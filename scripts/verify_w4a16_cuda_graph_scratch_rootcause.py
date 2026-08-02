"""Root-cause verification for the W4A16 fc1/fc2 ``c_tmp`` scratch mismatch
that used to break decode CUDA Graph capture (see notes/2026-08-03-std-
model-serving-acceptance.md section 3, and git history for
``Qwen36MLP._w4a16_c_tmp_scratch``, since deleted from this file).

That mismatch was sparkinfer's own bug: ``plan_w4a16_buffers``/
``make_w4a16_packed_buffers`` sized fc1/fc2 ``c_tmp`` via
``max_packed_route_slots`` (the *packed/grouped* route-kernel's bound), but
``run_w4a16_moe``'s small-M "direct top-k routes" / TC-decode fast path
(exactly what this repo's decode always takes) needs
``route_slots_for_scratch = m * topk * block_size_m`` instead -- a
different, and for this deployment's degenerate 1-expert/top-1 MoE, LARGER
number (16 vs 9 for decode batch=2). ``runtime/model/qwen36_model.py`` used
to work around this with a separate, conservatively-oversized persistent
scratch buffer (``Qwen36MLP._w4a16_c_tmp_scratch``) instead of using
``make_w4a16_packed_buffers``'s own ``fc1_c_tmp``/``fc2_c_tmp`` directly.

sparkinfer worktree ``/home/bot/project/spark-w-w4a16``
(``work/w4a16-scratch-20260803``) fixes the root cause in
``plan_w4a16_buffers`` itself (unions the packed-mode and direct-topk-routes
scratch bounds). This script confirms that fix makes the workaround
unnecessary -- and now that ``_forward_w4a16_fused`` has had the workaround
removed and gone back to passing ``buffers.fc1_c_tmp``/``buffers.fc2_c_tmp``
straight through, this exercises the REAL post-revert method, not a
stand-in copy: it builds ONE real MLP layer (real quantized weights off
disk -- cheap, no full 27B-parameter model load), then captures a
``torch.cuda.graph`` around ``mlp._forward_w4a16_fused`` at the exact decode
shape that broke before (m=2, degenerate topk=1/num_experts=1).

If capture succeeds and the replayed output matches eager bit-for-bit, the
workaround is confirmed unnecessary against the fixed sparkinfer.

*** Run from this worktree's root (``cd`` here first) so the relative
``sys.path`` insert below resolves; sparkinfer defaults to the FIXED
worktree via ``BF_SPARKINFER_PATH`` unless overridden, e.g.:

    cd /home/bot/project/qsr-w-w4a16fix
    ~/.venvs/vllm/bin/python scripts/verify_w4a16_cuda_graph_scratch_rootcause.py

Not a pytest test (needs the GPU lock + a real checkpoint on disk) -- run
manually, one shot, under /tmp/gpu_lock.sh, same convention as
``verify_nvfp4_gemm_single_layer.py`` (whose ``build_mlp`` this reuses).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__} "
    f"-- rerun with PYTHONPATH including {_ROOT}"
)

# Must resolve `sparkinfer` through the SAME controlled entry point
# `Qwen36MLP._ensure_w4a16_fused_ready` uses (runtime/backends/
# _sparkinfer_import.py::ensure_sparkinfer_path), and do it before anything
# else in this process touches `sparkinfer` for the first time -- otherwise
# ensure_sparkinfer_path() raises (sys.path edits can't retroactively
# redirect an already-imported package; see that module's docstring).
# BF_SPARKINFER_PATH picks which checkout; default here is the FIXED
# worktree, not the buggy main sparkinfer checkout.
import os  # noqa: E402

import torch  # noqa: E402

os.environ.setdefault("BF_SPARKINFER_PATH", "/home/bot/project/spark-w-w4a16")
from runtime.backends._sparkinfer_import import ensure_sparkinfer_path  # noqa: E402

ensure_sparkinfer_path()

import sparkinfer  # noqa: E402

_EXPECTED_SPARKINFER_ROOT = "/home/bot/project/spark-w-w4a16"
assert sparkinfer.__file__.startswith(_EXPECTED_SPARKINFER_ROOT), (
    f"sparkinfer resolved to {sparkinfer.__file__}, not the fixed worktree "
    f"{_EXPECTED_SPARKINFER_ROOT} -- rerun with BF_SPARKINFER_PATH set to it"
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_nvfp4_gemm_single_layer import (  # noqa: E402
    DEFAULT_CKPT_GLOB,
    LAYER,
    _find_ckpt,
    build_mlp,
)

DEVICE = "cuda"


def fused_forward_no_workaround(mlp, x: torch.Tensor) -> torch.Tensor:
    """Calls the REAL (post-revert) ``Qwen36MLP._forward_w4a16_fused`` --
    the ``_w4a16_c_tmp_scratch`` workaround has been deleted from that
    method, so this now exercises the actual production code path, not a
    hand-copied stand-in for it."""
    return mlp._forward_w4a16_fused(x)


def main() -> None:
    ckpt = _find_ckpt(DEFAULT_CKPT_GLOB)
    print(f"checkpoint: {ckpt}")
    print(f"=== layer {LAYER} MLP (fused gate/up/down_proj) ===")
    mlp, hidden_size, intermediate_size = build_mlp(ckpt, LAYER)
    mlp._ensure_w4a16_fused_ready()
    print(f"  hidden_size={hidden_size} intermediate_size={intermediate_size}")

    # The exact production repro shape (notes/2026-08-03-std-model-serving-acceptance.md).
    decode_m = 2
    torch.manual_seed(20260803)
    x = (
        torch.randn(decode_m, hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.02
    ).contiguous()
    static_x = x.clone()

    eager_out = fused_forward_no_workaround(mlp, x)
    torch.cuda.synchronize()
    print(f"  eager forward OK, output shape={tuple(eager_out.shape)}")

    # Standard torch.cuda.graph capture idiom: warm up on a side stream
    # first (kernels get JIT-compiled/cached here, same as real serving's
    # warm-up-before-capture), then capture on the default stream.
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            _ = fused_forward_no_workaround(mlp, static_x)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            graph_out = fused_forward_no_workaround(mlp, static_x)
    except RuntimeError as exc:
        print(f"\nCAPTURE FAILED (workaround still required): {exc}")
        raise
    print("  CUDA Graph capture OK (no workaround, buffers.fc1_c_tmp/fc2_c_tmp direct passthrough)")

    graph.replay()
    torch.cuda.synchronize()

    max_abs_diff = (graph_out.double() - eager_out.double()).abs().max().item()
    bit_exact = torch.equal(graph_out, eager_out)
    print(f"  replay vs eager: bit_exact={bit_exact} max_abs_diff={max_abs_diff:.3e}")
    assert bit_exact, "CUDA Graph replay must be bit-exact vs eager for the same static input"

    # Replay again with a different input written into the static buffer,
    # confirming the graph is reusable across decode steps like real serving
    # does (not just a one-shot capture-and-discard).
    torch.manual_seed(999)
    static_x.copy_(torch.randn(decode_m, hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.02)
    graph.replay()
    torch.cuda.synchronize()
    eager_out2 = fused_forward_no_workaround(mlp, static_x)
    torch.cuda.synchronize()
    bit_exact2 = torch.equal(graph_out, eager_out2)
    print(f"  second replay (new input) vs fresh eager: bit_exact={bit_exact2}")
    assert bit_exact2

    print("\nPASS: root-cause sparkinfer fix makes the qwen36_model.py workaround unnecessary.")


if __name__ == "__main__":
    main()
