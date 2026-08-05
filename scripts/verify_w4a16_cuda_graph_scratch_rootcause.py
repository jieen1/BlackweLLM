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

That root cause is fixed in ``plan_w4a16_buffers`` itself (unions the
packed-mode and direct-topk-routes scratch bounds), merged to sparkinfer
master as ``8242340``. This script confirms that fix makes the workaround
unnecessary -- and now that ``_forward_w4a16_fused`` has had the workaround
removed and gone back to passing ``buffers.fc1_c_tmp``/``buffers.fc2_c_tmp``
straight through, this exercises the REAL post-revert method, not a
stand-in copy: it builds ONE real MLP layer (real quantized weights off
disk -- cheap, no full 27B-parameter model load), then captures a
``torch.cuda.graph`` around ``mlp._forward_w4a16_fused`` at the exact decode
shape that broke before (m=2, degenerate topk=1/num_experts=1).

If capture succeeds and the replayed output matches eager bit-for-bit, the
workaround is confirmed unnecessary against the fixed sparkinfer.

Runs from wherever this file lives (``sys.path`` is derived from
``__file__``); sparkinfer resolves through ``BF_SPARKINFER_PATH`` or its
normal default, and the script asserts up front that whichever checkout it
got actually carries the scratch fix:

    ~/.venvs/vllm/bin/python scripts/verify_w4a16_cuda_graph_scratch_rootcause.py

Not a pytest test (needs the GPU lock + a real checkpoint on disk) -- run
manually, one shot, under /tmp/gpu_lock.sh, same convention as
``verify_nvfp4_gemm_single_layer.py`` (whose ``build_mlp`` this reuses).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

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
# BF_SPARKINFER_PATH picks which checkout, falling back to
# _sparkinfer_import's own default; the capability assert below is what
# decides whether the one we got is usable.
import torch  # noqa: E402

from runtime.backends._sparkinfer_import import ensure_sparkinfer_path  # noqa: E402

ensure_sparkinfer_path()

import sparkinfer  # noqa: E402

# Assert the CAPABILITY, not a path. This used to pin sparkinfer to
# /home/bot/project/spark-w-w4a16, the throwaway worktree the scratch fix was
# developed in; the fix has since merged to sparkinfer master (8242340) and
# that worktree is gone, so the path check could only ever fail from here on.
# What the script actually needs is a sparkinfer whose allocator covers
# decode's direct-topk path -- same contract tests/test_w4a16_scratch_contract.py
# pins, restated here because this script runs outside pytest.
from sparkinfer.moe._shared.kernels.w4a16.host import (  # noqa: E402
    packed_gemm_scratch_elements,
    plan_w4a16_buffers,
)

_probe_prepared = SimpleNamespace(
    num_experts=1,
    hidden_size=5120,
    intermediate_size=17408,
    is_gated=True,
)
_probe_plan = plan_w4a16_buffers(
    _probe_prepared,
    m=2,
    topk=1,
    route_num_experts=1,
    sms=128,
)
_probe_direct_slots = 2 * _probe_plan.block_size_m
_probe_fc1_needed = packed_gemm_scratch_elements(
    size_n=2 * _probe_prepared.intermediate_size,
    route_slots=_probe_direct_slots,
    moe_block_size=_probe_plan.block_size_m,
    sms=128,
)
_probe_fc2_needed = packed_gemm_scratch_elements(
    size_n=_probe_prepared.hidden_size,
    route_slots=_probe_direct_slots,
    moe_block_size=_probe_plan.block_size_m,
    sms=128,
)
_probe_scratch_covers_direct = (
    _probe_plan.fc1_c_tmp_elements >= _probe_fc1_needed
    and _probe_plan.fc2_c_tmp_elements >= _probe_fc2_needed
)
assert _probe_scratch_covers_direct, (
    f"sparkinfer at {sparkinfer.__file__} does not size W4A16 c_tmp scratch "
    f"for direct-topk decode (fc1={_probe_plan.fc1_c_tmp_elements}/"
    f"{_probe_fc1_needed}, fc2={_probe_plan.fc2_c_tmp_elements}/"
    f"{_probe_fc2_needed}); this checkout predates scratch-union fix 8242340."
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=int,
        default=2,
        help="static decode rows to capture (MTP target verify is anchor + K = 4)",
    )
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")

    ckpt = _find_ckpt(DEFAULT_CKPT_GLOB)
    print(f"checkpoint: {ckpt}")
    print(f"=== layer {LAYER} MLP (fused gate/up/down_proj) ===")
    mlp, hidden_size, intermediate_size = build_mlp(ckpt, LAYER)
    mlp._ensure_w4a16_fused_ready()
    print(f"  hidden_size={hidden_size} intermediate_size={intermediate_size}")

    # M=2 is the historical scratch repro; M=4 is Qwen's target verify
    # shape (anchor plus K=3 drafted continuations).
    decode_m = args.rows
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
