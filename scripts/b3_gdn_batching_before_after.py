"""B3: before/after timing + bit-exactness for the GDN spec_forward
batching optimization (2026-08-02, notes/2026-08-02-gdn-spec-forward-
batching.md).

Loads ONE real GDN layer (same pattern as
``scripts/b3_probe_gdn_spec_rollback.py`` -- no full 27B model load) and
runs the OLD (fully-sequential-per-position) and NEW (batched-except-the-
recurrence) ``spec_forward`` implementations against each other, in the
SAME process, on the SAME layer instance and weights, so the only variable
is the algorithm. "OLD" is loaded straight from this worktree's parent
commit's ``runtime/model/qwen36_model.py`` via git, not retyped -- the
comparison is against the literal code this session started from.

Checks:
  1. old and new spec_forward agree bit-exactly (same output, same
     snapshots, for every position) -- the optimization must not change
     ANY value, only which kernel call computes it.
  2. wall-clock ms/call for: old spec_forward, new spec_forward, K
     sequential decode() calls, one chunk_gated_delta_rule call -- same
     ``timed()`` harness and n_iters as the existing probe, so numbers are
     directly comparable to the B3 report's "12.6ms vs 1.8ms, 6.9x" figures.

Run: ~/.venvs/vllm/bin/python scripts/b3_gdn_batching_before_after.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

_ROOT = "/home/bot/project/qsr-w-gdnopt"
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__}"
)

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from runtime.model.qwen36_model import GdnLayerState, Qwen36GatedDeltaNet  # noqa: E402
from runtime.model_loading import _build_qwen36_model_config  # noqa: E402

MODEL_PATH = (
    "/home/bot/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/"
    "snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404"
)
LAYER_IDX = 0
DEVICE = torch.device("cuda")
torch.set_grad_enabled(False)
torch.cuda.set_per_process_memory_fraction(0.10, device=0)

_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")


def load_old_spec_forward():
    """Pull ``Qwen36GatedDeltaNet.spec_forward`` from the parent commit
    (``main``) -- i.e. the literal pre-optimization function this session
    started editing -- and load it under a distinct module name so it does
    not collide with the already-imported (new/batched) module. Only the
    function object is used, called unbound against a NEW-module layer
    instance below (identical weights, identical `self.forward` -- forward()
    itself was not touched by this session)."""
    old_src = subprocess.run(
        ["git", "show", "main:runtime/model/qwen36_model.py"],
        cwd=_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    old_path = Path("/tmp/_b3_qwen36_model_old_20260802.py")
    old_path.write_text(old_src)
    spec = importlib.util.spec_from_file_location("_b3_qwen36_model_old", old_path)
    old_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old_module  # dataclass() needs this in sys.modules to resolve
    spec.loader.exec_module(old_module)
    return old_module.Qwen36GatedDeltaNet.spec_forward


def load_layer(layer_idx: int) -> tuple[Qwen36GatedDeltaNet, int]:
    index = json.loads((Path(MODEL_PATH) / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}.linear_attn."
    names = [n for n in weight_map if n.startswith(prefix)]
    by_shard: dict[str, list[str]] = {}
    for n in names:
        by_shard.setdefault(weight_map[n], []).append(n)
    tensors: dict[str, torch.Tensor] = {}
    for shard, shard_names in by_shard.items():
        with safe_open(str(Path(MODEL_PATH) / shard), framework="pt", device="cpu") as f:
            for n in shard_names:
                tensors[n[len(prefix) :]] = f.get_tensor(n).to(DEVICE)

    model_config = _build_qwen36_model_config(MODEL_PATH)
    quantized = {
        f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_qkv": "FP8",
        f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_z": "FP8",
        f"model.language_model.layers.{layer_idx}.linear_attn.out_proj": "FP8",
    }
    with DEVICE:
        layer = Qwen36GatedDeltaNet(model_config, layer_idx, quantized).to(DEVICE).to(
            torch.bfloat16
        )
    params = dict(layer.named_parameters())
    name_map = {
        "in_proj_qkv.weight": "in_proj_qkv.weight",
        "in_proj_qkv.weight_scale": "in_proj_qkv.weight_scale",
        "in_proj_z.weight": "in_proj_z.weight",
        "in_proj_z.weight_scale": "in_proj_z.weight_scale",
        "out_proj.weight": "out_proj.weight",
        "out_proj.weight_scale": "out_proj.weight_scale",
        "in_proj_a.weight": "in_proj_a.weight",
        "in_proj_b.weight": "in_proj_b.weight",
        "dt_bias": "dt_bias",
        "A_log": "A_log",
        "conv1d.weight": "conv1d.weight",
        "norm.weight": "norm.weight",
    }
    missing = set(name_map) - set(tensors)
    assert not missing, f"checkpoint tensors missing: {missing}"
    for ckpt_name, param_name in name_map.items():
        params[param_name].data.copy_(tensors[ckpt_name].to(params[param_name].dtype))
    return layer, model_config["hidden_size"]


def clone_state(state: GdnLayerState) -> GdnLayerState:
    return GdnLayerState(
        conv_state=state.conv_state.clone(),
        recurrent_state=state.recurrent_state.clone(),
        has_previous_state=state.has_previous_state,
    )


def main() -> None:
    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    layer, hidden_size = load_layer(LAYER_IDX)
    old_spec_forward = load_old_spec_forward()
    print(f"loaded real layer {LAYER_IDX} weights, hidden_size={hidden_size}")

    torch.manual_seed(1234)
    anchor_state = layer.new_state(batch=1, device=DEVICE, dtype=torch.bfloat16)
    prefill_len = 6
    x_prefill = torch.randn(1, prefill_len, hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.1
    layer(x_prefill, anchor_state)
    record("anchor has_previous_state after prefill", anchor_state.has_previous_state is True)

    K = 16
    x_candidates = torch.randn(1, K, hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.1

    # -- Correctness: OLD and NEW spec_forward must agree bit-exactly ----
    print(f"\n=== old vs new spec_forward, bit-exactness, K={K} ===")
    anchor_conv_before = anchor_state.conv_state.clone()
    anchor_rec_before = anchor_state.recurrent_state.clone()
    old_out, old_snapshots = old_spec_forward(layer, x_candidates, anchor_state)
    new_out, new_snapshots = layer.spec_forward(x_candidates, anchor_state)

    record(
        "anchor untouched by either call",
        torch.equal(anchor_state.conv_state, anchor_conv_before)
        and torch.equal(anchor_state.recurrent_state, anchor_rec_before),
    )
    record("output: old == new (bit-exact)", torch.equal(old_out, new_out))
    record("snapshot count: old == new", len(old_snapshots) == len(new_snapshots) == K + 1)

    max_diff = 0.0
    for j in range(K + 1):
        conv_diff = (
            (old_snapshots[j].conv_state.float() - new_snapshots[j].conv_state.float())
            .abs().max().item()
        )
        rec_diff = (
            (old_snapshots[j].recurrent_state.float() - new_snapshots[j].recurrent_state.float())
            .abs().max().item()
        )
        max_diff = max(max_diff, conv_diff, rec_diff)
        bit_exact = torch.equal(
            old_snapshots[j].conv_state, new_snapshots[j].conv_state
        ) and torch.equal(old_snapshots[j].recurrent_state, new_snapshots[j].recurrent_state)
        record(
            f"snapshot[{j}]: old == new (bit-exact)",
            bit_exact,
            f"conv_max_abs_diff={conv_diff:.3g} recurrent_max_abs_diff={rec_diff:.3g}",
        )
    print(f"max diff old-vs-new across all snapshots (should be 0.0): {max_diff:.6g}")

    # -- Throughput: old spec_forward vs new spec_forward vs sequential
    # decode vs one chunk call -- same harness/n_iters as
    # scripts/b3_probe_gdn_spec_rollback.py, so numbers are directly
    # comparable to the B3 report's baseline. --------------------------
    print(f"\n=== throughput (this one GDN layer, K={K}) ===")
    torch.cuda.synchronize()
    n_iters = 50

    def timed(fn) -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iters * 1e3

    old_state = clone_state(anchor_state)
    old_state.has_previous_state = True
    t_old = timed(lambda: old_spec_forward(layer, x_candidates, old_state))

    new_state = clone_state(anchor_state)
    new_state.has_previous_state = True
    t_new = timed(lambda: layer.spec_forward(x_candidates, new_state))

    def run_sequential() -> None:
        s = clone_state(anchor_state)
        s.has_previous_state = True
        for t in range(K):
            layer(x_candidates[:, t : t + 1, :], s)

    t_sequential = timed(run_sequential)

    def run_chunk() -> None:
        s = clone_state(anchor_state)
        s.has_previous_state = True
        layer(x_candidates, s)

    t_chunk = timed(run_chunk)

    print(f"  OLD spec_forward (fully sequential, K={K} loop iterations):  {t_old:.3f} ms")
    print(f"  NEW spec_forward (batched conv/norm/small-proj, K={K} loop): {t_new:.3f} ms")
    print(f"  {K} ordinary sequential decode() calls (no verify at all):   {t_sequential:.3f} ms")
    print(f"  one chunk_gated_delta_rule call (no rollback capability):    {t_chunk:.3f} ms")
    print(f"\n  speedup, NEW vs OLD spec_forward: {t_old / t_new:.2f}x")
    print(f"  OLD spec_forward vs chunk: {t_old / t_chunk:.2f}x slower")
    print(f"  NEW spec_forward vs chunk: {t_new / t_chunk:.2f}x slower")

    print("\n== summary ==")
    failed = [n for n, ok, _ in _results if not ok]
    for name, ok, detail in _results:
        suffix = f"  ({detail})" if detail and not ok else ""
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{suffix}")
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
