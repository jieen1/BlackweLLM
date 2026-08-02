"""B3: does GDN speculative-verify rollback reproduce the non-speculative
path exactly?

Correctness bar is ``docs/b1-correctness-criterion.md`` §7's B3 judgement:
"接受/拒绝后GDN状态与非投机路径的状态张量对比" (compare the GDN state after
accept/reject against the non-speculative path's state) -- not
bit-exactness against HF, which §7 explicitly says B3 cannot ask for
(verify's ``seq_len>1`` runs the chunk algorithm; B1's own criterion is
built around the fact that chunk and ``fused_recurrent`` disagree by ~30
ULP for the same tokens -- "陷阱5" in this session's brief). This script's
whole point is that :meth:`Qwen36GatedDeltaNet.spec_forward` sidesteps that
trap rather than re-encountering it: it never calls the chunk path at all,
so there is no 30-ULP gap to account for -- rolled-back state should be
BIT-IDENTICAL to sequential decode, not merely close.

Uses ONE real GDN layer's real (FP8) checkpoint weights, loaded directly
via safetensors -- same safe pattern as ``scripts/b1_verify_gdn_layer.py``
(no full 27B model load). This is deliberate, not a shortcut: a session
earlier today OOM'd the shared card twice trying to exercise this through
the full model (every ``ModelOptFP8Linear``/``ModelOptNVFP4Linear``
dequantizes to BF16 lazily and CACHES it forever -- documented, deliberate
B1 scope, but it means touching every layer at least once inflates resident
memory from ~19 GiB to ~54+ GiB before a single KV byte, independent of
any --blocks-per-slot/--slots knob). The claim under test here is a
property of the GDN recurrence's control flow (does index j's snapshot
match j sequential decode steps), not of specific weight values, so a
single real layer proves it exactly as well as the full model would, at a
GPU memory footprint of tens of MiB instead of tens of GiB.

Run: ~/.venvs/vllm/bin/python scripts/b3_probe_gdn_spec_rollback.py
"""

from __future__ import annotations

import json
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

from runtime.model.qwen36_model import (  # noqa: E402
    GdnLayerState,
    Qwen36GatedDeltaNet,
    commit_spec_snapshot,
)
from runtime.model_loading import _build_qwen36_model_config  # noqa: E402

MODEL_PATH = (
    "/home/bot/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/"
    "snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404"
)
LAYER_IDX = 0
DEVICE = torch.device("cuda")
torch.set_grad_enabled(False)

# Defensive safety cap even though this script's real footprint is tiny
# (tens of MiB) -- cheap insurance on a card the user is actively using.
torch.cuda.set_per_process_memory_fraction(0.10, device=0)

_results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}")


def load_layer(layer_idx: int) -> Qwen36GatedDeltaNet:
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
    print(f"loaded real layer {LAYER_IDX} weights, hidden_size={hidden_size}")

    torch.manual_seed(1234)

    # -- Build a non-trivial anchor state: a real (short) prefill, not a
    # zero state -- a rollback mechanism that only "works" from zero state
    # is not a real test (B0-5's own operational requirement is that zero
    # state is the ONE thing guaranteed only for a fresh slot; verify never
    # runs on a fresh slot). --------------------------------------------
    anchor_state = layer.new_state(batch=1, device=DEVICE, dtype=torch.bfloat16)
    prefill_len = 6
    x_prefill = torch.randn(1, prefill_len, hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.1
    layer(x_prefill, anchor_state)  # mutates anchor_state in place; now has_previous_state=True
    record("anchor has_previous_state after prefill", anchor_state.has_previous_state is True)

    K = 16  # matches this repo's NUM_SPECULATIVE_TOKENS
    x_candidates = torch.randn(1, K, hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.1

    # -- spec_forward must not mutate the anchor --------------------------
    anchor_conv_before = anchor_state.conv_state.clone()
    anchor_rec_before = anchor_state.recurrent_state.clone()
    spec_out, snapshots = layer.spec_forward(x_candidates, anchor_state)
    record(
        "spec_forward left the anchor untouched",
        torch.equal(anchor_state.conv_state, anchor_conv_before)
        and torch.equal(anchor_state.recurrent_state, anchor_rec_before),
    )
    record("spec_forward returned K+1 snapshots", len(snapshots) == K + 1, f"got {len(snapshots)}")
    record(
        "snapshot[0] equals the anchor",
        torch.equal(snapshots[0].conv_state, anchor_conv_before)
        and torch.equal(snapshots[0].recurrent_state, anchor_rec_before),
    )

    # -- The core claim: for every m in [0, K], commit_spec_snapshot(m)
    # must reproduce EXACTLY what m sequential ordinary decode() calls from
    # the same anchor would produce -- not close, identical, because both
    # paths call the exact same fused_recurrent_gated_delta_rule kernel on
    # the exact same inputs. -------------------------------------------
    print(f"\n=== rollback vs sequential non-speculative decode, K={K} ===")
    max_diffs: list[float] = []
    for m in (0, 1, 5, K // 2, K - 1, K):
        # Reference: m ordinary single-token decode steps from a fresh
        # clone of the SAME anchor.
        ref_state = clone_state(anchor_state)
        ref_state.has_previous_state = True
        ref_outputs = []
        for t in range(m):
            ref_outputs.append(layer(x_candidates[:, t : t + 1, :], ref_state))

        # Candidate: roll spec_forward's snapshots back to m.
        live_state = clone_state(anchor_state)
        live_state.has_previous_state = True
        commit_spec_snapshot(live_state, snapshots, accepted_count=m)

        conv_diff = (
            (live_state.conv_state.float() - ref_state.conv_state.float()).abs().max().item()
        )
        rec_diff = (
            (live_state.recurrent_state.float() - ref_state.recurrent_state.float())
            .abs()
            .max()
            .item()
        )
        bit_exact = torch.equal(live_state.conv_state, ref_state.conv_state) and torch.equal(
            live_state.recurrent_state, ref_state.recurrent_state
        )
        max_diffs.append(max(conv_diff, rec_diff))
        record(
            f"m={m}: rolled-back state == sequential-decode state (bit-exact)",
            bit_exact,
            f"conv_max_abs_diff={conv_diff:.3g} recurrent_max_abs_diff={rec_diff:.3g}",
        )

        # Also check the per-position OUTPUT for positions [0, m) matches
        # (not just the final state) -- spec_forward's return value is
        # what the rest of the model would actually consume.
        if m > 0:
            spec_prefix = spec_out[:, :m, :]
            ref_cat = torch.cat(ref_outputs, dim=1)
            out_exact = torch.equal(spec_prefix, ref_cat)
            record(f"m={m}: spec_forward output[:m] == sequential output (bit-exact)", out_exact)

    # -- accepted_count out of range must raise, not silently clamp -------
    raised = False
    try:
        commit_spec_snapshot(clone_state(anchor_state), snapshots, accepted_count=K + 1)
    except ValueError:
        raised = True
    record("accepted_count > K raises ValueError", raised)

    # -- Throughput: what does this rollback-capable path cost relative to
    # (a) K ordinary sequential decode steps (the thing MTP verify exists
    # to avoid paying when acceptance is high) and (b) one chunk call over
    # K tokens (the "naive verify, no rollback support" alternative this
    # mechanism deliberately does NOT use, because it cannot expose
    # intermediate states). Real numbers for the B3 "GDN kernel tuning"
    # item, on this one real layer -- multiply by 48 (GDN layer count) for
    # a whole-model estimate, understanding that is linear extrapolation,
    # not a second measurement. ------------------------------------------
    print(f"\n=== throughput (this one GDN layer, K={K}) ===")
    torch.cuda.synchronize()
    n_iters = 50

    def timed(fn) -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iters * 1e3  # ms/call

    spec_state = clone_state(anchor_state)
    spec_state.has_previous_state = True
    t_spec = timed(lambda: layer.spec_forward(x_candidates, spec_state))

    seq_state = clone_state(anchor_state)
    seq_state.has_previous_state = True

    def run_sequential() -> None:
        s = clone_state(anchor_state)
        s.has_previous_state = True
        for t in range(K):
            layer(x_candidates[:, t : t + 1, :], s)

    t_sequential = timed(run_sequential)

    def run_chunk() -> None:
        s = clone_state(anchor_state)
        s.has_previous_state = True
        layer(x_candidates, s)  # seq_len=K>1 -> chunk_gated_delta_rule branch

    t_chunk = timed(run_chunk)

    print(f"  spec_forward ({K} sequential fused_recurrent calls, this mechanism): {t_spec:.3f} ms")
    print(f"  {K} ordinary sequential decode() calls (no verify at all):  {t_sequential:.3f} ms")
    print(f"  one chunk_gated_delta_rule call (no rollback capability):   {t_chunk:.3f} ms")
    print(
        f"  spec_forward vs sequential decode: {t_sequential / t_spec:.2f}x "
        f"(expected close to 1x -- same kernel, same call count, this IS the "
        "sequential path, just materializing snapshots along the way)"
    )
    print(
        f"  spec_forward vs chunk (the cost this mechanism trades away): "
        f"{t_spec / t_chunk:.2f}x slower, one GDN layer"
    )

    print("\n== summary ==")
    failed = [n for n, ok, _ in _results if not ok]
    for name, ok, detail in _results:
        suffix = f"  ({detail})" if detail and not ok else ""
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{suffix}")
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed")
    print(f"max diff across all tested m (should be 0.0 for bit-exact): {max(max_diffs):.6g}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
