"""B3: is *bit-exactness against the per-position loop* the right bar for
batching MTP verify's big GDN projections?

``notes/2026-08-02-gdn-spec-forward-batching.md`` left
``in_proj_qkv``/``in_proj_z``/``out_proj`` unbatched inside
:meth:`Qwen36GatedDeltaNet.spec_forward`'s ``for t in range(seq_len)``
loop, because batching them (either as a plain ``F.linear`` over all K
rows or via ``_bmm_project``) measurably stops reproducing the
per-position loop bit-for-bit once the output dim exceeds ~512. That
veto is what this script re-examines, using the same method
``docs/b1-correctness-criterion.md`` used to retire the original B1
gate: **stop asking whether two implementations agree bit-for-bit and
start asking whether they disagree by more than the noise floor.**

Four independent questions, all answerable from ONE real GDN layer plus
one real MLP/attention layer's weights -- no full-model load (see
``scripts/b3_probe_gdn_spec_rollback.py``'s docstring for why that
matters on this card):

**A. Is the bit-exact guarantee even intact today?**
:meth:`Qwen36TextModelSelfBuilt.verify_forward` calls ``layer.mlp`` and
``layer.self_attn`` on the whole ``[1, K, hidden]`` block for every one
of the 64 layers. Those are the same "batched ``F.linear`` with
output dim in the thousands" the note measured as non-bit-exact. If
they are non-bit-exact here too, then the hidden states entering GDN
layer *i>0*'s ``spec_forward`` already differ from what sequential
decode produced -- and protecting three projections inside that layer
buys a guarantee that was already spent one layer earlier.

**B. Which rounding is actually correct?**
Neither ``F.linear``-over-K-rows nor the K-separate-GEMV loop is a
reference; both are BF16 roundings of the same real number. Compared
against an FP64 evaluation of the same dot products, if the batched
result is no further from the truth than the loop, then "bit-exact vs
the loop" is a tie-break between two equally valid answers, not a
correctness property.

**C. How big is the perturbation the change would actually introduce?**
Measured at :meth:`spec_forward`'s own outputs (per-position hidden
output + every recurrent-state snapshot), in absolute terms AND in BF16
ULPs of the tensors involved -- the unit ``docs/b1-correctness-criterion.md``
calibrated its thresholds in.

**D. Does a perturbation of that size in the recurrent state grow or
decay?** This is the question the coordinator flagged as possibly having
a *different* answer from the projections: GDN state is the one thing
that survives across decode steps, so an error in it is inherited by
every later token. Injecting a perturbation of exactly the size B/C
measure and running N ordinary decode steps says whether the recurrence
contracts it or amplifies it -- measurement, not the assumption that
"recursive => must be bit-exact".

Run: ~/.venvs/vllm/bin/python scripts/b3_probe_batching_bar.py [--k 16]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__}"
)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402

from runtime.checkpoints import modelopt_checkpoint_path  # noqa: E402
from runtime.loading.modelopt import quantized_layers_map  # noqa: E402
from runtime.model.qwen36_model import (  # noqa: E402
    GdnLayerState,
    Qwen36GatedDeltaNet,
    Qwen36MLP,
    _bmm_project,
)
from runtime.model_loading import _build_qwen36_model_config  # noqa: E402

# Deliberately modelopt (nvidia), not the standard checkpoint: this script
# imports ``runtime.loading.modelopt.quantized_layers_map`` directly, which
# classifies against modelopt's own ``quantization_config.quantized_layers``
# schema -- the standard (compressed-tensors) checkpoint does not have that
# key at all, so this classifier does not even apply to it. Also builds
# ``ModelOptFP8Linear`` further down (see ``runtime/model/qwen36_model.py``'s
# ``_LINEAR_FACTORY_FOR_ALGO``), which is wrong for the standard checkpoint's
# per-channel FP8 scale layout -- see ``b1_verify_gdn_layer.py``'s equivalent
# comment for the full explanation.
MODEL_PATH = modelopt_checkpoint_path()
DEVICE = torch.device("cuda")
torch.set_grad_enabled(False)

RESULTS: dict[str, object] = {}


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """ULP of BF16 at each element's magnitude: 2^(exp-7), 7 explicit
    mantissa bits. Zeros map to the smallest normal ULP so the division
    below never blows up."""
    ax = x.abs().float()
    exp = torch.where(ax > 0, torch.floor(torch.log2(ax)), torch.full_like(ax, -126.0))
    return torch.exp2(exp - 7.0)


def diff_report(name: str, batched: torch.Tensor, perpos: torch.Tensor) -> dict[str, object]:
    """Difference between two BF16 results of the same computation.

    Reported in ULPs **at the tensor's own scale** (``ULP(max|ref|)``), not
    per-element ULP: the per-element version divides by the ULP of a
    near-zero element and produces meaningless five-digit ratios for a
    difference that is one bit in the last place of a denormal-ish value.
    The scale-relative version is the one that matters downstream, and it
    is the unit ``docs/b1-correctness-criterion.md`` calibrated in (its
    "2 ULP" is 0.125 against logits of magnitude ~10)."""
    d = (batched.float() - perpos.float()).abs()
    scale = perpos.abs().float().max().item()
    scale_ulp = 2.0 ** (torch.tensor(max(scale, 1e-30)).log2().floor().item() - 7.0)
    nz = perpos != 0
    per_elem_ulps = (d[nz] / bf16_ulp(perpos[nz])) if nz.any() else torch.zeros(1)
    rep = {
        "bit_exact": bool(torch.equal(batched, perpos)),
        "max_abs_diff": d.max().item(),
        "rms_diff": d.pow(2).mean().sqrt().item(),
        "ref_max_abs": scale,
        "scale_ulp": scale_ulp,
        "max_diff_scale_ulps": d.max().item() / scale_ulp,
        "rms_diff_scale_ulps": d.pow(2).mean().sqrt().item() / scale_ulp,
        "p99_elementwise_ulps": torch.quantile(per_elem_ulps.float(), 0.99).item(),
        "frac_elements_differing": (d > 0).float().mean().item(),
    }
    print(
        f"  {name:<34} bit_exact={str(rep['bit_exact']):<5} "
        f"max_abs={rep['max_abs_diff']:.4g} "
        f"max={rep['max_diff_scale_ulps']:.2f} rms={rep['rms_diff_scale_ulps']:.3f} "
        f"ULP(scale={scale:.3g})  differ={rep['frac_elements_differing']:.1%}"
    )
    return rep


# ---------------------------------------------------------------------------
# Loading: one real layer at a time, straight from safetensors.
# ---------------------------------------------------------------------------


def _load_prefix(prefix: str) -> dict[str, torch.Tensor]:
    index = json.loads((Path(MODEL_PATH) / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    names = [n for n in weight_map if n.startswith(prefix)]
    assert names, f"no checkpoint tensors under {prefix!r}"
    by_shard: dict[str, list[str]] = {}
    for n in names:
        by_shard.setdefault(weight_map[n], []).append(n)
    out: dict[str, torch.Tensor] = {}
    for shard, shard_names in by_shard.items():
        with safe_open(str(Path(MODEL_PATH) / shard), framework="pt", device="cpu") as f:
            for n in shard_names:
                out[n[len(prefix) :]] = f.get_tensor(n).to(DEVICE)
    return out


def _install(module: torch.nn.Module, tensors: dict[str, torch.Tensor]) -> None:
    params = dict(module.named_parameters())
    for name, param in params.items():
        assert name in tensors, f"checkpoint is missing {name!r} (have {sorted(tensors)})"
        param.data.copy_(tensors[name].to(param.dtype))


def load_gdn(config: dict, quantized: dict[str, str], layer_idx: int) -> Qwen36GatedDeltaNet:
    tensors = _load_prefix(f"model.language_model.layers.{layer_idx}.linear_attn.")
    with DEVICE:
        layer = Qwen36GatedDeltaNet(config, layer_idx, quantized).to(DEVICE).to(torch.bfloat16)
    _install(layer, tensors)
    return layer


def load_mlp(config: dict, quantized: dict[str, str], layer_idx: int) -> Qwen36MLP:
    tensors = _load_prefix(f"model.language_model.layers.{layer_idx}.mlp.")
    with DEVICE:
        mlp = Qwen36MLP(config, layer_idx, quantized).to(DEVICE).to(torch.bfloat16)
    _install(mlp, tensors)
    # part_a below calls mlp.gate_proj/mlp.down_proj directly (their own
    # legacy ModelOptNVFP4Linear.forward()) right after calling mlp(x) (the
    # fused w4a16 path) on this SAME instance -- the fused path frees the
    # raw NVFP4 Parameters by default once it builds its packed
    # representation (see Qwen36MLP.__init__'s docstring on
    # `_keep_raw_nvfp4_weights`), which would break those direct calls.
    # Opt out here since this diagnostic genuinely needs both live.
    mlp._keep_raw_nvfp4_weights = True
    return mlp


# ---------------------------------------------------------------------------
# A. Is the "bit-exact vs sequential decode" guarantee intact in
#    verify_forward TODAY, before any of this session's changes?
# ---------------------------------------------------------------------------


def part_a(config: dict, quantized: dict[str, str], k: int, hidden: int) -> None:
    print(f"\n=== A. verify_forward's OTHER batched sublayers, K={k} ===")
    print("    (these run batched over all K positions today, unmodified)")
    x = torch.randn(1, k, hidden, device=DEVICE, dtype=torch.bfloat16) * 0.1
    out: dict[str, object] = {}

    mlp = load_mlp(config, quantized, layer_idx=0)
    batched = mlp(x)
    perpos = torch.cat([mlp(x[:, t : t + 1, :]) for t in range(k)], dim=1)
    out["layer0_mlp"] = diff_report("layer0 mlp (dense, 17408)", batched, perpos)
    for sub, name in ((mlp.gate_proj, "gate_proj"), (mlp.down_proj, "down_proj")):
        xin = x if name == "gate_proj" else torch.randn(
            1, k, 17408, device=DEVICE, dtype=torch.bfloat16
        ) * 0.1
        b = sub(xin)
        p = torch.cat([sub(xin[:, t : t + 1, :]) for t in range(k)], dim=1)
        out[f"layer0_mlp_{name}"] = diff_report(f"layer0 mlp.{name}", b, p)
    del mlp
    torch.cuda.empty_cache()

    attn = _load_prefix("model.language_model.layers.3.self_attn.")
    from runtime.model.modelopt_linear import ModelOptFP8Linear  # noqa: PLC0415

    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        w = attn.get(f"{name}.weight")
        if w is None:
            continue
        scale = attn.get(f"{name}.weight_scale")
        out_f, in_f = w.shape
        lin = ModelOptFP8Linear(in_f, out_f, bias=False).to(DEVICE)
        lin.weight.data.copy_(w)
        lin.weight_scale.data.copy_(scale.to(lin.weight_scale.dtype))
        xin = torch.randn(1, k, in_f, device=DEVICE, dtype=torch.bfloat16) * 0.1
        b = lin(xin)
        p = torch.cat([lin(xin[:, t : t + 1, :]) for t in range(k)], dim=1)
        out[f"layer3_attn_{name}"] = diff_report(f"layer3 self_attn.{name} ({out_f})", b, p)
        del lin
    torch.cuda.empty_cache()
    RESULTS["part_a"] = out


# ---------------------------------------------------------------------------
# B. Against an FP64 oracle, which BF16 rounding is closer to the truth?
# ---------------------------------------------------------------------------


def part_b(gdn: Qwen36GatedDeltaNet, k: int, hidden: int) -> None:
    print(f"\n=== B. batched vs per-position vs an FP64 oracle, K={k} ===")
    print("    (a tie means 'bit-exact vs the loop' is a tie-break, not correctness)")
    x = torch.randn(1, k, hidden, device=DEVICE, dtype=torch.bfloat16) * 0.1
    out: dict[str, object] = {}
    projections = [
        ("in_proj_qkv", gdn.in_proj_qkv, x),
        ("in_proj_z", gdn.in_proj_z, x),
        ("in_proj_b", gdn.in_proj_b, x),
        (
            "out_proj",
            gdn.out_proj,
            torch.randn(1, k, gdn.value_dim, device=DEVICE, dtype=torch.bfloat16) * 0.1,
        ),
    ]
    for name, module, xin in projections:
        if hasattr(module, "_ensure_ready"):
            module._ensure_ready()
            w = module._weight_bf16
        else:
            w = module.weight
        # The oracle: the exact same BF16 inputs and BF16 dequantized
        # weights, but the dot products accumulated in FP64. Both BF16
        # results below are roundings OF THIS NUMBER -- nothing about the
        # inputs differs between them, only the reduction order.
        truth = (xin.reshape(k, -1).double() @ w.t().double()).reshape(1, k, -1)
        batched = module(xin)
        perpos = torch.cat([module(xin[:, t : t + 1, :]) for t in range(k)], dim=1)
        bmm = _bmm_project(module, xin)
        ideal = truth.to(torch.bfloat16)
        err_i = (ideal.double() - truth).abs()
        err_b = (batched.double() - truth).abs()
        err_p = (perpos.double() - truth).abs()
        err_m = (bmm.double() - truth).abs()
        closer_batched = (err_b < err_p).float().mean().item()
        closer_perpos = (err_p < err_b).float().mean().item()
        rep = {
            "rmse_batched": err_b.pow(2).mean().sqrt().item(),
            "rmse_perpos": err_p.pow(2).mean().sqrt().item(),
            "rmse_bmm": err_m.pow(2).mean().sqrt().item(),
            "rmse_ideal_rounding": err_i.pow(2).mean().sqrt().item(),
            "out_scale": truth.abs().max().item(),
            "max_err_batched": err_b.max().item(),
            "max_err_perpos": err_p.max().item(),
            "frac_batched_strictly_closer": closer_batched,
            "frac_perpos_strictly_closer": closer_perpos,
            "frac_tied": 1.0 - closer_batched - closer_perpos,
        }
        out[name] = rep
        print(
            f"  {name:<14} rmse: batched={rep['rmse_batched']:.6g} "
            f"perpos={rep['rmse_perpos']:.6g} bmm={rep['rmse_bmm']:.6g}   "
            f"ideal={rep['rmse_ideal_rounding']:.6g}   "
            f"batched closer on {closer_batched:.1%}, perpos closer on {closer_perpos:.1%}, "
            f"tied {rep['frac_tied']:.1%}"
        )
        scale_ulp = 2.0 ** (
            torch.tensor(max(rep["out_scale"], 1e-30)).log2().floor().item() - 7.0
        )
        print(
            f"                 in ULPs of the output scale ({rep['out_scale']:.3g}): "
            f"batched={rep['rmse_batched'] / scale_ulp:.4f}  "
            f"perpos={rep['rmse_perpos'] / scale_ulp:.4f}  "
            f"ideal(unavoidable)={rep['rmse_ideal_rounding'] / scale_ulp:.4f}"
        )
    RESULTS["part_b"] = out


# ---------------------------------------------------------------------------
# C. What does batching the three big projections actually change, at
#    spec_forward's own outputs?
# ---------------------------------------------------------------------------


def spec_forward_batched(
    self: Qwen36GatedDeltaNet, hidden_states: torch.Tensor, state: GdnLayerState
) -> tuple[torch.Tensor, list[GdnLayerState]]:
    """:meth:`Qwen36GatedDeltaNet.spec_forward` with in_proj_qkv/in_proj_z/
    out_proj batched over all K positions instead of looped. Byte-for-byte
    the shipped method otherwise -- the three ``for t in range(seq_len)``
    projection loops are the ONLY difference, so any measured delta is
    attributable to them alone."""
    batch_size, seq_len, _ = hidden_states.shape
    assert batch_size == 1
    if not state.has_previous_state:
        raise ValueError("spec_forward continues from a committed anchor")

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # BATCHED
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)  # BATCHED
    b = _bmm_project(self.in_proj_b, hidden_states)
    a = _bmm_project(self.in_proj_a, hidden_states)

    state_len = state.conv_state.shape[-1]
    catted = torch.cat([state.conv_state, mixed_qkv], dim=-1).to(self.conv1d.weight.dtype)
    conv_out = F.conv1d(catted, self.conv1d.weight, bias=None, padding=0, groups=self.conv_dim)
    mixed_qkv = F.silu(conv_out[:, :, -seq_len:]).transpose(1, 2)

    split_sizes = [self.key_dim, self.key_dim, self.value_dim]
    query, key, value = torch.split(mixed_qkv, split_sizes, dim=-1)
    query = query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.repeat > 1:
        query = query.repeat_interleave(self.repeat, dim=2)
        key = key.repeat_interleave(self.repeat, dim=2)

    from runtime.model.qwen36_model import fused_recurrent_gated_delta_rule  # noqa: PLC0415

    recurrent_state = state.recurrent_state.clone()
    recurrent_snapshots: list[torch.Tensor] = [recurrent_state]
    core_outs: list[torch.Tensor] = []
    for t in range(seq_len):
        core_attn_out, last_state = fused_recurrent_gated_delta_rule(
            query[:, t : t + 1], key[:, t : t + 1], value[:, t : t + 1],
            g=g[:, t : t + 1], beta=beta[:, t : t + 1],
            initial_state=recurrent_state, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        recurrent_state = torch.empty_like(state.recurrent_state)
        recurrent_state.copy_(last_state)
        recurrent_snapshots.append(recurrent_state)
        core_outs.append(core_attn_out)

    core_attn_out = torch.cat(core_outs, dim=1)
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z_flat = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z_flat)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)  # BATCHED

    snapshots = [
        GdnLayerState(
            conv_state=catted[:, :, j : j + state_len].clone(),
            recurrent_state=recurrent_snapshots[j],
            has_previous_state=True,
        )
        for j in range(seq_len + 1)
    ]
    return output, snapshots


def make_anchor(gdn: Qwen36GatedDeltaNet, hidden: int, prefill_len: int = 24) -> GdnLayerState:
    state = gdn.new_state(batch=1, device=DEVICE, dtype=torch.bfloat16)
    x = torch.randn(1, prefill_len, hidden, device=DEVICE, dtype=torch.bfloat16) * 0.1
    gdn(x, state)
    return state


def clone_state(s: GdnLayerState) -> GdnLayerState:
    return GdnLayerState(
        conv_state=s.conv_state.clone(),
        recurrent_state=s.recurrent_state.clone(),
        has_previous_state=s.has_previous_state,
    )


def part_c(gdn: Qwen36GatedDeltaNet, k: int, hidden: int) -> GdnLayerState:
    print(f"\n=== C. spec_forward: batched big projections vs shipped loop, K={k} ===")
    anchor = make_anchor(gdn, hidden)
    x = torch.randn(1, k, hidden, device=DEVICE, dtype=torch.bfloat16) * 0.1

    out_loop, snaps_loop = gdn.spec_forward(x, anchor)
    out_batch, snaps_batch = spec_forward_batched(gdn, x, anchor)

    rep: dict[str, object] = {}
    rep["output"] = diff_report("spec_forward output", out_batch, out_loop)
    worst_rec: dict[str, object] = {"max_diff_scale_ulps": -1.0}
    worst_conv: dict[str, object] = {"max_diff_scale_ulps": -1.0}
    for j in range(k + 1):
        r = diff_report(
            f"  snapshot[{j}].recurrent_state",
            snaps_batch[j].recurrent_state,
            snaps_loop[j].recurrent_state,
        )
        c = diff_report(
            f"  snapshot[{j}].conv_state",
            snaps_batch[j].conv_state,
            snaps_loop[j].conv_state,
        )
        if r["max_diff_scale_ulps"] > worst_rec["max_diff_scale_ulps"]:
            worst_rec = dict(r, j=j)
        if c["max_diff_scale_ulps"] > worst_conv["max_diff_scale_ulps"]:
            worst_conv = dict(c, j=j)
    rep["worst_recurrent_snapshot"] = worst_rec
    rep["worst_conv_snapshot"] = worst_conv

    # Timing: the whole point of the change.
    def timed(fn, n: int = 50) -> float:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    t_loop = timed(lambda: gdn.spec_forward(x, anchor))
    t_batch = timed(lambda: spec_forward_batched(gdn, x, anchor))
    rep["ms_loop"] = t_loop
    rep["ms_batched"] = t_batch
    rep["speedup"] = t_loop / t_batch
    print(
        f"  timing: shipped={t_loop:.3f} ms   fully-batched={t_batch:.3f} ms   "
        f"speedup={t_loop / t_batch:.2f}x"
    )
    RESULTS.setdefault("part_c", {})[f"k{k}"] = rep
    return anchor


# ---------------------------------------------------------------------------
# D. Does a perturbation of that size in the recurrent state grow or decay
#    over subsequent ordinary decode steps?
# ---------------------------------------------------------------------------


def part_d(gdn: Qwen36GatedDeltaNet, anchor: GdnLayerState, hidden: int, steps: int = 96) -> None:
    print(f"\n=== D. does a 1-ULP recurrent-state perturbation grow or decay? ({steps} steps) ===")
    print("    baseline: the SAME trajectory run with an FP32 recurrent state, i.e. the")
    print("    perturbation the shipped bf16 round-trip already injects every single step")
    tokens = torch.randn(1, steps, hidden, device=DEVICE, dtype=torch.bfloat16) * 0.1

    # ---- baseline: what does the runtime's own per-step BF16 rounding of
    # the recurrent state (the deliberate B0-4/B0-7 discipline, matching
    # transformers) cost, measured against keeping that state in FP32?
    # This is the noise floor any *additional* state perturbation has to
    # be judged against -- it is injected once per decode step, forever.
    ref32 = GdnLayerState(
        conv_state=anchor.conv_state.clone(),
        recurrent_state=anchor.recurrent_state.float(),
        has_previous_state=True,
    )
    ref16 = clone_state(anchor)
    base_traj = []
    for t in range(steps):
        xt = tokens[:, t : t + 1, :]
        gdn(xt, ref16)
        gdn(xt, ref32)
        base_traj.append(
            (ref16.recurrent_state.float() - ref32.recurrent_state.float()).abs().max().item()
        )
    base_scale = max(ref32.recurrent_state.float().abs().max().item(), 1e-30)
    print(
        f"  bf16-state vs fp32-state divergence: step1={base_traj[0]:.6g}  "
        f"step{steps}={base_traj[-1]:.6g}  (rel {base_traj[-1] / base_scale:.3%})"
    )

    clean = clone_state(anchor)
    dirty = clone_state(anchor)
    # Perturb by exactly one BF16 ULP per element, random sign -- the
    # largest perturbation the format can express as "one rounding away",
    # i.e. an upper bound on what any reduction-order change can do.
    ulp = bf16_ulp(dirty.recurrent_state)
    sign = torch.randint(0, 2, dirty.recurrent_state.shape, device=DEVICE).float() * 2 - 1
    dirty.recurrent_state.copy_((dirty.recurrent_state.float() + sign * ulp).to(torch.bfloat16))
    d0_state = (
        (dirty.recurrent_state.float() - clean.recurrent_state.float()).abs().max().item()
    )
    d0_rel = (
        (dirty.recurrent_state.float() - clean.recurrent_state.float()).abs().max().item()
        / clean.recurrent_state.float().abs().max().item()
    )
    print(f"  injected state perturbation: max_abs={d0_state:.6g} (rel to max |S| = {d0_rel:.3%})")

    traj = []
    for t in range(steps):
        xt = tokens[:, t : t + 1, :]
        y_clean = gdn(xt, clean)
        y_dirty = gdn(xt, dirty)
        s_diff = (dirty.recurrent_state.float() - clean.recurrent_state.float()).abs().max().item()
        s_rel = s_diff / max(clean.recurrent_state.float().abs().max().item(), 1e-30)
        o_diff = (y_dirty.float() - y_clean.float()).abs().max().item()
        o_ulp = (
            ((y_dirty.float() - y_clean.float()).abs() / bf16_ulp(y_clean)).max().item()
        )
        traj.append(
            {"step": t, "state_max_abs": s_diff, "state_rel": s_rel,
             "out_max_abs": o_diff, "out_max_ulps": o_ulp}
        )
        if t < 5 or (t + 1) % 16 == 0:
            print(
                f"  step {t + 1:>3}: state_diff={s_diff:.6g} (rel {s_rel:.3%})  "
                f"layer_out_diff={o_diff:.6g} ({o_ulp:.2f} ULP)"
            )
    first, last = traj[0], traj[-1]
    growth = last["state_max_abs"] / max(first["state_max_abs"], 1e-30)
    print(
        f"  growth over {steps} steps: state {first['state_max_abs']:.4g} -> "
        f"{last['state_max_abs']:.4g}  ({growth:.3f}x)"
    )
    print(
        f"  ONE 1-ULP kick after {steps} steps = {last['state_max_abs']:.4g}   vs   "
        f"{steps} steps of the shipped per-step rounding = {base_traj[-1]:.4g}   "
        f"(ratio {last['state_max_abs'] / max(base_traj[-1], 1e-30):.2f}x)"
    )
    RESULTS["part_d"] = {
        "initial": d0_state,
        "trajectory": traj,
        "bf16_vs_fp32_state_baseline": base_traj,
    }


# ---------------------------------------------------------------------------
# E. Is the "bit-exact against sequential decode" guarantee still intact
#    once TWO layers are stacked the way verify_forward stacks them?
# ---------------------------------------------------------------------------


def part_e(config: dict, quantized: dict[str, str], k: int, hidden: int) -> None:
    """Two GDN layers with the real MLP + real norms between them, run
    exactly the two ways :meth:`Qwen36TextModelSelfBuilt.verify_forward`
    and ordinary sequential decode run them.

    This uses the **shipped, unmodified** ``spec_forward`` -- nothing from
    this session is applied. If layer 1's snapshots already differ from
    sequential decode here, then the bit-exactness that
    ``in_proj_qkv``/``in_proj_z``/``out_proj`` are kept unbatched to
    preserve is a property the surrounding model has already spent, one
    layer earlier, in a line nobody proposed changing."""
    from runtime.model.qwen36_model import Qwen36RMSNorm  # noqa: PLC0415

    print(f"\n=== E. two stacked layers, verify-style vs sequential decode, K={k} ===")
    print("    (shipped spec_forward, NOTHING from this session applied)")
    gdns = [load_gdn(config, quantized, i) for i in (0, 1)]
    mlps = [load_mlp(config, quantized, i) for i in (0, 1)]
    norms = []
    for i in (0, 1):
        pair = []
        for which in ("input_layernorm", "post_attention_layernorm"):
            t = _load_prefix(f"model.language_model.layers.{i}.{which}.")
            n = Qwen36RMSNorm(hidden, eps=config["rms_norm_eps"]).to(DEVICE).to(torch.bfloat16)
            n.weight.data.copy_(t["weight"].to(torch.bfloat16))
            pair.append(n)
        norms.append(pair)

    anchors = [make_anchor(g, hidden) for g in gdns]
    x = torch.randn(1, k, hidden, device=DEVICE, dtype=torch.bfloat16) * 0.1

    # -- Attribution helpers: which of the two batched-but-claimed-safe
    # steps inside spec_forward actually is safe on THESE inputs?
    # RMSNorm's reduction is per row, and `_bmm_project` is the mechanism
    # in_proj_a/in_proj_b go through in spec_forward while ordinary
    # forward() calls the Linear directly -- both are claimed bit-exact.
    hn = norms[0][0](x)
    hn_pp = torch.cat([norms[0][0](x[:, t : t + 1, :]) for t in range(k)], dim=1)
    diff_report("input_layernorm batched", hn, hn_pp)
    diff_report(
        "_bmm_project(in_proj_b) @ seq_len=1",
        _bmm_project(gdns[0].in_proj_b, hn[:, :1, :]),
        gdns[0].in_proj_b(hn[:, :1, :]),
    )
    diff_report(
        "_bmm_project(in_proj_a) @ seq_len=1",
        _bmm_project(gdns[0].in_proj_a, hn[:, :1, :]),
        gdns[0].in_proj_a(hn[:, :1, :]),
    )
    diff_report(
        f"_bmm_project(in_proj_b) @ seq_len={k}",
        _bmm_project(gdns[0].in_proj_b, hn),
        torch.cat([gdns[0].in_proj_b(hn[:, t : t + 1, :]) for t in range(k)], dim=1),
    )

    # ---- verify-style: all K positions at once, exactly verify_forward's
    # body for two linear_attention layers.
    h = x
    verify_snaps = []
    for i in (0, 1):
        res = h
        h = norms[i][0](h)
        out, snaps = gdns[i].spec_forward(h, anchors[i])
        verify_snaps.append(snaps)
        h = res + out
        res = h
        h = norms[i][1](h)
        h = res + mlps[i](h)
    verify_hidden = h

    # ---- sequential decode: one position at a time, states advancing.
    seq_states = [clone_state(a) for a in anchors]
    seq_hidden = []
    for t in range(k):
        h = x[:, t : t + 1, :]
        for i in (0, 1):
            res = h
            h = norms[i][0](h)
            h = res + gdns[i](h, seq_states[i])
            res = h
            h = norms[i][1](h)
            h = res + mlps[i](h)
        seq_hidden.append(h)
    seq_hidden = torch.cat(seq_hidden, dim=1)

    out: dict[str, object] = {}
    for i in (0, 1):
        out[f"layer{i}_final_recurrent_state"] = diff_report(
            f"layer{i} recurrent_state after K",
            verify_snaps[i][k].recurrent_state,
            seq_states[i].recurrent_state,
        )
        out[f"layer{i}_final_conv_state"] = diff_report(
            f"layer{i} conv_state after K",
            verify_snaps[i][k].conv_state,
            seq_states[i].conv_state,
        )
    out["stack_output"] = diff_report("2-layer stack output", verify_hidden, seq_hidden)
    RESULTS["part_e"] = out
    for g in gdns:
        del g
    for m in mlps:
        del m
    torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    # Applied here, not at import: this module is also imported by
    # scripts/b3_verify_batching_logit_agreement.py purely for
    # `spec_forward_batched`, and that script DOES load the full model.
    # A module-level cap would silently strangle it.
    torch.cuda.set_per_process_memory_fraction(0.15, device=0)
    torch.manual_seed(1234)
    config = _build_qwen36_model_config(MODEL_PATH)
    quantized = quantized_layers_map(config)
    hidden = config["hidden_size"]

    part_a(config, quantized, args.k[-1], hidden)
    part_e(config, quantized, args.k[-1], hidden)

    gdn = load_gdn(config, quantized, layer_idx=0)
    part_b(gdn, args.k[-1], hidden)
    anchor = None
    for k in args.k:
        anchor = part_c(gdn, k, hidden)
    part_d(gdn, anchor, hidden, steps=args.steps)

    if args.out:
        Path(args.out).write_text(json.dumps(RESULTS, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
