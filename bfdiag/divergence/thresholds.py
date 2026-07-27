"""Composite, depth-relaxed divergence thresholds.

A single fixed cosine bar cannot work across an entire 48-layer decoder
stack: it either false-positives on deep layers (natural fp8/bf16
rounding-order drift compounds with depth) or false-negatives on shallow
layers (a real bug at layer 2 can still show cos=0.997, comfortably above a
loose global bar chosen to tolerate deep-layer noise).

Instead this module keeps a tight, evidence-grounded floor per submodule
*kind* at layer 0, then relaxes the allowed error budget with depth using a
capped ``sqrt(layer_idx)`` growth model (independent per-layer rounding
errors accumulate like a random walk: variance is additive over independent
steps, so the standard deviation grows with the square root of the number of
steps). Growth is capped so the relaxation cannot run away and hide a real
bug at the deepest layers.

Every constant below is argued from a real historical measurement recorded
in this repository's ``notes/`` -- see
notes/2026-07-27-bfdiag-oracle-divergence.md for the evidence table and the
worked examples per kind. None of this has been validated against a real
GPU capture yet (see that note's GPU-verification checklist).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import sqrt

# Coarse threshold "kinds". A captured submodule name is mapped onto one of
# these via ``kind_for_submodule`` before its threshold is looked up. Names
# are the real module-name suffixes read from runtime/backends/laguna.py
# (attention/MoE discovery) and the vendored vLLM model source for
# Qwen3_5DecoderLayer / Qwen3NextSparseMoeBlock (input_layernorm,
# post_attention_layernorm, self_attn, mlp.gate, mlp.experts,
# mlp.shared_expert) -- see notes/2026-07-27-bfdiag-oracle-divergence.md for
# the exact file/line evidence.
ATTN_OUT = "self_attn"
MOE_OUT = "mlp"
ROUTER_LOGITS = "mlp.gate"
INPUT_LAYERNORM = "input_layernorm"
POST_ATTENTION_LAYERNORM = "post_attention_layernorm"
HIDDEN_STATE = "hidden_state"

#: Overrides ``min_cosine`` uniformly for every kind/layer, bypassing the
#: depth model -- a quick knob for manual tightening/loosening without a
#: code change. The other three bars stay at the submodule's layer-0 floor.
ENV_THRESHOLD_OVERRIDE = "QSR_DIVERGENCE_THRESHOLD"

#: The per-layer error budget can at most triple by the deepest layers of a
#: 48-layer stack (Laguna-S-2.1: layers 0-47, see
#: runtime/backends/laguna_sparkinfer_moe.py's ``MOE_LAYER_IDS``). Chosen so
#: growth(layer=47) lands close to this cap: 1 + 0.3*sqrt(47) ~= 3.06.
_MAX_GROWTH = 3.0
_DEPTH_GROWTH_COEFFICIENT = 0.3

_TOP1_FLOOR = 0.4
_TOP5_FLOOR = 0.3
_MAX_REL_ABS_ERROR_CAP = 0.75


@dataclass(frozen=True)
class LayerThreshold:
    """Composite pass bar for one (layer, submodule) comparison."""

    min_cosine: float
    max_rel_abs_error: float
    min_top1_agreement: float
    min_top5_agreement: float


# Layer-0 floors per submodule kind. See notes/2026-07-27-bfdiag-oracle-
# divergence.md for the historical measurement each number is drawn from
# (short version: RMSNorm/elementwise ops are near bit-exact across kernel
# swaps; attention kernels validated at cos=0.999999 vs SDPA; NVFP4-quantized
# MoE has an inherent ~0.95-0.97 baseline vs bf16 truth even when correct;
# router logits are small vectors where a tiny error can flip which experts
# fire, so they get a tight bar close to attention's).
_BASE_THRESHOLDS: dict[str, LayerThreshold] = {
    INPUT_LAYERNORM: LayerThreshold(0.999999, 0.001, 1.0, 1.0),
    POST_ATTENTION_LAYERNORM: LayerThreshold(0.999999, 0.001, 1.0, 1.0),
    ATTN_OUT: LayerThreshold(0.9999, 0.01, 0.98, 0.95),
    ROUTER_LOGITS: LayerThreshold(0.999, 0.05, 0.99, 0.95),
    MOE_OUT: LayerThreshold(0.95, 0.35, 0.85, 0.75),
    HIDDEN_STATE: LayerThreshold(0.999, 0.02, 0.98, 0.95),
}
_DEFAULT_KIND_THRESHOLD = LayerThreshold(0.99, 0.05, 0.9, 0.8)


def kind_for_submodule(name: str) -> str:
    """Map a captured submodule key (e.g. ``mlp.gate``,
    ``model.layers.3.mlp``) onto one of the coarse threshold kinds above."""
    if name in _BASE_THRESHOLDS or name == HIDDEN_STATE:
        return name
    suffix = name.rsplit(".", 1)[-1]
    if suffix == "gate":
        return ROUTER_LOGITS
    if suffix in ("experts", "shared_expert"):
        return MOE_OUT
    if suffix == "mlp":
        return MOE_OUT
    if suffix == "self_attn":
        return ATTN_OUT
    if suffix == INPUT_LAYERNORM:
        return INPUT_LAYERNORM
    if suffix == POST_ATTENTION_LAYERNORM:
        return POST_ATTENTION_LAYERNORM
    return name


def _depth_growth(layer_idx: int) -> float:
    return min(_MAX_GROWTH, 1.0 + _DEPTH_GROWTH_COEFFICIENT * sqrt(max(layer_idx, 0)))


def _relax_lower_bound(base_value: float, growth: float, *, floor: float) -> float:
    """Widen the allowed ``(1 - base_value)`` budget by ``growth``x."""
    return max(floor, 1.0 - (1.0 - base_value) * growth)


def threshold_for(submodule: str, layer_idx: int) -> LayerThreshold:
    """Depth-relaxed composite threshold for one submodule at one layer.

    ``layer_idx`` is the absolute decoder-layer index (0-based); the growth
    model uses it directly rather than a fraction of total depth, so the
    same threshold applies regardless of how many layers a particular scan
    happens to cover.
    """
    kind = kind_for_submodule(submodule)
    base = _BASE_THRESHOLDS.get(kind, _DEFAULT_KIND_THRESHOLD)

    override = os.environ.get(ENV_THRESHOLD_OVERRIDE)
    if override is not None:
        return LayerThreshold(
            float(override),
            base.max_rel_abs_error,
            base.min_top1_agreement,
            base.min_top5_agreement,
        )

    growth = _depth_growth(layer_idx)
    min_cosine = _relax_lower_bound(base.min_cosine, growth, floor=0.0)
    max_rel_abs_error = min(_MAX_REL_ABS_ERROR_CAP, base.max_rel_abs_error * growth)
    min_top1 = _relax_lower_bound(base.min_top1_agreement, growth, floor=_TOP1_FLOOR)
    min_top5 = _relax_lower_bound(base.min_top5_agreement, growth, floor=_TOP5_FLOOR)
    return LayerThreshold(min_cosine, max_rel_abs_error, min_top1, min_top5)


if __name__ == "__main__":
    for layer_idx in (0, 1, 5, 17, 30, 47):
        print(f"-- layer {layer_idx} --")
        for kind in (
            INPUT_LAYERNORM,
            ATTN_OUT,
            ROUTER_LOGITS,
            MOE_OUT,
            HIDDEN_STATE,
        ):
            print(f"  {kind:26s} {threshold_for(kind, layer_idx)}")
