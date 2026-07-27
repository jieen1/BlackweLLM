"""Pure-function comparator for MoE routing tensors captured on two engines.

Compares per-(layer, token) top-k expert-id selections captured on our side
(``bfprobe/routing.py``, sites 300-302) against an oracle capture from vLLM
(see ``notes/2026-07-27-bfprobe-moe-routing-and-vllm-tap.md`` for how that
tap is obtained -- vLLM's own ``enable_return_routed_experts`` engine flag,
which returns exactly this ``topk_ids`` array with no source changes).

Deliberately framework- and bus-agnostic: every function here is pure
(no I/O, no global state, no GPU/CUDA access) and operates on anything
``numpy.asarray`` accepts -- numpy arrays, nested lists, or CPU tensors.
This is what lets the three core acceptance tests run against synthetic
fixtures with zero GPU access, and lets this module be reused unchanged
once real captures exist.

Semantic-parity note (see the design doc's checklist for the full
derivation): both engines call the identical vLLM function
(``fused_topk_bias`` -> ``ops.topk_sigmoid``) with identical arguments
(``renormalize``, ``e_score_correction_bias``, ``routed_scaling_factor=1.0``
at the router level), so the top-k order convention, renormalization, and
softcap handling are all identical by construction -- *if* the router
logits feed in identical, the ids should come out identical too. This
comparator does not assume that, though: it separately reports exact
top-1 match, exact full-sequence match, and the order-insensitive set
match / Jaccard similarity, so a case where only the *order* differs
(same top-k set) is distinguishable from a genuine divergence rather than
being misreported as one -- top-k selection kernels are not guaranteed to
return results in a fixed order and neither side's convention should be
assumed sight-unseen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LayerTokenCoord:
    """A (layer, token) coordinate into the routing grid."""

    layer: int
    token: int


@dataclass
class RoutingComparisonResult:
    """Result of comparing two engines' routing decisions.

    All per-(layer, token) arrays have shape ``(num_layers, num_tokens)``.
    """

    num_layers: int
    num_tokens: int
    top_k: int
    top1_match: np.ndarray
    """bool[layer, token]: does position-0 (highest-weight) expert id match?"""
    set_match: np.ndarray
    """bool[layer, token]: does the *set* of top-k expert ids match
    (order-insensitive)? This -- not ``sequence_match`` -- is what
    ``first_divergence`` and ``verdict`` are computed from."""
    sequence_match: np.ndarray
    """bool[layer, token]: does the exact ordered top-k sequence match?"""
    jaccard: np.ndarray
    """float[layer, token]: Jaccard similarity of the two top-k id sets."""
    first_divergence: LayerTokenCoord | None
    """First (layer, token), in layer-major scan order, where ``set_match``
    is False. ``None`` if routing agrees everywhere."""
    weight_cosine: float | None
    """Cosine similarity between the two flattened weight tensors, or
    ``None`` if weights were not supplied."""
    weight_max_abs_diff: float | None
    """Max absolute elementwise difference between the two flattened
    weight tensors, or ``None`` if weights were not supplied."""
    verdict: str
    """One-sentence, human-readable conclusion."""

    @property
    def top1_match_rate(self) -> float:
        return float(np.mean(self.top1_match)) if self.top1_match.size else 1.0

    @property
    def set_match_rate(self) -> float:
        return float(np.mean(self.set_match)) if self.set_match.size else 1.0

    @property
    def sequence_match_rate(self) -> float:
        return float(np.mean(self.sequence_match)) if self.sequence_match.size else 1.0

    @property
    def mean_jaccard(self) -> float:
        return float(np.mean(self.jaccard)) if self.jaccard.size else 1.0


def _as_ids_array(ids: Any) -> np.ndarray:
    arr = np.asarray(ids)
    if arr.ndim != 3:
        raise ValueError(
            f"expert-id tensors must be 3-D (num_layers, num_tokens, top_k), got shape {arr.shape}"
        )
    return arr


def compare_routing(
    ids_a: Any,
    ids_b: Any,
    weights_a: Any | None = None,
    weights_b: Any | None = None,
) -> RoutingComparisonResult:
    """Compare two engines' MoE routing decisions.

    Args:
        ids_a: Expert ids from engine A, shape ``(num_layers, num_tokens,
            top_k)``.
        ids_b: Expert ids from engine B, same shape as ``ids_a``.
        weights_a: Optional routing weights from engine A, any shape (only
            elementwise comparison against ``weights_b`` is done, so shape
            just has to match).
        weights_b: Optional routing weights from engine B, same shape as
            ``weights_a``.

    Returns:
        A :class:`RoutingComparisonResult`.

    Raises:
        ValueError: if ``ids_a``/``ids_b`` are not 3-D, their shapes
            disagree, or ``weights_a``/``weights_b`` shapes disagree.
    """
    arr_a = _as_ids_array(ids_a)
    arr_b = _as_ids_array(ids_b)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"routing shape mismatch: {arr_a.shape} vs {arr_b.shape}")

    num_layers, num_tokens, top_k = arr_a.shape

    top1_match = arr_a[:, :, 0] == arr_b[:, :, 0]
    sequence_match = np.all(arr_a == arr_b, axis=-1)

    set_match = np.zeros((num_layers, num_tokens), dtype=bool)
    jaccard = np.ones((num_layers, num_tokens), dtype=np.float64)
    first_divergence: LayerTokenCoord | None = None
    for layer in range(num_layers):
        for token in range(num_tokens):
            set_a = set(arr_a[layer, token].tolist())
            set_b = set(arr_b[layer, token].tolist())
            union = set_a | set_b
            inter = set_a & set_b
            jaccard[layer, token] = (len(inter) / len(union)) if union else 1.0
            is_match = set_a == set_b
            set_match[layer, token] = is_match
            if not is_match and first_divergence is None:
                first_divergence = LayerTokenCoord(layer=layer, token=token)

    weight_cosine: float | None = None
    weight_max_abs_diff: float | None = None
    if weights_a is not None or weights_b is not None:
        if weights_a is None or weights_b is None:
            raise ValueError("weights_a and weights_b must both be provided, or both omitted")
        wa = np.asarray(weights_a, dtype=np.float64).reshape(-1)
        wb = np.asarray(weights_b, dtype=np.float64).reshape(-1)
        if wa.shape != wb.shape:
            raise ValueError(f"weight shape mismatch: {wa.shape} vs {wb.shape}")
        denom = float(np.linalg.norm(wa) * np.linalg.norm(wb))
        weight_cosine = float(np.dot(wa, wb) / denom) if denom > 0 else float("nan")
        weight_max_abs_diff = float(np.max(np.abs(wa - wb))) if wa.size else 0.0

    if first_divergence is None:
        verdict = "路由完全一致,分歧只可能来自数值"
    else:
        verdict = f"路由在 layer {first_divergence.layer} token {first_divergence.token} 首次分叉"

    return RoutingComparisonResult(
        num_layers=num_layers,
        num_tokens=num_tokens,
        top_k=top_k,
        top1_match=top1_match,
        set_match=set_match,
        sequence_match=sequence_match,
        jaccard=jaccard,
        first_divergence=first_divergence,
        weight_cosine=weight_cosine,
        weight_max_abs_diff=weight_max_abs_diff,
        verdict=verdict,
    )
