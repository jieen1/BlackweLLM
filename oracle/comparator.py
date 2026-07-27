"""Dependency-light numerical checks for captured oracle tensors."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from typing import Any


def _as_values(value: Any) -> list[float]:
    """Accept lists and common tensor/array objects without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(item) for item in value]


def _top_indices(values: list[float], count: int) -> tuple[int, ...]:
    return tuple(sorted(range(len(values)), key=values.__getitem__, reverse=True)[:count])


@dataclass(frozen=True)
class ComparisonResult:
    count: int
    max_abs_error: float
    mean_abs_error: float
    cosine_similarity: float
    top_k_agreement: float

    def passes(self, *, max_abs_error: float, min_cosine: float, min_top_k: float) -> bool:
        return (
            self.max_abs_error <= max_abs_error
            and self.cosine_similarity >= min_cosine
            and self.top_k_agreement >= min_top_k
        )


def compare_values(
    reference: Iterable[float] | Any,
    candidate: Iterable[float] | Any,
    *,
    top_k: int = 10,
) -> ComparisonResult:
    """Measure error and logit-rank agreement for one captured activation."""
    expected = _as_values(reference)
    actual = _as_values(candidate)
    if not expected:
        raise ValueError("cannot compare an empty activation")
    if len(expected) != len(actual):
        raise ValueError("reference and candidate sizes differ")
    if not all(isfinite(value) for value in [*expected, *actual]):
        raise ValueError("comparison inputs must be finite")

    errors = [abs(left - right) for left, right in zip(expected, actual, strict=True)]
    dot = fsum(left * right for left, right in zip(expected, actual, strict=True))
    expected_norm = sqrt(fsum(value * value for value in expected))
    actual_norm = sqrt(fsum(value * value for value in actual))
    cosine = dot / (expected_norm * actual_norm) if expected_norm and actual_norm else 0.0
    k = min(top_k, len(expected))
    expected_top = set(_top_indices(expected, k))
    actual_top = set(_top_indices(actual, k))
    return ComparisonResult(
        count=len(expected),
        max_abs_error=max(errors),
        mean_abs_error=fsum(errors) / len(errors),
        cosine_similarity=cosine,
        top_k_agreement=len(expected_top & actual_top) / k,
    )


# ---------------------------------------------------------------------------
# Additive extensions for bfdiag's divergence scanner (bfdiag/divergence/).
# These do not change any behavior or signature above; they reuse the same
# private helpers (``_as_values``, ``_top_indices``) to add composite
# top-1/top-5 rank agreement and a depth-independent relative-error scale,
# both needed by bfdiag.divergence.scan.scan_layers.
# ---------------------------------------------------------------------------


def top_k_agreements(
    reference: Iterable[float] | Any,
    candidate: Iterable[float] | Any,
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[int, float]:
    """Rank agreement at several ``k`` values in a single pass over the data.

    Equivalent to calling ``compare_values(...).top_k_agreement`` once per
    ``k``, but only sorts each side once per requested ``k`` instead of once
    per ``compare_values`` call.
    """
    expected = _as_values(reference)
    actual = _as_values(candidate)
    if not expected:
        raise ValueError("cannot compare an empty activation")
    if len(expected) != len(actual):
        raise ValueError("reference and candidate sizes differ")
    agreements: dict[int, float] = {}
    for k in ks:
        bounded_k = min(k, len(expected))
        expected_top = set(_top_indices(expected, bounded_k))
        actual_top = set(_top_indices(actual, bounded_k))
        agreements[k] = len(expected_top & actual_top) / bounded_k
    return agreements


def activation_rms(values: Iterable[float] | Any) -> float:
    """Root-mean-square magnitude of a captured activation.

    Used to turn an absolute ``max_abs_error`` into a depth-independent
    relative error: hidden-state magnitude grows with decoder depth (residual
    stream accumulation), so a fixed absolute-error bar would misfire at
    shallow layers and go blind at deep ones.
    """
    data = _as_values(values)
    if not data:
        raise ValueError("cannot measure an empty activation")
    return sqrt(fsum(value * value for value in data) / len(data))


@dataclass(frozen=True)
class LayerComparison:
    """``ComparisonResult`` plus fixed top-1/top-5 rank agreement.

    This is the composite shape ``bfdiag.divergence.scan`` needs for its
    pass/fail judgement (cosine + relative error + top-1 + top-5), built
    entirely out of this module's existing primitives.
    """

    comparison: ComparisonResult
    top1_agreement: float
    top5_agreement: float


def compare_activation(
    reference: Iterable[float] | Any,
    candidate: Iterable[float] | Any,
    *,
    top_k: int = 10,
) -> LayerComparison:
    """``compare_values`` plus fixed top-1/top-5 rank agreement."""
    comparison = compare_values(reference, candidate, top_k=top_k)
    agreements = top_k_agreements(reference, candidate, ks=(1, 5))
    return LayerComparison(
        comparison=comparison,
        top1_agreement=agreements[1],
        top5_agreement=agreements[5],
    )
