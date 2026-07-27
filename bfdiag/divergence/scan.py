"""Pure divergence scanner: compares oracle vs candidate per-layer activations.

Never touches a model, a GPU, or any I/O -- inputs are plain mappings of
tensor-like values (lists, tuples, numpy arrays, and CPU torch tensors all
work equally well; see ``oracle.comparator._as_values``). That is what makes
the core algorithm exhaustively unit-testable on synthetic data without a
GPU: see tests/test_bfdiag_divergence.py, which constructs a 42-layer
synthetic trace, injects a bias at one layer/submodule, and asserts
``scan_layers`` finds exactly it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bfdiag.divergence.thresholds import LayerThreshold, threshold_for
from oracle.comparator import activation_rms, compare_activation

#: ``layer_idx -> {submodule_name: tensor_like}``. The same shape is used for
#: both the oracle side and the candidate (our engine) side.
ActivationTrace = Mapping[int, Mapping[str, Any]]


@dataclass(frozen=True)
class SubmoduleVerdict:
    """One (layer, submodule) comparison and whether it passed its threshold."""

    submodule: str
    cosine_similarity: float
    max_abs_error: float
    rel_max_abs_error: float
    mean_abs_error: float
    top1_agreement: float
    top5_agreement: float
    threshold: LayerThreshold
    passed: bool

    def reason(self) -> str:
        """Short human-readable explanation of why this verdict failed."""
        if self.passed:
            return "ok"
        reasons = []
        if self.cosine_similarity < self.threshold.min_cosine:
            reasons.append(f"cos={self.cosine_similarity:.4f}<{self.threshold.min_cosine:.4f}")
        if self.rel_max_abs_error > self.threshold.max_rel_abs_error:
            reasons.append(
                f"rel_max_abs_err={self.rel_max_abs_error:.4f}>"
                f"{self.threshold.max_rel_abs_error:.4f}"
            )
        if self.top1_agreement < self.threshold.min_top1_agreement:
            reasons.append(f"top1={self.top1_agreement:.4f}<{self.threshold.min_top1_agreement:.4f}")
        if self.top5_agreement < self.threshold.min_top5_agreement:
            reasons.append(f"top5={self.top5_agreement:.4f}<{self.threshold.min_top5_agreement:.4f}")
        return ", ".join(reasons) if reasons else "ok"


@dataclass(frozen=True)
class LayerVerdict:
    """All submodule verdicts captured for one decoder layer."""

    layer_idx: int
    submodules: tuple[SubmoduleVerdict, ...]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.submodules)

    @property
    def worst_submodules(self) -> tuple[SubmoduleVerdict, ...]:
        """Failing submodules ordered worst-first (lowest cosine first)."""
        failing = [item for item in self.submodules if not item.passed]
        return tuple(sorted(failing, key=lambda item: item.cosine_similarity))


@dataclass(frozen=True)
class DivergenceReport:
    """Full per-layer scan result, with the first divergent layer highlighted."""

    layers: tuple[LayerVerdict, ...]

    @property
    def first_divergent_layer(self) -> int | None:
        for layer in self.layers:
            if not layer.passed:
                return layer.layer_idx
        return None

    @property
    def first_divergent_submodules(self) -> tuple[str, ...]:
        layer_idx = self.first_divergent_layer
        if layer_idx is None:
            return ()
        layer = next(item for item in self.layers if item.layer_idx == layer_idx)
        return tuple(item.submodule for item in layer.worst_submodules)

    @property
    def has_divergence(self) -> bool:
        return self.first_divergent_layer is not None


def scan_layers(
    oracle: ActivationTrace,
    candidate: ActivationTrace,
    *,
    top_k: int = 10,
) -> DivergenceReport:
    """Compare oracle vs candidate activations layer by layer.

    Both ``oracle`` and ``candidate`` map ``layer_idx -> {submodule_name:
    tensor_like}``. Only layers present on both sides are scanned (in
    ascending layer order); only submodules present on both sides of a given
    layer are compared -- a submodule missing from one side is silently
    skipped rather than treated as a failure, since capture configs may
    legitimately hook different submodule sets on each side.

    Layers are evaluated in order and the first layer with >=1 failing
    submodule is reported as ``first_divergent_layer``; that layer's failing
    submodules are exposed via ``first_divergent_submodules``
    (worst-cosine-first), giving the "drill down to the offending
    submodule" behavior the CLI report renders.
    """
    if not oracle:
        raise ValueError("oracle activation trace is empty")
    if not candidate:
        raise ValueError("candidate activation trace is empty")
    shared_layers = sorted(set(oracle) & set(candidate))
    if not shared_layers:
        raise ValueError("oracle and candidate traces share no layer indices")

    layer_verdicts: list[LayerVerdict] = []
    for layer_idx in shared_layers:
        oracle_layer = oracle[layer_idx]
        candidate_layer = candidate[layer_idx]
        shared_submodules = sorted(set(oracle_layer) & set(candidate_layer))
        verdicts = []
        for name in shared_submodules:
            reference = oracle_layer[name]
            actual = candidate_layer[name]
            comparison = compare_activation(reference, actual, top_k=top_k)
            ref_rms = activation_rms(reference)
            rel_max_abs_error = (
                comparison.comparison.max_abs_error / ref_rms
                if ref_rms > 0
                else comparison.comparison.max_abs_error
            )
            threshold = threshold_for(name, layer_idx)
            passed = (
                comparison.comparison.cosine_similarity >= threshold.min_cosine
                and rel_max_abs_error <= threshold.max_rel_abs_error
                and comparison.top1_agreement >= threshold.min_top1_agreement
                and comparison.top5_agreement >= threshold.min_top5_agreement
            )
            verdicts.append(
                SubmoduleVerdict(
                    submodule=name,
                    cosine_similarity=comparison.comparison.cosine_similarity,
                    max_abs_error=comparison.comparison.max_abs_error,
                    rel_max_abs_error=rel_max_abs_error,
                    mean_abs_error=comparison.comparison.mean_abs_error,
                    top1_agreement=comparison.top1_agreement,
                    top5_agreement=comparison.top5_agreement,
                    threshold=threshold,
                    passed=passed,
                )
            )
        layer_verdicts.append(LayerVerdict(layer_idx=layer_idx, submodules=tuple(verdicts)))

    return DivergenceReport(layers=tuple(layer_verdicts))


if __name__ == "__main__":
    # Minimal self-test: two identical single-layer traces never diverge.
    trace = {0: {"self_attn": [1.0, 2.0, 3.0, 4.0]}}
    report = scan_layers(trace, trace)
    print("has_divergence:", report.has_divergence)
    assert not report.has_divergence
