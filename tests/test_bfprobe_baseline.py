"""Unit tests for bfprobe/baseline.py -- CPU-only, no GPU or model involved.

Covers: the depth-relaxed threshold growth model (shape-consistent with
bfdiag/divergence/thresholds.py's sqrt(layer_idx) growth), baseline
recording (max-absmax / mean-l2 folding across repeated observations),
JSON persistence round-trip, and the pure judge() decision function's four
required behaviors: NaN/Inf are always fatal, absmax/L2 bounds widen with
depth, and natural deep-layer drift within the relaxed bound is not a false
positive.
"""

from __future__ import annotations

import math

import pytest

from bfprobe.baseline import (
    Baseline,
    BaselineEntry,
    BaselineFingerprint,
    absmax_ratio_bound,
    judge,
    l2_rel_dev_bound,
    load_baseline,
    record_baseline,
    save_baseline,
)
from bfprobe.reduce import Signature


class TestDepthGrowthModel:
    def test_growth_increases_monotonically_with_depth(self):
        bounds = [absmax_ratio_bound(layer) for layer in (0, 1, 5, 17, 30, 47)]
        assert bounds == sorted(bounds)
        assert bounds[0] < bounds[-1]

    def test_growth_is_capped(self):
        # _MAX_GROWTH = 3.0 caps the multiplier; far beyond layer 47 the
        # bound should stop growing (matches bfdiag/divergence/thresholds
        # .py's _MAX_GROWTH cap, reused here for consistency).
        bound_at_47 = absmax_ratio_bound(47)
        bound_at_1000 = absmax_ratio_bound(1000)
        assert bound_at_1000 == pytest.approx(bound_at_47)

    def test_layer_zero_is_the_tightest_bound(self):
        assert absmax_ratio_bound(0) < absmax_ratio_bound(1)
        assert l2_rel_dev_bound(0) < l2_rel_dev_bound(1)

    def test_l2_bound_scales_the_same_way(self):
        bounds = [l2_rel_dev_bound(layer) for layer in (0, 5, 17, 47)]
        assert bounds == sorted(bounds)


class TestRecordBaseline:
    def test_empty_signatures_rejected(self):
        with pytest.raises(ValueError):
            record_baseline([], model_revision="rev", git_sha="sha")

    def test_single_observation_per_site_layer(self):
        sig = Signature(absmax=1.0, l2=2.0, mean=0.5, nan_count=0, inf_count=0, numel=10)
        baseline = record_baseline([(200, 3, sig)], model_revision="rev", git_sha="sha")
        entry = baseline.entry_for(200, 3)
        assert entry is not None
        assert entry.absmax == 1.0
        assert entry.l2 == 2.0
        assert entry.mean == 0.5
        assert entry.numel == 10
        assert baseline.fingerprint == BaselineFingerprint("rev", "sha")

    def test_absmax_takes_the_max_across_rounds(self):
        sigs = [
            (200, 0, Signature(1.0, 5.0, 0.1, 0, 0, 10)),
            (200, 0, Signature(3.0, 5.0, 0.1, 0, 0, 10)),
            (200, 0, Signature(2.0, 5.0, 0.1, 0, 0, 10)),
        ]
        baseline = record_baseline(sigs, model_revision="rev", git_sha="sha")
        assert baseline.entry_for(200, 0).absmax == 3.0

    def test_l2_and_mean_take_the_average_across_rounds(self):
        sigs = [
            (200, 0, Signature(1.0, 4.0, 0.2, 0, 0, 10)),
            (200, 0, Signature(1.0, 6.0, 0.4, 0, 0, 10)),
        ]
        baseline = record_baseline(sigs, model_revision="rev", git_sha="sha")
        entry = baseline.entry_for(200, 0)
        assert entry.l2 == pytest.approx(5.0)
        assert entry.mean == pytest.approx(0.3)

    def test_distinct_site_layer_pairs_kept_separate(self):
        sigs = [
            (200, 0, Signature(1.0, 1.0, 0.1, 0, 0, 10)),
            (200, 1, Signature(2.0, 2.0, 0.2, 0, 0, 10)),
            (201, 0, Signature(3.0, 3.0, 0.3, 0, 0, 10)),
        ]
        baseline = record_baseline(sigs, model_revision="rev", git_sha="sha")
        assert len(baseline.entries) == 3
        assert baseline.entry_for(200, 0).absmax == 1.0
        assert baseline.entry_for(200, 1).absmax == 2.0
        assert baseline.entry_for(201, 0).absmax == 3.0

    def test_missing_entry_returns_none(self):
        sigs = [(200, 0, Signature(1.0, 1.0, 0.1, 0, 0, 10))]
        baseline = record_baseline(sigs, model_revision="rev", git_sha="sha")
        assert baseline.entry_for(999, 999) is None


class TestBaselinePersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        sigs = [
            (200 + site, layer, Signature(float(site + layer), 1.0, 0.1, 0, 0, 100))
            for site in range(4)
            for layer in range(48)
        ]
        baseline = record_baseline(sigs, model_revision="qwen3.6-rev1", git_sha="abc123")
        path = tmp_path / "baseline.json"
        save_baseline(baseline, path)
        loaded = load_baseline(path)

        assert loaded.fingerprint == baseline.fingerprint
        assert set(loaded.entries) == set(baseline.entries)
        for key, entry in baseline.entries.items():
            assert loaded.entries[key] == entry

    def test_save_creates_parent_directories(self, tmp_path):
        sigs = [(200, 0, Signature(1.0, 1.0, 0.1, 0, 0, 10))]
        baseline = record_baseline(sigs, model_revision="rev", git_sha="sha")
        path = tmp_path / "nested" / "dir" / "baseline.json"
        save_baseline(baseline, path)
        assert path.exists()
        loaded = load_baseline(path)
        assert loaded.entry_for(200, 0).absmax == 1.0


class TestJudge:
    def _entry(self, layer: int, *, absmax: float = 1.0, l2: float = 10.0) -> BaselineEntry:
        return BaselineEntry(site_id=200, layer=layer, absmax=absmax, l2=l2, mean=0.1, numel=100)

    def test_matching_signature_passes(self):
        entry = self._entry(layer=10)
        current = Signature(absmax=1.0, l2=10.0, mean=0.1, nan_count=0, inf_count=0, numel=100)
        verdict = judge(entry, current)
        assert not verdict.out_of_band
        assert verdict.reasons == ()

    def test_nan_is_always_fatal_regardless_of_depth(self):
        for layer in (0, 5, 17, 47):
            entry = self._entry(layer=layer)
            current = Signature(
                absmax=1.0, l2=10.0, mean=0.1, nan_count=1, inf_count=0, numel=100
            )
            verdict = judge(entry, current)
            assert verdict.out_of_band, f"layer {layer} should be fatal on nan_count>0"
            assert any("nan_count" in reason for reason in verdict.reasons)

    def test_inf_is_always_fatal_regardless_of_depth(self):
        for layer in (0, 5, 17, 47):
            entry = self._entry(layer=layer)
            current = Signature(
                absmax=1.0, l2=10.0, mean=0.1, nan_count=0, inf_count=1, numel=100
            )
            verdict = judge(entry, current)
            assert verdict.out_of_band, f"layer {layer} should be fatal on inf_count>0"
            assert any("inf_count" in reason for reason in verdict.reasons)

    def test_absmax_spike_flagged_at_shallow_layer(self):
        entry = self._entry(layer=0, absmax=1.0)
        # Layer 0's bound is baseline * 1.5 -- a 50x spike is unambiguous.
        current = Signature(absmax=50.0, l2=10.0, mean=0.1, nan_count=0, inf_count=0, numel=100)
        verdict = judge(entry, current)
        assert verdict.out_of_band
        assert any("absmax" in reason for reason in verdict.reasons)

    def test_natural_deep_layer_drift_does_not_false_positive(self):
        # A modest absmax increase that would fail at layer 0 must pass at a
        # deep layer, because the bound has widened with depth -- this is
        # the whole point of the depth-relaxed model (bfdiag/divergence
        # /thresholds.py's rationale, reused here: independent per-layer
        # rounding errors accumulate like a random walk).
        entry = self._entry(layer=47, absmax=1.0, l2=10.0)
        bound = absmax_ratio_bound(47)
        assert bound > 1.5  # confirm the deep-layer bound is actually wider
        drifted_absmax = entry.absmax * (bound - 0.01)  # just inside the wider bound
        current = Signature(
            absmax=drifted_absmax, l2=10.0, mean=0.1, nan_count=0, inf_count=0, numel=100
        )
        verdict = judge(entry, current)
        assert not verdict.out_of_band, verdict.reasons

    def test_same_relative_drift_flagged_at_shallow_but_not_deep_layer(self):
        # The concrete "fixed threshold would misfire" scenario: a drift
        # ratio that sits between the shallow and deep bounds.
        shallow_bound = absmax_ratio_bound(0)
        deep_bound = absmax_ratio_bound(47)
        assert deep_bound > shallow_bound
        mid_ratio = (shallow_bound + deep_bound) / 2

        shallow_entry = self._entry(layer=0, absmax=1.0)
        deep_entry = self._entry(layer=47, absmax=1.0)
        drifted = Signature(
            absmax=mid_ratio, l2=10.0, mean=0.1, nan_count=0, inf_count=0, numel=100
        )

        shallow_verdict = judge(shallow_entry, drifted)
        deep_verdict = judge(deep_entry, drifted)
        assert shallow_verdict.out_of_band
        assert not deep_verdict.out_of_band

    def test_l2_relative_deviation_flagged(self):
        entry = self._entry(layer=0, l2=10.0)
        # 200% relative deviation, far past layer 0's 50% bound.
        current = Signature(absmax=1.0, l2=30.0, mean=0.1, nan_count=0, inf_count=0, numel=100)
        verdict = judge(entry, current)
        assert verdict.out_of_band
        assert any("l2_rel_dev" in reason for reason in verdict.reasons)

    def test_zero_baseline_absmax_any_nonzero_current_flagged(self):
        entry = self._entry(layer=0, absmax=0.0)
        current = Signature(absmax=0.1, l2=10.0, mean=0.1, nan_count=0, inf_count=0, numel=100)
        verdict = judge(entry, current)
        assert verdict.out_of_band

    def test_zero_baseline_and_zero_current_passes(self):
        entry = self._entry(layer=0, absmax=0.0, l2=0.0)
        current = Signature(absmax=0.0, l2=0.0, mean=0.0, nan_count=0, inf_count=0, numel=100)
        verdict = judge(entry, current)
        assert not verdict.out_of_band


def test_baseline_dataclass_is_frozen():
    fingerprint = BaselineFingerprint("rev", "sha")
    baseline = Baseline(fingerprint=fingerprint, entries={})
    with pytest.raises(Exception):
        baseline.fingerprint = fingerprint  # type: ignore[misc]


def test_absmax_ratio_bound_is_finite_and_positive():
    for layer in range(0, 100, 7):
        bound = absmax_ratio_bound(layer)
        assert math.isfinite(bound)
        assert bound > 1.0
