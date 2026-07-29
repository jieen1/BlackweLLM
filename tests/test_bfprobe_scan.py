"""Unit tests for bfprobe/scan.py -- CPU-only, synthetic data only.

The central acceptance test: construct a 48-layer signature sequence, inject
an absmax spike at layer 31, and assert the scan reports
``first_out_of_band.layer == 31``. Plus the two required companion cases
(natural deep-layer drift must not false-positive; NaN must be caught at any
depth) and CLI-facing report rendering / JSON shape checks.
"""

from __future__ import annotations

import json
from dataclasses import replace

from bfprobe.baseline import absmax_ratio_bound, record_baseline
from bfprobe.reduce import Signature
from bfprobe.scan import (
    SITE_ATTN_OUT,
    SITE_INPUT_LAYERNORM,
    SITE_MOE_OUT,
    SITE_POST_ATTENTION_LAYERNORM,
    format_text_report,
    scan,
    site_name,
    to_json_dict,
)
from bfprobe.signature import SignatureRecord

NUM_LAYERS = 48
ALL_SITES = (SITE_INPUT_LAYERNORM, SITE_ATTN_OUT, SITE_POST_ATTENTION_LAYERNORM, SITE_MOE_OUT)

_GOOD = Signature(absmax=1.0, l2=10.0, mean=0.1, nan_count=0, inf_count=0, numel=100)


def _make_baseline():
    signatures = [
        (site_id, layer, _GOOD) for layer in range(NUM_LAYERS) for site_id in ALL_SITES
    ]
    return record_baseline(signatures, model_revision="qwen3.6-demo", git_sha="deadbeef")


def _clean_round(round_idx: int) -> list[SignatureRecord]:
    """One full round of 48 layers x 4 taps, all matching the baseline
    exactly (so nothing here should ever be flagged)."""
    records = []
    seq = round_idx * NUM_LAYERS * len(ALL_SITES)
    for layer in range(NUM_LAYERS):
        for site_id in ALL_SITES:
            records.append(
                SignatureRecord(
                    seq=seq,
                    site_id=site_id,
                    round_idx=round_idx,
                    layer=layer,
                    absmax=_GOOD.absmax,
                    l2=_GOOD.l2,
                    mean=_GOOD.mean,
                    nan_count=0,
                    inf_count=0,
                    numel=_GOOD.numel,
                )
            )
            seq += 1
    return records


class TestFirstOutOfBandLocatesInjectedLayer:
    def test_absmax_spike_at_layer_31_is_located_exactly(self):
        baseline = _make_baseline()
        records = _clean_round(round_idx=0)
        # Inject an unmistakable absmax spike at layer 31, one tap only.
        spike_idx = next(
            i
            for i, r in enumerate(records)
            if r.layer == 31 and r.site_id == SITE_MOE_OUT
        )
        records[spike_idx] = replace(records[spike_idx], absmax=500.0)

        report = scan(records, baseline)
        assert report.has_out_of_band
        assert report.first_out_of_band.layer == 31
        assert report.first_out_of_band.site_id == SITE_MOE_OUT
        assert report.first_out_of_band.round_idx == 0

    def test_earlier_layer_wins_when_multiple_layers_are_bad(self):
        # If corruption at an early layer also drags down later layers, the
        # report must point at the *shallowest* bad layer -- that's the
        # likely root cause, not the propagated symptom.
        baseline = _make_baseline()
        records = _clean_round(round_idx=0)
        for i, r in enumerate(records):
            if r.layer in (10, 31, 40) and r.site_id == SITE_MOE_OUT:
                records[i] = replace(r, absmax=500.0)

        report = scan(records, baseline)
        assert report.first_out_of_band.layer == 10

    def test_spike_in_a_later_round_is_found_not_the_first_round(self):
        baseline = _make_baseline()
        records = _clean_round(round_idx=0) + _clean_round(round_idx=1)
        spike_idx = next(
            i
            for i, r in enumerate(records)
            if r.round_idx == 1 and r.layer == 20 and r.site_id == SITE_ATTN_OUT
        )
        records[spike_idx] = replace(records[spike_idx], absmax=500.0)
        report = scan(records, baseline)
        assert report.first_out_of_band.round_idx == 1
        assert report.first_out_of_band.layer == 20


class TestNaturalDeepLayerDriftDoesNotFalsePositive:
    def test_deep_layer_within_relaxed_bound_passes(self):
        baseline = _make_baseline()
        records = _clean_round(round_idx=0)
        bound = absmax_ratio_bound(47)
        drifted_absmax = _GOOD.absmax * (bound - 0.01)
        idx = next(
            i for i, r in enumerate(records) if r.layer == 47 and r.site_id == SITE_MOE_OUT
        )
        records[idx] = replace(records[idx], absmax=drifted_absmax)

        report = scan(records, baseline)
        assert not report.has_out_of_band
        assert report.first_out_of_band is None

    def test_same_drift_would_have_failed_at_layer_zero(self):
        # Sanity check that the "no false positive" case above is actually
        # testing something -- the same absolute drift ratio does trip the
        # tighter layer-0 bound.
        baseline = _make_baseline()
        bound = absmax_ratio_bound(47)
        drifted_absmax = _GOOD.absmax * (bound - 0.01)
        records = _clean_round(round_idx=0)
        idx = next(
            i for i, r in enumerate(records) if r.layer == 0 and r.site_id == SITE_MOE_OUT
        )
        records[idx] = replace(records[idx], absmax=drifted_absmax)
        report = scan(records, baseline)
        assert report.has_out_of_band
        assert report.first_out_of_band.layer == 0


class TestNanAlwaysCaughtRegardlessOfDepth:
    def test_nan_at_shallow_layer(self):
        baseline = _make_baseline()
        records = _clean_round(round_idx=0)
        idx = next(
            i for i, r in enumerate(records) if r.layer == 2 and r.site_id == SITE_INPUT_LAYERNORM
        )
        records[idx] = replace(records[idx], nan_count=1)
        report = scan(records, baseline)
        assert report.has_out_of_band
        assert report.first_out_of_band.layer == 2

    def test_nan_at_deepest_layer(self):
        baseline = _make_baseline()
        records = _clean_round(round_idx=0)
        idx = next(
            i for i, r in enumerate(records) if r.layer == 47 and r.site_id == SITE_MOE_OUT
        )
        records[idx] = replace(records[idx], nan_count=1)
        report = scan(records, baseline)
        assert report.has_out_of_band
        assert report.first_out_of_band.layer == 47
        assert any("nan_count" in reason for reason in report.first_out_of_band.reasons)


class TestScanReportShape:
    def test_all_clean_produces_no_out_of_band(self):
        baseline = _make_baseline()
        records = _clean_round(round_idx=0)
        report = scan(records, baseline)
        assert not report.has_out_of_band
        assert len(report.verdicts) == len(records)
        assert all(not v.verdict.out_of_band for v in report.verdicts)

    def test_missing_baseline_entries_are_skipped_not_failed(self):
        baseline = _make_baseline()
        records = _clean_round(round_idx=0)
        records.append(
            SignatureRecord(
                seq=99999,
                site_id=299,  # never in the baseline
                round_idx=0,
                layer=0,
                absmax=1.0,
                l2=1.0,
                mean=0.1,
                nan_count=0,
                inf_count=0,
                numel=10,
            )
        )
        report = scan(records, baseline)
        assert not report.has_out_of_band
        assert len(report.skipped_no_baseline) == 1
        assert report.skipped_no_baseline[0].site_id == 299

    def test_timeline_for_returns_all_rounds_for_one_tap(self):
        baseline = _make_baseline()
        records = _clean_round(0) + _clean_round(1) + _clean_round(2)
        report = scan(records, baseline)
        timeline = report.timeline_for(SITE_MOE_OUT, 5)
        assert len(timeline) == 3
        assert [v.record.round_idx for v in timeline] == [0, 1, 2]


class TestReportRendering:
    def test_json_report_shape_when_clean(self):
        baseline = _make_baseline()
        records = _clean_round(0)
        report = scan(records, baseline)
        payload = to_json_dict(report)
        assert payload["has_out_of_band"] is False
        assert payload["first_out_of_band"] is None
        assert payload["num_judged"] == len(records)
        assert payload["num_out_of_band"] == 0
        # Must be JSON-serializable as-is.
        json.dumps(payload)

    def test_json_report_shape_when_out_of_band(self):
        baseline = _make_baseline()
        records = _clean_round(0)
        idx = next(
            i for i, r in enumerate(records) if r.layer == 31 and r.site_id == SITE_MOE_OUT
        )
        records[idx] = replace(records[idx], absmax=500.0)
        report = scan(records, baseline)
        payload = to_json_dict(report)
        assert payload["has_out_of_band"] is True
        assert payload["first_out_of_band"]["layer"] == 31
        assert payload["first_out_of_band"]["site_name"] == site_name(SITE_MOE_OUT)
        json.dumps(payload)

    def test_text_report_mentions_layer_and_reason_when_out_of_band(self):
        baseline = _make_baseline()
        records = _clean_round(0)
        idx = next(
            i for i, r in enumerate(records) if r.layer == 31 and r.site_id == SITE_MOE_OUT
        )
        records[idx] = replace(records[idx], absmax=500.0)
        report = scan(records, baseline)
        text = format_text_report(report)
        assert "layer 31" in text
        assert "absmax" in text

    def test_text_report_clean_message(self):
        baseline = _make_baseline()
        records = _clean_round(0)
        report = scan(records, baseline)
        text = format_text_report(report)
        assert "未发现越界" in text
