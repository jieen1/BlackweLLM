from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("transformers")

from benchmarks.server_perf_grid import (
    _completion_evidence,
    diff_stats,
    snapshot_stats,
    wave_summary,
)


def test_acceptance_histogram_is_diffed_per_wave() -> None:
    before = {"requests_completed": 3, "mtp_acceptance_histogram": [1, 2, 3, 4]}
    after = {"requests_completed": 4, "mtp_acceptance_histogram": [2, 4, 6, 8]}

    delta = diff_stats(before, after)

    assert delta["requests_completed"] == 1
    assert delta["mtp_acceptance_histogram"] == [1, 2, 3, 4]
    assert delta["mtp_rounds"] == 10
    assert delta["mtp_accepted_tokens"] == 20
    assert delta["mtp_mean_accepted_per_round"] == 2.0
    assert delta["mtp_mean_committed_per_round"] == 3.0


def test_stats_snapshot_keeps_engine_acceptance_histogram() -> None:
    snapshot = snapshot_stats(
        {
            "requests_completed": 1,
            "prefix_cache_hits": 0,
            "prefix_cache_misses": 1,
            "mtp_acceptance_histogram": [7, 31, 21, 31],
            "_backend_stats_dbg": {},
        }
    )

    assert snapshot["mtp_acceptance_histogram"] == [7, 31, 21, 31]


def test_completion_evidence_is_exact_and_reproducible() -> None:
    text = "Qwen3.8 output \u2713"

    evidence = _completion_evidence(text)

    assert evidence["completion_text"] == text
    assert evidence["completion_chars"] == len(text)
    assert evidence["completion_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_wave_summary_retains_request_completion_evidence() -> None:
    request = {
        "error": None,
        "ttft_s": 0.5,
        "last_event_s": 0.9,
        **_completion_evidence("same greedy output"),
    }

    summary = wave_summary(
        [request],
        wall=1.0,
        stats_delta={"requests_completed": 1},
        metric_delta={},
    )

    assert summary["requests"] == [request]
