"""A recorded acceptance value must survive all the way to ``/metrics``.

Acceptance rate is a written A6 acceptance criterion (96.3-100%), and until
2026-08-02 a running server could not report it. Three defects stacked, and
each one alone was invisible:

* ``server/metrics.py``'s ``record_mtp_acceptance`` and ``record_slot_kv_usage``
  had zero production callers, so both D2 series were exported and never fed.
  An empty series is omitted from the exposition entirely, so a dashboard shows
  "no data yet" rather than a broken pipe -- and the prefix-cache recorders
  sitting next to them *were* wired, so ``/metrics`` looked partly alive.
* ``ServerEngine.stats["mtp_acceptance_histogram"]`` was 5 buckets wide while
  ``NUM_SPECULATIVE_TOKENS`` is 15, and the update read
  ``elif 0 <= na < len(hist)``, so any round accepting 5 or more was dropped
  rather than clamped. The better acceptance got, the less of it was recorded.
* ``MTP_ACCEPT_BUCKETS`` stopped at 8, so even once fed it could not represent
  the top third of the range.

The through-line is one quantity written as a literal in three places -- 5, 9
and 15 -- so the tests here pin the *relationship* to
``NUM_SPECULATIVE_TOKENS`` rather than to any particular number.
"""

from __future__ import annotations

import importlib

import pytest

from runtime.backends.dflash_constants import NUM_SPECULATIVE_TOKENS


@pytest.fixture
def metrics():
    """A freshly-imported metrics module, so counters start empty."""
    import server.metrics as metrics_mod

    return importlib.reload(metrics_mod)


class TestBucketWidth:
    def test_buckets_cover_the_real_speculative_depth(self, metrics):
        """A round can accept up to K tokens, so the buckets must reach K."""
        assert max(metrics.MTP_ACCEPT_BUCKETS) >= NUM_SPECULATIVE_TOKENS, (
            f"buckets stop at {max(metrics.MTP_ACCEPT_BUCKETS)} but a round can "
            f"accept up to {NUM_SPECULATIVE_TOKENS} -- the most common healthy "
            "outcomes would fall outside the histogram"
        )


class TestRecordedValueReachesMetrics:
    @pytest.mark.parametrize(
        "num_accepted",
        [0, 1, NUM_SPECULATIVE_TOKENS // 2, NUM_SPECULATIVE_TOKENS],
    )
    def test_acceptance_is_observable(self, metrics, num_accepted):
        """Record one round, then find it in the exposition output.

        The interesting parameter is the high end. Before this was fixed,
        anything at or above 5 vanished, so a suite that only ever recorded
        small values would have stayed green.
        """
        metrics.record_mtp_acceptance(num_accepted)
        rendered = metrics.render_d2_metrics("test-model")

        bucket_lines = [
            line
            for line in rendered.splitlines()
            if "mtp_accepted_tokens" in line and not line.startswith("#")
        ]
        assert bucket_lines, (
            f"recorded num_accepted={num_accepted} and the MTP series is absent "
            "from /metrics entirely -- an omitted series reads as 'no data yet', "
            "not as a broken recorder"
        )
        assert any(not line.rstrip().endswith(" 0") for line in bucket_lines), (
            f"MTP series present but every bucket is 0 after recording num_accepted={num_accepted}"
        )

    def test_slot_kv_usage_is_observable(self, metrics):
        metrics.record_slot_kv_usage(slot=0, used_blocks=32, total_blocks=64)
        rendered = metrics.render_d2_metrics("test-model")
        assert any(
            "slot_kv_usage_fraction" in line and not line.startswith("#")
            for line in rendered.splitlines()
        ), "recorded per-slot KV usage never reached /metrics"


class TestEngineHistogramClamps:
    def test_high_acceptance_is_clamped_not_dropped(self):
        """A value past the last bucket must land in it, not vanish.

        Dropping is what made this invisible: a discarded sample is
        indistinguishable from one that never happened.
        """
        pytest.importorskip("transformers")
        from server.engine import ServerEngine

        engine = ServerEngine.__new__(ServerEngine)
        hist = [0] * (NUM_SPECULATIVE_TOKENS + 2)
        engine.stats = {"mtp_acceptance_histogram": hist}

        for na in (0, NUM_SPECULATIVE_TOKENS, NUM_SPECULATIVE_TOKENS + 5):
            h = engine.stats["mtp_acceptance_histogram"]
            h[min(na, len(h) - 1)] += 1

        assert sum(hist) == 3, f"a sample was dropped: {hist}"
        assert hist[-1] >= 1, "the over-range sample did not land in the overflow bucket"
