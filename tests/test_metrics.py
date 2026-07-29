"""D2: tests for server/metrics.py — Prometheus-style metrics."""

import server.metrics as M


class TestHistogram:
    def test_observe_single(self):
        hist = M._Histogram((1.0, 5.0, 10.0))
        hist.observe(3.0)
        entry = hist.series[()]
        assert entry[0] == 0  # <= 1.0
        assert entry[1] == 1  # <= 5.0
        assert entry[2] == 1  # <= 10.0
        assert entry[-2] == 3.0  # sum
        assert entry[-1] == 1  # count

    def test_observe_multiple(self):
        hist = M._Histogram((1.0, 5.0, 10.0))
        hist.observe(0.5)
        hist.observe(3.0)
        hist.observe(7.0)
        entry = hist.series[()]
        assert entry[0] == 1  # <= 1.0
        assert entry[1] == 2  # <= 5.0
        assert entry[2] == 3  # <= 10.0
        assert entry[-2] == 10.5  # sum
        assert entry[-1] == 3  # count

    def test_observe_with_labels(self):
        hist = M._Histogram((1.0,))
        hist.observe(0.5, labels=("ep1",))
        hist.observe(0.5, labels=("ep2",))
        assert ("ep1",) in hist.series
        assert ("ep2",) in hist.series

    def test_observe_above_all_buckets(self):
        hist = M._Histogram((1.0, 5.0))
        hist.observe(100.0)
        entry = hist.series[()]
        assert entry[0] == 0
        assert entry[1] == 0
        assert entry[-1] == 1


class TestCounter:
    def test_inc_default(self):
        counter = M._Counter()
        counter.inc()
        assert counter.series[()] == 1.0

    def test_inc_amount(self):
        counter = M._Counter()
        counter.inc(5.0)
        assert counter.series[()] == 5.0

    def test_inc_with_labels(self):
        counter = M._Counter()
        counter.inc(1.0, labels=("a", "b"))
        counter.inc(2.0, labels=("a", "b"))
        assert counter.series[("a", "b")] == 3.0


class TestRecordRequest:
    def test_record_request_updates_all(self):
        # Reset global state
        M.E2E_LATENCY.series.clear()
        M.TTFT.series.clear()
        M.TPOT.series.clear()
        M.PROMPT_TOKENS_HIST.series.clear()
        M.GENERATION_TOKENS_HIST.series.clear()
        M.PROMPT_TOKENS_TOTAL.series.clear()
        M.GENERATION_TOKENS_TOTAL.series.clear()
        M.REQUEST_SUCCESS.series.clear()

        M.record_request(
            endpoint="/v1/chat/completions",
            prompt_tokens=100,
            generation_tokens=50,
            finish_reason="stop",
            e2e_seconds=2.0,
            ttft_seconds=0.1,
        )
        ep = ("/v1/chat/completions",)
        assert M.PROMPT_TOKENS_TOTAL.series[ep] == 100.0
        assert M.GENERATION_TOKENS_TOTAL.series[ep] == 50.0
        assert M.REQUEST_SUCCESS.series[("/v1/chat/completions", "stop")] == 1.0
        assert ep in M.E2E_LATENCY.series
        assert ep in M.TTFT.series
        assert ep in M.TPOT.series

    def test_record_request_no_ttft(self):
        M.TTFT.series.clear()
        M.TPOT.series.clear()
        M.record_request(
            endpoint="/v1/completions",
            prompt_tokens=10,
            generation_tokens=1,
            finish_reason="length",
            e2e_seconds=0.5,
            ttft_seconds=None,
        )
        ep = ("/v1/completions",)
        assert ep not in M.TTFT.series


class TestRecordError:
    def test_record_error(self):
        M.REQUEST_ERRORS.series.clear()
        M.record_error("/v1/chat/completions", 500)
        assert M.REQUEST_ERRORS.series[("/v1/chat/completions", "500")] == 1.0


class TestRenderPrometheus:
    def test_render_lines(self):
        M.E2E_LATENCY.series.clear()
        M.PROMPT_TOKENS_TOTAL.series.clear()
        M.record_request("/v1/chat", 10, 5, "stop", 1.0)
        lines = M.render("test-model")
        text = "\n".join(lines)
        assert "blackwellm:e2e_request_latency_seconds" in text
        assert "blackwellm:prompt_tokens_total" in text
        assert "blackwellm:request_success_total" in text


class TestD2Metrics:
    def test_mtp_acceptance(self):
        M.mtp_acceptance_histogram.series.clear()
        M.record_mtp_acceptance(3)
        M.record_mtp_acceptance(1)
        entry = M.mtp_acceptance_histogram.series[()]
        assert entry[-1] == 2  # count

    def test_prefix_cache_hit_miss(self):
        # Reset module-level globals
        M._prefix_cache_hits = 0
        M._prefix_cache_misses = 0
        M._prefix_cache_hit_depth_sum = 0
        M.record_prefix_cache_hit(depth_blocks=5)
        M.record_prefix_cache_hit(depth_blocks=3)
        M.record_prefix_cache_miss()
        assert M._prefix_cache_hits == 2
        assert M._prefix_cache_misses == 1
        assert M._prefix_cache_hit_depth_sum == 8

    def test_slot_kv_usage(self):
        M._slot_kv_usage.clear()
        M.record_slot_kv_usage(slot=0, used_blocks=50, total_blocks=100)
        M.record_slot_kv_usage(slot=1, used_blocks=75, total_blocks=100)
        assert M._slot_kv_usage[0] == 0.5
        assert M._slot_kv_usage[1] == 0.75

    def test_render_d2_metrics(self):
        M.mtp_acceptance_histogram.series.clear()
        M._prefix_cache_hits = 0
        M._prefix_cache_misses = 0
        M._prefix_cache_hit_depth_sum = 0
        M._slot_kv_usage.clear()
        M.record_mtp_acceptance(2)
        M.record_prefix_cache_hit(4)
        M.record_slot_kv_usage(0, 10, 100)
        output = M.render_d2_metrics()
        assert "blackwellm:mtp_accepted_tokens" in output
        assert "blackwellm:prefix_cache_hits_total" in output
        assert "blackwellm:slot_kv_usage_fraction" in output

    def test_render_d2_no_data(self):
        M.mtp_acceptance_histogram.series.clear()
        M._prefix_cache_hits = 0
        M._prefix_cache_misses = 0
        M._prefix_cache_hit_depth_sum = 0
        M._slot_kv_usage.clear()
        output = M.render_d2_metrics()
        assert "blackwellm:prefix_cache_hits_total 0" in output
        assert "blackwellm:prefix_cache_misses_total 0" in output
