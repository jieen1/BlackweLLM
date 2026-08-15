"""D3: tests for server/tracing.py — request-level tracing."""

import time

from server.tracing import RequestTrace, RequestTracer


class TestRequestTrace:
    def test_basic_lifecycle(self):
        trace = RequestTrace(request_id="req-1", slot=0, prompt_len=100)
        trace.admitted_at = time.perf_counter()
        trace.prefill_ms = 5.0
        trace.total_rounds = 3
        trace.total_tokens = 9
        trace.decode_rounds = [(0, 3, 10.0), (1, 3, 10.0), (2, 3, 10.0)]
        trace.finished_at = trace.admitted_at + 0.05
        trace.finish_reason = "stop"
        assert trace.total_ms > 0
        assert trace.decode_ms == 30.0
        assert trace.avg_round_ms == 10.0
        assert trace.tokens_per_sec > 0

    def test_empty_trace(self):
        trace = RequestTrace(request_id="req-2")
        assert trace.total_ms == 0.0
        assert trace.decode_ms == 0.0
        assert trace.avg_round_ms == 0.0
        assert trace.tokens_per_sec == 0.0

    def test_to_dict(self):
        trace = RequestTrace(request_id="req-3", slot=1, prompt_len=50)
        trace.admitted_at = time.perf_counter()
        trace.finished_at = trace.admitted_at + 0.1
        trace.prefill_ms = 2.0
        trace.total_rounds = 2
        trace.total_tokens = 6
        trace.decode_rounds = [(0, 3, 15.0), (1, 3, 15.0)]
        trace.finish_reason = "stop"
        d = trace.to_dict()
        assert d["request_id"] == "req-3"
        assert d["slot"] == 1
        assert d["prompt_len"] == 50
        assert d["total_rounds"] == 2
        assert d["total_tokens"] == 6
        assert d["finish_reason"] == "stop"
        assert "total_ms" in d
        assert "prefill_ms" in d
        assert "decode_ms" in d
        assert "avg_round_ms" in d
        assert "tokens_per_sec" in d
        assert d["decode_rounds"] == [
            {"round_idx": 0, "tokens_committed": 3, "round_ms": 15.0},
            {"round_idx": 1, "tokens_committed": 3, "round_ms": 15.0},
        ]


class TestRequestTracer:
    def test_full_lifecycle(self):
        tracer = RequestTracer(enabled=True, sample_rate=1.0)
        tracer.request_admitted("r1", slot=0, prompt_len=100)
        tracer.prefill_done("r1", prefill_ms=5.0)
        tracer.decode_round("r1", round_idx=0, tokens_committed=3, round_ms=10.0)
        tracer.decode_round("r1", round_idx=1, tokens_committed=2, round_ms=8.0)
        tracer.request_finished("r1", finish_reason="stop")
        trace = tracer.get_trace("r1")
        assert trace is not None
        assert trace["request_id"] == "r1"
        assert trace["total_rounds"] == 2
        assert trace["total_tokens"] == 5
        assert trace["finish_reason"] == "stop"

    def test_disabled_tracer(self):
        tracer = RequestTracer(enabled=False)
        tracer.request_admitted("r1", slot=0, prompt_len=100)
        tracer.prefill_done("r1", prefill_ms=5.0)
        tracer.request_finished("r1", finish_reason="stop")
        assert tracer.get_trace("r1") is None
        assert tracer.get_stats()["total_requests"] == 0

    def test_slow_request_detection(self):
        tracer = RequestTracer(enabled=True, slow_threshold_ms=0.001)
        tracer.request_admitted("slow-1", slot=0, prompt_len=10)
        time.sleep(0.01)
        tracer.request_finished("slow-1", finish_reason="stop")
        slow = tracer.get_slow_requests()
        assert len(slow) == 1
        assert slow[0]["request_id"] == "slow-1"

    def test_get_recent(self):
        tracer = RequestTracer(enabled=True)
        for i in range(5):
            rid = f"r{i}"
            tracer.request_admitted(rid, slot=0, prompt_len=10)
            tracer.request_finished(rid, finish_reason="stop")
        recent = tracer.get_recent(limit=3)
        assert len(recent) == 3
        assert recent[0]["request_id"] == "r4"

    def test_get_stats_empty(self):
        tracer = RequestTracer(enabled=True)
        stats = tracer.get_stats()
        assert stats["total_requests"] == 0

    def test_get_stats_with_data(self):
        tracer = RequestTracer(enabled=True)
        tracer.request_admitted("r1", slot=0, prompt_len=10)
        tracer.prefill_done("r1", prefill_ms=5.0)
        tracer.decode_round("r1", 0, 3, 10.0)
        tracer.request_finished("r1", "stop")
        stats = tracer.get_stats()
        assert stats["total_requests"] == 1
        assert stats["completed"] == 1

    def test_unknown_request_ignored(self):
        tracer = RequestTracer(enabled=True)
        tracer.prefill_done("nonexistent", prefill_ms=5.0)
        tracer.decode_round("nonexistent", 0, 1, 1.0)
        tracer.request_finished("nonexistent", "stop")
        assert tracer.get_trace("nonexistent") is None

    def test_render_prometheus(self):
        tracer = RequestTracer(enabled=True)
        tracer.request_admitted("r1", slot=0, prompt_len=10)
        tracer.request_finished("r1", "stop")
        output = tracer.render_prometheus()
        assert "blackwellm:trace_total_requests" in output
        assert "blackwellm:trace_active_requests" in output
        assert "blackwellm:trace_slow_requests" in output

    def test_max_rounds_per_trace(self):
        tracer = RequestTracer(enabled=True, max_rounds_per_trace=3)
        tracer.request_admitted("r1", slot=0, prompt_len=10)
        for i in range(10):
            tracer.decode_round("r1", i, 1, 1.0)
        tracer.request_finished("r1", "stop")
        trace = tracer.get_trace("r1")
        assert trace["total_rounds"] == 10
        assert trace["total_tokens"] == 10

    def test_active_request_trace(self):
        tracer = RequestTracer(enabled=True)
        tracer.request_admitted("active-1", slot=2, prompt_len=200)
        trace = tracer.get_trace("active-1")
        assert trace is not None
        assert trace["request_id"] == "active-1"
        assert trace["slot"] == 2


class TestPrometheusOnFreshServer:
    """`/metrics` used to 500 until the first request completed.

    `get_stats()` short-circuited to a two-key dict while `_completed` was
    empty, and `render_prometheus` reads six. A scraper pointed at a freshly
    started server therefore hit `KeyError: 'slow_count'` -- observed live on
    2026-08-01, repeating on every scrape for the whole warm-up window.
    """

    def test_render_prometheus_before_any_request_completes(self):
        from server.tracing import RequestTracer

        tracer = RequestTracer(enabled=True)
        rendered = tracer.render_prometheus("some-model")
        assert "blackwellm:trace_slow_requests" in rendered
        assert "blackwellm:trace_total_requests" in rendered

    def test_get_stats_key_set_does_not_depend_on_completion(self):
        from server.tracing import RequestTracer

        empty_keys = set(RequestTracer(enabled=True).get_stats())

        tracer = RequestTracer(enabled=True, sample_rate=1.0)
        tracer.request_admitted("r1", slot=0, prompt_len=100)
        tracer.prefill_done("r1", prefill_ms=5.0)
        tracer.decode_round("r1", round_idx=0, tokens_committed=3, round_ms=10.0)
        tracer.request_finished("r1", finish_reason="stop")
        assert set(tracer.get_stats()) == empty_keys
