"""D2: tests for server/metrics.py — Prometheus-style metrics."""

import pytest

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


class TestMetricsUnderLoad:
    """/metrics must not 500 while the server is busy.

    `LagunaBackend.slot_kv_len` is a list indexed by slot; the endpoint read
    it as a mapping. The expression only evaluates for slots in
    `engine.active`, so an idle server scraped fine and a busy one returned
    500 -- observed live 2026-08-01 during a 68 s request, with successful
    scrapes either side of it hiding the failure.
    """

    def test_snapshot_is_read_through_the_contract(self):
        pytest.importorskip("fastapi")
        from runtime.backends.protocol import BackendSnapshot, SlotSnapshot
        from server.app import _backend_snapshot

        class _Runner:
            def snapshot(self):
                return BackendSnapshot(
                    slots=(SlotSnapshot(slot=1, kv_len=4096, is_fresh=False),),
                    prefix=(),
                )

        snapshot = _backend_snapshot(_Runner())
        assert snapshot is not None
        assert {s.slot: s.kv_len for s in snapshot.slots}[1] == 4096

    def test_backend_without_snapshot_degrades_instead_of_raising(self):
        pytest.importorskip("fastapi")
        from server.app import _backend_snapshot

        class _NoContract:
            pass

        assert _backend_snapshot(_NoContract()) is None

    def test_a_raising_backend_cannot_take_metrics_down(self):
        # The older constraint, restated at the new seam: metrics must never
        # be the thing that fails. A backend whose snapshot explodes should
        # cost accuracy, not the monitoring signal.
        pytest.importorskip("fastapi")
        from server.app import _backend_snapshot

        class _Exploding:
            def snapshot(self):
                raise RuntimeError("backend is having a bad day")

        assert _backend_snapshot(_Exploding()) is None


class _FakeSnapshotRunner:
    """Backend stand-in shaped like the contract, not like LagunaBackend.

    Deliberately shares no internals with the real class: if the endpoints
    ever reach past `snapshot()` again, this fake stops satisfying them and
    the tests below go red.
    """

    def __init__(
        self,
        kv_lens,
        prefix=(),
        dflash_cg_status=(),
        runtime_stats=(),
        cg_fallback_reasons=(),
    ):
        from runtime.backends.protocol import BackendSnapshot, PrefixSnapshot, SlotSnapshot

        self._snapshot = BackendSnapshot(
            slots=tuple(
                SlotSnapshot(slot=i, kv_len=n, is_fresh=n == 0) for i, n in enumerate(kv_lens)
            ),
            prefix=tuple(
                PrefixSnapshot(slot=i, cached_kv_len=kv, cached_tokens=len(t), head=tuple(t[:5]))
                for i, (kv, t) in enumerate(prefix)
            ),
            dflash_cg_status=dflash_cg_status,
            runtime_stats=runtime_stats,
            cg_fallback_reasons=cg_fallback_reasons,
        )

    def snapshot(self):
        return self._snapshot


class _FakeEngine:
    MODEL = "test/model"

    def __init__(self, runner, active=None, block_size=16, num_slots=4, blocks_per_slot=8):
        self.runner = runner
        self.active = active if active is not None else {}
        self.waiting = []
        self.free_slots = list(range(num_slots))
        self.block_size = block_size
        self.num_slots = num_slots
        self.blocks_per_slot = blocks_per_slot
        self.capacity_tokens_per_slot = block_size * blocks_per_slot
        self.stats = {}


class TestMetricsEndpointItself:
    """Route-level coverage, which /metrics had none of.

    Both 500s shipped because the tests exercised `render_prometheus` and a
    helper in isolation while nothing ever called the endpoint. These drive
    the two states that actually broke: freshly started, and busy.
    """

    @staticmethod
    def _call(monkeypatch, engine) -> str:
        """Scrape /metrics and return the rendered text.

        The endpoint returns a PlainTextResponse, so the body is read out
        here -- asserting against the object's repr would pass no matter what
        the endpoint rendered.
        """
        import asyncio

        pytest.importorskip("fastapi")
        import server.app as app_module

        monkeypatch.setattr(app_module, "engine", engine)
        response = asyncio.run(app_module.metrics_endpoint())
        return response.body.decode()

    def test_cold_start_scrape_succeeds(self, monkeypatch):
        # Nothing has completed yet -- the window a scraper arrives in first,
        # and the window `get_stats()` used to short-circuit inside.
        engine = _FakeEngine(_FakeSnapshotRunner([0, 0, 0, 0]))
        body = self._call(monkeypatch, engine)
        assert "blackwellm:kv_cache_used_blocks" in body

    def test_busy_scrape_succeeds_and_counts_blocks(self, monkeypatch):
        # `engine.active` non-empty is the exact condition that turned the
        # slot-length read into a 500.
        engine = _FakeEngine(
            _FakeSnapshotRunner([0, 4096, 32, 0]),
            active={1: {}, 2: {}},
            block_size=16,
        )
        body = self._call(monkeypatch, engine)
        # 4096/16 = 256 blocks, 32/16 = 2 blocks.
        assert 'blackwellm:kv_cache_used_blocks{model_name="test/model"} 258' in body

    def test_scrape_counts_chunked_prefill_as_running(self, monkeypatch):
        # A long prompt owns its slot before the first anchor promotes it to
        # ``engine.active``.  Metrics must not report an idle server or zero
        # KV usage during that admission-to-activation window.
        engine = _FakeEngine(_FakeSnapshotRunner([2048, 0]), block_size=16)
        engine.free_slots = [1]
        engine._pending_prefill = object()
        engine._pending_prefill_reqs = [(0, object())]
        body = self._call(monkeypatch, engine)
        assert 'blackwellm:num_requests_running{model_name="test/model"} 1' in body
        assert 'blackwellm:num_requests_prefill{model_name="test/model"} 1' in body
        assert 'blackwellm:num_occupied_slots{model_name="test/model"} 3' in body
        assert 'blackwellm:kv_cache_used_blocks{model_name="test/model"} 128' in body

    def test_health_counts_chunked_prefill(self, monkeypatch):
        import asyncio

        pytest.importorskip("fastapi")
        import server.app as app_module

        engine = _FakeEngine(_FakeSnapshotRunner([2048, 0]), num_slots=2)
        engine.free_slots = [1]
        engine._pending_prefill = object()
        engine._pending_prefill_reqs = [(0, object())]
        monkeypatch.setattr(app_module, "engine", engine)

        result = asyncio.run(app_module.health())
        assert result["active"] == 1
        assert result["active_generating"] == 0
        assert result["prefill"] == 1
        assert result["occupied_slots"] == 1
        assert result["waiting"] == 0

    def test_health_counts_synchronous_prefill_begin(self, monkeypatch):
        import asyncio

        pytest.importorskip("fastapi")
        import server.app as app_module

        engine = _FakeEngine(_FakeSnapshotRunner([0]), num_slots=2)
        engine.free_slots = [1]
        engine._admission_inflight_reqs = [(0, object())]
        monkeypatch.setattr(app_module, "engine", engine)

        result = asyncio.run(app_module.health())
        assert result["active"] == 1
        assert result["active_generating"] == 0
        assert result["prefill"] == 1
        assert result["occupied_slots"] == 1

    def test_scrape_survives_a_backend_outside_the_contract(self, monkeypatch):
        class _NoContract:
            pass

        engine = _FakeEngine(_NoContract(), active={0: {}})
        body = self._call(monkeypatch, engine)
        # Falls back to the slot-count approximation rather than failing.
        assert "blackwellm:kv_cache_used_blocks" in body
        # Third failure mode this metric must not add (see
        # notes/2026-08-01-c1-c2-gpu-investigation.md): a backend outside the
        # snapshot contract at all must still get a well-formed, empty
        # dflash_cg_captured block, not a crash and not a missing HELP/TYPE
        # header.
        assert "# TYPE blackwellm:dflash_cg_captured gauge" in body
        assert "dflash_cg_captured{" not in body

    def test_scrape_reports_no_dflash_cg_series_when_dflash_disabled(self, monkeypatch):
        # The common case: DFlash never enabled, or enabled but no capture
        # attempted yet (self._dflash is None / cg_status == {} in
        # LagunaBackend.snapshot()). Must still be a valid, non-raising
        # Prometheus block with zero series -- not absent, not an exception.
        engine = _FakeEngine(_FakeSnapshotRunner([0, 0, 0, 0]))
        body = self._call(monkeypatch, engine)
        assert "# HELP blackwellm:dflash_cg_captured" in body
        assert "# TYPE blackwellm:dflash_cg_captured gauge" in body
        assert "dflash_cg_captured{" not in body

    def test_scrape_reports_dflash_cg_status_per_graph(self, monkeypatch):
        # This is the actual point of C-1's second fix: a CUDA Graph capture
        # failure must be visible to Prometheus, not just to someone grepping
        # startup logs. "captured" -> 1, "failed" -> 0, one series per graph
        # DFlash actually attempted.
        engine = _FakeEngine(
            _FakeSnapshotRunner(
                [0, 0],
                dflash_cg_status=(("draft", "captured"), ("verify", "failed")),
            )
        )
        body = self._call(monkeypatch, engine)
        assert 'blackwellm:dflash_cg_captured{model_name="test/model",graph="draft"} 1' in body
        assert 'blackwellm:dflash_cg_captured{model_name="test/model",graph="verify"} 0' in body

    def test_scrape_reports_flashnext_gdn_projection_mode_as_captured(self, monkeypatch):
        # Flash-Next stores the GDN projection execution contract (a mode
        # string) in the same status map, never the literal "captured".
        # A "batched*" mode is the fast path and must report 1; "per_row" /
        # "disabled" are the eager fallbacks and must report 0.
        engine = _FakeEngine(
            _FakeSnapshotRunner(
                [0, 0],
                dflash_cg_status=(
                    ("gdn_projections", "batched_bf16"),
                    ("decode", "captured"),
                ),
            )
        )
        body = self._call(monkeypatch, engine)
        assert (
            'blackwellm:dflash_cg_captured{model_name="test/model",'
            'graph="gdn_projections"} 1'
        ) in body
        assert 'blackwellm:dflash_cg_captured{model_name="test/model",graph="decode"} 1' in body

        engine = _FakeEngine(
            _FakeSnapshotRunner(
                [0, 0],
                dflash_cg_status=(("gdn_projections", "per_row"),),
            )
        )
        body = self._call(monkeypatch, engine)
        assert (
            'blackwellm:dflash_cg_captured{model_name="test/model",'
            'graph="gdn_projections"} 0'
        ) in body

    def test_scrape_reports_backend_cuda_graph_activity_and_fallback_reason(self, monkeypatch):
        engine = _FakeEngine(
            _FakeSnapshotRunner(
                [12],
                runtime_stats=(
                    ("decode_graph_capture_attempts", 1),
                    ("decode_graph_capture_successes", 0),
                    ("decode_graph_capture_failures", 1),
                    ("decode_graph_replays", 7),
                    ("decode_eager_fallbacks", 2),
                ),
                cg_fallback_reasons=(("capture_failed", 2),),
            )
        )
        body = self._call(monkeypatch, engine)
        assert (
            'blackwellm:decode_graph_replays_total{model_name="test/model"} 7' in body
        )
        assert (
            'blackwellm:decode_eager_fallbacks_total{model_name="test/model"} 2' in body
        )
        assert (
            'blackwellm:decode_eager_fallback_reason_total{model_name="test/model",'
            'reason="capture_failed"} 2'
        ) in body

    def test_debug_stats_reports_snapshot_activity_without_private_reads(self, monkeypatch):
        import asyncio

        pytest.importorskip("fastapi")
        import server.app as app_module

        runner = _FakeSnapshotRunner(
            [4],
            dflash_cg_status=(("decode", "captured"),),
            runtime_stats=(("decode_graph_replays", 3),),
            cg_fallback_reasons=(("not_captured", 1),),
        )
        engine = _FakeEngine(runner)
        monkeypatch.setattr(app_module, "engine", engine)

        result = asyncio.run(app_module.debug_stats())

        assert result["_cuda_graph_dbg"] == {"decode": "captured"}
        assert result["_backend_snapshot_stats_dbg"] == {"decode_graph_replays": 3}
        assert result["_cuda_graph_fallback_reasons_dbg"] == {"not_captured": 1}

    def test_debug_stats_does_not_block_the_event_loop(self, monkeypatch):
        # Regression for the 2026-09-01 freeze: on the live Flash-Next server
        # one /debug/stats scrape ran 37 s of blocking object-graph walking
        # ON THE EVENT LOOP, and /health plus every in-flight streaming chat
        # response were unresponsive for the whole duration. The collector
        # must run off-loop: a heartbeat task keeps ticking while a slow
        # memory_breakdown is in flight.
        import asyncio
        import time

        pytest.importorskip("fastapi")
        import server.app as app_module

        class _SlowMemoryRunner(_FakeSnapshotRunner):
            def memory_breakdown(self):
                time.sleep(0.5)
                return {"model_tensor_bytes": 1}

        engine = _FakeEngine(_SlowMemoryRunner([0, 0]))
        monkeypatch.setattr(app_module, "engine", engine)

        async def scenario():
            ticks: list[int] = [0]

            async def heartbeat():
                while True:
                    ticks[0] += 1
                    await asyncio.sleep(0.02)

            beat = asyncio.create_task(heartbeat())
            result = await app_module.debug_stats()
            beat.cancel()
            return result, ticks[0]

        result, ticks = asyncio.run(scenario())
        assert result["_memory_breakdown_dbg"] == {"model_tensor_bytes": 1}
        assert ticks >= 5, f"event loop starved during debug_stats (ticks={ticks})"

    def test_debug_stats_uses_cached_memory_while_busy(self, monkeypatch):
        import asyncio

        pytest.importorskip("fastapi")
        import server.app as app_module

        class _BusyMemoryRunner(_FakeSnapshotRunner):
            calls = 0

            def memory_breakdown(self):
                self.calls += 1
                raise AssertionError("busy debug scrape must not walk the object graph")

        runner = _BusyMemoryRunner([2048])
        engine = _FakeEngine(runner, active={0: {}})
        engine.stats["_memory_breakdown_dbg"] = {"model_tensor_bytes": 7}
        monkeypatch.setattr(app_module, "engine", engine)

        result = asyncio.run(app_module.debug_stats())
        assert result["_memory_breakdown_dbg"] == {"model_tensor_bytes": 7}
        assert result["_memory_breakdown_source"] == "startup_cache"
        assert result["_memory_breakdown_stale"] is True
        assert runner.calls == 0

    def test_endpoint_does_not_reach_past_the_contract(self, monkeypatch):
        # The nail this whole step is driven around. This backend carries a
        # real-shaped `slot_kv_len` list *and* a snapshot that disagrees with
        # it. Reading the attribute -- the habit that produced both 500s --
        # yields 1 block; going through the contract yields 256. Only one of
        # those numbers can appear, and it must be the contract's.
        class _TwoFaced(_FakeSnapshotRunner):
            def __init__(self):
                super().__init__([0, 4096])
                self.slot_kv_len = [0, 16]

        engine = _FakeEngine(_TwoFaced(), active={1: {}}, block_size=16)
        body = self._call(monkeypatch, engine)
        assert 'blackwellm:kv_cache_used_blocks{model_name="test/model"} 256' in body
