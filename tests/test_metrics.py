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

    def __init__(self, kv_lens, prefix=(), dflash_cg_status=()):
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
