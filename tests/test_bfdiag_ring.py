"""CPU-only tests for ``bfdiag.trace.ring``/``bfdiag.trace.timing``: the
flight recorder's hot-path core.

Every ``RoundRing``/``Timeline`` constructed here passes ``use_cuda=False``
explicitly -- never relies on real CUDA auto-detection (this sandbox has no
GPU access and must not probe for one; see the CUDA-event/perf_counter
fallback design in ``bfdiag/trace/timing.py``).
"""

from __future__ import annotations

import timeit

from bfdiag.trace import events
from bfdiag.trace.ring import TRACE_ENABLED, RoundRing
from bfdiag.trace.timing import Timeline


class TestTimelineCpuFallback:
    def test_begin_mark_resolve_roundtrip(self):
        tl = Timeline(2, 3, use_cuda=False)
        tl.begin(0)
        tl.mark(0, 1)
        tl.mark(0, 2)
        deltas = tl.resolve_deltas_ms(0, 3)
        assert len(deltas) == 2
        assert all(d >= 0.0 for d in deltas)

    def test_fewer_than_two_marks_has_no_deltas(self):
        tl = Timeline(2, 3, use_cuda=False)
        tl.begin(0)
        assert tl.resolve_deltas_ms(0, 1) == []

    def test_rows_are_independent(self):
        tl = Timeline(3, 2, use_cuda=False)
        tl.begin(0)
        tl.begin(1)
        tl.mark(0, 1)
        tl.mark(1, 1)
        # Both rows resolve independently without cross-contamination.
        d0 = tl.resolve_deltas_ms(0, 2)
        d1 = tl.resolve_deltas_ms(1, 2)
        assert len(d0) == 1
        assert len(d1) == 1

    def test_rejects_nonpositive_capacity(self):
        import pytest

        with pytest.raises(ValueError):
            Timeline(0, 3, use_cuda=False)
        with pytest.raises(ValueError):
            Timeline(3, 0, use_cuda=False)


class TestRoundRingBasics:
    def test_begin_mark_finish_snapshot(self):
        ring = RoundRing(4, use_cuda=False)
        row = ring.begin_round(slot=2, kv_len_before=50)
        ring.mark(row, events.PHASE_VERIFY)
        ring.mark(row, events.PHASE_COMMIT)
        ring.finish_round(
            row,
            events.PHASE_DRAFT,
            path=events.Path.CG_REPLAY,
            cg_miss_reason=events.CgMissReason.NONE,
            draft_tokens_n=15,
            accepted_n=12,
            reject_position=12,
            bonus_token=99,
        )
        snap = ring.snapshot()
        assert len(snap) == 1
        ev = snap[0]
        assert ev.slot == 2
        assert ev.kv_len_before == 50
        assert ev.path == "cg_replay"
        assert ev.cg_miss_reason == "none"
        assert ev.draft_tokens_n == 15
        assert ev.accepted_n == 12
        assert ev.reject_position == 12
        assert ev.bonus_token == 99
        assert ev.t_verify > 0.0 or ev.t_verify == 0.0  # timing is monotone, may be ~0
        assert ev.t_round >= 0.0

    def test_unwritten_rows_are_excluded_from_snapshot(self):
        ring = RoundRing(4, use_cuda=False)
        ring.begin_round(0, 0)  # never finished
        assert ring.snapshot() == []

    def test_wraparound_overwrites_oldest_row(self):
        capacity = 3
        ring = RoundRing(capacity, use_cuda=False)
        for i in range(capacity + 2):  # write 2 more rounds than capacity
            row = ring.begin_round(slot=0, kv_len_before=i)
            ring.finish_round(
                row,
                events.PHASE_VERIFY,
                path=events.Path.CG_REPLAY,
                cg_miss_reason=events.CgMissReason.NONE,
                draft_tokens_n=15,
                accepted_n=15,
                reject_position=-1,
                bonus_token=i,
            )
        snap = ring.snapshot()
        # Only `capacity` rows survive; they are the most recent ones, in order.
        assert len(snap) == capacity
        kv_lens = [ev.kv_len_before for ev in snap]
        assert kv_lens == [2, 3, 4]
        round_idxs = [ev.round_idx for ev in snap]
        assert round_idxs == sorted(round_idxs)

    def test_finish_dflash_round_translates_decision_dict(self):
        from bfdiag.trace import ring as ring_module

        ring_module.reset(2, use_cuda=False)

        row = ring_module.begin_round(0, 10)
        ring_module.mark(row, events.PHASE_VERIFY)
        decision = {"num_accepted": 5, "rejected_at": 5}
        ring_module.finish_dflash_round(row, True, True, 15, decision, bonus_token=7)
        snap = ring_module.get_ring().snapshot()
        assert snap[0].path == "cg_replay"
        assert snap[0].accepted_n == 5
        assert snap[0].reject_position == 5
        assert snap[0].bonus_token == 7

        row2 = ring_module.begin_round(0, 20)
        decision_full = {"num_accepted": 15, "rejected_at": None}
        ring_module.finish_dflash_round(row2, False, True, 15, decision_full, bonus_token=8)
        snap2 = ring_module.get_ring().snapshot()
        full = [e for e in snap2 if e.kv_len_before == 20][0]
        assert full.path == "eager"
        assert full.cg_miss_reason == "cg_unavailable"
        assert full.reject_position == -1

    def test_record_decode_batch_path_cg_miss_reasons(self):
        from bfdiag.trace import ring as ring_module
        from bfdiag.trace.ring import record_decode_batch_path

        ring_module.reset(8, use_cuda=False)

        class _Params:
            def __init__(self, is_greedy):
                self.is_greedy = is_greedy

        # cg not captured at all -> eager
        record_decode_batch_path([0], [10], None, False, False, [_Params(True)])
        snap = ring_module.get_ring().snapshot()
        assert snap[-1].path == "eager"
        assert snap[-1].cg_miss_reason == "cg_unavailable"

        # cg exists but batch size didn't match (dynamic per-call miss)
        record_decode_batch_path([0, 1], [10, 20], object(), False, False, [_Params(True)] * 2)
        snap = ring_module.get_ring().snapshot()
        recent = snap[-2:]
        assert all(e.path == "cg_miss" for e in recent)
        assert all(e.cg_miss_reason == "batch_size_mismatch" for e in recent)

        # cg exists, eligible -> cg_replay
        record_decode_batch_path([0], [10], object(), True, False, [_Params(True)])
        snap = ring_module.get_ring().snapshot()
        assert snap[-1].path == "cg_replay"
        assert snap[-1].cg_miss_reason == "none"


class TestDisabledPathOverhead:
    """Proves the QSR_TRACE=0 (default) hot path costs (close to) nothing.

    This mirrors the *exact* guard shape used at the real integration call
    sites in runtime/backends/laguna_dflash.py and
    runtime/backends/laguna.py: a single ``if bfdiag_trace.TRACE_ENABLED:``
    check per hook, never a bare function call into bfdiag first.
    """

    def test_module_default_is_disabled_without_env_var(self):
        # This test only means anything if the ambient test environment
        # didn't opt into tracing -- assert the precondition explicitly
        # rather than silently asserting nothing.
        assert TRACE_ENABLED is False, (
            "QSR_TRACE was set to 1 in this test environment; the disabled-"
            "path microbenchmark below requires the default (off) state"
        )

    def test_disabled_round_overhead_under_100ns(self):
        from bfdiag.trace import ring as bfdiag_trace

        def disabled_round_hooks() -> None:
            slot, kv_len = 0, 100
            row = bfdiag_trace.begin_round(slot, kv_len) if bfdiag_trace.TRACE_ENABLED else -1
            if bfdiag_trace.TRACE_ENABLED:
                bfdiag_trace.mark(row, events.PHASE_VERIFY)
            if bfdiag_trace.TRACE_ENABLED:
                bfdiag_trace.mark(row, events.PHASE_COMMIT)
            if bfdiag_trace.TRACE_ENABLED:
                bfdiag_trace.finish_round(
                    row,
                    events.PHASE_DRAFT,
                    path=0,
                    cg_miss_reason=0,
                    draft_tokens_n=15,
                    accepted_n=15,
                    reject_position=-1,
                    bonus_token=1,
                )

        number = 200_000
        # min-of-repeats is the standard way to squeeze measurement noise
        # out of a timeit microbenchmark (each repeat's mean is inflated by
        # any scheduler hiccup; the true cost is the best case observed).
        best = min(timeit.repeat(disabled_round_hooks, repeat=5, number=number))
        per_call_ns = (best / number) * 1e9
        assert per_call_ns < 100.0, (
            f"disabled-path overhead {per_call_ns:.1f}ns/round exceeds the 100ns budget"
        )
