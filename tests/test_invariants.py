"""CPU-only tests for ``bfdiag.invariants``: the ``check()`` API and the
concrete DFlash/Laguna invariants built on it.

Every test manipulates ``registry.ASSERT_LEVEL`` directly (rather than
re-importing the module with an env var set) and restores it afterwards --
the module reads ``QSR_ASSERT_LEVEL`` once at import time, so mutating the
env var after import wouldn't take effect anyway.
"""

from __future__ import annotations

import pytest

from bfdiag.invariants import checks
from bfdiag.invariants.registry import InvariantViolation, check


@pytest.fixture
def assert_level(request):
    """Set ``registry.ASSERT_LEVEL`` for the duration of one test, restore
    the original value afterwards."""
    import bfdiag.invariants.registry as registry

    original = registry.ASSERT_LEVEL
    registry.ASSERT_LEVEL = request.param
    yield request.param
    registry.ASSERT_LEVEL = original


class TestCheckApi:
    @pytest.mark.parametrize("assert_level", [0], indirect=True)
    def test_disabled_by_default_never_raises(self, assert_level):
        # Even an obviously-false condition is a no-op at level 0.
        check(1, "anything", False, x=1)
        check(2, "anything_else", False, y=2)

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_level_1_check_raises_on_false_condition(self, assert_level):
        with pytest.raises(InvariantViolation, match="my_invariant"):
            check(1, "my_invariant", False, foo=1, bar="baz")

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_level_1_active_but_level_2_check_is_skipped(self, assert_level):
        # A level-2 check should still no-op when QSR_ASSERT_LEVEL=1.
        check(2, "expensive_invariant", False)

    @pytest.mark.parametrize("assert_level", [2], indirect=True)
    def test_level_2_enables_both_levels(self, assert_level):
        with pytest.raises(InvariantViolation):
            check(1, "cheap", False)
        with pytest.raises(InvariantViolation):
            check(2, "expensive", False)

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_true_condition_never_raises(self, assert_level):
        check(1, "always_true", True, anything="goes")

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_violation_message_carries_context(self, assert_level):
        with pytest.raises(InvariantViolation) as exc_info:
            check(1, "ctx_check", False, slot=3, kv_len=128)
        message = str(exc_info.value)
        assert "ctx_check" in message
        assert "slot" in message
        assert "128" in message


class TestConcreteChecks:
    """Each of these mirrors a real invariant wired into
    runtime/backends/laguna_dflash.py, runtime/backends/laguna.py, or
    runtime/block_pool.py -- see checks.py's docstrings for the exact code
    citation each one is grounded in."""

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_kv_len_matches_committed_tokens(self, assert_level):
        checks.check_kv_len_matches_committed(slot=0, kv_len=10, committed_len=10)
        with pytest.raises(InvariantViolation):
            checks.check_kv_len_matches_committed(slot=0, kv_len=10, committed_len=9)

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_accepted_bound(self, assert_level):
        k = 15
        checks.check_accepted_bound(slot=0, committed_n=16, k=k)  # 15 drafts + 1 bonus
        with pytest.raises(InvariantViolation):
            checks.check_accepted_bound(slot=0, committed_n=17, k=k)

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_no_duplicate_ids(self, assert_level):
        checks.check_no_duplicate_ids("block_pool.allocate", [1, 2, 3])
        with pytest.raises(InvariantViolation):
            checks.check_no_duplicate_ids("block_pool.allocate", [1, 2, 2])

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_kv_len_monotonic(self, assert_level):
        checks.check_kv_len_monotonic(slot=0, prev_kv_len=10, new_kv_len=16)
        checks.check_kv_len_monotonic(slot=0, prev_kv_len=10, new_kv_len=10)  # equal is fine
        with pytest.raises(InvariantViolation):
            checks.check_kv_len_monotonic(slot=0, prev_kv_len=10, new_kv_len=9)

    @pytest.mark.parametrize("assert_level", [2], indirect=True)
    def test_aux_hidden_alignment(self, assert_level):
        checks.check_aux_hidden_alignment(slot=0, prompt_len=100, aux_len=40, aux_offset=60)
        with pytest.raises(InvariantViolation):
            # aux_len > prompt_len -> negative offset, real bug class
            checks.check_aux_hidden_alignment(slot=0, prompt_len=100, aux_len=140, aux_offset=-40)
        with pytest.raises(InvariantViolation):
            # offset/len don't add back up to prompt_len
            checks.check_aux_hidden_alignment(slot=0, prompt_len=100, aux_len=40, aux_offset=50)

    @pytest.mark.parametrize("assert_level", [2], indirect=True)
    def test_cg_replay_slot_consistency(self, assert_level):
        checks.check_cg_replay_slot_consistency(slot=2, replay_slot=2)
        with pytest.raises(InvariantViolation):
            checks.check_cg_replay_slot_consistency(slot=2, replay_slot=3)

    @pytest.mark.parametrize("assert_level", [0], indirect=True)
    def test_all_concrete_checks_are_noops_at_level_0(self, assert_level):
        checks.check_kv_len_matches_committed(slot=0, kv_len=1, committed_len=2)
        checks.check_accepted_bound(slot=0, committed_n=999, k=15)
        checks.check_no_duplicate_ids("x", [1, 1, 1])
        checks.check_kv_len_monotonic(slot=0, prev_kv_len=10, new_kv_len=1)
        checks.check_aux_hidden_alignment(slot=0, prompt_len=1, aux_len=999, aux_offset=-998)
        checks.check_cg_replay_slot_consistency(slot=1, replay_slot=2)


class TestRecentTraceContextInMessage:
    def test_violation_embeds_recent_trace_events_when_tracing_enabled(self, monkeypatch):
        import bfdiag.invariants.registry as registry
        from bfdiag.trace import events
        from bfdiag.trace.ring import RoundRing

        ring = RoundRing(4, use_cuda=False)
        row = ring.begin_round(0, 10)
        ring.finish_round(
            row,
            events.PHASE_VERIFY,
            path=events.Path.CG_REPLAY,
            cg_miss_reason=events.CgMissReason.NONE,
            draft_tokens_n=15,
            accepted_n=15,
            reject_position=-1,
            bonus_token=1,
        )

        monkeypatch.setattr("bfdiag.trace.ring.TRACE_ENABLED", True)
        monkeypatch.setattr("bfdiag.trace.ring._ring", ring)

        original_level = registry.ASSERT_LEVEL
        registry.ASSERT_LEVEL = 1
        try:
            with pytest.raises(InvariantViolation) as exc_info:
                check(1, "traced_failure", False, slot=0)
            message = str(exc_info.value)
            assert "trace event" in message
            assert "RoundEvent" in message
        finally:
            registry.ASSERT_LEVEL = original_level

    def test_violation_has_no_trace_section_when_tracing_disabled(self):
        with pytest.raises(InvariantViolation) as exc_info:
            import bfdiag.invariants.registry as registry

            original = registry.ASSERT_LEVEL
            registry.ASSERT_LEVEL = 1
            try:
                check(1, "untraced_failure", False)
            finally:
                registry.ASSERT_LEVEL = original
        assert "trace event" not in str(exc_info.value)
