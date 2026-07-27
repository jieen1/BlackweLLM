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
    def test_committed_ahead_of_kv_by_one(self, assert_level):
        # The real relationship is +1, NOT equality -- see
        # TestRealCodeRegression below for why equality was wrong.
        checks.check_committed_ahead_of_kv_by_one(slot=0, kv_len=10, committed_len=11)
        with pytest.raises(InvariantViolation):
            checks.check_committed_ahead_of_kv_by_one(slot=0, kv_len=10, committed_len=10)
        with pytest.raises(InvariantViolation):
            checks.check_committed_ahead_of_kv_by_one(slot=0, kv_len=10, committed_len=9)
        with pytest.raises(InvariantViolation):
            checks.check_committed_ahead_of_kv_by_one(slot=0, kv_len=10, committed_len=12)

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
            # aux_len > prompt_len -> negative offset, the real bug class
            checks.check_aux_hidden_alignment(slot=0, prompt_len=100, aux_len=140, aux_offset=-40)

    @pytest.mark.parametrize("assert_level", [2], indirect=True)
    def test_cg_replay_slot_consistency(self, assert_level):
        checks.check_cg_replay_slot_consistency(slot=2, replay_slot=2)
        with pytest.raises(InvariantViolation):
            checks.check_cg_replay_slot_consistency(slot=2, replay_slot=3)

    @pytest.mark.parametrize("assert_level", [0], indirect=True)
    def test_all_concrete_checks_are_noops_at_level_0(self, assert_level):
        checks.check_committed_ahead_of_kv_by_one(slot=0, kv_len=1, committed_len=1)
        checks.check_accepted_bound(slot=0, committed_n=999, k=15)
        checks.check_no_duplicate_ids("x", [1, 1, 1])
        checks.check_kv_len_monotonic(slot=0, prev_kv_len=10, new_kv_len=1)
        checks.check_aux_hidden_alignment(slot=0, prompt_len=1, aux_len=999, aux_offset=-998)
        checks.check_cg_replay_slot_consistency(slot=1, replay_slot=2)


class TestRealCodeRegression:
    """Every invariant here was re-derived from scratch against the actual
    formulas in runtime/backends/laguna.py and runtime/backends/
    laguna_dflash.py after a coordinator review caught
    ``check_committed_ahead_of_kv_by_one`` shipped as a wrong equality check
    (would have fired on every DFlash round in production). Each test below
    plugs the REAL code's formula through the check and, where the old
    version was wrong, shows it disagreeing with reality -- so a future edit
    that reintroduces the same mistake fails a test instead of shipping."""

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_prefill_produces_committed_len_kv_len_plus_one_not_equal(self, assert_level):
        """Mirrors runtime/backends/laguna.py's prefill (and its 4 sibling
        variants, all identical in shape): after sampling ``first_token``,
        ``slot_kv_len[slot] = len(prompt_ids)`` while
        ``slot_committed_tokens[slot] = list(prompt_ids) + [first_token]``."""
        prompt_ids = list(range(100))  # len(prompt_ids) == 100
        first_token = 12345

        kv_len = len(prompt_ids)
        committed_tokens = [*prompt_ids, first_token]
        committed_len = len(committed_tokens)

        assert committed_len == kv_len + 1  # the real relationship
        old_buggy_condition = kv_len == committed_len
        assert old_buggy_condition is False, (
            "the original (wrong) check would have asserted this False "
            "condition True and raised on every real post-prefill state"
        )

        # The fixed check accepts the real state; a hypothetical revert to
        # equality would reject it (proven above), which is exactly the
        # false-positive-on-every-round bug the coordinator caught.
        checks.check_committed_ahead_of_kv_by_one(0, kv_len, committed_len)

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_dflash_round_preserves_the_plus_one_gap_for_every_accept_count(self, assert_level):
        """Mirrors dflash_round: ``slot_kv_len[slot] += context_count`` where
        ``context_count = 1 + num_accepted`` (``_verify_only_accept_reject``,
        ``laguna_dflash.py``), immediately followed by appending
        ``len(committed) == context_count`` tokens -- same increment on both
        sides, every round, for every possible ``num_accepted`` in 0..K."""
        kv_len, committed_len = 100, 101  # post-prefill starting state
        k = 15
        for num_accepted in range(k + 1):
            context_count = 1 + num_accepted
            kv_len += context_count
            committed_len += context_count  # len(committed) == context_count
            # Should never raise: the +1 gap is preserved by construction.
            checks.check_committed_ahead_of_kv_by_one(0, kv_len, committed_len)
        assert committed_len == kv_len + 1

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_decode_batch_sampled_preserves_the_plus_one_gap(self, assert_level):
        """Mirrors decode_batch_sampled's non-DFlash path: both
        ``slot_kv_len[slot] += 1`` and one token appended to
        ``slot_committed_tokens[slot]`` -- same +1 increment on both sides."""
        kv_len, committed_len = 50, 51
        for _ in range(10):
            kv_len += 1
            committed_len += 1
            checks.check_committed_ahead_of_kv_by_one(0, kv_len, committed_len)

    def test_aux_offset_plus_aux_len_equals_prompt_len_is_tautological(self):
        """The removed clause (``aux_offset + aux_len == prompt_len``) was a
        re-derivation of ``aux_offset``'s own defining formula
        (``aux_offset = prompt_len - aux_len``, laguna_dflash.py) computed at
        the call site -- plugging that definition back in is true for every
        integer pair, valid or not, so it never added discriminating power.
        Demonstrated directly: it holds even for the invalid case the real
        check DOES catch (aux_len > prompt_len)."""
        for prompt_len, aux_len in [(100, 40), (100, 100), (100, 0), (100, 140), (1, 999)]:
            aux_offset = prompt_len - aux_len
            assert aux_offset + aux_len == prompt_len  # true unconditionally, valid or not
        # The ONE clause that actually distinguishes valid from invalid:
        assert (100 - 40) >= 0  # valid: aux_len <= prompt_len
        assert (100 - 140) < 0  # invalid: aux_len > prompt_len -- this is the real bug class

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_no_duplicate_ids_does_not_check_across_separate_calls(self, assert_level):
        """Documents the within-call-only scope as a test, not just prose:
        two separate ``allocate``-style calls returning overlapping ids each
        individually pass (this check cannot and does not claim to catch
        cross-call aliasing -- that's ``BlockPool``'s own hard ``ref_cnt``
        RuntimeErrors, not this invariant)."""
        checks.check_no_duplicate_ids("call_1", [5, 6, 7])
        checks.check_no_duplicate_ids("call_2", [7, 8, 9])  # 7 repeats across calls: not caught
        with pytest.raises(InvariantViolation):
            checks.check_no_duplicate_ids("call_3", [7, 7])  # repeats WITHIN one call: caught

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_accepted_bound_matches_every_real_committed_length(self, assert_level):
        """``len(committed) == num_accepted + 1`` always (see
        ``determine_accept_reject_from_predictions``: both the early-return-
        on-reject branch and the full-accept branch append exactly one
        token per loop iteration up to and including position
        ``num_accepted``) -- so the real ``committed_n`` never even
        approaches, let alone exceeds, ``k + 1``."""
        k = 15
        for num_accepted in range(k + 1):
            committed_n = num_accepted + 1
            checks.check_accepted_bound(0, committed_n, k)


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
