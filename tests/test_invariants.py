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

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_page_table_covers_seqlen(self, assert_level):
        # Enough pages filled to cover the declared length: fine, including
        # the exact-boundary case (pages_needed == n_filled_pages).
        checks.check_page_table_covers_seqlen(
            "swa", cache_seqlens=128, n_filled_pages=1, page_size=128
        )
        checks.check_page_table_covers_seqlen(
            "swa", cache_seqlens=129, n_filled_pages=2, page_size=128
        )
        with pytest.raises(InvariantViolation) as exc_info:
            # Declares 129 tokens (needs 2 pages of 128) but only 1 page filled.
            checks.check_page_table_covers_seqlen(
                "swa", cache_seqlens=129, n_filled_pages=1, page_size=128
            )
        message = str(exc_info.value)
        assert "cache_seqlens" in message and "129" in message
        assert "pages_needed" in message and "n_filled_pages" in message
        assert "deficit" in message

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_prefix_hit_state_le_kv(self, assert_level):
        # docs/a3-cache-coordinator-design.md §3's worked example.
        checks.check_prefix_hit_state_le_kv(slot=0, kv_hit=900, state_hit=400)
        checks.check_prefix_hit_state_le_kv(slot=0, kv_hit=64, state_hit=64)  # equal is fine
        with pytest.raises(InvariantViolation, match="INV-A3-2|prefix_hit_state_hit_le_kv_hit"):
            checks.check_prefix_hit_state_le_kv(slot=0, kv_hit=100, state_hit=101)

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_lockstep_eviction(self, assert_level):
        # Forward: KV hash dropped -> checkpoint dropped too. Fine either
        # with or without the co-keyed block still referenced (kv_ref_cnt is
        # only meaningful for the reverse direction below).
        checks.check_lockstep_eviction(
            key=1, kv_hash_dropped=True, checkpoint_dropped=True, kv_ref_cnt=0
        )
        # Reverse: checkpoint dropped by budget pressure, KV still
        # referenced -> hash correctly left alone (the asymmetry).
        checks.check_lockstep_eviction(
            key=1, kv_hash_dropped=False, checkpoint_dropped=True, kv_ref_cnt=5
        )
        with pytest.raises(InvariantViolation):
            # Forward violated: hash dropped, checkpoint survived.
            checks.check_lockstep_eviction(
                key=1, kv_hash_dropped=True, checkpoint_dropped=False, kv_ref_cnt=0
            )
        with pytest.raises(InvariantViolation):
            # Reverse violated: a still-referenced block's hash was dropped
            # anyway -- exactly the use-after-free-shaped bug INV-A3-3's
            # asymmetry exists to prevent.
            checks.check_lockstep_eviction(
                key=1, kv_hash_dropped=True, checkpoint_dropped=True, kv_ref_cnt=3
            )

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_referenced_resource_never_evicted(self, assert_level):
        checks.check_referenced_resource_never_evicted(resource_id=1, ref_cnt=1, was_evicted=False)
        checks.check_referenced_resource_never_evicted(resource_id=1, ref_cnt=0, was_evicted=True)
        with pytest.raises(InvariantViolation):
            checks.check_referenced_resource_never_evicted(
                resource_id=1, ref_cnt=1, was_evicted=True
            )

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_reserved_physical_slots_agree(self, assert_level):
        checks.check_reserved_physical_slots_agree(
            "laguna", state_pool_reserved=0, backend_reserved=0
        )
        with pytest.raises(InvariantViolation):
            # The real, documented §1.8 divergence: block_pool.py=1, laguna.py=0.
            checks.check_reserved_physical_slots_agree(
                "laguna", state_pool_reserved=1, backend_reserved=0
            )

    @pytest.mark.parametrize("assert_level", [0], indirect=True)
    def test_all_concrete_checks_are_noops_at_level_0(self, assert_level):
        checks.check_committed_ahead_of_kv_by_one(slot=0, kv_len=1, committed_len=1)
        checks.check_accepted_bound(slot=0, committed_n=999, k=15)
        checks.check_no_duplicate_ids("x", [1, 1, 1])
        checks.check_kv_len_monotonic(slot=0, prev_kv_len=10, new_kv_len=1)
        checks.check_aux_hidden_alignment(slot=0, prompt_len=1, aux_len=999, aux_offset=-998)
        checks.check_cg_replay_slot_consistency(slot=1, replay_slot=2)
        checks.check_page_table_covers_seqlen(
            "swa", cache_seqlens=999, n_filled_pages=0, page_size=1
        )
        checks.check_prefix_hit_state_le_kv(slot=0, kv_hit=1, state_hit=999)
        checks.check_lockstep_eviction(
            key=1, kv_hash_dropped=True, checkpoint_dropped=False, kv_ref_cnt=999
        )
        checks.check_referenced_resource_never_evicted(resource_id=1, ref_cnt=999, was_evicted=True)
        checks.check_reserved_physical_slots_agree("x", state_pool_reserved=1, backend_reserved=999)


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

    @staticmethod
    def _ring_blocks_for_window(window: int, block_size: int, qo_max: int = 16) -> int:
        """Verbatim copy of ``runtime/backends/laguna.py:50``'s
        ``_ring_blocks_for_window`` (``cdiv(window - 1 + qo_max, block_size)
        + 1``) -- not imported because that module hard-imports torch/vllm
        at module level; reproduced here so this test exercises the REAL
        formula without needing the full runtime import chain (this repo's
        own convention for CPU-only tests, see ``tests/test_dflash_engine.py``
        importing the owned Laguna runtime directly for the same reason)."""
        return -(-(window - 1 + qo_max) // block_size) + 1

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    @pytest.mark.parametrize("block_size", [64, 128])
    def test_page_table_covers_seqlen_at_real_production_block_sizes(
        self, assert_level, block_size
    ):
        """Mirrors ``LagunaCudaGraphVerify._fill_buffers``'s SWA branch
        (``runtime/backends/laguna_cuda_graph.py:641-655``) at the two real
        block sizes this runtime has actually shipped
        (``notes/2026-07-27-block-size-128-migration-and-tie-break-noise.md``):
        ``window=512`` (``DRAFT_WINDOW``/the main model's SWA window, both
        512), ``qo_max=16`` (``NUM_QUERY_PER_REQ``). ``aligned_len``'s worst
        case (maximum alignment slack) is ``(window - 1 + qo_max) + (bs -
        1)`` -- one full window-plus-query-batch, plus up to ``bs - 1``
        extra from floor-aligning ``window_start`` down to a block boundary.
        At both real block sizes, ``cdiv(aligned_len, bs)`` lands EXACTLY on
        ``ring_blocks_per_slot`` -- the coordinator's "生产配置下两者刚好卡
        在边界上所以不触发" observation, reproduced numerically here rather
        than just asserted in prose."""
        window, qo_max = 512, 16
        ring_blocks_per_slot = self._ring_blocks_for_window(window, block_size, qo_max)
        worst_case_aligned_len = (window - 1 + qo_max) + (block_size - 1)
        pages_needed = -(-worst_case_aligned_len // block_size)
        assert pages_needed == ring_blocks_per_slot, (
            "production sizing assumption changed -- re-derive the boundary"
        )
        # The real code's min() clip: min(pages_needed, ring_blocks_per_slot).
        n_ring = min(pages_needed, ring_blocks_per_slot)
        checks.check_page_table_covers_seqlen(
            "swa", cache_seqlens=worst_case_aligned_len, n_filled_pages=n_ring, page_size=block_size
        )

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_page_table_covers_seqlen_fires_when_alignment_outgrows_the_ring(self, assert_level):
        """One alignment-granularity step past the production boundary
        (above): the window's real span needs 7 pages of 128 but the ring
        was only ever sized (by ``_ring_blocks_for_window``) or clipped
        (by ``LagunaCudaGraphVerify._fill_buffers``'s ``min()``, line 648)
        to 6 -- the exact latent bug the coordinator found, reproduced with
        concrete numbers instead of only the symbolic formula. This is the
        scenario that produces "output looks fine but predictions quietly
        degrade": the kernel reads page_table entries beyond index 6 that
        were never (re)written for this slot/round."""
        block_size = 128
        ring_blocks_per_slot = 6  # sized for the real production boundary (see test above)
        aligned_len = 800  # needs cdiv(800, 128) == 7 pages -- one more than the ring has
        pages_needed = -(-aligned_len // block_size)
        assert pages_needed == 7 and pages_needed > ring_blocks_per_slot
        n_ring = min(pages_needed, ring_blocks_per_slot)  # the real code's clip: n_ring == 6
        with pytest.raises(InvariantViolation) as exc_info:
            checks.check_page_table_covers_seqlen(
                "swa", cache_seqlens=aligned_len, n_filled_pages=n_ring, page_size=block_size
            )
        message = str(exc_info.value)
        assert "800" in message  # declared length
        assert "7" in message  # pages needed
        assert "6" in message  # pages actually filled
        assert "1" in message  # deficit

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_page_table_covers_seqlen_full_attention_never_clips(self, assert_level):
        """Full-attention branches (``LagunaCudaGraphDecode._fill_buffers``/
        ``_fill_buffers_b1`` lines 161-174/216-224, and
        ``LagunaCudaGraphVerify._fill_buffers`` lines 660-666) have no
        ``min()`` cap anywhere: ``n_blocks = cdiv(new_kv, ps)`` is used
        directly as both the fill count and the basis for ``cache_seqlens``
        (``cache_seqlens = new_kv``), so ``cdiv(cache_seqlens, ps) ==
        n_blocks`` always, for any ``new_kv``."""
        block_size = 128
        for new_kv in [1, 127, 128, 129, 5000, 65536]:
            n_blocks = -(-new_kv // block_size)
            checks.check_page_table_covers_seqlen(
                "full", cache_seqlens=new_kv, n_filled_pages=n_blocks, page_size=block_size
            )

    @pytest.mark.parametrize("assert_level", [1], indirect=True)
    def test_page_table_covers_seqlen_swa_decode_class_never_clips_either(self, assert_level):
        """``LagunaCudaGraphDecode``'s SWA branch (``_fill_buffers``/
        ``_fill_buffers_b1``, lines 175-198/225-247) computes ``n_ring =
        cdiv(aligned_len, ps)`` with NO ``min()`` cap (unlike
        ``LagunaCudaGraphVerify`` above) -- only ``LagunaCudaGraphVerify``
        has the bug. The "only rewrite page_table if n_ring changed"
        optimization (``if n_ring != self._swa_prev_n_blocks[i]:``, lines
        188/237) doesn't matter for THIS invariant: whether or not this
        call actually rewrote entries, the CURRENT valid count is always
        this call's own ``n_ring`` (that's exactly why skipping the
        rewrite is correct in the first place -- the count didn't change).
        So passing this call's computed ``n_ring`` as ``n_filled_pages``
        (not "count of entries this call physically wrote", which could be
        0 on a cache-hit) is always correct here, and the check always
        passes for any window/kv_len combination -- there is no clip to
        trigger."""
        block_size = 64
        window = 512
        for kv_len in [0, 1, 100, 511, 512, 513, 8192, 65536]:
            new_kv = kv_len + 1
            window_start = max(0, kv_len - window + 1)
            aligned_start = (window_start // block_size) * block_size
            aligned_len = new_kv - aligned_start
            n_ring = -(-aligned_len // block_size)  # no min() cap in this class
            checks.check_page_table_covers_seqlen(
                "swa", cache_seqlens=aligned_len, n_filled_pages=n_ring, page_size=block_size
            )


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
