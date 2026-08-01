"""A3 step 7-c (docs/a3-cache-coordinator-design.md §7): pure-Python tests
for the new state-allocator skeleton, ``runtime.recurrent_state_pool``.

No model, no GPU, no caller yet -- these test the module's own bookkeeping
in isolation, the same way ``benchmarks/prefix_cache_eviction_check.py``'s
pure-Python checks (``lockstep_eviction``/``refcnt_never_evicted``/
``byte_budget``) test ``BlockPool``'s bookkeeping without a GPU. Methodology
ported, not the code -- see that module's docstring and
``runtime/recurrent_state_pool.py``'s.
"""

from __future__ import annotations

import pytest

from runtime.recurrent_state_pool import (
    RESERVED_PHYSICAL_SLOTS,
    RecurrentStatePool,
    physical_slot,
    spec_row,
)


class TestAddressing:
    """Fixed-slot, non-paged addressing -- ported shape of
    runtime/block_pool.py's _physical_slot/_ssm_spec_row."""

    def test_physical_slot_offsets_by_reserved(self):
        assert physical_slot(0, reserved=0) == 0
        assert physical_slot(0, reserved=1) == 1
        assert physical_slot(5, reserved=1) == 6

    def test_physical_slot_default_reserved_is_zero(self):
        # This module's default is 0 (Laguna's convention), NOT
        # block_pool.py's 1 -- see the "RESERVED_PHYSICAL_SLOTS: explicitly
        # decided, not inherited" section of the module docstring.
        assert physical_slot(3) == 3

    def test_physical_slot_rejects_negative_logical_slot(self):
        with pytest.raises(ValueError):
            physical_slot(-1)

    def test_spec_row_column_zero_matches_physical_slot(self):
        # Bootstrap correctness: the first spec-verify round right after a
        # real prefill reads column 0, which must be the SAME row the
        # ordinary chunked/non-spec path already wrote.
        for logical_slot in range(4):
            assert spec_row(logical_slot, 0, total_physical_slots=8, num_spec=3) == physical_slot(
                logical_slot
            )

    def test_spec_row_columns_are_disjoint_across_slots_and_columns(self):
        total_physical_slots = 6
        num_spec = 3
        seen: dict[int, tuple[int, int]] = {}
        for logical_slot in range(total_physical_slots):
            for col in range(num_spec + 1):
                row = spec_row(logical_slot, col, total_physical_slots, num_spec)
                if row in seen:
                    raise AssertionError(
                        f"row {row} reused by (slot={logical_slot}, col={col}) "
                        f"and {seen[row]}"
                    )
                seen[row] = (logical_slot, col)

    def test_spec_row_columns_live_past_every_slots_column_zero_row(self):
        total_physical_slots = 6
        num_spec = 3
        max_col0_row = max(physical_slot(s) for s in range(total_physical_slots))
        for logical_slot in range(total_physical_slots):
            for col in range(1, num_spec + 1):
                row = spec_row(logical_slot, col, total_physical_slots, num_spec)
                assert row > max_col0_row

    def test_spec_row_rejects_out_of_range_column(self):
        with pytest.raises(ValueError):
            spec_row(0, -1, total_physical_slots=4, num_spec=2)
        with pytest.raises(ValueError):
            spec_row(0, 3, total_physical_slots=4, num_spec=2)


class TestReservedPhysicalSlotsDivergenceIsIntentional:
    """§1.8/INV-A3-8: block_pool.py=1 vs laguna.py=0 is a real, documented
    divergence. This module picked 0 (Laguna's) explicitly -- these tests
    make that pick visible and would fail loudly if a future edit silently
    "fixed" it to match block_pool.py instead."""

    def test_module_default_matches_laguna_not_block_pool(self):
        from runtime.block_pool import RESERVED_PHYSICAL_SLOTS as BLOCK_POOL_RESERVED

        assert RESERVED_PHYSICAL_SLOTS == 0
        assert BLOCK_POOL_RESERVED == 1
        assert RESERVED_PHYSICAL_SLOTS != BLOCK_POOL_RESERVED

    def test_module_default_matches_laguna_backend_constant(self):
        pytest.importorskip("torch")
        from runtime.backends.laguna import RESERVED_PHYSICAL_SLOTS as LAGUNA_RESERVED

        assert RESERVED_PHYSICAL_SLOTS == LAGUNA_RESERVED == 0


class TestRegistrationAndLookup:
    def test_register_then_get_by_hash(self):
        pool = RecurrentStatePool(byte_budget=10**9)
        pool.register(1, hash_value=100, num_tokens=64, nbytes=50)
        assert pool.get_by_hash(100) == 1
        assert 1 in pool
        assert len(pool) == 1
        assert pool.total_bytes == 50

    def test_get_by_hash_miss_returns_none(self):
        pool = RecurrentStatePool(byte_budget=10**9)
        assert pool.get_by_hash(999) is None

    def test_register_duplicate_key_raises(self):
        pool = RecurrentStatePool(byte_budget=10**9)
        pool.register(1, hash_value=100, num_tokens=64, nbytes=50)
        with pytest.raises(RuntimeError, match="already registered"):
            pool.register(1, hash_value=200, num_tokens=128, nbytes=50)

    def test_touch_unregistered_key_raises(self):
        pool = RecurrentStatePool(byte_budget=10**9)
        with pytest.raises(RuntimeError):
            pool.touch(1)

    def test_touch_revives_lru_recency(self):
        pool = RecurrentStatePool(byte_budget=300)
        pool.register(1, hash_value=100, num_tokens=64, nbytes=100)
        pool.register(2, hash_value=200, num_tokens=64, nbytes=100)
        pool.register(3, hash_value=300, num_tokens=64, nbytes=100)
        pool.touch(1)  # 1 is now MRU; LRU order is now 2, 3, 1
        # Force one eviction: budget 300, incoming 100 -> must drop exactly
        # one to fit 400 into 300. Oldest (2) must go, not the touched one.
        pool.evict_for_budget(100)
        assert 2 not in pool
        assert 1 in pool
        assert 3 in pool


class TestLockstepEviction:
    """INV-A3-3: bidirectional, asymmetric lockstep."""

    def test_forward_direction_drops_checkpoint_with_no_kv_callbacks_wired(self):
        # Today's real state: nothing wires a KV pool to this pool at all
        # (Laguna has no recurrent layers). The KV side calling evict()
        # after evicting its own block must still cleanly drop the
        # checkpoint even with should_drop_kv_hash/drop_kv_hash both None.
        pool = RecurrentStatePool(byte_budget=10**9)
        pool.register(1, hash_value=100, num_tokens=64, nbytes=50)
        pool.evict(1)
        assert 1 not in pool
        assert pool.get_by_hash(100) is None
        assert pool.total_bytes == 0

    def test_evict_is_idempotent_on_missing_key(self):
        pool = RecurrentStatePool(byte_budget=10**9)
        pool.evict(999)  # must not raise
        pool.register(1, hash_value=100, num_tokens=64, nbytes=50)
        pool.evict(1)
        pool.evict(1)  # second call: no-op, must not raise

    def test_reverse_direction_drops_kv_hash_when_predicate_says_so(self):
        drop_calls: list[int] = []
        pool = RecurrentStatePool(
            byte_budget=100,
            should_drop_kv_hash=lambda key: True,  # e.g. co-keyed KV ref_cnt == 0
            drop_kv_hash=drop_calls.append,
        )
        pool.register(1, hash_value=100, num_tokens=64, nbytes=100)
        pool.evict_for_budget(100)  # forces eviction of key 1 to fit budget
        assert 1 not in pool
        assert drop_calls == [1]

    def test_reverse_direction_keeps_kv_hash_when_predicate_says_no(self):
        # INV-A3-3's asymmetry: losing only the checkpoint (co-keyed KV
        # block still ref_cnt > 0) is a SAFE compute miss, not a ghost hit --
        # drop_kv_hash must NOT be called.
        drop_calls: list[int] = []
        pool = RecurrentStatePool(
            byte_budget=100,
            should_drop_kv_hash=lambda key: False,  # co-keyed KV still referenced
            drop_kv_hash=drop_calls.append,
        )
        pool.register(1, hash_value=100, num_tokens=64, nbytes=100)
        pool.evict_for_budget(100)
        assert 1 not in pool  # checkpoint itself is still gone
        assert drop_calls == []  # but the KV hash was left alone


class TestPinnedNeverEvicted:
    """INV-A3-4, generalized to this pool's own resource: a pinned
    (live, in-flight) checkpoint is never evicted -- ported analog of
    benchmarks/prefix_cache_eviction_check.py's refcnt_never_evicted."""

    def test_pin_unregistered_key_raises(self):
        pool = RecurrentStatePool(byte_budget=10**9)
        with pytest.raises(RuntimeError):
            pool.pin(1)

    def test_pinned_key_survives_budget_pressure_that_would_otherwise_evict_it(self):
        pool = RecurrentStatePool(byte_budget=100)
        pool.register(1, hash_value=100, num_tokens=64, nbytes=100)  # oldest
        pool.register(2, hash_value=200, num_tokens=64, nbytes=100)  # newer
        pool.pin(1)
        # Incoming 100 bytes needs 100 freed; oldest (1) is pinned, so the
        # allocator must skip it and evict the next-oldest (2) instead.
        pool.evict_for_budget(100)
        assert 1 in pool
        assert 2 not in pool

    def test_budget_exceeded_when_everything_evictable_is_pinned(self):
        # Soft cap: going over budget is tolerated when every checkpoint is
        # pinned (unlike BlockPool.allocate's hard RuntimeError on true KV
        # exhaustion -- see class docstring in recurrent_state_pool.py).
        pool = RecurrentStatePool(byte_budget=100)
        pool.register(1, hash_value=100, num_tokens=64, nbytes=100)
        pool.pin(1)
        pool.evict_for_budget(100)  # must not raise, even though it can't help
        assert 1 in pool  # pinned survivor, unevicted
        # total_bytes(100) + incoming(100) > byte_budget(100): the budget
        # WOULD be exceeded once the incoming checkpoint actually lands --
        # evict_for_budget cannot prevent that here (nothing evictable), and
        # tolerating it (rather than raising) is the point of this test.
        assert pool.total_bytes + 100 > pool.byte_budget
        assert pool.total_bytes == 100  # nothing was evicted to try

    def test_unpin_then_evictable_again(self):
        pool = RecurrentStatePool(byte_budget=100)
        pool.register(1, hash_value=100, num_tokens=64, nbytes=100)
        pool.pin(1)
        pool.unpin(1)
        pool.register(2, hash_value=200, num_tokens=64, nbytes=100)
        pool.evict_for_budget(100)
        assert 1 not in pool  # no longer protected, and it's the oldest

    def test_evict_raises_on_pinned_key_deliberate_violation(self):
        # Deliberately constructed violation, not just a positive-path test
        # (docs/a3-cache-coordinator-design.md §7's 7-e discipline applied
        # one step early): forward-direction evict() must refuse a pinned
        # key rather than silently corrupting bookkeeping.
        pool = RecurrentStatePool(byte_budget=10**9)
        pool.register(1, hash_value=100, num_tokens=64, nbytes=50)
        pool.pin(1)
        with pytest.raises(RuntimeError, match="INV-A3-4"):
            pool.evict(1)
        assert 1 in pool  # the raise must not have partially evicted it

    def test_unpin_unknown_key_is_a_no_op(self):
        pool = RecurrentStatePool(byte_budget=10**9)
        pool.unpin(999)  # must not raise


class TestByteBudgetLRU:
    """Ported analog of benchmarks/prefix_cache_eviction_check.py's
    byte_budget check."""

    def test_evicts_lru_oldest_first_until_incoming_fits(self):
        pool = RecurrentStatePool(byte_budget=200)
        for key in (1, 2, 3):
            pool.register(key, hash_value=1000 + key, num_tokens=key * 16, nbytes=100)
        assert pool.total_bytes == 300
        pool.evict_for_budget(100)  # need room for one more 100-byte checkpoint
        assert pool.total_bytes + 100 <= pool.byte_budget
        assert list(pool._lru.keys()) == [3]  # only the newest survives
        assert pool.get_by_hash(1001) is None
        assert pool.get_by_hash(1002) is None
        assert pool.get_by_hash(1003) == 3

    def test_no_eviction_needed_when_already_within_budget(self):
        pool = RecurrentStatePool(byte_budget=1000)
        pool.register(1, hash_value=100, num_tokens=64, nbytes=50)
        pool.evict_for_budget(50)
        assert 1 in pool  # nothing needed to move
