"""Phase 1 tests: pure-metadata page-bundle allocator.

``.omx/plans/qwen38-dynamic-context-vllm-plan.md`` Phase 1 -- "实现独立
page-bundle allocator": 先把 allocator 作为纯元数据组件做正确,再接 GPU
tensor. Torch-free by construction (the module has no torch import), so the
torch-free CI job covers it fully.

Covers the plan §7 core invariants at every mutating call (assert_invariants
on by default), the §6.4 lifecycle state machine, COW detach, prefix-cache
hit lookup, cached eviction, true exhaustion, double-free, and stale-owner
bookkeeping. vLLM semantics are referenced in the module docstring; these
tests lock the behavior that Phase 2's tensor wiring will build on.
"""

from __future__ import annotations

import pytest

from runtime.model.qwen36_kv_arena import (
    BlockKey,
    QwenKVUsage,
    QwenPageBundlePool,
)


def _pool(num_bundles: int = 16, **kwargs) -> QwenPageBundlePool:
    return QwenPageBundlePool(num_bundles=num_bundles, **kwargs)


def _key(value: int, tokens: int = 128) -> BlockKey:
    return BlockKey(value=value, num_tokens=tokens)


class TestConstruction:
    def test_reserves_null_bundle_zero(self) -> None:
        pool = _pool(8)
        assert pool.reserved == 1
        assert pool.null_bundle.bundle_id == 0
        assert pool.null_bundle.is_null
        assert pool.num_free_bundles() == 7

    def test_refuses_zero_or_too_small_pool(self) -> None:
        with pytest.raises(ValueError):
            _pool(0)
        with pytest.raises(ValueError):
            _pool(1)  # num_bundles == reserved

    def test_null_bundle_never_allocatable_or_cachable(self) -> None:
        pool = _pool(8)
        with pytest.raises(RuntimeError, match="pool exhausted"):
            # Only bundle 0 reserved; 1..7 usable. Force exhaustion by
            # allocating 7, then try 8 more.
            pool.allocate(7, owner="a")
            pool.allocate(8, owner="b")
        with pytest.raises(RuntimeError, match="null bundle can never be cached"):
            pool.publish_full_block(0, _key(99))

    def test_usage_snapshot_balances(self) -> None:
        pool = _pool(10, watermark_bundles=2)
        u = pool.usage()
        assert isinstance(u, QwenKVUsage)
        assert u.total_bundles == 10
        assert u.reserved_bundles == 1
        assert u.free_bundles == 9
        assert u.watermark_bundles == 2
        assert u.live_bundles == 0
        assert u.cached_bundles == 0


class TestLifecycleStateMachine:
    """Plan §6.4: FREE_UNHASHED -> LIVE_PRIVATE -> LIVE_SHARED -> CACHED_REF0."""

    def test_allocate_hands_out_unique_live_bundles(self) -> None:
        pool = _pool(8)
        ids = pool.allocate(3, owner="a")
        assert len(ids) == len(set(ids)) == 3
        assert all(i >= pool.reserved for i in ids)
        assert pool.num_free_bundles() == 4
        assert pool._ref_live_unique() == 3

    def test_full_free_cached_live_conservation(self) -> None:
        # INV1: free_unhashed + cached_ref0 + live_unique == total_usable at
        # every step.
        pool = _pool(10)
        usable = pool.num_bundles - pool.reserved
        ids = pool.allocate(4, owner="a")
        keys = [_key(10 + i, 128 * (i + 1)) for i in range(4)]
        pool.cache_full_blocks(keys, ids)
        pool.decref(ids[:2], owner="a")
        live = pool._ref_live_unique()
        cached = pool._ref_cached_ref0()
        assert (pool._free_len - cached) + cached + live == usable
        pool.allocate(2, owner="b")
        live = pool._ref_live_unique()
        cached = pool._ref_cached_ref0()
        assert (pool._free_len - cached) + cached + live == usable

    def test_cached_bundle_returns_to_lru_tail(self) -> None:
        pool = _pool(6)
        ids = pool.allocate(3, owner="a")
        pool.cache_full_blocks([_key(1), _key(2), _key(3)], ids)
        pool.decref(ids, owner="a")  # all ref_cnt=0, hashed
        # CACHED_REF0: parked, still hit-able.
        assert pool._ref_cached_ref0() == 3
        hit = pool.lookup_longest_prefix([_key(1), _key(2), _key(3)])
        assert hit.hit and hit.num_blocks == 3

    def test_unhashed_bundle_returns_to_lifo_head(self) -> None:
        pool = _pool(6)
        ids = pool.allocate(2, owner="a")
        pool.decref(ids, owner="a")  # never cached -> FREE_UNHASHED head
        assert pool._ref_cached_ref0() == 0
        # LIFO: next allocate pops the most-recently-freed first.
        again = pool.allocate(1, owner="b")
        assert again[0] == ids[1]

    def test_incref_revives_cached_bundle_from_free_queue(self) -> None:
        pool = _pool(6)
        ids = pool.allocate(1, owner="a")
        pool.cache_full_blocks([_key(7)], ids)
        pool.decref(ids, owner="a")
        assert pool._ref_cached_ref0() == 1
        # vLLM touch: reviving a cached bundle raises ref_cnt 0 -> 1 and
        # removes it from the free queue in O(1).
        pool.incref(ids, owner="b")
        assert pool.bundles[ids[0]].ref_cnt == 1
        assert not pool._free_contains(pool.bundles[ids[0]])

    def test_double_free_is_refused(self) -> None:
        pool = _pool(6)
        ids = pool.allocate(1, owner="a")
        pool.decref(ids, owner="a")
        with pytest.raises(RuntimeError, match="double-free"):
            pool.decref(ids, owner="a")

    def test_incref_of_freed_unhashed_bundle_is_refused(self) -> None:
        pool = _pool(6)
        ids = pool.allocate(1, owner="a")
        pool.decref(ids, owner="a")  # FREE_UNHASHED, gone
        with pytest.raises(RuntimeError):
            pool.incref(ids, owner="b")

    def test_null_bundle_never_referenced(self) -> None:
        pool = _pool(6)
        with pytest.raises(RuntimeError, match="INV7"):
            pool.incref([0], owner="a")

    def test_all_mutations_preserve_invariants(self) -> None:
        pool = _pool(12, assert_invariants=True)
        ids = pool.allocate(5, owner="a")
        pool.cache_full_blocks([_key(i) for i in range(5)], ids)
        pool.decref(ids[:3], owner="a")
        pool.incref(ids[:2], owner="b")
        pool.ensure_writable(ids[0])
        pool.evict_cached(2)
        pool.allocate(3, owner="c")
        pool.decref(ids[3:], owner="a")
        pool._assert_invariants()  # explicit: nothing raised


class TestCow:
    def test_private_bundle_is_writable_as_is(self) -> None:
        pool = _pool(6)
        ids = pool.allocate(1, owner="a")
        assert pool.ensure_writable(ids[0]) == ids[0]
        assert not pool.drain_pending_cow()

    def test_shared_bundle_clones_and_remaps(self) -> None:
        pool = _pool(8)
        ids = pool.allocate(1, owner="a")
        pool.incref(ids, owner="b")  # ref_cnt=2: shared
        fresh = pool.ensure_writable(ids[0])
        assert fresh != ids[0]
        assert fresh >= pool.reserved
        assert fresh not in ids
        assert pool.drain_pending_cow() == [(fresh, ids[0])]
        # The clone is a private writable copy now.
        assert pool.bundles[fresh].ref_cnt == 1

    def test_cow_clone_then_both_sharers_release_balance(self) -> None:
        pool = _pool(10)
        ids = pool.allocate(1, owner="a")
        pool.cache_full_blocks([_key(5)], ids)
        pool.incref(ids, owner="b")
        fresh = pool.ensure_writable(ids[0])
        # Original released by its two sharers; clone released too.
        pool.decref(ids, owner="a")
        pool.decref(ids, owner="b")
        pool.decref([fresh], owner="cow")
        live = pool._ref_live_unique()
        assert live == 0
        assert pool._free_len == pool.num_bundles - pool.reserved

    def test_ensure_writable_on_free_bundle_is_refused(self) -> None:
        pool = _pool(6)
        ids = pool.allocate(1, owner="a")
        pool.decref(ids, owner="a")
        with pytest.raises(RuntimeError, match="allocate first"):
            pool.ensure_writable(ids[0])


class TestPrefixCache:
    def test_lookup_walks_contiguous_prefix(self) -> None:
        pool = _pool(8)
        ids = pool.allocate(3, owner="a")
        # vLLM block_hash_num_tokens is the FULL-PREFIX count at that block
        # boundary, so the keys carry cumulative token counts.
        keys = [_key(1, 128), _key(2, 256), _key(3, 384)]
        pool.cache_full_blocks(keys, ids)
        pool.decref(ids, owner="a")
        hit = pool.lookup_longest_prefix(keys)
        assert hit.hit
        assert hit.num_blocks == 3
        assert hit.effective_tokens == 384
        assert hit.bundle_ids == tuple(ids)

    def test_lookup_stops_at_first_miss(self) -> None:
        pool = _pool(8)
        ids = pool.allocate(3, owner="a")
        keys = [_key(1, 128), _key(2, 256), _key(3, 384)]
        pool.cache_full_blocks(keys, ids)
        pool.decref(ids, owner="a")
        query = [_key(1, 128), _key(2, 256), _key(999, 384)]  # diverges at block 3
        hit = pool.lookup_longest_prefix(query)
        assert hit.num_blocks == 2
        assert hit.effective_tokens == 256
        assert hit.bundle_ids == tuple(ids[:2])

    def test_lookup_with_shared_prefix_returns_first_candidate(self) -> None:
        pool = _pool(10)
        # Two requests share the same first page content.
        a = pool.allocate(1, owner="a")
        b = pool.allocate(1, owner="b")
        pool.cache_full_blocks([_key(1)], a)
        pool.cache_full_blocks([_key(1)], b)  # same content, two bundles
        pool.decref(a, owner="a")
        pool.decref(b, owner="b")
        hit = pool.lookup_longest_prefix([_key(1)])
        assert hit.hit and hit.num_blocks == 1
        assert hit.bundle_ids[0] in (a[0], b[0])

    def test_cache_publish_requires_live_bundle(self) -> None:
        pool = _pool(6)
        ids = pool.allocate(1, owner="a")
        pool.decref(ids, owner="a")
        with pytest.raises(RuntimeError, match="ref_cnt=0"):
            pool.publish_full_block(ids[0], _key(5))

    def test_invariant_8_draft_tokens_never_published(self) -> None:
        # The allocator cannot know what "draft" means; the rule is that only
        # publish_full_block is reachable for finalized pages. Lock the
        # property that publishing is an explicit opt-in, never implied by
        # allocation.
        pool = _pool(6)
        ids = pool.allocate(1, owner="a")
        assert pool.bundles[ids[0]].block_hash is None  # no hash on allocate


class TestEviction:
    def test_evict_cached_drops_hash_and_fires_callback(self) -> None:
        evicted: list[int] = []
        pool = _pool(8, on_evict_cached=lambda bid: evicted.append(bid))
        ids = pool.allocate(3, owner="a")
        pool.cache_full_blocks([_key(1), _key(2), _key(3)], ids)
        pool.decref(ids, owner="a")
        n = pool.evict_cached(2)
        assert n == 2
        assert evicted == ids[:2]
        assert pool._ref_cached_ref0() == 1
        # Evicted content is no longer hit-able.
        assert not pool.lookup_longest_prefix([_key(1), _key(2)]).hit

    def test_allocate_evicts_cached_when_pool_pressured(self) -> None:
        pool = _pool(5)
        a = pool.allocate(4, owner="a")  # uses 1..4
        pool.cache_full_blocks([_key(i) for i in range(4)], a)
        pool.decref(a, owner="a")  # 4 cached, 0 free
        # Allocate 1: must evict one cached bundle to satisfy.
        fresh = pool.allocate(1, owner="b")
        assert len(fresh) == 1
        assert pool._ref_cached_ref0() == 3
        assert pool._ref_live_unique() == 1

    def test_evict_cached_is_atomic_with_callback(self) -> None:
        # INV10: the callback fires exactly once per evicted bundle, and the
        # hash is gone before the callback (no half-evicted ghost).
        evicted: list[int] = []
        pool = _pool(8, on_evict_cached=lambda bid: evicted.append(bid))
        ids = pool.allocate(2, owner="a")
        pool.cache_full_blocks([_key(1), _key(2)], ids)
        pool.decref(ids, owner="a")
        pool.evict_cached(1)
        assert evicted == [ids[0]]
        assert pool.bundles[ids[0]].block_hash is None


class TestAdmission:
    def test_full_sequence_must_fit_check(self) -> None:
        pool = _pool(10)  # 9 usable (bundle 0 reserved as null)
        assert pool._check_admission_fits(8, owner="a")
        assert pool._check_admission_fits(9, owner="a")  # exactly fits
        assert not pool._check_admission_fits(10, owner="a")  # > 9 fails

    def test_watermark_reduces_admissible_capacity(self) -> None:
        pool = _pool(10, watermark_bundles=2)
        # 9 usable - 2 watermark = 7 admissible.
        assert pool._check_admission_fits(7, owner="a")
        assert not pool._check_admission_fits(8, owner="a")

    def test_cached_blocks_count_toward_capacity(self) -> None:
        # vLLM semantics: cached (ref_cnt==0, hashed) blocks are evictable,
        # so they count toward admission capacity.
        pool = _pool(10)
        ids = pool.allocate(4, owner="a")
        pool.cache_full_blocks([_key(i) for i in range(4)], ids)
        pool.decref(ids, owner="a")  # 4 cached, 5 free-unhashed
        # free(5) + cached(4) = 9 admissible.
        assert pool._check_admission_fits(9, owner="b")
        assert not pool._check_admission_fits(10, owner="b")

    def test_exhaustion_raises_runtime_error(self) -> None:
        pool = _pool(5)
        ids = pool.allocate(4, owner="a")
        pool.incref(ids, owner="b")  # all 4 live (shared), 0 free
        with pytest.raises(RuntimeError, match="pool exhausted"):
            pool.allocate(1, owner="c")

    def test_invariant_11_admitted_request_never_midflight_fails(self) -> None:
        # Once admitted (full_sequence_must_fit), the request's declared max
        # is reserved; chunked prefill grows into that reservation. Lock the
        # arithmetic: a request admitted for N never hits pool-exhausted.
        pool = _pool(20, watermark_bundles=2)
        assert pool._check_admission_fits(15, owner="long")
        # Its blocks are allocated up to its declared max without exhaustion.
        chunks = pool.allocate(15, owner="long")
        assert len(chunks) == 15


class TestStaleOwner:
    def test_owner_bookkeeping_is_strict(self) -> None:
        pool = _pool(8)
        with pytest.raises(ValueError, match="owner"):
            pool.allocate(1, owner="")
        with pytest.raises(ValueError, match="owner"):
            pool.incref([1], owner="")

    def test_reset_releases_all_owner_bundles(self) -> None:
        # Slot reset/reuse must not leak bundles (plan §7 invariant 7: 无旧
        # epoch page ownership). The allocator's decref-per-owner-release is
        # the primitive; this test locks that a full release returns every
        # bundle to the pool.
        pool = _pool(10)
        a = pool.allocate(4, owner="a")
        pool.cache_full_blocks([_key(i) for i in range(4)], a)
        pool.decref(a, owner="a")
        cached = pool._ref_cached_ref0()
        assert pool._free_len == 5 + cached
        # Next request reuses the pool without leaking.
        b = pool.allocate(4, owner="b")
        assert len(set(b)) == 4
        assert set(b).isdisjoint({0})
        pool.decref(b, owner="b")
        cached = pool._ref_cached_ref0()
        assert pool._free_len == 5 + cached
