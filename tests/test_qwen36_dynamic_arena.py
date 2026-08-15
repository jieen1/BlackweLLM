"""Phase 2 CPU tests: dynamic-arena mode of ``Qwen36SlotPool``.

``.omx/plans/qwen38-dynamic-context-vllm-plan.md`` Phase 2 -- "接入全局物理
arena,先完成 strict 模式". The CPU-verifiable half: the slot pool hands KV
physical pages out of a GLOBAL bundle pool sized by ``pool_bundles`` instead
of ``(num_slots + 1) * pages_per_slot`` fixed rows; every logical row starts
empty (null bundle); writes allocate on demand; sharing aliases with
refcounts; COW detaches before a shared suffix write; reset returns bundles.

The GPU half (eager/graph attention reading the dynamic page table, MTP
migration, base-pointer stability across replay) is verified by the Phase 5
matrix on real hardware; this file locks the bookkeeping the GPU path
depends on.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fla")
# sparkinfer 仓库自 2026-08-09 upstream merge 后以 ``b12x`` 为包名；
# runtime 通过 runtime.backends._sparkinfer_import.ensure_sparkinfer_path()
# 解析其路径。这里用 b12x 作为存在性探针。
pytest.importorskip("b12x")

from runtime.model.qwen36_slots import Qwen36SlotPool  # noqa: E402
from tests.test_qwen36_slot_pool import _stub_model  # noqa: E402


def _dynamic_pool(
    num_slots: int = 3,
    max_seq_len: int = 256,
    pool_bundles: int | None = None,
    watermark_bundles: int = 0,
) -> Qwen36SlotPool:
    return Qwen36SlotPool(
        _stub_model(["full_attention", "linear_attention", "linear_attention"]),
        num_slots=num_slots,
        max_seq_len=max_seq_len,
        device="cpu",
        dtype=torch.float32,
        dynamic_arena=True,
        pool_bundles=pool_bundles,
        watermark_bundles=watermark_bundles,
    )


class TestDynamicGeometry:
    def test_kv_tensor_spans_the_global_pool_not_fixed_rows(self) -> None:
        pool = _dynamic_pool(num_slots=3, max_seq_len=256, pool_bundles=20)
        # 3 slots + scratch = 4 rows x 2 pages = 8 fixed; the arena is 20.
        assert pool.pool_bundles == 20
        assert pool.k_pools[0].shape[0] == 20
        assert pool.dynamic_arena

    def test_default_pool_is_full_concurrent_capacity_plus_reserve(self) -> None:
        pool = _dynamic_pool(num_slots=3, max_seq_len=256)
        # strict default: 3 slots x 2 pages + 8 reserve.
        assert pool.pool_bundles == 3 * 2 + 8

    def test_logical_rows_start_empty_all_null(self) -> None:
        pool = _dynamic_pool()
        assert all(page == 0 for page in pool._page_table_host[0])
        assert all(page == 0 for page in pool._page_table_host[pool.scratch_row])
        assert torch.all(pool._global_page_table == 0)

    def test_capacity_snapshot_uses_pool_bundles(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=128, pool_bundles=12)
        snap = pool.capacity_snapshot()
        assert snap["qwen_kv_total_bundles"] == 12
        assert snap["kv_bytes_total"] == snap["kv_bytes_measured"]
        pool.assert_kv_storage_consistent()


class TestPhase4Reservations:
    def test_slot_reservation_is_consumed_by_growth_and_released_on_reset(self) -> None:
        pool = _dynamic_pool(
            num_slots=2,
            max_seq_len=256,
            pool_bundles=8,
            watermark_bundles=1,
        )

        assert pool.reserve_kv_capacity(0, 256)
        assert pool.kv_reservation_remaining(0) == 2
        pool.prepare_kv_writes(0, 0, 1)
        assert pool.kv_reservation_remaining(0) == 1

        pool.reset_slot(0)

        assert pool.kv_reservation_remaining(0) == 0
        assert pool._arena.usage().request_reserved_bundles == 0

    def test_partial_shared_tail_keeps_cow_reserve(self) -> None:
        pool = _dynamic_pool(
            num_slots=2,
            max_seq_len=256,
            pool_bundles=8,
            watermark_bundles=1,
        )
        pool.prepare_kv_writes(0, 0, 64)
        assert pool.reserve_kv_capacity(1, 256)

        pool.share_prefix_kv(0, 1, 64)
        assert pool.kv_reservation_remaining(1) == 2
        pool.prepare_kv_writes(1, 64, 1)
        assert pool.kv_reservation_remaining(1) == 1

    def test_full_shared_page_releases_that_part_of_reservation(self) -> None:
        pool = _dynamic_pool(
            num_slots=2,
            max_seq_len=256,
            pool_bundles=8,
            watermark_bundles=1,
        )
        pool.prepare_kv_writes(0, 0, 128)
        assert pool.reserve_kv_capacity(1, 256)

        pool.share_prefix_kv(0, 1, 128)

        assert pool.kv_reservation_remaining(1) == 1


class TestOnDemandAllocation:
    def test_prepare_kv_writes_allocates_on_first_write(self) -> None:
        pool = _dynamic_pool()
        pool.prepare_kv_writes(0, 0, 128)
        row0 = pool._page_table_host[0]
        assert row0[0] != 0  # first page now real
        assert row0[1] == 0  # second page still null
        assert pool._arena.bundles[row0[0]].ref_cnt == 1

    def test_write_index_follows_allocated_bundles(self) -> None:
        pool = _dynamic_pool()
        pool.prepare_kv_writes(0, 0, 128)
        assert pool.write_index(0, 0) == pool._page_table_host[0][0] * pool.page_size
        assert pool.write_index(0, 64) == pool._page_table_host[0][0] * pool.page_size + 64

    def test_cross_page_write_allocates_both_pages(self) -> None:
        pool = _dynamic_pool(max_seq_len=256)
        pool.prepare_kv_writes(0, 100, 100)  # touches pages 0 and 1
        assert pool._page_table_host[0][0] != 0
        assert pool._page_table_host[0][1] != 0
        assert pool._page_table_host[0][0] != pool._page_table_host[0][1]

    def test_allocated_bundles_are_distinct_per_slot(self) -> None:
        pool = _dynamic_pool()
        pool.prepare_kv_writes(0, 0, 128)
        pool.prepare_kv_writes(1, 0, 128)
        assert pool._page_table_host[0][0] != pool._page_table_host[1][0]


class TestCowUnderDynamicArena:
    def test_shared_prefix_detaches_before_suffix_write(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256)
        pool.prepare_kv_writes(0, 0, 128)
        pool.prepare_kv_writes(0, 128, 1)
        source = pool._page_table_host[0][0]
        pool.k_pools[0][source].fill_(7.0)

        # Slot 1 shares slot 0's prefix (both pages), then writes the suffix
        # at token 64 -- inside page 0, so that page must COW-detach.
        pool.share_prefix_kv(0, 1, 64)
        assert pool._page_table_host[1][0] == source
        assert pool._arena.bundles[source].ref_cnt == 2

        pool.prepare_kv_writes(1, 64, 1)
        target = pool._page_table_host[1][0]
        assert target != source
        assert pool._arena.bundles[source].ref_cnt == 1
        assert pool._arena.bundles[target].ref_cnt == 1
        # COW preserved the untouched prefix half.
        assert torch.all(pool.k_pools[0][target] == 7.0)
        pool.k_pools[0][target].fill_(11.0)
        assert torch.all(pool.k_pools[0][source] == 7.0)

    def test_shared_suffix_page_detaches_too(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256)
        pool.prepare_kv_writes(0, 0, 200)  # pages 0 and 1
        source0 = pool._page_table_host[0][0]
        source1 = pool._page_table_host[0][1]
        pool.share_prefix_kv(0, 1, 200)
        assert pool._arena.bundles[source1].ref_cnt == 2
        # Slot 1 writes at token 192 -- page 1 (shared) must detach, page 0 stays shared.
        pool.prepare_kv_writes(1, 192, 64)
        assert pool._page_table_host[1][0] == source0  # untouched
        assert pool._page_table_host[1][1] != source1  # detached
        assert pool._arena.bundles[source0].ref_cnt == 2
        assert pool._arena.bundles[source1].ref_cnt == 1


class TestScratchUnderDynamicArena:
    def test_scratch_snapshot_allocates_its_own_bundles(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256)
        pool.prepare_kv_writes(0, 0, 128)
        source = pool._page_table_host[0][0]
        pool.k_pools[0][source].fill_(7.0)
        pool.copy_prefix_to_scratch(0, 64)
        scratch = pool._page_table_host[pool.scratch_row][0]
        assert scratch != 0 and scratch != source
        assert torch.all(pool.k_pools[0][scratch] == 7.0)

    def test_scratch_share_aliases_and_detaches_on_write(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256)
        pool.prepare_kv_writes(0, 0, 128)
        source = pool._page_table_host[0][0]
        pool.k_pools[0][source].fill_(7.0)
        pool.copy_prefix_to_scratch(0, 64)
        scratch = pool._page_table_host[pool.scratch_row][0]

        pool.share_scratch_prefix(1, 64)
        assert pool._page_table_host[1][0] == scratch
        assert pool._arena.bundles[scratch].ref_cnt == 2

        pool.prepare_kv_writes(1, 64, 1)
        target = pool._page_table_host[1][0]
        assert target != scratch
        assert torch.all(pool.k_pools[0][target] == 7.0)
        pool.k_pools[0][target].fill_(11.0)
        assert torch.all(pool.k_pools[0][scratch] == 7.0)


class TestResetReturnsBundles:
    def test_reset_releases_all_slot_bundles(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256, pool_bundles=16)
        pool.prepare_kv_writes(0, 0, 200)  # 2 bundles
        owned_before = len(pool._slot_bundles[0])
        assert owned_before == 2
        pool.reset_slot(0)
        assert pool._slot_bundles[0] == set()
        assert all(page == 0 for page in pool._page_table_host[0])
        # All bundles are back in the free queue (none leaked, none shared).
        assert pool._arena.num_free_bundles() == 15  # 16 - null

    def test_reset_does_not_release_shared_bundles_held_by_others(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256, pool_bundles=16)
        pool.prepare_kv_writes(0, 0, 128)
        source = pool._page_table_host[0][0]
        pool.share_prefix_kv(0, 1, 64)  # slot 1 also references source
        pool.reset_slot(0)
        # Slot 1 still holds it; refcnt drops by 1 only.
        assert pool._arena.bundles[source].ref_cnt == 1
        assert pool._arena.num_free_bundles() == 14

    def test_invariant_7_no_old_epoch_ownership_after_reset(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256, pool_bundles=16)
        for _round in range(3):
            pool.prepare_kv_writes(0, 0, 128)
            pool.reset_slot(0)
        # No leaks across reuse rounds: 16 - null = 15 free.
        assert pool._arena.num_free_bundles() == 15
        assert pool._arena._ref_live_unique() == 0


class TestDynamicInvariants:
    def test_arena_invariants_hold_after_mixed_ops(self) -> None:
        pool = _dynamic_pool(num_slots=3, max_seq_len=256, pool_bundles=24)
        pool.prepare_kv_writes(0, 0, 128)
        pool.prepare_kv_writes(1, 0, 128)
        pool.share_prefix_kv(0, 2, 64)
        pool.prepare_kv_writes(2, 64, 64)  # COW
        pool.copy_prefix_to_scratch(0, 64)
        pool.reset_slot(1)
        pool._arena._assert_invariants()

    def test_pool_exhaustion_is_an_explicit_error(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256, pool_bundles=6)
        # 5 usable (1 null); each full-slot write needs 2 bundles. Two slot
        # writes use 4, leaving 1 free; a scratch snapshot of a full slot
        # wants 2 more -> pool exhausted, raised loudly.
        pool.prepare_kv_writes(0, 0, 256)  # 2 bundles
        pool.prepare_kv_writes(1, 0, 256)  # 2 bundles
        with pytest.raises(RuntimeError, match="pool exhausted"):
            pool.copy_prefix_to_scratch(0, 256)  # 2 more -> only 1 free


class TestPhase3PrefixCache:
    """Phase 3: the prefix KV survives a slot reset as arena-owned cached
    blocks (plan §6.4 state machine: FREE_UNHASHED -> LIVE_PRIVATE ->
    LIVE_SHARED -> CACHED_REF0 -> revived on lookup), instead of a fixed
    scratch row or an alias of the source slot's retained pages.

    Hash granularity is the BLOCK (block_size=64), not the page: one
    physical page (page_size=128) carries two hash blocks, both pointing at
    the same bundle -- vLLM's hash_block_size < block_size fine-grained
    lookup (plan §5.1C).
    """

    _BLOCK = 64

    def _keys(self, n_blocks: int, base: int = 7) -> list:
        from runtime.model.qwen36_kv_arena import BlockKey

        parent = None
        keys = []
        for i in range(n_blocks):
            parent = (parent or 0) * 31 + base + i
            keys.append(BlockKey(value=parent, num_tokens=self._BLOCK * (i + 1)))
        return keys

    def test_chained_keys_hash_each_token_once(self, monkeypatch) -> None:
        from runtime import block_pool
        from runtime.backends.qwen36 import _chained_block_keys

        calls: list[tuple[int | None, list[int]]] = []

        def _hash(parent, token_ids, extra_keys):
            calls.append((parent, list(token_ids)))
            return len(calls)

        monkeypatch.setattr(block_pool, "hash_block_tokens", _hash)

        keys = _chained_block_keys(list(range(12)), 12, 4)

        assert calls == [
            (None, [0, 1, 2, 3]),
            (1, [4, 5, 6, 7]),
            (2, [8, 9, 10, 11]),
        ]
        assert [key.num_tokens for key in keys] == [4, 8, 12]

    def test_bulk_publish_checks_invariants_once(self, monkeypatch) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256, pool_bundles=16)
        pool.prepare_kv_writes(0, 0, 128)
        calls = 0

        def _check_once():
            nonlocal calls
            calls += 1

        monkeypatch.setattr(pool._arena, "_maybe_invariant_check", _check_once)
        pool.publish_committed_blocks(0, 128, self._keys(2), self._BLOCK)

        assert calls == 1

    def test_publish_then_reset_leaves_bundles_cached_ref0(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256, pool_bundles=16)
        pool.prepare_kv_writes(0, 0, 128)
        bundle = pool._page_table_host[0][0]
        keys = self._keys(2)  # two 64-token blocks in one 128-token page
        n = pool.publish_committed_blocks(0, 128, keys, self._BLOCK)
        assert n == 2
        assert pool._arena.bundles[bundle].block_hash is not None
        pool.reset_slot(0)  # decref -> refcnt=0, hash retained (CACHED_REF0)
        assert pool._arena.bundles[bundle].ref_cnt == 0
        assert pool._arena.bundles[bundle].block_hash is not None
        assert pool._arena._ref_cached_ref0() == 1

    def test_partial_block_is_not_published(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256, pool_bundles=16)
        pool.prepare_kv_writes(0, 0, 100)  # one page, 100 tokens (1 full block)
        keys = self._keys(2)  # blocks at 64 and 128
        n = pool.publish_committed_blocks(0, 100, keys, self._BLOCK)
        assert n == 1  # only the 64-token block is committed (plan §5.1C)
        assert pool._arena._ref_cached_ref0() == 0  # bundle still live

    def test_partial_page_second_block_published_when_committed(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256, pool_bundles=16)
        pool.prepare_kv_writes(0, 0, 192)  # 192 = page0 full + 64 of page1
        keys = self._keys(3)  # blocks at 64, 128, 192
        n = pool.publish_committed_blocks(0, 192, keys, self._BLOCK)
        assert n == 3  # block at 192 (page 1, partial page, full block) too
        assert pool._arena._ref_cached_ref0() == 0  # all still live

    def test_restore_revives_cached_bundles_into_another_slot(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256, pool_bundles=16)
        pool.prepare_kv_writes(0, 0, 128)
        pool.prepare_kv_writes(0, 128, 64)  # page 1 partial (1 full block)
        keys = self._keys(3)  # 64, 128, 192
        pool.publish_committed_blocks(0, 192, keys, self._BLOCK)
        pool.reset_slot(0)
        b0 = pool._arena.lookup_longest_prefix(keys).bundle_ids[0]
        assert pool._arena.bundles[b0].ref_cnt == 0  # CACHED_REF0

        # Slot 1 restores the prefix: touch -> LIVE_SHARED, row points at it.
        restored, bundle_ids = pool.restore_prefix_from_arena(1, 192, keys)
        assert restored == 192
        assert len(set(bundle_ids)) == 2  # two pages
        assert pool._page_table_host[1][0] == bundle_ids[0]
        assert pool._page_table_host[1][1] == bundle_ids[1]
        # The bundle is the physical ownership unit: two hash blocks in one
        # page revive it once (ref_cnt=1), not once per block.
        assert pool._arena.bundles[b0].ref_cnt == 1

    def test_partial_page_restore_consumes_one_reserved_bundle(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256, pool_bundles=16)
        pool.prepare_kv_writes(0, 0, 64)
        keys = self._keys(1)
        pool.publish_committed_blocks(0, 64, keys, self._BLOCK)
        pool.reset_slot(0)
        assert pool.reserve_kv_capacity(1, 256)
        before = pool.kv_reservation_remaining(1)

        restored, bundle_ids = pool.restore_prefix_from_arena(1, 64, keys)

        assert restored == 64
        assert len(set(bundle_ids)) == 1
        assert pool.kv_reservation_remaining(1) == before - 1

    def test_miss_returns_zero(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256, pool_bundles=16)
        pool.prepare_kv_writes(0, 0, 128)
        pool.publish_committed_blocks(0, 128, self._keys(2, base=7), self._BLOCK)
        pool.reset_slot(0)
        restored, bundle_ids = pool.restore_prefix_from_arena(
            1,
            128,
            self._keys(2, base=99),  # different content
        )
        assert restored == 0
        assert bundle_ids == []

    def test_restored_prefix_detaches_on_first_write(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256, pool_bundles=16)
        pool.prepare_kv_writes(0, 0, 128)
        pool.prepare_kv_writes(0, 128, 1)  # page 1 exists for suffix
        keys = self._keys(2)
        pool.publish_committed_blocks(0, 128, keys, self._BLOCK)
        pool.reset_slot(0)
        restored, bundle_ids = pool.restore_prefix_from_arena(1, 128, keys)
        assert restored == 128
        # Suffix write at token 128 touches page 1 (fresh) -- must not
        # corrupt the restored shared page 0.
        pool.prepare_kv_writes(1, 128, 64)
        assert pool._page_table_host[1][0] == bundle_ids[0]
        assert pool._arena.bundles[bundle_ids[0]].ref_cnt == 1
        pool._arena._assert_invariants()

    def test_reset_releases_restored_bundles_back_to_cache(self) -> None:
        pool = _dynamic_pool(num_slots=2, max_seq_len=256, pool_bundles=16)
        pool.prepare_kv_writes(0, 0, 128)
        keys = self._keys(2)
        pool.publish_committed_blocks(0, 128, keys, self._BLOCK)
        pool.reset_slot(0)
        pool.restore_prefix_from_arena(1, 128, keys)
        assert pool._arena._ref_live_unique() == 1
        pool.reset_slot(1)
        # Back to CACHED_REF0 (refcnt=0, hash retained).
        assert pool._arena._ref_cached_ref0() == 1
        assert pool._arena._ref_live_unique() == 0
