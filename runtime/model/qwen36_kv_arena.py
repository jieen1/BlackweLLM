"""Phase 1: pure-metadata page-bundle allocator for Qwen3.8 dynamic KV.

``.omx/plans/qwen38-dynamic-context-vllm-plan.md`` Phase 1 -- "实现独立
page-bundle allocator": 先把 allocator 作为纯元数据组件做正确,再接 GPU
tensor。This module owns **bundle ownership only** (free queue, refcount,
block hash, LRU, COW); the physical backbone/MTP cache tensors stay out of
it until Phase 2 wires them to the same bundle ids.

Why a *bundle* rather than a block
----------------------------------
A logical Qwen KV page spans 17 physical tensors that must be allocated,
freed, aliased, and evicted **in lockstep**: the 16 full-attention FP8 K/V
backbone pages and the single MTP BF16 K/V page. vLLM handles this with a
``kv_cache_group`` per tensor family and a coordinator that keeps them in
sync; this runtime's model-specific contract (fixed 16 full-attention
layers, one MTP layer, one GPU) makes the lockstep the *unit* rather than a
reconciled afterthought. One ``bundle_id`` names all 17, so backbone and
MTP can never diverge (plan §6.1 -- the "MTP graph owner" work package).

Design provenance (plan §5, vLLM ``acb0f1dc``)
-----------------------------------------------
* ``KVCacheBlock`` (ref_cnt / block_hash / is_null / intrusive free-list
  links)  -> :class:`QwenPageBundle`.
* ``FreeKVCacheBlockQueue`` (O(1) popleft/append/prepend/remove LRU queue,
  ``block_hashes[num_cached_blocks:]``)  -> the free queue below.
* ``BlockPool.get_new_blocks``/``free_blocks``/``touch``/
  ``_maybe_evict_cached_block``  -> :meth:`QwenPageBundlePool.allocate` /
  ``decref`` / ``incref`` / ``_evict_cached_on_allocate``.
* ``BlockPool.cache_full_blocks``/``cache_partial_block`` -> the block-hash
  publish path.
* ``SingleTypeKVCacheManager`` state machine (FREE_UNHASHED -> LIVE_PRIVATE
  -> LIVE_SHARED -> CACHED_REF0 -> FREE_UNHASHED, plan §6.4) is the
  lifecycle :meth:`_assert_lifecycle` locks.

This is a torch-free module: CI runs a torch-free job, and this allocator's
correctness is a pure bookkeeping claim that must not depend on a torch
import. It is imported at module level by ``runtime/`` (via the arena
facade), so keep it free of third-party deps and of import-time side
effects (bfdiag rule).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BlockKey:
    """Content identity of one full published page bundle.

    Mirrors ``runtime/block_pool.py``'s ``BlockHash`` shape (``value`` +
    ``num_tokens``) but stands alone so this allocator stays decoupled from
    the Laguna/oracle allocator it was extracted alongside of. ``num_tokens``
    is the whole-prefix token count at the block boundary, enabling a cheap
    paranoid first-block token-count verify on a hit (same R7 idea).
    """

    value: int
    num_tokens: int


@dataclass
class QwenPageBundle:
    """Metadata of one physical page bundle (all 17 KV tensors).

    ``ref_cnt`` counts *logical page-table entries* pointing at this bundle
    (a shared prefix raises it; COW detach lowers it). ``block_hash`` is the
    published content hash, populated only for a **full, finalized, cached**
    page (plan §5.1C / §7 invariant 8: draft/rejected tokens never publish a
    hash). ``is_null`` marks the reserved null bundle that every unallocated
    page-table slot points at (plan §6.2 / §7 invariant 5: a null page can
    never be a KV write target).

    ``prev_free``/``next_free`` are the intrusive free-list links; both
    ``None`` exactly when the bundle is not parked in the free queue
    (allocated at ``ref_cnt > 0``, or a just-popped block). Same contract as
    ``runtime/block_pool.py``'s ``FreeBlockQueue``.
    """

    bundle_id: int
    ref_cnt: int = 0
    block_hash: BlockKey | None = None
    is_null: bool = False
    prev_free: QwenPageBundle | None = None
    next_free: QwenPageBundle | None = None


@dataclass
class QwenKVUsage:
    """One-time usage snapshot (plan §6.5 / Phase 4 metrics)."""

    total_bundles: int
    free_bundles: int
    live_bundles: int
    cached_bundles: int
    reserved_bundles: int
    request_reserved_bundles: int
    watermark_bundles: int
    cow_pending: int
    live_unique_bundles: int


@dataclass
class PrefixHit:
    """Longest full-page prefix hit resolved across the cache family.

    ``bundle_ids`` are the ``num_blocks`` physical bundles covering the
    common prefix (contiguous, in logical order); ``effective_tokens`` is the
    block-aligned token boundary all required cache families agree on (plan
    §5.1D invariant 9: ``state_hit <= kv_hit`` -- the caller must still
    truncate against the GDN state checkpoint family; this allocator owns KV
    blocks only).
    """

    num_blocks: int
    effective_tokens: int
    bundle_ids: tuple[int, ...] = ()
    hit: bool = False


class QwenPageBundlePool:
    """Fixed physical pool + dynamic logical allocation for Qwen3.8 KV.

    Built once at process start with ``num_bundles`` physical bundles plus
    ``reserved`` (default 1: the null bundle). Ownership is dynamic and
    content-addressed:

    * :meth:`allocate` hands out free-queue-front bundles, evicting a
      still-cached (ref_cnt == 0, hashed) bundle first (vLLM
      ``get_new_blocks`` + ``_maybe_evict_cached_block``).
    * :meth:`incref`/``decref`` track sharing; a bundle at ``ref_cnt == 0``
      keeps its hash and stays hit-able (LRU tail) or loses it and returns
      to the free queue head (LIFO reuse), matching vLLM ``free_blocks``
      order.
    * :meth:`cache_full_blocks` publishes hashes only for finalized full
      pages; :meth:`lookup_longest_prefix` walks the request's key chain
      back-to-front for the deepest contiguous full-page hit.
    * :meth:`ensure_writable` returns a bundle the caller may write to:
      the same bundle when it is private (ref_cnt == 1), or a freshly
      allocated private clone when the page is shared -- the COW primitive
      that keeps a shared tail from being overwritten (plan §6.4).

    Lifecycle state machine (plan §6.4), locked by :meth:`_assert_lifecycle`::

        FREE_UNHASHED -> LIVE_PRIVATE -> LIVE_SHARED -> CACHED_REF0
        FREE_UNHASHED -> reused (allocate) ; CACHED_REF0 -> LIVE_SHARED (incref)

    Twelve core invariants (plan §7) are checked at every mutating call when
    ``assert_invariants`` is enabled (test default; production may disable
    for speed once the state machine is proven).
    """

    def __init__(
        self,
        num_bundles: int,
        *,
        reserved: int = 1,
        watermark_bundles: int = 0,
        enable_caching: bool = True,
        assert_invariants: bool = True,
        on_evict_cached: Callable[[int], None] | None = None,
    ) -> None:
        if num_bundles < 1:
            raise ValueError(f"num_bundles must be >= 1, got {num_bundles}")
        if reserved < 1:
            raise ValueError("must reserve at least the null bundle (INV5/INV7)")
        if num_bundles <= reserved:
            raise ValueError(f"num_bundles={num_bundles} must exceed reserved={reserved}")
        if watermark_bundles < 0:
            raise ValueError(f"watermark_bundles must be >= 0, got {watermark_bundles}")

        self.num_bundles = num_bundles
        self.reserved = reserved
        self.enable_caching = enable_caching
        self.assert_invariants = assert_invariants
        #: Called when a still-cached bundle (ref_cnt == 0, hashed) is
        #: evicted by :meth:`allocate` or :meth:`evict_cached`. The caller
        #: (Phase 3's persistent-prefix layer) drops the co-keyed GDN state
        #: checkpoint in lockstep -- plan §6.4 "eviction 顺序必须是一个原子
        #: 事务" / §7 invariant 10. ``None`` for a stand-alone allocator.
        self._on_evict_cached = on_evict_cached

        self.bundles: list[QwenPageBundle] = [
            QwenPageBundle(bundle_id=i) for i in range(num_bundles)
        ]
        # Null bundle = bundle 0 (vLLM ``null_block``). Never cached, never
        # allocated, never a write target; every unallocated page-table slot
        # points at it.
        self.null_bundle = self.bundles[0]
        self.null_bundle.is_null = True

        # Free queue: intrusive LRU doubly-linked list. Front (``popleft``) =
        # evict-next; tail (``append``) = most-recently-freed. Excludes every
        # reserved id by construction.
        self._free_queue: list[QwenPageBundle] = []
        self._free_head: QwenPageBundle | None = None
        self._free_tail: QwenPageBundle | None = None
        self._free_len = 0
        for bundle_id in range(reserved, num_bundles):
            self._free_append(self.bundles[bundle_id])

        # Content index: ``block_hash.value -> list of physical bundles``.
        # A hash may map to several bundles after a COW clone publishes the
        # same content (vLLM ``BlockHashToBlockMap`` keeps a dict); the
        # lookup returns the first (LRU-oldest) candidate.
        self._hash_to_bundles: dict[int, list[int]] = {}
        # Reverse: bundle_id -> published BlockKeys (a bundle can be
        # reachable from more than one boundary after partial promotion, vLLM
        # ``cached_block_hashes_by_block``).
        self._bundle_hashes: dict[int, dict[int, BlockKey]] = {}
        #: COW clones issued but not yet copied (bundle_id -> source). Phase 2
        #: drains this to drive the device copies and clears it; until then it
        #: is the plan's "pending copy" ledger.
        self._pending_cow: dict[int, int] = {}
        self._watermark_bundles = watermark_bundles
        # owner -> logical pages still guaranteed to an admitted request but
        # not yet satisfied by an allocation or an existing shared page.
        # The engine thread is the only production mutator, so this needs no
        # lock; keeping it here makes check+reserve one atomic Python call.
        self._reservations: dict[str, int] = {}

    # -- free-queue primitives (vLLM FreeKVCacheBlockQueue shape) -----------

    def _free_append(self, bundle: QwenPageBundle) -> None:
        assert bundle.prev_free is None and bundle.next_free is None
        if self._free_tail is None:
            self._free_head = bundle
            self._free_tail = bundle
        else:
            self._free_tail.next_free = bundle
            bundle.prev_free = self._free_tail
            self._free_tail = bundle
        self._free_len += 1

    def _free_prepend(self, bundle: QwenPageBundle) -> None:
        assert bundle.prev_free is None and bundle.next_free is None
        if self._free_head is None:
            self._free_head = bundle
            self._free_tail = bundle
        else:
            self._free_head.prev_free = bundle
            bundle.next_free = self._free_head
            self._free_head = bundle
        self._free_len += 1

    def _free_popleft(self) -> QwenPageBundle:
        if self._free_head is None:
            raise RuntimeError("free queue is empty")
        bundle = self._free_head
        nxt = bundle.next_free
        if nxt is None:
            self._free_head = None
            self._free_tail = None
        else:
            nxt.prev_free = None
            self._free_head = nxt
        bundle.prev_free = None
        bundle.next_free = None
        self._free_len -= 1
        return bundle

    def _free_remove(self, bundle: QwenPageBundle) -> None:
        if not self._free_contains(bundle):
            raise RuntimeError(f"remove() on bundle {bundle.bundle_id} not in the free queue")
        prev = bundle.prev_free
        nxt = bundle.next_free
        if prev is None:
            self._free_head = nxt
        else:
            prev.next_free = nxt
        if nxt is None:
            self._free_tail = prev
        else:
            nxt.prev_free = prev
        bundle.prev_free = None
        bundle.next_free = None
        self._free_len -= 1

    def _free_contains(self, bundle: QwenPageBundle) -> bool:
        # A single-element queue has head == tail == bundle with BOTH links
        # None, so link presence alone cannot distinguish "in the queue"
        # from "popped". Head/tail identity closes the gap.
        return (
            bundle is self._free_head
            or bundle is self._free_tail
            or bundle.prev_free is not None
            or bundle.next_free is not None
        )

    def num_free_bundles(self) -> int:
        return self._free_len

    # -- content index ------------------------------------------------------

    def _drop_hash(self, bundle: QwenPageBundle) -> list[BlockKey]:
        """Remove every published hash from ``bundle`` (eviction / re-publish).

        Returns the dropped keys. ``None`` content (never cached) drops
        nothing. This is the single place hashes leave the index, so INV2
        (a bundle is never both indexed as old content AND handed out for
        new) holds by construction.
        """
        keys = self._bundle_hashes.pop(bundle.bundle_id, None)
        if not keys:
            bundle.block_hash = None
            return []
        dropped: list[BlockKey] = []
        for value, key in keys.items():
            holders = self._hash_to_bundles.get(value)
            if holders:
                holders.remove(bundle.bundle_id)
                if not holders:
                    self._hash_to_bundles.pop(value, None)
            dropped.append(key)
        bundle.block_hash = None
        return dropped

    def _publish(self, bundle: QwenPageBundle, key: BlockKey) -> None:
        """Publish ``key`` for ``bundle`` (full finalized page only)."""
        if bundle.is_null:
            raise RuntimeError("null bundle can never be cached (INV5)")
        self._bundle_hashes.setdefault(bundle.bundle_id, {})[key.value] = key
        holders = self._hash_to_bundles.setdefault(key.value, [])
        if bundle.bundle_id not in holders:
            holders.append(bundle.bundle_id)
        bundle.block_hash = key

    # -- lifecycle helpers --------------------------------------------------

    def _ref_live_unique(self) -> int:
        return sum(1 for b in self.bundles if b.ref_cnt > 0)

    def _ref_cached_ref0(self) -> int:
        return sum(1 for b in self.bundles if b.ref_cnt == 0 and b.block_hash is not None)

    def _assert_lifecycle(self) -> None:
        """Lock the plan §6.4 state machine at every mutation.

        Every bundle is in exactly one of FREE_UNHASHED (free queue, no
        hash), LIVE_PRIVATE/LIVE_SHARED (ref_cnt > 0), or CACHED_REF0
        (ref_cnt == 0, hashed, parked in the free queue). A hashed bundle
        MUST be in the free queue (it is hit-able, LRU); an unhashed
        ref_cnt == 0 bundle MUST be in the free queue too (LIFO reuse).
        ref_cnt > 0 bundles are NEVER in the free queue (INV9).
        """
        for bundle in self.bundles:
            if bundle.is_null:
                if bundle.ref_cnt != 0 or bundle.block_hash is not None:
                    raise RuntimeError("null bundle must stay ref_cnt=0 and unhashed")
                continue
            in_queue = self._free_contains(bundle)
            if bundle.ref_cnt > 0:
                if in_queue:
                    raise RuntimeError(
                        f"INV9: bundle {bundle.bundle_id} is in the free queue at ref_cnt>0"
                    )
            elif bundle.block_hash is not None:
                # CACHED_REF0: freed but hit-able, parked in the LRU tail.
                if not in_queue:
                    raise RuntimeError(
                        f"cached bundle {bundle.bundle_id} (ref_cnt=0, hashed) is not parked"
                    )
            else:
                if not in_queue:
                    raise RuntimeError(
                        f"free-unhashed bundle {bundle.bundle_id} is not in the free queue"
                    )

    def _assert_invariants(self) -> None:
        """Check the plan §7 core invariants that this allocator owns."""
        self._assert_lifecycle()
        total = self.num_bundles - self.reserved
        live = self._ref_live_unique()
        cached = self._ref_cached_ref0()
        # INV1: free + cached_ref0 + live_unique == total_usable_bundles.
        # The free queue holds both CACHED_REF0 (hashed) and FREE_UNHASHED
        # bundles, so free-unhashed is the queue length minus the cached.
        free = self._free_len - cached
        if free + cached + live != total:
            raise RuntimeError(
                f"INV1 violated: free={free} cached_ref0={cached} live_unique={live} "
                f"!= usable={total}"
            )
        # INV2: a bundle is never both in the free queue and live/cached-owned.
        # (enforced by _assert_lifecycle: ref_cnt>0 => not in queue)
        # INV4: ref_cnt > 0 bundles are never evicted (enforced in allocate).
        # INV10: eviction drops KV + state together (callback contract, Phase 3).
        # INV5: null page never a write target -- the only caller-side check;
        # this allocator refuses to allocate/cache the null bundle.

    def _maybe_invariant_check(self) -> None:
        if self.assert_invariants:
            self._assert_invariants()

    # -- public API ---------------------------------------------------------

    def allocate(self, count: int, *, owner: str) -> list[int]:
        """Hand out ``count`` bundles from the free-queue front to ``owner``.

        A popped bundle that still carries a published hash is a CACHED_REF0
        entry and is EVICTED first -- its hash leaves the content index and,
        in lockstep, the co-keyed GDN checkpoint is dropped via
        ``_on_evict_cached`` (plan §6.4, INV10). A popped bundle is always
        ``ref_cnt == 0`` by construction.

        Raises only on true exhaustion: every non-reserved bundle is either
        live (ref_cnt > 0, never evictable -- INV4) or cached (ref_cnt == 0
        but eviction has to run; if the free queue is empty there are no
        cached candidates either). Callers (Phase 4 admission) size the pool
        generously so this is rare; Phase 4 turns it into a deterministic
        wait/reject *before* the first GPU write.
        """
        if count < 0:
            raise ValueError(f"cannot allocate a negative count ({count})")
        if not isinstance(owner, str) or not owner:
            raise ValueError("owner must be a non-empty string")
        if count > self._free_len:
            raise RuntimeError(
                f"bundle pool exhausted: requested {count}, only {self._free_len} free "
                f"(of {self.num_bundles - self.reserved} usable, excluding {self.reserved} "
                "reserved); every other bundle is ref_cnt > 0 or cached"
            )
        owner_reserved = self._reservations.get(owner, 0)
        if owner_reserved:
            if count > owner_reserved:
                raise RuntimeError(
                    f"owner {owner!r} requested {count} bundles with only {owner_reserved} reserved"
                )
        else:
            unreserved = self._unreserved_free_bundles()
            if count > unreserved:
                raise RuntimeError(
                    f"bundle allocation would consume reserved capacity: owner={owner!r} "
                    f"requested={count}, unreserved={unreserved}, "
                    f"request_reserved={sum(self._reservations.values())}, "
                    f"watermark={self._watermark_bundles}"
                )
        ids: list[int] = []
        for _ in range(count):
            bundle = self._free_popleft()
            if bundle.bundle_id < self.reserved:
                raise RuntimeError(
                    f"INV7: reserved bundle {bundle.bundle_id} was in the free queue"
                )
            if bundle.ref_cnt != 0:
                raise RuntimeError(
                    f"bundle {bundle.bundle_id} was in the free queue at ref_cnt={bundle.ref_cnt}"
                )
            if bundle.block_hash is not None:
                # CACHED_REF0 -> reused: evict the cached content first.
                self._drop_hash(bundle)
                if self._on_evict_cached is not None:
                    self._on_evict_cached(bundle.bundle_id)
            bundle.ref_cnt = 1
            ids.append(bundle.bundle_id)
        self._consume_reservation(owner, count)
        self._maybe_invariant_check()
        return ids

    def incref(self, bundle_ids: Sequence[int], *, owner: str) -> None:
        """Reference already-allocated bundles (vLLM ``touch``).

        The sharing primitive: a sibling request aliases an existing prefix's
        physical bundles, so their ``ref_cnt`` rises and they leave the free
        queue if a cached hit just revived them (LIVE_SHARED). Refusing to
        incref a ``ref_cnt == 0`` block whose hash is gone would alias stale
        content -- the same guard vLLM's ``touch`` enforces by only touching
        blocks that exist.
        """
        if not isinstance(owner, str) or not owner:
            raise ValueError("owner must be a non-empty string")
        bundle_ids = list(bundle_ids)
        for bundle_id in bundle_ids:
            if bundle_id < self.reserved:
                raise RuntimeError(f"INV7: cannot reference reserved bundle {bundle_id}")
            if bundle_id >= self.num_bundles:
                raise RuntimeError(
                    f"bundle {bundle_id} is out of range (num_bundles={self.num_bundles})"
                )
        revivals = sum(1 for bundle_id in bundle_ids if self.bundles[bundle_id].ref_cnt == 0)
        owner_reserved = self._reservations.get(owner, 0)
        if (
            revivals > owner_reserved
            and revivals - owner_reserved > self._unreserved_free_bundles()
        ):
            raise RuntimeError(
                f"cached hit would consume reserved capacity: owner={owner!r}, "
                f"revivals={revivals}, owner_reserved={owner_reserved}, "
                f"unreserved={self._unreserved_free_bundles()}"
            )
        for bundle_id in bundle_ids:
            if bundle_id < self.reserved:
                raise RuntimeError(f"INV7: cannot reference reserved bundle {bundle_id}")
            if bundle_id >= self.num_bundles:
                raise RuntimeError(
                    f"bundle {bundle_id} is out of range (num_bundles={self.num_bundles})"
                )
            bundle = self.bundles[bundle_id]
            if bundle.ref_cnt == 0:
                if self._free_contains(bundle):
                    # A freed bundle is only a valid prefix-cache hit if it
                    # still carries the published hash -- an unhashed freed
                    # bundle has no cached content to revive and would alias
                    # garbage. vLLM's ``touch`` can revive a hashed freed
                    # block (the late-hit revival primitive); the unhashed
                    # case is a caller bug (INV8-adjacent: never reuse stale
                    # content as a hit).
                    if bundle.block_hash is None:
                        raise RuntimeError(
                            f"cannot reference freed-unhashed bundle {bundle_id} "
                            "(no cached content to revive)"
                        )
                    self._free_remove(bundle)
                else:
                    raise RuntimeError(
                        f"cannot reference bundle {bundle_id} at ref_cnt=0 not in free queue"
                    )
            bundle.ref_cnt += 1
        self._maybe_invariant_check()

    def decref(self, bundle_ids: Sequence[int], *, owner: str) -> None:
        """Release bundles back to the pool (vLLM ``free_blocks`` order).

        ``ref_cnt -= 1``; at 0, re-queue: a bundle that KEEPS its hash goes
        to the LRU tail (stays hit-able -- CACHED_REF0); a bundle with no
        hash goes to the head (evicted first -- FREE_UNHASHED LIFO reuse).
        Written as a decrement so a legitimately shared bundle
        (ref_cnt > 1) is not silently masked; only the final release
        re-queues. ``owner`` is bookkeeping for Phase 4's per-request
        reservation; this allocator does not gate on it yet.
        """
        if not isinstance(owner, str) or not owner:
            raise ValueError("owner must be a non-empty string")
        for bundle_id in bundle_ids:
            if bundle_id < self.reserved:
                raise RuntimeError(f"INV7: cannot free reserved bundle {bundle_id}")
            if bundle_id >= self.num_bundles:
                raise RuntimeError(
                    f"bundle {bundle_id} is out of range (num_bundles={self.num_bundles})"
                )
            bundle = self.bundles[bundle_id]
            if bundle.ref_cnt <= 0:
                raise RuntimeError(f"double-free of bundle {bundle_id} (ref_cnt={bundle.ref_cnt})")
            bundle.ref_cnt -= 1
            if bundle.ref_cnt == 0:
                if bundle.block_hash is not None:
                    self._free_append(bundle)  # cached: LRU tail
                else:
                    self._free_prepend(bundle)  # unhashed: LIFO head
        self._maybe_invariant_check()

    def ensure_writable(self, bundle_id: int, *, owner: str = "cow") -> int:
        """Return a bundle the caller may write to, cloning if shared.

        A private bundle (``ref_cnt == 1``, not null) is writable as-is and
        returned unchanged. A shared bundle (``ref_cnt > 1``) or a
        freed-but-cached one is COW-detached: a fresh bundle is allocated and
        recorded in ``_pending_cow[bundle_id] -> source`` so Phase 2 can copy
        the source bytes before the first write (plan §6.4 "shared tail 在任
        何写入前 COW"; §7 invariant 3 -- a writable tail's refcount must be 1).
        """
        if bundle_id < self.reserved:
            raise RuntimeError(f"INV7: cannot write to reserved bundle {bundle_id}")
        bundle = self.bundles[bundle_id]
        if bundle.ref_cnt == 1 and not bundle.is_null:
            return bundle_id
        if bundle.ref_cnt == 0:
            raise RuntimeError(
                f"ensure_writable on free bundle {bundle_id} (ref_cnt=0) -- allocate first"
            )
        # Shared: clone.
        fresh = self.allocate(1, owner=owner)
        target = fresh[0]
        self._pending_cow[target] = bundle_id
        return target

    def publish_full_block(self, bundle_id: int, key: BlockKey) -> None:
        """Publish a finalized full page's hash (vLLM ``cache_full_blocks``).

        Only called for a full, committed, finalized page -- draft and
        rejected tokens never reach here (plan §5.1C / §7 invariant 8).
        Idempotent for an already-published value on the same bundle.
        """
        if bundle_id >= self.num_bundles:
            raise RuntimeError(
                f"bundle {bundle_id} is out of range (num_bundles={self.num_bundles})"
            )
        bundle = self.bundles[bundle_id]
        if bundle.is_null:
            raise RuntimeError("null bundle can never be cached (INV5)")
        if bundle_id < self.reserved:
            raise RuntimeError(f"INV7: cannot cache reserved bundle {bundle_id}")
        if bundle.ref_cnt < 1:
            raise RuntimeError(f"cannot publish a hash for free bundle {bundle_id} (ref_cnt=0)")
        self._publish(bundle, key)
        self._maybe_invariant_check()

    def cache_full_blocks(self, keys: Sequence[BlockKey], bundle_ids: Sequence[int]) -> None:
        """Publish hashes for a run of finalized full pages in logical order."""
        if len(keys) != len(bundle_ids):
            raise ValueError("keys and bundle_ids must have equal length")
        for key, bundle_id in zip(keys, bundle_ids, strict=True):
            if bundle_id >= self.num_bundles:
                raise RuntimeError(
                    f"bundle {bundle_id} is out of range (num_bundles={self.num_bundles})"
                )
            bundle = self.bundles[bundle_id]
            if bundle.is_null:
                raise RuntimeError("null bundle can never be cached (INV5)")
            if bundle_id < self.reserved:
                raise RuntimeError(f"INV7: cannot cache reserved bundle {bundle_id}")
            if bundle.ref_cnt < 1:
                raise RuntimeError(f"cannot publish a hash for free bundle {bundle_id} (ref_cnt=0)")
            self._publish(bundle, key)
        # The single-key API checks after every mutation. A run is one
        # transaction, so checking once avoids O(keys * pool_bundles) work
        # when invariant assertions are enabled on a 128K diagnostic run.
        self._maybe_invariant_check()

    def lookup_longest_prefix(
        self, keys: Sequence[BlockKey], *, max_blocks: int | None = None
    ) -> PrefixHit:
        """Deepest contiguous full-page prefix hit over ``keys``.

        Walks the request's key chain in logical order and returns the
        longest run of hashes that all resolve, stopping at the first miss
        (the chained-hash property means divergence at any earlier token
        changes every later block's hash, so a miss halts the search -- vLLM
        ``find_longest_cache_hit``). The returned ``bundle_ids`` are the
        LRU-oldest candidates per hash; the caller (Phase 3) will
        :meth:`incref` them so they stay pinned. ``effective_tokens`` is the
        block-aligned boundary; the caller truncates against GDN state
        family before restoring (plan §5.1D).
        """
        if not keys:
            return PrefixHit(num_blocks=0, effective_tokens=0, hit=False)
        limit = len(keys) if max_blocks is None else min(max_blocks, len(keys))
        ids: list[int] = []
        for key in keys[:limit]:
            holders = self._hash_to_bundles.get(key.value)
            if not holders:
                break
            ids.append(holders[0])
        if not ids:
            return PrefixHit(num_blocks=0, effective_tokens=0, hit=False)
        num_tokens = keys[len(ids) - 1].num_tokens
        return PrefixHit(
            num_blocks=len(ids),
            effective_tokens=num_tokens,
            bundle_ids=tuple(ids),
            hit=True,
        )

    def evict_cached(self, count: int) -> int:
        """Evict up to ``count`` cached (ref_cnt == 0, hashed) bundles.

        Walks the free queue from the evict-next (head) end, dropping hashes
        and firing ``_on_evict_cached`` for the co-keyed GDN checkpoint --
        an atomic transaction (plan §6.4). Returns the number actually
        evicted (may be < ``count`` when fewer cached candidates exist).
        """
        if count < 0:
            raise ValueError(f"cannot evict a negative count ({count})")
        evicted = 0
        # Only cached bundles (hash present, ref_cnt == 0) are candidates;
        # unhashed free bundles are evicted implicitly by allocate and carry
        # no cached content to drop.
        candidates = [b for b in self.bundles if b.ref_cnt == 0 and b.block_hash is not None]
        for bundle in candidates:
            if evicted >= count:
                break
            if self._free_contains(bundle):
                self._free_remove(bundle)
            self._drop_hash(bundle)
            if self._on_evict_cached is not None:
                self._on_evict_cached(bundle.bundle_id)
            self._free_prepend(bundle)  # now free-unhashed, LIFO reuse
            evicted += 1
        self._maybe_invariant_check()
        return evicted

    def drain_pending_cow(self) -> list[tuple[int, int]]:
        """Return and clear the pending COW clones ``(dst, src)``.

        Phase 2 calls this before the device copy launch and clears the
        ledger once the copies have run; until then a pending clone's dst is
        not writable through the normal path.
        """
        pairs = sorted(self._pending_cow.items())
        self._pending_cow.clear()
        return [(dst, src) for dst, src in pairs]

    def usage(self) -> QwenKVUsage:
        """Usage snapshot for admission/metadata (plan §6.5 / Phase 4)."""
        live = self._ref_live_unique()
        cached = self._ref_cached_ref0()
        return QwenKVUsage(
            total_bundles=self.num_bundles,
            free_bundles=self._free_len,
            live_bundles=live,
            cached_bundles=cached,
            reserved_bundles=self.reserved,
            request_reserved_bundles=sum(self._reservations.values()),
            watermark_bundles=self._watermark_bundles,
            cow_pending=len(self._pending_cow),
            live_unique_bundles=live,
        )

    def _check_admission_fits(
        self, num_required_bundles: int, *, owner: str, include_watermark: bool = True
    ) -> bool:
        """vLLM ``allocate_slots`` admission gate (plan §6.5).

        ``full_sequence_must_fit`` reservation: the request's complete
        declared max length must fit in the pool's reclaimable bundles minus
        the watermark reserve. The free queue already holds every
        ``ref_cnt == 0`` bundle -- both free-unhashed (LIFO reuse) and
        CACHED_REF0 (LRU, evictable on allocation) -- so its length IS the
        reclaimable capacity, exactly vLLM's
        ``required_blocks > block_pool.get_num_free_blocks()`` in
        ``KVCacheManager.allocate_slots``. Returns False (never raises) so
        Phase 4 can queue or reject deterministically before the first GPU
        write (plan §6.5 "capacity 不足进入 waiting queue 或明确返回容量原
        因;不得先运行后失败").
        """
        if not isinstance(owner, str) or not owner:
            raise ValueError("owner must be a non-empty string")
        # A retry for the same owner may reuse its existing guarantee. Other
        # owners' promises are unavailable even though those pages have not
        # necessarily been materialized yet.
        available = self._free_len - sum(
            count for reserved_owner, count in self._reservations.items() if reserved_owner != owner
        )
        if include_watermark:
            available -= self._watermark_bundles
        return num_required_bundles <= available

    def reserve(self, count: int, *, owner: str) -> bool:
        """Atomically guarantee ``count`` future logical pages to ``owner``.

        Returns ``False`` without mutation when the global pool cannot honor
        the guarantee. A duplicate owner is a lifecycle bug: callers must
        release or reset the slot before attempting another admission.
        """
        if count < 0:
            raise ValueError(f"cannot reserve a negative count ({count})")
        if not isinstance(owner, str) or not owner:
            raise ValueError("owner must be a non-empty string")
        if owner in self._reservations:
            raise RuntimeError(f"owner {owner!r} already has a KV reservation")
        if not self._check_admission_fits(count, owner=owner):
            return False
        if count:
            self._reservations[owner] = count
        return True

    def reserved_for(self, owner: str) -> int:
        """Return the unconsumed request reservation for diagnostics/tests."""
        return self._reservations.get(owner, 0)

    def release_reservation(self, *, owner: str) -> int:
        """Release only ``owner``'s unmaterialized guarantee.

        Live pages remain owned until the normal ``decref`` lifecycle runs.
        The return value is the number of promised pages released.
        """
        return self._reservations.pop(owner, 0)

    def consume_reservation(self, count: int, *, owner: str) -> None:
        """Mark ``count`` promised pages satisfied without allocating them.

        Prefix sharing calls this for complete read-only pages. A partial
        shared tail deliberately remains reserved because its first suffix
        write must COW-clone it before touching bytes.
        """
        if count < 0:
            raise ValueError(f"cannot consume a negative count ({count})")
        self._consume_reservation(owner, count)

    def _consume_reservation(self, owner: str, count: int) -> None:
        remaining = self._reservations.get(owner)
        if remaining is None or count == 0:
            return
        if count > remaining:
            raise RuntimeError(
                f"owner {owner!r} consumed {count} pages with only {remaining} reserved"
            )
        remaining -= count
        if remaining:
            self._reservations[owner] = remaining
        else:
            self._reservations.pop(owner, None)

    def _unreserved_free_bundles(self) -> int:
        return max(
            0,
            self._free_len - sum(self._reservations.values()) - self._watermark_bundles,
        )
