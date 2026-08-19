"""B5 模块化：分页/前缀缓存域。

从 direct_model_runner.py 提取的 Block/BlockPool/FreeBlockQueue/hash 基础设施。
纯移动不改逻辑（B5 parity 门禁）。
"""

from __future__ import annotations

import array
import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass

from bfdiag.invariants import checks as bfdiag_checks

# something about index 0 (padding/NULL_BLOCK_ID-adjacent) makes the model
# read/write the wrong state. Fix: reserve physical index 0 permanently
# and offset every logical slot by +1 when computing a physical address.
RESERVED_PHYSICAL_SLOTS = 1


def _physical_slot(logical_slot: int) -> int:
    return logical_slot + RESERVED_PHYSICAL_SLOTS


def _initial_block_table(logical_slot: int, blocks_per_slot: int) -> list[int]:
    """P0 "thin allocator" (``notes/prefix-cache-design.md`` sec 5, P0 --
    block-table indirection substrate): hands out the SAME contiguous
    physical block ids the arange-based addressing has always used --
    ``[first_block, first_block + blocks_per_slot)`` where ``first_block =
    _physical_slot(logical_slot) * blocks_per_slot``. This is the only
    thing P1's dynamic free-list allocator will change; every downstream
    consumer (``build_attention_metadata``/``_batch``, ``_slot_mapping``/
    ``_batch``, ``CapturedBatchDecodeGraph``/``CapturedMTPDraftStepGraph``
    ``._fill_buffers``) reads ``block_table[slot]`` without caring how it
    was populated -- behavior-identical by construction, since the values
    returned here are byte-identical to what the old formula computed
    inline.
    """
    first_block = _physical_slot(logical_slot) * blocks_per_slot
    return list(range(first_block, first_block + blocks_per_slot))


def _ssm_spec_row(logical_slot: int, col: int, total_physical_slots: int, num_spec: int) -> int:
    """Physical SSM-state row for MTP verify's K+1 candidate positions
    (Phase 2, 2026-07-18 -- ``notes/2026-07-17-post-ragged-round-next-steps.md``
    section 10/11). Re-derived and reconfirmed directly against the real
    kernel this round (``fused_sigmoid_gating_delta_rule_update_kernel``,
    ``vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py``): that
    kernel reads its initial state from
    ``ssm_state_indices[i_n, num_accepted_tokens[i_n] - 1]`` and
    unconditionally WRITES a result into ``ssm_state_indices[i_n, t]`` for
    every candidate position ``t`` in this round's batch -- i.e. real
    vLLM's own ``spec_state_indices_tensor`` is a FIXED per-(request,
    column) physical-row mapping (there, a slice of that request's own
    persistent block table, allocated once and never reshuffled across
    rounds), and it is ``num_accepted_tokens`` -- supplied fresh each
    round from the PREVIOUS round's real accept/reject outcome -- that
    selects which of those fixed rows holds the valid incoming state.
    This function is the fixed-slot-runtime analogue of that same fixed
    per-(slot, column) mapping: column 0 always resolves to
    ``_physical_slot(logical_slot)`` -- the SAME row the ordinary
    chunked/non-spec GDN path already writes to (this is what makes the
    bootstrap case -- the first spec verify round right after a real
    prefill -- correct: passing ``num_accepted_tokens_prev=1`` selects
    ``i_t=0``, i.e. column 0, i.e. exactly the row the prefill's chunked
    forward wrote into). Columns 1..num_spec each get their own
    dedicated row in a separate address range (past every slot's column-0
    row), written unconditionally every verify round regardless of which
    candidate later turns out to be the real one -- a rejected
    candidate's row is simply never selected by any future round's
    ``num_accepted_tokens``-derived read, which is what eliminates the
    need for any explicit snapshot/restore or recompute-forward repair.
    """
    physical = _physical_slot(logical_slot)
    if col == 0:
        return physical
    return total_physical_slots + physical * num_spec + (col - 1)


def _derive_none_hash() -> int:
    """Process-global seed for the chained block hash (P3, R7). Derived once
    from a fixed salt + ``PYTHONHASHSEED`` so hashing is reproducible within
    a run but cross-run collisions are impossible (different seed -> different
    chain). The persistent cache is per-process/in-memory, so the seed only
    needs to be a fixed, well-mixed, non-zero value within one process."""
    salt = b"qwen-sm120-runtime/prefix-cache/NONE_HASH"
    seed = os.environ.get("PYTHONHASHSEED", "0").encode("utf-8")
    return int.from_bytes(hashlib.blake2b(salt + seed, digest_size=16).digest(), "big")


NONE_HASH: int = _derive_none_hash()


def hash_block_tokens(parent_hash: int | None, token_ids: list[int], extra_keys: tuple) -> int:
    """Full-width 128-bit chained block hash (``notes/prefix-cache-design.md``
    sec 3.2; vLLM ``hash_block_tokens`` shape, adapted not copied). Block i's
    hash depends on the whole prefix via the chained ``parent_hash`` plus the
    current block's ``token_ids`` (or the ``NONE_HASH`` seed for block 0), so two prompts
    diverging at any earlier token get different hashes for every block from
    the divergence on -- which is exactly why prefix lookup can stop at the
    first miss.

    ``extra_keys`` MUST carry the ``kv_cache_dtype`` (the runner passes
    ``(self.kv_cache_dtype,)``) so fp8 vs nvfp4 KV never collide (R7). Uses
    ``hashlib.blake2b(digest_size=16)`` -> a 128-bit int; collision probability
    is negligible at this runtime's <=4-slot traffic (R7)."""
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update((parent_hash or NONE_HASH).to_bytes(16, "big"))
    hasher.update(array.array("Q", token_ids).tobytes())
    if extra_keys:
        hasher.update(repr(extra_keys).encode("utf-8"))
    return int.from_bytes(hasher.digest(), "big")


@dataclass(frozen=True)
class BlockHash:
    """Content identity of one full published attention-KV block (P3).
    ``value`` is the 128-bit chained hash; ``num_tokens = (i+1)*block_size``
    enables the cheap paranoid first-block token-count verify on a hit/dedup
    (R7). Stored on ``Block.block_hash`` for every published block."""

    value: int
    num_tokens: int


@dataclass
class Block:
    """One physical attention-KV block. ``ref_cnt``/``block_hash`` are the
    same fields ``notes/prefix-cache-design.md`` sec 3.2 specifies for
    vLLM-v1-style block-pool bookkeeping. ``block_hash`` is populated by P3's
    persistent content-addressed cache (``BlockPool.cache_block``) for every
    published full block and RETAINED when the block is freed back to
    ``ref_cnt == 0`` -- that retention is exactly what keeps a cached prefix
    hit-able across ``reset_slot`` (R10).

    ``prev_free``/``next_free`` (P3.2) are the intrusive doubly-linked-list
    pointers ``FreeBlockQueue`` manipulates in O(1) (the design doc sec 3.2
    fields, vLLM ``KVCacheBlock.prev_free_block``/``next_free_block`` shape).
    Both are ``None`` exactly when the block is NOT parked in the free queue
    (allocated at ``ref_cnt > 0``, or a just-popped/removed block); a block in
    the queue has both non-``None``. The two ``FreeBlockQueue`` sentinels are
    the only blocks whose links point at a sentinel."""

    block_id: int
    ref_cnt: int = 0
    block_hash: BlockHash | None = None
    prev_free: Block | None = None
    next_free: Block | None = None


@dataclass
class ChunkedPrefillState:
    """A5/B4: Tracks an in-progress incremental chunked prefill.

    The engine advances this one chunk at a time via
    ``DirectModelRunner.prefill_chunked_step()``, interleaving decode
    rounds for active slots between chunks. This prevents long prefills
    (32K+ tokens) from starving active decode slots.
    """

    done: bool = False
    result: dict | None = None
    slots: list = None
    prompts_per_slot: list = None
    suffix_per_slot: list = None
    suffix_lens: list = None
    kv_offsets: list = None
    L_per_slot: list = None
    chunk_size: int = 512
    chunk_start: int = 0
    total_len: int = 0
    step0_logits: object = None
    step0_hidden: object = None
    anchors: dict = None
    # Optional per-slot token forcing used by reasoning-budget requests.  The
    # value is carried across incremental prefill steps so the first sampled
    # anchor can be constrained before speculative drafts are built from it.
    force_token_ids: dict[int, int] | None = None

    def __post_init__(self):
        if self.slots is None:
            self.slots = []
        if self.anchors is None:
            self.anchors = {}


class FreeBlockQueue:
    """Intrusive doubly-linked LRU free-block queue (P3.2; design doc sec 3.2,
    vLLM ``FreeKVCacheBlockQueue`` shape, adapted). Replaces P1's plain
    ``collections.deque`` to add O(1) MIDDLE removal -- needed to revive a
    ``ref_cnt == 0`` cached block on a late hit (``BlockPool.touch`` yanks it
    out before reuse) without an O(n) scan.

    Order semantics (vLLM ``free_blocks``/``get_new_blocks`` faithful):
    front (head side, ``popleft``) = evict-next; tail = most-recently-freed.
    ``append`` parks a block at the TAIL (a freed-but-HASHED block stays
    hit-able as long as possible -- LRU tail, evicted last); ``appendleft``
    parks it at the HEAD (a freed block with NO hash carries no cached content,
    so it is evicted FIRST). ``popleft`` returns the evict-next block;
    ``remove(block)`` unlinkes an arbitrary node in O(1). Two sentinel
    ``Block`` s (``block_id`` -1/-2, never popped) eliminate edge branching.

    Links live ON the ``Block`` objects (``prev_free``/``next_free``); the
    queue holds a back-reference to ``blocks`` only so ``__contains__`` can
    resolve a bare ``block_id`` (the P3.1 gate probes ``blk in pool._free_queue``
    with an int)."""

    def __init__(self, blocks: list[Block]) -> None:
        self._blocks = blocks
        self.head = Block(block_id=-1)  # sentinel: head.next = evict-next
        self.tail = Block(block_id=-2)  # sentinel: tail.prev = MRU
        self.head.next_free = self.tail
        self.tail.prev_free = self.head
        self._len = 0

    def __len__(self) -> int:
        return self._len

    def __contains__(self, item: Block | int) -> bool:
        block = item if isinstance(item, Block) else self._blocks[item]
        return block.prev_free is not None and block.next_free is not None

    def append(self, block: Block) -> None:
        """Park ``block`` at the TAIL (most-recently-freed / hashed LRU tail,
        evicted last). O(1)."""
        if block.prev_free is not None or block.next_free is not None:
            raise RuntimeError(f"block {block.block_id} is already linked in the free queue")
        last = self.tail.prev_free
        assert last is not None
        last.next_free = block
        block.prev_free = last
        block.next_free = self.tail
        self.tail.prev_free = block
        self._len += 1

    def appendleft(self, block: Block) -> None:
        """Park ``block`` at the HEAD (evict-next end; freed hashless blocks go
        here so they are reclaimed before any cached block). O(1)."""
        if block.prev_free is not None or block.next_free is not None:
            raise RuntimeError(f"block {block.block_id} is already linked in the free queue")
        first = self.head.next_free
        assert first is not None
        first.prev_free = block
        block.next_free = first
        block.prev_free = self.head
        self.head.next_free = block
        self._len += 1

    def popleft(self) -> Block:
        """Remove and return the evict-next (head) block. O(1)."""
        block = self.head.next_free
        if block is self.tail or block is None:
            raise RuntimeError("popleft from an empty free queue")
        self._unlink(block)
        return block

    def remove(self, block: Block) -> None:
        """Unlink an arbitrary node in O(1) (the late-hit revival primitive)."""
        if block.prev_free is None or block.next_free is None:
            raise RuntimeError(f"remove() on block {block.block_id} not in the free queue")
        self._unlink(block)

    def _unlink(self, block: Block) -> None:
        assert block.prev_free is not None and block.next_free is not None
        block.prev_free.next_free = block.next_free
        block.next_free.prev_free = block.prev_free
        block.prev_free = None
        block.next_free = None
        self._len -= 1


class BlockPool:
    """Dynamic free-list allocator + reference counting over the shared
    physical attention-KV block pool (``notes/prefix-cache-design.md`` sec
    5, "P1 -- Dynamic free-list allocator + reference counting").

    Replaces P0's ``_initial_block_table`` static per-slot partition (which
    just handed every logical slot the SAME contiguous
    ``[_physical_slot(slot) * blocks_per_slot, ...)`` range P0 was proven
    behavior-identical with the old arange addressing): blocks are now
    handed out from ONE shared free queue, on demand, to whichever slot
    asks next -- so a single slot's own block ids may end up non-contiguous
    over time (this is the intended, exercised behavior this phase proves
    safe, not a defect).

    **Free queue (P3.2)**: ``_free_queue`` is an intrusive ``FreeBlockQueue``
    (O(1) ``append``/``appendleft``/``popleft``/``remove``), replacing P1's
    plain ``collections.deque``. ``allocate`` pops from the front (evict-next);
    a popped block that still carries a published hash is EVICTED first -- its
    hash is dropped from ``hash_to_block`` and, in lockstep, the co-keyed GDN
    checkpoint is dropped via the ``_on_evict_block`` callback the runner wires
    after construction (INV2/INV3/R5). ``free`` re-queues a ``ref_cnt == 0``
    block at the TAIL if it keeps a hash (stays hit-able, LRU) or at the HEAD
    if hashless (no cached content, evicted first) -- vLLM ``free_blocks``
    order. ``touch`` yanks a revived ``ref_cnt == 0`` block out in O(1)
    (INV2/INV9). With a generously-sized pool and no pressure, ``allocate``
    never evicts a hashed block, so behavior is byte-for-byte P3.1.

    **INV7 (reserved physical slot 0)**: physical block ids
    ``[0, reserved)`` (``reserved=RESERVED_PHYSICAL_SLOTS`` by default,
    i.e. just block 0) are carved out at construction and NEVER enter the
    free queue via any path -- ``allocate``/``free`` both refuse to
    hand out or accept a reserved id, so a caller bug cannot leak block 0
    into real use, and neither can this class's own logic.
    """

    def __init__(self, num_blocks: int, reserved: int = RESERVED_PHYSICAL_SLOTS) -> None:
        if reserved < 1:
            raise ValueError("must reserve at least physical block 0 (INV7)")
        if num_blocks <= reserved:
            raise ValueError(f"num_blocks={num_blocks} must exceed reserved={reserved}")
        self.num_blocks = num_blocks
        self.reserved = reserved
        self.blocks: list[Block] = [Block(block_id=i) for i in range(num_blocks)]
        # Free queue: intrusive LRU (FreeBlockQueue). Front (``popleft()``) =
        # evict-next; tail (``append()``) = most-recently-freed. Excludes every
        # reserved id by construction -- reserved ids are never appended by
        # ``free`` either (guarded there too, defense in depth). Initial blocks
        # are linked in ASCENDING id order (append each), so the no-pressure
        # ``popleft`` order matches P1's ``deque(range(reserved, num_blocks))``
        # exactly (byte-for-byte P3.1 when eviction never triggers).
        self._free_queue: FreeBlockQueue = FreeBlockQueue(self.blocks)
        for _block_id in range(reserved, num_blocks):
            self._free_queue.append(self.blocks[_block_id])
        # P3.2 lockstep eviction hook: when ``_evict_one`` reclaims a still-
        # hashed attention block, the runner drops the co-keyed GDN checkpoint
        # too (INV3/R5, both directions). ``None`` for a stand-alone BlockPool
        # (pure-Python tests); the runner sets it to ``evict_gdn_checkpoint``
        # after construction. Kept as a plain callable to avoid a BlockPool ->
        # DirectModelRunner import cycle.
        self._on_evict_block: Callable[[int], None] | None = None
        # P3 persistent content-addressed index (notes/prefix-cache-design.md
        # sec 3.2): maps a full block's 128-bit chained hash VALUE to the
        # physical Block holding that prefix's KV. Populated by cache_block at
        # a cached prefix's completion boundary; probed by get_cached_block on
        # admission. A published block keeps its block_hash (and thus stays in
        # this index) even when freed back to ref_cnt == 0 -- that is exactly
        # what keeps a cached prefix hit-able across reset_slot (R10); the hash
        # is dropped only on real eviction (P3.2).
        self.hash_to_block: dict[int, Block] = {}

    def num_free_blocks(self) -> int:
        return len(self._free_queue)

    def _evict_one(self) -> Block:
        """Pop the evict-next (free-queue front) block and make it reusable
        (P3.2, design doc sec 3.2 ``_maybe_evict`` + sec 3.9 lockstep). If it
        still carries a published hash (a freed-but-cached block at
        ``ref_cnt == 0``), drop that hash from ``hash_to_block`` FIRST -- only
        then is the cached content gone (INV2: a block is never both indexed as
        old content AND handed out for new) -- and, in LOCKSTEP, drop the
        co-keyed GDN checkpoint via ``_on_evict_block`` (INV3/R5: an attention
        tail block and its checkpoint are always evicted together; no
        half-evicted ghost can produce a wrong ``L``). Returns the block at
        ``ref_cnt == 0`` / ``block_hash is None``, ready to hand out. Only
        reachable under real pool pressure (a generous pool never evicts a
        hashed block in the no-pressure path => byte-for-byte P3.1)."""
        block = self._free_queue.popleft()
        if block.block_hash is not None:
            self.hash_to_block.pop(block.block_hash.value, None)
            block.block_hash = None
            if self._on_evict_block is not None:
                self._on_evict_block(block.block_id)
        return block

    def allocate(self, n: int) -> list[int]:
        """Hand out ``n`` blocks from the free-queue front, setting
        ``ref_cnt = 1`` on each and returning their ids in pop order
        (evict-next-first). Every free-queue block is ``ref_cnt == 0`` by
        construction (``free`` re-queues only at 0; ``touch``/``allocate``
        remove on revive), so a popped block is always safe to reuse -- but a
        popped block may still be CACHED (hash retained at ``ref_cnt == 0``),
        in which case ``_evict_one`` evicts it (drops the hash + lockstep GDN
        checkpoint) before hand-out.

        Raises only on TRUE exhaustion: the free queue holds fewer than ``n``
        blocks, i.e. every other block is ``ref_cnt > 0`` (held by an active
        slot) and thus NEVER evictable (INV9). Callers
        (``DirectModelRunner._ensure_blocks``) size the pool generously (this
        project's ``num_blocks = (num_slots + RESERVED_PHYSICAL_SLOTS) *
        blocks_per_slot`` sizing, unchanged from P0), so eviction is rare and
        true exhaustion rarer still."""
        if n < 0:
            raise ValueError(f"cannot allocate a negative count ({n})")
        if n > len(self._free_queue):
            raise RuntimeError(
                f"block pool exhausted: requested {n}, only "
                f"{len(self._free_queue)} free (of {self.num_blocks - self.reserved} total, "
                f"excluding {self.reserved} reserved); every other block is ref_cnt > 0"
            )
        ids: list[int] = []
        for _ in range(n):
            block = self._evict_one()
            if block.block_id < self.reserved:
                raise RuntimeError(
                    f"INV7 violation: reserved block {block.block_id} was in the free queue"
                )
            if block.ref_cnt != 0:
                raise RuntimeError(
                    f"block {block.block_id} was in the free queue with "
                    f"ref_cnt={block.ref_cnt} != 0"
                )
            block.ref_cnt = 1
            ids.append(block.block_id)
        bfdiag_checks.check_no_duplicate_ids("block_pool.allocate", ids)
        return ids

    def free(self, block_ids: list[int]) -> None:
        """Release blocks back to the pool: ``ref_cnt -= 1``; at 0, re-queue
        (P3.2, vLLM ``free_blocks`` order): a block that KEEPS a published
        hash is ``append``-ed to the TAIL (stays hit-able, LRU -- evicted last
        among free blocks); a block with NO hash carries no cached content so
        it is ``appendleft``-ed to the HEAD (evicted FIRST, before any cached
        block). Written as a decrement (not a hard reset to 0) so a caller
        legitimately holding ``ref_cnt > 1`` (P2 fan-out / P3 hit sharing) is
        not silently masked -- only the final release re-queues."""
        for block_id in block_ids:
            if block_id < self.reserved:
                raise RuntimeError(f"INV7 violation: attempted to free reserved block {block_id}")
            if block_id >= self.num_blocks:
                raise RuntimeError(
                    f"block {block_id} is out of range (num_blocks={self.num_blocks})"
                )
            block = self.blocks[block_id]
            if block.ref_cnt <= 0:
                raise RuntimeError(f"double-free of block {block_id} (ref_cnt={block.ref_cnt})")
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                if block.block_hash is not None:
                    self._free_queue.append(block)
                else:
                    self._free_queue.appendleft(block)

    def reference(self, block_ids: list[int]) -> None:
        """Add a reference to already-allocated blocks (``ref_cnt += 1`` on
        each), the P2 fan-out primitive that lets a sibling slot share the
        leader's ``[0, Lc)`` attention blocks instead of recomputing them
        (``notes/prefix-cache-design.md`` sec 5, "P2 -- Fan-out fork", and
        sec 3.5 step 1: "touch each (ref_cnt += 1) for all 17 attention
        layers' shared namespace"). The 17 attention layers (16 target + 1
        MTP draft) share ONE block-id namespace, so referencing the block
        ids once covers every layer's KV for those token positions.

        Mirror-image of ``free``'s bookkeeping (which decrements and only
        re-queues at ``ref_cnt == 0``): a referenced block stays out of the
        free queue until EVERY referencer has freed it, so a shared block is
        never handed to ``allocate`` while any slot still references it
        (INV9). Carries the SAME INV7/range guards as ``allocate``/``free``
        (refuse reserved ids ``< reserved`` and out-of-range ids), plus a
        not-currently-allocated guard -- referencing a ``ref_cnt == 0``
        block would resurrect a block that may already be back in the free
        queue (and thus handed to another slot), an immediate aliasing bug.
        """
        for block_id in block_ids:
            if block_id < self.reserved:
                raise RuntimeError(
                    f"INV7 violation: attempted to reference reserved block {block_id}"
                )
            if block_id >= self.num_blocks:
                raise RuntimeError(
                    f"block {block_id} is out of range (num_blocks={self.num_blocks})"
                )
            block = self.blocks[block_id]
            if block.ref_cnt <= 0:
                try:
                    self._free_queue.remove(block)
                except ValueError:
                    raise RuntimeError(
                        f"cannot reference block {block_id} with ref_cnt={block.ref_cnt} "
                        "(not currently allocated -- it may already be back in the free queue)"
                    )
            block.ref_cnt += 1

    def cache_block(self, block_id: int, block_hash: BlockHash) -> None:
        # Publish a full block to the content index (P3, sec 3.2): tag
        # blocks[block_id].block_hash and record hash_to_block[value].
        # Idempotent guard: if value is ALREADY present this is the write-time
        # dedup signal (see DirectModelRunner._publish_committed_blocks) -- the
        # existing entry is the canonical block for that prefix and is NOT
        # overwritten (the caller swaps onto it and frees the duplicate fresh
        # block instead).
        if block_id < self.reserved:
            raise RuntimeError(f"INV7 violation: attempted to publish reserved block {block_id}")
        if block_id >= self.num_blocks:
            raise RuntimeError(f"block {block_id} is out of range (num_blocks={self.num_blocks})")
        if block_hash.value in self.hash_to_block:
            return
        self.blocks[block_id].block_hash = block_hash
        self.hash_to_block[block_hash.value] = self.blocks[block_id]

    def get_cached_block(self, hash_value: int) -> Block | None:
        # Content-index probe (P3, sec 3.4 attention match): return the
        # published Block whose chained hash equals hash_value, or None on a
        # miss. O(1) dict lookup.
        return self.hash_to_block.get(hash_value)

    def touch(self, block_ids: list[int]) -> None:
        # Cache-hit reference primitive (P3, sec 3.2/3.5 step 1): ref_cnt += 1
        # on each block, REVIVING a block parked at ref_cnt == 0 in the free
        # queue by removing it first. Mirror of reference() but LEGAL for
        # ref_cnt == 0 -- reference() stays the same-round-fork primitive (it
        # rejects ref_cnt == 0), touch() is the persistent-cache hit primitive
        # (a published block freed back to ref_cnt == 0 by reset_slot is still
        # hit-able and must be yanked out of the free queue before reuse --
        # INV2/INV9). P3.2: the removal is O(1) via the intrusive FreeBlockQueue
        # (replacing P3.1's O(n) deque remove).
        for block_id in block_ids:
            if block_id < self.reserved:
                raise RuntimeError(f"INV7 violation: attempted to touch reserved block {block_id}")
            if block_id >= self.num_blocks:
                raise RuntimeError(
                    f"block {block_id} is out of range (num_blocks={self.num_blocks})"
                )
            block = self.blocks[block_id]
            if block.ref_cnt == 0:
                # Parked in the free queue (a freed-but-published block):
                # revive it so allocate never hands it to another slot while
                # this hit references it.
                self._free_queue.remove(block)
            block.ref_cnt += 1
