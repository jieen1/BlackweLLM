"""Correctness gate for prefix-cache P1 -- dynamic free-list allocator +
reference counting (``notes/prefix-cache-design.md`` sec 5, "P1 -- Dynamic
free-list allocator + reference counting").

P1 replaces P0's ``_initial_block_table`` static per-slot partition (every
logical slot pre-populated with its own fixed contiguous ``blocks_per_slot``
range, byte-identical to the old arange addressing) with a real
``BlockPool``: one shared free queue + ``ref_cnt`` bookkeeping over every
physical block except reserved block 0 (INV7). Slots now grow their own
``block_table[slot]`` ON DEMAND (``DirectModelRunner._ensure_blocks``,
called from every attention-metadata/slot-mapping/CUDA-graph-fill call
site) and release blocks back to the pool on ``reset_slot``/
``_finish_request`` -- still with **no cross-slot sharing** this phase
(every block has exactly one referencer), so end-to-end request/response
*behavior* stays identical to P0; only physical block *placement* becomes
genuinely dynamic (a single slot's own blocks may end up non-contiguous
over time). This is the design doc's own explicit purpose for this phase:
prove the block-table + CUDA-graph path tolerates non-contiguous block
ids, not just P0's trivial contiguous case.

This script is pure Python -- **no GPU, no model load** -- both because
the design doc scopes this dedicated test to the ALLOCATOR's own
invariants (real end-to-end model/logits correctness under P1's dynamic
addressing is covered elsewhere: ``prefix_cache_block_table_check.py``'s
numeric-equivalence re-run, the two fragmented-CUDA-graph re-runs of
``cudagraph_decode_regression.py``/``mtp_verify_cudagraph_check.py``, and
the full ``mtp_*_check`` battery + 4K/c=4 headline -- see
``notes/prefix-cache-implementation-log.md``'s P1 section for that
evidence), and because ``BlockPool.allocate``/``free`` and
``DirectModelRunner._ensure_blocks``/``reset_slot``'s block-freeing logic
are pure bookkeeping with zero GPU/tensor operations. ``_ensure_blocks``/
``reset_slot`` are exercised here as the REAL, unmodified, production
bound methods (called against a lightweight duck-typed stand-in object
that only defines the handful of attributes those two methods actually
touch) -- not a reimplementation of their logic -- so this is a genuine
test of the production code path, just isolated from the (expensive,
GPU-bound) model/tensor machinery those methods' CALLERS also touch.

Checks, each mapped to the design doc's own P1 test list and correctness
invariants (sec 4):

1. ``_check_alloc_free_correctness`` -- ``BlockPool.allocate``/``free``
   round-trip: ids returned are distinct, in range, excluding reserved;
   free returns them to the pool; over-allocation raises; double-free
   raises.
2. ``_check_ref_cnt_bookkeeping`` -- ``Block.ref_cnt`` goes 0->1 on
   allocate, 1->0 on free, and a block can be correctly re-allocated
   (0->1 again) after being freed once.
3. ``_check_free_queue_ordering`` -- P3.2 LRU eviction ordering of the
   intrusive ``FreeBlockQueue`` (vLLM ``free_blocks`` order): hashless
   freed blocks are prepended to the evict-next front (reclaimed before
   any cached block, newest-free-call first); hashed (cached) freed
   blocks are appended to the LRU tail (oldest-freed-first among them);
   hashless are always evicted before hashed. (P1's plain-FIFO deque
   behavior was intentionally superseded by this eviction policy in
   P3.2; see ``BlockPool``/``FreeBlockQueue`` docstrings.)
4. ``_check_inv7_reserved_block_zero`` -- block 0 (and, generally, every
   id ``< reserved``) is never handed out by ``allocate`` and never
   accepted by ``free``, across many allocate/free cycles, and the pool
   refuses to be constructed with ``reserved < 1``.
5. ``_check_append_only_growth`` (INV6) -- ``DirectModelRunner
   ._ensure_blocks`` only ever APPENDS to ``block_table[slot]``: an
   already-assigned logical position's physical block id never changes
   across repeated/idempotent/growing calls, and the capacity check
   (``> blocks_per_slot`` pages) still raises before ever touching the
   pool.
6. ``_check_reset_slot_frees_blocks`` -- the real ``DirectModelRunner
   .reset_slot`` releases every block a slot held back to the pool
   (``ref_cnt`` back to 0, re-enters the free queue) and clears
   ``block_table[slot]`` to ``[]`` -- the design doc's R10 risk-register
   entry this phase is required to close.
7. ``_check_fragmentation_after_churn`` -- the actual scenario P1 exists
   to prove safe: after an explicit allocate/reset/reallocate churn cycle
   (one slot's blocks freed and reallocated interleaved with another
   slot's own growth), a single slot's ``block_table`` entry is
   PROVABLY non-contiguous (not a contiguous arithmetic run) -- directly
   asserted here, not just claimed. (The model-level, CUDA-graph-replay
   proof that this fragmented state still produces correct output is the
   job of the two dedicated fragmented-CUDA-graph re-runs named in the
   design doc's P1 test list, not this script.)

Usage:
    python -m benchmarks.prefix_cache_allocator_check
"""

from __future__ import annotations

import argparse
import sys


class _StubRunner:
    """Duck-typed stand-in exposing only the attributes
    ``DirectModelRunner._ensure_blocks``/``.reset_slot`` actually touch --
    NOT a reimplementation of either method (both are called as real,
    unbound ``DirectModelRunner`` methods against this object below)."""

    def __init__(self, num_slots: int, block_size: int, blocks_per_slot: int, pool) -> None:
        self.block_size = block_size
        self.blocks_per_slot = blocks_per_slot
        self.block_table: list[list[int]] = [[] for _ in range(num_slots)]
        self.block_pool = pool
        self.slot_kv_len = [0] * num_slots
        self.slot_gdn_initialized = [False] * num_slots
        self.slot_draft_sync_len = [0] * num_slots
        self.slot_pending_draft_tokens: list[list[int] | None] = [None] * num_slots
        self.slot_num_accepted_tokens = [1] * num_slots
        # P3.1 (notes/2026-07-19-p3-implementation-plan.md step 4): reset_slot
        # now also clears the per-slot hash-chain cursor/count. Built
        # unconditionally in the real __init__, so the stub must expose them
        # too (this stub mirrors exactly what reset_slot touches).
        self.slot_block_hashes: list[list] = [[] for _ in range(num_slots)]
        self.slot_published_blocks: list[int] = [0] * num_slots
        # P3.2 decode-position populate: reset_slot now also clears the per-slot
        # committed-token record. Built unconditionally in the real __init__, so
        # the stub mirrors it (this stub mirrors exactly what reset_slot touches).
        self.slot_committed_tokens: list[list[int]] = [[] for _ in range(num_slots)]


def _check_alloc_free_correctness() -> dict:
    from oracle.qwen36_vllm.direct_model_runner import BlockPool

    errors = []
    pool = BlockPool(num_blocks=20, reserved=1)
    if pool.num_free_blocks() != 19:
        errors.append(f"expected 19 free blocks initially, got {pool.num_free_blocks()}")

    ids = pool.allocate(5)
    if len(set(ids)) != 5:
        errors.append(f"allocate(5) returned non-distinct ids: {ids}")
    if any(i < pool.reserved or i >= pool.num_blocks for i in ids):
        errors.append(f"allocate(5) returned out-of-range id(s): {ids}")
    if pool.num_free_blocks() != 14:
        errors.append(f"expected 14 free blocks after allocate(5), got {pool.num_free_blocks()}")

    pool.free(ids)
    if pool.num_free_blocks() != 19:
        errors.append(f"expected 19 free blocks after freeing all 5 back, got {pool.num_free_blocks()}")

    # Over-allocation must raise, not silently return a short/duplicate list.
    raised = False
    try:
        pool.allocate(20)
    except RuntimeError:
        raised = True
    if not raised:
        errors.append("allocate(20) on a 19-free pool did not raise")

    # Double-free must raise.
    ids2 = pool.allocate(2)
    pool.free(ids2)
    raised = False
    try:
        pool.free(ids2)
    except RuntimeError:
        raised = True
    if not raised:
        errors.append("double-free did not raise")

    # allocate(0) is a legitimate no-op (a slot needing zero NEW blocks).
    if pool.allocate(0) != []:
        errors.append("allocate(0) did not return an empty list")

    return {"passed": not errors, "errors": errors}


def _check_ref_cnt_bookkeeping() -> dict:
    from oracle.qwen36_vllm.direct_model_runner import BlockPool

    errors = []
    pool = BlockPool(num_blocks=10, reserved=1)
    ids = pool.allocate(3)
    for i in ids:
        if pool.blocks[i].ref_cnt != 1:
            errors.append(f"block {i} ref_cnt={pool.blocks[i].ref_cnt} != 1 right after allocate")

    pool.free([ids[0]])
    if pool.blocks[ids[0]].ref_cnt != 0:
        errors.append(f"block {ids[0]} ref_cnt={pool.blocks[ids[0]].ref_cnt} != 0 after free")
    if pool.blocks[ids[1]].ref_cnt != 1 or pool.blocks[ids[2]].ref_cnt != 1:
        errors.append("freeing one block changed another still-held block's ref_cnt")

    # Re-allocate the whole pool and confirm the previously-freed block
    # (ref_cnt 0) can correctly become ref_cnt 1 again (multiple full
    # alloc/free cycles must not leave ref_cnt drifting).
    pool.free([ids[1], ids[2]])
    all_ids = pool.allocate(9)
    for i in all_ids:
        if pool.blocks[i].ref_cnt != 1:
            errors.append(f"block {i} ref_cnt={pool.blocks[i].ref_cnt} != 1 on second full allocate")
    pool.free(all_ids)
    for i in all_ids:
        if pool.blocks[i].ref_cnt != 0:
            errors.append(f"block {i} ref_cnt={pool.blocks[i].ref_cnt} != 0 after second free cycle")

    return {"passed": not errors, "errors": errors}


def _check_free_queue_ordering() -> dict:
    from oracle.qwen36_vllm.direct_model_runner import BlockHash, BlockPool, hash_block_tokens

    errors = []
    extra = ("fp8_e4m3",)

    # (a) HASHLESS freed blocks are prepended to the evict-next front: freed in
    #     one call [7,2,9,0], each appendleft jumps ahead of the previous, so
    #     popleft returns them newest-in-call-first (the reverse) -- they are
    #     reclaimed before any cached block (vLLM free_blocks order).
    pool = BlockPool(num_blocks=11, reserved=1)  # ids 1..10
    all_ids = pool.allocate(10)
    if pool.num_free_blocks() != 0:
        errors.append("pool not fully drained after allocating every block")
    free_order = [all_ids[7], all_ids[2], all_ids[9], all_ids[0]]
    pool.free(free_order)  # all hashless => appendleft each => front reversed
    realloc = pool.allocate(4)
    expected_hashless = list(reversed(free_order))
    if realloc != expected_hashless:
        errors.append(
            f"hashless prepend ordering violated: freed {free_order}, "
            f"expected {expected_hashless}, got {realloc}"
        )

    # (b) HASHLESS blocks are evicted BEFORE hashed (cached) blocks. Cache a
    #     pair (append => LRU tail) and free a hashless pair (appendleft =>
    #     head); the next two allocations must be the hashless pair first.
    pool2 = BlockPool(num_blocks=5, reserved=1)  # ids 1..4 (drained exactly)
    h_ids = pool2.allocate(2)            # become cached (hashed)
    plain_ids = pool2.allocate(2)        # stay hashless; pool now fully drained
    for i, bid in enumerate(h_ids):
        pool2.cache_block(bid, BlockHash(hash_block_tokens(None, [i + 1] * 16, extra), 16))
    pool2.free(h_ids)      # hashed  => append (tail)
    pool2.free(plain_ids)  # hashless => appendleft (head, evicted first)
    first_two = pool2.allocate(2)
    if first_two != list(reversed(plain_ids)):
        errors.append(
            f"hashless not evicted before hashed: got {first_two}, "
            f"expected {list(reversed(plain_ids))}"
        )
    # The two hashed blocks remain free (tail) and still cached...
    if pool2.num_free_blocks() != 2:
        errors.append(f"expected 2 hashed blocks still free, got {pool2.num_free_blocks()}")
    # ...and allocating them DROPS their hashes (INV2 eviction-before-hand-out).
    last_two = pool2.allocate(2)
    if set(last_two) != set(h_ids):
        errors.append(f"hashed blocks not returned after hashless: got {last_two}")
    for bid in h_ids:
        if pool2.blocks[bid].block_hash is not None:
            errors.append(f"evicted hashed block {bid} kept its hash (INV2)")

    # (c) Among HASHED blocks, oldest-freed-first (LRU tail order): a freed
    #     before b => a closer to the head => evicted first.
    pool3 = BlockPool(num_blocks=3, reserved=1)  # ids 1,2
    a, b = pool3.allocate(2)  # drains the pool
    pool3.cache_block(a, BlockHash(hash_block_tokens(None, [1] * 16, extra), 16))
    pool3.cache_block(b, BlockHash(hash_block_tokens(None, [2] * 16, extra), 16))
    pool3.free([a])  # hashed => append first  (older, closer to head)
    pool3.free([b])  # hashed => append second (newer, closer to tail)
    order = pool3.allocate(2)
    if order != [a, b]:
        errors.append(f"hashed LRU ordering violated: expected [{a}, {b}], got {order}")

    return {"passed": not errors, "errors": errors}


def _check_inv7_reserved_block_zero() -> dict:
    from oracle.qwen36_vllm.direct_model_runner import BlockPool

    errors = []

    # Construction must refuse reserved < 1.
    raised = False
    try:
        BlockPool(num_blocks=10, reserved=0)
    except ValueError:
        raised = True
    if not raised:
        errors.append("BlockPool(reserved=0) did not raise -- block 0 must always be excluded")

    pool = BlockPool(num_blocks=6, reserved=1)  # ids 0..5, 0 reserved
    seen_ids: set[int] = set()
    # Exhaust and recycle the pool several times over -- block 0 must NEVER
    # appear among allocated ids, across every cycle.
    for _cycle in range(4):
        ids = pool.allocate(pool.num_free_blocks())
        seen_ids.update(ids)
        pool.free(ids)
    if 0 in seen_ids:
        errors.append(f"block 0 was handed out by allocate() at some point: {sorted(seen_ids)}")
    if any(i < 1 for i in seen_ids):
        errors.append(f"a reserved id < 1 was handed out: {sorted(seen_ids)}")

    # free() must reject block 0 even if a caller mistakenly tries.
    raised = False
    try:
        pool.free([0])
    except RuntimeError:
        raised = True
    if not raised:
        errors.append("free([0]) did not raise -- reserved block 0 must never re-enter the pool")

    # Multi-reserved-slot configuration (defensive: RESERVED_PHYSICAL_SLOTS
    # is currently 1 in production, but the class itself must honor
    # whatever `reserved` it's given).
    pool2 = BlockPool(num_blocks=10, reserved=3)
    ids2 = pool2.allocate(7)
    if any(i < 3 for i in ids2):
        errors.append(f"reserved=3 pool handed out an id < 3: {ids2}")

    return {"passed": not errors, "errors": errors}


def _check_append_only_growth() -> dict:
    from oracle.qwen36_vllm.direct_model_runner import BlockPool, DirectModelRunner

    errors = []
    pool = BlockPool(num_blocks=50, reserved=1)
    runner = _StubRunner(num_slots=2, block_size=64, blocks_per_slot=1, pool=pool)  # capacity 64 tokens/slot

    DirectModelRunner._ensure_blocks(runner, 0, 5)  # needs 1 page
    if len(runner.block_table[0]) != 1:
        errors.append(f"expected 1 block after _ensure_blocks(0, 5), got {runner.block_table[0]}")
    first_block = runner.block_table[0][0]

    # Idempotent: same kv_len_needed must not grow or reorder anything.
    DirectModelRunner._ensure_blocks(runner, 0, 5)
    if runner.block_table[0] != [first_block]:
        errors.append(f"idempotent _ensure_blocks call mutated block_table[0]: {runner.block_table[0]}")

    # Growth must be a pure APPEND -- position 0's physical id must never
    # change once assigned (INV6: "a logical position's physical block
    # never changes mid-life").
    DirectModelRunner._ensure_blocks(runner, 0, 20)  # needs 2 pages
    if runner.block_table[0][0] != first_block:
        errors.append(
            f"position 0's block id changed on growth: was {first_block}, now {runner.block_table[0][0]}"
        )
    if len(runner.block_table[0]) != 2:
        errors.append(f"expected 2 blocks after _ensure_blocks(0, 20), got {runner.block_table[0]}")
    second_block = runner.block_table[0][1]

    DirectModelRunner._ensure_blocks(runner, 0, 33)  # needs 3 pages (ceil(33/16))
    if runner.block_table[0][:2] != [first_block, second_block]:
        errors.append(f"earlier positions mutated on further growth: {runner.block_table[0]}")
    if len(runner.block_table[0]) != 3:
        errors.append(f"expected 3 blocks after _ensure_blocks(0, 33), got {runner.block_table[0]}")

    # Per-slot capacity ceiling (blocks_per_slot=1 -> 64 tokens) must still
    # be enforced, and must raise BEFORE consuming a block from the pool.
    free_before = pool.num_free_blocks()
    raised = False
    try:
        DirectModelRunner._ensure_blocks(runner, 0, 65)
    except RuntimeError:
        raised = True
    if not raised:
        errors.append("_ensure_blocks did not raise when exceeding blocks_per_slot capacity")
    if pool.num_free_blocks() != free_before:
        errors.append("a capacity-exceeding _ensure_blocks call consumed a block before raising")

    return {"passed": not errors, "errors": errors}


def _check_reset_slot_frees_blocks() -> dict:
    from oracle.qwen36_vllm.direct_model_runner import BlockPool, DirectModelRunner

    errors = []
    pool = BlockPool(num_blocks=50, reserved=1)
    runner = _StubRunner(num_slots=2, block_size=64, blocks_per_slot=1, pool=pool)

    DirectModelRunner._ensure_blocks(runner, 0, 40)  # 3 pages
    held = list(runner.block_table[0])
    if len(held) != 3:
        errors.append(f"setup: expected 3 blocks held, got {held}")
    for i in held:
        if pool.blocks[i].ref_cnt != 1:
            errors.append(f"setup: block {i} ref_cnt != 1 before reset_slot")

    free_before = pool.num_free_blocks()
    DirectModelRunner.reset_slot(runner, 0)

    if runner.block_table[0] != []:
        errors.append(f"reset_slot did not clear block_table[0]: {runner.block_table[0]}")
    if pool.num_free_blocks() != free_before + len(held):
        errors.append(
            f"reset_slot did not return all {len(held)} blocks to the pool: "
            f"free count {free_before} -> {pool.num_free_blocks()}"
        )
    for i in held:
        if pool.blocks[i].ref_cnt != 0:
            errors.append(f"block {i} ref_cnt={pool.blocks[i].ref_cnt} != 0 after reset_slot")

    # reset_slot on a slot that never held any blocks (block_table[slot] ==
    # []) must be a safe no-op, not an error -- the real scenario when
    # enable_block_table was False for that slot's whole lifetime.
    try:
        DirectModelRunner.reset_slot(runner, 1)
    except Exception as e:  # noqa: BLE001
        errors.append(f"reset_slot on a never-grown slot raised: {e!r}")

    # Also confirm the OTHER per-slot bookkeeping reset_slot is responsible
    # for still happens (not a P1 regression of P0-era behavior).
    if runner.slot_kv_len[0] != 0 or runner.slot_gdn_initialized[0] is not False:
        errors.append("reset_slot no longer resets slot_kv_len/slot_gdn_initialized")

    return {"passed": not errors, "errors": errors}


def _check_fragmentation_after_churn() -> dict:
    from oracle.qwen36_vllm.direct_model_runner import BlockPool, DirectModelRunner

    errors = []
    details: dict[str, object] = {}
    pool = BlockPool(num_blocks=50, reserved=1)
    # slot 0 = the slot under test, slot 1 = a concurrent slot whose own
    # allocation pressure forces slot 0's block_table to be non-contiguous --
    # the real fragmentation scenario this phase exists to prove safe.
    runner = _StubRunner(num_slots=2, block_size=64, blocks_per_slot=2, pool=pool)  # capacity 128 tokens/slot

    # Step 1: slot 0 grows by one block (its first physical id).
    DirectModelRunner._ensure_blocks(runner, 0, 10)
    # Step 2: slot 1 grows by SEVERAL blocks and HOLDS them. Under P3.2's
    # appendleft-for-hashless LRU policy a freed hashless block is prepended to
    # the evict-next front (so plain allocate/free churn just recycles the same
    # low id back to the head -- it no longer rotates the queue the way P1's
    # FIFO deque did). The realistic way a gap therefore forms is a concurrent
    # slot HOLDING the ids adjacent to slot 0's first block, removing them from
    # the free queue so the head advances past them.
    DirectModelRunner._ensure_blocks(runner, 1, 40)  # 3 blocks, held by slot 1
    # Step 3: slot 0 grows again -- the free-queue head is now past slot 1's
    # held blocks, so slot 0's second block is non-adjacent to its first.
    DirectModelRunner._ensure_blocks(runner, 0, 25)
    # Step 4: slot 1 grows further (holds more), then slot 0 grows a third time
    # -- again drawing from past the held range, staying non-contiguous.
    DirectModelRunner._ensure_blocks(runner, 1, 57)
    DirectModelRunner._ensure_blocks(runner, 0, 40)

    table0 = runner.block_table[0]
    details["slot0_block_table"] = table0
    if len(table0) != 3:
        errors.append(f"expected slot 0 to hold 3 blocks after this churn recipe, got {table0}")

    is_contiguous = all(table0[i + 1] - table0[i] == 1 for i in range(len(table0) - 1))
    if is_contiguous:
        errors.append(
            f"slot 0's block_table {table0} is a contiguous run -- the churn recipe "
            "failed to produce the non-contiguous placement this phase must tolerate"
        )

    # INV6 sanity even under fragmentation: still append-only / no
    # duplicate ids / no reserved id 0 present.
    if len(set(table0)) != len(table0):
        errors.append(f"slot 0's block_table contains duplicate ids: {table0}")
    if 0 in table0:
        errors.append("slot 0's block_table contains reserved block 0")

    return {"passed": not errors, "errors": errors, "details": details}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    checks = [
        ("alloc_free_correctness", _check_alloc_free_correctness),
        ("ref_cnt_bookkeeping", _check_ref_cnt_bookkeeping),
        ("free_queue_ordering", _check_free_queue_ordering),
        ("inv7_reserved_block_zero", _check_inv7_reserved_block_zero),
        ("append_only_growth_inv6", _check_append_only_growth),
        ("reset_slot_frees_blocks", _check_reset_slot_frees_blocks),
        ("fragmentation_after_churn", _check_fragmentation_after_churn),
    ]

    overall = True
    for name, fn in checks:
        result = fn()
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{name}: {status}")
        if not result["passed"]:
            overall = False
            for err in result["errors"]:
                print(f"  - {err}")
        if result.get("details"):
            print(f"  details: {result['details']}")

    print(f"\n=== overall: {'PASS' if overall else 'FAIL'} ===")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
