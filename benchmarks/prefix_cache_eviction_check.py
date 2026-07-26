"""Correctness gate for prefix-cache P3.2 -- eviction, lockstep GDN eviction,
and the full populate path (``notes/2026-07-19-p3-implementation-plan.md``,
"### P3.2 -- Eviction, lockstep GDN eviction, and the full populate path";
``notes/prefix-cache-design.md`` sec 3.2/3.3/3.9, sec 4 invariants
INV2/INV3/INV9, sec 6 risks R4/R5/R7/R8).

P3.2 is the production-hardening round: the plain FIFO free deque becomes an
intrusive O(1) LRU ``FreeBlockQueue``; ``allocate`` evicts from the front
(dropping a popped cached block's hash BEFORE re-hand-out, INV2) and evicts the
co-keyed GDN checkpoint in LOCKSTEP (INV3/R5, both directions); a `ref_cnt > 0`
block is NEVER evicted (INV9/R4); the GDN checkpoint pool is capped by a real
byte-budget LRU (R8); decode-position populate publishes newly-full committed
blocks; and chunk-boundary GDN checkpoints give 8192-granular partial sharing.

Rollback-safe boundary: eviction only triggers when ``allocate`` cannot satisfy
from the free queue, so with no pressure behavior is byte-for-byte P3.1; the
populate paths only ADD index/checkpoint state under the flag and never change
produced tokens. ``enable_persistent_prefix_cache=False`` => byte-for-byte P2.

Checks (pure-Python + real-GPU), mirroring ``prefix_cache_persistent_hit_check
.py`` methodology: same-path token comparisons are exact (identical machinery =>
bit-identical tokens); the ONE cross-path case (``chunk_boundary_partial_share``,
hit continue-prefill vs a cold full prefill) proves INV1 R1-style instead -- the
restored @L GDN state is bytewise a true-cold compute of B[0,L) and the
continue-prefill anchor matches cold exactly -- because cross-path DECODE tokens
differ by fp8 split-KV non-associativity (the near-tie regime
``NEAR_TIE_LOGIT_MARGIN = 2.0``, R6) even when the restored state is correct:

Pure Python:
* ``lru_middle_removal`` -- intrusive queue O(1) remove of an arbitrary node;
  ``popleft`` order = oldest-appended-first (the LRU tail order).
* ``evict_drops_hash`` -- a hashed block popped for reuse loses its
  ``hash_to_block`` entry (INV2).
* ``lockstep_eviction`` -- evicting an attention tail block drops the co-keyed
  GDN checkpoint AND vice-versa; BOTH halves go together (INV3/R5).
* ``refcnt_never_evicted`` -- a ``ref_cnt > 0`` block is never popped, even as
  the LRU front (INV9).
* ``byte_budget`` -- materializing past the budget evicts the LRU checkpoint;
  pool bytes stay <= budget (R8).

Real GPU (one ``DirectModelRunner``, ``enable_block_table=True`` +
``enable_prefix_cache=True`` + ``enable_persistent_prefix_cache=True``, MTP
K=3):
* ``evict_then_recompute`` -- populate P, sponge the pool until P's hash is
  evicted, re-request P => clean COLD recompute (``L=0``), tokens match the
  cold reference (no half-evicted ghost hit; INV1/INV3).
* ``admission_under_pressure`` -- fill the pool so a new admission must evict
  while other slots are ACTIVE; assert no active ``ref_cnt > 0`` block is
  reclaimed and active committed tokens stay correct (INV9/R4).
* ``chunk_boundary_partial_share`` -- A = 20000-token cold prefill (chunked at
  8192 => checkpoints at 8192/16384); B shares A's first 18000 then diverges =>
  B hits at ``L=16384`` (deepest chunk boundary <= attention match) and reuses
  exactly that cached checkpoint (INV3); INV1 proved R1-style: B's restored @L
  GDN state is bytewise a true-cold chunked compute of B[0,L) (layer-0 exact +
  full-stack near-tie) and the continue-prefill anchor matches a true-cold full
  prefill of B exactly (INV1).
* ``a_gt_0_g_eq_0_dedup`` -- the compute-miss case (attention cached, no GDN
  checkpoint): reconcile returns ``L=0``; the fresh recompute's write-time
  dedup reclaims the duplicate attention blocks (free count recovers; no leak);
  tokens correct.
* ``no_leak_churn`` -- many admit/finish cycles under pressure; free count
  returns to baseline; ``cuda_allocated_mib`` stays flat (no leak; R10/R8).

Usage:
    python -m benchmarks.prefix_cache_eviction_check
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3
NUM_ROUNDS = 12
SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"}

NEAR_TIE_LOGIT_MARGIN = 2.0  # established methodology (R6).


# ---------------------------------------------------------------------------
# Pure-Python part (no GPU, no model load).
# ---------------------------------------------------------------------------


def _check_lru_middle_removal() -> dict:
    from runtime.direct_model_runner import BlockPool

    errors = []
    # Build a queue in a known order: drain the construction queue, then append
    # blocks 1..7 to the TAIL so popleft order = ascending (oldest-appended-first).
    pool2 = BlockPool(num_blocks=8, reserved=1)
    blocks = [pool2.blocks[i] for i in range(1, 8)]
    pool2.allocate(7)
    for b in blocks:
        pool2._free_queue.append(b)
    if len(pool2._free_queue) != 7:
        errors.append(f"queue length {len(pool2._free_queue)} != 7 after appends")
    # O(1) remove of an ARBITRARY middle node (blocks[3]).
    pool2._free_queue.remove(blocks[3])
    if len(pool2._free_queue) != 6:
        errors.append(f"queue length {len(pool2._free_queue)} != 6 after middle remove")
    if blocks[3] in pool2._free_queue:
        errors.append("removed middle block still reports as in the queue")
    # popleft order = oldest-appended-first, skipping the removed node.
    popped = [pool2._free_queue.popleft().block_id for _ in range(6)]
    expected = [1, 2, 3, 5, 6, 7]  # blocks[3] is block_id 4, removed
    if popped != expected:
        errors.append(f"popleft order {popped} != expected {expected} (oldest-first, middle gone)")
    # The removed node's links are cleared (re-addable without corruption).
    if blocks[3].prev_free is not None or blocks[3].next_free is not None:
        errors.append("removed node still has dangling free-list links")
    pool2._free_queue.append(blocks[3])
    if blocks[3] not in pool2._free_queue:
        errors.append("re-appended removed node is not back in the queue")
    return {"passed": not errors, "errors": errors}


def _check_evict_drops_hash() -> dict:
    from runtime.direct_model_runner import BlockHash, BlockPool, hash_block_tokens

    errors = []
    extra = ("fp8_e4m3",)
    pool = BlockPool(num_blocks=4, reserved=1)  # ids 1..3
    h = hash_block_tokens(None, list(range(16)), extra)
    b = pool.allocate(1)[0]
    pool.cache_block(b, BlockHash(h, 16))
    pool.free([b])  # ref_cnt 0, in queue (hashed => tail), hash retained
    if pool.get_cached_block(h) is None:
        errors.append("freed published block lost its hash before eviction")
    # Exhaust the pool so the hashed block is popped for reuse => evicted.
    pool.allocate(3)
    if pool.get_cached_block(h) is not None:
        errors.append("allocate did not drop the hash of a popped hashed block (INV2)")
    if pool.blocks[b].block_hash is not None:
        errors.append("evicted block still carries its block_hash field")
    return {"passed": not errors, "errors": errors}


class _StubCheckpointOwner:
    """Duck-typed stand-in exposing only the attributes the real
    ``DirectModelRunner.evict_gdn_checkpoint`` / ``_evict_gdn_checkpoints_for_
    budget`` touch -- both are called as real, unbound methods against this
    object below (NOT a reimplementation). Mirrors the GDN checkpoint pool's
    bookkeeping dicts so the lockstep / byte-budget logic is testable without a
    GPU (no tensor materialization needed for the bookkeeping path)."""

    def __init__(self, pool, byte_budget: int, per_checkpoint_bytes: int) -> None:
        self.block_pool = pool
        self.gdn_checkpoint_byte_budget = byte_budget
        self.gdn_ckpt_per_checkpoint_bytes = per_checkpoint_bytes
        self.gdn_ckpt_meta: dict[int, dict] = {}
        self._gdn_ckpt_by_hash: dict[int, int] = {}
        self._gdn_ckpt_free: list[int] = list(range(64))
        self._gdn_ckpt_lru: OrderedDict[int, None] = OrderedDict()

    def add_fake_ckpt(self, key: int, hash_value: int, num_tokens: int, nbytes: int) -> None:
        pool_slot = self._gdn_ckpt_free.pop()
        self.gdn_ckpt_meta[key] = {
            "key": key,
            "hash_value": hash_value,
            "num_tokens": num_tokens,
            "pool_slot": pool_slot,
            "bytes": nbytes,
            "__slot__": 0,
        }
        self._gdn_ckpt_by_hash[hash_value] = key
        self._gdn_ckpt_lru[key] = None


def _check_lockstep_eviction() -> dict:
    from runtime.direct_model_runner import (
        BlockHash,
        BlockPool,
        DirectModelRunner,
        hash_block_tokens,
    )

    errors = []
    extra = ("fp8_e4m3",)

    # --- Forward direction: evicting an attention tail block drops the co-keyed
    #     GDN checkpoint. ---
    pool = BlockPool(num_blocks=6, reserved=1)  # ids 1..5
    stub = _StubCheckpointOwner(pool, byte_budget=10**9, per_checkpoint_bytes=100)
    pool._on_evict_block = lambda key: DirectModelRunner.evict_gdn_checkpoint(stub, key)
    h = hash_block_tokens(None, list(range(16)), extra)
    b = pool.allocate(1)[0]
    pool.cache_block(b, BlockHash(h, 16))
    stub.add_fake_ckpt(key=b, hash_value=h, num_tokens=16, nbytes=100)  # co-keyed
    pool.free([b])  # cached, ref_cnt 0, in queue
    if b not in stub.gdn_ckpt_meta:
        errors.append("setup: checkpoint not registered")
    # Force eviction of block b (drain the pool => b is popped for reuse).
    pool.allocate(5)
    if pool.get_cached_block(h) is not None:
        errors.append("forward lockstep: attention hash survived eviction")
    if b in stub.gdn_ckpt_meta:
        errors.append("forward lockstep: co-keyed GDN checkpoint survived attention eviction")
    if h in stub._gdn_ckpt_by_hash:
        errors.append("forward lockstep: checkpoint hash index not cleared")

    # --- Reverse direction: evicting the GDN checkpoint drops the co-keyed
    #     attention block's hash (when that block is ref_cnt == 0). ---
    pool2 = BlockPool(num_blocks=6, reserved=1)
    stub2 = _StubCheckpointOwner(pool2, byte_budget=10**9, per_checkpoint_bytes=100)
    h2 = hash_block_tokens(None, list(range(16, 32)), extra)
    b2 = pool2.allocate(1)[0]
    pool2.cache_block(b2, BlockHash(h2, 16))
    stub2.add_fake_ckpt(key=b2, hash_value=h2, num_tokens=16, nbytes=100)
    pool2.free([b2])  # cached, ref_cnt 0
    DirectModelRunner.evict_gdn_checkpoint(stub2, b2)  # reverse lockstep
    if b2 in stub2.gdn_ckpt_meta:
        errors.append("reverse lockstep: checkpoint meta survived")
    if pool2.get_cached_block(h2) is not None:
        errors.append("reverse lockstep: co-keyed attention hash survived checkpoint eviction")
    if pool2.blocks[b2].block_hash is not None:
        errors.append("reverse lockstep: attention block_hash field not cleared")

    # --- Reverse direction must NOT drop the hash of an ACTIVE (ref_cnt>0)
    #     block -- losing only the checkpoint is a safe compute-miss, not a
    #     ghost (INV3 still holds via L=G<=A). ---
    pool3 = BlockPool(num_blocks=6, reserved=1)
    stub3 = _StubCheckpointOwner(pool3, byte_budget=10**9, per_checkpoint_bytes=100)
    h3 = hash_block_tokens(None, list(range(32, 48)), extra)
    b3 = pool3.allocate(1)[0]  # ref_cnt 1 (active, NOT in free queue)
    pool3.cache_block(b3, BlockHash(h3, 16))
    stub3.add_fake_ckpt(key=b3, hash_value=h3, num_tokens=16, nbytes=100)
    DirectModelRunner.evict_gdn_checkpoint(stub3, b3)
    if b3 in stub3.gdn_ckpt_meta:
        errors.append("active-block reverse lockstep: checkpoint meta survived")
    if pool3.get_cached_block(h3) is None:
        errors.append("active-block reverse lockstep wrongly dropped an active block's hash")

    return {"passed": not errors, "errors": errors}


def _check_refcnt_never_evicted() -> dict:
    from runtime.direct_model_runner import BlockHash, BlockPool, hash_block_tokens

    errors = []
    extra = ("fp8_e4m3",)
    pool = BlockPool(num_blocks=5, reserved=1)  # ids 1..4
    # Allocate two blocks; keep the FIRST active (ref_cnt 1), free the second
    # as a hashed (cached) block.
    active = pool.allocate(1)[0]
    cached = pool.allocate(1)[0]
    h = hash_block_tokens(None, list(range(16)), extra)
    pool.cache_block(cached, BlockHash(h, 16))
    pool.free([cached])  # cached block: ref_cnt 0, in queue
    # The active block is ref_cnt 1 and therefore NOT in the free queue at all.
    if active in pool._free_queue:
        errors.append("active (ref_cnt>0) block is in the free queue (INV9 precondition)")
    # Drain EVERYTHING the allocator can hand out. The active block must NEVER
    # be among the popped ids, no matter how much we allocate.
    handed = pool.allocate(pool.num_free_blocks())
    if active in handed:
        errors.append("INV9 violated: a ref_cnt>0 block was handed out by allocate")
    if pool.blocks[active].ref_cnt != 1:
        errors.append(f"active block ref_cnt={pool.blocks[active].ref_cnt} != 1 after drain")
    # True exhaustion: with the active block still held, the pool cannot
    # satisfy one more block.
    raised = False
    try:
        pool.allocate(1)
    except RuntimeError:
        raised = True
    if not raised:
        errors.append("allocate did not raise on true exhaustion (active block unevictable)")
    return {"passed": not errors, "errors": errors}


def _check_byte_budget() -> dict:
    from runtime.direct_model_runner import BlockPool, DirectModelRunner

    errors = []
    per_bytes = 100
    # Budget fits exactly 2 checkpoints (200 bytes). Seed 3 (LRU order k0,k1,k2)
    # then ask the budget eviction to make room for one incoming checkpoint.
    pool = BlockPool(num_blocks=8, reserved=1)
    stub = _StubCheckpointOwner(pool, byte_budget=2 * per_bytes, per_checkpoint_bytes=per_bytes)
    # The budget helper calls self.evict_gdn_checkpoint internally; bind the real
    # (unbound) method to the stub so the lockstep bookkeeping runs for real.
    stub.evict_gdn_checkpoint = lambda key: DirectModelRunner.evict_gdn_checkpoint(stub, key)
    for key in (1, 2, 3):
        stub.add_fake_ckpt(key=key, hash_value=1000 + key, num_tokens=key * 16, nbytes=per_bytes)
    total_before = sum(m["bytes"] for m in stub.gdn_ckpt_meta.values())
    if total_before != 3 * per_bytes:
        errors.append(f"setup: total bytes {total_before} != {3 * per_bytes}")
    # Evict until adding one more (per_bytes) fits the 2*per_bytes budget.
    DirectModelRunner._evict_gdn_checkpoints_for_budget(stub, per_bytes)
    total_after = sum(m["bytes"] for m in stub.gdn_ckpt_meta.values())
    if total_after + per_bytes > stub.gdn_checkpoint_byte_budget:
        errors.append(
            f"byte budget exceeded after eviction: {total_after} + {per_bytes} > "
            f"{stub.gdn_checkpoint_byte_budget}"
        )
    # LRU eviction: the oldest (key 1) is evicted first, then key 2; the newest
    # (key 3) survives -- the 200-byte budget fits exactly one existing (100)
    # plus the incoming (100) checkpoint.
    if sorted(stub.gdn_ckpt_meta) != [3]:
        errors.append(
            f"expected only the newest checkpoint [3] to survive, got {sorted(stub.gdn_ckpt_meta)}"
        )
    if 1001 in stub._gdn_ckpt_by_hash:
        errors.append("evicted checkpoint's hash index entry not cleared")
    return {"passed": not errors, "errors": errors}


# ---------------------------------------------------------------------------
# Real-GPU part.
# ---------------------------------------------------------------------------


def _reset_all(runner, slots) -> None:
    for s in slots:
        if runner.slot_kv_len[s] != 0 or runner.block_table[s]:
            runner.reset_slot(s)


def _clear_persistent_cache(runner) -> None:
    # Test-only isolation: reset the persistent content index + GDN checkpoint
    # pool to a clean slate (the real cache persists by design).
    _reset_all(runner, list(range(runner.num_slots)))
    runner.block_pool.hash_to_block.clear()
    for b in runner.block_pool.blocks:
        b.block_hash = None
    runner.gdn_ckpt_meta.clear()
    runner._gdn_ckpt_by_hash.clear()
    runner._gdn_ckpt_free = list(range(runner.gdn_ckpt_max_checkpoints))
    runner._gdn_ckpt_lru.clear()
    for s in range(runner.num_slots):
        runner.slot_block_hashes[s] = []
        runner.slot_published_blocks[s] = 0
        runner.slot_committed_tokens[s] = []


def _make_prompt_ids(tok, kind: str, n_tokens: int) -> list[int]:
    if kind == "nl":
        base = (
            "You are a careful coding assistant working inside a large repository. "
            "The weather today is mild and pleasant. The repository contains many "
            "modules, tests, and documentation files that must be kept consistent. "
            "Always read the existing code before changing it, and always run the "
            "relevant tests after a change. Keep diffs small and reviewable. "
        )
    elif kind == "code":
        base = (
            "def compute_prefix_cache(tokens, block_size):\n"
            "    hashes = []\n"
            "    parent = 0\n"
            "    for i in range(0, len(tokens), block_size):\n"
            "        block = tokens[i:i + block_size]\n"
            "        parent = blake2b(parent, block)\n"
            "        hashes.append(parent)\n"
            "    return hashes\n"
        )
    else:  # a distinct second natural-language stream (for divergence)
        base = (
            "In a distant galaxy the stars burn with a cold blue light. "
            "Spacecraft drift between the planets carrying rare minerals. "
            "The navigation computer recalculates the trajectory every hour. "
            "Crew members take turns watching the shimmering aurora outside. "
            "Each mission log is stored in a durable crystalline memory bank. "
        )
    text = base * max(1, (n_tokens // 8) + 4)
    ids = tok.encode(text, add_special_tokens=False)
    while len(ids) < n_tokens:
        ids += tok.encode(base, add_special_tokens=False)
    return ids[:n_tokens]


def _decode_rounds(runner, slot: int, anchor: int, draft_tokens: list[int], n: int) -> list[int]:
    """Run n batched verify-commit rounds on one slot; return the committed
    token stream (real_new_tokens concatenated)."""
    tokens: list[int] = []
    anchors = {slot: anchor}
    drafts = {slot: draft_tokens}
    for _ in range(n):
        decs = runner.mtp_verify_and_commit_batch([slot], anchors, drafts)
        d = decs[slot]
        tokens.extend([anchors[slot]] + d["committed"][:-1])
        anchors[slot] = d["next_anchor"]
        drafts[slot] = d["next_draft_tokens"]
    return tokens


def _decode_from_prefill(runner, slot: int, prefill_result: dict, n: int = NUM_ROUNDS) -> list[int]:
    """Convenience: decode n rounds starting from a prefill result dict."""
    pr = prefill_result[slot]
    return _decode_rounds(runner, slot, pr["anchor"], pr["draft_tokens"], n)


def _gdn_stack_compare(runner, slot_a: int, slot_b: int) -> tuple[bool, bool, float, float]:
    """R1-style GDN-state comparison between two slots: layer-0 BYTEWISE-exact
    (so fp8 noise cannot hide a wrong-block / wrong-prefix read) plus the full
    48-layer stack near-tie (``atol=rtol=1e-2``, matching the P3.1 R1 proof).
    Returns ``(l0_bytewise, stack_near_tie, max_conv_diff, max_ssm_diff)``.

    The conv state is compared over COMMITTED token-position rows only. Each
    physical slot's conv state holds ``(kernel_width - 1) + K`` token-position
    rows (SD layout: axis 0 of the per-slot tensor): the first
    ``kernel_width - 1`` are the committed rows the runtime reads as initial
    state, and the trailing ``K = num_speculative_tokens`` are dead
    spec-extension rows. A prior MTP decode leaves those spec rows stale, and
    ``reset_slot`` / ``_clear_persistent_cache`` deliberately do not zero
    tensors while a fresh prefill rewrites only the committed rows -- so a
    full-tensor compare across a slot-reuse-after-decode sees a spurious diff
    in dead rows only. Masking to committed rows compares exactly what the
    runtime reads (proven benign: see
    notes/2026-07-20-cold-prefill-rootcause-plan.md section 3). The ssm compare
    needs no masking: ``ssm_state[pa]`` is the single committed column-0 row
    (spec rows live in a separate address range, see ``_ssm_spec_row``)."""
    import torch

    from runtime.direct_model_runner import _physical_slot

    pa, pb = _physical_slot(slot_a), _physical_slot(slot_b)
    num_spec = runner.num_speculative_tokens or 0
    max_conv = 0.0
    max_ssm = 0.0
    l0_bytewise = True
    stack_near_tie = True
    for i, name in enumerate(runner.gdn_layer_names):
        conv_state, ssm_state = runner.kv_caches[name]
        conv_a, conv_b = conv_state[pa], conv_state[pb]
        # Committed token-position rows = total - K (axis 0 of the per-slot
        # conv tensor is the state_len axis under the active SD layout).
        committed = conv_a.shape[0] - num_spec
        ca, cb = conv_a[:committed].float(), conv_b[:committed].float()
        sa, sb = ssm_state[pa].float(), ssm_state[pb].float()
        max_conv = max(max_conv, float((ca - cb).abs().max().item()))
        max_ssm = max(max_ssm, float((sa - sb).abs().max().item()))
        if not (
            torch.allclose(ca, cb, atol=1e-2, rtol=1e-2)
            and torch.allclose(sa, sb, atol=1e-2, rtol=1e-2)
        ):
            stack_near_tie = False
        if i == 0:
            l0_bytewise = bool(
                torch.equal(conv_a[:committed], conv_b[:committed])
                and torch.equal(ssm_state[pa], ssm_state[pb])
            )
    return l0_bytewise, stack_near_tie, max_conv, max_ssm



def _run_evict_then_recompute(runner, tok) -> dict:
    """Populate P, force-evict P's blocks (sponge the whole free queue), then
    re-request P => clean COLD recompute (L=0), tokens match the original cold
    reference (no half-evicted ghost hit; INV1/INV3)."""
    _clear_persistent_cache(runner)
    result: dict = {"checks": {}}
    slot = 0
    prompt = _make_prompt_ids(tok, "nl", 320)

    # Cold populate (two-phase) => attention blocks hashed + GDN checkpoint.
    pr = runner.mtp_prefill_with_cache([slot], [prompt])
    L_populated = runner.reconcile_prefix_hit(prompt)
    result["L_populated"] = L_populated
    cold_tokens = _decode_from_prefill(runner, slot, pr)
    runner.reset_slot(slot)
    if L_populated <= 0:
        result["checks"]["populated_then_hit_able"] = False
        result["error"] = "populate did not make P hit-able"
        return result
    result["checks"]["populated_then_hit_able"] = True

    # Force-evict EVERYTHING cached: allocate the entire free queue (each popped
    # cached block is evicted -- hash dropped + lockstep GDN checkpoint dropped),
    # then hand the blocks back (now hashless).
    sponge = runner.block_pool.allocate(runner.block_pool.num_free_blocks())
    L_after_evict = runner.reconcile_prefix_hit(prompt)
    result["L_after_evict"] = L_after_evict
    result["checks"]["eviction_actually_happened"] = (L_after_evict == 0)
    runner.block_pool.free(sponge)

    # Re-request P => must be a clean COLD recompute (L=0), tokens matching the
    # original cold reference (no ghost hit on a half-evicted prefix).
    pr2 = runner.mtp_prefill_with_cache([slot], [prompt])
    L_recompute = runner.reconcile_prefix_hit(prompt)  # re-populated now, but the
    # recompute itself ran cold; capture the pre-recompute L via the path taken.
    recompute_tokens = _decode_from_prefill(runner, slot, pr2)
    result["checks"]["cold_recompute_tokens_match"] = (recompute_tokens == cold_tokens)
    result["checks"]["recompute_anchor_match"] = (pr2[slot]["anchor"] == pr[slot]["anchor"])
    _reset_all(runner, [slot])
    _ = L_recompute
    return result


def _run_admission_under_pressure(runner, tok) -> dict:
    """Fill the pool so a new admission must evict while other slots are ACTIVE;
    assert no active ref_cnt>0 block is reclaimed and active committed tokens
    stay correct (INV9/R4). Reuses the sponge technique (hold blocks directly)."""
    _clear_persistent_cache(runner)
    result: dict = {"checks": {}}
    active_slot, q_slot, admit_slot, ref_slot = 0, 1, 2, 3
    active_prompt = _make_prompt_ids(tok, "nl", 300)
    q_prompt = _make_prompt_ids(tok, "code", 300)
    # The admitting request is a DISTINCT, never-seen prompt (cold): it cannot
    # hit the cache, so its prefill must draw fresh blocks from the free queue
    # -- evicting Q's cached blocks -- rather than sharing the active prefix.
    admit_prompt = _make_prompt_ids(tok, "diverge", 200)

    # Cold reference for the active slot (unperturbed full decode).
    ref_pr = runner.mtp_prefill_with_cache([ref_slot], [active_prompt])
    ref_tokens = _decode_from_prefill(runner, ref_slot, ref_pr)
    runner.reset_slot(ref_slot)

    # Make the active slot live (ref_cnt>0 on its blocks).
    act_pr = runner.mtp_prefill_with_cache([active_slot], [active_prompt])
    active_blocks = list(runner.block_table[active_slot])
    active_refcnts_before = [runner.block_pool.blocks[b].ref_cnt for b in active_blocks]

    # Create cached (ref_cnt==0, hashed) blocks: prefill Q then reset (Q stays
    # hit-able in the free queue).
    runner.mtp_prefill_with_cache([q_slot], [q_prompt])
    runner.reset_slot(q_slot)
    if runner.reconcile_prefix_hit(q_prompt) <= 0:
        result["checks"]["q_cached"] = False
        result["error"] = "Q not cached before pressure"
        _reset_all(runner, [active_slot, ref_slot])
        return result
    result["checks"]["q_cached"] = True

    # Sponge the free queue down to ONLY Q's cached blocks: pop every hashless
    # free block (Q's hashed blocks sit at the LRU tail). The active slot's
    # blocks are ref_cnt>0 and therefore NOT in the free queue -- the sponge
    # cannot touch them.
    pool = runner.block_pool
    num_cached_in_queue = sum(
        1 for b in pool.blocks if b.ref_cnt == 0 and b.block_hash is not None
    )
    sponge_n = pool.num_free_blocks() - num_cached_in_queue
    sponge = pool.allocate(sponge_n) if sponge_n > 0 else []
    # The free queue now holds exactly Q's cached blocks.
    result["free_before_admit"] = pool.num_free_blocks()

    # Admit the cold distinct request: the (sponge-shrunk) free queue holds
    # Q's cached blocks, so its prefill allocates+evicts them (drops their
    # hashes + lockstep), never the active slot's ref_cnt>0 blocks.
    admit_pr = runner.mtp_prefill_with_cache([admit_slot], [admit_prompt])
    result["checks"]["admission_succeeded"] = bool(admit_pr.get(admit_slot))

    # INV9: every active block survived (ref_cnt unchanged, still > 0).
    active_refcnts_after = [runner.block_pool.blocks[b].ref_cnt for b in active_blocks]
    result["checks"]["inv9_active_blocks_survived"] = (
        active_refcnts_after == active_refcnts_before
        and all(rc > 0 for rc in active_refcnts_after)
    )
    result["checks"]["active_block_table_unchanged"] = (
        list(runner.block_table[active_slot]) == active_blocks
    )
    # Eviction really engaged: Q's cached blocks were reclaimed (Q no longer
    # hit-able), and the admitting slot drew from the pool.
    result["checks"]["eviction_engaged"] = (runner.reconcile_prefix_hit(q_prompt) == 0)

    # Active committed tokens stay correct: decode the active slot and match
    # the unperturbed cold reference (its KV was never evicted/overwritten).
    active_tokens = _decode_from_prefill(runner, active_slot, act_pr)
    result["checks"]["active_tokens_correct"] = (active_tokens == ref_tokens)

    runner.block_pool.free(sponge)
    _reset_all(runner, [active_slot, admit_slot, ref_slot])
    return result


def _run_chunk_boundary_partial_share(runner, tok) -> dict:
    """A = 20000-token cold prefill chunked at 8192 => GDN checkpoints at
    8192/16384. B shares A's first 18000 tokens then diverges => B hits at
    L=16384 (deepest chunk boundary <= the attention match) and reuses EXACTLY
    that cached checkpoint (INV3). INV1 is proved at the restore boundary, R1
    style extended cross-prompt: the GDN state B restores @L is BYTEWISE the
    state an independent true-cold chunked prefill of B[0,L) produces (so the
    hit reused a correct prefix state, not a stale/wrong-prefix ghost), and the
    hit's continue-prefill anchor matches a true-cold full prefill of B exactly.

    The decode token stream is deliberately NOT compared cold-vs-hit: the hit
    continue-prefills the suffix via ``_prefill_hit_with_cache`` while any cold
    reference uses the chunked-loop / two-phase populate path -- different GDN
    spec-row qo_len bookkeeping, so fp8 split-KV non-associativity flips
    near-tie decode tokens (R6) even though the restored state is bytewise-exact
    and the anchor matches. This is the same reason P3.1's partial-share INV3
    case gates on the reuse boundary + state, not cross-path decode tokens."""
    _clear_persistent_cache(runner)
    result: dict = {"checks": {}}
    a_slot, b_slot, restore_slot, cold_slot, cold_full_slot = 0, 1, 2, 3, 4
    chunk = 8192
    shared = 18000
    a_len = 20000

    base = _make_prompt_ids(tok, "nl", shared)
    a_prompt = base + _make_prompt_ids(tok, "code", a_len - shared)
    b_prompt = base + _make_prompt_ids(tok, "diverge", 1200)  # shares [0,18000), then diverges

    # Populate A via the CHUNKED prefill (chunk_size=8192) so the chunk-boundary
    # checkpoints at 8192/16384 are materialized (the fork-2 coarse policy).
    runner.mtp_prefill_batch([a_slot], [a_prompt], chunk_size=chunk)
    ckpt_boundaries = sorted(meta["num_tokens"] for meta in runner.gdn_ckpt_meta.values())
    result["a_checkpoint_boundaries"] = ckpt_boundaries
    result["checks"]["a_has_chunk_checkpoints"] = (
        chunk in ckpt_boundaries and 2 * chunk in ckpt_boundaries
    )
    runner.reset_slot(a_slot)  # free A's blocks; the checkpoint persists (R10)

    # B reconciles: attention matches [0,18000) (18000 == 16*1125, block-aligned);
    # the deepest GDN checkpoint <= that is 16384.
    L = runner.reconcile_prefix_hit(b_prompt)
    result["L"] = L
    result["checks"]["hit_at_expected_boundary"] = (L == 2 * chunk)

    # Hit B => restore @16384 + continue-prefill the suffix [16384, 19200).
    num_L_blocks = L // runner.block_size
    hit_pr = runner.mtp_prefill_with_cache([b_slot], [b_prompt])
    hit_anchor = hit_pr[b_slot]["anchor"]
    restored_ids = list(runner.block_table[b_slot][:num_L_blocks])
    # The restored [0,L) blocks ARE A's cached blocks (the hit actually fired).
    a_block_ids_for_prefix = runner._compute_prompt_block_hashes(b_prompt, L)
    cached_ids = [
        runner.block_pool.get_cached_block(bh.value).block_id
        for bh in a_block_ids_for_prefix
        if runner.block_pool.get_cached_block(bh.value) is not None
    ]
    result["checks"]["hit_reuses_cached_blocks"] = (
        L > 0 and bool(restored_ids) and restored_ids == cached_ids
        and all(runner.block_pool.blocks[b].ref_cnt >= 1 for b in restored_ids)
    )

    # INV1 (restore-boundary addressing proof): expose the restored @L state via
    # restore_cached_prefix into a spare slot, then independently compute B[0,L)
    # true-cold (mtp_prefill_batch is always a fresh compute, never a hit) with
    # the SAME chunk_size so the [0,L) state is built by identical chunk
    # boundaries. The two must match layer-0 bytewise + full-stack near-tie.
    runner.reset_slot(restore_slot)
    runner.restore_cached_prefix(restore_slot, b_prompt, L)
    runner.reset_slot(cold_slot)
    runner.mtp_prefill_batch([cold_slot], [b_prompt[:L]], chunk_size=chunk)
    l0_exact, stack_near_tie, max_conv, max_ssm = _gdn_stack_compare(
        runner, restore_slot, cold_slot
    )
    result["inv1_restore_max_conv_diff"] = max_conv
    result["inv1_restore_max_ssm_diff"] = max_ssm
    result["checks"]["inv1_restore_state_matches_cold"] = l0_exact and stack_near_tie

    # INV1 (continue-prefill output): the hit's prefill anchor matches a true-
    # cold full prefill of B exactly. The anchor is the greedy token at the
    # final prefill position; it depends on the whole [0,prompt_len) compute, so
    # an exact match proves the suffix continue-prefill produced correct output.
    runner.reset_slot(cold_full_slot)
    cold_full_pr = runner.mtp_prefill_batch([cold_full_slot], [b_prompt], chunk_size=chunk)
    result["cold_anchor"] = cold_full_pr[cold_full_slot]["anchor"]
    result["checks"]["inv1_anchor_matches_cold"] = (
        hit_anchor == cold_full_pr[cold_full_slot]["anchor"]
    )

    _reset_all(runner, [a_slot, b_slot, restore_slot, cold_slot, cold_full_slot])
    return result


def _run_a_gt_0_g_eq_0_dedup(runner, tok) -> dict:
    """Compute-miss case: attention blocks cached but NO GDN checkpoint at a
    usable boundary => reconcile returns L=0; the fresh recompute's write-time
    dedup reclaims the duplicate attention blocks (free count recovers; no
    leak); tokens correct."""
    _clear_persistent_cache(runner)
    result: dict = {"checks": {}}
    slot = 0
    prompt = _make_prompt_ids(tok, "nl", 300)

    # Populate ATTENTION ONLY via single-shot mtp_prefill_batch (no GDN
    # completion checkpoint -- a single-shot prefill's live GDN state is at
    # prompt_len, not at the block-aligned boundary G).
    runner.mtp_prefill_batch([slot], [prompt])
    # Attention match A > 0 ...
    hashes = runner._compute_prompt_block_hashes(prompt, len(prompt) - 1)
    matched = 0
    for bh in hashes:
        if runner.block_pool.get_cached_block(bh.value) is None:
            break
        matched += 1
    A = matched * runner.block_size
    result["A"] = A
    runner.reset_slot(slot)
    # ... but no GDN checkpoint => reconcile returns L=0 (compute miss).
    L = runner.reconcile_prefix_hit(prompt)
    result["L"] = L
    result["checks"]["compute_miss_L0"] = (A > 0 and L == 0)

    # Re-request cold via the two-phase path: its write-time dedup must reclaim
    # the recomputed attention blocks onto the already-cached ones (no leak).
    free_before = runner.block_pool.num_free_blocks()
    pr = runner.mtp_prefill_with_cache([slot], [prompt])
    # The slot's published prefix blocks must be the SAME physical blocks that
    # were already cached (dedup swapped onto them, freeing the fresh dups).
    num_blocks = (len(prompt) - 1) // runner.block_size
    table_ids = list(runner.block_table[slot][:num_blocks])
    cached_ids = [
        runner.block_pool.get_cached_block(bh.value).block_id
        for bh in runner._compute_prompt_block_hashes(prompt, num_blocks * runner.block_size)
    ]
    result["checks"]["dedup_reclaimed_duplicates"] = (
        bool(table_ids) and table_ids == cached_ids
    )
    tokens = _decode_rounds(runner, slot, pr[slot]["anchor"], pr[slot]["draft_tokens"], NUM_ROUNDS)
    runner.reset_slot(slot)
    free_after = runner.block_pool.num_free_blocks()
    result["free_before"] = free_before
    result["free_after"] = free_after
    result["checks"]["no_leak_free_recovered"] = (free_after == free_before)
    # Tokens correct: match an independent cold reference.
    _reset_all(runner, [slot])
    ref_pr = runner.mtp_prefill_with_cache([slot], [prompt])
    ref_tokens = _decode_from_prefill(runner, slot, ref_pr)
    result["checks"]["tokens_correct"] = (tokens == ref_tokens)
    _reset_all(runner, [slot])
    return result


def _run_no_leak_churn(runner, tok) -> dict:
    """Many admit/finish cycles under pressure; pool free count returns to
    baseline and cuda_allocated_mib stays flat (no leak; R10/R8)."""
    import torch

    _clear_persistent_cache(runner)
    result: dict = {"checks": {}}
    slot, scratch = 0, 1
    prompt = _make_prompt_ids(tok, "nl", 400)
    cycles = 8

    # Warm up the lazily-allocated GDN checkpoint pool slots so the memory
    # flatness check measures steady-state (not first-touch allocation).
    for _ in range(2):
        runner.mtp_prefill_with_cache([slot], [prompt])
        runner.reset_slot(slot)
    torch.cuda.synchronize()
    free_baseline = runner.block_pool.num_free_blocks()
    mem_baseline = torch.cuda.memory_allocated() / (1024 * 1024)

    mem_readings = [mem_baseline]
    free_ok = True
    for c in range(cycles):
        # Admit (cold or hit), decode a few rounds, finish -- under pressure
        # (sponge most of the pool each cycle so eviction actually runs).
        sponge_n = max(0, runner.block_pool.num_free_blocks() - 40)
        sponge = runner.block_pool.allocate(sponge_n) if sponge_n > 0 else []
        pr = runner.mtp_prefill_with_cache([slot], [prompt])
        _decode_rounds(runner, slot, pr[slot]["anchor"], pr[slot]["draft_tokens"], 4)
        runner.reset_slot(slot)
        runner.block_pool.free(sponge)
        # Scratch churn to rotate the queue.
        runner.mtp_prefill_batch([scratch], [prompt[:64]])
        runner.reset_slot(scratch)
        if runner.block_pool.num_free_blocks() != free_baseline:
            free_ok = False
        torch.cuda.synchronize()
        mem_readings.append(torch.cuda.memory_allocated() / (1024 * 1024))

    free_after = runner.block_pool.num_free_blocks()
    leaked = [b.block_id for b in runner.block_pool.blocks if b.ref_cnt != 0]
    result["free_baseline"] = free_baseline
    result["free_after"] = free_after
    result["checks"]["free_count_returns_to_baseline"] = (free_after == free_baseline) and free_ok
    result["checks"]["no_stuck_refcnt"] = (not leaked)
    # Memory flatness: the spread of allocated MiB across cycles is tiny (the
    # fixed-address pools never grow once warmed up). Allow a small slack for
    # allocator rounding.
    mem_spread = max(mem_readings) - min(mem_readings)
    result["mem_baseline_mib"] = round(mem_baseline, 1)
    result["mem_spread_mib"] = round(mem_spread, 1)
    result["checks"]["cuda_allocated_flat"] = (mem_spread < 64.0)
    _reset_all(runner, [slot, scratch])
    return result


def _run_gpu_checks() -> dict:
    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=22528,
        gpu_memory_utilization=0.85,
        speculative_config=SPECULATIVE_CONFIG,
    )
    # num_slots=6 (active/admit/reference slots for the pressure cases);
    # blocks_per_slot=352 (22528-token capacity) fits the 20000-token
    # chunk-boundary prompt plus decode. enable_cudagraph stays False -- this
    # gate targets eviction/populate CORRECTNESS (INV2/3/9, R4/5/7/8); CUDA-graph
    # parity over hit tables is INV5, scoped to P3.3.
    runner = DirectModelRunner(
        vllm_config,
        num_slots=6,
        block_size=64,
        blocks_per_slot=352,
        enable_block_table=True,
        enable_prefix_cache=True,
        enable_persistent_prefix_cache=True,
        enable_cudagraph=False,
    )
    tok = AutoTokenizer.from_pretrained(MODEL)

    results: dict = {"cases": {}}
    overall = True

    for name, fn in [
        ("evict_then_recompute", lambda: _run_evict_then_recompute(runner, tok)),
        ("admission_under_pressure", lambda: _run_admission_under_pressure(runner, tok)),
        ("a_gt_0_g_eq_0_dedup", lambda: _run_a_gt_0_g_eq_0_dedup(runner, tok)),
        ("chunk_boundary_partial_share", lambda: _run_chunk_boundary_partial_share(runner, tok)),
        ("no_leak_churn", lambda: _run_no_leak_churn(runner, tok)),
    ]:
        case = fn()
        results["cases"][name] = case
        for ok in case.get("checks", {}).values():
            if not ok:
                overall = False

    results["passed"] = overall
    return results


def main() -> int:
    print("=== prefix_cache_eviction_check (P3.2 eviction + full populate) ===")

    pure_checks = [
        ("lru_middle_removal", _check_lru_middle_removal),
        ("evict_drops_hash", _check_evict_drops_hash),
        ("lockstep_eviction", _check_lockstep_eviction),
        ("refcnt_never_evicted", _check_refcnt_never_evicted),
        ("byte_budget", _check_byte_budget),
    ]
    pure_ok = True
    for name, fn in pure_checks:
        r = fn()
        print(f"{name}: {'PASS' if r['passed'] else 'FAIL'}")
        for e in r["errors"]:
            print(f"  - {e}")
        if not r["passed"]:
            pure_ok = False

    gpu = _run_gpu_checks()
    for name, case in gpu["cases"].items():
        print(f"{name}:")
        for k, v in case.items():
            if k == "checks":
                for cname, ok in v.items():
                    print(f"  {cname}: {'PASS' if ok else 'FAIL'}")
            else:
                print(f"  {k}: {v}")

    overall = pure_ok and gpu["passed"]
    print(f"\npassed: {str(overall).lower()}")
    print(f"=== overall: {'PASS' if overall else 'FAIL'} ===")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
