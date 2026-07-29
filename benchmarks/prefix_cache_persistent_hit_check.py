"""Correctness gate for prefix-cache P3.1 -- persistent content-addressed
cache hit equivalence (``notes/2026-07-19-p3-implementation-plan.md``,
"### P3.1 -- Persistent-cache hit equivalence"; ``notes/prefix-cache-design
.md`` sec 4 invariants INV1/INV3/INV4/INV6/INV7, sec 6 risks R1/R10).

P3.1 proves the load-bearing INV1 reduction with a PERSISTENT (cross-
``reset_slot``, cross-round) GDN checkpoint + content-addressed attention
blocks: a request served by "restore checkpoint @ L + reference the [0, L)
attention blocks + continue-prefill [L, prompt)" produces the same committed
tokens as a cold prefill, with GDN layer 0 BYTEWISE-EXACT as the addressing
proof (R1, so fp8 noise cannot hide a wrong-block read).

The hit path is driven ONLY by this test (via ``mtp_prefill_with_cache``);
the production prefill entrypoint is untouched in P3.1, so production
behavior is byte-for-byte P2 (flag defaults off).

Checks (pure-Python part + real-GPU part), mirroring ``prefix_cache_fanout_
check.py``'s methodology (near-tie ``NEAR_TIE_LOGIT_MARGIN = 2.0`` per R6 for
token/logit comparisons -- NOT bytewise -- EXCEPT the GDN-layer-0 addressing
proof which is exact per R1):

Pure Python:
* ``hash_chain_determinism`` -- same tokens => same chain; one-token
  divergence => every hash from divergence on differs; ``extra_keys`` dtype
  change => different hash (R7).
* ``index_keepalive`` -- publish => ``free`` => ``ref_cnt == 0`` but
  ``hash_to_block`` still resolves; ``touch`` revives from the free queue;
  ``free`` of a published block retains its hash (R10); allocate drops the
  hash of a popped hashed block (INV2).

Real GPU (one ``DirectModelRunner``, ``enable_block_table=True`` +
``enable_prefix_cache=True`` + ``enable_persistent_prefix_cache=True``,
MTP K=3):
* ``r1_gdn_layer0_exact`` (R1) -- the restored GDN layer-0 (conv, ssm) state
  at the hit boundary L is BYTEWISE identical to an independent cold prefill
  of [0, L)'s layer-0 state (the addressing proof); the full 48-layer stack
  is near-tie.
* ``inv1_cold_vs_hit`` (INV1) -- same prompt served cold vs via persistent hit
  produces matching committed tokens (near-tie) over 20+ MTP decode rounds,
  at several L (a 100-token prompt hitting 96; a 5000-token prompt hitting
  the block-aligned len-1; a code prompt), for a natural-language AND a code
  prompt. Includes the fork_actually_engaged analogue: the hit ACTUALLY fired
  (L > 0 and the [0, L) blocks ARE the cached ones), so a silent cold
  fallback cannot pass trivially.
* ``inv4_multiround_after_hit`` (INV4) -- multi-round MTP after a hit stays
  oracle-aligned per step (``draft_sync_len == kv_len`` + near-tie next-token)
  plus acceptance-rate sanity vs a cold baseline.
* ``inv3_mismatched_prefix`` (INV3) -- a request sharing only the first Lc
  tokens reuses EXACTLY the cached checkpoint boundary (not more); a request
  whose attention matches deeper than the only checkpoint reuses only up to
  that checkpoint (L = G <= A).
* ``persistence_across_reset`` (R10) -- populate survives ``reset_slot`` of
  the producing slot (blocks stay hit-able at ``ref_cnt == 0``); a SECOND
  request after the first slot resets still hits.
* ``inv6_inv7`` (INV6/INV7) -- published blocks are append-only/immutable;
  reserved block 0 is never published.
* ``no_block_leak`` (R10/INV9) -- after hit + decode + ``reset_slot``, the
  free count returns to baseline minus the still-cached (ref_cnt == 0, hashed)
  blocks; every live ``ref_cnt == 0``; ``block_table[slot] == []``.

Usage:
    python -m benchmarks.prefix_cache_persistent_hit_check
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3
NUM_ROUNDS = 22  # 20+ MTP decode rounds (plan P3.1 test slice)
SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"}

NEAR_TIE_LOGIT_MARGIN = 2.0  # established methodology (R6): a real near-exact
# tie is kernel-path-sensitive fp8/batch non-associativity noise, NOT state
# corruption; distinct real candidates are typically separated by 8-13+ logit
# units. A mismatch is only a real failure if the reference's own logit for the
# hit's committed token is NOT within this margin of the reference's top.


# ---------------------------------------------------------------------------
# Pure-Python part (no GPU, no model load).
# ---------------------------------------------------------------------------


def _check_hash_chain_determinism() -> dict:
    from oracle.qwen36_vllm.direct_model_runner import BlockHash, hash_block_tokens

    errors = []
    extra = ("fp8_e4m3",)
    toks_a = list(range(16))
    h1 = hash_block_tokens(None, toks_a, extra)
    h1b = hash_block_tokens(None, toks_a, extra)
    if h1 != h1b:
        errors.append("same tokens did not produce the same hash (not deterministic)")
    if not (isinstance(h1, int) and 0 < h1.bit_length() <= 128):
        errors.append(f"hash is not a positive <=128-bit int: bit_length={h1.bit_length()}")

    # One-token divergence => this block's hash differs...
    toks_b = list(range(15)) + [99]
    hb = hash_block_tokens(None, toks_b, extra)
    if h1 == hb:
        errors.append("one-token divergence did not change the block hash")
    # ...and EVERY chained hash from the divergence on differs too.
    chain_a = hash_block_tokens(h1, list(range(16, 32)), extra)
    chain_b = hash_block_tokens(hb, list(range(16, 32)), extra)
    if chain_a == chain_b:
        errors.append("chained hash did not propagate the divergence (chain_a == chain_b)")

    # extra_keys dtype change => different hash (fp8 vs nvfp4 KV never collide, R7).
    h_nvfp4 = hash_block_tokens(None, toks_a, ("nvfp4",))
    if h1 == h_nvfp4:
        errors.append("changing extra_keys (kv_cache_dtype) did not change the hash")

    # BlockHash is a frozen (value) dataclass carrying num_tokens for the
    # paranoid first-block verify (R7).
    bh = BlockHash(h1, 16)
    if bh.value != h1 or bh.num_tokens != 16:
        errors.append("BlockHash fields wrong")
    try:
        bh.value = 5  # type: ignore[misc]
        errors.append("BlockHash is not frozen (mutable)")
    except Exception:
        pass
    return {"passed": not errors, "errors": errors}


def _check_index_keepalive() -> dict:
    from oracle.qwen36_vllm.direct_model_runner import BlockHash, BlockPool, hash_block_tokens

    errors = []
    extra = ("fp8_e4m3",)
    h0 = hash_block_tokens(None, list(range(16)), extra)
    pool = BlockPool(num_blocks=20, reserved=1)

    blk = pool.allocate(1)[0]
    pool.cache_block(blk, BlockHash(h0, 16))
    if pool.get_cached_block(h0) is None or pool.get_cached_block(h0).block_id != blk:
        errors.append("cache_block did not register the block in hash_to_block")

    # Idempotent (write-time-dedup signal): re-caching the same hash value does
    # NOT overwrite the canonical block.
    blk2 = pool.allocate(1)[0]
    pool.cache_block(blk2, BlockHash(h0, 16))
    if pool.get_cached_block(h0).block_id != blk:
        errors.append("cache_block overwrote an existing hash entry (must be idempotent)")

    # free of a published block retains its hash (R10 keepalive): the block is
    # back at ref_cnt == 0 / in the free queue, but still hit-able.
    pool.free([blk])
    if pool.blocks[blk].ref_cnt != 0:
        errors.append(f"published block ref_cnt={pool.blocks[blk].ref_cnt} != 0 after free")
    if pool.get_cached_block(h0) is None:
        errors.append("freeing a published block dropped its hash (must stay hit-able, R10)")
    if pool.blocks[blk].block_hash is None:
        errors.append("freeing a published block cleared its block_hash field")

    # touch revives a ref_cnt == 0 published block from the free queue.
    pool.touch([blk])
    if pool.blocks[blk].ref_cnt != 1:
        errors.append(f"touch did not raise ref_cnt to 1 (got {pool.blocks[blk].ref_cnt})")
    if blk in pool._free_queue:
        errors.append("touch did not yank the revived block out of the free queue")

    # INV7: reserved block 0 is never publishable / touchable.
    for bad_op in ("cache", "touch"):
        raised = False
        try:
            if bad_op == "cache":
                pool.cache_block(0, BlockHash(12345, 16))
            else:
                pool.touch([0])
        except RuntimeError:
            raised = True
        if not raised:
            errors.append(f"{bad_op} did not reject reserved block 0 (INV7)")

    # allocate hash-drop under pressure (design doc sec 3.2 _maybe_evict, INV2):
    # a hashed block popped from the free queue has its hash dropped first.
    pool2 = BlockPool(num_blocks=4, reserved=1)
    h1 = hash_block_tokens(None, list(range(16)), ("nvfp4",))
    b = pool2.allocate(1)[0]
    pool2.cache_block(b, BlockHash(h1, 16))
    pool2.free([b])  # ref_cnt 0, in queue, hash retained
    if pool2.get_cached_block(h1) is None:
        errors.append("freed published block lost its hash before pressure")
    pool2.allocate(3)  # exhausts the pool, popping the hashed block
    if pool2.get_cached_block(h1) is not None:
        errors.append("allocate did not drop the hash of a popped hashed block (INV2)")

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
    # pool to a clean slate so each case starts from an empty cache (the real
    # cache persists by design -- this just lets each case populate+hit within
    # itself without cross-case prefix overlap, e.g. the NL prompts sharing a
    # base prefix).
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


def _run_inv1_case(runner, tok, prompt_ids: list[int], label: str) -> dict:
    """One INV1 cold-vs-hit case. The cold reference is a full cold MTP run
    (cold ``mtp_prefill_with_cache`` + ``mtp_verify_and_commit_batch`` decode)
    -- the SAME machinery the hit uses -- so the committed-token comparison
    isolates the cold-vs-hit PREFILL difference (which R1 proves byte-exact),
    not a verify-vs-decode kernel-path artifact (fp8 split-KV is
    non-associative across different qo_len, which a cross-path reference
    would falsely flag at long context). The cold run also populates the
    persistent cache (two-phase cold path); after a ``reset_slot`` (R10
    persistence) the hit run reuses it.

    Asserts: the hit actually fired (L>0 + the [0,L) blocks ARE the cached
    ones); the prefill anchor matches the cold run AND an independent plain
    ``prefill`` (a third path); 20+ MTP decode rounds commit the SAME tokens
    cold vs hit (INV1) with ``draft_sync_len == kv_len`` each round (INV4);
    acceptance-rate sanity vs the cold baseline.
    """
    ref_slot, hit_slot, plain_slot = 0, 1, 5
    _clear_persistent_cache(runner)
    result: dict = {"label": label, "prompt_len": len(prompt_ids), "checks": {}}

    # --- Cold reference MTP run (cache empty => cold; populates the cache). ---
    ref_pr = runner.mtp_prefill_with_cache([ref_slot], [prompt_ids])
    cold_anchor = ref_pr[ref_slot]["anchor"]
    L = runner.reconcile_prefix_hit(prompt_ids)  # > 0 now (cold run populated)
    num_L_blocks = L // runner.block_size
    published_ids = list(runner.block_table[ref_slot][:num_L_blocks]) if num_L_blocks else []
    result["L"] = L
    result["published_ids"] = published_ids

    cold_rounds = []
    cold_anchors = {ref_slot: cold_anchor}
    cold_drafts = {ref_slot: ref_pr[ref_slot]["draft_tokens"]}
    cold_total_draft = 0
    cold_total_acc = 0
    for _r in range(NUM_ROUNDS):
        decs = runner.mtp_verify_and_commit_batch([ref_slot], cold_anchors, cold_drafts)
        d = decs[ref_slot]
        cold_rounds.append(
            {
                "real_new_tokens": [cold_anchors[ref_slot]] + d["committed"][:-1],
                "next_anchor": d["next_anchor"],
                "num_accepted": d["num_accepted"],
            }
        )
        cold_total_draft += len(cold_drafts[ref_slot])
        cold_total_acc += d["num_accepted"] + 1
        cold_anchors[ref_slot] = d["next_anchor"]
        cold_drafts[ref_slot] = d["next_draft_tokens"]
    cold_acceptance = (
        cold_total_acc / (cold_total_draft + cold_total_acc)
        if (cold_total_draft + cold_total_acc) else 0.0
    )
    result["cold_acceptance_rate"] = cold_acceptance

    # --- Persistence across reset (R10): reset the producing slot; the
    #     published blocks must stay hit-able (hash retained at ref_cnt==0). ---
    runner.reset_slot(ref_slot)
    still_indexed = bool(published_ids) and all(
        runner.block_pool.blocks[b].block_hash is not None
        and runner.block_pool.get_cached_block(
            runner.block_pool.blocks[b].block_hash.value
        ) is not None
        for b in published_ids
    )
    result["checks"]["persistence_index_survives_reset"] = still_indexed and L > 0

    # --- Hit run (reuses the cache the cold run populated). ---
    hit_pr = runner.mtp_prefill_with_cache([hit_slot], [prompt_ids])
    hit_anchor = hit_pr[hit_slot]["anchor"]
    restored_ids = list(runner.block_table[hit_slot][:num_L_blocks])
    hit_engaged = (
        L > 0
        and bool(published_ids)
        and restored_ids == published_ids
        and all(runner.block_pool.blocks[b].ref_cnt >= 1 for b in restored_ids)
    )
    result["checks"]["hit_actually_engaged"] = hit_engaged
    result["restored_ids"] = restored_ids

    # --- INV1 prefill anchor: cold run == hit == independent plain prefill. ---
    runner.reset_slot(plain_slot)
    plain_anchor = runner.prefill(plain_slot, prompt_ids)
    result["checks"]["inv1_prefill_anchor"] = (cold_anchor == hit_anchor == plain_anchor)
    result["cold_anchor"] = cold_anchor
    result["hit_anchor"] = hit_anchor
    result["plain_anchor"] = plain_anchor

    # --- INV1 decode + INV4: 20+ MTP rounds commit the SAME tokens cold vs hit
    #     (same machinery => exact match expected; R1 proves the prefill state
    #     is identical), with draft_sync_len == kv_len each round (INV4). ---
    hit_rounds = []
    hit_anchors = {hit_slot: hit_anchor}
    hit_drafts = {hit_slot: hit_pr[hit_slot]["draft_tokens"]}
    hit_total_draft = 0
    hit_total_acc = 0
    inv4_sync_ok = True
    for _r in range(NUM_ROUNDS):
        decs = runner.mtp_verify_and_commit_batch([hit_slot], hit_anchors, hit_drafts)
        d = decs[hit_slot]
        hit_rounds.append(
            {
                "real_new_tokens": [hit_anchors[hit_slot]] + d["committed"][:-1],
                "next_anchor": d["next_anchor"],
                "num_accepted": d["num_accepted"],
            }
        )
        if runner.slot_draft_sync_len[hit_slot] != runner.slot_kv_len[hit_slot]:
            inv4_sync_ok = False
        hit_total_draft += len(hit_drafts[hit_slot])
        hit_total_acc += d["num_accepted"] + 1
        hit_anchors[hit_slot] = d["next_anchor"]
        hit_drafts[hit_slot] = d["next_draft_tokens"]
    hit_acceptance = (
        hit_total_acc / (hit_total_draft + hit_total_acc)
        if (hit_total_draft + hit_total_acc) else 0.0
    )
    result["hit_acceptance_rate"] = hit_acceptance

    inv1_decode_ok = all(
        c["real_new_tokens"] == h["real_new_tokens"] and c["next_anchor"] == h["next_anchor"]
        for c, h in zip(cold_rounds, hit_rounds)
    )
    result["checks"]["inv1_decode_equiv"] = inv1_decode_ok
    result["checks"]["inv4_multiround_mtp"] = inv1_decode_ok and inv4_sync_ok
    result["checks"]["inv4_acceptance_sanity"] = abs(hit_acceptance - cold_acceptance) < 0.20
    # Report the first cold-vs-hit token divergence (if any) for diagnostics.
    first_mismatch = next(
        (
            {"round": i, "cold": c["real_new_tokens"], "hit": h["real_new_tokens"]}
            for i, (c, h) in enumerate(zip(cold_rounds, hit_rounds))
            if c["real_new_tokens"] != h["real_new_tokens"]
            or c["next_anchor"] != h["next_anchor"]
        ),
        None,
    )
    result["first_mismatch"] = first_mismatch

    _reset_all(runner, [ref_slot, hit_slot, plain_slot])
    return result


def _run_r1_gdn_layer0_exact(runner, tok, prompt_ids: list[int]) -> dict:
    """R1 addressing proof: at the hit boundary L, the restored GDN layer-0
    (conv, ssm) state is BYTEWISE identical to an independent cold prefill of
    [0, L)'s layer-0 state (so fp8 noise cannot hide a wrong-block read). The
    full 48-layer stack is near-tie."""
    import torch

    from oracle.qwen36_vllm.direct_model_runner import _physical_slot

    produce_slot, hit_slot, ref_slot = 0, 1, 4
    _clear_persistent_cache(runner)
    result: dict = {"checks": {}}

    # Produce cold => checkpoint at G = block_align_down(prompt_len - 1).
    runner.mtp_prefill_with_cache([produce_slot], [prompt_ids])
    L = runner.reconcile_prefix_hit(prompt_ids)
    result["L"] = L
    runner.reset_slot(produce_slot)
    if L <= 0:
        result["checks"]["r1_gdn_layer0_exact"] = False
        result["error"] = "no hit (L==0); cannot run R1 proof"
        return result

    # Independent fresh compute of [0, L) into ref_slot via the SAME code path
    # the producing phase-1 used (_forward_batch, batch=1) => identical kernel,
    # so any difference isolates a real addressing/state bug (not a code-path
    # artifact).
    runner._forward_batch(
        [ref_slot], [prompt_ids[:L]], [0], qo_len=L, commit=True,
        return_hidden=False, is_decode=False, logits_last_position_only=True,
    )
    # Restore the checkpoint into hit_slot.
    runner.restore_cached_prefix(hit_slot, prompt_ids, L)

    gdn0 = runner.gdn_layer_names[0]
    rp = _physical_slot(ref_slot)
    hp = _physical_slot(hit_slot)
    ref_conv, ref_ssm = runner.kv_caches[gdn0]
    layer0_exact = bool(
        torch.equal(runner.kv_caches[gdn0][0][hp], ref_conv[rp])
        and torch.equal(runner.kv_caches[gdn0][1][hp], ref_ssm[rp])
    )
    result["checks"]["r1_gdn_layer0_exact"] = layer0_exact

    # Full 48-layer stack near-tie.
    stack_ok = True
    max_conv_diff = 0.0
    max_ssm_diff = 0.0
    for name in runner.gdn_layer_names:
        conv_state, ssm_state = runner.kv_caches[name]
        cd = (conv_state[hp].float() - conv_state[rp].float()).abs().max().item()
        sd = (ssm_state[hp].float() - ssm_state[rp].float()).abs().max().item()
        max_conv_diff = max(max_conv_diff, cd)
        max_ssm_diff = max(max_ssm_diff, sd)
        if not (
            torch.allclose(conv_state[hp].float(), conv_state[rp].float(), atol=1e-2, rtol=1e-2)
            and torch.allclose(ssm_state[hp].float(), ssm_state[rp].float(), atol=1e-2, rtol=1e-2)
        ):
            stack_ok = False
    result["checks"]["r1_full_stack_near_tie"] = stack_ok
    result["max_conv_diff"] = max_conv_diff
    result["max_ssm_diff"] = max_ssm_diff

    _reset_all(runner, [produce_slot, hit_slot, ref_slot])
    return result


def _run_inv3_mismatched_prefix(runner, tok, base_ids: list[int]) -> dict:
    """INV3: a request sharing only the first Lc tokens reuses EXACTLY the
    cached checkpoint boundary (not more); a request whose attention matches
    deeper than the only checkpoint reuses only up to that checkpoint (L=G<=A).

    Produce R_short (a prefix of base) => checkpoint at boundary B. Then:
    * Q_share = R_short + extra tokens (shares more than B tokens with R_short,
      but R_short's partial tail past B is never published) => reuses exactly B.
    * r_long published via single-shot mtp_prefill_batch (attention deeper than
      the checkpoint, no new checkpoint) => a matching request reuses only B.
    """
    _clear_persistent_cache(runner)
    result: dict = {"checks": {}}
    block_size = runner.block_size
    produce_slot, produce2 = 0, 1

    # R_short: 100 tokens => completion boundary B = block_align_down(99) = 96.
    short_len = 100
    r_short = base_ids[:short_len]
    runner.mtp_prefill_with_cache([produce_slot], [r_short])
    B = runner.reconcile_prefix_hit(r_short)
    result["B"] = B
    runner.reset_slot(produce_slot)

    # Q_share shares all 100 tokens of R_short then extends. Attention matches
    # the 6 full blocks [0,96); R_short's partial tail [96,100) was never
    # published and its only checkpoint is at B=96 => reuses EXACTLY B.
    q_share = r_short + base_ids[short_len : short_len + 60]
    L_share = runner.reconcile_prefix_hit(q_share)
    result["L_share"] = L_share
    result["checks"]["inv3_reuses_exactly_boundary"] = (B > 0) and (L_share == B)

    # A > G case: publish attention deeper than the checkpoint via a single-shot
    # mtp_prefill_batch (attention-only populate, no new GDN checkpoint).
    _reset_all(runner, [produce2])
    long_len = 200
    r_long = base_ids[:long_len]  # extends R_short; shares [0,100) with it
    runner.mtp_prefill_batch([produce2], [r_long])
    runner.reset_slot(produce2)
    # Attention walk now matches deeper than B, but the only checkpoint is at B
    # => L = G = B <= A.
    hashes = runner._compute_prompt_block_hashes(r_long, len(r_long) - 1)
    matched = 0
    for bh in hashes:
        if runner.block_pool.get_cached_block(bh.value) is None:
            break
        matched += 1
    A_long = matched * block_size
    L_long = runner.reconcile_prefix_hit(r_long)
    result["A_long"] = A_long
    result["L_long"] = L_long
    result["checks"]["inv3_L_eq_G_le_A"] = (A_long > B) and (L_long == B)

    _reset_all(runner, list(range(runner.num_slots)))
    return result


def _run_inv6_inv7(runner, tok, prompt_ids: list[int]) -> dict:
    """INV6 (append-only, immutable-published, private-tail) + INV7 (reserved
    block 0 never published)."""
    _clear_persistent_cache(runner)
    result: dict = {"checks": {}}
    produce_slot, produce2 = 0, 1

    runner.mtp_prefill_with_cache([produce_slot], [prompt_ids])
    L = runner.reconcile_prefix_hit(prompt_ids)
    num_blocks = L // runner.block_size
    published = list(runner.block_table[produce_slot][:num_blocks])

    # INV7: reserved block 0 is never among the published blocks, and no index
    # entry points at it (the pool excludes it; the index never references it).
    inv7_ok = (
        bool(published)
        and (0 not in published)
        and all(blk.block_id != 0 for blk in runner.block_pool.hash_to_block.values())
    )
    result["checks"]["inv7_reserved_block0_never_published"] = inv7_ok

    # INV6: published blocks are immutable -- a SECOND identical produce must
    # dedup onto the SAME physical blocks (not republish a different block under
    # the same hash); each published hash keeps mapping to the same block id.
    hashes_before = {runner.block_pool.blocks[b].block_hash.value: b for b in published}
    runner.reset_slot(produce_slot)
    _reset_all(runner, [produce2])
    runner.mtp_prefill_with_cache([produce2], [prompt_ids])
    published2 = list(runner.block_table[produce2][:num_blocks])
    inv6_ok = (published2 == published) and all(
        runner.block_pool.blocks[b].block_hash.value == h for h, b in hashes_before.items()
    )
    result["checks"]["inv6_append_only_immutable"] = inv6_ok
    result["published"] = published
    result["published2"] = published2

    _reset_all(runner, list(range(runner.num_slots)))
    return result


def _run_no_block_leak(runner, tok, prompt_ids: list[int]) -> dict:
    """no_block_leak (R10/INV9): after hit + decode + reset_slot, the free
    count returns to baseline minus the still-cached (ref_cnt == 0, hashed)
    blocks; every live ref_cnt == 0; block_table[slot] == []."""
    errors = []
    _clear_persistent_cache(runner)
    free_baseline = runner.block_pool.num_free_blocks()
    produce_slot, hit_slot = 0, 1

    # Produce (populates), reset, then hit + a few decode rounds, then reset.
    runner.mtp_prefill_with_cache([produce_slot], [prompt_ids])
    L = runner.reconcile_prefix_hit(prompt_ids)
    runner.reset_slot(produce_slot)

    hit_pr = runner.mtp_prefill_with_cache([hit_slot], [prompt_ids])
    anchor = hit_pr[hit_slot]["anchor"]
    drafts = hit_pr[hit_slot]["draft_tokens"]
    for _ in range(3):
        dec = runner.mtp_verify_and_commit(hit_slot, anchor, drafts)
        anchor, drafts = dec["next_anchor"], dec["next_draft_tokens"]
    runner.reset_slot(hit_slot)

    # The still-cached (ref_cnt==0, hashed) blocks are IN the free queue (a
    # freed-but-published block is re-queued at ref_cnt==0, hash retained so it
    # stays hit-able -- R10). So they count as free; the no-leak invariant is
    # simply free_after == free_baseline (no block stuck at ref_cnt>0).
    cached_unreferenced = sum(
        1 for b in runner.block_pool.blocks if b.ref_cnt == 0 and b.block_hash is not None
    )
    free_after = runner.block_pool.num_free_blocks()
    if free_after != free_baseline:
        errors.append(
            f"free count {free_after} != baseline {free_baseline} "
            f"(a block leaked: stuck at ref_cnt>0 or double-freed)"
        )
    leaked = [b.block_id for b in runner.block_pool.blocks if b.ref_cnt != 0]
    if leaked:
        errors.append(f"non-zero ref_cnt blocks remain after reset: {leaked[:10]}")
    for s in (produce_slot, hit_slot):
        if runner.block_table[s] != []:
            errors.append(f"block_table[{s}] not cleared after reset: {runner.block_table[s]}")

    _reset_all(runner, list(range(runner.num_slots)))
    return {"passed": not errors, "errors": errors, "L": L,
            "cached_unreferenced": cached_unreferenced}


def _make_prompt_ids(tok, kind: str, n_tokens: int) -> list[int]:
    """Build a prompt of exactly n_tokens tokens (natural-language or code)."""
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
    else:  # math -- a third distinct first block (no shared prefix with
        # nl/code), so a multi-prompt case can cold-produce several DIFFERENT
        # cache depths without one prompt hitting another's cached prefix.
        base = (
            "The sum of the first n natural numbers is n times n plus one "
            "divided by two. The product of the first n natural numbers is n "
            "factorial. A prime number has exactly two distinct positive "
            "divisors, one and itself. The golden ratio is one plus the square "
            "root of five, all divided by two. "
        )
    text = base * max(1, (n_tokens // 8) + 4)
    ids = tok.encode(text, add_special_tokens=False)
    while len(ids) < n_tokens:
        ids += tok.encode(base, add_special_tokens=False)
    return ids[:n_tokens]


def _replay_ref_check(runner, ref_slot, real_new_tokens, next_anchor) -> dict:
    """Independent cold single-slot reference replay of one round's real
    committed tokens (mtp_batch_verify_check.py / prefix_cache_fanout_check.py's
    established pattern): the reference slot consumes the REAL tokens the hit
    path committed and its own next-token prediction is compared against the
    hit's next anchor within the near-tie margin. Decouples each round from any
    prior round's possible near-tie divergence (R6)."""
    ref_logits = runner._forward(
        ref_slot,
        real_new_tokens,
        start_pos=runner.slot_kv_len[ref_slot],
        is_decode=(len(real_new_tokens) == 1),
    )
    ref_last = ref_logits[-1].float()
    ref_predicted_next = int(ref_last.argmax(dim=-1).item())
    ref_top1_logit = float(ref_last.max().item())
    ref_logit_for_choice = float(ref_last[next_anchor].item())
    near_tie_margin = ref_top1_logit - ref_logit_for_choice
    exact_match = ref_predicted_next == next_anchor
    return {
        "exact_match": exact_match,
        "near_tie_margin": near_tie_margin,
        "content_ok": exact_match or near_tie_margin < NEAR_TIE_LOGIT_MARGIN,
    }


def _gdn_layer0_committed_exact(runner, slot_a, slot_b) -> dict:
    """GDN layer-0 addressing proof on COMMITTED rows only: slot_a's and
    slot_b's layer-0 conv state must agree BYTEWISE on the committed
    token-position rows (masking the dead K spec-extension rows that a
    non-spec prefill leaves stale -- notes/2026-07-20-cold-prefill-rootcause-
    plan.md sec 3) and their ssm state must agree bytewise. A wrong-block /
    wrong-prefix read would show a large diff here that fp8 noise cannot
    mimic (R1)."""
    import torch

    from oracle.qwen36_vllm.direct_model_runner import _physical_slot

    gdn0 = runner.gdn_layer_names[0]
    conv_state, ssm_state = runner.kv_caches[gdn0]
    pa = _physical_slot(slot_a)
    pb = _physical_slot(slot_b)
    committed = conv_state[pa].shape[0] - runner.num_speculative_tokens
    conv_a = conv_state[pa][:committed].float()
    conv_b = conv_state[pb][:committed].float()
    ssm_a = ssm_state[pa].float()
    ssm_b = ssm_state[pb].float()
    return {
        "committed_rows": committed,
        "conv_exact": bool(torch.equal(conv_state[pa][:committed], conv_state[pb][:committed])),
        "conv_max_diff": (conv_a - conv_b).abs().max().item(),
        "ssm_exact": bool(torch.equal(ssm_state[pa], ssm_state[pb])),
        "ssm_max_diff": (ssm_a - ssm_b).abs().max().item(),
    }


def _run_multi_slot_ragged_hit(runner, tok) -> dict:
    """P3.3a -- the unified entrypoint's NEW batched ragged-hit path. Produce
    cached prefixes of DIFFERENT depths, then admit 3 hit slots with different
    L_s (ragged suffixes) in ONE ``mtp_prefill_with_cache`` call; assert each
    matches its own cold prefill (per-slot anchor exact + GDN-layer-0 committed-
    rows exact addressing proof + near-tie committed tokens over a few MTP
    rounds). Then a MIXED hit+cold batch (one hits, one cold L=0) in ONE call
    to exercise the hit/cold split + merge. Methodology mirrors this file's
    existing near-tie (R6) / layer-0-exact (R1) gates and prefix_cache_fanout_
    check.py's decode-replay reference pattern."""
    result: dict = {"checks": {}}
    block_size = runner.block_size
    n_decode_rounds = 6

    # Three prompts of different lengths => three different hit depths L_s =>
    # genuinely ragged hit suffixes in the single batched call.
    # Distinct first blocks (nl / code / math) so no producer hits another's
    # cached prefix -- each cold-produces its OWN completion checkpoint at a
    # DIFFERENT depth (96 / 192 / 288) => genuinely ragged hit suffixes.
    prompts = [
        _make_prompt_ids(tok, "nl", 100),    # L = block_align_down(99)  = 96
        _make_prompt_ids(tok, "code", 200),  # L = block_align_down(199) = 192
        _make_prompt_ids(tok, "math", 300),  # L = block_align_down(299) = 288
    ]
    _clear_persistent_cache(runner)

    # --- Phase A: cold-produce each prefix (populates the cache). The producer
    #     IS the cold reference: record its anchor + the cached [0, L) block ids
    #     (before reset), then reset (R10: blocks stay hit-able at ref_cnt==0). ---
    producer_slots = [0, 1, 2]
    cold_anchor = {}
    depth = {}
    cached_ids = {}
    for s, p in zip(producer_slots, prompts):
        pr = runner.mtp_prefill_with_cache([s], [p])
        cold_anchor[s] = pr[s]["anchor"]
        L = runner.reconcile_prefix_hit(p)
        depth[s] = L
        cached_ids[s] = list(runner.block_table[s][: L // block_size]) if L else []
        runner.reset_slot(s)
    result["depths"] = {str(s): depth[s] for s in producer_slots}
    if any(L < block_size for L in depth.values()):
        result["checks"]["produced_full_blocks"] = False
        result["error"] = "a producer did not publish a full cached block"
        return result
    result["checks"]["depths_differ"] = len(set(depth.values())) == len(prompts)

    # --- Phase B: ONE batched ragged hit over all three (different L_s). ---
    hit_slots = [3, 4, 5]
    hit_pr = runner.mtp_prefill_with_cache(hit_slots, prompts)
    hit_anchor = {hit_slots[i]: hit_pr[hit_slots[i]]["anchor"] for i in range(len(hit_slots))}

    # The hit batch must have ACTUALLY hit (every slot restored its own [0, L_s)
    # cached blocks), not silently fallen back to cold.
    hit_engaged = all(
        list(runner.block_table[hit_slots[i]])[: depth[producer_slots[i]] // block_size]
        == cached_ids[producer_slots[i]]
        and all(runner.block_pool.blocks[b].ref_cnt >= 1 for b in cached_ids[producer_slots[i]])
        for i in range(len(hit_slots))
    )
    result["checks"]["ragged_hit_actually_engaged"] = hit_engaged

    # Per-slot anchor match: each ragged hit slot's anchor == its cold anchor.
    result["checks"]["per_slot_anchor_match"] = all(
        hit_anchor[hit_slots[i]] == cold_anchor[producer_slots[i]] for i in range(len(hit_slots))
    )

    # --- Phase C: GDN-layer-0 committed-rows exact addressing proof per depth.
    #     Restore the checkpoint @ L_s into a fresh slot and compare against an
    #     independent fresh cold compute of [0, L_s) (same code path the
    #     producing phase-1 used) -- bytewise on committed rows (R1). ---
    layer0_ok = True
    layer0_detail = {}
    for i, (s_prod, p) in enumerate(zip(producer_slots, prompts)):
        restore_slot, fresh_slot = 6, 7
        _reset_all(runner, [restore_slot, fresh_slot])
        L = depth[s_prod]
        runner.restore_cached_prefix(restore_slot, p, L)
        runner._forward_batch(
            [fresh_slot], [p[:L]], [0], qo_len=L, commit=True,
            return_hidden=False, is_decode=False, logits_last_position_only=True,
        )
        cmp = _gdn_layer0_committed_exact(runner, restore_slot, fresh_slot)
        layer0_detail[str(L)] = cmp
        if not (cmp["conv_exact"] and cmp["ssm_exact"]):
            layer0_ok = False
        _reset_all(runner, [restore_slot, fresh_slot])
    result["checks"]["gdn_layer0_committed_exact"] = layer0_ok
    result["layer0_detail"] = layer0_detail

    # --- Phase D: near-tie committed tokens over a few MTP rounds. Cold-prefill
    #     one reference slot per hit slot (slots 0,1,2, freed in Phase A), decode
    #     the ragged hit batch, and replay each round's committed tokens on the
    #     matching reference (near-tie next-token, R6). ---
    ref_slots = [0, 1, 2]
    for ref_s, p in zip(ref_slots, prompts):
        runner.prefill(ref_s, p)  # plain cold prefill (cache present => dedup)
    cur_anchors = {hit_slots[i]: hit_anchor[hit_slots[i]] for i in range(len(hit_slots))}
    cur_drafts = {hit_slots[i]: hit_pr[hit_slots[i]]["draft_tokens"] for i in range(len(hit_slots))}
    decode_ok = True
    sync_ok = True
    for _r in range(n_decode_rounds):
        decisions = runner.mtp_verify_and_commit_batch(hit_slots, cur_anchors, cur_drafts)
        for i, s in enumerate(hit_slots):
            d = decisions[s]
            real_new_tokens = [cur_anchors[s]] + d["committed"][:-1]
            rep = _replay_ref_check(runner, ref_slots[i], real_new_tokens, d["next_anchor"])
            if not rep["content_ok"]:
                decode_ok = False
            if runner.slot_draft_sync_len[s] != runner.slot_kv_len[s]:
                sync_ok = False
            cur_anchors[s], cur_drafts[s] = d["next_anchor"], d["next_draft_tokens"]
    result["checks"]["near_tie_committed_tokens"] = decode_ok
    result["checks"]["inv4_draft_sync_len_matches_kv_len"] = sync_ok
    _reset_all(runner, hit_slots + ref_slots)

    # --- Phase E: MIXED hit+cold batch in ONE call (one hits, one cold L=0).
    #     Exercises the hit/cold split + merge in the unified entrypoint. ---
    _clear_persistent_cache(runner)
    mix_hit_prompt = _make_prompt_ids(tok, "nl", 200)   # will be cached -> hits
    mix_cold_prompt = _make_prompt_ids(tok, "code", 150)  # distinct -> L=0 (cold)
    runner.mtp_prefill_with_cache([0], [mix_hit_prompt])  # produce the hit prefix
    mix_L = runner.reconcile_prefix_hit(mix_hit_prompt)
    mix_cached_ids = list(runner.block_table[0][: mix_L // block_size]) if mix_L else []
    mix_cold_anchor_ref = runner.mtp_prefill_with_cache([2], [mix_cold_prompt])[2]["anchor"]
    runner.reset_slot(0)
    runner.reset_slot(2)
    # Admit hit + cold together.
    mixed = runner.mtp_prefill_with_cache([3, 4], [mix_hit_prompt, mix_cold_prompt])
    mix_hit_engaged = (
        mix_L >= block_size
        and list(runner.block_table[3][: mix_L // block_size]) == mix_cached_ids
        and all(runner.block_pool.blocks[b].ref_cnt >= 1 for b in mix_cached_ids)
    )
    # The cold slot must have taken the cold path: its blocks are freshly
    # allocated for mix_cold_prompt (distinct tokens => distinct hashes), so it
    # did NOT restore the hit's cached prefix blocks. (The cached prefix blocks
    # are ref_cnt>=1 from the hit slot, hence unallocatable by the cold slot.)
    mix_cold_is_cold = (
        bool(runner.block_table[4])
        and bool(mix_cached_ids)
        and runner.block_table[4][0] != mix_cached_ids[0]
    )
    result["checks"]["mixed_hit_engaged"] = mix_hit_engaged
    result["checks"]["mixed_cold_is_cold"] = mix_cold_is_cold
    mix_hit_ref_anchor = runner.prefill(6, mix_hit_prompt)
    result["checks"]["mixed_hit_anchor_match"] = mixed[3]["anchor"] == mix_hit_ref_anchor
    result["checks"]["mixed_cold_anchor_match"] = mixed[4]["anchor"] == mix_cold_anchor_ref
    result["checks"]["mixed_both_returned"] = (3 in mixed and 4 in mixed)
    _reset_all(runner, [3, 4, 6])

    return result



def _run_gpu_checks() -> dict:
    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner, build_vllm_config

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=6144,
        gpu_memory_utilization=0.55,
        speculative_config=SPECULATIVE_CONFIG,
    )
    # num_slots=8: slots 0-3 for produce/hit, 4-7 for independent cold
    # references / baselines. blocks_per_slot=96 (6144-token capacity) fits the
    # 5000-token prompt + decode. enable_cudagraph stays False -- this gate
    # targets persistent-hit CORRECTNESS (INV1/3/4, R1); CUDA-graph parity over
    # hit-populated tables is INV5, scoped to P3.3.
    runner = DirectModelRunner(
        vllm_config,
        num_slots=8,
        block_size=64,
        blocks_per_slot=96,
        enable_block_table=True,
        enable_prefix_cache=True,
        enable_persistent_prefix_cache=True,
        enable_cudagraph=False,
    )
    tok = AutoTokenizer.from_pretrained(MODEL)

    results: dict = {"cases": {}}
    overall = True

    # R1 addressing proof (natural-language, 100 tokens => L=96).
    nl_short = _make_prompt_ids(tok, "nl", 100)
    r1 = _run_r1_gdn_layer0_exact(runner, tok, nl_short)
    results["cases"]["r1_gdn_layer0_exact"] = r1
    for ok in r1["checks"].values():
        if not ok:
            overall = False

    # INV1/INV4 cold-vs-hit at several L, NL + code.
    inv1_specs = [
        ("nl_100", "nl", 100),     # hits 96 (partial: 4 tokens recomputed)
        ("nl_5000", "nl", 5000),   # hits block_align_down(4999)=4992
        ("code_300", "code", 300),  # code prompt, hits 288
    ]
    for label, kind, n in inv1_specs:
        prompt_ids = _make_prompt_ids(tok, kind, n)
        case = _run_inv1_case(runner, tok, prompt_ids, label)
        results["cases"][f"inv1_{label}"] = case
        for ok in case["checks"].values():
            if not ok:
                overall = False

    # INV3 mismatched prefix.
    base_ids = _make_prompt_ids(tok, "nl", 400)
    inv3 = _run_inv3_mismatched_prefix(runner, tok, base_ids)
    results["cases"]["inv3_mismatched_prefix"] = inv3
    for ok in inv3["checks"].values():
        if not ok:
            overall = False

    # INV6/INV7.
    inv67 = _run_inv6_inv7(runner, tok, nl_short)
    results["cases"]["inv6_inv7"] = inv67
    for ok in inv67["checks"].values():
        if not ok:
            overall = False

    # no_block_leak.
    leak = _run_no_block_leak(runner, tok, _make_prompt_ids(tok, "nl", 200))
    results["no_block_leak"] = leak
    if not leak["passed"]:
        overall = False

    # P3.3a: multi-slot ragged hit + mixed hit/cold batch (the unified
    # entrypoint's new batched hit path).
    multi = _run_multi_slot_ragged_hit(runner, tok)
    results["cases"]["multi_slot_ragged_hit"] = multi
    for ok in multi["checks"].values():
        if not ok:
            overall = False

    results["passed"] = overall
    return results


def main() -> int:
    print("=== prefix_cache_persistent_hit_check (P3.1 persistent-cache hit) ===")

    hcd = _check_hash_chain_determinism()
    print(f"hash_chain_determinism: {'PASS' if hcd['passed'] else 'FAIL'}")
    for e in hcd["errors"]:
        print(f"  - {e}")

    ik = _check_index_keepalive()
    print(f"index_keepalive: {'PASS' if ik['passed'] else 'FAIL'}")
    for e in ik["errors"]:
        print(f"  - {e}")

    gpu = _run_gpu_checks()

    r1 = gpu["cases"]["r1_gdn_layer0_exact"]
    print("r1_gdn_layer0_exact:")
    print(f"  L={r1.get('L')} max_conv_diff={r1.get('max_conv_diff')} "
          f"max_ssm_diff={r1.get('max_ssm_diff')}")
    for name, ok in r1["checks"].items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    for key in ("inv1_nl_100", "inv1_nl_5000", "inv1_code_300"):
        case = gpu["cases"][key]
        print(f"{key}: (prompt_len={case['prompt_len']}, L={case.get('L')})")
        print(f"  hit_actually_engaged={case['checks'].get('hit_actually_engaged')} "
              f"cold_anchor={case.get('cold_anchor')} hit_anchor={case.get('hit_anchor')} "
              f"plain_anchor={case.get('plain_anchor')}")
        for name, ok in case["checks"].items():
            print(f"  {name}: {'PASS' if ok else 'FAIL'}")
        print(f"  hit_acceptance_rate={case.get('hit_acceptance_rate'):.4f} "
              f"cold_acceptance_rate={case.get('cold_acceptance_rate'):.4f}")
        if case.get("first_mismatch") is not None:
            print(f"    first_mismatch: {case['first_mismatch']}")

    inv3 = gpu["cases"]["inv3_mismatched_prefix"]
    print(f"inv3_mismatched_prefix: B={inv3.get('B')} L_share={inv3.get('L_share')} "
          f"A_long={inv3.get('A_long')} L_long={inv3.get('L_long')}")
    for name, ok in inv3["checks"].items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    inv67 = gpu["cases"]["inv6_inv7"]
    print(f"inv6_inv7: published={inv67.get('published')} published2={inv67.get('published2')}")
    for name, ok in inv67["checks"].items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    leak = gpu["no_block_leak"]
    print(f"no_block_leak: {'PASS' if leak['passed'] else 'FAIL'} "
          f"(L={leak.get('L')}, cached_unreferenced={leak.get('cached_unreferenced')})")
    for e in leak["errors"]:
        print(f"  - {e}")

    multi = gpu["cases"]["multi_slot_ragged_hit"]
    print(f"multi_slot_ragged_hit: depths={multi.get('depths')}")
    for name, ok in multi["checks"].items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    for L, cmp in multi.get("layer0_detail", {}).items():
        print(f"  layer0@L={L}: conv_exact={cmp['conv_exact']} ssm_exact={cmp['ssm_exact']} "
              f"conv_max_diff={cmp['conv_max_diff']} ssm_max_diff={cmp['ssm_max_diff']}")

    overall = hcd["passed"] and ik["passed"] and gpu["passed"]
    print(f"\npassed: {str(overall).lower()}")
    print(f"=== overall: {'PASS' if overall else 'FAIL'} ===")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
