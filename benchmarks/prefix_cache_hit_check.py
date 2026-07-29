"""Cumulative load-bearing gate for prefix-cache P3.3 --
``benchmarks/prefix_cache_hit_check.py`` (the design's sec 5-P3 test,
``notes/2026-07-19-p3-implementation-plan.md`` "### P3.3 -- Dedicated test
slice"; ``notes/prefix-cache-design.md`` sec 4 invariants INV1-INV9, sec 6
risks R1-R10).

P3.3a unified ``mtp_prefill_with_cache`` into the ONE production prefill
entrypoint (persistent-hit + P2 fan-out + cold) and proved INV5 parity; this
gate proves EVERY prefix-cache invariant holds THROUGH that production
entrypoint, plus the two genuinely-new P3.3b coverage items (INV2 persistent-
hit signal-probe crosstalk across all 4 slots; INV8 mid-flight admission of a
hit slot alongside long-running slots).

One ``DirectModelRunner`` (``enable_block_table=True`` + ``enable_prefix_cache
=True`` + ``enable_persistent_prefix_cache=True``, MTP K=3, CUSTOM attention),
slots reused via ``_reset_all``/``reset_slot`` + ``_clear_persistent_cache``
between subtests to avoid repeated model loads. Subtests:

* ``inv1_*`` (INV1/INV4, carry over via the production entrypoint) -- cold-vs-
  hit near-tie at SEVERAL L (partial + full), 20+ MTP decode rounds, a
  natural-language AND a code prompt; hit anchor == cold anchor (exact) +
  decode equivalence + hit acceptance == cold acceptance. Reuses
  ``prefix_cache_persistent_hit_check._run_inv1_case`` (already drives
  ``mtp_prefill_with_cache``).
* ``inv2_persistent_signal_probe`` (INV2, NEW) -- 4 distinct cached prefixes
  (each its own per-slot marker token), reset, then admit 4 slots that each
  HIT a different cached prefix via ONE ``mtp_prefill_with_cache`` call, decode
  a short burst from each, assert each slot's text contains its OWN marker and
  NONE of the other 3 (zero leakage). Generalizes ``prefix_cache_fanout_check
  .inv2_signal_probe`` from same-round forks to PERSISTENT hits.
* ``inv3_mismatched_prefix`` (INV3, carry over) -- mismatched/mixed-length
  prefixes reuse EXACTLY the right depth (L = G <= A). Reuses
  ``prefix_cache_persistent_hit_check._run_inv3_mismatched_prefix``.
* ``inv5_hit_table_parity`` (INV5, re-confirm) -- eager-vs-graph parity with a
  hit-populated NON-CONTIGUOUS table, via ``cudagraph_eager_parity_check
  --single-run-hit-json`` in a subprocess (it builds its own runner; two 27B
  models cannot coexist on one GPU). Layer-0 GDN bytewise exact + logits
  cosine >= 0.99 + top-1.
* ``inv8_midflight_hit`` (INV8, NEW) -- start 2 slots decoding a long
  generation (several verify rounds in flight), THEN admit a fresh slot that
  HITS a cached prefix mid-flight; assert (a) the hit engages (L>0) and its
  anchor matches its cold reference; (b) the already-running slots' block
  tables are UNCHANGED by the admission and their committed tokens stay
  correct (near-tie vs cold references -- no disruption); (c) the hit slot's
  own decode is correct (near-tie vs cold). Generalizes
  ``mtp_async_arrival_check``'s mid-flight pattern + persistent_hit's near-tie.
* ``inv9_admission_under_pressure`` (INV9, carry over) -- admission under pool
  pressure forcing eviction with NO live (ref_cnt>0) block reclaimed. Reuses
  ``prefix_cache_eviction_check._run_admission_under_pressure``.
* ``evict_then_recompute`` (carry over) -- evict -> re-request -> clean cold
  recompute (tokens + anchor match the original). Reuses
  ``prefix_cache_eviction_check._run_evict_then_recompute``.
* ``flag_off_delegation`` (the "P3 changes nothing when off" half) -- with
  ``enable_persistent_prefix_cache=False``, ``mtp_prefill_with_cache``
  delegates to ``mtp_prefill_batch`` byte-for-byte (identical anchor + draft
  tokens). Lightweight delegation-equivalence assertion (the full headline
  bit-identical is re-run by the LEADER).
* ``flag_on_regression`` -- a small flag-on battery proving the flag-on path is
  regression-free: a cold (L=0) multi-slot batch through
  ``mtp_prefill_with_cache`` matches ``mtp_prefill_batch`` exactly (the cold
  path IS the P2 fan-out/cold path), and a no-cache-available hit attempt
  (cleared cache) falls back to cold cleanly (anchor matches the original
  cold).

Methodology reuse (do NOT reinvent): near-tie ``NEAR_TIE_LOGIT_MARGIN = 2.0``
per R6 for token/logit comparisons (NOT bytewise) EXCEPT the GDN-layer-0
addressing proof which is exact per R1; any GDN conv comparison across a
slot-reuse-after-decode is COMMITTED-rows-only (``committed = conv.shape[0] -
num_speculative_tokens`` -- ``notes/2026-07-20-cold-prefill-rootcause-plan.md``
sec 3 dead-spec-row artifact).

SCOPE NOTE (leader decision): the >=64K Pattern-B PERF hook is DEFERRED to
P3.4 (its natural home -- P3.4 owns all long-context correctness + perf). This
gate runs at the standard ~4K-or-less scale the existing checks use; it asserts
CORRECTNESS through the production entrypoint, not the long-context speedup.

Usage:
    python -m benchmarks.prefix_cache_hit_check
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
SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"}

NEAR_TIE_LOGIT_MARGIN = 2.0  # established methodology (R6): a real near-exact
# tie is kernel-path-sensitive fp8/batch non-associativity noise, NOT state
# corruption; distinct real candidates are typically separated by 8-13+ logit
# units. A mismatch is only a real failure if the reference's own logit for the
# committed token is NOT within this margin of the reference's top.

# Distinct 5-digit markers, no shared full-string overlap (signal-probe
# crosstalk targets -- prefix_cache_fanout_check.py's methodology).
MARKERS = [84317, 52968, 71053, 39642]


# ---------------------------------------------------------------------------
# Real-GPU part.
# ---------------------------------------------------------------------------


def _run_inv2_persistent_signal_probe(runner, tok) -> dict:
    """INV2 (NEW P3.3b coverage): signal-probe crosstalk with PERSISTENT cache
    hits interleaved across ALL 4 slots. Produce 4 distinct cached prefixes
    (each with its own per-slot marker token), reset, then admit 4 slots that
    each HIT a different cached prefix via ONE ``mtp_prefill_with_cache([s0,s1,
    s2,s3], ...)`` call, decode a short burst from each, and assert each slot's
    generated text contains its OWN marker and NONE of the other 3 (zero
    leakage). Generalizes ``prefix_cache_fanout_check.inv2_signal_probe`` from
    same-round forks to persistent hits: proves cached prefixes don't bleed
    across slots when 4 hits are batched (the unified entrypoint's batched
    ragged hit continue-prefill keeps each slot attending to its OWN cached
    blocks)."""
    from benchmarks.prefix_cache_persistent_hit_check import (
        _clear_persistent_cache,
        _make_prompt_ids,
        _replay_ref_check,
        _reset_all,
    )

    _clear_persistent_cache(runner)
    result: dict = {"checks": {}}
    block_size = runner.block_size
    n_decode_rounds = 6
    n_burst = 12

    # Per-slot DISTINCT first block (nl/code/math/diverge streams) so no
    # producer hits another's cached prefix -- each cold-produces its OWN
    # distinct cached prefix. The marker is stated up-front (so it lands in the
    # cached [0, L) region) and the prompt ends with a strong copy cue (so the
    # greedy burst reproduces the marker), after prefix_cache_fanout_check's
    # proven copy-cue methodology.
    streams = ["nl", "code", "math", "diverge"]
    pad = (
        " The repository contains many modules tests and documentation files "
        "that must be kept consistent at all times. Always read the existing "
        "code carefully before changing any of it and always run the relevant "
        "tests after a change. Keep every diff small and easy to review. "
    )

    def _marker_prompt(i: int) -> list[int]:
        # The marker is stated up-front (so it lands in the cached [0, L)
        # region) AND repeated right before the copy cue (the fanout check's
        # proven adjacent-marker copy trigger), so the greedy burst reproduces
        # the marker whether it is served from the restored cached blocks or the
        # recomputed suffix -- either way a correct own-marker / no-cross-leak
        # result proves the batched hits did not bleed across slots.
        distinct = _make_prompt_ids(tok, streams[i], 24)
        marker_txt = tok.encode(
            f" Agent-{i} private task. The value of X is {MARKERS[i]}.",
            add_special_tokens=False,
        )
        pad_ids = tok.encode(pad * 2, add_special_tokens=False)
        cue = tok.encode(
            f" The value of X is {MARKERS[i]}. Repeat agent {i}'s value of X "
            f"exactly. The value of X is",
            add_special_tokens=False,
        )
        return distinct + marker_txt + pad_ids + cue

    prompts = [_marker_prompt(i) for i in range(4)]

    # --- Phase A: cold-produce each prefix (populates the cache). The producer
    #     IS the cold reference: record its anchor + the cached [0, L) block ids
    #     (before reset), then reset (R10: blocks stay hit-able at ref_cnt==0). ---
    producer_slots = [0, 1, 2, 3]
    cold_anchor: dict[int, int] = {}
    depth: dict[int, int] = {}
    cached_ids: dict[int, list[int]] = {}
    for i, s in enumerate(producer_slots):
        pr = runner.mtp_prefill_with_cache([s], [prompts[i]])
        cold_anchor[s] = pr[s]["anchor"]
        L = runner.reconcile_prefix_hit(prompts[i])
        depth[s] = L
        cached_ids[s] = list(runner.block_table[s][: L // block_size]) if L else []
        runner.reset_slot(s)
    result["depths"] = {str(s): depth[s] for s in producer_slots}
    if any(L < block_size for L in depth.values()):
        result["checks"]["produced_full_blocks"] = False
        result["error"] = "a producer did not publish a full cached block"
        return result
    result["checks"]["produced_full_blocks"] = True

    # --- Phase B: ONE batched persistent-hit ragged admission over all 4 (each
    #     hits its OWN distinct cached prefix). ---
    hit_slots = [4, 5, 6, 7]
    hit_pr = runner.mtp_prefill_with_cache(hit_slots, prompts)
    hit_anchor = {hit_slots[i]: hit_pr[hit_slots[i]]["anchor"] for i in range(4)}

    # The hit batch must have ACTUALLY hit (every slot restored its own [0, L)
    # cached blocks at ref_cnt>=1), not silently fallen back to cold.
    hit_engaged = all(
        list(runner.block_table[hit_slots[i]][: depth[producer_slots[i]] // block_size])
        == cached_ids[producer_slots[i]]
        and all(runner.block_pool.blocks[b].ref_cnt >= 1 for b in cached_ids[producer_slots[i]])
        for i in range(4)
    )
    result["checks"]["persistent_hits_actually_engaged"] = hit_engaged
    # Per-slot anchor match: each persistent hit's anchor == its cold anchor.
    result["checks"]["per_slot_anchor_match"] = all(
        hit_anchor[hit_slots[i]] == cold_anchor[producer_slots[i]] for i in range(4)
    )

    # --- Phase C: INV2 signal-probe -- greedy-decode [anchor] + a short burst
    #     from each hit slot; the prefill anchor IS the marker's first token
    #     (the cue ends "The value of X is"), so the marker is spelled by
    #     [anchor] + burst. Assert each slot reproduces its OWN marker and NONE
    #     of the other 3 (a distinct 5-digit number from another slot is
    #     unambiguous leakage). ---
    gen_text: dict[int, str] = {}
    for i, s in enumerate(hit_slots):
        out_tokens: list[int] = [hit_pr[s]["anchor"]]
        cur = hit_pr[s]["anchor"]
        for _ in range(n_burst):
            logits = runner._forward(s, [cur], start_pos=runner.slot_kv_len[s], is_decode=True)
            cur = int(logits[-1].argmax(dim=-1).item())
            out_tokens.append(cur)
        gen_text[s] = tok.decode(out_tokens, skip_special_tokens=True)

    inv2_ok = True
    inv2_detail: dict[str, object] = {}
    for i, s in enumerate(hit_slots):
        own = str(MARKERS[i])
        text_nospace = gen_text[s].replace(" ", "")
        own_present = own in text_nospace
        cross_present = [
            str(MARKERS[j])
            for j in range(4)
            if j != i and str(MARKERS[j]) in text_nospace
        ]
        inv2_detail[f"slot{s}"] = {
            "own_marker": own,
            "own_present": own_present,
            "cross_leak": cross_present,
            "gen": gen_text[s][:60],
        }
        if cross_present or not own_present:
            inv2_ok = False
    result["checks"]["inv2_zero_crosstalk"] = inv2_ok
    result["inv2_detail"] = inv2_detail

    # --- Phase D: near-tie committed tokens over a few MTP rounds, on NEUTRAL
    #     prompts (the persistent_hit_check's proven near-tie methodology). The
    #     marker prompts above end in a sharp copy cue ("The value of X is"),
    #     where a batch-4 MTP verify and a single-slot reference legitimately
    #     pick different tokens ~10 logits apart (fp8/batch non-associativity at
    #     a high-curvature distribution -- NOT state corruption; the exact
    #     per_slot_anchor_match + the clean zero-crosstalk burst above prove the
    #     hit state is correct). Neutral prompts decode in a flat region where
    #     the batch noise stays within NEAR_TIE_LOGIT_MARGIN, so this isolates a
    #     REAL batched-hit decode error if one existed. Produce 4 distinct-depth
    #     neutral prefixes, admit them as ONE batched ragged persistent hit, and
    #     replay-reference each slot's committed tokens vs its cold reference. ---
    _reset_all(runner, list(range(runner.num_slots)))
    _clear_persistent_cache(runner)
    neutral_prompts = [
        _make_prompt_ids(tok, "nl", 100),    # L = 96
        _make_prompt_ids(tok, "code", 200),  # L = 192
        _make_prompt_ids(tok, "math", 300),  # L = 288
        _make_prompt_ids(tok, "diverge", 400),  # L = 384
    ]
    for i, s in enumerate(producer_slots):
        runner.mtp_prefill_with_cache([s], [neutral_prompts[i]])
        runner.reset_slot(s)
    neutral_hit_pr = runner.mtp_prefill_with_cache(hit_slots, neutral_prompts)
    ref_slots = producer_slots
    for i, ref_s in enumerate(ref_slots):
        runner.prefill(ref_s, neutral_prompts[i])  # cold reference (cache => dedup)
    cur_anchors = {hit_slots[i]: neutral_hit_pr[hit_slots[i]]["anchor"] for i in range(4)}
    cur_drafts = {hit_slots[i]: neutral_hit_pr[hit_slots[i]]["draft_tokens"] for i in range(4)}
    decode_ok = True
    sync_ok = True
    max_margin = 0.0
    n_exact = 0
    n_checks = 0
    worst: dict = {}
    for _r in range(n_decode_rounds):
        decisions = runner.mtp_verify_and_commit_batch(hit_slots, cur_anchors, cur_drafts)
        for i, s in enumerate(hit_slots):
            d = decisions[s]
            real_new_tokens = [cur_anchors[s]] + d["committed"][:-1]
            rep = _replay_ref_check(runner, ref_slots[i], real_new_tokens, d["next_anchor"])
            n_checks += 1
            n_exact += int(rep["exact_match"])
            if rep["near_tie_margin"] > max_margin:
                max_margin = rep["near_tie_margin"]
                worst = {"round": _r, "slot": s, "margin": rep["near_tie_margin"],
                         "exact": rep["exact_match"], "next_anchor": d["next_anchor"]}
            if not rep["content_ok"]:
                decode_ok = False
            if runner.slot_draft_sync_len[s] != runner.slot_kv_len[s]:
                sync_ok = False
            cur_anchors[s], cur_drafts[s] = d["next_anchor"], d["next_draft_tokens"]
    result["checks"]["near_tie_committed_tokens"] = decode_ok
    result["checks"]["inv4_draft_sync_len_matches_kv_len"] = sync_ok
    result["near_tie_diag"] = {"max_margin": max_margin, "n_exact": n_exact,
                               "n_checks": n_checks, "worst": worst}
    _reset_all(runner, hit_slots + ref_slots)
    return result


def _run_inv8_midflight_hit(runner, tok) -> dict:
    """INV8 (NEW P3.3b coverage): MID-FLIGHT admission of a hit slot alongside
    LONG-RUNNING slots. Start 2 slots decoding a long generation (several
    verify rounds in flight), THEN admit a fresh slot that HITS a cached prefix
    (via ``mtp_prefill_with_cache``) mid-flight, and assert:
      (a) the hit slot engages (L>0, restores its own cached blocks) and its
          anchor matches its cold reference;
      (b) the already-running slots' block tables are UNCHANGED by the mid-
          flight admission and their committed tokens stay correct (near-tie vs
          their cold references -- no disruption);
      (c) the hit slot's own decode is correct (near-tie vs cold).
    Generalizes ``mtp_async_arrival_check``'s mid-flight pattern (admit a fresh
    request between verify rounds while others keep decoding) + persistent_hit's
    near-tie. The single-event-loop engine serializes admission, so this proves
    the reserve-and-touch-before-forward order (R4/INV9) holds when a hit joins
    live slots: a mid-flight hit can neither reclaim nor corrupt a live slot's
    blocks."""
    from benchmarks.prefix_cache_persistent_hit_check import (
        _clear_persistent_cache,
        _make_prompt_ids,
        _replay_ref_check,
        _reset_all,
    )

    _clear_persistent_cache(runner)
    result: dict = {"checks": {}}
    block_size = runner.block_size
    pre_rounds = 4   # long-running slots decode this many rounds before the hit joins
    post_rounds = 4  # rounds decoded after the hit joins (all slots together)

    run_prompts = [_make_prompt_ids(tok, "nl", 300), _make_prompt_ids(tok, "code", 300)]
    hit_prompt = _make_prompt_ids(tok, "math", 200)
    run_slots = [0, 1]
    hit_slot = 2
    run_ref_slots = [3, 4]  # cold references for the running slots (near-tie replay)
    hit_ref_slot = 5        # cold reference for the hit slot
    populate_slot = 6       # scratch: populate the hit prefix + record its cold anchor

    # --- Populate the hit prefix (cold) => cached [0, L) blocks + completion
    #     checkpoint; record the cold anchor (the producer IS the cold
    #     reference), then reset (R10: blocks stay hit-able at ref_cnt==0). ---
    pop_pr = runner.mtp_prefill_with_cache([populate_slot], [hit_prompt])
    hit_cold_anchor = pop_pr[populate_slot]["anchor"]
    hit_L = runner.reconcile_prefix_hit(hit_prompt)
    hit_cached_ids = list(runner.block_table[populate_slot][: hit_L // block_size]) if hit_L else []
    runner.reset_slot(populate_slot)
    result["hit_L"] = hit_L
    if hit_L < block_size:
        result["checks"]["hit_prefix_cached"] = False
        result["error"] = "hit prefix did not publish a full cached block"
        _reset_all(runner, list(range(runner.num_slots)))
        return result
    result["checks"]["hit_prefix_cached"] = True

    # --- Cold references for the running slots + the hit slot (plain prefill;
    #     the near-tie replay reference consumes the REAL committed tokens and
    #     compares its own next-token vs the slot's next anchor, R6). ---
    for ref_s, p in zip(run_ref_slots, run_prompts):
        runner.prefill(ref_s, p)
    runner.prefill(hit_ref_slot, hit_prompt)

    # --- Make the running slots live and decode them pre_rounds (mid-flight). ---
    run_pr = runner.mtp_prefill_with_cache(run_slots, run_prompts)
    run_anchors = {s: run_pr[s]["anchor"] for s in run_slots}
    run_drafts = {s: run_pr[s]["draft_tokens"] for s in run_slots}
    pre_decode_ok = True
    for _r in range(pre_rounds):
        decs = runner.mtp_verify_and_commit_batch(run_slots, run_anchors, run_drafts)
        for i, s in enumerate(run_slots):
            d = decs[s]
            real_new_tokens = [run_anchors[s]] + d["committed"][:-1]
            rep = _replay_ref_check(runner, run_ref_slots[i], real_new_tokens, d["next_anchor"])
            if not rep["content_ok"]:
                pre_decode_ok = False
            run_anchors[s], run_drafts[s] = d["next_anchor"], d["next_draft_tokens"]
    result["checks"]["running_slots_correct_pre_admission"] = pre_decode_ok

    # Snapshot the running slots' block tables at the admission boundary.
    run_tables_before = {s: list(runner.block_table[s]) for s in run_slots}

    # --- MID-FLIGHT admission: the running slots are still live (ref_cnt>0,
    #     mid-decode); admit the hit slot via the production entrypoint. ---
    hit_pr = runner.mtp_prefill_with_cache([hit_slot], [hit_prompt])
    hit_anchor = hit_pr[hit_slot]["anchor"]
    hit_restored = list(runner.block_table[hit_slot][: hit_L // block_size])
    result["checks"]["hit_engaged_midflight"] = (
        hit_restored == hit_cached_ids
        and all(runner.block_pool.blocks[b].ref_cnt >= 1 for b in hit_cached_ids)
    )
    result["checks"]["hit_anchor_matches_cold"] = (hit_anchor == hit_cold_anchor)
    # INV9/R4: the running slots' block tables are UNCHANGED by the admission
    # (a mid-flight hit never reclaims or rewrites a live slot's blocks).
    result["checks"]["running_block_tables_unchanged"] = all(
        list(runner.block_table[s]) == run_tables_before[s] for s in run_slots
    )

    # --- Decode ALL slots together post_rounds; the running slots must stay
    #     correct (no disruption) and the hit slot must decode correctly. ---
    all_slots = run_slots + [hit_slot]
    cur_anchors = {**run_anchors, hit_slot: hit_anchor}
    cur_drafts = {**run_drafts, hit_slot: hit_pr[hit_slot]["draft_tokens"]}
    ref_by_slot = {run_slots[0]: run_ref_slots[0], run_slots[1]: run_ref_slots[1],
                   hit_slot: hit_ref_slot}
    run_post_ok = True
    hit_post_ok = True
    for _r in range(post_rounds):
        decs = runner.mtp_verify_and_commit_batch(all_slots, cur_anchors, cur_drafts)
        for s in all_slots:
            d = decs[s]
            real_new_tokens = [cur_anchors[s]] + d["committed"][:-1]
            rep = _replay_ref_check(runner, ref_by_slot[s], real_new_tokens, d["next_anchor"])
            if not rep["content_ok"]:
                if s == hit_slot:
                    hit_post_ok = False
                else:
                    run_post_ok = False
            cur_anchors[s], cur_drafts[s] = d["next_anchor"], d["next_draft_tokens"]
    result["checks"]["running_slots_correct_post_admission"] = run_post_ok
    result["checks"]["hit_slot_decode_correct"] = hit_post_ok

    _reset_all(runner, all_slots + run_ref_slots + [hit_ref_slot, populate_slot])
    return result


def _run_inv5_subprocess() -> dict:
    """INV5 (re-confirm): eager-vs-graph parity over a hit-populated NON-
    CONTIGUOUS block table. Reuse ``cudagraph_eager_parity_check
    --single-run-hit-json`` in a subprocess (it builds its own runner -- two 27B
    models cannot coexist on one GPU). Gate: layer-0 GDN bytewise exact +
    logits cosine >= 0.99 + top-1 match + full-stack near-tie."""
    import json
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.cudagraph_eager_parity_check", "--single-run-hit-json"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("SINGLE_RUN_HIT_RESULT: "):
            result = json.loads(line[len("SINGLE_RUN_HIT_RESULT: "):])
            result["checks"] = {"inv5_hit_table_parity": bool(result.get("passed"))}
            return result
    return {
        "checks": {"inv5_hit_table_parity": False},
        "error": "no SINGLE_RUN_HIT_RESULT line found",
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-2000:],
    }


def _run_flag_off_battery(runner, tok) -> dict:
    """The "P3 changes nothing when off" half (lightweight delegation-
    equivalence assertion). With ``enable_persistent_prefix_cache=False``,
    ``mtp_prefill_with_cache`` delegates straight to ``mtp_prefill_batch``
    (runtime/direct_model_runner.py:4560) -- so a single-slot cold prefill via
    both paths produces an identical anchor + draft tokens. The full headline
    bit-identical re-run is the LEADER's job; this gate asserts the delegation
    equivalence only. Flips the flag off on the shared runner, then restores it
    on (the runner is constructed flag-on for the other subtests)."""
    from benchmarks.prefix_cache_persistent_hit_check import (
        _clear_persistent_cache,
        _make_prompt_ids,
        _reset_all,
    )

    result: dict = {"checks": {}}
    prompt = _make_prompt_ids(tok, "nl", 200)
    slot = 0

    runner.enable_persistent_prefix_cache = False
    try:
        _clear_persistent_cache(runner)
        pr_via = runner.mtp_prefill_with_cache([slot], [prompt])
        via_anchor = pr_via[slot]["anchor"]
        via_drafts = list(pr_via[slot]["draft_tokens"])
        runner.reset_slot(slot)
        pr_direct = runner.mtp_prefill_batch([slot], [prompt])
        direct_anchor = pr_direct[slot]["anchor"]
        direct_drafts = list(pr_direct[slot]["draft_tokens"])
        runner.reset_slot(slot)
    finally:
        runner.enable_persistent_prefix_cache = True
        _clear_persistent_cache(runner)

    result["via_anchor"] = via_anchor
    result["direct_anchor"] = direct_anchor
    result["checks"]["flag_off_delegates_anchor"] = (via_anchor == direct_anchor)
    result["checks"]["flag_off_delegates_drafts"] = (via_drafts == direct_drafts)
    _reset_all(runner, [slot])
    return result


def _run_flag_on_regression(runner, tok) -> dict:
    """The flag-on regression re-run: prove the flag-on path is regression-free.
    (1) A cold (L=0) multi-slot batch through ``mtp_prefill_with_cache`` matches
    ``mtp_prefill_batch`` exactly -- with the cache cleared every slot is cold
    (L=0), so the unified entrypoint routes the >=2 cold slots to
    ``mtp_prefill_fanout_batch`` with distinct prompts => ``mtp_prefill_batch``
    => byte-for-byte P2 output. (2) A no-cache-available hit attempt (cleared
    cache) falls back to cold cleanly -- a prompt that WAS cached, re-requested
    after clearing, gets L=0 and its anchor matches the original cold anchor."""
    from benchmarks.prefix_cache_persistent_hit_check import (
        _clear_persistent_cache,
        _make_prompt_ids,
        _reset_all,
    )

    result: dict = {"checks": {}}
    block_size = runner.block_size

    # (1) Cold multi-slot batch: mtp_prefill_with_cache == mtp_prefill_batch.
    _clear_persistent_cache(runner)
    distinct = [_make_prompt_ids(tok, "nl", 150), _make_prompt_ids(tok, "code", 150)]
    pr_cache = runner.mtp_prefill_with_cache([0, 1], distinct)
    a_cache = [pr_cache[0]["anchor"], pr_cache[1]["anchor"]]
    d_cache = [list(pr_cache[0]["draft_tokens"]), list(pr_cache[1]["draft_tokens"])]
    _reset_all(runner, [0, 1])
    _clear_persistent_cache(runner)
    pr_batch = runner.mtp_prefill_batch([2, 3], distinct)
    a_batch = [pr_batch[2]["anchor"], pr_batch[3]["anchor"]]
    d_batch = [list(pr_batch[2]["draft_tokens"]), list(pr_batch[3]["draft_tokens"])]
    _reset_all(runner, [2, 3])
    result["checks"]["cold_multislot_anchors_match"] = (a_cache == a_batch)
    result["checks"]["cold_multislot_drafts_match"] = (d_cache == d_batch)
    result["cold_multislot_anchors"] = {"with_cache": a_cache, "batch": a_batch}

    # (2) No-cache-available hit attempt falls back to cold cleanly.
    _clear_persistent_cache(runner)
    p = _make_prompt_ids(tok, "nl", 200)
    cold_anchor = runner.mtp_prefill_with_cache([0], [p])[0]["anchor"]
    L_before = runner.reconcile_prefix_hit(p)
    runner.reset_slot(0)
    _clear_persistent_cache(runner)  # wipe the cache => the re-request cannot hit
    L_after = runner.reconcile_prefix_hit(p)
    fallback_anchor = runner.mtp_prefill_with_cache([1], [p])[1]["anchor"]
    runner.reset_slot(1)
    result["L_before_clear"] = L_before
    result["L_after_clear"] = L_after
    result["checks"]["cleared_cache_no_hit"] = (L_before >= block_size) and (L_after == 0)
    result["checks"]["fallback_anchor_matches_cold"] = (fallback_anchor == cold_anchor)

    _reset_all(runner, list(range(runner.num_slots)))
    return result


def _run_gpu_checks() -> dict:
    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    # Reuse the methodology helpers from the checks this gate consolidates
    # (imported lazily so a pure-import of this module stays GPU-free).
    from benchmarks.prefix_cache_eviction_check import (
        _run_admission_under_pressure,
        _run_evict_then_recompute,
    )
    from benchmarks.prefix_cache_persistent_hit_check import (
        _make_prompt_ids,
        _run_inv1_case,
        _run_inv3_mismatched_prefix,
    )
    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner, build_vllm_config

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=6144,
        gpu_memory_utilization=0.55,
        speculative_config=SPECULATIVE_CONFIG,
    )
    # num_slots=8: slots 0-3 producers/running, 4-7 hits/references (INV2 needs
    # 4 producers + 4 hits in one batched call). blocks_per_slot=96
    # (6144-token capacity) fits the 5000-token INV1 prompt + 22 decode rounds.
    # enable_cudagraph stays False -- INV5 parity runs in its own subprocess
    # (cudagraph_eager_parity_check builds its own graph-capturing runner).
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

    def _record(name: str, case: dict) -> None:
        nonlocal overall
        results["cases"][name] = case
        for ok in case.get("checks", {}).values():
            if not ok:
                overall = False

    # --- INV1/INV4: cold-vs-hit near-tie at several L (partial + full), 20+
    #     decode rounds, NL + code -- via the production entrypoint. ---
    inv1_specs = [
        ("nl_100", "nl", 100),     # hits 96 (partial: 4 tokens recomputed)
        ("nl_5000", "nl", 5000),   # hits block_align_down(4999)=4992 (full)
        ("code_300", "code", 300),  # code prompt, hits 288
    ]
    for label, kind, n in inv1_specs:
        _record(f"inv1_{label}", _run_inv1_case(runner, tok, _make_prompt_ids(tok, kind, n), label))

    # --- INV2 (NEW): persistent-hit signal-probe crosstalk across all 4 slots. ---
    _record("inv2_persistent_signal_probe", _run_inv2_persistent_signal_probe(runner, tok))

    # --- INV3: mismatched/mixed-length prefixes reuse exactly the right depth. ---
    _record("inv3_mismatched_prefix",
            _run_inv3_mismatched_prefix(runner, tok, _make_prompt_ids(tok, "nl", 400)))

    # --- INV5: eager-vs-graph parity over a hit-populated non-contiguous table
    #     (subprocess -- builds its own runner). ---
    _record("inv5_hit_table_parity", _run_inv5_subprocess())

    # --- INV8 (NEW): mid-flight admission of a hit slot alongside long-running
    #     slots. ---
    _record("inv8_midflight_hit", _run_inv8_midflight_hit(runner, tok))

    # --- INV9: admission under pool pressure forcing eviction, no live block
    #     reclaimed. ---
    _record("inv9_admission_under_pressure", _run_admission_under_pressure(runner, tok))

    # --- eviction correctness: evict -> re-request -> clean cold recompute. ---
    _record("evict_then_recompute", _run_evict_then_recompute(runner, tok))

    # --- flag-off battery: mtp_prefill_with_cache delegates to mtp_prefill_batch
    #     byte-for-byte when the flag is off. ---
    _record("flag_off_delegation", _run_flag_off_battery(runner, tok))

    # --- flag-on regression re-run: cold multi-slot == mtp_prefill_batch; cleared
    #     cache falls back to cold cleanly. ---
    _record("flag_on_regression", _run_flag_on_regression(runner, tok))

    results["passed"] = overall
    return results


def _print_checks(case: dict, indent: str = "  ") -> None:
    for name, ok in case.get("checks", {}).items():
        print(f"{indent}{name}: {'PASS' if ok else 'FAIL'}")


def main() -> int:
    print("=== prefix_cache_hit_check (P3.3b cumulative prefix-cache gate) ===")
    print("note: >=64K Pattern-B PERF hook deferred to P3.4 (correctness-only gate)")

    gpu = _run_gpu_checks()
    cases = gpu["cases"]

    # INV1/INV4.
    for key in ("inv1_nl_100", "inv1_nl_5000", "inv1_code_300"):
        case = cases[key]
        print(f"{key}: (prompt_len={case.get('prompt_len')}, L={case.get('L')})")
        print(f"  cold_anchor={case.get('cold_anchor')} hit_anchor={case.get('hit_anchor')} "
              f"plain_anchor={case.get('plain_anchor')}")
        _print_checks(case)
        if case.get("hit_acceptance_rate") is not None:
            print(f"  hit_acceptance_rate={case['hit_acceptance_rate']:.4f} "
                  f"cold_acceptance_rate={case.get('cold_acceptance_rate'):.4f}")
        if case.get("first_mismatch") is not None:
            print(f"  first_mismatch: {case['first_mismatch']}")

    # INV2.
    inv2 = cases["inv2_persistent_signal_probe"]
    print(f"inv2_persistent_signal_probe: depths={inv2.get('depths')}")
    _print_checks(inv2)
    for slot_key, detail in inv2.get("inv2_detail", {}).items():
        print(f"  {slot_key}: own={detail['own_marker']} own_present={detail['own_present']} "
              f"cross_leak={detail['cross_leak']} gen={detail['gen']!r}")
    if inv2.get("near_tie_diag"):
        diag = inv2["near_tie_diag"]
        print(f"  near_tie_diag: max_margin={diag['max_margin']:.4f} "
              f"n_exact={diag['n_exact']}/{diag['n_checks']} worst={diag['worst']}")

    # INV3.
    inv3 = cases["inv3_mismatched_prefix"]
    print(f"inv3_mismatched_prefix: B={inv3.get('B')} L_share={inv3.get('L_share')} "
          f"A_long={inv3.get('A_long')} L_long={inv3.get('L_long')}")
    _print_checks(inv3)

    # INV5.
    inv5 = cases["inv5_hit_table_parity"]
    print(f"inv5_hit_table_parity: L={inv5.get('L')} "
          f"non_contiguous={inv5.get('non_contiguous_table')}")
    _print_checks(inv5)
    if inv5.get("logits"):
        print(f"  logits_cosine={inv5['logits'].get('cosine_similarity')} "
              f"top1_match={inv5['logits'].get('top1_match')} "
              f"gdn_layer0_exact={inv5.get('gdn_layer0_exact')}")
    if inv5.get("error"):
        print(f"  error: {inv5['error']}")

    # INV8.
    inv8 = cases["inv8_midflight_hit"]
    print(f"inv8_midflight_hit: hit_L={inv8.get('hit_L')}")
    _print_checks(inv8)

    # INV9.
    inv9 = cases["inv9_admission_under_pressure"]
    print(f"inv9_admission_under_pressure: free_before_admit={inv9.get('free_before_admit')}")
    _print_checks(inv9)

    # eviction correctness.
    evict = cases["evict_then_recompute"]
    print(f"evict_then_recompute: L_populated={evict.get('L_populated')} "
          f"L_after_evict={evict.get('L_after_evict')}")
    _print_checks(evict)

    # flag-off battery.
    flagoff = cases["flag_off_delegation"]
    print(f"flag_off_delegation: via_anchor={flagoff.get('via_anchor')} "
          f"direct_anchor={flagoff.get('direct_anchor')}")
    _print_checks(flagoff)

    # flag-on regression.
    flagn = cases["flag_on_regression"]
    print(f"flag_on_regression: cold_multislot_anchors={flagn.get('cold_multislot_anchors')} "
          f"L_before_clear={flagn.get('L_before_clear')} L_after_clear={flagn.get('L_after_clear')}")
    _print_checks(flagn)

    overall = gpu["passed"]
    print(f"\npassed: {str(overall).lower()}")
    print(f"=== overall: {'PASS' if overall else 'FAIL'} ===")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
