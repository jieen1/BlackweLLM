"""Correctness verification for the CUDA-graph wiring into the REAL
``mtp_verify_and_commit_batch`` entry point (``runtime/direct_model_runner.py``),
per ``notes/2026-07-17-post-ragged-round-next-steps.md``'s Phase 3 (verify
forward + K-1 draft steps, then a fast-iteration follow-up round -- full
precapture and step-0 draft-resync generalization) and its Phase 2
(native spec-decode GDN path, then this round's CUDA-graph reconciliation
of that path).

**History, briefly**: Phase 3 (2026-07-17/18) built ``CapturedBatchDecodeGraph``
around the then-current chunked GDN metadata path, including a
"recompute-forward graph-reuse" special case (a SEPARATE recompute
forward, needed on every partial-reject round, could reuse the
verify-graph cache at whatever qo_len the ragged recompute group's
committed_len happened to be). Phase 2 (2026-07-18) rewrote
``mtp_verify_and_commit_batch`` to use the real spec-decode GDN mechanism
(``build_gdn_metadata_spec_batch`` -- K+1 dedicated SSM rows,
acceptance-aware addressing) and eliminated the separate recompute
forward ENTIRELY (every slot's hidden states, regardless of accept/
reject, are now a ragged slice of the ONE verify forward) -- but that
rewrite initially fell back to eager dispatch for the verify step itself,
since ``CapturedBatchDecodeGraph`` was still built for the old mechanism.
**This round** reconciles the two: ``CapturedBatchDecodeGraph``'s qo_len>1
branch now fills GDN metadata via the SAME spec-decode mechanism
(``static_spec_state_indices``/``static_num_accepted_tokens``, refilled
per replay from real slot ids/accept-reject outcomes -- see that class's
docstring), so verify-step CUDA-graph replay works again. Since there is
no more separate recompute forward at all (with or without graphs), the
old "recompute-forward graph-reuse" special case and its own
``qo_len in 1..k`` precapture range are GONE too -- verify now only ever
replays at exactly ``qo_len=k+1``, for every slot, every round,
regardless of accept/reject outcome.

Unlike ``cudagraph_eager_parity_check.py``/``cudagraph_mtp_regression.py``
(which drive ``CapturedBatchDecodeGraph`` directly via ``.capture()``/
``.replay()`` in isolation), this test constructs a runner with
``enable_cudagraph=True`` and drives the REAL production call path
(``mtp_prefill_batch`` -> repeated ``mtp_verify_and_commit_batch``) --
exercising every graph-or-eager dispatch decision inside that method and
``_mtp_sync_and_propose_batch``, not just the underlying primitives.

Methodology: the project's established "independent-reference-replay"
pattern (``mtp_batch_verify_check.py``/``mtp_ragged_recompute_verify_check.py``)
-- feed each round's REAL committed tokens into an independent single-slot
``_forward`` reference call every round, comparing its own greedy argmax
against the batched path's ``next_anchor`` within a near-tie tolerance.
This decouples one round's possible near-tie tie-break flip from
corrupting the next round's check, and needs only ONE model load (one
runner; ``ref_slots`` on the always-eager single-slot path serve as the
oracle for ``mtp_slots`` on the graph-enabled batched path).

Coverage, and which code path each round is designed to exercise:

0-1. Organic rounds -- whatever the model actually does (usually full
     accept given this project's ~72% measured per-position acceptance
     rate). Exercises the common-case verify-graph + step-0-graph +
     draft-step-graph path.
2.   Mixed forced-reject (2 of 4 slots) -- the SAME verify-graph replay
     covers both the organically-accepting and forced-reject slots in
     one call (no more per-outcome branching at all, graph or eager);
     the draft resync afterward still splits by each slot's own real
     committed_len.
3.   Fully ragged forced-reject (every slot rejects at a DIFFERENT
     position) -- confirms the single verify-graph replay handles a
     genuinely ragged accept/reject OUTCOME correctly even though the
     replay's own INPUT shape (qo_len=k+1) never varies.
4.   Organic again, immediately after a ragged-reject round -- confirms
     the verify graph's static buffers (especially the new
     ``static_num_accepted_tokens``, which must correctly reflect the
     PREVIOUS round's real per-slot outcome) are refilled correctly
     round to round, not left stale.
5.   Single-slot forced reject.
6.   UNIFORM forced-reject, committed_len=2 for all 4 slots -- exercises
     the draft model's OWN step-0 resync graph at qo_len=2
     (``_get_draft_step_graph(4, 2)``, unrelated to the target-model
     verify graph, which is unaffected by committed_len since it always
     replays at qo_len=k+1 regardless).
7.   UNIFORM forced-reject, committed_len=1 for all 4 slots -- same
     draft-step-graph coverage at qo_len=1.
Shrink.  Drop from 4 active slots to 2 (simulating some requests finishing
     early, the real W1-S tail pattern) -- confirms a SECOND (batch_size=2)
     verify graph and draft-step graph get captured/used correctly.

Also confirms, via ``replay_count`` (not just cache-key presence, which
precapture alone would already satisfy regardless of whether anything
ever replays that shape):
- the verify graph is actually replayed at both batch_size=4 and 2 --
  the real pass/fail gate this round restores (was informational-only
  between the Phase 2 rewrite and this round's reconciliation);
- the draft-model step-0 graph is actually replayed (the generalization
  that, during development, needed an explicit ``_MAX_DECODE_QO_LEN``
  guard -- mtp_prefill_batch's own step-0 call uses num_new_tokens=
  prompt_len, e.g. 4096, which is also "uniform" and would otherwise have
  been forced through the decode-kernel dispatch it was never validated
  for; this test's own prefill call is the direct regression check for
  that guard actually firing);
- the draft-model qo_len=1 continuation-step graph is actually replayed.

Usage:
    python -m benchmarks.mtp_verify_cudagraph_check
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3
SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"}

NEAR_TIE_LOGIT_MARGIN = 2.0

FOUR_PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Germany is",
    "The capital of Italy is",
]


def _reset_if_needed(runner, slots):
    for slot in slots:
        if runner.slot_kv_len[slot] != 0 or runner.slot_draft_sync_len[slot] != 0:
            runner.reset_slot(slot)


def _rotate_pool(runner, scratch_slots: list[int], tok, sponge_reserve: int, rotations: int = 15) -> list[int]:
    """P1 (notes/prefix-cache-design.md sec 5) fragmentation recipe for this
    test: shrink the shared ``BlockPool``'s immediately-available working
    set down to ``sponge_reserve`` blocks (sponging up -- allocating and
    holding, returned here for the caller to free back once the real
    traffic below has completed -- everything else), then rotate that
    small working set via real allocate/reset/reallocate cycles
    (``DirectModelRunner.prefill``/``reset_slot``) on ``scratch_slots``
    (never the real ``mtp_slots``/``ref_slots`` this test drives) BEFORE
    any real slot touches the pool. Every real slot's subsequent growth
    (``mtp_prefill_batch``'s initial allocation, each round's further
    growth, ``ref_slots``' own independent single-slot growth) then draws
    from this already-rotated small set instead of this test's
    generously-sized, otherwise-untouched ascending virgin supply -- the
    real scenario this phase exists to prove the block-table + CUDA-graph
    path (verify graph AND draft-step graph) tolerates, not just P0's
    trivial always-contiguous case. See
    ``benchmarks/cudagraph_decode_regression.py``'s identically-motivated
    ``_fragment_pool_via_churn`` for the same recipe applied to a single
    big allocation instead of a whole multi-slot test's worth of organic
    growth."""
    pool = runner.block_pool
    free_now = pool.num_free_blocks()
    sponge_n = max(0, free_now - sponge_reserve)
    sponge = pool.allocate(sponge_n) if sponge_n > 0 else []
    tiny_ids = tok.encode("Q", add_special_tokens=False)
    for _ in range(rotations):
        for s in scratch_slots:
            runner.prefill(s, tiny_ids)
            runner.reset_slot(s)
    return sponge


def _is_contiguous(table: list[int]) -> bool:
    return all(table[i + 1] - table[i] == 1 for i in range(len(table) - 1))


def _decoy_at(drafts: list[int], position: int) -> list[int]:
    """Corrupt drafts[position] with a value guaranteed to differ from the
    real draft (and from itself across positions) so
    determine_accept_reject_batch rejects at EXACTLY this position,
    forcing committed_len = position + 1."""
    out = list(drafts)
    decoy = 100 + position
    while decoy in drafts:
        decoy += 1
    out[position] = decoy
    return out


def _ref_check(runner, ref_slot, real_new_tokens, mtp_next_anchor) -> dict:
    ref_logits = runner._forward(
        ref_slot,
        real_new_tokens,
        start_pos=runner.slot_kv_len[ref_slot],
        is_decode=(len(real_new_tokens) == 1),
    )
    ref_last = ref_logits[-1].float()
    ref_predicted_next = int(ref_last.argmax(dim=-1).item())
    ref_top1_logit = float(ref_last.max().item())
    ref_logit_for_mtp_choice = float(ref_last[mtp_next_anchor].item())
    near_tie_margin = ref_top1_logit - ref_logit_for_mtp_choice
    exact_match = ref_predicted_next == mtp_next_anchor
    return {
        "ref_predicted_next_matches_mtp_next_anchor": exact_match,
        "near_tie_margin": near_tie_margin,
        "content_ok_within_near_tie_tolerance": exact_match or near_tie_margin < NEAR_TIE_LOGIT_MARGIN,
    }


def _run_once(enable_block_table: bool = False) -> dict:
    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    tok = AutoTokenizer.from_pretrained(MODEL)
    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=2048,
        gpu_memory_utilization=0.6,
        speculative_config=SPECULATIVE_CONFIG,
    )
    # mtp_slots (real, graph-enabled batched path) + ref_slots (independent
    # eager single-slot reference oracle) + CapturedBatchDecodeGraph's/
    # CapturedMTPDraftStepGraph's own internally reserved warmup slots
    # (batch_size up to 4 -> 4 slots) = 3*4 = 12. block_size/blocks_per_slot
    # kept small (matches cudagraph_mtp_regression.py's own convention) --
    # this test only needs correctness, not production capacity.
    #
    # P1 (notes/prefix-cache-design.md sec 5): when enable_block_table, TWO
    # extra scratch slots (12, 13) are reserved purely for this test's own
    # deliberate pool-fragmentation churn (_rotate_pool) -- placed BETWEEN
    # the real slots [0-7] and CapturedBatchDecodeGraph's/
    # CapturedMTPDraftStepGraph's own warmup range (which shifts to stay
    # the LAST 4 slots of num_slots either way), so they never collide.
    num_slots = 12 + (2 if enable_block_table else 0)
    runner = DirectModelRunner(
        vllm_config,
        num_slots=num_slots,
        block_size=64,
        blocks_per_slot=32,
        enable_cudagraph=True,
        enable_block_table=enable_block_table,
    )

    mtp_slots = [0, 1, 2, 3]
    ref_slots = [4, 5, 6, 7]
    scratch_slots = [8, 9] if enable_block_table else []
    prompt_ids_per_slot = [tok.encode(p, add_special_tokens=False) for p in FOUR_PROMPTS]
    _reset_if_needed(runner, mtp_slots + ref_slots)

    # P1: fragment the shared BlockPool BEFORE any real slot (mtp_slots/
    # ref_slots) ever touches it, so their entire real-traffic growth below
    # (initial prefill, every round's further growth, the shrunk-batch
    # tail) draws from an already-rotated small working set instead of
    # this test's generously-sized, otherwise-untouched ascending virgin
    # supply -- see _rotate_pool's docstring. sponge_reserve is sized
    # generously above this test's real total demand (8 real slots x up to
    # ~3 blocks each over the whole run, plus headroom) so nothing runs out
    # mid-test; freed back at the very end.
    pool_sponge: list[int] = []
    if enable_block_table:
        pool_sponge = _rotate_pool(runner, scratch_slots=scratch_slots, tok=tok, sponge_reserve=48)

    # mtp_prefill_batch's own step-0 resync call uses num_new_tokens=
    # prompt_len (a large, uniform value) -- the direct regression check
    # for the _MAX_DECODE_QO_LEN guard that keeps this from being
    # mistakenly routed through the draft-step decode-kernel graph.
    prefill_result = runner.mtp_prefill_batch(mtp_slots, prompt_ids_per_slot)
    anchors = {s: prefill_result[s]["anchor"] for s in mtp_slots}
    drafts = {s: prefill_result[s]["draft_tokens"] for s in mtp_slots}
    for i, s in enumerate(mtp_slots):
        ref_first = runner.prefill(ref_slots[i], prompt_ids_per_slot[i])
        if ref_first != anchors[s]:
            return {"passed": False, "error": f"prefill anchor mismatch for slot {s}"}

    per_slot_rounds: dict[int, list[dict]] = {s: [] for s in mtp_slots}

    # Force patterns per round: None = organic (whatever the model does),
    # dict = {slot: reject_position} for slots to force-reject this round.
    force_patterns = [
        None,  # round 0: organic
        None,  # round 1: organic
        {mtp_slots[0]: 0, mtp_slots[2]: 1},  # round 2: mixed
        {mtp_slots[0]: 0, mtp_slots[1]: 1, mtp_slots[2]: 2, mtp_slots[3]: 0},  # round 3: fully ragged
        None,  # round 4: organic again, right after a ragged-recompute round
        {mtp_slots[1]: 2},  # round 5: single-slot forced reject
        # round 6: uniform reject, committed_len=2 for all 4 -- recompute
        # forward reuses the verify-graph cache instead of eager.
        {mtp_slots[0]: 1, mtp_slots[1]: 1, mtp_slots[2]: 1, mtp_slots[3]: 1},
        # round 7: uniform reject, committed_len=1 for all 4 -- same reuse
        # path at qo_len=1 (caught a real flatten bug during development).
        {mtp_slots[0]: 0, mtp_slots[1]: 0, mtp_slots[2]: 0, mtp_slots[3]: 0},
    ]

    for r, pattern in enumerate(force_patterns):
        submitted = {}
        for s in mtp_slots:
            if pattern and s in pattern:
                submitted[s] = _decoy_at(drafts[s], pattern[s])
            else:
                submitted[s] = drafts[s]

        decisions = runner.mtp_verify_and_commit_batch(mtp_slots, anchors, submitted)

        for i, s in enumerate(mtp_slots):
            decision = decisions[s]
            real_new_tokens = [anchors[s]] + decision["committed"][:-1]
            ref_report = _ref_check(runner, ref_slots[i], real_new_tokens, decision["next_anchor"])
            per_slot_rounds[s].append(
                {
                    "round": r,
                    "pattern": pattern.get(s) if pattern else None,
                    "num_accepted": decision["num_accepted"],
                    "kv_len_matches_draft_sync_len": runner.slot_kv_len[s] == runner.slot_draft_sync_len[s],
                    **ref_report,
                }
            )
            anchors[s], drafts[s] = decision["next_anchor"], decision["next_draft_tokens"]

    verify_graph_b4 = runner._verify_graphs.get((4, K + 1))
    # Round 6 forces ALL 4 slots to reject with a UNIFORM committed_len=2
    # -- this deterministically drives _mtp_sync_and_propose_batch's
    # step-0 resync through _get_draft_step_graph(4, 2) (the qo_len>1
    # generalization), unlike the organic full-accept case's own
    # qo_len=k+1 step-0 shape, which depends on ALL 4 slots organically
    # fully-accepting in the SAME round simultaneously (~2% per round
    # given this project's own measured ~72% per-position acceptance
    # rate) -- too unreliable to gate a deterministic test on. The
    # target-model verify graph itself is UNAFFECTED by committed_len
    # (always replays at qo_len=k+1 regardless -- see module docstring),
    # so this round exercises only the draft-side graph, not a distinct
    # verify-side shape.
    draft_step0_qo2_graph = runner._draft_step_graphs.get((4, 2))
    # (4, 1) is hit by BOTH the K-1 continuation loop (every round,
    # regardless of step-0's own qo_len) and round 7's step-0
    # (committed_len=1 uniform) -- guaranteed to be exercised every round.
    draft_cont_graph_b4 = runner._draft_step_graphs.get((4, 1))

    # --- Shrinking-batch check: drop to 2 active slots (simulating slots
    # 2/3 finishing early), continue for a few more rounds -- exercises a
    # SECOND (batch_size=2) graph via _get_verify_graph/_get_draft_step_graph. ---
    shrunk_slots = mtp_slots[:2]
    shrunk_ref = ref_slots[:2]
    for r in range(3):
        decisions = runner.mtp_verify_and_commit_batch(
            shrunk_slots, {s: anchors[s] for s in shrunk_slots}, {s: drafts[s] for s in shrunk_slots}
        )
        for i, s in enumerate(shrunk_slots):
            decision = decisions[s]
            real_new_tokens = [anchors[s]] + decision["committed"][:-1]
            ref_report = _ref_check(runner, shrunk_ref[i], real_new_tokens, decision["next_anchor"])
            per_slot_rounds[s].append(
                {
                    "round": f"shrunk-{r}",
                    "num_accepted": decision["num_accepted"],
                    "kv_len_matches_draft_sync_len": runner.slot_kv_len[s] == runner.slot_draft_sync_len[s],
                    **ref_report,
                }
            )
            anchors[s], drafts[s] = decision["next_anchor"], decision["next_draft_tokens"]

    verify_graph_b2 = runner._verify_graphs.get((2, K + 1))

    per_slot_ok = {}
    for s in mtp_slots:
        rounds = per_slot_rounds[s]
        per_slot_ok[s] = all(rd["kv_len_matches_draft_sync_len"] for rd in rounds) and all(
            rd["content_ok_within_near_tie_tolerance"] for rd in rounds
        )

    # Coverage gates: not just "was this shape captured" (precapture alone
    # satisfies that regardless of use) but "was it actually REPLAYED"
    # (replay_count > 0), i.e. the real round loop actually took this code
    # path at least once, not silently falling back to eager throughout.
    #
    # 2026-07-18, Phase 2 CUDA-graph reconciliation: the two verify-side
    # entries are REAL pass/fail gates again (they were informational-only
    # between the Phase 2 GDN-mechanism rewrite and this round's
    # reconciliation of CapturedBatchDecodeGraph with the new spec-decode
    # metadata -- see module docstring). The old "recompute-reuse"
    # entries are REMOVED entirely, not just demoted: Phase 2 eliminated
    # the separate recompute forward completely, so there is no longer
    # any (batch_size, qo_len<k+1) verify-graph shape for anything to
    # reuse, with or without graphs -- _precapture_verify_graphs no
    # longer even builds those shapes (confirmed by their absence from
    # verify_graph_cache_keys below).
    coverage = {
        "verify_graph_batch4_replayed": bool(verify_graph_b4 and verify_graph_b4.replay_count > 0),
        "verify_graph_batch2_replayed": bool(verify_graph_b2 and verify_graph_b2.replay_count > 0),
        "draft_step0_qo2_graph_replayed": bool(draft_step0_qo2_graph and draft_step0_qo2_graph.replay_count > 0),
        "draft_continuation_graph_replayed": bool(draft_cont_graph_b4 and draft_cont_graph_b4.replay_count > 0),
    }

    fragmentation_proof: dict | None = None
    if enable_block_table:
        # P1: release the sponge held by _rotate_pool now that every real
        # slot's growth this test drives is done, then confirm the
        # deliberate pre-fragmentation actually left at least one real
        # slot's own block_table non-contiguous -- the concrete proof this
        # phase's fragmented-CUDA-graph re-run must produce, not just a
        # claim. (The graph-replay CORRECTNESS proof under that
        # fragmentation is everything above -- per_slot_ok/coverage --
        # already having passed while those slots' block_tables held
        # these non-contiguous ids the whole time.)
        runner.block_pool.free(pool_sponge)
        real_tables = {s: list(runner.block_table[s]) for s in mtp_slots + ref_slots}
        non_contiguous_slots = [s for s, t in real_tables.items() if len(t) > 1 and not _is_contiguous(t)]
        fragmentation_proof = {
            "real_block_tables": real_tables,
            "non_contiguous_slots": non_contiguous_slots,
            "ok": len(non_contiguous_slots) > 0,
        }

    passed = bool(all(per_slot_ok.values()) and all(coverage.values()))
    if fragmentation_proof is not None:
        passed = passed and fragmentation_proof["ok"]
    return {
        "passed": passed,
        "per_slot_ok": per_slot_ok,
        "coverage": coverage,
        "fragmentation_proof": fragmentation_proof,
        "verify_graph_cache_keys": sorted(str(k) for k in runner._verify_graphs.keys()),
        "draft_step_graph_cache_keys": sorted(str(k) for k in runner._draft_step_graphs.keys()),
        "per_slot_rounds": per_slot_rounds,
    }


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    # P1 (notes/prefix-cache-design.md sec 5): default False preserves this
    # script's original behavior byte-for-byte. When passed, constructs the
    # runner with enable_block_table=True and deliberately fragments the
    # shared BlockPool (_rotate_pool) before any real slot touches it, so
    # the whole real-traffic round battery above -- verify-graph AND
    # draft-step-graph replay alike -- runs against provably non-contiguous
    # block tables (INV5), not just P0's trivial always-contiguous case.
    parser.add_argument("--enable-block-table", action="store_true")
    args = parser.parse_args()

    result = _run_once(enable_block_table=args.enable_block_table)
    summary = {
        "passed": result["passed"],
        "per_slot_ok": result.get("per_slot_ok"),
        "coverage": result.get("coverage"),
        "fragmentation_proof": (
            {k: v for k, v in result["fragmentation_proof"].items() if k != "real_block_tables"}
            if result.get("fragmentation_proof")
            else None
        ),
        "verify_graph_cache_keys": result.get("verify_graph_cache_keys"),
        "draft_step_graph_cache_keys": result.get("draft_step_graph_cache_keys"),
    }
    print(json.dumps(summary, indent=2, default=str))
    if not result["passed"]:
        print(json.dumps(result, indent=2, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
