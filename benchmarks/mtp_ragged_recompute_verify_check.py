"""Correctness verification for the 2026-07-17 ragged-qo_len generalization
of the recompute-fallback path (``mtp_verify_and_commit_batch``'s
NEEDS-RECOMPUTE group is now batched in ONE call across slots with
DIFFERENT real committed lengths, instead of a per-slot Python loop).

Three checks, following this project's established methodology
(strict batch=1 equivalence + per-round independent-reference-replay
numerical-twin + forced mixed-stage), specifically targeting the new
ragged code path:

0. Batch=1 forced-reject equivalence: a SINGLE slot forced to reject at
   a specific draft position (so the recompute path is definitely
   exercised, not just organically maybe-hit) is driven through BOTH the
   single-slot ``mtp_verify_and_commit`` and the batched
   ``mtp_verify_and_commit_batch`` (batch of 1). Originally (through
   2026-07-17) this required committed tokens/next_anchor/bookkeeping to
   match BIT-EXACTLY, no near-tie tolerance -- valid at the time because
   both entry points shared the exact SAME underlying mechanism (chunked
   GDN metadata + snapshot/restore/recompute-forward), differing only in
   call convention, so any divergence was necessarily a real bug (this
   is what caught the real ``decode_qo_len`` formula gap that round).

   **2026-07-18, Phase 2**: ``mtp_verify_and_commit_batch`` now uses a
   GENUINELY DIFFERENT mechanism (the real spec-decode GDN path -- K+1
   dedicated SSM rows, acceptance-aware addressing, no snapshot/restore/
   recompute-forward -- see ``runtime/direct_model_runner.py``'s
   ``mtp_verify_and_commit_batch``/``verify_batch_spec``/
   ``build_gdn_metadata_spec_batch``) while ``mtp_verify_and_commit``
   (the singular/looped sibling) deliberately still uses the OLD chunked
   mechanism. Bit-exact agreement between two genuinely different
   numerical paths is no longer a valid invariant -- confirmed by a
   direct, isolated measurement (not just inferred from the failure):
   running this exact scenario found round 3 (force_position=0) diverges
   to committed token 271 (looped) vs. 198 (batched) for the identical
   prior state, and a direct raw-logit dump at that exact point showed
   ``looped margin (top1 - logit[198]) = 0.125``, ``batched margin (top1
   - logit[271]) = 0.0`` -- both far under this project's established
   ``NEAR_TIE_LOGIT_MARGIN=2.0`` -- with a full-vocab max_abs_diff of
   6.84 and mean_abs_diff of 0.80 between the two mechanisms' raw logit
   vectors at that position, unremarkable and consistent with the same
   class of bf16 batching/kernel-path noise this project already accepts
   elsewhere (e.g. attention's kv_split_size noise, the "271 vs 198"
   near-tie ``mtp_batch_verify_check.py`` independently documented for
   this SAME prompt). This is a genuine near-tie tie-break flip, not a
   correctness bug. Fixed by adopting this file's own already-established
   per-round independent-reference-replay methodology (checks 1/2 below)
   for check 0 too: each mechanism's own committed choice is validated
   against its OWN untouched reference slot (near-tie tolerant); a
   divergence between the two mechanisms is only a FAILURE if either
   side's own choice fails ITS OWN reference check (an "unexplained"
   mismatch) -- a near-tie-explained divergence is recorded but does not
   fail the test. Cross-mechanism kv_len/draft_sync_len equality is
   likewise no longer required once a real near-tie divergence has
   legitimately split the two trajectories (their commit COUNTS can
   still differ once content differs) -- each mechanism's own
   kv_len==draft_sync_len internal self-consistency is the real
   bookkeeping gate now.

1. Multi-slot RAGGED recompute: 4 slots, each forced to reject at a
   DIFFERENT draft position (committed_len 1, 2, 3, and 1 again),
   batched into ONE ``mtp_verify_and_commit_batch`` call -- this is the
   genuinely new scenario (previously this always degenerated into 4
   separate single-slot calls; now it must be one ragged batched call).
   Verified via per-round independent-reference-replay (feed each slot's
   own real committed tokens into an independent single-slot reference
   forward every round), over several rounds, to catch any cross-slot
   GDN-state contamination that would only show up after multiple
   rounds of reuse.

2. True mixed-stage: 2 slots forced to reject at DIFFERENT positions
   (ragged recompute) while the other 2 organically full-accept, all in
   the SAME ``mtp_verify_and_commit_batch`` call -- the combinatorial
   case the coordinator specifically asked to cover (full-accept +
   ragged-recompute together, not just ragged-recompute alone).

Usage:
    python -m benchmarks.mtp_ragged_recompute_verify_check
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
NUM_ROUNDS = 6
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


def _decoy_at(drafts: list[int], position: int) -> list[int]:
    """Corrupt drafts[position] with a value guaranteed to differ from the
    real draft (and from itself across the 3 possible positions) so
    ``determine_accept_reject`` rejects at EXACTLY this position,
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


# ---------------------------------------------------------------------------
# Check 0: strict batch=1 forced-reject equivalence.
# ---------------------------------------------------------------------------


def _check_batch1_forced_reject_equivalence(runner, tok) -> dict:
    looped_slot, batched_slot = 0, 1
    # 2026-07-18, Phase 2: independent, untouched reference slots -- one
    # per mechanism -- so a genuine near-tie divergence between looped and
    # batched (see module docstring) can still be judged correct/incorrect
    # on its own merits, instead of only ever being comparable to the
    # OTHER mechanism's (now not-necessarily-matching) choice.
    looped_ref_slot, batched_ref_slot = 2, 3
    prompt = FOUR_PROMPTS[0]
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    _reset_if_needed(runner, [looped_slot, batched_slot, looped_ref_slot, batched_ref_slot])

    pr_looped = runner.mtp_prefill(looped_slot, prompt_ids)
    pr_batched = runner.mtp_prefill_batch([batched_slot], [prompt_ids])[batched_slot]
    anchor_l, drafts_l = pr_looped["anchor"], pr_looped["draft_tokens"]
    anchor_b, drafts_b = pr_batched["anchor"], pr_batched["draft_tokens"]
    if anchor_l != anchor_b or drafts_l != drafts_b:
        return {"passed": False, "error": "prefill mismatch before forced-reject rounds even started"}

    ref_first_l = runner.prefill(looped_ref_slot, prompt_ids)
    ref_first_b = runner.prefill(batched_ref_slot, prompt_ids)
    if ref_first_l != anchor_l or ref_first_b != anchor_b:
        return {"passed": False, "error": "reference prefill anchor mismatch"}

    exact_mismatches = []
    near_tie_divergences = []
    ref_failures = []
    landed_elsewhere_notes = []
    bookkeeping_notes = []
    for r in range(NUM_ROUNDS):
        force_position = r % K  # cycles through 0, 1, 2 -- committed_len 1, 2, 3
        submitted_l = _decoy_at(drafts_l, force_position)
        submitted_b = _decoy_at(drafts_b, force_position)

        decision_l = runner.mtp_verify_and_commit(looped_slot, anchor_l, submitted_l)
        decision_b = runner.mtp_verify_and_commit_batch(
            [batched_slot], {batched_slot: anchor_b}, {batched_slot: submitted_b}
        )[batched_slot]

        # Informational only (not a pass/fail signal): whether the forced
        # decoy landed at the exact intended position depends on the
        # model's own organic behavior at earlier positions too (an
        # organic earlier reject is real, expected model behavior, not a
        # runtime bug) -- see the module docstring.
        if decision_l["num_accepted"] != force_position:
            landed_elsewhere_notes.append({"round": r, "mechanism": "looped", "num_accepted": decision_l["num_accepted"]})
        if decision_b["num_accepted"] != force_position:
            landed_elsewhere_notes.append({"round": r, "mechanism": "batched", "num_accepted": decision_b["num_accepted"]})

        # Independent-reference-replay validation, per mechanism (this
        # file's own established check1/check2 pattern) -- the real
        # correctness bar now: each mechanism's own committed choice must
        # check out against ITS OWN untouched reference slot.
        real_new_tokens_l = [anchor_l] + decision_l["committed"][:-1]
        real_new_tokens_b = [anchor_b] + decision_b["committed"][:-1]
        ref_report_l = _ref_check(runner, looped_ref_slot, real_new_tokens_l, decision_l["next_anchor"])
        ref_report_b = _ref_check(runner, batched_ref_slot, real_new_tokens_b, decision_b["next_anchor"])
        if not ref_report_l["content_ok_within_near_tie_tolerance"]:
            ref_failures.append({"round": r, "mechanism": "looped", **ref_report_l})
        if not ref_report_b["content_ok_within_near_tie_tolerance"]:
            ref_failures.append({"round": r, "mechanism": "batched", **ref_report_b})

        if decision_l["committed"] != decision_b["committed"]:
            entry = {
                "round": r,
                "force_position": force_position,
                "looped_committed": decision_l["committed"],
                "batched_committed": decision_b["committed"],
            }
            # A divergence between the two mechanisms is acceptable ONLY
            # if both sides' own committed choice independently checks out
            # against their own reference -- otherwise this is a real,
            # unexplained mismatch (a genuine bug, not a near-tie).
            if ref_report_l["content_ok_within_near_tie_tolerance"] and ref_report_b["content_ok_within_near_tie_tolerance"]:
                near_tie_divergences.append(entry)
            else:
                exact_mismatches.append(entry)

        bookkeeping_notes.append(
            {
                "round": r,
                "kv_len_matches_across_mechanisms": runner.slot_kv_len[looped_slot] == runner.slot_kv_len[batched_slot],
                "looped_self_consistent": runner.slot_kv_len[looped_slot] == runner.slot_draft_sync_len[looped_slot],
                "batched_self_consistent": runner.slot_kv_len[batched_slot] == runner.slot_draft_sync_len[batched_slot],
            }
        )

        anchor_l, drafts_l = decision_l["next_anchor"], decision_l["next_draft_tokens"]
        anchor_b, drafts_b = decision_b["next_anchor"], decision_b["next_draft_tokens"]

    # The real pass/fail gate (2026-07-18, Phase 2): no UNEXPLAINED
    # mismatch (one that fails independent reference validation on either
    # side), no reference-check failure standing alone, and each
    # mechanism's OWN kv_len/draft_sync_len bookkeeping stays internally
    # self-consistent every round. Cross-mechanism kv_len equality is
    # demoted to an informational note -- see module docstring for why
    # that is no longer guaranteed once a real near-tie divergence has
    # legitimately split the two trajectories.
    self_consistent = all(b["looped_self_consistent"] and b["batched_self_consistent"] for b in bookkeeping_notes)
    passed = bool(len(exact_mismatches) == 0 and len(ref_failures) == 0 and self_consistent)
    return {
        "passed": passed,
        "num_rounds": NUM_ROUNDS,
        "exact_mismatches": exact_mismatches,
        "near_tie_divergences": near_tie_divergences,
        "ref_failures": ref_failures,
        "landed_elsewhere_notes": landed_elsewhere_notes,
        "self_consistent": self_consistent,
        "bookkeeping_notes": bookkeeping_notes,
    }


# ---------------------------------------------------------------------------
# Check 1: multi-slot ragged recompute (all 4 slots forced to DIFFERENT
# committed_len values, batched into ONE call).
# ---------------------------------------------------------------------------


def _check_ragged_recompute(runner, tok) -> dict:
    mtp_slots = [0, 1, 2, 3]
    ref_slots = [4, 5, 6, 7]
    # Force committed_len 1, 2, 3, 1 respectively (positions 0, 1, 2, 0) --
    # genuinely ragged: no two slots share the same committed_len except
    # slots 0 and 3, which is fine (tests both "all different" and "some
    # coincide" within one ragged batch).
    force_positions = {0: 0, 1: 1, 2: 2, 3: 0}
    prompt_ids_per_slot = [tok.encode(p, add_special_tokens=False) for p in FOUR_PROMPTS]
    _reset_if_needed(runner, mtp_slots + ref_slots)

    prefill_result = runner.mtp_prefill_batch(mtp_slots, prompt_ids_per_slot)
    anchors = {s: prefill_result[s]["anchor"] for s in mtp_slots}
    drafts = {s: prefill_result[s]["draft_tokens"] for s in mtp_slots}
    for i, s in enumerate(mtp_slots):
        ref_first = runner.prefill(ref_slots[i], prompt_ids_per_slot[i])
        if ref_first != anchors[s]:
            return {"passed": False, "error": f"prefill anchor mismatch for slot {s}"}

    per_slot_rounds = {s: [] for s in mtp_slots}
    for r in range(NUM_ROUNDS):
        submitted = {s: _decoy_at(drafts[s], force_positions[s]) for s in mtp_slots}
        decisions = runner.mtp_verify_and_commit_batch(mtp_slots, anchors, submitted)

        for i, s in enumerate(mtp_slots):
            decision = decisions[s]
            forced_ok = decision["num_accepted"] == force_positions[s]
            real_new_tokens = [anchors[s]] + decision["committed"][:-1]
            ref_report = _ref_check(runner, ref_slots[i], real_new_tokens, decision["next_anchor"])
            per_slot_rounds[s].append(
                {
                    "round": r,
                    "forced_position": force_positions[s],
                    "num_accepted": decision["num_accepted"],
                    "forced_reject_landed_correctly": forced_ok,
                    "draft_sync_len_matches_kv_len": runner.slot_draft_sync_len[s] == runner.slot_kv_len[s],
                    **ref_report,
                }
            )
            anchors[s], drafts[s] = decision["next_anchor"], decision["next_draft_tokens"]

    # NOTE (found while diagnosing a real GDN-fast-path bug this round):
    # ``forced_reject_landed_correctly`` is deliberately NOT part of the
    # pass/fail criterion. ``_decoy_at`` only corrupts ONE position; if
    # the model's own ORGANIC draft proposal is already wrong at an
    # EARLIER position (a real, expected outcome given this shape's
    # ~68% real per-position acceptance rate), the reject lands earlier
    # than intended -- this is benign test-harness noise, not a runtime
    # bug, and demanding it never happen would make this check flaky by
    # construction. What actually matters -- and IS enforced -- is that
    # regardless of exactly where each round's reject organically landed,
    # (a) the committed content matches an independent reference replay,
    # and (b) bookkeeping stays consistent. Cross-round variety in the
    # REAL (not necessarily intended) committed_len values across slots
    # is reported separately (``observed_ragged_diversity``) so a run
    # that accidentally never exercises genuine raggedness is visible,
    # not silently reported as a clean pass.
    per_slot_ok = {}
    for s in mtp_slots:
        rounds = per_slot_rounds[s]
        per_slot_ok[s] = (
            all(r["draft_sync_len_matches_kv_len"] for r in rounds)
            and all(r["content_ok_within_near_tie_tolerance"] for r in rounds)
        )

    observed_committed_lens_per_round = [
        {s: per_slot_rounds[s][r]["num_accepted"] + 1 for s in mtp_slots} for r in range(NUM_ROUNDS)
    ]
    ragged_rounds = sum(1 for cl in observed_committed_lens_per_round if len(set(cl.values())) > 1)

    return {
        "passed": bool(all(per_slot_ok.values())),
        "num_rounds": NUM_ROUNDS,
        "force_positions": force_positions,
        "per_slot_ok": per_slot_ok,
        "per_slot_rounds": per_slot_rounds,
        "observed_committed_lens_per_round": observed_committed_lens_per_round,
        "num_genuinely_ragged_rounds": ragged_rounds,
    }


# ---------------------------------------------------------------------------
# Check 2: true mixed-stage -- 2 slots ragged-recompute (different forced
# positions), 2 slots organic full-accept, all in ONE batched call.
# ---------------------------------------------------------------------------


def _check_mixed_ragged_and_full_accept(runner, tok) -> dict:
    slots = [0, 1, 2, 3]
    ref_slots = [4, 5, 6, 7]
    force_positions = {0: 0, 2: 1}  # slots 0, 2 forced (committed_len 1, 2); slots 1, 3 organic
    prompt_ids_per_slot = [tok.encode(p, add_special_tokens=False) for p in FOUR_PROMPTS]
    _reset_if_needed(runner, slots + ref_slots)

    prefill_result = runner.mtp_prefill_batch(slots, prompt_ids_per_slot)
    anchors = {s: prefill_result[s]["anchor"] for s in slots}
    drafts = {s: prefill_result[s]["draft_tokens"] for s in slots}
    for i, s in enumerate(slots):
        ref_first = runner.prefill(ref_slots[i], prompt_ids_per_slot[i])
        if ref_first != anchors[s]:
            return {"passed": False, "error": f"prefill anchor mismatch for slot {s}"}

    rounds = []
    for r in range(NUM_ROUNDS):
        submitted = {}
        for s in slots:
            if s in force_positions:
                submitted[s] = _decoy_at(drafts[s], force_positions[s])
            else:
                submitted[s] = drafts[s]

        decisions = runner.mtp_verify_and_commit_batch(slots, anchors, submitted)

        round_report = {"round": r, "per_slot": {}}
        all_ok = True
        for i, s in enumerate(slots):
            decision = decisions[s]
            real_new_tokens = [anchors[s]] + decision["committed"][:-1]
            ref_report = _ref_check(runner, ref_slots[i], real_new_tokens, decision["next_anchor"])
            slot_report = {
                "forced_position": force_positions.get(s),
                "num_accepted": decision["num_accepted"],
                "kv_len_matches_draft_sync_len": runner.slot_kv_len[s] == runner.slot_draft_sync_len[s],
                **ref_report,
            }
            # NOTE: whether the forced decoy landed at the EXACT intended
            # position is informational only, not a pass/fail signal --
            # see _check_ragged_recompute's identical note (organic
            # earlier rejects are expected, real model behavior, not a
            # runtime bug). Only content/bookkeeping correctness gates
            # pass/fail here.
            if s in force_positions and decision["num_accepted"] != force_positions[s]:
                slot_report["forced_reject_landed_at_different_position"] = True
            if not slot_report["content_ok_within_near_tie_tolerance"] or not slot_report["kv_len_matches_draft_sync_len"]:
                all_ok = False
            round_report["per_slot"][s] = slot_report
            anchors[s], drafts[s] = decision["next_anchor"], decision["next_draft_tokens"]
        round_report["all_ok"] = all_ok
        rounds.append(round_report)

    passed = all(r["all_ok"] for r in rounds)
    return {
        "passed": bool(passed),
        "num_rounds": NUM_ROUNDS,
        "force_positions": force_positions,
        "rounds": rounds,
    }


def _run_once() -> dict:
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
    runner = DirectModelRunner(vllm_config, num_slots=8, block_size=64, blocks_per_slot=640)

    result = {}
    result["check0_batch1_forced_reject_equivalence"] = _check_batch1_forced_reject_equivalence(runner, tok)
    result["check1_ragged_recompute"] = _check_ragged_recompute(runner, tok)
    result["check2_mixed_ragged_and_full_accept"] = _check_mixed_ragged_and_full_accept(runner, tok)
    result["passed"] = bool(
        result["check0_batch1_forced_reject_equivalence"]["passed"]
        and result["check1_ragged_recompute"]["passed"]
        and result["check2_mixed_ragged_and_full_accept"]["passed"]
    )
    return result


def main() -> int:
    import json

    result = _run_once()
    summary = {
        "passed": result["passed"],
        "check0_batch1_forced_reject_equivalence": {
            "passed": result["check0_batch1_forced_reject_equivalence"]["passed"],
            "exact_mismatches": result["check0_batch1_forced_reject_equivalence"].get("exact_mismatches"),
            "near_tie_divergences": result["check0_batch1_forced_reject_equivalence"].get("near_tie_divergences"),
            "ref_failures": result["check0_batch1_forced_reject_equivalence"].get("ref_failures"),
            "self_consistent": result["check0_batch1_forced_reject_equivalence"].get("self_consistent"),
        },
        "check1_ragged_recompute": {
            "passed": result["check1_ragged_recompute"]["passed"],
            "force_positions": result["check1_ragged_recompute"].get("force_positions"),
            "per_slot_ok": result["check1_ragged_recompute"].get("per_slot_ok"),
            "num_genuinely_ragged_rounds": result["check1_ragged_recompute"].get("num_genuinely_ragged_rounds"),
            "observed_committed_lens_per_round": result["check1_ragged_recompute"].get(
                "observed_committed_lens_per_round"
            ),
        },
        "check2_mixed_ragged_and_full_accept": {
            "passed": result["check2_mixed_ragged_and_full_accept"]["passed"],
            "force_positions": result["check2_mixed_ragged_and_full_accept"].get("force_positions"),
            "rounds_all_ok": [r["all_ok"] for r in result["check2_mixed_ragged_and_full_accept"].get("rounds", [])],
        },
    }
    print(json.dumps(summary, indent=2, default=str))
    if not result["passed"]:
        print(json.dumps(result, indent=2, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
