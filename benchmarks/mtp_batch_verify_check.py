"""Correctness verification for the 2026-07-17 cross-slot batched MTP
coordinator (``mtp_prefill_batch``/``mtp_verify_and_commit_batch``,
``runtime/direct_model_runner.py``).

Four checks, per the coordinator's explicit instruction to continue the
established methodology (signal-probe + numerical-twin) with SPECIFIC new
coverage of the "mixed-stage" scenario. Methodology note (2026-07-17,
after a real divergence-diagnosis round -- see ``mtp_batch_divergence_diag.py``
and ``notes/direct-model-runner-design.md``): an EARLIER version of this
script compared two INDEPENDENT multi-round trajectories (looped path vs.
batched path) token-for-token, which is the WRONG methodology for a greedy
decoder -- any single near-exact-tie tie-break flip (a real, already-
documented, kernel-numerics-dependent phenomenon in this codebase, see
``mtp_multiround_check.py``'s ``NEAR_TIE_LOGIT_MARGIN``) cascades into two
fully different-looking completions even when every individual step was
computed correctly. Fixed by adopting ``mtp_multiround_check.py``'s own
per-round INDEPENDENT-REFERENCE-REPLAY methodology instead (feed the
batched run's OWN real committed tokens into an independent single-slot
reference forward every round, so one round's tie-break flip cannot
compound into the next round's check).

0. Batch=1 equivalence: the batched-path API called with a SINGLE slot is
   compared against the original single-slot path's output. Through
   2026-07-17 this required BIT-EXACT agreement (no near-tie tolerance) --
   valid at the time because both paths shared the exact same underlying
   mechanism (chunked GDN metadata), differing only in call convention, so
   any divergence was necessarily a real bug (this is what actually caught
   the real ``decode_qo_len`` formula gap that round).

   **2026-07-18, Phase 2**: ``mtp_verify_and_commit_batch`` now uses a
   genuinely different mechanism from ``mtp_verify_and_commit`` (the real
   spec-decode GDN path -- K+1 dedicated SSM rows, no snapshot/restore/
   recompute-forward -- vs. the singular path's unchanged chunked +
   snapshot/restore/recompute mechanism; see ``runtime/direct_model_runner.py``).
   Bit-exact agreement between two genuinely different numerical paths is
   no longer a valid invariant in general -- a direct measurement in
   ``benchmarks/mtp_ragged_recompute_verify_check.py`` (this file's
   sibling, exercising the SAME two mechanisms) found a real near-tie
   tie-break flip at this exact prompt (committed token 271 vs. 198,
   margins 0.125/0.0 -- both far under ``NEAR_TIE_LOGIT_MARGIN``) once a
   forced reject was involved. This check happens not to hit that
   specific near-tie in its own (organic, non-forced) scenario, but is
   structurally just as exposed to the same risk, so it now uses this
   file's own already-established per-round independent-reference-replay
   methodology (matching checks 1-3 below) instead of a raw bit-exact
   comparison, for the same reason those checks needed it: a real
   near-tie divergence must not fail this test, only an UNEXPLAINED one
   (a mismatch that also fails independent reference validation) should.
   Kept here as a permanent regression guard for real bugs -- just no
   longer conflating "real bug" with "expected near-tie noise between two
   now-legitimately-different mechanisms."

1. Numerical-twin (real batch=4): 4 slots run through the batched path
   concurrently; each round, each slot's real committed tokens are
   replayed through an INDEPENDENT single-slot reference slot, and the
   reference's own next-token argmax is compared against the batched
   path's ``next_anchor`` (near-tie tolerance applies, matching
   ``mtp_multiround_check.py``'s own established margin).

2. Signal-probe: the batched path alone, run on 4 slots with 4 clearly
   distinguishable prompts, confirms no cross-slot content contamination.

3. Forced mixed-stage: in ONE ``mtp_verify_and_commit_batch`` call spanning
   4 slots, 2 slots are given a corrupted (decoy) draft token guaranteeing
   an organic reject (needs-recompute branch) while the other 2 keep their
   real, organically-accepted draft (full-accept branch) -- directly
   exercising the "same batched round, different slots at different
   post-verify stages" case the coordinator specifically called out.

Usage:
    python -m benchmarks.mtp_batch_verify_check
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

# Same constant, same rationale as mtp_multiround_check.py: distinct real
# candidates are typically separated by 8-13+ logit units at this
# prompt/position; a gap this small is a genuine near-tie in the model's
# OWN distribution (kernel-path-sensitive by nature -- e.g. batch=1 vs.
# batch=4 dispatch, already observed this round at the EXACT SAME
# token pair (271 vs 198) this project's earlier near-tie finding
# documented), not evidence of a batching logic bug.
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


def _ref_check(runner, ref_slot, real_new_tokens, mtp_next_anchor) -> dict:
    """Independent single-slot reference replay of one round's real
    committed tokens (mtp_multiround_check.py's own established pattern) --
    decouples this round's check from any PRIOR round's possible
    near-tie divergence, since ref_slot always consumes the REAL tokens
    the batched path actually committed, not its own previously-generated
    (and potentially already-diverged) history."""
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
# Check 0: strict batch=1 equivalence (the check that actually caught this
# round's real bug -- kept as a permanent regression guard).
# ---------------------------------------------------------------------------


def _check_batch1_equivalence(runner, tok) -> dict:
    looped_slot, batched_slot = 0, 1
    # 2026-07-18, Phase 2: independent, untouched reference slots -- see
    # this module's docstring update for why bit-exact looped-vs-batched
    # agreement is no longer a valid invariant on its own.
    looped_ref_slot, batched_ref_slot = 2, 3
    prompt = FOUR_PROMPTS[0]
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    _reset_if_needed(runner, [looped_slot, batched_slot, looped_ref_slot, batched_ref_slot])

    pr_looped = runner.mtp_prefill(looped_slot, prompt_ids)
    pr_batched = runner.mtp_prefill_batch([batched_slot], [prompt_ids])[batched_slot]
    anchor_l, drafts_l = pr_looped["anchor"], pr_looped["draft_tokens"]
    anchor_b, drafts_b = pr_batched["anchor"], pr_batched["draft_tokens"]

    exact_mismatches = []
    if anchor_l != anchor_b or drafts_l != drafts_b:
        exact_mismatches.append({"stage": "prefill", "looped": pr_looped, "batched": pr_batched})

    ref_first_l = runner.prefill(looped_ref_slot, prompt_ids)
    ref_first_b = runner.prefill(batched_ref_slot, prompt_ids)
    if ref_first_l != anchor_l or ref_first_b != anchor_b:
        exact_mismatches.append(
            {"stage": "reference_prefill", "ref_first_l": ref_first_l, "ref_first_b": ref_first_b}
        )

    near_tie_divergences = []
    ref_failures = []
    for r in range(NUM_ROUNDS):
        decision_l = runner.mtp_verify_and_commit(looped_slot, anchor_l, drafts_l)
        decision_b = runner.mtp_verify_and_commit_batch(
            [batched_slot], {batched_slot: anchor_b}, {batched_slot: drafts_b}
        )[batched_slot]

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
                "looped_committed": decision_l["committed"],
                "batched_committed": decision_b["committed"],
            }
            if ref_report_l["content_ok_within_near_tie_tolerance"] and ref_report_b["content_ok_within_near_tie_tolerance"]:
                near_tie_divergences.append(entry)
            else:
                exact_mismatches.append(entry)

        anchor_l, drafts_l = decision_l["next_anchor"], decision_l["next_draft_tokens"]
        anchor_b, drafts_b = decision_b["next_anchor"], decision_b["next_draft_tokens"]

    self_consistent = (
        runner.slot_kv_len[looped_slot] == runner.slot_draft_sync_len[looped_slot]
        and runner.slot_kv_len[batched_slot] == runner.slot_draft_sync_len[batched_slot]
    )
    passed = bool(len(exact_mismatches) == 0 and len(ref_failures) == 0 and self_consistent)
    return {
        "passed": passed,
        "num_rounds": NUM_ROUNDS,
        "exact_mismatches": exact_mismatches,
        "near_tie_divergences": near_tie_divergences,
        "ref_failures": ref_failures,
        "self_consistent": self_consistent,
        "final_looped_kv_len": runner.slot_kv_len[looped_slot],
        "final_batched_kv_len": runner.slot_kv_len[batched_slot],
    }


# ---------------------------------------------------------------------------
# Check 1: numerical-twin, real batch=4, per-round independent-reference
# replay (matching mtp_multiround_check.py's established methodology).
# ---------------------------------------------------------------------------


def _check_numerical_twin(runner, tok) -> dict:
    mtp_slots = [0, 1, 2, 3]
    ref_slots = [4, 5, 6, 7]
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
        decisions = runner.mtp_verify_and_commit_batch(mtp_slots, anchors, drafts)
        for i, s in enumerate(mtp_slots):
            decision = decisions[s]
            real_new_tokens = [anchors[s]] + decision["committed"][:-1]
            ref_report = _ref_check(runner, ref_slots[i], real_new_tokens, decision["next_anchor"])
            per_slot_rounds[s].append(
                {
                    "round": r,
                    "num_accepted": decision["num_accepted"],
                    "draft_sync_len_matches_kv_len": runner.slot_draft_sync_len[s] == runner.slot_kv_len[s],
                    **ref_report,
                }
            )
            anchors[s], drafts[s] = decision["next_anchor"], decision["next_draft_tokens"]

    per_slot_ok = {}
    for s in mtp_slots:
        rounds = per_slot_rounds[s]
        per_slot_ok[s] = all(r["draft_sync_len_matches_kv_len"] for r in rounds) and all(
            r["content_ok_within_near_tie_tolerance"] for r in rounds
        )

    return {
        "passed": bool(all(per_slot_ok.values())),
        "num_rounds": NUM_ROUNDS,
        "per_slot_ok": per_slot_ok,
        "per_slot_rounds": per_slot_rounds,
    }


# ---------------------------------------------------------------------------
# Check 2: signal-probe on the batched path alone.
# ---------------------------------------------------------------------------


def _check_signal_probe(runner, tok) -> dict:
    """2026-07-17 methodology fix: this check's original pass criterion
    ("no two slots' committed sequences are ever identical") turned out
    to be a real, reproducible false-positive trap for THESE SPECIFIC 4
    prompts -- all share an identical template ("The capital of X is"),
    differing only in one embedded country name/token. A deep, multi-stage
    diagnosis (mtp_slot_identity_pinpoint_diag.py +
    mtp_prefill_draft_logits_diag.py) confirmed this is genuine,
    deterministic model behavior, NOT a cross-slot data-sharing bug: each
    slot's OWN raw logits (both the draft model's step-0 logits and the
    target model's verify logits) are substantially different between
    slots (e.g. slot1-vs-slot2 max_abs_diff ~7.6, the SAME order of
    magnitude as slot1-vs-slot3's ~7.8, i.e. no suspicious closeness) --
    but BOTH slots independently have a near-tie (margin < this
    project's own established NEAR_TIE_LOGIT_MARGIN=2.0) between the
    SAME two generic continuation-token candidates after the shared
    template's factual statement, and that near-tie happens to resolve
    the same way for both. This is the identical CLASS of phenomenon
    already documented elsewhere in this codebase (e.g. the "271 vs 198"
    near-tie found earlier this session) -- batch-size-dependent kernel
    numerics can tip a genuine near-tie either way, and a template shared
    across all 4 probe prompts makes hitting the SAME near-tie for
    multiple slots far more likely than for prompts with less shared
    structure. Fixed by cross-checking each slot's own committed content
    against an INDEPENDENT reference replay (the same robust methodology
    ``_check_numerical_twin`` already uses) instead of the fragile
    "must differ from every other slot" assumption -- that assumption is
    demoted to an informational signal only (``no_cross_contamination_signal``),
    not a pass/fail gate.
    """
    slots = [0, 1, 2, 3]
    ref_slots = [4, 5, 6, 7]
    prompt_ids_per_slot = [tok.encode(p, add_special_tokens=False) for p in FOUR_PROMPTS]
    _reset_if_needed(runner, slots + ref_slots)

    prefill_result = runner.mtp_prefill_batch(slots, prompt_ids_per_slot)
    anchors = {s: prefill_result[s]["anchor"] for s in slots}
    drafts = {s: prefill_result[s]["draft_tokens"] for s in slots}
    for i, s in enumerate(slots):
        ref_first = runner.prefill(ref_slots[i], prompt_ids_per_slot[i])
        if ref_first != anchors[s]:
            return {"passed": False, "error": f"prefill anchor mismatch for slot {s}"}

    committed_sequences = {s: [] for s in slots}
    per_slot_rounds = {s: [] for s in slots}
    for r in range(NUM_ROUNDS):
        decisions = runner.mtp_verify_and_commit_batch(slots, anchors, drafts)
        for i, s in enumerate(slots):
            decision = decisions[s]
            committed_sequences[s].extend(decision["committed"])
            real_new_tokens = [anchors[s]] + decision["committed"][:-1]
            ref_report = _ref_check(runner, ref_slots[i], real_new_tokens, decision["next_anchor"])
            per_slot_rounds[s].append(
                {
                    "round": r,
                    "draft_sync_len_matches_kv_len": runner.slot_draft_sync_len[s] == runner.slot_kv_len[s],
                    **ref_report,
                }
            )
            anchors[s], drafts[s] = decision["next_anchor"], decision["next_draft_tokens"]

    per_slot_ok = {}
    for s in slots:
        rounds = per_slot_rounds[s]
        per_slot_ok[s] = all(r["draft_sync_len_matches_kv_len"] for r in rounds) and all(
            r["content_ok_within_near_tie_tolerance"] for r in rounds
        )

    seqs = list(committed_sequences.values())
    no_cross_contamination_signal = all(
        seqs[i] != seqs[j] for i in range(len(seqs)) for j in range(i + 1, len(seqs))
    )
    decoded = {s: tok.decode(committed_sequences[s]) for s in slots}
    return {
        "passed": bool(all(per_slot_ok.values())),
        "per_slot_ok": per_slot_ok,
        "no_cross_contamination_signal": no_cross_contamination_signal,
        "committed_sequences": committed_sequences,
        "decoded_completions": decoded,
    }


# ---------------------------------------------------------------------------
# Check 3: forced mixed-stage -- some slots reject, some fully accept, in
# the SAME mtp_verify_and_commit_batch call.
# ---------------------------------------------------------------------------


def _check_mixed_stage(runner, tok) -> dict:
    slots = [0, 1, 2, 3]
    ref_slots = [4, 5, 6, 7]
    reject_slots = {slots[0], slots[2]}
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
            if s in reject_slots:
                decoy = 100 if 100 not in drafts[s] else 200
                submitted[s] = [decoy] + drafts[s][1:]
            else:
                submitted[s] = drafts[s]

        decisions = runner.mtp_verify_and_commit_batch(slots, anchors, submitted)

        round_report = {"round": r, "per_slot": {}}
        all_ok = True
        for i, s in enumerate(slots):
            decision = decisions[s]
            was_forced_reject = s in reject_slots
            real_new_tokens = [anchors[s]] + decision["committed"][:-1]
            ref_report = _ref_check(runner, ref_slots[i], real_new_tokens, decision["next_anchor"])
            slot_report = {
                "forced_reject": was_forced_reject,
                "num_accepted": decision["num_accepted"],
                "committed": decision["committed"],
                "kv_len_matches_draft_sync_len": runner.slot_kv_len[s] == runner.slot_draft_sync_len[s],
                **ref_report,
            }
            if was_forced_reject and decision["num_accepted"] != 0:
                slot_report["forced_reject_but_not_rejected"] = True
                all_ok = False
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
        "reject_slots": sorted(reject_slots),
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
    result["check0_batch1_equivalence"] = _check_batch1_equivalence(runner, tok)
    result["check1_numerical_twin"] = _check_numerical_twin(runner, tok)
    result["check2_signal_probe"] = _check_signal_probe(runner, tok)
    result["check3_mixed_stage"] = _check_mixed_stage(runner, tok)
    result["passed"] = bool(
        result["check0_batch1_equivalence"]["passed"]
        and result["check1_numerical_twin"]["passed"]
        and result["check2_signal_probe"]["passed"]
        and result["check3_mixed_stage"]["passed"]
    )
    return result


def main() -> int:
    import json

    result = _run_once()
    summary = {
        "passed": result["passed"],
        "check0_batch1_equivalence": {
            "passed": result["check0_batch1_equivalence"]["passed"],
            "exact_mismatches": result["check0_batch1_equivalence"]["exact_mismatches"],
            "near_tie_divergences": result["check0_batch1_equivalence"].get("near_tie_divergences"),
            "ref_failures": result["check0_batch1_equivalence"].get("ref_failures"),
            "self_consistent": result["check0_batch1_equivalence"]["self_consistent"],
        },
        "check1_numerical_twin": {
            "passed": result["check1_numerical_twin"]["passed"],
            "per_slot_ok": result["check1_numerical_twin"].get("per_slot_ok"),
        },
        "check2_signal_probe": {
            "passed": result["check2_signal_probe"]["passed"],
            "per_slot_ok": result["check2_signal_probe"].get("per_slot_ok"),
            "no_cross_contamination_signal": result["check2_signal_probe"]["no_cross_contamination_signal"],
            "decoded_completions": result["check2_signal_probe"]["decoded_completions"],
        },
        "check3_mixed_stage": {
            "passed": result["check3_mixed_stage"]["passed"],
            "reject_slots": result["check3_mixed_stage"].get("reject_slots"),
            "rounds_all_ok": [r["all_ok"] for r in result["check3_mixed_stage"].get("rounds", [])],
        },
    }
    print(json.dumps(summary, indent=2, default=str))
    if not result["passed"]:
        # Full detail on failure, so a real regression is diagnosable
        # without re-running.
        print(json.dumps(result, indent=2, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
