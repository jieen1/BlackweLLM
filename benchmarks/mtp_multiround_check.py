"""Verification gradient steps 5-6: multi-round continuous MTP (concurrency=1)
and 4-slot concurrent isolation. Real numerical twin comparisons at every
round (not signal-probe alone), per the coordinator's explicit instruction
not to rely on signal-probe for this class of check.

Step 5 (multi-round, concurrency=1): drives the real
``mtp_prefill`` -> loop of ``mtp_verify_and_commit`` cycle (this is now the
FULL real cycle -- verify+commit+resync+propose folded into one call, see
``mtp_verify_and_commit``'s docstring) for several rounds on ONE slot,
mixing organic-accept and forced-reject rounds. At EVERY round, an
INDEPENDENT reference slot replays the SAME real committed tokens via the
plain, long-verified ``_forward`` path and its own next-token prediction is
compared against the MTP slot's own ``next_anchor`` -- this is round-by-round,
not just a final check, so drift would show up immediately, not average out.

Step 6 (4-slot isolation): the SAME per-round numerical-twin check,
independently, for 4 DIFFERENT (mtp_slot, ref_slot) pairs, INTERLEAVED
round-robin (not run sequentially) -- this is what actually exercises
whether the draft model's own per-slot KV addressing and the GDN
snapshot/restore mechanism cross-contaminate when multiple slots are
simultaneously active. Combined with a signal-probe layer (4 prompts about
4 different, easily-distinguishable topics/countries) as the coordinator
asked for both methods.

Scope note (reported honestly, not glossed over): this round's 4-slot test
interleaves independent SINGLE-slot ``mtp_prefill``/``mtp_verify_and_commit``
calls across 4 slots -- NOT one batched ``verify_batch`` call spanning all
4 slots at once (that would require generalizing the MTP coordinator
methods to accept slot lists, matching how ``_forward_batch``/
``verify_batch`` already do for the plain target-only path -- not done this
round). This still directly tests the coordinator's stated concern (does
per-slot KV/GDN state stay isolated when multiple slots are simultaneously
active), just not via a single fused kernel launch across slots.

Usage:
    python -m benchmarks.mtp_multiround_check
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
NUM_ROUNDS = 8
SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"}

# 4 clearly distinguishable prompts for the signal-probe layer of step 6 --
# an eyeballable sanity check (does slot i's own continuation stay on-topic
# for its own prompt) alongside the decisive numerical-twin check.
FOUR_PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Germany is",
    "The capital of Italy is",
]


NEAR_TIE_LOGIT_MARGIN = 2.0  # see the design doc: a real, evidenced near-exact
# tie (271 vs 198 both at 25.375 in one kernel path, 24.25 vs 24.0 in
# another) was found and diagnosed this round -- distinct real candidates
# at this prompt/position are typically separated by 8-13+ logit units, so
# a margin this small is a genuine near-tie in the model's OWN
# distribution (kernel-path-sensitive by nature), not evidence of state
# corruption. A mismatch is only treated as a real problem if the
# reference's own logit for MTP's committed token is NOT within this
# margin of the reference's own top candidate.


def _run_round(runner, mtp_slot, ref_slot, anchor, draft_tokens, force_reject, decoy):
    """One real MTP cycle on mtp_slot + the matching independent-reference
    replay on ref_slot. Returns (decision, round_report)."""
    if force_reject:
        submitted = [decoy] + draft_tokens[1:]
    else:
        submitted = draft_tokens

    decision = runner.mtp_verify_and_commit(mtp_slot, anchor, submitted)
    real_new_tokens = [anchor] + decision["committed"][:-1]

    ref_logits = runner._forward(
        ref_slot, real_new_tokens, start_pos=runner.slot_kv_len[ref_slot], is_decode=(len(real_new_tokens) == 1)
    )
    ref_last = ref_logits[-1].float()
    ref_predicted_next = int(ref_last.argmax(dim=-1).item())
    ref_top1_logit = float(ref_last.max().item())
    ref_logit_for_mtp_choice = float(ref_last[decision["next_anchor"]].item())
    near_tie_margin = ref_top1_logit - ref_logit_for_mtp_choice

    exact_match = ref_predicted_next == decision["next_anchor"]
    report = {
        "forced_reject": force_reject,
        "submitted_draft": submitted,
        "num_accepted": decision["num_accepted"],
        "committed": decision["committed"],
        "draft_sync_len_matches_kv_len": runner.slot_draft_sync_len[mtp_slot] == runner.slot_kv_len[mtp_slot],
        "ref_predicted_next_matches_mtp_next_anchor": exact_match,
        "near_tie_margin": near_tie_margin,
        "content_ok_within_near_tie_tolerance": exact_match or near_tie_margin < NEAR_TIE_LOGIT_MARGIN,
    }
    return decision, report


def _multiround_single_slot(runner, tok, mtp_slot: int, ref_slot: int, prompt: str) -> dict:
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    for slot in (mtp_slot, ref_slot):
        if runner.slot_kv_len[slot] != 0:
            runner.reset_slot(slot)
            runner.slot_draft_sync_len[slot] = 0

    pr = runner.mtp_prefill(mtp_slot, prompt_ids)
    ref_first = runner.prefill(ref_slot, prompt_ids)
    if ref_first != pr["anchor"]:
        return {"passed": False, "error": "prefill anchor mismatch between mtp_slot and ref_slot"}

    anchor, draft_tokens = pr["anchor"], pr["draft_tokens"]
    rounds = []
    for r in range(NUM_ROUNDS):
        force_reject = r % 3 == 2
        decoy = 100 if 100 not in draft_tokens else 200
        decision, report = _run_round(runner, mtp_slot, ref_slot, anchor, draft_tokens, force_reject, decoy)
        rounds.append(report)
        anchor, draft_tokens = decision["next_anchor"], decision["next_draft_tokens"]

    all_invariants_ok = all(r["draft_sync_len_matches_kv_len"] for r in rounds)
    num_exact_content_matches = sum(1 for r in rounds if r["ref_predicted_next_matches_mtp_next_anchor"])
    all_content_ok = all(r["content_ok_within_near_tie_tolerance"] for r in rounds)
    num_forced_reject_rounds = sum(1 for r in rounds if r["forced_reject"])
    num_organic_reject_rounds = sum(
        1 for r in rounds if not r["forced_reject"] and r["num_accepted"] < K
    )
    return {
        "passed": bool(all_invariants_ok and all_content_ok),
        "num_rounds": NUM_ROUNDS,
        "num_forced_reject_rounds": num_forced_reject_rounds,
        "num_organic_reject_rounds": num_organic_reject_rounds,
        "all_invariants_ok": all_invariants_ok,
        "num_exact_content_matches": num_exact_content_matches,
        "all_content_ok": all_content_ok,
        "rounds": rounds,
        "final_kv_len": runner.slot_kv_len[mtp_slot],
        "final_draft_sync_len": runner.slot_draft_sync_len[mtp_slot],
    }


def _four_slot_isolation(runner, tok) -> dict:
    mtp_slots = [0, 1, 2, 3]
    ref_slots = [4, 5, 6, 7]
    prompt_ids_per_slot = [tok.encode(p, add_special_tokens=False) for p in FOUR_PROMPTS]

    for slot in mtp_slots + ref_slots:
        if runner.slot_kv_len[slot] != 0:
            runner.reset_slot(slot)
            runner.slot_draft_sync_len[slot] = 0

    anchors = {}
    drafts = {}
    for i, (mtp_slot, ref_slot) in enumerate(zip(mtp_slots, ref_slots)):
        pr = runner.mtp_prefill(mtp_slot, prompt_ids_per_slot[i])
        ref_first = runner.prefill(ref_slot, prompt_ids_per_slot[i])
        if ref_first != pr["anchor"]:
            return {"passed": False, "error": f"prefill anchor mismatch for slot pair {i}"}
        anchors[mtp_slot] = pr["anchor"]
        drafts[mtp_slot] = pr["draft_tokens"]

    per_slot_rounds = {mtp_slot: [] for mtp_slot in mtp_slots}
    committed_sequences = {mtp_slot: [] for mtp_slot in mtp_slots}
    for r in range(NUM_ROUNDS):
        # Round-robin interleave -- all 4 slots' rounds happen "at the same
        # time" from the runtime's perspective (no slot completes all its
        # rounds before another starts), which is what actually exercises
        # cross-slot addressing rather than 4 independent sequential runs.
        for i, mtp_slot in enumerate(mtp_slots):
            ref_slot = ref_slots[i]
            force_reject = r % 3 == 2
            draft_tokens = drafts[mtp_slot]
            decoy = 100 if 100 not in draft_tokens else 200
            decision, report = _run_round(
                runner, mtp_slot, ref_slot, anchors[mtp_slot], draft_tokens, force_reject, decoy
            )
            report["slot"] = mtp_slot
            report["prompt"] = FOUR_PROMPTS[i]
            per_slot_rounds[mtp_slot].append(report)
            committed_sequences[mtp_slot].extend(decision["committed"])
            anchors[mtp_slot], drafts[mtp_slot] = decision["next_anchor"], decision["next_draft_tokens"]

    # Signal-probe layer: no two slots' committed token sequences should be
    # identical (they're 4 different prompts/topics -- if slot state
    # cross-contaminated, this is the cheap first sign something's wrong).
    seqs = list(committed_sequences.values())
    no_cross_contamination_signal = all(
        seqs[i] != seqs[j] for i in range(len(seqs)) for j in range(i + 1, len(seqs))
    )

    per_slot_ok = {}
    for mtp_slot in mtp_slots:
        rounds = per_slot_rounds[mtp_slot]
        per_slot_ok[mtp_slot] = all(r["draft_sync_len_matches_kv_len"] for r in rounds) and all(
            r["content_ok_within_near_tie_tolerance"] for r in rounds
        )

    return {
        "passed": bool(no_cross_contamination_signal and all(per_slot_ok.values())),
        "no_cross_contamination_signal": no_cross_contamination_signal,
        "per_slot_ok": per_slot_ok,
        "committed_sequences": committed_sequences,
        "per_slot_rounds": per_slot_rounds,
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
    result["step5_multiround_c1"] = _multiround_single_slot(
        runner, tok, mtp_slot=0, ref_slot=1, prompt=FOUR_PROMPTS[0]
    )
    result["step6_four_slot_isolation"] = _four_slot_isolation(runner, tok)

    result["passed"] = bool(result["step5_multiround_c1"]["passed"] and result["step6_four_slot_isolation"]["passed"])
    return result


def main() -> int:
    import json

    result = _run_once()
    # Trim per-round detail from the printed summary (kept in the returned
    # dict for anyone re-running interactively) so the console output stays
    # readable -- print pass/fail + key aggregate numbers only.
    summary = {
        "passed": result["passed"],
        "step5_multiround_c1": {
            k: v for k, v in result["step5_multiround_c1"].items() if k != "rounds"
        },
        "step6_four_slot_isolation": {
            "passed": result["step6_four_slot_isolation"]["passed"],
            "no_cross_contamination_signal": result["step6_four_slot_isolation"]["no_cross_contamination_signal"],
            "per_slot_ok": result["step6_four_slot_isolation"]["per_slot_ok"],
            "committed_sequences": result["step6_four_slot_isolation"]["committed_sequences"],
        },
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
