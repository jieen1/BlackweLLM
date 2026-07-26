"""Diagnostic for the real divergence ``mtp_batch_verify_check.py`` found
between the looped single-slot MTP path and the new batched path. Goal:
distinguish (a) a genuine LOGIC bug in the new batched coordinator code
from (b) a batch-size-dependent kernel numerical difference (already an
established, tolerated phenomenon in this codebase -- see
``mtp_multiround_check.py``'s ``NEAR_TIE_LOGIT_MARGIN`` and
``build_attention_metadata_batch``'s own "batch=1/2/3/4" ladder note).

Test A: run the SAME prompt through the batched-path API but with a batch
of exactly 1 slot (``mtp_prefill_batch([s], [prompt])`` /
``mtp_verify_and_commit_batch([s], ...)``), and compare token-for-token
against the ORIGINAL single-slot API (``mtp_prefill``/
``mtp_verify_and_commit``) on a different slot. If these match EXACTLY,
the batched code's control flow/indexing is correct at batch=1, and any
divergence only appears when batch size > 1 -- pointing at kernel-level
numerics, not a logic bug in this round's new code.

Test B: for the REAL 4-slot batched run vs. the looped run, compare raw
per-step logits (not just argmax) at the FIRST round where committed
token sequences diverge, reporting the logit gap between the two
candidates -- large gaps (many logit units) indicate a real bug; a gap on
the same order as this codebase's own established near-tie margin
(~2 logit units) indicates the already-known kernel-numerics phenomenon.

Usage:
    python -m benchmarks.mtp_batch_divergence_diag
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
PROMPT = "The capital of France is"


def _reset_if_needed(runner, slots):
    for slot in slots:
        if runner.slot_kv_len[slot] != 0 or runner.slot_draft_sync_len[slot] != 0:
            runner.reset_slot(slot)


def _run_looped(runner, tok, slot: int) -> dict:
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    _reset_if_needed(runner, [slot])
    pr = runner.mtp_prefill(slot, prompt_ids)
    anchor, drafts = pr["anchor"], pr["draft_tokens"]
    committed_all = []
    per_round_logits = []
    for r in range(NUM_ROUNDS):
        decision = runner.mtp_verify_and_commit(slot, anchor, drafts)
        committed_all.extend(decision["committed"])
        anchor, drafts = decision["next_anchor"], decision["next_draft_tokens"]
    return {"committed": committed_all}


def _run_batched_size1(runner, tok, slot: int) -> dict:
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    _reset_if_needed(runner, [slot])
    pr = runner.mtp_prefill_batch([slot], [prompt_ids])
    anchor, drafts = pr[slot]["anchor"], pr[slot]["draft_tokens"]
    committed_all = []
    for r in range(NUM_ROUNDS):
        decisions = runner.mtp_verify_and_commit_batch([slot], {slot: anchor}, {slot: drafts})
        decision = decisions[slot]
        committed_all.extend(decision["committed"])
        anchor, drafts = decision["next_anchor"], decision["next_draft_tokens"]
    return {"committed": committed_all}


def _run_batched_size4_with_logit_trace(runner, tok, slots: list[int]) -> dict:
    """Like the real 4-slot batched check, but also records this slot's own
    raw verify logits (float32) at every round, keyed by slot, so a later
    per-step comparison against the looped path's own logits is possible."""
    prompt_ids_per_slot = [tok.encode(PROMPT, add_special_tokens=False) for _ in slots]
    _reset_if_needed(runner, slots)
    pr = runner.mtp_prefill_batch(slots, prompt_ids_per_slot)
    anchors = {s: pr[s]["anchor"] for s in slots}
    drafts = {s: pr[s]["draft_tokens"] for s in slots}
    committed_all = {s: [] for s in slots}
    round_logits = {s: [] for s in slots}
    for r in range(NUM_ROUNDS):
        # Manually redo the verify step to capture raw logits per slot
        # (mirrors mtp_verify_and_commit_batch's own internals exactly) --
        # snapshot/restore GDN state around this EXTRA diagnostic-only call
        # so it does not itself corrupt state (GDN's recurrent state
        # physically advances on every real forward regardless of
        # ``commit``, so calling verify_batch twice back-to-back on the
        # same input would otherwise double-advance it before the real
        # ``mtp_verify_and_commit_batch`` call below ever runs).
        k = len(drafts[slots[0]])
        draft_lists = [[anchors[s]] + drafts[s] for s in slots]
        kv_lens_before = [runner.slot_kv_len[s] for s in slots]
        diag_snapshots = {s: runner.snapshot_gdn_state(s) for s in slots}
        verify_logits = runner.verify_batch(slots, draft_lists, kv_lens_before, return_hidden=False)
        for i, s in enumerate(slots):
            row_logits = verify_logits[i * (k + 1) : (i + 1) * (k + 1)].float().cpu()
            round_logits[s].append(row_logits)
        for s in slots:
            runner.restore_gdn_state(s, diag_snapshots[s])

        decisions = runner.mtp_verify_and_commit_batch(slots, anchors, drafts)
        for s in slots:
            committed_all[s].extend(decisions[s]["committed"])
            anchors[s], drafts[s] = decisions[s]["next_anchor"], decisions[s]["next_draft_tokens"]
    return {"committed": committed_all, "round_logits": round_logits}


def _run_looped_with_logit_trace(runner, tok, slot: int) -> dict:
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    _reset_if_needed(runner, [slot])
    pr = runner.mtp_prefill(slot, prompt_ids)
    anchor, drafts = pr["anchor"], pr["draft_tokens"]
    committed_all = []
    round_logits = []
    for r in range(NUM_ROUNDS):
        k = len(drafts)
        draft = [anchor] + drafts
        kv_len_before = runner.slot_kv_len[slot]
        diag_snapshot = runner.snapshot_gdn_state(slot)
        verify_logits = runner.verify_batch([slot], [draft], [kv_len_before], return_hidden=False)
        round_logits.append(verify_logits.float().cpu())
        runner.restore_gdn_state(slot, diag_snapshot)

        decision = runner.mtp_verify_and_commit(slot, anchor, drafts)
        committed_all.extend(decision["committed"])
        anchor, drafts = decision["next_anchor"], decision["next_draft_tokens"]
    return {"committed": committed_all, "round_logits": round_logits}


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

    # Test A: batched-path-at-batch=1 vs. original single-slot path.
    looped = _run_looped(runner, tok, slot=0)
    batched1 = _run_batched_size1(runner, tok, slot=1)
    result["test_a_batch1_vs_looped"] = {
        "match": looped["committed"] == batched1["committed"],
        "looped_committed": looped["committed"],
        "batched1_committed": batched1["committed"],
    }

    # Test B: real 4-slot batched run vs. looped run, with logit traces.
    looped_traced = _run_looped_with_logit_trace(runner, tok, slot=2)
    batched4_traced = _run_batched_size4_with_logit_trace(runner, tok, slots=[3, 4, 5, 6])
    # slot 3 in the 4-slot group is the one directly comparable (same
    # prompt, same round structure) to slot 2's looped run.
    target_slot = 3
    looped_committed = looped_traced["committed"]
    batched_committed = batched4_traced["committed"][target_slot]

    first_diverge_round = None
    logit_gap_report = None
    running_looped_idx = 0
    running_batched_idx = 0
    for r in range(NUM_ROUNDS):
        looped_logits = looped_traced["round_logits"][r]  # [k+1, vocab]
        batched_logits = batched4_traced["round_logits"][target_slot][r]  # [k+1, vocab]
        k_plus_1 = looped_logits.shape[0]
        for p in range(k_plus_1):
            looped_top = int(looped_logits[p].argmax(dim=-1).item())
            batched_top = int(batched_logits[p].argmax(dim=-1).item())
            if looped_top != batched_top:
                first_diverge_round = r
                looped_val_looped_top = float(looped_logits[p, looped_top].item())
                looped_val_batched_top = float(looped_logits[p, batched_top].item())
                batched_val_looped_top = float(batched_logits[p, looped_top].item())
                batched_val_batched_top = float(batched_logits[p, batched_top].item())
                max_abs_diff = float((looped_logits[p] - batched_logits[p]).abs().max().item())
                logit_gap_report = {
                    "round": r,
                    "position_in_draft": p,
                    "looped_argmax_token": looped_top,
                    "batched_argmax_token": batched_top,
                    "looped_logits_own_top1": looped_val_looped_top,
                    "looped_logits_at_batched_top1": looped_val_batched_top,
                    "looped_margin": looped_val_looped_top - looped_val_batched_top,
                    "batched_logits_own_top1": batched_val_batched_top,
                    "batched_logits_at_looped_top1": batched_val_looped_top,
                    "batched_margin": batched_val_batched_top - batched_val_looped_top,
                    "max_abs_logit_diff_this_row": max_abs_diff,
                }
                break
        if first_diverge_round is not None:
            break

    result["test_b_batch4_vs_looped"] = {
        "match": looped_committed == batched_committed,
        "looped_committed": looped_committed,
        "batched_committed": batched_committed,
        "first_diverge_round": first_diverge_round,
        "logit_gap_report": logit_gap_report,
    }

    result["passed"] = bool(result["test_a_batch1_vs_looped"]["match"])
    return result


def main() -> int:
    import json

    result = _run_once()
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
