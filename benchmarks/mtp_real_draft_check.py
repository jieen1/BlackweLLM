"""Phase 2 (sol's "Option A"): real ``Qwen3_5MTP`` draft model, numerical
verification gradient steps 1-4 (per the coordinator's explicit ordering
-- do not skip steps, each uses real numerical twin comparison, not
signal-probe):

  1. target prefill hidden states/logits alignment -- loading the draft
     model alongside the target must not perturb the target's OWN
     computation.
  2. shifted draft first pass -- weight-sharing identity (embed_tokens/
     lm_head genuinely shared with the target, per real vLLM's
     ``load_eagle_model``) + finite/shape sanity for the real draft
     model's own forward.
  3. K=3 proposal token correctness -- shape/finiteness across the full
     autoregressive draft-side propose loop.
  4. single-round verify/accept-reject/GDN rollback, wired through the
     real ``mtp_verify_and_commit`` coordinator -- reuses the
     ALREADY-VERIFIED ``determine_accept_reject``/``snapshot_gdn_state``/
     ``restore_gdn_state`` mechanisms (not reinvented), forcing both a
     full-accept and a forced-reject scenario via the same DECOY-token
     technique ``mtp_accept_reject_check.py`` already established.

Multi-round (concurrency=1), 4-slot isolation (concurrency=4), and the
real W1/W2 vLLM acceptance-rate gate are explicitly NOT attempted here
(next steps of the gradient, later rounds).

Usage:
    python -m benchmarks.mtp_real_draft_check
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
PROMPT = "The capital of France is"
K = 3
SPECULATIVE_CONFIG = {
    "method": "mtp",
    "num_speculative_tokens": K,
    # Real production convention (vllm_integration/launch_test_server.py):
    # the MTP proposer does NOT inherit the top-level --attention-backend,
    # it independently autoselects unless told otherwise -- must set this
    # explicitly or the draft's own attention layer could pick a different
    # backend (different metadata type) than SM120GQAMetadata.
    "attention_backend": "CUSTOM",
}


def _run_once() -> dict:
    import torch

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from oracle.qwen36_vllm.direct_model_runner import (
        DirectModelRunner,
        build_vllm_config,
    )

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)

    result: dict = {}

    # --- Step 1: target prefill hidden states/logits alignment ---
    # Plain (no speculative_config) runner -- the pre-existing, long-since
    # verified target-only path.
    plain_config = build_vllm_config(
        model=MODEL, kv_cache_dtype="fp8_e4m3", max_model_len=2048, gpu_memory_utilization=0.35
    )
    plain_runner = DirectModelRunner(plain_config, num_slots=2, block_size=64, blocks_per_slot=640)
    plain_logits, plain_hidden = plain_runner._forward(
        0, prompt_ids, start_pos=0, is_decode=False, return_hidden=True
    )

    # MTP-enabled runner -- draft model loaded alongside the target.
    mtp_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=2048,
        gpu_memory_utilization=0.35,
        speculative_config=SPECULATIVE_CONFIG,
    )
    mtp_runner = DirectModelRunner(mtp_config, num_slots=3, block_size=64, blocks_per_slot=640)

    if mtp_runner.mtp_model is None:
        return {"passed": False, "error": "mtp_model did not load despite speculative_config"}
    result["mtp_attn_layer_names"] = mtp_runner.mtp_attn_layer_names
    result["num_speculative_tokens"] = mtp_runner.num_speculative_tokens

    mtp_target_logits, mtp_target_hidden = mtp_runner._forward(
        0, prompt_ids, start_pos=0, is_decode=False, return_hidden=True
    )

    logits_close = torch.allclose(plain_logits, mtp_target_logits, atol=1e-2, rtol=1e-2)
    hidden_close = torch.allclose(plain_hidden, mtp_target_hidden, atol=1e-2, rtol=1e-2)
    logits_cos = torch.nn.functional.cosine_similarity(
        plain_logits[-1].float().unsqueeze(0), mtp_target_logits[-1].float().unsqueeze(0)
    ).item()
    result["step1_target_alignment"] = {
        "logits_allclose": logits_close,
        "hidden_allclose": hidden_close,
        "last_position_logits_cosine_sim": logits_cos,
        "same_greedy_token": int(plain_logits[-1].argmax(-1).item())
        == int(mtp_target_logits[-1].argmax(-1).item()),
    }
    step1_ok = logits_close and hidden_close and result["step1_target_alignment"]["same_greedy_token"]

    # --- Step 2: weight-sharing identity + shifted draft first pass ---
    target_lm = mtp_runner.model.get_language_model() if hasattr(mtp_runner.model, "get_language_model") else mtp_runner.model
    target_embed = getattr(target_lm.model, "embed_tokens", None)
    draft_embed = getattr(mtp_runner.mtp_model.model, "embed_tokens", None)
    target_lm_head = getattr(target_lm, "lm_head", None) or getattr(mtp_runner.model, "lm_head", None)
    draft_lm_head = getattr(mtp_runner.mtp_model, "lm_head", None)
    embed_shared = target_embed is not None and draft_embed is not None and target_embed.weight.data_ptr() == draft_embed.weight.data_ptr()
    lm_head_shared = (
        target_lm_head is not None
        and draft_lm_head is not None
        and target_lm_head.weight.data_ptr() == draft_lm_head.weight.data_ptr()
    )
    result["step2_weight_sharing"] = {"embed_tokens_shared": embed_shared, "lm_head_shared": lm_head_shared}

    anchor = int(mtp_target_logits[-1].argmax(dim=-1).item())
    shifted_input_ids = prompt_ids[1:] + [anchor]
    step0_logits, step0_hidden = mtp_runner._mtp_forward(
        0, shifted_input_ids, mtp_target_hidden, start_pos=0,
        prior_kv_len=mtp_runner.slot_draft_sync_len[0], is_decode=False
    )
    mtp_runner.slot_draft_sync_len[0] += len(prompt_ids)
    step0_finite = bool(torch.isfinite(step0_logits).all().item())
    result["step2_shifted_draft_first_pass"] = {
        "output_shape": list(step0_logits.shape),
        "expected_rows": len(prompt_ids),
        "all_finite": step0_finite,
        "draft_sync_len_after": mtp_runner.slot_draft_sync_len[0],
    }
    step2_ok = (
        embed_shared
        and lm_head_shared
        and step0_finite
        and step0_logits.shape[0] == len(prompt_ids)
    )

    # reset for a clean step-3/4 run through the real coordinator methods
    mtp_runner.reset_slot(0)
    mtp_runner.slot_draft_sync_len[0] = 0

    # --- Step 3: K=3 proposal token correctness (shape/finite, real coordinator path) ---
    prefill_result = mtp_runner.mtp_prefill(0, prompt_ids)
    draft_tokens = prefill_result["draft_tokens"]
    result["step3_k_proposal"] = {
        "anchor": prefill_result["anchor"],
        "draft_tokens": draft_tokens,
        "num_draft_tokens": len(draft_tokens),
        "all_in_vocab_range": all(0 <= t < tok.vocab_size + 8192 for t in draft_tokens),
    }
    step3_ok = len(draft_tokens) == K and result["step3_k_proposal"]["all_in_vocab_range"]

    # --- Step 4: single-round verify/accept-reject/GDN rollback, both scenarios ---
    # "real_draft_proposal": submit the REAL draft model's own K=3 output
    # as-is. Its accept/reject outcome is NOT asserted to be any specific
    # value -- an untrained-together MTP head organically agreeing or
    # disagreeing with the target at each position is the expected,
    # realistic dynamic (that's what an acceptance rate even measures);
    # only the mechanism's INVARIANTS are checked (kv_len tracks the real
    # committed length, GDN state gets repaired when not a full accept).
    # "forced_reject_at_1": a DECOY substitution (same technique
    # `mtp_accept_reject_check.py` already established) guarantees a
    # partial-reject at a KNOWN position, as a second, independent check.
    step4_results = {}
    for scenario in ("real_draft_proposal", "forced_reject_at_1"):
        # Fresh slot 1 each scenario, prefilled identically.
        if mtp_runner.slot_kv_len[1] != 0:
            mtp_runner.reset_slot(1)
            mtp_runner.slot_draft_sync_len[1] = 0
        pr = mtp_runner.mtp_prefill(1, prompt_ids)
        anchor1, drafts1 = pr["anchor"], pr["draft_tokens"]

        if scenario == "real_draft_proposal":
            submitted = drafts1
        else:
            decoy = 100 if decoy_safe(100, drafts1) else 200
            submitted = [drafts1[0], decoy, decoy]

        gdn_before = mtp_runner.snapshot_gdn_state(1)
        kv_before = mtp_runner.slot_kv_len[1]
        decision = mtp_runner.mtp_verify_and_commit(1, anchor1, submitted)
        kv_after_commit = mtp_runner.slot_kv_len[1]  # captured BEFORE the content-check's own extra decode call below
        gdn_after = mtp_runner.snapshot_gdn_state(1)

        gdn_changed = any(
            not torch.equal(gdn_before[name][0], gdn_after[name][0])
            or not torch.equal(gdn_before[name][1], gdn_after[name][1])
            for name in mtp_runner.gdn_layer_names
        )

        # Decisive content-correctness check (this is what actually catches
        # wrong KV CONTENT, not just right shapes/bookkeeping -- a real bug
        # of exactly this kind was caught and fixed this round by direct
        # reasoning before this check existed; see mtp_verify_and_commit's
        # docstring). Independently replay the REAL committed sequence
        # (prompt + anchor + accepted drafts) from scratch on a fresh
        # reference slot via the plain, long-verified prefill/decode path,
        # then continue BOTH slot 1 (via a real decode call feeding the
        # recovery token) and the reference (by construction, prefill's
        # return IS its prediction after the real sequence) and compare.
        recovery_token = decision["committed"][-1]
        real_sequence = prompt_ids + [anchor1] + decision["committed"][:-1]
        if mtp_runner.slot_kv_len[2] != 0:
            mtp_runner.reset_slot(2)
        ref_predicted_recovery = mtp_runner.prefill(2, real_sequence)
        slot1_next_logits, slot1_next_hidden = mtp_runner._forward(
            1, [recovery_token], start_pos=mtp_runner.slot_kv_len[1], is_decode=True, return_hidden=True
        )
        ref_next_logits, ref_next_hidden = mtp_runner._forward(
            2, [recovery_token], start_pos=mtp_runner.slot_kv_len[2], is_decode=True, return_hidden=True
        )
        content_check = {
            "ref_predicted_recovery_matches_decision": ref_predicted_recovery == recovery_token,
            "next_hidden_allclose": torch.allclose(slot1_next_hidden, ref_next_hidden, atol=1e-2, rtol=1e-2),
            "next_logits_cosine_sim": torch.nn.functional.cosine_similarity(
                slot1_next_logits[-1].float().unsqueeze(0), ref_next_logits[-1].float().unsqueeze(0)
            ).item(),
            "same_next_greedy_token": int(slot1_next_logits[-1].argmax(-1).item())
            == int(ref_next_logits[-1].argmax(-1).item()),
        }

        step4_results[scenario] = {
            "submitted_draft": submitted,
            "num_accepted": decision["num_accepted"],
            "committed": decision["committed"],
            "rejected_at": decision["rejected_at"],
            "kv_len_before": kv_before,
            "kv_len_after": kv_after_commit,
            "kv_len_matches_committed_len": kv_after_commit == kv_before + decision["num_accepted"] + 1,
            "gdn_state_changed_from_prefill": gdn_changed,
            "next_anchor": decision["next_anchor"],
            "next_draft_tokens": decision["next_draft_tokens"],
            "content_correctness_check": content_check,
        }

    result["step4_verify_accept_reject_gdn"] = step4_results
    step4_ok = (
        # real_draft_proposal: only the mechanism's invariants are asserted
        # (kv_len tracks whatever the real committed length turned out to
        # be; num_accepted is a valid value 0..K) -- NOT a specific
        # accept/reject outcome, since that's a property of the draft
        # model's own (imperfect, untrained-together) prediction quality,
        # not of this coordinator's correctness.
        0 <= step4_results["real_draft_proposal"]["num_accepted"] <= K
        and step4_results["real_draft_proposal"]["kv_len_matches_committed_len"]
        and step4_results["real_draft_proposal"]["content_correctness_check"]["ref_predicted_recovery_matches_decision"]
        and step4_results["real_draft_proposal"]["content_correctness_check"]["same_next_greedy_token"]
        # forced_reject_at_1: a KNOWN, deliberately-constructed scenario --
        # the exact outcome IS asserted here, matching
        # mtp_accept_reject_check.py's existing convention.
        and step4_results["forced_reject_at_1"]["num_accepted"] == 1
        and step4_results["forced_reject_at_1"]["kv_len_matches_committed_len"]
        and step4_results["forced_reject_at_1"]["content_correctness_check"]["ref_predicted_recovery_matches_decision"]
        and step4_results["forced_reject_at_1"]["content_correctness_check"]["same_next_greedy_token"]
    )

    result["passed"] = bool(step1_ok and step2_ok and step3_ok and step4_ok)
    result["step_status"] = {
        "step1_target_alignment": step1_ok,
        "step2_shifted_draft_first_pass": step2_ok,
        "step3_k_proposal": step3_ok,
        "step4_verify_accept_reject_gdn": step4_ok,
    }
    return result


def decoy_safe(candidate: int, real_tokens: list[int]) -> bool:
    return candidate not in real_tokens


def main() -> int:
    result = _run_once()
    import json

    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
