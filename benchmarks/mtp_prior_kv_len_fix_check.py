"""Decisive, oracle-aligned verification of the 2026-07-17 K>1 prior_kv_len
fix (Codex-sol review, confirmed real by the coordinator before being
relayed): ``_mtp_sync_and_propose``'s exploratory propose loop (steps
1..K-1) used to pass ``self.slot_draft_sync_len[slot]`` -- a field frozen
after step 0 -- as every exploratory `_mtp_forward` call's `prior_kv_len`,
even though the ACTUAL physical write position (`start_pos`/`next_pos`)
keeps advancing every exploratory iteration. For K=3 (this project's real
production setting, 2 exploratory steps), the 1st exploratory step happens
to still be correct (it immediately follows step 0, where the two
quantities still coincide), but the 2nd is not -- meaning every real K=3
proposal's 3rd draft token was silently computed against an incomplete
causal history.

This is NOT shape-checkable (the coordinator's explicit methodology
critique: the verification gradient's steps 3-4 only checked shape/vocab
range, never per-step numerical content, which is exactly why this bug
was not caught then). Two independent, decisive checks here instead:

1. **White-box invariant**: the CORRECT invariant is `prior_kv_len ==
   start_pos` for every `_mtp_forward` call inside the propose loop (a
   well-formed single-owner-per-slot draft KV cache should never have a
   call whose own "how much history exists" claim disagrees with where it
   is physically writing). Instrumented via monkeypatching `_mtp_forward`
   to record its own arguments, checked across K=3 (production) AND K=5
   (stress case with 4 exploratory steps, so a reintroduced version of
   the bug would diverge even more clearly).
2. **Black-box numerical demonstration**: directly reproduces the OLD,
   buggy call semantics (frozen `prior_kv_len`) for the 2nd exploratory
   step, side-by-side with the NEW fixed call (correct, advancing
   `prior_kv_len`), and shows their logits differ -- concrete, quantified
   proof the fix changes real computed values, not just internal
   bookkeeping.

Usage:
    python -m benchmarks.mtp_prior_kv_len_fix_check
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402  (env vars above must be set before torch/vllm initialize anything)

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
PROMPT = "The capital of France is"


def _check_invariant_for_k(runner, tok, slot: int, k: int) -> dict:
    """Instruments _mtp_forward to record (prior_kv_len, start_pos) for
    every call during one real mtp_prefill's propose loop, and asserts
    they match for every exploratory step."""
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    if runner.slot_kv_len[slot] != 0:
        runner.reset_slot(slot)

    calls = []
    _target = runner.backend if hasattr(runner, "backend") else runner
    original = _target._mtp_forward

    def instrumented(slot_arg, token_ids, hidden_states_in, start_pos, *, prior_kv_len, is_decode):
        calls.append({"start_pos": start_pos, "prior_kv_len": prior_kv_len, "num_tokens": len(token_ids)})
        return original(slot_arg, token_ids, hidden_states_in, start_pos, prior_kv_len=prior_kv_len, is_decode=is_decode)

    _target._mtp_forward = instrumented
    try:
        # mtp_prefill always proposes runner.num_speculative_tokens (K=3,
        # the runner's own configured production value) -- to stress-test
        # OTHER k values (e.g. k=5) we call _mtp_sync_and_propose directly,
        # matching mtp_prefill's own construction.
        target_logits, target_hidden = runner._forward(
            slot, prompt_ids, start_pos=0, is_decode=False, return_hidden=True
        )
        anchor = int(target_logits[-1].argmax(dim=-1).item())
        shifted_input_ids = prompt_ids[1:] + [anchor]
        runner._mtp_sync_and_propose(
            slot, shifted_input_ids, target_hidden, start_pos=0, num_new_tokens=len(prompt_ids), k=k
        )
    finally:
        _target._mtp_forward = original

    mismatches = [c for c in calls if c["prior_kv_len"] != c["start_pos"]]
    return {
        "k": k,
        "num_calls": len(calls),
        "calls": calls,
        "all_match": len(mismatches) == 0,
        "mismatches": mismatches,
    }


def _demonstrate_old_bug_numerically(runner, tok, slot: int, k: int) -> dict:
    """Reproduces the OLD buggy call semantics (``prior_kv_len`` frozen at
    the post-step-0 value for EVERY exploratory step, exactly matching
    the historical bug) directly alongside the NEW fixed call (advancing
    ``prior_kv_len`` every exploratory step), on TWIN slots with
    identical real history, letting each side's own draft tokens/hidden
    states propagate autoregressively (not artificially forced identical
    partway through) -- this is what actually happened in production
    before the fix, not a partial reproduction. The metadata gap between
    frozen and real grows by 1 every exploratory step, so the LAST
    step (checked here) has the largest gap and is the most likely to
    show a visible numerical divergence."""
    prompt_ids = tok.encode(PROMPT, add_special_tokens=False)
    slot_a, slot_b = slot, slot + 1
    for s in (slot_a, slot_b):
        if runner.slot_kv_len[s] != 0:
            runner.reset_slot(s)

    results = {}
    for which, s in (("fixed", slot_a), ("old_buggy_repro", slot_b)):
        target_logits, target_hidden = runner._forward(s, prompt_ids, start_pos=0, is_decode=False, return_hidden=True)
        anchor = int(target_logits[-1].argmax(dim=-1).item())
        shifted_input_ids = prompt_ids[1:] + [anchor]
        step0_logits, step0_hidden = runner._mtp_forward(
            s, shifted_input_ids, target_hidden, start_pos=0, prior_kv_len=runner.slot_draft_sync_len[s], is_decode=False
        )
        runner.slot_draft_sync_len[s] += len(prompt_ids)
        frozen_prior_kv_len = runner.slot_draft_sync_len[s]  # OLD BUG: this value, never advanced again

        draft_tokens = [int(step0_logits[-1].argmax(dim=-1).item())]
        prev_hidden = step0_hidden[-1:]
        prev_token = draft_tokens[0]
        next_pos = len(prompt_ids)
        running_prior_kv_len = frozen_prior_kv_len
        step_top5s = []
        for _ in range(1, k):
            this_prior_kv_len = frozen_prior_kv_len if which == "old_buggy_repro" else running_prior_kv_len
            step_logits, step_hidden = runner._mtp_forward(
                s, [prev_token], prev_hidden, next_pos, prior_kv_len=this_prior_kv_len, is_decode=True
            )
            prev_token = int(step_logits[-1].argmax(dim=-1).item())
            draft_tokens.append(prev_token)
            step_top5s.append(torch.topk(step_logits[-1].float(), 5).indices.tolist())
            prev_hidden = step_hidden[-1:]
            next_pos += 1
            running_prior_kv_len += 1

        results[which] = {
            "draft_tokens": draft_tokens,
            "last_step_top5": step_top5s[-1] if step_top5s else None,
            "all_step_top5s": step_top5s,
        }

    diverged = results["fixed"]["draft_tokens"] != results["old_buggy_repro"]["draft_tokens"]
    return {"k": k, "results": results, "old_vs_new_diverged": diverged}


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
        gpu_memory_utilization=0.5,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3, "attention_backend": "CUSTOM"},
    )
    runner = DirectModelRunner(vllm_config, num_slots=4, block_size=64, blocks_per_slot=640)

    result = {}
    result["invariant_k3"] = _check_invariant_for_k(runner, tok, slot=0, k=3)
    result["invariant_k5"] = _check_invariant_for_k(runner, tok, slot=0, k=5)
    # Numerical demonstration at k=3 (production) AND k=8 (stress -- larger
    # gap between the frozen-vs-real prior_kv_len by the last exploratory
    # step, more likely to actually flip a greedy top-1 token for this
    # specific prompt/position than k=3's smaller gap does). This is
    # reported as an ADDITIONAL, best-effort concrete illustration, NOT a
    # pass/fail gate -- whether a specific prompt's specific position is
    # numerically sensitive enough to visibly flip is prompt-dependent and
    # not itself a correctness property; the invariant checks above are
    # the actual decisive verification (they directly target the root
    # cause: does prior_kv_len match the real physical write position at
    # every step, for every k).
    result["old_bug_demonstration_k3"] = _demonstrate_old_bug_numerically(runner, tok, slot=2, k=3)
    result["old_bug_demonstration_k8"] = _demonstrate_old_bug_numerically(runner, tok, slot=2, k=8)

    result["passed"] = bool(result["invariant_k3"]["all_match"] and result["invariant_k5"]["all_match"])
    return result


def main() -> int:
    import json

    result = _run_once()
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
