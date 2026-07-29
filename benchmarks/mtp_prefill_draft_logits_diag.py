"""Follow-up to mtp_slot_identity_pinpoint_diag.py: that diagnostic showed
the TARGET model's own verify-step logits genuinely differ between slots
1 and 2 (max_abs_diff=9.98, ruling out literal shared-memory/duplicated
data for that computation), yet both slots' argmax sequences coincided,
producing identical committed content -- consistent with genuine model
coincidence (short generic continuations converging on common punctuation
tokens), not necessarily a bug.

This script checks the OTHER candidate location for an index/addressing
bug: the DRAFT model's OWN step-0 sync-forward during ``mtp_prefill_batch``
(the call that produced IDENTICAL draft tokens `[13, 248046, 198]` for
slots 1 and 2). It manually replicates ``mtp_prefill_batch``'s exact logic
but calls ``_mtp_forward_batch`` directly for the draft model's step-0
call to capture RAW step-0 logits per slot (not just their final argmax
draft-token choices), checking whether slot 1's and slot 2's raw
step-0 logits are:
- substantially different (like the target model's verify step) --
  supports genuine coincidence (same mechanism, different stage), or
- suspiciously near-identical/bit-identical -- supports a real
  indexing/addressing bug specific to the draft model's hidden-state
  slicing during batched prefill.

Usage:
    python -m benchmarks.mtp_prefill_draft_logits_diag
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

FOUR_PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Germany is",
    "The capital of Italy is",
]


def _run_once() -> dict:
    import torch

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner, build_vllm_config

    tok = AutoTokenizer.from_pretrained(MODEL)
    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=2048,
        gpu_memory_utilization=0.6,
        speculative_config=SPECULATIVE_CONFIG,
    )
    runner = DirectModelRunner(vllm_config, num_slots=8, block_size=64, blocks_per_slot=640)

    slots = [0, 1, 2, 3]
    prompt_ids_per_slot = [tok.encode(p, add_special_tokens=False) for p in FOUR_PROMPTS]
    report = {"prompt_ids_per_slot": prompt_ids_per_slot}

    # Replicates mtp_prefill_batch's own logic exactly, up through the
    # target-model forward (unmodified, real call) -- then manually
    # invokes the draft model's step-0 call directly (instead of via
    # _mtp_sync_and_propose_batch) so the RAW logits can be captured
    # before any argmax collapses them to a single token id.
    num_reqs = len(slots)
    prompt_len = len(prompt_ids_per_slot[0])
    target_logits, target_hidden = runner._forward_batch(
        slots, prompt_ids_per_slot, [0] * num_reqs,
        qo_len=prompt_len, commit=True, return_hidden=True, is_decode=False,
    )
    anchors = {}
    shifted_per_slot = []
    for i, s in enumerate(slots):
        row = target_logits[i * prompt_len + prompt_len - 1]
        anchor = int(row.argmax(dim=-1).item())
        anchors[s] = anchor
        shifted_per_slot.append(prompt_ids_per_slot[i][1:] + [anchor])
    report["anchors"] = anchors

    prior_kv_lens_step0 = [runner.slot_draft_sync_len[s] for s in slots]
    step0_logits, step0_hidden = runner._mtp_forward_batch(
        slots, shifted_per_slot, target_hidden, prior_kv_lens_step0, [0] * num_reqs,
        qo_len=prompt_len, is_decode=(prompt_len == 1),
    )
    # step0_logits is [sum(qo_lens), vocab] in request-then-position order;
    # each slot's OWN last row (position prompt_len-1) is what produces its
    # first draft token.
    last_row_logits = {}
    for i, s in enumerate(slots):
        last_idx = i * prompt_len + prompt_len - 1
        last_row_logits[s] = step0_logits[last_idx].float().cpu()

    report["draft_step0_argmax"] = {s: int(last_row_logits[s].argmax(dim=-1).item()) for s in slots}
    report["draft_step0_logits_1_eq_2"] = bool(torch.equal(last_row_logits[1], last_row_logits[2]))
    report["draft_step0_logits_max_abs_diff_1_vs_2"] = float(
        (last_row_logits[1] - last_row_logits[2]).abs().max().item()
    )
    report["draft_step0_logits_max_abs_diff_1_vs_3"] = float(
        (last_row_logits[1] - last_row_logits[3]).abs().max().item()
    )
    # Top-2 margin for slot 1 and slot 2's OWN distributions -- if either
    # is a near-exact tie between its own top-1 and top-2 candidates, a
    # coincidental cross-slot argmax match is far more plausible (matches
    # this project's own established near-tie precedent) than if both
    # slots have a large, confident margin.
    for s in (1, 2):
        vals, idxs = last_row_logits[s].topk(2)
        report[f"slot{s}_own_top1_top2_margin"] = float((vals[0] - vals[1]).item())
        report[f"slot{s}_own_top1_token"] = int(idxs[0].item())
        report[f"slot{s}_own_top2_token"] = int(idxs[1].item())

    return report


def main() -> int:
    import json

    result = _run_once()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
