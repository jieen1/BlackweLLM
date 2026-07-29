"""Pinpoint diagnostic for the confirmed check2_signal_probe regression:
slots 1 (Japan) and 2 (Germany) commit BYTE-FOR-BYTE IDENTICAL content
starting from round 0 (the very first verify round after prefill) --
not a gradually-accumulating divergence. This script captures raw data
at EVERY pipeline stage for round 0 specifically (prefill anchor/draft
tokens, raw verify logits per slot, accept/reject decision) to find the
EXACT stage where slots 1 and 2's data first becomes identical.

Usage:
    python -m benchmarks.mtp_slot_identity_pinpoint_diag
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

    from oracle.qwen36_vllm.direct_model_runner import (
        DirectModelRunner,
        build_vllm_config,
        determine_accept_reject,
    )

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

    prefill_result = runner.mtp_prefill_batch(slots, prompt_ids_per_slot)
    anchors = {s: prefill_result[s]["anchor"] for s in slots}
    drafts = {s: prefill_result[s]["draft_tokens"] for s in slots}
    report["prefill_anchors"] = anchors
    report["prefill_drafts"] = drafts
    report["anchor_1_eq_2"] = anchors[1] == anchors[2]
    report["drafts_1_eq_2"] = drafts[1] == drafts[2]

    # Manually redo round 0's verify step with extra instrumentation
    # (snapshot/restore around this diagnostic-only extra call so it
    # doesn't corrupt state before the real call below -- same pattern
    # as mtp_batch_divergence_diag.py's own diagnostic convention).
    k = K
    draft_lists = [[anchors[s]] + drafts[s] for s in slots]
    kv_lens_before = [runner.slot_kv_len[s] for s in slots]
    diag_snapshots = {s: runner.snapshot_gdn_state(s) for s in slots}
    verify_logits = runner.verify_batch(slots, draft_lists, kv_lens_before, return_hidden=False)
    report["draft_lists"] = draft_lists
    report["draft_list_1_eq_2"] = draft_lists[1] == draft_lists[2]

    row_logits = {}
    for i, s in enumerate(slots):
        row_logits[s] = verify_logits[i * (k + 1) : (i + 1) * (k + 1)].float().cpu()
    report["verify_logits_1_eq_2"] = bool(torch.equal(row_logits[1], row_logits[2]))
    report["verify_logits_1_argmax"] = [int(row_logits[1][p].argmax(dim=-1).item()) for p in range(k + 1)]
    report["verify_logits_2_argmax"] = [int(row_logits[2][p].argmax(dim=-1).item()) for p in range(k + 1)]
    # Max abs diff per row, to distinguish "bit-identical" from "near-tie
    # coincidentally same argmax".
    report["verify_logits_max_abs_diff"] = float((row_logits[1] - row_logits[2]).abs().max().item())

    for s in slots:
        runner.restore_gdn_state(s, diag_snapshots[s])

    decisions = {}
    for i, s in enumerate(slots):
        decisions[s] = determine_accept_reject(draft_lists[i], verify_logits[i * (k + 1) : (i + 1) * (k + 1)])
    report["decisions_1"] = decisions[1]
    report["decisions_2"] = decisions[2]
    report["decisions_1_eq_2"] = decisions[1]["committed"] == decisions[2]["committed"]

    # Now run the REAL round 0 via the actual coordinator method, to
    # confirm this reproduces what mtp_signal_probe_isolated_repro.py saw.
    real_decisions = runner.mtp_verify_and_commit_batch(slots, anchors, drafts)
    report["real_round0_committed_1"] = real_decisions[1]["committed"]
    report["real_round0_committed_2"] = real_decisions[2]["committed"]
    report["real_round0_1_eq_2"] = real_decisions[1]["committed"] == real_decisions[2]["committed"]

    return report


def main() -> int:
    import json

    result = _run_once()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
