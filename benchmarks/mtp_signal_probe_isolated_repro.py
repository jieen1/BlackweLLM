"""Isolated repro for the still-open check2_signal_probe regression:
runs ONLY the signal-probe scenario (4 slots, organic accept/reject, no
forced anything, no check0/check1 beforehand) on a completely fresh
runner, to determine whether the bug requires prior slot reuse (check0/
check1 already having used these physical slots earlier in the same
process) or reproduces on a truly first-ever use.

Usage:
    python -m benchmarks.mtp_signal_probe_isolated_repro
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

FOUR_PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Germany is",
    "The capital of Italy is",
]


def _run_once() -> dict:
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

    prefill_result = runner.mtp_prefill_batch(slots, prompt_ids_per_slot)
    anchors = {s: prefill_result[s]["anchor"] for s in slots}
    drafts = {s: prefill_result[s]["draft_tokens"] for s in slots}

    per_round_decisions = []
    committed_sequences = {s: [] for s in slots}
    for r in range(NUM_ROUNDS):
        decisions = runner.mtp_verify_and_commit_batch(slots, anchors, drafts)
        round_info = {"round": r}
        for s in slots:
            round_info[s] = {
                "num_accepted": decisions[s]["num_accepted"],
                "committed": decisions[s]["committed"],
            }
            committed_sequences[s].extend(decisions[s]["committed"])
            anchors[s], drafts[s] = decisions[s]["next_anchor"], decisions[s]["next_draft_tokens"]
        per_round_decisions.append(round_info)

    seqs = list(committed_sequences.values())
    no_cross_contamination = all(
        seqs[i] != seqs[j] for i in range(len(seqs)) for j in range(i + 1, len(seqs))
    )
    decoded = {s: tok.decode(committed_sequences[s]) for s in slots}

    return {
        "no_cross_contamination_signal": no_cross_contamination,
        "decoded_completions": decoded,
        "committed_sequences": committed_sequences,
        "per_round_decisions": per_round_decisions,
    }


def main() -> int:
    import json

    result = _run_once()
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["no_cross_contamination_signal"] else 1


if __name__ == "__main__":
    sys.exit(main())
