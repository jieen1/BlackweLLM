"""Measures THIS RUNTIME's own real MTP acceptance rate on a W1 or
W2-shaped workload (concurrency=4, real random-token prompts matching
real vLLM's own `RandomDataset` convention -- deterministic pseudo-random
token ids, NOT real coherent text, so the input distribution is
comparable to native vLLM's `--dataset-name random` acceptance-rate
measurement, not artificially inflated by real language's stronger local
predictability), using the ALREADY-VERIFIED-CORRECT `mtp_prefill`/
`mtp_verify_and_commit` machinery as a black box (this script measures
throughput/acceptance, it does not re-verify correctness -- that was
done across the prior several rounds).

Reports using the EXACT SAME formulas vLLM's own
`vllm/v1/spec_decode/metrics.py`'s `SpecDecodingLogging.log()` uses
(`draft_acceptance_rate = num_accepted_tokens/num_draft_tokens*100`,
`mean_acceptance_length = 1 + num_accepted_tokens/num_drafts`), so this
number is directly comparable to the "Avg Draft acceptance rate"/"Mean
acceptance length" native vLLM logs for the SAME workload shape.

Round-robin across 4 slots (not batched into one call -- this runtime's
current MTP coordinator methods are single-slot; see
notes/direct-model-runner-design.md's step-6 scope note) until each slot
has committed at least `target_output_len` tokens.

Usage:
    python -m benchmarks.mtp_our_runtime_acceptance --workload w1
    python -m benchmarks.mtp_our_runtime_acceptance --workload w2
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3
CONCURRENCY = 4
SEED = 12345

WORKLOADS = {
    "w1": {"input_len": 4096, "output_len": 1024},
    "w2": {"input_len": 32768, "output_len": 1024},
}


def _random_prompt(allowed_tokens: list[int], length: int, offset: int) -> list[int]:
    """Matches real vLLM's own `RandomDataset.generate_token_sequence()`
    formula EXACTLY (`allowed_tokens[(offset + index + arange(input_len))
    % len(allowed_tokens)]`) -- 2026-07-17 fix: an earlier version of this
    function used i.i.d. uniform random token sampling instead, a
    meaningfully DIFFERENT (and less locally-predictable) distribution
    than vLLM's actual sequential-run-of-ascending-ids scheme, found to
    be a real confound (alongside a temperature-sampling mismatch) in
    this round's first W1 comparison attempt -- see
    notes/direct-model-runner-design.md. Using the SAME formula here
    means both sides measure acceptance rate against the SAME input
    distribution, not just the same input LENGTH."""
    n = len(allowed_tokens)
    return [allowed_tokens[(offset + i) % n] for i in range(length)]


def _run_once(workload: str, target_output_len: int, blocks_per_slot: int) -> dict:

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner, build_vllm_config

    cfg = WORKLOADS[workload]
    input_len = cfg["input_len"]

    tok = AutoTokenizer.from_pretrained(MODEL)
    vocab_size = tok.vocab_size
    special_ids = set(tok.all_special_ids)
    allowed_tokens = [t for t in range(vocab_size) if t not in special_ids]
    rng = random.Random(SEED)

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=max(40960, input_len + target_output_len + 1024),
        gpu_memory_utilization=0.85,
        speculative_config={"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"},
    )
    runner = DirectModelRunner(
        vllm_config, num_slots=CONCURRENCY, block_size=64, blocks_per_slot=blocks_per_slot
    )

    slots = list(range(CONCURRENCY))
    committed_len = {s: 0 for s in slots}
    anchor = {}
    draft_tokens = {}

    t_prefill_start = time.perf_counter()
    for s in slots:
        offset = rng.randrange(len(allowed_tokens))
        prompt_ids = _random_prompt(allowed_tokens, input_len, offset)
        pr = runner.mtp_prefill(s, prompt_ids)
        anchor[s] = pr["anchor"]
        draft_tokens[s] = pr["draft_tokens"]
    t_prefill_end = time.perf_counter()

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    accepted_per_pos = [0] * K
    finished = set()
    round_idx = 0
    t_rounds_start = time.perf_counter()

    while len(finished) < CONCURRENCY:
        round_idx += 1
        for s in slots:
            if s in finished:
                continue
            decision = runner.mtp_verify_and_commit(s, anchor[s], draft_tokens[s])
            n_acc = decision["num_accepted"]
            num_drafts += 1
            num_draft_tokens += K
            num_accepted_tokens += n_acc
            for p in range(n_acc):
                accepted_per_pos[p] += 1
            committed_len[s] += n_acc + 1
            anchor[s], draft_tokens[s] = decision["next_anchor"], decision["next_draft_tokens"]
            if committed_len[s] >= target_output_len:
                finished.add(s)
        if round_idx % 20 == 0:
            done = sum(committed_len.values())
            print(
                f"  ... round {round_idx}: {done}/{target_output_len * CONCURRENCY} tokens "
                f"committed across {CONCURRENCY} slots ({len(finished)} finished)",
                flush=True,
            )

    t_rounds_end = time.perf_counter()

    draft_acceptance_rate = (
        num_accepted_tokens / num_draft_tokens * 100.0 if num_draft_tokens > 0 else float("nan")
    )
    mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else float("nan")
    per_position_rate = [c / num_drafts for c in accepted_per_pos] if num_drafts > 0 else []

    wall_s = t_rounds_end - t_rounds_start
    total_committed = sum(committed_len.values())

    return {
        "workload": workload,
        "input_len": input_len,
        "target_output_len_per_slot": target_output_len,
        "concurrency": CONCURRENCY,
        "k": K,
        "prefill_wall_s": t_prefill_end - t_prefill_start,
        "rounds_wall_s": wall_s,
        "num_rounds_total_calls": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "mean_acceptance_length": mean_acceptance_length,
        "draft_acceptance_rate_pct": draft_acceptance_rate,
        "per_position_acceptance_rate": per_position_rate,
        "total_committed_tokens": total_committed,
        "accepted_tokens_per_sec": total_committed / wall_s if wall_s > 0 else float("nan"),
        "committed_len_by_slot": committed_len,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["w1", "w2"], required=True)
    parser.add_argument("--target-output-len", type=int, default=None, help="override per-slot output length")
    parser.add_argument("--blocks-per-slot", type=int, default=2560)
    args = parser.parse_args()

    target_output_len = args.target_output_len or WORKLOADS[args.workload]["output_len"]
    result = _run_once(args.workload, target_output_len, args.blocks_per_slot)

    import json

    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
