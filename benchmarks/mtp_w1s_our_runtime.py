"""This runtime's side of the W1-S (controlled synthetic) acceptance-rate
comparison. Loads the SAME frozen, versioned prompt token ids
(`benchmarks/fixtures/w1s_prompts.json`) `w1s_native_bench.py` sends to
native vLLM -- both sides provably run the IDENTICAL input, not just "the
same distribution."

Processes all `num_requests` prompts in sequential batches of `concurrency`
(4 slots active at a time, matching `W1_S.concurrency`), reusing slots
across batches via `reset_slot` -- this is "more independent trajectories
via more batches, not larger concurrency," per the coordinator's explicit
instruction. Reports both the aggregate acceptance rate (directly
comparable to native's own aggregate) AND a per-trajectory breakdown (each
of the 16 requests' own acceptance rate), so a skewed aggregate caused by
one or two outlier trajectories is visible, not hidden.

Usage:
    python -m benchmarks.mtp_w1s_our_runtime --max-tokens 256 --concurrency 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3


def _run_batch(runner, prompts_batch: list[list[int]], target_output_len: int) -> list[dict]:
    """Runs one batch of len(prompts_batch) <= runner.num_slots requests to
    completion (each committing target_output_len tokens), round-robin
    across the batch's own slots -- same mechanism as
    mtp_multiround_check.py's 4-slot isolation test."""
    num = len(prompts_batch)
    slots = list(range(num))
    for s in slots:
        if runner.slot_kv_len[s] != 0:
            runner.reset_slot(s)

    anchor = {}
    draft_tokens = {}
    committed_len = {s: 0 for s in slots}
    per_slot_stats = {s: {"num_drafts": 0, "num_draft_tokens": 0, "num_accepted_tokens": 0} for s in slots}

    for i, s in enumerate(slots):
        pr = runner.mtp_prefill(s, prompts_batch[i])
        anchor[s] = pr["anchor"]
        draft_tokens[s] = pr["draft_tokens"]

    finished = set()
    while len(finished) < num:
        for s in slots:
            if s in finished:
                continue
            decision = runner.mtp_verify_and_commit(s, anchor[s], draft_tokens[s])
            n_acc = decision["num_accepted"]
            per_slot_stats[s]["num_drafts"] += 1
            per_slot_stats[s]["num_draft_tokens"] += K
            per_slot_stats[s]["num_accepted_tokens"] += n_acc
            committed_len[s] += n_acc + 1
            anchor[s], draft_tokens[s] = decision["next_anchor"], decision["next_draft_tokens"]
            if committed_len[s] >= target_output_len:
                finished.add(s)

    return [per_slot_stats[s] for s in slots]


def _run_once(max_tokens: int, concurrency: int, fixture_key: str, num_requests: int | None = None) -> dict:
    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401

    from benchmarks.workloads import W1_S_FIXTURE, W1_S_FIXTURE_N128, load_prompt_token_ids
    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    fixture = {"n16": W1_S_FIXTURE, "n128": W1_S_FIXTURE_N128}[fixture_key]
    prompts = load_prompt_token_ids(fixture)
    if num_requests is not None:
        # A prefix of an already-frozen fixture is still exactly
        # reproducible (no new fixture file needed) -- the frozen file's
        # own generation formula produces request i's prompt independent
        # of how many total requests follow it.
        prompts = prompts[:num_requests]

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=max(40960, fixture.prompt_len + max_tokens + 1024),
        gpu_memory_utilization=0.85,
        speculative_config={"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"},
    )
    runner = DirectModelRunner(vllm_config, num_slots=concurrency, block_size=64, blocks_per_slot=640)

    per_trajectory: list[dict] = []
    t_start = time.perf_counter()
    num_batches = (len(prompts) + concurrency - 1) // concurrency
    for batch_idx, batch_start in enumerate(range(0, len(prompts), concurrency)):
        batch = prompts[batch_start : batch_start + concurrency]
        batch_stats = _run_batch(runner, batch, max_tokens)
        per_trajectory.extend(batch_stats)
        elapsed = time.perf_counter() - t_start
        print(
            f"  ... batch {batch_idx + 1}/{num_batches} done "
            f"({len(per_trajectory)}/{len(prompts)} trajectories, {elapsed:.0f}s elapsed)",
            flush=True,
        )

    wall_s = time.perf_counter() - t_start
    total_drafts = sum(t["num_drafts"] for t in per_trajectory)
    total_draft_tokens = sum(t["num_draft_tokens"] for t in per_trajectory)
    total_accepted = sum(t["num_accepted_tokens"] for t in per_trajectory)

    per_trajectory_rates = [
        {
            "num_drafts": t["num_drafts"],
            "acceptance_rate_pct": (
                t["num_accepted_tokens"] / t["num_draft_tokens"] * 100.0 if t["num_draft_tokens"] > 0 else float("nan")
            ),
        }
        for t in per_trajectory
    ]

    return {
        "passed": True,
        "num_requests": len(prompts),
        "max_tokens": max_tokens,
        "concurrency": concurrency,
        "k": K,
        "wall_s": wall_s,
        "num_drafts": total_drafts,
        "num_draft_tokens": total_draft_tokens,
        "num_accepted_tokens": total_accepted,
        "draft_acceptance_rate_pct": total_accepted / total_draft_tokens * 100.0 if total_draft_tokens > 0 else float("nan"),
        "mean_acceptance_length": 1 + (total_accepted / total_drafts) if total_drafts > 0 else float("nan"),
        "per_trajectory_acceptance_rate_pct": per_trajectory_rates,
        "fixture": fixture.path,
        "fixture_seed": fixture.seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--fixture", choices=["n16", "n128"], default="n16")
    parser.add_argument("--num-requests", type=int, default=None, help="use only the first N of the fixture's prompts")
    args = parser.parse_args()

    result = _run_once(args.max_tokens, args.concurrency, args.fixture, args.num_requests)

    import json

    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
