"""Diagnostic for hypothesis (1) in the coordinator's remaining-8.7x-gap
investigation: is the cross-slot batched MTP coordinator ACTUALLY fully
batched in practice, or does something make it silently degrade back
toward per-slot execution?

``mtp_verify_and_commit_batch`` always batches the VERIFY step across
every active slot, but its post-verify draft catch-up+propose step only
batches the FULL-ACCEPT subgroup -- any slot with a partial reject
("needs-recompute") falls back to the EXISTING single-slot path
(``_forward_batch([slot],...)`` + ``_mtp_sync_and_propose(slot,...)``,
called once per affected slot, in a Python loop). The design doc's own
stated assumption when this was built was "batch the common case
(full-accept), fall back for the uncommon case (recompute)" -- this
script directly measures whether recompute is actually uncommon at the
W1-S shape's real ~68% draft acceptance rate, and whether recompute-heavy
rounds are disproportionately expensive (the two things needed to confirm
or refute that assumption).

For every real ``mtp_verify_and_commit_batch`` round on the W1-S fixture
(same n16/max_tokens=256/concurrency=4 shape as the perf comparison),
records: how many of the round's active slots needed recompute
(0..concurrency), and this round's total wall-clock time. Reports the
round-count histogram by recompute-slot-count and the mean/total
wall-clock time per bucket -- if recompute-heavy rounds are common AND
disproportionately slow, that is direct, decisive evidence of where a
chunk of the remaining gap lives (not requiring ncu/nsys to establish
this specific point, though nsys is still useful for the kernel-level
breakdown of what stays after this cause is quantified).

Usage:
    python -m benchmarks.mtp_batch_recompute_cost_diag
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3
MAX_TOKENS = 256
CONCURRENCY = 4


def _run_batch(torch, runner, prompts_batch: list[list[int]], target_output_len: int) -> list[dict]:
    num = len(prompts_batch)
    slots = list(range(num))
    for s in slots:
        if runner.slot_kv_len[s] != 0:
            runner.reset_slot(s)

    prefill_result = runner.mtp_prefill_batch(slots, prompts_batch)
    anchors = {s: prefill_result[s]["anchor"] for s in slots}
    drafts = {s: prefill_result[s]["draft_tokens"] for s in slots}
    committed_len = {s: 0 for s in slots}

    rounds = []
    active = list(slots)
    while active:
        t0 = time.perf_counter()
        decisions = runner.mtp_verify_and_commit_batch(
            active, {s: anchors[s] for s in active}, {s: drafts[s] for s in active}
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        round_wall_s = t1 - t0

        num_recompute = 0
        newly_finished = []
        for s in active:
            decision = decisions[s]
            n_acc = decision["num_accepted"]
            if n_acc != K:
                num_recompute += 1
            committed_len[s] += n_acc + 1
            anchors[s], drafts[s] = decision["next_anchor"], decision["next_draft_tokens"]
            if committed_len[s] >= target_output_len:
                newly_finished.append(s)
        rounds.append(
            {
                "num_active_slots": len(active),
                "num_recompute_slots": num_recompute,
                "round_wall_s": round_wall_s,
            }
        )
        for s in newly_finished:
            active.remove(s)

    return rounds


def _run_once() -> dict:
    import torch

    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401

    from benchmarks.workloads import W1_S_FIXTURE, load_prompt_token_ids
    from runtime.direct_model_runner import DirectModelRunner, build_vllm_config

    prompts = load_prompt_token_ids(W1_S_FIXTURE)

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=max(40960, W1_S_FIXTURE.prompt_len + MAX_TOKENS + 1024),
        gpu_memory_utilization=0.85,
        speculative_config={"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"},
    )
    runner = DirectModelRunner(vllm_config, num_slots=CONCURRENCY, block_size=64, blocks_per_slot=640)

    all_rounds = []
    for batch_start in range(0, len(prompts), CONCURRENCY):
        batch = prompts[batch_start : batch_start + CONCURRENCY]
        all_rounds.extend(_run_batch(torch, runner, batch, MAX_TOKENS))
        print(f"  ... batch at offset {batch_start} done ({len(all_rounds)} rounds so far)", flush=True)

    # Bucket by num_recompute_slots (0 = fully batched round, 4 = every
    # active slot needed the single-slot fallback this round).
    buckets: dict[int, list[float]] = {}
    for r in all_rounds:
        buckets.setdefault(r["num_recompute_slots"], []).append(r["round_wall_s"])

    bucket_report = {}
    total_wall_s = sum(r["round_wall_s"] for r in all_rounds)
    for n_recompute, times in sorted(buckets.items()):
        bucket_report[n_recompute] = {
            "num_rounds": len(times),
            "pct_of_rounds": len(times) / len(all_rounds) * 100.0,
            "mean_round_wall_ms": sum(times) / len(times) * 1000.0,
            "total_wall_s_this_bucket": sum(times),
            "pct_of_total_wall_time": sum(times) / total_wall_s * 100.0 if total_wall_s > 0 else float("nan"),
        }

    fully_batched_rounds = buckets.get(0, [])
    any_recompute_rounds = [r["round_wall_s"] for r in all_rounds if r["num_recompute_slots"] > 0]

    return {
        "total_rounds": len(all_rounds),
        "total_wall_s": total_wall_s,
        "bucket_report": bucket_report,
        "pct_rounds_with_any_recompute": len(any_recompute_rounds) / len(all_rounds) * 100.0,
        "mean_wall_ms_fully_batched_rounds": (
            sum(fully_batched_rounds) / len(fully_batched_rounds) * 1000.0 if fully_batched_rounds else float("nan")
        ),
        "mean_wall_ms_any_recompute_rounds": (
            sum(any_recompute_rounds) / len(any_recompute_rounds) * 1000.0 if any_recompute_rounds else float("nan")
        ),
    }


def main() -> int:
    import json

    result = _run_once()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
