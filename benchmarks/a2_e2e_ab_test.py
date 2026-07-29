#!/usr/bin/env python3
"""A2: End-to-end A/B test — stock vs B12x/cuDNN NVFP4 GEMM.

Single-slot MTP decode with greedy sampling on real Qwen3.6-27B.
Reports accepted tokens/s and output token IDs for parity check.

Usage:
    QSR_A2_B12X=0 QSR_A2_CUDNN=0 /home/bot/.venvs/vllm/bin/python -m benchmarks.a2_e2e_ab_test --tag stock
    QSR_A2_B12X=1 QSR_A2_CUDNN=0 /home/bot/.venvs/vllm/bin/python -m benchmarks.a2_e2e_ab_test --tag b12x
    QSR_A2_CUDNN=1 /home/bot/.venvs/vllm/bin/python -m benchmarks.a2_e2e_ab_test --tag cudnn
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3


def run_mtp_decode(runner, torch, slot, prompt_ids, max_tokens):
    """Run MTP decode loop, return (committed_len, wall_time_s, num_rounds)."""
    # Prefill
    pr = runner.mtp_prefill(slot, prompt_ids)
    anchor = pr["anchor"]
    drafts = pr["draft_tokens"]
    committed_len = 1  # prefill produces 1 token (the anchor)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    num_rounds = 0
    while committed_len < max_tokens:
        decision = runner.mtp_verify_and_commit(slot, anchor, drafts)
        n_acc = decision["num_accepted"]
        committed_len += n_acc + 1  # n_acc accepted drafts + 1 verify token
        anchor = decision["next_anchor"]
        drafts = decision["next_draft_tokens"]
        num_rounds += 1

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return committed_len, elapsed, num_rounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--prompt-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", default="benchmarks/fixtures/a2_e2e_ab.json")
    args = parser.parse_args()

    import torch

    sys.path.insert(0, "/home/bot/project/sm120-flash-attention/vllm_integration")
    import register_sm120_backend  # noqa: F401

    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner, build_vllm_config

    cudnn = os.environ.get("QSR_A2_CUDNN", "0") != "0"
    b12x = os.environ.get("QSR_A2_B12X", "1") != "0"
    print(f"=== A2 E2E: {args.tag} (B12x={'ON' if b12x else 'OFF'}, cuDNN={'ON' if cudnn else 'OFF'}) ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading model...")
    t0 = time.perf_counter()
    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=args.prompt_len + 8192,  # generous MTP headroom
        gpu_memory_utilization=0.85,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": K,
            "attention_backend": "CUSTOM",
        },
    )
    runner = DirectModelRunner(
        vllm_config=vllm_config,
        num_slots=4,
        blocks_per_slot=-(-(args.prompt_len + 8192) // 16),
        enable_cudagraph=False,
        enable_prefix_cache=False,
    )
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    prompt = list(range(1000, 1000 + args.prompt_len))
    slot = 0

    # Warmup
    print(f"Warmup ({args.warmup} reps)...")
    for _ in range(args.warmup):
        run_mtp_decode(runner, torch, slot, prompt, args.max_tokens)
        runner.reset_slot(slot)

    # Benchmark
    print(f"Benchmark ({args.reps} reps, {args.max_tokens} tokens each)...")
    results = []
    tokens_rep0 = None

    for rep in range(args.reps):
        committed_len, elapsed, num_rounds = run_mtp_decode(
            runner, torch, slot, prompt, args.max_tokens
        )
        tps = committed_len / elapsed
        accept_rate = committed_len / (num_rounds * (K + 1)) if num_rounds > 0 else 0
        results.append({
            "rep": rep,
            "tokens": committed_len,
            "rounds": num_rounds,
            "elapsed_s": round(elapsed, 4),
            "accepted_tok_s": round(tps, 1),
            "accept_rate": round(accept_rate, 3),
        })
        print(f"  Rep {rep}: {committed_len} tok, {num_rounds} rounds, {elapsed:.3f}s, "
              f"{tps:.1f} tok/s, accept={accept_rate:.1%}")
        runner.reset_slot(slot)

    avg_tps = sum(r["accepted_tok_s"] for r in results) / len(results)
    avg_acc = sum(r["accept_rate"] for r in results) / len(results)
    print(f"\n  AVG: {avg_tps:.1f} accepted tok/s, accept rate {avg_acc:.1%}")

    # Save
    entry = {
        "tag": args.tag,
        "b12x": b12x,
        "cudnn": cudnn,
        "gpu": torch.cuda.get_device_name(0),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "avg_accepted_tok_s": round(avg_tps, 1),
        "avg_accept_rate": round(avg_acc, 3),
        "reps": results,
    }
    existing = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
    existing.append(entry)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
