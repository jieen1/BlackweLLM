"""A1a: GDN per-layer profiling — decode step time occupancy ledger.

Produces a kernel-level breakdown of one MTP verify round, categorizing
GPU time into: attention, GDN (linear attention), GEMM, MTP draft, other.

Usage:
    python -m benchmarks.a1a_gdn_profile [--prompt-len 4096] [--rounds 10]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_NATIVEFP8_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3

GDN_KEYWORDS = [
    "gdn", "delta_rule", "gated_delta", "fla", "chunk_gated",
    "fused_sigmoid_gating", "causal_conv1d", "rms_norm",
    "rmsnorm", "gemma_rms",
]
ATTN_KEYWORDS = [
    "flash_attn", "sm120", "flash_fwd", "decode_v2", "attention",
    "paged_kv", "splitkv",
]
GEMM_KEYWORDS = [
    "gemm", "cutlass", "nvfp4", "fp4", "matmul", "mm_",
    "linear", "cublas", "sgemm", "hgemm",
]
MTP_KEYWORDS = [
    "mtp", "draft", "eagle", "spec",
]


def _categorize_kernel(name: str) -> str:
    name_lower = name.lower()
    for kw in ATTN_KEYWORDS:
        if kw in name_lower:
            return "attention"
    for kw in GDN_KEYWORDS:
        if kw in name_lower:
            return "gdn"
    for kw in GEMM_KEYWORDS:
        if kw in name_lower:
            return "gemm"
    for kw in MTP_KEYWORDS:
        if kw in name_lower:
            return "mtp_draft"
    return "other"


def main() -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile

    parser = argparse.ArgumentParser(description="A1a GDN profiling")
    parser.add_argument("--prompt-len", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    sys.path.insert(0, "/home/bot/project/sm120-flash-attention/vllm_integration")
    import register_sm120_backend  # noqa: F401

    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner, build_vllm_config

    print(f"Loading model ({MODEL})...")
    t0 = time.perf_counter()
    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=args.prompt_len + 8192,
        gpu_memory_utilization=0.85,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": K,
            "attention_backend": "CUSTOM",
        },
    )
    runner = DirectModelRunner(
        vllm_config=vllm_config,
        num_slots=max(args.concurrency * 2, 4),
        blocks_per_slot=-(-(args.prompt_len + 8192) // 16),
        enable_cudagraph=False,
        enable_prefix_cache=False,
    )
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    prompt = list(range(1000, 1000 + args.prompt_len))
    slots = list(range(args.concurrency))

    print(f"Prefilling {args.concurrency} slot(s), prompt_len={args.prompt_len}...")
    for s in slots:
        runner.reset_slot(s)
    result = runner.mtp_prefill_with_cache(slots, [prompt] * args.concurrency)
    torch.cuda.synchronize()
    print("Prefill done.")

    anchors = {s: result[s]["anchor"] for s in slots}
    drafts = {s: result[s]["draft_tokens"] for s in slots}

    print(f"Running {args.rounds} MTP verify rounds with torch.profiler...")
    category_times: dict[str, float] = {}
    kernel_details: dict[str, float] = {}
    total_gpu_us = 0.0

    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(args.rounds):
            decisions = runner.mtp_verify_and_commit_batch(slots, anchors, drafts)
            for s in slots:
                anchors[s] = decisions[s]["next_anchor"]
                drafts[s] = decisions[s]["next_draft_tokens"]
        torch.cuda.synchronize()

    print("\n" + "=" * 70)
    print("A1a: DECODE STEP TIME OCCUPANCY LEDGER")
    print(f"    prompt_len={args.prompt_len}, concurrency={args.concurrency}, "
          f"rounds={args.rounds}, K={K}")
    print("=" * 70)

    events = prof.key_averages()
    for evt in events:
        if evt.device_time_total > 0:
            cat = _categorize_kernel(evt.key)
            us = evt.device_time_total
            category_times[cat] = category_times.get(cat, 0) + us
            kernel_details[evt.key] = kernel_details.get(evt.key, 0) + us
            total_gpu_us += us

    print(f"\n{'Category':<15} {'GPU time (ms)':>14} {'% of total':>12}")
    print("-" * 43)
    for cat in sorted(category_times, key=category_times.get, reverse=True):
        ms = category_times[cat] / 1000
        pct = category_times[cat] / total_gpu_us * 100 if total_gpu_us > 0 else 0
        print(f"{cat:<15} {ms:>14.2f} {pct:>11.1f}%")
    print("-" * 43)
    print(f"{'TOTAL':<15} {total_gpu_us/1000:>14.2f} {'100.0%':>12}")

    per_round_ms = total_gpu_us / 1000 / args.rounds
    print(f"\nPer-round GPU time: {per_round_ms:.2f} ms")
    print(f"Per-round per-slot: {per_round_ms / args.concurrency:.2f} ms")

    print("\nTop 20 kernels by GPU time:")
    print(f"{'Kernel':<60} {'ms':>10} {'%':>7} {'Category':<12}")
    print("-" * 91)
    sorted_kernels = sorted(kernel_details.items(), key=lambda x: -x[1])[:20]
    for name, us in sorted_kernels:
        ms = us / 1000
        pct = us / total_gpu_us * 100
        cat = _categorize_kernel(name)
        short_name = name[:58] if len(name) > 58 else name
        print(f"{short_name:<60} {ms:>10.3f} {pct:>6.1f}% {cat:<12}")

    print(f"\nGPU memory: {torch.cuda.memory_allocated()/2**30:.1f} GiB allocated, "
          f"{torch.cuda.memory_reserved()/2**30:.1f} GiB reserved")


if __name__ == "__main__":
    main()
