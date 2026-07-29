"""A2 前置调查：NVFP4 GEMM shape 与占比分解。

用 torch.profiler 的 record_shapes=True 捕获每个 GEMM kernel 的
输入维度，按 shape 聚合统计，找出哪些 shape 占了多少时间。

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.a2_gemm_shape_survey
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_NATIVEFP8_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3


def main() -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile

    sys.path.insert(0, "/home/bot/project/sm120-flash-attention/vllm_integration")
    import register_sm120_backend  # noqa: F401

    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner, build_vllm_config

    prompt_len = 4096
    print(f"Loading model, prompt_len={prompt_len}...")
    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=prompt_len + 8192,
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
        blocks_per_slot=-(-(prompt_len + 8192) // 16),
        enable_cudagraph=False,
        enable_prefix_cache=False,
    )
    print("Model loaded.")

    prompt = list(range(1000, 1000 + prompt_len))
    runner.reset_slot(0)
    result = runner.mtp_prefill_with_cache([0], [prompt])
    anchor = result[0]["anchor"]
    drafts = result[0]["draft_tokens"]
    torch.cuda.synchronize()

    rounds = 5
    print(f"Profiling {rounds} MTP verify rounds with shapes...")

    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(rounds):
            decisions = runner.mtp_verify_and_commit_batch(
                [0], {0: anchor}, {0: drafts}
            )
            anchor = decisions[0]["next_anchor"]
            drafts = decisions[0]["next_draft_tokens"]
        torch.cuda.synchronize()

    print("\n" + "=" * 80)
    print("A2: GEMM SHAPE SURVEY (decode step, MTP verify)")
    print("=" * 80)

    gemm_keywords = ["cutlass", "gemm", "gemvx", "cublas", "sgemm", "hgemm",
                     "nvfp4", "fp4", "matmul", "mm_", "linear", "nvjet"]

    shape_times: dict[str, float] = defaultdict(float)
    total_gpu_us = 0.0

    for evt in prof.key_averages(group_by_input_shape=True):
        if evt.device_time_total > 0:
            total_gpu_us += evt.device_time_total
            name_lower = evt.key.lower()
            is_gemm = any(kw in name_lower for kw in gemm_keywords)
            if is_gemm:
                shape_str = str(evt.input_shapes) if evt.input_shapes else "unknown"
                key = f"{evt.key[:50]} | shapes={shape_str}"
                shape_times[key] += evt.device_time_total

    print(f"\nTotal GPU time: {total_gpu_us/1000:.1f} ms ({rounds} rounds)")
    print(f"\nGEMM kernels by shape ({len(shape_times)} unique):")
    print(f"{'Kernel + Shape':<75} {'ms':>8} {'%':>6}")
    print("-" * 91)
    for key, us in sorted(shape_times.items(), key=lambda x: -x[1])[:25]:
        ms = us / 1000
        pct = us / total_gpu_us * 100
        short = key[:73] if len(key) > 73 else key
        print(f"{short:<75} {ms:>8.3f} {pct:>5.1f}%")

    gemm_total = sum(shape_times.values())
    print(f"\n{'GEMM TOTAL':<75} {gemm_total/1000:>8.2f} {gemm_total/total_gpu_us*100:>5.1f}%")


if __name__ == "__main__":
    main()
