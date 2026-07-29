#!/usr/bin/env python3
"""Laguna-S-2.1-NVFP4: Run with custom runtime patches + basic perf/quality tests.

Tests:
1. Load model with CutlassDirect NVFP4 patch (A2 optimization)
2. Greedy determinism (same prompt → same output, twice)
3. Basic throughput measurement (prefill + decode)
4. Output quality sanity check (coherent text)

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.laguna_runtime_test
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL = "poolside/Laguna-S-2.1-NVFP4"


def main():
    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL}")
    print()

    # ── Step 1: Apply CutlassDirect NVFP4 patch ──
    print("=== Step 1: Apply CutlassDirect NVFP4 patch ===")
    from oracle.qwen36_vllm.nvfp4_cutlass_direct_patch import patch_nvfp4_prefer_cutlass_direct

    patched = patch_nvfp4_prefer_cutlass_direct()
    print(f"  CutlassDirect patch applied: {patched}")

    # ── Step 2: Load model via vLLM ──
    print("\n=== Step 2: Load model via vLLM ===")
    from vllm import LLM, SamplingParams

    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        dtype="bfloat16",
        disable_log_stats=True,
    )
    load_time = time.perf_counter() - t0
    gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Loaded in {load_time:.1f}s, GPU mem: {gpu_mem:.0f} MiB")

    # ── Step 3: Greedy determinism test ──
    print("\n=== Step 3: Greedy determinism test ===")
    prompt = "The capital of France is"
    greedy = SamplingParams(temperature=0, max_tokens=50)

    out1 = llm.generate([prompt], greedy)[0].outputs[0].text
    out2 = llm.generate([prompt], greedy)[0].outputs[0].text
    determinism_pass = out1 == out2
    print(f"  Run 1: {out1!r}")
    print(f"  Run 2: {out2!r}")
    print(f"  Determinism: {'✅ PASS' if determinism_pass else '❌ FAIL'}")

    # ── Step 4: Throughput benchmark ──
    print("\n=== Step 4: Throughput benchmark ===")
    # Short prompt for decode-heavy test
    short_prompt = "Write a detailed explanation of quantum computing:"
    bench_params = SamplingParams(temperature=0, max_tokens=128)

    # Warmup
    llm.generate([short_prompt], bench_params)

    # Benchmark
    reps = 5
    times = []
    token_counts = []
    for i in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = llm.generate([short_prompt], bench_params)[0]
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_tokens = len(result.outputs[0].token_ids)
        times.append(elapsed)
        token_counts.append(n_tokens)
        tps = n_tokens / elapsed
        print(f"  Rep {i}: {n_tokens} tokens, {elapsed:.3f}s, {tps:.1f} tok/s")

    avg_tps = sum(t / e for t, e in zip(token_counts, times)) / reps
    avg_itl = sum(times) / sum(token_counts) * 1000
    print(f"\n  AVG: {avg_tps:.1f} tok/s, ITL: {avg_itl:.1f} ms")

    # ── Step 5: Output quality sanity ──
    print("\n=== Step 5: Output quality sanity ===")
    quality_prompts = [
        ("Math", "What is 15 * 37? Show your work step by step."),
        ("Code", "Write a Python function to check if a number is prime."),
        ("Knowledge", "Explain the theory of relativity in simple terms."),
    ]
    quality_params = SamplingParams(temperature=0, max_tokens=200)
    quality_results = []
    for name, qp in quality_prompts:
        result = llm.generate([qp], quality_params)[0]
        text = result.outputs[0].text
        n_tok = len(result.outputs[0].token_ids)
        # Basic sanity: output should be non-trivial
        passed = len(text) > 20 and n_tok > 10
        quality_results.append({"name": name, "passed": passed, "n_tokens": n_tok})
        print(f"  {name}: {n_tok} tokens, {'✅' if passed else '❌'}")
        print(f"    Preview: {text[:120]}...")

    # ── Step 6: Concurrent requests ──
    print("\n=== Step 6: Concurrent requests (batch=4) ===")
    batch_prompts = [
        "The meaning of life is",
        "In machine learning, gradient descent is",
        "The best programming language for beginners is",
        "Climate change is primarily caused by",
    ]
    batch_params = SamplingParams(temperature=0, max_tokens=50)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    batch_results = llm.generate(batch_prompts, batch_params)
    torch.cuda.synchronize()
    batch_elapsed = time.perf_counter() - t0
    batch_tokens = sum(len(r.outputs[0].token_ids) for r in batch_results)
    batch_tps = batch_tokens / batch_elapsed
    concurrent_pass = all(len(r.outputs[0].text) > 10 for r in batch_results)
    print(f"  {batch_tokens} tokens in {batch_elapsed:.3f}s = {batch_tps:.1f} tok/s (batch)")
    print(f"  All outputs non-trivial: {'✅' if concurrent_pass else '❌'}")

    # ── Summary ──
    all_pass = determinism_pass and concurrent_pass and all(q["passed"] for q in quality_results)
    print(f"\n{'='*60}")
    print(f"OVERALL: {'✅ ALL PASS' if all_pass else '❌ SOME FAILED'}")
    print(f"  Greedy determinism: {'✅' if determinism_pass else '❌'}")
    print(f"  Throughput: {avg_tps:.1f} tok/s (single), {batch_tps:.1f} tok/s (batch=4)")
    print(f"  ITL: {avg_itl:.1f} ms")
    print(f"  Quality: {sum(q['passed'] for q in quality_results)}/{len(quality_results)}")
    print(f"  Concurrent: {'✅' if concurrent_pass else '❌'}")
    print(f"  CutlassDirect NVFP4: {'✅' if patched else '❌'}")
    print(f"{'='*60}")

    # Save results
    results = {
        "model": MODEL,
        "date": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "cutlass_direct": patched,
        "load_time_s": round(load_time, 1),
        "gpu_mem_mib": round(gpu_mem),
        "greedy_determinism": determinism_pass,
        "greedy_output": out1,
        "throughput": {
            "single_tps": round(avg_tps, 1),
            "batch4_tps": round(batch_tps, 1),
            "itl_ms": round(avg_itl, 1),
            "reps": reps,
        },
        "quality": quality_results,
        "concurrent": concurrent_pass,
        "all_pass": all_pass,
    }
    out_path = Path("benchmarks/fixtures/laguna_runtime_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
