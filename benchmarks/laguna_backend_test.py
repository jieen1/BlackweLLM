#!/usr/bin/env python3
"""Laguna-S-2.1 direct-backend smoke and performance checks.

Tests:
1. Load model via LagunaBackend (direct forward, no vLLM LLM engine)
2. Greedy determinism (same prompt → same output, twice)
3. Throughput measurement (prefill + decode)
4. Batch decode test

Usage:
    # Direct LagunaBackend fact smoke only; does not start vLLM's LLM engine.
    /home/bot/.venvs/vllm/bin/python -m benchmarks.laguna_backend_test --fact-only

    # Full direct-backend smoke and performance suite.
    /home/bot/.venvs/vllm/bin/python -m benchmarks.laguna_backend_test
"""

from __future__ import annotations

import argparse
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
FACT_PROMPT = "The capital of France is"
EXPECTED_FACT_FIRST_TOKEN = "Paris"


def build_laguna_config(max_model_len: int = 4096):
    """Build VllmConfig for Laguna (FlashInfer backend, no SM120GQA)."""
    from oracle.qwen36_vllm.vllm_compat import EngineArgs

    args = EngineArgs(
        model=MODEL,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        dtype="bfloat16",
        disable_log_stats=True,
        async_scheduling=False,
    )
    return args.create_engine_config()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fact-only",
        action="store_true",
        help="run only the direct LagunaBackend Paris fact smoke check",
    )
    args = parser.parse_args()

    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL}")
    print()

    # ── Step 1: Load via LagunaBackend ──
    print("=== Step 1: Load via LagunaBackend ===")
    config = build_laguna_config()
    from runtime.backends.laguna import LagunaBackend

    t0 = time.perf_counter()
    # SparkInfer paged attention requires 64-token pages.  The fact-only
    # check needs no long-context capacity, so keep its cache allocation small.
    blocks_per_slot = 128 if args.fact_only else 512
    backend = LagunaBackend(
        config,
        num_slots=4,
        block_size=64,
        blocks_per_slot=blocks_per_slot,
    )
    load_time = time.perf_counter() - t0
    gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Loaded in {load_time:.1f}s, GPU mem: {gpu_mem:.0f} MiB")
    print(f"  Attn layers: {len(backend.attn_layer_names)}")
    print(f"  Window groups: { {str(k): len(v) for k, v in backend._layer_groups.items()} }")

    # ── Step 2: Greedy prefill + decode ──
    print("\n=== Step 2: Greedy prefill + decode ===")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt = FACT_PROMPT
    prompt_ids = tokenizer.encode(prompt)
    print(f"  Prompt: {prompt!r} -> {len(prompt_ids)} tokens")

    # Run 1
    backend.reset_slot(0)
    first_token = backend.prefill(0, prompt_ids)
    tokens_r1 = [first_token]
    for _ in range(49):
        tok = backend.decode(0, tokens_r1[-1])
        tokens_r1.append(tok)
        if tok in (2, 24):  # Laguna EOS
            break
    text_r1 = tokenizer.decode(tokens_r1, skip_special_tokens=True)
    print(f"  Run 1: {text_r1!r}")

    # Run 2 (determinism check)
    backend.reset_slot(0)
    first_token2 = backend.prefill(0, prompt_ids)
    tokens_r2 = [first_token2]
    for _ in range(49):
        tok = backend.decode(0, tokens_r2[-1])
        tokens_r2.append(tok)
        if tok in (2, 24):  # Laguna EOS
            break
    text_r2 = tokenizer.decode(tokens_r2, skip_special_tokens=True)
    print(f"  Run 2: {text_r2!r}")

    determinism_pass = tokens_r1 == tokens_r2
    first_token_text = tokenizer.decode([first_token], skip_special_tokens=True).strip()
    fact_pass = first_token_text == EXPECTED_FACT_FIRST_TOKEN
    print(f"  Determinism: {'✅ PASS' if determinism_pass else '❌ FAIL'}")
    print(
        "  Fact first token: "
        f"{first_token_text!r} "
        f"({'✅ PASS' if fact_pass else f'❌ expected {EXPECTED_FACT_FIRST_TOKEN!r}'})"
    )

    if args.fact_only:
        all_pass = determinism_pass and fact_pass
        print(f"\nFACT SMOKE: {'✅ PASS' if all_pass else '❌ FAIL'}")
        raise SystemExit(0 if all_pass else 1)

    # ── Step 3: Throughput benchmark ──
    print("\n=== Step 3: Throughput benchmark ===")
    bench_prompt = "Write a detailed explanation of quantum computing:"
    bench_ids = tokenizer.encode(bench_prompt)

    # Warmup
    backend.reset_slot(0)
    tok = backend.prefill(0, bench_ids)
    for _ in range(10):
        tok = backend.decode(0, tok)
    backend.reset_slot(0)

    reps = 5
    times_list = []
    token_counts = []
    for i in range(reps):
        backend.reset_slot(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tok = backend.prefill(0, bench_ids)
        n = 1
        for _ in range(127):
            tok = backend.decode(0, tok)
            n += 1
            if tok in (2, 24):  # Laguna EOS
                break
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times_list.append(elapsed)
        token_counts.append(n)
        tps = n / elapsed
        print(f"  Rep {i}: {n} tokens, {elapsed:.3f}s, {tps:.1f} tok/s")

    avg_tps = sum(t / e for t, e in zip(token_counts, times_list)) / reps
    avg_itl = sum(times_list) / sum(token_counts) * 1000
    print(f"\n  AVG: {avg_tps:.1f} tok/s, ITL: {avg_itl:.1f} ms")

    # ── Step 4: Batch decode test ──
    print("\n=== Step 4: Batch decode (4 slots) ===")
    batch_prompts = [
        "The meaning of life is",
        "In machine learning, gradient descent is",
        "The best programming language for beginners is",
        "Climate change is primarily caused by",
    ]
    # Prefill all 4 slots
    first_tokens = []
    for i, bp in enumerate(batch_prompts):
        backend.reset_slot(i)
        ids = tokenizer.encode(bp)
        ft = backend.prefill(i, ids)
        first_tokens.append(ft)

    # Batch decode
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    current_tokens = first_tokens[:]
    batch_token_count = 4  # count the first tokens
    for step in range(49):
        next_tokens = backend.decode_batch([0, 1, 2, 3], current_tokens)
        current_tokens = next_tokens
        batch_token_count += 4
    torch.cuda.synchronize()
    batch_elapsed = time.perf_counter() - t0
    batch_tps = batch_token_count / batch_elapsed
    print(f"  {batch_token_count} tokens in {batch_elapsed:.3f}s = {batch_tps:.1f} tok/s (batch=4)")

    # Check outputs are non-trivial
    for i in range(4):
        text = tokenizer.decode(
            backend.slot_committed_tokens[i][len(tokenizer.encode(batch_prompts[i])) :],
            skip_special_tokens=True,
        )
        print(f"  Slot {i}: {text[:80]}...")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    all_pass = determinism_pass and fact_pass
    print(f"OVERALL: {'✅ ALL PASS' if all_pass else '❌ SOME FAILED'}")
    print(f"  Greedy determinism: {'✅' if determinism_pass else '❌'}")
    print(f"  Paris first token: {'✅' if fact_pass else '❌'}")
    print(f"  Throughput: {avg_tps:.1f} tok/s (single), {batch_tps:.1f} tok/s (batch=4)")
    print(f"  ITL: {avg_itl:.1f} ms")
    print(f"{'=' * 60}")

    # Save results
    results = {
        "model": MODEL,
        "backend": "LagunaBackend (direct forward)",
        "date": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "load_time_s": round(load_time, 1),
        "gpu_mem_mib": round(gpu_mem),
        "greedy_determinism": determinism_pass,
        "greedy_output": text_r1,
        "fact_first_token": first_token_text,
        "fact_pass": fact_pass,
        "throughput": {
            "single_tps": round(avg_tps, 1),
            "batch4_tps": round(batch_tps, 1),
            "itl_ms": round(avg_itl, 1),
            "reps": reps,
        },
        "all_pass": all_pass,
    }
    out_path = Path("benchmarks/fixtures/laguna_backend_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
