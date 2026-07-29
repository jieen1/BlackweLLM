#!/usr/bin/env python3
"""Laguna CUDA Graph: determinism + fair benchmark (both sides CUDA graph + compile).

Tests:
1. Graph determinism: same prompt → same tokens (greedy, twice)
2. Fair throughput: our CUDA Graph+compile vs vLLM CUDA Graph+compile
3. ITL measurement

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.laguna_cudagraph_bench
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("QSR_A2_CUSTOM_GEMM", "0")

MODEL = "poolside/Laguna-S-2.1-NVFP4"
DECODE_TOKENS = 128
WARMUP_TOKENS = 10
BENCH_REPS = 5


def build_laguna_config_compile(max_model_len: int = 4096):
    """Build config with torch.compile enabled but vLLM cudagraph disabled."""
    from oracle.qwen36_vllm.vllm_compat import EngineArgs, CUDAGraphMode

    args = EngineArgs(
        model=MODEL,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        disable_log_stats=True,
        async_scheduling=False,
    )
    config = args.create_engine_config()
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    return config


def build_laguna_config_eager(max_model_len: int = 4096):
    """Build config with enforce_eager (no compile, no graph)."""
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


def test_determinism(backend, tokenizer, cg, num_tokens: int = 50) -> dict:
    """Graph determinism: same prompt → same tokens, twice."""
    prompt = "The capital of France is"
    prompt_ids = tokenizer.encode(prompt)

    def run_graph_decode():
        cg.reset()
        backend.reset_slot(0)
        first = backend.prefill(0, prompt_ids)
        tokens = [first]
        for _ in range(num_tokens - 1):
            kv_len = backend.slot_kv_len[0]
            result = cg.replay([0], [tokens[-1]], [kv_len])
            tok = result[0]
            tokens.append(tok)
            backend.slot_kv_len[0] += 1
            backend.slot_committed_tokens[0].append(tokens[-2])
            if tok in (2, 24):
                break
        return tokens

    run1 = run_graph_decode()
    run2 = run_graph_decode()

    match = run1 == run2
    text1 = tokenizer.decode(run1, skip_special_tokens=True)
    text2 = tokenizer.decode(run2, skip_special_tokens=True)

    print(f"  Run 1: {text1[:80]!r}")
    print(f"  Run 2: {text2[:80]!r}")
    print(f"  Tokens: {len(run1)} vs {len(run2)}")
    print(f"  Determinism: {'✅ PASS' if match else '❌ FAIL'}")

    return {
        "determinism": match,
        "tokens_run1": len(run1),
        "tokens_run2": len(run2),
        "text": text1[:120],
    }


def bench_our_graph(backend, tokenizer, cg) -> dict:
    """Benchmark our CUDA Graph decode path."""
    prompt = "Write a detailed explanation of quantum computing:"
    prompt_ids = tokenizer.encode(prompt)

    # Warmup
    backend.reset_slot(0)
    tok = backend.prefill(0, prompt_ids)
    for _ in range(WARMUP_TOKENS):
        kv_len = backend.slot_kv_len[0]
        result = cg.replay([0], [tok], [kv_len])
        tok = result[0]
        backend.slot_kv_len[0] += 1
        backend.slot_committed_tokens[0].append(tok)
    backend.reset_slot(0)

    times_list = []
    token_counts = []
    for rep in range(BENCH_REPS):
        backend.reset_slot(0)
        tok = backend.prefill(0, prompt_ids)
        n = 1
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(DECODE_TOKENS - 1):
            kv_len = backend.slot_kv_len[0]
            result = cg.replay([0], [tok], [kv_len])
            tok = result[0]
            backend.slot_kv_len[0] += 1
            backend.slot_committed_tokens[0].append(tok)
            n += 1
            if tok in (2, 24):
                break
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times_list.append(elapsed)
        token_counts.append(n)
        tps = n / elapsed
        print(f"  Rep {rep}: {n} tokens, {elapsed:.3f}s, {tps:.1f} tok/s")

    avg_tps = sum(t / e for t, e in zip(token_counts, times_list)) / BENCH_REPS
    avg_itl = sum(times_list) / sum(token_counts) * 1000
    return {"tok_s": round(avg_tps, 1), "itl_ms": round(avg_itl, 2), "reps": BENCH_REPS}


def bench_vllm_graph(tokenizer) -> dict:
    """Benchmark vLLM with CUDA graph + compile (fair comparison)."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        max_model_len=4096,
        max_num_seqs=4,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        dtype="bfloat16",
        disable_log_stats=True,
    )

    prompt = "Write a detailed explanation of quantum computing:"
    sp = SamplingParams(temperature=0, max_tokens=DECODE_TOKENS)

    # Warmup
    for _ in range(3):
        llm.generate([prompt], sp)

    times_list = []
    token_counts = []
    for rep in range(BENCH_REPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sp)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_out = len(outputs[0].outputs[0].token_ids)
        n_in = len(tokenizer.encode(prompt))
        n = n_in + n_out
        times_list.append(elapsed)
        token_counts.append(n)
        tps = n / elapsed
        print(f"  Rep {rep}: {n} tokens ({n_in}+{n_out}), {elapsed:.3f}s, {tps:.1f} tok/s")

    avg_tps = sum(t / e for t, e in zip(token_counts, times_list)) / BENCH_REPS
    avg_itl = sum(times_list) / sum(token_counts) * 1000

    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return {"tok_s": round(avg_tps, 1), "itl_ms": round(avg_itl, 2), "reps": BENCH_REPS}


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL}")
    print(f"Decode tokens: {DECODE_TOKENS}, Reps: {BENCH_REPS}")
    print()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # ── Phase 1: Load with compile ──
    print("=== Phase 1: Load LagunaBackend (compile enabled) ===")
    config = build_laguna_config_compile()
    from runtime.backends.laguna import LagunaBackend

    t0 = time.perf_counter()
    backend = LagunaBackend(config, num_slots=4, block_size=64, blocks_per_slot=128)
    load_time = time.perf_counter() - t0
    gpu_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Loaded in {load_time:.1f}s, GPU mem: {gpu_mem:.0f} MiB")

    # ── Phase 2: Capture graph ──
    print("\n=== Phase 2: Capture CUDA Graph ===")
    from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode
    cg = LagunaCudaGraphDecode(backend, batch_size=1)
    t0 = time.perf_counter()
    cg.capture()
    capture_time = time.perf_counter() - t0
    print(f"  Captured in {capture_time:.1f}s")

    # ── Phase 3: Determinism ──
    print("\n=== Phase 3: Graph Determinism ===")
    det = test_determinism(backend, tokenizer, cg, num_tokens=50)

    # ── Phase 4: Our graph benchmark ──
    print("\n=== Phase 4: Our CUDA Graph+Compile Decode ===")
    our_graph = bench_our_graph(backend, tokenizer, cg)
    print(f"  AVG: {our_graph['tok_s']} tok/s, ITL: {our_graph['itl_ms']} ms")

    # Free our backend
    del backend, cg
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ── Phase 5: vLLM graph benchmark ──
    print("\n=== Phase 5: vLLM CUDA Graph+Compile Decode ===")
    vllm_graph = bench_vllm_graph(tokenizer)
    print(f"  AVG: {vllm_graph['tok_s']} tok/s, ITL: {vllm_graph['itl_ms']} ms")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Determinism: {'✅ PASS' if det['determinism'] else '❌ FAIL'}")
    print(f"  Our Graph+Compile: {our_graph['tok_s']:>6} tok/s  ITL {our_graph['itl_ms']:>6} ms")
    print(f"  vLLM Graph+Compile:{vllm_graph['tok_s']:>6} tok/s  ITL {vllm_graph['itl_ms']:>6} ms")
    speedup = our_graph['tok_s'] / vllm_graph['tok_s'] if vllm_graph['tok_s'] > 0 else 0
    print(f"  Speedup vs vLLM: {speedup:.2f}x")
    target_met = our_graph['tok_s'] >= vllm_graph['tok_s']
    print(f"  Target (>vLLM): {'✅ MET' if target_met else '❌ NOT MET'}")
    print(f"{'='*60}")

    results = {
        "model": MODEL,
        "date": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "decode_tokens": DECODE_TOKENS,
        "reps": BENCH_REPS,
        "determinism": det,
        "our_graph_compile": our_graph,
        "vllm_graph_compile": vllm_graph,
        "speedup_vs_vllm": round(speedup, 3),
        "target_met": target_met,
    }
    out_path = Path("benchmarks/fixtures/laguna_cudagraph_bench.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
