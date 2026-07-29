"""Comprehensive benchmark: all metrics in one model load.

Usage: /home/bot/.venvs/vllm/bin/python benchmarks/comprehensive_bench.py
"""
import os, sys, time, json
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")
os.environ["QSR_DFLASH_CUDA_GRAPH"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
import torch

def make_prompt(tokenizer, n):
    base = "The quick brown fox jumps over the lazy dog. "
    tokens = []
    chunk = tokenizer.encode(base, add_special_tokens=False)
    while len(tokens) < n:
        tokens.extend(chunk)
    return tokens[:n]

def gpu_mem_mb():
    return torch.cuda.memory_allocated() / 1024**2

def main():
    moe_backend = os.environ.get("QSR_MOE_BACKEND", "marlin")
    print("=" * 70)
    print(f"Comprehensive Benchmark (MoE={moe_backend})")
    print("=" * 70)

    from oracle.qwen36_vllm.vllm_compat import EngineArgs
    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
        "snapshots/07614121b31898586430f189d27a25a0be310843/"
    )
    engine_args = EngineArgs(
        model=model_path, dtype="bfloat16", max_model_len=131072,
        gpu_memory_utilization=0.88, enforce_eager=True, trust_remote_code=True,
        moe_backend=moe_backend,
    )
    vllm_config = engine_args.create_engine_config()

    print(f"\n[1] Loading model... (mem before: {gpu_mem_mb():.0f} MB)")
    t0 = time.perf_counter()
    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=2112)
    t_load = time.perf_counter() - t0
    print(f"  Loaded in {t_load:.1f}s, mem after: {gpu_mem_mb():.0f} MB")

    print("\n[2] Initializing DFlash...")
    from runtime.backends.laguna_dflash import DFlashEngine
    engine = DFlashEngine(backend)
    print(f"  DFlash ready, mem: {gpu_mem_mb():.0f} MB")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    results = {"moe_backend": moe_backend, "load_time_s": t_load}

    # Test each context length
    for ctx_k in [64, 128]:
        ctx_len = ctx_k * 1024
        ctx_name = f"{ctx_k}K"
        print(f"\n{'='*50}")
        print(f"  {ctx_name} context")
        print(f"{'='*50}")

        prompt = make_prompt(tokenizer, ctx_len)
        print(f"  Prompt: {len(prompt)} tokens, mem before: {gpu_mem_mb():.0f} MB")

        # Warmup (CG capture)
        print("  Warmup...")
        t_w0 = time.perf_counter()
        _, ws = engine.generate(prompt, max_tokens=128)
        t_warm = time.perf_counter() - t_w0
        print(f"  Warmup: {ws['tok_per_s']:.1f} tok/s, accept={ws['acceptance_rate']:.1%}, "
              f"prefill={ws['prefill_ms']:.0f}ms, total={t_warm:.1f}s")
        print(f"  CG: verify={engine._verify_cg is not None}, draft={engine._draft_cg is not None}")
        print(f"  Mem after warmup: {gpu_mem_mb():.0f} MB")

        # Measured
        print("  Measured (256 tokens)...")
        tokens, stats = engine.generate(prompt, max_tokens=256)
        itl = stats['decode_ms'] / max(stats['num_tokens'] - 1, 1)
        print(f"  Result: {stats['tok_per_s']:.1f} tok/s, accept={stats['acceptance_rate']:.1%}, "
              f"tok/step={stats['tokens_per_step']:.2f}")
        print(f"  Prefill: {stats['prefill_ms']:.0f}ms, Decode: {stats['decode_ms']:.0f}ms, "
              f"ITL: {itl:.2f}ms, Steps: {stats['num_steps']}")
        print(f"  Mem: {gpu_mem_mb():.0f} MB")

        # Baseline
        print("  Baseline (eager, 128 tokens)...")
        slot = 0
        backend.reset_slot(slot)
        torch.cuda.empty_cache()
        first = backend.prefill(slot, prompt)
        t_pf = time.perf_counter()
        toks = [first]
        for _ in range(127):
            tok = backend.decode(slot, toks[-1])
            toks.append(tok)
            if tok in (2, 24): break
        t_end = time.perf_counter()
        dec_ms = (t_end - t_pf) * 1000
        n = len(toks) - 1
        base_tps = n / (dec_ms / 1000)
        base_itl = dec_ms / n
        print(f"  Baseline: {base_tps:.1f} tok/s, ITL {base_itl:.2f}ms")
        backend.reset_slot(slot)

        speedup = stats['tok_per_s'] / max(base_tps, 1e-6)
        print(f"  Speedup: {speedup:.2f}×")

        results[ctx_name] = {
            "dflash_tps": stats['tok_per_s'],
            "dflash_itl_ms": itl,
            "acceptance": stats['acceptance_rate'],
            "tok_per_step": stats['tokens_per_step'],
            "prefill_ms": stats['prefill_ms'],
            "baseline_tps": base_tps,
            "baseline_itl_ms": base_itl,
            "speedup": speedup,
            "gpu_mem_mb": gpu_mem_mb(),
        }

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"MoE backend: {moe_backend}")
    print(f"{'Ctx':<6} {'DFlash':<10} {'Baseline':<10} {'ITL(ms)':<10} {'Accept%':<10} {'Tok/Step':<10} {'Speedup':<8} {'Mem(MB)':<10}")
    print("-" * 75)
    for ctx_name in ["64K", "128K"]:
        if ctx_name in results:
            r = results[ctx_name]
            print(f"{ctx_name:<6} {r['dflash_tps']:<10.1f} {r['baseline_tps']:<10.1f} "
                  f"{r['dflash_itl_ms']:<10.2f} {r['acceptance']:<10.1%} "
                  f"{r['tok_per_step']:<10.2f} {r['speedup']:<8.2f}× {r['gpu_mem_mb']:<10.0f}")

    out = "benchmarks/fixtures/comprehensive_bench.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
