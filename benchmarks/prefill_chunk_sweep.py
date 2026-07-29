"""Sweep prefill chunk sizes: 4096, 8192, 16384 for 64K and 128K contexts."""
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
    chunk_sizes = [int(x) for x in os.environ.get("QSR_CHUNK_SIZES", "4096,8192,16384").split(",")]
    ctx_lengths = [int(x) for x in os.environ.get("QSR_CTX_LENGTHS", "65536,131072").split(",")]
    
    print("=" * 70)
    print(f"Prefill Chunk Sweep (MoE={moe_backend})")
    print(f"Chunk sizes: {chunk_sizes}")
    print(f"Context lengths: {ctx_lengths}")
    print("=" * 70)

    from runtime.legacy_qwen36_vllm import EngineArgs
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

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    results = []

    for chunk_size in chunk_sizes:
        os.environ["QSR_PREFILL_CHUNK"] = str(chunk_size)
        print(f"\n{'='*60}")
        print(f"  Chunk size: {chunk_size}")
        print(f"{'='*60}")
        
        # Need fresh backend for each chunk size (it reads env at init)
        torch.cuda.empty_cache()
        import gc; gc.collect()
        
        from runtime.backends.laguna import LagunaBackend
        backend = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=2112)
        
        from runtime.backends.laguna_dflash import DFlashEngine
        engine = DFlashEngine(backend)
        
        mem_after_init = gpu_mem_mb()
        print(f"  Mem after init: {mem_after_init:.0f} MB")
        
        for ctx_len in ctx_lengths:
            ctx_name = f"{ctx_len//1024}K"
            prompt = make_prompt(tokenizer, ctx_len)
            print(f"\n  --- {ctx_name} context ({len(prompt)} tokens) ---")
            
            # Warmup
            backend.reset_slot(0)
            torch.cuda.empty_cache()
            _, ws = engine.generate(prompt, max_tokens=32)
            print(f"  Warmup: prefill={ws['prefill_ms']:.0f}ms, mem={gpu_mem_mb():.0f}MB")
            
            # Measured (3 runs)
            prefill_times = []
            for run in range(3):
                backend.reset_slot(0)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _, stats = engine.generate(prompt, max_tokens=64)
                torch.cuda.synchronize()
                total = (time.perf_counter() - t0) * 1000
                prefill_times.append(stats['prefill_ms'])
                print(f"  Run {run+1}: prefill={stats['prefill_ms']:.0f}ms, "
                      f"decode={stats['tok_per_s']:.1f}tok/s, total={total:.0f}ms, "
                      f"mem={gpu_mem_mb():.0f}MB")
            
            avg_prefill = sum(prefill_times) / len(prefill_times)
            results.append({
                "chunk_size": chunk_size,
                "ctx_len": ctx_len,
                "ctx_name": ctx_name,
                "avg_prefill_ms": avg_prefill,
                "prefill_runs": prefill_times,
                "mem_mb": gpu_mem_mb(),
            })
            print(f"  Avg prefill: {avg_prefill:.0f}ms")
        
        # Cleanup for next chunk size
        del engine, backend
        torch.cuda.empty_cache()
        gc.collect()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Chunk':<8} {'Ctx':<6} {'Avg Prefill(ms)':<18} {'Mem(MB)':<10}")
    print("-" * 45)
    for r in results:
        print(f"{r['chunk_size']:<8} {r['ctx_name']:<6} {r['avg_prefill_ms']:<18.0f} {r['mem_mb']:<10.0f}")
    
    out = "benchmarks/fixtures/prefill_chunk_sweep.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
