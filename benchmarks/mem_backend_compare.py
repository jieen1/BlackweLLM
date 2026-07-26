"""Compare GPU memory: MARLIN vs CUTLASS MoE backends."""
import os, sys, time, json
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
import torch

def gpu_mem_mb():
    return torch.cuda.memory_allocated() / 1024**2

def test_backend(moe_backend):
    print(f"\n{'='*60}")
    print(f"  Testing MoE backend: {moe_backend}")
    print(f"{'='*60}")
    
    # Clean GPU
    torch.cuda.empty_cache()
    import gc; gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    from runtime.compat_vllm import EngineArgs
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
    
    t0 = time.perf_counter()
    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=2112)
    t_load = time.perf_counter() - t0
    
    mem_after_load = gpu_mem_mb()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    # Count parameters by type
    total_params = 0
    moe_params = 0
    attn_params = 0
    other_params = 0
    for name, p in backend.model.named_parameters():
        nbytes = p.nelement() * p.element_size()
        total_params += nbytes
        if 'experts' in name or 'gate' in name:
            moe_params += nbytes
        elif 'attn' in name or 'qkv' in name or 'o_proj' in name:
            attn_params += nbytes
        else:
            other_params += nbytes
    
    print(f"  Load time: {t_load:.1f}s")
    print(f"  GPU mem after load: {mem_after_load:.0f} MB ({mem_after_load/1024:.1f} GB)")
    print(f"  Peak GPU mem: {peak_mem:.0f} MB ({peak_mem/1024:.1f} GB)")
    print(f"  Model params total: {total_params/1024**3:.2f} GB")
    print(f"    MoE params: {moe_params/1024**3:.2f} GB")
    print(f"    Attn params: {attn_params/1024**3:.2f} GB")
    print(f"    Other params: {other_params/1024**3:.2f} GB")
    
    # Check weight dtypes
    dtype_counts = {}
    for name, p in backend.model.named_parameters():
        dt = str(p.dtype)
        if dt not in dtype_counts:
            dtype_counts[dt] = 0
        dtype_counts[dt] += p.nelement() * p.element_size()
    print("  Weight dtypes:")
    for dt, sz in sorted(dtype_counts.items(), key=lambda x: -x[1]):
        print(f"    {dt}: {sz/1024**3:.2f} GB")
    
    result = {
        "backend": moe_backend,
        "load_time_s": t_load,
        "gpu_mem_mb": mem_after_load,
        "peak_mem_mb": peak_mem,
        "total_params_gb": total_params / 1024**3,
        "moe_params_gb": moe_params / 1024**3,
        "attn_params_gb": attn_params / 1024**3,
        "other_params_gb": other_params / 1024**3,
    }
    
    # Cleanup
    del backend
    torch.cuda.empty_cache()
    import gc; gc.collect()
    
    return result

if __name__ == "__main__":
    results = []
    for backend_name in ["flashinfer_cutlass", "marlin"]:
        r = test_backend(backend_name)
        results.append(r)
    
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Backend':<20} {'Load(s)':<10} {'GPU(MB)':<10} {'Peak(MB)':<10} {'Params(GB)':<12} {'MoE(GB)':<10}")
    print("-" * 72)
    for r in results:
        print(f"{r['backend']:<20} {r['load_time_s']:<10.1f} {r['gpu_mem_mb']:<10.0f} "
              f"{r['peak_mem_mb']:<10.0f} {r['total_params_gb']:<12.2f} {r['moe_params_gb']:<10.2f}")
    
    with open("benchmarks/fixtures/mem_backend_compare.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to benchmarks/fixtures/mem_backend_compare.json")
