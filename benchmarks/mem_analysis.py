"""GPU memory breakdown analysis."""
import os, sys
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")
os.environ["QSR_DFLASH_CUDA_GRAPH"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
import torch

def mem_mb():
    return torch.cuda.memory_allocated() / 1024**2

def mem_reserved_mb():
    return torch.cuda.memory_reserved() / 1024**2

def main():
    backend_name = sys.argv[1] if len(sys.argv) > 1 else "marlin"
    print(f"Memory Analysis: MoE={backend_name}")
    print(f"{'='*60}")
    
    torch.cuda.reset_peak_memory_stats()
    print(f"[0] Clean state: alloc={mem_mb():.0f} MB, reserved={mem_reserved_mb():.0f} MB")
    
    from oracle.qwen36_vllm.vllm_compat import EngineArgs
    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
        "snapshots/07614121b31898586430f189d27a25a0be310843/"
    )
    args = EngineArgs(
        model=model_path, dtype="bfloat16", max_model_len=131072,
        gpu_memory_utilization=0.88, enforce_eager=True, trust_remote_code=True,
        moe_backend=backend_name,
    )
    vllm_config = args.create_engine_config()
    
    m0 = mem_mb()
    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=2112)
    m1 = mem_mb()
    print(f"[1] After model load: alloc={m1:.0f} MB (+{m1-m0:.0f} MB)")
    
    # Count model parameters
    total_params = sum(p.numel() for p in backend.model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in backend.model.parameters())
    print(f"    Model params: {total_params/1e9:.2f}B, param memory: {param_bytes/1024**2:.0f} MB")
    
    # KV cache size
    kv_total = 0
    for name, kv in backend.kv_caches.items():
        kv_total += kv.numel() * kv.element_size()
    print(f"    KV cache: {kv_total/1024**2:.0f} MB ({len(backend.kv_caches)} layers)")
    
    # SWA scratch
    swa_total = 0
    if hasattr(backend, '_swa_scratch') and backend._swa_scratch:
        for name, kv in backend._swa_scratch.items():
            swa_total += kv.numel() * kv.element_size()
    print(f"    SWA scratch: {swa_total/1024**2:.0f} MB")
    
    m_before_dflash = mem_mb()
    from runtime.backends.laguna_dflash import DFlashEngine
    engine = DFlashEngine(backend)
    m2 = mem_mb()
    print(f"[2] After DFlash init: alloc={m2:.0f} MB (+{m2-m_before_dflash:.0f} MB)")
    
    # Draft KV cache
    draft_kv_total = 0
    for name, kv in engine._draft_kv_caches.items():
        draft_kv_total += kv.numel() * kv.element_size()
    print(f"    Draft KV cache: {draft_kv_total/1024**2:.0f} MB ({len(engine._draft_kv_caches)} layers)")
    
    # Decode CG
    if engine._cuda_graph:
        print(f"    Decode CG: captured={engine._cuda_graph._captured}")
    
    # Run a short generate to trigger CG capture
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base = "The quick brown fox jumps over the lazy dog. "
    tokens = []
    chunk = tokenizer.encode(base, add_special_tokens=False)
    while len(tokens) < 4096:
        tokens.extend(chunk)
    prompt = tokens[:4096]
    
    m3 = mem_mb()
    engine.generate(prompt, max_tokens=64)
    m4 = mem_mb()
    print(f"[3] After warmup generate (4K): alloc={m4:.0f} MB (+{m4-m3:.0f} MB)")
    print(f"    Peak: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
    print(f"    Reserved: {mem_reserved_mb():.0f} MB")
    print(f"    CG: verify={engine._verify_cg is not None}, draft={engine._draft_cg is not None}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"MEMORY SUMMARY ({backend_name})")
    print(f"{'='*60}")
    print(f"  Model + KV cache:  {m1:.0f} MB")
    print(f"  DFlash (draft+CG): {m2-m1:.0f} MB")
    print(f"  CG capture:        {m4-m3:.0f} MB")
    print(f"  Total allocated:   {m4:.0f} MB")
    print(f"  Total reserved:    {mem_reserved_mb():.0f} MB")
    print(f"  GPU total:         {torch.cuda.get_device_properties(0).total_mem/1024**2:.0f} MB")
    print(f"  Free:              {torch.cuda.get_device_properties(0).total_mem/1024**2 - mem_reserved_mb():.0f} MB")

if __name__ == "__main__":
    main()
