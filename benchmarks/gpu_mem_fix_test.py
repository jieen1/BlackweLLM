"""Quick GPU test: verify 20K/64K prompt prefill no longer OOMs."""
import os, sys, time
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")

import torch

def mem_gb():
    return torch.cuda.memory_allocated() / 1e9

def build_config():
    from oracle.qwen36_vllm.vllm_compat import EngineArgs
    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
        "snapshots/07614121b31898586430f189d27a25a0be310843/"
    )
    engine_args = EngineArgs(
        model=model_path,
        dtype="bfloat16",
        max_model_len=131072,
        gpu_memory_utilization=0.88,
        enforce_eager=True,
        trust_remote_code=True,
    )
    return engine_args.create_engine_config()

def main():
    print(f"[0] GPU free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")
    
    print("[1] Loading model...")
    t0 = time.perf_counter()
    vllm_config = build_config()
    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=4096)
    print(f"    Loaded in {time.perf_counter()-t0:.0f}s, GPU used: {mem_gb():.1f} GB")
    print(f"    SWA scratch blocks: {backend._swa_scratch_blocks}")
    
    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
            "snapshots/07614121b31898586430f189d27a25a0be310843/"
        ),
        trust_remote_code=True,
    )
    
    base_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In a world of artificial intelligence and machine learning, "
        "the importance of efficient inference cannot be overstated. "
    )
    
    for target in [5000, 20000, 64000]:
        tokens = tokenizer.encode(base_text * (target * 4 // len(base_text) + 1),
                                  add_special_tokens=False)[:target]
        print(f"\n[{target//1000}K] Prefill {len(tokens)} tokens...")
        backend.reset_slot(0)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            first = backend.prefill(0, tokens)
            elapsed = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"    OK: first_token={first}, {elapsed:.1f}s, peak={peak:.1f}GB")
            
            # Quick decode test
            tok = backend.decode(0, first)
            print(f"    Decode OK: next={tok}")
        except Exception as e:
            print(f"    FAILED: {e}")
        backend.reset_slot(0)
    
    print("\n[2] DFlash engine init...")
    from runtime.backends.laguna_dflash import DFlashEngine
    engine = DFlashEngine(backend)
    print(f"    DFlash ready, GPU used: {mem_gb():.1f} GB")
    
    # DFlash with 5K context
    tokens_5k = tokenizer.encode(base_text * (5000 * 4 // len(base_text) + 1),
                                  add_special_tokens=False)[:5000]
    print(f"\n[3] DFlash generate with {len(tokens_5k)} token prompt...")
    result_tokens, stats = engine.generate(tokens_5k, max_tokens=32)
    print(f"    Generated {len(result_tokens)} tokens")
    print(f"    Acceptance: {stats.get('acceptance_rate', 0):.1%}, "
          f"tok/s: {stats.get('tok_per_s', 0):.1f}")
    
    print("\n=== ALL PASSED ===")

if __name__ == "__main__":
    main()
