"""Quick test: TRTLLM MoE backend vs CUTLASS."""
import os, sys, time
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

def main():
    from runtime.legacy_qwen36_vllm import EngineArgs
    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
        "snapshots/07614121b31898586430f189d27a25a0be310843/"
    )
    
    backend = sys.argv[1] if len(sys.argv) > 1 else "flashinfer_trtllm"
    ctx_k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ctx_len = ctx_k * 1024
    
    print(f"Testing MoE backend: {backend}, context: {ctx_k}K")
    
    engine_args = EngineArgs(
        model=model_path, dtype="bfloat16", max_model_len=131072,
        gpu_memory_utilization=0.88, enforce_eager=True, trust_remote_code=True,
        moe_backend=backend,
    )
    vllm_config = engine_args.create_engine_config()
    
    print("Loading model...")
    t0 = time.perf_counter()
    from runtime.backends.laguna import LagunaBackend
    bk = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=max(4096, ctx_len//16+256))
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")
    
    from runtime.backends.laguna_dflash import DFlashEngine
    engine = DFlashEngine(bk)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = make_prompt(tokenizer, ctx_len)
    
    # Warmup (CG capture)
    print("Warmup...")
    _, ws = engine.generate(prompt, max_tokens=128)
    print(f"  Warmup: {ws['tok_per_s']:.1f} tok/s, accept={ws['acceptance_rate']:.1%}")
    
    # Measured
    print("Measured...")
    _, stats = engine.generate(prompt, max_tokens=256)
    print(f"  Result: {stats['tok_per_s']:.1f} tok/s, accept={stats['acceptance_rate']:.1%}, "
          f"tok/step={stats['tokens_per_step']:.2f}, ITL={stats['decode_ms']/(stats['num_tokens']-1):.2f}ms")
    
    # Baseline
    print("Baseline (eager)...")
    slot = 0
    bk.reset_slot(slot)
    torch.cuda.empty_cache()
    first = bk.prefill(slot, prompt)
    t_pf = time.perf_counter()
    toks = [first]
    for _ in range(127):
        tok = bk.decode(slot, toks[-1])
        toks.append(tok)
        if tok in (2, 24): break
    t_end = time.perf_counter()
    dec_ms = (t_end - t_pf) * 1000
    n = len(toks) - 1
    print(f"  Baseline: {n/(dec_ms/1000):.1f} tok/s, ITL {dec_ms/n:.2f}ms")
    bk.reset_slot(slot)

if __name__ == "__main__":
    main()
