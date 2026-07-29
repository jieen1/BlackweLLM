"""Discriminant: 8192 (single chunk) vs 8193 (triggers chunked) acceptance rate.
Same context length, only variable is the chunked prefill path."""
import os, sys
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")
os.environ["QSR_DFLASH_CUDA_GRAPH"] = "0"  # eager for clean comparison
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")

def make_prompt(tokenizer, n):
    base = "The quick brown fox jumps over the lazy dog. "
    tokens = []
    chunk = tokenizer.encode(base, add_special_tokens=False)
    while len(tokens) < n:
        tokens.extend(chunk)
    return tokens[:n]

def test_acceptance(engine, backend, tokenizer, ctx_len, num_steps=20):
    """Run speculative decode and return acceptance rate."""
    prompt = make_prompt(tokenizer, ctx_len)
    tokens, stats = engine.generate(prompt, max_tokens=num_steps * 16)
    return stats['acceptance_rate'], stats['tokens_per_step'], stats['tok_per_s']

def main():
    from oracle.qwen36_vllm.vllm_compat import EngineArgs
    model_path = os.path.expanduser(
        "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
        "snapshots/07614121b31898586430f189d27a25a0be310843/"
    )
    engine_args = EngineArgs(
        model=model_path, dtype="bfloat16", max_model_len=131072,
        gpu_memory_utilization=0.88, enforce_eager=True, trust_remote_code=True,
        moe_backend="marlin",
    )
    vllm_config = engine_args.create_engine_config()

    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=2112)

    from runtime.backends.laguna_dflash import DFlashEngine
    engine = DFlashEngine(backend)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    chunk_size = backend._prefill_chunk_tokens
    print(f"Prefill chunk size: {chunk_size}")
    print(f"{'='*70}")
    print(f"{'Ctx Len':<10} {'Chunked?':<10} {'Accept%':<10} {'Tok/Step':<10} {'Tok/s':<10}")
    print(f"{'-'*50}")

    # Test around the chunk boundary
    test_cases = [
        (4096, "single"),
        (chunk_size, "single (exact)"),
        (chunk_size + 1, "CHUNKED (+1)"),
        (chunk_size + 100, "CHUNKED (+100)"),
        (chunk_size * 2, "CHUNKED (2x)"),
        (16384, "CHUNKED (16K)"),
        (32768, "CHUNKED (32K)"),
        (65536, "CHUNKED (64K)"),
    ]

    for ctx_len, label in test_cases:
        accept, tps, tok_s = test_acceptance(engine, backend, tokenizer, ctx_len, num_steps=15)
        is_chunked = "YES" if ctx_len > chunk_size else "no"
        print(f"{ctx_len:<10} {is_chunked:<10} {accept:<10.1%} {tps:<10.2f} {tok_s:<10.1f}  [{label}]")

    print(f"\n{'='*70}")
    print("If acceptance drops sharply at chunk_size+1 → chunked aux/KV bug")
    print("If acceptance degrades smoothly → architectural limitation")

if __name__ == "__main__":
    main()
