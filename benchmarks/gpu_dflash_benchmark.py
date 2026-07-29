"""GPU Benchmark: DFlash speculative decoding vs baseline at 64K/128K context.

Requires: vLLM venv, GPU access.
Usage: /home/bot/.venvs/vllm/bin/python benchmarks/gpu_dflash_benchmark.py

Measures:
- DFlash acceptance rate with real prompts
- Throughput (accepted tok/s) vs baseline
- ITL (inter-token latency) comparison
"""
import os
import sys
import time
import json

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")

sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")



def build_vllm_config():
    """Build VllmConfig for Laguna-S-2.1-NVFP4."""
    from runtime.legacy_qwen36_vllm import EngineArgs

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


def generate_real_prompt(tokenizer, target_len: int) -> list[int]:
    """Generate a real prompt of approximately target_len tokens."""
    # Use repetitive but natural text to fill context
    base_text = (
        "The quick brown fox jumps over the lazy dog. "
        "In a world of artificial intelligence and machine learning, "
        "the importance of efficient inference cannot be overstated. "
        "Modern language models require careful optimization of memory "
        "bandwidth and compute utilization to achieve real-time performance. "
        "Speculative decoding offers a promising approach by using a smaller "
        "draft model to propose multiple tokens in parallel, which are then "
        "verified by the larger target model in a single forward pass. "
    )
    # Repeat to fill target length
    repeats = (target_len * 4) // len(base_text) + 1  # ~4 chars per token
    full_text = base_text * repeats
    tokens = tokenizer.encode(full_text, add_special_tokens=False)
    return tokens[:target_len]


def run_baseline(backend, prompt_ids: list[int], max_tokens: int = 128) -> dict:
    """Run non-speculative baseline decode."""
    slot = 0
    backend.reset_slot(slot)

    t0 = time.perf_counter()
    first_token = backend.prefill(slot, prompt_ids)
    t_prefill = time.perf_counter()

    tokens = [first_token]
    for _ in range(max_tokens - 1):
        tok = backend.decode(slot, tokens[-1])
        tokens.append(tok)
        if tok in (2, 24):
            break
    t_end = time.perf_counter()

    backend.reset_slot(slot)

    decode_ms = (t_end - t_prefill) * 1000
    num_decode_tokens = len(tokens) - 1
    return {
        "num_tokens": len(tokens),
        "prefill_ms": (t_prefill - t0) * 1000,
        "decode_ms": decode_ms,
        "itl_ms": decode_ms / max(num_decode_tokens, 1),
        "tok_per_s": num_decode_tokens / max(decode_ms / 1000, 1e-6),
    }


def run_dflash(engine, prompt_ids: list[int], max_tokens: int = 128) -> dict:
    """Run DFlash speculative decode."""
    tokens, stats = engine.generate(prompt_ids, max_tokens=max_tokens)
    return stats


def main():
    print("=" * 70)
    print("DFlash Benchmark: 64K / 128K Context")
    print("=" * 70)

    # Load model
    print("\n[1/3] Loading main model...")
    t0 = time.perf_counter()
    vllm_config = build_vllm_config()

    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=4096)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # Initialize DFlash
    print("\n[2/3] Initializing DFlash engine...")
    from runtime.backends.laguna_dflash import DFlashEngine
    engine = DFlashEngine(backend)
    print("  DFlash ready")

    # Get tokenizer for real prompts
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
            "snapshots/07614121b31898586430f189d27a25a0be310843/"
        ),
        trust_remote_code=True,
    )

    # Run benchmarks
    print("\n[3/3] Running benchmarks...")
    results = {}
    max_tokens = 128

    for ctx_len in [64 * 1024]:  # 128K needs blocks_per_slot=8192 + more VRAM
        ctx_name = f"{ctx_len // 1024}K"
        print(f"\n  --- {ctx_name} context ---")

        # Generate real prompt
        prompt_ids = generate_real_prompt(tokenizer, ctx_len)
        print(f"  Prompt: {len(prompt_ids)} tokens")

        # Baseline
        print(f"  Running baseline ({max_tokens} tokens)...")
        base_stats = run_baseline(backend, prompt_ids, max_tokens)
        print(f"    Baseline: {base_stats['tok_per_s']:.1f} tok/s, "
              f"ITL {base_stats['itl_ms']:.2f}ms")

        # DFlash
        print(f"  Running DFlash ({max_tokens} tokens)...")
        dflash_stats = run_dflash(engine, prompt_ids, max_tokens)
        print(f"    DFlash: {dflash_stats['tok_per_s']:.1f} tok/s, "
              f"acceptance {dflash_stats['acceptance_rate']:.1%}, "
              f"tokens/step {dflash_stats['tokens_per_step']:.2f}")

        speedup = dflash_stats['tok_per_s'] / max(base_stats['tok_per_s'], 1e-6)
        print(f"    Speedup: {speedup:.2f}×")

        results[ctx_name] = {
            "baseline": base_stats,
            "dflash": dflash_stats,
            "speedup": speedup,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Context':<10} {'Baseline':<15} {'DFlash':<15} {'Accept%':<10} {'Speedup':<10}")
    print("-" * 60)
    for ctx_name, r in results.items():
        print(f"{ctx_name:<10} {r['baseline']['tok_per_s']:<15.1f} "
              f"{r['dflash']['tok_per_s']:<15.1f} "
              f"{r['dflash']['acceptance_rate']:<10.1%} "
              f"{r['speedup']:<10.2f}×")

    # Save results
    out_path = "benchmarks/fixtures/dflash_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
