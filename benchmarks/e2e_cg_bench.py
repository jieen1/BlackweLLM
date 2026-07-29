"""End-to-end DFlash benchmark with CUDA Graph warmup.

Runs two generates per context length:
  1st = warmup (captures verify/draft CGs)
  2nd = measured (all 3 CGs active)

Usage: /home/bot/.venvs/vllm/bin/python benchmarks/e2e_cg_bench.py [64|128|both]
"""
import os, sys, time, json
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
                      "/home/bot/project/qwen-sm120-runtime/.autotune_cache")
os.environ["QSR_DFLASH_CUDA_GRAPH"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")

import torch

def build_vllm_config(moe_backend="marlin"):
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
    return engine_args.create_engine_config()

def make_prompt(tokenizer, target_len):
    """Generate a prompt of exactly target_len tokens."""
    base = (
        "The quick brown fox jumps over the lazy dog. "
        "In a world of artificial intelligence and machine learning, "
        "the importance of efficient inference cannot be overstated. "
        "Modern language models require careful optimization of memory "
        "bandwidth and compute utilization to achieve real-time performance. "
        "Speculative decoding offers a promising approach by using a smaller "
        "draft model to propose multiple tokens in parallel, which are then "
        "verified by the larger target model in a single forward pass. "
    )
    # Iteratively build to exact token count
    tokens = []
    chunk = tokenizer.encode(base, add_special_tokens=False)
    while len(tokens) < target_len:
        tokens.extend(chunk)
    return tokens[:target_len]

def main():
    moe_backend = os.environ.get("QSR_MOE_BACKEND", "marlin")
    ctx_arg = sys.argv[1] if len(sys.argv) > 1 else "64"
    if ctx_arg == "both":
        ctx_list = [64, 128]
    else:
        ctx_list = [int(ctx_arg)]

    print("=" * 70)
    print(f"E2E DFlash + CUDA Graph Benchmark ({', '.join(f'{c}K' for c in ctx_list)}, MoE={moe_backend})")
    print("=" * 70)

    # Determine blocks_per_slot based on max context
    max_ctx = max(ctx_list) * 1024
    blocks_per_slot = max(4096, (max_ctx + 15) // 16 + 256)  # +256 margin
    print(f"  blocks_per_slot={blocks_per_slot} (max_ctx={max_ctx})")

    print("\n[1/4] Loading model...")
    t0 = time.perf_counter()
    vllm_config = build_vllm_config(moe_backend)
    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(vllm_config, num_slots=1, blocks_per_slot=blocks_per_slot)
    print(f"  Model loaded in {time.perf_counter()-t0:.1f}s")

    print("\n[2/4] Initializing DFlash...")
    from runtime.backends.laguna_dflash import DFlashEngine
    engine = DFlashEngine(backend)
    print(f"  DFlash ready, decode_cg={engine._cuda_graph is not None}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
            "snapshots/07614121b31898586430f189d27a25a0be310843/"
        ), trust_remote_code=True,
    )

    max_tokens = 256
    results = {}

    for ctx_k in ctx_list:
        ctx_len = ctx_k * 1024
        ctx_name = f"{ctx_k}K"
        print(f"\n{'='*50}")
        print(f"  === {ctx_name} context ===")
        print(f"{'='*50}")

        prompt_ids = make_prompt(tokenizer, ctx_len)
        print(f"  Prompt: {len(prompt_ids)} tokens")

        # Warmup generate (captures CGs on first call)
        print("  [Warmup] Running generate (CG capture)...")
        t_w0 = time.perf_counter()
        _, warm_stats = engine.generate(prompt_ids, max_tokens=max_tokens)
        t_warm = time.perf_counter() - t_w0
        print(f"  [Warmup] {warm_stats['tok_per_s']:.1f} tok/s, "
              f"accept={warm_stats['acceptance_rate']:.1%}, "
              f"tok/step={warm_stats['tokens_per_step']:.2f}, "
              f"cg_captured={engine._cg_captured}")
        print(f"  [Warmup] verify_cg={engine._verify_cg is not None}, "
              f"draft_cg={engine._draft_cg is not None}")

        # Measured generate (all CGs active)
        print("  [Measured] Running generate (all CGs)...")
        t_m0 = time.perf_counter()
        tokens, stats = engine.generate(prompt_ids, max_tokens=max_tokens)
        t_meas = time.perf_counter() - t_m0
        print(f"  [Measured] {stats['tok_per_s']:.1f} tok/s, "
              f"accept={stats['acceptance_rate']:.1%}, "
              f"tok/step={stats['tokens_per_step']:.2f}")
        print(f"  [Measured] Prefill: {stats['prefill_ms']:.0f}ms, "
              f"Decode: {stats['decode_ms']:.0f}ms, "
              f"Steps: {stats['num_steps']}, "
              f"Tokens: {stats['num_tokens']}")
        itl = stats['decode_ms'] / max(stats['num_tokens'] - 1, 1)
        print(f"  [Measured] ITL: {itl:.2f}ms")

        # Baseline (no DFlash, eager decode)
        print("  [Baseline] Running eager decode...")
        slot = 0
        backend.reset_slot(slot)
        torch.cuda.empty_cache()
        t_b0 = time.perf_counter()
        first = backend.prefill(slot, prompt_ids)
        t_pf = time.perf_counter()
        toks = [first]
        for _ in range(max_tokens - 1):
            tok = backend.decode(slot, toks[-1])
            toks.append(tok)
            if tok in (2, 24):
                break
        t_bend = time.perf_counter()
        decode_ms = (t_bend - t_pf) * 1000
        n_dec = len(toks) - 1
        base_tps = n_dec / max(decode_ms / 1000, 1e-6)
        base_itl = decode_ms / max(n_dec, 1)
        print(f"  [Baseline] {base_tps:.1f} tok/s, ITL {base_itl:.2f}ms, {n_dec} tokens")
        backend.reset_slot(slot)

        speedup = stats['tok_per_s'] / max(base_tps, 1e-6)
        print(f"\n  >>> Speedup vs eager baseline: {speedup:.2f}×")

        results[ctx_name] = {
            "warmup": warm_stats,
            "measured": stats,
            "baseline_tps": base_tps,
            "baseline_itl_ms": base_itl,
            "speedup": speedup,
            "itl_ms": itl,
        }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Ctx':<6} {'Baseline':<12} {'DFlash(CG)':<12} {'ITL(ms)':<10} {'Accept%':<10} {'Tok/Step':<10} {'Speedup':<8}")
    print("-" * 70)
    for ctx_name, r in results.items():
        m = r["measured"]
        print(f"{ctx_name:<6} {r['baseline_tps']:<12.1f} {m['tok_per_s']:<12.1f} "
              f"{r['itl_ms']:<10.2f} {m['acceptance_rate']:<10.1%} "
              f"{m['tokens_per_step']:<10.2f} {r['speedup']:<8.2f}×")

    out = "benchmarks/fixtures/e2e_cg_bench.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
