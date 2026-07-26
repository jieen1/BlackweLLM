"""BlackweLLM full benchmark: prefix cache + CG + DFlash.

Usage:
    python -m benchmarks.full_comparison_ours [ctx_len]
    
Tests warm (prefix cached) and cold (no cache) scenarios.
3 rounds per scenario. Results saved to benchmarks/fixtures/full_comparison_ours.json
"""
import gc, json, os, sys, time
os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
os.environ.setdefault("QSR_DFLASH_CUDA_GRAPH", "1")
os.environ["SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_grad_enabled(False)

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)
SUFFIX_LEN = 10000  # 10K prompt suffix
MAX_TOKENS = 256
NUM_ROUNDS = 3

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# Deterministic prompt construction (same as vLLM script)
BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In a world of artificial intelligence and machine learning, "
    "the importance of efficient inference cannot be overstated. "
    "Modern language models require careful optimization of memory "
    "bandwidth and compute utilization to achieve real-time performance. "
    "Speculative decoding offers a promising approach by using a smaller "
    "draft model to propose multiple tokens in parallel, which are then "
    "verified by the larger target model in a single forward pass. "
)
CHUNK_IDS = tok.encode(BASE_TEXT, add_special_tokens=False)

SUFFIX_TEXT = (
    "Deep learning architectures have revolutionized natural language processing. "
    "Transformer models use self-attention mechanisms to capture long-range dependencies "
    "in sequential data. The key innovation is the multi-head attention mechanism which "
    "allows the model to jointly attend to information from different representation "
    "subspaces at different positions. Mixture of experts models further improve "
    "efficiency by routing tokens to specialized expert networks. Quantization "
    "techniques reduce memory footprint while maintaining model quality. "
)
SUFFIX_IDS = tok.encode(SUFFIX_TEXT, add_special_tokens=False)

def make_ids(target_len: int, use_suffix: bool = False) -> list[int]:
    """Build deterministic token sequence of exact length."""
    source = SUFFIX_IDS if use_suffix else CHUNK_IDS
    ids = []
    while len(ids) < target_len:
        ids.extend(source)
    return ids[:target_len]


def run_context(ctx_len: int):
    """Run warm + cold benchmarks for one context length."""
    from runtime.compat_vllm import EngineArgs
    from runtime.backends.laguna import LagunaBackend
    from runtime.backends.laguna_dflash import DFlashEngine

    base_len = ctx_len - SUFFIX_LEN
    base_ids = make_ids(base_len)
    suffix_ids = make_ids(SUFFIX_LEN, use_suffix=True)
    full_ids = base_ids + suffix_ids
    total_len = len(full_ids)

    max_model_len = max(total_len + MAX_TOKENS + 1024, 262144)
    bps = (total_len + 63) // 64 + 512

    print(f"\n{'='*70}", file=sys.stderr)
    print(f"[OURS] ctx={ctx_len}: base={base_len}, suffix={SUFFIX_LEN}, "
          f"total={total_len}, max_model_len={max_model_len}", file=sys.stderr)

    vc = EngineArgs(
        model=MODEL, dtype="bfloat16", max_model_len=max_model_len,
        gpu_memory_utilization=0.90, enforce_eager=True,
        trust_remote_code=True, disable_log_stats=True,
    ).create_engine_config()
    backend = LagunaBackend(vc, num_slots=1, block_size=64, blocks_per_slot=bps)
    engine = DFlashEngine(backend)

    # Warmup: small generate to trigger JIT + CG capture
    backend.reset_slot(0)
    for kv in engine._draft_kv_caches.values():
        kv.zero_()
    engine.generate_verify_only(base_ids[:256], max_tokens=5, temperature=0.0,
                                slot=0, enable_prefix_cache=False)
    torch.cuda.synchronize()
    print(f"  Init done, CG captured={engine._cg_captured}", file=sys.stderr)

    results = {"ctx_len": ctx_len, "base_len": base_len, "suffix_len": SUFFIX_LEN,
               "engine": "blackwellm", "warm": [], "cold": []}

    # === COLD scenario: no prefix cache, full prefill ===
    print(f"  [COLD] {NUM_ROUNDS} rounds...", file=sys.stderr)
    for r in range(NUM_ROUNDS):
        backend.reset_slot(0)
        for kv in engine._draft_kv_caches.values():
            kv.zero_()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens, stats = engine.generate_verify_only(
            full_ids, max_tokens=MAX_TOKENS, temperature=0.0,
            slot=0, enable_prefix_cache=False)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        stats["wall_s"] = round(wall, 2)
        stats["round"] = r
        results["cold"].append(stats)
        print(f"    cold r{r}: {stats['tok_per_s']:.1f} tok/s, "
              f"acc={stats['acceptance_rate']:.1%}, "
              f"pf={stats['prefill_ms']/1000:.1f}s, wall={wall:.1f}s", file=sys.stderr)

    # === WARM scenario: prefill base first (populates prefix cache), then full ===
    print(f"  [WARM] prefilling base ({base_len})...", file=sys.stderr)
    backend.reset_slot(0)
    for kv in engine._draft_kv_caches.values():
        kv.zero_()
    engine.generate_verify_only(base_ids, max_tokens=5, temperature=0.0,
                                slot=0, enable_prefix_cache=True)
    torch.cuda.synchronize()
    print(f"  [WARM] {NUM_ROUNDS} rounds (prefix cached)...", file=sys.stderr)
    for r in range(NUM_ROUNDS):
        # Do not reset either main or draft KV: both are part of the prefix
        # cache state. The engine falls back to cold prefill if ring history
        # is no longer sufficient for a safe rewind.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens, stats = engine.generate_verify_only(
            full_ids, max_tokens=MAX_TOKENS, temperature=0.0,
            slot=0, enable_prefix_cache=True)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        stats["wall_s"] = round(wall, 2)
        stats["round"] = r
        results["warm"].append(stats)
        print(f"    warm r{r}: {stats['tok_per_s']:.1f} tok/s, "
              f"acc={stats['acceptance_rate']:.1%}, "
              f"pf={stats['prefill_ms']/1000:.1f}s, wall={wall:.1f}s", file=sys.stderr)

    # Cleanup
    del engine, backend
    torch.cuda.empty_cache()
    gc.collect()
    return results


def main():
    ctx_lengths = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [65536]
    all_results = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": "blackwellm",
        "config": {
            "model": "poolside/Laguna-S-2.1-NVFP4",
            "draft": "poolside/Laguna-S-2.1-DFlash-NVFP4",
            "max_tokens": MAX_TOKENS,
            "suffix_len": SUFFIX_LEN,
            "num_rounds": NUM_ROUNDS,
            "prefix_cache": True,
            "cuda_graph": True,
            "dflash": True,
            "block_size": 64,
            "gpu_memory_utilization": 0.90,
        },
        "results": [],
    }
    for ctx in ctx_lengths:
        result = run_context(ctx)
        all_results["results"].append(result)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fixtures", "full_comparison_ours.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}", file=sys.stderr)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
