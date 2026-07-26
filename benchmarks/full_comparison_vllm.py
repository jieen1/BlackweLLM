"""vLLM 0.26.0 full benchmark: prefix cache + CG + DFlash.

Usage:
    python -m benchmarks.full_comparison_vllm [ctx_len]

IDENTICAL parameters to full_comparison_ours.py.
Results saved to benchmarks/fixtures/full_comparison_vllm.json
"""
import gc, json, os, sys, time
os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

import torch
torch.set_grad_enabled(False)

MODEL = "poolside/Laguna-S-2.1-NVFP4"
DRAFT = "poolside/Laguna-S-2.1-DFlash-NVFP4"
SUFFIX_LEN = 10000
MAX_TOKENS = 256
NUM_ROUNDS = 3

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# IDENTICAL prompt construction as ours
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

def make_ids(target_len: int, seed_offset: int = 0) -> list[int]:
    ids = []
    while len(ids) < target_len:
        ids.extend(CHUNK_IDS)
    if seed_offset > 0:
        ids = [(t + seed_offset) % 100352 for t in ids]
    return ids[:target_len]


def run_context(ctx_len: int):
    from vllm import LLM, SamplingParams

    base_len = ctx_len - SUFFIX_LEN
    base_ids = make_ids(base_len)
    suffix_ids = make_ids(SUFFIX_LEN, use_suffix=True)
    full_ids = base_ids + suffix_ids
    total_len = len(full_ids)

    max_model_len = max(total_len + MAX_TOKENS + 1024, 262144)

    base_text = tok.decode(base_ids)
    full_text = tok.decode(full_ids)

    print(f"\n{'='*70}", file=sys.stderr)
    print(f"[vLLM] ctx={ctx_len}: base={base_len}, suffix={SUFFIX_LEN}, "
          f"total={total_len}, max_model_len={max_model_len}", file=sys.stderr)

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        enforce_eager=False,  # CG enabled
        dtype="bfloat16",
        disable_log_stats=True,
        max_num_seqs=1,
        moe_backend="marlin",
        enable_prefix_caching=True,
        spec_method="dflash",
        spec_model=DRAFT,
        spec_tokens=15,
    )

    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0)
    results = {"ctx_len": ctx_len, "base_len": base_len, "suffix_len": SUFFIX_LEN,
               "engine": "vllm", "warm": [], "cold": []}

    # Warmup JIT
    llm.generate(["Hello"], params)
    torch.cuda.synchronize()
    print("  JIT warmup done", file=sys.stderr)

    # === COLD: no prefix cache hit (use unique prefix per round) ===
    print(f"  [COLD] {NUM_ROUNDS} rounds...", file=sys.stderr)
    for r in range(NUM_ROUNDS):
        # Reset prefix cache by using llm.reset_prefix_cache() if available
        if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'reset_prefix_cache'):
            llm.llm_engine.reset_prefix_cache()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate([full_text], params)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        out = outputs[0]
        n_tokens = len(out.outputs[0].token_ids)
        stats = {
            "round": r,
            "num_tokens": n_tokens,
            "wall_s": round(wall, 2),
            "tok_per_s": round(n_tokens / max(wall, 1e-6), 1),
            "preview": out.outputs[0].text[:80],
        }
        results["cold"].append(stats)
        print(f"    cold r{r}: {stats['tok_per_s']} tok/s, wall={wall:.1f}s", file=sys.stderr)

    # === WARM: prefill base first, then full (prefix cache hit) ===
    print(f"  [WARM] prefilling base ({base_len})...", file=sys.stderr)
    llm.generate([base_text], SamplingParams(max_tokens=1, temperature=0))
    torch.cuda.synchronize()
    print(f"  [WARM] {NUM_ROUNDS} rounds (prefix cached)...", file=sys.stderr)
    for r in range(NUM_ROUNDS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate([full_text], params)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        out = outputs[0]
        n_tokens = len(out.outputs[0].token_ids)
        stats = {
            "round": r,
            "num_tokens": n_tokens,
            "wall_s": round(wall, 2),
            "tok_per_s": round(n_tokens / max(wall, 1e-6), 1),
            "preview": out.outputs[0].text[:80],
        }
        results["warm"].append(stats)
        print(f"    warm r{r}: {stats['tok_per_s']} tok/s, wall={wall:.1f}s", file=sys.stderr)

    del llm
    torch.cuda.empty_cache()
    gc.collect()
    return results


def main():
    ctx_lengths = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [65536]
    all_results = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": "vllm-0.26.0",
        "config": {
            "model": MODEL,
            "draft": DRAFT,
            "max_tokens": MAX_TOKENS,
            "suffix_len": SUFFIX_LEN,
            "num_rounds": NUM_ROUNDS,
            "prefix_cache": True,
            "cuda_graph": True,
            "dflash": True,
            "moe_backend": "marlin",
            "gpu_memory_utilization": 0.90,
        },
        "results": [],
    }
    for ctx in ctx_lengths:
        result = run_context(ctx)
        all_results["results"].append(result)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "fixtures", "full_comparison_vllm.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}", file=sys.stderr)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
