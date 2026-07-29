import torch
"""End-to-end daemon benchmark: TTFT, ITL, tokens/s for real requests.

Run via: bf exec benchmarks/e2e_daemon_bench.py --socket /tmp/qsr-accept-test/bfd.sock --timeout-s 600

Measures:
  - TTFT (time to first token) = prefill + first draft + first verify
  - ITL (inter-token latency) = decode_time / (num_tokens - 1)
  - Throughput (tokens/s) = num_tokens / total_time
  - Acceptance rate
  - Tokens per step

Saves results to benchmarks/fixtures/e2e_daemon_bench_<date>.json
"""
# ``bf exec`` injects the live daemon objects used by this benchmark.
# ruff: noqa: F821
import json, time, os, statistics
from datetime import datetime

# ── Prompts at different context lengths ──
PROMPTS = {
    "4K_english": {
        "text": "The quick brown fox jumps over the lazy dog. In a world of artificial intelligence and machine learning, the importance of efficient inference cannot be overstated. Modern language models require careful optimization of memory bandwidth and compute utilization to achieve real-time performance. Speculative decoding offers a promising approach by using a smaller draft model to propose multiple tokens in parallel, which are then verified by the larger target model in a single forward pass. ",
        "target_tokens": 4096,
    },
    "16K_english": {
        "text": "The quick brown fox jumps over the lazy dog. In a world of artificial intelligence and machine learning, the importance of efficient inference cannot be overstated. Modern language models require careful optimization of memory bandwidth and compute utilization to achieve real-time performance. Speculative decoding offers a promising approach by using a smaller draft model to propose multiple tokens in parallel, which are then verified by the larger target model in a single forward pass. ",
        "target_tokens": 16384,
    },
    "64K_english": {
        "text": "The quick brown fox jumps over the lazy dog. In a world of artificial intelligence and machine learning, the importance of efficient inference cannot be overstated. Modern language models require careful optimization of memory bandwidth and compute utilization to achieve real-time performance. Speculative decoding offers a promising approach by using a smaller draft model to propose multiple tokens in parallel, which are then verified by the larger target model in a single forward pass. ",
        "target_tokens": 65536,
    },
    "4K_code": {
        "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n\n",
        "target_tokens": 4096,
    },
    "4K_qa": {
        "text": "Question: What is the capital of France? Answer: The capital of France is Paris. It is the largest city in France and serves as the country's political, economic, and cultural center. Paris is known for its iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. The city has a rich history dating back to the Roman era and has been a major European city for centuries. ",
        "target_tokens": 4096,
    },
}

MAX_TOKENS = 256
WARMUP_ROUNDS = 1
MEASURE_ROUNDS = 3

def make_prompt_ids(text, target_tokens):
    """Build prompt of exactly target_tokens using over-generate + truncate."""
    chunk = tokenizer.encode(text, add_special_tokens=False)
    tokens = []
    while len(tokens) < target_tokens:
        tokens.extend(chunk)
    return tokens[:target_tokens]

def run_one(prompt_ids, max_tokens=MAX_TOKENS):
    """Run one generate and return stats dict."""
    backend.reset_slot(0)
    t0 = time.perf_counter()
    tokens, stats = engine.generate(prompt_ids, max_tokens=max_tokens)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    n_tok = stats.get("num_tokens", len(tokens))
    prefill_ms = stats.get("prefill_ms", 0)
    decode_ms = stats.get("decode_ms", 0)
    n_steps = stats.get("num_steps", 0)
    accept = stats.get("acceptance_rate", 0)
    tok_per_s = stats.get("tok_per_s", 0)
    tok_per_step = stats.get("tokens_per_step", 0)

    # TTFT = prefill + first round (approx from stats)
    ttft_ms = prefill_ms + (decode_ms / max(n_steps, 1)) if n_steps > 0 else prefill_ms
    itl_ms = decode_ms / max(n_tok - 1, 1)

    return {
        "wall_s": round(wall, 4),
        "num_tokens": n_tok,
        "num_steps": n_steps,
        "prefill_ms": round(prefill_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "ttft_ms": round(ttft_ms, 2),
        "itl_ms": round(itl_ms, 2),
        "tok_per_s": round(tok_per_s, 1),
        "tok_per_step": round(tok_per_step, 2),
        "acceptance_rate": round(accept, 4),
    }

# ── Main ──
print("=" * 70)
print("E2E Daemon Benchmark")
print(f"Date: {datetime.now().isoformat()}")
print(f"Git HEAD: {os.popen('git rev-parse --short HEAD').read().strip()}")
print(f"Max tokens per request: {MAX_TOKENS}")
print(f"Warmup rounds: {WARMUP_ROUNDS}, Measure rounds: {MEASURE_ROUNDS}")
print("=" * 70)

results = {}
for name, cfg in PROMPTS.items():
    prompt_ids = make_prompt_ids(cfg["text"], cfg["target_tokens"])
    print(f"\n{'─'*50}")
    print(f"  {name}: {len(prompt_ids)} prompt tokens")
    print(f"{'─'*50}")

    # Warmup
    for w in range(WARMUP_ROUNDS):
        r = run_one(prompt_ids)
        print(f"  [warmup {w+1}] {r['tok_per_s']:.1f} tok/s, accept={r['acceptance_rate']:.1%}, "
              f"TTFT={r['ttft_ms']:.0f}ms, ITL={r['itl_ms']:.2f}ms")

    # Measured
    runs = []
    for m in range(MEASURE_ROUNDS):
        r = run_one(prompt_ids)
        runs.append(r)
        print(f"  [run {m+1}] {r['tok_per_s']:.1f} tok/s, accept={r['acceptance_rate']:.1%}, "
              f"TTFT={r['ttft_ms']:.0f}ms, ITL={r['itl_ms']:.2f}ms, "
              f"tok/step={r['tok_per_step']:.2f}")

    # Aggregate
    agg = {}
    for key in runs[0]:
        vals = [r[key] for r in runs]
        agg[f"{key}_mean"] = round(statistics.mean(vals), 3)
        agg[f"{key}_p50"] = round(statistics.median(vals), 3)
        agg[f"{key}_min"] = round(min(vals), 3)
        agg[f"{key}_max"] = round(max(vals), 3)

    results[name] = {
        "prompt_tokens": len(prompt_ids),
        "max_tokens": MAX_TOKENS,
        "runs": runs,
        "aggregate": agg,
    }

# ── Summary table ──
print(f"\n{'='*90}")
print("SUMMARY (mean of measured runs)")
print(f"{'='*90}")
print(f"{'Prompt':<16} {'Ctx':<8} {'tok/s':<10} {'TTFT(ms)':<12} {'ITL(ms)':<10} {'Accept%':<10} {'Tok/Step':<10} {'Prefill(ms)':<12}")
print("─" * 90)
for name, r in results.items():
    a = r["aggregate"]
    print(f"{name:<16} {r['prompt_tokens']:<8} {a['tok_per_s_mean']:<10.1f} "
          f"{a['ttft_ms_mean']:<12.1f} {a['itl_ms_mean']:<10.2f} "
          f"{a['acceptance_rate_mean']*100:<10.1f} {a['tok_per_step_mean']:<10.2f} "
          f"{a['prefill_ms_mean']:<12.1f}")

# Overall stats
all_tps = [r["aggregate"]["tok_per_s_mean"] for r in results.values()]
all_itl = [r["aggregate"]["itl_ms_mean"] for r in results.values()]
all_accept = [r["aggregate"]["acceptance_rate_mean"] for r in results.values()]
print("─" * 90)
print(f"{'OVERALL':<16} {'':<8} {statistics.mean(all_tps):<10.1f} "
      f"{'':<12} {statistics.mean(all_itl):<10.2f} "
      f"{statistics.mean(all_accept)*100:<10.1f}")

# ── Save ──
out_path = f"benchmarks/fixtures/e2e_daemon_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
save_data = {
    "date": datetime.now().isoformat(),
    "git_head": os.popen("git rev-parse --short HEAD").read().strip(),
    "config": {
        "max_tokens": MAX_TOKENS,
        "warmup_rounds": WARMUP_ROUNDS,
        "measure_rounds": MEASURE_ROUNDS,
    },
    "results": results,
    "summary": {
        "tok_per_s_mean": round(statistics.mean(all_tps), 1),
        "tok_per_s_p50": round(statistics.median(all_tps), 1),
        "itl_ms_mean": round(statistics.mean(all_itl), 2),
        "acceptance_mean": round(statistics.mean(all_accept), 4),
    },
}
with open(out_path, "w") as f:
    json.dump(save_data, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
