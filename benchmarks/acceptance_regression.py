"""DFlash acceptance rate regression benchmark.

Standardized prompt suite for tracking acceptance rate across code changes.
Run via bf exec (warm daemon) for steady-state measurement:

    bf exec benchmarks/acceptance_regression.py --socket <sock> --timeout-s 600

Or cold:
    /home/bot/.venvs/vllm/bin/python benchmarks/acceptance_regression.py

Reports P50, mean, min, max across the prompt suite.
Results saved to benchmarks/fixtures/acceptance_regression_<date>.json
"""


import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# Prompt suite — DO NOT change without bumping "suite_version"
# ---------------------------------------------------------------------------
SUITE_VERSION = "1.0"

# Each entry: (label, category, prompt_builder)
# prompt_builder(tokenizer) -> list[int]

def _repeat_text(tokenizer, text: str, target: int) -> list[int]:
    """Encode repeated text to exactly target tokens.

    IMPORTANT: encode the full repeated string in ONE call, then truncate.
    Do NOT compute repeat count from single-phrase token count -- tokenizer
    boundary merging makes N*len(encode(phrase)) != len(encode(phrase*N)).
    """
    # Estimate chars per token (~4 for English), over-generate by 2x
    chars_needed = target * 8
    big = text * max(chars_needed // max(len(text), 1) + 1, 10)
    ids = tokenizer.encode(big, add_special_tokens=False)
    assert len(ids) >= target, f"Need {target} tokens but only got {len(ids)}"
    return ids[:target]

def _make_suite():
    """Return list of (label, category, builder_fn)."""
    import itertools
    suite = []

    # --- Natural repeating text (best case for spec decode) ---
    suite.append(("fox-4K", "repeat",
        lambda tok: _repeat_text(tok, "The quick brown fox jumps over the lazy dog. ", 4096)))
    suite.append(("galaxy-4K", "repeat",
        lambda tok: _repeat_text(tok, "In a galaxy far far away, there lived a brave explorer. ", 4096)))
    suite.append(("ml-4K", "repeat",
        lambda tok: _repeat_text(tok, "Machine learning models require large datasets for training. ", 4096)))

    # --- Code (repeating) ---
    _code = (
        "import torch\nimport torch.nn as nn\n\nclass Block(nn.Module):\n"
        "    def __init__(self, d, h):\n        super().__init__()\n"
        "        self.attn = nn.MultiheadAttention(d, h)\n"
        "        self.ff = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))\n"
        "    def forward(self, x):\n        x = x + self.attn(x, x, x)[0]\n"
        "        return x + self.ff(x)\n"
    )
    suite.append(("code-4K", "code",
        lambda tok: _repeat_text(tok, _code, 4096)))

    # --- QA / instruction (model generates novel text) ---
    suite.append(("qa-relativity", "qa",
        lambda tok: _repeat_text(tok, "Explain the theory of relativity in simple terms. ", 2048)))
    suite.append(("qa-quicksort", "qa",
        lambda tok: _repeat_text(tok, "Write a Python function to sort a list using quicksort algorithm. ", 2048)))
    suite.append(("qa-tcp-udp", "qa",
        lambda tok: _repeat_text(tok, "What are the main differences between TCP and UDP protocols? ", 2048)))
    suite.append(("qa-photosynthesis", "qa",
        lambda tok: _repeat_text(tok, "Describe the process of photosynthesis step by step in detail. ", 2048)))

    # --- Chinese ---
    suite.append(("cn-repeat-4K", "chinese",
        lambda tok: _repeat_text(tok, "人工智能正在改变世界的方方面面，从医疗到教育，从交通到金融。", 4096)))
    suite.append(("cn-qa", "chinese",
        lambda tok: _repeat_text(tok, "请详细解释量子计算的基本原理，包括量子比特、叠加态和纠缠态的概念。", 2048)))

    # --- Synthetic token IDs (stress test, NOT representative of real workloads) ---
    suite.append(("ids-cycle-4K", "synthetic",
        lambda tok: list(itertools.islice(itertools.cycle(range(1000, 1100)), 4096))))
    suite.append(("ids-cycle-512", "synthetic",
        lambda tok: list(itertools.islice(itertools.cycle(range(1000, 1100)), 512))))

    # --- Long context (64K) ---
    suite.append(("fox-64K", "long",
        lambda tok: _repeat_text(tok, "The quick brown fox jumps over the lazy dog. ", 65536)))

    return suite

SUITE = _make_suite()

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
MAX_TOKENS_SHORT = 128
MAX_TOKENS_LONG = 256
WARMUP_ROUNDS = 1
MEASURE_ROUNDS = 3

def run_suite(backend, engine, tokenizer, warmup_rounds=WARMUP_ROUNDS,
              measure_rounds=MEASURE_ROUNDS):
    import torch
    results = []
    for label, category, builder in SUITE:
        prompt_ids = builder(tokenizer)
        max_tok = MAX_TOKENS_LONG if len(prompt_ids) > 30000 else MAX_TOKENS_SHORT
        def run_once():
            backend.reset_slot(0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tokens, stats = engine.generate_verify_only(
                prompt_ids=prompt_ids, max_tokens=max_tok,
                enable_prefix_cache=False, slot=0,
            )
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            return {
                "accept": stats["acceptance_rate"],
                "tps": stats["tokens_per_step"],
                "tok_s": stats["tok_per_s"],
                "wall_s": round(wall, 3),
                "steps": stats["num_steps"],
            }

        warmup = [run_once() for _ in range(warmup_rounds)]
        per_round = [run_once() for _ in range(measure_rounds)]
        avg_accept = sum(x["accept"] for x in per_round) / len(per_round)
        tok_s = sorted(x["tok_s"] for x in per_round)[len(per_round) // 2]
        tps = sorted(x["tps"] for x in per_round)[len(per_round) // 2]
        results.append({
            "label": label,
            "category": category,
            "prompt_len": len(prompt_ids),
            "accept": round(avg_accept, 4),
            "accept_avg": round(avg_accept, 4),
            "tps": round(tps, 2),
            "tok_s": round(tok_s, 1),
            "warmup": warmup,
            "rounds": per_round,
        })
        print(f"  {label:25s} [{category:9s}]  accept={avg_accept*100:5.1f}%  "
              f"tps={tps:5.2f}  tok/s={tok_s:6.1f}")
    return results


def summarize(results):
    import statistics
    accepts = [r["accept"] for r in results]
    # Exclude synthetic for "real workload" stats
    real = [r["accept"] for r in results if r["category"] != "synthetic"]
    accepts_sorted = sorted(accepts)
    real_sorted = sorted(real)

    def p50(xs):
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2

    summary = {
        "all": {
            "n": len(accepts),
            "mean": round(statistics.mean(accepts), 4),
            "p50": round(p50(accepts), 4),
            "min": round(min(accepts), 4),
            "max": round(max(accepts), 4),
        },
        "real_workload": {
            "n": len(real),
            "mean": round(statistics.mean(real), 4),
            "p50": round(p50(real), 4),
            "min": round(min(real), 4),
            "max": round(max(real), 4),
        },
        "by_category": {},
    }
    cats = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r["accept"])
    for cat, vals in sorted(cats.items()):
        summary["by_category"][cat] = {
            "n": len(vals),
            "mean": round(statistics.mean(vals), 4),
            "p50": round(p50(vals), 4),
        }
    return summary


# ---------------------------------------------------------------------------
# Main (bf exec path: backend/engine/tokenizer injected as globals)
# ---------------------------------------------------------------------------
def main():
    # bf exec injects these; cold-start builds them
    g = globals()
    if "backend" not in g or "engine" not in g:
        os.environ.setdefault("USE_LIBUV", "0")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from runtime.legacy_qwen36_vllm import EngineArgs
        model_path = os.path.expanduser(
            "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
            "snapshots/07614121b31898586430f189d27a25a0be310843/")
        ea = EngineArgs(model=model_path, dtype="bfloat16", max_model_len=131072,
                        gpu_memory_utilization=0.90, enforce_eager=True, trust_remote_code=True)
        cfg = ea.create_engine_config()
        from runtime.backends.laguna import LagunaBackend
        from runtime.backends.laguna_dflash import DFlashEngine
        g["backend"] = LagunaBackend(cfg, num_slots=1, blocks_per_slot=2048)
        g["engine"] = DFlashEngine(g["backend"])
        from transformers import AutoTokenizer
        g["tokenizer"] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("=" * 80)
    print(f"DFlash Acceptance Regression  (suite v{SUITE_VERSION})")
    print(f"Prompts: {len(SUITE)}  Warmup/measure: {WARMUP_ROUNDS}/{MEASURE_ROUNDS}")
    print("=" * 80)

    from bfdiag.record import run_record
    from runtime.backends.dflash_constants import NUM_SPECULATIVE_TOKENS

    with run_record(
        script="bf exec: Laguna DFlash acceptance regression",
        workload={
            "k": NUM_SPECULATIVE_TOKENS,
            "block_size": g["backend"].block_size,
            "capacity": g["backend"].num_slots,
            "max_model_len": g["backend"].vllm_config.model_config.max_model_len,
        },
        extra={
            "workload_extra": {
                "kind": "laguna-dflash-acceptance-regression",
                "suite_version": SUITE_VERSION,
                "warmup_rounds": WARMUP_ROUNDS,
                "measure_rounds": MEASURE_ROUNDS,
            }
        },
    ) as rec:
        results = run_suite(g["backend"], g["engine"], g["tokenizer"])
        summary = summarize(results)
        for result in results:
            rec.metric(f"accept_{result['label']}", result["accept"])
            rec.metric(f"tok_s_{result['label']}", result["tok_s"])
        rec.metric("accept_all_mean", summary["all"]["mean"])
        rec.metric("accept_real_mean", summary["real_workload"]["mean"])

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for scope in ("all", "real_workload"):
        s = summary[scope]
        print(f"  {scope:15s}  n={s['n']:2d}  mean={s['mean']*100:5.1f}%  "
              f"P50={s['p50']*100:5.1f}%  min={s['min']*100:5.1f}%  max={s['max']*100:5.1f}%")
    for cat, s in summary["by_category"].items():
        print(f"    {cat:13s}  n={s['n']:2d}  mean={s['mean']*100:5.1f}%  P50={s['p50']*100:5.1f}%")

    # Save fixture
    fixture = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "suite_version": SUITE_VERSION,
        "warmup_rounds": WARMUP_ROUNDS,
        "measure_rounds": MEASURE_ROUNDS,
        "summary": summary,
        "results": results,
    }
    out = Path(os.environ.get(
        "QSR_ACCEPTANCE_REGRESSION_OUT",
        _REPO / "benchmarks" / "fixtures" / f"acceptance_regression_{time.strftime('%Y%m%d')}.json",
    ))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out}")

main()
