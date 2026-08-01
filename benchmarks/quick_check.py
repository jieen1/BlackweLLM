"""Quick 1-round benchmark for fast iteration. Usage: bf exec benchmarks/quick_check.py"""
from benchmarks.acceptance_regression import SUITE, run_once

def main(backend, engine, tokenizer):
    results = []
    for item in SUITE:
        label = item["label"]
        r = run_once(backend, engine, tokenizer, item)
        results.append({"label": label, "accept": r["accept"], "tok_s": r["tok_s"]})
        print(f"  {label:25s} accept={r['accept']:.3f}  tok/s={r['tok_s']:.1f}")
    mean_acc = sum(r["accept"] for r in results) / len(results)
    print(f"\n  mean accept: {mean_acc:.3f}")
    return results
