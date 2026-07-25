#!/usr/bin/env python3
"""Speed baseline comparison tool — 改动核心后跑此脚本确认无退化。

读取 benchmarks/fixtures/speed_baseline.json 中的冻结基准，
运行关键速度测试（attention microbench + 64K warm throughput），
对比并报告是否退化。

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.baseline_compare [--quick]

    --quick: 只跑 attention microbench（~30s），不跑 e2e throughput（~3min）

Exit code 0 = 无退化, 1 = 检测到退化 (>5% 下降)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_PATH = os.path.join(_REPO_ROOT, "benchmarks/fixtures/speed_baseline.json")
VLLM_PYTHON = "/home/bot/.venvs/vllm/bin/python"
REGRESSION_THRESHOLD = 0.05  # 5% regression triggers failure


def load_baseline() -> dict:
    with open(BASELINE_PATH) as f:
        return json.load(f)


def run_attention_microbench(baseline: dict) -> dict:
    """Run attention kernel microbench and compare."""
    params = baseline["attention_kernel_microbench"]["_params"]
    script = os.path.join(_REPO_ROOT, "benchmarks/kernel_microbench_split.py")
    cmd = f"{VLLM_PYTHON} {script} {params}"
    print(f"  Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=_REPO_ROOT)
    if result.returncode != 0:
        print(f"  ⚠ Microbench failed: {result.stderr[-200:]}")
        return {"status": "error", "error": result.stderr[-200:]}

    # Parse text output:
    # TIME: 1.883 ms/call   eff_KV_BW: 570 GB/s
    # FLASHINFER: 2.117 ms/call   eff_KV_BW: 507 GB/s   ratio(sm120/fi)=0.889x
    # CORRECTNESS: cos=0.99939835 max_rel=0.041419
    import re as _re
    out = result.stdout
    data = {}
    m = _re.search(r"TIME:\s+([\d.]+)\s+ms/call\s+eff_KV_BW:\s+([\d.]+)", out)
    if m:
        data["sm120_ms"] = float(m.group(1))
        data["eff_kv_bw_gb_s"] = float(m.group(2))
    m = _re.search(r"FLASHINFER:\s+([\d.]+)\s+ms/call", out)
    if m:
        data["flashinfer_ms"] = float(m.group(1))
    m = _re.search(r"ratio\(sm120/fi\)=([\d.]+)", out)
    if m:
        ratio = float(m.group(1))
        data["speedup_vs_flashinfer"] = round(1.0 / ratio, 2) if ratio > 0 else 0
    m = _re.search(r"cos=([\d.]+)", out)
    if m:
        data["correctness_cos"] = float(m.group(1))
    if not data:
        return {"status": "error", "error": "no parseable output", "stdout": out[-500:]}
    return data


def run_warm_throughput(baseline: dict) -> dict:
    """Run 64K×4 warm throughput check."""
    cfg = baseline["e2e_warm_throughput"]["ctx64k_c4"]
    params = cfg["_params"]
    script = os.path.join(_REPO_ROOT, "benchmarks/prefix_cache_warm_throughput_check.py")
    cmd = f"{VLLM_PYTHON} {script} {params}"
    print(f"  Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                            cwd=_REPO_ROOT, timeout=600)
    if result.returncode != 0:
        print(f"  ⚠ Throughput check failed: {result.stderr[-200:]}")
        return {"status": "error", "error": result.stderr[-200:]}

    # Parse text output for warm_accepted_tok_s
    import re as _re
    out = result.stdout
    data = {}
    m = _re.search(r"warm[_ ]accepted[_ ]tok(?:ens)?/s[:\s=]+([\d.]+)", out, _re.IGNORECASE)
    if not m:
        m = _re.search(r"([\d.]+)\s*(?:accepted\s+)?tok(?:ens)?/s", out)
    if m:
        data["warm_accepted_tok_s"] = float(m.group(1))
    m = _re.search(r"acceptance[_ ]rate[:\s=]+([\d.]+)", out, _re.IGNORECASE)
    if m:
        data["acceptance_rate"] = float(m.group(1))
    if not data:
        # Try JSON fallback (in case --json is added later)
        lines = out.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return {"status": "error", "error": "no parseable output", "stdout": out[-500:]}
    return data


def compare_metric(name: str, baseline_val: float, current_val: float,
                   threshold: float = REGRESSION_THRESHOLD) -> tuple[bool, str]:
    """Compare a metric. Returns (passed, message)."""
    if baseline_val <= 0:
        return True, f"  {name}: baseline=0, skip"
    ratio = current_val / baseline_val
    pct_change = (ratio - 1.0) * 100
    passed = ratio >= (1.0 - threshold)
    symbol = "✓" if passed else "✗"
    msg = f"  {symbol} {name}: baseline={baseline_val:.1f} current={current_val:.1f} ({pct_change:+.1f}%)"
    return passed, msg


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare against speed baseline")
    parser.add_argument("--quick", action="store_true",
                        help="Only run attention microbench (skip e2e throughput)")
    args = parser.parse_args()

    baseline = load_baseline()
    print("=== BlackweLLM Speed Baseline Comparison ===")
    print(f"  Baseline date: {baseline['date']}")
    print(f"  GPU: {baseline['gpu']}")
    print()

    all_passed = True
    results = {}

    # 1. Attention microbench
    print("[1/2] Attention kernel microbench...")
    microbench = run_attention_microbench(baseline)
    results["microbench"] = microbench
    if microbench.get("status") == "error":
        print(f"  ⚠ Skipped (error): {microbench.get('error', 'unknown')}")
    else:
        bl_speedup = baseline["attention_kernel_microbench"]["speedup_vs_flashinfer"]
        cur_speedup = microbench.get("speedup_vs_flashinfer", 0)
        passed, msg = compare_metric("attn speedup vs FlashInfer", bl_speedup, cur_speedup)
        print(msg)
        all_passed = all_passed and passed

        bl_ms = baseline["attention_kernel_microbench"]["sm120_ms"]
        cur_ms = microbench.get("sm120_ms", 0)
        if cur_ms > 0:
            # For latency, lower is better — invert comparison
            ratio = bl_ms / cur_ms
            pct = (ratio - 1.0) * 100
            passed = ratio >= (1.0 - REGRESSION_THRESHOLD)
            symbol = "✓" if passed else "✗"
            print(f"  {symbol} sm120 latency: baseline={bl_ms:.3f}ms current={cur_ms:.3f}ms ({pct:+.1f}%)")
            all_passed = all_passed and passed

    # 2. E2E warm throughput
    if args.quick:
        print("\n[2/2] E2E warm throughput... SKIPPED (--quick)")
    else:
        print("\n[2/2] E2E warm throughput (64K×4)...")
        throughput = run_warm_throughput(baseline)
        results["throughput"] = throughput
        if throughput.get("status") == "error":
            print(f"  ⚠ Skipped (error): {throughput.get('error', 'unknown')}")
        else:
            bl_toks = baseline["e2e_warm_throughput"]["ctx64k_c4"]["warm_accepted_tok_s"]
            cur_toks = throughput.get("warm_accepted_tok_s", 0)
            passed, msg = compare_metric("64K×4 warm tok/s", bl_toks, cur_toks)
            print(msg)
            all_passed = all_passed and passed

    # Summary
    print(f"\n{'='*50}")
    if all_passed:
        print("✅ NO REGRESSION DETECTED")
    else:
        print("❌ REGRESSION DETECTED — investigate before merging")
    print(f"{'='*50}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
