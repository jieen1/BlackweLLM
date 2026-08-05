#!/usr/bin/env python3
"""Regression gate: compare a candidate quality report against a baseline.

Exits non-zero if any dimension regresses beyond its tolerance. Used to ensure
large changes do not silently degrade inference quality of the custom runtime
relative to an earlier baseline on the identical harness.

Tolerances are absolute fractions and intentionally small-but-forgiving because
greedy decoding on tiny self-contained sets has run-to-run noise:
  code (HumanEval+/MBPP+ pass@1) : 0.03
  tool (full accuracy)           : 0.05
  agent (final-answer accuracy)  : 0.10  (only 4 scenarios -> coarse)
  longctx (retrieval accuracy)   : 0.05

Usage:
  python benchmarks/quality_compare.py \
      --candidate evalplus_results/quality/our_runtime.json \
      --baseline  evalplus_results/quality/baseline.json
"""
import argparse
import json
import sys

TOL = {
    "code": 0.03,
    "tool": 0.05,
    "agent": 0.10,
    "longctx": 0.05,
}

# (dimension, metric key path, higher_is_better)
METRICS = [
    ("code", ["humaneval_pass_at_1"], True),
    ("code", ["humaneval_plus_pass_at_1"], True),
    ("tool", ["accuracy"], True),
    ("tool", ["name_accuracy"], True),
    ("agent", ["accuracy"], True),
    ("agent", ["tool_invocation_rate"], True),
    ("longctx", ["accuracy"], True),
]


def _get(d, dim, path):
    node = d.get("dims", {}).get(dim)
    if node is None:
        return None
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--strict", action="store_true",
                   help="fail on any regression (ignore tolerance)")
    args = p.parse_args()

    cand = json.load(open(args.candidate))
    base = json.load(open(args.baseline))
    print(f"Comparing candidate={cand.get('label')} ({args.candidate})")
    print(f"        vs baseline={base.get('label')} ({args.baseline})\n")

    rows = []
    regressions = 0
    print(f"{'dimension':<10} {'metric':<26} {'baseline':>9} {'candidate':>10} "
          f"{'delta':>8} {'tol':>6}  status")
    for dim, path, higher in METRICS:
        b = _get(base, dim, path)
        c = _get(cand, dim, path)
        if b is None or c is None:
            continue
        metric = ".".join(path)
        delta = c - b
        tol = 0.0 if args.strict else TOL[dim]
        reg = (delta < -tol) if higher else (delta > tol)
        status = "REGRESSION" if reg else "ok"
        if reg:
            regressions += 1
        print(f"{dim:<10} {metric:<26} {b:>9.3f} {c:>10.3f} "
              f"{delta:>+8.3f} {tol:>6.2f}  {status}")
        rows.append((dim, metric, b, c, delta, reg))

    # longctx per-length breakdown (informational + gated)
    bl = _get(base, "longctx", ["by_length"]) or {}
    cl = _get(cand, "longctx", ["by_length"]) or {}
    if bl and cl:
        print("\nlongctx by context length:")
        for k in sorted(set(bl) | set(cl), key=lambda x: int(x)):
            bv = bl.get(k); cv = cl.get(k)
            if bv is None or cv is None:
                continue
            delta = cv - bv
            reg = delta < -TOL["longctx"]
            if reg:
                regressions += 1
            print(f"  {int(k):>7} tokens: baseline={bv:.3f} candidate={cv:.3f} "
                  f"delta={delta:+.3f}  {'REGRESSION' if reg else 'ok'}")

    print()
    if regressions:
        print(f"RESULT: FAIL - {regressions} metric(s) regressed beyond tolerance.")
        sys.exit(1)
    print("RESULT: PASS - no quality regression detected.")
    sys.exit(0)


if __name__ == "__main__":
    main()
