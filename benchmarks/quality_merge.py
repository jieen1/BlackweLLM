#!/usr/bin/env python3
"""Merge multiple quality_regression reports (each possibly holding a subset of
dimensions) into a single report. Useful when dimensions are run separately
(e.g. fast dims now, slow code gen later, or separate parallel runs).

Usage:
  python benchmarks/quality_merge.py --label our_runtime \
      --out evalplus_results/quality/our_runtime.json \
      evalplus_results/quality/our_runtime_fast.json \
      evalplus_results/quality/our_runtime_longctx.json \
      evalplus_results/quality/our_runtime_code.json
"""
import argparse
import datetime
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("inputs", nargs="+")
    args = p.parse_args()

    merged = {"label": args.label, "model": None, "base_url": None,
              "date": datetime.datetime.now().isoformat(), "dims": {}}
    for path in args.inputs:
        d = json.load(open(path))
        merged["model"] = merged["model"] or d.get("model")
        merged["base_url"] = merged["base_url"] or d.get("base_url")
        for dim, val in d.get("dims", {}).items():
            merged["dims"][dim] = val
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {len(args.inputs)} report(s) -> {args.out} "
          f"(dims: {sorted(merged['dims'])})")


if __name__ == "__main__":
    main()
