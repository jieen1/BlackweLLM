"""``bf sensitivity`` -- allocator-sensitivity probes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_SWEEP = ("none", "gc", "pad16", "pad256", "gc+reset")


def _run_sweep(args: argparse.Namespace) -> int:
    from bfdiag.sensitivity.measure import derive_shape
    from bfdiag.sensitivity.verdict import Measurement, format_table, judge

    shape = derive_shape(args.ctx, args.block_size, max_tokens=args.max_tokens)
    print(
        f"block_size={shape.block_size} ctx={shape.ctx} "
        f"max_model_len={shape.max_model_len} blocks_per_slot={shape.blocks_per_slot}",
        file=sys.stderr,
    )
    measurements: list[Measurement] = []
    with tempfile.TemporaryDirectory() as td:
        for pert in args.perturbation or list(DEFAULT_SWEEP):
            out = Path(td) / f"{pert}.json"
            # A fresh process per perturbation: an earlier generate() in
            # this process would leave exactly the state under test.
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bfdiag.sensitivity.measure",
                    pert,
                    str(args.block_size),
                    str(args.ctx),
                    str(out),
                ],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0 or not out.exists():
                print(f"  {pert:>12}  FAILED\n{proc.stderr[-2000:]}", file=sys.stderr)
                continue
            row = json.loads(out.read_text())
            measurements.append(
                Measurement(
                    perturbation=row["perturbation"],
                    metric=row["acceptance_rate"],
                    output_hash=row["output_hash"],
                    allocator=row["allocator"],
                )
            )
            print(f"  {pert:>12}  accept={row['acceptance_rate']:.6f}", file=sys.stderr)

    if not measurements:
        print("no measurement succeeded", file=sys.stderr)
        return 1
    verdict = judge(measurements)
    if args.json:
        print(json.dumps({"verdict": verdict.summary, "stable": verdict.stable}, indent=2))
    else:
        print(format_table(measurements, verdict))
    return 0 if verdict.stable else 2


def _run_cycles(args: argparse.Namespace) -> int:
    from bfdiag.sensitivity.cycles import find_cycle_held_cuda_tensors, format_report

    print(format_report(find_cycle_held_cuda_tensors(top=args.top)))
    return 0


def register(subparsers) -> None:
    p = subparsers.add_parser("sensitivity", help="probe allocator sensitivity of a measurement")
    sub = p.add_subparsers(dest="sensitivity_cmd")

    sw = sub.add_parser("sweep", help="same workload under several allocator perturbations")
    sw.add_argument("--block-size", type=int, required=True)
    sw.add_argument("--ctx", type=int, required=True)
    sw.add_argument("--max-tokens", type=int, default=256)
    sw.add_argument("--perturbation", action="append", help="repeatable; default: a fixed set")
    sw.add_argument("--json", action="store_true")
    sw.set_defaults(func=_run_sweep)

    cy = sub.add_parser("cycles", help="CUDA tensors kept alive only by reference cycles")
    cy.add_argument("--top", type=int, default=10)
    cy.set_defaults(func=_run_cycles)
