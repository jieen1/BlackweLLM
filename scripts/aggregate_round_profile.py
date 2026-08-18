"""Aggregate one QSR_PROFILE_ROUNDS=1 server log into per-phase statistics.

Each round emits one JSON line (label=mtp_round_bN or dspark_ragged_round_bN) with wall-time phases
plus CUDA-event GPU spans (verify_gpu_ms / sync_gpu_ms / draft_gpu_ms).
This prints count, median, p90 and max for every phase/note so a run can be
compared against an earlier one without hand-picking log lines.

Usage:
    python scripts/aggregate_round_profile.py logs/quality/<file>.log
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def _summarize(values: list[float]) -> str:
    if not values:
        return "n=0"
    vals = sorted(values)
    p90 = vals[min(len(vals) - 1, int(len(vals) * 0.90))]
    return (
        f"n={len(vals)} med={statistics.median(vals):.2f} "
        f"mean={statistics.mean(vals):.2f} p90={p90:.2f} max={max(vals):.2f}"
    )


def main(path: str) -> None:
    phases: dict[str, list[float]] = defaultdict(list)
    notes: dict[str, list[float]] = defaultdict(list)
    rounds = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("label", "").startswith(("mtp_round", "dspark_ragged_round")):
                rounds += 1
                for name, ms in rec.get("phases", []):
                    phases[name].append(float(ms))
                for name, ms in rec.get("notes", {}).items():
                    if name in ("t_begin", "t_end"):
                        continue
                    notes[name].append(float(ms))
    print(f"rounds={rounds}")
    for name in sorted(phases):
        print(f"phase {name:22s} {_summarize(phases[name])}")
    for name in sorted(notes):
        print(f"note  {name:22s} {_summarize(notes[name])}")


if __name__ == "__main__":
    main(sys.argv[1])
