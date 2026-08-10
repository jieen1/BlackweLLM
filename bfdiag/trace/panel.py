"""Aggregation and rendering for ``bf trace show``/``bf trace diff``: the
"vital signs panel" a human actually reads after a run, so a failing run's
full trace answers questions instead of requiring a re-run with extra
prints.

Pure functions over ``list[events.RoundEvent]`` -- no file I/O, no argparse,
so this module is trivially unit-testable (``tests/test_bfdiag_trace.py``).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field

from bfdiag.trace.events import RoundEvent

# A round is "outlier-slow" when its t_round is both far (in robust
# z-score terms) from the run's typical round AND large in absolute terms --
# the second guard keeps sub-millisecond jitter in an otherwise-fast run from
# being reported as an "outlier" (e.g. notes/2026-07-27-dflash-concurrency-
# handoff.md's 270-second mystery is ~1e6x a normal ~ms-scale round; a robust
# z-score alone would also flag a 2ms round in a 0.5ms-median run, which is
# real variance but not diagnostically interesting).
OUTLIER_Z_THRESHOLD = 8.0
OUTLIER_MIN_ABS_MS = 50.0

PHASES = ("t_main_forward", "t_draft", "t_verify", "t_commit", "t_round")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _median_abs_deviation(values: list[float], med: float) -> float:
    if not values:
        return 0.0
    return statistics.median([abs(v - med) for v in values])


@dataclass
class RunStats:
    num_rounds: int
    dflash_rounds: int
    acceptance_rate: float | None
    reject_position_histogram: dict[int, int]
    path_counts: dict[str, int]
    cg_hit_rate: float
    cg_miss_reason_histogram: dict[str, int]
    phase_percentiles: dict[str, dict[str, float]]
    outliers: list[dict] = field(default_factory=list)
    # Rounds overwritten by ring wraparound before the earliest surviving
    # row -- never silently lost: round_idx is a monotonic counter assigned
    # by RoundRing.begin_round regardless of physical-row reuse, so the
    # earliest surviving row's round_idx IS the drop count (0 if the ring
    # never wrapped). See ring.py's module docstring.
    dropped: int = 0

    def to_dict(self) -> dict:
        return {
            "num_rounds": self.num_rounds,
            "dflash_rounds": self.dflash_rounds,
            "acceptance_rate": self.acceptance_rate,
            "reject_position_histogram": self.reject_position_histogram,
            "path_counts": self.path_counts,
            "cg_hit_rate": self.cg_hit_rate,
            "cg_miss_reason_histogram": self.cg_miss_reason_histogram,
            "phase_percentiles": self.phase_percentiles,
            "outliers": self.outliers,
            "dropped": self.dropped,
        }


def compute_stats(rows: list[RoundEvent]) -> RunStats:
    num_rounds = len(rows)
    dropped = rows[0].round_idx if rows else 0
    dflash_rows = [r for r in rows if r.draft_tokens_n > 0]
    decode_rows = [r for r in rows if r.event_kind == "decode_round"]

    total_draft = sum(r.draft_tokens_n for r in dflash_rows)
    total_accepted = sum(r.accepted_n for r in dflash_rows)
    acceptance_rate = (total_accepted / total_draft) if total_draft > 0 else None

    reject_hist: dict[int, int] = {}
    for r in dflash_rows:
        reject_hist[r.reject_position] = reject_hist.get(r.reject_position, 0) + 1

    path_counts: dict[str, int] = {}
    for r in decode_rows:
        path_counts[r.path] = path_counts.get(r.path, 0) + 1
    cg_hit_rate = (path_counts.get("cg_replay", 0) / len(decode_rows)) if decode_rows else 0.0

    miss_reason_hist: dict[str, int] = {}
    for r in decode_rows:
        if r.path != "cg_replay":
            miss_reason_hist[r.cg_miss_reason] = miss_reason_hist.get(r.cg_miss_reason, 0) + 1

    phase_percentiles: dict[str, dict[str, float]] = {}
    for phase in PHASES:
        values = [getattr(r, phase) for r in rows if getattr(r, phase) > 0.0]
        phase_percentiles[phase] = {
            "p50": _percentile(values, 0.50),
            "p99": _percentile(values, 0.99),
        }

    outliers = _find_outliers(rows)

    return RunStats(
        num_rounds=num_rounds,
        dflash_rounds=len(dflash_rows),
        acceptance_rate=acceptance_rate,
        reject_position_histogram=reject_hist,
        path_counts=path_counts,
        cg_hit_rate=cg_hit_rate,
        cg_miss_reason_histogram=miss_reason_hist,
        phase_percentiles=phase_percentiles,
        outliers=outliers,
        dropped=dropped,
    )


def _find_outliers(rows: list[RoundEvent]) -> list[dict]:
    t_rounds = [r.t_round for r in rows if r.t_round > 0.0]
    if len(t_rounds) < 4:
        return []
    med = statistics.median(t_rounds)
    mad = _median_abs_deviation(t_rounds, med)
    # 1.4826x MAD approximates a standard deviation for normally-distributed
    # data (the usual robust-z-score constant); guard mad==0 (a perfectly
    # uniform run) by falling back to the absolute-ms floor alone.
    scale = mad * 1.4826 if mad > 0 else 0.0
    outliers = []
    for r in rows:
        if r.t_round <= 0.0:
            continue
        z = (r.t_round - med) / scale if scale > 0 else float("inf")
        if r.t_round >= OUTLIER_MIN_ABS_MS and (scale == 0.0 or z >= OUTLIER_Z_THRESHOLD):
            outliers.append(
                {
                    "round_idx": r.round_idx,
                    "slot": r.slot,
                    "t_round_ms": r.t_round,
                    "z_score": z if scale > 0 else None,
                    "median_ms": med,
                }
            )
    return outliers


def render_round_table(rows: list[RoundEvent], limit: int = 50) -> str:
    """Per-round table, most recent ``limit`` rows (0 = all)."""
    shown = rows if limit <= 0 else rows[-limit:]
    header = (
        f"{'round':>6} {'slot':>4} {'kind':>13} {'pos':>6} {'rows':>5} {'ratio':>5} "
        f"{'win':>5} {'r4':>5} {'r128':>5} {'kv_before':>9} {'path':>10} "
        f"{'miss_reason':>20} {'draft_n':>7} {'acc_n':>5} {'rej_pos':>7} {'t_round(ms)':>11}"
    )
    lines = [header, "-" * len(header)]
    for r in shown:
        lines.append(
            f"{r.round_idx:>6} {r.slot:>4} {r.event_kind:>13} {r.position:>6} {r.row_count:>5} "
            f"{r.compressor_ratio:>5} {r.window_entries:>5} {r.ratio4_entries:>5} "
            f"{r.ratio128_entries:>5} {r.kv_len_before:>9} {r.path:>10} "
            f"{r.cg_miss_reason:>20} {r.draft_tokens_n:>7} {r.accepted_n:>5} "
            f"{r.reject_position:>7} {r.t_round:>11.3f}"
        )
    if limit > 0 and len(rows) > limit:
        lines.insert(0, f"(showing last {limit} of {len(rows)} rounds; pass --limit 0 for all)")
    return "\n".join(lines)


def render_summary(stats: RunStats) -> str:
    lines = ["=== bfdiag trace summary ===", f"total rounds: {stats.num_rounds}"]
    if stats.dropped:
        lines.append(
            f"dropped (ring wraparound, overwritten before the earliest surviving "
            f"round): {stats.dropped}"
        )
    lines.append(f"DFlash rounds (draft_tokens_n > 0): {stats.dflash_rounds}")
    if stats.acceptance_rate is not None:
        lines.append(f"acceptance rate: {stats.acceptance_rate:.3%}")
    else:
        lines.append("acceptance rate: n/a (no DFlash rounds)")

    lines.append("")
    lines.append("-- reject_position histogram (DFlash rounds; -1 = full accept) --")
    for pos in sorted(stats.reject_position_histogram):
        count = stats.reject_position_histogram[pos]
        lines.append(f"  {pos:>3}: {count}")

    lines.append("")
    lines.append("-- CUDA Graph path --")
    lines.append(f"  cg_hit_rate: {stats.cg_hit_rate:.3%}")
    for path, count in sorted(stats.path_counts.items()):
        lines.append(f"  {path}: {count}")
    if stats.cg_miss_reason_histogram:
        lines.append("  eager/cg_miss reason distribution:")
        for reason, count in sorted(stats.cg_miss_reason_histogram.items()):
            lines.append(f"    {reason}: {count}")

    lines.append("")
    lines.append("-- phase latency (ms), rounds where phase ran --")
    for phase in PHASES:
        p = stats.phase_percentiles[phase]
        lines.append(f"  {phase:<14} p50={p['p50']:.3f}  p99={p['p99']:.3f}")

    lines.append("")
    lines.append("-- outliers (t_round far from the run's own median) --")
    if not stats.outliers:
        lines.append("  none")
    else:
        for o in stats.outliers:
            z_str = f"z={o['z_score']:.1f}" if o["z_score"] is not None else "z=n/a"
            lines.append(
                f"  round {o['round_idx']} (slot {o['slot']}): "
                f"{o['t_round_ms']:.1f}ms vs median {o['median_ms']:.3f}ms ({z_str})"
            )
    return "\n".join(lines)


def render_json(rows: list[RoundEvent], stats: RunStats) -> str:
    return json.dumps(
        {"rounds": [r.__dict__ for r in rows], "summary": stats.to_dict()},
        indent=2,
    )


# --------------------------------------------------------------------------
# bf trace diff
# --------------------------------------------------------------------------

# Fields compared for divergence. Timing fields are deliberately excluded --
# they vary run to run even when nothing is structurally wrong, so a diff
# based on them would never stabilize.
_DIFF_FIELDS = (
    "event_kind",
    "slot",
    "position",
    "row_count",
    "compressor_ratio",
    "window_entries",
    "ratio4_entries",
    "ratio128_entries",
    "kv_len_before",
    "path",
    "cg_miss_reason",
    "draft_tokens_n",
    "accepted_n",
    "reject_position",
    "bonus_token",
)


@dataclass
class DiffResult:
    first_divergence_round: int | None
    diverging_fields: dict[str, tuple[object, object]]
    len_a: int
    len_b: int

    def to_dict(self) -> dict:
        return {
            "first_divergence_round": self.first_divergence_round,
            "diverging_fields": {
                k: {"a": v[0], "b": v[1]} for k, v in self.diverging_fields.items()
            },
            "len_a": self.len_a,
            "len_b": self.len_b,
        }


def diff_traces(a: list[RoundEvent], b: list[RoundEvent]) -> DiffResult:
    """Align two traces round-by-round (by position, since ``round_idx`` is
    only unique within a single run) and report the first round where any
    structural field diverges."""
    for i, (ra, rb) in enumerate(zip(a, b)):
        diverging = {
            field: (getattr(ra, field), getattr(rb, field))
            for field in _DIFF_FIELDS
            if getattr(ra, field) != getattr(rb, field)
        }
        if diverging:
            return DiffResult(
                first_divergence_round=i,
                diverging_fields=diverging,
                len_a=len(a),
                len_b=len(b),
            )
    return DiffResult(
        first_divergence_round=None,
        diverging_fields={},
        len_a=len(a),
        len_b=len(b),
    )


def render_diff(result: DiffResult) -> str:
    if result.first_divergence_round is None:
        if result.len_a != result.len_b:
            return (
                f"no field divergence in the shared prefix, but lengths differ: "
                f"A has {result.len_a} rounds, B has {result.len_b} rounds "
                f"(one run ended earlier)"
            )
        return f"no divergence: {result.len_a} rounds match on every structural field"
    lines = [f"first divergence at round {result.first_divergence_round}:"]
    for field_name, (va, vb) in result.diverging_fields.items():
        lines.append(f"  {field_name}: A={va!r}  B={vb!r}")
    return "\n".join(lines)


if __name__ == "__main__":
    _rows = [
        RoundEvent(
            round_idx=i,
            slot=0,
            kv_len_before=i,
            path="cg_replay",
            cg_miss_reason="none",
            draft_tokens_n=15,
            accepted_n=15 if i != 3 else 2,
            reject_position=-1 if i != 3 else 2,
            bonus_token=1,
            mem_allocated=0,
            t_main_forward=0.0,
            t_draft=1.0,
            t_verify=1.0,
            t_commit=0.5,
            t_round=2.5 if i != 7 else 9000.0,
        )
        for i in range(10)
    ]
    _stats = compute_stats(_rows)
    assert _stats.acceptance_rate is not None
    assert len(_stats.outliers) == 1 and _stats.outliers[0]["round_idx"] == 7
    print(render_summary(_stats))
    _diff = diff_traces(_rows, _rows)
    assert _diff.first_divergence_round is None
    print("panel.py self-test OK")
