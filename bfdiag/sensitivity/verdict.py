"""Turn a set of per-perturbation measurements into a verdict.

Pure functions: no torch, no GPU, no model. The whole point is that the
decision rule is testable and stated once, rather than re-eyeballed from
a terminal every time.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Measurement:
    """One measurement under one named perturbation."""

    perturbation: str
    metric: float
    output_hash: str
    allocator: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    stable: bool
    distinct_outputs: tuple[str, ...]
    distinct_metrics: tuple[float, ...]
    metric_spread: float
    # Perturbations that left allocator state identical to the reference
    # but still moved the result: those cannot be explained by layout and
    # point at genuine nondeterminism instead.
    unexplained: tuple[str, ...]
    summary: str


_ALLOC_KEYS = ("reserved_mib", "segments", "inactive_split_blocks", "alloc_mib")


def _alloc_key(m: Measurement) -> tuple:
    return tuple(m.allocator.get(k) for k in _ALLOC_KEYS)


def judge(measurements: list[Measurement], *, reference: str = "none") -> Verdict:
    """Stable == every perturbation produced byte-identical output.

    A run is *not* rescued by metrics matching: two different token
    sequences that happen to give the same acceptance rate are still a
    reproducibility failure, so the output hash is the primary signal.
    """
    if not measurements:
        raise ValueError("no measurements")

    hashes = tuple(sorted({m.output_hash for m in measurements}))
    metrics = tuple(sorted({round(m.metric, 9) for m in measurements}))
    spread = (max(metrics) - min(metrics)) if metrics else 0.0
    stable = len(hashes) == 1

    ref = next((m for m in measurements if m.perturbation == reference), measurements[0])
    ref_alloc = _alloc_key(ref)
    unexplained = tuple(
        m.perturbation
        for m in measurements
        if m.output_hash != ref.output_hash
        and any(v is not None for v in _alloc_key(m))
        and _alloc_key(m) == ref_alloc
    )

    if stable:
        summary = f"stable: all {len(measurements)} perturbations gave one output"
    else:
        summary = (
            f"SENSITIVE: {len(hashes)} distinct outputs across {len(measurements)} "
            f"perturbations, metric spread {spread:.6f}"
        )
        if unexplained:
            summary += (
                f"; {len(unexplained)} changed with allocator state UNCHANGED "
                f"({', '.join(unexplained)}) -- not explainable by layout"
            )
    return Verdict(
        stable=stable,
        distinct_outputs=hashes,
        distinct_metrics=metrics,
        metric_spread=spread,
        unexplained=unexplained,
        summary=summary,
    )


def format_table(measurements: list[Measurement], verdict: Verdict) -> str:
    lines = [
        f"{'perturbation':>14}  {'metric':>10}  {'output':>16}  "
        f"{'reserved_MiB':>12}  {'segs':>5}  {'inact_split':>11}",
        "-" * 78,
    ]
    for m in measurements:
        a = m.allocator
        lines.append(
            f"{m.perturbation:>14}  {m.metric:>10.6f}  {m.output_hash[:16]:>16}  "
            f"{a.get('reserved_mib', float('nan')):>12}  "
            f"{a.get('segments', '?'):>5}  {a.get('inactive_split_blocks', '?'):>11}"
        )
    lines += ["", verdict.summary]
    return "\n".join(lines)
