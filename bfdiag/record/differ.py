"""Compare two RunRecords: config diff, metric diff, and a comparability verdict.

The comparability check is the entire reason this tool exists. On
2026-07-27 two acceptance-rate measurements were compared -- 1.000 vs
0.687 -- and treated as evidence the backend had caught up. They
had used different prompts. Nobody noticed until a full day of investigation
had been spent chasing the wrong hypothesis (see
``notes/2026-07-27-bfdiag-run-records.md``). ``check_comparability`` below,
and the "NOT COMPARABLE" banner it produces, is the mechanism that makes that
specific mistake impossible to miss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bfdiag.record.schema import RunRecord

# Dotted paths (rooted at the full RunRecord dict, i.e. always under
# "fingerprint.") that must match between two runs for any metric comparison
# between them to mean anything. Names mirror the leaf names used in the
# bfdiag shared contract (e.g. "workload.prompt_hash").
DEFAULT_COMPARABLE_FIELDS: tuple[str, ...] = (
    "fingerprint.workload.contract",
    "fingerprint.workload.contract_version",
    "fingerprint.workload.workload_name",
    "fingerprint.workload.prompt_hash",
    "fingerprint.workload.generated_tokens",
    "fingerprint.workload.batch",
    "fingerprint.model.revision",
    "fingerprint.git.qwen-sm120-runtime.sha",
    "fingerprint.git.sparkinfer.sha",
    "fingerprint.workload.k",
    "fingerprint.workload.greedy",
    "fingerprint.workload.block_size",
    "fingerprint.workload.max_model_len",
    "fingerprint.workload.max_q_rows",
    "fingerprint.workload.cuda_graph_status",
    "fingerprint.workload.warm_only",
    # blocks_per_slot sets the sparkinfer decode workspace's max_pages, so it
    # can shift kernel tiling and float reduction order -- i.e. it moves
    # acceptance rate, not just memory. capacity (concurrent slots) changes
    # batching and the CUDA-Graph eligibility branch. Both were missing here
    # on 2026-07-27, which is why `bf diff` did NOT flag a warm-daemon run
    # (blocks_per_slot=4096) against a cold-start run (blocks_per_slot=130)
    # whose acceptance rates differed by 0.22.
    "fingerprint.workload.blocks_per_slot",
    "fingerprint.workload.capacity",
    # QSR_FORCE_SYNC (see bfdiag/determinism.py) breaks DFlash's async
    # pipeline on purpose -- any perf metric is meaningless if only one side
    # of a comparison had it on. Nested under `extra` (not a first-class
    # `fingerprint.determinism.*` path) because the determinism report is
    # carried in `Fingerprint.extra["determinism"]`, not a dedicated schema
    # field -- see bfdiag/record/fingerprint.py's capture_determinism().
    "fingerprint.extra.determinism.force_sync",
)

# SM/memory clocks are dynamic hardware observations, so they must not make
# otherwise identical runs formally non-comparable.  They do, however, make a
# throughput delta unsafe to attribute to code without a second controlled
# sample.  Keep this separate from the hard configuration gate above.
_FINGERPRINT_PREFIX = "fingerprint."


def _get_path(record_dict: dict[str, Any], path: str) -> Any:
    node: Any = record_dict
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _display_field(path: str) -> str:
    return path[len(_FINGERPRINT_PREFIX) :] if path.startswith(_FINGERPRINT_PREFIX) else path


def _short(value: Any, width: int = 12) -> str:
    text = "∅" if value is None else str(value)  # empty-set symbol for "missing"
    if len(text) > width:
        return text[:width] + "…"  # horizontal ellipsis
    return text


def _flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict to dotted-path -> scalar. Lists (argv, git shas
    tuple-like fields, layer_types, ...) are compared as opaque values, not
    descended into -- an index-by-index list diff is rarely what you want
    here.
    """
    out: dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(value, child_prefix))
    else:
        out[prefix] = node
    return out


@dataclass
class FieldDiff:
    path: str
    a: Any
    b: Any


@dataclass
class MetricDiff:
    name: str
    a: float | None
    b: float | None

    @property
    def delta_pct(self) -> float | None:
        if self.a is None or self.b is None or self.a == 0:
            return None
        return (self.b - self.a) / abs(self.a) * 100.0


@dataclass
class DiffResult:
    comparable: bool
    comparability_breaks: list[FieldDiff]
    config_diffs: list[FieldDiff]
    metric_diffs: list[MetricDiff]
    run_a: RunRecord
    run_b: RunRecord


def diff_configs(a: RunRecord, b: RunRecord) -> list[FieldDiff]:
    """Every fingerprint field that differs between the two runs, sorted by
    dotted path.
    """
    flat_a = _flatten(a.fingerprint.to_dict(), prefix="fingerprint")
    flat_b = _flatten(b.fingerprint.to_dict(), prefix="fingerprint")
    keys = sorted(set(flat_a) | set(flat_b))
    return [
        FieldDiff(path=key, a=flat_a.get(key), b=flat_b.get(key))
        for key in keys
        if flat_a.get(key) != flat_b.get(key)
    ]


def diff_metrics(a: RunRecord, b: RunRecord) -> list[MetricDiff]:
    names = sorted(set(a.metrics) | set(b.metrics))
    return [MetricDiff(name=n, a=a.metrics.get(n), b=b.metrics.get(n)) for n in names]


def check_comparability(
    a: RunRecord,
    b: RunRecord,
    fields: tuple[str, ...] = DEFAULT_COMPARABLE_FIELDS,
) -> list[FieldDiff]:
    """Which comparability-critical fields differ between the two runs.

    Empty list means the runs are comparable (as far as this field list
    knows -- see notes for why this list, not "everything", is the gate).
    """
    dict_a, dict_b = a.to_dict(), b.to_dict()
    breaks = []
    for path in fields:
        va, vb = _get_path(dict_a, path), _get_path(dict_b, path)
        if va != vb:
            breaks.append(FieldDiff(path=path, a=va, b=vb))
    return breaks


def diff_records(
    a: RunRecord,
    b: RunRecord,
    fields: tuple[str, ...] = DEFAULT_COMPARABLE_FIELDS,
) -> DiffResult:
    breaks = check_comparability(a, b, fields)
    return DiffResult(
        comparable=not breaks,
        comparability_breaks=breaks,
        config_diffs=diff_configs(a, b),
        metric_diffs=diff_metrics(a, b),
        run_a=a,
        run_b=b,
    )


def format_text(result: DiffResult) -> str:
    lines: list[str] = []

    for fd in result.comparability_breaks:
        lines.append(
            f"⚠ NOT COMPARABLE: {_display_field(fd.path)} differs "
            f"({_short(fd.a)} → {_short(fd.b)})"
        )
    if result.comparability_breaks:
        lines.append("")

    lines.append(f"A: {result.run_a.run_id}  {result.run_a.started_at}  {result.run_a.script}")
    lines.append(f"B: {result.run_b.run_id}  {result.run_b.started_at}  {result.run_b.script}")
    lines.append("")

    lines.append("== config diff ==")
    if not result.config_diffs:
        lines.append("  (identical)")
    for fd in result.config_diffs:
        lines.append(f"  {_display_field(fd.path)}: {fd.a!r} → {fd.b!r}")
    lines.append("")

    lines.append("== metrics diff ==")
    if not result.metric_diffs:
        lines.append("  (no metrics recorded)")
    for md in result.metric_diffs:
        a_str = "-" if md.a is None else str(md.a)
        b_str = "-" if md.b is None else str(md.b)
        pct = "n/a" if md.delta_pct is None else f"{md.delta_pct:+.1f}%"
        lines.append(f"  {md.name}: {a_str} → {b_str}  ({pct})")
    lines.append("")

    lines.append("== verdict ==")
    if result.comparable:
        lines.append("  OK: all comparability-critical fields match.")
    else:
        fields_str = ", ".join(_display_field(fd.path) for fd in result.comparability_breaks)
        lines.append(f"  NOT COMPARABLE: {fields_str} differ between the two runs.")

    return "\n".join(lines)


def to_jsonable(result: DiffResult) -> dict[str, Any]:
    return {
        "comparable": result.comparable,
        "comparability_breaks": [
            {"field": _display_field(fd.path), "a": fd.a, "b": fd.b}
            for fd in result.comparability_breaks
        ],
        "config_diffs": [
            {"field": _display_field(fd.path), "a": fd.a, "b": fd.b} for fd in result.config_diffs
        ],
        "metric_diffs": [
            {"name": md.name, "a": md.a, "b": md.b, "delta_pct": md.delta_pct}
            for md in result.metric_diffs
        ],
        "run_a": result.run_a.run_id,
        "run_b": result.run_b.run_id,
    }
