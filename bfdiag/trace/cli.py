"""``bf trace`` subcommand: ``show`` (vital-signs panel for one run) and
``diff`` (first-divergence report between two runs).

``register(subparsers)`` is the contract ``bfdiag/cli.py``'s auto-discovery
dispatcher calls (see the module docstring in that file -- not owned by this
module/agent). Uses the standard ``argparse`` ``set_defaults(func=...)``
pattern so the dispatcher can just do ``args.func(args)`` after parsing,
without needing to know this subcommand exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bfdiag.trace import panel
from bfdiag.trace.dump import read_trace, trace_path_for_run
from bfdiag.trace.ring import BFDIAG_DIR


def _trace_path(args: argparse.Namespace, run_id: str) -> Path:
    bfdiag_dir = Path(args.bfdiag_dir) if args.bfdiag_dir else BFDIAG_DIR
    return trace_path_for_run(bfdiag_dir, run_id)


def _cmd_show(args: argparse.Namespace) -> int:
    path = _trace_path(args, args.run_id)
    if not path.exists():
        print(f"no trace found for run {args.run_id!r} (looked at {path})", file=sys.stderr)
        return 1
    rows = read_trace(path)
    stats = panel.compute_stats(rows)
    if args.json:
        print(panel.render_json(rows, stats))
        return 0
    print(panel.render_round_table(rows, limit=args.limit))
    print()
    print(panel.render_summary(stats))
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    path_a = _trace_path(args, args.run_id_a)
    path_b = _trace_path(args, args.run_id_b)
    for label, path in (("A", path_a), ("B", path_b)):
        if not path.exists():
            print(f"no trace found for run {label} (looked at {path})", file=sys.stderr)
            return 1
    rows_a = read_trace(path_a)
    rows_b = read_trace(path_b)
    result = panel.diff_traces(rows_a, rows_b)
    if args.json:
        import json

        print(json.dumps(result.to_dict(), indent=2))
        return 0
    print(panel.render_diff(result))
    return 0 if result.first_divergence_round is None else 1


def register(subparsers: argparse._SubParsersAction) -> None:
    """Mount ``trace show``/``trace diff`` onto ``bf``'s subparsers."""
    trace_parser = subparsers.add_parser(
        "trace", help="flight-recorder trace inspection (show / diff)"
    )
    trace_sub = trace_parser.add_subparsers(dest="trace_command", required=True)

    _bfdiag_dir_help = "override the bfdiag storage root ($QSR_BFDIAG_DIR or <repo>/.bfdiag)"

    show_p = trace_sub.add_parser("show", help="vital-signs panel for one run's trace")
    show_p.add_argument("run_id")
    show_p.add_argument("--json", action="store_true", help="machine-readable output")
    show_p.add_argument(
        "--limit", type=int, default=50, help="rows to show in the per-round table (0 = all)"
    )
    show_p.add_argument("--bfdiag-dir", dest="bfdiag_dir", default=None, help=_bfdiag_dir_help)
    show_p.set_defaults(func=_cmd_show)

    diff_p = trace_sub.add_parser("diff", help="first-divergence report between two runs")
    diff_p.add_argument("run_id_a", metavar="RUN_A")
    diff_p.add_argument("run_id_b", metavar="RUN_B")
    diff_p.add_argument("--json", action="store_true", help="machine-readable output")
    diff_p.add_argument("--bfdiag-dir", dest="bfdiag_dir", default=None, help=_bfdiag_dir_help)
    diff_p.set_defaults(func=_cmd_diff)


def _build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bfdiag.trace.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register(subparsers)
    return parser


if __name__ == "__main__":
    _parser = _build_standalone_parser()
    _args = _parser.parse_args()
    raise SystemExit(_args.func(_args))
