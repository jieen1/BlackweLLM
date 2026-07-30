"""``bf ls`` / ``bf show`` / ``bf diff`` -- the CLI surface for run records.

Follows the dispatcher contract from ``bfdiag/cli.py``: :func:`register`
adds subparsers and attaches a ``func(args) -> int`` handler to each via
``set_defaults(func=...)``; the dispatcher calls ``args.func(args)``.
"""

from __future__ import annotations

import argparse
import json
import sys

from bfdiag.record.differ import diff_records, format_text, to_jsonable
from bfdiag.record.schema import RunRecord
from bfdiag.record.store import RunStore, default_store


def _resolve(store: RunStore, ref: str) -> RunRecord:
    run_id = store.resolve_run_id(ref)
    return store.load(run_id)


def _display_status(record: RunRecord) -> str:
    """Make an unfinalized record visibly non-comparable in ``bf ls``."""
    return "running" if record.finished_at is None else record.status


def _cmd_ls(args: argparse.Namespace) -> int:
    store = default_store()
    runs = store.list_runs(limit=args.n)
    if args.json:
        print(json.dumps([r.to_dict() for r in runs], indent=2, ensure_ascii=False))
        return 0
    if not runs:
        print("(no runs recorded yet)")
        return 0
    for r in runs:
        acceptance = r.metrics.get("acceptance_rate")
        suffix = f"  acceptance_rate={acceptance}" if acceptance is not None else ""
        print(f"{r.run_id}  {r.started_at}  {_display_status(r):7s}  {r.script}{suffix}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    store = default_store()
    try:
        record = _resolve(store, args.run_id)
    except (KeyError, ValueError) as exc:
        print(f"bf show: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))
        return 0
    print(f"run_id:      {record.run_id}")
    print(f"script:      {record.script}")
    print(f"argv:        {record.argv}")
    print(f"status:      {record.status}")
    print(f"started_at:  {record.started_at}")
    print(f"finished_at: {record.finished_at}")
    if record.error:
        print(f"error:       {record.error.strip().splitlines()[-1]}")
    print("fingerprint:")
    print(json.dumps(record.fingerprint.to_dict(), indent=2, ensure_ascii=False))
    print("metrics:")
    for name, value in sorted(record.metrics.items()):
        print(f"  {name}: {value}")
    print("artifacts:")
    for name, relpath in sorted(record.artifacts.items()):
        print(f"  {name}: {relpath}")
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    store = default_store()
    if args.a is None and args.b is None:
        runs = store.list_runs(limit=2)
        if len(runs) < 2:
            print("bf diff: fewer than two recorded runs; nothing to compare", file=sys.stderr)
            return 1
        run_b, run_a = runs[0], runs[1]  # list_runs() is newest-first
    else:
        if args.a is None or args.b is None:
            print("bf diff: pass both A and B, or neither to compare the last two", file=sys.stderr)
            return 1
        try:
            run_a = _resolve(store, args.a)
            run_b = _resolve(store, args.b)
        except (KeyError, ValueError) as exc:
            print(f"bf diff: {exc}", file=sys.stderr)
            return 1

    result = diff_records(run_a, run_b)
    if args.json:
        print(json.dumps(to_jsonable(result), indent=2, ensure_ascii=False))
    else:
        print(format_text(result))
    return 0 if result.comparable else 2


def register(subparsers) -> None:
    ls_parser = subparsers.add_parser("ls", help="list recorded runs, newest first")
    ls_parser.add_argument("-n", type=int, default=20, help="max runs to show")
    ls_parser.add_argument("--json", action="store_true", help="machine-readable output")
    ls_parser.set_defaults(func=_cmd_ls)

    show_parser = subparsers.add_parser("show", help="show one run record")
    show_parser.add_argument("run_id", help="run id or unique prefix")
    show_parser.add_argument("--json", action="store_true", help="machine-readable output")
    show_parser.set_defaults(func=_cmd_show)

    diff_parser = subparsers.add_parser("diff", help="diff two run records")
    diff_parser.add_argument(
        "a", nargs="?", default=None, help="run id/prefix (default: 2nd most recent)"
    )
    diff_parser.add_argument(
        "b", nargs="?", default=None, help="run id/prefix (default: most recent)"
    )
    diff_parser.add_argument("--json", action="store_true", help="machine-readable output")
    diff_parser.set_defaults(func=_cmd_diff)
