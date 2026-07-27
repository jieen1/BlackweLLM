"""``bf probe scan`` -- offline T1 signature scan against a recorded baseline.

Auto-discovered by the (not-yet-built, owned by another agent) top-level
``bfprobe/cli.py`` dispatcher via ``register(subparsers) -> None``, following
the same convention as ``bfdiag/divergence/cli.py``. Until that dispatcher
exists, this subcommand is fully self-testable standalone::

    python -m bfprobe.scan_cli --signatures run.json --baseline baseline.json

Both inputs are plain JSON produced by this package's own dump helpers
(``bfprobe.signature.dump_json`` / ``bfprobe.baseline.save_baseline``) --
no GPU, model, or live engine needed to exercise this path, which is why it
is fully covered by tests/test_bfprobe_scan.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bfprobe.baseline import load_baseline
from bfprobe.scan import format_text_report, scan, to_json_dict
from bfprobe.signature import load_json


def register(subparsers: argparse._SubParsersAction) -> None:
    """Wire ``bf probe scan`` into the shared ``bfprobe`` CLI dispatcher."""
    parser = subparsers.add_parser(
        "scan", help="offline T1 signature scan against a recorded baseline", description=__doc__
    )
    _add_arguments(parser)
    parser.set_defaults(func=_run)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bf probe scan", description="offline T1 signature scan against a recorded baseline"
    )
    _add_arguments(parser)
    return parser


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--signatures",
        required=True,
        help="path to a T1 signature dump JSON file (bfprobe.signature.dump_json format)",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="path to a recorded baseline JSON file (bfprobe.baseline.save_baseline format)",
    )
    parser.add_argument("--json", action="store_true", help="emit a machine-readable JSON report")


def _run(args: argparse.Namespace) -> int:
    signatures_path = Path(args.signatures)
    baseline_path = Path(args.baseline)
    if not signatures_path.exists():
        print(f"no signature dump at {signatures_path!s}", file=sys.stderr)
        return 2
    if not baseline_path.exists():
        print(f"no baseline at {baseline_path!s}", file=sys.stderr)
        return 2

    read_result = load_json(signatures_path)
    if read_result.dropped:
        print(
            f"warning: signature dump reports {read_result.dropped} dropped records "
            "(ring overwrote them before they were drained) -- scan proceeds on what remains",
            file=sys.stderr,
        )
    baseline = load_baseline(baseline_path)

    report = scan(read_result.records, baseline)
    if args.json:
        print(json.dumps(to_json_dict(report), indent=2))
    else:
        print(format_text_report(report))
    return 1 if report.has_out_of_band else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
