"""``bf probe routing`` subcommand: compare two engines' MoE routing captures.

``register(subparsers)`` is the contract the top-level ``bf`` dispatcher
calls (same auto-discovery convention ``bfdiag/cli.py`` uses, per
``notes/2026-07-27-probe-system-design-and-plan.md``'s "shared contract").
Uses the standard ``argparse`` ``set_defaults(func=...)`` pattern so the
dispatcher can just do ``args.func(args)`` after parsing.

Current input format: each of ``--ids-a``/``--ids-b``/``--weights-a``/
``--weights-b`` is a path to a ``.npy`` file holding one engine's captured
routing tensor. This is a placeholder until ``bfprobe/bus.py``'s storage
backend (``${QSR_BFDIAG_DIR:-<repo>/.bfdiag}/``) exists; once it does, this
CLI's data loading should be swapped to read a run id from that store
instead of raw ``.npy`` paths -- the comparison logic in
``bfprobe/routing_compare.py`` does not change either way.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from bfprobe.report import render_json, render_text
from bfprobe.routing_compare import compare_routing


def _load(path: str) -> Any:
    return np.load(path)


def _cmd_routing(args: argparse.Namespace) -> int:
    ids_a = _load(args.ids_a)
    ids_b = _load(args.ids_b)
    weights_a = _load(args.weights_a) if args.weights_a else None
    weights_b = _load(args.weights_b) if args.weights_b else None

    result = compare_routing(ids_a, ids_b, weights_a, weights_b)
    print(render_json(result) if args.json else render_text(result))
    return 0 if result.first_divergence is None else 1


def register(subparsers: argparse._SubParsersAction) -> None:
    """Mount ``probe routing`` onto ``bf``'s subparsers."""
    probe_parser = subparsers.add_parser(
        "probe", help="bfprobe: tiered, always-on extraction of engine internals"
    )
    probe_sub = probe_parser.add_subparsers(dest="probe_command", required=True)

    routing_p = probe_sub.add_parser(
        "routing",
        help="compare MoE routing (topk_ids/topk_weights) between two captured runs",
    )
    routing_p.add_argument("--ids-a", required=True, help="path to engine A's topk_ids .npy")
    routing_p.add_argument("--ids-b", required=True, help="path to engine B's topk_ids .npy")
    routing_p.add_argument("--weights-a", default=None, help="path to engine A's topk_weights .npy")
    routing_p.add_argument("--weights-b", default=None, help="path to engine B's topk_weights .npy")
    routing_p.add_argument("--json", action="store_true", help="machine-readable output")
    routing_p.set_defaults(func=_cmd_routing)


def _build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bfprobe.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register(subparsers)
    return parser


if __name__ == "__main__":
    _parser = _build_standalone_parser()
    _args = _parser.parse_args()
    raise SystemExit(_args.func(_args))
