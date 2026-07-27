"""``bf determinism`` subcommand: ``show`` (current status) and ``env``
(export statements for load-time switches that must be set before the
process starts).

Follows the dispatcher contract described in ``bfdiag/cli.py``'s module
docstring: :func:`register` adds subparsers and attaches a
``func(args) -> int`` handler via ``set_defaults(func=...)``.

Known gap (spec-vs-code, documented per the task brief's rule 7 rather than
worked around by editing ``bfdiag/cli.py``, which is out of scope for this
change): ``bfdiag/cli.py``'s ``_candidate_module_names()`` only auto-discovers
``bfdiag.<subpackage>.cli`` modules (one per subpackage dir) plus, as a
special case, ``bfprobe``'s *flat* ``cli``/``*_cli`` modules. It does *not*
scan flat top-level modules of ``bfdiag`` itself, so this module -- named
``bfdiag/det_cli.py`` per the task brief, deliberately not a subpackage --
is not wired into ``bf`` yet. See
``notes/2026-07-27-bfdiag-determinism-and-sync.md`` for the one-line fix
this needs in ``bfdiag/cli.py`` (left for whoever owns that file). Until
then, use this module directly: ``python -m bfdiag.det_cli determinism show``
/ ``... determinism env --deterministic --force-sync`` both work standalone
(see ``_build_standalone_parser`` below, same pattern as
``bfdiag/trace/cli.py``'s own standalone runner), and :func:`register` is
fully testable by mounting it on a scratch parser (see
``tests/test_bfdiag_determinism.py``).
"""

from __future__ import annotations

import argparse
import json
import sys

from bfdiag import determinism


def _format_item(item: determinism.BundleItem) -> list[str]:
    lt = "load-time (cannot hot-switch)" if item.load_time else "hot-switchable"
    lines = [f"  [{item.status}] {item.name}  ({lt})"]
    lines.append(f"      does: {item.does}")
    lines.append(f"      cost: {item.perf_cost}")
    if item.detail:
        lines.append(f"      detail: {item.detail}")
    return lines


def _cmd_show(args: argparse.Namespace) -> int:
    report = determinism.apply(mutate=False)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"QSR_FORCE_SYNC:    {'on' if report.force_sync else 'off'}")
    if report.force_sync:
        print(
            "  WARNING: every trace mark point synchronizes CUDA -- perf numbers from "
            "this run are not valid performance data."
        )
    print(f"QSR_DETERMINISTIC: {'on' if report.deterministic else 'off'}")
    print()
    print("bundle:")
    for item in report.bundle:
        for line in _format_item(item):
            print(line)
    return 0


def _cmd_env(args: argparse.Namespace) -> int:
    lines: list[str] = []
    if args.deterministic:
        lines.append("export QSR_DETERMINISTIC=1")
        lines.append(f"export QSR_SEED={args.seed}")
        lines.append(f"export CUBLAS_WORKSPACE_CONFIG={determinism.CUBLAS_WORKSPACE_CONFIG_VALUE}")
        lines.append(
            f"export {determinism.AUTOTUNE_CACHE_ENV}={determinism.default_autotune_cache_dir()}"
        )
    if args.force_sync:
        lines.append("export QSR_FORCE_SYNC=1")
    if args.disable_cuda_graph:
        lines.extend(f"export {name}=0" for name in determinism.CUDA_GRAPH_ENV_VARS)

    if not lines:
        print(
            "# nothing requested -- pass --deterministic and/or --force-sync "
            "(optionally --disable-cuda-graph)",
            file=sys.stderr,
        )
        return 1

    print("# eval this before starting the process -- load-time vars only take effect")
    print("# if set before the engine is constructed:")
    for line in lines:
        print(line)
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Mount ``determinism show``/``determinism env`` onto ``bf``'s
    subparsers (see this module's docstring for the current auto-discovery
    gap)."""
    det_parser = subparsers.add_parser(
        "determinism",
        help="forced-sync / deterministic-mode switches (QSR_FORCE_SYNC, QSR_DETERMINISTIC)",
    )
    det_sub = det_parser.add_subparsers(dest="determinism_command", required=True)

    show_p = det_sub.add_parser("show", help="print current determinism-mode status")
    show_p.add_argument("--json", action="store_true", help="machine-readable output")
    show_p.set_defaults(func=_cmd_show)

    env_p = det_sub.add_parser(
        "env", help="print `export ...` lines to eval before starting the process"
    )
    env_p.add_argument("--deterministic", action="store_true", help="QSR_DETERMINISTIC=1 + friends")
    env_p.add_argument(
        "--force-sync", dest="force_sync", action="store_true", help="QSR_FORCE_SYNC=1"
    )
    env_p.add_argument(
        "--disable-cuda-graph",
        dest="disable_cuda_graph",
        action="store_true",
        help="also emit the three load-time QSR_*_CUDA_GRAPH=0 exports",
    )
    env_p.add_argument("--seed", type=int, default=determinism.DEFAULT_SEED, help="QSR_SEED value")
    env_p.set_defaults(func=_cmd_env)


def _build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bfdiag.det_cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    register(subparsers)
    return parser


if __name__ == "__main__":
    _parser = _build_standalone_parser()
    _args = _parser.parse_args()
    raise SystemExit(_args.func(_args))
