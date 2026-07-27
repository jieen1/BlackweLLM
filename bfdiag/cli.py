"""``bf`` -- the bfdiag command-line dispatcher.

This module owns subcommand discovery for the whole ``bfdiag`` platform.
Every subpackage of ``bfdiag`` (``bfdiag.record``, and eventually
``bfdiag.trace``, ``bfdiag.invariants``, ``bfdiag.oracle`` from the other
bfdiag work streams) may define a ``cli.py`` with a
``register(subparsers) -> None`` function; this dispatcher imports each one
it finds and lets it add its own subcommands.

Discovery is defensive on purpose: the other subpackages are being built in
parallel and may not exist yet in a given checkout, or may temporarily fail
to import while under construction. A missing or broken subpackage must
never break ``bf`` for the subcommands that *do* work -- it's silently
skipped (pass ``--debug`` to see why).

Contract for ``register(subparsers)``: it should call
``subparsers.add_parser(name)`` for each subcommand it owns, and attach a
handler with ``parser.set_defaults(func=handler)`` where
``handler(args: argparse.Namespace) -> int`` returns a process exit code.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from types import ModuleType

import bfdiag


def _iter_subpackages() -> list[str]:
    return sorted(name for _, name, is_pkg in pkgutil.iter_modules(bfdiag.__path__) if is_pkg)


def _discover_registrars(debug: bool) -> list[ModuleType]:
    registrars: list[ModuleType] = []
    for sub in _iter_subpackages():
        module_name = f"bfdiag.{sub}.cli"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - sibling subpackages may not exist yet
            if debug:
                print(f"bf --debug: skipping {module_name}: {exc!r}", file=sys.stderr)
            continue
        register = getattr(module, "register", None)
        if not callable(register):
            if debug:
                print(f"bf --debug: {module_name} has no register(subparsers)", file=sys.stderr)
            continue
        registrars.append(module)
    return registrars


def build_parser(debug: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bf", description="BlackForge diagnostics (bfdiag) command line"
    )
    parser.add_argument(
        "--debug", action="store_true", help="print subcommand-discovery diagnostics to stderr"
    )
    subparsers = parser.add_subparsers(dest="command")
    for module in _discover_registrars(debug):
        module.register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    debug = "--debug" in argv
    parser = build_parser(debug=debug)
    args = parser.parse_args(argv)

    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args) or 0


if __name__ == "__main__":
    sys.exit(main())
