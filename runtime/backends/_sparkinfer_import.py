"""Single controlled entry point for resolving the ``sparkinfer`` package.

Problem this fixes (see
notes/2026-08-01-sparkinfer-patch-recovery-and-repro.md §7.2 and
docs/diagnostics-guide.md): ``BF_SPARKINFER_PATH`` was only honored by
``laguna_sparkinfer_attn.py`` and ``laguna_sparkinfer_moe.py``, each doing
its own ``sys.path.insert(0, os.environ.get("BF_SPARKINFER_PATH", ...))``
immediately before its own ``from sparkinfer... import``. But
``laguna.py``'s ``_patch_moe_sparkinfer`` did its OWN direct
``from sparkinfer.moe.fused_moe._impl import allocate_tp_moe_workspace_pool``
*before* importing anything from ``laguna_sparkinfer_moe`` -- and (confirmed
with an ``__import__`` traceback hook) that direct import is the actual
first touch of the top-level ``sparkinfer`` name in the real
``LagunaBackend.__init__`` startup path. Once a package name is bound in
``sys.modules``, submodule imports resolve through *that* package's cached
``__path__``, not through ``sys.path`` again -- so a later ``sys.path``
change cannot retroactively redirect an already-imported package. Net
effect: ``BF_SPARKINFER_PATH`` silently did nothing on the real Laguna
startup path, no error, no warning.

Fix: every Laguna call site that is about to touch ``sparkinfer`` for the
first time calls :func:`ensure_sparkinfer_path` here FIRST, before its own
``from sparkinfer...`` import. This module has no other responsibility, so
import order across call sites stops mattering -- whichever one runs first
resolves the path for everyone; a second call is a cheap no-op check.

If something *else* (outside every known Laguna call site -- e.g. a
production server preflight probe importing ``sparkinfer`` directly) beats
all of them to the import, this raises loudly instead of pretending the
override still took effect, because at that point it genuinely didn't.
"""

from __future__ import annotations

import os
import pathlib
import sys

# Matches the historical default in laguna_sparkinfer_attn.py, and the path
# the venv's own pip-editable install of sparkinfer already points at.
DEFAULT_SPARKINFER_PATH = "/home/bot/project/sparkinfer"


def requested_sparkinfer_path() -> str:
    """The checkout ``BF_SPARKINFER_PATH`` asks for, or the default."""
    return os.environ.get("BF_SPARKINFER_PATH", DEFAULT_SPARKINFER_PATH)


def ensure_sparkinfer_path() -> str:
    """Make the *next* ``import sparkinfer`` (from anywhere in the process)
    resolve under :func:`requested_sparkinfer_path`.

    Call this before your module's own first ``from sparkinfer...`` or
    ``import sparkinfer`` statement -- not after. Safe to call more than
    once (idempotent) and safe to call from multiple modules in any order,
    as long as each one calls it before its *own* first sparkinfer import.

    Raises ``RuntimeError`` if ``sparkinfer`` is already imported from a
    location other than the one requested -- at that point a ``sys.path``
    edit can no longer help, so failing loudly beats silently keeping the
    wrong checkout.
    """
    requested = requested_sparkinfer_path()
    already = sys.modules.get("sparkinfer")
    if already is not None:
        loaded_file = getattr(already, "__file__", None)
        loaded_root = (
            str(pathlib.Path(loaded_file).resolve().parent.parent) if loaded_file else "<unknown>"
        )
        if loaded_root != requested:
            raise RuntimeError(
                f"ensure_sparkinfer_path(): sparkinfer is ALREADY imported from "
                f"{loaded_root!r}, but {requested!r} was requested (via "
                "BF_SPARKINFER_PATH or the default). It is too late to "
                "redirect it -- sys.path changes cannot retroactively affect "
                "an already-imported module. Something imported `sparkinfer` "
                "without going through ensure_sparkinfer_path() first; see "
                "this module's docstring."
            )
        return requested  # already resolved to exactly what was requested

    if requested and requested not in sys.path:
        sys.path.insert(0, requested)
    return requested
