"""BF_SPARKINFER_PATH bootstrap shim.

Root cause (found 2026-08-01, see
notes/2026-08-01-sparkinfer-patch-recovery-and-repro.md): BF_SPARKINFER_PATH
is only honored by two files -- runtime/backends/laguna_sparkinfer_attn.py
and laguna_sparkinfer_moe.py -- each doing
`sys.path.insert(0, os.environ.get("BF_SPARKINFER_PATH", ...))` right before
their own `from sparkinfer... import`. But
`runtime/backends/laguna.py`'s `_patch_moe_sparkinfer` does its OWN direct
`from sparkinfer.moe.fused_moe._impl import allocate_tp_moe_workspace_pool`
*before* importing anything from laguna_sparkinfer_moe.py -- and that direct
import is the actual first touch of the `sparkinfer` name in the whole
daemon process (confirmed with an `__import__` traceback hook: the very
first `import sparkinfer.*` frame is
`runtime/backends/laguna.py:593 _patch_moe_sparkinfer`, called from
`LagunaBackend.__init__` during `provider.load()` -- both
laguna_sparkinfer_{attn,moe}.py are imported later than that). Once Python
caches a top-level module name in sys.modules, no later sys.path change can
redirect it. Net effect: BF_SPARKINFER_PATH is silently ignored -- no error,
no warning, just the pip-editable-installed default
(/home/bot/project/sparkinfer) every time, regardless of what the env var
says.

This shim works around it without touching `runtime/backends/laguna.py`
(a fix there is a one-line reorder -- move the
`from runtime.backends.laguna_sparkinfer_moe import ...` above the direct
`from sparkinfer.moe...` import, or route the workspace-pool import through
laguna_sparkinfer_moe.py instead of importing sparkinfer directly -- but
that file is live production code with other work in flight; this
directory is scoped to notes/Makefile/scripts, so the fix lives here as a
loader-level workaround instead).

`sitecustomize` is auto-imported by Python's `site` module at interpreter
startup, before user code (including `python -m bfdiag.daemon.server`)
runs. Putting this directory first on PYTHONPATH guarantees the sys.path
insertion below happens before ANY `import sparkinfer` anywhere in the
process -- closing the race outright rather than hoping a particular
project file happens to import first.

Usage -- put this directory *ahead of* the repo root on PYTHONPATH when
starting a daemon that should load a non-default SparkInfer checkout:

    PYTHONPATH=scripts/bf_sparkinfer_bootstrap:/path/to/repo \\
    BF_SPARKINFER_PATH=/home/bot/project/sparkinfer-ctrl \\
    scripts/bf-t0.sh daemon start --provider laguna

(`scripts/bf-t0.sh` does not add this directory automatically -- it is only
needed when overriding BF_SPARKINFER_PATH away from the default checkout;
see its own docstring.)

Still verify what actually loaded afterwards with
`scripts/verify_sparkinfer_load.py` (via `bf exec`, or `make
verify-sparkinfer`) -- this shim makes BF_SPARKINFER_PATH reliable, it does
not make checking unnecessary.
"""

import os
import sys

_path = os.environ.get("BF_SPARKINFER_PATH", "")
if _path and _path not in sys.path:
    sys.path.insert(0, _path)
