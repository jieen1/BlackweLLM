"""Single resolution point for the two local Qwen3.6-27B-NVFP4 checkpoints.

Before this module existed, ~22 scripts under ``scripts/`` each declared
their own ``MODEL_PATH = (...)`` string constant, hardcoding a full
``~/.cache/huggingface/hub/models--<org>--Qwen3.6-27B-NVFP4/snapshots/<hash>``
path inline. Every one of them independently pinned
``nvidia/Qwen3.6-27B-NVFP4`` (modelopt quantization format), because that
was the only checkpoint available when B1 started. The user has since
declared ``unsloth/Qwen3.6-27B-NVFP4`` (compressed-tensors ``mixed-precision``
format) the standard/served checkpoint (``3c2d0a8``, "Serve the standard
model... for the first time") -- but nothing forced the 22 hardcoded copies
to notice. That is the bug this module closes: one place names the standard
checkpoint, so there is exactly one place to point at a different snapshot,
and no script can quietly keep grading a checkpoint nobody ships anymore.

**The two checkpoints are different quantization *formats*, not just
different copies of the same weights** -- this module resolves paths only;
it does not paper over that difference:

- :func:`standard_checkpoint_path` (``unsloth/Qwen3.6-27B-NVFP4``) is
  compressed-tensors ``mixed-precision``: tensors named ``weight_packed``/
  ``weight_scale``/``weight_global_scale``/``input_global_scale``. Loaded by
  ``runtime/loading/compressed_tensors.py`` +
  ``runtime/model/compressed_tensors_linear.py``. This is the checkpoint
  every script measuring "the model we ship" should resolve.
- :func:`modelopt_checkpoint_path` (``nvidia/Qwen3.6-27B-NVFP4``) is NVIDIA
  ModelOpt: tensors named ``weight``/``weight_scale``/``weight_scale_2``/
  ``input_scale``. Loaded by ``runtime/loading/modelopt.py`` +
  ``runtime/model/modelopt_linear.py``. Kept as a separate, explicitly-named
  function (not deleted) for the handful of scripts whose entire point is
  exercising the modelopt adapter itself, or reproducing a historical
  measurement that was taken against this specific checkpoint -- see each
  such script's own module docstring for which reason applies. A script
  calling this function is declaring "I need modelopt specifically",
  greppable rather than an unexplained inline string literal.

Note ``weight_global_scale == 1 / weight_scale_2`` (a reciprocal, not merely
a rename) -- see ``runtime/model/compressed_tensors_linear.py``'s
``CompressedTensorsNVFP4Linear`` docstring. Nothing in this module needs to
know that; it is called out here only so nobody mistakes the two formats for
interchangeable because their paths now come from the same module.

Dynamic snapshot resolution, not a pinned hash
-----------------------------------------------
Each resolver function resolves ``<hub_cache>/models--<org>--Qwen3.6-27B-
NVFP4/snapshots/`` and picks its one entry, rather than hardcoding the
snapshot hash the way every migrated script used to. This is robust
exactly because a real HF hub cache normally holds one snapshot per
revision actually downloaded (verified against both checkpoints on this
machine, 2026-08-03: one entry each) -- re-downloading a checkpoint
rotates the hash without anyone having to hunt down 22 string literals
again. If a cache directory ever holds more than one snapshot (a second
revision pulled down alongside the first), resolution refuses to guess
which one is current and raises :class:`CheckpointNotFoundError` naming
both candidates -- pin an exact snapshot via the env var below rather than
have this module silently pick one.

Overriding
----------
``QSR_QWEN36_STANDARD_CHECKPOINT`` / ``QSR_QWEN36_MODELOPT_CHECKPOINT``
(``QSR_`` prefix matches this repo's existing env var convention -- see
``docs/diagnostics-guide.md``'s env var table and ``server/app.py``'s
``QSR_SERVER_MODEL_PATH``; ``BF_SPARKINFER_PATH`` is the one legacy holdout,
not the convention to follow) point resolution at an arbitrary local
directory instead -- a different snapshot, a checkpoint outside the HF hub
cache layout entirely, or a test fixture. Set once, honored everywhere,
instead of editing a string constant in whichever script you happen to be
running.

Functions, not eager module-level constants
--------------------------------------------
Deliberately not resolved at import time: ``import runtime.checkpoints``
must always succeed, including with an empty HF cache and in the
torch-free CI job, so nothing that merely imports this module (a test
enumerating its exports, a script that imports both functions but only
calls one) is forced to have both checkpoints on disk. Resolution --
and the possible :class:`CheckpointNotFoundError` -- happens only when a
caller actually asks for a path, same as every migrated script's own
``MODEL_PATH = ...`` line always did (it just used to fail later, inside
``AutoTokenizer.from_pretrained`` or ``load_qwen36_model``, with a less
specific error).

Neither constant silently falls back to the other checkpoint if the
requested one is missing -- a missing/misconfigured standard checkpoint
must fail loudly, not quietly grade modelopt and let the result be
mistaken for "the model we ship."
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repo IDs, not paths -- how the checkpoints are named everywhere else in
#: this codebase's docs/commit messages, and what turns into the HF hub
#: cache's ``models--<org>--<name>`` directory name below.
STANDARD_CHECKPOINT_REPO = "unsloth/Qwen3.6-27B-NVFP4"
MODELOPT_CHECKPOINT_REPO = "nvidia/Qwen3.6-27B-NVFP4"

#: Env vars that redirect resolution to an arbitrary local directory,
#: bypassing the HF hub cache lookup entirely.
QSR_QWEN36_STANDARD_CHECKPOINT = "QSR_QWEN36_STANDARD_CHECKPOINT"
QSR_QWEN36_MODELOPT_CHECKPOINT = "QSR_QWEN36_MODELOPT_CHECKPOINT"

#: Same default the rest of this codebase assumes (scripts have always
#: hardcoded this literally; HF_HOME/HUGGINGFACE_HUB_CACHE are not honored
#: here on purpose -- one clear override lever, the env var above, not two
#: that can disagree).
_DEFAULT_HF_HUB_CACHE = Path("~/.cache/huggingface/hub").expanduser()


class CheckpointNotFoundError(FileNotFoundError):
    """Raised when a checkpoint cannot be resolved, naming the env var that fixes it."""


def _repo_cache_dirname(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _resolve_snapshot_dir(repo_id: str, *, env_var: str) -> Path:
    """Resolve ``repo_id`` to a real local directory.

    ``env_var``, if set, is used verbatim (no existence check skipped --
    still validated below) so a misspelled override fails loudly rather
    than silently resolving the default checkpoint instead.
    """
    override = os.environ.get(env_var)
    if override:
        path = Path(override).expanduser()
        if not path.is_dir():
            raise CheckpointNotFoundError(
                f"{env_var}={override!r} does not exist or is not a directory. "
                f"Unset {env_var} to use the default HF hub cache location, or "
                "point it at a real local checkpoint directory."
            )
        return path

    cache_dir = _DEFAULT_HF_HUB_CACHE / _repo_cache_dirname(repo_id)
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        raise CheckpointNotFoundError(
            f"No local checkpoint found for {repo_id!r} -- expected a snapshot "
            f"under {snapshots_dir}. Download it first (e.g. `huggingface-cli "
            f"download {repo_id}`), or set {env_var}=/path/to/checkpoint to "
            "point at one already on disk elsewhere."
        )
    entries = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
    if len(entries) == 0:
        raise CheckpointNotFoundError(
            f"{snapshots_dir} exists but has no snapshot directories -- the "
            f"download for {repo_id!r} looks incomplete or corrupted. Re-download "
            f"it, or set {env_var}=/path/to/checkpoint to point at a working copy."
        )
    if len(entries) > 1:
        names = ", ".join(p.name for p in entries)
        raise CheckpointNotFoundError(
            f"{snapshots_dir} has {len(entries)} snapshot directories ({names}) -- "
            f"cannot pick one automatically. Set {env_var}=/path/to/one/of/them "
            "to disambiguate."
        )
    return entries[0]


def standard_checkpoint_path() -> str:
    """The standard/served checkpoint (unsloth, compressed-tensors mixed-precision).

    Every script measuring behavior of "the model we ship" should resolve
    this, not :func:`modelopt_checkpoint_path`.
    """
    return str(
        _resolve_snapshot_dir(STANDARD_CHECKPOINT_REPO, env_var=QSR_QWEN36_STANDARD_CHECKPOINT)
    )


def modelopt_checkpoint_path() -> str:
    """The modelopt reference checkpoint (nvidia), for scripts that need
    that format specifically -- see this module's docstring for why a
    script would want this instead of :func:`standard_checkpoint_path`.
    """
    return str(
        _resolve_snapshot_dir(MODELOPT_CHECKPOINT_REPO, env_var=QSR_QWEN36_MODELOPT_CHECKPOINT)
    )
