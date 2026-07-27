"""Single source of truth for BlackForge's forced-sync and deterministic-mode
switches. See ``notes/2026-07-27-bfdiag-determinism-and-sync.md`` for the
full design rationale and the GPU-verification TODOs.

Two orthogonal switches, both env-var driven and both off by default:

``QSR_FORCE_SYNC`` (0/1, default 0)
    Inserts ``torch.cuda.synchronize()`` at every ``bfdiag.trace.ring`` mark
    point (``begin_round``/``mark``/``finish_round``). Does *not* add any new
    integration point in ``runtime/`` -- the flight recorder's mark() calls
    already sit at the exact phase boundaries (post-verify, post-commit) a
    debugging sync wants, because ``dflash_round`` already calls them there
    for timing. Reusing them means zero new runtime call sites.

    What it buys you:
      1. Any pending async CUDA error from that phase's kernels raises at the
         mark() call, not several kernels (or phases) later -- much tighter
         fault localization when chasing a crash/NaN.
      2. On the CPU (``time.perf_counter``) timing fallback -- i.e. no CUDA,
         or ``Timeline`` is forced off it -- marks would otherwise only
         capture kernel-*launch* time, not completion; the sync makes them
         reflect real completion.

    What it does NOT buy you: on the normal CUDA-available path,
    ``bfdiag.trace.timing.Timeline`` already timestamps marks via
    ``torch.cuda.Event`` (see that module's docstring), which correctly
    captures GPU-side completion time whether or not the host synchronizes
    first -- ``elapsed_time()`` between two events is accurate either way.
    So on that path ``QSR_FORCE_SYNC`` does not change whether recorded phase
    *durations* are correct (they already are); it changes whether the host
    blocks until they're actually true, which is what gives you (1) above,
    at the direct cost of (2) below.

    Cost: forcing every phase boundary to block until the GPU has actually
    caught up serializes exactly the async pipeline DFlash depends on for its
    throughput. Round/phase timings and any tok/s-style numbers taken while
    this is on are **not valid performance data** -- they measure a
    fully-synchronous variant of the engine, not the real one. ``apply()``
    warns loudly when this is on, and it gets folded into the run's
    fingerprint (``fingerprint.extra.determinism.force_sync``) specifically
    so ``bf diff`` can flag two runs as NOT COMPARABLE when only this differs
    (see ``bfdiag/record/differ.py``'s ``DEFAULT_COMPARABLE_FIELDS``).

``QSR_DETERMINISTIC`` (0/1, default 0)
    A bundle of independent knobs, each verified against real code before
    being included here (see the module-level ``_apply_*`` functions below
    and the per-item ``does``/``perf_cost``/``load_time`` metadata returned
    by :func:`apply`). None of them are invented -- every one either already
    exists in this repo (grep-verified) or is a standard, documented
    PyTorch/CUDA mechanism.

Design note on ``apply()`` vs. a separate "status" function: there isn't
one. ``apply(mutate=False)`` *is* the read-only status/report call --
:func:`bfdiag.record.fingerprint.capture_determinism` and ``bf determinism
show`` both call it that way. This matters because ``fingerprint.capture()``
must never have side effects (it's called defensively, possibly mid-run, to
snapshot "what mode is this process in" -- it must not itself reseed RNGs or
flip a global torch flag as a side effect of taking a snapshot). Giving
``apply()`` a ``mutate`` switch, rather than a second function that
duplicates the same per-item logic, keeps the "what does this environment
actually look like right now" logic in exactly one place.

Every ``torch.cuda.*`` call in this module is written for real and guarded
so it only ever executes when ``torch`` is installed, CUDA is available, and
the relevant switch is on. This module was developed and tested entirely on
a CUDA-unavailable sandbox: every test in
``tests/test_bfdiag_determinism.py`` monkeypatches ``torch.cuda.is_available``
/``torch.cuda.synchronize`` rather than letting them run for real, and no
test ever sets ``QSR_FORCE_SYNC``/``QSR_DETERMINISTIC`` in a way that would
reach the real functions unpatched.
"""

from __future__ import annotations

import os
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - exercised on boxes with no torch
    torch = None  # type: ignore[assignment]

try:
    import numpy
except ImportError:  # pragma: no cover - numpy is behind the `cuda` extra
    numpy = None  # type: ignore[assignment]


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"


def _repo_root() -> Path:
    # bfdiag/determinism.py -> bfdiag -> <repo root>
    return Path(__file__).resolve().parent.parent


def _cuda_available() -> bool:
    """``torch.cuda.is_available()`` is a real GPU-driver query; every
    caller of this function that could run during this task's own tests
    monkeypatches it. Never called eagerly at import time or on any path
    that doesn't already require the corresponding switch to be on.
    """
    return torch is not None and torch.cuda.is_available()


# --------------------------------------------------------------------------
# QSR_FORCE_SYNC
# --------------------------------------------------------------------------

FORCE_SYNC: bool = _env_flag("QSR_FORCE_SYNC")

if FORCE_SYNC:
    warnings.warn(
        "QSR_FORCE_SYNC=1: every bfdiag trace mark point now calls "
        "torch.cuda.synchronize(). This breaks DFlash's async pipeline on "
        "purpose (see bfdiag/determinism.py's module docstring) -- phase "
        "timings, round timings, and any tok/s-style throughput number "
        "measured in this run are NOT valid performance data. It is "
        "recorded in this run's fingerprint (determinism.force_sync) so "
        "`bf diff` will flag comparisons against it as NOT COMPARABLE.",
        stacklevel=2,
    )


def maybe_sync() -> bool:
    """Call ``torch.cuda.synchronize()`` iff ``QSR_FORCE_SYNC`` is on *and*
    CUDA is actually available; a no-op (returns ``False``) otherwise --
    this is the "auto-degrade to no-op when CUDA is unavailable" contract
    ``bfdiag.trace.ring``'s hot path depends on. Returns whether it actually
    synchronized, so tests can assert on it and reports can distinguish
    "synced" from "skipped, no CUDA".
    """
    if not FORCE_SYNC:
        return False
    if not _cuda_available():
        return False
    torch.cuda.synchronize()  # the one call this module exists to make
    return True


# --------------------------------------------------------------------------
# QSR_DETERMINISTIC bundle
# --------------------------------------------------------------------------

DETERMINISTIC: bool = _env_flag("QSR_DETERMINISTIC")

SEED_ENV = "QSR_SEED"
DEFAULT_SEED = 0

CUBLAS_WORKSPACE_CONFIG_VALUE = ":4096:8"

AUTOTUNE_CACHE_ENV = "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"
AUTOTUNE_CACHE_DIRNAME = ".autotune_cache"

# Real env var, verified in runtime/backends/laguna_sparkinfer_moe.py (grepped,
# not guessed): that module does
#   os.environ.setdefault("SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
# at import time, unconditionally forcing this bugfix (sparkinfer commit
# 989723d, "fix(moe): make dynamic MoE physical-row assignment deterministic")
# on by default. bfdiag deliberately never sets or overrides this env var --
# it only observes and reports the current value, per the task brief: this
# is a correctness fix, not a knob, and its default must not move.
SPARKINFER_MOE_DETERMINISTIC_ENV = "SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT"

# Real env vars, verified in runtime/backends/laguna.py:359 and
# runtime/backends/laguna_dflash.py:171,387 (grepped, not guessed) -- each is
# read exactly once inside __init__/_init_cuda_graph. bfdiag/daemon/
# provider.py's LOAD_TIME_ENV_VARS independently documents the same three as
# "setting this on an already-loaded hot daemon has NO effect".
CUDA_GRAPH_ENV_VARS: tuple[str, ...] = (
    "QSR_DECODE_CUDA_GRAPH",
    "QSR_DFLASH_CUDA_GRAPH",
    "QSR_VERIFY_CUDA_GRAPH",
)


@dataclass
class BundleItem:
    """One checklist entry of the ``QSR_DETERMINISTIC`` bundle.

    ``status`` vocabulary (deliberately not an enum -- this is a
    human-readable report, not another dispatch table):
      - ``not_enabled``: bundle is off, or this specific item wasn't
        requested (e.g. the opt-in ``cuda_graph_disable`` item).
      - ``applied``: the mutation happened this call (or was already true,
        observed directly rather than just trusted -- see module docstring).
      - ``not_yet_applied``: bundle is on but this call was ``mutate=False``
        (a read-only status snapshot) and the underlying state doesn't show
        it's actually in effect yet.
      - ``skipped_no_cuda`` / ``skipped_no_torch``: honestly reporting a
        skip, never silently pretending the item applied (see acceptance
        criterion in the task brief this module implements).
      - ``observed_only``: this item is never mutated by bfdiag at all
        (sparkinfer's MoE determinism flag) -- always just reports the
        current real value.
      - ``error``: the mutation was attempted and raised; the exception is
        captured in ``detail`` rather than propagating (fingerprint capture
        must never raise).
    """

    name: str
    does: str
    perf_cost: str
    load_time: bool
    status: str = "not_enabled"
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "does": self.does,
            "perf_cost": self.perf_cost,
            "load_time": self.load_time,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class DeterminismReport:
    force_sync: bool
    deterministic: bool
    bundle: list[BundleItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "force_sync": self.force_sync,
            "deterministic": self.deterministic,
            "bundle": {item.name: item.to_dict() for item in self.bundle},
        }


def default_autotune_cache_dir() -> Path:
    return _repo_root() / AUTOTUNE_CACHE_DIRNAME


def _cuda_graph_env_summary() -> str:
    return " ".join(f"{name}={os.environ.get(name, '1(default)')}" for name in CUDA_GRAPH_ENV_VARS)


# -- item 1: torch.use_deterministic_algorithms + CUBLAS_WORKSPACE_CONFIG --


def _apply_torch_deterministic_algorithms(is_on: bool, *, mutate: bool) -> BundleItem:
    item = BundleItem(
        name="torch_deterministic_algorithms",
        does=(
            "torch.use_deterministic_algorithms(True) + "
            f"CUBLAS_WORKSPACE_CONFIG={CUBLAS_WORKSPACE_CONFIG_VALUE!r}"
        ),
        perf_cost=(
            "can be significantly slower (forces deterministic kernel variants); "
            "some ops have no deterministic implementation and will raise instead "
            "of silently being nondeterministic"
        ),
        load_time=True,
    )
    if torch is None:
        item.status = "skipped_no_torch"
        return item

    if is_on and mutate:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG_VALUE)
        torch.use_deterministic_algorithms(True)

    currently_enabled = bool(torch.are_deterministic_algorithms_enabled())
    cublas_cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    if not is_on:
        item.status = "not_enabled"
    elif currently_enabled:
        item.status = "applied"
    else:
        item.status = "not_yet_applied"

    item.detail = (
        f"torch.are_deterministic_algorithms_enabled()={currently_enabled}; "
        f"CUBLAS_WORKSPACE_CONFIG={cublas_cfg!r} -- this env var only takes effect "
        "if set before the first CUDA/cuBLAS context init in the process; setting "
        "it later (this call included, if cuBLAS already initialized) has no effect"
    )
    return item


# -- item 2: seed torch / numpy / python random -----------------------------


def _apply_seed_all(is_on: bool, seed: int, *, mutate: bool) -> BundleItem:
    item = BundleItem(
        name="seed_all",
        does=f"random.seed({seed}) + numpy.random.seed({seed}) + torch.manual_seed({seed})",
        perf_cost="negligible",
        load_time=False,
    )
    if not is_on:
        item.status = "not_enabled"
        item.detail = "not requested"
        return item
    if not mutate:
        item.status = "not_yet_applied"
        item.detail = (
            "read-only status snapshot -- RNG state isn't introspectable, so this "
            "item can only report what it *would* do until mutate=True actually seeds"
        )
        return item

    applied: list[str] = ["random"]
    random.seed(seed)
    skipped: list[str] = []
    if numpy is not None:
        numpy.random.seed(seed)
        applied.append("numpy")
    else:
        skipped.append("numpy")
    if torch is not None:
        torch.manual_seed(seed)
        applied.append("torch")
    else:
        skipped.append("torch")

    item.status = "applied"
    detail = f"seeded: {', '.join(applied)}"
    if skipped:
        detail += f"; skipped (not installed): {', '.join(skipped)}"
    item.detail = detail
    return item


# -- item 3: sparkinfer MoE deterministic_output (observation only) --------


def _observe_sparkinfer_moe_deterministic() -> BundleItem:
    value = os.environ.get(SPARKINFER_MOE_DETERMINISTIC_ENV)
    item = BundleItem(
        name="sparkinfer_moe_deterministic_output",
        does=(
            f"read-only observation of {SPARKINFER_MOE_DETERMINISTIC_ENV} -- bfdiag never "
            "sets or overrides it. runtime/backends/laguna_sparkinfer_moe.py forces it to "
            "'1' via os.environ.setdefault(...) at import time (sparkinfer commit 989723d, "
            "a real numerical-correctness fix for dynamic MoE physical-row assignment, not "
            "a perf knob) -- this item exists purely to fingerprint which mode a run was in"
        ),
        perf_cost="n/a (observation only; not controlled by this module)",
        load_time=True,  # read by sparkinfer at MoE kernel/binding construction (model load)
        status="observed_only",
        detail=(
            f"{SPARKINFER_MOE_DETERMINISTIC_ENV}={value!r}"
            if value is not None
            else (
                f"{SPARKINFER_MOE_DETERMINISTIC_ENV} unset in this process "
                "(laguna_sparkinfer_moe.py's own setdefault will force it to "
                "'1' once that module is imported)"
            )
        ),
    )
    return item


# -- item 4: fixed autotune cache directory ---------------------------------


def _apply_autotune_cache(is_on: bool, *, mutate: bool) -> BundleItem:
    item = BundleItem(
        name="autotune_cache",
        does=f"{AUTOTUNE_CACHE_ENV}=<repo>/{AUTOTUNE_CACHE_DIRNAME} (already the convention "
        "used throughout benchmarks/*.py)",
        perf_cost=(
            "first process to hit a given (shape, kernel) pair still pays the autotune "
            "search once; every later run reads the cached choice instead of re-searching, "
            "so kernel selection stops varying run-to-run"
        ),
        load_time=True,
    )
    if is_on and mutate:
        target = str(default_autotune_cache_dir())
        os.environ.setdefault(AUTOTUNE_CACHE_ENV, target)
        try:
            Path(os.environ[AUTOTUNE_CACHE_ENV]).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            item.status = "error"
            item.detail = repr(exc)
            return item

    current = os.environ.get(AUTOTUNE_CACHE_ENV)
    if not is_on:
        item.status = "not_enabled"
    elif current:
        item.status = "applied"
    else:
        item.status = "not_yet_applied"
    item.detail = (
        f"{AUTOTUNE_CACHE_ENV}={current!r}"
        if current
        else f"{AUTOTUNE_CACHE_ENV} not set. Exact read timing inside vllm/flashinfer isn't "
        "verified from this sandbox (no vllm installed here, see notes) -- treat as load-time "
        "like every other convention in benchmarks/*.py that sets it before importing vllm"
    )
    return item


# -- item 5 (optional): disable CUDA Graph capture --------------------------


def _apply_cuda_graph_disable(requested: bool, *, mutate: bool) -> BundleItem:
    item = BundleItem(
        name="cuda_graph_disable",
        does=(
            "opt-in: QSR_DECODE_CUDA_GRAPH=0, QSR_DFLASH_CUDA_GRAPH=0, "
            "QSR_VERIFY_CUDA_GRAPH=0 -- forces eager execution, removing captured-graph "
            "state as a variable while debugging"
        ),
        perf_cost="eager decode/verify is substantially slower (every kernel re-launched "
        "each round instead of one graph replay)",
        load_time=True,
    )
    if not requested:
        item.status = "not_enabled"
        item.detail = "opt-in only (pass disable_cuda_graph=True); current env: " + (
            _cuda_graph_env_summary()
        )
        return item
    if not _cuda_available():
        item.status = "skipped_no_cuda"
        item.detail = _cuda_graph_env_summary()
        return item
    if mutate:
        for name in CUDA_GRAPH_ENV_VARS:
            os.environ[name] = "0"
        item.status = "applied"
        item.detail = (
            "load-time only: these are read exactly once inside LagunaBackend.__init__ / "
            "DFlashEngine.__init__ (runtime/backends/laguna.py:359, "
            "runtime/backends/laguna_dflash.py:171,387) -- this only has any effect if the "
            "engine has not been constructed yet in this process; an already-loaded/hot "
            "engine ignores it and needs a process restart. " + _cuda_graph_env_summary()
        )
    else:
        item.status = "not_yet_applied"
        item.detail = _cuda_graph_env_summary()
    return item


# --------------------------------------------------------------------------
# apply(): the one entry point. mutate=True actually flips the switches
# (idempotent -- see per-item logic above, every mutation either checks
# real current state first or uses setdefault); mutate=False is a read-only
# status snapshot with the exact same per-item report shape, used by
# bfdiag.record.fingerprint.capture_determinism() and `bf determinism show`.
# --------------------------------------------------------------------------


def apply(
    *,
    deterministic: bool | None = None,
    force_sync: bool | None = None,
    seed: int | None = None,
    disable_cuda_graph: bool = False,
    mutate: bool = True,
) -> DeterminismReport:
    """Apply (or, with ``mutate=False``, just report on) the
    ``QSR_DETERMINISTIC`` bundle. Idempotent: calling this repeatedly with
    the same arguments converges to (and then just re-reports) the same
    state -- no item's mutation is unsafe to repeat.

    ``deterministic``/``force_sync`` default to the module-level flags
    derived from ``QSR_DETERMINISTIC``/``QSR_FORCE_SYNC`` at import time;
    pass explicitly to override (mainly for tests and ``bf determinism
    env``'s preview). ``seed`` defaults to ``$QSR_SEED`` (or
    :data:`DEFAULT_SEED` if unset). ``disable_cuda_graph`` is the one
    bundle item that's opt-in even when the rest of the bundle is on (see
    module docstring).
    """
    is_deterministic = DETERMINISTIC if deterministic is None else deterministic
    is_force_sync = FORCE_SYNC if force_sync is None else force_sync
    seed_value = seed if seed is not None else int(os.environ.get(SEED_ENV, str(DEFAULT_SEED)))

    bundle = [
        _apply_torch_deterministic_algorithms(is_deterministic, mutate=mutate),
        _apply_seed_all(is_deterministic, seed_value, mutate=mutate),
        _observe_sparkinfer_moe_deterministic(),
        _apply_autotune_cache(is_deterministic, mutate=mutate),
        _apply_cuda_graph_disable(is_deterministic and disable_cuda_graph, mutate=mutate),
    ]
    return DeterminismReport(
        force_sync=is_force_sync, deterministic=is_deterministic, bundle=bundle
    )


if __name__ == "__main__":
    # Self-test: CPU-only, never touches real CUDA (torch.cuda.is_available()
    # is only reached if QSR_FORCE_SYNC/QSR_DETERMINISTIC were manually set in
    # the environment running this, which the sandbox this was built in never
    # does).
    report = apply(mutate=False)
    assert report.force_sync is False
    assert report.deterministic is False
    names = {item.name for item in report.bundle}
    assert names == {
        "torch_deterministic_algorithms",
        "seed_all",
        "sparkinfer_moe_deterministic_output",
        "autotune_cache",
        "cuda_graph_disable",
    }
    idempotent_report = apply(mutate=False)
    assert report.to_dict() == idempotent_report.to_dict()
    print("determinism.py self-test OK:", report.to_dict())
