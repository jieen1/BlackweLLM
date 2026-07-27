"""Capture the environment fingerprint that makes two runs comparable.

Every function here is defensive: git missing, sparkinfer/vllm checkouts
absent, no GPU, no nvidia-smi -- none of it raises. Missing data becomes
``None`` and the caller gets a best-effort ``Fingerprint`` back. This is the
whole point: ``bf diff`` needs to work on a laptop with no GPU just as well
as on the workstation, so a run recorded on a GPU-less machine is still
useful evidence (e.g. "this config change alone doesn't explain the delta").

GPU info is read via a single read-only ``nvidia-smi --query-gpu=...`` call
(never a CUDA context, never a tensor) -- see AGENTS.md / the bfdiag task
brief for why this is the one GPU-adjacent command this module is allowed
to run.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys
from importlib import metadata as _importlib_metadata
from pathlib import Path
from typing import Any

from bfdiag import determinism as _determinism
from bfdiag.record.schema import (
    Fingerprint,
    GitRepoInfo,
    GpuInfo,
    ModelInfo,
    PythonInfo,
    WorkloadInfo,
)

# Any environment variable whose name starts with one of these prefixes is
# considered part of the fingerprint (per the shared bfdiag contract).
ENV_PREFIXES: tuple[str, ...] = ("QSR_", "VLLM_", "CUDA_", "TORCH_", "NVIDIA_", "FLASHINFER_")

# Default sibling-repo locations, overridable by environment variable so CI
# or another developer's machine can point elsewhere.
_REPO_ENV_OVERRIDES: dict[str, str] = {
    "qwen-sm120-runtime": "QSR_REPO_QWEN_SM120_RUNTIME",
    "sparkinfer": "QSR_REPO_SPARKINFER",
    "vllm": "QSR_REPO_VLLM",
}

_DEFAULT_REPO_PATHS: dict[str, str] = {
    "qwen-sm120-runtime": "/home/bot/project/qwen-sm120-runtime",
    "sparkinfer": "/home/bot/project/sparkinfer",
    "vllm": "/home/bot/vllm",
}

_GPU_CORE_FIELDS = (
    "name,driver_version,clocks.sm,clocks.mem,power.limit,memory.total,persistence_mode"
)


def _run(cmd: list[str], cwd: str | None = None, timeout: float = 5.0) -> str | None:
    """Run a subprocess and return stripped stdout, or None on any failure.

    Never raises: missing binary, nonzero exit, and timeout are all treated
    as "data unavailable" rather than an error worth propagating.
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _repo_paths(overrides: dict[str, str] | None = None) -> dict[str, str]:
    paths = dict(_DEFAULT_REPO_PATHS)
    for name, env_var in _REPO_ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            paths[name] = value
    if overrides:
        paths.update(overrides)
    return paths


def capture_git_repo(path: str | None) -> GitRepoInfo:
    """Fingerprint one repo: HEAD sha, dirty flag, branch. All-None if
    ``path`` isn't a directory or isn't a git repo (no exception either way).
    """
    if not path or not Path(path).is_dir():
        return GitRepoInfo()
    sha = _run(["git", "rev-parse", "HEAD"], cwd=path)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=path)
    status = _run(["git", "status", "--porcelain"], cwd=path)
    dirty = None if status is None else bool(status)
    return GitRepoInfo(sha=sha, dirty=dirty, branch=branch)


def capture_git(repo_paths: dict[str, str] | None = None) -> dict[str, GitRepoInfo]:
    """Fingerprint the three repos this runtime depends on.

    Names match the shared bfdiag contract: qwen-sm120-runtime, sparkinfer,
    vllm. Each is independently overridable via ``QSR_REPO_<NAME>`` or the
    ``repo_paths`` argument, and independently tolerant of not existing.
    """
    return {name: capture_git_repo(path) for name, path in _repo_paths(repo_paths).items()}


def capture_env() -> dict[str, str]:
    """All environment variables matching the shared bfdiag prefixes."""
    return {
        key: value for key, value in sorted(os.environ.items()) if key.startswith(ENV_PREFIXES)
    }


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    parsed = _parse_float(value)
    return int(parsed) if parsed is not None else None


def capture_gpu() -> GpuInfo:
    """Read-only ``nvidia-smi --query-gpu=...``; all-None if unavailable.

    Split into two queries because ``cuda_version`` is not a valid
    ``--query-gpu`` field on every driver version -- if it fails, the other
    (widely supported) fields should still populate.
    """
    if shutil.which("nvidia-smi") is None:
        return GpuInfo()

    core = _run(["nvidia-smi", f"--query-gpu={_GPU_CORE_FIELDS}", "--format=csv,noheader,nounits"])
    if not core:
        return GpuInfo()
    first_line = core.splitlines()[0]
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) < 7:
        return GpuInfo()
    name, driver, sm_clock, mem_clock, power_limit, total_mem, persistence = parts[:7]

    cuda_version = _run(["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"])

    return GpuInfo(
        name=name or None,
        driver=driver or None,
        cuda=cuda_version or None,
        sm_clock_mhz=_parse_int(sm_clock),
        mem_clock_mhz=_parse_int(mem_clock),
        power_limit_w=_parse_float(power_limit),
        total_mem_mib=_parse_int(total_mem),
        persistence_mode=persistence or None,
    )


def _pkg_version(dist_name: str) -> str | None:
    """Installed package version without importing the package itself --
    safe to call even for heavy/CUDA-touching packages like torch/vllm.
    """
    try:
        return _importlib_metadata.version(dist_name)
    except _importlib_metadata.PackageNotFoundError:
        return None


def capture_determinism() -> dict[str, Any]:
    """Read-only snapshot of ``QSR_FORCE_SYNC``/``QSR_DETERMINISTIC`` mode --
    "what mode was this run actually in", for the fingerprint. Never raises
    (same defensive contract as every other ``capture_*`` here) and never
    mutates process state: uses ``determinism.apply(mutate=False)``, so
    taking a fingerprint never itself reseeds RNGs or flips torch's
    deterministic-algorithms flag as a side effect.

    Nested under ``Fingerprint.extra["determinism"]`` rather than a new
    top-level ``Fingerprint`` field, specifically to avoid touching
    ``bfdiag/record/schema.py`` (out of scope for this change) -- see
    ``notes/2026-07-27-bfdiag-determinism-and-sync.md``. This is also why
    ``bfdiag.record.differ.DEFAULT_COMPARABLE_FIELDS`` references it as
    ``fingerprint.extra.determinism.force_sync`` rather than
    ``fingerprint.determinism.force_sync``.
    """
    try:
        return _determinism.apply(mutate=False).to_dict()
    except Exception:  # noqa: BLE001 - a fingerprint capture must never raise
        return {}


def capture_python() -> PythonInfo:
    return PythonInfo(
        version=sys.version.split()[0],
        torch=_pkg_version("torch"),
        vllm=_pkg_version("vllm"),
        transformers=_pkg_version("transformers"),
    )


def _filtered_kwargs(
    dc_type: type, data: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split ``data`` into (known-field kwargs, leftover) for ``dc_type``.

    Unknown keys never raise -- they're preserved in ``leftover`` so callers
    can stash them under ``fingerprint.extra`` instead of silently dropping
    them.
    """
    data = dict(data or {})
    names = {f.name for f in dataclasses.fields(dc_type)}
    known = {k: v for k, v in data.items() if k in names}
    leftover = {k: v for k, v in data.items() if k not in names}
    return known, leftover


def capture(
    *,
    model: dict[str, Any] | None = None,
    workload: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    repo_paths: dict[str, str] | None = None,
) -> Fingerprint:
    """Assemble a full Fingerprint. Never raises.

    ``model``/``workload`` are caller-supplied (they describe the specific
    experiment, not the machine) and are matched against the
    ``ModelInfo``/``WorkloadInfo`` schema fields; anything unrecognized is
    kept under ``extra`` rather than dropped or raised on.
    """
    model_known, model_extra = _filtered_kwargs(ModelInfo, model)
    workload_known, workload_extra = _filtered_kwargs(WorkloadInfo, workload)

    merged_extra: dict[str, Any] = dict(extra or {})
    if model_extra:
        merged_extra["model_extra"] = model_extra
    if workload_extra:
        merged_extra["workload_extra"] = workload_extra
    # Always recorded (not just when QSR_DETERMINISTIC/QSR_FORCE_SYNC are on)
    # so `bf diff` can catch "these two runs differ in determinism mode"
    # regardless of which side turned anything on. Computed last so it wins
    # over an (unlikely) caller-supplied "determinism" key in `extra`.
    merged_extra["determinism"] = capture_determinism()

    return Fingerprint(
        git=capture_git(repo_paths),
        env=capture_env(),
        gpu=capture_gpu(),
        python=capture_python(),
        model=ModelInfo(**model_known),
        workload=WorkloadInfo(**workload_known),
        extra=merged_extra,
    )
