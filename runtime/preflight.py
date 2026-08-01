"""Startup preflight validation — run this before any model weights load.

This module answers one question: "is the environment sane enough to start
loading a checkpoint," and answers it *before* the expensive, slow-to-fail
part of startup (streaming safetensors into a 6000-series GPU) gets a chance
to run for a while and then die confusingly. See docs/roadmap.md Track 0
(T0-6, "依赖版本合同统一") and Track D (D3, "启动前置检查").

Not wired into `server/app.py` here on purpose — that module is being
rewritten in parallel by another agent. The intended call site is the very
first thing the server startup path does, before
`runtime.model_loading.load_laguna_model(...)`:

    from runtime.preflight import run_preflight

    report = run_preflight(checkpoint_path)
    for check in report.checks:
        logger.info("preflight: %s", check) if check.passed else logger.warning(...)
    if not report.ok:
        raise SystemExit(1)  # or translate into a FastAPI startup failure

Design rules (same import discipline as `bfdiag/`, because this module, like
`bfdiag/`, may end up imported at module level by `runtime/` call sites
before any GPU work happens):

  - Pure stdlib at import time. `torch` and `sparkinfer` are imported lazily,
    inside the probe functions that need them, so this module imports (and
    unit-tests) cleanly on a machine with neither installed.
  - No printing, no logging, no `sys.exit`. Every check is a `CheckResult`;
    the caller decides how to render, log, or escalate it.
  - Every check function accepts its probe as an optional argument. Passing
    one in (as the test suite does) skips the real hardware/library probe
    entirely — that is what makes this testable without a GPU.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
from pathlib import Path
from typing import Literal

Severity = Literal["fatal", "warning"]

# --- Version contract (docs/roadmap.md T0-6) --------------------------------
# Kept in sync with pyproject.toml's `cuda` extra. See that file's comments
# for why these are floors/ceilings rather than exact pins, and for the exact
# build verified working as of 2026-08-01.
MIN_TORCH_VERSION: tuple[int, int, int] = (2, 12, 0)
MAX_TORCH_VERSION: tuple[int, int, int] = (2, 14, 0)  # exclusive
VERIFIED_TORCH_VERSION = (
    "2.13.0a0+gitcf30153"  # ~/.venvs/vllm, from-source build of /home/bot/pytorch-build
)
VERIFIED_CUDA_RUNTIME_VERSION = "13.3"  # torch.version.cuda in the verified environment
MIN_CUDA_RUNTIME_VERSION: tuple[int, int] = (13, 0)
MIN_DRIVER_VERSION: tuple[int, ...] = (
    550,
)  # conservative Blackwell floor; verified driver is 610.47
VERIFIED_DRIVER_VERSION = "610.47"
REQUIRED_COMPUTE_CAPABILITY: tuple[int, int] = (12, 0)  # SM120 only — docs/architecture.md §1
MIN_SPARKINFER_VERSION: tuple[int, int, int] = (1, 0, 0)
VERIFIED_SPARKINFER_VERSION = "1.0.1"  # editable install of /home/bot/project/sparkinfer

# Real, unsharded (TP=1) full-attention shape SparkInfer's paged-attention
# adapter runs in production — see runtime/backends/laguna_sparkinfer_attn.py
# module docstring. Used to functionally probe whether the installed
# SparkInfer's analytic-decode gate accepts our shape or only the upstream
# TP=2 shape (24 query heads / 4 KV heads) it ships pre-tuned for.
PRODUCTION_FULL_ATTENTION_NUM_Q_HEADS = 48
PRODUCTION_FULL_ATTENTION_NUM_KV_HEADS = 8
PRODUCTION_HEAD_DIM = 128
PRODUCTION_PAGE_SIZE = 64  # current production block_size; see docs/sparkinfer-upstream-handoff.md


@dataclasses.dataclass(frozen=True)
class CheckResult:
    """One preflight check's outcome. Structured, not rendered.

    `severity` distinguishes "must not start" (fatal — e.g. wrong GPU
    architecture, which is a hardware-contract violation per
    docs/architecture.md §1) from "will start, but you should know"
    (warning — e.g. an unpatched SparkInfer that silently falls back to a
    slower kernel). The caller decides what to do with either; this module
    never exits or blocks on its own.
    """

    name: str
    passed: bool
    severity: Severity
    actual: str
    expected: str
    remediation: str = ""

    def __str__(self) -> str:  # pragma: no cover - convenience for callers
        status = "PASS" if self.passed else self.severity.upper()
        base = f"[{status}] {self.name}: expected {self.expected!r}, got {self.actual!r}"
        return f"{base} — {self.remediation}" if self.remediation and not self.passed else base


@dataclasses.dataclass(frozen=True)
class PreflightReport:
    """The full set of preflight results, plus the derived pass/fail verdict."""

    checks: tuple[CheckResult, ...]

    @property
    def fatal_failures(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed and c.severity == "fatal")

    @property
    def warnings(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed and c.severity == "warning")

    @property
    def ok(self) -> bool:
        """True when nothing fatal failed. Warnings never block startup."""
        return len(self.fatal_failures) == 0


def _parse_version_prefix(version: str) -> tuple[int, ...] | None:
    """Parse the leading `N.N.N` (or shorter) numeric prefix of a version
    string, ignoring any pre-release/local suffix (e.g. `2.13.0a0+gitabc`
    parses to `(2, 13, 0)`). Returns None if there is no numeric prefix.
    """
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", version.strip())
    if not match:
        return None
    return tuple(int(g) for g in match.groups() if g is not None)


def _version_tuple_in_range(
    actual: tuple[int, ...] | None,
    minimum: tuple[int, ...],
    maximum: tuple[int, ...] | None = None,
) -> bool:
    if actual is None:
        return False
    if actual < minimum:
        return False
    if maximum is not None and actual >= maximum:
        return False
    return True


# --- GPU / torch / driver probe ---------------------------------------------


@dataclasses.dataclass(frozen=True)
class GpuProbe:
    """Snapshot of GPU/CUDA/torch facts, gathered once.

    Build one via `probe_gpu()` in production. Tests construct one by hand
    (or monkeypatch nothing at all — this dataclass has no behavior) to
    simulate any hardware/software combination without a GPU.
    """

    torch_importable: bool
    torch_version: str | None
    cuda_available: bool
    device_name: str | None
    compute_capability: tuple[int, int] | None
    cuda_runtime_version: str | None
    driver_version: str | None
    error: str | None = None


def _probe_driver_version() -> str | None:
    """Best-effort `nvidia-smi` query. Returns None if unavailable for any
    reason (no GPU, no nvidia-smi on PATH, sandboxed environment, etc.) —
    the driver check degrades to "unknown" rather than raising.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except Exception:
        return None
    line = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    return line or None


def probe_gpu() -> GpuProbe:
    """Real hardware/software probe. Imports torch lazily; safe to call on a
    CPU-only box (returns a GpuProbe describing the absence, not a raise).
    """
    try:
        import torch
    except Exception as exc:
        return GpuProbe(
            torch_importable=False,
            torch_version=None,
            cuda_available=False,
            device_name=None,
            compute_capability=None,
            cuda_runtime_version=None,
            driver_version=_probe_driver_version(),
            error=f"torch is not importable: {exc}",
        )

    torch_version = getattr(torch, "__version__", None)
    cuda_runtime_version = getattr(getattr(torch, "version", None), "cuda", None)

    if not torch.cuda.is_available():
        return GpuProbe(
            torch_importable=True,
            torch_version=torch_version,
            cuda_available=False,
            device_name=None,
            compute_capability=None,
            cuda_runtime_version=cuda_runtime_version,
            driver_version=_probe_driver_version(),
            error="torch.cuda.is_available() is False",
        )

    try:
        device_name = torch.cuda.get_device_name(0)
        compute_capability = tuple(torch.cuda.get_device_capability(0))
    except Exception as exc:
        return GpuProbe(
            torch_importable=True,
            torch_version=torch_version,
            cuda_available=True,
            device_name=None,
            compute_capability=None,
            cuda_runtime_version=cuda_runtime_version,
            driver_version=_probe_driver_version(),
            error=f"could not query CUDA device 0: {exc}",
        )

    return GpuProbe(
        torch_importable=True,
        torch_version=torch_version,
        cuda_available=True,
        device_name=device_name,
        compute_capability=compute_capability,  # type: ignore[arg-type]
        cuda_runtime_version=cuda_runtime_version,
        driver_version=_probe_driver_version(),
        error=None,
    )


def check_gpu_present(gpu: GpuProbe | None = None) -> CheckResult:
    gpu = gpu if gpu is not None else probe_gpu()
    if gpu.cuda_available and gpu.compute_capability is not None:
        cc = gpu.compute_capability
        return CheckResult(
            name="gpu_present",
            passed=True,
            severity="fatal",
            actual=f"{gpu.device_name} (cc {cc[0]}.{cc[1]})",
            expected="an available CUDA device",
        )
    return CheckResult(
        name="gpu_present",
        passed=False,
        severity="fatal",
        actual=gpu.error or "no CUDA device available",
        expected="an available CUDA device",
        remediation=(
            "This runtime only runs on the SM120 workstation it targets — check "
            "`nvidia-smi` on the host, and that torch was built/installed with "
            f"CUDA support (torch_importable={gpu.torch_importable}, "
            f"torch_version={gpu.torch_version})."
        ),
    )


def check_compute_capability(gpu: GpuProbe | None = None) -> CheckResult:
    gpu = gpu if gpu is not None else probe_gpu()
    cc = gpu.compute_capability
    expected = f"{REQUIRED_COMPUTE_CAPABILITY[0]}.{REQUIRED_COMPUTE_CAPABILITY[1]} (SM120)"
    if cc == REQUIRED_COMPUTE_CAPABILITY:
        return CheckResult(
            name="compute_capability",
            passed=True,
            severity="fatal",
            actual=f"{cc[0]}.{cc[1]}",
            expected=expected,
        )
    actual = f"{cc[0]}.{cc[1]}" if cc is not None else "unknown (no CUDA device)"
    return CheckResult(
        name="compute_capability",
        passed=False,
        severity="fatal",
        actual=actual,
        expected=expected,
        remediation=(
            "This runtime is hard-scoped to NVIDIA Blackwell SM120 (CC 12.0) — "
            "see docs/architecture.md §1. Refusing to start on any other "
            "compute capability is intentional: the kernels, NVFP4/FP8 "
            "quantization paths, and CUDA Graph capture logic are all "
            "SM120-specific and are not merely slow elsewhere, they may "
            "silently miscompute."
        ),
    )


def check_cuda_driver_version(gpu: GpuProbe | None = None) -> CheckResult:
    gpu = gpu if gpu is not None else probe_gpu()
    expected = f">= {MIN_DRIVER_VERSION[0]} (verified: {VERIFIED_DRIVER_VERSION})"
    if gpu.driver_version is None:
        return CheckResult(
            name="cuda_driver_version",
            passed=False,
            severity="warning",
            actual="could not determine driver version (nvidia-smi unavailable or failed)",
            expected=expected,
            remediation="Ensure `nvidia-smi` is on PATH; this check is advisory only.",
        )
    parsed = _parse_version_prefix(gpu.driver_version)
    ok = parsed is not None and parsed[:1] >= MIN_DRIVER_VERSION
    return CheckResult(
        name="cuda_driver_version",
        passed=ok,
        severity="warning",
        actual=gpu.driver_version,
        expected=expected,
        remediation=""
        if ok
        else "Update the NVIDIA driver; Blackwell SM120 needs a reasonably recent one.",
    )


def check_cuda_runtime_version(gpu: GpuProbe | None = None) -> CheckResult:
    gpu = gpu if gpu is not None else probe_gpu()
    min_major, min_minor = MIN_CUDA_RUNTIME_VERSION
    expected = f">= {min_major}.{min_minor} (verified: {VERIFIED_CUDA_RUNTIME_VERSION})"
    if gpu.cuda_runtime_version is None:
        return CheckResult(
            name="cuda_runtime_version",
            passed=False,
            severity="warning",
            actual="torch reports no CUDA runtime (torch.version.cuda is None)",
            expected=expected,
            remediation="Install a CUDA-enabled torch build; see pyproject.toml's `cuda` extra.",
        )
    parsed = _parse_version_prefix(gpu.cuda_runtime_version)
    ok = parsed is not None and parsed[:2] >= MIN_CUDA_RUNTIME_VERSION
    remediation = "" if ok else "torch's linked CUDA runtime is older than verified; recheck SM120."
    return CheckResult(
        name="cuda_runtime_version",
        passed=ok,
        severity="warning",
        actual=gpu.cuda_runtime_version,
        expected=expected,
        remediation=remediation,
    )


def check_torch_version(gpu: GpuProbe | None = None) -> CheckResult:
    gpu = gpu if gpu is not None else probe_gpu()
    expected = (
        f">= {'.'.join(map(str, MIN_TORCH_VERSION))}, "
        f"< {'.'.join(map(str, MAX_TORCH_VERSION))} "
        f"(verified: {VERIFIED_TORCH_VERSION})"
    )
    if not gpu.torch_importable or gpu.torch_version is None:
        return CheckResult(
            name="torch_version",
            passed=False,
            severity="fatal",
            actual="torch is not importable",
            expected=expected,
            remediation="Install torch per pyproject.toml's `cuda` extra: `pip install -e .[cuda]`",
        )
    parsed = _parse_version_prefix(gpu.torch_version)
    ok = _version_tuple_in_range(parsed, MIN_TORCH_VERSION, MAX_TORCH_VERSION)
    return CheckResult(
        name="torch_version",
        passed=ok,
        severity="fatal",
        actual=gpu.torch_version,
        expected=expected,
        remediation=(
            ""
            if ok
            else (
                "SparkInfer itself hard-requires torch>=2.12 "
                "(/home/bot/project/sparkinfer/pyproject.toml). Rebuild/reinstall "
                "the torch used by this venv; see pyproject.toml's `cuda` extra "
                "comment for why an exact PyPI pin is not possible here."
            )
        ),
    )


# --- SparkInfer probe --------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SparkInferProbe:
    """Snapshot of SparkInfer's install state and the functional gating
    behavior that governs whether our production shape gets SparkInfer's
    fast warp-specialized analytic decode kernel or its generic fallback.

    See docs/sparkinfer-upstream-handoff.md for the full story.
    """

    importable: bool
    version: str | None
    module_path: str | None
    gate_found: bool
    gate_accepts_production_shape: bool | None  # None = could not be evaluated
    error: str | None = None


def probe_sparkinfer(*, gpu: GpuProbe | None = None) -> SparkInferProbe:
    """Real probe. Imports sparkinfer (and torch) lazily. The functional
    gate check additionally requires a real CUDA device — on a CPU-only box
    it degrades to `gate_accepts_production_shape=None` rather than raising.
    """
    try:
        import sparkinfer
    except Exception as exc:
        return SparkInferProbe(
            importable=False,
            version=None,
            module_path=None,
            gate_found=False,
            gate_accepts_production_shape=None,
            error=f"sparkinfer is not importable: {exc}",
        )

    version = getattr(sparkinfer, "__version__", None)
    if version is None:
        try:
            from importlib import metadata

            version = metadata.version("sparkinfer")
        except Exception:
            version = None
    module_path = getattr(sparkinfer, "__file__", None)

    try:
        from sparkinfer.attention.paged import planner as _planner
    except Exception as exc:
        return SparkInferProbe(
            importable=True,
            version=version,
            module_path=module_path,
            gate_found=False,
            gate_accepts_production_shape=None,
            error=f"sparkinfer.attention.paged.planner is not importable: {exc}",
        )

    gate = getattr(_planner, "_is_laguna_fp8_gqa6_analytic_decode_graph", None)
    if gate is None:
        return SparkInferProbe(
            importable=True,
            version=version,
            module_path=module_path,
            gate_found=False,
            gate_accepts_production_shape=None,
            error=(
                "gate function _is_laguna_fp8_gqa6_analytic_decode_graph not found "
                "— SparkInfer's internal API has changed since "
                "docs/sparkinfer-upstream-handoff.md was written; re-derive the probe"
            ),
        )

    gpu = gpu if gpu is not None else probe_gpu()
    if not gpu.cuda_available:
        return SparkInferProbe(
            importable=True,
            version=version,
            module_path=module_path,
            gate_found=True,
            gate_accepts_production_shape=None,
            error="no CUDA device available; cannot evaluate the gate's device-capability check",
        )

    try:
        import torch

        accepts = bool(
            gate(
                device=torch.device("cuda", torch.cuda.current_device()),
                q_dtype=torch.bfloat16,
                kv_dtype=torch.float8_e4m3fn,
                num_q_heads=PRODUCTION_FULL_ATTENTION_NUM_Q_HEADS,
                num_kv_heads=PRODUCTION_FULL_ATTENTION_NUM_KV_HEADS,
                head_dim_qk=PRODUCTION_HEAD_DIM,
                head_dim_vo=PRODUCTION_HEAD_DIM,
                page_size=PRODUCTION_PAGE_SIZE,
                batch=1,
                window_left=-1,
            )
        )
    except TypeError as exc:
        return SparkInferProbe(
            importable=True,
            version=version,
            module_path=module_path,
            gate_found=True,
            gate_accepts_production_shape=None,
            error=f"gate signature changed, cannot evaluate: {exc}",
        )

    return SparkInferProbe(
        importable=True,
        version=version,
        module_path=module_path,
        gate_found=True,
        gate_accepts_production_shape=accepts,
        error=None,
    )


def check_sparkinfer_contract(sparkinfer_probe: SparkInferProbe | None = None) -> CheckResult:
    probe = sparkinfer_probe if sparkinfer_probe is not None else probe_sparkinfer()
    expected = (
        f">= {'.'.join(map(str, MIN_SPARKINFER_VERSION))} (verified: {VERIFIED_SPARKINFER_VERSION})"
    )
    if not probe.importable:
        return CheckResult(
            name="sparkinfer_contract",
            passed=False,
            severity="fatal",
            actual=probe.error or "sparkinfer is not importable",
            expected=expected,
            remediation=(
                "SparkInfer is a private local dependency, not on PyPI: "
                "`pip install -e /home/bot/project/sparkinfer` (after activating "
                "the venv where this project's `cuda` extra is installed)."
            ),
        )
    parsed = _parse_version_prefix(probe.version) if probe.version else None
    ok = parsed is not None and parsed >= MIN_SPARKINFER_VERSION
    return CheckResult(
        name="sparkinfer_contract",
        passed=ok,
        severity="fatal",
        actual=f"{probe.version} at {probe.module_path}",
        expected=expected,
        remediation=""
        if ok
        else "Update the SparkInfer checkout to a version meeting the contract.",
    )


def check_sparkinfer_analytic_decode_gate(
    sparkinfer_probe: SparkInferProbe | None = None,
) -> CheckResult:
    """The T0-5 check: does the installed SparkInfer unlock the fast
    warp-specialized analytic decode path for our real production shape
    (48 query heads / 8 KV heads, gqa_group_size=6), or only for the
    upstream TP=2 shape (24/4) it ships pre-tuned for?

    This is a performance check, not a correctness one — the runtime falls
    back to SparkInfer's generic paged-attention kernel either way and
    produces correct output. It is `severity="warning"` for exactly that
    reason. But it must be a *loud* warning: see
    docs/sparkinfer-upstream-handoff.md — the 2026-07-31 throughput numbers
    (353-401 tok/s at 4K, 353-368 tok/s at 64K) were measured against a
    locally patched SparkInfer that is not upstream and, as of this writing,
    is not even present in the checked-out /home/bot/project/sparkinfer
    working tree. Silently running on the unpatched checkout and still
    expecting those numbers is exactly the failure mode this check exists to
    prevent.
    """
    probe = sparkinfer_probe if sparkinfer_probe is not None else probe_sparkinfer()
    expected = "accepts production shape (48 q heads / 8 kv heads, gqa_group_size=6)"

    if not probe.importable:
        return CheckResult(
            name="sparkinfer_analytic_decode_gate",
            passed=False,
            severity="warning",
            actual="sparkinfer not importable; cannot evaluate",
            expected=expected,
            remediation="Fix the sparkinfer_contract check first.",
        )
    if probe.gate_accepts_production_shape is None:
        return CheckResult(
            name="sparkinfer_analytic_decode_gate",
            passed=False,
            severity="warning",
            actual=probe.error or "could not evaluate the gate",
            expected=expected,
            remediation=(
                "Could not determine gate status (no CUDA device at preflight "
                "time, or SparkInfer's internal API changed). Re-run on the GPU; "
                "if it still cannot evaluate, see docs/sparkinfer-upstream-handoff.md."
            ),
        )
    if probe.gate_accepts_production_shape:
        return CheckResult(
            name="sparkinfer_analytic_decode_gate",
            passed=True,
            severity="warning",
            actual="accepts production shape",
            expected=expected,
        )
    return CheckResult(
        name="sparkinfer_analytic_decode_gate",
        passed=False,
        severity="warning",
        actual=(
            "rejects production shape (48/8) — falls back to SparkInfer's "
            "generic paged-attention kernel; only the upstream TP=2 shape "
            "(24 q heads / 4 kv heads) is unlocked on this checkout"
        ),
        expected=expected,
        remediation=(
            "This SparkInfer checkout is unpatched. The 2026-07-31 throughput "
            "numbers (353-401 tok/s @4K, 353-368 tok/s @64K) do NOT apply here "
            "— expect a measurable regression versus those figures. See "
            "docs/sparkinfer-upstream-handoff.md for the exact gating change, "
            "why it is safe, and how to verify it. This repo does not modify "
            "SparkInfer source directly; the patch must be applied by the "
            "SparkInfer team or reproduced locally by someone with authority "
            "to edit /home/bot/project/sparkinfer."
        ),
    )


# --- Checkpoint probe --------------------------------------------------------


def check_checkpoint_path(checkpoint_path: str | Path) -> CheckResult:
    path = Path(checkpoint_path)
    if path.is_dir():
        return CheckResult(
            name="checkpoint_path",
            passed=True,
            severity="fatal",
            actual=str(path),
            expected="an existing directory",
        )
    remediation = f"Check the model path passed at startup; `{path}` must exist before load."
    return CheckResult(
        name="checkpoint_path",
        passed=False,
        severity="fatal",
        actual=f"{path} does not exist or is not a directory",
        expected="an existing directory",
        remediation=remediation,
    )


def check_checkpoint_config(checkpoint_path: str | Path) -> CheckResult:
    config_path = Path(checkpoint_path) / "config.json"
    if not config_path.is_file():
        return CheckResult(
            name="checkpoint_config",
            passed=False,
            severity="fatal",
            actual=f"{config_path} is missing",
            expected="config.json present in the checkpoint directory",
            remediation="Verify the checkpoint directory contains a HuggingFace-style config.json.",
        )
    try:
        parsed = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return CheckResult(
            name="checkpoint_config",
            passed=False,
            severity="fatal",
            actual=f"cannot read/parse {config_path}: {exc}",
            expected="valid JSON",
            remediation="Re-download or repair the checkpoint; config.json is corrupt/unreadable.",
        )
    if not isinstance(parsed, dict):
        remediation = "config.json is malformed; expected a top-level object with model_type."
        return CheckResult(
            name="checkpoint_config",
            passed=False,
            severity="fatal",
            actual=f"{config_path} does not contain a JSON object",
            expected="a JSON object",
            remediation=remediation,
        )
    actual = f"model_type={parsed.get('model_type')!r} arch={parsed.get('architectures')!r}"
    return CheckResult(
        name="checkpoint_config",
        passed=True,
        severity="fatal",
        actual=actual,
        expected="valid JSON with model_type/architectures",
    )


# --- Entry point --------------------------------------------------------------


def run_preflight(
    checkpoint_path: str | Path,
    *,
    gpu: GpuProbe | None = None,
    sparkinfer: SparkInferProbe | None = None,
) -> PreflightReport:
    """Run every startup check and return a structured report.

    Call this once, before any model weights are loaded. Passing `gpu=` /
    `sparkinfer=` lets a caller reuse probes it already computed elsewhere
    (or lets tests inject a fabricated environment); leave both None in
    production to probe the live machine.

    This function does not raise, print, or exit on failure — inspect
    `report.ok` (blocks on any fatal failure) and `report.warnings`
    (informational, never blocks) and decide what to do.
    """
    gpu = gpu if gpu is not None else probe_gpu()
    sparkinfer = sparkinfer if sparkinfer is not None else probe_sparkinfer(gpu=gpu)

    checks = (
        check_gpu_present(gpu),
        check_compute_capability(gpu),
        check_cuda_driver_version(gpu),
        check_cuda_runtime_version(gpu),
        check_torch_version(gpu),
        check_sparkinfer_contract(sparkinfer),
        check_sparkinfer_analytic_decode_gate(sparkinfer),
        check_checkpoint_path(checkpoint_path),
        check_checkpoint_config(checkpoint_path),
    )
    return PreflightReport(checks=checks)
