"""Keep external runtime-dependency migration surfaces explicit and bounded.

This is intentionally a source-only test: it can run in a CPU-only
environment without importing the runtime or requiring a local vLLM build.
Direct vLLM and FlashInfer imports are legacy migration sites.  Their exact
lists are frozen here so new production code cannot add another dependency
bypass silently.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# Each entry is an existing migration site.  Remove entries as their callers
# are moved behind runtime.compat_vllm.py or replaced by self-owned code.
_APPROVED_DIRECT_IMPORT_FILES = {
    "runtime/compat_vllm.py",
    "runtime/backends/laguna.py",
    "runtime/backends/laguna_dflash.py",
    "runtime/backends/laguna_dflash_cudagraph.py",
    # Added 2026-07-27: these three call vllm._custom_ops.reshape_and_cache_flash
    # directly. runtime/kernels/fused_kv_scatter.py was written as a drop-in
    # Triton replacement but is NOT yet safe to wire in -- swapping it in and
    # bit-exact-testing against benchmarks/ab_dflash_block_size_64_vs_128.py
    # (bs=64, ctx=10240) collapsed DFlash's accept rate from the established
    # 0.718182 baseline to 0.028839 (deterministic, not a race). Given DFlash's
    # demonstrated extreme sensitivity to any numeric difference (see
    # notes/2026-07-27-block-size-128-accept-rate-root-cause-CLOSED.md), this
    # needs a dedicated numerical debugging pass before it can replace the
    # vLLM call, not a quick wire-in. Tracked as follow-up work under the
    # vLLM-removal plan (notes/2026-07-27-vllm-complete-removal-implementation-plan.md).
    "runtime/backends/bf_attention.py",
    "runtime/backends/laguna_cuda_graph.py",
    "runtime/backends/laguna_sparkinfer_attn.py",
}

_APPROVED_DIRECT_FLASHINFER_IMPORT_FILES = {
    "runtime/backends/laguna_dflash_cudagraph.py",
}


def _imports_package(path: Path, package: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == package or alias.name.startswith(f"{package}.")
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == package or (node.module and node.module.startswith(f"{package}.")):
                return True
    return False


def _direct_import_files(package: str) -> set[str]:
    return {
        path.relative_to(_ROOT).as_posix()
        for directory in ("runtime", "server")
        for path in (_ROOT / directory).rglob("*.py")
        if _imports_package(path, package)
    }


def test_vllm_direct_imports_are_an_explicit_migration_ledger() -> None:
    observed = _direct_import_files("vllm")

    assert observed <= _APPROVED_DIRECT_IMPORT_FILES, (
        "New direct vLLM import outside the migration ledger: "
        f"{sorted(observed - _APPROVED_DIRECT_IMPORT_FILES)}"
    )
    assert observed == _APPROVED_DIRECT_IMPORT_FILES, (
        "A dependency was removed; update the ledger to make the reduction explicit: "
        f"stale={sorted(_APPROVED_DIRECT_IMPORT_FILES - observed)}"
    )


def test_flashinfer_direct_imports_are_an_explicit_migration_ledger() -> None:
    observed = _direct_import_files("flashinfer")

    assert observed <= _APPROVED_DIRECT_FLASHINFER_IMPORT_FILES, (
        "New direct FlashInfer import outside the migration ledger: "
        f"{sorted(observed - _APPROVED_DIRECT_FLASHINFER_IMPORT_FILES)}"
    )
    assert observed == _APPROVED_DIRECT_FLASHINFER_IMPORT_FILES, (
        "A dependency was removed; update the ledger to make the reduction explicit: "
        f"stale={sorted(_APPROVED_DIRECT_FLASHINFER_IMPORT_FILES - observed)}"
    )
