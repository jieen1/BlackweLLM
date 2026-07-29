"""Keep external runtime-dependency migration surfaces explicit and bounded.

This is intentionally a source-only test: it can run in a CPU-only
environment without importing the runtime or requiring a local vLLM build.
Direct vLLM and FlashInfer imports are forbidden in production. Their exact
empty ledgers are frozen here so new production code cannot add a dependency
bypass silently.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PRODUCTION_SOURCE_DIRECTORIES = ("runtime", "server", "bfdiag")

# Qwen3.6's historical runner is archived under oracle/qwen36_vllm and is
# excluded from production distributions. Runtime and server now have none.
_APPROVED_DIRECT_IMPORT_FILES: set[str] = set()

_APPROVED_DIRECT_FLASHINFER_IMPORT_FILES: set[str] = set()
# Empty since 2026-07-28 (任务#41): laguna_dflash_cudagraph.py's only
# FlashInfer import was inside the same dead DFlashVerifyCudaGraph class
# removed above -- see _APPROVED_DIRECT_IMPORT_FILES's comment on that file.


def _is_type_checking_guard(node: ast.AST) -> bool:
    """True for `if TYPE_CHECKING:` / `if typing.TYPE_CHECKING:`."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _walk_runtime_reachable(node: ast.AST):
    """Like ``ast.walk``, but skips the body of ``if TYPE_CHECKING:``
    guards -- those imports only run for static type checkers (mypy/
    pyright/IDEs), never at actual process runtime, so they don't create
    a real dependency on the guarded package. See e.g.
    runtime/model_loading.py's ``TYPE_CHECKING``-guarded ``VllmConfig``
    import (阶段7, vLLM removal plan): the class is used purely as a type
    annotation there, never instantiated or isinstance-checked, and
    ``from __future__ import annotations`` makes the annotation a lazy
    string anyway -- so guarding it removes the real runtime import
    without losing type-checking value.
    """
    for child in ast.iter_child_nodes(node):
        if _is_type_checking_guard(child):
            continue
        yield child
        yield from _walk_runtime_reachable(child)


def _imports_package(path: Path, package: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in _walk_runtime_reachable(tree):
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
        for directory in _PRODUCTION_SOURCE_DIRECTORIES
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


def test_production_code_does_not_import_archived_qwen_oracle() -> None:
    assert not _direct_import_files("oracle.qwen36_vllm")


def test_distribution_metadata_does_not_offer_vllm() -> None:
    """The production wheel must not advertise vLLM as an installable extra."""
    pyproject = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    extras = pyproject.split("[project.optional-dependencies]", maxsplit=1)[1]
    extras = extras.split("\n[", maxsplit=1)[0]

    assert "vllm" not in extras.lower()
