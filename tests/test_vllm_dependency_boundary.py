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
    # Added 2026-07-28 (任务#42): the qwen36/DirectModelRunner-exclusive
    # counterpart to compat_vllm.py -- GDNAttentionMetadata/
    # SM120GQAMetadata/AttentionBackendEnum/register_backend/FLA chunk
    # helpers/compute_causal_conv1d_metadata, split out of compat_vllm.py
    # because their only real consumers (runtime/metadata_builders.py,
    # runtime/cuda_graphs.py, runtime/direct_model_runner.py) are
    # exclusively the qwen36 tenant (server/engine.py's _load_model()
    # dispatches "laguna" to LagunaBackend, never DirectModelRunner) --
    # having them as unconditional module-level imports in the SHARED
    # compat_vllm.py meant Laguna's own import chain transitively required
    # them too, purely as a file-sharing accident (found via a coordinator
    # cross-check, not this project's own earlier audits). qwen3.6/
    # DirectModelRunner itself was explicitly put out of scope for this
    # whole vLLM-removal effort at 阶段0 ("qwen3.6(DirectModelRunner)路径
    # 本次不动") -- this file is intentionally NOT part of that removal
    # target, it exists so Laguna's ledger entries above don't have to
    # carry qwen36's dependency weight.
    "runtime/compat_vllm_qwen36.py",
    # runtime/backends/laguna_dflash_cudagraph.py removed 2026-07-28
    # (任务#41, vLLM removal plan 阶段8): its only vLLM/FlashInfer import
    # was inside DFlashVerifyCudaGraph, a FlashInfer-based main-model
    # verify CUDA graph -- re-verified (not just carried forward from the
    # 阶段0 audit's suspicion) to have zero real callers since commit
    # d4354e939e9 (2026-07-25); the active verify-CUDA-graph path is
    # LagunaCudaGraphVerify (runtime/backends/laguna_cuda_graph.py,
    # sparkinfer-only, never imported vLLM/FlashInfer). Deleted the dead
    # class entirely rather than leave it on the ledger unused.
    # runtime/model_loading.py, runtime/model/laguna_model.py,
    # runtime/model/laguna_decoder.py, runtime/model/laguna_dflash_model.py
    # removed 2026-07-28 (任务#41, vLLM removal plan 阶段8): their last
    # remaining real vLLM imports (default_weight_loader/
    # maybe_remap_kv_scale_name, fused_moe_make_expert_params_mapping,
    # get_tensor_model_parallel_rank, extract_layer_index,
    # set_default_torch_dtype) were all small, checkpoint/TP=1-specific
    # utilities -- self-built narrowed ports (runtime/model/_weight_
    # loading.py, runtime/model/laguna_decoder.py's _extract_layer_index,
    # runtime/model_loading.py's _default_torch_dtype) or, for the MoE
    # expert-params-mapping call, confirmed 100% dead weight and deleted
    # outright (LagunaMoESelfBuilt has had no experts submodule to match
    # against since 阶段6) rather than ported. runtime/model/plain_linear.py,
    # plain_embedding.py, plain_attention.py, nvfp4_linear.py, and _prefix.py
    # (same directory) already had zero vLLM imports by design.
}

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
