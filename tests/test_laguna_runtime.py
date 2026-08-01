"""CPU contracts for Laguna's owned runtime boundary."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.laguna_runtime import LagunaAttentionMetadata, bind_laguna_kv_cache


def test_owned_attention_metadata_preserves_sparkinfer_adapter_fields() -> None:
    metadata = LagunaAttentionMetadata(
        query_start_loc="qsl",
        query_start_loc_cpu="cpu-qsl",
        seq_lens="lengths",
        num_reqs=2,
        num_actual_tokens=5,
        max_query_len=3,
        max_seq_len=17,
        block_table_tensor="pages",
        slot_mapping="slots",
        causal=True,
    )

    assert metadata.num_actual_tokens == 5
    assert metadata.block_table_tensor == "pages"
    assert metadata.causal is True


def test_owned_kv_binding_sorts_layers_and_updates_placeholders() -> None:
    layers = {
        "model.layers.10.attn": SimpleNamespace(kv_cache=None),
        "model.layers.2.attn": SimpleNamespace(kv_cache=None),
    }
    caches = {
        "model.layers.10.attn": "cache-10",
        "model.layers.2.attn": "cache-2",
    }
    runner_caches: list[object] = []

    bind_laguna_kv_cache(caches, layers, runner_caches)

    assert runner_caches == ["cache-2", "cache-10"]
    assert layers["model.layers.2.attn"].kv_cache == "cache-2"
    assert layers["model.layers.10.attn"].kv_cache == "cache-10"


def test_laguna_runtime_modules_do_not_import_qwen_legacy_vllm() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "runtime/backends/laguna.py",
        "runtime/backends/laguna_dflash.py",
    ):
        tree = ast.parse((root / relative_path).read_text(encoding="utf-8"), filename=relative_path)
        assert not any(
            (
                isinstance(node, ast.ImportFrom)
                and node.module
                in {
                    "runtime.compat_vllm",
                    "runtime.compat_vllm_qwen36",
                    "oracle.qwen36_vllm.vllm_compat",
                    "oracle.qwen36_vllm.attention_compat",
                }
            )
            or (
                isinstance(node, ast.Import)
                and any(
                    alias.name
                    in {
                        "runtime.compat_vllm",
                        "runtime.compat_vllm_qwen36",
                        "oracle.qwen36_vllm.vllm_compat",
                        "oracle.qwen36_vllm.attention_compat",
                    }
                    for alias in node.names
                )
            )
            for node in ast.walk(tree)
        ), f"{relative_path} must not depend on the Qwen vLLM legacy tenant"


def test_laguna_backend_has_no_dead_vllm_patch_hooks() -> None:
    source = (Path(__file__).resolve().parents[1] / "runtime" / "backends" / "laguna.py").read_text(
        encoding="utf-8"
    )

    assert "patch_nvfp4_" not in source
    assert "_patch_rmsnorm_triton" not in source


def test_laguna_backend_import_does_not_require_vllm() -> None:
    # The subprocess below imports runtime.backends.laguna, which imports torch
    # eagerly -- so torch is what this test actually needs, not numpy. Guarding
    # on numpy alone happened to skip under CI's dev extras (pytest + ruff, no
    # numpy) while failing outright in any environment that has numpy but not
    # torch. The guard now names the dependency the subprocess really requires.
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    blocker = """
import importlib.abc
import sys

class BlockVllm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'vllm' or fullname.startswith('vllm.'):
            raise ModuleNotFoundError('vllm intentionally blocked')
        return None

sys.meta_path.insert(0, BlockVllm())
import server.app
import server.engine
import runtime.backends.laguna
import runtime.backends.laguna_dflash
"""
    completed = subprocess.run(
        [sys.executable, "-c", blocker],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_server_app_defaults_to_the_only_supported_backend() -> None:
    pytest.importorskip("fastapi")
    completed = subprocess.run(
        [sys.executable, "-c", "import server.app; print(server.app.SERVER_MODEL_BACKEND)"],
        check=False,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env={key: value for key, value in os.environ.items() if key != "QSR_SERVER_MODEL_BACKEND"},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "laguna"
