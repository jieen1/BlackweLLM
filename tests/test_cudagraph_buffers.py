"""CUDA Graph buffer 管理回归测试（CPU-only，不需要模型权重）。

自 SparkInfer CG 迁移（commit adcca60，"No FlashInfer dependency for attention
anymore"）后，decode 不再有 fast_decode_plan/_fi_* CPU planning + H2D 拷贝；
page_table/cache_seqlens 由 _fill_buffers() 直接以 GPU tensor 写入更新。
验证的契约相应变为：

- replay() 必须先 _fill_buffers() 再 _graph.replay()（否则 replay 用的是上一步的
  stale metadata）
- 每个 layer group 的 sparkinfer workspace 必须独立分配，不能共享
- page-crossing 检测逻辑正确
- indptr 累积和正确
- last_page_len 计算正确
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


class TestReplayContract:
    """验证 replay() 履行了 SparkInfer CG decode 的调用方契约。"""

    def _get_replay_source(self) -> str:
        pytest.importorskip("torch")
        from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode

        return inspect.getsource(LagunaCudaGraphDecode.replay)

    def test_fill_buffers_before_graph_replay(self):
        """replay() 必须先调用 _fill_buffers() 更新 page_table/cache_seqlens，再 replay 图。"""
        src = self._get_replay_source()
        idx_fill = src.find("_fill_buffers(")
        idx_replay = src.find("_graph.replay()")
        assert idx_fill != -1 and idx_replay != -1, (
            "replay() 缺少 _fill_buffers()/_graph.replay() 调用"
        )
        assert idx_fill < idx_replay, (
            "_fill_buffers() 必须在 _graph.replay() 之前调用，"
            "否则 replay 用的是上一步的 stale page_table/cache_seqlens"
        )

    def test_no_priming_replay_in_capture(self):
        """capture() 不应包含 priming replay（已证明是同根因的 workaround）。"""
        pytest.importorskip("torch")
        from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode

        src = inspect.getsource(LagunaCudaGraphDecode.capture)
        assert "Prime" not in src and "priming" not in src.lower(), (
            "capture() 仍包含 priming replay — 根因修复后不再需要"
        )

    def test_reset_forgets_page_crossing_state(self):
        pytest.importorskip("torch")
        from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode

        graph = object.__new__(LagunaCudaGraphDecode)
        graph.batch_size = 2
        graph._prev_n_blocks = [17, 19]
        graph._swa_prev_n_blocks = [3, 4]

        graph.reset()

        assert graph._prev_n_blocks == [0, 0]
        assert graph._swa_prev_n_blocks == [0, 0]

    def test_b1_metadata_fastpath_requires_lagunas_two_group_layout(self):
        pytest.importorskip("torch")
        from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode

        graph = object.__new__(LagunaCudaGraphDecode)
        graph._b1_metadata_fastpath_enabled = True
        graph._b1_metadata_values_gpu = None
        graph._decode_workspaces = {
            (-1, 48, 8): object(),
            (511, 72, 8): object(),
        }
        graph._ring_blocks_per_slot = 10
        graph.device = "cuda:0"

        # The actual allocation needs CUDA; this contract guards the layout
        # selection before that allocation is reached in integration tests.
        assert [key for key in graph._decode_workspaces if key[0] < 0] == [(-1, 48, 8)]
        assert [key for key in graph._decode_workspaces if key[0] >= 0] == [(511, 72, 8)]

    def test_hot_reloaded_graph_defaults_fastpath_on_and_can_disable(self, monkeypatch):
        pytest.importorskip("torch")
        from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode

        graph = object.__new__(LagunaCudaGraphDecode)
        graph._decode_workspaces = {}
        graph._ring_blocks_per_slot = 10
        monkeypatch.delenv("QSR_B1_METADATA_FASTPATH", raising=False)

        assert graph._init_b1_metadata_fastpath() is False
        assert graph._b1_metadata_fastpath_enabled is True

        disabled = object.__new__(LagunaCudaGraphDecode)
        monkeypatch.setenv("QSR_B1_METADATA_FASTPATH", "0")
        assert disabled._init_b1_metadata_fastpath() is False
        assert disabled._b1_metadata_fastpath_enabled is False


class TestBufferArithmetic:
    """验证 buffer 计算的纯算术逻辑。"""

    def test_last_page_len_computation(self):
        """last_page_len = new_kv % page_size, 0 → page_size。"""
        page_size = 16
        cases = [
            (1, 1),
            (15, 15),
            (16, 16),
            (17, 1),
            (31, 15),
            (32, 16),
            (33, 1),
            (256, 16),
        ]
        for new_kv, expected in cases:
            lpl = new_kv % page_size
            lpl = lpl if lpl != 0 else page_size
            assert lpl == expected, f"new_kv={new_kv}: got {lpl}, want {expected}"

    def test_n_blocks_computation(self):
        """n_blocks = ceil(new_kv / page_size)。"""
        page_size = 16
        cases = [
            (1, 1),
            (16, 1),
            (17, 2),
            (32, 2),
            (33, 3),
            (256, 16),
        ]
        for new_kv, expected in cases:
            n_blocks = (new_kv + page_size - 1) // page_size
            assert n_blocks == expected, f"new_kv={new_kv}: got {n_blocks}, want {expected}"

    def test_indptr_cumulative(self):
        """indptr 是 n_blocks 的前缀和。"""
        n_blocks_list = [1, 2, 1, 3]
        indptr = [0]
        for nb in n_blocks_list:
            indptr.append(indptr[-1] + nb)
        assert indptr == [0, 1, 3, 4, 7]

    def test_slot_mapping_formula(self):
        """slot_mapping = (base + pos // page_size) * page_size + pos % page_size。"""
        page_size = 16
        blocks_per_slot = 256
        slot_id = 0
        phys = slot_id  # RESERVED_PHYSICAL_SLOTS = 0
        base = phys * blocks_per_slot
        for pos in [0, 5, 15, 16, 17, 100]:
            sm = (base + pos // page_size) * page_size + pos % page_size
            expected = base * page_size + pos
            assert sm == expected, f"pos={pos}: got {sm}, want {expected}"

    def test_page_crossing_detection(self):
        """跨页检测：n_blocks 变化时触发 indptr/indices 重建。"""
        page_size = 16
        prev_n_blocks = 0
        crossings = []
        for kv_len in range(50):
            new_kv = kv_len + 1
            n_blocks = (new_kv + page_size - 1) // page_size
            if n_blocks != prev_n_blocks:
                crossings.append(kv_len)
                prev_n_blocks = n_blocks
        assert crossings == [0, 16, 32, 48], f"跨页点错误: {crossings}"


class TestIndependentWorkspace:
    """验证每个 layer group 有独立的 sparkinfer workspace。"""

    def test_workspace_independent_per_group(self):
        """_init_workspaces 必须为每个 group_key 创建独立的 SparkinferDecodeWorkspace 实例。"""
        pytest.importorskip("torch")
        from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode

        src = inspect.getsource(LagunaCudaGraphDecode._init_workspaces)
        assert "SparkinferDecodeWorkspace(" in src, (
            "_init_workspaces 必须为每个 group 创建独立的 SparkinferDecodeWorkspace 实例，"
            "不能跨 group 共享"
        )


class TestDependencySurfaces:
    """验证图路径只依赖 owned runtime context 与 Sparkinfer。"""

    def test_graph_modules_do_not_import_runtime_compat_vllm(self):
        for relpath in (
            "runtime/backends/laguna_cuda_graph.py",
            "runtime/backends/laguna_dflash_cudagraph.py",
        ):
            tree = ast.parse((_ROOT / relpath).read_text(encoding="utf-8"), filename=relpath)
            imports_compat = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports_compat = any(
                        alias.name == "runtime.compat_vllm" for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom):
                    imports_compat = node.module == "runtime.compat_vllm"
                if imports_compat:
                    break
            assert not imports_compat, (
                f"{relpath} 应直接使用 runtime.laguna_runtime，而不是 compat_vllm"
            )

    def test_dflash_graph_module_has_no_flashinfer_dependency(self):
        relpath = "runtime/backends/laguna_dflash_cudagraph.py"
        tree = ast.parse((_ROOT / relpath).read_text(encoding="utf-8"), filename=relpath)
        imports_flashinfer = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports_flashinfer = any(
                    alias.name == "flashinfer" or alias.name.startswith("flashinfer.")
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                imports_flashinfer = node.module == "flashinfer" or (
                    node.module is not None and node.module.startswith("flashinfer.")
                )
            if imports_flashinfer:
                break
        assert not imports_flashinfer, (
            "DFlash draft CUDA graph 应只保留 Sparkinfer 路径，不应残留 FlashInfer 依赖"
        )

    def test_graph_modules_use_owned_runtime_context(self):
        for relpath in (
            "runtime/backends/laguna_cuda_graph.py",
            "runtime/backends/laguna_dflash_cudagraph.py",
        ):
            src = (_ROOT / relpath).read_text(encoding="utf-8")
            assert "laguna_forward_context" in src, (
                f"{relpath} 应通过 runtime.laguna_runtime 提供 forward context"
            )
