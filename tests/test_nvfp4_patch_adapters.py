"""CPU-only regression tests for archived Qwen NVFP4 patch adapters.

The oracle patch modules obtain upstream symbols only through
``oracle.qwen36_vllm.vllm_compat`` when a patch is actually enabled.
"""

from __future__ import annotations

import sys
import types

import pytest


def _install_compat_stub(monkeypatch, **symbols: object) -> None:
    compat = types.ModuleType("oracle.qwen36_vllm.vllm_compat")
    for name, value in symbols.items():
        setattr(compat, name, value)
    monkeypatch.setitem(sys.modules, "oracle.qwen36_vllm.vllm_compat", compat)


def test_b12x_patch_uses_compat_registry(monkeypatch) -> None:
    from oracle.qwen36_vllm import nvfp4_b12x_patch

    class PlatformEnum:
        CUDA = "cuda"

    class B12xKernel:
        @staticmethod
        def is_supported():
            return True, ""

    kernels = {PlatformEnum.CUDA: []}
    _install_compat_stub(
        monkeypatch,
        get_nvfp4_b12x_kernel_components=lambda: (kernels, B12xKernel, PlatformEnum),
    )
    monkeypatch.setattr(nvfp4_b12x_patch, "_patched", False)
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(get_device_capability=lambda: (12, 0))
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert nvfp4_b12x_patch.patch_nvfp4_prefer_b12x() is True
    assert kernels[PlatformEnum.CUDA] == [B12xKernel]


def test_cutlass_patch_uses_compat_registry(monkeypatch) -> None:
    from oracle.qwen36_vllm import nvfp4_cutlass_direct_patch

    class PlatformEnum:
        CUDA = "cuda"

    class CutlassKernel:
        @staticmethod
        def is_supported():
            return True, ""

    class OtherKernel:
        pass

    kernels = {PlatformEnum.CUDA: [OtherKernel, CutlassKernel]}
    _install_compat_stub(
        monkeypatch,
        get_nvfp4_cutlass_kernel_components=lambda: (
            kernels,
            CutlassKernel,
            PlatformEnum,
        ),
    )
    monkeypatch.setattr(nvfp4_cutlass_direct_patch, "_patched", False)

    assert nvfp4_cutlass_direct_patch.patch_nvfp4_prefer_cutlass_direct() is True
    assert kernels[PlatformEnum.CUDA] == [CutlassKernel, OtherKernel]


def test_custom_gemm_disabled_path_does_not_import_compat(monkeypatch) -> None:
    from oracle.qwen36_vllm import nvfp4_custom_gemm

    monkeypatch.setattr(nvfp4_custom_gemm, "_patched", False)
    monkeypatch.setenv("QSR_A2_CUSTOM_GEMM", "0")

    assert nvfp4_custom_gemm.patch_nvfp4_custom_gemm() is False


@pytest.mark.parametrize(
    ("width", "expected_config"),
    [
        (100352, 3),
        (17408, 3),
        (8192, 0),
        (5120, 1),
        (3072, 1),
    ],
)
def test_custom_gemm_selects_the_frozen_nvfp4_tile_config(width, expected_config) -> None:
    from oracle.qwen36_vllm.nvfp4_custom_gemm import _select_config

    assert _select_config(width) == expected_config


def test_cudnn_disabled_path_does_not_import_compat(monkeypatch) -> None:
    from oracle.qwen36_vllm import nvfp4_cudnn_patch

    monkeypatch.setattr(nvfp4_cudnn_patch, "_patched", False)
    monkeypatch.setenv("QSR_A2_CUDNN", "0")

    assert nvfp4_cudnn_patch.patch_nvfp4_to_cudnn() is False
