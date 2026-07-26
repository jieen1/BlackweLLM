"""A2: Force CutlassNvFp4LinearKernel (vLLM's own CUTLASS, less Python overhead).

The FlashInfer wrapper adds Python indirection per GEMM call. With ~304 GEMM
calls per decode round, this overhead is non-trivial. CutlassNvFp4LinearKernel
calls cutlass_scaled_fp4_mm directly (C++ custom op), bypassing FlashInfer.

Usage:
    from runtime.nvfp4_cutlass_direct_patch import patch_nvfp4_prefer_cutlass_direct
    patch_nvfp4_prefer_cutlass_direct()  # call BEFORE get_model()
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("qwen_sm120_runtime.nvfp4_cutlass_direct_patch")

_patched = False


def patch_nvfp4_prefer_cutlass_direct() -> bool:
    """Move CutlassNvFp4LinearKernel to front of NVFP4 priority list.

    Controlled by QSR_A2_CUTLASS_DIRECT env var (default: 1 = enabled).
    """
    global _patched
    if _patched:
        return False

    if os.environ.get("QSR_A2_CUTLASS_DIRECT", "1") == "0":
        return False

    try:
        from runtime.compat_vllm import get_nvfp4_cutlass_kernel_components

        (
            _POSSIBLE_NVFP4_KERNELS,
            CutlassNvFp4LinearKernel,
            PlatformEnum,
        ) = get_nvfp4_cutlass_kernel_components()
    except ImportError:
        logger.warning("A2: cannot import vLLM kernel registry")
        return False

    cuda_kernels = _POSSIBLE_NVFP4_KERNELS.get(PlatformEnum.CUDA, [])

    supported, reason = CutlassNvFp4LinearKernel.is_supported()
    if not supported:
        logger.warning("A2: CutlassNvFp4LinearKernel not supported: %s", reason)
        return False

    if cuda_kernels and cuda_kernels[0] is CutlassNvFp4LinearKernel:
        _patched = True
        return True

    cuda_kernels.remove(CutlassNvFp4LinearKernel)
    cuda_kernels.insert(0, CutlassNvFp4LinearKernel)
    _POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA] = cuda_kernels
    _patched = True
    logger.info("A2: CutlassNvFp4LinearKernel moved to front (direct C++ CUTLASS path)")
    return True
