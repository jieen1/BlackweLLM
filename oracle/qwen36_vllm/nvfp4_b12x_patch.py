"""A2: Patch NVFP4 kernel selection to prefer FlashInfer B12x on SM120+.

B12x is a warp-level MMA kernel specifically designed for SM120 (Blackwell
GeForce / RTX PRO). Micro-benchmarks show it is bit-exact vs CUTLASS on all
model shapes and ~6% faster on weighted decode GEMM total.

The kernel exists in vLLM but is excluded from auto-selection pending an
upstream CUTLASS SM121 MMA op guard fix. This patch re-inserts it at the
front of the priority list so it is chosen on SM120 hardware.

Usage:
    from oracle.qwen36_vllm.nvfp4_b12x_patch import patch_nvfp4_prefer_b12x
    patch_nvfp4_prefer_b12x()  # call BEFORE get_model()
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("qwen_sm120_oracle.qwen36_vllm.nvfp4_b12x_patch")

_patched = False


def patch_nvfp4_prefer_b12x() -> bool:
    """Insert FlashInferB12xNvFp4LinearKernel at front of NVFP4 priority list.

    Returns True if patch was applied, False if skipped.
    Controlled by QSR_A2_B12X env var (default: 1 = enabled).
    """
    global _patched
    if _patched:
        return False

    if os.environ.get("QSR_A2_B12X", "1") == "0":
        logger.info("A2: B12x patch disabled by QSR_A2_B12X=0")
        return False

    try:
        import torch

        cc = torch.cuda.get_device_capability()
        if cc[0] < 12:
            logger.info("A2: B12x patch skipped (SM%d < SM120)", cc[0] * 10 + cc[1])
            return False
    except Exception:
        return False

    try:
        from oracle.qwen36_vllm.vllm_compat import get_nvfp4_b12x_kernel_components

        (
            _POSSIBLE_NVFP4_KERNELS,
            FlashInferB12xNvFp4LinearKernel,
            PlatformEnum,
        ) = get_nvfp4_b12x_kernel_components()
    except ImportError:
        logger.warning("A2: cannot import vLLM kernel registry, skipping B12x patch")
        return False

    cuda_kernels = _POSSIBLE_NVFP4_KERNELS.get(PlatformEnum.CUDA, [])
    if FlashInferB12xNvFp4LinearKernel in cuda_kernels:
        logger.info("A2: B12x already in NVFP4 kernel list")
        _patched = True
        return True

    supported, reason = FlashInferB12xNvFp4LinearKernel.is_supported()
    if not supported:
        logger.warning("A2: B12x kernel not supported: %s", reason)
        return False

    cuda_kernels.insert(0, FlashInferB12xNvFp4LinearKernel)
    _POSSIBLE_NVFP4_KERNELS[PlatformEnum.CUDA] = cuda_kernels
    _patched = True
    logger.info(
        "A2: B12x kernel inserted at front of NVFP4 priority list "
        "(SM120 warp-level MMA, bit-exact, ~6%% faster)"
    )
    return True
