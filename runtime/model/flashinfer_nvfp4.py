"""Optional FlashInfer NVFP4 activation quantization and GEMM helpers.

SGLang's NVFP4 path uses FlashInfer for both the activation quantizer and the
FP4 GEMM.  Keep the import lazy because the runtime's CPU test job does not
install FlashInfer, while the SM120 serving image has a locally validated
Python/cubin patch-version mismatch.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from typing import Any

import torch

logger = logging.getLogger("qwen_sm120_runtime.flashinfer_nvfp4")

_OPS: tuple[Any, Any] | None = None
_IMPORT_ERROR: BaseException | None = None
_IMPORT_REPORTED = False
_SILU_AND_MUL_OP: Any | None = None
_SILU_AND_MUL_IMPORT_ERROR: BaseException | None = None


def load_flashinfer_nvfp4_ops() -> tuple[Any, Any] | None:
    """Return ``(fp4_quantize, mm_fp4)`` or ``None`` when unavailable."""

    global _OPS, _IMPORT_ERROR, _IMPORT_REPORTED
    if _OPS is not None:
        return _OPS
    if _IMPORT_ERROR is not None:
        return None

    # FlashInfer JIT compilation invokes ninja.  The reference vLLM venv owns
    # the executable, but the service launcher does not always put its bin
    # directory on PATH.
    venv_ninja = os.path.join(os.path.dirname(sys.executable), "ninja")
    if os.path.isfile(venv_ninja) and shutil.which("ninja") is None:
        os.environ["PATH"] = os.path.dirname(venv_ninja) + os.pathsep + os.environ.get(
            "PATH", ""
        )

    # The local image has flashinfer-python 0.6.16.post3 and cached SM120
    # cubins from 0.6.13.  The cubins are already exercised by the runtime;
    # retain the existing explicit opt-out of the patch-version gate.
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    try:
        from flashinfer import fp4_quantize, mm_fp4
    except BaseException as exc:  # optional dependency; caller chooses fallback
        _IMPORT_ERROR = exc
        if not _IMPORT_REPORTED:
            logger.warning("FlashInfer NVFP4 helpers unavailable: %s", exc)
            _IMPORT_REPORTED = True
        return None

    _OPS = (fp4_quantize, mm_fp4)
    return _OPS


def quantize_nvfp4_activation(
    input: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    backend: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16/FP16 activations using FlashInfer's swizzled layout.

    The returned scale is the standard two-dimensional FlashInfer tensor
    ``[M_padded, K/16_padded]``.  The caller owns the conversion to its
    backend-specific grouped view; keeping that conversion at the call site
    makes the helper usable by both b12x and ``mm_fp4``.
    """

    ops = load_flashinfer_nvfp4_ops()
    if ops is None:
        raise RuntimeError("FlashInfer NVFP4 quantization is unavailable") from _IMPORT_ERROR
    fp4_quantize, _ = ops
    return fp4_quantize(
        input,
        global_scale,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
        backend=backend,
    )


def silu_and_mul_nvfp4_quantize(
    input: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse SwiGLU and NVFP4 activation quantization.

    This is the FlashInfer CuTe-DSL operation used by SGLang's Blackwell
    W4A4 path.  ``input`` is contiguous ``[M, 2K]`` with the gate half first;
    the result is the same swizzled ``(packed, scale)`` pair consumed by the
    local block-scaled GEMM adapter.  Keep this separate from
    :func:`quantize_nvfp4_activation`: the fused operation has no CUDA backend
    and is intentionally only called on the opt-in FlashInfer MLP path.
    """

    global _SILU_AND_MUL_OP, _SILU_AND_MUL_IMPORT_ERROR
    if _SILU_AND_MUL_OP is None:
        if load_flashinfer_nvfp4_ops() is None:
            raise RuntimeError(
                "FlashInfer NVFP4 quantization is unavailable"
            ) from _IMPORT_ERROR
        try:
            from flashinfer import silu_and_mul_nvfp4_quantize as op
        except BaseException as exc:  # optional dependency; caller owns fallback
            _SILU_AND_MUL_IMPORT_ERROR = exc
            raise RuntimeError(
                "FlashInfer fused SwiGLU NVFP4 quantization is unavailable"
            ) from exc
        _SILU_AND_MUL_OP = op
    if _SILU_AND_MUL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "FlashInfer fused SwiGLU NVFP4 quantization is unavailable"
        ) from _SILU_AND_MUL_IMPORT_ERROR
    return _SILU_AND_MUL_OP(
        input,
        global_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=True,
    )


def mm_fp4(
    input_fp4: torch.Tensor,
    weight_fp4: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    backend: str = "cutlass",
) -> torch.Tensor:
    """Dispatch one FlashInfer FP4xFP4 GEMM."""

    ops = load_flashinfer_nvfp4_ops()
    if ops is None:
        raise RuntimeError("FlashInfer NVFP4 GEMM is unavailable") from _IMPORT_ERROR
    _, flashinfer_mm_fp4 = ops
    return flashinfer_mm_fp4(
        input_fp4,
        weight_fp4,
        input_scale,
        weight_scale,
        alpha,
        out_dtype,
        backend=backend,
    )
