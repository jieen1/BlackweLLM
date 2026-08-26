"""Offline NVFP4 weight quantization matching the checkpoint contract.

Day-0 prep for Qwen3.8-Flash-Next (see
notes/2026-08-26-qwen38-flash-next-day0-survey.md): the official FP8
checkpoint does not fit a 96 GB card, so serving runs an NVFP4 quant of the
BF16 weights. This module produces the exact triple the loading path
already consumes for every Linear-like weight tensor:

* ``weight_packed``  ``[N, K/2]`` uint8, two e2m1 codes per byte;
* ``weight_scale``   ``[N, K/16]`` F8_E4M3 block scales, LINEAR layout (the
  loader/preparer applies sparkinfer's swizzle, exactly as the real
  checkpoints are stored -- verified against the Laguna snapshot shapes);
* ``weight_global_scale`` scalar fp32, stored as ``2688 / amax`` -- the
  large-scale convention of the compressed-tensors/Laguna checkpoints the
  MoE prepare path consumes (``w*_alpha = 1/gs``); dequant is therefore
  ``code_deq * sf / gs``. The quantizer runs with the SAME value as its
  global-scale argument (sparkinfer's oracle calls it the "reciprocal
  global scale" relative to the nvidia-modelopt small-scale convention --
  the two families store reciprocals of each other, a mismatch already
  documented on ``laguna_sparkinfer_moe.py``). The direction was pinned
  empirically (2026-08-26: only ``2688/amax`` + divide-dequant reproduces
  the weights; the flipped convention gives nRMSE ~1.0).

2688 = 6 (e2m1 max) * 448 (e4m3 max): the choice that puts the tensor's
largest 16-block at the top of the e4m3 range.
"""

from __future__ import annotations

import torch

from runtime.kernels.nvfp4_quant import quantize_nvfp4_activation

_NVFP4_RANGE = 6.0 * 448.0


def quantize_nvfp4_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize one ``[N, K]`` BF16/FP32 weight to the checkpoint triple.

    Returns ``(packed_uint8 [N, K/2], scale_fp8 [N, K/16], global_scale
    scalar fp32)``. K must be a multiple of 32 (the kernel's contract;
    every Qwen-family projection satisfies it).
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be rank-2, got {tuple(weight.shape)}")
    w = (
        weight.to(torch.bfloat16).contiguous()
        if weight.dtype != torch.bfloat16
        else weight.contiguous()
    )
    amax = w.float().abs().amax()
    if bool(amax.isinf()) or bool(amax.isnan()):
        raise ValueError("weight contains non-finite values")
    if float(amax) == 0.0:
        global_scale = torch.ones((), dtype=torch.float32, device=weight.device)
    else:
        global_scale = (_NVFP4_RANGE / amax).to(torch.float32).reshape(())
    packed, sf_linear = quantize_nvfp4_activation(w, global_scale)
    scale = sf_linear[: w.shape[0], : w.shape[1] // 16].view(torch.float8_e4m3fn).contiguous()
    return packed, scale, global_scale


def dequantize_nvfp4_weight(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct ``code_deq * sf / gs`` -- the compressed-tensors/Laguna
    convention this module stores (the nvidia-modelopt family multiplies by
    a small reciprocal scale instead; see the module docstring).

    Verification companion of :func:`quantize_nvfp4_weight`; mirrors the
    e2m1 decode of ``quantize_dequantize_nvfp4_roundtrip``.
    """
    n, k_half = packed.shape
    codes = torch.stack([packed & 0x0F, packed >> 4], dim=-1).reshape(n, k_half * 2)
    mag = codes & 0x07
    sign = (codes >> 3) & 0x01
    mag_f = mag.to(torch.float32)
    val = torch.where(mag == 1, 0.5, torch.zeros_like(mag_f))
    val = torch.where(mag == 2, 1.0, val)
    val = torch.where(mag == 3, 1.5, val)
    val = torch.where(mag == 4, 2.0, val)
    val = torch.where(mag == 5, 3.0, val)
    val = torch.where(mag == 6, 4.0, val)
    val = torch.where(mag == 7, 6.0, val)
    val = val * (1.0 - 2.0 * sign.to(torch.float32))
    sf = scale.view(torch.float8_e4m3fn).to(torch.float32).repeat_interleave(16, dim=-1)
    return val * sf / global_scale.to(torch.float32)
