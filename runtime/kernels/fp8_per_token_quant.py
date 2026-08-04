"""Bit-exact Triton per-token E4M3 activation quantizer for the torch
``_scaled_mm`` FP8 route.

Replaces the 6-op torch chain in
``compressed_tensors_linear._quantize_fp8_activation_for_torch_scaled_mm``
with one kernel, BIT-EXACTLY: the per-row amax is order-independent (max is
associative), every other op is elementwise, and the e4m3 cast uses the
same rn-satfinite conversion -- so the produced codes and scales are
identical to the torch chain and the acceptance anchor cannot move.
This is why norm-style fusions (order-dependent variance) cannot use the
same argument but this quantizer can.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_FP8_MAX = 448.0
_SCALE_FLOOR = 1.0 / (_FP8_MAX * 512.0)


@triton.jit
def _fp8_per_token_quant_kernel(
    X_ptr,
    OUT_ptr,
    SCALE_ptr,
    stride_x_row,
    stride_out_row,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    x = tl.load(
        X_ptr + row * stride_x_row + cols, mask=mask, other=0.0
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(amax / 448.0, 1.0 / (448.0 * 512.0))
    # div.rn.f32 via inline asm: torch's CUDA fdiv is div.rn.f32; triton's
    # div.full / fdiv(ieee_rounding=True) both measured +-1ulp off, which
    # flips e4m3 subnormal-midpoint rounding decisions (~2e-5 of codes).
    q = tl.inline_asm_elementwise(
        asm="div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[x, tl.broadcast_to(scale, x.shape)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    tl.store(
        OUT_ptr + row * stride_out_row + cols,
        q.to(tl.float8e4nv),
        mask=mask,
    )
    tl.store(SCALE_ptr + row, scale)


def fp8_per_token_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bit-exact per-token E4M3 quantization matching the torch chain:
    ``scale = clamp_min(amax/448, 1/(448*512)); q = clamp(x/scale, ±448)``.
    Returns ``(x_fp8 [M, K], scale [M, 1] fp32)``."""
    if x.ndim != 2 or not x.is_contiguous():
        x = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x.shape
    out = torch.empty(m, k, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty(m, 1, dtype=torch.float32, device=x.device)
    block_k = triton.next_power_of_2(k)
    num_warps = 4 if block_k <= 4096 else 8
    _fp8_per_token_quant_kernel[(m,)](
        x,
        out,
        scale,
        x.stride(0),
        out.stride(0),
        K=k,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out, scale
