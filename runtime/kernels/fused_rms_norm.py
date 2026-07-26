"""Triton fused RMSNorm + residual add — zero external dependency.

Single-pass algorithm: loads data once into registers, computes variance,
normalizes and stores — no intermediate global memory round-trip.

For M=1 (decode), H=3072: entire row fits in one thread block's registers.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rms_norm_kernel(
    X_ptr,
    RES_ptr,
    W_ptr,
    OUT_ptr,
    RES_OUT_ptr,
    stride_x_row,
    stride_res_row,
    stride_out_row,
    stride_res_out_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass fused: residual_out = x + residual; out = rmsnorm(residual_out) * weight.

    Loads x+residual into registers, computes variance via reduction,
    then normalizes and stores — one global read, one global write per tensor.
    """
    row = tl.program_id(0)
    x_offset = row * stride_x_row
    res_offset = row * stride_res_row
    out_offset = row * stride_out_row
    res_out_offset = row * stride_res_out_row

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Single load of x and residual
    x = tl.load(X_ptr + x_offset + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(RES_ptr + res_offset + cols, mask=mask, other=0.0).to(tl.float32)

    # Compute new_residual = x + residual (kept in registers)
    val = x + res

    # Store new residual
    tl.store(RES_OUT_ptr + res_out_offset + cols, val.to(tl.bfloat16), mask=mask)

    # Variance reduction (in-register, no extra memory access)
    var = tl.sum(val * val, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Load weight and normalize
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    out = val * rstd * w
    tl.store(OUT_ptr + out_offset + cols, out.to(tl.bfloat16), mask=mask)


@triton.jit
def _rms_norm_kernel(
    X_ptr,
    W_ptr,
    OUT_ptr,
    stride_x_row,
    stride_out_row,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Single-pass standalone RMSNorm: out = rmsnorm(x) * weight."""
    row = tl.program_id(0)
    x_offset = row * stride_x_row
    out_offset = row * stride_out_row

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Single load
    x = tl.load(X_ptr + x_offset + cols, mask=mask, other=0.0).to(tl.float32)

    # Variance
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and store
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    out = x * rstd * w
    tl.store(OUT_ptr + out_offset + cols, out.to(tl.bfloat16), mask=mask)


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual add + RMSNorm. Returns (normed_output, new_residual).

    new_residual = x + residual
    normed_output = rmsnorm(new_residual) * weight
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    res_2d = residual.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape

    out = torch.empty_like(x_2d)
    res_out = torch.empty_like(res_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)

    _fused_add_rms_norm_kernel[(M,)](
        x_2d,
        res_2d,
        weight,
        out,
        res_out,
        x_2d.stride(0),
        res_2d.stride(0),
        out.stride(0),
        res_out.stride(0),
        N=N,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return out.view(orig_shape), res_out.view(orig_shape)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Standalone RMSNorm: out = rmsnorm(x) * weight."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape

    out = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)

    _rms_norm_kernel[(M,)](
        x_2d,
        weight,
        out,
        x_2d.stride(0),
        out.stride(0),
        N=N,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return out.view(orig_shape)
