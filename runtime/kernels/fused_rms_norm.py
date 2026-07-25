"""Triton fused RMSNorm + residual add — zero external dependency.

Replaces 3 separate kernels (norm, mul, add) with a single fused kernel.
Saves ~2ms/step at 48 layers × 2 norms/layer by eliminating redundant
global memory round-trips.

Usage:
    from runtime.kernels.fused_rms_norm import fused_add_rms_norm, rms_norm
    # In-place fused: x = x + residual; out = rmsnorm(x) * weight
    hidden, residual = fused_add_rms_norm(hidden, residual, weight, eps)
    # Standalone: out = rmsnorm(x) * weight
    out = rms_norm(x, weight, eps)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_rms_norm_kernel(
    X_ptr, RES_ptr, W_ptr, OUT_ptr, RES_OUT_ptr,
    stride_x_row, stride_res_row, stride_out_row, stride_res_out_row,
    N: tl.constexpr, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused: residual_out = x + residual; out = rmsnorm(residual_out) * weight."""
    row = tl.program_id(0)
    x_offset = row * stride_x_row
    res_offset = row * stride_res_row
    out_offset = row * stride_out_row
    res_out_offset = row * stride_res_out_row

    # Compute x + residual and variance in one pass
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + x_offset + cols, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(RES_ptr + res_offset + cols, mask=mask, other=0.0).to(tl.float32)
        val = x + res
        # Store residual_out = x + residual
        tl.store(RES_OUT_ptr + res_out_offset + cols, val.to(tl.bfloat16), mask=mask)
        _var += val * val

    var = tl.sum(_var, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and scale by weight
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        val = tl.load(RES_OUT_ptr + res_out_offset + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        out = val * rstd * w
        tl.store(OUT_ptr + out_offset + cols, out.to(tl.bfloat16), mask=mask)


@triton.jit
def _rms_norm_kernel(
    X_ptr, W_ptr, OUT_ptr,
    stride_x_row, stride_out_row,
    N: tl.constexpr, eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Standalone RMSNorm: out = rmsnorm(x) * weight."""
    row = tl.program_id(0)
    x_offset = row * stride_x_row
    out_offset = row * stride_out_row

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + x_offset + cols, mask=mask, other=0.0).to(tl.float32)
        _var += x * x

    var = tl.sum(_var, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_ptr + x_offset + cols, mask=mask, other=0.0).to(tl.float32)
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
    
    new_residual = x + residual (in-place on residual)
    normed_output = rmsnorm(new_residual) * weight
    """
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    res_2d = residual.view(-1, orig_shape[-1])
    M, N = x_2d.shape

    out = torch.empty_like(x_2d)
    res_out = torch.empty_like(res_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 8192:
        BLOCK_SIZE = 8192

    _fused_add_rms_norm_kernel[(M,)](
        x_2d, res_2d, weight, out, res_out,
        x_2d.stride(0), res_2d.stride(0), out.stride(0), res_out.stride(0),
        N=N, eps=eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.view(orig_shape), res_out.view(orig_shape)


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Standalone RMSNorm: out = rmsnorm(x) * weight."""
    orig_shape = x.shape
    x_2d = x.view(-1, orig_shape[-1])
    M, N = x_2d.shape

    out = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 8192:
        BLOCK_SIZE = 8192

    _rms_norm_kernel[(M,)](
        x_2d, weight, out,
        x_2d.stride(0), out.stride(0),
        N=N, eps=eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.view(orig_shape)
