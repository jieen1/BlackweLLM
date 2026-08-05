"""Triton fused RMSNorm + residual add — zero external dependency.

Single-pass algorithm: loads data once into registers, computes variance,
normalizes and stores — no intermediate global memory round-trip.

For M=1 (decode), H=3072: entire row fits in one thread block's registers.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


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


class TritonRMSNorm(torch.nn.Module):
    """nn.Module wrapper so this kernel can drop directly into a model graph.

    Mirrors the two call shapes vLLM's ``RMSNorm(CustomOp)`` supports and
    that ``LagunaDecoderLayer``/``LagunaModel`` rely on:
    ``forward(x)`` (no residual, first call) -> normed tensor;
    ``forward(x, residual)`` -> ``(normed_output, new_residual)`` tuple.
    No CustomOp base class, no ``dispatch_forward()`` binding -- this is
    meant for code that owns its own module construction and can just
    call the Triton kernel directly (see runtime/model/laguna_model.py).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return rms_norm(x, self.weight, self.eps)
        return fused_add_rms_norm(x, residual, self.weight, self.eps)


@triton.jit
def _rms_norm_tail_kernel(
    X_ptr,
    RSTD_ptr,
    W_ptr,
    OUT_ptr,
    stride_x_row,
    stride_out_row,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Bit-exact norm tail: ``out = bf16(x_f32 * rstd * w1)``.

    The variance/rsqrt prefix stays in torch (reduction order must stay
    torch's); this fuses only the two final fp32 multiplies and the bf16
    round, which are deterministic RN ops -- bit-identical to the torch
    chain (tests/test_norm_tail_bit_parity.py).
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row)
    w = tl.load(W_ptr + cols, mask=mask, other=1.0)
    out = x * rstd * w
    tl.store(OUT_ptr + row * stride_out_row + cols, out.to(tl.bfloat16), mask=mask)


def rms_norm_tail(
    x_f32: torch.Tensor, rstd: torch.Tensor, weight_plus_one: torch.Tensor
) -> torch.Tensor:
    """Fused tail of the zero-centred RMSNorm; returns bf16 ``[M, N]``."""
    m, n = x_f32.shape
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x_f32.device)
    block_n = triton.next_power_of_2(n)
    _rms_norm_tail_kernel[(m,)](
        x_f32,
        rstd,
        weight_plus_one,
        out,
        x_f32.stride(0),
        out.stride(0),
        N=n,
        BLOCK_SIZE=block_n,
        num_warps=4 if block_n <= 4096 else 8,
    )
    return out


@triton.jit
def _gated_norm_tail_kernel(
    X_ptr,
    RSTD_ptr,
    W_ptr,
    G_ptr,
    OUT_ptr,
    stride_x_row,
    stride_out_row,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Bit-exact tail of Qwen36RMSNormGated.

    Torch chain kept exact: ``xn = x_f32 * rstd``; ``wm = bf16(w *
    bf16(xn))`` (weight multiply in input dtype); ``silu(gate)`` via the
    div form ``g / (1 + exp(-g))`` with libdevice exp + div.rn (measured
    bit-identical to torch silu); final ``bf16(wm * silu)``. Variance/rsqrt
    stay in torch (reduction order). All steps are deterministic RN ops --
    bit-identical to the torch chain (tests/test_gated_norm_tail_bit_parity.py).
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_ptr + row)
    w = tl.load(W_ptr + cols, mask=mask, other=1.0)
    g = tl.load(G_ptr + row * stride_x_row + cols, mask=mask, other=0.0).to(tl.float32)
    xn = x * rstd
    # torch promotes weight(f32) * x(bf16) to F32 and does NOT round back to
    # bf16 before the silu multiply -- keep wm in f32 (measured: rounding it
    # to bf16 flips 26% of outputs).
    wm = w.to(tl.float32) * xn.to(tl.bfloat16).to(tl.float32)
    silu_g = tl.div_rn(g, 1.0 + libdevice.exp(-g))
    out = wm * silu_g
    tl.store(OUT_ptr + row * stride_out_row + cols, out.to(tl.bfloat16), mask=mask)


def gated_norm_tail(
    x_f32: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Fused tail of the GDN gated norm; returns bf16 ``[M, N]``."""
    m, n = x_f32.shape
    out = torch.empty(m, n, dtype=torch.bfloat16, device=x_f32.device)
    block_n = triton.next_power_of_2(n)
    _gated_norm_tail_kernel[(m,)](
        x_f32,
        rstd,
        weight,
        gate.reshape(m, n),
        out,
        x_f32.stride(0),
        out.stride(0),
        N=n,
        BLOCK_SIZE=block_n,
        num_warps=4 if block_n <= 4096 else 8,
    )
    return out
