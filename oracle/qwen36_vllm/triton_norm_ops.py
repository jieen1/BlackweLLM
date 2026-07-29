"""Triton-fused RMSNorm + fused_add_rms_norm IR op implementations.

The vLLM C extension (_C.abi3.so) on this machine lacks rms_norm /
fused_add_rms_norm symbols, so the IR op system falls back to the native
PyTorch path (8+ separate kernel launches per call).  Registering Triton
implementations collapses each call to a single kernel launch.

Key fix: GemmaRMSNorm passes float32 weight (self.weight.float() + 1.0)
with bfloat16 input.  The supports_args check must allow mixed dtypes.

Usage – call once after create_engine_config()::

    from oracle.qwen36_vllm.triton_norm_ops import install_triton_norm_ops
    install_triton_norm_ops()
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _rms_norm_triton_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start = input_ptr + row_idx * input_row_stride
    out_start = output_ptr + row_idx * output_row_stride

    sum_sq = tl.zeros([1], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < n_cols
        vals = tl.load(row_start + idx, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(tl.where(mask, vals * vals, 0.0))

    inv_rms = tl.rsqrt(sum_sq / n_cols + eps)

    for off in range(0, n_cols, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < n_cols
        vals = tl.load(row_start + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        out = vals * inv_rms * w
        tl.store(
            out_start + idx, out.to(tl.load(row_start + idx, mask=mask, other=0.0).dtype), mask=mask
        )


@triton.jit
def _fused_add_rms_norm_triton_kernel(
    input_ptr,
    residual_ptr,
    weight_ptr,
    input_row_stride,
    residual_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """In-place: residual += input; input = rms_norm(residual) * weight."""
    row_idx = tl.program_id(0).to(tl.int64)
    inp_start = input_ptr + row_idx * input_row_stride
    res_start = residual_ptr + row_idx * residual_row_stride

    sum_sq = tl.zeros([1], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < n_cols
        x = tl.load(inp_start + idx, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_start + idx, mask=mask, other=0.0).to(tl.float32)
        s = x + r
        tl.store(
            res_start + idx, s.to(tl.load(res_start + idx, mask=mask, other=0.0).dtype), mask=mask
        )
        sum_sq += tl.sum(tl.where(mask, s * s, 0.0))

    inv_rms = tl.rsqrt(sum_sq / n_cols + eps)

    for off in range(0, n_cols, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < n_cols
        r = tl.load(res_start + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        out = r * inv_rms * w
        tl.store(
            inp_start + idx, out.to(tl.load(inp_start + idx, mask=mask, other=0.0).dtype), mask=mask
        )


# ---------------------------------------------------------------------------
# Python wrappers (match IR op signatures)
# ---------------------------------------------------------------------------


def _triton_rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    assert variance_size is None
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    n_rows, n_cols = x_2d.shape
    output = torch.empty_like(x_2d)
    w = (
        weight.contiguous()
        if weight is not None
        else torch.ones(n_cols, device=x.device, dtype=torch.float32)
    )
    BLOCK = min(triton.next_power_of_2(n_cols), 4096)
    _rms_norm_triton_kernel[(n_rows,)](
        x_2d,
        w,
        output,
        x_2d.stride(0),
        output.stride(0),
        n_cols,
        epsilon,
        BLOCK_SIZE=BLOCK,
    )
    return output.view(orig_shape)


def _triton_fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    assert variance_size is None
    assert x.shape == x_residual.shape
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    r_2d = x_residual.reshape(-1, x_residual.shape[-1])
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    if not r_2d.is_contiguous():
        r_2d = r_2d.contiguous()
    n_rows, n_cols = x_2d.shape
    w = (
        weight.contiguous()
        if weight is not None
        else torch.ones(n_cols, device=x.device, dtype=torch.float32)
    )
    BLOCK = min(triton.next_power_of_2(n_cols), 4096)
    _fused_add_rms_norm_triton_kernel[(n_rows,)](
        x_2d,
        r_2d,
        w,
        x_2d.stride(0),
        r_2d.stride(0),
        n_cols,
        epsilon,
        BLOCK_SIZE=BLOCK,
    )
    return x_2d.view(orig_shape), r_2d.view(orig_shape)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_installed = False


def install_triton_norm_ops() -> None:
    """Register Triton implementations for rms_norm / fused_add_rms_norm
    IR ops and set them as highest priority.  Idempotent.

    KEY: supports_args allows mixed dtypes (float32 weight + bfloat16 input)
    because GemmaRMSNorm passes ``self.weight.float() + 1.0`` as weight."""
    global _installed
    if _installed:
        return
    _installed = True

    from oracle.qwen36_vllm.vllm_compat import get_vllm_ir

    ir = get_vllm_ir()

    def _no_var(x, weight, epsilon, variance_size=None):
        return variance_size is None

    def _no_var_add(x, x_residual, weight, epsilon, variance_size=None):
        return variance_size is None

    ir.ops.rms_norm.register_impl(
        "triton",
        supports_args=_no_var,
        supported=True,
    )(_triton_rms_norm)

    ir.ops.fused_add_rms_norm.register_impl(
        "triton",
        supports_args=_no_var_add,
        supported=True,
        inplace=True,
    )(_triton_fused_add_rms_norm)

    ir.ops.rms_norm.set_default(["triton", "native"])
    ir.ops.fused_add_rms_norm.set_default(["triton", "native"])


# ---------------------------------------------------------------------------
# Triton SiluAndMul (fused SwiGLU activation)
# ---------------------------------------------------------------------------


@triton.jit
def _silu_and_mul_triton_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    d,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    inp_start = input_ptr + row_idx * input_row_stride
    out_start = output_ptr + row_idx * output_row_stride

    for off in range(0, d, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < d
        gate = tl.load(inp_start + idx, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(inp_start + d + idx, mask=mask, other=0.0).to(tl.float32)
        silu_gate = gate * tl.sigmoid(gate)
        out = (silu_gate * up).to(tl.load(inp_start + idx, mask=mask, other=0.0).dtype)
        tl.store(out_start + idx, out, mask=mask)


def triton_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Fused SiLU-and-mul (SwiGLU activation): x -> silu(x[:d]) * x[d:]."""
    d = x.shape[-1] // 2
    orig_shape = x.shape[:-1] + (d,)
    x_2d = x.reshape(-1, x.shape[-1])
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()
    n_rows = x_2d.shape[0]
    output = torch.empty(n_rows, d, device=x.device, dtype=x.dtype)
    BLOCK = min(triton.next_power_of_2(d), 4096)
    _silu_and_mul_triton_kernel[(n_rows,)](
        x_2d,
        output,
        x_2d.stride(0),
        output.stride(0),
        d,
        BLOCK_SIZE=BLOCK,
    )
    return output.view(orig_shape)
