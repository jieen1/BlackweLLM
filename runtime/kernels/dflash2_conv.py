"""CUDA kernel for DFlash2 grouped dynamic causal convolution.

The reference implementation is intentionally easy to audit, but it builds a
zero buffer and a shifted/masked temporary for every tap.  DFlash2 has a
fixed two-tap, group-size-16 path, so keeping the tap loop in one Triton
program removes those intermediate global-memory round trips while retaining
the reference multiply/addcmul ordering.
"""

from __future__ import annotations

import torch

try:  # Triton is optional for the torch-free CI interpreter.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by the torch-free gate
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _grouped_dynamic_conv_kernel(
        hidden_ptr,
        dynamic_ptr,
        base_ptr,
        output_ptr,
        hidden_stride_b,
        hidden_stride_l,
        hidden_stride_h,
        dynamic_stride_b,
        dynamic_stride_l,
        dynamic_stride_t,
        dynamic_stride_g,
        base_stride_t,
        base_stride_g,
        base_stride_d,
        output_stride_b,
        output_stride_l,
        output_stride_h,
        length,
        block_size,
        groups,
        GROUP_SIZE: tl.constexpr,
        TAPS: tl.constexpr,
        BLOCK_GROUP: tl.constexpr,
    ):
        row = tl.program_id(0)
        group = tl.program_id(1)
        lanes = tl.arange(0, BLOCK_GROUP)
        lane_mask = lanes < GROUP_SIZE
        hidden_index = group * GROUP_SIZE + lanes
        valid_group = group < groups
        mask = lane_mask & valid_group

        batch = row // length
        position = row - batch * length
        block_position = position % block_size

        out = tl.zeros((BLOCK_GROUP,), dtype=tl.bfloat16)
        output_ptrs = (
            output_ptr
            + batch * output_stride_b
            + position * output_stride_l
            + hidden_index * output_stride_h
        )
        for tap in range(TAPS):
            valid_position = block_position >= tap
            value_position = tl.maximum(position - tap, 0)
            value = tl.load(
                hidden_ptr
                + batch * hidden_stride_b
                + value_position * hidden_stride_l
                + hidden_index * hidden_stride_h,
                mask=mask & valid_position,
                other=0.0,
            )
            kernel = tl.load(
                base_ptr + tap * base_stride_t + group * base_stride_g + lanes * base_stride_d,
                mask=mask,
                other=0.0,
            )
            dynamic = tl.load(
                dynamic_ptr
                + batch * dynamic_stride_b
                + position * dynamic_stride_l
                + tap * dynamic_stride_t
                + group * dynamic_stride_g,
                mask=valid_group,
                other=0.0,
            )
            # Keep the reference order: output += base * value, followed by
            # torch.addcmul(output, dynamic, value).  Do not combine these
            # into (base + dynamic) * value; BF16 rounding differs.  Triton
            # otherwise keeps the accumulator in FP32 across the two
            # expressions, so the output buffer is used as an explicit BF16
            # round-trip between the statements.  This is still one fused
            # launch and is considerably cheaper than the reference's
            # per-tap temporary tensors.
            if tap > 0:
                out = tl.load(output_ptrs, mask=mask, other=0.0)
            out = out + kernel * value
            tl.store(output_ptrs, out, mask=mask)
            out = tl.load(output_ptrs, mask=mask, other=0.0)
            out = out + dynamic * value
            tl.store(output_ptrs, out, mask=mask)


def fused_grouped_dynamic_convolve(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
    block_size: int,
) -> torch.Tensor | None:
    """Run the fused CUDA path, or return ``None`` when it is not applicable.

    The caller owns the reference fallback.  Returning ``None`` instead of
    silently changing dtype/layout keeps this helper safe for CPU tests and
    for future DFlash checkpoints with a different convolution geometry.
    """

    if triton is None or not hidden.is_cuda:
        return None
    if hidden.ndim != 3 or hidden.dtype != torch.bfloat16:
        return None
    if dynamic.dtype != hidden.dtype or base.dtype != hidden.dtype:
        return None
    if group_size <= 0 or block_size <= 0:
        return None
    batch, length, hidden_size = hidden.shape
    if hidden_size % group_size:
        return None
    groups = hidden_size // group_size
    if base.ndim != 2 or base.shape[1] != hidden_size:
        return None
    taps = int(base.shape[0])
    if taps < 1 or taps > 4:
        return None
    try:
        dynamic_4d = dynamic.reshape(batch, length, taps, groups)
        base_3d = base.reshape(taps, groups, group_size)
    except RuntimeError:
        return None

    output = torch.empty_like(hidden)
    block_group = triton.next_power_of_2(group_size)
    num_warps = 1 if block_group <= 32 else 2 if block_group <= 64 else 4
    _grouped_dynamic_conv_kernel[(batch * length, groups)](
        hidden,
        dynamic_4d,
        base_3d,
        output,
        hidden.stride(0),
        hidden.stride(1),
        hidden.stride(2),
        dynamic_4d.stride(0),
        dynamic_4d.stride(1),
        dynamic_4d.stride(2),
        dynamic_4d.stride(3),
        base_3d.stride(0),
        base_3d.stride(1),
        base_3d.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        length,
        block_size,
        groups,
        GROUP_SIZE=group_size,
        TAPS=taps,
        BLOCK_GROUP=block_group,
        num_warps=num_warps,
        num_stages=2,
    )
    return output
