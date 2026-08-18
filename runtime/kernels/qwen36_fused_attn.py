"""Qwen3.5/Qwen3.8 full-attention epilogue kernels.

These are the local equivalents of SGLang's
``fused_qk_gemma_rmsnorm_rope_gate`` and ``fused_sigmoid_mul``.  Keeping the
implementation here avoids making the runtime depend on SGLang while using
the same tensor layout and operation ordering:

* q/gate is read from the interleaved ``[q_h, gate_h]`` projection;
* Q/K Gemma RMSNorm, partial NeoX RoPE, and gate extraction happen in one
  launch;
* the attention output gate is applied in one launch over a strided gate.

The kernels deliberately have no model-specific weight packing or cache
ownership.  They are usable by eager prefill and by graph paths once their
fixed geometry has been captured.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_rmsnorm_rope_gate_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    stride_qg_token,
    stride_qg_head,
    stride_k_token,
    stride_k_head,
    stride_q_out_token,
    stride_k_out_token,
    stride_gate_token,
    stride_gate_head,
    stride_cos_token,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    head_block: tl.constexpr,
    rotary_block: tl.constexpr,
    eps: tl.constexpr,
    fp16: tl.constexpr,
    has_pass: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)
    out_dtype = tl.float16 if fp16 else tl.bfloat16

    if is_k:
        input_base = k_ptr + token * stride_k_token + local_head * stride_k_head
        weight_ptr = k_weight_ptr
        output_base = k_out_ptr + token * stride_k_out_token + local_head * head_dim
    else:
        input_base = q_gate_ptr + token * stride_qg_token + local_head * stride_qg_head
        weight_ptr = q_weight_ptr
        output_base = q_out_ptr + token * stride_q_out_token + local_head * head_dim

    offsets = tl.arange(0, head_block)
    mask = offsets < head_dim
    x = tl.load(input_base + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / head_dim
    inv_rms = tl.rsqrt(variance + eps)
    normalized = (x * inv_rms * (weight + 1.0)).to(out_dtype).to(tl.float32)

    if has_pass:
        pass_mask = mask & (offsets >= rotary_dim)
        tl.store(output_base + offsets, normalized, mask=pass_mask)

    rotary_offsets = tl.arange(0, rotary_block)
    rotary_mask = rotary_offsets < half_rotary
    x0 = tl.load(input_base + rotary_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
    x1 = tl.load(
        input_base + half_rotary + rotary_offsets,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    w0 = tl.load(weight_ptr + rotary_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
    w1 = tl.load(
        weight_ptr + half_rotary + rotary_offsets,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    x0 = (x0 * inv_rms * (w0 + 1.0)).to(out_dtype).to(tl.float32)
    x1 = (x1 * inv_rms * (w1 + 1.0)).to(out_dtype).to(tl.float32)

    position = tl.load(positions_ptr + token).to(tl.int64)
    cache_base = cos_sin_cache_ptr + position * stride_cos_token
    cos = tl.load(cache_base + rotary_offsets, mask=rotary_mask, other=0.0).to(tl.float32)
    sin = tl.load(
        cache_base + half_rotary + rotary_offsets,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(output_base + rotary_offsets, x0 * cos - x1 * sin, mask=rotary_mask)
    tl.store(
        output_base + half_rotary + rotary_offsets,
        x1 * cos + x0 * sin,
        mask=rotary_mask,
    )

    if not is_k:
        gate_base = gate_out_ptr + token * stride_gate_token + local_head * stride_gate_head
        gate = tl.load(input_base + head_dim + offsets, mask=mask, other=0.0)
        tl.store(gate_base + offsets, gate, mask=mask)


def fused_qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse Q/K Gemma RMSNorm, partial NeoX RoPE, and gate extraction.

    ``q_gate`` is ``[T, num_q_heads, 2 * head_dim]`` and ``k`` is
    ``[T, num_kv_heads, head_dim]``.  Outputs are contiguous Q/K flat rows
    plus a contiguous ``[T, num_q_heads, head_dim]`` gate.
    """
    if q_gate.ndim != 3 or k.ndim != 3:
        raise ValueError(f"expected 3D q_gate/k, got {q_gate.shape} and {k.shape}")
    if q_gate.shape[0] != k.shape[0]:
        raise ValueError("q_gate and k must have the same token count")
    if q_gate.shape[1:] != (num_q_heads, 2 * head_dim):
        raise ValueError(f"unexpected q_gate shape {tuple(q_gate.shape)}")
    if k.shape[1:] != (num_kv_heads, head_dim):
        raise ValueError(f"unexpected k shape {tuple(k.shape)}")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be an even value in 1..{head_dim}, got {rotary_dim}")
    if positions.numel() != q_gate.shape[0]:
        raise ValueError("positions must contain one entry per token")

    tokens = q_gate.shape[0]
    q_out = torch.empty(tokens, num_q_heads * head_dim, dtype=q_gate.dtype, device=q_gate.device)
    k_out = torch.empty(tokens, num_kv_heads * head_dim, dtype=k.dtype, device=k.device)
    gate_out = torch.empty(
        tokens, num_q_heads, head_dim, dtype=q_gate.dtype, device=q_gate.device
    )
    head_block = triton.next_power_of_2(head_dim)
    rotary_block = triton.next_power_of_2(rotary_dim // 2)
    _qk_rmsnorm_rope_gate_kernel[(tokens, num_q_heads + num_kv_heads)](
        q_gate,
        k,
        q_out,
        k_out,
        gate_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_gate.stride(0),
        q_gate.stride(1),
        k.stride(0),
        k.stride(1),
        q_out.stride(0),
        k_out.stride(0),
        gate_out.stride(0),
        gate_out.stride(1),
        cos_sin_cache.stride(0),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        half_rotary=rotary_dim // 2,
        head_block=head_block,
        rotary_block=rotary_block,
        eps=eps,
        fp16=q_gate.dtype == torch.float16,
        has_pass=rotary_dim < head_dim,
    )
    return q_out, k_out, gate_out


@triton.jit
def _sigmoid_mul_kernel(
    output_ptr,
    gate_ptr,
    gate_stride_token,
    gate_stride_head,
    hidden_dim: tl.constexpr,
    head_dim: tl.constexpr,
    block_h: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    offsets = block * block_h + tl.arange(0, block_h)
    mask = offsets < hidden_dim
    head = offsets // head_dim
    dim = offsets - head * head_dim
    output_offset = token * hidden_dim + offsets
    gate_offset = token * gate_stride_token + head * gate_stride_head + dim
    output = tl.load(output_ptr + output_offset, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + gate_offset, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + output_offset, output * tl.sigmoid(gate), mask=mask)


def fused_sigmoid_mul(attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply ``sigmoid(gate)`` to a contiguous ``[T, H*D]`` output in place."""
    if attn_output.ndim != 2 or gate.ndim != 3:
        raise ValueError(
            "expected output [T,H*D] and gate [T,H,D], "
            f"got {attn_output.shape} / {gate.shape}"
        )
    tokens, hidden_dim = attn_output.shape
    if gate.shape[0] != tokens or gate.shape[1] * gate.shape[2] != hidden_dim:
        raise ValueError("attention output and gate shapes are incompatible")
    block_h = 1024 if tokens < 1024 else 2048
    _sigmoid_mul_kernel[(tokens, triton.cdiv(hidden_dim, block_h))](
        attn_output,
        gate,
        gate.stride(0),
        gate.stride(1),
        hidden_dim=hidden_dim,
        head_dim=gate.shape[2],
        block_h=block_h,
        num_warps=4,
    )
    return attn_output
