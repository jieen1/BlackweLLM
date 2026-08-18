"""Fused FP8 KV cache scatter: one kernel replaces 6 separate ops.

Eliminates per-layer kernel overhead in CUDA Graph capture:
  Old: slot_mapping//bs → slot_mapping%bs → key/scale → .to(fp8) → scatterK → scatterV
  New: single fused kernel per layer
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kv_scatter_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    k_scale_ptr,
    v_scale_ptr,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_cb,
    stride_cs,
    stride_chh,
    stride_cdd,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    hid = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + pid)
    if slot < 0:
        # Padding token (CUDA-graph fixed-shape batches use slot=-1 to mark
        # positions that should be skipped). Matches vLLM's own
        # triton_reshape_and_cache_flash reference kernel.
        return
    block_idx = slot // block_size
    block_off = slot % block_size

    k_scale = tl.load(k_scale_ptr)
    v_scale = tl.load(v_scale_ptr)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    k_val = tl.load(
        key_ptr + pid * stride_kt + hid * stride_kh + offs_d * stride_kd,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)
    v_val = tl.load(
        value_ptr + pid * stride_vt + hid * stride_vh + offs_d * stride_vd,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)

    k_fp8 = (k_val / k_scale).to(tl.float8e4nv)
    v_fp8 = (v_val / v_scale).to(tl.float8e4nv)

    cache_base = block_idx * stride_cb + block_off * stride_cs + hid * stride_chh
    tl.store(
        k_cache_ptr + cache_base + offs_d * stride_cdd,
        k_fp8,
        mask=mask_d,
    )
    tl.store(
        v_cache_ptr + cache_base + offs_d * stride_cdd,
        v_fp8,
        mask=mask_d,
    )


@triton.jit
def _fused_kv_scatter_row_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    k_scale_ptr,
    v_scale_ptr,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_cb,
    stride_cs,
    stride_chh,
    stride_cdd,
    num_kv_heads,
    head_dim,
    block_size,
    total_cols,
    BLOCK_COLS: tl.constexpr,
):
    """Write one complete flattened KV row per program.

    SGLang's ``store_cache`` path also treats one token's KV vector as the
    unit of work.  The original local kernel used one program per
    ``(token, kv_head)``; that is a poor match for Qwen3.8's 4x256 geometry
    during long prefill/injection.  Keeping the head index in the flattened
    offset preserves arbitrary strides while reducing the launch grid from
    ``tokens * kv_heads`` to ``tokens`` for the production shape.
    """

    token = tl.program_id(0)
    chunk = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + token)
    if slot < 0:
        return

    cols = chunk * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = cols < total_cols
    head = cols // head_dim
    dim = cols - head * head_dim

    k_scale = tl.load(k_scale_ptr)
    v_scale = tl.load(v_scale_ptr)
    k_val = tl.load(
        key_ptr
        + token * stride_kt
        + head * stride_kh
        + dim * stride_kd,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    v_val = tl.load(
        value_ptr
        + token * stride_vt
        + head * stride_vh
        + dim * stride_vd,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    k_fp8 = (k_val / k_scale).to(tl.float8e4nv)
    v_fp8 = (v_val / v_scale).to(tl.float8e4nv)
    block_idx = slot // block_size
    block_off = slot - block_idx * block_size
    cache_base = block_idx * stride_cb + block_off * stride_cs
    k_dst = cache_base + head * stride_chh + dim * stride_cdd
    v_dst = cache_base + head * stride_chh + dim * stride_cdd
    tl.store(k_cache_ptr + k_dst, k_fp8, mask=mask)
    tl.store(v_cache_ptr + v_dst, v_fp8, mask=mask)


def fused_kv_scatter(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """Fused FP8 KV cache write. CG-compatible.

    key/value: [num_tokens, num_kv_heads, head_dim] (bf16)
    k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim] (fp8/uint8)
    slot_mapping: [num_tokens] (int64)
    """
    num_tokens = key.shape[0]
    num_kv_heads = key.shape[1]
    head_dim = key.shape[2]
    block_size = k_cache.shape[1]

    if k_cache.dtype == torch.uint8:
        k_cache = k_cache.view(torch.float8_e4m3fn)
        v_cache = v_cache.view(torch.float8_e4m3fn)

    # Long prefill and DSpark hidden-KV injection use a dense Qwen geometry
    # (4 KV heads x 256 dimensions).  Match SGLang's row-oriented writer for
    # that regime; M=1 decode and unusual layouts retain the established
    # head-oriented kernel because its smaller launch footprint is better
    # there and it is already covered by the existing correctness tests.
    total_cols = num_kv_heads * head_dim
    if num_tokens >= 128 and total_cols <= 1024:
        block_cols = 1024
        _fused_kv_scatter_row_kernel[
            (num_tokens, triton.cdiv(total_cols, block_cols))
        ](
            key,
            value,
            k_cache,
            v_cache,
            slot_mapping,
            k_scale,
            v_scale,
            key.stride(0),
            key.stride(1),
            key.stride(2),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            num_kv_heads,
            head_dim,
            block_size,
            total_cols,
            BLOCK_COLS=block_cols,
            num_warps=8,
            num_stages=2,
        )
        return

    BLOCK_D = triton.next_power_of_2(head_dim)

    _fused_kv_scatter_kernel[(num_tokens, num_kv_heads)](
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        k_scale,
        v_scale,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        BLOCK_D=BLOCK_D,
    )
