"""NVFP4 KV cache packing/unpacking toolchain for Qwen3.8 (S3 of the NVFP4
KV plan, notes/2026-08-16-nvfp4-kv-plan.md).

Pool layout (per (page, token, kv_head) row, head_dim=256 = 16 block-16
groups):

    codes  [.., 128] uint8  -- e2m1 nibble pairs, block-16 along head_dim
    scales [..,  16] uint8  -- e4m3 block scales (one per 16 elements)

The quantization recipe is exactly ``quantize_nvfp4_activation`` (bit-identical
to the oracle; ``tests/test_nvfp4_quant_triton.py``).  The unpack kernel
reconstructs ``x ~= code * sf_deq`` with the e2m1 magnitude table and a
hardware-exact e4m3 decode, emitting FP8 E4M3 values for the b12x read side.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from runtime.kernels.nvfp4_quant import quantize_nvfp4_activation

_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


@triton.jit
def _e4m3_byte_to_f32(byte):
    """e4m3fn byte -> f32 (same bit recipe as nvfp4_quant's LUT)."""
    b16 = byte.to(tl.uint16)
    sign = (b16 >> 7) & 0x1
    exp4 = (b16 >> 3) & 0xF
    mant = b16 & 0x7
    sub = mant.to(tl.float32) * 0.001953125
    norm = (8.0 + mant.to(tl.float32)) * tl.exp2(exp4.to(tl.float32) - 10.0)
    val = tl.where(exp4 == 0, sub, norm)
    return tl.where(sign == 1, -val, val)


@triton.jit
def _nvfp4_kv_unpack_kernel(
    CODES,
    SCALES,
    OUT,
    total_rows,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Unpack one (page, token, kv_head) row: 128 code bytes + 16 scale bytes
    -> HEAD_DIM fp8 values.  Grid = (total_rows,)."""
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)  # BLOCK == HEAD_DIM // 2 code bytes
    c = tl.load(CODES + row * BLOCK + col)
    lo = (c & 0x0F).to(tl.int32)
    hi = ((c >> 4) & 0x0F).to(tl.int32)
    nib = tl.reshape(tl.join(lo, hi), (2 * BLOCK,))
    mag = nib & 0x07
    sign = (nib >> 3) & 0x01
    mag_f = mag.to(tl.float32)
    val = tl.where(mag == 1, 0.5, tl.zeros_like(mag_f))
    val = tl.where(mag == 2, 1.0, val)
    val = tl.where(mag == 3, 1.5, val)
    val = tl.where(mag == 4, 2.0, val)
    val = tl.where(mag == 5, 3.0, val)
    val = tl.where(mag == 6, 4.0, val)
    val = tl.where(mag == 7, 6.0, val)
    val = val * (1.0 - 2.0 * sign.to(tl.float32))
    sf_idx = tl.arange(0, HEAD_DIM) // 16
    sf = tl.load(SCALES + row * (HEAD_DIM // 16) + sf_idx)
    sf_deq = _e4m3_byte_to_f32(sf)
    out = (val * sf_deq).to(tl.float8e4nv)
    tl.store(OUT + row * HEAD_DIM + tl.arange(0, HEAD_DIM), out)


def unpack_nvfp4_kv(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    head_dim: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    """Unpack a [R, head_dim/2] codes pool + [R, head_dim/16] scales pool
    into an FP8 [R, head_dim] tensor (grid over rows)."""
    total_rows = codes.shape[0]
    assert scales.shape[0] == total_rows
    assert codes.shape[1] == head_dim // 2
    assert scales.shape[1] == head_dim // 16
    out = torch.empty((total_rows, head_dim), dtype=torch.float8_e4m3fn, device=codes.device)
    _nvfp4_kv_unpack_kernel[(total_rows,)](
        codes,
        scales,
        out,
        total_rows,
        HEAD_DIM=head_dim,
        BLOCK=head_dim // 2,
        num_warps=num_warps,
    )
    return out


def pack_nvfp4_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    head_dim: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ``key``/``value`` ([M, kv_heads, head_dim] BF16) into the pool
    row layout: returns ``(k_codes, k_scales, v_codes, v_scales)`` each
    [M * kv_heads, head_dim/2 or /16] uint8."""
    flat_k = key.reshape(-1, head_dim)
    flat_v = value.reshape(-1, head_dim)
    gs = torch.ones(1, dtype=torch.float32, device=key.device)
    k_codes, k_sf = quantize_nvfp4_activation(flat_k, gs)
    v_codes, v_sf = quantize_nvfp4_activation(flat_v, gs)
    rows = flat_k.shape[0]
    k_scales = k_sf[:rows, : head_dim // 16].contiguous()
    v_scales = v_sf[:rows, : head_dim // 16].contiguous()
    return k_codes, k_scales, v_codes, v_scales


def pack_nvfp4_kv_into_pool(
    key: torch.Tensor,
    value: torch.Tensor,
    k_codes_pool: torch.Tensor,
    k_scales_pool: torch.Tensor,
    v_codes_pool: torch.Tensor,
    v_scales_pool: torch.Tensor,
    write_rows: torch.Tensor,
    *,
    head_dim: int = 256,
) -> None:
    """Quantize ``key``/``value`` ([total_q, kv_heads, head_dim]) and write
    the pool row layout ``[num_rows, kv_heads, head_dim/2]`` / ``[/16]`` at
    ``write_rows`` (one flat row per token; head axis inside the row)."""
    kv_heads = key.shape[1]
    k_codes, k_scales, v_codes, v_scales = pack_nvfp4_kv(key, value, head_dim=head_dim)
    rows = write_rows.shape[0]
    # Pools are [num_pages, page_size, kv_heads, ...]; write_index rows are
    # global flattened rows, so index through the flattened view.
    k_codes_pool.view(-1, kv_heads, head_dim // 2)[write_rows] = k_codes.view(
        rows, kv_heads, head_dim // 2
    )
    k_scales_pool.view(-1, kv_heads, head_dim // 16)[write_rows] = k_scales.view(
        rows, kv_heads, head_dim // 16
    )
    v_codes_pool.view(-1, kv_heads, head_dim // 2)[write_rows] = v_codes.view(
        rows, kv_heads, head_dim // 2
    )
    v_scales_pool.view(-1, kv_heads, head_dim // 16)[write_rows] = v_scales.view(
        rows, kv_heads, head_dim // 16
    )
