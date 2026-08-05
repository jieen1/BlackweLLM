"""Triton fused NVFP4 activation quantizer — zero external dependency.

Single-pass block-16 NVFP4 quantization of a ``[M, K]`` BF16 activation:
e2m1 packed codes + e4m3 block scales, bit-identical to sparkinfer's own
oracle quantizer ``quantize_grouped_nvfp4_torch`` (same recipe: ``sf =
e4m3_rn(gs * amax16 / 6)``, ``code = e2m1_snap(x / (sf_deq / gs))``, zero
codes for zero scales, same e2m1 tie boundaries). Verified bit-for-bit
against that oracle at load time by
``tests/test_nvfp4_quant_triton.py`` (GPU) — the two are interchangeable
operands of ``sparkinfer.gemm.blockscaled.mm``.

Scales are returned LINEAR (``[M_padded, C_padded]`` e4m3, zero-padded to
the swizzle alignment); the caller applies sparkinfer's own
``swizzle_block_scale`` + ``as_grouped_scale_view`` so the operand layout
stays defined in exactly one place. The padding copy is a bounded-size
indexed write, not a second pass over the data.

The swizzle is permutation-only with zeros elsewhere, so zero-padding
BEFORE the swizzle (this file) is the same tensor as swizzling then
zero-padding (what ``quantize_grouped_nvfp4_torch`` + the operand builder
do today).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_FP4_MAG_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@triton.jit
def _rne_int(x):
    """round-to-nearest-even of a non-negative f32 to int32."""
    t = tl.floor(x)
    frac = x - t
    ti = t.to(tl.int32)
    return ti + (frac > 0.5).to(tl.int32) + (((frac == 0.5) & ((ti & 1) == 1)).to(tl.int32))


@triton.jit
def _e4m3_rn_byte(v):
    """f32 -> e4m3fn byte with round-to-nearest-even + satfinite, matching
    PTX ``cvt.rn.satfinite.e4m3x2.f32`` (and torch's float8_e4m3fn cast):
    normal path in integer bit arithmetic, subnormal path (|v| < 2^-6) via
    RNE of v*512 into the 3-bit mantissa."""
    v = tl.minimum(tl.maximum(v, -448.0), 448.0)
    bits = v.to(tl.int32, bitcast=True)
    sign = (bits >> 31) & 0x1
    mag = bits & 0x7FFFFFFF
    # --- normal path (|v| >= 2^-6): RNE on the 23-bit f32 mantissa ---
    lsb = (mag >> 20) & 0x1
    rounded = mag + 0x7FFFF + lsb
    exp8 = (rounded >> 23) & 0xFF
    mant = (rounded >> 20) & 0x7
    out_exp = exp8 - 127 + 7
    norm_byte = (sign << 7) | (out_exp << 3) | mant
    # exp field 15 with mant 7 is NaN in e4m3fn; satfinite maps it to max
    norm_byte = tl.where((out_exp == 15) & (mant == 7), (sign << 7) | 0x7E, norm_byte)
    # --- subnormal path (|v| < 2^-6): value = mant * 2^-9 ---
    mant_sub = _rne_int(tl.abs(v) * 512.0)
    sub_byte = (sign << 7) | mant_sub
    # mantissa carry 8 == smallest normal 2^-6 (byte 0x08)
    sub_byte = tl.where(mant_sub > 7, (sign << 7) | 0x08, sub_byte)
    byte = tl.where(mag < 0x3C800000, sub_byte, norm_byte)
    return byte.to(tl.uint8)


@triton.jit
def _e4m3_byte_to_f32_lut(byte):
    """e4m3fn byte -> f32 via a 256-entry LUT built bit-wise (no NaN entry
    is reachable: the quantizer only ever produces rn.satfinite bytes)."""
    sign = (byte.to(tl.uint16) >> 7) & 0x1
    exp4 = (byte.to(tl.uint16) >> 3) & 0xF
    mant = byte.to(tl.uint16) & 0x7
    # subnormal: value = mant * 2^-9 ; normal: (8+mant) * 2^(exp4-10)
    sub = mant.to(tl.float32) * 0.001953125
    norm = (8.0 + mant.to(tl.float32)) * tl.exp2(exp4.to(tl.float32) - 10.0)
    val = tl.where(exp4 == 0, sub, norm)
    return tl.where(sign == 1, -val, val)


@triton.jit
def _e2m1_snap_nibble(x):
    """|x| -> e2m1 magnitude index (0..7) with the oracle's exact tie
    boundaries, then sign bit -> nibble byte."""
    mag_idx = tl.zeros(x.shape, dtype=tl.int32)
    mag_idx = tl.where(x > 0.25, 1, mag_idx)
    mag_idx = tl.where(x >= 0.75, 2, mag_idx)
    mag_idx = tl.where(x > 1.25, 3, mag_idx)
    mag_idx = tl.where(x >= 1.75, 4, mag_idx)
    mag_idx = tl.where(x > 2.5, 5, mag_idx)
    mag_idx = tl.where(x >= 3.5, 6, mag_idx)
    mag_idx = tl.where(x > 5.0, 7, mag_idx)
    return mag_idx


@triton.jit
def _quantize_nvfp4_kernel(
    X_ptr,
    OUT_ptr,
    SF_ptr,
    GS_ptr,
    stride_x_row,
    stride_out_row,
    stride_sf_row,
    K: tl.constexpr,
    NBLK: tl.constexpr,
):
    gs = tl.load(GS_ptr).to(tl.float32)
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    # load as adjacent (even, odd) pairs: packed byte p covers columns 2p/2p+1
    pair_idx = tl.arange(0, NBLK // 2)[:, None] * 2 + tl.arange(0, 2)[None, :]
    pair_cols = pid_b * NBLK + tl.reshape(pair_idx, (NBLK,))
    offs = pid_m * stride_x_row + pair_cols
    x = tl.load(X_ptr + offs).to(tl.float32)
    xb = tl.reshape(x, (NBLK // 16, 16))  # same 16-column blocks as the oracle
    bmax = tl.max(tl.abs(xb), axis=1)
    # oracle order matters: gs * (bmax / 6), NOT (bmax * gs) / 6 -- the two
    # round differently and flip e4m3 ties at block boundaries.
    sf_f = gs * (bmax / 6.0)
    sf_byte = _e4m3_rn_byte(sf_f)
    sf_deq = _e4m3_byte_to_f32_lut(sf_byte.to(tl.uint16))
    # code = e2m1(x * inv) -- replicate the oracle's exact f32 arithmetic:
    # inv = 1 / (sf_deq * (1/gs)) with TWO rounded reciprocals (torch
    # `_reciprocal_or_zero_torch` chain), not one combined division; the
    # 1-ulp difference flips values sitting on e2m1 snap boundaries.
    rcp_gs = 1.0 / gs
    t = sf_deq * rcp_gs
    safe = t != 0.0
    inv = tl.where(safe, 1.0 / t, 1.0)
    xs = xb * inv[:, None]
    # the oracle's snap maps the |x|<=0.25 dead zone to +0.0 BEFORE the sign
    # is read (x * sign(x) == 0.0), so dead-zone values carry NO sign bit;
    # zero them here first or -tiny picks up a spurious sign nibble.
    xs = tl.where(tl.abs(xs) <= 0.25, 0.0, xs)
    idx = _e2m1_snap_nibble(tl.abs(xs))
    sign_bit = tl.where(xs < 0, 8, 0)
    nib = (idx | sign_bit).to(tl.uint8)
    nib = tl.where(safe[:, None], nib, 0)
    # pair up: last dim 2 -> tl.split gives (even, odd) nibbles per byte
    lo, hi = tl.split(tl.reshape(nib, (NBLK // 2, 2)))
    byte = lo | (hi << 4)
    out_offs = pid_m * stride_out_row + pid_b * (NBLK // 2) + tl.arange(0, NBLK // 2)
    tl.store(OUT_ptr + out_offs, byte)
    sf_offs = pid_m * stride_sf_row + pid_b * (NBLK // 16) + tl.arange(0, NBLK // 16)
    tl.store(SF_ptr + sf_offs, sf_byte)


def _align_up(v: int, a: int) -> int:
    return (v + a - 1) // a * a


def quantize_nvfp4_activation(
    x: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[M, K]`` BF16 ``x`` to NVFP4 block-16 codes + e4m3 scales.

    Returns ``(packed_uint8 [M, K/2], linear_scales [M_pad, K/16_pad]
    e4m3-as-uint8)``; feed the scales through sparkinfer's
    ``swizzle_block_scale`` + ``as_grouped_scale_view`` to build the
    ``blockscaled.mm`` operand. Bit-identical to
    ``quantize_grouped_nvfp4_torch`` for the same inputs (tested)."""
    if x.ndim != 2 or not x.is_contiguous():
        x = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x.shape
    if k % 32 != 0:
        raise ValueError(f"K must be a multiple of 32, got {k}")
    gs_t = global_scale.reshape(-1).to(torch.float32).contiguous()
    packed = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    sf_cols = k // 16
    sf = torch.zeros((_align_up(m, 128), _align_up(sf_cols, 4)), dtype=torch.uint8, device=x.device)
    sf_view = sf[:m, :sf_cols]
    nblk = 64
    while nblk > 16 and k // nblk < 4:
        nblk //= 2
    grid = (m, k // nblk)
    _quantize_nvfp4_kernel[grid](
        x,
        packed,
        sf_view,
        gs_t,
        x.stride(0),
        packed.stride(0),
        sf.stride(0),
        K=k,
        NBLK=nblk,
        num_warps=4,
    )
    return packed, sf
