"""Fused Q8_0 dequant-GEMM for the DSV4 dense projections (Phase 5).

The attention/HC/MoE-shared projections (``PackedQ8_0Linear``) dequantize
their full Q8_0 weight to bf16/fp32 every forward, then run ``F.linear``.
At M=1 decode that dequant is the dominant per-layer cost (measured:
wq_a+wq_b 0.59 ms, wkv 0.22 ms per layer -- the dequant, not the
matmul).  This kernel dequantizes in-register while accumulating the
GEMM, so no bf16/fp32 weight tensor is materialized.

Q8_0 block (34 bytes, 32 values): [2] fp16 d + [32] int8 q; value =
d * q (bit-exact with dequantize_q8_0).  One program per (token,
BLOCK_COLS output columns); the weight is [out, in] row-major packed
with w_row_stride bytes per out row.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_Q8_0_BLOCK_BYTES = 34
_Q8_0_BLOCK_ELEMS = 32


@triton.jit
def _q8_0_dequant_gemm_kernel(
    x_ptr,  # [M, K] bf16 activations
    w_ptr,  # [out, in] Q8_0 packed, w_row_stride bytes/row
    out_ptr,  # [M, out] fp32
    M,
    K: tl.constexpr,
    OUT: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,  # packed bytes per out row = in/32*34
    BLOCK_COLS: tl.constexpr,
    BLK_ELEMS: tl.constexpr = 32,
    BLK_BYTES: tl.constexpr = 34,
):
    """One program per (token, BLOCK_COLS output columns)."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    col0 = pid_c * BLOCK_COLS

    x_u16 = x_ptr.to(tl.pointer_type(tl.uint16))
    x_base = pid_m * K

    crow = col0 + tl.arange(0, BLOCK_COLS)
    acc = tl.zeros((BLOCK_COLS,), dtype=tl.float32)
    c32 = tl.arange(0, BLK_ELEMS)

    for kb in range(0, K, BLK_ELEMS):
        kb_block = kb // BLK_ELEMS
        col_byte = crow * W_ROW_STRIDE + kb_block * BLK_BYTES  # [BC]
        base = w_ptr + col_byte  # [BC]
        # d: fp16 at bytes 0..1
        d_lo = tl.load(base).to(tl.uint32)
        d_hi = tl.load(base + 1).to(tl.uint32)
        d_u16 = d_lo.to(tl.uint16) | (d_hi.to(tl.uint16) << 8)
        d_f = d_u16.to(tl.float16, bitcast=True).to(tl.float32)  # [BC]
        # qs: int8 at bytes 2..34
        qs = tl.load(base[:, None] + 2 + c32[None, :]).to(tl.int8, bitcast=True).to(tl.float32)
        # x values for this K-block
        xv = (tl.load(x_u16 + x_base + kb + c32).to(tl.uint32) << 16).to(
            tl.float32, bitcast=True
        )
        # out[bc] += sum_{32} x * (d * q)
        acc += tl.sum(qs * d_f[:, None] * xv[None, :], axis=1)

    tl.store(out_ptr + pid_m * OUT + crow, acc)


def q8_0_dequant_gemm(
    x: torch.Tensor,
    packed: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    BLOCK_COLS: int = 64,
) -> torch.Tensor:
    """``x @ W^T`` with W = Q8_0 packed [out, in], dequant in-kernel."""
    M = x.shape[0]
    x = x.to(torch.bfloat16).contiguous()
    out = torch.empty((M, out_features), dtype=torch.float32, device=x.device)
    if in_features % _Q8_0_BLOCK_ELEMS:
        raise ValueError(
            f"in_features {in_features} is not a multiple of the "
            f"{_Q8_0_BLOCK_ELEMS}-value Q8_0 block"
        )
    w_row_stride = (in_features // _Q8_0_BLOCK_ELEMS) * _Q8_0_BLOCK_BYTES
    assert (out_features * w_row_stride) == packed.numel(), (
        f"packed {packed.numel()} != out {out_features} x stride {w_row_stride}"
    )
    _q8_0_dequant_gemm_kernel[(M, out_features // BLOCK_COLS)](
        x,
        packed,
        out,
        M,
        K=in_features,
        OUT=out_features,
        W_ROW_STRIDE=w_row_stride,
        BLOCK_COLS=BLOCK_COLS,
        BLK_ELEMS=32,
        BLK_BYTES=34,
    )
    return out
