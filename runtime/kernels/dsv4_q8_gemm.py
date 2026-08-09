"""Fused Q8_0 dequant-GEMM via tensor cores (native, no dequant cache).

The eager ``PackedQ8_0Linear`` dequantizes its full Q8_0 weight to bf16
every forward (598 dense modules, 6.8 GiB packed -- re-dequantizing that
is the dominant per-step elementwise cost, measured 209 ms/step on the
attention side).  Caching the bf16 dequant is forbidden (13.6 GiB
resident, the Qwen3.6 dequant-cache trap).  This kernel dequantizes the
Q8_0 weight in-register to a bf16 tile and feeds ``tl.dot`` (tensor
cores): Q8_0 block = [2] fp16 d + [32] int8 q, value = d * q.

Data flow (tensor-core friendly): for each K-block of 32 values, every
output column reads the SAME 32 int8 values (packed 34 bytes) once and
dequantizes to a [32, BLOCK_N] bf16 tile, then tl.dot with the
activation [BLOCK_M, 32] tile.  The weight stays packed (int8) -- never
materialized as bf16.  M=1 decode is a GEMV but still uses tensor cores.

Note: the eager ``dequantize_q8_0`` computes ``d * q`` in fp32 (d fp16
promoted, q int8 -> fp32).  This kernel computes the same product in
bf16 after rounding d*q to bf16 -- matching the ``weight_dtype=bfloat16``
production regime the reference uses for bf16-declared linears, which is
what the attention projections run at.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _q8_0_dequant_gemm_tc_kernel(
    x_ptr,  # [M, K] bf16 activations
    w_ptr,  # [out, in] Q8_0 packed, w_row_stride bytes/row
    out_ptr,  # [M, out] fp32
    M,
    K: tl.constexpr,
    OUT: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,  # packed bytes per out row = in/32*34
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLK_ELEMS: tl.constexpr,  # 32 (Q8_0 values per block)
    BLK_BYTES: tl.constexpr,  # 34 (Q8_0 packed bytes per block)
):
    """One program per (BLOCK_M token x BLOCK_N output tile)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLK_ELEMS):
        kb = k // BLK_ELEMS
        # activation [BLOCK_M, BLK_ELEMS] bf16 -> fp32
        a_u16 = (x_ptr + offs_m[:, None] * K + k + tl.arange(0, BLK_ELEMS)[None, :]).to(
            tl.pointer_type(tl.uint16)
        )
        a_tile = (tl.load(a_u16, mask=m_mask[:, None], other=0).to(tl.uint32) << 16).to(
            tl.float32, bitcast=True
        )
        # weight: every output col reads the same packed 34-byte block.
        # int8 qs at bytes 2..34: [BLOCK_N, BLK_ELEMS]
        q_ptrs = (
            w_ptr
            + offs_n[:, None] * W_ROW_STRIDE
            + kb * BLK_BYTES
            + 2
            + tl.arange(0, BLK_ELEMS)[None, :]
        )
        qs = tl.load(q_ptrs).to(tl.int8, bitcast=True).to(tl.float32)
        # fp16 d at bytes 0..1 (per output row, same for all 32 k)
        d_lo = tl.load(w_ptr + offs_n * W_ROW_STRIDE + kb * BLK_BYTES).to(tl.uint32)
        d_hi = tl.load(w_ptr + offs_n * W_ROW_STRIDE + kb * BLK_BYTES + 1).to(tl.uint32)
        d_u16 = d_lo.to(tl.uint16) | (d_hi.to(tl.uint16) << 8)
        d_f = d_u16.to(tl.float16, bitcast=True).to(tl.float32)  # [BLOCK_N]
        # weight tile [BLOCK_N, 32] (output x K), transposed to [32, BLOCK_N]
        # for the tl.dot A[BLOCK_M,32] @ B[32,BLOCK_N].
        w_tile = (qs * d_f[:, None]).to(tl.bfloat16)
        w_t = tl.trans(w_tile)  # [32, BLOCK_N]
        acc = tl.dot(a_tile.to(tl.bfloat16), w_t, acc)

    out_ptrs = out_ptr + offs_m[:, None] * OUT + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=m_mask[:, None])


def q8_0_dequant_gemm(
    x: torch.Tensor,
    packed: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    BLOCK_M: int = 16,
    BLOCK_N: int = 64,
) -> torch.Tensor:
    """``x @ W^T`` with W = Q8_0 packed [out, in], dequant-in-kernel."""
    M = x.shape[0]
    x = x.to(torch.bfloat16).contiguous()
    out = torch.empty((M, out_features), dtype=torch.float32, device=x.device)
    if in_features % 32:
        raise ValueError(f"in_features {in_features} is not a multiple of 32")
    w_row_stride = (in_features // 32) * 34
    assert (out_features * w_row_stride) == packed.numel(), (
        f"packed {packed.numel()} != out {out_features} x stride {w_row_stride}"
    )
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(out_features, BLOCK_N))
    _q8_0_dequant_gemm_tc_kernel[grid](
        x,
        packed,
        out,
        M,
        K=in_features,
        OUT=out_features,
        W_ROW_STRIDE=w_row_stride,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLK_ELEMS=32,
        BLK_BYTES=34,
    )
    return out
