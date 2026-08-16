"""Triton NVFP4-KV decode attention for Qwen3.8 (S2 of the NVFP4 KV plan).

Replaces the b12x paged decode path when ``QSR_QWEN36_NVFP4_KV=1``: reads
the packed nvfp4 KV pools (codes + e4m3 block scales), unpacks in-kernel,
and runs split-KV flash-attention decode with a two-pass merge.

Geometry (Qwen3.8-27B): 24 q heads / 4 kv heads (GQA 6), head_dim 256,
page_size 32 tokens.  KV pools are row-major over physical rows
(``[num_rows, kv_heads, 128]`` codes + ``[num_rows, kv_heads, 16]`` scales).

Kernel 1 (partial): grid = (N_SPLIT, NUM_KV_HEADS, batch).  One program =
one request's KV segment for ONE kv head: unpacks the segment's K/V once,
runs QK (the GQA group's 6 q heads) + online softmax + PV, emits
(m, l, acc) partials.
Kernel 2 (merge): grid = (batch * 24,).  Combines the N_SPLIT * 4 partials
per (request, q head) into the final output.

SMEM budget: QK B operand [BLOCK_K, 256] fp8 and PV B operand [BLOCK_K, 256]
fp8 at BLOCK_K=128 -> 32 KiB each, fits SM120's 99 KiB.
"""

from __future__ import annotations
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3_byte_to_f32(byte):
    b16 = byte.to(tl.uint16)
    sign = (b16 >> 7) & 0x1
    exp4 = (b16 >> 3) & 0xF
    mant = b16 & 0x7
    sub = mant.to(tl.float32) * 0.001953125
    norm = (8.0 + mant.to(tl.float32)) * tl.exp2(exp4.to(tl.float32) - 10.0)
    val = tl.where(exp4 == 0, sub, norm)
    return tl.where(sign == 1, -val, val)


@triton.jit
def _unpack_kv_half(
    CODES,
    SCALES,
    phys,
    pid_h,
    H,
    half_start,
    row_mask,
    D: tl.constexpr,
    BK: tl.constexpr,
):
    """Unpack one half (D/2 columns) of a [BK, D] kv-head row into BF16.

    Column-split keeps the f32 where-chain intermediates at [BK, D/2] so
    BLOCK_K=64 fits SM120's 99 KiB smem (a full [64, 256] f32 tile is
    64 KiB by itself)."""
    # one half is D/2 elements = D/4 code bytes; nibble-unpack to D/2 elems
    byte_cols = tl.arange(0, D // 4)
    c_off = phys[:, None] * (H * (D // 2)) + pid_h * (D // 2) + (half_start // 2 + byte_cols)[None, :]
    kc = tl.load(CODES + c_off, mask=row_mask[:, None], other=0)
    lo = (kc & 0x0F).to(tl.int32)
    hi = ((kc >> 4) & 0x0F).to(tl.int32)
    nib = tl.reshape(tl.join(lo, hi), (BK, D // 2))
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
    sf_idx = (half_start + tl.arange(0, D // 2)) // 16
    sf_off = phys[:, None] * (H * (D // 16)) + pid_h * (D // 16) + sf_idx[None, :]
    sf = tl.load(SCALES + sf_off, mask=row_mask[:, None], other=0)
    sf_deq = _e4m3_byte_to_f32(sf)
    return (val * sf_deq).to(tl.bfloat16)


@triton.jit
def nvfp4_decode_partial_kernel(
    Q,
    K_CODES,
    K_SCALES,
    V_CODES,
    V_SCALES,
    OUT_M,
    OUT_L,
    OUT_ACC,
    row_offsets,
    cache_seqlens,
    query_to_request,
    query_positions,
    N_SPLIT,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PADDED_GQA: tl.constexpr,
):
    """One (request, kv head, KV segment) partial flash decode."""
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    req = tl.load(query_to_request + pid_q)
    cache_len = tl.load(cache_seqlens + req)
    q_pos = tl.load(query_positions + pid_q)
    page0 = tl.load(row_offsets + req)
    seg_len = cache_len // N_SPLIT
    seg_start = pid_s * seg_len
    seg_end = tl.minimum(seg_start + seg_len, cache_len)
    if seg_start >= seg_end:
        return

    # this program's GQA query heads: [pid_h*GQA, pid_h*GQA+GQA), padded
    q_rows = tl.arange(0, PADDED_GQA)
    q_rmask = q_rows < GQA
    q_off = pid_q * NUM_Q_HEADS * HEAD_DIM + (pid_h * GQA + q_rows)[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
    q = tl.load(Q + q_off, mask=q_rmask[:, None], other=0.0)  # [PADDED_GQA, D] bf16

    m_i = tl.full((PADDED_GQA, 1), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((PADDED_GQA, 1), dtype=tl.float32)
    acc = tl.zeros((PADDED_GQA, HEAD_DIM), dtype=tl.float32)

    scale = 1.0 / (HEAD_DIM ** 0.5)
    for k_start in range(seg_start, seg_end, BLOCK_K):
        n = tl.minimum(k_start + BLOCK_K, seg_end) - k_start
        rows = tl.arange(0, BLOCK_K)
        row_mask = rows < n
        phys = page0 + k_start + rows
        # ---- unpack K for this kv head in column halves ----
        k_lo = _unpack_kv_half(
            K_CODES, K_SCALES, phys, pid_h, NUM_KV_HEADS, 0, row_mask, HEAD_DIM, BLOCK_K
        )
        k_hi = _unpack_kv_half(
            K_CODES, K_SCALES, phys, pid_h, NUM_KV_HEADS, HEAD_DIM // 2, row_mask,
            HEAD_DIM, BLOCK_K,
        )
        k = tl.reshape(tl.permute(tl.join(k_lo, k_hi), (0, 2, 1)), (BLOCK_K, HEAD_DIM))
        # ---- unpack V ----
        # ---- unpack V in column halves ----
        v_lo = _unpack_kv_half(
            V_CODES, V_SCALES, phys, pid_h, NUM_KV_HEADS, 0, row_mask, HEAD_DIM, BLOCK_K
        )
        v_hi = _unpack_kv_half(
            V_CODES, V_SCALES, phys, pid_h, NUM_KV_HEADS, HEAD_DIM // 2, row_mask,
            HEAD_DIM, BLOCK_K,
        )
        v = tl.reshape(tl.permute(tl.join(v_lo, v_hi), (0, 2, 1)), (BLOCK_K, HEAD_DIM))

        # ---- QK: [GQA, D] x [D, BLOCK_K] (bf16 dot: SM120 tl.dot has no
        # fp8 LHS) ----
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
        causal = (k_start + rows) <= q_pos
        qk = tl.where(row_mask[None, :] & causal[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1, keep_dims=True))
        alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
        p = tl.exp2((qk - m_new) * 1.4426950408889634)
        l_i = l_i * alpha + tl.sum(p, axis=1, keep_dims=True)
        pv = tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
        acc = acc * alpha + pv
        m_i = m_new

    # write partials (real GQA rows only; slice pads with 0 rows, stored
    # positions are the real heads)
    store_q = tl.arange(0, PADDED_GQA)
    store_mask = store_q < GQA
    base = pid_q * (N_SPLIT * NUM_KV_HEADS * GQA) + pid_h * (N_SPLIT * GQA) + pid_s * GQA
    tl.store(OUT_M + base + store_q, tl.reshape(m_i, (PADDED_GQA,)), mask=store_mask)
    tl.store(OUT_L + base + store_q, tl.reshape(l_i, (PADDED_GQA,)), mask=store_mask)
    acc_off = (pid_q * (N_SPLIT * NUM_KV_HEADS * GQA * HEAD_DIM)
               + pid_h * (N_SPLIT * GQA * HEAD_DIM)
               + pid_s * (GQA * HEAD_DIM)
               + store_q[:, None] * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :])
    tl.store(OUT_ACC + acc_off, acc, mask=store_mask[:, None])


@triton.jit
def nvfp4_decode_merge_kernel(
    OUT_M,
    OUT_L,
    OUT_ACC,
    OUT,
    N_SPLIT: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA: tl.constexpr,
):
    pid = tl.program_id(0)  # request * NUM_Q_HEADS + global q head
    r = pid // NUM_Q_HEADS
    qh = pid % NUM_Q_HEADS
    kh = qh // GQA
    ql = qh % GQA
    nparts = N_SPLIT * NUM_KV_HEADS
    m = tl.load(OUT_M + r * nparts * GQA + kh * N_SPLIT * GQA + ql + tl.arange(0, N_SPLIT) * GQA)
    l = tl.load(OUT_L + r * nparts * GQA + kh * N_SPLIT * GQA + ql + tl.arange(0, N_SPLIT) * GQA)
    acc = tl.load(OUT_ACC + r * nparts * GQA * HEAD_DIM + kh * N_SPLIT * GQA * HEAD_DIM + ql * HEAD_DIM + tl.arange(0, N_SPLIT)[:, None] * GQA * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :])
    m_max = tl.max(m)
    scale = tl.exp2((m - m_max) * 1.4426950408889634)
    lsum = tl.sum(l * scale)
    acc_sum = tl.sum(acc * scale[:, None], axis=0)
    out = acc_sum / lsum
    tl.store(OUT + r * NUM_Q_HEADS * HEAD_DIM + qh * HEAD_DIM + tl.arange(0, HEAD_DIM), out.to(tl.bfloat16))


def nvfp4_decode_attention(
    q: torch.Tensor,
    k_codes: torch.Tensor,
    k_scales: torch.Tensor,
    v_codes: torch.Tensor,
    v_scales: torch.Tensor,
    row_offsets: torch.Tensor,
    cache_seqlens: torch.Tensor,
    query_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    num_q_heads: int = 24,
    num_kv_heads: int = 4,
    head_dim: int = 256,
    gqa: int = 6,
    n_split: int = 32,
    block_k: int = 64,
    num_warps: int = 8,
) -> torch.Tensor:
    """NVFP4-KV flash decode attention (partial + merge).

    ``q`` is ``[total_q, num_q_heads, head_dim]`` BF16 (request-major rows).
    KV pools are ``[num_rows, kv_heads, head_dim/2]`` codes and
    ``[num_rows, kv_heads, head_dim/16]`` scales.  ``row_offsets`` is
    ``[num_requests]`` with each request's first physical KV row (its page
    table's first page id times the page size).  Returns
    ``[total_q, num_q_heads, head_dim]`` BF16.
    """
    total_q = q.shape[0]
    padded_gqa = max(8, (gqa + 7) // 8 * 8)
    n_parts = n_split * num_kv_heads
    out_m = torch.full((total_q, n_split, num_kv_heads, gqa), -float("inf"), device=q.device)
    out_l = torch.zeros_like(out_m)
    out_acc = torch.zeros((total_q, n_split, num_kv_heads, gqa, head_dim), device=q.device)
    out = torch.empty((total_q, num_q_heads, head_dim), dtype=torch.bfloat16, device=q.device)
    if os.environ.get("QSR_NVFP4_ATTN_DEBUG", "0") == "1":
        import time as _t

        _t0 = _t.perf_counter()
    nvfp4_decode_partial_kernel[(n_split, num_kv_heads, total_q)](
        q, k_codes, k_scales, v_codes, v_scales, out_m, out_l, out_acc,
        row_offsets, cache_seqlens, query_to_request, query_positions, n_split,
        NUM_Q_HEADS=num_q_heads, NUM_KV_HEADS=num_kv_heads, HEAD_DIM=head_dim,
        GQA=gqa, BLOCK_K=block_k, PADDED_GQA=padded_gqa, num_warps=num_warps,
    )
    nvfp4_decode_merge_kernel[(total_q * num_q_heads,)](
        out_m, out_l, out_acc, out, n_split, num_kv_heads,
        NUM_Q_HEADS=num_q_heads, HEAD_DIM=head_dim, GQA=gqa, num_warps=4,
    )
    if os.environ.get("QSR_NVFP4_ATTN_DEBUG", "0") == "1":
        torch.cuda.synchronize()
        print(
            f"[nvfp4-attn] total_q={total_q} partial={(_t.perf_counter()-_t0)*1e3:.2f}ms",
            flush=True,
        )
    return out
