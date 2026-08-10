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


def _select_q8_0_block_n(m: int, in_features: int, out_features: int) -> int:
    """Choose the measured SM120 tile for DSV4's fixed projection shapes."""
    if m <= 4 and (in_features, out_features) in ((4096, 64), (4096, 1024)):
        return 8
    if m <= 4 and (in_features, out_features) == (1024, 8192):
        return 32 if m == 2 else 16
    if m <= 4 and (in_features, out_features) in ((4096, 512), (8192, 4096)):
        return 16
    if in_features == 1024 and out_features == 32768:
        return 64
    if m > 4 and out_features >= 65536:
        return 64
    return 32


def _select_q8_0_block_m(m: int, in_features: int, out_features: int) -> int:
    """Choose the measured SM120 row tile for DSV4 decode projections."""
    if m <= 4 and (in_features, out_features) in (
        (1024, 8192),
        (4096, 512),
        (4096, 1024),
        (8192, 4096),
    ):
        return 8
    if m <= 2 and (in_features, out_features) == (4096, 64):
        return 8
    return 16


def _select_q8_0_grouped_block_n(rows_per_group: int) -> int:
    """Choose the measured SM120 tile for MLA's grouped output projection."""
    return 16 if rows_per_group <= 4 else 64


def _select_q8_0_grouped_block_m(rows_per_group: int) -> int:
    """Choose the measured SM120 row tile for decode's grouped MLA projection."""
    return 8 if rows_per_group <= 4 else 16


def _select_q8_0_fp32_block_cols(m: int, in_features: int, out_features: int) -> int:
    """Choose the exact-FP32 shared-expert tile measured on SM120."""
    if m in (1, 2, 4) and (in_features, out_features) in (
        (4096, 2048),
        (2048, 4096),
    ):
        return 8
    return 32


@triton.jit
def _q8_0_dequant_gemv_fp32_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    K: tl.constexpr,
    OUT: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    BLK_ELEMS: tl.constexpr,
    BLK_BYTES: tl.constexpr,
):
    """Q8_0 GEMV with the eager path's FP32 dequant/accumulation semantics."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_c * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    row_mask = rows < OUT
    elems = tl.arange(0, BLK_ELEMS)
    x_u16 = x_ptr.to(tl.pointer_type(tl.uint16))
    x_base = pid_m * K
    acc = tl.zeros((BLOCK_COLS,), dtype=tl.float32)

    for k in range(0, K, BLK_ELEMS):
        block = k // BLK_ELEMS
        packed_block = w_ptr + rows * W_ROW_STRIDE + block * BLK_BYTES
        d_lo = tl.load(packed_block, mask=row_mask, other=0).to(tl.uint32)
        d_hi = tl.load(packed_block + 1, mask=row_mask, other=0).to(tl.uint32)
        d_u16 = d_lo.to(tl.uint16) | (d_hi.to(tl.uint16) << 8)
        scale = d_u16.to(tl.float16, bitcast=True).to(tl.float32)
        quant = tl.load(
            packed_block[:, None] + 2 + elems[None, :],
            mask=row_mask[:, None],
            other=0,
        ).to(tl.int8, bitcast=True).to(tl.float32)
        activation = (
            tl.load(x_u16 + x_base + k + elems).to(tl.uint32) << 16
        ).to(tl.float32, bitcast=True)
        acc += tl.sum(quant * scale[:, None] * activation[None, :], axis=1)

    tl.store(out_ptr + pid_m * OUT + rows, acc, mask=row_mask)




def q8_0_dequant_gemv_fp32(
    x: torch.Tensor,
    packed: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    BLOCK_COLS: int | None = None,
) -> torch.Tensor:
    """Packed Q8_0 linear preserving FP32 weight-product semantics.

    This decode-oriented path avoids materializing the full FP32 weight while
    retaining the reference regime used by FP32-declared DSV4 linears.  It is
    intentionally separate from :func:`q8_0_dequant_gemm`, whose BF16 weight
    tile is substantially faster but is not numerically safe when repeated
    through every shared expert layer.  The fixed shared-expert shapes use a
    measured 8-column SM120 tile for B1/B2/B4; explicit overrides remain
    available for kernel qualification.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be rank 2, got shape {tuple(x.shape)}")
    if in_features % 32:
        raise ValueError(f"in_features {in_features} is not a multiple of 32")
    x = x.to(torch.bfloat16).contiguous()
    row_stride = (in_features // 32) * 34
    if out_features * row_stride != packed.numel():
        raise ValueError(
            f"packed {packed.numel()} != out {out_features} x stride {row_stride}"
        )
    if BLOCK_COLS is None:
        BLOCK_COLS = _select_q8_0_fp32_block_cols(
            int(x.shape[0]), in_features, out_features
        )
    out = torch.empty((x.shape[0], out_features), dtype=torch.float32, device=x.device)
    grid = (x.shape[0], triton.cdiv(out_features, BLOCK_COLS))
    _q8_0_dequant_gemv_fp32_kernel[grid](
        x,
        packed,
        out,
        x.shape[0],
        K=in_features,
        OUT=out_features,
        W_ROW_STRIDE=row_stride,
        BLOCK_COLS=BLOCK_COLS,
        BLK_ELEMS=32,
        BLK_BYTES=34,
    )
    return out


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
    n_mask = offs_n < OUT

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
        qs = tl.load(q_ptrs, mask=n_mask[:, None], other=0).to(tl.int8, bitcast=True).to(
            tl.float32
        )
        # fp16 d at bytes 0..1 (per output row, same for all 32 k)
        d_lo = tl.load(
            w_ptr + offs_n * W_ROW_STRIDE + kb * BLK_BYTES, mask=n_mask, other=0
        ).to(tl.uint32)
        d_hi = tl.load(
            w_ptr + offs_n * W_ROW_STRIDE + kb * BLK_BYTES + 1, mask=n_mask, other=0
        ).to(tl.uint32)
        d_u16 = d_lo.to(tl.uint16) | (d_hi.to(tl.uint16) << 8)
        d_f = d_u16.to(tl.float16, bitcast=True).to(tl.float32)  # [BLOCK_N]
        # weight tile [BLOCK_N, 32] (output x K), transposed to [32, BLOCK_N]
        # for the tl.dot A[BLOCK_M,32] @ B[32,BLOCK_N].
        w_tile = (qs * d_f[:, None]).to(tl.bfloat16)
        w_t = tl.trans(w_tile)  # [32, BLOCK_N]
        acc = tl.dot(a_tile.to(tl.bfloat16), w_t, acc)

    out_ptrs = out_ptr + offs_m[:, None] * OUT + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=m_mask[:, None] & n_mask[None, :])


def q8_0_dequant_gemm(
    x: torch.Tensor,
    packed: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    BLOCK_M: int | None = None,
    BLOCK_N: int | None = None,
) -> torch.Tensor:
    """``x @ W^T`` with W = Q8_0 packed [out, in], dequant-in-kernel."""
    M = x.shape[0]
    if BLOCK_M is None:
        BLOCK_M = _select_q8_0_block_m(M, in_features, out_features)
    if BLOCK_N is None:
        # Real-weight SM120 sweeps show that decode's narrow M=1 surface is
        # normally limited by the work carried by each output tile, while the
        # 1024->32768 query projection and large multi-row output head retain
        # enough parallel work to favour 64 columns.  Keep explicit overrides
        # available for kernel qualification tests.
        BLOCK_N = _select_q8_0_block_n(M, in_features, out_features)
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


@triton.jit
def _q8_0_grouped_dequant_gemm_tc_kernel(
    x_ptr,  # [G * ROWS_PER_G, K] bf16 activations, group-contiguous
    w_ptr,  # [G * R, in] Q8_0 packed, w_row_stride bytes/row
    out_ptr,  # [G * ROWS_PER_G, R] fp32
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  # R output rows per group
    ROWS_PER_G: tl.constexpr,  # activation rows per group (1 for decode)
    K: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLK_ELEMS: tl.constexpr,
    BLK_BYTES: tl.constexpr,
):
    """Grouped ``x[m, :] @ W[group(m), r, :]^T``.

    x rows are group-contiguous (``ROWS_PER_G`` rows of group 0, then
    group 1, ...).  A program covers ``BLOCK_M`` rows of ONE group, so the
    group index -- and therefore the weight-row base -- is a scalar, and a
    plain 2D ``tl.dot`` works.  Grid x = groups, grid y = row blocks
    within one group, and grid z = output-column blocks.
    """
    pid_g = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    local_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = pid_g * ROWS_PER_G + local_m
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = local_m < ROWS_PER_G
    n_mask = offs_n < GROUP_SIZE

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    w_base = pid_g * GROUP_SIZE  # scalar: this program's whole group

    for k in range(0, K, BLK_ELEMS):
        kb = k // BLK_ELEMS
        a_u16 = (x_ptr + offs_m[:, None] * K + k + tl.arange(0, BLK_ELEMS)[None, :]).to(
            tl.pointer_type(tl.uint16)
        )
        a_tile = (tl.load(a_u16, mask=m_mask[:, None], other=0).to(tl.uint32) << 16).to(
            tl.float32, bitcast=True
        )
        w_rows = w_base + offs_n  # [BLOCK_N] scalar base + vector
        q_ptrs = (
            w_ptr
            + w_rows[:, None] * W_ROW_STRIDE
            + kb * BLK_BYTES
            + 2
            + tl.arange(0, BLK_ELEMS)[None, :]
        )
        qs = tl.load(q_ptrs, mask=n_mask[:, None], other=0).to(
            tl.int8, bitcast=True
        ).to(tl.float32)
        d_lo = tl.load(
            w_ptr + w_rows * W_ROW_STRIDE + kb * BLK_BYTES,
            mask=n_mask,
            other=0,
        ).to(tl.uint32)
        d_hi = tl.load(
            w_ptr + w_rows * W_ROW_STRIDE + kb * BLK_BYTES + 1,
            mask=n_mask,
            other=0,
        ).to(tl.uint32)
        d_u16 = d_lo.to(tl.uint16) | (d_hi.to(tl.uint16) << 8)
        d_f = d_u16.to(tl.float16, bitcast=True).to(tl.float32)  # [BLOCK_N]
        w_tile = (qs * d_f[:, None]).to(tl.bfloat16)  # [BLOCK_N, 32]
        w_t = tl.trans(w_tile)  # [32, BLOCK_N]
        acc = tl.dot(a_tile.to(tl.bfloat16), w_t, acc)

    out_ptrs = out_ptr + offs_m[:, None] * GROUP_SIZE + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=m_mask[:, None] & n_mask[None, :])


def q8_0_grouped_dequant_gemm(
    x: torch.Tensor,
    packed: torch.Tensor,
    *,
    num_groups: int,
    group_size: int,
    in_features: int,
    rows_per_group: int = 1,
    BLOCK_M: int | None = None,
    BLOCK_N: int | None = None,
) -> torch.Tensor:
    """Grouped ``x[m, :] @ W[group(m), r, :]^T`` with W = Q8_0 packed.

    ``x`` is [num_groups * rows_per_group, K], group-contiguous (the
    MLA wo_a contraction ``o[bs, seq, g, d] @ wo_a[g, r, d]`` flattened).
    W is packed [num_groups * group_size, in].  Returns [M, group_size]
    fp32 -- the dequantized weight is never materialized.

    Triton has no batched tl.dot, so each group/row tile is handled by a
    2D dot program with a scalar weight-row base.
    """
    M = x.shape[0]
    assert M == num_groups * rows_per_group, (M, num_groups, rows_per_group)
    if BLOCK_M is None:
        # Real-weight SM120 decode sweeps over wo_a's grouped contraction show
        # that a narrower 8-row tile wins for B1/B2/B4 while remaining
        # bit-exact against the established 16x16 kernel. Prefill keeps the
        # existing 16-row tile.
        BLOCK_M = _select_q8_0_grouped_block_m(rows_per_group)
    if BLOCK_N is None:
        # Real-weight SM120 sweeps favour finer output tiling for decode's
        # B1/B2/B4 surface.  Wider prefill row groups amortize a 64-column
        # tile better.  Explicit overrides remain available for qualification.
        BLOCK_N = _select_q8_0_grouped_block_n(rows_per_group)
    x = x.to(torch.bfloat16).contiguous()
    out = torch.empty((M, group_size), dtype=torch.float32, device=x.device)
    if in_features % 32:
        raise ValueError(f"in_features {in_features} is not a multiple of 32")
    w_row_stride = (in_features // 32) * 34
    assert (num_groups * group_size * w_row_stride) == packed.numel(), (
        f"packed {packed.numel()} != groups {num_groups} x {group_size} x stride {w_row_stride}"
    )
    # Keep BLOCK_M power-of-two (required by tl.arange) and mask the final
    # row tile. A separate grid dimension covers every tile when prefill's
    # rows_per_group exceeds BLOCK_M; decode remains one masked tile.
    grid = (
        num_groups,
        triton.cdiv(rows_per_group, BLOCK_M),
        triton.cdiv(group_size, BLOCK_N),
    )
    _q8_0_grouped_dequant_gemm_tc_kernel[grid](
        x,
        packed,
        out,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
        ROWS_PER_G=rows_per_group,
        K=in_features,
        W_ROW_STRIDE=w_row_stride,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLK_ELEMS=32,
        BLK_BYTES=34,
    )
    return out


@triton.jit
def _q8_0_dequant_gemm_soa_tc_kernel(
    x_ptr,  # [M, K] bf16 activations
    q_ptr,  # [out, K] int8 code plane, row-contiguous (K bytes/row)
    d_ptr,  # [out, K/32] fp16 scale plane
    out_ptr,  # [M, out] fp32
    M,
    K: tl.constexpr,
    OUT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """SoA-layout Q8_0 dequant-GEMM: code plane separate from scale plane.

    The interleaved 34-byte Q8_0 block (2-byte fp16 d + 32 int8 q) makes the
    int8 code stream only 2-byte aligned, forcing 16-bit loads that underuse
    DRAM on cold M=1 decode (measured 103 vs 166 GB/s for a flat code plane).
    Splitting into a row-contiguous int8 code plane (K bytes/row) and a
    [out, K/32] fp16 scale plane lets the code loads be full-width.  Same
    arithmetic (fp32 accumulate, d*q, bf16 tl.dot), bit-exact vs interleaved.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < OUT
    x_u16 = x_ptr.to(tl.pointer_type(tl.uint16))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, 32):
        kb = k // 32
        a_u16 = x_u16 + offs_m[:, None] * K + k + tl.arange(0, 32)[None, :]
        a_tile = (tl.load(a_u16, mask=m_mask[:, None], other=0).to(tl.uint32) << 16).to(
            tl.float32, bitcast=True
        )
        q_ptrs = q_ptr + offs_n[:, None] * K + k + tl.arange(0, 32)[None, :]
        qs = tl.load(q_ptrs, mask=n_mask[:, None], other=0).to(tl.int8, bitcast=True).to(
            tl.float32
        )
        d = tl.load(d_ptr + offs_n * (K // 32) + kb, mask=n_mask, other=0).to(
            tl.float16
        ).to(tl.float32)
        w_tile = (qs * d[:, None]).to(tl.bfloat16)  # [BLOCK_N, 32]
        w_t = tl.trans(w_tile)  # [32, BLOCK_N]
        acc = tl.dot(a_tile.to(tl.bfloat16), w_t, acc)
    out_ptrs = out_ptr + offs_m[:, None] * OUT + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=m_mask[:, None] & n_mask[None, :])


def repack_q8_0_soa(
    packed: torch.Tensor, out_features: int, in_features: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split interleaved Q8_0 [out*in/32*34] into (code_plane, scale_plane).

    Returns (q [out, in] uint8 row-contiguous, d [out, in/32] fp16).  The
    code plane is K bytes per row (already 64B aligned for K>=64), so the
    kernel loads it with full-width reads instead of 16-bit pairs.
    """
    if in_features % 32:
        raise ValueError(f"in_features {in_features} is not a multiple of 32")
    wv = packed.view(out_features, in_features // 32, 34)
    q = wv[:, :, 2:].reshape(out_features, in_features).contiguous()
    d = wv[:, :, :2].view(torch.float16).squeeze(-1).contiguous()
    return q, d


def q8_0_dequant_gemm_soa(
    x: torch.Tensor,
    q: torch.Tensor,
    d: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    BLOCK_M: int | None = None,
    BLOCK_N: int | None = None,
) -> torch.Tensor:
    """SoA-layout ``x @ W^T`` (see _q8_0_dequant_gemm_soa_tc_kernel)."""
    M = x.shape[0]
    if BLOCK_M is None:
        BLOCK_M = _select_q8_0_block_m(M, in_features, out_features)
    if BLOCK_N is None:
        BLOCK_N = _select_q8_0_block_n(M, in_features, out_features)
    x = x.to(torch.bfloat16).contiguous()
    out = torch.empty((M, out_features), dtype=torch.float32, device=x.device)
    if q.shape != (out_features, in_features):
        raise ValueError(f"q must be [{out_features}, {in_features}], got {tuple(q.shape)}")
    if d.shape != (out_features, in_features // 32):
        raise ValueError(f"d must be [{out_features}, {in_features//32}], got {tuple(d.shape)}")
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(out_features, BLOCK_N))
    _q8_0_dequant_gemm_soa_tc_kernel[grid](
        x,
        q,
        d,
        out,
        M,
        K=in_features,
        OUT=out_features,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return out
