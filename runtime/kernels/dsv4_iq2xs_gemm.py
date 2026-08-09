"""Fused IQ2_XS dequant-GEMM for the DSV4 MoE routed experts (Phase 5).

The eager MoE dequantizes every routed expert to fp32 before each matmul
(measured 0.67 ms per 2048x4096 expert dequant vs 0.04 ms for the M=1
matmul -- the dequant is ~95% of the MoE step cost).  This kernel reads
the packed IQ2_XS bytes and dequantizes in-register while accumulating
the GEMM, so no fp32 weight tensor is ever materialized -- the native
path the roadmap requires.

Layout: each expert is a [rows=inter, cols=hidden] matrix stored as
``rows * (cols/256)`` 74-byte IQ2_XS blocks, row-major (the same byte
order ``dequantize_iq2_xs(...).reshape(rows, cols)`` consumes).  The
MoE computes ``xs @ W^T`` (W is [inter, hidden], output [M, inter]): a
program owns a tile of output columns (inter dim) and sums over the
hidden dim, dequantizing the hidden row of each owned output column
in-register.

IQ2_XS block (74 bytes, 256 values = 8 sub-blocks of 32):
    [2]  fp16 d
    [64] 32 x int16 codes
    [8]  8 scale bytes (lo nibble -> sub-blocks 0,1; hi -> 2,3 of each
         32-group)
Dequant per code c (mirrors dequantize_iq2_xs bit-for-bit):
    grid[code & 511]        -> 8 magnitudes (sub-block j = bits 8j..8j+7)
    ksigns[code >> 9]       -> sign byte; kmask bit j negates sub-block j
    delta_j = d*(0.5+nibble)*0.25, group = sub-block j // 4 (lo/hi nibble)
    value = sign * magnitude * delta
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_IQ2_XS_BLOCK_BYTES = 74
_IQ2_XS_BLOCK_ELEMS = 256


@triton.jit
def _iq2xs_dequant_gemm_batch_kernel(
    x_ptr,  # [E, M, K] bf16 activations (routed expert x tokens x hidden)
    w_ptr,  # [E, rows, cols] IQ2_XS packed, row-major, w_row_stride bytes
    grid_ptr,  # [512] int64 magnitudes (8 packed per entry)
    ksigns_ptr,  # [128] int32 sign bytes
    out_ptr,  # [E, M, rows] fp32 out
    E,
    M,
    K: tl.constexpr,  # hidden 4096
    ROWS: tl.constexpr,  # inter 2048 (output rows of W^T = output cols)
    W_ROW_STRIDE: tl.constexpr,  # packed bytes per W row = cols/256*74
    BLOCK_COLS: tl.constexpr,  # output columns (ROWS dim) per program
    BLK_ELEMS: tl.constexpr,  # 256 values per IQ2_XS block
    BLK_BYTES: tl.constexpr,  # 74 bytes per IQ2_XS block
):
    """One program per (expert, token, BLOCK_COLS output columns)."""
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    col0 = pid_c * BLOCK_COLS

    # activation row base in bf16 (raw-bytes bitcast, per dsv4_mhc)
    x_u16 = x_ptr.to(tl.pointer_type(tl.uint16))
    x_base = pid_e * M * K + pid_m * K

    crows = col0 + tl.arange(0, BLOCK_COLS)  # [BC]
    acc = tl.zeros((BLOCK_COLS,), dtype=tl.float32)

    # Roll over the hidden dim in 256-value blocks.
    for kb in range(0, K, BLK_ELEMS):
        kb_block = kb // BLK_ELEMS
        # per-block byte offset for each owned output column
        col_byte = (
            pid_e * ROWS * W_ROW_STRIDE
            + crows * W_ROW_STRIDE
            + kb_block * BLK_BYTES
        )  # [BC]
        base = w_ptr + col_byte  # [BC]

        # 32 int16 codes at offset 2..66 (64 bytes)
        c32 = tl.arange(0, 32)
        lo = tl.load(base[:, None] + 2 + c32[None, :] * 2).to(tl.uint32)
        hi = tl.load(base[:, None] + 2 + c32[None, :] * 2 + 1).to(tl.uint32)
        codes = (lo | (hi << 8)) & 0xFFFF  # [BLOCK_COLS, 32]

        # grid[code & 511] -> int64 with 8 magnitudes (sub-block j = bits 8j..)
        g = tl.load(grid_ptr + (codes & 511))  # [BLOCK_COLS, 32]
        # signs from ksigns[code >> 9]
        sb = tl.load(ksigns_ptr + (codes >> 9))  # [BLOCK_COLS, 32]

        # d (fp16) at bytes 0..1
        d_lo = tl.load(base).to(tl.uint32)  # byte 0
        d_hi = tl.load(base + 1).to(tl.uint32)  # byte 1
        d_u16 = d_lo.to(tl.uint16) | (d_hi.to(tl.uint16) << 8)
        d_f = d_u16.to(tl.float16, bitcast=True).to(tl.float32)  # [BLOCK_COLS]

        # dequantize 256 values: [BC, 32 codes, 8 sub-blocks].  Accumulate
        # directly into the GEMM: out[bc] += sum_{256} x * value.
        for j in tl.static_range(8):
            magj = ((g >> (8 * j)) & 0xFF).to(tl.float32)  # [BC, 32]
            signj = tl.where((sb & (1 << j)) != 0, -1.0, 1.0)  # [BC, 32]
            # scale byte j: sub-blocks j (group j//4) use lo/hi nibble
            scj = tl.load(base + 66 + j)  # [BC] uint8
            nib = tl.where((j % 4) < 2, scj & 0xF, scj >> 4).to(tl.float32)  # [BC]
            deltaj = d_f * (0.5 + nib) * 0.25  # [BC]
            valj = signj * magj * deltaj[:, None]  # [BC, 32]
            # value index for (code s, sub-block j): code-major, value
            # v = s*8 + j (verified against the eager dequant ordering)
            xv = (tl.load(
                x_u16 + x_base + kb + c32 * 8 + j
            ).to(tl.uint32) << 16).to(tl.float32, bitcast=True)  # [32]
            acc += tl.sum(valj * xv[None, :], axis=1)

    tl.store(out_ptr + pid_e * M * ROWS + pid_m * ROWS + crows, acc)


def iq2xs_dequant_gemm_batch(
    x: torch.Tensor,
    packed: torch.Tensor,
    *,
    rows: int,
    cols: int,
    grid_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    BLOCK_COLS: int = 32,
) -> torch.Tensor:
    """Batched ``x @ W^T`` over ``E`` experts in one launch.

    ``x`` is [E, M, cols]; ``packed`` is [E, rows*stride] (each expert's
    packed bytes row-major); returns [E, M, rows].  One kernel launch for
    the whole expert batch -- collapses the MoE's per-expert launches.
    """
    E, M, _ = x.shape
    x = x.to(torch.bfloat16).contiguous()
    out = torch.empty((E, M, rows), dtype=torch.float32, device=x.device)
    grid_t, ksigns_t, _ = grid_tables
    if cols % _IQ2_XS_BLOCK_ELEMS:
        raise ValueError(
            f"cols {cols} is not a multiple of the {_IQ2_XS_BLOCK_ELEMS}-value "
            "IQ2_XS block"
        )
    w_row_stride = (cols // _IQ2_XS_BLOCK_ELEMS) * _IQ2_XS_BLOCK_BYTES
    assert (E * rows * w_row_stride) == packed.numel(), (
        f"packed {packed.numel()} != experts {E} x rows {rows} x stride {w_row_stride}"
    )
    _iq2xs_dequant_gemm_batch_kernel[(E, M, rows // BLOCK_COLS)](
        x,
        packed,
        grid_t,
        ksigns_t,
        out,
        E,
        M,
        K=cols,
        ROWS=rows,
        W_ROW_STRIDE=w_row_stride,
        BLOCK_COLS=BLOCK_COLS,
        BLK_ELEMS=256,
        BLK_BYTES=74,
    )
    return out


def iq2xs_dequant_gemm(
    x: torch.Tensor,
    packed: torch.Tensor,
    *,
    rows: int,
    cols: int,
    grid_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    BLOCK_COLS: int = 32,
) -> torch.Tensor:
    """Single-expert ``x @ W^T`` (wrapper over the batched kernel)."""
    return iq2xs_dequant_gemm_batch(
        x.unsqueeze(0),
        packed.unsqueeze(0),
        rows=rows,
        cols=cols,
        grid_tables=grid_tables,
        BLOCK_COLS=BLOCK_COLS,
    )[0]

