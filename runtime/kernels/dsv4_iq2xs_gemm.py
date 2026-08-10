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
    [8]  8 scale bytes (one per consecutive four codes / 32 values;
         lo nibble -> codes 0,1; hi -> codes 2,3 in that group)
Dequant per code c (mirrors dequantize_iq2_xs bit-for-bit):
    grid[code & 511]        -> 8 magnitudes (sub-block j = bits 8j..8j+7)
    ksigns[code >> 9]       -> sign byte; kmask bit j negates sub-block j
    delta = d*(0.5+nibble)*0.25, scale = code//4, lo/hi = code%4 < 2
    value = sign * magnitude * delta
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

_IQ2_XS_BLOCK_BYTES = 74
_IQ2_XS_BLOCK_ELEMS = 256
_DEFAULT_IQ2XS_BLOCK_COLS = 8
_DEFAULT_IQ2XS_NUM_WARPS = 4
_DEFAULT_IQ2XS_NUM_STAGES = 3


@triton.jit
def _swiglu_bf16_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    n_elements,
    limit: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    gate = tl.minimum(tl.load(gate_ptr + offsets, mask=mask), limit)
    up = tl.maximum(tl.minimum(tl.load(up_ptr + offsets, mask=mask), limit), -limit)
    out = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + offsets, out, mask=mask)


def swiglu_bf16(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """Fuse MoE clamp, SiLU, multiply, and the down-input BF16 cast."""
    if gate.shape != up.shape or gate.dtype != torch.float32 or up.dtype != torch.float32:
        raise ValueError("gate and up must have the same shape and fp32 dtype")
    out = torch.empty_like(gate, dtype=torch.bfloat16)
    if gate.numel() == 0:
        return out
    block = 256
    _swiglu_bf16_kernel[(triton.cdiv(gate.numel(), block),)](
        gate,
        up,
        out,
        gate.numel(),
        limit=limit,
        BLOCK=block,
    )
    return out


def _check_iq2xs_shape(rows: int, cols: int, block_cols: int) -> int:
    if rows <= 0:
        raise ValueError(f"rows must be > 0, got {rows}")
    if cols <= 0:
        raise ValueError(f"cols must be > 0, got {cols}")
    if block_cols <= 0:
        raise ValueError(f"BLOCK_COLS must be > 0, got {block_cols}")
    if cols % _IQ2_XS_BLOCK_ELEMS:
        raise ValueError(
            f"cols {cols} is not a multiple of the {_IQ2_XS_BLOCK_ELEMS}-value IQ2_XS block"
        )
    if rows % block_cols:
        raise ValueError(f"rows {rows} must be divisible by BLOCK_COLS {block_cols}")
    return (cols // _IQ2_XS_BLOCK_ELEMS) * _IQ2_XS_BLOCK_BYTES


def _prepare_iq2xs_indexed_inputs(
    x: torch.Tensor,
    packed_all: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    rows: int,
    cols: int,
    block_cols: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    if x.ndim != 3:
        raise ValueError(f"x must be rank 3 [E, M, K], got shape {tuple(x.shape)}")
    if x.shape[2] != cols:
        raise ValueError(f"x hidden {x.shape[2]} != cols {cols}")
    w_row_stride = _check_iq2xs_shape(rows, cols, block_cols)
    if x.device.type != "cuda":
        raise ValueError("iq2xs indexed kernels require x on CUDA")
    if packed_all.device != x.device:
        raise ValueError("packed_all must be on the same CUDA device as x")
    if packed_all.dtype != torch.uint8:
        raise ValueError(f"packed_all must be uint8, got {packed_all.dtype}")

    expert_stride = rows * w_row_stride
    if packed_all.numel() % expert_stride:
        raise ValueError(
            f"packed_all {packed_all.numel()} is not divisible by expert stride {expert_stride}"
        )
    num_experts = packed_all.numel() // expert_stride
    if num_experts <= 0:
        raise ValueError("packed_all must contain at least one expert")

    x = x.to(torch.bfloat16).contiguous()
    packed_all = packed_all.contiguous()
    expert_ids = expert_ids.to(device=x.device, dtype=torch.int64).contiguous()
    if expert_ids.ndim != 1:
        raise ValueError(f"expert_ids must be rank 1, got shape {tuple(expert_ids.shape)}")
    if expert_ids.numel() != x.shape[0]:
        raise ValueError(f"expert_ids {expert_ids.numel()} != batch experts {x.shape[0]}")
    # Expert ids are produced by the device-side router.  Do not inspect their
    # values here: ``min().item()`` would introduce a GPU->CPU synchronization
    # inside CUDA Graph capture.  Shape/table bounds remain host-validated;
    # route-id range is an upstream router invariant.
    return x, packed_all, expert_ids, w_row_stride, num_experts


def _prepare_iq2xs_dual_indexed_inputs(
    x: torch.Tensor,
    packed_gate_all: torch.Tensor,
    packed_up_all: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    rows: int,
    cols: int,
    block_cols: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    x, packed_gate_all, expert_ids, w_row_stride, num_experts = _prepare_iq2xs_indexed_inputs(
        x,
        packed_gate_all,
        expert_ids,
        rows=rows,
        cols=cols,
        block_cols=block_cols,
    )
    if packed_up_all.device != x.device:
        raise ValueError("packed_up_all must be on the same CUDA device as x")
    if packed_up_all.dtype != torch.uint8:
        raise ValueError(f"packed_up_all must be uint8, got {packed_up_all.dtype}")
    packed_up_all = packed_up_all.contiguous()
    expert_stride = rows * w_row_stride
    if packed_up_all.numel() != num_experts * expert_stride:
        raise ValueError(
            "packed_up_all must contain the same number of experts as packed_gate_all "
            f"({num_experts} x stride {expert_stride}), got {packed_up_all.numel()} bytes"
        )
    return x, packed_gate_all, packed_up_all, expert_ids, w_row_stride


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
        col_byte = pid_e * ROWS * W_ROW_STRIDE + crows * W_ROW_STRIDE + kb_block * BLK_BYTES  # [BC]
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
            # One scale byte covers four consecutive codes (32 values).
            # Its low nibble serves codes 0/1 and high nibble codes 2/3.
            # This is indexed by c32, not by the within-code value j.
            scj = tl.load(base[:, None] + 66 + (c32[None, :] // 4))
            nib = tl.where((c32[None, :] % 4) < 2, scj & 0xF, scj >> 4).to(tl.float32)
            deltaj = d_f[:, None] * (0.5 + nib) * 0.25
            valj = signj * magj * deltaj
            # value index for (code s, sub-block j): code-major, value
            # v = s*8 + j (verified against the eager dequant ordering)
            xv = (tl.load(x_u16 + x_base + kb + c32 * 8 + j).to(tl.uint32) << 16).to(
                tl.float32, bitcast=True
            )  # [32]
            acc += tl.sum(valj * xv[None, :], axis=1)

    tl.store(out_ptr + pid_e * M * ROWS + pid_m * ROWS + crows, acc)


@triton.jit
def _iq2xs_dequant_gemm_batch_indexed_kernel(
    x_ptr,
    w_ptr,
    expert_ids_ptr,
    grid_ptr,
    ksigns_ptr,
    out_ptr,
    E,
    M,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    BLK_ELEMS: tl.constexpr,
    BLK_BYTES: tl.constexpr,
):
    """Route-batched variant selecting experts from one global packed table."""
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    col0 = pid_c * BLOCK_COLS

    x_u16 = x_ptr.to(tl.pointer_type(tl.uint16))
    x_base = pid_e * M * K + pid_m * K
    eid = tl.load(expert_ids_ptr + pid_e).to(tl.int32)

    crows = col0 + tl.arange(0, BLOCK_COLS)
    acc = tl.zeros((BLOCK_COLS,), dtype=tl.float32)

    for kb in range(0, K, BLK_ELEMS):
        kb_block = kb // BLK_ELEMS
        col_byte = eid * ROWS * W_ROW_STRIDE + crows * W_ROW_STRIDE + kb_block * BLK_BYTES
        base = w_ptr + col_byte

        c32 = tl.arange(0, 32)
        lo = tl.load(base[:, None] + 2 + c32[None, :] * 2).to(tl.uint32)
        hi = tl.load(base[:, None] + 2 + c32[None, :] * 2 + 1).to(tl.uint32)
        codes = (lo | (hi << 8)) & 0xFFFF

        g = tl.load(grid_ptr + (codes & 511))
        sb = tl.load(ksigns_ptr + (codes >> 9))

        d_lo = tl.load(base).to(tl.uint32)
        d_hi = tl.load(base + 1).to(tl.uint32)
        d_u16 = d_lo.to(tl.uint16) | (d_hi.to(tl.uint16) << 8)
        d_f = d_u16.to(tl.float16, bitcast=True).to(tl.float32)

        for j in tl.static_range(8):
            magj = ((g >> (8 * j)) & 0xFF).to(tl.float32)
            signj = tl.where((sb & (1 << j)) != 0, -1.0, 1.0)
            scj = tl.load(base[:, None] + 66 + (c32[None, :] // 4))
            nib = tl.where((c32[None, :] % 4) < 2, scj & 0xF, scj >> 4).to(tl.float32)
            deltaj = d_f[:, None] * (0.5 + nib) * 0.25
            valj = signj * magj * deltaj
            xv = (tl.load(x_u16 + x_base + kb + c32 * 8 + j).to(tl.uint32) << 16).to(
                tl.float32, bitcast=True
            )
            acc += tl.sum(valj * xv[None, :], axis=1)

    tl.store(out_ptr + pid_e * M * ROWS + pid_m * ROWS + crows, acc)


@triton.jit
def _iq2xs_dequant_gemm_batch_indexed_dual_kernel(
    x_ptr,
    gate_w_ptr,
    up_w_ptr,
    expert_ids_ptr,
    grid_ptr,
    ksigns_ptr,
    gate_out_ptr,
    up_out_ptr,
    E,
    M,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    BLK_ELEMS: tl.constexpr,
    BLK_BYTES: tl.constexpr,
):
    """Indexed dual-path IQ2_XS dequant-GEMM sharing activation loads."""
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)
    col0 = pid_c * BLOCK_COLS

    x_u16 = x_ptr.to(tl.pointer_type(tl.uint16))
    x_base = pid_e * M * K + pid_m * K
    eid = tl.load(expert_ids_ptr + pid_e).to(tl.int32)

    crows = col0 + tl.arange(0, BLOCK_COLS)
    c32 = tl.arange(0, 32)
    gate_acc = tl.zeros((BLOCK_COLS,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_COLS,), dtype=tl.float32)

    for kb in range(0, K, BLK_ELEMS):
        kb_block = kb // BLK_ELEMS
        col_byte = eid * ROWS * W_ROW_STRIDE + crows * W_ROW_STRIDE + kb_block * BLK_BYTES
        gate_base = gate_w_ptr + col_byte
        up_base = up_w_ptr + col_byte

        gate_lo = tl.load(gate_base[:, None] + 2 + c32[None, :] * 2).to(tl.uint32)
        gate_hi = tl.load(gate_base[:, None] + 2 + c32[None, :] * 2 + 1).to(tl.uint32)
        gate_codes = (gate_lo | (gate_hi << 8)) & 0xFFFF
        gate_g = tl.load(grid_ptr + (gate_codes & 511))
        gate_sb = tl.load(ksigns_ptr + (gate_codes >> 9))
        gate_d_lo = tl.load(gate_base).to(tl.uint32)
        gate_d_hi = tl.load(gate_base + 1).to(tl.uint32)
        gate_d_u16 = gate_d_lo.to(tl.uint16) | (gate_d_hi.to(tl.uint16) << 8)
        gate_d_f = gate_d_u16.to(tl.float16, bitcast=True).to(tl.float32)

        up_lo = tl.load(up_base[:, None] + 2 + c32[None, :] * 2).to(tl.uint32)
        up_hi = tl.load(up_base[:, None] + 2 + c32[None, :] * 2 + 1).to(tl.uint32)
        up_codes = (up_lo | (up_hi << 8)) & 0xFFFF
        up_g = tl.load(grid_ptr + (up_codes & 511))
        up_sb = tl.load(ksigns_ptr + (up_codes >> 9))
        up_d_lo = tl.load(up_base).to(tl.uint32)
        up_d_hi = tl.load(up_base + 1).to(tl.uint32)
        up_d_u16 = up_d_lo.to(tl.uint16) | (up_d_hi.to(tl.uint16) << 8)
        up_d_f = up_d_u16.to(tl.float16, bitcast=True).to(tl.float32)

        scale_idx = 66 + (c32[None, :] // 4)
        gate_sc = tl.load(gate_base[:, None] + scale_idx)
        up_sc = tl.load(up_base[:, None] + scale_idx)
        is_lo = (c32[None, :] % 4) < 2

        for j in tl.static_range(8):
            xv = (tl.load(x_u16 + x_base + kb + c32 * 8 + j).to(tl.uint32) << 16).to(
                tl.float32, bitcast=True
            )

            gate_magj = ((gate_g >> (8 * j)) & 0xFF).to(tl.float32)
            gate_signj = tl.where((gate_sb & (1 << j)) != 0, -1.0, 1.0)
            gate_nib = tl.where(is_lo, gate_sc & 0xF, gate_sc >> 4).to(tl.float32)
            gate_deltaj = gate_d_f[:, None] * (0.5 + gate_nib) * 0.25
            gate_valj = gate_signj * gate_magj * gate_deltaj
            gate_acc += tl.sum(gate_valj * xv[None, :], axis=1)

            up_magj = ((up_g >> (8 * j)) & 0xFF).to(tl.float32)
            up_signj = tl.where((up_sb & (1 << j)) != 0, -1.0, 1.0)
            up_nib = tl.where(is_lo, up_sc & 0xF, up_sc >> 4).to(tl.float32)
            up_deltaj = up_d_f[:, None] * (0.5 + up_nib) * 0.25
            up_valj = up_signj * up_magj * up_deltaj
            up_acc += tl.sum(up_valj * xv[None, :], axis=1)

    out_base = pid_e * M * ROWS + pid_m * ROWS + crows
    tl.store(gate_out_ptr + out_base, gate_acc)
    tl.store(up_out_ptr + out_base, up_acc)


@triton.jit
def _iq2xs_dequant_gemm_batch_indexed_dual_swiglu_b1_kernel(
    x_ptr,
    gate_w_ptr,
    up_w_ptr,
    expert_ids_ptr,
    grid_ptr,
    ksigns_ptr,
    out_ptr,
    E,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    BLK_ELEMS: tl.constexpr,
    BLK_BYTES: tl.constexpr,
):
    """B1 dual IQ2_XS projections followed by exact BF16 SwiGLU."""
    pid_e = tl.program_id(0)
    pid_c = tl.program_id(1)
    col0 = pid_c * BLOCK_COLS

    x_u16 = x_ptr.to(tl.pointer_type(tl.uint16))
    x_base = pid_e * K
    eid = tl.load(expert_ids_ptr + pid_e).to(tl.int32)

    crows = col0 + tl.arange(0, BLOCK_COLS)
    c32 = tl.arange(0, 32)
    gate_acc = tl.zeros((BLOCK_COLS,), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_COLS,), dtype=tl.float32)

    for kb in range(0, K, BLK_ELEMS):
        kb_block = kb // BLK_ELEMS
        col_byte = eid * ROWS * W_ROW_STRIDE + crows * W_ROW_STRIDE + kb_block * BLK_BYTES
        gate_base = gate_w_ptr + col_byte
        up_base = up_w_ptr + col_byte

        gate_lo = tl.load(gate_base[:, None] + 2 + c32[None, :] * 2).to(tl.uint32)
        gate_hi = tl.load(gate_base[:, None] + 2 + c32[None, :] * 2 + 1).to(tl.uint32)
        gate_codes = (gate_lo | (gate_hi << 8)) & 0xFFFF
        gate_g = tl.load(grid_ptr + (gate_codes & 511))
        gate_sb = tl.load(ksigns_ptr + (gate_codes >> 9))
        gate_d_lo = tl.load(gate_base).to(tl.uint32)
        gate_d_hi = tl.load(gate_base + 1).to(tl.uint32)
        gate_d_u16 = gate_d_lo.to(tl.uint16) | (gate_d_hi.to(tl.uint16) << 8)
        gate_d_f = gate_d_u16.to(tl.float16, bitcast=True).to(tl.float32)

        up_lo = tl.load(up_base[:, None] + 2 + c32[None, :] * 2).to(tl.uint32)
        up_hi = tl.load(up_base[:, None] + 2 + c32[None, :] * 2 + 1).to(tl.uint32)
        up_codes = (up_lo | (up_hi << 8)) & 0xFFFF
        up_g = tl.load(grid_ptr + (up_codes & 511))
        up_sb = tl.load(ksigns_ptr + (up_codes >> 9))
        up_d_lo = tl.load(up_base).to(tl.uint32)
        up_d_hi = tl.load(up_base + 1).to(tl.uint32)
        up_d_u16 = up_d_lo.to(tl.uint16) | (up_d_hi.to(tl.uint16) << 8)
        up_d_f = up_d_u16.to(tl.float16, bitcast=True).to(tl.float32)

        scale_idx = 66 + (c32[None, :] // 4)
        gate_sc = tl.load(gate_base[:, None] + scale_idx)
        up_sc = tl.load(up_base[:, None] + scale_idx)
        is_lo = (c32[None, :] % 4) < 2

        for j in tl.static_range(8):
            xv = (tl.load(x_u16 + x_base + kb + c32 * 8 + j).to(tl.uint32) << 16).to(
                tl.float32, bitcast=True
            )

            gate_magj = ((gate_g >> (8 * j)) & 0xFF).to(tl.float32)
            gate_signj = tl.where((gate_sb & (1 << j)) != 0, -1.0, 1.0)
            gate_nib = tl.where(is_lo, gate_sc & 0xF, gate_sc >> 4).to(tl.float32)
            gate_deltaj = gate_d_f[:, None] * (0.5 + gate_nib) * 0.25
            gate_valj = gate_signj * gate_magj * gate_deltaj
            gate_acc += tl.sum(gate_valj * xv[None, :], axis=1)

            up_magj = ((up_g >> (8 * j)) & 0xFF).to(tl.float32)
            up_signj = tl.where((up_sb & (1 << j)) != 0, -1.0, 1.0)
            up_nib = tl.where(is_lo, up_sc & 0xF, up_sc >> 4).to(tl.float32)
            up_deltaj = up_d_f[:, None] * (0.5 + up_nib) * 0.25
            up_valj = up_signj * up_magj * up_deltaj
            up_acc += tl.sum(up_valj * xv[None, :], axis=1)

    # Preserve the exact operation and rounding boundaries of swiglu_bf16:
    # fp32 clamp -> fp32 SiLU/multiply -> one BF16 store.
    gate = tl.minimum(gate_acc, LIMIT)
    up = tl.maximum(tl.minimum(up_acc, LIMIT), -LIMIT)
    out = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + pid_e * ROWS + crows, out)


def iq2xs_dequant_gemm_batch(
    x: torch.Tensor,
    packed: torch.Tensor,
    *,
    rows: int,
    cols: int,
    grid_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    BLOCK_COLS: int = _DEFAULT_IQ2XS_BLOCK_COLS,
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
    w_row_stride = _check_iq2xs_shape(rows, cols, BLOCK_COLS)
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


def iq2xs_dequant_gemm_batch_indexed(
    x: torch.Tensor,
    packed_all: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    rows: int,
    cols: int,
    grid_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    BLOCK_COLS: int = _DEFAULT_IQ2XS_BLOCK_COLS,
) -> torch.Tensor:
    """Batched ``x @ W_e^T`` over route ids indexing one packed expert table."""
    x, packed_all, expert_ids, w_row_stride, _ = _prepare_iq2xs_indexed_inputs(
        x,
        packed_all,
        expert_ids,
        rows=rows,
        cols=cols,
        block_cols=BLOCK_COLS,
    )
    E, M, _ = x.shape
    out = torch.empty((E, M, rows), dtype=torch.float32, device=x.device)
    grid_t, ksigns_t, _ = grid_tables
    _iq2xs_dequant_gemm_batch_indexed_kernel[(E, M, rows // BLOCK_COLS)](
        x,
        packed_all,
        expert_ids,
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


def iq2xs_dequant_gemm_batch_indexed_dual(
    x: torch.Tensor,
    packed_gate_all: torch.Tensor,
    packed_up_all: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    rows: int,
    cols: int,
    grid_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    BLOCK_COLS: int = _DEFAULT_IQ2XS_BLOCK_COLS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Indexed dual-path ``x @ W_gate^T`` and ``x @ W_up^T`` in one launch."""
    x, packed_gate_all, packed_up_all, expert_ids, w_row_stride = (
        _prepare_iq2xs_dual_indexed_inputs(
            x,
            packed_gate_all,
            packed_up_all,
            expert_ids,
            rows=rows,
            cols=cols,
            block_cols=BLOCK_COLS,
        )
    )
    E, M, _ = x.shape
    gate_out = torch.empty((E, M, rows), dtype=torch.float32, device=x.device)
    up_out = torch.empty_like(gate_out)
    grid_t, ksigns_t, _ = grid_tables
    _iq2xs_dequant_gemm_batch_indexed_dual_kernel[(E, M, rows // BLOCK_COLS)](
        x,
        packed_gate_all,
        packed_up_all,
        expert_ids,
        grid_t,
        ksigns_t,
        gate_out,
        up_out,
        E,
        M,
        K=cols,
        ROWS=rows,
        W_ROW_STRIDE=w_row_stride,
        BLOCK_COLS=BLOCK_COLS,
        BLK_ELEMS=256,
        BLK_BYTES=74,
    )
    return gate_out, up_out


def iq2xs_dequant_gemm_batch_indexed_dual_swiglu_b1(
    x: torch.Tensor,
    packed_gate_all: torch.Tensor,
    packed_up_all: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    rows: int,
    cols: int,
    grid_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    limit: float,
    BLOCK_COLS: int = _DEFAULT_IQ2XS_BLOCK_COLS,
) -> torch.Tensor:
    """B1-only indexed gate/up IQ2_XS GEMMs fused with BF16 SwiGLU.

    The result is ``swiglu_bf16(*iq2xs_..._dual(...), limit)`` with the
    gate/up fp32 accumulators kept in registers.  This deliberately accepts
    only ``x.shape == [E, 1, cols]``; prefill and general-M paths remain on
    the standalone kernels.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be rank 3 [E, M, K], got shape {tuple(x.shape)}")
    E, M, _ = x.shape
    if E <= 0:
        raise ValueError("B1 dual-SwiGLU requires at least one routed expert")
    if M != 1:
        raise ValueError(f"B1 dual-SwiGLU requires exactly one token per expert, got M={M}")
    limit = float(limit)
    if not 0.0 < limit < float("inf"):
        raise ValueError(f"limit must be finite and > 0, got {limit}")

    x, packed_gate_all, packed_up_all, expert_ids, w_row_stride = (
        _prepare_iq2xs_dual_indexed_inputs(
            x,
            packed_gate_all,
            packed_up_all,
            expert_ids,
            rows=rows,
            cols=cols,
            block_cols=BLOCK_COLS,
        )
    )

    out = torch.empty((E, 1, rows), dtype=torch.bfloat16, device=x.device)
    grid_t, ksigns_t, _ = grid_tables
    _iq2xs_dequant_gemm_batch_indexed_dual_swiglu_b1_kernel[(E, rows // BLOCK_COLS)](
        x,
        packed_gate_all,
        packed_up_all,
        expert_ids,
        grid_t,
        ksigns_t,
        out,
        E,
        K=cols,
        ROWS=rows,
        W_ROW_STRIDE=w_row_stride,
        LIMIT=limit,
        BLOCK_COLS=BLOCK_COLS,
        BLK_ELEMS=_IQ2_XS_BLOCK_ELEMS,
        BLK_BYTES=_IQ2_XS_BLOCK_BYTES,
    )
    return out


def compile_iq2xs_dequant_gemm_batch_indexed_dual_sm120(
    *,
    rows: int = 2048,
    cols: int = 4096,
    block_cols: int = _DEFAULT_IQ2XS_BLOCK_COLS,
    num_warps: int = _DEFAULT_IQ2XS_NUM_WARPS,
    num_stages: int = _DEFAULT_IQ2XS_NUM_STAGES,
):
    """Offline-compile the standalone dual indexed IQ2_XS kernel for SM120."""
    w_row_stride = _check_iq2xs_shape(rows, cols, block_cols)
    _iq2xs_dequant_gemm_batch_indexed_dual_kernel.create_binder()
    src = ASTSource(
        fn=_iq2xs_dequant_gemm_batch_indexed_dual_kernel,
        signature={
            "x_ptr": "*bf16",
            "gate_w_ptr": "*u8",
            "up_w_ptr": "*u8",
            "expert_ids_ptr": "*i64",
            "grid_ptr": "*i64",
            "ksigns_ptr": "*i32",
            "gate_out_ptr": "*fp32",
            "up_out_ptr": "*fp32",
            "E": "i32",
            "M": "i32",
            "K": "constexpr",
            "ROWS": "constexpr",
            "W_ROW_STRIDE": "constexpr",
            "BLOCK_COLS": "constexpr",
            "BLK_ELEMS": "constexpr",
            "BLK_BYTES": "constexpr",
        },
        constexprs={
            "K": cols,
            "ROWS": rows,
            "W_ROW_STRIDE": w_row_stride,
            "BLOCK_COLS": block_cols,
            "BLK_ELEMS": _IQ2_XS_BLOCK_ELEMS,
            "BLK_BYTES": _IQ2_XS_BLOCK_BYTES,
        },
    )
    return triton.compile(
        src,
        target=GPUTarget("cuda", 120, 32),
        options={
            "num_warps": num_warps,
            "num_stages": num_stages,
        },
    )


def compile_iq2xs_dequant_gemm_batch_indexed_dual_swiglu_b1_sm120(
    *,
    rows: int = 2048,
    cols: int = 4096,
    limit: float = 10.0,
    block_cols: int = _DEFAULT_IQ2XS_BLOCK_COLS,
    num_warps: int = _DEFAULT_IQ2XS_NUM_WARPS,
    num_stages: int = _DEFAULT_IQ2XS_NUM_STAGES,
):
    """Offline-compile the B1 dual IQ2_XS + SwiGLU candidate for SM120."""
    w_row_stride = _check_iq2xs_shape(rows, cols, block_cols)
    limit = float(limit)
    if not 0.0 < limit < float("inf"):
        raise ValueError(f"limit must be finite and > 0, got {limit}")
    src = ASTSource(
        fn=_iq2xs_dequant_gemm_batch_indexed_dual_swiglu_b1_kernel,
        signature={
            "x_ptr": "*bf16",
            "gate_w_ptr": "*u8",
            "up_w_ptr": "*u8",
            "expert_ids_ptr": "*i64",
            "grid_ptr": "*i64",
            "ksigns_ptr": "*i32",
            "out_ptr": "*bf16",
            "E": "i32",
            "K": "constexpr",
            "ROWS": "constexpr",
            "W_ROW_STRIDE": "constexpr",
            "LIMIT": "constexpr",
            "BLOCK_COLS": "constexpr",
            "BLK_ELEMS": "constexpr",
            "BLK_BYTES": "constexpr",
        },
        constexprs={
            "K": cols,
            "ROWS": rows,
            "W_ROW_STRIDE": w_row_stride,
            "LIMIT": limit,
            "BLOCK_COLS": block_cols,
            "BLK_ELEMS": _IQ2_XS_BLOCK_ELEMS,
            "BLK_BYTES": _IQ2_XS_BLOCK_BYTES,
        },
    )
    return triton.compile(
        src,
        target=GPUTarget("cuda", 120, 32),
        options={
            "num_warps": num_warps,
            "num_stages": num_stages,
        },
    )


def iq2xs_dequant_gemm(
    x: torch.Tensor,
    packed: torch.Tensor,
    *,
    rows: int,
    cols: int,
    grid_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    BLOCK_COLS: int = _DEFAULT_IQ2XS_BLOCK_COLS,
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
