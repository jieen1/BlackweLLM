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

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


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
    M_PAD: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    BLK_ELEMS: tl.constexpr,
    BLK_BYTES: tl.constexpr,
):
    """Route-batched IQ2_XS dequant-GEMM with per-expert M-token batching.

    One block covers one (expert, row-block) and dequantizes its packed
    weights ONCE, then reuses them across all ``M`` tokens assigned to that
    expert.  The previous layout launched one block per token, re-reading
    the same weight block ``M`` times from HBM (that is why M>1 prefill was
    only ~4x faster than M=1 decode instead of M-times faster).  ``acc`` is
    [M, BLOCK_COLS] so the K-accumulation stays correct across the kb loop.
    """
    pid_e = tl.program_id(0)
    pid_c = tl.program_id(1)
    col0 = pid_c * BLOCK_COLS

    x_u16 = x_ptr.to(tl.pointer_type(tl.uint16))
    eid = tl.load(expert_ids_ptr + pid_e).to(tl.int32)

    crows = col0 + tl.arange(0, BLOCK_COLS)
    c32 = tl.arange(0, 32)

    if M_PAD == 1:
        # M=1: 2D reduction bit-identical to the legacy per-token kernel
        # (the B1 decode kernel keeps this exact accumulation order).
        acc = tl.zeros((BLOCK_COLS,), dtype=tl.float32)
        for kb in range(0, K, BLK_ELEMS):
            kb_block = kb // BLK_ELEMS
            col_byte = eid * ROWS * W_ROW_STRIDE + crows * W_ROW_STRIDE + kb_block * BLK_BYTES
            base = w_ptr + col_byte

            lo = tl.load(base[:, None] + 2 + c32[None, :] * 2).to(tl.uint32)
            hi = tl.load(base[:, None] + 2 + c32[None, :] * 2 + 1).to(tl.uint32)
            codes = (lo | (hi << 8)) & 0xFFFF

            g = tl.load(grid_ptr + (codes & 511))
            sb = tl.load(ksigns_ptr + (codes >> 9))

            d_lo = tl.load(base).to(tl.uint32)
            d_hi = tl.load(base + 1).to(tl.uint32)
            d_u16 = d_lo.to(tl.uint16) | (d_hi.to(tl.uint16) << 8)
            d_f = d_u16.to(tl.float16, bitcast=True).to(tl.float32)
            sc = tl.load(base[:, None] + 66 + (c32[None, :] // 4))
            nib = tl.where((c32[None, :] % 4) < 2, sc & 0xF, sc >> 4).to(tl.float32)
            deltab = d_f[:, None] * (0.5 + nib) * 0.25

            for j in tl.static_range(8):
                magj = ((g >> (8 * j)) & 0xFF).to(tl.float32)
                signj = tl.where((sb & (1 << j)) != 0, -1.0, 1.0)
                valj = signj * magj * deltab
                xv = (tl.load(x_u16 + pid_e * K + kb + c32 * 8 + j).to(tl.uint32) << 16).to(
                    tl.float32, bitcast=True
                )
                acc += tl.sum(valj * xv[None, :], axis=1)

        tl.store(out_ptr + pid_e * ROWS + crows, acc)
        return

    for mm in tl.static_range(M_PAD):
        # Per-row 2D reduction, bit-identical to the M_PAD==1 branch: one
        # [BLOCK_COLS, 32] tl.sum per token keeps the fp32 accumulation order
        # stable across triton versions (a batched [M, BLOCK_COLS, 32] reduce
        # rebalances its tree under triton 3.7 and breaks the M-batched vs
        # M=1 bit-exact contract). Weight dequant is re-read per row inside
        # the kb loop; M>1 prefill correctness is preserved at the cost of
        # re-reading the packed weight block per token (the 3D path hoisted
        # it). The numeric gate is the priority: same tree as M=1.
        acc = tl.zeros((BLOCK_COLS,), dtype=tl.float32)
        for kb in range(0, K, BLK_ELEMS):
            kb_block = kb // BLK_ELEMS
            col_byte = eid * ROWS * W_ROW_STRIDE + crows * W_ROW_STRIDE + kb_block * BLK_BYTES
            base = w_ptr + col_byte

            lo = tl.load(base[:, None] + 2 + c32[None, :] * 2).to(tl.uint32)
            hi = tl.load(base[:, None] + 2 + c32[None, :] * 2 + 1).to(tl.uint32)
            codes = (lo | (hi << 8)) & 0xFFFF

            g = tl.load(grid_ptr + (codes & 511))
            sb = tl.load(ksigns_ptr + (codes >> 9))

            d_lo = tl.load(base).to(tl.uint32)
            d_hi = tl.load(base + 1).to(tl.uint32)
            d_u16 = d_lo.to(tl.uint16) | (d_hi.to(tl.uint16) << 8)
            d_f = d_u16.to(tl.float16, bitcast=True).to(tl.float32)
            sc = tl.load(base[:, None] + 66 + (c32[None, :] // 4))
            nib = tl.where((c32[None, :] % 4) < 2, sc & 0xF, sc >> 4).to(tl.float32)
            deltab = d_f[:, None] * (0.5 + nib) * 0.25

            for j in tl.static_range(8):
                magj = ((g >> (8 * j)) & 0xFF).to(tl.float32)
                signj = tl.where((sb & (1 << j)) != 0, -1.0, 1.0)
                valj = signj * magj * deltab
                xaddr = pid_e * M_PAD * K + mm * K + kb + c32 * 8 + j
                xv = (tl.load(x_u16 + xaddr).to(tl.uint32) << 16).to(
                    tl.float32, bitcast=True
                )
                acc += tl.sum(valj * xv[None, :], axis=1)

        tl.store(out_ptr + pid_e * M_PAD * ROWS + mm * ROWS + crows, acc)


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
    M_PAD: tl.constexpr,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    BLK_ELEMS: tl.constexpr,
    BLK_BYTES: tl.constexpr,
):
    """Indexed dual-path IQ2_XS dequant-GEMM sharing activation loads.

    Weights are dequantized once per (expert, row-block) and reused across
    the M tokens assigned to that expert; ``acc`` is [M, BLOCK_COLS] so the
    K-accumulation accumulates correctly across the kb loop.
    """
    pid_e = tl.program_id(0)
    pid_c = tl.program_id(1)
    col0 = pid_c * BLOCK_COLS

    x_u16 = x_ptr.to(tl.pointer_type(tl.uint16))
    eid = tl.load(expert_ids_ptr + pid_e).to(tl.int32)

    crows = col0 + tl.arange(0, BLOCK_COLS)
    c32 = tl.arange(0, 32)
    scale_idx = 66 + (c32[None, :] // 4)
    is_lo = (c32[None, :] % 4) < 2

    if M_PAD == 1:
        # M=1: 2D reduction, bit-identical to the legacy per-token dual
        # kernel (kept so the B1 decode kernel's split-exactly contract holds).
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
            gate_sc = tl.load(gate_base[:, None] + scale_idx)
            gate_nib = tl.where(is_lo, gate_sc & 0xF, gate_sc >> 4).to(tl.float32)
            gate_deltab = gate_d_f[:, None] * (0.5 + gate_nib) * 0.25

            up_lo = tl.load(up_base[:, None] + 2 + c32[None, :] * 2).to(tl.uint32)
            up_hi = tl.load(up_base[:, None] + 2 + c32[None, :] * 2 + 1).to(tl.uint32)
            up_codes = (up_lo | (up_hi << 8)) & 0xFFFF
            up_g = tl.load(grid_ptr + (up_codes & 511))
            up_sb = tl.load(ksigns_ptr + (up_codes >> 9))
            up_d_lo = tl.load(up_base).to(tl.uint32)
            up_d_hi = tl.load(up_base + 1).to(tl.uint32)
            up_d_u16 = up_d_lo.to(tl.uint16) | (up_d_hi.to(tl.uint16) << 8)
            up_d_f = up_d_u16.to(tl.float16, bitcast=True).to(tl.float32)
            up_sc = tl.load(up_base[:, None] + scale_idx)
            up_nib = tl.where(is_lo, up_sc & 0xF, up_sc >> 4).to(tl.float32)
            up_deltab = up_d_f[:, None] * (0.5 + up_nib) * 0.25

            for j in tl.static_range(8):
                xv = (tl.load(x_u16 + pid_e * K + kb + c32 * 8 + j).to(tl.uint32) << 16).to(
                    tl.float32, bitcast=True
                )
                gate_magj = ((gate_g >> (8 * j)) & 0xFF).to(tl.float32)
                gate_signj = tl.where((gate_sb & (1 << j)) != 0, -1.0, 1.0)
                gate_valj = gate_signj * gate_magj * gate_deltab
                gate_acc += tl.sum(gate_valj * xv[None, :], axis=1)
                up_magj = ((up_g >> (8 * j)) & 0xFF).to(tl.float32)
                up_signj = tl.where((up_sb & (1 << j)) != 0, -1.0, 1.0)
                up_valj = up_signj * up_magj * up_deltab
                up_acc += tl.sum(up_valj * xv[None, :], axis=1)

        out_base = pid_e * ROWS + crows
        tl.store(gate_out_ptr + out_base, gate_acc)
        tl.store(up_out_ptr + out_base, up_acc)
        return

    m = tl.arange(0, M_PAD)
    gate_acc = tl.zeros((M_PAD, BLOCK_COLS), dtype=tl.float32)
    up_acc = tl.zeros((M_PAD, BLOCK_COLS), dtype=tl.float32)

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
        gate_sc = tl.load(gate_base[:, None] + scale_idx)
        gate_nib = tl.where(is_lo, gate_sc & 0xF, gate_sc >> 4).to(tl.float32)
        gate_deltab = gate_d_f[:, None] * (0.5 + gate_nib) * 0.25

        up_lo = tl.load(up_base[:, None] + 2 + c32[None, :] * 2).to(tl.uint32)
        up_hi = tl.load(up_base[:, None] + 2 + c32[None, :] * 2 + 1).to(tl.uint32)
        up_codes = (up_lo | (up_hi << 8)) & 0xFFFF
        up_g = tl.load(grid_ptr + (up_codes & 511))
        up_sb = tl.load(ksigns_ptr + (up_codes >> 9))
        up_d_lo = tl.load(up_base).to(tl.uint32)
        up_d_hi = tl.load(up_base + 1).to(tl.uint32)
        up_d_u16 = up_d_lo.to(tl.uint16) | (up_d_hi.to(tl.uint16) << 8)
        up_d_f = up_d_u16.to(tl.float16, bitcast=True).to(tl.float32)
        up_sc = tl.load(up_base[:, None] + scale_idx)
        up_nib = tl.where(is_lo, up_sc & 0xF, up_sc >> 4).to(tl.float32)
        up_deltab = up_d_f[:, None] * (0.5 + up_nib) * 0.25

        for j in tl.static_range(8):
            xaddr = pid_e * M_PAD * K + m[:, None] * K + kb + c32[None, :] * 8 + j
            xv = (tl.load(x_u16 + xaddr).to(tl.uint32) << 16).to(
                tl.float32, bitcast=True
            )

            gate_magj = ((gate_g >> (8 * j)) & 0xFF).to(tl.float32)
            gate_signj = tl.where((gate_sb & (1 << j)) != 0, -1.0, 1.0)
            gate_valj = gate_signj * gate_magj * gate_deltab
            gate_acc += tl.sum(gate_valj[None, :, :] * xv[:, None, :], axis=2)

            up_magj = ((up_g >> (8 * j)) & 0xFF).to(tl.float32)
            up_signj = tl.where((up_sb & (1 << j)) != 0, -1.0, 1.0)
            up_valj = up_signj * up_magj * up_deltab
            up_acc += tl.sum(up_valj[None, :, :] * xv[:, None, :], axis=2)

    out_addr = pid_e * M_PAD * ROWS + m[:, None] * ROWS + crows[None, :]
    tl.store(gate_out_ptr + out_addr, gate_acc)
    tl.store(up_out_ptr + out_addr, up_acc)


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
    m_pad = _next_pow2(M)
    if m_pad != M:
        padded = torch.zeros((E, m_pad, cols), dtype=x.dtype, device=x.device)
        padded[:, :M] = x
        x = padded
    out = torch.empty((E, m_pad, rows), dtype=torch.float32, device=x.device)
    grid_t, ksigns_t, _ = grid_tables
    _iq2xs_dequant_gemm_batch_indexed_kernel[(E, rows // BLOCK_COLS)](
        x,
        packed_all,
        expert_ids,
        grid_t,
        ksigns_t,
        out,
        E,
        m_pad,
        K=cols,
        ROWS=rows,
        W_ROW_STRIDE=w_row_stride,
        BLOCK_COLS=BLOCK_COLS,
        BLK_ELEMS=256,
        BLK_BYTES=74,
    )
    return out[:, :M]


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
    m_pad = _next_pow2(M)
    if m_pad != M:
        padded = torch.zeros((E, m_pad, cols), dtype=x.dtype, device=x.device)
        padded[:, :M] = x
        x = padded
    gate_out = torch.empty((E, m_pad, rows), dtype=torch.float32, device=x.device)
    up_out = torch.empty_like(gate_out)
    grid_t, ksigns_t, _ = grid_tables
    _iq2xs_dequant_gemm_batch_indexed_dual_kernel[(E, rows // BLOCK_COLS)](
        x,
        packed_gate_all,
        packed_up_all,
        expert_ids,
        grid_t,
        ksigns_t,
        gate_out,
        up_out,
        E,
        m_pad,
        K=cols,
        ROWS=rows,
        W_ROW_STRIDE=w_row_stride,
        BLOCK_COLS=BLOCK_COLS,
        BLK_ELEMS=256,
        BLK_BYTES=74,
    )
    return gate_out[:, :M], up_out[:, :M]


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

    gate_out = torch.empty((E, 1, rows), dtype=torch.float32, device=x.device)
    up_out = torch.empty_like(gate_out)
    grid_t, ksigns_t, _ = grid_tables
    # Gate/up are stored as raw fp32 and SwiGLU runs outside the kernel.
    # Fusing the SwiGLU tail changed the compiler's register allocation for
    # the GEMM accumulation loop, which under triton 3.7 rebalanced the fp32
    # tl.sum reduce tree and broke the bit-exact split contract at bf16
    # rounding edges. Keeping this kernel structurally identical to the dual
    # reference kernel makes it bit-identical to it on every triton version;
    # swiglu_bf16 below matches the reference pipeline exactly.
    _iq2xs_dequant_gemm_batch_indexed_dual_b1_kernel_rawfp32[
        (E, rows // BLOCK_COLS)
    ](
        x,
        packed_gate_all,
        packed_up_all,
        expert_ids,
        grid_t,
        ksigns_t,
        gate_out,
        up_out,
        E,
        K=cols,
        ROWS=rows,
        W_ROW_STRIDE=w_row_stride,
        LIMIT=limit,
        BLOCK_COLS=BLOCK_COLS,
        BLK_ELEMS=_IQ2_XS_BLOCK_ELEMS,
        BLK_BYTES=_IQ2_XS_BLOCK_BYTES,
    )
    return swiglu_bf16(gate_out, up_out, limit)



@triton.jit
def _iq2xs_dequant_gemm_batch_indexed_dual_b1_kernel_rawfp32(
    x_ptr,
    gate_w_ptr,
    up_w_ptr,
    expert_ids_ptr,
    grid_ptr,
    ksigns_ptr,
    gate_out_ptr,
    up_out_ptr,
    E,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    BLK_ELEMS: tl.constexpr,
    BLK_BYTES: tl.constexpr,
):
    """B1 dual IQ2_XS storing raw fp32 gate/up; SwiGLU applied outside.

    Deliberately NOT fused with the SwiGLU tail: fusing it changed the
    compiler's register allocation for the GEMM accumulation loop, which
    under triton 3.7 rebalanced the fp32 ``tl.sum`` reduce tree and broke the
    bit-exact split contract at bf16 rounding edges (2/2048 elements, and a
    DSV4 MoE decode batch). Structurally identical to
    ``_iq2xs_dequant_gemm_batch_indexed_dual_kernel``'s M_PAD==1 path, so it
    is bit-identical to it on every triton version; the caller applies
    :func:`swiglu_bf16`, matching the reference pipeline exactly.
    """
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

    out_base = pid_e * ROWS + crows
    tl.store(gate_out_ptr + out_base, gate_acc)
    tl.store(up_out_ptr + out_base, up_acc)

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


def preq_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to int8 with per-32-element fp32 scales."""
    k = x.shape[-1]
    xr = x.reshape(-1, k // 32, 32)
    scale = xr.abs().max(-1, keepdim=True).values / 127.0
    scale = torch.clamp(scale, min=1e-8)
    xq = (xr / scale).round().clamp(-128, 127).to(torch.int8)
    return xq.reshape_as(x), scale.reshape(-1, k // 32)


@triton.jit
def _iq2xs_dequant_gemm_indexed_dp4a_kernel(
    xq_ptr,
    xs_ptr,
    w_ptr,
    eids_ptr,
    grid_ptr,
    ksigns_ptr,
    out_ptr,
    E,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,
    BR: tl.constexpr,
):
    """Route-batched IQ2_XS dequant-GEMM with int8 dp4a inner product.

    Activations arrive int8-quantized (``xq`` + per-32 ``xs`` scales).  The
    packed IQ2 weights are decoded in-register to signed int8 magnitudes and
    the 8-j code inner product runs on the PTX ``dp4a`` instruction; the
    per-code scale (d x nibble x activation scale) is applied before the
    code reduction.  One block covers one (route, BR output rows) x full K.
    """
    pid_e = tl.program_id(0)
    pid_c = tl.program_id(1)
    eid = tl.load(eids_ptr + pid_e).to(tl.int32)
    r = pid_c * BR + tl.arange(0, BR)
    c32 = tl.arange(0, 32)
    acc = tl.zeros((BR,), dtype=tl.float32)
    for kb in range(0, K, 256):
        kb_block = kb // 256
        base = (
            w_ptr
            + eid.to(tl.int64) * ROWS * W_ROW_STRIDE
            + r.to(tl.int64) * W_ROW_STRIDE
            + kb_block * 74
        )
        lo = tl.load(base[:, None] + 2 + c32[None, :] * 2).to(tl.uint32)
        hi = tl.load(base[:, None] + 2 + c32[None, :] * 2 + 1).to(tl.uint32)
        codes = (lo | (hi << 8)) & 0xFFFF
        g = tl.load(grid_ptr + (codes & 511)).to(tl.int64)
        sb = tl.load(ksigns_ptr + (codes >> 9))
        d_lo = tl.load(base).to(tl.uint32)
        d_hi = tl.load(base + 1).to(tl.uint32)
        d_f = (d_lo.to(tl.uint16) | (d_hi.to(tl.uint16) << 8)).to(
            tl.float16, bitcast=True
        ).to(tl.float32)
        sc = tl.load(base[:, None] + 66 + (c32[None, :] // 4))
        nib = tl.where((c32[None, :] % 4) < 2, sc & 0xF, sc >> 4).to(tl.float32)
        res_code = tl.zeros((BR, 32), dtype=tl.int32)
        for half in tl.static_range(2):
            wp = tl.zeros((BR, 32), dtype=tl.int32)
            xp = tl.zeros((32,), dtype=tl.int32)
            for j in tl.static_range(4):
                jj = half * 4 + j
                mag = ((g >> (8 * jj)) & 0xFF).to(tl.int32)
                mag = tl.where((sb & (1 << jj)) != 0, -mag, mag) & 0xFF
                xv = (
                    tl.load(xq_ptr + pid_e.to(tl.int64) * K + kb + c32 * 8 + jj).to(tl.int32)
                    & 0xFF
                )
                wp = wp | (mag << (8 * j))
                xp = xp | (xv << (8 * j))
            xpb = tl.broadcast_to(xp[None, :], (BR, 32))
            acc0 = tl.zeros((BR, 32), dtype=tl.int32)
            ah = tl.inline_asm_elementwise(
                "dp4a.s32.s32 $0, $1, $2, $3;", "=r,r,r,r",
                [wp, xpb, acc0], tl.int32, is_pure=False, pack=1,
            )
            res_code = res_code + ah
        xs = tl.load(xs_ptr + pid_e.to(tl.int64) * (K // 32) + kb // 32 + c32 // 4)
        scale = d_f[:, None] * (0.5 + nib) * 0.25 * xs[None, :]
        acc += tl.sum(res_code.to(tl.float32) * scale, axis=1)
    tl.store(out_ptr + pid_e.to(tl.int64) * ROWS + r, acc)


@triton.jit
def _iq2xs_dequant_gemm_indexed_dual_dp4a_kernel(
    xq_ptr,
    xs_ptr,
    gate_w_ptr,
    up_w_ptr,
    eids_ptr,
    grid_ptr,
    ksigns_ptr,
    gate_out_ptr,
    up_out_ptr,
    E,
    K: tl.constexpr,
    ROWS: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,
    BR: tl.constexpr,
):
    """Dual-path (gate+up) dp4a IQ2 GEMM sharing activation loads."""
    pid_e = tl.program_id(0)
    pid_c = tl.program_id(1)
    eid = tl.load(eids_ptr + pid_e).to(tl.int32)
    r = pid_c * BR + tl.arange(0, BR)
    c32 = tl.arange(0, 32)
    gate_acc = tl.zeros((BR,), dtype=tl.float32)
    up_acc = tl.zeros((BR,), dtype=tl.float32)
    for kb in range(0, K, 256):
        kb_block = kb // 256
        gate_base = (
            gate_w_ptr
            + eid.to(tl.int64) * ROWS * W_ROW_STRIDE
            + r.to(tl.int64) * W_ROW_STRIDE
            + kb_block * 74
        )
        up_base = (
            up_w_ptr
            + eid.to(tl.int64) * ROWS * W_ROW_STRIDE
            + r.to(tl.int64) * W_ROW_STRIDE
            + kb_block * 74
        )
        g_lo = tl.load(gate_base[:, None] + 2 + c32[None, :] * 2).to(tl.uint32)
        g_hi = tl.load(gate_base[:, None] + 2 + c32[None, :] * 2 + 1).to(tl.uint32)
        g_codes = (g_lo | (g_hi << 8)) & 0xFFFF
        g_g = tl.load(grid_ptr + (g_codes & 511)).to(tl.int64)
        g_sb = tl.load(ksigns_ptr + (g_codes >> 9))
        g_d_lo = tl.load(gate_base).to(tl.uint32)
        g_d_hi = tl.load(gate_base + 1).to(tl.uint32)
        g_d = (g_d_lo.to(tl.uint16) | (g_d_hi.to(tl.uint16) << 8)).to(
            tl.float16, bitcast=True
        ).to(tl.float32)
        g_sc = tl.load(gate_base[:, None] + 66 + (c32[None, :] // 4))
        g_nib = tl.where((c32[None, :] % 4) < 2, g_sc & 0xF, g_sc >> 4).to(tl.float32)

        u_lo = tl.load(up_base[:, None] + 2 + c32[None, :] * 2).to(tl.uint32)
        u_hi = tl.load(up_base[:, None] + 2 + c32[None, :] * 2 + 1).to(tl.uint32)
        u_codes = (u_lo | (u_hi << 8)) & 0xFFFF
        u_g = tl.load(grid_ptr + (u_codes & 511)).to(tl.int64)
        u_sb = tl.load(ksigns_ptr + (u_codes >> 9))
        u_d_lo = tl.load(up_base).to(tl.uint32)
        u_d_hi = tl.load(up_base + 1).to(tl.uint32)
        u_d = (u_d_lo.to(tl.uint16) | (u_d_hi.to(tl.uint16) << 8)).to(
            tl.float16, bitcast=True
        ).to(tl.float32)
        u_sc = tl.load(up_base[:, None] + 66 + (c32[None, :] // 4))
        u_nib = tl.where((c32[None, :] % 4) < 2, u_sc & 0xF, u_sc >> 4).to(tl.float32)

        g_res = tl.zeros((BR, 32), dtype=tl.int32)
        u_res = tl.zeros((BR, 32), dtype=tl.int32)
        for half in tl.static_range(2):
            g_wp = tl.zeros((BR, 32), dtype=tl.int32)
            u_wp = tl.zeros((BR, 32), dtype=tl.int32)
            xp = tl.zeros((32,), dtype=tl.int32)
            for j in tl.static_range(4):
                jj = half * 4 + j
                g_mag = ((g_g >> (8 * jj)) & 0xFF).to(tl.int32)
                g_mag = tl.where((g_sb & (1 << jj)) != 0, -g_mag, g_mag) & 0xFF
                u_mag = ((u_g >> (8 * jj)) & 0xFF).to(tl.int32)
                u_mag = tl.where((u_sb & (1 << jj)) != 0, -u_mag, u_mag) & 0xFF
                xv = (
                    tl.load(xq_ptr + pid_e.to(tl.int64) * K + kb + c32 * 8 + jj).to(tl.int32)
                    & 0xFF
                )
                g_wp = g_wp | (g_mag << (8 * j))
                u_wp = u_wp | (u_mag << (8 * j))
                xp = xp | (xv << (8 * j))
            xpb = tl.broadcast_to(xp[None, :], (BR, 32))
            acc0 = tl.zeros((BR, 32), dtype=tl.int32)
            g_ah = tl.inline_asm_elementwise(
                "dp4a.s32.s32 $0, $1, $2, $3;", "=r,r,r,r",
                [g_wp, xpb, acc0], tl.int32, is_pure=False, pack=1,
            )
            u_ah = tl.inline_asm_elementwise(
                "dp4a.s32.s32 $0, $1, $2, $3;", "=r,r,r,r",
                [u_wp, xpb, acc0], tl.int32, is_pure=False, pack=1,
            )
            g_res = g_res + g_ah
            u_res = u_res + u_ah
        xs = tl.load(xs_ptr + pid_e.to(tl.int64) * (K // 32) + kb // 32 + c32 // 4)
        g_scale = g_d[:, None] * (0.5 + g_nib) * 0.25 * xs[None, :]
        u_scale = u_d[:, None] * (0.5 + u_nib) * 0.25 * xs[None, :]
        gate_acc += tl.sum(g_res.to(tl.float32) * g_scale, axis=1)
        up_acc += tl.sum(u_res.to(tl.float32) * u_scale, axis=1)
    out_base = pid_e.to(tl.int64) * ROWS + r
    tl.store(gate_out_ptr + out_base, gate_acc)
    tl.store(up_out_ptr + out_base, up_acc)


def iq2xs_dequant_gemm_indexed_dp4a(
    xq: torch.Tensor,
    xs: torch.Tensor,
    packed: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    rows: int,
    cols: int,
    grid_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    BR: int = 8,
) -> torch.Tensor:
    """dp4a route-batched ``x @ W_e^T``; xq/xs are preq-quantized activations."""
    E = expert_ids.numel()
    out = torch.empty((E, rows), dtype=torch.float32, device=xq.device)
    grid_t, ksigns_t, _ = grid_tables
    w_row_stride = _check_iq2xs_shape(rows, cols, BR)
    _iq2xs_dequant_gemm_indexed_dp4a_kernel[(E, rows // BR)](
        xq, xs, packed, expert_ids, grid_t, ksigns_t, out, E,
        K=cols, ROWS=rows, W_ROW_STRIDE=w_row_stride, BR=BR,
    )
    return out


def iq2xs_dequant_gemm_indexed_dual_dp4a(
    xq: torch.Tensor,
    xs: torch.Tensor,
    packed_gate: torch.Tensor,
    packed_up: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    rows: int,
    cols: int,
    grid_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    BR: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """dp4a dual-path gate+up GEMM over routed experts."""
    E = expert_ids.numel()
    gate_out = torch.empty((E, rows), dtype=torch.float32, device=xq.device)
    up_out = torch.empty_like(gate_out)
    grid_t, ksigns_t, _ = grid_tables
    w_row_stride = _check_iq2xs_shape(rows, cols, BR)
    _iq2xs_dequant_gemm_indexed_dual_dp4a_kernel[(E, rows // BR)](
        xq, xs, packed_gate, packed_up, expert_ids, grid_t, ksigns_t,
        gate_out, up_out, E, K=cols, ROWS=rows, W_ROW_STRIDE=w_row_stride, BR=BR,
    )
    return gate_out, up_out
