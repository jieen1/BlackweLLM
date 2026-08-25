"""Tensor-core GGML K-quant GEMM for the Qwen3.8 packed target.

The CUDA ABI in :mod:`runtime.kernels.gguf_qk` is the low-level fallback and
the graph-safe Q8_1 path.  This module is the BF16 tensor-core path used when
the runtime needs the closest arithmetic contract to ``F.linear`` without
materialising a model-sized dequantized weight cache.  Each program decodes a
small packed K tile into BF16 registers and immediately feeds it to
``tl.dot``; the GGML bytes remain the only resident parameter representation.

The four formats present in the Qwen3.8 UD checkpoint are accepted in both
their loader layout and the native split layout: Q4_K, Q5_K, Q6_K,
Q8_0, Q6_K_SPLIT, and Q8_0_SPLIT.  The split layouts keep the same row size
as GGML but move each block's FP16 ``d`` value into a row tail.  That lets the
decode kernel use aligned code loads without forcing tensor-core prefill to
materialize a second standard-format payload.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import torch
import triton
import triton.language as tl

_TYPE_IDS: Mapping[str, int] = {
    "Q4_K": 0,
    "Q5_K": 1,
    "Q6_K": 2,
    "Q8_0": 3,
    "Q6_K_SPLIT": 4,
    "Q8_0_SPLIT": 5,
}
_BLOCK_ELEMENTS = (256, 256, 256, 32, 256, 32)
# Smaller decode tiles reduce the live decoded-weight/register footprint for
# K-quant blocks while preserving the exact block boundaries.  Q8_0 is already
# a 32-element block and stays on its existing path.
_DECODE_ELEMENTS = (64, 64, 64, 32, 64, 32)
# Bytes occupied by the code/scale data for one block.  Split rows keep the
# original total row width, but their FP16 d values live in the row tail.
_DATA_BLOCK_BYTES = (144, 176, 210, 34, 208, 32)
_ROW_BLOCK_BYTES = (144, 176, 210, 34, 210, 34)


def _tensor_core_q6_small_m_down_enabled() -> bool:
    """Gate the narrow Q6 down-projection verify specialization."""

    return os.environ.get("QSR_GGUF_TC_Q6_SMALL_M_DOWN", "1").strip() != "0"


def _tensor_core_block_n(
    *,
    type_name: str | None = None,
    rows: int | None = None,
    n: int | None = None,
    k: int | None = None,
) -> int:
    """Return the packed tensor-core N tile, with an A/B override.

    The Q6 down projection is the one small-M shape where a 16-column tile
    wins on SM120.  Its ``N=5120,K=17408,M=8`` verify instance has enough
    K-side decode work that a 32-column program carries excess live state and
    lowers memory-level parallelism.  Keep the narrower tile local to the
    fixed DFlash2 verify shape; large prefill keeps the measured 32-column
    default, and an explicit environment value still wins for experiments.
    """

    raw = os.environ.get("QSR_GGUF_TC_BLOCK_N", "auto").strip().casefold()
    if raw not in {"", "auto"}:
        try:
            value = int(raw)
        except ValueError:
            return 32
        return value if value in (16, 32, 64, 128) else 32
    if (
        _tensor_core_q6_small_m_down_enabled()
        and type_name in {"Q6_K", "Q6_K_SPLIT"}
        and rows is not None
        and rows <= 8
        and n is not None
        and k is not None
        and n < k
    ):
        return 16
    return 32

def _tensor_core_block_m(
    rows: int,
    *,
    type_name: str | None = None,
) -> int:
    """Return the packed-K GEMM M tile, with an A/B override.

    ``BLOCK_M=8`` is the safe verify/decode tile.  Larger tiles let the
    Triton program reuse one decoded Q6 weight tile across more activation
    rows, which is the same reuse axis that SGLang's MMQ kernel exposes for
    batched work.  ``auto`` keeps the fixed-width DFlash2 verify tile at 8.
    Q5's decode footprint benefits from a 64-row tile as soon as the input is
    batched; Q6 reaches that point at 64 rows, while the smaller 32-row Q6
    tile avoids wasting half a program.  An explicit numeric value remains
    available for focused A/B runs.
    """

    raw = os.environ.get("QSR_GGUF_TC_BLOCK_M", "auto").strip().casefold()
    if raw in {"", "auto"}:
        if type_name == "Q5_K" and rows >= 32:
            return 64
        if type_name in {"Q6_K", "Q6_K_SPLIT"} and rows >= 64:
            return 64
        return 32 if rows >= 32 else 8
    try:
        value = int(raw)
    except ValueError:
        return 8
    return value if value in (8, 16, 32, 64) else 8


def _tensor_core_num_warps(type_name: str, rows: int) -> int:
    """Return the packed tensor-core warp count for this format and shape.

    Q5_K has a more expensive two-plane high-bit decode than Q6_K.  On the
    large prefill tile, the extra resident warps hide that decode latency and
    materially improve occupancy on SM120.  The fixed M=8 verify tile is
    still faster with four warps, so keep the specialization shape-aware.
    An explicit environment value remains an override for A/B experiments.
    """

    raw = os.environ.get("QSR_GGUF_TC_WARPS", "auto").strip().casefold()
    if raw in {"", "auto"}:
        return 8 if type_name == "Q5_K" and rows >= 32 else 4
    try:
        value = int(raw)
    except ValueError:
        return 4
    return value if value in (2, 4, 8) else 4


def _tensor_core_num_stages(type_name: str, rows: int) -> int:
    """Return the packed tensor-core pipeline depth for this shape."""

    raw = os.environ.get("QSR_GGUF_TC_STAGES", "auto").strip().casefold()
    if raw in {"", "auto"}:
        # Q6's inline decode has a larger live register footprint than the
        # generic K-quant path.  A second Triton pipeline stage therefore
        # lowers SM120 occupancy for the fixed DFlash2 M=8 tile instead of
        # hiding useful memory latency.  Keep the measured stage-1 default;
        # stage-2 remains available as an explicit A/B override.
        return 2 if type_name == "Q5_K" and rows >= 32 else 1
    try:
        value = int(raw)
    except ValueError:
        return 1
    return value if value in (1, 2, 3, 4) else 1


def _tensor_core_decode_elements(type_id: int, *, n: int, k: int, rows: int | None = None) -> int:
    """Choose the K sub-tile without crossing a packed-format block.

    Small sub-tiles reduce the live Q4/Q5/Q6 decode footprint for output-wide
    projections.  Large input-wide down projections retain one full
    256-value K-quant block; the fixed M=8 Q6 verify shape is the measured
    exception.  Q8_0 already has 32-value blocks.
    The environment override is intentionally narrow and exists for kernel
    A/B runs, not as a model-level tuning knob.
    """

    raw = os.environ.get("QSR_GGUF_TC_DECODE_ELEMENTS", "auto").strip().casefold()
    if raw not in {"", "auto"}:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value in {32, 64, 128, 256} and value <= _BLOCK_ELEMENTS[type_id]:
            return value
    if type_id in {3, 5}:
        return _DECODE_ELEMENTS[type_id]
    if (
        type_id in {2, 4}
        and (rows is None or rows <= 8)
        and _tensor_core_q6_small_m_down_enabled()
    ):
        # A 128-value Q6 subtile is the measured SM120 minimum for both
        # DFlash2 M=8 MLP directions: it halves the live decode footprint
        # versus a full block without the extra loop overhead of 64 values.
        # The larger prefill path retains the old shape-dependent choice.
        return 128
    return _DECODE_ELEMENTS[type_id] if n >= k else _BLOCK_ELEMENTS[type_id]


@triton.jit
def _gguf_qk_gemm_kernel(
    x_ptr,
    packed_ptr,
    out_ptr,
    M,
    N,
    K: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    TYPE_ID: tl.constexpr,
    BLOCK_ELEMENTS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DECODE_ELEMENTS: tl.constexpr,
    TILE_MAJOR: tl.constexpr,
    TILE_BYTES: tl.constexpr,
):
    """Decode one packed weight tile and run a BF16 tensor-core dot."""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    column_offsets = tl.arange(0, BLOCK_N)
    columns = pid_n * BLOCK_N + column_offsets
    row_mask = rows < M
    column_mask = columns < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, DECODE_ELEMENTS):
        block_index = k0 // BLOCK_ELEMENTS
        block_offset = k0 - block_index * BLOCK_ELEMENTS
        values = block_offset + tl.arange(0, DECODE_ELEMENTS)
        if TILE_MAJOR:
            tile_base = packed_ptr + pid_n * TILE_BYTES
            row_base = tile_base + column_offsets[:, None] * BLOCK_BYTES
            weight_base = row_base + block_index * BLOCK_BYTES * BLOCK_N
        else:
            row_base = packed_ptr + columns[:, None] * ROW_BYTES
            weight_base = row_base + block_index * BLOCK_BYTES
        if TYPE_ID == 2:
            d_base = weight_base + 208
        elif TYPE_ID == 4:
            d_base = (
                weight_base + 208
                if TILE_MAJOR
                else row_base + (K // BLOCK_ELEMENTS) * 208 + block_index * 2
            )
        elif TYPE_ID == 5:
            d_base = (
                weight_base + 32
                if TILE_MAJOR
                else row_base + (K // BLOCK_ELEMENTS) * 32 + block_index * 2
            )
        else:
            d_base = weight_base

        if TYPE_ID == 0:
            chunk = values // 64
            local = values % 64
            nibble = local // 32
            qbyte = local % 32
            packed_q = tl.load(
                weight_base + 16 + chunk[None, :] * 32 + qbyte[None, :],
                mask=column_mask[:, None],
                other=0,
            ).to(tl.uint32)
            q = (packed_q >> (nibble[None, :] * 4)) & 0x0F
            scale_index = chunk * 2 + nibble
            scale_bytes = weight_base + 4
            scale_lo = tl.load(
                scale_bytes + scale_index[None, :],
                mask=column_mask[:, None],
                other=0,
            ).to(tl.uint32)
            scale_hi = tl.load(
                scale_bytes + (scale_index + 4)[None, :],
                mask=column_mask[:, None],
                other=0,
            ).to(tl.uint32)
            scale_prev = tl.load(
                scale_bytes + (scale_index - 4)[None, :],
                mask=column_mask[:, None],
                other=0,
            ).to(tl.uint32)
            scale = tl.where(
                scale_index[None, :] < 4,
                scale_lo & 0x3F,
                (scale_hi & 0x0F) | ((scale_prev >> 6) << 4),
            )
            min_value = tl.where(
                scale_index[None, :] < 4,
                scale_hi & 0x3F,
                (scale_hi >> 4) | ((scale_lo >> 6) << 4),
            )
            d_bits = tl.load(
                weight_base.to(tl.pointer_type(tl.uint16)),
                mask=column_mask[:, None],
                other=0,
            )
            dmin_bits = tl.load(weight_base + 2, mask=column_mask[:, None], other=0).to(
                tl.uint16
            ) | (tl.load(weight_base + 3, mask=column_mask[:, None], other=0).to(tl.uint16) << 8)
            d = d_bits.to(tl.float16, bitcast=True).to(tl.float32)
            dmin = dmin_bits.to(tl.float16, bitcast=True).to(tl.float32)
            weight = (
                d * scale.to(tl.float32) * q.to(tl.float32)
                - dmin * min_value.to(tl.float32)
            )
        elif TYPE_ID == 1:
            chunk = values // 64
            local = values % 64
            nibble = local // 32
            qbyte = local % 32
            low = tl.load(
                weight_base + 48 + chunk[None, :] * 32 + qbyte[None, :],
                mask=column_mask[:, None],
                other=0,
            ).to(tl.uint32)
            high = tl.load(
                weight_base + 16 + qbyte[None, :],
                mask=column_mask[:, None],
                other=0,
            ).to(tl.uint32)
            high_bit = (high >> (chunk[None, :] * 2 + nibble[None, :])) & 1
            q = ((low >> (nibble[None, :] * 4)) & 0x0F) | (high_bit << 4)
            scale_index = chunk * 2 + nibble
            scale_bytes = weight_base + 4
            scale_lo = tl.load(
                scale_bytes + scale_index[None, :],
                mask=column_mask[:, None],
                other=0,
            ).to(tl.uint32)
            scale_hi = tl.load(
                scale_bytes + (scale_index + 4)[None, :],
                mask=column_mask[:, None],
                other=0,
            ).to(tl.uint32)
            scale_prev = tl.load(
                scale_bytes + (scale_index - 4)[None, :],
                mask=column_mask[:, None],
                other=0,
            ).to(tl.uint32)
            scale = tl.where(
                scale_index[None, :] < 4,
                scale_lo & 0x3F,
                (scale_hi & 0x0F) | ((scale_prev >> 6) << 4),
            )
            min_value = tl.where(
                scale_index[None, :] < 4,
                scale_hi & 0x3F,
                (scale_hi >> 4) | ((scale_lo >> 6) << 4),
            )
            d_bits = tl.load(
                weight_base.to(tl.pointer_type(tl.uint16)),
                mask=column_mask[:, None],
                other=0,
            )
            dmin_bits = tl.load(weight_base + 2, mask=column_mask[:, None], other=0).to(
                tl.uint16
            ) | (tl.load(weight_base + 3, mask=column_mask[:, None], other=0).to(tl.uint16) << 8)
            d = d_bits.to(tl.float16, bitcast=True).to(tl.float32)
            dmin = dmin_bits.to(tl.float16, bitcast=True).to(tl.float32)
            weight = (
                d * scale.to(tl.float32) * q.to(tl.float32)
                - dmin * min_value.to(tl.float32)
            )
        elif TYPE_ID == 2 or TYPE_ID == 4:
            half = values // 128
            local = values % 128
            group = local // 32
            j = local % 32
            low_offset = half * 64 + (group & 1) * 32 + j
            high_offset = half * 32 + j
            low_shift = tl.where(group >= 2, 4, 0)
            high_shift = group * 2
            low = tl.load(
                weight_base + low_offset[None, :],
                mask=column_mask[:, None],
                other=0,
            ).to(tl.uint32)
            high = tl.load(
                weight_base + 128 + high_offset[None, :],
                mask=column_mask[:, None],
                other=0,
            ).to(tl.uint32)
            q = ((low >> low_shift[None, :]) & 0x0F) | (
                ((high >> high_shift[None, :]) & 0x03) << 4
            )
            scale_index = half * 8 + group * 2 + j // 16
            scale_byte = (
                tl.load(
                    weight_base + 192 + scale_index[None, :],
                    mask=column_mask[:, None],
                    other=0,
                )
                .to(tl.int8, bitcast=True)
                .to(tl.float32)
            )
            d_bits = tl.load(
                d_base.to(tl.pointer_type(tl.uint16)),
                mask=column_mask[:, None],
                other=0,
            )
            d = d_bits.to(tl.float16, bitcast=True).to(tl.float32)
            weight = d * scale_byte * (q.to(tl.float32) - 32.0)
        else:
            code_offset = 0 if TYPE_ID == 5 else 2
            q = (
                tl.load(
                    weight_base + code_offset + values[None, :],
                    mask=column_mask[:, None],
                    other=0,
                )
                .to(tl.int8, bitcast=True)
                .to(tl.float32)
            )
            d_bits = tl.load(
                d_base.to(tl.pointer_type(tl.uint16)),
                mask=column_mask[:, None],
                other=0,
            )
            d = d_bits.to(tl.float16, bitcast=True).to(tl.float32)
            weight = d * q

        activation = tl.load(
            x_ptr + rows[:, None] * K + block_index * BLOCK_ELEMENTS + values[None, :],
            mask=row_mask[:, None],
            other=0,
        )
        acc = tl.dot(activation, tl.trans(weight.to(tl.bfloat16)), acc)

    tl.store(
        out_ptr + rows[:, None] * N + columns[None, :],
        acc.to(tl.bfloat16),
        mask=row_mask[:, None] & column_mask[None, :],
    )



def gguf_qk_gemm(
    x: torch.Tensor,
    packed: torch.Tensor,
    *,
    m: int,
    n: int,
    k: int,
    row_bytes: int,
    type_name: str,
) -> torch.Tensor:
    """Run a packed GGML K-quant GEMM through BF16 tensor cores."""

    type_id = _TYPE_IDS.get(type_name)
    if type_id is None:
        raise ValueError(f"unsupported GGUF Triton type {type_name!r}")
    if x.device.type != "cuda" or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("GGUF Triton GEMM expects contiguous CUDA BF16 input")
    if packed.device != x.device or packed.dtype != torch.uint8 or not packed.is_contiguous():
        raise ValueError("GGUF Triton GEMM expects a contiguous CUDA uint8 payload")
    if x.shape != (m, k):
        raise ValueError(f"GGUF Triton input shape {tuple(x.shape)} != ({m}, {k})")
    if packed.numel() != n * row_bytes:
        raise ValueError(f"GGUF Triton payload {packed.numel()} != {n * row_bytes}")
    if (
        k % _BLOCK_ELEMENTS[type_id]
        or row_bytes != (k // _BLOCK_ELEMENTS[type_id]) * _ROW_BLOCK_BYTES[type_id]
    ):
        raise ValueError("GGUF Triton GEMM geometry does not match the packed format")

    block_m = _tensor_core_block_m(m, type_name=type_name)
    block_n = _tensor_core_block_n(type_name=type_name, rows=m, n=n, k=k)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _gguf_qk_gemm_kernel[grid](
        x,
        packed,
        out,
        m,
        n,
        K=k,
        ROW_BYTES=row_bytes,
        TYPE_ID=type_id,
        BLOCK_ELEMENTS=_BLOCK_ELEMENTS[type_id],
        BLOCK_BYTES=_DATA_BLOCK_BYTES[type_id],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        DECODE_ELEMENTS=_tensor_core_decode_elements(type_id, n=n, k=k, rows=m),
        TILE_MAJOR=False,
        TILE_BYTES=0,
        num_warps=_tensor_core_num_warps(type_name, m),
        num_stages=_tensor_core_num_stages(type_name, m),
    )
    return out


def gguf_qk_repack_for_tensor_core(
    packed: torch.Tensor,
    *,
    n: int,
    k: int,
    row_bytes: int,
    type_name: str,
    block_n: int,
) -> tuple[torch.Tensor, int]:
    """Repack GGUF rows as ``[N-tile, K-block, N-lane, byte]``.

    The regular loader layout is row-major: adjacent output rows are separated
    by the complete K row.  The tensor-core decoder consumes one N tile at a
    time, so that layout makes the same K-block bytes arrive with a large
    stride.  This representation keeps a block's output rows adjacent while
    preserving the exact GGML bytes.  It is an opt-in experiment; callers own
    the returned allocation and may cache it for graph replay.
    """

    type_id = _TYPE_IDS.get(type_name)
    if type_id is None:
        raise ValueError(f"unsupported GGUF Triton type {type_name!r}")
    if block_n not in (16, 32, 64, 128):
        raise ValueError(f"unsupported tensor-core tile width {block_n}")
    block_elements = _BLOCK_ELEMENTS[type_id]
    block_bytes = _ROW_BLOCK_BYTES[type_id]
    blocks = k // block_elements
    if k % block_elements or row_bytes != blocks * block_bytes:
        raise ValueError("GGUF repack geometry does not match the packed format")
    if packed.device.type != "cuda" or packed.dtype != torch.uint8 or not packed.is_contiguous():
        raise ValueError("GGUF repack expects contiguous CUDA uint8 storage")
    if packed.numel() != n * row_bytes:
        raise ValueError(f"GGUF repack payload {packed.numel()} != {n * row_bytes}")

    source = packed.view(n, row_bytes)
    if type_id in {4, 5}:
        data_bytes = _DATA_BLOCK_BYTES[type_id]
        data = source[:, : blocks * data_bytes].view(n, blocks, data_bytes)
        scales = source[:, blocks * data_bytes :].view(n, blocks, 2)
        interleaved = torch.empty(
            (n, blocks, block_bytes), dtype=packed.dtype, device=packed.device
        )
        interleaved[..., :data_bytes].copy_(data)
        interleaved[..., data_bytes:].copy_(scales)
    else:
        interleaved = source.view(n, blocks, block_bytes)

    padded_n = triton.cdiv(n, block_n) * block_n
    if padded_n != n:
        padded = torch.zeros(
            (padded_n, blocks, block_bytes), dtype=packed.dtype, device=packed.device
        )
        padded[:n].copy_(interleaved)
        interleaved = padded
    tile_major = interleaved.view(padded_n // block_n, block_n, blocks, block_bytes)
    tile_major = tile_major.permute(0, 2, 1, 3).contiguous()
    return tile_major.reshape(-1), padded_n


def gguf_qk_gemm_tile_major(
    x: torch.Tensor,
    packed: torch.Tensor,
    *,
    m: int,
    n: int,
    k: int,
    type_name: str,
    block_n: int,
) -> torch.Tensor:
    """Run the experimental tile-major tensor-core decoder."""

    type_id = _TYPE_IDS.get(type_name)
    if type_id is None:
        raise ValueError(f"unsupported GGUF Triton type {type_name!r}")
    if x.device.type != "cuda" or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("GGUF Triton GEMM expects contiguous CUDA BF16 input")
    if packed.device != x.device or packed.dtype != torch.uint8 or not packed.is_contiguous():
        raise ValueError("GGUF Triton GEMM expects a contiguous CUDA uint8 payload")
    if x.shape != (m, k):
        raise ValueError(f"GGUF Triton input shape {tuple(x.shape)} != ({m}, {k})")
    block_elements = _BLOCK_ELEMENTS[type_id]
    block_bytes = _ROW_BLOCK_BYTES[type_id]
    blocks = k // block_elements
    padded_n = triton.cdiv(n, block_n) * block_n
    tile_bytes = blocks * block_bytes * block_n
    if k % block_elements or packed.numel() != (padded_n // block_n) * tile_bytes:
        raise ValueError("GGUF tile-major payload geometry does not match the packed format")

    block_m = _tensor_core_block_m(m, type_name=type_name)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    grid = (triton.cdiv(m, block_m), padded_n // block_n)
    _gguf_qk_gemm_kernel[grid](
        x,
        packed,
        out,
        m,
        n,
        K=k,
        ROW_BYTES=0,
        TYPE_ID=type_id,
        BLOCK_ELEMENTS=block_elements,
        BLOCK_BYTES=block_bytes,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        DECODE_ELEMENTS=_tensor_core_decode_elements(type_id, n=n, k=k, rows=m),
        TILE_MAJOR=True,
        TILE_BYTES=tile_bytes,
        num_warps=_tensor_core_num_warps(type_name, m),
        num_stages=_tensor_core_num_stages(type_name, m),
    )
    return out
