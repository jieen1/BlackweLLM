"""CUDA-Graph-safe decode index generation for the DSV4 attention layers.

The eager ``window_topk_idxs`` / ``compress_topk_idxs`` build per-step
attention index tensors from a Python ``start_pos`` using torch.arange /
cat / pad. Those ops are capturable, but the values depend on the current
decode position, which a replayed graph must read from device memory.

This module regenerates the same decode layouts from device-side position
scalars. It now also supports true batched decode:

- ``positions`` is a 1-D ``[B]`` int64 tensor, one decode position per row.
- Optional ``slot_ids`` + page metadata shift each row into its slot-global
  raw cache-id space while preserving the ``-1`` sentinel.
- The legacy B1 call contract is unchanged: callers that pass only one
  position row and no slot offsets still receive the same local ids.

The compressed helper remains generic over the compression ratio, so the same
surface serves both ratio-4 and ratio-128 DSV4 layers.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

_INT32_MAX = (1 << 31) - 1
_DEFAULT_NUM_WARPS = 1
_DEFAULT_NUM_STAGES = 1


def _resolve_device(device: torch.device | str | None, tensor: torch.Tensor) -> torch.device:
    return torch.device(device) if device is not None else tensor.device


def _as_1d_int64(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if tensor.ndim == 0:
        tensor = tensor.reshape(1)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.int64:
        raise ValueError(f"{name} must be int64, got {tensor.dtype}")
    same_device = (
        tensor.device.type == device.type
        and (device.index is None or tensor.device.index == device.index)
    )
    if not same_device:
        raise ValueError(f"{name} device {tensor.device} does not match {device}")
    return tensor


def _prepare_inputs(
    positions: torch.Tensor,
    *,
    device: torch.device | str | None,
    slot_ids: torch.Tensor | None,
    pages_per_slot: int | None,
    page_size: int | None,
    max_local_id: int,
) -> tuple[torch.device, torch.Tensor, torch.Tensor | None, int]:
    resolved = _resolve_device(device, positions)
    positions = _as_1d_int64("positions", positions, device=resolved)
    if pages_per_slot is not None and pages_per_slot <= 0:
        raise ValueError(f"pages_per_slot must be positive, got {pages_per_slot}")
    if page_size is not None and page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    if slot_ids is None:
        if pages_per_slot is not None or page_size is not None:
            raise ValueError("pages_per_slot/page_size require slot_ids")
        return resolved, positions, None, 0

    if pages_per_slot is None or page_size is None:
        raise ValueError("slot_ids require both pages_per_slot and page_size")
    slot_ids = _as_1d_int64("slot_ids", slot_ids, device=resolved)
    if slot_ids.shape != positions.shape:
        raise ValueError(
            "slot_ids shape "
            f"{tuple(slot_ids.shape)} does not match positions {tuple(positions.shape)}"
        )

    page_offset_scale = pages_per_slot * page_size
    if resolved.type != "cuda" and slot_ids.numel():
        if int(slot_ids.min().item()) < 0:
            raise ValueError("slot_ids must be non-negative")
        max_raw_id = int(slot_ids.max().item()) * page_offset_scale + max_local_id
        if max_raw_id > _INT32_MAX:
            raise ValueError(
                f"raw decode index {max_raw_id} exceeds int32 capacity {_INT32_MAX}"
            )
    return resolved, positions, slot_ids, page_offset_scale


def _decode_swa_indices_reference(
    positions: torch.Tensor,
    window: int,
    *,
    slot_ids: torch.Tensor | None,
    page_offset_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cols = torch.arange(window, dtype=torch.int64, device=positions.device).unsqueeze(0)
    rows = positions.unsqueeze(1)
    full = rows >= (window - 1)
    local = torch.where(
        full,
        (rows - window + 1 + cols) % window,
        torch.where(cols <= rows, cols, -1),
    )
    if slot_ids is not None:
        offsets = slot_ids.unsqueeze(1) * page_offset_scale
        local = torch.where(local >= 0, local + offsets, -1)
    lengths = torch.minimum(rows.squeeze(1) + 1, torch.full_like(positions, window))
    return local.to(torch.int32).contiguous(), lengths.to(torch.int32).contiguous()


def _decode_comp_indices_reference(
    positions: torch.Tensor,
    ratio: int,
    max_comp: int,
    *,
    slot_ids: torch.Tensor | None,
    page_offset_scale: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cols = torch.arange(max_comp, dtype=torch.int64, device=positions.device).unsqueeze(0)
    counts = ((positions + 1) // ratio).unsqueeze(1)
    local = torch.where(cols < counts, cols, -1)
    if slot_ids is not None:
        offsets = slot_ids.unsqueeze(1) * page_offset_scale
        local = torch.where(local >= 0, local + offsets, -1)
    lengths = torch.minimum(counts.squeeze(1), torch.full_like(positions, max_comp))
    return local.to(torch.int32).contiguous(), lengths.to(torch.int32).contiguous()


@triton.jit
def _decode_swa_indices_kernel(
    positions_ptr,  # [B] int64
    swa_ptr,  # [B, win] int32 out
    lengths_ptr,  # [B] int32 out
    swa_row_stride,
    win: tl.constexpr,
):
    col = tl.program_id(0)
    row = tl.program_id(1)
    pos = tl.load(positions_ptr + row)
    full = pos >= (win - 1)
    local = tl.where(full, (pos - win + 1 + col) % win, tl.where(col <= pos, col, -1))
    tl.store(swa_ptr + row * swa_row_stride + col, local.to(tl.int32))
    if col == 0:
        tl.store(lengths_ptr + row, tl.minimum(pos + 1, win).to(tl.int32))


@triton.jit
def _decode_swa_indices_raw_kernel(
    positions_ptr,  # [B] int64
    slot_ids_ptr,  # [B] int64
    swa_ptr,  # [B, win] int32 out
    lengths_ptr,  # [B] int32 out
    swa_row_stride,
    page_offset_scale,
    win: tl.constexpr,
):
    col = tl.program_id(0)
    row = tl.program_id(1)
    pos = tl.load(positions_ptr + row)
    slot = tl.load(slot_ids_ptr + row)
    full = pos >= (win - 1)
    local = tl.where(full, (pos - win + 1 + col) % win, tl.where(col <= pos, col, -1))
    shifted = tl.where(local >= 0, local + slot * page_offset_scale, -1)
    tl.store(swa_ptr + row * swa_row_stride + col, shifted.to(tl.int32))
    if col == 0:
        tl.store(lengths_ptr + row, tl.minimum(pos + 1, win).to(tl.int32))


@triton.jit
def _decode_comp_indices_kernel(
    positions_ptr,  # [B] int64
    comp_ptr,  # [B, max_comp] int32 out
    lengths_ptr,  # [B] int32 out
    comp_row_stride,
    ratio: tl.constexpr,
    max_comp: tl.constexpr,
):
    col = tl.program_id(0)
    row = tl.program_id(1)
    pos = tl.load(positions_ptr + row)
    count = (pos + 1) // ratio
    tl.store(
        comp_ptr + row * comp_row_stride + col,
        tl.where(col < count, col.to(tl.int64), -1).to(tl.int32),
    )
    if col == 0:
        tl.store(lengths_ptr + row, tl.minimum(count, max_comp).to(tl.int32))


@triton.jit
def _decode_comp_indices_raw_kernel(
    positions_ptr,  # [B] int64
    slot_ids_ptr,  # [B] int64
    comp_ptr,  # [B, max_comp] int32 out
    lengths_ptr,  # [B] int32 out
    comp_row_stride,
    page_offset_scale,
    ratio: tl.constexpr,
    max_comp: tl.constexpr,
):
    col = tl.program_id(0)
    row = tl.program_id(1)
    pos = tl.load(positions_ptr + row)
    slot = tl.load(slot_ids_ptr + row)
    count = (pos + 1) // ratio
    local = tl.where(col < count, col.to(tl.int64), -1)
    shifted = tl.where(local >= 0, local + slot * page_offset_scale, -1)
    tl.store(comp_ptr + row * comp_row_stride + col, shifted.to(tl.int32))
    if col == 0:
        tl.store(lengths_ptr + row, tl.minimum(count, max_comp).to(tl.int32))


def decode_swa_indices(
    positions: torch.Tensor,
    window: int,
    *,
    device: torch.device | str | None = None,
    slot_ids: torch.Tensor | None = None,
    pages_per_slot: int | None = None,
    page_size: int | None = None,
    return_lengths: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Ring-ordered decode-window ids for each row in ``positions``.

    With only ``positions`` the result matches the historical B1 contract:
    ``[B, window]`` int32 local ring ids, padded with ``-1``.

    When ``slot_ids`` and page metadata are provided, every non-sentinel entry
    is shifted into the slot-global raw cache-id space:

    ``raw_id = local_id + slot_id * pages_per_slot * page_size``

    Set ``return_lengths=True`` to also receive the valid-count vector
    ``[B]`` int32.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    resolved, positions, slot_ids, page_offset_scale = _prepare_inputs(
        positions,
        device=device,
        slot_ids=slot_ids,
        pages_per_slot=pages_per_slot,
        page_size=page_size,
        max_local_id=max(window - 1, 0),
    )
    batch = int(positions.shape[0])
    out = torch.empty((batch, window), dtype=torch.int32, device=resolved)
    lengths = torch.empty((batch,), dtype=torch.int32, device=resolved)
    if batch == 0:
        return (out, lengths) if return_lengths else out

    if resolved.type != "cuda":
        out, lengths = _decode_swa_indices_reference(
            positions,
            window,
            slot_ids=slot_ids,
            page_offset_scale=page_offset_scale,
        )
    else:
        grid = (window, batch)
        if slot_ids is None:
            _decode_swa_indices_kernel[grid](positions, out, lengths, out.stride(0), win=window)
        else:
            _decode_swa_indices_raw_kernel[grid](
                positions,
                slot_ids,
                out,
                lengths,
                out.stride(0),
                page_offset_scale,
                win=window,
            )
    if return_lengths:
        return out, lengths
    return out


def decode_comp_indices(
    positions: torch.Tensor,
    ratio: int,
    max_comp: int,
    *,
    device: torch.device | str | None = None,
    slot_ids: torch.Tensor | None = None,
    pages_per_slot: int | None = None,
    page_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compressed decode ids for every row in ``positions``.

    Returns ``([B, max_comp] int32 ids, [B] int32 lengths)``. Values past the
    valid prefix are ``-1``. When slot metadata is present the ids are shifted
    into the slot-global raw cache-id space with the same rule as
    :func:`decode_swa_indices`.
    """
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if max_comp < 0:
        raise ValueError(f"max_comp must be non-negative, got {max_comp}")
    resolved, positions, slot_ids, page_offset_scale = _prepare_inputs(
        positions,
        device=device,
        slot_ids=slot_ids,
        pages_per_slot=pages_per_slot,
        page_size=page_size,
        max_local_id=max(max_comp - 1, 0),
    )
    batch = int(positions.shape[0])
    comp = torch.empty((batch, max_comp), dtype=torch.int32, device=resolved)
    lengths = torch.empty((batch,), dtype=torch.int32, device=resolved)
    if batch == 0 or max_comp == 0:
        lengths.zero_()
        return comp, lengths

    if resolved.type != "cuda":
        return _decode_comp_indices_reference(
            positions,
            ratio,
            max_comp,
            slot_ids=slot_ids,
            page_offset_scale=page_offset_scale,
        )

    grid = (max_comp, batch)
    if slot_ids is None:
        _decode_comp_indices_kernel[grid](
            positions,
            comp,
            lengths,
            comp.stride(0),
            ratio=ratio,
            max_comp=max_comp,
        )
    else:
        _decode_comp_indices_raw_kernel[grid](
            positions,
            slot_ids,
            comp,
            lengths,
            comp.stride(0),
            page_offset_scale,
            ratio=ratio,
            max_comp=max_comp,
        )
    return comp, lengths


def compile_decode_swa_indices_sm120(*, window: int, with_slot_offsets: bool = False):
    """Offline-compile the decode SWA index kernel for SM120."""
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    kernel = _decode_swa_indices_raw_kernel if with_slot_offsets else _decode_swa_indices_kernel
    signature = {
        "positions_ptr": "*i64",
        "swa_ptr": "*i32",
        "lengths_ptr": "*i32",
        "swa_row_stride": "i64",
        "win": "constexpr",
    }
    if with_slot_offsets:
        signature = {
            "positions_ptr": "*i64",
            "slot_ids_ptr": "*i64",
            "swa_ptr": "*i32",
            "lengths_ptr": "*i32",
            "swa_row_stride": "i64",
            "page_offset_scale": "i64",
            "win": "constexpr",
        }
    src = ASTSource(
        fn=kernel,
        signature=signature,
        constexprs={"win": window},
    )
    return triton.compile(
        src,
        target=GPUTarget("cuda", 120, 32),
        options={
            "num_warps": _DEFAULT_NUM_WARPS,
            "num_stages": _DEFAULT_NUM_STAGES,
        },
    )


def compile_decode_comp_indices_sm120(
    *,
    ratio: int,
    max_comp: int,
    with_slot_offsets: bool = False,
):
    """Offline-compile the decode compressed-index kernel for SM120."""
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if max_comp < 0:
        raise ValueError(f"max_comp must be non-negative, got {max_comp}")
    kernel = _decode_comp_indices_raw_kernel if with_slot_offsets else _decode_comp_indices_kernel
    signature = {
        "positions_ptr": "*i64",
        "comp_ptr": "*i32",
        "lengths_ptr": "*i32",
        "comp_row_stride": "i64",
        "ratio": "constexpr",
        "max_comp": "constexpr",
    }
    if with_slot_offsets:
        signature = {
            "positions_ptr": "*i64",
            "slot_ids_ptr": "*i64",
            "comp_ptr": "*i32",
            "lengths_ptr": "*i32",
            "comp_row_stride": "i64",
            "page_offset_scale": "i64",
            "ratio": "constexpr",
            "max_comp": "constexpr",
        }
    src = ASTSource(
        fn=kernel,
        signature=signature,
        constexprs={
            "ratio": ratio,
            "max_comp": max_comp,
        },
    )
    return triton.compile(
        src,
        target=GPUTarget("cuda", 120, 32),
        options={
            "num_warps": _DEFAULT_NUM_WARPS,
            "num_stages": _DEFAULT_NUM_STAGES,
        },
    )
