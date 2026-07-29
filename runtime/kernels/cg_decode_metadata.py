"""Small fused metadata update for Laguna's B=1 decode CUDA graph path."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _write_b1_decode_metadata(
    values,
    input_ids,
    positions,
    full_slot_mapping,
    swa_slot_mapping,
    full_cache_seqlens,
    swa_cache_seqlens,
    full_slot_base: tl.constexpr,
    swa_ring_base: tl.constexpr,
    ring_slots: tl.constexpr,
    swa_window: tl.constexpr,
    block_size: tl.constexpr,
):
    token_id = tl.load(values)
    kv_len = tl.load(values + 1)

    tl.store(input_ids, token_id)
    tl.store(positions, kv_len)
    tl.store(full_slot_mapping, full_slot_base * block_size + kv_len)

    new_kv = kv_len + 1
    tl.store(full_cache_seqlens, new_kv)

    window_start = tl.maximum(0, kv_len - swa_window + 1)
    aligned_start = (window_start // block_size) * block_size
    tl.store(swa_cache_seqlens, new_kv - aligned_start)
    ring_position = kv_len % ring_slots
    tl.store(
        swa_slot_mapping,
        (swa_ring_base + ring_position // block_size) * block_size + ring_position % block_size,
    )


def write_laguna_b1_decode_metadata(
    values: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    full_slot_mapping: torch.Tensor,
    swa_slot_mapping: torch.Tensor,
    full_cache_seqlens: torch.Tensor,
    swa_cache_seqlens: torch.Tensor,
    *,
    full_slot_base: int,
    swa_ring_base: int,
    ring_slots: int,
    swa_window: int,
    block_size: int,
) -> None:
    """Write B=1 input and the two Laguna attention-group metadata records.

    The caller has already copied ``[token_id, kv_len]`` to ``values`` with
    one pinned H2D transfer.  This replaces six Python tensor scalar writes
    on the streaming path.  It intentionally covers only Laguna's one full
    and one SWA group; other layouts retain the generic correct path.
    """
    _write_b1_decode_metadata[(1,)](
        values,
        input_ids,
        positions,
        full_slot_mapping,
        swa_slot_mapping,
        full_cache_seqlens,
        swa_cache_seqlens,
        full_slot_base=full_slot_base,
        swa_ring_base=swa_ring_base,
        ring_slots=ring_slots,
        swa_window=swa_window,
        block_size=block_size,
    )
