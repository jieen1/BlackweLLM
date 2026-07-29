"""Regression test for ``LagunaBackend.reset_slot``'s KV-cache block axis.

``kv_caches[name]`` is ``[2, num_blocks, block_size, num_kv_heads,
head_dim]`` -- dim 0 is K/V, dim 1 is the block axis
(``runtime/backends/laguna.py``, the ``torch.zeros(shape, ...)``
allocation). ``reset_slot`` originally sliced dim 0 with a block range,
which silently:

* clamped to ``[0:2]`` for slot 0, wiping **every** slot's blocks, and
* produced an **empty** slice for every slot >= 1, clearing nothing --
  so the previous request's KV survived into the next one.

Both were masked at ``num_slots == 1`` (DFlash pins capacity to 1),
where ``num_blocks == blocks_per_slot`` makes the two forms equivalent.
These tests exercise the slicing arithmetic directly on plain tensors so
they need neither a GPU nor model weights.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

BLOCK_SIZE = 128
BLOCKS_PER_SLOT = 130
NUM_KV_HEADS = 8
HEAD_DIM = 4


def _kv_cache(num_slots: int) -> torch.Tensor:
    """A stand-in with the real layout, filled so zeros are detectable."""
    return torch.ones(
        2,
        num_slots * BLOCKS_PER_SLOT,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
    )


def _zero_slot(kv: torch.Tensor, slot: int) -> None:
    """The fixed slicing from ``reset_slot``."""
    start = slot * BLOCKS_PER_SLOT
    kv[:, start : start + BLOCKS_PER_SLOT].zero_()


def _blocks_zeroed(kv: torch.Tensor) -> set[int]:
    """Indices along the block axis that are fully zero, for both K and V."""
    return {int(i) for i in (kv == 0).all(dim=0).flatten(1).all(dim=1).nonzero().flatten()}


@pytest.mark.parametrize("slot", [0, 1, 2, 3])
def test_reset_slot_clears_exactly_its_own_blocks(slot: int) -> None:
    kv = _kv_cache(num_slots=4)
    _zero_slot(kv, slot)
    expected = set(range(slot * BLOCKS_PER_SLOT, (slot + 1) * BLOCKS_PER_SLOT))
    assert _blocks_zeroed(kv) == expected


def test_reset_slot_is_not_a_no_op_for_slots_above_zero() -> None:
    """The exact failure mode of the dim-0 slice: nothing got cleared."""
    kv = _kv_cache(num_slots=4)
    _zero_slot(kv, 1)
    assert (kv == 0).any(), "reset_slot(1) cleared nothing -- stale KV would leak"


def test_reset_slot_zero_does_not_wipe_other_slots() -> None:
    """The other failure mode: slot 0 clamped to [0:2] and wiped everything."""
    kv = _kv_cache(num_slots=4)
    _zero_slot(kv, 0)
    survivors = set(range(BLOCKS_PER_SLOT, 4 * BLOCKS_PER_SLOT))
    assert _blocks_zeroed(kv).isdisjoint(survivors)


def test_single_slot_is_unchanged_by_the_fix() -> None:
    """Why this fix is safe to land mid-investigation: at num_slots == 1
    the old (dim-0) and new (dim-1) slicings clear the same elements, so
    every DFlash/benchmark run today is bit-identical either way."""
    old, new = _kv_cache(num_slots=1), _kv_cache(num_slots=1)
    old[0 : BLOCKS_PER_SLOT].zero_()  # the buggy form, clamped to [0:2]
    _zero_slot(new, 0)
    assert torch.equal(old, new)
