"""Tests for ``bfdiag/checkpoint/state.py``: the declarative state
manifest, geometry resolution, block-range arithmetic, and the volume
estimate table. Nothing here touches the GPU -- ``state.py`` has zero
``runtime.*``/``torch`` dependency at all.
"""

from __future__ import annotations

import torch

from bfdiag.checkpoint import state
from bfdiag.checkpoint.testing import FakeBackend, FakeDFlashEngine

# --- declarative checklist ---------------------------------------------------

_KNOWN_CATEGORIES = {
    "host_scalar",
    "host_list",
    "device_tensor",
    "derived_no_store",
    "not_applicable",
    "bug_found_not_fixed",
}


def test_slot_state_items_nonempty_and_well_formed() -> None:
    items = state.SLOT_STATE_ITEMS
    assert len(items) >= 10
    names = [item.name for item in items]
    assert len(names) == len(set(names)), "duplicate StateItem names"
    for item in items:
        assert item.category in _KNOWN_CATEGORIES
        assert item.code_ref, f"{item.name} has no code_ref citation"
        assert item.note, f"{item.name} has no note"


def test_describe_state_items_is_json_safe() -> None:
    items = state.describe_state_items()
    assert isinstance(items, list)
    assert len(items) == len(state.SLOT_STATE_ITEMS)
    for d in items:
        assert set(d) == {"name", "category", "per_layer", "source", "code_ref", "note"}
        assert isinstance(d["name"], str)
        assert isinstance(d["per_layer"], bool)


def test_device_tensor_items_cover_full_swa_draft() -> None:
    tensor_items = [i for i in state.SLOT_STATE_ITEMS if i.category == "device_tensor"]
    names = {i.name for i in tensor_items}
    assert "full-attention KV cache blocks" in names
    assert "SWA ring KV cache blocks" in names
    assert "DFlash draft KV cache ring blocks" in names
    assert len(tensor_items) == 3


def test_not_applicable_items_match_reset_checklist_findings() -> None:
    """GDN state and the content-addressed prefix cache/BlockPool are named
    in the task spec by analogy with DirectModelRunner but do not apply to
    LagunaBackend -- same conclusion as
    bfdiag/daemon/session.py::RESET_CHECKLIST, re-derived independently
    here rather than merely trusted."""
    na_names = {i.name for i in state.SLOT_STATE_ITEMS if i.category == "not_applicable"}
    assert any("GDN" in name for name in na_names)
    assert any("prefix cache" in name or "BlockPool" in name for name in na_names)


def test_bug_found_item_is_documented() -> None:
    bug_items = [i for i in state.SLOT_STATE_ITEMS if i.category == "bug_found_not_fixed"]
    assert len(bug_items) == 1
    assert "reset_slot" in bug_items[0].code_ref


# --- ring_blocks_for_window matches the real formula -------------------------


def test_ring_blocks_for_window_matches_known_production_values() -> None:
    # runtime/backends/laguna.py's SWA_QO_MAX=16, swa_window=512 (production):
    # cdiv(512 - 1 + 16, 64) + 1 = cdiv(527, 64) + 1 = 9 + 1 = 10
    assert state.ring_blocks_for_window(512, 64, 16) == 10
    # cdiv(527, 128) + 1 = 5 + 1 = 6
    assert state.ring_blocks_for_window(512, 128, 16) == 6


# --- geometry resolution + block ranges, against the FakeBackend/Engine -----


def _fake_pair(**kwargs):
    backend = FakeBackend(**kwargs)
    engine = FakeDFlashEngine(backend, draft_window=40, num_draft_layers=2)
    return backend, engine


def test_slot_geometry_duck_types_correctly() -> None:
    backend, engine = _fake_pair(num_slots=3, block_size=16, blocks_per_slot=8, swa_window=40)
    geom = state.slot_geometry(backend, engine, slot=1)
    assert geom.slot == 1
    assert geom.physical_slot == 1  # RESERVED_PHYSICAL_SLOTS == 0
    assert geom.reserved_physical_slots == 0
    assert geom.num_slots == 3
    assert geom.block_size == 16
    assert geom.blocks_per_slot == 8
    assert geom.ring_blocks_per_slot == backend._ring_blocks_per_slot
    assert geom.draft_blocks_per_slot == engine._draft_blocks_per_slot
    assert geom.swa_window == 40
    assert geom.full_layer_names == tuple(backend._full_layer_names)
    assert geom.swa_layer_names == tuple(backend._swa_layer_names)
    assert geom.draft_layer_names == tuple(engine._draft_layer_names)


def test_full_block_range_is_ceil_of_kv_len_not_full_capacity() -> None:
    backend, engine = _fake_pair(num_slots=2, block_size=16, blocks_per_slot=8, swa_window=40)
    geom = state.slot_geometry(backend, engine, slot=1)
    # slot 1 -> physical_slot 1 -> base offset = 1 * blocks_per_slot = 8
    start, end = state.full_block_range(geom, kv_len=0)
    assert (start, end) == (8, 8)
    start, end = state.full_block_range(geom, kv_len=1)
    assert (start, end) == (8, 9)  # ceil(1/16) = 1 block
    start, end = state.full_block_range(geom, kv_len=16)
    assert (start, end) == (8, 9)  # exactly one block
    start, end = state.full_block_range(geom, kv_len=17)
    assert (start, end) == (8, 10)  # spills into a second block
    # Never exceeds the static blocks_per_slot allocation even if kv_len claims to.
    start, end = state.full_block_range(geom, kv_len=10_000)
    assert end - start == geom.blocks_per_slot


def test_ring_ranges_are_always_full_capacity_regardless_of_kv_len() -> None:
    backend, engine = _fake_pair(num_slots=2, block_size=16, blocks_per_slot=8, swa_window=40)
    geom = state.slot_geometry(backend, engine, slot=0)
    swa_start, swa_end = state.swa_ring_block_range(geom)
    assert swa_end - swa_start == geom.ring_blocks_per_slot
    draft_start, draft_end = state.draft_ring_block_range(geom)
    assert draft_end - draft_start == geom.draft_blocks_per_slot
    # Physical-slot offset scales with the ring's own per-slot block count,
    # not with blocks_per_slot (full-attention's own static capacity).
    geom1 = state.slot_geometry(backend, engine, slot=1)
    swa_start1, _ = state.swa_ring_block_range(geom1)
    assert swa_start1 == geom.ring_blocks_per_slot  # slot 1 starts right after slot 0's ring


# --- volume estimate table ---------------------------------------------------


def test_bytes_per_token_per_layer_matches_task_brief() -> None:
    # Task brief: "全局层每 token 24 KiB (12 层 x 2(K+V) x 8 头 x 128 dim x 1B, FP8)"
    assert state.bytes_per_token_per_layer() == 2 * 8 * 128 * 1  # 2048 B = 2 KiB
    assert state.NUM_FULL_LAYERS * state.bytes_per_token_per_layer() == 24 * 1024


def test_estimate_checkpoint_bytes_64k_matches_task_brief() -> None:
    # Task brief: "64K -> 1.5 GiB"
    est = state.estimate_checkpoint_bytes(65536, block_size=64)
    assert est.full_attn_bytes == 65536 * 24 * 1024
    assert abs(est.full_attn_bytes / (1024**3) - 1.5) < 1e-9

    est128 = state.estimate_checkpoint_bytes(65536, block_size=128)
    assert est128.full_attn_bytes == est.full_attn_bytes  # full-attn size is block_size-agnostic


def test_estimate_checkpoint_bytes_swa_ring_is_fixed_per_slot() -> None:
    """SWA ring size does not grow with context length -- only with
    block_size (via the ring's block-rounding)."""
    small = state.estimate_checkpoint_bytes(4096, block_size=64)
    large = state.estimate_checkpoint_bytes(262144, block_size=64)
    assert small.swa_ring_bytes == large.swa_ring_bytes
    assert small.draft_ring_bytes == large.draft_ring_bytes
    assert small.full_attn_bytes < large.full_attn_bytes


# --- regression test for the reset_slot axis bug -----------------------------


def test_reset_slot_axis_bug_is_real_and_this_package_does_not_replicate_it() -> None:
    """Documents (and proves, via plain tensor slicing -- no runtime.*
    import needed) the bug recorded in SLOT_STATE_ITEMS'
    'bug_found_not_fixed' entry: ``runtime/backends/laguna.py:1647,1653``
    slices ``tensor[start:end]`` on a KV cache tensor shaped
    ``(2, num_blocks, block_size, kv_heads, head_dim)`` -- dim 0 is the K/V
    axis (size 2), so that slice hits the WRONG axis. For slot 0 (start=0)
    it clips to the whole dim 0 and leaves every block untouched by the
    slice itself, i.e. ``.zero_()`` wipes the ENTIRE tensor (every slot's
    blocks); for any slot > 0 the slice is empty and ``.zero_()`` is a
    silent no-op.

    This package's own code (state.py's ``full_block_range`` used with a
    leading ``:,`` in store.py/restore.py/testing.py) does NOT have this
    bug -- demonstrated here by comparing the two slicing forms directly.
    """
    num_slots = 2
    blocks_per_slot = 4
    tensor = torch.arange(2 * num_slots * blocks_per_slot, dtype=torch.uint8).reshape(
        2, num_slots * blocks_per_slot, 1, 1, 1
    )

    # Slot 0's "intended" range: blocks [0, blocks_per_slot).
    full_start, full_end = 0 * blocks_per_slot, 1 * blocks_per_slot

    buggy = tensor.clone()
    buggy[full_start:full_end].zero_()  # the real reset_slot's exact form
    correct = tensor.clone()
    correct[:, full_start:full_end].zero_()  # this package's form

    # The buggy slice zeros BOTH K and V for slot 0's range only if that
    # happened to equal the whole tensor; in general (num_slots > 1) it
    # zeros strictly more than intended (all of dim 0, every block) --
    # provably different from the correct, block-scoped zeroing.
    assert not torch.equal(buggy, correct), (
        "the buggy and correct slicing forms produced identical results -- "
        "this test's tensor shape no longer demonstrates the bug; revisit "
        "the shape parameters"
    )
    # Correct form only touches slot 0's blocks; slot 1's blocks (and
    # anything beyond blocks_per_slot in dim 1) must be untouched.
    assert torch.equal(correct[:, blocks_per_slot:], tensor[:, blocks_per_slot:])
    # Buggy form, by contrast, zeros dim-1 positions belonging to OTHER
    # slots too (since it left dim 1 entirely unrestricted).
    assert not torch.equal(buggy[:, blocks_per_slot:], tensor[:, blocks_per_slot:])

    # And for slot 1 (start=blocks_per_slot >= dim-0 size 2), the buggy
    # slice is empty -- .zero_() is a silent no-op, unlike the correct form.
    full_start1, full_end1 = 1 * blocks_per_slot, 2 * blocks_per_slot
    buggy1 = tensor.clone()
    buggy1[full_start1:full_end1].zero_()
    assert torch.equal(buggy1, tensor), "expected the buggy slice for slot>0 to be a no-op"
    correct1 = tensor.clone()
    correct1[:, full_start1:full_end1].zero_()
    assert not torch.equal(correct1, tensor), "the correct slice must actually zero slot 1's blocks"
