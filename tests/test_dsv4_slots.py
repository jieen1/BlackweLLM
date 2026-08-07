"""Slot-pool tests: geometry, static view slicing, reset semantics."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.model.dsv4_config import Dsv4Config  # noqa: E402
from runtime.model.dsv4_slots import Dsv4SlotPool  # noqa: E402


def pool_config() -> Dsv4Config:
    return Dsv4Config(
        vocab_size=32,
        hidden_size=256,
        num_layers=5,
        compress_ratios=(0, 4, 128, 4, 128),
        index_head_dim=128,
    )


def test_layer_class_partition() -> None:
    pool = Dsv4SlotPool(pool_config(), num_slots=2, max_seq_len=512)
    assert pool.csa_layer_ids == (1, 3)
    assert pool.hca_layer_ids == (2, 4)
    assert pool.csa_entries == 128
    assert pool.hca_entries == 4


def test_view_shapes_and_storage() -> None:
    pool = Dsv4SlotPool(pool_config(), num_slots=3, max_seq_len=512)
    w = pool.slot_window(1)
    assert w.shape == (5, 128, 512)
    assert pool.slot_csa_comp(2).shape == (2, 128, 512)
    assert pool.slot_hca_comp(0).shape == (2, 4, 512)
    assert pool.slot_idx_k(1).shape == (2, 128, 128)
    # views address the pool's storage at the slot stride (no new storage)
    assert (
        pool.slot_window(1).data_ptr() == pool.window_pool.data_ptr() + w.element_size() * w.numel()
    )
    kv, score = pool.slot_csa_state(0)
    assert kv.shape == (2, 8, 1024) and score.shape == (2, 8, 1024)
    kv_h, score_h = pool.slot_hca_state(0)
    assert kv_h.shape == (2, 128, 512)
    kv_i, score_i = pool.slot_idx_state(0)
    assert kv_i.shape == (2, 8, 256)


def test_reset_zeroes_recursive_state_only() -> None:
    pool = Dsv4SlotPool(pool_config(), num_slots=2, max_seq_len=512)
    pool.csa_kv_state[1].fill_(3.0)
    pool.hca_score_state[1].fill_(0.5)
    pool.idx_kv_state[1].fill_(-2.0)
    pool.slot_window(1).fill_(1.0)  # KV regions must survive reset
    pool.csa_kv_state[0].fill_(7.0)  # marker: other slots must stay untouched
    pool.reset_slot(1)
    assert torch.equal(pool.csa_kv_state[1], torch.zeros_like(pool.csa_kv_state[1]))
    assert torch.isinf(pool.hca_score_state[1]).all() and (pool.hca_score_state[1] < 0).all()
    assert torch.equal(pool.idx_kv_state[1], torch.zeros_like(pool.idx_kv_state[1]))
    assert (pool.slot_window(1) == 1.0).all()
    # other slots untouched by slot 1's reset
    assert (pool.csa_kv_state[0] == 7.0).all()


def test_geometry_math() -> None:
    pool = Dsv4SlotPool(pool_config(), num_slots=2, max_seq_len=512)
    geo = pool.geometry()
    entry = 512 * 2
    expected_per_slot = (
        5 * 128 * entry  # window
        + 2 * 128 * entry  # csa compressed
        + 2 * 4 * entry  # hca compressed
        + 2 * 128 * 128 * 2  # indexer K
        + 2 * 8 * 1024 * 4 * 2  # csa state
        + 2 * 128 * 512 * 4 * 2  # hca state
        + 2 * 8 * 256 * 4 * 2  # indexer state
    )
    assert geo.bytes_per_slot == expected_per_slot
    assert geo.total_bytes == expected_per_slot * 2
    actual = sum(
        t.numel() * t.element_size()
        for t in (
            pool.window_pool,
            pool.csa_comp_pool,
            pool.hca_comp_pool,
            pool.idx_k_pool,
            pool.csa_kv_state,
            pool.csa_score_state,
            pool.hca_kv_state,
            pool.hca_score_state,
            pool.idx_kv_state,
            pool.idx_score_state,
        )
    )
    assert actual == geo.total_bytes


def test_rejects_unknown_layout() -> None:
    with pytest.raises(ValueError, match="unsupported slot-pool layout"):
        Dsv4SlotPool(pool_config(), num_slots=1, max_seq_len=512, layout="fp8")
