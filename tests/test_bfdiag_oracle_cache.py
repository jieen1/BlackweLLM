"""Unit tests for bfdiag.divergence.cache: the tiered oracle activation cache.

The reduction logic (stats, quantiles, sample-position selection, cache-key
hashing) is pure Python and tested unconditionally. The actual disk I/O
(``write_oracle_cache``/``read_oracle_cache``) needs numpy + safetensors, so
those tests use ``pytest.importorskip`` the same way
tests/test_capture_hooks.py already does for torch/safetensors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bfdiag.divergence.cache import (
    CacheKey,
    CaptureConfig,
    build_cache_entries,
    compute_prompt_hash,
    compute_stats,
    select_sample_positions,
    to_activation_trace,
)


def test_select_sample_positions_covers_edges_and_middle() -> None:
    positions = select_sample_positions(100, k_edge=8, r_random=8, seed=0)
    assert positions == tuple(sorted(set(positions)))
    assert set(range(8)) <= set(positions)
    assert set(range(92, 100)) <= set(positions)
    assert len(positions) == 24  # 8 head + 8 tail + 8 random, no overlap at this size


def test_select_sample_positions_deterministic_for_a_fixed_seed() -> None:
    first = select_sample_positions(1000, k_edge=4, r_random=6, seed=7)
    second = select_sample_positions(1000, k_edge=4, r_random=6, seed=7)
    assert first == second


def test_select_sample_positions_handles_short_sequences_without_duplicates() -> None:
    positions = select_sample_positions(10, k_edge=8, r_random=8, seed=0)
    assert positions == tuple(sorted(set(positions)))
    assert all(0 <= p < 10 for p in positions)
    assert len(positions) <= 10


def test_select_sample_positions_empty_sequence() -> None:
    assert select_sample_positions(0, k_edge=8, r_random=8) == ()


def test_compute_stats_matches_hand_computed_values() -> None:
    matrix = [[1.0, 3.0], [2.0, 4.0]]
    stats, dim_mean = compute_stats(matrix)
    assert stats.num_tokens == 2
    assert stats.dim == 2
    assert stats.mean == pytest.approx(2.5)
    assert stats.min == pytest.approx(1.0)
    assert stats.max == pytest.approx(4.0)
    assert stats.absmax == pytest.approx(4.0)
    assert stats.l2 == pytest.approx((1 + 4 + 9 + 16) ** 0.5)
    assert dim_mean == pytest.approx((1.5, 3.5))
    assert len(stats.quantiles) == 9


def test_compute_stats_rejects_empty_matrix() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_stats([])


def test_build_cache_entries_and_to_activation_trace_round_trip_small_activation() -> None:
    trace = {0: {"self_attn": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}}
    config = CaptureConfig(k_edge_tokens=1, r_random_tokens=0)
    entries = build_cache_entries(config, trace)

    entry = entries[0]["self_attn"]
    assert entry.stats.num_tokens == 3
    assert entry.sample_positions == (0, 2)  # first 1 + last 1, dedup at n=3
    assert entry.sample_tokens == ((1.0, 2.0), (5.0, 6.0))
    assert entry.full is None

    scan_trace = to_activation_trace(entries)
    assert scan_trace[0]["self_attn"] == [1.0, 2.0, 5.0, 6.0]


def test_build_cache_entries_full_tier_stores_every_token() -> None:
    trace = {0: {"self_attn": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}}
    config = CaptureConfig(full=True)
    entries = build_cache_entries(config, trace)
    entry = entries[0]["self_attn"]
    assert entry.full == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))

    scan_trace = to_activation_trace(entries)
    assert scan_trace[0]["self_attn"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_to_activation_trace_falls_back_to_dim_mean_with_no_sampling() -> None:
    trace = {0: {"self_attn": [[1.0, 2.0], [3.0, 4.0]]}}
    config = CaptureConfig(k_edge_tokens=0, r_random_tokens=0)
    entries = build_cache_entries(config, trace)
    entry = entries[0]["self_attn"]
    assert entry.sample_tokens == ()
    scan_trace = to_activation_trace(entries)
    assert scan_trace[0]["self_attn"] == [2.0, 3.0]  # dim_mean of [[1,2],[3,4]]


def test_compute_prompt_hash_is_stable_and_order_sensitive() -> None:
    assert compute_prompt_hash([1, 2, 3]) == compute_prompt_hash([1, 2, 3])
    assert compute_prompt_hash([1, 2, 3]) != compute_prompt_hash([3, 2, 1])


def test_cache_key_hash_changes_with_capture_config() -> None:
    base = CacheKey("rev", "hash123", "all", CaptureConfig())
    changed = CacheKey("rev", "hash123", "all", CaptureConfig(k_edge_tokens=16))
    assert base.config_hash() != changed.config_hash()


def test_cache_key_hash_changes_with_model_revision() -> None:
    a = CacheKey("rev-a", "hash123", "all", CaptureConfig())
    b = CacheKey("rev-b", "hash123", "all", CaptureConfig())
    assert a.config_hash() != b.config_hash()


def test_write_then_read_oracle_cache_round_trips(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("safetensors.numpy")
    from bfdiag.divergence.cache import read_oracle_cache, write_oracle_cache

    trace = {
        0: {"self_attn": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]},
        1: {"mlp": [[0.1, 0.2], [0.3, 0.4]]},
    }
    key = CacheKey(
        model_revision="rev-1",
        prompt_hash=compute_prompt_hash([10, 20, 30]),
        layer_set="all",
        capture_config=CaptureConfig(k_edge_tokens=1, r_random_tokens=0),
    )

    assert read_oracle_cache(key, root=tmp_path) is None  # clean miss before writing

    write_result = write_oracle_cache(key, trace, root=tmp_path)
    assert not write_result.hit
    assert write_result.path.exists()

    read_result = read_oracle_cache(key, root=tmp_path)
    assert read_result is not None
    entries, lookup = read_result
    assert lookup.hit

    original_entries = build_cache_entries(key.capture_config, trace)
    for layer_idx, submodules in original_entries.items():
        for name, original in submodules.items():
            restored = entries[layer_idx][name]
            assert restored.stats.mean == pytest.approx(original.stats.mean, abs=1e-5)
            assert restored.stats.absmax == pytest.approx(original.stats.absmax, abs=1e-5)
            assert restored.sample_positions == original.sample_positions
            for restored_row, original_row in zip(
                restored.sample_tokens, original.sample_tokens, strict=True
            ):
                assert restored_row == pytest.approx(original_row, abs=1e-5)


def test_read_oracle_cache_misses_when_capture_config_differs(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("safetensors.numpy")
    from bfdiag.divergence.cache import read_oracle_cache, write_oracle_cache

    trace = {0: {"self_attn": [[1.0, 2.0], [3.0, 4.0]]}}
    prompt_hash = compute_prompt_hash([1, 2, 3])
    written_key = CacheKey("rev-1", prompt_hash, "all", CaptureConfig(k_edge_tokens=1))
    write_oracle_cache(written_key, trace, root=tmp_path)

    different_config_key = CacheKey("rev-1", prompt_hash, "all", CaptureConfig(k_edge_tokens=2))
    assert read_oracle_cache(different_config_key, root=tmp_path) is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
