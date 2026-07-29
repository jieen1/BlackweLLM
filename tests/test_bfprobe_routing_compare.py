"""Tests for bfprobe/routing_compare.py -- the pure-function routing
comparator.

Fixtures are synthetic 47-layer x 16-token x top-10 grids, matching
Laguna-S-2.1's real MoE layer count / DFlash verify-round token count /
top_k (see notes/2026-07-27-probe-system-design-and-plan.md section 4).
Everything here runs on CPU with plain numpy -- no GPU access is needed to
validate the comparator's logic, and none of these tests import torch.
"""

from __future__ import annotations

# Numpy is an optional probe dependency in the CPU-only CI environment.
# ruff: noqa: E402, I001

import pytest

np = pytest.importorskip("numpy")

from bfprobe.routing_compare import LayerTokenCoord, compare_routing

NUM_LAYERS = 47
NUM_TOKENS = 16
TOP_K = 10
NUM_EXPERTS = 256


def _synthetic_ids(seed: int = 0) -> np.ndarray:
    """A deterministic (layers, tokens, top_k) grid of distinct expert ids
    per (layer, token)."""
    rng = np.random.default_rng(seed)
    ids = np.empty((NUM_LAYERS, NUM_TOKENS, TOP_K), dtype=np.int32)
    for layer in range(NUM_LAYERS):
        for token in range(NUM_TOKENS):
            ids[layer, token] = rng.choice(NUM_EXPERTS, size=TOP_K, replace=False)
    return ids


# ---------------------------------------------------------------------------
# The three core acceptance tests from the task spec.
# ---------------------------------------------------------------------------


def test_case_a_identical_routing_is_reported_as_fully_consistent():
    ids = _synthetic_ids()
    result = compare_routing(ids, ids.copy())

    assert result.first_divergence is None
    assert result.top1_match_rate == 1.0
    assert result.set_match_rate == 1.0
    assert result.sequence_match_rate == 1.0
    assert result.mean_jaccard == 1.0
    assert result.verdict == "路由完全一致,分歧只可能来自数值"


def test_case_b_top1_substitution_is_localized_to_the_exact_coordinate():
    ids_a = _synthetic_ids()
    ids_b = ids_a.copy()

    target_layer, target_token = 23, 7
    original = ids_b[target_layer, target_token].copy()
    # Replace the top-1 (position 0) id with one guaranteed not to already
    # be in this token's top-10 set, so the *set* changes -- an
    # unambiguous routing divergence, not merely a reorder.
    replacement = next(e for e in range(NUM_EXPERTS) if e not in original.tolist())
    ids_b[target_layer, target_token, 0] = replacement

    result = compare_routing(ids_a, ids_b)

    assert result.first_divergence == LayerTokenCoord(layer=target_layer, token=target_token)
    assert not result.set_match[target_layer, target_token]
    assert not result.top1_match[target_layer, target_token]
    # Every other (layer, token) still agrees -- the divergence is localized.
    assert int(result.set_match.sum()) == NUM_LAYERS * NUM_TOKENS - 1
    assert "layer 23 token 7" in result.verdict


def test_case_c_reordering_without_a_set_change_is_not_a_false_divergence():
    ids_a = _synthetic_ids()
    ids_b = ids_a.copy()

    target_layer, target_token = 10, 3
    # Reverse this token's top-10: same set, different order. Must NOT be
    # reported as a routing divergence -- only as an order/sequence
    # difference, which is exactly the ambiguity a differing kernel-native
    # top-k order convention between two engines could otherwise cause a
    # naive positional diff to misreport.
    ids_b[target_layer, target_token] = ids_b[target_layer, target_token][::-1]

    result = compare_routing(ids_a, ids_b)

    assert result.first_divergence is None
    assert result.verdict == "路由完全一致,分歧只可能来自数值"
    # But the comparator still exposes the order-only difference for anyone
    # who wants to look:
    assert result.set_match[target_layer, target_token]
    assert result.jaccard[target_layer, target_token] == 1.0
    assert not result.sequence_match[target_layer, target_token]
    assert result.sequence_match_rate < 1.0


# ---------------------------------------------------------------------------
# Supporting behavior.
# ---------------------------------------------------------------------------


def test_shape_mismatch_raises():
    ids_a = _synthetic_ids()
    ids_b = ids_a[:, :, :-1]
    with pytest.raises(ValueError):
        compare_routing(ids_a, ids_b)


def test_non_3d_input_raises():
    flat = np.zeros((NUM_LAYERS, NUM_TOKENS))
    with pytest.raises(ValueError):
        compare_routing(flat, flat)


def test_partial_overlap_jaccard():
    ids_a = np.array([[[0, 1, 2, 3]]], dtype=np.int32)  # (layers=1, tokens=1, top_k=4)
    ids_b = np.array([[[0, 1, 4, 5]]], dtype=np.int32)
    result = compare_routing(ids_a, ids_b)
    # intersection {0, 1} has size 2; union {0, 1, 2, 3, 4, 5} has size 6.
    assert result.jaccard[0, 0] == pytest.approx(2 / 6)
    assert not result.set_match[0, 0]
    assert result.first_divergence == LayerTokenCoord(layer=0, token=0)


def test_weight_comparison_cosine_and_max_abs_diff():
    ids = _synthetic_ids()
    weights_a = np.ones((NUM_LAYERS, NUM_TOKENS, TOP_K), dtype=np.float32)
    weights_b = weights_a * 2.0
    result = compare_routing(ids, ids.copy(), weights_a, weights_b)
    assert result.weight_cosine == pytest.approx(1.0)
    assert result.weight_max_abs_diff == pytest.approx(1.0)


def test_weight_shape_mismatch_raises():
    ids = _synthetic_ids()
    with pytest.raises(ValueError):
        compare_routing(ids, ids.copy(), np.zeros(3), np.zeros(4))


def test_weights_must_both_be_provided_or_both_omitted():
    ids = _synthetic_ids()
    with pytest.raises(ValueError):
        compare_routing(ids, ids.copy(), weights_a=np.zeros(3))
