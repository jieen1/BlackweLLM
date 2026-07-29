"""Tests for bfprobe/routing.py: the MoE routing probe integration point.

Uses the local ``bfprobe._bus_stub`` -- the same stand-in ``routing.py``
falls back to via ``try/except ImportError`` when the real ``bfprobe.bus``
(owned by the P1 agent) doesn't exist yet. No GPU access; all tensors here
are plain numpy arrays standing in for what would be CPU/CUDA torch tensors
in production -- ``capture_routing`` never inspects its arguments beyond
passing them through, so numpy is a faithful substitute for this module's
own tests.
"""

from __future__ import annotations

# Numpy is an optional probe dependency in the CPU-only CI environment.
# ruff: noqa: E402, I001

import pytest

np = pytest.importorskip("numpy")

from bfprobe import _bus_stub as stub
from bfprobe import routing


@pytest.fixture(autouse=True)
def _reset_probe_state():
    stub.reset()
    routing.PROBE_ENABLED = False
    stub.PROBE_ENABLED = False
    yield
    stub.reset()
    routing.PROBE_ENABLED = False
    stub.PROBE_ENABLED = False


def _synthetic_layer(num_tokens: int = 16, num_experts: int = 256, top_k: int = 10):
    rng = np.random.default_rng(0)
    router_logits = rng.standard_normal((num_tokens, num_experts)).astype(np.float32)
    topk_ids = rng.integers(0, num_experts, size=(num_tokens, top_k)).astype(np.int32)
    topk_weights = rng.random((num_tokens, top_k)).astype(np.float32)
    return router_logits, topk_ids, topk_weights


def test_site_ids_are_stable_and_in_assigned_band():
    # bfprobe's routing probe owns site ids 300-399; router_logits/topk_ids/
    # topk_weights are specifically 300/301/302 per the design doc.
    assert routing.SITE_ROUTER_LOGITS == 300
    assert routing.SITE_TOPK_IDS == 301
    assert routing.SITE_TOPK_WEIGHTS == 302


def test_disabled_probe_emits_nothing():
    routing.PROBE_ENABLED = False
    router_logits, topk_ids, topk_weights = _synthetic_layer()
    routing.capture_routing(router_logits, topk_ids, topk_weights)
    assert stub.recorded_tensors == []


def test_disabled_probe_short_circuits_before_touching_arguments():
    routing.PROBE_ENABLED = False
    # If the guard didn't short-circuit before the emit_tensor calls, this
    # would still succeed against the stub (which only appends to a list),
    # but it demonstrates the call returns immediately regardless of what
    # garbage is passed in -- matching the "zero overhead when disabled"
    # contract.
    routing.capture_routing(None, None, None)
    assert stub.recorded_tensors == []


def test_enabled_probe_emits_all_three_tensors_in_order():
    routing.PROBE_ENABLED = True
    stub.PROBE_ENABLED = True
    router_logits, topk_ids, topk_weights = _synthetic_layer()
    routing.capture_routing(router_logits, topk_ids, topk_weights)

    assert [site_id for site_id, _ in stub.recorded_tensors] == [
        routing.SITE_ROUTER_LOGITS,
        routing.SITE_TOPK_IDS,
        routing.SITE_TOPK_WEIGHTS,
    ]
    (_, emitted_logits), (_, emitted_ids), (_, emitted_weights) = stub.recorded_tensors
    assert emitted_logits is router_logits
    assert emitted_ids is topk_ids
    assert emitted_weights is topk_weights


def test_enabled_probe_skips_prefill_sized_routing_tensors():
    routing.PROBE_ENABLED = True
    stub.PROBE_ENABLED = True
    router_logits, topk_ids, topk_weights = _synthetic_layer(num_tokens=17)
    routing.capture_routing(router_logits, topk_ids, topk_weights)
    assert stub.recorded_tensors == []


def test_multiple_layers_preserve_call_order_for_offline_reconstruction():
    routing.PROBE_ENABLED = True
    stub.PROBE_ENABLED = True
    for _ in range(3):
        router_logits, topk_ids, topk_weights = _synthetic_layer()
        routing.capture_routing(router_logits, topk_ids, topk_weights)

    # 3 layers x 3 tensors/layer, in fixed (router_logits, topk_ids,
    # topk_weights) x ascending-layer order -- offline consumers rely on
    # this fixed call order (no explicit layer index is emitted) to
    # reconstruct the (round, layer, token) grid.
    site_sequence = [site_id for site_id, _ in stub.recorded_tensors]
    assert (
        site_sequence
        == [
            routing.SITE_ROUTER_LOGITS,
            routing.SITE_TOPK_IDS,
            routing.SITE_TOPK_WEIGHTS,
        ]
        * 3
    )
