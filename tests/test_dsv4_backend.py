"""DSV4 backend tests: protocol conformance + eager single-slot serving.

Phase 3 serves one slot eagerly; the gate is protocol conformance (GPU-
free, signature-level) plus greedy/sampled token identity and slot
bookkeeping on a tiny zeroed graph (no weights needed: every buffer
zeroed, so logits are identically zero and greedy argmax is token 0,
deterministically).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.backends.dsv4 import DeepseekV4Backend  # noqa: E402
from runtime.backends.protocol import (  # noqa: E402
    BackendCapabilities,
    check_conformance,
)
from runtime.model.dsv4_config import Dsv4Config  # noqa: E402
from runtime.model.dsv4_model import Dsv4Transformer  # noqa: E402
from runtime.sampling import SamplingParams  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

TINY = Dsv4Config(
    vocab_size=128,
    hidden_size=256,
    num_layers=1,
    max_position_embeddings=256,
    norm_eps=1e-6,
    num_heads=2,
    head_dim=128,
    rope_head_dim=64,
    q_lora_rank=16,
    o_groups=2,
    o_lora_rank=8,
    window_size=8,
    compress_ratios=(4,),
    rope_theta=10000.0,
    rope_factor=16.0,
    rope_original_seq_len=64,
    beta_fast=32,
    beta_slow=1,
    compress_rope_theta=160000.0,
    index_n_heads=2,
    index_head_dim=64,
    index_topk=4,
    hc_mult=4,
    hc_sinkhorn_iters=4,
    hc_eps=1e-6,
    n_routed_experts=8,
    n_shared_experts=1,
    n_activated_experts=2,
    moe_intermediate_size=256,
    route_scale=1.5,
    swiglu_limit=10.0,
    n_hash_layers=0,
)


def _zeroed_model(max_seq_len: int = 64) -> Dsv4Transformer:
    model = Dsv4Transformer(TINY, max_seq_len=max_seq_len, device="cuda")
    for buf in model.buffers():
        buf.zero_()
    for module in model.modules():
        packed = getattr(module, "packed", None)
        if packed is not None:
            packed.zero_()
    return model


# -- protocol conformance (GPU-free) -----------------------------------------


def test_backend_conforms_with_no_capabilities() -> None:
    problems = check_conformance(
        DeepseekV4Backend,
        BackendCapabilities(False, False, False, False, False),
    )
    assert problems == [], problems


def test_backend_declares_no_capabilities() -> None:
    caps = DeepseekV4Backend.capabilities.fget(None)
    assert caps.speculative_decode is False
    assert caps.prefix_cache is False
    assert caps.cuda_graph is False
    assert caps.chunked_prefill is False
    assert caps.warm_continue is False


def test_capabilities_property_is_instance_level() -> None:
    assert isinstance(DeepseekV4Backend.capabilities, property)


# -- eager serving on a tiny zeroed graph ------------------------------------


def test_prefill_and_decode_advance_slot() -> None:
    backend = DeepseekV4Backend(_zeroed_model(), TINY, device="cuda")
    assert backend.slot_state(0).is_fresh

    first = backend.prefill(0, [1, 2, 3])
    assert first == 0  # zeroed weights -> uniform logits -> argmax 0
    state = backend.slot_state(0)
    assert state.kv_len == 3
    assert state.committed_tokens == (1, 2, 3)
    assert not state.is_fresh

    out = backend.decode_batch_sampled([0], [7], [3], [SamplingParams(temperature=0.0)])
    assert out == [0]
    assert backend.slot_state(0).kv_len == 4
    assert backend.slot_state(0).committed_tokens == (1, 2, 3, 0)


def test_sampled_decode_uses_params_seed() -> None:
    backend = DeepseekV4Backend(_zeroed_model(), TINY, device="cuda")
    backend.prefill(0, [5])
    params = SamplingParams(temperature=0.8, seed=42)
    a = backend.decode_batch_sampled([0], [9], [1], [params])
    backend.reset_slot(0)
    backend.prefill(0, [5])
    b = backend.decode_batch_sampled([0], [9], [1], [params])
    assert a == b  # same seed, same step, same weights -> same token


def test_reset_slot_clears_bookkeeping_and_state() -> None:
    backend = DeepseekV4Backend(_zeroed_model(), TINY, device="cuda")
    backend.prefill(0, [1, 2, 3])
    backend.decode_batch_sampled([0], [7], [3], [SamplingParams(temperature=0.0)])
    backend.reset_slot(0)
    state = backend.slot_state(0)
    assert state.kv_len == 0
    assert state.committed_tokens == ()
    assert state.is_fresh


def test_prefill_without_reset_raises() -> None:
    backend = DeepseekV4Backend(_zeroed_model(), TINY, device="cuda")
    backend.prefill(0, [1, 2])
    with pytest.raises(RuntimeError, match="reset_slot"):
        backend.prefill(0, [3, 4])


def test_decode_kv_length_mismatch_raises() -> None:
    backend = DeepseekV4Backend(_zeroed_model(), TINY, device="cuda")
    backend.prefill(0, [1, 2])
    with pytest.raises(RuntimeError, match="reset_slot"):
        backend.decode_batch_sampled([0], [7], [99], [SamplingParams(temperature=0.0)])


def test_snapshot_shapes() -> None:
    backend = DeepseekV4Backend(_zeroed_model(), TINY, device="cuda")
    snap = backend.snapshot()
    assert len(snap.slots) == 1
    assert snap.slots[0].slot == 0
    assert len(snap.prefix) == 1
    assert snap.dflash_cg_status == ()


def test_multislot_rejected() -> None:
    with pytest.raises(NotImplementedError, match="Phase 3"):
        DeepseekV4Backend(_zeroed_model(), TINY, num_slots=2, device="cuda")


def test_logprobs_rejected() -> None:
    backend = DeepseekV4Backend(_zeroed_model(), TINY, device="cuda")
    backend.prefill(0, [1])
    with pytest.raises(NotImplementedError, match="Phase 4"):
        backend.decode_batch_sampled(
            [0], [7], [1], [SamplingParams(temperature=0.0)], return_logprobs=True
        )
