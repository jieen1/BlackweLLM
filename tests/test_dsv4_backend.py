"""DSV4 backend tests: protocol conformance + multi-slot serving.

The serving contract is exercised with a stubbed ``forward_fn`` on a tiny
zeroed graph (no weights needed: every buffer zeroed, so logits are
identically zero and greedy argmax is token 0, deterministically).  The
kernel-path forward itself is exercised separately on the real model
(scripts/dsv4_align_eager_vs_kernel.py); this file pins the backend's
server-facing surface: protocol conformance, slot bookkeeping, prefix/
chunked no-op surfaces, and the engine call sequence.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.backends.dsv4 import DeepseekV4Backend  # noqa: E402
from runtime.backends.protocol import (  # noqa: E402
    BackendCapabilities,
    PrefixHit,
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


def _make_backend(num_slots: int = 2) -> DeepseekV4Backend:
    """Backend with a stubbed forward: zero logits -> greedy argmax 0.

    The stub records (slot, start_pos, seq_len) per call so tests can
    assert the serving contract's call sequence without weights.
    """
    calls: list[tuple[int, int, int]] = []

    def forward_fn(slot: int, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        calls.append((slot, start_pos, input_ids.shape[1]))
        vocab = TINY.vocab_size
        logits = torch.zeros(
            1, input_ids.shape[1], vocab, dtype=torch.float32, device=input_ids.device
        )
        # Force a deterministic non-zero token so decode identity is
        # distinguishable from prefill's argmax-0: place 2.0 on token 3.
        logits[0, -1, 3] = 2.0
        return logits

    backend = DeepseekV4Backend(
        _zeroed_model(),
        TINY,
        num_slots=num_slots,
        max_seq_len=64,
        device="cuda",
        forward_fn=forward_fn,
    )
    backend._test_calls = calls  # noqa: SLF001 -- test hook
    return backend


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


# -- multi-slot serving on a tiny zeroed graph -------------------------------


def test_prefill_and_decode_advance_slot() -> None:
    backend = _make_backend()
    assert backend.slot_state(0).is_fresh

    first = backend.prefill(0, [1, 2, 3])
    assert first == 3  # stub places 2.0 on token 3
    state = backend.slot_state(0)
    assert state.kv_len == 3
    assert state.committed_tokens == (1, 2, 3)
    assert not state.is_fresh

    out = backend.decode_batch_sampled([0], [7], [3], [SamplingParams(temperature=0.0)])
    assert out == [3]
    assert backend.slot_state(0).kv_len == 4
    assert backend.slot_state(0).committed_tokens == (1, 2, 3, 3)
    # prefill called at start_pos 0 with the full prompt; decode at kv_len.
    assert backend._test_calls == [(0, 0, 3), (0, 3, 1)]


def test_sampled_decode_uses_params_seed() -> None:
    backend = _make_backend()
    backend.prefill(0, [5])
    params = SamplingParams(temperature=0.8, seed=42)
    a = backend.decode_batch_sampled([0], [9], [1], [params])
    backend.reset_slot(0)
    backend.prefill(0, [5])
    b = backend.decode_batch_sampled([0], [9], [1], [params])
    assert a == b  # same seed, same step, same weights -> same token


def test_reset_slot_clears_bookkeeping_and_state() -> None:
    backend = _make_backend()
    backend.prefill(0, [1, 2, 3])
    backend.decode_batch_sampled([0], [7], [3], [SamplingParams(temperature=0.0)])
    backend.reset_slot(0)
    state = backend.slot_state(0)
    assert state.kv_len == 0
    assert state.committed_tokens == ()
    assert state.is_fresh


def test_prefill_without_reset_raises() -> None:
    backend = _make_backend()
    backend.prefill(0, [1, 2])
    with pytest.raises(RuntimeError, match="reset_slot"):
        backend.prefill(0, [3, 4])


def test_decode_kv_length_mismatch_raises() -> None:
    backend = _make_backend()
    backend.prefill(0, [1, 2])
    with pytest.raises(RuntimeError, match="reset_slot"):
        backend.decode_batch_sampled([0], [7], [99], [SamplingParams(temperature=0.0)])


def test_snapshot_shapes() -> None:
    backend = _make_backend(num_slots=2)
    snap = backend.snapshot()
    assert len(snap.slots) == 2
    assert [s.slot for s in snap.slots] == [0, 1]
    assert len(snap.prefix) == 2
    assert snap.dflash_cg_status == ()


def test_multislot_serving() -> None:
    backend = _make_backend(num_slots=2)
    assert backend.prefill(0, [1, 2]) == 3
    assert backend.prefill(1, [4, 5]) == 3
    out = backend.decode_batch_sampled(
        [0, 1],
        [7, 8],
        [2, 2],
        [SamplingParams(temperature=0.0), SamplingParams(temperature=0.0)],
    )
    assert out == [3, 3]
    assert backend.slot_state(0).kv_len == 3
    assert backend.slot_state(1).kv_len == 3


def test_out_of_range_slot_raises() -> None:
    backend = _make_backend(num_slots=1)
    with pytest.raises(IndexError, match="out of range"):
        backend.prefill(1, [1])
    with pytest.raises(IndexError, match="out of range"):
        backend.slot_state(1)


def test_logprobs_rejected() -> None:
    backend = _make_backend()
    backend.prefill(0, [1])
    with pytest.raises(NotImplementedError, match="logprobs"):
        backend.decode_batch_sampled(
            [0], [7], [1], [SamplingParams(temperature=0.0)], return_logprobs=True
        )


# -- prefix-cache no-op surface ----------------------------------------------


def test_reconcile_prefix_hit_is_zero() -> None:
    backend = _make_backend()
    hit = backend.reconcile_prefix_hit([1, 2, 3])
    assert isinstance(hit, PrefixHit)
    assert hit.kv_hit == 0 and hit.state_hit == 0 and hit.effective == 0


def test_find_best_slot_picks_first_free() -> None:
    backend = _make_backend(num_slots=2)
    slot, hit = backend.find_best_slot_for_prompt([1, 2], [1, 0])
    assert slot == 1 and hit == 0


def test_no_speculative_decode() -> None:
    backend = _make_backend()
    assert backend.has_speculative_decode is False


# -- chunked prefill surface (one-shot) --------------------------------------


def test_prefill_chunked_begin_completes_immediately() -> None:
    backend = _make_backend(num_slots=2)
    state = backend.prefill_chunked_begin([0, 1], [[1, 2, 3], [4, 5]])
    assert state.done is True
    assert state.result[0]["anchor"] == 3
    assert state.result[0]["draft_tokens"] == []
    assert state.result[1]["anchor"] == 3
    assert backend.slot_state(0).kv_len == 3
    assert backend.slot_state(1).kv_len == 2


def test_prefill_chunked_begin_sampled_anchor() -> None:
    backend = _make_backend()
    params = SamplingParams(temperature=0.8, seed=7)
    state = backend.prefill_chunked_begin(
        [0], [[1, 2]], params_per_slot={0: params}
    )
    assert state.done is True
    # The sampled anchor comes from the stub logits (uniform + 2.0 on 3);
    # sampling with seed is deterministic -- just assert it is a valid token.
    assert 0 <= state.result[0]["anchor"] < TINY.vocab_size


def test_prefill_chunked_step_returns_done() -> None:
    backend = _make_backend()
    state = backend.prefill_chunked_begin([0], [[1]])
    assert backend.prefill_chunked_step(state) is True
