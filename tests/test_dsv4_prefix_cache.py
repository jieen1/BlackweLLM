from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.backends.dsv4 import (  # noqa: E402
    DSV4_PREFIX_BLOCK_SIZE,
    DeepseekV4Backend,
)
from runtime.model.dsv4_config import Dsv4Config  # noqa: E402
from runtime.model.dsv4_model import Dsv4Transformer  # noqa: E402
from runtime.sampling import SamplingParams  # noqa: E402

TINY = Dsv4Config(
    vocab_size=128,
    hidden_size=256,
    num_layers=1,
    max_position_embeddings=512,
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


def _backend(num_slots: int = 1) -> tuple[DeepseekV4Backend, list[tuple[int, int, int]]]:
    calls: list[tuple[int, int, int]] = []

    def forward_fn(slot: int, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        calls.append((slot, start_pos, input_ids.shape[1]))
        logits = torch.zeros(1, input_ids.shape[1], TINY.vocab_size)
        logits[..., 7] = 1.0
        return logits

    model = Dsv4Transformer(TINY, max_seq_len=512, device="cpu")
    backend = DeepseekV4Backend(
        model,
        TINY,
        num_slots=num_slots,
        max_seq_len=512,
        max_q_rows=32,
        device="cpu",
        forward_fn=forward_fn,
    )
    return backend, calls


def test_same_slot_exact_prefix_reuses_checkpoint_and_anchor() -> None:
    backend, calls = _backend()
    prompt = [index % TINY.vocab_size for index in range(DSV4_PREFIX_BLOCK_SIZE)]

    assert backend.prefill(0, prompt) == 7
    cold_calls = list(calls)
    backend.reset_slot(0)

    assert backend.capabilities.prefix_cache is True
    assert backend.reconcile_prefix_hit(prompt).effective == DSV4_PREFIX_BLOCK_SIZE
    assert backend.find_best_slot_for_prompt(prompt, [0]) == (0, DSV4_PREFIX_BLOCK_SIZE)
    assert backend.prefill(0, prompt) == 7

    assert calls == cold_calls
    assert backend.slot_state(0).kv_len == DSV4_PREFIX_BLOCK_SIZE
    assert backend.stats["prefix_same_slot_restores"] == 1


def test_same_slot_prefix_only_computes_suffix() -> None:
    backend, calls = _backend()
    prompt = [index % TINY.vocab_size for index in range(DSV4_PREFIX_BLOCK_SIZE)]
    backend.prefill(0, prompt)
    backend.reset_slot(0)
    calls.clear()

    extended = prompt + [3, 4, 5, 6]
    backend.prefill(0, extended)

    assert calls == [(0, DSV4_PREFIX_BLOCK_SIZE, 4)]
    assert backend.slot_state(0).kv_len == len(extended)
    assert backend.stats["prefill_tokens"] == DSV4_PREFIX_BLOCK_SIZE + 4


def test_prefix_mismatch_falls_back_to_cold_prefill() -> None:
    backend, calls = _backend()
    prompt = [index % TINY.vocab_size for index in range(DSV4_PREFIX_BLOCK_SIZE)]
    backend.prefill(0, prompt)
    backend.reset_slot(0)
    calls.clear()
    mismatch = list(prompt)
    mismatch[0] += 1

    backend.prefill(0, mismatch)

    assert calls[0] == (0, 0, 32)
    assert len(calls) == DSV4_PREFIX_BLOCK_SIZE // 32
    assert backend.stats["prefix_same_slot_restores"] == 0


def test_cross_slot_exact_prefix_copies_checkpoint_and_anchor() -> None:
    backend, calls = _backend(num_slots=2)
    prompt = [index % TINY.vocab_size for index in range(DSV4_PREFIX_BLOCK_SIZE)]
    backend.prefill(0, prompt)
    backend.reset_slot(0)
    calls.clear()

    assert backend.find_best_slot_for_prompt(prompt, [1]) == (1, DSV4_PREFIX_BLOCK_SIZE)
    assert backend.prefill(1, prompt) == 7

    assert calls == []
    assert backend.slot_state(1).kv_len == DSV4_PREFIX_BLOCK_SIZE
    assert backend.stats["prefix_cross_slot_restores"] == 1
    backend.reset_slot(1)
    assert backend.find_best_slot_for_prompt(prompt, [1]) == (1, DSV4_PREFIX_BLOCK_SIZE)


def test_cross_slot_restore_does_not_depend_on_pending_scheduler_hint() -> None:
    backend, calls = _backend(num_slots=2)
    prompt = [index % TINY.vocab_size for index in range(DSV4_PREFIX_BLOCK_SIZE)]
    backend.prefill(0, prompt)
    backend.reset_slot(0)
    calls.clear()

    assert backend.cross_slot_prefix_hit(prompt).effective == DSV4_PREFIX_BLOCK_SIZE
    assert backend.prefill(1, prompt) == 7

    assert calls == []
    assert backend.stats["prefix_cross_slot_restores"] == 1


def test_prefix_restore_recovers_immutable_window_ring_same_and_cross_slot() -> None:
    backend, _calls = _backend(num_slots=2)
    prompt = [index % TINY.vocab_size for index in range(DSV4_PREFIX_BLOCK_SIZE)]

    class _FakeLayer:
        def __init__(self) -> None:
            self.compressor = None
            self.indexer = None
            self.window_pages = torch.zeros(2, 1, 8, dtype=torch.uint8)

        def reset_caches(self, slot: int) -> None:
            return None

        def hard_clear_slot(self, slot: int) -> None:
            self.window_pages[slot].zero_()

        def copy_prefix(self, source_slot: int, destination_slot: int, length: int) -> None:
            self.window_pages[destination_slot].copy_(self.window_pages[source_slot])

        def clear_after_prefix(self, slot: int, length: int) -> None:
            return None

    layer = _FakeLayer()
    backend.slot_layers = [layer]
    layer.window_pages[0].fill_(17)
    backend.prefill(0, prompt)
    backend.reset_slot(0)

    # Continuation decode overwrites the live ring, but not its retained
    # checkpoint. Same-slot restore must recover the checkpoint bytes.
    layer.window_pages[0].fill_(91)
    backend.prefill(0, prompt)
    assert torch.equal(layer.window_pages[0], torch.full_like(layer.window_pages[0], 17))
    backend.reset_slot(0)

    # Cross-slot restore must also copy from the immutable checkpoint rather
    # than from the source slot's subsequently mutated live ring.
    layer.window_pages[0].fill_(123)
    backend.prefill(1, prompt)
    assert torch.equal(layer.window_pages[1], torch.full_like(layer.window_pages[1], 17))


def test_decode_failure_invalidates_live_and_retained_slot_state() -> None:
    backend, _calls = _backend()
    prompt = [index % TINY.vocab_size for index in range(DSV4_PREFIX_BLOCK_SIZE)]
    backend.prefill(0, prompt)
    backend.reset_slot(0)
    assert backend.prefill(0, prompt) == 7

    def fail_forward(slot: int, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        raise RuntimeError("injected decode failure")

    backend._forward_fn = fail_forward
    with pytest.raises(RuntimeError, match="injected decode failure"):
        backend.decode_batch_sampled(
            [0],
            [7],
            [DSV4_PREFIX_BLOCK_SIZE],
            [SamplingParams(temperature=0.0)],
        )

    assert backend.slot_state(0).is_fresh
    assert backend.reconcile_prefix_hit(prompt).effective == 0
