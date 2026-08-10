from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.backends.dsv4 import DeepseekV4Backend, _decode_batch_chunks  # noqa: E402
from runtime.model.dsv4_attn_kernel import Dsv4AttnKernelLayer  # noqa: E402
from runtime.model.dsv4_config import Dsv4Config  # noqa: E402
from runtime.model.dsv4_model import Dsv4Compressor, Dsv4Indexer  # noqa: E402
from runtime.sampling import SamplingParams  # noqa: E402

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


def test_layer_slot_bases_use_page_capacity_not_live_lengths() -> None:
    layer = object.__new__(Dsv4AttnKernelLayer)
    layer.window_pages = torch.empty(2, 1, 16, dtype=torch.uint8)
    layer.prefill_pages = torch.empty(2, 2, 16, dtype=torch.uint8)
    layer.csa_pages = torch.empty(2, 3, 8, dtype=torch.uint8)

    assert Dsv4AttnKernelLayer._slot_raw_base(1, layer.window_pages, 128) == 128
    assert Dsv4AttnKernelLayer._slot_raw_base(1, layer.prefill_pages, 128) == 256
    assert Dsv4AttnKernelLayer._slot_raw_base(1, layer.csa_pages, 64) == 192
    assert Dsv4AttnKernelLayer._offset_valid_ids(torch.tensor([0, -1, 5]), 192).tolist() == [
        192,
        -1,
        197,
    ]
    batched = Dsv4AttnKernelLayer._offset_valid_ids_batch(
        torch.tensor([[0, -1, 5], [2, 3, -1]], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int64),
        192,
    )
    assert batched.dtype is torch.int32
    assert batched.is_contiguous()
    assert batched.tolist() == [[192, -1, 197], [2, 3, -1]]


def test_layer_reset_and_page_clear_only_touch_selected_slot() -> None:
    class _FakeStateOwner:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def reset_slot(self, slot: int) -> None:
            self.calls.append(slot)

        def reset_state(self, slot: int) -> None:
            self.calls.append(slot)

    layer = object.__new__(Dsv4AttnKernelLayer)
    layer.num_slots = 2
    layer.ratio = 4
    layer.compressor = _FakeStateOwner()
    layer.indexer = _FakeStateOwner()
    layer.window_pages = torch.full((2, 1, 4), 7, dtype=torch.uint8)
    layer.prefill_pages = torch.full((2, 2, 4), 9, dtype=torch.uint8)
    layer.csa_pages = torch.full((2, 3, 4), 11, dtype=torch.uint8)
    layer.hca_pages = torch.empty(2, 0, 0, dtype=torch.uint8)

    layer.reset_caches(1)
    layer.clear_pages(1)

    assert layer.compressor.calls == [1]
    assert layer.indexer.calls == [1]
    assert torch.equal(layer.window_pages[0], torch.full((1, 4), 7, dtype=torch.uint8))
    assert torch.equal(layer.prefill_pages[0], torch.full((2, 4), 9, dtype=torch.uint8))
    assert torch.equal(layer.csa_pages[0], torch.full((3, 4), 11, dtype=torch.uint8))
    assert torch.count_nonzero(layer.window_pages[1]) == 0
    assert torch.count_nonzero(layer.prefill_pages[1]) == 0
    assert torch.count_nonzero(layer.csa_pages[1]) == 0


@pytest.mark.parametrize(
    ("ratio", "compressed_attr", "entries"),
    ((4, "csa_pages", 64), (128, "hca_pages", 2)),
)
def test_clear_after_prefix_removes_only_selected_slot_tail(
    ratio: int,
    compressed_attr: str,
    entries: int,
) -> None:
    class _FakeCompressor:
        def __init__(self) -> None:
            self.kv_cache = torch.full((2, entries + 3, 2), 7, dtype=torch.bfloat16)

    class _FakeIndexer:
        def __init__(self) -> None:
            self.kv_cache = torch.full((2, entries + 3, 2), 11, dtype=torch.bfloat16)

    layer = object.__new__(Dsv4AttnKernelLayer)
    layer.num_slots = 2
    layer.ratio = ratio
    layer.compressor = _FakeCompressor()
    layer.indexer = _FakeIndexer() if ratio == 4 else None
    layer.csa_pages = torch.full((2, 3, 4), 13, dtype=torch.uint8)
    layer.hca_pages = torch.full((2, 3, 4), 17, dtype=torch.uint8)

    layer.clear_after_prefix(1, 256)

    compressed_pages = getattr(layer, compressed_attr)
    assert torch.count_nonzero(compressed_pages[1, :1]) == compressed_pages[1, :1].numel()
    assert torch.count_nonzero(compressed_pages[1, 1:]) == 0
    assert torch.count_nonzero(compressed_pages[0]) == compressed_pages[0].numel()
    assert torch.count_nonzero(layer.compressor.kv_cache[1, :entries]) > 0
    assert torch.count_nonzero(layer.compressor.kv_cache[1, entries:]) == 0
    assert torch.count_nonzero(layer.compressor.kv_cache[0]) > 0
    if layer.indexer is not None:
        assert torch.count_nonzero(layer.indexer.kv_cache[1, :entries]) > 0
        assert torch.count_nonzero(layer.indexer.kv_cache[1, entries:]) == 0
        assert torch.count_nonzero(layer.indexer.kv_cache[0]) > 0


def test_compressor_reset_slot_keeps_other_slot_state() -> None:
    compressor = Dsv4Compressor(TINY, 0, num_slots=2, quantize=False, device="cpu")
    compressor.kv_cache = torch.zeros(2, 16, TINY.head_dim, dtype=torch.bfloat16)
    compressor.kv_state[0].fill_(1.0)
    compressor.kv_state[1].fill_(2.0)
    compressor.score_state[0].fill_(3.0)
    compressor.score_state[1].fill_(4.0)

    compressor.reset_slot(1)

    assert torch.equal(compressor.kv_state[0], torch.ones_like(compressor.kv_state[0]))
    assert torch.equal(
        compressor.score_state[0],
        torch.full_like(compressor.score_state[0], 3.0),
    )
    assert torch.count_nonzero(compressor.kv_state[1]) == 0
    assert torch.isneginf(compressor.score_state[1]).all()


def test_indexer_reset_slot_keeps_other_slot_cache_and_recursive_state() -> None:
    indexer = Dsv4Indexer(TINY, 0, num_slots=2, max_seq_len=64, device="cpu")
    indexer.kv_cache[0].fill_(1.0)
    indexer.kv_cache[1].fill_(2.0)
    indexer.compressor.kv_state[0].fill_(5.0)
    indexer.compressor.kv_state[1].fill_(6.0)
    indexer.compressor.score_state[0].fill_(7.0)
    indexer.compressor.score_state[1].fill_(8.0)

    indexer.reset_slot(1)

    assert torch.equal(indexer.kv_cache[0], torch.ones_like(indexer.kv_cache[0]))
    assert torch.equal(
        indexer.compressor.kv_state[0],
        torch.full_like(indexer.compressor.kv_state[0], 5.0),
    )
    assert torch.equal(
        indexer.compressor.score_state[0],
        torch.full_like(indexer.compressor.score_state[0], 7.0),
    )
    assert torch.count_nonzero(indexer.kv_cache[1]) == 0
    assert torch.count_nonzero(indexer.compressor.kv_state[1]) == 0
    assert torch.isneginf(indexer.compressor.score_state[1]).all()


def test_indexer_state_reset_preserves_prefix_cache_bytes() -> None:
    indexer = Dsv4Indexer(TINY, 0, num_slots=2, max_seq_len=64, device="cpu")
    indexer.kv_cache[1].fill_(2.0)
    indexer.compressor.kv_state[1].fill_(6.0)
    indexer.compressor.score_state[1].fill_(8.0)

    indexer.reset_state(1)

    assert torch.equal(indexer.kv_cache[1], torch.full_like(indexer.kv_cache[1], 2.0))
    assert torch.count_nonzero(indexer.compressor.kv_state[1]) == 0
    assert torch.isneginf(indexer.compressor.score_state[1]).all()


def test_backend_serial_slot_order_works_with_shared_slot_stack_contract() -> None:
    calls: list[tuple[int, int, int]] = []

    def forward_fn(slot: int, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        calls.append((slot, start_pos, input_ids.shape[1]))
        logits = torch.zeros(
            1, input_ids.shape[1], TINY.vocab_size, dtype=torch.float32, device=input_ids.device
        )
        logits[0, -1, 10 + slot] = 2.0
        return logits

    backend = DeepseekV4Backend(
        object(),
        TINY,
        num_slots=2,
        max_seq_len=64,
        device="cpu",
        forward_fn=forward_fn,
    )

    assert backend.prefill(1, [4]) == 11
    assert backend.prefill(0, [9]) == 10

    out = backend.decode_batch_sampled(
        [1, 0],
        [5, 6],
        [1, 1],
        [SamplingParams(temperature=0.0), SamplingParams(temperature=0.0)],
    )

    assert out == [11, 10]
    assert calls == [(1, 0, 1), (0, 0, 1), (1, 1, 1), (0, 1, 1)]
    assert backend.slot_state(0).committed_tokens == (9, 6)
    assert backend.slot_state(1).committed_tokens == (4, 5)


def test_decode_batch_chunks_prefers_largest_native_bucket() -> None:
    assert _decode_batch_chunks(0) == ()
    assert _decode_batch_chunks(1) == (1,)
    assert _decode_batch_chunks(3) == (2, 1)
    assert _decode_batch_chunks(4) == (4,)
    assert _decode_batch_chunks(7) == (4, 2, 1)


def test_backend_reserves_decode_rows_without_widening_prefill_chunks() -> None:
    backend = DeepseekV4Backend(
        object(),
        TINY,
        num_slots=4,
        max_seq_len=64,
        max_q_rows=1,
        device="cpu",
        forward_fn=lambda *_args: None,
    )

    assert backend.max_q_rows == 1
    assert backend._kernel_max_q_rows == 4


def test_backend_native_b4_graph_replays_once_and_preserves_slot_order() -> None:
    backend = DeepseekV4Backend(
        object(),
        TINY,
        num_slots=4,
        max_seq_len=64,
        device="cpu",
        forward_fn=lambda *_args: None,
    )
    backend._forward_fn = None
    backend._native_decode_batch_available = True
    backend._kv_len = [1, 1, 1, 1]
    calls: list[tuple[list[int], list[int], list[int], int | None]] = []

    class FakeGraph:
        def replay_host(
            self,
            input_ids: list[int],
            positions: list[int],
            slot_ids: list[int],
            *,
            max_index_entries: int | None,
        ) -> torch.Tensor:
            calls.append((input_ids, positions, slot_ids, max_index_entries))
            logits = torch.zeros(4, 1, TINY.vocab_size)
            for row, slot in enumerate(slot_ids):
                logits[row, 0, 20 + slot] = 3.0
            return logits

    backend._decode_graphs = {4: FakeGraph()}
    out = backend.decode_batch_sampled(
        [3, 1, 0, 2],
        [7, 8, 9, 10],
        [1, 1, 1, 1],
        [SamplingParams(temperature=0.0)] * 4,
    )

    assert out == [23, 21, 20, 22]
    assert calls == [([7, 8, 9, 10], [1, 1, 1, 1], [3, 1, 0, 2], None)]
    assert backend.stats["decode_graph_replays"] == 1
    assert backend.stats["decode_tokens"] == 4
    assert backend._kv_len == [2, 2, 2, 2]


def test_backend_rejects_duplicate_slot_before_mutation() -> None:
    backend = DeepseekV4Backend(
        object(),
        TINY,
        num_slots=2,
        max_seq_len=64,
        device="cpu",
        forward_fn=lambda *_args: None,
    )

    with pytest.raises(ValueError, match="same slot"):
        backend.decode_batch_sampled(
            [1, 1],
            [7, 8],
            [0, 0],
            [SamplingParams(temperature=0.0)] * 2,
        )

    assert backend._kv_len == [0, 0]
    assert backend._committed == [[], []]


def test_backend_unavailable_native_batch_falls_back_to_serial_b1() -> None:
    backend = DeepseekV4Backend(
        object(),
        TINY,
        num_slots=2,
        max_seq_len=64,
        device="cpu",
        forward_fn=lambda *_args: None,
    )
    backend._forward_fn = None
    backend._native_decode_batch_available = False
    calls: list[tuple[int, int]] = []

    def serial_forward(slot: int, input_ids: torch.Tensor, position: int) -> torch.Tensor:
        calls.append((slot, position))
        logits = torch.zeros(1, 1, TINY.vocab_size)
        logits[0, 0, 30 + slot] = 2.0
        return logits

    backend._forward = serial_forward
    out = backend.decode_batch_sampled(
        [1, 0],
        [5, 6],
        [0, 0],
        [SamplingParams(temperature=0.0)] * 2,
    )

    assert out == [31, 30]
    assert calls == [(1, 0), (0, 0)]
    assert backend.stats["decode_eager_fallbacks"] == 2
