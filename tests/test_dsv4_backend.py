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

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from runtime.backends.dsv4 import (  # noqa: E402
    DeepseekV4Backend,
    _enable_serving_q8_kernels,
)
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


def test_serving_q8_policy_enables_lm_head_without_blanket_fp32_conversion() -> None:
    model = _zeroed_model()

    assert model.lm_head.weight_dtype is None
    assert model.blocks[0].moe.shared_w1.weight_dtype is None
    assert model.blocks[0].attn.wq_a.weight_dtype is torch.bfloat16

    _enable_serving_q8_kernels(model)

    assert model.lm_head.fused_q8 is True
    assert model.blocks[0].attn.wq_a.fused_q8 is True
    assert model.blocks[0].moe.shared_w1.fused_q8 is False
    assert model.blocks[0].moe.shared_w1.fused_q8_fp32 is True


# -- protocol conformance (GPU-free) -----------------------------------------


def test_backend_conforms_with_cuda_graph_capability() -> None:
    problems = check_conformance(
        DeepseekV4Backend,
        BackendCapabilities(False, True, True, False, False),
    )
    assert problems == [], problems


def test_backend_declares_cuda_graph_capability() -> None:
    caps = DeepseekV4Backend.capabilities.fget(None)
    assert caps.speculative_decode is False
    assert caps.prefix_cache is True
    assert caps.cuda_graph is True
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
    assert backend.slot_state(0).committed_tokens == (1, 2, 3, 7)
    # prefill is chunked at max_q_rows (=1 in this tiny backend): one forward
    # per token at absolute positions 0,1,2, then decode at kv_len=3.
    assert backend._test_calls == [(0, 0, 1), (0, 1, 1), (0, 2, 1), (0, 3, 1)]
    assert backend.stats["prefill_calls"] == 1
    assert backend.stats["prefill_chunks"] == 3
    assert backend.stats["prefill_tokens"] == 3
    assert backend.stats["decode_rounds"] == 1
    assert backend.stats["decode_tokens"] == 1
    assert backend.stats["decode_eager_fallbacks"] == 1
    assert backend.cg_fallback_reasons == {"not_captured": 1}


def test_prefill_rejects_capacity_overflow_before_forward_mutation() -> None:
    backend = _make_backend()

    with pytest.raises(IndexError, match="prefill length 65 exceeds max_seq_len=64"):
        backend.prefill(0, list(range(65)))

    assert backend._test_calls == []
    assert backend.slot_state(0).is_fresh


def test_decode_uses_slot_cuda_graph_when_captured() -> None:
    backend = _make_backend()
    backend.prefill(0, [1, 2])
    calls: list[tuple[int, int, int]] = []

    class FakeGraph:
        def replay(self, slot: int, token: int, position: int) -> torch.Tensor:
            calls.append((slot, token, position))
            logits = torch.zeros(1, 1, TINY.vocab_size, device="cuda")
            logits[0, 0, 9] = 3
            return logits

    eager_calls = list(backend._test_calls)
    backend._decode_graphs[0] = FakeGraph()
    out = backend.decode_batch_sampled([0], [7], [2], [SamplingParams(temperature=0.0)])

    assert out == [9]
    assert calls == [(0, 7, 2)]
    assert backend._test_calls == eager_calls
    assert backend.stats["decode_graph_replays"] == 1
    assert backend.stats["decode_eager_fallbacks"] == 0


def test_prefill_and_decode_emit_dsv4_flight_recorder_rows(monkeypatch) -> None:
    from bfdiag.trace import ring as bfdiag_trace

    monkeypatch.setattr(bfdiag_trace, "TRACE_ENABLED", True)
    bfdiag_trace.reset(16, use_cuda=False)
    backend = _make_backend()

    anchor = backend.prefill(0, [1, 2, 3])
    backend.decode_batch_sampled(
        [0],
        [anchor],
        [3],
        [SamplingParams(temperature=0.0)],
    )

    rows = bfdiag_trace.get_ring().snapshot()
    assert [row.event_kind for row in rows] == [
        "prefill_chunk",
        "prefill_chunk",
        "prefill_chunk",
        "decode_round",
    ]
    assert [(row.position, row.row_count) for row in rows] == [
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
    ]
    decode = rows[-1]
    assert decode.path == "eager"
    assert decode.cg_miss_reason == "not_captured"
    assert decode.window_entries == 4
    assert decode.ratio4_entries == 1
    assert decode.ratio128_entries == 0
    assert decode.t_round > 0.0


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


def test_decode_batch_sampled_rejects_any_parameter_length_mismatch() -> None:
    backend = _make_backend(num_slots=2)
    backend.prefill(0, [1])
    with pytest.raises(ValueError, match="equal length"):
        backend.decode_batch_sampled(
            [0],
            [7, 8],
            [1],
            [SamplingParams(temperature=0.0), SamplingParams(temperature=0.0)],
        )


def test_snapshot_shapes() -> None:
    backend = _make_backend(num_slots=2)
    snap = backend.snapshot()
    assert len(snap.slots) == 2
    assert [s.slot for s in snap.slots] == [0, 1]
    assert len(snap.prefix) == 2
    assert snap.dflash_cg_status == ()
    assert dict(snap.runtime_stats)["decode_graph_replays"] == 0
    assert snap.cg_fallback_reasons == ()


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


def test_decode_graphs_are_slot_isolated_across_interleaved_replay_and_reset() -> None:
    backend = _make_backend(num_slots=2)
    assert backend.prefill(0, [1, 2]) == 3
    assert backend.prefill(1, [4, 5]) == 3

    class FakeLayer:
        def __init__(self) -> None:
            self.reset_slots: list[int] = []

        def reset_caches(self, slot: int) -> None:
            self.reset_slots.append(slot)

    class FakeGraph:
        def __init__(self, slot: int, out_token: int) -> None:
            self.slot = slot
            self.out_token = out_token
            self.calls: list[tuple[int, int, int]] = []

        def replay(self, slot: int, token: int, position: int) -> torch.Tensor:
            self.calls.append((slot, token, position))
            logits = torch.zeros(1, 1, TINY.vocab_size, device="cuda")
            logits[0, 0, self.out_token] = 5.0
            return logits

    layers = [FakeLayer(), FakeLayer()]
    slot0_graph = FakeGraph(0, 11)
    slot1_graph = FakeGraph(1, 17)
    backend.slot_layers = layers
    backend._decode_graphs = {0: slot0_graph, 1: slot1_graph}
    eager_calls = list(backend._test_calls)

    out = backend.decode_batch_sampled(
        [1, 0],
        [8, 7],
        [2, 2],
        [SamplingParams(temperature=0.0), SamplingParams(temperature=0.0)],
    )

    assert out == [17, 11]
    assert slot0_graph.calls == [(0, 7, 2)]
    assert slot1_graph.calls == [(1, 8, 2)]
    assert backend._test_calls == eager_calls
    assert backend.slot_state(0).kv_len == 3
    assert backend.slot_state(0).committed_tokens == (1, 2, 7)
    assert backend.slot_state(1).kv_len == 3
    assert backend.slot_state(1).committed_tokens == (4, 5, 8)

    backend.reset_slot(0)

    assert [layer.reset_slots for layer in layers] == [[0], [0]]
    assert backend.slot_state(0).is_fresh
    assert backend.slot_state(0).committed_tokens == ()
    assert backend.slot_state(1).kv_len == 3
    assert backend.slot_state(1).committed_tokens == (4, 5, 8)

    out = backend.decode_batch_sampled([1], [9], [3], [SamplingParams(temperature=0.0)])

    assert out == [17]
    assert slot0_graph.calls == [(0, 7, 2)]
    assert slot1_graph.calls == [(1, 8, 2), (1, 9, 3)]
    assert backend.slot_state(1).kv_len == 4
    assert backend.slot_state(1).committed_tokens == (4, 5, 8, 9)


def test_capture_decode_graphs_rejects_active_slots() -> None:
    backend = _make_backend(num_slots=1)
    backend.prefill(0, [1])
    backend._forward_fn = None
    backend.slot_layers = [object()]
    backend._native_decode_batch_available = True

    with pytest.raises(RuntimeError, match="before slot admission.*0"):
        backend.capture_decode_cuda_graph()


def test_capture_decode_graphs_is_atomic_and_idempotent(monkeypatch) -> None:
    backend = _make_backend(num_slots=2)
    backend._forward_fn = None
    backend._native_decode_batch_available = True

    class FakeLayer:
        def __init__(self) -> None:
            self.reset_slots: list[int] = []

        def reset_caches(self, slot: int) -> None:
            self.reset_slots.append(slot)

    layers = [FakeLayer(), FakeLayer()]
    backend.slot_layers = layers
    built: list[object] = []

    class FakeDriver:
        def __init__(self, **kwargs) -> None:
            self.backend = kwargs["backend"]
            self.captures = 0
            built.append(self)

        def capture(self) -> None:
            self.captures += 1

    monkeypatch.setattr(
        "runtime.backends.dsv4_cudagraph.build_batched_decode_graph_driver", FakeDriver
    )

    assert backend.capture_decode_cuda_graph() == 2
    assert set(backend._decode_graphs) == {1, 2}
    assert len(built) == 1
    assert built[0].captures == 1
    assert backend._decode_graphs[1] is backend._decode_graphs[2] is built[0]
    assert [layer.reset_slots for layer in layers] == [[0, 1], [0, 1]]

    backend._kv_len[0] = 1
    assert backend.capture_decode_cuda_graph() == 2
    assert [layer.reset_slots for layer in layers] == [[0, 1], [0, 1]]


def test_capture_decode_graphs_rolls_back_partial_failure(monkeypatch) -> None:
    backend = _make_backend(num_slots=2)
    backend._forward_fn = None
    backend._native_decode_batch_available = True

    class FakeLayer:
        def __init__(self) -> None:
            self.reset_slots: list[int] = []

        def reset_caches(self, slot: int) -> None:
            self.reset_slots.append(slot)

    layers = [FakeLayer(), FakeLayer()]
    backend.slot_layers = layers

    class FailingDriver:
        def __init__(self, **kwargs) -> None:
            return None

        def capture(self) -> None:
            raise RuntimeError("capture failed")

    monkeypatch.setattr(
        "runtime.backends.dsv4_cudagraph.build_batched_decode_graph_driver", FailingDriver
    )

    assert backend.capture_decode_cuda_graph() is None
    assert backend._decode_graphs == {}
    assert backend.snapshot().dflash_cg_status == (("decode", "failed"),)
    assert [layer.reset_slots for layer in layers] == [[0, 1], [0, 1]]


def test_bucketed_decode_graph_driver_picks_smallest_covering_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from runtime.backends import dsv4_cudagraph

    class FakeIndexer:
        def __init__(self) -> None:
            self.kv_cache = torch.zeros(1, 16384, 4)

    class FakeLayer:
        def __init__(self) -> None:
            self.indexer = FakeIndexer()

    seen_caps: list[tuple[int | None, int]] = []

    class FakeDriver:
        def __init__(self, *, max_index_entries=None, graph_pool=None, **kwargs) -> None:
            self.max_index_entries = max_index_entries
            self.graph_pool = graph_pool or object()

        def capture(self, slot: int) -> None:
            self.slot = slot

        def replay(self, slot: int, token: int, position: int) -> torch.Tensor:
            seen_caps.append((self.max_index_entries, position))
            return torch.zeros(1, 1, TINY.vocab_size, device="cuda")

    monkeypatch.setattr(dsv4_cudagraph, "Dsv4DecodeGraphDriver", FakeDriver)

    graph = dsv4_cudagraph.build_decode_graph_driver(
        model=object(),
        kernel_layers=[FakeLayer()],
        max_seq_len=65536,
        device="cuda",
    )
    graph.capture(0)

    assert [cap for cap, _driver in graph._drivers] == [512, 1024, 4096, 16384]

    graph.replay(0, 7, 3)
    graph.replay(0, 7, 4099)
    graph.replay(0, 7, 65000)

    assert seen_caps == [
        (512, 3),
        (4096, 4099),
        (16384, 65000),
    ]


def test_q8_block_n_selection_matches_sm120_real_weight_sweep() -> None:
    from runtime.kernels.dsv4_q8_gemm import (
        _select_q8_0_block_n,
        _select_q8_0_grouped_block_n,
    )

    assert _select_q8_0_block_n(1, 4096, 1024) == 8
    assert _select_q8_0_block_n(1, 1024, 32768) == 64
    assert _select_q8_0_block_n(1, 4096, 129280) == 32
    assert _select_q8_0_block_n(32, 4096, 129280) == 64
    assert _select_q8_0_grouped_block_n(1) == 16
    assert _select_q8_0_grouped_block_n(2) == 16
    assert _select_q8_0_grouped_block_n(4) == 16
    assert _select_q8_0_grouped_block_n(5) == 64
    assert _select_q8_0_grouped_block_n(32) == 64


def test_out_of_range_slot_raises() -> None:
    backend = _make_backend(num_slots=1)
    with pytest.raises(IndexError, match="out of range"):
        backend.prefill(1, [1])
    with pytest.raises(IndexError, match="out of range"):
        backend.slot_state(1)


def test_logprobs_return_chosen_and_top_tokens() -> None:
    backend = _make_backend()
    backend.prefill(0, [1])
    tokens, logprobs = backend.decode_batch_sampled(
        [0],
        [7],
        [1],
        [SamplingParams(temperature=0.0)],
        return_logprobs=True,
        top_logprobs=3,
    )

    assert len(tokens) == len(logprobs) == 1
    assert logprobs[0]["token_id"] == tokens[0]
    assert len(logprobs[0]["top_logprobs"]) == 3
    assert logprobs[0]["top_logprobs"][0]["token_id"] == tokens[0]
    assert logprobs[0]["logprob"] == pytest.approx(logprobs[0]["top_logprobs"][0]["logprob"])


def test_empty_logprobs_batch_preserves_protocol_shape() -> None:
    backend = _make_backend()
    assert backend.decode_batch_sampled([], [], [], [], return_logprobs=True) == ([], [])


# -- empty prefix-cache surface ----------------------------------------------


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
    state = backend.prefill_chunked_begin([0], [[1, 2]], params_per_slot={0: params})
    assert state.done is True
    # The sampled anchor comes from the stub logits (uniform + 2.0 on 3);
    # sampling with seed is deterministic -- just assert it is a valid token.
    assert 0 <= state.result[0]["anchor"] < TINY.vocab_size


def test_prefill_chunked_step_returns_done() -> None:
    backend = _make_backend()
    state = backend.prefill_chunked_begin([0], [[1]])
    assert backend.prefill_chunked_step(state) is True


# -- prefill CUDA-graph dispatch (tail-chunk fallback) -----------------------


class _IdentityAttnLayer:
    """Stands in for Dsv4AttnKernelLayer: the kernel stack needs real 512-dim
    shapes, but ``_forward``'s MoE dispatch is shape-independent."""

    def __call__(self, x: torch.Tensor, start_pos: int, slot: int = 0) -> torch.Tensor:
        return x


class _FakePrefillGraph:
    """Mimics Dsv4PrefillGraphDriver's fixed-M replay surface."""

    def __init__(self, m: int = 64) -> None:
        self.m = m
        self.calls: list[tuple[int, int]] = []

    def replay_layer(self, layer_id: int, flat: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        self.calls.append((layer_id, flat.shape[0]))
        return torch.zeros_like(flat)


def _backend_with_prefill_graph(max_q_rows: int = 64) -> DeepseekV4Backend:
    """Tiny zeroed backend with a fake prefill graph and identity attention,
    so ``_forward`` runs on a 1-layer model without the 512-dim kernel stack."""
    model = _zeroed_model()
    backend = DeepseekV4Backend(
        model, TINY, num_slots=1, max_seq_len=64, max_q_rows=max_q_rows, device="cuda"
    )
    backend.slot_layers = [_IdentityAttnLayer()]
    backend._prefill_graph = _FakePrefillGraph(m=max_q_rows)
    return backend


def test_prefill_graph_used_only_at_exact_m_rows() -> None:
    backend = _backend_with_prefill_graph(max_q_rows=64)
    graph = backend._prefill_graph
    assert isinstance(graph, _FakePrefillGraph)

    # A full 64-row chunk goes through the CUDA graph.
    backend._forward(0, torch.tensor([[7] * 64], dtype=torch.long, device="cuda"), 0)
    assert graph.calls == [(0, 64)]


def test_prefill_tail_chunk_falls_back_to_eager_moe() -> None:
    backend = _backend_with_prefill_graph(max_q_rows=64)
    graph = backend._prefill_graph
    assert isinstance(graph, _FakePrefillGraph)
    block = backend.model.blocks[0]
    orig_forward = block.moe.forward
    eager_calls: list[int] = []

    def spy_moe(x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        eager_calls.append(input_ids.shape[1])
        return orig_forward(x, input_ids)

    block.moe.forward = spy_moe  # type: ignore[method-assign]
    try:
        # A short tail chunk (10 rows, prompt not a multiple of 64) must NOT
        # hit the graph: the graph's replay contract is fixed (M, H).
        backend._forward(0, torch.tensor([[3] * 10], dtype=torch.long, device="cuda"), 64)
        assert graph.calls == []
        assert eager_calls == [10]
    finally:
        block.moe.forward = orig_forward  # type: ignore[method-assign]


def test_prefill_graph_dispatch_handles_m1_decode_fallback() -> None:
    backend = _backend_with_prefill_graph(max_q_rows=64)
    graph = backend._prefill_graph
    assert isinstance(graph, _FakePrefillGraph)
    block = backend.model.blocks[0]
    orig_forward = block.moe.forward
    eager_calls: list[int] = []

    def spy_moe(x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        eager_calls.append(input_ids.shape[1])
        return orig_forward(x, input_ids)

    block.moe.forward = spy_moe  # type: ignore[method-assign]
    try:
        # The serial decode fallback also funnels through ``_forward`` with a
        # single-row input; it must stay on the eager path.
        backend._forward(0, torch.tensor([[9]], dtype=torch.long, device="cuda"), 74)
        assert graph.calls == []
        assert eager_calls == [1]
    finally:
        block.moe.forward = orig_forward  # type: ignore[method-assign]


class _FakeSuperchunkLayer:
    def __init__(self) -> None:
        self.ratio = 4
        self.eager_calls: list[tuple[int, int, int]] = []
        self.prefill_calls: list[tuple[int | None, int | None, bool, int]] = []
        self.precompute_calls: list[tuple[int, int]] = []
        self.reset_calls = 0

    def __call__(self, x: torch.Tensor, start_pos: int, slot: int = 0) -> torch.Tensor:
        self.eager_calls.append((start_pos, x.shape[1], slot))
        return x

    def precompute_cold_prefill_compressors(
        self,
        x: torch.Tensor,
        *,
        completed_rows: int,
        slot: int = 0,
    ) -> None:
        self.precompute_calls.append((completed_rows, slot))

    def forward_graph_prefill(
        self,
        x: torch.Tensor,
        pos_t: torch.Tensor,
        *,
        slot: int = 0,
        graph_max_index_entries: int | None = None,
        host_start_pos: int | None = None,
        compressors_precomputed: bool = False,
    ) -> torch.Tensor:
        self.prefill_calls.append(
            (int(pos_t.reshape(-1)[0]), host_start_pos, compressors_precomputed, slot)
        )
        return x

    def reset_caches(self, slot: int = 0) -> None:
        self.reset_calls += 1


class _FakeSuperchunkMoE:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...], object]] = []

    def forward_dynamic(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        *,
        workspace: object | None = None,
    ) -> torch.Tensor:
        self.calls.append((tuple(x.shape), tuple(ids.shape), workspace))
        return x


class _FakeSuperchunkBlock:
    def __init__(self, hidden: int) -> None:
        self.eps = 1e-6
        self.hc_attn_fn = None
        self.hc_attn_scale = 1.0
        self.hc_attn_base = 0.0
        self.hc_ffn_fn = None
        self.hc_ffn_scale = 1.0
        self.hc_ffn_base = 0.0
        self.attn_norm_weight = torch.ones(hidden)
        self.ffn_norm_weight = torch.ones(hidden)
        self.moe = _FakeSuperchunkMoE()

    def hc_pre(
        self,
        h: torch.Tensor,
        _fn: object,
        _scale: float,
        _base: float,
    ) -> tuple[torch.Tensor, None, None]:
        return h.flatten(2) if h.ndim == 4 else h, None, None

    def hc_post(
        self,
        x: torch.Tensor,
        _residual: torch.Tensor,
        _post: None,
        _comb: None,
    ) -> torch.Tensor:
        return x


class _FakeSuperchunkModel:
    def __init__(self, hidden: int = 8) -> None:
        self.hc_mult = 1
        self.norm_weight = torch.ones(hidden)
        self.eps = 1e-6
        self.config = SimpleNamespace(
            hidden_size=hidden,
            n_activated_experts=2,
            moe_intermediate_size=16,
        )
        self.blocks = [_FakeSuperchunkBlock(hidden)]

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, ids.shape[1], self.config.hidden_size, dtype=torch.bfloat16)

    def hc_head(self, h: torch.Tensor) -> torch.Tensor:
        return h.flatten(2)

    def lm_head(self, h: torch.Tensor) -> torch.Tensor:
        return h.float()


class _FakeDynamicWorkspace:
    @staticmethod
    def create(*_args: object, **_kwargs: object) -> object:
        return object()


class _FakeSuperchunkGraph:
    def __init__(self, *, rows: int = 1024, tile: int = 64, prefix_len: int = 0) -> None:
        self.m = rows
        self.tile = tile
        self.prefix_len = prefix_len
        self.calls: list[tuple[int, tuple[int, ...]]] = []

    def matches(self, *, slot: int, prefix_len: int, tile: int, rows: int) -> bool:
        return slot == 0 and prefix_len == self.prefix_len and tile == self.tile and rows == self.m

    def replay_layer(self, layer_id: int, x: torch.Tensor) -> torch.Tensor:
        self.calls.append((layer_id, tuple(x.shape)))
        return x + 1


def _make_fake_superchunk_backend() -> tuple[DeepseekV4Backend, _FakeSuperchunkLayer]:
    backend = DeepseekV4Backend.__new__(DeepseekV4Backend)
    backend.model = _FakeSuperchunkModel()
    backend.config = backend.model.config
    backend.num_slots = 1
    backend.max_seq_len = 2048
    backend.max_q_rows = 64
    backend.device = "cpu"
    backend._forward_fn = None
    layer = _FakeSuperchunkLayer()
    backend.slot_layers = [layer]
    backend._kv_len = [0]
    backend._committed = [[]]
    backend._cg_status = {}
    backend._prefill_graph = None
    backend._superchunk_prefill_graph = None
    backend._capture_prefix_checkpoint = lambda *_args: None
    return backend, layer


def test_superchunk_prefill_graph_replays_only_on_exact_shape_and_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, layer = _make_fake_superchunk_backend()
    graph = _FakeSuperchunkGraph()
    backend._superchunk_prefill_graph = graph
    monkeypatch.setitem(
        sys.modules,
        "runtime.kernels.iq2_mma16_tc",
        SimpleNamespace(DynamicMoEWorkspace=_FakeDynamicWorkspace),
    )

    logits = backend._prefill_superchunk_logits(0, list(range(1024)), tile=64, prefix_len=0)

    assert tuple(logits.shape) == (1, 1024, backend.model.config.hidden_size)
    assert graph.calls == [(0, (1, 1024, backend.model.config.hidden_size))]
    assert layer.eager_calls == []
    assert layer.prefill_calls == []
    assert layer.precompute_calls == []
    assert len(backend.model.blocks[0].moe.calls) == 1


def test_superchunk_prefill_graph_mismatch_falls_back_to_tile_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, layer = _make_fake_superchunk_backend()
    backend._superchunk_prefill_graph = _FakeSuperchunkGraph(rows=1024, tile=64, prefix_len=0)
    monkeypatch.setitem(
        sys.modules,
        "runtime.kernels.iq2_mma16_tc",
        SimpleNamespace(DynamicMoEWorkspace=_FakeDynamicWorkspace),
    )

    logits = backend._prefill_superchunk_logits(0, list(range(960)), tile=64, prefix_len=0)

    # The final non-aligned tail is a separate superchunk so the preceding
    # 768-token boundary can be published as a restorable prefix checkpoint.
    assert tuple(logits.shape) == (1, 192, backend.model.config.hidden_size)
    assert layer.eager_calls == [(0, 64, 0)]
    assert [call[:3] for call in layer.prefill_calls] == [
        (64, 64, True),
        (128, 128, True),
        (192, 192, True),
        (256, 256, True),
        (320, 320, True),
        (384, 384, True),
        (448, 448, True),
        (512, 512, True),
        (576, 576, True),
        (640, 640, True),
        (704, 704, True),
        (768, 768, False),
        (832, 832, False),
        (896, 896, False),
    ]
    assert layer.precompute_calls == [(64, 0)]


def test_kernel_superchunk_prefill_publishes_last_aligned_prefix_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, _layer = _make_fake_superchunk_backend()
    backend.stats = {"prefill_calls": 0, "prefill_chunks": 0, "prefill_tokens": 0}
    captures: list[tuple[int, int, list[int], tuple[int, ...]]] = []
    monkeypatch.setattr(backend, "_apply_same_slot_prefix", lambda _slot, _ids: (0, None))
    monkeypatch.setattr(
        backend,
        "_capture_prefix_checkpoint",
        lambda slot, length, token_ids, logits: captures.append(
            (slot, length, list(token_ids), tuple(logits.shape))
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "runtime.kernels.iq2_mma16_tc",
        SimpleNamespace(DynamicMoEWorkspace=_FakeDynamicWorkspace),
    )
    monkeypatch.setenv("QSR_DSV4_MOE_CHUNK", "256")
    prompt = list(range(384))

    backend._prefill_logits(0, prompt)

    assert captures == [(0, 256, prompt, (1, 256, backend.model.config.hidden_size))]
    assert backend._kv_len == [384]
    assert backend._committed == [prompt]


def test_capture_prefill_cuda_graph_uses_superchunk_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = DeepseekV4Backend.__new__(DeepseekV4Backend)
    backend._superchunk_prefill_graph = None
    legacy_graph = object()
    backend._prefill_graph = legacy_graph
    backend._forward_fn = None
    backend.slot_layers = [object()]
    backend.device = "cuda"
    backend.num_slots = 1
    backend.max_q_rows = 64
    backend._kv_len = [0]
    backend._committed = [[]]
    backend._cg_status = {}

    captures: list[DeepseekV4Backend] = []

    class FakeDriver:
        def __init__(self, arg_backend: DeepseekV4Backend) -> None:
            captures.append(arg_backend)

        def capture(self) -> None:
            captures.append(backend)

    monkeypatch.setattr("runtime.backends.dsv4.Dsv4SuperchunkPrefillGraphDriver", FakeDriver)
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda _device: (int(4 * 2**30), int(8 * 2**30)),
    )

    assert backend.capture_prefill_cuda_graph() is True
    assert captures == [backend, backend]
    assert isinstance(backend._superchunk_prefill_graph, FakeDriver)
    assert backend._prefill_graph is legacy_graph
    assert backend._cg_status["prefill"] == "captured"
