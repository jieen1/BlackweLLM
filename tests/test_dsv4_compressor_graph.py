"""Regression coverage for the DSV4 compressor CUDA-Graph state machine."""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from runtime.model.dsv4_attention import precompute_freqs_cis  # noqa: E402
from runtime.model.dsv4_config import Dsv4Config  # noqa: E402
from runtime.model.dsv4_model import Dsv4Compressor, Dsv4Indexer  # noqa: E402

CONFIG = Dsv4Config(
    vocab_size=32,
    hidden_size=8,
    num_layers=1,
    max_position_embeddings=64,
    num_heads=1,
    head_dim=8,
    rope_head_dim=4,
    q_lora_rank=8,
    o_groups=1,
    o_lora_rank=8,
    window_size=8,
    compress_ratios=(4,),
    index_n_heads=1,
    index_head_dim=4,
    index_topk=4,
    hc_mult=1,
    n_routed_experts=1,
    n_shared_experts=1,
    n_activated_experts=1,
    moe_intermediate_size=8,
    n_hash_layers=0,
)


def _compressor(config: Dsv4Config = CONFIG) -> Dsv4Compressor:
    coefficient = config.compressor_coeff(0)
    compressor = Dsv4Compressor(
        config,
        0,
        head_dim=config.head_dim,
        quantize=False,
        device="cpu",
    )
    output_size = coefficient * config.head_dim
    compressor.wkv = torch.nn.Linear(config.hidden_size, output_size, bias=False)
    compressor.wgate = torch.nn.Linear(config.hidden_size, output_size, bias=False)
    with torch.no_grad():
        values = torch.arange(output_size * config.hidden_size, dtype=torch.float32)
        compressor.wkv.weight.copy_(values.reshape(output_size, config.hidden_size) / 100)
        compressor.wgate.weight.copy_(values.flip(0).reshape(output_size, config.hidden_size) / 200)
        compressor.ape.copy_(
            torch.arange(compressor.ape.numel(), dtype=torch.float32).reshape_as(compressor.ape)
            / 1000
        )
        compressor.norm_weight.fill_(1)
    compressor.freqs_cis = precompute_freqs_cis(
        config.rope_head_dim,
        config.max_position_embeddings,
        original_seq_len=config.rope_original_seq_len,
        base=config.compress_rope_theta,
        factor=config.rope_factor,
        beta_fast=config.beta_fast,
        beta_slow=config.beta_slow,
    )
    compressor.kv_cache = torch.zeros(
        1,
        config.max_position_embeddings // compressor.ratio,
        config.head_dim,
        dtype=torch.bfloat16,
    )
    return compressor


def test_forward_graph_overlap_state_matches_eager_across_steps() -> None:
    eager = _compressor()
    graph = _compressor()
    graph.load_state_dict(eager.state_dict())

    generator = torch.Generator().manual_seed(1234)
    prefill = torch.randn(1, 1, CONFIG.hidden_size, generator=generator, dtype=torch.bfloat16)
    assert eager(prefill, 0) is None
    assert graph(prefill, 0) is None

    for position in range(1, 16):
        x = torch.randn(1, 1, CONFIG.hidden_size, generator=generator, dtype=torch.bfloat16)
        eager(x, position)
        graph.forward_graph(x, torch.tensor([position], dtype=torch.int64))

        assert not torch.isnan(graph.score_state).any(), f"NaN at position {position}"
        assert torch.equal(graph.kv_state, eager.kv_state), position
        assert torch.equal(graph.score_state, eager.score_state), position
        assert torch.equal(graph.kv_cache, eager.kv_cache), position


def test_forward_graph_ratio128_matches_eager_across_boundary() -> None:
    config = replace(
        CONFIG,
        max_position_embeddings=256,
        compress_ratios=(128,),
    )
    eager = _compressor(config)
    graph = _compressor(config)
    graph.load_state_dict(eager.state_dict())

    generator = torch.Generator().manual_seed(5678)
    prefill = torch.randn(1, 1, config.hidden_size, generator=generator, dtype=torch.bfloat16)
    assert eager(prefill, 0) is None
    assert graph(prefill, 0) is None

    for position in range(1, 130):
        x = torch.randn(1, 1, config.hidden_size, generator=generator, dtype=torch.bfloat16)
        eager(x, position)
        graph.forward_graph(x, torch.tensor([position], dtype=torch.int64))

        assert not torch.isnan(graph.score_state).any(), f"NaN at position {position}"
        assert torch.equal(graph.kv_state, eager.kv_state), position
        assert torch.equal(graph.score_state, eager.score_state), position
        assert torch.equal(graph.kv_cache, eager.kv_cache), position


def test_forward_graph_cpu_keeps_oracle_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compressor = _compressor()
    generator = torch.Generator().manual_seed(91011)
    prefill = torch.randn(1, 1, CONFIG.hidden_size, generator=generator, dtype=torch.bfloat16)
    assert compressor(prefill, 0) is None

    from runtime.kernels import dsv4_compressor

    monkeypatch.setattr(
        dsv4_compressor,
        "fused_decode_postgemv",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("CPU must stay on oracle path")),
    )

    x = torch.randn(1, 1, CONFIG.hidden_size, generator=generator, dtype=torch.bfloat16)
    compressor.forward_graph(x, torch.tensor([1], dtype=torch.int64))


def test_forward_graph_prefill_uses_known_host_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compressor = _compressor()
    x = torch.randn(1, 4, CONFIG.hidden_size, dtype=torch.bfloat16)
    opaque_device_position = object()

    from runtime.kernels import dsv4_compressor

    monkeypatch.setattr(dsv4_compressor, "supports_fused_decode_postgemv", lambda **_: True)

    def fused_seq(**kwargs) -> torch.Tensor:
        assert kwargs["pos0"] is opaque_device_position
        return kwargs["out"].zero_()

    monkeypatch.setattr(dsv4_compressor, "fused_decode_postgemv_seq", fused_seq)

    result = compressor.forward_graph_prefill(
        x,
        opaque_device_position,  # type: ignore[arg-type]
        host_start_pos=4,
    )

    assert result.shape == (1, 1, CONFIG.head_dim)


def test_capture_pack_uses_current_compressed_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    from runtime.model import dsv4_attn_kernel

    layer = object.__new__(dsv4_attn_kernel.Dsv4AttnKernelLayer)
    torch.nn.Module.__init__(layer)
    layer.num_slots = 1
    layer.ratio = 4
    layer.head_dim = 8
    layer.register_buffer("csa_pages", torch.empty(1, 1, dtype=torch.uint8))
    seen_ids: list[torch.Tensor] = []

    def record_pack(_entry, _pages, token_ids, **_kwargs) -> None:
        seen_ids.append(token_ids.clone())

    monkeypatch.setattr(dsv4_attn_kernel, "pack_latent_kv", record_pack)
    entry = torch.zeros(1, 1, layer.head_dim, dtype=torch.bfloat16)

    for position in (3, 4, 5, 7, 8):
        layer._pack_compressed(  # noqa: SLF001 -- focused address regression
            0,
            entry,
            -1,
            1,
            capture=True,
            pos_tensor=torch.tensor([position], dtype=torch.int64),
        )

    assert [int(ids.item()) for ids in seen_ids] == [0, 1, 1, 1, 2]


class _SpyCompressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ratio = 4
        self.kv_cache = None
        self.freqs_cis = None
        self.calls = 0

    def forward_graph(
        self, _x: torch.Tensor, _pos: torch.Tensor, *, slot: int = 0
    ) -> torch.Tensor:
        assert slot == 0
        self.calls += 1
        self.kv_cache[:, 0].fill_(1)
        return self.kv_cache[:, :1]

    def forward(self, _x: torch.Tensor, _start_pos: int, *, slot: int = 0) -> torch.Tensor:
        assert slot == 0
        self.calls += 1
        self.kv_cache[:, 0].fill_(1)
        return self.kv_cache[:, :1]


def test_indexer_forward_graph_advances_its_own_compressor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from runtime.model import dsv4_model

    indexer = object.__new__(Dsv4Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.num_slots = 1
    indexer.n_heads = 1
    indexer.head_dim = 4
    indexer.rope_head_dim = 4
    indexer.index_topk = 2
    indexer.softmax_scale = 1.0
    indexer.wq_b = torch.nn.Linear(4, 4, bias=False)
    indexer.weights_proj = torch.nn.Linear(4, 1, bias=False)
    indexer.compressor = _SpyCompressor()
    indexer.register_buffer("kv_cache", torch.zeros(1, 8, 4, dtype=torch.float32))
    indexer.register_buffer("freqs_cis", torch.zeros(16, 2))
    with torch.no_grad():
        indexer.wq_b.weight.copy_(torch.eye(4))
        indexer.weights_proj.weight.fill_(1)

    monkeypatch.setattr(dsv4_model, "apply_rotary_emb", lambda value, _freqs: value)
    monkeypatch.setattr(dsv4_model, "hadamard_transform", lambda value, _scale: value)
    monkeypatch.setattr(dsv4_model, "fp4_act_quant_simulate", lambda value, _block: value)

    x = torch.ones(1, 1, 4, dtype=torch.float32)
    qr = torch.ones(1, 1, 4, dtype=torch.float32)
    indices = indexer.forward_graph(
        x,
        qr,
        torch.tensor([3], dtype=torch.int64),
        max_entries=2,
    )

    assert indexer.compressor.calls == 1
    assert indexer.compressor.kv_cache is indexer.kv_cache
    assert indices.tolist() == [[[0, -1]]]


def test_indexer_forward_graph_skips_scoring_when_all_entries_are_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from runtime.model import dsv4_model

    class _UnexpectedProjection(torch.nn.Module):
        def forward(self, _x: torch.Tensor) -> torch.Tensor:
            raise AssertionError("the all-entries path must not score")

    indexer = object.__new__(Dsv4Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.num_slots = 1
    indexer.n_heads = 1
    indexer.head_dim = 4
    indexer.rope_head_dim = 4
    indexer.index_topk = 4
    indexer.softmax_scale = 1.0
    indexer.wq_b = _UnexpectedProjection()
    indexer.weights_proj = _UnexpectedProjection()
    indexer.compressor = _SpyCompressor()
    indexer.register_buffer("kv_cache", torch.zeros(1, 8, 4, dtype=torch.float32))
    indexer.register_buffer("freqs_cis", torch.zeros(16, 2))

    monkeypatch.setattr(dsv4_model, "apply_rotary_emb", lambda value, _freqs: value)

    indices = indexer.forward_graph(
        torch.ones(1, 1, 4),
        torch.ones(1, 1, 4),
        torch.tensor([7], dtype=torch.int64),
        max_entries=4,
    )

    assert indexer.compressor.calls == 1
    assert indices.tolist() == [[[0, 1, -1, -1]]]


def test_indexer_eager_decode_uses_same_padded_identity_indices() -> None:
    class _UnexpectedProjection(torch.nn.Module):
        def forward(self, _x: torch.Tensor) -> torch.Tensor:
            raise AssertionError("the all-entries path must not score")

    indexer = object.__new__(Dsv4Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.num_slots = 1
    indexer.n_heads = 1
    indexer.head_dim = 4
    indexer.rope_head_dim = 4
    indexer.index_topk = 4
    indexer.softmax_scale = 1.0
    indexer.wq_b = _UnexpectedProjection()
    indexer.weights_proj = _UnexpectedProjection()
    indexer.compressor = _SpyCompressor()
    indexer.register_buffer("kv_cache", torch.zeros(1, 8, 4, dtype=torch.float32))
    indexer.register_buffer("freqs_cis", torch.zeros(16, 2))

    indices = indexer.forward(
        torch.ones(1, 1, 4),
        torch.ones(1, 1, 4),
        start_pos=7,
        offset=10,
    )

    assert indexer.compressor.calls == 1
    assert indices.tolist() == [[[10, 11, -1, -1]]]


def test_indexer_forward_graph_rejects_bucket_smaller_than_topk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from runtime.model import dsv4_model

    indexer = object.__new__(Dsv4Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.num_slots = 1
    indexer.n_heads = 1
    indexer.head_dim = 4
    indexer.rope_head_dim = 4
    indexer.index_topk = 2
    indexer.softmax_scale = 1.0
    indexer.wq_b = torch.nn.Linear(4, 4, bias=False)
    indexer.weights_proj = torch.nn.Linear(4, 1, bias=False)
    indexer.compressor = _SpyCompressor()
    indexer.register_buffer("kv_cache", torch.zeros(1, 8, 4, dtype=torch.float32))
    indexer.register_buffer("freqs_cis", torch.zeros(16, 2))

    monkeypatch.setattr(dsv4_model, "apply_rotary_emb", lambda value, _freqs: value)
    monkeypatch.setattr(dsv4_model, "hadamard_transform", lambda value, _scale: value)
    monkeypatch.setattr(dsv4_model, "fp4_act_quant_simulate", lambda value, _block: value)

    x = torch.ones(1, 1, 4, dtype=torch.float32)
    qr = torch.ones(1, 1, 4, dtype=torch.float32)

    with pytest.raises(ValueError, match=">= index_topk"):
        indexer.forward_graph(
            x,
            qr,
            torch.tensor([3], dtype=torch.int64),
            max_entries=1,
        )
