from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from runtime.model.flashnext.mtp import (  # noqa: E402
    BF16MoE,
    FlashNextMTP,
    _shared_sparse_captured_len,
    _shared_sparse_reuse,
    quantize_mtp_expert_weight,
)
from runtime.model.flashnext.mtp_kernels import (  # noqa: E402
    mtp_expert_matvec,
    mtp_weighted_route_reduce,
)
from runtime.model.flashnext.spec import (  # noqa: E402
    FlashNextSpecEngine,
    FlashNextVerifyGraph,
)


def _initialize(moe: BF16MoE) -> None:
    with torch.no_grad():
        for parameter in moe.parameters():
            parameter.normal_(mean=0.0, std=0.02)


def test_bf16_moe_cpu_uses_indexed_reference() -> None:
    torch.manual_seed(7)
    moe = BF16MoE(hidden=16, num_experts=8, inter=12, top_k=2).to(torch.bfloat16)
    _initialize(moe)
    x = torch.randn(5, 16, dtype=torch.bfloat16)

    got = moe(x)

    logits = moe.gate(x)
    weights, ids = torch.topk(torch.softmax(logits.float(), dim=-1), 2, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    expert_out = moe._indexed_experts(x, ids)
    routed = (expert_out.float() * weights.unsqueeze(-1)).sum(dim=1).to(x.dtype)
    shared = moe.shared_down(
        torch.nn.functional.silu(moe.shared_gate(x)) * moe.shared_up(x)
    )
    expected = routed + torch.sigmoid(moe.shared_gate_lin(x)) * shared

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_fp8_moe_cpu_fallback_dequantizes_per_output_row() -> None:
    torch.manual_seed(9)
    dtype = torch.float8_e4m3fn
    moe = BF16MoE(
        hidden=16,
        num_experts=8,
        inter=12,
        top_k=2,
        expert_dtype=dtype,
    )
    with torch.no_grad():
        for parameter in moe.parameters():
            parameter.normal_(mean=0.0, std=0.02)
        gate_up = torch.randn(8, 24, 16)
        down = torch.randn(8, 16, 12)
        gate_up_q, gate_up_scale = quantize_mtp_expert_weight(gate_up, dtype=dtype)
        down_q, down_scale = quantize_mtp_expert_weight(down, dtype=dtype)
        moe.set_fp8_expert_weights(
            gate_up_q,
            gate_up_scale,
            down_q,
            down_scale,
        )
        for parameter in moe.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)
    x = torch.randn(5, 16, dtype=torch.bfloat16)
    got = moe(x)

    logits = moe.gate(x)
    weights, ids = torch.topk(torch.softmax(logits.float(), dim=-1), 2, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    expert_out = moe._indexed_experts(x, ids)
    routed = (expert_out.float() * weights.unsqueeze(-1)).sum(dim=1).to(x.dtype)
    shared = moe.shared_down(
        torch.nn.functional.silu(moe.shared_gate(x)) * moe.shared_up(x)
    )
    expected = routed + torch.sigmoid(moe.shared_gate_lin(x)) * shared

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def _make_fp8_moe_with_weights() -> BF16MoE:
    """Build a small quantized MTP MoE for storage-contract tests."""
    dtype = torch.float8_e4m3fn
    torch.manual_seed(101)
    moe = BF16MoE(
        hidden=16,
        num_experts=4,
        inter=12,
        top_k=2,
        expert_dtype=dtype,
    )
    with torch.no_grad():
        for parameter in moe.parameters():
            parameter.normal_(mean=0.0, std=0.02)
        gate_up_q, gate_up_scale = quantize_mtp_expert_weight(
            torch.randn(4, 24, 16), dtype=dtype
        )
        down_q, down_scale = quantize_mtp_expert_weight(
            torch.randn(4, 16, 12), dtype=dtype
        )
        moe.set_fp8_expert_weights(
            gate_up_q,
            gate_up_scale,
            down_q,
            down_scale,
        )
    return moe


def test_fp8_moe_state_dict_roundtrip_preserves_expert_payload() -> None:
    moe = _make_fp8_moe_with_weights()
    state = moe.state_dict()

    for name in (
        "gate_up_proj",
        "gate_up_proj_scale",
        "down_proj",
        "down_proj_scale",
    ):
        assert name in state

    restored = BF16MoE(
        hidden=16,
        num_experts=4,
        inter=12,
        top_k=2,
        expert_dtype=torch.float8_e4m3fn,
    )
    restored.load_state_dict(state)
    for name in (
        "gate_up_proj",
        "gate_up_proj_scale",
        "down_proj",
        "down_proj_scale",
    ):
        torch.testing.assert_close(getattr(restored, name), getattr(moe, name))


def test_fp8_moe_loader_cast_keeps_compact_expert_storage() -> None:
    moe = _make_fp8_moe_with_weights()
    moe.to("cpu")
    with torch.no_grad():
        for parameter in moe.parameters():
            parameter.data = parameter.data.to(torch.bfloat16)

    assert moe.gate.weight.dtype == torch.bfloat16
    assert moe.shared_gate.weight.dtype == torch.bfloat16
    assert moe.gate_up_proj.dtype == torch.float8_e4m3fn
    assert moe.down_proj.dtype == torch.float8_e4m3fn
    assert moe.gate_up_proj_scale.dtype == torch.float32
    assert moe.down_proj_scale.dtype == torch.float32


def test_bf16_moe_graph_direct_batches_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(17)
    moe = BF16MoE(hidden=16, num_experts=8, inter=12, top_k=2).to(torch.bfloat16)
    _initialize(moe)
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    calls = {}

    def _batched_stub(
        hidden: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        gate_up: torch.Tensor,
        down: torch.Tensor,
    ) -> torch.Tensor:
        calls["hidden_shape"] = tuple(hidden.shape)
        calls["ids_shape"] = tuple(ids.shape)
        calls["weights_shape"] = tuple(weights.shape)
        calls["gate_up_shape"] = tuple(gate_up.shape)
        calls["down_shape"] = tuple(down.shape)
        return torch.zeros(hidden.shape[0], hidden.shape[1], dtype=hidden.dtype)

    monkeypatch.setattr("runtime.model.flashnext.mtp.mtp_expert_matvec", _batched_stub)

    got = moe(x, graph_direct=True)

    assert calls["hidden_shape"] == (3, 16)
    assert calls["ids_shape"] == (3, 2)
    assert calls["weights_shape"] == (3, 2)
    assert calls["gate_up_shape"] == tuple(moe.gate_up_proj.shape)
    assert calls["down_shape"] == tuple(moe.down_proj.shape)
    assert got.shape == (3, 16)


def test_mtp_weighted_route_reduce_preserves_route_order() -> None:
    rows, top_k, hidden = 3, 2, 5
    original = torch.arange(rows * top_k * hidden, dtype=torch.bfloat16).view(
        rows * top_k, hidden
    )
    order = torch.tensor([3, 0, 5, 2, 1, 4], dtype=torch.long)
    sorted_out = original.index_select(0, order)
    weights = torch.tensor(
        [[0.25, 0.75], [0.6, 0.4], [0.1, 0.9]],
        dtype=torch.float32,
    )

    got = mtp_weighted_route_reduce(sorted_out, order, weights)
    expected = (original.view(rows, top_k, hidden).float() * weights.unsqueeze(-1)).sum(
        dim=1
    ).to(torch.bfloat16)

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", [(1, 2, 17), (7, 4, 64), (19, 10, 257)])
def test_mtp_weighted_route_reduce_cuda_matches_reference(shape) -> None:
    rows, top_k, hidden = shape
    generator = torch.Generator(device="cuda").manual_seed(rows * 1000 + hidden)
    original = torch.randn(
        rows * top_k,
        hidden,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    route_ids = torch.randint(
        0,
        max(3, rows),
        (rows * top_k,),
        generator=generator,
        device="cuda",
    )
    order = torch.argsort(route_ids, stable=True)
    sorted_out = original.index_select(0, order)
    weights = torch.rand(
        rows,
        top_k,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    weights /= weights.sum(dim=-1, keepdim=True)

    got = mtp_weighted_route_reduce(sorted_out, order, weights)
    expected = (original.view(rows, top_k, hidden).float() * weights.unsqueeze(-1)).sum(
        dim=1
    ).to(torch.bfloat16)

    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_mtp_sparse_reuse_tensor_length_updates_eager_mirror() -> None:
    mtp = object.__new__(FlashNextMTP)
    sparse = SimpleNamespace(
        shared_indices=torch.empty(1, 2, dtype=torch.long),
        shared_valid=torch.empty(1, 2, dtype=torch.bool),
        shared_captured_len=torch.zeros(1, dtype=torch.long),
    )
    sess = SimpleNamespace(
        sparse_graph_buffers=sparse,
        shared_sparse_captured_len=0,
    )
    idx = torch.tensor([[2, 5]], dtype=torch.long)
    valid = torch.tensor([[True, False]])

    mtp._store_sparse_reuse(sess, idx, valid, torch.tensor([7], dtype=torch.long))

    assert sess.shared_sparse_captured_len == 7
    assert _shared_sparse_captured_len(sess) == 7
    torch.testing.assert_close(sess.shared_sparse_indices, idx)
    torch.testing.assert_close(sess.shared_sparse_valid, valid)


def test_mtp_sparse_reuse_prefers_static_buffers_after_graph_replay() -> None:
    static_indices = torch.tensor([[2, 5]], dtype=torch.long)
    static_valid = torch.tensor([[True, False]])
    sess = SimpleNamespace(
        shared_sparse_indices=torch.tensor([[99, 99]], dtype=torch.long),
        shared_sparse_valid=torch.tensor([[False, False]]),
        sparse_graph_buffers=SimpleNamespace(
            shared_indices=static_indices,
            shared_valid=static_valid,
        ),
    )

    indices, valid = _shared_sparse_reuse(sess)

    assert indices is static_indices
    assert valid is static_valid


def test_mtp_cache_guards_reject_fixed_pool_overflow() -> None:
    engine = object.__new__(FlashNextSpecEngine)
    engine.max_seq = 8
    engine.mtp_session = SimpleNamespace(
        mtp_k_pool=torch.zeros(8, 1, 1),
        mtp_v_pool=torch.zeros(7, 1, 1),
        mtp_idx_k_pool=torch.zeros(8, 1),
    )
    with pytest.raises(ValueError, match="exceeds cache capacity"):
        engine._validate_mtp_range(6, 2, "MTP sync")

    verify = object.__new__(FlashNextVerifyGraph)
    verify.max_seq = 8
    verify.qo_len = 4
    with pytest.raises(ValueError, match="exceeds target cache capacity"):
        verify._validate_past_len(5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires grouped GEMM on CUDA")
def test_bf16_moe_grouped_experts_preserve_routes() -> None:
    torch.manual_seed(11)
    moe = BF16MoE(hidden=64, num_experts=16, inter=32, top_k=4).cuda().to(torch.bfloat16)
    _initialize(moe)
    x = torch.randn(9, 64, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, 16, (9, 4), device="cuda")

    got = moe._grouped_experts(x, ids)
    expected = moe._indexed_experts(x, ids)

    assert got.shape == expected.shape
    assert torch.isfinite(got).all()
    # Grouped GEMM may choose a different BF16 reduction order than einsum,
    # but routing/order restoration must retain close expert outputs.
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=1.0)

    weights = torch.rand(9, 4, device="cuda")
    weights /= weights.sum(dim=-1, keepdim=True)
    routed = moe._grouped_experts(x, ids, routing_weights=weights)
    expected_routed = (expected.float() * weights.unsqueeze(-1)).sum(dim=1).to(x.dtype)
    torch.testing.assert_close(routed, expected_routed, rtol=2e-2, atol=1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mtp_direct_expert_matvec_is_graph_safe_and_matches_grouped() -> None:
    torch.manual_seed(19)
    moe = BF16MoE(hidden=64, num_experts=16, inter=32, top_k=4).cuda().to(torch.bfloat16)
    _initialize(moe)
    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    ids = torch.stack(
        [
            torch.randperm(16, device="cuda")[:4],
            torch.randperm(16, device="cuda")[:4],
            torch.randperm(16, device="cuda")[:4],
        ]
    )
    weights = torch.rand(3, 4, device="cuda")
    weights /= weights.sum(dim=-1, keepdim=True)

    expected_experts = moe._grouped_experts(x, ids)
    expected = (
        expected_experts.float() * weights.unsqueeze(-1)
    ).sum(dim=1).to(torch.bfloat16)
    got = mtp_expert_matvec(
        x,
        ids,
        weights,
        moe.gate_up_proj,
        moe.down_proj,
    )
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)

    # Capture the production multi-row shape; replay must consume updated
    # device inputs without falling back to one graph launch per row.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = mtp_expert_matvec(
            x,
            ids,
            weights,
            moe.gate_up_proj,
            moe.down_proj,
        )
    before = replayed.clone()
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(replayed, before)


def test_mtp_forward_sparse_graph_uses_fixed_sparse_buffers() -> None:
    class _Mixer:
        def mix(self, value):
            return value, value

        def combine(self, value, residuals):
            del residuals
            return value

    class _Indexer:
        compress_ratio = 4
        head_dim = 3
        block_topk = 2

        def __init__(self) -> None:
            self.fixed_score_called = False
            self.fixed_select_called = False
            self.fixed_gather_called = False
            self.fixed_reuse_called = False
            self.group_offsets = torch.arange(self.compress_ratio, dtype=torch.long)

        def project_qk(self, mixed, positions):
            rows = positions.shape[0]
            qi = torch.ones(rows, 1, self.head_dim, dtype=torch.bfloat16)
            ki = torch.arange(rows * self.head_dim, dtype=torch.bfloat16).view(rows, self.head_dim)
            return qi, ki

        def pool_key_groups_at_positions(self, groups, group_positions):
            del group_positions
            return groups.float().mean(dim=1).to(torch.bfloat16)

        def update_index_cache_fixed(self, raw_cache, pooled_cache, keys, positions):
            raw_cache.index_copy_(0, positions.remainder(raw_cache.shape[0]), keys)
            group_positions = positions // self.compress_ratio * self.compress_ratio
            token_positions = (
                group_positions.unsqueeze(1) + self.group_offsets.unsqueeze(0)
            ).remainder(raw_cache.shape[0])
            pooled_cache.index_copy_(
                0,
                group_positions // self.compress_ratio,
                self.pool_key_groups_at_positions(
                    raw_cache[token_positions],
                    group_positions,
                ),
            )

        def score_blocks_fixed(self, q, pooled_k, row_block_ends, *, out, column_ids):
            del q, pooled_k, column_ids
            self.fixed_score_called = True
            out.fill_(-float("inf"))
            for row, end in enumerate(row_block_ends.tolist()):
                out[row, : max(int(end), 1)] = torch.arange(max(int(end), 1), dtype=out.dtype)
            return out

        def select_blocks_fixed(self, logits, row_block_ends, *, out):
            del logits
            self.fixed_select_called = True
            out.fill_(-1)
            for row, end in enumerate(row_block_ends.tolist()):
                if end > 0:
                    out[row, 0] = 0
            return out

        def batch_decode_gather_indices_fixed(
            self,
            block_indices,
            positions,
            pad_to,
            *,
            out_tokens,
            out_valid,
            tail_tokens,
        ):
            del block_indices, positions, tail_tokens
            self.fixed_gather_called = True
            out_tokens.zero_()
            out_valid.zero_()
            out_tokens[:, :4] = torch.tensor([0, 1, 2, 3], dtype=torch.long)
            out_valid[:, :4] = True
            assert out_tokens.shape[1] == pad_to
            return out_tokens, out_valid

        def batch_decode_reuse_indices_fixed(
            self,
            shared_indices,
            shared_valid,
            positions,
            captured_len,
            *,
            out_tokens,
            out_valid,
            tail_tokens,
        ):
            del positions, captured_len, tail_tokens
            self.fixed_reuse_called = True
            out_tokens.zero_()
            out_valid.zero_()
            out_tokens[:, : shared_indices.shape[1]] = shared_indices.expand(
                out_tokens.shape[0], -1
            )
            out_valid[:, : shared_valid.shape[1]] = shared_valid.expand(
                out_valid.shape[0], -1
            )
            return out_tokens, out_valid

    class _AttnProject:
        num_kv_heads = 1
        head_dim = 3

        def project(self, mixed, positions):
            rows = positions.shape[0]
            q = torch.ones(rows, 1, 3, dtype=torch.bfloat16)
            k = torch.ones(rows, 1, 3, dtype=torch.bfloat16)
            v = torch.full((rows, 1, 3), 2, dtype=torch.bfloat16)
            gate = torch.zeros(rows, 1, 3, dtype=torch.bfloat16)
            return q, k, v, gate

    mtp = object.__new__(FlashNextMTP)
    mtp._dtype_hooks = True
    mtp.fc_embedding = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16))
    mtp.fc_hidden = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16))
    mtp.fuse = lambda embeds, hc_hidden: embeds + hc_hidden
    mtp.attn_hc = _Mixer()
    mtp.mlp_hc = _Mixer()
    mtp.hyper_connection_mixer = _Mixer()
    mtp.indexer = _Indexer()
    mtp.attn = _AttnProject()
    mtp.mlp = lambda x, graph_direct=False: x + (1 if graph_direct else 0)
    mtp.qsa_pad = 9
    captured = {}
    mtp.decode_attn = lambda q, gate, k_pool, v_pool, idx, valid, selected_counts=None: (
        captured.setdefault(
            "args",
            (q.clone(), gate.clone(), idx.clone(), valid.clone()),
        )
        and torch.zeros(q.shape[0], 4, dtype=torch.bfloat16)
    )

    sparse_buffers = SimpleNamespace(
        pooled_source=torch.empty(8, 3, dtype=torch.bfloat16),
        pooled_positions=torch.arange(0, 8, 4, dtype=torch.long),
        pooled_columns=torch.arange(2, dtype=torch.long).unsqueeze(0),
        shared_indices=torch.empty(1, 9, dtype=torch.long),
        shared_valid=torch.empty(1, 9, dtype=torch.bool),
        shared_captured_len=torch.zeros(1, dtype=torch.long),
        row_block_ends=torch.empty(2, dtype=torch.long),
        block_logits=torch.empty(2, 2, dtype=torch.float32),
        block_indices=torch.empty(2, 2, dtype=torch.long),
        gather_indices=torch.empty(2, 9, dtype=torch.long),
        gather_valid=torch.empty(2, 9, dtype=torch.bool),
        reuse_indices=torch.empty(2, 9, dtype=torch.long),
        reuse_valid=torch.empty(2, 9, dtype=torch.bool),
        tail_indices=torch.empty(2, 3, dtype=torch.long),
        reuse_tail_indices=torch.empty(2, 0, dtype=torch.long),
    )
    sess = SimpleNamespace(
        mtp_k_pool=torch.zeros(8, 1, 3, dtype=torch.bfloat16),
        mtp_v_pool=torch.zeros(8, 1, 3, dtype=torch.bfloat16),
        mtp_idx_k_pool=torch.zeros(8, 3, dtype=torch.bfloat16),
        mtp_pooled_k_pool=torch.zeros(2, 3, dtype=torch.bfloat16),
        sparse_graph_buffers=sparse_buffers,
    )
    embeds = torch.ones(2, 4, dtype=torch.bfloat16)
    hc_hidden = torch.ones(2, 4, dtype=torch.bfloat16)
    positions = torch.tensor([4, 5], dtype=torch.long)

    mixed, own_hidden = FlashNextMTP.forward(
        mtp,
        embeds,
        hc_hidden,
        positions,
        sess,
        graph_sparse_capacity=8,
    )

    assert mtp.indexer.fixed_score_called
    assert mtp.indexer.fixed_select_called
    assert mtp.indexer.fixed_gather_called
    assert "args" in captured
    assert captured["args"][2].shape == (2, 9)
    assert captured["args"][3].shape == (2, 9)
    assert mixed.shape == own_hidden.shape == (2, 4)


def test_mtp_forward_sparse_graph_reuses_captured_sync_row_without_rescore() -> None:
    class _Mixer:
        def mix(self, value):
            return value, value

        def combine(self, value, residuals):
            del residuals
            return value

    class _Indexer:
        compress_ratio = 4
        head_dim = 3
        block_topk = 2

        def __init__(self) -> None:
            self.fixed_score_called = False
            self.fixed_select_called = False
            self.fixed_gather_called = False
            self.fixed_reuse_called = False
            self.group_offsets = torch.arange(self.compress_ratio, dtype=torch.long)

        def project_qk(self, mixed, positions):
            del mixed, positions
            raise AssertionError("reuse path must not project/score/select")

        def pool_key_groups_at_positions(self, groups, group_positions):
            del groups, group_positions
            raise AssertionError("reuse path must not pool groups")

        def score_blocks_fixed(self, q, pooled_k, row_block_ends, *, out, column_ids):
            del q, pooled_k, row_block_ends, out, column_ids
            self.fixed_score_called = True
            raise AssertionError("reuse path must not rescore blocks")

        def select_blocks_fixed(self, logits, row_block_ends, *, out):
            del logits, row_block_ends, out
            self.fixed_select_called = True
            raise AssertionError("reuse path must not reselect blocks")

        def batch_decode_gather_indices_fixed(
            self,
            block_indices,
            positions,
            pad_to,
            *,
            out_tokens,
            out_valid,
            tail_tokens,
        ):
            del block_indices, positions, pad_to, out_tokens, out_valid, tail_tokens
            self.fixed_gather_called = True
            raise AssertionError("reuse path must not regather sparse blocks")

        def batch_decode_reuse_indices_fixed(
            self,
            shared_indices,
            shared_valid,
            positions,
            captured_len,
            *,
            out_tokens,
            out_valid,
            tail_tokens,
        ):
            self.fixed_reuse_called = True
            assert positions.tolist() == [23472]
            assert captured_len.tolist() == [23472]
            out_tokens.zero_()
            out_valid.zero_()
            out_tokens[:, : shared_indices.shape[1]] = shared_indices
            out_valid[:, : shared_valid.shape[1]] = shared_valid
            return out_tokens, out_valid

    class _AttnProject:
        num_kv_heads = 1
        head_dim = 3

        def project(self, mixed, positions):
            rows = positions.shape[0]
            q = torch.ones(rows, 1, 3, dtype=torch.bfloat16)
            k = torch.ones(rows, 1, 3, dtype=torch.bfloat16)
            v = torch.full((rows, 1, 3), 2, dtype=torch.bfloat16)
            gate = torch.zeros(rows, 1, 3, dtype=torch.bfloat16)
            return q, k, v, gate

    mtp = object.__new__(FlashNextMTP)
    mtp._dtype_hooks = True
    mtp.fc_embedding = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16))
    mtp.fc_hidden = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16))
    mtp.fuse = lambda embeds, hc_hidden: embeds + hc_hidden
    mtp.attn_hc = _Mixer()
    mtp.mlp_hc = _Mixer()
    mtp.hyper_connection_mixer = _Mixer()
    mtp.indexer = _Indexer()
    mtp.attn = _AttnProject()
    mtp.mlp = lambda x, graph_direct=False: x + (1 if graph_direct else 0)
    mtp.qsa_pad = 9
    captured = {}
    mtp.decode_attn = lambda q, gate, k_pool, v_pool, idx, valid, selected_counts=None: (
        captured.setdefault(
            "args",
            (q.clone(), gate.clone(), idx.clone(), valid.clone()),
        )
        and torch.zeros(q.shape[0], 4, dtype=torch.bfloat16)
    )

    sparse_buffers = SimpleNamespace(
        pooled_source=torch.empty(32768, 3, dtype=torch.bfloat16),
        pooled_positions=torch.arange(0, 32768, 4, dtype=torch.long),
        pooled_columns=torch.arange(8192, dtype=torch.long).unsqueeze(0),
        shared_indices=torch.tensor([[0, 4, 8, 12, 16, 20, 24, 28, 31]], dtype=torch.long),
        shared_valid=torch.tensor([[True, True, True, True, True, True, True, True, True]]),
        shared_captured_len=torch.tensor([23472], dtype=torch.long),
        row_block_ends=torch.empty(1, dtype=torch.long),
        block_logits=torch.empty(1, 8192, dtype=torch.float32),
        block_indices=torch.empty(1, 2, dtype=torch.long),
        gather_indices=torch.empty(1, 9, dtype=torch.long),
        gather_valid=torch.empty(1, 9, dtype=torch.bool),
        reuse_indices=torch.empty(1, 9, dtype=torch.long),
        reuse_valid=torch.empty(1, 9, dtype=torch.bool),
        tail_indices=torch.empty(1, 3, dtype=torch.long),
        reuse_tail_indices=torch.empty(1, 0, dtype=torch.long),
    )
    sess = SimpleNamespace(
        mtp_k_pool=torch.zeros(32768, 1, 3, dtype=torch.bfloat16),
        mtp_v_pool=torch.zeros(32768, 1, 3, dtype=torch.bfloat16),
        mtp_idx_k_pool=torch.zeros(32768, 3, dtype=torch.bfloat16),
        mtp_pooled_k_pool=torch.zeros(8192, 3, dtype=torch.bfloat16),
        sparse_graph_buffers=sparse_buffers,
    )

    mixed, own_hidden = FlashNextMTP.forward(
        mtp,
        torch.ones(1, 4, dtype=torch.bfloat16),
        torch.ones(1, 4, dtype=torch.bfloat16),
        torch.tensor([23472], dtype=torch.long),
        sess,
        reuse_sparse_indices=True,
        graph_sparse_capacity=32768,
    )

    assert mtp.indexer.fixed_reuse_called
    assert not mtp.indexer.fixed_score_called
    assert not mtp.indexer.fixed_select_called
    assert not mtp.indexer.fixed_gather_called
    torch.testing.assert_close(captured["args"][2], sparse_buffers.shared_indices, rtol=0, atol=0)
    torch.testing.assert_close(captured["args"][3], sparse_buffers.shared_valid, rtol=0, atol=0)
    assert mixed.shape == own_hidden.shape == (1, 4)


def test_mtp_forward_capture_sparse_indices_syncs_static_reuse_buffers() -> None:
    class _Mixer:
        def mix(self, value):
            return value, value

        def combine(self, value, residuals):
            del residuals
            return value

    class _Indexer:
        compress_ratio = 4
        head_dim = 3
        block_topk = 2

        def __init__(self) -> None:
            self.group_offsets = torch.arange(self.compress_ratio, dtype=torch.long)

        def project_qk(self, mixed, positions):
            rows = positions.shape[0]
            qi = torch.ones(rows, 1, self.head_dim, dtype=torch.bfloat16)
            ki = torch.arange(rows * self.head_dim, dtype=torch.bfloat16).view(rows, self.head_dim)
            return qi, ki

        def pool_keys(self, idx_pool):
            del idx_pool
            return torch.ones(5870, self.head_dim, dtype=torch.bfloat16)

        def update_index_cache_eager(self, raw_cache, pooled_cache, keys, *, start):
            positions = torch.arange(start, start + keys.shape[0])
            raw_cache.index_copy_(0, positions.remainder(raw_cache.shape[0]), keys)
            pooled_cache.fill_(1)

        def score_blocks(self, qi, pooled, ends):
            del qi, pooled, ends
            return torch.zeros(1, 2, dtype=torch.float32)

        def select_blocks(self, scores, ends):
            del scores, ends
            return torch.tensor([[0, -1]], dtype=torch.long)

        def batch_decode_gather_indices(self, blocks, pos, pad_to):
            del blocks, pos, pad_to
            idx = torch.tensor([[0, 4, 8, 12, 16, 20, 24, 28, 31]], dtype=torch.long)
            valid = torch.tensor([[True, True, True, True, True, True, True, True, True]])
            return idx, valid

    class _AttnProject:
        num_kv_heads = 1
        head_dim = 3

        def project(self, mixed, positions):
            rows = positions.shape[0]
            q = torch.ones(rows, 1, 3, dtype=torch.bfloat16)
            k = torch.ones(rows, 1, 3, dtype=torch.bfloat16)
            v = torch.full((rows, 1, 3), 2, dtype=torch.bfloat16)
            gate = torch.zeros(rows, 1, 3, dtype=torch.bfloat16)
            return q, k, v, gate

    mtp = object.__new__(FlashNextMTP)
    mtp._dtype_hooks = True
    mtp.fc_embedding = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16))
    mtp.fc_hidden = SimpleNamespace(weight=torch.empty(1, dtype=torch.bfloat16))
    mtp.fuse = lambda embeds, hc_hidden: embeds + hc_hidden
    mtp.attn_hc = _Mixer()
    mtp.mlp_hc = _Mixer()
    mtp.hyper_connection_mixer = _Mixer()
    mtp.indexer = _Indexer()
    mtp.attn = _AttnProject()
    mtp.mlp = lambda x, graph_direct=False: x + (1 if graph_direct else 0)
    mtp.qsa_pad = 9
    mtp.decode_attn = lambda q, gate, k_pool, v_pool, idx, valid, selected_counts=None: (
        torch.zeros(q.shape[0], 4, dtype=torch.bfloat16)
    )

    sparse_buffers = SimpleNamespace(
        pooled_source=torch.empty(32768, 3, dtype=torch.bfloat16),
        pooled_positions=torch.arange(0, 32768, 4, dtype=torch.long),
        pooled_columns=torch.arange(8192, dtype=torch.long).unsqueeze(0),
        shared_indices=torch.zeros(1, 9, dtype=torch.long),
        shared_valid=torch.zeros(1, 9, dtype=torch.bool),
        shared_captured_len=torch.zeros(1, dtype=torch.long),
        row_block_ends=torch.empty(1, dtype=torch.long),
        block_logits=torch.empty(1, 8192, dtype=torch.float32),
        block_indices=torch.empty(1, 2, dtype=torch.long),
        gather_indices=torch.empty(1, 9, dtype=torch.long),
        gather_valid=torch.empty(1, 9, dtype=torch.bool),
        reuse_indices=torch.empty(1, 9, dtype=torch.long),
        reuse_valid=torch.empty(1, 9, dtype=torch.bool),
        tail_indices=torch.empty(1, 3, dtype=torch.long),
        reuse_tail_indices=torch.empty(1, 0, dtype=torch.long),
    )
    sess = SimpleNamespace(
        mtp_k_pool=torch.zeros(32768, 1, 3, dtype=torch.bfloat16),
        mtp_v_pool=torch.zeros(32768, 1, 3, dtype=torch.bfloat16),
        mtp_idx_k_pool=torch.zeros(32768, 3, dtype=torch.bfloat16),
        mtp_pooled_k_pool=torch.zeros(8192, 3, dtype=torch.bfloat16),
        sparse_graph_buffers=sparse_buffers,
        shared_sparse_indices=None,
        shared_sparse_valid=None,
        shared_sparse_captured_len=0,
    )
    sess.mtp_idx_k_pool[:23471].fill_(7)

    FlashNextMTP.forward(
        mtp,
        torch.ones(1, 4, dtype=torch.bfloat16),
        torch.ones(1, 4, dtype=torch.bfloat16),
        torch.tensor([23471], dtype=torch.long),
        sess,
        capture_sparse_indices=True,
    )

    assert sess.shared_sparse_captured_len == 23472
    torch.testing.assert_close(sparse_buffers.shared_captured_len, torch.tensor([23472]))
    torch.testing.assert_close(
        sparse_buffers.shared_indices,
        sess.shared_sparse_indices,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        sparse_buffers.shared_valid,
        sess.shared_sparse_valid,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        sess.mtp_idx_k_pool[23471],
        torch.tensor([0, 1, 2], dtype=torch.bfloat16),
        rtol=0,
        atol=0,
    )
