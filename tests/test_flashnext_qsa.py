"""QSA indexer and fused fixed-pool key pooling tests."""

from __future__ import annotations

import pathlib

import pytest

torch = pytest.importorskip("torch")

CKPT = pathlib.Path("/home/bot/models/Qwen3.8-Flash-Next-NVFP4-RadixArk")

from runtime.model.flashnext.qsa import (  # noqa: E402
    QSAIndexer,
    load_qsa_attention,
    load_qsa_indexer,
    qsa_cache_index_copy_,
    qsa_index_cache_rows,
    qsa_kv_cache_dtype,
    quantize_qsa_kv,
)


@pytest.fixture(scope="module")
def indexer() -> QSAIndexer:
    if not CKPT.is_dir():
        pytest.skip("RadixArk checkpoint not downloaded")
    return load_qsa_indexer(CKPT, layer_idx=3)


def test_int8_qsa_cache_uses_row_scaled_symmetric_quantization(monkeypatch):
    monkeypatch.setenv("QSR_FLASHNEXT_QSA_KV_DTYPE", "int8")
    assert qsa_kv_cache_dtype() == torch.int8

    torch.manual_seed(41)
    values = torch.randn(7, 2, 32, dtype=torch.bfloat16) * 0.3
    destination = torch.empty_like(values, dtype=torch.int8)
    scales = torch.empty(7, 2, dtype=torch.float16)
    quantize_qsa_kv(values, destination, scales)
    reconstructed = destination.float() * scales.float().unsqueeze(-1)
    relative_error = (
        (reconstructed - values.float()).abs().mean()
        / values.float().abs().mean()
    )

    assert torch.isfinite(reconstructed).all()
    assert float(relative_error) < 0.01
    assert int(destination.abs().max()) <= 127


def test_qsa_cache_defaults_to_fp8_for_flashnext(monkeypatch):
    monkeypatch.delenv("QSR_FLASHNEXT_QSA_KV_DTYPE", raising=False)
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None:
        pytest.skip("torch.float8_e4m3fn is unavailable")
    assert qsa_kv_cache_dtype() == fp8


def test_qsa_cache_index_copy_supports_fp8():
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None:
        pytest.skip("torch.float8_e4m3fn is unavailable")
    destination = torch.zeros(5, 2, 4, dtype=fp8)
    source = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4).to(fp8)
    positions = torch.tensor([1, 3], dtype=torch.long)

    qsa_cache_index_copy_(destination, positions, source)

    torch.testing.assert_close(destination[positions].float(), source.float())


def test_project_qk_shapes_and_finite(indexer):
    torch.manual_seed(0)
    hidden = torch.randn(8, 2560, dtype=torch.bfloat16) * 0.02
    pos = torch.arange(8)
    q, k = indexer.project_qk(hidden, pos)
    assert tuple(q.shape) == (8, 4, 128)
    assert tuple(k.shape) == (8, 128)
    assert torch.isfinite(q.float()).all()
    assert torch.isfinite(k.float()).all()


def test_rope_at_position_zero_is_identity(indexer):
    torch.manual_seed(1)
    hidden = torch.randn(2, 2560, dtype=torch.bfloat16) * 0.02
    q, k = indexer.project_qk(hidden, torch.zeros(2, dtype=torch.long))
    qk = indexer.index_qk_proj(hidden)
    q_raw = qk[:, : 4 * 128].view(-1, 4, 128)
    k_raw = qk[:, 4 * 128 :].view(-1, 128)
    q_normed = indexer._gemma_norm(q_raw, indexer.q_layernorm, 1e-6)
    assert torch.allclose(q.float(), q_normed.float(), atol=1e-5)
    assert torch.equal(k, k_raw)


def test_mrope_interleaving_matches_scalar_rope_when_axes_coincide():
    indexer = QSAIndexer(
        hidden_size=16,
        n_heads=1,
        kv_heads=1,
        head_dim=8,
        rotary_dim=8,
        compress_ratio=2,
        block_topk=2,
        mrope_section=(1, 1, 2),
        mrope_interleaved=True,
    )
    torch.manual_seed(101)
    x = torch.randn(3, 1, 8, dtype=torch.bfloat16)
    scalar = torch.arange(3, dtype=torch.long)
    matrix = scalar.view(1, -1).expand(3, -1)
    torch.testing.assert_close(indexer._rope(x, scalar), indexer._rope(x, matrix))

    distinct = matrix.clone()
    distinct[1] += 7
    assert not torch.equal(indexer._rope(x, matrix), indexer._rope(x, distinct))


def test_mrope_group_pool_uses_first_member_coordinates():
    indexer = QSAIndexer(
        hidden_size=16,
        n_heads=1,
        kv_heads=1,
        head_dim=8,
        rotary_dim=8,
        compress_ratio=2,
        block_topk=2,
        mrope_section=(1, 1, 2),
        mrope_interleaved=True,
    )
    keys = torch.arange(32, dtype=torch.float32).view(4, 8).to(torch.bfloat16)
    rope = torch.stack(
        [
            torch.arange(4),
            torch.arange(4) * 2,
            torch.arange(4) * 3,
        ]
    )
    got = indexer.pool_keys(keys, group_positions=rope)
    mean = keys.float().view(2, 2, 8).mean(1).to(torch.bfloat16)
    expected = indexer._gemma_norm(mean, indexer.k_layernorm, 1e-6)
    expected = indexer._rope(expected, rope[:, ::2])
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_pool_and_select_respect_causality(indexer):
    torch.manual_seed(2)
    seq = 2048  # exactly 512 blocks of 4 -> matches block_topk
    hidden = torch.randn(seq, 2560, dtype=torch.bfloat16) * 0.02
    pos = torch.arange(seq)
    q, k = indexer.project_qk(hidden, pos)
    pooled = indexer.pool_keys(k)
    assert tuple(pooled.shape) == (seq // 4, 128)
    # complete blocks fully below the causal frontier: block b covers
    # tokens [4b, 4b+3], visible iff 4b+3 <= i
    ends = torch.tensor([max(0, (i - 3) // 4 + 1) for i in range(seq)])
    assert int(ends[-1]) == seq // 4
    logits = indexer.score_blocks(q, pooled, ends)
    assert tuple(logits.shape) == (seq, seq // 4)
    # mask: blocks at/after the causal end are -inf
    mid = 100
    assert torch.isinf(logits[mid, ends[mid] :]).all()
    assert logits[mid, 0].isfinite()
    # last row sees all 512 blocks; every selection is finite and in range
    sel_last = indexer.select_blocks(logits[seq - 1 : seq], ends[-1:])
    assert sel_last.shape[1] == 512
    assert int(sel_last.max()) < 512
    assert torch.isfinite(logits[seq - 1][sel_last[0]]).all()


def test_short_context_selection_uses_completed_groups_plus_tail(indexer):
    logits = torch.full((1, 1024), -float("inf"))
    logits[0, 0] = 1.0
    blocks = indexer.select_blocks(logits, torch.tensor([1]))
    assert blocks[0, 0].item() == 0
    assert (blocks[0, 1:] == -1).all()

    # Position 5 sees completed group [0,1,2,3] plus pending tail [4,5].
    indices, valid = indexer.batch_decode_gather_indices(blocks, torch.tensor([5]), pad_to=2051)
    assert indices[0, valid[0]].tolist() == [0, 1, 2, 3, 4, 5]
    assert valid[0, :6].all()
    assert not valid[0, 6:].any()

    compact, compact_valid, counts = indexer.batch_prefill_gather_indices(
        blocks, torch.tensor([5]), pad_to=2051
    )
    assert counts.tolist() == [6]
    assert compact[0, compact_valid[0]].tolist() == [0, 1, 2, 3, 4, 5]
    assert compact_valid[0, :6].all()
    assert not compact_valid[0, 6:].any()


def test_fixed_sparse_score_and_gather_match_dynamic_reference():
    indexer = QSAIndexer(
        hidden_size=32,
        n_heads=2,
        kv_heads=1,
        head_dim=8,
        rotary_dim=4,
        compress_ratio=4,
        block_topk=4,
    )
    torch.manual_seed(23)
    q = torch.randn(2, 2, 8, dtype=torch.bfloat16)
    pooled = torch.randn(8, 8, dtype=torch.bfloat16)
    ends = torch.tensor([3, 8], dtype=torch.long)
    score_out = torch.empty(2, 8, dtype=torch.float32)
    columns = torch.arange(8, dtype=torch.long).unsqueeze(0)

    fixed_scores = indexer.score_blocks_fixed(
        q,
        pooled,
        ends,
        out=score_out,
        column_ids=columns,
    )
    dynamic_scores = indexer.score_blocks(q, pooled, ends)
    torch.testing.assert_close(fixed_scores, dynamic_scores, rtol=0, atol=0)

    fixed_blocks = indexer.select_blocks_fixed(
        fixed_scores,
        ends,
        out=torch.empty(2, 4, dtype=torch.long),
    )
    dynamic_blocks = indexer.select_blocks(dynamic_scores, ends)
    torch.testing.assert_close(fixed_blocks, dynamic_blocks, rtol=0, atol=0)

    positions = torch.tensor([5, 23472], dtype=torch.long)
    fixed_idx, fixed_valid = indexer.batch_decode_gather_indices_fixed(
        fixed_blocks,
        positions,
        pad_to=17,
        out_tokens=torch.empty(2, 17, dtype=torch.long),
        out_valid=torch.empty(2, 17, dtype=torch.bool),
        tail_tokens=torch.empty(2, 3, dtype=torch.long),
    )
    dynamic_idx, dynamic_valid = indexer.batch_decode_gather_indices(
        fixed_blocks,
        positions,
        pad_to=17,
    )
    torch.testing.assert_close(fixed_idx, dynamic_idx, rtol=0, atol=0)
    torch.testing.assert_close(fixed_valid, dynamic_valid, rtol=0, atol=0)


def test_fixed_sparse_reuse_matches_dynamic_reference_at_long_context():
    indexer = QSAIndexer(
        hidden_size=32,
        n_heads=2,
        kv_heads=1,
        head_dim=8,
        rotary_dim=4,
        compress_ratio=4,
        block_topk=4,
    )
    shared_idx = torch.tensor([[0, 4, 8, 12, 23468, 23469, 23470, 23471]], dtype=torch.long)
    shared_valid = torch.tensor([[True, True, True, True, True, True, True, True]])
    positions = torch.tensor([23472, 23473], dtype=torch.long)
    captured_len = torch.tensor([23472], dtype=torch.long)

    fixed_idx, fixed_valid = indexer.batch_decode_reuse_indices_fixed(
        shared_idx,
        shared_valid,
        positions,
        captured_len,
        out_tokens=torch.empty(2, 10, dtype=torch.long),
        out_valid=torch.empty(2, 10, dtype=torch.bool),
        tail_tokens=torch.empty(2, 2, dtype=torch.long),
    )

    dynamic_idx = torch.zeros_like(fixed_idx)
    dynamic_valid = torch.zeros_like(fixed_valid)
    for row, pos in enumerate(positions.tolist()):
        tail = torch.arange(23472, pos + 1, dtype=torch.long).unsqueeze(0)
        row_idx = torch.cat([shared_idx, tail], dim=1)
        row_valid = torch.cat([shared_valid, torch.ones_like(tail, dtype=torch.bool)], dim=1)
        dynamic_idx[row, : row_idx.shape[1]] = row_idx[0]
        dynamic_valid[row, : row_valid.shape[1]] = row_valid[0]
    torch.testing.assert_close(fixed_idx, dynamic_idx, rtol=0, atol=0)
    torch.testing.assert_close(fixed_valid, dynamic_valid, rtol=0, atol=0)


def test_pool_normalizes_after_fp32_group_average(indexer):
    torch.manual_seed(6)
    raw = torch.randn(8, 128, dtype=torch.bfloat16)
    got = indexer.pool_keys(raw)
    mean = raw.float().view(2, 4, 128).mean(dim=1).to(torch.bfloat16)
    expected = indexer._gemma_norm(mean, indexer.k_layernorm, 1e-6)
    expected = indexer._rope(expected, torch.tensor([0, 4]))
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_incremental_pool_matches_full_pool(indexer):
    torch.manual_seed(8)
    keys = torch.randn(28, indexer.head_dim, dtype=torch.bfloat16)
    expected = indexer.pool_keys(keys)
    first = indexer.pool_key_groups(
        keys[:12].view(-1, indexer.compress_ratio, indexer.head_dim),
        group_start=0,
    )
    second = indexer.pool_key_groups(
        keys[12:].view(-1, indexer.compress_ratio, indexer.head_dim),
        group_start=3,
    )
    torch.testing.assert_close(torch.cat([first, second]), expected, rtol=0, atol=0)


def test_index_ring_eager_chunks_match_full_history(indexer):
    torch.manual_seed(81)
    keys = torch.randn(37, indexer.head_dim, dtype=torch.bfloat16)
    ring = torch.zeros(8, indexer.head_dim, dtype=torch.bfloat16)
    pooled = torch.zeros(10, indexer.head_dim, dtype=torch.bfloat16)

    boundaries = (0, 3, 11, 26, 37)
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        indexer.update_index_cache_eager(
            ring,
            pooled,
            keys[start:end],
            start=start,
        )

    complete = keys.shape[0] // indexer.compress_ratio
    expected = indexer.pool_keys(keys[: complete * indexer.compress_ratio])
    torch.testing.assert_close(pooled[:complete], expected, rtol=0, atol=0)
    tail_positions = torch.arange(29, 37).remainder(ring.shape[0])
    torch.testing.assert_close(ring[tail_positions], keys[29:37], rtol=0, atol=0)


def test_index_ring_fixed_updates_wrap_and_complete_groups(indexer):
    torch.manual_seed(82)
    keys = torch.randn(20, indexer.head_dim, dtype=torch.bfloat16)
    ring = torch.zeros(8, indexer.head_dim, dtype=torch.bfloat16)
    pooled = torch.zeros(5, indexer.head_dim, dtype=torch.bfloat16)

    for start in range(0, keys.shape[0], 4):
        positions = torch.arange(start, start + 4)
        indexer.update_index_cache_fixed(
            ring,
            pooled,
            keys[start : start + 4],
            positions,
        )

    torch.testing.assert_close(pooled, indexer.pool_keys(keys), rtol=0, atol=0)


def test_mrope_index_ring_pools_and_stores_axis_coordinates():
    indexer = QSAIndexer(
        hidden_size=16,
        n_heads=1,
        kv_heads=1,
        head_dim=8,
        rotary_dim=8,
        compress_ratio=2,
        block_topk=2,
        mrope_section=(1, 1, 2),
        mrope_interleaved=True,
    )
    torch.manual_seed(83)
    keys = torch.randn(10, 8, dtype=torch.bfloat16)
    rope = torch.stack(
        [
            torch.arange(10),
            torch.arange(10) * 2,
            torch.arange(10) * 3,
        ]
    )
    ring = torch.zeros(6, 8, dtype=torch.bfloat16)
    rope_ring = torch.zeros(6, 3, dtype=torch.long)
    pooled = torch.zeros(5, 8, dtype=torch.bfloat16)
    for start, end in ((0, 3), (3, 7), (7, 10)):
        indexer.update_index_cache_eager(
            ring,
            pooled,
            keys[start:end],
            start=start,
            rope_cache=rope_ring,
            rope_positions=rope[:, start:end],
        )
    expected = indexer.pool_keys(keys, group_positions=rope)
    torch.testing.assert_close(pooled[:5], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        rope_ring[torch.arange(4, 10).remainder(6)],
        rope[:, 4:10].transpose(0, 1),
        rtol=0,
        atol=0,
    )


def test_index_ring_capacity_profile_is_opt_in(monkeypatch):
    monkeypatch.delenv("QSR_FLASHNEXT_QSA_INDEX_RING", raising=False)
    assert qsa_index_cache_rows(262144, 4, fixed_rows=4) == 262144
    monkeypatch.setenv("QSR_FLASHNEXT_QSA_INDEX_RING", "1")
    assert qsa_index_cache_rows(262144, 4, fixed_rows=4) == 8


def test_scoring_uses_relu_and_head_sum(indexer):
    torch.manual_seed(3)
    q = torch.randn(1, 4, 128, dtype=torch.bfloat16)
    k = torch.randn(3, 128, dtype=torch.bfloat16)
    logits = indexer.score_blocks(q, k, torch.tensor([3]))
    scores = torch.einsum("mhd,nd->mnh", q.float(), k.float())
    expect = torch.relu(scores).sum(-1) / (128**0.5)
    assert torch.allclose(logits, expect, atol=1e-3)


@pytest.mark.skipif(not CKPT.is_dir(), reason="RadixArk checkpoint not downloaded")
def test_qsa_attention_project_and_sparse_decode():
    attn = load_qsa_attention(CKPT, layer_idx=3)
    torch.manual_seed(4)
    hidden = torch.randn(2, 2560, dtype=torch.bfloat16) * 0.02
    pos = torch.arange(2)
    q, k, v, gate = attn.project(hidden, pos)
    assert tuple(q.shape) == (2, 24, 256)
    assert tuple(k.shape) == (2, 2, 256)
    assert tuple(gate.shape) == (2, 24, 256)
    assert torch.isfinite(q.float()).all()
    # KV pool over 16 context tokens; decode step selects 6 rows
    kv_rows = 16
    k_pool = torch.randn(kv_rows, 2, 256, dtype=torch.bfloat16) * 0.02
    v_pool = torch.randn(kv_rows, 2, 256, dtype=torch.bfloat16) * 0.02
    q_step = q[1:2]
    gate_step = gate[1:2]
    selected = torch.tensor([[0, 2, 4, 8, 12, 15]])
    out = attn.sparse_decode(q_step, gate_step, k_pool, v_pool, selected)
    assert tuple(out.shape) == (1, 2560)
    assert torch.isfinite(out.float()).all()


@pytest.mark.skipif(not CKPT.is_dir(), reason="RadixArk checkpoint not downloaded")
def test_qsa_batched_decode_matches_single_rows():
    from runtime.model.flashnext.qsa import QsaDecodeAttention

    attn = load_qsa_attention(CKPT, layer_idx=3)
    decode = QsaDecodeAttention(attn, pad_to=8)
    torch.manual_seed(5)
    hidden = torch.randn(3, 2560, dtype=torch.bfloat16) * 0.02
    positions = torch.tensor([7, 11, 15])
    q, _, _, gate = attn.project(hidden, positions)
    k_pool = torch.randn(16, 2, 256, dtype=torch.bfloat16) * 0.02
    v_pool = torch.randn_like(k_pool)
    indices = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 4, 6, 8, 9, 10, 11], [1, 3, 5, 7, 9, 11, 13, 15]]
    )
    valid = torch.ones_like(indices, dtype=torch.bool)
    valid[0, -2:] = False

    batched = decode(q, gate, k_pool, v_pool, indices, valid)
    rows = torch.cat(
        [
            decode(q[i : i + 1], gate[i : i + 1], k_pool, v_pool, indices[i], valid[i])
            for i in range(3)
        ],
        dim=0,
    )
    torch.testing.assert_close(batched.float(), rows.float(), rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not CKPT.is_dir(), reason="RadixArk checkpoint not downloaded")
def test_qsa_causal_prefix_matches_fixed_width_decode():
    from runtime.model.flashnext.qsa import QsaDecodeAttention

    attn = load_qsa_attention(CKPT, layer_idx=3)
    decode = QsaDecodeAttention(attn, pad_to=9)
    torch.manual_seed(8)
    hidden = torch.randn(6, 2560, dtype=torch.bfloat16) * 0.02
    positions = torch.arange(6)
    q, k, v, gate = attn.project(hidden, positions)

    indices = torch.zeros(6, 6, dtype=torch.long)
    valid = torch.zeros(6, 6, dtype=torch.bool)
    for row in range(6):
        indices[row, : row + 1] = torch.arange(row + 1)
        valid[row, : row + 1] = True

    fixed = decode(q, gate, k, v, indices, valid)
    dense = decode.causal_prefix(q, gate, k, v, positions)

    torch.testing.assert_close(dense.float(), fixed.float(), rtol=1e-4, atol=1e-4)

    capacity = 12
    k_pool = torch.zeros(capacity, *k.shape[1:], dtype=k.dtype)
    v_pool = torch.zeros_like(k_pool)
    k_pool[: k.shape[0]].copy_(k)
    v_pool[: v.shape[0]].copy_(v)
    graph_dense = decode.causal_prefix_fixed(
        q[-1:],
        gate[-1:],
        k_pool,
        v_pool,
        positions[-1:],
        capacity,
    )
    row_dense = decode.causal_prefix(
        q[-1:],
        gate[-1:],
        k_pool,
        v_pool,
        positions[-1:],
    )
    torch.testing.assert_close(graph_dense.float(), row_dense.float(), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not CKPT.is_dir(), reason="RadixArk checkpoint not downloaded")
def test_qsa_sparse_prefill_matches_dense_prefix():
    from runtime.model.flashnext.qsa import QsaDecodeAttention

    indexer = load_qsa_indexer(CKPT, layer_idx=3)
    attn = load_qsa_attention(CKPT, layer_idx=3)
    decode = QsaDecodeAttention(attn, pad_to=2051)
    torch.manual_seed(17)
    hidden = torch.randn(9, 2560, dtype=torch.bfloat16) * 0.02
    positions = torch.arange(9)
    index_q, index_k = indexer.project_qk(hidden, positions)
    q, k, v, gate = attn.project(hidden, positions)

    sparse = decode.sparse_prefill(
        indexer,
        index_q,
        q,
        gate,
        k,
        v,
        index_k,
        positions,
        logits_workspace_bytes=128,
    )
    dense = decode.causal_prefix(q, gate, k, v, positions)

    torch.testing.assert_close(sparse.float(), dense.float(), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qsa_prefill_gather_indices_matches_cpu_reference():
    from runtime.model.flashnext.qsa_kernels import qsa_prefill_gather_indices

    indexer = QSAIndexer()
    torch.manual_seed(9)
    positions = torch.tensor([2048, 2201, 4094], dtype=torch.long)
    blocks = torch.stack(
        [torch.randperm(int(position // 4 + 1))[:512] for position in positions]
    ).to(torch.int64)
    expected = indexer.batch_prefill_gather_indices(blocks, positions, pad_to=2051)
    got = qsa_prefill_gather_indices(
        blocks.cuda(),
        positions.cuda(),
        pad_to=2051,
        compress_ratio=4,
    )
    for got_tensor, expected_tensor in zip(got, expected, strict=True):
        torch.testing.assert_close(got_tensor.cpu(), expected_tensor, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qsa_mqa_prefill_matches_torch_reference():
    from runtime.model.flashnext.qsa_kernels import qsa_mqa_prefill

    torch.manual_seed(18)
    q = torch.randn(67, 4, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(73, 128, dtype=torch.bfloat16, device="cuda")
    ends = torch.arange(7, 74, dtype=torch.int32, device="cuda")
    got = qsa_mqa_prefill(q, k, ends)
    scores = torch.einsum("mhd,nd->mnh", q.float(), k.float())
    expected = torch.relu(scores).sum(dim=-1) / (128**0.5)
    columns = torch.arange(k.shape[0], device="cuda").unsqueeze(0)
    expected.masked_fill_(columns >= ends.unsqueeze(1), -float("inf"))

    finite = torch.isfinite(expected)
    torch.testing.assert_close(got[finite], expected[finite], rtol=3e-2, atol=3e-2)
    assert torch.equal(torch.isfinite(got), finite)


def test_qsa_mqa_block_q_avoids_decode_padding():
    from runtime.model.flashnext.qsa_kernels import _qsa_mqa_block_q

    # QSA production uses four index heads.  Verify (4 rows) and decode
    # (1 row) should use the smallest valid SM120 GEMM tile instead of doing
    # the old 32-row calculation and discarding most of it.
    assert _qsa_mqa_block_q(1, 4) == 8
    assert _qsa_mqa_block_q(4, 4) == 8
    assert _qsa_mqa_block_q(9, 4) == 16
    assert _qsa_mqa_block_q(32, 4) == 32

    # Preserve the validated legacy tile for head counts that do not divide
    # the 128-wide base tile.
    assert _qsa_mqa_block_q(4, 24) == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_qsa_topk_is_score_ordered_by_default(monkeypatch):
    """The unordered SM120 set must be reranked before sparse reduction."""

    from runtime.kernels.flashnext_qsa_topk import load_native_flashnext_qsa_topk

    native = load_native_flashnext_qsa_topk()
    if native is None:
        pytest.skip("native SM120 QSA top-k artifact is not built")

    monkeypatch.delenv("QSR_FLASHNEXT_QSA_TOPK_RERANK", raising=False)
    indexer = QSAIndexer()
    torch.manual_seed(2026)
    logits = torch.randn(1, 1024, device="cuda", dtype=torch.float32)
    ends = torch.tensor([1024], device="cuda", dtype=torch.int64)
    expected = torch.topk(logits, 512, dim=-1).indices

    for _ in range(3):
        got = indexer.select_blocks(logits, ends)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_qsa_topk_handles_a_broad_coarse_bin():
    """The radix selector must not drop indices when the half-float bin is wide."""

    from runtime.kernels.flashnext_qsa_topk import load_native_flashnext_qsa_topk

    native = load_native_flashnext_qsa_topk()
    if native is None:
        pytest.skip("native SM120 QSA top-k artifact is not built")

    # All values are close enough to share the coarse half-float range while
    # the FP32 ramp still gives the selector a unique top-k boundary.  This
    # exceeds the old 4096-entry scratch capacity and exercises the exact
    # row-rescan fallback.
    values = (2.0 + torch.arange(8192, device="cuda", dtype=torch.float32) * 2.3e-7).view(
        1, -1
    )
    lengths = torch.tensor([values.shape[1]], device="cuda", dtype=torch.int64)
    selected = native.select(values, lengths)
    expected = torch.topk(values, 512, dim=-1, sorted=False).indices
    torch.testing.assert_close(
        torch.sort(selected, dim=-1).values,
        torch.sort(expected, dim=-1).values,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_qsa_matches_reference_and_replays_cuda_graph():
    from runtime.model.flashnext.qsa_kernels import (
        qsa_sparse_attention,
        qsa_sparse_prefill_attention,
    )

    torch.manual_seed(19)
    rows, heads, kv_heads, dim, selected = 2, 24, 2, 256, 71
    q = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(q)
    k_cache = torch.randn(96, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    indices = torch.randint(0, 96, (rows, selected), device="cuda")
    valid = torch.ones(rows, selected, device="cuda", dtype=torch.bool)
    valid[0, -7:] = False

    gathered_k = k_cache[indices.reshape(-1)].view(rows, selected, kv_heads, dim)
    gathered_v = v_cache[indices.reshape(-1)].view_as(gathered_k)
    gathered_k = gathered_k.repeat_interleave(heads // kv_heads, dim=2)
    gathered_v = gathered_v.repeat_interleave(heads // kv_heads, dim=2)
    scores = torch.einsum("mhd,mshd->mhs", q.float(), gathered_k.float()) * (dim**-0.5)
    scores = torch.where(valid.unsqueeze(1), scores, torch.finfo(torch.float32).min)
    reference = torch.einsum("mhs,mshd->mhd", torch.softmax(scores, dim=-1), gathered_v.float()).to(
        q.dtype
    )
    reference *= torch.sigmoid(gate.float()).to(q.dtype)

    for _ in range(3):
        fused = qsa_sparse_attention(q, gate, k_cache, v_cache, indices, valid)
    packed_counts = valid.sum(dim=1, dtype=torch.int32)
    packed_fused = qsa_sparse_attention(
        q,
        gate,
        k_cache,
        v_cache,
        indices,
        valid,
        selected_counts=packed_counts,
    )
    prefill_fused = qsa_sparse_prefill_attention(
        q,
        gate,
        k_cache,
        v_cache,
        indices,
        valid,
        valid.sum(dim=1, dtype=torch.int32),
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = qsa_sparse_attention(q, gate, k_cache, v_cache, indices, valid)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(fused.float(), reference.float(), rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(packed_fused.float(), reference.float(), rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(prefill_fused.float(), reference.float(), rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(captured.float(), fused.float(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_qsa_all_invalid_padding_blocks_stay_finite():
    """Fixed-width decode must ignore padding-only online-softmax blocks.

    Position zero has one valid causal token followed by the full 2051-lane
    gather padding.  An online reduction initialized at ``-inf`` turns an
    all-invalid block into ``-inf - -inf`` and poisons the accumulator with
    NaNs; this is the exact shape used by the target CUDA Graph.
    """

    from runtime.model.flashnext.qsa_kernels import (
        qsa_sparse_attention,
        qsa_sparse_prefill_attention,
    )

    torch.manual_seed(191)
    rows, heads, kv_heads, dim, selected = 1, 24, 2, 256, 2051
    q = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(q)
    k_cache = torch.randn(8, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    indices = torch.zeros(rows, selected, device="cuda", dtype=torch.long)
    valid = torch.zeros(rows, selected, device="cuda", dtype=torch.bool)
    valid[:, 0] = True

    fused = qsa_sparse_attention(q, gate, k_cache, v_cache, indices, valid)
    prefill = qsa_sparse_prefill_attention(
        q,
        gate,
        k_cache,
        v_cache,
        indices,
        valid,
        torch.ones(rows, device="cuda", dtype=torch.int32),
    )
    torch.cuda.synchronize()

    selected_k = k_cache[indices[:, :1].reshape(-1)].view(rows, 1, kv_heads, dim)
    selected_v = v_cache[indices[:, :1].reshape(-1)].view_as(selected_k)
    selected_k = selected_k.repeat_interleave(heads // kv_heads, dim=2)
    selected_v = selected_v.repeat_interleave(heads // kv_heads, dim=2)
    scores = torch.einsum("mhd,mshd->mhs", q.float(), selected_k.float()) * (dim**-0.5)
    reference = torch.einsum(
        "mhs,mshd->mhd", torch.softmax(scores, dim=-1), selected_v.float()
    ).to(q.dtype)
    reference *= torch.sigmoid(gate.float()).to(q.dtype)

    assert torch.isfinite(fused.float()).all()
    assert torch.isfinite(prefill.float()).all()
    torch.testing.assert_close(fused.float(), reference.float(), rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(prefill.float(), reference.float(), rtol=2e-2, atol=2e-3)
