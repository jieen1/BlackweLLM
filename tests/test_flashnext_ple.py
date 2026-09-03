"""PLE hasher/table/layer gates against hand derivations + real checkpoint."""

from __future__ import annotations

import pathlib

import pytest

torch = pytest.importorskip("torch")

CKPT = pathlib.Path("/home/bot/models/Qwen3.8-Flash-Next-NVFP4-RadixArk")

if not CKPT.is_dir():
    pytest.skip("RadixArk checkpoint not downloaded", allow_module_level=True)

from runtime.model.flashnext.ple import (  # noqa: E402
    FlashNextPleHasher,
    FlashNextPLELayer,
    FlashNextPleTable,
)

EOS = 248044


@pytest.fixture(scope="module")
def table() -> FlashNextPleTable:
    t = FlashNextPleTable(CKPT, layer_idx=1)
    yield t
    t.close()


@pytest.fixture(scope="module")
def hasher(table) -> FlashNextPleHasher:
    return FlashNextPleHasher(table, eos_token_id=EOS)


def test_hash_matches_hand_derivation_no_eos(hasher, table):
    tokens = torch.tensor([100, 200, 300])
    ids = hasher.sequence_ids(tokens)
    assert tuple(ids.shape) == (3, 16)
    m = table.layer_multipliers.tolist()
    sizes = table.head_sizes.tolist()
    offs = table.head_offsets.tolist()
    # position 2, bigram head 0: (300*m0) XOR (200*m1) mod size0 + off0
    expect = ((300 * m[0]) ^ (200 * m[1])) % sizes[0] + offs[0]
    assert ids[2, 0].item() == expect
    # position 2, trigram head 8 (first trigram head)
    expect3 = ((300 * m[0]) ^ (200 * m[1]) ^ (100 * m[2])) % sizes[8] + offs[8]
    assert ids[2, 8].item() == expect3
    # position 0 has no left context: shifted positions read EOS
    expect0 = ((100 * m[0]) ^ (EOS * m[1])) % sizes[0] + offs[0]
    assert ids[0, 0].item() == expect0


def test_hash_eos_boundary_resets_window(hasher, table):
    tokens = torch.tensor([100, EOS, 300])
    ids = hasher.sequence_ids(tokens)
    m = table.layer_multipliers.tolist()
    sizes = table.head_sizes.tolist()
    offs = table.head_offsets.tolist()
    # position 2: its 1-shift is the EOS token itself (valid, in-segment)
    expect = ((300 * m[0]) ^ (EOS * m[1])) % sizes[0] + offs[0]
    assert ids[2, 0].item() == expect
    # trigram at position 2: 2-shift crosses the EOS -> reads EOS not 100
    expect3 = ((300 * m[0]) ^ (EOS * m[1]) ^ (EOS * m[2])) % sizes[8] + offs[8]
    assert ids[2, 8].item() == expect3


def test_hash_ids_stay_in_head_ranges(hasher, table):
    torch.manual_seed(0)
    tokens = torch.randint(0, 248044, (512,))
    ids = hasher.sequence_ids(tokens)
    for h in range(16):
        lo = int(table.head_offsets[h])
        hi = lo + int(table.head_sizes[h])
        col = ids[:, h]
        assert int(col.min()) >= lo and int(col.max()) < hi


def test_table_gather_matches_direct_read(table):
    import json

    from safetensors import safe_open

    with open(CKPT / "model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]
    key = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.weight"
    with safe_open(str(CKPT / wm[key]), framework="pt", device="cpu") as f:
        shard3 = f.get_tensor(key)
    local_rows = torch.tensor([0, 7, 2500011])
    global_ids = (3 * table.shard_rows + local_rows).view(3, 1).expand(3, 16)
    got = table.gather(global_ids)
    expect_fp8 = shard3[local_rows]
    expect = expect_fp8.to(torch.float8_e4m3fn).to(torch.bfloat16).float() * table.weight_scale
    assert torch.allclose(got[:, 0].float(), expect, atol=1e-3)


def test_table_gather_writes_into_out_buffer(table):
    local_rows = torch.tensor([3, 19, 42, 77], dtype=torch.long)
    ids = (5 * table.shard_rows + local_rows).view(4, 1).expand(4, 16)
    expected = table.gather(ids)
    out = torch.empty_like(expected)
    got = table.gather(ids, out=out)
    assert got is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_table_async_gather_matches_synchronous_bytes(table):
    local_rows = torch.tensor([11, 37, 101], dtype=torch.long)
    ids = (9 * table.shard_rows + local_rows).view(3, 1).expand(3, 16)
    pending = table.start_gather(ids)
    got = table.finish_gather(pending)
    expected = table.gather(ids)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_table_lazy_async_gather_matches_synchronous_bytes(table):
    local_rows = torch.tensor([13, 41, 109], dtype=torch.long)
    ids = (11 * table.shard_rows + local_rows).view(3, 1).expand(3, 16)
    pending = table.start_gather_lazy(lambda: ids, token_count=ids.shape[0])
    got = table.finish_gather(pending)
    expected = table.gather(ids)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_table_default_keeps_page_cache_off():
    cached = FlashNextPleTable(CKPT, layer_idx=1)
    try:
        assert cached._page_cache_cap == 0
    finally:
        cached.close()


def test_table_fifo_cache_evicts_stale_mapping():
    cached = FlashNextPleTable(CKPT, layer_idx=1, cache_rows=2)
    try:
        row0 = torch.zeros((1, 16), dtype=torch.long)
        row1 = torch.ones((1, 16), dtype=torch.long)
        row2 = torch.full((1, 16), 2, dtype=torch.long)
        expected0 = cached.gather(row0)
        cached.gather(row1)
        hits_before = cached.cache_hits
        torch.testing.assert_close(cached.gather(row0), expected0, rtol=0, atol=0)
        assert cached.cache_hits == hits_before + 16

        cached.gather(row2)  # FIFO slot 0 now belongs to row 2, not row 0.
        misses_before = cached.cache_misses
        torch.testing.assert_close(cached.gather(row0), expected0, rtol=0, atol=0)
        assert cached.cache_misses == misses_before + 16
    finally:
        cached.close()


def test_table_page_cache_reuses_aligned_reads():
    cached = FlashNextPleTable(CKPT, layer_idx=1, cache_rows=0, cache_pages=128)
    try:
        # Include rows around page boundaries; 160 does not divide 4096, so
        # some rows require two aligned pages.
        local = torch.tensor([24, 25, 50, 51])
        ids = (7 * cached.shard_rows + local).view(4, 1).expand(4, 16)
        expected = cached.gather(ids)
        misses = cached.page_cache_misses
        hits = cached.page_cache_hits
        torch.testing.assert_close(cached.gather(ids), expected, rtol=0, atol=0)
        assert cached.page_cache_misses == misses
        assert cached.page_cache_hits > hits
    finally:
        cached.close()


def test_table_stats_snapshot_reports_cache_rates():
    cached = FlashNextPleTable(CKPT, layer_idx=1, cache_rows=2, cache_pages=128)
    try:
        ids = torch.tensor([[cached.head_offsets[0].item()] * 16], dtype=torch.long)
        cached.gather(ids)
        cached.gather(ids)
        stats = cached.stats_snapshot()
        assert stats["row_cache_hits"] == 16
        assert stats["row_cache_misses"] == 16
        assert stats["row_cache_hit_rate"] == pytest.approx(0.5)
        assert stats["page_cache_capacity"] == 128
        assert stats["page_cache_entries"] > 0
    finally:
        cached.close()


def test_ple_layer_embed_writes_into_out_buffer(table):
    layer = FlashNextPLELayer(CKPT, table, layer_idx=1)
    ids = torch.tensor([[table.head_offsets[0].item()] * 16], dtype=torch.long)
    out = torch.empty(1, 2560, dtype=torch.bfloat16)
    returned = layer.embed(ids, out=out)
    assert returned is out
    assert returned.shape == out.shape
    assert torch.isfinite(returned.float()).all()


def test_ple_layer_inject_shapes_and_finite(table):
    layer = FlashNextPLELayer(CKPT, table, layer_idx=1)
    torch.manual_seed(1)
    t = 5
    emb = torch.randn(t, 2560, dtype=torch.bfloat16)
    hidden = torch.randn(t, 4 * 2560, dtype=torch.bfloat16)
    gated_flat, normed_flat = layer.inject(emb, hidden)
    assert tuple(gated_flat.shape) == (t, 4 * 2560)
    assert torch.isfinite(gated_flat.float()).all()
    out = layer.prefill_conv(normed_flat, seq_lens=[2, 3])
    assert tuple(out.shape) == (t, 4 * 2560)
    assert torch.isfinite(out.float()).all()


def test_ple_layer_decode_conv_state_consistency(table):
    layer = FlashNextPLELayer(CKPT, table, layer_idx=1)
    torch.manual_seed(2)
    seq = torch.randn(6, 4 * 2560, dtype=torch.bfloat16)
    full = layer.prefill_conv(seq, seq_lens=[6])
    state = torch.zeros(1, 4 * 2560, layer.state_len, dtype=torch.bfloat16)
    outs = []
    for i in range(6):
        y, state = layer.decode_conv(seq[i : i + 1], state)
        outs.append(y)
    dec = torch.cat(outs, dim=0)
    assert torch.allclose(dec.float(), full.float(), atol=1e-2)


@pytest.mark.parametrize("seq_len", [1, 8, 9, 10, 17])
def test_ple_stateful_prefill_matches_full_history(table, seq_len):
    layer = FlashNextPLELayer(CKPT, table, layer_idx=1)
    torch.manual_seed(23)
    state = torch.randn(1, 4 * 2560, layer.state_len, dtype=torch.bfloat16)
    sequence = torch.randn(seq_len, 4 * 2560, dtype=torch.bfloat16)
    reference_input = torch.cat([state, sequence.t().unsqueeze(0)], dim=-1)
    expected = torch.nn.functional.silu(
        torch.nn.functional.conv1d(
            reference_input,
            layer.conv_weight,
            groups=reference_input.shape[1],
            dilation=layer.conv_dilation,
        )
    ).squeeze(0).t()
    expected_state = reference_input[:, :, -layer.state_len :]

    actual_state = state.clone()
    actual = layer.prefill_conv_with_state(sequence, actual_state)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [10, 512])
def test_ple_stateful_prefill_fused_path_is_bit_exact(table, seq_len):
    from runtime.kernels.gdn_conv import fused_causal_conv_silu

    layer = FlashNextPLELayer(CKPT, table, layer_idx=1)
    layer.conv_weight = layer.conv_weight.cuda()
    torch.manual_seed(29)
    state = torch.randn(
        1,
        4 * 2560,
        layer.state_len,
        device="cuda",
        dtype=torch.bfloat16,
    )
    sequence = torch.randn(
        seq_len,
        4 * 2560,
        device="cuda",
        dtype=torch.bfloat16,
    )
    reference_input = torch.cat([state, sequence.t().unsqueeze(0)], dim=-1)
    expected = fused_causal_conv_silu(
        reference_input,
        layer.conv_weight,
        padding=0,
        out_len=seq_len,
        dilation=layer.conv_dilation,
    )
    assert expected is not None

    actual_state = state.clone()
    actual = layer.prefill_conv_with_state(sequence, actual_state)

    torch.testing.assert_close(actual, expected.squeeze(0).t(), rtol=0, atol=0)
    torch.testing.assert_close(
        actual_state,
        reference_input[:, :, -layer.state_len :],
        rtol=0,
        atol=0,
    )


def test_ple_short_conv_applies_silu(table):
    layer = FlashNextPLELayer(CKPT, table, layer_idx=1)
    torch.manual_seed(4)
    seq = torch.randn(3, 4 * 2560, dtype=torch.bfloat16)
    state = torch.zeros(1, 4 * 2560, layer.state_len, dtype=torch.bfloat16)
    conv_input = torch.cat([state, seq[:1].unsqueeze(-1)], dim=-1)
    raw = torch.nn.functional.conv1d(
        conv_input,
        layer.conv_weight,
        groups=conv_input.shape[1],
        dilation=layer.conv_dilation,
    ).squeeze(-1)
    got, _ = layer.decode_conv(seq[:1], state)
    torch.testing.assert_close(
        got,
        torch.nn.functional.silu(raw),
        rtol=0,
        atol=0,
    )


def test_ple_spec_conv_retains_each_commit_candidate(table):
    layer = FlashNextPLELayer(CKPT, table, layer_idx=1)
    torch.manual_seed(3)
    seq = torch.randn(4, 4 * 2560, dtype=torch.bfloat16)
    initial = torch.randn(1, 4 * 2560, layer.state_len, dtype=torch.bfloat16)
    original = initial.clone()

    expected_outputs = []
    expected_states = []
    state = initial.clone()
    for row in range(seq.shape[0]):
        out, state = layer.decode_conv(seq[row : row + 1], state)
        expected_outputs.append(out)
        expected_states.append(state.clone())

    fixed_rows = [torch.empty_like(initial) for _ in range(seq.shape[0])]
    outputs, snapshots = layer.spec_conv(seq, initial, fixed_rows)
    torch.testing.assert_close(outputs, torch.cat(expected_outputs), rtol=0, atol=0)
    for got, expected in zip(snapshots, expected_states, strict=True):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.testing.assert_close(initial, original, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seq_len", [1, 17, 109, 512])
def test_dilated_ple_conv_fusion_is_bit_exact(seq_len):
    from runtime.kernels.gdn_conv import fused_causal_conv_silu

    torch.manual_seed(19)
    channels = 10240
    dilation = 3
    kernel = 4
    state_len = (kernel - 1) * dilation
    x = torch.randn(
        1,
        channels,
        state_len + seq_len,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        channels,
        1,
        kernel,
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected = torch.nn.functional.silu(
        torch.nn.functional.conv1d(
            x,
            weight,
            groups=channels,
            dilation=dilation,
        )
    )
    got = fused_causal_conv_silu(
        x,
        weight,
        padding=0,
        out_len=seq_len,
        dilation=dilation,
    )
    assert got is not None
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
