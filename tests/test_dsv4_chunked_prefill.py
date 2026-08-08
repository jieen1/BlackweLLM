"""Chunked-prefill equivalence tests (CPU/GPU-free): mid-sequence state.

The serving backend prefills long prompts in chunks of 128 rows.  A
mid-sequence chunk must produce exactly the same compressor KV state,
compressed entries, and attention indices as the single-shot prefill
oracle (start_pos==0 with the whole prompt).  These tests exercise the
state machines directly on a tiny config -- no model weights, no kernel.

The single-shot path is the executable definition: a chunk of tokens at
start_pos>0 is L sequential per-token decode steps, so the chunked
compressor output must equal the concatenation of L single-token decode
steps, and the chunked attention indices must equal the single-shot
indices for the same rows.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.model.dsv4_attention import compress_topk_idxs, window_topk_idxs  # noqa: E402
from runtime.model.dsv4_config import Dsv4Config  # noqa: E402
from runtime.model.dsv4_model import Dsv4Compressor  # noqa: E402

TINY = Dsv4Config(
    vocab_size=128,
    hidden_size=64,
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

#: Non-overlap compressor variant (ratio-128 layers use coff=1).
TINY128 = Dsv4Config(
    vocab_size=128,
    hidden_size=64,
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
    compress_ratios=(128,),
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


def _make_compressor(ratio: int = 4) -> Dsv4Compressor:
    cfg = TINY if ratio == 4 else TINY128
    comp = Dsv4Compressor(cfg, 0, quantize=False, device="cpu")
    # small deterministic packed Q8_0 payloads (d=1.0, qs in [-3, 3]) so
    # the dequant stays in a sane range -- random bytes explode the gated
    # pooling softmax (measured: kv_state to ~1e7 before overflow -> NaN)
    for mod in (comp.wkv, comp.wgate):
        n_blocks = mod.packed.numel() // 34  # Q8_0 block: 2 fp16 + 32 int8
        qs = torch.randint(-3, 4, (n_blocks, 32), dtype=torch.int8)
        d = torch.full((n_blocks, 1), 1.0, dtype=torch.float16)
        packed = torch.cat([d.view(torch.uint8), qs.view(torch.uint8)], dim=1)
        mod.packed.copy_(packed.reshape(-1))
    comp.kv_cache = torch.zeros(1, 64, cfg.head_dim, dtype=torch.bfloat16)
    comp.freqs_cis = torch.zeros(64, cfg.rope_head_dim // 2, dtype=torch.bfloat16)
    comp.ape.normal_()  # uninitialized buffer in the real graph; filled by GGUF
    comp.norm_weight.normal_()
    return comp


def _rand_x(seqlen: int) -> torch.Tensor:
    return torch.randn(1, seqlen, TINY.hidden_size, dtype=torch.bfloat16)


def _run_single_step(comp: Dsv4Compressor, x: torch.Tensor, pos: int):
    """One token at absolute pos through the decode branch; returns the
    emitted entry (or None) and a snapshot of the compressed cache."""
    out = comp(x, pos)
    return out, comp.kv_cache.clone()


def test_compressor_chunk_equals_sequential_decode() -> None:
    """A 6-token mid-sequence chunk must equal 6 single decode steps."""
    _check_chunk_vs_sequential(4)


def test_compressor_chunk_equals_sequential_overlap() -> None:
    # ratio-128: start just before a boundary so the chunk emits entries
    _check_chunk_vs_sequential(128, start_pos=126)


def _check_chunk_vs_sequential(ratio: int, start_pos: int = 10) -> None:
    comp_a = _make_compressor(ratio)  # chunked
    comp_b = _make_compressor(ratio)  # sequential
    comp_b.wkv.packed.copy_(comp_a.wkv.packed)
    comp_b.wgate.packed.copy_(comp_a.wgate.packed)
    comp_b.ape.copy_(comp_a.ape)
    comp_b.norm_weight.copy_(comp_a.norm_weight)

    x = _rand_x(6)

    for p in range(start_pos):
        xp = _rand_x(1)
        _run_single_step(comp_a, xp, p)
        _run_single_step(comp_b, xp, p)

    chunk_out, chunk_cache = _run_single_step(comp_a, x, start_pos)

    seq_outs: list = []
    for i in range(6):
        xi = x[:, i : i + 1]
        out, _ = _run_single_step(comp_b, xi, start_pos + i)
        if out is not None:
            seq_outs.append(out)
    seq_cache = comp_b.kv_cache.clone()

    assert (chunk_out is None) == (not seq_outs), (
        f"emission pattern differs: chunk={chunk_out is not None} seq={bool(seq_outs)}"
    )
    if chunk_out is None:
        assert torch.allclose(chunk_cache, seq_cache, atol=1e-6)
        return
    chunk_flat = chunk_out.reshape(-1)
    seq_flat = torch.cat(seq_outs, dim=1).reshape(-1)
    assert chunk_flat.shape == seq_flat.shape
    assert torch.allclose(chunk_flat, seq_flat, atol=1e-6), (
        f"chunk/seq mismatch: max {(chunk_flat - seq_flat).abs().max().item()}"
    )
    assert torch.allclose(chunk_cache, seq_cache, atol=1e-6)


def test_window_indices_chunk_matches_single_shot() -> None:
    """Mid-sequence chunk window indices == single-shot prefill rows.

    The chunk path reads the window RING (window_pages): each row's causal
    window [p-win+1, p] maps to ring slots pos%win.  The single-shot
    prefill (start_pos=0) reads absolute positions; converting its rows to
    ring slots must reproduce the chunk rows' valid subsequences (the
    kernel skips -1, so padding placement is irrelevant; the ORDER of
    valid slots is what matters).
    """
    win, bsz = TINY.window_size, 1
    start_pos, seqlen = 5, 6  # crosses the window boundary at p=7
    chunk = window_topk_idxs(win, bsz, seqlen, start_pos, "cpu")[0]

    total = start_pos + seqlen
    single = window_topk_idxs(win, bsz, total, 0, "cpu")[0]  # absolute positions
    rows_abs = single[start_pos : start_pos + seqlen]
    for row_c, row_a in zip(chunk, rows_abs):
        valid_c = [int(v) for v in row_c if v >= 0]
        valid_a = [int(v % win) for v in row_a if v >= 0]
        assert valid_c == valid_a, f"{valid_c} != {valid_a}"


def test_window_indices_ring_order_for_late_rows() -> None:
    """Rows past the window must be ring-ordered, ending at their own slot."""
    win, bsz = TINY.window_size, 1
    start_pos, seqlen = 40, 3
    chunk = window_topk_idxs(win, bsz, seqlen, start_pos, "cpu")[0]
    for i, row in enumerate(chunk):
        p = start_pos + i
        assert row[-1] == p % win
        # all win entries valid (past the window), ring wrap correct
        assert (row >= 0).all()
        assert len(set(row.tolist())) == win


def test_compress_indices_chunk_matches_single_shot() -> None:
    """Chunked compress indices == single-shot prefill's causal rows."""
    ratio, bsz = 4, 1
    start_pos, seqlen = 6, 8
    offset = 16
    chunk = compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset, "cpu")[0]

    rows = torch.arange(seqlen)
    n = (start_pos + 1 + rows) // ratio
    maxn = (start_pos + seqlen) // ratio
    expect = torch.arange(maxn).unsqueeze(0).repeat(seqlen, 1)
    expect = torch.where(expect >= n.unsqueeze(1), torch.full_like(expect, -1), expect)
    expect = torch.where(expect >= 0, expect + offset, expect)
    assert torch.equal(chunk, expect.int())


def test_indexer_chunk_equals_sequential() -> None:
    """The indexer's per-row top-k for a mid-sequence chunk must equal the
    single-shot prefill's rows for the same absolute positions."""
    from runtime.model.dsv4_model import Dsv4Indexer

    def _make_indexer() -> Dsv4Indexer:
        idx = Dsv4Indexer(TINY, 0, max_seq_len=64, device="cpu")
        for mod in (idx.wq_b, idx.weights_proj):
            mod.packed.copy_(
                torch.randint(-3, 4, mod.packed.shape, dtype=torch.int8).view(torch.uint8)
            )
        for mod in (idx.compressor.wkv, idx.compressor.wgate):
            n_blocks = mod.packed.numel() // 34
            qs = torch.randint(-3, 4, (n_blocks, 32), dtype=torch.int8)
            d = torch.full((n_blocks, 1), 1.0, dtype=torch.float16)
            packed = torch.cat([d.view(torch.uint8), qs.view(torch.uint8)], dim=1)
            mod.packed.copy_(packed.reshape(-1))
        idx.compressor.ape.normal_()
        idx.compressor.norm_weight.normal_()
        # wire the compressor into the indexer's scoring cache (the model
        # graph does this in _wire_subcaches)
        idx.compressor.kv_cache = idx.kv_cache
        idx.compressor.freqs_cis = torch.zeros(
            64, TINY.rope_head_dim // 2, dtype=torch.bfloat16
        )
        idx.freqs_cis = idx.compressor.freqs_cis
        return idx

    idx_a = _make_indexer()
    idx_b = _make_indexer()
    idx_b.wq_b.packed.copy_(idx_a.wq_b.packed)
    idx_b.weights_proj.packed.copy_(idx_a.weights_proj.packed)
    idx_b.compressor.wkv.packed.copy_(idx_a.compressor.wkv.packed)
    idx_b.compressor.wgate.packed.copy_(idx_a.compressor.wgate.packed)
    idx_b.compressor.ape.copy_(idx_a.compressor.ape)
    idx_b.compressor.norm_weight.copy_(idx_a.compressor.norm_weight)

    start_pos, seqlen = 8, 6
    x = _rand_x(seqlen)  # hidden_size
    qr = torch.randn(1, seqlen, TINY.q_lora_rank, dtype=torch.bfloat16)
    for p in range(start_pos):
        xp = _rand_x(1)
        idx_a.compressor(xp, p)
        idx_b.compressor(xp, p)

    chunk = idx_a(x, qr, start_pos, offset=0)
    for i in range(seqlen):
        r = idx_b(x[:, i : i + 1], qr[:, i : i + 1], start_pos + i, offset=0)[0, 0]
        row_c = chunk[0, i]
        valid_c = sorted(int(v) for v in row_c if v >= 0)
        valid_s = sorted(int(v) for v in r if v >= 0)
        assert valid_c == valid_s, f"row {i}: chunk {valid_c} != seq {valid_s}"
