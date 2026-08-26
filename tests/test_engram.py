"""Engram hash + embedding parity against hand-derived reference values.

The hash contract is pinned from SGLang's ``ngram_embedding.cuh``
``ComputeNGramIdsDecodeKernel`` and vLLM's ``longcat_flash_ngram.py``
(read 2026-08-26): base-vocab polynomial hash per embedder, walk truncated
at sequence start / negative markers / EOS-in-lookback, global ids offset
by the exclusive table-size prefix sums; the fused embedding is the mean of
the word embedding and one projection per embedder. CPU-only by design.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.model.engram import (  # noqa: E402
    EngramConfig,
    NgramEmbedding,
    build_engram_tables,
    compute_ngram_ids,
)


def _config(vocab=10, hidden=8, ratio=1, k=1, n=3, pad=None, eos=9) -> EngramConfig:
    return EngramConfig(
        vocab_size=vocab,
        hidden_size=hidden,
        ngram_vocab_size_ratio=ratio,
        emb_split_num=k,
        emb_neighbor_num=n,
        pad_token_id=pad,
        eos_token_id=eos,
    )


def test_table_layout_matches_vllm_formulas():
    cfg = _config()
    ne_mods, ne_weights, offsets = build_engram_tables(cfg)
    # num_embedders = k*(n-1) = 2; mods m+2r+1 with m = ratio*vocab = 10.
    assert ne_mods.tolist() == [[11], [13]]
    assert offsets == [0, 11, 24]
    # ne_weights[i][j][delta] = vocab^delta mod mod.
    assert ne_weights[0, 0].tolist() == [pow(10, d, 11) for d in range(3)]
    assert ne_weights[1, 0].tolist() == [pow(10, d, 13) for d in range(3)]


def test_full_context_hash_matches_hand_derivation():
    cfg = _config()
    tokens = torch.tensor([2, 3, 4])
    ids = compute_ngram_ids(tokens, cfg)
    # Position 2: 4*10^0 + 3*10^1 + 2*10^2 = 234.
    assert ids[2, 0].item() == 234 % 11 + 0  # embedder 0, mod 11, offset 0
    assert ids[2, 1].item() == 234 % 13 + 11  # embedder 1, mod 13, offset 11
    # Position 1 has only two tokens of context: 3 + 2*10 = 23.
    assert ids[1, 0].item() == 23 % 11
    # Position 0: single token.
    assert ids[0, 0].item() == 2 % 11


def test_eos_stops_lookback_but_not_current_token():
    cfg = _config(eos=9)
    tokens = torch.tensor([5, 9, 7])
    ids = compute_ngram_ids(tokens, cfg)
    # Position 2 (token 7): lookback hits EOS at j=1 -> hash is 7 alone.
    assert ids[2, 0].item() == 7 % 11
    # Position 1 (the EOS itself): j=0 allowed -> 9 + 5*10 = 59.
    assert ids[1, 0].item() == 59 % 11


def test_negative_marker_is_boundary():
    cfg = _config()
    tokens = torch.tensor([-1, 6])
    ids = compute_ngram_ids(tokens, cfg)
    assert ids[1, 0].item() == 6 % 11
    # The negative position itself accumulates nothing.
    assert ids[0, 0].item() == 0


def test_ids_stay_inside_per_embedder_ranges():
    cfg = _config(vocab=97, k=3, n=4)
    _, _, offsets = build_engram_tables(cfg)
    torch.manual_seed(0)
    tokens = torch.randint(0, 97, (256,))
    ids = compute_ngram_ids(tokens, cfg)
    assert tuple(ids.shape) == (256, cfg.num_embedders)
    for col in range(cfg.num_embedders):
        assert ids[:, col].min().item() >= offsets[col]
        assert ids[:, col].max().item() < offsets[col + 1]


def test_embed_batched_is_mean_of_word_and_projections():
    cfg = EngramConfig(
        vocab_size=8,
        hidden_size=4,
        ngram_vocab_size_ratio=1,
        emb_split_num=1,
        emb_neighbor_num=2,
        pad_token_id=None,
        eos_token_id=7,
    )
    # num_embedders = 1, oe_dim = 4, table rows = m+1 = 9.
    word = torch.nn.Embedding(8, 4)
    with torch.no_grad():
        word.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(8, 4))
    eng = NgramEmbedding(cfg, word)
    with torch.no_grad():
        eng.oe_embedder.weight.copy_(torch.arange(9 * 4, dtype=torch.float32).reshape(9, 4))
        eng.oe_projection.copy_(torch.eye(4).unsqueeze(0) * 2.0)

    tokens = torch.tensor([3, 5])
    ids = eng.compute_ids(tokens)
    out = eng.embed_batched(tokens, ids)

    # Hand derivation: embedder mod is 9; pos0 id = 3, pos1 id = 5 + 3*8 = 29
    # -> 29 % 9 = 2, offset 0.
    assert ids.tolist() == [[3], [2]]
    expected = torch.stack(
        [
            (word.weight[3] + eng.oe_embedder.weight[3] @ (2 * torch.eye(4))) / 2,
            (word.weight[5] + eng.oe_embedder.weight[2] @ (2 * torch.eye(4))) / 2,
        ]
    )
    torch.testing.assert_close(out, expected)


def test_forward_end_to_end_is_deterministic():
    cfg = _config(vocab=50, hidden=16, k=2, n=3)
    word = torch.nn.Embedding(50, 16)
    eng = NgramEmbedding(cfg, word)
    with torch.no_grad():
        eng.oe_embedder.weight.normal_(0, 0.1)
        eng.oe_projection.normal_(0, 0.1)
    tokens = torch.randint(0, 50, (64,))
    a = eng(tokens)
    b = eng(tokens)
    assert tuple(a.shape) == (64, 16)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
