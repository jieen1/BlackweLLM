"""Engram (n-gram conditional-memory) embedding for the Qwen4 family.

Day-0 prep for Qwen3.8-Flash-Next (see
notes/2026-08-26-qwen38-flash-next-day0-survey.md). Ported from vLLM's
``longcat_flash_ngram.py`` + the SGLang ``ngram_embedding.cuh`` reference
(both read 2026-08-26; the mechanism is the same Kimi-delta-style "Conditional
Memory via Scalable Lookup" construction Flash-Next uses):

* ``num_embedders = k * (n - 1)`` independent hash tables, embedder
  ``(i, j)`` having modulus ``m + 2*(i*k+j) + 1`` where
  ``m = ngram_vocab_size_ratio * vocab_size``;
* per-position id: base-``vocab`` polynomial hash of the current token plus
  up to ``n_idx + 1`` left-context tokens, ``hash = sum_j tok[c-j] *
  vocab^j mod mod`` (``ne_weights[i][j][delta] = vocab^delta mod mod``),
  truncated at sequence start, pad/boundary markers, and -- only when
  looking back (``j > 0``) -- at EOS tokens;
* one concatenated lookup table (per-embedder offsets are the exclusive
  prefix sums of the table sizes ``m + 2*r + 1``) plus stacked projections;
  the fused embedding is ``mean(word_embed, proj_0, ..., proj_{ne-1})`` --
  vLLM's exact ``embed_batched`` formula.

The hash walk is vectorized pure-torch (no custom kernel): a later
performance step can swap in a CUDA port of ``ComputeNGramIdsDecodeKernel``
without changing this module's contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class EngramConfig:
    """The three engram config keys (family naming pinned from vLLM's
    ``FlashConfig`` usage)."""

    vocab_size: int
    hidden_size: int
    ngram_vocab_size_ratio: int
    emb_split_num: int  # k
    emb_neighbor_num: int  # n
    pad_token_id: int | None
    eos_token_id: int

    @property
    def table_size(self) -> int:
        return self.ngram_vocab_size_ratio * self.vocab_size

    @property
    def num_embedders(self) -> int:
        return self.emb_split_num * (self.emb_neighbor_num - 1)

    @property
    def oe_dim(self) -> int:
        dim = self.hidden_size // self.num_embedders
        if dim * self.num_embedders != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by "
                f"num_embedders {self.num_embedders}"
            )
        return dim


def build_engram_tables(config: EngramConfig) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Return ``(ne_mods, ne_weights, offsets)``.

    ``ne_mods`` ``[n-1, k]`` int32; ``ne_weights`` ``[n-1, k, n]`` int32 with
    ``ne_weights[i][j][delta] = vocab^delta mod ne_mods[i][j]``; ``offsets``
    the exclusive prefix sums of the per-embedder table sizes (len
    ``num_embedders + 1``). Matches vLLM ``NgramEmbedding._init_ngram_embeddings``.
    """
    m = config.table_size
    k, n = config.emb_split_num, config.emb_neighbor_num
    ne_mods = torch.zeros((n - 1, k), dtype=torch.int32)
    ne_weights = torch.zeros((n - 1, k, n), dtype=torch.int32)
    sizes: list[int] = []
    offsets = [0]
    for i in range(n - 1):
        for j in range(k):
            mod = int(m + 2 * (i * k + j) + 1)
            ne_mods[i, j] = mod
            sizes.append(mod)
            offsets.append(offsets[-1] + mod)
            for delta in range(n):
                ne_weights[i, j, delta] = pow(config.vocab_size, delta, mod)
    return ne_mods, ne_weights, offsets


def compute_ngram_ids(
    tokens: torch.Tensor,
    config: EngramConfig,
    *,
    ne_mods: torch.Tensor | None = None,
    ne_weights: torch.Tensor | None = None,
    offsets: list[int] | None = None,
) -> torch.Tensor:
    """Compute global n-gram ids for one flat token sequence.

    ``tokens`` is ``[N]`` int; negative entries (e.g. the -1 pad used by
    fresh-request left context) are boundaries. Returns
    ``[N, num_embedders]`` int64 global ids (per-embedder offset applied).
    """
    if ne_mods is None or ne_weights is None or offsets is None:
        ne_mods, ne_weights, offsets = build_engram_tables(config)
    toks = tokens.long()
    n_tokens = toks.shape[0]
    k, n = config.emb_split_num, config.emb_neighbor_num
    device = toks.device
    ne_mods = ne_mods.to(device)
    ne_weights = ne_weights.to(device)

    # valid[c] = token participates at all; the walk for position c stops at
    # the nearest strict predecessor that is negative or EOS.
    ids = torch.empty((n_tokens, config.num_embedders), dtype=torch.int64, device=device)
    for i in range(n - 1):
        for j in range(k):
            col = i * k + j
            mod = int(ne_mods[i, j])
            acc = torch.zeros(n_tokens, dtype=torch.int64, device=device)
            alive = torch.ones(n_tokens, dtype=torch.bool, device=device)
            for delta in range(n):
                if delta == 0:
                    window = toks
                else:
                    window = torch.full_like(toks, -1)
                    window[delta:] = toks[:-delta]
                boundary = (window < 0) | ((window == config.eos_token_id) & (delta > 0))
                alive &= ~boundary
                w = int(ne_weights[i, j, delta])
                term = (window.clamp_min(0) * w) % mod
                acc = torch.where(alive, (acc + term) % mod, acc)
            ids[:, col] = acc + offsets[col]
    return ids


class NgramEmbedding(nn.Module):
    """Concatenated n-gram table + stacked projections, fused into the word
    embedding exactly like vLLM's ``NgramEmbedding.embed_batched``."""

    def __init__(self, config: EngramConfig, word_embeddings: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.word_embeddings = word_embeddings
        ne_mods, ne_weights, offsets = build_engram_tables(config)
        self.register_buffer("ne_mods", ne_mods, persistent=False)
        self.register_buffer("ne_weights", ne_weights, persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int32), persistent=False)
        total_rows = offsets[-1]
        self.oe_embedder = nn.Embedding(total_rows, config.oe_dim)
        self.oe_projection = nn.Parameter(
            torch.empty(config.num_embedders, config.oe_dim, config.hidden_size),
            requires_grad=False,
        )

    def compute_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        return compute_ngram_ids(
            tokens,
            self.config,
            ne_mods=self.ne_mods,
            ne_weights=self.ne_weights,
            offsets=self.offsets.tolist(),
        )

    def embed_batched(self, input_ids: torch.Tensor, oe_ids: torch.Tensor) -> torch.Tensor:
        """``input_ids`` ``[N]``, ``oe_ids`` ``[N, num_embedders]`` global ids
        -> ``[N, hidden]``, the mean of the word embedding and one projection
        per embedder (vLLM's exact formula)."""
        word = self.word_embeddings(input_ids)
        flat = oe_ids.permute(1, 0).contiguous()
        oe = self.oe_embedder(flat)
        proj = torch.bmm(oe, self.oe_projection)
        all_h = torch.cat([word.unsqueeze(0), proj], dim=0)
        return all_h.mean(dim=0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_batched(input_ids, self.compute_ids(input_ids))
