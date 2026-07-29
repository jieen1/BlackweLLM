"""Self-built VocabParallelEmbedding / ParallelLMHead / LogitsProcessor --
阶段6 of the vLLM removal plan (self-build "剩下的主要缺口之一").

TP=1 simplified, same design as ``plain_linear.py``: own the weight_loader
closure, skip vLLM's TP-sharding protocol entirely rather than port it
unused. Verified against real vLLM source
(vllm/model_executor/layers/vocab_parallel_embedding.py,
vllm/model_executor/layers/logits_processor.py) and this runtime's actual
construction sites before simplifying, not assumed:

- ``VocabParallelEmbedding``'s entire vocab-padding/TP-sharding/LoRA-added-
  vocab machinery is provably inert for every real construction site in
  this runtime: Laguna's ``vocab_size=100352`` is already an exact
  multiple of ``DEFAULT_VOCAB_PADDING_SIZE=64`` (``100352 % 64 == 0``,
  verified directly), ``draft_vocab_size == vocab_size`` (checked in
  ``LagunaDraftForCausalLMSelfBuilt.__init__``), there is no LoRA
  (``SupportsLoRA`` never exercised, per laguna_model.py's docstring), and
  TP is always 1. At those values every shard/padding index collapses to
  the trivial case (``org_vocab_size == num_embeddings_padded ==
  num_embeddings``, single shard covering the whole tensor) and
  ``forward()`` reduces to a plain ``F.embedding`` lookup with a no-op
  ``tensor_model_parallel_all_reduce`` (world_size=1) dropped entirely.
  TP sharding is a documented future extension point, not implemented --
  see class docstrings below for exactly what would need to change.
- ``LogitsProcessor.forward`` -> ``_get_logits`` calls
  ``lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)``,
  which for the unquantized case (Laguna's lm_head/embed_tokens are BF16,
  confirmed by the阶段2 checkpoint audit -- "ignore": ["lm_head", ...])
  resolves to ``default_unquantized_gemm`` -- literally
  ``torch.nn.functional.linear(x, weight, bias)`` (traced through
  vllm/model_executor/layers/utils.py, not assumed). Note this passes the
  caller's ``embedding_bias`` argument (always None at every call site in
  this runtime) to the GEMM, NOT ``lm_head.bias`` -- matched exactly below,
  even though it means a hypothetical lm_head bias would never actually
  apply through this path (moot for Laguna: bias=False everywhere).
  ``_gather_logits`` (TP all-gather/gather) and the
  ``logits[..., :org_vocab_size]`` padding-strip are both no-ops at TP=1
  with no vocab padding, so both are dropped. ``soft_cap``/``scale`` are
  never set at this runtime's one LogitsProcessor(config.vocab_size)
  construction site (both default to None/1.0, i.e. inert) -- kept as
  real (if currently-inert) parameters rather than deleted, matching
  vLLM's own construction shape.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PlainEmbedding(nn.Module):
    """TP=1 embedding lookup. Self-built equivalent of vLLM's
    ``VocabParallelEmbedding`` for this runtime's actual usage (no LoRA, no
    vocab padding needed -- see module docstring). Multi-GPU support would
    mean re-adding per-rank vocab sharding + ``tensor_model_parallel_all_
    reduce`` in ``forward`` -- a documented extension point, not started.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.data.shape == loaded_weight.shape, (
            f"PlainEmbedding weight: dst {tuple(param.data.shape)} vs "
            f"loaded {tuple(loaded_weight.shape)}"
        )
        param.data.copy_(loaded_weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids.long(), self.weight)


class PlainLMHead(nn.Module):
    """Weight (+ optional bias) container for the LM head. Not directly
    callable -- like vLLM's ``ParallelLMHead``, its weights are meant to be
    read by a ``PlainLogitsProcessor``, matching the same "raise if called
    directly" contract vLLM's version has.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self._weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(num_embeddings))
            self.bias.weight_loader = self._weight_loader
        else:
            self.register_parameter("bias", None)

    def _weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        assert param.data.shape == loaded_weight.shape, (
            f"PlainLMHead: dst {tuple(param.data.shape)} vs loaded {tuple(loaded_weight.shape)}"
        )
        param.data.copy_(loaded_weight)

    def tie_weights(self, embed_tokens: PlainEmbedding) -> PlainLMHead:
        """Object-reference tie, matching vLLM's UnquantizedEmbeddingMethod.
        tie_weights exactly: `layer.weight = embed_tokens.weight` (same
        Parameter object, not a copy)."""
        self.weight = embed_tokens.weight
        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        del input_ids
        raise RuntimeError("PlainLMHead's weights should be used via PlainLogitsProcessor.")


class PlainLogitsProcessor(nn.Module):
    """Self-built equivalent of vLLM's ``LogitsProcessor`` for this
    runtime's one real call shape: unquantized lm_head, TP=1, no vocab
    padding, no soft_cap/scale ever configured (see module docstring)."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        lm_head: PlainLMHead,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(hidden_states, lm_head.weight, embedding_bias)
