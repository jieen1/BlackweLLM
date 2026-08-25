"""Self-built Qwen3.8 DFlash2 draft model.

The target remains the Q6_K_XL GGUF Qwen3.8 graph.  This module implements
only the separate DFlash2 drafter and deliberately follows the existing
``Qwen36DSparkDraftForCausalLM`` boundary so the backend can own paged KV
storage, target hidden-state injection, and verify/commit state transitions.

The official DFlash2 checkpoint has four pieces that distinguish it from the
older DSpark draft:

* two-tap grouped dynamic causal convolutions around attention and MLP;
* a 5-layer Qwen3 dense backbone with non-causal sliding attention;
* a target-hidden projector shared with the target's configured taps;
* a top-k candidate selector with predecessor/successor codebooks.

The backend captures the greedy selector path together with the non-causal
sliding-window attention in ``qwen36_dspark_cudagraph``.  Temperature-based
sampling remains eager because its request-local multinomial generator is not
part of the fixed CUDA Graph contract.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from runtime.backends.flashinfer_dspark_attn import (
    flashinfer_deterministic_topk as _flashinfer_top_k,
)
from runtime.dflash2_config import DFlash2DraftConfig
from runtime.kernels.fused_rms_norm import TritonRMSNorm
from runtime.model._prefix import maybe_prefix
from runtime.model._weight_loading import default_weight_loader
from runtime.model.laguna_decoder import LagunaDecoderLayerSelfBuilt
from runtime.model.plain_embedding import PlainEmbedding, PlainLMHead
from runtime.model.plain_linear import PlainLinear
from runtime.model.qwen36_dspark import Qwen36DSparkDraftModel


def _dflash2_hf_config(config: DFlash2DraftConfig) -> SimpleNamespace:
    """Adapt DFlash2 JSON fields to the existing self-built decoder layer."""

    return SimpleNamespace(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        max_position_embeddings=262144,
        layer_types=["sliding_attention"] * config.num_hidden_layers,
        sliding_window=config.sliding_window,
        is_causal=False,
        rope_parameters=dict(config.rope_parameters),
        rms_norm_eps=config.rms_norm_eps,
        hidden_act="silu",
        qkv_bias=False,
        attention_bias=False,
        gating=False,
        num_experts=0,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )


def _grouped_dynamic_convolve(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    group_size: int,
    block_size: int,
) -> torch.Tensor:
    """Apply grouped dynamic causal convolution with the runtime block reset."""

    if hidden.ndim != 3:
        raise ValueError(f"DFlash2 dynamic convolution expects [B,L,H], got {hidden.shape}")
    if block_size <= 0:
        raise ValueError("DFlash2 dynamic convolution block_size must be positive")
    batch, length, hidden_size = hidden.shape
    if hidden_size % group_size:
        raise ValueError("DFlash2 hidden size must be divisible by conv group size")
    groups = hidden_size // group_size
    taps = base.shape[0]
    blocks = hidden.reshape(batch, length, groups, group_size)
    dynamic = dynamic.reshape(batch, length, taps, groups, 1).to(hidden.dtype)
    output = torch.zeros_like(blocks)
    positions = torch.arange(length, device=hidden.device)
    if block_size & (block_size - 1) == 0:
        positions = positions & (block_size - 1)
    else:
        positions = positions.remainder(block_size)
    for offset in range(taps):
        if offset == 0:
            values = blocks
        else:
            values = torch.zeros_like(blocks)
            if offset < length:
                values[:, offset:] = blocks[:, :-offset]
            values = values * (positions >= offset).view(1, length, 1, 1)
        kernel = base[offset].reshape(1, 1, groups, group_size).to(hidden.dtype)
        # Keep the reference operation order.  ``base + dynamic`` followed by
        # one multiply is algebraically equivalent, but it changes BF16
        # rounding at every tap and compounds through five draft layers.
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, :, offset], values)
    return output.reshape_as(hidden)


class GroupedDynamicCausalConv(nn.Module):
    """Two-tap dynamic grouped convolution used by every DFlash2 layer."""

    def __init__(
        self, hidden_size: int, kernel_size: int, group_size: int, block_size: int
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError("DFlash2 hidden_size must be divisible by conv_group_size")
        if block_size <= 0:
            raise ValueError("DFlash2 block_size must be positive")
        self.kernel_size = kernel_size
        self.group_size = group_size
        self.block_size = block_size
        groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, kernel_size, hidden_size))
        self.kernel_projection = PlainLinear(hidden_size, 2 * kernel_size * groups, bias=False)

    def prepare(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        groups = hidden.shape[-1] // self.group_size
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1], 2, self.kernel_size, groups
        )
        return (
            _grouped_dynamic_convolve(
                hidden,
                dynamic[..., 0, :, :],
                self.base_kernel[0],
                self.group_size,
                self.block_size,
            ),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        return _grouped_dynamic_convolve(
            hidden, dynamic, self.base_kernel[1], self.group_size, self.block_size
        )


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Return the official DFlash2 edge scores with shape ``[B,L,K,K]``."""

    if candidate_ids.ndim != 3 or unary_logits.shape != candidate_ids.shape:
        raise ValueError("DFlash2 candidate tables must both have shape [B,L,K]")
    if candidate_ids.shape[-1] != top_k:
        raise ValueError("DFlash2 candidate table width does not match top_k")
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids.reshape(-1, 1, 1).expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits.unsqueeze(-1) + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden.unsqueeze(2), successors
    )


def _dflash2_topk(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic candidate rows using SGLang's fast top-k path."""

    if logits.is_cuda:
        flashinfer_result = _dflash2_flashinfer_topk(logits, top_k)
        if flashinfer_result is not None:
            return flashinfer_result
    return torch.topk(logits, top_k, dim=-1, sorted=True)


def _dflash2_flashinfer_topk(
    logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run FlashInfer top-k over arbitrary leading batch/position axes."""

    # FlashInfer's radix_topk API intentionally accepts only ``[N, V]``.
    # DFlash2 scores one shared-head row per draft position, so the model
    # naturally presents ``[B, L, V]`` here.  Flatten only the row axes;
    # restoring them is important because the selector walks positions in
    # order when it builds the predecessor path.
    row_shape = logits.shape[:-1]
    rows = logits.reshape(-1, logits.shape[-1])
    result = _flashinfer_top_k(rows, top_k)
    if result is None:
        return None
    values, indices = result
    return values.reshape(*row_shape, top_k), indices.reshape(*row_shape, top_k)


class DFlash2CandidateSelector(nn.Module):
    """Top-k predecessor/successor path selector from the official model."""

    def __init__(self, config: DFlash2DraftConfig) -> None:
        super().__init__()
        self.top_k = config.selector_top_k
        self.predecessor_codebook = nn.Embedding(config.vocab_size, config.selector_rank)
        self.successor_codebook = nn.Embedding(config.vocab_size, config.selector_rank)
        self.hidden_projection = PlainLinear(config.hidden_size, config.selector_rank, bias=False)

    def select(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        anchor_ids: torch.Tensor,
        *,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if hidden.ndim != 3 or logits.ndim != 3:
            raise ValueError("DFlash2 selector expects hidden/logits shaped [B,K,*]")
        if temperature < 0:
            raise ValueError("DFlash2 selector temperature must be non-negative")
        # SGLang uses FlashInfer's deterministic radix top-k on CUDA.  Besides
        # being materially cheaper for the 248K-wide shared head, sorted rows
        # make candidate ordering stable across graph captures and tie cases.
        unary, candidates = _dflash2_topk(logits, self.top_k)
        projected = self.hidden_projection(hidden)
        predecessor = anchor_ids.long()
        path: list[torch.Tensor] = []
        q_rows: list[torch.Tensor] = []
        for position in range(hidden.shape[1]):
            scores = unary[:, position] + torch.einsum(
                "br,bkr->bk",
                self.predecessor_codebook(predecessor) * projected[:, position],
                self.successor_codebook(candidates[:, position]),
            )
            if temperature > 0:
                q = torch.softmax(scores.float() / temperature, dim=-1)
                index = torch.multinomial(q, 1, generator=generator)[:, 0]
                q_rows.append(q)
            else:
                index = scores.argmax(dim=-1)
            predecessor = candidates[:, position].gather(-1, index[:, None])[:, 0]
            path.append(predecessor)
        q = torch.stack(q_rows, dim=1) if q_rows else None
        return torch.stack(path, dim=1), candidates, q


class Qwen38DFlash2DecoderLayer(LagunaDecoderLayerSelfBuilt):
    """Existing self-built Qwen decoder plus the two DFlash2 convs."""

    def __init__(
        self,
        config: SimpleNamespace,
        *,
        cache_config: object | None,
        prefix: str,
        layer_idx: int,
        attention_prefix: str,
        conv_kernel_size: int,
        conv_group_size: int,
    ) -> None:
        super().__init__(
            config=config,
            cache_config=cache_config,
            quant_config=None,
            prefix=prefix,
            layer_idx=layer_idx,
            attention_prefix=attention_prefix,
            # The official DFlash2 reference keeps its five-layer draft KV
            # cache in BF16. The target model's independent KV family stays
            # FP8; this is deliberately scoped to the draft layers.
            kv_cache_dtype="bfloat16",
        )
        self.attention_conv = GroupedDynamicCausalConv(
            config.hidden_size, conv_kernel_size, conv_group_size, config.block_size
        )
        self.mlp_conv = GroupedDynamicCausalConv(
            config.hidden_size, conv_kernel_size, conv_group_size, config.block_size
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError(
                "DFlash2 draft hidden states must have shape [B,K,H]; "
                f"got {tuple(hidden_states.shape)}"
            )
        if positions.ndim == 2:
            flat_positions = positions.reshape(-1)
        elif positions.ndim == 1:
            flat_positions = positions
        else:
            raise ValueError("DFlash2 positions must be rank 1 or 2")

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, attention_dynamic = self.attention_conv.prepare(hidden_states)
        batch, length, width = hidden_states.shape
        attention_out = self.self_attn(
            positions=flat_positions,
            hidden_states=hidden_states.reshape(batch * length, width),
        ).reshape(batch, length, width)
        hidden_states = self.attention_conv.finish(attention_out, attention_dynamic)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, mlp_dynamic = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, mlp_dynamic)
        return hidden_states, residual


class Qwen38DFlash2DraftModel(Qwen36DSparkDraftModel):
    """DFlash2 backbone with inherited target-KV projection/scatter helpers."""

    def __init__(self, config: DFlash2DraftConfig, *, target_layer_count: int) -> None:
        # We intentionally reuse the inherited context-KV methods, but build a
        # different layer body and do not invoke the DSpark constructor.
        nn.Module.__init__(self)
        self.draft_config = config
        decoder_config = _dflash2_hf_config(config)
        self.layers = nn.ModuleList(
            [
                Qwen38DFlash2DecoderLayer(
                    decoder_config,
                    cache_config=None,
                    prefix=maybe_prefix("model", f"layers.{i}"),
                    layer_idx=i,
                    attention_prefix=maybe_prefix("model", f"layers.{i + target_layer_count}"),
                    conv_kernel_size=config.conv_kernel_size,
                    conv_group_size=config.conv_group_size,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = PlainLinear(
            config.hidden_size * len(config.target_layer_ids), config.hidden_size, bias=False
        )
        self.hidden_norm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.candidate_selector = DFlash2CandidateSelector(config)
        self.block_size = config.block_size

    def set_block_size(self, block_size: int) -> None:
        """Propagate the worker-resolved block width into both convolutions.

        SGLang applies ``--speculative-num-draft-tokens`` after loading the
        draft.  The local engine currently resolves the width from the
        checkpoint, but keeping this setter at the model boundary prevents a
        future override from silently using stale convolution reset positions.
        """

        block_size = int(block_size)
        if block_size <= 0:
            raise ValueError(f"DFlash2 block_size must be positive, got {block_size}")
        self.block_size = block_size
        for layer in self.layers:
            layer.attention_conv.block_size = block_size
            layer.mlp_conv.block_size = block_size

    def forward(self, inputs_embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        if inputs_embeds.ndim != 3:
            raise ValueError(
                "DFlash2 draft inputs_embeds must have shape [B,K,H]; "
                f"got {tuple(inputs_embeds.shape)}"
            )
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        residual: torch.Tensor | None = None
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        if residual is None:
            return self.norm(hidden_states)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen38DFlash2DraftForCausalLM(nn.Module):
    """Backend-compatible DFlash2 wrapper with sparse draft distributions."""

    is_dflash2 = True
    confidence_head = None
    markov_head = None

    def __init__(self, config: DFlash2DraftConfig, *, target_layer_count: int) -> None:
        super().__init__()
        self.config = config
        if config.block_size < 2:
            raise ValueError(
                "DFlash2 block_size must include an anchor and at least one draft token"
            )
        # The official block contains the anchor plus ``block_size - 1``
        # proposed continuations. The backend's gamma counts proposals only.
        self.gamma = config.block_size - 1
        self.embed_tokens = PlainEmbedding(config.vocab_size, config.hidden_size)
        self.lm_head = PlainLMHead(config.vocab_size, config.hidden_size)
        self.model = Qwen38DFlash2DraftModel(config, target_layer_count=target_layer_count)
        self.model.set_block_size(config.block_size)
        self._sampling_temperature = 0.0
        self._sampling_generator: torch.Generator | None = None
        self.last_candidate_indices: torch.Tensor | None = None
        self.last_candidate_probs: torch.Tensor | None = None

    def attach_shared_modules(self, *, embed_tokens: nn.Module, lm_head: nn.Module) -> None:
        del self.embed_tokens, self.lm_head
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def set_sampling_context(
        self, *, temperature: float, generator: torch.Generator | None
    ) -> None:
        self._sampling_temperature = float(temperature)
        self._sampling_generator = generator

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids) * self.config.input_embedding_scale
        return embeds.to(dtype=self.model.fc.weight.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        elif inputs_embeds.dtype != self.model.fc.weight.dtype:
            inputs_embeds = inputs_embeds.to(dtype=self.model.fc.weight.dtype)
        return self.model(inputs_embeds, positions)

    def compute_base_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if isinstance(self.lm_head, PlainLMHead):
            if hidden_states.dtype != self.lm_head.weight.dtype:
                hidden_states = hidden_states.to(self.lm_head.weight.dtype)
            return F.linear(hidden_states, self.lm_head.weight)
        return self.lm_head(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.compute_base_logits(hidden_states)
        logits = logits * self.config.output_multiplier
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits

    def sample_block(
        self,
        hidden_states: torch.Tensor,
        *,
        anchor_tokens: torch.Tensor,
        sampler: Callable[[torch.Tensor, int], torch.Tensor],
        capture_confidence: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        del sampler, capture_confidence
        if hidden_states.ndim != 3:
            raise ValueError(
                f"DFlash2 block hidden states must be rank 3, got {hidden_states.ndim}"
            )
        logits = self.compute_logits(hidden_states)
        tokens, candidates, probs = self.model.candidate_selector.select(
            hidden_states,
            logits,
            anchor_tokens,
            temperature=self._sampling_temperature,
            generator=self._sampling_generator,
        )
        self.last_candidate_indices = candidates
        self.last_candidate_probs = probs
        return tokens, logits, None

    def propose(
        self,
        hidden_states: torch.Tensor,
        *,
        anchor_tokens: torch.Tensor,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        self.set_sampling_context(temperature=temperature, generator=generator)
        tokens, _, _ = self.sample_block(
            hidden_states,
            anchor_tokens=anchor_tokens,
            sampler=lambda logits, _step: logits.argmax(dim=-1),
        )
        assert self.last_candidate_indices is not None
        return tokens, self.last_candidate_indices, self.last_candidate_probs

    def precompute_and_store_context_kv(
        self,
        target_hidden: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            target_hidden, context_positions, context_slot_mapping
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Map the flat official DFlash2 tensor names into this graph."""

        params = dict(self.named_parameters())
        loaded: set[str] = set()
        shard_map = {
            "q_proj": ("qkv_proj", "q"),
            "k_proj": ("qkv_proj", "k"),
            "v_proj": ("qkv_proj", "v"),
        }
        for name, tensor in weights:
            mapped: str | None = None
            shard_id: str | None = None
            parts = name.split(".")
            if len(parts) >= 5 and parts[0] == "layers" and parts[2] == "self_attn":
                projection = parts[3]
                if projection in shard_map and parts[4] == "weight":
                    fused, shard_id = shard_map[projection]
                    mapped = f"model.layers.{parts[1]}.self_attn.{fused}.weight"
                elif parts[4] == "weight":
                    mapped = f"model.layers.{parts[1]}.self_attn.{projection}.weight"
            if mapped is None:
                if name.startswith("candidate_selector."):
                    mapped = f"model.{name}" if name.endswith(".weight") else f"model.{name}.weight"
                elif name.startswith(("fc.", "hidden_norm.", "norm.", "layers.")):
                    mapped = f"model.{name}"
                else:
                    raise RuntimeError(f"unexpected DFlash2 checkpoint tensor: {name}")
            param = params.get(mapped)
            if param is None:
                raise RuntimeError(
                    f"DFlash2 tensor {name!r} maps to missing runtime parameter {mapped!r}"
                )
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                loader(param, tensor)
            else:
                loader(param, tensor, shard_id)
            loaded.add(mapped)
        return loaded
