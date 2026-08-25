"""Qwen3.8's external DSpark draft graph.

The Qwen3.8 target remains the existing hybrid Qwen36 graph.  This module
owns only the separate five-layer dense draft described by
``RadixArk/Qwen3.8-27B-DSpark``:

* target embeddings and the target LM head are shared by object reference;
* target post-layer hidden states are projected through ``fc`` and
  ``hidden_norm`` into the draft context;
* the draft backbone is ordinary Qwen3 full attention;
* a vanilla Markov head supplies the semi-autoregressive token correction.

Attention cache allocation/dispatch is deliberately kept outside this model,
matching ``LagunaDraftForCausalLMSelfBuilt``.  The backend owns the cache
addresses and binds ``SelfBuiltAttentionPlaceholder`` to the runtime's one
attention implementation before the first forward.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from runtime.dspark_config import DSparkDraftConfig
from runtime.kernels.fused_rms_norm import TritonRMSNorm, rms_norm
from runtime.kernels.rope import apply_rotary_embedding_inplace
from runtime.model._prefix import maybe_prefix
from runtime.model._weight_loading import default_weight_loader
from runtime.model.laguna_decoder import LagunaDecoderLayerSelfBuilt
from runtime.model.plain_embedding import PlainEmbedding, PlainLMHead
from runtime.model.plain_linear import PlainLinear


def _draft_hf_config(config: DSparkDraftConfig) -> SimpleNamespace:
    """Adapt the validated JSON fields to the existing self-built decoder."""

    return SimpleNamespace(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        vocab_size=config.vocab_size,
        max_position_embeddings=262144,
        layer_types=["full_attention"] * config.num_hidden_layers,
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


class VanillaMarkov(nn.Module):
    """The official DSpark vanilla Markov head, TP=1."""

    def __init__(self, *, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        self.markov_w1 = PlainEmbedding(vocab_size, markov_rank)
        self.markov_w2 = PlainLinear(markov_rank, vocab_size, bias=False)

    def compute_step_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.markov_w1(token_ids.long()))


class DSparkConfidenceHead(nn.Module):
    """Confidence projection present in the official Qwen3.8 draft."""

    def __init__(self, *, hidden_size: int, markov_rank: int, with_markov: bool = True) -> None:
        super().__init__()
        self.with_markov = bool(with_markov)
        input_size = hidden_size + (markov_rank if self.with_markov else 0)
        self.proj = PlainLinear(input_size, 1, bias=True)
        self.register_buffer(
            "sts_temperatures", torch.ones((), dtype=torch.float32), persistent=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        markov_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.with_markov:
            if markov_embeddings is None:
                raise ValueError("DSpark confidence head requires Markov embeddings")
            features = torch.cat([hidden_states, markov_embeddings.to(hidden_states.dtype)], dim=-1)
        else:
            features = hidden_states
        return self.proj(features).squeeze(-1)

    def apply_sts(
        self, confidence_raw: torch.Tensor, *, position: int | None = None
    ) -> torch.Tensor:
        temperatures = self.sts_temperatures
        if position is not None and temperatures.numel() > 1:
            temperatures = temperatures[position]
        return torch.sigmoid(confidence_raw.float() / temperatures)


class Qwen36DSparkDraftModel(nn.Module):
    """Dense DFlash backbone plus target-context projection."""

    def __init__(self, config: DSparkDraftConfig, *, target_layer_count: int) -> None:
        super().__init__()
        self.draft_config = config
        decoder_config = _draft_hf_config(config)
        self.layers = nn.ModuleList(
            [
                LagunaDecoderLayerSelfBuilt(
                    config=decoder_config,
                    cache_config=None,
                    quant_config=None,
                    prefix=maybe_prefix("model", f"layers.{i}"),
                    layer_idx=i,
                    # The target and draft share one static attention context;
                    # offset draft layer names so discovery cannot collide.
                    attention_prefix=maybe_prefix("model", f"layers.{i + target_layer_count}"),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = PlainLinear(
            config.hidden_size * len(config.target_layer_ids), config.hidden_size, bias=False
        )
        self.hidden_norm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @property
    def num_context_features(self) -> int:
        return len(self.draft_config.target_layer_ids)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Project concatenated target taps into draft hidden space."""

        expected = self.fc.input_size
        if target_hidden.ndim != 2 or target_hidden.shape[-1] != expected:
            raise ValueError(
                "DSpark target_hidden feature dim mismatch: expected [N, "
                f"{expected}], got {tuple(target_hidden.shape)}"
            )
        # The Q6 GGUF target keeps correctness-critical taps in F32 while the
        # official DFlash2 draft weights are BF16.  This projector is the
        # deliberate precision boundary between the two models; make it
        # explicit instead of relying on F.linear to reject mixed dtypes.
        if target_hidden.dtype != self.fc.weight.dtype:
            target_hidden = target_hidden.to(dtype=self.fc.weight.dtype)
        return self.hidden_norm(self.fc(target_hidden))

    def forward(self, inputs_embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if inputs_embeds.ndim == 3:
            batch, seq_len, hidden = inputs_embeds.shape
            hidden_states = inputs_embeds.reshape(batch * seq_len, hidden)
            if positions.ndim == 2:
                positions = positions.reshape(-1)
        elif inputs_embeds.ndim == 2:
            hidden_states = inputs_embeds
        else:
            raise ValueError(
                f"DSpark draft inputs_embeds must be rank 2 or 3, got {inputs_embeds.ndim}"
            )
        residual: torch.Tensor | None = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        if residual is None:
            return self.norm(hidden_states)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    # -- Context KV projection -----------------------------------------

    def _build_fused_kv_buffers(self) -> None:
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        self._kv_weights = torch.stack(
            [attn.qkv_proj.weight[attn.q_size :] for attn in layers_attn], dim=0
        ).contiguous()
        self._kv_biases: torch.Tensor | None = None
        if attn0.qkv_proj.bias is not None:
            self._kv_biases = torch.stack(
                [attn.qkv_proj.bias[attn.q_size :] for attn in layers_attn], dim=0
            ).contiguous()
        self._k_norm_weights = torch.stack(
            [attn.k_norm.weight.data for attn in layers_attn], dim=0
        ).contiguous()
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        if not attn0.rotary_emb.is_neox_style:
            raise ValueError("Qwen DSpark draft requires NeoX-style RoPE")
        for attn in layers_attn[1:]:
            if (
                attn.rotary_emb.head_size != self._rope_head_size
                or not attn.rotary_emb.is_neox_style
            ):
                raise ValueError("all Qwen DSpark draft layers must share RoPE geometry")
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.eps
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]

    def _project_context_kv(
        self, context_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project normalized context states into every draft layer."""

        num_ctx = context_states.shape[0]
        context_states = context_states.unsqueeze(0).expand(self._num_attn_layers, -1, -1)
        all_kv_flat = torch.bmm(context_states, self._kv_weights.transpose(1, 2))
        if self._kv_biases is not None:
            all_kv_flat = all_kv_flat + self._kv_biases[:, None, :]
        return (
            all_kv_flat.view(self._num_attn_layers, num_ctx, 2, self._num_kv_heads, self._head_dim)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )

    def precompute_and_store_context_kv(
        self,
        target_hidden: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Project target taps and write draft K/V at their absolute positions."""

        if not hasattr(self, "_num_attn_layers"):
            self._build_fused_kv_buffers()
        context_states = self.project_target_hidden(target_hidden)
        all_k, all_v = self._project_context_kv(context_states)
        normed_k = torch.empty_like(all_k)
        for layer_idx in range(self._num_attn_layers):
            normed_k[layer_idx] = rms_norm(
                all_k[layer_idx], self._k_norm_weights[layer_idx], self._rms_norm_eps
            )
        all_k_flat = normed_k.view(self._num_attn_layers * context_states.shape[0], self._kv_size)
        apply_rotary_embedding_inplace(
            context_positions.repeat(self._num_attn_layers),
            all_k_flat,
            self._rope_head_size,
            self._rope_cos_sin_cache,
        )
        if context_slot_mapping is None:
            return
        all_k_final = all_k_flat.view(
            self._num_attn_layers,
            context_states.shape[0],
            self._num_kv_heads,
            self._head_dim,
        )
        per_layer = isinstance(context_slot_mapping, (list, tuple))

        for layer_idx, attn in enumerate(self._attn_layers):
            mapping = context_slot_mapping[layer_idx] if per_layer else context_slot_mapping
            if mapping is None:
                continue
            attn.impl.do_kv_cache_update(
                attn, all_k_final[layer_idx], all_v[layer_idx], attn.kv_cache, mapping
            )

    # -- Weight loading ------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        stacked = (
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        )
        for name, tensor in weights:
            for param_name, weight_name, shard_id in stacked:
                if f".{weight_name}." not in f".{name}":
                    continue
                mapped = name.replace(weight_name, param_name).removeprefix("model.")
                param = params.get(mapped)
                if param is None:
                    continue
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, tensor, shard_id)
                loaded.add(f"model.{mapped}")
                break
            else:
                mapped = name.removeprefix("model.")
                param = params.get(mapped)
                if param is None:
                    continue
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, tensor)
                loaded.add(f"model.{mapped}")
        return loaded


class Qwen36DSparkDraftForCausalLM(nn.Module):
    """DSpark draft wrapper with shared target embedding/LM head."""

    def __init__(self, config: DSparkDraftConfig, *, target_layer_count: int) -> None:
        super().__init__()
        self.config = config
        self.gamma = config.block_size
        self.embed_tokens = PlainEmbedding(config.vocab_size, config.hidden_size)
        self.lm_head = PlainLMHead(config.vocab_size, config.hidden_size)
        self.model = Qwen36DSparkDraftModel(config, target_layer_count=target_layer_count)
        self.markov_head = VanillaMarkov(
            vocab_size=config.vocab_size, markov_rank=config.markov_rank
        )
        self.confidence_head = (
            DSparkConfidenceHead(
                hidden_size=config.hidden_size,
                markov_rank=config.markov_rank,
                with_markov=config.confidence_head_with_markov,
            )
            if config.confidence_head
            else None
        )

    def attach_shared_modules(self, *, embed_tokens: nn.Module, lm_head: nn.Module) -> None:
        del self.embed_tokens, self.lm_head
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        return self.model(inputs_embeds, positions)

    def compute_base_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The production Qwen3.8 target shares its NVFP4/FP8 lm_head with the
        # draft.  Calling F.linear on that raw FP8 storage is not supported by
        # CUDA; the target module's forward path performs the required
        # dequantization or scaled matmul.  Keep the plain test head fallback
        # so the draft remains usable without a full target model.
        if not isinstance(self.lm_head, PlainLMHead):
            return self.lm_head(hidden_states)
        if hidden_states.dtype != self.lm_head.weight.dtype:
            hidden_states = hidden_states.to(self.lm_head.weight.dtype)
        return F.linear(hidden_states, self.lm_head.weight)

    def markov_logits(
        self, base_logits: torch.Tensor, *, previous_tokens: torch.Tensor
    ) -> torch.Tensor:
        return base_logits + self.markov_head.compute_step_bias(previous_tokens)

    def sample_greedy_block(
        self, hidden_states: torch.Tensor, *, anchor_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply the Markov recurrence and greedily produce one DSpark block."""

        return self.sample_block(
            hidden_states,
            anchor_tokens=anchor_tokens,
            sampler=lambda logits, _step: logits.argmax(dim=-1),
        )

    def sample_block(
        self,
        hidden_states: torch.Tensor,
        *,
        anchor_tokens: torch.Tensor,
        sampler: Callable[[torch.Tensor, int], torch.Tensor],
        capture_confidence: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run the DSpark Markov recurrence with a caller-provided sampler.

        SGLang keeps the recurrence independent from greedy versus sampled
        token selection.  Keeping that same boundary here matters because
        speculative rejection sampling needs the corrected draft logits, not
        just the chosen token ids.
        """

        if hidden_states.ndim != 3:
            raise ValueError(f"DSpark block hidden states must be rank 3, got {hidden_states.ndim}")
        base_logits = self.compute_base_logits(hidden_states)
        outputs: list[torch.Tensor] = []
        corrected: list[torch.Tensor] = []
        confidence: list[torch.Tensor] = []
        previous = anchor_tokens.long()
        for step in range(hidden_states.shape[1]):
            markov_embedding = self.markov_head.markov_w1(previous)
            logits = base_logits[:, step, :] + self.markov_head.markov_w2(markov_embedding)
            token = sampler(logits, step)
            if token.shape != previous.shape:
                raise ValueError(
                    "DSpark sampler must return one token per batch row; "
                    f"expected {tuple(previous.shape)}, got {tuple(token.shape)}"
                )
            outputs.append(token)
            corrected.append(logits.unsqueeze(1))
            if capture_confidence and self.confidence_head is not None:
                confidence.append(
                    self.confidence_head(
                        hidden_states[:, step, :],
                        markov_embedding if self.confidence_head.with_markov else None,
                    ).unsqueeze(1)
                )
            previous = token
        # SGLang exposes calibrated acceptance probabilities to its verify
        # planner, not the raw confidence-head logits.  Keep this conversion
        # in the draft model so eager and CUDA-Graph proposal paths feed the
        # same quantity into compact scheduling.
        confidence_out = (
            self.confidence_head.apply_sts(torch.cat(confidence, dim=1))
            if confidence and self.confidence_head is not None
            else None
        )
        return torch.stack(outputs, dim=1), torch.cat(corrected, dim=1), confidence_out

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
        """Load the flat official checkpoint names into model/heads."""

        params = dict(self.named_parameters())
        loaded: set[str] = set()
        backbone: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            if name.startswith(("embed_tokens.", "lm_head.", "rotary_emb.")):
                continue
            if name.startswith(("markov_head.", "confidence_head.")):
                param = params.get(name)
                if param is None:
                    continue
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, tensor)
                loaded.add(name)
            else:
                backbone.append((name, tensor))
        loaded.update(self.model.load_weights(backbone))
        return loaded
