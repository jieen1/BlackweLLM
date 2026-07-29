"""Self-built DFlash draft model -- Phase 3 of the vLLM removal plan.

Replaces vLLM's ``DFlashLagunaForCausalLM``/``DFlashLagunaModel``
(vllm/model_executor/models/laguna_dflash.py, inheriting shared algorithm
methods from ``DFlashQwen3Model`` in vllm/model_executor/models/
qwen3_dflash.py) with a direct equivalent built on the same self-built
building blocks Phase 1/2 already validated for the main model
(``LagunaDecoderLayerSelfBuilt``, ``TritonRMSNorm``, ``PlainLinear``).

Read before writing, not assumed:
- The real draft checkpoint (models--poolside--Laguna-S-2.1-DFlash-NVFP4)
  has NO ``quantization_config`` at all -- every weight is plain BF16.
  6 layers, all ``sliding_attention``, ``num_experts=0`` (dense MLP only,
  no MoE branch ever taken) -- so every projection in every draft layer
  is a ``PlainLinear``/``LagunaDecoderLayerSelfBuilt`` with
  ``quant_config=None``, no NVFP4 concern anywhere in this file.
- Weight names in the checkpoint (verified via direct safetensors
  inspection) are ``layers.N.{self_attn.*,mlp.*,input_layernorm,
  post_attention_layernorm}``, ``aux_hidden_norms.{0..5}.weight``,
  ``fc.weight``, ``hidden_norm.weight``, ``norm.weight`` -- structurally
  identical to the main model's per-layer naming (reuses
  ``LagunaDecoderLayerSelfBuilt`` unmodified), plus the DFlash-specific
  top-level extras handled directly in this file. No ``embed_tokens.*``/
  ``lm_head.*`` keys -- shared with the target model (see
  ``runtime/model_loading.py::load_laguna_dflash_draft_model``).
- ``dflash_config.causal=True``, ``target_layer_ids=[1,10,19,29,38,47]``
  (0-indexed, "after this main-model layer"), matching
  ``dflash_constants.AUX_LAYER_IDS=[2,11,20,30,39,48]`` (vLLM's
  post-layer/1-indexed convention == target_layer_ids[i]+1).

Two details that would have been silent, hard-to-diagnose bugs if this
file had reused ``LagunaDecoderLayerSelfBuilt`` naively without reading
vLLM's actual ``DFlashLagunaModel.__init__``/``_build_context_kv_buffers``/
``_project_context_kv`` (vllm/model_executor/models/laguna_dflash.py):

1. **Attention layer naming offset.** vLLM constructs each draft layer's
   underlying ``Attention`` op with
   ``attention_prefix=maybe_prefix(prefix, f"layers.{i + target_layer_count}")``
   (target_layer_count=48) instead of the layer's own prefix -- so draft
   layer 0's ``Attention`` registers in ``static_forward_context`` as
   ``"...layers.48.attn"``, not ``"...layers.0.attn"`` (which is already
   the MAIN model's layer 0). ``runtime/backends/laguna_dflash.py``'s
   ``_alloc_draft_kv_cache`` (unchanged, existing code) discovers draft
   attention layers via ``layer_idx >= 48`` parsed from this exact name.
   Getting this offset wrong would silently collide draft and main-model
   attention op registrations. ``_build_draft_layers`` below replicates it.
2. **Laguna's ``_project_context_kv`` override is NOT the ``DFlashQwen3Model``
   base-class version.** The Qwen3 base class normalizes the combined
   context state ONCE with a single shared ``hidden_norm`` before one big
   fused GEMM. Laguna's own override (which is what actually runs, since
   ``DFlashLagunaModel`` in vLLM defines its own ``_build_context_kv_buffers``/
   ``_project_context_kv``/``_normalize_context_k``) instead applies EACH
   layer's own ``input_layernorm`` weight to the (already ``hidden_norm``'d,
   from ``combine_hidden_states``) context state before that layer's own
   K/V projection, via a batched ``bmm`` over per-layer weights -- not a
   single shared norm. This is NOT double-normalization by mistake; it is
   deliberately model-specific and ported verbatim below, not "fixed" to
   match the more general base-class version.

RoPE: the fused, multi-layer, K-only, in-place rotation
(``runtime/kernels/rope.py::apply_rotary_embedding_inplace``) is the one
genuinely new kernel in this runtime -- validated bit-exact against
vLLM's own ``torch.ops._C.rotary_embedding`` for both full- and
partial-rotary configurations before being wired in here. It only
replaces the *application* of rotation to a pre-existing ``cos_sin_cache``;
constructing that cache (YaRN scaling etc.) still goes through vLLM's
``get_rope()`` inside ``LagunaAttentionSelfBuilt`` (Phase 2, unchanged) --
each draft layer's own ``rotary_emb`` module is reused only to read its
already-built ``cos_sin_cache``/``head_size`` buffers, asserted uniform
across layers exactly like vLLM's reference does.

The grouped per-layer RMSNorm calls in vLLM's reference (a single vLLM
kernel call with a ``[L, ...]``-shaped weight, broadcasting by the
tensor's outermost dim -- confirmed by reading
csrc/libtorch_stable/layernorm_kernels.cu, not guessed) are ported here
as a plain Python loop over L=6 calling the already-validated
``fused_rms_norm.rms_norm`` once per layer -- semantically identical
(each layer's norm only ever depends on that layer's own weight and the
underlying data), and L=6 makes the extra kernel launches cheap enough
that a second fused kernel isn't worth the added risk here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from runtime.kernels.fused_rms_norm import TritonRMSNorm, rms_norm
from runtime.kernels.rope import apply_rotary_embedding_inplace
from runtime.model._prefix import maybe_prefix
from runtime.model._weight_loading import default_weight_loader
from runtime.model.laguna_decoder import LagunaDecoderLayerSelfBuilt
from runtime.model.plain_embedding import PlainEmbedding, PlainLMHead, PlainLogitsProcessor
from runtime.model.plain_linear import PlainLinear


class LagunaDraftModelSelfBuilt(nn.Module):
    """Self-built equivalent of vLLM's ``DFlashLagunaModel``."""

    def __init__(self, *, runtime_config: Any, prefix: str = "") -> None:
        super().__init__()
        self.config = runtime_config.speculative_config.draft_model_config.hf_config
        config = self.config
        self.vocab_size = config.vocab_size

        dflash_config = getattr(config, "dflash_config", None) or {}
        target_layer_ids = dflash_config.get("target_layer_ids")
        if not target_layer_ids:
            raise ValueError(
                "Laguna DFlash config requires non-empty `dflash_config.target_layer_ids`."
            )

        # Placeholder -- replaced with the target model's embed_tokens object
        # (not a weight copy) by load_laguna_dflash_draft_model. The real
        # checkpoint has no embed_tokens.* key at all.
        self.embed_tokens = PlainEmbedding(config.vocab_size, config.hidden_size)

        self.mask_token_id = dflash_config.get("mask_token_id")
        # Buffer, not a Parameter -- matches vLLM's own DFlashLagunaModel
        # exactly (register_buffer(..., persistent=False), not
        # nn.Parameter). Getting this wrong (an earlier version of this
        # file used nn.Parameter) makes it show up in named_parameters()
        # and fail the "every parameter must receive a checkpoint tensor"
        # loader assertion, even though this checkpoint legitimately has
        # no mask_embedding key at all (has_separate_mask_embedding stays
        # False below) -- caught via real GPU validation, not by reading
        # the vLLM source carefully enough the first time.
        self.register_buffer("mask_embedding", torch.zeros(config.hidden_size), persistent=False)
        # No `has_separate_mask_embedding`-triggering tensor in the real
        # checkpoint (verified: no `mask_embedding` key in the safetensors
        # file) -- stays False, embed_input_ids falls through to the plain
        # embedding lookup. Kept as a real (if currently inert) flag rather
        # than deleted, matching vLLM's own conditional exactly.
        self.has_separate_mask_embedding = False

        target_layer_count = runtime_config.model_config.get_num_layers(
            runtime_config.parallel_config
        )
        self.layers = nn.ModuleList(
            [
                LagunaDecoderLayerSelfBuilt(
                    config=config,
                    cache_config=runtime_config.cache_config,
                    quant_config=None,  # draft checkpoint is entirely unquantized
                    prefix=maybe_prefix(prefix, f"layers.{i}"),
                    layer_idx=i,
                    # Offset so static_forward_context names don't collide
                    # with the target model's own layers 0..47 -- see
                    # module docstring point 1.
                    attention_prefix=maybe_prefix(prefix, f"layers.{i + target_layer_count}"),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        for layer in self.layers:
            # DFlash injects verifier-context K/V at absolute cache slots;
            # the ring-sized KV allocation is still the only capacity limit.
            # Disabling the attention op's own SWA compute-time mask avoids
            # it incorrectly excluding legitimately-relevant injected
            # context entries. Matches vLLM's DFlashLagunaModel.__init__.
            if getattr(layer.self_attn, "sliding_window", None) is not None:
                layer.self_attn.attn.sliding_window = None

        num_features = len(target_layer_ids)
        self.num_aux_slices = num_features
        target_hidden_size = runtime_config.model_config.get_hidden_size()
        fc_input_size = target_hidden_size * num_features
        self.aux_hidden_norms = nn.ModuleList(
            [
                TritonRMSNorm(fc_input_size // num_features, eps=config.rms_norm_eps)
                for _ in range(num_features)
            ]
        )
        self.fc = PlainLinear(fc_input_size, config.hidden_size, bias=False)
        self.hidden_norm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids)
        if self.has_separate_mask_embedding and self.mask_token_id is not None:
            is_mask = (input_ids == self.mask_token_id).unsqueeze(-1)
            embeds = torch.where(is_mask, self.mask_embedding.to(embeds.dtype), embeds)
        return embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        )
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    # -- DFlash context-KV precompute -----------------------------------

    def _build_fused_kv_buffers(self) -> None:
        """One-time (post weight-load) fused-buffer construction.

        Must run after embed_tokens/lm_head sharing and weight loading.
        Ported from vLLM's DFlashQwen3Model._build_fused_kv_buffers +
        DFlashLagunaModel._build_context_kv_buffers (the Laguna-specific
        override -- see module docstring point 2).
        """
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        has_bias = attn0.qkv_proj.bias is not None

        self._kv_weights = torch.stack(
            [a.qkv_proj.weight[a.q_size :] for a in layers_attn], dim=0
        ).contiguous()
        if has_bias:
            self._kv_biases: torch.Tensor | None = torch.stack(
                [a.qkv_proj.bias[a.q_size :] for a in layers_attn], dim=0
            ).contiguous()
        else:
            self._kv_biases = None
        self._input_layernorm_weights = torch.stack(
            [layer.input_layernorm.weight.data for layer in self.layers], dim=0
        ).contiguous()
        self._k_norm_weights = torch.stack(
            [a.k_norm.weight.data for a in layers_attn], dim=0
        ).contiguous()

        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        assert attn0.rotary_emb.is_neox_style, (
            "runtime.kernels.rope only implements NeoX-style rotation"
        )
        for attn in layers_attn[1:]:
            assert (
                attn.rotary_emb.head_size == self._rope_head_size and attn.rotary_emb.is_neox_style
            ), "All draft layers must share RoPE parameters for context-KV precompute"

        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.eps
        for attn in layers_attn[1:]:
            assert (
                attn.kv_size == self._kv_size
                and attn.head_dim == self._head_dim
                and attn.num_kv_heads == self._num_kv_heads
                and attn.q_norm.eps == self._rms_norm_eps
            ), "All draft layers must share attention config for context-KV precompute"

        self._attn_layers = [layer.self_attn.attn for layer in self.layers]

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = context_states.shape[-1]
        normed_context_states = torch.empty(
            (num_layers, num_ctx, hidden),
            dtype=context_states.dtype,
            device=context_states.device,
        )
        for layer_idx in range(num_layers):
            normed_context_states[layer_idx] = rms_norm(
                context_states, self._input_layernorm_weights[layer_idx], self._rms_norm_eps
            )
        all_kv_flat = torch.bmm(normed_context_states, self._kv_weights.transpose(1, 2))
        if self._kv_biases is not None:
            all_kv_flat = all_kv_flat + self._kv_biases[:, None, :]
        all_kv = (
            all_kv_flat.view(num_layers, num_ctx, 2, num_kv_heads, head_dim)
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )
        return all_kv[0], all_kv[1]

    def _normalize_context_k(self, all_k: torch.Tensor) -> torch.Tensor:
        num_layers = all_k.shape[0]
        out = torch.empty_like(all_k)
        for layer_idx in range(num_layers):
            out[layer_idx] = rms_norm(
                all_k[layer_idx], self._k_norm_weights[layer_idx], self._rms_norm_eps
            )
        return out

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        if not hasattr(self, "_num_attn_layers"):
            self._build_fused_kv_buffers()

        num_ctx = context_states.shape[0]
        num_layers = self._num_attn_layers
        kv = self._kv_size
        head_dim = self._head_dim
        num_kv_heads = self._num_kv_heads

        all_k, all_v = self._project_context_kv(
            context_states, num_ctx, num_layers, num_kv_heads, head_dim
        )
        all_k_normed = self._normalize_context_k(all_k)

        all_k_flat = all_k_normed.view(num_layers * num_ctx, kv)
        positions_repeated = context_positions.repeat(num_layers)
        apply_rotary_embedding_inplace(
            positions_repeated, all_k_flat, self._rope_head_size, self._rope_cos_sin_cache
        )

        if context_slot_mapping is None:
            return

        all_k_final = all_k_flat.view(num_layers, num_ctx, num_kv_heads, head_dim)
        per_layer = isinstance(context_slot_mapping, (list, tuple))
        for i in range(num_layers):
            slot_mapping = context_slot_mapping[i] if per_layer else context_slot_mapping
            if slot_mapping is None:
                continue
            attn = self._attn_layers[i]
            kv_cache = attn.kv_cache
            attn.impl.do_kv_cache_update(attn, all_k_final[i], all_v[i], kv_cache, slot_mapping)

    # -- Weight loading ---------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """No MoE (num_experts=0, always dense) and no attention_sink in the
        draft config -- simpler than LagunaModelSelfBuilt.load_weights,
        just the QKV stacked-shard mapping plus a direct-copy fallback for
        everything else (fc/hidden_norm/norm/aux_hidden_norms.*, all plain
        single-shard params).
        """
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader is default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params


class LagunaDraftForCausalLMSelfBuilt(nn.Module):
    """Self-built equivalent of vLLM's ``DFlashLagunaForCausalLM``."""

    def __init__(self, *, runtime_config: Any, prefix: str = "") -> None:
        super().__init__()
        self.config = runtime_config.speculative_config.draft_model_config.hf_config
        config = self.config
        if getattr(config, "draft_vocab_size", None) is None:
            raise ValueError("Laguna DFlash config requires `draft_vocab_size`.")

        target_vocab_size = runtime_config.model_config.get_vocab_size()
        if config.draft_vocab_size != target_vocab_size:
            raise ValueError(
                "Laguna DFlash shares the target lm_head and requires "
                "`draft_vocab_size` to match the target vocabulary size "
                f"({config.draft_vocab_size} != {target_vocab_size})."
            )

        # Both unconditionally shared with the target model -- see
        # runtime/model_loading.py::load_laguna_dflash_draft_model.
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False

        self.model = LagunaDraftModelSelfBuilt(
            runtime_config=runtime_config, prefix=maybe_prefix(prefix, "model")
        )
        # Placeholder -- replaced with the target model's lm_head object by
        # load_laguna_dflash_draft_model. No lm_head.* key in the checkpoint.
        self.lm_head = PlainLMHead(config.draft_vocab_size, config.hidden_size)
        self.logits_processor = PlainLogitsProcessor(config.draft_vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Normalize each verifier hidden-state slice independently, then
        # concatenate and project into the drafter hidden size used as
        # DFlash context. Ported verbatim from vLLM's
        # DFlashLagunaForCausalLM.combine_hidden_states.
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        num_slices = self.model.num_aux_slices
        slice_size = hidden_states.shape[-1] // num_slices
        slices = hidden_states.view(hidden_states.shape[0], num_slices, slice_size)
        normed = torch.empty_like(slices)
        for i, norm in enumerate(self.model.aux_hidden_norms):
            normed[:, i, :] = norm(slices[:, i, :])
        hidden_states = normed.reshape(hidden_states.shape[0], -1)
        result = self.model.fc(hidden_states)
        result = self.model.hidden_norm(result)
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    # -- SupportsEagle3-shaped surface (see vLLM's interfaces.py) --------
    # Not exercised by this runtime's hand-rolled DFlashEngine (which never
    # goes through vLLM's generic spec-decode dispatch), kept for interface
    # parity / potential future reuse. The draft model itself never needs
    # its own aux hidden states -- DFlash's aux states come from the target
    # model (see runtime/model/laguna_model.py, already Phase-1 complete).
    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        dflash_config = getattr(self.config, "dflash_config", None) or {}
        target_layer_ids = dflash_config.get("target_layer_ids") or ()
        return tuple(i + 1 for i in target_layer_ids)

    # -- Weight loading ---------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Checkpoint keys have no top-level prefix at all (verified: raw
        keys are `layers.*`/`fc.weight`/etc, no `model.`/`lm_head.` -- the
        checkpoint has no lm_head weights, shared with the target instead).
        So this is a direct pass-through to the inner model's loader,
        unlike LagunaForCausalLMSelfBuilt.load_weights (which does have a
        real model./lm_head. split to handle for the main checkpoint).
        """
        loaded = self.model.load_weights(weights)
        return {f"model.{n}" for n in loaded}
