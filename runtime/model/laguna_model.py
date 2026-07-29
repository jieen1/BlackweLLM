"""Self-built Laguna model graph — Phase 1 of the vLLM removal plan.

Replaces vLLM's ``get_model()`` orchestration (registry lookup, model
construction inside ``set_current_vllm_config``, weight loading via
``AutoWeightsLoader``/``DefaultModelLoader``) with a direct, hand-written
equivalent scoped to exactly what this runtime needs: a single model
(Laguna), single pipeline-parallel rank, no LoRA, no online quantization.

Deliberately still reuses vLLM's building blocks for everything this phase
is NOT scoped to touch (see notes/2026-07-27-vllm-complete-removal-
implementation-plan.md, "阶段1"/"阶段2"):

- ``LagunaDecoderLayer``/``LagunaAttention``/``LagunaMoE``/``LagunaMLP`` are
  now self-built (``runtime/model/laguna_decoder.py``,
  ``LagunaDecoderLayerSelfBuilt`` and friends) -- Phase 2. The one deliberate
  exception is ``LagunaMoESelfBuilt.experts``, which stays vLLM's
  ``FusedMoE``: the checkpoint's NVFP4 quantization applies ONLY to routed
  MoE expert weights (verified against the real config.json, not assumed --
  see laguna_decoder.py's module docstring), and
  ``LagunaBackend._patch_moe_sparkinfer`` already loads those weights
  directly from safetensors and discards vLLM's copy before any inference
  runs. Reimplementing FusedMoE's expert-parallel dispatch is out of scope
  for a Linear/Embedding phase.
- vLLM's per-layer ``RMSNorm(CustomOp)`` instances inside the decoder layer
  are now also ``TritonRMSNorm`` (same class as the top-level final norm
  below) -- ``_patch_rmsnorm_triton``'s double-patch trick is only still
  needed for paths not yet migrated (e.g. DirectModelRunner).
- ``VocabParallelEmbedding``/``ParallelLMHead``/``LogitsProcessor`` are now
  self-built too (阶段6, ``runtime/model/plain_embedding.py``) --
  ``PlainEmbedding``/``PlainLMHead``/``PlainLogitsProcessor``. Traced
  through vLLM's real ``LogitsProcessor``/``UnquantizedEmbeddingMethod``
  source first: at this runtime's actual values (TP=1, no LoRA, Laguna's
  vocab_size already a multiple of the default padding alignment, no
  soft_cap/scale ever configured), the whole TP-sharding/vocab-padding/
  gather machinery is provably a no-op, and ``LogitsProcessor.forward``
  reduces to one plain ``F.linear`` call -- see that module's docstring
  for the exact trace.
- vLLM's NVFP4 ``process_weights_after_loading`` is still called (via the
  loader in ``runtime/model_loading.py``) for ``LagunaMoESelfBuilt.experts``
  only, for the same FusedMoE reason above -- it runs, produces weights that
  are immediately discarded by the sparkinfer patch, and is otherwise inert.

What IS new here, versus vLLM's ``vllm/model_executor/models/laguna.py``:
- No ``@support_torch_compile`` decorator (confirmed dead code in
  production: every real forward call site passes ``skip_compiled=True``).
- No pipeline-parallel branches (this runtime is PP=1, always; the
  ``get_pp_group().is_first_rank``/``is_last_rank`` checks in vLLM's
  version are permanently-true dead branches here).
- No ``IntermediateTensors``/``inputs_embeds`` plumbing (PP-only paths).
- Registry-free construction: callers import ``LagunaForCausalLMSelfBuilt``
  directly instead of going through vLLM's ``get_model_architecture()``
  dict lookup.

Migration invariant (same as compat_vllm.py's B7-V1 contract): swapping
``get_model()`` for this module must preserve bit-level parity on the
greedy fixed-prompt suite. This has NOT been GPU-validated yet -- see the
handoff note in this phase's final report. Do not wire this into
``LagunaBackend`` as the default path until that validation lands.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from runtime.kernels.fused_rms_norm import TritonRMSNorm
from runtime.model._prefix import maybe_prefix as _maybe_prefix
from runtime.model._weight_loading import default_weight_loader, remap_kv_scale_name
from runtime.model.laguna_decoder import LagunaDecoderLayerSelfBuilt
from runtime.model.plain_embedding import PlainEmbedding, PlainLMHead, PlainLogitsProcessor


class LagunaModelSelfBuilt(nn.Module):
    """Self-built equivalent of vLLM's ``LagunaModel``.

    Same forward dataflow (embed -> N decoder layers -> final norm), minus
    the @support_torch_compile decoration and PP branches vLLM's version
    carries. See module docstring for the full list of what's intentionally
    still borrowed from vLLM.
    """

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        # Same rationale as vLLM's LagunaModel.__init__: Laguna drives SWA
        # per-layer via layer_types + per-layer window_left, not via the
        # model-level cache_config.sliding_window fallback. Without this,
        # global (full-attention) layers would silently pick up a 512-token
        # window from config.sliding_window.
        if cache_config is not None:
            cache_config.sliding_window = None

        self.vocab_size = config.vocab_size

        self.embed_tokens = PlainEmbedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [
                LagunaDecoderLayerSelfBuilt(
                    config=config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                    enable_eplb=vllm_config.parallel_config.enable_eplb,
                    layer_idx=layer_idx,
                    max_model_len=vllm_config.model_config.max_model_len,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # Final norm: we own this module's construction, so skip vLLM's
        # RMSNorm(CustomOp) class entirely -- go straight to the same
        # Triton kernel fused_rms_norm.py already provides, no CustomOp
        # dispatch_forward()/double-patch dance needed for it.
        self.norm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # EagleModelMixin equivalent (DFlash/Eagle3 aux hidden states).
        # Kept as plain attributes/methods here rather than a mixin class --
        # this is the entire mixin's real logic (see interfaces.py
        # EagleModelMixin, ~15 lines), no benefit to importing it separately.
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.aux_hidden_state_layers = layers

    def _maybe_add_hidden_state(
        self,
        aux_hidden_states: list[torch.Tensor],
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if layer_idx in self.aux_hidden_state_layers:
            value = hidden_states + residual if residual is not None else hidden_states
            aux_hidden_states.append(value)
        return aux_hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        residual = None

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(aux_hidden_states, layer_idx + 1, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load ``model.*``-scoped weights (prefix already stripped by caller).

        Line-for-line port of vLLM's ``LagunaModel.load_weights`` minus the
        pipeline-parallel ``is_pp_missing_parameter`` guards (dead checks
        at PP=1 -- always False, so the bodies they guard always run; kept
        removed rather than kept-but-inert to avoid importing vLLM's PP
        utils for no behavioral effect) and the expert_params_mapping-
        driven loading branch (任务#41): verified directly, not assumed,
        that it was 100% dead weight -- ``LagunaMoESelfBuilt`` has had no
        ``self.experts``/``routed_experts`` submodule since 阶段6 (its
        NVFP4 expert weights are loaded straight from the checkpoint by
        ``runtime/backends/laguna_sparkinfer_moe.py`` instead), so no
        parameter name that ``fused_moe_make_expert_params_mapping``
        could ever produce was in ``params_dict`` to match against.
        Checkpoint keys that used to (fail to) match through that branch
        (``mlp.experts.N.{gate,up,down}_proj.weight`` etc.) now just fall
        through to the same final ``if name not in params_dict: continue``
        below with an identical, silent no-op outcome -- the expert
        weight_scale/input_scale/e_score_correction_bias keys among them
        were already being caught earlier by ``ignore_suffixes``
        regardless (``_bias``/``_weight_scale``/``_input_scale``), so
        removing the branch changes nothing about which keys get loaded,
        only removes the dead machinery that used to (fail to) look at
        them.
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # gate_proj/up_proj stay separate Linears (see LagunaMLP) --
            # no merge entry needed. See notes/2026-07-27-vllm-complete-
            # removal-implementation-plan.md 阶段2 for why this must not
            # change: merging would collapse two independent per-Linear
            # NVFP4 global scales into one via .max(), losing precision.
        ]

        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_rank = 0  # TP=1 always in this runtime.

        for name, loaded_weight in weights:
            if "sink" in name:
                param = params_dict.get(name)
                if param is not None:
                    layer_heads_per_rank = param.shape[0]
                    layer_head_start = tp_rank * layer_heads_per_rank
                    narrow_weight = loaded_weight.narrow(0, layer_head_start, layer_heads_per_rank)
                    param.data.copy_(narrow_weight)
                    loaded_params.add(name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name and "shared_expert" not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)

                if mapped_name.endswith(ignore_suffixes) and mapped_name not in params_dict:
                    continue
                if mapped_name.endswith("scale"):
                    mapped_name = remap_kv_scale_name(mapped_name, params_dict)
                    if mapped_name is None:
                        continue
                if mapped_name not in params_dict:
                    continue

                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                break
            else:
                name = remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params


class LagunaForCausalLMSelfBuilt(nn.Module):
    """Self-built equivalent of vLLM's ``LagunaForCausalLM``.

    Not decorated, not PP-aware, no LoRA (``SupportsLoRA`` was never
    exercised by this runtime -- LagunaBackend never sets up a LoRA
    manager). Keeps the ``SupportsEagle3``-shaped surface
    (``set_aux_hidden_state_layers``, forward returning an
    ``(hidden_states, aux_hidden_states)`` tuple) because DFlash's
    ``_enable_aux_hidden_states`` (runtime/backends/laguna_dflash.py)
    hasattr-probes for these rather than requiring the vLLM Protocol.
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    }

    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.model = LagunaModelSelfBuilt(
            vllm_config=vllm_config, prefix=_maybe_prefix(prefix, "model")
        )

        self.lm_head = PlainLMHead(config.vocab_size, config.hidden_size)
        self.tie_word_embeddings = bool(config.tie_word_embeddings)
        if self.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = PlainLogitsProcessor(config.vocab_size)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.set_aux_hidden_state_layers(layers)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Hand-rolled replacement for vLLM's ``AutoWeightsLoader`` wrapper.

        vLLM's ``LagunaForCausalLM.load_weights`` is just
        ``AutoWeightsLoader(self, skip_prefixes=["lm_head."] if tied else
        None).load_weights(weights)`` -- a fully generic recursive
        module-tree walker. We don't need generic: there are exactly two
        top-level weight groups in this checkpoint (``model.*`` and
        ``lm_head.*``), and ``model.*`` already has a complete, hand-written
        loader (``LagunaModelSelfBuilt.load_weights``). So: split by prefix,
        strip it, delegate ``model.*`` to that loader, handle the (untied)
        ``lm_head.*`` case with the same per-parameter weight_loader pattern
        AutoWeightsLoader would have used generically.
        """
        model_weights: list[tuple[str, torch.Tensor]] = []
        lm_head_weights: list[tuple[str, torch.Tensor]] = []
        other: list[str] = []

        for name, tensor in weights:
            if name.startswith("model."):
                model_weights.append((name[len("model.") :], tensor))
            elif name.startswith("lm_head."):
                if self.tie_word_embeddings:
                    # Tied: lm_head shares model.embed_tokens's Parameter
                    # object already (see __init__). Loading a separate
                    # copy here would be redundant at best, and at worst
                    # overwrite the tied Parameter with a second, distinct
                    # tensor read from the checkpoint. Skip, matching
                    # vLLM's skip_prefixes=["lm_head."] behavior for the
                    # tied case.
                    continue
                lm_head_weights.append((name[len("lm_head.") :], tensor))
            else:
                other.append(name)

        if other:
            raise ValueError(
                f"LagunaForCausalLMSelfBuilt.load_weights: unrecognized top-level "
                f"weight prefix on {len(other)} tensor(s), e.g. {other[:3]!r}. "
                "Expected every checkpoint key to start with 'model.' or 'lm_head.'."
            )

        loaded = self.model.load_weights(iter(model_weights))
        loaded = {f"model.{n}" for n in loaded}

        if not self.tie_word_embeddings:
            params_dict = dict(self.lm_head.named_parameters())
            for name, loaded_weight in lm_head_weights:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(f"lm_head.{name}")

        return loaded
