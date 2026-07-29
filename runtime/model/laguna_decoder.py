"""Self-built Laguna decoder-layer graph -- Phase 2 of the vLLM removal plan.

Replaces vLLM's ``LagunaAttention``/``LagunaMLP``/``LagunaMoE``/
``LagunaDecoderLayer`` (vllm/model_executor/models/laguna.py) with direct
equivalents scoped to what this runtime actually needs: TP=1, no LoRA, no
pipeline parallelism.

Before writing this, the real checkpoint's quantization_config was read
(models--poolside--Laguna-S-2.1-NVFP4/.../config.json) rather than assumed.
Findings that shape every design decision below:

- ``ignore`` covers every ``self_attn.{q,k,v,o,g}_proj`` (all 48 layers),
  ``model.layers.0.mlp.{gate,up,down}_proj`` (the sole ``mlp_only_layers``
  entry), every ``mlp.gate`` (MoE router), and every
  ``mlp.shared_expert.{gate,up,down}_proj``. All of these are plain BF16
  on disk -- NOT NVFP4. They use ``PlainLinear`` (runtime/model/
  plain_linear.py), not ``NvFp4Linear``.
- ``config_groups.group_0.targets`` is exactly
  ``experts.[0-9]+.(gate_proj|up_proj|down_proj)`` -- NVFP4 quantization
  in this checkpoint applies ONLY to MoE routed-expert weights, i.e. only
  inside ``FusedMoE``.
- ``LagunaBackend._patch_moe_sparkinfer`` (runtime/backends/laguna.py)
  already loads those routed-expert weights directly from safetensors
  (runtime/backends/laguna_sparkinfer_moe.py, zero vLLM dependency), and
  replaces this module's ``forward`` entirely before any inference happens.

**2026-07-28 update (阶段6, revisiting the "keep FusedMoE" call from
Phase 2):** re-read what ``_patch_moe_sparkinfer`` actually reads off
vLLM's constructed ``FusedMoE``/``RoutedExperts`` object before this
class stopped constructing one. Every attribute it touched, traced
precisely (not from the naming, from the actual code):
``routed.w13_weight`` -- existence-checked, then immediately freed,
value never read. ``routed.w13_weight_scale``/``w13_weight_scale_2``/
``w2_weight_scale``/``w2_weight_scale_2`` -- freed, never read at all.
``experts_obj.e_score_correction_bias`` and ``routed.w13_input_scale``/
``w2_input_scale`` -- the only three values genuinely consumed (fed into
sparkinfer's ``prepare_sparkinfer_layer``). So constructing vLLM's
``FusedMoE`` bought this runtime nothing except three small tensors that
are just as easy to read straight from the checkpoint (verified against
a live GPU run of the FusedMoE-derived values before switching, see
``runtime/backends/laguna_sparkinfer_moe.py::load_moe_layer_
activation_gscales``/``load_moe_layer_e_score_correction_bias``) -- the
big NVFP4-packed weight tensors it loaded were 100% discarded overhead
regardless of TP/EP size, not "dead code at TP=1" as the original Phase 2
framing assumed (the actual waste has nothing to do with distributed vs.
single-GPU). ``self.experts`` and every FusedMoE/EPLB/TP/EP construction
argument are gone from this class as of this pass; ``_patch_moe_
sparkinfer`` now reads its three inputs directly from the checkpoint
instead of from a constructed-then-discarded FusedMoE instance.

Consequently ``NvFp4Linear`` (runtime/model/nvfp4_linear.py, built earlier
in Phase 2 under the wrong assumption that these four classes needed it)
still has no live call site in the Laguna production path.

Also folded in per the same "we own construction now" reasoning already
applied to the top-level norm in laguna_model.py: ``q_norm``/``k_norm``/
``input_layernorm``/``post_attention_layernorm`` all switch to
``TritonRMSNorm`` (fused_rms_norm.py's ``rms_norm`` flattens every
leading dim before the reduction, so the ``[..., num_heads, head_dim]``
per-head-norm call shape is exactly as safe as the 2D top-level-norm case
Phase 1 already validated bit-exact -- confirmed by reading the kernel,
not assumed). This removes the ``_patch_rmsnorm_triton`` double-patch
trick's remaining reason to exist for every RMSNorm below the top level,
though that patch function itself is untouched here (still needed until
DirectModelRunner/other paths are migrated too).
"""

from __future__ import annotations

import typing

import torch
import torch.nn.functional as F
from torch import nn

from runtime.kernels.fused_rms_norm import TritonRMSNorm
# Module-level RoPE cache sharing: layers with identical (rotary_dim,
# max_position, rope_type, rope_theta) share ONE cos_sin_cache tensor.
# vLLM's get_model() does this implicitly via its RotaryEmbedding registry;
# the self-built path must do it explicitly to avoid 54 separate allocations
# (2.84 GB waste at 262144 positions).
_ROPE_CACHE_REGISTRY: dict[tuple, torch.Tensor] = {}

from runtime.kernels.rope import (
    SelfBuiltRotaryEmbedding,
    compute_cos_sin_cache_default,
    compute_cos_sin_cache_yarn,
)
from runtime.model._prefix import maybe_prefix
from runtime.model.plain_attention import SelfBuiltAttentionPlaceholder
from runtime.model.plain_linear import PlainLinear

def _extract_layer_index(layer_name: str) -> int:
    """Narrowed port of vLLM's ``extract_layer_index`` (vllm/model_
    executor/models/utils.py) -- only the ``num_attn_module=1`` (default)
    behavior: pull the sole integer out of a dotted module path, e.g.
    ``"model.layers.3.self_attn"`` -> ``3``. Both real call sites below
    always use the default, never pass ``num_attn_module`` (verified by
    grep before dropping it) -- the real function's other branch (multi-
    int paths for MLA-style attention submodule nesting) is unreachable
    here and intentionally not ported.
    """
    int_vals = [int(part) for part in layer_name.split(".") if part.isdigit()]
    assert len(int_vals) == 1, f"layer name {layer_name!r} should contain exactly one integer"
    return int_vals[0]


class LagunaMLPSelfBuilt(nn.Module):
    """Dense MLP (only ``model.layers.0.mlp`` uses this in the real config --
    every other layer is MoE, see module docstring). gate_proj/up_proj kept
    as separate ``PlainLinear`` matching vLLM's ``LagunaMLP`` layout
    (unrelated to the NVFP4 global-scale merge concern that motivated the
    same split in ``NvFp4Linear`` -- this class is unquantized -- kept
    unstacked purely for structural parity with the vLLM reference and the
    checkpoint's own per-tensor key names)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        reduce_results: bool = True,  # unused at TP=1; kept for signature parity
        prefix: str = "",
    ) -> None:
        super().__init__()
        del reduce_results
        self.gate_proj = PlainLinear(hidden_size, intermediate_size, bias=False)
        self.up_proj = PlainLinear(hidden_size, intermediate_size, bias=False)
        self.down_proj = PlainLinear(intermediate_size, hidden_size, bias=False)
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class LagunaMoESelfBuilt(nn.Module):
    """Sparse MoE block. ``gate`` (router) and ``shared_expert`` are plain
    BF16 (self-built) per the checkpoint audit in the module docstring.

    No routed-expert weight container here at all (not FusedMoE, not a
    self-built replacement) -- LagunaBackend._patch_moe_sparkinfer loads
    the NVFP4 expert weights directly from safetensors and replaces
    forward() below before any inference happens, and (as of 阶段6) also
    reads e_score_correction_bias/the two activation global scales
    directly from the checkpoint instead of off a constructed FusedMoE
    instance -- see module docstring for what changed and why.
    """

    def __init__(
        self,
        config,
        quant_config: typing.Any = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        del quant_config, enable_eplb  # only ever needed for FusedMoE construction
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = float(
            getattr(config, "moe_routed_scaling_factor", 1.0)
        )

        # Router gate -- plain BF16 (checkpoint: "ignore": [..., "re:.*\\.mlp\\.gate$"]).
        self.gate = PlainLinear(config.hidden_size, config.num_experts, bias=False)

        self.shared_expert: LagunaMLPSelfBuilt | None
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = LagunaMLPSelfBuilt(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                reduce_results=False,
                prefix=maybe_prefix(prefix, "shared_expert"),
            )
        else:
            self.shared_expert = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "LagunaMoESelfBuilt.forward must be patched by "
            "LagunaBackend._patch_moe_sparkinfer before use -- sparkinfer "
            "owns all routed-expert compute in this runtime; this class "
            "only holds gate/shared_expert construction, never a real "
            "forward path of its own."
        )


class LagunaAttentionSelfBuilt(nn.Module):
    """Laguna attention with optional softplus output gating. TP=1 only --
    the ``tp_size``/rank arithmetic vLLM's version carries is dropped
    entirely (was always degenerate at TP=1: num_heads // 1, etc.)."""

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 131072,
        max_model_len: int | None = None,
        head_dim: int | None = None,
        cache_config: typing.Any = None,
        quant_config: typing.Any = None,
        prefix: str = "",
        attention_sink: bool = False,
        layer_idx: int | None = None,
        attention_prefix: str | None = None,
    ) -> None:
        super().__init__()
        if layer_idx is None:
            layer_idx = _extract_layer_index(prefix)
        if attention_prefix is None:
            attention_prefix = prefix
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.gating = config.gating

        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None:
            is_sliding = layer_types[layer_idx] == "sliding_attention"
            self.sliding_window = config.sliding_window if is_sliding else None
        else:
            self.sliding_window = None

        # Plain BF16 (checkpoint ignores every self_attn.{q,k,v,o,g}_proj).
        self.qkv_proj = PlainLinear(
            self.hidden_size,
            self.q_size + 2 * self.kv_size,
            shard_sizes=[self.q_size, self.kv_size, self.kv_size],
            bias=config.qkv_bias,
        )
        self.o_proj = PlainLinear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        if self.gating:
            gate_per_head = self.gating is True or self.gating == "per-head"
            g_out = self.num_heads if gate_per_head else self.num_heads * self.head_dim
            self.g_proj = PlainLinear(hidden_size, g_out, bias=False)
            self.gate_per_head = gate_per_head
        else:
            self.g_proj = None
            self.gate_per_head = False

        sinks = None
        if attention_sink:
            self.sink = torch.nn.Parameter(
                torch.empty(self.num_heads, requires_grad=False)
            )
            sinks = self.sink

        layer_type = (
            layer_types[layer_idx] if layer_types is not None else "full_attention"
        )
        is_sliding = layer_type == "sliding_attention"

        top_rope = getattr(config, "rope_parameters", None) or {}
        if any(isinstance(v, dict) for v in top_rope.values()):
            base_rope = top_rope.get(layer_type) or top_rope.get("full_attention") or {}
        else:
            base_rope = top_rope

        swa_rope = getattr(config, "swa_rope_parameters", None)
        rope_params = swa_rope if (is_sliding and swa_rope is not None) else base_rope
        top_partial = getattr(config, "partial_rotary_factor", None)
        if top_partial is not None and "partial_rotary_factor" not in rope_params:
            rope_params = {**rope_params, "partial_rotary_factor": top_partial}

        # Self-built cache construction (阶段7) -- replaces vLLM's get_rope().
        # rope_type/rope_theta/partial_rotary_factor read exactly as
        # get_rope() itself would (same rope_params dict, same defaults);
        # see runtime/kernels/rope.py's module docstring for the
        # attn_factor/attention_factor naming subtlety and the bit-exact
        # verification against vLLM's real output this is based on.
        rope_base = rope_params.get("rope_theta", 10000)
        rope_scaling_type = rope_params.get("rope_type", "default")
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
        rotary_dim = int(self.head_dim * partial_rotary_factor)
        rope_dtype = torch.get_default_dtype()
        rope_device = torch.empty(0).device
        # Build a cache key for RoPE sharing across layers
        if rope_scaling_type == "yarn":
            _cache_key = ("yarn", rotary_dim, rope_base,
                          rope_params["original_max_position_embeddings"],
                          rope_params["factor"], max_position_embeddings)
        elif rope_scaling_type == "default":
            _cache_key = ("default", rotary_dim, rope_base, max_position_embeddings)
        else:
            _cache_key = None

        if _cache_key is not None and _cache_key in _ROPE_CACHE_REGISTRY:
            cos_sin_cache = _ROPE_CACHE_REGISTRY[_cache_key]
        elif rope_scaling_type == "yarn":
            cos_sin_cache = compute_cos_sin_cache_yarn(
                rotary_dim=rotary_dim,
                original_max_position=rope_params["original_max_position_embeddings"],
                base=rope_base,
                scaling_factor=rope_params["factor"],
                dtype=rope_dtype,
                device=rope_device,
                extrapolation_factor=rope_params.get("extrapolation_factor", 1.0),
                attn_factor=rope_params.get("attn_factor", 1.0),
                beta_fast=rope_params.get("beta_fast", 32),
                beta_slow=rope_params.get("beta_slow", 1),
                truncate=rope_params.get("truncate", True),
            )
        elif rope_scaling_type == "default":
            cos_sin_cache = compute_cos_sin_cache_default(
                rotary_dim=rotary_dim,
                max_position=max_position_embeddings,
                base=rope_base,
                dtype=rope_dtype,
                device=rope_device,
            )
        else:
            raise NotImplementedError(
                f"SelfBuiltRotaryEmbedding: rope_type={rope_scaling_type!r} not "
                "implemented -- Laguna's real config only ever uses 'default' "
                "(sliding_attention layers, draft model) and 'yarn' "
                "(full_attention layers), verified against the real checkpoint."
            )
        if _cache_key is not None:
            _ROPE_CACHE_REGISTRY[_cache_key] = cos_sin_cache
        self.rotary_emb = SelfBuiltRotaryEmbedding(
            head_size=self.head_dim, cos_sin_cache=cos_sin_cache, is_neox_style=True
        )

        # Self-built op-dispatch placeholder (阶段7 item 2/4, replacing
        # vLLM's Attention construction -- see runtime/model/
        # plain_attention.py's module docstring for the full evaluation
        # and exact attribute contract). bf_attention.py's
        # replace_vllm_attention() discovers attention layers via
        # static_forward_context (now populated externally by
        # LagunaBackend.__init__, see laguna.py); this is the actual
        # compute-ownership boundary, not this class.
        # quant_config is still threaded through here (unlike qkv_proj/o_proj,
        # which are plain PlainLinear with no quant_config at all) -- it is
        # NOT about weight quantization (attention has none, see module
        # docstring) but about KV-cache quantization: the checkpoint declares
        # an FP8 kv_cache_scheme (config.json quantization_config), and
        # SelfBuiltAttentionPlaceholder.__init__ is what sets up k_scale/
        # v_scale handling for it.
        self.attn = SelfBuiltAttentionPlaceholder(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=self.sliding_window,
            prefix=maybe_prefix(attention_prefix, "attn"),
            sinks=sinks,
        )

        self.q_norm = TritonRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = TritonRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)

        if self.gating and self.g_proj is not None:
            gate = self.g_proj(hidden_states)
            gate = F.softplus(gate.float()).type_as(attn_output)
            if self.gate_per_head:
                attn_shape = attn_output.shape
                attn_output = (
                    attn_output.view(*attn_shape[:-1], self.num_heads, self.head_dim)
                    * gate.unsqueeze(-1)
                ).view(attn_shape)
            else:
                attn_output = attn_output * gate

        return self.o_proj(attn_output)


class LagunaDecoderLayerSelfBuilt(nn.Module):
    def __init__(
        self,
        config,
        cache_config: typing.Any = None,
        quant_config: typing.Any = None,
        prefix: str = "",
        enable_eplb: bool = False,
        layer_idx: int | None = None,
        attention_prefix: str | None = None,
        max_model_len: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        if layer_idx is None:
            layer_idx = _extract_layer_index(prefix)

        layer_types = getattr(config, "layer_types", None)
        is_sliding = (
            layer_types is not None and layer_types[layer_idx] == "sliding_attention"
        )
        attention_sink = is_sliding and getattr(
            config, "swa_attention_sink_enabled", False
        )

        per_layer_heads = getattr(config, "num_attention_heads_per_layer", None)
        layer_num_heads = (
            per_layer_heads[layer_idx]
            if per_layer_heads is not None
            else config.num_attention_heads
        )

        self.self_attn = LagunaAttentionSelfBuilt(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=layer_num_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "self_attn"),
            attention_sink=attention_sink,
            layer_idx=layer_idx,
            attention_prefix=(
                maybe_prefix(attention_prefix, "self_attn")
                if attention_prefix is not None
                else None
            ),
        )

        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        self.is_moe_layer = (
            (layer_idx not in mlp_only_layers)
            and (config.num_experts > 0)
            and ((layer_idx + 1) % config.decoder_sparse_step == 0)
        )

        if self.is_moe_layer:
            self.mlp = LagunaMoESelfBuilt(
                config=config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "mlp"),
                enable_eplb=enable_eplb,
            )
        else:
            self.mlp = LagunaMLPSelfBuilt(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                prefix=maybe_prefix(prefix, "mlp"),
            )

        self.input_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TritonRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
