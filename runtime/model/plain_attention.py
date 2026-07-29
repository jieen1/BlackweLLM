"""Self-built replacement for vLLM's ``Attention`` construction-time
machinery -- 阶段7 item 2/4 (see notes/2026-07-27-vllm-complete-removal-
implementation-plan.md, "Attention op-dispatch ABC 结论修正").

**Architectural premise, stated explicitly so it's never mistaken for an
oversight (same reason plain_linear.py/plain_embedding.py spell out their
TP=1 assumption instead of leaving it implicit): sparkinfer is this
runtime's ONE and ONLY attention kernel. This is not a placeholder or a
"good enough for now" simplification of a would-be pluggable backend
system -- there is no second kernel anywhere on this runtime's roadmap,
and this class is written on the assumption that there never will be.**
Every attribute below is hardcoded to sparkinfer's real, single
requirement (FP8 KV cache, no attention sinks, per-tensor not per-head KV
scales) rather than derived through config-driven branching the way real
vLLM's ``Attention`` has to, because vLLM genuinely serves many backends/
model families and this runtime genuinely does not. If this runtime ever
needs a second attention kernel, that is a real architecture change --
redesigning how attention layers get discovered, configured, and
dispatched -- not a matter of adding an ``if`` branch to this class or
reviving vLLM's ``AttentionBackendEnum``/``get_attn_backend`` machinery.
Do not "future-proof" this file for that case; get instructed to do the
redesign instead, same as every other TP=1/single-architecture
simplification in this codebase.

Scope is precisely bounded by what this runtime's own consuming code
actually reads off an attention-layer object -- verified by grepping
runtime/backends/laguna.py, runtime/backends/bf_attention.py,
runtime/backends/laguna_sparkinfer_attn.py, runtime/backends/
laguna_cuda_graph.py, not guessed from vLLM's own internal usage:

- ``num_heads``/``num_kv_heads``/``head_size``: read by laguna.py (KV
  cache shape/discovery) and bf_attention.py (BFAttention construction).
- ``get_attn_backend`` (method, NEVER called): a duck-typed marker both
  laguna.py's and bf_attention.py's discovery loops use to recognize
  "this is an attention layer" (``hasattr(layer, "get_attn_backend")``).
  Stubbed here purely to satisfy that check.
- ``impl``: unconditionally overwritten by ``LagunaBackend.__init__``
  right after discovery (``layer.impl = SparkinferAttentionImpl(...)``,
  runtime/backends/laguna.py) before anything else reads it -- confirmed
  by tracing construction order. Left ``None`` here; nothing needs a
  real value before that assignment happens.
- ``kv_cache``: bound externally via vLLM's ``bind_kv_cache`` utility
  (plain ``forward_context[name].kv_cache = tensor`` assignment, no
  isinstance check -- verified against vllm/v1/worker/utils.py's real
  source) and read/written throughout laguna_cuda_graph.py for CUDA-graph
  capture. Placeholder value only; always overwritten before first
  real forward.
- ``kv_cache_dtype``/``kv_cache_torch_dtype``: read by bf_attention.py /
  laguna.py respectively. ``kv_cache_torch_dtype`` specifically is only
  consumed on a branch this runtime's real config never takes in
  production (FP8 KV cache is always active -- verified via
  EngineArgs.create_engine_config()), so it's computed defensively
  rather than left unset, but is not GPU-bit-exact-critical the way
  the scale handling below is.
- ``sliding_window``/``is_swa``: NOT read off this object by laguna.py's
  layer-group bookkeeping (that logic recomputes window/group info
  straight from ``hf_config.layer_types``/``hf_config.sliding_window``,
  independent of the attention layer object) -- but IS read by the
  ``get_kv_cache_spec()`` replacement (see laguna.py's KV-cache-spec
  classification loop), which used to call
  ``layer.get_kv_cache_spec(vllm_config)`` and read
  ``type(spec).__name__``/``spec.sliding_window`` off the result. Grep-
  verified real consumption of that return value: ``spec.block_size`` is
  NEVER read (KV-cache block_size comes from LagunaBackend's own
  externally-supplied constructor arg instead) -- so this class exposes
  ``is_swa``/``sliding_window`` directly, and laguna.py's consuming code
  reads those two attributes instead of calling
  ``get_kv_cache_spec()``/constructing a ``FullAttentionSpec``/
  ``SlidingWindowSpec``.
- ``_k_scale``/``_v_scale``: read by bf_attention.py (copied onto
  BFAttention), laguna_sparkinfer_attn.py, laguna_cuda_graph.py. These
  must hold the REAL checkpoint scale after weight loading -- see
  ``_apply_kv_cache_scale_post_load`` in runtime/model_loading.py, the
  self-built replacement for the ONE real vLLM method that actually
  populates them for this checkpoint's quantization scheme:
  ``CompressedTensorsKVCacheMethod.process_weights_after_loading``
  (vllm/model_executor/layers/quantization/compressed_tensors/
  compressed_tensors.py:1121-1147) -- NOT the more generic
  ``BaseKVCacheMethod.process_weights_after_loading`` (kv_cache.py:74),
  which looked like the relevant one at first read but is never actually
  invoked for this checkpoint (``CompressedTensorsKVCacheMethod``
  overrides it). Caught and corrected before writing this: the base
  class's "positive/duplicate/no-scale" branching logic does NOT apply
  here at all.

This runtime only ever runs one attention kernel (sparkinfer) against one
checkpoint format (NVFP4, FP8 KV cache) -- so unlike vLLM's ``Attention``,
which has to stay generic over many backends/quant schemes/model
families, this class hardcodes that one real case directly instead of
branching on config values to re-derive it (no "is kv_cache_dtype auto,
fp8, or something else" resolution, no backend registry, no
``AttentionBackendEnum``/``get_attn_backend`` real implementation,
no per-strategy scale-shape dispatch). ``laguna.py`` unconditionally
overwrites ``.impl`` with ``SparkinferAttentionImpl`` right after
discovery, before anything downstream reads it, so there is no real
backend-selection decision left for this runtime to make in the first
place -- adding one back here would be reintroducing exactly the
complexity 阶段7-补充's cross-engine research (nano-vllm/sglang) showed
isn't needed once the kernel is fixed. Also not created at all:
``q_scale``/``k_zero_point``/``v_zero_point``/``q_zero_point``/
``prob_scale`` -- real ``CompressedTensorsKVCacheMethod.create_weights``
creates these too (to support llm-compressor checkpoints using
asymmetric/per-query-head schemes), but this checkpoint never provides
values for any of them (verified directly against its safetensors: only
``self_attn.{k,v}_scale`` exist per layer) and nothing in this runtime's
forward path reads ``_q_scale``/``_q_scale_float``/zero-points
(grep-verified). If a future checkpoint ever uses a different KV-cache
quantization strategy (e.g. per-head instead of per-tensor scales), this
class needs revisiting, not generalizing in advance -- it fails loud
(``NotImplementedError``) rather than silently doing the wrong thing.
"""

from __future__ import annotations

import typing

import torch
from torch import nn

if typing.TYPE_CHECKING:
    from vllm.config import CacheConfig
    from vllm.model_executor.layers.quantization import QuantizationConfig


class SelfBuiltAttentionPlaceholder(nn.Module):
    """TP=1 replacement for constructing a real ``vllm.model_executor.
    layers.attention.Attention`` instance. See module docstring for the
    exact, grep-verified attribute contract this must satisfy.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        cache_config: CacheConfig,
        quant_config: QuantizationConfig | None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        sinks: torch.Tensor | None = None,
    ) -> None:
        # sparkinfer has no attention-sink kernel, and Laguna's real
        # config never sets attention_sink=True (verified against the
        # checkpoint's config.json) -- accepted only so this call site
        # matches laguna_decoder.py's existing signature, same idiom as
        # LagunaMoESelfBuilt.__init__'s `del quant_config, enable_eplb`.
        del sinks
        super().__init__()

        self.layer_name = prefix
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.sliding_window = per_layer_sliding_window
        self.is_swa = per_layer_sliding_window is not None

        # This runtime's one real KV-cache dtype, hardcoded rather than
        # re-derived (see module docstring's architectural premise): FP8,
        # always -- true for BOTH real models this runtime loads, verified
        # directly rather than assumed to be the same reason for both:
        # the main model's checkpoint declares a real FP8 kv_cache_scheme
        # (real Attention.__init__ would resolve cache_dtype to "fp8" from
        # it too); the DFlash draft model's checkpoint has NO
        # quantization_config at all (confirmed against its config.json),
        # but its KV cache is still unconditionally allocated as FP8
        # (stored uint8) by runtime/backends/laguna_dflash.py's
        # _alloc_draft_kv_cache -- a hardcoded decision independent of
        # cache_config/quant_config, not something this class needs to
        # rederive either.
        self.kv_cache_dtype = "fp8"
        self.kv_cache_torch_dtype = torch.float8_e4m3fn

        # Placeholder; always overwritten by bind_kv_cache()/laguna.py
        # before any real forward call (see module docstring).
        self.kv_cache: torch.Tensor = torch.tensor([])
        # Placeholder; unconditionally overwritten by laguna.py's/
        # laguna_dflash.py's SparkinferAttentionImpl patch before any real
        # forward call (see module docstring) -- never read in its unset
        # state.
        self.impl = None

        # What BFAttention/SparkinferAttentionImpl/fused_kv_scatter
        # actually read at runtime. Default 1.0 (no-op scale).
        self.register_buffer("_k_scale", torch.ones(1, dtype=torch.float32))
        self.register_buffer("_v_scale", torch.ones(1, dtype=torch.float32))

        # Whether a real per-tensor checkpoint scale exists to load is a
        # genuine, checkpoint-driven fact that differs between this
        # runtime's two real models -- not a hypothetical "future kernel"
        # branch (see module docstring's architectural premise, which is
        # about kernel choice, not this). Verified directly, not assumed:
        # the main model's checkpoint declares an FP8 kv_cache_scheme and
        # ships real ``self_attn.{k,v}_scale`` tensors; the DFlash draft
        # model's checkpoint has no quantization_config and ships neither.
        # Real vLLM's Attention would reach the identical outcome for the
        # draft model too (quant_config is None there, so
        # should_load_quant_weights is False and the k_scale/v_scale
        # Parameters never get created) -- its KV cache is FP8 regardless
        # (see kv_cache_dtype above) but permanently uses the hardcoded
        # _k_scale/_v_scale default of 1.0 above, no checkpoint value ever
        # overrides it. Loadable Parameters created here only get read by
        # _apply_kv_cache_scale_post_load (runtime/model_loading.py) when
        # this is True, matching this checkpoint's real "tensor" (not
        # "attn_head") KV-cache quantization strategy -- see module
        # docstring.
        self.has_checkpoint_kv_scale = (
            quant_config is not None and quant_config.kv_cache_scheme is not None
        )
        if self.has_checkpoint_kv_scale:
            cache_config.cache_dtype = "fp8"
            cache_config.calculate_kv_scales = False
            # Loadable from checkpoint (model.layers.N.self_attn.{k,v}_scale,
            # remapped to model.layers.N.self_attn.attn.{k,v}_scale by
            # vLLM's existing maybe_remap_kv_scale_name -- untouched,
            # still in use from runtime/model/laguna_model.py).
            self.k_scale = nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=False
            )
            self.v_scale = nn.Parameter(
                torch.ones(1, dtype=torch.float32), requires_grad=False
            )

    def get_attn_backend(self) -> None:
        """Never called -- duck-typed marker only. laguna.py's and
        bf_attention.py's attention-layer discovery loops use
        ``hasattr(layer, "get_attn_backend")`` to recognize this object
        (see module docstring); nothing calls the method itself."""
        raise NotImplementedError("marker attribute only, not meant to be called")
