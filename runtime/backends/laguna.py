"""E3: Laguna-S-2.1 Backend — direct model.forward() without vLLM's LLM engine.

Loads the model via compat_vllm.get_model(), allocates KV caches, builds
FlashInfer attention metadata via vLLM's own FlashInferMetadataBuilder, and
drives prefill/decode forward passes directly.

Architecture: 48 layers (12 full attn 48-head + 36 SWA 72-head window=512),
47 MoE layers, 8 KV heads / head_dim=128, NVFP4 quantized.

Roadmap ref: E3 Laguna L2 = "LagunaBackend — 过质量链"
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch

from runtime.block_pool import ChunkedPrefillState
from runtime.compat_vllm import (
    VllmConfig,
    bind_kv_cache,
    get_distributed_init_method,
    get_model,
    get_open_port,
    init_worker_distributed_environment,
    set_current_vllm_config,
    set_forward_context,
)
from runtime.logprobs import compute_logprobs
from runtime.model_spec import ModelSpec
from runtime.sampling import SamplingParams, make_generator, sample_from_logits

logger = logging.getLogger("qwen_sm120_runtime.laguna_backend")

RESERVED_PHYSICAL_SLOTS = 1

# Ring KV for SWA layers: parameterized for DFlash verify qo_max=16
# Formula: cdiv(window - 1 + qo_max, block_size) + 1
# qo_max=1 → 33, qo_max=16 → 34 (审查阻断①)
SWA_QO_MAX = 16

# Upper bound on num_tokens for any CUDA-graph-captured forward step:
# decode (batch_size<=4, MultiBatchGraphManager) and DFlash verify/draft
# (fixed NUM_QUERY_PER_REQ=16, dflash_constants.py) both stay <= 16.
# LagunaMoEB12x instances built for these hot paths must be constructed
# with use_cuda_graph=True and max_num_tokens>=this bound -- FlashInfer's
# B12xMoEWrapper.run() raises RuntimeError if invoked during CUDA graph
# capture with use_cuda_graph=False (dynamic per-call torch.empty alloc
# is not graph-capturable). Prefill (chunked at 2048 tokens, never
# graph-captured) keeps using a separate use_cuda_graph=False instance --
# sizing the graph-safe workspace to 2048 instead of 16 would multiply its
# fixed VRAM footprint ~128x across all 47 MoE layers for no benefit.
MOE_GRAPH_SAFE_MAX_TOKENS = 16


def _ring_blocks_for_window(window: int, block_size: int, qo_max: int = SWA_QO_MAX) -> int:
    return -(-( window - 1 + qo_max) // block_size) + 1  # cdiv + 1


def _physical_slot(slot: int) -> int:
    return slot + RESERVED_PHYSICAL_SLOTS


class LagunaBackend:
    """Direct model runner for Laguna-S-2.1-NVFP4.

    Uses vLLM's own FlashInferMetadataBuilder for correct attention metadata.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_slots: int = 4,
        block_size: int = 16,
        blocks_per_slot: int = 512,
    ) -> None:
        import os as _os
        import sys as _sys

        _venv_bin = _os.path.dirname(_sys.executable)
        if _venv_bin not in _os.environ.get("PATH", ""):
            _os.environ["PATH"] = _venv_bin + ":" + _os.environ.get("PATH", "")

        torch.set_grad_enabled(False)

        self.vllm_config = vllm_config
        self.num_slots = num_slots
        self.block_size = block_size
        self.blocks_per_slot = blocks_per_slot
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        # Apply A2 patches before loading
        from runtime.nvfp4_custom_gemm import patch_nvfp4_custom_gemm
        from runtime.nvfp4_cutlass_direct_patch import patch_nvfp4_prefer_cutlass_direct

        patch_nvfp4_prefer_cutlass_direct()
        patch_nvfp4_custom_gemm()

        # Load model
        with set_current_vllm_config(vllm_config):
            init_method = get_distributed_init_method("127.0.0.1", get_open_port())
            init_worker_distributed_environment(
                vllm_config, rank=0, distributed_init_method=init_method, local_rank=0
            )
            self.model = get_model(vllm_config=vllm_config)

        # Initialize workspace manager for MoE layers
        from runtime.compat_vllm import init_flashinfer_workspace

        init_flashinfer_workspace(self.device)

        # Patch MoE layers with direct kernel (自研 kernel 集成)
        self._moe_b12x_kernels: list = []
        self._moe_sparkinfer_layers: list = []
        if _os.environ.get("QSR_MOE_SPARKINFER", "0") != "0":
            self._patch_moe_sparkinfer()
        elif _os.environ.get("QSR_MOE_B12X", "0") != "0":
            self._patch_moe_b12x()

        # Discover attention layers from static_forward_context
        sfc = vllm_config.compilation_config.static_forward_context
        self.static_forward_context = sfc
        self.attn_layer_names: list[str] = []
        for name, layer in sfc.items():
            if hasattr(layer, "get_attn_backend"):
                self.attn_layer_names.append(name)
        logger.info("Laguna: %d attention layers discovered", len(self.attn_layer_names))

        # Group layers by (num_qo_heads, num_kv_heads, window_left)
        # Each group gets its own FlashInferMetadataBuilder
        hf_config = vllm_config.model_config.hf_config
        layer_types = getattr(hf_config, "layer_types", None)
        sliding_window = getattr(hf_config, "sliding_window", None)

        self._layer_groups: dict[tuple, list[str]] = {}
        for name in self.attn_layer_names:
            layer = sfc[name]
            nqh = layer.num_heads
            nkvh = layer.num_kv_heads
            parts = name.split(".")
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                    except ValueError:
                        pass
                    break
            if layer_types is not None and sliding_window is not None:
                if layer_idx is not None and layer_idx < len(layer_types):
                    is_sliding = layer_types[layer_idx] == "sliding_attention"
                    wl = (sliding_window - 1) if is_sliding else -1
                else:
                    wl = -1
            else:
                wl = -1
            key = (wl, nqh, nkvh)
            self._layer_groups.setdefault(key, []).append(name)

        logger.info(
            "Laguna: layer groups: %s",
            {f"wl={k[0]},qh={k[1]},kvh={k[2]}": len(v) for k, v in self._layer_groups.items()},
        )

        # Create FlashInferMetadataBuilder for each group
        from runtime.compat_vllm import get_flashinfer_metadata_builder
        FlashInferMetadataBuilder = get_flashinfer_metadata_builder()

        self._metadata_builders: dict[tuple, FlashInferMetadataBuilder] = {}
        with set_current_vllm_config(vllm_config):
            for group_key, layer_names in self._layer_groups.items():
                first_layer = sfc[layer_names[0]]
                kv_cache_spec = first_layer.get_kv_cache_spec(vllm_config)
                builder = FlashInferMetadataBuilder(
                    kv_cache_spec=kv_cache_spec,
                    layer_names=layer_names,
                    vllm_config=vllm_config,
                    device=self.device,
                )
                builder.disable_split_kv = True
                self._metadata_builders[group_key] = builder

        # ── Classify layers: full attention vs SWA ──
        cache_dtype_str = vllm_config.cache_config.cache_dtype
        self._cache_dtype_str = cache_dtype_str
        self._full_layer_names: list[str] = []
        self._swa_layer_names: list[str] = []
        self._swa_window: int = 0
        for name in self.attn_layer_names:
            layer = sfc[name]
            spec = layer.get_kv_cache_spec(vllm_config)
            spec_cls = type(spec).__name__
            if spec_cls == "SlidingWindowSpec":
                self._swa_layer_names.append(name)
                self._swa_window = spec.sliding_window
            else:
                self._full_layer_names.append(name)

        self._ring_blocks_per_slot = (
            _ring_blocks_for_window(self._swa_window, block_size)
            if self._swa_window > 0
            else 0
        )
        self._ring_slots_per_slot = self._ring_blocks_per_slot * block_size
        logger.info(
            "Laguna: %d full layers, %d SWA layers (window=%d, ring_blocks=%d/slot)",
            len(self._full_layer_names),
            len(self._swa_layer_names),
            self._swa_window,
            self._ring_blocks_per_slot,
        )

        # ── Allocate KV caches: per-group ──
        num_phys = num_slots + RESERVED_PHYSICAL_SLOTS
        full_num_blocks = num_phys * blocks_per_slot
        ring_num_blocks = num_phys * self._ring_blocks_per_slot
        self.kv_caches: dict[str, torch.Tensor] = {}
        for name in self.attn_layer_names:
            layer = sfc[name]
            backend_cls = layer.get_attn_backend()
            is_swa = name in self._swa_layer_names
            n_blocks = ring_num_blocks if is_swa else full_num_blocks
            shape = backend_cls.get_kv_cache_shape(
                n_blocks, block_size, layer.num_kv_heads, layer.head_size, cache_dtype_str
            )
            self.kv_caches[name] = torch.zeros(
                shape, dtype=layer.kv_cache_torch_dtype, device=self.device
            )
        runner_kv_caches: list[torch.Tensor] = []
        bind_kv_cache(self.kv_caches, sfc, runner_kv_caches)

        # ── Persistent prefill scratch for SWA layers (审查非阻断③) ──
        # Allocated once, reused across slots. Not zeroed (causal mask
        # guarantees no read-before-write within the window).
        self._swa_scratch: dict[str, torch.Tensor] = {}
        # SWA scratch: sized for overlap (window) + one prefill chunk.
        # Chunked prefill copies the last `window` tokens from ring into
        # scratch before each chunk, then processes chunk_tokens new tokens.
        # Total scratch capacity = window + chunk_tokens.
        self._prefill_chunk_tokens = int(os.environ.get("QSR_PREFILL_CHUNK", "8192"))
        _scratch_tokens = (self._swa_window if self._swa_window > 0 else 0) + self._prefill_chunk_tokens
        self._swa_scratch_blocks = min(
            blocks_per_slot,
            -(-_scratch_tokens // block_size),  # cdiv
        )
        if self._swa_layer_names:
            for name in self._swa_layer_names:
                layer = sfc[name]
                backend_cls = layer.get_attn_backend()
                shape = backend_cls.get_kv_cache_shape(
                    self._swa_scratch_blocks, block_size, layer.num_kv_heads,
                    layer.head_size, cache_dtype_str,
                )
                self._swa_scratch[name] = torch.empty(
                    shape, dtype=layer.kv_cache_torch_dtype, device=self.device
                )

        # Per-slot state
        self.slot_kv_len: list[int] = [0] * num_slots
        self.slot_committed_tokens: list[list[int]] = [[] for _ in range(num_slots)]
        # E1: mirrors DirectModelRunner.block_table's role as a per-slot
        # "has this slot ever been touched" dirty flag for admission. Laguna
        # has no block-table indirection (physical slot is a direct
        # arithmetic mapping, see _physical_slot) -- this list is never
        # populated, only kept empty/falsy so ServerEngine's shared admission
        # check (`slot_kv_len[slot] != 0 or block_table[slot]`) works
        # unmodified against either backend.
        self.block_table: list[list[int]] = [[] for _ in range(num_slots)]

        # Pre-allocated decode buffers (avoid per-step tensor allocation)
        max_batch = num_slots
        self._decode_input_ids = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        self._decode_positions = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        self._decode_seq_lens = torch.zeros(max_batch, dtype=torch.int32, device=self.device)
        self._decode_block_table = torch.zeros(
            max_batch, blocks_per_slot, dtype=torch.int32, device=self.device
        )
        self._decode_slot_mapping = torch.zeros(max_batch, dtype=torch.long, device=self.device)
        # query_start_loc: [0, 1, 2, ..., batch_size] for decode (qo_len=1)
        self._decode_qsl_gpu = torch.arange(max_batch + 1, dtype=torch.int32, device=self.device)
        self._decode_qsl_cpu = torch.arange(max_batch + 1, dtype=torch.int32, pin_memory=True)

        # SWA ring decode buffers (separate from full-attention buffers)
        if self._ring_blocks_per_slot > 0:
            self._swa_decode_block_table = torch.zeros(
                max_batch, self._ring_blocks_per_slot, dtype=torch.int32, device=self.device
            )
            self._swa_decode_slot_mapping = torch.zeros(
                max_batch, dtype=torch.long, device=self.device
            )
            self._swa_decode_seq_lens = torch.zeros(
                max_batch, dtype=torch.int32, device=self.device
            )

        # Expose for engine compatibility
        self.num_speculative_tokens = 0
        model_id = getattr(vllm_config.model_config, "model", "poolside/Laguna-S-2.1-NVFP4")
        architecture = getattr(hf_config, "architectures", ["LagunaForCausalLM"])[0]
        # E1: no MTP draft model and no GDN layers -- Laguna has neither
        # speculative decoding (DFlash is planned, roadmap L3, not wired
        # into this backend yet) nor a GDN/SSM recursive state; every
        # discovered layer is a (full or sliding-window) attention layer.
        self.spec = ModelSpec.from_runner_init(
            model_id=model_id,
            architecture=architecture,
            attn_layer_names=self.attn_layer_names,
            gdn_layer_names=[],
            mtp_model_id=None,
            num_speculative_tokens=0,
            kv_dtype=cache_dtype_str,
            block_size=block_size,
        )
        self.num_qo_heads = sfc[self.attn_layer_names[0]].num_heads
        self.num_kv_heads = sfc[self.attn_layer_names[0]].num_kv_heads
        self.head_dim = sfc[self.attn_layer_names[0]].head_size

        logger.info(
            "LagunaBackend initialized: %d slots, block_size=%d",
            num_slots,
            block_size,
        )

    def _patch_moe_b12x(self) -> None:
        """Replace vLLM FusedMoE dispatch with direct FlashInfer B12x kernel.

        Monkey-patches each LagunaMoE.forward to:
        1. gate → sigmoid topk routing (vLLM's fused_topk_bias)
        2. B12x expert compute (single fused kernel, CUDA Graph captured)
        3. shared expert + combine

        Each layer gets TWO LagunaMoEB12x instances, dispatched by num_tokens
        at call time:
        - graph-safe (use_cuda_graph=True, max_num_tokens=MOE_GRAPH_SAFE_MAX_TOKENS):
          for decode/DFlash-verify calls that run inside an outer
          torch.cuda.graph() capture (LagunaCudaGraphDecode / DFlashVerifyCudaGraph
          / DFlashDraftCudaGraph all call backend.model.forward() directly inside
          `with torch.cuda.graph(...)`). FlashInfer's B12xMoEWrapper.run() raises
          RuntimeError if called during graph capture unless it was constructed
          with use_cuda_graph=True (see flashinfer/fused_moe/cute_dsl/b12x_moe.py,
          the _is_cuda_graph_capturing() guard) -- this is the previously-observed
          "CUDA graph不兼容" failure. The graph-safe wrapper also uses pre-allocated
          fixed buffers instead of a fresh torch.empty() per call, which is the
          fix for the paired "性能不及预期" finding too (eager B12xMoEWrapper
          dynamic-alloc mode is measurably slower, see notes/2026-07-23-laguna-
          moe-b12x-direct-kernel.md).
        - eager (use_cuda_graph=False): for prefill, whose chunk size (2048) is
          never graph-captured and far exceeds MOE_GRAPH_SAFE_MAX_TOKENS.
        """
        from runtime.backends.laguna_moe_kernel import LagunaMoEB12x
        from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
            fused_topk_bias,
        )

        model = self.model
        hf_config = self.vllm_config.model_config.hf_config
        num_experts = getattr(hf_config, "num_experts", 256)
        top_k = getattr(hf_config, "num_experts_per_tok", 10)
        hidden_size = getattr(hf_config, "hidden_size", 3072)
        intermediate_size = getattr(hf_config, "moe_intermediate_size", 1024)

        patched = 0
        for name, module in model.named_modules():
            if not hasattr(module, "gate") or not hasattr(module, "experts"):
                continue
            experts_obj = module.experts
            if not hasattr(experts_obj, "routed_experts"):
                continue
            routed = experts_obj.routed_experts
            if not hasattr(routed, "w13_weight"):
                continue

            moe_module = module
            shared_expert = getattr(moe_module, "shared_expert", None)
            routed_scaling = getattr(moe_module, "routed_scaling_factor", 1.0)
            renormalize = getattr(hf_config, "norm_topk_prob", True)
            softcap = getattr(hf_config, "moe_router_logit_softcapping", 0.0) or 0.0
            e_bias = getattr(experts_obj, "e_score_correction_bias", None)
            apply_on_input = getattr(hf_config, "moe_apply_router_weight_on_input", False)

            weight_kwargs = dict(
                w13_weight=routed.w13_weight,
                w13_weight_scale=routed.w13_weight_scale,
                w13_weight_scale_2=routed.w13_weight_scale_2,
                w2_weight=routed.w2_weight,
                w2_weight_scale=routed.w2_weight_scale,
                w2_weight_scale_2=routed.w2_weight_scale_2,
            )

            b12x_graph = LagunaMoEB12x(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                device=self.device,
                use_cuda_graph=True,
                max_num_tokens=MOE_GRAPH_SAFE_MAX_TOKENS,
            )
            b12x_graph.load_weights(**weight_kwargs)

            b12x_eager = LagunaMoEB12x(
                num_experts=num_experts,
                top_k=top_k,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                device=self.device,
                use_cuda_graph=False,
            )
            b12x_eager.load_weights(**weight_kwargs)

            self._moe_b12x_kernels.append(b12x_graph)
            self._moe_b12x_kernels.append(b12x_eager)

            def _make_patched_forward(
                moe_mod, _b12x_graph, _b12x_eager, _shared, _scaling,
                _renorm, _softcap, _e_bias, _top_k, _apply_on_input,
            ):
                def _patched_forward(hidden_states: torch.Tensor) -> torch.Tensor:
                    orig_shape = hidden_states.shape
                    hs = hidden_states.view(-1, hidden_states.shape[-1])
                    router_logits, _ = moe_mod.gate(hs)
                    router_logits = router_logits.float()
                    if _softcap > 0:
                        router_logits = torch.tanh(router_logits / _softcap) * _softcap
                    topk_weights, topk_ids = fused_topk_bias(
                        hs, router_logits, "sigmoid", _e_bias,
                        _top_k, _renorm, routed_scaling_factor=_scaling if not _apply_on_input else 1.0,
                    )
                    b12x_kernel = (
                        _b12x_graph if hs.shape[0] <= MOE_GRAPH_SAFE_MAX_TOKENS else _b12x_eager
                    )
                    routed_out = b12x_kernel.forward(hs, topk_ids, topk_weights)
                    if _apply_on_input:
                        routed_out = routed_out * _scaling
                    if _shared is not None:
                        shared_out = _shared(hs)
                        routed_out = routed_out + shared_out
                    return routed_out.view(orig_shape)
                return _patched_forward

            moe_module.forward = _make_patched_forward(
                moe_module, b12x_graph, b12x_eager, shared_expert, routed_scaling,
                renormalize, softcap, e_bias, top_k, apply_on_input,
            )
            patched += 1

        logger.info("Laguna: patched %d MoE layers with B12x direct kernel", patched)


    def _patch_moe_sparkinfer(self) -> None:
        """Replace vLLM FusedMoE expert kernel with sparkinfer.

        Loads weights directly from checkpoint (bypasses vLLM scale reformat),
        uses runtime-alpha path (Phase 1 validated: cos=0.964 vs bf16 truth).
        Patches each MoE layer's forward to: router → sparkinfer → shared.

        After preparing sparkinfer weights for a layer, frees vLLM's copy of
        that layer's expert weights to avoid double memory usage.

        Gated by QSR_MOE_SPARKINFER=1.
        """
        from runtime.backends.laguna_sparkinfer_moe import (
            SparkinferMoELayer,
            _find_checkpoint,
            load_moe_layer_weights,
            prepare_sparkinfer_layer,
            sparkinfer_version,
        )
        from sparkinfer.moe.fused_moe._impl import allocate_tp_moe_workspace_pool
        from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
            fused_topk_bias,
        )

        model = self.model
        hf_config = self.vllm_config.model_config.hf_config
        top_k = getattr(hf_config, "num_experts_per_tok", 10)
        renormalize = getattr(hf_config, "norm_topk_prob", True)
        softcap = getattr(hf_config, "moe_router_logit_softcapping", 0.0) or 0.0
        apply_on_input = getattr(hf_config, "moe_apply_router_weight_on_input", False)

        workspace = allocate_tp_moe_workspace_pool()
        ckpt = _find_checkpoint()
        logger.info(
            "sparkinfer MoE patch (checkpoint-direct, alpha path): sparkinfer@%s",
            sparkinfer_version(),
        )

        patched = 0
        for name, module in model.named_modules():
            if not hasattr(module, "gate") or not hasattr(module, "experts"):
                continue
            experts_obj = module.experts
            if not hasattr(experts_obj, "routed_experts"):
                continue
            routed = experts_obj.routed_experts
            if not hasattr(routed, "w13_weight"):
                continue

            parts = name.split(".")
            layer_idx = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                    except ValueError:
                        pass
            if layer_idx is None or layer_idx == 0:
                continue

            moe_module = module
            shared_expert = getattr(moe_module, "shared_expert", None)
            routed_scaling = getattr(moe_module, "routed_scaling_factor", 1.0)
            e_bias = getattr(experts_obj, "e_score_correction_bias", None)

            # Load weights directly from checkpoint (alpha path)
            raw = load_moe_layer_weights(ckpt, layer_idx, self.device)
            # a1_gscale = 1/input_scale (sparkinfer convention: reciprocal)
            inp13 = getattr(routed, "w13_input_scale", None)
            inp2 = getattr(routed, "w2_input_scale", None)
            a1g = (1.0 / inp13.float().max()).item() if inp13 is not None and inp13.numel() > 0 else None
            a2g = (1.0 / inp2.float().max()).item() if inp2 is not None and inp2.numel() > 0 else None
            si_experts = prepare_sparkinfer_layer(raw, self.device, a1_gscale=a1g, a2_gscale=a2g)
            del raw
            si_layer = SparkinferMoELayer(si_experts, workspace, self.device)
            self._moe_sparkinfer_layers.append(si_layer)

            # Free vLLM's expert weights to reclaim memory
            for attr in ("w13_weight", "w13_weight_scale", "w13_weight_scale_2",
                         "w2_weight", "w2_weight_scale", "w2_weight_scale_2",
                         "w13_input_scale", "w2_input_scale"):
                if hasattr(routed, attr):
                    t = getattr(routed, attr)
                    if isinstance(t, torch.nn.Parameter):
                        t.data = torch.empty(0, device=t.device)
                    elif isinstance(t, torch.Tensor):
                        setattr(routed, attr, torch.empty(0, device=t.device))
            torch.cuda.empty_cache()

            def _make_patched_forward(
                moe_mod, _si_layer, _shared, _scaling,
                _renorm, _softcap, _e_bias, _top_k, _apply_on_input,
            ):
                def _patched_forward(hidden_states: torch.Tensor) -> torch.Tensor:
                    orig_shape = hidden_states.shape
                    hs = hidden_states.view(-1, hidden_states.shape[-1])
                    router_logits, _ = moe_mod.gate(hs)
                    router_logits = router_logits.float()
                    if _softcap > 0:
                        router_logits = torch.tanh(router_logits / _softcap) * _softcap
                    topk_weights, topk_ids = fused_topk_bias(
                        hs, router_logits, "sigmoid", _e_bias,
                        _top_k, _renorm,
                        routed_scaling_factor=1.0,
                    )
                    routed_out = _si_layer.forward(hs, topk_ids, topk_weights)
                    if _shared is not None:
                        shared_out = _shared(hs)
                        if _scaling != 1.0:
                            routed_out = routed_out * _scaling
                        routed_out = routed_out + shared_out
                    elif _scaling != 1.0:
                        routed_out = routed_out * _scaling
                    return routed_out.view(orig_shape)
                return _patched_forward

            moe_module.forward = _make_patched_forward(
                moe_module, si_layer, shared_expert, routed_scaling,
                renormalize, softcap, e_bias, top_k, apply_on_input,
            )
            patched += 1
            if patched % 10 == 0:
                logger.info("sparkinfer MoE: patched %d layers...", patched)

        logger.info("Laguna: patched %d MoE layers with sparkinfer kernel", patched)

    def _fill_decode_buffers(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
    ) -> None:
        """Fill pre-allocated buffers for decode (avoids per-step tensor allocation).

        Full-attention layers: standard contiguous block_table.
        SWA layers: ring block_table (block-aligned window) + ring slot_mapping.
        """
        batch_size = len(slot_ids)
        bs = self.block_size
        for i in range(batch_size):
            self._decode_input_ids[i] = token_ids[i]
            self._decode_positions[i] = kv_lengths[i]
            self._decode_seq_lens[i] = kv_lengths[i] + 1

            phys = _physical_slot(slot_ids[i])
            pos = kv_lengths[i]
            new_kv_len = kv_lengths[i] + 1

            # ── Full-attention block_table / slot_mapping ──
            full_base = phys * self.blocks_per_slot
            n_blocks = (new_kv_len + bs - 1) // bs
            self._decode_block_table[i, :n_blocks] = torch.arange(
                full_base, full_base + n_blocks, dtype=torch.int32, device=self.device
            )
            if n_blocks < self.blocks_per_slot:
                self._decode_block_table[i, n_blocks:] = 0
            self._decode_slot_mapping[i] = (
                (full_base + pos // bs) * bs + pos % bs
            )

            # ── SWA ring block_table / slot_mapping ──
            if self._ring_blocks_per_slot > 0:
                ring_base = phys * self._ring_blocks_per_slot
                ring_slots = self._ring_slots_per_slot
                window = self._swa_window

                # Block-aligned window start
                window_start = max(0, pos - window + 1)
                aligned_start = (window_start // bs) * bs
                aligned_len = pos + 1 - aligned_start
                n_ring = (aligned_len + bs - 1) // bs

                for j in range(n_ring):
                    actual_pos = aligned_start + j * bs
                    ring_block = (actual_pos % ring_slots) // bs
                    self._swa_decode_block_table[i, j] = ring_base + ring_block
                if n_ring < self._ring_blocks_per_slot:
                    self._swa_decode_block_table[i, n_ring:] = 0

                self._swa_decode_seq_lens[i] = aligned_len

                # Ring slot_mapping for the new decode token
                ring_block = (pos % ring_slots) // bs
                ring_off = pos % bs
                self._swa_decode_slot_mapping[i] = (
                    (ring_base + ring_block) * bs + ring_off
                )

    def _build_common_attn_metadata(
        self,
        slot_ids: list[int],
        kv_lengths: list[int],
        qo_lens: list[int],
        is_decode: bool,
    ):
        """Build CommonAttentionMetadata for full-attention layers."""
        from runtime.compat_vllm import get_common_attn_metadata_cls
        CommonAttentionMetadata = get_common_attn_metadata_cls()

        num_reqs = len(slot_ids)
        num_actual_tokens = sum(qo_lens)
        page_size = self.block_size
        new_kv_lens = [kv_len + qo for kv_len, qo in zip(kv_lengths, qo_lens)]

        if is_decode and max(qo_lens) == 1:
            query_start_loc = self._decode_qsl_gpu[:num_reqs + 1]
            query_start_loc_cpu = self._decode_qsl_cpu[:num_reqs + 1]
        else:
            qo_indptr = np.zeros(num_reqs + 1, dtype=np.int32)
            np.cumsum(qo_lens, dtype=np.int32, out=qo_indptr[1:])
            query_start_loc = torch.from_numpy(qo_indptr).to(self.device)
            query_start_loc_cpu = torch.from_numpy(qo_indptr)

        if is_decode and max(qo_lens) == 1:
            seq_lens = self._decode_seq_lens[:num_reqs]
            max_blocks = max((kvl + page_size - 1) // page_size for kvl in new_kv_lens)
            block_table = self._decode_block_table[:num_reqs, :max_blocks]
            slot_mapping = self._decode_slot_mapping[:num_reqs]
        else:
            seq_lens_np = np.array(new_kv_lens, dtype=np.int32)
            seq_lens = torch.from_numpy(seq_lens_np).to(self.device)
            max_blocks = max((kvl + page_size - 1) // page_size for kvl in new_kv_lens)
            block_table = torch.zeros(num_reqs, max_blocks, dtype=torch.int32, device=self.device)
            for i, (slot, n_blocks) in enumerate(
                zip(slot_ids, [(kvl + page_size - 1) // page_size for kvl in new_kv_lens])
            ):
                phys = _physical_slot(slot)
                base = phys * self.blocks_per_slot
                block_table[i, :n_blocks] = torch.arange(
                    base, base + n_blocks, dtype=torch.int32, device=self.device
                )
            # Vectorized slot_mapping
            mappings = []
            for slot, kv_len, qo in zip(slot_ids, kv_lengths, qo_lens):
                phys = _physical_slot(slot)
                base = phys * self.blocks_per_slot
                pos = torch.arange(kv_len, kv_len + qo, device=self.device)
                sm = (base + pos // self.block_size) * self.block_size + pos % self.block_size
                mappings.append(sm)
            slot_mapping = torch.cat(mappings) if len(mappings) > 1 else mappings[0]

        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max(qo_lens),
            max_seq_len=max(new_kv_lens),
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=True,
        )

    def _build_swa_attn_metadata(
        self,
        slot_ids: list[int],
        kv_lengths: list[int],
        qo_lens: list[int],
        is_decode: bool,
        swa_mode: str = "auto",
    ):
        """Build CommonAttentionMetadata for SWA layers.

        swa_mode: explicit routing — "decode_ring", "verify_ring",
                  "prefill_scratch", or "auto" (infer from is_decode/qo).
        """
        from runtime.compat_vllm import get_common_attn_metadata_cls
        CommonAttentionMetadata = get_common_attn_metadata_cls()

        num_reqs = len(slot_ids)
        num_actual_tokens = sum(qo_lens)
        bs = self.block_size
        ring_slots = self._ring_slots_per_slot

        # Resolve mode
        if swa_mode == "auto":
            if is_decode and max(qo_lens) == 1:
                swa_mode = "decode_ring"
            else:
                swa_mode = "prefill_scratch"

        if swa_mode == "decode_ring":
            query_start_loc = self._decode_qsl_gpu[:num_reqs + 1]
            query_start_loc_cpu = self._decode_qsl_cpu[:num_reqs + 1]
            seq_lens = self._swa_decode_seq_lens[:num_reqs]
            max_blocks = max(
                int(self._swa_decode_seq_lens[i].item()) for i in range(num_reqs)
            )
            max_blocks = (max_blocks + bs - 1) // bs
            block_table = self._swa_decode_block_table[:num_reqs, :max_blocks]
            slot_mapping = self._swa_decode_slot_mapping[:num_reqs]
            max_seq = int(seq_lens.max().item())
        elif swa_mode == "prefill_scratch":
            # Prefill: use full block_table (scratch KV is full-size)
            qo_indptr = np.zeros(num_reqs + 1, dtype=np.int32)
            np.cumsum(qo_lens, dtype=np.int32, out=qo_indptr[1:])
            query_start_loc = torch.from_numpy(qo_indptr).to(self.device)
            query_start_loc_cpu = torch.from_numpy(qo_indptr)

            new_kv_lens = [kv_len + qo for kv_len, qo in zip(kv_lengths, qo_lens)]
            seq_lens_np = np.array(new_kv_lens, dtype=np.int32)
            seq_lens = torch.from_numpy(seq_lens_np).to(self.device)
            max_blocks = max((kvl + bs - 1) // bs for kvl in new_kv_lens)
            block_table = torch.zeros(num_reqs, max_blocks, dtype=torch.int32, device=self.device)
            for i, (slot, n_blocks) in enumerate(
                zip(slot_ids, [(kvl + bs - 1) // bs for kvl in new_kv_lens])
            ):
                block_table[i, :n_blocks] = torch.arange(
                    n_blocks, dtype=torch.int32, device=self.device
                )
            mappings = []
            for slot, kv_len, qo in zip(slot_ids, kv_lengths, qo_lens):
                pos = torch.arange(kv_len, kv_len + qo, device=self.device)
                sm = (pos // bs) * bs + pos % bs
                mappings.append(sm)
            slot_mapping = torch.cat(mappings) if len(mappings) > 1 else mappings[0]
            max_seq = max(new_kv_lens)
        elif swa_mode == "verify_ring":
            # Verify (qo>1, ring buffer active): ring block_table + ring slot_mapping
            qo_indptr = np.zeros(num_reqs + 1, dtype=np.int32)
            np.cumsum(qo_lens, dtype=np.int32, out=qo_indptr[1:])
            query_start_loc = torch.from_numpy(qo_indptr).to(self.device)
            query_start_loc_cpu = torch.from_numpy(qo_indptr)

            window = self._swa_window
            ring_blocks_per_slot = self._ring_blocks_per_slot
            new_kv_lens = [kv_len + qo for kv_len, qo in zip(kv_lengths, qo_lens)]
            max_seq = max(new_kv_lens)

            # Window must cover [earliest_query - window + 1, latest_query].
            # Earliest query is at kv_len (first verify token), so
            # window_start = kv_len - window + 1  (NOT nkv - window).
            seq_lens_list = []
            for kv_len, qo in zip(kv_lengths, qo_lens):
                nkv = kv_len + qo
                ws = max(0, kv_len - window + 1)
                aligned_start = (ws // bs) * bs
                seq_lens_list.append(nkv - aligned_start)
            seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=self.device)

            max_blocks = ring_blocks_per_slot
            block_table = torch.zeros(num_reqs, max_blocks, dtype=torch.int32, device=self.device)
            for i, (slot, kv_len, qo) in enumerate(zip(slot_ids, kv_lengths, qo_lens)):
                phys = _physical_slot(slot)
                ring_base = phys * ring_blocks_per_slot
                nkv = kv_len + qo
                ws = max(0, kv_len - window + 1)
                aligned_start = (ws // bs) * bs
                aligned_len = nkv - aligned_start
                n_ring = min((aligned_len + bs - 1) // bs, ring_blocks_per_slot)
                for j in range(n_ring):
                    actual_pos = aligned_start + j * bs
                    ring_block = (actual_pos % ring_slots) // bs
                    block_table[i, j] = ring_base + ring_block

            # Ring slot_mapping for new tokens
            mappings = []
            for slot, kv_len, qo in zip(slot_ids, kv_lengths, qo_lens):
                phys = _physical_slot(slot)
                ring_base = phys * ring_blocks_per_slot
                for j in range(qo):
                    pos = kv_len + j
                    ring_block = (pos % ring_slots) // bs
                    ring_off = pos % bs
                    mappings.append((ring_base + ring_block) * bs + ring_off)
            slot_mapping = torch.tensor(mappings, dtype=torch.long, device=self.device)

        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max(qo_lens),
            max_seq_len=max_seq,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
            causal=True,
        )

    def _forward(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
        qo_len: int = 1,
        is_decode: bool = True,
        swa_kv_lengths: list[int] | None = None,
        skip_logits: bool = False,
    ) -> torch.Tensor | None:
        """Run one forward pass for a batch of slots.

        swa_kv_lengths: override kv_lengths for SWA layers (used by chunked
            prefill where SWA scratch has relative positions).
        """
        num_reqs = len(slot_ids)
        qo_lens = [qo_len] * num_reqs

        if is_decode and qo_len == 1:
            self._fill_decode_buffers(slot_ids, token_ids, kv_lengths)

        # Build CommonAttentionMetadata
        common_meta = self._build_common_attn_metadata(
            slot_ids, kv_lengths, qo_lens, is_decode
        )

        # Build per-group FlashInferMetadata using vLLM's builder
        # SWA groups use ring metadata; full groups use standard metadata
        attn_metadata_dict: dict[str, Any] = {}
        slot_mapping_dict: dict[str, torch.Tensor] = {}

        swa_meta = None
        if self._ring_blocks_per_slot > 0 and self._swa_layer_names:
            effective_swa_kv = swa_kv_lengths if swa_kv_lengths is not None else kv_lengths
            swa_meta = self._build_swa_attn_metadata(
                slot_ids, effective_swa_kv, qo_lens, is_decode
            )

        for group_key, builder in self._metadata_builders.items():
            wl = group_key[0]
            is_swa_group = wl >= 0
            meta = swa_meta if (is_swa_group and swa_meta is not None) else common_meta
            with set_current_vllm_config(self.vllm_config):
                metadata = builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=meta,
                )
            for name in self._layer_groups[group_key]:
                attn_metadata_dict[name] = metadata
                slot_mapping_dict[name] = meta.slot_mapping

        # Build input tensors (use pre-allocated buffers for decode)
        if is_decode and qo_len == 1:
            input_ids = self._decode_input_ids[:num_reqs]
            positions = self._decode_positions[:num_reqs]
        else:
            if qo_len == 1:
                flat_token_ids = token_ids
            elif num_reqs == 1:
                flat_token_ids = token_ids
            else:
                flat_token_ids = [tok for slot_tokens in token_ids for tok in slot_tokens]

            input_ids = torch.tensor(flat_token_ids, dtype=torch.long, device=self.device)
            positions_list = []
            for kv_len, qo in zip(kv_lengths, qo_lens):
                positions_list.extend(range(kv_len, kv_len + qo))
            positions = torch.tensor(positions_list, dtype=torch.long, device=self.device)

        with set_forward_context(
            attn_metadata_dict, self.vllm_config, slot_mapping=slot_mapping_dict,
            skip_compiled=not is_decode,
        ):
            result = self.model.forward(input_ids, positions)

        # Handle tuple return when aux_hidden_state_layers is set (DFlash)
        if isinstance(result, tuple):
            hidden_states = result[0]
        else:
            hidden_states = result

        if skip_logits:
            return None
        logits = self.model.compute_logits(hidden_states)
        return logits

    def _prefill_with_swa_scratch(
        self, slot: int, prompt_ids: list[int]
    ) -> torch.Tensor:
        """Run prefill with SWA layers rebound to scratch, then copy to ring."""
        sfc = self.static_forward_context
        bs = self.block_size

        # Rebind SWA layers to scratch KV
        if self._swa_scratch:
            for name in self._swa_layer_names:
                sfc[name].kv_cache = self._swa_scratch[name]

        try:
            logits = self._forward(
                [slot], prompt_ids, [0], qo_len=len(prompt_ids), is_decode=False
            )

            # Copy last window from scratch to ring — slab copy (审查非阻断④)
            if self._swa_scratch:
                prompt_len = len(prompt_ids)
                window = self._swa_window
                ring_slots = self._ring_slots_per_slot
                phys = _physical_slot(slot)
                ring_base = phys * self._ring_blocks_per_slot
                window_start = max(0, prompt_len - window)

                slabs: list[tuple[int, int, int]] = []
                pos = window_start
                while pos < prompt_len:
                    ring_slot = pos % ring_slots
                    until_wrap = ring_slots - ring_slot
                    src_off = pos % bs
                    until_block_end = bs - src_off
                    count = min(until_wrap, until_block_end, prompt_len - pos)
                    slabs.append((pos, ring_slot, count))
                    pos += count

                for name in self._swa_layer_names:
                    scratch = self._swa_scratch[name]
                    ring = self.kv_caches[name]
                    for src_pos, dst_ring_slot, count in slabs:
                        sb = src_pos // bs
                        so = src_pos % bs
                        db = dst_ring_slot // bs + ring_base
                        do = dst_ring_slot % bs
                        ring[db, :, do:do + count] = scratch[sb, :, so:so + count]
        finally:
            # Always rebind SWA layers back to ring KV (审查 P3a)
            if self._swa_scratch:
                for name in self._swa_layer_names:
                    sfc[name].kv_cache = self.kv_caches[name]

        return logits

    def _copy_ring_to_scratch(
        self, slot: int, abs_start: int, count: int
    ) -> None:
        """Copy `count` tokens from ring KV to scratch starting at scratch pos 0.

        Reads ring positions [abs_start, abs_start+count) and writes them to
        scratch positions [0, count).
        """
        if count <= 0:
            return
        bs = self.block_size
        ring_slots = self._ring_slots_per_slot
        phys = _physical_slot(slot)
        ring_base = phys * self._ring_blocks_per_slot

        slabs: list[tuple[int, int, int]] = []
        pos = 0
        while pos < count:
            abs_pos = abs_start + pos
            ring_slot_idx = abs_pos % ring_slots
            until_wrap = ring_slots - ring_slot_idx
            dst_off = pos % bs
            src_off = ring_slot_idx % bs
            until_src_block = bs - src_off
            until_dst_block = bs - dst_off
            n = min(until_wrap, until_src_block, until_dst_block, count - pos)
            slabs.append((ring_slot_idx, pos, n))
            pos += n

        for name in self._swa_layer_names:
            scratch = self._swa_scratch[name]
            ring = self.kv_caches[name]
            for ring_slot_idx, dst_pos, n in slabs:
                sb = ring_slot_idx // bs + ring_base
                so = ring_slot_idx % bs
                db = dst_pos // bs
                do = dst_pos % bs
                scratch[db, :, do:do + n] = ring[sb, :, so:so + n]

    def _copy_scratch_to_ring(
        self, slot: int, scratch_start: int, abs_start: int, count: int
    ) -> None:
        """Copy `count` tokens from scratch[scratch_start:] to ring at abs positions."""
        if count <= 0:
            return
        bs = self.block_size
        ring_slots = self._ring_slots_per_slot
        phys = _physical_slot(slot)
        ring_base = phys * self._ring_blocks_per_slot

        slabs: list[tuple[int, int, int]] = []
        pos = 0
        while pos < count:
            abs_pos = abs_start + pos
            ring_slot_idx = abs_pos % ring_slots
            until_wrap = ring_slots - ring_slot_idx
            src_off = (scratch_start + pos) % bs
            dst_off = ring_slot_idx % bs
            until_src_block = bs - src_off
            until_dst_block = bs - dst_off
            n = min(until_wrap, until_src_block, until_dst_block, count - pos)
            slabs.append((scratch_start + pos, ring_slot_idx, n))
            pos += n

        for name in self._swa_layer_names:
            scratch = self._swa_scratch[name]
            ring = self.kv_caches[name]
            for src_pos, ring_slot_idx, n in slabs:
                sb = src_pos // bs
                so = src_pos % bs
                db = ring_slot_idx // bs + ring_base
                do = ring_slot_idx % bs
                ring[db, :, do:do + n] = scratch[sb, :, so:so + n]

    def _prefill_with_swa_chunked(
        self, slot: int, prompt_ids: list[int]
    ) -> torch.Tensor:
        """Chunked prefill for prompts longer than SWA scratch capacity.

        Each chunk: copy last `window` tokens from ring → scratch (overlap),
        then process chunk_tokens new tokens. Full-attention layers use
        absolute kv_length; SWA layers use relative positions in scratch.
        """
        sfc = self.static_forward_context
        bs = self.block_size
        chunk_tokens = self._prefill_chunk_tokens
        prompt_len = len(prompt_ids)
        window = self._swa_window

        all_logits = None
        for chunk_start in range(0, prompt_len, chunk_tokens):
            chunk_end = min(chunk_start + chunk_tokens, prompt_len)
            chunk = prompt_ids[chunk_start:chunk_end]
            chunk_len = len(chunk)

            # Overlap: copy last `window` tokens from ring to scratch
            overlap = min(window, chunk_start)
            if overlap > 0:
                self._copy_ring_to_scratch(slot, chunk_start - overlap, overlap)

            # Rebind SWA layers to scratch
            for name in self._swa_layer_names:
                sfc[name].kv_cache = self._swa_scratch[name]

            try:
                # Forward: full-attn uses absolute kv_length=chunk_start,
                # SWA uses relative kv_length=overlap (positions in scratch)
                is_last_chunk = chunk_end >= prompt_len
                logits = self._forward(
                    [slot], chunk, [chunk_start],
                    qo_len=chunk_len, is_decode=False,
                    swa_kv_lengths=[overlap],
                    skip_logits=not is_last_chunk,
                )
                if logits is not None:
                    all_logits = logits

                # Copy the last `window` tokens from scratch to ring
                total_in_scratch = overlap + chunk_len
                copy_count = min(window, total_in_scratch)
                copy_scratch_start = total_in_scratch - copy_count
                copy_abs_start = chunk_start + chunk_len - copy_count
                self._copy_scratch_to_ring(slot, copy_scratch_start, copy_abs_start, copy_count)
            finally:
                # Always rebind SWA layers back to ring (审查 P3a)
                for name in self._swa_layer_names:
                    sfc[name].kv_cache = self.kv_caches[name]

        return all_logits

    def prefill(self, slot: int, prompt_ids: list[int]) -> int:
        """Prefill prompt and return the greedy first token."""
        if self.slot_kv_len[slot] != 0:
            raise RuntimeError(
                f"slot {slot} is not fresh (kv_len={self.slot_kv_len[slot]})"
            )
        if self._swa_scratch:
            prompt_len = len(prompt_ids)
            if prompt_len <= self._prefill_chunk_tokens:
                logits = self._prefill_with_swa_scratch(slot, prompt_ids)
            else:
                logits = self._prefill_with_swa_chunked(slot, prompt_ids)
        else:
            logits = self._forward(
                [slot], prompt_ids, [0], qo_len=len(prompt_ids), is_decode=False
            )
        first_token = int(logits[-1].argmax(dim=-1).item())
        self.slot_kv_len[slot] = len(prompt_ids)
        self.slot_committed_tokens[slot] = list(prompt_ids) + [first_token]
        return first_token

    def prefill_sampled(
        self, slot: int, prompt_ids: list[int], params: SamplingParams
    ) -> int:
        if self.slot_kv_len[slot] != 0:
            raise RuntimeError(
                f"slot {slot} is not fresh (kv_len={self.slot_kv_len[slot]})"
            )
        if self._swa_scratch:
            logits = self._prefill_with_swa_scratch(slot, prompt_ids)
        else:
            logits = self._forward(
                [slot], prompt_ids, [0], qo_len=len(prompt_ids), is_decode=False
            )
        last_logits = logits[-1].unsqueeze(0)
        gen = make_generator(params.seed)
        first_token = int(
            sample_from_logits(last_logits, params, generator=gen).item()
        )
        self.slot_kv_len[slot] = len(prompt_ids)
        self.slot_committed_tokens[slot] = list(prompt_ids) + [first_token]
        return first_token

    def prefill_with_aux(
        self, slot: int, prompt_ids: list[int]
    ) -> tuple[int, list[torch.Tensor] | None]:
        """Prefill prompt and return (first_token, aux_hidden_states).

        Processes the prompt in chunks of PREFILL_CHUNK_SIZE tokens to
        reduce peak GPU memory. Only returns aux hidden states from the
        last chunk (sufficient for DFlash's initial context precompute).
        """
        if self.slot_kv_len[slot] != 0:
            raise RuntimeError(
                f"slot {slot} is not fresh (kv_len={self.slot_kv_len[slot]})"
            )
        prompt_len = len(prompt_ids)
        PREFILL_CHUNK = self._prefill_chunk_tokens

        if prompt_len <= PREFILL_CHUNK and self._swa_scratch:
            # Short prompt: use scratch path (single forward)
            sfc = self.static_forward_context
            bs = self.block_size
            for name in self._swa_layer_names:
                sfc[name].kv_cache = self._swa_scratch[name]

            try:
                logits, aux = self._forward_with_aux(
                    [slot], prompt_ids, [0], qo_len=prompt_len, is_decode=False
                )

                # Copy last window from scratch to ring
                window = self._swa_window
                ring_slots = self._ring_slots_per_slot
                phys = _physical_slot(slot)
                ring_base = phys * self._ring_blocks_per_slot
                window_start = max(0, prompt_len - window)
                slabs = []
                pos = window_start
                while pos < prompt_len:
                    ring_slot_idx = pos % ring_slots
                    until_wrap = ring_slots - ring_slot_idx
                    src_off = pos % bs
                    until_block_end = bs - src_off
                    count = min(until_wrap, until_block_end, prompt_len - pos)
                    slabs.append((pos, ring_slot_idx, count))
                    pos += count
                for name in self._swa_layer_names:
                    scratch = self._swa_scratch[name]
                    ring = self.kv_caches[name]
                    for src_pos, dst_ring_slot, count in slabs:
                        sb = src_pos // bs
                        so = src_pos % bs
                        db = dst_ring_slot // bs + ring_base
                        do = dst_ring_slot % bs
                        ring[db, :, do:do + count] = scratch[sb, :, so:so + count]
            finally:
                for name in self._swa_layer_names:
                    sfc[name].kv_cache = self.kv_caches[name]

        elif prompt_len <= PREFILL_CHUNK:
            # Short prompt, no SWA scratch
            logits, aux = self._forward_with_aux(
                [slot], prompt_ids, [0], qo_len=prompt_len, is_decode=False
            )

        else:
            # Long prompt: chunked prefill with overlap-aware SWA scratch
            sfc = self.static_forward_context
            window = self._swa_window
            aux = None

            for chunk_start in range(0, prompt_len, PREFILL_CHUNK):
                chunk_end = min(chunk_start + PREFILL_CHUNK, prompt_len)
                chunk = prompt_ids[chunk_start:chunk_end]
                chunk_len = len(chunk)
                is_last = (chunk_end == prompt_len)

                # Overlap: copy last `window` tokens from ring to scratch
                overlap = min(window, chunk_start) if self._swa_scratch else 0
                if overlap > 0:
                    self._copy_ring_to_scratch(slot, chunk_start - overlap, overlap)

                # Rebind SWA to scratch for this chunk
                if self._swa_scratch:
                    for name in self._swa_layer_names:
                        sfc[name].kv_cache = self._swa_scratch[name]

                try:
                    # Forward: full-attn uses absolute kv_length=chunk_start,
                    # SWA uses relative kv_length=overlap
                    swa_kv = [overlap] if self._swa_scratch else None
                    if is_last:
                        logits, aux = self._forward_with_aux(
                            [slot], chunk, [chunk_start], qo_len=chunk_len,
                            is_decode=False, swa_kv_lengths=swa_kv,
                        )
                    else:
                        self._forward(
                            [slot], chunk, [chunk_start], qo_len=chunk_len,
                            is_decode=False, swa_kv_lengths=swa_kv,
                            skip_logits=True,
                        )

                    # Copy the last `window` tokens from scratch to ring
                    if self._swa_scratch:
                        total_in_scratch = overlap + chunk_len
                        copy_count = min(window, total_in_scratch)
                        copy_scratch_start = total_in_scratch - copy_count
                        copy_abs_start = chunk_start + chunk_len - copy_count
                        self._copy_scratch_to_ring(
                            slot, copy_scratch_start, copy_abs_start, copy_count
                        )
                finally:
                    # Always rebind SWA layers back to ring (审查 P3a)
                    if self._swa_scratch:
                        for name in self._swa_layer_names:
                            sfc[name].kv_cache = self.kv_caches[name]

        first_token = int(logits[-1].argmax(dim=-1).item())
        self.slot_kv_len[slot] = prompt_len
        self.slot_committed_tokens[slot] = list(prompt_ids) + [first_token]
        return first_token, aux

    def _forward_with_aux(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
        qo_len: int = 1,
        is_decode: bool = True,
        swa_kv_lengths: list[int] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Like _forward but also returns aux_hidden_states."""
        num_reqs = len(slot_ids)
        qo_lens = [qo_len] * num_reqs

        if is_decode and qo_len == 1:
            self._fill_decode_buffers(slot_ids, token_ids, kv_lengths)

        common_meta = self._build_common_attn_metadata(
            slot_ids, kv_lengths, qo_lens, is_decode
        )

        attn_metadata_dict: dict[str, Any] = {}
        slot_mapping_dict: dict[str, torch.Tensor] = {}

        swa_meta = None
        if self._ring_blocks_per_slot > 0 and self._swa_layer_names:
            effective_swa_kv = swa_kv_lengths if swa_kv_lengths is not None else kv_lengths
            swa_meta = self._build_swa_attn_metadata(
                slot_ids, effective_swa_kv, qo_lens, is_decode
            )

        for group_key, builder in self._metadata_builders.items():
            wl = group_key[0]
            is_swa_group = wl >= 0
            meta = swa_meta if (is_swa_group and swa_meta is not None) else common_meta
            with set_current_vllm_config(self.vllm_config):
                metadata = builder.build(
                    common_prefix_len=0,
                    common_attn_metadata=meta,
                )
            for name in self._layer_groups[group_key]:
                attn_metadata_dict[name] = metadata
                slot_mapping_dict[name] = meta.slot_mapping

        if is_decode and qo_len == 1:
            input_ids = self._decode_input_ids[:num_reqs]
            positions = self._decode_positions[:num_reqs]
        else:
            if qo_len == 1:
                flat_token_ids = token_ids
            elif num_reqs == 1:
                flat_token_ids = token_ids
            else:
                flat_token_ids = [tok for slot_tokens in token_ids for tok in slot_tokens]
            input_ids = torch.tensor(flat_token_ids, dtype=torch.long, device=self.device)
            positions_list = []
            for kv_len, qo in zip(kv_lengths, qo_lens):
                positions_list.extend(range(kv_len, kv_len + qo))
            positions = torch.tensor(positions_list, dtype=torch.long, device=self.device)

        with set_forward_context(
            attn_metadata_dict, self.vllm_config, slot_mapping=slot_mapping_dict,
            skip_compiled=True,
        ):
            result = self.model.forward(input_ids, positions)

        if isinstance(result, tuple):
            hidden_states, aux_hidden_states = result
        else:
            hidden_states = result
            aux_hidden_states = None

        logits = self.model.compute_logits(hidden_states)
        return logits, aux_hidden_states

    def decode(self, slot: int, token_id: int) -> int:
        kv_len = self.slot_kv_len[slot]
        logits = self._forward([slot], [token_id], [kv_len], qo_len=1, is_decode=True)
        next_token = int(logits[-1].argmax(dim=-1).item())
        self.slot_kv_len[slot] += 1
        self.slot_committed_tokens[slot].append(token_id)
        return next_token

    def decode_sampled(
        self, slot: int, token_id: int, params: SamplingParams
    ) -> int:
        kv_len = self.slot_kv_len[slot]
        logits = self._forward([slot], [token_id], [kv_len], qo_len=1, is_decode=True)
        last_logits = logits[-1].unsqueeze(0)
        gen = make_generator(params.seed)
        next_token = int(
            sample_from_logits(last_logits, params, generator=gen).item()
        )
        self.slot_kv_len[slot] += 1
        self.slot_committed_tokens[slot].append(token_id)
        return next_token

    def decode_batch(self, slot_ids: list[int], token_ids: list[int]) -> list[int]:
        kv_lengths = [self.slot_kv_len[s] for s in slot_ids]
        logits = self._forward(
            slot_ids, token_ids, kv_lengths, qo_len=1, is_decode=True
        )
        next_tokens = []
        for i, slot in enumerate(slot_ids):
            next_token = int(logits[i].argmax(dim=-1).item())
            next_tokens.append(next_token)
            self.slot_kv_len[slot] += 1
            self.slot_committed_tokens[slot].append(token_ids[i])
        return next_tokens

    def decode_batch_sampled(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
        params_list: list[SamplingParams],
        *,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> list[int] | tuple[list[int], list[dict]]:
        """Decode one token per slot with per-request sampling params.

        Signature matches ``DirectModelRunner.decode_batch_sampled`` (E1: the
        two backends share ServerEngine's calling convention) -- greedy is
        temperature=0, a plain special case of sampling, per B1.
        """
        logits = self._forward(
            slot_ids, token_ids, kv_lengths, qo_len=1, is_decode=True
        )
        next_tokens: list[int] = []
        for i, (slot, params) in enumerate(zip(slot_ids, params_list)):
            if params.is_greedy:
                tok = int(logits[i].argmax(dim=-1).item())
            else:
                row = logits[i].unsqueeze(0)
                gen = make_generator(params.seed)
                tok = int(sample_from_logits(row, params, generator=gen).item())
            next_tokens.append(tok)
            self.slot_kv_len[slot] += 1
            self.slot_committed_tokens[slot].append(token_ids[i])
        if return_logprobs:
            lp_list = [
                compute_logprobs(logits[i].unsqueeze(0), [next_tokens[i]], top_k=top_logprobs)[0]
                for i in range(len(next_tokens))
            ]
            return next_tokens, lp_list
        return next_tokens

    def assert_swa_rebind(self) -> None:
        """Assert all SWA layers point to ring KV, not scratch (审查 P3a)."""
        sfc = self.static_forward_context
        for name in self._swa_layer_names:
            actual = sfc[name].kv_cache
            expected = self.kv_caches[name]
            assert actual is expected, (
                f"SWA layer {name} kv_cache is not ring KV "
                f"(got {id(actual):#x}, expected {id(expected):#x}). "
                f"Rebind leak on exception path?"
            )

    def reset_slot(self, slot: int) -> None:
        self.slot_kv_len[slot] = 0
        self.slot_committed_tokens[slot] = []
        phys = _physical_slot(slot)
        # Full-attention layers: clear blocks_per_slot blocks
        full_start = phys * self.blocks_per_slot
        full_end = full_start + self.blocks_per_slot
        for name in self._full_layer_names:
            self.kv_caches[name][full_start:full_end].zero_()
        # SWA layers: clear only ring_blocks_per_slot blocks
        if self._ring_blocks_per_slot > 0:
            ring_start = phys * self._ring_blocks_per_slot
            ring_end = ring_start + self._ring_blocks_per_slot
            for name in self._swa_layer_names:
                self.kv_caches[name][ring_start:ring_end].zero_()

    def reconcile_prefix_hit(self, token_ids: list[int]) -> int:
        """E1 stub: Laguna has no persistent content-addressed prefix cache
        yet (roadmap L2/L3 TODO) -- every admission is a cold miss."""
        return 0

    def prefill_chunked_begin(
        self,
        slots: list[int],
        prompts_per_slot: list[list[int]],
        chunk_size: int = 512,
    ) -> ChunkedPrefillState:
        """E1: one-shot prefill wrapper matching DirectModelRunner's chunked-
        prefill contract so ServerEngine's admission path is backend-neutral.

        Laguna has no incremental chunking yet (TODO, tracked for roadmap
        L2/L3): this processes each slot's WHOLE prompt in one call and
        always returns ``done=True`` immediately. A single very long prompt
        will therefore block the engine thread for its entire prefill
        instead of interleaving with other slots' decode rounds -- unlike
        the Qwen path's true incremental chunking (A5/B4).
        """
        if len(slots) != len(prompts_per_slot):
            raise ValueError("slots and prompts_per_slot must have equal length")
        if not slots:
            return ChunkedPrefillState(done=True, result={})
        result: dict[int, dict] = {}
        for slot, prompt in zip(slots, prompts_per_slot):
            first_token = self.prefill(slot, prompt)
            result[slot] = {"anchor": first_token, "draft_tokens": []}
        return ChunkedPrefillState(done=True, result=result)

    def prefill_chunked_step(self, state: ChunkedPrefillState) -> bool:
        """Laguna prefill is never incremental; state is always already done."""
        return state.done

    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> list[int]:
        slot = 0
        self.reset_slot(slot)
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p if top_p < 1.0 else 1.0,
            top_k=top_k if top_k > 0 else 0,
        )
        if temperature == 0:
            first = self.prefill(slot, prompt_ids)
        else:
            first = self.prefill_sampled(slot, prompt_ids, params)
        tokens = [first]
        for _ in range(max_tokens - 1):
            if temperature == 0:
                tok = self.decode(slot, tokens[-1])
            else:
                tok = self.decode_sampled(slot, tokens[-1], params)
            tokens.append(tok)
            if tok in (2, 24):  # Laguna EOS (generation_config.json)
                break
        self.reset_slot(slot)
        return tokens

    # ── Prefix cache support ──────────────────────────────────────────────

    def find_prefix_match(self, slot: int, prompt_ids: list[int]) -> int:
        """Find the longest prefix of prompt_ids that matches cached tokens.

        Returns the number of matching tokens (block-aligned down).
        The slot's KV cache must still be valid (not reset).
        """
        cached = self.slot_committed_tokens[slot]
        if not cached or self.slot_kv_len[slot] == 0:
            return 0
        n = 0
        for a, b in zip(cached, prompt_ids):
            if a != b:
                break
            n += 1
        # Align down to block boundary for full-attention KV correctness
        bs = self.block_size
        aligned = (n // bs) * bs
        # Never exceed the cached KV length
        return min(aligned, self.slot_kv_len[slot])

    def continue_prefill_with_aux(
        self, slot: int, prompt_ids: list[int], start_pos: int
    ) -> tuple[int, list[torch.Tensor] | None]:
        """Continue prefill from start_pos, reusing cached KV for [0, start_pos).

        The slot's KV cache must contain valid data for positions [0, start_pos).
        Only processes prompt_ids[start_pos:].
        Returns (first_token, aux_hidden_states_from_last_chunk).
        """
        prompt_len = len(prompt_ids)
        # Invalidate stale KV beyond start_pos (from previous generation)
        self.slot_kv_len[slot] = start_pos
        if start_pos >= prompt_len:
            # No new tokens — decode the last cached token to get logits
            last_token = prompt_ids[start_pos - 1]
            logits = self._forward(
                [slot], [last_token], [start_pos - 1], qo_len=1, is_decode=True
            )
            first_token = int(logits[0].argmax(dim=-1).item())
            self.slot_kv_len[slot] = start_pos
            self.slot_committed_tokens[slot] = list(prompt_ids) + [first_token]
            return first_token, None

        PREFILL_CHUNK = self._prefill_chunk_tokens
        suffix_len = prompt_len - start_pos

        if suffix_len <= PREFILL_CHUNK and self._swa_scratch:
            # Short suffix: single chunk with scratch
            sfc = self.static_forward_context
            window = self._swa_window
            bs = self.block_size

            # Copy overlap from ring to scratch
            overlap = min(window, start_pos)
            if overlap > 0:
                self._copy_ring_to_scratch(slot, start_pos - overlap, overlap)

            for name in self._swa_layer_names:
                sfc[name].kv_cache = self._swa_scratch[name]
            try:
                suffix = prompt_ids[start_pos:]
                logits, aux = self._forward_with_aux(
                    [slot], suffix, [start_pos], qo_len=suffix_len,
                    is_decode=False, swa_kv_lengths=[overlap],
                )
                # Copy last window from scratch to ring
                total_in_scratch = overlap + suffix_len
                copy_count = min(window, total_in_scratch)
                copy_scratch_start = total_in_scratch - copy_count
                copy_abs_start = start_pos + suffix_len - copy_count
                self._copy_scratch_to_ring(
                    slot, copy_scratch_start, copy_abs_start, copy_count
                )
            finally:
                for name in self._swa_layer_names:
                    sfc[name].kv_cache = self.kv_caches[name]

        elif suffix_len <= PREFILL_CHUNK:
            suffix = prompt_ids[start_pos:]
            logits, aux = self._forward_with_aux(
                [slot], suffix, [start_pos], qo_len=suffix_len, is_decode=False
            )

        else:
            # Long suffix: chunked prefill
            sfc = self.static_forward_context
            window = self._swa_window
            aux = None
            logits = None

            for chunk_start in range(start_pos, prompt_len, PREFILL_CHUNK):
                chunk_end = min(chunk_start + PREFILL_CHUNK, prompt_len)
                chunk = prompt_ids[chunk_start:chunk_end]
                chunk_len = len(chunk)
                is_last = (chunk_end == prompt_len)

                overlap = min(window, chunk_start) if self._swa_scratch else 0
                if overlap > 0:
                    self._copy_ring_to_scratch(slot, chunk_start - overlap, overlap)

                if self._swa_scratch:
                    for name in self._swa_layer_names:
                        sfc[name].kv_cache = self._swa_scratch[name]
                try:
                    swa_kv = [overlap] if self._swa_scratch else None
                    if is_last:
                        logits, aux = self._forward_with_aux(
                            [slot], chunk, [chunk_start], qo_len=chunk_len,
                            is_decode=False, swa_kv_lengths=swa_kv,
                        )
                    else:
                        self._forward(
                            [slot], chunk, [chunk_start], qo_len=chunk_len,
                            is_decode=False, swa_kv_lengths=swa_kv,
                            skip_logits=True,
                        )
                    if self._swa_scratch:
                        total_in_scratch = overlap + chunk_len
                        copy_count = min(window, total_in_scratch)
                        copy_scratch_start = total_in_scratch - copy_count
                        copy_abs_start = chunk_start + chunk_len - copy_count
                        self._copy_scratch_to_ring(
                            slot, copy_scratch_start, copy_abs_start, copy_count
                        )
                finally:
                    if self._swa_scratch:
                        for name in self._swa_layer_names:
                            sfc[name].kv_cache = self.kv_caches[name]

        first_token = int(logits[-1].argmax(dim=-1).item())
        self.slot_kv_len[slot] = prompt_len
        self.slot_committed_tokens[slot] = list(prompt_ids) + [first_token]
        return first_token, aux
