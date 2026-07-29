"""E3: Laguna-S-2.1 Backend — direct model.forward() without vLLM's LLM engine.

Loads the model via runtime.model_loading.load_laguna_model() (self-built,
zero vLLM get_model() dependency since 任务#46), allocates KV caches, builds
SparkInfer paged attention for prefill and decode.
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

from bfdiag.trace import ring as bfdiag_trace
from bfprobe.routing import capture_routing
from runtime.backends.bf_attention import bf_attn_context
from runtime.block_pool import ChunkedPrefillState
from runtime.laguna_config import SelfBuiltVllmConfig
from runtime.laguna_runtime import (
    LagunaAttentionMetadata,
    bind_laguna_kv_cache,
    get_distributed_init_method,
    get_open_port,
)
from runtime.logprobs import compute_logprobs
from runtime.model_spec import ModelSpec
from runtime.sampling import SamplingParams, make_generator, sample_from_logits

logger = logging.getLogger("qwen_sm120_runtime.laguna_backend")

RESERVED_PHYSICAL_SLOTS = 0

# Ring KV for SWA layers: parameterized for DFlash verify qo_max=16
# Formula: cdiv(window - 1 + qo_max, block_size) + 1
# qo_max=1 → 33, qo_max=16 → 34 (审查阻断①)
SWA_QO_MAX = 16


def _ring_blocks_for_window(window: int, block_size: int, qo_max: int = SWA_QO_MAX) -> int:
    return -(-(window - 1 + qo_max) // block_size) + 1  # cdiv + 1


def _physical_slot(slot: int) -> int:
    return slot + RESERVED_PHYSICAL_SLOTS


def _prefill_chunk_ranges(
    start: int,
    end: int,
    chunk_tokens: int,
    *,
    min_final_tokens: int = 0,
) -> list[tuple[int, int]]:
    """Partition ``[start, end)`` while keeping the final chunk large enough.

    DFlash reconstructs its sliding-window KV from the aux hidden states
    returned for the final prefill chunk.  A short modulo remainder would
    otherwise leave part of that window uninitialized.
    """
    if start < 0 or end < start:
        raise ValueError(f"invalid prefill range [{start}, {end})")
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    if min_final_tokens < 0 or min_final_tokens > chunk_tokens:
        raise ValueError(
            "min_final_tokens must be in [0, chunk_tokens], got "
            f"{min_final_tokens} for chunk_tokens={chunk_tokens}"
        )
    if start == end:
        return []

    ranges = [
        (chunk_start, min(chunk_start + chunk_tokens, end))
        for chunk_start in range(start, end, chunk_tokens)
    ]
    if len(ranges) < 2 or min_final_tokens == 0:
        return ranges

    final_start, final_end = ranges[-1]
    final_len = final_end - final_start
    if final_len >= min_final_tokens:
        return ranges

    previous_start, previous_end = ranges[-2]
    deficit = min_final_tokens - final_len
    new_boundary = previous_end - deficit
    if new_boundary <= previous_start:
        raise ValueError(
            "cannot reserve the requested final prefill tail: "
            f"range=[{start}, {end}), chunk_tokens={chunk_tokens}, "
            f"min_final_tokens={min_final_tokens}"
        )
    ranges[-2] = (previous_start, new_boundary)
    ranges[-1] = (new_boundary, final_end)
    return ranges


class LagunaBackend:
    """Direct model runner for Laguna-S-2.1-NVFP4.

    Uses SparkInfer paged attention (replaces FlashInfer).
    """

    def __init__(
        self,
        vllm_config: SelfBuiltVllmConfig,
        *,
        num_slots: int = 4,
        block_size: int = 64,
        blocks_per_slot: int = 1088,
    ) -> None:
        # e66d254 (2026-07-26) pinned this to 64 because the sparkinfer
        # version at the time only supported 64-token pages. sparkinfer
        # master (merged 2026-07-27, notes/2026-07-27-sparkinfer-branch-
        # master-canonical.md) supports both 64 and 128 throughout its
        # planner/workspace/traits stack. Note: 128 does NOT unlock
        # sparkinfer's Laguna-specific kernel traits (select_paged_forward_
        # traits_from_plan) -- those additionally require num_kv_heads==4
        # (a TP=2 shard count), but this runtime is TP=1 with num_kv_heads=8,
        # so none of those traits ever fire regardless of page_size (see
        # notes/2026-07-27-laguna-real-shapes-correction-and-page-size-
        # migration-plan.md). Correctness verified against sparkinfer's own
        # reference at Laguna's real shapes (cos>=0.999991 for both page
        # sizes, see notes/2026-07-27-verify-cg-mode-fix-and-block-size-
        # eval.md) and against a deep numerical bisection that traced an
        # apparent accept-rate regression down to a genuine floating-point
        # tie-break flip, not a correctness bug (notes/2026-07-27-block-
        # size-128-migration-and-tie-break-noise.md). Fail fast on anything
        # else before model loading instead of reaching sparkinfer's opaque
        # planner error after allocating all weights.
        if block_size not in (64, 128):
            raise ValueError(
                "LagunaBackend requires block_size in (64, 128) for sparkinfer "
                f"paged attention; got {block_size}."
            )

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
        # Read once before router closures or any CUDA Graph capture. The
        # native A/B path uses this fixed prefill bound to allocate a
        # grow-never output arena during construction.
        self._prefill_chunk_tokens = int(os.environ.get("QSR_PREFILL_CHUNK", "8192"))
        self._laguna_router_library = None
        self._laguna_router_arena = None

        # The self-built model has no vLLM global-config reader.  Establish
        # the one-rank torch process group, then construct it directly.
        init_method = get_distributed_init_method("127.0.0.1", get_open_port())
        from runtime.laguna_config import init_laguna_distributed_environment

        init_laguna_distributed_environment(
            rank=0, distributed_init_method=init_method, local_rank=0
        )
        from runtime.model_loading import load_laguna_model

        self.model = load_laguna_model(vllm_config)
        # Set IR op priority (fused RMSNorm C++ kernels) — normally done by worker init
        vllm_config.kernel_config.ir_op_priority.set_default()

        # Patch MoE layers with the sparkinfer kernel (自研 kernel 集成,
        # zero vLLM dependency for the expert compute itself)
        self._moe_sparkinfer_layers: list = []
        self._initialize_laguna_router()
        self._patch_moe_sparkinfer()

        # Discover attention layers from static_forward_context. Real vLLM
        # Attention.__init__ self-registers into this dict as a side
        # effect of construction; SelfBuiltAttentionPlaceholder (阶段7
        # item 2, replacing Attention construction in laguna_decoder.py)
        # does not, so this loop populates it externally instead -- the
        # rest of this method (and bf_attention.py/laguna_cuda_graph.py)
        # keeps reading from ``sfc``/``self.static_forward_context``
        # completely unchanged.
        from runtime.model.plain_attention import SelfBuiltAttentionPlaceholder

        sfc = vllm_config.compilation_config.static_forward_context
        for name, module in self.model.named_modules():
            if isinstance(module, SelfBuiltAttentionPlaceholder):
                sfc[name] = module
        self.static_forward_context = sfc
        self.attn_layer_names: list[str] = []
        for name, layer in sfc.items():
            if hasattr(layer, "get_attn_backend"):
                self.attn_layer_names.append(name)
        logger.info("Laguna: %d attention layers discovered", len(self.attn_layer_names))

        # Group layers by (num_qo_heads, num_kv_heads, window_left)
        # Each group gets sparkinfer impl patched
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

        # Patch attention layers to use sparkinfer instead of FlashInfer
        from runtime.backends.laguna_sparkinfer_attn import SparkinferAttentionImpl

        self._metadata_builders = {}  # kept for CG compat, unused for prefill
        for group_key, layer_names in self._layer_groups.items():
            wl, nqh, nkvh = group_key
            for name in layer_names:
                layer = sfc[name]
                layer.impl = SparkinferAttentionImpl(
                    num_heads=nqh,
                    head_size=128,
                    scale=128**-0.5,
                    num_kv_heads=nkvh,
                    window_left=wl,
                )
            logger.info(
                "SparkInfer attn patched: wl=%d, qh=%d, kvh=%d (%d layers)",
                wl,
                nqh,
                nkvh,
                len(layer_names),
            )

        # ── Classify layers: full attention vs SWA ──
        cache_dtype_str = vllm_config.cache_config.cache_dtype
        self._cache_dtype_str = cache_dtype_str
        self._full_layer_names: list[str] = []
        self._swa_layer_names: list[str] = []
        self._swa_window: int = 0
        # Reads SelfBuiltAttentionPlaceholder's own is_swa/sliding_window
        # attributes directly -- that's genuinely all this loop ever
        # needed off the old vLLM get_kv_cache_spec()-derived spec object
        # (阶段7 item 2; spec.block_size was never consumed -- verified by
        # grep, see runtime/model/plain_attention.py's module docstring).
        # Every layer here is a SelfBuiltAttentionPlaceholder unconditionally
        # since 任务#46 removed the QSR_LAGUNA_MODEL_LOADER=vllm escape
        # hatch (real vLLM Attention has no is_swa attribute, which is why
        # this used to need a hasattr() fallback -- no longer reachable).
        for name in self.attn_layer_names:
            layer = sfc[name]
            is_swa = layer.is_swa
            sliding_window = layer.sliding_window
            if is_swa:
                self._swa_layer_names.append(name)
                self._swa_window = sliding_window
            else:
                self._full_layer_names.append(name)

        self._ring_blocks_per_slot = (
            _ring_blocks_for_window(self._swa_window, block_size) if self._swa_window > 0 else 0
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
            is_swa = name in self._swa_layer_names
            n_blocks = ring_num_blocks if is_swa else full_num_blocks
            # Self-allocated KV cache in sparkinfer-native format:
            # [2, num_blocks, block_size, num_kv_heads, head_dim]
            # dim=0: 0=K, 1=V. FP8 stored as uint8.
            kv_dtype = (
                torch.uint8 if "fp8" in (cache_dtype_str or "") else layer.kv_cache_torch_dtype
            )
            shape = (2, n_blocks, block_size, layer.num_kv_heads, layer.head_size)
            self.kv_caches[name] = torch.zeros(shape, dtype=kv_dtype, device=self.device)
        runner_kv_caches: list[torch.Tensor] = []
        bind_laguna_kv_cache(self.kv_caches, sfc, runner_kv_caches)
        # Replace vLLM Attention modules with self-developed BFAttention
        from runtime.backends.bf_attention import replace_vllm_attention

        replace_vllm_attention(self.model, sfc, self.kv_caches)

        # ── Persistent prefill scratch for SWA layers (审查非阻断③) ──
        # Allocated once, reused across slots. Not zeroed (causal mask
        # guarantees no read-before-write within the window).
        self._swa_scratch: dict[str, torch.Tensor] = {}
        # SWA scratch: sized for overlap (window) + one prefill chunk.
        # Chunked prefill copies the last `window` tokens from ring into
        # scratch before each chunk, then processes chunk_tokens new tokens.
        # Total scratch capacity = window + chunk_tokens.
        _scratch_tokens = (
            self._swa_window if self._swa_window > 0 else 0
        ) + self._prefill_chunk_tokens
        self._swa_scratch_blocks = min(
            blocks_per_slot,
            -(-_scratch_tokens // block_size),  # cdiv
        )
        if self._swa_layer_names:
            for name in self._swa_layer_names:
                layer = sfc[name]
                kv_dtype = (
                    torch.uint8 if "fp8" in (cache_dtype_str or "") else layer.kv_cache_torch_dtype
                )
                shape = (
                    2,
                    self._swa_scratch_blocks,
                    block_size,
                    layer.num_kv_heads,
                    layer.head_size,
                )
                self._swa_scratch[name] = torch.empty(shape, dtype=kv_dtype, device=self.device)

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

        # M=1 decode CUDA Graph (lazily captured on first generate call)
        self._decode_cg = None
        self._decode_cg_enabled = _os.environ.get("QSR_DECODE_CUDA_GRAPH", "1") != "0"

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
        # E1: no GDN layers -- Laguna has no GDN/SSM recursive state, every
        # discovered layer is a (full or sliding-window) attention layer.
        # MTP fields start empty; ServerEngine._load_laguna_model flips them
        # (via dataclasses.replace on self.spec) once a DFlashEngine is
        # wired up via self._dflash, so classify_decode_slots routes greedy
        # requests through the MTP-shaped mtp_verify_and_commit_batch path.
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
        # E1: set by ServerEngine._load_laguna_model to a DFlashEngine
        # instance when speculative decoding is enabled for this backend
        # (requires capacity == 1 -- DFlash's draft/verify CUDA Graphs are
        # captured for a single physical slot, see DFlashEngine._init_buffers).
        self._dflash: Any = None
        self.num_qo_heads = sfc[self.attn_layer_names[0]].num_heads
        self.num_kv_heads = sfc[self.attn_layer_names[0]].num_kv_heads
        self.head_dim = sfc[self.attn_layer_names[0]].head_size

        logger.info(
            "LagunaBackend initialized: %d slots, block_size=%d",
            num_slots,
            block_size,
        )

    def _initialize_laguna_router(self) -> None:
        """Prepare the native router before any CUDA Graph capture."""
        from runtime.laguna_router import (
            LagunaRouterArena,
            LagunaRouterLibrary,
            router_max_rows,
        )

        max_rows = router_max_rows(
            self._prefill_chunk_tokens,
            self.num_slots,
            swa_qo_max=SWA_QO_MAX,
        )
        self._laguna_router_library = LagunaRouterLibrary.load()
        self._laguna_router_arena = LagunaRouterArena(max_rows, self.device)
        logger.info("Laguna native router: fixed output arena rows=%d", max_rows)

    def _warmup_laguna_router(self, correction_bias: torch.Tensor) -> None:
        """Resolve the native module before CUDA Graph capture, never in forward."""
        assert self._laguna_router_library is not None
        assert self._laguna_router_arena is not None
        logits = torch.zeros((1, 256), dtype=torch.float32, device=self.device)
        self._laguna_router_library.launch(
            logits,
            correction_bias,
            self._laguna_router_arena.weights,
            self._laguna_router_arena.ids,
        )
        torch.cuda.synchronize(self.device)
        del logits

    def _patch_moe_sparkinfer(self) -> None:
        """Replace MoE compute with sparkinfer for every LagunaMoESelfBuilt layer.

        Loads all NVFP4 expert weights AND the two activation global scales
        AND e_score_correction_bias directly from checkpoint (阶段6, vLLM
        removal plan: LagunaMoESelfBuilt no longer constructs vLLM's
        FusedMoE at all -- see runtime/model/laguna_decoder.py's module
        docstring for what was actually consumed from it before this
        change, and runtime/backends/laguna_sparkinfer_moe.py's
        load_moe_layer_activation_gscales for the verified-against-a-live-
        run formula this replaces). Patches each MoE layer's forward to:
        router → sparkinfer → shared.
        """
        from sparkinfer.moe.fused_moe._impl import allocate_tp_moe_workspace_pool

        from runtime.backends.laguna_sparkinfer_moe import (
            SparkinferMoELayer,
            SparkinferMoEOutputArena,
            _find_checkpoint,
            load_moe_layer_activation_gscales,
            load_moe_layer_e_score_correction_bias,
            load_moe_layer_weights,
            prepare_sparkinfer_layer,
            sparkinfer_version,
        )
        from runtime.model.laguna_decoder import LagunaMoESelfBuilt

        model = self.model
        hf_config = self.vllm_config.model_config.hf_config
        top_k = getattr(hf_config, "num_experts_per_tok", 10)
        renormalize = getattr(hf_config, "norm_topk_prob", True)
        softcap = getattr(hf_config, "moe_router_logit_softcapping", 0.0) or 0.0
        apply_on_input = getattr(hf_config, "moe_apply_router_weight_on_input", False)
        if top_k != 10 or not renormalize:
            raise RuntimeError(
                "native Laguna router requires num_experts_per_tok=10 and norm_topk_prob=True"
            )

        workspace = allocate_tp_moe_workspace_pool()
        # All patched forwards immediately consume routed output into a new
        # routed-plus-shared tensor, and ServerEngine owns one CUDA execution
        # thread.  One grow-only arena therefore replaces 47 long-prefill
        # allocations without making standalone SparkinferMoEModel callers
        # share output storage implicitly.
        output_arena = SparkinferMoEOutputArena()
        self._moe_sparkinfer_output_arena = output_arena
        native_router = self._laguna_router_library
        native_router_arena = self._laguna_router_arena
        assert native_router is not None
        assert native_router_arena is not None
        ckpt = _find_checkpoint()
        logger.info(
            "sparkinfer MoE patch (checkpoint-direct, alpha path): sparkinfer@%s",
            sparkinfer_version(),
        )

        patched = 0
        for name, module in model.named_modules():
            if not isinstance(module, LagunaMoESelfBuilt):
                continue

            parts = name.split(".")
            layer_idx = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                    except ValueError:
                        pass
            if layer_idx is None:
                continue

            moe_module = module
            shared_expert = getattr(moe_module, "shared_expert", None)
            routed_scaling = getattr(moe_module, "routed_scaling_factor", 1.0)
            e_bias = load_moe_layer_e_score_correction_bias(ckpt, layer_idx, self.device)

            # Load weights + activation global scales directly from checkpoint
            raw = load_moe_layer_weights(ckpt, layer_idx, self.device)
            a1g, a2g = load_moe_layer_activation_gscales(ckpt, layer_idx)
            si_experts = prepare_sparkinfer_layer(raw, self.device, a1_gscale=a1g, a2_gscale=a2g)
            del raw
            si_layer = SparkinferMoELayer(
                si_experts,
                workspace,
                self.device,
                # A no-shared-expert configuration returns routed output
                # directly, so it must retain private storage.
                output_arena=output_arena if shared_expert is not None else None,
            )
            self._moe_sparkinfer_layers.append(si_layer)

            def _make_patched_forward(
                moe_mod,
                _si_layer,
                _shared,
                _scaling,
                _renorm,
                _softcap,
                _e_bias,
                _top_k,
                _apply_on_input,
                _native_router,
                _native_router_arena,
            ):
                def _patched_forward(hidden_states: torch.Tensor) -> torch.Tensor:
                    orig_shape = hidden_states.shape
                    hs = hidden_states.view(-1, hidden_states.shape[-1])
                    # moe_mod.gate is PlainLinear (runtime/model/plain_linear.py),
                    # returns a plain tensor -- not vLLM's ReplicatedLinear
                    # (output, bias) tuple convention this line used to match.
                    router_logits = moe_mod.gate(hs)
                    router_logits = router_logits.float()
                    if _softcap > 0:
                        router_logits = torch.tanh(router_logits / _softcap) * _softcap
                    topk_weights, topk_ids = _native_router.launch(
                        router_logits,
                        _e_bias,
                        _native_router_arena.weights,
                        _native_router_arena.ids,
                    )
                    capture_routing(router_logits, topk_ids, topk_weights)  # bfprobe P-TOPK
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
                moe_module,
                si_layer,
                shared_expert,
                routed_scaling,
                renormalize,
                softcap,
                e_bias,
                top_k,
                apply_on_input,
                native_router,
                native_router_arena,
            )
            patched += 1
            if patched % 10 == 0:
                logger.info("sparkinfer MoE: patched %d layers...", patched)

        if patched == 0:
            raise RuntimeError("native Laguna router found no MoE layers to patch")
        self._warmup_laguna_router(e_bias)
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
                self._decode_block_table[i, n_blocks:] = full_base
            self._decode_slot_mapping[i] = (full_base + pos // bs) * bs + pos % bs

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
                    self._swa_decode_block_table[i, n_ring:] = ring_base

                self._swa_decode_seq_lens[i] = aligned_len

                # Ring slot_mapping for the new decode token
                ring_block = (pos % ring_slots) // bs
                ring_off = pos % bs
                self._swa_decode_slot_mapping[i] = (ring_base + ring_block) * bs + ring_off

    def _build_common_attn_metadata(
        self,
        slot_ids: list[int],
        kv_lengths: list[int],
        qo_lens: list[int],
        is_decode: bool,
    ):
        """Build owned metadata for full-attention layers."""
        num_reqs = len(slot_ids)
        num_actual_tokens = sum(qo_lens)
        page_size = self.block_size
        new_kv_lens = [kv_len + qo for kv_len, qo in zip(kv_lengths, qo_lens)]

        if is_decode and max(qo_lens) == 1:
            query_start_loc = self._decode_qsl_gpu[: num_reqs + 1]
            query_start_loc_cpu = self._decode_qsl_cpu[: num_reqs + 1]
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

        return LagunaAttentionMetadata(
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

    def _build_sparkinfer_metadata(
        self, common_meta, window_left: int = -1, mode: str | None = None
    ):
        """Convert owned attention metadata to ``SparkinferAttnMetadata``."""
        from runtime.backends.laguna_sparkinfer_attn import SparkinferAttnMetadata

        num_reqs = common_meta.num_reqs
        seq_lens = common_meta.seq_lens  # GPU tensor [num_reqs]
        block_table = common_meta.block_table_tensor  # [num_reqs, max_blocks]
        query_start_loc = common_meta.query_start_loc  # GPU [num_reqs+1]
        num_actual_tokens = common_meta.num_actual_tokens

        if mode is None:
            mode = "extend" if num_actual_tokens > num_reqs else "decode"
        return SparkinferAttnMetadata(
            mode=mode,
            page_table=block_table.int(),
            cache_seqlens=seq_lens.int(),
            cu_seqlens_q=query_start_loc.int(),
            num_actual_tokens=num_actual_tokens,
            window_left=window_left,
        )

    def _build_swa_attn_metadata(
        self,
        slot_ids: list[int],
        kv_lengths: list[int],
        qo_lens: list[int],
        is_decode: bool,
        swa_mode: str = "auto",
    ):
        """Build owned attention metadata for SWA layers.

        swa_mode: explicit routing — "decode_ring", "verify_ring",
                  "prefill_scratch", or "auto" (infer from is_decode/qo).
        """
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
            query_start_loc = self._decode_qsl_gpu[: num_reqs + 1]
            query_start_loc_cpu = self._decode_qsl_cpu[: num_reqs + 1]
            seq_lens = self._swa_decode_seq_lens[:num_reqs]
            max_blocks = max(int(self._swa_decode_seq_lens[i].item()) for i in range(num_reqs))
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

        return LagunaAttentionMetadata(
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

        # Build owned attention metadata.
        common_meta = self._build_common_attn_metadata(slot_ids, kv_lengths, qo_lens, is_decode)

        # Build sparkinfer attention metadata per group

        attn_metadata_dict: dict[str, Any] = {}
        slot_mapping_dict: dict[str, torch.Tensor] = {}

        # Full-attention metadata
        full_meta = self._build_sparkinfer_metadata(common_meta, window_left=-1)

        # SWA metadata (ring or scratch depending on mode)
        swa_meta_obj = None
        if self._ring_blocks_per_slot > 0 and self._swa_layer_names:
            effective_swa_kv = swa_kv_lengths if swa_kv_lengths is not None else kv_lengths
            swa_common = self._build_swa_attn_metadata(
                slot_ids, effective_swa_kv, qo_lens, is_decode
            )
            swa_wl = self._swa_window - 1 if self._swa_window > 0 else -1
            swa_meta_obj = self._build_sparkinfer_metadata(swa_common, window_left=swa_wl)

        for group_key, layer_names in self._layer_groups.items():
            wl = group_key[0]
            is_swa_group = wl >= 0
            meta = swa_meta_obj if (is_swa_group and swa_meta_obj is not None) else full_meta
            for name in layer_names:
                attn_metadata_dict[name] = meta
                slot_mapping_dict[name] = (
                    common_meta.slot_mapping
                    if not is_swa_group
                    else (swa_common.slot_mapping if swa_common else common_meta.slot_mapping)
                )

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

        with bf_attn_context(attn_metadata_dict, slot_mapping_dict):
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

    def _debug_check_ring_to_scratch_copy(
        self, slot: int, abs_start: int, count: int, chunk_idx: int
    ) -> None:
        """TEMP diagnostic (QSR_DEBUG_CHUNK_CHECK=1): mirror of
        _debug_check_scratch_to_ring_copy for the other copy direction
        (ring->scratch overlap at the start of a chunk). Destination is
        always scratch[0:count) per _copy_ring_to_scratch's contract.
        """
        if not self._swa_layer_names:
            return
        bs = self.block_size
        ring_slots = self._ring_slots_per_slot
        phys = _physical_slot(slot)
        ring_base = phys * self._ring_blocks_per_slot
        name = self._swa_layer_names[0]
        ring = self.kv_caches[name]
        scratch = self._swa_scratch[name]
        verbose = os.environ.get("QSR_DEBUG_CHUNK_CHECK") == "2"
        max_abs_diff = 0.0
        n_mismatch = 0
        sample_positions = sorted(set([0, count - 1] + list(range(0, count, max(1, count // 8)))))
        for i in sample_positions:
            a_pos = abs_start + i
            s_pos = i
            ring_slot_idx = a_pos % ring_slots
            rb, ro = ring_slot_idx // bs + ring_base, ring_slot_idx % bs
            sb, so = s_pos // bs, s_pos % bs
            rv = ring[:, rb, ro].float()
            sv = scratch[:, sb, so].float()
            diff = (sv - rv).abs().max().item()
            max_abs_diff = max(max_abs_diff, diff)
            if diff > 0:
                n_mismatch += 1
        status = "OK" if max_abs_diff == 0.0 else "MISMATCH"
        if status == "MISMATCH" or verbose:
            logger.warning(
                "CHUNK_CHECK chunk=%d bs=%d ring[%d:%d)->scratch[0:%d) status=%s "
                "max_abs_diff=%.6g n_mismatch=%d/%d",
                chunk_idx,
                bs,
                abs_start,
                abs_start + count,
                count,
                status,
                max_abs_diff,
                n_mismatch,
                len(sample_positions),
            )

    def _debug_check_scratch_to_ring_copy(
        self, slot: int, scratch_start: int, abs_start: int, count: int, chunk_idx: int
    ) -> None:
        """TEMP diagnostic (QSR_DEBUG_CHUNK_CHECK=1): verify _copy_scratch_to_ring
        actually landed the right values at the right ring addresses, by
        independently re-deriving both addresses token-by-token (no slab
        batching, so this can't share a bug with the code under test) and
        diffing. Self-consistent within one run -- doesn't need cross-
        block_size comparison. Prints only on mismatch or always if
        QSR_DEBUG_CHUNK_CHECK=2.
        """
        if not self._swa_layer_names:
            return
        bs = self.block_size
        ring_slots = self._ring_slots_per_slot
        phys = _physical_slot(slot)
        ring_base = phys * self._ring_blocks_per_slot
        name = self._swa_layer_names[0]
        ring = self.kv_caches[name]
        scratch = self._swa_scratch[name]
        verbose = os.environ.get("QSR_DEBUG_CHUNK_CHECK") == "2"
        max_abs_diff = 0.0
        n_mismatch = 0
        sample_positions = sorted(set([0, count - 1] + list(range(0, count, max(1, count // 8)))))
        for i in sample_positions:
            s_pos = scratch_start + i
            a_pos = abs_start + i
            sb, so = s_pos // bs, s_pos % bs
            ring_slot_idx = a_pos % ring_slots
            rb, ro = ring_slot_idx // bs + ring_base, ring_slot_idx % bs
            sv = scratch[:, sb, so].float()
            rv = ring[:, rb, ro].float()
            diff = (sv - rv).abs().max().item()
            max_abs_diff = max(max_abs_diff, diff)
            if diff > 0:
                n_mismatch += 1
        status = "OK" if max_abs_diff == 0.0 else "MISMATCH"
        if status == "MISMATCH" or verbose:
            logger.warning(
                "CHUNK_CHECK chunk=%d bs=%d scratch[%d:%d)->ring[%d:%d) status=%s "
                "max_abs_diff=%.6g n_mismatch=%d/%d",
                chunk_idx,
                bs,
                scratch_start,
                scratch_start + count,
                abs_start,
                abs_start + count,
                status,
                max_abs_diff,
                n_mismatch,
                len(sample_positions),
            )

    def _prefill_with_swa_scratch(self, slot: int, prompt_ids: list[int]) -> torch.Tensor:
        """Run prefill with SWA layers rebound to scratch, then copy to ring."""
        sfc = self.static_forward_context
        bs = self.block_size

        # Rebind SWA layers to scratch KV
        if self._swa_scratch:
            for name in self._swa_layer_names:
                sfc[name].kv_cache = self._swa_scratch[name]

        try:
            logits = self._forward([slot], prompt_ids, [0], qo_len=len(prompt_ids), is_decode=False)

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
                        ring[:, db, do : do + count] = scratch[:, sb, so : so + count]
        finally:
            # Always rebind SWA layers back to ring KV (审查 P3a)
            if self._swa_scratch:
                for name in self._swa_layer_names:
                    sfc[name].kv_cache = self.kv_caches[name]

        return logits

    def _copy_ring_to_scratch(self, slot: int, abs_start: int, count: int) -> None:
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
                scratch[:, db, do : do + n] = ring[:, sb, so : so + n]

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
                ring[:, db, do : do + n] = scratch[:, sb, so : so + n]

    def _prefill_with_swa_chunked(self, slot: int, prompt_ids: list[int]) -> torch.Tensor:
        """Chunked prefill for prompts longer than SWA scratch capacity.

        Each chunk: copy last `window` tokens from ring → scratch (overlap),
        then process chunk_tokens new tokens. Full-attention layers use
        absolute kv_length; SWA layers use relative positions in scratch.
        """
        sfc = self.static_forward_context
        chunk_tokens = self._prefill_chunk_tokens
        prompt_len = len(prompt_ids)
        window = self._swa_window

        all_logits = None
        chunk_ranges = _prefill_chunk_ranges(
            0,
            prompt_len,
            chunk_tokens,
            min_final_tokens=window,
        )
        for chunk_start, chunk_end in chunk_ranges:
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
                    [slot],
                    chunk,
                    [chunk_start],
                    qo_len=chunk_len,
                    is_decode=False,
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
            raise RuntimeError(f"slot {slot} is not fresh (kv_len={self.slot_kv_len[slot]})")
        if self._swa_scratch:
            prompt_len = len(prompt_ids)
            if prompt_len <= self._prefill_chunk_tokens:
                logits = self._prefill_with_swa_scratch(slot, prompt_ids)
            else:
                logits = self._prefill_with_swa_chunked(slot, prompt_ids)
        else:
            logits = self._forward([slot], prompt_ids, [0], qo_len=len(prompt_ids), is_decode=False)
        first_token = int(logits[-1].argmax(dim=-1).item())
        self.slot_kv_len[slot] = len(prompt_ids)
        self.slot_committed_tokens[slot] = list(prompt_ids) + [first_token]
        return first_token

    def prefill_sampled(self, slot: int, prompt_ids: list[int], params: SamplingParams) -> int:
        if self.slot_kv_len[slot] != 0:
            raise RuntimeError(f"slot {slot} is not fresh (kv_len={self.slot_kv_len[slot]})")
        if self._swa_scratch:
            logits = self._prefill_with_swa_scratch(slot, prompt_ids)
        else:
            logits = self._forward([slot], prompt_ids, [0], qo_len=len(prompt_ids), is_decode=False)
        last_logits = logits[-1].unsqueeze(0)
        gen = make_generator(params.seed)
        first_token = int(sample_from_logits(last_logits, params, generator=gen).item())
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
            raise RuntimeError(f"slot {slot} is not fresh (kv_len={self.slot_kv_len[slot]})")
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
                        ring[:, db, do : do + count] = scratch[:, sb, so : so + count]
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

            chunk_ranges = _prefill_chunk_ranges(
                0,
                prompt_len,
                PREFILL_CHUNK,
                min_final_tokens=window,
            )
            debug_chunk_check = os.environ.get("QSR_DEBUG_CHUNK_CHECK") in ("1", "2")
            for _chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_ranges):
                chunk = prompt_ids[chunk_start:chunk_end]
                chunk_len = len(chunk)
                is_last = chunk_end == prompt_len

                # Overlap: copy last `window` tokens from ring to scratch
                overlap = min(window, chunk_start) if self._swa_scratch else 0
                if overlap > 0:
                    self._copy_ring_to_scratch(slot, chunk_start - overlap, overlap)
                    if debug_chunk_check:
                        self._debug_check_ring_to_scratch_copy(
                            slot, chunk_start - overlap, overlap, _chunk_idx
                        )
                if debug_chunk_check:
                    logger.warning(
                        "CHUNK_CHECK chunk=%d bs=%d chunk_start=%d chunk_end=%d overlap=%d",
                        _chunk_idx,
                        self.block_size,
                        chunk_start,
                        chunk_end,
                        overlap,
                    )

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
                            [slot],
                            chunk,
                            [chunk_start],
                            qo_len=chunk_len,
                            is_decode=False,
                            swa_kv_lengths=swa_kv,
                        )
                    else:
                        self._forward(
                            [slot],
                            chunk,
                            [chunk_start],
                            qo_len=chunk_len,
                            is_decode=False,
                            swa_kv_lengths=swa_kv,
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
                        if debug_chunk_check:
                            self._debug_check_scratch_to_ring_copy(
                                slot, copy_scratch_start, copy_abs_start, copy_count, _chunk_idx
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

        common_meta = self._build_common_attn_metadata(slot_ids, kv_lengths, qo_lens, is_decode)

        attn_metadata_dict: dict[str, Any] = {}
        slot_mapping_dict: dict[str, torch.Tensor] = {}

        full_meta = self._build_sparkinfer_metadata(common_meta, window_left=-1)

        swa_meta_obj = None
        swa_common = None
        if self._ring_blocks_per_slot > 0 and self._swa_layer_names:
            effective_swa_kv = swa_kv_lengths if swa_kv_lengths is not None else kv_lengths
            swa_common = self._build_swa_attn_metadata(
                slot_ids, effective_swa_kv, qo_lens, is_decode
            )
            swa_wl = self._swa_window - 1 if self._swa_window > 0 else -1
            swa_meta_obj = self._build_sparkinfer_metadata(swa_common, window_left=swa_wl)

        for group_key, layer_names in self._layer_groups.items():
            wl = group_key[0]
            is_swa_group = wl >= 0
            meta = swa_meta_obj if (is_swa_group and swa_meta_obj is not None) else full_meta
            for name in layer_names:
                attn_metadata_dict[name] = meta
                slot_mapping_dict[name] = (
                    common_meta.slot_mapping
                    if not is_swa_group
                    else (swa_common.slot_mapping if swa_common else common_meta.slot_mapping)
                )

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

        with bf_attn_context(attn_metadata_dict, slot_mapping_dict):
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

    def decode_sampled(self, slot: int, token_id: int, params: SamplingParams) -> int:
        kv_len = self.slot_kv_len[slot]
        logits = self._forward([slot], [token_id], [kv_len], qo_len=1, is_decode=True)
        last_logits = logits[-1].unsqueeze(0)
        gen = make_generator(params.seed)
        next_token = int(sample_from_logits(last_logits, params, generator=gen).item())
        self.slot_kv_len[slot] += 1
        self.slot_committed_tokens[slot].append(token_id)
        return next_token

    def decode_batch(self, slot_ids: list[int], token_ids: list[int]) -> list[int]:
        kv_lengths = [self.slot_kv_len[s] for s in slot_ids]
        logits = self._forward(slot_ids, token_ids, kv_lengths, qo_len=1, is_decode=True)
        next_tokens = []
        for i, slot in enumerate(slot_ids):
            next_token = int(logits[i].argmax(dim=-1).item())
            next_tokens.append(next_token)
            self.slot_kv_len[slot] += 1
            self.slot_committed_tokens[slot].append(token_ids[i])
        return next_tokens

    def _decode_cg_batch_eligible(
        self,
        slot_ids: list[int],
        params_list: list[SamplingParams],
        return_logprobs: bool,
    ) -> bool:
        """Whether this decode step can safely replay the captured decode CG.

        The graph is captured once at a fixed batch size (see
        ``_ensure_decode_cg`` -- currently 1, matching Laguna's default
        server capacity). Replaying at a different batch size would run the
        graph's fixed-shape kernels over stale rows left by a previous
        replay: since ``_physical_slot`` maps 1:1 onto real KV-cache slot
        ranges, any padding row would write live attention output into
        whatever physical slot its stale metadata happens to point at --
        silently corrupting a slot outside this batch, not just wasting
        compute. Only an exact batch-size match is provably safe, so this
        stays conservative instead of attempting padding.

        CG replay also only returns an argmax token id (greedy is baked
        into the captured graph itself, see ``LagunaCudaGraphDecode.capture``),
        so any request needing logprobs or non-greedy sampling must fall
        back to eager.
        """
        return (
            self._decode_cg is not None
            and not return_logprobs
            and len(slot_ids) == self._decode_cg.batch_size
            and all(p.is_greedy for p in params_list)
        )

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

        When the batch is a single exact-size, all-greedy, non-logprobs
        match for the captured decode CUDA Graph (see
        ``_decode_cg_batch_eligible``), replays it instead of running eager
        forward -- this is the path ``ServerEngine._step_sync`` actually
        calls, so it is what makes CUDA Graph decode reach real requests.
        """
        _bf_cg_ok = self._decode_cg_batch_eligible(slot_ids, params_list, return_logprobs)
        if bfdiag_trace.TRACE_ENABLED:
            bfdiag_trace.record_decode_batch_path(
                slot_ids, kv_lengths, self._decode_cg, _bf_cg_ok, return_logprobs, params_list
            )
        if _bf_cg_ok:
            next_tokens = self._decode_cg.replay(slot_ids, token_ids, kv_lengths)
            for i, slot in enumerate(slot_ids):
                self.slot_kv_len[slot] += 1
                self.slot_committed_tokens[slot].append(token_ids[i])
            return next_tokens

        logits = self._forward(slot_ids, token_ids, kv_lengths, qo_len=1, is_decode=True)
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
        # Full-attention layers: clear blocks_per_slot blocks.
        # The block axis is dim 1 -- kv_cache is
        # [2, num_blocks, block_size, num_kv_heads, head_dim] with dim 0
        # being K/V (see the allocation above). Slicing dim 0 instead
        # silently clamps to [0:2] for slot 0 (wiping EVERY slot's blocks)
        # and yields an empty slice for every slot >= 1 (clearing nothing
        # at all, so the previous request's KV leaks into the next one).
        # It went unnoticed because DFlash pins capacity to 1, where
        # num_blocks == blocks_per_slot makes both forms equivalent.
        full_start = phys * self.blocks_per_slot
        full_end = full_start + self.blocks_per_slot
        for name in self._full_layer_names:
            self.kv_caches[name][:, full_start:full_end].zero_()
        # SWA layers: clear only ring_blocks_per_slot blocks
        if self._ring_blocks_per_slot > 0:
            ring_start = phys * self._ring_blocks_per_slot
            ring_end = ring_start + self._ring_blocks_per_slot
            for name in self._swa_layer_names:
                self.kv_caches[name][:, ring_start:ring_end].zero_()

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
            if self._dflash is not None:
                result[slot] = self._dflash.dflash_prefill_bootstrap(slot, prompt)
            else:
                first_token = self.prefill(slot, prompt)
                result[slot] = {"anchor": first_token, "draft_tokens": []}
        return ChunkedPrefillState(done=True, result=result)

    def prefill_chunked_step(self, state: ChunkedPrefillState) -> bool:
        """Laguna prefill is never incremental; state is always already done."""
        return state.done

    def mtp_verify_and_commit_batch(
        self,
        slots: list[int],
        anchors: dict[int, int],
        drafts: dict[int, list[int]],
        *,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> dict[int, dict]:
        """E1: DFlash's sibling of ``DirectModelRunner.mtp_verify_and_commit_batch``,
        called by the SAME ``ServerEngine._step_sync`` greedy-MTP branch
        (``classify_decode_slots`` routes here once ``self.spec.has_mtp`` is
        true -- see ``self._dflash`` / ``ServerEngine._load_laguna_model``).

        DFlash's draft/verify CUDA Graphs are captured for exactly one
        physical slot (see ``DFlashEngine._init_buffers``), so this only
        ever runs with ``len(slots) == 1`` in practice --
        ``ServerEngine.__init__`` requires ``capacity == 1`` whenever DFlash
        is enabled. The loop below still handles >1 slots correctly (just
        sequentially, no batched replay) rather than silently assuming the
        constraint, in case that capacity guard is ever loosened without
        updating this method.
        """
        if self._dflash is None:
            raise RuntimeError("mtp_verify_and_commit_batch called without a DFlashEngine wired up")
        return {
            slot: self._dflash.dflash_round(
                slot,
                anchors[slot],
                drafts[slot],
                return_logprobs=return_logprobs,
                top_logprobs=top_logprobs,
            )
            for slot in slots
        }

    def _ensure_decode_cg(self) -> None:
        """Lazily capture M=1 decode CUDA Graph on first use."""
        if not self._decode_cg_enabled or self._decode_cg is not None:
            return
        from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode

        try:
            cg = LagunaCudaGraphDecode(self, batch_size=1)
            cg.capture()
            self._decode_cg = cg
            logger.info("Laguna: M=1 decode CUDA Graph captured")
        except Exception as e:
            logger.warning("Laguna: decode CG capture failed (falling back to eager): %s", e)
            self._decode_cg_enabled = False

    def _unpatch_impls_for_prefill(self) -> None:
        """Restore original attention impls so prefill works after CG capture."""
        if self._decode_cg is not None:
            self._decode_cg.unpatch_impls()

    def _repatch_impls_for_cg(self) -> None:
        """Re-apply CG decode impls after prefill."""
        if self._decode_cg is not None:
            self._decode_cg.repatch_impls()

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

        # Use CUDA Graph for greedy decode (temperature=0)
        if temperature == 0 and self._decode_cg_enabled:
            self._ensure_decode_cg()
            if self._decode_cg is not None:
                self._decode_cg.reset()
                tokens = self._decode_cg.generate_fast(
                    slot=slot, first_token=first, max_tokens=max_tokens, eos_tokens=(2, 24)
                )
                self.reset_slot(slot)
                return tokens

        # Fallback: eager decode
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
            logits = self._forward([slot], [last_token], [start_pos - 1], qo_len=1, is_decode=True)
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
            # Copy overlap from ring to scratch
            overlap = min(window, start_pos)
            if overlap > 0:
                self._copy_ring_to_scratch(slot, start_pos - overlap, overlap)

            for name in self._swa_layer_names:
                sfc[name].kv_cache = self._swa_scratch[name]
            try:
                suffix = prompt_ids[start_pos:]
                logits, aux = self._forward_with_aux(
                    [slot],
                    suffix,
                    [start_pos],
                    qo_len=suffix_len,
                    is_decode=False,
                    swa_kv_lengths=[overlap],
                )
                # Copy last window from scratch to ring
                total_in_scratch = overlap + suffix_len
                copy_count = min(window, total_in_scratch)
                copy_scratch_start = total_in_scratch - copy_count
                copy_abs_start = start_pos + suffix_len - copy_count
                self._copy_scratch_to_ring(slot, copy_scratch_start, copy_abs_start, copy_count)
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

            chunk_ranges = _prefill_chunk_ranges(
                start_pos,
                prompt_len,
                PREFILL_CHUNK,
                min_final_tokens=window,
            )
            for chunk_start, chunk_end in chunk_ranges:
                chunk = prompt_ids[chunk_start:chunk_end]
                chunk_len = len(chunk)
                is_last = chunk_end == prompt_len

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
                            [slot],
                            chunk,
                            [chunk_start],
                            qo_len=chunk_len,
                            is_decode=False,
                            swa_kv_lengths=swa_kv,
                        )
                    else:
                        self._forward(
                            [slot],
                            chunk,
                            [chunk_start],
                            qo_len=chunk_len,
                            is_decode=False,
                            swa_kv_lengths=swa_kv,
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
