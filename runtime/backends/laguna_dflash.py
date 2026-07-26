"""DFlash Speculative Decoding Engine for Laguna-S-2.1.

Integrates the DFlash draft model with the main Laguna backend to achieve
~25× decode speedup via parallel draft + verify speculative decoding.

Architecture:
- Main model: 48 layers (12 full + 36 SWA), NVFP4 quantized
- Draft model: 6 layers (all SWA window=512), bf16, shares embed+lm_head
- Aux hidden states extracted at layers [2, 11, 20, 30, 39, 48] (vLLM post-layer indexing)
- combine_hidden_states: concat 6×[N,3072] → fc → hidden_norm → [N,3072]
- precompute_and_store_context_kv: project combined → draft KV cache
- Draft forward: 16 tokens (1 bonus + 15 mask) → sample 15 draft tokens
- Verify: main model forward 16 tokens → greedy accept/reject

Pipeline per speculative step:
1. Main decode (1 token) → logits + aux_hidden_states
2. combine + precompute_context_kv → draft KV updated
3. Draft forward (16 tokens) → 15 draft tokens
4. Main verify (16 tokens) → accept/reject
5. Accept N tokens → next step starts from token N+1
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import torch

from runtime.backends.bf_attention import bf_attn_context
from runtime.backends.dflash_constants import (
    AUX_LAYER_IDS,
    DFLASH_MODEL_PATH,
    DRAFT_HEAD_DIM,
    DRAFT_NUM_KV_HEADS,
    DRAFT_NUM_LAYERS,
    DRAFT_NUM_QO_HEADS,
    DRAFT_WINDOW,
    MASK_TOKEN_ID,
    NUM_QUERY_PER_REQ,
    NUM_SPECULATIVE_TOKENS,
)
from runtime.backends.laguna import LagunaBackend, _physical_slot, _ring_blocks_for_window
from runtime.compat_vllm import (
    VllmConfig,
    set_current_vllm_config,
    set_forward_context,
)
from runtime.mtp_accept import determine_accept_reject_from_predictions

logger = logging.getLogger("qwen_sm120_runtime.dflash")


def _greedy_accept_reject(
    verify_argmax: list[int],
    draft_tokens: list[int],
    bonus_token: int,
) -> tuple[list[int], int]:
    """Greedy speculative-decode accept/reject: walk draft_tokens in order,
    accepting each one that matches the target model's own argmax at that
    position; on the first mismatch, accept the target's correction instead
    and stop (everything after a rejection is discarded and redrafted next
    round). Returns (accepted_tokens including bonus_token, num_accepted).

    Pure function (no GPU/model access) so both the eager (_verify) and
    CUDA-Graph (_accept_reject) verify paths can share one implementation
    instead of keeping two copies of this loop in sync by hand.
    """
    accepted = [bonus_token]
    num_accepted = 0
    for verify_tok, draft_tok in zip(verify_argmax, draft_tokens):
        if verify_tok == draft_tok:
            accepted.append(draft_tok)
            num_accepted += 1
        else:
            accepted.append(verify_tok)
            num_accepted += 1
            break
    return accepted, num_accepted


def _verify_only_accept_reject(
    all_argmax: list[int],
    draft_tokens: list[int],
    bonus_token: int,
) -> dict:
    """Resolve one verify-only round without conflating output and KV state."""
    decision = determine_accept_reject_from_predictions(
        [bonus_token] + draft_tokens,
        all_argmax,
    )
    committed = decision["committed"]
    return {
        **decision,
        # The old anchor and matching drafts were verifier inputs and now have
        # valid target/draft context KV. The recovery/bonus remains pending.
        "context_count": 1 + decision["num_accepted"],
        "next_anchor": committed[-1],
    }


class DFlashEngine:
    """DFlash speculative decoding engine wrapping LagunaBackend.

    Manages the draft model, its KV cache, and the speculative decode loop.
    """

    def __init__(
        self,
        backend: LagunaBackend,
        dflash_model_path: str | None = None,
    ) -> None:
        self.backend = backend
        self.device = backend.device
        self.vllm_config = backend.vllm_config
        self.block_size = backend.block_size
        self.num_slots = backend.num_slots

        # Verify SWA layers are rebound to ring KV (审查 P3a: rebind leak guard)
        backend.assert_swa_rebind()

        # Load DFlash draft model
        self.draft_model = self._load_draft_model(dflash_model_path)

        # Set aux hidden state layers on main model
        self._enable_aux_hidden_states()

        # Allocate draft KV cache and bind to draft model
        self._alloc_draft_kv_cache()

        # Patch draft attention layers to use sparkinfer (replaces FlashInfer)
        self._patch_draft_sparkinfer()

        # FlashInfer metadata builder skipped — sparkinfer handles draft attention
        # self._init_draft_metadata_builder()

        # Pre-allocated buffers
        self._init_buffers()

        # CUDA Graph for main model decode (M=1) -- captured lazily on first
        # use by _ensure_decode_cg(), see there for why.
        self._cuda_graph = None
        self._decode_cg_attempted = False
        self._verify_cg = None
        self._draft_cg = None
        self._cg_captured = False
        self._use_cuda_graph = os.environ.get("QSR_DFLASH_CUDA_GRAPH", "1") != "0"
        if self._use_cuda_graph:
            self._init_cuda_graph()

        logger.info(
            "DFlashEngine initialized: K=%d speculative tokens, draft %d layers, cuda_graph=%s",
            NUM_SPECULATIVE_TOKENS, DRAFT_NUM_LAYERS, self._use_cuda_graph,
        )

    def _load_draft_model(self, model_path: str | None) -> Any:
        """Load the DFlash draft model via vLLM's load_dflash_model."""
        from vllm.config import ModelConfig, SpeculativeConfig
        from vllm.config import replace as vllm_replace

        if model_path is None:
            model_path = os.path.expanduser(DFLASH_MODEL_PATH)

        # Build SpeculativeConfig for the draft model
        target_model_config = self.vllm_config.model_config
        spec_config = SpeculativeConfig(
            model=model_path,
            method="dflash",
            num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
            target_model_config=target_model_config,
            target_parallel_config=self.vllm_config.parallel_config,
        )
        # Set draft_model_config directly (skip full post-init resolution)
        spec_config.draft_model_config = ModelConfig(
            model=model_path,
            runner="draft",
            tokenizer=target_model_config.tokenizer,
            tokenizer_mode=target_model_config.tokenizer_mode,
            trust_remote_code=target_model_config.trust_remote_code,
            dtype=target_model_config.dtype,
            seed=target_model_config.seed,
            max_model_len=DRAFT_WINDOW + NUM_QUERY_PER_REQ + 128,
            spec_target_max_model_len=target_model_config.max_model_len,
            enforce_eager=True,
        )

        draft_vllm_config = vllm_replace(
            self.vllm_config,
            speculative_config=spec_config,
        )

        # Use vLLM's load_dflash_model which handles weight sharing
        from vllm.v1.worker.gpu.spec_decode.dflash.utils import load_dflash_model

        with set_current_vllm_config(draft_vllm_config):
            draft_model = load_dflash_model(
                target_model=self.backend.model,
                vllm_config=draft_vllm_config,
            )

        draft_model.eval()
        logger.info("DFlash draft model loaded from %s", model_path)
        return draft_model

    def _enable_aux_hidden_states(self) -> None:
        """Enable aux hidden state extraction on the main model."""
        model = self.backend.model
        # SupportsEagle3 interface
        if hasattr(model, "set_aux_hidden_state_layers"):
            model.set_aux_hidden_state_layers(AUX_LAYER_IDS)
        elif hasattr(model, "model") and hasattr(model.model, "_set_aux_hidden_state_layers"):
            model.model._set_aux_hidden_state_layers(AUX_LAYER_IDS)
        else:
            raise RuntimeError(
                "Main model does not support aux hidden state extraction. "
                "Expected SupportsEagle3 interface."
            )
        logger.info("Aux hidden state layers enabled: %s", AUX_LAYER_IDS)

    def _alloc_draft_kv_cache(self) -> None:
        """Allocate KV cache for the draft model's 6 SWA layers."""
        from runtime.compat_vllm import bind_kv_cache

        # Discover draft model's attention layers from static_forward_context
        sfc = self.vllm_config.compilation_config.static_forward_context

        self._draft_layer_names: list[str] = []
        self._draft_attn_layers: dict[str, Any] = {}

        for name, layer in sfc.items():
            if not hasattr(layer, "get_attn_backend"):
                continue
            # Extract layer index from name
            parts = name.split(".")
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                    except ValueError:
                        pass
                    break
            # Draft layers have indices >= 48 (main model's num_hidden_layers)
            if layer_idx is not None and layer_idx >= 48:
                self._draft_layer_names.append(name)
                self._draft_attn_layers[name] = layer

        if not self._draft_layer_names:
            # Fallback: discover from draft model directly
            draft_inner = (
                self.draft_model.model
                if hasattr(self.draft_model, "model")
                else self.draft_model
            )
            if hasattr(draft_inner, "layers"):
                for layer in draft_inner.layers:
                    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "attn"):
                        attn = layer.self_attn.attn
                        name = attn.layer_name
                        self._draft_layer_names.append(name)
                        self._draft_attn_layers[name] = attn

        logger.info(
            "DFlash: %d draft attention layers discovered",
            len(self._draft_layer_names),
        )

        # Allocate KV cache for draft layers
        num_phys = self.num_slots + 1  # +1 reserved
        draft_blocks_per_slot = _ring_blocks_for_window(
            DRAFT_WINDOW, self.block_size, NUM_QUERY_PER_REQ
        )
        self._draft_blocks_per_slot = draft_blocks_per_slot
        total_blocks = num_phys * draft_blocks_per_slot

        self._draft_kv_caches: dict[str, torch.Tensor] = {}
        for name in self._draft_layer_names:
            attn = self._draft_attn_layers[name]
            # Self-allocated: [blocks, 2, bs, kv_heads, head_dim], FP8 as uint8
            shape = (total_blocks, 2, self.block_size, attn.num_kv_heads, attn.head_size)
            self._draft_kv_caches[name] = torch.zeros(
                shape, dtype=torch.uint8, device=self.device
            )

        # Bind draft KV caches to draft attention layers
        bind_kv_cache(self._draft_kv_caches, self._draft_attn_layers, [])
        logger.info(
            "DFlash: draft KV allocated: %d blocks/slot × %d layers",
            draft_blocks_per_slot, len(self._draft_layer_names),
        )


    def _patch_draft_sparkinfer(self) -> None:
        """Patch draft model attention layers to use sparkinfer (zero FlashInfer dep)."""
        from runtime.backends.laguna_sparkinfer_attn import SparkinferAttentionImpl
        for name in self._draft_layer_names:
            attn = self._draft_attn_layers[name]
            attn.impl = SparkinferAttentionImpl(
                num_heads=attn.num_heads, head_size=attn.head_size,
                scale=attn.head_size ** -0.5,
                num_kv_heads=attn.num_kv_heads, window_left=DRAFT_WINDOW - 1,
            )
        logger.info("DFlash: draft attention patched with sparkinfer (%d layers)",
                    len(self._draft_layer_names))

    def _init_draft_metadata_builder(self) -> None:
        """Initialize FlashInfer metadata builder for draft model attention."""
        from runtime.compat_vllm import get_flashinfer_metadata_builder

        FlashInferMetadataBuilder = get_flashinfer_metadata_builder()

        # All draft layers share the same config (72 QO / 8 KV, SWA window=512)
        first_attn = self._draft_attn_layers[self._draft_layer_names[0]]
        kv_cache_spec = first_attn.get_kv_cache_spec(self.vllm_config)

        with set_current_vllm_config(self.vllm_config):
            self._draft_metadata_builder = FlashInferMetadataBuilder(
                kv_cache_spec=kv_cache_spec,
                layer_names=self._draft_layer_names,
                vllm_config=self.vllm_config,
                device=self.device,
            )
        self._draft_metadata_builder.disable_split_kv = True
        logger.info("DFlash: FlashInfer metadata builder initialized for draft")

    def _init_buffers(self) -> None:
        """Pre-allocate buffers for the speculative decode loop."""
        device = self.device
        max_tokens = NUM_QUERY_PER_REQ  # 16

        # Draft input buffers
        self._draft_input_ids = torch.zeros(max_tokens, dtype=torch.long, device=device)
        self._draft_positions = torch.zeros(max_tokens, dtype=torch.long, device=device)

        # Draft attention metadata buffers
        self._draft_seq_lens = torch.zeros(1, dtype=torch.int32, device=device)
        self._draft_block_table = torch.zeros(
            1, self._draft_blocks_per_slot, dtype=torch.int32, device=device
        )
        self._draft_slot_mapping = torch.zeros(max_tokens, dtype=torch.long, device=device)
        self._draft_qsl = torch.tensor([0, max_tokens], dtype=torch.int32, device=device)
        self._draft_qsl_cpu = torch.tensor([0, max_tokens], dtype=torch.int32)

    def _init_cuda_graph(self) -> None:
        """Set up CUDA Graph state for verify and draft.

        Verify CG (M=16) is captured eagerly — it uses extend-mode sparkinfer
        workspaces and is deterministic across KV lengths. This eliminates
        the MoE kernel non-determinism that causes 0% acceptance in eager mode.
        """
        self._verify_cg = None
        self._draft_cg = None
        self._cg_captured = False

        # Capture verify CG immediately (deterministic MoE via CG replay)
        if self._use_cuda_graph:
            self._capture_verify_cg()

    def _capture_verify_cg(self) -> None:
        """Capture M=16 verify CUDA Graph with sparkinfer extend attention."""
        from runtime.backends.laguna_cuda_graph import LagunaCudaGraphVerify
        try:
            cg = LagunaCudaGraphVerify(self.backend, num_tokens=NUM_QUERY_PER_REQ)
            cg.capture()
            self._verify_cg = cg
            logger.info("DFlash: verify CG captured (M=%d)", NUM_QUERY_PER_REQ)
        except Exception as e:
            logger.warning("DFlash: verify CG capture failed: %s", e)
            import traceback; traceback.print_exc()
            self._verify_cg = None

    def _ensure_decode_cg(self) -> None:
        """Lazily capture the M=1 main-decode CUDA Graph on first use.

        Unlike verify/draft's lazy capture, this isn't gated on KV cache
        state: LagunaCudaGraphDecode.capture() warms up on backend's own
        reserved slots (never slot 0), so it's safe to capture at any point,
        including before the first prefill -- lazy here purely to skip the
        cost entirely for callers (generate_verify_only) that never need it.
        """
        if not self._use_cuda_graph or self._cuda_graph is not None:
            return
        from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode

        try:
            cg = LagunaCudaGraphDecode(self.backend, batch_size=1)
            cg.capture()
            self._cuda_graph = cg
            logger.info("DFlash: CUDA Graph captured for main decode (M=1, lazy)")
        except Exception as e:
            logger.warning("DFlash: main decode CUDA Graph failed: %s", e)
            self._cuda_graph = None

    def _forward_main_with_aux(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
        qo_len: int = 1,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Run main model forward and return (logits, aux_hidden_states)."""
        backend = self.backend
        num_reqs = len(slot_ids)
        qo_lens = [qo_len] * num_reqs
        is_decode = qo_len == 1

        if is_decode:
            backend._fill_decode_buffers(slot_ids, token_ids, kv_lengths)

        # Build attention metadata
        common_meta = backend._build_common_attn_metadata(
            slot_ids, kv_lengths, qo_lens, is_decode
        )

        attn_metadata_dict: dict[str, Any] = {}
        slot_mapping_dict: dict[str, torch.Tensor] = {}

        swa_meta = None
        if backend._ring_blocks_per_slot > 0 and backend._swa_layer_names:
            # Explicit routing: qo>1 with ring active = verify, not prefill
            mode = "verify_ring" if qo_len > 1 else "decode_ring"
            swa_meta = backend._build_swa_attn_metadata(
                slot_ids, kv_lengths, qo_lens, is_decode, swa_mode=mode
            )

        full_spark_meta = backend._build_sparkinfer_metadata(common_meta, window_left=-1)
        swa_spark_meta = None
        if swa_meta is not None:
            swa_wl = backend._swa_window - 1 if backend._swa_window > 0 else -1
            swa_spark_meta = backend._build_sparkinfer_metadata(swa_meta, window_left=swa_wl)
        for group_key, layer_names in backend._layer_groups.items():
            wl = group_key[0]
            is_swa_group = wl >= 0
            meta = swa_spark_meta if (is_swa_group and swa_spark_meta is not None) else full_spark_meta
            for name in layer_names:
                attn_metadata_dict[name] = meta
                slot_mapping_dict[name] = common_meta.slot_mapping if not is_swa_group else (swa_meta.slot_mapping if swa_meta else common_meta.slot_mapping)

        # Build input tensors
        if is_decode:
            input_ids = backend._decode_input_ids[:num_reqs]
            positions = backend._decode_positions[:num_reqs]
        else:
            if num_reqs == 1:
                flat_token_ids = token_ids
            else:
                flat_token_ids = [
                    tok for slot_tokens in token_ids for tok in slot_tokens
                ]
            input_ids = torch.tensor(
                flat_token_ids, dtype=torch.long, device=self.device
            )
            positions_list = []
            for kv_len, qo in zip(kv_lengths, qo_lens):
                positions_list.extend(range(kv_len, kv_len + qo))
            positions = torch.tensor(
                positions_list, dtype=torch.long, device=self.device
            )

        with bf_attn_context(attn_metadata_dict, slot_mapping_dict):
            with set_forward_context(
                attn_metadata_dict, backend.vllm_config, slot_mapping=slot_mapping_dict,
                skip_compiled=True,
            ):
                result = backend.model.forward(input_ids, positions)

        # Handle tuple return (hidden_states, aux_hidden_states)
        if isinstance(result, tuple):
            hidden_states, aux_hidden_states = result
        else:
            hidden_states = result
            aux_hidden_states = None

        logits = backend.model.compute_logits(hidden_states)
        return logits, aux_hidden_states

    def _build_draft_attn_metadata(self, slot: int, kv_len: int, num_tokens: int):
        """Build CommonAttentionMetadata for draft model forward.

        Draft KV cache is a ring buffer (SWA window=512). All positions
        must be wrapped modulo ring_slots to avoid OOB writes.
        """
        from runtime.compat_vllm import get_common_attn_metadata_cls

        CommonAttentionMetadata = get_common_attn_metadata_cls()

        bs = self.block_size
        phys = _physical_slot(slot)
        draft_base = phys * self._draft_blocks_per_slot
        ring_slots = self._draft_blocks_per_slot * bs
        new_kv_len = kv_len + num_tokens

        # Ring block table: cover the SWA window aligned to block boundary
        window_start = max(0, kv_len - DRAFT_WINDOW + 1)
        aligned_start = (window_start // bs) * bs
        aligned_len = new_kv_len - aligned_start
        n_ring = min(
            -(-aligned_len // bs),  # cdiv
            self._draft_blocks_per_slot,
        )
        for j in range(n_ring):
            actual_pos = aligned_start + j * bs
            ring_block = (actual_pos % ring_slots) // bs
            self._draft_block_table[0, j] = draft_base + ring_block

        # Seq lens: window-aligned, not absolute
        self._draft_seq_lens[0] = aligned_len

        # Slot mapping: ring-wrapped positions
        for j in range(num_tokens):
            pos = kv_len + j
            ring_block = (pos % ring_slots) // bs
            ring_off = pos % bs
            self._draft_slot_mapping[j] = (draft_base + ring_block) * bs + ring_off

        # Query start loc
        self._draft_qsl[1] = num_tokens
        self._draft_qsl_cpu[1] = num_tokens

        return CommonAttentionMetadata(
            query_start_loc=self._draft_qsl[:2],
            query_start_loc_cpu=self._draft_qsl_cpu[:2],
            seq_lens=self._draft_seq_lens[:1],
            num_reqs=1,
            num_actual_tokens=num_tokens,
            max_query_len=num_tokens,
            max_seq_len=aligned_len,
            block_table_tensor=self._draft_block_table[:1, :n_ring],
            slot_mapping=self._draft_slot_mapping[:num_tokens],
            causal=True,
        )

    def _draft_forward(
        self,
        slot: int,
        bonus_token: int,
        kv_len: int,
    ) -> list[int]:
        """Run draft model forward with 16 tokens (1 bonus + 15 mask).

        Returns 15 draft tokens (greedy argmax).
        """
        num_tokens = NUM_QUERY_PER_REQ  # 16

        # Fill input: [bonus_token, mask, mask, ..., mask]
        self._draft_input_ids[0] = bonus_token
        self._draft_input_ids[1:num_tokens] = MASK_TOKEN_ID

        # Positions: [kv_len, kv_len+1, ..., kv_len+15]
        self._draft_positions[:num_tokens] = torch.arange(
            kv_len, kv_len + num_tokens, dtype=torch.long, device=self.device
        )

        # Build draft attention metadata
        common_meta = self._build_draft_attn_metadata(slot, kv_len, num_tokens)

        # Build sparkinfer metadata for draft (extend mode, qo=16)
        from runtime.backends.laguna_sparkinfer_attn import SparkinferAttnMetadata
        draft_meta = SparkinferAttnMetadata(
            mode="extend",
            page_table=common_meta.block_table_tensor,
            cache_seqlens=common_meta.seq_lens,
            cu_seqlens_q=common_meta.query_start_loc,
            num_actual_tokens=num_tokens,
            window_left=DRAFT_WINDOW - 1,
        )

        # Create metadata dict for all draft layers
        attn_metadata_dict = {
            name: draft_meta for name in self._draft_layer_names
        }
        slot_mapping_dict = {
            name: self._draft_slot_mapping[:num_tokens]
            for name in self._draft_layer_names
        }

        # Run draft model forward
        with set_forward_context(
            attn_metadata_dict, self.vllm_config, slot_mapping=slot_mapping_dict
        , skip_compiled=True):
            draft_hidden = self.draft_model(
                input_ids=self._draft_input_ids[:num_tokens],
                positions=self._draft_positions[:num_tokens],
                inputs_embeds=None,
            )

        # Compute draft logits and sample greedily. Position 0's logits are
        # never used (only positions 1..15, the mask positions, predict the
        # next tokens) -- slice before compute_logits so the vocab-size GEMM
        # only pays for the 15 positions that matter.
        draft_logits = self.draft_model.compute_logits(draft_hidden[1:num_tokens])
        draft_tokens = draft_logits.argmax(dim=-1)
        return draft_tokens.tolist()

    def _precompute_context_kv(
        self,
        slot: int,
        combined_hidden: torch.Tensor,
        position: int,
    ) -> None:
        """Precompute and store context KV for the draft model."""
        bs = self.block_size
        phys = _physical_slot(slot)
        draft_base = phys * self._draft_blocks_per_slot
        ring_slots = self._draft_blocks_per_slot * bs

        # Slot mapping: ring-wrapped position
        ring_block = (position % ring_slots) // bs
        ring_off = position % bs
        slot_mapping_val = (draft_base + ring_block) * bs + ring_off
        context_positions = torch.tensor(
            [position], dtype=torch.long, device=self.device
        )
        context_slot_mapping = torch.tensor(
            [slot_mapping_val], dtype=torch.long, device=self.device
        )

        self.draft_model.precompute_and_store_context_kv(
            combined_hidden,
            context_positions,
            context_slot_mapping,
        )

    def _accept_reject(
        self,
        verify_logits: torch.Tensor,
        draft_tokens: list[int],
        bonus_token: int,
    ) -> tuple[list[int], int]:
        """Greedy accept/reject from verify logits (CUDA Graph path)."""
        num_tokens = 1 + len(draft_tokens)
        verify_argmax = verify_logits[:num_tokens - 1].argmax(dim=-1).tolist()
        return _greedy_accept_reject(verify_argmax, draft_tokens, bonus_token)

    def _verify(
        self,
        slot: int,
        bonus_token: int,
        draft_tokens: list[int],
        kv_len: int,
    ) -> tuple[list[int], int]:
        """Verify draft tokens with main model (parallel, single forward).

        Runs main model forward with [bonus_token] + draft_tokens (16 tokens)
        in a single pass. Uses decode-style ring metadata extended for qo>1.

        Returns (accepted_tokens, num_accepted).
        """
        backend = self.backend
        num_tokens = 1 + len(draft_tokens)  # 16
        verify_tokens = [bonus_token] + draft_tokens

        # Build attention metadata for parallel verify
        # Use decode-style buffers but with qo_len=num_tokens
        logits, _ = self._forward_verify(slot, verify_tokens, kv_len, num_tokens)

        # Greedy verification:
        # logits[i] predicts token at position kv_len + i + 1
        # logits[0] → should match draft_tokens[0]
        verify_argmax = logits[:num_tokens - 1].argmax(dim=-1).tolist()
        return _greedy_accept_reject(verify_argmax, draft_tokens, bonus_token)

    def _forward_verify(
        self,
        slot: int,
        tokens: list[int],
        kv_len: int,
        num_tokens: int,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass for verify: qo_len>1 with correct ring metadata.

        Builds attention metadata that correctly maps to the ring buffer
        for SWA layers and contiguous blocks for full layers.
        """
        backend = self.backend
        bs = backend.block_size
        device = self.device

        # Input tensors
        input_ids = torch.tensor(tokens, dtype=torch.long, device=device)
        positions = torch.arange(kv_len, kv_len + num_tokens, dtype=torch.long, device=device)

        # Build full-attention metadata (standard contiguous blocks)
        from runtime.compat_vllm import get_common_attn_metadata_cls, set_current_vllm_config
        CommonAttentionMetadata = get_common_attn_metadata_cls()

        phys = _physical_slot(slot)
        new_kv_len = kv_len + num_tokens
        n_blocks_full = (new_kv_len + bs - 1) // bs
        full_base = phys * backend.blocks_per_slot

        # Full block table
        full_bt = torch.zeros(1, n_blocks_full, dtype=torch.int32, device=device)
        full_bt[0, :n_blocks_full] = torch.arange(
            full_base, full_base + n_blocks_full, dtype=torch.int32, device=device
        )

        # Full slot mapping
        full_sm = torch.zeros(num_tokens, dtype=torch.long, device=device)
        for j in range(num_tokens):
            pos = kv_len + j
            full_sm[j] = (full_base + pos // bs) * bs + pos % bs

        qsl = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
        qsl_cpu = torch.tensor([0, num_tokens], dtype=torch.int32)
        seq_lens = torch.tensor([new_kv_len], dtype=torch.int32, device=device)

        full_meta = CommonAttentionMetadata(
            query_start_loc=qsl,
            query_start_loc_cpu=qsl_cpu,
            seq_lens=seq_lens,
            num_reqs=1,
            num_actual_tokens=num_tokens,
            max_query_len=num_tokens,
            max_seq_len=new_kv_len,
            block_table_tensor=full_bt,
            slot_mapping=full_sm,
            causal=True,
        )

        # Build SWA ring metadata for verify (qo>1)
        if backend._ring_blocks_per_slot > 0:
            ring_base = phys * backend._ring_blocks_per_slot
            ring_slots = backend._ring_slots_per_slot
            window = backend._swa_window

            # Window start for the earliest query position
            window_start = max(0, kv_len - window + 1)
            aligned_start = (window_start // bs) * bs
            aligned_len = new_kv_len - aligned_start
            n_ring = min((aligned_len + bs - 1) // bs, backend._ring_blocks_per_slot)

            ring_bt = torch.zeros(1, n_ring, dtype=torch.int32, device=device)
            for j in range(n_ring):
                actual_pos = aligned_start + j * bs
                ring_block = (actual_pos % ring_slots) // bs
                ring_bt[0, j] = ring_base + ring_block

            ring_sm = torch.zeros(num_tokens, dtype=torch.long, device=device)
            for j in range(num_tokens):
                pos = kv_len + j
                ring_block = (pos % ring_slots) // bs
                ring_off = pos % bs
                ring_sm[j] = (ring_base + ring_block) * bs + ring_off

            ring_seq_lens = torch.tensor([aligned_len], dtype=torch.int32, device=device)

            swa_meta = CommonAttentionMetadata(
                query_start_loc=qsl,
                query_start_loc_cpu=qsl_cpu,
                seq_lens=ring_seq_lens,
                num_reqs=1,
                num_actual_tokens=num_tokens,
                max_query_len=num_tokens,
                max_seq_len=aligned_len,
                block_table_tensor=ring_bt,
                slot_mapping=ring_sm,
                causal=True,
            )
        else:
            swa_meta = None

        # Build FlashInfer metadata for each group
        attn_metadata_dict = {}
        slot_mapping_dict = {}
        for group_key, builder in backend._metadata_builders.items():
            wl = group_key[0]
            is_swa = wl >= 0
            meta = swa_meta if (is_swa and swa_meta is not None) else full_meta
            with set_current_vllm_config(backend.vllm_config):
                metadata = builder.build(common_prefix_len=0, common_attn_metadata=meta)
            for name in backend._layer_groups[group_key]:
                attn_metadata_dict[name] = metadata
                slot_mapping_dict[name] = meta.slot_mapping

        with bf_attn_context(attn_metadata_dict, slot_mapping_dict):
            with set_forward_context(
                attn_metadata_dict, backend.vllm_config, slot_mapping=slot_mapping_dict,
                skip_compiled=True,
            ):
                result = backend.model.forward(input_ids, positions)

        if isinstance(result, tuple):
            hidden_states = result[0]
        else:
            hidden_states = result

        logits = backend.model.compute_logits(hidden_states)
        return logits, None

    def speculative_decode_step(
        self,
        slot: int,
        last_token: int,
    ) -> list[int]:
        """Execute one full speculative decode step.

        Returns list of accepted tokens (1-16 tokens).
        """
        backend = self.backend
        kv_len = backend.slot_kv_len[slot]

        # Step 1: Main model decode with aux hidden states
        if self._cuda_graph is not None:
            next_tokens, aux_hidden_states = self._cuda_graph.replay_with_aux(
                [slot], [last_token], [kv_len]
            )
            bonus_token = next_tokens[0]
        else:
            logits, aux_hidden_states = self._forward_main_with_aux(
                [slot], [last_token], [kv_len], qo_len=1
            )
            bonus_token = int(logits[0].argmax(dim=-1).item())
        backend.slot_kv_len[slot] += 1

        # Step 2: Combine hidden states and precompute context KV
        if aux_hidden_states is not None:
            combined_input = torch.cat(aux_hidden_states, dim=-1)  # [1, 18432]
            combined = self.draft_model.combine_hidden_states(
                combined_input
            )  # [1, 3072]
            self._precompute_context_kv(slot, combined, kv_len)

        # Step 3: Draft forward → 15 draft tokens
        if self._draft_cg is not None:
            draft_tokens = self._draft_cg.replay(slot, bonus_token, kv_len + 1)
        else:
            draft_tokens = self._draft_forward(slot, bonus_token, kv_len + 1)

        # Step 4: Verify
        if self._verify_cg is not None:
            verify_tokens = [bonus_token] + draft_tokens
            verify_logits = self._verify_cg.replay(slot, verify_tokens, kv_len + 1)
            accepted, num_accepted = self._accept_reject(
                verify_logits, draft_tokens, bonus_token
            )
        else:
            accepted, num_accepted = self._verify(
                slot, bonus_token, draft_tokens, kv_len + 1
            )

        # Update slot state
        backend.slot_kv_len[slot] += num_accepted
        for tok in accepted:
            backend.slot_committed_tokens[slot].append(tok)

        return accepted

    def _bulk_precompute_context_kv(
        self,
        slot: int,
        aux_hidden_states: list[torch.Tensor],
        num_positions: int,
        position_offset: int = 0,
    ) -> None:
        """Precompute draft context KV from captured aux hidden states.

        Args:
            slot: slot index
            aux_hidden_states: list of 6 tensors [N, 3072]
            num_positions: number of positions to precompute
            position_offset: absolute position offset (for chunked prefill)
        """
        # Combine hidden states: [N, 18432] → [N, 3072]
        combined_input = torch.cat(aux_hidden_states, dim=-1)
        combined = self.draft_model.combine_hidden_states(combined_input)

        # Precompute context KV (ring-wrapped positions)
        bs = self.block_size
        phys = _physical_slot(slot)
        draft_base = phys * self._draft_blocks_per_slot
        ring_slots = self._draft_blocks_per_slot * bs

        context_positions = torch.arange(
            position_offset, position_offset + num_positions,
            dtype=torch.long, device=self.device
        )
        slot_mappings = torch.zeros(num_positions, dtype=torch.long, device=self.device)
        for i in range(num_positions):
            pos = position_offset + i
            ring_block = (pos % ring_slots) // bs
            ring_off = pos % bs
            slot_mappings[i] = (draft_base + ring_block) * bs + ring_off

        self.draft_model.precompute_and_store_context_kv(
            combined,
            context_positions,
            slot_mappings,
        )
        logger.info(
            "DFlash: precomputed context KV for %d positions (offset=%d)",
            num_positions, position_offset,
        )

    def _lazy_capture_cg(self) -> None:
        """Capture draft CUDA Graph after KV cache is populated.

        Verify CG disabled: after SWA window fix (verify_ring used nkv-window
        instead of kv_len-window+1, missing 15 tokens), eager verify achieves
        55% acceptance / 60.6 tok/s at 64K — better than CG verify (34% / 38).
        Eager is also 43-63ms faster per forward at 128K (191ms vs 249ms).
        Draft CG kept: 6.5ms replay vs ~15ms eager (6-layer SWA model).
        """
        self._verify_cg = None
        try:
            from runtime.backends.laguna_dflash_cudagraph import DFlashDraftCudaGraph
            dcg = DFlashDraftCudaGraph(self)
            dcg.capture()
            self._draft_cg = dcg
            logger.info("DFlash: draft CUDA Graph captured (lazy)")
        except Exception as e:
            logger.warning("DFlash: draft CG failed: %s", e)
        self._cg_captured = True

    def generate(
        self,
        prompt_ids: list[int],
        max_tokens: int = 128,
        temperature: float = 0.0,
        eos_tokens: tuple[int, ...] = (2, 24),
        slot: int = 0,
        enable_prefix_cache: bool = True,
    ) -> tuple[list[int], dict[str, float]]:
        """Generate tokens using DFlash speculative decoding (verify-only).

        Uses verify-only design: no redundant M=1 decode forward.
        Returns (tokens, stats).
        """
        return self.generate_verify_only(
            prompt_ids, max_tokens, temperature, eos_tokens,
            slot=slot, enable_prefix_cache=enable_prefix_cache,
        )

    def generate_legacy(
        self,
        prompt_ids: list[int],
        max_tokens: int = 128,
        temperature: float = 0.0,
        eos_tokens: tuple[int, ...] = (2, 24),
    ) -> tuple[list[int], dict[str, float]]:
        """Legacy generate with separate decode+verify (kept for comparison)."""
        backend = self.backend
        slot = 0
        backend.reset_slot(slot)
        # Reset draft KV cache to avoid stale data from previous generates
        for kv_tensor in self._draft_kv_caches.values():
            kv_tensor.zero_()
        torch.cuda.empty_cache()

        t0 = time.perf_counter()

        # Prefill with aux hidden state capture (single forward, no re-run)
        prompt_len = len(prompt_ids)
        first_token, aux_hidden_states = backend.prefill_with_aux(slot, prompt_ids)

        # Bulk precompute draft context KV from captured aux states
        # For long prompts (chunked prefill), aux is only from the last chunk.
        # Precompute draft KV for those positions (offset from prompt start).
        if aux_hidden_states is not None:
            aux_len = aux_hidden_states[0].shape[0]
            aux_offset = prompt_len - aux_len
            self._bulk_precompute_context_kv(slot, aux_hidden_states, aux_len, aux_offset)

        # Free fragmented memory from prefill before decode phase
        del aux_hidden_states
        torch.cuda.empty_cache()
        t_prefill = time.perf_counter()

        tokens = [first_token]
        total_draft = 0
        total_accepted = 0
        num_steps = 0

        while len(tokens) < max_tokens:
            last_token = tokens[-1]
            accepted = self.speculative_decode_step(slot, last_token)
            tokens.extend(accepted)
            num_steps += 1
            total_draft += NUM_SPECULATIVE_TOKENS
            total_accepted += len(accepted) - 1  # -1 for bonus

            # Check EOS
            found_eos = False
            for tok in accepted:
                if tok in eos_tokens:
                    idx = len(tokens) - len(accepted) + accepted.index(tok)
                    tokens = tokens[: idx + 1]
                    found_eos = True
                    break
            if found_eos:
                break

        t_total = time.perf_counter()

        # Lazy-capture verify/draft CGs after first generate completes
        # (capture warmup writes to KV cache; must not corrupt active session)
        if self._use_cuda_graph and not self._cg_captured:
            self._lazy_capture_cg()

        tokens = tokens[:max_tokens]

        stats = {
            "prefill_ms": (t_prefill - t0) * 1000,
            "decode_ms": (t_total - t_prefill) * 1000,
            "total_ms": (t_total - t0) * 1000,
            "num_tokens": len(tokens),
            "num_steps": num_steps,
            "acceptance_rate": total_accepted / max(total_draft, 1),
            "tokens_per_step": (len(tokens) - 1) / max(num_steps, 1),
            "tok_per_s": (len(tokens) - 1) / max(t_total - t_prefill, 1e-6),
        }

        return tokens, stats

    def generate_verify_only(
        self,
        prompt_ids: list[int],
        max_tokens: int = 128,
        temperature: float = 0.0,
        eos_tokens: tuple[int, ...] = (2, 24),
        slot: int = 0,
        enable_prefix_cache: bool = True,
    ) -> tuple[list[int], dict[str, float]]:
        """Generate using verify-only speculative decoding (no redundant decode).

        Eliminates the M=1 decode forward by extracting bonus_token and aux
        from the verify (M=16) forward. ~50% step latency reduction.

        Prefix cache: if enable_prefix_cache and the slot has a matching
        cached prefix, skip re-prefilling the prefix and only process the
        suffix. This is the core optimization for agent workloads where
        each turn re-sends the full conversation history.

        Returns (tokens, stats).
        """
        backend = self.backend
        prompt_len = len(prompt_ids)

        # Prefix cache: find matching prefix in cached KV
        prefix_len = 0
        if enable_prefix_cache:
            prefix_len = backend.find_prefix_match(slot, prompt_ids)

        if prefix_len > 0 and prefix_len < prompt_len:
            # Partial match: continue from cached prefix
            logger.info(
                "Prefix cache HIT: %d/%d tokens cached, prefilling %d suffix",
                prefix_len, prompt_len, prompt_len - prefix_len,
            )
        elif prefix_len >= prompt_len:
            # Full match: no prefill needed
            logger.info(
                "Prefix cache FULL HIT: %d/%d tokens cached",
                prefix_len, prompt_len,
            )
        else:
            # No match: full reset and prefill
            backend.reset_slot(slot)
            for kv_tensor in self._draft_kv_caches.values():
                kv_tensor.zero_()

        t0 = time.perf_counter()

        # Always zero draft KV — it will be recomputed from aux hidden states
        for kv_tensor in self._draft_kv_caches.values():
            kv_tensor.zero_()

        if prefix_len > 0:
            # Continue from cached prefix
            first_token, aux_hidden_states = backend.continue_prefill_with_aux(
                slot, prompt_ids, prefix_len
            )
        else:
            # Full prefill
            first_token, aux_hidden_states = backend.prefill_with_aux(slot, prompt_ids)

        # Bulk precompute draft context KV from prefill aux
        if aux_hidden_states is not None:
            aux_len = aux_hidden_states[0].shape[0]
            if prefix_len > 0:
                aux_offset = prompt_len - aux_len
            else:
                aux_offset = prompt_len - aux_len
            self._bulk_precompute_context_kv(slot, aux_hidden_states, aux_len, aux_offset)

        del aux_hidden_states
        t_prefill = time.perf_counter()

        # Bootstrap: initial bonus = first_token, run draft to get initial 15 tokens
        bonus_token = first_token
        kv_len = backend.slot_kv_len[slot]  # = prompt_len

        if self._draft_cg is not None:
            draft_tokens = self._draft_cg.replay(slot, bonus_token, kv_len)
        else:
            draft_tokens = self._draft_forward(slot, bonus_token, kv_len)

        tokens = [first_token]
        total_draft = 0
        total_accepted = 0
        num_steps = 0

        while len(tokens) < max_tokens:
            num_steps += 1
            total_draft += NUM_SPECULATIVE_TOKENS
            kv_len = backend.slot_kv_len[slot]

            # Step 1: Verify (M=16 full model forward) — replaces decode+verify
            verify_tokens = [bonus_token] + draft_tokens
            if self._verify_cg is not None:
                verify_logits, verify_aux = self._verify_cg.replay_with_aux(
                    slot, verify_tokens, kv_len
                )
            else:
                verify_logits, verify_aux = self._forward_verify_with_aux(
                    slot, verify_tokens, kv_len, len(verify_tokens)
                )

            # Step 2: Accept/reject (single GPU→CPU sync for all 16 argmax)
            all_argmax = verify_logits[:NUM_QUERY_PER_REQ].argmax(dim=-1).tolist()
            decision = _verify_only_accept_reject(
                all_argmax, draft_tokens, bonus_token
            )
            num_accepted = decision["num_accepted"]
            new_tokens = decision["committed"]
            total_accepted += num_accepted

            # The recovery token on rejection, or target bonus after a full
            # accept, is the pending anchor for the next draft round.
            new_bonus = decision["next_anchor"]

            # The verifier wrote the old anchor plus every matching draft token.
            # The recovery/bonus token remains pending and is not in KV yet.
            context_count = decision["context_count"]

            # Step 4: Precompute draft context KV from committed verifier inputs.
            if verify_aux is not None:
                aux_slice = [a[:context_count] for a in verify_aux]
                combined_input = torch.cat(aux_slice, dim=-1)
                combined = self.draft_model.combine_hidden_states(combined_input)
                bs = self.block_size
                phys = _physical_slot(slot)
                draft_base = phys * self._draft_blocks_per_slot
                ring_slots = self._draft_blocks_per_slot * bs
                context_positions = torch.arange(
                    kv_len, kv_len + context_count,
                    dtype=torch.long, device=self.device
                )
                ring_blocks = (context_positions % ring_slots) // bs
                ring_offs = context_positions % bs
                slot_mappings = (draft_base + ring_blocks) * bs + ring_offs
                self.draft_model.precompute_and_store_context_kv(
                    combined, context_positions, slot_mappings
                )

            # Step 5: Update state
            backend.slot_kv_len[slot] += context_count
            for tok in new_tokens:
                backend.slot_committed_tokens[slot].append(tok)
            tokens.extend(new_tokens)

            # Check EOS
            found_eos = False
            for tok in new_tokens:
                if tok in eos_tokens:
                    idx = len(tokens) - len(new_tokens) + new_tokens.index(tok)
                    tokens = tokens[:idx + 1]
                    found_eos = True
                    break
            if found_eos:
                break

            # Step 6: Draft next 15 tokens
            bonus_token = new_bonus
            new_kv_len = backend.slot_kv_len[slot]
            if self._draft_cg is not None:
                draft_tokens = self._draft_cg.replay(slot, bonus_token, new_kv_len)
            else:
                draft_tokens = self._draft_forward(slot, bonus_token, new_kv_len)

        t_total = time.perf_counter()
        # Prefix cache: preserve slot KV for next turn

        if self._use_cuda_graph and not self._cg_captured:
            self._lazy_capture_cg()

        tokens = tokens[:max_tokens]
        stats = {
            "prefill_ms": (t_prefill - t0) * 1000,
            "decode_ms": (t_total - t_prefill) * 1000,
            "total_ms": (t_total - t0) * 1000,
            "num_tokens": len(tokens),
            "num_steps": num_steps,
            "acceptance_rate": total_accepted / max(total_draft, 1),
            "tokens_per_step": (len(tokens) - 1) / max(num_steps, 1),
            "tok_per_s": (len(tokens) - 1) / max(t_total - t_prefill, 1e-6),
        }
        return tokens, stats

    def _forward_verify_with_aux(
        self,
        slot: int,
        tokens: list[int],
        kv_len: int,
        num_tokens: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Eager verify forward that returns (logits, aux_hidden_states)."""
        backend = self.backend
        input_ids = torch.tensor(tokens, dtype=torch.long, device=self.device)
        positions = torch.arange(kv_len, kv_len + num_tokens, dtype=torch.long, device=self.device)

        # Build metadata (reuse existing _forward_verify logic)
        from runtime.compat_vllm import get_common_attn_metadata_cls, set_current_vllm_config
        CommonAttentionMetadata = get_common_attn_metadata_cls()

        phys = _physical_slot(slot)
        new_kv_len = kv_len + num_tokens
        bs = backend.block_size
        n_blocks_full = (new_kv_len + bs - 1) // bs
        full_base = phys * backend.blocks_per_slot

        full_bt = torch.zeros(1, n_blocks_full, dtype=torch.int32, device=self.device)
        full_bt[0, :n_blocks_full] = torch.arange(
            full_base, full_base + n_blocks_full, dtype=torch.int32, device=self.device
        )
        full_sm = torch.zeros(num_tokens, dtype=torch.long, device=self.device)
        for j in range(num_tokens):
            pos = kv_len + j
            full_sm[j] = (full_base + pos // bs) * bs + pos % bs

        qsl = torch.tensor([0, num_tokens], dtype=torch.int32, device=self.device)
        qsl_cpu = torch.tensor([0, num_tokens], dtype=torch.int32)
        seq_lens = torch.tensor([new_kv_len], dtype=torch.int32, device=self.device)

        full_meta = CommonAttentionMetadata(
            query_start_loc=qsl, query_start_loc_cpu=qsl_cpu,
            seq_lens=seq_lens, num_reqs=1, num_actual_tokens=num_tokens,
            max_query_len=num_tokens, max_seq_len=new_kv_len,
            block_table_tensor=full_bt, slot_mapping=full_sm, causal=True,
        )

        # SWA ring metadata
        swa_meta = None
        if backend._ring_blocks_per_slot > 0:
            swa_meta = backend._build_swa_attn_metadata(
                [slot], [kv_len], [num_tokens], False, swa_mode="verify_ring"
            )

        from runtime.backends.laguna_sparkinfer_attn import SparkinferAttnMetadata
        attn_metadata_dict = {}
        slot_mapping_dict = {}
        full_spark_meta = backend._build_sparkinfer_metadata(full_meta, window_left=-1)
        swa_spark_meta = None
        if swa_meta is not None:
            swa_wl = backend._swa_window - 1 if backend._swa_window > 0 else -1
            swa_spark_meta = backend._build_sparkinfer_metadata(swa_meta, window_left=swa_wl)
        for group_key, layer_names in backend._layer_groups.items():
            wl = group_key[0]
            is_swa = wl >= 0
            meta = swa_spark_meta if (is_swa and swa_spark_meta is not None) else full_spark_meta
            for name in layer_names:
                attn_metadata_dict[name] = meta
                slot_mapping_dict[name] = full_sm if not is_swa else swa_meta.slot_mapping

        from runtime.compat_vllm import set_forward_context
        with set_current_vllm_config(backend.vllm_config):
            with bf_attn_context(attn_metadata_dict, slot_mapping_dict):
                with set_forward_context(
                    attn_metadata_dict, backend.vllm_config,
                    slot_mapping=slot_mapping_dict, skip_compiled=True,
                ):
                    result = backend.model.forward(input_ids, positions)

        if isinstance(result, tuple):
            hidden_states, aux = result
        else:
            hidden_states, aux = result, None
        logits = backend.model.compute_logits(hidden_states)
        return logits, aux
