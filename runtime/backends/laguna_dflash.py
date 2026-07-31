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
from collections.abc import Callable
from typing import Any

import torch

from bfdiag.invariants import checks as bfdiag_checks
from bfdiag.trace import events as bfdiag_events
from bfdiag.trace import ring as bfdiag_trace
from runtime.backends.bf_attention import bf_attn_context
from runtime.backends.dflash_constants import (
    AUX_LAYER_IDS,
    DFLASH_MODEL_PATH,
    DRAFT_NUM_LAYERS,
    DRAFT_WINDOW,
    MASK_TOKEN_ID,
    NUM_QUERY_PER_REQ,
    NUM_SPECULATIVE_TOKENS,
)
from runtime.backends.laguna import (
    RESERVED_PHYSICAL_SLOTS,
    LagunaBackend,
    _physical_slot,
    _ring_blocks_for_window,
)
from runtime.laguna_runtime import LagunaAttentionMetadata, bind_laguna_kv_cache
from runtime.mtp_accept import determine_accept_reject_from_predictions

logger = logging.getLogger("qwen_sm120_runtime.dflash")


def _ring_prefix_reuse_is_safe(
    cached_kv_len: int,
    prefix_len: int,
    ring_specs: tuple[tuple[int, int], ...],
) -> bool:
    """Whether every ring still retains the prefix boundary's attention window.

    A ring with ``capacity - window`` spare positions can survive that many
    appended KV writes without overwriting the old prefix window.  Rewinding
    farther cannot be repaired by recomputing one window: that recomputation
    itself needs an even older window which the ring no longer contains.
    """
    rewind = cached_kv_len - prefix_len
    if prefix_len <= 0 or rewind < 0:
        return False
    return all(rewind <= max(0, capacity - window) for capacity, window in ring_specs)


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
        *,
        defer_cuda_graph_capture: bool = False,
    ) -> None:
        self.backend = backend
        self.device = backend.device
        self.runtime_config = backend.runtime_config
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
        self._cuda_graph_capture_deferred = bool(defer_cuda_graph_capture)
        if self._use_cuda_graph and not self._cuda_graph_capture_deferred:
            self._init_cuda_graph()

        logger.info(
            "DFlashEngine initialized: K=%d speculative tokens, draft %d layers, cuda_graph=%s",
            NUM_SPECULATIVE_TOKENS,
            DRAFT_NUM_LAYERS,
            self._use_cuda_graph,
        )

    def capture_cuda_graphs(self) -> None:
        """Capture deferred DFlash graphs after the eager engine is fully built.

        Normal production construction captures immediately. Diagnostics may
        defer only this stage to attribute its persistent CUDA allocations;
        the captured graph ABI and runtime behavior remain unchanged.
        """
        if self._use_cuda_graph and not self._cg_captured:
            self._init_cuda_graph()
        self._cuda_graph_capture_deferred = False

    def _load_draft_model(self, model_path: str | None) -> Any:
        """Load the DFlash draft model.

        Both the DFlash model graph and its configuration are owned by this
        runtime.  Only the narrow fields consumed by the local loader are
        constructed; vLLM's general speculative-config resolution is not.
        """
        from runtime.laguna_config import (
            build_laguna_dflash_config,
            load_laguna_draft_hf_config,
        )

        if model_path is None:
            model_path = os.path.expanduser(DFLASH_MODEL_PATH)

        draft_hf_config = load_laguna_draft_hf_config(
            model_path,
        )
        draft_runtime_config = build_laguna_dflash_config(
            self.runtime_config,
            model=model_path,
            hf_config=draft_hf_config,
            num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
            max_model_len=DRAFT_WINDOW + NUM_QUERY_PER_REQ + 128,
        )

        from runtime.model_loading import load_laguna_dflash_draft_model

        draft_model = load_laguna_dflash_draft_model(
            target_model=self.backend.model,
            draft_runtime_config=draft_runtime_config,
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
        # Discover draft model's attention layers from static_forward_context
        sfc = self.runtime_config.compilation_config.static_forward_context

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
                self.draft_model.model if hasattr(self.draft_model, "model") else self.draft_model
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
        num_phys = self.num_slots + RESERVED_PHYSICAL_SLOTS
        draft_blocks_per_slot = _ring_blocks_for_window(
            DRAFT_WINDOW, self.block_size, NUM_QUERY_PER_REQ
        )
        self._draft_blocks_per_slot = draft_blocks_per_slot
        total_blocks = num_phys * draft_blocks_per_slot

        self._draft_kv_caches: dict[str, torch.Tensor] = {}
        for name in self._draft_layer_names:
            attn = self._draft_attn_layers[name]
            # Self-allocated: [2, blocks, bs, kv_heads, head_dim], FP8 as uint8
            shape = (2, total_blocks, self.block_size, attn.num_kv_heads, attn.head_size)
            self._draft_kv_caches[name] = torch.zeros(shape, dtype=torch.uint8, device=self.device)

        # Bind draft KV caches to draft attention layers
        bind_laguna_kv_cache(self._draft_kv_caches, self._draft_attn_layers, [])
        logger.info(
            "DFlash: draft KV allocated: %d blocks/slot × %d layers",
            draft_blocks_per_slot,
            len(self._draft_layer_names),
        )

    def _patch_draft_sparkinfer(self) -> None:
        """Patch draft model attention layers to use sparkinfer (zero FlashInfer dep).

        阶段7-补充: also replaces each draft attention layer's whole module
        with ``BFAttention`` (``replace_laguna_attention()``, same call the
        main model already goes through in ``LagunaBackend.__init__``) --
        previously this only swapped ``.impl``, leaving ``self.attn``
        itself as a real vLLM ``Attention`` instance (now
        ``SelfBuiltAttentionPlaceholder``). That was fine as long as
        vLLM's own ``Attention.forward()`` handled the
        get_forward_context()/custom-op bridge into ``.impl.forward()``,
        but ``SelfBuiltAttentionPlaceholder`` doesn't implement that (see
        its module docstring) -- it was only ever built to satisfy the
        construction-time attribute contract, not to be called as a real
        forward() target. Routing through BFAttention instead sidesteps
        needing to replicate that bridge at all: BFAttention.forward()
        already reads ``bf_attn_context`` (a lightweight context this
        file already sets up around every draft-model forward call, see
        e.g. the ``with bf_attn_context(...):`` blocks below -- it was
        unused for the draft model until now, only ever consumed by the
        main model's BFAttention layers). ``.impl`` must be set to the
        real ``SparkinferAttentionImpl`` (with the draft-specific
        ``window_left=DRAFT_WINDOW - 1``, different from the main
        model's per-layer-group window) BEFORE calling
        ``replace_laguna_attention()``, since it reads ``attn_layer.impl.
        scale``/``.window_left`` to construct each BFAttention -- same
        order LagunaBackend.__init__ already uses for the main model.

        ``replace_laguna_attention()``'s default parent-resolution logic
        (split ``layer_name`` on ``"."``, walk the model tree by those
        exact path components) does NOT work here and needs its
        ``resolve_parent`` override: draft attention layers are
        registered under a global-index ``layer_name`` (e.g.
        ``"...layers.48.attn"``, laguna_dflash_model.py's module
        docstring explains why -- offset past the main model's 48 layers
        so both share one static_forward_context without key collisions)
        that doesn't match this draft model's own local module tree
        (``self.draft_model.model.layers`` is only ever indices 0-5).
        Confirmed by an actual GPU run, not spotted by inspection -- the
        default logic raised ``IndexError: index 48 is out of range``
        trying to index ``layers[48]``. Resolved the same way
        ``_alloc_draft_kv_cache``'s fallback discovery branch above
        already does it: walk ``draft_model.model.layers`` directly by
        real attribute access, matching each layer's ``self_attn.attn``
        against ``self._draft_attn_layers`` by ``layer_name``.
        """
        from runtime.backends.bf_attention import replace_laguna_attention
        from runtime.backends.laguna_sparkinfer_attn import SparkinferAttentionImpl

        for name in self._draft_layer_names:
            attn = self._draft_attn_layers[name]
            attn.impl = SparkinferAttentionImpl(
                num_heads=attn.num_heads,
                head_size=attn.head_size,
                scale=attn.head_size**-0.5,
                num_kv_heads=attn.num_kv_heads,
                window_left=DRAFT_WINDOW - 1,
            )

        draft_inner = (
            self.draft_model.model if hasattr(self.draft_model, "model") else self.draft_model
        )
        parents_by_name: dict[str, Any] = {}
        for layer in draft_inner.layers:
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is not None and hasattr(self_attn, "attn"):
                parents_by_name[self_attn.attn.layer_name] = self_attn

        def _resolve_parent(layer_name: str) -> tuple[Any, str]:
            return parents_by_name[layer_name], "attn"

        replace_laguna_attention(
            self.draft_model,
            self._draft_attn_layers,
            self._draft_kv_caches,
            resolve_parent=_resolve_parent,
        )
        logger.info(
            "DFlash: draft attention patched with sparkinfer (%d layers)",
            len(self._draft_layer_names),
        )

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
        """Capture the runtime-safe DFlash CUDA Graph.

        The draft graph has a fixed 16-token query shape and is safe across
        ring-relative KV lengths after address-aware rebinding.  Main-model
        verify CG uses sparkinfer's update_prefill_graph_replay_metadata to
        recompute the worklist (block_valid_mask, kv_chunk_size,
        window_start_tokens) on each replay, fixing the stale-worklist bug
        that previously caused acceptance collapse at SWA page boundaries.
        """
        self._verify_cg = None
        self._draft_cg = None
        self._cg_captured = False

        if self._use_cuda_graph:
            self._capture_draft_cg()
            if os.environ.get("QSR_VERIFY_CUDA_GRAPH", "1") == "1":
                self._capture_verify_cg()
            self._cg_captured = True

    def _capture_verify_cg(self) -> None:
        """Capture main-model verify graph (M=16 extend planner with runtime worklist update)."""
        from runtime.backends.laguna_cuda_graph import LagunaCudaGraphVerify

        try:
            cg = LagunaCudaGraphVerify(self.backend, num_tokens=NUM_QUERY_PER_REQ)
            cg.capture()
            self._verify_cg = cg
            logger.info("DFlash: verify CG captured (M=%d)", NUM_QUERY_PER_REQ)
        except Exception as e:
            logger.warning("DFlash: verify CG capture failed: %s", e)
            import traceback

            traceback.print_exc()
            self._verify_cg = None

    def _capture_draft_cg(self) -> None:
        """Capture the M=16 draft graph before any request owns slot state."""
        try:
            from runtime.backends.laguna_dflash_cudagraph import DFlashDraftCudaGraph

            cg = DFlashDraftCudaGraph(self)
            cg.capture()
            self._draft_cg = cg
            logger.info("DFlash: draft CUDA Graph captured")
        except Exception as e:
            logger.warning("DFlash: draft CG failed: %s", e)
            self._draft_cg = None

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
        common_meta = backend._build_common_attn_metadata(slot_ids, kv_lengths, qo_lens, is_decode)

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
            meta = (
                swa_spark_meta if (is_swa_group and swa_spark_meta is not None) else full_spark_meta
            )
            sm = (
                (swa_meta.slot_mapping if swa_meta else common_meta.slot_mapping)
                if is_swa_group
                else common_meta.slot_mapping
            )
            for name in layer_names:
                attn_metadata_dict[name] = meta
                slot_mapping_dict[name] = sm

        # Build input tensors
        if is_decode:
            input_ids = backend._decode_input_ids[:num_reqs]
            positions = backend._decode_positions[:num_reqs]
        else:
            if num_reqs == 1:
                flat_token_ids = token_ids
            else:
                flat_token_ids = [tok for slot_tokens in token_ids for tok in slot_tokens]
            input_ids = torch.tensor(flat_token_ids, dtype=torch.long, device=self.device)
            positions_list = []
            for kv_len, qo in zip(kv_lengths, qo_lens):
                positions_list.extend(range(kv_len, kv_len + qo))
            positions = torch.tensor(positions_list, dtype=torch.long, device=self.device)

        with bf_attn_context(attn_metadata_dict, slot_mapping_dict):
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
        """Build owned attention metadata for the draft-model forward.

        Draft KV cache is a ring buffer (SWA window=512). All positions
        must be wrapped modulo ring_slots to avoid OOB writes.
        """
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

        return LagunaAttentionMetadata(
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
        attn_metadata_dict = {name: draft_meta for name in self._draft_layer_names}
        slot_mapping_dict = {
            name: self._draft_slot_mapping[:num_tokens] for name in self._draft_layer_names
        }

        # Run draft model forward. bf_attn_context added 阶段7-补充: draft
        # attention layers are now BFAttention instances too (see
        # _patch_draft_sparkinfer's docstring), which read this context
        # instead of vLLM's real get_forward_context() -- this call site
        # previously only set up set_forward_context because the real
        # (now-replaced) Attention.forward() used that exclusively.
        # Caught by an actual GPU run (RuntimeError: BFAttention was
        # called without a scoped attention context), not by inspection.
        with bf_attn_context(attn_metadata_dict, slot_mapping_dict):
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
        context_positions = torch.tensor([position], dtype=torch.long, device=self.device)
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
        verify_argmax = verify_logits[: num_tokens - 1].argmax(dim=-1).tolist()
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
        num_tokens = 1 + len(draft_tokens)  # 16
        verify_tokens = [bonus_token] + draft_tokens

        # Build attention metadata for parallel verify
        # Use decode-style buffers but with qo_len=num_tokens
        logits, _ = self._forward_verify(slot, verify_tokens, kv_len, num_tokens)

        # Greedy verification:
        # logits[i] predicts token at position kv_len + i + 1
        # logits[0] → should match draft_tokens[0]
        verify_argmax = logits[: num_tokens - 1].argmax(dim=-1).tolist()
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

        full_meta = LagunaAttentionMetadata(
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

            swa_meta = LagunaAttentionMetadata(
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
            metadata = builder.build(common_prefix_len=0, common_attn_metadata=meta)
            for name in backend._layer_groups[group_key]:
                attn_metadata_dict[name] = metadata
                slot_mapping_dict[name] = meta.slot_mapping

        with bf_attn_context(attn_metadata_dict, slot_mapping_dict):
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
            combined = self.draft_model.combine_hidden_states(combined_input)  # [1, 3072]
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
            accepted, num_accepted = self._accept_reject(verify_logits, draft_tokens, bonus_token)
        else:
            accepted, num_accepted = self._verify(slot, bonus_token, draft_tokens, kv_len + 1)

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

        # A chunked-prefill final chunk can be much larger than the draft
        # ring's capacity (chunk_len is sized for the *main* model's SWA
        # window/prefetch, independent of the draft ring, which only needs
        # to hold DRAFT_WINDOW-ish positions). Writing more positions than
        # ring_slots means multiple positions alias the same ring slot;
        # advanced-indexing assignment on CUDA does not guarantee "last
        # write wins" for duplicate destination indices, so the ring could
        # end up with a scrambled mix of stale and fresh KV instead of the
        # intended most-recent-window content. Only the last ring_slots
        # positions matter going forward (older ones would be overwritten
        # by the ring wrap anyway), so clip to them -- this also guarantees
        # every slot_mapping in this call is unique, sidestepping the
        # duplicate-index hazard entirely rather than relying on it being
        # benign.
        if num_positions > ring_slots:
            drop = num_positions - ring_slots
            combined = combined[drop:]
            position_offset += drop
            num_positions = ring_slots

        if os.environ.get("QSR_DEBUG_CHUNK_CHECK") in ("1", "2"):
            logger.warning(
                "CHUNK_CHECK combined_stats bs=%d shape=%s mean=%.6g std=%.6g "
                "first_row_mean=%.6g last_row_mean=%.6g",
                bs,
                tuple(combined.shape),
                combined.float().mean().item(),
                combined.float().std().item(),
                combined[0].float().mean().item(),
                combined[-1].float().mean().item(),
            )

        context_positions = torch.arange(
            position_offset, position_offset + num_positions, dtype=torch.long, device=self.device
        )
        slot_mappings = torch.zeros(num_positions, dtype=torch.long, device=self.device)
        for i in range(num_positions):
            pos = position_offset + i
            ring_block = (pos % ring_slots) // bs
            ring_off = pos % bs
            slot_mappings[i] = (draft_base + ring_block) * bs + ring_off

        if os.environ.get("QSR_DEBUG_CHUNK_CHECK") in ("1", "2"):
            n_unique = torch.unique(slot_mappings).numel()
            logger.warning(
                "CHUNK_CHECK draft_kv_precompute bs=%d num_positions=%d ring_slots=%d "
                "position_offset=%d n_unique_slot_mappings=%d (dup=%d)",
                bs,
                num_positions,
                ring_slots,
                position_offset,
                n_unique,
                num_positions - n_unique,
            )

        self.draft_model.precompute_and_store_context_kv(
            combined,
            context_positions,
            slot_mappings,
        )

        if os.environ.get("QSR_DEBUG_CHUNK_CHECK") in ("1", "2"):
            name0 = self._draft_layer_names[0]
            kv = self._draft_kv_caches[name0]
            sample_idx = [0, num_positions // 2, num_positions - 1]
            for i in sample_idx:
                sm = int(slot_mappings[i].item())
                blk, off = sm // bs, sm % bs
                k_val = kv[0, blk, off].float()
                v_val = kv[1, blk, off].float()
                logger.warning(
                    "CHUNK_CHECK draft_kv_readback bs=%d i=%d pos=%d slot_mapping=%d "
                    "block=%d off=%d k_abs_sum=%.6g v_abs_sum=%.6g k_nonzero=%s",
                    bs,
                    i,
                    position_offset + i,
                    sm,
                    blk,
                    off,
                    k_val.abs().sum().item(),
                    v_val.abs().sum().item(),
                    bool((k_val != 0).any().item()),
                )

        logger.info(
            "DFlash: precomputed context KV for %d positions (offset=%d)",
            num_positions,
            position_offset,
        )

    def _lazy_capture_cg(self) -> None:
        """Compatibility hook for engines initialized without an eager capture."""
        if self._draft_cg is None:
            self._capture_draft_cg()
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
            prompt_ids,
            max_tokens,
            temperature,
            eos_tokens,
            slot=slot,
            enable_prefix_cache=enable_prefix_cache,
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
        prefill_observer: Callable[[int, int, int, list[torch.Tensor] | None], None] | None = None,
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
        cached_kv_len = backend.slot_kv_len[slot]

        # Prefix cache: find matching prefix in cached KV
        prefix_len = 0
        exact_cold_replay = False
        if enable_prefix_cache:
            prefix_len = backend.find_prefix_match(slot, prompt_ids)
            if 0 < prefix_len < prompt_len:
                matched_prefix_len = prefix_len
                prefix_len = backend.prepare_exact_prefix_replay(
                    slot, prompt_ids, matched_prefix_len
                )
                exact_cold_replay = prefix_len is not None
                if prefix_len is None:
                    logger.info(
                        "Prefix cache MISS: partial reuse lacks cold-boundary SWA provenance"
                    )
                    prefix_len = 0
            elif prefix_len >= prompt_len:
                ring_specs = (
                    (backend._ring_slots_per_slot, backend._swa_window),
                    (self._draft_blocks_per_slot * self.block_size, DRAFT_WINDOW),
                )
                if not _ring_prefix_reuse_is_safe(
                    cached_kv_len,
                    prefix_len,
                    ring_specs,
                ):
                    prefix_len = backend.prepare_exact_prefix_replay(
                        slot, prompt_ids, prompt_len
                    )
                    exact_cold_replay = prefix_len is not None
                    if exact_cold_replay:
                        logger.info(
                            "Prefix cache REPLAY: full hit rewinds %d positions from "
                            "cold boundary %d",
                            cached_kv_len - prompt_len,
                            prefix_len,
                        )
                    else:
                        logger.info(
                            "Prefix cache MISS: rewinding %d KV positions exceeds ring history",
                            cached_kv_len - prompt_len,
                        )
                        logger.info(
                            "Prefix cache MISS: expired full hit lacks cold-boundary SWA provenance"
                        )
                        prefix_len = 0

        if prefix_len > 0 and prefix_len < prompt_len:
            # Partial match: continue from cached prefix
            logger.info(
                "Prefix cache HIT: %d/%d tokens cached, prefilling %d suffix",
                prefix_len,
                prompt_len,
                prompt_len - prefix_len,
            )
        elif prefix_len >= prompt_len:
            # Full match: keep the retained prompt-boundary ring state intact.
            # The safety check above guarantees later generation has not
            # overwritten the attention window needed at this boundary.
            logger.info(
                "Prefix cache FULL HIT: %d/%d tokens cached",
                prefix_len,
                prompt_len,
            )
        else:
            # No match: full reset and prefill
            backend.reset_slot(slot)
            for kv_tensor in self._draft_kv_caches.values():
                kv_tensor.zero_()

        t0 = time.perf_counter()

        if prefix_len > 0:
            # Continue from cached prefix
            first_token, aux_hidden_states = backend.continue_prefill_with_aux(
                slot, prompt_ids, prefix_len, exact_cold_replay=exact_cold_replay
            )
        else:
            # Full prefill
            first_token, aux_hidden_states = backend.prefill_with_aux(slot, prompt_ids)

        if prefill_observer is not None:
            prefill_observer(slot, prefix_len, first_token, aux_hidden_states)

        # Bulk precompute draft context KV from prefill aux
        if aux_hidden_states is not None:
            aux_len = aux_hidden_states[0].shape[0]
            if prefix_len > 0:
                aux_offset = prompt_len - aux_len
            else:
                aux_offset = prompt_len - aux_len
            if os.environ.get("QSR_DEBUG_CHUNK_CHECK") in ("1", "2"):
                logger.warning(
                    "CHUNK_CHECK generate_verify_only prompt_len=%d prefix_len=%d "
                    "aux_len=%d aux_offset=%d n_aux_tensors=%d",
                    prompt_len,
                    prefix_len,
                    aux_len,
                    aux_offset,
                    len(aux_hidden_states),
                )
            self._bulk_precompute_context_kv(slot, aux_hidden_states, aux_len, aux_offset)

        del aux_hidden_states
        t_prefill = time.perf_counter()

        # Bootstrap: initial bonus = first_token, run draft to get initial 15 tokens
        bonus_token = first_token
        kv_len = backend.slot_kv_len[slot]  # = prompt_len

        _force_sync = os.environ.get("QSR_DEBUG_FORCE_SYNC") == "1"

        if self._draft_cg is not None:
            draft_tokens = self._draft_cg.replay(slot, bonus_token, kv_len)
        else:
            draft_tokens = self._draft_forward(slot, bonus_token, kv_len)
        if _force_sync:
            torch.cuda.synchronize()

        if os.environ.get("QSR_DEBUG_CHUNK_CHECK") in ("1", "2"):
            logger.warning(
                "CHUNK_CHECK initial_draft bs=%d bonus_token=%d kv_len=%d draft_tokens=%s",
                self.block_size,
                bonus_token,
                kv_len,
                draft_tokens,
            )

        tokens = [first_token]
        total_draft = 0
        total_accepted = 0
        num_steps = 0

        # Cache loop-invariant values outside the hot decode loop
        _trace_enabled = bfdiag_trace.TRACE_ENABLED
        _chunk_check = os.environ.get("QSR_DEBUG_CHUNK_CHECK") in ("1", "2")
        _verify_dump_path = os.environ.get("QSR_DEBUG_VERIFY_LOGITS_FILE")
        _bs = self.block_size
        _phys = _physical_slot(slot)
        _draft_base = _phys * self._draft_blocks_per_slot
        _ring_slots = self._draft_blocks_per_slot * _bs
        # Pre-allocate context position buffer (max M=16 tokens per round)
        _ctx_pos_buf = torch.arange(
            NUM_QUERY_PER_REQ, dtype=torch.long, device=self.device
        )

        while len(tokens) < max_tokens:
            num_steps += 1
            total_draft += NUM_SPECULATIVE_TOKENS
            kv_len = backend.slot_kv_len[slot]
            _bf_row = (
                bfdiag_trace.begin_round(slot, kv_len) if _trace_enabled else -1
            )

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
            if bfdiag_trace.TRACE_ENABLED:
                bfdiag_trace.mark(_bf_row, bfdiag_events.PHASE_VERIFY)
            if _force_sync:
                torch.cuda.synchronize()

            # Step 2: Accept/reject (single GPU→CPU sync for all 16 argmax)
            all_argmax = verify_logits[:NUM_QUERY_PER_REQ].argmax(dim=-1).tolist()
            if _force_sync:
                torch.cuda.synchronize()

            if _verify_dump_path:
                vl = verify_logits[:NUM_QUERY_PER_REQ].float()
                vtop2 = vl.topk(2, dim=-1)
                vpositions = []
                for j in range(vl.shape[0]):
                    vpositions.append(
                        {
                            "top1_tok": int(vtop2.indices[j, 0]),
                            "top1_val": round(float(vtop2.values[j, 0]), 6),
                            "top2_tok": int(vtop2.indices[j, 1]),
                            "top2_val": round(float(vtop2.values[j, 1]), 6),
                        }
                    )
                import json as _json

                with open(_verify_dump_path, "a") as _f:
                    _f.write(
                        _json.dumps(
                            {
                                "bs": self.block_size,
                                "kv_len": kv_len,
                                "bonus_token": bonus_token,
                                "draft_tokens": draft_tokens,
                                "positions": vpositions,
                            }
                        )
                        + "\n"
                    )

            decision = _verify_only_accept_reject(all_argmax, draft_tokens, bonus_token)
            num_accepted = decision["num_accepted"]
            new_tokens = decision["committed"]
            total_accepted += num_accepted

            # The recovery token on rejection, or target bonus after a full
            # accept, is the pending anchor for the next draft round.
            new_bonus = decision["next_anchor"]

            # The verifier wrote the old anchor plus every matching draft token.
            # The recovery/bonus token remains pending and is not in KV yet.
            context_count = decision["context_count"]

            if _chunk_check:
                logger.warning(
                    "CHUNK_CHECK round bs=%d num_steps=%d kv_len=%d bonus_token=%d "
                    "draft_tokens=%s num_accepted=%d new_tokens=%s new_bonus=%d",
                    self.block_size,
                    num_steps,
                    kv_len,
                    bonus_token,
                    draft_tokens,
                    num_accepted,
                    new_tokens,
                    new_bonus,
                )

            # Step 4: Precompute draft context KV from committed verifier inputs.
            if verify_aux is not None and context_count > 0:
                aux_slice = [a[:context_count] for a in verify_aux]
                combined_input = torch.cat(aux_slice, dim=-1)
                combined = self.draft_model.combine_hidden_states(combined_input)
                context_positions = _ctx_pos_buf[:context_count] + kv_len
                ring_blocks = (context_positions % _ring_slots) // _bs
                ring_offs = context_positions % _bs
                slot_mappings = (_draft_base + ring_blocks) * _bs + ring_offs
                self.draft_model.precompute_and_store_context_kv(
                    combined, context_positions, slot_mappings
                )
            if _force_sync:
                torch.cuda.synchronize()

            # Step 5: Update state
            backend.slot_kv_len[slot] += context_count
            backend.slot_committed_tokens[slot].extend(new_tokens)
            tokens.extend(new_tokens)
            if _trace_enabled:
                bfdiag_trace.mark(_bf_row, bfdiag_events.PHASE_COMMIT)

            # Check EOS
            found_eos = False
            for tok in new_tokens:
                if tok in eos_tokens:
                    idx = len(tokens) - len(new_tokens) + new_tokens.index(tok)
                    tokens = tokens[: idx + 1]
                    found_eos = True
                    break
            if found_eos:
                if bfdiag_trace.TRACE_ENABLED:
                    bfdiag_trace.finish_dflash_round(
                        _bf_row,
                        self._verify_cg is not None,
                        self._use_cuda_graph,
                        len(draft_tokens),
                        decision,
                        new_bonus,
                    )
                break

            # Step 6: Draft next 15 tokens
            bonus_token = new_bonus
            new_kv_len = backend.slot_kv_len[slot]
            if self._draft_cg is not None:
                draft_tokens = self._draft_cg.replay(slot, bonus_token, new_kv_len)
            else:
                draft_tokens = self._draft_forward(slot, bonus_token, new_kv_len)
            if _force_sync:
                torch.cuda.synchronize()
            if bfdiag_trace.TRACE_ENABLED:
                bfdiag_trace.finish_dflash_round(
                    _bf_row,
                    self._verify_cg is not None,
                    self._use_cuda_graph,
                    len(draft_tokens),
                    decision,
                    new_bonus,
                )

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

    def dflash_prefill_bootstrap(self, slot: int, prompt_ids: list[int]) -> dict:
        """E1: DFlash-aware prefill for ServerEngine's admission path.

        Mirrors ``generate_verify_only``'s prefill + initial-draft bootstrap
        (no prefix-cache reuse -- matches ``LagunaBackend.prefill_chunked_begin``'s
        existing simplicity; Laguna prefix caching is a separate, unbuilt
        roadmap item per ``reconcile_prefix_hit``). Returns the same
        ``{"anchor": int, "draft_tokens": list[int]}`` shape
        ``prefill_chunked_begin`` already returns for the non-DFlash path, so
        callers don't need to special-case it.
        """
        backend = self.backend
        prompt_len = len(prompt_ids)
        kv_before = backend.slot_kv_len[slot]  # prefix hit offset (0 for cold)
        first_token, aux_hidden_states = backend.prefill_with_aux(slot, prompt_ids)

        if aux_hidden_states is not None:
            aux_len = aux_hidden_states[0].shape[0]
            # aux_offset is relative to the FULL context (prefix + suffix)
            total_kv = backend.slot_kv_len[slot]
            aux_offset = total_kv - aux_len
            self._bulk_precompute_context_kv(slot, aux_hidden_states, aux_len, aux_offset)
        del aux_hidden_states

        bonus_token = first_token
        kv_len = backend.slot_kv_len[slot]
        if self._draft_cg is not None:
            draft_tokens = self._draft_cg.replay(slot, bonus_token, kv_len)
        else:
            draft_tokens = self._draft_forward(slot, bonus_token, kv_len)

        if self._use_cuda_graph and not self._cg_captured:
            self._lazy_capture_cg()

        return {"anchor": bonus_token, "draft_tokens": draft_tokens}

    def dflash_round(
        self,
        slot: int,
        anchor: int,
        draft_tokens: list[int],
        *,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> dict:
        """E1: one draft+verify+accept round for ServerEngine's per-step decode
        loop -- the step-wise sibling of ``generate_verify_only``'s while-loop
        body, extracted so ``ServerEngine._step_sync`` can drive DFlash one
        round per engine step instead of running a whole generation
        to completion. Matches the decision-dict contract
        ``DirectModelRunner.mtp_verify_and_commit_batch`` already returns
        (``committed``/``num_accepted``/``next_anchor``/``next_draft_tokens``[/``logprobs``])
        so ``LagunaBackend.mtp_verify_and_commit_batch`` can hand this straight
        to the existing, unmodified ``_step_sync`` greedy-MTP branch.

        EOS/max_tokens truncation is intentionally NOT done here -- the caller
        (``_step_sync``) already walks ``committed`` looking for EOS / the
        length limit for the qwen36 MTP path, and applies unchanged here. A
        drafted-but-never-consumed next round when the caller stops after
        this one is harmless (the physical slot's KV/committed-token state
        gets reset before its next real use).
        """
        backend = self.backend
        bonus_token = anchor
        kv_len = backend.slot_kv_len[slot]
        _bf_row = bfdiag_trace.begin_round(slot, kv_len) if bfdiag_trace.TRACE_ENABLED else -1

        verify_tokens = [bonus_token] + draft_tokens
        if self._verify_cg is not None:
            verify_logits, verify_aux = self._verify_cg.replay_with_aux(slot, verify_tokens, kv_len)
        else:
            verify_logits, verify_aux = self._forward_verify_with_aux(
                slot, verify_tokens, kv_len, len(verify_tokens)
            )
        if bfdiag_trace.TRACE_ENABLED:
            bfdiag_trace.mark(_bf_row, bfdiag_events.PHASE_VERIFY)

        all_argmax = verify_logits[:NUM_QUERY_PER_REQ].argmax(dim=-1).tolist()
        decision = _verify_only_accept_reject(all_argmax, draft_tokens, bonus_token)
        committed = decision["committed"]
        new_bonus = decision["next_anchor"]
        context_count = decision["context_count"]
        bfdiag_checks.check_accepted_bound(slot, len(committed), len(draft_tokens))

        logprobs_list = None
        if return_logprobs:
            from runtime.logprobs import compute_logprobs

            logprobs_list = [
                compute_logprobs(verify_logits[p].unsqueeze(0), [committed[p]], top_k=top_logprobs)[
                    0
                ]
                for p in range(len(committed))
            ]

        if verify_aux is not None and context_count > 0:
            aux_slice = [a[:context_count] for a in verify_aux]
            combined_input = torch.cat(aux_slice, dim=-1)
            combined = self.draft_model.combine_hidden_states(combined_input)
            bs = self.block_size
            phys = _physical_slot(slot)
            draft_base = phys * self._draft_blocks_per_slot
            ring_slots = self._draft_blocks_per_slot * bs
            context_positions = torch.arange(
                kv_len, kv_len + context_count, dtype=torch.long, device=self.device
            )
            ring_blocks = (context_positions % ring_slots) // bs
            ring_offs = context_positions % bs
            slot_mappings = (draft_base + ring_blocks) * bs + ring_offs
            self.draft_model.precompute_and_store_context_kv(
                combined, context_positions, slot_mappings
            )

        backend.slot_kv_len[slot] += context_count
        backend.slot_committed_tokens[slot].extend(committed)
        bfdiag_checks.check_kv_len_monotonic(slot, kv_len, backend.slot_kv_len[slot])
        bfdiag_checks.check_committed_ahead_of_kv_by_one(
            slot, backend.slot_kv_len[slot], len(backend.slot_committed_tokens[slot])
        )
        if bfdiag_trace.TRACE_ENABLED:
            bfdiag_trace.mark(_bf_row, bfdiag_events.PHASE_COMMIT)

        new_kv_len = backend.slot_kv_len[slot]
        if self._draft_cg is not None:
            next_draft_tokens = self._draft_cg.replay(slot, new_bonus, new_kv_len)
        else:
            next_draft_tokens = self._draft_forward(slot, new_bonus, new_kv_len)
        if bfdiag_trace.TRACE_ENABLED:
            bfdiag_trace.finish_dflash_round(
                _bf_row,
                self._verify_cg is not None,
                self._use_cuda_graph,
                len(draft_tokens),
                decision,
                new_bonus,
            )

        result = {
            "committed": committed,
            "num_accepted": decision["num_accepted"],
            "next_anchor": new_bonus,
            "next_draft_tokens": next_draft_tokens,
        }
        if logprobs_list is not None:
            result["logprobs"] = logprobs_list
        return result

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

        full_meta = LagunaAttentionMetadata(
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

        # SWA ring metadata
        swa_meta = None
        if backend._ring_blocks_per_slot > 0:
            swa_meta = backend._build_swa_attn_metadata(
                [slot], [kv_len], [num_tokens], False, swa_mode="verify_ring"
            )

        attn_metadata_dict = {}
        slot_mapping_dict = {}
        full_spark_meta = backend._build_sparkinfer_metadata(
            full_meta, window_left=-1, mode="verify"
        )
        swa_spark_meta = None
        if swa_meta is not None:
            swa_wl = backend._swa_window - 1 if backend._swa_window > 0 else -1
            swa_spark_meta = backend._build_sparkinfer_metadata(
                swa_meta, window_left=swa_wl, mode="verify"
            )
        for group_key, layer_names in backend._layer_groups.items():
            wl = group_key[0]
            is_swa = wl >= 0
            meta = swa_spark_meta if (is_swa and swa_spark_meta is not None) else full_spark_meta
            for name in layer_names:
                attn_metadata_dict[name] = meta
                slot_mapping_dict[name] = full_sm if not is_swa else swa_meta.slot_mapping

        with bf_attn_context(attn_metadata_dict, slot_mapping_dict):
            result = backend.model.forward(input_ids, positions)

        if isinstance(result, tuple):
            hidden_states, aux = result
        else:
            hidden_states, aux = result, None
        logits = backend.model.compute_logits(hidden_states)
        return logits, aux
