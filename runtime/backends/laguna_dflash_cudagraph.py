"""CUDA Graph wrapper for DFlash speculative decoding.

The legacy verify graph still has a FlashInfer implementation below, but the
active DFlash draft graph uses Sparkinfer's paged-verify CUDA-graph path.

Per speculative step:
1. Main decode (M=1): LagunaCudaGraphDecode (existing)
2. combine + precompute_context_kv: eager (small ops)
3. Draft forward (M=16): CUDA Graph with prefill wrapper
4. Verify forward (M=16): CUDA Graph with prefill wrapper
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from runtime.backends.dflash_constants import (
    DRAFT_WINDOW,
    MASK_TOKEN_ID,
    NUM_QUERY_PER_REQ,
)
from runtime.backends.laguna import LagunaBackend, _physical_slot

logger = logging.getLogger("qwen_sm120_runtime.dflash_cudagraph")


class DFlashVerifyCudaGraph:
    """CUDA Graph for main model verify forward (M=16).

    Uses FlashInfer prefill wrappers with use_cuda_graph=True.
    Separate wrappers for full-attention and SWA layer groups.
    """

    def __init__(self, backend: LagunaBackend) -> None:
        self.backend = backend
        self.device = backend.device
        self.block_size = backend.block_size
        self.blocks_per_slot = backend.blocks_per_slot
        self.num_tokens = NUM_QUERY_PER_REQ  # 16

        # Ring buffer params
        self._ring_blocks_per_slot = backend._ring_blocks_per_slot
        self._ring_slots_per_slot = backend._ring_slots_per_slot
        self._swa_window = backend._swa_window

        # Pre-allocated input buffers (fixed address for graph)
        self._input_ids = torch.zeros(self.num_tokens, dtype=torch.long, device=self.device)
        self._positions = torch.zeros(self.num_tokens, dtype=torch.long, device=self.device)

        # Full-attention FlashInfer buffers
        max_full_pages = self.blocks_per_slot
        self._full_qo_indptr = torch.tensor(
            [0, self.num_tokens], dtype=torch.int32, device=self.device
        )
        self._full_kv_indptr_cpu = torch.zeros(2, dtype=torch.int32, pin_memory=True)
        self._full_kv_indptr_gpu = torch.zeros(2, dtype=torch.int32, device=self.device)
        self._full_kv_indices = torch.zeros(max_full_pages, dtype=torch.int32, device=self.device)
        self._full_last_page_len_cpu = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self._full_last_page_len_gpu = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._full_slot_mapping = torch.zeros(self.num_tokens, dtype=torch.long, device=self.device)

        # SWA ring FlashInfer buffers
        if self._ring_blocks_per_slot > 0:
            max_swa_pages = self._ring_blocks_per_slot
            self._swa_qo_indptr = torch.tensor(
                [0, self.num_tokens], dtype=torch.int32, device=self.device
            )
            self._swa_kv_indptr_cpu = torch.zeros(2, dtype=torch.int32, pin_memory=True)
            self._swa_kv_indptr_gpu = torch.zeros(2, dtype=torch.int32, device=self.device)
            self._swa_kv_indices = torch.zeros(max_swa_pages, dtype=torch.int32, device=self.device)
            self._swa_last_page_len_cpu = torch.zeros(1, dtype=torch.int32, pin_memory=True)
            self._swa_last_page_len_gpu = torch.zeros(1, dtype=torch.int32, device=self.device)
            self._swa_slot_mapping = torch.zeros(
                self.num_tokens, dtype=torch.long, device=self.device
            )

        # FlashInfer prefill wrappers (one per layer group)
        self._prefill_wrappers: dict[tuple, Any] = {}
        self._workspaces: list[torch.Tensor] = []

        # Graph state
        self._graph: torch.cuda.CUDAGraph | None = None
        self._aux_hidden_states: list[torch.Tensor] | None = None
        self._logits: torch.Tensor | None = None
        self._captured = False

    def _init_wrappers(self) -> None:
        """Create FlashInfer prefill wrappers with use_cuda_graph=True."""
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

        backend = self.backend
        for group_key, builder in backend._metadata_builders.items():
            workspace = torch.empty(
                builder._get_workspace_buffer().numel(),
                dtype=torch.uint8,
                device=self.device,
            )
            wl = group_key[0]
            is_swa = wl >= 0 and self._ring_blocks_per_slot > 0
            if is_swa:
                wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    workspace,
                    "NHD",
                    use_cuda_graph=True,
                    qo_indptr_buf=self._swa_qo_indptr,
                    paged_kv_indptr_buf=self._swa_kv_indptr_gpu,
                    paged_kv_indices_buf=self._swa_kv_indices,
                    paged_kv_last_page_len_buf=self._swa_last_page_len_gpu,
                )
            else:
                wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    workspace,
                    "NHD",
                    use_cuda_graph=True,
                    qo_indptr_buf=self._full_qo_indptr,
                    paged_kv_indptr_buf=self._full_kv_indptr_gpu,
                    paged_kv_indices_buf=self._full_kv_indices,
                    paged_kv_last_page_len_buf=self._full_last_page_len_gpu,
                )
            self._prefill_wrappers[group_key] = wrapper
            self._workspaces.append(workspace)

    def _fill_buffers(self, slot: int, kv_len: int) -> None:
        """Update pre-allocated buffers for verify replay."""
        bs = self.block_size
        phys = _physical_slot(slot)
        new_kv_len = kv_len + self.num_tokens

        # Positions
        self._positions[: self.num_tokens] = torch.arange(
            kv_len, kv_len + self.num_tokens, dtype=torch.long, device=self.device
        )

        # Full-attention buffers
        full_base = phys * self.blocks_per_slot
        n_full_blocks = (new_kv_len + bs - 1) // bs
        self._full_kv_indices[:n_full_blocks] = torch.arange(
            full_base, full_base + n_full_blocks, dtype=torch.int32, device=self.device
        )
        self._full_kv_indptr_cpu[0] = 0
        self._full_kv_indptr_cpu[1] = n_full_blocks
        lpl = new_kv_len % bs
        self._full_last_page_len_cpu[0] = lpl if lpl != 0 else bs
        self._full_kv_indptr_gpu[:2].copy_(self._full_kv_indptr_cpu[:2], non_blocking=True)
        self._full_last_page_len_gpu[:1].copy_(self._full_last_page_len_cpu[:1], non_blocking=True)

        # Full slot mapping (vectorized) -- reuse self._positions, already
        # holds this exact arange(kv_len, kv_len+num_tokens) from above.
        _pos = self._positions[: self.num_tokens]
        self._full_slot_mapping[: self.num_tokens] = (full_base + _pos // bs) * bs + _pos % bs

        # SWA ring buffers
        if self._ring_blocks_per_slot > 0:
            ring_base = phys * self._ring_blocks_per_slot
            ring_slots = self._ring_slots_per_slot
            window = self._swa_window

            window_start = max(0, kv_len - window + 1)
            aligned_start = (window_start // bs) * bs
            aligned_len = new_kv_len - aligned_start
            n_ring = min((aligned_len + bs - 1) // bs, self._ring_blocks_per_slot)

            # Vectorized ring block indices
            _ap = torch.arange(
                aligned_start,
                aligned_start + n_ring * bs,
                bs,
                device=self.device,
                dtype=torch.long,
            )
            self._swa_kv_indices[:n_ring] = ring_base + (_ap % ring_slots) // bs

            self._swa_kv_indptr_cpu[0] = 0
            self._swa_kv_indptr_cpu[1] = n_ring
            swa_lpl = aligned_len % bs
            self._swa_last_page_len_cpu[0] = swa_lpl if swa_lpl != 0 else bs
            self._swa_kv_indptr_gpu[:2].copy_(self._swa_kv_indptr_cpu[:2], non_blocking=True)
            self._swa_last_page_len_gpu[:1].copy_(
                self._swa_last_page_len_cpu[:1], non_blocking=True
            )

            # Vectorized SWA slot mapping -- reuse self._positions (same
            # arange as above, no need to recompute a third time).
            _sp = self._positions[: self.num_tokens]
            _rb = (_sp % ring_slots) // bs
            _ro = _sp % bs
            self._swa_slot_mapping[: self.num_tokens] = (ring_base + _rb) * bs + _ro

    def _run_plan(self) -> None:
        """Run FlashInfer plan on all prefill wrappers."""

        backend = self.backend
        for group_key, wrapper in self._prefill_wrappers.items():
            wl, nqh, nkvh = group_key
            head_dim = backend.head_dim
            page_size = self.block_size
            is_swa = wl >= 0 and self._ring_blocks_per_slot > 0

            kv_dtype = torch.float8_e4m3fn if "fp8" in backend._cache_dtype_str else torch.bfloat16
            builder_sm_scale = backend._metadata_builders[group_key].sm_scale

            if is_swa:
                qo_indptr = self._swa_qo_indptr
                kv_indptr = self._swa_kv_indptr_gpu
                kv_indices = self._swa_kv_indices
                last_page_len = self._swa_last_page_len_gpu
                window_left = wl
            else:
                qo_indptr = self._full_qo_indptr
                kv_indptr = self._full_kv_indptr_gpu
                kv_indices = self._full_kv_indices
                last_page_len = self._full_last_page_len_gpu
                window_left = -1

            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_indptr,
                paged_kv_indices=kv_indices,
                paged_kv_last_page_len=last_page_len,
                num_qo_heads=nqh,
                num_kv_heads=nkvh,
                head_dim_qk=head_dim,
                page_size=page_size,
                causal=True,
                pos_encoding_mode="NONE",
                window_left=window_left,
                logits_soft_cap=None,
                q_data_type=torch.bfloat16,
                kv_data_type=kv_dtype,
                sm_scale=builder_sm_scale,
                disable_split_kv=True,
            )

    def _build_metadata_and_forward(self) -> torch.Tensor:
        """Build FlashInferMetadata from pre-allocated buffers and run forward."""
        from vllm.v1.attention.backends.flashinfer import FIPrefill, FlashInferMetadata

        from runtime.compat_vllm import set_current_vllm_config, set_forward_context

        backend = self.backend
        attn_metadata_dict: dict[str, Any] = {}
        slot_mapping_dict: dict[str, torch.Tensor] = {}

        for group_key, wrapper in self._prefill_wrappers.items():
            wl = group_key[0]
            is_swa = wl >= 0 and self._ring_blocks_per_slot > 0
            sm = (
                self._swa_slot_mapping[: self.num_tokens]
                if is_swa
                else self._full_slot_mapping[: self.num_tokens]
            )
            metadata = FlashInferMetadata(
                num_actual_tokens=self.num_tokens,
                slot_mapping=sm,
                q_data_type_prefill=torch.bfloat16,
                q_data_type_decode=torch.bfloat16,
                num_decodes=0,
                num_decode_tokens=0,
                num_prefills=1,
                num_prefill_tokens=self.num_tokens,
                causal=True,
                use_cascade=False,
                prefill=FIPrefill(wrapper=wrapper),
                decode=None,
                cascade_wrapper=None,
            )
            for name in backend._layer_groups[group_key]:
                attn_metadata_dict[name] = metadata
                slot_mapping_dict[name] = sm

        with set_current_vllm_config(backend.vllm_config):
            with set_forward_context(
                attn_metadata_dict, backend.vllm_config, slot_mapping=slot_mapping_dict
            ):
                result = backend.model.forward(
                    self._input_ids[: self.num_tokens],
                    self._positions[: self.num_tokens],
                )

        if isinstance(result, tuple):
            hidden_states, self._aux_hidden_states = result
        else:
            hidden_states = result
            self._aux_hidden_states = None
        return backend.model.compute_logits(hidden_states)

    def capture(self) -> None:
        """Warmup + plan + capture the verify forward."""
        if self._captured:
            return

        logger.info("Capturing DFlash verify CUDA Graph (M=%d)...", self.num_tokens)
        self._init_wrappers()

        # Warmup with dummy data (use max kv_len for grid size headroom)
        capture_kv = self.blocks_per_slot * self.block_size - self.num_tokens
        self._input_ids[: self.num_tokens] = 1
        self._fill_buffers(0, capture_kv)

        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                self._fill_buffers(0, capture_kv)
                self._run_plan()
                self._build_metadata_and_forward()
        side_stream.synchronize()

        # Final fill + plan before capture
        self._fill_buffers(0, capture_kv)
        self._run_plan()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._logits = self._build_metadata_and_forward()

        self._graph = graph
        self._captured = True
        logger.info("DFlash verify CUDA Graph captured")

    def replay(self, slot: int, tokens: list[int], kv_len: int) -> torch.Tensor:
        """Replay verify graph with real data. Returns logits [16, vocab]."""
        if self._graph is None:
            return None

        num_tokens = len(tokens)
        self._input_ids[:num_tokens] = torch.tensor(tokens, dtype=torch.long, device=self.device)
        self._fill_buffers(slot, kv_len)
        self._run_plan()
        self._graph.replay()
        return self._logits

    def replay_with_aux(
        self, slot: int, tokens: list[int], kv_len: int
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Replay verify graph and return (logits, aux_hidden_states)."""
        logits = self.replay(slot, tokens, kv_len)
        return logits, self._aux_hidden_states


class DFlashDraftCudaGraph:
    """CUDA Graph for draft model forward (M=16).

    Draft model has 6 SWA layers (window=512). It uses Sparkinfer's
    paged-verify graph path with a ring KV cache and no FlashInfer metadata or
    planning in the active graph path.

    Although this forward belongs to the draft model, its attention shape is
    a 16-token verification-style query.  Sparkinfer's ``extend`` graph plan
    freezes a non-split schedule from the capture-time KV length; replaying it
    against the shorter, ring-relative runtime lengths produces incorrect
    attention.  ``verify`` keeps the graph schedule valid across those
    changing lengths.
    """

    def __init__(self, engine) -> None:
        self.engine = engine
        self.device = engine.device
        self.block_size = engine.block_size
        self.num_tokens = NUM_QUERY_PER_REQ

        self._draft_blocks_per_slot = engine._draft_blocks_per_slot
        self._ring_slots = self._draft_blocks_per_slot * self.block_size

        # Pre-allocated input buffers
        self._input_ids = torch.zeros(self.num_tokens, dtype=torch.long, device=self.device)
        self._positions = torch.zeros(self.num_tokens, dtype=torch.long, device=self.device)

        # Sparkinfer runtime metadata for draft (all layers are SWA).
        max_pages = self._draft_blocks_per_slot
        self._page_table = torch.zeros(1, max_pages, dtype=torch.int32, device=self.device)
        self._cache_seqlens = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._cu_seqlens_q = torch.tensor(
            [0, self.num_tokens], dtype=torch.int32, device=self.device
        )
        self._slot_mapping = torch.zeros(self.num_tokens, dtype=torch.long, device=self.device)

        self._workspace = None
        self._metadata = None
        self._attn_metadata_dict: dict[str, Any] | None = None
        self._slot_mapping_dict: dict[str, torch.Tensor] | None = None
        self._original_impls: dict[str, Any] = {}
        self._graph: torch.cuda.CUDAGraph | None = None
        self._logits: torch.Tensor | None = None
        self._captured = False

    def _init_workspace(self) -> None:
        """Create a fixed-address Sparkinfer verify workspace for draft CG."""
        from sparkinfer.attention.paged.planner import create_paged_plan
        from sparkinfer.attention.paged.workspace import PagedAttentionWorkspace

        from runtime.backends.dflash_constants import (
            DRAFT_HEAD_DIM,
            DRAFT_NUM_KV_HEADS,
            DRAFT_NUM_QO_HEADS,
        )
        from runtime.backends.laguna_cuda_graph import _SparkinferCGExtendMetadata

        max_pages = self._draft_blocks_per_slot
        q = torch.zeros(
            self.num_tokens,
            DRAFT_NUM_QO_HEADS,
            DRAFT_HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )
        k_cache = torch.zeros(
            max_pages,
            self.block_size,
            DRAFT_NUM_KV_HEADS,
            DRAFT_HEAD_DIM,
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        v_cache = torch.zeros_like(k_cache)
        workspace = PagedAttentionWorkspace.for_tensors(
            mode="verify",
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            use_cuda_graph=True,
        )

        max_kv = max_pages * self.block_size - 1
        capture_page_table = torch.arange(
            max_pages, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        capture_cache_seqlens = torch.tensor([max_kv], dtype=torch.int32, device=self.device)
        plan = create_paged_plan(
            q,
            k_cache,
            v_cache,
            capture_page_table,
            capture_cache_seqlens,
            self._cu_seqlens_q,
            mode="verify",
            enable_cuda_graph=True,
            window_left=DRAFT_WINDOW - 1,
        )
        workspace._ensure_capacity(plan)
        workspace._copy_runtime_metadata(
            capture_page_table, capture_cache_seqlens, self._cu_seqlens_q
        )
        workspace._copy_plan_metadata(plan)
        workspace._plan = plan

        self._workspace = workspace
        self._metadata = _SparkinferCGExtendMetadata(workspace, self.num_tokens)

    def _fill_buffers(self, slot: int, kv_len: int) -> None:
        bs = self.block_size
        phys = _physical_slot(slot)
        draft_base = phys * self._draft_blocks_per_slot
        ring_slots = self._ring_slots
        new_kv_len = kv_len + self.num_tokens

        self._positions[: self.num_tokens] = torch.arange(
            kv_len, kv_len + self.num_tokens, dtype=torch.long, device=self.device
        )

        window_start = max(0, kv_len - DRAFT_WINDOW + 1)
        aligned_start = (window_start // bs) * bs
        aligned_len = new_kv_len - aligned_start
        n_ring = min((aligned_len + bs - 1) // bs, self._draft_blocks_per_slot)

        # Vectorized ring page table. The physical indices address the shared
        # draft KV tensor directly; the Sparkinfer plan remains fixed.
        _ap = torch.arange(
            aligned_start,
            aligned_start + n_ring * bs,
            bs,
            device=self.device,
            dtype=torch.long,
        )
        self._page_table[0, :n_ring] = draft_base + (_ap % ring_slots) // bs
        self._cache_seqlens[0] = aligned_len

        # Vectorized slot mapping -- reuse self._positions (same arange as
        # above, no need to recompute).
        _sp = self._positions[: self.num_tokens]
        _rb = (_sp % ring_slots) // bs
        _ro = _sp % bs
        self._slot_mapping[: self.num_tokens] = (draft_base + _rb) * bs + _ro
        self._workspace._copy_runtime_metadata(
            self._page_table, self._cache_seqlens, self._cu_seqlens_q
        )

    def _patch_impls_for_cg(self) -> None:
        """Temporarily bind draft attention to the Sparkinfer graph workspace."""
        from runtime.backends.laguna_cuda_graph import _SparkinferCGExtendImpl

        for name in self.engine._draft_layer_names:
            attn = self.engine._draft_attn_layers[name]
            self._original_impls[name] = attn.impl
            attn.impl = _SparkinferCGExtendImpl(self._workspace, self.num_tokens)

    def _restore_impls(self) -> None:
        for name, impl in self._original_impls.items():
            self.engine._draft_attn_layers[name].impl = impl

    def _build_metadata_and_forward(self) -> torch.Tensor:
        from runtime.compat_vllm import set_current_vllm_config, set_forward_context

        engine = self.engine
        with set_current_vllm_config(engine.vllm_config):
            with set_forward_context(
                self._attn_metadata_dict,
                engine.vllm_config,
                slot_mapping=self._slot_mapping_dict,
                skip_compiled=True,
            ):
                draft_hidden = engine.draft_model(
                    input_ids=self._input_ids[: self.num_tokens],
                    positions=self._positions[: self.num_tokens],
                    inputs_embeds=None,
                )

        # Position 0's logits are never read (replay() only uses positions
        # 1..num_tokens-1 -- the 15 draft-token predictions; position 0's
        # hidden state is a required byproduct of causal attention over the
        # other 15 positions, but its own vocab projection is pure waste).
        # Slice before compute_logits, not after, so the vocab-size GEMM
        # only pays for the 15 positions that matter.
        return engine.draft_model.compute_logits(draft_hidden[1 : self.num_tokens])

    def capture(self) -> None:
        if self._captured:
            return

        logger.info("Capturing DFlash draft CUDA Graph (M=%d)...", self.num_tokens)
        self._init_workspace()
        self._patch_impls_for_cg()
        self._attn_metadata_dict = {name: self._metadata for name in self.engine._draft_layer_names}
        self._slot_mapping_dict = {
            name: self._slot_mapping for name in self.engine._draft_layer_names
        }

        try:
            capture_kv = 2048
            self._input_ids[0] = 1
            self._input_ids[1 : self.num_tokens] = MASK_TOKEN_ID
            self._fill_buffers(0, capture_kv)

            side_stream = torch.cuda.Stream()
            with torch.cuda.stream(side_stream):
                for _ in range(3):
                    self._fill_buffers(0, capture_kv)
                    self._build_metadata_and_forward()
            side_stream.synchronize()

            self._fill_buffers(0, capture_kv)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._logits = self._build_metadata_and_forward()

            self._graph = graph
            self._captured = True
            logger.info("DFlash draft CUDA Graph captured (sparkinfer verify)")
        finally:
            # Replay executes the captured graph directly. Restore the eager
            # Sparkinfer implementation so any fallback remains valid.
            self._restore_impls()

    def replay(self, slot: int, bonus_token: int, kv_len: int) -> list[int]:
        """Replay draft graph. Returns 15 draft tokens."""
        if self._graph is None:
            return []

        self._input_ids[0] = bonus_token
        self._input_ids[1 : self.num_tokens] = MASK_TOKEN_ID
        self._fill_buffers(slot, kv_len)
        self._graph.replay()
        # self._logits is already positions [1:num_tokens] -- see
        # _build_metadata_and_forward, which slices before compute_logits.
        draft_tokens = self._logits.argmax(dim=-1)
        return draft_tokens.tolist()
