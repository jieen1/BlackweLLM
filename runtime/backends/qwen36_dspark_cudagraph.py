"""CUDA Graphs for the external Qwen3.8 DSpark draft block.

The target verify graph lives next to the existing Qwen36 MTP verify graph:
DSpark only changes the source of the speculative tokens and the hidden taps
that are returned.  The draft is different from MTP's autoregressive head:
all ``gamma`` draft hidden states are produced by one masked dense forward,
then the small Markov recurrence is unrolled inside the same captured graph.

Each serving slot owns one graph because its draft KV pages are fixed tensor
addresses.  The graph's page table, sequence length, input anchor, and token
positions remain replay inputs; the model forward, shared LM head, Markov
correction, and greedy argmax stay inside the graph.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

from runtime.backends.bf_attention import bf_attn_context
from runtime.backends.flashinfer_dspark_attn import FlashInferDSparkAttentionImpl
from runtime.backends.laguna_cuda_graph import (
    _SparkinferCGExtendImpl,
    _SparkinferCGExtendMetadata,
)
from runtime.backends.laguna_sparkinfer_attn import SparkinferAttnMetadata

if TYPE_CHECKING:
    from runtime.backends.qwen36_dspark import Qwen36DSparkEngine

logger = logging.getLogger("qwen_sm120_runtime.qwen36_dspark_cudagraph")


class Qwen36DSparkDraftCudaGraph:
    """Capture and replay one fixed-slot, greedy DSpark draft block."""

    def __init__(self, engine: Qwen36DSparkEngine, slot: int) -> None:
        self.engine = engine
        self.slot = slot
        self.device = engine.device
        self.num_tokens = engine.k
        self.page_size = engine.page_size
        self.pages_per_slot = engine.pages_per_slot
        self.max_seq_len = engine.max_seq_len

        if slot < 0 or slot >= engine.backend.num_slots:
            raise ValueError(f"invalid DSpark draft graph slot: {slot}")

        self._input_ids = torch.full(
            (1, self.num_tokens),
            engine.draft_model.config.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        self._positions = torch.zeros(self.num_tokens, dtype=torch.long, device=self.device)
        self._page_table = torch.zeros(
            self.num_tokens, self.pages_per_slot, dtype=torch.int32, device=self.device
        )
        self._cache_seqlens = torch.zeros(self.num_tokens, dtype=torch.int32, device=self.device)
        self._cu_seqlens_q = torch.arange(
            self.num_tokens + 1, dtype=torch.int32, device=self.device
        )
        self._slot_mapping = torch.zeros(self.num_tokens, dtype=torch.long, device=self.device)
        self._pos_offset = torch.arange(self.num_tokens, dtype=torch.long, device=self.device)
        self._page_offset = torch.arange(self.pages_per_slot, dtype=torch.int32, device=self.device)
        self._tokens = torch.zeros(1, self.num_tokens, dtype=torch.long, device=self.device)

        self._workspace: Any | None = None
        self._metadata: Any | None = None
        self._flashinfer_impl: FlashInferDSparkAttentionImpl | None = None
        self._flash_metadata: SparkinferAttnMetadata | None = None
        self._attn_metadata: dict[str, Any] | None = None
        self._slot_mappings: dict[str, torch.Tensor] | None = None
        self._original_impls: dict[str, Any] = {}
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_pool: object | None = getattr(engine, "_draft_graph_pool", None)
        self._captured = False

    def _init_workspace(self) -> None:
        """Build one fixed-capacity graph-mode non-causal workspace.

        FlashInfer owns the graph path when available because the official
        Qwen3.8 DSpark draft attention is non-causal.  The SparkInfer branch
        below remains a compatibility fallback for environments without that
        optional backend.
        """

        if self.engine.use_flashinfer_draft:
            self._init_flashinfer_workspace()
            return

        from b12x.attention.paged.planner import create_paged_plan
        from b12x.attention.paged.workspace import PagedAttentionWorkspace

        first_attn = next(iter(self.engine._draft_attn_layers.values()))
        workspace = PagedAttentionWorkspace.for_contract(
            mode="extend",
            device=self.device,
            dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            num_q_heads=first_attn.num_heads,
            num_kv_heads=first_attn.num_kv_heads,
            head_dim_qk=first_attn.head_size,
            head_dim_vo=first_attn.head_size,
            page_size=self.page_size,
            max_total_q=self.num_tokens,
            num_cache_pages=self.pages_per_slot,
            use_cuda_graph=True,
        )
        assert workspace._plan_q is not None
        assert workspace._plan_k_cache is not None
        assert workspace._plan_v_cache is not None

        max_kv = self.pages_per_slot * self.page_size - 1
        capture_page_table = (
            torch.arange(
                self.slot * self.pages_per_slot,
                (self.slot + 1) * self.pages_per_slot,
                dtype=torch.int32,
                device=self.device,
            )
            .view(1, -1)
            .expand(self.num_tokens, -1)
        )
        capture_cache_seqlens = torch.full(
            (self.num_tokens,), max_kv, dtype=torch.int32, device=self.device
        )
        plan = create_paged_plan(
            workspace._plan_q,
            workspace._plan_k_cache,
            workspace._plan_v_cache,
            capture_page_table,
            capture_cache_seqlens,
            self._cu_seqlens_q,
            mode="extend",
            enable_cuda_graph=True,
            window_left=-1,
        )
        workspace._ensure_capacity(plan)
        workspace._copy_runtime_metadata(
            capture_page_table, capture_cache_seqlens, self._cu_seqlens_q
        )
        workspace._copy_plan_metadata(plan)
        workspace._plan = plan

        self._workspace = workspace
        self._metadata = _SparkinferCGExtendMetadata(workspace, self.num_tokens)

    def _init_flashinfer_workspace(self) -> None:
        first_attn = next(iter(self.engine._draft_attn_layers.values()))
        self._flashinfer_impl = FlashInferDSparkAttentionImpl(
            num_heads=first_attn.num_heads,
            head_size=first_attn.head_size,
            scale=first_attn.scale,
            num_kv_heads=first_attn.num_kv_heads,
            page_size=self.page_size,
            max_pages=self.pages_per_slot,
            num_tokens=self.num_tokens,
            device=self.device,
            use_cuda_graph=True,
            slot=self.slot,
            workspace_buffer=self.engine._flashinfer_workspace_buffer,
        )
        page_table = torch.arange(
            self.slot * self.pages_per_slot,
            (self.slot + 1) * self.pages_per_slot,
            dtype=torch.int32,
            device=self.device,
        ).view(1, -1)
        self._flash_metadata = SparkinferAttnMetadata(
            mode="extend",
            page_table=page_table,
            cache_seqlens=torch.zeros(1, dtype=torch.int32, device=self.device),
            cu_seqlens_q=torch.tensor([0, self.num_tokens], dtype=torch.int32, device=self.device),
            num_actual_tokens=self.num_tokens,
            window_left=-1,
        )

    def _fill_buffers(self, anchor_token: int, kv_len: int) -> None:
        if kv_len < 0 or kv_len + self.num_tokens > self.max_seq_len:
            raise RuntimeError(
                f"DSpark draft graph block does not fit: kv_len={kv_len}, "
                f"gamma={self.num_tokens}, max_seq_len={self.max_seq_len}"
            )

        self._input_ids.fill_(self.engine.draft_model.config.mask_token_id)
        self._input_ids[0, 0] = int(anchor_token)
        torch.add(self._pos_offset, kv_len, out=self._positions)
        page_row = self.slot * self.pages_per_slot + self._page_offset
        self._page_table.copy_(page_row.view(1, -1).expand(self.num_tokens, -1))
        self._cache_seqlens.fill_(kv_len + self.num_tokens)
        self._slot_mapping.copy_(
            self.slot * self.pages_per_slot * self.page_size + self._positions,
            non_blocking=True,
        )
        if self.engine.use_flashinfer_draft:
            assert self._flashinfer_impl is not None
            assert self._flash_metadata is not None
            self._flashinfer_impl.update_graph_metadata(kv_len)
            self._flash_metadata.cache_seqlens[0] = kv_len + self.num_tokens
            return
        assert self._workspace is not None
        self._workspace.update_prefill_graph_replay_metadata(
            self._page_table,
            self._cache_seqlens,
            self._cu_seqlens_q,
            window_left=-1,
        )

    def _patch_impls_for_capture(self) -> None:
        if self.engine.use_flashinfer_draft:
            assert self._flashinfer_impl is not None
            for name, attn in self.engine._draft_attn_layers.items():
                self._original_impls[name] = attn.impl
                attn.impl = self._flashinfer_impl
            return
        assert self._workspace is not None
        for name, attn in self.engine._draft_attn_layers.items():
            self._original_impls[name] = attn.impl
            attn.impl = _SparkinferCGExtendImpl(
                self._workspace,
                self.num_tokens,
                batch_size=self.num_tokens,
            )

    def _restore_impls(self) -> None:
        for name, impl in self._original_impls.items():
            self.engine._draft_attn_layers[name].impl = impl

    def _forward(self) -> torch.Tensor:
        assert self._metadata is not None or self._flash_metadata is not None
        assert self._attn_metadata is not None
        assert self._slot_mappings is not None
        draft_model = self.engine.draft_model
        with bf_attn_context(self._attn_metadata, self._slot_mappings):
            hidden = draft_model(self._input_ids, self._positions)
        hidden = hidden.reshape(1, self.num_tokens, -1)
        base_logits = draft_model.compute_base_logits(hidden)

        # Keep the official DSpark recurrence in the captured region.  This
        # is intentionally the greedy path only; temperature/top-p sampling
        # requires its per-request RNG and remains on the eager fallback.
        previous = self._input_ids[:, 0]
        for step in range(self.num_tokens):
            markov_embedding = draft_model.markov_head.markov_w1(previous)
            logits = base_logits[:, step, :] + draft_model.markov_head.markov_w2(markov_embedding)
            previous = logits.argmax(dim=-1)
            self._tokens[:, step].copy_(previous)
        return self._tokens

    def capture(self) -> None:
        if self._captured:
            return

        self._init_workspace()
        self._patch_impls_for_capture()
        self._attn_metadata = {
            name: self._flash_metadata if self.engine.use_flashinfer_draft else self._metadata
            for name in self.engine._draft_layer_names
        }
        self._slot_mappings = {name: self._slot_mapping for name in self.engine._draft_layer_names}
        capture_kv = min(2048, max(0, self.max_seq_len - self.num_tokens))
        try:
            self._fill_buffers(1, capture_kv)
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                for _ in range(3):
                    self._fill_buffers(1, capture_kv)
                    self._forward()
            side_stream.synchronize()

            self._fill_buffers(1, capture_kv)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._graph_pool):
                self._forward()
            self._graph = graph
            self._graph_pool = graph.pool()
            self.engine._draft_graph_pool = self._graph_pool  # noqa: SLF001
            self._captured = True
            logger.info(
                "Captured Qwen3.8 DSpark draft CUDA Graph: slot=%d K=%d",
                self.slot,
                self.num_tokens,
            )
        finally:
            self._restore_impls()

    def replay(self, anchor_token: int, kv_len: int) -> torch.Tensor:
        """Replay and return the graph-owned ``[K]`` token row.

        Returning the device view is deliberate.  The next target verify
        graph consumes this row directly; converting it to a Python list here
        would force a stream synchronization once per speculative round.
        The caller copies the row into its persistent verify-input buffer
        before this graph is replayed again.
        """

        if self._graph is None:
            raise RuntimeError("DSpark draft graph replay requested before capture")
        self._fill_buffers(anchor_token, kv_len)
        self._graph.replay()
        return self._tokens[0]


class Qwen36DSparkDraftBatchCudaGraph:
    """SGLang-shaped ``B*K`` masked draft CUDA Graph.

    The legacy class above is retained as a compatibility fallback for
    callers that explicitly construct a slot graph.  Production DSpark uses
    this batch-bucket graph: every layer sees one flattened ``B*K`` forward,
    one non-causal paged-attention launch, and one Markov recurrence over the
    B rows.  Slot ids and context lengths are replay metadata, never graph
    captured addresses.
    """

    def __init__(self, engine: Qwen36DSparkEngine, batch_size: int) -> None:
        self.engine = engine
        self.batch_size = int(batch_size)
        self.device = engine.device
        self.num_tokens = engine.k
        self.page_size = engine.page_size
        self.pages_per_slot = engine.pages_per_slot
        self.max_seq_len = engine.max_seq_len
        if not 1 <= self.batch_size <= engine.backend.num_slots:
            raise ValueError(f"invalid DSpark draft graph batch size: {batch_size}")

        self._input_ids = torch.full(
            (self.batch_size, self.num_tokens),
            engine.draft_model.config.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        # Replay inputs are persistent tensors.  Constructing CUDA tensors
        # from Python lists inside every speculative round creates a small
        # allocator/synchronization tax which is especially visible after
        # the target verify has been fused across B requests.
        self._anchor_host = torch.empty(
            self.batch_size, dtype=torch.long, device="cpu", pin_memory=True
        )
        self._kv_len_host = torch.empty(
            self.batch_size, dtype=torch.long, device="cpu", pin_memory=True
        )
        self._slot_host = torch.empty(
            self.batch_size, dtype=torch.long, device="cpu", pin_memory=True
        )
        self._anchors = torch.empty(self.batch_size, dtype=torch.long, device=self.device)
        self._kv_lens = torch.empty(self.batch_size, dtype=torch.long, device=self.device)
        self._slot_ids = torch.empty(self.batch_size, dtype=torch.long, device=self.device)
        self._slot_bases = torch.empty(self.batch_size, dtype=torch.long, device=self.device)
        self._slot_page_tables = torch.arange(
            engine.backend.num_slots * self.pages_per_slot,
            dtype=torch.int32,
            device=self.device,
        ).view(engine.backend.num_slots, self.pages_per_slot)
        self._positions = torch.zeros(
            self.batch_size * self.num_tokens, dtype=torch.long, device=self.device
        )
        self._page_table = torch.zeros(
            self.batch_size, self.pages_per_slot, dtype=torch.int32, device=self.device
        )
        self._cache_seqlens = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        self._cu_seqlens_q = (
            torch.arange(self.batch_size + 1, dtype=torch.int32, device=self.device)
            * self.num_tokens
        )
        self._slot_mapping = torch.zeros(
            self.batch_size * self.num_tokens, dtype=torch.long, device=self.device
        )
        self._pos_offset = torch.arange(self.num_tokens, dtype=torch.long, device=self.device)
        self._tokens = torch.zeros(
            self.batch_size, self.num_tokens, dtype=torch.long, device=self.device
        )
        self._confidence = (
            torch.zeros(
                self.batch_size,
                self.num_tokens,
                dtype=torch.float32,
                device=self.device,
            )
            if (
                self.engine.capture_confidence
                and self.engine.draft_model.confidence_head is not None
            )
            else None
        )

        self._workspace: Any | None = None
        self._metadata: Any | None = None
        self._flashinfer_impl: FlashInferDSparkAttentionImpl | None = None
        self._flash_metadata: SparkinferAttnMetadata | None = None
        self._attn_metadata: dict[str, Any] | None = None
        self._slot_mappings: dict[str, torch.Tensor] | None = None
        self._original_impls: dict[str, Any] = {}
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_pool: object | None = getattr(engine, "_draft_graph_pool", None)
        self._captured = False

    def _init_workspace(self) -> None:
        if self.engine.use_flashinfer_draft:
            first_attn = next(iter(self.engine._draft_attn_layers.values()))
            self._flashinfer_impl = FlashInferDSparkAttentionImpl(
                num_heads=first_attn.num_heads,
                head_size=first_attn.head_size,
                scale=first_attn.scale,
                num_kv_heads=first_attn.num_kv_heads,
                page_size=self.page_size,
                max_pages=self.pages_per_slot,
                num_tokens=self.num_tokens,
                batch_size=self.batch_size,
                device=self.device,
                use_cuda_graph=True,
                workspace_buffer=self.engine._flashinfer_workspace_buffer,
            )
            page_table = torch.arange(
                self.batch_size * self.pages_per_slot,
                dtype=torch.int32,
                device=self.device,
            ).view(self.batch_size, self.pages_per_slot)
            self._flash_metadata = SparkinferAttnMetadata(
                mode="extend",
                page_table=page_table,
                cache_seqlens=self._cache_seqlens,
                cu_seqlens_q=self._cu_seqlens_q,
                num_actual_tokens=self.batch_size * self.num_tokens,
                window_left=-1,
            )
            return

        from b12x.attention.paged.planner import create_paged_plan
        from b12x.attention.paged.workspace import PagedAttentionWorkspace

        first_attn = next(iter(self.engine._draft_attn_layers.values()))
        workspace = PagedAttentionWorkspace.for_contract(
            mode="extend",
            device=self.device,
            dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
            num_q_heads=first_attn.num_heads,
            num_kv_heads=first_attn.num_kv_heads,
            head_dim_qk=first_attn.head_size,
            head_dim_vo=first_attn.head_size,
            page_size=self.page_size,
            max_total_q=self.batch_size * self.num_tokens,
            num_cache_pages=self.pages_per_slot,
            use_cuda_graph=True,
        )
        assert workspace._plan_q is not None
        assert workspace._plan_k_cache is not None
        assert workspace._plan_v_cache is not None
        max_kv = self.pages_per_slot * self.page_size - 1
        capture_page_table = torch.arange(
            self.batch_size * self.pages_per_slot,
            dtype=torch.int32,
            device=self.device,
        ).view(self.batch_size, self.pages_per_slot)
        capture_cache_seqlens = torch.full(
            (self.batch_size,), max_kv, dtype=torch.int32, device=self.device
        )
        plan = create_paged_plan(
            workspace._plan_q,
            workspace._plan_k_cache,
            workspace._plan_v_cache,
            capture_page_table,
            capture_cache_seqlens,
            self._cu_seqlens_q,
            mode="extend",
            enable_cuda_graph=True,
            window_left=-1,
        )
        workspace._ensure_capacity(plan)
        workspace._copy_runtime_metadata(
            capture_page_table, capture_cache_seqlens, self._cu_seqlens_q
        )
        workspace._copy_plan_metadata(plan)
        workspace._plan = plan
        self._workspace = workspace
        self._metadata = _SparkinferCGExtendMetadata(workspace, self.batch_size * self.num_tokens)

    def _patch_impls_for_capture(self) -> None:
        if self.engine.use_flashinfer_draft:
            assert self._flashinfer_impl is not None
            for name, attn in self.engine._draft_attn_layers.items():
                self._original_impls[name] = attn.impl
                attn.impl = self._flashinfer_impl
            return
        assert self._workspace is not None
        for name, attn in self.engine._draft_attn_layers.items():
            self._original_impls[name] = attn.impl
            attn.impl = _SparkinferCGExtendImpl(
                self._workspace,
                self.batch_size * self.num_tokens,
                batch_size=self.batch_size,
            )

    def _restore_impls(self) -> None:
        for name, impl in self._original_impls.items():
            self.engine._draft_attn_layers[name].impl = impl

    def _fill_buffers(self, slots: list[int], anchors: list[int], kv_lens: list[int]) -> None:
        if len(slots) != self.batch_size or len(anchors) != self.batch_size:
            raise ValueError(
                f"DSpark draft B={self.batch_size} replay expects matching slots/anchors"
            )
        if len(kv_lens) != self.batch_size:
            raise ValueError(f"DSpark draft B={self.batch_size} replay expects kv_lens")
        if any(slot < 0 or slot >= self.engine.backend.num_slots for slot in slots):
            raise ValueError(f"invalid DSpark draft graph slots: {slots}")
        if any(kv_len < 0 or kv_len + self.num_tokens > self.max_seq_len for kv_len in kv_lens):
            raise RuntimeError(
                f"DSpark draft graph block does not fit: kv_lens={kv_lens}, "
                f"gamma={self.num_tokens}, max_seq_len={self.max_seq_len}"
            )

        self._input_ids.fill_(self.engine.draft_model.config.mask_token_id)
        for row, (slot, anchor, kv_len) in enumerate(zip(slots, anchors, kv_lens, strict=True)):
            self._slot_host[row] = slot
            self._anchor_host[row] = anchor
            self._kv_len_host[row] = kv_len
        self._anchors.copy_(self._anchor_host, non_blocking=True)
        self._kv_lens.copy_(self._kv_len_host, non_blocking=True)
        self._slot_ids.copy_(self._slot_host, non_blocking=True)
        self._input_ids[:, 0].copy_(self._anchors)
        torch.add(
            self._kv_lens[:, None],
            self._pos_offset[None, :],
            out=self._positions.view(self.batch_size, self.num_tokens),
        )
        for row, slot in enumerate(slots):
            # DSpark owns a separate contiguous KV allocation.  The target
            # pool's dynamic page table is not a valid address map here: its
            # physical rows may be reused or non-contiguous, while this
            # draft cache is always laid out as slot*pages_per_slot.
            self._page_table[row].copy_(self._slot_page_tables[slot])
        torch.add(
            self._kv_lens,
            self.num_tokens,
            out=self._cache_seqlens,
        )
        self._slot_bases.copy_(self._slot_ids)
        self._slot_bases.mul_(self.pages_per_slot * self.page_size)
        torch.add(
            self._slot_bases[:, None],
            self._positions.view(self.batch_size, -1),
            out=self._slot_mapping.view(self.batch_size, self.num_tokens),
        )
        if self.engine.use_flashinfer_draft:
            assert self._flashinfer_impl is not None
            self._flashinfer_impl.update_graph_metadata(kv_lens, page_table=self._page_table)
            return
        assert self._workspace is not None
        self._workspace.update_prefill_graph_replay_metadata(
            self._page_table,
            self._cache_seqlens,
            self._cu_seqlens_q,
            window_left=-1,
        )

    def _forward(self) -> torch.Tensor:
        assert self._metadata is not None or self._flash_metadata is not None
        assert self._attn_metadata is not None and self._slot_mappings is not None
        draft_model = self.engine.draft_model
        with bf_attn_context(self._attn_metadata, self._slot_mappings):
            hidden = draft_model(self._input_ids, self._positions)
        hidden = hidden.reshape(self.batch_size, self.num_tokens, -1)
        base_logits = draft_model.compute_base_logits(hidden)
        previous = self._input_ids[:, 0]
        for step in range(self.num_tokens):
            markov_embedding = draft_model.markov_head.markov_w1(previous)
            logits = base_logits[:, step, :] + draft_model.markov_head.markov_w2(markov_embedding)
            if self._confidence is not None:
                self._confidence[:, step].copy_(
                    draft_model.confidence_head.apply_sts(
                        draft_model.confidence_head(
                            hidden[:, step, :],
                            markov_embedding if draft_model.confidence_head.with_markov else None,
                        ),
                        position=step,
                    )
                )
            previous = logits.argmax(dim=-1)
            self._tokens[:, step].copy_(previous)
        return self._tokens

    def capture(self) -> None:
        if self._captured:
            return
        self._init_workspace()
        self._patch_impls_for_capture()
        self._attn_metadata = {
            name: self._flash_metadata if self.engine.use_flashinfer_draft else self._metadata
            for name in self.engine._draft_layer_names
        }
        self._slot_mappings = {name: self._slot_mapping for name in self.engine._draft_layer_names}
        capture_kv = min(2048, max(0, self.max_seq_len - self.num_tokens))
        slots = list(range(self.batch_size))
        anchors = [1] * self.batch_size
        kv_lens = [capture_kv] * self.batch_size
        try:
            self._fill_buffers(slots, anchors, kv_lens)
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                for _ in range(3):
                    self._fill_buffers(slots, anchors, kv_lens)
                    self._forward()
            side_stream.synchronize()
            self._fill_buffers(slots, anchors, kv_lens)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._graph_pool):
                self._forward()
            self._graph = graph
            self._graph_pool = graph.pool()
            self.engine._draft_graph_pool = self._graph_pool  # noqa: SLF001
            self._captured = True
            logger.info(
                "Captured Qwen3.8 DSpark draft BxK CUDA Graph: batch=%d K=%d",
                self.batch_size,
                self.num_tokens,
            )
        finally:
            self._restore_impls()

    def replay(self, slots: list[int], anchors: list[int], kv_lens: list[int]) -> torch.Tensor:
        if self._graph is None:
            raise RuntimeError("DSpark draft batch graph replay requested before capture")
        self._fill_buffers(slots, anchors, kv_lens)
        self._graph.replay()
        return self._tokens

    @property
    def confidence(self) -> torch.Tensor | None:
        """Graph-owned ``[B, K]`` acceptance probabilities, if available."""

        return self._confidence
