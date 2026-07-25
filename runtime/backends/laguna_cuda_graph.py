"""Laguna CUDA Graph decode — SparkInfer paged attention with on-device metadata.

SparkInfer's decode graph mode captures the metadata rebuild kernel IN the graph.
Per-step cost: GPU int32 writes (cache_seqlens + page_table on boundary) + replay.
No CPU plan, no H2D copies, no Python dispatch per step.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from runtime.backends.laguna import LagunaBackend

logger = logging.getLogger("qwen_sm120_runtime.laguna_cuda_graph")

def _physical_slot(slot: int) -> int:
    return slot + 1  # RESERVED_PHYSICAL_SLOTS = 1


class SparkinferDecodeCGMetadata:
    """Metadata for CG decode: references pre-created sparkinfer decode workspace."""
    __slots__ = ("workspace", "num_actual_tokens", "window_left")

    def __init__(self, workspace, num_actual_tokens: int, window_left: int = -1):
        self.workspace = workspace
        self.num_actual_tokens = num_actual_tokens
        self.window_left = window_left


class LagunaCudaGraphDecode:
    """CUDA-graph-captured decode using SparkInfer paged attention.

    Per (batch_size) instance. Captures the full decode forward with sparkinfer
    decode workspaces. Per-step: update cache_seqlens + page_table, replay.
    """

    def __init__(self, backend: LagunaBackend, batch_size: int) -> None:
        self.backend = backend
        self.batch_size = batch_size
        self.device = backend.device
        self.block_size = backend.block_size
        self.blocks_per_slot = backend.blocks_per_slot
        self.max_kv_len = backend.blocks_per_slot * backend.block_size

        self._graph: torch.cuda.CUDAGraph | None = None
        self._captured = False
        self._logits: torch.Tensor | None = None

        # ── Pre-allocated input buffers (fixed address) ──
        self._input_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._positions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._slot_mapping = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # ── Per-group sparkinfer decode workspaces ──
        self._decode_workspaces: dict[tuple, Any] = {}
        self._cg_metadata: dict[tuple, SparkinferDecodeCGMetadata] = {}

        # ── Page table buffers (one per group) ──
        self._page_tables: dict[tuple, torch.Tensor] = {}
        self._cache_seqlens: dict[tuple, torch.Tensor] = {}

        # ── SWA ring state ──
        rbps = backend._ring_blocks_per_slot
        self._ring_blocks_per_slot = rbps
        self._ring_slots_per_slot = rbps * self.block_size
        self._swa_window = backend._swa_window
        self._swa_slot_mapping = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # ── Per-slot page-crossing tracker ──
        self._prev_n_blocks: list[int] = [0] * batch_size
        self._swa_prev_n_blocks: list[int] = [0] * batch_size

        # ── DFlash aux hidden states (captured in graph) ──
        self._aux_hidden_states: list[torch.Tensor] | None = None

    def _init_workspaces(self) -> None:
        """Create sparkinfer decode workspaces per layer group."""
        from runtime.backends.laguna_sparkinfer_attn import SparkinferDecodeWorkspace

        backend = self.backend
        bs = self.batch_size

        for group_key, layer_names in backend._layer_groups.items():
            wl, nqh, nkvh = group_key
            is_swa = wl >= 0 and self._ring_blocks_per_slot > 0

            if is_swa:
                max_pages = self._ring_blocks_per_slot
            else:
                max_pages = self.blocks_per_slot

            ws = SparkinferDecodeWorkspace(
                num_q_heads=nqh, num_kv_heads=nkvh, head_dim=128,
                max_pages=max_pages, window_left=wl, device=str(self.device))

            self._decode_workspaces[group_key] = ws
            self._page_tables[group_key] = ws.page_table
            self._cache_seqlens[group_key] = ws.cache_seqlens
            self._cg_metadata[group_key] = SparkinferDecodeCGMetadata(
                workspace=ws, num_actual_tokens=bs, window_left=wl)

            logger.info("CG decode workspace: wl=%d qh=%d kvh=%d max_pages=%d",
                        wl, nqh, nkvh, max_pages)

    def _bind_kv_caches(self) -> None:
        """Bind real KV cache tensors to workspaces (before capture)."""
        backend = self.backend
        sfc = backend.static_forward_context

        for group_key, ws in self._decode_workspaces.items():
            wl = group_key[0]
            is_swa = wl >= 0 and self._ring_blocks_per_slot > 0
            # Use first layer in group to get KV cache
            first_name = backend._layer_groups[group_key][0]
            layer = sfc[first_name]
            kv_cache = layer.kv_cache
            # kv_cache: [num_blocks, 2, block_size, num_kv_heads, head_dim]
            key_cache, value_cache = kv_cache.unbind(1)
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(torch.float8_e4m3fn)
                value_cache = value_cache.view(torch.float8_e4m3fn)

            # Q and output buffers (will be set during capture from model internals)
            # For now, use the workspace's dummy tensors — they'll be rebound
            # during the actual capture when we know the real Q/output addresses.

    def _fill_buffers(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
    ) -> None:
        """Update pre-allocated buffers for replay. Optimized for batch=1."""
        bs = len(slot_ids)
        ps = self.block_size
        bps = self.blocks_per_slot

        if bs == 1:
            self._fill_buffers_b1(slot_ids[0], token_ids[0], kv_lengths[0])
            return

        # Generic batch path (batch > 1)
        for i in range(bs):
            self._input_ids[i] = token_ids[i]
            self._positions[i] = kv_lengths[i]

        for group_key, ws in self._decode_workspaces.items():
            wl = group_key[0]
            is_swa = wl >= 0 and self._ring_blocks_per_slot > 0

            if not is_swa:
                for i in range(bs):
                    phys = slot_ids[i] + 1
                    base = phys * bps
                    new_kv = kv_lengths[i] + 1
                    n_blocks = (new_kv + ps - 1) // ps
                    if n_blocks != self._prev_n_blocks[i]:
                        self._prev_n_blocks[i] = n_blocks
                        pt = self._page_tables[group_key]
                        pt[0, :n_blocks] = torch.arange(
                            base, base + n_blocks, dtype=torch.int32, device=self.device)
                    self._cache_seqlens[group_key][i] = new_kv
                    pos = kv_lengths[i]
                    self._slot_mapping[i] = (base + pos // ps) * ps + pos % ps
            else:
                rbps = self._ring_blocks_per_slot
                ring_slots = self._ring_slots_per_slot
                window = self._swa_window
                for i in range(bs):
                    phys = slot_ids[i] + 1
                    ring_base = phys * rbps
                    pos = kv_lengths[i]
                    new_kv = pos + 1
                    window_start = max(0, pos - window + 1)
                    aligned_start = (window_start // ps) * ps
                    aligned_len = new_kv - aligned_start
                    n_ring = (aligned_len + ps - 1) // ps
                    if n_ring != self._swa_prev_n_blocks[i]:
                        self._swa_prev_n_blocks[i] = n_ring
                        pt = self._page_tables[group_key]
                        for j in range(n_ring):
                            actual = aligned_start + j * ps
                            rb = (actual % ring_slots) // ps
                            pt[i, j] = ring_base + rb
                    self._cache_seqlens[group_key][i] = aligned_len
                    rb_dec = (pos % ring_slots) // ps
                    ro_dec = pos % ps
                    self._swa_slot_mapping[i] = (ring_base + rb_dec) * ps + ro_dec

    def _fill_buffers_b1(self, slot: int, token_id: int, kv_len: int) -> None:
        """Optimized fill for batch=1: minimal Python, pre-cached constants."""
        ps = self.block_size
        bps = self.blocks_per_slot
        phys = slot + 1
        base = phys * bps
        new_kv = kv_len + 1

        # Scalar writes (each is a tiny CUDA memcpy, ~2us each)
        self._input_ids[0] = token_id
        self._positions[0] = kv_len

        for group_key in self._decode_workspaces:
            wl = group_key[0]
            is_swa = wl >= 0 and self._ring_blocks_per_slot > 0

            if not is_swa:
                # Full attention: page_table only changes on block boundary
                n_blocks = (new_kv + ps - 1) // ps
                if n_blocks != self._prev_n_blocks[0]:
                    self._prev_n_blocks[0] = n_blocks
                    self._page_tables[group_key][0, :n_blocks] = torch.arange(
                        base, base + n_blocks, dtype=torch.int32, device=self.device)
                self._cache_seqlens[group_key][0] = new_kv
                self._slot_mapping[0] = base * ps + kv_len
            else:
                # SWA ring buffer
                rbps = self._ring_blocks_per_slot
                ring_slots = self._ring_slots_per_slot
                ring_base = phys * rbps
                window = self._swa_window

                window_start = max(0, kv_len - window + 1)
                aligned_start = (window_start // ps) * ps
                aligned_len = new_kv - aligned_start
                n_ring = (aligned_len + ps - 1) // ps

                if n_ring != self._swa_prev_n_blocks[0]:
                    self._swa_prev_n_blocks[0] = n_ring
                    pt = self._page_tables[group_key]
                    for j in range(n_ring):
                        actual = aligned_start + j * ps
                        rb = (actual % ring_slots) // ps
                        pt[0, j] = ring_base + rb

                self._cache_seqlens[group_key][0] = aligned_len
                rb_dec = (kv_len % ring_slots) // ps
                ro_dec = kv_len % ps
                self._swa_slot_mapping[0] = (ring_base + rb_dec) * ps + ro_dec

    def _build_metadata_and_forward(self) -> torch.Tensor:
        """Build sparkinfer CG metadata and run forward."""
        from runtime.compat_vllm import set_current_vllm_config, set_forward_context

        backend = self.backend
        bs = self.batch_size

        attn_metadata_dict: dict[str, Any] = {}
        slot_mapping_dict: dict[str, torch.Tensor] = {}

        for group_key, layer_names in backend._layer_groups.items():
            wl = group_key[0]
            is_swa = wl >= 0 and self._ring_blocks_per_slot > 0
            meta = self._cg_metadata[group_key]
            sm = self._swa_slot_mapping[:bs] if is_swa else self._slot_mapping[:bs]
            for name in layer_names:
                attn_metadata_dict[name] = meta
                slot_mapping_dict[name] = sm

        with set_current_vllm_config(backend.vllm_config):
            with set_forward_context(
                attn_metadata_dict, backend.vllm_config, slot_mapping=slot_mapping_dict,
                skip_compiled=True,
            ):
                result = backend.model.forward(
                    self._input_ids[:bs], self._positions[:bs]
                )

        if isinstance(result, tuple):
            hidden_states, self._aux_hidden_states = result
        else:
            hidden_states = result
            self._aux_hidden_states = None
        return backend.model.compute_logits(hidden_states)

    def capture(self) -> None:
        """Warmup → capture the decode forward with sparkinfer workspaces."""
        if self._captured:
            return

        backend = self.backend
        bs = self.batch_size

        # Reserve warmup slots (last bs slots)
        warmup_slots = list(range(backend.num_slots - bs, backend.num_slots))
        dummy_tokens = [1] * bs
        capture_kv = self.blocks_per_slot * self.block_size - 1
        dummy_kv_lens = [capture_kv] * bs

        logger.info("Capturing Laguna CUDA Graph (sparkinfer): batch_size=%d", bs)

        # Initialize sparkinfer workspaces
        self._init_workspaces()

        # Patch attention impls to use CG decode workspaces
        self._patch_impls_for_cg()

        # Fill buffers with capture-time data
        self._fill_buffers(warmup_slots, dummy_tokens, dummy_kv_lens)

        # Warmup (3x)
        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                self._fill_buffers(warmup_slots, dummy_tokens, dummy_kv_lens)
                self._build_metadata_and_forward()
        side_stream.synchronize()

        # Final fill before capture
        self._fill_buffers(warmup_slots, dummy_tokens, dummy_kv_lens)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._logits = self._build_metadata_and_forward()
            self._input_ids[0] = self._logits[0].argmax(dim=-1).to(torch.long)

        self._graph = graph
        self._captured = True
        logger.info("Laguna CUDA Graph captured (sparkinfer): batch_size=%d", bs)

    def _patch_impls_for_cg(self) -> None:
        """Patch attention impls to dispatch to sparkinfer CG workspaces."""
        backend = self.backend
        sfc = backend.static_forward_context

        # Save original impls for prefill restore
        if not hasattr(self, '_original_impls'):
            self._original_impls = {}
            for group_key, layer_names in backend._layer_groups.items():
                for name in layer_names:
                    self._original_impls[name] = sfc[name].impl

        for group_key, layer_names in backend._layer_groups.items():
            meta = self._cg_metadata[group_key]
            ws = self._decode_workspaces[group_key]
            for name in layer_names:
                layer = sfc[name]
                layer.impl = _SparkinferCGDecodeImpl(ws, meta)

    def unpatch_impls(self) -> None:
        """Restore original attention impls (for prefill after CG capture)."""
        if not hasattr(self, '_original_impls'):
            return
        sfc = self.backend.static_forward_context
        for name, impl in self._original_impls.items():
            sfc[name].impl = impl

    def repatch_impls(self) -> None:
        """Re-apply CG decode impls (after prefill, before decode replay)."""
        self._patch_impls_for_cg()

    def replay(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
    ) -> list[int]:
        """Replay the captured graph with updated metadata."""
        if not self._captured:
            raise RuntimeError("Graph not captured. Call capture() first.")

        self._fill_buffers(slot_ids, token_ids, kv_lengths)
        self._graph.replay()

        bs = len(slot_ids)
        if bs == 1:
            return [int(self._input_ids[0].item())]
        return [int(self._input_ids[i].item()) for i in range(bs)]

    def replay_with_aux(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
    ) -> tuple[list[int], list[torch.Tensor] | None]:
        """Replay graph and return (next_tokens, aux_hidden_states)."""
        next_tokens = self.replay(slot_ids, token_ids, kv_lengths)
        return next_tokens, self._aux_hidden_states

    def generate(
        self,
        slot: int,
        first_token: int,
        kv_len: int,
        max_new_tokens: int = 128,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """Generate tokens using CG replay loop."""
        tokens = []
        tok = first_token
        current_kv = kv_len
        for _ in range(max_new_tokens):
            results = self.replay([slot], [tok], [current_kv])
            tok = results[0]
            tokens.append(tok)
            current_kv += 1
            if eos_token_id is not None and tok == eos_token_id:
                break
        return tokens


class _SparkinferCGDecodeImpl:
    """Attention impl that dispatches to a pre-created sparkinfer CG workspace.

    Used during CUDA graph capture/replay. The workspace's forward() is
    captured in the graph. Per-step metadata updates (cache_seqlens, page_table)
    are GPU writes that happen BEFORE graph.replay().
    """

    def __init__(self, workspace, metadata):
        self.workspace = workspace
        self.metadata = metadata
        self.num_heads = workspace.num_q_heads
        self.head_size = workspace.head_dim
        self.num_kv_heads = workspace.num_kv_heads
        self.scale = workspace.head_dim ** -0.5
        self.kv_cache_dtype = "fp8_e4m3"
        self.supports_quant_query_input = False

    def process_weights_after_loading(self, act_dtype):
        pass

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """Write K/V into paged cache."""
        k_cache = kv_cache[:, 0]
        v_cache = kv_cache[:, 1]
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key, value, k_cache, v_cache, slot_mapping,
            self.kv_cache_dtype, layer._k_scale, layer._v_scale,
        )

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        """Run sparkinfer decode attention (captured in CG)."""
        num_actual_tokens = attn_metadata.num_actual_tokens
        q = query[:num_actual_tokens]
        key_cache, value_cache = kv_cache.unbind(1)
        if key_cache.dtype == torch.uint8:
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)

        # Bind real tensors to workspace
        ws = self.workspace
        ws._q = q
        ws._k_cache = key_cache
        ws._v_cache = value_cache
        ws._output = output[:num_actual_tokens]

        # Run sparkinfer forward (captured in graph)
        ws.forward()
        return output


class LagunaCudaGraphVerify:
    """CUDA-graph-captured M=16 verify forward using SparkInfer extend attention.

    Captures the full model forward with 16 tokens (1 bonus + 15 draft).
    Per-step: update input_ids, positions, page_table, cache_seqlens, slot_mapping.
    Returns logits (16 × vocab) + aux hidden states.
    """

    def __init__(self, backend: "LagunaBackend", num_tokens: int = 16) -> None:
        self.backend = backend
        self.num_tokens = num_tokens
        self.device = backend.device
        self.block_size = backend.block_size
        self.blocks_per_slot = backend.blocks_per_slot
        self.max_kv_len = backend.blocks_per_slot * backend.block_size

        self._graph: torch.cuda.CUDAGraph | None = None
        self._captured = False
        self._logits: torch.Tensor | None = None
        self._aux_hidden_states: list[torch.Tensor] | None = None

        # Pre-allocated input buffers (fixed address for CG)
        self._input_ids = torch.zeros(num_tokens, dtype=torch.long, device=self.device)
        self._positions = torch.zeros(num_tokens, dtype=torch.long, device=self.device)
        self._slot_mapping = torch.zeros(num_tokens, dtype=torch.long, device=self.device)

        # Per-group extend workspaces
        self._extend_workspaces: dict[tuple, Any] = {}
        self._page_tables: dict[tuple, torch.Tensor] = {}
        self._cache_seqlens: dict[tuple, torch.Tensor] = {}
        self._cu_seqlens_q = torch.tensor([0, num_tokens], dtype=torch.int32, device=self.device)

        # SWA ring state
        rbps = backend._ring_blocks_per_slot
        self._ring_blocks_per_slot = rbps
        self._ring_slots_per_slot = rbps * self.block_size
        self._swa_window = backend._swa_window
        self._swa_slot_mapping = torch.zeros(num_tokens, dtype=torch.long, device=self.device)

    def _init_workspaces(self) -> None:
        """Create sparkinfer extend workspaces per layer group (CG mode)."""
        from sparkinfer.attention.paged.workspace import PagedAttentionWorkspace
        from sparkinfer.attention.paged.planner import create_paged_plan

        backend = self.backend
        nt = self.num_tokens

        for group_key, layer_names in backend._layer_groups.items():
            wl, nqh, nkvh = group_key
            is_swa = wl >= 0 and self._ring_blocks_per_slot > 0

            if is_swa:
                max_pages = self._ring_blocks_per_slot
            else:
                max_pages = self.blocks_per_slot

            # Dummy tensors for workspace creation
            q = torch.zeros(nt, nqh, 128, dtype=torch.bfloat16, device=self.device)
            k_cache = torch.zeros(max_pages, self.block_size, nkvh, 128,
                                  dtype=torch.float8_e4m3fn, device=self.device)
            v_cache = torch.zeros(max_pages, self.block_size, nkvh, 128,
                                  dtype=torch.float8_e4m3fn, device=self.device)

            ws = PagedAttentionWorkspace.for_tensors(
                mode="extend", q=q, k_cache=k_cache, v_cache=v_cache,
                use_cuda_graph=True)

            # Create CG-compatible plan at max context
            max_kv = max_pages * self.block_size - 1
            page_table = torch.arange(max_pages, dtype=torch.int32, device=self.device).unsqueeze(0)
            cache_seqlens = torch.tensor([max_kv], dtype=torch.int32, device=self.device)

            plan = create_paged_plan(
                q, k_cache, v_cache, page_table, cache_seqlens, self._cu_seqlens_q,
                mode="extend", enable_cuda_graph=True, window_left=wl)
            ws._ensure_capacity(plan)
            ws._copy_runtime_metadata(page_table, cache_seqlens, self._cu_seqlens_q)
            ws._copy_plan_metadata(plan)
            ws._plan = plan

            self._extend_workspaces[group_key] = ws
            self._page_tables[group_key] = page_table
            self._cache_seqlens[group_key] = cache_seqlens

            logger.info("CG verify workspace: wl=%d qh=%d kvh=%d max_pages=%d",
                        wl, nqh, nkvh, max_pages)

    def _bind_kv_caches(self) -> None:
        """Bind real KV cache tensors to workspaces."""
        backend = self.backend
        sfc = backend.static_forward_context

        for group_key, ws in self._extend_workspaces.items():
            wl = group_key[0]
            is_swa = wl >= 0 and self._ring_blocks_per_slot > 0
            first_name = backend._layer_groups[group_key][0]
            layer = sfc[first_name]
            kv_cache = layer.kv_cache
            key_cache, value_cache = kv_cache.unbind(1)
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(torch.float8_e4m3fn)
                value_cache = value_cache.view(torch.float8_e4m3fn)
            ws._k_cache = key_cache
            ws._v_cache = value_cache

    def _patch_impls_for_cg(self) -> None:
        """Patch attention impls to dispatch to extend CG workspaces."""
        backend = self.backend
        sfc = backend.static_forward_context

        if not hasattr(self, '_original_impls'):
            self._original_impls = {}
            for group_key, layer_names in backend._layer_groups.items():
                for name in layer_names:
                    self._original_impls[name] = sfc[name].impl

        for group_key, layer_names in backend._layer_groups.items():
            ws = self._extend_workspaces[group_key]
            for name in layer_names:
                layer = sfc[name]
                layer.impl = _SparkinferCGExtendImpl(ws, self.num_tokens)

    def unpatch_impls(self) -> None:
        """Restore original attention impls."""
        if not hasattr(self, '_original_impls'):
            return
        sfc = self.backend.static_forward_context
        for name, impl in self._original_impls.items():
            sfc[name].impl = impl

    def repatch_impls(self) -> None:
        """Re-apply CG verify impls."""
        self._patch_impls_for_cg()

    def _fill_buffers(self, slot: int, token_ids: list[int], kv_len: int) -> None:
        """Fill input buffers for verify replay."""
        backend = self.backend
        nt = self.num_tokens
        bs = self.block_size
        phys = _physical_slot(slot)
        new_kv_len = kv_len + nt

        # Input IDs and positions
        for i, tok in enumerate(token_ids[:nt]):
            self._input_ids[i] = tok
        for i in range(nt):
            self._positions[i] = kv_len + i

        # Full-attention: page table and slot mapping
        full_base = phys * self.blocks_per_slot
        n_blocks_full = (new_kv_len + bs - 1) // bs
        for group_key, ws in self._extend_workspaces.items():
            wl = group_key[0]
            is_swa = wl >= 0 and self._ring_blocks_per_slot > 0

            if is_swa:
                # SWA ring: page table covers window-aligned blocks
                ring_base = phys * self._ring_blocks_per_slot
                ring_slots = self._ring_slots_per_slot
                window_start = max(0, kv_len - self._swa_window + 1)
                aligned_start = (window_start // bs) * bs
                aligned_len = new_kv_len - aligned_start
                n_ring = min(-(-aligned_len // bs), self._ring_blocks_per_slot)
                pt = self._page_tables[group_key]
                for j in range(n_ring):
                    actual_pos = aligned_start + j * bs
                    ring_block = (actual_pos % ring_slots) // bs
                    pt[0, j] = ring_base + ring_block
                self._cache_seqlens[group_key][0] = aligned_len
                # SWA slot mapping: ring-wrapped
                for j in range(nt):
                    pos = kv_len + j
                    ring_block = (pos % ring_slots) // bs
                    ring_off = pos % bs
                    self._swa_slot_mapping[j] = (ring_base + ring_block) * bs + ring_off
            else:
                # Full attention: sequential blocks
                pt = self._page_tables[group_key]
                for j in range(n_blocks_full):
                    pt[0, j] = full_base + j
                self._cache_seqlens[group_key][0] = new_kv_len

        # Full-attention slot mapping
        for j in range(nt):
            pos = kv_len + j
            self._slot_mapping[j] = (full_base + pos // bs) * bs + pos % bs

        # Copy runtime metadata to workspaces
        for group_key, ws in self._extend_workspaces.items():
            ws._copy_runtime_metadata(
                self._page_tables[group_key],
                self._cache_seqlens[group_key],
                self._cu_seqlens_q)

    def capture(self) -> None:
        """Warmup → capture the verify forward."""
        if self._captured:
            return

        backend = self.backend
        nt = self.num_tokens
        sfc = backend.static_forward_context

        logger.info("Capturing Laguna Verify CUDA Graph (M=%d, sparkinfer extend)", nt)

        self._init_workspaces()
        self._bind_kv_caches()
        self._patch_impls_for_cg()

        # Build metadata and slot mapping dicts (reused across warmup/capture)
        dummy_tokens = [1] * nt
        warmup_kv = 64
        self._fill_buffers(0, dummy_tokens, warmup_kv)

        slot_mapping_dict = {}
        for group_key, layer_names in backend._layer_groups.items():
            wl = group_key[0]
            is_swa = wl >= 0 and self._ring_blocks_per_slot > 0
            sm = self._swa_slot_mapping if is_swa else self._slot_mapping
            for name in layer_names:
                slot_mapping_dict[name] = sm

        attn_meta = {}
        for group_key, layer_names in backend._layer_groups.items():
            ws = self._extend_workspaces[group_key]
            meta = _SparkinferCGExtendMetadata(ws, nt)
            for name in layer_names:
                attn_meta[name] = meta

        from runtime.compat_vllm import set_forward_context

        # Warmup (3x)
        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                self._fill_buffers(0, dummy_tokens, warmup_kv)
                with set_forward_context(attn_meta, backend.vllm_config,
                                         slot_mapping=slot_mapping_dict, skip_compiled=True):
                    result = backend.model.forward(self._input_ids, self._positions)
                if isinstance(result, tuple):
                    hidden_states, aux = result
                else:
                    hidden_states, aux = result, None
                backend.model.compute_logits(hidden_states)
        side_stream.synchronize()

        # Final fill before capture
        self._fill_buffers(0, dummy_tokens, warmup_kv)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with set_forward_context(attn_meta, backend.vllm_config,
                                     slot_mapping=slot_mapping_dict, skip_compiled=True):
                result = backend.model.forward(self._input_ids, self._positions)
            if isinstance(result, tuple):
                hidden_states, self._aux_hidden_states = result
            else:
                hidden_states, self._aux_hidden_states = result, None
            self._logits = backend.model.compute_logits(hidden_states)

        self._graph = graph
        self._captured = True
        logger.info("Laguna Verify CUDA Graph captured (M=%d)", nt)

    def replay_with_aux(
        self, slot: int, token_ids: list[int], kv_len: int
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Replay verify graph. Returns (logits[16×vocab], aux_hidden_states)."""
        if not self._captured:
            raise RuntimeError("Verify CG not captured")
        self._fill_buffers(slot, token_ids, kv_len)
        self._graph.replay()
        return self._logits, self._aux_hidden_states


class _SparkinferCGExtendMetadata:
    """Lightweight metadata for CG extend impl."""
    __slots__ = ("workspace", "num_actual_tokens")
    def __init__(self, workspace, num_actual_tokens: int):
        self.workspace = workspace
        self.num_actual_tokens = num_actual_tokens


class _SparkinferCGExtendImpl:
    """Attention impl that dispatches to CG extend workspace.
    
    Creates the binding on first forward call (warmup), then reuses it
    for CG capture and replay. The binding references fixed-address tensors
    so it remains valid across replays.
    """

    def __init__(self, workspace, num_tokens: int):
        self._ws = workspace
        self._num_tokens = num_tokens
        self._binding = None
        self._descale = None
        self.kv_cache_dtype = "fp8_e4m3"
        self.supports_quant_query_input = False

    def process_weights_after_loading(self, act_dtype):
        pass

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        k_cache = kv_cache[:, 0]
        v_cache = kv_cache[:, 1]
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key, value, k_cache, v_cache, slot_mapping,
            self.kv_cache_dtype, layer._k_scale, layer._v_scale)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        if attn_metadata is None:
            return output.fill_(0)
        num_actual_tokens = attn_metadata.num_actual_tokens
        q = query[:num_actual_tokens]
        key_cache, value_cache = kv_cache.unbind(1)
        if key_cache.dtype == torch.uint8:
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
        out = output[:num_actual_tokens]

        if self._binding is None:
            from sparkinfer.attention.paged._scratch import build_paged_attention_binding
            self._descale = torch.ones(1, dtype=torch.float32, device=q.device)
            self._binding = build_paged_attention_binding(
                scratch=self._ws, q=q, k_cache=key_cache, v_cache=value_cache,
                output=out, k_descale=self._descale, v_descale=self._descale)

        from sparkinfer.attention.paged._forward import paged_attention_forward
        paged_attention_forward(binding=self._binding)
        return output
