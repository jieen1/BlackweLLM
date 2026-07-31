"""Laguna CUDA Graph decode — SparkInfer paged attention with on-device metadata.

SparkInfer's decode graph mode captures the metadata rebuild kernel IN the graph.
Per-step cost: GPU int32 writes (cache_seqlens + page_table on boundary) + replay.
No CPU plan, no H2D copies, no Python dispatch per step.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import torch

from bfdiag.invariants import checks as bfdiag_checks
from runtime.kernels.cg_decode_metadata import write_laguna_b1_decode_metadata
from runtime.kernels.fused_kv_scatter import fused_kv_scatter

if TYPE_CHECKING:
    from runtime.backends.laguna import LagunaBackend

from runtime.backends.bf_attention import bf_attn_context

logger = logging.getLogger("qwen_sm120_runtime.laguna_cuda_graph")


def _debug_swa_align_gran() -> int:
    """QSR_DEBUG_SWA_ALIGN_GRANULARITY: diagnostic-only override for the SWA
    ring's window_start rounding granularity (normally == block_size). Read
    per-call (not cached) so it can be toggled across `bf exec` calls in a
    resident daemon without a reload -- block_size itself stays a load-time
    constant, but this knob isolates "alignment slack" as a variable
    independent of it. 0/unset = disabled (use block_size, production
    behavior).
    """
    return int(os.environ.get("QSR_DEBUG_SWA_ALIGN_GRANULARITY", "0"))


def _physical_slot(slot: int) -> int:
    return slot  # RESERVED_PHYSICAL_SLOTS = 0


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

        # Default only for the exact Laguna B=1 layout (one full-attention
        # group plus one SWA group).  Set QSR_B1_METADATA_FASTPATH=0 to
        # restore scalar metadata writes while diagnosing a regression.
        self._b1_metadata_fastpath_enabled = os.environ.get("QSR_B1_METADATA_FASTPATH", "1") != "0"
        self._b1_metadata_values_cpu: torch.Tensor | None = None
        self._b1_metadata_values_gpu: torch.Tensor | None = None
        self._b1_full_group: tuple | None = None
        self._b1_swa_group: tuple | None = None

    def _init_b1_metadata_fastpath(self) -> bool:
        """Allocate the pinned staging pair for the fixed Laguna B=1 layout."""
        # A resident daemon can rebind this class onto a graph captured by an
        # older implementation; initialize newly introduced state lazily so
        # the hot-reload path never requires a model reload.
        if not hasattr(self, "_b1_metadata_values_gpu"):
            self._b1_metadata_fastpath_enabled = (
                os.environ.get("QSR_B1_METADATA_FASTPATH", "1") != "0"
            )
            self._b1_metadata_values_cpu = None
            self._b1_metadata_values_gpu = None
            self._b1_full_group = None
            self._b1_swa_group = None
        if not self._b1_metadata_fastpath_enabled:
            return False
        if self._b1_metadata_values_gpu is not None:
            return True

        full_groups = [
            key
            for key in self._decode_workspaces
            if not (key[0] >= 0 and self._ring_blocks_per_slot > 0)
        ]
        swa_groups = [
            key for key in self._decode_workspaces if key[0] >= 0 and self._ring_blocks_per_slot > 0
        ]
        if len(full_groups) != 1 or len(swa_groups) != 1:
            return False

        self._b1_full_group = full_groups[0]
        self._b1_swa_group = swa_groups[0]
        self._b1_metadata_values_cpu = torch.empty(2, dtype=torch.long, pin_memory=True)
        self._b1_metadata_values_gpu = torch.empty(2, dtype=torch.long, device=self.device)
        return True

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
                # CG capture capacity for full attention layers.
                # Default: capture at blocks_per_slot+16 (full capacity).
                # Override with QSR_VERIFY_CG_MAX_PAGES to capture at a
                # representative context (e.g. 1040 for 64K) — shorter
                # capture → smaller chunks → more useful CTAs at that
                # context length.  Contexts longer than capture capacity
                # still work (fewer useful chunks, graceful degradation).
                _cg_cap = int(os.environ.get("QSR_VERIFY_CG_MAX_PAGES", "0"))
                if _cg_cap > 0:
                    max_pages = _cg_cap
                else:
                    max_pages = self.blocks_per_slot + 16

            ws = SparkinferDecodeWorkspace(
                num_q_heads=nqh,
                num_kv_heads=nkvh,
                head_dim=128,
                max_pages=max_pages,
                window_left=wl,
                device=str(self.device),
                page_size=self.block_size,
            )

            self._decode_workspaces[group_key] = ws
            self._page_tables[group_key] = ws.page_table
            self._cache_seqlens[group_key] = ws.cache_seqlens
            self._cg_metadata[group_key] = SparkinferDecodeCGMetadata(
                workspace=ws, num_actual_tokens=bs, window_left=wl
            )

            logger.info(
                "CG decode workspace: wl=%d qh=%d kvh=%d max_pages=%d", wl, nqh, nkvh, max_pages
            )

    def _bind_kv_caches(self) -> None:
        """Bind real KV cache tensors to workspaces (before capture)."""
        backend = self.backend
        sfc = backend.static_forward_context

        for group_key, ws in self._decode_workspaces.items():
            # Use first layer in group to get KV cache
            first_name = backend._layer_groups[group_key][0]
            layer = sfc[first_name]
            kv_cache = layer.kv_cache
            # kv_cache: [2, num_blocks, block_size, num_kv_heads, head_dim]
            key_cache, value_cache = kv_cache.unbind(0)
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(torch.float8_e4m3fn)
                value_cache = value_cache.view(torch.float8_e4m3fn)

            # Replace the zero-stride planning views with real cache storage
            # before graph capture. Q and output are rebound per-layer during
            # capture from model internals.
            ws._k_cache = key_cache
            ws._v_cache = value_cache

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
                    phys = _physical_slot(slot_ids[i])
                    base = phys * bps
                    new_kv = kv_lengths[i] + 1
                    n_blocks = (new_kv + ps - 1) // ps
                    if n_blocks != self._prev_n_blocks[i]:
                        self._prev_n_blocks[i] = n_blocks
                        pt = self._page_tables[group_key]
                        pt[0, :n_blocks] = torch.arange(
                            base, base + n_blocks, dtype=torch.int32, device=self.device
                        )
                    self._cache_seqlens[group_key][i] = new_kv
                    pos = kv_lengths[i]
                    self._slot_mapping[i] = (base + pos // ps) * ps + pos % ps
            else:
                rbps = self._ring_blocks_per_slot
                ring_slots = self._ring_slots_per_slot
                window = self._swa_window
                for i in range(bs):
                    phys = _physical_slot(slot_ids[i])
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
        phys = _physical_slot(slot)
        base = phys * bps
        new_kv = kv_len + 1

        if self._init_b1_metadata_fastpath():
            assert self._b1_metadata_values_cpu is not None
            assert self._b1_metadata_values_gpu is not None
            assert self._b1_full_group is not None
            assert self._b1_swa_group is not None

            self._b1_metadata_values_cpu[0] = token_id
            self._b1_metadata_values_cpu[1] = kv_len
            self._b1_metadata_values_gpu.copy_(self._b1_metadata_values_cpu, non_blocking=True)
            write_laguna_b1_decode_metadata(
                self._b1_metadata_values_gpu,
                self._input_ids,
                self._positions,
                self._slot_mapping,
                self._swa_slot_mapping,
                self._cache_seqlens[self._b1_full_group],
                self._cache_seqlens[self._b1_swa_group],
                full_slot_base=base,
                swa_ring_base=phys * self._ring_blocks_per_slot,
                ring_slots=self._ring_slots_per_slot,
                swa_window=self._swa_window,
                block_size=ps,
            )
            self._update_b1_page_tables(slot, kv_len, new_kv)
            return

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
                        base, base + n_blocks, dtype=torch.int32, device=self.device
                    )
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

    def _update_b1_page_tables(self, slot: int, kv_len: int, new_kv: int) -> None:
        """Apply only the rare page-table writes omitted by the fused fast path."""
        assert self._b1_full_group is not None
        assert self._b1_swa_group is not None
        ps = self.block_size
        phys = _physical_slot(slot)

        n_blocks = (new_kv + ps - 1) // ps
        if n_blocks != self._prev_n_blocks[0]:
            self._prev_n_blocks[0] = n_blocks
            base = phys * self.blocks_per_slot
            self._page_tables[self._b1_full_group][0, :n_blocks] = torch.arange(
                base, base + n_blocks, dtype=torch.int32, device=self.device
            )

        window_start = max(0, kv_len - self._swa_window + 1)
        aligned_start = (window_start // ps) * ps
        aligned_len = new_kv - aligned_start
        n_ring = (aligned_len + ps - 1) // ps
        if n_ring != self._swa_prev_n_blocks[0]:
            self._swa_prev_n_blocks[0] = n_ring
            ring_base = phys * self._ring_blocks_per_slot
            ring_slots = self._ring_slots_per_slot
            page_table = self._page_tables[self._b1_swa_group]
            for index in range(n_ring):
                actual = aligned_start + index * ps
                ring_block = (actual % ring_slots) // ps
                page_table[0, index] = ring_base + ring_block

    def _build_metadata_and_forward(self) -> torch.Tensor:
        """Build sparkinfer CG metadata and run forward."""
        from runtime.laguna_runtime import laguna_forward_context

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

        with bf_attn_context(attn_metadata_dict, slot_mapping_dict):
            with laguna_forward_context(
                attn_metadata_dict,
                backend.runtime_config,
                slot_mapping=slot_mapping_dict,
                skip_compiled=True,
            ):
                result = backend.model.forward(self._input_ids[:bs], self._positions[:bs])

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

        # Restore original attention impls so the eager path keeps working.
        # CG replay uses the captured graph directly, not the impl attribute.
        self.unpatch_impls()

        logger.info("Laguna CUDA Graph captured (sparkinfer): batch_size=%d", bs)

    def _patch_impls_for_cg(self) -> None:
        """Patch attention impls to dispatch to sparkinfer CG workspaces."""
        backend = self.backend
        sfc = backend.static_forward_context

        # Save original impls for prefill restore
        if not hasattr(self, "_original_impls"):
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
        if not hasattr(self, "_original_impls"):
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

    def reset(self) -> None:
        """Forget per-slot page-table state before a fresh generation."""
        self._prev_n_blocks = [0] * self.batch_size
        self._swa_prev_n_blocks = [0] * self.batch_size

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
        self.scale = workspace.head_dim**-0.5
        self.kv_cache_dtype = "fp8_e4m3"
        self.supports_quant_query_input = False

    def process_weights_after_loading(self, act_dtype):
        pass

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        """Write K/V into paged cache via fused Triton kernel (CG-safe)."""
        k_cache = kv_cache[0].view(torch.float8_e4m3fn)
        v_cache = kv_cache[1].view(torch.float8_e4m3fn)
        fused_kv_scatter(key, value, k_cache, v_cache, slot_mapping, layer._k_scale, layer._v_scale)

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        """Run sparkinfer decode attention (captured in CG)."""
        num_actual_tokens = attn_metadata.num_actual_tokens
        q = query[:num_actual_tokens]
        key_cache, value_cache = kv_cache.unbind(0)
        if key_cache.dtype == torch.uint8:
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)

        # Bind real tensors to workspace
        ws = self.workspace
        ws._q = q
        ws._k_cache = key_cache
        ws._v_cache = value_cache
        ws._output = output[:num_actual_tokens]
        ws._k_descale = layer._k_scale.detach()
        ws._v_descale = layer._v_scale.detach()

        # Run sparkinfer forward (captured in graph)
        ws.forward()
        return output


class LagunaCudaGraphVerify:
    """CUDA-graph-captured M=16 verify forward using SparkInfer extend attention.

    Captures the full model forward with 16 tokens (1 bonus + 15 draft).
    Per-step: update input_ids, positions, page_table, cache_seqlens, slot_mapping.
    Returns logits (16 × vocab) + aux hidden states.

    The graph uses the non-split extend planner even though this is logically a
    verification forward.  Its KV length and SWA ring alignment change on each
    replay; the split verify planner captures a length-specific worklist, while
    the extend kernel reads the runtime metadata inside a fixed query worklist.
    """

    def __init__(self, backend: LagunaBackend, num_tokens: int = 16) -> None:
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

        # Per-group extend workspaces for the logical verify forward
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
        from sparkinfer.attention.paged.planner import create_paged_plan
        from sparkinfer.attention.paged.workspace import PagedAttentionWorkspace

        backend = self.backend
        nt = self.num_tokens

        for group_key, layer_names in backend._layer_groups.items():
            wl, nqh, nkvh = group_key
            is_swa = wl >= 0 and self._ring_blocks_per_slot > 0

            if is_swa:
                max_pages = self._ring_blocks_per_slot
            else:
                max_pages = self.blocks_per_slot + 16  # margin for generated tokens
                _cg_cap = int(os.environ.get("QSR_VERIFY_CG_MAX_PAGES", "0"))
                max_pages = _cg_cap if _cg_cap > 0 else self.blocks_per_slot + 16
            # Keep only shape-only K/V planning views until real cache
            # storage is bound before capture.
            ws = PagedAttentionWorkspace.for_contract(
                mode="verify",
                device=self.device,
                dtype=torch.bfloat16,
                kv_dtype=torch.float8_e4m3fn,
                num_q_heads=nqh,
                num_kv_heads=nkvh,
                head_dim_qk=128,
                head_dim_vo=128,
                page_size=self.block_size,
                max_total_q=nt,
                num_cache_pages=max_pages,
                use_cuda_graph=True,
            )
            assert ws._plan_q is not None
            assert ws._plan_k_cache is not None
            assert ws._plan_v_cache is not None
            q = ws._plan_q
            k_cache = ws._plan_k_cache
            v_cache = ws._plan_v_cache

            # Create CG-compatible plan at max context
            max_kv = max_pages * self.block_size - 1
            page_table = torch.arange(max_pages, dtype=torch.int32, device=self.device).unsqueeze(0)
            cache_seqlens = torch.tensor([max_kv], dtype=torch.int32, device=self.device)

            _ctas_per_sm = int(os.environ.get("QSR_VERIFY_CG_CTAS_PER_SM", "0")) or None
            plan = create_paged_plan(
                q,
                k_cache,
                v_cache,
                page_table,
                cache_seqlens,
                self._cu_seqlens_q,
                mode="verify",
                enable_cuda_graph=True,
                window_left=wl,
                graph_ctas_per_sm=_ctas_per_sm,
            )
            ws._ensure_capacity(plan)
            ws._copy_runtime_metadata(page_table, cache_seqlens, self._cu_seqlens_q)
            ws._copy_plan_metadata(plan)
            ws._plan = plan

            self._extend_workspaces[group_key] = ws
            self._page_tables[group_key] = page_table
            self._cache_seqlens[group_key] = cache_seqlens

            logger.info(
                "CG verify workspace: wl=%d qh=%d kvh=%d max_pages=%d", wl, nqh, nkvh, max_pages
            )

    def _bind_kv_caches(self) -> None:
        """Bind real KV cache tensors to workspaces."""
        backend = self.backend
        sfc = backend.static_forward_context

        for group_key, ws in self._extend_workspaces.items():
            first_name = backend._layer_groups[group_key][0]
            layer = sfc[first_name]
            kv_cache = layer.kv_cache
            key_cache, value_cache = kv_cache.unbind(0)
            if key_cache.dtype == torch.uint8:
                key_cache = key_cache.view(torch.float8_e4m3fn)
                value_cache = value_cache.view(torch.float8_e4m3fn)
            ws._k_cache = key_cache
            ws._v_cache = value_cache

    def _patch_impls_for_cg(self) -> None:
        """Patch attention impls to dispatch to extend CG workspaces."""
        backend = self.backend
        sfc = backend.static_forward_context

        if not hasattr(self, "_original_impls"):
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
        if not hasattr(self, "_original_impls"):
            return
        sfc = self.backend.static_forward_context
        for name, impl in self._original_impls.items():
            sfc[name].impl = impl

    def repatch_impls(self) -> None:
        """Re-apply CG verify impls."""
        self._patch_impls_for_cg()

    def _fill_buffers(self, slot: int, token_ids: list[int], kv_len: int) -> None:
        """Fill input buffers for verify replay.

        Vectorized (was a per-element Python loop writing up to
        blocks_per_slot ~1024 scalar tensor elements per replay at 64K
        context -- torch.profiler showed this alone as ~1100 aten::copy_
        calls / ~180ms of CPU dispatch overhead per replay, dwarfing the
        ~40ms of actual GPU kernel time. See notes/2026-07-27-dflash-
        profiling-and-optimization.md.

        Further optimized: pre-allocated position offset and sequential
        page table buffers eliminate per-round torch.arange allocations.
        """
        nt = self.num_tokens
        bs = self.block_size
        phys = _physical_slot(slot)
        new_kv_len = kv_len + nt

        # Input IDs: use pre-allocated CPU staging buffer for async copy
        self._input_ids[:nt] = torch.as_tensor(
            token_ids[:nt], dtype=self._input_ids.dtype, device=self.device
        )
        # Positions: pre-allocated offset buffer [0,1,...,nt-1] + kv_len
        pos_range = self._pos_offset[:nt] + kv_len
        self._positions[:nt] = pos_range

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
                align_gran = _debug_swa_align_gran() or bs
                aligned_start = (window_start // align_gran) * align_gran
                aligned_len = new_kv_len - aligned_start
                n_ring = min(-(-aligned_len // bs), self._ring_blocks_per_slot)
                pt = self._page_tables[group_key]
                # Use pre-allocated SWA block index buffer
                block_starts = self._swa_block_starts[:n_ring] * bs + aligned_start
                pt[0, :n_ring] = (ring_base + (block_starts % ring_slots) // bs).to(pt.dtype)
                self._cache_seqlens[group_key][0] = aligned_len
                bfdiag_checks.check_page_table_covers_seqlen(group_key, aligned_len, n_ring, bs)
                # SWA slot mapping: ring-wrapped
                ring_block = (pos_range % ring_slots) // bs
                ring_off = pos_range % bs
                self._swa_slot_mapping[:nt] = (ring_base + ring_block) * bs + ring_off
            else:
                # Full attention: use pre-allocated sequential page table
                pt = self._page_tables[group_key]
                pt[0, :n_blocks_full] = self._seq_page_table[full_base:full_base + n_blocks_full]
                self._cache_seqlens[group_key][0] = new_kv_len

        # Full-attention slot mapping
        self._slot_mapping[:nt] = (full_base + pos_range // bs) * bs + pos_range % bs

        # Update runtime metadata AND worklist for CG replay.
        # update_prefill_graph_replay_metadata copies page_table/cache_seqlens
        # then runs a Triton kernel to recompute block_valid_mask,
        # kv_chunk_size, and window_start_tokens from the new KV lengths.
        # Without this, the worklist stays frozen at capture-time values
        # and SWA ring alignment changes produce wrong attention output.
        for group_key, ws in self._extend_workspaces.items():
            wl = group_key[0]
            ws.update_prefill_graph_replay_metadata(
                self._page_tables[group_key],
                self._cache_seqlens[group_key],
                self._cu_seqlens_q,
                window_left=wl,
            )

    def _init_fill_buffers(self) -> None:
        """Pre-allocate buffers used by _fill_buffers to avoid per-round allocations."""
        nt = self.num_tokens
        # Position offset: [0, 1, ..., nt-1]
        self._pos_offset = torch.arange(nt, dtype=torch.long, device=self.device)
        # Sequential page table: [0, 1, 2, ..., blocks_per_slot * num_slots - 1]
        total_blocks = self.blocks_per_slot * max(1, self.backend.num_slots)
        self._seq_page_table = torch.arange(total_blocks, dtype=torch.int32, device=self.device)
        # SWA block index: [0, 1, 2, ..., ring_blocks_per_slot - 1]
        max_ring = max(self._ring_blocks_per_slot, 1)
        self._swa_block_starts = torch.arange(max_ring, dtype=torch.long, device=self.device)

    def capture(self) -> None:
        """Warmup → capture the verify forward."""
        if self._captured:
            return

        self._init_fill_buffers()
        backend = self.backend
        nt = self.num_tokens
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

        from runtime.laguna_runtime import laguna_forward_context

        # Warmup (3x)
        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                self._fill_buffers(0, dummy_tokens, warmup_kv)
                with bf_attn_context(attn_meta, slot_mapping_dict):
                    with laguna_forward_context(
                        attn_meta,
                        backend.runtime_config,
                        slot_mapping=slot_mapping_dict,
                        skip_compiled=True,
                    ):
                        result = backend.model.forward(self._input_ids, self._positions)
                if isinstance(result, tuple):
                    hidden_states, _aux = result
                else:
                    hidden_states, _aux = result, None
                backend.model.compute_logits(hidden_states)
        side_stream.synchronize()

        # Final fill before capture
        self._fill_buffers(0, dummy_tokens, warmup_kv)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with bf_attn_context(attn_meta, slot_mapping_dict):
                with laguna_forward_context(
                    attn_meta,
                    backend.runtime_config,
                    slot_mapping=slot_mapping_dict,
                    skip_compiled=True,
                ):
                    result = backend.model.forward(self._input_ids, self._positions)
            if isinstance(result, tuple):
                hidden_states, self._aux_hidden_states = result
            else:
                hidden_states, self._aux_hidden_states = result, None
            self._logits = backend.model.compute_logits(hidden_states)

        self._graph = graph
        self._captured = True

        # Restore original attention impls so the eager path keeps working.
        # CG replay uses the captured graph directly, not the impl attribute.
        self.unpatch_impls()

        logger.info("Laguna Verify CUDA Graph captured (M=%d)", nt)

    def replay_with_aux(
        self, slot: int, token_ids: list[int], kv_len: int
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Replay verify graph. Returns (logits[16×vocab], aux_hidden_states)."""
        if not self._captured:
            raise RuntimeError("Verify CG not captured")
        self._fill_buffers(slot, token_ids, kv_len)
        self._graph.replay()

        dump_kv_len = os.environ.get("QSR_DEBUG_AUX_DUMP_KV_LEN")
        if (
            dump_kv_len is not None
            and kv_len >= int(dump_kv_len)
            and not getattr(self, "_aux_dump_done", False)
            and self._aux_hidden_states is not None
        ):
            self._aux_dump_done = True
            dump_path = os.environ.get("QSR_DEBUG_AUX_DUMP_PATH")
            torch.save(
                {
                    "kv_len": kv_len,
                    "block_size": self.block_size,
                    "aux": [t.float().clone().cpu() for t in self._aux_hidden_states],
                },
                dump_path,
            )
            logger.warning(
                "AUX_DUMP saved kv_len=%d bs=%d n_layers=%d to %s",
                kv_len,
                self.block_size,
                len(self._aux_hidden_states),
                dump_path,
            )

        return self._logits, self._aux_hidden_states


class _SparkinferCGExtendMetadata:
    """Lightweight metadata for CG extend impl."""

    __slots__ = ("workspace", "num_actual_tokens")

    def __init__(self, workspace, num_actual_tokens: int):
        self.workspace = workspace
        self.num_actual_tokens = num_actual_tokens


class _SparkinferCGExtendImpl:
    """Attention impl that dispatches to CG extend workspace.

    Rebuilds the binding as warmup tensors move, then captures the binding
    backed by the graph-private fixed-address tensors used during replay.
    """

    def __init__(self, workspace, num_tokens: int):
        self._ws = workspace
        self._num_tokens = num_tokens
        self._binding = None
        self._binding_signature = None
        self._k_descale = None
        self._v_descale = None
        self.kv_cache_dtype = "fp8_e4m3"
        self.supports_quant_query_input = False

    def process_weights_after_loading(self, act_dtype):
        pass

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        k_cache = kv_cache[0].view(torch.float8_e4m3fn)
        v_cache = kv_cache[1].view(torch.float8_e4m3fn)
        block_size = k_cache.shape[1]
        block_idx = slot_mapping // block_size
        block_off = slot_mapping % block_size
        k_cache[block_idx, block_off] = (key / layer._k_scale).to(torch.float8_e4m3fn)
        v_cache[block_idx, block_off] = (value / layer._v_scale).to(torch.float8_e4m3fn)

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if attn_metadata is None:
            return output.fill_(0)
        num_actual_tokens = attn_metadata.num_actual_tokens
        q = query[:num_actual_tokens]
        key_cache, value_cache = kv_cache.unbind(0)
        if key_cache.dtype == torch.uint8:
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
        out = output[:num_actual_tokens]

        from runtime.backends.laguna_sparkinfer_attn import _paged_descale

        if self._k_descale is None:
            self._k_descale = _paged_descale(
                layer._k_scale,
                batch_size=1,
                num_kv_heads=int(key_cache.shape[2]),
            ).clone()
            self._v_descale = _paged_descale(
                layer._v_scale,
                batch_size=1,
                num_kv_heads=int(value_cache.shape[2]),
            ).clone()
        else:
            self._k_descale.copy_(
                _paged_descale(
                    layer._k_scale,
                    batch_size=1,
                    num_kv_heads=int(key_cache.shape[2]),
                )
            )
            self._v_descale.copy_(
                _paged_descale(
                    layer._v_scale,
                    batch_size=1,
                    num_kv_heads=int(value_cache.shape[2]),
                )
            )

        # Model warmup forwards allocate fresh Q/output tensors.  The binding
        # holds their raw addresses, so caching the first warmup binding makes
        # the captured graph read and write stale buffers.  Rebind when capture
        # moves tensors into the graph-private memory pool (or whenever any
        # bound tensor address changes).
        binding_signature = (
            q.data_ptr(),
            key_cache.data_ptr(),
            value_cache.data_ptr(),
            out.data_ptr(),
        )
        if self._binding is None or binding_signature != self._binding_signature:
            from sparkinfer.attention.paged._scratch import build_paged_attention_binding

            self._binding = build_paged_attention_binding(
                scratch=self._ws,
                q=q,
                k_cache=key_cache,
                v_cache=value_cache,
                output=out,
                k_descale=self._k_descale,
                v_descale=self._v_descale,
            )
            self._binding_signature = binding_signature

        from sparkinfer.attention.paged._forward import paged_attention_forward

        paged_attention_forward(binding=self._binding)
        return output
