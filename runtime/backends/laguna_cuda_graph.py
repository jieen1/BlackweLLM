"""Laguna CUDA Graph decode — FlashInfer cudagraph wrapper + fast_decode_plan。

利用 FlashInfer 原生 cudagraph 模式（BatchDecodeWithPagedKVCacheWrapper(use_cuda_graph=True)）
和 vLLM vendor 的 fast_decode_plan（每步只做 indptr/last_page_len 的小 H2D copy）。

简化红利：block table 每槽连续（base = phys * blocks_per_slot + j），
paged_kv_indices 在槽位生命周期内不变，每步真正变的只有 last_page_len 和
跨页时的 indptr。

性能目标：消除 51ms Python dispatch overhead，ITL 从 66ms → ~15ms。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from runtime.backends.laguna import LagunaBackend

logger = logging.getLogger("qwen_sm120_runtime.laguna_cuda_graph")


class LagunaCudaGraphDecode:
    """CUDA-graph-captured decode for a FIXED batch size.

    Per (batch_size) instance. Captures the full decode forward
    (metadata build + model.forward + compute_logits) using FlashInfer's
    cudagraph-enabled decode wrappers.

    Usage:
        cg = LagunaCudaGraphDecode(backend, batch_size=1)
        cg.capture()  # warmup + plan + capture

        # In decode loop:
        next_tokens = cg.replay(slot_ids, token_ids, kv_lengths)
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

        # ── FlashInfer cudagraph buffers (full-attention groups) ──
        max_pages = batch_size * self.blocks_per_slot
        self._fi_indptr_cpu = torch.zeros(batch_size + 1, dtype=torch.int32, pin_memory=True)
        self._fi_indptr_gpu = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
        self._fi_indices_gpu = torch.zeros(max_pages, dtype=torch.int32, device=self.device)
        self._fi_last_page_len_cpu = torch.zeros(batch_size, dtype=torch.int32, pin_memory=True)
        self._fi_last_page_len_gpu = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        self._fi_last_page_len_staging = torch.zeros(batch_size, dtype=torch.int32)

        # ── SWA ring buffers (separate from full-attention) ──
        rbps = backend._ring_blocks_per_slot
        self._ring_blocks_per_slot = rbps
        self._ring_slots_per_slot = rbps * self.block_size
        self._swa_window = backend._swa_window
        if rbps > 0:
            swa_max_pages = batch_size * rbps
            self._swa_fi_indptr_cpu = torch.zeros(batch_size + 1, dtype=torch.int32, pin_memory=True)
            self._swa_fi_indptr_gpu = torch.zeros(batch_size + 1, dtype=torch.int32, device=self.device)
            self._swa_fi_indices_gpu = torch.zeros(swa_max_pages, dtype=torch.int32, device=self.device)
            self._swa_fi_last_page_len_cpu = torch.zeros(batch_size, dtype=torch.int32, pin_memory=True)
            self._swa_fi_last_page_len_gpu = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
            self._swa_fi_last_page_len_staging = torch.zeros(batch_size, dtype=torch.int32)
            self._swa_slot_mapping = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            self._swa_block_table = torch.zeros(
                batch_size, rbps, dtype=torch.int32, device=self.device
            )
            self._swa_prev_n_blocks: list[int] = [0] * batch_size

        # ── CommonAttentionMetadata pre-allocated fields ──
        self._qsl_gpu = torch.arange(batch_size + 1, dtype=torch.int32, device=self.device)
        self._qsl_cpu = torch.arange(batch_size + 1, dtype=torch.int32, pin_memory=True)
        self._seq_lens_gpu = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        self._block_table = torch.zeros(
            batch_size, self.blocks_per_slot, dtype=torch.int32, device=self.device
        )

        # ── FlashInfer cudagraph decode wrappers (one per layer group) ──
        self._decode_wrappers: dict[tuple, Any] = {}
        self._fi_metadata: dict[tuple, Any] = {}
        self._workspaces: list[torch.Tensor] = []

        # ── Per-slot page-crossing tracker ──
        self._prev_n_blocks: list[int] = [0] * batch_size

        # ── DFlash aux hidden states (captured in graph) ──
        self._aux_hidden_states: list[torch.Tensor] | None = None

    def _init_wrappers(self) -> None:
        """Create FlashInfer cudagraph-enabled decode wrappers per layer group.

        Each wrapper gets its OWN workspace buffer (not shared with the eager
        builder) to prevent prefill from polluting decode's scheduling area.
        SWA groups get separate indptr/indices/last_page_len buffers (ring KV).
        """
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

        backend = self.backend
        bs = self.batch_size

        for group_key, builder in backend._metadata_builders.items():
            workspace = torch.empty(
                builder._get_workspace_buffer().numel(),
                dtype=torch.uint8,
                device=self.device,
            )
            wl = group_key[0]
            is_swa = wl >= 0 and self._ring_blocks_per_slot > 0
            if is_swa:
                indptr_buf = self._swa_fi_indptr_gpu[:bs + 1]
                indices_buf = self._swa_fi_indices_gpu
                lpl_buf = self._swa_fi_last_page_len_gpu[:bs]
            else:
                indptr_buf = self._fi_indptr_gpu[:bs + 1]
                indices_buf = self._fi_indices_gpu
                lpl_buf = self._fi_last_page_len_gpu[:bs]
            wrapper = BatchDecodeWithPagedKVCacheWrapper(
                workspace,
                "NHD",
                use_cuda_graph=True,
                paged_kv_indptr_buffer=indptr_buf,
                paged_kv_indices_buffer=indices_buf,
                paged_kv_last_page_len_buffer=lpl_buf,
                use_tensor_cores=True,
            )
            self._decode_wrappers[group_key] = wrapper
            self._workspaces.append(workspace)

    def _fill_buffers(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
    ) -> None:
        """Update pre-allocated buffers for replay (vectorized)."""
        from runtime.backends.laguna import _physical_slot

        bs = len(slot_ids)
        ps = self.block_size

        # Vectorized: input_ids, positions, seq_lens
        self._input_ids[:bs] = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        kv_t = torch.tensor(kv_lengths, dtype=torch.long, device=self.device)
        self._positions[:bs] = kv_t
        new_kv = kv_t.int() + 1
        self._seq_lens_gpu[:bs] = new_kv

        # Per-slot: block_table, slot_mapping, last_page_len, indices
        n_blocks_t = (new_kv + ps - 1) // ps
        self._fi_indptr_cpu[0] = 0
        for i in range(bs):
            phys = _physical_slot(slot_ids[i])
            base = phys * self.blocks_per_slot
            nb = int(n_blocks_t[i].item())

            self._block_table[i, :nb] = torch.arange(
                base, base + nb, dtype=torch.int32, device=self.device
            )

            pos = kv_lengths[i]
            self._slot_mapping[i] = (base + pos // ps) * ps + pos % ps

            lpl = int(new_kv[i].item()) % ps
            self._fi_last_page_len_cpu[i] = lpl if lpl != 0 else ps

            self._fi_indptr_cpu[i + 1] = self._fi_indptr_cpu[i] + nb

            start = int(self._fi_indptr_cpu[i].item())
            self._fi_indices_gpu[start:start + nb] = torch.arange(
                base, base + nb, dtype=torch.int32, device=self.device
            )

        self._fi_indptr_gpu[:bs + 1].copy_(self._fi_indptr_cpu[:bs + 1], non_blocking=True)
        self._fi_last_page_len_gpu[:bs].copy_(self._fi_last_page_len_cpu[:bs], non_blocking=True)

        # ── SWA ring buffers ──
        if self._ring_blocks_per_slot > 0:
            rbps = self._ring_blocks_per_slot
            ring_slots = self._ring_slots_per_slot
            window = self._swa_window
            self._swa_fi_indptr_cpu[0] = 0
            for i in range(bs):
                phys = _physical_slot(slot_ids[i])
                ring_base = phys * rbps
                pos = kv_lengths[i]
                new_kv = pos + 1

                window_start = max(0, pos - window + 1)
                aligned_start = (window_start // ps) * ps
                aligned_len = new_kv - aligned_start
                n_ring = (aligned_len + ps - 1) // ps

                for j in range(n_ring):
                    actual = aligned_start + j * ps
                    rb = (actual % ring_slots) // ps
                    self._swa_block_table[i, j] = ring_base + rb

                rb_dec = (pos % ring_slots) // ps
                ro_dec = pos % ps
                self._swa_slot_mapping[i] = (ring_base + rb_dec) * ps + ro_dec

                lpl = aligned_len % ps
                self._swa_fi_last_page_len_staging[i] = lpl if lpl != 0 else ps

                self._swa_fi_indptr_cpu[i + 1] = self._swa_fi_indptr_cpu[i] + n_ring
                start = int(self._swa_fi_indptr_cpu[i].item())
                self._swa_fi_indices_gpu[start:start + n_ring] = self._swa_block_table[i, :n_ring]

            self._swa_fi_last_page_len_cpu[:bs].copy_(self._swa_fi_last_page_len_staging[:bs])
            self._swa_fi_indptr_gpu[:bs + 1].copy_(self._swa_fi_indptr_cpu[:bs + 1], non_blocking=True)
            self._swa_fi_last_page_len_gpu[:bs].copy_(self._swa_fi_last_page_len_cpu[:bs], non_blocking=True)

    def _run_plan(self, slot_ids: list[int], kv_lengths: list[int]) -> None:
        """Run fast_decode_plan on all layer group wrappers."""
        from vllm.v1.attention.backends.flashinfer import fast_plan_decode


        backend = self.backend
        bs = len(slot_ids)

        for group_key, wrapper in self._decode_wrappers.items():
            wl, nqh, nkvh = group_key
            head_dim = backend.head_dim
            page_size = self.block_size

            kv_dtype = torch.float8_e4m3fn if "fp8" in backend._cache_dtype_str else torch.bfloat16

            # Per-group buffers: SWA uses ring buffers
            is_swa = wl >= 0 and self._ring_blocks_per_slot > 0
            if is_swa:
                indptr_cpu = self._swa_fi_indptr_cpu[:bs + 1]
                indices = self._swa_fi_indices_gpu
                lpl_cpu = self._swa_fi_last_page_len_cpu[:bs]
            else:
                indptr_cpu = self._fi_indptr_cpu[:bs + 1]
                indices = self._fi_indices_gpu
                lpl_cpu = self._fi_last_page_len_cpu[:bs]

            builder_sm_scale = backend._metadata_builders[group_key].sm_scale
            fast_plan_decode(
                wrapper,
                indptr_cpu=indptr_cpu,
                indices=indices,
                last_page_len_cpu=lpl_cpu,
                num_qo_heads=nqh,
                num_kv_heads=nkvh,
                head_dim=head_dim,
                page_size=page_size,
                pos_encoding_mode="NONE",
                window_left=wl,
                logits_soft_cap=None,
                q_data_type=torch.bfloat16,
                kv_data_type=kv_dtype,
                sm_scale=builder_sm_scale,
                non_blocking=True,
                fixed_split_size=-1,
                disable_split_kv=False,
            )
            wrapper._sm_scale = builder_sm_scale

    def _build_metadata_and_forward(self) -> torch.Tensor:
        """Build FlashInferMetadata from pre-allocated buffers and run forward."""
        from runtime.compat_vllm import (
            set_current_vllm_config,
            set_forward_context,
        )

        backend = self.backend
        bs = self.batch_size

        # Build FlashInferMetadata using cudagraph wrappers
        from vllm.v1.attention.backends.flashinfer import FIDecode, FlashInferMetadata

        attn_metadata_dict: dict[str, Any] = {}
        slot_mapping_dict: dict[str, torch.Tensor] = {}

        for group_key, wrapper in self._decode_wrappers.items():
            wl = group_key[0]
            is_swa = wl >= 0 and self._ring_blocks_per_slot > 0
            sm = self._swa_slot_mapping[:bs] if is_swa else self._slot_mapping[:bs]
            metadata = FlashInferMetadata(
                num_actual_tokens=bs,
                slot_mapping=sm,
                q_data_type_prefill=torch.bfloat16,
                q_data_type_decode=torch.bfloat16,
                num_decodes=bs,
                num_decode_tokens=bs,
                num_prefills=0,
                num_prefill_tokens=0,
                causal=True,
                use_cascade=False,
                prefill=None,
                decode=FIDecode(wrapper=wrapper),
                cascade_wrapper=None,
            )
            for name in backend._layer_groups[group_key]:
                attn_metadata_dict[name] = metadata
                slot_mapping_dict[name] = sm

        with set_current_vllm_config(backend.vllm_config):
            with set_forward_context(
                attn_metadata_dict, backend.vllm_config, slot_mapping=slot_mapping_dict,
                skip_compiled=True,
            ):
                result = backend.model.forward(
                    self._input_ids[:bs], self._positions[:bs]
                )

        # Handle tuple return when aux_hidden_state_layers is set (DFlash)
        if isinstance(result, tuple):
            hidden_states, self._aux_hidden_states = result
        else:
            hidden_states = result
            self._aux_hidden_states = None
        return backend.model.compute_logits(hidden_states)

    def capture(self) -> None:
        """Warmup → plan → capture the decode forward."""
        if self._captured:
            return

        backend = self.backend
        bs = self.batch_size

        # Reserve warmup slots (last bs slots)
        warmup_slots = list(range(backend.num_slots - bs, backend.num_slots))
        dummy_tokens = [1] * bs
        # Capture at max kv_len so FlashInfer grid covers all context lengths
        capture_kv = self.blocks_per_slot * self.block_size - 1
        dummy_kv_lens = [capture_kv] * bs

        logger.info("Capturing Laguna CUDA Graph: batch_size=%d", bs)

        # Initialize wrappers
        self._init_wrappers()

        # Fill buffers with dummy data
        self._fill_buffers(warmup_slots, dummy_tokens, dummy_kv_lens)

        # Warmup: run plan + forward on side stream (3x)
        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                self._fill_buffers(warmup_slots, dummy_tokens, dummy_kv_lens)
                self._run_plan(warmup_slots, dummy_kv_lens)
                self._build_metadata_and_forward()
        side_stream.synchronize()

        # Final fill + plan before capture
        self._fill_buffers(warmup_slots, dummy_tokens, dummy_kv_lens)
        self._run_plan(warmup_slots, dummy_kv_lens)

        # Capture: forward + argmax + write next token (self-feeding graph)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._logits = self._build_metadata_and_forward()
            # Argmax fused into graph: no separate kernel launch per step
            self._input_ids[0] = self._logits[0].argmax(dim=-1).to(torch.long)

        self._graph = graph
        self._captured = True

        logger.info("Laguna CUDA Graph captured: batch_size=%d", bs)

    def replay(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
    ) -> list[int]:
        """Replay the captured graph with real data.

        FlashInfer contract: plan→run must be paired every step.
        fast_plan_decode is cheap (small H2D copies), never skip it.
        """
        if not self._captured:
            raise RuntimeError("Graph not captured. Call capture() first.")

        bs = len(slot_ids)
        ps = self.block_size
        bps = self.blocks_per_slot

        for i in range(bs):
            kvl = kv_lengths[i]
            new_kv = kvl + 1

            self._input_ids[i] = token_ids[i]
            self._positions[i] = kvl

            phys = slot_ids[i] + 1  # inlined _physical_slot
            base = phys * bps
            self._slot_mapping[i] = base * ps + kvl

            lpl = new_kv % ps
            self._fi_last_page_len_staging[i] = lpl if lpl != 0 else ps

            n_blocks = (new_kv + ps - 1) // ps
            if n_blocks != self._prev_n_blocks[i]:
                self._prev_n_blocks[i] = n_blocks
                self._block_table[i, :n_blocks] = torch.arange(
                    base, base + n_blocks, dtype=torch.int32, device=self.device
                )
                self._fi_indptr_cpu[0] = 0
                for j in range(bs):
                    nb = self._prev_n_blocks[j]
                    self._fi_indptr_cpu[j + 1] = self._fi_indptr_cpu[j] + nb
                    p2 = slot_ids[j] + 1
                    b2 = p2 * bps
                    start = int(self._fi_indptr_cpu[j].item())
                    self._fi_indices_gpu[start:start + nb] = torch.arange(
                        b2, b2 + nb, dtype=torch.int32, device=self.device
                    )

        self._fi_last_page_len_cpu[:bs].copy_(self._fi_last_page_len_staging[:bs])
        self._fi_indptr_gpu[:bs + 1].copy_(self._fi_indptr_cpu[:bs + 1], non_blocking=True)
        self._fi_last_page_len_gpu[:bs].copy_(self._fi_last_page_len_cpu[:bs], non_blocking=True)

        # SWA ring update
        if self._ring_blocks_per_slot > 0:
            rbps = self._ring_blocks_per_slot
            ring_slots = self._ring_slots_per_slot
            window = self._swa_window
            self._swa_fi_indptr_cpu[0] = 0
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
                    for j in range(n_ring):
                        actual = aligned_start + j * ps
                        rb = (actual % ring_slots) // ps
                        self._swa_block_table[i, j] = ring_base + rb
                    self._swa_fi_indptr_cpu[0] = 0
                    for jj in range(bs):
                        nb = self._swa_prev_n_blocks[jj]
                        self._swa_fi_indptr_cpu[jj + 1] = self._swa_fi_indptr_cpu[jj] + nb
                        p2 = slot_ids[jj] + 1
                        rb2 = p2 * rbps
                        ws2 = max(0, kv_lengths[jj] - window + 1)
                        as2 = (ws2 // ps) * ps
                        al2 = kv_lengths[jj] + 1 - as2
                        nr2 = (al2 + ps - 1) // ps
                        st = int(self._swa_fi_indptr_cpu[jj].item())
                        for k in range(nr2):
                            ap = as2 + k * ps
                            self._swa_fi_indices_gpu[st + k] = rb2 + (ap % ring_slots) // ps

                rb_dec = (pos % ring_slots) // ps
                ro_dec = pos % ps
                self._swa_slot_mapping[i] = (ring_base + rb_dec) * ps + ro_dec
                lpl = aligned_len % ps
                self._swa_fi_last_page_len_staging[i] = lpl if lpl != 0 else ps

            self._swa_fi_last_page_len_cpu[:bs].copy_(self._swa_fi_last_page_len_staging[:bs])
            self._swa_fi_indptr_gpu[:bs + 1].copy_(self._swa_fi_indptr_cpu[:bs + 1], non_blocking=True)
            self._swa_fi_last_page_len_gpu[:bs].copy_(self._swa_fi_last_page_len_cpu[:bs], non_blocking=True)

        self._run_plan(slot_ids, kv_lengths)
        self._graph.replay()

        # Argmax is fused in graph → input_ids already has next token(s)
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
        max_tokens: int,
        eos_tokens: tuple[int, ...] = (2, 24),
    ) -> list[int]:
        """Zero-sync decode loop: argmax stays on GPU, single sync at end.

        Eliminates per-step .item() GPU synchronization by:
        1. Writing argmax directly to _input_ids on GPU (GPU→GPU, no sync)
        2. Accumulating tokens in a GPU tensor
        3. Syncing ONCE at the end to read all tokens + check EOS

        This pipelines Python buffer prep with GPU compute:
          CPU: [prep N+1] [plan N+1] [replay N+1] [prep N+2] ...
          GPU: [====compute N====]   [====compute N+1====]   ...
        """
        if not self._captured:
            raise RuntimeError("Graph not captured. Call capture() first.")

        backend = self.backend
        ps = self.block_size
        bps = self.blocks_per_slot
        phys = slot + 1
        base = phys * bps

        # GPU token accumulator (avoids per-step sync)
        out_tokens_gpu = torch.empty(max_tokens, dtype=torch.long, device=self.device)
        out_tokens_gpu[0] = first_token
        self._input_ids[0] = first_token

        for step in range(max_tokens - 1):
            kvl = backend.slot_kv_len[slot]
            new_kv = kvl + 1

            self._positions[0] = kvl
            self._slot_mapping[0] = base * ps + kvl

            # SWA ring slot_mapping
            if self._ring_blocks_per_slot > 0:
                ring_base = phys * self._ring_blocks_per_slot
                ring_slots = self._ring_slots_per_slot
                rb = (kvl % ring_slots) // ps
                ro = kvl % ps
                self._swa_slot_mapping[0] = (ring_base + rb) * ps + ro

            lpl = new_kv % ps
            self._fi_last_page_len_staging[0] = lpl if lpl != 0 else ps

            n_blocks = (new_kv + ps - 1) // ps
            if n_blocks != self._prev_n_blocks[0]:
                self._prev_n_blocks[0] = n_blocks
                self._block_table[0, :n_blocks] = torch.arange(
                    base, base + n_blocks, dtype=torch.int32, device=self.device
                )
                self._fi_indptr_cpu[0] = 0
                self._fi_indptr_cpu[1] = n_blocks
                self._fi_indices_gpu[:n_blocks] = torch.arange(
                    base, base + n_blocks, dtype=torch.int32, device=self.device
                )

            # SWA ring block_table update
            if self._ring_blocks_per_slot > 0:
                rbps = self._ring_blocks_per_slot
                ring_slots = self._ring_slots_per_slot
                window = self._swa_window
                ring_base_g = phys * rbps
                ws = max(0, kvl - window + 1)
                as_ = (ws // ps) * ps
                al = new_kv - as_
                nr = (al + ps - 1) // ps
                if nr != self._swa_prev_n_blocks[0]:
                    self._swa_prev_n_blocks[0] = nr
                    for j in range(nr):
                        ap = as_ + j * ps
                        self._swa_block_table[0, j] = ring_base_g + (ap % ring_slots) // ps
                    self._swa_fi_indptr_cpu[0] = 0
                    self._swa_fi_indptr_cpu[1] = nr
                    self._swa_fi_indices_gpu[:nr] = self._swa_block_table[0, :nr]
                lpl_swa = al % ps
                self._swa_fi_last_page_len_staging[0] = lpl_swa if lpl_swa != 0 else ps

            # SWA last_page_len + indptr must be copied EVERY step
            # (last_page_len changes every step, not just on block boundary)
            if self._ring_blocks_per_slot > 0:
                self._swa_fi_last_page_len_cpu[0] = self._swa_fi_last_page_len_staging[0]
                self._swa_fi_indptr_gpu[:2].copy_(self._swa_fi_indptr_cpu[:2], non_blocking=True)
                self._swa_fi_last_page_len_gpu[:1].copy_(self._swa_fi_last_page_len_cpu[:1], non_blocking=True)

            self._fi_last_page_len_cpu[0] = self._fi_last_page_len_staging[0]
            self._fi_indptr_gpu[:2].copy_(self._fi_indptr_cpu[:2], non_blocking=True)
            self._fi_last_page_len_gpu[:1].copy_(self._fi_last_page_len_cpu[:1], non_blocking=True)
            self._run_plan([slot], [kvl])
            self._graph.replay()

            # Argmax is fused in graph → input_ids[0] already has next token
            out_tokens_gpu[step + 1] = self._input_ids[0]

            backend.slot_kv_len[slot] += 1

        # Single sync at end: read all tokens, truncate at EOS
        torch.cuda.synchronize()
        all_tokens = out_tokens_gpu[:max_tokens].tolist()
        tokens = [first_token]
        for tok in all_tokens[1:]:
            tokens.append(tok)
            backend.slot_committed_tokens[slot].append(tok)
            if tok in eos_tokens:
                break

        return tokens

    def reset(self) -> None:
        """Reset per-slot tracking state for a fresh generation.

        Zeros workspace buffers to prevent capture-warmup residue from
        affecting the first replay (run1 vs run2 divergence).
        """
        self._prev_n_blocks = [0] * self.batch_size
        if self._ring_blocks_per_slot > 0:
            self._swa_prev_n_blocks = [0] * self.batch_size
        for ws in self._workspaces:
            ws.zero_()

    @property
    def is_captured(self) -> bool:
        return self._captured


class MultiBatchGraphManager:
    """Manages CUDA Graphs for batch_size=1..max_bs, dispatches by actual batch.

    Production形态是多 slot 并发。每个 batch_size 捕获一张独立 graph，
    运行时按实际 batch 大小选择对应 graph replay。
    """

    def __init__(self, backend: LagunaBackend, max_batch_size: int = 4) -> None:
        self.backend = backend
        self.max_batch_size = max_batch_size
        self._graphs: dict[int, LagunaCudaGraphDecode] = {}

    def capture_all(self) -> None:
        """Capture graphs for all batch sizes 1..max_batch_size."""
        for bs in range(1, self.max_batch_size + 1):
            logger.info("Capturing graph for batch_size=%d", bs)
            cg = LagunaCudaGraphDecode(self.backend, batch_size=bs)
            cg.capture()
            self._graphs[bs] = cg

    def get(self, batch_size: int) -> LagunaCudaGraphDecode:
        """Get the graph for a specific batch size."""
        if batch_size not in self._graphs:
            raise RuntimeError(
                f"No graph captured for batch_size={batch_size}. "
                f"Available: {sorted(self._graphs.keys())}"
            )
        return self._graphs[batch_size]

    def replay(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
    ) -> list[int]:
        """Dispatch to the correct batch-size graph."""
        bs = len(slot_ids)
        return self.get(bs).replay(slot_ids, token_ids, kv_lengths)

    def generate(
        self,
        slot: int,
        first_token: int,
        max_tokens: int,
        eos_tokens: tuple[int, ...] = (2, 24),
    ) -> list[int]:
        """Single-slot generate using bs=1 graph."""
        return self.get(1).generate(slot, first_token, max_tokens, eos_tokens)

    def reset(self, batch_size: int | None = None) -> None:
        """Reset specific or all graphs."""
        if batch_size is not None:
            if batch_size in self._graphs:
                self._graphs[batch_size].reset()
        else:
            for cg in self._graphs.values():
                cg.reset()

    @property
    def captured_sizes(self) -> list[int]:
        return sorted(self._graphs.keys())
