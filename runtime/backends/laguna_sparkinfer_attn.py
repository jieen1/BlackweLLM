"""SparkInfer paged attention — full replacement for FlashInfer in Laguna.

Handles both prefill (extend mode) and decode (CG mode) for all layer groups:
- Full attention: window_left=-1, 24 Q heads, 8 KV heads, head_dim=128
- SWA: window_left=511, same head config

KV cache layout: vLLM stores [num_blocks, 2, block_size, num_kv_heads, head_dim].
After unbind(1): k_cache/v_cache = [num_blocks, block_size, num_kv_heads, head_dim]
which is exactly sparkinfer's expected [num_pages, page_size, num_kv_heads, head_dim].

Integration: monkey-patch each Attention layer's impl after model load.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

import torch

logger = logging.getLogger("qwen_sm120_runtime.sparkinfer_attn")

_BF_SPARKINFER_PATH = os.environ.get("BF_SPARKINFER_PATH", "/home/bot/project/sparkinfer")
if _BF_SPARKINFER_PATH and _BF_SPARKINFER_PATH not in sys.path:
    sys.path.insert(0, _BF_SPARKINFER_PATH)

PAGE_SIZE = 64  # Must match LagunaBackend.block_size


class SparkinferAttnMetadata:
    """Lightweight metadata passed through forward context for sparkinfer."""
    __slots__ = ("mode", "page_table", "cache_seqlens", "cu_seqlens_q",
                 "num_actual_tokens", "window_left")

    def __init__(self, mode: str, page_table: torch.Tensor,
                 cache_seqlens: torch.Tensor, cu_seqlens_q: torch.Tensor,
                 num_actual_tokens: int, window_left: int = -1):
        self.mode = mode
        self.page_table = page_table
        self.cache_seqlens = cache_seqlens
        self.cu_seqlens_q = cu_seqlens_q
        self.num_actual_tokens = num_actual_tokens
        self.window_left = window_left


class SparkinferPrefillWorkspace:
    """Manages sparkinfer extend-mode workspaces for prefill (eager, no CG).

    One workspace per layer group (full-attn vs SWA). Rebuilt per forward call
    since prefill shapes vary. Overhead is minimal vs the actual compute.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._descale = torch.ones(1, dtype=torch.float32, device=device)

    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        window_left: int = -1,
    ) -> None:
        """Run extend-mode attention (prefill)."""
        from sparkinfer.attention.paged.workspace import PagedAttentionWorkspace
        from sparkinfer.attention.paged.planner import create_paged_plan
        from sparkinfer.attention.paged._forward import paged_attention_forward
        from sparkinfer.attention.paged._scratch import build_paged_attention_binding

        ws = PagedAttentionWorkspace.for_tensors(
            mode="extend", q=q, k_cache=k_cache, v_cache=v_cache,
            use_cuda_graph=False)

        plan = create_paged_plan(
            q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
            mode="extend", enable_cuda_graph=False, window_left=window_left)
        ws._ensure_capacity(plan)
        ws._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
        ws._copy_plan_metadata(plan)
        ws._plan = plan

        binding = build_paged_attention_binding(
            scratch=ws, q=q, k_cache=k_cache, v_cache=v_cache,
            output=output, k_descale=self._descale, v_descale=self._descale)
        paged_attention_forward(binding=binding)


class SparkinferDecodeWorkspace:
    """Manages sparkinfer decode-mode workspace for CG replay.

    One per layer group. Captured in CUDA graph. Per-step update is just
    writing cache_seqlens (GPU int32 write) + graph.replay().
    """

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_pages: int,
        window_left: int = -1,
        device: str = "cuda",
    ):
        from sparkinfer.attention.paged.workspace import PagedAttentionWorkspace
        from sparkinfer.attention.paged.planner import create_paged_plan

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_pages = max_pages
        self.window_left = window_left
        self.device = torch.device(device)

        # Dummy tensors for workspace creation (real ones bound at capture)
        self._q = torch.zeros(1, num_q_heads, head_dim, dtype=torch.bfloat16, device=self.device)
        self._k_cache = torch.zeros(max_pages, PAGE_SIZE, num_kv_heads, head_dim,
                                    dtype=torch.float8_e4m3fn, device=self.device)
        self._v_cache = torch.zeros(max_pages, PAGE_SIZE, num_kv_heads, head_dim,
                                    dtype=torch.float8_e4m3fn, device=self.device)
        self._output = torch.zeros(1, num_q_heads, head_dim, dtype=torch.bfloat16, device=self.device)
        self._descale = torch.ones(1, dtype=torch.float32, device=self.device)

        # Create graph-mode workspace with prepare_decode_graph_replay_state.
        # Requires sparkinfer commit 0a7b143+ (fixes capacity underestimation
        # for windowed attention with small page counts).
        self._workspace = PagedAttentionWorkspace.for_tensors(
            mode="decode", q=self._q, k_cache=self._k_cache,
            v_cache=self._v_cache, use_cuda_graph=True)
        self._workspace.prepare_decode_graph_replay_state(
            batch=1, max_page_table_width=max_pages, window_left=window_left)

        # Bind runtime metadata at max context for capture
        capture_page_table = torch.arange(max_pages, dtype=torch.int32, device=self.device).unsqueeze(0)
        capture_cache_seqlens = torch.tensor([max_pages * PAGE_SIZE - 1], dtype=torch.int32, device=self.device)
        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        self._workspace._copy_runtime_metadata(capture_page_table, capture_cache_seqlens, cu_seqlens_q)

        self._cu_seqlens_q = cu_seqlens_q
        logger.info(
            "SparkinferDecodeWorkspace: q_heads=%d kv_heads=%d head_dim=%d "
            "max_pages=%d window_left=%d",
            num_q_heads, num_kv_heads, head_dim, max_pages, window_left)

    def bind_kv(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Bind real tensors (called before CG capture)."""
        self._q = q
        self._k_cache = k_cache
        self._v_cache = v_cache
        self._output = output

    def forward(self) -> torch.Tensor:
        """Run attention (captured in CUDA graph)."""
        from sparkinfer.attention.paged._forward import paged_attention_forward
        from sparkinfer.attention.paged._scratch import build_paged_attention_binding

        binding = build_paged_attention_binding(
            scratch=self._workspace,
            q=self._q,
            k_cache=self._k_cache,
            v_cache=self._v_cache,
            output=self._output,
            k_descale=self._descale,
            v_descale=self._descale,
        )
        paged_attention_forward(binding=binding)
        return self._output

    @property
    def cache_seqlens(self) -> torch.Tensor:
        return self._workspace.cache_seqlens

    @property
    def page_table(self) -> torch.Tensor:
        return self._workspace.page_table


class SparkinferAttentionImpl:
    """Drop-in replacement for FlashInferImpl on attention layers.

    Reads SparkinferAttnMetadata from forward context and dispatches to
    sparkinfer paged attention. Handles both prefill (extend) and decode.
    """

    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int, window_left: int = -1, **kwargs):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.window_left = window_left
        self.kv_cache_dtype = "fp8_e4m3"
        self.supports_quant_query_input = False
        self._prefill_ws: SparkinferPrefillWorkspace | None = None

    def _get_prefill_ws(self, device: torch.device) -> SparkinferPrefillWorkspace:
        if self._prefill_ws is None:
            self._prefill_ws = SparkinferPrefillWorkspace(device)
        return self._prefill_ws

    def process_weights_after_loading(self, act_dtype):
        pass

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write K/V into paged cache using vLLM's reshape_and_cache_flash."""
        k_cache = kv_cache[:, 0]
        v_cache = kv_cache[:, 1]
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key, value, k_cache, v_cache, slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        if num_actual_tokens == 0:
            return output.fill_(0)

        q = query[:num_actual_tokens]
        # KV cache: [num_blocks, 2, block_size, num_kv_heads, head_dim]
        key_cache, value_cache = kv_cache.unbind(1)
        # vLLM stores FP8 as uint8; sparkinfer expects float8_e4m3fn
        if key_cache.dtype == torch.uint8:
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
        # Now: [num_blocks, block_size, num_kv_heads, head_dim] — sparkinfer layout

        # CG decode path: metadata has a workspace attribute
        if hasattr(attn_metadata, "workspace"):
            ws = attn_metadata.workspace
            ws._q = q
            ws._k_cache = key_cache
            ws._v_cache = value_cache
            ws._output = output[:num_actual_tokens]
            ws.forward()
            return output

        # Prefill/extend path: create ephemeral workspace
        ws = self._get_prefill_ws(q.device)
        ws.forward(
            q=q,
            k_cache=key_cache,
            v_cache=value_cache,
            output=output[:num_actual_tokens],
            page_table=attn_metadata.page_table,
            cache_seqlens=attn_metadata.cache_seqlens,
            cu_seqlens_q=attn_metadata.cu_seqlens_q,
            window_left=attn_metadata.window_left,
        )
        return output
