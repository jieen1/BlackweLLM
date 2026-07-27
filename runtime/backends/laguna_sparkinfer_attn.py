"""SparkInfer paged attention — full replacement for FlashInfer in Laguna.

Handles both prefill (extend mode) and decode (CG mode) for all layer groups:
- Full attention: window_left=-1, 48 Q heads, 8 KV heads (gqa_group_size=6), head_dim=128
- SWA: window_left=511, 72 Q heads, 8 KV heads (gqa_group_size=9), head_dim=128

These are the real unsharded weight shapes (verified against the checkpoint's
safetensors tensors directly, not the config). Production runs TP=1 (no tensor
parallelism implemented), so these are also the shapes seen at runtime -- not
the TP=2-sharded num_kv_heads=4 that some upstream sparkinfer kernel
specializations are tuned for.

KV cache layout: vLLM stores [2, num_blocks, block_size, num_kv_heads, head_dim].
After unbind(0): k_cache/v_cache = [num_blocks, block_size, num_kv_heads, head_dim]
which is exactly sparkinfer's expected [num_pages, page_size, num_kv_heads, head_dim].

Integration: monkey-patch each Attention layer's impl after model load.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import torch
from vllm._custom_ops import reshape_and_cache_flash

logger = logging.getLogger("qwen_sm120_runtime.sparkinfer_attn")

_BF_SPARKINFER_PATH = os.environ.get("BF_SPARKINFER_PATH", "/home/bot/project/sparkinfer")
if _BF_SPARKINFER_PATH and _BF_SPARKINFER_PATH not in sys.path:
    sys.path.insert(0, _BF_SPARKINFER_PATH)

PAGE_SIZE = 64  # Default for SparkinferDecodeWorkspace.page_size; callers should
# pass the real LagunaBackend.block_size explicitly (64 or 128 both supported).


def _paged_descale(
    scale: torch.Tensor,
    *,
    batch_size: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Normalize vLLM KV scales to sparkinfer's per-request contract.

    vLLM attention layers expose scalar, per-head, or singleton-expanded
    scales depending on whether the checkpoint stores FP8 KV scale tensors.
    SparkInfer requires a rank-1 ``[batch]`` or rank-2
    ``[batch, num_kv_heads]`` descale tensor.  In particular, DFlash's BF16
    checkpoint-backed draft layers expose a rank-0 default scale.
    """
    scale = scale.detach().to(dtype=torch.float32)
    count = scale.numel()
    if count == 1:
        return scale.reshape(1).expand(batch_size).contiguous()
    if count == batch_size:
        return scale.reshape(batch_size).contiguous()
    if count == num_kv_heads:
        return scale.reshape(1, num_kv_heads).expand(batch_size, -1).contiguous()
    if count == batch_size * num_kv_heads:
        return scale.reshape(batch_size, num_kv_heads).contiguous()
    raise ValueError(
        "KV descale must be scalar, per-request, per-head, or per-request/per-head; "
        f"got shape {tuple(scale.shape)} for batch_size={batch_size}, "
        f"num_kv_heads={num_kv_heads}."
    )


class SparkinferAttnMetadata:
    """Lightweight metadata passed through forward context for sparkinfer."""

    __slots__ = (
        "mode",
        "page_table",
        "cache_seqlens",
        "cu_seqlens_q",
        "num_actual_tokens",
        "window_left",
    )

    def __init__(
        self,
        mode: str,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        num_actual_tokens: int,
        window_left: int = -1,
    ):
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
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
        mode: str = "extend",
    ) -> None:
        """Run extend/verify-mode attention (prefill or speculative verify)."""
        from sparkinfer.attention.paged._forward import paged_attention_forward
        from sparkinfer.attention.paged._scratch import build_paged_attention_binding
        from sparkinfer.attention.paged.planner import create_paged_plan
        from sparkinfer.attention.paged.workspace import PagedAttentionWorkspace

        if k_descale is None:
            k_descale = self._descale
        if v_descale is None:
            v_descale = self._descale

        ws = PagedAttentionWorkspace.for_tensors(
            mode=mode, q=q, k_cache=k_cache, v_cache=v_cache, use_cuda_graph=False
        )

        plan = create_paged_plan(
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            mode=mode,
            enable_cuda_graph=False,
            window_left=window_left,
        )
        ws._ensure_capacity(plan)
        ws._copy_runtime_metadata(page_table, cache_seqlens, cu_seqlens_q)
        ws._copy_plan_metadata(plan)
        ws._plan = plan

        binding = build_paged_attention_binding(
            scratch=ws,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            output=output,
            k_descale=k_descale,
            v_descale=v_descale,
        )
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
        page_size: int = PAGE_SIZE,
    ):
        from sparkinfer.attention.paged.workspace import PagedAttentionWorkspace

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_pages = max_pages
        self.window_left = window_left
        self.device = torch.device(device)
        self.page_size = page_size

        # Dummy tensors for workspace creation (real ones bound at capture)
        self._q = torch.zeros(1, num_q_heads, head_dim, dtype=torch.bfloat16, device=self.device)
        self._k_cache = torch.zeros(
            max_pages,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        self._v_cache = torch.zeros(
            max_pages,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        self._output = torch.zeros(
            1, num_q_heads, head_dim, dtype=torch.bfloat16, device=self.device
        )
        self._descale = torch.ones(1, dtype=torch.float32, device=self.device)
        self._k_descale = self._descale
        self._v_descale = self._descale

        # Create graph-mode workspace with prepare_decode_graph_replay_state.
        # Requires sparkinfer commit 0a7b143+ (fixes capacity underestimation
        # for windowed attention with small page counts).
        self._workspace = PagedAttentionWorkspace.for_tensors(
            mode="decode",
            q=self._q,
            k_cache=self._k_cache,
            v_cache=self._v_cache,
            use_cuda_graph=True,
        )
        self._workspace.prepare_decode_graph_replay_state(
            batch=1, max_page_table_width=max_pages, window_left=window_left
        )

        # Bind runtime metadata at max context for capture
        capture_page_table = torch.arange(
            max_pages, dtype=torch.int32, device=self.device
        ).unsqueeze(0)
        capture_cache_seqlens = torch.tensor(
            [max_pages * page_size - 1], dtype=torch.int32, device=self.device
        )
        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=self.device)
        self._workspace._copy_runtime_metadata(
            capture_page_table, capture_cache_seqlens, cu_seqlens_q
        )

        self._cu_seqlens_q = cu_seqlens_q
        logger.info(
            "SparkinferDecodeWorkspace: q_heads=%d kv_heads=%d head_dim=%d "
            "max_pages=%d window_left=%d",
            num_q_heads,
            num_kv_heads,
            head_dim,
            max_pages,
            window_left,
        )

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
            k_descale=self._k_descale,
            v_descale=self._v_descale,
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

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        window_left: int = -1,
        **kwargs,
    ):
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
        """Write K/V into paged cache (self-contained, zero vLLM dependency).

        kv_cache: [2, num_blocks, block_size, num_kv_heads, head_dim] (uint8/FP8)
        key/value: [num_tokens, num_kv_heads, head_dim] (bf16)
        slot_mapping: [num_tokens] (int64, flat index = block_idx * block_size + block_off)
        """
        k_cache = kv_cache[0].view(torch.float8_e4m3fn)
        v_cache = kv_cache[1].view(torch.float8_e4m3fn)
        reshape_and_cache_flash(
            key,
            value,
            k_cache,
            v_cache,
            slot_mapping,
            "fp8_e4m3",
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
        # KV cache: [2, num_blocks, block_size, num_kv_heads, head_dim]
        key_cache, value_cache = kv_cache.unbind(0)
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

        batch_size = int(attn_metadata.cache_seqlens.numel())
        num_kv_heads = int(key_cache.shape[2])
        k_descale = _paged_descale(
            layer._k_scale,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
        )
        v_descale = _paged_descale(
            layer._v_scale,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
        )

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
            k_descale=k_descale,
            v_descale=v_descale,
            mode=getattr(attn_metadata, "mode", "extend"),
        )
        return output
