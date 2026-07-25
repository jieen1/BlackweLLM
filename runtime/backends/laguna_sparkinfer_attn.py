"""SparkInfer paged attention adapter for Laguna CG decode.

Replaces FlashInfer BatchDecodeWithPagedKVCacheWrapper with SparkInfer's
PagedAttentionWorkspace + on-device metadata rebuild.

Key advantage: metadata update is captured IN the CUDA graph.
Per-step cost: 1 GPU int32 write (cache_seqlens) + graph.replay().
No CPU plan, no H2D copies, no Python dispatch per step.

Requires page_size=64 (SparkInfer constraint).
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

import torch

logger = logging.getLogger("qwen_sm120_runtime.sparkinfer_attn")

_BF_SPARKINFER_PATH = os.environ.get("BF_SPARKINFER_PATH", "")
if _BF_SPARKINFER_PATH and _BF_SPARKINFER_PATH not in sys.path:
    sys.path.insert(0, _BF_SPARKINFER_PATH)

PAGE_SIZE = 64  # SparkInfer hard requirement


class SparkinferDecodeAttention:
    """Manages SparkInfer paged attention for one layer group in CG decode.

    Lifecycle:
      1. __init__: allocate workspace, prepare decode graph replay state
      2. setup_for_capture: load capture plan, bind tensors
      3. (caller captures CUDA graph around `forward()`)
      4. update_seqlen: per-step cache_seqlens update (GPU write)
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
        self._captured = False

        # Dummy tensors for workspace creation (real ones bound at capture)
        self._q = torch.zeros(1, num_q_heads, head_dim, dtype=torch.bfloat16, device=self.device)
        self._k_cache = torch.zeros(max_pages, PAGE_SIZE, num_kv_heads, head_dim,
                                    dtype=torch.float8_e4m3fn, device=self.device)
        self._v_cache = torch.zeros(max_pages, PAGE_SIZE, num_kv_heads, head_dim,
                                    dtype=torch.float8_e4m3fn, device=self.device)
        self._output = torch.zeros(1, num_q_heads, head_dim, dtype=torch.bfloat16, device=self.device)
        self._descale = torch.ones(1, dtype=torch.float32, device=self.device)

        # Create workspace
        self._workspace = PagedAttentionWorkspace.for_tensors(
            mode="decode", q=self._q, k_cache=self._k_cache,
            v_cache=self._v_cache, use_cuda_graph=True)
        self._workspace.prepare_decode_graph_replay_state(
            batch=1, max_page_table_width=max_pages, window_left=window_left)

        # Capture plan (at max context)
        capture_page_table = torch.arange(max_pages, dtype=torch.int32, device=self.device).unsqueeze(0)
        capture_cache_seqlens = torch.tensor([max_pages * PAGE_SIZE - 1], dtype=torch.int32, device=self.device)
        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=self.device)

        capture_plan = create_paged_plan(
            self._q, self._k_cache, self._v_cache,
            capture_page_table, capture_cache_seqlens, cu_seqlens_q,
            mode="decode", enable_cuda_graph=True, graph_chunk_policy=True)
        self._workspace._ensure_capacity(capture_plan)
        self._workspace._copy_runtime_metadata(capture_page_table, capture_cache_seqlens, cu_seqlens_q)
        self._workspace._copy_plan_metadata(capture_plan)
        self._workspace._plan = capture_plan

        self._cu_seqlens_q = cu_seqlens_q
        logger.info(
            "SparkinferDecodeAttention: q_heads=%d kv_heads=%d head_dim=%d "
            "max_pages=%d window_left=%d",
            num_q_heads, num_kv_heads, head_dim, max_pages, window_left)

    def bind_kv(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
    ) -> None:
        """Bind real tensors (called before CG capture)."""
        self._q = q
        self._k_cache = k_cache
        self._v_cache = v_cache
        self._output = output
        self._descale_k = k_descale or self._descale
        self._descale_v = v_descale or self._descale

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
            k_descale=self._descale_k,
            v_descale=self._descale_v,
        )
        paged_attention_forward(binding=binding)
        return self._output

    @property
    def cache_seqlens(self) -> torch.Tensor:
        """GPU tensor to update per-step (just write cache_seqlens[0] = new_kv_len)."""
        return self._workspace.cache_seqlens

    def update_seqlen_gpu(self, seqlen_tensor: torch.Tensor) -> None:
        """Update cache_seqlens from a pre-computed GPU tensor (GPU→GPU copy)."""
        self._workspace.cache_seqlens[:1] = seqlen_tensor
