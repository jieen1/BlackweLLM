"""BlackForge Attention — self-developed replacement for vLLM's Attention class.

Handles KV cache write (FP8 quantization + paged scatter) and sparkinfer
attention forward directly. Zero dependency on vLLM's unified_kv_cache_update,
unified_attention_with_output, or get_attention_context dispatch.

Metadata (slot_mapping, page_table, seq_lens) is passed via a lightweight
thread-local context set by LagunaBackend before each model.forward() call.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

from runtime.kernels.fused_kv_scatter import fused_kv_scatter

logger = logging.getLogger("qwen_sm120_runtime.bf_attention")

# ── Lightweight forward context (replaces vLLM's ForwardContext for attention) ──
_ctx = threading.local()


class BFAttnContext:
    """Per-forward-call context for BFAttention layers."""

    __slots__ = ("attn_metadata", "slot_mapping")

    def __init__(
        self,
        attn_metadata: dict[str, Any],
        slot_mapping: dict[str, torch.Tensor],
    ):
        self.attn_metadata = attn_metadata
        self.slot_mapping = slot_mapping


def set_bf_attn_context(
    attn_metadata: dict[str, Any],
    slot_mapping: dict[str, torch.Tensor],
) -> None:
    """Set the attention context for the current forward call."""
    _ctx.current = BFAttnContext(attn_metadata, slot_mapping)


def get_bf_attn_context() -> BFAttnContext:
    """Get the current attention context."""
    context = getattr(_ctx, "current", None)
    if context is None:
        raise RuntimeError(
            "BFAttention was called without a scoped attention context. "
            "Wrap model.forward() in bf_attn_context()."
        )
    return context


def clear_bf_attn_context() -> None:
    _ctx.current = None


@contextmanager
def bf_attn_context(
    attn_metadata: dict[str, Any],
    slot_mapping: dict[str, torch.Tensor],
):
    """Scope BFAttention metadata to one model forward call.

    A forward that is not explicitly scoped must fail instead of reusing
    metadata from a previous request. Nested forwards restore their caller's
    context when they complete.
    """
    previous = getattr(_ctx, "current", None)
    set_bf_attn_context(attn_metadata, slot_mapping)
    try:
        yield
    finally:
        _ctx.current = previous


class BFAttention(nn.Module):
    """Drop-in replacement for vLLM's Attention module.

    Stores its own KV cache reference and sparkinfer impl. On forward:
    1. Reshape Q/K/V
    2. Write K/V to paged FP8 cache (scatter by slot_mapping)
    3. Run sparkinfer paged attention
    4. Return [M, num_heads * head_dim]
    """

    def __init__(
        self,
        layer_name: str,
        num_heads: int,
        head_size: int,
        num_kv_heads: int,
        scale: float,
        window_left: int = -1,
        kv_cache_dtype: str = "fp8_e4m3",
    ):
        super().__init__()
        self.layer_name = layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.head_size_v = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.window_left = window_left
        self.kv_cache_dtype = kv_cache_dtype

        # Set after model load + KV cache allocation
        self.kv_cache: torch.Tensor | None = None
        self._k_scale: torch.Tensor = torch.ones(1, dtype=torch.float32, device="cuda")
        self._v_scale: torch.Tensor = torch.ones(1, dtype=torch.float32, device="cuda")

        # Sparkinfer impl
        from runtime.backends.laguna_sparkinfer_attn import SparkinferAttentionImpl

        self.impl = SparkinferAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            window_left=window_left,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = query.dtype
        num_tokens = query.shape[0]
        if output_shape is None:
            output_shape = torch.Size((num_tokens, self.num_heads * self.head_size_v))
        output = torch.empty(output_shape, dtype=output_dtype, device=query.device)

        # Reshape: [M, hidden] -> [M, heads, dim]
        q = query.view(-1, self.num_heads, self.head_size)
        k = key.view(-1, self.num_kv_heads, self.head_size) if key is not None else None
        v = value.view(-1, self.num_kv_heads, self.head_size_v) if value is not None else None
        out = output.view(-1, self.num_heads, self.head_size_v)

        # Get per-call metadata
        ctx = get_bf_attn_context()
        sm = ctx.slot_mapping.get(self.layer_name)
        meta = ctx.attn_metadata.get(self.layer_name)
        if sm is None or meta is None:
            raise RuntimeError(
                f"BFAttention context is missing metadata or slot mapping for {self.layer_name!r}."
            )

        # KV cache write via self-developed fused Triton kernel (replaces
        # vLLM's compiled C++ reshape_and_cache_flash). Single kernel per
        # layer instead of 6 Python ops (288→48 kernels/step).
        if sm is not None and k is not None and v is not None and self.kv_cache is not None:
            k_cache = self.kv_cache[0]
            v_cache = self.kv_cache[1]
            if k_cache.dtype == torch.uint8:
                k_cache = k_cache.view(torch.float8_e4m3fn)
                v_cache = v_cache.view(torch.float8_e4m3fn)
            fused_kv_scatter(k, v, k_cache, v_cache, sm, self._k_scale, self._v_scale)

        # Sparkinfer attention forward
        self.impl.forward(self, q, k, v, self.kv_cache, meta, out)

        return output


def replace_vllm_attention(
    model: nn.Module,
    sfc: dict[str, Any],
    kv_caches: dict[str, torch.Tensor],
) -> int:
    """Replace all vLLM Attention modules in model with BFAttention.

    Walks the model tree, finds Attention instances registered in sfc,
    and replaces them with BFAttention that has the same config + our
    sparkinfer impl + self-allocated KV cache.

    Returns number of layers replaced.
    """
    replaced = 0

    for layer_name, attn_layer in sfc.items():
        if not hasattr(attn_layer, "get_attn_backend"):
            continue

        # Read config from the original vLLM Attention layer
        num_heads = attn_layer.num_heads
        head_size = attn_layer.head_size
        num_kv_heads = attn_layer.num_kv_heads
        scale = attn_layer.impl.scale if hasattr(attn_layer.impl, "scale") else head_size**-0.5
        window_left = getattr(attn_layer.impl, "window_left", -1)

        # Create BFAttention replacement
        bf_attn = BFAttention(
            layer_name=layer_name,
            num_heads=num_heads,
            head_size=head_size,
            num_kv_heads=num_kv_heads,
            scale=scale,
            window_left=window_left,
            kv_cache_dtype=getattr(attn_layer, "kv_cache_dtype", "fp8_e4m3"),
        )

        # Copy KV scales from original layer
        if hasattr(attn_layer, "_k_scale"):
            bf_attn._k_scale = attn_layer._k_scale
        if hasattr(attn_layer, "_v_scale"):
            bf_attn._v_scale = attn_layer._v_scale

        # Bind KV cache
        bf_attn.kv_cache = kv_caches[layer_name]

        # Replace in the model tree: find parent module and attribute name
        parts = layer_name.split(".")
        parent = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        attr_name = parts[-1]
        setattr(parent, attr_name, bf_attn)

        # Also update sfc to point to our BFAttention
        sfc[layer_name] = bf_attn

        replaced += 1

    logger.info("Replaced %d vLLM Attention modules with BFAttention", replaced)
    return replaced
