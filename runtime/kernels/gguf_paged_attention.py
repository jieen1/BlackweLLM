"""Native F32 paged attention for the Qwen3.8 GGUF target.

The GGML reference keeps the Qwen3.8 full-attention reduction in F32.  The
existing SparkInfer and FlashInfer adapters intentionally expose BF16/FP16
query contracts, so using either one here would silently round the target
path before the attention reduction.  This module owns the missing contract:
one Triton launch per ``(query token, KV head)`` reads the physical pages
directly, performs grouped-query attention in F32, and writes the caller's
stable output buffer.

All runtime metadata is device-resident.  ``page_table``, ``cache_seqlens``,
``positions`` and (for compact verify) ``cu_seqlens_q`` are kernel inputs;
there is no host length readback or Python request loop in the captured path.
The launch is therefore safe to put inside a CUDA Graph and can serve decode,
uniform verify/prefill, and compact ragged verify with the same implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634
_PAGE_SIZE = 128
# Two F32 [K/V] tiles at D=256 must fit SM120's ~99 KiB shared-memory
# ceiling.  32 rows keeps the pair below the limit while still reusing each
# page row across the six grouped query heads.
_BLOCK_K = 32


@triton.jit
def _paged_f32_attention_kernel(
    query,
    k_cache,
    v_cache,
    output,
    page_table,
    cache_seqlens,
    positions,
    cu_seqlens_q,
    page_table_stride,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GQA: tl.constexpr,
    PADDED_GQA: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_PAGE_COUNT: tl.constexpr,
    MAX_TOKENS: tl.constexpr,
    FIXED_Q: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOG2E: tl.constexpr,
):
    """Online-softmax paged attention for one KV-head group.

    A program owns all ``GQA`` query heads belonging to one KV head.  This
    reuses every loaded K/V tile across the six Qwen3.8 query heads while
    keeping the reduction order stable and the temporary tile bounded.
    """

    query_row = tl.program_id(0)
    kv_head = tl.program_id(1)

    if FIXED_Q > 0:
        request = query_row // FIXED_Q
        query_valid = request < BATCH_SIZE
    else:
        request_ids = tl.arange(0, BATCH_SIZE)
        request_ends = tl.load(cu_seqlens_q + request_ids + 1)
        request = tl.sum(query_row >= request_ends, axis=0)
        request = tl.minimum(request, BATCH_SIZE - 1)
        total_query_rows = tl.load(cu_seqlens_q + BATCH_SIZE)
        query_valid = query_row < total_query_rows

    # The clamp keeps masked tail rows in-bounds.  Their stores are masked by
    # ``query_valid`` and therefore cannot affect a compact ragged replay.
    request = tl.minimum(request, BATCH_SIZE - 1)
    cache_len = tl.load(cache_seqlens + request)
    query_position = tl.load(positions + query_row)

    q_rows = tl.arange(0, PADDED_GQA)
    q_dims = tl.arange(0, HEAD_DIM)
    q_heads = kv_head * GQA + q_rows
    q_ptrs = (
        query + query_row * NUM_Q_HEADS * HEAD_DIM + q_heads[:, None] * HEAD_DIM + q_dims[None, :]
    )
    q_mask = (q_rows < GQA) & query_valid
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    running_max = tl.full((PADDED_GQA, 1), -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros((PADDED_GQA, 1), dtype=tl.float32)
    accumulator = tl.zeros((PADDED_GQA, HEAD_DIM), dtype=tl.float32)

    scale = 1.0 / (HEAD_DIM**0.5)
    for key_start in tl.range(0, MAX_TOKENS, BLOCK_K):
        key_rows = tl.arange(0, BLOCK_K)
        logical_rows = key_start + key_rows
        valid_rows = (logical_rows < cache_len) & (logical_rows <= query_position)
        page_index = logical_rows // PAGE_SIZE
        page_index = tl.minimum(page_index, MAX_PAGE_COUNT - 1)
        physical_page = tl.load(
            page_table + request * page_table_stride + page_index,
        )
        page_row = logical_rows % PAGE_SIZE
        cache_base = (
            physical_page * PAGE_SIZE + page_row
        ) * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
        k_ptrs = k_cache + cache_base[:, None] + q_dims[None, :]
        v_ptrs = v_cache + cache_base[:, None] + q_dims[None, :]
        k = tl.load(k_ptrs, mask=valid_rows[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=valid_rows[:, None], other=0.0).to(tl.float32)

        scores = (
            tl.dot(
                q,
                tl.trans(k),
                out_dtype=tl.float32,
                input_precision="ieee",
            )
            * scale
        )
        scores = tl.where(valid_rows[None, :], scores, -float("inf"))
        block_max = tl.max(scores, axis=1, keep_dims=True)
        new_max = tl.maximum(running_max, block_max)
        has_value = new_max != -float("inf")
        safe_max = tl.where(has_value, new_max, 0.0)
        rescale = tl.where(
            has_value,
            tl.exp2((running_max - safe_max) * LOG2E),
            0.0,
        )
        probabilities = tl.where(
            valid_rows[None, :],
            tl.exp2((scores - safe_max) * LOG2E),
            0.0,
        )
        running_sum = running_sum * rescale + tl.sum(probabilities, axis=1, keep_dims=True)
        accumulator = accumulator * rescale + tl.dot(
            probabilities,
            v,
            out_dtype=tl.float32,
            input_precision="ieee",
        )
        running_max = tl.where(has_value, new_max, running_max)

    denominator = tl.maximum(running_sum, 1.0e-20)
    result = accumulator / denominator
    output_ptrs = (
        output + query_row * NUM_Q_HEADS * HEAD_DIM + q_heads[:, None] * HEAD_DIM + q_dims[None, :]
    )
    # Ragged graph capacity includes scratch tail rows.  Write those rows as
    # zero rather than leaving the previous replay's bytes visible to the
    # following output projection.
    tl.store(output_ptrs, tl.where(query_valid, result, 0.0), mask=(q_rows < GQA)[:, None])


def paged_f32_attention(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    tokens_per_request: int | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run graph-safe F32 paged attention and return ``output``.

    ``query``/``output`` are token-major ``[total_q, q_heads, head_dim]``;
    pools are ``[physical_pages, page_size, kv_heads, head_dim]``.  A fixed
    ``tokens_per_request`` selects uniform decode/prefill/verify.  Passing
    ``None`` selects compact ragged verify and uses ``cu_seqlens_q`` to map
    every query row to its request without a host synchronization.
    """

    if page_size != _PAGE_SIZE:
        raise ValueError(f"Qwen3.8 F32 attention requires page_size={_PAGE_SIZE}, got {page_size}")
    if query.ndim != 3 or query.shape[1:] != (num_q_heads, head_dim):
        raise ValueError(
            "Qwen3.8 F32 attention query must be "
            f"[tokens,{num_q_heads},{head_dim}], got {tuple(query.shape)}"
        )
    expected_pool_shape = (page_size, num_kv_heads, head_dim)
    if k_cache.ndim != 4 or tuple(k_cache.shape[1:]) != expected_pool_shape:
        raise ValueError(
            "Qwen3.8 F32 attention K pool must have trailing shape "
            f"{expected_pool_shape}, got {tuple(k_cache.shape)}"
        )
    if v_cache.shape != k_cache.shape:
        raise ValueError("Qwen3.8 F32 attention K/V pools must have identical shapes")
    if (
        query.dtype != torch.float32
        or k_cache.dtype != torch.float32
        or v_cache.dtype != torch.float32
    ):
        raise TypeError("Qwen3.8 F32 attention requires F32 query and K/V pools")
    if (
        query.device.type != "cuda"
        or k_cache.device != query.device
        or v_cache.device != query.device
    ):
        raise ValueError("Qwen3.8 F32 attention requires all tensors on one CUDA device")
    if page_table.ndim != 2 or cache_seqlens.ndim != 1 or positions.shape != (query.shape[0],):
        raise ValueError("Qwen3.8 F32 attention metadata has incompatible shapes")
    batch_size, page_count = page_table.shape
    if cache_seqlens.shape != (batch_size,):
        raise ValueError("cache_seqlens must have one entry per page-table row")
    if tokens_per_request is not None:
        fixed_q = int(tokens_per_request)
        if fixed_q <= 0 or query.shape[0] != batch_size * fixed_q:
            raise ValueError("fixed F32 attention query shape does not match batch geometry")
    else:
        fixed_q = 0
        if cu_seqlens_q is None or cu_seqlens_q.shape != (batch_size + 1,):
            raise ValueError("ragged F32 attention requires cu_seqlens_q with B+1 entries")

    if output is None:
        output = torch.empty_like(query)
    elif (
        output.shape != query.shape
        or output.dtype != torch.float32
        or output.device != query.device
    ):
        raise ValueError("F32 attention output must match query shape, dtype, and device")

    gqa = num_q_heads // num_kv_heads
    if gqa * num_kv_heads != num_q_heads:
        raise ValueError("Qwen3.8 F32 attention requires an integer GQA ratio")
    padded_gqa = max(8, (gqa + 7) // 8 * 8)
    max_tokens = int(page_count) * page_size
    _paged_f32_attention_kernel[(query.shape[0], num_kv_heads)](
        query,
        k_cache,
        v_cache,
        output,
        page_table,
        cache_seqlens,
        positions,
        cu_seqlens_q if cu_seqlens_q is not None else page_table,
        page_table.stride(0),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        GQA=gqa,
        PADDED_GQA=padded_gqa,
        PAGE_SIZE=page_size,
        MAX_PAGE_COUNT=page_count,
        MAX_TOKENS=max_tokens,
        FIXED_Q=fixed_q,
        BATCH_SIZE=batch_size,
        BLOCK_K=_BLOCK_K,
        LOG2E=_LOG2E,
        num_warps=4,
    )
    return output
