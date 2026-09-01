"""Fused sparse QSA decode attention for Flash-Next.

The torch bring-up path gathered ``[rows, selected, kv_heads, head_dim]`` and
then repeated the two KV heads twelve times before two FP32 einsums.  At the
production width (4 verify rows, 2051 selected tokens) that creates several
200 MiB temporaries per QSA layer.  This kernel keeps GQA grouped: one program
loads a KV head and evaluates all twelve query heads with an online softmax.
"""

from __future__ import annotations

import torch

try:  # Triton is optional for the torch-free CI interpreter.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

try:  # TileLang is supplied by the SM120 b12x runtime environment.
    import tilelang
    from tilelang import language as T
except ImportError:  # pragma: no cover
    tilelang = None
    T = None


def qsa_sparse_attention_supported(q: torch.Tensor) -> bool:
    return triton is not None and q.is_cuda and q.dtype == torch.bfloat16


def qsa_mqa_prefill_supported(q: torch.Tensor) -> bool:
    return tilelang is not None and q.is_cuda and q.dtype == torch.bfloat16


def qsa_prefill_gather_indices_supported(block_indices: torch.Tensor) -> bool:
    return triton is not None and block_indices.is_cuda


def _qsa_mqa_block_q(rows: int, heads: int) -> int:
    """Choose a TensorCore-friendly query tile without padding decode rows.

    The original QSA kernel always used ``block_q=128 // heads``.  That is a
    good prefill tile, but verify only supplies four rows (and decode supplies
    one), so the kernel performed the GEMM and reduction for 32 padded rows
    before slicing the result back to the live rows.  Keep at least 32 query
    lanes per GEMM (the SM120 warp layout requires that width), then round to
    a power-of-two tile so TileLang gets a small, reusable set of compiled
    variants.  For unusual head counts that do not divide the 128-wide base
    tile, retain the validated legacy choice rather than guessing at a warp
    layout.
    """
    if rows <= 0 or heads <= 0:
        raise ValueError(f"rows and heads must be positive, got rows={rows}, heads={heads}")
    base = max(1, 128 // heads)
    if 128 % heads:
        return base

    minimum = max(1, (32 + heads - 1) // heads)
    target = max(rows, minimum)
    tile = 1 << (target - 1).bit_length()
    return min(base, tile)


if tilelang is not None:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_qsa_mqa_prefill_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 64,
        block_q: int = 32,
        num_stages: int = 3,
        threads: int = 512,
    ):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            query: T.Tensor([rows * heads, head_dim], T.bfloat16),
            key: T.Tensor([keys, head_dim], T.bfloat16),
            logits: T.Tensor([rows, keys], T.float32),
            starts: T.Tensor([rows], T.int32),
            ends: T.Tensor([rows], T.int32),
        ):
            with T.Kernel(T.ceildiv(rows, block_q), threads=threads) as block:
                query_shared = T.alloc_shared([block_q * heads, head_dim], T.bfloat16)
                key_shared = T.alloc_shared([block_n, head_dim], T.bfloat16)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                row_base = block * block_q
                start_min = T.alloc_var(T.int32)
                end_max = T.alloc_var(T.int32)
                start_min = 2147483647
                end_max = -2147483648
                for row in T.serial(block_q):
                    start_min = T.min(
                        start_min,
                        T.min(starts[row_base + row], keys),
                    )
                    end_max = T.max(
                        end_max,
                        T.min(ends[row_base + row], keys),
                    )

                T.copy(query[row_base * heads, 0], query_shared)
                for key_block in T.Pipelined(
                    T.ceildiv(end_max - start_min, block_n),
                    num_stages=num_stages,
                ):
                    T.copy(
                        key[start_min + key_block * block_n, 0],
                        key_shared,
                    )
                    T.gemm(
                        key_shared,
                        query_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for key_row, query_row, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[key_row, query_row, head] = T.max(
                            scores_3d[key_row, query_row, head],
                            0.0,
                        )
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for query_row, key_row in T.Parallel(block_q, block_n):
                        logits[
                            row_base + query_row,
                            start_min + key_block * block_n + key_row,
                        ] = reduced[key_row, query_row]

        return kernel

    @tilelang.jit
    def _tilelang_qsa_mqa_mask_kernel(threads: int = 512, block_k: int = 4096):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            logits: T.Tensor([rows, keys], T.float32),
            starts: T.Tensor([rows], T.int32),
            ends: T.Tensor([rows], T.int32),
        ):
            with T.Kernel(rows, threads=threads) as block:
                thread = T.thread_binding(0, threads, thread="threadIdx.x")
                for key_block in T.Pipelined(T.ceildiv(keys, block_k)):
                    for item in T.serial(block_k // threads):
                        column = key_block * block_k + item * threads + thread
                        if column < starts[block] or column >= ends[block]:
                            logits[block, column] = -T.infinity(T.float32)

        return kernel


def qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_ends: torch.Tensor,
) -> torch.Tensor:
    """Score compressed QSA keys with FP32 accumulation on SM120 tensor cores."""
    if not qsa_mqa_prefill_supported(q):
        raise RuntimeError("QSA MQA prefill requires CUDA BF16 with TileLang")
    if q.ndim != 3 or k.ndim != 2 or q.shape[-1] != k.shape[-1]:
        raise ValueError(
            "QSA MQA expects q [rows,heads,dim] and k [keys,dim], got "
            f"{tuple(q.shape)} and {tuple(k.shape)}"
        )
    rows, heads, head_dim = q.shape
    keys = k.shape[0]
    if row_ends.shape != (rows,):
        raise ValueError(f"QSA row ends must have shape ({rows},), got {tuple(row_ends.shape)}")
    if rows == 0 or keys == 0:
        return torch.full(
            (rows, keys),
            -float("inf"),
            dtype=torch.float32,
            device=q.device,
        )

    block_q = _qsa_mqa_block_q(rows, heads)
    padding = (-rows) % block_q
    padded_rows = rows + padding
    logits = torch.empty(
        (padded_rows, keys),
        dtype=torch.float32,
        device=q.device,
    )
    query = q.contiguous()
    starts = torch.zeros(padded_rows, dtype=torch.int32, device=q.device)
    ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    if padding:
        query = torch.cat(
            [query, query.new_zeros(padding, heads, head_dim)],
            dim=0,
        )
        ends = torch.cat([ends, ends[-1:].expand(padding)])

    _tilelang_qsa_mqa_prefill_kernel(
        heads=heads,
        head_dim=head_dim,
        block_q=block_q,
    )(
        query.reshape(-1, head_dim),
        k.contiguous(),
        logits,
        starts,
        ends,
    )
    logits = logits[:rows]
    logits.mul_(head_dim**-0.5)
    _tilelang_qsa_mqa_mask_kernel()(logits, starts[:rows], ends[:rows])
    return logits


if triton is not None:

    @triton.jit
    def _qsa_prefill_gather_indices_kernel(
        block_indices_ptr,
        positions_ptr,
        tokens_ptr,
        valid_ptr,
        selected_counts_ptr,
        block_stride_row: tl.constexpr,
        output_stride_row: tl.constexpr,
        block_topk: tl.constexpr,
        pad_to: tl.constexpr,
        compress_ratio: tl.constexpr,
        block_output: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, block_output)
        visible = tl.load(positions_ptr + row) + 1
        complete_groups = visible // compress_ratio
        valid_blocks = tl.minimum(complete_groups, block_topk)
        selected_tokens = valid_blocks * compress_ratio

        source_block = columns // compress_ratio
        offsets = columns % compress_ratio
        blocks = tl.load(
            block_indices_ptr + row * block_stride_row + source_block,
            mask=(columns < selected_tokens) & (source_block < block_topk),
            other=-1,
        )
        expanded = blocks * compress_ratio + offsets
        expanded_valid = (
            (columns < selected_tokens) & (blocks >= 0) & (expanded >= 0) & (expanded < visible)
        )

        tail_offset = columns - selected_tokens
        tail_start = complete_groups * compress_ratio
        tail_count = visible - tail_start
        tail = tail_start + tail_offset
        tail_valid = (
            (tail_offset >= 0)
            & (tail_offset < tail_count)
            & (tail_offset < compress_ratio - 1)
            & (columns < pad_to)
        )
        lane_valid = expanded_valid | tail_valid
        token = tl.where(expanded_valid, expanded, tl.where(tail_valid, tail, 0))
        tl.store(
            tokens_ptr + row * output_stride_row + columns,
            token,
            mask=columns < pad_to,
        )
        tl.store(
            valid_ptr + row * output_stride_row + columns,
            lane_valid,
            mask=columns < pad_to,
        )
        tl.store(
            selected_counts_ptr + row,
            tl.minimum(selected_tokens + tail_count, pad_to),
        )

    @triton.jit
    def _qsa_sparse_attention_kernel(
        q_ptr,
        gate_ptr,
        k_ptr,
        v_ptr,
        idx_ptr,
        valid_ptr,
        out_ptr,
        selected_counts_ptr,
        q_stride_row: tl.constexpr,
        q_stride_head: tl.constexpr,
        q_stride_dim: tl.constexpr,
        gate_stride_row: tl.constexpr,
        gate_stride_head: tl.constexpr,
        gate_stride_dim: tl.constexpr,
        kv_stride_token: tl.constexpr,
        kv_stride_head: tl.constexpr,
        kv_stride_dim: tl.constexpr,
        k_scale_ptr,
        v_scale_ptr,
        k_scale_stride_token: tl.constexpr,
        k_scale_stride_head: tl.constexpr,
        v_scale_stride_token: tl.constexpr,
        v_scale_stride_head: tl.constexpr,
        idx_stride_row: tl.constexpr,
        idx_stride_col: tl.constexpr,
        valid_stride_row: tl.constexpr,
        valid_stride_col: tl.constexpr,
        out_stride_row: tl.constexpr,
        out_stride_head: tl.constexpr,
        out_stride_dim: tl.constexpr,
        num_kv_heads: tl.constexpr,
        gqa: tl.constexpr,
        head_dim: tl.constexpr,
        selected: tl.constexpr,
        block_gqa: tl.constexpr,
        block_k: tl.constexpr,
        limit_selected: tl.constexpr,
        kv_is_quantized: tl.constexpr,
    ):
        program = tl.program_id(0)
        row = program // num_kv_heads
        kv_head = program % num_kv_heads

        gqa_offsets = tl.arange(0, block_gqa)
        dim_offsets = tl.arange(0, head_dim)
        gqa_mask = gqa_offsets < gqa
        query_head = kv_head * gqa + gqa_offsets
        q = tl.load(
            q_ptr
            + row * q_stride_row
            + query_head[:, None] * q_stride_head
            + dim_offsets[None, :] * q_stride_dim,
            mask=gqa_mask[:, None],
            other=0.0,
        )

        # Start from a finite reference exponent.  The fixed-width decode
        # gather contains a long tail of invalid padding lanes (at position
        # zero only one of 2051 lanes is valid).  Keeping ``-inf`` here makes
        # an all-invalid block evaluate ``-inf - -inf`` below, poisoning the
        # online softmax with NaNs before the next valid block can recover.
        # A zero baseline is algebraically identical to the usual max=0
        # softmax formulation and keeps empty blocks a no-op.
        running_max = tl.zeros((block_gqa, 1), tl.float32)
        running_sum = tl.zeros((block_gqa, 1), tl.float32)
        accumulator = tl.zeros((block_gqa, head_dim), tl.float32)
        scale = 1.0 / (head_dim**0.5)

        row_selected = selected
        if limit_selected:
            row_selected = tl.load(selected_counts_ptr + row)
        for start in range(0, row_selected, block_k):
            key_offsets = start + tl.arange(0, block_k)
            in_range = key_offsets < selected
            token_idx = tl.load(
                idx_ptr + row * idx_stride_row + key_offsets * idx_stride_col,
                mask=in_range,
                other=0,
            )
            lane_valid = in_range & tl.load(
                valid_ptr + row * valid_stride_row + key_offsets * valid_stride_col,
                mask=in_range,
                other=0,
            )
            kv_offsets = (
                token_idx[:, None] * kv_stride_token
                + kv_head * kv_stride_head
                + dim_offsets[None, :] * kv_stride_dim
            )
            keys = tl.load(k_ptr + kv_offsets, mask=lane_valid[:, None], other=0.0)
            values = tl.load(v_ptr + kv_offsets, mask=lane_valid[:, None], other=0.0)
            if kv_is_quantized:
                k_scales = tl.load(
                    k_scale_ptr
                    + token_idx * k_scale_stride_token
                    + kv_head * k_scale_stride_head,
                    mask=lane_valid,
                    other=1.0,
                ).to(tl.float32)
                v_scales = tl.load(
                    v_scale_ptr
                    + token_idx * v_scale_stride_token
                    + kv_head * v_scale_stride_head,
                    mask=lane_valid,
                    other=1.0,
                ).to(tl.float32)
                keys = keys.to(tl.bfloat16)
                values = values.to(tl.bfloat16)
                keys = keys * k_scales[:, None].to(tl.bfloat16)
                values = values * v_scales[:, None].to(tl.bfloat16)

            scores = tl.dot(q, tl.trans(keys), out_dtype=tl.float32) * scale
            scores = tl.where(gqa_mask[:, None] & lane_valid[None, :], scores, -float("inf"))
            new_max = tl.maximum(running_max, tl.max(scores, axis=1, keep_dims=True))
            old_scale = tl.exp2((running_max - new_max) * 1.4426950408889634)
            probabilities = tl.exp2((scores - new_max) * 1.4426950408889634)
            running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1, keep_dims=True)
            accumulator = accumulator * old_scale + tl.dot(
                probabilities.to(tl.bfloat16),
                values,
                out_dtype=tl.float32,
            )
            running_max = new_max

        output = accumulator / running_sum
        gate = tl.load(
            gate_ptr
            + row * gate_stride_row
            + query_head[:, None] * gate_stride_head
            + dim_offsets[None, :] * gate_stride_dim,
            mask=gqa_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        output *= tl.sigmoid(gate)
        tl.store(
            out_ptr
            + row * out_stride_row
            + query_head[:, None] * out_stride_head
            + dim_offsets[None, :] * out_stride_dim,
            output,
            mask=gqa_mask[:, None],
        )


def qsa_prefill_gather_indices(
    block_indices: torch.Tensor,
    positions: torch.Tensor,
    pad_to: int,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand QSA blocks and append the partial causal tail in one launch."""
    if triton is None or not block_indices.is_cuda:
        raise RuntimeError("QSA prefill gather indices require CUDA with Triton")
    if block_indices.ndim != 2 or positions.shape != (block_indices.shape[0],):
        raise ValueError("QSA blocks/positions must be [rows, topk] and [rows]")
    rows, block_topk = block_indices.shape
    tokens = torch.empty(rows, pad_to, dtype=torch.long, device=block_indices.device)
    valid = torch.empty(rows, pad_to, dtype=torch.bool, device=block_indices.device)
    selected_counts = torch.empty(rows, dtype=torch.int32, device=block_indices.device)
    if rows:
        _qsa_prefill_gather_indices_kernel[(rows,)](
            block_indices,
            positions,
            tokens,
            valid,
            selected_counts,
            block_stride_row=block_indices.stride(0),
            output_stride_row=tokens.stride(0),
            block_topk=block_topk,
            pad_to=pad_to,
            compress_ratio=compress_ratio,
            block_output=triton.next_power_of_2(pad_to),
            num_warps=8,
        )
    return tokens, valid, selected_counts


def qsa_sparse_attention(
    q: torch.Tensor,
    gate: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx: torch.Tensor,
    valid: torch.Tensor,
    k_scales: torch.Tensor | None = None,
    v_scales: torch.Tensor | None = None,
    selected_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run graph-safe grouped QSA attention without expanding KV heads.

    ``selected_counts`` is optional for compatibility with callers that pass
    arbitrary sparse masks.  Decode/verify gatherers pack valid lanes at the
    front of their fixed-width row, so supplying the exact valid prefix lets
    the kernel stop before scanning the padding tail (up to 2051 lanes at
    long context) without changing the attention math.
    """
    if not qsa_sparse_attention_supported(q):
        raise RuntimeError("fused QSA attention requires CUDA BF16 with Triton")
    if q.ndim != 3 or gate.shape != q.shape:
        raise ValueError("QSA q and gate must have matching [rows, heads, dim] shapes")
    if idx.ndim != 2 or valid.shape != idx.shape or idx.shape[0] != q.shape[0]:
        raise ValueError("QSA idx/valid must be matching [rows, selected] tensors")
    if selected_counts is not None:
        if selected_counts.shape != (q.shape[0],):
            raise ValueError(
                "QSA selected counts must have shape "
                f"({q.shape[0]},), got {tuple(selected_counts.shape)}"
            )
        if selected_counts.device != idx.device:
            raise ValueError("QSA selected counts must be on the index device")
        if selected_counts.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "QSA selected counts must use int32 or int64, "
                f"got {selected_counts.dtype}"
            )
    if k_cache.shape != v_cache.shape or k_cache.ndim != 3:
        raise ValueError("QSA K/V caches must have matching [tokens, kv_heads, dim] shapes")
    rows, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    if k_cache.shape[2] != head_dim or num_heads % num_kv_heads:
        raise ValueError("incompatible QSA query and KV head geometry")
    gqa = num_heads // num_kv_heads
    if gqa > 16 or head_dim != 256:
        raise ValueError(f"unsupported QSA geometry: gqa={gqa}, head_dim={head_dim}")

    output = torch.empty_like(q)
    if k_scales is None:
        k_scales = torch.ones(1, dtype=torch.float16, device=k_cache.device)
    if v_scales is None:
        v_scales = torch.ones(1, dtype=torch.float16, device=v_cache.device)
    _qsa_sparse_attention_kernel[(rows * num_kv_heads,)](
        q,
        gate,
        k_cache,
        v_cache,
        idx,
        valid,
        output,
        selected_counts if selected_counts is not None else valid,
        q_stride_row=q.stride(0),
        q_stride_head=q.stride(1),
        q_stride_dim=q.stride(2),
        gate_stride_row=gate.stride(0),
        gate_stride_head=gate.stride(1),
        gate_stride_dim=gate.stride(2),
        kv_stride_token=k_cache.stride(0),
        kv_stride_head=k_cache.stride(1),
        kv_stride_dim=k_cache.stride(2),
        k_scale_ptr=k_scales,
        v_scale_ptr=v_scales,
        k_scale_stride_token=k_scales.stride(0) if k_scales.ndim > 1 else 0,
        k_scale_stride_head=k_scales.stride(1) if k_scales.ndim > 1 else 0,
        v_scale_stride_token=v_scales.stride(0) if v_scales.ndim > 1 else 0,
        v_scale_stride_head=v_scales.stride(1) if v_scales.ndim > 1 else 0,
        idx_stride_row=idx.stride(0),
        idx_stride_col=idx.stride(1),
        valid_stride_row=valid.stride(0),
        valid_stride_col=valid.stride(1),
        out_stride_row=output.stride(0),
        out_stride_head=output.stride(1),
        out_stride_dim=output.stride(2),
        num_kv_heads=num_kv_heads,
        gqa=gqa,
        head_dim=head_dim,
        selected=idx.shape[1],
        block_gqa=16,
        block_k=32,
        limit_selected=selected_counts is not None,
        kv_is_quantized=k_cache.dtype
        in {torch.int8, getattr(torch, "float8_e4m3fn", None)},
        num_warps=8,
        num_stages=2,
    )
    return output


def qsa_sparse_prefill_attention(
    q: torch.Tensor,
    gate: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx: torch.Tensor,
    valid: torch.Tensor,
    selected_counts: torch.Tensor,
    k_scales: torch.Tensor | None = None,
    v_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Grouped sparse QSA with a per-row loop bound for causal prefill."""
    if not qsa_sparse_attention_supported(q):
        raise RuntimeError("fused QSA prefill requires CUDA BF16 with Triton")
    if idx.ndim != 2 or valid.shape != idx.shape or idx.shape[0] != q.shape[0]:
        raise ValueError("QSA idx/valid must be matching [rows, selected] tensors")
    if selected_counts.shape != (q.shape[0],):
        raise ValueError(
            f"QSA selected counts must have shape ({q.shape[0]},), "
            f"got {tuple(selected_counts.shape)}"
        )
    rows, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    gqa = num_heads // num_kv_heads
    output = torch.empty_like(q)
    if k_scales is None:
        k_scales = torch.ones(1, dtype=torch.float16, device=k_cache.device)
    if v_scales is None:
        v_scales = torch.ones(1, dtype=torch.float16, device=v_cache.device)
    _qsa_sparse_attention_kernel[(rows * num_kv_heads,)](
        q,
        gate,
        k_cache,
        v_cache,
        idx,
        valid,
        output,
        selected_counts,
        q_stride_row=q.stride(0),
        q_stride_head=q.stride(1),
        q_stride_dim=q.stride(2),
        gate_stride_row=gate.stride(0),
        gate_stride_head=gate.stride(1),
        gate_stride_dim=gate.stride(2),
        kv_stride_token=k_cache.stride(0),
        kv_stride_head=k_cache.stride(1),
        kv_stride_dim=k_cache.stride(2),
        k_scale_ptr=k_scales,
        v_scale_ptr=v_scales,
        k_scale_stride_token=k_scales.stride(0) if k_scales.ndim > 1 else 0,
        k_scale_stride_head=k_scales.stride(1) if k_scales.ndim > 1 else 0,
        v_scale_stride_token=v_scales.stride(0) if v_scales.ndim > 1 else 0,
        v_scale_stride_head=v_scales.stride(1) if v_scales.ndim > 1 else 0,
        idx_stride_row=idx.stride(0),
        idx_stride_col=idx.stride(1),
        valid_stride_row=valid.stride(0),
        valid_stride_col=valid.stride(1),
        out_stride_row=output.stride(0),
        out_stride_head=output.stride(1),
        out_stride_dim=output.stride(2),
        num_kv_heads=num_kv_heads,
        gqa=gqa,
        head_dim=head_dim,
        selected=idx.shape[1],
        block_gqa=16,
        block_k=32,
        limit_selected=True,
        kv_is_quantized=k_cache.dtype
        in {torch.int8, getattr(torch, "float8_e4m3fn", None)},
        num_warps=8,
        num_stages=2,
    )
    return output
