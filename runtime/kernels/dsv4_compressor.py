"""Triton post-GEMV decode kernels for narrow DSV4 compressor paths.

This is the narrow production contract only:

- decode M=1 / seqlen=1
- main compressor: ratio-4/128, head_dim=512, no rotate/quantize
- indexer compressor: ratio-4 overlap, head_dim=128, Hadamard + FP4

The eager `Dsv4Compressor.forward_graph` path remains the oracle for every
other shape/variant.  The indexer entry point is deliberately separate so it
can be parity/performance gated before the model starts calling it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def _hadamard_fp4_query_kernel(
    query_ptr,
    out_ptr,
    rows,
    row_stride,
    BLOCK_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    in_offs = tl.arange(0, 128)
    out_offs = block * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    query_u16 = query_ptr.to(tl.pointer_type(tl.uint16))
    values = (
        tl.load(
            query_u16 + row * row_stride + in_offs,
            mask=row < rows,
            other=0,
        ).to(tl.uint32)
        << 16
    ).to(tl.float32, bitcast=True)

    parity = in_offs[:, None] & out_offs[None, :]
    parity = parity ^ (parity >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    signs = 1.0 - 2.0 * (parity & 1).to(tl.float32)
    rotated = tl.sum(values[:, None] * signs, axis=0) * 0.08838834764831845
    rotated = rotated.to(tl.bfloat16).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(rotated), axis=0), 6.0 * 1.1754943508222875e-38)
    v = amax * (1.0 / 6.0)
    bits = v.to(tl.uint32, bitcast=True)
    exp_bits = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    exponent = exp_bits.to(tl.int32) - 127 + tl.where(mant != 0, 1, 0)
    scale_bits = (exponent + 127).to(tl.uint32) << 23
    scale = scale_bits.to(tl.float32, bitcast=True)
    scaled = rotated / scale
    magnitude = tl.minimum(tl.abs(scaled), 6.0)
    snapped = tl.full(magnitude.shape, 6.0, tl.float32)
    snapped = tl.where(magnitude <= 5.0, 4.0, snapped)
    snapped = tl.where(magnitude < 3.5, 3.0, snapped)
    snapped = tl.where(magnitude <= 2.5, 2.0, snapped)
    snapped = tl.where(magnitude < 1.75, 1.5, snapped)
    snapped = tl.where(magnitude <= 1.25, 1.0, snapped)
    snapped = tl.where(magnitude < 0.75, 0.5, snapped)
    snapped = tl.where(magnitude <= 0.25, 0.0, snapped)
    quantized = (snapped * tl.where(scaled < 0, -1.0, 1.0) * scale).to(tl.bfloat16)
    tl.store(
        out_ptr + row * row_stride + out_offs,
        quantized,
        mask=row < rows,
    )


def hadamard_fp4_query(query: torch.Tensor) -> torch.Tensor:
    """Fuse the indexer query H128 transform and block-32 FP4 simulation."""
    if query.dtype != torch.bfloat16 or query.shape[-1] != 128:
        raise ValueError(f"query must end in 128 bf16 values, got {query.shape} {query.dtype}")
    query = query.contiguous()
    rows = query.numel() // 128
    out = torch.empty_like(query)
    if rows == 0:
        return out
    _hadamard_fp4_query_kernel[(rows, 4)](
        query,
        out,
        rows,
        query.stride(-2),
        BLOCK_DIM=32,
        num_warps=4,
    )
    return out


def _is_contiguous_exact(tensor: torch.Tensor) -> bool:
    return tensor.is_contiguous() and tensor.storage_offset() == 0


def _require_contiguous_exact(name: str, tensor: torch.Tensor) -> None:
    if not _is_contiguous_exact(tensor):
        raise ValueError(f"{name} must be contiguous with zero storage offset")


def _require_contiguous_view(name: str, tensor: torch.Tensor) -> None:
    """Accept a contiguous slice whose data pointer already includes its offset."""
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_batch_tensor_devices(
    *,
    device: torch.device,
    tensors: dict[str, torch.Tensor],
) -> None:
    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def _check_main_batch_contract(
    *,
    kv_i: torch.Tensor,
    score_i: torch.Tensor,
    pos: torch.Tensor,
    slot_ids: torch.Tensor,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    overlap: bool,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
) -> int:
    batch_size = kv_i.shape[0]
    coeff = 2 if overlap else 1
    full_dim = coeff * head_dim
    expected_state_rows = coeff * ratio
    for name, tensor in {
        "kv_i": kv_i,
        "score_i": score_i,
        "ape": ape,
        "norm_weight": norm_weight,
        "freqs_cis": freqs_cis,
        "kv_state": kv_state,
        "score_state": score_state,
        "kv_cache": kv_cache,
        "out": out,
    }.items():
        _require_contiguous_exact(name, tensor)
    # CUDA Graph replay packs the three dynamic integer inputs into one
    # allocation and exposes row slices here.  Triton receives each view's
    # already-offset data pointer, so a non-zero storage offset is valid; only
    # the logical row itself must remain contiguous.
    _require_contiguous_view("pos", pos)
    _require_contiguous_view("slot_ids", slot_ids)
    _check_batch_tensor_devices(
        device=kv_i.device,
        tensors={
            "score_i": score_i,
            "pos": pos,
            "slot_ids": slot_ids,
            "ape": ape,
            "norm_weight": norm_weight,
            "freqs_cis": freqs_cis,
            "kv_state": kv_state,
            "score_state": score_state,
            "kv_cache": kv_cache,
            "out": out,
        },
    )
    if batch_size not in (1, 2, 4):
        raise ValueError(f"batch_size must be one of (1, 2, 4), got {batch_size}")
    if kv_i.shape != (batch_size, 1, full_dim) or score_i.shape != (batch_size, 1, full_dim):
        raise ValueError(
            f"expected kv_i/score_i shape ({batch_size}, 1, {full_dim}), got "
            f"{tuple(kv_i.shape)} / {tuple(score_i.shape)}"
        )
    if pos.shape != (batch_size,):
        raise ValueError(f"expected pos shape ({batch_size},), got {tuple(pos.shape)}")
    if slot_ids.shape != (batch_size,):
        raise ValueError(f"expected slot_ids shape ({batch_size},), got {tuple(slot_ids.shape)}")
    if ape.shape != (ratio, full_dim):
        raise ValueError(f"unexpected ape shape {tuple(ape.shape)}")
    if norm_weight.shape != (head_dim,):
        raise ValueError(f"unexpected norm_weight shape {tuple(norm_weight.shape)}")
    if kv_state.shape[1:] != (expected_state_rows, full_dim):
        raise ValueError(f"unexpected kv_state shape {tuple(kv_state.shape)}")
    if score_state.shape != kv_state.shape:
        raise ValueError(f"unexpected score_state shape {tuple(score_state.shape)}")
    if (
        kv_cache.ndim != 3
        or kv_cache.shape[0] != kv_state.shape[0]
        or kv_cache.shape[2] != head_dim
    ):
        raise ValueError(f"unexpected kv_cache shape {tuple(kv_cache.shape)}")
    if out.shape != (batch_size, 1, head_dim):
        raise ValueError(f"unexpected out shape {tuple(out.shape)}")
    if kv_i.dtype != torch.float32 or score_i.dtype != torch.float32:
        raise ValueError("kv_i and score_i must be float32")
    if pos.dtype != torch.int64 or slot_ids.dtype != torch.int64:
        raise ValueError("pos and slot_ids must be int64")
    if ape.dtype != torch.float32 or norm_weight.dtype != torch.float32:
        raise ValueError("ape and norm_weight must be float32")
    if kv_state.dtype != torch.float32 or score_state.dtype != torch.float32:
        raise ValueError("kv_state and score_state must be float32")
    if kv_cache.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError("kv_cache and out must be bfloat16")
    if not torch.is_complex(freqs_cis):
        raise ValueError("freqs_cis must be complex")
    # Do not inspect device values here.  These wrappers run inside CUDA Graph
    # capture, where unique/min/max/item/tolist would introduce a forbidden
    # device-to-host synchronization.  The backend validates distinct slot
    # ids and position bounds from its host-side request metadata before it
    # fills the persistent graph input tensors.
    return batch_size


def _check_indexer_batch_contract(
    *,
    kv_i: torch.Tensor,
    score_i: torch.Tensor,
    pos: torch.Tensor,
    slot_ids: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
) -> int:
    return _check_main_batch_contract(
        kv_i=kv_i,
        score_i=score_i,
        pos=pos,
        slot_ids=slot_ids,
        ratio=4,
        head_dim=128,
        rope_head_dim=64,
        overlap=True,
        ape=ape,
        norm_weight=norm_weight,
        freqs_cis=freqs_cis,
        kv_state=kv_state,
        score_state=score_state,
        kv_cache=kv_cache,
        out=out,
    )


@triton.jit
def _decode_postgemv_kernel(
    kv_i_ptr,
    score_i_ptr,
    pos_ptr,
    ape_ptr,
    norm_weight_ptr,
    freqs_ri_ptr,
    kv_state_ptr,
    score_state_ptr,
    kv_cache_ptr,
    out_ptr,
    ape_row_stride,
    kv_state_row_stride,
    score_state_row_stride,
    kv_cache_row_stride,
    out_row_stride,
    freqs_row_stride,
    eps,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    FULL_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
):
    pos = tl.load(pos_ptr)
    slot = pos % RATIO
    should_compress = ((pos + 1) % RATIO) == 0
    entry_slot = pos // RATIO
    offs_full = tl.arange(0, FULL_DIM)
    offs_head = tl.arange(0, HEAD_DIM)
    offs_pair = tl.arange(0, HEAD_DIM // 2)

    kv_i = tl.load(kv_i_ptr + offs_full).to(tl.float32)
    score_i = tl.load(score_i_ptr + offs_full).to(tl.float32)
    score_i += tl.load(ape_ptr + slot * ape_row_stride + offs_full).to(tl.float32)

    state_row = tl.where(OVERLAP, RATIO + slot, slot)
    tl.store(kv_state_ptr + state_row * kv_state_row_stride + offs_full, kv_i)
    tl.store(score_state_ptr + state_row * score_state_row_stride + offs_full, score_i)

    max_score = tl.full((HEAD_DIM,), float("-inf"), tl.float32)
    for r in range(RATIO):
        score_a = tl.load(score_state_ptr + r * score_state_row_stride + offs_head).to(tl.float32)
        if not OVERLAP:
            current_score_a = tl.load(score_i_ptr + offs_head).to(tl.float32)
            current_score_a += tl.load(ape_ptr + slot * ape_row_stride + offs_head).to(tl.float32)
            score_a = tl.where(slot == r, current_score_a, score_a)
        max_score = tl.maximum(max_score, score_a)
        if OVERLAP:
            score_b = tl.load(
                score_state_ptr + (RATIO + r) * score_state_row_stride + HEAD_DIM + offs_head
            ).to(tl.float32)
            current_score_b = tl.load(score_i_ptr + HEAD_DIM + offs_head).to(tl.float32)
            current_score_b += tl.load(ape_ptr + slot * ape_row_stride + HEAD_DIM + offs_head).to(
                tl.float32
            )
            score_b = tl.where(slot == r, current_score_b, score_b)
            max_score = tl.maximum(max_score, score_b)

    denom = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    pooled = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for r in range(RATIO):
        score_a = tl.load(score_state_ptr + r * score_state_row_stride + offs_head).to(tl.float32)
        value_a = tl.load(kv_state_ptr + r * kv_state_row_stride + offs_head).to(tl.float32)
        if not OVERLAP:
            current_score_a = tl.load(score_i_ptr + offs_head).to(tl.float32)
            current_score_a += tl.load(ape_ptr + slot * ape_row_stride + offs_head).to(tl.float32)
            current_value_a = tl.load(kv_i_ptr + offs_head).to(tl.float32)
            score_a = tl.where(slot == r, current_score_a, score_a)
            value_a = tl.where(slot == r, current_value_a, value_a)
        exp_a = tl.exp(score_a - max_score)
        denom += exp_a
        pooled += value_a * exp_a
        if OVERLAP:
            score_b = tl.load(
                score_state_ptr + (RATIO + r) * score_state_row_stride + HEAD_DIM + offs_head
            ).to(tl.float32)
            value_b = tl.load(
                kv_state_ptr + (RATIO + r) * kv_state_row_stride + HEAD_DIM + offs_head
            ).to(tl.float32)
            current_score_b = tl.load(score_i_ptr + HEAD_DIM + offs_head).to(tl.float32)
            current_score_b += tl.load(ape_ptr + slot * ape_row_stride + HEAD_DIM + offs_head).to(
                tl.float32
            )
            current_value_b = tl.load(kv_i_ptr + HEAD_DIM + offs_head).to(tl.float32)
            score_b = tl.where(slot == r, current_score_b, score_b)
            value_b = tl.where(slot == r, current_value_b, value_b)
            exp_b = tl.exp(score_b - max_score)
            denom += exp_b
            pooled += value_b * exp_b
    pooled = pooled / denom

    if OVERLAP:
        for r in range(RATIO):
            upper_kv = tl.load(kv_state_ptr + (RATIO + r) * kv_state_row_stride + offs_full).to(
                tl.float32
            )
            upper_score = tl.load(
                score_state_ptr + (RATIO + r) * score_state_row_stride + offs_full
            ).to(tl.float32)
            lower_kv = tl.load(kv_state_ptr + r * kv_state_row_stride + offs_full).to(tl.float32)
            lower_score = tl.load(score_state_ptr + r * score_state_row_stride + offs_full).to(
                tl.float32
            )
            tl.store(
                kv_state_ptr + r * kv_state_row_stride + offs_full,
                tl.where(should_compress, upper_kv, lower_kv),
            )
            tl.store(
                score_state_ptr + r * score_state_row_stride + offs_full,
                tl.where(should_compress, upper_score, lower_score),
            )

    # The eager contract casts the pooled latent to bf16 before RMSNorm,
    # then promotes that rounded value back to fp32 for the variance and
    # scaling math.  Preserve that numerically significant round point.
    pooled = pooled.to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / HEAD_DIM + eps)
    norm_weight = tl.load(norm_weight_ptr + offs_head).to(tl.float32)
    normed = pooled * inv_rms * norm_weight

    normed_bf16 = normed.to(tl.bfloat16)
    pair_even, pair_odd = tl.split(tl.reshape(normed_bf16, (HEAD_DIM // 2, 2)))
    rope_pair_start = (HEAD_DIM - ROPE_DIM) // 2
    rope_mask = offs_pair >= rope_pair_start
    rope_pair = offs_pair - rope_pair_start
    # Non-boundary early decode rows discard the finalized value, but the
    # fixed graph still executes this load. Clamp its inert RoPE row so prompts
    # shorter than the compression ratio never form a negative pointer.
    freq_base = tl.maximum(pos + 1 - RATIO, 0) * freqs_row_stride
    cos = tl.load(
        freqs_ri_ptr + freq_base + 2 * rope_pair,
        mask=rope_mask,
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        freqs_ri_ptr + freq_base + 2 * rope_pair + 1,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    pair_even_fp32 = pair_even.to(tl.float32)
    pair_odd_fp32 = pair_odd.to(tl.float32)
    new_even = tl.where(
        rope_mask,
        pair_even_fp32 * cos - pair_odd_fp32 * sin,
        pair_even_fp32,
    ).to(tl.bfloat16)
    new_odd = tl.where(
        rope_mask,
        pair_odd_fp32 * cos + pair_even_fp32 * sin,
        pair_odd_fp32,
    ).to(tl.bfloat16)

    cache_base = entry_slot * kv_cache_row_stride
    out_base = 0 * out_row_stride
    existing_even = tl.load(kv_cache_ptr + cache_base + 2 * offs_pair)
    existing_odd = tl.load(kv_cache_ptr + cache_base + 2 * offs_pair + 1)
    merged_even = tl.where(should_compress, new_even, existing_even)
    merged_odd = tl.where(should_compress, new_odd, existing_odd)
    tl.store(kv_cache_ptr + cache_base + 2 * offs_pair, merged_even)
    tl.store(kv_cache_ptr + cache_base + 2 * offs_pair + 1, merged_odd)
    tl.store(out_ptr + out_base + 2 * offs_pair, merged_even)
    tl.store(out_ptr + out_base + 2 * offs_pair + 1, merged_odd)


@triton.jit
def _decode_postgemv_batch_kernel(
    kv_i_ptr,
    score_i_ptr,
    pos_ptr,
    slot_ids_ptr,
    ape_ptr,
    norm_weight_ptr,
    freqs_ri_ptr,
    kv_state_ptr,
    score_state_ptr,
    kv_cache_ptr,
    out_ptr,
    kv_i_batch_stride,
    score_i_batch_stride,
    ape_row_stride,
    kv_state_slot_stride,
    kv_state_row_stride,
    score_state_slot_stride,
    score_state_row_stride,
    kv_cache_slot_stride,
    kv_cache_row_stride,
    out_batch_stride,
    out_row_stride,
    freqs_row_stride,
    eps,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    FULL_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
):
    batch = tl.program_id(0)
    pos = tl.load(pos_ptr + batch)
    arena_slot = tl.load(slot_ids_ptr + batch)
    slot = pos % RATIO
    should_compress = ((pos + 1) % RATIO) == 0
    entry_slot = pos // RATIO
    offs_full = tl.arange(0, FULL_DIM)
    offs_head = tl.arange(0, HEAD_DIM)
    offs_pair = tl.arange(0, HEAD_DIM // 2)

    kv_i_base = batch * kv_i_batch_stride
    score_i_base = batch * score_i_batch_stride
    kv_i = tl.load(kv_i_ptr + kv_i_base + offs_full).to(tl.float32)
    score_i = tl.load(score_i_ptr + score_i_base + offs_full).to(tl.float32)
    score_i += tl.load(ape_ptr + slot * ape_row_stride + offs_full).to(tl.float32)

    kv_state_base = arena_slot * kv_state_slot_stride
    score_state_base = arena_slot * score_state_slot_stride
    state_row = tl.where(OVERLAP, RATIO + slot, slot)
    tl.store(kv_state_ptr + kv_state_base + state_row * kv_state_row_stride + offs_full, kv_i)
    tl.store(
        score_state_ptr + score_state_base + state_row * score_state_row_stride + offs_full,
        score_i,
    )

    max_score = tl.full((HEAD_DIM,), float("-inf"), tl.float32)
    for r in range(RATIO):
        score_a = tl.load(
            score_state_ptr + score_state_base + r * score_state_row_stride + offs_head
        ).to(tl.float32)
        if not OVERLAP:
            current_score_a = tl.load(score_i_ptr + score_i_base + offs_head).to(tl.float32)
            current_score_a += tl.load(ape_ptr + slot * ape_row_stride + offs_head).to(tl.float32)
            score_a = tl.where(slot == r, current_score_a, score_a)
        max_score = tl.maximum(max_score, score_a)
        if OVERLAP:
            score_b = tl.load(
                score_state_ptr
                + score_state_base
                + (RATIO + r) * score_state_row_stride
                + HEAD_DIM
                + offs_head
            ).to(tl.float32)
            current_score_b = tl.load(score_i_ptr + score_i_base + HEAD_DIM + offs_head).to(
                tl.float32
            )
            current_score_b += tl.load(ape_ptr + slot * ape_row_stride + HEAD_DIM + offs_head).to(
                tl.float32
            )
            score_b = tl.where(slot == r, current_score_b, score_b)
            max_score = tl.maximum(max_score, score_b)

    denom = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    pooled = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for r in range(RATIO):
        score_a = tl.load(
            score_state_ptr + score_state_base + r * score_state_row_stride + offs_head
        ).to(tl.float32)
        value_a = tl.load(kv_state_ptr + kv_state_base + r * kv_state_row_stride + offs_head).to(
            tl.float32
        )
        if not OVERLAP:
            current_score_a = tl.load(score_i_ptr + score_i_base + offs_head).to(tl.float32)
            current_score_a += tl.load(ape_ptr + slot * ape_row_stride + offs_head).to(tl.float32)
            current_value_a = tl.load(kv_i_ptr + kv_i_base + offs_head).to(tl.float32)
            score_a = tl.where(slot == r, current_score_a, score_a)
            value_a = tl.where(slot == r, current_value_a, value_a)
        exp_a = tl.exp(score_a - max_score)
        denom += exp_a
        pooled += value_a * exp_a
        if OVERLAP:
            score_b = tl.load(
                score_state_ptr
                + score_state_base
                + (RATIO + r) * score_state_row_stride
                + HEAD_DIM
                + offs_head
            ).to(tl.float32)
            value_b = tl.load(
                kv_state_ptr
                + kv_state_base
                + (RATIO + r) * kv_state_row_stride
                + HEAD_DIM
                + offs_head
            ).to(tl.float32)
            current_score_b = tl.load(score_i_ptr + score_i_base + HEAD_DIM + offs_head).to(
                tl.float32
            )
            current_score_b += tl.load(ape_ptr + slot * ape_row_stride + HEAD_DIM + offs_head).to(
                tl.float32
            )
            current_value_b = tl.load(kv_i_ptr + kv_i_base + HEAD_DIM + offs_head).to(tl.float32)
            score_b = tl.where(slot == r, current_score_b, score_b)
            value_b = tl.where(slot == r, current_value_b, value_b)
            exp_b = tl.exp(score_b - max_score)
            denom += exp_b
            pooled += value_b * exp_b
    pooled = pooled / denom

    if OVERLAP:
        for r in range(RATIO):
            upper_kv = tl.load(
                kv_state_ptr + kv_state_base + (RATIO + r) * kv_state_row_stride + offs_full
            ).to(tl.float32)
            upper_score = tl.load(
                score_state_ptr
                + score_state_base
                + (RATIO + r) * score_state_row_stride
                + offs_full
            ).to(tl.float32)
            lower_kv = tl.load(
                kv_state_ptr + kv_state_base + r * kv_state_row_stride + offs_full
            ).to(tl.float32)
            lower_score = tl.load(
                score_state_ptr + score_state_base + r * score_state_row_stride + offs_full
            ).to(tl.float32)
            tl.store(
                kv_state_ptr + kv_state_base + r * kv_state_row_stride + offs_full,
                tl.where(should_compress, upper_kv, lower_kv),
            )
            tl.store(
                score_state_ptr + score_state_base + r * score_state_row_stride + offs_full,
                tl.where(should_compress, upper_score, lower_score),
            )

    pooled = pooled.to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / HEAD_DIM + eps)
    norm_weight = tl.load(norm_weight_ptr + offs_head).to(tl.float32)
    normed = pooled * inv_rms * norm_weight

    normed_bf16 = normed.to(tl.bfloat16)
    pair_even, pair_odd = tl.split(tl.reshape(normed_bf16, (HEAD_DIM // 2, 2)))
    rope_pair_start = (HEAD_DIM - ROPE_DIM) // 2
    rope_mask = offs_pair >= rope_pair_start
    rope_pair = offs_pair - rope_pair_start
    freq_base = tl.maximum(pos + 1 - RATIO, 0) * freqs_row_stride
    cos = tl.load(
        freqs_ri_ptr + freq_base + 2 * rope_pair,
        mask=rope_mask,
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        freqs_ri_ptr + freq_base + 2 * rope_pair + 1,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    pair_even_fp32 = pair_even.to(tl.float32)
    pair_odd_fp32 = pair_odd.to(tl.float32)
    new_even = tl.where(
        rope_mask,
        pair_even_fp32 * cos - pair_odd_fp32 * sin,
        pair_even_fp32,
    ).to(tl.bfloat16)
    new_odd = tl.where(
        rope_mask,
        pair_odd_fp32 * cos + pair_even_fp32 * sin,
        pair_odd_fp32,
    ).to(tl.bfloat16)

    cache_base = arena_slot * kv_cache_slot_stride + entry_slot * kv_cache_row_stride
    out_base = batch * out_batch_stride
    existing_even = tl.load(kv_cache_ptr + cache_base + 2 * offs_pair)
    existing_odd = tl.load(kv_cache_ptr + cache_base + 2 * offs_pair + 1)
    merged_even = tl.where(should_compress, new_even, existing_even)
    merged_odd = tl.where(should_compress, new_odd, existing_odd)
    tl.store(kv_cache_ptr + cache_base + 2 * offs_pair, merged_even)
    tl.store(kv_cache_ptr + cache_base + 2 * offs_pair + 1, merged_odd)
    tl.store(out_ptr + out_base + 2 * offs_pair, merged_even)
    tl.store(out_ptr + out_base + 2 * offs_pair + 1, merged_odd)


@triton.jit
def _indexer_decode_postgemv_kernel(
    kv_i_ptr,
    score_i_ptr,
    pos_ptr,
    ape_ptr,
    norm_weight_ptr,
    freqs_ri_ptr,
    kv_state_ptr,
    score_state_ptr,
    kv_cache_ptr,
    out_ptr,
    ape_row_stride,
    kv_state_row_stride,
    score_state_row_stride,
    kv_cache_row_stride,
    out_row_stride,
    freqs_row_stride,
    eps,
):
    """Ratio-4 indexer path, four CTAs with one FP4 block per CTA.

    Each CTA owns a disjoint 64-value current-state slice and 32-value output
    block. Pooling/norm/RoPE are intentionally duplicated to limit the
    Hadamard working set to 128x32 instead of the spill-heavy 128x128
    single-CTA formulation. State migration is a following tiny kernel: it
    must not race another CTA still reading the old lower state.
    """
    pid = tl.program_id(0)
    ratio: tl.constexpr = 4
    head_dim: tl.constexpr = 128
    rope_dim: tl.constexpr = 64

    pos = tl.load(pos_ptr)
    slot = pos % ratio
    should_compress = ((pos + 1) % ratio) == 0
    entry_slot = pos // ratio
    offs_state = pid * 64 + tl.arange(0, 64)
    offs_head = tl.arange(0, head_dim)
    offs_pair = tl.arange(0, head_dim // 2)

    # Partition the current-row state write across the four CTAs.  Pooling
    # explicitly substitutes the current upper-half row, so no CTA consumes a
    # sibling CTA's just-written values and no grid synchronization is needed.
    kv_slice = tl.load(kv_i_ptr + offs_state).to(tl.float32)
    score_slice = tl.load(score_i_ptr + offs_state).to(tl.float32)
    score_slice += tl.load(ape_ptr + slot * ape_row_stride + offs_state).to(tl.float32)
    state_row = ratio + slot
    tl.store(kv_state_ptr + state_row * kv_state_row_stride + offs_state, kv_slice)
    tl.store(score_state_ptr + state_row * score_state_row_stride + offs_state, score_slice)

    current_score_b = tl.load(score_i_ptr + head_dim + offs_head).to(tl.float32)
    current_score_b += tl.load(ape_ptr + slot * ape_row_stride + head_dim + offs_head).to(
        tl.float32
    )
    current_value_b = tl.load(kv_i_ptr + head_dim + offs_head).to(tl.float32)
    max_score = tl.full((head_dim,), float("-inf"), tl.float32)
    for r in range(ratio):
        score_a = tl.load(score_state_ptr + r * score_state_row_stride + offs_head).to(tl.float32)
        score_b = tl.load(
            score_state_ptr + (ratio + r) * score_state_row_stride + head_dim + offs_head
        ).to(tl.float32)
        score_b = tl.where(slot == r, current_score_b, score_b)
        max_score = tl.maximum(max_score, tl.maximum(score_a, score_b))

    denom = tl.zeros((head_dim,), tl.float32)
    pooled = tl.zeros((head_dim,), tl.float32)
    for r in range(ratio):
        score_a = tl.load(score_state_ptr + r * score_state_row_stride + offs_head).to(tl.float32)
        value_a = tl.load(kv_state_ptr + r * kv_state_row_stride + offs_head).to(tl.float32)
        score_b = tl.load(
            score_state_ptr + (ratio + r) * score_state_row_stride + head_dim + offs_head
        ).to(tl.float32)
        value_b = tl.load(
            kv_state_ptr + (ratio + r) * kv_state_row_stride + head_dim + offs_head
        ).to(tl.float32)
        score_b = tl.where(slot == r, current_score_b, score_b)
        value_b = tl.where(slot == r, current_value_b, value_b)
        exp_a = tl.exp(score_a - max_score)
        exp_b = tl.exp(score_b - max_score)
        denom += exp_a + exp_b
        pooled += value_a * exp_a + value_b * exp_b
    pooled = pooled / denom

    # BF16 round -> RMS -> BF16 -> RoPE, matching the eager finalize order.
    pooled = pooled.to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / head_dim + eps)
    norm_weight = tl.load(norm_weight_ptr + offs_head).to(tl.float32)
    normed = (pooled * inv_rms * norm_weight).to(tl.bfloat16)
    pair_even, pair_odd = tl.split(tl.reshape(normed, (head_dim // 2, 2)))
    rope_pair_start = (head_dim - rope_dim) // 2
    rope_mask = offs_pair >= rope_pair_start
    rope_pair = offs_pair - rope_pair_start
    freq_base = tl.maximum(pos + 1 - ratio, 0) * freqs_row_stride
    cos = tl.load(freqs_ri_ptr + freq_base + 2 * rope_pair, mask=rope_mask, other=1.0).to(
        tl.float32
    )
    sin = tl.load(freqs_ri_ptr + freq_base + 2 * rope_pair + 1, mask=rope_mask, other=0.0).to(
        tl.float32
    )
    pair_even_fp32 = pair_even.to(tl.float32)
    pair_odd_fp32 = pair_odd.to(tl.float32)
    post_rope_even = tl.where(
        rope_mask,
        pair_even_fp32 * cos - pair_odd_fp32 * sin,
        pair_even_fp32,
    ).to(tl.bfloat16)
    post_rope_odd = tl.where(
        rope_mask,
        pair_odd_fp32 * cos + pair_even_fp32 * sin,
        pair_odd_fp32,
    ).to(tl.bfloat16)
    post_rope = tl.reshape(tl.join(post_rope_even, post_rope_odd), (head_dim,)).to(tl.float32)

    # One exact Sylvester-Hadamard output/FP4 block per CTA.
    out_offs = pid * 32 + tl.arange(0, 32)
    had_in = offs_head[:, None]
    had_out = out_offs[None, :]
    parity = had_in & had_out
    parity = parity ^ (parity >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    signs = 1.0 - 2.0 * (parity & 1).to(tl.float32)
    rotated = tl.sum(post_rope[:, None] * signs, axis=0) * 0.08838834764831845
    rotated = rotated.to(tl.bfloat16).to(tl.float32)

    # Exact power-of-two ceil scale and e2m1 RTNE thresholds for block 32.
    amax = tl.maximum(tl.max(tl.abs(rotated), axis=0), 6.0 * 1.1754943508222875e-38)
    v = amax * (1.0 / 6.0)
    bits = v.to(tl.uint32, bitcast=True)
    exp_bits = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    exponent = exp_bits.to(tl.int32) - 127 + tl.where(mant != 0, 1, 0)
    scale_bits = (exponent + 127).to(tl.uint32) << 23
    scale = scale_bits.to(tl.float32, bitcast=True)
    scaled = rotated / scale
    magnitude = tl.minimum(tl.abs(scaled), 6.0)
    snapped = tl.full(magnitude.shape, 6.0, tl.float32)
    snapped = tl.where(magnitude <= 5.0, 4.0, snapped)
    snapped = tl.where(magnitude < 3.5, 3.0, snapped)
    snapped = tl.where(magnitude <= 2.5, 2.0, snapped)
    snapped = tl.where(magnitude < 1.75, 1.5, snapped)
    snapped = tl.where(magnitude <= 1.25, 1.0, snapped)
    snapped = tl.where(magnitude < 0.75, 0.5, snapped)
    snapped = tl.where(magnitude <= 0.25, 0.0, snapped)
    quantized = (snapped * tl.where(scaled < 0, -1.0, 1.0) * scale).to(tl.bfloat16)

    cache_base = entry_slot * kv_cache_row_stride
    existing = tl.load(kv_cache_ptr + cache_base + out_offs)
    merged = tl.where(should_compress, quantized, existing)
    tl.store(kv_cache_ptr + cache_base + out_offs, merged)
    tl.store(out_ptr + out_offs, merged)


@triton.jit
def _indexer_decode_postgemv_batch_kernel(
    kv_i_ptr,
    score_i_ptr,
    pos_ptr,
    slot_ids_ptr,
    ape_ptr,
    norm_weight_ptr,
    freqs_ri_ptr,
    kv_state_ptr,
    score_state_ptr,
    kv_cache_ptr,
    out_ptr,
    kv_i_batch_stride,
    score_i_batch_stride,
    ape_row_stride,
    kv_state_slot_stride,
    kv_state_row_stride,
    score_state_slot_stride,
    score_state_row_stride,
    kv_cache_slot_stride,
    kv_cache_row_stride,
    out_batch_stride,
    out_row_stride,
    freqs_row_stride,
    eps,
):
    """Ratio-4 indexer path, four CTAs per batch row."""
    pid = tl.program_id(0)
    batch = pid // 4
    block = pid % 4
    ratio: tl.constexpr = 4
    head_dim: tl.constexpr = 128
    rope_dim: tl.constexpr = 64

    pos = tl.load(pos_ptr + batch)
    arena_slot = tl.load(slot_ids_ptr + batch)
    slot = pos % ratio
    should_compress = ((pos + 1) % ratio) == 0
    entry_slot = pos // ratio
    offs_state = block * 64 + tl.arange(0, 64)
    offs_head = tl.arange(0, head_dim)
    offs_pair = tl.arange(0, head_dim // 2)
    kv_i_base = batch * kv_i_batch_stride
    score_i_base = batch * score_i_batch_stride
    kv_state_base = arena_slot * kv_state_slot_stride
    score_state_base = arena_slot * score_state_slot_stride

    kv_slice = tl.load(kv_i_ptr + kv_i_base + offs_state).to(tl.float32)
    score_slice = tl.load(score_i_ptr + score_i_base + offs_state).to(tl.float32)
    score_slice += tl.load(ape_ptr + slot * ape_row_stride + offs_state).to(tl.float32)
    state_row = ratio + slot
    tl.store(kv_state_ptr + kv_state_base + state_row * kv_state_row_stride + offs_state, kv_slice)
    tl.store(
        score_state_ptr + score_state_base + state_row * score_state_row_stride + offs_state,
        score_slice,
    )

    current_score_b = tl.load(score_i_ptr + score_i_base + head_dim + offs_head).to(tl.float32)
    current_score_b += tl.load(ape_ptr + slot * ape_row_stride + head_dim + offs_head).to(
        tl.float32
    )
    current_value_b = tl.load(kv_i_ptr + kv_i_base + head_dim + offs_head).to(tl.float32)
    max_score = tl.full((head_dim,), float("-inf"), tl.float32)
    for r in range(ratio):
        score_a = tl.load(
            score_state_ptr + score_state_base + r * score_state_row_stride + offs_head
        ).to(tl.float32)
        score_b = tl.load(
            score_state_ptr
            + score_state_base
            + (ratio + r) * score_state_row_stride
            + head_dim
            + offs_head
        ).to(tl.float32)
        score_b = tl.where(slot == r, current_score_b, score_b)
        max_score = tl.maximum(max_score, tl.maximum(score_a, score_b))

    denom = tl.zeros((head_dim,), tl.float32)
    pooled = tl.zeros((head_dim,), tl.float32)
    for r in range(ratio):
        score_a = tl.load(
            score_state_ptr + score_state_base + r * score_state_row_stride + offs_head
        ).to(tl.float32)
        value_a = tl.load(kv_state_ptr + kv_state_base + r * kv_state_row_stride + offs_head).to(
            tl.float32
        )
        score_b = tl.load(
            score_state_ptr
            + score_state_base
            + (ratio + r) * score_state_row_stride
            + head_dim
            + offs_head
        ).to(tl.float32)
        value_b = tl.load(
            kv_state_ptr + kv_state_base + (ratio + r) * kv_state_row_stride + head_dim + offs_head
        ).to(tl.float32)
        score_b = tl.where(slot == r, current_score_b, score_b)
        value_b = tl.where(slot == r, current_value_b, value_b)
        exp_a = tl.exp(score_a - max_score)
        exp_b = tl.exp(score_b - max_score)
        denom += exp_a + exp_b
        pooled += value_a * exp_a + value_b * exp_b
    pooled = pooled / denom

    pooled = pooled.to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / head_dim + eps)
    norm_weight = tl.load(norm_weight_ptr + offs_head).to(tl.float32)
    normed = (pooled * inv_rms * norm_weight).to(tl.bfloat16)
    pair_even, pair_odd = tl.split(tl.reshape(normed, (head_dim // 2, 2)))
    rope_pair_start = (head_dim - rope_dim) // 2
    rope_mask = offs_pair >= rope_pair_start
    rope_pair = offs_pair - rope_pair_start
    freq_base = tl.maximum(pos + 1 - ratio, 0) * freqs_row_stride
    cos = tl.load(freqs_ri_ptr + freq_base + 2 * rope_pair, mask=rope_mask, other=1.0).to(
        tl.float32
    )
    sin = tl.load(freqs_ri_ptr + freq_base + 2 * rope_pair + 1, mask=rope_mask, other=0.0).to(
        tl.float32
    )
    pair_even_fp32 = pair_even.to(tl.float32)
    pair_odd_fp32 = pair_odd.to(tl.float32)
    post_rope_even = tl.where(
        rope_mask,
        pair_even_fp32 * cos - pair_odd_fp32 * sin,
        pair_even_fp32,
    ).to(tl.bfloat16)
    post_rope_odd = tl.where(
        rope_mask,
        pair_odd_fp32 * cos + pair_even_fp32 * sin,
        pair_odd_fp32,
    ).to(tl.bfloat16)
    post_rope = tl.reshape(tl.join(post_rope_even, post_rope_odd), (head_dim,)).to(tl.float32)

    out_offs = block * 32 + tl.arange(0, 32)
    had_in = offs_head[:, None]
    had_out = out_offs[None, :]
    parity = had_in & had_out
    parity = parity ^ (parity >> 4)
    parity = parity ^ (parity >> 2)
    parity = parity ^ (parity >> 1)
    signs = 1.0 - 2.0 * (parity & 1).to(tl.float32)
    rotated = tl.sum(post_rope[:, None] * signs, axis=0) * 0.08838834764831845
    rotated = rotated.to(tl.bfloat16).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(rotated), axis=0), 6.0 * 1.1754943508222875e-38)
    v = amax * (1.0 / 6.0)
    bits = v.to(tl.uint32, bitcast=True)
    exp_bits = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    exponent = exp_bits.to(tl.int32) - 127 + tl.where(mant != 0, 1, 0)
    scale_bits = (exponent + 127).to(tl.uint32) << 23
    scale = scale_bits.to(tl.float32, bitcast=True)
    scaled = rotated / scale
    magnitude = tl.minimum(tl.abs(scaled), 6.0)
    snapped = tl.full(magnitude.shape, 6.0, tl.float32)
    snapped = tl.where(magnitude <= 5.0, 4.0, snapped)
    snapped = tl.where(magnitude < 3.5, 3.0, snapped)
    snapped = tl.where(magnitude <= 2.5, 2.0, snapped)
    snapped = tl.where(magnitude < 1.75, 1.5, snapped)
    snapped = tl.where(magnitude <= 1.25, 1.0, snapped)
    snapped = tl.where(magnitude < 0.75, 0.5, snapped)
    snapped = tl.where(magnitude <= 0.25, 0.0, snapped)
    quantized = (snapped * tl.where(scaled < 0, -1.0, 1.0) * scale).to(tl.bfloat16)

    cache_base = arena_slot * kv_cache_slot_stride + entry_slot * kv_cache_row_stride
    out_base = batch * out_batch_stride
    existing = tl.load(kv_cache_ptr + cache_base + out_offs)
    merged = tl.where(should_compress, quantized, existing)
    tl.store(kv_cache_ptr + cache_base + out_offs, merged)
    tl.store(out_ptr + out_base + out_offs, merged)


@triton.jit
def _indexer_migrate_state_kernel(
    pos_ptr,
    kv_state_ptr,
    score_state_ptr,
    kv_state_row_stride,
    score_state_row_stride,
):
    """Migrate ratio-4 overlap state after all pooling CTAs have completed."""
    pid = tl.program_id(0)
    ratio: tl.constexpr = 4
    offs = pid * 64 + tl.arange(0, 64)
    pos = tl.load(pos_ptr)
    should_compress = ((pos + 1) % ratio) == 0
    for r in range(ratio):
        upper_kv = tl.load(kv_state_ptr + (ratio + r) * kv_state_row_stride + offs)
        upper_score = tl.load(score_state_ptr + (ratio + r) * score_state_row_stride + offs)
        lower_kv = tl.load(kv_state_ptr + r * kv_state_row_stride + offs)
        lower_score = tl.load(score_state_ptr + r * score_state_row_stride + offs)
        tl.store(
            kv_state_ptr + r * kv_state_row_stride + offs,
            tl.where(should_compress, upper_kv, lower_kv),
        )
        tl.store(
            score_state_ptr + r * score_state_row_stride + offs,
            tl.where(should_compress, upper_score, lower_score),
        )


@triton.jit
def _indexer_migrate_state_batch_kernel(
    pos_ptr,
    slot_ids_ptr,
    kv_state_ptr,
    score_state_ptr,
    kv_state_slot_stride,
    kv_state_row_stride,
    score_state_slot_stride,
    score_state_row_stride,
):
    """Migrate ratio-4 overlap state after all pooling CTAs have completed."""
    pid = tl.program_id(0)
    batch = pid // 4
    block = pid % 4
    ratio: tl.constexpr = 4
    offs = block * 64 + tl.arange(0, 64)
    pos = tl.load(pos_ptr + batch)
    arena_slot = tl.load(slot_ids_ptr + batch)
    should_compress = ((pos + 1) % ratio) == 0
    kv_state_base = arena_slot * kv_state_slot_stride
    score_state_base = arena_slot * score_state_slot_stride
    for r in range(ratio):
        upper_kv = tl.load(kv_state_ptr + kv_state_base + (ratio + r) * kv_state_row_stride + offs)
        upper_score = tl.load(
            score_state_ptr + score_state_base + (ratio + r) * score_state_row_stride + offs
        )
        lower_kv = tl.load(kv_state_ptr + kv_state_base + r * kv_state_row_stride + offs)
        lower_score = tl.load(
            score_state_ptr + score_state_base + r * score_state_row_stride + offs
        )
        tl.store(
            kv_state_ptr + kv_state_base + r * kv_state_row_stride + offs,
            tl.where(should_compress, upper_kv, lower_kv),
        )
        tl.store(
            score_state_ptr + score_state_base + r * score_state_row_stride + offs,
            tl.where(should_compress, upper_score, lower_score),
        )


def supports_fused_decode_postgemv(
    *,
    ratio: int,
    rotate: bool,
    quantize: bool,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    head_dim: int,
    rope_head_dim: int,
) -> bool:
    """Strict gate for the exact Triton decode kernel contract."""
    return (
        device.type == "cuda"
        and batch_size == 1
        and seq_len == 1
        and ratio in (4, 128)
        and not rotate
        and not quantize
        and head_dim == 512
        and rope_head_dim == 64
    )


def supports_fused_decode_postgemv_batch(
    *,
    ratio: int,
    rotate: bool,
    quantize: bool,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    head_dim: int,
    rope_head_dim: int,
) -> bool:
    """Strict gate for the exact batched Triton decode kernel contract."""
    return (
        device.type == "cuda"
        and batch_size in (1, 2, 4)
        and seq_len == 1
        and ratio in (4, 128)
        and not rotate
        and not quantize
        and head_dim == 512
        and rope_head_dim == 64
    )


def supports_fused_indexer_decode_postgemv(
    *,
    ratio: int,
    rotate: bool,
    quantize: bool,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    head_dim: int,
    rope_head_dim: int,
) -> bool:
    """Strict gate for the ratio-4 indexer compressor experiment."""
    return (
        device.type == "cuda"
        and batch_size == 1
        and seq_len == 1
        and ratio == 4
        and rotate
        and quantize
        and head_dim == 128
        and rope_head_dim == 64
    )


def supports_fused_indexer_decode_postgemv_batch(
    *,
    ratio: int,
    rotate: bool,
    quantize: bool,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    head_dim: int,
    rope_head_dim: int,
) -> bool:
    """Strict gate for the batched ratio-4 indexer compressor contract."""
    return (
        device.type == "cuda"
        and batch_size in (1, 2, 4)
        and seq_len == 1
        and ratio == 4
        and rotate
        and quantize
        and head_dim == 128
        and rope_head_dim == 64
    )


def fused_decode_postgemv_batch(
    *,
    kv_i: torch.Tensor,
    score_i: torch.Tensor,
    pos: torch.Tensor,
    slot_ids: torch.Tensor,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    overlap: bool,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the batched Triton decode kernel into ``out`` and return it.

    ``slot_ids`` must be distinct and all device values must be in bounds;
    the host-side backend validates those capture-time preconditions.
    """
    batch_size = _check_main_batch_contract(
        kv_i=kv_i,
        score_i=score_i,
        pos=pos,
        slot_ids=slot_ids,
        ratio=ratio,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        overlap=overlap,
        ape=ape,
        norm_weight=norm_weight,
        freqs_cis=freqs_cis,
        kv_state=kv_state,
        score_state=score_state,
        kv_cache=kv_cache,
        out=out,
    )
    if not supports_fused_decode_postgemv_batch(
        ratio=ratio,
        rotate=False,
        quantize=False,
        device=kv_i.device,
        batch_size=batch_size,
        seq_len=kv_i.shape[1],
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
    ):
        raise ValueError(
            "fused_decode_postgemv_batch requires the exact CUDA main-compressor contract"
        )
    coeff = 2 if overlap else 1
    full_dim = coeff * head_dim
    freqs_ri = torch.view_as_real(freqs_cis).reshape(freqs_cis.shape[0], rope_head_dim)
    num_warps = 16 if ratio == 128 else 8
    _decode_postgemv_batch_kernel[(batch_size,)](
        kv_i,
        score_i,
        pos,
        slot_ids,
        ape,
        norm_weight,
        freqs_ri,
        kv_state,
        score_state,
        kv_cache,
        out,
        kv_i.stride(0),
        score_i.stride(0),
        ape.stride(0),
        kv_state.stride(0),
        kv_state.stride(1),
        score_state.stride(0),
        score_state.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        out.stride(0),
        out.stride(1),
        freqs_ri.stride(0),
        eps,
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_head_dim,
        FULL_DIM=full_dim,
        RATIO=ratio,
        OVERLAP=overlap,
        num_warps=num_warps,
    )
    return out


def fused_decode_postgemv(
    *,
    kv_i: torch.Tensor,
    score_i: torch.Tensor,
    pos: torch.Tensor,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    overlap: bool,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the single-launch Triton decode kernel into ``out`` and return it."""
    if not supports_fused_decode_postgemv(
        ratio=ratio,
        rotate=False,
        quantize=False,
        device=kv_i.device,
        batch_size=kv_i.shape[0],
        seq_len=kv_i.shape[1],
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
    ):
        raise ValueError("fused_decode_postgemv requires the exact CUDA main-compressor contract")
    coeff = 2 if overlap else 1
    full_dim = coeff * head_dim
    if kv_i.shape != (1, 1, full_dim) or score_i.shape != (1, 1, full_dim):
        raise ValueError(
            f"expected kv_i/score_i shape (1, 1, {full_dim}), got "
            f"{tuple(kv_i.shape)} / {tuple(score_i.shape)}"
        )
    if kv_state.shape != (1, coeff * ratio, full_dim):
        raise ValueError(f"unexpected kv_state shape {tuple(kv_state.shape)}")
    if score_state.shape != (1, coeff * ratio, full_dim):
        raise ValueError(f"unexpected score_state shape {tuple(score_state.shape)}")
    if kv_cache.shape[0] != 1 or kv_cache.shape[2] != head_dim:
        raise ValueError(f"unexpected kv_cache shape {tuple(kv_cache.shape)}")
    if out.shape != (1, 1, head_dim):
        raise ValueError(f"unexpected out shape {tuple(out.shape)}")

    freqs_ri = torch.view_as_real(freqs_cis).reshape(freqs_cis.shape[0], rope_head_dim)
    num_warps = 16 if ratio == 128 else 8
    _decode_postgemv_kernel[(1,)](
        kv_i,
        score_i,
        pos,
        ape,
        norm_weight,
        freqs_ri,
        kv_state,
        score_state,
        kv_cache,
        out,
        ape.stride(0),
        kv_state.stride(1),
        score_state.stride(1),
        kv_cache.stride(1),
        out.stride(1),
        freqs_ri.stride(0),
        eps,
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_head_dim,
        FULL_DIM=full_dim,
        RATIO=ratio,
        OVERLAP=overlap,
        num_warps=num_warps,
    )
    return out


def fused_indexer_decode_postgemv_batch(
    *,
    kv_i: torch.Tensor,
    score_i: torch.Tensor,
    pos: torch.Tensor,
    slot_ids: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the batched ratio-4 indexer compressor into ``out``.

    ``slot_ids`` must be distinct and all device values must be in bounds;
    the host-side backend validates those capture-time preconditions.
    """
    ratio = 4
    head_dim = 128
    rope_head_dim = 64
    batch_size = _check_indexer_batch_contract(
        kv_i=kv_i,
        score_i=score_i,
        pos=pos,
        slot_ids=slot_ids,
        ape=ape,
        norm_weight=norm_weight,
        freqs_cis=freqs_cis,
        kv_state=kv_state,
        score_state=score_state,
        kv_cache=kv_cache,
        out=out,
    )
    if not supports_fused_indexer_decode_postgemv_batch(
        ratio=ratio,
        rotate=True,
        quantize=True,
        device=kv_i.device,
        batch_size=batch_size,
        seq_len=kv_i.shape[1],
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
    ):
        raise ValueError("fused_indexer_decode_postgemv_batch requires the exact CUDA contract")
    freqs_ri = torch.view_as_real(freqs_cis).reshape(freqs_cis.shape[0], rope_head_dim)
    _indexer_decode_postgemv_batch_kernel[(batch_size * 4,)](
        kv_i,
        score_i,
        pos,
        slot_ids,
        ape,
        norm_weight,
        freqs_ri,
        kv_state,
        score_state,
        kv_cache,
        out,
        kv_i.stride(0),
        score_i.stride(0),
        ape.stride(0),
        kv_state.stride(0),
        kv_state.stride(1),
        score_state.stride(0),
        score_state.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        out.stride(0),
        out.stride(1),
        freqs_ri.stride(0),
        eps,
        num_warps=8,
    )
    _indexer_migrate_state_batch_kernel[(batch_size * 4,)](
        pos,
        slot_ids,
        kv_state,
        score_state,
        kv_state.stride(0),
        kv_state.stride(1),
        score_state.stride(0),
        score_state.stride(1),
        num_warps=4,
    )
    return out


def fused_indexer_decode_postgemv(
    *,
    kv_i: torch.Tensor,
    score_i: torch.Tensor,
    pos: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the experimental ratio-4 indexer compressor into ``out``.

    This API is intentionally not wired into ``Dsv4Compressor``.  Its exact
    full-state parity and end-to-end quality must be established on the target
    SM120 GPU before production routing is enabled.
    """
    ratio = 4
    head_dim = 128
    rope_head_dim = 64
    full_dim = 256
    if not supports_fused_indexer_decode_postgemv(
        ratio=ratio,
        rotate=True,
        quantize=True,
        device=kv_i.device,
        batch_size=kv_i.shape[0],
        seq_len=kv_i.shape[1],
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
    ):
        raise ValueError("fused_indexer_decode_postgemv requires the exact CUDA contract")
    if kv_i.shape != (1, 1, full_dim) or score_i.shape != (1, 1, full_dim):
        raise ValueError(
            f"expected kv_i/score_i shape (1, 1, {full_dim}), got "
            f"{tuple(kv_i.shape)} / {tuple(score_i.shape)}"
        )
    if ape.shape != (ratio, full_dim):
        raise ValueError(f"unexpected ape shape {tuple(ape.shape)}")
    if norm_weight.shape != (head_dim,):
        raise ValueError(f"unexpected norm_weight shape {tuple(norm_weight.shape)}")
    if kv_state.shape != (1, 2 * ratio, full_dim):
        raise ValueError(f"unexpected kv_state shape {tuple(kv_state.shape)}")
    if score_state.shape != (1, 2 * ratio, full_dim):
        raise ValueError(f"unexpected score_state shape {tuple(score_state.shape)}")
    if kv_cache.ndim != 3 or kv_cache.shape[0] != 1 or kv_cache.shape[2] != head_dim:
        raise ValueError(f"unexpected kv_cache shape {tuple(kv_cache.shape)}")
    if out.shape != (1, 1, head_dim):
        raise ValueError(f"unexpected out shape {tuple(out.shape)}")

    freqs_ri = torch.view_as_real(freqs_cis).reshape(freqs_cis.shape[0], rope_head_dim)
    _indexer_decode_postgemv_kernel[(4,)](
        kv_i,
        score_i,
        pos,
        ape,
        norm_weight,
        freqs_ri,
        kv_state,
        score_state,
        kv_cache,
        out,
        ape.stride(0),
        kv_state.stride(1),
        score_state.stride(1),
        kv_cache.stride(1),
        out.stride(1),
        freqs_ri.stride(0),
        eps,
        num_warps=8,
    )
    _indexer_migrate_state_kernel[(4,)](
        pos,
        kv_state,
        score_state,
        kv_state.stride(1),
        score_state.stride(1),
        num_warps=4,
    )
    return out


def compile_fused_decode_postgemv_batch_sm120(
    *,
    ratio: int,
    overlap: bool,
    num_warps: int | None = None,
    num_stages: int = 3,
):
    """Offline-compile the batched main-compressor decode kernel for SM120."""
    if ratio not in (4, 128):
        raise ValueError(f"ratio must be 4 or 128, got {ratio}")
    head_dim = 512
    rope_head_dim = 64
    full_dim = head_dim * (2 if overlap else 1)
    compile_num_warps = 16 if ratio == 128 else 8
    if num_warps is not None:
        compile_num_warps = num_warps
    src = ASTSource(
        fn=_decode_postgemv_batch_kernel,
        signature={
            "kv_i_ptr": "*fp32",
            "score_i_ptr": "*fp32",
            "pos_ptr": "*i64",
            "slot_ids_ptr": "*i64",
            "ape_ptr": "*fp32",
            "norm_weight_ptr": "*fp32",
            "freqs_ri_ptr": "*fp32",
            "kv_state_ptr": "*fp32",
            "score_state_ptr": "*fp32",
            "kv_cache_ptr": "*bf16",
            "out_ptr": "*bf16",
            "kv_i_batch_stride": "i64",
            "score_i_batch_stride": "i64",
            "ape_row_stride": "i64",
            "kv_state_slot_stride": "i64",
            "kv_state_row_stride": "i64",
            "score_state_slot_stride": "i64",
            "score_state_row_stride": "i64",
            "kv_cache_slot_stride": "i64",
            "kv_cache_row_stride": "i64",
            "out_batch_stride": "i64",
            "out_row_stride": "i64",
            "freqs_row_stride": "i64",
            "eps": "fp32",
            "HEAD_DIM": "constexpr",
            "ROPE_DIM": "constexpr",
            "FULL_DIM": "constexpr",
            "RATIO": "constexpr",
            "OVERLAP": "constexpr",
        },
        constexprs={
            "HEAD_DIM": head_dim,
            "ROPE_DIM": rope_head_dim,
            "FULL_DIM": full_dim,
            "RATIO": ratio,
            "OVERLAP": overlap,
        },
    )
    return triton.compile(
        src,
        target=GPUTarget("cuda", 120, 32),
        options={"num_warps": compile_num_warps, "num_stages": num_stages},
    )


def compile_fused_indexer_decode_postgemv_batch_sm120(
    *,
    num_warps: int = 8,
    num_stages: int = 3,
    migrate_num_warps: int = 4,
    migrate_num_stages: int = 2,
):
    """Offline-compile the batched ratio-4 indexer decode and migrate kernels."""
    decode_src = ASTSource(
        fn=_indexer_decode_postgemv_batch_kernel,
        signature={
            "kv_i_ptr": "*fp32",
            "score_i_ptr": "*fp32",
            "pos_ptr": "*i64",
            "slot_ids_ptr": "*i64",
            "ape_ptr": "*fp32",
            "norm_weight_ptr": "*fp32",
            "freqs_ri_ptr": "*fp32",
            "kv_state_ptr": "*fp32",
            "score_state_ptr": "*fp32",
            "kv_cache_ptr": "*bf16",
            "out_ptr": "*bf16",
            "kv_i_batch_stride": "i64",
            "score_i_batch_stride": "i64",
            "ape_row_stride": "i64",
            "kv_state_slot_stride": "i64",
            "kv_state_row_stride": "i64",
            "score_state_slot_stride": "i64",
            "score_state_row_stride": "i64",
            "kv_cache_slot_stride": "i64",
            "kv_cache_row_stride": "i64",
            "out_batch_stride": "i64",
            "out_row_stride": "i64",
            "freqs_row_stride": "i64",
            "eps": "fp32",
        },
        constexprs={},
    )
    migrate_src = ASTSource(
        fn=_indexer_migrate_state_batch_kernel,
        signature={
            "pos_ptr": "*i64",
            "slot_ids_ptr": "*i64",
            "kv_state_ptr": "*fp32",
            "score_state_ptr": "*fp32",
            "kv_state_slot_stride": "i64",
            "kv_state_row_stride": "i64",
            "score_state_slot_stride": "i64",
            "score_state_row_stride": "i64",
        },
        constexprs={},
    )
    target = GPUTarget("cuda", 120, 32)
    return (
        triton.compile(
            decode_src,
            target=target,
            options={"num_warps": num_warps, "num_stages": num_stages},
        ),
        triton.compile(
            migrate_src,
            target=target,
            options={"num_warps": migrate_num_warps, "num_stages": migrate_num_stages},
        ),
    )
