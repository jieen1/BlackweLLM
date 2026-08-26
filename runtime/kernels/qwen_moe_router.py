"""Fused softmax + top-k router for the Qwen MoE family on SM120.

Day-0 prep for Qwen3.8-Flash-Next (see
notes/2026-08-26-qwen38-flash-next-day0-survey.md). The Qwen3-Next /
Qwen3.5-MoE / Flash-Next family routing contract is the reference
``Qwen3NextTopKRouter`` (transformers ``modeling_qwen3_next.py``, pinned
2026-08-26 from the installed transformers 5.x):

    probs = softmax(logits.float(), dim=-1)
    values, indices = topk(probs, top_k)
    if norm_topk_prob: values /= values.sum(dim=-1, keepdim=True)

This is NOT Laguna's sigmoid + correction-bias contract, so
``kernels/laguna_router_sm120.cu`` deliberately does not serve this path.

Numerical contract, verified against the eager torch reference by
``tests/test_qwen_moe_router.py``:

* logits upcast to FP32 for the softmax (the reference computes softmax in
  FP32 regardless of the logits dtype);
* top-k selection breaks exact ties toward the LOWER expert index, matching
  ``torch.topk``;
* renormalization divides by the exact selected sum (denominator floored at
  1.0 only for the all-zero row, mirroring the reference's plain division
  behavior on normal rows);
* weights are cast back to the logits dtype, ids are int32.
"""

from __future__ import annotations

import torch

try:  # Triton is optional for the torch-free CI interpreter.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by the torch-free gate
    triton = None
    tl = None


def qwen_moe_router_reference(
    logits: torch.Tensor,
    top_k: int,
    *,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager torch implementation of the Qwen MoE family router contract."""
    probs = torch.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(probs, top_k, dim=-1)
    if renormalize:
        values = values / values.sum(dim=-1, keepdim=True)
    return values.to(logits.dtype), indices.to(torch.int32)


if triton is not None:

    @triton.jit
    def _softmax_topk_kernel(
        logits_ptr,
        weights_ptr,
        ids_ptr,
        num_experts,
        stride_logits_row,
        stride_out_row,
        BLOCK_E: tl.constexpr,
        TOP_K: tl.constexpr,
        RENORMALIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        experts = tl.arange(0, BLOCK_E)
        mask = experts < num_experts

        x = tl.load(
            logits_ptr + row * stride_logits_row + experts,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        x_max = tl.max(x, axis=0)
        exp_x = tl.exp(x - x_max)
        probs = exp_x / tl.sum(exp_x, axis=0)

        selected_sum = 0.0
        for k in tl.static_range(TOP_K):
            best = tl.max(probs, axis=0)
            is_best = probs == best
            # Lowest expert index wins exact ties, matching torch.topk.
            candidate = tl.where(is_best, experts, num_experts)
            idx = tl.min(candidate, axis=0)
            tl.store(weights_ptr + row * stride_out_row + k, best)
            tl.store(ids_ptr + row * stride_out_row + k, idx)
            selected_sum += best
            probs = tl.where(experts == idx, 0.0, probs)

        if RENORMALIZE:
            denom = tl.where(selected_sum > 0.0, selected_sum, 1.0)
            for k in tl.static_range(TOP_K):
                slot = weights_ptr + row * stride_out_row + k
                tl.store(slot, tl.load(slot) / denom)


def qwen_moe_softmax_topk(
    logits: torch.Tensor,
    top_k: int,
    *,
    renormalize: bool = True,
    weights_out: torch.Tensor | None = None,
    ids_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused softmax + top-k routing for one router call.

    ``logits`` is ``[rows, num_experts]`` BF16 or FP32, CUDA, contiguous.
    Returns ``(weights, ids)`` shaped ``[rows, top_k]`` in the logits dtype
    and int32. The optional output tensors let a caller keep CUDA-Graph-safe
    address-stable arenas (they must already be the right shape/dtype/device).
    """
    if logits.ndim != 2:
        raise ValueError(f"router logits must be rank-2, got {tuple(logits.shape)}")
    if not logits.is_cuda or not logits.is_contiguous():
        raise ValueError("router logits must be contiguous CUDA tensors")
    if logits.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"router logits must be BF16 or FP32, got {logits.dtype}")
    rows, num_experts = logits.shape
    if top_k <= 0 or top_k > num_experts:
        raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
    block_e = triton.next_power_of_2(num_experts)

    if weights_out is None:
        weights_out = torch.empty((rows, top_k), dtype=logits.dtype, device=logits.device)
    if ids_out is None:
        ids_out = torch.empty((rows, top_k), dtype=torch.int32, device=logits.device)
    if weights_out.shape != (rows, top_k) or weights_out.dtype != logits.dtype:
        raise ValueError("weights_out must match [rows, top_k] and the logits dtype")
    if ids_out.shape != (rows, top_k) or ids_out.dtype != torch.int32:
        raise ValueError("ids_out must match [rows, top_k] and be int32")
    if rows == 0:
        return weights_out, ids_out

    _softmax_topk_kernel[(rows,)](
        logits,
        weights_out,
        ids_out,
        num_experts,
        logits.stride(0),
        weights_out.stride(0),
        BLOCK_E=block_e,
        TOP_K=top_k,
        RENORMALIZE=renormalize,
        num_warps=4 if block_e <= 512 else 8,
    )
    return weights_out, ids_out
