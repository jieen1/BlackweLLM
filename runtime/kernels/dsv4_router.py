"""Triton router kernel for DSV4 scored (non-hash) MoE layers.

Fuses the post-GEMM half of the gate (plan Phase 3 §4: the gate GEMM
itself stays in the model graph): softplus -> sqrt -> bias shift ->
top-6 selection -> gather original scores -> renormalize -> x1.5.

Semantics mirror the reference Gate exactly (parity-proven in
tests/test_dsv4_reference_parts.py): the bias moves the SELECTION only;
the routing weights come from the unbiased sqrtsoftplus scores,
renormalized over the chosen six and scaled by route_scale (1.5).

One program per token row; E=256 fits one block, so top-k is a static
six-pass argmax-with-mask loop -- no workspace, fixed shape, CUDA Graph
safe. Tie-breaking is first-occurrence (argmax); torch.topk agrees on
real (tie-free) score distributions.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _dsv4_router_kernel(
    logits_ptr,  # [T, E] fp32 gate logits (post-GEMM)
    bias_ptr,  # [E] fp32 selection bias (exp_probs_b)
    weights_ptr,  # [T, K] fp32 out: renormalized routing weights
    indices_ptr,  # [T, K] int64 out: expert ids, descending score order
    stride_row,
    K: tl.constexpr,
    ROUTE_SCALE: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < E
    logits = tl.load(logits_ptr + row * stride_row + cols, mask=mask, other=float("-inf"))
    # torch softplus: identity above the threshold (20), log1p(exp(x)) below
    sp = tl.where(logits > 20.0, logits, tl.log(1.0 + tl.exp(logits)))
    scores = tl.sqrt(sp)
    bias = tl.load(bias_ptr + cols, mask=mask, other=float("-inf"))
    selection = scores + bias

    w_sum = 0.0
    for k in tl.static_range(K):
        idx = tl.argmax(selection, axis=0)
        gathered = tl.sum(tl.where(cols == idx, scores, 0.0), axis=0)
        tl.store(indices_ptr + row * K + k, idx.to(tl.int64))
        tl.store(weights_ptr + row * K + k, gathered)
        w_sum += gathered
        selection = tl.where(cols == idx, float("-inf"), selection)

    for k in tl.static_range(K):
        raw = tl.load(weights_ptr + row * K + k)
        tl.store(weights_ptr + row * K + k, raw / w_sum * ROUTE_SCALE)


def dsv4_route_scores(
    logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """[T, E] fp32 logits + [E] bias -> ([T, K] weights, [T, K] int64 ids)."""
    if logits.dtype != torch.float32:
        raise ValueError(f"router logits must be fp32, got {logits.dtype}")
    tokens, experts = logits.shape
    weights = torch.empty(tokens, top_k, dtype=torch.float32, device=logits.device)
    indices = torch.empty(tokens, top_k, dtype=torch.int64, device=logits.device)
    if tokens == 0:
        return weights, indices
    block = triton.next_power_of_2(experts)
    _dsv4_router_kernel[(tokens,)](
        logits,
        bias,
        weights,
        indices,
        logits.stride(0),
        K=top_k,
        ROUTE_SCALE=route_scale,
        E=experts,
        BLOCK=block,
    )
    return weights, indices
