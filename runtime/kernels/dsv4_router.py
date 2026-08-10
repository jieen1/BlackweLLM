"""Triton router kernels for DSV4 scored and hashed MoE layers.

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

The first three hashed layers receive their six expert ids from the checkpoint's
``tid2eid`` table.  Their native candidate skips the unnecessary 256-wide score
materialization: it evaluates sqrt-softplus for those six ids directly, then
normalizes and scales them in the same program.  The supplied id order (and any
duplicate ids) is preserved exactly.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

_HASHED_EXPERTS = 256
_HASHED_TOP_K = 6
_HASHED_NUM_WARPS = 1
_HASHED_NUM_STAGES = 1


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


@triton.jit
def _dsv4_hashed_router_kernel(
    logits_ptr,  # [T, 256] fp32 gate logits
    supplied_ids_ptr,  # [T, 6] int32 checkpoint tid2eid rows
    weights_ptr,  # [T, 6] fp32 out: normalized routing weights
    indices_ptr,  # [T, 6] int64 out: supplied expert ids
    logits_stride_row,
    logits_stride_col,
    ids_stride_row,
    ids_stride_col,
    K: tl.constexpr,
    ROUTE_SCALE: tl.constexpr,
):
    """Route one token from checkpoint-supplied ids without a score workspace."""
    row = tl.program_id(0)
    weight_row = weights_ptr + row * K
    index_row = indices_ptr + row * K
    w_sum = 0.0

    # Keep this scalar static loop in the same order as the accepted non-hash
    # router.  K=6 is compile-time fixed, so there is no device/host sync and no
    # dynamic loop in CUDA Graph replay.
    for k in tl.static_range(K):
        expert_id = tl.load(supplied_ids_ptr + row * ids_stride_row + k * ids_stride_col)
        logit = tl.load(
            logits_ptr + row * logits_stride_row + expert_id.to(tl.int64) * logits_stride_col
        )
        # Match torch.nn.functional.softplus(beta=1, threshold=20), as in the
        # non-hash route kernel above.
        softplus = tl.where(logit > 20.0, logit, tl.log(1.0 + tl.exp(logit)))
        score = tl.sqrt(softplus)
        tl.store(index_row + k, expert_id.to(tl.int64))
        tl.store(weight_row + k, score)
        w_sum += score

    for k in tl.static_range(K):
        score = tl.load(weight_row + k)
        tl.store(weight_row + k, score / w_sum * ROUTE_SCALE)


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


def dsv4_route_hashed_scores(
    logits: torch.Tensor,
    supplied_ids: torch.Tensor,
    route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route fixed DSV4 hash layers from device-side ``tid2eid`` ids.

    This deliberately narrow candidate accepts the production decode/prefill
    contract: contiguous fp32 ``logits[T, 256]`` and contiguous int32
    ``supplied_ids[T, 6]`` on one CUDA device.  It performs no value-range
    validation because reading ids back to the host would break CUDA Graph
    safety; checkpoint loading owns the invariant ``0 <= id < 256``.
    """
    if logits.device.type != "cuda" or supplied_ids.device.type != "cuda":
        raise ValueError("hashed router requires CUDA logits and supplied_ids")
    if logits.device != supplied_ids.device:
        raise ValueError("hashed router requires logits and supplied_ids on one device")
    if logits.dtype != torch.float32:
        raise ValueError(f"hashed router logits must be fp32, got {logits.dtype}")
    if supplied_ids.dtype != torch.int32:
        raise ValueError(f"hashed router supplied_ids must be int32, got {supplied_ids.dtype}")
    if logits.ndim != 2 or logits.shape[1] != _HASHED_EXPERTS:
        raise ValueError(
            f"hashed router logits must have shape [T, {_HASHED_EXPERTS}], "
            f"got {tuple(logits.shape)}"
        )
    if supplied_ids.ndim != 2 or supplied_ids.shape != (
        logits.shape[0],
        _HASHED_TOP_K,
    ):
        raise ValueError(
            f"hashed router supplied_ids must have shape [T, {_HASHED_TOP_K}], "
            f"got {tuple(supplied_ids.shape)}"
        )
    if not logits.is_contiguous() or not supplied_ids.is_contiguous():
        raise ValueError("hashed router requires contiguous logits and supplied_ids")

    tokens = logits.shape[0]
    weights = torch.empty((tokens, _HASHED_TOP_K), dtype=torch.float32, device=logits.device)
    indices = torch.empty((tokens, _HASHED_TOP_K), dtype=torch.int64, device=logits.device)
    if tokens == 0:
        return weights, indices
    _dsv4_hashed_router_kernel[(tokens,)](
        logits,
        supplied_ids,
        weights,
        indices,
        logits.stride(0),
        logits.stride(1),
        supplied_ids.stride(0),
        supplied_ids.stride(1),
        K=_HASHED_TOP_K,
        ROUTE_SCALE=route_scale,
        num_warps=_HASHED_NUM_WARPS,
        num_stages=_HASHED_NUM_STAGES,
    )
    return weights, indices


def compile_dsv4_hashed_router_sm120(
    *,
    route_scale: float = 1.5,
    num_warps: int = _HASHED_NUM_WARPS,
    num_stages: int = _HASHED_NUM_STAGES,
):
    """Offline-compile the fixed hashed-router candidate for SM120."""
    src = ASTSource(
        fn=_dsv4_hashed_router_kernel,
        signature={
            "logits_ptr": "*fp32",
            "supplied_ids_ptr": "*i32",
            "weights_ptr": "*fp32",
            "indices_ptr": "*i64",
            "logits_stride_row": "i64",
            "logits_stride_col": "i64",
            "ids_stride_row": "i64",
            "ids_stride_col": "i64",
            "K": "constexpr",
            "ROUTE_SCALE": "constexpr",
        },
        constexprs={
            "K": _HASHED_TOP_K,
            "ROUTE_SCALE": float(route_scale),
        },
    )
    return triton.compile(
        src,
        target=GPUTarget("cuda", 120, 32),
        options={
            "num_warps": num_warps,
            "num_stages": num_stages,
        },
    )
