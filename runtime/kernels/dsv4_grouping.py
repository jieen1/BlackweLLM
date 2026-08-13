"""Triton device-side MoE grouping: counts, within, and activation tile fill.

Replaces the Python argsort/nonzero glue (dynamic shapes, not CUDA-graph
safe) with fixed-shape kernels.  routes [R] int32 -> counts[256] (per-expert
route count), within[R] (route index within its expert, 0..count-1), and the
activation tile [E, BUCKET, H] filled from flat quantized activations.

The within order is arbitrary (atomic claim), which is fine: gate/up/down are
per-expert, and combine writes back by route index, so tile order does not
affect the result.  Requires a second pass to claim slots after counts are
known (two kernel launches, both fixed-shape).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _count_kernel(
    routes_ptr,   # [R] int32
    counts_ptr,   # [E] int32 out (zeroed by caller)
    R,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    for r in range(pid, R, step):
        e = tl.load(routes_ptr + r)
        tl.atomic_add(counts_ptr + e, 1)


@triton.jit
def _within_kernel(
    routes_ptr,      # [R] int32
    base_ptr,        # [E] int32 (exclusive offsets, fixed)
    cursor_ptr,      # [E] int32 (running cursor, zeroed by caller)
    within_ptr,      # [R] int32 out
    R,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    for r in range(pid, R, step):
        e = tl.load(routes_ptr + r)
        base = tl.load(base_ptr + e)
        slot = tl.atomic_add(cursor_ptr + e, 1)
        tl.store(within_ptr + r, slot - base)


@triton.jit
def _fill_kernel(
    src_ptr,       # [R, H] int8 (quantized flat activations)
    routes_ptr,    # [R] int32
    within_ptr,    # [R] int32
    tile_ptr,      # [E, BUCKET, H] int8 out (zeroed by caller)
    R,
    H: tl.constexpr,
    BUCKET: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    for r in range(pid, R, step):
        e = tl.load(routes_ptr + r)
        w = tl.load(within_ptr + r)
        if w < BUCKET:
            offs = tl.arange(0, 32)
            base = e * BUCKET * H + w * H
            src = r * H
            for h in tl.static_range(H // 32):
                v = tl.load(src_ptr + src + h * 32 + offs, boundary_check=None)
                tl.store(tile_ptr + base + h * 32 + offs, v)


def device_group_counts(routes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Counts per expert and within indices.  Fixed shape, graph-safe.

    Allocates its scratch buffers per call -- safe for eager use, but NOT
    for inside a CUDA graph capture body.  Use
    :func:`device_group_counts_into` with caller-owned buffers there.
    """
    R = routes.numel()
    E = 256
    counts = torch.zeros(E, dtype=torch.int32, device=routes.device)
    within = torch.zeros(R, dtype=torch.int32, device=routes.device)
    offsets = torch.zeros(E, dtype=torch.int32, device=routes.device)
    # arange carrier must be >= R (see device_group_counts_into's contract)
    cursor = torch.zeros(max(E, R), dtype=torch.int32, device=routes.device)
    return device_group_counts_into(routes, counts, within, offsets, cursor)


def device_group_counts_into(
    routes: torch.Tensor,
    counts: torch.Tensor,
    within: torch.Tensor,
    offsets: torch.Tensor,
    cursor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Caller-owned variant: writes into preallocated buffers (graph-safe).

    Deterministic (unlike the atomic-cursor variant): routes are sorted
    stably, so ``within[r]`` is a pure function of the route order.  This is
    what makes the result reproducible across CUDA-graph replays -- atomics
    give a different slot order every replay, which corrupts the combine.

    ``cursor`` must be at least ``R = routes.numel()`` elements long: it is
    used as the ``torch.arange(R)`` output carrier.  It is NOT a per-expert
    buffer here (that role died with the atomic variant); sizing it to
    ``n_experts`` resizes it on the first call, which breaks CUDA-graph
    capture's fixed-shape contract.
    """
    R = routes.numel()
    if cursor.numel() < R:
        raise ValueError(
            f"cursor must hold R={R} arange elements, got {cursor.numel()}"
        )
    # stable sort routes -> sorted_eids, order
    sorted_eids, order = torch.sort(routes, stable=True)
    # per-expert counts via scatter-add of a ones vector into counts
    counts.zero_()
    counts.scatter_add_(0, sorted_eids, torch.ones_like(sorted_eids))
    # exclusive per-expert offsets: cumsum - counts
    torch.cumsum(counts, 0, out=offsets)
    offsets.sub_(counts)
    # within[r] = position among same-expert routes = arange(R) - base[expert]
    base = offsets[sorted_eids]  # exclusive start of this route's expert
    torch.arange(R, device=routes.device, out=cursor)
    within.copy_(cursor - base)  # in sorted order, then unsort by `order`
    # unsort within back to original route order
    inv = torch.empty_like(order)
    inv[order] = torch.arange(R, device=routes.device)
    within.copy_(within[inv])
    return counts, within, offsets


def device_group_fill(
    src: torch.Tensor,   # [R, H] int8 (or [R] -> scale)
    routes: torch.Tensor,
    within: torch.Tensor,
    tile: torch.Tensor,  # [E, BUCKET, H] int8
) -> None:
    R = routes.numel()
    H = tile.shape[-1]
    _fill_kernel[(64,)](src, routes, within, tile, R, H=H, BUCKET=tile.shape[1])
