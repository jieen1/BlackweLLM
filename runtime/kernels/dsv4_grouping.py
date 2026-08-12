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
        if w >= 0 and w < BUCKET:
            for h in tl.static_range(H // 16):
                v = tl.load(src_ptr + r * H + h * 16, boundary_check=None)
                tl.store(tile_ptr + e * BUCKET * H + w * H + h * 16, v)


def device_group_counts(routes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Counts per expert and within indices.  Fixed shape, graph-safe."""
    R = routes.numel()
    E = 256
    counts = torch.zeros(E, dtype=torch.int32, device=routes.device)
    within = torch.zeros(R, dtype=torch.int32, device=routes.device)
    offsets = torch.zeros(E, dtype=torch.int32, device=routes.device)
    cursor = torch.zeros(E, dtype=torch.int32, device=routes.device)
    # zero inside graph each replay so atomics accumulate from a clean state
    counts.zero_()
    _count_kernel[(64,)](routes, counts, R)
    torch.cumsum(counts, 0, out=offsets)
    offsets.sub_(counts)
    cursor.copy_(offsets)   # cursor starts at each expert's base
    _within_kernel[(64,)](routes, offsets, cursor, within, R)
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


@triton.jit
def _fill_tiles_kernel(
    xq_ptr,        # [R, H] int8 (pre-gathered per-route activations)
    xs_ptr,        # [R, H/32] float32
    w_ptr,         # [R] float32
    routes_ptr,    # [R] int32
    within_ptr,    # [R] int32
    xq_tile_ptr,   # [E, BUCKET, H] int8
    xs_tile_ptr,   # [E, BUCKET, H/32] float32
    w_tile_ptr,    # [E, BUCKET] float32
    R,
    E,
    BUCKET: tl.constexpr,
    H: tl.constexpr,
):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    HS = H // 32
    for r in range(pid, R, step):
        e = tl.load(routes_ptr + r)
        w = tl.load(within_ptr + r)
        if w >= 0 and w < BUCKET:
            for h in tl.range(0, H):
                vq = tl.load(xq_ptr + r * H + h)
                tl.store(xq_tile_ptr + e * BUCKET * H + w * H + h, vq)
            for h in tl.range(0, HS):
                vxs = tl.load(xs_ptr + r * HS + h)
                tl.store(xs_tile_ptr + e * BUCKET * HS + w * HS + h, vxs)
            vw = tl.load(w_ptr + r)
            tl.store(w_tile_ptr + e * BUCKET + w, vw)


def device_fill_tiles(
    xq_per_route,  # [R, H] int8
    xs_per_route,  # [R, H/32] float32
    rw,            # [R] float32
    routes,        # [R] int32
    within,        # [R] int32
    xq_tile,       # [E, BUCKET, H] int8
    xs_tile,       # [E, BUCKET, H/32]
    w_tile,        # [E, BUCKET]
):
    R = routes.numel()
    E = xq_tile.shape[0]
    _fill_tiles_kernel[(64,)](xq_per_route, xs_per_route, rw, routes, within,
                              xq_tile, xs_tile, w_tile, R, E,
                              BUCKET=xq_tile.shape[1], H=xq_tile.shape[2])


def device_group_counts_deterministic(routes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic per-expert counts and within indices (argsort-based).

    Unlike the atomic-claim version, the within index of every route is a pure
    function of ``routes``, so CUDA-graph replay reproduces eager output
    exactly.  All operations are fixed-shape and graph-safe.
    """
    R = routes.numel()
    E = 256
    counts = torch.zeros(E, dtype=torch.int32, device=routes.device)
    within = torch.empty(R, dtype=torch.int32, device=routes.device)
    sorted_idx = torch.argsort(routes, stable=True)
    sr = routes[sorted_idx]
    counts.zero_()
    _count_kernel[(64,)](routes, counts, R)
    starts = torch.cumsum(counts, 0) - counts
    start_r = starts[sr]
    within_sorted = (torch.arange(R, device=routes.device) - start_r).to(torch.int32)
    within.scatter_(0, sorted_idx, within_sorted)
    return counts, within, starts
