"""Graph-safe routed BF16 expert kernels for Flash-Next MTP decode.

MTP decode has one row and ten selected experts.  Materialising
``weight[expert_ids]`` copies roughly 100 MiB per layer invocation, while
``torch.grouped_mm`` cannot currently be captured by a CUDA Graph because its
group metadata is staged through the host.  These kernels read the resident
expert tensors directly using the device expert ids.
"""

from __future__ import annotations

import torch

try:  # Triton is optional for the torch-free CI interpreter.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


def mtp_expert_matvec_supported(x: torch.Tensor) -> bool:
    return triton is not None and x.is_cuda and x.dtype == torch.bfloat16


def mtp_weighted_route_reduce_supported(
    sorted_out: torch.Tensor,
    order: torch.Tensor,
    routing_weights: torch.Tensor,
) -> bool:
    """Return whether the large-M sorted-route reducer can run on Triton."""
    return (
        triton is not None
        and sorted_out.is_cuda
        and sorted_out.dtype == torch.bfloat16
        and sorted_out.is_contiguous()
        and order.is_cuda
        and order.dtype == torch.int64
        and order.is_contiguous()
        and routing_weights.is_cuda
        and routing_weights.dtype == torch.float32
        and routing_weights.is_contiguous()
    )


if triton is not None:

    @triton.jit
    def _mtp_gate_up_matvec_kernel(
        x_ptr,
        ids_ptr,
        gate_up_ptr,
        gate_up_scale_ptr,
        out_ptr,
        rows: tl.constexpr,
        hidden: tl.constexpr,
        gate_up_width: tl.constexpr,
        top_k: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        quantized: tl.constexpr,
    ):
        route_idx = tl.program_id(0)
        row = route_idx // top_k
        cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
        col_mask = cols < gate_up_width
        expert = tl.load(ids_ptr + route_idx)
        weight_base = expert * gate_up_width * hidden
        acc = tl.zeros((block_n,), dtype=tl.float32)
        for start in range(0, hidden, block_k):
            inner = start + tl.arange(0, block_k)
            inner_mask = inner < hidden
            values = tl.load(
                x_ptr + row * hidden + inner,
                mask=inner_mask,
                other=0.0,
            )
            weights = tl.load(
                gate_up_ptr + weight_base + cols[:, None] * hidden + inner[None, :],
                mask=col_mask[:, None] & inner_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            if quantized:
                weights *= tl.load(
                    gate_up_scale_ptr + expert * gate_up_width + cols,
                    mask=col_mask,
                    other=1.0,
                )[:, None]
            acc += tl.sum(weights * values.to(tl.float32), axis=1)
        tl.store(out_ptr + route_idx * gate_up_width + cols, acc, mask=col_mask)

    @triton.jit
    def _mtp_down_weighted_matvec_kernel(
        gate_up_out_ptr,
        ids_ptr,
        routing_weights_ptr,
        down_ptr,
        down_scale_ptr,
        out_ptr,
        rows: tl.constexpr,
        hidden: tl.constexpr,
        inter: tl.constexpr,
        top_k: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        quantized: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.program_id(1) * block_n + tl.arange(0, block_n)
        col_mask = cols < hidden
        total = tl.zeros((block_n,), dtype=tl.float32)
        for route in range(top_k):
            route_idx = row * top_k + route
            expert = tl.load(ids_ptr + route_idx)
            route_weight = tl.load(routing_weights_ptr + route_idx).to(tl.float32)
            expert_acc = tl.zeros((block_n,), dtype=tl.float32)
            for start in range(0, inter, block_k):
                inner = start + tl.arange(0, block_k)
                inner_mask = inner < inter
                gate = tl.load(
                    gate_up_out_ptr + route_idx * (2 * inter) + inner,
                    mask=inner_mask,
                    other=0.0,
                ).to(tl.float32)
                up = tl.load(
                    gate_up_out_ptr + route_idx * (2 * inter) + inter + inner,
                    mask=inner_mask,
                    other=0.0,
                ).to(tl.float32)
                # Match the BF16 activation tensor consumed by grouped_mm.
                activated = ((gate * tl.sigmoid(gate)).to(tl.bfloat16) * up).to(
                    tl.bfloat16
                )
                weights = tl.load(
                    down_ptr
                    + expert * hidden * inter
                    + cols[:, None] * inter
                    + inner[None, :],
                    mask=col_mask[:, None] & inner_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                if quantized:
                    weights *= tl.load(
                        down_scale_ptr + expert * hidden + cols,
                        mask=col_mask,
                        other=1.0,
                    )[:, None]
                expert_acc += tl.sum(
                    weights * activated[None, :].to(tl.float32),
                    axis=1,
                )
            # grouped_mm writes BF16 before the FP32 routing reduction.
            total += expert_acc.to(tl.bfloat16).to(tl.float32) * route_weight
        tl.store(out_ptr + row * hidden + cols, total, mask=col_mask)

    @triton.jit
    def _mtp_weighted_route_reduce_kernel(
        sorted_out_ptr,
        inverse_order_ptr,
        routing_weights_ptr,
        out_ptr,
        hidden: tl.constexpr,
        top_k: tl.constexpr,
        block_hidden: tl.constexpr,
    ):
        """Restore routes and reduce them without materialising [T,K,H]."""
        row = tl.program_id(0)
        cols = tl.program_id(1) * block_hidden + tl.arange(0, block_hidden)
        mask = cols < hidden
        acc = tl.zeros((block_hidden,), dtype=tl.float32)
        for route in range(top_k):
            original_row = row * top_k + route
            sorted_row = tl.load(inverse_order_ptr + original_row)
            values = tl.load(
                sorted_out_ptr + sorted_row * hidden + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(routing_weights_ptr + original_row).to(tl.float32)
            acc += values * weight
        # The reference path casts the FP32 routing sum back to BF16.
        tl.store(out_ptr + row * hidden + cols, acc, mask=mask)


def mtp_expert_matvec(
    x: torch.Tensor,
    ids: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    *,
    gate_up_scales: torch.Tensor | None = None,
    down_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate and route up to a few MTP rows without copying selected weights."""
    if not mtp_expert_matvec_supported(x):
        raise RuntimeError("MTP expert matvec requires CUDA BF16 with Triton")
    if x.ndim != 2 or ids.ndim != 2 or routing_weights.ndim != 2:
        raise ValueError("mtp_expert_matvec expects x [T,H], ids [T,K], weights [T,K]")
    if x.shape[0] != ids.shape[0] or ids.shape != routing_weights.shape:
        raise ValueError("mtp_expert_matvec inputs must agree on row and top-k shape")
    top_k = ids.shape[1]
    rows = x.shape[0]
    hidden = x.shape[1]
    inter = down.shape[2]
    gate_up_width = gate_up.shape[1]
    if gate_up_width != 2 * inter or down.shape[1] != hidden:
        raise ValueError("incompatible MTP expert tensor shapes")
    quantized = gate_up.dtype == getattr(torch, "float8_e4m3fn", None)
    if quantized:
        if down.dtype != gate_up.dtype:
            raise ValueError("quantized MTP expert tensors must use one dtype")
        if gate_up_scales is None or down_scales is None:
            raise ValueError("quantized MTP expert matvec requires row scales")
        if tuple(gate_up_scales.shape) != (gate_up.shape[0], gate_up.shape[1]):
            raise ValueError("invalid gate_up expert row-scale shape")
        if tuple(down_scales.shape) != (down.shape[0], down.shape[1]):
            raise ValueError("invalid down expert row-scale shape")
    elif gate_up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16:
        raise ValueError("MTP expert matvec supports BF16 or FP8 E4M3 weights")

    # The pointers are unused in the BF16 specialization.  Reusing the weight
    # tensors avoids an allocation inside CUDA Graph capture; ``quantized`` is
    # a constexpr so Triton removes the branch and pointer loads entirely.
    gate_up_scale_ptr = gate_up_scales if gate_up_scales is not None else gate_up
    down_scale_ptr = down_scales if down_scales is not None else down

    projected = torch.empty(
        rows * top_k,
        gate_up_width,
        dtype=x.dtype,
        device=x.device,
    )
    _mtp_gate_up_matvec_kernel[(rows * top_k, triton.cdiv(gate_up_width, 16))](
        x,
        ids,
        gate_up,
        gate_up_scale_ptr,
        projected,
        rows=rows,
        hidden=hidden,
        gate_up_width=gate_up_width,
        top_k=top_k,
        block_n=16,
        block_k=64,
        quantized=quantized,
        num_warps=4,
    )
    output = torch.empty(rows, hidden, dtype=x.dtype, device=x.device)
    _mtp_down_weighted_matvec_kernel[(rows, triton.cdiv(hidden, 16))](
        projected,
        ids,
        routing_weights,
        down,
        down_scale_ptr,
        output,
        rows=rows,
        hidden=hidden,
        inter=inter,
        top_k=top_k,
        block_n=16,
        block_k=64,
        quantized=quantized,
        num_warps=4,
    )
    return output


def mtp_weighted_route_reduce(
    sorted_out: torch.Tensor,
    order: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """Reduce grouped expert outputs directly into token-major BF16 rows.

    ``order`` is the stable ``argsort`` permutation used to produce
    ``sorted_out``: sorted row ``s`` belongs to original route ``order[s]``.
    The old implementation first copied those rows into ``[T,K,H]`` and then
    expanded the entire tensor to FP32 for routing.  For long prompt sync that
    transient is over a GiB, so the CUDA path reconstructs each route in a
    Triton program and accumulates the weighted result directly into ``[T,H]``.

    The fallback intentionally retains the reference operation order for CPU,
    unsupported dtypes, and environments without Triton.
    """
    if sorted_out.ndim != 2 or order.ndim != 1 or routing_weights.ndim != 2:
        raise ValueError("expected sorted_out [routes,H], order [routes], weights [T,K]")
    routes, hidden = sorted_out.shape
    rows, top_k = routing_weights.shape
    if routes != rows * top_k or order.numel() != routes:
        raise ValueError("incompatible sorted routes, permutation, and routing weights")
    if order.device != sorted_out.device or routing_weights.device != sorted_out.device:
        raise ValueError("sorted routes, permutation, and routing weights must share a device")

    if not mtp_weighted_route_reduce_supported(sorted_out, order, routing_weights):
        restored = torch.empty_like(sorted_out)
        restored.index_copy_(0, order, sorted_out)
        return (
            restored.view(rows, top_k, hidden).float()
            * routing_weights.unsqueeze(-1)
        ).sum(dim=1).to(sorted_out.dtype)

    # ``order`` maps sorted rows to original route rows.  Invert it once so
    # each output program can walk its K routes without a global search.
    inverse_order = torch.empty_like(order)
    inverse_order.scatter_(
        0,
        order,
        torch.arange(routes, device=order.device, dtype=order.dtype),
    )
    output = torch.empty(rows, hidden, dtype=sorted_out.dtype, device=sorted_out.device)
    block_hidden = 256 if hidden >= 256 else 128
    _mtp_weighted_route_reduce_kernel[(rows, triton.cdiv(hidden, block_hidden))](
        sorted_out,
        inverse_order,
        routing_weights,
        output,
        hidden=hidden,
        top_k=top_k,
        block_hidden=block_hidden,
        num_warps=8 if block_hidden == 256 else 4,
    )
    return output
