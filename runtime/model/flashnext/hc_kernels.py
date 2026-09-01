"""Triton kernels for Flash-Next decode (M=1 / small M).

The eager hyper-connection path costs ~6 GEMV/elementwise launches per
block (96 blocks per token); at M=1 every launch is overhead-dominated
(profile 2026-08-27: gemvx + elementwise ≈ half the replay GPU time).
The complete projection fusions remain available for isolated microbenchmarks,
but production keeps the canonical GEMM/reduction order because their tiled
reductions are not bitwise equivalent to ``hyper_connection.GatedResidual`` on
the real checkpoint.  Production does use the smaller reduction-preserving
epilogues: ATen still computes the RMS reduction and GEMMs, while Triton
collapses only the following pointwise chains.  Those kernels retain the
reference BF16 cast boundaries and are guarded by focused bitwise tests.
"""

from __future__ import annotations

import torch

try:  # Triton is optional for the torch-free CI interpreter.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


_HC_BLOCK_HIDDEN = 256


def hc_fusion_supported(x: torch.Tensor) -> bool:
    """Return whether the Triton HC path is qualified for production use.

    The fused kernels are intentionally kept available for isolated kernel
    experiments, but they are not the runtime's numerical contract.  Their
    tiled ``tl.dot`` reductions do not reproduce the reference BF16 GEMM
    accumulation order closely enough: on the real Flash-Next shape the first
    greedy token can diverge from the canonical PyTorch path.  A capability
    probe based only on CUDA/dtype therefore makes model output depend on the
    backend rather than on the checkpoint.  Until a fused implementation is
    validated against the exact reference contract, ``GatedResidual`` must use
    its canonical path for both eager and CUDA-Graph execution.

    Keeping this as a function (rather than deleting the kernels) leaves the
    experimental entry points usable by focused microbenchmarks without an
    environment switch that can silently trade correctness for speed.
    """
    del x
    return False


def hc_norm_fusion_supported(x: torch.Tensor) -> bool:
    """Return whether the reduction-fused grouped RMSNorm experiment is enabled.

    The complete HC mix/combine fusion remains intentionally disabled because
    its tiled GEMM/reduction order is not the model's numerical contract.  The
    standalone grouped reduction is kept as an explicit A/B switch too: it is
    close, but its Triton reduction tree can move a BF16 boundary value by one
    ulp on the real checkpoint.  The default path instead uses
    :func:`hc_norm_apply_fusion_supported`, which leaves this reduction in
    ATen and fuses only the exact epilogue.
    """
    import os

    # Keep the quality-safe canonical path as the default.  The fused kernel
    # is an explicit A/B switch because its reduction tree can move an
    # isolated BF16 boundary value by one ulp; a full long-run model gate is
    # required before making that numerical change implicit.
    enabled = os.environ.get("QSR_FLASHNEXT_HC_NORM_FUSION", "0").strip().lower()
    return enabled in {"1", "true", "on"} and triton is not None and x.is_cuda and x.dtype in {
        torch.bfloat16,
        torch.float16,
    }


def hc_norm_apply_fusion_supported(x: torch.Tensor) -> bool:
    """Return whether the reduction-preserving norm epilogue is available.

    This is deliberately separate from :func:`hc_norm_fusion_supported`.
    The latter fuses the reduction itself and is only an explicit experiment
    because its Triton reduction tree can move a BF16 boundary value.  The
    apply-only variant keeps the variance and ``rsqrt`` on ATen, then fuses
    only the two pointwise multiplies that follow it.  It therefore preserves
    the reference reduction order while removing two tiny graph nodes per HC
    norm.
    """
    import os

    enabled = os.environ.get("QSR_FLASHNEXT_HC_NORM_APPLY_FUSION", "1").strip().lower()
    return enabled in {"1", "true", "on"} and triton is not None and x.is_cuda and x.dtype in {
        torch.bfloat16,
        torch.float16,
    }


def hc_pointwise_fusion_supported(x: torch.Tensor) -> bool:
    """Return whether exact BF16 HC pointwise epilogues are available.

    Flash-Next serving is BF16.  Keep FP16 on the canonical path until the
    separate output-dtype reduction gate is qualified; silently converting an
    FP16 branch through a BF16 intermediate would be a correctness regression.
    """
    import os

    enabled = os.environ.get("QSR_FLASHNEXT_HC_POINTWISE_FUSION", "1").strip().lower()
    return (
        enabled in {"1", "true", "on"}
        and triton is not None
        and x.is_cuda
        and x.dtype == torch.bfloat16
    )


if triton is not None:

    @triton.jit
    def _grid_barrier(counter_ptr, num_ctas):
        tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
        while tl.atomic_add(counter_ptr, 0, sem="acq_rel", scope="gpu") < num_ctas:
            pass

    @triton.jit
    def _hc_mix_persistent_kernel(
        x_ptr,
        w_down_ptr,
        w_up_ptr,
        t_raw_ptr,
        out_ptr,
        counters_ptr,
        k,
        lowrank,
        hs,
        num_rows,
        num_ctas,
        inv_hc,
        rows_pad: tl.constexpr,
        hc: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        block_j: tl.constexpr,
        block_r: tl.constexpr,
    ):
        """Persistent down -> SiLU -> up -> gated branch reduction."""
        pid = tl.program_id(0)
        rows = tl.arange(0, rows_pad)
        rows_mask = rows < num_rows

        k_cols = tl.arange(0, block_k)
        n_cols = tl.arange(0, block_n)
        n_blocks = tl.cdiv(lowrank, block_n)
        k_blocks = tl.cdiv(k, block_k)
        for tile in range(pid, n_blocks * k_blocks, num_ctas):
            n_block = tile % n_blocks
            k_block = tile // n_blocks
            n = n_block * block_n + n_cols
            k_offsets = k_block * block_k + k_cols
            n_mask = n < lowrank
            x = tl.load(
                x_ptr + rows[:, None] * k + k_offsets[None, :],
                mask=rows_mask[:, None],
                other=0.0,
            )
            weight = tl.load(
                w_down_ptr + n[:, None] * k + k_offsets[None, :],
                mask=n_mask[:, None],
                other=0.0,
            )
            partial = tl.dot(x, tl.trans(weight))
            # Keep every K tile separate.  The old atomic accumulation made
            # the FP32 reduction order depend on CTA scheduling, which was
            # enough to move recurrent draft logits across token boundaries
            # between otherwise identical processes.
            tl.store(
                t_raw_ptr
                + k_block * rows_pad * lowrank
                + rows[:, None] * lowrank
                + n[None, :],
                partial,
                mask=n_mask[None, :],
            )
        _grid_barrier(counters_ptr, num_ctas)

        # Reduce K tiles in a fixed ascending order.  Reuse tile zero as the
        # compact [rows, lowrank] buffer consumed by the up projection.
        for n_block in range(pid, n_blocks, num_ctas):
            n = n_block * block_n + n_cols
            n_mask = n < lowrank
            reduced = tl.zeros((rows_pad, block_n), dtype=tl.float32)
            for k_block in range(k_blocks):
                reduced += tl.load(
                    t_raw_ptr
                    + k_block * rows_pad * lowrank
                    + rows[:, None] * lowrank
                    + n[None, :],
                    mask=rows_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )
            tl.store(
                t_raw_ptr + rows[:, None] * lowrank + n[None, :],
                reduced,
                mask=rows_mask[:, None] & n_mask[None, :],
            )
        _grid_barrier(counters_ptr + 1, num_ctas)

        j_cols = tl.arange(0, block_j)
        r_cols = tl.arange(0, block_r)
        branches = tl.arange(0, hc)
        j_blocks = tl.cdiv(hs, block_j)
        for j_block in range(pid, j_blocks, num_ctas):
            j = j_block * block_j + j_cols
            j_mask = j < hs
            branch_j = branches[:, None] * hs + j[None, :]
            branch_j_flat = tl.reshape(branch_j, (hc * block_j,))
            branch_j_mask = tl.reshape(
                tl.broadcast_to(j_mask[None, :], (hc, block_j)),
                (hc * block_j,),
            )
            accumulator = tl.zeros((rows_pad, hc * block_j), dtype=tl.float32)
            for r_start in range(0, lowrank, block_r):
                r = r_start + r_cols
                r_mask = r < lowrank
                reduced = tl.load(
                    t_raw_ptr + rows[:, None] * lowrank + r[None, :],
                    mask=r_mask[None, :],
                    other=0.0,
                )
                reduced = reduced * inv_hc
                activated = (reduced * tl.sigmoid(reduced)).to(x_ptr.dtype.element_ty)
                up_weight = tl.load(
                    w_up_ptr + branch_j_flat[:, None] * lowrank + r[None, :],
                    mask=branch_j_mask[:, None] & r_mask[None, :],
                    other=0.0,
                )
                accumulator = tl.dot(activated, tl.trans(up_weight), accumulator)
            gate = tl.sigmoid(tl.reshape(accumulator, (rows_pad, hc, block_j)))
            branch_input = tl.load(
                x_ptr
                + rows[:, None, None] * (hc * hs)
                + branches[None, :, None] * hs
                + j[None, None, :],
                mask=rows_mask[:, None, None] & j_mask[None, None, :],
                other=0.0,
            ).to(tl.float32)
            output = tl.sum(gate * branch_input, axis=1) * inv_hc
            tl.store(
                out_ptr + rows[:, None] * hs + j[None, :],
                output,
                mask=rows_mask[:, None] & j_mask[None, :],
            )

        ticket = tl.atomic_add(counters_ptr + 2, 1, sem="acq_rel", scope="gpu")
        if ticket == num_ctas - 1:
            tl.store(counters_ptr, 0)
            tl.store(counters_ptr + 1, 0)
            tl.store(counters_ptr + 2, 0)

    @triton.jit
    def _grouped_gemma_rmsnorm_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        rows: tl.constexpr,
        total: tl.constexpr,
        group_size: tl.constexpr,
        groups: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        rg = tl.program_id(0)
        row = rg // groups
        group = rg - row * groups
        cols = tl.arange(0, block)
        mask = cols < group_size
        base = row * total + group * group_size
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / group_size
        scale = tl.rsqrt(variance + eps)
        weight = tl.load(
            weight_ptr + group * group_size + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        # Keep the two reference multiplications as distinct FP32 steps:
        # ATen computes ``x * rsqrt(var + eps)`` and only then applies the
        # Gemma ``(1 + weight)`` scale.  Writing one algebraic expression lets
        # the Triton optimizer reassociate the products and can move a BF16
        # boundary value by one ulp after the store.
        normalized = x * scale
        tl.store(
            out_ptr + base + cols,
            normalized * (1.0 + weight),
            mask=mask,
        )

    @triton.jit
    def _grouped_gemma_rmsnorm_apply_kernel(
        x_ptr,
        scale_ptr,
        weight_ptr,
        out_ptr,
        total: tl.constexpr,
        group_size: tl.constexpr,
        groups: tl.constexpr,
        block: tl.constexpr,
    ):
        """Apply a precomputed per-group RMS scale in one pointwise pass.

        ``scale`` is produced by ATen's reduction/rsqrt path.  Keep the two
        multiplies as distinct FP32 expressions to match the canonical
        ``x * rsqrt(...)`` followed by Gemma ``(1 + weight)`` sequence before
        the BF16 store.
        """
        rg = tl.program_id(0)
        row = rg // groups
        group = rg - row * groups
        cols = tl.arange(0, block)
        mask = cols < group_size
        base = row * total + group * group_size
        value = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + rg).to(tl.float32)
        weight = tl.load(
            weight_ptr + group * group_size + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        normalized = value * scale
        tl.store(
            out_ptr + base + cols,
            normalized * (1.0 + weight),
            mask=mask,
        )

    @triton.jit
    def _silu_scale_kernel(
        x_ptr,
        out_ptr,
        size: tl.constexpr,
        scale: tl.constexpr,
        block: tl.constexpr,
    ):
        cols = tl.program_id(0) * block + tl.arange(0, block)
        mask = cols < size
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32) * scale
        tl.store(out_ptr + cols, x * tl.sigmoid(x), mask=mask)

    @triton.jit
    def _mix_finish_kernel(
        up_ptr,
        normed_ptr,
        out_ptr,
        rows: tl.constexpr,
        hc: tl.constexpr,
        hs: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        col0 = tl.program_id(1) * block
        cols = col0 + tl.arange(0, block)
        mask = cols < hs
        acc = tl.zeros((block,), dtype=tl.float32)
        for branch in range(hc):
            offset = row * hc * hs + branch * hs + cols
            gate = tl.sigmoid(
                tl.load(up_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            )
            value = tl.load(normed_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            acc += gate * value
        tl.store(out_ptr + row * hs + cols, acc / hc, mask=mask)

    @triton.jit
    def _mix_finish_exact_kernel(
        up_ptr,
        normed_ptr,
        out_ptr,
        hc: tl.constexpr,
        hs: tl.constexpr,
        block: tl.constexpr,
    ):
        """Fuse sigmoid/cast/multiply while preserving BF16 mean semantics."""
        row = tl.program_id(0)
        col0 = tl.program_id(1) * block
        cols = col0 + tl.arange(0, block)
        mask = cols < hs
        acc = tl.zeros((block,), dtype=tl.float32)
        for branch in range(hc):
            offset = row * hc * hs + branch * hs + cols
            # The reference casts sigmoid to the input dtype before the
            # branch multiply.  Keep that boundary explicit; accumulating the
            # BF16 product in FP32 matches torch.mean's reduction contract.
            gate = tl.sigmoid(
                tl.load(up_ptr + offset, mask=mask, other=0.0).to(tl.float32)
            ).to(tl.bfloat16)
            value = tl.load(normed_ptr + offset, mask=mask, other=0.0)
            acc += (gate * value).to(tl.float32)
        tl.store(out_ptr + row * hs + cols, acc / hc, mask=mask)

    @triton.jit
    def _combine_gate_kernel(
        normed_ptr,
        weight_ptr,
        gate_ptr,
        rows: tl.constexpr,
        total: tl.constexpr,
        hc: tl.constexpr,
        block: tl.constexpr,
    ):
        rg = tl.program_id(0)
        row = rg // hc
        branch = rg - row * hc
        cols = tl.arange(0, block)
        mask = cols < total
        normed = tl.load(
            normed_ptr + row * total + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + branch * total + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(normed * weight, axis=0) / hc
        tl.store(gate_ptr + row * hc + branch, 2.0 * tl.sigmoid(dot))

    @triton.jit
    def _combine_apply_kernel(
        block_output_ptr,
        residual_ptr,
        gate_ptr,
        out_ptr,
        size: tl.constexpr,
        total: tl.constexpr,
        hc: tl.constexpr,
        hs: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < size
        row = offsets // total
        col = offsets - row * total
        branch = col // hs
        hidden_col = col - branch * hs
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        value = tl.load(
            block_output_ptr + row * hs + hidden_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(gate_ptr + row * hc + branch, mask=mask, other=0.0)
        product = (value * gate).to(tl.bfloat16)
        tl.store(out_ptr + offsets, residual + product, mask=mask)

    @triton.jit
    def _silu_matvec_kernel(
        x_ptr, w_ptr, y_ptr,
        K: tl.constexpr, N: tl.constexpr, HC: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """y[n] = silu(sum_k x[k] * w[n, k] / HC) -- one row of down-proj."""
        n = tl.program_id(0)
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            xv = tl.load(x_ptr + ks, mask=ks < K, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + n * K + ks, mask=ks < K, other=0.0).to(tl.float32)
            acc += xv * wv
        s = tl.sum(acc, axis=0) / HC
        y = s / (1.0 + tl.exp(-s))
        tl.store(y_ptr + n, y)

    @triton.jit
    def _scale_sigmoid_kernel(
        x_ptr,
        out_ptr,
        size: tl.constexpr,
        scale: tl.constexpr,
        multiplier: tl.constexpr,
        block: tl.constexpr,
    ):
        cols = tl.program_id(0) * block + tl.arange(0, block)
        mask = cols < size
        value = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        value = value * scale
        tl.store(
            out_ptr + cols,
            multiplier * tl.sigmoid(value),
            mask=mask,
        )

    @triton.jit
    def _gate_weight_mean_kernel(
        y_ptr, w_ptr, normed_ptr, out_ptr,
        N: tl.constexpr, HC: tl.constexpr, HS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """out[j] = mean_c sigmoid(up@gate)[c,j] * normed[c,j].

        grid over HS blocks; each program loads the full [HC, BLOCK_N] gate
        tile (recomputing the up-proj for its columns) and the matching
        normed values, then reduces over branches."""
        j0 = tl.program_id(0) * BLOCK_N
        js = j0 + tl.arange(0, BLOCK_N)
        acc = tl.zeros((HC, BLOCK_N), dtype=tl.float32)
        for i0 in range(0, N, 32):
            inner = i0 + tl.arange(0, 32)
            yv = tl.load(y_ptr + inner, mask=inner < N, other=0.0).to(tl.float32)
            # w is [HC*HS, N] row-major; element (c, j) at (c*HS + j)*N + i
            wv = tl.load(
                w_ptr + (tl.arange(0, HC)[:, None] * HS + js[None, :]) * N + inner[None, :],
                mask=(inner[None, :] < N),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(wv * yv[None, :], axis=1)[:, None]
        g = tl.sigmoid(acc)  # [HC, BLOCK_N]
        nv = tl.load(normed_ptr + tl.arange(0, HC)[:, None] * HS + js[None, :]).to(tl.float32)
        prod = g * nv
        tl.store(out_ptr + js, tl.sum(prod, axis=0) / HC)


def hc_mix_fused(
    normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    """Fused mix math for a small row batch.

    ``normed`` [hc*hs] bf16; ``w_down`` [lowrank, hc*hs]; ``w_up`` [hc*hs,
    lowrank]. Math identical to GatedResidual.mix's gate path."""
    rows = normed.numel() // normed.shape[-1]
    flat = normed.reshape(rows, normed.shape[-1])
    if rows <= 16 and flat.shape[-1] % 2048 == 0:
        return _hc_mix_persistent(flat, w_down, w_up, hc_count)
    down = torch.nn.functional.linear(flat, w_down)
    activated = torch.empty_like(down)
    size = down.numel()
    _silu_scale_kernel[(triton.cdiv(size, 256),)](
        down,
        activated,
        size=size,
        scale=1.0 / hc_count,
        block=256,
    )
    up = torch.nn.functional.linear(activated, w_up)
    hs = normed.shape[-1] // hc_count
    out = torch.empty((rows, hs), dtype=normed.dtype, device=normed.device)
    _mix_finish_kernel[(rows, triton.cdiv(hs, _HC_BLOCK_HIDDEN))](
        up,
        flat,
        out,
        rows=rows,
        hc=hc_count,
        hs=hs,
        block=_HC_BLOCK_HIDDEN,
    )
    return out.reshape(*normed.shape[:-1], hs)


_persistent_counters: dict[tuple[torch.device, int], torch.Tensor] = {}


def _persistent_counters_for_current_stream(device: torch.device) -> torch.Tensor:
    """Return barrier storage owned by the current CUDA stream.

    The persistent kernel uses grid-wide atomic barriers.  Sharing their
    counters between independent streams lets overlapping eager launches or
    graph executions observe each other's tickets.  Capture streams are also
    distinct here, so separately captured graphs keep separate barrier state.
    """
    stream = torch.cuda.current_stream(device)
    key = (device, int(stream.cuda_stream))
    counters = _persistent_counters.get(key)
    if counters is None:
        counters = torch.zeros(3, dtype=torch.int32, device=device)
        _persistent_counters[key] = counters
    return counters


def _hc_mix_persistent(
    normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    """One persistent kernel for the complete small-batch HC mix chain."""
    rows, width = normed.shape
    lowrank = w_down.shape[0]
    hs = width // hc_count
    device = normed.device
    counters = _persistent_counters_for_current_stream(device)
    k_blocks = triton.cdiv(width, 256)
    partial = torch.empty(
        (k_blocks, 16, lowrank),
        dtype=torch.float32,
        device=device,
    )
    out = torch.empty((rows, hs), dtype=normed.dtype, device=device)
    num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    _hc_mix_persistent_kernel[(num_ctas,)](
        normed,
        w_down,
        w_up,
        partial,
        out,
        counters,
        width,
        lowrank,
        hs,
        rows,
        num_ctas,
        1.0 / hc_count,
        rows_pad=16,
        hc=hc_count,
        block_n=32,
        block_k=256,
        block_j=32,
        block_r=64,
        num_warps=8,
    )
    return out
def grouped_gemma_rmsnorm_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float,
) -> torch.Tensor:
    """One-kernel grouped Gemma RMSNorm for CUDA BF16/FP16 rows."""
    total = x.shape[-1]
    rows = x.numel() // total
    groups = total // group_size
    out = torch.empty_like(x)
    block = triton.next_power_of_2(group_size)
    _grouped_gemma_rmsnorm_kernel[(rows * groups,)](
        x,
        weight,
        out,
        rows=rows,
        total=total,
        group_size=group_size,
        groups=groups,
        eps=eps,
        block=block,
        num_warps=8,
    )
    return out


def grouped_gemma_rmsnorm_apply_fused(
    x: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Apply precomputed grouped RMS scales without changing reduction math."""
    total = x.shape[-1]
    rows = x.numel() // total
    groups = total // group_size
    out = torch.empty_like(x)
    block = triton.next_power_of_2(group_size)
    _grouped_gemma_rmsnorm_apply_kernel[(rows * groups,)](
        x,
        scale.reshape(-1),
        weight,
        out,
        total=total,
        group_size=group_size,
        groups=groups,
        block=block,
        num_warps=8,
    )
    return out


def silu_scale_inplace_fused(
    x: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Apply ``silu(x * scale)`` in-place with the canonical BF16 contract."""
    if triton is None:
        raise RuntimeError("Triton is required for the fused HC SiLU epilogue")
    if x.ndim == 0 or not x.is_contiguous():
        raise ValueError("fused HC SiLU expects a contiguous non-scalar tensor")
    _silu_scale_kernel[(triton.cdiv(x.numel(), 256),)](
        x,
        x,
        size=x.numel(),
        scale=scale,
        block=256,
    )
    return x


def scale_sigmoid_inplace_fused(
    x: torch.Tensor,
    scale: float,
    multiplier: float,
) -> torch.Tensor:
    """Apply ``multiplier * sigmoid(x * scale)`` in-place."""
    if triton is None:
        raise RuntimeError("Triton is required for the fused HC sigmoid epilogue")
    if x.ndim == 0 or not x.is_contiguous():
        raise ValueError("fused HC sigmoid expects a contiguous non-scalar tensor")
    _scale_sigmoid_kernel[(triton.cdiv(x.numel(), 256),)](
        x,
        x,
        size=x.numel(),
        scale=scale,
        multiplier=multiplier,
        block=256,
    )
    return x


def mix_finish_exact_fused(
    up: torch.Tensor,
    normed: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    """Fuse the final HC mix epilogue with the canonical BF16 boundaries."""
    if triton is None:
        raise RuntimeError("Triton is required for the fused HC mix epilogue")
    if up.ndim != 2 or normed.ndim != 2 or up.shape != normed.shape:
        raise ValueError("fused HC mix expects matching 2-D up/normed tensors")
    if not up.is_contiguous() or not normed.is_contiguous():
        raise ValueError("fused HC mix expects contiguous tensors")
    rows, total = up.shape
    if total % hc_count:
        raise ValueError("HC width must be divisible by the branch count")
    hs = total // hc_count
    out = torch.empty((rows, hs), dtype=normed.dtype, device=normed.device)
    _mix_finish_exact_kernel[(rows, triton.cdiv(hs, _HC_BLOCK_HIDDEN))](
        up,
        normed,
        out,
        hc=hc_count,
        hs=hs,
        block=_HC_BLOCK_HIDDEN,
        num_warps=4,
    )
    return out


def hc_combine_fused(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    normed: torch.Tensor,
    inject_weight: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    """Two-kernel HC inject + residual update for a small row batch."""
    total = residual.shape[-1]
    hs = total // hc_count
    rows = residual.numel() // total
    gates = torch.empty((rows, hc_count), dtype=torch.float32, device=residual.device)
    block = triton.next_power_of_2(total)
    _combine_gate_kernel[(rows * hc_count,)](
        normed,
        inject_weight,
        gates,
        rows=rows,
        total=total,
        hc=hc_count,
        block=block,
        num_warps=8,
    )
    out = torch.empty_like(residual)
    size = residual.numel()
    _combine_apply_kernel[(triton.cdiv(size, _HC_BLOCK_HIDDEN),)](
        block_output,
        residual,
        gates,
        out,
        size=size,
        total=total,
        hc=hc_count,
        hs=hs,
        block=_HC_BLOCK_HIDDEN,
    )
    return out
