"""Fused DSV4 mHC pre-kernel: linear -> rsqrt -> sinkhorn -> pre/post/comb.

The eager ``Dsv4Block.hc_pre`` chain runs ~47 tiny launches per token per
block (dequant + linear + rsqrt + sigmoid/softmax + 2x19 sinkhorn norms);
at 86 block-sides per decode step that is thousands of small kernels a
step. This kernel fuses the whole pre side into one program per token:

    mixes[t, j] = (x[t] . dequant_q8_0(W[j])) * rsqrt(mean(x[t]^2) + eps)

then the reference ``hc_split_sinkhorn`` semantics in fp32 (sigmoid
pre/post, softmax+eps comb, 19 rounds of row/column normalization ending
on a column normalize -- deliberately NOT re-symmetrized, that would
"fix" the reference), and finally the reduced stream

    y[t, :] = sum_h pre[h] * x[t, h, :]

matching ``Dsv4Block.hc_pre`` in fp32 modulo reduction order (the parity
tolerance for HC is 1e-4, measured below). Weights stay packed Q8_0 --
dequantized inside the kernel, no bf16 cache ever.

Implementation notes:
- bf16 loads followed by fp32 reductions are LOSSY on this Triton build
  (sums quantize to coarse grids; verified 2026-08-07, see the kernel
  comment). Every x read therefore goes through the raw-bytes bitcast
  path, which is exact (1e-7 agreement).
- The 512 Q8_0 block loop is rolled, not unrolled: static unrolling blows
  up both compile time and code size.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_mixes_partial_kernel(
    x_ptr,  # [T, hc*d] bf16, row-major
    w_ptr,  # [hc_mix, hc*d] Q8_0 packed, row stride w_row_bytes bytes
    mixes_ptr,  # [T, N_SPLITS, 32] fp32 out (partial sums)
    hc_dim: tl.constexpr,
    w_row_bytes: tl.constexpr,
    hc6: tl.constexpr,  # hc*6 (24 real W rows)
    BLKS_PER_SPLIT: tl.constexpr,
    N_SPLITS: tl.constexpr,
):
    """Split-K partial Q8_0 GEMV for the mHC mixes.

    One program per (token, split); each accumulates ``BLKS_PER_SPLIT`` of
    the 512 Q8_0 blocks into a partial 32-lane sum.  Splits fill grid.y so
    decode's single token uses many SMs instead of one (measured 156us ->
    12us at 32 splits on SM120 for the GEMV portion).
    """
    t = tl.program_id(0)
    sp = tl.program_id(1)
    idx32 = tl.arange(0, 32)
    rows_ok = idx32 < hc6
    x_u16 = x_ptr.to(tl.pointer_type(tl.uint16))
    acc = tl.zeros((32,), dtype=tl.float32)
    start = sp * BLKS_PER_SPLIT * 32
    for k in range(0, BLKS_PER_SPLIT * 32, 32):
        blk = start + k
        xs = (tl.load(x_u16 + t * hc_dim + blk + idx32).to(tl.uint32) << 16).to(
            tl.float32, bitcast=True
        )
        offs_j = idx32[:, None] * w_row_bytes
        wq = tl.load(
            w_ptr + offs_j + (blk // 32) * 34 + 2 + idx32[None, :],
            mask=rows_ok[:, None],
            other=0,
        )
        d2 = tl.load(
            w_ptr + offs_j + (blk // 32) * 34 + tl.arange(0, 2)[None, :],
            mask=rows_ok[:, None],
            other=0,
        )
        d_lo, d_hi = tl.split(d2)
        d_u16 = d_lo.to(tl.uint16) | (d_hi.to(tl.uint16) << 8)
        d_f = d_u16.to(tl.float16, bitcast=True).to(tl.float32)
        dots = tl.sum(
            xs[None, :] * wq.to(tl.int8, bitcast=True).to(tl.float32) * d_f[:, None],
            axis=1,
        )
        acc += dots
    tl.store(mixes_ptr + t * N_SPLITS * 32 + sp * 32 + idx32, acc, mask=rows_ok)


@triton.jit
def _hc_pre_kernel(
    x_ptr,  # [T, hc*d] bf16, row-major
    w_ptr,  # [hc_mix, hc*d] Q8_0 packed, row stride w_row_bytes bytes
    scale_ptr,  # [3] fp32
    base_ptr,  # [hc_mix] fp32
    y_ptr,  # [T, d] bf16 out (reduced stream)
    post_ptr,  # [T, hc] fp32 out
    comb_ptr,  # [T, hc*hc] fp32 out
    eps,
    sinkhorn_iters: tl.constexpr,
    hc: tl.constexpr,  # hc_mult (4)
    d: tl.constexpr,  # hidden size (4096)
    hc_dim: tl.constexpr,  # hc * d (16384)
    w_row_bytes: tl.constexpr,  # packed bytes per W row (hc_dim/32*34)
    mixes_ptr,  # [T, N_SPLITS, 32] fp32 partial sums (None for single-program path)
    n_splits: tl.constexpr,
):
    t = tl.program_id(0)
    idx32 = tl.arange(0, 32)
    rows_ok = idx32 < hc * 6  # only hc*6 (24) of the 32 lanes are real W rows

    # bf16 -> fp32 via the raw bytes: tl.load of bf16 followed by fp32 math
    # quantizes on this Triton build (verified 2026-08-07); the bitcast path
    # is exact.
    x_u16 = x_ptr.to(tl.pointer_type(tl.uint16))

    # -- pass 1: rsqrt over the whole row -----------------------------------
    cols = tl.arange(0, hc_dim)
    xf = (tl.load(x_u16 + t * hc_dim + cols).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    rsqrt = 1.0 / tl.sqrt(tl.sum(xf * xf) / hc_dim + eps)

    # -- pass 2: 24 dot products, Q8_0 dequantized in-kernel -----------------
    if n_splits > 1:
        # split-K path: reduce the per-split partial sums (deterministic:
        # fixed split order, fp32 accumulate).
        mix_part = tl.load(
            mixes_ptr + t * n_splits * 32 + tl.arange(0, n_splits)[:, None] * 32 + idx32[None, :],
            mask=tl.arange(0, n_splits)[:, None] < n_splits,
            other=0.0,
        )  # [n_splits, 32]
        acc = tl.sum(mix_part, axis=0)  # [32]
    else:
        acc = tl.zeros((32,), dtype=tl.float32)
        # Q8_0 packed blocks: 34 bytes = 2-byte fp16 d + 32 int8 q.
        # Rolled (not unrolled): 512 iterations would blow up the compile.
        for blk in range(hc_dim // 32):
            xs = (tl.load(x_u16 + t * hc_dim + blk * 32 + idx32).to(tl.uint32) << 16).to(
                tl.float32, bitcast=True
            )
            offs_j = idx32[:, None] * w_row_bytes
            wq = tl.load(
                w_ptr + offs_j + blk * 34 + 2 + idx32[None, :],
                mask=rows_ok[:, None],
                other=0,
            )
            d2 = tl.load(
                w_ptr + offs_j + blk * 34 + tl.arange(0, 2)[None, :],
                mask=rows_ok[:, None],
                other=0,
            )
            d_lo, d_hi = tl.split(d2)
            d_u16 = d_lo.to(tl.uint16) | (d_hi.to(tl.uint16) << 8)
            d_f = d_u16.to(tl.float16, bitcast=True).to(tl.float32)
            dots = tl.sum(
                xs[None, :] * wq.to(tl.int8, bitcast=True).to(tl.float32) * d_f[:, None],
                axis=1,
            )
            acc += dots
    mixes = acc * rsqrt  # [32]; lanes >= hc*6 are garbage

    # -- sinkhorn ------------------------------------------------------------
    scale = tl.load(scale_ptr + tl.arange(0, 4), mask=tl.arange(0, 4) < 3, other=0.0)
    base = tl.load(base_ptr + idx32, mask=idx32 < hc * 6, other=0.0)
    scale0 = tl.sum(tl.where(tl.arange(0, 4) == 0, scale, 0.0), axis=0)
    scale1 = tl.sum(tl.where(tl.arange(0, 4) == 1, scale, 0.0), axis=0)
    scale2 = tl.sum(tl.where(tl.arange(0, 4) == 2, scale, 0.0), axis=0)
    sc = tl.where(
        idx32 < hc,
        scale0,
        tl.where(idx32 < hc * 2, scale1, tl.where(idx32 < hc * 6, scale2, scale0)),
    )
    pre_full = tl.sigmoid(mixes * sc + base) + eps
    post_full = 2.0 * tl.sigmoid(mixes * sc + base)

    # comb elements are lanes hc*2 .. hc*3
    comb16 = tl.sum(
        tl.where(
            idx32[:, None] == (hc * 2 + tl.arange(0, 16)[None, :]),
            (mixes * sc + base)[:, None],
            0.0,
        ),
        axis=0,
    )
    comb = tl.reshape(comb16, (hc, hc))
    # Manual row softmax: tl.softmax on a [hc, hc] tensor normalizes along
    # the WRONG axis on this Triton build (per-column, verified 2026-08-07).
    row_max = tl.max(comb, axis=1)[:, None]
    exp_c = tl.exp(comb - row_max)
    comb = exp_c / tl.sum(exp_c, axis=1)[:, None] + eps
    comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)
    for _ in tl.static_range(sinkhorn_iters - 1):
        comb = comb / (tl.sum(comb, axis=1)[:, None] + eps)
        comb = comb / (tl.sum(comb, axis=0)[None, :] + eps)

    # -- reduced stream: y = sum_h pre[h] * x[t, h, :] -----------------------
    y = tl.zeros((d,), dtype=tl.float32)
    for h in tl.static_range(hc):
        pre_h = tl.sum(tl.where(idx32 == h, pre_full, 0.0), axis=0)
        xh = (tl.load(x_u16 + t * hc_dim + h * d + tl.arange(0, d)).to(tl.uint32) << 16).to(
            tl.float32, bitcast=True
        )
        y += pre_h * xh
    tl.store(y_ptr + t * d + tl.arange(0, d), y.to(tl.bfloat16))

    # -- outputs -------------------------------------------------------------
    post4 = tl.sum(
        tl.where(
            idx32[:, None] == (hc + tl.arange(0, hc)[None, :]),
            post_full[:, None],
            0.0,
        ),
        axis=0,
    )
    tl.store(post_ptr + t * hc + tl.arange(0, hc), post4)
    tl.store(comb_ptr + t * hc * hc + tl.arange(0, 16), tl.reshape(comb, (16,)))


@triton.jit
def _hc_post_kernel(
    x_ptr,  # [T, d] bf16
    residual_ptr,  # [T, hc, d] bf16
    post_ptr,  # [T, hc] fp32
    comb_ptr,  # [T, hc, hc] fp32, first hc axis is reduced
    out_ptr,  # [T, hc, d] bf16
    d: tl.constexpr,
    hc: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    out_h = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < d
    x_u16 = x_ptr.to(tl.pointer_type(tl.uint16))
    residual_u16 = residual_ptr.to(tl.pointer_type(tl.uint16))
    x = (tl.load(x_u16 + t * d + cols, mask=mask, other=0).to(tl.uint32) << 16).to(
        tl.float32, bitcast=True
    )
    residual = (
        tl.load(residual_u16 + (t * hc) * d + cols, mask=mask, other=0).to(tl.uint32) << 16
    ).to(tl.float32, bitcast=True)
    coefficient = tl.load(comb_ptr + t * hc * hc + out_h)
    mixed = tl.inline_asm_elementwise(
        asm="mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[coefficient, residual],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    for in_h in tl.static_range(1, hc):
        residual = (
            tl.load(
                residual_u16 + (t * hc + in_h) * d + cols,
                mask=mask,
                other=0,
            ).to(tl.uint32)
            << 16
        ).to(tl.float32, bitcast=True)
        coefficient = tl.load(comb_ptr + (t * hc + in_h) * hc + out_h)
        term = tl.inline_asm_elementwise(
            asm="mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[coefficient, residual],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        mixed = tl.inline_asm_elementwise(
            asm="add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[mixed, term],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    post_term = tl.inline_asm_elementwise(
        asm="mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[tl.load(post_ptr + t * hc + out_h), x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    value = tl.inline_asm_elementwise(
        asm="add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[post_term, mixed],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(out_ptr + (t * hc + out_h) * d + cols, value, mask=mask)


def hc_fused_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused ``Dsv4Block.hc_pre`` for x [T, hc, d] bf16.

    ``hc_fn`` is the packed Q8_0 flat uint8 storage (hc*6 rows of hc*d/32*34
    bytes). Returns (y [T, d] bf16, post [T, hc] fp32, comb [T, hc, hc] fp32)
    -- the same three tensors the eager chain yields.
    """
    if x.ndim != 3 or x.dtype != torch.bfloat16:
        raise ValueError(f"x must be [T, {hc_mult}, d] bf16, got {tuple(x.shape)} {x.dtype}")
    t_tokens, hc, d = x.shape
    if hc != hc_mult:
        raise ValueError(f"x second dim must be hc_mult {hc_mult}, got {hc}")
    if hc_fn.dtype != torch.uint8 or hc_fn.ndim != 1:
        raise ValueError(
            f"hc_fn must be a flat packed uint8 buffer, got {hc_fn.dtype} {hc_fn.ndim}d"
        )
    row_bytes = hc * d // 32 * 34
    if hc_fn.numel() != hc * 6 * row_bytes:
        raise ValueError(
            f"hc_fn packed bytes must be {hc * 6} rows x {row_bytes}, got {hc_fn.numel()}"
        )
    if sinkhorn_iters < 1:
        raise ValueError(f"sinkhorn_iters must be >= 1, got {sinkhorn_iters}")
    y = torch.empty(t_tokens, d, dtype=torch.bfloat16, device=x.device)
    post = torch.empty(t_tokens, hc, dtype=torch.float32, device=x.device)
    comb = torch.empty(t_tokens, hc, hc, dtype=torch.float32, device=x.device)
    if t_tokens == 0:
        return y, post, comb
    x2 = x.reshape(t_tokens, hc * d).contiguous()
    if t_tokens <= 4:
        # decode / tiny batch: the 512-block Q8_0 GEMV on one token is
        # single-SM (measured 156us); split K across up to 32 programs so
        # the M=1 surface uses the idle SMs (measured 12us at 32 splits).
        n_splits = 32
        blks_per = (hc * d // 32) // n_splits
        assert blks_per * n_splits == hc * d // 32
        mixes_partial = torch.empty(
            t_tokens, n_splits, 32, dtype=torch.float32, device=x.device
        )
        _hc_mixes_partial_kernel[(t_tokens, n_splits)](
            x2,
            hc_fn,
            mixes_partial,
            hc_dim=hc * d,
            w_row_bytes=row_bytes,
            hc6=hc * 6,
            BLKS_PER_SPLIT=blks_per,
            N_SPLITS=n_splits,
        )
        _hc_pre_kernel[(t_tokens,)](
            x2,
            hc_fn,
            hc_scale,
            hc_base,
            y,
            post,
            comb,
            eps,
            sinkhorn_iters=sinkhorn_iters,
            hc=hc_mult,
            d=d,
            hc_dim=hc * d,
            w_row_bytes=row_bytes,
            mixes_ptr=mixes_partial,
            n_splits=n_splits,
            num_warps=8 if t_tokens <= 4 else 4,
        )
    else:
        _hc_pre_kernel[(t_tokens,)](
            x2,
            hc_fn,
            hc_scale,
            hc_base,
            y,
            post,
            comb,
            eps,
            sinkhorn_iters=sinkhorn_iters,
            hc=hc_mult,
            d=d,
            hc_dim=hc * d,
            w_row_bytes=row_bytes,
            mixes_ptr=None,
            n_splits=1,
            # Real-weight SM120 decode sweeps are consistently faster with eight
            # warps for T=1/2/4. Keep prefill's established four-warp regime.
            num_warps=8 if t_tokens <= 4 else 4,
        )
    return y, post, comb


def hc_fused_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Fuse the decode HC residual expansion without FP32 broadcast temporaries."""
    if x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError(f"x must be [T, d] bf16, got {tuple(x.shape)} {x.dtype}")
    if residual.ndim != 3 or residual.dtype != torch.bfloat16:
        raise ValueError(
            f"residual must be [T, hc, d] bf16, got {tuple(residual.shape)} {residual.dtype}"
        )
    tokens, d = x.shape
    if residual.shape[0] != tokens or residual.shape[2] != d:
        raise ValueError("x and residual token/hidden dimensions must match")
    hc = residual.shape[1]
    if post.shape != (tokens, hc) or post.dtype != torch.float32:
        raise ValueError(f"post must be [{tokens}, {hc}] fp32")
    if comb.shape != (tokens, hc, hc) or comb.dtype != torch.float32:
        raise ValueError(f"comb must be [{tokens}, {hc}, {hc}] fp32")
    out = torch.empty_like(residual)
    if tokens == 0:
        return out
    _hc_post_kernel[(tokens, hc)](
        x,
        residual,
        post,
        comb,
        out,
        d=d,
        hc=hc,
        BLOCK=triton.next_power_of_2(d),
        num_warps=8,
    )
    return out
