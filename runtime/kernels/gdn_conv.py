"""Fused causal depthwise conv1d + SiLU for the Qwen3.8 GDN target path.

Replaces ``F.conv1d(depthwise) + F.silu`` in
``Qwen36GatedDeltaNet._conv1d_window`` with one launch and one read/write of
the activation instead of a generic depthwise conv plus a separate SiLU pass.

Two kernels share one wrapper, selected by output length:

* ``_causal_conv_silu_kernel`` — length-tiled, for prefill windows
  (``out_len >= 64``). The tile's inner axis is channels (stride-1 on the
  production ``transpose(1, 2)`` view), so every access coalesces.
* ``_conv_step_kernel`` — channel-parallel, for tiny decode windows
  (single-token update: ``padding == 0``, ``out_len == 1``). The length tile
  variant would launch one mostly-masked program per slot here; this one
  fills the machine over channels instead.

Numerical contract, verified bit-for-bit against the eager pair by
``tests/test_qwen36_gdn_conv_fusion.py``:

* BF16 checkpoint path (safetensors): all K taps accumulate in FP32 in tap
  order, the sum rounds to BF16 exactly where ``F.conv1d``'s output tensor
  sits, and SiLU runs on that rounded value in FP32 using ATen's exact form
  (``x / (1 + exp(-x))`` — not ``x * sigmoid(x)``, which differs in ULP).
* FP32 checkpoint path (GGUF scalars): deliberately NOT fused — the FP32
  eager conv contracts FMAs differently and fusing changed results by 1 ULP;
  the caller keeps the eager reference there.
"""

from __future__ import annotations

import torch

try:  # Triton is optional for the torch-free CI interpreter.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by the torch-free gate
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _causal_conv_silu_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        L,
        out_len,
        stride_x_b,
        stride_x_c,
        stride_x_l,
        stride_w_c,
        stride_w_k,
        stride_o_b,
        stride_o_c,
        stride_o_l,
        pad,
        num_c_blocks: tl.constexpr,
        K: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        pid_b = tl.program_id(1)
        l_block = tl.program_id(0)

        ls = l_block * BLOCK_L + tl.arange(0, BLOCK_L)
        l_mask = ls < out_len

        # A runtime loop keeps the generated code small; each tile's inner
        # dimension is channels (stride-1), so every access coalesces.
        for c_block in range(num_c_blocks):
            cs = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
            acc = tl.zeros((BLOCK_L, BLOCK_C), dtype=tl.float32)
            for k in tl.static_range(K):
                src = ls + k - pad
                xv = tl.load(
                    x_ptr
                    + pid_b * stride_x_b
                    + cs[None, :] * stride_x_c
                    + src[:, None] * stride_x_l,
                    mask=l_mask[:, None]
                    & (src[:, None] >= 0)
                    & (src[:, None] < L),
                    other=0.0,
                )
                wv = tl.load(w_ptr + cs * stride_w_c + k * stride_w_k)
                acc += wv[None, :].to(tl.float32) * xv.to(tl.float32)

            r = tl.cast(acc, tl.bfloat16).to(tl.float32)
            y = r / (1.0 + tl.exp(-r))
            tl.store(
                out_ptr
                + pid_b * stride_o_b
                + cs[None, :] * stride_o_c
                + ls[:, None] * stride_o_l,
                tl.cast(y, tl.bfloat16),
                mask=l_mask[:, None],
            )

    @triton.jit
    def _conv_step_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        L,
        out_len,
        stride_x_b,
        stride_x_c,
        stride_x_l,
        stride_w_c,
        stride_w_k,
        stride_o_b,
        stride_o_c,
        stride_o_l,
        pad,
        C,
        K: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_bt = tl.program_id(1)
        b = pid_bt // out_len
        t = pid_bt % out_len

        cs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = cs < C
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for k in tl.static_range(K):
            src = t + k - pad
            xv = tl.load(
                x_ptr + b * stride_x_b + cs * stride_x_c + src * stride_x_l,
                mask=c_mask & (src >= 0) & (src < L),
                other=0.0,
            )
            wv = tl.load(w_ptr + cs * stride_w_c + k * stride_w_k, mask=c_mask, other=0.0)
            acc += wv.to(tl.float32) * xv.to(tl.float32)

        r = tl.cast(acc, tl.bfloat16).to(tl.float32)
        y = r / (1.0 + tl.exp(-r))
        tl.store(
            out_ptr + b * stride_o_b + cs * stride_o_c + t * stride_o_l,
            tl.cast(y, tl.bfloat16),
            mask=c_mask,
        )


def fused_causal_conv_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    padding: int,
    out_len: int | None = None,
) -> torch.Tensor | None:
    """Run the fused causal depthwise conv+SiLU, or return ``None``.

    Mirrors ``F.silu(F.conv1d(x, weight, padding=padding, groups=C))`` on the
    first ``out_len`` time steps (default: exactly the columns the caller
    keeps). Returns ``None`` when the geometry or environment is unsupported
    so the caller can take the eager reference; returning a different dtype
    or silently contiguifying inputs would break the byte-exactness contract
    this helper exists to protect.
    """

    if triton is None or not x.is_cuda:
        return None
    if x.ndim != 3 or weight.ndim != 3:
        return None
    batch, channels, seq_len = x.shape
    if weight.shape[0] != channels or weight.shape[1] != 1:
        return None
    k = weight.shape[2]
    if k < 1 or k > 8:
        return None
    if padding != k - 1 and padding != 0:
        return None
    full_len = seq_len + 2 * padding - k + 1
    if out_len is None:
        if padding == k - 1:
            # The caller truncates the right-padded tail away
            # (``[:, :, :input_len]``), so those columns are never consumed;
            # computing them would only add work and read past the input.
            out_len = min(seq_len, full_len)
        else:
            # Decode window branch: ``[:, :, -seq_len:]`` keeps the tail.
            out_len = full_len
    if out_len <= 0 or out_len > full_len:
        return None
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        # The FP32 (GGUF) eager conv contracts FMAs differently; fusing it
        # changed results by 1 ULP, so it stays on the eager path.
        return None
    if padding == k - 1 and channels % 128 != 0:
        return None

    out = torch.empty((batch, channels, out_len), dtype=x.dtype, device=x.device)
    if out_len < 64:
        block_c = 1024
        grid = (triton.cdiv(channels, block_c), batch * out_len)
        _conv_step_kernel[grid](
            x,
            weight,
            out,
            seq_len,
            out_len,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            weight.stride(0),
            weight.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            padding,
            channels,
            K=k,
            BLOCK_C=block_c,
            num_warps=4,
        )
        return out

    block_l = 64
    grid = (triton.cdiv(out_len, block_l), batch)
    _causal_conv_silu_kernel[grid](
        x,
        weight,
        out,
        seq_len,
        out_len,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        weight.stride(0),
        weight.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        padding,
        num_c_blocks=channels // 128,
        K=k,
        BLOCK_C=128,
        BLOCK_L=block_l,
        num_warps=4,
    )
    return out
