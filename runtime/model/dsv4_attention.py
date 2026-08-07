"""Functional attention parts for the DSV4 model graph (eager semantics).

Transcriptions of the reference math (notes/dsv4flash-ref/inference/):
YaRN RoPE, the QAT simulation quantizers (fp8 block-64 on the nope part of
the latent KV, fp4 block-32 in the indexer path), the sparse gather
attention with learned attn_sink, and the Hadamard rotation. Parity against
the reference tilelang kernels is tested in tests/test_dsv4_attention_parts.py.

Phase 3 replaces the hot paths with kernels; the semantics defined here are
the contract those kernels must keep.
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# RoPE (YaRN) -- transcription of reference precompute_freqs_cis /
# apply_rotary_emb (model.py), fp32 with complex64 phasors.
# ---------------------------------------------------------------------------


def _find_correction_dim(num_rotations: float, dim: int, base: float, max_seq_len: int) -> float:
    return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    *,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Complex phasors [seqlen, dim//2]; YaRN interpolation when original_seq_len > 0."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    if original_seq_len > 0:
        low = math.floor(_find_correction_dim(beta_fast, dim, base, original_seq_len))
        high = math.ceil(_find_correction_dim(beta_slow, dim, base, original_seq_len))
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp(
            (torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / (high - low), 0, 1
        )
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen, dtype=torch.float32, device=device)
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, *, inverse: bool = False
) -> torch.Tensor:
    """Rotate the last dim pair-wise in place (reference semantics: y.copy_)."""
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    phasor = freqs_cis.conj() if inverse else freqs_cis
    if xc.ndim == 3:
        phasor = phasor.view(1, xc.size(1), xc.size(-1))
    else:
        phasor = phasor.view(1, xc.size(1), 1, xc.size(-1))
    xc = torch.view_as_real(xc * phasor).flatten(-2)
    y.copy_(xc)
    return y


# ---------------------------------------------------------------------------
# QAT simulation quantizers (reference kernel.py act_quant / fp4_act_quant,
# inplace=True mode: quantize then dequantize back to the input dtype).
# ---------------------------------------------------------------------------

_FP8_MAX = 448.0
_FP4_MAX = 6.0


def _pow2_scale(amax: torch.Tensor, qmax_inv: float) -> torch.Tensor:
    """2^ceil(log2(amax * qmax_inv)), exact via frexp (matches the reference
    bit-trick fast_round_scale)."""
    v = amax * qmax_inv
    mant, exp = torch.frexp(v)  # v = mant * 2^exp, mant in [0.5, 1)
    return torch.ldexp(torch.ones_like(v), torch.where(mant == 0.5, exp - 1, exp).to(torch.int32))


def act_quant_simulate(x: torch.Tensor, block_size: int, *, ue8m0: bool) -> torch.Tensor:
    """Block-wise fp8-e4m3 quant-dequant simulation along the last dim.

    Reference semantics: amax floored at 1e-4; scale = amax/448 (or the
    power-of-two ceiling under ue8m0); clamp to +-448; cast e4m3; multiply
    the scale back; return in the input dtype.
    """
    if x.shape[-1] % block_size:
        raise ValueError(f"last dim {x.shape[-1]} not a multiple of block {block_size}")
    dtype = x.dtype
    shape = x.shape
    xf = x.float().reshape(-1, shape[-1] // block_size, block_size)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = _pow2_scale(amax, 1 / _FP8_MAX) if ue8m0 else amax / _FP8_MAX
    q = (xf / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return (q.to(torch.float32) * scale).reshape(shape).to(dtype)


_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even onto the e2m1 magnitude set {0,.5,1,1.5,2,3,4,6}.

    Tie resolution follows IEEE RTNE with the e2m1 encoding: 0.25->0,
    0.75->1, 1.25->1, 1.75->2, 2.5->2, 3.5->4, 5->4.
    """
    a = x.abs()
    out = torch.full_like(a, 6.0)
    out = torch.where(a <= 5.0, torch.full_like(a, 4.0), out)
    out = torch.where(a < 3.5, torch.full_like(a, 3.0), out)
    out = torch.where(a <= 2.5, torch.full_like(a, 2.0), out)
    out = torch.where(a < 1.75, torch.full_like(a, 1.5), out)
    out = torch.where(a <= 1.25, torch.full_like(a, 1.0), out)
    out = torch.where(a < 0.75, torch.full_like(a, 0.5), out)
    out = torch.where(a <= 0.25, torch.zeros_like(a), out)
    return out * x.sign()


def fp4_act_quant_simulate(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Block-wise e2m1 quant-dequant simulation (indexer path).

    Reference semantics: amax floored at 6*2^-126; scale is always the
    power-of-two ceiling of amax/6; clamp to +-6; cast e2m1; scale back.
    """
    if x.shape[-1] % block_size:
        raise ValueError(f"last dim {x.shape[-1]} not a multiple of block {block_size}")
    dtype = x.dtype
    shape = x.shape
    xf = x.float().reshape(-1, shape[-1] // block_size, block_size)
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(_FP4_MAX * 2.0**-126)
    scale = _pow2_scale(amax, 1 / _FP4_MAX)
    q = _round_e2m1((xf / scale).clamp(-_FP4_MAX, _FP4_MAX))
    return (q * scale).reshape(shape).to(dtype)


# ---------------------------------------------------------------------------
# Sparse gather attention with learned sink.
# ---------------------------------------------------------------------------


def sparse_attention_eager(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference sparse_attn semantics in plain torch (fp32 math).

    q [b,m,h,d] bf16; kv [b,n,d] bf16 (latent K=V); topk_idxs [b,m,topk] int
    with -1 = invalid; attn_sink [h] fp32. The sink is a virtual entry with
    score attn_sink[h] and zero value: it joins the softmax denominator but
    not the max tracking, and the gathered softmax weights round to bf16
    before accumulating (mirroring the kernel's acc_s_cast).
    """
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    valid = topk_idxs >= 0
    safe = topk_idxs.clamp(min=0).reshape(b, m * topk, 1).expand(b, m * topk, d)
    k_gathered = torch.gather(kv, 1, safe).reshape(b, m, topk, d)
    scores = torch.einsum("bmhd,bmtd->bmht", q.float(), k_gathered.float()) * scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    max_score = scores.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - max_score)
    exp_scores = torch.where(valid.unsqueeze(2), exp_scores, torch.zeros_like(exp_scores))
    denom = exp_scores.sum(dim=-1, keepdim=True) + torch.exp(
        attn_sink.view(1, 1, h, 1).float() - max_score
    )
    weights = exp_scores.to(torch.bfloat16)
    acc = torch.einsum("bmht,bmtd->bmhd", weights.float(), k_gathered.float())
    return (acc / denom).to(q.dtype)


# ---------------------------------------------------------------------------
# Hadamard rotation (fast_hadamard_transform is not installable here).
# ---------------------------------------------------------------------------

_hadamard_cache: dict[tuple[torch.device, int], torch.Tensor] = {}


def hadamard_matrix(dim: int, device: torch.device) -> torch.Tensor:
    """Sylvester construction, fp32, cached per (device, dim)."""
    key = (device, dim)
    cached = _hadamard_cache.get(key)
    if cached is None:
        if dim & (dim - 1) or dim <= 0:
            raise ValueError(f"hadamard dimension must be a power of 2, got {dim}")
        h = torch.ones(1, 1, dtype=torch.float32, device=device)
        while h.shape[0] < dim:
            h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        cached = h
        _hadamard_cache[key] = cached
    return cached


def hadamard_transform(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Drop-in semantics of fast_hadamard_transform.hadamard_transform."""
    h = hadamard_matrix(x.shape[-1], x.device)
    return (x.float() @ h) * scale
