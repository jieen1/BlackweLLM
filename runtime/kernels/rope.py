"""Triton NeoX-style rotary embedding -- zero external dependency.

Ported from vLLM's compiled CUDA kernel (csrc/libtorch_stable/
pos_encoding_kernels.cu, ``apply_token_rotary_embedding``/
``rotary_embedding_kernel``), NeoX-only (the only style Laguna ever
constructs its ``RotaryEmbedding`` with -- see
``runtime/model/laguna_decoder.py``'s ``get_rope(..., is_neox_style=True,
...)`` call). Traced against that source, not re-derived: for each token
at position ``pos`` and each head, splits the first ``rot_dim`` elements
of the head into two halves at ``[0, embed_dim)`` / ``[embed_dim,
rot_dim)`` (``embed_dim = rot_dim // 2``) and rotates them by
``(cos, sin) = cos_sin_cache[pos, :embed_dim], cos_sin_cache[pos,
embed_dim:rot_dim]``. Elements at ``[rot_dim, head_size)`` (the
partial-rotary tail, if ``rot_dim < head_size``) are left untouched.

This is the one entry point in the runtime that applies rotation to a
raw, flat ``[num_tokens, num_heads * head_size]`` tensor directly (no
Q/K pairing) -- needed by ``runtime/model/laguna_dflash_model.py``'s
``precompute_and_store_context_kv``, which rotates a K-only tensor
flattened across all draft layers in one batch. ``LagunaAttentionSelfBuilt``
(runtime/model/laguna_decoder.py) calls it twice per forward (once for Q,
once for K) instead of vLLM's combined-Q+K ``ops.rotary_embedding`` call --
same math, independent tensors, no need to extend this kernel for the
combined case.

``compute_cos_sin_cache_default``/``compute_cos_sin_cache_yarn`` below
(阶段7, vLLM removal plan) replace vLLM's ``get_rope()`` cache
CONSTRUCTION (a one-time, load-time computation -- not the hot path this
kernel serves). Ported from vLLM's real source
(vllm/model_executor/layers/rotary_embedding/base.py's
``_compute_inv_freq``/``_compute_cos_sin_cache``, yarn_scaling_rope.py's
``YaRNScalingRotaryEmbedding``, common.py's ``yarn_find_correction_range``/
``yarn_get_mscale``/``yarn_linear_ramp_mask``) and verified bit-exact
against vLLM's real, live-constructed ``cos_sin_cache`` tensor for both of
the two rope_type values Laguna's real config actually uses ("default" for
sliding_attention layers and the DFlash draft model, "yarn" for
full_attention layers) -- not just the formulas transcribed correctly, the
actual output tensors compared with ``torch.equal``. One subtlety that
would have been silently wrong without that live comparison: vLLM's
``get_rope`` YaRN branch only reads an ``attn_factor`` key out of
``rope_parameters`` (default ``1.0`` if absent) -- but Laguna's checkpoint
config names the equivalent field ``attention_factor`` (extra syllable),
which vLLM's dispatcher does NOT recognize and silently ignores. Laguna's
``attention_factor`` value happens to exactly equal
``yarn_get_mscale(scaling_factor)`` anyway (it was derived from that same
formula when the checkpoint's config.json was generated), so the discarded
value coincidentally doesn't change the result for this specific
checkpoint -- but the functions below correctly ignore
``attention_factor``/default ``attn_factor=1.0`` to match vLLM's real
(if arguably surprising) behavior, not what the config's field naming
would suggest.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    POS_ptr,
    X_ptr,
    COS_SIN_ptr,
    stride_x_row,
    cos_sin_stride,
    head_size: tl.constexpr,
    embed_dim: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    pos = tl.load(POS_ptr + row)

    idx = tl.arange(0, BLOCK)
    mask = idx < NUM_HEADS * embed_dim
    head_idx = idx // embed_dim
    rot_offset = idx % embed_dim

    cos_sin_base = COS_SIN_ptr + pos * cos_sin_stride
    cos = tl.load(cos_sin_base + rot_offset, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(cos_sin_base + embed_dim + rot_offset, mask=mask, other=0.0).to(tl.float32)

    x_base = X_ptr + row * stride_x_row + head_idx * head_size
    x_off = rot_offset
    y_off = embed_dim + rot_offset

    x_val = tl.load(x_base + x_off, mask=mask, other=0.0).to(tl.float32)
    y_val = tl.load(x_base + y_off, mask=mask, other=0.0).to(tl.float32)

    new_x = x_val * cos - y_val * sin
    new_y = y_val * cos + x_val * sin

    tl.store(x_base + x_off, new_x.to(tl.bfloat16), mask=mask)
    tl.store(x_base + y_off, new_y.to(tl.bfloat16), mask=mask)


def apply_rotary_embedding_inplace(
    positions: torch.Tensor,
    x: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
) -> None:
    """In-place NeoX rotary embedding on a flat ``[num_tokens, num_heads *
    head_size]`` tensor. ``rot_dim = cos_sin_cache.shape[-1]`` may be less
    than ``head_size`` (partial rotary factor); the tail per head is left
    untouched, matching vLLM's op.
    """
    assert x.is_contiguous()
    num_tokens = x.shape[0]
    num_heads = x.shape[1] // head_size
    rot_dim = cos_sin_cache.shape[-1]
    embed_dim = rot_dim // 2

    if cos_sin_cache.dtype != x.dtype:
        cos_sin_cache = cos_sin_cache.to(dtype=x.dtype)

    BLOCK = triton.next_power_of_2(num_heads * embed_dim)
    _rope_kernel[(num_tokens,)](
        positions,
        x,
        cos_sin_cache,
        x.stride(0),
        cos_sin_cache.stride(0),
        head_size=head_size,
        embed_dim=embed_dim,
        NUM_HEADS=num_heads,
        BLOCK=BLOCK,
    )


def compute_cos_sin_cache_default(
    rotary_dim: int,
    max_position: int,
    base: float,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    """Plain (non-scaled) RoPE cache. Verbatim port of vLLM's
    ``RotaryEmbeddingBase._compute_inv_freq``/``_compute_cos_sin_cache``.
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float, device=device) / rotary_dim)
    )
    t = torch.arange(max_position, dtype=torch.float, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache.to(dtype)


def _yarn_find_correction_dim(
    num_rotations: int, dim: int, base: float, max_position_embeddings: int
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float,
    max_position_embeddings: int,
    truncate: bool,
) -> tuple[float, float]:
    low = _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim, dtype=dtype, device=device) - low) / (high - low)
    return torch.clamp(linear_func, 0, 1)


def _yarn_get_mscale(scale: float) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def compute_cos_sin_cache_yarn(
    rotary_dim: int,
    original_max_position: int,
    base: float,
    scaling_factor: float,
    dtype: torch.dtype,
    device: torch.device | str,
    *,
    extrapolation_factor: float = 1.0,
    attn_factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
    truncate: bool = True,
) -> torch.Tensor:
    """YaRN-scaled RoPE cache. Verbatim port of vLLM's
    ``YaRNScalingRotaryEmbedding``. ``attn_factor`` intentionally defaults
    to 1.0 and is NOT read from a checkpoint's ``attention_factor`` config
    field -- see this module's docstring for why that's correct, not a gap.
    """
    mscale = _yarn_get_mscale(scaling_factor) * attn_factor

    pos_freqs = base ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float, device=device) / rotary_dim
    )
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

    low, high = _yarn_find_correction_range(
        beta_fast, beta_slow, rotary_dim, base, original_max_position, truncate
    )
    inv_freq_mask = (
        1 - _yarn_linear_ramp_mask(low, high, rotary_dim // 2, torch.float, device)
    ) * extrapolation_factor
    inv_freq = (
        inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
    )

    t = torch.arange(original_max_position * scaling_factor, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos() * mscale
    sin = freqs.sin() * mscale
    cache = torch.cat((cos, sin), dim=-1)
    return cache.to(dtype)


class SelfBuiltRotaryEmbedding(torch.nn.Module):
    """Replaces vLLM's ``get_rope(...)``-constructed ``RotaryEmbedding``
    object in ``LagunaAttentionSelfBuilt`` (阶段7, vLLM removal plan).
    Holds a ``cos_sin_cache`` built by ``compute_cos_sin_cache_default``/
    ``_yarn`` above (bit-exact verified against vLLM's real output) and
    applies it via ``apply_rotary_embedding_inplace``, called twice (once
    for Q, once for K -- same math as vLLM's combined-tensor
    ``ops.rotary_embedding`` call, no need for this kernel to support a
    combined call shape). Exposes ``head_size``/``cos_sin_cache``/
    ``is_neox_style`` under the same names vLLM's ``RotaryEmbedding``
    does, since ``LagunaDraftModelSelfBuilt._build_fused_kv_buffers``
    (runtime/model/laguna_dflash_model.py) reads them directly.
    """

    def __init__(self, head_size: int, cos_sin_cache: torch.Tensor, is_neox_style: bool) -> None:
        super().__init__()
        assert is_neox_style, "apply_rotary_embedding_inplace only implements NeoX style"
        self.head_size = head_size
        self.is_neox_style = is_neox_style
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        apply_rotary_embedding_inplace(positions, query, self.head_size, self.cos_sin_cache)
        apply_rotary_embedding_inplace(positions, key, self.head_size, self.cos_sin_cache)
        return query, key
