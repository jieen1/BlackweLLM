"""Triton KV pack kernel for the DSV4 compressed-MLA FP8 page layout.

The sparkinfer fork's ``attention.compressed_mla`` kernels consume one
latent KV entry per token in this byte layout (authoritative definition:
``compressed_reference.py`` in the fork):

    payload region: 576 B/token = 448 e4m3 nope bytes + 128 B bf16 rope
    scale region:    8 B/token  = 7 ue8m0 scale bytes (one per 64-dim
                  nope group) + 1 pad byte
    page:   [page_size * 576 payload][page_size * 8 scales][pad to
            576 multiple]

This kernel quantizes and packs raw post-norm, post-rope latent rows
(bf16 [N, 512]) into that layout. The quantization is a bit-exact
reproduction of ``dsv4_attention.act_quant_simulate(x, 64, ue8m0=True)``
-- the eager graph is the executable definition, and the Phase 3 parity
gate is drawn against it, so the kernel must not invent its own rounding:

    amax per 64-block, floored at 1e-4;
    v = amax * fp32(1/448);  k = ceil(log2(v)) via the exact frexp
        bit-trick (not logf -- libm is not exact at power-of-two
        boundaries, torch.frexp is);
    scale = 2^k;  q = round-to-nearest-even e4m3(clamp(x / scale, +-448))
        (the e4m3 cast is the hardware cvt.rn.satfinite, verified
        bit-identical to torch's ``.to(torch.float8_e4m3fn)``);
    ue8m0 byte = k + 127 (decoded as 2^(byte - 127) by the kernels).

The rope 64 dims are stored as raw bf16 little-endian bytes.

One program per token; pages are addressed by flat token id (page =
id // page_size, slot = id % page_size), which is exactly how the slot
pool's window ring and compressed regions are indexed.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
NOPE_GROUP_SIZE = 64
NOPE_GROUPS = NOPE_DIM // NOPE_GROUP_SIZE
#: Payload bytes per token: 448 e4m3 + 64 bf16 rope (2 B each).
PAYLOAD_BYTES_PER_TOKEN = NOPE_DIM + ROPE_DIM * 2
#: Scale bytes per token: 7 ue8m0 + 1 pad.
SCALE_BYTES_PER_TOKEN = 8
BYTES_PER_TOKEN = PAYLOAD_BYTES_PER_TOKEN + SCALE_BYTES_PER_TOKEN
#: SGLang-compatible DSV4 physical page (kernel contract).
DSV4_PAGE_SIZE = 256
UE8M0_BIAS = 127
FP8_MAX = 448.0


def page_nbytes(page_size: int) -> int:
    """Padded byte count of one compressed-MLA KV page (576-multiple)."""
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    unpadded = page_size * BYTES_PER_TOKEN
    return (
        (unpadded + PAYLOAD_BYTES_PER_TOKEN - 1)
        // PAYLOAD_BYTES_PER_TOKEN
        * PAYLOAD_BYTES_PER_TOKEN
    )


def scale_region_offset(page_size: int) -> int:
    """Byte offset at which the ue8m0 scale region starts inside a page."""
    return page_size * PAYLOAD_BYTES_PER_TOKEN


@triton.jit
def _pack_latent_kv_kernel(
    kv_ptr,  # [N, 512] bf16, row-major
    buf_ptr,  # [num_pages, page_nbytes] uint8 page buffer (flat base)
    token_ids_ptr,  # [N] int64 flat token ids (page = id // page_size)
    inv_fp8_max,  # fp32 constant 1/448 (must match eager's Python 1/448)
    page_size: tl.constexpr,
    page_nbytes: tl.constexpr,
    scale_offset: tl.constexpr,
    payload_bytes: tl.constexpr,
    scale_bytes: tl.constexpr,
    head_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    nope_groups: tl.constexpr,
    nope_group_size: tl.constexpr,
    rope_dim: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = tl.load(token_ids_ptr + pid)
    page = tid // page_size
    slot = tid % page_size
    base = page * page_nbytes + slot * payload_bytes
    sbase = page * page_nbytes + scale_offset + slot * scale_bytes

    nope_offs = tl.arange(0, head_dim)  # 512: masked to the 448 nope dims
    x_nope = tl.reshape(
        tl.load(
            kv_ptr + pid * head_dim + nope_offs,
            mask=nope_offs < nope_dim,
            other=0.0,
        ),
        (nope_groups + 1, nope_group_size),
    )
    rope_offs = tl.arange(0, rope_dim)
    x_rope = tl.load(kv_ptr + pid * head_dim + nope_dim + rope_offs)

    amax = tl.max(tl.abs(x_nope), axis=1)
    amax = tl.maximum(amax, 1e-4)
    v = amax * inv_fp8_max
    bits = v.to(tl.uint32, bitcast=True)
    exp_bits = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    k = exp_bits.to(tl.int32) - 127 + tl.where(mant != 0, 1, 0)
    scale = (k + 127).to(tl.uint32) << 23
    scale_f = scale.to(tl.float32, bitcast=True)

    scale_2d = tl.reshape(scale_f, (nope_groups + 1, 1))
    q8 = tl.cast(x_nope / scale_2d, tl.float8e4nv)
    q8u = q8.to(tl.uint8, bitcast=True)
    tl.store(
        buf_ptr + base + nope_offs,
        tl.reshape(q8u, (head_dim,)),
        mask=nope_offs < nope_dim,
    )

    rope_u16 = x_rope.to(tl.bfloat16).to(tl.uint16, bitcast=True)
    byte_off = nope_dim + rope_offs * 2
    tl.store(buf_ptr + base + byte_off, (rope_u16 & 0xFF).to(tl.uint8))
    tl.store(buf_ptr + base + byte_off + 1, (rope_u16 >> 8).to(tl.uint8))

    scale_offs = tl.arange(0, scale_bytes)
    scale_byte = (k + 127).to(tl.uint8)  # UE8M0_BIAS: decoded as 2^(byte - 127)
    tl.store(buf_ptr + sbase + scale_offs, scale_byte, mask=scale_offs < nope_groups)
    tl.store(buf_ptr + sbase + scale_bytes - 1, 0)


def pack_latent_kv(
    kv: torch.Tensor,
    page_buffer: torch.Tensor,
    token_ids: torch.Tensor,
    page_size: int = DSV4_PAGE_SIZE,
) -> None:
    """Quantize+pack bf16 latent rows into the compressed-MLA page layout.

    ``kv`` is [N, 512] bf16 (post kv_norm, post rope; the nope part is
    still raw -- the kernel performs the eager act_quant_simulate
    quantization). ``page_buffer`` is [num_pages, page_nbytes] uint8
    (see :func:`page_nbytes`); each ``token_ids[i]`` addresses the flat
    token slot written by row ``i``. In-place; returns None.
    """
    if kv.ndim != 2 or kv.shape[-1] != HEAD_DIM:
        raise ValueError(f"kv must have shape [N, {HEAD_DIM}], got {tuple(kv.shape)}")
    if kv.dtype != torch.bfloat16:
        raise ValueError(f"kv must be bf16, got {kv.dtype}")
    n_tokens = kv.shape[0]
    if token_ids.shape != (n_tokens,):
        raise ValueError(f"token_ids must have shape [{n_tokens}], got {tuple(token_ids.shape)}")
    if token_ids.dtype != torch.int64:
        raise ValueError(f"token_ids must be int64, got {token_ids.dtype}")
    pn = page_nbytes(page_size)
    if page_buffer.ndim != 2 or page_buffer.shape[1] != pn:
        raise ValueError(f"page_buffer must be [num_pages, {pn}], got {tuple(page_buffer.shape)}")
    if page_buffer.dtype != torch.uint8:
        raise ValueError(f"page_buffer must be uint8, got {page_buffer.dtype}")
    if n_tokens == 0:
        return
    max_id = int(token_ids.max().item())
    if max_id >= page_buffer.shape[0] * page_size:
        raise ValueError(
            f"token id {max_id} exceeds page capacity {page_buffer.shape[0] * page_size}"
        )
    _pack_latent_kv_kernel[(n_tokens,)](
        kv,
        page_buffer,
        token_ids,
        1.0 / FP8_MAX,
        page_size=page_size,
        page_nbytes=pn,
        scale_offset=scale_region_offset(page_size),
        payload_bytes=PAYLOAD_BYTES_PER_TOKEN,
        scale_bytes=SCALE_BYTES_PER_TOKEN,
        head_dim=HEAD_DIM,
        nope_dim=NOPE_DIM,
        nope_groups=NOPE_GROUPS,
        nope_group_size=NOPE_GROUP_SIZE,
        rope_dim=ROPE_DIM,
    )
