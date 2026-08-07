"""KV pack kernel parity: packed FP8 layout vs the eager act_quant_simulate.

The eager graph (dsv4_attention.act_quant_simulate, frexp-exact scale +
hardware e4m3 cast) is the executable definition the Phase 3 gate draws
against; this pins the pack kernel's bytes to it bit-exactly. The layout
itself is the sparkinfer fork's compressed-MLA page contract
(payload/scales regions, ue8m0 exponent bias) -- the P3-2 attention
wiring feeds these buffers straight to the fork kernels.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from runtime.kernels.dsv4_kv_pack import (  # noqa: E402
    BYTES_PER_TOKEN,
    HEAD_DIM,
    NOPE_DIM,
    NOPE_GROUP_SIZE,
    NOPE_GROUPS,
    PAYLOAD_BYTES_PER_TOKEN,
    ROPE_DIM,
    SCALE_BYTES_PER_TOKEN,
    UE8M0_BIAS,
    pack_latent_kv,
    page_nbytes,
    scale_region_offset,
)
from runtime.model.dsv4_attention import act_quant_simulate  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


# -- layout constants (pure ints; also runs under the torch-free CI sim) ----


def test_layout_constants() -> None:
    assert PAYLOAD_BYTES_PER_TOKEN == 448 + 64 * 2 == 576
    assert SCALE_BYTES_PER_TOKEN == 8
    assert BYTES_PER_TOKEN == 584
    assert NOPE_GROUPS == 7
    assert NOPE_GROUP_SIZE == 64
    assert scale_region_offset(256) == 256 * 576 == 147456
    assert page_nbytes(256) == 149760  # ceil(256*584/576)*576
    assert page_nbytes(64) == 37440  # ceil(64*584/576)*576 = 65*576
    assert page_nbytes(2) == 1728  # ceil(1168/576)*576 = 3*576
    assert UE8M0_BIAS == 127


def test_page_nbytes_matches_fork_formula() -> None:
    """Same closed form as compressed_mla_page_nbytes in the fork."""
    for ps in (1, 2, 64, 256):
        unpadded = ps * BYTES_PER_TOKEN
        expected = (unpadded + 575) // 576 * 576
        assert page_nbytes(ps) == expected, ps


# -- bit-exact parity with the eager round-trip -----------------------------


def dequantize_entries(
    page_buffer: torch.Tensor, token_ids: torch.Tensor, page_size: int
) -> torch.Tensor:
    """Unpack full [N, 512] fp32 rows from the packed page layout."""
    pn = page_nbytes(page_size)
    so = scale_region_offset(page_size)
    flat = page_buffer.reshape(-1)
    pages = token_ids // page_size
    slots = token_ids % page_size
    token_base = pages * pn + slots * PAYLOAD_BYTES_PER_TOKEN

    nope_offs = token_base[:, None] + torch.arange(NOPE_DIM, device=flat.device)
    rope_offs = token_base[:, None] + NOPE_DIM + torch.arange(ROPE_DIM * 2, device=flat.device)
    scale_offs = (
        pages[:, None] * pn
        + so
        + slots[:, None] * SCALE_BYTES_PER_TOKEN
        + torch.arange(NOPE_GROUPS, device=flat.device)
    )
    nope = (
        flat[nope_offs]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
        .view(-1, NOPE_GROUPS, NOPE_GROUP_SIZE)
        * torch.ldexp(
            torch.ones_like(flat[scale_offs], dtype=torch.float32),
            flat[scale_offs].to(torch.int32) - UE8M0_BIAS,
        ).view(-1, NOPE_GROUPS, 1)
    ).reshape(-1, NOPE_DIM)
    rope = flat[rope_offs].contiguous().view(torch.bfloat16).view(-1, 64).float()
    return torch.cat([nope, rope], dim=-1)


def eager_roundtrip(kv: torch.Tensor) -> torch.Tensor:
    """The eager graph's stored value for the nope part (fp32 view)."""
    out = kv.float().clone()
    out[..., :NOPE_DIM] = act_quant_simulate(kv[..., :NOPE_DIM], 64, ue8m0=True).float()
    return out


def make_page_buffer(num_pages: int, page_size: int, device) -> torch.Tensor:
    return torch.zeros(num_pages, page_nbytes(page_size), dtype=torch.uint8, device=device)


@pytest.mark.parametrize("n_tokens", [1, 7, 129, 513])
@pytest.mark.parametrize("page_size", [64, 256])
def test_pack_matches_eager_bit_exact(n_tokens: int, page_size: int) -> None:
    gen = torch.Generator(device="cuda").manual_seed(20260807 + n_tokens)
    kv = (torch.randn(n_tokens, HEAD_DIM, generator=gen, device="cuda") * 2).to(torch.bfloat16)
    num_pages = (n_tokens + page_size - 1) // page_size
    buf = make_page_buffer(max(num_pages, 1), page_size, kv.device)
    token_ids = torch.arange(n_tokens, dtype=torch.int64, device=kv.device)
    pack_latent_kv(kv, buf, token_ids, page_size=page_size)

    back = dequantize_entries(buf, token_ids, page_size)
    expected = eager_roundtrip(kv)
    assert back.shape == expected.shape
    assert torch.equal(back, expected), (
        "packed dequant diverges from the eager act_quant_simulate round-trip"
    )


def test_pack_small_block_magnitudes() -> None:
    """Blocks whose amax lands near the 1e-4 floor and around powers of two
    (the frexp-vs-logf boundary cases the bit-trick exists for)."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    kv = (torch.randn(64, HEAD_DIM, generator=gen, device="cuda") * 1e-4).to(torch.bfloat16)
    # force a handful of blocks to exact power-of-two amax patterns
    kv[0, :64] = torch.full((64,), 0.0002, device="cuda").to(torch.bfloat16)
    kv[1, :64] = torch.full((64,), 0.0, device="cuda").to(torch.bfloat16)
    kv[2, :64] = torch.full((64,), 224.0, device="cuda").to(torch.bfloat16)
    kv[3, :64] = torch.full((64,), 448.0, device="cuda").to(torch.bfloat16)

    buf = make_page_buffer(1, 64, kv.device)
    token_ids = torch.arange(64, dtype=torch.int64, device=kv.device)
    pack_latent_kv(kv, buf, token_ids, page_size=64)
    back = dequantize_entries(buf, token_ids, 64)
    assert torch.equal(back, eager_roundtrip(kv))


def test_pack_scattered_token_ids_ring_wrap() -> None:
    """Distinct ids land on their own slots (page boundaries crossed), and
    sequential ring-style overwrites are deterministic (calls are ordered;
    a single call must never write one slot twice)."""
    kv = (torch.randn(150, HEAD_DIM, device="cuda") * 2).to(torch.bfloat16)
    token_ids = torch.arange(150, dtype=torch.int64, device="cuda")
    buf = make_page_buffer(3, 64, kv.device)  # 192 slots: 63|64 and 127|128 split
    pack_latent_kv(kv, buf, token_ids, page_size=64)
    back = dequantize_entries(buf, token_ids, 64)
    assert torch.equal(back, eager_roundtrip(kv))

    # sequential ring overwrite: fill a 128-slot ring, then wrap 22 fresh
    # tokens over slots 0..21 -- a later call's bytes win, like the window
    # ring reusing its slots
    ring = (torch.randn(128, HEAD_DIM, device="cuda") * 2).to(torch.bfloat16)
    rid = torch.arange(128, dtype=torch.int64, device="cuda")
    pack_latent_kv(ring, buf, rid, page_size=64)
    new = (torch.randn(22, HEAD_DIM, device="cuda") * 2).to(torch.bfloat16)
    pack_latent_kv(new, buf, torch.arange(22, dtype=torch.int64, device="cuda"), page_size=64)

    back2 = dequantize_entries(buf, rid, 64)
    expected2 = eager_roundtrip(ring)
    expected2[:22] = eager_roundtrip(new)
    assert torch.equal(back2, expected2)


def test_pack_rope_bytes_are_raw_bf16() -> None:
    """The rope section must be the untouched bf16 bit pattern."""
    kv = torch.randn(1, HEAD_DIM, device="cuda").to(torch.bfloat16)
    buf = make_page_buffer(1, 64, kv.device)
    pack_latent_kv(kv, buf, torch.zeros(1, dtype=torch.int64, device="cuda"), page_size=64)
    stored = buf[0, NOPE_DIM : NOPE_DIM + ROPE_DIM * 2].contiguous().view(torch.bfloat16).view(-1)
    assert torch.equal(stored, kv[0, NOPE_DIM:])


def test_pack_empty_noop() -> None:
    buf = make_page_buffer(1, 64, "cuda")
    kv = torch.empty(0, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    ids = torch.empty(0, dtype=torch.int64, device="cuda")
    pack_latent_kv(kv, buf, ids, page_size=64)
    assert torch.count_nonzero(buf) == 0


def test_pack_validates_shapes() -> None:
    buf = make_page_buffer(1, 64, "cuda")
    kv = torch.empty(2, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        pack_latent_kv(
            kv,
            buf,
            torch.tensor([0], dtype=torch.int64, device="cuda"),
            page_size=64,
        )
    with pytest.raises(ValueError):
        pack_latent_kv(
            kv,
            torch.empty(1, 10, dtype=torch.uint8, device="cuda"),
            torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
            page_size=64,
        )
    with pytest.raises(ValueError):
        pack_latent_kv(
            kv,
            buf,
            torch.tensor([0, 64], dtype=torch.int64, device="cuda"),
            page_size=64,
        )
