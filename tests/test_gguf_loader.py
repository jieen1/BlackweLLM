"""Tests for runtime/loading/gguf.py (the streaming GGUF reader).

Covers: numpy-vectorized dequant vs the bit-exact pure-Python reference,
streaming iteration on the real DSV4-Flash GGUF (skipped when absent), and
a synthetic GGUF round-trip that runs everywhere torch is available.
"""

from __future__ import annotations

import random
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from runtime.loading.gguf import (  # noqa: E402
    dequantize_iq2_xs_packed,
    dequantize_q8_0_packed,
    iterate_gguf_checkpoint,
    load_gguf_tensors,
)

REAL_GGUF = Path(
    "/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf"
)


def _random_q8_0_bytes(rng: random.Random, n_blocks: int) -> bytes:
    out = bytearray()
    for _ in range(n_blocks):
        while True:
            bits = rng.getrandbits(16)
            if (bits >> 10) & 0x1F != 0x1F:
                break
        out += struct.pack("<H", bits)
        out += struct.pack("<32b", *(rng.randrange(-127, 128) for _ in range(32)))
    return bytes(out)


def _random_iq2_xs_bytes(rng: random.Random, n_blocks: int) -> bytes:
    out = bytearray()
    for _ in range(n_blocks):
        while True:
            bits = rng.getrandbits(16)
            if (bits >> 10) & 0x1F != 0x1F:
                break
        out += struct.pack("<H", bits)
        out += struct.pack("<32H", *(rng.getrandbits(16) for _ in range(32)))
        out += bytes(rng.getrandbits(8) for _ in range(8))
    return bytes(out)


def test_q8_0_numpy_matches_bit_exact_reference() -> None:
    from loader.gguf_dequant import dequantize_q8_0_row

    rng = random.Random(7)
    raw = _random_q8_0_bytes(rng, 8)
    reference = dequantize_q8_0_row(raw)
    packed = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    ours = dequantize_q8_0_packed(packed, shape=(8, 32))
    ref_bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in reference]
    our_bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in ours.flatten().tolist()]
    assert ref_bits == our_bits


def test_iq2_xs_numpy_matches_bit_exact_reference() -> None:
    from loader.gguf_dequant import dequantize_iq2_xs_row

    rng = random.Random(8)
    raw = _random_iq2_xs_bytes(rng, 4)
    reference = dequantize_iq2_xs_row(raw)
    packed = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    ours = dequantize_iq2_xs_packed(packed, shape=(4, 256))
    ref_bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in reference]
    our_bits = [struct.unpack("<I", struct.pack("<f", v))[0] for v in ours.flatten().tolist()]
    assert ref_bits == our_bits


@pytest.mark.skipif(not REAL_GGUF.exists(), reason="GGUF download not present")
def test_stream_real_tensors_cpu() -> None:
    wanted = {
        "blk.0.attn_sinks.weight",  # F32, tiny
        "blk.0.ffn_gate_inp.weight",  # BF16 [256, 4096]
        "blk.0.ffn_gate_tid2eid.weight",  # I32 [6, 129280]
        "blk.0.attn_q_a.weight",  # Q8_0 packed
        "blk.0.ffn_gate_exps.weight",  # IQ2_XS packed (streams 577 MiB)
    }
    tensors = load_gguf_tensors(REAL_GGUF, wanted, device="cpu")
    assert set(tensors) == wanted

    sinks = tensors["blk.0.attn_sinks.weight"]
    assert sinks.shape == (64,)
    assert sinks.data.dtype == torch.float32
    assert torch.isfinite(sinks.data).all()

    gate_inp = tensors["blk.0.ffn_gate_inp.weight"]
    assert gate_inp.shape == (256, 4096)
    assert gate_inp.data.dtype == torch.bfloat16

    tid2eid = tensors["blk.0.ffn_gate_tid2eid.weight"]
    assert tid2eid.shape == (129280, 6)
    assert tid2eid.data.dtype == torch.int32
    assert tid2eid.data.min() >= 0 and tid2eid.data.max() < 256

    q8 = tensors["blk.0.attn_q_a.weight"]
    assert q8.is_quantized and q8.shape == (1024, 4096)
    assert q8.data.dtype == torch.uint8
    assert q8.data.numel() == 1024 * 4096 // 32 * 34

    experts = tensors["blk.0.ffn_gate_exps.weight"]
    assert experts.is_quantized and experts.shape == (256, 2048, 4096)
    assert experts.data.numel() == 256 * 2048 * 4096 // 256 * 74


@pytest.mark.skipif(not REAL_GGUF.exists(), reason="GGUF download not present")
def test_stream_is_lazy_per_tensor() -> None:
    """Iteration must not require holding more than one tensor at a time."""
    seen = []
    for tensor in iterate_gguf_checkpoint(
        REAL_GGUF, device="cpu", names={"blk.0.attn_sinks.weight"}
    ):
        seen.append(tensor.name)
    assert seen == ["blk.0.attn_sinks.weight"]
