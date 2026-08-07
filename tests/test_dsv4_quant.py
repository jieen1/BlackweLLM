"""Bit-exact tests for runtime/model/dsv4_quant.py against the numpy
reference (which is itself bit-exact with llama.cpp, see
tests/test_gguf_dequant_golden.py)."""

from __future__ import annotations

import random
import struct
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from runtime.loading.gguf import (  # noqa: E402
    dequantize_iq2_xs_packed,
    dequantize_q8_0_packed,
    load_gguf_tensors,
)
from runtime.model.dsv4_quant import dequantize_iq2_xs, dequantize_q8_0  # noqa: E402

REAL_GGUF = Path(
    "/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf"
)


def _bits(values) -> list[int]:
    return [struct.unpack("<I", struct.pack("<f", v))[0] for v in values]


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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_q8_0_bit_exact(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no GPU")
    rng = random.Random(11)
    raw = _random_q8_0_bytes(rng, 16)
    reference = dequantize_q8_0_packed(
        torch.frombuffer(bytearray(raw), dtype=torch.uint8), (16, 32)
    )
    ours = dequantize_q8_0(torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device))
    assert _bits(reference.flatten().tolist()) == _bits(ours.cpu().tolist())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_iq2_xs_bit_exact(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("no GPU")
    rng = random.Random(12)
    raw = _random_iq2_xs_bytes(rng, 8)
    reference = dequantize_iq2_xs_packed(
        torch.frombuffer(bytearray(raw), dtype=torch.uint8), (8, 256)
    )
    ours = dequantize_iq2_xs(torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device))
    assert _bits(reference.flatten().tolist()) == _bits(ours.cpu().tolist())


def test_payload_validation() -> None:
    with pytest.raises(ValueError, match="multiple of the 34-byte block"):
        dequantize_q8_0(torch.zeros(35, dtype=torch.uint8))
    with pytest.raises(ValueError, match="multiple of the 74-byte block"):
        dequantize_iq2_xs(torch.zeros(75, dtype=torch.uint8))
    with pytest.raises(ValueError, match="must be uint8"):
        dequantize_q8_0(torch.zeros(34, dtype=torch.int8))


@pytest.mark.skipif(not REAL_GGUF.exists(), reason="GGUF download not present")
def test_real_tensor_q8_0_bit_exact() -> None:
    tensors = load_gguf_tensors(REAL_GGUF, {"blk.0.attn_q_a.weight"}, device="cpu")
    packed = tensors["blk.0.attn_q_a.weight"].data
    # first 64 blocks (enough to be meaningful without dequantizing 4 MiB)
    head = packed[: 64 * 34]
    reference = dequantize_q8_0_packed(head, (64, 32))
    ours = dequantize_q8_0(head)
    assert _bits(reference.flatten().tolist()) == _bits(ours.tolist())


@pytest.mark.skipif(not REAL_GGUF.exists(), reason="GGUF download not present")
def test_real_tensor_iq2_xs_bit_exact() -> None:
    tensors = load_gguf_tensors(REAL_GGUF, {"blk.0.ffn_gate_exps.weight"}, device="cpu")
    packed = tensors["blk.0.ffn_gate_exps.weight"].data
    head = packed[: 32 * 74]
    reference = dequantize_iq2_xs_packed(head, (32, 256))
    ours = dequantize_iq2_xs(head)
    assert _bits(reference.flatten().tolist()) == _bits(ours.tolist())
