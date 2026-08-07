"""CPU reference dequantizers for GGUF Q8_0 and IQ2_XS tensors.

Offline verification tool only — deliberately pure stdlib so the torch-free CI
job can run it, and deliberately slow-but-obvious so it can serve as the
readable definition of the block formats. Every arithmetic step reproduces the
C fp32 rounding of llama.cpp's dequantize_row_q8_0 / dequantize_row_iq2_xs
(commit 79bba02a, ggml/src/ggml-quants.c), so outputs are expected to be
bit-exact, not merely close.

Block layouts (ggml/src/ggml-common.h):
  Q8_0   : { ggml_half d; int8_t qs[32]; }                    34 B / 32 elems
  IQ2_XS : { ggml_half d; uint16_t qs[32]; uint8_t scales[8]; } 74 B / 256 elems
    - each of the 32 qs codes addresses one 8-element group:
        low 9 bits  -> iq2xs_grid entry (512 x uint64, little-endian bytes)
        high 7 bits -> ksigns_iq2xs sign-mask byte (128 entries)
    - scales[ib32] carries two 4-bit sub-block scales for the 32 elements
      starting at 32*ib32: low nibble for groups l=0,1, high nibble for l=2,3
    - sub-block delta = d * (0.5 + nibble) * 0.25
"""

from __future__ import annotations

import struct

from loader.gguf_quant_tables import IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS

Q8_0_BLOCK_BYTES = 34
Q8_0_BLOCK_ELEMENTS = 32
IQ2_XS_BLOCK_BYTES = 74
IQ2_XS_BLOCK_ELEMENTS = 256


def fp16_bits_to_fp32(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def _f32(value: float) -> float:
    """Round to IEEE-754 binary32 exactly, matching C float arithmetic."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def dequantize_q8_0_block(block: bytes) -> list[float]:
    """32 floats; matches dequantize_row_q8_0 element semantics."""
    if len(block) != Q8_0_BLOCK_BYTES:
        raise ValueError(f"Q8_0 block must be {Q8_0_BLOCK_BYTES} bytes, got {len(block)}")
    d = fp16_bits_to_fp32(struct.unpack("<H", block[:2])[0])
    quants = struct.unpack("<32b", block[2:])
    return [_f32(d * q) for q in quants]


def dequantize_iq2_xs_block(block: bytes) -> list[float]:
    """256 floats; matches dequantize_row_iq2_xs element semantics."""
    if len(block) != IQ2_XS_BLOCK_BYTES:
        raise ValueError(f"IQ2_XS block must be {IQ2_XS_BLOCK_BYTES} bytes, got {len(block)}")
    d = fp16_bits_to_fp32(struct.unpack("<H", block[:2])[0])
    codes = struct.unpack("<32H", block[2:66])
    scales = block[66:74]
    out = [0.0] * IQ2_XS_BLOCK_ELEMENTS
    for ib32 in range(8):
        scale_byte = scales[ib32]
        db0 = _f32(_f32(d * _f32(0.5 + (scale_byte & 0xF))) * 0.25)
        db1 = _f32(_f32(d * _f32(0.5 + (scale_byte >> 4))) * 0.25)
        for group in range(4):
            code = codes[4 * ib32 + group]
            grid = IQ2XS_GRID[code & 511]
            signs = KSIGNS_IQ2XS[code >> 9]
            delta = db0 if group < 2 else db1
            base = 32 * ib32 + 8 * group
            for j in range(8):
                magnitude = (grid >> (8 * j)) & 0xFF
                if signs & KMASK_IQ2XS[j]:
                    magnitude = -magnitude
                out[base + j] = _f32(delta * magnitude)
    return out


def dequantize_q8_0_row(data: bytes) -> list[float]:
    if len(data) % Q8_0_BLOCK_BYTES:
        raise ValueError("Q8_0 row length is not a multiple of the block size")
    out: list[float] = []
    for start in range(0, len(data), Q8_0_BLOCK_BYTES):
        out.extend(dequantize_q8_0_block(data[start : start + Q8_0_BLOCK_BYTES]))
    return out


def dequantize_iq2_xs_row(data: bytes) -> list[float]:
    if len(data) % IQ2_XS_BLOCK_BYTES:
        raise ValueError("IQ2_XS row length is not a multiple of the block size")
    out: list[float] = []
    for start in range(0, len(data), IQ2_XS_BLOCK_BYTES):
        out.extend(dequantize_iq2_xs_block(data[start : start + IQ2_XS_BLOCK_BYTES]))
    return out
