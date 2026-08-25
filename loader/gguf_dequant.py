"""CPU reference dequantizers for the GGUF quantizers used by this runtime.

Offline verification tool only — deliberately pure stdlib so the torch-free CI
job can run it, and deliberately slow-but-obvious so it can serve as the
readable definition of the block formats. Every arithmetic step reproduces the
C fp32 rounding of llama.cpp's dequantize_row_q8_0 / dequantize_row_iq2_xs
(commit 79bba02a, ggml/src/ggml-quants.c), so outputs are expected to be
bit-exact, not merely close.

Block layouts (ggml/src/ggml-common.h):
  Q4_K   : { ggml_half d, dmin; uint8_t scales[12]; qs[128]; }
                                                               144 B / 256 elems
  Q5_K   : Q4_K plus qh[32]                                  176 B / 256 elems
  Q6_K   : { ql[128]; qh[64]; int8_t scales[16]; ggml_half d; }
                                                               210 B / 256 elems
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
Q4_K_BLOCK_BYTES = 144
Q4_K_BLOCK_ELEMENTS = 256
Q5_K_BLOCK_BYTES = 176
Q5_K_BLOCK_ELEMENTS = 256
Q6_K_BLOCK_BYTES = 210
Q6_K_BLOCK_ELEMENTS = 256
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


def _get_scale_min_k4(scales: bytes, index: int) -> tuple[int, int]:
    """Decode one of GGML's eight packed 6-bit K-quant scales/minima."""

    if index < 4:
        return scales[index] & 63, scales[index + 4] & 63
    return (
        (scales[index + 4] & 0xF) | ((scales[index - 4] >> 6) << 4),
        (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
    )


def dequantize_q4_K_block(block: bytes) -> list[float]:
    """256 floats; matches llama.cpp's ``dequantize_row_q4_K``."""

    if len(block) != Q4_K_BLOCK_BYTES:
        raise ValueError(f"Q4_K block must be {Q4_K_BLOCK_BYTES} bytes, got {len(block)}")
    d = fp16_bits_to_fp32(struct.unpack_from("<H", block, 0)[0])
    dmin = fp16_bits_to_fp32(struct.unpack_from("<H", block, 2)[0])
    scales = block[4:16]
    qs = block[16:]
    out = [0.0] * Q4_K_BLOCK_ELEMENTS
    for chunk in range(4):
        sc0, m0 = _get_scale_min_k4(scales, 2 * chunk)
        sc1, m1 = _get_scale_min_k4(scales, 2 * chunk + 1)
        d0 = _f32(d * sc0)
        d1 = _f32(d * sc1)
        min0 = _f32(dmin * m0)
        min1 = _f32(dmin * m1)
        q = qs[32 * chunk : 32 * (chunk + 1)]
        base = 64 * chunk
        for index, value in enumerate(q):
            out[base + index] = _f32(_f32(d0 * (value & 0xF)) - min0)
            out[base + 32 + index] = _f32(_f32(d1 * (value >> 4)) - min1)
    return out


def dequantize_q5_K_block(block: bytes) -> list[float]:
    """256 floats; matches llama.cpp's ``dequantize_row_q5_K``."""

    if len(block) != Q5_K_BLOCK_BYTES:
        raise ValueError(f"Q5_K block must be {Q5_K_BLOCK_BYTES} bytes, got {len(block)}")
    d = fp16_bits_to_fp32(struct.unpack_from("<H", block, 0)[0])
    dmin = fp16_bits_to_fp32(struct.unpack_from("<H", block, 2)[0])
    scales = block[4:16]
    qh = block[16:48]
    ql = block[48:]
    out = [0.0] * Q5_K_BLOCK_ELEMENTS
    for chunk in range(4):
        sc0, m0 = _get_scale_min_k4(scales, 2 * chunk)
        sc1, m1 = _get_scale_min_k4(scales, 2 * chunk + 1)
        d0 = _f32(d * sc0)
        d1 = _f32(d * sc1)
        min0 = _f32(dmin * m0)
        min1 = _f32(dmin * m1)
        low_bit = 1 << (2 * chunk)
        high_bit = 2 << (2 * chunk)
        q = ql[32 * chunk : 32 * (chunk + 1)]
        base = 64 * chunk
        for index, value in enumerate(q):
            q0 = (value & 0xF) + (16 if qh[index] & low_bit else 0)
            q1 = (value >> 4) + (16 if qh[index] & high_bit else 0)
            out[base + index] = _f32(_f32(d0 * q0) - min0)
            out[base + 32 + index] = _f32(_f32(d1 * q1) - min1)
    return out


def dequantize_q6_K_block(block: bytes) -> list[float]:
    """256 floats; matches llama.cpp's ``dequantize_row_q6_K``."""

    if len(block) != Q6_K_BLOCK_BYTES:
        raise ValueError(f"Q6_K block must be {Q6_K_BLOCK_BYTES} bytes, got {len(block)}")
    ql = block[:128]
    qh = block[128:192]
    scales = struct.unpack_from("<16b", block, 192)
    d = fp16_bits_to_fp32(struct.unpack_from("<H", block, 208)[0])
    out = [0.0] * Q6_K_BLOCK_ELEMENTS
    for half in range(2):
        ql_half = ql[64 * half : 64 * (half + 1)]
        qh_half = qh[32 * half : 32 * (half + 1)]
        sc = scales[8 * half : 8 * (half + 1)]
        base = 128 * half
        for index in range(32):
            scale_index = index // 16
            q1 = ((ql_half[index] & 0xF) | ((qh_half[index] & 3) << 4)) - 32
            q2 = ((ql_half[index + 32] & 0xF) | (((qh_half[index] >> 2) & 3) << 4)) - 32
            q3 = ((ql_half[index] >> 4) | (((qh_half[index] >> 4) & 3) << 4)) - 32
            q4 = ((ql_half[index + 32] >> 4) | (((qh_half[index] >> 6) & 3) << 4)) - 32
            out[base + index] = _f32(d * sc[scale_index + 0] * q1)
            out[base + 32 + index] = _f32(d * sc[scale_index + 2] * q2)
            out[base + 64 + index] = _f32(d * sc[scale_index + 4] * q3)
            out[base + 96 + index] = _f32(d * sc[scale_index + 6] * q4)
    return out


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
                magnitude = float((grid >> (8 * j)) & 0xFF)
                if signs & KMASK_IQ2XS[j]:
                    magnitude = -magnitude  # keeps -0.0, matching C float semantics
                out[base + j] = _f32(delta * magnitude)
    return out


def dequantize_q8_0_row(data: bytes) -> list[float]:
    if len(data) % Q8_0_BLOCK_BYTES:
        raise ValueError("Q8_0 row length is not a multiple of the block size")
    out: list[float] = []
    for start in range(0, len(data), Q8_0_BLOCK_BYTES):
        out.extend(dequantize_q8_0_block(data[start : start + Q8_0_BLOCK_BYTES]))
    return out


def _dequantize_row(data: bytes, block_bytes: int, block_fn) -> list[float]:
    if len(data) % block_bytes:
        raise ValueError(f"row length is not a multiple of the block size {block_bytes}")
    out: list[float] = []
    for start in range(0, len(data), block_bytes):
        out.extend(block_fn(data[start : start + block_bytes]))
    return out


def dequantize_q4_K_row(data: bytes) -> list[float]:
    return _dequantize_row(data, Q4_K_BLOCK_BYTES, dequantize_q4_K_block)


def dequantize_q5_K_row(data: bytes) -> list[float]:
    return _dequantize_row(data, Q5_K_BLOCK_BYTES, dequantize_q5_K_block)


def dequantize_q6_K_row(data: bytes) -> list[float]:
    return _dequantize_row(data, Q6_K_BLOCK_BYTES, dequantize_q6_K_block)


def dequantize_iq2_xs_row(data: bytes) -> list[float]:
    if len(data) % IQ2_XS_BLOCK_BYTES:
        raise ValueError("IQ2_XS row length is not a multiple of the block size")
    out: list[float] = []
    for start in range(0, len(data), IQ2_XS_BLOCK_BYTES):
        out.extend(dequantize_iq2_xs_block(data[start : start + IQ2_XS_BLOCK_BYTES]))
    return out
