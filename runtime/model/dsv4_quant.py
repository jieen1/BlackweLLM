"""Torch dequantizers for GGUF Q8_0 / IQ2_XS payloads.

The eager correctness scaffold of the DSV4 model graph: consumes the packed
bytes that ``runtime/loading/gguf.py`` streams off disk and produces fp32
values with the exact semantics of the bit-exact reference chain
(llama.cpp C <=> pure-Python reference <=> numpy vectorized -- see
``loader/gguf_dequant.py`` and ``tests/test_gguf_dequant_golden.py``).

Phase 3 replaces these call sites with fused dequant-GEMM kernels; those
kernels must keep these numerics, which is why this module stays as the
executable definition of "correct dequant" for the model graph.

Device note: lookup tables are materialized lazily per device.
"""

from __future__ import annotations

import torch

from loader.gguf_quant_tables import IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS

Q8_0_BLOCK_BYTES = 34
Q8_0_BLOCK_ELEMENTS = 32
IQ2_XS_BLOCK_BYTES = 74
IQ2_XS_BLOCK_ELEMENTS = 256

_tables_cache: dict[torch.device, dict[str, torch.Tensor]] = {}


def _tables(device: torch.device) -> dict[str, torch.Tensor]:
    cached = _tables_cache.get(device)
    if cached is None:
        cached = {
            "grid": torch.tensor(IQ2XS_GRID, dtype=torch.int64, device=device),
            "ksigns": torch.tensor(KSIGNS_IQ2XS, dtype=torch.int32, device=device),
            "kmask": torch.tensor(KMASK_IQ2XS, dtype=torch.int32, device=device),
            "subblock": torch.arange(IQ2_XS_BLOCK_ELEMENTS // 8, device=device) // 4,
            "low_half": (torch.arange(IQ2_XS_BLOCK_ELEMENTS // 8, device=device) % 4) < 2,
        }
        _tables_cache[device] = cached
    return cached


def _fp16_scale_column(block_bytes: torch.Tensor) -> torch.Tensor:
    """(N, 2) uint8 little-endian fp16 scales -> (N,) fp32."""
    return block_bytes.contiguous().view(torch.float16).to(torch.float32).squeeze(-1)


def dequantize_q8_0(packed: torch.Tensor) -> torch.Tensor:
    """Q8_0 packed uint8 bytes -> flat fp32 values.

    Block: {fp16 d; int8 qs[32]}. y = d * qs, matching the reference chain
    bit-for-bit (fp32 multiply, sign-of-zero preserved).
    """
    if packed.dtype != torch.uint8:
        raise ValueError(f"Q8_0 payload must be uint8, got {packed.dtype}")
    flat = packed.reshape(-1)
    if flat.numel() % Q8_0_BLOCK_BYTES:
        raise ValueError("Q8_0 payload is not a multiple of the 34-byte block")
    blocks = flat.reshape(-1, Q8_0_BLOCK_BYTES)
    d = _fp16_scale_column(blocks[:, :2])
    qs = blocks[:, 2:].view(torch.int8).to(torch.float32)
    return (d.unsqueeze(1) * qs).reshape(-1)


def dequantize_iq2_xs(packed: torch.Tensor) -> torch.Tensor:
    """IQ2_XS packed uint8 bytes -> flat fp32 values.

    Block: {fp16 d; uint16 codes[32]; uint8 scales[8]}. Each code selects an
    8-magnitude grid entry (low 9 bits) and a sign mask (high 7 bits); the
    sub-block delta is d * (0.5 + nibble) * 0.25 with the low nibble serving
    groups 0,1 and the high nibble groups 2,3 of each 32-element sub-block.
    """
    if packed.dtype != torch.uint8:
        raise ValueError(f"IQ2_XS payload must be uint8, got {packed.dtype}")
    flat = packed.reshape(-1)
    if flat.numel() % IQ2_XS_BLOCK_BYTES:
        raise ValueError("IQ2_XS payload is not a multiple of the 74-byte block")
    blocks = flat.reshape(-1, IQ2_XS_BLOCK_BYTES)
    device = flat.device
    tables = _tables(device)

    d = _fp16_scale_column(blocks[:, :2])
    codes = blocks[:, 2:66].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    scales = blocks[:, 66:74].to(torch.int32)

    entries = tables["grid"][codes & 511]  # (N, 32) int64
    magnitudes = torch.stack([(entries >> (8 * j)) & 0xFF for j in range(8)], dim=-1).to(
        torch.float32
    )
    sign_bytes = tables["ksigns"][codes >> 9]  # (N, 32)
    sign_mask = (sign_bytes.unsqueeze(-1) & tables["kmask"]) != 0
    signed = torch.where(sign_mask, -magnitudes, magnitudes)

    lo = (scales & 0xF).to(torch.float32)
    hi = (scales >> 4).to(torch.float32)
    db0 = d.unsqueeze(1) * (0.5 + lo) * 0.25  # (N, 8)
    db1 = d.unsqueeze(1) * (0.5 + hi) * 0.25
    subblock = tables["subblock"]
    deltas = torch.where(tables["low_half"], db0[:, subblock], db1[:, subblock])
    return (signed * deltas.unsqueeze(-1)).reshape(-1)


def q8_0_weight(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Dequantize a Q8_0 linear weight into logical torch order."""
    return dequantize_q8_0(packed).reshape(shape)


def iq2_xs_weight(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Dequantize an IQ2_XS weight into logical torch order."""
    return dequantize_iq2_xs(packed).reshape(shape)
