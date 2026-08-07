"""Streaming GGUF checkpoint reader for the DeepSeek-V4-Flash backend.

The GGUF sibling of iterate_safetensors_checkpoint (runtime/loading/common.py).
Same load-bearing contract: one tensor at a time, never the whole file in
host memory (23 GiB host RAM vs an 82 GiB checkpoint). Quantized payloads
(IQ2_XS / Q8_0) stay packed — dequantization happens on-GPU inside the
kernels (see plan D2), never as a BF16 host/GPU copy.

GGML dim order is fastest-first; torch shape is the reverse. A GGUF tensor
with dims (4096, 2048, 256) becomes torch shape (256, 2048, 4096) — the
expert-major [E, ...] layout the fused MoE paths want.
"""

from __future__ import annotations

import mmap
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from loader.gguf_dequant import (
    IQ2_XS_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES,
)
from loader.gguf_header import GgufTensorInfo, read_gguf_header

# Types that must remain packed end-to-end (dequantized only inside kernels).
PACKED_QUANT_TYPES = frozenset({"IQ2_XS", "Q8_0"})
_PLAIN_DTYPES = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "I8": torch.int8,
}


@dataclass(frozen=True)
class GgufTensor:
    """One streamed tensor: raw payload plus enough metadata to interpret it."""

    name: str
    type_name: str
    shape: tuple[int, ...]  # torch order (outermost dim first)
    data: torch.Tensor  # packed uint8 for quant types, typed view otherwise

    @property
    def is_quantized(self) -> bool:
        return self.type_name in PACKED_QUANT_TYPES

    @property
    def numel(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result


def _torch_shape(info: GgufTensorInfo) -> tuple[int, ...]:
    return tuple(reversed(info.dims))


def iterate_gguf_checkpoint(
    path: Path,
    *,
    device: str = "cuda",
    names: set[str] | None = None,
) -> Iterator[GgufTensor]:
    """Yield tensors one at a time; quantized payloads stay packed uint8."""
    header = read_gguf_header(Path(path))
    with Path(path).open("rb") as source:
        mm = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for info in header.tensors:
                if names is not None and info.name not in names:
                    continue
                start = header.absolute_offset(info)
                raw = mm[start : start + info.nbytes]
                if len(raw) != info.nbytes:
                    raise ValueError(f"truncated tensor payload: {info.name}")
                payload = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                dtype = _PLAIN_DTYPES.get(info.type_name)
                if dtype is not None:
                    payload = payload.view(dtype)
                yield GgufTensor(
                    name=info.name,
                    type_name=info.type_name,
                    shape=_torch_shape(info),
                    data=payload.to(device),
                )
        finally:
            mm.close()


def load_gguf_tensors(
    path: Path,
    names: set[str],
    *,
    device: str = "cuda",
) -> dict[str, GgufTensor]:
    """Convenience wrapper for tests/tooling that need a few named tensors."""
    return {t.name: t for t in iterate_gguf_checkpoint(path, device=device, names=names)}


# ---------------------------------------------------------------------------
# Vectorized host-side dequantizers (numpy). Used by offline tooling and the
# parity tests to materialize reference weights; NOT the serving path.
# ---------------------------------------------------------------------------


def dequantize_q8_0_packed(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Q8_0 packed uint8 bytes -> float32 tensor of logical shape."""
    raw = packed.cpu().numpy()
    if raw.size % Q8_0_BLOCK_BYTES:
        raise ValueError("packed Q8_0 payload is not a multiple of the block size")
    blocks = raw.reshape(-1, Q8_0_BLOCK_BYTES)
    d = np.ascontiguousarray(blocks[:, :2]).view(np.float16).astype(np.float32).reshape(-1)
    qs = np.ascontiguousarray(blocks[:, 2:]).view(np.int8).astype(np.float32)
    values = (d[:, None] * qs).reshape(-1)
    numel = 1
    for dim in shape:
        numel *= dim
    if values.size != numel:
        raise ValueError(f"Q8_0 dequant size {values.size} != numel {numel}")
    return torch.from_numpy(values).reshape(shape)


def dequantize_iq2_xs_packed(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """IQ2_XS packed uint8 bytes -> float32 tensor of logical shape."""
    from loader.gguf_quant_tables import IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS

    raw = packed.cpu().numpy()
    if raw.size % IQ2_XS_BLOCK_BYTES:
        raise ValueError("packed IQ2_XS payload is not a multiple of the block size")
    n_blocks = raw.size // IQ2_XS_BLOCK_BYTES
    blocks = raw.reshape(n_blocks, IQ2_XS_BLOCK_BYTES)
    d = np.ascontiguousarray(blocks[:, :2]).view(np.float16).astype(np.float32).reshape(-1)
    codes = np.ascontiguousarray(blocks[:, 2:66]).view(np.uint16)  # (B, 32)
    scales = blocks[:, 66:74]  # (B, 8) uint8

    grid = np.array(IQ2XS_GRID, dtype=np.uint64)
    ksigns = np.array(KSIGNS_IQ2XS, dtype=np.uint8)
    kmask = np.array(KMASK_IQ2XS, dtype=np.uint8)

    grid_idx = codes & np.uint16(511)
    sign_idx = codes >> np.uint16(9)
    # (B, 32, 8): the 8 magnitude bytes of each selected grid entry
    magnitudes = np.zeros((*codes.shape, 8), dtype=np.uint8)
    for j in range(8):
        magnitudes[..., j] = (grid[grid_idx] >> np.uint64(8 * j)) & np.uint64(0xFF)
    sign_bytes = ksigns[sign_idx]  # (B, 32)
    sign_mask = (sign_bytes[..., None] & kmask[None, None, :]) != 0
    signed = np.where(sign_mask, -magnitudes.astype(np.float32), magnitudes.astype(np.float32))

    # sub-block deltas: scales[ib32] carries two 4-bit nibbles; groups l=0,1
    # of a sub-block use the low nibble, groups l=2,3 the high one (C: db[l/2])
    nib_lo = (scales & 0xF).astype(np.float32)
    nib_hi = (scales >> 4).astype(np.float32)
    db0 = d[:, None] * (0.5 + nib_lo) * 0.25  # (B, 8)
    db1 = d[:, None] * (0.5 + nib_hi) * 0.25  # (B, 8)
    code_idx = np.arange(32)
    subblock = code_idx // 4
    low_half = (code_idx % 4) < 2
    deltas = np.where(low_half[None, :], db0[:, subblock], db1[:, subblock])
    values = (signed * deltas[:, :, None]).reshape(-1)
    numel = 1
    for dim in shape:
        numel *= dim
    if values.size != numel:
        raise ValueError(f"IQ2_XS dequant size {values.size} != numel {numel}")
    return torch.from_numpy(values).reshape(shape)
