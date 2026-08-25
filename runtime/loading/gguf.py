"""Streaming GGUF checkpoint reader and correctness-first dequantizers.

The GGUF sibling of iterate_safetensors_checkpoint (runtime/loading/common.py).
Same load-bearing contract: one tensor at a time, never the whole file in
host memory (23 GiB host RAM vs an 82 GiB checkpoint). Quantized payloads
(IQ2_XS / Q4_K / Q5_K / Q6_K / Q8_0) stay packed. The DSV4 path still hands
IQ2_XS bytes to its native kernels; the Qwen3.8 bring-up path additionally
exposes a Torch byte-unpacker that runs on the tensor's existing device.
The unpacker remains an explicit correctness oracle; the SM120 serving path
uses the native packed Q/K kernels when ``QSR_GGUF_NATIVE`` is enabled.

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
    Q4_K_BLOCK_BYTES,
    Q5_K_BLOCK_BYTES,
    Q6_K_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES,
    dequantize_q4_K_row,
    dequantize_q5_K_row,
    dequantize_q6_K_row,
)
from loader.gguf_header import GgufTensorInfo, read_gguf_header

# Types that must remain packed end-to-end (dequantized only inside kernels).
PACKED_QUANT_TYPES = frozenset({"IQ2_XS", "Q4_K", "Q5_K", "Q6_K", "Q8_0"})
GGUF_BLOCK_BYTES = {
    "IQ2_XS": IQ2_XS_BLOCK_BYTES,
    "Q4_K": Q4_K_BLOCK_BYTES,
    "Q5_K": Q5_K_BLOCK_BYTES,
    "Q6_K": Q6_K_BLOCK_BYTES,
    "Q8_0": Q8_0_BLOCK_BYTES,
}
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
                    # plain tensors arrive shaped; packed quant payloads stay flat
                    payload = payload.view(dtype).reshape(_torch_shape(info))
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


def _dequantize_k_packed(
    packed: torch.Tensor,
    shape: tuple[int, ...],
    *,
    type_name: str,
    block_bytes: int,
    row_dequantizer,
) -> torch.Tensor:
    raw = packed.detach().cpu().numpy().tobytes()
    if len(raw) % block_bytes:
        raise ValueError(f"packed {type_name} payload is not a multiple of the block size")
    values = np.asarray(row_dequantizer(raw), dtype=np.float32)
    numel = 1
    for dim in shape:
        numel *= dim
    if values.size != numel:
        raise ValueError(f"{type_name} dequant size {values.size} != numel {numel}")
    return torch.from_numpy(values.reshape(shape))


def dequantize_q4_K_packed(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """CPU reference materialization for GGML Q4_K."""

    return _dequantize_k_packed(
        packed,
        shape,
        type_name="Q4_K",
        block_bytes=Q4_K_BLOCK_BYTES,
        row_dequantizer=dequantize_q4_K_row,
    )


def dequantize_q5_K_packed(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """CPU reference materialization for GGML Q5_K."""

    return _dequantize_k_packed(
        packed,
        shape,
        type_name="Q5_K",
        block_bytes=Q5_K_BLOCK_BYTES,
        row_dequantizer=dequantize_q5_K_row,
    )


def dequantize_q6_K_packed(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """CPU reference materialization for GGML Q6_K."""

    return _dequantize_k_packed(
        packed,
        shape,
        type_name="Q6_K",
        block_bytes=Q6_K_BLOCK_BYTES,
        row_dequantizer=dequantize_q6_K_row,
    )


def _reshape_dequantized(
    values: torch.Tensor, shape: tuple[int, ...], type_name: str
) -> torch.Tensor:
    expected = 1
    for dim in shape:
        expected *= dim
    if values.numel() != expected:
        raise ValueError(f"{type_name} dequant size {values.numel()} != numel {expected}")
    return values.reshape(shape)


def _fp16_scale(packed: torch.Tensor, start: int) -> torch.Tensor:
    return (
        packed[:, start : start + 2]
        .contiguous()
        .view(torch.float16)
        .to(torch.float32)
        .squeeze(1)
    )


def _k4_scales_and_mins(scales: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    decoded_scales: list[torch.Tensor] = []
    decoded_mins: list[torch.Tensor] = []
    for index in range(8):
        if index < 4:
            decoded_scales.append((scales[:, index] & 63).to(torch.float32))
            decoded_mins.append((scales[:, index + 4] & 63).to(torch.float32))
        else:
            decoded_scales.append(
                ((scales[:, index + 4] & 0xF) | ((scales[:, index - 4] >> 6) << 4)).to(
                    torch.float32
                )
            )
            decoded_mins.append(
                ((scales[:, index + 4] >> 4) | ((scales[:, index] >> 6) << 4)).to(
                    torch.float32
                )
            )
    return decoded_scales, decoded_mins


def dequantize_gguf_packed(
    packed: torch.Tensor,
    shape: tuple[int, ...],
    type_name: str,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Materialize one packed GGUF tensor without moving it through the CPU.

    The implementation mirrors the standard GGML block layouts and is kept
    deliberately explicit for bring-up and parity tests. It is not a claim
    that byte-unpacking plus ordinary ``F.linear`` is the final performance
    implementation for Q6_K on SM120.
    """

    if type_name == "IQ2_XS":
        values = dequantize_iq2_xs_packed(packed, shape).to(packed.device)
        return values.to(dtype=dtype) if dtype is not None else values
    block_bytes = GGUF_BLOCK_BYTES.get(type_name)
    if block_bytes is None:
        raise ValueError(f"unsupported packed GGUF type {type_name!r}")
    if packed.dtype != torch.uint8 or packed.numel() % block_bytes:
        raise ValueError(f"packed {type_name} payload has invalid dtype or byte length")
    blocks = packed.reshape(-1, block_bytes)
    if type_name == "Q8_0":
        d = _fp16_scale(blocks, 0)
        qs = blocks[:, 2:].contiguous().view(torch.int8).to(torch.float32)
        values = (d[:, None] * qs).reshape(-1)
    elif type_name in {"Q4_K", "Q5_K"}:
        d = _fp16_scale(blocks, 0)
        dmin = _fp16_scale(blocks, 2)
        scales, mins = _k4_scales_and_mins(blocks[:, 4:16])
        ql_start = 16 if type_name == "Q4_K" else 48
        ql = blocks[:, ql_start:].to(torch.int16)
        qh = blocks[:, 16:48].to(torch.int16) if type_name == "Q5_K" else None
        parts: list[torch.Tensor] = []
        for chunk in range(4):
            q = ql[:, 32 * chunk : 32 * (chunk + 1)]
            q0 = q & 0xF
            q1 = q >> 4
            if qh is not None:
                q0 = q0 + ((qh & (1 << (2 * chunk)) != 0).to(torch.int16) * 16)
                q1 = q1 + ((qh & (2 << (2 * chunk)) != 0).to(torch.int16) * 16)
            p0 = d[:, None] * scales[2 * chunk][:, None] * q0.to(torch.float32)
            p1 = d[:, None] * scales[2 * chunk + 1][:, None] * q1.to(torch.float32)
            p0 = p0 - dmin[:, None] * mins[2 * chunk][:, None]
            p1 = p1 - dmin[:, None] * mins[2 * chunk + 1][:, None]
            parts.extend((p0, p1))
        values = torch.cat(parts, dim=1).reshape(-1)
    else:
        d = _fp16_scale(blocks, 208)
        ql = blocks[:, :128].to(torch.int16)
        qh = blocks[:, 128:192].to(torch.int16)
        scales = blocks[:, 192:208].contiguous().view(torch.int8).to(torch.float32)
        halves: list[torch.Tensor] = []
        index = torch.arange(32, device=packed.device)
        scale_index = index // 16
        for half in range(2):
            low = ql[:, 64 * half : 64 * (half + 1)]
            high = qh[:, 32 * half : 32 * (half + 1)]
            sc = scales[:, 8 * half : 8 * (half + 1)]
            q1 = (low[:, index] & 0xF) | ((high[:, index] & 3) << 4)
            q2 = (low[:, index + 32] & 0xF) | (((high[:, index] >> 2) & 3) << 4)
            q3 = (low[:, index] >> 4) | (((high[:, index] >> 4) & 3) << 4)
            q4 = (low[:, index + 32] >> 4) | (((high[:, index] >> 6) & 3) << 4)
            q1 = q1.to(torch.float32) - 32
            q2 = q2.to(torch.float32) - 32
            q3 = q3.to(torch.float32) - 32
            q4 = q4.to(torch.float32) - 32
            halves.append(
                torch.cat(
                    [
                        d[:, None]
                        * sc.gather(1, scale_index[None, :].expand(sc.shape[0], -1))
                        * q1,
                        d[:, None]
                        * sc.gather(1, (scale_index + 2)[None, :].expand(sc.shape[0], -1))
                        * q2,
                        d[:, None]
                        * sc.gather(1, (scale_index + 4)[None, :].expand(sc.shape[0], -1))
                        * q3,
                        d[:, None]
                        * sc.gather(1, (scale_index + 6)[None, :].expand(sc.shape[0], -1))
                        * q4,
                    ],
                    dim=1,
                )
            )
        values = torch.cat(halves, dim=1).reshape(-1)
    values = _reshape_dequantized(values, shape, type_name)
    return values.to(dtype=dtype) if dtype is not None else values


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
