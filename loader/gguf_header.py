"""Minimal GGUF v3 header reader for metadata-only checkpoint validation.

Parses only the header region (metadata KV pairs + tensor index), never the
tensor payloads, so it works on partially downloaded files and costs O(header)
bytes. Pure stdlib by design: this module is imported by offline tooling that
must run in the torch-free CI job.

GGML type ids and block geometries follow ggml.h (llama.cpp). Byte sizes for
block-quantized types are exact (bytes_per_block / block_size), not bpw
approximations.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GGUF_MAGIC = b"GGUF"

# Reasonable upper bounds that catch misaligned/corrupt parses early without
# rejecting real models (DSV4-Flash: 63 KV pairs, 1328 tensors).
_MAX_KV_COUNT = 4096
_MAX_TENSOR_COUNT = 1_000_000
_MAX_NAME_LENGTH = 4096
_MAX_STRING_VALUE_LENGTH = 1_000_000  # chat templates and similar KV values
_MAX_ARRAY_LENGTH = 4_000_000  # DSV4 tokenizer vocab/merges are ~129k entries

# ggml type id -> (name, elements per block, bytes per block).
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}

# GGUF metadata value types (v3 spec).
_KV_U8, _KV_I8, _KV_U16, _KV_I16, _KV_U32, _KV_I32 = 0, 1, 2, 3, 4, 5
_KV_F32, _KV_BOOL, _KV_STRING, _KV_ARRAY, _KV_U64, _KV_I64, _KV_F64 = (
    6,
    7,
    8,
    9,
    10,
    11,
    12,
)
_SCALAR_FORMATS = {
    _KV_U8: "<B",
    _KV_I8: "<b",
    _KV_U16: "<H",
    _KV_I16: "<h",
    _KV_U32: "<I",
    _KV_I32: "<i",
    _KV_F32: "<f",
    _KV_BOOL: "<?",
    _KV_U64: "<Q",
    _KV_I64: "<q",
    _KV_F64: "<d",
}
_ARRAY_ELEMENT_FORMATS = {
    _KV_U8: "<B",
    _KV_I8: "<b",
    _KV_U16: "<H",
    _KV_I16: "<h",
    _KV_U32: "<I",
    _KV_I32: "<i",
    _KV_F32: "<f",
    _KV_BOOL: "<?",
    _KV_U64: "<Q",
    _KV_I64: "<q",
    _KV_F64: "<d",
}


class GgufHeaderError(ValueError):
    """Raised for malformed, truncated, or unsupported GGUF headers."""


@dataclass(frozen=True)
class GgufTensorInfo:
    """One tensor index entry; dims are in GGML order (dims[0] fastest)."""

    name: str
    dims: tuple[int, ...]
    type_id: int
    offset: int  # relative to the start of the (aligned) data section

    @property
    def type_name(self) -> str:
        entry = GGML_TYPES.get(self.type_id)
        return entry[0] if entry is not None else f"UNKNOWN_{self.type_id}"

    @property
    def numel(self) -> int:
        result = 1
        for dim in self.dims:
            result *= dim
        return result

    @property
    def nbytes(self) -> int:
        entry = GGML_TYPES.get(self.type_id)
        if entry is None:
            raise GgufHeaderError(
                f"cannot size tensor {self.name!r}: unknown ggml type {self.type_id}"
            )
        _, block_size, block_bytes = entry
        if self.numel % block_size:
            raise GgufHeaderError(
                f"tensor {self.name!r}: numel {self.numel} not a multiple of "
                f"block size {block_size}"
            )
        return self.numel // block_size * block_bytes


@dataclass(frozen=True)
class GgufHeader:
    version: int
    kv: dict[str, Any]
    tensors: tuple[GgufTensorInfo, ...]
    header_end: int  # byte offset just past the tensor index
    data_start: int  # byte offset of the tensor data section (aligned)

    @property
    def alignment(self) -> int:
        value = self.kv.get("general.alignment", 32)
        if not isinstance(value, int) or value <= 0 or value & (value - 1):
            raise GgufHeaderError(f"invalid general.alignment: {value!r}")
        return value

    def tensor(self, name: str) -> GgufTensorInfo:
        for entry in self.tensors:
            if entry.name == name:
                return entry
        raise KeyError(name)

    def absolute_offset(self, tensor: GgufTensorInfo) -> int:
        return self.data_start + tensor.offset


class _Reader:
    """Bounds-checked little-endian cursor over an open binary file."""

    def __init__(self, source: Any, path: Path) -> None:
        self._source = source
        self._path = path

    def take(self, count: int) -> bytes:
        data = self._source.read(count)
        if len(data) != count:
            raise GgufHeaderError(
                f"truncated GGUF header in {self._path}: wanted {count} bytes at "
                f"offset {self._source.tell() - len(data)}, got {len(data)}"
            )
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self, max_length: int = _MAX_NAME_LENGTH) -> str:
        length = self.u64()
        if length > max_length:
            raise GgufHeaderError(f"implausible string length {length} in {self._path}")
        return self.take(length).decode("utf-8", "replace")

    def scalar(self, type_id: int) -> Any:
        fmt = _SCALAR_FORMATS.get(type_id)
        if fmt is None:
            raise GgufHeaderError(f"unsupported GGUF scalar value type {type_id}")
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def value(self) -> Any:
        type_id = self.u32()
        if type_id == _KV_STRING:
            return self.string(_MAX_STRING_VALUE_LENGTH)
        if type_id == _KV_ARRAY:
            elem_type = self.u32()
            count = self.u64()
            if count > _MAX_ARRAY_LENGTH:
                raise GgufHeaderError(f"implausible array length {count} in {self._path}")
            if elem_type == _KV_STRING:
                return [self.string(_MAX_STRING_VALUE_LENGTH) for _ in range(count)]
            fmt = _ARRAY_ELEMENT_FORMATS.get(elem_type)
            if fmt is None:
                raise GgufHeaderError(f"unsupported GGUF array element type {elem_type}")
            width = struct.calcsize(fmt)
            return list(struct.unpack(f"<{count}{fmt[-1]}", self.take(width * count)))
        return self.scalar(type_id)


def read_gguf_header(path: Path) -> GgufHeader:
    """Parse the GGUF metadata and tensor index without touching tensor data."""
    with path.open("rb") as source:
        reader = _Reader(source, path)
        if reader.take(4) != GGUF_MAGIC:
            raise GgufHeaderError(f"not a GGUF file: {path}")
        version = reader.u32()
        if version != 3:
            raise GgufHeaderError(f"unsupported GGUF version {version} (only v3): {path}")
        tensor_count = reader.u64()
        kv_count = reader.u64()
        if tensor_count > _MAX_TENSOR_COUNT or kv_count > _MAX_KV_COUNT:
            raise GgufHeaderError(
                f"implausible counts (tensors={tensor_count}, kv={kv_count}) in {path}"
            )
        kv: dict[str, Any] = {}
        for _ in range(kv_count):
            key = reader.string()
            kv[key] = reader.value()
        tensors: list[GgufTensorInfo] = []
        for _ in range(tensor_count):
            name = reader.string()
            n_dims = reader.u32()
            if not 0 < n_dims <= 8:
                raise GgufHeaderError(f"tensor {name!r}: implausible n_dims {n_dims}")
            dims = tuple(reader.u64() for _ in range(n_dims))
            type_id = reader.u32()
            offset = reader.u64()
            tensors.append(GgufTensorInfo(name=name, dims=dims, type_id=type_id, offset=offset))
        header_end = source.tell()

    alignment = 32
    alignment_value = kv.get("general.alignment")
    if alignment_value is not None:
        if (
            not isinstance(alignment_value, int)
            or alignment_value <= 0
            or alignment_value & (alignment_value - 1)
        ):
            raise GgufHeaderError(f"invalid general.alignment: {alignment_value!r}")
        alignment = alignment_value
    data_start = (header_end + alignment - 1) // alignment * alignment
    return GgufHeader(
        version=version,
        kv=kv,
        tensors=tuple(tensors),
        header_end=header_end,
        data_start=data_start,
    )
