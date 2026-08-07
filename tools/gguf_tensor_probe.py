"""Read one tensor's payload out of a (possibly partial) GGUF and dequantize it.

Offline loader-probe: exercises the exact mmap + offset + dequant path the
Phase-2 streaming loader will use, on real model tensors, without loading the
rest of the file. Prints dequantized-value statistics for sanity checks.

Usage: python tools/gguf_tensor_probe.py <file.gguf> <tensor_name> [--blocks N]
"""

from __future__ import annotations

import argparse
import math
import mmap
import os
from pathlib import Path

from loader.gguf_dequant import (
    IQ2_XS_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES,
    dequantize_iq2_xs_row,
    dequantize_q8_0_row,
)
from loader.gguf_header import read_gguf_header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf_file", type=Path)
    parser.add_argument("tensor_name")
    parser.add_argument(
        "--blocks", type=int, default=64, help="how many leading blocks to dequantize"
    )
    arguments = parser.parse_args()

    header = read_gguf_header(arguments.gguf_file)
    tensor = header.tensor(arguments.tensor_name)
    total_bytes = tensor.nbytes
    want = min(total_bytes, arguments.blocks * max(Q8_0_BLOCK_BYTES, IQ2_XS_BLOCK_BYTES))
    start = header.absolute_offset(tensor)

    size = os.path.getsize(arguments.gguf_file)
    if start + want > size:
        raise SystemExit(
            f"tensor payload not downloaded yet: need bytes [{start}, {start + want}), "
            f"file is {size} bytes"
        )

    with (
        arguments.gguf_file.open("rb") as source,
        mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as mm,
    ):
        raw = bytes(mm[start : start + want])

    print(f"tensor  : {tensor.name}")
    print(f"dims    : {tensor.dims} (GGML order)  type: {tensor.type_name}")
    print(f"payload : {total_bytes} bytes at file offset {start}; probing {want} bytes")

    if tensor.type_name == "Q8_0":
        n_blocks = want // Q8_0_BLOCK_BYTES
        values = dequantize_q8_0_row(raw[: n_blocks * Q8_0_BLOCK_BYTES])
        print(f"probed  : {n_blocks} blocks -> {len(values)} values")
    elif tensor.type_name == "IQ2_XS":
        n_blocks = want // IQ2_XS_BLOCK_BYTES
        values = dequantize_iq2_xs_row(raw[: n_blocks * IQ2_XS_BLOCK_BYTES])
        print(f"probed  : {n_blocks} blocks -> {len(values)} values")
    elif tensor.type_name in ("F32", "BF16"):
        import struct

        if tensor.type_name == "F32":
            values = list(struct.unpack(f"<{want // 4}f", raw[: want // 4 * 4]))
        else:
            halves = struct.unpack(f"<{want // 2}H", raw[: want // 2 * 2])
            values = [struct.unpack("<f", struct.pack("<I", h << 16))[0] for h in halves]
        print(f"probed  : {len(values)} values")
    else:
        raise SystemExit(f"unsupported probe type {tensor.type_name}")

    finite = [v for v in values if math.isfinite(v)]
    nonzero = [v for v in finite if v != 0.0]
    print(f"finite  : {len(finite)}  nonzero: {len(nonzero)}")
    if finite:
        print(f"min={min(finite):.6g}  max={max(finite):.6g}  mean={sum(finite) / len(finite):.6g}")
    if nonzero:
        absvals = sorted(abs(v) for v in nonzero)
        print(
            f"abs median={absvals[len(absvals) // 2]:.6g}  "
            f"p99={absvals[int(len(absvals) * 0.99)]:.6g}"
        )
    print("sample  :", [f"{v:.4g}" for v in values[:8]])


if __name__ == "__main__":
    main()
