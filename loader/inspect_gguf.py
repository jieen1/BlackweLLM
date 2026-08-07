"""Print a validated GGUF metadata summary without reading tensor payloads.

Works on partially downloaded files (only the header region is read), which
makes it the progress/completeness probe for large GGUF downloads.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from loader.gguf_header import GgufHeader, read_gguf_header


def summarize(header: GgufHeader, file_size: int | None = None) -> dict:
    bytes_by_type: dict[str, int] = collections.defaultdict(int)
    count_by_type: dict[str, int] = collections.defaultdict(int)
    for tensor in header.tensors:
        bytes_by_type[tensor.type_name] += tensor.nbytes
        count_by_type[tensor.type_name] += 1
    total_payload = sum(bytes_by_type.values())
    summary = {
        "version": header.version,
        "architecture": header.kv.get("general.architecture"),
        "name": header.kv.get("general.name"),
        "tensor_count": len(header.tensors),
        "kv_count": len(header.kv),
        "header_end": header.header_end,
        "data_start": header.data_start,
        "total_payload_bytes": total_payload,
        "bytes_by_type": dict(bytes_by_type),
        "count_by_type": dict(count_by_type),
    }
    if file_size is not None:
        summary["file_size"] = file_size
        summary["complete"] = file_size >= header.data_start + total_payload
        summary["download_fraction"] = round(
            min(file_size, header.data_start + total_payload) / (header.data_start + total_payload),
            4,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf_file", type=Path)
    parser.add_argument("--tensors", action="store_true", help="list every tensor index entry")
    parser.add_argument("--kv", action="store_true", help="dump all metadata KV pairs")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    arguments = parser.parse_args()

    header = read_gguf_header(arguments.gguf_file)
    file_size = arguments.gguf_file.stat().st_size if arguments.gguf_file.exists() else None

    if arguments.json:
        payload: dict = {"summary": summarize(header, file_size)}
        if arguments.kv:
            payload["kv"] = header.kv
        if arguments.tensors:
            payload["tensors"] = [
                {
                    "name": t.name,
                    "dims": list(t.dims),
                    "type": t.type_name,
                    "offset": t.offset,
                    "nbytes": t.nbytes,
                }
                for t in header.tensors
            ]
        print(json.dumps(payload, indent=1, ensure_ascii=False))
        return

    summary = summarize(header, file_size)
    gib = 2**30
    print(f"file        : {arguments.gguf_file}")
    print(
        f"gguf version: {summary['version']}  arch: {summary['architecture']}  "
        f"name: {summary['name']}"
    )
    print(f"tensors     : {summary['tensor_count']}  kv pairs: {summary['kv_count']}")
    print(f"data_start  : {summary['data_start']}  header_end: {summary['header_end']}")
    print(f"payload     : {summary['total_payload_bytes'] / gib:.2f} GiB")
    if file_size is not None:
        state = (
            "COMPLETE"
            if summary["complete"]
            else f"{summary['download_fraction'] * 100:.1f}% downloaded"
        )
        print(f"on disk     : {file_size / gib:.2f} GiB  [{state}]")
    print("by type:")
    for type_name in sorted(
        summary["bytes_by_type"], key=summary["bytes_by_type"].__getitem__, reverse=True
    ):
        print(
            f"  {type_name:>10}: {summary['count_by_type'][type_name]:5d} tensors, "
            f"{summary['bytes_by_type'][type_name] / gib:8.2f} GiB"
        )
    if arguments.kv:
        print("metadata:")
        for key, value in header.kv.items():
            text = str(value)
            if len(text) > 120:
                text = text[:120] + "...<truncated>"
            print(f"  {key} = {text}")
    if arguments.tensors:
        print("tensors:")
        for tensor in header.tensors:
            print(
                f"  {tensor.name}  dims={tensor.dims}  {tensor.type_name}  "
                f"off={tensor.offset}  {tensor.nbytes} B"
            )


if __name__ == "__main__":
    main()
