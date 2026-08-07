"""Torch-free tests for loader/gguf_header.py + loader/inspect_gguf.py.

Builds a synthetic GGUF v3 file (header only) so the CI-sim job can run it.
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from loader.gguf_header import (
    GgufHeaderError,
    read_gguf_header,
)
from loader.inspect_gguf import summarize


def _str(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _kv(key: str, type_id: int, payload: bytes) -> bytes:
    return _str(key) + struct.pack("<I", type_id) + payload


def _array(elem_type: int, fmt: str, values: list) -> bytes:
    return struct.pack("<IIQ", elem_type, 9, len(values))[:8] + b""  # placeholder, unused


def build_synthetic_gguf(path: Path) -> None:
    """Write a minimal GGUF v3 header: 6 KV pairs of varied types, 3 tensors."""
    out = bytearray()
    out += b"GGUF" + struct.pack("<I", 3)
    out += struct.pack("<QQ", 3, 6)  # tensor_count, kv_count

    out += _kv("general.architecture", 8, _str("deepseek4"))
    out += _kv("deepseek4.block_count", 4, struct.pack("<I", 43))
    out += _kv("deepseek4.rope.freq_base", 6, struct.pack("<f", 10000.0))
    out += _kv("tokenizer.ggml.add_bos_token", 7, struct.pack("<?", False))
    out += _kv("general.alignment", 4, struct.pack("<I", 64))
    # i32 array (like compress_ratios)
    ratios = [0, 0, 4, 128]
    out += _kv(
        "deepseek4.attention.compress_ratios",
        9,
        struct.pack("<I", 5)
        + struct.pack("<Q", len(ratios))
        + struct.pack(f"<{len(ratios)}i", *ratios),
    )

    def tensor_entry(name: str, dims: tuple[int, ...], type_id: int, offset: int) -> bytes:
        data = _str(name) + struct.pack("<I", len(dims))
        for dim in dims:
            data += struct.pack("<Q", dim)
        return data + struct.pack("<IQ", type_id, offset)

    # F32 [4,4] (64B), Q8_0 [64,128] (8704B), IQ2_XS [256,512] (37888B); dims in GGML order
    out += tensor_entry("blk.0.ffn_gate_inp.weight", (256, 4096), 30, 0)  # BF16: 2MiB
    out += tensor_entry("blk.0.attn_sinks.weight", (64,), 0, 2 * 1024 * 1024)  # F32: 256B
    out += tensor_entry("blk.0.ffn_gate_exps.weight", (4096, 2048, 256), 17, 3 * 1024 * 1024)
    path.write_bytes(bytes(out))


@pytest.fixture()
def synthetic_gguf(tmp_path: Path) -> Path:
    target = tmp_path / "synthetic.gguf"
    build_synthetic_gguf(target)
    return target


def test_header_parse(synthetic_gguf: Path) -> None:
    header = read_gguf_header(synthetic_gguf)
    assert header.version == 3
    assert len(header.tensors) == 3
    assert header.kv["general.architecture"] == "deepseek4"
    assert header.kv["deepseek4.block_count"] == 43
    assert header.kv["deepseek4.attention.compress_ratios"] == [0, 0, 4, 128]
    assert header.kv["tokenizer.ggml.add_bos_token"] is False
    # alignment 64: header end must be rounded up to 64
    assert header.data_start % 64 == 0
    assert header.data_start >= header.header_end


def test_tensor_sizing(synthetic_gguf: Path) -> None:
    header = read_gguf_header(synthetic_gguf)
    bf16 = header.tensor("blk.0.ffn_gate_inp.weight")
    assert bf16.type_name == "BF16"
    assert bf16.numel == 256 * 4096
    assert bf16.nbytes == 256 * 4096 * 2
    sinks = header.tensor("blk.0.attn_sinks.weight")
    assert sinks.nbytes == 64 * 4
    experts = header.tensor("blk.0.ffn_gate_exps.weight")
    assert experts.type_name == "IQ2_XS"
    # 256-element blocks at 74 bytes each
    assert experts.nbytes == (256 * 2048 * 4096) // 256 * 74
    with pytest.raises(KeyError):
        header.tensor("does.not.exist")


def test_absolute_offset(synthetic_gguf: Path) -> None:
    header = read_gguf_header(synthetic_gguf)
    tensor = header.tensor("blk.0.attn_sinks.weight")
    assert header.absolute_offset(tensor) == header.data_start + 2 * 1024 * 1024


def test_summarize_reports_incomplete_file(synthetic_gguf: Path) -> None:
    header = read_gguf_header(synthetic_gguf)
    summary = summarize(header, file_size=synthetic_gguf.stat().st_size)
    # synthetic file contains only the header, so payload is missing
    assert summary["complete"] is False
    assert summary["total_payload_bytes"] > 0
    full_size = header.data_start + summary["total_payload_bytes"]
    assert summarize(header, file_size=full_size)["complete"] is True


def test_truncated_file_raises(tmp_path: Path) -> None:
    target = tmp_path / "truncated.gguf"
    target.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 1, 1) + b"\x05hello")
    with pytest.raises(GgufHeaderError):
        read_gguf_header(target)


def test_bad_magic_raises(tmp_path: Path) -> None:
    target = tmp_path / "notgguf.gguf"
    target.write_bytes(b"GGML" + b"\x00" * 32)
    with pytest.raises(GgufHeaderError):
        read_gguf_header(target)


def test_unknown_type_can_still_index_but_not_size(tmp_path: Path) -> None:
    target = tmp_path / "unknown.gguf"
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 1, 0)
    out += _str("weird.weight") + struct.pack("<I", 1) + struct.pack("<Q", 256)
    out += struct.pack("<IQ", 999, 0)  # unknown type id 999
    target.write_bytes(out)
    header = read_gguf_header(target)
    tensor = header.tensor("weird.weight")
    assert tensor.type_name == "UNKNOWN_999"
    with pytest.raises(GgufHeaderError):
        tensor.nbytes


def test_cli_json_output(synthetic_gguf: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "loader.inspect_gguf", str(synthetic_gguf), "--json", "--kv"],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    payload = json.loads(result.stdout)
    assert payload["summary"]["architecture"] == "deepseek4"
    assert payload["summary"]["tensor_count"] == 3
    assert payload["kv"]["deepseek4.block_count"] == 43
