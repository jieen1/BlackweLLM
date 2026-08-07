"""Bit-exact cross-check of loader/gguf_dequant.py against llama.cpp.

Compiles tools/gguf_dequant_golden.c against a local llama.cpp build (env
LLAMA_CPP_DIR, default /home/bot/project/llama.cpp) and compares its fp32
outputs with our pure-Python dequantizers on seeded-random blocks. Skips
cleanly where the checkout, compiler, or libraries are absent (e.g. CI), so
the torch-free job still collects it.
"""

from __future__ import annotations

import os
import random
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from loader.gguf_dequant import (
    dequantize_iq2_xs_row,
    dequantize_q8_0_row,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _llama_cpp_dir() -> Path | None:
    candidate = Path(os.environ.get("LLAMA_CPP_DIR", "/home/bot/project/llama.cpp"))
    if (candidate / "ggml" / "src" / "ggml-common.h").exists():
        return candidate
    return None


def _find_ggml_lib(build_dir: Path) -> Path | None:
    for candidate in sorted(build_dir.rglob("libggml-base.so")):
        return candidate.parent
    return None


@pytest.fixture(scope="module")
def harness(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("gcc") is None:
        pytest.skip("gcc not available")
    llama_dir = _llama_cpp_dir()
    if llama_dir is None:
        pytest.skip("llama.cpp checkout not found (set LLAMA_CPP_DIR)")
    build_dir = llama_dir / "build-sm120"
    lib_dir = _find_ggml_lib(build_dir)
    if lib_dir is None:
        pytest.skip("llama.cpp build-sm120 with libggml-base.so not found")
    binary = tmp_path_factory.mktemp("golden") / "gguf_dequant_golden"
    command = [
        "gcc",
        "-O2",
        "-DGGML_COMMON_DECL_C",
        f"-I{llama_dir / 'ggml' / 'src'}",
        str(REPO_ROOT / "tools" / "gguf_dequant_golden.c"),
        f"-L{lib_dir}",
        "-lggml-base",
        f"-Wl,-rpath,{lib_dir}",
        "-o",
        str(binary),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"harness build failed: {result.stderr[-500:]}")
    return binary


def _run_harness(harness: Path, kind: str, blocks: bytes, tmp_path: Path) -> list[float]:
    blocks_path = tmp_path / f"{kind}.bin"
    output_path = tmp_path / f"{kind}.f32"
    blocks_path.write_bytes(blocks)
    result = subprocess.run(
        [str(harness), kind, str(blocks_path), str(output_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    raw = output_path.read_bytes()
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def _float_bits(values: list[float]) -> list[int]:
    return [struct.unpack("<I", struct.pack("<f", v))[0] for v in values]


def _finite_fp16_bits(rng: random.Random) -> int:
    while True:
        bits = rng.getrandbits(16)
        if (bits >> 10) & 0x1F != 0x1F:  # exclude inf/nan payloads
            return bits


def test_q8_0_bit_exact(harness: Path, tmp_path: Path) -> None:
    rng = random.Random(20260807)
    blocks = bytearray()
    for _ in range(16):
        d = struct.pack("<H", _finite_fp16_bits(rng))
        qs = struct.pack("<32b", *(rng.randrange(-127, 128) for _ in range(32)))
        blocks += d + qs
    golden = _run_harness(harness, "q8_0", bytes(blocks), tmp_path)
    ours = dequantize_q8_0_row(bytes(blocks))
    assert len(golden) == len(ours) == 16 * 32
    assert _float_bits(golden) == _float_bits(ours)


def test_iq2_xs_bit_exact(harness: Path, tmp_path: Path) -> None:
    rng = random.Random(20260808)
    blocks = bytearray()
    for _ in range(16):
        d = struct.pack("<H", _finite_fp16_bits(rng))
        codes = struct.pack("<32H", *(rng.getrandbits(16) for _ in range(32)))
        scales = bytes(rng.getrandbits(8) for _ in range(8))
        blocks += d + codes + scales
    golden = _run_harness(harness, "iq2_xs", bytes(blocks), tmp_path)
    ours = dequantize_iq2_xs_row(bytes(blocks))
    assert len(golden) == len(ours) == 16 * 256
    assert _float_bits(golden) == _float_bits(ours)
