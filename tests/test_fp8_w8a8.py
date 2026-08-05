from __future__ import annotations

import hashlib
import json

import pytest

from runtime.fp8_w8a8 import (
    ABI_VERSION,
    TARGET_SM,
    FP8W8A8Error,
    NativeFP8W8A8Library,
    _require_sm120_cuda,
    _validate_geometry,
    artifact_paths,
)


def test_artifact_paths_are_single_canonical_generated_locations() -> None:
    library, manifest = artifact_paths()

    assert library.name == "fp8_w8a8_sm120.so"
    assert manifest.name == "fp8_w8a8_sm120.manifest.json"
    assert library.parent == manifest.parent
    assert library.parent.name == "_generated"


def test_loader_fails_fast_when_library_is_missing(tmp_path) -> None:
    with pytest.raises(FP8W8A8Error, match="library is missing"):
        NativeFP8W8A8Library.load(
            library_path=tmp_path / "missing.so", manifest_path=tmp_path / "missing.json"
        )


def test_loader_rejects_manifest_abi_before_loading_library(tmp_path) -> None:
    library = tmp_path / "native.so"
    library.write_bytes(b"not a shared library")
    manifest = tmp_path / "native.json"
    manifest.write_text(
        json.dumps(
            {
                "abi_version": ABI_VERSION + 1,
                "target_sm": TARGET_SM,
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FP8W8A8Error, match="ABI mismatch"):
        NativeFP8W8A8Library.load(library_path=library, manifest_path=manifest)


def test_geometry_rejects_non_tensorcore_aligned_dimensions() -> None:
    with pytest.raises(FP8W8A8Error, match="multiples of 16"):
        _validate_geometry(m=1, n=15, k=4096)


def test_sm120_guard_rejects_cpu_only_torch(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch())
    with pytest.raises(FP8W8A8Error, match="available SM120"):
        _require_sm120_cuda()


def test_torch_scaled_mm_quantizer_uses_fused_raw_fp8_contract_when_available(
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch", reason="torch-free CI job")
    import runtime.model.compressed_tensors_linear as linear

    class FakeQuantizer:
        calls = 0

        def quantize_per_token(self, x, out_fp8, scale):
            self.calls += 1
            out_fp8.copy_(x.to(torch.float8_e4m3fn))
            scale.fill_(0.25)

    quantizer = FakeQuantizer()
    monkeypatch.setattr(linear, "_native_w8a8_quantizer_for_cuda", lambda: quantizer)
    monkeypatch.setenv(linear.QSR_NATIVE_W8A8_QUANT_ENV, "1")
    x = torch.arange(32, dtype=torch.bfloat16).reshape(2, 16)

    x_fp8, scale = linear._quantize_fp8_activation_for_torch_scaled_mm(x, 16)

    assert quantizer.calls == 1
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert x_fp8.shape == x.shape
    assert scale.dtype == torch.float32
    assert torch.equal(scale, torch.full((2, 1), 0.25, dtype=torch.float32))
