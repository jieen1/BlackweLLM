from __future__ import annotations

import hashlib
import json

import pytest

from runtime.laguna_router import (
    ABI_VERSION,
    TARGET_SM,
    LagunaRouterArena,
    LagunaRouterError,
    LagunaRouterLibrary,
    _require_sm120_cuda,
    _validate_tensors,
    artifact_paths,
    router_max_rows,
)


def test_artifact_paths_are_single_canonical_generated_locations() -> None:
    library, manifest = artifact_paths()

    assert library.name == "laguna_router_sm120.so"
    assert manifest.name == "laguna_router_sm120.manifest.json"
    assert library.parent == manifest.parent
    assert library.parent.name == "_generated"


def test_loader_fails_fast_when_library_is_missing(tmp_path) -> None:
    with pytest.raises(LagunaRouterError, match="library is missing"):
        LagunaRouterLibrary.load(
            library_path=tmp_path / "missing.so", manifest_path=tmp_path / "missing.json"
        )


def test_loader_rejects_manifest_abi_before_loading_library(tmp_path) -> None:
    library = tmp_path / "router.so"
    library.write_bytes(b"not a shared library")
    manifest = tmp_path / "router.json"
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

    with pytest.raises(LagunaRouterError, match="ABI mismatch"):
        LagunaRouterLibrary.load(library_path=library, manifest_path=manifest)


def test_loader_rejects_library_hash_mismatch_before_loading_library(tmp_path) -> None:
    library = tmp_path / "router.so"
    library.write_bytes(b"not a shared library")
    manifest = tmp_path / "router.json"
    manifest.write_text(
        json.dumps(
            {
                "abi_version": ABI_VERSION,
                "target_sm": TARGET_SM,
                "library_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(LagunaRouterError, match="SHA256"):
        LagunaRouterLibrary.load(library_path=library, manifest_path=manifest)


def test_router_arena_capacity_is_fixed_from_prefill_verify_and_slots() -> None:
    assert router_max_rows(8192, 4, swa_qo_max=16) == 8192
    assert router_max_rows(8, 32, swa_qo_max=16) == 32


def test_router_arena_constructor_rejects_nonpositive_capacity_without_cuda() -> None:
    with pytest.raises(LagunaRouterError, match="max rows"):
        LagunaRouterArena(0, "cuda")


def test_tensor_validation_accepts_bf16_logits_without_changing_output_contract() -> None:
    class FakeTorch:
        float32 = object()
        bfloat16 = object()
        int32 = object()

    class FakeTensor:
        def __init__(self, dtype, shape) -> None:
            self.dtype = dtype
            self.shape = shape
            self.ndim = len(shape)
            self.is_cuda = True
            self.device = "cuda:0"

        def is_contiguous(self) -> bool:
            return True

    torch = FakeTorch()
    logits = FakeTensor(torch.bfloat16, (4, 256))
    bias = FakeTensor(torch.float32, (256,))
    weights = FakeTensor(torch.float32, (8, 10))
    ids = FakeTensor(torch.int32, (8, 10))

    assert _validate_tensors(logits, bias, weights, ids, torch_module=torch) == 4


def test_sm120_guard_rejects_cpu_only_torch(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch())
    with pytest.raises(LagunaRouterError, match="available SM120"):
        _require_sm120_cuda()
