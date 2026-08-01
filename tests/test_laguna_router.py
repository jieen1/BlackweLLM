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


class TestBuildAndLoadAgreeOnTarget:
    """The Makefile's build target and this module's accept-list must agree.

    B-5 (d9b635e) changed `make build-laguna-router` to stamp `sm_120f` while
    `TARGET_SM` still read `sm_120a`, and the check in `_load_library` is a
    real gate -- so every freshly built router raised, taking the whole Laguna
    model down with it. Nothing caught it: the existing tests here build their
    fixture manifests *from* `TARGET_SM`, so the two sides can never disagree
    inside this file no matter what the Makefile does.

    This test reads the Makefile instead, which is the only way the mismatch
    is observable from the test suite.
    """

    def _makefile_target_sm(self) -> str:
        import re
        from pathlib import Path

        makefile = Path(__file__).resolve().parent.parent / "Makefile"
        text = makefile.read_text(encoding="utf-8")
        # The manifest's target_sm is written by the python -c block in
        # build-laguna-router as a literal.
        match = re.search(r'"target_sm"\s*:\s*"(sm_\w+)"', text)
        assert match, "could not find target_sm literal in Makefile"
        return match.group(1)

    def test_makefile_target_is_accepted_by_the_loader(self):
        from runtime.laguna_router import ACCEPTED_TARGET_SM

        built = self._makefile_target_sm()
        assert built in ACCEPTED_TARGET_SM, (
            f"make build-laguna-router stamps target_sm={built!r}, which "
            f"_load_library rejects (accepts {ACCEPTED_TARGET_SM}). A freshly "
            f"built router would fail to load."
        )

    def test_target_sm_is_the_current_build_target(self):
        from runtime.laguna_router import TARGET_SM

        assert TARGET_SM == self._makefile_target_sm()
