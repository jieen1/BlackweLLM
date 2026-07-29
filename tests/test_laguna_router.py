from __future__ import annotations

import hashlib
import json

import pytest

from runtime.laguna_router import (
    ABI_VERSION,
    TARGET_SM,
    LagunaRouterError,
    LagunaRouterLibrary,
    artifact_paths,
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
