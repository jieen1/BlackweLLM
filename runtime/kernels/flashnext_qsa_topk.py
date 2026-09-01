"""Optional native SM120 top-k selection for Flash-Next QSA.

The Python fallback remains the correctness baseline when the generated
artifact is not present.  Serving images that build the runtime artifact use
the standalone radix selector to avoid the large ``torch.topk`` scan over the
unused tail of a fixed 256K pooled-key cache.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

# ABI 2 returns int64 indices directly.  The previous int32 output required a
# separate device cast on every QSA layer/round, which erased the selector's
# gain once the result entered the long-index gather path.
ABI_VERSION = 2
TARGET_SM = "sm_120f"
ACCEPTED_TARGET_SM = ("sm_120f", "sm_120a")
TOPK = 512

_GENERATED_DIR = Path(__file__).with_name("_generated")
_LIBRARY_PATH = _GENERATED_DIR / "flashnext_qsa_topk_sm120.so"
_MANIFEST_PATH = _GENERATED_DIR / "flashnext_qsa_topk_sm120.manifest.json"


class FlashNextQsaTopKError(RuntimeError):
    """The optional native QSA top-k artifact is invalid or unusable."""


@dataclass(frozen=True)
class _Manifest:
    abi_version: int
    target_sm: str
    library_sha256: str

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> _Manifest:
        try:
            return cls(
                abi_version=int(value["abi_version"]),
                target_sm=str(value["target_sm"]),
                library_sha256=str(value["library_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise FlashNextQsaTopKError("invalid Flash-Next QSA top-k manifest") from error


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest() -> _Manifest:
    try:
        value = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FlashNextQsaTopKError(
            f"cannot read Flash-Next QSA top-k manifest: {_MANIFEST_PATH}"
        ) from error
    manifest = _Manifest.from_json(value)
    if manifest.abi_version != ABI_VERSION:
        raise FlashNextQsaTopKError(
            f"Flash-Next QSA top-k ABI mismatch: expected {ABI_VERSION}, got {manifest.abi_version}"
        )
    if manifest.target_sm not in ACCEPTED_TARGET_SM:
        raise FlashNextQsaTopKError(
            "Flash-Next QSA top-k target mismatch: expected one of "
            f"{', '.join(ACCEPTED_TARGET_SM)}, got {manifest.target_sm}"
        )
    if _sha256(_LIBRARY_PATH) != manifest.library_sha256:
        raise FlashNextQsaTopKError(
            "Flash-Next QSA top-k library SHA256 does not match its manifest"
        )
    return manifest


def _require_sm120_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise FlashNextQsaTopKError("Flash-Next QSA top-k requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise FlashNextQsaTopKError(
            f"Flash-Next QSA top-k requires CUDA capability (12, 0), got {capability}"
        )


class NativeFlashNextQsaTopK:
    """Raw-pointer adapter for the fixed-width 512-way radix selector."""

    def __init__(self, library: ctypes.CDLL) -> None:
        self._library = library
        self._select = library.qsr_flashnext_qsa_topk_sm120
        self._select.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        self._select.restype = ctypes.c_int

    @classmethod
    def load(cls) -> NativeFlashNextQsaTopK:
        if not _LIBRARY_PATH.is_file() or not _MANIFEST_PATH.is_file():
            raise FlashNextQsaTopKError(
                f"Flash-Next QSA top-k artifact is missing: {_LIBRARY_PATH}"
            )
        _load_manifest()
        _require_sm120_cuda()
        try:
            library = ctypes.CDLL(str(_LIBRARY_PATH))
        except OSError as error:
            raise FlashNextQsaTopKError(
                f"cannot load Flash-Next QSA top-k library: {_LIBRARY_PATH}"
            ) from error
        abi = library.qsr_flashnext_qsa_topk_abi_version
        abi.argtypes = []
        abi.restype = ctypes.c_int
        if abi() != ABI_VERSION:
            raise FlashNextQsaTopKError("Flash-Next QSA top-k library ABI is incompatible")
        return cls(library)

    def select(self, scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        import torch

        if (
            scores.device.type != "cuda"
            or scores.dtype != torch.float32
            or scores.ndim != 2
            or not scores.is_contiguous()
            or lengths.device != scores.device
            or lengths.dtype != torch.int64
            or lengths.ndim != 1
            or lengths.shape[0] != scores.shape[0]
            or not lengths.is_contiguous()
        ):
            raise FlashNextQsaTopKError(
                "native Flash-Next QSA top-k expects contiguous CUDA F32 scores and "
                "contiguous CUDA int64 row lengths"
            )
        # Downstream QSA gather/index_copy operations consume torch.long.  Keep
        # that dtype in the native output so graph replay does not need an
        # extra int32->int64 conversion kernel for every layer.
        output = torch.empty(scores.shape[0], TOPK, dtype=torch.int64, device=scores.device)
        status = self._select(
            ctypes.c_void_p(scores.data_ptr()),
            ctypes.c_void_p(lengths.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            scores.shape[0],
            scores.stride(0),
            ctypes.c_void_p(torch.cuda.current_stream(scores.device).cuda_stream),
        )
        if status != 0:
            raise FlashNextQsaTopKError(
                f"native Flash-Next QSA top-k launch failed with status {status}"
            )
        return output


@lru_cache(maxsize=1)
def load_native_flashnext_qsa_topk() -> NativeFlashNextQsaTopK | None:
    """Load the optional artifact once; return ``None`` for source-only installs."""

    try:
        return NativeFlashNextQsaTopK.load()
    except FlashNextQsaTopKError:
        return None


def artifact_paths() -> tuple[Path, Path]:
    """Return the generated library and provenance manifest paths."""

    return _LIBRARY_PATH, _MANIFEST_PATH


__all__ = [
    "ABI_VERSION",
    "TOPK",
    "FlashNextQsaTopKError",
    "NativeFlashNextQsaTopK",
    "artifact_paths",
    "load_native_flashnext_qsa_topk",
]
