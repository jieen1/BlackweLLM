"""Native scale-amortized IQ2_XS -> INT8 MMA grouped-MoE adapter (Phase 2B-0).

Wraps ``iq2_mma16_tc.cu``: folds the per-K16 IQ2 delta into the INT8 B
fragment (qB = sign*round(mag*delta/sB)), accumulates INT32 across a
K-scale group of 32 values, and applies one I2F+FFMA per group per
accumulator.  Same raw-pointer ABI as ``iq2_mma16.py`` (the exact oracle);
kept as a separate artifact so the two kernels' results are distinguishable.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ABI_VERSION = 1
TARGET_SM = "sm_120f"
ACCEPTED_TARGET_SM = ("sm_120f", "sm_120a")

_KERNEL_DIR = Path(__file__).resolve().parent
_GENERATED_DIR = _KERNEL_DIR / "_generated"
_LIBRARY_PATH = _GENERATED_DIR / "iq2_mma16_tc.so"
_MANIFEST_PATH = _GENERATED_DIR / "iq2_mma16_tc.manifest.json"


class IQ2MMA16TCError(RuntimeError):
    """The native IQ2 MMA16 TC artifact cannot satisfy its fixed contract."""


@dataclass(frozen=True)
class IQ2MMA16TCManifest:
    abi_version: int
    target_sm: str
    library_sha256: str

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> IQ2MMA16TCManifest:
        try:
            return cls(
                abi_version=int(value["abi_version"]),
                target_sm=str(value["target_sm"]),
                library_sha256=str(value["library_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IQ2MMA16TCError("invalid IQ2 MMA16 TC manifest") from error


def artifact_paths() -> tuple[Path, Path]:
    return _LIBRARY_PATH, _MANIFEST_PATH


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> IQ2MMA16TCManifest:
    if not path.is_file():
        raise IQ2MMA16TCError(f"IQ2 MMA16 TC manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IQ2MMA16TCError(f"cannot read IQ2 MMA16 TC manifest: {path}") from error
    return IQ2MMA16TCManifest.from_json(value)


class NativeIQ2MMA16TCLibrary:
    """Loaded raw-pointer ABI for the scale-amortized K-group kernel."""

    def __init__(self, library: ctypes.CDLL) -> None:
        self._library = library
        self._launch = library.iq2_mma16_tc_launch
        self._launch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self._launch.restype = None

    @classmethod
    def load(cls) -> NativeIQ2MMA16TCLibrary:
        if not _LIBRARY_PATH.is_file():
            raise IQ2MMA16TCError(f"IQ2 MMA16 TC library is missing: {_LIBRARY_PATH}")
        manifest = _load_manifest(_MANIFEST_PATH)
        if manifest.abi_version != ABI_VERSION:
            raise IQ2MMA16TCError("IQ2 MMA16 TC ABI mismatch")
        if manifest.target_sm not in ACCEPTED_TARGET_SM:
            raise IQ2MMA16TCError("IQ2 MMA16 TC target mismatch")
        if _sha256(_LIBRARY_PATH) != manifest.library_sha256:
            raise IQ2MMA16TCError("IQ2 MMA16 TC library SHA256 does not match its manifest")
        try:
            library = ctypes.CDLL(str(_LIBRARY_PATH))
        except OSError as error:
            raise IQ2MMA16TCError(f"cannot load IQ2 MMA16 TC library: {_LIBRARY_PATH}") from error
        return cls(library)

    def grouped_gate_up(
        self,
        xq,
        xs,
        packed_gate,
        packed_up,
        eids,
        grid,
        ksigns,
        *,
        rows: int,
        cols: int,
        stride: int,
        m_pad: int,
    ):
        import torch

        if xq.device.type != "cuda":
            raise IQ2MMA16TCError("iq2_mma16_tc requires CUDA tensors")
        E = int(eids.numel())
        for name, t in [("xq", xq), ("xs", xs), ("packed_gate", packed_gate),
                        ("packed_up", packed_up), ("eids", eids), ("grid", grid),
                        ("ksigns", ksigns)]:
            if not t.is_contiguous():
                raise IQ2MMA16TCError(f"{name} must be contiguous")
        out_gate = torch.empty((E, m_pad, rows), dtype=torch.float32, device=xq.device)
        out_up = torch.empty((E, m_pad, rows), dtype=torch.float32, device=xq.device)
        self._launch(
            ctypes.c_void_p(xq.data_ptr()),
            ctypes.c_void_p(xs.data_ptr()),
            ctypes.c_void_p(packed_gate.data_ptr()),
            ctypes.c_void_p(packed_up.data_ptr()),
            ctypes.c_void_p(eids.data_ptr()),
            ctypes.c_void_p(grid.data_ptr()),
            ctypes.c_void_p(ksigns.data_ptr()),
            ctypes.c_void_p(out_gate.data_ptr()),
            ctypes.c_void_p(out_up.data_ptr()),
            E, rows, cols, stride, m_pad,
        )
        return out_gate, out_up


def make_manifest(library_path: Path = _LIBRARY_PATH, source_path: Path | None = None) -> dict:
    payload = {
        "abi_version": ABI_VERSION,
        "target_sm": TARGET_SM,
        "library_sha256": _sha256(library_path),
    }
    if source_path is not None:
        payload["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return payload
