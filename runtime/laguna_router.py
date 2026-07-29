"""Fixed-contract native router adapter for Laguna's SM120 MoE path.

The module is deliberately safe to import in CPU-only tests.  Loading the
generated CUDA library is explicit and never happens at import time.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

ABI_VERSION = 1
EXPERTS = 256
TOP_K = 10
TARGET_SM = "sm_120a"
ROUTER_MODES = frozenset(("vllm", "native"))

_KERNEL_DIR = Path(__file__).with_name("kernels")
_GENERATED_DIR = _KERNEL_DIR / "_generated"
_LIBRARY_PATH = _GENERATED_DIR / "laguna_router_sm120.so"
_MANIFEST_PATH = _GENERATED_DIR / "laguna_router_sm120.manifest.json"


class LagunaRouterError(RuntimeError):
    """The fixed native router cannot satisfy its production contract."""


def resolve_router_mode(value: str | None) -> str:
    """Return the explicit temporary A/B router mode or reject typos early."""
    mode = value or "vllm"
    if mode not in ROUTER_MODES:
        choices = ", ".join(sorted(ROUTER_MODES))
        raise LagunaRouterError(f"invalid QSR_LAGUNA_ROUTER={mode!r}; expected one of: {choices}")
    return mode


def router_max_rows(prefill_chunk_tokens: int, num_slots: int, *, swa_qo_max: int) -> int:
    """Compute the fixed router arena capacity before CUDA Graph capture."""
    max_rows = max(prefill_chunk_tokens, swa_qo_max, num_slots)
    if max_rows <= 0:
        raise LagunaRouterError(f"Laguna router max rows must be positive, got {max_rows}")
    return max_rows


@dataclass(frozen=True)
class RouterManifest:
    abi_version: int
    target_sm: str
    library_sha256: str

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> RouterManifest:
        try:
            return cls(
                abi_version=int(value["abi_version"]),
                target_sm=str(value["target_sm"]),
                library_sha256=str(value["library_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise LagunaRouterError("invalid Laguna router manifest") from error


def artifact_paths() -> tuple[Path, Path]:
    """Return the single supported generated library and manifest paths."""
    return _LIBRARY_PATH, _MANIFEST_PATH


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> RouterManifest:
    if not path.is_file():
        raise LagunaRouterError(f"Laguna router manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LagunaRouterError(f"cannot read Laguna router manifest: {path}") from error
    return RouterManifest.from_json(value)


def _require_sm120_cuda() -> None:
    """Reject a matching artifact when the active CUDA device is not SM120."""
    import torch

    if not torch.cuda.is_available():
        raise LagunaRouterError("Laguna router requires an available SM120 CUDA device")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise LagunaRouterError(f"Laguna router requires CUDA capability (12, 0), got {capability}")


class LagunaRouterLibrary:
    """Loaded C ABI library with strict contract validation and no fallback."""

    def __init__(self, library: ctypes.CDLL) -> None:
        self._library = library
        self._launch = library.qsr_laguna_router_f32
        self._launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]
        self._launch.restype = ctypes.c_int

    @classmethod
    def load(
        cls,
        *,
        library_path: Path = _LIBRARY_PATH,
        manifest_path: Path = _MANIFEST_PATH,
    ) -> LagunaRouterLibrary:
        if not library_path.is_file():
            raise LagunaRouterError(f"Laguna router library is missing: {library_path}")
        manifest = _load_manifest(manifest_path)
        if manifest.abi_version != ABI_VERSION:
            raise LagunaRouterError(
                f"Laguna router ABI mismatch: expected {ABI_VERSION}, got {manifest.abi_version}"
            )
        if manifest.target_sm != TARGET_SM:
            raise LagunaRouterError(
                f"Laguna router target mismatch: expected {TARGET_SM}, got {manifest.target_sm}"
            )
        if _sha256(library_path) != manifest.library_sha256:
            raise LagunaRouterError("Laguna router library SHA256 does not match its manifest")
        _require_sm120_cuda()
        try:
            library = ctypes.CDLL(str(library_path))
        except OSError as error:
            raise LagunaRouterError(f"cannot load Laguna router library: {library_path}") from error
        abi = library.qsr_laguna_router_abi_version
        abi.argtypes = []
        abi.restype = ctypes.c_int
        if abi() != ABI_VERSION:
            raise LagunaRouterError("Laguna router library exports an incompatible ABI")
        return cls(library)

    def launch(
        self,
        logits: torch.Tensor,
        correction_bias: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write fixed-contract routing output into caller-owned CUDA tensors."""
        import torch

        rows = _validate_tensors(
            logits,
            correction_bias,
            topk_weights,
            topk_ids,
            torch_module=torch,
        )
        if rows == 0:
            return topk_weights[:0], topk_ids[:0]
        stream = torch.cuda.current_stream(logits.device).cuda_stream
        status = self._launch(
            ctypes.c_void_p(logits.data_ptr()),
            ctypes.c_void_p(correction_bias.data_ptr()),
            ctypes.c_void_p(topk_weights.data_ptr()),
            ctypes.c_void_p(topk_ids.data_ptr()),
            ctypes.c_int32(rows),
            ctypes.c_void_p(stream),
        )
        if status != 0:
            raise LagunaRouterError(f"Laguna router launch failed with status {status}")
        return topk_weights[:rows], topk_ids[:rows]


class LagunaRouterArena:
    """Address-stable caller-owned router outputs for one Laguna engine thread."""

    def __init__(self, max_rows: int, device: Any) -> None:
        if max_rows <= 0:
            raise LagunaRouterError(f"Laguna router max rows must be positive, got {max_rows}")
        import torch

        self.max_rows = max_rows
        self.weights = torch.empty((max_rows, TOP_K), dtype=torch.float32, device=device)
        self.ids = torch.empty((max_rows, TOP_K), dtype=torch.int32, device=device)


def _validate_tensors(
    logits: Any,
    correction_bias: Any,
    topk_weights: Any,
    topk_ids: Any,
    *,
    torch_module: Any,
) -> int:
    tensors = (logits, correction_bias, topk_weights, topk_ids)
    if any(not tensor.is_cuda for tensor in tensors):
        raise LagunaRouterError("Laguna router requires CUDA tensors")
    if logits.dtype != torch_module.float32 or correction_bias.dtype != torch_module.float32:
        raise LagunaRouterError("Laguna router requires FP32 logits and correction bias")
    if topk_weights.dtype != torch_module.float32 or topk_ids.dtype != torch_module.int32:
        raise LagunaRouterError("Laguna router requires FP32 weights and int32 ids")
    if logits.ndim != 2 or logits.shape[1] != EXPERTS:
        raise LagunaRouterError(f"Laguna router logits must have shape [M, {EXPERTS}]")
    if correction_bias.shape != (EXPERTS,):
        raise LagunaRouterError(f"Laguna router correction bias must have shape [{EXPERTS}]")
    rows = logits.shape[0]
    if topk_weights.ndim != 2 or topk_ids.ndim != 2:
        raise LagunaRouterError("Laguna router outputs must be rank-2")
    if topk_weights.shape[0] < rows or topk_ids.shape[0] < rows:
        raise LagunaRouterError("Laguna router output arena is smaller than the input batch")
    if topk_weights.shape[1] != TOP_K or topk_ids.shape[1] != TOP_K:
        raise LagunaRouterError(f"Laguna router outputs must have {TOP_K} columns")
    devices = {str(tensor.device) for tensor in tensors}
    if len(devices) != 1:
        raise LagunaRouterError("Laguna router tensors must share one CUDA device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise LagunaRouterError("Laguna router tensors must be contiguous")
    return rows
