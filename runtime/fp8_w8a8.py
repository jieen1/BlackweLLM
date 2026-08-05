"""Fixed-contract native E4M3 W8A8 GEMM adapter for Qwen3.6 dense layers.

The generated library deliberately has a raw CUDA-pointer ABI.  It owns no
PyTorch or vLLM headers, and this module imports ``torch`` only when a caller
explicitly loads or launches the CUDA artifact so CPU-only test collection is
safe.
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

# Version 2 adds the native dynamic FP8 activation-quantization entry point.
# Keep this in lockstep with the C ABI so a stale GEMM-only artifact fails
# during load instead of reaching a missing ctypes symbol at inference time.
ABI_VERSION = 2
TARGET_SM = "sm_120f"
ACCEPTED_TARGET_SM = ("sm_120f", "sm_120a")

_KERNEL_DIR = Path(__file__).with_name("kernels")
_GENERATED_DIR = _KERNEL_DIR / "_generated"
_LIBRARY_PATH = _GENERATED_DIR / "fp8_w8a8_sm120.so"
_MANIFEST_PATH = _GENERATED_DIR / "fp8_w8a8_sm120.manifest.json"


class FP8W8A8Error(RuntimeError):
    """The native FP8 W8A8 artifact cannot satisfy its fixed contract."""


@dataclass(frozen=True)
class FP8W8A8Manifest:
    abi_version: int
    target_sm: str
    library_sha256: str

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> FP8W8A8Manifest:
        try:
            return cls(
                abi_version=int(value["abi_version"]),
                target_sm=str(value["target_sm"]),
                library_sha256=str(value["library_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise FP8W8A8Error("invalid FP8 W8A8 manifest") from error


def artifact_paths() -> tuple[Path, Path]:
    """Return the single supported generated library and manifest paths."""
    return _LIBRARY_PATH, _MANIFEST_PATH


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> FP8W8A8Manifest:
    if not path.is_file():
        raise FP8W8A8Error(f"FP8 W8A8 manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FP8W8A8Error(f"cannot read FP8 W8A8 manifest: {path}") from error
    return FP8W8A8Manifest.from_json(value)


def _require_sm120_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise FP8W8A8Error("FP8 W8A8 requires an available SM120 CUDA device")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise FP8W8A8Error(f"FP8 W8A8 requires CUDA capability (12, 0), got {capability}")


class NativeFP8W8A8Library:
    """Loaded raw-pointer ABI with strict artifact and tensor validation."""

    def __init__(self, library: ctypes.CDLL) -> None:
        self._library = library
        self._quantize = library.qsr_fp8_w8a8_dynamic_per_token_quant_sm120
        self._workspace_size = library.qsr_fp8_w8a8_workspace_size_sm120
        self._launch = library.qsr_fp8_w8a8_scaled_mm_sm120
        self._quantize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._quantize.restype = ctypes.c_int
        self._workspace_size.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._workspace_size.restype = ctypes.c_int
        self._launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._launch.restype = ctypes.c_int

    @classmethod
    def load(
        cls,
        *,
        library_path: Path = _LIBRARY_PATH,
        manifest_path: Path = _MANIFEST_PATH,
    ) -> NativeFP8W8A8Library:
        if not library_path.is_file():
            raise FP8W8A8Error(f"FP8 W8A8 library is missing: {library_path}")
        manifest = _load_manifest(manifest_path)
        if manifest.abi_version != ABI_VERSION:
            raise FP8W8A8Error(
                f"FP8 W8A8 ABI mismatch: expected {ABI_VERSION}, got {manifest.abi_version}"
            )
        if manifest.target_sm not in ACCEPTED_TARGET_SM:
            raise FP8W8A8Error(
                "FP8 W8A8 target mismatch: expected one of "
                f"{', '.join(ACCEPTED_TARGET_SM)}, got {manifest.target_sm}"
            )
        if _sha256(library_path) != manifest.library_sha256:
            raise FP8W8A8Error("FP8 W8A8 library SHA256 does not match its manifest")
        _require_sm120_cuda()
        try:
            library = ctypes.CDLL(str(library_path))
        except OSError as error:
            raise FP8W8A8Error(f"cannot load FP8 W8A8 library: {library_path}") from error
        abi = library.qsr_fp8_w8a8_abi_version
        abi.argtypes = []
        abi.restype = ctypes.c_int
        if abi() != ABI_VERSION:
            raise FP8W8A8Error("FP8 W8A8 library exports an incompatible ABI")
        return cls(library)

    def quantize_per_token(
        self, x: torch.Tensor, out_fp8: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize contiguous BF16 ``[M, K]`` into E4M3 plus FP32 ``[M, 1]``."""
        import torch

        if (
            not x.is_cuda
            or not out_fp8.is_cuda
            or not scale.is_cuda
            or x.dtype != torch.bfloat16
            or out_fp8.dtype != torch.float8_e4m3fn
            or scale.dtype != torch.float32
            or x.ndim != 2
            or out_fp8.shape != x.shape
            or scale.shape != (x.shape[0], 1)
            or not x.is_contiguous()
            or not out_fp8.is_contiguous()
            or not scale.is_contiguous()
            or x.shape[1] % 16
            or len({str(x.device), str(out_fp8.device), str(scale.device)}) != 1
        ):
            raise FP8W8A8Error("invalid BF16-to-E4M3 dynamic per-token quantization tensors")
        status = self._quantize(
            ctypes.c_void_p(out_fp8.data_ptr()),
            ctypes.c_void_p(scale.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            x.shape[0],
            x.shape[1],
            ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream),
        )
        if status != 0:
            raise FP8W8A8Error(f"FP8 W8A8 dynamic quantization failed with status {status}")
        return out_fp8, scale

    def workspace_bytes(self, *, m: int, n: int, k: int, batch_invariant: bool) -> int:
        """Return caller-owned workspace bytes for one immutable launch geometry."""
        _validate_geometry(m=m, n=n, k=k)
        workspace_bytes = ctypes.c_size_t()
        status = self._workspace_size(ctypes.byref(workspace_bytes), m, n, k, int(batch_invariant))
        if status != 0:
            raise FP8W8A8Error(f"FP8 W8A8 workspace-size query failed with status {status}")
        return workspace_bytes.value

    def launch(
        self,
        x_fp8: torch.Tensor,
        weight_t: torch.Tensor,
        activation_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
        *,
        batch_invariant: bool,
    ) -> torch.Tensor:
        """Write one fused per-token/per-channel scaled GEMM into ``out``.

        ``weight_t`` must be the non-contiguous transpose view of checkpoint
        storage ``[N, K]``.  Its physical layout is therefore column-major
        ``[K, N]`` without duplicating raw E4M3 weights.
        """
        import torch

        m, n, k = _validate_tensors(
            x_fp8,
            weight_t,
            activation_scale,
            weight_scale,
            out,
            workspace,
            torch_module=torch,
        )
        status = self._launch(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(x_fp8.data_ptr()),
            ctypes.c_void_p(weight_t.data_ptr()),
            ctypes.c_void_p(activation_scale.data_ptr()),
            ctypes.c_void_p(weight_scale.data_ptr()),
            ctypes.c_void_p(workspace.data_ptr()) if workspace.numel() else None,
            workspace.numel(),
            m,
            n,
            k,
            int(batch_invariant),
            ctypes.c_void_p(torch.cuda.current_stream(x_fp8.device).cuda_stream),
        )
        if status != 0:
            raise FP8W8A8Error(f"FP8 W8A8 launch failed with status {status}")
        return out


def _validate_geometry(*, m: int, n: int, k: int) -> None:
    if m <= 0 or n <= 0 or k <= 0 or n % 16 or k % 16:
        raise FP8W8A8Error(f"FP8 W8A8 requires M>0 and N/K multiples of 16, got ({m}, {n}, {k})")


def _validate_tensors(
    x_fp8: Any,
    weight_t: Any,
    activation_scale: Any,
    weight_scale: Any,
    out: Any,
    workspace: Any,
    *,
    torch_module: Any,
) -> tuple[int, int, int]:
    tensors = (x_fp8, weight_t, activation_scale, weight_scale, out, workspace)
    if any(not tensor.is_cuda for tensor in tensors):
        raise FP8W8A8Error("FP8 W8A8 requires CUDA-resident tensors")
    if x_fp8.dtype != torch_module.float8_e4m3fn or weight_t.dtype != torch_module.float8_e4m3fn:
        raise FP8W8A8Error("FP8 W8A8 requires E4M3 activation and weight tensors")
    if activation_scale.dtype != torch_module.float32 or weight_scale.dtype != torch_module.float32:
        raise FP8W8A8Error("FP8 W8A8 requires FP32 activation and weight scales")
    if out.dtype != torch_module.bfloat16:
        raise FP8W8A8Error("FP8 W8A8 requires BF16 output")
    if workspace.dtype != torch_module.uint8:
        raise FP8W8A8Error("FP8 W8A8 workspace must be a uint8 CUDA tensor")
    if x_fp8.ndim != 2 or weight_t.ndim != 2 or out.ndim != 2:
        raise FP8W8A8Error("FP8 W8A8 activation, weight, and output must be rank-2")
    m, k = x_fp8.shape
    weight_k, n = weight_t.shape
    _validate_geometry(m=m, n=n, k=k)
    if weight_k != k or out.shape != (m, n):
        raise FP8W8A8Error(
            f"FP8 W8A8 shape mismatch: A={tuple(x_fp8.shape)}, B={tuple(weight_t.shape)}, "
            f"D={tuple(out.shape)}"
        )
    if activation_scale.numel() != m or weight_scale.numel() != n:
        raise FP8W8A8Error("FP8 W8A8 scale lengths must equal M and N respectively")
    if not x_fp8.is_contiguous() or not out.is_contiguous():
        raise FP8W8A8Error("FP8 W8A8 activation and output must be contiguous")
    if weight_t.stride() != (1, k):
        raise FP8W8A8Error("FP8 W8A8 weight must be the column-major [K, N] transpose view")
    if not activation_scale.is_contiguous() or not weight_scale.is_contiguous():
        raise FP8W8A8Error("FP8 W8A8 scale tensors must be contiguous")
    if not workspace.is_contiguous():
        raise FP8W8A8Error("FP8 W8A8 workspace must be contiguous")
    devices = {str(tensor.device) for tensor in tensors}
    if len(devices) != 1:
        raise FP8W8A8Error("FP8 W8A8 tensors must share one CUDA device")
    return m, n, k
