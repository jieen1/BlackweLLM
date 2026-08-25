"""Raw-pointer adapter for the native GGML K-quant SM120 kernels.

The adapter deliberately owns no weight conversion.  ``GgufLinear`` keeps the
checkpoint's flat ``uint8`` payload and passes it directly to the CUDA shared
library.  A manifest and SHA-256 check make an accidentally stale local
artifact fail at load instead of silently selecting a slow or incompatible
kernel.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

# Dispatch id 6 is a private row-layout variant, so stale artifacts must not
# be accepted merely because their exported function signatures still match.
ABI_VERSION = 12
TARGET_SM = "sm_120f"
ACCEPTED_TARGET_SM = ("sm_120f", "sm_120a")

_GENERATED_DIR = Path(__file__).with_name("_generated")
_LIBRARY_PATH = _GENERATED_DIR / "gguf_qk_sm120.so"
_MANIFEST_PATH = _GENERATED_DIR / "gguf_qk_sm120.manifest.json"

# These are the private ABI's compact dispatch ids.  They deliberately do
# not reuse GGML's global enum values because the CUDA kernel only exposes the
# four formats this checkpoint actually contains.
_TYPE_IDS = {
    "Q4_K": 0,
    "Q5_K": 1,
    "Q6_K": 2,
    "Q8_0": 3,
    # Private runtime layout: Q6_K blocks padded to 224 bytes for aligned
    # 32-bit payload loads.  The checkpoint itself remains standard Q6_K.
    "Q6_K_ALIGNED": 4,
    # Private runtime layout: 208-byte payload blocks followed by a row-tail
    # array of FP16 d values; total storage stays equal to standard Q6_K.
    "Q6_K_SPLIT": 5,
    # Private runtime layout: 32-byte Q8_0 payload blocks followed by a
    # row-tail array of FP16 d values; total storage stays equal to Q8_0.
    "Q8_0_SPLIT": 6,
}


class GgufQKError(RuntimeError):
    """The native GGUF Q/K artifact cannot satisfy its fixed contract."""


@dataclass(frozen=True)
class GgufQKManifest:
    abi_version: int
    target_sm: str
    library_sha256: str

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> GgufQKManifest:
        try:
            return cls(
                abi_version=int(value["abi_version"]),
                target_sm=str(value["target_sm"]),
                library_sha256=str(value["library_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise GgufQKError("invalid GGUF Q/K manifest") from error


def artifact_paths() -> tuple[Path, Path]:
    """Return the generated native library and its provenance manifest."""

    return _LIBRARY_PATH, _MANIFEST_PATH


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> GgufQKManifest:
    if not path.is_file():
        raise GgufQKError(f"GGUF Q/K manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GgufQKError(f"cannot read GGUF Q/K manifest: {path}") from error
    return GgufQKManifest.from_json(value)


def _require_sm120_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise GgufQKError("native GGUF Q/K requires an available SM120 CUDA device")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise GgufQKError(f"native GGUF Q/K requires CUDA capability (12, 0), got {capability}")


def _q8_activation_cache_enabled() -> bool:
    value = os.environ.get("QSR_GGUF_Q8_CACHE_ACTIVATION", "0").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def _q8_activation_shared_bytes(k: int) -> int:
    padded_k = ((k + 511) // 512) * 512
    return padded_k // 32 * 36


class NativeGgufQK:
    """Loaded raw-pointer ABI for Q4_K/Q5_K/Q6_K/Q8_0 GEMM and row gather."""

    def __init__(self, library: ctypes.CDLL) -> None:
        self._library = library
        # The Q8_1 activation buffer is scratch, not model state.  Keep one
        # buffer per fixed (device, M, padded-K) shape so every packed Linear
        # does not enter the CUDA allocator on the eager path.  The runtime
        # is single-stream by contract; graph capture warms each shape before
        # capture and then reuses the same address on replay.
        self._activation_workspaces: dict[tuple[int, int, int], Any] = {}
        self._quantize_q8 = library.qsr_gguf_quantize_q8_sm120
        self._quantize_q8_f32 = library.qsr_gguf_quantize_q8_f32_sm120
        self._gemm_q8 = library.qsr_gguf_gemm_q8_sm120
        self._gemm_q8_prequantized = library.qsr_gguf_gemm_q8_prequantized_sm120
        self._gemm_q8_prequantized_cached = library.qsr_gguf_gemm_q8_prequantized_cached_sm120
        self._gemm_q8_mmq = library.qsr_gguf_gemm_q8_mmq_sm120
        self._gemm_q8_f32 = library.qsr_gguf_gemm_q8_f32_sm120
        self._gemm_q8_prequantized_f32 = library.qsr_gguf_gemm_q8_prequantized_f32_sm120
        self._gemm_q8_prequantized_cached_f32 = (
            library.qsr_gguf_gemm_q8_prequantized_cached_f32_sm120
        )
        self._gemm_q8_mixed = library.qsr_gguf_gemm_q8_mixed_sm120
        self._gemm_q8_mixed_f32 = library.qsr_gguf_gemm_q8_mixed_f32_sm120
        self._gemm_q8_mixed_cached = library.qsr_gguf_gemm_q8_mixed_cached_sm120
        self._gemm_q8_mixed_cached_f32 = library.qsr_gguf_gemm_q8_mixed_cached_f32_sm120
        self._gemm_direct_mixed = library.qsr_gguf_gemm_direct_mixed_sm120
        self._gemm_direct_mixed_f32 = library.qsr_gguf_gemm_direct_mixed_f32_sm120
        self._gemm_direct_mixed_cached = library.qsr_gguf_gemm_direct_mixed_cached_sm120
        self._gemm_direct_mixed_cached_f32 = library.qsr_gguf_gemm_direct_mixed_cached_f32_sm120
        self._gemm = library.qsr_gguf_gemm_sm120
        self._gemm_f32 = library.qsr_gguf_gemm_f32_sm120
        self._gemm_direct_cached = library.qsr_gguf_gemm_direct_cached_sm120
        self._gemm_direct_cached_f32 = library.qsr_gguf_gemm_direct_cached_f32_sm120
        self._dequant_rows = library.qsr_gguf_dequant_rows_sm120
        self._dequant_rows_f32 = library.qsr_gguf_dequant_rows_f32_sm120
        self._gemm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._gemm.restype = ctypes.c_int
        self._gemm_f32.argtypes = self._gemm.argtypes
        self._gemm_f32.restype = ctypes.c_int
        self._gemm_direct_cached.argtypes = self._gemm.argtypes
        self._gemm_direct_cached.restype = ctypes.c_int
        self._gemm_direct_cached_f32.argtypes = self._gemm.argtypes
        self._gemm_direct_cached_f32.restype = ctypes.c_int
        self._gemm_q8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._gemm_q8.restype = ctypes.c_int
        self._quantize_q8.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._quantize_q8.restype = ctypes.c_int
        self._quantize_q8_f32.argtypes = self._quantize_q8.argtypes
        self._quantize_q8_f32.restype = ctypes.c_int
        self._gemm_q8_prequantized.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._gemm_q8_prequantized.restype = ctypes.c_int
        self._gemm_q8_prequantized_cached.argtypes = self._gemm_q8_prequantized.argtypes
        self._gemm_q8_prequantized_cached.restype = ctypes.c_int
        self._gemm_q8_mmq.argtypes = self._gemm_q8_prequantized.argtypes
        self._gemm_q8_mmq.restype = ctypes.c_int
        self._gemm_q8_f32.argtypes = self._gemm_q8.argtypes
        self._gemm_q8_f32.restype = ctypes.c_int
        self._gemm_q8_prequantized_f32.argtypes = self._gemm_q8_prequantized.argtypes
        self._gemm_q8_prequantized_f32.restype = ctypes.c_int
        self._gemm_q8_prequantized_cached_f32.argtypes = self._gemm_q8_prequantized.argtypes
        self._gemm_q8_prequantized_cached_f32.restype = ctypes.c_int
        self._gemm_q8_mixed.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._gemm_q8_mixed.restype = ctypes.c_int
        self._gemm_q8_mixed_f32.argtypes = self._gemm_q8_mixed.argtypes
        self._gemm_q8_mixed_f32.restype = ctypes.c_int
        self._gemm_q8_mixed_cached.argtypes = self._gemm_q8_mixed.argtypes
        self._gemm_q8_mixed_cached.restype = ctypes.c_int
        self._gemm_q8_mixed_cached_f32.argtypes = self._gemm_q8_mixed.argtypes
        self._gemm_q8_mixed_cached_f32.restype = ctypes.c_int
        self._gemm_direct_mixed.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._gemm_direct_mixed.restype = ctypes.c_int
        self._gemm_direct_mixed_f32.argtypes = self._gemm_direct_mixed.argtypes
        self._gemm_direct_mixed_f32.restype = ctypes.c_int
        self._gemm_direct_mixed_cached.argtypes = self._gemm_direct_mixed.argtypes
        self._gemm_direct_mixed_cached.restype = ctypes.c_int
        self._gemm_direct_mixed_cached_f32.argtypes = self._gemm_direct_mixed.argtypes
        self._gemm_direct_mixed_cached_f32.restype = ctypes.c_int
        self._dequant_rows.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._dequant_rows.restype = ctypes.c_int
        self._dequant_rows_f32.argtypes = self._dequant_rows.argtypes
        self._dequant_rows_f32.restype = ctypes.c_int

    @classmethod
    def load(cls) -> NativeGgufQK:
        if not _LIBRARY_PATH.is_file():
            raise GgufQKError(
                f"native GGUF Q/K library is missing: {_LIBRARY_PATH}; "
                "run `make build-gguf-qk PYTHON=/home/bot/.venvs/torch-nightly/bin/python`"
            )
        manifest = _load_manifest(_MANIFEST_PATH)
        if manifest.abi_version != ABI_VERSION:
            raise GgufQKError(
                f"native GGUF Q/K ABI mismatch: expected {ABI_VERSION}, got {manifest.abi_version}"
            )
        if manifest.target_sm not in ACCEPTED_TARGET_SM:
            raise GgufQKError(
                "native GGUF Q/K target mismatch: expected one of "
                f"{', '.join(ACCEPTED_TARGET_SM)}, got {manifest.target_sm}"
            )
        if _sha256(_LIBRARY_PATH) != manifest.library_sha256:
            raise GgufQKError("native GGUF Q/K library SHA256 does not match its manifest")
        _require_sm120_cuda()
        try:
            library = ctypes.CDLL(str(_LIBRARY_PATH))
        except OSError as error:
            raise GgufQKError(f"cannot load native GGUF Q/K library: {_LIBRARY_PATH}") from error
        abi = library.qsr_gguf_qk_abi_version
        abi.argtypes = []
        abi.restype = ctypes.c_int
        if abi() != ABI_VERSION:
            raise GgufQKError("native GGUF Q/K library exports an incompatible ABI")
        return cls(library)

    @staticmethod
    def _validate_common(x: torch.Tensor, packed: torch.Tensor, *, bf16_only: bool = False) -> None:
        import torch

        supported_dtype = x.dtype in (torch.bfloat16, torch.float32)
        if (
            x.device.type != "cuda"
            or packed.device != x.device
            or not supported_dtype
            or (bf16_only and x.dtype != torch.bfloat16)
            or packed.dtype != torch.uint8
            or not x.is_contiguous()
            or not packed.is_contiguous()
        ):
            expected = "contiguous CUDA BF16" if bf16_only else "contiguous CUDA BF16/F32"
            raise GgufQKError(f"native GGUF Q/K expects {expected} input and uint8 weight")

    def gemm(
        self,
        x: torch.Tensor,
        packed: torch.Tensor,
        *,
        m: int,
        n: int,
        k: int,
        row_bytes: int,
        type_name: str,
    ) -> torch.Tensor:
        import torch

        self._validate_common(x, packed, bf16_only=True)
        if x.shape != (m, k):
            raise GgufQKError(f"input shape {tuple(x.shape)} != ({m}, {k})")
        if packed.numel() != n * row_bytes:
            raise GgufQKError(f"packed bytes {packed.numel()} != {n * row_bytes}")
        type_id = _TYPE_IDS.get(type_name)
        if type_id is None:
            raise GgufQKError(f"unsupported native GGUF type {type_name!r}")
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        padded_k = ((k + 511) // 512) * 512
        activation_bytes = m * padded_k // 32 * 36
        device_key = x.device.index if x.device.index is not None else 0
        workspace_key = (device_key, m, padded_k)
        activation_workspace = self._activation_workspaces.get(workspace_key)
        workspace_words = (activation_bytes + 3) // 4
        if activation_workspace is None or activation_workspace.numel() != workspace_words:
            if torch.cuda.is_current_stream_capturing():
                raise GgufQKError(
                    "native GGUF Q/K activation workspace is missing during CUDA Graph "
                    "capture; run an eager warmup for this (M,K) shape first"
                )
            activation_workspace = torch.empty(
                workspace_words,
                dtype=torch.int32,
                device=x.device,
            )
            self._activation_workspaces[workspace_key] = activation_workspace
        status = self._gemm_q8(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(packed.data_ptr()),
            ctypes.c_void_p(activation_workspace.data_ptr()),
            m,
            n,
            k,
            row_bytes,
            type_id,
            ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream),
        )
        if status != 0:
            raise GgufQKError(f"native GGUF Q/K GEMM failed with status {status}")
        return out

    def _activation_workspace(self, x: torch.Tensor, *, m: int, k: int) -> tuple[torch.Tensor, int]:
        import torch

        padded_k = ((k + 511) // 512) * 512
        activation_bytes = m * padded_k // 32 * 36
        device_key = x.device.index if x.device.index is not None else 0
        workspace_key = (device_key, m, padded_k)
        activation_workspace = self._activation_workspaces.get(workspace_key)
        workspace_words = (activation_bytes + 3) // 4
        if activation_workspace is None or activation_workspace.numel() != workspace_words:
            if torch.cuda.is_current_stream_capturing():
                raise GgufQKError(
                    "native GGUF Q/K activation workspace is missing during CUDA Graph "
                    "capture; run an eager warmup for this (M,K) shape first"
                )
            activation_workspace = torch.empty(
                workspace_words,
                dtype=torch.int32,
                device=x.device,
            )
            self._activation_workspaces[workspace_key] = activation_workspace
        return activation_workspace, padded_k

    def gemm_q8_f32(
        self,
        x: torch.Tensor,
        packed: torch.Tensor,
        *,
        m: int,
        n: int,
        k: int,
        row_bytes: int,
        type_name: str,
    ) -> torch.Tensor:
        """Run SGLang-style F32-input Q8_1 GEMV without BF16 boundary rounding."""

        import torch

        if x.dtype != torch.float32 or m != 1:
            raise GgufQKError("F32 Q8_1 GGUF GEMV expects F32 input with M=1")
        self._validate_common(x, packed)
        if x.shape != (m, k):
            raise GgufQKError(f"input shape {tuple(x.shape)} != ({m}, {k})")
        if packed.numel() != n * row_bytes:
            raise GgufQKError(f"packed bytes {packed.numel()} != {n * row_bytes}")
        type_id = _TYPE_IDS.get(type_name)
        if type_id is None:
            raise GgufQKError(f"unsupported native GGUF type {type_name!r}")
        activation_workspace, _ = self._activation_workspace(x, m=m, k=k)
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        status = self._gemm_q8_f32(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(packed.data_ptr()),
            ctypes.c_void_p(activation_workspace.data_ptr()),
            m,
            n,
            k,
            row_bytes,
            type_id,
            ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream),
        )
        if status != 0:
            raise GgufQKError(f"native F32 GGUF Q/K GEMV failed with status {status}")
        return out

    def quantize_q8_1(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize one BF16/F32 activation matrix into Q8_1 scratch."""

        import torch

        if (
            x.device.type != "cuda"
            or x.dtype not in (torch.bfloat16, torch.float32)
            or x.ndim != 2
            or not x.is_contiguous()
        ):
            raise GgufQKError("Q8_1 activation quantization expects contiguous CUDA BF16/F32 [M,K]")
        activation_workspace, _ = self._activation_workspace(x, m=x.shape[0], k=x.shape[1])
        quantize = self._quantize_q8 if x.dtype == torch.bfloat16 else self._quantize_q8_f32
        status = quantize(
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(activation_workspace.data_ptr()),
            x.shape[0],
            x.shape[1],
            ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream),
        )
        if status != 0:
            raise GgufQKError(
                f"native GGUF Q/K activation quantization failed with status {status}"
            )
        return activation_workspace

    def gemm_q8_prequantized(
        self,
        activation_workspace: torch.Tensor,
        packed: torch.Tensor,
        *,
        m: int,
        n: int,
        k: int,
        row_bytes: int,
        type_name: str,
        output_dtype: Any | None = None,
        cache_activation: bool | None = None,
    ) -> torch.Tensor:
        """Run a packed GEMM using an already quantized Q8_1 activation."""

        import torch

        if (
            activation_workspace.device.type != "cuda"
            or activation_workspace.dtype != torch.int32
            or not activation_workspace.is_contiguous()
        ):
            raise GgufQKError("prequantized GGUF Q/K GEMM expects contiguous CUDA Q8_1 scratch")
        padded_k = ((k + 511) // 512) * 512
        expected_words = m * padded_k // 32 * 9
        if activation_workspace.numel() != expected_words:
            raise GgufQKError(
                f"Q8_1 activation workspace {activation_workspace.numel()} words != "
                f"{expected_words} for ({m}, {k})"
            )
        if (
            packed.device != activation_workspace.device
            or packed.dtype != torch.uint8
            or not packed.is_contiguous()
        ):
            raise GgufQKError("prequantized GGUF Q/K GEMM expects matching CUDA uint8 weights")
        if packed.numel() != n * row_bytes:
            raise GgufQKError(f"packed bytes {packed.numel()} != {n * row_bytes}")
        type_id = _TYPE_IDS.get(type_name)
        if type_id is None:
            raise GgufQKError(f"unsupported native GGUF type {type_name!r}")
        if output_dtype is None:
            output_dtype = torch.bfloat16
        if output_dtype not in (torch.bfloat16, torch.float32):
            raise GgufQKError(f"unsupported GGUF Q/K output dtype {output_dtype}")
        if output_dtype == torch.float32 and m != 1:
            raise GgufQKError("F32 prequantized GGUF Q/K output currently requires M=1")
        cache_requested = (
            _q8_activation_cache_enabled() if cache_activation is None else cache_activation
        )
        activation_shared_bytes = _q8_activation_shared_bytes(k)
        use_cached = cache_requested and (
            (m == 1 and activation_shared_bytes <= 48 * 1024)
            or (
                m == 8
                and output_dtype == torch.bfloat16
                and type_name == "Q6_K_SPLIT"
                and activation_shared_bytes * m <= 48 * 1024
            )
        )
        out = torch.empty((m, n), dtype=output_dtype, device=packed.device)
        gemm = (
            self._gemm_q8_prequantized_cached
            if use_cached and output_dtype == torch.bfloat16
            else self._gemm_q8_prequantized_cached_f32
            if use_cached
            else self._gemm_q8_prequantized
            if output_dtype == torch.bfloat16
            else self._gemm_q8_prequantized_f32
        )
        status = gemm(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(activation_workspace.data_ptr()),
            ctypes.c_void_p(packed.data_ptr()),
            m,
            n,
            k,
            row_bytes,
            type_id,
            ctypes.c_void_p(torch.cuda.current_stream(packed.device).cuda_stream),
        )
        if status != 0:
            raise GgufQKError(f"native prequantized GGUF Q/K GEMM failed with status {status}")
        return out

    def gemm_q8_mmq(
        self,
        activation_workspace: torch.Tensor,
        packed: torch.Tensor,
        *,
        m: int,
        n: int,
        k: int,
        row_bytes: int,
        type_name: str,
    ) -> torch.Tensor:
        """Run the opt-in SGLang-style Q5_K/Q6_K/Q8_0 MMQ verify tile."""

        import torch

        if (
            activation_workspace.device.type != "cuda"
            or activation_workspace.dtype != torch.int32
            or not activation_workspace.is_contiguous()
            or packed.device != activation_workspace.device
            or packed.dtype != torch.uint8
            or not packed.is_contiguous()
        ):
            raise GgufQKError("MMQ expects matching CUDA Q8_1 scratch and uint8 weights")
        if type_name == "Q6_K_SPLIT":
            if m <= 0 or n <= 0 or k <= 0 or k % 256 or row_bytes != (k // 256) * 210:
                raise GgufQKError("Q6 MMQ geometry must use standard-size Q6_K_SPLIT rows")
        elif type_name == "Q5_K":
            if m <= 0 or n <= 0 or k <= 0 or k % 256 or row_bytes != (k // 256) * 176:
                raise GgufQKError("Q5 MMQ geometry must use standard-size Q5_K rows")
        elif type_name in {"Q8_0", "Q8_0_SPLIT"}:
            if m <= 0 or n <= 0 or k <= 0 or k % 128 or row_bytes != (k // 32) * 34:
                raise GgufQKError("Q8 MMQ geometry must use Q8_0 rows")
        else:
            raise GgufQKError(f"MMQ does not support {type_name!r}")
        padded_k = ((k + 511) // 512) * 512
        expected_words = m * padded_k // 32 * 9
        if activation_workspace.numel() != expected_words:
            raise GgufQKError(
                f"Q8_1 activation workspace {activation_workspace.numel()} words != "
                f"{expected_words} for ({m}, {k})"
            )
        if packed.numel() != n * row_bytes:
            raise GgufQKError(f"packed bytes {packed.numel()} != {n * row_bytes}")
        out = torch.empty((m, n), dtype=torch.bfloat16, device=packed.device)
        status = self._gemm_q8_mmq(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(activation_workspace.data_ptr()),
            ctypes.c_void_p(packed.data_ptr()),
            m,
            n,
            k,
            row_bytes,
            _TYPE_IDS[type_name],
            ctypes.c_void_p(torch.cuda.current_stream(packed.device).cuda_stream),
        )
        if status != 0:
            raise GgufQKError(f"native MMQ GEMM failed with status {status}")
        return out

    def gemm_q8_mixed(
        self,
        activation_workspace: torch.Tensor,
        descriptors: torch.Tensor,
        *,
        projection_count: int,
        total_n: int,
        k: int,
        output_dtype: Any | None = None,
        cache_activation: bool | None = None,
    ) -> torch.Tensor:
        """Run one decode GEMV over adjacent Q4/Q5/Q6/Q8 output segments."""

        import torch

        if (
            activation_workspace.device.type != "cuda"
            or activation_workspace.dtype != torch.int32
            or not activation_workspace.is_contiguous()
            or descriptors.device != activation_workspace.device
            or descriptors.dtype != torch.int64
            or descriptors.ndim != 2
            or descriptors.shape != (projection_count, 4)
            or not descriptors.is_contiguous()
        ):
            raise GgufQKError("mixed GGUF Q/K GEMM expects contiguous CUDA descriptors and scratch")
        padded_k = ((k + 511) // 512) * 512
        expected_words = padded_k // 32 * 9
        if activation_workspace.numel() != expected_words:
            raise GgufQKError(
                f"Q8_1 activation workspace {activation_workspace.numel()} words != "
                f"{expected_words} for ({1}, {k})"
            )
        if output_dtype is None:
            output_dtype = torch.bfloat16
        if output_dtype not in (torch.bfloat16, torch.float32):
            raise GgufQKError(f"unsupported GGUF Q/K output dtype {output_dtype}")
        use_cached = (
            _q8_activation_cache_enabled() if cache_activation is None else cache_activation
        ) and _q8_activation_shared_bytes(k) <= 48 * 1024
        out = torch.empty((1, total_n), dtype=output_dtype, device=descriptors.device)
        gemm = (
            self._gemm_q8_mixed_cached
            if use_cached and output_dtype == torch.bfloat16
            else self._gemm_q8_mixed_cached_f32
            if use_cached
            else self._gemm_q8_mixed
            if output_dtype == torch.bfloat16
            else self._gemm_q8_mixed_f32
        )
        status = gemm(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(activation_workspace.data_ptr()),
            ctypes.c_void_p(descriptors.data_ptr()),
            projection_count,
            total_n,
            k,
            ctypes.c_void_p(torch.cuda.current_stream(descriptors.device).cuda_stream),
        )
        if status != 0:
            raise GgufQKError(f"native mixed GGUF Q/K GEMM failed with status {status}")
        return out

    def gemm_direct(
        self,
        x: torch.Tensor,
        packed: torch.Tensor,
        *,
        m: int,
        n: int,
        k: int,
        row_bytes: int,
        type_name: str,
        cache_activation: bool = False,
    ) -> torch.Tensor:
        """Run the packed BF16 reference-quality native GEMM.

        This path decodes each packed value directly and does not quantize the
        activation to Q8_1 first.  It is slower than :meth:`gemm`, but is a
        useful production fallback for numerical bisects and workloads whose
        recurrent state is unusually sensitive to activation quantization.
        It still keeps the checkpoint packed and never creates a BF16 weight
        matrix.
        """

        import torch

        self._validate_common(x, packed)
        if x.shape != (m, k):
            raise GgufQKError(f"input shape {tuple(x.shape)} != ({m}, {k})")
        if packed.numel() != n * row_bytes:
            raise GgufQKError(f"packed bytes {packed.numel()} != {n * row_bytes}")
        type_id = _TYPE_IDS.get(type_name)
        if type_id is None:
            raise GgufQKError(f"unsupported native GGUF type {type_name!r}")
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        use_cached = cache_activation and m == 1 and k * x.element_size() <= 48 * 1024
        if use_cached:
            gemm = (
                self._gemm_direct_cached
                if x.dtype == torch.bfloat16
                else self._gemm_direct_cached_f32
            )
        else:
            gemm = self._gemm if x.dtype == torch.bfloat16 else self._gemm_f32
        status = gemm(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(packed.data_ptr()),
            m,
            n,
            k,
            row_bytes,
            type_id,
            ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream),
        )
        if status != 0:
            raise GgufQKError(f"native direct GGUF Q/K GEMM failed with status {status}")
        return out

    def gemm_direct_mixed(
        self,
        x: torch.Tensor,
        descriptors: torch.Tensor,
        *,
        projection_count: int,
        total_n: int,
        k: int,
        cache_activation: bool = False,
    ) -> torch.Tensor:
        """Run one exact BF16/F32 GEMV over adjacent mixed-format projections."""

        import torch

        if (
            x.device.type != "cuda"
            or x.dtype not in (torch.bfloat16, torch.float32)
            or x.ndim != 2
            or x.shape != (1, k)
            or not x.is_contiguous()
            or descriptors.device != x.device
            or descriptors.dtype != torch.int64
            or descriptors.ndim != 2
            or descriptors.shape != (projection_count, 4)
            or not descriptors.is_contiguous()
        ):
            raise GgufQKError(
                "direct mixed GGUF Q/K GEMM expects contiguous CUDA BF16/F32 input and descriptors"
            )
        out = torch.empty((1, total_n), dtype=x.dtype, device=x.device)
        use_cached = cache_activation and k * x.element_size() <= 48 * 1024
        if use_cached:
            gemm = (
                self._gemm_direct_mixed_cached
                if x.dtype == torch.bfloat16
                else self._gemm_direct_mixed_cached_f32
            )
        else:
            gemm = (
                self._gemm_direct_mixed
                if x.dtype == torch.bfloat16
                else self._gemm_direct_mixed_f32
            )
        status = gemm(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(descriptors.data_ptr()),
            projection_count,
            total_n,
            k,
            ctypes.c_void_p(torch.cuda.current_stream(x.device).cuda_stream),
        )
        if status != 0:
            raise GgufQKError(f"native direct mixed GGUF Q/K GEMM failed with status {status}")
        return out

    def gemm_tensor_core(
        self,
        x: torch.Tensor,
        packed: torch.Tensor,
        *,
        m: int,
        n: int,
        k: int,
        row_bytes: int,
        type_name: str,
    ) -> torch.Tensor:
        """Run packed Q/K GEMM through the BF16 tensor-core decoder.

        Triton is imported lazily so the torch-free test job can still import
        the model graph.  The kernel decodes each GGML block into a BF16 tile
        and feeds that tile to ``tl.dot``; it is therefore a native packed
        path, not a request to materialize a persistent dequantized weight.
        """

        from runtime.kernels.gguf_qk_triton import gguf_qk_gemm

        return gguf_qk_gemm(
            x,
            packed,
            m=m,
            n=n,
            k=k,
            row_bytes=row_bytes,
            type_name=type_name,
        )

    def gemm_tensor_core_tile_major(
        self,
        x: torch.Tensor,
        packed: torch.Tensor,
        *,
        m: int,
        n: int,
        k: int,
        type_name: str,
        block_n: int,
    ) -> torch.Tensor:
        """Run the exact packed decoder over a cached N-tile-major payload."""

        from runtime.kernels.gguf_qk_triton import gguf_qk_gemm_tile_major

        return gguf_qk_gemm_tile_major(
            x,
            packed,
            m=m,
            n=n,
            k=k,
            type_name=type_name,
            block_n=block_n,
        )

    def dequant_rows(
        self,
        input_ids: torch.Tensor,
        packed: torch.Tensor,
        *,
        rows: int,
        k: int,
        row_bytes: int,
        type_name: str,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        import torch

        if (
            input_ids.device.type != "cuda"
            or packed.device != input_ids.device
            or input_ids.dtype != torch.int64
            or packed.dtype != torch.uint8
            or input_ids.ndim != 1
            or input_ids.numel() != rows
            or not input_ids.is_contiguous()
            or not packed.is_contiguous()
        ):
            raise GgufQKError("native GGUF row gather expects a contiguous int64 id vector")
        type_id = _TYPE_IDS.get(type_name)
        if type_id is None:
            raise GgufQKError(f"unsupported native GGUF type {type_name!r}")
        if dtype is None:
            dtype = torch.bfloat16
        if dtype not in (torch.bfloat16, torch.float32):
            raise GgufQKError(f"native GGUF row gather does not support output dtype {dtype}")
        out_dtype = dtype
        dequant_rows = self._dequant_rows if dtype == torch.bfloat16 else self._dequant_rows_f32
        out = torch.empty((rows, k), dtype=out_dtype, device=input_ids.device)
        status = dequant_rows(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(input_ids.data_ptr()),
            ctypes.c_void_p(packed.data_ptr()),
            rows,
            k,
            row_bytes,
            type_id,
            ctypes.c_void_p(torch.cuda.current_stream(input_ids.device).cuda_stream),
        )
        if status != 0:
            raise GgufQKError(f"native GGUF row gather failed with status {status}")
        return out
