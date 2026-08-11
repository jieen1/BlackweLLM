"""Native IQ2_XS -> INT8 tensor-core grouped-MoE adapter (Phase 2).

Wraps the hand-written ``iq2_mma16.cu`` m16n8k16 kernel: exact IQ2_XS
decode (per-K16 fp32 scale ``d*(0.5+nibble)*0.25``), fused dual gate+up,
B-decode reuse across an expert's M tokens, and per-K16 scale applied to
the mma int32 result.  Numerics match ``dequantize_iq2_xs`` bit-for-bit
up to fp32 accumulation order (cos >= 0.9999).

The generated library has a raw CUDA-pointer ABI; this module imports
``torch`` only when a caller actually loads or launches the artifact so
CPU-only test collection stays safe.
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
TARGET_SM = "sm_120f"
ACCEPTED_TARGET_SM = ("sm_120f", "sm_120a")

_KERNEL_DIR = Path(__file__).resolve().parent
_GENERATED_DIR = _KERNEL_DIR / "_generated"
_LIBRARY_PATH = _GENERATED_DIR / "iq2_mma16.so"
_MANIFEST_PATH = _GENERATED_DIR / "iq2_mma16.manifest.json"


class IQ2MMA16Error(RuntimeError):
    """The native IQ2 MMA16 artifact cannot satisfy its fixed contract."""


@dataclass(frozen=True)
class IQ2MMA16Manifest:
    abi_version: int
    target_sm: str
    library_sha256: str

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> IQ2MMA16Manifest:
        try:
            return cls(
                abi_version=int(value["abi_version"]),
                target_sm=str(value["target_sm"]),
                library_sha256=str(value["library_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise IQ2MMA16Error("invalid IQ2 MMA16 manifest") from error


def artifact_paths() -> tuple[Path, Path]:
    """Return the single supported generated library and manifest paths."""
    return _LIBRARY_PATH, _MANIFEST_PATH


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> IQ2MMA16Manifest:
    if not path.is_file():
        raise IQ2MMA16Error(f"IQ2 MMA16 manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IQ2MMA16Error(f"cannot read IQ2 MMA16 manifest: {path}") from error
    return IQ2MMA16Manifest.from_json(value)


class NativeIQ2MMA16Library:
    """Loaded raw-pointer ABI with strict artifact and tensor validation."""

    def __init__(self, library: ctypes.CDLL) -> None:
        self._library = library
        self._launch = library.iq2_mma16_launch
        self._launch.argtypes = [
            ctypes.c_void_p,  # xq [E, M_PAD, K] int8
            ctypes.c_void_p,  # xs [E, M_PAD, K/32] fp32
            ctypes.c_void_p,  # packed_gate [E, ROWS*STRIDE] uint8
            ctypes.c_void_p,  # packed_up   [E, ROWS*STRIDE] uint8
            ctypes.c_void_p,  # eids [E] int64
            ctypes.c_void_p,  # grid [512] int64
            ctypes.c_void_p,  # ksigns [128] int32
            ctypes.c_void_p,  # out_gate [E, M_PAD, ROWS] fp32
            ctypes.c_void_p,  # out_up   [E, M_PAD, ROWS] fp32
            ctypes.c_int,     # E
            ctypes.c_int,     # ROWS
            ctypes.c_int,     # COLS
            ctypes.c_int,     # STRIDE (bytes per weight row)
            ctypes.c_int,     # M_PAD
        ]
        self._launch.restype = None

    @classmethod
    def load(cls) -> NativeIQ2MMA16Library:
        if not _LIBRARY_PATH.is_file():
            raise IQ2MMA16Error(f"IQ2 MMA16 library is missing: {_LIBRARY_PATH}")
        manifest = _load_manifest(_MANIFEST_PATH)
        if manifest.abi_version != ABI_VERSION:
            raise IQ2MMA16Error(
                f"IQ2 MMA16 ABI mismatch: expected {ABI_VERSION}, got {manifest.abi_version}"
            )
        if manifest.target_sm not in ACCEPTED_TARGET_SM:
            raise IQ2MMA16Error(
                f"IQ2 MMA16 target mismatch: expected one of "
                f"{', '.join(ACCEPTED_TARGET_SM)}, got {manifest.target_sm}"
            )
        if _sha256(_LIBRARY_PATH) != manifest.library_sha256:
            raise IQ2MMA16Error("IQ2 MMA16 library SHA256 does not match its manifest")
        try:
            library = ctypes.CDLL(str(_LIBRARY_PATH))
        except OSError as error:
            raise IQ2MMA16Error(f"cannot load IQ2 MMA16 library: {_LIBRARY_PATH}") from error
        return cls(library)

    def grouped_gate_up(
        self,
        xq: torch.Tensor,
        xs: torch.Tensor,
        packed_gate: torch.Tensor,
        packed_up: torch.Tensor,
        eids: torch.Tensor,
        grid: torch.Tensor,
        ksigns: torch.Tensor,
        *,
        rows: int,
        cols: int,
        stride: int,
        m_pad: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused grouped gate+up ``x @ W^T`` for all routed experts.

        ``xq``/``xs`` are preq-quantized activations shaped ``[E, M_PAD, K]``
        and ``[E, M_PAD, K/32]``; each expert ``e`` occupies ``eids[e]``'s
        packed weight rows.  Returns ``(gate, up)`` each ``[E, M_PAD, ROWS]``.
        """
        import torch

        if xq.device.type != "cuda":
            raise IQ2MMA16Error("iq2_mma16 requires CUDA tensors")
        E = int(eids.numel())
        for name, t in [
            ("xq", xq), ("xs", xs), ("packed_gate", packed_gate),
            ("packed_up", packed_up), ("eids", eids), ("grid", grid),
            ("ksigns", ksigns),
        ]:
            if not t.is_contiguous():
                raise IQ2MMA16Error(f"{name} must be contiguous")
        if xq.shape != (E, m_pad, cols):
            raise IQ2MMA16Error(f"xq shape {tuple(xq.shape)} != ({E}, {m_pad}, {cols})")
        if xs.shape != (E, m_pad, cols // 32):
            raise IQ2MMA16Error(f"xs shape {tuple(xs.shape)} != ({E}, {m_pad}, {cols // 32})")
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
    """Build the provenance manifest (used by the Makefile target)."""
    payload = {
        "abi_version": ABI_VERSION,
        "target_sm": TARGET_SM,
        "library_sha256": _sha256(library_path),
    }
    if source_path is not None:
        payload["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return payload


def grouped_moe_prefill(
    flat: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    gate_packed: torch.Tensor,
    up_packed: torch.Tensor,
    down_packed: torch.Tensor,
    grid: torch.Tensor,
    ksigns: torch.Tensor,
    *,
    inter: int,
    hidden: int,
    swiglu_limit: float,
    m_pad: int = 32,
    library: NativeIQ2MMA16Library | None = None,
) -> torch.Tensor:
    """Grouped MoE prefill: route -> gather -> gate/up -> SwiGLU -> down -> reduce.

    ``flat`` is ``[M, hidden]`` fp32 activations; ``weights``/``indices`` are
    the routed ``[M, top_k]`` from the gate.  Routes are grouped by expert into
    ``[E_active, m_pad, hidden]`` activation tiles, run through the exact
    tensor-core gate/up/down kernels, then scattered back and reduced in stable
    expert-id order to match the eager reference.

    Returns the routed MoE output ``[M, hidden]`` fp32 (shared expert not
    included; the caller adds it).
    """
    import torch

    if flat.device.type != "cuda":
        raise IQ2MMA16Error("grouped_moe_prefill requires CUDA")
    if library is None:
        library = NativeIQ2MMA16Library.load()
    M, top_k = indices.shape
    routes = indices.reshape(-1)                      # [R]
    rweights = weights.reshape(-1)                    # [R]
    # Group routes by expert id (stable): sort eids, gather token/slot/weight.
    order = torch.argsort(routes, stable=True)
    eids_sorted = routes[order]
    # per-expert route ranges
    change = eids_sorted[1:] != eids_sorted[:-1]
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=flat.device),
                        torch.nonzero(change, as_tuple=False).squeeze(-1) + 1])
    n_experts = starts.numel()
    ends = torch.cat([starts[1:], torch.tensor([routes.numel()], dtype=torch.long,
                                               device=flat.device)])
    # cap m_pad at the max routes per expert (no padding waste)
    counts = ends - starts
    max_routes = int(counts.max().item())
    eff_pad = max(16, (max_routes + 15) // 16 * 16)
    eff_pad = min(eff_pad, m_pad)
    if eff_pad < max_routes:
        eff_pad = (max_routes + 15) // 16 * 16
    eff_pad = max(eff_pad, 16)

    # Build [E, eff_pad, hidden] activation tile via vectorized scatter.
    rt = torch.arange(M, device=flat.device).repeat_interleave(top_k)[order]
    rw = rweights[order]
    # per-route slot within its expert = position among routes of the same expert
    within = torch.arange(routes.numel(), device=flat.device) - torch.repeat_interleave(
        starts, ends - starts)
    xq_all = torch.zeros(n_experts, eff_pad, hidden, dtype=torch.int8, device=flat.device)
    xs_all = torch.zeros(n_experts, eff_pad, hidden // 32, dtype=torch.float32, device=flat.device)
    w_all = torch.zeros(n_experts, eff_pad, dtype=torch.float32, device=flat.device)
    # quantize activations (per 32)
    xq_flat, xs_flat = _preq(flat)
    eidx = torch.repeat_interleave(torch.arange(n_experts, device=flat.device),
                                   ends - starts)
    xq_all[eidx, within] = xq_flat[rt]
    xs_all[eidx, within] = xs_flat[rt]
    w_all[eidx, within] = rw
    eids = eids_sorted[starts]                        # [E]
    stride = (hidden // 256) * 74
    # gate/up
    gate, up = library.grouped_gate_up(
        xq_all, xs_all, gate_packed, up_packed, eids, grid, ksigns,
        rows=inter, cols=hidden, stride=stride, m_pad=eff_pad,
    )
    # SwiGLU with clamp
    up_c = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    gate_c = torch.clamp(gate, max=swiglu_limit)
    h = torch.nn.functional.silu(gate_c) * up_c        # [E, eff_pad, inter]
    hq, hs = _preq(h.reshape(-1, inter))
    hq = hq.reshape(n_experts, eff_pad, inter)
    hs = hs.reshape(n_experts, eff_pad, inter // 32)
    stride_d = (inter // 256) * 74
    # down: single matrix, use gate=up=down, take first output
    down, _ = library.grouped_gate_up(
        hq, hs, down_packed, down_packed, eids, grid, ksigns,
        rows=hidden, cols=inter, stride=stride_d, m_pad=eff_pad,
    )
    # scatter back to routes (vectorized) and reduce in stable expert-id order.
    inv = torch.argsort(order)
    contrib = (down * w_all.unsqueeze(-1))[eidx, within]        # [R, hidden]
    contrib = contrib[inv].reshape(M, top_k, hidden)
    order = torch.argsort(indices, dim=1, stable=True)
    contrib = contrib.gather(1, order.unsqueeze(-1).expand_as(contrib))
    y = contrib.sum(dim=1)
    return y


def _preq(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """int8 per-32 quantization of ``[..., K]`` (matches preq_activation)."""
    import torch

    shape = x.shape
    k = shape[-1]
    xr = x.reshape(*shape[:-1], k // 32, 32)
    scale = xr.abs().max(-1, keepdim=True).values / 127.0
    scale = torch.clamp(scale, min=1e-8)
    xq = (xr / scale).round().clamp(-128, 127).to(torch.int8)
    return xq.reshape_as(x), scale.reshape(*shape[:-1], k // 32)
