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
_SINGLE_LIBRARY_PATH = _GENERATED_DIR / "iq2_mma16_tc_single.so"


class IQ2MMA16TCError(RuntimeError):
    """The native IQ2 MMA16 TC artifact cannot satisfy its fixed contract."""


@dataclass(frozen=True)
class IQ2MMA16TCManifest:
    abi_version: int
    target_sm: str
    library_sha256: str
    source_sha256: str | None = None

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> IQ2MMA16TCManifest:
        try:
            return cls(
                abi_version=int(value["abi_version"]),
                target_sm=str(value["target_sm"]),
                library_sha256=str(value["library_sha256"]),
                source_sha256=str(value.get("source_sha256")),
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
        _source_path = _KERNEL_DIR / "iq2_mma16_tc.cu"
        source_sha = getattr(manifest, "source_sha256", None)
        if source_sha is None:
            raise IQ2MMA16TCError("IQ2 MMA16 TC manifest is missing source_sha256")
        if _source_path.is_file() and _sha256(_source_path) != source_sha:
            raise IQ2MMA16TCError(
                "IQ2 MMA16 TC source changed after artifact build; rebuild with "
                "`make build-iq2-mma16-tc` (stale artifact guard)"
            )
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

    def single_down(
        self,
        xq,
        xs,
        packed,
        eids,
        grid,
        ksigns,
        *,
        rows: int,
        cols: int,
        stride: int,
        m_pad: int,
    ):
        """Single-output down ``x @ W^T`` for one packed matrix.

        Uses the dedicated single-output artifact (``iq2_mma16_tc_single.cu``),
        avoiding the gate/up dual kernel's wasted second output.  Returns
        ``[E, m_pad, rows]``.
        """
        import torch

        if not _SINGLE_LIBRARY_PATH.is_file():
            raise IQ2MMA16TCError(
                f"IQ2 MMA16 TC single library is missing: {_SINGLE_LIBRARY_PATH}"
            )
        if xq.device.type != "cuda":
            raise IQ2MMA16TCError("iq2_mma16_tc single requires CUDA tensors")
        E = int(eids.numel())
        for name, t in [("xq", xq), ("xs", xs), ("packed", packed),
                        ("eids", eids), ("grid", grid), ("ksigns", ksigns)]:
            if not t.is_contiguous():
                raise IQ2MMA16TCError(f"{name} must be contiguous")
        library = ctypes.CDLL(str(_SINGLE_LIBRARY_PATH))
        launch = library.iq2_mma16_tc_launch_single
        launch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        launch.restype = None
        out = torch.empty((E, m_pad, rows), dtype=torch.float32, device=xq.device)
        launch(
            ctypes.c_void_p(xq.data_ptr()),
            ctypes.c_void_p(xs.data_ptr()),
            ctypes.c_void_p(packed.data_ptr()),
            ctypes.c_void_p(eids.data_ptr()),
            ctypes.c_void_p(grid.data_ptr()),
            ctypes.c_void_p(ksigns.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            E, rows, cols, stride, m_pad,
        )
        return out


def make_manifest(library_path: Path = _LIBRARY_PATH, source_path: Path | None = None) -> dict:
    payload = {
        "abi_version": ABI_VERSION,
        "target_sm": TARGET_SM,
        "library_sha256": _sha256(library_path),
    }
    if source_path is not None:
        payload["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return payload


def grouped_moe_prefill_k32(
    flat,
    weights,
    indices,
    gate_packed,
    up_packed,
    down_packed,
    grid,
    ksigns,
    *,
    inter: int,
    hidden: int,
    swiglu_limit: float,
    bucket: int = 32,
    library: NativeIQ2MMA16TCLibrary | None = None,
):
    """Complete K32 grouped MoE with split batches (quality-first).

    Like ``iq2_mma16.grouped_moe_prefill`` but uses the K32 scale-amortized
    kernel (routed cos ~0.9999) with single-output down, and fixes eff_pad at
    ``bucket`` (default 32).  Experts with more than ``bucket`` routes are
    split: the first ``bucket`` routes go in the full batch, the remainder in a
    small second batch.  Returns ``[M, hidden]`` routed output (shared expert
    not included; caller adds it).
    """
    import torch

    if flat.device.type != "cuda":
        raise IQ2MMA16TCError("grouped_moe_prefill_k32 requires CUDA")
    if library is None:
        library = NativeIQ2MMA16TCLibrary.load()
    M, top_k = indices.shape
    routes = indices.reshape(-1)
    rweights = weights.reshape(-1)

    order = torch.argsort(routes, stable=True)
    eids_sorted = routes[order]
    change = eids_sorted[1:] != eids_sorted[:-1]
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=flat.device),
                        torch.nonzero(change, as_tuple=False).squeeze(-1) + 1])
    n_experts = starts.numel()
    ends = torch.cat([starts[1:], torch.tensor([routes.numel()], dtype=torch.long,
                                               device=flat.device)])
    counts = ends - starts
    rt = torch.arange(M, device=flat.device).repeat_interleave(top_k)[order]
    rw = rweights[order]
    within = torch.arange(routes.numel(), device=flat.device) - torch.repeat_interleave(
        starts, ends - starts)
    from .iq2_mma16 import _preq
    xq_flat, xs_flat = _preq(flat)
    eids = eids_sorted[starts]
    stride_g = (hidden // 256) * 74
    stride_d = (inter // 256) * 74

    contrib = torch.zeros(routes.numel(), hidden, device=flat.device)

    # batch 1: first `bucket` routes of every expert
    b1_mask = within < bucket
    gidx1 = torch.nonzero(b1_mask).squeeze(-1)
    seg1 = torch.minimum(counts, torch.tensor(bucket, device=flat.device))
    n1 = int(seg1.sum().item())
    e1 = torch.repeat_interleave(torch.arange(n_experts, device=flat.device), seg1)
    within1 = torch.arange(n1, device=flat.device) - torch.repeat_interleave(
        torch.cat([torch.zeros(1, dtype=torch.long, device=flat.device),
                   torch.cumsum(seg1, 0)[:-1]]), seg1)
    xq = torch.zeros(n_experts, bucket, hidden, dtype=torch.int8, device=flat.device)
    xs = torch.zeros(n_experts, bucket, hidden // 32, dtype=torch.float32, device=flat.device)
    w = torch.zeros(n_experts, bucket, dtype=torch.float32, device=flat.device)
    xq[e1, within1] = xq_flat[rt[gidx1]]
    xs[e1, within1] = xs_flat[rt[gidx1]]
    w[e1, within1] = rw[gidx1]
    gate, up = library.grouped_gate_up(
        xq, xs, gate_packed, up_packed, eids, grid, ksigns,
        rows=inter, cols=hidden, stride=stride_g, m_pad=bucket,
    )
    h = torch.nn.functional.silu(torch.clamp(gate, max=swiglu_limit)) * \
        torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    hq, hs = _preq(h.reshape(-1, inter))
    hq = hq.reshape(n_experts, bucket, inter)
    hs = hs.reshape(n_experts, bucket, inter // 32)
    down = library.single_down(
        hq, hs, down_packed, eids, grid, ksigns,
        rows=hidden, cols=inter, stride=stride_d, m_pad=bucket,
    )
    contrib[gidx1] = (down * w.unsqueeze(-1))[e1, within1]

    # batch 2: remainder of over-bucket experts (up to 16 routes each)
    over = counts > bucket
    n_over = int(over.sum().item())
    if n_over:
        over_eids = eids[over]
        b2_mask = (within >= bucket) & torch.repeat_interleave(over, counts)
        gidx2 = torch.nonzero(b2_mask).squeeze(-1)
        over_exp_idx = torch.nonzero(over).squeeze(-1)   # expert ids with >bucket
        g2l = torch.full((n_experts,), -1, dtype=torch.long, device=flat.device)
        g2l[over_exp_idx] = torch.arange(n_over, device=flat.device)
        e2_global = torch.repeat_interleave(torch.arange(n_experts, device=flat.device),
                                            counts)[gidx2]
        e2 = g2l[e2_global]
        b2_bucket = 16
        within2 = within[gidx2] - bucket
        xq = torch.zeros(n_over, b2_bucket, hidden, dtype=torch.int8, device=flat.device)
        xs = torch.zeros(n_over, b2_bucket, hidden // 32, dtype=torch.float32, device=flat.device)
        w = torch.zeros(n_over, b2_bucket, dtype=torch.float32, device=flat.device)
        xq[e2, within2] = xq_flat[rt[gidx2]]
        xs[e2, within2] = xs_flat[rt[gidx2]]
        w[e2, within2] = rw[gidx2]
        gate, up = library.grouped_gate_up(
            xq, xs, gate_packed, up_packed, over_eids, grid, ksigns,
            rows=inter, cols=hidden, stride=stride_g, m_pad=b2_bucket,
        )
        h = torch.nn.functional.silu(torch.clamp(gate, max=swiglu_limit)) * \
            torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        hq, hs = _preq(h.reshape(-1, inter))
        hq = hq.reshape(n_over, b2_bucket, inter)
        hs = hs.reshape(n_over, b2_bucket, inter // 32)
        down = library.single_down(
            hq, hs, down_packed, over_eids, grid, ksigns,
            rows=hidden, cols=inter, stride=stride_d, m_pad=b2_bucket,
        )
        contrib[gidx2] = (down * w.unsqueeze(-1))[e2, within2]

    # unsort to original route order, reshape, reduce in stable expert-id order
    inv = torch.argsort(order)
    c = contrib[inv].reshape(M, top_k, hidden)
    final_order = torch.argsort(indices, dim=1, stable=True)
    c = c.gather(1, final_order.unsqueeze(-1).expand_as(c))
    return c.sum(dim=1)
