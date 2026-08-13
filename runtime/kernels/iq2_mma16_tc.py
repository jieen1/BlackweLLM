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

import torch

from runtime.kernels.dsv4_grouping import (
    device_group_counts_into,
)
from runtime.kernels.iq2_mma16 import _preq_into

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
        self._single_lib: ctypes.CDLL | None = None
        self._launch = library.iq2_mma16_tc_launch
        self._launch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._launch.restype = None

    def _single_library(self) -> ctypes.CDLL:
        if self._single_lib is None:
            self._single_lib = ctypes.CDLL(str(_SINGLE_LIBRARY_PATH))
        return self._single_lib

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
        self.grouped_gate_up_into(xq, xs, packed_gate, packed_up, eids, grid, ksigns,
                                  out_gate, out_up, rows=rows, cols=cols, stride=stride,
                                  m_pad=m_pad)
        return out_gate, out_up

    def grouped_gate_up_into(
        self,
        xq,
        xs,
        packed_gate,
        packed_up,
        eids,
        grid,
        ksigns,
        out_gate,
        out_up,
        *,
        rows: int,
        cols: int,
        stride: int,
        m_pad: int,
    ):
        """Like ``grouped_gate_up`` but writes into caller-owned buffers.

        No internal allocation, so it is safe inside a CUDA graph capture.
        """

        if xq.device.type != "cuda":
            raise IQ2MMA16TCError("iq2_mma16_tc requires CUDA tensors")
        E = int(eids.numel())
        for name, t in [("xq", xq), ("xs", xs), ("packed_gate", packed_gate),
                        ("packed_up", packed_up), ("eids", eids), ("grid", grid),
                        ("ksigns", ksigns), ("out_gate", out_gate), ("out_up", out_up)]:
            if not t.is_contiguous():
                raise IQ2MMA16TCError(f"{name} must be contiguous")
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
            ctypes.c_void_p(torch.cuda.current_stream(xq.device).cuda_stream),
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
        self.single_down_into(xq, xs, packed, eids, grid, ksigns, out,
                              rows=rows, cols=cols, stride=stride, m_pad=m_pad)
        return out

    def single_down_into(
        self,
        xq,
        xs,
        packed,
        eids,
        grid,
        ksigns,
        out,
        *,
        rows: int,
        cols: int,
        stride: int,
        m_pad: int,
    ):
        """Like ``single_down`` but writes into a caller-owned buffer (graph-safe)."""

        if not _SINGLE_LIBRARY_PATH.is_file():
            raise IQ2MMA16TCError(
                f"IQ2 MMA16 TC single library is missing: {_SINGLE_LIBRARY_PATH}"
            )
        if xq.device.type != "cuda":
            raise IQ2MMA16TCError("iq2_mma16_tc single requires CUDA tensors")
        E = int(eids.numel())
        for name, t in [("xq", xq), ("xs", xs), ("packed", packed),
                        ("eids", eids), ("grid", grid), ("ksigns", ksigns), ("out", out)]:
            if not t.is_contiguous():
                raise IQ2MMA16TCError(f"{name} must be contiguous")
        library = ctypes.CDLL(str(_SINGLE_LIBRARY_PATH))
        launch = library.iq2_mma16_tc_launch_single
        launch.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_void_p,
        ]
        launch.restype = None
        launch(
            ctypes.c_void_p(xq.data_ptr()),
            ctypes.c_void_p(xs.data_ptr()),
            ctypes.c_void_p(packed.data_ptr()),
            ctypes.c_void_p(eids.data_ptr()),
            ctypes.c_void_p(grid.data_ptr()),
            ctypes.c_void_p(ksigns.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            E, rows, cols, stride, m_pad,
            ctypes.c_void_p(torch.cuda.current_stream(xq.device).cuda_stream),
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
    xs[e1, within1] = xs_flat[rt[gidx1]].to(torch.float32)
    w[e1, within1] = rw[gidx1]
    gate = torch.empty(n_experts, bucket, inter, dtype=torch.float32, device=flat.device)
    up = torch.empty_like(gate)
    library.grouped_gate_up_into(
        xq, xs, gate_packed, up_packed, eids, grid, ksigns, gate, up,
        rows=inter, cols=hidden, stride=stride_g, m_pad=bucket,
    )
    h = torch.nn.functional.silu(torch.clamp(gate, max=swiglu_limit)) * \
        torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
    hq, hs = _preq(h.reshape(-1, inter))
    hq = hq.reshape(n_experts, bucket, inter)
    hs = hs.reshape(n_experts, bucket, inter // 32)
    down = torch.empty(n_experts, bucket, hidden, dtype=torch.float32, device=flat.device)
    library.single_down_into(
        hq, hs, down_packed, eids, grid, ksigns, down,
        rows=hidden, cols=inter, stride=stride_d, m_pad=bucket,
    )
    contrib[gidx1] = (down * w.unsqueeze(-1))[e1, within1]

    # batch 2: remainder of over-bucket experts.  ``b2_bucket`` must cover
    # the FULL over-run (up to the largest route count in the chunk), not a
    # fixed 16 -- a 64-token chunk can route hundreds of slots to one expert
    # (e.g. hash layers, or a dominant token), and truncating the batch
    # would both OOB the scatter below and drop routes.
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
        within2 = within[gidx2] - bucket
        # The over-run can exceed the kernel's M_PAD cap for a dominant
        # token, but DSV4's real 64-token route distribution tops out around
        # 21 over-routes (max route ~53, bucket=32).  Use a single batch2
        # kernel sized to the actual over-run (clamped to the kernel's 48-row
        # support) so the common case is ONE extra kernel per over-expert,
        # not a per-48-row segment loop.
        B2_MAX = 48
        over_run = int(within2.max().item()) + 1 if within2.numel() else 0
        for lo in range(0, over_run, B2_MAX):
            hi = min(lo + B2_MAX, over_run)
            seg_mask = (within2 >= lo) & (within2 < hi)
            gs = gidx2[seg_mask]
            if not gs.numel():
                continue
            # The kernel only instantiates M_PAD in {16, 32, 48, 64}; a raw
            # segment width like 28 falls through to the <64> template and
            # reads/writes past the real [n_over, 28, ...] buffer.  Pad the
            # segment bucket UP to the next supported width (<=64).
            seg_bucket = min(64, ((hi - lo) + 15) // 16 * 16)
            e2s = e2[seg_mask]
            within2s = within2[seg_mask] - lo
            xq = torch.zeros(n_over, seg_bucket, hidden, dtype=torch.int8, device=flat.device)
            xs = torch.zeros(
                n_over, seg_bucket, hidden // 32, dtype=torch.float32, device=flat.device
            )
            w = torch.zeros(n_over, seg_bucket, dtype=torch.float32, device=flat.device)
            xq[e2s, within2s] = xq_flat[rt[gs]]
            xs[e2s, within2s] = xs_flat[rt[gs]].to(torch.float32)
            w[e2s, within2s] = rw[gs]
            gate2 = torch.empty(n_over, seg_bucket, inter, dtype=torch.float32, device=flat.device)
            up2 = torch.empty_like(gate2)
            library.grouped_gate_up_into(
                xq, xs, gate_packed, up_packed, over_eids, grid, ksigns, gate2, up2,
                rows=inter, cols=hidden, stride=stride_g, m_pad=seg_bucket,
            )
            h = torch.nn.functional.silu(torch.clamp(gate2, max=swiglu_limit)) * \
                torch.clamp(up2, min=-swiglu_limit, max=swiglu_limit)
            hq, hs = _preq(h.reshape(-1, inter))
            hq = hq.reshape(n_over, seg_bucket, inter)
            hs = hs.reshape(n_over, seg_bucket, inter // 32)
            down = torch.empty(n_over, seg_bucket, hidden, dtype=torch.float32, device=flat.device)
            library.single_down_into(
                hq, hs, down_packed, over_eids, grid, ksigns, down,
                rows=hidden, cols=inter, stride=stride_d, m_pad=seg_bucket,
            )
            contrib[gs] = (down * w.unsqueeze(-1))[e2s, within2s]

    # unsort to original route order, reshape, reduce in stable expert-id order
    inv = torch.argsort(order)
    c = contrib[inv].reshape(M, top_k, hidden)
    final_order = torch.argsort(indices, dim=1, stable=True)
    c = c.gather(1, final_order.unsqueeze(-1).expand_as(c))
    return c.sum(dim=1)


# ---------------------------------------------------------------------------
# CUDA-graph-safe complete K32 MoE (Phase 1K service path)
# ---------------------------------------------------------------------------
#
# The eager ``grouped_moe_prefill_k32`` above spends ~2.4 s of CPU per
# 64-token chunk on Python glue (argsort/nonzero/repeat_interleave/scatter,
# measured 2026-08-12) against ~270 ms of GPU kernels -- CPU-bound.  This
# graph-safe sibling replaces every dynamic-shape Python op with fixed-shape
# device kernels and caller-owned buffers so the whole 43-layer MoE body can
# be captured in one CUDA graph and replayed with per-chunk input contents.
#
# ``Dsv4PrefillMoEWorkspace`` owns every intermediate buffer; the run function
# performs zero allocations and zero dynamic branching, so it is capturable.
# BUCKET=64 covers DSV4's real 64-token route distribution (measured max
# route ~53) in a SINGLE batch -- no over-bucket split in the common case --
# matching what the decode-side kernels already assume.
_GRAPH_BUCKET = 64


@dataclass
class Dsv4PrefillMoEWorkspace:
    """Caller-owned fixed-shape buffers for one graph-captured K32 MoE run.

    Shapes are pinned to a 64-token chunk: ``M`` rows, top-k 6, 256 experts,
    ``bucket`` 64.  ``flat``/``indices``/``weights`` contents change between
    replays; everything else is a reusable scratch area.
    """

    device: str
    hidden: int
    inter: int
    m: int = 64
    top_k: int = 6
    n_experts: int = 256
    bucket: int = _GRAPH_BUCKET

    def __post_init__(self) -> None:
        d = torch.device(self.device)
        self.flat = torch.empty(self.m, self.hidden, dtype=torch.bfloat16, device=d)
        self.indices = torch.empty(self.m, self.top_k, dtype=torch.int64, device=d)
        self.weights = torch.empty(self.m, self.top_k, dtype=torch.float32, device=d)
        r = self.m * self.top_k
        # quantized activations of the M input rows (flat, NOT route-expanded)
        self.xq_flat = torch.empty(self.m, self.hidden, dtype=torch.int8, device=d)
        self.xs_flat = torch.empty(self.m, self.hidden // 32, dtype=torch.float32, device=d)
        # route-expanded activations [R, hidden] gathered from xq_flat by rt
        self.xq_route = torch.empty(r, self.hidden, dtype=torch.int8, device=d)
        self.xs_route = torch.empty(r, self.hidden // 32, dtype=torch.float32, device=d)
        # SwiGLU output quantized (flat [E*BUCKET, inter])
        self.hq_flat = torch.empty(
            self.n_experts * self.bucket * self.inter, dtype=torch.int8, device=d
        )
        self.hs_flat = torch.empty(
            self.n_experts * self.bucket * (self.inter // 32), dtype=torch.float32, device=d
        )
        # device grouping (fixed shape)
        self.routes = torch.empty(r, dtype=torch.int32, device=d)
        self.rweights = torch.empty(r, dtype=torch.float32, device=d)
        self.counts = torch.empty(self.n_experts, dtype=torch.int32, device=d)
        self.within = torch.empty(r, dtype=torch.int32, device=d)
        # overflow-batch within: (within - bucket) for within >= bucket, the
        # second fixed batch's slot index (graph-safe; see batch2 split below).
        self.within2 = torch.empty(r, dtype=torch.int32, device=d)
        self.offsets = torch.empty(self.n_experts, dtype=torch.int32, device=d)
        # arange carrier for device_group_counts_into: must be >= R = m*top_k.
        # Sizing it to n_experts would silently resize during graph capture.
        self.cursor = torch.empty(r, dtype=torch.int32, device=d)
        self.rt = torch.empty(r, dtype=torch.int64, device=d)  # token index per route
        # batch tile (single BUCKET, covers real 64-token distribution)
        self.xq = torch.empty(
            self.n_experts, self.bucket, self.hidden, dtype=torch.int8, device=d
        )
        self.xs = torch.empty(
            self.n_experts, self.bucket, self.hidden // 32, dtype=torch.float32, device=d
        )
        self.w = torch.empty(self.n_experts, self.bucket, dtype=torch.float32, device=d)
        self.gate = torch.empty(
            self.n_experts, self.bucket, self.inter, dtype=torch.float32, device=d
        )
        self.up = torch.empty_like(self.gate)
        self.hq = torch.empty(
            self.n_experts, self.bucket, self.inter, dtype=torch.int8, device=d
        )
        self.hs = torch.empty(
            self.n_experts, self.bucket, self.inter // 32, dtype=torch.float32, device=d
        )
        self.down = torch.empty(
            self.n_experts, self.bucket, self.hidden, dtype=torch.float32, device=d
        )
        self.contrib = torch.empty(r, self.hidden, dtype=torch.float32, device=d)
        self.eids = torch.arange(self.n_experts, dtype=torch.int64, device=d)
        self.out = torch.empty(self.m, self.hidden, dtype=torch.float32, device=d)
        # flattened (expert, within) index for 1D tile scatter (graph-safe)
        self.flat_idx = torch.empty(r, dtype=torch.int64, device=d)
        # 0..R-1 route positions for combine (graph-safe)
        self.route_range = torch.arange(r, dtype=torch.int64, device=d)
        # stable expert-id order of each token's top-k slots (graph-safe);
        # the combine must sum in this order to be bit-exact with the eager
        # path (which gathers on argsort(indices) before summing -- the sum
        # order matters for fp32 accumulation, and a mismatch leaks ~1e-5
        # that the sparse-attention topk then amplifies into token drift).
        self.final_order = torch.empty(self.m, self.top_k, dtype=torch.int64, device=d)
        # ctypes-launched kernels read raw pointers, so torch knows no
        # dependency edge; a tiny scalar copy bridges each kernel to its
        # inputs so CUDA graph capture orders them correctly (same pattern as
        # the decode driver's input dependencies).
        self.dep_gate = torch.empty(1, dtype=torch.float32, device=d)
        self.dep_down = torch.empty(1, dtype=torch.float32, device=d)


def _preq_flat_into(x: torch.Tensor, xq_out: torch.Tensor, xs_out: torch.Tensor) -> None:
    """Quantize ``[R, K]`` into caller-owned flat buffers (graph-safe)."""
    _preq_into(x, xq_out, xs_out)


def grouped_moe_prefill_k32_graph(
    ws: Dsv4PrefillMoEWorkspace,
    flat: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    gate_packed: torch.Tensor,
    up_packed: torch.Tensor,
    down_packed: torch.Tensor,
    grid: torch.Tensor,
    ksigns: torch.Tensor,
    *,
    inter: int,
    hidden: int,
    swiglu_limit: float,
    library: NativeIQ2MMA16TCLibrary | None = None,
) -> torch.Tensor:
    """Graph-capturable complete K32 MoE for one fixed 64-token chunk.

    Zero allocations, zero dynamic shapes: device grouping replaces the
    Python argsort/nonzero glue, and the fixed ``bucket=64`` single batch
    covers DSV4's real route distribution.  ``ws`` owns every intermediate;
    ``flat``/``indices``/``weights`` are copied into it (contents-only, shape
    must match the workspace).  Returns ``[M, hidden]`` routed output.
    """
    if library is None:
        library = NativeIQ2MMA16TCLibrary.load()
    if flat.shape != (ws.m, hidden) or indices.shape != (ws.m, ws.top_k):
        raise ValueError(
            f"graph MoE expects flat {tuple((ws.m, hidden))} indices "
            f"{tuple((ws.m, ws.top_k))}, got {tuple(flat.shape)} {tuple(indices.shape)}"
        )
    # The caller pre-loads ws.flat/ws.indices/ws.weights BEFORE capture (and
    # mutates them between replays).  The graph body reads ONLY ws buffers so
    # a replay needs no parameter round-trip and new inputs take effect by
    # writing into the workspace.  ``flat``/``indices``/``weights`` are
    # accepted for shape validation only and are copied by the caller into
    # the workspace (they are captured as graph inputs, so the copy here
    # would otherwise pin the capture-time contents into every replay).
    ws.routes.copy_(ws.indices.reshape(-1).to(torch.int32))
    ws.rweights.copy_(ws.weights.reshape(-1))
    ws.rt.copy_(torch.arange(ws.m, device=ws.device).repeat_interleave(ws.top_k))

    # device grouping (fixed-shape atomics)
    device_group_counts_into(ws.routes, ws.counts, ws.within, ws.offsets, ws.cursor)

    # quantize activations of the M input rows, then expand to R routes
    _preq_flat_into(ws.flat, ws.xq_flat, ws.xs_flat)
    ws.xq_route.copy_(ws.xq_flat[ws.rt])
    ws.xs_route.copy_(ws.xs_flat[ws.rt])

    stride_g = (hidden // 256) * 74
    stride_d = (inter // 256) * 74
    # Two fixed batches cover up to 2*bucket routes per expert.  The eager
    # path splits exactly this way (bucket first, then an overflow batch) --
    # a single batch cannot cover the hash layers' route distribution (max
    # route ~51 at 128 tokens, ~315 at 2048), and clamping the overflow into
    # slot bucket-1 silently corrupts the output (measured cos 0.995 -> token
    # drift).  Batch 1 takes within < bucket; batch 2 takes the remainder with
    # within2 = within - bucket.  Each fill masks the OTHER batch's routes to
    # zero and uses accumulate so the masked routes add nothing.
    ws.contrib.zero_()
    bucket = ws.bucket
    b1 = ws.within < bucket  # [R] bool, fixed shape
    w1 = ws.within.clamp(max=bucket - 1)

    # ---- batch 1 ---------------------------------------------------------
    ws.xq.zero_()
    ws.xs.zero_()
    ws.w.zero_()
    ws.xq.index_put_(
        (ws.routes.long(), w1),
        torch.where(b1.unsqueeze(-1), ws.xq_route, torch.zeros_like(ws.xq_route)),
        accumulate=True,
    )
    ws.xs.index_put_(
        (ws.routes.long(), w1),
        torch.where(b1.unsqueeze(-1), ws.xs_route, torch.zeros_like(ws.xs_route)),
        accumulate=True,
    )
    ws.w.index_put_(
        (ws.routes.long(), w1),
        torch.where(b1, ws.rweights, torch.zeros_like(ws.rweights)),
        accumulate=True,
    )
    ws.dep_gate.copy_(ws.xq.sum().to(torch.float32) + ws.xs.sum())
    library.grouped_gate_up_into(
        ws.xq, ws.xs, gate_packed, up_packed, ws.eids, grid, ksigns,
        ws.gate, ws.up, rows=inter, cols=hidden, stride=stride_g, m_pad=bucket,
    )
    h = torch.nn.functional.silu(torch.clamp(ws.gate, max=swiglu_limit)) * torch.clamp(
        ws.up, min=-swiglu_limit, max=swiglu_limit
    )
    _preq_flat_into(h.reshape(-1, inter), ws.hq_flat, ws.hs_flat)
    ws.hq.view(ws.n_experts, bucket, inter).copy_(
        ws.hq_flat.view(ws.n_experts, bucket, inter)
    )
    ws.hs.view(ws.n_experts, bucket, inter // 32).copy_(
        ws.hs_flat.view(ws.n_experts, bucket, inter // 32)
    )
    ws.dep_down.copy_(ws.hq.sum().to(torch.float32) + ws.hs.sum())
    library.single_down_into(
        ws.hq, ws.hs, down_packed, ws.eids, grid, ksigns, ws.down,
        rows=hidden, cols=inter, stride=stride_d, m_pad=bucket,
    )
    ws.contrib.index_put_(
        (ws.route_range,),
        torch.where(
            b1.unsqueeze(-1),
            ws.down[ws.routes.long(), w1] * ws.rweights.unsqueeze(-1),
            torch.zeros((ws.m * ws.top_k, hidden), dtype=torch.float32, device=ws.down.device),
        ),
        accumulate=True,
    )

    # ---- batch 2 (overflow) ---------------------------------------------
    b2 = ws.within >= bucket
    ws.within2.copy_((ws.within - bucket).clamp(min=0, max=bucket - 1))
    ws.xq.zero_()
    ws.xs.zero_()
    ws.w.zero_()
    ws.xq.index_put_(
        (ws.routes.long(), ws.within2),
        torch.where(b2.unsqueeze(-1), ws.xq_route, torch.zeros_like(ws.xq_route)),
        accumulate=True,
    )
    ws.xs.index_put_(
        (ws.routes.long(), ws.within2),
        torch.where(b2.unsqueeze(-1), ws.xs_route, torch.zeros_like(ws.xs_route)),
        accumulate=True,
    )
    ws.w.index_put_(
        (ws.routes.long(), ws.within2),
        torch.where(b2, ws.rweights, torch.zeros_like(ws.rweights)),
        accumulate=True,
    )
    ws.dep_gate.copy_(ws.xq.sum().to(torch.float32) + ws.xs.sum())
    library.grouped_gate_up_into(
        ws.xq, ws.xs, gate_packed, up_packed, ws.eids, grid, ksigns,
        ws.gate, ws.up, rows=inter, cols=hidden, stride=stride_g, m_pad=bucket,
    )
    h = torch.nn.functional.silu(torch.clamp(ws.gate, max=swiglu_limit)) * torch.clamp(
        ws.up, min=-swiglu_limit, max=swiglu_limit
    )
    _preq_flat_into(h.reshape(-1, inter), ws.hq_flat, ws.hs_flat)
    ws.hq.view(ws.n_experts, bucket, inter).copy_(
        ws.hq_flat.view(ws.n_experts, bucket, inter)
    )
    ws.hs.view(ws.n_experts, bucket, inter // 32).copy_(
        ws.hs_flat.view(ws.n_experts, bucket, inter // 32)
    )
    ws.dep_down.copy_(ws.hq.sum().to(torch.float32) + ws.hs.sum())
    library.single_down_into(
        ws.hq, ws.hs, down_packed, ws.eids, grid, ksigns, ws.down,
        rows=hidden, cols=inter, stride=stride_d, m_pad=bucket,
    )
    ws.contrib.index_put_(
        (ws.route_range,),
        torch.where(
            b2.unsqueeze(-1),
            ws.down[ws.routes.long(), ws.within2] * ws.rweights.unsqueeze(-1),
            torch.zeros((ws.m * ws.top_k, hidden), dtype=torch.float32, device=ws.down.device),
        ),
        accumulate=True,
    )

    # combine: contrib[route r] = sum over its (expert, within) batches.
    # The eager path gathers on argsort(indices) BEFORE summing its top-k
    # contributions; sum order is not numerically commutative in fp32, so
    # reproduce the exact same order here or the ~1e-5 mismatch is amplified
    # into token drift by the sparse-attention topk across 43 layers.
    c = ws.contrib.reshape(ws.m, ws.top_k, hidden)
    ws.final_order.copy_(ws.indices.argsort(dim=1, stable=True))
    c = c.gather(1, ws.final_order.unsqueeze(-1).expand_as(c))
    ws.out.copy_(c.sum(dim=1))
    return ws.out


def grouped_moe_prefill_k32_dynamic(
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
    library: NativeIQ2MMA16TCLibrary | None = None,
):
    """Per-expert COMPACT K32 MoE (llama.cpp mm_ids_helper grouping).

    Grouping (one warp per expert), expert-sorted gather, and per-expert
    dynamic M (expert_bounds) GEMMs replace the eager argsort/nonzero/
    repeat_interleave/index_put Python glue and the fixed-bucket tile waste.
    Bit-exact with ``grouped_moe_prefill_k32``; ~1.6x faster (16.8 vs 27.3 ms
    at 1024 tokens).  Returns ``[M, hidden]`` routed output (no shared expert).
    """
    import torch

    if flat.device.type != "cuda":
        raise IQ2MMA16TCError("grouped_moe_prefill_k32_dynamic requires CUDA")
    if library is None:
        library = NativeIQ2MMA16TCLibrary.load()
    M, top_k = indices.shape
    E = 256
    stream = torch.cuda.current_stream(flat.device).cuda_stream
    indices32 = indices.reshape(-1).to(torch.int32)
    rweights = weights.reshape(-1)
    R = M * top_k

    compact_route = torch.zeros(R, dtype=torch.int32, device=flat.device)
    compact_iex = torch.zeros(R, dtype=torch.int32, device=flat.device)
    expert_bounds = torch.empty(E + 1, dtype=torch.int32, device=flat.device)
    library._library.moe_group_launch(
        ctypes.c_void_p(indices32.data_ptr()),
        ctypes.c_void_p(compact_route.data_ptr()),
        ctypes.c_void_p(compact_iex.data_ptr()),
        ctypes.c_void_p(expert_bounds.data_ptr()),
        M, top_k, E, ctypes.c_void_p(stream),
    )

    from .iq2_mma16 import _preq

    xq_flat, xs_flat = _preq(flat)
    compact_xq = torch.empty(R, hidden, dtype=torch.int8, device=flat.device)
    compact_xs = torch.empty(R, hidden // 32, dtype=torch.float32, device=flat.device)
    library._library.moe_gather_xq_launch(
        ctypes.c_void_p(xq_flat.data_ptr()),
        ctypes.c_void_p(xs_flat.data_ptr()),
        ctypes.c_void_p(compact_route.data_ptr()),
        ctypes.c_void_p(compact_xq.data_ptr()),
        ctypes.c_void_p(compact_xs.data_ptr()),
        R, hidden, ctypes.c_void_p(stream),
    )

    eids = torch.arange(E, dtype=torch.int64, device=flat.device)
    stride_g = (hidden // 256) * 74
    out_gate = torch.empty(R, inter, dtype=torch.float32, device=flat.device)
    out_up = torch.empty(R, inter, dtype=torch.float32, device=flat.device)
    library._library.iq2_mma16_tc_dynamic_launch(
        ctypes.c_void_p(compact_xq.data_ptr()),
        ctypes.c_void_p(compact_xs.data_ptr()),
        ctypes.c_void_p(gate_packed.data_ptr()),
        ctypes.c_void_p(up_packed.data_ptr()),
        ctypes.c_void_p(eids.data_ptr()),
        ctypes.c_void_p(grid.data_ptr()),
        ctypes.c_void_p(ksigns.data_ptr()),
        ctypes.c_void_p(expert_bounds.data_ptr()),
        ctypes.c_void_p(out_gate.data_ptr()),
        ctypes.c_void_p(out_up.data_ptr()),
        E, inter, hidden, stride_g, ctypes.c_void_p(stream),
    )

    h = torch.nn.functional.silu(torch.clamp(out_gate, max=swiglu_limit)) * torch.clamp(
        out_up, min=-swiglu_limit, max=swiglu_limit
    )
    hq, hs = _preq(h.reshape(-1, inter))
    hq = hq.reshape(R, inter)
    hs = hs.reshape(R, inter // 32)
    stride_d = (inter // 256) * 74
    down = torch.empty(R, hidden, dtype=torch.float32, device=flat.device)
    library._single_library().iq2_mma16_tc_launch_single_dynamic(
        ctypes.c_void_p(hq.data_ptr()),
        ctypes.c_void_p(hs.data_ptr()),
        ctypes.c_void_p(down_packed.data_ptr()),
        ctypes.c_void_p(eids.data_ptr()),
        ctypes.c_void_p(grid.data_ptr()),
        ctypes.c_void_p(ksigns.data_ptr()),
        ctypes.c_void_p(expert_bounds.data_ptr()),
        ctypes.c_void_p(down.data_ptr()),
        E, hidden, inter, stride_d, ctypes.c_void_p(stream),
    )

    # combine: scatter compact down back to route order, then sum in the eager
    # stable expert-id order (fp32 sum order is not commutative).
    contrib = torch.zeros(R, hidden, dtype=torch.float32, device=flat.device)
    route_pos = compact_route.long() * top_k + compact_iex.long()
    contrib[route_pos] = down * rweights[route_pos].unsqueeze(-1)
    c = contrib.reshape(M, top_k, hidden)
    final_order = torch.argsort(indices, dim=1, stable=True)
    c = c.gather(1, final_order.unsqueeze(-1).expand_as(c))
    return c.sum(dim=1)
