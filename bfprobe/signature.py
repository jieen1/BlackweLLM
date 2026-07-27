"""T1 signature ring: a fixed-capacity, overwrite-oldest ring of per-tensor
fingerprints (see ``bfprobe.reduce.Signature``).

One DFlash round produces 48 layers x 4 taps = 192 signatures (see
notes/2026-07-27-probe-system-design-and-plan.md section 4). The ring stores
each as a flat row of scalars -- ``(seq, site_id, round_idx, layer, absmax,
l2, mean, nan_count, inf_count, numel)`` -- in preallocated, columnar
(structure-of-arrays) storage so the hot-path write is index math plus
in-place scalar assignment: no per-call Python object allocation, no
f-strings, no dict construction.

Storage backend is pluggable (``SignatureRingBackend``): today's
``HostSignatureRingBackend`` is plain host memory (numpy arrays), matching
the task's "host-side implementation is fine for now, but leave the
interface ready" instruction. A future GPU-resident backend (P3's staging
buffer + ring, per the design doc's section 3/section 5) implements the same
``write``/``read`` methods; ``SignatureRing`` itself would not need to
change.

Overwrite + drop accounting: the ring never blocks and never grows. Once
``total_written`` exceeds ``capacity``, old rows are silently overwritten --
but "silently" only in the sense that the *storage* doesn't complain. Every
row carries its own monotonic ``seq`` (the value of ``total_written`` at the
time it was written), so ``read_all()`` can always compute exactly how many
rows were dropped (``max(0, total_written - capacity)``) and report it
explicitly via ``ReadResult.dropped`` -- this package's hard requirement is
that data loss is always visible, never silent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from bfprobe.reduce import Signature, reduce_reference

try:
    from bfprobe.bus import PROBE_ENABLED, TIER_SIGNATURE
except ImportError:
    from bfprobe._bus_stub import PROBE_ENABLED, TIER_SIGNATURE

#: This package's probe-site id band (see notes/2026-07-27-bfprobe-t1-
#: signatures.md's "site_id 分配规则" section): 200-299, non-overlapping
#: with the MoE-routing agent's 300-399. Only a handful of ids are actually
#: used -- see ``bfprobe.scan``'s ``SITE_*`` constants -- because ``site_id``
#: identifies the *tap kind* (input_layernorm/self_attn/...), not
#: ``(layer, kind)``; the layer index already has its own column
#: (``SignatureRecord.layer``), so it does not need to be folded into
#: ``site_id`` at all.
SITE_ID_BAND = range(200, 300)

#: 200,000 rows * ~72 bytes/row (9 int64/float64 columns) ~= 14.4 MB --
#: negligible next to T2's 256 MiB default budget (section 4/6 of the design
#: doc), and at 192 records/round holds >1000 rounds of history.
DEFAULT_CAPACITY = 200_000

_TIER_SIGNATURE_TAG = TIER_SIGNATURE  # referenced for self-documentation only


@dataclass(frozen=True)
class SignatureRecord:
    """One ring row, read back in time order by ``SignatureRing.read_all``."""

    seq: int
    site_id: int
    round_idx: int
    layer: int
    absmax: float
    l2: float
    mean: float
    nan_count: int
    inf_count: int
    numel: int

    def as_signature(self) -> Signature:
        """Drop the ring bookkeeping fields, keep just the fingerprint."""
        return Signature(
            absmax=self.absmax,
            l2=self.l2,
            mean=self.mean,
            nan_count=self.nan_count,
            inf_count=self.inf_count,
            numel=self.numel,
        )


@dataclass(frozen=True)
class ReadResult:
    """A full, time-ordered readback of the ring, plus the drop count."""

    records: tuple[SignatureRecord, ...]
    dropped: int


class SignatureRingBackend(Protocol):
    """Storage backend contract. ``SignatureRing`` only ever calls
    ``write``/``read``/``capacity`` -- everything else (ring math, drop
    accounting, ordering) lives in ``SignatureRing`` itself, so a backend
    implementation is just "N preallocated columns with random access"."""

    capacity: int

    def write(
        self,
        idx: int,
        *,
        seq: int,
        site_id: int,
        round_idx: int,
        layer: int,
        absmax: float,
        l2: float,
        mean: float,
        nan_count: int,
        inf_count: int,
        numel: int,
    ) -> None:
        """In-place write of one row at physical slot ``idx``. Must not
        allocate (this is the hot path)."""
        ...

    def read(
        self, idx: int
    ) -> tuple[int, int, int, int, float, float, float, int, int, int]:
        """Read back row ``idx`` as ``(seq, site_id, round_idx, layer,
        absmax, l2, mean, nan_count, inf_count, numel)``. Off hot path."""
        ...


class HostSignatureRingBackend:
    """Preallocated, columnar host-memory backend (numpy arrays).

    This is today's only backend -- the "host-side implementation" the task
    asks for. A GPU-resident backend (P3) would hold the same nine columns
    as device tensors and implement the same two methods; nothing in
    ``SignatureRing`` assumes host memory.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        self._seq = np.zeros(capacity, dtype=np.int64)
        self._site_id = np.zeros(capacity, dtype=np.int32)
        self._round_idx = np.zeros(capacity, dtype=np.int64)
        self._layer = np.zeros(capacity, dtype=np.int32)
        self._absmax = np.zeros(capacity, dtype=np.float64)
        self._l2 = np.zeros(capacity, dtype=np.float64)
        self._mean = np.zeros(capacity, dtype=np.float64)
        self._nan_count = np.zeros(capacity, dtype=np.int64)
        self._inf_count = np.zeros(capacity, dtype=np.int64)
        self._numel = np.zeros(capacity, dtype=np.int64)

    def write(
        self,
        idx: int,
        *,
        seq: int,
        site_id: int,
        round_idx: int,
        layer: int,
        absmax: float,
        l2: float,
        mean: float,
        nan_count: int,
        inf_count: int,
        numel: int,
    ) -> None:
        # Each of these is an in-place scalar store into a preallocated
        # array -- no new numpy object is created.
        self._seq[idx] = seq
        self._site_id[idx] = site_id
        self._round_idx[idx] = round_idx
        self._layer[idx] = layer
        self._absmax[idx] = absmax
        self._l2[idx] = l2
        self._mean[idx] = mean
        self._nan_count[idx] = nan_count
        self._inf_count[idx] = inf_count
        self._numel[idx] = numel

    def read(
        self, idx: int
    ) -> tuple[int, int, int, int, float, float, float, int, int, int]:
        return (
            int(self._seq[idx]),
            int(self._site_id[idx]),
            int(self._round_idx[idx]),
            int(self._layer[idx]),
            float(self._absmax[idx]),
            float(self._l2[idx]),
            float(self._mean[idx]),
            int(self._nan_count[idx]),
            int(self._inf_count[idx]),
            int(self._numel[idx]),
        )


class SignatureRing:
    """Fixed-capacity, overwrite-oldest ring of ``SignatureRecord`` rows."""

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        backend: SignatureRingBackend | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._backend: SignatureRingBackend = (
            backend if backend is not None else HostSignatureRingBackend(capacity)
        )
        if self._backend.capacity != capacity:
            raise ValueError(
                f"backend capacity ({self._backend.capacity}) does not match "
                f"requested capacity ({capacity})"
            )
        self._capacity = capacity
        self._next_seq = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_written(self) -> int:
        """Monotonic count of every ``record`` call ever made, including
        ones that have since been overwritten."""
        return self._next_seq

    def record(
        self,
        site_id: int,
        round_idx: int,
        layer: int,
        absmax: float,
        l2: float,
        mean: float,
        nan_count: int,
        inf_count: int,
        numel: int,
    ) -> None:
        """Hot-path write: one row, wrapping at ``capacity``. Zero
        allocation -- takes raw scalar fields rather than a ``Signature``
        object specifically to avoid constructing one per call."""
        seq = self._next_seq
        idx = seq % self._capacity
        self._backend.write(
            idx,
            seq=seq,
            site_id=site_id,
            round_idx=round_idx,
            layer=layer,
            absmax=absmax,
            l2=l2,
            mean=mean,
            nan_count=nan_count,
            inf_count=inf_count,
            numel=numel,
        )
        self._next_seq = seq + 1

    def read_all(self) -> ReadResult:
        """Off-hot-path readback, oldest-to-newest, with an explicit drop
        count. Never silently drops data: if ``total_written > capacity``,
        the overwritten rows are counted in ``ReadResult.dropped`` rather
        than pretending they never existed."""
        total = self._next_seq
        valid = min(total, self._capacity)
        dropped = max(0, total - self._capacity)
        start_seq = total - valid
        records = []
        for i in range(valid):
            seq = start_seq + i
            idx = seq % self._capacity
            row = self._backend.read(idx)
            records.append(SignatureRecord(*row))
        return ReadResult(records=tuple(records), dropped=dropped)


def dump_json(result: ReadResult, path: Path) -> None:
    """Serialize a ring readback to JSON -- the on-disk format
    ``bfprobe.scan_cli``'s ``bf probe scan`` reads. Off hot path: this is a
    dump operation, called well after the rounds it describes have already
    finished (see ``bfprobe.signature``'s module docstring on the drain
    thread this would eventually run on)."""
    payload = {
        "dropped": result.dropped,
        "records": [
            {
                "seq": record.seq,
                "site_id": record.site_id,
                "round_idx": record.round_idx,
                "layer": record.layer,
                "absmax": record.absmax,
                "l2": record.l2,
                "mean": record.mean,
                "nan_count": record.nan_count,
                "inf_count": record.inf_count,
                "numel": record.numel,
            }
            for record in result.records
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> ReadResult:
    """Read back a ring dump written by ``dump_json``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = tuple(SignatureRecord(**row) for row in payload["records"])
    return ReadResult(records=records, dropped=int(payload["dropped"]))


def emit(
    ring: SignatureRing, site_id: int, round_idx: int, layer: int, tensor: object
) -> None:
    """Convenience call-site wrapper: single ``PROBE_ENABLED`` check, then
    reduce + record.

    This is what a forward hook would call in the always-on-in-production
    design (section 5 of the design doc): when disabled, the cost is exactly
    one module-attribute load and a branch -- ``tensor`` is never touched,
    ``reduce_reference`` is never called. See
    tests/test_bfprobe_signature.py's zero-overhead test, which times this
    exact function with probing disabled.

    Uses ``reduce_reference`` (CPU-only) rather than ``bfprobe.reduce
    .reduce_gpu`` because this repo's current environment has no GPU access
    to exercise the Triton path; production wiring (P1/P3) would call
    ``reduce_gpu`` + stage the raw accumulators into a GPU-resident ring
    slot instead of calling this function as-is. See
    notes/2026-07-27-bfprobe-t1-signatures.md.
    """
    if not PROBE_ENABLED:
        return
    signature = reduce_reference(tensor)
    ring.record(
        site_id,
        round_idx,
        layer,
        signature.absmax,
        signature.l2,
        signature.mean,
        signature.nan_count,
        signature.inf_count,
        signature.numel,
    )


if __name__ == "__main__":
    demo_ring = SignatureRing(capacity=4)
    for round_idx in range(6):
        demo_ring.record(200, round_idx, 0, 1.0, 1.0, 0.5, 0, 0, 100)
    result = demo_ring.read_all()
    print("dropped:", result.dropped)
    print("records:", result.records)
