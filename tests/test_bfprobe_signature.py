"""Unit tests for bfprobe/signature.py -- CPU-only, no GPU involved.

Covers: ring overwrite-oldest behavior, monotonic seq / drop accounting,
time-ordered readback, JSON round-trip, the disabled-path zero-overhead
guarantee, and the enabled-path emit -> record wiring.
"""

from __future__ import annotations

import timeit

import pytest

torch = pytest.importorskip("torch")

import bfprobe.signature as signature_module  # noqa: E402
from bfprobe.signature import (  # noqa: E402
    HostSignatureRingBackend,
    ReadResult,
    SignatureRecord,
    SignatureRing,
    dump_json,
    emit,
    load_json,
)


class TestSignatureRingBasics:
    def test_capacity_must_be_positive(self):
        with pytest.raises(ValueError):
            SignatureRing(capacity=0)
        with pytest.raises(ValueError):
            SignatureRing(capacity=-1)

    def test_backend_capacity_must_match(self):
        backend = HostSignatureRingBackend(capacity=8)
        with pytest.raises(ValueError):
            SignatureRing(capacity=4, backend=backend)

    def test_empty_ring_reads_back_nothing(self):
        ring = SignatureRing(capacity=4)
        result = ring.read_all()
        assert result.records == ()
        assert result.dropped == 0
        assert ring.total_written == 0

    def test_single_write_read_round_trip(self):
        ring = SignatureRing(capacity=4)
        ring.record(200, 0, 5, absmax=1.5, l2=2.5, mean=0.1, nan_count=0, inf_count=0, numel=99)
        result = ring.read_all()
        assert result.dropped == 0
        assert len(result.records) == 1
        record = result.records[0]
        assert record.seq == 0
        assert record.site_id == 200
        assert record.round_idx == 0
        assert record.layer == 5
        assert record.absmax == pytest.approx(1.5)
        assert record.l2 == pytest.approx(2.5)
        assert record.mean == pytest.approx(0.1)
        assert record.nan_count == 0
        assert record.inf_count == 0
        assert record.numel == 99


class TestSignatureRingWrapAndDrop:
    def test_under_capacity_no_drops(self):
        ring = SignatureRing(capacity=8)
        for i in range(5):
            ring.record(200, i, 0, float(i), float(i), float(i), 0, 0, 10)
        result = ring.read_all()
        assert result.dropped == 0
        assert len(result.records) == 5
        assert [r.round_idx for r in result.records] == [0, 1, 2, 3, 4]

    def test_exactly_at_capacity_no_drops(self):
        ring = SignatureRing(capacity=4)
        for i in range(4):
            ring.record(200, i, 0, 1.0, 1.0, 1.0, 0, 0, 10)
        result = ring.read_all()
        assert result.dropped == 0
        assert len(result.records) == 4

    def test_overwrite_oldest_and_drop_count(self):
        ring = SignatureRing(capacity=4)
        for i in range(10):
            ring.record(200, i, 0, 1.0, 1.0, 1.0, 0, 0, 10)
        result = ring.read_all()
        # 10 writes into a capacity-4 ring: the first 6 are overwritten.
        assert result.dropped == 6
        assert len(result.records) == 4
        assert ring.total_written == 10

    def test_readback_order_is_time_order_after_wraparound(self):
        ring = SignatureRing(capacity=4)
        for i in range(10):
            ring.record(200, i, 0, 1.0, 1.0, 1.0, 0, 0, 10)
        result = ring.read_all()
        # Oldest-surviving-to-newest: rounds 6,7,8,9.
        assert [r.round_idx for r in result.records] == [6, 7, 8, 9]
        assert [r.seq for r in result.records] == [6, 7, 8, 9]

    def test_seq_is_monotonic_and_gapless_across_wraparound(self):
        ring = SignatureRing(capacity=3)
        for i in range(20):
            ring.record(200, i, 0, 1.0, 1.0, 1.0, 0, 0, 10)
        result = ring.read_all()
        seqs = [r.seq for r in result.records]
        assert seqs == sorted(seqs)
        assert seqs[-1] - seqs[0] == len(seqs) - 1  # no gaps among survivors
        assert ring.total_written == 20
        assert result.dropped == 20 - 3

    def test_distinct_sites_and_layers_all_survive_within_capacity(self):
        ring = SignatureRing(capacity=200)
        # One full round: 48 layers x 4 taps.
        for layer in range(48):
            for site_id in (200, 201, 202, 203):
                ring.record(site_id, 0, layer, 1.0, 1.0, 1.0, 0, 0, 10)
        result = ring.read_all()
        assert result.dropped == 0
        assert len(result.records) == 192
        seen = {(r.site_id, r.layer) for r in result.records}
        assert len(seen) == 192


class TestSignatureRecordConversion:
    def test_as_signature_drops_ring_bookkeeping_fields(self):
        record = SignatureRecord(
            seq=7,
            site_id=201,
            round_idx=3,
            layer=10,
            absmax=1.0,
            l2=2.0,
            mean=0.5,
            nan_count=0,
            inf_count=0,
            numel=42,
        )
        signature = record.as_signature()
        assert signature.absmax == 1.0
        assert signature.l2 == 2.0
        assert signature.mean == 0.5
        assert signature.nan_count == 0
        assert signature.inf_count == 0
        assert signature.numel == 42


class TestJsonRoundTrip:
    def test_dump_and_load_json(self, tmp_path):
        ring = SignatureRing(capacity=4)
        for i in range(6):
            ring.record(200 + (i % 2), i, i % 3, float(i), float(i), 0.1, 0, 0, 10)
        result = ring.read_all()

        path = tmp_path / "dump.json"
        dump_json(result, path)
        loaded = load_json(path)

        assert loaded.dropped == result.dropped
        assert loaded.records == result.records

    def test_load_json_returns_read_result(self, tmp_path):
        empty = ReadResult(records=(), dropped=0)
        path = tmp_path / "empty.json"
        dump_json(empty, path)
        loaded = load_json(path)
        assert loaded.records == ()
        assert loaded.dropped == 0


class TestZeroOverheadWhenDisabled:
    def test_probe_enabled_defaults_false(self):
        # Import-time default from the local bus stub (bfprobe.bus does not
        # exist yet in this worktree) -- see bfprobe/_bus_stub.py.
        assert signature_module.PROBE_ENABLED is False

    def test_emit_is_a_no_op_when_disabled(self, monkeypatch):
        monkeypatch.setattr(signature_module, "PROBE_ENABLED", False)
        ring = SignatureRing(capacity=4)
        emit(ring, 200, 0, 0, object())  # would blow up in reduce_reference if not short-circuited
        assert ring.total_written == 0

    def test_emit_records_when_enabled(self, monkeypatch):
        monkeypatch.setattr(signature_module, "PROBE_ENABLED", True)
        ring = SignatureRing(capacity=4)
        tensor = torch.tensor([1.0, 2.0, 3.0])
        emit(ring, 201, 5, 3, tensor)
        result = ring.read_all()
        assert len(result.records) == 1
        record = result.records[0]
        assert record.site_id == 201
        assert record.round_idx == 5
        assert record.layer == 3
        assert record.numel == 3

    def test_disabled_call_is_under_100ns(self, monkeypatch):
        monkeypatch.setattr(signature_module, "PROBE_ENABLED", False)
        ring = SignatureRing(capacity=4)
        sentinel = object()

        def call() -> None:
            emit(ring, 200, 0, 0, sentinel)

        number = 200_000
        elapsed = timeit.timeit(call, number=number)
        per_call_ns = (elapsed / number) * 1e9
        assert per_call_ns < 100, f"disabled emit() took {per_call_ns:.1f} ns/call, expected < 100"
        assert ring.total_written == 0
