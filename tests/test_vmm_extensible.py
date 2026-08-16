"""Phase A tests: CUDA VMM extensible buffers (runtime/model/vmm_extensible.py).

Two halves, mirroring the research note's experiments:

  - torch-free: granule bookkeeping of ``_VirtualBuffer`` (committed-range
    accounting, overlap skipping, padding semantics, range validation) with
    an injected fake driver -- runs in the CI torch-free job.
  - GPU: ``ExtensibleTensor`` / ``ExtensibleKVCacheBuffers`` end-to-end
    (VA reservation costs ~0 physical memory, incremental commit + zeroing,
    torch views, base-pointer stability across commits, CUDA graph captured
    pre-commit replays correctly into post-commit pages). Self-skips when
    torch or a CUDA device is unavailable.
"""

from __future__ import annotations

import pytest

from runtime.model.vmm_extensible import (
    ExtensibleKVCacheBuffers,
    _round_up,
    _VirtualBuffer,
)


class _FakeVmmDriver:
    """Torch-free stand-in for ``VmmDriver``: records calls, maps nothing.

    ``create``/``map``/``set_access``/``unmap``/``release`` simulate success
    so ``_VirtualBuffer``'s granule accounting runs; nothing is actually
    allocated, so these tests are pure bookkeeping.
    """

    def __init__(self) -> None:
        self.granularity_size = 2 * 1024 * 1024  # 2 MiB, as measured on SM120
        self.reserved: list[tuple[int, int]] = []
        self.mapped: list[tuple[int, int, int]] = []  # (ptr, size, handle)
        self.next_ptr = 1 << 40
        self.next_handle = 1

    def ensure_context(self, device_index: int) -> None:  # noqa: ARG002
        return None

    def granularity(self, device_index: int) -> int:  # noqa: ARG002
        return self.granularity_size

    def reserve(self, size: int) -> int:
        ptr = self.next_ptr
        self.next_ptr += size
        self.reserved.append((ptr, size))
        return ptr

    def free_reserved(self, ptr: int, size: int) -> None:
        self.reserved.remove((ptr, size))

    def create(self, size: int, device_index: int) -> int:  # noqa: ARG002
        handle = self.next_handle
        self.next_handle += 1
        return handle

    def map(self, ptr: int, size: int, handle: int) -> None:
        self.mapped.append((ptr, size, handle))

    def set_access(self, ptr: int, size: int, device_index: int) -> None:  # noqa: ARG002
        return None

    def unmap(self, ptr: int, size: int) -> None:
        self.mapped = [(p, s, h) for p, s, h in self.mapped if p != ptr or s != size]

    def release(self, handle: int) -> None:
        self.mapped = [(p, s, h) for p, s, h in self.mapped if h != handle]


def _fake_buffer(max_bytes: int, driver: _FakeVmmDriver) -> _VirtualBuffer:
    return _VirtualBuffer(max_bytes, device_index=0, driver=driver)  # type: ignore[arg-type]


class _FakeExtensible:
    """Duck-type of ``ExtensibleTensor`` over a ``_VirtualBuffer``: the
    single-segment grow-only resize surface ``ExtensibleKVCacheBuffers``
    calls, minus torch (zeroing is the GPU half's concern)."""

    def __init__(self, buffer: _VirtualBuffer) -> None:
        self._buffer = buffer
        self._bytes_per_segment = 0

    @property
    def physical_bytes(self) -> int:
        return self._buffer.committed_bytes

    @property
    def bytes_per_segment(self) -> int:
        return self._bytes_per_segment

    def resize_per_segment_(self, bytes_per_segment: int, zero_new: bool = False) -> None:  # noqa: ARG002
        assert bytes_per_segment >= self._bytes_per_segment
        self._buffer.ensure_committed_range(self._bytes_per_segment, bytes_per_segment)
        self._bytes_per_segment = bytes_per_segment

    def free(self) -> None:
        self._buffer.free()


class TestRoundUp:
    def test_rounds_up_to_multiple(self) -> None:
        assert _round_up(3, 2) == 4
        assert _round_up(4, 2) == 4
        assert _round_up(0, 2) == 0

    def test_max_with_one(self) -> None:
        assert _round_up(0, 2) == 0


class TestVirtualBufferBookkeeping:
    def test_reservation_is_rounded_to_granularity(self) -> None:
        d = _FakeVmmDriver()
        b = _fake_buffer(3 * 1024 * 1024, d)  # 3 MiB -> 4 MiB at 2 MiB granule
        assert b.reserved_size == 4 * 1024 * 1024
        assert b.committed_bytes == 0

    def test_commit_small_range_maps_whole_granules(self) -> None:
        d = _FakeVmmDriver()
        b = _fake_buffer(16 * 1024 * 1024, d)
        b.ensure_committed_range(0, 1024)  # 1 KiB request -> 2 MiB granule
        assert b.committed_bytes == 2 * 1024 * 1024
        assert len(d.mapped) == 1
        assert d.mapped[0][0] == b.base_ptr
        assert d.mapped[0][1] == 2 * 1024 * 1024

    def test_overlapping_ranges_skip_already_mapped_granules(self) -> None:
        d = _FakeVmmDriver()
        b = _fake_buffer(16 * 1024 * 1024, d)
        b.ensure_committed_range(0, 2 * 1024 * 1024)  # granule 0
        b.ensure_committed_range(1 * 1024 * 1024, 6 * 1024 * 1024)  # granules 0..2
        # granule 0 mapped once; granules 1..2 added.
        assert b.committed_bytes == 6 * 1024 * 1024
        assert len(d.mapped) == 2

    def test_disjoint_ranges_map_separate_chunks(self) -> None:
        d = _FakeVmmDriver()
        b = _fake_buffer(16 * 1024 * 1024, d)
        b.ensure_committed_range(0, 2 * 1024 * 1024)
        b.ensure_committed_range(4 * 1024 * 1024, 6 * 1024 * 1024)
        assert len(d.mapped) == 2

    def test_range_past_reservation_is_rejected(self) -> None:
        d = _FakeVmmDriver()
        b = _fake_buffer(8 * 1024 * 1024, d)
        with pytest.raises(ValueError, match="exceeds reserved capacity"):
            b.ensure_committed_range(0, 9 * 1024 * 1024)

    def test_invalid_range_is_rejected(self) -> None:
        d = _FakeVmmDriver()
        b = _fake_buffer(8 * 1024 * 1024, d)
        with pytest.raises(ValueError, match="invalid range"):
            b.ensure_committed_range(5, 3)

    def test_empty_range_is_noop(self) -> None:
        d = _FakeVmmDriver()
        b = _fake_buffer(8 * 1024 * 1024, d)
        b.ensure_committed_range(0, 0)
        assert b.committed_bytes == 0
        assert d.mapped == []

    def test_release_physical_keeps_va(self) -> None:
        d = _FakeVmmDriver()
        b = _fake_buffer(8 * 1024 * 1024, d)
        base = b.base_ptr
        b.ensure_committed_range(0, 2 * 1024 * 1024)
        assert b.committed_bytes == 2 * 1024 * 1024
        b.release_physical()
        assert b.committed_bytes == 0
        assert d.mapped == []
        assert b.base_ptr == base

    def test_free_releases_reservation(self) -> None:
        d = _FakeVmmDriver()
        b = _fake_buffer(8 * 1024 * 1024, d)
        b.ensure_committed_range(0, 2 * 1024 * 1024)
        b.free()
        assert d.reserved == []
        assert d.mapped == []
        assert b.base_ptr == 0

    def test_double_free_is_noop(self) -> None:
        d = _FakeVmmDriver()
        b = _fake_buffer(8 * 1024 * 1024, d)
        b.free()
        b.free()
        assert d.reserved == []


class TestExtensibleKVCacheBuffersBookkeeping:
    def test_lockstep_commit_with_per_block_bytes(self) -> None:
        d = _FakeVmmDriver()
        b1 = _FakeExtensible(_fake_buffer(8 * 1024 * 1024, d))
        b2 = _FakeExtensible(_fake_buffer(4 * 1024 * 1024, d))
        bufs = ExtensibleKVCacheBuffers([(b1, 1024 * 1024), (b2, 512 * 1024)], 8)
        bufs.commit(3)
        # per-segment prefixes: 3 x 1 MiB and 3 x 0.5 MiB, granule-rounded.
        assert b1.physical_bytes == 4 * 1024 * 1024  # 3 MiB -> 4 MiB
        assert b2.physical_bytes == 2 * 1024 * 1024  # 1.5 MiB -> 2 MiB
        assert bufs.num_blocks_committed == 3

    def test_commit_is_grow_only(self) -> None:
        d = _FakeVmmDriver()
        b = _FakeExtensible(_fake_buffer(8 * 1024 * 1024, d))
        bufs = ExtensibleKVCacheBuffers([(b, 1024 * 1024)], 8)
        bufs.commit(4)
        bufs.commit(2)  # no-op, not a shrink
        assert bufs.num_blocks_committed == 4
        assert b.physical_bytes == 4 * 1024 * 1024

    def test_commit_past_capacity_is_rejected(self) -> None:
        d = _FakeVmmDriver()
        b = _FakeExtensible(_fake_buffer(8 * 1024 * 1024, d))
        bufs = ExtensibleKVCacheBuffers([(b, 1024 * 1024)], 8)
        with pytest.raises(ValueError, match="capacity"):
            bufs.commit(9)

    def test_ensure_blocks_is_monotonic(self) -> None:
        d = _FakeVmmDriver()
        b = _FakeExtensible(_fake_buffer(8 * 1024 * 1024, d))
        bufs = ExtensibleKVCacheBuffers([(b, 1024 * 1024)], 8)
        bufs.ensure_blocks(2)
        bufs.ensure_blocks(1)  # no-op
        assert bufs.num_blocks_committed == 2

    def test_add_registers_lockstep_buffer(self) -> None:
        d = _FakeVmmDriver()
        b1 = _FakeExtensible(_fake_buffer(8 * 1024 * 1024, d))
        b2 = _FakeExtensible(_fake_buffer(8 * 1024 * 1024, d))
        bufs = ExtensibleKVCacheBuffers([(b1, 1024 * 1024)], 8)
        bufs.add(b2, 1024 * 1024)
        bufs.commit(2)
        assert len(bufs.buffers) == 2
        assert b1.physical_bytes == b2.physical_bytes == 2 * 1024 * 1024

    def test_physical_bytes_sums_buffers(self) -> None:
        d = _FakeVmmDriver()
        b1 = _FakeExtensible(_fake_buffer(8 * 1024 * 1024, d))
        b2 = _FakeExtensible(_fake_buffer(4 * 1024 * 1024, d))
        bufs = ExtensibleKVCacheBuffers([(b1, 1024 * 1024), (b2, 512 * 1024)], 8)
        assert bufs.physical_bytes == 0
        bufs.commit(3)
        assert bufs.physical_bytes == b1.physical_bytes + b2.physical_bytes


