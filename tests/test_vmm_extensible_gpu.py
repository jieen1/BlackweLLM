"""Phase A GPU tests: CUDA VMM extensible buffers (runtime/model/vmm_extensible.py).

The GPU half of ``tests/test_vmm_extensible.py`` (kept separate so the
torch-free job collects the pure-bookkeeping half). Covers the research
note's E1 claims on real hardware: 36 GiB VA reservation costs ~0 physical
memory, incremental commit + zeroing, K/V-split lockstep segments, base-
pointer stability across commits, a CUDA graph captured while only a small
prefix is committed replaying correctly into post-commit pages, and
release/recommit.

Self-skips when torch or a CUDA device is unavailable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)

from runtime.model.vmm_extensible import (  # noqa: E402
    ExtensibleKVCacheBuffers,
    ExtensibleTensor,
    get_vmm_driver,
)

MB = 1024**2
GB = 1024**3


class TestExtensibleTensorGpu:
    def test_va_reservation_costs_no_physical_memory(self) -> None:
        free_before, _ = torch.cuda.mem_get_info()
        et = ExtensibleTensor(max_num_bytes=36 * GB, device_index=0)
        torch.cuda.synchronize()
        free_after, _ = torch.cuda.mem_get_info()
        try:
            assert et.physical_bytes == 0
            assert free_before - free_after < 4 * MB, "36 GiB VA must be ~free"
        finally:
            et.free()
            torch.cuda.synchronize()

    def test_granularity_matches_driver(self) -> None:
        assert get_vmm_driver().granularity(0) == 2 * MB
        et = ExtensibleTensor(max_num_bytes=4 * MB, device_index=0)
        try:
            assert et._buffer.granularity == 2 * MB  # noqa: SLF001 - test probe
        finally:
            et.free()

    def test_incremental_commit_and_zeroing(self) -> None:
        et = ExtensibleTensor(max_num_bytes=8 * MB, device_index=0)
        try:
            et.resize_per_segment_(1 * MB, zero_new=True)
            assert et.physical_bytes == 2 * MB  # granule-rounded
            full = et.full_view()
            assert full.data_ptr() == et.base_ptr
            # freshly committed bytes are zero
            assert full[0 : 64 * 1024].sum().item() == 0
            # write into the committed prefix survives
            full[0:4096].copy_(torch.arange(4096, dtype=torch.uint8, device="cuda"))
            torch.cuda.synchronize()
            assert torch.equal(full[0:4096].cpu(), torch.arange(4096, dtype=torch.uint8))
            # growing again preserves old data and zeroes only new bytes
            et.resize_per_segment_(4 * MB, zero_new=True)
            assert torch.equal(full[0:4096].cpu(), torch.arange(4096, dtype=torch.uint8))
            assert full[1 * MB : 1 * MB + 4096].sum().item() == 0
        finally:
            et.free()

    def test_kv_split_two_segments_grow_in_lockstep(self) -> None:
        et = ExtensibleTensor(max_num_bytes=16 * MB, num_segments=2)
        try:
            et.resize_per_segment_(1 * MB, zero_new=True)
            assert et.bytes_per_segment == 1 * MB
            assert et.physical_bytes == 4 * MB  # 1 MiB + 1 MiB, granule-rounded
            full = et.full_view()
            seg1 = full[8 * MB : 8 * MB + 1 * MB]
            seg1[:1024].copy_(torch.arange(1024, dtype=torch.uint8, device="cuda"))
            torch.cuda.synchronize()
            et.resize_per_segment_(2 * MB, zero_new=True)
            assert torch.equal(
                full[8 * MB : 8 * MB + 1024].cpu(),
                torch.arange(1024, dtype=torch.uint8),
            )
            # new per-segment prefixes are zeroed: [1MB,2MB) of seg0 and
            # [9MB,10MB) of seg1 (both within committed granules)
            assert full[1 * MB : 1 * MB + 1024].sum().item() == 0
            assert full[9 * MB : 9 * MB + 1024].sum().item() == 0
        finally:
            et.free()

    def test_grow_only_rejects_shrink(self) -> None:
        et = ExtensibleTensor(max_num_bytes=8 * MB, device_index=0)
        try:
            et.resize_per_segment_(2 * MB)
            with pytest.raises(ValueError, match="grow-only"):
                et.resize_per_segment_(1 * MB)
        finally:
            et.free()

    def test_cuda_graph_captured_pre_commit_replays_post_commit(self) -> None:
        # THE core claim (research note E1): capture with a tiny committed
        # prefix, commit more pages under the same base pointer, replay.
        et = ExtensibleTensor(max_num_bytes=64 * MB, device_index=0)
        try:
            et.resize_per_segment_(1 * MB, zero_new=True)
            full = et.full_view()
            idx = torch.arange(31 * MB, 32 * MB, dtype=torch.int64, device="cuda")
            g = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                full.index_fill_(0, idx, 0xAB)
            torch.cuda.synchronize()
            et.resize_per_segment_(32 * MB, zero_new=True)
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            region = full[31 * MB : 32 * MB]
            assert region.sum().item() == 0xAB * (1 * MB)
            # untouched committed region stays zero
            assert full[30 * MB : 30 * MB + 64 * 1024].sum().item() == 0
        finally:
            et.free()

    def test_release_physical_keeps_base_pointer(self) -> None:
        et = ExtensibleTensor(max_num_bytes=8 * MB, device_index=0)
        try:
            et.resize_per_segment_(2 * MB, zero_new=True)
            full = et.full_view()
            base = et.base_ptr
            assert full.data_ptr() == base
            et.release_physical()
            torch.cuda.synchronize()
            assert et.base_ptr == base
            assert et.physical_bytes == 0
            et.resize_per_segment_(2 * MB, zero_new=True)
            torch.cuda.synchronize()
            assert full.data_ptr() == base
            assert full[0:1024].sum().item() == 0, "re-committed pages re-zeroed"
        finally:
            et.free()


class TestExtensibleKVCacheBuffersGpu:
    def test_lockstep_commit_zeroes_new_blocks_only(self) -> None:
        et1 = ExtensibleTensor(max_num_bytes=16 * MB)
        et2 = ExtensibleTensor(max_num_bytes=8 * MB)
        try:
            bufs = ExtensibleKVCacheBuffers([(et1, 1 * MB), (et2, 512 * 1024)], 16)
            bufs.commit(2)
            assert bufs.num_blocks_committed == 2
            assert et1.bytes_per_segment == 2 * MB
            assert et2.bytes_per_segment == 1 * MB
            f1 = et1.full_view()
            f1[0:1024].copy_(torch.arange(1024, dtype=torch.uint8, device="cuda"))
            torch.cuda.synchronize()
            bufs.commit(4)
            assert torch.equal(f1[0:1024].cpu(), torch.arange(1024, dtype=torch.uint8))
            assert f1[3 * MB : 3 * MB + 1024].sum().item() == 0
            assert bufs.physical_bytes > 0
        finally:
            et1.free()
            et2.free()
