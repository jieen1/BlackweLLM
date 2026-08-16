"""B4 experiments: CUDA VMM extensible KV buffer mechanics + perf parity (SM120).

Two suites, matching the research note
`notes/2026-08-16-vllm-extensible-kv-cache.md`:

  E1 -- mechanics & CUDA-graph safety:
        - VMM granularity on this card
        - 36 GiB VA reservation costs ~0 physical memory
        - incremental physical commit (K/V-split, num_segments=2)
        - torch (DLPack) views over VMM memory: write / zero / read
        - THE core claim: a CUDA graph captured while only a small prefix
          is committed stays valid after committing MORE pages under the
          same base pointer; newly committed pages are zeroed
        - release_physical() + recommit() keeps the base pointer stable

  E2 -- performance parity (the runtime's decode sits at the DRAM bandwidth
        floor, so VMM-backed KV must not cost per-step bandwidth):
        - 4 GiB bulk copy / read / write: VMM vs torch.empty
        - random 128 KiB page walks (paged-attention block-table pattern)
        - growth (commit + zero) cost, amortized per 128 KiB page vs a
          30 ms decode step

Usage:  ~/.venvs/vllm/bin/python scripts/b4_vmm_extensible_experiments.py [e1|e2]
Memory budget: ~10 GiB GPU (4 GiB torch + 4 GiB VMM x2 in E2).
"""

from __future__ import annotations

import ctypes
import sys
import time
from functools import cache

import torch

MB = 1024**2
GB = 1024**3

# ---------------------------------------------------------------------------
# Minimal CUDA VMM driver (subset of vLLM's vmm_driver.py, same call shapes)
# ---------------------------------------------------------------------------

_SUCCESS = 0
_MEM_ALLOCATION_TYPE_PINNED = 1
_MEM_LOCATION_TYPE_DEVICE = 1
_MEM_ALLOC_GRANULARITY_MINIMUM = 0
_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
_MEM_ALLOCATION_COMP_NONE = 0

_DevicePtr = ctypes.c_ulonglong
_MemHandle = ctypes.c_ulonglong
_Context = ctypes.c_void_p


class _MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _MemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _MemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _MemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _MemAllocFlags),
    ]


class _MemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _MemLocation), ("flags", ctypes.c_int)]


class _VmmDriver:
    """Uniform interface over cuMem* VMM entry points (ctypes)."""

    _symbols = {
        "get_granularity": "cuMemGetAllocationGranularity",
        "address_reserve": "cuMemAddressReserve",
        "create": "cuMemCreate",
        "map": "cuMemMap",
        "set_access": "cuMemSetAccess",
        "unmap": "cuMemUnmap",
        "release": "cuMemRelease",
        "address_free": "cuMemAddressFree",
    }

    def __init__(self) -> None:
        import ctypes as c

        self._lib = c.CDLL("libcuda.so.1")
        self._fns = {k: getattr(self._lib, v) for k, v in self._symbols.items()}
        c_uint64 = ctypes.c_uint64
        c_size_t = ctypes.c_size_t
        self._fns["get_granularity"].argtypes = [
            c.POINTER(c_size_t), c.POINTER(_MemAllocationProp), c.c_int
        ]
        self._fns["address_reserve"].argtypes = [
            c.POINTER(_DevicePtr), c_size_t, c_uint64, c_uint64, c_uint64
        ]
        self._fns["create"].argtypes = [
            c.POINTER(_MemHandle), c_size_t, c.POINTER(_MemAllocationProp), c_uint64
        ]
        self._fns["map"].argtypes = [_DevicePtr, c_size_t, c_size_t, _MemHandle, c_uint64]
        self._fns["set_access"].argtypes = [
            _DevicePtr, c_size_t, c.POINTER(_MemAccessDesc), c_size_t
        ]
        self._fns["unmap"].argtypes = [_DevicePtr, c_size_t]
        self._fns["release"].argtypes = [_MemHandle]
        self._fns["address_free"].argtypes = [_DevicePtr, c_size_t]
        for fn in self._fns.values():
            fn.restype = ctypes.c_int

    def _check(self, result: int) -> None:
        if result != _SUCCESS:
            raise RuntimeError(f"CUDA VMM error {result}")

    def granularity(self) -> int:
        g = ctypes.c_size_t()
        prop = self._make_prop()
        self._check(self._fns["get_granularity"](
            ctypes.byref(g), ctypes.byref(prop), _MEM_ALLOC_GRANULARITY_MINIMUM))
        return g.value

    def reserve(self, size: int) -> int:
        ptr = _DevicePtr()
        self._check(self._fns["address_reserve"](ctypes.byref(ptr), size, 0, 0, 0))
        return ptr.value

    def free_reserved(self, ptr: int, size: int) -> None:
        self._check(self._fns["address_free"](ptr, size))

    def create(self, size: int) -> int:
        h = _MemHandle()
        self._check(self._fns["create"](ctypes.byref(h), size, ctypes.byref(self._make_prop()), 0))
        return h.value

    def map(self, ptr: int, size: int, handle: int) -> None:
        self._check(self._fns["map"](ptr, size, 0, handle, 0))

    def set_access(self, ptr: int, size: int) -> None:
        desc = _MemAccessDesc()
        desc.location.type = _MEM_LOCATION_TYPE_DEVICE
        desc.location.id = 0
        desc.flags = _MEM_ACCESS_FLAGS_PROT_READWRITE
        self._check(self._fns["set_access"](ptr, size, ctypes.byref(desc), 1))

    def unmap(self, ptr: int, size: int) -> None:
        self._check(self._fns["unmap"](ptr, size))

    def release(self, handle: int) -> None:
        self._check(self._fns["release"](handle))

    def _make_prop(self) -> _MemAllocationProp:
        prop = _MemAllocationProp()
        prop.type = _MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = _MEM_LOCATION_TYPE_DEVICE
        prop.location.id = 0
        prop.allocFlags.compressionType = _MEM_ALLOCATION_COMP_NONE
        return prop


@cache
def driver() -> _VmmDriver:
    return _VmmDriver()


# ---------------------------------------------------------------------------
# Minimal ExtensibleTensor (subset of vLLM's extensible_tensor.py)
# ---------------------------------------------------------------------------

_K_DL_UINT = 1
_UINT8_BITS = 8


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLDeleter),
]

_KEEPALIVE: dict[int, tuple[object, object, object]] = {}
_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


def _uint8_view(ptr: int, num_bytes: int) -> torch.Tensor:
    shape_arr = (ctypes.c_int64 * 1)(num_bytes)
    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(ptr)
    managed.dl_tensor.device = _DLDevice(2, 0)  # kDLCUDA
    managed.dl_tensor.ndim = 1
    managed.dl_tensor.dtype = _DLDataType(_K_DL_UINT, _UINT8_BITS, 1)
    managed.dl_tensor.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))
    managed.dl_tensor.strides = None
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None
    key = ctypes.addressof(managed)
    deleter = _DLDeleter(lambda _p: _KEEPALIVE.pop(key, None))
    managed.deleter = deleter
    _KEEPALIVE[key] = (managed, shape_arr, deleter)
    return torch.from_dlpack(_PyCapsule_New(ctypes.addressof(managed), b"dltensor", None))


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class ExtensibleTensor:
    """Grow-only CUDA byte buffer: one VA reservation, incremental commit."""

    def __init__(self, max_num_bytes: int, num_segments: int = 1) -> None:
        d = driver()
        self.granularity = d.granularity()
        self.max_num_bytes = max_num_bytes
        self.num_segments = num_segments
        self.segment_capacity = max_num_bytes // num_segments
        self.reserved_size = _round_up(max(max_num_bytes, 1), self.granularity)
        self.base_ptr = d.reserve(self.reserved_size)
        self._bytes_per_segment = 0
        self._mapped: set[int] = set()
        self._handles: list[tuple[int, int, int]] = []

    def full_view(self) -> torch.Tensor:
        return _uint8_view(self.base_ptr, self.max_num_bytes)

    @property
    def physical_bytes(self) -> int:
        return len(self._mapped) * self.granularity

    @property
    def bytes_per_segment(self) -> int:
        return self._bytes_per_segment

    def resize_per_segment(self, bytes_per_segment: int, zero_new: bool = False) -> None:
        old = self._bytes_per_segment
        assert bytes_per_segment >= old and bytes_per_segment <= self.segment_capacity
        if bytes_per_segment == old:
            return
        for i in range(self.num_segments):
            start = i * self.segment_capacity
            self._ensure_range(start + old, start + bytes_per_segment)
        self._bytes_per_segment = bytes_per_segment
        if zero_new:
            full = self.full_view()
            for i in range(self.num_segments):
                start = i * self.segment_capacity
                full[start + old : start + bytes_per_segment].zero_()

    def _ensure_range(self, start: int, end: int) -> None:
        first = start // self.granularity
        last = (end + self.granularity - 1) // self.granularity  # exclusive
        run_start = None
        for g in range(first, last + 1):
            unmapped = g < last and g not in self._mapped
            if unmapped and run_start is None:
                run_start = g
            elif not unmapped and run_start is not None:
                self._map_chunk(run_start * self.granularity, (g - run_start) * self.granularity)
                self._mapped.update(range(run_start, g))
                run_start = None

    def _map_chunk(self, offset: int, size: int) -> None:
        d = driver()
        handle = d.create(size)
        try:
            d.map(self.base_ptr + offset, size, handle)
        except Exception:
            d.release(handle)
            raise
        d.set_access(self.base_ptr + offset, size)
        self._handles.append((handle, offset, size))

    def release_physical(self) -> None:
        torch.cuda.synchronize()
        d = driver()
        for handle, offset, size in self._handles:
            d.unmap(self.base_ptr + offset, size)
            d.release(handle)
        self._handles = []
        self._mapped = set()
        self._bytes_per_segment = 0

    def free(self) -> None:
        if self.base_ptr:
            self.release_physical()
            driver().free_reserved(self.base_ptr, self.reserved_size)
            self.base_ptr = 0


# ---------------------------------------------------------------------------
# E1: mechanics + CUDA-graph safety
# ---------------------------------------------------------------------------


def e1() -> None:
    torch.cuda.init()
    print(f"torch {torch.__version__} cuda {torch.version.cuda} dev {torch.cuda.get_device_name()}")
    d = driver()
    gran = d.granularity()
    print(f"VMM allocation granularity: {gran / 2**20:.2f} MiB")
    assert gran % MB == 0

    # 36 GiB VA reservation: zero physical cost?
    pool_bytes = int(36.0 * GB)
    t0 = time.monotonic()
    probe_ptr = d.reserve(pool_bytes)
    dt = time.monotonic() - t0
    d.free_reserved(probe_ptr, pool_bytes)
    print(f"reserve+free 36 GiB VA: {dt * 1e3:.1f} ms")
    free_before, _ = torch.cuda.mem_get_info()
    buf = ExtensibleTensor(pool_bytes, num_segments=2)
    torch.cuda.synchronize()
    free_after, _ = torch.cuda.mem_get_info()
    print(f"36 GiB VA reserved x2 segments, physical committed: "
          f"{buf.physical_bytes / MB:.2f} MiB "
          f"(free delta {(free_before - free_after) / MB:.1f} MiB)")
    assert buf.physical_bytes == 0

    # commit 1 bundle per segment (1 MiB each) + zero
    buf.resize_per_segment(1 * MB, zero_new=True)
    print(f"after commit(1 bundle/seg): physical={buf.physical_bytes / MB:.2f} MiB")

    full = buf.full_view()
    assert full.data_ptr() == buf.base_ptr
    seg1 = full[buf.segment_capacity : buf.segment_capacity + buf.bytes_per_segment]
    seg1[:4096].copy_(torch.arange(4096, dtype=torch.uint8, device="cuda"))
    torch.cuda.synchronize()
    assert torch.equal(seg1[:4096].cpu(), torch.arange(4096, dtype=torch.uint8))
    assert seg1[4096:8192].sum().item() == 0
    print("write + fresh-page-zero on committed prefix: OK")

    # CUDA graph: capture while 1 bundle committed, replay after committing 512
    ida = torch.arange(511 * MB, 512 * MB, dtype=torch.int64, device="cuda")
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        full.index_fill_(0, ida, 0xAB)
    torch.cuda.synchronize()
    buf.resize_per_segment(512 * MB, zero_new=True)
    torch.cuda.synchronize()
    print(f"after commit(512 bundles): physical={buf.physical_bytes / MB:.2f} MiB")
    g.replay()
    torch.cuda.synchronize()
    new_region = full[511 * MB : 512 * MB]
    assert new_region.sum().item() == 0xAB * (1 * MB), (
        "graph replay into newly-committed pages failed"
    )
    assert full[510 * MB : 510 * MB + 64 * 1024].sum().item() == 0
    print("CUDA graph captured pre-commit, replayed post-commit: "
          "OK (base-pointer stability proven)")

    # release_physical + recommit keeps VA
    base_before = buf.base_ptr
    buf.release_physical()
    torch.cuda.synchronize()
    assert buf.base_ptr == base_before
    buf.resize_per_segment(512 * MB, zero_new=True)
    torch.cuda.synchronize()
    assert full[0:1024].sum().item() == 0
    print("release_physical + recommit: VA stable, pages re-zeroed: OK")

    buf.free()
    torch.cuda.synchronize()
    print("E1 ALL PASS")


# ---------------------------------------------------------------------------
# E2: performance parity (DRAM bandwidth floor matters for decode)
# ---------------------------------------------------------------------------


def _bench(fn, reps=5):
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times = sorted(times)
    return times[len(times) // 2], times[0]  # median, min


def e2() -> None:
    torch.cuda.init()
    print(f"torch {torch.__version__} cuda {torch.version.cuda} dev {torch.cuda.get_device_name()}")
    N = 4 * GB
    src_t = torch.empty(N, dtype=torch.uint8, device="cuda")
    dst_t = torch.empty(N, dtype=torch.uint8, device="cuda")
    src_v = ExtensibleTensor(N)
    dst_v = ExtensibleTensor(N)
    src_v.resize_per_segment(N, zero_new=False)
    dst_v.resize_per_segment(N, zero_new=False)
    torch.cuda.synchronize()
    dst_t.copy_(src_t)
    dst_v.full_view()[:N].copy_(src_v.full_view()[:N])
    torch.cuda.synchronize()

    def report(name, tt, tv, nbytes, unit=1e9):
        line = f"{name}: torch {nbytes / tt / unit:.0f} GB/s | "
        line += f"vmm {nbytes / tv / unit:.0f} GB/s | ratio {tv / tt:.3f}"
        print(line)

    m, _ = _bench(lambda: dst_t.copy_(src_t), 7)
    mv, _ = _bench(lambda: dst_v.full_view()[:N].copy_(src_v.full_view()[:N]), 7)
    report("copy 4GiB", m, mv, N)
    m, _ = _bench(lambda: torch.sum(src_t), 7)
    mv, _ = _bench(lambda: torch.sum(src_v.full_view()[:N]), 7)
    report("read 4GiB", m, mv, N)
    m, _ = _bench(lambda: dst_t.fill_(7), 11)
    mv, _ = _bench(lambda: dst_v.full_view()[:N].fill_(7), 11)
    report("write 4GiB", m, mv, N)

    PAGE = 128 * 1024  # Qwen3.8 KV page (64 tok x 4 heads x 256 dim x FP8 x K+V)
    NPAGES = N // PAGE
    rng = torch.Generator(device="cuda").manual_seed(42)
    pages = torch.randint(0, NPAGES, (16384,), device="cuda", generator=rng)
    acc_t = torch.empty(16384, dtype=torch.uint8, device="cuda")
    acc_v = torch.empty(16384, dtype=torch.uint8, device="cuda")

    def page_walk(t_buf, out):
        gathered = t_buf.view(NPAGES, PAGE)[pages]
        out.copy_(gathered.max(dim=1).values)

    m, _ = _bench(lambda: page_walk(src_t, acc_t), 9)
    mv, _ = _bench(lambda: page_walk(src_v.full_view()[:N], acc_v), 9)
    report("paged-walk 2GiB", m, mv, 16384 * PAGE)
    assert torch.equal(acc_t, acc_v)

    grow = ExtensibleTensor(N)
    grow.resize_per_segment(2 * GB, zero_new=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    grow.resize_per_segment(4 * GB, zero_new=True)
    torch.cuda.synchronize()
    grow_s = time.perf_counter() - t0
    print(f"grow 2->4 GiB (commit+zero): {grow_s * 1e3:.1f} ms")
    per_page = (128 * 1024) / (2 * GB / grow_s)
    print(f"per-128KiB-page commit cost: {per_page * 1e6:.1f} us "
          f"= {per_page * 1e6 / 30000 * 100:.3f}% of a 30 ms decode step")

    for b in (src_v, dst_v, grow):
        b.free()
    print("E2 DONE")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "e1"
    {"e1": e1, "e2": e2}[which]()
