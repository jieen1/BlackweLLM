"""CUDA VMM-backed extensible GPU byte buffers (dynamic physical KV, Phase 5.5).

Research: ``notes/2026-08-16-vllm-extensible-kv-cache.md``. Concept adapted
from vLLM's ``extensible-kv-cache`` branch (``vllm/utils/vmm_driver.py`` /
``extensible_tensor.py``) -- same CUDA VMM call surface and the same
"reserve full VA up front, commit physical pages incrementally" semantics,
self-contained here (no vLLM import) and hardened with the SM120 experiments
in ``scripts/b4_vmm_extensible_experiments.py``:

  - VMM granularity on this card: 2 MiB; reserving 36 GiB of VA costs ~0
    physical memory and ~1 ms.
  - A CUDA graph captured while only a small prefix is committed stays valid
    after committing more pages under the stable base pointer (proven by
    capture -> commit -> replay in the experiments).
  - Bandwidth parity with cudaMalloc'd torch tensors within +/-2% on bulk
    copy/read/write and random 128 KiB page walks; growth commit+zero costs
    1.7 us per 128 KiB page (0.006% of a 30 ms decode step).

Semantics (locked by this module's docstrings and the tests):

  - Grow-only: physical memory is committed as a prefix of each segment and
    never shrinks; ``release_physical`` + recommit is the only reset path.
  - ``resize_per_segment_(n, zero_new=True)`` zeroes exactly the newly
    committed prefix. Granules rounded outward past ``n`` ARE committed but
    NOT zeroed -- callers must never read past the committed prefix (paged
    indexing satisfies this by construction).
  - The base pointer never moves while the buffer is alive; torch views
    built over ``full_view()`` stay valid across commits.

Import discipline: this module imports no torch at module scope (the CI
torch-free job collects ``tests/`` with a bare ``import torch`` failing);
torch is imported lazily inside the GPU-facing methods only.
"""

from __future__ import annotations

import ctypes
from contextlib import suppress
from functools import cache

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


class VmmDriver:
    """Uniform ctypes interface over the CUDA driver's virtual-memory APIs.

    The call signatures, struct layouts, and constants are the CUDA driver
    API's public VMM surface (``cuMemAddressReserve`` / ``cuMemCreate`` /
    ``cuMemMap`` / ``cuMemSetAccess`` / ``cuMemUnmap`` / ``cuMemRelease`` /
    ``cuMemAddressFree``), the same entry points PyTorch's
    expandable-segments allocator and vLLM's extensible KV cache use.
    Torch-free by design: context selection uses the driver API directly.
    """

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
        self._lib = ctypes.CDLL("libcuda.so.1")
        self._fns: dict[str, ctypes.CDLL] = {}
        for logical, symbol in self._symbols.items():
            self._fns[logical] = getattr(self._lib, symbol)
        self._configure_signatures()
        self._lib.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self._lib.cuGetErrorString.restype = ctypes.c_int
        self._lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(_Context)]
        self._lib.cuCtxGetCurrent.restype = ctypes.c_int
        self._lib.cuDevicePrimaryCtxRetain.argtypes = [
            ctypes.POINTER(_Context),
            ctypes.c_int,
        ]
        self._lib.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        self._lib.cuCtxSetCurrent.argtypes = [_Context]
        self._lib.cuCtxSetCurrent.restype = ctypes.c_int

    def _configure_signatures(self) -> None:
        pointer = ctypes.POINTER
        c_size_t = ctypes.c_size_t
        c_uint64 = ctypes.c_uint64
        fns = self._fns
        fns["get_granularity"].argtypes = [
            pointer(c_size_t),
            pointer(_MemAllocationProp),
            ctypes.c_int,
        ]
        fns["address_reserve"].argtypes = [
            pointer(_DevicePtr),
            c_size_t,
            c_uint64,
            c_uint64,
            c_uint64,
        ]
        fns["create"].argtypes = [
            pointer(_MemHandle),
            c_size_t,
            pointer(_MemAllocationProp),
            c_uint64,
        ]
        fns["map"].argtypes = [_DevicePtr, c_size_t, c_size_t, _MemHandle, c_uint64]
        fns["set_access"].argtypes = [
            _DevicePtr,
            c_size_t,
            pointer(_MemAccessDesc),
            c_size_t,
        ]
        fns["unmap"].argtypes = [_DevicePtr, c_size_t]
        fns["release"].argtypes = [_MemHandle]
        fns["address_free"].argtypes = [_DevicePtr, c_size_t]
        for fn in fns.values():
            fn.restype = ctypes.c_int

    def error_string(self, code: int) -> str:
        msg = ctypes.c_char_p()
        self._lib.cuGetErrorString(code, ctypes.byref(msg))
        return msg.value.decode() if msg.value else "unknown error"

    def _check(self, result: int) -> None:
        if result == 0:
            return
        raise RuntimeError(f"CUDA VMM error {result}: {self.error_string(result)}")

    def ensure_context(self, device_index: int) -> None:
        """Make a primary driver context for ``device_index`` current."""
        pctx = _Context()
        self._check(self._lib.cuCtxGetCurrent(ctypes.byref(pctx)))
        if pctx.value:
            return
        self._check(
            self._lib.cuDevicePrimaryCtxRetain(ctypes.byref(pctx), device_index)
        )
        self._check(self._lib.cuCtxSetCurrent(pctx))

    def _make_alloc_prop(self, device_index: int) -> _MemAllocationProp:
        prop = _MemAllocationProp()
        prop.type = _MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = _MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device_index
        prop.allocFlags.compressionType = _MEM_ALLOCATION_COMP_NONE
        return prop

    def granularity(self, device_index: int) -> int:
        prop = self._make_alloc_prop(device_index)
        g = ctypes.c_size_t()
        self._check(
            self._fns["get_granularity"](
                ctypes.byref(g), ctypes.byref(prop), _MEM_ALLOC_GRANULARITY_MINIMUM
            )
        )
        return g.value

    def reserve(self, size: int) -> int:
        """Reserve a virtual address range; returns the base pointer."""
        ptr = _DevicePtr()
        self._check(self._fns["address_reserve"](ctypes.byref(ptr), size, 0, 0, 0))
        return ptr.value

    def free_reserved(self, ptr: int, size: int) -> None:
        self._check(self._fns["address_free"](ptr, size))

    def create(self, size: int, device_index: int) -> int:
        """Create one physical memory handle of ``size`` bytes."""
        handle = _MemHandle()
        self._check(
            self._fns["create"](
                ctypes.byref(handle), size, ctypes.byref(self._make_alloc_prop(device_index)), 0
            )
        )
        return handle.value

    def map(self, ptr: int, size: int, handle: int) -> None:
        self._check(self._fns["map"](ptr, size, 0, handle, 0))

    def set_access(self, ptr: int, size: int, device_index: int) -> None:
        desc = _MemAccessDesc()
        desc.location.type = _MEM_LOCATION_TYPE_DEVICE
        desc.location.id = device_index
        desc.flags = _MEM_ACCESS_FLAGS_PROT_READWRITE
        self._check(self._fns["set_access"](ptr, size, ctypes.byref(desc), 1))

    def unmap(self, ptr: int, size: int) -> None:
        self._check(self._fns["unmap"](ptr, size))

    def release(self, handle: int) -> None:
        self._check(self._fns["release"](handle))


@cache
def get_vmm_driver() -> VmmDriver:
    return VmmDriver()


@cache
def vmm_unavailable_reason() -> str | None:
    """Probe VMM support; None if usable, else a human-readable reason."""
    try:
        driver = get_vmm_driver()
        driver.ensure_context(0)
        gran = driver.granularity(0)
        ptr = driver.reserve(gran)
        driver.free_reserved(ptr, gran)
    except Exception as exc:  # noqa: BLE001 - probe returns the reason
        return str(exc)
    return None


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class _VirtualBuffer:
    """One device VA reservation plus the physical chunks mapped into it.

    Physical memory is committed incrementally at granularity-sized granules
    via :meth:`ensure_committed_range`; granules already mapped by an earlier
    (possibly overlapping) range are skipped, so ranges may abut or overlap
    freely. ``driver`` is injectable for torch-free accounting tests.
    """

    def __init__(
        self,
        max_bytes: int,
        device_index: int = 0,
        driver: VmmDriver | None = None,
    ) -> None:
        self._driver = driver if driver is not None else get_vmm_driver()
        self.device_index = device_index
        self._driver.ensure_context(device_index)
        self.granularity = self._driver.granularity(device_index)
        self.reserved_size = _round_up(max(max_bytes, 1), self.granularity)
        self.base_ptr = self._driver.reserve(self.reserved_size)
        self._mapped_granules: set[int] = set()
        self._handles: list[tuple[int, int, int]] = []
        self._freed = False

    @property
    def committed_bytes(self) -> int:
        """Physically mapped bytes (a multiple of the granularity)."""
        return len(self._mapped_granules) * self.granularity

    def ensure_committed_range(self, start: int, end: int) -> None:
        """Map physical pages so the byte range ``[start, end)`` is backed.

        The range is widened outward to granule boundaries; granules mapped
        by earlier calls are skipped, so a granule shared by two requested
        ranges is mapped once.
        """
        if not 0 <= start <= end:
            raise ValueError(f"invalid range [{start}, {end})")
        if end > self.reserved_size:
            raise ValueError(
                f"range end {end} exceeds reserved capacity {self.reserved_size}"
            )
        if start == end:
            return
        first = start // self.granularity
        last = (end + self.granularity - 1) // self.granularity  # exclusive
        run_start: int | None = None
        for g in range(first, last + 1):
            unmapped = g < last and g not in self._mapped_granules
            if unmapped and run_start is None:
                run_start = g
            elif not unmapped and run_start is not None:
                self._map_chunk_at(
                    run_start * self.granularity, (g - run_start) * self.granularity
                )
                self._mapped_granules.update(range(run_start, g))
                run_start = None

    def _map_chunk_at(self, offset: int, size: int) -> None:
        """Create one physical chunk of ``size`` bytes and map it at ``offset``."""
        driver = self._driver
        driver.ensure_context(self.device_index)
        handle = driver.create(size, self.device_index)
        addr = self.base_ptr + offset
        try:
            driver.map(addr, size, handle)
        except Exception:
            driver.release(handle)
            raise
        driver.set_access(addr, size, self.device_index)
        self._handles.append((handle, offset, size))

    def release_physical(self) -> None:
        """Unmap and release all physical memory, keeping the VA reservation.

        The base pointer (and any tensor views over it) stays valid but
        unbacked; ``ensure_committed_range`` maps fresh physical pages again.
        """
        driver = self._driver
        driver.ensure_context(self.device_index)
        if self._handles:
            try:
                import torch  # local: module stays torch-free at import time

                torch.accelerator.synchronize(self.device_index)
            except ImportError:
                # torch-free accounting path: nothing real to synchronize.
                pass
        for handle, offset, size in self._handles:
            driver.unmap(self.base_ptr + offset, size)
            driver.release(handle)
        self._handles = []
        self._mapped_granules = set()

    def free(self) -> None:
        if self._freed:
            return
        self._freed = True
        self.release_physical()
        if self.base_ptr:
            self._driver.free_reserved(self.base_ptr, self.reserved_size)
        self.base_ptr = 0

    def __del__(self) -> None:
        with suppress(Exception):
            self.free()


_K_DL_UINT = 1
_UINT8_BITS = 8


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


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


def uint8_tensor_from_ptr(ptr: int, num_bytes: int, device_index: int) -> object:
    """A torch uint8 tensor viewing driver-mapped memory (DLPack import).

    The returned tensor owns a reference to the DLPack managed struct via
    ``_KEEPALIVE``; the pointer must stay valid for the tensor's lifetime.
    Lazily imports torch -- torch-free callers must not call this.
    """
    import torch  # local: module stays torch-free at import time

    shape_arr = (ctypes.c_int64 * 1)(num_bytes)
    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(ptr)
    managed.dl_tensor.device = _DLDevice(2, device_index)  # kDLCUDA
    managed.dl_tensor.ndim = 1
    managed.dl_tensor.dtype = _DLDataType(_K_DL_UINT, _UINT8_BITS, 1)
    managed.dl_tensor.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))
    managed.dl_tensor.strides = None
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None
    key = ctypes.addressof(managed)
    deleter = _DLDeleter(lambda _managed_ptr: _KEEPALIVE.pop(key, None))
    managed.deleter = deleter
    _KEEPALIVE[key] = (managed, shape_arr, deleter)
    capsule = _PyCapsule_New(ctypes.addressof(managed), b"dltensor", None)
    return torch.from_dlpack(capsule)


class ExtensibleTensor:
    """A 1-D CUDA byte buffer that grows without moving its base pointer.

    With ``num_segments > 1`` the reservation is divided into that many
    equal segments that grow in lockstep: the committed bytes form a prefix
    of each segment (segment ``i`` spans ``[i * seg_cap, (i+1) * seg_cap)``
    of ``full_view()``). This backs layouts whose block dimension is not
    outermost (e.g. a K/V-split cache with ``num_segments=2``).

    ``full_view()`` spans the full reserved capacity and is the ONLY view a
    caller should hold across commits: existing data and the base pointer
    stay valid as more pages are mapped underneath it. ``resize_per_segment_``
    is grow-only.
    """

    def __init__(
        self,
        max_num_bytes: int,
        *,
        device_index: int = 0,
        num_segments: int = 1,
    ) -> None:
        if max_num_bytes < 0:
            raise ValueError("max_num_bytes must be non-negative")
        if num_segments < 1:
            raise ValueError(f"num_segments must be positive, got {num_segments}")
        if max_num_bytes % num_segments != 0:
            raise ValueError(
                f"max_num_bytes ({max_num_bytes}) must be divisible by "
                f"num_segments ({num_segments})"
            )
        import torch  # local: module stays torch-free at import time

        torch.cuda.init()
        self._device_index = device_index
        self._max_num_bytes = max_num_bytes
        self._num_segments = num_segments
        self._segment_capacity_bytes = max_num_bytes // num_segments
        self._buffer = _VirtualBuffer(max_num_bytes, device_index=device_index)
        self._bytes_per_segment = 0

    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def max_num_bytes(self) -> int:
        return self._max_num_bytes

    @property
    def num_segments(self) -> int:
        return self._num_segments

    @property
    def segment_capacity_bytes(self) -> int:
        return self._segment_capacity_bytes

    @property
    def base_ptr(self) -> int:
        return self._buffer.base_ptr

    @property
    def bytes_per_segment(self) -> int:
        return self._bytes_per_segment

    @property
    def num_bytes(self) -> int:
        """Currently committed bytes, summed over all segments."""
        return self._bytes_per_segment * self._num_segments

    @property
    def physical_bytes(self) -> int:
        """Physically mapped bytes (committed size rounded up to granules)."""
        return self._buffer.committed_bytes

    def full_view(self) -> object:
        """A uint8 torch view spanning the full reserved capacity."""
        return uint8_tensor_from_ptr(
            self._buffer.base_ptr, self._max_num_bytes, self._device_index
        )

    def resize_per_segment_(self, bytes_per_segment: int, zero_new: bool = False) -> None:
        """Grow every segment's committed prefix to ``bytes_per_segment`` bytes.

        Existing bytes are preserved and the base pointer is unchanged. With
        ``zero_new=True`` the newly committed byte range of each segment is
        zeroed (bytes committed earlier are left intact). Raises if smaller
        than the current per-segment size (shrink is unsupported) or larger
        than ``segment_capacity_bytes``.
        """
        old = self._bytes_per_segment
        if bytes_per_segment < old:
            raise ValueError(
                f"ExtensibleTensor is grow-only: cannot resize from {old} "
                f"to {bytes_per_segment} bytes per segment"
            )
        if bytes_per_segment > self._segment_capacity_bytes:
            raise ValueError(
                f"requested {bytes_per_segment} bytes per segment exceeds "
                f"segment capacity {self._segment_capacity_bytes}"
            )
        if bytes_per_segment == old:
            return
        for i in range(self._num_segments):
            start = i * self._segment_capacity_bytes
            self._buffer.ensure_committed_range(start + old, start + bytes_per_segment)
        self._bytes_per_segment = bytes_per_segment
        if zero_new:
            full = self.full_view()
            for i in range(self._num_segments):
                start = i * self._segment_capacity_bytes
                full[start + old : start + bytes_per_segment].zero_()

    def release_physical(self) -> None:
        """Release all physical memory while keeping the VA reservation.

        Existing tensor views stay pointer-valid but must not be accessed
        until the buffer is committed again; the data is discarded.
        """
        self._buffer.release_physical()
        self._bytes_per_segment = 0

    def free(self) -> None:
        self._buffer.free()
        self._bytes_per_segment = 0


class ExtensibleKVCacheBuffers:
    """Grow-only physical backing for a set of KV buffers, committed in
    lockstep as a per-segment prefix of blocks.

    ``commit`` maps (and zeroes) physical pages for additional blocks while
    keeping every buffer's base pointer, existing data, and the logical
    views built over the full reserved capacity stable -- the property that
    keeps CUDA graphs captured before the final commit valid afterwards.
    """

    def __init__(
        self,
        buffers: list[tuple[ExtensibleTensor, int]],
        num_blocks_capacity: int,
    ) -> None:
        # Each entry is (buffer, bytes_per_block_per_segment).
        self.buffers = buffers
        self.num_blocks_capacity = num_blocks_capacity
        self.num_blocks_committed = 0

    @property
    def physical_bytes(self) -> int:
        return sum(buffer.physical_bytes for buffer, _ in self.buffers)

    def add(self, buffer: ExtensibleTensor, bytes_per_block_per_segment: int) -> None:
        """Register another lockstep buffer (e.g. the pooled MTP KV)."""
        self.buffers.append((buffer, bytes_per_block_per_segment))

    def commit(self, num_blocks: int) -> None:
        """Grow the committed prefix of every buffer to ``num_blocks`` blocks.

        Newly committed blocks are zeroed; previously committed ones are
        left intact. Grow-only.
        """
        if num_blocks > self.num_blocks_capacity:
            raise ValueError(
                f"cannot commit {num_blocks} blocks; capacity is "
                f"{self.num_blocks_capacity}"
            )
        if num_blocks <= self.num_blocks_committed:
            return
        for buffer, bytes_per_block_per_segment in self.buffers:
            buffer.resize_per_segment_(
                num_blocks * bytes_per_block_per_segment, zero_new=True
            )
        self.num_blocks_committed = num_blocks

    def ensure_blocks(self, num_blocks: int) -> None:
        """Grow the committed prefix to at least ``num_blocks`` (no-op when
        already committed further -- warmup/capture-time hook)."""
        self.commit(max(num_blocks, self.num_blocks_committed))

    def free(self) -> None:
        for buffer, _ in self.buffers:
            buffer.free()
        self.buffers = []
        self.num_blocks_committed = 0
