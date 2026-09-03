"""Flash-Next PLE (n-gram conditional-memory) injection layer.

Mechanics pinned from the reference implementation (sglang
``models/qwen4_exp.py`` ``Qwen4ExpNGramEmbedding`` / ``Qwen4ExpPLELayer``,
read 2026-08-27) and the RadixArk checkpoint tensors:

Hash (per PLE layer; this model has one at layer index 1):
* ``ngram_heads = (ngram_size-1) * heads_per_ngram = 16`` heads, each with
  its own prime modulus (~20M, ``ngram_heads_vocab_sizes``) and row offset
  (``ngram_heads_offsets``); the table is the concatenation of all heads'
  rows, split into 128 FP8 shards of ``[2500012, 160]``;
* ``layer_multipliers`` (3 odd int64, splitmix64-derived, stored in the
  checkpoint); the n-gram window is ``[t-2, t-1, t]`` with EOS-segment
  resets (positions closer than ``shift`` to the segment start read EOS);
* bigram heads: ``mix = tok[t]*m0 XOR tok[t-1]*m1``; trigram heads add
  ``XOR tok[t-2]*m2``; ``id = mix % size_h + offset_h``.

Injection (at the layer whose ``(layer_id + 1)`` is in ``ple_layer_ids``):
``embed [T,16,160]*scale -> flatten [T,2560] -> key/value proj -> grouped
Gemma-RMSNorms -> sigmoid(sqrt-soft gate from key.query/sqrt(hs)) * value
-> + SiLU(depthwise dilated causal conv1d(kernel 4, dilation 3, per-request
state of 9 steps))`` -> ``[T, hc*hs=10240]`` added to the widened hidden.

Table modes: ``stream`` (batched page reads with persistent high-QD workers,
page/row caches and pinned GPU staging; fits 23 GB RAM) and ``resident``
(whole table in pinned memory; for hosts with enough RAM -- set
``resident=True`` once the memory exists).
"""

from __future__ import annotations

import errno
import importlib.machinery
import importlib.util
import json
import mmap
import os
import pathlib
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import torch
from torch import nn

from runtime.model.flashnext.hyper_connection import GroupedGemmaRMSNorm

_PLE_PREFIX = "model.language_model.layers.{layer}.ple"
_PLE_IO_ENV = "QSR_FLASHNEXT_PLE_IO"
_PLE_IO_AUTO = "auto"
_PLE_IO_PREAD = "pread"
_PLE_IO_URING = "io_uring"
_PLE_IO_FALLBACK_ERRNOS = frozenset(
    {
        errno.ENOSYS,
        errno.EPERM,
        errno.EOPNOTSUPP,
        errno.EINVAL,
        errno.ENODEV,
        errno.ENXIO,
    }
)
_PLE_IO_URING_READER_TYPE = None


def _extension_matches_runtime(path: pathlib.Path) -> bool:
    """Return whether a tagged CPython extension can load in this process.

    The local SGLang checkout may contain a ``_storage`` extension built for a
    different Python minor (currently the common case is ``cpython-312`` while
    the validated serving runtime is Python 3.14).  Importing that extension by
    file path bypasses Python's normal suffix filtering; on CPython 3.14 it can
    segfault before raising an import error.  ABI3 and untagged extensions are
    left to the loader because they explicitly advertise cross-minor support.
    """
    name = path.name
    marker = ".cpython-"
    if marker not in name:
        return True
    tag = name.split(marker, 1)[1].split("-", 1)[0]
    current = f"{sys.version_info.major}{sys.version_info.minor}"
    return tag == current


def _resolve_ple_io_mode() -> str:
    raw = os.environ.get(_PLE_IO_ENV, _PLE_IO_AUTO).strip().lower()
    if raw in ("", _PLE_IO_AUTO):
        return _PLE_IO_AUTO
    if raw in ("0", "false", "off", "sync", "pread"):
        return _PLE_IO_PREAD
    if raw in ("1", "true", "on", "uring", "io_uring"):
        return _PLE_IO_URING
    raise ValueError(f"invalid {_PLE_IO_ENV}={raw!r}; expected auto, pread, or io_uring")


def _sglang_python_candidates() -> tuple[str, ...]:
    seen: set[str] = set()
    values: list[str] = []
    for raw in (
        os.environ.get("QSR_SGLANG_PYTHON_PATH"),
        os.environ.get("SGLANG_PYTHONPATH"),
    ):
        if not raw:
            continue
        for entry in raw.split(os.pathsep):
            path = os.fspath(pathlib.Path(entry).expanduser().resolve())
            if path in seen or not pathlib.Path(path).is_dir():
                continue
            seen.add(path)
            values.append(path)
    repo_candidate = pathlib.Path(__file__).resolve().parents[3].parent / "sglang" / "python"
    if repo_candidate.is_dir():
        path = os.fspath(repo_candidate.resolve())
        if path not in seen:
            values.append(path)
    return tuple(values)


def _load_io_uring_reader_type():
    """Load only SGLang's compiled storage module, without importing SGLang.

    Importing ``sglang`` itself mutates global Transformers/cache state and a
    partially imported incompatible installation can poison ``sys.modules``.
    The CPython extension is self-contained, so resolve it from Python roots
    and load it directly under the module name its ``PyInit`` symbol expects.
    """
    global _PLE_IO_URING_READER_TYPE

    if _PLE_IO_URING_READER_TYPE is not None:
        return _PLE_IO_URING_READER_TYPE
    roots = [*map(pathlib.Path, _sglang_python_candidates())]
    roots.extend(pathlib.Path(entry) for entry in sys.path if entry)
    seen: set[pathlib.Path] = set()
    load_errors: list[Exception] = []
    for root in roots:
        extension_dir = root / "sglang" / "srt" / "rust_extensions"
        try:
            extension_dir = extension_dir.resolve()
        except OSError:
            continue
        if extension_dir in seen or not extension_dir.is_dir():
            continue
        seen.add(extension_dir)
        paths = sorted(
            path
            for suffix in importlib.machinery.EXTENSION_SUFFIXES
            for path in extension_dir.glob(f"_storage*{suffix}")
            if _extension_matches_runtime(path)
        )
        for path in paths:
            try:
                spec = importlib.util.spec_from_file_location(
                    "sglang.srt.rust_extensions._storage", path
                )
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _PLE_IO_URING_READER_TYPE = getattr(module, "IoUringReader")
                return _PLE_IO_URING_READER_TYPE
            except Exception as exc:  # pragma: no cover - host/ABI dependent.
                load_errors.append(exc)
    if load_errors:
        raise load_errors[-1]
    raise ModuleNotFoundError("compiled SGLang io_uring storage extension was not found")


class _MappedShard:
    """Read-only row access to one safetensors FP8 shard via mmap."""

    def __init__(self, path: pathlib.Path, tensor_name: str) -> None:
        with open(path, "rb") as f:
            (header_len,) = __import__("struct").unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        meta = header[tensor_name]
        self.rows, self.dim = meta["shape"]
        self._row_bytes = self.dim  # one byte per fp8 element
        start, end = meta["data_offsets"]
        assert end - start == self.rows * self._row_bytes
        self._offset = 8 + header_len + start
        self._path = path
        self._file = path.open("rb")
        self._direct_fd: int | None = None
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    @property
    def fd(self) -> int:
        return self._file.fileno()

    @property
    def direct_fd(self) -> int:
        """A separate page-aligned direct-I/O descriptor for io_uring."""
        if self._direct_fd is None:
            direct = getattr(os, "O_DIRECT", 0)
            if not direct:
                raise OSError(errno.EOPNOTSUPP, "O_DIRECT is unavailable")
            self._direct_fd = os.open(self._path, os.O_RDONLY | direct)
        return self._direct_fd

    def close_direct(self) -> None:
        if self._direct_fd is not None:
            os.close(self._direct_fd)
            self._direct_fd = None

    def row_location(self, row: int) -> tuple[int, int]:
        if row < 0 or row >= self.rows:
            raise IndexError(f"PLE row {row} outside shard with {self.rows} rows")
        return self._offset + row * self._row_bytes, self._row_bytes

    def gather_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """``rows`` int64 1-D -> ``[len(rows), dim]`` uint8 (fp8 bits)."""
        idx = rows.numpy()
        buf = torch.frombuffer(self._map, dtype=torch.uint8)
        base = buf[self._offset : self._offset + self.rows * self._row_bytes]
        tiles = base.view(self.rows, self._row_bytes)
        return tiles[torch.from_numpy(idx)].clone()

    def all_rows(self) -> torch.Tensor:
        """Zero-copy uint8 view used only by the explicit resident loader."""
        buf = torch.frombuffer(self._map, dtype=torch.uint8)
        return buf[self._offset : self._offset + self.rows * self._row_bytes].view(
            self.rows, self._row_bytes
        )

    def close(self) -> None:
        self.close_direct()
        self._map.close()
        self._file.close()


class _PendingPleGather:
    """Future-backed PLE gather used to overlap NVMe reads with layer compute."""

    __slots__ = ("future", "shape")

    def __init__(
        self,
        future: Future[tuple[bytearray, tuple[int, int, int]]],
        shape: tuple[int, int, int],
    ) -> None:
        self.future = future
        self.shape = shape


class FlashNextPleTable:
    """The 128-shard FP8 n-gram table, streamed or resident."""

    def __init__(
        self,
        ckpt: pathlib.Path | str,
        layer_idx: int = 1,
        *,
        resident: bool = False,
        cache_rows: int = 131_072,
        cache_pages: int = 0,
        io_workers: int = 32,
    ) -> None:
        ckpt = pathlib.Path(ckpt)
        with open(ckpt / "model.safetensors.index.json") as f:
            weight_map = json.load(f)["weight_map"]
        prefix = _PLE_PREFIX.format(layer=layer_idx)
        self._load_buf = lambda name: self._tensor(ckpt, weight_map, f"{prefix}.{name}")  # noqa: E731
        self.weight_scale = float(
            self._load_buf("ple_embedding.ngram_embedding.weight_scale").float().item()
        )
        self.head_sizes = self._load_buf("ple_embedding.ngram_heads_vocab_sizes")
        self.head_offsets = self._load_buf("ple_embedding.ngram_heads_offsets")
        self.layer_multipliers = self._load_buf("ple_embedding.layer_multipliers")
        self.ngram_heads = int(self.head_sizes.numel())
        self.head_dim = 160

        shard_names = sorted(
            (k for k in weight_map if f"{prefix}.ple_embedding.ngram_embedding.shard_" in k),
            key=lambda k: int(k.rsplit("shard_", 1)[1].split(".")[0]),
        )
        self._shards: list[_MappedShard] = []
        for name in shard_names:
            self._shards.append(_MappedShard(ckpt / weight_map[name], name))
        self.shard_rows = self._shards[0].rows
        self.total_rows = self.shard_rows * len(self._shards)
        # Every checkpoint shard has the same fixed-width row layout.  Keep
        # the absolute data bases locally so the hot random-row path can
        # derive locations arithmetically instead of dispatching a Python
        # method/property pair for every miss.
        self._row_bytes = self._shards[0]._row_bytes
        self._shard_data_offsets = tuple(shard._offset for shard in self._shards)
        self._resident: torch.Tensor | None = None
        # FIFO row cache (n-gram locality) to avoid rotational-media random
        # reads (~5 ms/cold row). cache_rows=0 disables.
        self._cache_cap = int(cache_rows)
        self._cache_map: dict[int, int] = {}
        self._cache_keys: list[int | None] = [None] * self._cache_cap
        self._cache_rows: list[bytes | None] | None = None
        self._cache_next = 0
        self.cache_hits = 0
        self.cache_misses = 0
        if self._cache_cap > 0:
            self._cache_rows = [None] * self._cache_cap
        # Long prompts can stream millions of cold rows through the FIFO
        # cache and evict the small, repeated system prefix before the next
        # request arrives.  Keep a tiny hit-preserving tier populated only by
        # the first prefill chunk; it is deliberately bounded independently
        # of the cold-row capacity and stores the same FP8 bytes, not another
        # decoded tensor.  With cache_rows=0 the default tier is disabled too;
        # an explicit prefix-cap env var can still opt into just this tier.
        configured_prefix_cap = os.environ.get("QSR_FLASHNEXT_PLE_PREFIX_CACHE_ROWS")
        if configured_prefix_cap is None:
            configured_prefix_cap = str(min(65_536, max(0, self._cache_cap // 2)))
        self._prefix_cache_cap = max(0, int(configured_prefix_cap))
        self._prefix_rows: OrderedDict[int, bytes] = OrderedDict()
        self.prefix_row_hits = 0
        self.prefix_row_misses = 0
        # The storage device presents as rotational and the 160-byte rows are
        # distributed over a 47 GiB table.  Drive it at page granularity so
        # one verify batch becomes one high-QD read instead of a sequence of
        # synchronous mmap faults hidden inside torch advanced indexing.
        self._page_size = 4096
        # The real speculative trace has effectively no cross-round page
        # reuse once the much smaller row cache is enabled.  Keep this
        # optional rather than duplicating the OS page cache by default.
        self._page_cache_cap = max(0, int(cache_pages))
        self._page_cache: OrderedDict[tuple[int, int], bytes] = OrderedDict()
        self.page_cache_hits = 0
        self.page_cache_misses = 0
        self._io_workers = min(len(self._shards), max(1, int(io_workers)))
        self._io_pool: ThreadPoolExecutor | None = None
        # A single persistent foreground reader lets the model schedule the
        # next PLE lookup before it enters the first transformer layer.  The
        # page-read pool remains separate: it fans out over shards while this
        # executor keeps request ordering/cache mutation deterministic.
        self._prefetch_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="flashnext-ple-prefetch",
        )
        self._io_backend = _PLE_IO_PREAD
        self._io_mode = _PLE_IO_PREAD
        # A single io_uring ring leaves the NVMe queue under-filled when the
        # completion thread is copying a large batch of PyBytes objects.  A
        # small number of independent rings lets the kernel and the Python
        # bridge make progress concurrently.  Keep ``_uring_reader`` as the
        # compatibility alias used by the diagnostics/tests; production reads
        # use ``_uring_readers`` when it contains more than one ring.
        self._uring_reader = None
        self._uring_readers: tuple[object, ...] = ()
        self._uring_pool: ThreadPoolExecutor | None = None
        self._uring_pool_workers = 0
        self._closed = False
        self._uring_max_batch = max(
            1, int(os.environ.get("QSR_FLASHNEXT_PLE_IO_MAX_BATCH", "32768"))
        )
        self._uring_queue_depth = min(
            self._uring_max_batch,
            max(1, int(os.environ.get("QSR_FLASHNEXT_PLE_IO_QUEUE_DEPTH", "4096"))),
        )
        self._uring_reader_count = max(1, int(os.environ.get("QSR_FLASHNEXT_PLE_IO_READERS", "2")))
        self._large_batch_pread_pages = max(
            1,
            int(os.environ.get("QSR_FLASHNEXT_PLE_IO_LARGE_BATCH_PREAD_PAGES", "8192")),
        )
        self.pread_pages = 0
        self.io_uring_pages = 0
        # Diagnostic counters are opt-in because taking a clock around every
        # gather is measurable on the one-row decode path.  Prefill profiling
        # enables them automatically, so a cold request can distinguish hash,
        # page-I/O, and Python row assembly instead of attributing all of the
        # time to the target forward.
        self._profile_enabled = os.environ.get(
            "QSR_FLASHNEXT_PROFILE_PLE",
            os.environ.get("QSR_FLASHNEXT_PROFILE_PREFILL", "0"),
        ).strip().lower() in {"1", "true", "on"}
        self._profile_gather_ns = 0
        self._profile_lazy_factory_ns = 0
        self._profile_read_rows_ns = 0
        self._profile_page_load_ns = 0
        self._profile_page_io_ns = 0
        self._profile_row_assembly_ns = 0
        self._profile_rows_read = 0
        self._profile_gather_calls = 0
        self._profile_finish_wait_ns = 0
        self._profile_finish_calls = 0
        self._profile_finish_ready_calls = 0
        if not resident:
            self._io_mode = _resolve_ple_io_mode()
            self._init_stream_backend(self._io_mode)
        # Row/page caches and the double-buffered staging cursor are mutable.
        # Server-side request overlap may otherwise race a foreground gather
        # across requests.
        self._gather_lock = threading.Lock()
        self._stage_lock = threading.Lock()
        self._pinned_stages: list[torch.Tensor | None] = [None, None]
        self._stage_events: list[torch.cuda.Event | None] = [None, None]
        self._stage_index = 0
        if resident:
            self.make_resident()

    def _ensure_pread_pool(self) -> ThreadPoolExecutor:
        if self._io_pool is None:
            # Keep workers alive across decode steps.  Constructing an executor
            # per token both serialized thread start-up and limited the mmap page
            # faults to low queue depth; the table spans a virtual rotational
            # disk, where enough concurrent shard faults are essential.
            self._io_pool = ThreadPoolExecutor(max_workers=self._io_workers)
        return self._io_pool

    def _init_stream_backend(self, mode: str) -> None:
        if mode == _PLE_IO_PREAD:
            self._io_backend = _PLE_IO_PREAD
            self._ensure_pread_pool()
            return
        try:
            reader_type = _load_io_uring_reader_type()
            reader_count = min(len(self._shards), self._uring_reader_count)
            readers = tuple(
                reader_type(
                    self._uring_queue_depth,
                    self._uring_max_batch,
                    self._page_size,
                )
                for _ in range(reader_count)
            )
            self._uring_readers = readers
            self._uring_reader = readers[0]
            self._io_backend = _PLE_IO_URING
        except Exception:
            if mode == _PLE_IO_URING:
                raise
            self._uring_reader = None
            self._uring_readers = ()
            self._io_backend = _PLE_IO_PREAD
            self._ensure_pread_pool()

    @staticmethod
    def _should_fallback_io_uring(error: OSError) -> bool:
        code = getattr(error, "errno", None)
        return code in _PLE_IO_FALLBACK_ERRNOS

    @staticmethod
    def _tensor(ckpt: pathlib.Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
        from safetensors import safe_open

        with safe_open(str(ckpt / weight_map[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def make_resident(self) -> None:
        """Load the whole table into one pinned FP8 tensor (needs ~51 GB)."""
        if self._resident is not None:
            return
        required = self.total_rows * self.head_dim
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        if required > available:
            raise MemoryError(
                "PLE resident table requires "
                f"{required / 2**30:.2f} GiB but only {available / 2**30:.2f} GiB "
                "of host RAM is currently available; use stream mode"
            )
        # Allocate pinned storage directly.  Building a pageable 47.7-GiB
        # table and calling pin_memory() afterwards briefly required two full
        # copies, defeating the resident mode on otherwise adequate hosts.
        table = torch.empty(
            (self.total_rows, self.head_dim),
            dtype=torch.uint8,
            pin_memory=True,
        )
        for i, shard in enumerate(self._shards):
            table[i * shard.rows : (i + 1) * shard.rows].copy_(shard.all_rows())
        self._resident = table

    def _page_keys(self, shard_idx: int, row: int) -> tuple[tuple[int, int], ...]:
        offset, nbytes = self._shards[shard_idx].row_location(row)
        first = offset // self._page_size * self._page_size
        last = (offset + nbytes - 1) // self._page_size * self._page_size
        return tuple(
            (shard_idx, page_offset)
            for page_offset in range(first, last + self._page_size, self._page_size)
        )

    def _read_page_run(self, shard_idx: int, start: int, stop: int) -> dict[tuple[int, int], bytes]:
        size = stop - start
        if size <= 0:
            return {}
        chunk = os.pread(self._shards[shard_idx].fd, size, start)
        if not chunk:
            raise OSError(f"empty PLE page run: shard={shard_idx}, offset={start}, size={size}")
        pages: dict[tuple[int, int], bytes] = {}
        for page_offset, page_start in zip(
            range(start, stop, self._page_size),
            range(0, size, self._page_size),
            strict=True,
        ):
            page = chunk[page_start : page_start + self._page_size]
            if not page:
                raise OSError(f"PLE page run ended early: shard={shard_idx}, offset={page_offset}")
            pages[(shard_idx, page_offset)] = page
        return pages

    def _read_page_batch(
        self, shard_idx: int, offsets: Sequence[int]
    ) -> dict[tuple[int, int], bytes]:
        page_offsets = sorted(set(offsets))
        if not page_offsets:
            return {}
        pages: dict[tuple[int, int], bytes] = {}
        run_start = page_offsets[0]
        run_stop = run_start + self._page_size
        for offset in page_offsets[1:]:
            if offset == run_stop:
                run_stop += self._page_size
                continue
            pages.update(self._read_page_run(shard_idx, run_start, run_stop))
            run_start = offset
            run_stop = offset + self._page_size
        pages.update(self._read_page_run(shard_idx, run_start, run_stop))
        return pages

    def _load_pages(self, keys: Sequence[tuple[int, int]]) -> dict[tuple[int, int], bytes]:
        started = time.perf_counter_ns() if getattr(self, "_profile_enabled", False) else 0
        pages: dict[tuple[int, int], bytes] = {}
        misses: list[tuple[int, int]] = []
        for key in dict.fromkeys(keys):
            page = self._page_cache.get(key)
            if page is None:
                misses.append(key)
                self.page_cache_misses += 1
            else:
                self._page_cache.move_to_end(key)
                pages[key] = page
                self.page_cache_hits += 1
        if misses:
            misses.sort()
            io_started = time.perf_counter_ns() if started else 0
            try:
                use_io_uring = self._uring_reader is not None and not (
                    self._io_mode == _PLE_IO_AUTO and len(misses) >= self._large_batch_pread_pages
                )
                if use_io_uring:
                    self.io_uring_pages += len(misses)
                    loaded = self._load_pages_io_uring(misses)
                else:
                    self.pread_pages += len(misses)
                    loaded = self._load_pages_pread(misses)
            except OSError as error:
                if (
                    not use_io_uring
                    or self._uring_reader is None
                    or not self._should_fallback_io_uring(error)
                ):
                    raise
                self._uring_reader = None
                self._uring_readers = ()
                self._io_backend = _PLE_IO_PREAD
                uring_pool = getattr(self, "_uring_pool", None)
                if uring_pool is not None:
                    uring_pool.shutdown(wait=True)
                    self._uring_pool = None
                    self._uring_pool_workers = 0
                for shard in self._shards:
                    close_direct = getattr(shard, "close_direct", None)
                    if close_direct is not None:
                        close_direct()
                self.pread_pages += len(misses)
                loaded = self._load_pages_pread(misses)
            if io_started:
                self._profile_page_io_ns += time.perf_counter_ns() - io_started
            pages.update(loaded)
            if self._page_cache_cap:
                for key, page in loaded.items():
                    self._page_cache[key] = page
                    self._page_cache.move_to_end(key)
                    while len(self._page_cache) > self._page_cache_cap:
                        self._page_cache.popitem(last=False)
        if started:
            self._profile_page_load_ns += time.perf_counter_ns() - started
        return pages

    def _load_pages_pread(self, keys: Sequence[tuple[int, int]]) -> dict[tuple[int, int], bytes]:
        shard_batches: dict[int, list[int]] = {}
        for shard_idx, offset in keys:
            shard_batches.setdefault(shard_idx, []).append(offset)

        if len(shard_batches) == 1:
            shard_idx, offsets = next(iter(shard_batches.items()))
            loaded = [self._read_page_batch(shard_idx, offsets)]
        else:
            pool = self._ensure_pread_pool()
            futures = [
                pool.submit(self._read_page_batch, shard_idx, offsets)
                for shard_idx, offsets in shard_batches.items()
            ]
            loaded = [future.result() for future in futures]
        pages: dict[tuple[int, int], bytes] = {}
        for batch in loaded:
            pages.update(batch)
        return pages

    def _load_pages_io_uring(self, keys: Sequence[tuple[int, int]]) -> dict[tuple[int, int], bytes]:
        readers = getattr(self, "_uring_readers", ())
        if not readers:
            reader = getattr(self, "_uring_reader", None)
            readers = (reader,) if reader is not None else ()
        if not readers:
            raise RuntimeError("io_uring reader is not initialized")
        pages: dict[tuple[int, int], bytes] = {}
        deduped = list(dict.fromkeys(keys))

        def read_partition(reader, partition: Sequence[tuple[int, int]]):
            loaded: list[tuple[tuple[int, int], bytes]] = []
            for start in range(0, len(partition), self._uring_max_batch):
                batch_keys = partition[start : start + self._uring_max_batch]
                fds = [self._shards[shard_idx].direct_fd for shard_idx, _ in batch_keys]
                offsets = [offset for _, offset in batch_keys]
                batch_pages = reader.read_pages(fds, offsets)
                loaded.extend(
                    (key, bytes(page)) for key, page in zip(batch_keys, batch_pages, strict=True)
                )
            return loaded

        if len(readers) == 1:
            loaded_partitions = [read_partition(readers[0], deduped)]
        else:
            # Interleave sorted keys instead of assigning one contiguous
            # shard range to each ring.  Each ring then sees the same mixture
            # of files/offsets and the device scheduler can balance them; the
            # returned dictionary does not depend on completion order.
            partitions = [deduped[index :: len(readers)] for index in range(len(readers))]
            pool = getattr(self, "_uring_pool", None)
            workers = getattr(self, "_uring_pool_workers", 0)
            if pool is None or workers != len(readers):
                if pool is not None:
                    pool.shutdown(wait=True)
                pool = ThreadPoolExecutor(
                    max_workers=len(readers),
                    thread_name_prefix="flashnext-ple-uring",
                )
                self._uring_pool = pool
                self._uring_pool_workers = len(readers)
            futures = [
                pool.submit(read_partition, reader, partition)
                for reader, partition in zip(readers, partitions, strict=True)
            ]
            # Drain every future before propagating an error.  This guarantees
            # no ring is still touching its aligned buffer when a caller
            # falls back to pread and releases the reader objects.
            loaded_partitions = []
            first_error = None
            for future in futures:
                try:
                    loaded_partitions.append(future.result())
                except BaseException as error:  # pragma: no cover - kernel dependent
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error

        for loaded in loaded_partitions:
            for key, page in loaded:
                pages[key] = page
        return pages

    def _read_rows(self, row_ids: Sequence[int]) -> dict[int, bytes]:
        started = time.perf_counter_ns() if getattr(self, "_profile_enabled", False) else 0
        if started:
            self._profile_rows_read += len(row_ids)
        locations: list[tuple[int, int, int]] = []
        location_keys: list[tuple[tuple[int, int], ...]] = []
        for row_id in row_ids:
            if row_id < 0 or row_id >= self.total_rows:
                raise IndexError(f"PLE row {row_id} outside table with {self.total_rows} rows")
            shard_idx, local_row = divmod(row_id, self.shard_rows)
            offset = self._shard_data_offsets[shard_idx] + local_row * self._row_bytes
            nbytes = self._row_bytes
            locations.append((shard_idx, offset, nbytes))
            # Rows are 160 bytes in the current checkpoint, so they touch at
            # most two 4 KiB pages.  Derive the keys from the location we just
            # computed instead of calling row_location a second time through
            # _page_keys; retaining the tuple shape keeps the generic page
            # loader and its cache semantics unchanged.
            first = offset // self._page_size * self._page_size
            last = (offset + nbytes - 1) // self._page_size * self._page_size
            first_key = (shard_idx, first)
            if first == last:
                location_keys.append((first_key,))
            else:
                location_keys.append((first_key, (shard_idx, last)))
        pages = self._load_pages([key for keys in location_keys for key in keys])

        assembly_started = time.perf_counter_ns() if started else 0
        output: dict[int, bytes] = {}
        for row_id, (_, offset, nbytes), keys in zip(
            row_ids, locations, location_keys, strict=True
        ):
            if len(keys) == 1:
                key = keys[0]
                page = pages[key]
                within_page = offset - key[1]
                if within_page < 0 or within_page + nbytes > len(page):
                    raise OSError(f"PLE page does not cover row {row_id} at byte {offset}")
                # A direct slice avoids the temporary one-element list and
                # b"".join allocation on the overwhelmingly common path.
                output[row_id] = page[within_page : within_page + nbytes]
                continue
            if len(keys) == 2:
                first_key, second_key = keys
                first_page = pages[first_key]
                within_page = offset - first_key[1]
                first_nbytes = min(nbytes, len(first_page) - within_page)
                if within_page < 0 or first_nbytes <= 0:
                    raise OSError(f"PLE page does not cover row {row_id} at byte {offset}")
                remaining = nbytes - first_nbytes
                second_page = pages[second_key]
                if remaining > len(second_page):
                    raise OSError(f"incomplete PLE row {row_id}: {remaining} bytes remain")
                output[row_id] = (
                    first_page[within_page : within_page + first_nbytes] + second_page[:remaining]
                )
                continue
            remaining = nbytes
            cursor = offset
            parts = []
            for key in keys:
                page = pages[key]
                within_page = cursor - key[1]
                available = min(remaining, len(page) - within_page)
                if available <= 0:
                    raise OSError(f"PLE page does not cover row {row_id} at byte {cursor}")
                parts.append(page[within_page : within_page + available])
                cursor += available
                remaining -= available
            if remaining:
                raise OSError(f"incomplete PLE row {row_id}: {remaining} bytes remain")
            output[row_id] = b"".join(parts)
        if assembly_started:
            self._profile_row_assembly_ns += time.perf_counter_ns() - assembly_started
        if started:
            self._profile_read_rows_ns += time.perf_counter_ns() - started
        return output

    def _to_output(
        self,
        raw: bytearray,
        shape: tuple[int, int, int],
        device: torch.device | str | None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with self._stage_lock:
            return self._to_output_unlocked(raw, shape, device, out=out)

    def _to_output_unlocked(
        self,
        raw: bytearray,
        shape: tuple[int, int, int],
        device: torch.device | str | None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source = torch.frombuffer(raw, dtype=torch.uint8)
        target = out.view(shape) if out is not None else None
        if target is not None and device is None:
            device = target.device
        if device is None or torch.device(device).type == "cpu":
            if target is not None:
                target.copy_(source.view(torch.float8_e4m3fn).view(shape)).mul_(self.weight_scale)
                return out
            return (source.view(torch.float8_e4m3fn).to(torch.bfloat16) * self.weight_scale).view(
                shape
            )
        else:
            stage_idx = self._stage_index
            self._stage_index = (stage_idx + 1) % len(self._pinned_stages)
            event = self._stage_events[stage_idx]
            if event is not None:
                event.synchronize()
            stage = self._pinned_stages[stage_idx]
            if stage is None or stage.numel() < source.numel():
                stage = torch.empty(source.numel(), dtype=torch.uint8, pin_memory=True)
                self._pinned_stages[stage_idx] = stage
            stage[: source.numel()].copy_(source)
            device_bytes = stage[: source.numel()].to(device, non_blocking=True)
            if event is None:
                event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(torch.device(device)))
            self._stage_events[stage_idx] = event
            if target is not None:
                target.copy_((device_bytes.view(torch.float8_e4m3fn)).view(shape)).mul_(
                    self.weight_scale
                )
                return out
            return (
                device_bytes.view(torch.float8_e4m3fn).to(torch.bfloat16) * self.weight_scale
            ).view(shape)

    def _gather_raw_unlocked(
        self,
        ids: torch.Tensor,
        *,
        cache_hint: bool = False,
    ) -> tuple[bytearray, tuple[int, int, int]]:
        """Resolve/cache rows and return their checkpoint FP8 bytes."""
        started = time.perf_counter_ns() if getattr(self, "_profile_enabled", False) else 0
        t = ids.shape[0]
        flat_ids = ids.reshape(-1).long().tolist()
        shape = (t, self.ngram_heads, self.head_dim)
        if self._resident is not None:
            flat = torch.tensor(flat_ids, dtype=torch.long)
            raw = bytearray(self._resident[flat].contiguous().numpy().tobytes())
            return raw, shape

        rows: dict[int, bytes] = {}
        misses: list[int] = []
        for rid in flat_ids:
            protected = self._prefix_rows.get(rid)
            if protected is not None:
                self._prefix_rows.move_to_end(rid)
                rows[rid] = protected
                self.cache_hits += 1
                self.prefix_row_hits += 1
                continue
            if self._prefix_cache_cap:
                self.prefix_row_misses += 1
            if self._cache_rows is not None:
                slot = self._cache_map.get(rid)
                if slot is not None:
                    row = self._cache_rows[slot]
                    assert row is not None
                    rows[rid] = row
                    self.cache_hits += 1
                else:
                    misses.append(rid)
                    self.cache_misses += 1
            else:
                misses.append(rid)
                self.cache_misses += 1
        unique_misses = list(dict.fromkeys(misses))
        if unique_misses:
            loaded = self._read_rows(unique_misses)
            rows.update(loaded)
            if cache_hint and self._prefix_cache_cap:
                for rid in unique_misses:
                    self._prefix_rows[rid] = loaded[rid]
                    self._prefix_rows.move_to_end(rid)
                    while len(self._prefix_rows) > self._prefix_cache_cap:
                        self._prefix_rows.popitem(last=False)
            if self._cache_rows is not None:
                for rid in unique_misses:
                    slot = self._cache_next
                    self._cache_next = (slot + 1) % self._cache_cap
                    evicted = self._cache_keys[slot]
                    if evicted is not None and self._cache_map.get(evicted) == slot:
                        del self._cache_map[evicted]
                    self._cache_rows[slot] = loaded[rid]
                    self._cache_keys[slot] = rid
                    self._cache_map[rid] = slot
        raw = bytearray(b"".join(rows[rid] for rid in flat_ids))
        if started:
            self._profile_gather_calls += 1
            self._profile_gather_ns += time.perf_counter_ns() - started
        return raw, shape

    def _gather_raw_prefetch(
        self,
        ids: torch.Tensor,
        *,
        cache_hint: bool = False,
    ) -> tuple[bytearray, tuple[int, int, int]]:
        """Read rows on the persistent prefetch worker.

        Cache and page-LRU mutation use the same lock as synchronous gathers;
        keeping the lock in the worker (rather than around ``submit``) is what
        allows the caller to run the first model layer while NVMe is in flight.
        """
        with self._gather_lock:
            return self._gather_raw_unlocked(ids, cache_hint=cache_hint)

    def _lazy_gather_prefetch(
        self,
        ids_factory: Callable[[], torch.Tensor],
        *,
        cache_hint: bool = False,
    ) -> tuple[bytearray, tuple[int, int, int]]:
        """Build n-gram ids and read their rows entirely off the caller thread."""
        started = time.perf_counter_ns() if getattr(self, "_profile_enabled", False) else 0
        ids = ids_factory()
        if started:
            self._profile_lazy_factory_ns += time.perf_counter_ns() - started
        if not torch.is_tensor(ids) or ids.ndim != 2:
            shape = getattr(ids, "shape", None)
            raise ValueError(f"PLE gather factory must return [T, heads] ids, got {shape}")
        ids_cpu = ids.detach().to(device="cpu", dtype=torch.long).contiguous()
        return self._gather_raw_prefetch(ids_cpu, cache_hint=cache_hint)

    def start_gather(self, ids: torch.Tensor, *, cache_hint: bool = False) -> _PendingPleGather:
        """Schedule a PLE gather and return before its rows are available.

        ``ids`` is copied to CPU before submission so the worker never holds a
        CUDA tensor or a caller-owned storage view.  The paired
        :meth:`finish_gather` performs the existing pinned staging and FP8
        conversion exactly once, preserving the synchronous path's bytes.
        """
        if ids.ndim != 2:
            raise ValueError(f"PLE gather expects [T, heads] ids, got {tuple(ids.shape)}")
        with self._gather_lock:
            if self._closed:
                raise RuntimeError("PLE table is closed")
        ids_cpu = ids.detach().to(device="cpu", dtype=torch.long).contiguous()
        shape = (int(ids_cpu.shape[0]), self.ngram_heads, self.head_dim)
        # Submit while holding the same lifecycle lock used by ``close``.  A
        # close racing this call therefore either waits for the submission or
        # rejects it cleanly; it can never replace the executor between the
        # closed check and ``submit``.
        with self._gather_lock:
            if self._closed or self._prefetch_pool is None:
                raise RuntimeError("PLE table is closed")
            future = self._prefetch_pool.submit(
                self._gather_raw_prefetch,
                ids_cpu,
                cache_hint=cache_hint,
            )
        return _PendingPleGather(future, shape)

    def start_gather_lazy(
        self,
        ids_factory: Callable[[], torch.Tensor],
        *,
        token_count: int,
        cache_hint: bool = False,
    ) -> _PendingPleGather:
        """Schedule hash computation and the following row gather together.

        The factory must be side-effect free and return CPU or CUDA ids with
        shape ``[token_count, ngram_heads]``.  Keeping the factory on the
        persistent worker lets token-major prefill overlap CPU hash work,
        page reads, and the embedding/first-layer GPU work instead of doing
        the hash synchronously before the overlap window opens.
        """
        if token_count <= 0:
            raise ValueError(f"PLE gather token_count must be positive, got {token_count}")
        with self._gather_lock:
            if self._closed or self._prefetch_pool is None:
                raise RuntimeError("PLE table is closed")
            shape = (int(token_count), self.ngram_heads, self.head_dim)
            future = self._prefetch_pool.submit(
                self._lazy_gather_prefetch,
                ids_factory,
                cache_hint=cache_hint,
            )
        return _PendingPleGather(future, shape)

    def finish_gather(
        self,
        pending: _PendingPleGather,
        *,
        device: torch.device | str | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Wait for :meth:`start_gather` and materialise its BF16 output."""
        started = time.perf_counter_ns() if getattr(self, "_profile_enabled", False) else 0
        ready = pending.future.done() if started else False
        raw, shape = pending.future.result()
        if started:
            self._profile_finish_calls += 1
            self._profile_finish_wait_ns += time.perf_counter_ns() - started
            if ready:
                self._profile_finish_ready_calls += 1
        if shape != pending.shape:
            raise RuntimeError(
                f"PLE gather shape changed while reading: {shape} != {pending.shape}"
            )
        return self._to_output(raw, shape, device, out=out)

    def gather(
        self,
        ids: torch.Tensor,
        *,
        device: torch.device | str | None = None,
        out: torch.Tensor | None = None,
        cache_hint: bool = False,
    ) -> torch.Tensor:
        """``ids`` ``[T, ngram_heads]`` global row ids -> ``[T, heads, 160]`` bf16.

        Cold rows are reconstructed from deduplicated 4 KiB pages read in one
        persistent high-QD batch.  CPU output keeps FP8 conversion local;
        CUDA output stages raw bytes in pinned memory and converts on-device.
        """
        with self._gather_lock:
            if self._closed:
                raise RuntimeError("PLE table is closed")
            raw, shape = self._gather_raw_unlocked(ids, cache_hint=cache_hint)
            return self._to_output(raw, shape, device, out=out)

    def stats_snapshot(self) -> dict[str, int | float | str | bool]:
        """Return a point-in-time view of PLE cache and I/O activity.

        The counters are updated while holding ``_gather_lock``.  Reading the
        integer values without taking that lock is safe for the diagnostic
        endpoint (and avoids delaying a decode gather); a snapshot can be off
        by one in the middle of a gather, but it is never a fabricated rate.
        """
        row_total = self.cache_hits + self.cache_misses
        page_total = self.page_cache_hits + self.page_cache_misses
        return {
            "resident": self._resident is not None,
            "row_cache_capacity": self._cache_cap,
            "row_cache_entries": len(self._cache_map),
            "row_cache_hits": self.cache_hits,
            "row_cache_misses": self.cache_misses,
            "row_cache_hit_rate": self.cache_hits / row_total if row_total else 0.0,
            "prefix_row_cache_capacity": self._prefix_cache_cap,
            "prefix_row_cache_entries": len(self._prefix_rows),
            "prefix_row_cache_hits": self.prefix_row_hits,
            "prefix_row_cache_misses": self.prefix_row_misses,
            "page_cache_capacity": self._page_cache_cap,
            "page_cache_entries": len(self._page_cache),
            "page_cache_hits": self.page_cache_hits,
            "page_cache_misses": self.page_cache_misses,
            "page_cache_hit_rate": self.page_cache_hits / page_total if page_total else 0.0,
            "pread_pages": self.pread_pages,
            "io_uring_pages": self.io_uring_pages,
            "io_backend": self._io_backend,
            "io_uring_readers": len(getattr(self, "_uring_readers", ()))
            or int(getattr(self, "_uring_reader", None) is not None),
            "io_uring_queue_depth": getattr(self, "_uring_queue_depth", 0),
            "io_uring_max_batch": getattr(self, "_uring_max_batch", 0),
            "profile_gather_calls": getattr(self, "_profile_gather_calls", 0),
            "profile_gather_ns": getattr(self, "_profile_gather_ns", 0),
            "profile_lazy_factory_ns": getattr(self, "_profile_lazy_factory_ns", 0),
            "profile_read_rows_ns": getattr(self, "_profile_read_rows_ns", 0),
            "profile_page_load_ns": getattr(self, "_profile_page_load_ns", 0),
            "profile_page_io_ns": getattr(self, "_profile_page_io_ns", 0),
            "profile_row_assembly_ns": getattr(self, "_profile_row_assembly_ns", 0),
            "profile_rows_read": getattr(self, "_profile_rows_read", 0),
            "profile_finish_wait_ns": getattr(self, "_profile_finish_wait_ns", 0),
            "profile_finish_calls": getattr(self, "_profile_finish_calls", 0),
            "profile_finish_ready_calls": getattr(self, "_profile_finish_ready_calls", 0),
        }

    def close(self) -> None:
        with self._gather_lock:
            if self._closed:
                return
            # Mark the table closed before waiting for the worker.  The
            # worker intentionally does not re-check this flag: a gather that
            # was already submitted must drain before its file descriptors
            # and page cache are released.
            self._closed = True
        self._prefetch_pool.shutdown(wait=True)
        self._prefetch_pool = None
        if self._io_pool is not None:
            self._io_pool.shutdown(wait=True)
            self._io_pool = None
        with self._gather_lock:
            self._uring_reader = None
            self._uring_readers = ()
            if self._uring_pool is not None:
                self._uring_pool.shutdown(wait=True)
                self._uring_pool = None
                self._uring_pool_workers = 0
            for event in self._stage_events:
                if event is not None:
                    event.synchronize()
            self._resident = None
            self._cache_map.clear()
            self._cache_keys.clear()
            if self._cache_rows is not None:
                self._cache_rows.clear()
                self._cache_rows = None
            self._page_cache.clear()
            self._prefix_rows.clear()
            self._pinned_stages.clear()
            self._stage_events.clear()
            for shard in self._shards:
                shard.close()


class FlashNextPleHasher:
    """EOS-aware n-gram window hashing (checkpoint-provided constants)."""

    def __init__(self, table: FlashNextPleTable, eos_token_id: int, ngram_size: int = 3) -> None:
        self.multipliers = table.layer_multipliers.long()
        self.head_sizes = table.head_sizes.long()
        self.head_offsets = table.head_offsets.long()
        self.eos = int(eos_token_id)
        self.ngram_size = ngram_size
        self.heads_per_ngram = table.ngram_heads // (ngram_size - 1)

    def _shift(self, tokens: torch.Tensor, n: int) -> torch.Tensor:
        """``tokens [S]`` -> token ``n`` steps back, EOS-filled before the
        segment start (segment = after the previous EOS)."""
        if n == 0:
            return tokens
        s = tokens.shape[0]
        idx = torch.arange(s, device=tokens.device)
        eos_pos = torch.where(tokens == self.eos, idx, idx.new_full((), -1))
        prev_incl = torch.cummax(eos_pos, dim=0).values
        prev = torch.cat([eos_pos.new_full((1,), -1), prev_incl[:-1]])
        seg_start = prev + 1
        pos_in_seg = idx - seg_start
        src = idx - n
        shifted = tokens[torch.clamp(src, min=0)]
        valid = (pos_in_seg >= n) & (src >= 0)
        return torch.where(valid, shifted, tokens.new_full((), self.eos))

    def sequence_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens [S]`` (one request) -> ``[S, ngram_heads]`` global ids."""
        tokens = tokens.long()
        multipliers = self.multipliers.to(tokens.device)
        head_sizes = self.head_sizes.to(tokens.device)
        head_offsets = self.head_offsets.to(tokens.device)
        shifts = [self._shift(tokens, i) for i in range(self.ngram_size)]
        blocks = []
        for n in range(2, self.ngram_size + 1):
            mix = shifts[0] * multipliers[0]
            for pos in range(1, n):
                mix = torch.bitwise_xor(mix, shifts[pos] * multipliers[pos])
            start = (n - 2) * self.heads_per_ngram
            sizes = head_sizes[start : start + self.heads_per_ngram]
            offs = head_offsets[start : start + self.heads_per_ngram]
            ids = torch.remainder(mix.unsqueeze(-1), sizes.view(1, -1)) + offs.view(1, -1)
            blocks.append(ids)
        return torch.cat(blocks, dim=-1)

    def decode_ids(self, window: Sequence[int]) -> torch.Tensor:
        """``[t-2, t-1, t]`` context -> ``[1, ngram_heads]``."""
        return self.sequence_ids(torch.tensor(list(window), dtype=torch.long))[-1:]


class FlashNextPLELayer(nn.Module):
    """Key/value projection + gated dilated conv injection (plain torch)."""

    def __init__(
        self,
        ckpt: pathlib.Path | str,
        table: FlashNextPleTable,
        layer_idx: int = 1,
        *,
        hc_count: int = 4,
        hidden_size: int = 2560,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        ckpt = pathlib.Path(ckpt)
        with open(ckpt / "model.safetensors.index.json") as f:
            weight_map = json.load(f)["weight_map"]
        prefix = _PLE_PREFIX.format(layer=layer_idx)

        def load(name: str) -> torch.Tensor:
            key = f"{prefix}.{name}"
            from safetensors import safe_open

            with safe_open(str(ckpt / weight_map[key]), framework="pt", device="cpu") as f:
                return f.get_tensor(key)

        self.table = table
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.conv_kernel = 4
        self.conv_dilation = 3
        self.state_len = (self.conv_kernel - 1) * self.conv_dilation
        hc_hidden = hc_count * hidden_size

        self.key_proj = nn.Parameter(load("key_proj.weight").to(dtype))
        self.value_proj = nn.Parameter(load("value_proj.weight").to(dtype))
        self.norm_key = GroupedGemmaRMSNorm(hc_hidden, eps=eps, group_size=hidden_size)
        self.norm_query = GroupedGemmaRMSNorm(hc_hidden, eps=eps, group_size=hidden_size)
        self.norm_conv = GroupedGemmaRMSNorm(hc_hidden, eps=eps, group_size=hidden_size)
        self.norm_key.weight = nn.Parameter(load("norm_key.weight").to(dtype))
        self.norm_query.weight = nn.Parameter(load("norm_query.weight").to(dtype))
        self.norm_conv.weight = nn.Parameter(load("norm_conv.weight").to(dtype))
        conv_w = load("conv1d.weight").to(dtype)
        self.register_buffer("conv_weight", conv_w)
        self.to(dtype)

    def embed(
        self,
        ids: torch.Tensor,
        *,
        device: torch.device | str | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``ids [T, heads]`` -> flattened embedding ``[T, 2560]`` bf16."""
        if out is None:
            rows = self.table.gather(ids, device=device)
            return rows.flatten(start_dim=-2)
        return self.table.gather(ids, device=device, out=out)

    def _norm3(self, norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = norm(x.flatten(-2, -1))
        return y.unflatten(-1, (self.hc_count, self.hidden_size))

    def inject(self, embeddings: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """``embeddings [T,2560]``, ``hidden [T, hc*hs]`` -> ``[T, hc*hs]``."""
        t = hidden.shape[0]
        # These are separate projections in the reference graph.  Fusing
        # them changes cuBLAS' reduction order (observed non-bit-exact even
        # for M=1 and up to 0.0625 at M=4), which compounds through PLE.
        key = torch.nn.functional.linear(embeddings, self.key_proj)
        value = torch.nn.functional.linear(embeddings, self.value_proj)
        key = key.view(t, self.hc_count, self.hidden_size)
        query = hidden.view(t, self.hc_count, self.hidden_size)
        key_n = self._norm3(self.norm_key, key)
        query_n = self._norm3(self.norm_query, query)
        gate = (key_n * query_n).sum(dim=-1, keepdim=True) / (self.hidden_size**0.5)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = torch.sigmoid(gate.float()).to(hidden.dtype)
        gated = gate * value.unsqueeze(-2)
        gated_flat = gated.flatten(-2)
        normed_flat = self._norm3(self.norm_conv, gated).flatten(-2)
        return gated_flat, normed_flat

    def prefill_conv(self, normed_flat: torch.Tensor, seq_lens: Sequence[int]) -> torch.Tensor:
        """Causal dilated depthwise conv over concatenated per-request rows."""
        pad = self.state_len
        outs = []
        pos = 0
        for length in seq_lens:
            x = normed_flat[pos : pos + length].t().unsqueeze(0)  # [1, C, L]
            xp = torch.nn.functional.pad(x, (pad, 0))
            y = torch.nn.functional.conv1d(
                xp, self.conv_weight, groups=x.shape[1], dilation=self.conv_dilation
            )
            # Qwen4-Exp applies SiLU to the depthwise-convolution branch
            # before adding it to the gated value.  Keeping the activation
            # inside every conv entry point prevents eager decode, prefill,
            # and speculative verify from silently implementing different
            # PLE equations.
            outs.append(torch.nn.functional.silu(y.squeeze(0).t()))
            pos += length
        return torch.cat(outs, dim=0)

    def prefill_conv_with_state(
        self,
        normed_flat: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one prompt chunk without copying history beside every row.

        Only the first ``state_len`` outputs can observe the incoming history.
        Compute the long causal body directly from the chunk, then replace that
        short prefix with the history-aware result.  This preserves the exact
        convolution equation while avoiding ``cat([state, chunk])`` for the
        full sequence.
        """
        from runtime.kernels.gdn_conv import fused_causal_conv_silu

        if normed_flat.ndim != 2 or normed_flat.shape[0] == 0:
            raise ValueError(
                f"PLE stateful prefill expects non-empty [T,C], got {tuple(normed_flat.shape)}"
            )
        sequence = normed_flat.t().unsqueeze(0)
        seq_len = int(sequence.shape[-1])

        if seq_len <= self.state_len:
            conv_input = torch.cat([state, sequence], dim=-1)
            output = fused_causal_conv_silu(
                conv_input,
                self.conv_weight,
                padding=0,
                out_len=seq_len,
                dilation=self.conv_dilation,
            )
            if output is None:
                output = torch.nn.functional.silu(
                    torch.nn.functional.conv1d(
                        conv_input,
                        self.conv_weight,
                        groups=conv_input.shape[1],
                        dilation=self.conv_dilation,
                    )
                )
            state.copy_(conv_input[:, :, -self.state_len :])
            return output.squeeze(0).t()

        output = fused_causal_conv_silu(
            sequence,
            self.conv_weight,
            padding=self.state_len,
            out_len=seq_len,
            dilation=self.conv_dilation,
        )
        if output is None:
            output = torch.nn.functional.silu(
                torch.nn.functional.conv1d(
                    sequence,
                    self.conv_weight,
                    padding=self.state_len,
                    groups=sequence.shape[1],
                    dilation=self.conv_dilation,
                )[:, :, :seq_len]
            )

        prefix_input = torch.cat(
            [state, sequence[:, :, : self.state_len]],
            dim=-1,
        )
        prefix = fused_causal_conv_silu(
            prefix_input,
            self.conv_weight,
            padding=0,
            out_len=self.state_len,
            dilation=self.conv_dilation,
        )
        if prefix is None:
            prefix = torch.nn.functional.silu(
                torch.nn.functional.conv1d(
                    prefix_input,
                    self.conv_weight,
                    groups=prefix_input.shape[1],
                    dilation=self.conv_dilation,
                )
            )
        output[:, :, : self.state_len].copy_(prefix)
        state.copy_(sequence[:, :, -self.state_len :])
        return output.squeeze(0).t()

    def decode_conv(
        self, normed_flat: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One token per row; ``state [T, C, state_len]`` -> conv out + new state."""
        conv_input = torch.cat([state, normed_flat.unsqueeze(-1)], dim=-1)
        y = torch.nn.functional.conv1d(
            conv_input, self.conv_weight, groups=conv_input.shape[1], dilation=self.conv_dilation
        )
        new_state = torch.cat([state[:, :, 1:], normed_flat.unsqueeze(-1)], dim=-1)
        return torch.nn.functional.silu(y.squeeze(-1)), new_state

    def decode_conv_inplace(self, normed_flat: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """CUDA-Graph-safe: conv over ``[state, new]`` then shift ``state`` in
        place, so the state tensor keeps a fixed address across replays."""
        conv_input = torch.cat([state, normed_flat.unsqueeze(-1)], dim=-1)
        y = torch.nn.functional.conv1d(
            conv_input, self.conv_weight, groups=conv_input.shape[1], dilation=self.conv_dilation
        )
        state.copy_(torch.cat([state[:, :, 1:], normed_flat.unsqueeze(-1)], dim=-1))
        return torch.nn.functional.silu(y.squeeze(-1))

    def spec_conv(
        self,
        normed_flat: torch.Tensor,
        state: torch.Tensor,
        state_rows: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Advance a verify block while retaining every commit candidate.

        PLE is recurrent through its dilated-convolution window.  A partial
        speculative accept must restore the window after the accepted row,
        not after all candidates.  ``state_rows`` supplies fixed-address
        destinations for CUDA Graph capture; eager callers may omit it.
        The live ``state`` is read-only.
        """
        rows = normed_flat.shape[0]
        if state_rows is None:
            state_rows = [torch.empty_like(state) for _ in range(rows)]
        if len(state_rows) != rows:
            raise ValueError(
                f"PLE spec state rows must match verify rows: {len(state_rows)} != {rows}"
            )
        # A verify block is one causal sequence, so all candidate outputs are
        # one depthwise convolution over ``[live history, candidate rows]``.
        # The previous implementation launched one 10,240-group convolution
        # per row.  This preserves the exact recurrence while reducing K+1
        # launches to one; snapshots are overlapping fixed-address views of
        # the same combined history and remain available for partial commit.
        sequence = normed_flat.t().unsqueeze(0)
        conv_input = torch.cat([state, sequence], dim=-1)
        outputs = torch.nn.functional.silu(
            torch.nn.functional.conv1d(
                conv_input,
                self.conv_weight,
                groups=conv_input.shape[1],
                dilation=self.conv_dilation,
            )
            .squeeze(0)
            .t()
        )
        for row, destination in enumerate(state_rows):
            destination.copy_(conv_input[:, :, row + 1 : row + 1 + self.state_len])
        return outputs, state_rows
