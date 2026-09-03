from __future__ import annotations

import errno
import fcntl
import json
import os
import pathlib
import struct
import sys
import tempfile
from collections import OrderedDict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from runtime.model.flashnext.ple import (  # noqa: E402
    _PLE_IO_AUTO,
    _PLE_IO_ENV,
    _PLE_IO_PREAD,
    _PLE_IO_URING,
    FlashNextPleTable,
    _extension_matches_runtime,
    _load_io_uring_reader_type,
    _MappedShard,
    _resolve_ple_io_mode,
)


def test_io_uring_loader_rejects_foreign_cpython_extension():
    current = f"{sys.version_info.major}{sys.version_info.minor}"
    foreign = "312" if current != "312" else "314"
    assert not _extension_matches_runtime(pathlib.Path(f"_storage.cpython-{foreign}-x86_64.so"))
    assert _extension_matches_runtime(pathlib.Path("_storage.abi3.so"))
    assert _extension_matches_runtime(pathlib.Path("_storage.so"))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, _PLE_IO_AUTO),
        ("", _PLE_IO_AUTO),
        ("auto", _PLE_IO_AUTO),
        ("pread", _PLE_IO_PREAD),
        ("off", _PLE_IO_PREAD),
        ("io_uring", _PLE_IO_URING),
        ("uring", _PLE_IO_URING),
    ],
)
def test_resolve_ple_io_mode(value, expected, monkeypatch):
    if value is None:
        monkeypatch.delenv(_PLE_IO_ENV, raising=False)
    else:
        monkeypatch.setenv(_PLE_IO_ENV, value)
    assert _resolve_ple_io_mode() == expected


def test_resolve_ple_io_mode_rejects_invalid(monkeypatch):
    monkeypatch.setenv(_PLE_IO_ENV, "bogus")
    with pytest.raises(ValueError):
        _resolve_ple_io_mode()


def test_load_pages_falls_back_to_pread_on_io_uring_availability_error():
    table = object.__new__(FlashNextPleTable)
    table._page_cache = OrderedDict()
    table._page_cache_cap = 0
    table.page_cache_hits = 0
    table.page_cache_misses = 0
    table._io_backend = _PLE_IO_URING
    table._io_mode = _PLE_IO_URING
    table._uring_max_batch = 4
    table._large_batch_pread_pages = 8192
    table.pread_pages = 0
    table.io_uring_pages = 0
    table._shards = [
        SimpleNamespace(fd=-1, direct_fd=-1, close_direct=lambda: None),
        SimpleNamespace(fd=-1, direct_fd=-1, close_direct=lambda: None),
    ]
    table._uring_reader = SimpleNamespace(
        read_pages=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError(errno.ENOSYS, "io_uring unavailable")
        )
    )
    table._load_pages_pread = lambda keys: {key: b"pread" for key in keys}

    loaded = table._load_pages([(0, 0), (1, 4096)])

    assert loaded == {(0, 0): b"pread", (1, 4096): b"pread"}
    assert table._io_backend == _PLE_IO_PREAD
    assert table._uring_reader is None
    assert table.io_uring_pages == 2
    assert table.pread_pages == 2


def test_auto_mode_uses_coalesced_pread_for_large_page_batches():
    table = object.__new__(FlashNextPleTable)
    table._page_cache = OrderedDict()
    table._page_cache_cap = 0
    table.page_cache_hits = 0
    table.page_cache_misses = 0
    table._io_backend = _PLE_IO_URING
    table._io_mode = _PLE_IO_AUTO
    table._uring_reader = SimpleNamespace()
    table._large_batch_pread_pages = 2
    table.pread_pages = 0
    table.io_uring_pages = 0
    table._load_pages_io_uring = lambda _keys: pytest.fail("large batch used io_uring")
    table._load_pages_pread = lambda keys: {key: b"pread" for key in keys}

    loaded = table._load_pages([(0, 0), (0, 4096)])

    assert loaded == {(0, 0): b"pread", (0, 4096): b"pread"}
    assert table.pread_pages == 2
    assert table.io_uring_pages == 0


def test_auto_large_batch_pread_error_does_not_disable_io_uring():
    table = object.__new__(FlashNextPleTable)
    table._page_cache = OrderedDict()
    table._page_cache_cap = 0
    table.page_cache_hits = 0
    table.page_cache_misses = 0
    table._io_backend = _PLE_IO_URING
    table._io_mode = _PLE_IO_AUTO
    reader = SimpleNamespace()
    table._uring_reader = reader
    table._large_batch_pread_pages = 1
    table.pread_pages = 0
    table.io_uring_pages = 0
    table._load_pages_pread = lambda _keys: (_ for _ in ()).throw(
        OSError(errno.ENOSYS, "pread failure")
    )

    with pytest.raises(OSError, match="pread failure"):
        table._load_pages([(0, 0)])

    assert table._uring_reader is reader
    assert table._io_backend == _PLE_IO_URING


@pytest.mark.parametrize("row_id", [-1, 20])
def test_read_rows_rejects_out_of_range_ids(row_id):
    table = object.__new__(FlashNextPleTable)
    table.total_rows = 20
    table.shard_rows = 10
    table._shards = [SimpleNamespace(), SimpleNamespace()]
    with pytest.raises(IndexError, match="outside table"):
        table._read_rows([row_id])


def test_make_resident_rejects_insufficient_host_memory(monkeypatch):
    table = object.__new__(FlashNextPleTable)
    table._resident = None
    table.total_rows = 1024
    table.head_dim = 1024
    monkeypatch.setattr(os, "sysconf", lambda _name: 1)
    monkeypatch.setattr(torch, "empty", lambda *_args, **_kwargs: pytest.fail("allocated table"))

    with pytest.raises(MemoryError, match="use stream mode"):
        table.make_resident()


def test_load_pages_io_uring_reads_pages_in_requested_order():
    try:
        reader_type = _load_io_uring_reader_type()
        reader = reader_type(queue_depth=2, max_batch=4, page_size=4096)
    except ModuleNotFoundError as error:
        pytest.skip(f"sglang io_uring extension unavailable: {error}")
    except OSError as error:
        if error.errno in (errno.ENOSYS, errno.EPERM, errno.EOPNOTSUPP):
            pytest.skip(f"io_uring unavailable on this host: {error}")
        raise

    with tempfile.NamedTemporaryFile() as file:
        file.write(bytes([0x11]) * 4096)
        file.write(bytes([0x22]) * 4096)
        file.flush()
        table = object.__new__(FlashNextPleTable)
        table._page_cache = OrderedDict()
        table._page_cache_cap = 0
        table.page_cache_hits = 0
        table.page_cache_misses = 0
        table._io_backend = _PLE_IO_URING
        table._io_mode = _PLE_IO_URING
        table._uring_reader = reader
        table._uring_max_batch = 4
        table._large_batch_pread_pages = 8192
        table.pread_pages = 0
        table.io_uring_pages = 0
        table._shards = [SimpleNamespace(fd=file.fileno(), direct_fd=file.fileno())]

        loaded = table._load_pages([(0, 4096), (0, 0)])

    assert loaded[(0, 4096)] == bytes([0x22]) * 4096
    assert loaded[(0, 0)] == bytes([0x11]) * 4096


def test_load_pages_io_uring_distributes_work_across_readers():
    class FakeReader:
        def __init__(self, marker):
            self.marker = marker
            self.calls = []

        def read_pages(self, _fds, offsets):
            self.calls.append(tuple(offsets))
            return [bytes([self.marker]) for _ in offsets]

    table = object.__new__(FlashNextPleTable)
    table._uring_reader = None
    table._uring_readers = (FakeReader(0x11), FakeReader(0x22))
    table._uring_pool = None
    table._uring_pool_workers = 0
    table._uring_max_batch = 2
    table._shards = [SimpleNamespace(direct_fd=7)]
    try:
        loaded = table._load_pages_io_uring([(0, 0), (0, 4096), (0, 8192), (0, 12288), (0, 16384)])
        assert set(loaded) == {
            (0, 0),
            (0, 4096),
            (0, 8192),
            (0, 12288),
            (0, 16384),
        }
        assert sum(len(reader.calls) for reader in table._uring_readers) == 3
        assert {loaded[(0, 0)], loaded[(0, 4096)]} == {bytes([0x11]), bytes([0x22])}
    finally:
        if table._uring_pool is not None:
            table._uring_pool.shutdown(wait=True)


def test_io_uring_loader_does_not_import_sglang_package():
    before = {name for name in sys.modules if name == "sglang" or name.startswith("sglang.")}
    try:
        _load_io_uring_reader_type()
    except ModuleNotFoundError as error:
        pytest.skip(f"sglang io_uring extension unavailable: {error}")
    after = {name for name in sys.modules if name == "sglang" or name.startswith("sglang.")}
    assert after == before


@pytest.mark.skipif(not getattr(os, "O_DIRECT", 0), reason="O_DIRECT unavailable")
def test_mapped_shard_opens_separate_direct_io_descriptor(tmp_path: pathlib.Path):
    name = "ple.weight"
    metadata = {
        name: {
            "dtype": "F8_E4M3",
            "shape": [1, 160],
            "data_offsets": [0, 160],
        }
    }
    header = json.dumps(metadata).encode()
    path = tmp_path / "shard.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(160))

    shard = _MappedShard(path, name)
    try:
        assert fcntl.fcntl(shard.direct_fd, fcntl.F_GETFL) & os.O_DIRECT
        assert shard.direct_fd != shard.fd
    finally:
        shard.close()
