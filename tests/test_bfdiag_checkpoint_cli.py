"""Tests for ``bfdiag/checkpoint/cli.py``: dispatcher wiring, the pure
filesystem subcommands (list/show/rm/schema), graceful "no daemon running"
handling for save/restore, and one genuine end-to-end round trip through a
REAL (in-process, ``FakeEngineProvider``-style) warm daemon -- proving
``bf checkpoint save``/``restore`` actually work over the real daemon RPC
path, not just the lower-level Python API already covered by
``test_bfdiag_checkpoint_store.py``/``test_bfdiag_checkpoint_restore.py``.

Nothing here touches the GPU: the daemon-backed test uses a tiny
``EngineProvider``-compatible wrapper around
:mod:`bfdiag.checkpoint.testing`'s pure-CPU ``FakeBackend``/
``FakeDFlashEngine``, exactly mirroring how ``tests/test_bfdiag_daemon.py``
drives the real ``Daemon``/``Client`` machinery against
``FakeEngineProvider``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from bfdiag.checkpoint import cli, store
from bfdiag.checkpoint.testing import FakeBackend, FakeDFlashEngine
from bfdiag.daemon.client import Client
from bfdiag.daemon.server import Daemon


def _build_parser() -> tuple[argparse.ArgumentParser, Any]:
    parser = argparse.ArgumentParser(prog="bf")
    subparsers = parser.add_subparsers(dest="command")
    cli.register(subparsers)
    return parser, subparsers


def _run(parser: argparse.ArgumentParser, argv: list[str]) -> int:
    args = parser.parse_args(argv)
    return args.func(args)


# --- dispatcher wiring ---------------------------------------------------


def test_register_wires_up_all_subcommands() -> None:
    parser, _ = _build_parser()
    args = parser.parse_args(["checkpoint", "list"])
    assert callable(args.func)
    needs_name = {"show", "rm", "save", "restore"}
    for sub in ("list", "show", "rm", "schema", "save", "restore"):
        argv = ["checkpoint", sub] + (["x"] if sub in needs_name else [])
        args = parser.parse_args(argv)
        assert callable(args.func)


def test_checkpoint_with_no_subcommand_prints_help_and_returns_1(capsys) -> None:
    parser, _ = _build_parser()
    rc = _run(parser, ["checkpoint"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "checkpoint" in out


# --- pure filesystem subcommands (list/show/rm/schema) --------------------


def _make_checkpoint(root: Path, name: str) -> None:
    backend = FakeBackend(num_slots=2, block_size=16, blocks_per_slot=32, swa_window=40)
    engine = FakeDFlashEngine(backend, draft_window=40, num_draft_layers=2)
    prompt = list(range(1, 61))
    boot = engine.dflash_prefill_bootstrap(0, prompt)
    anchor, draft_tokens = boot["anchor"], boot["draft_tokens"]
    for _ in range(3):
        dec = engine.dflash_round(0, anchor, draft_tokens)
        anchor, draft_tokens = dec["next_anchor"], dec["next_draft_tokens"]
    store.save_checkpoint(engine, 0, name, root=root, baseline_steps=1)


def test_cmd_list_show_rm_schema(tmp_path: Path, capsys) -> None:
    parser, _ = _build_parser()
    _make_checkpoint(tmp_path, "alpha")
    _make_checkpoint(tmp_path, "beta")

    rc = _run(parser, ["checkpoint", "list", "--root", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "alpha" in out and "beta" in out

    rc = _run(parser, ["checkpoint", "list", "--root", str(tmp_path), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert {d["name"] for d in data} == {"alpha", "beta"}

    rc = _run(parser, ["checkpoint", "show", "alpha", "--root", str(tmp_path), "--json"])
    assert rc == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["name"] == "alpha"
    assert shown["fingerprint"]["laguna_geometry"]["block_size"] == 16

    rc = _run(parser, ["checkpoint", "show", "does-not-exist", "--root", str(tmp_path)])
    assert rc == 1
    capsys.readouterr()

    rc = _run(parser, ["checkpoint", "schema", "--json"])
    assert rc == 0
    items = json.loads(capsys.readouterr().out)
    assert len(items) >= 10

    rc = _run(parser, ["checkpoint", "rm", "alpha", "--root", str(tmp_path)])
    assert rc == 0
    capsys.readouterr()
    remaining = {m.name for m in store.list_checkpoints(root=tmp_path)}
    assert remaining == {"beta"}

    rc = _run(parser, ["checkpoint", "rm", "alpha", "--root", str(tmp_path)])
    assert rc == 1


# --- save/restore against a daemon that isn't running --------------------


def test_cmd_save_reports_no_daemon_running(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("QSR_BFD_SOCKET", str(tmp_path / "no-such.sock"))
    parser, _ = _build_parser()
    rc = _run(parser, ["checkpoint", "save", "x"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no warm daemon running" in err


def test_cmd_restore_reports_no_daemon_running(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("QSR_BFD_SOCKET", str(tmp_path / "no-such.sock"))
    parser, _ = _build_parser()
    rc = _run(parser, ["checkpoint", "restore", "x"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no warm daemon running" in err


# --- genuine end-to-end round trip through a real (in-process) daemon ----


class _FakeCheckpointProvider:
    """Minimal ``EngineProvider``-compatible wrapper around
    :mod:`bfdiag.checkpoint.testing`'s fakes -- enough surface for the
    daemon to load/reset/expose a namespace, nothing more. Mirrors the
    role ``FakeEngineProvider`` (bfdiag/daemon/provider.py) plays for the
    daemon's OWN test suite, adapted so ``namespace()`` exposes an
    ``engine``/``backend`` pair this package's ``save_checkpoint``/
    ``restore_checkpoint`` can operate on."""

    def __init__(self) -> None:
        self._backend: FakeBackend | None = None
        self._engine: FakeDFlashEngine | None = None

    def load(self) -> None:
        self._backend = FakeBackend(num_slots=2, block_size=16, blocks_per_slot=32, swa_window=40)
        self._engine = FakeDFlashEngine(self._backend, draft_window=40, num_draft_layers=2)

    def reset(self) -> None:
        assert self._backend is not None and self._engine is not None
        for slot in range(self._backend.num_slots):
            self._backend.reset_slot(slot)
        for tensor in self._engine._draft_kv_caches.values():
            tensor.zero_()

    def describe(self) -> dict[str, Any]:
        return {"kind": "fake-checkpoint", "loaded": self._engine is not None, "load_config": {}}

    def is_healthy(self) -> bool:
        return self._engine is not None

    def generate(
        self, prompt_ids: list[int], max_tokens: int, *, temperature: float = 0.0
    ) -> list[int]:
        raise NotImplementedError("canary is disabled for this test; generate() is unused")

    def namespace(self) -> dict[str, Any]:
        return {"engine": self._engine, "backend": self._backend, "provider": self}

    def unload(self) -> None:
        self._engine = None
        self._backend = None

    def memory_snapshot(self) -> dict[str, Any]:
        return {"kind": "fake-checkpoint"}


@pytest.fixture
def real_daemon(tmp_path):
    """A real ``Daemon``/``Client`` pair (in-process thread) backed by
    ``_FakeCheckpointProvider``, canary disabled (it would otherwise call
    the unimplemented ``generate()`` before every exec)."""
    socket_dir = Path(tempfile.mkdtemp())
    socket_path = socket_dir / "d.sock"
    daemon = Daemon(
        provider_factory=_FakeCheckpointProvider,
        socket_path=socket_path,
        canary_enabled=False,
    )
    daemon.serve_in_background()
    client = Client(socket_path=socket_path, timeout_s=10.0)
    yield client
    with contextlib.suppress(Exception):
        client.shutdown()
    with contextlib.suppress(Exception):
        shutil.rmtree(socket_dir, ignore_errors=True)
    time.sleep(0.05)


def test_cli_save_and_restore_round_trip_via_real_daemon(
    real_daemon: Client, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "_get_client", lambda timeout_s: real_daemon)

    prime_code = textwrap.dedent(
        """
        prompt = list(range(1, 91))
        boot = engine.dflash_prefill_bootstrap(0, prompt)
        anchor, draft_tokens = boot["anchor"], boot["draft_tokens"]
        for _ in range(5):
            dec = engine.dflash_round(0, anchor, draft_tokens)
            anchor, draft_tokens = dec["next_anchor"], dec["next_draft_tokens"]
        result = {"kv_len": backend.slot_kv_len[0]}
        """
    )
    resp = real_daemon.exec_code(prime_code)
    assert resp.ok, resp.error
    primed_kv_len = resp.result["kv_len"]
    assert primed_kv_len > 0

    parser, _ = _build_parser()

    rc = _run(
        parser,
        ["checkpoint", "save", "daemon-e2e", "--root", str(tmp_path), "--baseline-steps", "2"],
    )
    assert rc == 0

    manifest = store.load_manifest("daemon-e2e", root=tmp_path)
    assert manifest.slot_kv_len == primed_kv_len

    # A second exec resets the engine back to pristine (mirrors a fresh
    # daemon session that never saw this prompt), THEN we restore.
    resp = real_daemon.exec_code("provider.reset()\nresult = backend.slot_kv_len[0]")
    assert resp.ok, resp.error
    assert resp.result == 0

    rc = _run(
        parser,
        ["checkpoint", "restore", "daemon-e2e", "--root", str(tmp_path)],
    )
    assert rc == 0

    resp = real_daemon.exec_code(
        "result = {'kv_len': backend.slot_kv_len[0], "
        "'committed': backend.slot_committed_tokens[0]}"
    )
    assert resp.ok, resp.error
    # Restore ran with the default verify_after=True, which -- per the
    # documented safety-valve side effect -- runs manifest.baseline's real
    # decode rounds on top of the restored checkpoint, advancing kv_len
    # further. The restored history must be an exact PREFIX of the current
    # one (proving the checkpoint boundary itself was reproduced exactly).
    assert resp.result["kv_len"] >= primed_kv_len
    committed_len = len(manifest.slot_committed_tokens)
    assert resp.result["committed"][:committed_len] == manifest.slot_committed_tokens


def test_cli_restore_exec_file_sees_restored_anchor_and_draft_tokens(
    real_daemon: Client, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(cli, "_get_client", lambda timeout_s: real_daemon)

    prime_code = textwrap.dedent(
        """
        prompt = list(range(1, 61))
        boot = engine.dflash_prefill_bootstrap(0, prompt)
        anchor, draft_tokens = boot["anchor"], boot["draft_tokens"]
        for _ in range(3):
            dec = engine.dflash_round(0, anchor, draft_tokens)
            anchor, draft_tokens = dec["next_anchor"], dec["next_draft_tokens"]
        result = None
        """
    )
    assert real_daemon.exec_code(prime_code).ok

    parser, _ = _build_parser()
    rc = _run(parser, ["checkpoint", "save", "exec-file-e2e", "--root", str(tmp_path)])
    assert rc == 0

    assert real_daemon.exec_code("provider.reset()").ok

    followup_script = tmp_path / "followup.py"
    followup_script.write_text(
        "dec = engine.dflash_round(0, anchor, draft_tokens)\n"
        "result = {'accepted': dec['context_count']}\n"
    )
    rc = _run(
        parser,
        [
            "checkpoint",
            "restore",
            "exec-file-e2e",
            "--root",
            str(tmp_path),
            "--exec-file",
            str(followup_script),
        ],
    )
    assert rc == 0
