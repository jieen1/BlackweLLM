"""End-to-end tests for the bfdiag warm daemon: lifecycle, concurrency,
timeout/abandonment, crash recovery, and single-instance locking.

Everything here runs against ``FakeEngineProvider`` (pure Python, no
torch/CUDA) per this task's hard no-GPU constraint. Most tests run the
``Daemon`` in a background thread within the test process (fast,
deterministic); one test spawns a REAL subprocess running
``python -m bfdiag.daemon.server --provider fake`` to exercise the actual
process-lifecycle path ``bf daemon start`` drives -- still zero GPU/torch
involvement, since ``--provider fake`` never imports torch.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from bfdiag.daemon import cli
from bfdiag.daemon.client import Client, DaemonNotRunning
from bfdiag.daemon.provider import FakeEngineProvider
from bfdiag.daemon.server import AlreadyRunningError, Daemon


@pytest.fixture
def daemon_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Yields a callable that builds+starts a Daemon/Client pair with an
    isolated bfdiag dir and a short-path temp socket (AF_UNIX paths are
    capped around 107 bytes, so pytest's own -- often deep -- tmp_path is
    deliberately not used for the socket file itself)."""
    monkeypatch.setenv("QSR_BFDIAG_DIR", str(tmp_path))
    created: list[tuple[Daemon, Client, Path]] = []

    def _make(provider_factory=None, **daemon_kwargs) -> tuple[Daemon, Client]:
        factory = provider_factory or (lambda: FakeEngineProvider())
        socket_dir = Path(tempfile.mkdtemp())
        socket_path = socket_dir / "d.sock"
        daemon = Daemon(provider_factory=factory, socket_path=socket_path, **daemon_kwargs)
        daemon.serve_in_background()
        client = Client(socket_path=socket_path, timeout_s=5.0)
        created.append((daemon, client, socket_dir))
        return daemon, client

    yield _make

    for daemon, client, socket_dir in created:
        with contextlib.suppress(Exception):
            client.shutdown()
        with contextlib.suppress(Exception):
            shutil.rmtree(socket_dir, ignore_errors=True)
    time.sleep(0.1)


class TestLifecycle:
    def test_ping_and_status(self, daemon_factory):
        _daemon, client = daemon_factory()
        assert client.ping().ok is True
        status = client.status().result
        assert status["state"] == "READY"
        assert status["provider"]["kind"] == "fake"
        assert status["provider"]["loaded"] is True

    def test_exec_simple_code_returns_result_and_output(self, daemon_factory):
        _daemon, client = daemon_factory()
        response = client.exec_code("print('hello')\nresult = 1 + 1")
        assert response.ok is True
        assert response.result == 2
        assert "hello" in response.stdout

    def test_exec_sets_run_id_env_var_for_the_code(self, daemon_factory):
        _daemon, client = daemon_factory()
        response = client.exec_code("import os\nresult = os.environ.get('QSR_BFDIAG_RUN_ID')")
        assert response.ok is True
        assert response.result == response.run_id
        assert response.run_id is not None

    def test_exec_explicit_run_id_is_honored(self, daemon_factory):
        _daemon, client = daemon_factory()
        response = client.exec_code("result = 1", run_id="my-run-123")
        assert response.run_id == "my-run-123"

    def test_reset_op(self, daemon_factory):
        _daemon, client = daemon_factory()
        client.exec_code("provider.pollute()")
        assert client.reset().ok is True
        status = client.status().result
        assert status["provider"]["dirty"] == 0

    def test_exec_exception_is_caught_and_daemon_stays_alive(self, daemon_factory):
        _daemon, client = daemon_factory(canary_enabled=False)
        response = client.exec_code("raise ValueError('boom')")
        assert response.ok is False
        assert response.traceback is not None
        assert "ValueError: boom" in response.traceback
        assert response.state == "READY"  # a bare exception does not taint

        # Daemon must still be fully usable afterwards.
        follow_up = client.exec_code("result = 42")
        assert follow_up.ok is True
        assert follow_up.result == 42

    def test_shutdown_removes_socket_file(self, daemon_factory):
        daemon, client = daemon_factory()
        socket_path = daemon._socket_path
        assert socket_path.exists()
        assert client.shutdown().ok is True
        deadline = time.monotonic() + 2.0
        while socket_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not socket_path.exists()

    def test_client_raises_when_no_daemon_running(self, tmp_path: Path):
        client = Client(socket_path=tmp_path / "nonexistent.sock")
        with pytest.raises(DaemonNotRunning):
            client.ping()
        assert client.is_running() is False


class TestSingleInstanceLock:
    def test_second_daemon_on_same_socket_refuses_to_start(self, daemon_factory):
        daemon1, client1 = daemon_factory()
        daemon2 = Daemon(
            provider_factory=lambda: FakeEngineProvider(), socket_path=daemon1._socket_path
        )
        with pytest.raises(AlreadyRunningError):
            daemon2.start()
        # The original daemon must be completely unaffected.
        assert client1.ping().ok is True


class TestProtocolRobustness:
    def test_malformed_line_gets_error_response_and_daemon_survives(self, daemon_factory):
        import socket as socket_mod

        _daemon, client = daemon_factory()
        sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
        sock.connect(str(client.socket_path))
        sock.sendall(b"not even json\n")
        sock.settimeout(5.0)
        raw = sock.makefile("rb").readline()
        sock.close()
        assert b'"ok":false' in raw

        # Still usable for the next, well-formed request.
        assert client.ping().ok is True

    def test_unknown_op_bypasses_client_validation_gets_clean_error(self, daemon_factory):
        import socket as socket_mod

        _daemon, client = daemon_factory()
        sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
        sock.connect(str(client.socket_path))
        sock.sendall(json.dumps({"op": "not_a_real_op"}).encode("utf-8") + b"\n")
        sock.settimeout(5.0)
        raw = sock.makefile("rb").readline()
        sock.close()
        assert b'"ok":false' in raw
        assert client.ping().ok is True


class TestConcurrency:
    def test_concurrent_execs_never_overlap(self, daemon_factory):
        _daemon, client = daemon_factory(canary_enabled=False)
        # Give the fake provider extra scratch attributes to observe
        # concurrent engine access -- exactly the property the daemon's
        # single FIFO worker thread must guarantee on real (single-GPU)
        # hardware.
        _daemon._provider._active = 0
        _daemon._provider._max_active_seen = 0

        code = (
            "import time\n"
            "provider._active += 1\n"
            "provider._max_active_seen = max(provider._max_active_seen, provider._active)\n"
            "time.sleep(0.1)\n"
            "provider._active -= 1\n"
        )

        def _submit() -> None:
            resp = client.exec_code(code, timeout_s=5.0)
            assert resp.ok is True

        threads = [threading.Thread(target=_submit) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert _daemon._provider._max_active_seen == 1

    def test_ping_answers_immediately_during_a_long_exec(self, daemon_factory):
        _daemon, client = daemon_factory(canary_enabled=False)
        long_thread = threading.Thread(
            target=lambda: client.exec_code("import time\ntime.sleep(0.6)", timeout_s=5.0)
        )
        long_thread.start()
        time.sleep(0.15)  # let the long exec actually start
        t0 = time.perf_counter()
        assert client.ping().ok is True
        assert time.perf_counter() - t0 < 0.3
        long_thread.join(timeout=5.0)


class TestTimeoutAndAbandonment:
    def test_timeout_returns_quickly_and_taints_then_restarts(self, daemon_factory):
        load_count = {"n": 0}

        def factory() -> FakeEngineProvider:
            load_count["n"] += 1
            return FakeEngineProvider()

        daemon, client = daemon_factory(provider_factory=factory, canary_enabled=False)

        t0 = time.perf_counter()
        response = client.exec_code("import time\ntime.sleep(5)", timeout_s=0.2)
        elapsed = time.perf_counter() - t0

        assert response.ok is False
        assert "timed out" in (response.error or "")
        assert elapsed < 2.0  # much less than the 5s the abandoned code sleeps for

        # Restart-on-taint (default) means service resumes automatically.
        deadline = time.monotonic() + 2.0
        status = daemon.status()
        while status["state"] != "READY" and time.monotonic() < deadline:
            time.sleep(0.05)
            status = daemon.status()
        assert status["state"] == "READY"
        assert status["restart_count"] == 1
        assert load_count["n"] == 2  # original + restarted instance

        follow_up = client.exec_code("result = 'still alive'")
        assert follow_up.ok is True
        assert follow_up.result == "still alive"

    def test_timeout_with_restart_disabled_stays_tainted(self, daemon_factory):
        daemon, client = daemon_factory(canary_enabled=False, restart_on_taint=False)
        response = client.exec_code("import time\ntime.sleep(5)", timeout_s=0.2)
        assert response.ok is False

        deadline = time.monotonic() + 1.0
        while daemon.status()["state"] == "BUSY" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert daemon.status()["state"] == "TAINTED"

        refused = client.exec_code("result = 1")
        assert refused.ok is False
        assert "TAINTED" in (refused.error or "")


class TestCliSubprocessLifecycle:
    """Exercises the actual subprocess spawn/ping/status/exec/shutdown path
    that ``bf daemon start`` drives, against a real OS process -- but only
    ever with --provider fake, so this never touches torch/CUDA."""

    def test_start_status_exec_stop_round_trip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("QSR_BFDIAG_DIR", str(tmp_path))
        socket_dir = Path(tempfile.mkdtemp())
        socket_path = socket_dir / "d.sock"
        repo_root = Path(__file__).resolve().parents[1]

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "bfdiag.daemon.server",
                "--provider",
                "fake",
                "--socket",
                str(socket_path),
            ],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            client = Client(socket_path=socket_path, timeout_s=5.0)
            deadline = time.monotonic() + 10.0
            ready = False
            while time.monotonic() < deadline:
                if client.is_running():
                    ready = True
                    break
                time.sleep(0.1)
            assert ready, "daemon subprocess never became ready"

            status = client.status().result
            assert status["provider"]["kind"] == "fake"

            exec_response = client.exec_code("result = 1 + 1")
            assert exec_response.ok is True
            assert exec_response.result == 2

            shutdown_response = client.shutdown()
            assert shutdown_response.ok is True
            proc.wait(timeout=5.0)
        finally:
            if proc.poll() is None:
                proc.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5.0)
            shutil.rmtree(socket_dir, ignore_errors=True)

    def test_cli_daemon_start_reuses_running_instance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ):
        monkeypatch.setenv("QSR_BFDIAG_DIR", str(tmp_path))
        socket_dir = Path(tempfile.mkdtemp())
        socket_path = socket_dir / "d.sock"
        repo_root = Path(__file__).resolve().parents[1]

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "bfdiag.daemon.server",
                "--provider",
                "fake",
                "--socket",
                str(socket_path),
            ],
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            client = Client(socket_path=socket_path, timeout_s=5.0)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and not client.is_running():
                time.sleep(0.1)
            assert client.is_running()
            original_pid = client.status().result["pid"]

            args = argparse.Namespace(
                provider="fake",
                socket=str(socket_path),
                model_path=None,
                num_slots=1,
                no_canary=False,
                wait_s=2.0,
            )
            rc = cli._cmd_daemon_start(args)
            captured = capsys.readouterr()

            assert rc == 0
            assert "reusing" in captured.out
            assert client.status().result["pid"] == original_pid
        finally:
            with contextlib.suppress(DaemonNotRunning):
                client.shutdown()
            if proc.poll() is None:
                proc.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5.0)
            shutil.rmtree(socket_dir, ignore_errors=True)
