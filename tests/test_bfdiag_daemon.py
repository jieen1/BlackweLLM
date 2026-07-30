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
from bfdiag.daemon.protocol import Response
from bfdiag.daemon.provider import (
    LOAD_TIME_ENV_VARS,
    FakeEngineProvider,
    LagunaEngineProvider,
    _rebind_instance_class,
    _rebind_verify_graphs,
    requires_cold_restart,
)
from bfdiag.daemon.queue import check_sweep_is_hot_safe, submit
from bfdiag.daemon.server import AlreadyRunningError, Daemon, ManualClock


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


class TestHotReloadMechanics:
    def test_rebind_instance_class_uses_new_method_implementation(self):
        class OldImplementation:
            def marker(self):
                return "old"

        class NewImplementation:
            def marker(self):
                return "new"

        instance = OldImplementation()
        assert _rebind_instance_class(instance, NewImplementation, "test") is True
        assert instance.marker() == "new"

    def test_rebind_none_is_a_noop(self):
        assert _rebind_instance_class(None, object, "optional") is False

    def test_rebind_verify_graphs_includes_bounded_final_graphs(self):
        class OldGraph:
            pass

        class NewGraph:
            pass

        class Engine:
            _verify_cg = OldGraph()
            _partial_verify_cgs = {7: OldGraph()}

        engine = Engine()
        assert _rebind_verify_graphs(engine, NewGraph) == [
            "engine._verify_cg",
            "engine._partial_verify_cgs[7]",
        ]
        assert isinstance(engine._verify_cg, NewGraph)
        assert isinstance(engine._partial_verify_cgs[7], NewGraph)

    def test_laguna_provider_reset_restores_prefill_implementation(self, monkeypatch):
        provider = LagunaEngineProvider()
        calls: list[str] = []

        class Backend:
            def _unpatch_impls_for_prefill(self):
                calls.append("unpatch")

        provider._backend = Backend()
        provider._engine = object()
        monkeypatch.setattr(
            "bfdiag.daemon.session.reset_laguna_engine",
            lambda engine: calls.append("reset"),
        )

        provider.reset()
        assert calls == ["unpatch", "reset"]

    def test_reload_command_reloads_provider_before_runtime(self, monkeypatch):
        class ClientStub:
            code: str | None = None

            def exec_code(self, code, *, timeout_s):
                self.code = code
                assert timeout_s == 120.0
                return Response(ok=True, result={"canary_tokens": 8})

        client_stub = ClientStub()
        monkeypatch.setattr(cli, "_client_for", lambda _args: client_stub)

        assert cli._cmd_daemon_reload(argparse.Namespace(timeout_s=120.0)) == 0
        assert "importlib.reload(provider_module)" in client_stub.code
        assert "provider.__class__ = provider_module.LagunaEngineProvider" in client_stub.code
        assert "provider.hot_reload_code()" in client_stub.code


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

    def test_manual_shutdown_unloads_provider_once(self, daemon_factory):
        daemon, client = daemon_factory()
        provider = daemon._provider

        assert client.shutdown().ok is True
        deadline = time.monotonic() + 2.0
        while daemon._read_state() != "STOPPED" and time.monotonic() < deadline:
            time.sleep(0.02)

        assert daemon._read_state() == "STOPPED"
        assert provider._unload_count == 1

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


class TestTimeoutDefaults:
    def test_laguna_process_default_is_long_but_fake_default_stays_fast(self):
        from bfdiag.daemon.server import DEFAULT_LAGUNA_TIMEOUT_S, _resolve_default_timeout_s

        assert _resolve_default_timeout_s("laguna", None) == DEFAULT_LAGUNA_TIMEOUT_S
        assert _resolve_default_timeout_s("fake", None) is None
        assert _resolve_default_timeout_s("laguna", 123.0) == 123.0

    def test_device_provider_timeout_refuses_unsafe_in_process_recovery(self, daemon_factory):
        load_count = {"n": 0}

        class UnsafeDeviceLikeProvider(FakeEngineProvider):
            allow_in_process_recovery_after_taint = False

        def factory() -> UnsafeDeviceLikeProvider:
            load_count["n"] += 1
            return UnsafeDeviceLikeProvider()

        daemon, client = daemon_factory(provider_factory=factory, canary_enabled=False)
        response = client.exec_code("import time\ntime.sleep(5)", timeout_s=0.2)

        assert response.ok is False
        assert "stop this daemon" in (response.error or "")
        assert daemon.status()["state"] == "TAINTED"
        assert daemon.status()["restart_count"] == 0
        assert load_count["n"] == 1

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
                blocks_per_slot=4096,
                block_size=64,
                dtype="bfloat16",
                max_model_len=131072,
                gpu_memory_utilization=0.88,
                no_canary=False,
                idle_ttl_s=None,
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

    def test_daemon_start_refuses_reuse_on_load_config_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ):
        """The one real GPU this machine has must never get silently
        reconfigured out from under a running daemon: `bf daemon start`
        with a DIFFERENT load-time config than what's already running must
        refuse to just "reuse" it."""
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
                "--num-slots",
                "1",
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

            args = argparse.Namespace(
                provider="fake",
                socket=str(socket_path),
                model_path=None,
                num_slots=2,  # DIFFERENT from the running instance's 1
                blocks_per_slot=4096,
                block_size=64,
                dtype="bfloat16",
                max_model_len=131072,
                gpu_memory_utilization=0.88,
                no_canary=False,
                idle_ttl_s=None,
                wait_s=2.0,
            )
            rc = cli._cmd_daemon_start(args)
            captured = capsys.readouterr()

            assert rc == 1
            assert "DIFFERENT load-time config" in captured.err
            assert "num_slots" in captured.err
            # The running daemon must be completely unaffected.
            assert client.status().result["provider"]["load_config"]["num_slots"] == 1
        finally:
            with contextlib.suppress(DaemonNotRunning):
                client.shutdown()
            if proc.poll() is None:
                proc.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=5.0)
            shutil.rmtree(socket_dir, ignore_errors=True)


class TestIdleTTL:
    """Idle-TTL auto-release: the whole point is that a human owns this
    GPU too. Uses ManualClock throughout so tests fast-forward through a
    900-second default TTL without 900 real seconds of sleeping."""

    def test_idle_ttl_triggers_auto_shutdown(self, daemon_factory):
        clock = ManualClock()
        daemon, client = daemon_factory(clock=clock, idle_ttl_s=10.0, idle_check_interval_s=0.02)
        assert client.ping().ok is True

        clock.advance(11.0)
        deadline = time.monotonic() + 2.0
        while daemon._read_state() != "STOPPED" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert daemon._read_state() == "STOPPED"
        assert not daemon._socket_path.exists()
        assert daemon._provider._unload_count == 1

    def test_activity_resets_the_idle_timer(self, daemon_factory):
        clock = ManualClock()
        daemon, client = daemon_factory(
            clock=clock, idle_ttl_s=10.0, idle_check_interval_s=0.02, canary_enabled=False
        )

        clock.advance(8.0)
        assert client.exec_code("result = 1").ok is True  # bumps the timer back to "now"
        time.sleep(0.1)  # let the watchdog poll at least once

        clock.advance(8.0)  # 8s since the exec above -- still under the 10s ttl
        time.sleep(0.1)
        assert daemon._read_state() != "STOPPED"

        clock.advance(3.0)  # now 11s since the last activity -- past the ttl
        deadline = time.monotonic() + 2.0
        while daemon._read_state() != "STOPPED" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert daemon._read_state() == "STOPPED"

    def test_ping_and_status_do_not_reset_the_idle_timer(self, daemon_factory):
        clock = ManualClock()
        daemon, client = daemon_factory(clock=clock, idle_ttl_s=5.0, idle_check_interval_s=0.02)

        clock.advance(4.0)
        for _ in range(5):  # a polling script hammering ping/status ...
            assert client.ping().ok is True
            assert client.status().ok is True
        clock.advance(2.0)  # ... must not have kept the daemon alive past its ttl

        deadline = time.monotonic() + 2.0
        while daemon._read_state() != "STOPPED" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert daemon._read_state() == "STOPPED"

    def test_idle_ttl_zero_disables_auto_shutdown(self, daemon_factory):
        clock = ManualClock()
        daemon, client = daemon_factory(clock=clock, idle_ttl_s=0.0, idle_check_interval_s=0.02)
        assert daemon._idle_watchdog_thread is None

        clock.advance(10_000_000.0)
        time.sleep(0.1)
        assert client.ping().ok is True
        assert daemon._read_state() != "STOPPED"

    def test_idle_check_never_fires_mid_job_and_resets_on_completion(self, daemon_factory):
        clock = ManualClock()
        daemon, client = daemon_factory(
            clock=clock, idle_ttl_s=5.0, idle_check_interval_s=0.02, canary_enabled=False
        )
        # The clock already reads "expired" relative to the ttl BEFORE any
        # job starts -- if the BUSY guard didn't work, the watchdog would
        # shut the daemon down out from under the running experiment.
        clock.advance(10.0)

        done = threading.Event()

        def _submit_slow() -> None:
            client.exec_code("import time\ntime.sleep(0.3)", timeout_s=5.0)
            done.set()

        thread = threading.Thread(target=_submit_slow)
        thread.start()
        time.sleep(0.1)  # let it actually become BUSY
        assert daemon._read_state() == "BUSY"
        time.sleep(0.15)  # several watchdog polls while still BUSY
        assert daemon._read_state() == "BUSY"  # never yanked mid-job
        assert client.socket_path.exists()

        thread.join(timeout=5.0)
        assert done.is_set()
        # Completion resets the idle clock to "now" -- immediately after,
        # the daemon must still be alive (idle_s ~= 0), not stopped.
        assert daemon._read_state() != "STOPPED"

        clock.advance(6.0)  # now genuinely idle past the ttl
        deadline = time.monotonic() + 2.0
        while daemon._read_state() != "STOPPED" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert daemon._read_state() == "STOPPED"

    def test_status_exposes_idle_and_cold_start_fields(self, daemon_factory):
        clock = ManualClock()
        _daemon, client = daemon_factory(clock=clock, idle_ttl_s=100.0, idle_check_interval_s=1.0)

        status = client.status().result
        assert status["idle_ttl_s"] == 100.0
        assert status["idle_s"] == pytest.approx(0.0, abs=0.01)
        assert status["since_cold_start_s"] == pytest.approx(0.0, abs=0.01)
        assert status["exec_count"] == 0

        clock.advance(5.0)
        status = client.status().result
        assert status["idle_s"] == pytest.approx(5.0, abs=0.01)
        assert status["since_cold_start_s"] == pytest.approx(5.0, abs=0.01)

    def test_exec_count_increments_only_on_actual_execution(self, daemon_factory):
        clock = ManualClock()
        daemon, client = daemon_factory(clock=clock, idle_ttl_s=100.0, canary_enabled=False)
        client.exec_code("result = 1")
        client.exec_code("result = 2")
        assert daemon.status()["exec_count"] == 2
        # ping/status/reset must never count as an "experiment executed".
        client.ping()
        client.status()
        client.reset()
        assert daemon.status()["exec_count"] == 2


class TestMemorySnapshots:
    def test_exec_response_includes_memory_snapshots(self, daemon_factory):
        _daemon, client = daemon_factory()
        response = client.exec_code("result = 1")
        assert response.memory_before == {
            "kind": "fake",
            "allocated_bytes": None,
            "reserved_bytes": None,
            "num_alloc_retries": None,
            "fragmentation_ratio": None,
        }
        assert response.memory_after == response.memory_before

    def test_non_exec_ops_carry_no_memory_snapshot(self, daemon_factory):
        _daemon, client = daemon_factory()
        assert client.ping().memory_before is None
        assert client.status().memory_before is None
        assert client.reset().memory_before is None


class TestHotColdBoundary:
    """The other half of protecting the one shared GPU: some config is
    fixed the instant the engine is constructed, and pretending otherwise
    (silently sweeping it through an already-loaded hot daemon) produces
    measurements that look real but never actually changed anything."""

    def test_requires_cold_restart_detects_a_changed_key(self):
        current = {"num_slots": 1, "dtype": "bfloat16"}
        requested = {"num_slots": 4, "dtype": "bfloat16"}
        assert requires_cold_restart(current, requested) == ["num_slots"]

    def test_requires_cold_restart_no_diff_is_empty(self):
        cfg = {"num_slots": 1, "dtype": "bfloat16"}
        assert requires_cold_restart(cfg, dict(cfg)) == []

    def test_requires_cold_restart_ignores_unlocked_keys(self):
        current = {"num_slots": 1, "some_unlocked_thing": "a"}
        requested = {"num_slots": 1, "some_unlocked_thing": "b"}
        assert requires_cold_restart(current, requested) == []

    def test_requires_cold_restart_missing_on_one_side_counts_as_changed(self):
        current = {"num_slots": 1}
        requested = {"num_slots": 1, "dtype": "bfloat16"}
        assert requires_cold_restart(current, requested) == ["dtype"]

    def test_requires_cold_restart_both_missing_is_not_changed(self):
        assert requires_cold_restart({}, {}) == []

    def test_requires_cold_restart_custom_locked_keys(self):
        current = {"a": 1, "b": 2}
        requested = {"a": 1, "b": 99}
        assert requires_cold_restart(current, requested, locked_keys=frozenset({"a"})) == []
        assert requires_cold_restart(current, requested, locked_keys=frozenset({"b"})) == ["b"]

    def test_load_time_env_vars_are_the_audited_set(self):
        # Documents the exact, code-verified list (see provider.py's
        # comment citing runtime/backends/laguna*.py line numbers) --
        # changing this set should be a deliberate, reviewed edit.
        assert LOAD_TIME_ENV_VARS == {
            "QSR_PREFILL_CHUNK",
            "QSR_DECODE_CUDA_GRAPH",
            "QSR_DFLASH_CUDA_GRAPH",
            "QSR_VERIFY_CUDA_GRAPH",
        }

    def test_fake_provider_describe_exposes_load_config(self):
        provider = FakeEngineProvider(num_slots=3)
        assert provider.describe()["load_config"] == {"num_slots": 3}

    def test_laguna_provider_describe_exposes_load_config_without_gpu(self):
        # Constructing LagunaEngineProvider only sets plain-Python
        # attributes (all torch/runtime imports are deferred into method
        # bodies) -- __init__ + describe() is safe with zero GPU/torch.
        provider = LagunaEngineProvider(
            num_slots=2,
            blocks_per_slot=2048,
            block_size=128,
            dtype="float16",
            max_model_len=8192,
            gpu_memory_utilization=0.5,
        )
        load_config = provider.describe()["load_config"]
        assert load_config == {
            "model_path": provider.describe()["model_path"],
            "num_slots": 2,
            "blocks_per_slot": 2048,
            "block_size": 128,
            "dtype": "float16",
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.5,
            "dflash_model_path": None,
        }

    def test_check_sweep_is_hot_safe_allows_ordinary_sweeps(self):
        check_sweep_is_hot_safe(["QSR_ASSERT_LEVEL=0,2"])  # must not raise

    def test_check_sweep_is_hot_safe_rejects_load_time_env_var(self):
        with pytest.raises(ValueError, match="load-time-locked"):
            check_sweep_is_hot_safe(["QSR_DFLASH_CUDA_GRAPH=0,1"])

    def test_submit_refuses_load_time_sweep_before_touching_the_client(self, tmp_path: Path):
        script = tmp_path / "script.py"
        script.write_text("result = 1\n")

        class _PoisonClient:
            def exec_code(self, *args, **kwargs):
                raise AssertionError("submit() must not reach the client for an unsafe sweep")

        with pytest.raises(ValueError, match="load-time-locked"):
            submit(script, ["QSR_DECODE_CUDA_GRAPH=0,1"], client=_PoisonClient())

    def test_run_cold_runs_one_independent_process_per_sweep_variant(self, tmp_path: Path):
        results_file = tmp_path / "results.txt"
        script = tmp_path / "cold_script.py"
        script.write_text(
            "import os\n"
            f"with open({str(results_file)!r}, 'a') as f:\n"
            "    f.write(os.environ.get('MYVAR', 'MISSING') + chr(10))\n"
        )
        args = argparse.Namespace(script=str(script), cold=True, sweep=["MYVAR=a,b,c"])
        rc = cli._cmd_run(args)
        assert rc == 0
        assert results_file.read_text().splitlines() == ["a", "b", "c"]

    def test_run_requires_explicit_cold_flag_at_the_argparse_level(self):
        parser = argparse.ArgumentParser(prog="bf")
        subparsers = parser.add_subparsers(dest="command", required=True)
        cli.register(subparsers)
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "script.py"])  # missing --cold
