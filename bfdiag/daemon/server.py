"""The bfdiag warm daemon: a Unix-socket server holding one loaded
``EngineProvider`` for the lifetime of the process.

Design summary (see notes/2026-07-27-bfdiag-warm-daemon.md for the full
writeup):

* Single-instance guarantee: an ``flock(2)`` on a lock file next to the
  socket. Only one daemon may hold the GPU at a time (this machine has
  exactly one, shared with other agents); a second ``start()`` fails fast
  with ``AlreadyRunningError`` instead of silently fighting the first one
  for VRAM.
* All engine-touching work (``reset``, ``exec``) is funneled through a
  single FIFO worker thread, so two simultaneous ``bf exec`` clients can
  never touch the engine concurrently and always run in submission order.
  ``ping``/``status``/``shutdown`` are answered immediately by whichever
  connection-handler thread received them, without waiting behind a
  possibly-long-running exec job.
* Timeout handling cannot safely force-kill an arbitrary running Python
  thread (and, for the real engine, absolutely cannot interrupt a stuck
  CUDA kernel from another thread -- that requires killing the OS
  process). Instead: the worker thread runs each exec in a short-lived
  daemon thread and ``join(timeout_s)``s it. If it doesn't finish in time,
  the daemon answers the client immediately and marks itself TAINTED. CUDA
  providers remain TAINTED until their process is stopped and replaced;
  they never load a second runtime alongside an abandoned GPU thread. Pure
  Python providers may explicitly opt into the old in-process recovery path
  for lifecycle tests.
* Canary gating: every ``exec`` runs the canary check first (see
  ``canary.py``); a mismatch marks TAINTED and refuses to run the
  requested code at all, then restarts per the same policy as a timeout.
  A normal Python exception raised BY exec'd code is caught, its
  traceback is returned to the client, and the daemon does *not* taint
  itself for that alone (per spec: never let one experiment's bug kill or
  poison the whole daemon by itself -- the canary on the *next* exec is
  what catches it if the exception left bad state behind).
* Idle-TTL auto-release: this machine has exactly one GPU and its human
  owner also needs it. A daemon that just sits there loaded forever after
  the last experiment is a real cost, not a convenience. ``--idle-ttl-s``
  (default 900s / 15 minutes) tracks time since the last *completed*
  ``exec``/``reset`` (NOT ``ping``/``status`` -- a polling script must
  never be able to keep the daemon alive forever) and, once exceeded,
  unloads the provider and shuts the daemon down on its own, handing the
  GPU back. ``--idle-ttl-s 0`` disables this (an explicit choice to hold
  the GPU indefinitely). The idle check never fires while a job is
  actually running (state ``BUSY``) -- a long experiment is not "idle".
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import io
import logging
import os
import queue
import socket
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bfdiag.daemon.canary import DEFAULT_CANARY_PROMPT_IDS, DEFAULT_CANARY_STEPS, CanaryChecker
from bfdiag.daemon.protocol import ProtocolError, Request, read_line, write_line
from bfdiag.daemon.provider import EngineProvider, FakeEngineProvider, LagunaEngineProvider

logger = logging.getLogger("bfdiag.daemon")

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_LAGUNA_TIMEOUT_S = 900.0
DEFAULT_IDLE_TTL_S = 900.0  # 15 minutes -- see module docstring's idle-TTL bullet
_STATES = ("STARTING", "READY", "BUSY", "TAINTED", "STOPPED")


class AlreadyRunningError(RuntimeError):
    """Another daemon process already holds the single-instance lock."""


class ManualClock:
    """Injectable fake clock: a callable returning a manually-advanced,
    monotonic-like float. Passed as ``Daemon(..., clock=...)`` so idle-TTL
    tests can fast-forward through a 900-second default without an
    equivalent 900 real seconds of ``time.sleep`` -- see
    ``tests/test_bfdiag_daemon.py``'s idle-TTL tests. Never used by
    production code (``Daemon``'s default ``clock=time.monotonic``)."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def bfdiag_dir() -> Path:
    """Root storage directory: ``${QSR_BFDIAG_DIR:-<repo>/.bfdiag}``.

    Repo root is resolved relative to this file (``bfdiag/daemon/server.py``
    -> repo root is two parents up), not the current working directory, so
    this is stable regardless of where ``bf`` is invoked from.
    """
    override = os.environ.get("QSR_BFDIAG_DIR")
    if override:
        return Path(override)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / ".bfdiag"


def default_socket_path() -> Path:
    override = os.environ.get("QSR_BFD_SOCKET")
    if override:
        return Path(override)
    return bfdiag_dir() / "bfd.sock"


def _lock_path_for(socket_path: Path) -> Path:
    return socket_path.with_name(socket_path.name + ".lock")


def _generate_run_id() -> str:
    return f"bfd-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


@dataclass
class _Job:
    request: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None


@dataclass
class _ExecOutcome:
    ok: bool
    result: Any
    stdout: str
    stderr: str
    exc_traceback: str | None
    elapsed_s: float
    timed_out: bool


def _run_with_timeout(
    code: str,
    namespace: dict[str, Any],
    args: dict[str, Any] | None,
    timeout_s: float,
) -> _ExecOutcome:
    """Execute ``code`` against a copy of ``namespace`` in a short-lived
    thread, ``join(timeout_s)``-ing it. See module docstring for why a
    forced kill is not attempted: it is not safely possible for arbitrary
    Python code, and definitely not possible for a stuck CUDA call, short
    of terminating the OS process.
    """
    ns = dict(namespace)
    ns["__name__"] = "__main__"
    ns["args"] = args or {}

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    outcome_box: dict[str, Any] = {}

    def _target() -> None:
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                compiled = compile(code, "<bfdiag-exec>", "exec")
                exec(compiled, ns)
            outcome_box["ok"] = True
            outcome_box["result"] = ns.get("result")
            outcome_box["traceback"] = None
        except BaseException:
            outcome_box["ok"] = False
            outcome_box["result"] = None
            outcome_box["traceback"] = traceback.format_exc()
        finally:
            outcome_box["elapsed_s"] = time.perf_counter() - t0

    thread = threading.Thread(target=_target, daemon=True, name="bfdiag-exec")
    start = time.perf_counter()
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        # Best-effort abandonment -- see docstring. The thread may still be
        # writing to stdout_buf/stderr_buf concurrently with the read
        # below; under CPython's GIL this cannot corrupt the process, at
        # worst the captured snapshot is incomplete/torn.
        return _ExecOutcome(
            ok=False,
            result=None,
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            exc_traceback=None,
            elapsed_s=time.perf_counter() - start,
            timed_out=True,
        )
    return _ExecOutcome(
        ok=outcome_box["ok"],
        result=outcome_box.get("result"),
        stdout=stdout_buf.getvalue(),
        stderr=stderr_buf.getvalue(),
        exc_traceback=outcome_box.get("traceback"),
        elapsed_s=outcome_box["elapsed_s"],
        timed_out=False,
    )


def _safe_memory_snapshot(provider: EngineProvider) -> dict[str, Any]:
    """``provider.memory_snapshot()``, never allowed to break ``exec`` --
    a snapshot is diagnostic sugar, not something that should turn a
    working experiment into a failure if it raises."""
    try:
        return provider.memory_snapshot()
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": str(exc)}


class Daemon:
    """The warm daemon server. See module docstring for the design."""

    def __init__(
        self,
        provider_factory: Callable[[], EngineProvider],
        socket_path: str | Path | None = None,
        *,
        canary_enabled: bool | None = None,
        default_timeout_s: float | None = None,
        restart_on_taint: bool = True,
        canary_prompt_ids: list[int] | None = None,
        canary_steps: int | None = None,
        idle_ttl_s: float | None = None,
        idle_check_interval_s: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._provider_factory = provider_factory
        self._provider: EngineProvider = provider_factory()
        self._socket_path = Path(socket_path) if socket_path else default_socket_path()
        self._bfdiag_dir = bfdiag_dir()
        self._canary_enabled = (
            canary_enabled
            if canary_enabled is not None
            else os.environ.get("QSR_BFD_CANARY", "1") != "0"
        )
        self._default_timeout_s = (
            default_timeout_s
            if default_timeout_s is not None
            else float(os.environ.get("QSR_BFD_TIMEOUT_S", str(DEFAULT_TIMEOUT_S)))
        )
        self._restart_on_taint = restart_on_taint
        default_prompt_ids = list(DEFAULT_CANARY_PROMPT_IDS)
        self._canary = CanaryChecker(
            self._bfdiag_dir,
            prompt_ids=canary_prompt_ids if canary_prompt_ids is not None else default_prompt_ids,
            steps=canary_steps if canary_steps is not None else DEFAULT_CANARY_STEPS,
            enabled=self._canary_enabled,
        )
        self._idle_ttl_s = (
            idle_ttl_s
            if idle_ttl_s is not None
            else float(os.environ.get("QSR_BFD_IDLE_TTL_S", str(DEFAULT_IDLE_TTL_S)))
        )
        self._idle_check_interval_s = idle_check_interval_s
        self._clock = clock

        self._state = "STARTING"
        self._provider_lock = threading.Lock()
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._shutdown_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._idle_watchdog_thread: threading.Thread | None = None
        self._listen_sock: socket.socket | None = None
        self._lock_fd: int | None = None
        self._restart_count = 0
        self._exec_count = 0
        self._started_at: float | None = None
        self._started_clock_ts: float | None = None
        self._last_activity_ts: float = self._clock()

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        """Acquire the single-instance lock, load the provider, bind the
        socket, and start the FIFO worker thread. Does not accept
        connections yet -- call ``serve_forever()``/``serve_in_background()``
        (or ``_accept_loop()`` directly) for that."""
        self._bfdiag_dir.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()
        if self._socket_path.exists():
            # Safe: we hold the exclusive flock, so any leftover socket
            # file is guaranteed stale (a crashed previous daemon).
            self._socket_path.unlink()
        self._listen_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listen_sock.bind(str(self._socket_path))
        self._listen_sock.listen(16)
        self._listen_sock.settimeout(0.2)

        self._provider.load()
        self._set_state("READY")
        self._started_at = time.time()
        self._started_clock_ts = self._clock()
        self._last_activity_ts = self._clock()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="bfdiag-worker", daemon=True
        )
        self._worker_thread.start()
        if self._idle_ttl_s > 0:
            self._idle_watchdog_thread = threading.Thread(
                target=self._idle_watchdog_loop, name="bfdiag-idle-watchdog", daemon=True
            )
            self._idle_watchdog_thread.start()
        logger.info(
            "bfdiag daemon ready: socket=%s canary=%s timeout_s=%s idle_ttl_s=%s",
            self._socket_path,
            self._canary_enabled,
            self._default_timeout_s,
            self._idle_ttl_s if self._idle_ttl_s > 0 else "disabled",
        )

    def serve_forever(self) -> None:
        """Blocking: ``start()`` then accept connections until ``shutdown``."""
        self.start()
        self._accept_loop()

    def serve_in_background(self) -> threading.Thread:
        """``start()`` synchronously (so the socket is guaranteed bound by
        the time this returns), then run the accept loop in a background
        thread. Intended for tests -- callers can connect immediately."""
        self.start()
        thread = threading.Thread(target=self._accept_loop, name="bfdiag-accept", daemon=True)
        thread.start()
        return thread

    def _accept_loop(self) -> None:
        assert self._listen_sock is not None
        try:
            while not self._shutdown_event.is_set():
                try:
                    conn, _ = self._listen_sock.accept()
                except TimeoutError:
                    continue
                except OSError:
                    break
                threading.Thread(
                    target=self._handle_connection, args=(conn,), daemon=True
                ).start()
        finally:
            self._stop()

    def _stop(self) -> None:
        self._jobs.put(None)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
        if self._idle_watchdog_thread is not None:
            self._idle_watchdog_thread.join(timeout=5.0)
        if self._listen_sock is not None:
            with contextlib.suppress(OSError):
                self._listen_sock.close()
        with contextlib.suppress(OSError):
            self._socket_path.unlink(missing_ok=True)
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None
        self._set_state("STOPPED")

    def _acquire_lock(self) -> None:
        lock_path = _lock_path_for(self._socket_path)
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise AlreadyRunningError(
                f"another bfdiag daemon already holds the lock at {lock_path} "
                "(only one daemon may own the GPU at a time)"
            ) from exc
        self._lock_fd = fd

    # -- idle-TTL watchdog: give the GPU back when nobody is using it -----

    def _idle_watchdog_loop(self) -> None:
        """Polls every ``idle_check_interval_s``; once ``idle_s`` (time
        since the last completed ``exec``/``reset`` -- ``ping``/``status``
        never count) reaches ``idle_ttl_s``, unloads the provider and
        shuts the daemon down. Never fires while a job is actually running
        (``state == "BUSY"``) -- a long-running experiment is not "idle".
        Uses ``self._shutdown_event.wait(...)`` instead of ``time.sleep``
        so a manual ``shutdown``/``bf daemon stop`` wakes this thread
        immediately instead of leaving it to poll out its last interval.
        """
        while not self._shutdown_event.wait(self._idle_check_interval_s):
            if self._read_state() == "BUSY":
                continue
            idle_s = self._clock() - self._read_last_activity()
            if idle_s >= self._idle_ttl_s:
                logger.warning(
                    "bfdiag daemon: idle for %.1fs >= --idle-ttl-s=%.1fs, "
                    "auto-shutting down to release the GPU",
                    idle_s,
                    self._idle_ttl_s,
                )
                self._auto_shutdown_for_idle()
                break

    def _auto_shutdown_for_idle(self) -> None:
        with contextlib.suppress(Exception):
            self._provider.unload()
        self._shutdown_event.set()
        self._jobs.put(None)

    def _read_last_activity(self) -> float:
        with self._provider_lock:
            return self._last_activity_ts

    def _bump_activity(self) -> None:
        with self._provider_lock:
            self._last_activity_ts = self._clock()

    # -- connection handling ---------------------------------------------

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn:
            rfile = conn.makefile("rb")
            try:
                data = read_line(rfile)
            except ProtocolError as exc:
                self._safe_write(conn, {"ok": False, "error": f"protocol error: {exc}"})
                return
            if data is None:
                return
            try:
                request = Request.from_dict(data)
            except ProtocolError as exc:
                self._safe_write(conn, {"ok": False, "error": f"protocol error: {exc}"})
                return
            if request.op in ("ping", "status", "shutdown"):
                response = self._handle_immediate(request)
            else:
                response = self._submit_and_wait(request)
            self._safe_write(conn, response)

    def _safe_write(self, conn: socket.socket, data: dict[str, Any]) -> None:
        with contextlib.suppress(OSError):
            write_line(conn.makefile("wb"), data)

    def _handle_immediate(self, request: Request) -> dict[str, Any]:
        if request.op == "ping":
            return {"ok": True, "result": "pong", "state": self._read_state()}
        if request.op == "status":
            return {"ok": True, "result": self.status(), "state": self._read_state()}
        if request.op == "shutdown":
            self._shutdown_event.set()
            self._jobs.put(None)
            return {"ok": True, "result": "shutting down", "state": self._read_state()}
        return {"ok": False, "error": f"unexpected immediate op {request.op!r}"}

    def _submit_and_wait(self, request: Request) -> dict[str, Any]:
        job = _Job(request=request.to_dict())
        self._jobs.put(job)
        job.done.wait()
        return job.response or {"ok": False, "error": "no response produced (internal error)"}

    # -- worker thread: the ONLY thread allowed to touch the engine ------

    def _worker_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                break
            try:
                job.response = self._execute_job(job.request)
            except Exception as exc:  # pragma: no cover - defensive
                job.response = {
                    "ok": False,
                    "error": f"internal daemon error: {exc}",
                    "traceback": traceback.format_exc(),
                    "state": self._read_state(),
                }
            finally:
                # Idle clock restarts when a job FINISHES, not when it was
                # submitted -- a single exec running longer than
                # idle_ttl_s must not make the daemon look "idle" the
                # instant it completes (see _idle_watchdog_loop's BUSY
                # guard for the other half of this: it never fires mid-job
                # either, so a long job is never mistaken for idleness).
                self._bump_activity()
                job.done.set()

    def _execute_job(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        if op == "reset":
            return self._do_reset()
        if op == "exec":
            return self._do_exec(request)
        return {"ok": False, "error": f"unexpected queued op {op!r}", "state": self._read_state()}

    def _do_reset(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            self._provider.reset()
        except Exception as exc:
            self._set_state("TAINTED")
            self._maybe_restart()
            return {
                "ok": False,
                "error": f"reset() failed: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_s": time.perf_counter() - t0,
                "state": self._read_state(),
            }
        self._set_state("READY")
        return {
            "ok": True,
            "result": "reset",
            "elapsed_s": time.perf_counter() - t0,
            "state": self._read_state(),
        }

    def _do_exec(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._read_state() == "TAINTED":
            return {
                "ok": False,
                "error": "daemon is TAINTED (a previous canary/timeout failure was not "
                "cleared); refusing to exec until it recovers",
                "state": self._read_state(),
            }

        timeout_s = request.get("timeout_s") or self._default_timeout_s
        run_id = request.get("run_id") or _generate_run_id()

        if self._canary_enabled:
            canary_result = self._canary.check(self._provider)
            if not canary_result.ok:
                self._set_state("TAINTED")
                self._maybe_restart()
                return {
                    "ok": False,
                    "error": f"canary mismatch, refusing exec: {canary_result.detail}",
                    "state": self._read_state(),
                    "run_id": run_id,
                }

        with self._provider_lock:
            self._exec_count += 1
        self._set_state("BUSY")
        # Memory snapshot BEFORE the code runs -- see notes' "memory
        # visibility" section: a lone tok/s number from a long-lived hot
        # daemon is not comparable to another one unless the caching
        # allocator's state at measurement time is recorded alongside it.
        memory_before = _safe_memory_snapshot(self._provider)
        prev_run_id = os.environ.get("QSR_BFDIAG_RUN_ID")
        os.environ["QSR_BFDIAG_RUN_ID"] = run_id
        try:
            outcome = _run_with_timeout(
                code=request["code"],
                namespace=self._provider.namespace(),
                args=request.get("args"),
                timeout_s=timeout_s,
            )
        finally:
            if prev_run_id is None:
                os.environ.pop("QSR_BFDIAG_RUN_ID", None)
            else:
                os.environ["QSR_BFDIAG_RUN_ID"] = prev_run_id
        # ...and AFTER, taken before any taint/restart bookkeeping so a
        # timed-out call's "after" snapshot still reflects the provider
        # that was actually abandoned, not a freshly-restarted one.
        memory_after = _safe_memory_snapshot(self._provider)

        if outcome.timed_out:
            self._set_state("TAINTED")
            recovered = self._maybe_restart()
            recovery = (
                "provider recovered in-process"
                if recovered
                else "stop this daemon and start a fresh process before retrying"
            )
            return {
                "ok": False,
                "error": (
                    f"exec timed out after {timeout_s}s; daemon marked TAINTED; {recovery}"
                ),
                "stdout": outcome.stdout,
                "stderr": outcome.stderr,
                "elapsed_s": outcome.elapsed_s,
                "state": self._read_state(),
                "run_id": run_id,
                "memory_before": memory_before,
                "memory_after": memory_after,
            }

        self._set_state("READY")
        return {
            "ok": outcome.ok,
            "result": outcome.result,
            "stdout": outcome.stdout,
            "stderr": outcome.stderr,
            "traceback": outcome.exc_traceback,
            "elapsed_s": outcome.elapsed_s,
            "state": self._read_state(),
            "run_id": run_id,
            "memory_before": memory_before,
            "memory_after": memory_after,
        }

    def _maybe_restart(self) -> bool:
        if not self._restart_on_taint:
            return False
        if not getattr(self._provider, "allow_in_process_recovery_after_taint", False):
            logger.error(
                "bfdiag daemon: TAINTED provider %s requires process replacement; "
                "refusing unsafe in-process restart",
                type(self._provider).__name__,
            )
            return False
        logger.warning("bfdiag daemon: TAINTED, restarting provider")
        new_provider = self._provider_factory()
        new_provider.load()
        with self._provider_lock:
            self._provider = new_provider
            self._state = "READY"
        self._restart_count += 1
        return True

    # -- status ------------------------------------------------------------

    def _set_state(self, state: str) -> None:
        assert state in _STATES
        with self._provider_lock:
            self._state = state

    def _read_state(self) -> str:
        with self._provider_lock:
            return self._state

    def status(self) -> dict[str, Any]:
        with self._provider_lock:
            provider_desc = self._provider.describe()
            state = self._state
            last_activity = self._last_activity_ts
            exec_count = self._exec_count
        now = self._clock()
        since_cold_start = (
            now - self._started_clock_ts if self._started_clock_ts is not None else None
        )
        return {
            "state": state,
            "pid": os.getpid(),
            "socket": str(self._socket_path),
            "canary_enabled": self._canary_enabled,
            "default_timeout_s": self._default_timeout_s,
            "restart_count": self._restart_count,
            "exec_count": exec_count,
            "started_at": self._started_at,
            "since_cold_start_s": since_cold_start,
            "idle_s": now - last_activity,
            "idle_ttl_s": self._idle_ttl_s,
            "provider": provider_desc,
        }


def _build_provider(args: argparse.Namespace) -> EngineProvider:
    if args.provider == "fake":
        return FakeEngineProvider(num_slots=args.num_slots)
    if args.provider == "laguna":
        return LagunaEngineProvider(
            model_path=args.model_path,
            num_slots=args.num_slots,
            blocks_per_slot=args.blocks_per_slot,
            block_size=args.block_size,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    raise ValueError(f"unknown provider {args.provider!r}")


def _resolve_default_timeout_s(provider: str, configured: float | None) -> float | None:
    """Keep normal interactive defaults for fake/CPU providers while giving
    Laguna's complete fixed workloads enough time to finish."""
    if configured is not None:
        return configured
    return DEFAULT_LAGUNA_TIMEOUT_S if provider == "laguna" else None


def _add_provider_args(parser: argparse.ArgumentParser) -> None:
    """Shared between this module's own ``main()`` and ``cli.py``'s
    ``bf daemon start`` so the two never drift out of sync on what a
    Laguna load-time config actually looks like (see provider.py's
    ``LOAD_TIME_CONFIG_KEYS``)."""
    parser.add_argument("--provider", choices=["fake", "laguna"], default="fake")
    parser.add_argument("--socket", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--num-slots", type=int, default=1)
    parser.add_argument("--blocks-per-slot", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--no-canary", action="store_true")
    parser.add_argument("--idle-ttl-s", type=float, default=None)
    parser.add_argument(
        "--default-timeout-s",
        type=float,
        default=None,
        help=(
            "default per-exec timeout; Laguna defaults to 900 seconds so long "
            "fixed-contract benchmarks cannot accidentally taint the daemon"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bfdiag-daemon")
    _add_provider_args(parser)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s bfdiag-daemon %(levelname)s %(message)s"
    )

    socket_path = Path(args.socket) if args.socket else default_socket_path()
    canary_enabled = (not args.no_canary) and os.environ.get("QSR_BFD_CANARY", "1") != "0"
    default_timeout_s = _resolve_default_timeout_s(args.provider, args.default_timeout_s)

    daemon = Daemon(
        provider_factory=lambda: _build_provider(args),
        socket_path=socket_path,
        canary_enabled=canary_enabled,
        default_timeout_s=default_timeout_s,
        idle_ttl_s=args.idle_ttl_s,
    )
    try:
        daemon.serve_forever()
    except AlreadyRunningError as exc:
        print(f"bfdiag daemon: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
