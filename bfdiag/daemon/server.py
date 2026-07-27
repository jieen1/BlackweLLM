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
  the daemon (a) answers the client immediately with a timeout error,
  (b) marks itself TAINTED, and (c) -- if configured -- "restarts" by
  constructing a brand-new provider via the factory and swapping it in.
  The abandoned thread keeps a reference to the OLD provider and can never
  touch the new one; but note this is an in-process, Python-level
  swap-and-reload, NOT a process-level restart, so it does not free GPU
  memory or reclaim a genuinely hung CUDA context -- see notes' GPU
  validation TODO list.
* Canary gating: every ``exec`` runs the canary check first (see
  ``canary.py``); a mismatch marks TAINTED and refuses to run the
  requested code at all, then restarts per the same policy as a timeout.
  A normal Python exception raised BY exec'd code is caught, its
  traceback is returned to the client, and the daemon does *not* taint
  itself for that alone (per spec: never let one experiment's bug kill or
  poison the whole daemon by itself -- the canary on the *next* exec is
  what catches it if the exception left bad state behind).
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
_STATES = ("STARTING", "READY", "BUSY", "TAINTED", "STOPPED")


class AlreadyRunningError(RuntimeError):
    """Another daemon process already holds the single-instance lock."""


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

        self._state = "STARTING"
        self._provider_lock = threading.Lock()
        self._jobs: queue.Queue[_Job | None] = queue.Queue()
        self._shutdown_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._listen_sock: socket.socket | None = None
        self._lock_fd: int | None = None
        self._restart_count = 0
        self._started_at: float | None = None

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
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="bfdiag-worker", daemon=True
        )
        self._worker_thread.start()
        logger.info(
            "bfdiag daemon ready: socket=%s canary=%s timeout_s=%s",
            self._socket_path,
            self._canary_enabled,
            self._default_timeout_s,
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

        self._set_state("BUSY")
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

        if outcome.timed_out:
            self._set_state("TAINTED")
            self._maybe_restart()
            return {
                "ok": False,
                "error": f"exec timed out after {timeout_s}s; daemon marked TAINTED",
                "stdout": outcome.stdout,
                "stderr": outcome.stderr,
                "elapsed_s": outcome.elapsed_s,
                "state": self._read_state(),
                "run_id": run_id,
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
        }

    def _maybe_restart(self) -> None:
        if not self._restart_on_taint:
            return
        logger.warning("bfdiag daemon: TAINTED, restarting provider")
        new_provider = self._provider_factory()
        new_provider.load()
        with self._provider_lock:
            self._provider = new_provider
            self._state = "READY"
        self._restart_count += 1

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
        return {
            "state": state,
            "pid": os.getpid(),
            "socket": str(self._socket_path),
            "canary_enabled": self._canary_enabled,
            "default_timeout_s": self._default_timeout_s,
            "restart_count": self._restart_count,
            "started_at": self._started_at,
            "provider": provider_desc,
        }


def _build_provider(args: argparse.Namespace) -> EngineProvider:
    if args.provider == "fake":
        return FakeEngineProvider()
    if args.provider == "laguna":
        return LagunaEngineProvider(model_path=args.model_path, num_slots=args.num_slots)
    raise ValueError(f"unknown provider {args.provider!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bfdiag-daemon")
    parser.add_argument("--provider", choices=["fake", "laguna"], default="fake")
    parser.add_argument("--socket", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--num-slots", type=int, default=1)
    parser.add_argument("--no-canary", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s bfdiag-daemon %(levelname)s %(message)s"
    )

    socket_path = Path(args.socket) if args.socket else default_socket_path()
    canary_enabled = (not args.no_canary) and os.environ.get("QSR_BFD_CANARY", "1") != "0"

    daemon = Daemon(
        provider_factory=lambda: _build_provider(args),
        socket_path=socket_path,
        canary_enabled=canary_enabled,
    )
    try:
        daemon.serve_forever()
    except AlreadyRunningError as exc:
        print(f"bfdiag daemon: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
