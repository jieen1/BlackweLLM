"""Thin Unix-socket client for the bfdiag warm daemon.

Each request opens a fresh connection, writes one newline-delimited JSON
request line, reads one response line, and closes -- no persistent
connection is kept between calls (the daemon's own FIFO worker thread is
what serializes engine access; a persistent connection would buy nothing
here and would complicate reconnecting after a daemon restart).
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

from bfdiag.daemon.protocol import ProtocolError, Request, Response, read_line, write_line
from bfdiag.daemon.server import default_socket_path


class DaemonNotRunning(RuntimeError):
    """No daemon is listening on the configured socket."""


class Client:
    """Connects to one daemon socket. Stateless between calls -- safe to
    share across threads, cheap to construct per-call."""

    def __init__(
        self,
        socket_path: str | Path | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self._socket_path = Path(socket_path) if socket_path else default_socket_path()
        self._default_timeout_s = (
            timeout_s if timeout_s is not None else float(os.environ.get("QSR_BFD_TIMEOUT_S", "60"))
        )

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    def _connect(self, timeout_s: float) -> socket.socket:
        if not self._socket_path.exists():
            raise DaemonNotRunning(f"no daemon socket at {self._socket_path}")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(str(self._socket_path))
        except OSError as exc:
            sock.close()
            raise DaemonNotRunning(f"cannot connect to {self._socket_path}: {exc}") from exc
        # Generous padding over the server-side timeout: the server itself
        # enforces request-level timeouts (exec) and answers ping/status
        # immediately, so this is a network-level safety net, not the real
        # timeout policy.
        sock.settimeout(timeout_s + 10.0)
        return sock

    def _request(self, request: Request) -> Response:
        timeout_s = request.timeout_s if request.timeout_s is not None else self._default_timeout_s
        sock = self._connect(timeout_s)
        try:
            write_line(sock.makefile("wb"), request.to_dict())
            rfile = sock.makefile("rb")
            data = read_line(rfile)
            if data is None:
                raise ProtocolError("daemon closed the connection without a response")
            return Response.from_dict(data)
        finally:
            sock.close()

    def ping(self) -> Response:
        return self._request(Request(op="ping"))

    def status(self) -> Response:
        return self._request(Request(op="status"))

    def reset(self) -> Response:
        return self._request(Request(op="reset"))

    def shutdown(self) -> Response:
        return self._request(Request(op="shutdown"))

    def exec_code(
        self,
        code: str,
        *,
        args: dict[str, Any] | None = None,
        run_id: str | None = None,
        timeout_s: float | None = None,
    ) -> Response:
        return self._request(
            Request(op="exec", code=code, args=args, run_id=run_id, timeout_s=timeout_s)
        )

    def exec_file(
        self,
        path: str | Path,
        *,
        args: dict[str, Any] | None = None,
        run_id: str | None = None,
        timeout_s: float | None = None,
    ) -> Response:
        script_path = Path(path).resolve()
        code = f"__file__ = {str(script_path)!r}\n" + script_path.read_text()
        return self.exec_code(code, args=args, run_id=run_id, timeout_s=timeout_s)

    def is_running(self) -> bool:
        try:
            return self.ping().ok
        except DaemonNotRunning:
            return False

    def close(self) -> None:
        """No persistent resources are held between calls; kept for a
        symmetrical, future-proof API (e.g. if a pooled-connection mode is
        added later)."""


if __name__ == "__main__":
    client = Client()
    if client.is_running():
        print("daemon status:", client.status().result)
    else:
        print(f"no daemon running at {client.socket_path}")
