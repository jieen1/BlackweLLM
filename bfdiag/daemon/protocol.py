"""Newline-delimited JSON protocol for the bfdiag warm daemon's Unix socket.

Wire format: one JSON object per line (UTF-8, ``\\n``-terminated), in both
directions. Deliberately simple -- no length-prefix framing, no binary --
so the socket can be poked by hand (``socat -,rawer UNIX-CONNECT:bfd.sock``,
or a two-line ``python -c``) while debugging the diagnostics platform
itself.

Request ops: ``exec``, ``ping``, ``reset``, ``status``, ``shutdown``.

Everything in this module is pure (encode/decode dataclasses plus framing
helpers over any file-like object), so it is unit-testable without a real
socket or daemon -- see ``tests/test_bfdiag_protocol.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import IO, Any

VALID_OPS = frozenset({"exec", "ping", "reset", "status", "shutdown"})

#: Response fields, in this order, when re-serializing (kept for readability
#: of hand-inspected JSON lines; not load-bearing for decoding).
_RESPONSE_FIELD_ORDER = (
    "ok",
    "result",
    "stdout",
    "stderr",
    "traceback",
    "error",
    "elapsed_s",
    "state",
    "run_id",
)


class ProtocolError(ValueError):
    """A request/response line is malformed or violates the wire contract."""


@dataclass
class Request:
    """One request line.

    ``op`` selects the behavior; ``code`` is required (and must be
    non-empty) for ``op == "exec"``. ``args`` is an arbitrary JSON-safe dict
    made available to exec'd code as the ``args`` namespace binding.
    ``timeout_s`` overrides the daemon's default per-request timeout.
    ``run_id`` correlates this request with a ``.bfdiag/runs/<run_id>/``
    record (other bfdiag subsystems' run recorders key off this); the
    daemon generates one if omitted.
    """

    op: str
    code: str | None = None
    args: dict[str, Any] | None = None
    timeout_s: float | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if self.op not in VALID_OPS:
            raise ProtocolError(f"unknown op {self.op!r}; expected one of {sorted(VALID_OPS)}")
        if self.op == "exec" and not self.code:
            raise ProtocolError("op 'exec' requires a non-empty 'code' string")

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None} | {"op": self.op}

    def encode(self) -> bytes:
        """Serialize to one newline-terminated JSON line, UTF-8 encoded."""
        return _dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Request:
        if "op" not in data:
            raise ProtocolError("request is missing required field 'op'")
        try:
            return cls(
                op=data["op"],
                code=data.get("code"),
                args=data.get("args"),
                timeout_s=data.get("timeout_s"),
                run_id=data.get("run_id"),
            )
        except ProtocolError:
            raise
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"malformed request: {exc}") from exc

    @classmethod
    def decode(cls, line: bytes | str) -> Request:
        return cls.from_dict(_loads(line))


@dataclass
class Response:
    """One response line.

    ``ok`` is the only field callers must always check. ``result`` is
    whatever the exec'd code assigned to a module-level ``result`` name (or
    the op's own small JSON-safe payload for ``status``/``ping``/etc.);
    ``stdout``/``stderr`` are captured output; ``traceback`` is set when
    exec'd code raised; ``error`` is a short daemon-side message (protocol
    errors, timeouts, canary refusals); ``state`` is the daemon's state
    (``READY``/``BUSY``/``TAINTED``/...) after handling this request.
    """

    ok: bool
    result: Any = None
    stdout: str = ""
    stderr: str = ""
    traceback: str | None = None
    error: str | None = None
    elapsed_s: float = 0.0
    state: str | None = None
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {name: data[name] for name in _RESPONSE_FIELD_ORDER}

    def encode(self) -> bytes:
        return _dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Response:
        if "ok" not in data:
            raise ProtocolError("response is missing required field 'ok'")
        known = {f.name for f in fields(cls)}
        try:
            return cls(**{k: v for k, v in data.items() if k in known})
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"malformed response: {exc}") from exc

    @classmethod
    def decode(cls, line: bytes | str) -> Response:
        return cls.from_dict(_loads(line))


def _dumps(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, separators=(",", ":"), default=str) + "\n").encode("utf-8")


def _loads(line: bytes | str) -> dict[str, Any]:
    text = line.decode("utf-8") if isinstance(line, bytes) else line
    text = text.strip()
    if not text:
        raise ProtocolError("empty line (no message)")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProtocolError(f"expected a JSON object, got {type(data).__name__}")
    return data


def read_line(source: IO[bytes]) -> dict[str, Any] | None:
    """Read one newline-delimited JSON message from any object exposing a
    ``.readline()`` that returns ``bytes`` (a socket's ``.makefile('rb')``,
    a ``io.BytesIO``, ...). Returns ``None`` at EOF (no more data at all);
    raises ``ProtocolError`` for a non-empty line that fails to parse.
    """
    raw = source.readline()
    if not raw:
        return None
    return _loads(raw)


def write_line(target: IO[bytes], data: dict[str, Any]) -> None:
    """Write one newline-delimited JSON message to any object exposing
    ``.write(bytes)`` (and, ideally, ``.flush()``)."""
    target.write(json.dumps(data, separators=(",", ":"), default=str).encode("utf-8") + b"\n")
    flush = getattr(target, "flush", None)
    if flush is not None:
        flush()


if __name__ == "__main__":
    # Minimal self-test / usage demo -- no socket, no daemon required.
    req = Request(op="exec", code="1 + 1", timeout_s=5.0)
    print("request:", req.encode())
    print("round-trip ok:", Request.decode(req.encode()) == req)

    resp = Response(ok=True, result=2, elapsed_s=0.001, state="READY")
    print("response:", resp.encode())
    print("round-trip ok:", Response.decode(resp.encode()) == resp)
