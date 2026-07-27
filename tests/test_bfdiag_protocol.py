"""Unit tests for bfdiag/daemon/protocol.py -- CPU-only, no socket/daemon."""

from __future__ import annotations

import io

import pytest

from bfdiag.daemon.protocol import (
    VALID_OPS,
    ProtocolError,
    Request,
    Response,
    read_line,
    write_line,
)


class TestRequest:
    def test_valid_ops_round_trip(self):
        for op in VALID_OPS:
            code = "1 + 1" if op == "exec" else None
            req = Request(op=op, code=code)
            assert Request.decode(req.encode()) == req

    def test_unknown_op_rejected(self):
        with pytest.raises(ProtocolError, match="unknown op"):
            Request(op="frobnicate")

    def test_exec_requires_code(self):
        with pytest.raises(ProtocolError, match="requires a non-empty"):
            Request(op="exec")
        with pytest.raises(ProtocolError, match="requires a non-empty"):
            Request(op="exec", code="")

    def test_ping_does_not_require_code(self):
        req = Request(op="ping")
        assert req.code is None

    def test_encode_is_newline_terminated_utf8(self):
        req = Request(op="ping")
        encoded = req.encode()
        assert isinstance(encoded, bytes)
        assert encoded.endswith(b"\n")
        assert encoded.count(b"\n") == 1

    def test_full_field_round_trip(self):
        req = Request(op="exec", code="x = 1", args={"a": 1}, timeout_s=12.5, run_id="run-1")
        decoded = Request.decode(req.encode())
        assert decoded == req

    def test_from_dict_missing_op(self):
        with pytest.raises(ProtocolError, match="missing required field 'op'"):
            Request.from_dict({"code": "1"})

    def test_decode_rejects_non_json(self):
        with pytest.raises(ProtocolError, match="invalid JSON"):
            Request.decode(b"not json at all {")

    def test_decode_rejects_non_object(self):
        with pytest.raises(ProtocolError, match="expected a JSON object"):
            Request.decode(b"[1, 2, 3]\n")

    def test_decode_rejects_empty_line(self):
        with pytest.raises(ProtocolError, match="empty line"):
            Request.decode(b"\n")

    def test_decode_accepts_str_or_bytes(self):
        req = Request(op="ping")
        as_bytes = req.encode()
        as_str = as_bytes.decode("utf-8")
        assert Request.decode(as_bytes) == Request.decode(as_str)


class TestResponse:
    def test_default_round_trip(self):
        resp = Response(ok=True)
        assert Response.decode(resp.encode()) == resp

    def test_full_field_round_trip(self):
        resp = Response(
            ok=False,
            result=None,
            stdout="hi\n",
            stderr="",
            traceback="Traceback...",
            error="boom",
            elapsed_s=1.5,
            state="TAINTED",
            run_id="run-42",
        )
        assert Response.decode(resp.encode()) == resp

    def test_missing_ok_rejected(self):
        with pytest.raises(ProtocolError, match="missing required field 'ok'"):
            Response.from_dict({"result": 1})

    def test_unknown_extra_fields_ignored(self):
        # Forward-compatible: a future daemon adding a field should not
        # break an older client's decode.
        data = Response(ok=True, result=1).to_dict()
        data["totally_new_field"] = "surprise"
        resp = Response.from_dict(data)
        assert resp.ok is True
        assert resp.result == 1

    def test_non_json_serializable_result_uses_str_fallback(self):
        class Weird:
            def __str__(self) -> str:
                return "weird-object"

        resp = Response(ok=True, result=Weird())
        encoded = resp.encode()
        assert b"weird-object" in encoded


class TestFraming:
    def test_write_then_read_round_trip(self):
        buf = io.BytesIO()
        write_line(buf, {"ok": True, "result": 42})
        buf.seek(0)
        assert read_line(buf) == {"ok": True, "result": 42}

    def test_read_multiple_messages_sequentially(self):
        buf = io.BytesIO()
        write_line(buf, {"ok": True, "result": 1})
        write_line(buf, {"ok": True, "result": 2})
        buf.seek(0)
        assert read_line(buf) == {"ok": True, "result": 1}
        assert read_line(buf) == {"ok": True, "result": 2}
        assert read_line(buf) is None

    def test_read_line_eof_returns_none(self):
        buf = io.BytesIO(b"")
        assert read_line(buf) is None

    def test_read_line_bad_json_raises(self):
        buf = io.BytesIO(b"{not json}\n")
        with pytest.raises(ProtocolError):
            read_line(buf)

    def test_write_line_flushes_when_available(self):
        class Recorder:
            def __init__(self) -> None:
                self.written = b""
                self.flushed = False

            def write(self, data: bytes) -> None:
                self.written += data

            def flush(self) -> None:
                self.flushed = True

        rec = Recorder()
        write_line(rec, {"ok": True})
        assert rec.flushed is True
        assert rec.written == b'{"ok":true}\n'
