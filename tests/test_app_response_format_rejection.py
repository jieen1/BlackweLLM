"""N1 tests for server/app.py: response_format={"type": "json_object"/
"json_schema"} must be rejected loudly (400), not silently accepted and
ignored -- see server/app.py::_reject_unsupported_response_format and
docs/api-layer-design.md §7.1 for why this runtime chose "reject" over
"wire it in": the only reachable masking hook is never reached by the
prefill anchor token, CUDA-Graph decode replay, or the eager is_greedy
shortcut -- exactly the paths a default (temperature unset, i.e. greedy)
"give me JSON" request takes.

CPU-only: fake engine, no GPU/tokenizer/model required.
"""

from __future__ import annotations

import pytest

# server.app imports fastapi at module scope -- fastapi is a serving-extra,
# not a dev-extra (see AGENTS.md), so a CI venv with only `.[dev]` installed
# must skip this whole file cleanly, not ImportError.
pytest.importorskip("fastapi")


class TestRejectUnsupportedResponseFormat:
    def test_none_is_fine(self):
        from server import app as server_app

        server_app._reject_unsupported_response_format(None)  # no raise

    def test_text_type_is_fine(self):
        from server import app as server_app

        server_app._reject_unsupported_response_format({"type": "text"})  # no raise

    def test_json_object_rejected(self):
        from fastapi import HTTPException

        from server import app as server_app

        with pytest.raises(HTTPException) as exc:
            server_app._reject_unsupported_response_format({"type": "json_object"})
        assert exc.value.status_code == 400

    def test_json_schema_rejected(self):
        from fastapi import HTTPException

        from server import app as server_app

        with pytest.raises(HTTPException):
            server_app._reject_unsupported_response_format(
                {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
            )


class _CapturingEngine:
    """Fake engine returning a canned, always-successful result -- these
    tests only need to reach (or be stopped before reaching) engine.submit/
    submit_stream, not exercise engine internals."""

    MODEL = "laguna-test"
    capacity_tokens_per_slot = 4096

    class _FakeTok:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids if 32 <= i < 127)

    tok = _FakeTok()

    def capacity_ok(self, _prompt_tokens, _max_tokens):
        return True

    async def submit(self, *args, **kwargs):
        return {
            "committed_token_ids": [ord(c) for c in "ok"],
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "finish_reason": "stop",
            "matched_stop_sequence": None,
        }

    async def submit_stream(self, *args, **kwargs):
        for ch in "ok":
            yield [ord(ch)]
        yield {
            "finish_reason": "stop",
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "matched_stop_sequence": None,
        }


def _patch_common(monkeypatch, server_app, engine):
    async def _tokenize_chat(*_args, **_kwargs):
        return [0]

    async def _tokenize_encode(*_args, **_kwargs):
        return [0]

    async def _tokenize_decode(*_args, **_kwargs):
        return "ok"

    async def _noop(*_args, **_kwargs):
        return None

    def _sync_noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server_app, "engine", engine)
    monkeypatch.setattr(server_app, "_tokenize_chat", _tokenize_chat)
    monkeypatch.setattr(server_app, "_tokenize_encode", _tokenize_encode)
    monkeypatch.setattr(server_app, "_tokenize_decode", _tokenize_decode)
    monkeypatch.setattr(server_app, "_debug_log_input", _noop)
    monkeypatch.setattr(server_app, "_debug_log_output", _sync_noop)
    monkeypatch.setattr(server_app, "_debug_log_stream_output", _noop)


class TestResponseFormatRejectionViaRealHttpDispatch:
    def test_chat_completions_rejects_json_object(self, monkeypatch):
        from fastapi.testclient import TestClient

        from server import app as server_app

        monkeypatch.setattr(server_app, "engine", _CapturingEngine())
        client = TestClient(server_app.app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_object"},
            },
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["type"] == "invalid_request_error"
        assert "response_format" in body["error"]["message"]

    def test_chat_completions_streaming_also_rejects(self, monkeypatch):
        """The check runs before the stream/non-stream branch -- streaming
        requests must not sneak past it either."""
        from fastapi.testclient import TestClient

        from server import app as server_app

        monkeypatch.setattr(server_app, "engine", _CapturingEngine())
        client = TestClient(server_app.app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "response_format": {"type": "json_schema", "json_schema": {"schema": {}}},
            },
        )
        assert resp.status_code == 400

    def test_completions_legacy_endpoint_also_rejects(self, monkeypatch):
        from fastapi.testclient import TestClient

        from server import app as server_app

        monkeypatch.setattr(server_app, "engine", _CapturingEngine())
        client = TestClient(server_app.app)

        resp = client.post(
            "/v1/completions",
            json={"prompt": "hi", "response_format": {"type": "json_object"}},
        )
        assert resp.status_code == 400

    def test_chat_completions_without_response_format_unaffected(self, monkeypatch):
        from fastapi.testclient import TestClient

        from server import app as server_app

        engine = _CapturingEngine()
        _patch_common(monkeypatch, server_app, engine)
        client = TestClient(server_app.app)

        resp = client.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert resp.status_code == 200

    def test_chat_completions_text_type_unaffected(self, monkeypatch):
        from fastapi.testclient import TestClient

        from server import app as server_app

        engine = _CapturingEngine()
        _patch_common(monkeypatch, server_app, engine)
        client = TestClient(server_app.app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "text"},
            },
        )
        assert resp.status_code == 200
