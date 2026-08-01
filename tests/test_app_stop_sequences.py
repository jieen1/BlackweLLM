"""N2 tests for server/app.py's HTTP layer: `stop` (OpenAI, string or <=4
strings) / `stop_sequences` (Anthropic, no documented cap) parsing and
validation, and end-to-end wiring through to the engine + back out as
OpenAI finish_reason="stop" / Anthropic stop_reason="stop_sequence" +
stop_sequence=<matched>, for both streaming and non-streaming, both
protocols.

See tests/test_stop_matching.py for the pure text-matching primitives and
tests/test_engine_stop_sequences.py for the engine decode-loop behavior
(including the cross-token-boundary and mid-MTP-batch cases) -- this file
only covers the HTTP-layer parsing/validation/wiring, using a fake engine.

CPU-only: fake engine, no GPU/tokenizer/model required.
"""

from __future__ import annotations

import json

import pytest

# server.app imports fastapi at module scope -- fastapi is a serving-extra,
# not a dev-extra (see AGENTS.md), so a CI venv with only `.[dev]` installed
# must skip this whole file cleanly, not ImportError.
pytest.importorskip("fastapi")


class TestNormalizeStop:
    def test_none_stays_none(self):
        from server import app as server_app

        assert server_app._normalize_stop(None) is None

    def test_string_becomes_single_element_list(self):
        from server import app as server_app

        assert server_app._normalize_stop("STOP") == ["STOP"]

    def test_list_passthrough(self):
        from server import app as server_app

        assert server_app._normalize_stop(["A", "B"]) == ["A", "B"]

    def test_empty_strings_dropped(self):
        from server import app as server_app

        assert server_app._normalize_stop(["", "A", ""]) == ["A"]

    def test_all_empty_normalizes_to_none(self):
        from server import app as server_app

        assert server_app._normalize_stop(["", ""]) is None

    def test_openai_max_four_enforced(self):
        from fastapi import HTTPException

        from server import app as server_app

        with pytest.raises(HTTPException):
            server_app._normalize_stop(["A", "B", "C", "D", "E"], max_count=4)

    def test_openai_exactly_four_ok(self):
        from server import app as server_app

        assert server_app._normalize_stop(["A", "B", "C", "D"], max_count=4) == [
            "A",
            "B",
            "C",
            "D",
        ]

    def test_anthropic_no_cap(self):
        from server import app as server_app

        five = ["A", "B", "C", "D", "E"]
        assert server_app._normalize_stop(five, max_count=None) == five


class _CapturingEngine:
    """Fake engine that records the kwargs it was called with and returns
    a canned result carrying a configurable ``matched_stop_sequence`` --
    used to assert on what app.py threads through to submit()/
    submit_stream() and how it surfaces the result, without any engine
    internals running (that's tests/test_engine_stop_sequences.py's job)."""

    MODEL = "laguna-test"
    capacity_tokens_per_slot = 4096

    class _FakeTok:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids if 32 <= i < 127)

    tok = _FakeTok()

    def __init__(self, matched_stop_sequence: str | None = None):
        self.submit_calls: list[dict] = []
        self.submit_stream_calls: list[dict] = []
        self._matched_stop_sequence = matched_stop_sequence

    def capacity_ok(self, _prompt_tokens, _max_tokens):
        return True

    async def submit(self, *args, **kwargs):
        self.submit_calls.append(kwargs)
        return {
            "committed_token_ids": [ord(c) for c in "ok"],
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "finish_reason": "stop",
            "matched_stop_sequence": self._matched_stop_sequence,
        }

    async def submit_stream(self, *args, **kwargs):
        self.submit_stream_calls.append(kwargs)
        text = "ok"
        for ch in text:
            yield [ord(ch)]
        yield {
            "finish_reason": "stop",
            "prompt_tokens": 1,
            "completion_tokens": len(text),
            "matched_stop_sequence": self._matched_stop_sequence,
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


class TestOpenAIStopWiring:
    def test_string_stop_normalized_and_threaded_to_engine(self, monkeypatch):
        from fastapi.testclient import TestClient

        from server import app as server_app

        engine = _CapturingEngine()
        _patch_common(monkeypatch, server_app, engine)
        client = TestClient(server_app.app)

        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stop": "STOP"},
        )
        assert resp.status_code == 200
        assert engine.submit_calls[-1]["stop_sequences"] == ["STOP"]

    def test_list_stop_threaded_to_engine(self, monkeypatch):
        from fastapi.testclient import TestClient

        from server import app as server_app

        engine = _CapturingEngine()
        _patch_common(monkeypatch, server_app, engine)
        client = TestClient(server_app.app)

        resp = client.post(
            "/v1/completions",
            json={"prompt": "hi", "stop": ["A", "B"]},
        )
        assert resp.status_code == 200
        assert engine.submit_calls[-1]["stop_sequences"] == ["A", "B"]

    def test_more_than_four_stop_sequences_rejected(self, monkeypatch):
        from fastapi.testclient import TestClient

        from server import app as server_app

        monkeypatch.setattr(server_app, "engine", _CapturingEngine())
        client = TestClient(server_app.app)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stop": ["A", "B", "C", "D", "E"],
            },
        )
        assert resp.status_code == 400
        assert "stop" in resp.json()["error"]["message"]

    def test_streaming_chat_completions_threads_stop_sequences(self, monkeypatch):
        from fastapi.testclient import TestClient

        from server import app as server_app

        engine = _CapturingEngine()
        _patch_common(monkeypatch, server_app, engine)
        client = TestClient(server_app.app)

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stop": "STOP",
            },
        ) as resp:
            for _ in resp.iter_lines():
                pass
        assert engine.submit_stream_calls[-1]["stop_sequences"] == ["STOP"]


class TestAnthropicStopSequencesWiring:
    def test_stop_sequences_threaded_no_cap(self, monkeypatch):
        from fastapi.testclient import TestClient

        from server import app as server_app

        engine = _CapturingEngine()
        _patch_common(monkeypatch, server_app, engine)
        client = TestClient(server_app.app)

        five = ["A", "B", "C", "D", "E"]
        resp = client.post(
            "/v1/messages",
            json={
                "model": "t",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                "stop_sequences": five,
            },
        )
        assert resp.status_code == 200
        assert engine.submit_calls[-1]["stop_sequences"] == five

    def test_non_streaming_stop_reason_and_stop_sequence_populated(self, monkeypatch):
        from fastapi.testclient import TestClient

        from server import app as server_app

        engine = _CapturingEngine(matched_stop_sequence="STOP")
        _patch_common(monkeypatch, server_app, engine)
        client = TestClient(server_app.app)

        resp = client.post(
            "/v1/messages",
            json={
                "model": "t",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
                "stop_sequences": ["STOP"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["stop_reason"] == "stop_sequence"
        assert body["stop_sequence"] == "STOP"

    def test_non_streaming_without_match_reports_end_turn(self, monkeypatch):
        from fastapi.testclient import TestClient

        from server import app as server_app

        engine = _CapturingEngine(matched_stop_sequence=None)
        _patch_common(monkeypatch, server_app, engine)
        client = TestClient(server_app.app)

        resp = client.post(
            "/v1/messages",
            json={
                "model": "t",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["stop_reason"] == "end_turn"
        assert body["stop_sequence"] is None

    def test_streaming_message_delta_carries_stop_sequence(self, monkeypatch):
        from fastapi.testclient import TestClient

        from server import app as server_app

        engine = _CapturingEngine(matched_stop_sequence="STOP")
        _patch_common(monkeypatch, server_app, engine)
        client = TestClient(server_app.app)

        with client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": "t",
                "max_tokens": 10,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
                "stop_sequences": ["STOP"],
            },
        ) as resp:
            events = []
            event_name = None
            for line in resp.iter_lines():
                if line.startswith("event: "):
                    event_name = line[len("event: ") :]
                elif line.startswith("data: "):
                    events.append((event_name, json.loads(line[len("data: ") :])))

        message_deltas = [d for name, d in events if name == "message_delta"]
        assert message_deltas, "no message_delta event observed"
        assert message_deltas[-1]["delta"]["stop_reason"] == "stop_sequence"
        assert message_deltas[-1]["delta"]["stop_sequence"] == "STOP"
