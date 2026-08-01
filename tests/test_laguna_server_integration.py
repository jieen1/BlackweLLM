"""E1: LagunaBackend <-> ServerEngine wiring (CPU-only, no GPU/model load).

Covers:
- classify_decode_slots: the pure predicate that routes a decode round's
  active slots to the MTP path vs. the plain sampled path used by Laguna.
- ServerEngine backend selection: real (non-GPU) Laguna-only construction,
  verifying MODEL/K/eos_token_ids are set correctly.
- LagunaBackend's new E1 surface (reconcile_prefix_hit, prefill_chunked_begin/
  step, decode_batch_sampled's signature) via __new__ bypass -- these methods
  either don't touch GPU state at all, or only do so past an early return we
  never reach in these tests.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

pytest.importorskip("torch")

from server.engine import ServerEngine, classify_decode_slots


class TestClassifyDecodeSlots:
    def test_mtp_capable_reproduces_original_split(self):
        active = {
            1: {"sampled": False},
            2: {"sampled": True},
            3: {"sampled": False},
        }
        greedy, sampled = classify_decode_slots(
            [1, 2, 3], active, grammar_slots=[], mtp_capable=True
        )
        assert greedy == [1, 3]
        assert sampled == [2]

    def test_mtp_capable_grammar_slots_forced_to_sampled(self):
        active = {1: {"sampled": False}, 2: {"sampled": False}}
        greedy, sampled = classify_decode_slots([1, 2], active, grammar_slots=[2], mtp_capable=True)
        assert greedy == [1]
        assert sampled == [2]

    def test_non_mtp_backend_routes_everything_to_sampled(self):
        """Laguna (mtp_capable=False): even a 'greedy' slot skips MTP."""
        active = {1: {"sampled": False}, 2: {"sampled": True}}
        greedy, sampled = classify_decode_slots([1, 2], active, grammar_slots=[], mtp_capable=False)
        assert greedy == []
        assert sampled == [1, 2]

    def test_empty_active_slots(self):
        greedy, sampled = classify_decode_slots([], {}, grammar_slots=[], mtp_capable=True)
        assert greedy == []
        assert sampled == []


class TestServerEngineBackendSelection:
    """Real (GPU-free) ServerEngine construction -- __init__ never touches
    the GPU; model loading happens later, only on the engine thread via
    start(), which these tests never call."""

    def test_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="backend"):
            ServerEngine(backend="not-a-real-backend", capacity=1, num_slots=1)

    def test_laguna_is_the_default_backend(self):
        """vLLM removal: ServerEngine is now Laguna-only (BACKEND='laguna')."""
        engine = ServerEngine(capacity=1, num_slots=1, enable_cudagraph=False, production=True)
        assert engine.backend_name == "laguna"

    def test_laguna_backend_overrides_model_and_k(self):
        engine = ServerEngine(
            backend="laguna", capacity=1, num_slots=1, enable_cudagraph=False, production=True
        )
        assert engine.backend_name == "laguna"
        assert engine.MODEL == "poolside/Laguna-S-2.1-NVFP4"
        assert engine.K == 0
        # Laguna's generation_config.json declares eos_token_id: [2, 24] --
        # both must be in the live stop set, not just the tokenizer's single
        # eos_token (id 2).
        assert 2 in engine.eos_token_ids
        assert 24 in engine.eos_token_ids


class TestLagunaBackendE1Surface:
    """Exercises the new methods added to LagunaBackend without constructing
    a real instance (which requires a GPU + loaded model). __new__ bypasses
    __init__; every method under test either never touches `self` at all, or
    returns before reaching any GPU-backed attribute.

    Laguna imports are vLLM-free, so these tests intentionally import the
    real backend rather than masking its import closure with a test stub.
    """

    def _bare_backend(self):
        from runtime.backends.laguna import LagunaBackend

        return LagunaBackend.__new__(LagunaBackend)

    def test_rejects_unsupported_sparkinfer_page_size_before_model_load(self):
        """A stale 16-token launcher must fail before allocating GPU state."""
        from runtime.backends.laguna import LagunaBackend

        with pytest.raises(ValueError, match=r"block_size in \(64, 128\)"):
            LagunaBackend(None, block_size=16)

    def test_reconcile_prefix_hit_cold_miss(self):
        """No warm cache → always returns 0 (cold miss)."""
        backend = self._bare_backend()
        backend._prefix_cache_tokens = [None, None]
        backend._prefix_cache_kv_len = [0, 0]
        backend._pending_prefix_hits = {}
        backend.block_size = 64
        assert backend.reconcile_prefix_hit([1, 2, 3]) == 0
        assert backend.reconcile_prefix_hit([]) == 0

    def test_reconcile_prefix_hit_warm_match(self):
        """Warm cache with matching prefix → returns block-aligned hit depth."""
        backend = self._bare_backend()
        backend._prefix_cache_tokens = [[10, 20, 30, 40, 50] * 20, None]  # 100 tokens
        backend._prefix_cache_kv_len = [100, 0]
        backend._pending_prefix_hits = {}
        backend.block_size = 64
        # Prompt shares first 100 tokens → hit = 64 (block-aligned)
        prompt = [10, 20, 30, 40, 50] * 20 + [99, 98, 97]
        hit = backend.reconcile_prefix_hit(prompt)
        assert hit == 64  # block-aligned down from 100
        assert backend._pending_prefix_hits.get(0) == 64

    def test_slot_state_is_immutable_scheduler_snapshot(self):
        backend = self._bare_backend()
        backend.slot_kv_len = [3]
        backend.slot_committed_tokens = [[11, 12, 13, 14]]

        state = backend.slot_state(0)

        assert state.kv_len == 3
        assert state.committed_tokens == (11, 12, 13, 14)
        assert state.is_fresh is False
        backend.slot_committed_tokens[0].append(15)
        assert state.committed_tokens == (11, 12, 13, 14)


def test_server_engine_uses_laguna_owned_runtime_surface():
    """Keep Qwen compatibility fields out of the Laguna server path."""
    source = inspect.getsource(ServerEngine)
    for private_member in (
        "_decode_cg",
        "_dflash",
        "block_table",
        "slot_kv_len",
        "slot_committed_tokens",
    ):
        assert f"runner.{private_member}" not in source


class _FakeDecodeTok:
    """Fake tokenizer whose ``.decode()`` always returns a fixed string,
    regardless of the token ids passed in -- good enough for these tests,
    which fake ``_tokenize_decode`` too and only care about what
    ``StreamProcessor`` does with the decoded text."""

    def __init__(self, text: str):
        self._text = text

    def decode(self, _ids, skip_special_tokens=True):
        return self._text


def _fake_engine_returning(text: str):
    """A ``_FakeEngine`` whose single committed token decodes to ``text``
    (both directly, and via ``engine.tok.decode`` -- server/app.py's
    non-streaming handlers now build their own ``StreamProcessor`` fed by
    ``engine.tok``, matching the real code path, instead of only faking
    ``_tokenize_decode``)."""

    class _FakeEngine:
        MODEL = "laguna-test"
        capacity_tokens_per_slot = 128
        tok = _FakeDecodeTok(text)

        def capacity_ok(self, prompt_tokens, max_tokens):
            return True

        async def submit(self, *_args, **_kwargs):
            return {
                "committed_token_ids": [1],
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "finish_reason": "stop",
            }

    return _FakeEngine()


def _parse_sse(chunks: list[str]) -> list[dict]:
    """Parse ``event: X\\ndata: {...}\\n\\n`` chunks into
    ``[{"event": X, "data": {...}}, ...]``. The fake engines below feed
    tokens one at a time, so a single logical piece of text (e.g.
    "because X") can arrive split across many single-character SSE
    events -- callers should reconstruct it by concatenating ``delta``
    fields, not by substring-searching the raw joined SSE text."""
    import json as _json

    events = []
    for chunk in chunks:
        event_name = None
        data = None
        for line in chunk.splitlines():
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = _json.loads(line[len("data: ") :])
        if event_name is not None:
            events.append({"event": event_name, "data": data})
    return events


def _patch_server_app(monkeypatch, server_app, engine, raw_text):
    async def _tokenize_chat(*_args, **_kwargs):
        return [0]

    async def _tokenize_decode(*_args, **_kwargs):
        return raw_text

    async def _noop(*_args, **_kwargs):
        return None

    def _sync_noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server_app, "engine", engine)
    monkeypatch.setattr(server_app, "_tokenize_chat", _tokenize_chat)
    monkeypatch.setattr(server_app, "_tokenize_decode", _tokenize_decode)
    monkeypatch.setattr(server_app, "_debug_log_input", _noop)
    monkeypatch.setattr(server_app, "_debug_log_output", _sync_noop)


def test_laguna_chat_response_splits_generated_think_tags_into_reasoning_content(monkeypatch):
    """Laguna's chat template does not inject <think> (unlike Qwen3.6's),
    but Laguna has been observed to voluntarily open its own reasoning
    block as the first thing it generates. Per the reasoning/thinking
    contract (docs/api-layer-design.md, QSR_REASONING_MODE=expose is the
    default): content must NEVER contain reasoning -- the body goes to
    message.reasoning_content, and only the answer stays in content.

    This replaces the old (docs/roadmap.md R3) "must preserve verbatim"
    expectation, which encoded a contract this project no longer implements
    (unstripped <think> ending up in `content` where an ordinary client
    would never look for it)."""
    from server import app as server_app

    raw = "<think>generated by Laguna</think>answer"
    _patch_server_app(monkeypatch, server_app, _fake_engine_returning(raw), raw)

    response = asyncio.run(
        server_app.chat_completions(
            server_app.ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}]),
            request=None,
        )
    )

    message = response["choices"][0]["message"]
    assert message["content"] == "answer"
    assert message["reasoning_content"] == "generated by Laguna"


def test_laguna_chat_response_reasoning_omitted_in_strip_mode(monkeypatch):
    """QSR_REASONING_MODE=strip: reasoning is discarded, not surfaced."""
    from server import app as server_app

    raw = "<think>generated by Laguna</think>answer"
    _patch_server_app(monkeypatch, server_app, _fake_engine_returning(raw), raw)
    monkeypatch.setattr(server_app, "SERVER_REASONING_MODE", "strip")

    response = asyncio.run(
        server_app.chat_completions(
            server_app.ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}]),
            request=None,
        )
    )

    message = response["choices"][0]["message"]
    assert message["content"] == "answer"
    assert "reasoning_content" not in message


def test_laguna_chat_response_literal_think_mid_body_not_truncated(monkeypatch):
    """R4 regression: a <think> tag that is NOT the first thing generated
    is ordinary visible content (e.g. the model explaining how the tag
    works -- a high-frequency ask for a runtime that mostly serves
    code/agent workloads) and must reach the client byte-for-byte, not get
    silently truncated by a blind "strip to the first </think>" pass."""
    from server import app as server_app

    raw = (
        "Sure -- the <think> tag marks reasoning, e.g. "
        "<think>like this</think>, and </think> alone closes it."
    )
    _patch_server_app(monkeypatch, server_app, _fake_engine_returning(raw), raw)

    response = asyncio.run(
        server_app.chat_completions(
            server_app.ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}]),
            request=None,
        )
    )

    message = response["choices"][0]["message"]
    assert message["content"] == raw
    assert "reasoning_content" not in message


def test_anthropic_stream_reasoning_via_custom_event_not_thinking_block(monkeypatch):
    """Anthropic streaming: reasoning arrives as a custom
    ``reasoning_content_delta`` SSE event, never as a spec ``thinking``
    content block (see server/formats/anthropic.py's build_response
    docstring -- a real ``thinking`` block without a valid signature broke
    Claude Desktop before, commit f13fd4a; directive: do not re-add it)."""
    from server import app as server_app

    class _FakeTok:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids if 32 <= i < 127)

    class _FakeEngine:
        MODEL = "laguna-test"
        capacity_tokens_per_slot = 4096

        tok = _FakeTok()

        async def submit_stream(self, *_args, **_kwargs):
            text = "<think>because X</think>the answer"
            for ch in text:
                yield [ord(ch)]
            yield {
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": len(text),
            }

    class _FakeURL:
        query = ""

    class _FakeRequest:
        url = _FakeURL()

        async def is_disconnected(self):
            return False

        async def json(self):
            return {
                "model": "test",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }

    async def _tokenize_chat(*_args, **_kwargs):
        return [0]

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server_app, "engine", _FakeEngine())
    monkeypatch.setattr(server_app, "_tokenize_chat", _tokenize_chat)
    monkeypatch.setattr(server_app, "_debug_log_input", _noop)

    async def _run():
        resp = await server_app.anthropic_messages(_FakeRequest())
        events = []
        async for chunk in resp.body_iterator:
            events.append(chunk)
        return events

    events = asyncio.run(_run())
    joined = "".join(events)
    parsed = _parse_sse(events)

    reasoning = "".join(
        e["data"]["delta"] for e in parsed if e["event"] == "reasoning_content_delta"
    )
    content = "".join(
        e["data"]["delta"]["text"]
        for e in parsed
        if e["event"] == "content_block_delta" and e["data"]["delta"].get("type") == "text_delta"
    )
    assert reasoning == "because X"
    assert content == "the answer"
    # Never a spec thinking content block or its signature event.
    assert '"type": "thinking"' not in joined
    assert "signature_delta" not in joined


def test_anthropic_stream_reasoning_omitted_in_strip_mode(monkeypatch):
    """QSR_REASONING_MODE=strip: no reasoning_content_delta events at all,
    the answer text still streams normally."""
    from server import app as server_app

    class _FakeTok:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids if 32 <= i < 127)

    class _FakeEngine:
        MODEL = "laguna-test"
        capacity_tokens_per_slot = 4096

        tok = _FakeTok()

        async def submit_stream(self, *_args, **_kwargs):
            text = "<think>because X</think>the answer"
            for ch in text:
                yield [ord(ch)]
            yield {
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": len(text),
            }

    class _FakeURL:
        query = ""

    class _FakeRequest:
        url = _FakeURL()

        async def is_disconnected(self):
            return False

        async def json(self):
            return {
                "model": "test",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }

    async def _tokenize_chat(*_args, **_kwargs):
        return [0]

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server_app, "engine", _FakeEngine())
    monkeypatch.setattr(server_app, "_tokenize_chat", _tokenize_chat)
    monkeypatch.setattr(server_app, "_debug_log_input", _noop)
    monkeypatch.setattr(server_app, "SERVER_REASONING_MODE", "strip")

    async def _run():
        resp = await server_app.anthropic_messages(_FakeRequest())
        events = []
        async for chunk in resp.body_iterator:
            events.append(chunk)
        return events

    events = asyncio.run(_run())
    joined = "".join(events)
    parsed = _parse_sse(events)

    assert not any(e["event"] == "reasoning_content_delta" for e in parsed)
    content = "".join(
        e["data"]["delta"]["text"]
        for e in parsed
        if e["event"] == "content_block_delta" and e["data"]["delta"].get("type") == "text_delta"
    )
    assert content == "the answer"
    assert "because X" not in joined


def test_openai_stream_reasoning_content_in_delta(monkeypatch):
    """OpenAI streaming: reasoning arrives incrementally via
    delta.reasoning_content, never mixed into delta.content."""
    from server import app as server_app

    class _FakeTok:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids if 32 <= i < 127)

    class _FakeEngine:
        MODEL = "laguna-test"
        capacity_tokens_per_slot = 4096
        tok = _FakeTok()

        def capacity_ok(self, prompt_tokens, max_tokens):
            return True

        async def submit_stream(self, *_args, **_kwargs):
            text = "<think>because X</think>the answer"
            for ch in text:
                yield [ord(ch)]
            yield {
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": len(text),
            }

    class _FakeRequest:
        async def is_disconnected(self):
            return False

    async def _tokenize_chat(*_args, **_kwargs):
        return [0]

    async def _noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(server_app, "engine", _FakeEngine())
    monkeypatch.setattr(server_app, "_tokenize_chat", _tokenize_chat)
    monkeypatch.setattr(server_app, "_debug_log_input", _noop)

    async def _run():
        resp = await server_app.chat_completions(
            server_app.ChatCompletionRequest(
                messages=[{"role": "user", "content": "hi"}], stream=True
            ),
            request=_FakeRequest(),
        )
        chunks = []
        async for chunk in resp.body_iterator:
            chunks.append(chunk)
        return chunks

    import json as _json

    chunks = asyncio.run(_run())
    payloads = [
        _json.loads(c[len("data: ") :])
        for c in chunks
        if c.startswith("data: ") and c.strip() != "data: [DONE]"
    ]

    reasoning = "".join(
        p["choices"][0]["delta"]["reasoning_content"]
        for p in payloads
        if "reasoning_content" in p["choices"][0]["delta"]
    )
    content = "".join(
        p["choices"][0]["delta"]["content"]
        for p in payloads
        if "content" in p["choices"][0]["delta"]
    )
    assert reasoning == "because X"
    # First chunk's delta.content=="" (role announcement) is expected; the
    # rest of `content` must be exactly the answer, reasoning excluded.
    assert content == "the answer"


def test_openai_error_shapes_via_real_http_dispatch(monkeypatch):
    """E1 (docs/roadmap.md Track E, error-code semantics): confirmed via
    TestClient (real Starlette exception-handler dispatch, not just calling
    the handler function directly) that BOTH FastAPI default error paths
    were wrong before this fix:

    - HTTPException (e.g. our own 400s from `_invalid_request`): FastAPI's
      default handler wraps `detail` in an extra {"detail": ...} envelope,
      so a 400 meant to be {"error": {...}} actually reached the client as
      {"detail": {"error": {...}}}.
    - RequestValidationError (pydantic body validation, e.g. a missing
      required field): FastAPI's default 422 body is
      {"detail": [{"loc":..., "msg":..., "type":...}, ...]}, matching
      neither protocol at all.
    """
    from fastapi.testclient import TestClient

    from server import app as server_app

    monkeypatch.setattr(server_app, "engine", object())
    client = TestClient(server_app.app)

    bad_temperature = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "temperature": -1},
    )
    assert bad_temperature.status_code == 400
    body = bad_temperature.json()
    assert "detail" not in body
    assert body["error"]["type"] == "invalid_request_error"
    assert "temperature" in body["error"]["message"]

    missing_field = client.post("/v1/chat/completions", json={})
    assert missing_field.status_code == 422
    body2 = missing_field.json()
    assert "detail" not in body2
    assert body2["error"]["type"] == "invalid_request_error"
    assert "messages" in body2["error"]["message"]


def test_anthropic_error_shape_via_real_http_dispatch(monkeypatch):
    """Same fix, verified for /v1/messages: the error envelope must be
    Anthropic's {"type": "error", "error": {...}}, not FastAPI's default
    {"detail": ...} wrapper (nor OpenAI's un-nested {"error": {...}})."""
    from fastapi.testclient import TestClient

    from server import app as server_app

    monkeypatch.setattr(server_app, "engine", object())
    client = TestClient(server_app.app)

    resp = client.post(
        "/v1/messages",
        json={
            "model": "t",
            "max_tokens": 10,
            "temperature": -1,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    assert "detail" not in body
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"


def test_anthropic_no_messages_and_prompt_too_long_use_shared_invalid_request(monkeypatch):
    """These two checks used to hand-roll their own JSONResponse instead of
    raising through the shared `_invalid_request()` helper (the only reason
    they needed to was that the helper was OpenAI-shaped only). Now that
    `_http_exception_handler` reshapes per protocol, they route through the
    same helper as every other validation error -- one fewer bespoke
    error-construction path for a new protocol adapter to have to learn."""
    from fastapi.testclient import TestClient

    from server import app as server_app

    class _FakeEngine:
        MODEL = "laguna-test"
        capacity_tokens_per_slot = 10

    async def _tokenize_chat(*_args, **_kwargs):
        return [0] * 20  # longer than capacity_tokens_per_slot, for the 2nd case

    monkeypatch.setattr(server_app, "engine", _FakeEngine())
    monkeypatch.setattr(server_app, "_tokenize_chat", _tokenize_chat)
    client = TestClient(server_app.app)

    no_messages = client.post("/v1/messages", json={"model": "t", "max_tokens": 10, "messages": []})
    assert no_messages.status_code == 400
    body = no_messages.json()
    assert body == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "no messages provided"},
    }

    too_long = client.post(
        "/v1/messages",
        json={
            "model": "t",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi " * 50}],
        },
    )
    assert too_long.status_code == 400
    body2 = too_long.json()
    assert body2["type"] == "error"
    assert "prompt too long" in body2["error"]["message"]


def test_validate_capacity_error_metric_not_double_counted(monkeypatch):
    """_validate_capacity() must not call metrics.record_error itself
    anymore -- _http_exception_handler now records every HTTPException
    exactly once, uniformly. A stray call at the raise site would double
    the error count for this one path relative to every other validation
    error (docs/api-layer-design.md)."""
    from fastapi import HTTPException

    from server import app as server_app

    class _FakeEngine:
        capacity_tokens_per_slot = 10

        def capacity_ok(self, _prompt_tokens, _max_tokens):
            return False

    monkeypatch.setattr(server_app, "engine", _FakeEngine())
    calls = []
    monkeypatch.setattr(server_app.metrics, "record_error", lambda *a, **k: calls.append((a, k)))

    with pytest.raises(HTTPException):
        server_app._validate_capacity([1, 2, 3], 10_000_000)

    assert calls == []
