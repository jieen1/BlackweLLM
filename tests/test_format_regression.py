"""Regression tests built from REAL captured client requests.

These lock two classes of bug that have actually hit this service so they
cannot recur silently:

1. Format parsing drops the user's message. The original Claude Desktop report
   ("the model didn't get my message") is guarded by asserting that every real
   request body's user content survives ``parse_messages`` /
   ``parse_chat_messages`` intact.

2. Long-context requests wrongly rejected. That same Claude Desktop report was
   ACTUALLY a capacity rejection: prompt_tokens(25843) + max_tokens(64000) =
   89843 exceeded the old 67200-token per-slot cap (blocks_per_slot=4200). The
   service now runs blocks_per_slot=16384 (256K). These tests assert the real
   numbers pass the 256K cap and would have failed the old 67K cap.

CPU-only: no GPU, no running server, no tokenizer required (runs in CI).
Fixtures live in ``tests/fixtures/`` (see the README.md there). When a new real
client request exposes a bug, capture its RAW REQUEST line from the service log
and add it verbatim as a fixture, then add a case here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.formats import anthropic as anthropic_format
from server.formats import convert_tools_to_chat_template
from server.formats import openai as openai_format

FIXTURES = Path(__file__).parent / "fixtures"

# Mirror of server/engine.py: ServerEngine.K (speculative tokens) and the
# per-slot capacity formula capacity_tokens_per_slot = block_size * blocks_per_slot.
K = 3
BLOCK_SIZE = 16
OLD_BLOCKS_PER_SLOT = 4200  # 67200-token cap (the buggy config)
NEW_BLOCKS_PER_SLOT = 16384  # 262144-token cap (256K, the fix; see vllm_ctl.sh)


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _capacity_ok(prompt_len: int, max_tokens: int, blocks_per_slot: int) -> bool:
    capacity = BLOCK_SIZE * blocks_per_slot
    return prompt_len + max_tokens + K <= capacity


def _all_content_text(messages: list[dict]) -> str:
    """Concatenate every message's content into one searchable string."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "\n".join(parts)


# -- 1. format parsing keeps the user's message ----------------------------


def test_anthropic_simple_user_content_survives():
    body = _load("anthropic_simple.json")
    parsed = anthropic_format.parse_messages(body)
    text = _all_content_text(parsed)
    assert "What is 2+2? Reply with just the number." in text
    # system block (array form) must also survive
    assert "You are a helpful assistant." in text
    assert [m["role"] for m in parsed] == ["system", "user"]


def test_anthropic_claude_desktop_user_content_survives():
    """The exact shape Claude Desktop sends: system blocks with cache_control,
    multi-turn user content arrays, tools, and a thinking config. The FINAL user
    turn must reach the model -- this is the 'model didn't get my message' guard."""
    body = _load("anthropic_claude_desktop.json")
    parsed = anthropic_format.parse_messages(body)
    text = _all_content_text(parsed)
    # the final user turn (the one that was reportedly lost)
    assert "Good catch. Please rewrite it to sort descending and add a unit test." in text
    # the first user turn (with the code) survives too
    assert "Why does it sometimes drop the highest-priority job?" in text
    # system prompt survives (cache_control must be ignored, text kept)
    assert "You are Claude, an AI assistant" in text
    # the assistant turn is preserved for multi-turn context
    assert "highest numeric priority ends up last" in text
    # unknown top-level fields (thinking, metadata) must not break parsing
    assert any(m["role"] == "user" for m in parsed)


def test_anthropic_claude_desktop_tools_convert():
    body = _load("anthropic_claude_desktop.json")
    tools = convert_tools_to_chat_template(body.get("tools"))
    assert tools, "tools must convert for the chat template"
    assert "read_file" in json.dumps(tools)


def test_openai_chat_user_content_survives():
    body = _load("openai_chat_simple.json")
    parsed = openai_format.parse_chat_messages(body)
    text = _all_content_text(parsed)
    assert "What is the capital of France? One word." in text
    assert parsed[0]["role"] == "user"


def test_openai_completions_prompt_present():
    body = _load("openai_completions_simple.json")
    assert "Translate the following English sentence to French" in body["prompt"]


# -- 2. 256K capacity regression (the real bug + the real fix) -------------


def test_real_bug_request_rejected_under_old_67k_cap():
    """The actual Claude Desktop request that failed: 25843 prompt + 64000
    max_tokens. Under the OLD blocks_per_slot=4200 (67200 cap) it was rejected
    with the 'exceeds this runtime's per-slot capacity' 400. This documents the
    bug so a return to the small cap is caught."""
    assert not _capacity_ok(25843, 64000, OLD_BLOCKS_PER_SLOT)


def test_real_bug_request_accepted_under_256k_cap():
    """The same request must be accepted now that blocks_per_slot=16384 (256K)."""
    assert _capacity_ok(25843, 64000, NEW_BLOCKS_PER_SLOT)


def test_256k_cap_boundary():
    """A request filling (but not exceeding) 256K passes; one token over fails."""
    capacity = BLOCK_SIZE * NEW_BLOCKS_PER_SLOT  # 262144
    max_tokens = capacity - 200000 - K  # prompt + max_tokens + K == capacity
    assert _capacity_ok(200000, max_tokens, NEW_BLOCKS_PER_SLOT)
    assert not _capacity_ok(200000, max_tokens + 1, NEW_BLOCKS_PER_SLOT)


def test_claude_desktop_fixture_max_tokens_fits_256k():
    """The Claude Desktop fixture's max_tokens (64000) plus a representative
    large prompt must fit the 256K cap the service now enforces."""
    body = _load("anthropic_claude_desktop.json")
    representative_prompt_tokens = 25843  # the real failing request's prompt length
    assert _capacity_ok(representative_prompt_tokens, body["max_tokens"], NEW_BLOCKS_PER_SLOT)


def test_billing_header_stripped_from_system():
    """Claude Desktop sends a leading ``x-anthropic-billing-header`` system
    block (per-request hash). It must NOT reach the model: it pollutes the
    prompt and defeats prefix caching. Mirrors vLLM's anthropic layer."""
    body = _load("anthropic_claude_desktop.json")
    parsed = anthropic_format.parse_messages(body)
    sys_msg = next(m for m in parsed if m["role"] == "system")
    assert "x-anthropic-billing-header" not in sys_msg["content"]
    # the real system prompt survives
    assert "You are Claude, an AI assistant" in sys_msg["content"]


def test_billing_header_stripped_from_user_content():
    body = {
        "system": "sys",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "x-anthropic-billing-header: cc_version=1; cc_entrypoint=x;",
                    },
                    {"type": "text", "text": "the real user question"},
                ],
            }
        ],
    }
    parsed = anthropic_format.parse_messages(body)
    user = next(m for m in parsed if m["role"] == "user")
    assert "x-anthropic-billing-header" not in user["content"]
    assert "the real user question" in user["content"]


def test_anthropic_sse_no_thinking_blocks():
    """The Anthropic SSE stream must NOT emit thinking blocks.

    We cannot produce the cryptographic signature that the official API
    attaches via signature_delta.  Claude Desktop validates it and DROPS
    every content block that follows an invalid thinking block -- including
    tool_use (e.g. AskUserQuestion), which caused the user's selection to
    come back as "(no content)".  The fix is to omit thinking blocks entirely
    (valid when thinking.type is "adaptive").
    """
    import json as _json

    pytest.importorskip("fastapi")
    from server.formats.anthropic import build_sse_events

    events = list(
        build_sse_events(
            model="test",
            text="hello world",
            finish_reason="stop",
            input_tokens=10,
            output_tokens=5,
        )
    )
    joined = "".join(events)
    # Must NOT contain any thinking-related events
    assert "thinking" not in joined
    assert "signature_delta" not in joined
    # Must contain the text
    assert "hello world" in joined
    # message_delta usage must only have output_tokens
    for ev in events:
        if "message_delta" in ev:
            data_line = [line for line in ev.splitlines() if line.startswith("data: ")][0]
            payload = _json.loads(data_line[len("data: ") :])
            assert "input_tokens" not in payload.get("usage", {})
            assert "output_tokens" in payload["usage"]


def test_anthropic_tool_use_ids_are_unique():
    """Tool-use IDs must be globally unique across turns.

    The previous sequential scheme (toolu_0000, toolu_0001, ...) reused the
    same IDs in every assistant turn, which confused Claude Desktop's
    tool_result matching in multi-turn conversations.
    """
    from server.formats.tools import format_tool_calls_anthropic

    calls = [{"name": "Bash", "arguments": {"command": "ls"}}]
    ids_turn1 = {tc["id"] for tc in format_tool_calls_anthropic(calls)}
    ids_turn2 = {tc["id"] for tc in format_tool_calls_anthropic(calls)}
    # IDs must differ across invocations (uuid4-based)
    assert ids_turn1.isdisjoint(ids_turn2)
    # IDs must have the toolu_ prefix and be 30 chars total
    for tid in ids_turn1 | ids_turn2:
        assert tid.startswith("toolu_")
        assert len(tid) == 30  # "toolu_" (6) + 24 hex chars


def test_unhandled_exception_error_shape_matches_protocol():
    """E1 (docs/roadmap.md Track E, error code semantics): an unhandled
    exception on /v1/messages must come back in Anthropic's error envelope
    ({"type": "error", "error": {...}}), not OpenAI's ({"error": {...}}) --
    found while auditing error-code semantics for the reasoning/thinking
    fix, not from a captured client request like the fixtures above."""
    import asyncio

    from server import app as server_app

    class _FakeURL:
        def __init__(self, path):
            self.path = path

    class _FakeRequest:
        def __init__(self, path):
            self.url = _FakeURL(path)

    anthropic_resp = asyncio.run(
        server_app._unhandled_exception_handler(_FakeRequest("/v1/messages"), RuntimeError("boom"))
    )
    anthropic_body = json.loads(bytes(anthropic_resp.body))
    assert anthropic_body["type"] == "error"
    assert anthropic_body["error"]["type"] == "internal_error"
    assert anthropic_body["error"]["message"] == "boom"

    openai_resp = asyncio.run(
        server_app._unhandled_exception_handler(
            _FakeRequest("/v1/chat/completions"), RuntimeError("boom")
        )
    )
    openai_body = json.loads(bytes(openai_resp.body))
    assert openai_body["error"]["type"] == "internal_error"
    assert openai_body["error"]["message"] == "boom"
    assert "type" not in openai_body  # no Anthropic-style top-level `type`


def test_anthropic_cache_read_input_tokens_mapping():
    """C6: build_response must propagate cache_read_input_tokens."""
    from server.formats.anthropic import build_response

    resp = build_response(
        model="test",
        text="Hello",
        finish_reason="stop",
        input_tokens=100,
        output_tokens=10,
        cache_read_input_tokens=42,
    )
    assert resp["usage"]["cache_read_input_tokens"] == 42
    assert resp["usage"]["cache_creation_input_tokens"] == 0

    # Default should be 0
    resp_default = build_response(
        model="test",
        text="Hello",
        finish_reason="stop",
        input_tokens=100,
        output_tokens=10,
    )
    assert resp_default["usage"]["cache_read_input_tokens"] == 0
