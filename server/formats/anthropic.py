"""Anthropic Messages API formatting.

Handles request parsing and response formatting for the Anthropic-compatible
/v1/messages endpoint. Follows the same pattern as vLLM's anthropic serving
layer: convert Anthropic request -> internal chat messages -> format response.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from server.formats.content import extract_blocks, extract_text
from server.formats.tools import format_tool_calls_anthropic, parse_tool_calls

_BILLING_HEADER_PREFIX = "x-anthropic-billing-header"


def _strip_billing_blocks(blocks: list[dict]) -> list[dict]:
    """Drop Claude Code's per-request billing/attribution header blocks.

    These carry a per-request hash that (a) pollutes the system prompt and
    (b) defeats prefix caching. Mirrors vLLM's anthropic serving layer, which
    skips any text block starting with ``x-anthropic-billing-header``.
    """
    return [
        b
        for b in blocks
        if not (
            isinstance(b, dict)
            and b.get("type") == "text"
            and isinstance(b.get("text"), str)
            and b["text"].startswith(_BILLING_HEADER_PREFIX)
        )
    ]


# Anthropic server-side tool types that we cannot execute locally.
# These are passed in the tools array but must be skipped when converting
# to the chat template (the model cannot call them).
_SERVER_TOOL_TYPES = frozenset(
    {
        "web_search_20250305",
        "web_search_20250306",
        "code_execution_20250115",
        "computer_20250124",
        "text_editor_20250124",
        "bash_20250124",
    }
)

# Content block types in assistant messages that carry no actionable content
# for the chat template (thinking is internal, server_tool_use is opaque).
_ASSISTANT_SKIP_TYPES = frozenset(
    {"thinking", "redacted_thinking", "server_tool_use", "server_tool_result"}
)

# Content block types in user messages that we extract text from.
_USER_TEXT_EXTRACT_TYPES = frozenset(
    {"web_search_tool_result", "search_result", "code_execution_tool_result"}
)

# Content block types in user messages that are silently ignored.
_USER_IGNORE_TYPES = frozenset(
    {"image", "document", "mcp_tool_use", "mcp_tool_result", "container_upload"}
)


def parse_messages(body: dict) -> list[dict]:
    """Convert Anthropic Messages API request body to chat-template messages.

    Handles all content block types defined by the Anthropic Messages API:
    - system: string | list of text blocks (with cache_control etc.)
    - messages[].content: string | list of content blocks
    - text, thinking, redacted_thinking, tool_use, tool_result
    - server_tool_use, web_search_tool_result, search_result
    - image, document, mcp_tool_use, mcp_tool_result (gracefully ignored)
    - Multi-turn conversations with user/assistant roles
    """
    chat_messages: list[dict] = []

    # System message (strip Claude Code's billing-header block first)
    system_field = body.get("system")
    if isinstance(system_field, list):
        system_field = _strip_billing_blocks(system_field)
    elif isinstance(system_field, str) and system_field.startswith(_BILLING_HEADER_PREFIX):
        system_field = ""
    system_text = extract_text(system_field)
    if system_text:
        chat_messages.append({"role": "system", "content": system_text})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        blocks = _strip_billing_blocks(extract_blocks(msg.get("content")))

        if role == "assistant":
            text_parts = []
            tool_calls = []
            for block in blocks:
                btype = block.get("type", "text")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": block.get("input", {}),
                            },
                        }
                    )
                elif btype in _ASSISTANT_SKIP_TYPES:
                    pass  # thinking, redacted_thinking, server_tool_use: skip
            entry: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts)}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            chat_messages.append(entry)

        elif role == "user":
            text_parts = []
            tool_results = []
            for block in blocks:
                btype = block.get("type", "text")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_content = extract_text(result_content)
                    tool_results.append(
                        {
                            "role": "tool",
                            "content": str(result_content),
                            "tool_call_id": block.get("tool_use_id", ""),
                        }
                    )
                elif btype in _USER_TEXT_EXTRACT_TYPES:
                    # web_search_tool_result, search_result, etc.
                    # Extract any text content so the model sees search results.
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_content = extract_text(result_content)
                    elif isinstance(result_content, str) and result_content:
                        pass
                    else:
                        result_content = ""
                    if result_content:
                        text_parts.append(str(result_content))
                elif btype in _USER_IGNORE_TYPES:
                    pass  # image, document, mcp blocks: not supported, skip
            # tool results go first (they respond to the previous assistant turn)
            for tr in tool_results:
                chat_messages.append(tr)
            if text_parts:
                chat_messages.append({"role": "user", "content": "\n".join(text_parts)})
            elif not tool_results:
                chat_messages.append({"role": "user", "content": ""})
        else:
            chat_messages.append({"role": role, "content": extract_text(msg.get("content"))})

    return chat_messages


def build_response(
    model: str,
    text: str,
    finish_reason: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    reasoning_content: str | None = None,
) -> dict:
    """Build a non-streaming Anthropic Messages API response.

    ``text`` must already have any reasoning span removed (e.g. via
    ``StreamProcessor.content_text()``) -- this function only extracts
    tool calls out of it. ``reasoning_content`` (when the server is
    running with ``QSR_REASONING_MODE=expose``, the default) is surfaced
    as a top-level ``reasoning_content`` field, NOT as a spec ``thinking``
    content block: we cannot produce the cryptographic signature real
    Anthropic thinking blocks carry, and shipping one anyway is what broke
    Claude Desktop before (commit f13fd4a; "Directive: Do NOT re-add
    thinking block emission without a valid signature source"). An
    additive top-level field is ignored by permissive clients instead of
    being validated as a member of the ``content`` block type union.
    """
    visible_text, tool_calls = parse_tool_calls(text)
    stop_reason = "end_turn" if finish_reason == "stop" else "max_tokens"

    content_blocks: list[dict] = []
    if visible_text:
        content_blocks.append({"type": "text", "text": visible_text})
    if tool_calls:
        content_blocks.extend(format_tool_calls_anthropic(tool_calls))
        stop_reason = "tool_use"
    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    resp: dict[str, Any] = {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cache_read_input_tokens,
        },
    }
    if reasoning_content:
        resp["reasoning_content"] = reasoning_content
    return resp


def build_sse_events(
    model: str,
    text: str,
    finish_reason: str,
    input_tokens: int,
    output_tokens: int,
):
    """Generate Anthropic SSE stream events (yields strings)."""
    visible_text, tool_calls = parse_tool_calls(text)
    stop_reason = "end_turn" if finish_reason == "stop" else "max_tokens"
    if tool_calls:
        stop_reason = "tool_use"

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    msg_start = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(msg_start)}\n\n"

    block_index = 0
    if visible_text:
        bs = {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {"type": "text", "text": ""},
        }
        yield f"event: content_block_start\ndata: {json.dumps(bs)}\n\n"
        yield "event: ping\ndata: " + json.dumps({"type": "ping"}) + "\n\n"
        delta = {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": "text_delta", "text": visible_text},
        }
        yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
        yield (
            "event: content_block_stop\ndata: "
            + json.dumps({"type": "content_block_stop", "index": block_index})
            + "\n\n"
        )
        block_index += 1

    for tc in format_tool_calls_anthropic(tool_calls):
        bs = {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": {}},
        }
        yield f"event: content_block_start\ndata: {json.dumps(bs)}\n\n"
        delta = {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(tc["input"])},
        }
        yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
        yield (
            "event: content_block_stop\ndata: "
            + json.dumps({"type": "content_block_stop", "index": block_index})
            + "\n\n"
        )
        block_index += 1

    msg_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }
    yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
    yield "event: message_stop\ndata: " + json.dumps({"type": "message_stop"}) + "\n\n"
