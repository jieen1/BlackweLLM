"""OpenAI Chat Completions API formatting.

Handles request parsing and response formatting for the OpenAI-compatible
/v1/chat/completions endpoint.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from server.formats.content import (
    content_has_images,
    content_has_videos,
    extract_text,
    normalize_content_blocks,
)
from server.formats.tools import format_tool_calls_openai, parse_tool_calls


def parse_chat_messages(body: dict) -> list[dict]:
    """Parse OpenAI chat messages, handling flexible content types.

    Accepts content as string or list of content blocks.
    Also handles tool_calls in assistant messages and tool role messages.
    """
    messages = []

    def content_for_template(field: Any, *, allow_images: bool = True) -> Any:
        if allow_images and (content_has_images(field) or content_has_videos(field)):
            return normalize_content_blocks(field)
        return extract_text(field)

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        entry: dict[str, Any] = {"role": role}

        if role == "tool":
            entry["content"] = extract_text(msg.get("content"))
            if "tool_call_id" in msg:
                entry["tool_call_id"] = msg["tool_call_id"]
        elif role == "assistant" and "tool_calls" in msg:
            entry["content"] = content_for_template(msg.get("content"))
            # Chat template expects arguments as dict, not JSON string
            converted_calls = []
            for tc in msg["tool_calls"]:
                tc_copy = dict(tc)
                if "function" in tc_copy:
                    fn = dict(tc_copy["function"])
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"])
                        except (json.JSONDecodeError, ValueError):
                            pass
                    tc_copy["function"] = fn
                converted_calls.append(tc_copy)
            entry["tool_calls"] = converted_calls
        else:
            entry["content"] = content_for_template(msg.get("content"))

        # Qwen3.6 and Qwen3.8 chat templates consume this field when
        # preserve_thinking=True.  Dropping it here silently erased prior
        # assistant reasoning before apply_chat_template saw the messages.
        if role == "assistant" and isinstance(msg.get("reasoning_content"), str):
            entry["reasoning_content"] = msg["reasoning_content"]

        messages.append(entry)
    return messages


def build_response(
    model: str,
    text: str,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    committed_token_ids: list[int] | None = None,
    prompt_token_ids: list[int] | None = None,
    reasoning_content: str | None = None,
) -> dict:
    """Build a non-streaming OpenAI chat completion response.

    ``text`` must already have any reasoning span removed (e.g. via
    ``StreamProcessor.content_text()``) -- this function only extracts
    tool calls out of it. ``reasoning_content`` (when the server is
    running with ``QSR_REASONING_MODE=expose``, the default) is surfaced
    as ``message.reasoning_content``, mirroring vLLM's ``--reasoning-parser``
    convention -- never mixed into ``content``.
    """
    visible_text, tool_calls = parse_tool_calls(text)

    message: dict[str, Any] = {"role": "assistant"}
    if tool_calls:
        message["content"] = visible_text or None
        message["tool_calls"] = format_tool_calls_openai(tool_calls)
        finish_reason = "tool_calls"
    else:
        message["content"] = visible_text
    message["refusal"] = None
    if reasoning_content:
        message["reasoning_content"] = reasoning_content

    resp: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if committed_token_ids is not None:
        resp["debug_committed_token_ids"] = committed_token_ids
    if prompt_token_ids is not None:
        resp["debug_prompt_token_ids"] = prompt_token_ids
    return resp


def build_sse_chunks(
    cmpl_id: str,
    model: str,
    text: str,
    finish_reason: str,
    created: int,
):
    """Generate OpenAI SSE stream chunks (yields strings)."""
    visible_text, tool_calls = parse_tool_calls(text)

    if tool_calls:
        finish_reason = "tool_calls"
        if visible_text:
            chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": visible_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        for i, tc in enumerate(format_tool_calls_openai(tool_calls)):
            chunk = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": i, **tc}]},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
    else:
        chunk = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": visible_text},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    done = {
        "id": cmpl_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"
