"""OpenAI Responses API (/v1/responses) formatting adapter.

Codex CLI 0.146+ removed ``wire_api = "chat"`` and only talks to custom
providers through the Responses protocol. This module translates the
Responses request shape into this server's chat-message pipeline and
translates the engine output back into Responses ``output`` items and SSE
events, so the self-built runtime can serve as a Codex model provider
without any proxy.

Only the surface Codex actually uses is implemented: ``instructions`` +
``input`` (string | items), ``tools`` (function definitions), tool
``function_call`` / ``function_call_output`` items, streaming SSE events,
and usage with cached-token accounting. Unknown request fields are ignored
by the endpoint (the body is parsed as plain JSON, not a pydantic model).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from server.formats.tools import parse_tool_calls


def _block_text(block: Any) -> str:
    """Extract the text from a Responses content block or plain string."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        return str(block.get("text") or block.get("refusal") or "")
    return ""


def parse_input(body: dict) -> list[dict]:
    """Translate Responses ``instructions`` + ``input`` into chat messages.

    Returns a list of messages in this server's internal chat shape
    (``server/formats/openai.py::parse_chat_messages`` output shape), ready
    for ``_tokenize_chat``.
    """
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if instructions:
        if isinstance(instructions, str):
            messages.append({"role": "system", "content": instructions})
        elif isinstance(instructions, list):
            text = "\n".join(_block_text(b) for b in instructions if _block_text(b))
            if text:
                messages.append({"role": "system", "content": text})

    raw_input = body.get("input")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "message":
                role = item.get("role", "user")
                if role == "developer":
                    # Responses uses "developer"; the chat template wants
                    # system-level instructions.
                    role = "system"
                content = item.get("content")
                if isinstance(content, list):
                    text = "\n".join(_block_text(b) for b in content if _block_text(b))
                else:
                    text = _block_text(content)
                messages.append({"role": role, "content": text})
            elif itype == "function_call":
                args = item.get("arguments", "")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        pass
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": item.get("call_id")
                                or f"call_{uuid.uuid4().hex[:12]}",
                                "type": "function",
                                "function": {
                                    "name": item.get("name", ""),
                                    "arguments": args,
                                },
                            }
                        ],
                    }
                )
            elif itype == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "content": str(item.get("output", "")),
                        "tool_call_id": item.get("call_id", ""),
                    }
                )
            elif itype == "reasoning":
                summary = item.get("summary")
                if summary:
                    messages.append({"role": "assistant", "content": _block_text(summary)})
            # Other item types (computer_call, etc.) are not used by Codex
            # against a chat-style backend and are ignored.
    return messages


def message_item(item_id: str, text: str) -> dict:
    """A Responses ``message`` output item for the given visible text."""
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": text, "annotations": []}
        ],
    }


def function_call_item(item_id: str, name: str, arguments: str) -> dict:
    """A Responses ``function_call`` output item."""
    return {
        "id": item_id,
        "type": "function_call",
        "call_id": f"call_{uuid.uuid4().hex[:12]}",
        "name": name,
        "arguments": arguments,
    }


def snapshot(
    resp_id: str,
    created_at: int,
    model: str,
    status: str,
    output: list[dict],
    usage: dict | None,
) -> dict:
    """The full Responses object carried by created/in_progress/completed/done."""
    return {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": usage,
        "metadata": {},
        "user": None,
    }


def build_usage(
    prompt_tokens: int,
    completion_tokens: int,
    prefix_cache_hit_tokens: int = 0,
) -> dict:
    return {
        "input_tokens": prompt_tokens,
        "input_tokens_details": {"cached_tokens": prefix_cache_hit_tokens},
        "output_tokens": completion_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": prompt_tokens + completion_tokens,
    }


def build_response(
    model: str,
    text: str,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    committed_token_ids: list[int] | None = None,
    reasoning_content: str | None = None,
    prefix_cache_hit_tokens: int = 0,
) -> dict:
    """Build a non-streaming Responses response.

    ``text`` must already have any reasoning span removed and tool-call
    blocks parsed out (i.e. ``StreamProcessor.content_text()`` output).
    """
    visible_text, tool_calls = parse_tool_calls(text)
    resp_id = f"resp_{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())
    output: list[dict] = []
    output.append(message_item(f"msg_{uuid.uuid4().hex[:24]}", visible_text or ""))
    for tc in tool_calls:
        output.append(
            function_call_item(
                f"fc_{uuid.uuid4().hex[:24]}",
                tc["name"],
                json.dumps(tc["arguments"], ensure_ascii=False),
            )
        )
    resp = snapshot(
        resp_id,
        created_at,
        model,
        "completed",
        output,
        build_usage(prompt_tokens, completion_tokens, prefix_cache_hit_tokens),
    )
    if committed_token_ids is not None:
        resp["debug_committed_token_ids"] = committed_token_ids
    if reasoning_content:
        resp["debug_reasoning_content"] = reasoning_content
    return resp
