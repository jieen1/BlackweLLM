"""Tool-call parsing and formatting.

Models emit tool calls as XML-ish text inside their generation, sharing a
common outer ``<tool_call>...</tool_call>`` delimiter but disagreeing on the
interior shape:

- Qwen3.6: ``<function=NAME>...<parameter=K>V</parameter>...</function>``
- Laguna-S-2.1 (poolside_v1, per its ``chat_template.jinja``): bare ``NAME``
  followed by zero or more ``<arg_key>K</arg_key><arg_value>V</arg_value>``
  pairs -- no ``<function=>``/``<parameter=>`` wrapper at all.

``parse_tool_calls`` detects which shape a block uses (the two are mutually
exclusive) and parses accordingly. This module parses that text into
structured tool-call objects and formats them for each API style
(OpenAI / Anthropic).

``parse_qwen_params``/``parse_poolside_args`` (the per-shape arg-block
parsers) are also used directly by ``server/formats/stream.py``'s
``drain_tool_deltas`` for streaming responses, so a tool call's arguments
are parsed identically whether the request was streamed or not, and the
streamed ``function.arguments`` delta is always valid JSON (never the raw
``<parameter=>``/``<arg_key>`` XML -- see ``drain_tool_deltas`` for why that
distinction matters).
"""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_QWEN_FUNCTION_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_QWEN_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_POOLSIDE_ARG_RE = re.compile(
    r"<arg_key>([^<]*)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
)
# Real tool names are always simple identifiers (OpenAI's function-calling
# spec requires this shape). Since the Poolside interior has no wrapper tag
# at all, this guards against misreading arbitrary/malformed <tool_call>
# content (e.g. prose) as a bogus zero-argument call.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")


def _repair_json(value: str) -> str:
    """Attempt to repair common JSON formatting errors from model output.

    Models occasionally produce near-valid JSON with predictable mutations:
    - ``{("key": ...)}`` instead of ``[{"key": ...}]`` (set-literal confusion)
    - Trailing commas before ``]`` or ``}``
    """
    repaired = value.strip()
    # Pattern: {("key": val, ...)}] -> [{"key": val, ...}]
    # The model sometimes wraps a dict in set-literal syntax {( ... )}
    # instead of putting it in an array [{ ... }].
    if repaired.startswith("{("):
        inner = repaired[2:]  # strip leading {(
        if inner.endswith("})]"):
            inner = inner[:-3] + "}]"
        elif inner.endswith(")}"):
            inner = inner[:-2] + "}]"
        elif inner.endswith(")"):
            inner = inner[:-1] + "}]"
        repaired = "[{" + inner
    # Trailing commas: ,] or ,}
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def _parse_value(raw: str) -> Any:
    """Parse one argument value: JSON if possible, else the repaired JSON,
    else the raw string verbatim (models occasionally emit bare strings
    the template doesn't quote -- see chat_template.jinja's
    ``v if v is string else v | tojson``)."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        try:
            return json.loads(_repair_json(raw))
        except (json.JSONDecodeError, ValueError):
            return raw


def parse_qwen_params(params_block: str) -> dict:
    """Parse a Qwen ``<parameter=K>V</parameter>...`` block into an args dict.

    Shared by the final-text parser below and the streaming incremental
    parser (``server/formats/stream.py``) so the two never disagree on how
    a given fragment decodes.
    """
    return {
        m.group(1).strip(): _parse_value(m.group(2).strip())
        for m in _QWEN_PARAM_RE.finditer(params_block)
    }


def parse_poolside_args(args_block: str) -> dict:
    """Parse a Poolside ``<arg_key>K</arg_key><arg_value>V</arg_value>...``
    block into an args dict. Shared with the streaming parser -- see
    ``parse_qwen_params``."""
    return {
        m.group(1).strip(): _parse_value(m.group(2).strip())
        for m in _POOLSIDE_ARG_RE.finditer(args_block)
    }


def _parse_tool_call_block(block: str) -> dict | None:
    """Parse one ``<tool_call>...</tool_call>`` block's interior.

    Returns None if the block matches neither known shape (left as visible
    text by the caller, same as if it hadn't matched at all).
    """
    func_match = _QWEN_FUNCTION_RE.search(block)
    if func_match:
        name = func_match.group(1).strip()
        arguments = parse_qwen_params(func_match.group(2))
        return {"name": name, "arguments": arguments}

    # Poolside shape: bare NAME, optionally followed by <arg_key>/<arg_value>
    # pairs -- e.g. "get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
    # or a zero-argument call, just "get_weather".
    first_arg_idx = block.find("<arg_key>")
    name = block[:first_arg_idx].strip() if first_arg_idx >= 0 else block.strip()
    if not _IDENTIFIER_RE.match(name):
        return None
    args_block = block[first_arg_idx:] if first_arg_idx >= 0 else ""
    arguments = parse_poolside_args(args_block)
    return {"name": name, "arguments": arguments}


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Parse tool calls from model output.

    Returns (visible_text, tool_calls) where visible_text is the output
    with successfully-parsed tool_call blocks removed, and tool_calls is a
    list of dicts with keys: name, arguments (dict). A block matching
    neither known interior shape is left untouched in visible_text (same
    as a non-match, not counted as a tool call).
    """
    tool_calls: list[dict] = []
    spans: list[tuple[int, int]] = []
    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        parsed = _parse_tool_call_block(match.group(1))
        if parsed is None:
            continue
        tool_calls.append(parsed)
        spans.append(match.span())
    if not spans:
        return text.strip(), tool_calls
    pieces = []
    last = 0
    for start, end in spans:
        pieces.append(text[last:start])
        last = end
    pieces.append(text[last:])
    visible = "".join(pieces).strip()
    return visible, tool_calls


def format_tool_calls_openai(tool_calls: list[dict], start_id: int = 0) -> list[dict]:
    """Format parsed tool calls for OpenAI chat completion response."""
    result = []
    for i, tc in enumerate(tool_calls):
        result.append(
            {
                "id": f"call_{start_id + i:04d}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                },
            }
        )
    return result


def format_tool_calls_anthropic(tool_calls: list[dict], start_id: int = 0) -> list[dict]:
    """Format parsed tool calls as Anthropic tool_use content blocks.

    Each tool_use block gets a globally unique ID (uuid4-based) so that
    IDs never collide across turns in a multi-turn conversation.  The
    previous sequential scheme (toolu_0000, toolu_0001, ...) reused the
    same IDs in every assistant turn, which confused Claude Desktop's
    tool_result matching.
    """
    import uuid as _uuid

    result = []
    for tc in tool_calls:
        result.append(
            {
                "type": "tool_use",
                "id": f"toolu_{_uuid.uuid4().hex[:24]}",
                "name": tc["name"],
                "input": tc["arguments"],
            }
        )
    return result


# Anthropic server-side tool types that cannot be executed by a local model.
# These are skipped during tool conversion (the model cannot call them).
_SERVER_TOOL_TYPE_PREFIXES = (
    "web_search_",
    "code_execution_",
    "computer_",
    "text_editor_",
    "bash_",
)


def convert_tools_to_chat_template(tools: list[dict] | None) -> list[dict] | None:
    """Convert OpenAI/Anthropic tool definitions to the format expected
    by the Qwen3.6 chat template (list of function dicts).

    The chat template expects tools as a list of dicts, each with
    type=function and a function sub-dict with name/description/parameters.

    Anthropic server-side tools (web_search_20250305, code_execution_*, etc.)
    are skipped because they cannot be executed by a local model.
    """
    if not tools:
        return None
    converted = []
    for tool in tools:
        # Skip Anthropic server-side tools (web_search, code_execution, etc.)
        tool_type = tool.get("type", "")
        if any(tool_type.startswith(p) for p in _SERVER_TOOL_TYPE_PREFIXES):
            continue
        if "function" in tool:
            converted.append(tool)
        elif "name" in tool:
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", tool.get("parameters", {})),
                    },
                }
            )
        else:
            converted.append(tool)
    return converted or None


# -- Streaming tool-call detection ------------------------------------------

_TOOL_CALL_OPEN = chr(60) + "tool_call" + chr(62)


def find_tool_call_start(text: str) -> int:
    """Find the earliest position where a tool call block might be starting.

    Returns the index of the first character of the potential tool call,
    or -1 if no tool call start is detected.

    We look for progressively shorter prefixes of the opening tag to catch
    partial matches at the end of a streaming buffer (e.g. the model has
    emitted '<tool' but not yet '_call>').
    """
    # Full tag present
    idx = text.find(_TOOL_CALL_OPEN)
    if idx >= 0:
        return idx
    # Partial prefixes at the very end of the text (streaming edge case)
    for length in range(len(_TOOL_CALL_OPEN) - 1, 0, -1):
        prefix = _TOOL_CALL_OPEN[:length]
        if text.endswith(prefix):
            return len(text) - length
    return -1
