"""Tool-call parsing and formatting.

Each model family has its own on-the-wire shape for a tool call --
``server/formats/tool_parsers/`` holds one ``ToolCallParser`` per shape
(``poolside_v1`` for Laguna-S-2.1, ``qwen3_coder`` for Qwen3.6) selected via
a registry, mirroring vLLM's own ``--tool-call-parser NAME`` design. See
``tool_parsers/base.py`` for why a new model means a new parser module, not
a config lookup.

This module is the shape-agnostic layer above that registry: it scans text
for ``<tool_call>...</tool_call>`` blocks using the active (or an explicitly
passed) parser and formats the result for each API style (OpenAI /
Anthropic). It also backs ``server/formats/stream.py``'s streaming path, so
a tool call parses identically whether the request was streamed or not.
"""

from __future__ import annotations

import json
import uuid

from server.formats.tool_parsers import ToolCallParser, get_active_parser

__all__ = [
    "parse_tool_calls",
    "format_tool_calls_openai",
    "format_tool_calls_anthropic",
    "new_tool_call_id",
    "convert_tools_to_chat_template",
    "find_tool_call_start",
]


def parse_tool_calls(text: str, parser: ToolCallParser | None = None) -> tuple[str, list[dict]]:
    """Parse tool calls from model output.

    ``parser`` defaults to the process's active parser (set once at server
    startup from the loaded model's config) -- pass one explicitly to parse
    a specific shape regardless of what's active (e.g. in tests).

    Returns (visible_text, tool_calls) where visible_text is the output
    with successfully-parsed tool_call blocks removed, and tool_calls is a
    list of dicts with keys: name, arguments (dict). A block whose interior
    the parser doesn't recognize is left untouched in visible_text (same as
    a non-match, not counted as a tool call).
    """
    parser = parser or get_active_parser()
    tool_calls: list[dict] = []
    spans: list[tuple[int, int]] = []
    search_start = 0
    while True:
        start = text.find(parser.open_tag, search_start)
        if start < 0:
            break
        interior_start = start + len(parser.open_tag)
        end = text.find(parser.close_tag, interior_start)
        if end < 0:
            break
        block_end = end + len(parser.close_tag)
        parsed = parser.parse_block(text[interior_start:end])
        if parsed is not None:
            tool_calls.append(parsed)
            spans.append((start, block_end))
        search_start = block_end
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


def new_tool_call_id() -> str:
    """Return an opaque tool-call ID unique across response turns."""
    return f"call_{uuid.uuid4().hex[:24]}"


def format_tool_calls_openai(
    tool_calls: list[dict], start_id: int | None = None
) -> list[dict]:
    """Format parsed tool calls for an OpenAI chat completion response.

    The old default (``call_0000`` on every response) reused the same ID on
    every tool-call turn.  Anthropic-to-OpenAI relays use that ID to match
    tool results; reusing it makes a client treat later turns as the same
    call and can drive an agent into repeating the previous tool forever.
    ``start_id`` remains as an explicit deterministic test/compatibility
    escape hatch, while normal responses receive opaque unique IDs.
    """
    result = []
    for i, tc in enumerate(tool_calls):
        call_id = (
            f"call_{start_id + i:04d}" if start_id is not None else new_tool_call_id()
        )
        result.append(
            {
                "id": call_id,
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

_DEFAULT_TOOL_CALL_OPEN = chr(60) + "tool_call" + chr(62)


def find_tool_call_start(text: str, open_tag: str = _DEFAULT_TOOL_CALL_OPEN) -> int:
    """Find the earliest position where a tool call block might be starting.

    ``open_tag`` defaults to the ``<tool_call>`` wrapper every parser has
    used so far (see ``tool_parsers/base.py``); pass the active parser's
    own ``open_tag`` explicitly if it ever differs.

    Returns the index of the first character of the potential tool call,
    or -1 if no tool call start is detected.

    We look for progressively shorter prefixes of the opening tag to catch
    partial matches at the end of a streaming buffer (e.g. the model has
    emitted '<tool' but not yet '_call>').
    """
    # Full tag present
    idx = text.find(open_tag)
    if idx >= 0:
        return idx
    # Partial prefixes at the very end of the text (streaming edge case)
    for length in range(len(open_tag) - 1, 0, -1):
        prefix = open_tag[:length]
        if text.endswith(prefix):
            return len(text) - length
    return -1
