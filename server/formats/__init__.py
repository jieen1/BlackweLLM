"""Format compatibility layer for BlackweLLM server.

This package handles ALL input/output format conversion between external
API formats (OpenAI, Anthropic) and the internal engine representation.

Sub-modules:
- thinking: reasoning-span detection (find_reasoning_span) + <usage>
  artifact stripping; strip_thinking() is the text-only QSR_REASONING_MODE
  =strip helper
- stream: StreamProcessor -- the stateful reasoning/content/tool-call state
  machine both the streaming and non-streaming response paths reuse
- content: parse flexible content fields (string | array of blocks)
- tools: parse tool calls from model XML output, format for each API
- openai: OpenAI Chat Completions request/response formatting
- anthropic: Anthropic Messages API request/response formatting

Design principle: app.py handles routing and engine interaction only.
All format parsing/serialization lives in this package.
"""

from server.formats import anthropic as anthropic_format
from server.formats import openai as openai_format
from server.formats.content import extract_blocks, extract_text
from server.formats.stream import StreamProcessor
from server.formats.thinking import strip_thinking
from server.formats.tools import (
    convert_tools_to_chat_template,
    find_tool_call_start,
    format_tool_calls_anthropic,
    format_tool_calls_openai,
    parse_tool_calls,
)

__all__ = [
    "strip_thinking",
    "extract_text",
    "extract_blocks",
    "parse_tool_calls",
    "format_tool_calls_openai",
    "format_tool_calls_anthropic",
    "convert_tools_to_chat_template",
    "find_tool_call_start",
    "StreamProcessor",
    "openai_format",
    "anthropic_format",
]
