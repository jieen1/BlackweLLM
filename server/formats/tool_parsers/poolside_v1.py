"""Poolside's ``poolside_v1`` tool-call shape (Laguna-S-2.1).

Per its ``chat_template.jinja``: a bare function NAME directly inside
``<tool_call>...</tool_call>``, followed by zero or more
``<arg_key>K</arg_key><arg_value>V</arg_value>`` pairs -- no
``<function=>``/``<parameter=>`` wrapper at all::

    <tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>
"""

from __future__ import annotations

import re

from server.formats.tool_parsers.base import ToolCallParser
from server.formats.tool_parsers.value_parsing import parse_value

_ARG_KEY_OPEN = "<arg_key>"
_ARG_RE = re.compile(r"<arg_key>([^<]*)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
# Real tool names are always simple identifiers (OpenAI's function-calling
# spec requires this shape). Since this shape has no wrapper tag at all,
# this guards against misreading arbitrary/malformed <tool_call> content
# (e.g. prose) as a bogus zero-argument call.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")


class PoolsideV1ToolCallParser(ToolCallParser):
    name = "poolside_v1"

    def parse_block(self, interior: str) -> dict | None:
        first_arg_idx = interior.find(_ARG_KEY_OPEN)
        call_name = interior[:first_arg_idx].strip() if first_arg_idx >= 0 else interior.strip()
        if not _IDENTIFIER_RE.match(call_name):
            return None
        args_block = interior[first_arg_idx:] if first_arg_idx >= 0 else ""
        arguments = {
            m.group(1).strip(): parse_value(m.group(2).strip())
            for m in _ARG_RE.finditer(args_block)
        }
        return {"name": call_name, "arguments": arguments}

    def find_name_boundary(self, interior_so_far: str, block_closed: bool) -> str | None:
        arg_key_pos = interior_so_far.find(_ARG_KEY_OPEN)
        if arg_key_pos >= 0:
            call_name = interior_so_far[:arg_key_pos].strip()
        elif block_closed:
            # Zero-argument call: nothing follows the name but the close
            # tag. Only safe to conclude once the close tag has actually
            # arrived -- more text (an <arg_key>) could still be coming.
            call_name = interior_so_far.strip()
        else:
            return None
        return call_name if _IDENTIFIER_RE.match(call_name) else None
