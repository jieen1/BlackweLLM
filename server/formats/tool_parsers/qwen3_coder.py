"""Qwen3.6's ``qwen3_coder`` tool-call shape.

Per its ``chat_template.jinja`` (and the literal format example it prompts
the model with): ``<function=NAME>`` wraps zero or more
``<parameter=K>V</parameter>`` pairs, itself nested inside
``<tool_call>...</tool_call>``::

    <tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>
"""

from __future__ import annotations

import re

from server.formats.tool_parsers.base import ToolCallParser
from server.formats.tool_parsers.value_parsing import parse_value

_FUNC_OPEN = "<function="
_FUNC_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


class Qwen3CoderToolCallParser(ToolCallParser):
    name = "qwen3_coder"

    def parse_block(self, interior: str) -> dict | None:
        match = _FUNC_RE.search(interior)
        if not match:
            return None
        arguments = {
            m.group(1).strip(): parse_value(m.group(2).strip())
            for m in _PARAM_RE.finditer(match.group(2))
        }
        return {"name": match.group(1).strip(), "arguments": arguments}

    def find_name_boundary(self, interior_so_far: str, block_closed: bool) -> str | None:
        func_pos = interior_so_far.find(_FUNC_OPEN)
        if func_pos < 0:
            return None
        name_end = interior_so_far.find(">", func_pos + len(_FUNC_OPEN))
        if name_end < 0:
            return None
        call_name = interior_so_far[func_pos + len(_FUNC_OPEN) : name_end].strip()
        return call_name or None
