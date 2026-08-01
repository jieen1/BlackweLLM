"""Tool-call parser registry: name -> ``ToolCallParser`` instance.

This is the only file that needs a new line when a new model's parser is
added -- ``poolside_v1.py``/``qwen3_coder.py`` (and any future parser
module) are never imported by each other, only by this module.

The "active" parser is process-global, set once at server startup from
config (mirrors vLLM's ``--tool-call-parser NAME``) -- this runtime loads
one model per process, so there is exactly one tool-call shape in play at
any time. Call sites that need a specific parser regardless of what is
active (e.g. tests exercising one shape in isolation) can pass an explicit
``ToolCallParser`` instead of relying on the active one.
"""

from __future__ import annotations

from server.formats.tool_parsers.base import ToolCallParser
from server.formats.tool_parsers.poolside_v1 import PoolsideV1ToolCallParser
from server.formats.tool_parsers.qwen3_coder import Qwen3CoderToolCallParser

_PARSERS: dict[str, ToolCallParser] = {
    parser.name: parser for parser in (PoolsideV1ToolCallParser(), Qwen3CoderToolCallParser())
}

# Matches this project's currently (and so far only) production model,
# poolside/Laguna-S-2.1-NVFP4 -- see server/app.py's QSR_TOOL_CALL_PARSER /
# --tool-call-parser for how a differently-shaped model overrides this.
_active_name = "poolside_v1"


def available_parsers() -> list[str]:
    return sorted(_PARSERS)


def get_parser(name: str) -> ToolCallParser:
    try:
        return _PARSERS[name]
    except KeyError:
        raise ValueError(
            f"unknown tool_call_parser={name!r}; available: {available_parsers()}"
        ) from None


def set_active_parser(name: str) -> None:
    global _active_name
    get_parser(name)  # raises on an unknown name -- fail fast at startup
    _active_name = name


def get_active_parser() -> ToolCallParser:
    return _PARSERS[_active_name]
