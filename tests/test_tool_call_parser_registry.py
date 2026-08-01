"""Tests for the tool-call parser registry itself (server/formats/tool_parsers/).

Not the per-shape parsing logic (see tests/test_tool_calls.py) -- this file
covers the registry mechanics: lookup, validation, and active-parser
selection, which is what makes adding a new model's parser a one-line
registration rather than a change to shared code.
"""

import pytest

from server.formats.tool_parsers import (
    ToolCallParser,
    available_parsers,
    get_active_parser,
    get_parser,
    set_active_parser,
)
from server.formats.tool_parsers.poolside_v1 import PoolsideV1ToolCallParser
from server.formats.tool_parsers.qwen3_coder import Qwen3CoderToolCallParser
from server.formats.tool_parsers.registry import _active_name


class TestRegistryLookup:
    def test_known_parsers_registered(self):
        assert set(available_parsers()) == {"poolside_v1", "qwen3_coder"}

    def test_get_parser_returns_correct_type(self):
        assert isinstance(get_parser("poolside_v1"), PoolsideV1ToolCallParser)
        assert isinstance(get_parser("qwen3_coder"), Qwen3CoderToolCallParser)

    def test_get_parser_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown tool_call_parser"):
            get_parser("gpt5_pirate_speak")

    def test_get_parser_unknown_name_lists_available(self):
        with pytest.raises(ValueError, match="poolside_v1"):
            get_parser("nonexistent")

    def test_every_registered_parser_implements_the_interface(self):
        """Guards against a future parser module forgetting to implement
        the abstract interface -- registering it would otherwise only fail
        the first time a request actually exercises the missing method."""
        for name in available_parsers():
            parser = get_parser(name)
            assert isinstance(parser, ToolCallParser)
            assert parser.name == name
            assert parser.open_tag and parser.close_tag


class TestActiveParserSelection:
    def teardown_method(self):
        # Selecting the active parser is process-global state -- restore it
        # so this test file doesn't leak a selection into whichever test
        # runs next (including tests outside this file).
        set_active_parser("poolside_v1")

    def test_default_active_parser_is_poolside_v1(self):
        """Matches this project's currently (and so far only) production
        model, poolside/Laguna-S-2.1-NVFP4."""
        assert _active_name == "poolside_v1"
        assert isinstance(get_active_parser(), PoolsideV1ToolCallParser)

    def test_set_active_parser_switches_it(self):
        set_active_parser("qwen3_coder")
        assert isinstance(get_active_parser(), Qwen3CoderToolCallParser)

    def test_set_active_parser_unknown_name_raises_and_does_not_switch(self):
        set_active_parser("qwen3_coder")
        with pytest.raises(ValueError):
            set_active_parser("gpt5_pirate_speak")
        # A bad --tool-call-parser value must fail fast at startup, not
        # silently fall back to some other model's parser mid-request.
        assert isinstance(get_active_parser(), Qwen3CoderToolCallParser)
