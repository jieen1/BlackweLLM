"""Regression unit tests for all historically reported issues.

Run: python -m pytest tests/test_regression_unit.py -v
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.formats import anthropic as anthropic_format
from server.formats import openai as openai_format
from server.formats.content import extract_blocks, extract_text
from server.formats.stream import StreamProcessor
from server.formats.thinking import strip_thinking
from server.formats.tool_parsers import get_parser, set_active_parser
from server.formats.tools import (
    convert_tools_to_chat_template,
    find_tool_call_start,
    parse_tool_calls,
)

QWEN_PARSER = get_parser("qwen3_coder")

THINK_OPEN = chr(60) + "think" + chr(62)
THINK_CLOSE = chr(60) + "/think" + chr(62)
TOOL_OPEN = chr(60) + "tool_call" + chr(62)
TOOL_CLOSE = chr(60) + "/tool_call" + chr(62)
FUNC_OPEN = chr(60) + "function=get_weather" + chr(62)
FUNC_CLOSE = chr(60) + "/function" + chr(62)
PARAM_OPEN = chr(60) + "parameter=location" + chr(62)
PARAM_CLOSE = chr(60) + "/parameter" + chr(62)
FFFD = chr(0xFFFD)
NL = chr(10)


class TestFFFDFiltering:
    def test_removes_fffd(self):
        assert FFFD not in strip_thinking("Hello" + FFFD)

    def test_removes_multiple_fffd(self):
        assert FFFD not in strip_thinking(FFFD + "Hi" + FFFD)

    def test_fffd_in_think_block(self):
        r = strip_thinking(THINK_OPEN + FFFD + "x" + FFFD + THINK_CLOSE + "Ans")
        assert FFFD not in r and "Ans" in r and "x" not in r

    def test_fffd_only_empty(self):
        assert strip_thinking(FFFD * 3) == ""

    def test_normal_unchanged(self):
        assert strip_thinking("Hello!") == "Hello!"


class TestThinkBlockStripping:
    def test_paired(self):
        r = strip_thinking(THINK_OPEN + "reason" + THINK_CLOSE + "Ans")
        assert "reason" not in r and "Ans" in r

    def test_orphan_close_not_anchored_to_start_is_left_untouched(self):
        """R4 fix: a </think> with no matching <think> at position 0 is
        ordinary content, not a signal to delete everything up to it. The
        old behavior here (docs/roadmap.md R4) used an unanchored regex
        that deleted "Thinking Process:\\nreason\\n" -- i.e. real answer
        text -- whenever any </think> appeared anywhere in the response."""
        text = "Thinking Process:" + NL + "reason" + NL + THINK_CLOSE + NL + "Ans"
        assert strip_thinking(text) == text.strip()

    def test_unclosed(self):
        assert strip_thinking(THINK_OPEN + "incomplete") == ""

    def test_multiline(self):
        r = strip_thinking(THINK_OPEN + NL + "L1" + NL + THINK_CLOSE + "Vis")
        assert "L1" not in r and "Vis" in r

    def test_passthrough(self):
        assert strip_thinking("Normal") == "Normal"


class TestOpenAIResponse:
    def test_normal(self):
        r = openai_format.build_response(
            model="t", text="Hi!", finish_reason="stop", prompt_tokens=10, completion_tokens=5
        )
        assert r["choices"][0]["message"]["content"] == "Hi!"

    def test_empty(self):
        r = openai_format.build_response(
            model="t", text="", finish_reason="length", prompt_tokens=10, completion_tokens=100
        )
        assert r["choices"][0]["message"]["content"] == ""

    def test_tool_call(self):
        """build_response has no parser param of its own -- it goes through
        parse_tool_calls' default, i.e. whatever's process-active. Select
        qwen3_coder (this text's shape) for the duration of the test, then
        restore the poolside_v1 default other tests rely on."""
        text = (
            "X "
            + TOOL_OPEN
            + FUNC_OPEN
            + PARAM_OPEN
            + "Paris"
            + PARAM_CLOSE
            + FUNC_CLOSE
            + TOOL_CLOSE
        )
        set_active_parser("qwen3_coder")
        try:
            r = openai_format.build_response(
                model="t", text=text, finish_reason="stop", prompt_tokens=10, completion_tokens=20
            )
        finally:
            set_active_parser("poolside_v1")
        assert r["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert r["choices"][0]["finish_reason"] == "tool_calls"

    def test_usage(self):
        r = openai_format.build_response(
            model="t", text="Hi", finish_reason="stop", prompt_tokens=10, completion_tokens=5
        )
        assert r["usage"]["total_tokens"] == 15


class TestAnthropicResponse:
    def test_required_fields(self):
        r = anthropic_format.build_response(
            model="t", text="Hi", finish_reason="stop", input_tokens=10, output_tokens=5
        )
        for f in [
            "id",
            "type",
            "role",
            "content",
            "model",
            "stop_reason",
            "stop_sequence",
            "usage",
        ]:
            assert f in r, f"missing {f}"

    def test_stop_sequence_always_present(self):
        for reason in ["stop", "length"]:
            r = anthropic_format.build_response(
                model="t", text="Hi", finish_reason=reason, input_tokens=5, output_tokens=3
            )
            assert "stop_sequence" in r and r["stop_sequence"] is None

    def test_empty_text(self):
        r = anthropic_format.build_response(
            model="t", text="", finish_reason="length", input_tokens=10, output_tokens=100
        )
        assert len(r["content"]) >= 1

    def test_stop_reason_mapping(self):
        assert (
            anthropic_format.build_response(
                model="t", text="Hi", finish_reason="stop", input_tokens=1, output_tokens=1
            )["stop_reason"]
            == "end_turn"
        )
        assert (
            anthropic_format.build_response(
                model="t", text="Hi", finish_reason="length", input_tokens=1, output_tokens=1
            )["stop_reason"]
            == "max_tokens"
        )


class TestToolCallParsing:
    def test_single(self):
        text = (
            "X "
            + TOOL_OPEN
            + FUNC_OPEN
            + PARAM_OPEN
            + "Paris"
            + PARAM_CLOSE
            + FUNC_CLOSE
            + TOOL_CLOSE
        )
        vis, tools = parse_tool_calls(text, parser=QWEN_PARSER)
        assert len(tools) == 1 and tools[0]["name"] == "get_weather" and TOOL_OPEN not in vis

    def test_none(self):
        vis, tools = parse_tool_calls("Just text", parser=QWEN_PARSER)
        assert vis == "Just text" and tools == []

    def test_multiple(self):
        tc = TOOL_OPEN + FUNC_OPEN + PARAM_OPEN + "X" + PARAM_CLOSE + FUNC_CLOSE + TOOL_CLOSE
        _, tools = parse_tool_calls("A " + tc + " B " + tc, parser=QWEN_PARSER)
        assert len(tools) == 2


class TestStreamingToolDetection:
    def test_full_tag(self):
        assert find_tool_call_start("Hello " + TOOL_OPEN + "rest") == 6

    def test_partial_prefix(self):
        for i in range(1, len(TOOL_OPEN)):
            assert find_tool_call_start("Hello " + TOOL_OPEN[:i]) == 6

    def test_no_tool(self):
        assert find_tool_call_start("Hello world") == -1


class TestContentExtraction:
    def test_none(self):
        assert extract_text(None) == ""

    def test_string(self):
        assert extract_text("hello") == "hello"

    def test_blocks(self):
        assert (
            extract_text([{"type": "text", "text": "A"}, {"type": "text", "text": "B"}])
            == "A" + NL + "B"
        )

    def test_strings(self):
        assert extract_text(["A", "B"]) == "A" + NL + "B"

    def test_mixed(self):
        assert extract_text([{"type": "text", "text": "Hi"}, {"type": "image"}]) == "Hi"

    def test_extract_blocks_string(self):
        assert extract_blocks("hi") == [{"type": "text", "text": "hi"}]

    def test_extract_blocks_none(self):
        assert extract_blocks(None) == []


class _FakeTok:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if 32 <= i < 127)


class TestStreamProcessor:
    def test_thinking_then_content(self):
        p = StreamProcessor(_FakeTok())
        p.add_tokens([ord(c) for c in "think stuff"])
        assert len(p.drain_thinking()) > 0
        assert p.drain_content() == []
        close_tag = chr(60) + "/think" + chr(62)
        p.add_tokens([ord(c) for c in close_tag + "Answer here"])
        assert any("Answer" in c for c in p.drain_content())

    def test_finalize(self):
        p = StreamProcessor(_FakeTok())
        close_tag = chr(60) + "/think" + chr(62)
        p.add_tokens([ord(c) for c in close_tag + chr(10) + "Hello world"])
        vis, tools = p.finalize()
        assert "Hello world" in vis and tools == []

    def test_non_thinking_backend_splits_generated_think_tags(self):
        """thinking_capable=False (Laguna): no chat-template injection, but
        a <think> the model opens itself (as the very first thing it
        generates) is still a real reasoning span -- split it out, don't
        pass it through verbatim (that was the old, now-retired contract;
        see docs/roadmap.md R3 and tests/test_laguna_server_integration.py
        ::test_laguna_chat_response_splits_generated_think_tags_into_reasoning_content)."""
        p = StreamProcessor(_FakeTok(), thinking_capable=False)
        text = "<think>generated by Laguna</think>answer"
        p.add_tokens([ord(c) for c in text])

        assert p.drain_thinking() == ["generated by Laguna"]
        assert p.drain_content() == ["answer"]
        assert p.finalize() == ("answer", [])
        assert p.reasoning_content() == "generated by Laguna"

    def test_non_thinking_backend_leaves_mid_body_think_literal_untouched(self):
        """R4 regression: <think> not at position 0 is ordinary content."""
        p = StreamProcessor(_FakeTok(), thinking_capable=False)
        text = "Sure, the <think> tag marks reasoning."
        p.add_tokens([ord(c) for c in text])

        assert p.drain_thinking() == []
        assert p.drain_content() == [text]
        assert p.finalize() == (text, [])
        assert p.reasoning_content() is None


class TestAnthropicParse:
    def test_simple(self):
        m = anthropic_format.parse_messages({"messages": [{"role": "user", "content": "Hi"}]})
        assert m[0]["content"] == "Hi"

    def test_system_string(self):
        m = anthropic_format.parse_messages(
            {"system": "Help", "messages": [{"role": "user", "content": "Hi"}]}
        )
        assert m[0]["role"] == "system"

    def test_system_array(self):
        m = anthropic_format.parse_messages(
            {
                "system": [{"type": "text", "text": "A"}],
                "messages": [{"role": "user", "content": "Hi"}],
            }
        )
        assert "A" in m[0]["content"]

    def test_tool_use(self):
        m = anthropic_format.parse_messages(
            {
                "messages": [
                    {"role": "user", "content": "W?"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "tool_use", "id": "t1", "name": "gw", "input": {"l": "P"}}
                        ],
                    },
                ]
            }
        )
        assert [x for x in m if x["role"] == "assistant"][0]["tool_calls"][0]["function"][
            "name"
        ] == "gw"

    def test_tool_result(self):
        m = anthropic_format.parse_messages(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "22C"}],
                    },
                ]
            }
        )
        assert [x for x in m if x["role"] == "tool"][0]["content"] == "22C"


class TestContextCapacity:
    def test_256k_blocks_per_slot(self):
        import inspect

        pytest.importorskip("torch")
        from server.engine import ServerEngine

        sig = inspect.signature(ServerEngine.__init__)
        bps = sig.parameters["blocks_per_slot"].default
        bs = sig.parameters["block_size"].default
        assert bps * bs == 262144, f"{bps}*{bs}={bps * bs} != 262144"

    def test_default_max_tokens_16384(self):
        pytest.importorskip("fastapi")
        from server.app import DEFAULT_MAX_TOKENS

        assert DEFAULT_MAX_TOKENS == 16384


class TestOpenAIParse:
    def test_string(self):
        m = openai_format.parse_chat_messages({"messages": [{"role": "user", "content": "Hi"}]})
        assert m[0]["content"] == "Hi"

    def test_array(self):
        m = openai_format.parse_chat_messages(
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]}
        )
        assert m[0]["content"] == "Hi"

    def test_tool_role(self):
        m = openai_format.parse_chat_messages(
            {"messages": [{"role": "tool", "content": "22C", "tool_call_id": "c1"}]}
        )
        assert m[0]["tool_call_id"] == "c1"

    def test_assistant_tool_calls(self):
        m = openai_format.parse_chat_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "X",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "f", "arguments": json.dumps({"x": 1})},
                            }
                        ],
                    }
                ]
            }
        )
        assert isinstance(m[0]["tool_calls"][0]["function"]["arguments"], dict)

    def test_assistant_reasoning_content_survives_for_qwen_template(self):
        m = openai_format.parse_chat_messages(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "reasoning_content": "Prior reasoning",
                        "content": "Prior answer",
                    }
                ]
            }
        )
        assert m[0]["reasoning_content"] == "Prior reasoning"
        assert m[0]["content"] == "Prior answer"


class TestConvertTools:
    def test_openai_passthrough(self):
        t = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        assert convert_tools_to_chat_template(t) == t

    def test_anthropic_converted(self):
        t = [{"name": "f", "description": "d", "input_schema": {"type": "object"}}]
        assert convert_tools_to_chat_template(t)[0]["type"] == "function"

    def test_none(self):
        assert convert_tools_to_chat_template(None) is None


class TestSSE:
    def test_openai_done(self):
        chunks = list(openai_format.build_sse_chunks("id", "m", "Hi", "stop", 1))
        assert chunks[-1] == "data: [DONE]" + NL + NL

    def test_openai_finish(self):
        chunks = list(openai_format.build_sse_chunks("id", "m", "Hi", "stop", 1))
        assert json.loads(chunks[-2][6:])["choices"][0]["finish_reason"] == "stop"

    def test_anthropic_stop(self):
        evts = list(anthropic_format.build_sse_events("m", "Hi", "stop", 10, 5))
        assert any("message_stop" in e for e in evts)

    def test_anthropic_start(self):
        evts = list(anthropic_format.build_sse_events("m", "Hi", "stop", 10, 5))
        assert evts[0].startswith("event: message_start")


# ============================================================
# <usage> metadata block stripping (model artifact from training data)
# ============================================================


class TestUsageStripping:
    """Regression: model generates <usage>...</usage> blocks that must not
    leak into visible content (reported via Claude Desktop sub-agent output)."""

    def test_paired_usage_block_stripped(self):
        text = (
            "Here is the answer.<usage>subagent_tokens: 53407\n"
            "tool_uses: 146\nduration_ms: 353827</usage>"
        )
        result = strip_thinking(text)
        assert "<usage>" not in result
        assert "subagent_tokens" not in result
        assert "Here is the answer." in result

    def test_usage_block_with_surrounding_whitespace(self):
        text = "Result text.\n\n<usage>\ntokens: 100\n</usage>\n\n"
        result = strip_thinking(text)
        assert "<usage>" not in result
        assert "tokens: 100" not in result
        assert "Result text." in result

    def test_unclosed_usage_block_stripped(self):
        text = "Some output.<usage>subagent_tokens: 999"
        result = strip_thinking(text)
        assert "<usage>" not in result
        assert "subagent_tokens" not in result
        assert "Some output." in result

    def test_usage_after_thinking(self):
        text = (
            f"{THINK_OPEN}reasoning here{THINK_CLOSE}Visible answer."
            f"<usage>duration_ms: 1000</usage>"
        )
        result = strip_thinking(text)
        assert "reasoning" not in result
        assert "<usage>" not in result
        assert "Visible answer." in result

    def test_multiple_usage_blocks(self):
        text = "A.<usage>x: 1</usage> B.<usage>y: 2</usage>"
        result = strip_thinking(text)
        assert "<usage>" not in result
        assert "A." in result
        assert "B." in result

    def test_no_usage_unchanged(self):
        text = "Normal response without any metadata tags."
        result = strip_thinking(text)
        assert result == "Normal response without any metadata tags."

    def test_usage_in_openai_response(self):
        """Ensure <usage> doesn't appear in OpenAI response content."""
        from server.formats import openai as openai_format

        raw = "Answer here.<usage>subagent_tokens: 100</usage>"
        cleaned = strip_thinking(raw)
        resp = openai_format.build_response("test", cleaned, "stop", 10, 5)
        content = resp["choices"][0]["message"]["content"]
        assert "<usage>" not in content
        assert "Answer here." in content

    def test_usage_in_anthropic_response(self):
        """Ensure <usage> doesn't appear in Anthropic response content."""
        from server.formats import anthropic as anthropic_format

        raw = "Answer here.<usage>subagent_tokens: 100</usage>"
        cleaned = strip_thinking(raw)
        resp = anthropic_format.build_response("test", cleaned, "stop", 10, 5)
        text_block = resp["content"][0]
        assert "<usage>" not in text_block["text"]
        assert "Answer here." in text_block["text"]
