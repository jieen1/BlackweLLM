"""C4: tests for tool call parsing and streaming deltas."""

import json

from server.formats.stream import StreamProcessor
from server.formats.tool_parsers import get_parser
from server.formats.tool_parsers.value_parsing import repair_json as _repair_json
from server.formats.tools import (
    convert_tools_to_chat_template,
    find_tool_call_start,
    format_tool_calls_anthropic,
    format_tool_calls_openai,
    new_tool_call_id,
    parse_tool_calls,
)

QWEN_PARSER = get_parser("qwen3_coder")
POOLSIDE_PARSER = get_parser("poolside_v1")

TC_OPEN = chr(60) + "tool_call" + chr(62)
TC_CLOSE = chr(60) + "/tool_call" + chr(62)
FUNC_OPEN = chr(60) + "function="
FUNC_CLOSE = chr(60) + "/function" + chr(62)
PARAM_OPEN = chr(60) + "parameter="
PARAM_CLOSE = chr(60) + "/parameter" + chr(62)


def _make_tool_call(func_name, params):
    """Build a tool_call XML block from name and param dict."""
    parts = [TC_OPEN, FUNC_OPEN + func_name + chr(62)]
    for key, val in params.items():
        parts.append(PARAM_OPEN + key + chr(62))
        parts.append(val)
        parts.append(PARAM_CLOSE)
    parts.append(FUNC_CLOSE)
    parts.append(TC_CLOSE)
    return "".join(parts)


class TestParseToolCalls:
    """qwen3_coder shape -- parser passed explicitly so this class's
    behavior can't drift if the process's default active parser changes."""

    def test_no_tool_calls(self):
        text = "Hello, world!"
        visible, calls = parse_tool_calls(text, parser=QWEN_PARSER)
        assert visible == "Hello, world!"
        assert calls == []

    def test_single_tool_call(self):
        tc = _make_tool_call("get_weather", {"location": json.dumps("Paris")})
        text = "Sure! " + tc
        visible, calls = parse_tool_calls(text, parser=QWEN_PARSER)
        assert visible == "Sure!"
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"] == {"location": "Paris"}

    def test_multiple_tool_calls(self):
        tc1 = _make_tool_call("foo", {"x": "1"})
        tc2 = _make_tool_call("bar", {"y": json.dumps("hello")})
        text = "Result: " + tc1 + " and " + tc2
        visible, calls = parse_tool_calls(text, parser=QWEN_PARSER)
        assert visible == "Result:  and"
        assert len(calls) == 2
        assert calls[0]["name"] == "foo"
        assert calls[0]["arguments"] == {"x": 1}
        assert calls[1]["name"] == "bar"
        assert calls[1]["arguments"] == {"y": "hello"}

    def test_multiple_params(self):
        tc = _make_tool_call("search", {"query": json.dumps("python"), "limit": "10"})
        _, calls = parse_tool_calls(tc, parser=QWEN_PARSER)
        assert calls[0]["arguments"] == {"query": "python", "limit": 10}

    def test_invalid_json_falls_back_to_string(self):
        tc = _make_tool_call("run", {"code": "not json at all"})
        _, calls = parse_tool_calls(tc, parser=QWEN_PARSER)
        assert calls[0]["arguments"]["code"] == "not json at all"

    def test_json_repair_trailing_comma(self):
        tc = _make_tool_call("fn", {"items": "[1, 2, 3,]"})
        _, calls = parse_tool_calls(tc, parser=QWEN_PARSER)
        assert calls[0]["arguments"]["items"] == [1, 2, 3]

    def test_json_repair_set_literal(self):
        repaired = _repair_json('{("key": "val")}')
        assert json.loads(repaired) == [{"key": "val"}]

    def test_empty_arguments(self):
        tc = _make_tool_call("no_args", {})
        _, calls = parse_tool_calls(tc, parser=QWEN_PARSER)
        assert calls[0]["name"] == "no_args"
        assert calls[0]["arguments"] == {}

    def test_unrecognized_shape_left_as_visible_text(self):
        """A <tool_call> block matching neither known interior shape is
        left untouched -- same as a non-match, not counted as a call."""
        text = "Before " + TC_OPEN + "garbage, no known tags" + TC_CLOSE + " After"
        visible, calls = parse_tool_calls(text, parser=QWEN_PARSER)
        assert calls == []
        assert visible == text.strip()


ARG_KEY_OPEN = chr(60) + "arg_key" + chr(62)
ARG_KEY_CLOSE = chr(60) + "/arg_key" + chr(62)
ARG_VALUE_OPEN = chr(60) + "arg_value" + chr(62)
ARG_VALUE_CLOSE = chr(60) + "/arg_value" + chr(62)


def _make_poolside_tool_call(func_name, params):
    """Build a tool_call block in Laguna's poolside_v1 shape (per its
    chat_template.jinja): bare NAME directly inside <tool_call>, followed
    by <arg_key>/<arg_value> pairs -- no <function=>/<parameter=> wrapper."""
    parts = [TC_OPEN, func_name]
    for key, val in params.items():
        parts.append(ARG_KEY_OPEN + key + ARG_KEY_CLOSE)
        parts.append(ARG_VALUE_OPEN + val + ARG_VALUE_CLOSE)
    parts.append(TC_CLOSE)
    return "".join(parts)


class TestParseToolCallsPoolside:
    """Laguna-S-2.1's poolside_v1 tool-call shape (no <function=>/<parameter=>
    wrapper) -- see chat_template.jinja lines ~64-73. Parser passed
    explicitly for the same reason as TestParseToolCalls above -- it also
    happens to be the process default, but these tests don't rely on that."""

    def test_single_tool_call(self):
        tc = _make_poolside_tool_call("get_weather", {"city": "Paris"})
        text = "Sure! " + tc
        visible, calls = parse_tool_calls(text, parser=POOLSIDE_PARSER)
        assert visible == "Sure!"
        assert len(calls) == 1
        assert calls[0]["name"] == "get_weather"
        assert calls[0]["arguments"] == {"city": "Paris"}

    def test_multiple_args(self):
        tc = _make_poolside_tool_call("search", {"query": "python", "limit": "10"})
        _, calls = parse_tool_calls(tc, parser=POOLSIDE_PARSER)
        assert calls[0]["arguments"] == {"query": "python", "limit": 10}

    def test_multiple_tool_calls(self):
        tc1 = _make_poolside_tool_call("foo", {"x": "1"})
        tc2 = _make_poolside_tool_call("bar", {"y": json.dumps("hello")})
        text = "Result: " + tc1 + " and " + tc2
        visible, calls = parse_tool_calls(text, parser=POOLSIDE_PARSER)
        assert visible == "Result:  and"
        assert len(calls) == 2
        assert calls[0]["name"] == "foo"
        assert calls[0]["arguments"] == {"x": 1}
        assert calls[1]["name"] == "bar"
        assert calls[1]["arguments"] == {"y": "hello"}

    def test_zero_argument_call(self):
        tc = _make_poolside_tool_call("no_args", {})
        _, calls = parse_tool_calls(tc, parser=POOLSIDE_PARSER)
        assert calls[0]["name"] == "no_args"
        assert calls[0]["arguments"] == {}

    def test_invalid_json_falls_back_to_string(self):
        tc = _make_poolside_tool_call("run", {"code": "not json at all"})
        _, calls = parse_tool_calls(tc, parser=POOLSIDE_PARSER)
        assert calls[0]["arguments"]["code"] == "not json at all"

    def test_json_repair_trailing_comma(self):
        tc = _make_poolside_tool_call("fn", {"items": "[1, 2, 3,]"})
        _, calls = parse_tool_calls(tc, parser=POOLSIDE_PARSER)
        assert calls[0]["arguments"]["items"] == [1, 2, 3]

    def test_non_string_value_is_json_encoded_per_template(self):
        """Template: `v | tojson if v is not string else v` -- a bool/number
        argument arrives JSON-encoded, a string argument arrives raw."""
        tc = _make_poolside_tool_call("toggle", {"enabled": "true", "count": "3"})
        _, calls = parse_tool_calls(tc, parser=POOLSIDE_PARSER)
        assert calls[0]["arguments"] == {"enabled": True, "count": 3}


class TestFormatToolCallsOpenAI:
    def test_basic_format(self):
        calls = [{"name": "get_weather", "arguments": {"city": "London"}}]
        result = format_tool_calls_openai(calls)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["function"]["arguments"]) == {"city": "London"}
        assert result[0]["id"].startswith("call_")

    def test_start_id_offset(self):
        calls = [{"name": "fn", "arguments": {}}]
        result = format_tool_calls_openai(calls, start_id=5)
        assert result[0]["id"] == "call_0005"

    def test_multiple_calls_get_unique_default_ids(self):
        calls = [
            {"name": "a", "arguments": {}},
            {"name": "b", "arguments": {}},
        ]
        result = format_tool_calls_openai(calls)
        assert result[0]["id"].startswith("call_")
        assert result[1]["id"].startswith("call_")
        assert result[0]["id"] != result[1]["id"]

    def test_default_ids_are_unique_across_turns(self):
        calls = [{"name": "read", "arguments": {}}]
        first = format_tool_calls_openai(calls)
        second = format_tool_calls_openai(calls)
        assert first[0]["id"] != second[0]["id"]

    def test_new_tool_call_id_is_opaque_and_unique(self):
        assert new_tool_call_id() != new_tool_call_id()


class TestFormatToolCallsAnthropic:
    def test_basic_format(self):
        calls = [{"name": "get_weather", "arguments": {"city": "London"}}]
        result = format_tool_calls_anthropic(calls)
        assert len(result) == 1
        assert result[0]["type"] == "tool_use"
        assert result[0]["name"] == "get_weather"
        assert result[0]["input"] == {"city": "London"}
        assert result[0]["id"].startswith("toolu_")

    def test_unique_ids(self):
        calls = [{"name": "fn", "arguments": {}}]
        r1 = format_tool_calls_anthropic(calls)
        r2 = format_tool_calls_anthropic(calls)
        assert r1[0]["id"] != r2[0]["id"]


class TestConvertToolsToChatTemplate:
    def test_openai_format_passthrough(self):
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        result = convert_tools_to_chat_template(tools)
        assert result == tools

    def test_anthropic_format_conversion(self):
        tools = [{"name": "test", "description": "A test", "input_schema": {"type": "object"}}]
        result = convert_tools_to_chat_template(tools)
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "test"
        assert result[0]["function"]["parameters"] == {"type": "object"}

    def test_server_tools_filtered(self):
        tools = [
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "function", "function": {"name": "real_tool", "parameters": {}}},
        ]
        result = convert_tools_to_chat_template(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "real_tool"

    def test_none_input(self):
        assert convert_tools_to_chat_template(None) is None

    def test_empty_list(self):
        assert convert_tools_to_chat_template([]) is None

    def test_all_server_tools_returns_none(self):
        tools = [{"type": "code_execution_20250101", "name": "exec"}]
        assert convert_tools_to_chat_template(tools) is None


class TestFindToolCallStart:
    def test_full_tag(self):
        text = "Hello " + TC_OPEN + "stuff"
        assert find_tool_call_start(text) == 6

    def test_no_tag(self):
        assert find_tool_call_start("Hello world") == -1

    def test_partial_prefix_at_end(self):
        partial = TC_OPEN[:5]  # e.g. "<tool"
        text = "Hello " + partial
        assert find_tool_call_start(text) == 6

    def test_single_char_prefix(self):
        text = "Hello " + TC_OPEN[0]
        assert find_tool_call_start(text) == 6

    def test_empty_string(self):
        assert find_tool_call_start("") == -1

    def test_tag_at_start(self):
        text = TC_OPEN + "content"
        assert find_tool_call_start(text) == 0


# -- Streaming tool-call delta tests (C4) ------------------------------------


class _FakeTok:
    """Minimal tokenizer stub: maps token IDs to ASCII chars."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if 32 <= i < 127)


def _ids(text):
    return [ord(c) for c in text]


class TestStreamToolDeltas:
    def test_no_tool_call_no_deltas(self):
        proc = StreamProcessor(_FakeTok(), tool_parser=QWEN_PARSER)
        close_think = chr(60) + "/think" + chr(62)
        proc.add_tokens(_ids("thinking" + close_think + "Just text"))
        proc.drain_content()
        assert proc.drain_tool_deltas() == []

    def test_tool_name_delta(self):
        proc = StreamProcessor(_FakeTok(), tool_parser=QWEN_PARSER)
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        func_open = chr(60) + "function=get_weather" + chr(62)
        proc.add_tokens(_ids("think" + close_think + "Sure! " + tc_open + func_open))
        proc.drain_content()
        deltas = proc.drain_tool_deltas()
        name_deltas = [d for d in deltas if d["type"] == "name"]
        assert len(name_deltas) == 1
        assert name_deltas[0]["name"] == "get_weather"
        assert name_deltas[0]["index"] == 0

    def test_tool_name_ids_are_unique_across_streams(self):
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        func_open = chr(60) + "function=read" + chr(62)
        output = _ids("thinking" + close_think + tc_open + func_open)

        first = StreamProcessor(_FakeTok(), tool_parser=QWEN_PARSER)
        second = StreamProcessor(_FakeTok(), tool_parser=QWEN_PARSER)
        first.add_tokens(output)
        second.add_tokens(output)
        first.drain_content()
        second.drain_content()

        first_id = first.drain_tool_deltas()[0]["id"]
        second_id = second.drain_tool_deltas()[0]["id"]
        assert first_id.startswith("call_")
        assert second_id.startswith("call_")
        assert first_id != second_id

    def test_arguments_delta_withheld_until_block_closes(self):
        """No arguments_delta while </parameter></function> hasn't arrived
        yet -- streaming a partial XML-ish fragment isn't valid JSON, so
        there is nothing safe to emit until the whole block is parseable."""
        proc = StreamProcessor(_FakeTok(), tool_parser=QWEN_PARSER)
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        func_open = chr(60) + "function=fn" + chr(62)
        param_open = chr(60) + "parameter=x" + chr(62)
        proc.add_tokens(_ids("t" + close_think + tc_open + func_open + param_open + "1234"))
        proc.drain_content()
        deltas = proc.drain_tool_deltas()
        assert [d for d in deltas if d["type"] == "arguments_delta"] == []

    def test_arguments_delta_emitted_once_as_valid_json_on_close(self):
        """Regression test: arguments_delta must be a JSON string, not the
        raw <parameter=...> XML fragment (that XML, handed verbatim to an
        OpenAI-compatible client as `function.arguments`, fails to
        json-decode -- exactly the reported bug).

        "Block closed" is defined by the shared outer </tool_call> (not
        Qwen's own </function>) -- one definition shared by every parser,
        which real generations satisfy trivially: Qwen's own chat_template
        always emits '</function>\\n</tool_call>' back to back."""
        proc = StreamProcessor(_FakeTok(), tool_parser=QWEN_PARSER)
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        tc_close = chr(60) + "/tool_call" + chr(62)
        func_open = chr(60) + "function=fn" + chr(62)
        param_open = chr(60) + "parameter=x" + chr(62)
        param_close = chr(60) + "/parameter" + chr(62)
        func_close = chr(60) + "/function" + chr(62)
        proc.add_tokens(
            _ids("t" + close_think + tc_open + func_open + param_open + "1234" + param_close)
        )
        proc.drain_content()
        deltas = proc.drain_tool_deltas()  # "name" fires here; args must not
        assert [d for d in deltas if d["type"] == "arguments_delta"] == []
        proc.add_tokens(_ids(func_close))
        deltas = proc.drain_tool_deltas()  # </function> alone still isn't enough
        assert [d for d in deltas if d["type"] == "arguments_delta"] == []
        proc.add_tokens(_ids(tc_close))
        deltas = proc.drain_tool_deltas()
        arg_deltas = [d for d in deltas if d["type"] == "arguments_delta"]
        assert len(arg_deltas) == 1
        assert json.loads(arg_deltas[0]["delta"]) == {"x": 1234}
        # Emitted exactly once -- a later drain for the same call is a no-op.
        proc.add_tokens(_ids(""))
        assert proc.drain_tool_deltas() == []

    def test_finalize_after_tool_deltas(self):
        proc = StreamProcessor(_FakeTok(), tool_parser=QWEN_PARSER)
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        tc_close = chr(60) + "/tool_call" + chr(62)
        func_open = chr(60) + "function=add" + chr(62)
        func_close = chr(60) + "/function" + chr(62)
        param_open = chr(60) + "parameter=x" + chr(62)
        param_close = chr(60) + "/parameter" + chr(62)
        full = (
            "think"
            + close_think
            + "Result: "
            + tc_open
            + func_open
            + param_open
            + "42"
            + param_close
            + func_close
            + tc_close
        )
        proc.add_tokens(_ids(full))
        proc.drain_content()
        proc.drain_tool_deltas()
        visible, calls = proc.finalize()
        assert visible == "Result:"
        assert len(calls) == 1
        assert calls[0]["name"] == "add"
        assert calls[0]["arguments"] == {"x": 42}

    def test_content_frozen_after_tool_start(self):
        proc = StreamProcessor(_FakeTok(), tool_parser=QWEN_PARSER)
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        proc.add_tokens(_ids("think" + close_think + "Hello " + tc_open))
        content = proc.drain_content()
        assert any("Hello" in c for c in content)
        # After tool call starts, content should be frozen
        proc.add_tokens(_ids("more text"))
        assert proc.drain_content() == []


class TestStreamToolDeltasPoolside:
    """Real-time (streaming) tool-call deltas for Laguna's poolside_v1 shape
    -- bare NAME<arg_key>K</arg_key><arg_value>V</arg_value>..., no
    <function=>/<parameter=> wrapper. Mirrors TestStreamToolDeltas above."""

    def test_tool_name_delta_zero_arg(self):
        """Name boundary for a zero-argument call is only known once
        </tool_call> itself arrives -- not emitted before that."""
        proc = StreamProcessor(_FakeTok(), tool_parser=POOLSIDE_PARSER)
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        proc.add_tokens(_ids("think" + close_think + "Sure! " + tc_open + "get_time"))
        proc.drain_content()
        assert proc.drain_tool_deltas() == []  # no </tool_call> or <arg_key> yet
        tc_close = chr(60) + "/tool_call" + chr(62)
        proc.add_tokens(_ids(tc_close))
        deltas = proc.drain_tool_deltas()
        name_deltas = [d for d in deltas if d["type"] == "name"]
        assert len(name_deltas) == 1
        assert name_deltas[0]["name"] == "get_time"
        assert name_deltas[0]["index"] == 0

    def test_tool_name_delta_with_args(self):
        proc = StreamProcessor(_FakeTok(), tool_parser=POOLSIDE_PARSER)
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        arg_key_open = chr(60) + "arg_key" + chr(62)
        proc.add_tokens(_ids("think" + close_think + "Sure! " + tc_open + "get_weather"))
        proc.drain_content()
        assert proc.drain_tool_deltas() == []  # name boundary not known yet
        proc.add_tokens(_ids(arg_key_open))
        deltas = proc.drain_tool_deltas()
        name_deltas = [d for d in deltas if d["type"] == "name"]
        assert len(name_deltas) == 1
        assert name_deltas[0]["name"] == "get_weather"

    def test_arguments_delta_withheld_until_block_closes(self):
        """No arguments_delta while </tool_call> hasn't arrived yet --
        streaming the raw <arg_key>/<arg_value> fragment isn't valid JSON."""
        proc = StreamProcessor(_FakeTok(), tool_parser=POOLSIDE_PARSER)
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        arg_key_open = chr(60) + "arg_key" + chr(62)
        arg_key_close = chr(60) + "/arg_key" + chr(62)
        arg_value_open = chr(60) + "arg_value" + chr(62)
        proc.add_tokens(
            _ids(
                "t"
                + close_think
                + tc_open
                + "fn"
                + arg_key_open
                + "x"
                + arg_key_close
                + arg_value_open
                + "1234"
            )
        )
        proc.drain_content()
        deltas = proc.drain_tool_deltas()
        assert [d for d in deltas if d["type"] == "arguments_delta"] == []

    def test_arguments_delta_emitted_once_as_valid_json_on_close(self):
        """Regression test for the reported bug: an OpenAI-compatible client
        (Go's encoding/json, per the exact error text 'invalid character
        <lt> looking for beginning of value') tried to json-decode
        `function.arguments` and got the raw
        `<arg_key>path</arg_key><arg_value>.</arg_value>` XML instead of
        JSON. arguments_delta must carry valid, parseable JSON."""
        proc = StreamProcessor(_FakeTok(), tool_parser=POOLSIDE_PARSER)
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        tc_close = chr(60) + "/tool_call" + chr(62)
        arg_key_open = chr(60) + "arg_key" + chr(62)
        arg_key_close = chr(60) + "/arg_key" + chr(62)
        arg_value_open = chr(60) + "arg_value" + chr(62)
        arg_value_close = chr(60) + "/arg_value" + chr(62)
        proc.add_tokens(
            _ids(
                "t"
                + close_think
                + tc_open
                + "ls"
                + arg_key_open
                + "path"
                + arg_key_close
                + arg_value_open
                + "."
                + arg_value_close
            )
        )
        proc.drain_content()
        deltas = proc.drain_tool_deltas()  # "name" fires here; args must not
        assert [d for d in deltas if d["type"] == "arguments_delta"] == []
        proc.add_tokens(_ids(tc_close))
        deltas = proc.drain_tool_deltas()
        arg_deltas = [d for d in deltas if d["type"] == "arguments_delta"]
        assert len(arg_deltas) == 1
        assert "<arg_key>" not in arg_deltas[0]["delta"]
        assert json.loads(arg_deltas[0]["delta"]) == {"path": "."}
        # Emitted exactly once -- a later drain for the same call is a no-op.
        proc.add_tokens(_ids(""))
        assert proc.drain_tool_deltas() == []

    def test_finalize_after_tool_deltas(self):
        proc = StreamProcessor(_FakeTok(), tool_parser=POOLSIDE_PARSER)
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        tc_close = chr(60) + "/tool_call" + chr(62)
        arg_key_open = chr(60) + "arg_key" + chr(62)
        arg_key_close = chr(60) + "/arg_key" + chr(62)
        arg_value_open = chr(60) + "arg_value" + chr(62)
        arg_value_close = chr(60) + "/arg_value" + chr(62)
        full = (
            "think"
            + close_think
            + "Result: "
            + tc_open
            + "add"
            + arg_key_open
            + "x"
            + arg_key_close
            + arg_value_open
            + "42"
            + arg_value_close
            + tc_close
        )
        proc.add_tokens(_ids(full))
        proc.drain_content()
        proc.drain_tool_deltas()
        visible, calls = proc.finalize()
        assert visible == "Result:"
        assert len(calls) == 1
        assert calls[0]["name"] == "add"
        assert calls[0]["arguments"] == {"x": 42}

    def test_multiple_tool_calls_sequential(self):
        proc = StreamProcessor(_FakeTok(), tool_parser=POOLSIDE_PARSER)
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        tc_close = chr(60) + "/tool_call" + chr(62)
        full = "think" + close_think + tc_open + "foo" + tc_close + tc_open + "bar" + tc_close
        proc.add_tokens(_ids(full))
        proc.drain_content()
        deltas = proc.drain_tool_deltas()
        names = [d for d in deltas if d["type"] == "name"]
        assert [n["name"] for n in names] == ["foo", "bar"]
        assert [n["index"] for n in names] == [0, 1]

    def test_multi_arg_call_and_second_call_both_get_valid_json(self):
        """Multi-key-value regression: every key in a multi-arg call, and
        every call in a multi-call round, must independently concatenate to
        valid JSON -- not just the single-arg case above."""
        proc = StreamProcessor(_FakeTok(), tool_parser=POOLSIDE_PARSER)
        close_think = chr(60) + "/think" + chr(62)
        tc_open = chr(60) + "tool_call" + chr(62)
        tc_close = chr(60) + "/tool_call" + chr(62)
        ako, akc = chr(60) + "arg_key" + chr(62), chr(60) + "/arg_key" + chr(62)
        avo, avc = chr(60) + "arg_value" + chr(62), chr(60) + "/arg_value" + chr(62)

        def pair(k, v):
            return ako + k + akc + avo + v + avc

        call1 = tc_open + "search" + pair("query", "python") + pair("limit", "10") + tc_close
        call2 = tc_open + "get_time" + tc_close
        proc.add_tokens(_ids("think" + close_think + call1 + call2))
        proc.drain_content()
        deltas = proc.drain_tool_deltas()
        arg_deltas = {d["index"]: d["delta"] for d in deltas if d["type"] == "arguments_delta"}
        assert set(arg_deltas) == {0, 1}
        assert json.loads(arg_deltas[0]) == {"query": "python", "limit": 10}
        assert json.loads(arg_deltas[1]) == {}
