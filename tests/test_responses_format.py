"""Unit tests for the /v1/responses adapter (Codex provider surface)."""

import json

from server.formats import responses as responses_format
from server.formats.tool_parsers import set_active_parser


def test_parse_input_string_instructions_and_input():
    messages = responses_format.parse_input(
        {
            "instructions": "You are a helper.",
            "input": "Hello",
        }
    )
    assert messages == [
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": "Hello"},
    ]


def test_parse_input_items_with_tool_roundtrip():
    body = {
        "instructions": [{"type": "input_text", "text": "System A"}],
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Calculate"},
                    {"type": "input_text", "text": " 2+2"},
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "calculator",
                "arguments": '{"expr": "2+2"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "4",
            },
        ],
    }
    messages = responses_format.parse_input(body)
    assert messages[0] == {"role": "system", "content": "System A"}
    assert messages[1] == {"role": "user", "content": "Calculate\n 2+2"}
    assert messages[2]["role"] == "assistant"
    assert messages[2]["tool_calls"][0]["id"] == "call_1"
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == {"expr": "2+2"}
    assert messages[3] == {
        "role": "tool",
        "content": "4",
        "tool_call_id": "call_1",
    }


def test_developer_role_maps_to_system():
    messages = responses_format.parse_input(
        {
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": "be careful",
                }
            ]
        }
    )
    assert messages == [{"role": "system", "content": "be careful"}]


def test_build_response_text_only():
    resp = responses_format.build_response(
        model="qwen3.6",
        text="hello world",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=3,
        prefix_cache_hit_tokens=7,
    )
    assert resp["object"] == "response"
    assert resp["status"] == "completed"
    assert resp["model"] == "qwen3.6"
    assert len(resp["output"]) == 1
    item = resp["output"][0]
    assert item["type"] == "message"
    assert item["content"][0]["type"] == "output_text"
    assert item["content"][0]["text"] == "hello world"
    assert resp["usage"]["input_tokens"] == 10
    assert resp["usage"]["input_tokens_details"]["cached_tokens"] == 7
    assert resp["usage"]["total_tokens"] == 13
    assert resp["max_output_tokens"] is None
    assert resp["incomplete_details"] is None


def test_build_response_length_is_incomplete():
    resp = responses_format.build_response(
        model="qwen3.8",
        text="partial",
        finish_reason="length",
        prompt_tokens=10,
        completion_tokens=32,
        max_output_tokens=32,
    )

    assert resp["status"] == "incomplete"
    assert resp["incomplete_details"] == {"reason": "max_output_tokens"}
    assert resp["max_output_tokens"] == 32


def test_responses_sse_event_has_ordering_metadata():
    raw = responses_format.sse_event(
        "response.created",
        7,
        {"response": {"id": "resp_x"}},
    )

    assert raw.startswith("event: response.created\n")
    payload = json.loads(raw.split("data: ", 1)[1])
    assert payload == {
        "type": "response.created",
        "sequence_number": 7,
        "response": {"id": "resp_x"},
    }


def test_build_response_with_tool_call():
    set_active_parser("qwen3_coder")
    try:
        text = (
            "I will use the tool.\n"
            "<tool_call><function=calculator><parameter=expr>2+2</parameter>"
            "</function></tool_call>"
        )
        resp = responses_format.build_response(
            model="qwen3.6",
            text=text,
            finish_reason="tool_calls",
            prompt_tokens=5,
            completion_tokens=9,
        )
    finally:
        set_active_parser("poolside_v1")
    assert len(resp["output"]) == 2
    assert resp["output"][0]["type"] == "message"
    fc = resp["output"][1]
    assert fc["type"] == "function_call"
    assert fc["name"] == "calculator"
    assert json.loads(fc["arguments"]) == {"expr": "2+2"}


def test_snapshot_has_required_fields():
    snap = responses_format.snapshot(
        "resp_x",
        123,
        "qwen3.6",
        "in_progress",
        [],
        None,
        max_output_tokens=128,
    )
    assert snap["id"] == "resp_x"
    assert snap["object"] == "response"
    assert snap["status"] == "in_progress"
    assert snap["output"] == []
    assert snap["usage"] is None
    assert snap["max_output_tokens"] == 128
    assert snap["incomplete_details"] is None
    assert snap["error"] is None


def test_terminal_status_marks_stream_errors_failed():
    assert responses_format.terminal_status("error") == ("failed", None)


def test_parse_input_empty_is_falsey():
    assert responses_format.parse_input({}) == []
    assert responses_format.parse_input({"input": []}) == []
