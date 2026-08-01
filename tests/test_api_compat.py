#!/usr/bin/env python3
"""Strict integration test for BlackweLLM server API compatibility.

Tests both OpenAI and Anthropic formats for:
1. Normal chat (non-streaming)
2. Normal chat (streaming)
3. Tool calls (non-streaming)
4. Tool calls (streaming)
5. Multi-turn with tool results

Usage: python3 tests/test_api_compat.py [--base-url http://127.0.0.1:8000]
"""

import argparse
import json
import sys
import time

import requests

PASS_COUNT = 0
FAIL_COUNT = 0
RESULTS = []

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def report(name, ok, detail=""):
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    RESULTS.append((name, status, detail))
    msg = f"  [{status}] {name}"
    if detail and not ok:
        msg += " -- " + detail
    print(msg)


def check_no_thinking_leak(text, test_name):
    leaks = []
    if THINK_OPEN in text:
        leaks.append("think-open")
    if THINK_CLOSE in text:
        leaks.append("think-close")
    if leaks:
        report(test_name + ": no thinking leak", False, "found " + str(leaks))
        return False
    report(test_name + ": no thinking leak", True)
    return True


def check_no_tool_xml_leak(text, test_name):
    leaks = []
    if TOOL_CALL_OPEN in text:
        leaks.append("tool_call-open")
    if TOOL_CALL_CLOSE in text:
        leaks.append("tool_call-close")
    if leaks:
        report(test_name + ": no tool XML leak", False, "found " + str(leaks))
        return False
    report(test_name + ": no tool XML leak", True)
    return True


WEATHER_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
    },
}

WEATHER_TOOL_ANTHROPIC = {
    "name": "get_weather",
    "description": "Get the current weather in a given location",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City name"}},
        "required": ["location"],
    },
}


# ===== OpenAI Tests =====


def test_openai_chat(base_url):
    print("")
    print("=== OpenAI: Normal Chat (non-streaming) ===")
    resp = requests.post(
        base_url + "/v1/chat/completions",
        json={
            "model": "qwen3.6-rt",
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 512,
        },
        timeout=120,
    )
    report("openai chat: status 200", resp.status_code == 200, "got " + str(resp.status_code))
    if resp.status_code != 200:
        return
    data = resp.json()
    content = data["choices"][0]["message"]["content"] or ""
    report("openai chat: has content", len(content) > 0)
    check_no_thinking_leak(content, "openai chat")
    check_no_tool_xml_leak(content, "openai chat")
    fr = data["choices"][0]["finish_reason"]
    report("openai chat: finish_reason=stop", fr == "stop", "got " + str(fr))


def test_openai_chat_stream(base_url):
    print("")
    print("=== OpenAI: Normal Chat (streaming) ===")
    resp = requests.post(
        base_url + "/v1/chat/completions",
        json={
            "model": "qwen3.6-rt",
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "max_tokens": 512,
            "stream": True,
        },
        timeout=120,
        stream=True,
    )
    report("openai stream: status 200", resp.status_code == 200, "got " + str(resp.status_code))
    if resp.status_code != 200:
        return
    chunks = []
    full_content = ""
    finish_reason = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            report("openai stream: valid JSON chunk", False, "bad JSON: " + payload[:80])
            return
        chunks.append(chunk)
        delta = chunk["choices"][0].get("delta", {})
        if "content" in delta and delta["content"]:
            full_content += delta["content"]
        fr = chunk["choices"][0].get("finish_reason")
        if fr:
            finish_reason = fr
    report("openai stream: got chunks", len(chunks) > 0, "got " + str(len(chunks)))
    report(
        "openai stream: has content", len(full_content) > 0, "content=" + repr(full_content[:60])
    )
    check_no_thinking_leak(full_content, "openai stream")
    check_no_tool_xml_leak(full_content, "openai stream")
    report(
        "openai stream: finish_reason=stop", finish_reason == "stop", "got " + str(finish_reason)
    )


def test_openai_tool_call(base_url):
    print("")
    print("=== OpenAI: Tool Call (non-streaming) ===")
    resp = requests.post(
        base_url + "/v1/chat/completions",
        json={
            "model": "qwen3.6-rt",
            "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
            "tools": [WEATHER_TOOL_OPENAI],
            "max_tokens": 256,
        },
        timeout=120,
    )
    report("openai tool: status 200", resp.status_code == 200, "got " + str(resp.status_code))
    if resp.status_code != 200:
        return
    data = resp.json()
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls", [])
    check_no_thinking_leak(content, "openai tool")
    check_no_tool_xml_leak(content, "openai tool")
    if tool_calls:
        report("openai tool: has tool_calls", True)
        tc = tool_calls[0]
        report(
            "openai tool: has function name",
            "name" in tc.get("function", {}),
            "tc=" + json.dumps(tc)[:100],
        )
        fr = data["choices"][0]["finish_reason"]
        report("openai tool: finish_reason=tool_calls", fr == "tool_calls", "got " + str(fr))
        args = tc["function"].get("arguments", "{}")
        try:
            if isinstance(args, str):
                json.loads(args)
            report("openai tool: arguments valid JSON", True)
        except (json.JSONDecodeError, TypeError):
            report("openai tool: arguments valid JSON", False, "args=" + str(args)[:80])
    else:
        report(
            "openai tool: has tool_calls", False, "no tool_calls (model may not have called tool)"
        )
        fr = data["choices"][0]["finish_reason"]
        report("openai tool: finish_reason", fr in ("stop", "tool_calls"), "got " + str(fr))


def test_openai_tool_call_stream(base_url):
    print("")
    print("=== OpenAI: Tool Call (streaming) ===")
    resp = requests.post(
        base_url + "/v1/chat/completions",
        json={
            "model": "qwen3.6-rt",
            "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
            "tools": [WEATHER_TOOL_OPENAI],
            "max_tokens": 256,
            "stream": True,
        },
        timeout=120,
        stream=True,
    )
    report(
        "openai tool stream: status 200", resp.status_code == 200, "got " + str(resp.status_code)
    )
    if resp.status_code != 200:
        return
    full_content = ""
    tool_call_chunks = []
    finish_reason = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            report("openai tool stream: valid JSON", False, "bad: " + payload[:80])
            return
        delta = chunk["choices"][0].get("delta", {})
        if "content" in delta and delta["content"]:
            full_content += delta["content"]
        if "tool_calls" in delta:
            tool_call_chunks.extend(delta["tool_calls"])
        fr = chunk["choices"][0].get("finish_reason")
        if fr:
            finish_reason = fr
    check_no_thinking_leak(full_content, "openai tool stream")
    check_no_tool_xml_leak(full_content, "openai tool stream")
    if tool_call_chunks:
        report("openai tool stream: has tool_call chunks", True)
        report(
            "openai tool stream: finish_reason=tool_calls",
            finish_reason == "tool_calls",
            "got " + str(finish_reason),
        )
    else:
        report("openai tool stream: has tool_call chunks", False, "no tool_calls in stream")
        report(
            "openai tool stream: finish_reason",
            finish_reason in ("stop", "tool_calls"),
            "got " + str(finish_reason),
        )


def test_openai_multi_turn(base_url):
    print("")
    print("=== OpenAI: Multi-turn with Tool Result ===")
    messages = [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0000",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"location": "Paris"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0000", "content": "Paris: 22C, sunny"},
        {"role": "user", "content": "Should I bring an umbrella?"},
    ]
    resp = requests.post(
        base_url + "/v1/chat/completions",
        json={
            "model": "qwen3.6-rt",
            "messages": messages,
            "max_tokens": 512,
        },
        timeout=120,
    )
    report("openai multi-turn: status 200", resp.status_code == 200, "got " + str(resp.status_code))
    if resp.status_code != 200:
        return
    data = resp.json()
    content = data["choices"][0]["message"]["content"] or ""
    report("openai multi-turn: has content", len(content) > 0)
    check_no_thinking_leak(content, "openai multi-turn")
    check_no_tool_xml_leak(content, "openai multi-turn")


# ===== Anthropic Tests =====


def test_anthropic_chat(base_url):
    print("")
    print("=== Anthropic: Normal Chat (non-streaming) ===")
    resp = requests.post(
        base_url + "/v1/messages",
        json={
            "model": "qwen3.6-rt",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
        },
        timeout=120,
    )
    report("anthropic chat: status 200", resp.status_code == 200, "got " + str(resp.status_code))
    if resp.status_code != 200:
        return
    data = resp.json()
    content_blocks = data.get("content", [])
    text = " ".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
    report("anthropic chat: has content", len(text) > 0)
    check_no_thinking_leak(text, "anthropic chat")
    check_no_tool_xml_leak(text, "anthropic chat")
    sr = data.get("stop_reason")
    report("anthropic chat: stop_reason=end_turn", sr == "end_turn", "got " + str(sr))


def test_anthropic_chat_stream(base_url):
    print("")
    print("=== Anthropic: Normal Chat (streaming) ===")
    resp = requests.post(
        base_url + "/v1/messages",
        json={
            "model": "qwen3.6-rt",
            "max_tokens": 512,
            "stream": True,
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
        },
        timeout=120,
        stream=True,
    )
    report("anthropic stream: status 200", resp.status_code == 200, "got " + str(resp.status_code))
    if resp.status_code != 200:
        return
    full_text = ""
    events = []
    stop_reason = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("event: "):
            events.append(line[7:])
        elif line.startswith("data: "):
            try:
                d = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if d.get("type") == "content_block_delta":
                delta = d.get("delta", {})
                if delta.get("type") == "text_delta":
                    full_text += delta.get("text", "")
            elif d.get("type") == "message_delta":
                stop_reason = d.get("delta", {}).get("stop_reason")
    report("anthropic stream: got events", len(events) > 0, "got " + str(len(events)))
    report("anthropic stream: has content", len(full_text) > 0, "text=" + repr(full_text[:60]))
    check_no_thinking_leak(full_text, "anthropic stream")
    check_no_tool_xml_leak(full_text, "anthropic stream")
    report(
        "anthropic stream: stop_reason=end_turn",
        stop_reason == "end_turn",
        "got " + str(stop_reason),
    )


def test_anthropic_tool_call(base_url):
    print("")
    print("=== Anthropic: Tool Call (non-streaming) ===")
    resp = requests.post(
        base_url + "/v1/messages",
        json={
            "model": "qwen3.6-rt",
            "max_tokens": 256,
            "tools": [WEATHER_TOOL_ANTHROPIC],
            "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
        },
        timeout=120,
    )
    report("anthropic tool: status 200", resp.status_code == 200, "got " + str(resp.status_code))
    if resp.status_code != 200:
        return
    data = resp.json()
    content_blocks = data.get("content", [])
    text_blocks = [b for b in content_blocks if b.get("type") == "text"]
    tool_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]
    text = " ".join(b.get("text", "") for b in text_blocks)
    check_no_thinking_leak(text, "anthropic tool")
    check_no_tool_xml_leak(text, "anthropic tool")
    if tool_blocks:
        report("anthropic tool: has tool_use blocks", True)
        tb = tool_blocks[0]
        report("anthropic tool: has name", "name" in tb, "block=" + json.dumps(tb)[:100])
        sr = data.get("stop_reason")
        report("anthropic tool: stop_reason=tool_use", sr == "tool_use", "got " + str(sr))
    else:
        report("anthropic tool: has tool_use blocks", False, "no tool_use blocks")


def test_anthropic_tool_call_stream(base_url):
    print("")
    print("=== Anthropic: Tool Call (streaming) ===")
    resp = requests.post(
        base_url + "/v1/messages",
        json={
            "model": "qwen3.6-rt",
            "max_tokens": 256,
            "stream": True,
            "tools": [WEATHER_TOOL_ANTHROPIC],
            "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
        },
        timeout=120,
        stream=True,
    )
    report(
        "anthropic tool stream: status 200", resp.status_code == 200, "got " + str(resp.status_code)
    )
    if resp.status_code != 200:
        return
    full_text = ""
    tool_use_blocks = []
    stop_reason = None
    current_tool = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            try:
                d = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            dtype = d.get("type", "")
            if dtype == "content_block_start":
                cb = d.get("content_block", {})
                if cb.get("type") == "tool_use":
                    current_tool = {"name": cb.get("name"), "input_json": ""}
            elif dtype == "content_block_delta":
                delta = d.get("delta", {})
                if delta.get("type") == "text_delta":
                    full_text += delta.get("text", "")
                elif delta.get("type") == "input_json_delta":
                    if current_tool is not None:
                        current_tool["input_json"] += delta.get("partial_json", "")
            elif dtype == "content_block_stop":
                if current_tool is not None:
                    tool_use_blocks.append(current_tool)
                    current_tool = None
            elif dtype == "message_delta":
                stop_reason = d.get("delta", {}).get("stop_reason")
    check_no_thinking_leak(full_text, "anthropic tool stream")
    check_no_tool_xml_leak(full_text, "anthropic tool stream")
    if tool_use_blocks:
        report("anthropic tool stream: has tool_use blocks", True)
        report(
            "anthropic tool stream: stop_reason=tool_use",
            stop_reason == "tool_use",
            "got " + str(stop_reason),
        )
        tb = tool_use_blocks[0]
        try:
            if tb["input_json"]:
                json.loads(tb["input_json"])
            report("anthropic tool stream: input valid JSON", True)
        except (json.JSONDecodeError, TypeError):
            report(
                "anthropic tool stream: input valid JSON",
                False,
                "raw=" + str(tb["input_json"])[:80],
            )
    else:
        report("anthropic tool stream: has tool_use blocks", False, "no tool_use in stream")


def test_anthropic_multi_turn(base_url):
    print("")
    print("=== Anthropic: Multi-turn with Tool Result ===")
    resp = requests.post(
        base_url + "/v1/messages",
        json={
            "model": "qwen3.6-rt",
            "max_tokens": 512,
            "messages": [
                {"role": "user", "content": "What is the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_0000",
                            "name": "get_weather",
                            "input": {"location": "Paris"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_0000",
                            "content": "Paris: 22C, sunny",
                        }
                    ],
                },
                {"role": "user", "content": "Should I bring an umbrella?"},
            ],
        },
        timeout=120,
    )
    report(
        "anthropic multi-turn: status 200", resp.status_code == 200, "got " + str(resp.status_code)
    )
    if resp.status_code != 200:
        return
    data = resp.json()
    content_blocks = data.get("content", [])
    text = " ".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
    report("anthropic multi-turn: has content", len(text) > 0)
    check_no_thinking_leak(text, "anthropic multi-turn")
    check_no_tool_xml_leak(text, "anthropic multi-turn")


def test_anthropic_array_content(base_url):
    print("")
    print("=== Anthropic: Array Content (Claude Desktop compat) ===")
    resp = requests.post(
        base_url + "/v1/messages",
        json={
            "model": "qwen3.6-rt",
            "max_tokens": 512,
            "system": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Say hi in one word."}]}
            ],
        },
        timeout=120,
    )
    report(
        "anthropic array content: status 200",
        resp.status_code == 200,
        "got " + str(resp.status_code),
    )
    if resp.status_code != 200:
        report("anthropic array content: response", False, resp.text[:200])
        return
    data = resp.json()
    content_blocks = data.get("content", [])
    text = " ".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
    report("anthropic array content: has content", len(text) > 0)
    check_no_thinking_leak(text, "anthropic array content")


def run(base_url: str) -> tuple[int, int, list[tuple[str, str, str]]]:
    """Run every check against ``base_url`` and return (passed, failed,
    results) without exiting the process -- C-LIVE (scripts/c_live_smoke.py)
    calls this directly to reuse this file's dual-protocol x stream/
    non-stream x tool-call coverage instead of re-deriving it.

    PASS_COUNT/FAIL_COUNT/RESULTS are module-level (every ``report()`` call
    from any of the test_* functions above touches them via ``global``) --
    reset them here rather than threading a result object through ten
    functions that predate this refactor. Not safe for concurrent calls in
    the same interpreter; nothing here needs that.
    """
    global PASS_COUNT, FAIL_COUNT, RESULTS
    PASS_COUNT = 0
    FAIL_COUNT = 0
    RESULTS = []
    base = base_url.rstrip("/")

    print("BlackweLLM API Compatibility Test")
    print("Target: " + base)
    print("Time: " + time.strftime("%Y-%m-%d %H:%M:%S"))

    # OpenAI tests
    test_openai_chat(base)
    test_openai_chat_stream(base)
    test_openai_tool_call(base)
    test_openai_tool_call_stream(base)
    test_openai_multi_turn(base)

    # Anthropic tests
    test_anthropic_chat(base)
    test_anthropic_chat_stream(base)
    test_anthropic_tool_call(base)
    test_anthropic_tool_call_stream(base)
    test_anthropic_multi_turn(base)
    test_anthropic_array_content(base)

    print("")
    print("=" * 60)
    total = PASS_COUNT + FAIL_COUNT
    print(f"RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed, {total} total")
    if FAIL_COUNT > 0:
        print("")
        print("Failed tests:")
        for name, status, detail in RESULTS:
            if status == "FAIL":
                print("  - " + name + ": " + detail)
    print("=" * 60)
    return PASS_COUNT, FAIL_COUNT, RESULTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    _passed, failed, _results = run(args.base_url)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
