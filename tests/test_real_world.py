"""Real-world integration tests for BlackweLLM server.

Covers the actual scenarios that Claude Desktop / OpenAI clients hit:
- Anthropic API format compliance (all required fields)
- Event loop non-blocking (health during generation AND tokenization)
- Large prompt handling (10K+ tokens, apply_chat_template offloaded)
- Streaming TTFT and incremental delivery
- Tool-call loop termination
- Concurrent request handling
- HEAD / connectivity check
"""

import http.client
import json
import sys
import threading
import time

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
HOST = BASE.replace("http://", "").split(":")[0]
PORT = int(BASE.replace("http://", "").split(":")[1]) if ":" in BASE.replace("http://", "") else 80

passed = 0
failed = 0


def check(label, ok, detail=""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label} {detail}")
    return ok


def post(path, body, headers=None, timeout=120):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    conn.request("POST", path, json.dumps(body), hdrs)
    resp = conn.getresponse()
    data = resp.read().decode()
    conn.close()
    return resp.status, data


def get(path, timeout=5):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    conn.request("GET", path)
    resp = conn.getresponse()
    data = resp.read().decode()
    conn.close()
    return resp.status, data


def head(path, timeout=5):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    conn.request("HEAD", path)
    resp = conn.getresponse()
    resp.read()
    conn.close()
    return resp.status


def stream_post(path, body, headers=None, timeout=120):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    body["stream"] = True
    t0 = time.perf_counter()
    conn.request("POST", path, json.dumps(body), hdrs)
    resp = conn.getresponse()
    events = []
    first_event_t = None
    for raw in resp:
        line = raw.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            if first_event_t is None:
                first_event_t = time.perf_counter()
            try:
                events.append(json.loads(line[6:]))
            except Exception:
                pass
    t1 = time.perf_counter()
    conn.close()
    ttft = (first_event_t - t0) * 1000 if first_event_t else (t1 - t0) * 1000
    return events, ttft, (t1 - t0) * 1000


ANTHROPIC_HDRS = {"x-api-key": "test", "anthropic-version": "2023-06-01"}

# ============================================================
print("=== 1. HEAD / connectivity check ===")
# ============================================================
status = head("/")
check("HEAD / returns 200", status == 200, f"got {status}")

# ============================================================
print("\n=== 2. Anthropic non-streaming format compliance ===")
# ============================================================
status, raw = post(
    "/v1/messages",
    {
        "model": "qwen3.6-rt",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "Say hello"}],
    },
    ANTHROPIC_HDRS,
)
r = json.loads(raw)
check("status 200", status == 200, f"got {status}")
for field in ["id", "type", "role", "content", "model", "stop_reason", "stop_sequence", "usage"]:
    check(f"field '{field}' present", field in r, f"missing in {list(r.keys())}")
check("type is 'message'", r.get("type") == "message")
check("role is 'assistant'", r.get("role") == "assistant")
check("content is list", isinstance(r.get("content"), list))
check("usage has input_tokens", "input_tokens" in r.get("usage", {}))
check("usage has output_tokens", "output_tokens" in r.get("usage", {}))

# ============================================================
print("\n=== 3. Anthropic array system + array content ===")
# ============================================================
status, raw = post(
    "/v1/messages",
    {
        "model": "qwen3.6-rt",
        "max_tokens": 128,
        "system": [
            {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Be concise."},
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
    },
    ANTHROPIC_HDRS,
)
check("array system+content: 200", status == 200, f"got {status}: {raw[:200]}")
if status == 200:
    r = json.loads(raw)
    check("array: has content", len(r.get("content", [])) > 0)
    check("array: stop_sequence present", "stop_sequence" in r)

# ============================================================
print("\n=== 4. Anthropic streaming format ===")
# ============================================================
events, ttft, total = stream_post(
    "/v1/messages?beta=true",
    {
        "model": "qwen3.6-rt",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Count 1 to 5"}],
    },
    ANTHROPIC_HDRS,
)
types = [e.get("type") for e in events]
check("stream: has message_start", "message_start" in types)
check("stream: has content_block_start", "content_block_start" in types)
check("stream: has content_block_delta", "content_block_delta" in types)
check("stream: has content_block_stop", "content_block_stop" in types)
check("stream: has message_delta", "message_delta" in types)
check("stream: has message_stop", "message_stop" in types)
# Check message_start format
ms = next((e for e in events if e.get("type") == "message_start"), {})
msg = ms.get("message", {})
check("stream: message_start has stop_sequence", "stop_sequence" in msg)
# Check message_delta format
md = next((e for e in events if e.get("type") == "message_delta"), {})
check("stream: message_delta has stop_reason", "stop_reason" in md.get("delta", {}))
check("stream: message_delta has stop_sequence", "stop_sequence" in md.get("delta", {}))
print(f"  TTFT={ttft:.0f}ms total={total:.0f}ms events={len(events)}")

# ============================================================
print("\n=== 5. Tool call loop terminates ===")
# ============================================================
tools = [
    {
        "name": "get_time",
        "description": "Get current time",
        "input_schema": {
            "type": "object",
            "properties": {"tz": {"type": "string"}},
            "required": [],
        },
    }
]
msgs = [{"role": "user", "content": [{"type": "text", "text": "What time is it?"}]}]
loop_ok = False
for turn in range(5):
    status, raw = post(
        "/v1/messages",
        {"model": "qwen3.6-rt", "max_tokens": 4096, "tools": tools, "messages": msgs},
        ANTHROPIC_HDRS,
    )
    r = json.loads(raw)
    stop = r.get("stop_reason")
    if stop != "tool_use":
        loop_ok = True
        check(f"tool loop ends at turn {turn + 1} (stop={stop})", True)
        break
    msgs.append({"role": "assistant", "content": r["content"]})
    results = []
    for b in r["content"]:
        if b.get("type") == "tool_use":
            results.append({"type": "tool_result", "tool_use_id": b["id"], "content": "12:00 PM"})
    msgs.append({"role": "user", "content": results})
if not loop_ok:
    check("tool loop terminates within 5 turns", False, "STILL LOOPING")

# ============================================================
print("\n=== 6. Event loop non-blocking: health during generation ===")
# ============================================================
# Start a long generation in background
gen_done = threading.Event()


def long_gen():
    post(
        "/v1/chat/completions",
        {
            "model": "qwen3.6-rt",
            "messages": [{"role": "user", "content": "Write a very long essay about history"}],
            "max_tokens": 2048,
        },
        timeout=120,
    )
    gen_done.set()


t = threading.Thread(target=long_gen, daemon=True)
t.start()
time.sleep(2)

latencies = []
for _ in range(5):
    t0 = time.perf_counter()
    try:
        status, _ = get("/health", timeout=5)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
    except Exception:
        latencies.append(5000)
    time.sleep(0.3)

avg_lat = sum(latencies) / len(latencies)
max_lat = max(latencies)
check(f"health during gen: avg={avg_lat:.0f}ms < 1000ms", avg_lat < 1000, f"avg={avg_lat:.0f}ms")
check(f"health during gen: max={max_lat:.0f}ms < 2000ms", max_lat < 2000, f"max={max_lat:.0f}ms")
gen_done.wait(timeout=60)

# ============================================================
print("\n=== 7. Large prompt: event loop not blocked by tokenization ===")
# ============================================================
# Build a ~5K token system prompt
big_system = "You are a helpful assistant. " * 500  # ~3500 tokens
big_msgs = [{"role": "user", "content": "Hello"}]
# Add conversation history
for i in range(10):
    big_msgs.append({"role": "assistant", "content": f"Response {i} " * 50})
    big_msgs.append({"role": "user", "content": f"Follow up {i} " * 50})

# Start large tokenization in background
tokenize_done = threading.Event()


def big_request():
    post(
        "/v1/messages",
        {
            "model": "qwen3.6-rt",
            "max_tokens": 64,
            "system": [{"type": "text", "text": big_system}],
            "messages": big_msgs,
        },
        ANTHROPIC_HDRS,
        timeout=120,
    )
    tokenize_done.set()


t2 = threading.Thread(target=big_request, daemon=True)
t2.start()
time.sleep(0.5)  # let tokenization start

# Health check should still be fast during tokenization
t0 = time.perf_counter()
try:
    status, _ = get("/health", timeout=10)
    health_ms = (time.perf_counter() - t0) * 1000
except Exception:
    health_ms = 10000
check(
    f"health during large tokenize: {health_ms:.0f}ms < 200ms",
    health_ms < 200,
    f"got {health_ms:.0f}ms",
)
tokenize_done.wait(timeout=120)

# ============================================================
print("\n=== 8. OpenAI format compliance ===")
# ============================================================
status, raw = post(
    "/v1/chat/completions",
    {"model": "qwen3.6-rt", "max_tokens": 64, "messages": [{"role": "user", "content": "Hi"}]},
)
r = json.loads(raw)
check("openai: status 200", status == 200)
check("openai: has choices", "choices" in r)
check("openai: has usage", "usage" in r)
check("openai: choice has message", "message" in r.get("choices", [{}])[0])
check("openai: choice has finish_reason", "finish_reason" in r.get("choices", [{}])[0])

# ============================================================
print("\n=== 9. OpenAI streaming ===")
# ============================================================
events, ttft, total = stream_post(
    "/v1/chat/completions",
    {
        "model": "qwen3.6-rt",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Say hello"}],
    },
)
check("openai stream: has chunks", len(events) > 0)
check("openai stream: TTFT < 5000ms", ttft < 5000, f"TTFT={ttft:.0f}ms")
has_content = any(e.get("choices", [{}])[0].get("delta", {}).get("content") for e in events)
has_finish = any(e.get("choices", [{}])[0].get("finish_reason") for e in events)
check("openai stream: has content deltas", has_content)
check("openai stream: has finish_reason", has_finish)
print(f"  TTFT={ttft:.0f}ms total={total:.0f}ms chunks={len(events)}")

# ============================================================
print("\n=== 10. Concurrent requests (3 parallel) ===")
# ============================================================
results = [None, None, None]


def conc(idx):
    s, raw = post(
        "/v1/chat/completions",
        {
            "model": "qwen3.6-rt",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": f"Hello #{idx}"}],
        },
        timeout=120,
    )
    results[idx] = (s, json.loads(raw))


t0 = time.perf_counter()
threads = [threading.Thread(target=conc, args=(i,)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join(timeout=120)
conc_ms = (time.perf_counter() - t0) * 1000
all_ok = all(r is not None and r[0] == 200 for r in results)
check("3 concurrent: all 200", all_ok)
check(f"3 concurrent: completed in {conc_ms:.0f}ms < 60s", conc_ms < 60000, f"{conc_ms:.0f}ms")

# ============================================================
print(f"\n{'=' * 60}")
print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print(f"{'=' * 60}")
sys.exit(1 if failed else 0)
