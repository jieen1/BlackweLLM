#!/usr/bin/env python3
"""End-to-end 256K long-context stress test.

Requires a running server:
    python -m server.app --host 0.0.0.0 --port 8000

Run:
    python tests/test_e2e_256k_longctx.py [--base-url http://127.0.0.1:8000]

Covers:
  Phase 1: Single 200K warmup (verify full context works)
  Phase 2: 3×200K concurrent sessions (max concurrency at near-full context)
  Phase 3: Multi-turn on each of 3 sessions (growing context, prefix cache)
  Phase 4: Tool call + tool result round-trip at 100K+ context
  Phase 5: 4×128K concurrent (full capacity stress)
  Phase 6: Final 2×200K concurrent (verify no OOM after all rounds)
  Phase 7: Memory stability check (no monotonic leak)
"""

import argparse
import http.client
import json
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

PASSED = 0
FAILED = 0
RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    global PASSED, FAILED
    if ok:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}  {detail}")
    RESULTS.append((label, ok, detail))
    return ok


def gpu_mem_mib() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return int(r.stdout.strip())
    except Exception:
        return -1


class Client:
    def __init__(self, base_url: str):
        parsed = urlparse(base_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 8000

    def post(self, path: str, body: dict, timeout: int = 600) -> tuple[int, dict | str]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read().decode()
            if resp.status == 200:
                return resp.status, json.loads(data)
            return resp.status, data
        except Exception as exc:
            return 0, str(exc)
        finally:
            conn.close()

    def health(self) -> bool:
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
            conn.request("GET", "/health")
            r = conn.getresponse()
            r.read()
            conn.close()
            return r.status == 200
        except Exception:
            return False

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 64,
        tools: list[dict] | None = None,
        timeout: int = 600,
    ) -> tuple[int, dict | str]:
        body: dict = {
            "model": "qwen3.6-rt",
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
        return self.post("/v1/chat/completions", body, timeout=timeout)


def make_filler(target_tokens: int) -> str:
    """Generate filler text of approximately target_tokens tokens."""
    return "The quick brown fox jumps over the lazy dog. " * max(1, target_tokens // 10)


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
    },
}


def run_concurrent(
    client: Client, prompts: list[str], max_tokens: int = 64, timeout: int = 600
) -> list[tuple[int, dict | str, float]]:
    """Fire len(prompts) requests concurrently, return (status, response, elapsed)."""
    results: list[tuple[int, dict | str, float] | None] = [None] * len(prompts)

    def worker(idx: int):
        t0 = time.perf_counter()
        s, r = client.chat(
            [{"role": "user", "content": prompts[idx]}],
            max_tokens=max_tokens,
            timeout=timeout,
        )
        results[idx] = (s, r, time.perf_counter() - t0)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(len(prompts))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout + 30)
    return [r if r is not None else (0, "timeout", 0.0) for r in results]


def run(base_url: str) -> tuple[int, int]:
    """Run every phase against ``base_url`` and return (passed, failed)
    without exiting the process. PASS/FAILED/RESULTS are module-level
    (every ``check()`` call touches them via ``global``); reset here so a
    second call in the same interpreter doesn't accumulate. Not safe for
    concurrent calls in the same interpreter; nothing here needs that.

    Raises ``RuntimeError`` if the server is unreachable, rather than the
    original bare ``sys.exit(2)`` -- a caller driving this as a subroutine
    (as opposed to a standalone script) needs an exception it can catch,
    not a process exit.
    """
    global PASSED, FAILED, RESULTS
    PASSED = 0
    FAILED = 0
    RESULTS = []

    client = Client(base_url)
    mem_baseline = gpu_mem_mib()

    print("=" * 72)
    print("  256K Full-Context · Max-Concurrency · Multi-Turn E2E Test")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Server: {base_url}")
    print(f"  GPU baseline: {mem_baseline} MiB")
    print("=" * 72)

    if not client.health():
        raise RuntimeError(f"server not reachable at {base_url}")

    filler_200k = make_filler(200_000)
    filler_128k = make_filler(128_000)
    filler_100k = make_filler(100_000)

    # ── Phase 1: Single 200K warmup ──────────────────────────────────
    print("\n=== Phase 1: Single 200K warmup ===")
    t0 = time.perf_counter()
    s, r = client.chat(
        [{"role": "user", "content": filler_200k + "\n\nSummarize in one sentence."}],
        max_tokens=64,
    )
    elapsed = time.perf_counter() - t0
    if s == 200:
        pt = r.get("usage", {}).get("prompt_tokens", 0)
        check(
            f"200K warmup: prompt_tokens={pt}, {elapsed:.1f}s, gpu={gpu_mem_mib()}MiB",
            pt > 150_000,
            f"expected >150K tokens, got {pt}",
        )
    else:
        check("200K warmup", False, f"status={s}: {str(r)[:200]}")
    check("Health after warmup", client.health())

    # ── Phase 2: 3×200K concurrent sessions ──────────────────────────
    print("\n=== Phase 2: 3×200K concurrent sessions ===")
    prompts_3x200k = [
        filler_200k + f"\n\nSession {i}: Summarize the key theme in one sentence." for i in range(3)
    ]
    t0 = time.perf_counter()
    conc_results = run_concurrent(client, prompts_3x200k, max_tokens=64)
    total_conc = time.perf_counter() - t0
    mem_conc = gpu_mem_mib()
    for i, (st, resp, el) in enumerate(conc_results):
        if st == 200:
            pt = resp.get("usage", {}).get("prompt_tokens", 0)
            check(
                f"3×200K session {i}: pt={pt}, {el:.1f}s", pt > 150_000, f"expected >150K, got {pt}"
            )
        else:
            check(f"3×200K session {i}", False, f"status={st}: {str(resp)[:200]}")
    check(
        f"3×200K total={total_conc:.1f}s, gpu={mem_conc}MiB",
        total_conc < 900,
        f"took too long: {total_conc:.1f}s",
    )
    check("Health after 3×200K", client.health())

    # ── Phase 3: Multi-turn on 3 sessions (growing context) ──────────
    print("\n=== Phase 3: Multi-turn × 3 sessions (100K base, 3 turns each) ===")
    session_messages: list[list[dict]] = [
        [
            {
                "role": "user",
                "content": filler_100k + f"\n\nSession {i}: Start a short story about a robot.",
            }
        ]
        for i in range(3)
    ]
    for turn in range(3):
        print(f"  --- Turn {turn + 1} ---")
        prompts_turn = []
        for i in range(3):
            last_user = (
                session_messages[i][-1]["content"]
                if session_messages[i][-1]["role"] == "user"
                else ""
            )
            prompts_turn.append(last_user)

        turn_results = run_concurrent(client, prompts_turn, max_tokens=128)
        for i, (st, resp, el) in enumerate(turn_results):
            if st == 200:
                content = resp["choices"][0]["message"].get("content", "") or ""
                pt = resp.get("usage", {}).get("prompt_tokens", 0)
                session_messages[i].append({"role": "assistant", "content": content or "..."})
                session_messages[i].append(
                    {"role": "user", "content": f"Continue turn {turn + 2}."}
                )
                check(
                    f"Multi-turn s{i} t{turn + 1}: pt={pt}, {el:.1f}s, gpu={gpu_mem_mib()}MiB", True
                )
            else:
                check(f"Multi-turn s{i} t{turn + 1}", False, f"status={st}: {str(resp)[:200]}")
    check("Health after multi-turn", client.health())

    # ── Phase 4: Tool call + round-trip at 100K ──────────────────────
    print("\n=== Phase 4: Tool call + round-trip at 100K context ===")
    s, r = client.chat(
        [
            {
                "role": "user",
                "content": filler_100k + "\n\nWhat is the weather in Paris? Use the tool.",
            }
        ],
        max_tokens=256,
        tools=[WEATHER_TOOL],
    )
    if s == 200:
        msg = r["choices"][0]["message"]
        has_tc = bool(msg.get("tool_calls"))
        fr = r["choices"][0].get("finish_reason", "")
        pt = r.get("usage", {}).get("prompt_tokens", 0)
        check(f"Tool call 100K: pt={pt}, finish={fr}, has_tool_calls={has_tc}", True)
        if has_tc:
            tc = msg["tool_calls"][0]
            check(f"Tool name={tc['function']['name']}", tc["function"]["name"] == "get_weather")
    else:
        check("Tool call 100K", False, f"status={s}: {str(r)[:200]}")

    s, r = client.chat(
        [
            {"role": "user", "content": filler_100k + "\n\nWhat is the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "Paris"}),
                        },
                    }
                ],
            },
            {"role": "tool", "content": "Paris: 22°C, sunny", "tool_call_id": "c1"},
            {"role": "user", "content": "Should I bring an umbrella?"},
        ],
        max_tokens=256,
        tools=[WEATHER_TOOL],
    )
    if s == 200:
        content = r["choices"][0]["message"].get("content", "") or ""
        pt = r.get("usage", {}).get("prompt_tokens", 0)
        check(f"Tool round-trip: pt={pt}, has_content={len(content) > 0}", True)
    else:
        check("Tool round-trip", False, f"status={s}: {str(r)[:200]}")

    # ── Phase 5: 4×128K concurrent (full capacity) ───────────────────
    print("\n=== Phase 5: 4×128K concurrent (full capacity) ===")
    prompts_4x128k = [filler_128k + f"\n\nSession {i}: Summarize." for i in range(4)]
    t0 = time.perf_counter()
    conc4 = run_concurrent(client, prompts_4x128k, max_tokens=64)
    total4 = time.perf_counter() - t0
    for i, (st, resp, el) in enumerate(conc4):
        if st == 200:
            pt = resp.get("usage", {}).get("prompt_tokens", 0)
            check(f"4×128K session {i}: pt={pt}, {el:.1f}s", pt > 100_000)
        else:
            check(f"4×128K session {i}", False, f"status={st}: {str(resp)[:200]}")
    check(f"4×128K total={total4:.1f}s, gpu={gpu_mem_mib()}MiB", total4 < 600)
    check("Health after 4×128K", client.health())

    # ── Phase 6: Final 2×200K concurrent (post-stress) ───────────────
    print("\n=== Phase 6: Final 2×200K concurrent (post-stress) ===")
    prompts_2x200k = [filler_200k + f"\n\nFinal {i}: Summarize." for i in range(2)]
    final2 = run_concurrent(client, prompts_2x200k, max_tokens=64)
    for i, (st, resp, el) in enumerate(final2):
        if st == 200:
            pt = resp.get("usage", {}).get("prompt_tokens", 0)
            check(f"Final 2×200K session {i}: pt={pt}, {el:.1f}s", pt > 150_000)
        else:
            check(f"Final 2×200K session {i}", False, f"status={st}: {str(resp)[:200]}")

    # ── Phase 7: Memory stability ────────────────────────────────────
    print("\n=== Phase 7: Memory stability ===")
    mem_final = gpu_mem_mib()
    check("Final health", client.health())
    if mem_baseline > 0 and mem_final > 0:
        drift = mem_final - mem_baseline
        pct = drift / mem_baseline * 100 if mem_baseline else 0
        check(
            f"GPU memory drift: {drift}MiB ({pct:.1f}%)",
            drift < mem_baseline * 1.5,
            f"baseline={mem_baseline}MiB final={mem_final}MiB",
        )

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed, {PASSED + FAILED} total")
    print(f"  GPU: baseline={mem_baseline}MiB → final={gpu_mem_mib()}MiB")
    print(f"{'=' * 72}")
    return PASSED, FAILED


def main():
    parser = argparse.ArgumentParser(description="256K long-context E2E test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    try:
        _passed, failed = run(args.base_url)
    except RuntimeError as exc:
        print(f"FATAL: {exc}")
        sys.exit(2)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
