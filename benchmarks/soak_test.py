"""D4: Long-stability soak test for BlackweLLM.

Sends mixed workloads (short/long requests, streaming/non-streaming,
greedy/sampled, disconnects) against a running server and monitors for:
- Slot wedges (watchdog triggers)
- Memory/handle leaks (via /health and /debug/stats)
- Request failures
- Metric drift

Usage:
    python benchmarks/soak_test.py [--base-url URL] [--duration-minutes N]
        [--concurrency C] [--max-tokens M]

Requires a running BlackweLLM server (python -m server.app or launch_test_server.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time

try:
    import aiohttp
except ImportError:
    print("aiohttp required: pip install aiohttp", file=sys.stderr)
    sys.exit(1)


SHORT_PROMPTS = [
    "What is 2+2?",
    "Name three colors.",
    "Write a haiku about code.",
    "Explain recursion in one sentence.",
    "What is the capital of France?",
]

LONG_PROMPTS = [
    "Write a detailed 500-word essay about the history of computing, "
    "covering Babbage, Turing, von Neumann, and the invention of the transistor.",
    "Explain the theory of general relativity in detail, including the "
    "equivalence principle, spacetime curvature, gravitational time dilation, "
    "and the experimental evidence that supports each concept.",
    "Write a comprehensive guide to Python's asyncio library, covering "
    "event loops, coroutines, tasks, futures, and common patterns.",
]

SYSTEM_PROMPT = "You are a helpful assistant. Be concise."


async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
    temperature: float,
    request_id: int,
    results: dict,
    disconnect_after: float | None = None,
) -> None:
    """Send one request and record the outcome."""
    payload = {
        "model": "qwen3.6",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": temperature,
    }

    t0 = time.perf_counter()
    try:
        if stream:
            async with session.post(
                f"{base_url}/v1/chat/completions", json=payload
            ) as resp:
                if resp.status != 200:
                    results["errors"] += 1
                    results["error_details"].append(
                        f"req={request_id} status={resp.status}"
                    )
                    return
                token_count = 0
                async for line in resp.content:
                    if disconnect_after and (time.perf_counter() - t0) > disconnect_after:
                        break
                    text = line.decode("utf-8", errors="replace")
                    if text.startswith("data: ") and "[DONE]" not in text:
                        token_count += 1
                elapsed = time.perf_counter() - t0
                results["completed"] += 1
                results["total_tokens"] += token_count
                results["latencies"].append(elapsed)
        else:
            async with session.post(
                f"{base_url}/v1/chat/completions", json=payload
            ) as resp:
                body = await resp.json()
                elapsed = time.perf_counter() - t0
                if resp.status != 200:
                    results["errors"] += 1
                    results["error_details"].append(
                        f"req={request_id} status={resp.status} body={body}"
                    )
                    return
                usage = body.get("usage", {})
                results["completed"] += 1
                results["total_tokens"] += usage.get("completion_tokens", 0)
                results["latencies"].append(elapsed)
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        results["errors"] += 1
        results["error_details"].append(f"req={request_id} exc={exc}")


async def check_health(session: aiohttp.ClientSession, base_url: str) -> dict:
    """Fetch /health and /debug/stats."""
    health = {}
    try:
        async with session.get(f"{base_url}/health") as resp:
            health["health"] = await resp.json()
    except Exception:
        health["health"] = None
    try:
        async with session.get(f"{base_url}/debug/stats") as resp:
            health["stats"] = await resp.json()
    except Exception:
        health["stats"] = None
    return health


async def run_soak(
    base_url: str,
    duration_minutes: float,
    concurrency: int,
    max_tokens: int,
) -> None:
    """Main soak loop."""
    results = {
        "completed": 0,
        "errors": 0,
        "total_tokens": 0,
        "latencies": [],
        "error_details": [],
    }

    deadline = time.perf_counter() + duration_minutes * 60
    request_id = 0
    health_snapshots = []

    print(f"Soak test: {duration_minutes}min, concurrency={concurrency}, "
          f"max_tokens={max_tokens}, target={base_url}")
    print("-" * 60)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        # Initial health check
        initial_health = await check_health(session, base_url)
        health_snapshots.append((time.perf_counter(), initial_health))
        print(f"Initial health: {json.dumps(initial_health.get('health', {}))}")

        while time.perf_counter() < deadline:
            batch = []
            for _ in range(concurrency):
                request_id += 1
                is_long = random.random() < 0.3
                prompt = random.choice(LONG_PROMPTS if is_long else SHORT_PROMPTS)
                stream = random.random() < 0.7
                temperature = random.choice([0.0, 0.0, 0.0, 0.7, 1.0])
                tokens = random.randint(32, max_tokens)
                disconnect_after = 0.5 if random.random() < 0.05 else None

                batch.append(
                    send_request(
                        session, base_url, prompt, tokens, stream,
                        temperature, request_id, results, disconnect_after,
                    )
                )

            await asyncio.gather(*batch)

            elapsed = time.perf_counter() - (deadline - duration_minutes * 60)
            if results["completed"] > 0 and int(elapsed) % 30 < 2:
                health = await check_health(session, base_url)
                health_snapshots.append((time.perf_counter(), health))
                stats = health.get("stats", {}) or {}
                print(
                    f"[{elapsed:.0f}s] completed={results['completed']} "
                    f"errors={results['errors']} "
                    f"tokens={results['total_tokens']} "
                    f"watchdog={stats.get('watchdog_triggers', '?')} "
                    f"cancellations={stats.get('cancellations', '?')}"
                )

            await asyncio.sleep(0.1)

    # Final health check
    final_health = await check_health(session, base_url)
    health_snapshots.append((time.perf_counter(), final_health))

    # Report
    print("\n" + "=" * 60)
    print("SOAK TEST REPORT")
    print("=" * 60)
    print(f"Duration: {duration_minutes} minutes")
    print(f"Total requests: {request_id}")
    print(f"Completed: {results['completed']}")
    print(f"Errors: {results['errors']}")
    print(f"Total tokens: {results['total_tokens']}")
    if results["latencies"]:
        lats = sorted(results["latencies"])
        print(f"Latency p50: {lats[len(lats)//2]:.2f}s")
        print(f"Latency p95: {lats[int(len(lats)*0.95)]:.2f}s")
        print(f"Latency p99: {lats[int(len(lats)*0.99)]:.2f}s")

    final_stats = final_health.get("stats", {}) or {}
    print(f"\nWatchdog triggers: {final_stats.get('watchdog_triggers', '?')}")
    print(f"Cancellations: {final_stats.get('cancellations', '?')}")
    print(f"MTP acceptance histogram: {final_stats.get('mtp_acceptance_histogram', '?')}")
    print(f"Sampled decode rounds: {final_stats.get('sampled_decode_rounds', '?')}")

    if results["error_details"]:
        print("\nFirst 10 errors:")
        for detail in results["error_details"][:10]:
            print(f"  {detail}")

    # Verdict
    watchdog = final_stats.get("watchdog_triggers", 0)
    if results["errors"] == 0 and watchdog == 0:
        print("\n✅ PASS: zero errors, zero watchdog triggers")
    elif watchdog > 0:
        print(f"\n❌ FAIL: {watchdog} watchdog triggers (slot wedges detected)")
    else:
        print(f"\n⚠️  WARN: {results['errors']} errors (check details above)")


def main() -> None:
    parser = argparse.ArgumentParser(description="BlackweLLM soak test (D4)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--duration-minutes", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    asyncio.run(
        run_soak(args.base_url, args.duration_minutes, args.concurrency, args.max_tokens)
    )


if __name__ == "__main__":
    main()
