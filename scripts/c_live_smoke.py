#!/usr/bin/env python3
"""P0-B · C-LIVE: the live-server smoke gate (docs/implementation-plan.md §3).

"The goal is not coverage, it's turning 'assumptions only deployment facts
can falsify' into automated assertions" (§3, translated). Three real bugs
shipped past 1100+ passing unit tests on 2026-08-01 because all three
depended on deployment facts (a cold engine, a busy engine, a request
landing inside a specific timing window) that no unit test can construct.
This script hits a real, already-running server and checks exactly those
facts. See docs/architecture.md §3.5 / §0.1 for the three bugs' postmortems.

implementation-plan.md §3.1 says to collect the three existing e2e scripts
(tests/test_real_world.py, tests/test_api_compat.py,
tests/test_e2e_256k_longctx.py -- refactored to be importable in the C-LIVE
B-1 commit) rather than build a fourth from scratch. This script follows
that: item 4 below (dual protocol x stream/non-stream x tool call) is
tests/test_api_compat.py's existing coverage, called directly. Items 1, 2,
3, 5, 6 have no existing coverage anywhere (confirmed by reading all three
scripts before writing this one) because they are the specific historical
bugs from architecture.md §0.1 / notes/2026-07-27-p1-http-e2e-and-thinking-
strip-bug.md -- this file is the smallest amount of new code that closes
those six gaps.

The six checks (implementation-plan.md §3.2):
  1. /metrics in the COLD-START window (before any request has completed)
     returns 200 with all aggregate keys present.
  2. /metrics while a request is ACTIVELY RUNNING (engine.active non-empty)
     returns 200 -- this is the exact shape of the second 500 on 2026-08-01
     (a list read as a mapping).
  3. Two back-to-back conversation turns, the second fired < 200ms after
     the first's response lands -- the exact window of the lost-wakeup bug
     (server/engine.py's _step_sync comment: an agent's follow-up turn
     landed 152ms after the previous one finished and froze the engine).
  4. OpenAI + Anthropic, streaming + non-streaming, one tool call each --
     delegated to tests/test_api_compat.py.
  5. Thinking contract: on a real model whose chat template injects
     <think>, content's first characters must never be `</think>` (the
     comment-reversed bug that leaked the closing tag into visible text).
  6. /v1/completions returns its raw completion verbatim (no chat template,
     no thinking-strip wrapping) -- the exact P1 empty-output bug.

IMPORTANT precondition for check 1: run this immediately after starting a
fresh server, before issuing any other request to it (including via curl,
a browser, or blackwellm_ctl.sh's own /v1/models startup poll -- that one
is safe, see below). `stats["requests_completed"]` only increments when a
*generation* request finishes (server/engine.py _finish_request), so
health/models/metrics probes during startup do not consume the cold
window -- only a completed chat/completions or messages call does. If this
script is not the first generation request against the server, check 1 is
still executed but its result no longer means what it claims; this script
warns rather than silently mis-reporting when it detects that.

Usage:
    python scripts/c_live_smoke.py [--base-url http://127.0.0.1:8100]

Exit code 0 = all checks passed, 1 = at least one failed, 2 = server
unreachable at start.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

# tests/ has no __init__.py -- it is a namespace package. Reach it by path
# rather than requiring the caller to set PYTHONPATH, since this script is
# meant to be invoked directly (scripts/blackwellm_ctl.sh, `make smoke`) the
# same way tests/test_api_compat.py itself is.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests import test_api_compat  # noqa: E402

METRICS_KEYS = (
    "blackwellm:num_requests_running",
    "blackwellm:num_requests_waiting",
    "blackwellm:kv_cache_usage_perc",
    "blackwellm:num_free_slots",
    "blackwellm:requests_completed_total",
    "blackwellm:prefix_cache_hit_rate",
)


class Client:
    """Minimal http.client wrapper -- same shape as the other three e2e
    scripts use, deliberately not sharing a module with them (each of the
    four scripts is meant to run standalone; see AGENTS.md's fixed-scope
    guidance against speculative shared abstractions)."""

    def __init__(self, base_url: str):
        parsed = urlparse(base_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 80

    def request(
        self, method: str, path: str, body: dict | None = None, timeout: float = 30
    ) -> tuple[int, str]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            headers = {"Content-Type": "application/json"} if body is not None else {}
            conn.request(method, path, json.dumps(body) if body is not None else None, headers)
            resp = conn.getresponse()
            return resp.status, resp.read().decode()
        finally:
            conn.close()

    def get(self, path: str, timeout: float = 10) -> tuple[int, str]:
        return self.request("GET", path, timeout=timeout)

    def post(self, path: str, body: dict, timeout: float = 120) -> tuple[int, str]:
        return self.request("POST", path, body, timeout=timeout)


def run(base_url: str) -> tuple[int, int]:
    """Run every C-LIVE check against ``base_url``. Returns (passed, failed)
    without exiting the process, so a caller (e.g. a future bfdiag run
    record writer for B-4) can inspect the result.

    Raises ``RuntimeError`` if the server is unreachable at the start --
    same convention as tests/test_e2e_256k_longctx.py.run().
    """
    client = Client(base_url)
    passed = 0
    failed = 0

    def check(label: str, ok: bool, detail: str = "") -> bool:
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f"  [PASS] {label}")
        else:
            failed += 1
            print(f"  [FAIL] {label}  {detail}")
        return ok

    def assert_metrics_keys(context: str) -> None:
        status, body = client.get("/metrics")
        check(f"/metrics ({context}): status 200", status == 200, f"got {status}: {body[:200]}")
        if status != 200:
            return
        missing = [k for k in METRICS_KEYS if k not in body]
        check(
            f"/metrics ({context}): all {len(METRICS_KEYS)} aggregate keys present",
            not missing,
            f"missing={missing}",
        )

    print("=" * 72)
    print("  P0-B C-LIVE smoke gate")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Server: {base_url}")
    print("=" * 72)

    try:
        status, _ = client.get("/health")
    except OSError as exc:
        raise RuntimeError(f"server not reachable at {base_url}: {exc}") from exc
    if status != 200:
        raise RuntimeError(f"server not reachable at {base_url} (health status={status})")

    # ── 1. Cold-start /metrics ───────────────────────────────────────────
    # Must be the FIRST generation-adjacent check -- see module docstring.
    print("\n=== 1. /metrics: cold-start window (zero requests completed) ===")
    pre_status, pre_body = client.get("/metrics")
    already_warm = (
        pre_status == 200
        and "requests_completed_total" in pre_body
        and not all(
            "} 0" in line for line in pre_body.splitlines() if "requests_completed_total" in line
        )
    )
    if already_warm:
        print(
            "  WARNING: requests_completed_total is already non-zero -- this "
            "server has served a generation request before this script ran, "
            "so check 1 below does not exercise the cold-start branch. Run "
            "this script immediately after a fresh `restart`, before any "
            "other client hits it, to make this check meaningful."
        )
    assert_metrics_keys("cold-start")

    # ── 2. /v1/completions verbatim (P1 empty-output bug) ────────────────
    print("\n=== 2. /v1/completions returns raw completion verbatim ===")
    status, raw = client.post(
        "/v1/completions",
        {
            "model": "qwen3.6-rt",
            "prompt": "The capital of France is",
            "max_tokens": 16,
            "temperature": 0,
        },
    )
    check("/v1/completions: status 200", status == 200, f"got {status}: {raw[:200]}")
    if status == 200:
        data = json.loads(raw)
        text = data.get("choices", [{}])[0].get("text", "")
        check(
            "/v1/completions: non-empty text (notes/2026-07-27-p1-http-e2e-"
            "and-thinking-strip-bug.md)",
            len(text) > 0,
            f"got empty text; full response={raw[:300]}",
        )
        check(
            "/v1/completions: no <think>/</think> leaked into text",
            "<think>" not in text and "</think>" not in text,
            f"text={text!r}",
        )

    # ── 3. Thinking contract: content never starts with </think> ────────
    print("\n=== 3. Thinking contract: content's first chars are never </think> ===")
    status, raw = client.post(
        "/v1/chat/completions",
        {
            "model": "qwen3.6-rt",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
        },
    )
    check("chat completions: status 200", status == 200, f"got {status}: {raw[:200]}")
    if status == 200:
        data = json.loads(raw)
        content = data.get("choices", [{}])[0].get("message", {}).get("content") or ""
        check(
            "content does not start with </think> (architecture.md §0.1)",
            not content.startswith("</think>"),
            f"content={content!r}",
        )

    # ── 4. Dual protocol x stream/non-stream x tool call ─────────────────
    print("\n=== 4. Dual protocol coverage (delegated to tests/test_api_compat.py) ===")
    api_passed, api_failed, _results = test_api_compat.run(base_url)
    passed += api_passed
    failed += api_failed

    # ── 5. Lost-wakeup: back-to-back turns < 200ms apart ─────────────────
    print("\n=== 5. Back-to-back turns < 200ms apart (lost-wakeup window) ===")
    status1, raw1 = client.post(
        "/v1/chat/completions",
        {
            "model": "qwen3.6-rt",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
        },
        timeout=60,
    )
    check("turn 1: status 200", status1 == 200, f"got {status1}: {raw1[:200]}")
    # No sleep here on purpose -- the whole point is to fire turn 2 as fast
    # as this process can, which is comfortably under the 152ms window that
    # triggered the bug live.
    t0 = time.perf_counter()
    status2, raw2 = client.post(
        "/v1/chat/completions",
        {
            "model": "qwen3.6-rt",
            "max_tokens": 32,
            "messages": [
                {"role": "user", "content": "Say hello in one word."},
                {"role": "assistant", "content": "Hello."},
                {"role": "user", "content": "Say goodbye in one word."},
            ],
        },
        timeout=30,
    )
    turn2_ms = (time.perf_counter() - t0) * 1000
    check(
        f"turn 2 (fired <1ms after turn 1 landed): status 200 within 30s (took {turn2_ms:.0f}ms)",
        status2 == 200,
        f"got {status2}: {raw2[:200]} -- a hang here IS the lost-wakeup bug, "
        f"not a slow-model false positive (see server/engine.py _step_sync)",
    )

    # ── 6. Busy /metrics (engine.active non-empty) ───────────────────────
    print("\n=== 6. /metrics while a request is actively running ===")
    gen_done = threading.Event()

    def long_gen():
        client.post(
            "/v1/chat/completions",
            {
                "model": "qwen3.6-rt",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Write a long essay about history."}],
            },
            timeout=180,
        )
        gen_done.set()

    t = threading.Thread(target=long_gen, daemon=True)
    t.start()
    time.sleep(2)  # let admission happen before we check
    assert_metrics_keys("busy")
    gen_done.wait(timeout=180)

    print(f"\n{'=' * 72}")
    print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 72}")
    return passed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8100",
        help="Default matches scripts/blackwellm_ctl.sh's PORT (8100, not "
        "8000 -- 8000 is held by an unrelated service on this host).",
    )
    args = parser.parse_args()
    try:
        _passed, failed = run(args.base_url)
    except RuntimeError as exc:
        print(f"FATAL: {exc}")
        sys.exit(2)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
