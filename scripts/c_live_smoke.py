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
  3. KNOWN GAP, do not trust a PASS here as coverage: many back-to-back
     conversation turns over one persistent connection, no gap between
     them, reaching for the lost-wakeup bug's window (server/engine.py's
     _step_sync comment: an agent's follow-up turn landed 152ms after the
     previous one finished and froze the engine). Verified empirically on
     53f0dca (the bug's own parent commit, where it is 100% present): 20
     rapid-fire rounds over a persistent connection, zero sleep, zero
     print -- fastest 0.21s, slowest 0.28s, zero stalls. It did not catch
     it. Root cause (read from _step_sync, not guessed): the real window is
     the gap between two adjacent Python statements
     (_drain_requests()/_drain_pipe()) inside one engine round, not a
     client-observable span of wall-clock time -- no black-box HTTP timing
     can reliably land inside it. d9e52ce's own regression test needed to
     hook _drain_pipe directly to hit it deterministically. This check
     stays in as a cheap sanity net (it WOULD catch a coarser regression
     that turned the whole idle path pathologically slow) but a green run
     provides no evidence the specific lost-wakeup race is fixed or absent.
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
    # Only the value line matters -- the two comment lines Prometheus text
    # format always emits ("# HELP ...", "# TYPE ...") also contain the
    # metric name and would make this always look "warm" if not excluded.
    value_lines = [
        line
        for line in pre_body.splitlines()
        if line.startswith("blackwellm:requests_completed_total{")
    ]
    already_warm = (
        pre_status == 200 and value_lines and not all(line.endswith("} 0") for line in value_lines)
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

    # ── 5. Lost-wakeup: N rapid-fire turns, no gap between them ──────────
    # First cut of this check (single back-to-back pair over a fresh
    # http.client.HTTPConnection each time) did NOT reproduce the bug on
    # 53f0dca -- confirmed by running it there. Root cause, from reading
    # _step_sync itself: the race is a request landing between
    # _drain_requests() and _drain_pipe() at the top of the round where the
    # engine is *also* about to find active+waiting both empty and block --
    # a window on the order of the gap between two consecutive Python
    # statements, not 200ms. A fresh HTTPConnection per call spends that
    # 200ms budget on TCP handshake + Python overhead sitting AFTER the
    # engine has already reached its blocking read, where a new byte wakes
    # it normally -- i.e. exactly the case that is NOT buggy. This version
    # reuses ONE persistent connection across every turn (no new handshake,
    # no sleep, no print inside the loop) to shrink client-side latency as
    # close to zero as an HTTP client can get, and repeats it many times
    # since each idle transition is an independent chance to land in the
    # window, not a one-shot dice roll.
    print("\n=== 5. Rapid-fire back-to-back turns (lost-wakeup window) ===")
    n_rapid = 20
    slow_round_s = 8.0  # generous vs. a small completion's real latency, tiny vs. a real hang
    per_round: list[float | None] = []
    conn = http.client.HTTPConnection(client.host, client.port, timeout=slow_round_s + 5)
    for i in range(n_rapid):
        body = json.dumps(
            {
                "model": "qwen3.6-rt",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": f"Reply with just the number {i}."}],
            }
        )
        t0 = time.perf_counter()
        try:
            conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
            resp = conn.getresponse()
            resp.read()
            per_round.append(time.perf_counter() - t0)
        except OSError:
            # A dead/reset connection after a stall is itself a symptom, not
            # a script bug -- record it as a stall and get a fresh
            # connection so the remaining rounds can still run.
            per_round.append(None)
            try:
                conn.close()
            except Exception:
                pass
            conn = http.client.HTTPConnection(client.host, client.port, timeout=slow_round_s + 5)
    conn.close()
    stalled = [i for i, t in enumerate(per_round) if t is None or t > slow_round_s]
    fast_times = [t for t in per_round if t is not None]
    check(
        f"{n_rapid} rapid-fire turns: every round < {slow_round_s:.0f}s "
        f"(fastest={min(fast_times):.2f}s slowest={max(fast_times):.2f}s)"
        if fast_times
        else f"{n_rapid} rapid-fire turns: every round < {slow_round_s:.0f}s (none completed)",
        not stalled,
        f"stalled rounds (0-indexed): {stalled} -- a stall here IS the lost-wakeup bug "
        f"(see server/engine.py _step_sync), not a slow-model false positive; "
        f"per_round={['-' if t is None else round(t, 2) for t in per_round]}",
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
