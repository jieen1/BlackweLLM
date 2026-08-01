#!/usr/bin/env python3
"""Reproduce the prefix-cache slowdown against a running BlackweLLM server.

Symptom (observed live 2026-08-01): after a few multi-turn requests have warmed
the slots, a *tiny* request -- five prompt tokens, twenty output tokens -- takes
tens of seconds to well over a minute. During that time exactly one CPU core is
pinned at 100% and the GPU sits at 2-4%; the GPU only wakes for the final ~2
seconds. So the cost is neither prefill nor decode: it is single-threaded host
work on the request path, and it scales with what is *cached*, not with what was
asked.

Usage
-----
    python -m benchmarks.repro_prefix_cache_slowdown --base-url http://127.0.0.1:8100

    # A/B a fix: run against a server started with the cache off
    QSR_SERVER_ENABLE_PREFIX_CACHE=0 scripts/blackwellm_ctl.sh restart
    python -m benchmarks.repro_prefix_cache_slowdown --label no-prefix-cache

The script is deliberately self-contained (stdlib + requests) and talks HTTP
only, so it reproduces what a client actually experiences rather than what the
engine believes internally. It does not need a GPU of its own and does not
import the runtime.

What it does
------------
1. Baseline: time a tiny request against freshly restarted, cold slots.
2. Warm: send a long multi-turn conversation so slots hold large cached prefixes.
3. Probe: repeat the *same tiny request* several times and time each one.

A healthy server keeps step 3 close to step 1. The bug shows up as step 3 being
one to two orders of magnitude slower, growing with how much got cached in
step 2.

Exit status is non-zero when the slowdown factor exceeds --threshold, so this
doubles as a regression gate once the cause is fixed.

--fresh-lengths mode
--------------------
A *second*, independently-triggerable regression: SparkInfer's paged-attention
kernel used to JIT-compile per distinct request shape (~30s the first time any
given shape was seen -- see notes/2026-08-01-prefill-shape-buckets-root-cause.md).
Real agent traffic almost never repeats a prompt length exactly, so every turn
paid that ~30s cost. This is a *different* trigger from the warm-slot/tiny-probe
scenario above (which is about how much is cached, not about the prompt's own
length): a fresh server, hit with N structurally different prompt lengths in a
row, none repeated, is enough to reproduce it on its own -- no warm-up phase
needed.

    python -m benchmarks.repro_prefix_cache_slowdown --fresh-lengths

Sends one request per length in --fresh-lengths-list (approximate prompt
tokens, comma-separated; default spans 500-30000), each with distinct filler
content so no two prompts -- or their prefixes -- are identical. Exit status is
non-zero if any individual request exceeds --fresh-lengths-threshold-s, which
doubles as a regression gate: a healthy server keeps every one of these under
a few seconds once the model is loaded; the bug shows up as one or more
~30s-multiple spikes scattered across the run, correlated with previously
unseen shapes rather than with prompt length itself.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

TINY_PROMPT = "你好"
TINY_MAX_TOKENS = 20

# Repeated English filler tokenizes fairly predictably (~4 chars/token for
# most BPE tokenizers on common English text), which is good enough for
# "approximately this many tokens, and structurally distinct from every
# other request in this run" -- exactness is not the point; distinct,
# never-before-seen shapes are.
_CHARS_PER_TOKEN_ESTIMATE = 4
_DEFAULT_FRESH_LENGTHS = "500,1000,2000,4000,6000,10000,16000,24000,30000"


def _post(base_url: str, path: str, body: dict, timeout: float) -> tuple[dict, float]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    return payload, time.perf_counter() - t0


def _get(base_url: str, path: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as resp:
        return json.loads(resp.read())


def _chat(model: str, messages: list[dict], max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }


def _tiny(base_url: str, model: str, timeout: float) -> float:
    body = _chat(model, [{"role": "user", "content": TINY_PROMPT}], TINY_MAX_TOKENS)
    _, elapsed = _post(base_url, "/v1/chat/completions", body, timeout)
    return elapsed


def _warm(base_url: str, model: str, turns: int, filler_chars: int, timeout: float) -> list[float]:
    """Grow a conversation so slots accumulate large cached prefixes.

    Each turn re-sends the whole history, which is exactly what an agent client
    does and exactly the shape the prefix cache exists to exploit.
    """
    filler = "背景资料：" + ("这是一段用于把上下文撑大的填充文本。" * (filler_chars // 20))
    messages: list[dict] = [{"role": "system", "content": filler}]
    times: list[float] = []
    for i in range(turns):
        messages.append({"role": "user", "content": f"第 {i + 1} 个问题：请用一句话回答 1+{i} 等于几。"})
        body = _chat(model, messages, 48)
        payload, elapsed = _post(base_url, "/v1/chat/completions", body, timeout)
        times.append(elapsed)
        answer = payload["choices"][0]["message"].get("content") or ""
        messages.append({"role": "assistant", "content": answer})
        print(f"  warm turn {i + 1}/{turns}: {elapsed:7.2f}s  (prompt grows each turn)")
    return times


def _fresh_length_prompt(target_tokens: int, trial_index: int) -> str:
    """Build a prompt of ~target_tokens tokens, distinct from every other trial.

    Distinct content (not just distinct length) matters: it also defeats the
    prefix cache, so this mode measures shape-driven cost in isolation from
    the separate warm-slot/tiny-probe scenario above.
    """
    marker = f"[[trial {trial_index} unique marker {os.urandom(4).hex()}]] "
    phrase = (
        "The quick brown fox jumps over the lazy dog near the riverbank. "
        "In a galaxy far far away, a curious explorer studied ancient machines. "
    )
    target_chars = target_tokens * _CHARS_PER_TOKEN_ESTIMATE
    body = phrase * (target_chars // max(len(phrase), 1) + 1)
    return (marker + body)[: max(target_chars, len(marker))]


def _fresh_lengths(
    base_url: str, model: str, lengths: list[int], max_tokens: int, timeout: float
) -> list[float]:
    times: list[float] = []
    for i, target_tokens in enumerate(lengths):
        prompt = _fresh_length_prompt(target_tokens, i)
        body = _chat(model, [{"role": "user", "content": prompt}], max_tokens)
        _, elapsed = _post(base_url, "/v1/chat/completions", body, timeout)
        times.append(elapsed)
        print(
            f"  fresh length {i + 1}/{len(lengths)} (~{target_tokens} prompt tokens): "
            f"{elapsed:7.2f}s"
        )
    return times


def _run_fresh_lengths_mode(args: argparse.Namespace) -> int:
    try:
        health = _get(args.base_url, "/health")
    except (urllib.error.URLError, OSError) as exc:
        print(f"server not reachable at {args.base_url}: {exc}", file=sys.stderr)
        return 2
    print(f"server: {health}")
    if args.label:
        print(f"label : {args.label}")

    lengths = [int(x) for x in args.fresh_lengths_list.split(",") if x.strip()]
    print(
        f"\n[fresh-lengths] {len(lengths)} distinct, never-repeated prompt "
        f"lengths (~{min(lengths)}-{max(lengths)} tokens), one request each, "
        "no warm-up phase"
    )
    times = _fresh_lengths(
        args.base_url, args.model, lengths, args.fresh_lengths_max_tokens, args.timeout
    )

    print("\n== summary ==")
    for target_tokens, elapsed in zip(lengths, times):
        flag = "  <-- SLOW" if elapsed > args.fresh_lengths_threshold_s else ""
        print(f"  ~{target_tokens:6d} tokens: {elapsed:7.2f}s{flag}")
    slow = [
        (t, e) for t, e in zip(lengths, times) if e > args.fresh_lengths_threshold_s
    ]
    print(f"\n  threshold: {args.fresh_lengths_threshold_s:.1f}s  slow requests: {len(slow)}/{len(times)}")

    if slow:
        print(
            "\nFAIL: one or more never-before-seen prompt lengths took longer than "
            "the threshold -- looks like a per-shape JIT compile (or equivalent) is "
            "landing inside request latency instead of being paid for at startup."
        )
        return 1
    print("\nOK: every distinct prompt length returned within threshold.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="laguna-s-2.1")
    ap.add_argument("--warm-turns", type=int, default=4)
    ap.add_argument("--filler-chars", type=int, default=4000)
    ap.add_argument("--probes", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="fail if median probe time exceeds baseline by this factor",
    )
    ap.add_argument("--label", default="", help="tag for the printed summary")
    ap.add_argument(
        "--fresh-lengths",
        action="store_true",
        help=(
            "run the 'every request uses a brand-new, never-repeated prompt "
            "length' mode instead of the warm-slot/tiny-probe mode -- see "
            "module docstring"
        ),
    )
    ap.add_argument(
        "--fresh-lengths-list",
        default=_DEFAULT_FRESH_LENGTHS,
        help="comma-separated approximate prompt token counts, one request each",
    )
    ap.add_argument("--fresh-lengths-max-tokens", type=int, default=24)
    ap.add_argument(
        "--fresh-lengths-threshold-s",
        type=float,
        default=15.0,
        help="fail if any single fresh-length request exceeds this many seconds",
    )
    args = ap.parse_args()

    if args.fresh_lengths:
        return _run_fresh_lengths_mode(args)

    try:
        health = _get(args.base_url, "/health")
    except (urllib.error.URLError, OSError) as exc:
        print(f"server not reachable at {args.base_url}: {exc}", file=sys.stderr)
        return 2
    print(f"server: {health}")
    if args.label:
        print(f"label : {args.label}")

    print("\n[1/3] baseline -- tiny request against cold slots")
    baseline = _tiny(args.base_url, args.model, args.timeout)
    print(f"  baseline: {baseline:7.2f}s")

    print(f"\n[2/3] warming -- {args.warm_turns} growing turns")
    _warm(args.base_url, args.model, args.warm_turns, args.filler_chars, args.timeout)

    print(f"\n[3/3] probes -- the SAME tiny request, {args.probes}x")
    probes = []
    for i in range(args.probes):
        elapsed = _tiny(args.base_url, args.model, args.timeout)
        probes.append(elapsed)
        print(f"  probe {i + 1}/{args.probes}: {elapsed:7.2f}s")

    median = statistics.median(probes)
    factor = median / baseline if baseline > 0 else float("inf")

    try:
        stats = _get(args.base_url, "/debug/stats")
        cache = {
            k: stats.get(k)
            for k in (
                "prefix_cache_hits",
                "prefix_cache_misses",
                "prefix_cache_hit_tokens_saved",
            )
        }
    except Exception:  # noqa: BLE001 - diagnostics only
        cache = {}

    print("\n== summary ==")
    print(f"  baseline (cold)   : {baseline:7.2f}s")
    print(f"  probe median      : {median:7.2f}s")
    print(f"  probe max         : {max(probes):7.2f}s")
    print(f"  slowdown factor   : {factor:7.2f}x  (threshold {args.threshold}x)")
    if cache:
        print(f"  prefix cache      : {cache}")

    if factor > args.threshold:
        print("\nFAIL: identical tiny requests got dramatically slower once slots were warm.")
        return 1
    print("\nOK: warm slots did not slow identical requests down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
