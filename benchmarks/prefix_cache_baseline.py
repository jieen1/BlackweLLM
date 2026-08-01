#!/usr/bin/env python3
"""Collect a reproducible prefix-cache baseline against a running server.

Why this script exists
-----------------------
Track A step 7-g's acceptance criteria (`docs/implementation-plan.md` §6 step
7) include "prefix cache hit rate does not regress". The only number on record
before this script existed was a single e2e-script run
(`notes/prefix-cache-implementation-log.md`: `prefix_cache_hits=1 /
misses=10 / hit_rate=0.0909`) that aggregates hit/miss events across an entire
multi-subtest correctness suite -- most of those admissions are DIFFERENT,
unrelated prompt content (independent-reference-replay cases, concurrent-
batching cases, etc.), not turns of the one growing conversation the
prefix cache exists to serve. One number, no per-round sequence, and not
scoped to the workload it is meant to describe -- not usable as a regression
floor (see `notes/2026-08-02-prefix-cache-baseline.md` for the full
reasoning, including why `hit_rate` itself is a weak signal even when scoped
correctly, and why `prefix_cache_hit_tokens_saved` -- or better, the tokens
actually saved as a *fraction of what a healthy cache should have saved* --
is the metric this script is built to produce).

What this script measures
--------------------------
A fixed, repeatable synthetic agent workload: `--trials` independent
conversations, each `--turns` turns long, each turn RESENDING THE WHOLE
HISTORY plus one new question -- exactly what a real agent client does, and
exactly the shape the prefix cache exists to exploit (mirrors `_warm()` in
`benchmarks/repro_prefix_cache_slowdown.py`). Each trial's system preamble
carries a unique random marker so trials never share a prefix with each
other (a `--fresh-lengths`-style guard borrowed from the same repro script) --
turn 1 of every trial is therefore a genuine cold miss, never an accidental
hit off a previous trial's leftover cache.

For every turn this script records, from the client's own vantage point plus
one `/debug/stats` poll immediately after:

- wall-clock latency (client-observed, includes both prefill and decode --
  NOT a clean prefill-only signal; report it as directional context, not the
  regression gate)
- `usage.prompt_tokens` / `usage.completion_tokens` / `finish_reason`
- whether THIS turn's admission was an engine-counted hit or miss
  (`prefix_cache_hits`/`misses` delta -- requests are sent strictly
  sequentially by this script, so there is no concurrent admission to
  confuse the delta with)
- `hit_L` for hits (from the tail of `prefix_cache_hit_L_samples`, which grew
  by exactly one entry this turn)
- `ideal_L` -- the previous turn's own `prompt_tokens`, floored to
  `--block-size` -- i.e. the deepest hit a healthy cache could possibly have
  served for a strict prefix-extension turn. `hit_L / ideal_L` is this
  script's proposed regression signal: 1.0 means the cache captured
  everything it could have; a regression that still registers `hit_L > 0`
  (so `hit_rate` alone would stay a deceptive 1.0) but only a sliver of it
  shows up here as a ratio well below 1.0.

A single throwaway warmup request runs first (absorbs the one-time M=1 decode
CUDA Graph capture cost -- see `runtime/backends/laguna.py`
`_ensure_decode_cg`/`warmup_paged_attention_shapes`, the latter already runs
automatically at server startup, before `/v1/models` answers, so it is NOT
re-warmed here) and is reported separately, excluded from every statistic.

This script does not decide pass/fail. It prints the full per-round sequence
(never just a summary -- see the fox-64K lesson in
`notes/2026-08-02-track-a-step5-gpu-verification.md`: the same workload's
throughput swung ~60% depending on where in a call sequence it was measured,
and the original baseline recorded only aggregates, making it impossible to
tell which regime 353-368 tok/s came from) and writes the raw per-round data
as JSON so a future run (post-7-g) can be diffed against it by hand or by a
follow-up tool.

Usage
-----
    python -m benchmarks.prefix_cache_baseline --base-url http://127.0.0.1:8100

    # repeat identically against a post-7-g server for an A/B:
    python -m benchmarks.prefix_cache_baseline --base-url http://127.0.0.1:8100 \\
        --out benchmarks/fixtures/prefix_cache_baseline_post_7g.json --label post-7g

Self-contained (stdlib + urllib only, like `repro_prefix_cache_slowdown.py`):
talks HTTP to an already-running server, does not import `runtime`/`server`,
and does not need a GPU of its own.
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

DEFAULT_MAX_TOKENS = 32


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


def _trial_preamble(trial_index: int, filler_chars: int) -> str:
    """Distinct-content system preamble: never shares a prefix with any
    other trial (own random marker, mirrors `_fresh_length_prompt` in
    `repro_prefix_cache_slowdown.py`), so turn 1 of every trial is a
    genuine cold miss rather than an accidental hit off leftover cache."""
    marker = f"[[trial {trial_index} unique marker {os.urandom(4).hex()}]] "
    filler = "背景资料：" + ("这是一段用于把上下文撑大的填充文本，本轮基线测量专用。" * (filler_chars // 24 + 1))
    return (marker + filler)[: max(filler_chars, len(marker))]


def _run_warmup(base_url: str, model: str, timeout: float) -> dict:
    """One throwaway request to absorb one-time process-level costs (M=1
    decode CUDA Graph capture) outside the measured trials. Not counted in
    any statistic."""
    body = _chat(model, [{"role": "user", "content": "你好"}], 8)
    payload, elapsed = _post(base_url, "/v1/chat/completions", body, timeout)
    return {
        "wall_s": elapsed,
        "prompt_tokens": payload.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
    }


def run_baseline(
    base_url: str,
    model: str,
    trials: int,
    turns: int,
    filler_chars: int,
    max_tokens: int,
    block_size: int,
    timeout: float,
) -> tuple[dict, list[dict]]:
    """Run the fixed workload once; return (warmup_record, rounds)."""
    warmup = _run_warmup(base_url, model, timeout)

    stats = _get(base_url, "/debug/stats")
    prev_hits = stats.get("prefix_cache_hits", 0)
    prev_misses = stats.get("prefix_cache_misses", 0)
    prev_saved = stats.get("prefix_cache_hit_tokens_saved", 0)
    prev_samples_len = len(stats.get("prefix_cache_hit_L_samples", []))

    rounds: list[dict] = []
    for trial in range(1, trials + 1):
        messages: list[dict] = [
            {"role": "system", "content": _trial_preamble(trial, filler_chars)}
        ]
        prev_turn_prompt_tokens: int | None = None
        for turn in range(1, turns + 1):
            messages.append(
                {
                    "role": "user",
                    "content": f"第 {turn} 个问题：请用一句话回答 1+{turn} 等于几。",
                }
            )
            body = _chat(model, messages, max_tokens)
            payload, elapsed = _post(base_url, "/v1/chat/completions", body, timeout)
            usage = payload.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            choice = payload["choices"][0]
            finish_reason = choice.get("finish_reason")
            answer = choice["message"].get("content") or ""
            messages.append({"role": "assistant", "content": answer})

            stats = _get(base_url, "/debug/stats")
            hits = stats.get("prefix_cache_hits", 0)
            misses = stats.get("prefix_cache_misses", 0)
            saved = stats.get("prefix_cache_hit_tokens_saved", 0)
            samples = stats.get("prefix_cache_hit_L_samples", [])

            engine_hit = hits > prev_hits
            engine_miss = misses > prev_misses
            hit_L = 0
            if engine_hit and len(samples) > prev_samples_len:
                hit_L = samples[-1].get("hit_L", 0)

            ideal_L = None
            tokens_saved_ratio = None
            if turn > 1 and prev_turn_prompt_tokens is not None:
                ideal_L = (prev_turn_prompt_tokens // block_size) * block_size
                if ideal_L > 0:
                    tokens_saved_ratio = hit_L / ideal_L

            rounds.append(
                {
                    "trial": trial,
                    "turn": turn,
                    "wall_s": round(elapsed, 4),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "finish_reason": finish_reason,
                    "engine_hit": engine_hit,
                    "engine_miss": engine_miss,
                    "hit_L": hit_L,
                    "ideal_L": ideal_L,
                    "tokens_saved_ratio": (
                        round(tokens_saved_ratio, 4) if tokens_saved_ratio is not None else None
                    ),
                    "cum_hits": hits,
                    "cum_misses": misses,
                    "cum_tokens_saved": saved,
                }
            )

            prev_hits, prev_misses, prev_saved = hits, misses, saved
            prev_samples_len = len(samples)
            prev_turn_prompt_tokens = prompt_tokens

            print(
                f"  trial {trial}/{trials} turn {turn}/{turns}: "
                f"{elapsed:6.2f}s  prompt_tokens={prompt_tokens:<6} "
                f"{'HIT ' if engine_hit else 'MISS'} hit_L={hit_L:<6} "
                f"ideal_L={ideal_L!s:<6} "
                f"ratio={tokens_saved_ratio if tokens_saved_ratio is not None else '-'}"
            )

    return warmup, rounds


def _print_summary(rounds: list[dict], turns: int) -> None:
    print("\n== per-turn-position summary (across all trials) ==")
    for turn in range(1, turns + 1):
        at_turn = [r for r in rounds if r["turn"] == turn]
        if not at_turn:
            continue
        wall_times = [r["wall_s"] for r in at_turn]
        hit_rate = sum(1 for r in at_turn if r["engine_hit"]) / len(at_turn)
        ratios = [r["tokens_saved_ratio"] for r in at_turn if r["tokens_saved_ratio"] is not None]
        ratio_str = f"{statistics.mean(ratios):.4f}" if ratios else "n/a"
        print(
            f"  turn {turn}: n={len(at_turn):<3} "
            f"wall_s median={statistics.median(wall_times):6.2f} "
            f"mean={statistics.mean(wall_times):6.2f} "
            f"stdev={(statistics.stdev(wall_times) if len(wall_times) > 1 else 0.0):5.2f}  "
            f"hit_rate={hit_rate:.2f}  mean_tokens_saved_ratio={ratio_str}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="laguna-s-2.1")
    ap.add_argument("--trials", type=int, default=8, help="independent growing conversations")
    ap.add_argument("--turns", type=int, default=6, help="turns per conversation")
    ap.add_argument("--filler-chars", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="server QSR_SERVER_BLOCK_SIZE -- hit_L/ideal_L are block-aligned to this",
    )
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--label", default="", help="tag recorded in the output JSON")
    ap.add_argument(
        "--out",
        default="",
        help="write full per-round JSON here (meta + warmup + rounds); omit to skip",
    )
    args = ap.parse_args()

    try:
        health = _get(args.base_url, "/health")
    except (urllib.error.URLError, OSError) as exc:
        print(f"server not reachable at {args.base_url}: {exc}", file=sys.stderr)
        return 2
    print(f"server: {health}")
    if args.label:
        print(f"label : {args.label}")

    print("\n[warmup] one throwaway request (absorbs one-time decode-CG capture cost)")
    warmup, rounds = run_baseline(
        args.base_url,
        args.model,
        args.trials,
        args.turns,
        args.filler_chars,
        args.max_tokens,
        args.block_size,
        args.timeout,
    )
    print(f"  warmup: {warmup['wall_s']:.2f}s (prompt_tokens={warmup['prompt_tokens']})")

    print(f"\n[trials] {args.trials} independent {args.turns}-turn conversations, full history resent each turn")
    _print_summary(rounds, args.turns)

    try:
        final_stats = _get(args.base_url, "/debug/stats")
        engine_cache = {
            k: final_stats.get(k)
            for k in (
                "prefix_cache_hits",
                "prefix_cache_misses",
                "prefix_cache_hit_rate",
                "prefix_cache_hit_tokens_saved",
            )
        }
    except Exception:  # noqa: BLE001 - diagnostics only
        engine_cache = {}
    print(f"\n== final /debug/stats prefix-cache fields (whole run, includes warmup) ==\n  {engine_cache}")

    if args.out:
        out_doc = {
            "meta": {
                "base_url": args.base_url,
                "model": args.model,
                "trials": args.trials,
                "turns": args.turns,
                "filler_chars": args.filler_chars,
                "max_tokens": args.max_tokens,
                "block_size": args.block_size,
                "label": args.label,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "final_debug_stats_prefix_cache_fields": engine_cache,
            },
            "warmup": warmup,
            "rounds": rounds,
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_doc, f, indent=2, ensure_ascii=False)
        print(f"\nwrote raw per-round data: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
