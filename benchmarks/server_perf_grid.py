"""Server-side performance grid: exact context length x concurrency.

Runs against a LIVE BlackweLLM server (qwen36 backend, MTP K=3 + CUDA Graph
+ prefix cache, 3 x 256K slots -- e.g. ``scripts/run_qwen36_quality.sh
server start best``). For each context length and concurrency:

  1. COLD wave: ``c`` concurrent identical streaming requests against an
     empty prefix cache -> populate it.
  2. WARM wave(s): the same ``c`` concurrent requests again -> persistent
     prefix-cache hit.

All headline numbers are taken from the server's own Prometheus metrics
(``/metrics``) and engine counters (``/debug/stats``), not from client-side
SSE parsing: with MTP a single stream delta can carry several tokens, and
Qwen3.6 spends part of its budget on reasoning that never reaches the
visible text stream. The client only drives concurrency and records event
timestamps for the aggregate decode window.

Context length is STRICT and block-aligned: the chat template adds a fixed
special-token overhead (measured by a tiny probe at startup), and the raw
prompt is sized so that the SERVED prompt_tokens (raw + overhead) is a
multiple of block_size=16. The persistent prefix cache only stores
checkpoints on block-aligned boundaries, so this is also what makes the
WARM wave actually hit. Per-wave ``prompt_tokens_total`` delta must equal
``served_context_tokens x concurrency``.

Historically comparable anchors (documented in
``notes/2026-08-05-server-perf-grid-mtp-cg-prefix.md``):

* 2026-08-03 CG-vs-eager capacity sweep (decode-only tok/s, server path):
  CG 28.56 / 47.71 / 68.59 at c=1/2/4, MTP off
  (``notes/2026-08-03-cudagraph-vs-eager-decode-throughput.md``).
* 2026-08-03 MTP serving verification: MTP-on 7.80 tok/s vs MTP-off 28.0,
  BEFORE the MTP CUDA-Graph fix
  (``notes/2026-08-03-mtp-serving-gpu-verification.md``).
* 2026-08-05 quality rerun probe: 2+2 total time 21.8s (eager) -> 4.1s
  (MTP+CG); 4096-token HumanEval 726.9s -> 46.5s
  (``notes/2026-08-05-qwen36-quality-rerun.md`` timeline).
* Prefix-cache P3.4 cold-vs-warm TTFT methodology, 15.4x exact-repeat
  ceiling from native vLLM (``benchmarks/prefix_cache_warm_throughput_check.py``,
  ``notes/prefix-cache-design.md``) -- ours is completion-boundary-only
  reuse, so a documented fraction of that ceiling is expected.

Usage:
    /home/bot/.venvs/vllm/bin/python benchmarks/server_perf_grid.py \
        --base-url http://127.0.0.1:8300 --model qwen3.6 \
        --contexts 4k,32k,64k,128k,250k --concurrency 1,2,3 \
        --max-tokens 256 --warm-rounds 1

    # Historical Pattern-B protocol (cached prefix + 10240 fresh suffix,
    # raw tokenization, no chat template -- matches native_warm_compare):
    /home/bot/.venvs/vllm/bin/python benchmarks/server_perf_grid.py \
        --base-url http://127.0.0.1:8300 --model qwen3.6 \
        --endpoint completions --contexts 64k,128k,200k \
        --concurrency 1 --max-tokens 256 --warm-rounds 1 \
        --warm-suffix-tokens 10240 --filler-prefix '9876543210 '

Output: ``benchmarks/fixtures/server_perf_grid_<ts>.json`` (full per-wave
records: server metric deltas, per-request timestamps, prefix/MTP/CG
counter deltas).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import aiohttp  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from runtime.checkpoints import standard_checkpoint_path  # noqa: E402

# Chosen so decode -> re-encode round-trips EXACTLY at any length (verified
# against the served tokenizer: repeated digit+space fillers tokenize 1:1,
# unlike prose fillers which collapse whitespace during decode).
# ``FILLER`` is the prefix filler; ``SUFFIX_FILLER`` is only used for the
# optional historical-protocol warm suffix (--warm-suffix-tokens), and is a
# DIFFERENT digit string so the suffix is genuinely fresh content that is
# not part of the cached prefix.
FILLER = "0123456789 "
SUFFIX_FILLER = "1357902468 "

COUNTER_KEYS = [
    "requests_completed",
    "prefix_cache_hits",
    "prefix_cache_misses",
    "prefix_cache_hit_tokens_saved",
    "decode_rounds",
    "decode_tokens",
    "decode_graph_replays",
    "mtp_verify_graph_replays",
    "mtp_draft_graph_replays",
    "mtp_batched_sync_replays",
    "prefix_persistent_stores",
    "prefix_persistent_evictions",
    "prefix_persistent_restores",
    "checkpoints_taken",
]

# (metric suffix, kind) -- scraped for endpoint="chat".
METRIC_KEYS = [
    "prompt_tokens_total",
    "generation_tokens_total",
    "e2e_request_latency_seconds_sum",
    "e2e_request_latency_seconds_count",
    "time_to_first_token_seconds_sum",
    "time_to_first_token_seconds_count",
    "request_time_per_output_token_seconds_sum",
    "request_time_per_output_token_seconds_count",
]


def parse_ctx(value: str) -> int:
    value = value.strip().lower()
    mult = 1
    if value.endswith("k"):
        mult, value = 1024, value[:-1]
    return int(value) * mult


def make_prompt(tokenizer, n_tokens: int, filler: str = FILLER) -> str:
    """Return a decoded prompt whose raw tokenization is exactly n_tokens."""
    chunk = tokenizer.encode(filler, add_special_tokens=False)
    ids = (chunk * (n_tokens // len(chunk) + 1))[:n_tokens]
    assert len(ids) == n_tokens
    text = tokenizer.decode(ids, skip_special_tokens=True)
    again = tokenizer.encode(text, add_special_tokens=False)
    assert len(again) == n_tokens, (len(again), n_tokens)
    return text


def make_suffix(tokenizer, n_tokens: int, filler: str = SUFFIX_FILLER) -> str:
    """Return fresh suffix text whose raw tokenization is exactly n_tokens.

    The suffix is appended to the prefix on the WARM wave (historical
    Pattern-B protocol: re-send the cached prefix + a 10240-token NEW
    suffix, so only the suffix + one partial block is re-prefilled).
    ``n_tokens == 0`` returns "".
    """
    if n_tokens <= 0:
        return ""
    chunk = tokenizer.encode(filler, add_special_tokens=False)
    ids = (chunk * (n_tokens // len(chunk) + 1))[:n_tokens]
    assert len(ids) == n_tokens
    text = tokenizer.decode(ids, skip_special_tokens=True)
    again = tokenizer.encode(text, add_special_tokens=False)
    assert len(again) == n_tokens, (len(again), n_tokens)
    return text


def chat_token_count(tokenizer, text: str) -> int:
    """Served token count of a chat request carrying ``text``.

    ``tokenize=True`` returns a dict in some transformers versions, so the
    raw ``len()`` over that return is meaningless; use ``return_dict=False``
    to get the flat id list.
    """
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
    )
    return len(ids)


def find_raw_for_served(tokenizer, served: int, filler: str = FILLER) -> int | None:
    """Exact raw prompt length whose served (chat) token count is ``served``.

    The chat template's overhead is NOT a constant: at some lengths the
    prompt's tail token merges with the template's newline/special token,
    so ``served - overhead`` is off by one (measured 2026-08-05: raw 32758
    -> served 32767 and raw 32759 -> served 32769 -- 32768 itself is
    unreachable).  Scan a small window around the estimate and require an
    exact match.  Returns ``None`` when no raw length maps to ``served``.
    """
    for raw_n in range(max(1, served - 32), served):
        text = make_prompt(tokenizer, raw_n, filler)
        if chat_token_count(tokenizer, text) == served:
            return raw_n
    return None


def find_prompt_for_target(tokenizer, target: int, filler: str = FILLER) -> tuple[int, int]:
    """``(raw_n, served)`` for the largest reachable block-aligned served
    context at or below ``target``.

    The served (chat) token count must be a multiple of block_size=16 for
    the persistent prefix cache to store a checkpoint, and some aligned
    values (e.g. 32768) are unreachable because the template boundary
    merges tokens.  Walk down in 16-token steps and take the first exact
    match, so every grid cell is strictly aligned and actually cachable.
    """
    for served in range(target & ~15, max(16, target - 1024), -16):
        raw_n = find_raw_for_served(tokenizer, served, filler)
        if raw_n is not None:
            return raw_n, served
    raise RuntimeError(f"no reachable block-aligned served context <= {target}")


def snapshot_stats(stats: dict) -> dict:
    dbg = stats.get("_backend_stats_dbg", {}) or {}
    out = {k: dbg.get(k, 0) for k in COUNTER_KEYS}
    out["requests_completed"] = stats.get("requests_completed", 0)
    out["prefix_cache_hits"] = stats.get("prefix_cache_hits", 0)
    out["prefix_cache_misses"] = stats.get("prefix_cache_misses", 0)
    out["mtp_acceptance_histogram"] = list(stats.get("mtp_acceptance_histogram", []))
    return out


def diff_stats(before: dict, after: dict) -> dict:
    out = {}
    for key, before_value in before.items():
        after_value = after.get(key, [] if isinstance(before_value, list) else 0)
        if isinstance(before_value, list):
            width = max(len(before_value), len(after_value))
            before_hist = [*before_value, *([0] * (width - len(before_value)))]
            after_hist = [*after_value, *([0] * (width - len(after_value)))]
            out[key] = [int(a) - int(b) for a, b in zip(after_hist, before_hist)]
        else:
            out[key] = int(after_value) - int(before_value)

    hist = out.get("mtp_acceptance_histogram", [])
    mtp_rounds = sum(hist)
    mtp_accepted = sum(index * count for index, count in enumerate(hist))
    out["mtp_rounds"] = mtp_rounds
    out["mtp_accepted_tokens"] = mtp_accepted
    out["mtp_committed_tokens"] = mtp_rounds + mtp_accepted
    out["mtp_mean_accepted_per_round"] = round(mtp_accepted / mtp_rounds, 6) if mtp_rounds else None
    out["mtp_mean_committed_per_round"] = (
        round((mtp_rounds + mtp_accepted) / mtp_rounds, 6) if mtp_rounds else None
    )
    return out


def _completion_evidence(text: str) -> dict:
    encoded = text.encode("utf-8")
    return {
        "completion_text": text,
        "completion_chars": len(text),
        "completion_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _metric_re_for(endpoint: str) -> re.Pattern[str]:
    return re.compile(
        rf'^blackwellm:(?P<name>[a-z0-9_]+)\{{model_name="[^"]*",'
        rf'endpoint="{endpoint}"\}} (?P<value>\S+)$'
    )


async def scrape_metrics(session: aiohttp.ClientSession, base_url: str, endpoint: str) -> dict:
    out = {k: 0.0 for k in METRIC_KEYS}
    try:
        async with session.get(f"{base_url}/metrics") as resp:
            text = await resp.text()
        pattern = _metric_re_for(endpoint)
        for line in text.splitlines():
            m = pattern.match(line.strip())
            if m and m.group("name") in out:
                out[m.group("name")] = float(m.group("value"))
    except Exception:  # noqa: BLE001
        pass
    return out


def diff_metrics(before: dict, after: dict) -> dict:
    return {k: round(float(after.get(k, 0.0)) - float(before.get(k, 0.0)), 6) for k in before}


async def get_stats(session: aiohttp.ClientSession, base_url: str) -> dict:
    try:
        async with session.get(f"{base_url}/debug/stats") as resp:
            return snapshot_stats(await resp.json())
    except Exception:  # noqa: BLE001
        return {k: 0 for k in COUNTER_KEYS}


async def stream_one(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    endpoint: str = "chat",
) -> dict:
    """Send one request; record event timestamps only.

    ``endpoint="chat"`` uses streaming chat/completions (SSE).
    ``endpoint="completions"`` uses the legacy text-completions endpoint
    (non-streaming JSON): the raw prompt is tokenized WITHOUT the chat
    template, which is the exact token-array protocol the historical
    native_warm_compare benchmark used, so a warm P+10240 request shares
    its first P tokens bit-for-bit with the cold P request.
    """
    if endpoint == "completions":
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
    t0 = time.perf_counter()
    first_t = None
    last_t = None
    error = None
    try:
        url = (
            f"{base_url}/v1/completions"
            if endpoint == "completions"
            else f"{base_url}/v1/chat/completions"
        )
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = (await resp.text())[:500]
                return {"error": f"http {resp.status}: {body}"}
            if endpoint == "completions":
                body = await resp.json()
                wall = time.perf_counter() - t0
                choice = (body.get("choices") or [{}])[0]
                result = {
                    "error": None,
                    "ttft_s": None,  # non-streaming: server metrics own TTFT
                    "last_event_s": round(wall, 4),
                    "wall_s": round(wall, 4),
                    "usage_prompt": (body.get("usage") or {}).get("prompt_tokens"),
                    "usage_completion": (body.get("usage") or {}).get("completion_tokens"),
                }
                result.update(_completion_evidence(choice.get("text") or ""))
                return result
            completion_parts = []
            async for line in resp.content:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                try:
                    evt = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                choice = (evt.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    completion_parts.append(delta.get("content") or delta.get("reasoning_content"))
                    now = time.perf_counter()
                    if first_t is None:
                        first_t = now
                    last_t = now
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - t0
    result = {
        "error": error,
        "ttft_s": round(first_t - t0, 4) if first_t is not None else None,
        "last_event_s": round(last_t - t0, 4) if last_t is not None else None,
        "wall_s": round(wall, 4),
    }
    result.update(_completion_evidence("".join(completion_parts)))
    return result


def wave_summary(
    reqs: list[dict],
    wall: float,
    stats_delta: dict,
    metric_delta: dict,
) -> dict:
    errors = [r for r in reqs if r.get("error")]
    ok = [r for r in reqs if not r.get("error")]
    n = stats_delta.get("requests_completed", 0)
    prompt_total = int(metric_delta.get("prompt_tokens_total", 0.0))
    gen_total = int(metric_delta.get("generation_tokens_total", 0.0))
    ttft_count = int(metric_delta.get("time_to_first_token_seconds_count", 0.0))
    ttft_sum = metric_delta.get("time_to_first_token_seconds_sum", 0.0)
    tpot_count = int(metric_delta.get("request_time_per_output_token_seconds_count", 0.0))
    tpot_sum = metric_delta.get("request_time_per_output_token_seconds_sum", 0.0)
    e2e_sum = metric_delta.get("e2e_request_latency_seconds_sum", 0.0)

    ttfts = sorted(r["ttft_s"] for r in ok if r["ttft_s"] is not None)
    starts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    ends = [r["last_event_s"] for r in ok if r["last_event_s"] is not None]
    agg_decode_window = None
    if starts and ends and max(ends) > min(starts):
        agg_decode_window = round(max(ends) - min(starts), 4)

    return {
        "wall_s": round(wall, 4),
        "requests_sent": len(reqs),
        "requests_ok": len(ok),
        "requests_completed_delta": n,
        "errors": errors,
        # Keep the exact completion evidence beside the counters that
        # describe it.  Without this, a cold/warm acceptance change cannot
        # be distinguished from the model taking a different greedy path.
        "requests": reqs,
        "prompt_tokens_total": prompt_total,
        "generation_tokens_total": gen_total,
        "expected_prompt_tokens": None,  # filled by caller
        "ttft_s_per_request": [round(x, 4) for x in ttfts],
        "ttft_median_s": round(ttfts[len(ttfts) // 2], 4) if ttfts else None,
        "server_mean_ttft_s": round(ttft_sum / ttft_count, 4) if ttft_count else None,
        "server_mean_decode_tok_per_s": round(1.0 / (tpot_sum / tpot_count), 2)
        if tpot_count and tpot_sum > 0
        else None,
        "server_e2e_tok_per_s": round(gen_total / e2e_sum, 2) if e2e_sum > 0 else None,
        "aggregate_decode_window_s": agg_decode_window,
        "aggregate_decode_tok_per_s": round(gen_total / agg_decode_window, 2)
        if agg_decode_window
        else None,
        "aggregate_e2e_tok_per_s": round(gen_total / wall, 2) if wall > 0 else None,
        "stats_delta": stats_delta,
        "metric_delta": metric_delta,
    }


async def run_wave(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    expected_prompt_tokens: int,
    endpoint: str = "chat",
) -> dict:
    before_stats = await get_stats(session, base_url)
    before_metrics = await scrape_metrics(session, base_url, endpoint)
    t0 = time.perf_counter()
    reqs = await asyncio.gather(
        *[
            stream_one(session, base_url, model, prompt, max_tokens, endpoint)
            for _ in range(concurrency)
        ]
    )
    wall = time.perf_counter() - t0
    after_stats = await get_stats(session, base_url)
    after_metrics = await scrape_metrics(session, base_url, endpoint)
    summary = wave_summary(
        reqs,
        wall,
        diff_stats(before_stats, after_stats),
        diff_metrics(before_metrics, after_metrics),
    )
    summary["expected_prompt_tokens"] = expected_prompt_tokens
    return summary


def load_fixture(name: str) -> dict:
    """Load a historical token-id fixture (benchmarks/fixtures/<name>_prompts.json).

    These fixtures are the EXACT workloads behind the July 2026 128K/c4
    headline numbers: per-request arithmetic ramps over non-special token
    ids (vLLM RandomDataset formula). Their text round-trip is not
    identity-preserving, so they must be served as token-id lists through
    /v1/completions, never as decoded text.
    """
    path = Path(__file__).resolve().parent / "fixtures" / f"{name}_prompts.json"
    data = json.loads(path.read_text())
    ids = data["prompt_token_ids"]
    assert len(ids) == data["num_requests"], (len(ids), data["num_requests"])
    assert all(len(x) == data["prompt_len"] for x in ids)
    return data


async def run_fixture_wave(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt_ids_list: list,
    max_tokens: int,
) -> dict:
    """One wave: every fixture prompt sent once, concurrently.

    The historical protocol ran exactly num_requests distinct prompts (one
    per slot), so this wave sends each fixture prompt exactly once instead
    of repeating one prompt c times.
    """
    before_stats = await get_stats(session, base_url)
    before_metrics = await scrape_metrics(session, base_url, "completions")
    t0 = time.perf_counter()
    reqs = await asyncio.gather(
        *[
            stream_one(session, base_url, model, ids, max_tokens, "completions")
            for ids in prompt_ids_list
        ]
    )
    wall = time.perf_counter() - t0
    after_stats = await get_stats(session, base_url)
    after_metrics = await scrape_metrics(session, base_url, "completions")
    summary = wave_summary(
        reqs,
        wall,
        diff_stats(before_stats, after_stats),
        diff_metrics(before_metrics, after_metrics),
    )
    summary["expected_prompt_tokens"] = sum(len(x) for x in prompt_ids_list)
    return summary


async def run_fixture_cell(args, session, results) -> None:
    data = load_fixture(args.fixture)
    ids_list = data["prompt_token_ids"]
    if args.fixture_prompts:
        idx = [int(x) for x in args.fixture_prompts.split(",")]
        ids_list = [ids_list[i] for i in idx]
    prompt_len = data["prompt_len"]
    n_req = len(ids_list)
    print(
        f"fixture {args.fixture}: {n_req} prompts x {prompt_len} tokens "
        f"(generation_formula: {data['generation_formula'][:80]}...)"
    )
    cell = {
        "context_tokens": prompt_len,
        "served_context_tokens": prompt_len,
        "concurrency": n_req,
        "fixture": args.fixture,
        "generation_formula": data["generation_formula"],
        "seed": data["seed"],
    }
    print(f"=== COLD wave ({n_req} x {prompt_len}) ===")
    cold = await run_fixture_wave(session, args.base_url, args.model, ids_list, args.max_tokens)
    cell["cold"] = cold
    _print_wave("COLD", cold, prompt_len * n_req)
    warms = []
    for i in range(args.warm_rounds):
        w = await run_fixture_wave(session, args.base_url, args.model, ids_list, args.max_tokens)
        warms.append(w)
        _print_wave(f"WARM{i + 1}", w, prompt_len * n_req)
    cell["warm"] = warms
    results["cells"][str(prompt_len)] = {str(n_req): cell}


def _print_wave(kind: str, w: dict, expected_prompt: int) -> None:
    print(
        f"  {kind}: wall={w['wall_s']:.4f}s prompt={w['prompt_tokens_total']}/{expected_prompt} "
        f"gen={w['generation_tokens_total']} mean_ttft={w['server_mean_ttft_s']}s "
        f"mean_decode={w['server_mean_decode_tok_per_s']} tok/s "
        f"agg_e2e={w['aggregate_e2e_tok_per_s']} tok/s "
        f"hits={w['stats_delta'].get('prefix_cache_hits', 0)} "
        f"restores={w['stats_delta'].get('prefix_persistent_restores', 0)}"
    )


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8300")
    p.add_argument("--model", default="qwen3.6")
    p.add_argument(
        "--tokenizer-path",
        default=None,
        help="local tokenizer/checkpoint path; defaults to the standard Qwen3.6 checkpoint",
    )
    p.add_argument(
        "--server-label",
        default="live server; launch parameters recorded separately",
        help="exact server configuration written into the result artifact",
    )
    p.add_argument(
        "--endpoint",
        choices=["chat", "completions"],
        default="chat",
        help="chat/completions (SSE, chat template) or the legacy "
        "text-completions endpoint (raw tokenization, no "
        "template -- the historical token-array protocol)",
    )
    p.add_argument("--contexts", default="4k,32k,64k,128k,250k")
    p.add_argument("--concurrency", default="1,2,3")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument(
        "--warm-rounds",
        type=int,
        default=1,
        help="WARM waves per cell after the COLD populate wave",
    )
    p.add_argument(
        "--warm-suffix-tokens",
        type=int,
        default=0,
        help="append this many FRESH tokens to the prompt on WARM "
        "waves (historical Pattern-B protocol: cached prefix "
        "+ 10240-token new suffix, only the suffix is "
        "re-prefilled)",
    )
    p.add_argument(
        "--filler-prefix",
        default=FILLER,
        help="filler text for the prefix prompt (must tokenize "
        "1:1; different fillers isolate cache content)",
    )
    p.add_argument(
        "--filler-suffix",
        default=SUFFIX_FILLER,
        help="filler text for the warm suffix (must tokenize 1:1 "
        "and differ from the prefix filler)",
    )
    p.add_argument(
        "--fixture",
        default=None,
        help="historical token-id fixture name (ctx64k/ctx128k): "
        "serve the exact fixture prompts via token-id completions "
        "instead of generated filler text",
    )
    p.add_argument(
        "--fixture-prompts",
        default=None,
        help="comma-separated fixture prompt indices to serve, e.g. "
        "'0,0,1,1' serves four requests built from two distinct "
        "fixture prompts (the persistent arena holds two 128K "
        "entries, so four DISTINCT prompts cannot all stay warm)",
    )
    p.add_argument("--out", default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help="load a previous partial run from --out (or the resume file) and skip completed cells",
    )
    p.add_argument(
        "--smoke", action="store_true", help="single 4K x c=1 cell, for validating the harness"
    )
    args = p.parse_args()

    context_labels = [x.strip().lower() for x in args.contexts.split(",")]
    contexts = [parse_ctx(x) for x in context_labels]
    concurrencies = [int(x) for x in args.concurrency.split(",")]
    if args.smoke:
        context_labels, contexts, concurrencies = (
            context_labels[:1],
            contexts[:1],
            concurrencies[:1],
        )

    tokenizer_path = args.tokenizer_path or standard_checkpoint_path()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    results = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "endpoint": args.endpoint,
            "context_labels": context_labels,
            "context_targets": contexts,
            "concurrency": concurrencies,
            "max_tokens": args.max_tokens,
            "warm_rounds": args.warm_rounds,
            "warm_suffix_tokens": args.warm_suffix_tokens,
            "filler_prefix": args.filler_prefix,
            "filler_suffix": args.filler_suffix,
            "tokenizer": tokenizer_path,
            "server": args.server_label,
        },
        "cells": {},
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if args.fixture:
            if args.endpoint != "completions":
                raise SystemExit("--fixture requires --endpoint completions (token-id protocol)")
            results["config"]["fixture"] = args.fixture
            await run_fixture_cell(args, session, results)
            out_path = (
                Path(args.out)
                if args.out
                else Path(
                    f"benchmarks/fixtures/server_perf_grid_fixture_{args.fixture}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
            )
            results["finished_at"] = datetime.now(timezone.utc).isoformat()
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            print(f"\nresults written to {out_path}")
            return
        # Block-align the SERVED context (raw + template overhead) down to
        # the largest reachable value <= target, then find the exact raw
        # length that produces it.
        prompts = {}
        served_ctx = {}
        suffix_text = make_suffix(tokenizer, args.warm_suffix_tokens, args.filler_suffix)
        for label, target in zip(context_labels, contexts):
            if args.endpoint == "completions":
                # Raw text-completions protocol: tokenized WITHOUT the chat
                # template, so the served length IS the target (no template
                # overhead, no end-of-message token merge). This mirrors the
                # historical token-array fixtures exactly.
                prompts[target] = make_prompt(tokenizer, target, args.filler_prefix)
                served_ctx[target] = target
                raw_ids = tokenizer.encode(prompts[target], add_special_tokens=True)
                assert len(raw_ids) == target, (label, len(raw_ids), target)
            else:
                raw_n, served = find_prompt_for_target(tokenizer, target, args.filler_prefix)
                prompts[target] = make_prompt(tokenizer, raw_n, args.filler_prefix)
                served_ctx[target] = served
            if args.warm_suffix_tokens:
                if args.endpoint == "completions":
                    pfx_ids = tokenizer.encode(prompts[target], add_special_tokens=True)
                    warm_ids = tokenizer.encode(
                        prompts[target] + suffix_text, add_special_tokens=True
                    )
                    assert len(warm_ids) == len(pfx_ids) + args.warm_suffix_tokens
                    # The warm request must share its first P tokens with the
                    # cold request BIT-FOR-BIT, or the persistent cache cannot
                    # hit (this is exactly what the historical token-array
                    # protocol guaranteed by construction).
                    assert warm_ids[: len(pfx_ids)] == pfx_ids, label
                else:
                    warm_served = chat_token_count(tokenizer, prompts[target] + suffix_text)
                    assert warm_served == served_ctx[target] + args.warm_suffix_tokens, (
                        f"{label}: warm served {warm_served} != "
                        f"prefix {served_ctx[target]} + "
                        f"suffix {args.warm_suffix_tokens}"
                    )
        results["config"]["template_overhead_tokens"] = {
            label: (
                0
                if args.endpoint == "completions"
                else served_ctx[target]
                - len(tokenizer.encode(prompts[target], add_special_tokens=False))
            )
            for label, target in zip(context_labels, contexts)
        }
        results["config"]["served_context_tokens"] = {
            label: served_ctx[target] for label, target in zip(context_labels, contexts)
        }
        print(
            f"template_overhead={results['config']['template_overhead_tokens']}; "
            f"served (block-aligned) contexts: "
            f"{ {l: served_ctx[t] for l, t in zip(context_labels, contexts)} }",
            flush=True,
        )
        # Resumability: load a previous partial run and skip completed cells.
        out_path = Path(args.out) if args.out else None
        resume_path = out_path or (
            _REPO_ROOT / "benchmarks" / "fixtures" / "server_perf_grid_resume.json"
        )
        if args.resume and resume_path.exists():
            old = json.loads(resume_path.read_text())
            old_cells = old.get("cells", {})
            for ctx in contexts:
                results["cells"].setdefault(ctx, {}).update(old_cells.get(str(ctx), {}))
            print(f"resumed {len(results['cells'])} context cells from {resume_path}", flush=True)
        results["server_stats_before"] = await get_stats(session, args.base_url)
        for label, ctx in zip(context_labels, contexts):
            results["cells"].setdefault(ctx, {})
            for c in concurrencies:
                if str(c) in results["cells"][ctx]:
                    print(f"skip completed cell {label} x c={c}", flush=True)
                    continue
                print(
                    f"\n=== cell {label} (served={served_ctx[ctx]}) "
                    f"x c={c} (max_tokens={args.max_tokens}) ===",
                    flush=True,
                )
                expected = served_ctx[ctx] * c
                cold = await run_wave(
                    session,
                    args.base_url,
                    args.model,
                    prompts[ctx],
                    args.max_tokens,
                    c,
                    expected,
                    args.endpoint,
                )
                print(
                    f"  COLD : wall={cold['wall_s']}s "
                    f"prompt={cold['prompt_tokens_total']}/{expected} "
                    f"gen={cold['generation_tokens_total']} "
                    f"mean_ttft={cold['server_mean_ttft_s']}s "
                    f"mean_decode={cold['server_mean_decode_tok_per_s']} tok/s "
                    f"agg_e2e={cold['aggregate_e2e_tok_per_s']} tok/s "
                    f"hits={cold['stats_delta']['prefix_cache_hits']} "
                    f"restores={cold['stats_delta']['prefix_persistent_restores']}",
                    flush=True,
                )
                warms = []
                for _ in range(args.warm_rounds):
                    warm = await run_wave(
                        session,
                        args.base_url,
                        args.model,
                        prompts[ctx] + suffix_text,
                        args.max_tokens,
                        c,
                        (served_ctx[ctx] + args.warm_suffix_tokens) * c,
                        args.endpoint,
                    )
                    warms.append(warm)
                    print(
                        f"  WARM : wall={warm['wall_s']}s "
                        f"prompt={warm['prompt_tokens_total']}/{expected} "
                        f"gen={warm['generation_tokens_total']} "
                        f"mean_ttft={warm['server_mean_ttft_s']}s "
                        f"mean_decode={warm['server_mean_decode_tok_per_s']} tok/s "
                        f"agg_e2e={warm['aggregate_e2e_tok_per_s']} tok/s "
                        f"hits={warm['stats_delta']['prefix_cache_hits']} "
                        f"restores={warm['stats_delta']['prefix_persistent_restores']}",
                        flush=True,
                    )
                cell = {
                    "context_tokens": ctx,
                    "served_context_tokens": served_ctx[ctx],
                    "concurrency": c,
                    "cold": cold,
                    "warm": warms,
                }
                if warms:
                    w = warms[-1]
                    cell["warm_prefix_hit"] = (
                        w["stats_delta"].get("prefix_cache_hits", 0) > 0
                        or w["stats_delta"].get("prefix_persistent_restores", 0) > 0
                    )
                results["cells"][ctx][c] = cell
                # Persist after every cell so an interrupted run resumes
                # from exactly where it stopped.
                if out_path is not None:
                    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
                else:
                    resume_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    results["server_stats_after"] = await get_stats(session, args.base_url)
    results["finished_at"] = datetime.now(timezone.utc).isoformat()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        Path(args.out)
        if args.out
        else (_REPO_ROOT / "benchmarks" / "fixtures" / f"server_perf_grid_{ts}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nresults written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
