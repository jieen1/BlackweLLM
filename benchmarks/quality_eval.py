#!/usr/bin/env python3
"""Parallel HumanEval+ quality evaluation via OpenAI-compatible API.

Historical harness (restored from the 2026-07-21 quality run; deleted in the
T0-7 cleanup commit dfd37bf). It sends all 164 HumanEval+ prompts to a server
with greedy decoding, saves results in evalplus jsonl format for evaluation.

The README-recorded numbers for our runtime (44.5%/43.3%, 2026-07-21) were
produced with max_tokens=768 and concurrency=16 -- those remain the defaults
here so a rerun is comparable.

Comparability note (2026-07-21 vs 2026-08-05):
* 07-21 our-runtime server returned the FULL generation (thinking + answer)
  as ``message.content``, and the original harness sanitized that raw text
  directly (see evalplus_results/humaneval/our_runtime.raw.jsonl, median
  ~2380 chars of reasoning text).
* The current server splits the same raw generation into
  ``message.reasoning_content`` (thinking) + ``message.content`` (final
  answer). To keep the README row comparable we reconstruct
  ``raw = reasoning_content + content`` before sanitizing, exactly as the
  07-21 harness saw it. Generation parameters are unchanged.

Enhancements over the deleted original:
  * resume: tasks already present in the sanitized output file are skipped,
    so an interrupted run continues instead of regenerating everything;
  * --max-tokens / --concurrency are CLI-tunable (defaults preserved);
  * optional --evaluate runs `python -m evalplus.evaluate` and prints pass@1.

Usage:
  python benchmarks/quality_eval.py --base-url http://localhost:8300/v1 \
      --model qwen3.6 --output evalplus_results/humaneval/our_runtime.jsonl
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import aiohttp
from evalplus.data import get_human_eval_plus
from evalplus.sanitize import sanitize

INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
MAX_TOKENS = 768
CONCURRENCY = 16


def load_done(output_path):
    """Load task_id -> solution from the raw output jsonl (resume source)."""
    done = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["task_id"]] = rec.get("solution")
    return done


async def generate_one(session, base_url, model, task_id, prompt, semaphore,
                       max_tokens):
    message = INSTRUCTION_PREFIX + f"\n```python\n{prompt.strip()}\n```"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "n": 1,
    }
    async with semaphore:
        for attempt in range(3):
            try:
                async with session.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=3600),
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        print(f"  [{task_id}] HTTP {resp.status}: {text[:200]}",
                              flush=True)
                        await asyncio.sleep(2)
                        continue
                    data = await resp.json()
                    msg = data["choices"][0]["message"]
                    content = msg.get("content") or ""
                    reasoning = msg.get("reasoning_content") or ""
                    # Reconstruct the raw generation exactly as the 07-21
                    # server exposed it in ``content`` (see module docstring).
                    return task_id, reasoning + content
            except Exception as e:  # noqa: BLE001 - retry transient errors
                print(f"  [{task_id}] attempt {attempt + 1} error: {e}", flush=True)
                await asyncio.sleep(2)
    return task_id, None


async def run_all(base_url, model, dataset, output_path, concurrency,
                  max_tokens):
    raw_path = output_path.replace(".jsonl", ".raw.jsonl")
    done = load_done(raw_path)
    by_id = {task_id: task for task_id, task in dataset.items()}
    if os.path.exists(raw_path) and done:
        # Resume consistency: raw is the write-order authority (san is written
        # immediately before raw and both are flushed together, but a crash
        # between the writes -- or an earlier duplicate-append -- can leave
        # san inconsistent). Rebuilding san from raw on startup makes resume
        # idempotent and guarantees the eval input exactly mirrors the raw
        # audit file. Sanitizing ~160 short records is cheap.
        rebuilt = [
            json.dumps({"task_id": tid, "solution": sanitize(
                sol, entrypoint=by_id[tid]["entry_point"])})
            for tid, sol in done.items()
        ]
        with open(output_path, "w") as f:
            f.write("\n".join(rebuilt) + ("\n" if rebuilt else ""))
    tasks = []
    for task_id, task in dataset.items():
        if task_id in done:
            continue
        prompt = task["prompt"].strip() + "\n"
        tasks.append((task_id, task, prompt))
    print(f"Resuming: {len(done)}/{len(dataset)} already done; "
          f"generating {len(tasks)}", flush=True)

    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    results = dict(done)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "a") as f_san, open(raw_path, "a") as f_raw:
        async with aiohttp.ClientSession(connector=connector) as session:
            futures = [
                generate_one(session, base_url, model, task_id, prompt, semaphore,
                             max_tokens)
                for task_id, _task, prompt in tasks
            ]
            done_count = 0
            total = len(futures)
            start = time.time()
            for coro in asyncio.as_completed(futures):
                task_id, content = await coro
                done_count += 1
                if content is not None:
                    results[task_id] = content
                    sanitized = sanitize(
                        content, entrypoint=by_id[task_id]["entry_point"])
                    f_san.write(json.dumps(
                        {"task_id": task_id, "solution": sanitized}) + "\n")
                    f_raw.write(json.dumps(
                        {"task_id": task_id, "solution": content}) + "\n")
                    f_san.flush()
                    f_raw.flush()
                    if done_count % 10 == 0 or done_count == total:
                        elapsed = max(1e-6, time.time() - start)
                        rate = done_count / elapsed
                        eta = (total - done_count) / rate if rate > 0 else 0
                        print(f"  Progress: {done_count}/{total} "
                              f"({rate:.1f} problems/s, ETA {eta:.0f}s)",
                              flush=True)
                else:
                    print(f"  FAILED: {task_id}", flush=True)

    print(f"\nSaved {len(results)}/{len(dataset)} solutions to {output_path}")
    return len(results)


def evaluate(output_path):
    subprocess.run(
        [sys.executable, "-m", "evalplus.evaluate",
         "--dataset", "humaneval", "--samples", output_path],
        check=True, capture_output=True, text=True)
    # evalplus strips the .jsonl suffix before appending _eval_results.json.
    stem = output_path[:-len(".jsonl")] if output_path.endswith(".jsonl") else output_path
    results = stem + "_eval_results.json"
    d = json.load(open(results))
    ev = d.get("eval", {})
    nb = np_ = pb = pp = 0
    for reslist in ev.values():
        for r in reslist:
            nb += 1
            np_ += 1
            if r.get("base_status") == "pass":
                pb += 1
            if r.get("plus_status") == "pass":
                pp += 1
    base = pb / nb if nb else 0.0
    plus = pp / np_ if np_ else 0.0
    print(f"HumanEval pass@1={base:.3f} ({pb}/{nb}) "
          f"HumanEval+ pass@1={plus:.3f} ({pp}/{np_})")
    print(f"eval results -> {results}")
    return base, plus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--evaluate", action="store_true",
                        help="run evalplus.evaluate on the generated samples")
    args = parser.parse_args()

    print("Loading HumanEval+ dataset...")
    dataset = get_human_eval_plus()
    print(f"  {len(dataset)} problems loaded")
    print(f"Target: {args.base_url} model={args.model}")
    print(f"Concurrency: {args.concurrency} max_tokens={args.max_tokens}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    asyncio.run(run_all(args.base_url, args.model, dataset, args.output,
                        args.concurrency, args.max_tokens))
    if args.evaluate:
        evaluate(args.output)


if __name__ == "__main__":
    main()
