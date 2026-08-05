#!/usr/bin/env python3
"""MMLU-Pro evaluation -- faithful to the standard published methodology so the
score is comparable to the official Qwen3.6-27B number (MMLU-Pro = 86.2).

Methodology (matches the original MMLU-Pro paper / lm-eval `mmlu_pro` task):
  * Dataset: TIGER-Lab/MMLU-Pro, full 12,032-question test split.
  * 5-shot chain-of-thought, category-matched exemplars from the validation
    split (exactly 5 per category), using the dataset's `cot_content`.
  * Greedy decoding (temperature 0).
  * Score = accuracy of the extracted answer letter vs the gold `answer`.
  * Answer extraction: last "answer is (X)" in the model output (fallback to
    reasoning_content, then last standalone "(X)").

The model is queried through an OpenAI-compatible chat endpoint, so this runs
against our custom runtime unchanged.

Usage:
  python benchmarks/official/mmlu_pro_eval.py --base-url http://localhost:8000/v1 \
      --model qwen3.6 --out evalplus_results/official/mmlu_pro_our_runtime.json
  # quick estimate on a stratified subset:
  ... --limit 700 --concurrency 8
"""
import argparse
import concurrent.futures as cf
import datetime
import glob
import json
import os
import re
import time
import urllib.request

LETTERS = "ABCDEFGHIJ"
ANS_RE = re.compile(r"answer\s*is\s*\(?([A-J])\)?", re.I)
PAREN_RE = re.compile(r"\(([A-J])\)")


def chat(base_url, model, prompt, max_tokens, temperature=0.0, timeout=3600,
         chat_template_kwargs=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature, "n": 1,
    }
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs
    payload = json.dumps(body).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"chat failed: {last}")


def clean_cot(cot):
    s = cot.strip()
    if s.startswith("A:"):
        s = s[2:].strip()
    idx = s.find("step by step.")
    if idx >= 0:
        s = s[idx + len("step by step."):].strip()
    return s


def format_options(options):
    return "\n".join(f"({LETTERS[i]}) {opt}" for i, opt in enumerate(options))


def build_prompt(question, options, category, shots):
    header = (f"The following are multiple choice questions (with answers) about "
              f"{category}.\n\n")
    blocks = []
    for s in shots:
        blocks.append(
            f"Question:\n{s['question']}\nOptions:\n{format_options(s['options'])}\n"
            f"Answer: Let's think step by step.\n{clean_cot(s['cot_content'])}\n")
    test_block = (f"Question:\n{question}\nOptions:\n{format_options(options)}\n"
                  f"Answer: Let's think step by step.\n")
    return header + "\n".join(blocks) + "\n" + test_block


def extract_answer(content, reasoning):
    for text in (content or "", reasoning or ""):
        m = list(ANS_RE.finditer(text))
        if m:
            return m[-1].group(1).upper()
    for text in (content or "", reasoning or ""):
        m = list(PAREN_RE.finditer(text))
        if m:
            return m[-1].group(1).upper()
    return None


def build_report(records, model, base_url, limit, max_tokens, thinking):
    n = len(records)
    overall = sum(r["correct"] for r in records) / n if n else 0.0
    per_cat = {}
    cats = sorted({r["category"] for r in records})
    for c in cats:
        sub = [r for r in records if r["category"] == c]
        per_cat[c] = {"acc": round(sum(r["correct"] for r in sub) / len(sub), 4),
                      "n": len(sub)}
    trunc = sum(1 for r in records if r["finish_reason"] == "length")
    noans = sum(1 for r in records if r["pred"] is None)
    return {
        "benchmark": "MMLU-Pro", "official_qwen36_27b": 86.2,
        "model": model, "base_url": base_url,
        "date": datetime.datetime.now().isoformat(),
        "n": n, "subset": "full" if not limit else f"stratified-{n}",
        "methodology": "5-shot CoT, category-matched, greedy, last 'answer is (X)'",
        "thinking": "off" if not thinking else "on",
        "max_tokens": max_tokens,
        "accuracy": round(overall * 100, 2),
        "per_category": per_cat,
        "truncated": trunc, "no_answer_extracted": noans,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0,
                   help="0 = full 12032 test set; else stratified subset size")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--no-thinking", action="store_true",
                   help="send chat_template_kwargs={enable_thinking:False} (needs server fix)")
    p.add_argument("--checkpoint", default="",
                   help="jsonl path for incremental results (auto if empty)")
    p.add_argument("--shards", type=int, default=1,
                   help="run as shard N of --shards parallel workers")
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--merge", action="store_true",
                   help="merge <out>.shard*.jsonl into the final report (no eval)")
    args = p.parse_args()

    if args.merge:
        shard_glob = args.out.replace(".json", ".shard*.jsonl")
        paths = sorted(glob.glob(shard_glob))
        if not paths:
            print(f"merge: no shard checkpoints found for {shard_glob}", flush=True)
            raise SystemExit(2)
        records = []
        for path in paths:
            with open(path) as f:
                records.extend(json.loads(line) for line in f if line.strip())
        print(f"merge: loaded {len(records)} records from {len(paths)} shard(s)",
              flush=True)
        report = build_report(records, args.model, args.base_url, args.limit,
                              args.max_tokens, not args.no_thinking)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n=== MMLU-Pro accuracy = {report['accuracy']} "
              f"(official Qwen3.6-27B = 86.2) ===", flush=True)
        print(f"truncated={report['truncated']} "
              f"no_answer={report['no_answer_extracted']}", flush=True)
        print(f"Saved -> {args.out}", flush=True)
        return

    from datasets import load_dataset
    print("Loading TIGER-Lab/MMLU-Pro ...", flush=True)
    test = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    val = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")

    by_cat = {}
    for ex in val:
        by_cat.setdefault(ex["category"], []).append(ex)

    items = list(test)
    if args.limit and args.limit < len(items):
        # stratified: take proportional slice per category for a fair estimate
        from collections import defaultdict
        per = defaultdict(list)
        for ex in items:
            per[ex["category"]].append(ex)
        # Largest-remainder apportionment: floor quotas first, then give the
        # remaining seats to the categories with the largest fractional parts.
        # This reproduces the historical 414-question subset exactly (the
        # plain int() truncation only yields 408 and is NOT comparable to the
        # 2026-07-22 baseline: e.g. math 47, physics 45, chemistry 39, ...).
        total = args.limit
        quota = {cat: len(exs) * total / len(items) for cat, exs in per.items()}
        alloc = {cat: int(q) for cat, q in quota.items()}
        remain = total - sum(alloc.values())
        frac_order = sorted(
            quota, key=lambda c: (quota[c] - int(quota[c]), len(per[c])),
            reverse=True)
        for cat in frac_order[:remain]:
            alloc[cat] += 1
        items = []
        for cat, exs in per.items():
            items.extend(exs[:alloc[cat]])
    if args.shards > 1:
        if not (0 <= args.shard_idx < args.shards):
            raise SystemExit(f"--shard-idx {args.shard_idx} out of range "
                             f"0..{args.shards - 1}")
        items = [ex for i, ex in enumerate(items) if i % args.shards == args.shard_idx]
    print(f"Evaluating {len(items)} questions "
          f"({'full set' if not args.limit else 'stratified subset'}), "
          f"concurrency={args.concurrency}, max_tokens={args.max_tokens}, "
          f"shard={args.shard_idx}/{args.shards}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    suffix = f".shard{args.shard_idx}" if args.shards > 1 else ""
    ckpt = args.checkpoint or (args.out.replace(".json", f"{suffix}.jsonl"))
    report_out = args.out.replace(".json", f"{suffix}.json") if args.shards > 1 \
        else args.out
    done = {}
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            for line in f:
                r = json.loads(line)
                done[r["question_id"]] = r
        print(f"Resuming: {len(done)} already done", flush=True)

    ckpt_f = open(ckpt, "a")
    lock_write = [0.0]

    def one(ex):
        qid = ex["question_id"]
        if qid in done:
            return done[qid]
        shots = by_cat.get(ex["category"], [])[:5]
        prompt = build_prompt(ex["question"], ex["options"], ex["category"], shots)
        ctk = {"enable_thinking": False} if args.no_thinking else None
        try:
            resp = chat(args.base_url, args.model, prompt, args.max_tokens,
                        chat_template_kwargs=ctk)
            ch = resp["choices"][0]
            msg = ch["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            finish = ch.get("finish_reason")
            pred = extract_answer(content, reasoning)
        except Exception as e:  # noqa: BLE001
            content = reasoning = ""
            finish = "error"
            pred = None
        rec = {"question_id": qid, "category": ex["category"],
               "gold": ex["answer"], "pred": pred,
               "correct": pred == ex["answer"], "finish_reason": finish,
               "gen_tokens": (resp.get("usage", {}) or {}).get("completion_tokens")
               if 'resp' in dir() else None}
        ckpt_f.write(json.dumps(rec) + "\n")
        ckpt_f.flush()
        return rec

    t0 = time.time()
    results = []
    finished = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one, e) for e in items]
        for fut in cf.as_completed(futs):
            results.append(fut.result())
            finished += 1
            if finished % 50 == 0 or finished == len(items):
                acc = sum(r["correct"] for r in results) / len(results)
                rate = finished / max(1e-6, time.time() - t0)
                eta = (len(items) - finished) / rate if rate else 0
                print(f"  {finished}/{len(items)} acc={acc:.4f} "
                      f"({rate:.2f} q/s, ETA {eta/60:.1f}m)", flush=True)
    ckpt_f.close()

    report = build_report(results, args.model, args.base_url, args.limit,
                          args.max_tokens, not args.no_thinking)
    os.makedirs(os.path.dirname(report_out) or ".", exist_ok=True)
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n=== MMLU-Pro accuracy = {report['accuracy']} "
          f"(official Qwen3.6-27B = 86.2) ===", flush=True)
    print(f"truncated={report['truncated']} "
          f"no_answer={report['no_answer_extracted']}", flush=True)
    print(f"Saved -> {report_out}", flush=True)


if __name__ == "__main__":
    main()
