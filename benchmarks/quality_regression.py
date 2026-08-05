#!/usr/bin/env python3
"""Self-contained inference-quality regression harness.

Covers the four dimensions we care about, each mapped to an official Qwen3.6
capability (see notes/2026-07-22-quality-baseline-and-official-scores.md):

  code     -> evalplus HumanEval+ pass@1          (proxy: LiveCodeBench / SWE-bench)
  tool     -> BFCL-lite tool-call accuracy         (proxy: tool-calling / BFCL)
  agent    -> multi-turn tool-use loop accuracy    (proxy: tau2-bench / Claw-Eval)
  longctx  -> Needle-in-a-Haystack retrieval       (proxy: 262K native context)

Runs against any OpenAI-compatible server, targeting our custom runtime.
No network downloads: tool/agent/longctx sets are built inline; code uses the
already-cached evalplus dataset.

Usage:
  python benchmarks/quality_regression.py --base-url http://localhost:8000/v1 \
      --model qwen3.6 --label our_runtime --out evalplus_results/quality/our_runtime.json
  # fast smoke without slow code generation:
  python benchmarks/quality_regression.py ... --dims tool,agent,longctx
"""
import argparse
import concurrent.futures as cf
import datetime
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULT_MAX_TOKENS_CODE = 4096
DEFAULT_MAX_TOKENS = 2048
REQUEST_TIMEOUT = 3600


def _load_ckpt(path):
    """Load a per-dimension jsonl checkpoint into {item_id: record}."""
    done = {}
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["id"]] = rec
    return done


def _append_ckpt(path, item_id, record):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"id": item_id, **record}) + "\n")


def _strip_ckpt(record):
    return {k: v for k, v in record.items() if k != "id"}


def _ckpt_done(record):
    if record is None or record.get("error"):
        return False
    if "content" in record:
        # A code record whose whole token budget was consumed by thinking
        # arrives as {"content": ""}; treating it as done silently skips the
        # problem on resume, so require a non-empty answer here.
        return bool(record["content"] and str(record["content"]).strip())
    return "ok" in record or "name_ok" in record


def chat(base_url, model, messages, max_tokens=DEFAULT_MAX_TOKENS, temperature=0.0,
         tools=None, tool_choice=None):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": 1,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=data,
        headers={"Content-Type": "application/json"})
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:  # noqa: BLE001 - retry any transient server error
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"chat failed after 3 attempts: {last_err}")


def message_of(resp):
    return resp["choices"][0]["message"]


# --------------------------------------------------------------------------- #
# Dimension: tool calling (BFCL-lite, self-contained)
# --------------------------------------------------------------------------- #
TOOL_SPECS = [
    ("get_weather", {"city": "str"}, "What's the weather in Tokyo?",
     {"city": "Tokyo"}),
    ("get_weather", {"city": "str"}, "Check the weather for New York right now.",
     {"city": "New York"}),
    ("search_web", {"query": "str"}, "Search the web for the latest NVIDIA earnings.",
     {"query": "latest NVIDIA earnings"}),
    ("calculate", {"expression": "str"}, "What is 128 * 46?",
     {"expression": "128 * 46"}),
    ("calculate", {"expression": "str"}, "Compute (15 + 27) / 6 for me.",
     {"expression": "(15 + 27) / 6"}),
    ("send_email", {"to": "str", "subject": "str", "body": "str"},
     "Email alice@example.com with subject 'Meeting' and body 'See you at 3pm'.",
     {"to": "alice@example.com", "subject": "Meeting"}),
    ("book_flight", {"origin": "str", "destination": "str", "date": "str"},
     "Book a flight from SFO to LHR on 2026-08-01.",
     {"origin": "SFO", "destination": "LHR", "date": "2026-08-01"}),
    ("get_stock_price", {"ticker": "str"}, "What's the current price of AAPL stock?",
     {"ticker": "AAPL"}),
    ("convert_currency", {"amount": "num", "from": "str", "to": "str"},
     "Convert 100 USD to EUR.", {"amount": 100, "from": "USD", "to": "EUR"}),
    ("set_reminder", {"time": "str", "message": "str"},
     "Remind me at 9am tomorrow to call the dentist.",
     {"message": "call the dentist"}),
    ("translate", {"text": "str", "target_language": "str"},
     "Translate 'hello world' into French.",
     {"text": "hello world", "target_language": "French"}),
    ("get_calendar_events", {"date": "str"}, "What meetings do I have on 2026-07-25?",
     {"date": "2026-07-25"}),
    ("create_file", {"path": "str", "content": "str"},
     "Create a file at /tmp/notes.txt containing 'buy milk'.",
     {"path": "/tmp/notes.txt"}),
    ("run_sql", {"query": "str"}, "Run a SQL query to count all users.",
     {"query": "SELECT COUNT(*) FROM users"}),
    ("get_user_info", {"user_id": "str"}, "Look up information for user id 4287.",
     {"user_id": "4287"}),
    ("play_music", {"song": "str", "artist": "str"},
     "Play 'Bohemian Rhapsody' by Queen.",
     {"song": "Bohemian Rhapsody", "artist": "Queen"}),
    ("order_food", {"restaurant": "str", "items": "str"},
     "Order a margherita pizza from Luigi's.",
     {"restaurant": "Luigi's"}),
    ("get_directions", {"origin": "str", "destination": "str"},
     "Give me directions from Boston to Providence.",
     {"origin": "Boston", "destination": "Providence"}),
    ("summarize_text", {"text": "str"},
     "Summarize this: 'The quick brown fox jumps over the lazy dog.'",
     {}),
    ("generate_image", {"prompt": "str"},
     "Generate an image of a sunset over the ocean.",
     {"prompt": "sunset over the ocean"}),
]

_TYPE_MAP = {"str": "string", "num": "number"}


def _build_tools():
    tools = []
    seen = set()
    for name, params, _, _ in TOOL_SPECS:
        if name in seen:
            continue
        seen.add(name)
        props = {p: {"type": _TYPE_MAP[t]} for p, t in params.items()}
        required = [p for p, t in params.items() if t == "str"]
        tools.append({"type": "function", "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": props,
                           "required": required},
        }})
    return tools


def _args_match(got_json, expected):
    """Expected is a partial spec: every key in expected must match in got."""
    try:
        got = json.loads(got_json) if isinstance(got_json, str) else got_json
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(got, dict):
        return False
    for k, v in expected.items():
        if k not in got:
            return False
        gv = got[k]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                if float(gv) != float(v):
                    return False
            except (TypeError, ValueError):
                return False
        else:
            if str(v).strip().lower() not in str(gv).strip().lower():
                return False
    return True


def run_tool(base_url, model, concurrency, ckpt_path=None):
    tools = _build_tools()
    items = list(enumerate(TOOL_SPECS))
    done = _load_ckpt(ckpt_path)

    def one(item):
        idx, (name, _params, query, expected) = item
        try:
            resp = chat(base_url, model,
                        [{"role": "user", "content": query}],
                        max_tokens=DEFAULT_MAX_TOKENS, tools=tools)
            msg = message_of(resp)
            tcs = msg.get("tool_calls") or []
            if not tcs:
                return {"idx": idx, "name_ok": False, "args_ok": False,
                        "expected": name, "got": None, "query": query}
            fn = tcs[0]["function"]
            got_name = fn.get("name")
            name_ok = got_name == name
            args_ok = _args_match(fn.get("arguments", "{}"), expected)
            return {"idx": idx, "name_ok": name_ok, "args_ok": args_ok,
                    "expected": name, "got": got_name,
                    "got_args": fn.get("arguments"), "query": query}
        except Exception as e:  # noqa: BLE001
            return {"idx": idx, "name_ok": False, "args_ok": False,
                    "expected": name, "got": None, "error": str(e), "query": query}

    missing = [it for it in items if not _ckpt_done(done.get(it[0]))]
    if done:
        print(f"    [tool] checkpoint: {len(done)}/{len(items)} "
              f"done, generating {len(missing)}", flush=True)

    def _save(r):
        done[r["idx"]] = r
        _append_ckpt(ckpt_path, r["idx"], r)

    _parallel(one, missing, concurrency, on_done=_save) if missing else []
    results = [_strip_ckpt(done[i]) for i, _ in items]
    n = len(results)
    name_acc = sum(r["name_ok"] for r in results) / n
    full_acc = sum(r["name_ok"] and r["args_ok"] for r in results) / n
    failures = [r for r in results if not (r["name_ok"] and r["args_ok"])]
    print(f"  [tool] tool-name accuracy={name_acc:.3f} "
          f"full(name+args) accuracy={full_acc:.3f} ({n} cases)")
    return {"accuracy": round(full_acc, 4), "name_accuracy": round(name_acc, 4),
            "n": n, "failures": failures[:20]}


# --------------------------------------------------------------------------- #
# Dimension: agent (multi-turn tool-use loop, self-contained)
# --------------------------------------------------------------------------- #
def _calc(expr):
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 - arithmetic only
    except Exception:  # noqa: BLE001
        return "0"


AGENT_TOOLS = [{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a math expression and return the numeric result.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]},
    },
}]

AGENT_SCENARIOS = [
    {"q": "A store sells apples at $3 each. If I buy 7 apples and pay with a $50 "
          "bill, how much change do I get? Use the calculator, then answer with just "
          "the number.", "expr_hint": "50 - 3 * 7", "answer": "29"},
    {"q": "What is 144 divided by 12, then multiplied by 5? Use the calculator and "
          "give the final number.", "expr_hint": "144 / 12 * 5", "answer": "60"},
    {"q": "I run 4 miles a day for 6 days. Each mile burns 100 calories. How many "
          "calories total? Use the calculator, answer with the number.",
          "expr_hint": "4 * 6 * 100", "answer": "2400"},
    {"q": "A tank fills at 8 liters per minute. How many liters after 15 minutes? "
          "Use the calculator and reply with the number.",
          "expr_hint": "8 * 15", "answer": "120"},
]


def run_agent(base_url, model, concurrency, ckpt_path=None):
    items = list(enumerate(AGENT_SCENARIOS))
    done = _load_ckpt(ckpt_path)

    def one(item):
        idx, sc = item
        try:
            resp = chat(base_url, model, [{"role": "user", "content": sc["q"]}],
                        max_tokens=DEFAULT_MAX_TOKENS, tools=AGENT_TOOLS)
            msg = message_of(resp)
            tcs = msg.get("tool_calls") or []
            if not tcs:
                # maybe answered directly
                content = (msg.get("content") or "")
                return idx, {"ok": sc["answer"] in re.sub(r"[^\d]", "", content) or
                             sc["answer"] in content, "called_tool": False,
                             "q": sc["q"], "final": content[:120]}
            expr = json.loads(tcs[0]["function"].get("arguments", "{}")).get(
                "expression", sc["expr_hint"])
            result = _calc(expr)
            resp2 = chat(base_url, model, [
                {"role": "user", "content": sc["q"]},
                {"role": "assistant", "content": None, "tool_calls": tcs},
                {"role": "tool", "tool_call_id": tcs[0].get("id", "call_0"),
                 "content": result},
            ], max_tokens=DEFAULT_MAX_TOKENS)
            final = message_of(resp2).get("content") or ""
            ok = sc["answer"] in final.replace(",", "")
            return idx, {"ok": ok, "called_tool": True, "q": sc["q"],
                         "expr": expr, "tool_result": result,
                         "final": final[:160]}
        except Exception as e:  # noqa: BLE001
            return idx, {"ok": False, "called_tool": False, "q": sc["q"],
                         "error": str(e)}

    missing = [it for it in items if not _ckpt_done(done.get(it[0]))]
    if done:
        print(f"    [agent] checkpoint: {len(done)}/{len(items)} "
              f"done, generating {len(missing)}", flush=True)

    def _save(item):
        idx, rec = item
        done[idx] = rec
        _append_ckpt(ckpt_path, idx, rec)

    _parallel(one, missing, concurrency, on_done=_save) if missing else []
    results = [_strip_ckpt(done[i]) for i, _ in items]
    n = len(results)
    acc = sum(r["ok"] for r in results) / n
    tool_rate = sum(r["called_tool"] for r in results) / n
    print(f"  [agent] final-answer accuracy={acc:.3f} "
          f"tool-invocation rate={tool_rate:.3f} ({n} scenarios)")
    return {"accuracy": round(acc, 4), "tool_invocation_rate": round(tool_rate, 4),
            "n": n, "details": results}


# --------------------------------------------------------------------------- #
# Dimension: long context (Needle-in-a-Haystack, self-contained)
# --------------------------------------------------------------------------- #
_HAYSTACK_PARA = (
    "The quarterly report discussed steady growth across all regional markets. "
    "Revenue increased due to higher demand for cloud infrastructure services. "
    "Operating margins improved as the company optimized its supply chain. "
    "Management reaffirmed guidance for the remainder of the fiscal year. "
    "Customer retention remained strong throughout the period under review. ")


def _build_haystack(approx_tokens, needle, depth_frac):
    # This repetitive English filler tokenizes at ~6.4 chars/token (measured),
    # so treat approx_tokens as an approximate REAL token budget.
    target_chars = int(approx_tokens * 6.4)
    repeats = max(1, target_chars // len(_HAYSTACK_PARA))
    hay = _HAYSTACK_PARA * repeats
    pos = int(len(hay) * depth_frac)
    return hay[:pos] + f"\n>>> {needle} <<<\n" + hay[pos:]


def run_longctx(base_url, model, concurrency, ckpt_path=None,
                lengths=(8192, 32768, 65536, 131072), depths=(0.0, 0.5, 0.95)):
    rng = random.Random(1234)
    cases = []
    for length in lengths:
        for depth in depths:
            secret = rng.randint(100000, 999999)
            needle = f"The magic number is {secret}."
            cases.append({"length": length, "depth": depth, "secret": str(secret),
                          "needle": needle})
    items = list(enumerate(cases))
    done = _load_ckpt(ckpt_path)

    def one(item):
        idx, c = item
        try:
            haystack = _build_haystack(c["length"], c["needle"], c["depth"])
            prompt = (
                "Below is a long document. After reading it, answer the question.\n\n"
                f"DOCUMENT START\n{haystack}\nDOCUMENT END\n\n"
                "Question: What is the magic number stated in the document? "
                "Reply with only the number.")
            resp = chat(base_url, model, [{"role": "user", "content": prompt}],
                        max_tokens=2048)
            content = message_of(resp).get("content") or ""
            prompt_tokens = resp.get("usage", {}).get("prompt_tokens", -1)
            ok = c["secret"] in re.sub(r"[^\d]", " ", content)
            return idx, {"length": c["length"], "depth": c["depth"], "ok": ok,
                         "secret": c["secret"], "prompt_tokens": prompt_tokens,
                         "answer": content[:80]}
        except Exception as e:  # noqa: BLE001
            return idx, {"length": c["length"], "depth": c["depth"], "ok": False,
                         "error": str(e)}

    missing = [it for it in items if not _ckpt_done(done.get(it[0]))]
    if done:
        print(f"    [longctx] checkpoint: {len(done)}/{len(items)} "
              f"done, generating {len(missing)}", flush=True)

    def _save(item):
        idx, rec = item
        done[idx] = rec
        _append_ckpt(ckpt_path, idx, rec)

    _parallel(one, missing, min(concurrency, 4), progress_every=1,
              on_done=_save) if missing else []
    results = [_strip_ckpt(done[i]) for i, _ in items]
    n = len(results)
    acc = sum(r["ok"] for r in results) / n
    by_length = {}
    for length in lengths:
        sub = [r for r in results if r["length"] == length]
        by_length[length] = round(sum(r["ok"] for r in sub) / len(sub), 4) if sub else 0.0
    print(f"  [longctx] needle retrieval accuracy={acc:.3f} ({n} cases) "
          f"by_length={by_length}")
    return {"accuracy": round(acc, 4), "n": n, "by_length": by_length,
            "details": [r for r in results if not r["ok"]][:20]}


# --------------------------------------------------------------------------- #
# Dimension: code (evalplus HumanEval+, cached dataset)
# --------------------------------------------------------------------------- #
def run_code(base_url, model, concurrency, max_tokens, workdir, ckpt_path=None):
    from evalplus.data import get_human_eval_plus
    from evalplus.sanitize import sanitize
    dataset = get_human_eval_plus()
    prefix = ("Please provide a self-contained Python script that solves the "
              "following problem in a markdown code block:")
    items = list(dataset.items())
    done = _load_ckpt(ckpt_path)

    def one(kv):
        task_id, task = kv
        prompt = task["prompt"].strip() + "\n"
        msg = prefix + f"\n```python\n{prompt.strip()}\n```"
        try:
            resp = chat(base_url, model, [{"role": "user", "content": msg}],
                        max_tokens=max_tokens)
            m = message_of(resp)
            content = m.get("content") or ""
            reasoning = m.get("reasoning_content") or ""
            # Reconstruct the raw generation (thinking + final answer) so
            # sanitize() sees exactly what the 07-21 HumanEval harness saw:
            # that server returned the full generation as ``content``. The
            # current server splits it into reasoning_content/content; with
            # thinking-mode code tasks the code often lives in the reasoning
            # span while ``content`` is empty (max_tokens hit mid-think).
            return task_id, reasoning + content
        except Exception as e:  # noqa: BLE001
            print(f"    [code] {task_id} error: {e}", flush=True)
            return task_id, None

    missing = [kv for kv in items if not _ckpt_done(done.get(kv[0]))]
    if done:
        print(f"    [code] checkpoint: {len(done)}/{len(items)} "
              f"done, generating {len(missing)}", flush=True)
    t0 = time.time()

    def _save(kv):
        task_id, content = kv
        if content is None:
            return
        done[task_id] = {"content": content}
        _append_ckpt(ckpt_path, task_id, {"content": content})

    _parallel(one, missing, concurrency, progress_every=20,
              on_done=_save) if missing else []
    solutions = {tid: rec.get("content") for tid, rec in done.items()
                 if rec.get("content") is not None}
    os.makedirs(workdir, exist_ok=True)
    samples = os.path.join(workdir, "humaneval_plus_samples.jsonl")
    with open(samples, "w") as f:
        for task_id, task in dataset.items():
            if task_id not in solutions:
                continue
            san = sanitize(solutions[task_id], entrypoint=task["entry_point"])
            f.write(json.dumps({"task_id": task_id, "solution": san}) + "\n")
    print(f"  [code] generated {len(solutions)}/{len(dataset)} in "
          f"{time.time()-t0:.0f}s; evaluating pass@1 ...", flush=True)
    if len(solutions) < len(dataset):
        print(f"  [code] INCOMPLETE ({len(solutions)}/{len(dataset)}) -- "
              f"evaluation deferred; rerun the same command to resume "
              f"(checkpoint {ckpt_path})", flush=True)
        raise SystemExit(2)
    eval_json = _evalplus_evaluate(samples)
    base_acc, plus_acc = _parse_evalplus(eval_json)
    print(f"  [code] HumanEval pass@1={base_acc:.3f} HumanEval+ pass@1={plus_acc:.3f}")
    return {"humaneval_pass_at_1": round(base_acc, 4),
            "humaneval_plus_pass_at_1": round(plus_acc, 4),
            "n": len(dataset), "generated": len(solutions),
            "eval_results": eval_json}


def _evalplus_evaluate(samples_path):
    subprocess.run(
        [sys.executable, "-m", "evalplus.evaluate",
         "--dataset", "humaneval", "--samples", samples_path],
        check=True, capture_output=True, text=True)
    # evalplus strips the .jsonl suffix before appending _eval_results.json,
    # so "x.jsonl" becomes "x_eval_results.json".
    stem = samples_path[:-len(".jsonl")] if samples_path.endswith(".jsonl") else samples_path
    return stem + "_eval_results.json"


def _parse_evalplus(path):
    d = json.load(open(path))
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
    return (pb / nb if nb else 0.0), (pp / np_ if np_ else 0.0)


# --------------------------------------------------------------------------- #
def _parallel(fn, items, concurrency, progress_every=0, on_done=None):
    results = []
    done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = [ex.submit(fn, it) for it in items]
        for fut in cf.as_completed(futs):
            r = fut.result()
            results.append(r)
            if on_done is not None:
                on_done(r)
            done += 1
            if progress_every and (done % progress_every == 0 or done == len(items)):
                rate = done / max(1e-6, time.time() - t0)
                print(f"    progress {done}/{len(items)} ({rate:.2f}/s)", flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dims", default="tool,agent,longctx,code",
                   help="comma list subset of tool,agent,longctx,code")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-tokens-code", type=int, default=DEFAULT_MAX_TOKENS_CODE)
    p.add_argument("--code-workdir", default="evalplus_results/quality")
    p.add_argument("--workdir", default="",
                   help="checkpoint dir for incremental resume "
                        "(default: <out>.work)")
    args = p.parse_args()

    dims = [d.strip() for d in args.dims.split(",") if d.strip()]
    print(f"Quality regression: label={args.label} model={args.model} "
          f"url={args.base_url} dims={dims}", flush=True)
    workdir = args.workdir or (os.path.splitext(args.out)[0] + ".work")
    os.makedirs(workdir, exist_ok=True)
    code_workdir = args.code_workdir or workdir
    report = {"label": args.label, "model": args.model, "base_url": args.base_url,
              "date": datetime.datetime.now().isoformat(), "dims": {}}

    if "tool" in dims:
        report["dims"]["tool"] = run_tool(
            args.base_url, args.model, args.concurrency,
            os.path.join(workdir, "tool.jsonl"))
    if "agent" in dims:
        report["dims"]["agent"] = run_agent(
            args.base_url, args.model, args.concurrency,
            os.path.join(workdir, "agent.jsonl"))
    if "longctx" in dims:
        report["dims"]["longctx"] = run_longctx(args.base_url, args.model,
                                                args.concurrency,
                                                os.path.join(workdir, "longctx.jsonl"))
    if "code" in dims:
        report["dims"]["code"] = run_code(args.base_url, args.model,
                                          args.concurrency, args.max_tokens_code,
                                          code_workdir,
                                          os.path.join(workdir, "code_raw.jsonl"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report -> {args.out}")
    _print_summary(report)


def _print_summary(report):
    print("\n=== SUMMARY ===")
    d = report["dims"]
    if "code" in d:
        code = d["code"]
        base = code.get("humaneval_pass_at_1")
        plus = code.get("humaneval_plus_pass_at_1")
        if base is None or plus is None:
            print(f"  code    incomplete ({code.get('generated')}/{code.get('n')})")
        else:
            print(f"  code    HumanEval={base:.3f} HumanEval+={plus:.3f}")
    if "tool" in d:
        print(f"  tool    full_acc={d['tool']['accuracy']:.3f} "
              f"name_acc={d['tool']['name_accuracy']:.3f}")
    if "agent" in d:
        print(f"  agent   acc={d['agent']['accuracy']:.3f} "
              f"tool_rate={d['agent']['tool_invocation_rate']:.3f}")
    if "longctx" in d:
        print(f"  longctx acc={d['longctx']['accuracy']:.3f} "
              f"by_length={d['longctx']['by_length']}")


if __name__ == "__main__":
    main()
