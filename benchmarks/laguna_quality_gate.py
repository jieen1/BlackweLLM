#!/usr/bin/env python3
"""Laguna L2 质量链：native/oracle A/B 比对 + prompt ids 断言护栏。

两条路径在独立子进程中运行，避免 vLLM 全局状态污染。

Usage:
    /home/bot/.venvs/vllm/bin/python -m benchmarks.laguna_quality_gate
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

MODEL = "poolside/Laguna-S-2.1-NVFP4"
PYTHON = "/home/bot/.venvs/vllm/bin/python"

GREEDY_PROMPTS = [
    "The capital of France is",
    "Write a Python function to check if a number is prime.",
    "What is 15 * 37? Show your work step by step.",
    "Explain the theory of relativity in simple terms.",
]

LAGUNA_EOS = (2, 24)
MAX_DECODE_TOKENS = 80
ORACLE_TIMEOUT_SECONDS = 1800

BACKEND_SCRIPT = '''
import json, os, sys
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, {repo_root!r})
MODEL = {model!r}
PROMPTS = {prompts!r}
EOS = {eos!r}
MAX_TOK = {max_tok!r}

from transformers import AutoTokenizer
from runtime.backends.laguna import LagunaBackend
from runtime.laguna_config import build_laguna_config

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
runtime_config = build_laguna_config(
    MODEL,
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    dtype="bfloat16",
    trust_remote_code=True,
)
backend = LagunaBackend(
    runtime_config, num_slots=4, block_size=64, blocks_per_slot=128
)

results = []
for prompt in PROMPTS:
    ids = tokenizer.encode(prompt)
    backend.reset_slot(0)
    ft = backend.prefill(0, ids)
    tokens = [ft]
    for _ in range(MAX_TOK - 1):
        tok = backend.decode(0, tokens[-1])
        tokens.append(tok)
        if tok in EOS:
            break
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    results.append({{"prompt": prompt, "prompt_ids": ids, "output_ids": tokens, "text": text[:200]}})
    backend.reset_slot(0)

with open(sys.argv[1], "w") as f:
    json.dump(results, f)
print("Backend done:", len(results), "prompts")
'''

VLLM_SCRIPT = '''
import json, os, sys
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
MODEL = {model!r}
PROMPTS = {prompts!r}
MAX_TOK = {max_tok!r}

def main():
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL, max_model_len=4096, gpu_memory_utilization=0.85,
              enforce_eager=True, dtype="bfloat16", disable_log_stats=True)
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=0, max_tokens=MAX_TOK)

    results = []
    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt)
        out = llm.generate([prompt], params)[0]
        tokens = list(out.outputs[0].token_ids)
        text = out.outputs[0].text
        results.append({{"prompt": prompt, "prompt_ids": ids, "output_ids": tokens, "text": text[:200]}})

    with open(sys.argv[1], "w") as f:
        json.dump(results, f)
    print("vLLM done:", len(results), "prompts")


if __name__ == "__main__":
    main()
'''


def assert_prompt_ids_equal(prompt, ids_a, ids_b, path_a, path_b):
    if ids_a != ids_b:
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(ids_a, ids_b)) if a != b),
            min(len(ids_a), len(ids_b)),
        )
        raise AssertionError(
            f"Prompt token ids 不一致！\n"
            f"  Prompt: {prompt!r}\n"
            f"  {path_a}: {ids_a[:20]} (len={len(ids_a)})\n"
            f"  {path_b}: {ids_b[:20]} (len={len(ids_b)})\n"
            f"  首个分歧位置: {first_diff}"
        )


def run_in_subprocess(script: str, label: str) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(script)
        script_path = f.name
    try:
        print(f"\n{'='*60}")
        print(f"  Running {label} in subprocess...")
        print(f"{'='*60}")
        child_env = os.environ.copy()
        # FlashInfer's JIT invokes `ninja` by name.  The vLLM interpreter is
        # intentionally addressed by an absolute path, so its sibling scripts
        # are not otherwise guaranteed to be on PATH.
        python_bin = str(Path(PYTHON).parent)
        child_env["PATH"] = python_bin + os.pathsep + child_env.get("PATH", "")
        # FlashInfer otherwise starts one memory-hungry nvcc process per CPU.
        # The oracle host has 23 GiB of RAM; a two-job build avoids swap
        # thrashing without changing either inference implementation.
        child_env.setdefault("MAX_JOBS", "2")
        result = subprocess.run(
            [PYTHON, script_path, out_path],
            capture_output=True,
            text=True,
            timeout=ORACLE_TIMEOUT_SECONDS,
            env=child_env,
        )
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        if result.returncode != 0:
            failure_report = os.environ.get("QSR_QUALITY_FAILURE_REPORT")
            if failure_report:
                Path(failure_report).write_text(
                    json.dumps(
                        {
                            "label": label,
                            "returncode": result.returncode,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                        },
                        indent=2,
                    )
                )
            print(f"❌ {label} failed with code {result.returncode}")
            sys.exit(1)
        with open(out_path) as f:
            return json.load(f)
    finally:
        os.unlink(script_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def main():
    fmt = dict(
        repo_root=_REPO_ROOT,
        model=MODEL,
        prompts=GREEDY_PROMPTS,
        eos=LAGUNA_EOS,
        max_tok=MAX_DECODE_TOKENS,
    )

    # Phase 1: Run backend in subprocess
    backend_results = run_in_subprocess(
        BACKEND_SCRIPT.format(**fmt), "LagunaBackend"
    )

    # Phase 2: Run vLLM in subprocess
    vllm_results = run_in_subprocess(
        VLLM_SCRIPT.format(**fmt), "stock vLLM"
    )

    # Phase 3: Compare
    print(f"\n{'='*60}")
    print("  A/B Comparison")
    print(f"{'='*60}")

    all_match = True
    results = []
    for br, vr in zip(backend_results, vllm_results):
        prompt = br["prompt"]

        # Guardrail: assert prompt ids equal
        try:
            assert_prompt_ids_equal(
                prompt, br["prompt_ids"], vr["prompt_ids"],
                "Backend", "vLLM",
            )
            print(f"  ✅ Prompt ids match: {prompt[:40]!r} (len={len(br['prompt_ids'])}, BOS={br['prompt_ids'][0]})")
        except AssertionError as e:
            print(f"  ❌ {e}")
            all_match = False
            continue

        # Token-level comparison
        bt = br["output_ids"]
        vt = vr["output_ids"]
        min_len = min(len(bt), len(vt))
        first_diverge = None
        for i in range(min_len):
            if bt[i] != vt[i]:
                first_diverge = i
                break
        if first_diverge is None and len(bt) != len(vt):
            first_diverge = min_len

        match = first_diverge is None
        if not match:
            all_match = False

        status = "✅ MATCH" if match else f"❌ DIVERGE@{first_diverge}"
        print(f"  {status}: {prompt[:40]!r}")
        if not match:
            print(f"    Backend: {br['text'][:80]!r}")
            print(f"    vLLM:    {vr['text'][:80]!r}")

        results.append({
            "prompt": prompt,
            "prompt_ids_len": len(br["prompt_ids"]),
            "prompt_bos": br["prompt_ids"][0],
            "backend_tokens": len(bt),
            "vllm_tokens": len(vt),
            "match": match,
            "first_diverge_at": first_diverge,
            "backend_text": br["text"],
            "vllm_text": vr["text"],
        })

    print(f"\n{'='*60}")
    print(f"QUALITY GATE: {'✅ PASS' if all_match else '❌ FAIL'}")
    print(f"  Token-level A/B match: {sum(r['match'] for r in results)}/{len(results)}")
    print(f"{'='*60}")

    fixture = {
        "model": MODEL,
        "date": datetime.now().isoformat(timespec="seconds"),
        "guardrail": "prompt_ids_assertion + subprocess_isolation",
        "ab_match_all": all_match,
        "results": results,
    }
    out_path = Path("benchmarks/fixtures/laguna_quality_gate.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    if not all_match:
        sys.exit(1)
    return all_match


if __name__ == "__main__":
    from bfdiag.record import run_record  # demo: bf run-record integration

    with run_record(script=__file__) as rec:
        rec.metric("ab_match_all", 1.0 if main() else 0.0)
