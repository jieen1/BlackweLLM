"""Greedy alignment harness: our eager DSV4 graph vs the llama.cpp oracle.

Phase 3's correctness red line is token-level greedy agreement with
llama.cpp on the same quantized weights (plan Phase 3 gate). This harness
is the scaffolding for it, usable TODAY against the eager graph:

  # 1. run our side (needs GPU + QSR_DSV4_FULL_LOAD=1; slow, eager)
  scripts/dsv4_align_eager_vs_llama.py ours --n 128 --out /tmp/ours.json
  # 2. run the oracle side (needs the llama.cpp build; fast)
  scripts/dsv4_align_eager_vs_llama.py llama --n 128 --out /tmp/llama.json
  # 3. compare
  scripts/dsv4_align_eager_vs_llama.py compare /tmp/ours.json /tmp/llama.json

The llama.cpp side MUST run bare completion (-no-cnv): llama-completion
otherwise wraps the prompt in the chat template (fact-baseline §9.3).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GGUF = Path(
    "/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf"
)
_DEFAULT_LLAMA_COMPLETION = "/home/bot/project/llama.cpp/build-sm120/bin/llama-completion"
LLAMA_COMPLETION = Path(os.environ.get("QSR_LLAMA_COMPLETION", _DEFAULT_LLAMA_COMPLETION))
TOKENIZER = REPO_ROOT / "notes" / "dsv4flash-ref" / "tokenizer.json"

DEFAULT_PROMPTS = [
    "The meaning of life is",
    "Explain the theory of relativity in simple terms:",
]


def _tokenizer():
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER))


def run_ours(args) -> None:
    import torch

    sys.path.insert(0, str(REPO_ROOT))
    from runtime.model.dsv4_model import load_dsv4_from_gguf

    tok = _tokenizer()
    model, count = load_dsv4_from_gguf(GGUF, max_seq_len=args.max_seq_len, device="cuda")
    print(f"loaded {count} tensors", flush=True)
    records = []
    for prompt in DEFAULT_PROMPTS[: args.workloads]:
        ids = tok.encode(prompt)
        input_ids = torch.tensor([ids], device="cuda")
        generated: list[int] = []
        step_times: list[float] = []
        t0 = time.time()
        logits = model(input_ids, start_pos=0)
        step_times.append(time.time() - t0)
        pos = len(ids)
        next_id = int(logits[0, -1].argmax())
        for step in range(args.n):
            generated.append(next_id)
            t0 = time.time()
            out = model(torch.tensor([[next_id]], device="cuda"), start_pos=pos)
            step_times.append(time.time() - t0)
            next_id = int(out[0, -1].argmax())
            pos += 1
            if step % 10 == 9:
                print(f"  {step + 1}/{args.n} tokens", flush=True)
        records.append(
            {
                "prompt": prompt,
                "prompt_ids": ids,
                "generated_ids": generated,
                "generated_text": tok.decode(generated),
                "prefill_s": step_times[0],
                "decode_s_per_token": step_times[1:],
            }
        )
        print(f"workload done: {records[-1]['generated_text'][:80]!r}", flush=True)
    Path(args.out).write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


def run_llama(args) -> None:
    tok = _tokenizer()
    records = []
    for prompt in DEFAULT_PROMPTS[: args.workloads]:
        cmd = [
            str(LLAMA_COMPLETION),
            "-m",
            str(GGUF),
            "-p",
            prompt,
            "--temp",
            "-1",
            "-n",
            str(args.n),
            "-no-cnv",
            "--no-display-prompt",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        text = proc.stdout
        # llama-completion echoes nothing else with --no-display-prompt
        ids = tok.encode(text, add_special_tokens=False)
        records.append(
            {
                "prompt": prompt,
                "generated_ids": ids,
                "generated_text": text,
            }
        )
        print(f"workload done: {text[:80]!r}", flush=True)
    Path(args.out).write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


def compare(args) -> None:
    ours = json.loads(Path(args.ours).read_text())
    theirs = json.loads(Path(args.llama).read_text())
    assert len(ours) == len(theirs)
    total = 0
    agree = 0
    for o, t in zip(ours, theirs):
        assert o["prompt"] == t["prompt"]
        o_ids, t_ids = o["generated_ids"], t["generated_ids"]
        n = min(len(o_ids), len(t_ids))
        first_div = next((i for i in range(n) if o_ids[i] != t_ids[i]), None)
        agree_here = n if first_div is None else first_div
        total += n
        agree += agree_here
        print(f"prompt: {o['prompt']!r}")
        print(f"  tokens compared: {n}, agreement prefix: {agree_here}")
        if first_div is not None:
            print(f"  FIRST DIVERGENCE at token {first_div}:")
            print(f"    ours:   {o_ids[first_div : first_div + 8]}")
            print(f"    oracle: {t_ids[first_div : first_div + 8]}")
            ctx = o["prompt"] + _tokenizer().decode(o_ids[:first_div])
            print(f"    context: ...{ctx[-60:]!r}")
    print(f"TOTAL: {agree}/{total} tokens agree ({agree / max(total, 1):.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="side", required=True)
    for name, fn in (("ours", run_ours), ("llama", run_llama)):
        p = sub.add_parser(name)
        p.add_argument("--n", type=int, default=128)
        p.add_argument("--workloads", type=int, default=1)
        p.add_argument("--max-seq-len", type=int, default=512)
        p.add_argument("--out", required=True)
        p.set_defaults(fn=fn)
    p = sub.add_parser("compare")
    p.add_argument("ours")
    p.add_argument("llama")
    p.set_defaults(fn=compare)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
