"""B3-b: is the MTP draft head itself weak, or does chained self-conditioning
compound errors? Teacher-forced single-step accuracy isolates the former.

``scripts/b3b_k_sweep.py``'s ``position_accuracy`` curve is measured INSIDE
a real speculative chain: position p's input is the head's OWN (possibly
already-wrong) prediction at position p-1, and the backbone's verify context
for position p is the same possibly-wrong draft prefix. Any decay with p in
that curve therefore mixes two things: (a) the head's raw single-step
quality, and (b) compounding -- both the head's own recurrent state AND the
target's verify context drifting once an earlier step in the chain went off
the model's own true continuation.

This script removes (b). At every position along a real, coherent
sequential generation (this runtime's own free greedy decode -- the
already-B1-proven oracle), the MTP head is asked exactly ONE question,
conditioned on the TRUE token and the TRUE post-backbone hidden state at
that position (never on its own prior output): "what comes next?" -- and the
answer is compared to the actual next true token (which, since the
generation is greedy, IS the target model's own top-1 pick at that
position). This is "K=1, always-teacher-forced" -- the best case the draft
head could possibly see, with a much larger sample size (one comparison per
generated token, not one per round) than the chained K-sweep affords per
prompt.

Comparing this curve to ``b3b_k_sweep.py``'s per-slot-index curve at k
answers B3-b's central question directly:
  * if teacher-forced accuracy is already low and roughly matches K-sweep's
    position-0 accuracy -- the head itself is weak; more/fewer K or better
    scheduling will not fix it.
  * if teacher-forced accuracy is high but K-sweep's position-p accuracy
    (p>0) falls off fast -- the head is fine in isolation, and the problem
    is compounding within the chain (a K/scheduling question, not a head
    quality question).

Model APIs: identical to the other B3-b scripts
(``model.mtp_step``/``model.mtp_new_cache``, ``docs/b1-correctness-
criterion.md`` §7's oracle-is-our-own-non-speculative-path posture).

Run: ~/.venvs/vllm/bin/python scripts/b3b_teacher_forced_head_quality.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__}"
)

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from runtime.model.qwen36_model import Qwen36GenerationState  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402

MODEL_PATH = (
    "/home/bot/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/"
    "snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404"
)
DEVICE = torch.device("cuda")
MAX_SEQ_LEN = 512
N_TOKENS = 160

PROMPTS = {
    "prose": "Once upon a time, in a small village near the mountains,",
    "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "instruction": (
        "Explain, in plain language and without equations, why adding more "
        "layers to a neural network does not always improve its accuracy."
    ),
}


def _logits_for(
    model, token_ids: torch.Tensor, state: Qwen36GenerationState
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = model(token_ids, state)
    return model.compute_logits(hidden[:, -1:, :])[0, -1], hidden[:, -1:, :]


def free_greedy_decode_with_hidden(
    model, prompt_ids: list[int], n_tokens: int
) -> tuple[list[int], list[torch.Tensor], list[int]]:
    """Returns (tokens, hiddens, top5_second_choice).

    ``tokens[i]`` is the i-th generated token. ``hiddens[i]`` is the
    post-norm hidden state produced immediately after consuming
    ``tokens[i]`` -- i.e. exactly the ``prev_hidden`` MTP needs to predict
    ``tokens[i+1]`` (same convention ``b3_mtp_e2e_acceptance_throughput.py``
    uses for ``anchor_hidden``). ``top5_second_choice[i]`` is the true
    model's rank-2 token at the position predicting ``tokens[i+1]`` (kept
    for one extra diagnostic: is a "miss" close, i.e. the head's answer was
    the target's #2 pick, or nowhere near the target's ranking at all).
    """
    state = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    prompt = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)
    logits, _ = _logits_for(model, prompt, state)
    tokens: list[int] = []
    hiddens: list[torch.Tensor] = []
    top5_ranks: list[list[int]] = []
    for _ in range(n_tokens):
        token = int(logits.argmax().item())
        tokens.append(token)
        step = torch.tensor([[token]], device=DEVICE, dtype=torch.long)
        logits, hidden = _logits_for(model, step, state)
        hiddens.append(hidden)
        top5_ranks.append(torch.topk(logits, 5).indices.tolist())
    return tokens, hiddens, top5_ranks


def teacher_forced_accuracy(model, tokens: list[int], hiddens: list[torch.Tensor]):
    """At every position i (0..len(tokens)-2): condition the MTP head on the
    TRUE token[i] and TRUE hiddens[i] (never on the head's own prior
    output), predict a draft for position i+1, compare to the TRUE
    tokens[i+1]. mtp_cache advances by exactly 1 real position per step,
    continuously across the whole sequence -- this is what "teacher forcing"
    means for a head with its own recurrent KV state: every position it
    attends to really happened.
    """
    mtp_cache = model.mtp_new_cache(device=DEVICE, dtype=torch.bfloat16)
    records = []
    for i in range(len(tokens) - 1):
        input_tok = torch.tensor([[tokens[i]]], device=DEVICE, dtype=torch.long)
        draft_token, _ = model.mtp_step(input_tok, hiddens[i], mtp_cache.seq_len, mtp_cache)
        pred = int(draft_token.item())
        true_next = tokens[i + 1]
        records.append({"i": i, "pred": pred, "true": true_next, "match": pred == true_next})
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=N_TOKENS)
    ap.add_argument(
        "--out", type=str, default=".bfdiag/runs/b3b_teacher_forced_head_quality.json"
    )
    args = ap.parse_args()

    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    t_load0 = time.perf_counter()
    model = load_qwen36_model(
        MODEL_PATH, device=DEVICE, max_seq_len=MAX_SEQ_LEN, enable_mtp=True
    )
    print(f"model loaded in {time.perf_counter() - t_load0:.1f}s")

    results: dict = {"n_tokens": args.n_tokens, "prompts": {}}

    for label, text in PROMPTS.items():
        prompt_ids = tok(text, return_tensors=None)["input_ids"]
        print(f"\n=== {label!r} ({len(prompt_ids)} prompt tokens) ===")
        t0 = time.perf_counter()
        tokens, hiddens, top5 = free_greedy_decode_with_hidden(model, prompt_ids, args.n_tokens)
        print(f"  free greedy decode: {len(tokens)} tokens in {time.perf_counter() - t0:.1f}s")

        records = teacher_forced_accuracy(model, tokens, hiddens)
        n = len(records)
        matches = sum(r["match"] for r in records)
        acc = matches / n if n else float("nan")

        # rank-2 rescue rate: among misses, how often was the head's guess
        # the target's OWN #2 pick (a "close" miss vs a "nowhere near" miss).
        misses = [r for r in records if not r["match"]]
        close_misses = sum(
            1 for r in misses if r["pred"] in top5[r["i"]][1:5]
        )
        close_rate = close_misses / len(misses) if misses else float("nan")

        # first-quarter vs last-quarter accuracy -- does raw (non-chained)
        # quality drift as context grows, independent of any speculative
        # chaining at all?
        q = max(n // 4, 1)
        first_q_acc = sum(r["match"] for r in records[:q]) / q if n else float("nan")
        last_q_acc = sum(r["match"] for r in records[-q:]) / q if n else float("nan")

        print(f"  teacher-forced top-1 accuracy: {acc:.3f} ({matches}/{n})")
        print(f"  first-quarter acc: {first_q_acc:.3f}  last-quarter acc: {last_q_acc:.3f}")
        print(
            f"  among misses, head's guess was target's rank 2-5: "
            f"{close_rate:.3f} ({close_misses}/{len(misses)})"
        )

        results["prompts"][label] = {
            "n_positions": n,
            "accuracy": acc,
            "first_quarter_accuracy": first_q_acc,
            "last_quarter_accuracy": last_q_acc,
            "close_miss_rate_rank2_5": close_rate,
            "records": records,
        }

    out_path = Path(_ROOT) / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
