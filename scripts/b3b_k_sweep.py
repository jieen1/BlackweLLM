"""B3-b: acceptance rate as a function of K -- the cheapest lever left.

``docs/implementation-plan.md`` §7.1 B3 has ruled out every other lever
(sparkinfer kernel launch overhead, batched GDN projections, the anchor-fold)
down to a hard ceiling: even a free GDN verify only gets e2e to prose 1.09x /
code 0.905x. The one knob nobody has swept yet is K itself -- the shipped
default is K=8, chosen when the head was wired up, not because 8 was
measured to be good. This script measures the acceptance-rate / throughput
curve across K so that choice stops being an assumption.

Reuses ``scripts/b3_mtp_e2e_acceptance_throughput.py``'s exact draft/verify/
accept-reject/rollback loop (same model APIs, same greedy accept/reject via
``runtime.mtp_accept.determine_accept_reject_from_predictions``) -- this is
not a new algorithm, just the same one run at multiple K with one extra
piece of bookkeeping: verify computes logits for ALL K draft positions in
one forward regardless of where the greedy chain would first reject (that miss
is a decision, not a compute skip), so recording the FULL length-K match
vector per round (not just the first-mismatch index the accept/reject
function returns) costs nothing extra and turns into two things at once:

  * the existing reject_position histogram (first False in the vector, or
    "full accept" if all True) -- this is a decision derived from the vector,
    not a separate measurement.
  * a per-slot-index accuracy curve (mean of vector[p] across all rounds,
    for each p in 0..K-1) that separates "does the draft head go bad
    immediately" from "does it decay smoothly with position" -- the two
    hypotheses AGENTS.md's diagnostics table says the histogram's *shape*
    should distinguish.

One model load (``enable_mtp=True``), swept over every (K, prompt) pair, so
GPU thermal state and JIT-compiled kernel state are shared -- the run-to-run
"GPU thermal drift" this project has repeatedly documented
(``notes/2026-08-02-b3a-anchor-fold-verdict.md``) is a reason to compare
K values WITHIN one process, not across separate invocations.

Correctness posture: unchanged from the e2e script this is derived from --
committed sequences are compared against this runtime's own free greedy
decode (``docs/b1-correctness-criterion.md`` §7's B3 judgement), not HF.
Mismatches are recorded, not asserted away; ``scripts/b3b_divergence_gap.py``
is the follow-up that turns a mismatch into a gap-error measurement instead
of a bare boolean.

Run: ~/.venvs/vllm/bin/python scripts/b3b_k_sweep.py
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
from runtime.mtp_accept import determine_accept_reject_from_predictions  # noqa: E402

MODEL_PATH = (
    "/home/bot/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/"
    "snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404"
)
DEVICE = torch.device("cuda")
MAX_SEQ_LEN = 512
N_TOKENS = 64  # tokens generated per (K, prompt) combo

PROMPTS = {
    "prose": "Once upon a time, in a small village near the mountains,",
    "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
}


def _logits_for(
    model, token_ids: torch.Tensor, state: Qwen36GenerationState
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = model(token_ids, state)
    return model.compute_logits(hidden[:, -1:, :])[0, -1], hidden[:, -1:, :]


def free_greedy_decode(model, prompt_ids: list[int], n_tokens: int) -> tuple[list[int], float]:
    state = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    prompt = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits, _ = _logits_for(model, prompt, state)
    tokens: list[int] = []
    for _ in range(n_tokens):
        token = int(logits.argmax().item())
        tokens.append(token)
        step = torch.tensor([[token]], device=DEVICE, dtype=torch.long)
        logits, _ = _logits_for(model, step, state)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return tokens, elapsed


def speculative_decode(
    model, prompt_ids: list[int], n_tokens: int, k: int
) -> tuple[list[int], float, list[dict]]:
    """Same algorithm as b3_mtp_e2e_acceptance_throughput.py's function of
    the same name, plus a full length-k ``match`` vector per round (every
    position's raw predicted==drafted comparison, not just where the
    chain first breaks)."""
    state = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    mtp_cache = model.mtp_new_cache(device=DEVICE, dtype=torch.bfloat16)
    prompt = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    prompt_logits, _ = _logits_for(model, prompt, state)
    anchor_token = int(prompt_logits.argmax().item())
    anchor_step = torch.tensor([[anchor_token]], device=DEVICE, dtype=torch.long)
    anchor_logits, anchor_hidden = _logits_for(model, anchor_step, state)
    anchor_argmax = int(anchor_logits.argmax().item())

    committed: list[int] = [anchor_token]
    rounds: list[dict] = []

    while len(committed) < n_tokens:
        round_mtp_start = mtp_cache.seq_len
        round_t0 = time.perf_counter()
        draft_tokens: list[int] = []
        mtp_hidden = anchor_hidden
        next_input = torch.tensor([[anchor_token]], device=DEVICE, dtype=torch.long)
        for _step in range(k):
            draft_token, mtp_hidden = model.mtp_step(
                next_input, mtp_hidden, mtp_cache.seq_len, mtp_cache
            )
            draft_tokens.append(int(draft_token.item()))
            next_input = draft_token.view(1, 1)

        past_len = state.num_tokens_seen
        draft_tensor = torch.tensor([draft_tokens], device=DEVICE, dtype=torch.long)
        verify_hidden, gdn_snapshots = model.verify_forward(draft_tensor, state)
        verify_logits = model.compute_logits(verify_hidden)[0]  # [K, vocab]
        verify_argmax = verify_logits.argmax(dim=-1).tolist()

        predicted_tokens = [anchor_argmax] + verify_argmax  # length K+1
        # Full per-position match vector -- computed regardless of where
        # the chain would first reject (verify already has all K rows).
        match_vector = [predicted_tokens[p] == draft_tokens[p] for p in range(k)]

        decision = determine_accept_reject_from_predictions(
            [anchor_token] + draft_tokens, predicted_tokens
        )
        m = decision["num_accepted"]
        assert (m == k) or (not match_vector[m]), "match_vector disagrees with accept/reject"
        assert all(match_vector[:m]), "match_vector disagrees with accept/reject"

        model.commit_verify(state, gdn_snapshots, past_len=past_len, accepted_count=m)
        mtp_cache.seq_len = round_mtp_start + m

        committed.extend(decision["committed"])

        new_anchor = decision["committed"][-1]
        new_anchor_tensor = torch.tensor([[new_anchor]], device=DEVICE, dtype=torch.long)
        new_anchor_logits, new_anchor_hidden = _logits_for(model, new_anchor_tensor, state)

        torch.cuda.synchronize()
        round_elapsed = time.perf_counter() - round_t0
        rounds.append(
            {
                "k": k,
                "num_accepted": m,
                "rejected_at": decision["rejected_at"],
                "match_vector": match_vector,
                "round_s": round_elapsed,
            }
        )

        anchor_token = new_anchor
        anchor_hidden = new_anchor_hidden
        anchor_argmax = int(new_anchor_logits.argmax().item())

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return committed[:n_tokens], elapsed, rounds


def reject_position_histogram(rounds: list[dict], k: int) -> dict[str, int]:
    hist: dict[str, int] = {str(p): 0 for p in range(k)}
    hist["full_accept"] = 0
    for r in rounds:
        pos = r["rejected_at"]
        hist["full_accept" if pos is None else str(pos)] += 1
    return hist


def position_accuracy(rounds: list[dict], k: int) -> list[float]:
    """mean(match_vector[p]) across all rounds, for each p -- unconditional
    of early stopping, i.e. NOT gated on earlier positions having matched."""
    if not rounds:
        return [float("nan")] * k
    sums = [0] * k
    for r in rounds:
        for p in range(k):
            sums[p] += int(r["match_vector"][p])
    return [s / len(rounds) for s in sums]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--n-tokens", type=int, default=N_TOKENS)
    ap.add_argument("--out", type=str, default=".bfdiag/runs/b3b_k_sweep.json")
    args = ap.parse_args()

    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    t_load0 = time.perf_counter()
    model = load_qwen36_model(
        MODEL_PATH, device=DEVICE, max_seq_len=MAX_SEQ_LEN, enable_mtp=True
    )
    print(f"model loaded in {time.perf_counter() - t_load0:.1f}s")

    warm_ids = tok("Warm up the kernels before timing.", return_tensors=None)["input_ids"]
    free_greedy_decode(model, warm_ids, 4)
    print("warmup (free greedy) done")

    results: dict = {"n_tokens": args.n_tokens, "k_values": args.k_values, "prompts": {}}

    # One free-greedy baseline per prompt, measured once (shared across all
    # K values in this same warm process -- see module docstring on why
    # cross-process re-measurement is not comparable here).
    baselines: dict[str, tuple[list[int], float]] = {}
    for label, text in PROMPTS.items():
        prompt_ids = tok(text, return_tensors=None)["input_ids"]
        ref_tokens, ref_time = free_greedy_decode(model, prompt_ids, args.n_tokens)
        baselines[label] = (ref_tokens, ref_time)
        print(f"baseline[{label}]: {args.n_tokens} tok in {ref_time:.3f}s "
              f"= {args.n_tokens / ref_time:.2f} tok/s")

    for k in args.k_values:
        print(f"\n{'=' * 70}\nK={k}\n{'=' * 70}")
        # Per-K warmup: pay any K-specific extend-mode JIT compile before
        # timing anything real at this K.
        try:
            speculative_decode(model, warm_ids, min(2 * k, 8) + k, k)
        except Exception as exc:  # noqa: BLE001
            print(f"  K={k} warmup FAILED: {exc!r} -- skipping this K")
            results["prompts"].setdefault("__errors__", {})[str(k)] = repr(exc)
            continue

        results["prompts"].setdefault(str(k), {})
        for label, text in PROMPTS.items():
            prompt_ids = tok(text, return_tensors=None)["input_ids"]
            ref_tokens, ref_time = baselines[label]
            try:
                spec_tokens, spec_time, rounds = speculative_decode(
                    model, prompt_ids, args.n_tokens, k
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [{label}] K={k} FAILED: {exc!r}")
                results["prompts"][str(k)][label] = {"error": repr(exc)}
                continue

            match = ref_tokens == spec_tokens
            first_diff = None
            if not match:
                for i, (a, b) in enumerate(zip(ref_tokens, spec_tokens)):
                    if a != b:
                        first_diff = i
                        break

            total_accepted = sum(r["num_accepted"] for r in rounds)
            total_drafted = sum(r["k"] for r in rounds)
            accept_rate = total_accepted / total_drafted if total_drafted else 0.0
            mean_accept_per_round = total_accepted / len(rounds) if rounds else 0.0
            spec_tok_s = len(spec_tokens) / spec_time
            ref_tok_s = args.n_tokens / ref_time
            speedup = spec_tok_s / ref_tok_s if ref_tok_s else float("nan")
            hist = reject_position_histogram(rounds, k)
            pos_acc = position_accuracy(rounds, k)

            print(
                f"  [{label:6s}] rounds={len(rounds):3d} accept_rate={accept_rate:.3f} "
                f"mean_accepted/round={mean_accept_per_round:.2f} "
                f"spec_tok/s={spec_tok_s:.2f} speedup={speedup:.2f}x "
                f"match={match}{'' if match else f' (first_diff={first_diff})'}"
            )
            print(f"    reject_position_histogram: {hist}")
            print(f"    position_accuracy (p=0..{k - 1}): "
                  f"{[round(x, 3) for x in pos_acc]}")

            entry = {
                "num_rounds": len(rounds),
                "accept_rate": accept_rate,
                "mean_accepted_per_round": mean_accept_per_round,
                "spec_tokens_per_sec": spec_tok_s,
                "ref_tokens_per_sec": ref_tok_s,
                "speedup": speedup,
                "tokens_match": match,
                "first_diff_index": first_diff,
                "reject_position_histogram": hist,
                "position_accuracy": pos_acc,
                "rounds": rounds,
            }
            results["prompts"][str(k)][label] = entry

    out_path = Path(_ROOT) / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
