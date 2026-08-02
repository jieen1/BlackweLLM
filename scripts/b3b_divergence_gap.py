"""B3-b: the code-prompt ``tokens_match=False`` from B3-a's e2e report,
judged by gap error instead of token equality.

``notes/2026-08-02-b3-mtp-e2e-acceptance-throughput.md`` found that the code
prompt's committed speculative-decode sequence diverges from this runtime's
own free greedy decode at position 2 (``n-1`` vs ``n - 1``), flagged it as
"not the same as a bug" (both continuations are valid Python), and
explicitly listed "measure the real logit gap at that position, not just
the token difference" as unfinished work
(docs/b1-correctness-criterion.md §7's B3 judgement: the reference frame is
our OWN non-speculative path, not HF, and the judgement instrument is
B1-R's gap-error metric -- ``docs/b1-correctness-criterion.md`` §6.1's
calibrated bars -- not token equality, which B1's own history already
proved to be an over-strict criterion for this exact "same math, different
kernel schedule" class of disagreement).

This script reruns that same comparison (free greedy decode vs speculative
decode, both this runtime's own paths, same algorithm as
``scripts/b3_mtp_e2e_acceptance_throughput.py``) but keeps every logits row
that COULD have produced a committed token (not just the argmax id), so
that if/when the two committed sequences first diverge, the exact pair of
logits rows that disagreement came from can be pulled out and run through
``bfdiag.divergence.logit_agreement.compare_step`` -- the SAME metric
``scripts/b3_verify_batching_logit_agreement.py`` uses, so the number lands
in the same units as ``docs/b1-correctness-criterion.md`` §6.1's table.

Why the two rows are legitimately comparable (not a mismatched-context
artifact): by construction, both paths' committed sequences are IDENTICAL
up to (not including) the first-divergence index -- that is what "first
divergence" means. So both logits rows were computed by a forward pass that
had consumed the exact same prefix. This is the same "same input, different
kernel path" comparison ``scripts/b1_forced_decode_agreement.py`` and
``scripts/b3_verify_batching_logit_agreement.py`` make via explicit
step-locking; here the shared prefix falls out of the real (unforced) run
instead of being forced, which is only valid up to the first divergence --
this script does not attempt to compare anything past that point (a later
position's context has already forked, so a logits comparison there would
not isolate the same kernel-path question).

``TopKRow``/``_row_map``/``agreement`` below are a direct copy of
``scripts/b3_verify_batching_logit_agreement.py``'s own helpers of the same
name (same STORE_TOP_K/CAPTURE_TOP_K/GAP_TOP_K constants, so results are
directly comparable) -- kept local rather than imported so this script does
not also import that script's unrelated ``spec_forward_batched`` dependency.

Run: ~/.venvs/vllm/bin/python scripts/b3b_divergence_gap.py
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

from bfdiag.divergence.logit_agreement import (  # noqa: E402
    CALIBRATED_THRESHOLDS,
    AgreementReport,
    WorkloadAgreement,
    compare_step,
    evaluate_summary,
    missing_from_intersection,
)
from runtime.model.qwen36_model import Qwen36GenerationState  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402
from runtime.mtp_accept import determine_accept_reject_from_predictions  # noqa: E402

MODEL_PATH = (
    "/home/bot/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/"
    "snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404"
)
DEVICE = torch.device("cuda")
torch.set_grad_enabled(False)
MAX_SEQ_LEN = 256
K = 8  # same K the B3-a e2e report used, for a like-for-like reproduction
N_TOKENS = 32

CAPTURE_TOP_K = 64
GAP_TOP_K = 8
STORE_TOP_K = 1024

PROMPTS = {
    "prose": "Once upon a time, in a small village near the mountains,",
    "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
}


class TopKRow:
    """Copy of b3_verify_batching_logit_agreement.py's TopKRow."""

    __slots__ = ("indices", "values", "logsumexp")

    def __init__(self, logits: torch.Tensor) -> None:
        row = logits.float()
        top = torch.topk(row, STORE_TOP_K)
        self.indices = top.indices.cpu()
        self.values = top.values.cpu()
        self.logsumexp = float(torch.logsumexp(row, dim=-1).item())

    def top(self, k: int) -> list[int]:
        return self.indices[:k].tolist()

    def lookup(self, ids: list[int]) -> dict[int, float] | None:
        table = dict(zip(self.indices.tolist(), self.values.tolist(), strict=True))
        if not all(i in table for i in ids):
            return None
        return {i: table[i] for i in ids}


def _row_map(mine: TopKRow, oracle: TopKRow) -> tuple[dict[int, float], dict[int, float]]:
    ids = sorted(set(mine.top(CAPTURE_TOP_K)) | set(oracle.top(CAPTURE_TOP_K)))
    mine_map = mine.lookup(ids)
    oracle_map = oracle.lookup(ids)
    assert mine_map is not None and oracle_map is not None, (
        f"a token in one side's top-{CAPTURE_TOP_K} fell outside the other's "
        f"stored top-{STORE_TOP_K}; raise STORE_TOP_K rather than dropping it"
    )
    return mine_map, oracle_map


def _logits_for(
    model, token_ids: torch.Tensor, state: Qwen36GenerationState
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = model(token_ids, state)
    return model.compute_logits(hidden[:, -1:, :])[0, -1], hidden[:, -1:, :]


def free_greedy_decode_capturing(model, prompt_ids: list[int], n_tokens: int):
    """Returns (tokens, rows): rows[i] is the logits row that produced
    tokens[i] (i.e. captured BEFORE argmax picks tokens[i])."""
    state = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    prompt = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)
    logits, _ = _logits_for(model, prompt, state)
    tokens: list[int] = []
    rows: list[TopKRow] = []
    for _ in range(n_tokens):
        rows.append(TopKRow(logits))
        token = int(logits.argmax().item())
        tokens.append(token)
        step = torch.tensor([[token]], device=DEVICE, dtype=torch.long)
        logits, _ = _logits_for(model, step, state)
    return tokens, rows


def speculative_decode_capturing(model, prompt_ids: list[int], n_tokens: int, k: int):
    """Same algorithm as b3_mtp_e2e_acceptance_throughput.py's
    speculative_decode, additionally keeping (per round) the TopKRow for
    every predicted_tokens[j] slot (j=0 is the anchor's own independent
    forward; j=1..k are verify_logits[0..k-1]) and the flat committed-index
    range each round fills, so a later divergence index can be mapped back
    to the exact logits row that produced it.
    """
    state = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    mtp_cache = model.mtp_new_cache(device=DEVICE, dtype=torch.bfloat16)
    prompt = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)

    prompt_logits, _ = _logits_for(model, prompt, state)
    anchor_token = int(prompt_logits.argmax().item())
    anchor_step = torch.tensor([[anchor_token]], device=DEVICE, dtype=torch.long)
    anchor_logits, anchor_hidden = _logits_for(model, anchor_step, state)
    anchor_argmax = int(anchor_logits.argmax().item())

    committed: list[int] = [anchor_token]
    round_meta: list[dict] = []  # {"committed_before": int, "num_accepted": int, "rows": [...]}

    while len(committed) < n_tokens:
        round_mtp_start = mtp_cache.seq_len
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

        predicted_tokens = [anchor_argmax] + verify_argmax
        predicted_rows = [TopKRow(anchor_logits)] + [TopKRow(verify_logits[p]) for p in range(k)]

        decision = determine_accept_reject_from_predictions(
            [anchor_token] + draft_tokens, predicted_tokens
        )
        m = decision["num_accepted"]

        committed_before = len(committed)
        model.commit_verify(state, gdn_snapshots, past_len=past_len, accepted_count=m)
        mtp_cache.seq_len = round_mtp_start + m
        committed.extend(decision["committed"])
        round_meta.append(
            {"committed_before": committed_before, "num_accepted": m, "rows": predicted_rows}
        )

        new_anchor = decision["committed"][-1]
        new_anchor_tensor = torch.tensor([[new_anchor]], device=DEVICE, dtype=torch.long)
        new_anchor_logits, new_anchor_hidden = _logits_for(model, new_anchor_tensor, state)

        anchor_token = new_anchor
        anchor_hidden = new_anchor_hidden
        anchor_logits = new_anchor_logits
        anchor_argmax = int(new_anchor_logits.argmax().item())

    return committed[:n_tokens], round_meta


def find_source_row(round_meta: list[dict], flat_index: int) -> TopKRow:
    for r in round_meta:
        lo = r["committed_before"]
        hi = lo + r["num_accepted"]  # inclusive: j in [0, num_accepted] fills [lo, hi]
        if lo <= flat_index <= hi:
            j = flat_index - lo
            return r["rows"][j]
    raise AssertionError(f"flat_index {flat_index} not covered by any round")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--n-tokens", type=int, default=N_TOKENS)
    ap.add_argument("--out", type=str, default=".bfdiag/runs/b3b_divergence_gap.json")
    args = ap.parse_args()

    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    t0 = time.perf_counter()
    model = load_qwen36_model(
        MODEL_PATH, device=DEVICE, max_seq_len=MAX_SEQ_LEN, enable_mtp=True
    )
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")

    warm_ids = tok("Warm up the kernels before timing.", return_tensors=None)["input_ids"]
    free_greedy_decode_capturing(model, warm_ids, 4)
    speculative_decode_capturing(model, warm_ids, 4 + args.k, args.k)
    print("warmup done")

    results: dict = {"k": args.k, "n_tokens": args.n_tokens, "prompts": {}}

    for label, text in PROMPTS.items():
        print(f"\n=== prompt={label!r} ===")
        prompt_ids = tok(text, return_tensors=None)["input_ids"]
        ref_tokens, seq_rows = free_greedy_decode_capturing(model, prompt_ids, args.n_tokens)
        spec_tokens, round_meta = speculative_decode_capturing(
            model, prompt_ids, args.n_tokens, args.k
        )

        match = ref_tokens == spec_tokens
        entry: dict = {"tokens_match": match, "ref_tokens": ref_tokens, "spec_tokens": spec_tokens}
        print(f"  tokens_match={match}")

        if match:
            results["prompts"][label] = entry
            continue

        first_diff = next(
            i for i, (a, b) in enumerate(zip(ref_tokens, spec_tokens)) if a != b
        )
        print(f"  first_diff at index {first_diff}: ref={ref_tokens[first_diff]} "
              f"({tok.decode([ref_tokens[first_diff]])!r}) vs "
              f"spec={spec_tokens[first_diff]} ({tok.decode([spec_tokens[first_diff]])!r})")

        oracle_row = seq_rows[first_diff]
        mine_row = find_source_row(round_meta, first_diff)

        mine_map, oracle_map = _row_map(mine_row, oracle_row)
        gaps = missing_from_intersection(mine_map, oracle_map, top_k=GAP_TOP_K)
        step = compare_step(
            0,
            ref_tokens[first_diff],
            mine_map,
            oracle_map,
            gap_top_k=GAP_TOP_K,
            mine_logsumexp=mine_row.logsumexp,
            oracle_logsumexp=oracle_row.logsumexp,
        )
        workload = WorkloadAgreement(
            workload_name=f"{label}_divergence", prompt_token_ids=(), steps=(step,)
        )
        report = AgreementReport(workloads=(workload,))
        metrics = report.summary_metrics()
        passed, reasons = evaluate_summary(metrics, CALIBRATED_THRESHOLDS)

        print(f"  gap_error={step.gap_error:.4f}  kl_topk={step.kl_topk:.3e}  "
              f"tie_slack_ulps={step.tie_slack_ulps:.1f}  agrees={step.agrees}  "
              f"mine_top1={step.mine_top1} oracle_top1={step.oracle_top1}")
        print(f"  judged against B1-R CALIBRATED_THRESHOLDS: passes={passed}")
        for r in reasons:
            print(f"    {r}")

        entry.update(
            {
                "first_diff_index": first_diff,
                "ref_token_text": tok.decode([ref_tokens[first_diff]]),
                "spec_token_text": tok.decode([spec_tokens[first_diff]]),
                "gap_error": step.gap_error,
                "kl_topk": step.kl_topk,
                "tie_slack_ulps": step.tie_slack_ulps,
                "step_agrees": step.agrees,
                "step_mine_top1": step.mine_top1,
                "step_oracle_top1": step.oracle_top1,
                "capture_gaps_missing_from_intersection": gaps,
                "metrics": metrics,
                "passes_calibrated_bars": passed,
                "reasons": list(reasons),
            }
        )
        results["prompts"][label] = entry

    out_path = Path(_ROOT) / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
