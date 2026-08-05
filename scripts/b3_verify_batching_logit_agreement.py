"""B3: gap error at the logits, for MTP verify with the big GDN
projections batched -- measured against **our own non-speculative path**.

``docs/b1-correctness-criterion.md`` §7 sets B3's reference frame
explicitly: "参照系应当是我们自己的非投机路径", not HF. This script is
that comparison, run with the criterion's own R1 metric
(:func:`bfdiag.divergence.logit_agreement.compare_step`). The sequential
pairings are informational baselines because verify already uses an
approximate batched execution form; only the batched-vs-shipped pairing
is judged as the candidate's incremental footprint.

Three logit trajectories over the SAME token sequence, one model load:

* ``sequential`` -- ordinary greedy decode, one token per forward. The
  B1-proven path. This is the oracle for everything below.
* ``verify_shipped`` -- the same tokens replayed through
  :meth:`Qwen36TextModelSelfBuilt.verify_forward` in blocks of ``K``,
  with today's ``spec_forward`` (``in_proj_qkv``/``in_proj_z``/
  ``out_proj`` looped per position).
* ``verify_batched`` -- identical, except those three projections are
  batched over all ``K`` positions.

Step-locking is what makes the three comparable: every trajectory is
forced through the sequential path's own greedy tokens, so no side can
wander onto a different prefix, exactly as
``scripts/b1_forced_decode_agreement.py`` does for B1-R (and for the
same reason -- see that script's docstring §1). Here it also removes the
acceptance rate as a confound: ``commit_verify(accepted_count=K)`` after
every block means both verify runs traverse an identical schedule of
verify blocks.

The three pairings answer three different questions:

* ``verify_shipped`` vs ``sequential`` -- the gap this runtime **already
  ships and has already accepted** (verify's attention runs an
  extend-mode kernel, its MLP and layernorms run batched over K rows;
  ``notes/2026-08-02-b3-mtp-e2e-acceptance-throughput.md`` names this
  and does not treat it as a bug).
* ``verify_batched`` vs ``sequential`` -- the same quantity after the
  change. If this is not materially worse than the line above, the
  change costs nothing that was not already being spent.
* ``verify_batched`` vs ``verify_shipped`` -- the change's own footprint,
  isolated.

Run: ~/.venvs/vllm/bin/python -u scripts/b3_verify_batching_logit_agreement.py
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
from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.model.qwen36_model import Qwen36GatedDeltaNet  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402

sys.path.insert(0, str(Path(_ROOT) / "scripts"))
MODEL_PATH = standard_checkpoint_path()
DEVICE = torch.device("cuda")
torch.set_grad_enabled(False)

#: Same capture/compare widths B1-R's own harness uses, so the numbers
#: land in the same units as docs/b1-correctness-criterion.md §6.1.
CAPTURE_TOP_K = 64
GAP_TOP_K = 8
#: Top-k remains the compact representation for gap/KL reporting. The
#: candidate-vs-shipped full-logit check additionally retains FP32 CPU rows
#: for one prompt's three trajectories, then releases them before the next
#: prompt. That costs roughly 672 MiB at the default workload, but avoids
#: BF16 host-rounding dominating a cosine whose calibrated bar is 0.9995.
STORE_TOP_K = 1024

PROMPTS = {
    "prose": "Once upon a time, in a small village near the mountains,",
    "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "instruction": (
        "Explain, in plain language and without equations, why adding more "
        "layers to a neural network does not always improve its accuracy."
    ),
}


class TopKRow:
    """One step's logits, with a compact top-k view and FP32 CPU row."""

    __slots__ = ("indices", "values", "logsumexp", "full")

    def __init__(self, logits: torch.Tensor) -> None:
        row = logits.float()
        top = torch.topk(row, STORE_TOP_K)
        self.indices = top.indices.cpu()
        self.values = top.values.cpu()
        self.logsumexp = float(torch.logsumexp(row, dim=-1).item())
        self.full = logits.detach().to(device="cpu", dtype=torch.float32)

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


def agreement(name: str, mine: list[TopKRow], oracle: list[TopKRow], tokens: list[int]):
    records = []
    capture_gaps: list[int] = []
    for i, (m, o) in enumerate(zip(mine, oracle, strict=True)):
        mine_map, oracle_map = _row_map(m, o)
        capture_gaps.extend(missing_from_intersection(mine_map, oracle_map, top_k=GAP_TOP_K))
        records.append(
            compare_step(
                i,
                tokens[i],
                mine_map,
                oracle_map,
                gap_top_k=GAP_TOP_K,
                mine_logsumexp=m.logsumexp,
                oracle_logsumexp=o.logsumexp,
            )
        )
    return WorkloadAgreement(workload_name=name, prompt_token_ids=(), steps=tuple(records))


def full_logit_metrics(
    mine: list[TopKRow], oracle: list[TopKRow], tokens: list[int]
) -> tuple[float, float]:
    """Return forced-token NLL excess and worst full-vocabulary cosine.

    The token sequence is locked to the sequential oracle, so this measures
    exactly the extra likelihood cost and logit-vector drift caused by the
    verify form. It intentionally uses full rows rather than the top-k slices
    used by :func:`agreement`.
    """
    mine_nll = 0.0
    oracle_nll = 0.0
    min_cosine = 1.0
    for mine_row, oracle_row, token in zip(mine, oracle, tokens, strict=True):
        mine_values = mine_row.full.float()
        oracle_values = oracle_row.full.float()
        mine_nll += mine_row.logsumexp - float(mine_values[token].item())
        oracle_nll += oracle_row.logsumexp - float(oracle_values[token].item())
        min_cosine = min(
            min_cosine,
            float(torch.nn.functional.cosine_similarity(mine_values, oracle_values, dim=0).item()),
        )
    return (mine_nll / oracle_nll - 1.0, min_cosine)


# ---------------------------------------------------------------------------
# The three trajectories.
# ---------------------------------------------------------------------------


def sequential_trajectory(model, prompt_ids: list[int], n_steps: int):
    """Greedy decode ``n_steps + 1`` predictions, one token per forward.

    Returns ``(rows, tokens)``: ``rows[i]`` is the prediction made *before*
    consuming ``tokens[i]``. ``rows`` has ``n_steps + 1`` entries so the
    verify replay below (which predicts one position ahead of each draft
    token it consumes) has an oracle for every position it produces.
    """
    state = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    ids = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)
    hidden = model(ids, state)
    logits = model.compute_logits(hidden[:, -1:, :])[0, -1]
    rows: list[TopKRow] = []
    tokens: list[int] = []
    for _ in range(n_steps + 1):
        rows.append(TopKRow(logits))
        token = int(logits.argmax().item())
        tokens.append(token)
        step = torch.tensor([[token]], device=DEVICE, dtype=torch.long)
        hidden = model(step, state)
        logits = model.compute_logits(hidden[:, -1:, :])[0, -1]
    return rows, tokens


def verify_trajectory(model, prompt_ids: list[int], tokens: list[int], k: int):
    """Replay ``tokens`` through ``verify_forward`` in blocks of ``k``.

    Position alignment: a verify block fed ``tokens[j:j+k]`` predicts
    ``tokens[j+1 .. j+k]``, so its row ``t`` is the oracle's row
    ``j + t + 1``. Every block is committed with ``accepted_count=k`` --
    we are step-locking, not measuring acceptance.
    """
    state = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    ids = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)
    model(ids, state)

    rows: list[TopKRow] = []
    positions: list[int] = []
    j = 0
    while j + k <= len(tokens) - 1:
        block = torch.tensor([tokens[j : j + k]], device=DEVICE, dtype=torch.long)
        past_len = state.num_tokens_seen
        hidden, snapshots = model.verify_forward(block, state)
        logits = model.compute_logits(hidden)[0]  # [k, vocab]
        for t in range(k):
            rows.append(TopKRow(logits[t]))
            positions.append(j + t + 1)
        model.commit_verify(state, snapshots, past_len=past_len, accepted_count=k)
        j += k
    return rows, positions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--steps", type=int, default=224)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--out", type=str, default=".bfdiag/runs/b3_verify_batching_agreement.json")
    args = ap.parse_args()

    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    t0 = time.perf_counter()
    model = load_qwen36_model(
        MODEL_PATH, device=DEVICE, max_seq_len=args.max_seq_len, enable_mtp=False
    )
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")

    production = Qwen36GatedDeltaNet.spec_forward

    def spec_forward_shipped(self, hidden_states, state, **kwargs):
        return production(
            self,
            hidden_states,
            state,
            batch_large_projections=False,
            **kwargs,
        )

    def spec_forward_batched(self, hidden_states, state, **kwargs):
        return production(
            self,
            hidden_states,
            state,
            batch_large_projections=True,
            **kwargs,
        )
    results: dict[str, object] = {
        "k": args.k,
        "steps_requested": args.steps,
        "workloads": {},
    }
    per_pairing: dict[str, list[WorkloadAgreement]] = {
        "verify_shipped_vs_sequential": [],
        "verify_batched_vs_sequential": [],
        "verify_batched_vs_verify_shipped": [],
    }
    full_metrics: dict[str, list[tuple[float, float]]] = {
        "verify_shipped_vs_sequential": [],
        "verify_batched_vs_sequential": [],
        "verify_batched_vs_verify_shipped": [],
    }

    for label, text in PROMPTS.items():
        prompt_ids = tok(text, return_tensors=None)["input_ids"]
        print(f"\n=== {label!r} ({len(prompt_ids)} prompt tokens) ===")
        t = time.perf_counter()
        seq_rows, tokens = sequential_trajectory(model, prompt_ids, args.steps)
        print(f"  sequential: {len(seq_rows)} steps in {time.perf_counter() - t:.1f}s")

        Qwen36GatedDeltaNet.spec_forward = spec_forward_shipped
        t = time.perf_counter()
        ship_rows, positions = verify_trajectory(model, prompt_ids, tokens, args.k)
        print(f"  verify (shipped): {len(ship_rows)} positions in {time.perf_counter() - t:.1f}s")

        Qwen36GatedDeltaNet.spec_forward = spec_forward_batched
        t = time.perf_counter()
        batch_rows, positions_b = verify_trajectory(model, prompt_ids, tokens, args.k)
        print(f"  verify (batched): {len(batch_rows)} positions in {time.perf_counter() - t:.1f}s")
        Qwen36GatedDeltaNet.spec_forward = production
        assert positions == positions_b

        oracle_rows = [seq_rows[p] for p in positions]
        oracle_tokens = [tokens[p] for p in positions]
        pairs = {
            "verify_shipped_vs_sequential": (ship_rows, oracle_rows),
            "verify_batched_vs_sequential": (batch_rows, oracle_rows),
            "verify_batched_vs_verify_shipped": (batch_rows, ship_rows),
        }
        entry: dict[str, object] = {"positions": len(positions)}
        for pairing, (mine, oracle) in pairs.items():
            w = agreement(label, mine, oracle, oracle_tokens)
            per_pairing[pairing].append(w)
            nll_excess, min_cosine = full_logit_metrics(mine, oracle, oracle_tokens)
            full_metrics[pairing].append((nll_excess, min_cosine))
            entry[pairing] = w.to_dict()["summary"]
            entry[pairing]["nll_relative_excess"] = nll_excess
            entry[pairing]["min_logits_cosine"] = min_cosine
            print(
                f"  {pairing:<34} p90={w.p90_gap_error:.4f} p99={w.p99_gap_error:.4f} "
                f"med={w.median_gap_error:.4f} maxKL={w.mean_kl_topk:.3e} "
                f"flips={w.num_disagreements} slack={w.max_tie_slack_ulps:.1f}"
            )
        results["workloads"][label] = entry

    print("\n=== combined ===")
    summary: dict[str, object] = {}
    for pairing, workloads in per_pairing.items():
        nll_excess, min_cosine = zip(*full_metrics[pairing], strict=True)
        report = AgreementReport(
            workloads=tuple(workloads),
            nll_relative_excess=max(nll_excess),
            min_logits_cosine=min(min_cosine),
        )
        metrics = report.summary_metrics()
        if pairing == "verify_batched_vs_verify_shipped":
            passed, reasons = evaluate_summary(metrics, CALIBRATED_THRESHOLDS)
        else:
            passed, reasons = None, ("informational baseline comparison; not a candidate gate",)
        summary[pairing] = {
            "metrics": metrics,
            "passes_calibrated_bars": passed,
            "reasons": list(reasons),
            "total_steps": report.total_steps,
        }
        print(f"\n  {pairing}  ({report.total_steps} steps)")
        for key in (
            "median_gap_error",
            "p90_gap_error",
            "p99_gap_error",
            "max_gap_error",
            "mean_kl_topk",
            "max_tie_slack_ulps",
            "disagreement_rate",
            "p90_logprob_error",
            "max_drift_ratio",
        ):
            print(f"    {key:<24} {metrics.get(key)}")
        print(f"    -> passes calibrated bars: {passed}")
        for r in reasons:
            print(f"       {r}")
    results["summary"] = summary

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
