"""Full-model B1-R gap-error check for FP8 KV cache (2026-08-03 follow-up,
``enable_fp8_kv``/``QSR_QWEN36_FP8_KV``, see ``runtime/model/qwen36_model.py``'s
module docstring): does replacing the full-attention layers' BF16 KV cache
with FP8-e4m3 (using the standard checkpoint's real per-layer ``k_scale``/
``v_scale``) stay inside B1-R's calibrated tolerance
(``docs/b1-correctness-criterion.md`` §6.1,
``bfdiag.divergence.logit_agreement.CALIBRATED_THRESHOLDS``)?

Adapted from ``scripts/verify_fp8_w8a8_activation_emulation_full_model_gap.py``
(same oracle/candidate/step-locked harness shape, same
``CALIBRATED_THRESHOLDS`` judgement, same capture-window-overflow hard-fail
handling -- kept verbatim, not re-derived, per that script's own comment on
why: two prior investigations (W4A4, FP8 W8A8) died exactly at a workload
overflowing the top-1024 capture window, and the bars below are computed on
survivors only unless that is treated as a hard fail explicitly).

**One structural difference from that script, forced by what FP8 KV
actually changes**: the W8A8 activation-emulation flag
(``QSR_EMULATE_FP8_ACTIVATION``) is read inside ``forward()`` at call time,
so oracle and candidate could share ONE loaded model instance, toggled
around each half of a workload. FP8 KV is NOT like that -- ``enable_fp8_kv``
changes the model's own structure at construction time (a real ``k_scale``/
``v_scale`` Parameter per full-attention layer, an FP8-dtype KV cache
buffer instead of BF16), so oracle (``enable_fp8_kv=False``, today's
shipped default) and candidate (``enable_fp8_kv=True``) are necessarily TWO
SEPARATE model instances built from the SAME checkpoint. Loading both at
once does not fit one GPU (each is 50+ GiB resident) -- so this script runs
in two phases instead: load the oracle, capture every workload's
free-running trajectory and tokens, free it, THEN load the candidate and
replay every workload's already-chosen tokens through it
(``forced_sequential_trajectory``, step-locked, same ``rows[i]`` =
prediction-before-consuming-``tokens[i]`` indexing as
``sequential_trajectory`` so the two are directly comparable position by
position). ``TopKRow`` only ever holds ``.cpu()`` tensors (see
``scripts/b3_verify_batching_logit_agreement.py``), so nothing from phase 1
pins the oracle model's GPU memory once ``del`` + ``torch.cuda.empty_cache()``
runs.

Run: PYTHONPATH=<this worktree> ~/.venvs/vllm/bin/python -u \\
    scripts/verify_fp8_kv_full_model_gap.py [--steps N]

*** MUST be run with PYTHONPATH pointing at this worktree -- see
``scripts/verify_nvfp4_gemm_full_model_gap.py``'s docstring for why.
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
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__} "
    f"-- rerun with PYTHONPATH={_ROOT}"
)

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from bfdiag.divergence.logit_agreement import (  # noqa: E402
    CALIBRATED_THRESHOLDS,
    AgreementReport,
    evaluate_summary,
)
from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402

sys.path.insert(0, str(Path(_ROOT) / "scripts"))
from b3_verify_batching_logit_agreement import (  # noqa: E402
    PROMPTS,
    agreement,
    sequential_trajectory,
)
from verify_nvfp4_gemm_full_model_gap import forced_sequential_trajectory  # noqa: E402

DEVICE = torch.device("cuda")
torch.set_grad_enabled(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--out", type=str, default=".bfdiag/runs/fp8_kv_full_model_gap.json")
    ap.add_argument("--model-path", type=str, default=standard_checkpoint_path())
    args = ap.parse_args()
    model_path = args.model_path

    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    print("model_path:", model_path)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # ---- Phase 1: oracle (enable_fp8_kv=False -- today's shipped BF16 KV
    # path), free-running greedy decode, one model load for every workload.
    t0 = time.perf_counter()
    oracle_model = load_qwen36_model(
        model_path,
        device=DEVICE,
        max_seq_len=args.max_seq_len,
        enable_mtp=False,
        enable_fp8_kv=False,
    )
    print(f"oracle model loaded in {time.perf_counter() - t0:.1f}s")
    print(f"allocated after oracle load: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    oracle_rows_by_label = {}
    tokens_by_label = {}
    prompt_ids_by_label = {}
    for label, text in PROMPTS.items():
        prompt_ids = tok(text, return_tensors=None)["input_ids"]
        print(f"\n=== oracle {label!r} ({len(prompt_ids)} prompt tokens) ===")
        t = time.perf_counter()
        oracle_rows, tokens = sequential_trajectory(oracle_model, prompt_ids, args.steps)
        print(
            f"  oracle (BF16 KV, today's default): {len(oracle_rows)} steps in "
            f"{time.perf_counter() - t:.1f}s"
        )
        oracle_rows_by_label[label] = oracle_rows
        tokens_by_label[label] = tokens
        prompt_ids_by_label[label] = prompt_ids

    del oracle_model
    torch.cuda.empty_cache()
    print(f"\nfreed oracle model; allocated now: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    # ---- Phase 2: candidate (enable_fp8_kv=True), step-locked through the
    # oracle's own token trajectory, one model load for every workload.
    t0 = time.perf_counter()
    candidate_model = load_qwen36_model(
        model_path,
        device=DEVICE,
        max_seq_len=args.max_seq_len,
        enable_mtp=False,
        enable_fp8_kv=True,
    )
    print(f"candidate model loaded in {time.perf_counter() - t0:.1f}s")
    print(f"allocated after candidate load: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    workloads = []
    results: dict[str, object] = {"steps": args.steps, "workloads": {}}

    for label in PROMPTS:
        prompt_ids = prompt_ids_by_label[label]
        tokens = tokens_by_label[label]
        oracle_rows = oracle_rows_by_label[label]
        print(f"\n=== candidate {label!r} ({len(prompt_ids)} prompt tokens) ===")
        t = time.perf_counter()
        cand_rows = forced_sequential_trajectory(candidate_model, prompt_ids, tokens)
        print(
            f"  candidate (FP8-e4m3 KV): {len(cand_rows)} steps in {time.perf_counter() - t:.1f}s"
        )

        try:
            w = agreement(label, cand_rows, oracle_rows, tokens)
        except AssertionError as exc:
            # The candidate diverged so far from the oracle that a token
            # left one side's stored top-1024 capture window entirely --
            # itself evidence of a much larger disagreement than anything
            # B1-R's calibration sweep produced (docs/b1-correctness-
            # criterion.md §5.3: even the strongest injected bug stayed
            # inside top-64). Record and move on instead of losing every
            # already-measured workload to one crash.
            print(f"  gap_error: CAPTURE WINDOW OVERFLOW ({exc}) -- treating as a hard fail")
            results["workloads"][label] = {"capture_overflow": str(exc)}
            continue
        workloads.append(w)
        results["workloads"][label] = w.to_dict()["summary"]
        print(
            f"  gap_error: p50={w.median_gap_error:.4f} p90={w.p90_gap_error:.4f} "
            f"p99={w.p99_gap_error:.4f} max={w.max_gap_error:.4f}  "
            f"flips={w.num_disagreements}/{w.num_steps} "
            f"tie_slack={w.max_tie_slack_ulps:.1f}ULP meanKL={w.mean_kl_topk:.3e}"
        )

    print(
        "\n=== combined, judged against CALIBRATED_THRESHOLDS "
        "(docs/b1-correctness-criterion.md §6.1) ==="
    )
    report = AgreementReport(workloads=tuple(workloads))
    metrics = report.summary_metrics()
    passed, reasons = evaluate_summary(metrics, CALIBRATED_THRESHOLDS)

    # A workload that overflowed the capture window contributed NOTHING to the
    # metrics above -- it was skipped before `workloads.append`. So every bar
    # here was judged on the survivors, with the worst case excluded by
    # construction. Without this, a run where one workload diverges beyond
    # measurement and the rest stay inside their bars would record
    # `passes_calibrated_bars: true`: a false green produced by the very
    # divergence the gate exists to catch.
    overflowed = sorted(
        label for label, w in results["workloads"].items() if "capture_overflow" in w
    )
    if overflowed:
        passed = False
        reasons = (
            *reasons,
            f"{len(overflowed)} workload(s) overflowed the top-1024 capture window "
            f"({', '.join(overflowed)}) and are NOT represented in any metric above -- "
            "the bars shown were computed on the survivors only. Diverging past "
            "measurability is a harder failure than exceeding a bar: B1-R's own "
            "calibration sweep never produced it from any injected bug "
            "(docs/b1-correctness-criterion.md §5.3, all stayed inside top-64).",
        )
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
        bar_attr = {
            "median_gap_error": "median_gap_error",
            "p90_gap_error": "p90_gap_error",
            "p99_gap_error": "p99_gap_error",
            "max_gap_error": "max_gap_error",
            "mean_kl_topk": "max_mean_kl_topk",
            "max_tie_slack_ulps": "max_tie_slack_ulps",
            "disagreement_rate": "max_disagreement_rate",
            "p90_logprob_error": "p90_logprob_error",
            "max_drift_ratio": "max_drift_ratio",
        }[key]
        bar = getattr(CALIBRATED_THRESHOLDS, bar_attr)
        print(f"    {key:<20} measured={metrics.get(key):<12} bar={bar}")
    # Worst-workload mean_kl_topk specifically (report §4 asks for this,
    # not just the combined-report figure metrics.get('mean_kl_topk')
    # already prints above -- print both so the worst single workload is
    # visible even if it isn't what summary_metrics aggregates).
    if workloads:
        worst = max(workloads, key=lambda w: w.mean_kl_topk)
        print(
            f"    worst-workload mean_kl_topk = {worst.mean_kl_topk:.3e} ({worst.workload_name!r})"
        )
    print(f"\n  total_steps={report.total_steps}  -> passes CALIBRATED_THRESHOLDS bars: {passed}")
    for r in reasons:
        print(f"    FAIL: {r}")

    results["summary"] = {
        "metrics": metrics,
        "passes_calibrated_bars": passed,
        "reasons": list(reasons),
        "total_steps": report.total_steps,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
