"""B3-b driver: run the K sweep, the teacher-forced head-quality probe, and
the code-prompt divergence gap-error check in ONE process, ONE model load.

Why one process: a cold load of the real 27B checkpoint measured 312s in
this session (``load_qwen36_model``, NVFP4 dequant + weight materialization)
-- reloading per experiment would spend more wall-clock on loading than on
the actual measurements. This script imports the three standalone,
independently-documented, independently-runnable probes
(``b3b_k_sweep.py``, ``b3b_teacher_forced_head_quality.py``,
``b3b_divergence_gap.py``) as modules and calls their pure functions
directly against ONE shared ``model``/``tok`` -- their own ``main()``
functions (each of which loads its own model) are never called here. Each
of those three files remains runnable standalone with its own docstring
explaining what it measures and why; this file is only the "run all three
against one warm model" convenience, not a fourth experiment design.

Run: ~/.venvs/vllm/bin/python scripts/b3b_run_all.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
sys.path.insert(0, str(Path(_ROOT) / "scripts"))
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__}"
)

import b3b_divergence_gap as dg  # noqa: E402
import b3b_k_sweep as ks  # noqa: E402
import b3b_teacher_forced_head_quality as tf  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from runtime.model_loading import load_qwen36_model  # noqa: E402

MODEL_PATH = ks.MODEL_PATH
DEVICE = ks.DEVICE
MAX_SEQ_LEN = 512

K_SWEEP_VALUES = [1, 2, 4, 8, 16]
K_SWEEP_N_TOKENS = 64
TEACHER_FORCED_N_TOKENS = 160
DIVERGENCE_K = 8
DIVERGENCE_N_TOKENS = 32


def run_k_sweep(model, tok) -> dict:
    print(f"\n{'#' * 70}\n# PART 1: K sweep\n{'#' * 70}")
    warm_ids = tok("Warm up the kernels before timing.", return_tensors=None)["input_ids"]
    ks.free_greedy_decode(model, warm_ids, 4)

    results: dict = {"n_tokens": K_SWEEP_N_TOKENS, "k_values": K_SWEEP_VALUES, "prompts": {}}
    baselines: dict[str, tuple[list[int], float]] = {}
    for label, text in ks.PROMPTS.items():
        prompt_ids = tok(text, return_tensors=None)["input_ids"]
        ref_tokens, ref_time = ks.free_greedy_decode(model, prompt_ids, K_SWEEP_N_TOKENS)
        baselines[label] = (ref_tokens, ref_time)
        print(f"baseline[{label}]: {K_SWEEP_N_TOKENS} tok in {ref_time:.3f}s "
              f"= {K_SWEEP_N_TOKENS / ref_time:.2f} tok/s")

    for k in K_SWEEP_VALUES:
        print(f"\n{'=' * 70}\nK={k}\n{'=' * 70}")
        try:
            ks.speculative_decode(model, warm_ids, min(2 * k, 8) + k, k)
        except Exception as exc:  # noqa: BLE001
            print(f"  K={k} warmup FAILED: {exc!r} -- skipping this K")
            results["prompts"].setdefault("__errors__", {})[str(k)] = repr(exc)
            continue

        results["prompts"].setdefault(str(k), {})
        for label, text in ks.PROMPTS.items():
            prompt_ids = tok(text, return_tensors=None)["input_ids"]
            ref_tokens, ref_time = baselines[label]
            try:
                spec_tokens, spec_time, rounds = ks.speculative_decode(
                    model, prompt_ids, K_SWEEP_N_TOKENS, k
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
            ref_tok_s = K_SWEEP_N_TOKENS / ref_time
            speedup = spec_tok_s / ref_tok_s if ref_tok_s else float("nan")
            hist = ks.reject_position_histogram(rounds, k)
            pos_acc = ks.position_accuracy(rounds, k)

            print(
                f"  [{label:6s}] rounds={len(rounds):3d} accept_rate={accept_rate:.3f} "
                f"mean_accepted/round={mean_accept_per_round:.2f} "
                f"spec_tok/s={spec_tok_s:.2f} speedup={speedup:.2f}x "
                f"match={match}{'' if match else f' (first_diff={first_diff})'}"
            )
            print(f"    reject_position_histogram: {hist}")
            print(f"    position_accuracy (p=0..{k - 1}): {[round(x, 3) for x in pos_acc]}")

            results["prompts"][str(k)][label] = {
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
    return results


def run_teacher_forced(model, tok) -> dict:
    print(f"\n{'#' * 70}\n# PART 2: teacher-forced draft head quality\n{'#' * 70}")
    results: dict = {"n_tokens": TEACHER_FORCED_N_TOKENS, "prompts": {}}
    for label, text in tf.PROMPTS.items():
        prompt_ids = tok(text, return_tensors=None)["input_ids"]
        print(f"\n=== {label!r} ({len(prompt_ids)} prompt tokens) ===")
        t0 = time.perf_counter()
        tokens, hiddens, top5 = tf.free_greedy_decode_with_hidden(
            model, prompt_ids, TEACHER_FORCED_N_TOKENS
        )
        print(f"  free greedy decode: {len(tokens)} tokens in {time.perf_counter() - t0:.1f}s")

        records = tf.teacher_forced_accuracy(model, tokens, hiddens)
        n = len(records)
        matches = sum(r["match"] for r in records)
        acc = matches / n if n else float("nan")
        misses = [r for r in records if not r["match"]]
        close_misses = sum(1 for r in misses if r["pred"] in top5[r["i"]][1:5])
        close_rate = close_misses / len(misses) if misses else float("nan")
        q = max(n // 4, 1)
        first_q_acc = sum(r["match"] for r in records[:q]) / q if n else float("nan")
        last_q_acc = sum(r["match"] for r in records[-q:]) / q if n else float("nan")

        print(f"  teacher-forced top-1 accuracy: {acc:.3f} ({matches}/{n})")
        print(f"  first-quarter acc: {first_q_acc:.3f}  last-quarter acc: {last_q_acc:.3f}")
        print(f"  among misses, head's guess was target's rank 2-5: "
              f"{close_rate:.3f} ({close_misses}/{len(misses)})")

        results["prompts"][label] = {
            "n_positions": n,
            "accuracy": acc,
            "first_quarter_accuracy": first_q_acc,
            "last_quarter_accuracy": last_q_acc,
            "close_miss_rate_rank2_5": close_rate,
            "records": records,
        }
    return results


def run_divergence_gap(model, tok) -> dict:
    print(f"\n{'#' * 70}\n# PART 3: code-prompt divergence, gap-error judged\n{'#' * 70}")
    warm_ids = tok("Warm up the kernels before timing.", return_tensors=None)["input_ids"]
    dg.free_greedy_decode_capturing(model, warm_ids, 4)
    dg.speculative_decode_capturing(model, warm_ids, 4 + DIVERGENCE_K, DIVERGENCE_K)

    results: dict = {"k": DIVERGENCE_K, "n_tokens": DIVERGENCE_N_TOKENS, "prompts": {}}
    for label, text in dg.PROMPTS.items():
        print(f"\n=== prompt={label!r} ===")
        prompt_ids = tok(text, return_tensors=None)["input_ids"]
        ref_tokens, seq_rows = dg.free_greedy_decode_capturing(
            model, prompt_ids, DIVERGENCE_N_TOKENS
        )
        spec_tokens, round_meta = dg.speculative_decode_capturing(
            model, prompt_ids, DIVERGENCE_N_TOKENS, DIVERGENCE_K
        )

        match = ref_tokens == spec_tokens
        entry: dict = {
            "tokens_match": match,
            "ref_tokens": ref_tokens,
            "spec_tokens": spec_tokens,
        }
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
        mine_row = dg.find_source_row(round_meta, first_diff)
        mine_map, oracle_map = dg._row_map(mine_row, oracle_row)
        gaps = dg.missing_from_intersection(mine_map, oracle_map, top_k=dg.GAP_TOP_K)
        step = dg.compare_step(
            0,
            ref_tokens[first_diff],
            mine_map,
            oracle_map,
            gap_top_k=dg.GAP_TOP_K,
            mine_logsumexp=mine_row.logsumexp,
            oracle_logsumexp=oracle_row.logsumexp,
        )
        workload = dg.WorkloadAgreement(
            workload_name=f"{label}_divergence", prompt_token_ids=(), steps=(step,)
        )
        report = dg.AgreementReport(workloads=(workload,))
        metrics = report.summary_metrics()
        passed, reasons = dg.evaluate_summary(metrics, dg.CALIBRATED_THRESHOLDS)

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
    return results


def main() -> None:
    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    t0 = time.perf_counter()
    model = load_qwen36_model(
        MODEL_PATH, device=DEVICE, max_seq_len=MAX_SEQ_LEN, enable_mtp=True
    )
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")

    all_results: dict = {"model_load_s": time.perf_counter() - t0}

    t_part = time.perf_counter()
    all_results["k_sweep"] = run_k_sweep(model, tok)
    print(f"\n[part 1 wall time: {time.perf_counter() - t_part:.1f}s]")

    t_part = time.perf_counter()
    all_results["teacher_forced"] = run_teacher_forced(model, tok)
    print(f"\n[part 2 wall time: {time.perf_counter() - t_part:.1f}s]")

    t_part = time.perf_counter()
    all_results["divergence_gap"] = run_divergence_gap(model, tok)
    print(f"\n[part 3 wall time: {time.perf_counter() - t_part:.1f}s]")

    out_path = Path(_ROOT) / ".bfdiag" / "runs" / "b3b_run_all.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
