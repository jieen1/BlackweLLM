"""Full-model B1-R gap-error check for the NVFP4-GEMM change
(``work/nvfp4-gemm-20260802``): does replacing ``ModelOptNVFP4Linear``'s
BF16-dequant-cache forward with a real block-scaled NVFP4xNVFP4 GEMM
(``sparkinfer.gemm.blockscaled.mm``) stay inside B1-R's calibrated
tolerance (``docs/b1-correctness-criterion.md`` §6.1,
``bfdiag.divergence.logit_agreement.CALIBRATED_THRESHOLDS``)?

Reuses ``scripts/b3_verify_batching_logit_agreement.py``'s exact
plumbing (``TopKRow``, ``agreement``, ``sequential_trajectory``,
``PROMPTS``, ``MODEL_PATH``) rather than reinventing it -- one model
load, two forward paths on the SAME instance:

* ``oracle`` -- ``ModelOptNVFP4Linear``'s legacy dequant-to-BF16 path
  (``_ensure_ready()`` + ``F.linear``), the path B1-R already calibrated
  against HF. Free-runs greedily (this is the trusted baseline).
* ``candidate`` -- the new direct NVFP4xNVFP4 GEMM path (this worktree's
  actual ``forward()``), step-locked: forced through the oracle's own
  greedy tokens one at a time (``forced_sequential_trajectory``, same
  ``rows[i]`` = prediction-before-consuming-``tokens[i]`` indexing as
  ``sequential_trajectory`` so the two are directly comparable position
  by position).

Both paths run on ONE loaded set of quantized weights (no reload) --
the legacy path is switched in/out via a class-level monkeypatch of
``ModelOptNVFP4Linear.forward``, and its BF16 dequant cache is dropped
(``_weight_bf16 = None`` on every instance + ``torch.cuda.empty_cache()``)
before the candidate run, matching this class's own "opt-in, not
resident" contract (see ``runtime/model/modelopt_linear.py`` module
docstring).

Run: PYTHONPATH=<this worktree> ~/.venvs/vllm/bin/python -u \\
    scripts/verify_nvfp4_gemm_full_model_gap.py [--steps N]

*** MUST be run with PYTHONPATH pointing at this worktree, not plain
``python scripts/....py`` -- the venv's editable install of BlackweLLM
resolves ``runtime`` to the MAIN worktree otherwise (script dir goes
first on sys.path, not cwd; found the hard way, 2026-08-03, see this
worktree's own history). The assertion right after importing ``runtime``
below is exactly this check, so a wrong invocation fails loud in the
first line of output rather than silently grading the wrong code.
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
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from bfdiag.divergence.logit_agreement import (  # noqa: E402
    CALIBRATED_THRESHOLDS,
    AgreementReport,
    evaluate_summary,
)
from runtime.model.modelopt_linear import ModelOptNVFP4Linear  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402

sys.path.insert(0, str(Path(_ROOT) / "scripts"))
from b3_verify_batching_logit_agreement import (  # noqa: E402
    MODEL_PATH,
    PROMPTS,
    TopKRow,
    agreement,
    sequential_trajectory,
)

DEVICE = torch.device("cuda")
torch.set_grad_enabled(False)

_NEW_FORWARD = ModelOptNVFP4Linear.forward


def _legacy_dequant_forward(self: ModelOptNVFP4Linear, x: torch.Tensor) -> torch.Tensor:
    """The B1-R-calibrated baseline: dequantize to BF16 once, plain F.linear."""
    self._ensure_ready()
    return F.linear(x, self._weight_bf16, self.bias)


def _drop_dequant_caches(model: torch.nn.Module) -> None:
    n = 0
    for m in model.modules():
        if isinstance(m, ModelOptNVFP4Linear) and m._weight_bf16 is not None:
            m._weight_bf16 = None
            n += 1
    torch.cuda.empty_cache()
    print(f"  dropped {n} legacy BF16 dequant caches")


def forced_sequential_trajectory(model, prompt_ids: list[int], tokens: list[int]):
    """Like ``sequential_trajectory``, but forced through an already-chosen
    ``tokens`` sequence instead of the model's own greedy argmax -- so
    ``rows[i]`` lines up with ``sequential_trajectory``'s ``rows[i]``
    (prediction made *before* consuming ``tokens[i]``) position for
    position, on a genuinely identical input trajectory."""
    state = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    ids = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)
    hidden = model(ids, state)
    logits = model.compute_logits(hidden[:, -1:, :])[0, -1]
    rows: list[TopKRow] = []
    for tok in tokens[:-1]:
        rows.append(TopKRow(logits))
        step = torch.tensor([[tok]], device=DEVICE, dtype=torch.long)
        hidden = model(step, state)
        logits = model.compute_logits(hidden[:, -1:, :])[0, -1]
    rows.append(TopKRow(logits))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument(
        "--out", type=str, default=".bfdiag/runs/nvfp4_gemm_full_model_gap.json"
    )
    args = ap.parse_args()

    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    print(
        "ModelOptNVFP4Linear.forward is the new GEMM path:",
        ModelOptNVFP4Linear.forward is _NEW_FORWARD,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    t0 = time.perf_counter()
    model = load_qwen36_model(
        MODEL_PATH, device=DEVICE, max_seq_len=args.max_seq_len, enable_mtp=False
    )
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")
    print(f"allocated after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    workloads = []
    results: dict[str, object] = {"steps": args.steps, "workloads": {}}

    for label, text in PROMPTS.items():
        prompt_ids = tok(text, return_tensors=None)["input_ids"]
        print(f"\n=== {label!r} ({len(prompt_ids)} prompt tokens) ===")

        # Oracle: legacy BF16-dequant path, free-running greedy decode --
        # this is the already-B1-R-calibrated baseline.
        ModelOptNVFP4Linear.forward = _legacy_dequant_forward
        t = time.perf_counter()
        oracle_rows, tokens = sequential_trajectory(model, prompt_ids, args.steps)
        print(
            f"  oracle (legacy dequant): {len(oracle_rows)} steps in "
            f"{time.perf_counter() - t:.1f}s"
        )
        _drop_dequant_caches(model)

        # Candidate: real NVFP4xNVFP4 GEMM path, forced through the
        # oracle's own tokens (step-locked).
        ModelOptNVFP4Linear.forward = _NEW_FORWARD
        t = time.perf_counter()
        cand_rows = forced_sequential_trajectory(model, prompt_ids, tokens)
        print(f"  candidate (NVFP4 GEMM): {len(cand_rows)} steps in {time.perf_counter() - t:.1f}s")

        w = agreement(label, cand_rows, oracle_rows, tokens)
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
