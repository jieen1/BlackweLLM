"""Full-model B1-R gap-error check for the FP8 W8A8 pre-flight's activation
emulation (``QSR_EMULATE_FP8_ACTIVATION=1``, ``runtime/model/
compressed_tensors_linear.py::emulate_fp8_activation_round_trip``): if ONLY
the activation-quantization error a genuine W8A8 GEMM would add is injected
-- weights already dequantized exactly from the checkpoint's real FP8
values on both sides, GEMM still runs BF16xBF16 via the same ``F.linear``
today's production path uses, no accumulation-order change -- does the
standard checkpoint's FP8-layer footprint (self_attn q/k/v/o_proj, GDN's
in_proj_qkv/in_proj_z/out_proj, lm_head, and layers 56-63's MLP -- 233
calls/decode-step per ``notes/2026-08-03-decode-kernel-profile.md``) still
clear ``bfdiag.divergence.logit_agreement.CALIBRATED_THRESHOLDS``?

This is a **lower bound** on real W8A8's error, not an equivalent of it --
see ``emulate_fp8_activation_round_trip``'s docstring for exactly what error
source it does and does not model. So the read on this script's result is
asymmetric: a FAIL here is decisive (a real kernel can only be worse, so
real W8A8 cannot pass either); a PASS here is necessary but not sufficient
(a real kernel's own accumulation-order difference is untested and could
still push it over the bars).

Sibling of ``scripts/verify_nvfp4_w4a4_gemm_full_model_gap.py`` (same
oracle/candidate/step-locked/``CALIBRATED_THRESHOLDS`` harness), but
**simpler in one load-bearing way**: that script needed to monkeypatch
``Qwen36MLP.forward`` because production BYPASSES per-Linear ``forward()``
for NVFP4 MLPs (a fused ``run_w4a16_moe`` kernel call reads raw Parameters
directly). FP8-channel Linears have no such fusion -- every one of the 233
calls/step already goes through :meth:`CompressedTensorsFP8ChannelLinear.
forward` in production (verified directly: ``Qwen36Attention.forward``
calls ``self.q_proj(hidden_states)`` etc., ``Qwen36GatedDeltaNet.forward``
calls ``self.in_proj_qkv(...)`` etc., ``Qwen36CausalLM`` calls
``self.lm_head(...)``, and ``Qwen36MLP.forward``'s non-fused branch --
taken for layers 56-63, where ``self._nvfp4_fused`` is False because those
MLPs are FP8-channel, not NVFP4 -- is exactly ``down_proj(F.silu(gate_proj(x))
* up_proj(x))``, three more direct calls). So toggling
``QSR_EMULATE_FP8_ACTIVATION`` around an ordinary forward pass directly
toggles production behavior for exactly these 233 calls and nothing else
(NVFP4's fused MLP path, layers 0-55, uses a different Linear class this
flag never touches) -- no monkeypatching, no raw-Parameter-freeing/
``_keep_raw_nvfp4_weights`` bookkeeping, no dequant-cache dropping between
workloads (the weight-side BF16 cache does not depend on this flag at all).

Oracle: production forward with the flag unset -- today's B1-R-calibrated
default, free-running greedy decode. Candidate: the SAME loaded model,
forced through the oracle's own token trajectory, with the flag set to
``"1"`` for that trajectory only.

Run: PYTHONPATH=<this worktree> ~/.venvs/vllm/bin/python -u \\
    scripts/verify_fp8_w8a8_activation_emulation_full_model_gap.py [--steps N]

*** MUST be run with PYTHONPATH pointing at this worktree -- see
``scripts/verify_nvfp4_gemm_full_model_gap.py``'s docstring for why.
"""

from __future__ import annotations

import argparse
import json
import os
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
from runtime.model.compressed_tensors_linear import QSR_EMULATE_FP8_ACTIVATION_ENV  # noqa: E402
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
    ap.add_argument(
        "--out",
        type=str,
        default=".bfdiag/runs/fp8_w8a8_activation_emulation_full_model_gap.json",
    )
    ap.add_argument("--model-path", type=str, default=standard_checkpoint_path())
    args = ap.parse_args()
    model_path = args.model_path

    # This script is the one thing that flips QSR_EMULATE_FP8_ACTIVATION on
    # and off around each half of every workload -- a value already set in
    # the invoking shell would make the "oracle" run silently not be the
    # oracle. Fail loud rather than produce a quietly meaningless gap of 0.
    assert QSR_EMULATE_FP8_ACTIVATION_ENV not in os.environ, (
        f"{QSR_EMULATE_FP8_ACTIVATION_ENV} is already set in the environment -- unset it "
        "before running this script, which controls it itself"
    )

    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    print("model_path:", model_path)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    t0 = time.perf_counter()
    model = load_qwen36_model(
        model_path, device=DEVICE, max_seq_len=args.max_seq_len, enable_mtp=False
    )
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")
    print(f"allocated after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    workloads = []
    results: dict[str, object] = {"steps": args.steps, "workloads": {}}

    for label, text in PROMPTS.items():
        prompt_ids = tok(text, return_tensors=None)["input_ids"]
        print(f"\n=== {label!r} ({len(prompt_ids)} prompt tokens) ===")

        os.environ.pop(QSR_EMULATE_FP8_ACTIVATION_ENV, None)
        t = time.perf_counter()
        oracle_rows, tokens = sequential_trajectory(model, prompt_ids, args.steps)
        print(
            f"  oracle (today's default, no activation emulation): {len(oracle_rows)} "
            f"steps in {time.perf_counter() - t:.1f}s"
        )

        os.environ[QSR_EMULATE_FP8_ACTIVATION_ENV] = "1"
        try:
            t = time.perf_counter()
            cand_rows = forced_sequential_trajectory(model, prompt_ids, tokens)
            print(
                f"  candidate (FP8 activation round-trip emulated): {len(cand_rows)} "
                f"steps in {time.perf_counter() - t:.1f}s"
            )
        finally:
            os.environ.pop(QSR_EMULATE_FP8_ACTIVATION_ENV, None)

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
