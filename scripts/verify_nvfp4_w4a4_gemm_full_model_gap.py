"""Full-model B1-R gap-error check for a genuine W4A4 NVFP4 MLP GEMM
(``sparkinfer.gemm.blockscaled.mm``, both operands pre-quantized) on the
standard checkpoint (``unsloth/Qwen3.6-27B-NVFP4``).

Sibling of ``scripts/verify_nvfp4_gemm_full_model_gap.py`` (same plumbing:
one model load, ``Qwen36MLP.forward`` monkeypatched per side, oracle vs
candidate compared with ``bfdiag.divergence.logit_agreement``'s
B1-R-calibrated bars) -- the difference is what the candidate forward does.
That script's candidate is the *production* W4A16 fused kernel
(``run_w4a16_moe``, weight-only -- dequantizes NVFP4 against the real BF16
activation, no activation-quantization error). This script's candidate
additionally quantizes the activation to NVFP4 (block size 16, using each
Linear's own checkpoint ``input_global_scale``) and calls
``sparkinfer.gemm.blockscaled.mm`` directly -- the checkpoint's OWN declared
W4A4 scheme for these 56 layers (``config_groups.group_1.input_activations``,
see ``runtime/loading/compressed_tensors.py``'s module docstring), not an
approximation forced onto a weight-only checkpoint the way the first
NVFP4-GEMM attempt on ``nvidia/Qwen3.6-27B-NVFP4`` was (see
``runtime/model/modelopt_linear.py``'s module docstring for why that one
was invalid on its own terms).

Oracle is unchanged from the sibling script: the legacy per-Linear BF16
dequant-and-cache path (``down_proj(F.silu(gate_proj(x)) * up_proj(x))``),
already the B1-R-calibrated baseline. Only NVFP4-fused MLPs whose
``gate_proj`` is :class:`CompressedTensorsNVFP4Linear` take the W4A4 path
(layers 0-55 on the standard checkpoint); every other MLP (layers 56-63,
FP8-channel) falls through to the same oracle-equivalent per-Linear
forward on both sides, so this measures the W4A4 substitution in isolation.

Run: PYTHONPATH=<this worktree> ~/.venvs/vllm/bin/python -u \\
    scripts/verify_nvfp4_w4a4_gemm_full_model_gap.py [--steps N]

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
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from bfdiag.divergence.logit_agreement import (  # noqa: E402
    CALIBRATED_THRESHOLDS,
    AgreementReport,
    evaluate_summary,
)
from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.model.compressed_tensors_linear import CompressedTensorsNVFP4Linear  # noqa: E402
from runtime.model.modelopt_linear import ModelOptNVFP4Linear  # noqa: E402
from runtime.model.qwen36_model import Qwen36MLP  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402

sys.path.insert(0, str(Path(_ROOT) / "scripts"))
from verify_nvfp4_gemm_full_model_gap import (  # noqa: E402
    _drop_dequant_caches,
    _legacy_per_linear_forward,
    forced_sequential_trajectory,
)
from verify_nvfp4_w4a4_gemm_single_layer import blockscaled_linear  # noqa: E402

sys.path.insert(0, str(Path(_ROOT) / "scripts"))
from b3_verify_batching_logit_agreement import (  # noqa: E402
    PROMPTS,
    agreement,
    sequential_trajectory,
)

DEVICE = torch.device("cuda")
torch.set_grad_enabled(False)


def _w4a4_candidate_forward(self: Qwen36MLP, x: torch.Tensor) -> torch.Tensor:
    """W4A4 candidate: genuine ``blockscaled.mm`` GEMM for both operands on
    every NVFP4-fused MLP backed by :class:`CompressedTensorsNVFP4Linear`
    (the only format with a real checkpoint ``input_global_scale``);
    identical to the oracle for everything else (FP8-channel layers 56-63,
    or any non-fused MLP), so those layers cannot contribute to the
    measured gap."""
    if not (self._nvfp4_fused and isinstance(self.gate_proj, CompressedTensorsNVFP4Linear)):
        return _legacy_per_linear_forward(self, x)

    orig_shape = x.shape
    x2d = x.reshape(-1, self.hidden_size).contiguous()

    gate_w, gate_scale, gate_gs, gate_igs = self.gate_proj.nvfp4_w4a4_components_for_fuse()
    up_w, up_scale, up_gs, up_igs = self.up_proj.nvfp4_w4a4_components_for_fuse()
    down_w, down_scale, down_gs, down_igs = self.down_proj.nvfp4_w4a4_components_for_fuse()

    gate_out = blockscaled_linear(
        x2d, gate_w, gate_scale, gate_gs, gate_igs, self.intermediate_size, self.hidden_size
    )
    up_out = blockscaled_linear(
        x2d, up_w, up_scale, up_gs, up_igs, self.intermediate_size, self.hidden_size
    )
    inter = (F.silu(gate_out.float()) * up_out.float()).to(torch.bfloat16)
    down_out = blockscaled_linear(
        inter, down_w, down_scale, down_gs, down_igs, self.hidden_size, self.intermediate_size
    )
    return down_out.reshape(*orig_shape[:-1], self.hidden_size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--out", type=str, default=".bfdiag/runs/nvfp4_w4a4_gemm_full_model_gap.json")
    ap.add_argument("--model-path", type=str, default=standard_checkpoint_path())
    args = ap.parse_args()
    model_path = args.model_path

    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    print("model_path:", model_path)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    t0 = time.perf_counter()
    model = load_qwen36_model(
        model_path, device=DEVICE, max_seq_len=args.max_seq_len, enable_mtp=False
    )
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")
    print(f"allocated after load: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

    n_w4a4 = 0
    n_other_fused = 0
    for m in model.modules():
        if isinstance(m, Qwen36MLP) and m._nvfp4_fused:
            m._keep_raw_nvfp4_weights = True
            if isinstance(m.gate_proj, CompressedTensorsNVFP4Linear):
                n_w4a4 += 1
            elif isinstance(m.gate_proj, ModelOptNVFP4Linear):
                n_other_fused += 1
    print(
        f"W4A4-candidate-eligible fused Qwen36MLP instances: {n_w4a4} "
        f"(modelopt-fused: {n_other_fused})"
    )
    if n_w4a4 == 0:
        raise SystemExit(
            "no CompressedTensorsNVFP4Linear-backed fused MLP found -- is --model-path "
            "really the standard (unsloth) checkpoint?"
        )

    workloads = []
    results: dict[str, object] = {"steps": args.steps, "workloads": {}}

    for label, text in PROMPTS.items():
        prompt_ids = tok(text, return_tensors=None)["input_ids"]
        print(f"\n=== {label!r} ({len(prompt_ids)} prompt tokens) ===")

        Qwen36MLP.forward = _legacy_per_linear_forward
        t = time.perf_counter()
        oracle_rows, tokens = sequential_trajectory(model, prompt_ids, args.steps)
        print(
            f"  oracle (legacy per-Linear dequant): {len(oracle_rows)} steps in "
            f"{time.perf_counter() - t:.1f}s"
        )
        _drop_dequant_caches(model)

        Qwen36MLP.forward = _w4a4_candidate_forward
        t = time.perf_counter()
        cand_rows = forced_sequential_trajectory(model, prompt_ids, tokens)
        print(
            f"  candidate (W4A4 blockscaled): {len(cand_rows)} steps in "
            f"{time.perf_counter() - t:.1f}s"
        )
        _drop_dequant_caches(model)

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
