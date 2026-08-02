"""Phase 2 of the B1-R criterion: step-locked agreement + injected-bug sweep.

Consumes the reference trajectory written by
``scripts/b1_reference_trajectory.py`` and, for each requested
configuration, drives **our** model through HF's token sequence one token
at a time, comparing full logit vectors at every step. See
``docs/b1-correctness-criterion.md`` for the criterion itself and
``bfdiag/divergence/logit_agreement.py`` for the metrics.

Two things about this script are load-bearing and easy to get wrong.

**1. Step-locking is not the teacher forcing that was ruled out.**
``notes/2026-08-02-b1-greedy-alignment-fails.md`` concluded that "任何'喂
进已知前缀看下一个 token'的验证方法对 Qwen3.6 都无效", because
``runtime/model/qwen36_model.py``'s GDN layer dispatches ``seq_len == 1``
with existing state to FLA's ``fused_recurrent_gated_delta_rule`` and
everything else to ``chunk_gated_delta_rule``, and the two differ by up to
1.94 in logits (~30 bf16 ULP) while an argmax tie is 1-2 ULP. That is
correct, and it kills the specific method it was measured on: re-prefilling
``prompt + oracle[:i]`` in ONE forward and treating the result as a proxy
for what the *incremental decode* path produced. Those are two different
algorithms; of course they disagree.

What this script does is different in exactly the way that matters. It
never re-prefills. It prefills the prompt once and then advances one token
per call, so every step takes the ``seq_len == 1`` branch -- the production
decode path, unchanged. Feeding a token that is not our own argmax changes
no dispatch, no shape and no kernel; it only changes which row of the
embedding table is read. HF's ``Qwen3_5GatedDeltaNet`` makes the same
dispatch on its own side, so both sides are on the recurrent algorithm at
every compared step. ``--selfcheck`` proves this empirically rather than by
argument: forcing our model with *its own* greedy tokens must reproduce its
free-running logits bit for bit.

**2. The control must be run inside the sweep, not remembered from before.**
Every configuration is measured against one loaded model with the
injection applied and removed around it, so a leaked patch would
contaminate later runs. The control is therefore re-measured as the first
configuration of every sweep, and its numbers are what the injected
configurations are read against.

Run with:
    ~/.venvs/vllm/bin/python scripts/b1_forced_decode_agreement.py [options]

    --steps N            steps per workload (default: all available)
    --workloads a,b      subset of workloads
    --configs a,b,c      injection specs (default: the standard sweep)
    --selfcheck          also prove forced == free-running on our own tokens
    --freerun            also record free-running first-divergence indices
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _REPO_ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_REPO_ROOT), (
    f"imported runtime from {runtime.__file__}, expected under {_REPO_ROOT}"
)

import torch  # noqa: E402

from bfdiag.divergence.logit_agreement import (  # noqa: E402
    UNCALIBRATED_THRESHOLDS,
    AgreementReport,
    WorkloadAgreement,
    compare_step,
    missing_from_intersection,
    passes_b1r_gate,
)
from bfdiag.divergence.qwen36_bug_injection import injected, parse_injection  # noqa: E402
from bfdiag.divergence.qwen36_capture import Qwen36EngineCaptureSource  # noqa: E402
from bfdiag.divergence.scan import scan_layers  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402

DEVICE = torch.device("cuda")
MAX_SEQ_LEN = 2048

#: Captured per side before taking the union. Wide enough that the union
#: almost always contains both sides' plausible candidates; anything that
#: still falls outside is reported by name, never silently dropped.
CAPTURE_TOP_K = 64
#: Only tokens that could plausibly become the argmax enter the gap error.
GAP_TOP_K = 8

DEFAULT_REFERENCE = Path(_REPO_ROOT) / ".bfdiag" / "runs" / "b1_reference_trajectory.pt"
DEFAULT_OUTPUT_DIR = Path(_REPO_ROOT) / ".bfdiag" / "runs" / "b1_agreement"

#: The standard sweep. Control first (see the module docstring), then one
#: structural bug to anchor the far end, then two continuous knobs walked
#: down by decades so the criterion's detection floor is measured rather
#: than asserted.
DEFAULT_CONFIGS: tuple[str, ...] = (
    "none",
    "drop-q-norm",
    "rope-theta-rel:1e-2",
    "rope-theta-rel:1e-3",
    "rope-theta-rel:1e-4",
    "rope-theta-rel:1e-5",
    "gdn-state-stale-every:1",
    "gdn-state-stale-every:8",
    "gdn-state-stale-every:64",
    "gdn-state-decay:1e-3",
    "gdn-state-decay:1e-4",
)


def _logits_for(model, token_ids: torch.Tensor, state) -> torch.Tensor:
    hidden = model(token_ids, state)
    return model.compute_logits(hidden[:, -1:, :])[0, -1]


def forced_decode_logits(model, prompt_ids: list[int], forced_tokens: list[int]):
    """Yield our logits at every step while consuming ``forced_tokens``.

    Step ``i``'s logits are the prediction *for* ``forced_tokens[i]``; the
    token itself is then fed in to produce step ``i + 1``. Only
    ``forced_tokens[:-1]`` is ever consumed, so the last step's prediction
    is compared but its (unknown) successor is never needed.
    """
    state = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    prompt = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)
    yield _logits_for(model, prompt, state)
    for token in forced_tokens[:-1]:
        step = torch.tensor([[token]], device=DEVICE, dtype=torch.long)
        yield _logits_for(model, step, state)


def free_greedy_tokens(model, prompt_ids: list[int], max_new_tokens: int, eos_id: int):
    """Our own free-running greedy generation, with its logits."""
    state = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    prompt = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)
    logits = _logits_for(model, prompt, state)
    tokens: list[int] = []
    rows: list[torch.Tensor] = []
    for _ in range(max_new_tokens):
        rows.append(logits.clone())
        token = int(logits.argmax().item())
        tokens.append(token)
        if token == eos_id or len(tokens) >= max_new_tokens:
            break
        step = torch.tensor([[token]], device=DEVICE, dtype=torch.long)
        logits = _logits_for(model, step, state)
    return tokens, rows


def measure_workload(
    model, name: str, entry: dict, steps: int | None
) -> tuple[WorkloadAgreement, list[int]]:
    """Step-locked comparison of one workload. Returns the agreement plus
    any token ids that fell outside one side's captured top-K."""
    prompt_ids = list(entry["prompt_ids"])
    forced = list(entry["tokens"])
    if steps is not None:
        forced = forced[:steps]
    reference = entry["logits"][: len(forced)].to(DEVICE)
    ref_lse = entry["logsumexp"][: len(forced)]

    records = []
    capture_gaps: list[int] = []
    for index, mine_row in enumerate(forced_decode_logits(model, prompt_ids, forced)):
        if index >= len(forced):
            break
        ref_row = reference[index]
        mine_idx = torch.topk(mine_row, CAPTURE_TOP_K).indices
        ref_idx = torch.topk(ref_row, CAPTURE_TOP_K).indices
        union = torch.unique(torch.cat([mine_idx, ref_idx]))
        ids = union.tolist()
        mine_map = dict(zip(ids, mine_row[union].float().tolist(), strict=True))
        ref_map = dict(zip(ids, ref_row[union].float().tolist(), strict=True))
        capture_gaps.extend(missing_from_intersection(mine_map, ref_map, top_k=GAP_TOP_K))
        records.append(
            compare_step(
                index,
                forced[index],
                mine_map,
                ref_map,
                gap_top_k=GAP_TOP_K,
                mine_logsumexp=float(torch.logsumexp(mine_row.float(), dim=-1).item()),
                oracle_logsumexp=float(ref_lse[index].item()),
            )
        )
    del reference
    return (
        WorkloadAgreement(
            workload_name=name,
            prompt_token_ids=tuple(prompt_ids),
            steps=tuple(records),
        ),
        sorted(set(capture_gaps)),
    )


def layer_scan(model, name: str, entry: dict) -> dict:
    """Per-layer prefill cosine scan against HF's captured activations."""
    mine_trace = Qwen36EngineCaptureSource(model=model).capture(list(entry["prompt_ids"]))
    mine_cpu = {
        layer: {k: v.detach().cpu().clone() for k, v in sub.items()}
        for layer, sub in mine_trace.items()
    }
    report = scan_layers(entry["hidden"], mine_cpu)
    worst = min(
        (s for layer in report.layers for s in layer.submodules),
        key=lambda s: s.cosine_similarity,
        default=None,
    )
    return {
        "workload": name,
        "first_divergent_layer": report.first_divergent_layer,
        "first_divergent_submodules": list(report.first_divergent_submodules),
        "worst_cosine": None if worst is None else worst.cosine_similarity,
        "worst_submodule": None if worst is None else worst.submodule,
    }


def measure_nll(model, token_ids: list[int]) -> tuple[float, int]:
    state = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    ids = torch.tensor([token_ids], device=DEVICE, dtype=torch.long)
    hidden = model(ids, state)
    logits = model.compute_logits(hidden)[0]
    targets = torch.tensor(token_ids[1:], device=DEVICE, dtype=torch.long)
    log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
    picked = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return float(-picked.sum().item()), int(targets.numel())


def run_config(
    model, reference: dict, spec, workloads: list[str], steps: int | None, output_dir: Path
) -> dict:
    started = time.time()
    with injected(model, spec):
        agreements = []
        capture_gaps: list[int] = []
        scans = []
        for name in workloads:
            entry = reference["workloads"][name]
            agreement, gaps = measure_workload(model, name, entry, steps)
            agreements.append(agreement)
            capture_gaps.extend(gaps)
            scans.append(layer_scan(model, name, entry))
        nll_total, nll_count = measure_nll(model, list(reference["nll"]["token_ids"]))

    hf_nll_mean = float(reference["nll"]["mean"])
    our_nll_mean = nll_total / nll_count
    report = AgreementReport(
        workloads=tuple(agreements),
        metadata={
            "injection": str(spec),
            "steps_per_workload": steps,
            "capture_top_k": CAPTURE_TOP_K,
            "gap_top_k": GAP_TOP_K,
            "tokens_outside_capture": capture_gaps,
            "layer_scan": scans,
            "nll_mean_ours": our_nll_mean,
            "nll_mean_hf": hf_nll_mean,
            "nll_relative_excess": (our_nll_mean - hf_nll_mean) / hf_nll_mean,
            "ppl_ours": float(torch.tensor(our_nll_mean).exp()),
            "ppl_hf": float(torch.tensor(hf_nll_mean).exp()),
            "elapsed_s": time.time() - started,
        },
    )
    passed, reasons = passes_b1r_gate(report, UNCALIBRATED_THRESHOLDS)
    report.metadata["uncalibrated_gate_passed"] = passed
    report.metadata["uncalibrated_gate_reasons"] = list(reasons)

    safe = str(spec).replace(":", "_").replace(".", "p")
    report.save(output_dir / f"{safe}.json")
    return {
        "injection": str(spec),
        "max_gap_error": report.max_gap_error,
        "p99_gap_error": report.p99_gap_error,
        "max_tie_slack_ulps": report.max_tie_slack_ulps,
        "disagreement_rate": report.disagreement_rate,
        "mean_kl_topk": report.mean_kl_topk,
        "max_logprob_error": report.max_logprob_error,
        "max_drift_ratio": report.max_drift_ratio,
        "nll_relative_excess": report.metadata["nll_relative_excess"],
        "ppl_ours": report.metadata["ppl_ours"],
        "first_divergent_layer": [s["first_divergent_layer"] for s in scans],
        "worst_cosine": min((s["worst_cosine"] for s in scans if s["worst_cosine"]), default=None),
        "tokens_outside_capture": len(capture_gaps),
        "elapsed_s": report.metadata["elapsed_s"],
    }


def selfcheck(model, reference: dict, workloads: list[str], steps: int) -> None:
    """Prove that step-locking does not perturb our own execution path.

    Free-run our model, then force it with its own tokens. Identical
    inputs through identical kernels must give identical bits; if they do
    not, forcing is changing something and every number this script
    produces is suspect.
    """
    for name in workloads:
        entry = reference["workloads"][name]
        prompt_ids = list(entry["prompt_ids"])
        tokens, free_rows = free_greedy_tokens(model, prompt_ids, steps, eos_id=-1)
        forced_rows = list(forced_decode_logits(model, prompt_ids, tokens))
        assert len(forced_rows) == len(free_rows), (len(forced_rows), len(free_rows))
        mismatch = [
            i for i, (a, b) in enumerate(zip(free_rows, forced_rows, strict=True))
            if not torch.equal(a, b)
        ]
        status = "bit-exact" if not mismatch else f"MISMATCH at steps {mismatch[:5]}"
        print(f"  selfcheck {name}: {len(tokens)} steps -- {status}")
        assert not mismatch, "step-locked forcing perturbed our own path"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--workloads", type=str, default=None)
    parser.add_argument("--configs", type=str, default=None)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--selfcheck-steps", type=int, default=32)
    args = parser.parse_args()

    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    torch.set_grad_enabled(False)

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    workloads = (
        args.workloads.split(",") if args.workloads else list(reference["workloads"])
    )
    specs = [
        parse_injection(text)
        for text in (args.configs.split(",") if args.configs else DEFAULT_CONFIGS)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = load_qwen36_model(
        reference["model_path"], device="cuda", dtype=torch.bfloat16, max_seq_len=MAX_SEQ_LEN
    )
    print(f"load_qwen36_model: {time.time() - t0:.1f}s")

    if args.selfcheck:
        print("\n-- selfcheck: forced-with-own-tokens must equal free-running --")
        selfcheck(model, reference, workloads, args.selfcheck_steps)

    rows = []
    for spec in specs:
        print(f"\n-- {spec} --")
        row = run_config(model, reference, spec, workloads, args.steps, args.output_dir)
        rows.append(row)
        print(
            f"  gap_max={row['max_gap_error']:.4g} p99={row['p99_gap_error']:.4g} "
            f"tie_ulp={row['max_tie_slack_ulps']:.4g} "
            f"disagree={row['disagreement_rate']:.4f} "
            f"kl={row['mean_kl_topk']:.4g} lp={row['max_logprob_error']:.4g} "
            f"drift={row['max_drift_ratio']:.3g} "
            f"nll_excess={row['nll_relative_excess']:+.4f} "
            f"ppl={row['ppl_ours']:.4g} "
            f"layer={row['first_divergent_layer']} "
            f"({row['elapsed_s']:.0f}s)"
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
