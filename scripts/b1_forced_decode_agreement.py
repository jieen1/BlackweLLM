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
import os
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
from torch import nn  # noqa: E402

from bfdiag.divergence.logit_agreement import (  # noqa: E402
    CALIBRATED_THRESHOLDS,
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
from runtime.model.compressed_tensors_linear import (  # noqa: E402
    QSR_EMULATE_FP8_ACTIVATION_ENV,
    QSR_NATIVE_W8A8_FP8_CHANNEL_ENV,
    QSR_NATIVE_W8A8_QUANT_ENV,
    QSR_TORCH_SCALED_MM_FP8_CHANNEL_ENV,
    CompressedTensorsFP8ChannelLinear,
)
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


class _MarlinW8A16Layer(nn.Module):
    """Offline-only holder for a checkpoint-native FP8 linear.

    vLLM's Marlin source is used here only as a numerical/performance oracle
    while a no-vLLM kernel is being ported.  This object owns all packed
    copies, so mutating it cannot modify the runtime module before that
    module is replaced by the candidate closure below.
    """

    def __init__(self, linear: CompressedTensorsFP8ChannelLinear) -> None:
        super().__init__()
        self.input_size_per_partition = linear.input_size
        self.output_size_per_partition = linear.output_size
        self.orig_dtype = torch.bfloat16
        self.weight_block_size = None
        self.logical_widths = [linear.output_size]
        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        self.weight_scale = nn.Parameter(
            linear.weight_scale.detach().clone(), requires_grad=False
        )
        self.register_parameter(
            "bias",
            None
            if linear.bias is None
            else nn.Parameter(linear.bias.detach().clone(), requires_grad=False),
        )
        self.register_parameter("input_scale", None)


def install_marlin_w8a16_oracle(model) -> int:
    """Replace FP8 linears with an offline BF16-activation/FP8-weight oracle.

    This deliberately lives in a diagnostic script, not ``runtime/``.  It
    validates the exact packing, scale and accumulation contract of the
    eventual native backend against the saved HF trajectory, while keeping
    the production no-vLLM import boundary intact.  The candidate never
    builds ``_weight_bf16``; once each private packed copy exists, the model's
    original raw FP8 storage is dropped to keep this single-GPU experiment
    below the card's memory budget.
    """
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_fp8_marlin_linear,
            prepare_fp8_layer_for_marlin,
        )
    except ImportError as exc:
        raise RuntimeError(
            "--marlin-w8a16-oracle needs the local vLLM oracle environment; "
            "it is not a runtime dependency"
        ) from exc

    replaced = 0
    for name, module in model.named_modules():
        if not isinstance(module, CompressedTensorsFP8ChannelLinear):
            continue
        if module.weight.data.numel() == 0:
            raise RuntimeError(
                f"{name}: raw FP8 weight was already released; reload with "
                "keep_fp8_raw_weights=True"
            )
        if module._weight_bf16 is not None:
            raise RuntimeError(
                f"{name}: BF16 FP8 cache already exists; oracle setup must run before a forward"
            )

        packed = _MarlinW8A16Layer(module)
        prepare_fp8_layer_for_marlin(packed, size_k_first=False)

        def marlin_forward(x, *, _packed=packed, _apply=apply_fp8_marlin_linear):
            return _apply(
                input=x,
                weight=_packed.weight,
                weight_scale=_packed.weight_scale,
                workspace=_packed.workspace,
                size_n=_packed.output_size_per_partition,
                size_k=_packed.input_size_per_partition,
                input_dtype=None,
                bias=_packed.bias,
            )

        # _call_impl invokes an instance attribute as ``forward(*args)``;
        # unlike a class method this closure intentionally receives no self.
        module.forward = marlin_forward
        module.weight.data = module.weight.data.new_empty(0)
        # Retain the packed tensor strongly without registering it as a child
        # module (which would make a diagnostic implementation look like a
        # production model component in state_dict walks).
        object.__setattr__(module, "_marlin_w8a16_oracle", packed)
        replaced += 1

    if not replaced:
        raise RuntimeError("Marlin oracle found no FP8-channel Linears")
    torch.cuda.empty_cache()
    print(f"installed offline Marlin W8A16 oracle on {replaced} FP8 Linears")
    return replaced


def install_cutlass_w8a8_oracle(model) -> int:
    """Install the historical FP8xFP8 CUTLASS arithmetic as an offline oracle.

    Qwen3.6's standard checkpoint declares dynamic per-token E4M3
    activations and channel-wise E4M3 weights.  This is the contract used by
    the pre-removal vLLM path, not the BF16-dequant compatibility fallback.
    The oracle keeps the checkpoint weight storage in its required column
    major ``[K, N]`` view and expands only its small per-output scale vector
    to FP32, as CUTLASS requires.  It never constructs ``_weight_bf16``.

    It is intentionally diagnostic-only: importing vLLM here lets the full
    B1 trajectory prove the historical arithmetic before its CUDA source is
    re-homed behind the runtime's own extension.
    """
    try:
        from vllm import _custom_ops as ops
    except ImportError as exc:
        raise RuntimeError(
            "--cutlass-w8a8-oracle needs the local vLLM oracle environment; "
            "it is not a runtime dependency"
        ) from exc

    replaced = 0
    for name, module in model.named_modules():
        if not isinstance(module, CompressedTensorsFP8ChannelLinear):
            continue
        if module.weight.data.numel() == 0:
            raise RuntimeError(
                f"{name}: raw FP8 weight was already released; reload with "
                "keep_fp8_raw_weights=True"
            )
        if module._weight_bf16 is not None:
            raise RuntimeError(
                f"{name}: BF16 FP8 cache already exists; oracle setup must run before a forward"
            )

        # CUTLASS consumes B as column-major [K, N].  Do not make this
        # transpose contiguous: stride(0)==1 is the kernel's physical-layout
        # contract.  The scale conversion is exact for checkpoint BF16 values
        # and costs only O(N), not O(K*N) persistent BF16 weight memory.
        weight_kn = module.weight.detach().t()
        scale_n = module.weight_scale.detach().t().to(torch.float32).contiguous()
        bias = module.bias
        input_size = module.input_size
        output_size = module.output_size

        def cutlass_forward(
            x,
            *,
            _ops=ops,
            _weight_kn=weight_kn,
            _scale_n=scale_n,
            _bias=bias,
            _input_size=input_size,
            _output_size=output_size,
        ):
            if x.shape[-1] != _input_size:
                raise RuntimeError(
                    f"CUTLASS W8A8 oracle input width {x.shape[-1]} != {_input_size}"
                )
            flat_x = x.reshape(-1, _input_size).contiguous()
            x_fp8, x_scale = _ops.scaled_fp8_quant(
                flat_x, None, use_per_token_if_dynamic=True
            )
            out = _ops.cutlass_scaled_mm(
                x_fp8,
                _weight_kn,
                scale_a=x_scale,
                scale_b=_scale_n,
                out_dtype=x.dtype,
                bias=_bias,
            )
            return out.reshape(*x.shape[:-1], _output_size)

        # A transpose view keeps the original raw storage alive after the
        # registered parameter is released.  This mirrors production's future
        # ownership model without adding a second full-matrix copy.
        module.forward = cutlass_forward
        module.weight.data = module.weight.data.new_empty(0)
        object.__setattr__(
            module,
            "_cutlass_w8a8_oracle",
            (weight_kn, scale_n),
        )
        replaced += 1

    if not replaced:
        raise RuntimeError("CUTLASS oracle found no FP8-channel Linears")
    torch.cuda.empty_cache()
    print(f"installed offline CUTLASS W8A8 oracle on {replaced} FP8 Linears")
    return replaced


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
    """Per-layer prefill cosine scan against HF's captured activations.

    Reports the hidden-state cosine curve and the logits entry
    *separately*, and records which bar of the composite threshold
    actually fired. A single "worst cosine over everything" number mixes
    two different kinds of tensor and cannot tell a real divergence from a
    mis-specified threshold -- which is exactly what the first run of this
    script found (see docs/b1-correctness-criterion.md §5.4).
    """
    mine_trace = Qwen36EngineCaptureSource(model=model).capture(list(entry["prompt_ids"]))
    mine_cpu = {
        layer: {k: v.detach().cpu().clone() for k, v in sub.items()}
        for layer, sub in mine_trace.items()
    }
    report = scan_layers(entry["hidden"], mine_cpu)

    hidden = [
        (layer.layer_idx, s)
        for layer in report.layers
        for s in layer.submodules
        if s.submodule == "hidden_state"
    ]
    logits = [s for layer in report.layers for s in layer.submodules if s.submodule == "logits"]
    min_layer, min_verdict = min(hidden, key=lambda pair: pair[1].cosine_similarity)
    first_reasons = []
    if report.first_divergent_layer is not None:
        layer = next(
            item for item in report.layers if item.layer_idx == report.first_divergent_layer
        )
        first_reasons = [f"{s.submodule}: {s.reason()}" for s in layer.worst_submodules]

    return {
        "workload": name,
        "first_divergent_layer": report.first_divergent_layer,
        "first_divergent_reasons": first_reasons,
        "min_hidden_cosine": min_verdict.cosine_similarity,
        "min_hidden_cosine_layer": min_layer,
        "last_hidden_cosine": hidden[-1][1].cosine_similarity,
        "max_hidden_rel_abs_error": max(s.rel_max_abs_error for _idx, s in hidden),
        "logits_cosine": logits[0].cosine_similarity if logits else None,
        "logits_top1_agreement": logits[0].top1_agreement if logits else None,
        "logits_top5_agreement": logits[0].top5_agreement if logits else None,
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
    model,
    reference: dict,
    spec,
    workloads: list[str],
    steps: int | None,
    output_dir: Path,
    *,
    emulate_fp8_activation: bool = False,
    marlin_w8a16_oracle: bool = False,
    cutlass_w8a8_oracle: bool = False,
    torch_w8a8_oracle: bool = False,
    native_w8a8_oracle: bool = False,
    native_w8a8_quant_oracle: bool = False,
) -> dict:
    started = time.time()
    with injected(model, spec):
        if emulate_fp8_activation:
            os.environ[QSR_EMULATE_FP8_ACTIVATION_ENV] = "1"
        try:
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
        finally:
            if emulate_fp8_activation:
                os.environ.pop(QSR_EMULATE_FP8_ACTIVATION_ENV, None)

    hf_nll_mean = float(reference["nll"]["mean"])
    our_nll_mean = nll_total / nll_count
    nll_relative_excess = (our_nll_mean - hf_nll_mean) / hf_nll_mean
    min_logits_cosine = min(s["logits_cosine"] for s in scans)
    report = AgreementReport(
        workloads=tuple(agreements),
        metadata={
            "injection": str(spec),
            "steps_per_workload": steps,
            "capture_top_k": CAPTURE_TOP_K,
            "gap_top_k": GAP_TOP_K,
            "emulate_fp8_activation": emulate_fp8_activation,
            "marlin_w8a16_oracle": marlin_w8a16_oracle,
            "cutlass_w8a8_oracle": cutlass_w8a8_oracle,
            "torch_w8a8_oracle": torch_w8a8_oracle,
            "native_w8a8_oracle": native_w8a8_oracle,
            "native_w8a8_quant_oracle": native_w8a8_quant_oracle,
            "tokens_outside_capture": capture_gaps,
            "layer_scan": scans,
            "nll_mean_ours": our_nll_mean,
            "nll_mean_hf": hf_nll_mean,
            "nll_relative_excess": nll_relative_excess,
            "ppl_ours": float(torch.tensor(our_nll_mean).exp()),
            "ppl_hf": float(torch.tensor(hf_nll_mean).exp()),
            "elapsed_s": time.time() - started,
        },
        nll_relative_excess=nll_relative_excess,
        min_logits_cosine=min_logits_cosine,
    )
    calibrated_passed, calibrated_reasons = passes_b1r_gate(report, CALIBRATED_THRESHOLDS)
    report.metadata["calibrated_gate_passed"] = calibrated_passed
    report.metadata["calibrated_gate_reasons"] = list(calibrated_reasons)
    uncalibrated_passed, uncalibrated_reasons = passes_b1r_gate(
        report, UNCALIBRATED_THRESHOLDS
    )
    report.metadata["uncalibrated_gate_passed"] = uncalibrated_passed
    report.metadata["uncalibrated_gate_reasons"] = list(uncalibrated_reasons)

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
        "min_hidden_cosine": min(s["min_hidden_cosine"] for s in scans),
        "logits_cosine": min(s["logits_cosine"] for s in scans),
        "tokens_outside_capture": len(capture_gaps),
        "calibrated_gate_passed": calibrated_passed,
        "calibrated_gate_reasons": list(calibrated_reasons),
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
            i
            for i, (a, b) in enumerate(zip(free_rows, forced_rows, strict=True))
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
    parser.add_argument(
        "--emulate-fp8-activation",
        action="store_true",
        help=(
            "evaluate the FP8 W8A8 activation-round-trip lower-bound candidate against "
            "the HF reference; production remains unchanged"
        ),
    )
    parser.add_argument(
        "--marlin-w8a16-oracle",
        action="store_true",
        help=(
            "offline-only BF16-activation/native-FP8 Marlin oracle; validates the "
            "future no-vLLM W8A16 backend against the HF trajectory"
        ),
    )
    parser.add_argument(
        "--cutlass-w8a8-oracle",
        action="store_true",
        help=(
            "offline-only historical FP8-activation/FP8-weight CUTLASS oracle; "
            "validates the future no-vLLM W8A8 backend against the HF trajectory"
        ),
    )
    parser.add_argument(
        "--torch-w8a8-oracle",
        action="store_true",
        help=(
            "exercise the runtime's all-layer raw-FP8 torch._scaled_mm candidate; "
            "no vLLM import or BF16 FP8 weight cache"
        ),
    )
    parser.add_argument(
        "--native-w8a8-oracle",
        action="store_true",
        help=(
            "exercise the self-owned SM120 all-layer raw-FP8 W8A8 candidate; "
            "no vLLM import or BF16 FP8 weight cache"
        ),
    )
    parser.add_argument(
        "--native-w8a8-quant-oracle",
        action="store_true",
        help=(
            "exercise the self-owned fused activation quantizer followed by the "
            "runtime's raw-FP8 torch._scaled_mm GEMM; no BF16 FP8 weight cache"
        ),
    )
    args = parser.parse_args()

    candidates = (
        args.emulate_fp8_activation,
        args.marlin_w8a16_oracle,
        args.cutlass_w8a8_oracle,
        args.torch_w8a8_oracle,
        args.native_w8a8_oracle,
        args.native_w8a8_quant_oracle,
    )
    if sum(candidates) > 1:
        parser.error("FP8 candidate oracle flags are mutually exclusive")

    assert QSR_EMULATE_FP8_ACTIVATION_ENV not in os.environ, (
        f"{QSR_EMULATE_FP8_ACTIVATION_ENV} is already set; this script owns the candidate "
        "switch so the reference meaning cannot be silently inverted"
    )
    assert QSR_TORCH_SCALED_MM_FP8_CHANNEL_ENV not in os.environ, (
        f"{QSR_TORCH_SCALED_MM_FP8_CHANNEL_ENV} is already set; use "
        "--torch-w8a8-oracle so this report records the selected path"
    )
    assert QSR_NATIVE_W8A8_FP8_CHANNEL_ENV not in os.environ, (
        f"{QSR_NATIVE_W8A8_FP8_CHANNEL_ENV} is already set; use --native-w8a8-oracle "
        "so this report records the selected path"
    )
    assert QSR_NATIVE_W8A8_QUANT_ENV not in os.environ, (
        f"{QSR_NATIVE_W8A8_QUANT_ENV} is already set; use "
        "--native-w8a8-quant-oracle so this report records the selected path"
    )
    if args.torch_w8a8_oracle:
        os.environ[QSR_TORCH_SCALED_MM_FP8_CHANNEL_ENV] = "all"
    if args.native_w8a8_oracle:
        os.environ[QSR_NATIVE_W8A8_FP8_CHANNEL_ENV] = "all"
    if args.native_w8a8_quant_oracle:
        os.environ[QSR_TORCH_SCALED_MM_FP8_CHANNEL_ENV] = "all"
        os.environ[QSR_NATIVE_W8A8_QUANT_ENV] = "1"

    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    torch.set_grad_enabled(False)

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    workloads = args.workloads.split(",") if args.workloads else list(reference["workloads"])
    specs = [
        parse_injection(text)
        for text in (args.configs.split(",") if args.configs else DEFAULT_CONFIGS)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = load_qwen36_model(
        reference["model_path"],
        device="cuda",
        dtype=torch.bfloat16,
        max_seq_len=MAX_SEQ_LEN,
        keep_fp8_raw_weights=(
            args.marlin_w8a16_oracle
            or args.cutlass_w8a8_oracle
            or args.torch_w8a8_oracle
            or args.native_w8a8_oracle
            or args.native_w8a8_quant_oracle
        ),
    )
    print(f"load_qwen36_model: {time.time() - t0:.1f}s")

    if args.marlin_w8a16_oracle:
        t0 = time.time()
        install_marlin_w8a16_oracle(model)
        print(f"prepare Marlin W8A16 oracle: {time.time() - t0:.1f}s")
    elif args.cutlass_w8a8_oracle:
        t0 = time.time()
        install_cutlass_w8a8_oracle(model)
        print(f"prepare CUTLASS W8A8 oracle: {time.time() - t0:.1f}s")

    if args.selfcheck:
        print("\n-- selfcheck: forced-with-own-tokens must equal free-running --")
        selfcheck(model, reference, workloads, args.selfcheck_steps)

    rows = []
    for spec in specs:
        print(f"\n-- {spec} --")
        row = run_config(
            model,
            reference,
            spec,
            workloads,
            args.steps,
            args.output_dir,
            emulate_fp8_activation=args.emulate_fp8_activation,
            marlin_w8a16_oracle=args.marlin_w8a16_oracle,
            cutlass_w8a8_oracle=args.cutlass_w8a8_oracle,
            torch_w8a8_oracle=args.torch_w8a8_oracle,
            native_w8a8_oracle=args.native_w8a8_oracle,
            native_w8a8_quant_oracle=args.native_w8a8_quant_oracle,
        )
        rows.append(row)
        print(
            f"  gap_max={row['max_gap_error']:.4g} p99={row['p99_gap_error']:.4g} "
            f"tie_ulp={row['max_tie_slack_ulps']:.4g} "
            f"disagree={row['disagreement_rate']:.4f} "
            f"kl={row['mean_kl_topk']:.4g} lp={row['max_logprob_error']:.4g} "
            f"drift={row['max_drift_ratio']:.3g} "
            f"nll_excess={row['nll_relative_excess']:+.4f} "
            f"ppl={row['ppl_ours']:.4g} "
            f"hcos={row['min_hidden_cosine']:.7f} lcos={row['logits_cosine']:.7f} "
            f"({row['elapsed_s']:.0f}s)"
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
