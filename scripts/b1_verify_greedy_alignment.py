"""B1 gate, literal form: "与 HF transformers 贪心逐 token 对齐（>= 3 工作负载
x 512 token）；逐层 logits 余弦相似度进 bfdiag" (``docs/implementation-plan.md``
§7.1).

**Written but NOT YET RUN** -- no GPU time was available for this in the
pass that wrote it (see the B1 handoff notes). Per the coordinator's
explicit ask ("准备 HF 逐 token 对齐的脚手架...写好之后等拿到 GPU 一次跑完"),
this script is meant to be run once, end to end, the next time the GPU is
free -- read it once before running, since three things in it are
designed-not-measured and should be double-checked against the real
output the first time, not assumed correct because they compiled:

1. **HF-side weight copying** (:func:`copy_dequantized_weights_into_hf`).
   Relies on this runtime's model graph (``runtime/model/qwen36_model.py``)
   using IDENTICAL dotted parameter names to HF's own
   ``Qwen3_5ForCausalLM`` (``model.layers.{i}.self_attn.q_proj.weight``,
   ``model.layers.{i}.linear_attn.A_log``, etc.) -- true by construction
   (this runtime's classes mirror HF's attribute names one-for-one, see
   that module's docstring). Partially exercised already: a 2-layer toy
   model (tiny hidden size, one GDN + one full-attention layer) dry-run on
   CPU caught a real bug in an earlier version of this function (it
   walked module *paths* keyed on ``.weight`` and silently missed
   ``A_log``/``dt_bias``, which live directly on ``Qwen36GatedDeltaNet``
   itself, not inside a ``.weight``-bearing leaf) -- see that function's
   docstring for the fix. The toy-model check confirmed 28/28 parameters
   copy with exactly matching values after the fix, including through the
   FP8 dequantization path. **Never exercised at the real 27B scale or
   against real checkpoint weights** -- the toy model's weights were
   random, and NVFP4 (only FP8 was reachable at that tiny hidden size)
   was not exercised in the toy check at all.
2. **HF-side per-layer indexing convention**
   (``bfdiag/divergence/qwen36_capture.py``'s ``hidden_states[i+1] ==
   after layer i``) -- documented there as read-from-HF-docs, not
   confirmed against a live forward pass.
3. **27B-in-BF16 memory headroom**: this script holds BOTH a quantized
   copy of the model (~19 GiB, this runtime's own) AND a fully
   BF16-dequantized HF copy (~54 GiB) resident on the GPU at once (~73
   GiB total) -- arithmetically fits a 96 GiB card per B0-7's own
   capacity math, but was never actually allocated together before this
   script runs it.

Also note: since the HF side's weights come from copying THIS runtime's
own dequantized tensors (there is no independent modelopt-aware HF loader
in this environment -- see ``runtime/loading/modelopt.py``'s module
docstring), a PASS here proves "does this runtime's Qwen3.6 FORWARD MATH
match HF's reference implementation" -- it does NOT independently confirm
the NVFP4 dequantization itself is bit-correct relative to what modelopt
intended (no oracle for that exists on this machine either; see that same
docstring for what the strongest available evidence for THAT question is
instead -- coherent, factually correct generation, already measured in
``scripts/b1_verify_full_model_smoke.py``).

Run with: ~/.venvs/vllm/bin/python scripts/b1_verify_greedy_alignment.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/bot/project/qsr-w-b1")
import runtime  # noqa: E402

assert runtime.__file__.startswith("/home/bot/project/qsr-w-b1"), runtime.__file__

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

from bfdiag.divergence.qwen36_capture import (  # noqa: E402
    Qwen36EngineCaptureSource,
    Qwen36HFOracleCaptureSource,
)
from bfdiag.divergence.qwen36_greedy_alignment import (  # noqa: E402
    GreedyAlignmentReport,
    compare_greedy_token_ids,
    passes_b1_gate,
)
from bfdiag.divergence.report import format_text_report  # noqa: E402
from bfdiag.divergence.scan import scan_layers  # noqa: E402
from runtime.model_loading import _build_qwen36_model_config, load_qwen36_model  # noqa: E402

MODEL_PATH = (
    "/home/bot/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/"
    "snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404"
)
MAX_NEW_TOKENS = 512
DEVICE = torch.device("cuda")

#: >= 3 workloads per the gate. Chosen for a mix of prompt lengths/shapes
#: (also exercises Qwen36AttentionWorkspace's fixed-capacity contract
#: across genuinely different (seq_len, cache_seqlens) pairs in one run,
#: which the smoke test's own three prompts already showed matters --
#: see the sibling commit fixing that).
WORKLOADS: tuple[tuple[str, str], ...] = (
    ("factual-short", "The first president of the United States was"),
    ("math-short", "2 + 2 ="),
    (
        "instruction-longer",
        "Write a short paragraph explaining, in simple terms, how a "
        "refrigerator keeps food cold.",
    ),
)


def copy_dequantized_weights_into_hf(
    mine: torch.nn.Module, hf_model: torch.nn.Module
) -> tuple[list[str], list[str]]:
    """Copy every real parameter from ``mine`` into the identically-named
    parameter in ``hf_model`` -- dequantized to BF16 first for anything
    that came from a quantized Linear. See this script's module docstring
    point 1 for why identical parameter names hold by construction.

    Iterates ``mine.named_parameters()`` directly (not a module-tree walk
    keyed on ``.weight``) specifically because two real parameters
    (``linear_attn.A_log``/``linear_attn.dt_bias``) live directly on
    ``Qwen36GatedDeltaNet`` itself, not inside a ``.weight``-bearing leaf
    submodule -- an earlier version of this function that only walked
    "leaf modules with a `.weight` attribute" silently missed exactly
    these two (caught by a CPU-scale dry run against a 2-layer toy model
    before this script was ever pointed at the real 27B checkpoint: 26/28
    real parameters copied, 0 reported as skipped -- the 2 missing ones
    never entered the walk at all, so they could not even show up as a
    logged skip. Fixed by iterating parameters, not modules.).
    ``.weight_scale``/``.weight_scale_2`` are the only real parameters
    deliberately never copied -- HF has no quantized-scale equivalent at
    all, by design (see runtime/model/modelopt_linear.py).

    Returns ``(copied_names, skipped_names)`` for the caller to sanity-log
    -- a real run against the full model should see zero skips; any skip
    means a name genuinely doesn't exist on the HF side and needs
    investigating before trusting the comparison at all.
    """
    for module in mine.modules():
        ensure_ready = getattr(module, "_ensure_ready", None)
        if ensure_ready is not None:
            ensure_ready()

    dequantized_by_module: dict[str, torch.Tensor] = {
        name: module._weight_bf16
        for name, module in mine.named_modules()
        if getattr(module, "_weight_bf16", None) is not None
    }

    hf_params = dict(hf_model.named_parameters())
    copied: list[str] = []
    skipped: list[str] = []
    for name, param in mine.named_parameters():
        if name.endswith((".weight_scale", ".weight_scale_2")):
            continue
        module_path = name.rsplit(".", 1)[0] if "." in name else ""
        is_dequantized_weight = name.endswith(".weight") and module_path in dequantized_by_module
        value = dequantized_by_module[module_path] if is_dequantized_weight else param.data

        hf_param = hf_params.get(name)
        if hf_param is None:
            skipped.append(name)
            continue
        hf_param.data.copy_(value.to(hf_param.dtype))
        copied.append(name)
    return copied, skipped


def build_hf_reference(model_config: dict) -> torch.nn.Module:
    text_config_dict = dict(model_config)
    text_config_dict.pop("quantization_config", None)
    hf_config = Qwen3_5TextConfig(**text_config_dict)
    hf_config._attn_implementation = "eager"
    with DEVICE:
        hf_model = Qwen3_5ForCausalLM(hf_config).to(torch.bfloat16)
    return hf_model.eval()


def greedy_generate_mine(
    model, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int
) -> list[int]:
    state = model.new_generation_state(device=input_ids.device, dtype=torch.bfloat16)
    hidden = model(input_ids, state)
    logits = model.compute_logits(hidden[:, -1:, :])
    next_token = int(logits[0, -1].argmax().item())
    generated = [next_token]
    for _ in range(max_new_tokens - 1):
        if generated[-1] == eos_id:
            break
        tok = torch.tensor([[generated[-1]]], device=input_ids.device, dtype=torch.long)
        hidden = model(tok, state)
        logits = model.compute_logits(hidden)
        generated.append(int(logits[0, -1].argmax().item()))
    return generated


def greedy_generate_hf(
    hf_model, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int
) -> list[int]:
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=hf_model.config)
    with torch.no_grad():
        outputs = hf_model(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    next_token = int(outputs.logits[0, -1].argmax().item())
    generated = [next_token]
    for _ in range(max_new_tokens - 1):
        if generated[-1] == eos_id:
            break
        tok = torch.tensor([[generated[-1]]], device=input_ids.device, dtype=torch.long)
        with torch.no_grad():
            outputs = hf_model(tok, past_key_values=cache, use_cache=True, logits_to_keep=1)
        generated.append(int(outputs.logits[0, -1].argmax().item()))
    return generated


def main() -> None:
    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_config = _build_qwen36_model_config(MODEL_PATH)

    t0 = time.time()
    mine = load_qwen36_model(MODEL_PATH, device="cuda", dtype=torch.bfloat16, max_seq_len=1024)
    print(f"load_qwen36_model: {time.time() - t0:.1f}s")

    t0 = time.time()
    hf_model = build_hf_reference(model_config)
    copied, skipped = copy_dequantized_weights_into_hf(mine, hf_model)
    print(
        f"build_hf_reference + copy: {time.time() - t0:.1f}s, "
        f"copied={len(copied)} skipped={len(skipped)}"
    )
    if skipped:
        print(f"WARNING: {len(skipped)} weight(s) not matched into HF, e.g. {skipped[:5]!r}")

    eos_id = tokenizer.eos_token_id

    token_results = []
    divergence_reports = []
    for name, prompt in WORKLOADS:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        prompt_ids = input_ids[0].tolist()

        engine_trace = Qwen36EngineCaptureSource(model=mine).capture(prompt_ids)
        oracle_trace = Qwen36HFOracleCaptureSource(hf_model=hf_model).capture(prompt_ids)
        report = scan_layers(oracle_trace, engine_trace)
        divergence_reports.append((name, report))
        print(f"\n--- {name}: per-layer divergence scan ---")
        print(format_text_report(report))

        mine_tokens = greedy_generate_mine(mine, input_ids, MAX_NEW_TOKENS, eos_id)
        hf_tokens = greedy_generate_hf(hf_model, input_ids, MAX_NEW_TOKENS, eos_id)
        result = compare_greedy_token_ids(name, prompt_ids, mine_tokens, hf_tokens)
        token_results.append(result)
        print(
            f"{name}: match_rate={result.match_rate:.4f} "
            f"first_divergence={result.first_divergence_index} "
            f"compared={result.num_tokens_compared}"
        )

    alignment_report = GreedyAlignmentReport(workloads=tuple(token_results))
    out_path = Path("/home/bot/project/qsr-w-b1/.bfdiag/runs/b1_greedy_alignment.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    alignment_report.save(out_path)
    print(f"\nsaved alignment report to {out_path}")

    passed, reason = passes_b1_gate(alignment_report)
    print(f"\nB1 GATE: {'PASS' if passed else 'FAIL'} -- {reason}")


if __name__ == "__main__":
    main()
