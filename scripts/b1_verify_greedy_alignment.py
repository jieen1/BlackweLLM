"""B1 gate, literal form: "与 HF transformers 贪心逐 token 对齐（>= 3 工作负载
x 512 token）；逐层 logits 余弦相似度进 bfdiag" (``docs/implementation-plan.md``
§7.1).

**Structural memory ordering, not just a coding fix** (2026-08-02, third
round): running this the first time surfaced three OOMs. Two were coding
slips (fp8-then-halve construction transiently needing ~108 GiB; caching
every dequantized weight before copying, materializing a third full
model) -- both fixed upstream of this docstring. The third is
structural: this runtime's own forward pass caches every quantized
Linear's dequantized BF16 weight *forever* once touched
(``runtime/model/modelopt_linear.py``'s ``_ensure_ready``, by design, so
repeated decode steps don't re-dequantize every token) -- so running our
own generation across all 64 layers eventually caches ~54 GiB of BF16
weights ON TOP OF the ~19 GiB quantized originals, ~73 GiB total. A
resident 54 GiB HF reference cannot coexist with that on a 96 GiB card
(73 + 54 = 127 GiB). B0-7's 73 GiB capacity estimate assumed this
runtime's own side stays quantized throughout -- true right up until a
forward pass actually runs, not after.

**Fix: never let both full-size copies be resident on the GPU at once.**
Greedy decoding is deterministic, so the two sides do not need to run
concurrently:

1. Load ``mine`` on GPU (~19 GiB quantized).
2. Stage every one of ``mine``'s real parameters to individual files on
   local disk, dequantizing on GPU one at a time and releasing each
   cache immediately after writing -- never holds more than one
   tensor's worth of extra memory. This runs BEFORE HF's reference is
   built at all, so it also serves as the source of truth for HF's
   weights later, without ever needing a second full BF16 copy resident
   anywhere while ``mine`` is doing the heavy lifting.

   (Originally planned to build HF's reference directly on the CPU
   during this step, per the coordinator's proposed ordering -- checked
   against this machine's actual host RAM first, per their explicit
   "先确认主机内存够放 54 GiB": ``free -h`` showed ~20 GiB available,
   not 54 GiB, so CPU residency was not viable here. Disk (170 GiB free)
   was -- the coordinator's own documented fallback ("不够就把 HF 权重
   落盘再分阶段加载") -- so this script stages to disk instead of RAM.)
3. Run OUR side's full generation + ``Qwen36EngineCaptureSource`` for
   every workload (GPU grows to ~73 GiB as ``mine``'s own caches fill in
   -- HF is not resident on the GPU at all yet, so this is fine).
   Captured per-layer activations are cloned to CPU immediately so they
   survive step 4.
4. ``del mine`` + ``torch.cuda.empty_cache()``.
5. Construct HF's reference for real, directly on the GPU in BF16 (using
   the same "set the default dtype during construction" fix that solved
   OOM #1, so the fp32-then-halve doubling never happens here either).
6. Load every staged weight from disk into HF's model, freeing each
   staged file as it's consumed.
7. Run HF's side's full generation + ``Qwen36HFOracleCaptureSource`` for
   every workload.
8. Compare: per-workload token-for-token match (against the traces
   captured in step 3, held in memory since then) and per-layer
   divergence scan.

Peak GPU usage across the whole run: ~73 GiB (step 3, our side alone) or
~54 GiB (step 7, HF alone) -- never both at once.

Also note: since HF's weights come from copying THIS runtime's own
dequantized tensors (there is no independent modelopt-aware HF loader in
this environment -- see ``runtime/loading/modelopt.py``'s module
docstring), a PASS here proves "does this runtime's Qwen3.6 FORWARD MATH
match HF's reference implementation" -- it does NOT independently confirm
the NVFP4 dequantization itself is bit-correct relative to what modelopt
intended (see ``notes/2026-08-02-b1-nvfp4-nibble-packing-unverified.md``
for what that would take).

Run with: ~/.venvs/vllm/bin/python scripts/b1_verify_greedy_alignment.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

# Derived from this file's own location, not hardcoded to the worktree it was
# written in: the editable blackwellm install pins `runtime`/`server` to a
# static path under the main worktree regardless of cwd, so a bare
# `python scripts/...` silently imports main's code. Asserting after the
# insert is what makes that impossible to do accidentally -- see AGENTS.md,
# "Verifying from a git worktree".
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _REPO_ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_REPO_ROOT), (
    f"imported runtime from {runtime.__file__}, expected under {_REPO_ROOT}"
)

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
from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.model_loading import _build_qwen36_model_config, load_qwen36_model  # noqa: E402

MODEL_PATH = standard_checkpoint_path()
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
        "Write a short paragraph explaining, in simple terms, how a refrigerator keeps food cold.",
    ),
)

#: Quantization metadata has no matching HF ``nn.Linear`` Parameter.  The
#: two NVFP4 formats differ: modelopt uses ``weight``/``weight_scale_2``;
#: compressed-tensors uses ``weight_packed``/``weight_global_scale``.  The
#: logical BF16 value of the latter is staged as ``.weight`` below.
_NEVER_STAGED_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_2",
    ".weight_global_scale",
    ".input_global_scale",
    ".input_scale",
    # Runtime-only FP8 KV-cache descales.  HF keeps K/V in its own cache
    # representation and exposes no matching model Parameters.
    ".k_scale",
    ".v_scale",
)


def _staged_filename(param_name: str) -> str:
    return param_name.replace("/", "_") + ".pt"


def stage_dequantized_weights_to_disk(mine: torch.nn.Module, scratch_dir: Path) -> list[str]:
    """Write every real parameter of ``mine`` to its own file under
    ``scratch_dir``, dequantized to BF16 first for anything that came from
    a quantized Linear -- one tensor at a time, releasing both the GPU
    dequant cache and the transient CPU tensor immediately after each
    write. See this script's module docstring for why this exists (avoids
    ever needing a second full-size copy of the model resident anywhere
    while ``mine`` is doing its own work).

    Iterates ``mine.named_parameters()`` directly (not a module-tree walk
    keyed on ``.weight``) because two real parameters
    (``linear_attn.A_log``/``linear_attn.dt_bias``) live directly on
    ``Qwen36GatedDeltaNet`` itself, not inside a ``.weight``-bearing leaf
    submodule -- an earlier version of this function's ancestor (the old
    ``copy_dequantized_weights_into_hf``) missed exactly these two for
    that reason; fixed by iterating parameters, not modules, and verified
    against a CPU-scale 2-layer toy model (28/28 parameters, exact value
    match) before ever running at the real 27B scale.

    Returns HF-compatible staged parameter names.  In particular, a
    compressed-tensors ``.weight_packed`` parameter becomes the dequantized
    ``.weight`` of the matching HF ``nn.Linear``.  The runtime module's own
    ``_ensure_ready`` remains the only implementation of the checkpoint's
    format-specific dequantization convention.
    """
    modules_by_path = dict(mine.named_modules())

    staged: list[str] = []
    staged_set: set[str] = set()
    for name, param in mine.named_parameters():
        if name.endswith(_NEVER_STAGED_SUFFIXES):
            continue
        module_path = name.rsplit(".", 1)[0] if "." in name else ""
        module = modules_by_path.get(module_path)
        ensure_ready = getattr(module, "_ensure_ready", None) if module is not None else None

        value = param.data
        staged_name = name
        if name.endswith((".weight", ".weight_packed")) and ensure_ready is not None:
            ensure_ready()
            cached = getattr(module, "_weight_bf16", None)
            if cached is not None:
                value = cached
                staged_name = f"{module_path}.weight"
        if name.endswith(".weight_packed") and staged_name == name:
            raise RuntimeError(
                f"{name} is packed but its module did not provide a BF16 cache "
                "after _ensure_ready()"
            )
        if staged_name in staged_set:
            raise RuntimeError(f"duplicate staged HF parameter name: {staged_name}")

        cpu_value = value.detach().to("cpu", torch.bfloat16).clone()
        torch.save(cpu_value, scratch_dir / _staged_filename(staged_name))
        staged.append(staged_name)
        staged_set.add(staged_name)
        del cpu_value

        # Release immediately -- this is the memory this function exists
        # to bound. The module re-dequantizes on demand if it is ever
        # used for a real forward (which it will be, in step 3).
        if module is not None and getattr(module, "_weight_bf16", None) is not None:
            module._weight_bf16 = None
    return staged


def load_staged_weights_into_hf(
    hf_model: torch.nn.Module, scratch_dir: Path, staged_names: list[str]
) -> tuple[list[str], list[str]]:
    """Load every file :func:`stage_dequantized_weights_to_disk` wrote into
    the identically-named parameter on ``hf_model``, deleting each staged
    file as it is consumed. Returns ``(loaded_names, missing_names)`` --
    ``missing_names`` is anything staged but not found on ``hf_model``
    (should be empty on a real run; see this script's module docstring
    point 1 for why identical names hold by construction).
    """
    hf_params = dict(hf_model.named_parameters())
    loaded: list[str] = []
    missing: list[str] = []
    for name in staged_names:
        path = scratch_dir / _staged_filename(name)
        hf_param = hf_params.get(name)
        if hf_param is None:
            missing.append(name)
            path.unlink(missing_ok=True)
            continue
        cpu_tensor = torch.load(path, map_location="cpu", weights_only=True)
        hf_param.data.copy_(cpu_tensor.to(hf_param.dtype))
        loaded.append(name)
        path.unlink()
    return loaded, missing


def build_hf_reference_on_device(model_config: dict, device: torch.device) -> torch.nn.Module:
    """Materialise HF's reference model directly in BF16 on ``device``.

    The obvious spelling -- ``Qwen3_5ForCausalLM(hf_config).to(torch.bfloat16)``
    -- builds every ``nn.Linear`` at the fp32 default first and only then
    halves it, so a 27B model transiently needs ~108 GiB rather than ~54.
    That OOM'd on the first real run (2026-08-02): the card has 95.59 GiB
    and PyTorch reported 108.26 GiB allocated, exactly 2x the BF16
    footprint. Setting the default dtype during construction means the
    fp32 copy never exists.
    """
    text_config_dict = dict(model_config)
    text_config_dict.pop("quantization_config", None)
    hf_config = Qwen3_5TextConfig(**text_config_dict)
    hf_config._attn_implementation = "eager"
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with device:
            hf_model = Qwen3_5ForCausalLM(hf_config)
    finally:
        torch.set_default_dtype(previous_dtype)
    return hf_model.eval()


def greedy_generate(
    model_call, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int
) -> list[int]:
    """Shared greedy loop -- ``model_call(tok)`` must return that step's
    logits for the last position, and close over whatever cache/state
    object it needs between calls (mine's ``Qwen36GenerationState`` or
    HF's ``DynamicCache``, set up by the caller before the first call)."""
    logits = model_call(input_ids)
    next_token = int(logits.argmax().item())
    generated = [next_token]
    for _ in range(max_new_tokens - 1):
        if generated[-1] == eos_id:
            break
        tok = torch.tensor([[generated[-1]]], device=input_ids.device, dtype=torch.long)
        logits = model_call(tok)
        generated.append(int(logits.argmax().item()))
    return generated


def greedy_generate_mine(
    model, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int
) -> list[int]:
    state = model.new_generation_state(device=input_ids.device, dtype=torch.bfloat16)

    def call(tok: torch.Tensor) -> torch.Tensor:
        hidden = model(tok, state)
        return model.compute_logits(hidden[:, -1:, :])[0, -1]

    return greedy_generate(call, input_ids, max_new_tokens, eos_id)


def greedy_generate_hf(
    hf_model, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int
) -> list[int]:
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=hf_model.config)

    def call(tok: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = hf_model(tok, past_key_values=cache, use_cache=True, logits_to_keep=1)
        return outputs.logits[0, -1]

    return greedy_generate(call, input_ids, max_new_tokens, eos_id)


TraceType = dict[int, dict[str, torch.Tensor]]


def _clone_trace_to_cpu(trace: TraceType) -> TraceType:
    """Detach a captured activation trace from the GPU entirely -- needed
    so ``mine``'s trace survives past ``del mine`` in step 4."""
    return {
        layer_idx: {name: tensor.detach().cpu().clone() for name, tensor in submodules.items()}
        for layer_idx, submodules in trace.items()
    }


def main() -> None:
    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_config = _build_qwen36_model_config(MODEL_PATH)
    eos_id = tokenizer.eos_token_id

    scratch_dir = Path(tempfile.mkdtemp(prefix="b1_hf_weights_"))
    print(f"staging directory: {scratch_dir}")

    try:
        # -- Step 1: load our own model --
        t0 = time.time()
        mine = load_qwen36_model(MODEL_PATH, device="cuda", dtype=torch.bfloat16, max_seq_len=1024)
        print(f"load_qwen36_model: {time.time() - t0:.1f}s")

        # -- Step 2: stage every real parameter to disk, dequantized --
        t0 = time.time()
        staged_names = stage_dequantized_weights_to_disk(mine, scratch_dir)
        print(
            f"stage_dequantized_weights_to_disk: {time.time() - t0:.1f}s, "
            f"staged={len(staged_names)}"
        )

        # -- Step 3: run OUR side's generation + capture for every workload --
        mine_traces: dict[str, dict] = {}
        mine_tokens: dict[str, list[int]] = {}
        prompt_ids_by_workload: dict[str, list[int]] = {}
        for name, prompt in WORKLOADS:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
            prompt_ids = input_ids[0].tolist()
            prompt_ids_by_workload[name] = prompt_ids

            t0 = time.time()
            trace = Qwen36EngineCaptureSource(model=mine).capture(prompt_ids)
            mine_traces[name] = _clone_trace_to_cpu(trace)
            mine_tokens[name] = greedy_generate_mine(mine, input_ids, MAX_NEW_TOKENS, eos_id)
            print(
                f"[mine] {name}: {time.time() - t0:.1f}s, generated {len(mine_tokens[name])} tokens"
            )

        mem_after_mine = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU allocated after our own generation: {mem_after_mine:.2f} GiB")

        # -- Step 4: free our model entirely before HF ever touches the GPU --
        del mine
        torch.cuda.empty_cache()
        mem_after_free = torch.cuda.memory_allocated() / (1024**3)
        print(f"GPU allocated after del mine + empty_cache: {mem_after_free:.2f} GiB")

        # -- Step 5+6: build HF's reference for real, load the staged weights --
        t0 = time.time()
        hf_model = build_hf_reference_on_device(model_config, DEVICE)
        loaded, missing = load_staged_weights_into_hf(hf_model, scratch_dir, staged_names)
        print(
            f"build_hf_reference_on_device + load: {time.time() - t0:.1f}s, "
            f"loaded={len(loaded)} missing={len(missing)}"
        )
        assert not missing, (
            f"{len(missing)} staged weight(s) not matched into HF, "
            f"e.g. {missing[:5]!r}; refusing a partial-reference comparison"
        )

        # -- Step 7: run HF's side's generation + capture for every workload --
        token_results = []
        for name, prompt_ids in prompt_ids_by_workload.items():
            input_ids = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)

            t0 = time.time()
            hf_trace = Qwen36HFOracleCaptureSource(hf_model=hf_model).capture(prompt_ids)
            hf_tokens = greedy_generate_hf(hf_model, input_ids, MAX_NEW_TOKENS, eos_id)
            print(f"[hf] {name}: {time.time() - t0:.1f}s, generated {len(hf_tokens)} tokens")

            # -- Step 8: compare --
            divergence = scan_layers(hf_trace, mine_traces[name])
            print(f"\n--- {name}: per-layer divergence scan ---")
            print(format_text_report(divergence))

            result = compare_greedy_token_ids(name, prompt_ids, mine_tokens[name], hf_tokens)
            token_results.append(result)
            print(
                f"{name}: match_rate={result.match_rate:.4f} "
                f"first_divergence={result.first_divergence_index} "
                f"compared={result.num_tokens_compared}"
            )

        alignment_report = GreedyAlignmentReport(workloads=tuple(token_results))
        out_path = Path(_REPO_ROOT) / ".bfdiag" / "runs" / "b1_greedy_alignment.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        alignment_report.save(out_path)
        print(f"\nsaved alignment report to {out_path}")

        passed, reason = passes_b1_gate(alignment_report)
        print(f"\nB1 GATE: {'PASS' if passed else 'FAIL'} -- {reason}")
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
