"""Phase 1 of the B1-R criterion: record HF's reference trajectory once.

The B1-R criterion (``docs/b1-correctness-criterion.md``) compares this
runtime against HF **step-locked** -- both sides walk the same token
sequence, so neither side's own argmax choice can send it down a different
trajectory and the comparison stays apples-to-apples at every one of 512
steps instead of collapsing to a single "first divergence index".

Phase 1 (this script) produces the reference side; phase 2
(``scripts/b1_forced_decode_agreement.py``) consumes it and can then sweep
many injected-bug configurations against a single load of our own model,
never needing HF resident again.

**Why the split is not optional.** Both models cannot be on the card at
once: our own side grows to ~73 GiB once its dequantisation caches fill,
HF's BF16 reference is ~54 GiB, and the card has 95.59 GiB -- see
``scripts/b1_verify_greedy_alignment.py``'s docstring for the full memory
argument and the three OOMs that established it. That script's remedy
(stage every dequantised weight to disk, evict our side, build HF) is
reused verbatim here rather than reimplemented. But it pays that cost
*per run*, and the injected-bug sweep needs a dozen runs of our side. HF's
answers do not depend on which bug we inject into ours, so they are
computed once and written to disk.

What gets recorded, per workload:

- ``prompt_ids`` and HF's own greedy continuation (``tokens``) -- the
  latter is the forced sequence phase 2 replays.
- ``logits``: the FULL vocabulary logit vector at every step, in bf16.
  Storing the whole vector rather than a top-K slice costs ~250 MB per
  workload and removes an entire class of caveat: with a top-K slice, a
  grossly broken candidate whose argmax falls outside HF's slice yields
  an unbounded, unquantifiable comparison. bf16 is what HF's lm_head
  actually produces, so this is lossless, not a compression choice.
- ``logsumexp``: fp32, computed at capture time from the fp32 upcast, so
  phase 2 can report the SGLang-comparable ``|delta log_softmax|`` bar.
- ``hidden``: HF's per-layer prefill activation trace, for the per-layer
  cosine scan leg (``bfdiag.divergence.scan``). Tiny (prompts are <= 17
  tokens).

Plus one workload-independent record: HF's teacher-forced negative
log-likelihood over a fixed passage. That is the criterion's semantic
floor -- a quantity no argmax tie can move, and the leg with by far the
largest dynamic range against a real defect. vLLM gates the same quantity
for its own model zoo at ``PPL_TOL = 0.01`` relative, one-sided
(``vllm/tests/models/language/generation_ppl_test/ppl_utils.py``).

Run with:
    ~/.venvs/vllm/bin/python scripts/b1_reference_trajectory.py
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _REPO_ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_REPO_ROOT), (
    f"imported runtime from {runtime.__file__}, expected under {_REPO_ROOT}"
)

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from bfdiag.divergence.qwen36_capture import Qwen36HFOracleCaptureSource  # noqa: E402
from runtime.model_loading import _build_qwen36_model_config, load_qwen36_model  # noqa: E402


def _load_sibling_script(name: str):
    """Import a sibling script as a module.

    ``scripts/`` is deliberately not a package (nothing in it is
    importable production code), so the three memory-discipline helpers in
    ``b1_verify_greedy_alignment.py`` -- which took three real OOMs to get
    right -- are loaded by path rather than copy-pasted. Copying them
    would mean the next person fixes the memory bug in one of the two
    copies.
    """
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_b1_sibling_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

MODEL_PATH = (
    "/home/bot/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/"
    "snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404"
)
DEVICE = torch.device("cuda")
MAX_NEW_TOKENS = 512
MAX_SEQ_LEN = 2048

WORKLOADS: tuple[tuple[str, str], ...] = (
    ("factual-short", "The first president of the United States was"),
    ("math-short", "2 + 2 ="),
    (
        "instruction-longer",
        "Write a short paragraph explaining, in simple terms, how a "
        "refrigerator keeps food cold.",
    ),
)

#: Fixed passage for the NLL leg. Deliberately ordinary expository English
#: with no repetition an under-trained state could exploit, and long enough
#: (>= 200 tokens) that a single unlucky token cannot move the mean.
NLL_TEXT = (
    "A refrigerator works by moving heat from the inside of an insulated "
    "box to the room around it. The key idea is that a liquid absorbs heat "
    "when it evaporates and releases heat when it condenses. A compressor "
    "raises the pressure of a refrigerant gas, which makes it hot, and the "
    "coils on the back of the appliance let that heat escape into the "
    "kitchen. As the refrigerant cools it turns back into a liquid. It then "
    "passes through a narrow valve into a low pressure region inside the "
    "cabinet, where it boils at a very low temperature and pulls heat out of "
    "the food and the air. The now warm gas returns to the compressor and "
    "the cycle repeats. Because the process only moves heat rather than "
    "creating cold, a refrigerator always warms the room it stands in. "
    "Engineers measure how well one works with a coefficient of performance, "
    "the ratio of heat removed to electrical energy consumed. Modern units "
    "improve on older ones mainly through better insulation, variable speed "
    "compressors, and refrigerants chosen for a smaller effect on the "
    "atmosphere. The same cycle, run in reverse, is what a heat pump uses to "
    "warm a house in winter, which is why the two appliances share so much "
    "of their design and why an engineer who understands one already "
    "understands most of the other."
)

DEFAULT_OUTPUT = Path(_REPO_ROOT) / ".bfdiag" / "runs" / "b1_reference_trajectory.pt"


def hf_greedy_with_logits(
    hf_model, prompt_ids: list[int], max_new_tokens: int, eos_id: int
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Greedy-decode and keep every step's full logit vector.

    Returns ``(tokens, logits[steps, vocab] bf16 on CPU, logsumexp[steps]
    fp32 on CPU)``. The logsumexp is taken on an fp32 upcast because
    ``logsumexp`` over a 250k-entry bf16 vector loses several digits and
    this value is subtracted from logits to form log-probabilities.
    """
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=hf_model.config)
    input_ids = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)

    tokens: list[int] = []
    logit_rows: list[torch.Tensor] = []
    lse_rows: list[float] = []

    step_input = input_ids
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = hf_model(
                step_input, past_key_values=cache, use_cache=True, logits_to_keep=1
            )
        row = outputs.logits[0, -1]
        logit_rows.append(row.detach().to("cpu", torch.bfloat16).clone())
        lse_rows.append(float(torch.logsumexp(row.float(), dim=-1).item()))
        next_token = int(row.argmax().item())
        tokens.append(next_token)
        if next_token == eos_id:
            break
        step_input = torch.tensor([[next_token]], device=DEVICE, dtype=torch.long)

    return tokens, torch.stack(logit_rows), torch.tensor(lse_rows, dtype=torch.float32)


def teacher_forced_nll(model_call, token_ids: list[int]) -> tuple[float, int]:
    """Sum of ``-log p(token[i] | token[:i])`` over a fixed passage, in one
    forward pass.

    Both sides run this the same way -- a single full-length forward, which
    on this architecture means both use FLA's ``chunk_gated_delta_rule``
    (``runtime/model/qwen36_model.py``'s ``seq_len == 1`` dispatch is not
    taken). That matched-mode requirement is exactly what
    ``notes/2026-08-02-b1-greedy-alignment-fails.md`` found to be violated
    by re-prefilling a prefix to predict what the *incremental* path would
    have produced; running one full forward on each side and comparing them
    to each other does not violate it.

    Returns ``(total_nll, num_predicted_tokens)``.
    """
    logits = model_call(token_ids)
    targets = torch.tensor(token_ids[1:], device=logits.device, dtype=torch.long)
    log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
    picked = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return float(-picked.sum().item()), int(targets.numel())


def main() -> None:
    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    torch.set_grad_enabled(False)

    sibling = _load_sibling_script("b1_verify_greedy_alignment")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model_config = _build_qwen36_model_config(MODEL_PATH)
    eos_id = tokenizer.eos_token_id

    prompt_ids_by_workload = {
        name: tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        for name, prompt in WORKLOADS
    }
    nll_token_ids = tokenizer(NLL_TEXT, return_tensors="pt").input_ids[0].tolist()
    print(f"NLL passage: {len(nll_token_ids)} tokens")

    scratch_dir = Path(tempfile.mkdtemp(prefix="b1_ref_weights_"))
    print(f"staging directory: {scratch_dir}")
    try:
        t0 = time.time()
        mine = load_qwen36_model(
            MODEL_PATH, device="cuda", dtype=torch.bfloat16, max_seq_len=MAX_SEQ_LEN
        )
        print(f"load_qwen36_model: {time.time() - t0:.1f}s")

        t0 = time.time()
        staged_names = sibling.stage_dequantized_weights_to_disk(mine, scratch_dir)
        print(f"staged {len(staged_names)} tensors in {time.time() - t0:.1f}s")

        del mine
        torch.cuda.empty_cache()
        print(f"GPU after evicting ours: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")

        t0 = time.time()
        hf_model = sibling.build_hf_reference_on_device(model_config, DEVICE)
        loaded, missing = sibling.load_staged_weights_into_hf(
            hf_model, scratch_dir, staged_names
        )
        print(
            f"HF built + loaded in {time.time() - t0:.1f}s "
            f"({len(loaded)} in, {len(missing)} missing)"
        )
        assert not missing, f"staged weights unmatched on HF: {missing[:5]}"

        artifact: dict = {
            "model_path": MODEL_PATH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "workloads": {},
        }

        for name, _prompt in WORKLOADS:
            prompt_ids = prompt_ids_by_workload[name]
            t0 = time.time()
            hidden = Qwen36HFOracleCaptureSource(hf_model=hf_model).capture(prompt_ids)
            tokens, logits, lse = hf_greedy_with_logits(
                hf_model, prompt_ids, MAX_NEW_TOKENS, eos_id
            )
            artifact["workloads"][name] = {
                "prompt_ids": prompt_ids,
                "tokens": tokens,
                "logits": logits,
                "logsumexp": lse,
                "hidden": {
                    layer: {k: v.detach().cpu().clone() for k, v in sub.items()}
                    for layer, sub in hidden.items()
                },
            }
            print(
                f"[hf] {name}: {len(tokens)} tokens in {time.time() - t0:.1f}s, "
                f"logits {tuple(logits.shape)}"
            )

        def hf_call(token_ids: list[int]) -> torch.Tensor:
            ids = torch.tensor([token_ids], device=DEVICE, dtype=torch.long)
            with torch.no_grad():
                out = hf_model(ids, use_cache=False)
            return out.logits[0]

        t0 = time.time()
        nll_total, nll_count = teacher_forced_nll(hf_call, nll_token_ids)
        artifact["nll"] = {
            "token_ids": nll_token_ids,
            "total": nll_total,
            "count": nll_count,
            "mean": nll_total / nll_count,
        }
        print(
            f"[hf] NLL: mean={nll_total / nll_count:.6f} nats/token "
            f"(ppl={float(torch.tensor(nll_total / nll_count).exp()):.4f}) "
            f"in {time.time() - t0:.1f}s"
        )

        out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, out_path)
        size_mb = out_path.stat().st_size / 1024**2
        print(f"\nsaved reference trajectory to {out_path} ({size_mb:.1f} MiB)")
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
