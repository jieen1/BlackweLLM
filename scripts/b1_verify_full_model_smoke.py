"""B1 step 3/4 (partial -- see script docstring at bottom of output for
what this does NOT cover): load the real 27B checkpoint, run a real
prompt through prefill + greedy decode end-to-end, and sanity-check the
output.

**Scope of what this establishes, stated honestly**: this is a smoke
test, not the full B1 gate. It proves the full 64-layer forward pass
(48 GDN + 16 sparkinfer-paged-attention layers, interleaved) runs without
NaN/Inf, produces syntactically valid token ids, and decodes to text a
human can eyeball for coherence. It does NOT do the full B1 gate's
required "greedy logit match against HF transformers, per-layer cosine
into bfdiag, >= 3 workloads x 512 tokens" -- that requires either (a) an
independent HF reference forward over the SAME real weights (blocked in
this environment: no modelopt-aware HF quantizer is installed, so
"independent" would still mean loading MY dequantized weights into an HF
module -- see runtime/loading/modelopt.py's module docstring), which needs
substantial additional weight-copying glue across all 64 layers plus
enough GPU memory for both a quantized copy (~19 GiB) and a fully
BF16-dequantized HF copy (~54 GiB) resident at once, or (b) restructuring
this script to run them sequentially with intermediate results cached to
CPU. Neither was completed in this pass -- see the B1 handoff notes for
exactly what is left.

Run with: ~/.venvs/vllm/bin/python scripts/b1_verify_full_model_smoke.py
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/bot/project/qsr-w-b1")
import runtime  # noqa: E402

assert runtime.__file__.startswith("/home/bot/project/qsr-w-b1"), runtime.__file__

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402

MODEL_PATH = standard_checkpoint_path()
MAX_NEW_TOKENS = 32


def run_prompt(model, tokenizer, prompt: str) -> None:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    print(f"\n--- prompt: {prompt!r} ({input_ids.shape[1]} tokens) ---")

    state = model.new_generation_state(device=input_ids.device, dtype=torch.bfloat16)

    t0 = time.time()
    hidden = model(input_ids, state)
    logits = model.compute_logits(hidden[:, -1:, :])
    torch.cuda.synchronize()
    print(f"prefill: {time.time()-t0:.2f}s, logits finite={torch.isfinite(logits).all().item()}")

    generated = [int(input_ids[0, -1].item())]
    next_token = int(logits[0, -1].argmax().item())
    generated.append(next_token)

    decode_times = []
    for step in range(MAX_NEW_TOKENS - 1):
        t0 = time.time()
        tok = torch.tensor([[next_token]], device=input_ids.device, dtype=torch.long)
        hidden = model(tok, state)
        logits = model.compute_logits(hidden)
        torch.cuda.synchronize()
        decode_times.append(time.time() - t0)
        if not torch.isfinite(logits).all():
            print(f"  step {step}: NON-FINITE LOGITS -- stopping")
            break
        next_token = int(logits[0, -1].argmax().item())
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated[1:])
    if len(decode_times) > 1:
        median_after_first = sorted(decode_times[1:])[len(decode_times[1:]) // 2]
    else:
        median_after_first = float("nan")
    print(
        f"decode: {len(decode_times)} steps, first={decode_times[0]:.3f}s "
        f"(includes sparkinfer JIT), median-after-first={median_after_first:.4f}s"
    )
    print(f"generated token ids: {generated[1:]}")
    print(f"generated text: {text!r}")


def main() -> None:
    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    t0 = time.time()
    model = load_qwen36_model(MODEL_PATH, device="cuda", dtype=torch.bfloat16, max_seq_len=512)
    print(f"load_qwen36_model: {time.time()-t0:.1f}s")

    run_prompt(model, tokenizer, "The capital of France is")
    run_prompt(model, tokenizer, "2 + 2 =")
    run_prompt(model, tokenizer, "The first president of the United States was")


if __name__ == "__main__":
    main()
