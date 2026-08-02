"""B3-a: isolate WHY the per-round "confirm new anchor" forward
(``scripts/b3_mtp_e2e_acceptance_throughput.py``'s ``speculative_decode``,
the ``_logits_for(model, new_anchor_tensor, state)`` call at the end of
every round) costs ~194-240ms, and whether routing that SAME single-token
step through ``verify_forward`` (K=1) instead of the ordinary ``forward()``
path measurably changes that cost.

This does NOT change any runtime code -- it is a measurement probe,
run before deciding whether "fold anchor into verify block" (B3-a) has
any real lever to pull, per the B3-a brief's instruction to verify the
dependency empirically rather than assume the previous session's
[推算]-only claim.

Method: load the real 27B backbone once, advance a generation state to a
realistic context length (~40 tokens, matching the e2e script's prompts +
partial generation), then alternately time:
  (a) ordinary ``model.forward(token, state)`` for 1 new token
      (what the e2e script's confirm step calls today)
  (b) ``model.verify_forward([[token]], state)`` + ``commit_verify(...,
      accepted_count=1)`` for 1 new token (the "verify block" code path)
  (c) ``model.verify_forward(draft_tokens, state)`` for K=8 new tokens
      (what a normal verify round costs, for scale comparison)

All three advance ``state`` by exactly 1 (or K) real tokens each iteration
(same token id, replayed) so every measurement is a steady-state decode
step at a growing but always-realistic context length -- not a cold/warm
mismatch.

Run: ~/.venvs/vllm/bin/python scripts/b3a_probe_confirm_cost.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__}"
)

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402

# Checkpoint choice matters here, not just as a formality: MTP acceptance
# behavior has been measured to differ by checkpoint *publisher*, not only
# by quantization format -- see scripts/mtpfix_unsloth_checkpoint_probe.py's
# module docstring. This script's own "~194-240ms" cost figures were
# measured on the nvidia checkpoint pre-migration; re-running on the
# standard checkpoint may shift them.
MODEL_PATH = standard_checkpoint_path()
DEVICE = torch.device("cuda")
MAX_SEQ_LEN = 256
N_REPEATS = 12
K_VERIFY = 8


def time_calls(fn, n: int) -> list[float]:
    times = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def summarize(label: str, times: list[float]) -> None:
    ts = sorted(times)
    mean = sum(ts) / len(ts)
    print(
        f"  {label}: mean={mean * 1000:.2f}ms median={ts[len(ts) // 2] * 1000:.2f}ms "
        f"min={ts[0] * 1000:.2f}ms max={ts[-1] * 1000:.2f}ms n={len(ts)}"
    )


def main() -> None:
    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    t0 = time.perf_counter()
    model = load_qwen36_model(MODEL_PATH, device=DEVICE, max_seq_len=MAX_SEQ_LEN, enable_mtp=True)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")

    prompt_ids = tok(
        "Once upon a time, in a small village near the mountains,", return_tensors=None
    )["input_ids"]

    # -- Warmup: pay JIT/compile for decode-mode AND extend-mode (K=1 and
    # K=8) before timing anything, on a throwaway state. -----------------
    warm_state = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    warm_prompt = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)
    warm_hidden = model(warm_prompt, warm_state)
    warm_tok = int(model.compute_logits(warm_hidden[:, -1:, :])[0, -1].argmax().item())
    for _ in range(3):
        step = torch.tensor([[warm_tok]], device=DEVICE, dtype=torch.long)
        model(step, warm_state)
    past_len = warm_state.num_tokens_seen
    draft = torch.tensor([[warm_tok] * K_VERIFY], device=DEVICE, dtype=torch.long)
    vh, snaps = model.verify_forward(draft, warm_state)
    model.commit_verify(warm_state, snaps, past_len=past_len, accepted_count=K_VERIFY)
    for _ in range(3):
        past_len = warm_state.num_tokens_seen
        vh, snaps = model.verify_forward(
            torch.tensor([[warm_tok]], device=DEVICE, dtype=torch.long), warm_state
        )
        model.commit_verify(warm_state, snaps, past_len=past_len, accepted_count=1)
    print("warmup done, context after warmup:", warm_state.num_tokens_seen)

    # -- (a) ordinary forward(), 1 new token, steady state ---------------
    state_a = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    hidden = model(warm_prompt, state_a)
    tok_a = int(model.compute_logits(hidden[:, -1:, :])[0, -1].argmax().item())

    def step_ordinary_forward():
        nonlocal tok_a
        step = torch.tensor([[tok_a]], device=DEVICE, dtype=torch.long)
        model(step, state_a)

    # a few throwaway iterations to reach a similar context length to (b)/(c)
    for _ in range(3):
        step_ordinary_forward()
    times_a = time_calls(step_ordinary_forward, N_REPEATS)
    print(f"context length after (a): {state_a.num_tokens_seen}")
    summarize("(a) ordinary forward(), K=1 (today's confirm step)", times_a)

    # -- (b) verify_forward + commit_verify, K=1, steady state -----------
    state_b = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    hidden = model(warm_prompt, state_b)
    tok_b = int(model.compute_logits(hidden[:, -1:, :])[0, -1].argmax().item())

    def step_verify_k1():
        nonlocal tok_b
        pl = state_b.num_tokens_seen
        draft1 = torch.tensor([[tok_b]], device=DEVICE, dtype=torch.long)
        vh, snaps = model.verify_forward(draft1, state_b)
        model.commit_verify(state_b, snaps, past_len=pl, accepted_count=1)

    for _ in range(3):
        step_verify_k1()
    times_b = time_calls(step_verify_k1, N_REPEATS)
    print(f"context length after (b): {state_b.num_tokens_seen}")
    summarize("(b) verify_forward(K=1) + commit_verify (candidate fold)", times_b)

    # -- (c) verify_forward, K=8, steady state (scale reference) ---------
    state_c = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    hidden = model(warm_prompt, state_c)
    tok_c = int(model.compute_logits(hidden[:, -1:, :])[0, -1].argmax().item())

    def step_verify_k8():
        nonlocal tok_c
        pl = state_c.num_tokens_seen
        draft8 = torch.tensor([[tok_c] * K_VERIFY], device=DEVICE, dtype=torch.long)
        vh, snaps = model.verify_forward(draft8, state_c)
        model.commit_verify(state_c, snaps, past_len=pl, accepted_count=K_VERIFY)

    for _ in range(3):
        step_verify_k8()
    times_c = time_calls(step_verify_k8, N_REPEATS)
    print(f"context length after (c): {state_c.num_tokens_seen}")
    summarize(f"(c) verify_forward(K={K_VERIFY}) (scale reference)", times_c)

    mean_a = sum(times_a) / len(times_a)
    mean_b = sum(times_b) / len(times_b)
    mean_c = sum(times_c) / len(times_c)
    print()
    print(f"ratio (a)/(b) [ordinary vs verify-K1]: {mean_a / mean_b:.3f}x")
    print(f"ratio (a)/(c) [K1-ordinary vs K8-verify]: {mean_a / mean_c:.3f}x")
    print(f"delta (a)-(b): {(mean_a - mean_b) * 1000:.2f}ms")


if __name__ == "__main__":
    main()
