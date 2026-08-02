"""mtpfix investigation (work/mtp-accept-fix-20260802): run the EXACT SAME
MTP speculative-decode driver as ``scripts/b3_mtp_e2e_acceptance_throughput.py``
(byte-for-byte identical ``speculative_decode``/``free_greedy_decode`` logic,
copied rather than imported so this script has no dependency on that file
changing under it) at a CONFIGURABLE K, to answer one question: **is
today's low acceptance rate (15.2% prose / 35.0% code at K=8) purely a
consequence of K being larger than the historical K=3 config, or does it
persist at K=3 too?**

**Result (measured, this session): it persists.** K=3 on this runtime's
self-built Qwen3.6 path (checkpoint ``nvidia/Qwen3.6-27B-NVFP4``) gives
mean committed length/round = 2.07/4 (prose, 52% of the K+1=4 cap) and
3.44/4 (code, 86% of cap) -- nowhere near the historical ~4.0/4 (~100%)
recorded in ``PROGRESS.md`` at commit 8f5c195 in
``/home/bot/project/qsr-hist-mtp`` (K=3 production setting). K alone does
not explain the gap (independently corroborated by
``notes/2026-08-02-b3b-acceptance-rate-vs-k.md``'s own K-sweep, which
measured 0.401 acceptance rate at K=3 -- same unit-normalized answer: mean
length ~2.2/4, not ~4/4).

Given that, this session's investigation moved to WHY: see
``notes/2026-08-02-mtpfix-historical-comparison.md`` for the structural
diff against ``runtime/direct_model_runner.py``'s ``_mtp_sync_and_propose``/
``_mtp_run_continuation_steps`` (commit 8f5c195) and the checkpoint-publisher
confound this script's sibling (``mtpfix_unsloth_checkpoint_probe.py``)
surfaces: the historical ~4.0 measurement ran vLLM's own native
``load_eagle_model`` MTP loader against ``unsloth/Qwen3.6-27B-NVFP4``, NOT
this repo's self-built ``runtime/model/qwen36_model.py`` path against
``nvidia/Qwen3.6-27B-NVFP4`` -- two confounds (execution stack AND
checkpoint publisher) stacked on top of the K difference.

Run: ~/.venvs/vllm/bin/python scripts/mtpfix_k_sweep_selfbuilt.py [K ...]
(defaults to 3 8 if no K given)
"""

from __future__ import annotations

import json
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

from runtime.checkpoints import modelopt_checkpoint_path  # noqa: E402
from runtime.model.qwen36_model import Qwen36GenerationState  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402
from runtime.mtp_accept import determine_accept_reject_from_predictions  # noqa: E402

# Deliberately modelopt (nvidia), not the standard checkpoint -- and unlike
# most of this round's migrations, not because of format-specific code.
# This script's entire point (see module docstring) is being the
# nvidia-checkpoint half of a deliberate two-script comparison against its
# sibling ``mtpfix_unsloth_checkpoint_probe.py`` (hardcoded to the standard
# checkpoint), isolating the checkpoint-*publisher* confound in the
# historical MTP acceptance-rate discrepancy. Pointing this script at the
# standard checkpoint too would collapse the comparison the pair exists to
# make.
MODEL_PATH = modelopt_checkpoint_path()
DEVICE = torch.device("cuda")
MAX_SEQ_LEN = 256
N_TOKENS = 32  # tokens to generate per prompt (both paths)

PROMPTS = {
    "prose": "Once upon a time, in a small village near the mountains,",
    "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
}


def _logits_for(
    model, token_ids: torch.Tensor, state: Qwen36GenerationState
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = model(token_ids, state)
    return model.compute_logits(hidden[:, -1:, :])[0, -1], hidden[:, -1:, :]


def free_greedy_decode(model, prompt_ids: list[int], n_tokens: int) -> tuple[list[int], float]:
    state = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    prompt = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits, _ = _logits_for(model, prompt, state)
    tokens: list[int] = []
    for _ in range(n_tokens):
        token = int(logits.argmax().item())
        tokens.append(token)
        step = torch.tensor([[token]], device=DEVICE, dtype=torch.long)
        logits, _ = _logits_for(model, step, state)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return tokens, elapsed


def speculative_decode(
    model, prompt_ids: list[int], n_tokens: int, k: int
) -> tuple[list[int], float, list[dict]]:
    """Byte-for-byte identical to
    ``scripts/b3_mtp_e2e_acceptance_throughput.py``'s own function, at
    commit dd758d2 -- only ``k`` is now a real parameter instead of a
    module-level constant."""
    state = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    mtp_cache = model.mtp_new_cache(device=DEVICE, dtype=torch.bfloat16)
    prompt = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    prompt_logits, _ = _logits_for(model, prompt, state)
    anchor_token = int(prompt_logits.argmax().item())
    anchor_step = torch.tensor([[anchor_token]], device=DEVICE, dtype=torch.long)
    anchor_logits, anchor_hidden = _logits_for(model, anchor_step, state)
    anchor_argmax = int(anchor_logits.argmax().item())

    committed: list[int] = [anchor_token]
    rounds: list[dict] = []

    while len(committed) < n_tokens:
        round_mtp_start = mtp_cache.seq_len
        draft_tokens: list[int] = []
        mtp_hidden = anchor_hidden
        next_input = torch.tensor([[anchor_token]], device=DEVICE, dtype=torch.long)
        for _step in range(k):
            draft_token, mtp_hidden = model.mtp_step(
                next_input, mtp_hidden, mtp_cache.seq_len, mtp_cache
            )
            draft_tokens.append(int(draft_token.item()))
            next_input = draft_token.view(1, 1)

        past_len = state.num_tokens_seen
        draft_tensor = torch.tensor([draft_tokens], device=DEVICE, dtype=torch.long)
        verify_hidden, gdn_snapshots = model.verify_forward(draft_tensor, state)
        verify_logits = model.compute_logits(verify_hidden)[0]  # [K, vocab]
        verify_argmax = verify_logits.argmax(dim=-1).tolist()

        predicted_tokens = [anchor_argmax] + verify_argmax
        decision = determine_accept_reject_from_predictions(
            [anchor_token] + draft_tokens, predicted_tokens
        )
        m = decision["num_accepted"]

        model.commit_verify(state, gdn_snapshots, past_len=past_len, accepted_count=m)
        mtp_cache.seq_len = round_mtp_start + m

        committed.extend(decision["committed"])
        rounds.append({"k": k, "num_accepted": m, "rejected_at": decision["rejected_at"]})

        new_anchor = decision["committed"][-1]
        new_anchor_tensor = torch.tensor([[new_anchor]], device=DEVICE, dtype=torch.long)
        new_anchor_logits, new_anchor_hidden = _logits_for(model, new_anchor_tensor, state)

        anchor_token = new_anchor
        anchor_hidden = new_anchor_hidden
        anchor_argmax = int(new_anchor_logits.argmax().item())

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return committed[:n_tokens], elapsed, rounds


def main() -> None:
    ks = [int(a) for a in sys.argv[1:]] or [3, 8]
    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    t_load0 = time.perf_counter()
    model = load_qwen36_model(MODEL_PATH, device=DEVICE, max_seq_len=MAX_SEQ_LEN, enable_mtp=True)
    print(f"model loaded in {time.perf_counter() - t_load0:.1f}s")

    warm_ids = tok("Warm up the kernels before timing.", return_tensors=None)["input_ids"]
    free_greedy_decode(model, warm_ids, 4)
    for k in ks:
        speculative_decode(model, warm_ids, 4, k)
    print("warmup done")

    results: dict = {"model": "nvidia/Qwen3.6-27B-NVFP4", "n_tokens": N_TOKENS, "by_k": {}}

    for k in ks:
        print(f"\n########## K={k} ##########")
        results["by_k"][k] = {"prompts": {}}
        for label, prompt_text in PROMPTS.items():
            prompt_ids = tok(prompt_text, return_tensors=None)["input_ids"]
            print(f"\n=== K={k} prompt={label!r} ({len(prompt_ids)} tokens) ===")

            ref_tokens, ref_time = free_greedy_decode(model, prompt_ids, N_TOKENS)
            spec_tokens, spec_time, rounds = speculative_decode(model, prompt_ids, N_TOKENS, k)

            match = ref_tokens == spec_tokens
            first_diff = None
            if not match:
                for i, (a, b) in enumerate(zip(ref_tokens, spec_tokens)):
                    if a != b:
                        first_diff = i
                        break

            total_accepted = sum(r["num_accepted"] for r in rounds)
            total_drafted = sum(r["k"] for r in rounds)
            accept_rate = total_accepted / total_drafted if total_drafted else 0.0
            mean_accept_per_round = total_accepted / len(rounds) if rounds else 0.0
            mean_len_per_round = (total_accepted + len(rounds)) / len(rounds) if rounds else 0.0

            ref_tok_s = N_TOKENS / ref_time
            spec_tok_s = len(spec_tokens) / spec_time
            speedup = spec_tok_s / ref_tok_s if ref_tok_s else float("nan")

            print(
                f"  acceptance rate: {accept_rate:.3f} ({total_accepted}/{total_drafted}); "
                f"mean accepted/round: {mean_accept_per_round:.2f}; "
                f"mean COMMITTED length/round (incl. bonus): {mean_len_per_round:.2f} "
                f"(cap = K+1 = {k + 1}); rounds={len(rounds)}; "
                f"speedup={speedup:.2f}x; tokens_match={match}"
                + ("" if match else f" (first diff at {first_diff})")
            )
            print(f"  per-round num_accepted: {[r['num_accepted'] for r in rounds]}")

            results["by_k"][k]["prompts"][label] = {
                "prompt_len": len(prompt_ids),
                "non_speculative": {"tokens_per_sec": ref_tok_s, "wall_s": ref_time},
                "speculative": {
                    "tokens_per_sec": spec_tok_s,
                    "wall_s": spec_time,
                    "rounds": rounds,
                    "acceptance_rate": accept_rate,
                    "mean_accepted_per_round": mean_accept_per_round,
                    "mean_committed_length_per_round": mean_len_per_round,
                },
                "speedup": speedup,
                "tokens_match": match,
                "first_diff_index": first_diff,
            }

    out_path = Path(_ROOT) / ".bfdiag" / "runs" / "mtpfix_k_sweep_selfbuilt.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
