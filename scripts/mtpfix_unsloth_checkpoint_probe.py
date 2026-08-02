"""mtpfix investigation (work/mtp-accept-fix-20260802): same driver as
``scripts/mtpfix_k_sweep_selfbuilt.py``, pointed at
``unsloth/Qwen3.6-27B-NVFP4`` instead of ``nvidia/Qwen3.6-27B-NVFP4``.

**Why**: grepping the historical repo (commit 8f5c195,
``/home/bot/project/qsr-hist-mtp``) shows every one of its 47 benchmark
scripts that set ``MODEL = "..."`` points at ``unsloth/Qwen3.6-27B-NVFP4``
(``grep -rln 'MODEL\\s*=\\s*"unsloth' benchmarks/*.py`` -> 47 hits, ``...nvidia``
-> 0 hits) -- the ~4.0/4 mean-acceptance-length (K=3) measurement this
investigation is chasing was taken against a DIFFERENT checkpoint publisher
than today's ``scripts/b3_mtp_e2e_acceptance_throughput.py``/
``scripts/b3b_*.py`` use. Both are locally cached
(``~/.cache/huggingface/hub/models--{nvidia,unsloth}--Qwen3.6-27B-NVFP4``),
both declare 15 ``mtp.*`` tensors with identical names, both use the
identical top-level prefix convention this repo's loader expects
(``model.language_model.``/``lm_head.``/``mtp.``/``model.visual.``) -- so
this looked like a clean drop-in swap to directly test whether the
CHECKPOINT (not the code) explains the acceptance-rate gap.

**Result: blocked, not a clean swap.** ``unsloth/Qwen3.6-27B-NVFP4``'s
``config.json`` declares a MIXED quantization layout (its
``quantization_config.config_groups`` has a ``group_0`` int8/float8-dynamic
scheme covering most projections, with NVFP4 (``group_1``) reserved for
only ``mlp.(gate|up|down)_proj`` in layers 56-63) -- unlike
``nvidia/Qwen3.6-27B-NVFP4``'s uniform NVFP4-everywhere layout this repo's
``runtime.loading``/``quantized_layers_map`` was built and tested against.
Running this script raises ``RuntimeError: load_qwen36_model: 168
parameter(s) never received a checkpoint tensor`` (``runtime/loading/
common.py::assert_all_params_loaded``) -- the loader's quantized/plain
layer-type inference does not recognize unsloth's mixed layout, so most
non-quantized (int8/float8 group_0) backbone MLP weights are left
unmapped. This is a REAL, checkpoint-format finding in its own right (the
two publishers' NVFP4 checkpoints are not just re-quantizations of the same
layout -- they use genuinely different per-layer precision plans), and it
means this investigation could NOT directly measure whether unsloth's MTP
head weights, run through today's self-built loader, would recover
acceptance closer to the historical ~4.0/4 -- adding proper mixed-precision
support to ``runtime.loading`` to unblock that comparison is future work,
out of scope for this investigation.

Run: ~/.venvs/vllm/bin/python scripts/mtpfix_unsloth_checkpoint_probe.py [K ...]
(currently fails at model load -- see docstring above)
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

from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.model.qwen36_model import Qwen36GenerationState  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402
from runtime.mtp_accept import determine_accept_reject_from_predictions  # noqa: E402

# checkpoint-unify-20260803: not one of the original 22 (this script already
# hardcoded the standard checkpoint deliberately, being the unsloth half of
# a comparison pair with mtpfix_k_sweep_selfbuilt.py's nvidia default) --
# migrated anyway so it resolves through the same single point everything
# else does, and so the new tests/test_checkpoint_scripts_no_hardcoded_
# path.py gate has nothing left to flag.
MODEL_PATH = standard_checkpoint_path()
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

    results: dict = {"model": "unsloth/Qwen3.6-27B-NVFP4", "n_tokens": N_TOKENS, "by_k": {}}

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

    out_path = Path(_ROOT) / ".bfdiag" / "runs" / "mtpfix_unsloth_checkpoint_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
