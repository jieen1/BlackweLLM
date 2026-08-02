"""B3: real MTP acceptance rate + throughput, full model, real weights.

This is the full-model integration this session's B3 report needs on top
of the two lighter-weight probes (``scripts/b3_probe_gdn_spec_rollback.py``
for one GDN layer, ``scripts/b3_probe_mtp_head.py`` for the MTP head
alone): loads the REAL 27B backbone + MTP head together and runs genuine
speculative decoding rounds (draft K tokens via the chained MTP head,
verify against the target model in one pass, greedy accept/reject, roll
back on partial accept) against real prompts, end to end.

**Correctness criterion** (``docs/b1-correctness-criterion.md`` §7's B3
judgement, restated): NOT bit-exactness against HF -- "参照系应当是我们自己
的非投机路径". This script's primary check is exactly that: run the SAME
prompt through (a) this runtime's own free greedy autoregressive decode
(no speculation at all -- the already-B1-proven path) and (b) this
runtime's own speculative decode (MTP draft + verify + greedy accept/
reject), and compare the two COMMITTED token sequences. Greedy speculative
decoding is only correct if these are token-for-token identical -- that is
the mathematical guarantee the accept/reject algorithm exists to provide
(on any mismatch, the target's own prediction becomes the correction, so
the committed sequence can never diverge from what greedy (a) alone would
have produced -- see runtime/mtp_accept.py's module docstring for the
formal argument, there stated for the non-greedy/rejection-sampling
sibling but the greedy case is the special case where q puts all its mass
on one token).

**Known potential source of disagreement that would NOT be a bug**: the
target model's full-attention layers process K draft positions in ONE
"extend"-mode kernel call during verify, vs one "decode"-mode kernel call
per position during plain sequential decode. These are different kernel
code paths (different KV-chunking / reduction order), analogous to (but
not the same as) the GDN chunk-vs-recurrent ~30-ULP gap this session's
brief flagged as a trap -- B2's own notes
(notes/2026-08-02-eager-verify-cg-verify-divergence.md) already document
a same-shape phenomenon (CG-replay decode vs eager decode disagreeing at
kv_len>=400 while both are correct, cos>=0.999997) for exactly this
"same math, different kernel schedule" reason. GDN layers carry NO such
risk here -- verify_forward always uses spec_forward's single-token
fused_recurrent path, never chunk_gated_delta_rule, by construction (see
Qwen36TextModelSelfBuilt.verify_forward's docstring). If the two token
sequences disagree, this script reports the first divergence position and
both diverging token ids (written to the output JSON) rather than
assuming either a bug or a benign tie-break -- a real logit-gap
measurement at that position (bfdiag's gap-error metric, the same one
B1-R uses) is the natural follow-up this script does not itself perform.

Run: ~/.venvs/vllm/bin/python scripts/b3_mtp_e2e_acceptance_throughput.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = "/home/bot/project/qsr-w-b3a"
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__}"
)

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from runtime.model.qwen36_model import Qwen36GenerationState  # noqa: E402
from runtime.model_loading import load_qwen36_model  # noqa: E402
from runtime.mtp_accept import determine_accept_reject_from_predictions  # noqa: E402

MODEL_PATH = (
    "/home/bot/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/"
    "snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404"
)
DEVICE = torch.device("cuda")
MAX_SEQ_LEN = 256
# Draft tokens per speculative round. 4, not 8.
#
# 8 was picked when the head was first wired up, for no measured reason. The
# K sweep (scripts/b3b_k_sweep.py, real 27B, two independent runs) shows
# acceptance falling monotonically with K while speedup peaks at K in 3..6:
#
#   K=3  prose 1.06x  code 1.27x
#   K=4  prose 1.11x  code 1.38x     <- here
#   K=5  prose 1.11x  code 1.41x
#   K=8  prose 0.88x  code 1.09x     <- the old default, net-negative on prose
#   K=16 prose 0.65x  code 0.97x
#
# K=8 was the single reason every earlier B3 measurement reported speculation
# as a net loss. Changing this constant is the whole fix; no other code moved.
#
# Caveat carried from that sweep: these ran in a standalone eager script, not
# ServerEngine, so "4 beats 8" is trustworthy and "1.38x" is not a production
# number. See notes/2026-08-02-b3b-acceptance-rate-vs-k.md.
K = 4
N_TOKENS = 32  # tokens to generate per prompt (both paths)

PROMPTS = {
    "prose": "Once upon a time, in a small village near the mountains,",
    "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
}


def _logits_for(
    model, token_ids: torch.Tensor, state: Qwen36GenerationState
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (last-position logits [vocab], last-position post-norm
    hidden state [1, 1, hidden_size])."""
    hidden = model(token_ids, state)
    return model.compute_logits(hidden[:, -1:, :])[0, -1], hidden[:, -1:, :]


def free_greedy_decode(model, prompt_ids: list[int], n_tokens: int) -> tuple[list[int], float]:
    """This runtime's own, already-B1-proven non-speculative path."""
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
    """Real MTP speculative decode: draft K tokens (chained MTP head),
    verify against the target in one pass (GDN spec_forward + ordinary
    multi-token attention), greedy accept/reject, roll back on partial
    accept, advance the new anchor through one ordinary target forward.
    """
    state = model.new_generation_state(device=DEVICE, dtype=torch.bfloat16)
    mtp_cache = model.mtp_new_cache(device=DEVICE, dtype=torch.bfloat16)
    prompt = torch.tensor([prompt_ids], device=DEVICE, dtype=torch.long)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Prefill picks the first generated token (anchor_token) from the
    # prompt's own logits -- but that forward has NOT consumed
    # anchor_token yet, so its hidden state / next-token prediction are
    # NOT what MTP or accept/reject need (those need the state/prediction
    # AFTER anchor_token has been processed). One more ordinary forward,
    # consuming anchor_token, produces both: anchor_hidden (MTP's
    # prev_hidden seed) and anchor_argmax (predicted_tokens[0] for round
    # 1's accept/reject) -- the exact same "process the new anchor" step
    # every later round performs at its own end, just needed once more
    # up front for the very first anchor.
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
    print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    t_load0 = time.perf_counter()
    model = load_qwen36_model(
        MODEL_PATH, device=DEVICE, max_seq_len=MAX_SEQ_LEN, enable_mtp=True
    )
    print(f"model loaded in {time.perf_counter() - t_load0:.1f}s")

    # -- Warmup: pay first-call JIT compiles (extend-mode at K, decode
    # mode for the MTP head's own attention, plain decode mode) on a
    # THROWAWAY prompt/state before timing anything real. ----------------
    warm_ids = tok("Warm up the kernels before timing.", return_tensors=None)["input_ids"]
    free_greedy_decode(model, warm_ids, 4)
    speculative_decode(model, warm_ids, 4, K)
    print("warmup done")

    results: dict = {
        "model": "nvidia/Qwen3.6-27B-NVFP4",
        "k": K,
        "n_tokens": N_TOKENS,
        "prompts": {},
    }

    for label, prompt_text in PROMPTS.items():
        prompt_ids = tok(prompt_text, return_tensors=None)["input_ids"]
        print(f"\n=== prompt={label!r} ({len(prompt_ids)} tokens) ===")

        ref_tokens, ref_time = free_greedy_decode(model, prompt_ids, N_TOKENS)
        spec_tokens, spec_time, rounds = speculative_decode(model, prompt_ids, N_TOKENS, K)

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

        ref_tok_s = N_TOKENS / ref_time
        spec_tok_s = len(spec_tokens) / spec_time
        speedup = spec_tok_s / ref_tok_s if ref_tok_s else float("nan")

        print(f"  non-speculative: {N_TOKENS} tok in {ref_time:.3f}s = {ref_tok_s:.2f} tok/s")
        print(
            f"  speculative:     {len(spec_tokens)} tok in {spec_time:.3f}s = "
            f"{spec_tok_s:.2f} tok/s, {len(rounds)} rounds"
        )
        print(
            f"  acceptance rate: {accept_rate:.3f} "
            f"({total_accepted}/{total_drafted} draft slots)"
        )
        print(f"  mean accepted per round (of K={K}): {mean_accept_per_round:.2f}")
        print(f"  speedup (tok/s ratio): {speedup:.2f}x")
        match_suffix = "" if match else f" (first diff at {first_diff})"
        print(f"  token sequences match: {match}{match_suffix}")

        entry = {
            "prompt_len": len(prompt_ids),
            "non_speculative": {"tokens_per_sec": ref_tok_s, "wall_s": ref_time},
            "speculative": {
                "tokens_per_sec": spec_tok_s,
                "wall_s": spec_time,
                "rounds": rounds,
                "acceptance_rate": accept_rate,
                "mean_accepted_per_round": mean_accept_per_round,
            },
            "speedup": speedup,
            "tokens_match": match,
        }
        if not match:
            entry["first_diff_index"] = first_diff
            entry["ref_tokens"] = ref_tokens
            entry["spec_tokens"] = spec_tokens
            print(f"  ref_tokens={ref_tokens}")
            print(f"  spec_tokens={spec_tokens}")
        results["prompts"][label] = entry

    out_path = Path(_ROOT) / ".bfdiag" / "runs" / "b3_mtp_e2e_acceptance_throughput.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")

    all_match = all(e["tokens_match"] for e in results["prompts"].values())
    print(f"\n== summary: all prompts token-match non-speculative path: {all_match} ==")
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
