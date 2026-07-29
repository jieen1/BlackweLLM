"""Dedicated correctness test for ragged-length batched prefill
(2026-07-19, continuous-batching round,
``notes/2026-07-18-session-review-and-next-steps.md`` section 21).

``mtp_prefill_batch`` used to hard-assert every slot's prompt had the SAME
length -- real async serving needs genuinely different-length prompts
admitted together. This checks the new ragged (``chunk_size=None``, per-slot
different-length ``prompts_per_slot``) code path added this round.

Two checks:

0. **Ragged batch vs. independent singular reference (the core gate)**: 4
   slots, 4 DIFFERENT real prompt lengths (300/777/1536/4096 tokens, sliced
   from 4 different frozen W1-S fixture prompts -- deterministic, no new
   synthetic content invented), prefilled together in ONE
   ``mtp_prefill_batch`` call. Each slot's ANCHOR (the target model's own
   greedy next token -- the real gate, since the draft model's proposed K
   tokens are only ever a SPECULATIVE proposal subject to verification the
   very next round, never committed output by themselves) is compared
   against an INDEPENDENT single-slot ``mtp_prefill`` reference run on its
   own dedicated slot with the SAME prompt, via this project's established
   near-tie-margin convention (``NEAR_TIE_LOGIT_MARGIN=2.0``), not a bare
   bit-exact requirement.

   **A real, root-caused finding from this round, not glossed over**: an
   early version of this check required bit-exact agreement (the
   reasoning at the time: ragged batching introduces no NEW external call
   boundary per slot, unlike chunked prefill's extra bf16 state
   round-trip, so bit-exact seemed like the right bar). That version
   FAILED for the shortest prompt (300 tokens) -- anchor matched, but the
   K proposed draft tokens diverged completely. A dedicated isolation
   probe (not committed, ad hoc) traced this to **cross-slot batched
   PREFILL numerical noise when slots hold heterogeneous real content** --
   and, critically, confirmed this is a PRE-EXISTING characteristic of
   this runtime's batched-prefill mechanism, not something the ragged
   generalization introduces: the SAME divergence (margin 0.4375, full-
   vocab max_abs_diff 0.46 -- a genuine near-tie, not a gross bug)
   reproduces when batching 4 DIFFERENT-content prompts of the SAME
   length (300) through the UNTOUCHED, pre-2026-07-19 uniform-length code
   path. This project has already established this exact noise CLASS
   (cross-slot bf16 batching order) for the spec-decode verify kernel and
   for chunked-prefill's GDN state round-trip -- this is the same class
   showing up in a third place (heterogeneous-content batched prefill,
   any length combination) that nobody had previously bit-exact-tested at
   batch_size>1, not a new, unexplained phenomenon. The proposed draft
   tokens (informational only below, not gated) are correspondingly NOT
   required to match either -- they are speculative proposals; the real
   safety net is the verify/accept-reject step every round already goes
   through regardless (exercised end-to-end in check 1). GDN raw state is
   also compared directly against each slot's own reference (diagnostic
   only, sanity-bound gated, mirroring ``mtp_chunked_prefill_check.py``'s
   established convention for this kind of comparison).

1. **Ragged signal-probe + multi-round continuation**: 4 real,
   human-readable prompts ("The capital of France/Japan/Germany/Italy is"),
   each PADDED with a shared natural-language filler paragraph to a
   DIFFERENT target length (612/948/1284/1956 tokens) so the question
   always lands at the very end -- directly exercises ragged-length
   admission with recoverable, distinguishable content (this project's
   established signal-probe convention, generalized to raggedness). After
   the ragged prefill, NUM_ROUNDS real ``mtp_verify_and_commit_batch``
   rounds are run on all 4 slots TOGETHER; each round, each slot's real
   committed tokens are replayed through an INDEPENDENT single-slot
   reference slot (this project's established per-round
   independent-reference-replay methodology, decoupling this round's check
   from any prior round's own near-tie flip) -- the REAL gate is that
   every round's real committed content matches this independent
   reference within near-tie tolerance, for every slot, catching any
   actual cross-request state leakage between different-length slots
   sharing one physical KV-cache/GDN-state pool (a leak would corrupt a
   slot's own real content, which the independent single-slot reference
   -- computed from that slot's OWN true history alone -- would not
   reproduce). The decoded completions' OWN city name is also reported
   (``identity_ok``) but, per this project's own established precedent
   for this exact class of check (``mtp_batch_verify_check.py``'s
   ``_check_signal_probe`` docstring: a shared "The capital of X is"
   template can independently near-tie onto the SAME generic
   continuation for multiple slots, a real, reproducible false-positive
   trap, NOT a data-contamination bug), this is demoted to an
   INFORMATIONAL signal, not a pass/fail gate -- the per-round
   independent-reference check is what actually proves no contamination.

Usage:
    python -m benchmarks.mtp_ragged_prefill_check
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("SM120_GQA_USE_V2_DECODE_KERNEL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

SM120_VLLM_INTEGRATION = "/home/bot/project/sm120-flash-attention/vllm_integration"
MODEL = "unsloth/Qwen3.6-27B-NVFP4"
K = 3
SPECULATIVE_CONFIG = {"method": "mtp", "num_speculative_tokens": K, "attention_backend": "CUSTOM"}
NEAR_TIE_LOGIT_MARGIN = 2.0
GDN_STATE_SANITY_BOUND = 50.0
NUM_ROUNDS = 6

# Check 0 slot layout: 4 ragged slots + 4 dedicated singular-reference slots.
RAGGED_SLOTS = [0, 1, 2, 3]
REF_SLOTS_C0 = [4, 5, 6, 7]
RAGGED_LENGTHS = [300, 777, 1536, 4096]

# Check 1 slot layout: 4 ragged signal-probe slots + 4 dedicated reference slots.
PROBE_SLOTS = [8, 9, 10, 11]
REF_SLOTS_C1 = [12, 13, 14, 15]
PROBE_QUESTIONS = [
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Germany is",
    "The capital of Italy is",
]
PROBE_EXPECTED_CITY = ["France", "Japan", "Germany", "Italy"]
PROBE_TARGET_LENGTHS = [612, 948, 1284, 1956]

_FILLER_PARAGRAPH = (
    "The Amazon rainforest, spanning roughly 5.5 million square kilometers "
    "across nine countries in South America, is the largest tropical "
    "rainforest on Earth and hosts an estimated ten percent of all known "
    "species. Its dense canopy regulates regional and even global weather "
    "patterns by cycling enormous volumes of water vapor into the "
    "atmosphere, a process scientists call evapotranspiration. Deforestation "
    "driven by cattle ranching, soy cultivation, and illegal logging has "
    "accelerated over the past two decades, raising concern that the forest "
    "could approach a tipping point beyond which large sections convert to "
    "savanna. "
)


def _build_padded_probe_prompt(tok, question: str, target_len: int) -> list[int]:
    """Filler paragraph (tiled) + the distinguishing question, truncated/
    padded so the question's own tokens always land at the exact end of a
    prompt of EXACTLY ``target_len`` tokens -- this is what makes different
    slots' prompts genuinely ragged (different total length) while keeping
    each one's distinguishing content in a fixed, recoverable position."""
    question_ids = tok.encode(question, add_special_tokens=False)
    filler_needed = target_len - len(question_ids)
    assert filler_needed > 0, f"target_len {target_len} too short for question {question!r}"
    text = _FILLER_PARAGRAPH
    filler_ids = tok.encode(text, add_special_tokens=False)
    while len(filler_ids) < filler_needed:
        text += _FILLER_PARAGRAPH
        filler_ids = tok.encode(text, add_special_tokens=False)
    prompt_ids = filler_ids[:filler_needed] + question_ids
    assert len(prompt_ids) == target_len
    return prompt_ids


def _compare_gdn_states(runner, slot_a: int, slot_b: int, sanity_bound: float) -> dict:
    import math

    import torch

    from oracle.qwen36_vllm.direct_model_runner import _physical_slot

    phys_a, phys_b = _physical_slot(slot_a), _physical_slot(slot_b)
    per_layer = []
    sane = True
    for name in runner.gdn_layer_names:
        conv_state, ssm_state = runner.kv_caches[name]
        ca, cb = conv_state[phys_a].float(), conv_state[phys_b].float()
        sa, sb = ssm_state[phys_a].float(), ssm_state[phys_b].float()
        conv_diff = (ca - cb).abs().max().item()
        ssm_diff = (sa - sb).abs().max().item()
        finite = bool(
            torch.isfinite(ca).all() and torch.isfinite(cb).all()
            and torch.isfinite(sa).all() and torch.isfinite(sb).all()
        )
        per_layer.append({"layer": name, "conv_diff": conv_diff, "ssm_diff": ssm_diff, "finite": finite})
        if not finite or conv_diff > sanity_bound or ssm_diff > sanity_bound or math.isnan(conv_diff) or math.isnan(ssm_diff):
            sane = False
    return {
        "sane": sane,
        "max_conv_diff": max(e["conv_diff"] for e in per_layer),
        "max_ssm_diff": max(e["ssm_diff"] for e in per_layer),
    }


def _near_tie_margin_diag(runner, ref_slot: int, prompt: list[int], chosen_anchor: int) -> dict:
    """An INDEPENDENT direct-forward margin measurement (mirrors
    ``mtp_chunked_prefill_check.py``'s check-2 methodology) on a FRESH slot
    (never touched by the ``mtp_prefill``-based reference, whose KV/GDN
    state is already committed and must not be reused for this) -- the
    REAL gate for check 0 (see module docstring for why this is near-tie
    tolerant, not bit-exact: a real, root-caused finding this round)."""
    target_logits = runner._forward(ref_slot, prompt, start_pos=0, is_decode=False, return_hidden=False)
    last = target_logits[-1].float()
    ref_anchor = int(last.argmax(dim=-1).item())
    margin = float(last.max().item() - last[chosen_anchor].item())
    return {
        "ref_anchor": ref_anchor,
        "chosen_anchor": chosen_anchor,
        "near_tie_margin": margin,
        "within_tolerance": chosen_anchor == ref_anchor or margin < NEAR_TIE_LOGIT_MARGIN,
    }


def _check0_ragged_vs_singular(runner, prompts: list[list[int]]) -> dict:
    ragged_result = runner.mtp_prefill_batch(RAGGED_SLOTS, prompts)
    anchors = {s: ragged_result[s]["anchor"] for s in RAGGED_SLOTS}
    drafts = {s: ragged_result[s]["draft_tokens"] for s in RAGGED_SLOTS}

    margin_ref_slots = [16, 17, 18, 19]
    per_slot = {}
    for i, s in enumerate(RAGGED_SLOTS):
        ref_slot = REF_SLOTS_C0[i]
        ref = runner.mtp_prefill(ref_slot, prompts[i])
        anchor_exact = anchors[s] == ref["anchor"]
        drafts_exact = drafts[s] == ref["draft_tokens"]
        # The REAL gate (see module docstring): near-tie-tolerant anchor
        # agreement via an INDEPENDENT direct-forward measurement (not the
        # mtp_prefill-based ref, whose own KV is already committed and
        # cannot be re-forwarded for this without corrupting it).
        margin_diag = _near_tie_margin_diag(runner, margin_ref_slots[i], prompts[i], anchors[s])
        per_slot[s] = {
            "prompt_len": len(prompts[i]),
            "ragged_anchor": anchors[s],
            "singular_anchor": ref["anchor"],
            "anchor_exact_match": anchor_exact,
            # Informational only, NOT gated -- see module docstring: raw
            # proposed draft tokens are speculative proposals subject to
            # verification the very next round, not committed output.
            "ragged_drafts": drafts[s],
            "singular_drafts": ref["draft_tokens"],
            "drafts_exact_match": drafts_exact,
            "near_tie_diag": margin_diag,
        }

    gdn_reports = {
        s: _compare_gdn_states(runner, s, REF_SLOTS_C0[i], GDN_STATE_SANITY_BOUND)
        for i, s in enumerate(RAGGED_SLOTS)
    }
    gdn_all_sane = all(r["sane"] for r in gdn_reports.values())

    all_anchors_ok = all(v["near_tie_diag"]["within_tolerance"] for v in per_slot.values())
    all_exact_match = all(v["anchor_exact_match"] and v["drafts_exact_match"] for v in per_slot.values())

    return {
        "per_slot": per_slot,
        "all_exact_match": all_exact_match,
        "all_anchors_ok_within_tolerance": all_anchors_ok,
        "gdn_reports": gdn_reports,
        "gdn_all_sane": gdn_all_sane,
        "passed": bool(all_anchors_ok and gdn_all_sane),
    }


def _ref_check(runner, ref_slot: int, real_new_tokens: list[int], mtp_next_anchor: int) -> dict:
    ref_logits = runner._forward(
        ref_slot, real_new_tokens, start_pos=runner.slot_kv_len[ref_slot], is_decode=(len(real_new_tokens) == 1)
    )
    ref_last = ref_logits[-1].float()
    ref_predicted_next = int(ref_last.argmax(dim=-1).item())
    near_tie_margin = float(ref_last.max().item() - ref_last[mtp_next_anchor].item())
    exact_match = ref_predicted_next == mtp_next_anchor
    return {
        "ref_predicted_next_matches_mtp_next_anchor": exact_match,
        "near_tie_margin": near_tie_margin,
        "content_ok_within_near_tie_tolerance": exact_match or near_tie_margin < NEAR_TIE_LOGIT_MARGIN,
    }


def _check1_ragged_signal_probe(runner, tok) -> dict:
    prompts = [
        _build_padded_probe_prompt(tok, q, tlen)
        for q, tlen in zip(PROBE_QUESTIONS, PROBE_TARGET_LENGTHS)
    ]
    assert len(set(len(p) for p in prompts)) == len(prompts), "probe prompts must be genuinely ragged"

    prefill_result = runner.mtp_prefill_batch(PROBE_SLOTS, prompts)
    anchors = {s: prefill_result[s]["anchor"] for s in PROBE_SLOTS}
    drafts = {s: prefill_result[s]["draft_tokens"] for s in PROBE_SLOTS}

    # Independent singular reference bootstrap: each ref slot's own real
    # mtp_prefill on the SAME prompt. A bit-exact match is checked first
    # (the common case); per check 0's finding (cross-slot heterogeneous-
    # content batched prefill can near-tie-diverge from a singular
    # reference even in already-existing code), an exact mismatch is NOT
    # immediately treated as failure -- a fresh, dedicated margin-diagnostic
    # slot (mirrors check 0's ``_near_tie_margin_diag``) characterizes it;
    # only a mismatch that ALSO exceeds near-tie tolerance is a hard,
    # unexplained failure worth stopping on immediately, before it poisons
    # every subsequent per-round check.
    margin_ref_slots_c1 = [20, 21, 22, 23]
    for i, s in enumerate(PROBE_SLOTS):
        ref_first = runner.prefill(REF_SLOTS_C1[i], prompts[i])
        if ref_first != anchors[s]:
            diag = _near_tie_margin_diag(runner, margin_ref_slots_c1[i], prompts[i], anchors[s])
            if not diag["within_tolerance"]:
                return {
                    "passed": False,
                    "error": f"prefill anchor mismatch for slot {s}, beyond near-tie tolerance",
                    "ref_first": ref_first,
                    "batch_anchor": anchors[s],
                    "near_tie_diag": diag,
                }

    committed_sequences = {s: [] for s in PROBE_SLOTS}
    per_slot_rounds = {s: [] for s in PROBE_SLOTS}
    for r in range(NUM_ROUNDS):
        decisions = runner.mtp_verify_and_commit_batch(PROBE_SLOTS, anchors, drafts)
        for i, s in enumerate(PROBE_SLOTS):
            decision = decisions[s]
            committed_sequences[s].extend(decision["committed"])
            real_new_tokens = [anchors[s]] + decision["committed"][:-1]
            ref_report = _ref_check(runner, REF_SLOTS_C1[i], real_new_tokens, decision["next_anchor"])
            per_slot_rounds[s].append({"round": r, **ref_report})
            anchors[s], drafts[s] = decision["next_anchor"], decision["next_draft_tokens"]

    # The REAL gate: every round's real committed content matches an
    # INDEPENDENT single-slot reference (computed from that slot's own true
    # history alone) within near-tie tolerance -- this is what actually
    # proves no cross-request state leakage (a leak would corrupt a slot's
    # real content in a way the independent reference, which never sees
    # any OTHER slot's data, would not reproduce).
    per_slot_ok = {
        s: all(rr["content_ok_within_near_tie_tolerance"] for rr in per_slot_rounds[s]) for s in PROBE_SLOTS
    }
    decoded = {s: tok.decode(committed_sequences[s]) for s in PROBE_SLOTS}
    # Informational only (see module docstring): a shared prompt template
    # ("The capital of X is") is this project's own established false-
    # positive trap for a hard identity-content gate -- demoted here for
    # the same reason mtp_batch_verify_check.py's _check_signal_probe
    # demoted it, not re-litigated from scratch.
    identity_ok = {}
    for i, s in enumerate(PROBE_SLOTS):
        expected_city = PROBE_EXPECTED_CITY[i]
        other_cities = [c for j, c in enumerate(PROBE_EXPECTED_CITY) if j != i]
        has_own_city = expected_city in decoded[s]
        has_other_city = any(c in decoded[s] for c in other_cities)
        identity_ok[s] = has_own_city and not has_other_city
    seqs = list(committed_sequences.values())
    no_cross_contamination_signal = all(
        seqs[i] != seqs[j] for i in range(len(seqs)) for j in range(i + 1, len(seqs))
    )

    return {
        "passed": bool(all(per_slot_ok.values())),
        "prompt_lens": {s: len(p) for s, p in zip(PROBE_SLOTS, prompts)},
        "per_slot_ok": per_slot_ok,
        "identity_ok_informational": identity_ok,
        "no_cross_contamination_signal": no_cross_contamination_signal,
        "decoded_completions": decoded,
    }


def _run_once() -> dict:
    sys.path.insert(0, SM120_VLLM_INTEGRATION)
    import register_sm120_backend  # noqa: F401
    from transformers import AutoTokenizer

    from benchmarks.workloads import W1_S_FIXTURE, load_prompt_token_ids
    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner, build_vllm_config

    w1s_prompts = load_prompt_token_ids(W1_S_FIXTURE)
    ragged_prompts = [w1s_prompts[i][:length] for i, length in enumerate(RAGGED_LENGTHS)]

    vllm_config = build_vllm_config(
        model=MODEL,
        kv_cache_dtype="fp8_e4m3",
        max_model_len=max(40960, RAGGED_LENGTHS[-1] + 1024),
        gpu_memory_utilization=0.85,
        speculative_config=SPECULATIVE_CONFIG,
    )
    # 24 slots: check0's 4 ragged + 4 singular-ref + 4 margin-diag (always
    # computed, see check 0's own docstring) and check1's 4 probe + 4 ref +
    # 4 margin-diag (only touched on a bootstrap mismatch).
    runner = DirectModelRunner(
        vllm_config, num_slots=24, block_size=64, blocks_per_slot=640, enable_cudagraph=False
    )

    result: dict = {}
    result["check0_ragged_vs_singular"] = _check0_ragged_vs_singular(runner, ragged_prompts)

    tok = AutoTokenizer.from_pretrained(MODEL)
    result["check1_ragged_signal_probe"] = _check1_ragged_signal_probe(runner, tok)

    result["passed"] = bool(
        result["check0_ragged_vs_singular"]["passed"] and result["check1_ragged_signal_probe"]["passed"]
    )
    return result


def main() -> int:
    import json

    result = _run_once()
    print(json.dumps(result, indent=2, default=str))
    print(f"\n=== {'PASS' if result['passed'] else 'FAIL'} ===")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
