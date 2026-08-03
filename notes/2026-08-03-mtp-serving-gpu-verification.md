# MTP serving-path GPU verification: pairing fix moves acceptance, but MTP loses badly to CUDA Graph on throughput

Date: 2026-08-03 · Worktree `work/mtp-serving-20260803` @ merge of `088698f` ·
Model: `unsloth/Qwen3.6-27B-NVFP4` (standard checkpoint) · Single RTX PRO 6000
Blackwell Max-Q under `/tmp/gpu_lock.sh` · K=4

## Headline

The (token, hidden) pairing fix (`runtime/backends/qwen36_mtp.py`'s module
docstring) is real and moves acceptance in the expected direction, but by a
modest amount, not a dramatic one. Separately, and more decisively: **MTP's
verify/draft round is 100% eager and never touches the CUDA Graph decode
path, so turning MTP on with CUDA Graph enabled makes the server's real
tok/s ~3.6x *worse*, not better.** Acceptance rate was not the binding
constraint this round -- throughput composition with CUDA Graph is.

## 1. Acceptance rate, standard checkpoint, K=4 (`scripts/b3_mtp_e2e_acceptance_throughput.py`, fixed)

Same script, same prompts, same K=4, same checkpoint as the recorded
baseline (`notes/2026-08-03-mtp-acceptance-on-standard-checkpoint.md`) --
only the pairing fix differs.

| prompt | accept rate (before → after) | mean accepted/round (before → after) | e2e speedup, eager (before → after) |
|---|---:|---:|---:|
| prose | 0.300 → **0.385** | 1.20 → **1.54** | 0.83x → 0.97x |
| code | 0.417 → **0.455** | 1.67 → **1.82** | 1.07x → 1.10x |

Token sequences match the non-speculative path exactly on both prompts
(`tokens_match: True`), same as before the fix. The fix helped -- prose
acceptance up 28% relative, code up 9% -- but the ceiling documented in
`notes/2026-08-02-mtpfix-historical-comparison.md` (teacher-forced,
zero-compounding single-step accuracy 62.9%/82.4%/71.1% -- itself measured
with the *unfixed* pairing, so possibly still an underestimate) stands:
this checkpoint's draft head is not going to reach anywhere near 100%
regardless of the pairing fix. Raw data: `.bfdiag/runs/
b3_mtp_e2e_acceptance_throughput.json` (gitignored, not checked in).

## 2. Does `enable_mtp=True` load against the standard checkpoint?

**Yes.** `load_qwen36_model(standard_checkpoint_path(), enable_mtp=True)`
loads cleanly (267s cold, 48-70s once `~/.cache/sparkinfer`/page cache are
warm). One informational-only warning: `k_scale x16, v_scale x16` checkpoint
tensors reach no consumer -- expected, this run doesn't enable FP8 KV, not a
defect.

## 3. Through the server, CUDA Graph on: e2e tok/s, MTP off vs on

Methodology matches `notes/2026-08-03-cudagraph-vs-eager-decode-throughput.md`:
same prompt ("merge two sorted lists"), `capacity=1`, `num_slots=2`,
`QSR_SERVER_ENABLE_CUDAGRAPH=1` in both arms, 1 warm request discarded, 3
timed streaming runs, median reported. This model spends nearly its entire
token budget on `<think>` at `max_tokens=256` (confirmed empirically, matches
`notes/2026-08-03-std-model-serving-acceptance.md`'s own finding) -- tok/s is
measured over ALL text deltas (reasoning + content), token count taken by
re-tokenizing the accumulated text after the stream completes (not by
counting SSE events, which do not correspond 1:1 with tokens once MTP can
commit several tokens per event).

| | MTP off (CG captured) | MTP on (K=4, resync off) | ratio |
|---|---:|---:|---:|
| tok/s (median of 3, max_tokens=256) | **28.0-28.1** (2 independent runs, 6 total) | **7.80** | **0.28x** |

28.0-28.1 corroborates the recorded 28.85 CUDA Graph baseline (this
harness, different prompt-completion split between reasoning/content,
~3% off, healthy agreement). **MTP-on is ~3.6x SLOWER**, not faster.

**Root cause, structural, not a bug to fix quietly**: `Qwen36MTPEngine.round`
calls `self.model(anchor_input, state)` (plain eager) and
`model.verify_forward(...)` (eager, extend-mode) -- neither goes through
`Qwen36Backend.decode_batch_sampled`/`_decode_forward_batched`, which is the
only path `capture_decode_cuda_graph()` captures. Once
`has_speculative_decode` is true, `classify_decode_slots` routes **every**
active slot's round through the MTP branch (same split DFlash already
established for Laguna) -- so with MTP on, **no request's decode ever
replays the captured graph at all**. This matches the task's own framing
exactly ("if MTP and CG cannot coexist yet, say so explicitly rather than
silently disabling capture") -- capture is not disabled and does not fail,
it is simply never reached once MTP is on. DFlash's own Laguna
implementation has its own dedicated draft/verify CUDA Graphs
(`DFlashEngine._verify_cg`/`_draft_cg`, `runtime/backends/laguna_dflash.py`)
for exactly this reason; `Qwen36MTPEngine` has no analogue, and did not
attempt to build one in this landing.

**Conclusion: MTP should not be enabled in production as currently
implemented.** Even with the pairing fix's acceptance improvement, an eager
verify/draft round losing the 4.7x CUDA Graph advantage on every single
decode step is not something a K=4, ~0.4 acceptance rate can pay back.
`QSR_SERVER_ENABLE_MTP` stays default OFF, correctly.

## 4. C-LIVE with MTP on

**66/67** (`scripts/c_live_smoke.py` against a live server, `QSR_TOOL_CALL_
PARSER=qwen3_coder`, same config as above). Matches/exceeds the recorded
64/67 MTP-off baseline. The one failure is the SAME already-documented
genuine model behavior from `notes/2026-08-03-std-model-serving-acceptance.md`
(`/v1/completions` raw-completion check: the model spontaneously emits
`<think>` even with no chat template, which `/v1/completions` is correct to
pass through unstripped). The other two failures recorded in that baseline
(`openai multi-turn`/`anthropic array content`, both `max_tokens=512`
thinking-budget-overflow cases) did not reproduce this run -- plausibly just
run-to-run variance in exactly how much of the budget thinking consumes for
those specific prompts, not attributable to MTP one way or the other.
**No MTP-caused C-LIVE regression.**

## 5. Token-identity through the server, long generation (~2000 tokens)

The 32-token standalone-script check (item 1) passes cleanly. A separate,
longer (`max_tokens=2048`) same-prompt check through the live server (MTP on
vs MTP off) was added to get past the reasoning-only phase into real answer
content, and **it does diverge** -- first divergence at character 215 of a
shared prefix. Traced before concluding anything:

| comparison | first divergence |
|---|---:|
| MTP-on (eager anchor) vs MTP-off (CUDA Graph decode) | char 215 |
| plain eager, MTP off, **no CUDA Graph**, vs MTP-off+CUDA-Graph | char 215 (identical position) |
| MTP-on vs plain eager, MTP off, no CUDA Graph (**both eager**) | char 307 |

The char-215 divergence is **not attributable to MTP at all**: it occurs,
at the exact same position, between plain eager decode and CUDA-Graph-replay
decode with MTP completely absent from both arms. This is the already-
documented `notes/2026-08-02-eager-verify-cg-verify-divergence.md`
phenomenon ("CG-replay decode vs eager decode disagreeing... while both are
correct, cos>=0.999997") -- a pre-existing property of this runtime's decode
kernels, not something this landing introduced.

There **is** a second, smaller, MTP-attributable divergence at char 307
(MTP's eager anchor+extend-mode-verify path vs plain eager sequential
decode, both non-CUDA-Graph). This one is the phenomenon
`scripts/b3_mtp_e2e_acceptance_throughput.py`'s own module docstring
pre-registered as "a known potential source of disagreement that would NOT
be a bug": full-attention layers process K draft positions in ONE
extend-mode kernel call during verify vs one decode-mode call per position
during plain sequential decode -- different kernel schedules computing the
same math, occasionally disagreeing at a near-tied logit after enough
tokens for one to occur (32 tokens: never seen; ~90 tokens in this run:
seen once). This does not violate the accept/reject algorithm's own
guarantee (a token is only ever committed when it exactly matches the
target's OWN prediction from that same verify call) -- it means the
target's own prediction at a near-tied position is not perfectly kernel-
invariant, which is a fact about the decode kernels, established
independently of MTP, now also visible through MTP's own eager verify path.

**Stated plainly, not softened**: greedy MTP through the server is **not**
bit-exact-identical to non-speculative decode at long (~2000-token)
generation lengths. It IS exactly token-identical at the 32-token scale the
correctness gate's own reference script tests, and the long-generation
divergence traces cleanly to two independently-pre-existing, already-
documented decode-kernel numerical-noise sources -- not a defect in the
accept/reject implementation this landing added. Whether "not bit-exact
over ~2000 tokens, for reasons that predate and are independent of MTP" is
an acceptable answer to "greedy speculative must commit token-for-token
identical sequences" is a judgment call the numbers alone don't resolve;
recorded here without resolving it either way.

## 6. Resync A/B (`QSR_SERVER_MTP_RESYNC`)

**Not run this session** -- out of GPU time budget after items 1-5 (lowest
explicit priority, "if you have budget left"). Still no data on it.
`QSR_SERVER_MTP_RESYNC` stays default OFF.

## What changed in this repo from this session

`scripts/b3_mtp_e2e_acceptance_throughput.py`: the same pairing fix
`runtime/backends/qwen36_mtp.py` already carries, applied to this reference
script so its numbers are comparable to the recorded 0.300/0.417 baseline
(item 1's table). No other code changes -- this was a measurement session,
not an implementation one.

## Harness

One-time, not checked in (matches this repo's own precedent for `$CLAUDE_
JOB_DIR/tmp/cg_vs_eager.py`): `/tmp/mtp_gpu_verify/serve_bench.py`. Starts
`python -m server.app` as a subprocess with the config under test, waits for
`/health`, runs warmup + N timed streaming chat completions (tok/s from
first text delta to last, token count by re-tokenizing after the stream),
optionally a larger-budget identity-check call and `scripts/c_live_smoke.py`
against the live server, then tears the server down. Raw JSON results for
every arm are under `/tmp/mtp_gpu_verify/*.json` on this machine (not
committed).
