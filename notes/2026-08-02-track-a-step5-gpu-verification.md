# Track A step 5 GPU verification (2026-08-02)

Evidence for the four A6-style gates on `docs/implementation-plan.md` §6 step 5
(registry becomes the source of truth for backend selection; `ServerEngine.MODEL`
/ `BACKEND` / `server/app.py`'s `SERVER_MODEL_BACKEND` deleted — commit `d9ecbd1`,
merged into `work/trackA-20260801` at `cd11ab2`).

## 1. Greedy bit-exact: confirmed, not inferred

Reasoning alone says this must hold: step 5 touches only `ServerEngine.__init__`'s
parameter handling and `server/app.py`'s `lifespan()`, both of which run *before*
`_load_laguna_model()`; zero lines of `LagunaBackend`, `DFlashEngine`, or any kernel
changed. But this repo's own incident history is exactly "an obviously-safe change
that wasn't", so it was checked for real:

- Started the real HTTP server on the post-step-5 commit (`cd11ab2`, this worktree),
  sent three `/v1/completions` requests (`temperature=0`, `max_tokens=64`): "The
  capital of France is", "Explain the theory of relativity in one paragraph:",
  "def fibonacci(n):". Saved the exact `choices[0].text` for each.
- Created a throwaway detached worktree at the same commit, ran
  `git revert --no-commit d9ecbd1` in it (clean revert, two files auto-merged,
  zero conflicts — confirmed by re-reading `server/engine.py`/`server/app.py`
  afterward: the old `MODEL`/`BACKEND`/`SERVER_MODEL_BACKEND` are back, everything
  else from the `main` merge is untouched). Started a second real HTTP server there
  and sent the identical three requests.
- Result: all three `choices[0].text` values are **byte-for-byte identical**
  between the two servers, and `finish_reason` matches (`length` all three times).

## 2. Performance vs. baseline (fox-64K 353–368 / fox-4K 353–357 / galaxy-4K
395–401 / code-4K 341–359 tok/s) and 3. Acceptance rate (96.3–100%)

Measured via `bf exec benchmarks/acceptance_regression.py` against the warm
`bfdiag` daemon (provider=laguna, num-slots=3, blocks-per-slot=4096, block-size=64,
gpu-memory-utilization=0.88 — same config the `gpu-agent` worktree used earlier
today, confirmed by its own `pgrep` output before it released the GPU).

**Important scope note, read before trusting these numbers as "step 5's gate"**:
the bfdiag Laguna provider (`bfdiag/daemon/provider.py`) constructs `LagunaBackend`
/ `DFlashEngine` directly — it never goes through `ServerEngine` or
`server/app.py`'s `lifespan()` at all. This measurement exercises the exact same
code regardless of whether step 5 landed. It is included here as corroborating
evidence that the underlying engine is healthy on this exact checkout, not as a
verification of step 5's own diff (§1's before/after HTTP comparison is that).

First run used only two of the three env vars documented in
`notes/2026-07-31-session-summary.md` / `notes/2026-08-01-sparkinfer-patch-recovery-
and-repro.md` as "current best configuration" (missed
`SPARKINFER_PAGED_DECODE_GRAPH_CHUNK_PAGES=32` initially, then also missed
`QSR_VERIFY_CG_MAX_PAGES`/`QSR_VERIFY_CG_CTAS_PER_SM` on the very first attempt) —
each miss produced a materially different fox-64K number (255.7 → 305.3 → 288.5
tok/s across three daemon restarts), which is precisely the "not comparable"
trap `bf diff`/AGENTS.md warns about. Recorded here so the next person doesn't
repeat the three restarts: **set all three env vars before `bf daemon start`,
not just the two mentioned in the shorter recap**.

Final run (all three env vars set), suite v1.0, run record `756410d0d592` /
`37eae3d34bc3` / `5712b8379d8f` (the last of the three restarts):

| Workload | Baseline | Measured | Accept baseline | Accept measured | Verdict |
|---|---|---|---|---|---|
| fox-4K | 353–357 | 368.2 | 96.3–97.0% | 97.0% | tok/s slightly above range (matches the pattern in the recovery note's own repro: 357.1/360.8, also slightly above); accept ✅ |
| galaxy-4K | 395–401 | 410.2 | 100% | 100.0% | tok/s slightly above range (recovery note repro: 398.3/387.4, similar spread); accept ✅ exact |
| code-4K | 341–359 | 358.6 | 97.8% | 97.8% | ✅ both, accept exact match |
| fox-64K | 353–368 | 288.5 | 96.9% | 96.9% | **tok/s ~19% below floor; accept exact match** |

Acceptance rate matches the documented baseline almost to the digit on every
workload, including fox-64K (96.9% both) — strong evidence the *computation* is
unaffected. The fox-64K throughput gap is the one number that does not clear the
"within 3% of baseline" bar.

### fox-64K throughput: investigated, C-1 capacity fix ruled out by direct A/B, still not root-caused

Within a single run's raw fixture data (`benchmarks/fixtures/
acceptance_regression_20260802.json`, `fox-64K` entry): `steps` is identical (17)
across the warmup round and all three measured rounds — same amount of DFlash
work every time — but wall-clock time jumps from 13.8s (warmup) to a stable
~21.8s (all three measured rounds). Whatever is slower is not doing more work; it
is doing the same work slower, and it stabilizes after the first call rather than
drifting further, which reads more like a one-time state change (memory layout,
allocator behavior, or a clock/thermal step) than continuous degradation.

**Coordinator's specific hypothesis, checked directly**: the timeline lines up
(the 353–368 baseline was measured 2026-07-31/08-01, *before* `e9ee7de`'s C-1
verify-capacity fix, `dfa22f9`+`7903997`), fox-64K is exactly the long-context
workload the verify path is most sensitive on, and a bigger declared capacity
could plausibly change kernel selection, split-KV policy, or workspace memory
pressure. Tested directly rather than argued about:

- In a throwaway worktree (same commit otherwise), edited
  `SparkinferPrefillWorkspace._work_item_capacity`'s `mode == "verify"` branch
  to route through the same `eager_extend_work_items_capacity` call
  extend/decode use — the exact pre-`dfa22f9`/`7903997` behavior ("always sized
  via the extend estimator regardless of mode", per `dfa22f9`'s own commit
  message) — while leaving `declare_verify_capacity`, `_attempt_cg_capture`,
  `cg_status`, and `QSR_DFLASH_REQUIRE_CG`'s default untouched, per instruction
  not to conflate the throughput question with the (unrelated) CG-requirement
  default. A mechanical `git revert --no-commit` of both commits was tried
  first and conflicts with `85f7fb9` (the later `REQUIRE_CG` default flip,
  which touches the same function's neighborhood) — the manual edit was the
  clean way to isolate exactly one variable.
- Started a daemon from that worktree with the identical config (all three
  baseline env vars, same num-slots/blocks-per-slot/block-size/gpu-mem-util).
  It reached `READY` with `cg_status: {"verify": "captured"}` — the smaller,
  reverted capacity was still sufficient for verify-CG capture to succeed at
  this shape, so the A/B is a real apples-to-apples comparison, not "one side
  crashed."
- Ran the identical `benchmarks/acceptance_regression.py` suite:
  **fox-64K = 298.1 tok/s (accept 96.9%, `steps=17` all four calls) — not back
  to 353–368.** The per-round timing signature reproduced exactly: warmup
  13.684s, three measured rounds 21.938s/21.951s/22.092s. Same shape, same
  magnitude, same stabilization-after-first-call pattern, with the capacity
  fix's code path not even reachable.

**Conclusion (the coordinator's second enumerated outcome): the C-1 capacity
fix is ruled out as the cause.** Reverting it reproduces the same throughput
and the same timing signature, which is only possible if whatever is actually
responsible sits elsewhere. Filed as an open, unattributed performance item —
still not root-caused, but one specific, plausible suspect has now been
eliminated by direct measurement rather than by argument.

**Second hypothesis, checked by reading the code instead of another GPU restart
(the coordinator's inverted-warmup observation: first call *faster* than every
later identical call, not the usual slow-then-fast JIT-warmup shape — 22s→298
tok/s vs. 13.7s→~480 tok/s on the first call, well above baseline)**: prefix
cache doing same-slot reuse + SWA-ring rebuild on every call after the first,
and that path being slower than cold prefill at long context.

Ruled out without touching the GPU, by reading the exact call path
`benchmarks/acceptance_regression.py`'s `run_once()` exercises:

- `run_once()` calls `engine.generate_verify_only(..., enable_prefix_cache=False,
  slot=0)` on *every* call, warmup included (`runtime/backends/laguna_dflash.py`
  `generate_verify_only`, the `if enable_prefix_cache:` guard around
  `backend.find_prefix_match`/`reconcile_prefix_hit`/`prepare_exact_prefix_replay`
  — `prefix_len` stays 0 when the flag is `False`, unconditionally).
- With `prefix_len == 0`, `generate_verify_only` always takes the "No match:
  full reset and prefill" branch and calls `backend.prefill_with_aux(slot,
  prompt_ids)` with no `prefix_hit` argument, which defaults to `prefix_hit=0`
  → the cold-path branch in `prefill_with_aux`, not `_prefill_with_prefix_hit`
  (the same-slot-reuse-plus-SWA-ring-rebuild function the coordinator's
  hypothesis names). That function is provably unreached in this benchmark, on
  every single call, not just after the flag check but by the keyword default
  a level below it too.
- There is no separate provider/daemon-level prefix-cache construction switch
  to additionally flip: checked `LagunaBackend.__init__`'s full signature —
  `num_slots`, `block_size`, `blocks_per_slot`, no `enable_prefix_cache`
  parameter at all. "Prefix cache" here is entirely the caller's per-call
  decision, which is the one parameter already confirmed `False` above.
  `ServerEngine`'s own `enable_prefix_cache` construction flag is a different
  layer (governs whether `ServerEngine._finish_request` retains a slot's
  history across separate HTTP requests) that this benchmark's direct
  backend/engine calls never go through at all.
- One real, unconditional side effect worth naming so it isn't mistaken for
  the mechanism above: `LagunaBackend.reset_slot()` (called explicitly before
  every `run_once()`, and again internally inside `generate_verify_only` when
  `prefix_len == 0`) *always* saves `slot_committed_tokens`/`slot_kv_len` into
  `self._prefix_cache_tokens`/`_prefix_cache_kv_len` "for prefix cache", with
  no flag check — but its own docstring says it does not zero KV tensors and
  only touches small Python-side bookkeeping (an int and a list), so this is
  not a plausible multi-second cost regardless of whether it is "for" a
  feature this benchmark never reads back from.

**Conclusion: ruled out, and ruled out more cheaply than a GPU restart would
have — the exact code path the hypothesis names cannot execute given how this
benchmark calls the engine.** Confirmed both the top-level flag and the
one-level-deeper keyword default independently disable it, and confirmed there
is no separate construction-time switch to have missed.

Not chased further because: (a) `nvidia-smi` showed no other compute process, low
utilization/power/temperature at the time, so no obvious external contention; (b)
the bfdiag Laguna provider bypasses `ServerEngine` entirely regardless (§ above),
so this number was never attributable to step 5 either way; (c) the natural next
step for whoever picks this up is isolating the warmup-vs-measured timing jump
itself directly (e.g. `torch.cuda.memory_stats()`/allocator snapshots before and
after the first vs. second fox-64K call, on the theory that a repeated
alloc/free cycle of a ~64K-token KV region interacts with PyTorch's CUDA
caching allocator differently on the first request for that size class than on
later ones -- a generic allocator hypothesis, not a Laguna-specific one, and
notably consistent with "4K workloads don't show it" since their allocation
size class is common enough to already be warm from other traffic) rather than
another capacity- or cache-shaped hypothesis.

**On whether the 353–368 baseline itself is trustworthy**: asked directly, so
answered directly. The three per-workload numbers this session actually
reproduced closely (fox-4K, galaxy-4K, code-4K) all landed within or just
above their documented ranges, consistently, across every restart -- I have no
reason to doubt those three. fox-64K specifically is now a different story: this
session has shown that *where in a sequence of identical calls* you measure
fox-64K changes its throughput by ~60%, and the 07-31/08-01 baseline notes
record the summary numbers but not each run's warmup/measured call ordering in
enough detail to know which regime produced 353–368. It is entirely possible
07-31's baseline run happened to measure something closer to this session's
"first call" number, or used a different warmup convention, and 353-368 was
never a *steady-state* number for this specific workload to begin with. That
would make it not-wrong exactly, but not-comparable to a steady-state
measurement either -- exactly the shape of problem `bf diff` exists to catch,
one level removed (comparing against a written summary instead of another live
run, so there is no config/fingerprint to diff against). Recommend: next time
fox-64K is measured, record and publish the full warmup+per-round sequence (not
just the summary), so this question has an answer instead of a memory.

**Recommendation**: file as its own investigation item (perf, not correctness) —
does not block Track A step 5 or step 6. Bit-exact (§1) and acceptance rate
(§2 table) are the gates that actually test step 5's diff, and both hold.

## 4. `bf diff` before comparing

Applied in spirit rather than literally: the three fixture-comparability traps
this session actually hit and fixed were config mismatches against the
*documented* baseline (missing env vars, §2), not a same-tool `bf diff` between
two `bf ls` run IDs (this worktree's `.bfdiag` store had no prior runs to diff
against — each worktree keeps its own store, confirmed via `bf ls` returning
empty before this session's first run). All three of this session's own runs are
on record (`bf ls`) for a future `bf diff` if their comparability is ever
questioned.

## 5. C-LIVE (`make smoke`)

67/67 passed on the post-step-5, post-merge commit (`cd11ab2`), including a
genuinely cold cold-start check (`requests_completed_total` confirmed `0` via a
direct `/metrics` read immediately before running the script, not assumed).
