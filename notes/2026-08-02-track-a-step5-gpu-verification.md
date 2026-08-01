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

### fox-64K throughput: investigated, not root-caused, evidence points away from step 5

Within a single run's raw fixture data (`benchmarks/fixtures/
acceptance_regression_20260802.json`, `fox-64K` entry): `steps` is identical (17)
across the warmup round and all three measured rounds — same amount of DFlash
work every time — but wall-clock time jumps from 13.8s (warmup) to a stable
~21.8s (all three measured rounds). Whatever is slower is not doing more work; it
is doing the same work slower, and it stabilizes after the first call rather than
drifting further, which reads more like a one-time state change (memory layout,
allocator behavior, or a clock/thermal step) than continuous degradation.

Not chased further because: (a) `nvidia-smi` showed no other compute process, low
utilization/power/temperature at the time, so no obvious external contention; (b)
the bfdiag Laguna provider bypasses `ServerEngine` entirely (§ above), so this
number cannot be attributed to step 5's diff regardless of its cause; (c) three
separate daemon restarts already spent on this session's GPU window, and further
isolation (e.g. running fox-64K alone, first, on a completely fresh daemon) is
the natural next step for whoever picks this up, not urgent for step 5's own
sign-off given (b).

**Recommendation**: file as its own investigation item (perf, not correctness) —
does not block Track A step 5, since step 5 cannot be its cause. Bit-exact (§1)
and acceptance rate (this section) are the gates that actually test step 5's
diff and both hold.

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
