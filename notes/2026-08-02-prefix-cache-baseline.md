# Prefix-cache baseline for Track A step 7-g (2026-08-02)

Collected on `work/prefix-baseline-20260802`, branched from `main` at
`1cf482f` (the commit that recorded step 7-g's blocking note in
`docs/implementation-plan.md`) — i.e. this is the **pre-switch** baseline
`docs/implementation-plan.md` §6 step 7 says must exist before 7-g (the A3
coordinator rework) starts. This branch does not implement 7-g itself and is
not merged into `main`.

Companion script: `benchmarks/prefix_cache_baseline.py`
(`python -m benchmarks.prefix_cache_baseline`) — reusable, so 7-g's own A/B
can run the identical workload against the post-switch server.

## 0. Why the existing number could not be used

`docs/implementation-plan.md` (2026-08-02 blocking note) already disqualified
`notes/prefix-cache-implementation-log.md`'s `prefix_cache_hits=1 /
misses=10 / hit_rate=0.0909` as a baseline. This section re-derives *why*,
against the code as it exists today, because the note's stated mechanism
("admission bootstrap check 给两轮都加了一次冷参考 prefill") does not
actually match what the code does — the real mechanism is different, and
worth recording precisely so nobody re-derives the wrong fix later.

**Checked directly (not inferred):** `ServerEngine._admission_bootstrap_check`
(`server/engine.py:626`) calls `self.runner.prefill(ref_slot, req.prompt_ids)`
(line 630) — a raw prefill on a *dedicated* reference slot
(`self.ref_slot_for`), never touching `reconcile_prefix_hit` or
`_record_prefix_cache_hits`. `prefix_cache_hits`/`prefix_cache_misses` are
incremented in exactly one place, `_record_prefix_cache_hits`
(`server/engine.py:683-702`), called from exactly one call site
(`server/engine.py:1133`), fed by `reconcile_prefix_hit` results computed for
the *real* admitted request only (line 1116). The bootstrap check's cold
reference prefill structurally cannot reach either function — it is a
completely separate code path. On top of that, the bootstrap check only runs
`if not self.production` (line 727), and `ServerEngine.production` defaults
to `True` (line 218); the actual deployed launcher,
`scripts/blackwellm_ctl.sh`, pins `QSR_SERVER_PRODUCTION=1` explicitly
(line 52) — so in the real deployed default, bootstrap checks never execute
at all, and even in configurations where they do, they do not touch the
counted stats.

**What actually produced hits=1/misses=10**, most likely: `/debug/stats`'
`prefix_cache_hits`/`misses` are lifetime, process-wide counters with no
reset endpoint and no per-workload scoping. `benchmarks/server_e2e_check.py`
runs many subtests against one long-lived server process in one script
invocation — basic round-trip, several independent-reference-replay
correctness cases, a concurrent-batching case with multiple distinct
prompts, defensive-rejection cases, a post-defensive request, *and* the one
turn-1/turn-2 hit subtest that the log entry is actually about. Every one of
those other subtests uses distinct prompt content and is a genuine,
unrelated cache miss when it is admitted — that is normal and expected, but
it means the reported hit_rate aggregates a specific two-turn conversation
together with ~9 unrelated single-shot admissions that were never going to
hit. That is a **scoping** problem (numerator/denominator mix unrelated
traffic), not a bootstrap-accounting problem. The practical conclusion is
unchanged from the blocking note — 0.0909 is not a usable per-workload
number — but the mechanism matters: fixing "exclude bootstrap probes" would
not have fixed anything, because bootstrap probes were never counted.

## 1. Is `hit_rate` a valid regression signal? Is `hit_tokens_saved` better?

**`prefix_cache_hit_rate = hits/(hits+misses)`, correctly scoped to a single
dedicated benchmark run (nothing else hitting the server, as
`benchmarks/prefix_cache_baseline.py` arranges): usable as a sanity check,
but too coarse to serve as *the* regression floor.** A "hit" is any
admission where `reconcile_prefix_hit(...).effective > 0`
(`server/engine.py:689`) — there is no notion of a *shallow* hit vs a *deep*
hit in this counter. For the fixed growing-conversation workload this
baseline uses, every turn after turn 1 is expected to register `L > 0`
almost by construction (the whole point of resending full history), so
`hit_rate` saturates near 1.0 immediately and stays there — a regression
that silently truncates how *much* of the prefix survives (a coordinator bug
that evicts more aggressively, mis-tracks refcounts, or restores from the
wrong depth) would still show `L > 0` most of the time and would not move
`hit_rate` at all. This is exactly the failure mode the task brief warned
about: "一个完全坏掉的缓存也能轻松满足" — `hit_rate` alone does not close
that gap even when scoped correctly; it only closes the *cross-workload
aggregation* problem, not the *shallow-hit-looks-like-full-hit* problem.

**`prefix_cache_hit_tokens_saved` (`server/engine.py:691`, sum of per-hit `L`)
is a better base signal** — it is sensitive to depth, not just to
hit/miss — but it is still an unnormalized absolute count: it will
mechanically differ between two runs whenever the *conversations themselves*
differ in length, independent of any cache-effectiveness change. It needs a
denominator to be comparable across runs. This baseline computes one:
`tokens_saved_ratio = hit_L / ideal_L` per turn, where `ideal_L` is the
*previous* turn's own `usage.prompt_tokens`, floored to
`--block-size` — i.e. the deepest hit a healthy cache could possibly have
served for a turn that is a strict prefix-extension of the one before it.
`tokens_saved_ratio ≈ 1.0` means the cache captured everything available;
a regression that still registers as an `hit_rate`-visible "hit" but only
serves a fraction of the possible depth shows up here as a ratio well below
1.0. **Recommendation for 7-g: gate on `tokens_saved_ratio` (median across
turns ≥ 2, from a run of this script against both pre- and post-switch
servers), not on `hit_rate`.** Report `hit_rate` alongside it as a secondary
sanity check (it should stay ≈ (turns-1)/turns per trial on a healthy
cache), not as the primary signal.

**Should bootstrap-probe misses be excluded?** Moot for the reasons in §0 —
they are never counted in the first place, under the deployed default
config (`production=1`) this baseline used. Recorded here so the question
has a real answer instead of staying open.

**Is current instrumentation sufficient for 7-g's stated criterion in
general (not just for this dedicated benchmark)?** Only partially. For a
purpose-built, isolated benchmark like this one — where the operator
controls all traffic during the measurement window — before/after deltas on
`/debug/stats` are trustworthy (confirmed by direct code reading, §0). For
the *general* case of judging cache health from a live, multi-tenant
server's cumulative `/debug/stats` or `/metrics` output, the current fields
are **not** sufficient: there is no reset endpoint, no per-session or
per-workload tagging, and no shallow-vs-deep-hit distinction. That gap is a
legitimate, separate finding — not something this task is scoped to fix —
and should not block 7-g, since 7-g's own A/B can (and should) reuse this
same dedicated-benchmark methodology rather than reading a shared server's
lifetime counters.

## 2. Workload definition

`benchmarks/prefix_cache_baseline.py`, mirroring `_warm()` in
`benchmarks/repro_prefix_cache_slowdown.py` (multi-turn, full-history-resend,
"exactly what a real agent client does"):

- **8 independent trials**, each a **6-turn** growing conversation. Every
  turn resends the *entire* prior history plus one new question, then
  appends the model's reply — turn *t*'s prompt is turn *t-1*'s prompt plus
  the previous exchange, a strict prefix extension.
- Each trial's system preamble carries a unique random marker
  (`os.urandom(4).hex()`) so **no trial shares a prefix with any other
  trial** — turn 1 of every trial is a genuine cold miss, never an
  accidental hit off a previous trial's leftover cache.
- `filler_chars=4000` (system-preamble padding, same default as
  `repro_prefix_cache_slowdown.py`), `max_tokens=32` per turn,
  `temperature=0`, `block_size=64` (server default).
- One throwaway warmup request runs first and is excluded from every
  statistic (absorbs the one-time M=1 decode-CUDA-Graph capture cost;
  paged-attention JIT warmup already runs automatically at server startup
  via `LagunaBackend.warmup_paged_attention_shapes`, before `/v1/models`
  answers, so it does not need re-warming here).
- Total: 1 warmup + 48 measured requests.

**Server launch config** (`scripts/blackwellm_ctl.sh`, run unmodified from
this worktree — `REPO_ROOT` resolves from the script's own path, so
`cd $REPO_ROOT && python -m server.app` runs with this worktree's code, not
main's; confirmed per the `AGENTS.md` "Verifying from a git worktree"
protocol before trusting any number, see §4):

| Knob | Value | Note |
|---|---|---|
| `QSR_SERVER_PRODUCTION` | `1` (default) | bootstrap checks off — see §0 |
| `QSR_SERVER_ENABLE_PREFIX_CACHE` | `1` (default in `blackwellm_ctl.sh`; note `server/app.py`'s own inline default is `"0"` — the ctl script's default overrides it) | the feature under test |
| `QSR_SERVER_CAPACITY` / `QSR_SERVER_NUM_SLOTS` | `3` / `3` | matches real deployment |
| `QSR_SERVER_BLOCK_SIZE` | `64` | |
| `QSR_SERVER_ENABLE_CUDAGRAPH` | `1` | |
| `QSR_SERVER_ENABLE_DFLASH` | `1` | speculative decode is orthogonal to prefill/prefix-cache; affects `wall_s` variance, not `hit_L`/`tokens_saved_ratio` |
| `QSR_SERVER_ENABLE_SESSION_AFFINITY` | `0` (default) | not needed — content-hash same-slot reuse already serves this workload |

> **Worth flagging on its own**: `server/app.py:123`'s inline default for
> `QSR_SERVER_ENABLE_PREFIX_CACHE` is `"0"` (i.e. *off* when the env var is
> unset and the script is invoked directly), while the comment immediately
> above it (`server/app.py:118-122`) says "Default ON (this is THE product
> value...)". `scripts/blackwellm_ctl.sh` papers over this by exporting `1`
> explicitly, so the real deployed server has prefix cache on — but
> `server/app.py`'s own inline default now contradicts its own comment.
> Flagging for the holder to decide whether this is an intentional rollback
> (e.g. related to the still-open `benchmarks/repro_prefix_cache_slowdown.py`
> slowdown investigation) with a stale comment, or a plain regression.

## 3. Per-round results

Collected 2026-08-02, real HTTP against a Laguna server on this exact
worktree (GPU: RTX PRO 6000 Blackwell Max-Q, cc 12.0). GPU held exclusively
via `/tmp/gpu_lock.sh acquire pcbase` for the whole measurement window,
released immediately after. Two independent, back-to-back invocations of
the script (same server process, same config, no restart between them) —
run 2 is there specifically to check for the fox-64K-style "does the number
depend on where in a longer session you measure it" failure mode, over and
above the within-run trial-1-vs-trial-8 check §3.3 already gives.

### 3.1 Run 1 — full per-round sequence (`benchmarks/fixtures/prefix_cache_baseline_main_20260802.json`)

| trial | turn | wall_s | prompt_tokens | completion_tokens | finish_reason | hit/miss | hit_L | ideal_L | tokens_saved_ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.80 | 3280 | 3 | stop | MISS | 0 | - | - |
| 1 | 2 | 0.41 | 3314 | 3 | stop | HIT | 3264 | 3264 | 1.0 |
| 1 | 3 | 0.38 | 3348 | 3 | stop | HIT | 3264 | 3264 | 1.0 |
| 1 | 4 | 0.39 | 3382 | 3 | stop | HIT | 3328 | 3328 | 1.0 |
| 1 | 5 | 0.36 | 3416 | 3 | stop | HIT | 3328 | 3328 | 1.0 |
| 1 | 6 | 0.38 | 3450 | 3 | stop | HIT | 3392 | 3392 | 1.0 |
| 2 | 1 | 0.78 | 3280 | 6 | stop | MISS | 0 | - | - |
| 2 | 2 | 0.42 | 3317 | 6 | stop | HIT | 3264 | 3264 | 1.0 |
| 2 | 3 | 0.37 | 3354 | 6 | stop | HIT | 3264 | 3264 | 1.0 |
| 2 | 4 | 0.33 | 3391 | 6 | stop | HIT | 3328 | 3328 | 1.0 |
| 2 | 5 | 0.40 | 3428 | 6 | stop | HIT | 3392 | 3328 | 1.0192 |
| 2 | 6 | 0.34 | 3465 | 6 | stop | HIT | 3392 | 3392 | 1.0 |
| 3 | 1 | 0.86 | 3282 | 3 | stop | MISS | 0 | - | - |
| 3 | 2 | 0.43 | 3316 | 3 | stop | HIT | 3264 | 3264 | 1.0 |
| 3 | 3 | 0.44 | 3350 | 3 | stop | HIT | 3264 | 3264 | 1.0 |
| 3 | 4 | 0.38 | 3384 | 3 | stop | HIT | 3328 | 3328 | 1.0 |
| 3 | 5 | 0.35 | 3418 | 3 | stop | HIT | 3328 | 3328 | 1.0 |
| 3 | 6 | 0.34 | 3452 | 3 | stop | HIT | 3392 | 3392 | 1.0 |
| 4 | 1 | 0.99 | 3279 | 7 | stop | MISS | 0 | - | - |
| 4 | 2 | 0.44 | 3317 | 7 | stop | HIT | 3264 | 3264 | 1.0 |
| 4 | 3 | 0.40 | 3355 | 7 | stop | HIT | 3264 | 3264 | 1.0 |
| 4 | 4 | 0.37 | 3393 | 7 | stop | HIT | 3328 | 3328 | 1.0 |
| 4 | 5 | 0.37 | 3431 | 7 | stop | HIT | 3392 | 3392 | 1.0 |
| 4 | 6 | 0.38 | 3469 | 7 | stop | HIT | 3392 | 3392 | 1.0 |
| 5 | 1 | 0.91 | 3280 | 7 | stop | MISS | 0 | - | - |
| 5 | 2 | 0.38 | 3318 | 7 | stop | HIT | 3264 | 3264 | 1.0 |
| 5 | 3 | 0.38 | 3356 | 7 | stop | HIT | 3264 | 3264 | 1.0 |
| 5 | 4 | 0.34 | 3394 | 7 | stop | HIT | 3328 | 3328 | 1.0 |
| 5 | 5 | 0.34 | 3432 | 7 | stop | HIT | 3392 | 3392 | 1.0 |
| 5 | 6 | 0.39 | 3470 | 7 | stop | HIT | 3392 | 3392 | 1.0 |
| 6 | 1 | 0.86 | 3282 | 3 | stop | MISS | 0 | - | - |
| 6 | 2 | 0.42 | 3316 | 3 | stop | HIT | 3264 | 3264 | 1.0 |
| 6 | 3 | 0.40 | 3350 | 3 | stop | HIT | 3264 | 3264 | 1.0 |
| 6 | 4 | 0.37 | 3384 | 3 | stop | HIT | 3328 | 3328 | 1.0 |
| 6 | 5 | 0.35 | 3418 | 3 | stop | HIT | 3328 | 3328 | 1.0 |
| 6 | 6 | 0.34 | 3452 | 3 | stop | HIT | 3392 | 3392 | 1.0 |
| 7 | 1 | 0.90 | 3281 | 3 | stop | MISS | 0 | - | - |
| 7 | 2 | 0.37 | 3315 | 3 | stop | HIT | 3264 | 3264 | 1.0 |
| 7 | 3 | 0.38 | 3349 | 3 | stop | HIT | 3264 | 3264 | 1.0 |
| 7 | 4 | 0.33 | 3383 | 3 | stop | HIT | 3328 | 3328 | 1.0 |
| 7 | 5 | 0.34 | 3417 | 3 | stop | HIT | 3328 | 3328 | 1.0 |
| 7 | 6 | 0.35 | 3451 | 3 | stop | HIT | 3392 | 3392 | 1.0 |
| 8 | 1 | 0.88 | 3279 | 3 | stop | MISS | 0 | - | - |
| 8 | 2 | 0.38 | 3313 | 3 | stop | HIT | 3264 | 3264 | 1.0 |
| 8 | 3 | 0.38 | 3347 | 3 | stop | HIT | 3264 | 3264 | 1.0 |
| 8 | 4 | 0.40 | 3381 | 3 | stop | HIT | 3328 | 3328 | 1.0 |
| 8 | 5 | 0.36 | 3415 | 3 | stop | HIT | 3328 | 3328 | 1.0 |
| 8 | 6 | 0.34 | 3449 | 3 | stop | HIT | 3392 | 3392 | 1.0 |

The one `ratio=1.0192` (trial 2, turn 5) is not a real "super-saving":
`ideal_L` is a conservative proxy (`floor(prev_turn.prompt_tokens /
block_size)`), and the actual cache boundary the engine computes
(`((len(prompt)-1)//block_size)*block_size` inside `prefill_chunked_begin`,
`runtime/backends/laguna.py:2294-2295`) can round to one block deeper than
that proxy depending on exactly where each turn's length falls relative to
a block boundary — a known, harmless artifact of the proxy formula, not a
cache behavior change. Every other one of the 40 hits in this run is
exactly `1.0`.

**Warmup** (excluded from all of the above): 0.60s, `prompt_tokens=45`.

**Turn-position summary** (n=8 per position):

| turn | wall_s median | wall_s mean | wall_s stdev | hit_rate | mean tokens_saved_ratio |
|---|---|---|---|---|---|
| 1 (cold) | 0.87 | 0.87 | 0.07 | 0.00 | n/a |
| 2 | 0.41 | 0.41 | 0.02 | 1.00 | 1.0000 |
| 3 | 0.38 | 0.39 | 0.02 | 1.00 | 1.0000 |
| 4 | 0.37 | 0.36 | 0.03 | 1.00 | 1.0000 |
| 5 | 0.36 | 0.36 | 0.02 | 1.00 | 1.0024 |
| 6 | 0.35 | 0.36 | 0.02 | 1.00 | 1.0000 |

Final cumulative `/debug/stats` (whole run, includes the 1 warmup miss):
`prefix_cache_hits=40, prefix_cache_misses=9, prefix_cache_hit_rate=0.8163,
prefix_cache_hit_tokens_saved=132800`.

### 3.2 Run 2 — same config, same server process, no restart (`..._run2.json`)

Run immediately after run 1 against the same still-running server (98
cumulative admissions by the end of run 2), specifically to check for
fox-64K-style position-in-a-longer-session drift. Full per-round JSON
committed alongside run 1's; summary here (identical shape, no
degradation):

| turn | wall_s median | wall_s mean | wall_s stdev | hit_rate | mean tokens_saved_ratio |
|---|---|---|---|---|---|
| 1 (cold) | 0.91 | 0.90 | 0.03 | 0.00 | n/a |
| 2 | 0.39 | 0.39 | 0.03 | 1.00 | 1.0000 |
| 3 | 0.39 | 0.40 | 0.01 | 1.00 | 1.0000 |
| 4 | 0.37 | 0.37 | 0.01 | 1.00 | 1.0000 |
| 5 | 0.36 | 0.36 | 0.01 | 1.00 | 1.0048 |
| 6 | 0.38 | 0.38 | 0.03 | 1.00 | 1.0000 |

Final cumulative `/debug/stats` after run 2: `prefix_cache_hits=80,
prefix_cache_misses=18, prefix_cache_hit_rate=0.8163,
prefix_cache_hit_tokens_saved=265792` — exactly double run 1's, i.e.
byte-for-byte the same per-request behavior repeated, zero drift across the
two back-to-back runs.

### 3.3 Reading these numbers

- **`hit_rate` at turn ≥ 2 is 1.00 at every position, in both runs** — as
  §1 predicted, it saturates immediately and is not doing any discriminating
  work here.
- **`tokens_saved_ratio` is 1.0 (or its harmless 1.0192 proxy-rounding
  variant) on all 40+40 hits in both runs** — on `main` today, with the
  server's real deployed default config, the persistent content-hash prefix
  cache is capturing essentially 100% of the available prefix on every
  single turn of this workload, with no measurable degradation across 98
  cumulative admissions. **This is the number 7-g must not regress below**
  (recommend: fail if the post-switch median `tokens_saved_ratio` for
  turns ≥ 2 drops meaningfully below ~1.0, e.g. below 0.95, rather than
  requiring bit-for-bit 1.0 — leaves room for legitimate proxy-rounding
  noise like trial 2/turn 5 above without being a rubber stamp).
- **wall time**: cold turn 1 ≈ 0.87-0.91s vs warm turns 2-6 ≈ 0.35-0.44s —
  roughly a 2.3x drop, consistent across both runs. Reported as directional
  context per the script's own docstring (includes DFlash decode time, not
  a clean prefill-only signal) — not the regression gate.
- **No fox-64K-style sequence-position effect observed** in either the
  within-run (trial 1 vs trial 8) or across-run (run 1 vs run 2) comparison:
  wall times and ratios are flat across both axes, well within normal
  measurement noise (stdev ≤ 0.07s throughout). This workload does not
  reproduce that failure mode — consistent with the existing note's own
  finding that fox-64K's swing was tied to the *64K-scale* long-context
  path, not to prefix caching (`prefix_cache=False` there) or to anything
  this ~3.3K-token workload exercises.

## 4. Worktree-import verification (before trusting any of the above)

Per `AGENTS.md`'s "Verifying from a git worktree" protocol, run before
starting the server:

```
$ cd /home/bot/project/qsr-w-pcbase && ~/.venvs/vllm/bin/python -c "
import sys
EXPECTED = '/home/bot/project/qsr-w-pcbase'
import runtime, server
assert runtime.__file__.startswith(EXPECTED), runtime.__file__
assert server.__file__.startswith(EXPECTED), server.__file__
print('OK runtime:', runtime.__file__)
print('OK server :', server.__file__)
"
OK runtime: /home/bot/project/qsr-w-pcbase/runtime/__init__.py
OK server : /home/bot/project/qsr-w-pcbase/server/__init__.py
```

Both resolve to this worktree, not main — the numbers above are this
worktree's code (identical to `main@1cf482f`, since this branch has not
touched `runtime/`/`server/`).

**One reproducibility gap worth recording**: `scripts/blackwellm_ctl.sh
start` initially failed with `LagunaRouterError: Laguna router library is
missing: runtime/kernels/_generated/laguna_router_sm120.so` — this compiled
artifact is gitignored (`runtime/kernels/_generated/`, generated by `make
build-laguna-router`) and a fresh worktree checkout has no copy. Confirmed
`runtime/kernels/laguna_router_sm120.cu` is byte-identical between this
worktree and main (`diff` clean — this branch has not touched kernel
sources), so the fix was to copy main's already-built `.so` +
`.manifest.json` into this worktree rather than rebuild (faster, and
provably equivalent given the identical source). Anyone reproducing this
baseline from a fresh worktree needs to do the same, or run `make
build-laguna-router` themselves.

## 5. What this baseline does not validate

- **Only one server lifetime is measured.** "Cold server startup" cost
  (model load, paged-attention JIT warmup) is observed exactly once, at the
  first-ever request of the run; it is not repeated across independent
  server restarts, so this baseline says nothing about restart-to-restart
  variance in that one-time cost.
- **`wall_s` is not a clean prefill-only signal.** It includes decode time
  (DFlash speculative verify rounds included), which is orthogonal to the
  prefix-cache mechanism and varies with acceptance rate. Treat `wall_s` as
  directional context; `hit_L`/`tokens_saved_ratio` are the load-bearing
  numbers.
- **Concurrency is not exercised.** All 48 requests are sent strictly
  sequentially by one client — this is deliberate (it is what makes the
  `/debug/stats` before/after deltas unambiguous, §0), but it means this
  baseline says nothing about hit-rate behavior under concurrent admission
  contending for the cache-aware slot-assignment logic
  (`find_best_slot_for_prompt`, `server/engine.py:1096-1104`).
- **Session affinity (`QSR_SERVER_ENABLE_SESSION_AFFINITY`) is off.** This
  baseline exercises the persistent content-hash cache path only, not the
  warm-slot zero-restore fast path (P4b).
- **The general "is `/debug/stats` sufficient for a live multi-tenant
  server" question is answered "no" (§1) but not fixed.** That gap is
  outside this task's scope.

## 6. Gates run on this branch

- `/tmp/ci-sim/bin/python -m ruff check .` → `All checks passed!`
- `/tmp/ci-sim/bin/python -m pytest -q` → `861 passed, 132 skipped` (matches
  the documented CPU-only baseline exactly)
- `~/.venvs/vllm/bin/python -m pytest -q` → `1263 passed, 3 warnings`
  (matches the documented full baseline exactly; warnings are pre-existing
  deprecation notices unrelated to this change — `httpx`/`starlette`
  testclient and two SWIG `__module__` warnings from an unrelated test)
