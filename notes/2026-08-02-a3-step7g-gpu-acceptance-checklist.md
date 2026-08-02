# A3 step 7-g: GPU acceptance checklist (2026-08-02)

Written on `work/7g-20260802` (worktree `/home/bot/project/qsr-w-7g`), branched
from `main` at `9e5c340`. **This note documents the checklist; it does not run
it.** This task is scoped zero-GPU (a separate agent is on the GPU running the
Qwen3.6 fact baseline, which is the critical path today) and does not acquire
`/tmp/gpu_lock.sh`. Everything below is written so whoever picks up GPU
verification next can run it without re-deriving commands or thresholds.

## 0. What changed, in one paragraph

`server/app.py`'s `lifespan()` already called `resolve_checkpoint(...)`,
getting back a `Resolution` with `.spec: ArchitectureSpec`, but only
`.backend` was passed to `ServerEngine(...)` -- `.spec` was read and
discarded. This step threads `.spec` through as `architecture_spec=`, and
`ServerEngine.slot_resources` (a new property returning a
`runtime.slot_resource_manager.SlotResourceManager` bound to
`(self.runner, self.architecture_spec)`) is what `server/engine.py`'s two
`capabilities.prefix_cache` call sites (`find_best_slot_for_prompt`,
`reconcile_prefix_hit`) now read instead of `self.runner` directly. For
every checkpoint this runtime serves today, `needs_two_cache_families` is
`False`, so `SlotResourceManager` is a pure forward -- see
`docs/a3-cache-coordinator-design.md` §5 for why that branch's behavior is
argued to be identical to calling the backend directly, and this task's own
CPU-level verification (§1 below) for what was actually checked, not just
argued, before any GPU time is spent.

Diff footprint: `server/app.py` (+9 lines, one new kwarg), `server/engine.py`
(+91 lines: one property, one module-level default-spec constant, two call
sites swapped), `runtime/slot_resource_manager.py` (docstring only, no code
change), `tests/test_engine_prefix_cache_admission.py` (+74 lines, two new
tests proving the coordinator is actually in the call path, not just that
the outcome is unchanged). No line of `runtime/backends/laguna.py` or any
kernel changed.

## 1. CPU-level evidence already collected (read this before spending GPU time)

- `/tmp/ci-sim/bin/python -m ruff check .` → `All checks passed!`
- `/tmp/ci-sim/bin/python -m pytest -q` → `871 passed, 152 skipped` (matches
  the documented baseline exactly -- `tests/test_engine_prefix_cache_
  admission.py`'s two new tests are inside a `pytest.importorskip("torch")`
  module, so this CPU-only job still skips the whole file as one unit, same
  as before).
- `timeout --signal=ABRT 400 ~/.venvs/vllm/bin/python -X faulthandler -m
  pytest -q` → `1299 passed` in ~77s (baseline `1297 passed` + 2 new tests;
  no hang, ran to completion).
- **Red/green check on the new wiring tests, not just a green run**:
  temporarily reverted the two call sites back to `self.runner.
  find_best_slot_for_prompt`/`.reconcile_prefix_hit` (bypassing the
  coordinator) and reran `tests/test_engine_prefix_cache_admission.py::
  TestCoordinatorWiring` -- it went red (`spy_find_slot.call_count == 0`,
  expected `2`). Restored the real wiring; green again. This is the
  concrete evidence that the coordinator is actually in the call path, not
  merely that pre-existing assertions about admission outcomes still pass
  (which they also do, unmodified, from 7-b).
- `__pycache__` cleared with `find . -name __pycache__ -type d -not -path
  './.venv/*' -exec rm -rf {} +` before every one of the above runs (the
  stale-bytecode trap this task's brief warned about).

None of this substitutes for the four GPU gates below -- it is the reason to
expect them to pass, not a replacement for running them.

## 2. Before touching the GPU

```bash
# 1. Confirm GPU is free / coordinate with whoever holds /tmp/gpu_lock.sh.
#    Do NOT acquire it if the Qwen3.6 fact-baseline workstream is running.
cat /tmp/gpu_lock.sh   # inspect; use its acquire/release/status subcommands

# 2. Clear stale bytecode (do this on every worktree you start a server from).
find . -name __pycache__ -type d -not -path './.venv/*' -exec rm -rf {} +

# 3. Worktree-import sanity check (AGENTS.md "Verifying from a git worktree" --
#    ~/.venvs/vllm's editable install hard-wires runtime/server to the MAIN
#    worktree's path unless invoked with -m or from the right cwd).
~/.venvs/vllm/bin/python -c "
import sys
EXPECTED = '/home/bot/project/qsr-w-7g'
import runtime, server
assert runtime.__file__.startswith(EXPECTED), runtime.__file__
assert server.__file__.startswith(EXPECTED), server.__file__
print('OK runtime:', runtime.__file__)
print('OK server :', server.__file__)
"

# 4. Laguna router .so is gitignored -- a fresh worktree checkout has no copy.
#    Confirm runtime/kernels/laguna_router_sm120.cu is byte-identical to main
#    (it is -- this branch never touches runtime/kernels/) and copy main's
#    prebuilt artifact rather than rebuild:
diff /home/bot/project/qwen-sm120-runtime/runtime/kernels/laguna_router_sm120.cu \
     runtime/kernels/laguna_router_sm120.cu   # expect: no output
cp /home/bot/project/qwen-sm120-runtime/runtime/kernels/_generated/laguna_router_sm120.so* \
   runtime/kernels/_generated/   2>/dev/null || true
```

## 3. Gate 1 -- Greedy bit-exact

**Method**: same "real HTTP server, before vs. after, byte-for-byte" pattern
`notes/2026-08-02-track-a-step5-gpu-verification.md` §1 and
`notes/2026-08-02-track-a-step6-gpu-verification.md` §2 used, because the
change here is the same shape (a call-path indirection with no argued
numeric difference) as both of those.

```bash
# "After" = this worktree (work/7g-20260802).
cd /home/bot/project/qsr-w-7g
scripts/blackwellm_ctl.sh start
# send fixed prompts, temperature=0, max_tokens=64, save choices[0].text + finish_reason:
#   "The capital of France is"
#   "Explain the theory of relativity in one paragraph:"
#   "def fibonacci(n):"
#   "Write a haiku about the ocean."
#   "The three laws of thermodynamics are:"
scripts/blackwellm_ctl.sh stop

# "Before" = a throwaway detached worktree at main's tip this branch forked from.
git -C /home/bot/project/qwen-sm120-runtime worktree add --detach /tmp/qsr-7g-before 9e5c340
cp runtime/kernels/_generated/laguna_router_sm120.so* /tmp/qsr-7g-before/runtime/kernels/_generated/
cd /tmp/qsr-7g-before && scripts/blackwellm_ctl.sh start
# send the identical five prompts, save the identical two fields
scripts/blackwellm_ctl.sh stop
git -C /home/bot/project/qwen-sm120-runtime worktree remove /tmp/qsr-7g-before
```

**Expected**: all five `choices[0].text` values byte-for-byte identical
between the two servers, `finish_reason` matching on every prompt (`length`
at `max_tokens=64` for all five, per the step 5/6 precedent).

**Judgment**: any single-byte difference fails the gate -- go find which of
the two touched call sites diverges before assuming it is noise.

## 4. Gate 2 -- Acceptance rate (96.3-100%)

**Method**: `bf exec benchmarks/acceptance_regression.py` against a warm
`bfdiag` daemon, same invocation `notes/2026-08-02-track-a-step5-gpu-
verification.md` §2 used. Set all three env vars documented there
BEFORE `bf daemon start` (`SPARKINFER_PAGED_DECODE_GRAPH_CHUNK_PAGES=32`,
`QSR_VERIFY_CG_MAX_PAGES`, `QSR_VERIFY_CG_CTAS_PER_SM` -- missing any one
produced a materially different fox-64K number last time, a config-mismatch
trap, not a real regression).

```bash
cd /home/bot/project/qsr-w-7g
export SPARKINFER_PAGED_DECODE_GRAPH_CHUNK_PAGES=32
export QSR_VERIFY_CG_MAX_PAGES=<value from notes/2026-07-31-session-summary.md>
export QSR_VERIFY_CG_CTAS_PER_SM=<same>
bf daemon start --provider laguna --num-slots 3 --blocks-per-slot 4096 \
    --block-size 64 --gpu-memory-utilization 0.88
bf exec benchmarks/acceptance_regression.py
```

**Important scope note (carried over from the step 5 note, still true
here)**: the bfdiag Laguna provider constructs `LagunaBackend`/`DFlashEngine`
directly -- it never goes through `ServerEngine` or `server/app.py`'s
`lifespan()` at all, so this measurement cannot by itself prove step 7-g's
diff is inert. It corroborates that the underlying engine is healthy on this
checkout; §3's real-HTTP-server bit-exact check (which *does* go through
`ServerEngine`) is what actually exercises 7-g's own diff for acceptance-rate
purposes, in the sense that it's the same 5-request round trip through the
new call path. If a dedicated acceptance-rate re-measurement of the HTTP
path itself is wanted, run it through `scripts/blackwellm_ctl.sh start` +
whatever client drives multi-round MTP verify/commit at scale (there is no
existing benchmark that does this through the HTTP layer specifically;
`acceptance_regression.py` is the one that exists and it bypasses
`ServerEngine`).

**Expected**, per-workload (baseline table, `notes/2026-08-02-track-a-
step5-gpu-verification.md` §2 -- reproduce this table, don't just eyeball a
pass/fail):

| Workload | tok/s baseline | Accept baseline |
|---|---|---|
| fox-4K | 353-357 | 96.3-97.0% |
| galaxy-4K | 395-401 | 100% |
| code-4K | 341-359 | 97.8% |
| fox-64K | 353-368 (see caveat below) | 96.9% |

**Judgment**: acceptance rate within (or matching to the digit) the baseline
range on every workload is the gate; this step's diff has no plausible
mechanism to move acceptance rate (it touches only which object answers
`reconcile_prefix_hit`/`find_best_slot_for_prompt`, both upstream of any
DFlash verify/commit decision) so any acceptance-rate delta here would be a
signal to stop and investigate, not average away. tok/s is directional
context, not the gate -- fox-64K specifically has an open, unattributed
~19% throughput gap versus its documented baseline (ruled out: the C-1
verify-capacity fix; see that note's own investigation) that predates this
step and is not this step's problem to fix or to be blocked by.

## 5. Gate 3 -- Prefix-cache hit rate does not regress

**Method**: `benchmarks/prefix_cache_baseline.py`, run identically against
this worktree's server, A/B'd against the pre-switch baseline already
collected in `notes/2026-08-02-prefix-cache-baseline.md`.

```bash
cd /home/bot/project/qsr-w-7g
export QSR_SERVER_PRODUCTION=1
export QSR_SERVER_ENABLE_PREFIX_CACHE=1
export QSR_SERVER_CAPACITY=3 QSR_SERVER_NUM_SLOTS=3
export QSR_SERVER_BLOCK_SIZE=64
export QSR_SERVER_ENABLE_CUDAGRAPH=1
export QSR_SERVER_ENABLE_DFLASH=1
export QSR_SERVER_ENABLE_SESSION_AFFINITY=0
scripts/blackwellm_ctl.sh start
python -m benchmarks.prefix_cache_baseline --label post-7g-switch
scripts/blackwellm_ctl.sh stop
```

**Expected / judgment (do NOT gate on `hit_rate`)**: per
`notes/2026-08-02-prefix-cache-baseline.md` §1/§3.3, `hit_rate` saturates to
`1.00` at every turn ≥ 2 almost by construction of this workload and cannot
detect a coordinator that silently serves a shallower hit than it should.
**Gate on `tokens_saved_ratio = hit_L / ideal_L`, median across turns ≥ 2**:
the pre-switch baseline measured this at `1.0` (or the harmless
proxy-rounding variant `~1.02`) on all 40/40 hits across two independent
runs. Recommend: **fail if the post-switch median drops below ~0.95**
(leaves room for the same proxy-rounding noise the pre-switch baseline
already documented, without being a rubber stamp). Report `hit_rate`
alongside as a secondary sanity check only (expect it to stay ≈
`(turns-1)/turns` per trial, i.e. `5/6 ≈ 0.83` per trial, `~0.82` cumulative
including the one warmup miss -- matching the pre-switch run's
`prefix_cache_hit_rate=0.8163`).

Also compare the raw per-round table (`hit_L`, `ideal_L`, `wall_s`) directly
against `notes/2026-08-02-prefix-cache-baseline.md` §3.1's table -- a
coordinator bug that evicts more aggressively or mistracks which slot's
history matches would show up as a specific turn's `hit_L` dropping below
its pre-switch value, which the summary statistic alone could still average
over if it happened on only one or two turns out of 48.

## 6. Gate 4 -- C-LIVE smoke

```bash
cd /home/bot/project/qsr-w-7g
scripts/blackwellm_ctl.sh start   # if not already running from gate 3
make smoke SMOKE_BASE_URL=http://127.0.0.1:8100
# equivalently: python scripts/c_live_smoke.py --base-url http://127.0.0.1:8100
```

**Expected**: `67/67` passed, per the step 5 precedent
(`notes/2026-08-02-track-a-step5-gpu-verification.md` §5) -- confirm the
cold-start check specifically ran cold (`requests_completed_total == 0` via
a direct `/metrics` read taken *before* running the script, not assumed;
that check is only meaningful against a genuinely fresh server, and this
step's server was likely already warmed by gates 3/4 above if run in the
same session -- restart it if a genuinely cold C-LIVE run is wanted, or note
explicitly that the cold-start check was skipped/not-genuinely-cold if not).

## 7. `bf diff` before trusting any comparison

Per `AGENTS.md`: before treating gate 2's numbers as comparable to the
documented baseline, run `bf diff` (or the manual env-var checklist above,
which is what actually bit the step 5 session) against the baseline run's
recorded config/fingerprint, not just its summary numbers.

## 8. What this checklist does not cover (left for whoever runs it)

- **Concurrency.** All four gates above are single-client, sequential (same
  scope limitation the pre-switch baseline itself documents, §5 of that
  note). `find_best_slot_for_prompt`'s cache-aware slot assignment under
  *concurrent* admission is not exercised by any of the four gates as
  written.
- **A genuinely fresh cold-start C-LIVE run**, if gates 3/4 share one server
  process (§6's note).
- **Session affinity** (`QSR_SERVER_ENABLE_SESSION_AFFINITY=1`) is off in
  every command above, matching the pre-switch baseline's own scope --
  this step's diff touches the persistent content-hash path only, but the
  warm-slot session-affinity path also calls through `capabilities.
  prefix_cache`-gated code and was not separately re-measured here.
- **Real Qwen3.6/Track B checkpoints** are out of scope by construction:
  `needs_two_cache_families` is `False` for every checkpoint this runtime
  can currently load (`IMPLEMENTED_BACKENDS == {"laguna"}`), so the
  `SlotResourceManager` branch that actually merges two resources
  (`NotImplementedError` today) has no real-checkpoint GPU coverage and
  cannot have any until Track B lands a second backend.
