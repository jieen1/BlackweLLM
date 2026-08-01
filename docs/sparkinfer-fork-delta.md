# SparkInfer fork delta: what our fork carries beyond upstream

> Roadmap reference: `docs/roadmap.md` R5 / T0-5.

## 0. Two remotes, two rules

`/home/bot/project/sparkinfer` has two remotes, and they are not the same
thing:

- **`origin`** = `https://github.com/jieen1/sparkinfer.git` — **this
  project's own fork.** Editable, ours to change, and — as of this writing —
  the fork **is** production: `pip install -e /home/bot/project/sparkinfer`
  resolves to a checkout of this fork, `origin/master` is checked out by
  default, and BlackweLLM's SparkInfer dependency runs whatever is on that
  branch with no environment variable needed to opt in.
- **`upstream`** = `https://github.com/local-inference-lab/sparkinfer.git` —
  the other team's canonical repo. `AGENTS.md`'s "do not modify SparkInfer
  source directly" rule is about *this* remote, not the fork. This repo does
  not push to `upstream`, has no write access there, and any change destined
  for it goes through a PR, not a direct commit.

**Verified pin:** sparkinfer commit **`0844a4f`** on fork `origin/master`
(based on upstream `3bd3a2e`, the last upstream sync merged into the fork).
This is the exact commit this document, and `runtime/preflight.py`'s
functional gate probe, were verified against — record it alongside the rest
of this project's dependency contract (`pyproject.toml`'s SparkInfer note)
when that pin changes.

## 1. What our fork carries beyond upstream

`origin/master` = upstream `3bd3a2e` ("Merge branch
'local-inference-lab:master' into master") + two fork-only commits, both
2026-07-31, both gating-only (no kernel/CUDA source touched):

### `7a1d69d` — unlock the Laguna analytic decode kernel for TP=1

`sparkinfer/attention/paged/_forward.py` (-6/+2) and
`sparkinfer/attention/paged/planner.py` (-6/+4). In 4 specific gate
functions (see §2, locations #6/#7/#10/#12), replaces
`num_q_heads == 24 and num_kv_heads == 4` with a `gqa_group_size`-only
check, and widens `page_size == 128` to `page_size in (64, 128)`:

```python
# _forward.py: _use_laguna_verify_forward_kernel / _use_laguna_decode_analytic_kernel
-        and plan.page_size == 128
+        and plan.page_size in (64, 128)
         and plan.dtype == torch.bfloat16
         and plan.kv_dtype == torch.float8_e4m3fn
-        and plan.num_q_heads == 24
-        and plan.num_kv_heads == 4
         and plan.gqa_group_size == 6

# planner.py: _is_laguna_fp8_gqa6_analytic_decode_graph / _is_laguna_fp8_gqa6_full_prefill_graph
-        and int(num_q_heads) == 24
-        and int(num_kv_heads) == 4
+        and int(num_q_heads) // max(int(num_kv_heads), 1) == 6
         and int(head_dim_qk) == 128
         and int(head_dim_vo) == 128
-        and int(page_size) == 128
+        and int(page_size) in (64, 128)
```

**Why this is safe:** the kernel bodies these gates dispatch to are already
parameterized by `gqa_group_size` (a property of the model's GQA ratio,
identical whether TP=1 shards KV heads as 8 or TP=2 shards them as 4), not
by the absolute head count. Reading the CTA-internal resource formulas in
`select_paged_forward_traits_from_plan` (traits.py, §2 locations #1-5,
untouched by this commit but illustrative of the same kernels' internals):

- `compact_sync_rows = plan.gqa_group_size` — already gqa-ratio-generic.
- `exact_num_mma_kv` and `minimum_shared_storage_bytes` are computed from
  `cta_tile_q`, `head_dim_qk`, `head_dim_vo`, and a fixed KV-tile width (32)
  — none of these are functions of `num_kv_heads`. They describe **one
  CTA's** internal register/shared-memory budget for processing one KV-head
  group's worth of query rows (`gqa_group_size` of them), identical whether
  that KV head is one of 4 (TP=2) or one of 8 (TP=1).
- `_paged_determine_cta_tile_q` computes `packed_qo_len = qo_len *
  gqa_group_size` — also already TP-invariant.

The absolute head counts existed only in the four *admission* predicates
this commit touched — not in anything that actually shapes the kernel
launch. The commit's own trailer backs this with direct evidence: SMEM
36KB ≤ 50KB budget, register budgets, the LSE fill literal `Int32(6)`
matching `gqa=6`, a TMA `page_size=64` branch existing, and the row-0 merge
fastpath supporting `group_size in {6, 8}` — all checked before landing.
`Tested: Full acceptance regression suite (13 workloads, 3 measurements
each)`. Its own `Not-tested`: `batch>2 analytic scope, ncu occupancy
validation at kv_heads=8` — flagged honestly as open, see §3.

It also deliberately did **not** touch verify-graph identification (#11),
any of the five CTA-trait-selection branches (#1-5), the inline
merge-pairing predicate (#8), the SWA one-wave chunk budget (#9), or the
CUDA-graph replay path selection (#13) — its own commit message: "SWA layers
(72/8, gqa=9) remain on the generic path by construction... no gqa=9
register fastpath", and self-assessed `Scope-risk: moderate`. That was a
deliberate scope decision, not an oversight — see §3 for what that leaves
open.

### `0844a4f` — decouple FP8 PV from FP8 QK for the verify path

`sparkinfer/attention/paged/_forward.py` (-2/+7), inside
`_resolve_native_fp8_attention_mma_flags`. Unrelated to the head-count
gating above — a distinct FP8-precision change:

```python
     use_native_fp8_pv = (
-        use_native_fp8_qk
-        and plan.mode in ("decode", "verify")
+        plan.kv_dtype == torch.float8_e4m3fn
+        and plan.mode == "verify"
         and plan.kv_chunk_size <= 384
+        and plan.cta_tile_q == 64
     )
```

Enables FP8 PV (probability × value) MMA independently of FP8 QK ("TURBO"),
restricted to the verify path only. Rationale from the commit: PV operands
are bounded in `[0, 1]` (well within FP8 e4m3 range) and PV errors average
out under accumulation, so FP8 PV is numerically safer than FP8 QK; the
`plan.mode == "verify"` restriction is load-bearing, not incidental — FP8 PV
MMA requires `num_warps_kv == 1`, true only for the verify path
(`cta_tile_q == 64`); the decode path uses `num_warps_kv == 4` and a prior
attempt enabling FP8 PV there caused a CUDA-graph capture hang (see the
commit's own `Rejected:` line). Self-assessed `Scope-risk: narrow`,
`Confidence: medium`, `Tested: quick_bench 1-round, quality preserved`,
`Not-tested: 3-round statistical validation, interaction with TURBO`.

## 2. Exact gate inventory (verified against fork HEAD `0844a4f`)

All paths relative to `/home/bot/project/sparkinfer/`. Re-verified directly
against the current checkout (read-only `git show`/`grep`/`sed`, no source
changed) — line numbers below supersede any earlier revision of this
document, which was checked against a stale pre-sync baseline (`1621c1e`).
Three intervening upstream commits (`b38a60e`, `6a2babc`, `9b852b2` — SM121
paged-attention decode/capacity work, all outside the Laguna/SM120 gates
below) shifted line numbers in `planner.py` by inserting a new
`_sm121_gqa8_decode_chunk_budget` function between locations #9 and #10;
`traits.py` and `workspace.py` were untouched and kept identical line
numbers to before.

| # | File | Lines | Function | Family | Status |
|---|---|---|---|---|---|
| 1 | `sparkinfer/attention/paged/traits.py` | 377–404 | `select_paged_forward_traits_from_plan`, branch 1 | SWA extend | untouched (strict: `num_q_heads==36`, `num_kv_heads==4`, `page_size==128`) |
| 2 | `sparkinfer/attention/paged/traits.py` | 405–432 | same, branch 2 | full-attn extend | untouched (strict: `num_q_heads==24`, `num_kv_heads==4`, `page_size==128`) |
| 3 | `sparkinfer/attention/paged/traits.py` | 433–453 | same, branch 3 | decode (split_kv) | untouched (strict) |
| 4 | `sparkinfer/attention/paged/traits.py` | 454–471 | same, branch 4 | SWA decode | untouched (strict) |
| 5 | `sparkinfer/attention/paged/traits.py` | 472–494 | same, branch 5 | verify (split_kv) | untouched (strict) |
| 6 | `sparkinfer/attention/paged/_forward.py` | 432–464 | `_use_laguna_verify_forward_kernel` | verify kernel dispatch | **OPEN** — `gqa_group_size==6`, `page_size in (64,128)` |
| 7 | `sparkinfer/attention/paged/_forward.py` | 467–502 | `_use_laguna_decode_analytic_kernel` | decode kernel dispatch | **OPEN** — same relaxed form |
| 8 | `sparkinfer/attention/paged/_forward.py` | 1479–1499 | `paged_attention_forward`, local `pair_bf16_merge_partial_loads` | verify-mode merge-kernel pairing | untouched (strict, in the `mode=="verify"` branch of an `or`) |
| 9 | `sparkinfer/attention/paged/planner.py` | 219–256 | `_laguna_page128_one_wave_chunk_budget` | SWA one-wave chunk budget | untouched (strict) |
| 10 | `sparkinfer/attention/paged/planner.py` | 307–332 | `_is_laguna_fp8_gqa6_analytic_decode_graph` | decode graph identification | **OPEN** — `num_q_heads // num_kv_heads == 6`, `page_size in (64,128)` |
| 11 | `sparkinfer/attention/paged/planner.py` | 335–363 | `_is_laguna_fp8_gqa6_analytic_verify_graph` | verify graph identification | untouched (strict) |
| 12 | `sparkinfer/attention/paged/planner.py` | 366–391 | `_is_laguna_fp8_gqa6_full_prefill_graph` | prefill graph identification | **OPEN** — same relaxed form as #10 |
| 13 | `sparkinfer/attention/paged/workspace.py` | 1025–1047 | `_uses_laguna_verify_analytic_schedule` | CUDA-graph replay path selection | untouched (strict) |

Functionally confirmed on this checkout: calling `_is_laguna_fp8_gqa6_analytic_decode_graph`
directly with BlackweLLM's real production shape (`num_q_heads=48`,
`num_kv_heads=8`, `page_size=64`, FP8 KV, bf16, `window_left=-1`) returns
`True`. This is exactly what `runtime/preflight.py`'s
`check_sparkinfer_analytic_decode_gate` checks at every startup.

## 3. What's not relaxed yet — future optimization opportunities

Locations #1, #2, #3, #4, #5, #8, #9, #11, #13 — 9 of the 13 — still require
the exact upstream TP=2 shape and reject BlackweLLM's production shape. This
is not an oversight in the fork; `7a1d69d`'s commit message explicitly scoped
around it (`Scope-risk: moderate`, SWA gating rejected on purpose). It is,
however, real headroom left on the table:

- **#9, #11 (planner.py graph identification), #13 (workspace.py CUDA-graph
  replay), #8 (_forward.py merge pairing)** would follow the same pattern as
  the two already-relaxed functions — drop the absolute head-count conjuncts,
  keep `gqa_group_size`, widen `page_size`. Mechanically similar to
  `7a1d69d`, same review bar.
- **#1-5 (`traits.py`'s CTA-trait-selection branches)** are a different kind
  of change: these don't just gate *whether* the specialized kernel runs,
  they select `exact_num_mma_kv`, `minimum_shared_storage_bytes`, and
  `compact_sync_rows` for it — the CTA-internal resource formulas argued
  safe in §1. Relaxing these would let the *SWA* family (gqa=9) and the
  verify/extend branches also reach the specialized traits, not just decode
  and full-prefill.

**The one risk that applies to all of the above, not repeated per-item:**
`num_kv_heads` determines how many *independent* CTA groups the decode graph
must schedule across the device's 188 SMs — going from 4 (upstream TP=2) to
8 (our TP=1) **doubles** that count. The whole-grid budget functions
(`_decode_graph_heuristic_max_chunks_per_req`, `heuristic_decode_graph_chunk_pages`,
`_laguna_page128_one_wave_chunk_budget` — all in `planner.py`) hardcode chunk
budgets (`total_chunk_budget = 96`, "94 total work items on 188-SM Laguna",
etc.) derived empirically for the 4-KV-head case. These numbers are
SM-occupancy-derived, not shape-derived, and **must be re-measured, not
assumed, for `num_kv_heads=8`** before extending the relaxation to anything
that touches grid-level scheduling (which #1-5 and #9 do; #8/#11/#13 are
CTA-selection/replay-path predicates and lower-risk by comparison). This is
exactly the risk `7a1d69d`'s own `Not-tested` line flags (`ncu occupancy
validation at kv_heads=8`) and exactly what an earlier investigation in this
repo (`notes/2026-07-27-sparkinfer-generalize-kv-heads-4-to-8-spec.md`)
warned about: "改一行就完事" is the wrong mental model for the occupancy
functions specifically, even though it turned out to be roughly right for
the 4 gate predicates `7a1d69d` did relax.

## 4. Upstreaming to `local-inference-lab/sparkinfer` — optional future goal, not blocking

Contributing `7a1d69d`'s generalization (and, separately, whichever of the 9
remaining locations get relaxed) back to `upstream` remains worth doing —
other consumers of that repo presumably hit the same TP=1-shape miss we did
— but it is **not required for BlackweLLM**. Our fork is self-consistent:
`origin/master` is production, the gate is open for our shape today, and
nothing here depends on upstream's cooperation or timeline.

If/when someone does prepare an upstream PR, the safety argument in §1 and
the verification method below are the starting material:

**Verification method** (repeatable, no BlackweLLM engine required — the
same kernel-isolation methodology validated during this repo's block_size-128
investigation, `notes/2026-07-27-block-size-128-accept-rate-root-cause-CLOSED.md`):

1. **Isolated numerical correctness.** Construct paged-attention inputs with
   `num_kv_heads=8`, `num_q_heads=48` (`gqa_group_size=6`) and `num_q_heads=72`
   (`gqa_group_size=9`), FP8 KV cache, bf16 activations, at both `page_size=64`
   and `page_size=128`, across a range of KV lengths spanning the chunk-budget
   boundaries noted in §3 (the `224 <= max_effective_kv_pages <= 288` /
   `480..544` / `992..1056` bands visible in `planner.py`'s
   `_sm121_gqa8_decode_chunk_budget` are a good template for bucket
   boundaries to test, by analogy). Compare against SparkInfer's own generic
   (ungated) kernel or a plain PyTorch reference, same methodology as
   `isolate_kernel_test_v2.py` referenced in
   `notes/2026-07-27-sparkinfer-generalize-kv-heads-4-to-8-spec.md`. Require
   cosine similarity parity with the existing `num_kv_heads=4`
   specialized-vs-generic comparison (this repo's own bar for page_size=128
   was cos ≥ 0.999991).
2. **CUDA Graph capture/replay stability.** Specifically exercise capture at
   one batch/KV-length combination and replay at several others within the
   same graph's declared capacity — this repo's history includes a real
   incident where a KV-layout change silently produced stale CUDA-graph
   addresses and crashed acceptance to 0.13%
   (`notes/2026-07-27-block-size-128-migration-and-tie-break-noise.md`); any
   change touching the analytic decode graph path deserves the same
   scrutiny.
3. **Bandwidth/occupancy re-measurement**, not extrapolation — re-run this
   repo's bandwidth-roofline method
   (`notes/2026-07-31-sm120-flash-attention-kernel-research-for-sparkinfer.md`
   §5#1) on the `num_kv_heads=8` shape for whichever location is being
   relaxed next. Do not assume the published `2.1x/1.4x` figures for
   `num_kv_heads=4` transfer unchanged; §3's occupancy-budget risk applies
   directly here.
4. **Regression guard.** `runtime/preflight.py`'s
   `check_sparkinfer_analytic_decode_gate` (calling
   `_is_laguna_fp8_gqa6_analytic_decode_graph` directly with the production
   48/8 shape) is a one-line functional smoke test
   (`python -c "from runtime.preflight import probe_sparkinfer; print(probe_sparkinfer())"`)
   that already exercises exactly this — useful for confirming any future
   `upstream` sync or fork rebase didn't regress the open gates back closed.

## 5. Summary

- **Where we stand:** `origin/master` (fork HEAD `0844a4f`) is production
  and is what BlackweLLM's editable SparkInfer install runs by default, no
  flags needed. 4 of 13 identified gate locations are open for our TP=1
  shape (48/8 heads); the other 9 remain on upstream's strict TP=2-only
  gating, by deliberate scope choice in `7a1d69d`, not by accident.
- **Why the 4 that are open are safe:** the kernel bodies they dispatch to
  are already parameterized by `gqa_group_size`, a TP-invariant quantity —
  the deleted conjuncts were checking a proxy (absolute head count) for a
  quantity the kernel never actually used that way.
- **Why the other 9 aren't "just the same fix nine more times":** #1-5 also
  select CTA-internal resource constants, not just admission; and anything
  touching grid-level scheduling (#1-5, #9) inherits the occupancy-budget
  risk in §3 — those constants are `num_kv_heads=4`-derived and unverified
  at `num_kv_heads=8`.
- **Upstreaming:** a nice-to-have, not a dependency. This repo does not
  write to `upstream` directly and isn't blocked on it doing so.
