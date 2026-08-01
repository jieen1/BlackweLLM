# SparkInfer upstream handoff: generalize the Laguna analytic-decode gating

> Written from BlackweLLM (`qwen-sm120-runtime`), for the SparkInfer team.
> BlackweLLM does not modify SparkInfer source — see
> `/home/bot/project/sparkinfer`'s owning team notice in this repo's
> `AGENTS.md` ("External dependencies you must not edit directly"). Everything
> below is a read-only audit of `/home/bot/project/sparkinfer` plus a
> concrete, safety-argued change proposal. Roadmap reference: `docs/roadmap.md`
> R5 / T0-5.

## 0. Headline finding — read this first

**As of `/home/bot/project/sparkinfer` HEAD `1621c1e` (2026-07-30, branch
`master`, clean working tree), the gating relaxation described below is NOT
present in the checkout.** All 13 call sites listed in §2 still require the
exact TP=2 shape (24 or 36 query heads, 4 KV heads, `page_size==128`).

This matters because `notes/2026-07-31-session-summary.md` in this repo
records a **local, in-place edit** to `sparkinfer/attention/paged/_forward.py`
and `planner.py` that relaxed exactly this gating, and reports throughput
gains measured against that edited checkout (4K: 353-401 tok/s, 64K: 353-368
tok/s, "Analytic Decode Path (no TURBO) — RECOMMENDED"). That edit was never
committed to `/home/bot/project/sparkinfer` — `git log`/`git status`/`git
reflog`/pickaxe search across every local branch found no trace of it — so it
is not just "not upstream," it is **currently absent from the local checkout
that produced those numbers**. Whoever benchmarks this stack next, without
first re-applying a patch, should expect a real regression from the
2026-07-31 figures, not a reproduction of them.

`runtime/preflight.py`'s `check_sparkinfer_analytic_decode_gate` (this repo)
detects this condition functionally (calls the live gate with production
shape parameters) and reports it as a loud, non-fatal startup warning — see
that module's docstring for how it works and where it is meant to be wired
in. Right now, on this checkout, that check fails (warns).

## 1. Background: why the fast path never fires in production

BlackweLLM runs Qwen/Laguna-family models with `TP=1` (no tensor
parallelism — see `AGENTS.md`, "one GPU, one process"). The real, unsharded
attention shapes are (verified against checkpoint tensors directly, not
`config.json`; see `runtime/backends/laguna_sparkinfer_attn.py`'s module
docstring):

| Layer group | `num_q_heads` | `num_kv_heads` | `gqa_group_size` | `window_left` |
|---|---|---|---|---|
| Full attention | 48 | 8 | 6 | -1 |
| SWA | 72 | 8 | 9 | 511 |

`head_dim_qk == head_dim_vo == 128` for both. Current production
`page_size` (this repo's `block_size`) is **64**; a separate, already-verified
migration to `page_size=128` is tracked independently in this repo (see
`notes/2026-07-27-laguna-real-shapes-correction-and-page-size-migration-plan.md`
— correctness at `page_size=128` on these real shapes is confirmed,
cos ≥ 0.999991 across four scenarios).

SparkInfer ships five hand-tuned "Laguna family" kernel specializations
(warp-specialized producer/consumer split, device-side analytic split
mapping, compact GQA-scoped sync) that give large attention-bandwidth gains
over the generic paged-attention fallback — SparkInfer's own published
numbers for the `num_kv_heads=4` shape are 2.1x/1.4x, and this repo's roofline
work independently found production attention running at only ~37% of the
measured bandwidth ceiling while MoE and dense GEMM are at 86-100% (see
`notes/2026-07-31-sm120-flash-attention-kernel-research-for-sparkinfer.md`
§5#1). But every one of those specializations gates on the **exact TP=2
shape** — `num_q_heads in (24, 36)`, `num_kv_heads == 4`, `page_size == 128`
— which is the shape you get by sharding Laguna's real heads across 2 tensor-
parallel ranks. BlackweLLM runs `TP=1`, so none of the nine absolute-head-count
checks ever match, and every call falls through to SparkInfer's generic
kernel. `gqa_group_size` (6 for full attention, 9 for SWA) is **identical**
between the TP=1 and TP=2 shapes — it is a property of the model's GQA ratio,
invariant to how you shard KV heads — so the gates are checking a proxy
(absolute head count) for a quantity (warp/CTA-internal work shape) that is
actually determined by `gqa_group_size`.

## 2. Exact locations (verified against HEAD `1621c1e`)

All paths relative to `/home/bot/project/sparkinfer/`.

| # | File | Lines | Function | Family | Current gate (abbreviated) |
|---|---|---|---|---|---|
| 1 | `sparkinfer/attention/paged/traits.py` | 377–404 | `select_paged_forward_traits_from_plan`, branch 1 | SWA extend | `mode=="extend"`, `window_left==511`, `page_size==128`, `num_q_heads==36`, `num_kv_heads==4`, `gqa_group_size==9` |
| 2 | `sparkinfer/attention/paged/traits.py` | 405–432 | same, branch 2 | full-attn extend | `mode=="extend"`, `window_left<0`, `page_size==128`, `num_q_heads==24`, `num_kv_heads==4`, `gqa_group_size==6` |
| 3 | `sparkinfer/attention/paged/traits.py` | 433–453 | same, branch 3 | decode (split_kv) | `mode=="decode"`, `window_left<0`, `page_size==128`, `num_q_heads==24`, `num_kv_heads==4`, `gqa_group_size==6` |
| 4 | `sparkinfer/attention/paged/traits.py` | 454–471 | same, branch 4 | SWA decode | `mode=="decode"`, `page_size==128`, `num_q_heads==36`, `num_kv_heads==4`, `gqa_group_size==9` |
| 5 | `sparkinfer/attention/paged/traits.py` | 472–494 | same, branch 5 | verify (split_kv) | `mode=="verify"`, `window_left<0`, `page_size==128`, `num_q_heads==24`, `num_kv_heads==4`, `gqa_group_size==6` |
| 6 | `sparkinfer/attention/paged/_forward.py` | 427–461 | `_use_laguna_verify_forward_kernel` | verify kernel dispatch | same head/page conditions as #5 |
| 7 | `sparkinfer/attention/paged/_forward.py` | 464–501 | `_use_laguna_decode_analytic_kernel` | decode kernel dispatch | same head/page conditions as #3 |
| 8 | `sparkinfer/attention/paged/_forward.py` | 1478–1493 | `paged_attention_forward`, local `pair_bf16_merge_partial_loads` | verify-mode merge-kernel pairing | same head/page conditions as #5 |
| 9 | `sparkinfer/attention/paged/planner.py` | 219–256 | `_laguna_page128_one_wave_chunk_budget` | SWA one-wave chunk budget | `num_q_heads==36`, `num_kv_heads==4`, `page_size==128`, `batch==1` |
| 10 | `sparkinfer/attention/paged/planner.py` | 259–285 | `_is_laguna_fp8_gqa6_analytic_decode_graph` | decode graph identification | `num_q_heads==24`, `num_kv_heads==4`, `page_size==128` |
| 11 | `sparkinfer/attention/paged/planner.py` | 288–316 | `_is_laguna_fp8_gqa6_analytic_verify_graph` | verify graph identification | `num_q_heads==24`, `num_kv_heads==4`, `page_size==128`, `query_len==8` |
| 12 | `sparkinfer/attention/paged/planner.py` | 319–347 | `_is_laguna_fp8_gqa6_full_prefill_graph` | prefill graph identification | `num_q_heads==24`, `num_kv_heads==4`, `page_size==128` |
| 13 | `sparkinfer/attention/paged/workspace.py` | 1025–1047 | `_uses_laguna_verify_analytic_schedule` | CUDA-graph replay path selection | `num_q_heads==24`, `num_kv_heads==4`, `gqa_group_size==6`, `page_size==128` |

All 13 currently reject BlackweLLM's production shape for **two independent
reasons**: `num_kv_heads` (8, not 4) and `page_size` (64, not 128 — pending
this repo's own separate migration). Both need addressing for the fast path
to activate; §4 covers the split of responsibility.

## 3. Why relaxing the gate is safe (and where it is *not* free)

**Safe part — CTA-internal tuning is already parameterized by
`gqa_group_size`, not by absolute head count.** Reading
`select_paged_forward_traits_from_plan` (locations #1-5) directly:

- `compact_sync_rows = plan.gqa_group_size` (traits.py, decode branch) — this
  constant is *already* gqa-ratio-generic; it does not need to change at all.
- `exact_num_mma_kv` and `minimum_shared_storage_bytes` are computed from
  `cta_tile_q`, `head_dim_qk`, `head_dim_vo`, and a fixed KV-tile width (32) —
  none of these are functions of `num_kv_heads`. They describe **one CTA's**
  internal register/shared-memory budget for processing one KV-head group's
  worth of query rows (`gqa_group_size` of them), which is identical whether
  that KV head is one of 4 (TP=2) or one of 8 (TP=1).
- `_paged_determine_cta_tile_q` (referenced from
  `notes/2026-07-27-sparkinfer-generalize-kv-heads-4-to-8-spec.md`, already
  written by this repo) computes `packed_qo_len = qo_len * gqa_group_size`
  — also already TP-invariant.

This is the load-bearing argument: **the kernel bodies are shape-generic
already; only the nine absolute-head-count gate predicates need to change.**
No kernel/CUDA source needs touching for the change proposed in §4.

**Not free — grid-level occupancy tuning assumes `num_kv_heads==4`.**
`num_kv_heads` determines how many *independent* CTA groups the decode graph
must schedule across the device's 188 SMs — going from 4 to 8 **doubles**
that count. The whole-grid budget functions (`_decode_graph_heuristic_max_chunks_per_req`,
`heuristic_decode_graph_chunk_pages`, `_laguna_page128_one_wave_chunk_budget`,
all in `planner.py`) hardcode chunk budgets (`total_chunk_budget = 96`, "94
total work items on 188-SM Laguna", etc.) derived empirically for the 4-KV-head
case. These numbers are SM-occupancy-derived, not shape-derived, and **must
be re-measured, not assumed, for `num_kv_heads=8`.** This is exactly the risk
this repo's earlier spec note (`notes/2026-07-27-sparkinfer-generalize-kv-heads-4-to-8-spec.md`)
flagged: "改一行就完事" is the wrong mental model here.

## 4. Recommended change, split by who does what

**Upstream (SparkInfer team, this handoff):** at the 13 locations in §2,
replace the absolute-head-count conjuncts
(`num_q_heads == 24/36 and num_kv_heads == 4`) with a check on
`gqa_group_size` alone (`gqa_group_size in (6, 9)`, or more precisely
`gqa_group_size == 6` for the full-attention family and `gqa_group_size == 9`
for the SWA family — do not drop that distinction, the two families use
different `window_left`/`cta_tile_q` values). Also accept `page_size in (64,
128)` at each of these sites rather than requiring exactly 128 (see §5 for why
this half of the change is independent and lower-risk). Then, **separately**:
re-derive the grid-occupancy constants in `planner.py`'s three chunk-budget
functions for the `num_kv_heads == 8` case — do not reuse the `num_kv_heads
== 4` constants unchanged. The safest sequencing is: relax the CTA-selection
gates first (traits.py, _forward.py, workspace.py — locations #1-8, #13),
correctness-test that on real `num_kv_heads=8` inputs, *then* tackle the
occupancy-budget functions (#9-12) as a distinct follow-up with their own
benchmark pass, since a wrong occupancy constant degrades performance without
being a correctness bug and is easy to miss in a quick regression check.

**Downstream (this repo, already scoped, not part of this ask):** the
`page_size` 64→128 migration is tracked independently
(`notes/2026-07-27-laguna-real-shapes-correction-and-page-size-migration-plan.md`);
it does not require any SparkInfer change and its correctness is already
verified. It is listed here only because both halves are required
simultaneously for the fast path to activate on production traffic — relaxing
`num_kv_heads` alone, without either this repo's `page_size=128` migration or
the `page_size` relaxation suggested above, still leaves every gate false on
production's current `page_size=64`.

## 5. Correctness verification method (repeatable, no BlackweLLM engine required)

This repo already has the methodology, validated during the block_size-128
investigation (`notes/2026-07-27-block-size-128-accept-rate-root-cause-CLOSED.md`):
kernel isolation testing against a reference implementation, not full
end-to-end engine runs. Recommended verification sequence for whoever lands
the generalization:

1. **Isolated numerical correctness.** Construct paged-attention inputs with
   `num_kv_heads=8`, `num_q_heads=48` (`gqa_group_size=6`) and `num_q_heads=72`
   (`gqa_group_size=9`), FP8 KV cache, bf16 activations, at both `page_size=64`
   and `page_size=128`, across a range of KV lengths spanning the chunk-budget
   boundaries in §3 (the `224 <= max_effective_kv_pages <= 288` / `480..544` /
   `992..1056` bands visible in `planner.py`'s existing GB10 budget function
   are a good template for the bucket boundaries to test). Compare against
   SparkInfer's own generic (ungated) kernel or a plain PyTorch reference,
   same methodology as `isolate_kernel_test_v2.py` referenced in
   `notes/2026-07-27-sparkinfer-generalize-kv-heads-4-to-8-spec.md`. Require
   cosine similarity parity with the existing `num_kv_heads=4` specialized-vs-
   generic comparison (this repo's own bar for page_size=128 was
   cos ≥ 0.999991).
2. **CUDA Graph capture/replay stability.** Specifically exercise capture at
   one batch/KV-length combination and replay at several others within the
   same graph's declared capacity — this repo's history includes a real
   incident where a KV-layout change silently produced stale CUDA-graph
   addresses and crashed acceptance to 0.13%
   (`notes/2026-07-27-block-size-128-migration-and-tie-break-noise.md`); any
   change touching the analytic decode graph path deserves the same scrutiny.
3. **Bandwidth/occupancy re-measurement**, not extrapolation. Re-run this
   repo's bandwidth-roofline method (`notes/2026-07-31-sm120-flash-attention-kernel-research-for-sparkinfer.md`
   §5#1 references the exact roofline approach) on the `num_kv_heads=8` shape
   after landing the change. Do not assume the published `2.1x/1.4x` figures
   for `num_kv_heads=4` transfer unchanged — the explicit expectation-
   management note in that section is "page_size=128 alone does NOT grant the
   specializations... re-measure after; do not extrapolate."
4. **Regression guard once patched.** `runtime/preflight.py` in this repo
   (`check_sparkinfer_analytic_decode_gate`, calling
   `_is_laguna_fp8_gqa6_analytic_decode_graph` directly with the production
   48/8 shape) will flip from warning to passing once a patched SparkInfer is
   installed — this is a convenient one-line functional smoke test
   (`python -c "from runtime.preflight import probe_sparkinfer; print(probe_sparkinfer())"`
   on a machine with both packages installed) that does not require standing
   up the full engine.

## 6. Summary for reviewers

- **What changed conceptually:** nothing in any `.cu`/kernel-launch source.
  Nine boolean gate predicates, across 4 Python files, that decide *which*
  already-compiled kernel variant runs for a given attention call.
- **Why it's safe to relax the CTA-selection half:** the CTA-internal
  resource formulas these gates guard are already parameterized by
  `gqa_group_size`, a TP-invariant quantity — not by the absolute head counts
  the gates currently check.
- **Why it's not "just flip a flag":** the grid-level occupancy budget
  functions in `planner.py` *do* hardcode `num_kv_heads=4`-specific SM
  occupancy constants and need independent re-derivation and re-benchmarking
  for `num_kv_heads=8`, per §3-4.
- **What this repo needs from SparkInfer:** the generalization in §4,
  verified per §5. This repo will not modify SparkInfer source directly.
