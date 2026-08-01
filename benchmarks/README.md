# benchmarks/

**Before adding a file here, read this.** This directory used to hold 136
one-off scripts with zero compounding value (see `AGENTS.md` and
`docs/diagnostics-guide.md`). 34 of them were removed in the T0-7 cleanup
(2026-08-01) — see "What was removed" below. **Do not repeat the pattern.**

> **New experiment? Submit it to the warm engine instead of adding a file
> here: `bf exec <script>` (see `docs/diagnostics-guide.md`).** The daemon
> keeps the model loaded, records a run record automatically, and the flight
> recorder (`QSR_TRACE=1`) gives you per-round history without a re-run.
> Only add a file to `benchmarks/` if it is a **reusable, reproducible gate**
> that will be run again — a regression check, a correctness gate, a frozen
> baseline comparison — and wire it into a note, test, or the Makefile so
> the next cleanup can tell it is still used.

## How these scripts are invoked

Two conventions coexist:

- **Cold-start module scripts** (`Usage: /home/bot/.venvs/vllm/bin/python -m
  benchmarks.<name>`) — load the model themselves, need the vLLM venv, need
  GPU. Use these for load-time config (block size, capacity, quantization
  backend) per the diagnostics guide's cold-vs-warm table.
- **`bf exec`-ready scripts** (marked `[bf-exec]` below) — do **not** load a
  model; `bf exec benchmarks/<name>.py` injects the already-loaded `engine` /
  `backend` / `tokenizer` objects from the warm daemon into the script's
  namespace (hence the `# ruff: noqa: F821` at the top of each). Prefer
  these for anything that isn't measuring cold start or load-time config.

Scripts with a real CLI (argparse) accept `--help`; check each file's
docstring for the exact invocation and any required flags.

## Correctness gates (run these after touching the execution path)

These are the load-bearing regression checks. If you change `runtime/` or
`server/`, at least skim this list for anything relevant.

| Script | Gates |
|---|---|
| `baseline_compare.py` | Frozen speed baseline (`fixtures/speed_baseline.json`) — attention microbench + 64K warm throughput. `--quick` for the fast half. |
| `verify_golden_fixtures.py` + `record_golden_fixtures.py` | Golden-fixture bit-exact parity: committed tokens, MTP accept sequence, GDN state norm, top-16 logits. `record_golden_fixtures.py` re-freezes the fixtures in `fixtures/golden/`; `verify_golden_fixtures.py` is the judge. Don't re-record casually — it moves the baseline everyone else is compared against. |
| `laguna_quality_gate.py` | Native vs oracle A/B comparison + prompt-id assertion guardrails. |
| `acceptance_regression.py` `[bf-exec]` | DFlash acceptance-rate regression suite; wired into `tests/test_acceptance_regression.py`. |
| `quality_regression.py` + `quality_compare.py` (+ `run_quality_ab.sh`, `quality_merge.py`) | Inference-quality non-regression: code/tool/agent/longctx dims against an OpenAI-compatible server. `run_quality_ab.sh {ours\|vllm\|compare}` orchestrates the full A/B; `quality_merge.py` is for when you ran dims separately and need to stitch reports back together. |
| `cudagraph_decode_regression.py`, `cudagraph_mtp_regression.py`, `cudagraph_eager_parity_check.py`, `verify_decode_cg_integration.py` | CUDA Graph capture/replay correctness vs eager, decode and MTP-verify paths. |
| `prefix_cache_*_check.py` (8 files) | Prefix-cache correctness gates, one per phase of the implementation (`P0` block-table, `P1` allocator, `P2` fan-out, `P3.1` persistent hit, `P3.2` eviction, `P3.3` cumulative, `P3.4` long-context perf). See `notes/prefix-cache-design.md` / `notes/prefix-cache-implementation-log.md` for what each phase means. |
| `prefix_cache_baseline.py` | Reproducible prefix-cache hit-depth/tokens-saved baseline against a running server (multi-trial growing-conversation workload, full per-round output, not just a summary). Frozen pre-A3 baseline + methodology: `notes/2026-08-02-prefix-cache-baseline.md`. Re-run identically post-7-g (`docs/implementation-plan.md` §6 step 7) for the A/B. |
| `mtp_*_check.py` (most of the `mtp_` family) | Correctness verification for MTP speculative decode: accept/reject boundary, GDN rollback, chunked/ragged prefill, cross-slot batching, CUDA-graph wiring. Each docstring names the specific regression/finding it locked in — see `notes/direct-model-runner-design.md` and `notes/2026-07-19-comprehensive-audit-and-forward-plan.md` for the narrative. |
| `verify_ring_kv_math.py`, `verify_ring_flashinfer_gpu.py` | SWA ring-buffer KV index math, CPU-only and GPU. |
| `single_prefill_regression.py`, `batch_decode_regression.py` | Fixed-input regression probes for the direct model runner's decode/prefill paths. |
| `server_e2e_check.py` | Real HTTP end-to-end validation of `server/app.py`. |
| `soak_test.py` | Long-running stability soak against a live server: slot wedges, leaks, request failures, metric drift. `--duration-minutes` / `--concurrency`. |
| `sparkinfer_moe_test.py`, `sparkinfer_standalone_test.py` | SparkInfer MoE correctness/perf: `_moe_test.py` hits the raw upstream kernel binding directly (bypasses our adapter) against a dequantized fp32 reference; `_standalone_test.py` goes through our own `runtime.backends.laguna_sparkinfer_moe` adapter. Both matter — SparkInfer is an externally-owned dependency (see `AGENTS.md`); if a correctness issue shows up, use these to tell whether it's upstream or in our adapter, then write it up for the SparkInfer team rather than patching in place. |

## Performance / profiling tools

| Script | What it measures |
|---|---|
| `decode_step_profile.py`, `native_decode_step_profile.py`, `native_nsys_profile.py` | Per-step kernel breakdown of the MTP decode path (ours vs native vLLM/FlashInfer), in-process profiler and `nsys`. |
| `a1a_gdn_profile.py` | Per-layer GDN occupancy ledger for one verify round. |
| `profile_dflash_round.py`, `profile_dflash_round_v2.py` `[bf-exec]`, `profile_dflash_verify_kernels.py`, `profile_verify_kernels.py` `[bf-exec]` | Stage- and kernel-level timing of a DFlash speculative-decode round. |
| `measure_decode_cg_throughput.py` | `decode_batch_sampled` eager vs CG-routed throughput, server-like single-slot calls. |
| `kernel_microbench_nvfp4kv.py`, `kernel_microbench_nvfp4kv_v2.py`, `kernel_microbench_split.py` | Decode-attention microbenches: NVFP4-KV vs FP8-KV, v2 tensor-core kernel, production split config. Feed `fixtures/speed_baseline.json`. |
| `flashinfer_decode_feasibility.py` | FlashInfer vs SM120-native GQA NATIVEFP8 decode kernel feasibility. |
| `ncu_splitkv_occupancy_probe.py` | `ncu` occupancy probe for split-KV decode. |
| `mem_backend_compare.py` | GPU memory: MARLIN vs CUTLASS MoE backends. |
| `laguna_moe_node_trace.py` | Single-load kernel ledger across the MoE node matrix. |
| `w1s_native_bench.py`, `native_warm_compare.py` | Native vLLM side of the W1-S acceptance/perf comparisons (paired with `mtp_w1s_our_runtime*.py` — see `workloads.py`). |
| `mtp_w1s_our_runtime.py`, `mtp_w1s_our_runtime_perf.py`, `mtp_our_runtime_acceptance.py` | This runtime's side of the same W1/W1-S comparisons. |
| `full_benchmark.py`, `full_comparison_ours.py`, `full_comparison_vllm.py`, `comprehensive_bench.py`, `laguna_vllm_dflash_baseline.py`, `laguna_backend_test.py`, `repro_80tok_m1_decode_cg.py`, `e2e_cg_bench.py`, `e2e_daemon_bench.py` `[bf-exec]` | Broader end-to-end throughput/ITL/acceptance sweeps at various context lengths, prefix-cache and CUDA-graph on/off. Several replicate a specific historical commit's methodology for a fair before/after — check the docstring before assuming two runs are comparable, and prefer `bf diff` over eyeballing. |
| `acceptance_sweep_quick.py` `[bf-exec]`, `quick_check.py` `[bf-exec]`, `diag_acceptance_v2.py` | Fast iteration acceptance checks (seconds, not a full model reload). |

## A2 — NVFP4 GEMM investigation (kernel/backend selection)

| Script | Role |
|---|---|
| `a2_gemm_shape_survey.py`, `a2_gemm_shape_profile.py` | Which (M, N, K) shapes actually occur at decode time, and their share of total GEMM time. |
| `a2_gemm_microbench.py`, `a2_native_baseline.py` | Kernel-level timing for those shapes; full native baseline. |
| `a2_autotune.py` | Per-shape tile-config autotuning; feeds `fixtures/a2_autotune_table.json`. |
| `a2_backend_sweep.py`, `a2_e2e_ab_test.py` | Sweep FlashInfer NVFP4 backends; stock vs custom-GEMM end-to-end A/B. |

See `notes/2026-07-22-a2-gemm-autotune-investigation.md` and
`notes/2026-07-23-a2-gemm-baseline-and-cudnn-findings.md` for what was
concluded.

## Diagnostics kept for provenance (evidence chain, not routine reruns)

These are one-off investigation scripts. They are kept — not because you
should run them again, but because a `notes/` file or another kept script's
comment cites them as where a specific finding was proven, and deleting them
would break that citation. Read the docstring before assuming a script like
this is a template; most hardcode a specific historical scenario.

`_diag_warm_suffix.py`, `mtp_signal_probe_isolated_repro.py`,
`mtp_slot_identity_pinpoint_diag.py`, `mtp_batch_divergence_diag.py`,
`mtp_batch_recompute_cost_diag.py`, `mtp_prefill_batch_memory_diag.py`,
`mtp_prefill_draft_logits_diag.py`, `mtp_trace_driven_probe.py`,
`d1_decode_round_kvlen_diag.py`, `d1_prefill_shape_nsys_diag.py`,
`phase0_nsys_gap_ledger_diag.py`, `memory_growth_diag.py`,
`batch_decode_signal_probe.py`, `capacity_w1w2_check.py`,
`cudagraph_decode_sanitizer_repro.py`, `cudagraph_sanitizer_micro.py`.

## `b1_d1_gpu_verify.py` — kept as a Track B starting point

Targets roadmap Track B's `B1` (Qwen3.6-27B correctness) milestone, which
hasn't started yet. Uses `oracle.qwen36_vllm.direct_model_runner` and an
external path (`/home/bot/project/sm120-flash-attention/vllm_integration`)
that lives outside this repo. Flagged during the T0-7 cleanup and resolved
by the user (2026-08-01): keep it as a head start for B1 rather than delete
and rebuild from scratch once Track A's backend abstraction lands. Expect
it to need rework against whatever `ModelSpec`/backend-protocol shape
Track A produces before it runs again.

## `official/`, `phase1/`, `fixtures/`

- `official/mmlu_pro_eval.py` — official MMLU-Pro eval harness; feeds
  `fixtures/speed_baseline.json`.
- `phase1/phase1_*.py` — bf16 ground-truth vs SparkInfer comparison and
  alpha/folding precision investigation; cited from
  `notes/2026-07-24-phase1-ground-truth.md`. Don't move these — `notes/`
  files reference them by path.
- `fixtures/` — frozen prompt sets, recorded baselines, and golden fixtures
  consumed by the scripts above. Treat anything under `fixtures/golden/` and
  `fixtures/speed_baseline.json` as a checked-in baseline, not scratch
  output: re-generating them moves what every other script is compared
  against.

## What was removed (2026-08-01, T0-7)

29 scripts with zero references anywhere in the repo (no `notes/`, no
`docs/`, no test, no Makefile target, no import, no fixture provenance
field) were deleted. Most matched the exact anti-patterns
`docs/diagnostics-guide.md` warns about: hardcoded absolute paths into the
*other* worktree, direct construction of `LagunaBackend`/`EngineArgs`
bypassing the server (the guide's "don't hand-roll a decode loop calling
private methods" anti-pattern — two of them called
`engine._forward_main_with_aux` directly), or a superseded duplicate of a
script that *is* kept (e.g. `quality_eval.py` duplicated the code-eval logic
`quality_regression.py` now does inline; `profile_decode_step.py` was an
early duplicate of the kept `decode_step_profile.py`;
`repro_80tok_m1_decode_cg_mainrepo.py` duplicated the kept
`repro_80tok_m1_decode_cg.py`). See the git log for the full list and
per-file reasoning.

A second pass the same day removed the whole vLLM-removal bit-exact
validation family — `_phase1_bitexact_validate.py` (+ `_long`),
`_phase3_dflash_bitexact_validate.py` (+ `_long`), and
`_phase5_e2e_bitexact_validate.py` — per explicit user decision (the T0-7
cleanup had originally flagged this family as "needs a human decision"
rather than deleting it outright, since `_phase5` was cited from
`notes/2026-07-27-vllm-complete-removal-implementation-plan.md`; the user
confirmed vLLM removal is complete and the family serves no further
purpose). That note now carries a superseded-pointer at the top saying so.
One dangling citation could not be fixed from here: `runtime/laguna_config.py`
around line 310 cites `benchmarks/_phase5_e2e_bitexact_validate.py` in a
docstring explaining `build_laguna_config`'s kwarg surface — `runtime/` is
outside this cleanup's file boundary (another agent owns it), so that
comment still names a file that no longer exists. The claim itself (which
call sites were grepped) is unaffected; only the path in the comment is
now stale. Whoever next touches `runtime/laguna_config.py` should drop or
reword that clause.

All 34 removed scripts are fully recoverable from git history
(`git log --diff-filter=D -- benchmarks/<name>.py`).
