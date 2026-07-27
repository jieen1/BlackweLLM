# Repository Guidelines

## Project Structure & Module Organization

Actual structure (post-B5 modularization, 2026-07-22):

- `runtime/`: Core inference engine (B5 模块化后拆分为多个域):
  - `direct_model_runner.py`: Qwen3.6 main runner class (MTP/prefill/decode)
  - `block_pool.py`: Paging/prefix-cache infrastructure (Block, BlockPool, hash)
  - `metadata_builders.py`: Attention/GDN metadata construction
  - `cuda_graphs.py`: CUDA Graph capture/replay (CapturedBatchDecodeGraph, CapturedMTPDraftStepGraph)
  - `mtp_accept.py`: MTP accept/reject logic (pure functions)
  - `compat_vllm.py`: B7-V1 single-point vLLM dependency consolidation
  - `sampling.py`: Temperature/top-k/top-p sampling primitives
  - `backends/laguna.py`: Laguna-S-2.1 backend (sparkinfer MoE kernel, SWA ring KV, DFlash)
- `server/`: OpenAI + Anthropic dual-protocol API (streaming, tools, thinking)
  - `app.py`: FastAPI application
  - `engine.py`: Continuous-batching server engine
  - `formats/`: Protocol adapters (openai, anthropic, stream, tools, thinking)
- `benchmarks/`: Reproducible performance measurements + fixtures
  - `fixtures/`: speed_baseline.json, golden/, laguna_vllm_baseline.json
- `tests/`: CPU-only unit tests, no model weights required
- `notes/`: Design documents and investigation records
- `docs/`: roadmap.md, architecture.md

Keep components small and layer boundaries explicit. Do not embed
backend-specific calls throughout model code; route all vLLM imports through
`runtime/compat_vllm.py`.

## Build, Test, and Development Commands

Install the lightweight development tools, then run these repository-root
commands:

```bash
python -m pip install -e '.[dev]'        # development dependencies
python -m pip install -e '.[cuda]'       # PyTorch CUDA runtime
python -m pytest -q                      # correctness suite
python -m pytest tests/test_gdn.py -q   # focused regression
python -m benchmarks.workloads            # print frozen W1/W2 contracts
```

Keep environment setup, CUDA/toolchain versions, and model-location variables
in `README.md` or dedicated developer documentation. Do not make unit tests require
downloading model weights unless explicitly marked as integration tests.

## Diagnostics — read this before debugging anything

This machine has **one GPU and no parallelism**; a test run costs minutes. The
only lever on iteration speed is **how much each GPU run tells you**. A
diagnostics platform (`bfdiag`, CLI entry point `bf`) exists for exactly this.
**Full guide: `docs/diagnostics-guide.md` — read it before writing any
diagnostic code.**

Three rules that override habit:

1. **Do not write another one-off script under `benchmarks/`.** There are
   already 144 of them (32710 lines) with zero compounding value. Submit
   experiments to the warm engine instead: `bf exec <script>`.
2. **Run `bf diff <A> <B>` before comparing any two numbers.** On 2026-07-27
   two acceptance rates (1.000 vs 0.687) were compared as evidence the backend
   had caught up with vLLM; the two runs had used different prompts, and a full
   day was lost. `bf diff` prints the config delta and refuses to call
   incomparable runs comparable.
3. **When something fails, read the existing trace first — do not re-run.** The
   flight recorder is on-by-default-cheap; the failing run's per-round history
   is already on disk. Re-running costs minutes and may not reproduce.

Symptom → tool:

| Symptom | Sequence |
|---|---|
| Acceptance rate dropped | `bf diff` → `bf trace show` (**reject_position histogram** — its *shape* separates several distinct bug classes) → `bf divergence` |
| Garbage output / NaN | `QSR_ASSERT_LEVEL=1` re-run → `bf trace show` → `bf divergence` |
| Suddenly slower | `bf diff` → `bf trace show` (CG hit rate, eager-fallback reasons, round-time outliers) |
| Intermittent, won't reproduce | `bf trace show <failing run>` — do not re-run |

Key env vars: `QSR_TRACE=1` (flight recorder), `QSR_ASSERT_LEVEL=1`
(invariant assertions), `QSR_BFDIAG_DIR` (artifact root).

**Warm engine vs cold start** — using the wrong one produces plausible but
false numbers with no error: the warm engine (`bf exec`) is valid for
steady-state decode performance, acceptance rates, and numerical experiments.
It is **not** valid for cold-prefill performance, OOM/memory-pressure limits,
or any **load-time** config (`block_size`, `capacity`,
`gpu_memory_utilization`, `max_model_len`, quantization backend) — those are
fixed when the model loads and require a fresh process.

`bfdiag` is pure-stdlib and `runtime/` imports it at module level; keep it
free of third-party dependencies and of import-time side effects.

## Coding Style & Naming Conventions

Use Python with four-space indentation, `snake_case` for modules/functions,
`PascalCase` for classes, and type annotations on public interfaces. Name CUDA
sources descriptively (for example, `nvfp4_gemm_sm120.cu`). Favor clear,
fixed-scope interfaces over generic abstractions: this runtime supports one
model family, one GPU architecture, one GPU, and at most four concurrent slots.

Add a formatter and linter with the first Python code (for example, Ruff), and
run them before submitting changes. Keep generated packed weights, profiles,
and large checkpoints out of Git.

## Testing Guidelines

Add tests beside each capability using `test_<unit>.py` and
`test_<behavior>` names. Compare model layers, logits, MTP acceptance, and GDN
state against the vLLM oracle. Cover prefill, multi-step decode, slot reset and
reuse, batches 1--4, and CUDA Graph replay. For quantized kernels, record
error metrics and top-k-logit agreement; greedy fixtures must not show
systematic token drift.

## Commits & Pull Requests

There is no Git history yet, so use concise imperative commit subjects, e.g.
`Add GDN state reset test`. Keep commits narrowly scoped. Pull requests should
state the affected workload, correctness evidence, benchmark comparison (using
accepted tokens/s and ITL), hardware/CUDA environment, and any changed runtime
assumptions. Include profiler screenshots only when they substantiate a
performance claim.
