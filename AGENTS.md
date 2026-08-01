# Repository Guidelines

> Last verified against the tree: 2026-08-01, commit `ce21eb5`.
> If something here disagrees with the code, the code wins — and fix this file.

## What this project is

A single-node inference runtime for **NVIDIA Blackwell SM120 only**. One GPU
architecture, one machine, one process, `TP = PP = EP = 1` as a premise rather
than a config option. Currently serves `poolside/Laguna-S-2.1-NVFP4`;
Qwen3.6 support is the active roadmap priority.

**Read before making non-trivial changes:**
[`docs/roadmap.md`](docs/roadmap.md) (where the project is going and what is
currently broken) · [`docs/architecture.md`](docs/architecture.md) (how it is
layered) · [`docs/diagnostics-guide.md`](docs/diagnostics-guide.md)
(**mandatory before writing any diagnostic code**).

## Project structure

```
runtime/                  Core inference engine
  backends/
    laguna.py                 LagunaBackend — slots, KV ownership, attn metadata,
                              SWA ring KV, CUDA Graph lifecycle, MoE patching
    laguna_cuda_graph.py      decode / verify graph capture & replay
    laguna_dflash.py          DFlash speculative engine (draft + verify)
    laguna_dflash_cudagraph.py
    laguna_sparkinfer_attn.py SparkInfer paged-attention adapter
    laguna_sparkinfer_moe.py  SparkInfer fused-MoE adapter
    bf_attention.py           KV write (FP8 quant + paged scatter) + attn dispatch
  model/                  Self-built model graph
    laguna_model.py · laguna_decoder.py · laguna_dflash_model.py
    plain_linear.py · plain_embedding.py · plain_attention.py · fp8_linear.py
  kernels/                laguna_router_sm120.cu · nvfp4_gemm_sm120.cu
                          rope.py · fused_rms_norm.py · fused_kv_scatter.py (Triton)
  block_pool.py           Paged KV + prefix cache (block hash, refcount, LRU)
  model_loading.py        Self-built streaming safetensors loader
  laguna_config.py        Self-built runtime config (replaces vLLM's VllmConfig)
  laguna_router.py        Native SM120 router adapter (EXPERTS=256, TOP_K=10)
  model_spec.py           Architecture description (currently thin — see roadmap A1)
  mtp_accept.py · sampling.py · logprobs.py · structured_output.py

server/                   HTTP layer
  app.py                  FastAPI endpoints + /metrics + /debug/*
  engine.py               ServerEngine — admission, fixed slots, continuous batching
  formats/                openai · anthropic · stream · tools · thinking · content
  metrics.py · tracing.py

bfdiag/                   Diagnostics platform, CLI `bf`. Pure stdlib, imported at
                          module level by runtime/ — keep it free of third-party
                          deps and of import-time side effects.
bfprobe/                  Runtime probes
loader/                   Checkpoint inspection utilities
model/                    Architecture config parsing
benchmarks/               Perf & correctness checks (136 files — see warning below)
tests/                    Unit tests, no model weights required
docs/                     Live documentation (see docs/README.md)
notes/                    Investigation records (see notes/README.md for the index)
oracle/                   Offline reference only. NOT packaged, NOT importable from
                          production code. Contains the retired Qwen3.6/vLLM path.
```

Keep components small and layer boundaries explicit. **Production code must not
import `vllm` or anything under `oracle/`** — that boundary is tested, and
`tools/verify_no_vllm_laguna.py` exists to check it.

## Build, test, and development

```bash
make install        # editable install with dev + serving extras
make lint           # ruff check . — whole repo, must stay green
make format         # ruff auto-fix + format runtime server loader model oracle tests tools
make test           # pytest -q
make verify-cuda    # confirm an SM120 CUDA op executes
make serve          # start the server (tune via QSR_* env vars)
make build-laguna-router   # build the SM120 router .so + provenance manifest
```

Lint and format are enforced by `ruff` (config in `pyproject.toml`) and run in CI
on every push and PR. `benchmarks/` and `tests/debug/` are lint-relaxed for style
rules but keep the bug-catching rules (F821, F811, F401, E9xx) active.

**Current state of the gates (2026-08-01): CI is red and 4 tests fail.**
See [`docs/roadmap.md`](docs/roadmap.md) §1.2. Do not assume green.

## Diagnostics — read this before debugging anything

This machine has **one GPU and no parallelism**; a test run costs minutes. The
only lever on iteration speed is **how much each GPU run tells you**. That is
what `bfdiag` (CLI `bf`) is for. **Full guide:
[`docs/diagnostics-guide.md`](docs/diagnostics-guide.md) — read it before writing
any diagnostic code.**

Three rules that override habit:

1. **Do not write another one-off script under `benchmarks/`.** There are already
   136 of them with zero compounding value. Submit experiments to the warm
   engine instead: `bf exec <script>`.
2. **Run `bf diff <A> <B>` before comparing any two numbers.** On 2026-07-27 two
   acceptance rates (1.000 vs 0.687) were compared as evidence the backend had
   caught up; the two runs had used different prompts, and a full day was lost.
   `bf diff` prints the config delta and refuses to call incomparable runs
   comparable.
3. **When something fails, read the existing trace first — do not re-run.** The
   flight recorder is on-by-default-cheap; the failing run's per-round history is
   already on disk. Re-running costs minutes and may not reproduce.

| Symptom | Sequence |
|---|---|
| Acceptance rate dropped | `bf diff` → `bf trace show` (**reject_position histogram** — its *shape* separates several distinct bug classes) → `bf divergence` |
| Garbage output / NaN | `QSR_ASSERT_LEVEL=1` re-run → `bf trace show` → `bf divergence` |
| Suddenly slower | `bf diff` → `bf trace show` (CG hit rate, eager-fallback reasons, round-time outliers) |
| Intermittent, won't reproduce | `bf trace show <failing run>` — do not re-run |

Key env vars: `QSR_TRACE=1` (flight recorder), `QSR_ASSERT_LEVEL=1` (invariant
assertions), `QSR_BFDIAG_DIR` (artifact root).

**Warm engine vs cold start** — using the wrong one produces plausible but false
numbers *with no error*. The warm engine (`bf exec`) is valid for steady-state
decode performance, acceptance rates, and numerical experiments. It is **not**
valid for cold-prefill performance, OOM/memory-pressure limits, or any
**load-time** config (`block_size`, `capacity`, `gpu_memory_utilization`,
`max_model_len`, quantization backend) — those are fixed when the model loads and
require a fresh process.

## External dependencies you must not edit directly

- **SparkInfer** (`/home/bot/project/sparkinfer`) — SM120 kernel library, owned by
  a separate team. Read-only profiling is fine; **source changes must be written
  up and handed over, not made in place.** Note that the current performance
  numbers depend on local gating patches that are not upstream yet (roadmap T0-5).

## Coding style

Python, four-space indent, `snake_case` modules/functions, `PascalCase` classes,
type annotations on public interfaces. Name CUDA sources descriptively
(e.g. `nvfp4_gemm_sm120.cu`).

**Favor fixed-scope interfaces over generic abstractions.** This runtime supports
one GPU architecture, one machine, and a handful of concurrent slots. An
abstraction whose only justification is "we might need multi-GPU later" is a
liability — the roadmap explicitly rules that out. The one abstraction the
roadmap *does* call for (the model layer) is being built by generalising from a
real second model, not by anticipating a hypothetical one.

Keep generated packed weights, profiles, and large checkpoints out of Git.

## Testing

Add tests beside each capability using `test_<unit>.py` / `test_<behavior>` names.
Cover prefill, multi-step decode, slot reset and reuse, batches 1–4, and CUDA
Graph replay. For quantized kernels, record error metrics and top-k-logit
agreement; greedy fixtures must not show systematic token drift.

Unit tests must not require downloading model weights unless explicitly marked as
integration tests. Torch-dependent tests should self-skip via
`pytest.importorskip` — **three modules currently violate this and break CI**
(`test_swa_scratch_lifecycle.py`, `test_bfdiag_cold_capacity.py`,
`test_bfdiag_ring.py`); fixing that is roadmap T0-1.

Correctness before performance, always. The most expensive bugs in this
repository's history — the block_size-128 acceptance regression, the
fused_kv_scatter value-stride bug, the FP8 rounding tie — all came from a
performance change that ran ahead of a correctness gate. See
[`docs/model-support.md`](docs/model-support.md) §6 for the full trap list.

## Commits & pull requests

Concise imperative subjects (`Add GDN state reset test`). Keep commits narrowly
scoped. The established commit-body convention in this repo carries the reasoning:

```
Constraint: <what could not be violated>
Rejected: <alternative> | <why it was rejected>
Confidence: high | medium | low
Scope-risk: narrow | moderate | broad
Reversibility: clean | messy
Tested: <what was actually run>
Not-tested: <what was not, and what would close it>
```

`Tested:` must reflect what was actually run. `d52a3b1` claimed "unit tests pass"
while shipping a red test — that is how a contradiction reaches `main`.

PRs should state the affected workload, correctness evidence, benchmark
comparison (accepted tokens/s and ITL), hardware/CUDA environment, and any
changed runtime assumptions. Include profiler output only when it substantiates
a performance claim.

## Documentation

`docs/` holds live documentation; `notes/` holds investigation records and
evidence. When a document's premises stop being true, move it to `docs/archive/`
and record why in `docs/archive/README.md` — do not leave it rotting in place.
`notes/` files are referenced by path from code comments, so **do not move
them**; mark their status in `notes/README.md` instead.
