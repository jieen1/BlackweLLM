# Repository Guidelines

> Last verified against the tree: 2026-08-05, commit `8ed33e9` (plus the
> 2026-08-05 working-tree changes documented in
> [`notes/2026-08-05-persistent-prefix-full-hit-fix-and-codex-integration.md`](notes/2026-08-05-persistent-prefix-full-hit-fix-and-codex-integration.md)).
> If something here disagrees with the code, the code wins — and fix this file.

## What this project is

A single-node inference runtime for **NVIDIA Blackwell SM120 only**. One GPU
architecture, one machine, one process, `TP = PP = EP = 1` as a premise rather
than a config option. Serves `poolside/Laguna-S-2.1-NVFP4` (LagunaBackend,
DFlash) and `unsloth/Qwen3.6-27B-NVFP4` (Qwen36Backend, MTP K=3 + MTP/decode
CUDA Graphs + persistent prefix cache + FP8 KV, servable since 2026-08-05);
`Qwen3.6-25B-A3B` is the next roadmap target.

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
    qwen36.py                 Qwen36Backend — persistent prefix + GDN recurrent
                              state, dual-cache-family checkpointing, MTP wiring
    qwen36_mtp.py             MTP speculative engine (draft/sync/verify CUDA Graphs)
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
  formats/                openai · anthropic · responses · stream · tools · thinking · content
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
scripts/                  Runbooks — `run_qwen36_quality.sh` (parallel/resumable
                          quality rerun; server profiles suite/longctx/mmlu/best)
```

Keep components small and layer boundaries explicit. **Production code must not
import `vllm` or anything under `oracle/`** — that boundary is tested, and
`tools/verify_no_vllm_laguna.py` exists to check it.

Local agent-tooling config (`.codex/blackwellm.config.toml`,
`.claude/settings.json`) is intentionally gitignored; the contents and the
Codex/Claude Code end-to-end verification are recorded in
[`notes/2026-08-05-persistent-prefix-full-hit-fix-and-codex-integration.md`](notes/2026-08-05-persistent-prefix-full-hit-fix-and-codex-integration.md)
and
[`notes/2026-08-05-claude-code-via-local-runtime.md`](notes/2026-08-05-claude-code-via-local-runtime.md).

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

**Current state of the gates (2026-08-05): green.** `ruff check .` passes, the
torch-free CI-sim suite passes (1150 passed, 192 skipped), and the full venv
suite passes (1871 passed, 3 skipped). Re-verify rather than trusting it — that
is the point of the date stamp.

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

### `sparkinfer.*.is_supported()` is not a capability signal here

Measured 2026-08-03: **all five** kernels report `False` —
`attention.paged`, `moe.fused_moe`, `gemm.blockscaled`,
`gemm.tensor_fp8_linear`, `norm.mhc` — while production serves Laguna
through the first two and C-LIVE passes 67/67. The gate is
`_lib/gating.py`'s `MIN_CUTLASS_DSL = "4.6.0"` against an installed 4.5.2,
and it is a version-string floor rather than a functional probe.

So `False` here means "the declared floor is not met", not "this kernel does
not work". Nothing in `runtime/` or `server/` branches on it today, and
nothing should start: a fallback taken on this signal would be taken
unconditionally on this machine.

The floor is not arbitrary — `_lib/` carries real 4.5-vs-4.6 workarounds
(MLIR helper relocation, `cute.kernel_smem_size` lowering). Raising or
relaxing it is a sparkinfer change with its own verification, not a
one-line edit.

## External dependencies

- **SparkInfer** (`/home/bot/project/sparkinfer`) — SM120 kernel library.
  **Directly editable as of 2026-08-02**, superseding the earlier hand-it-over
  rule. `origin` is `jieen1/sparkinfer`, this project's own fork, and is where
  changes go; `upstream` is `local-inference-lab/sparkinfer` and is not ours.
  Work in a worktree and merge, as with this repo — sparkinfer has its own
  branch rhythm and has collided with in-flight branch switches before, so
  check `git status` and `git log` before starting. Note that the current
  performance numbers depend on local gating patches that are not upstream yet
  (roadmap T0-5).

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
integration tests. Torch-dependent tests must self-skip via `pytest.importorskip`.
CI runs a torch-free job, so a bare `import torch` at module scope fails
collection there even when the whole suite is green locally. Verify with the
CPU-only interpreter, not just the full one:

```bash
/tmp/ci-sim/bin/python -m pytest -q     # torch-free, mirrors CI job 1
~/.venvs/vllm/bin/python -m pytest -q   # full, mirrors CI job 2
```

### Qwen3.6 on a shared card

`ModelOptFP8Linear`/`ModelOptNVFP4Linear` dequantize weights to BF16 lazily and
cache them for the process's lifetime, so resident memory goes from ~19 GB to
~54 GB+ the first time a forward pass touches every layer. **No memory knob
controls this** -- `GPU_MEM_UTIL`, `num_slots` and `blocks_per_slot` all size
the KV cache, not the weight cache. Lowering them and expecting a smaller
footprint is the natural guess and it is wrong; it was tried twice on
2026-08-02 and the card still filled.

Laguna does **not** behave this way -- verified 2026-08-02 both in code (its
non-MoE layers are plain BF16 on disk and never quantized; MoE experts go
straight to sparkinfer's CUTLASS kernel on packed FP4) and by measurement
(0.83 GiB of growth between weights-settled and graph capture, against
49.72 GiB for Qwen3.6). So "Laguna fits, therefore Qwen3.6 fits" does not
follow: Laguna's footprint is static, Qwen3.6's steps up on first full
forward.

To verify something on a shared card, drive a **single layer** rather than the
whole model. `scripts/b3_probe_gdn_spec_rollback.py` is the worked example: one
real GDN layer with real FP8 weights, no full-model load. See
[`notes/2026-08-02-qwen36-dequant-cache-memory-floor.md`](notes/2026-08-02-qwen36-dequant-cache-memory-floor.md).

### Verifying from a git worktree

`~/.venvs/vllm` has `blackwellm` installed **editable**, and that install's
finder hard-wires `runtime`/`server` to a static path pointing at the *main*
worktree, regardless of cwd. Measured 2026-08-02:

| How you invoke it | `runtime.__file__` resolves to |
|---|---|
| `python -m pytest` from a worktree | that worktree ✅ |
| `python <script>` where the script sits outside the worktree root | **the main worktree** ❌ silent |

`-m` puts cwd on `sys.path[0]`, which is why the test gates are trustworthy
from any worktree.

Related trap, same family, caught 2026-08-02: **stale `__pycache__` bytecode**.
Python validates a `.pyc` against the source's `(mtime, size)`, so an edit that
preserves both — flipping a `"1"` to a `"0"` and back, which is exactly what
red/green verification of a one-character default looks like — can leave the
old bytecode in place. `module.__file__` still reports the `.py`, so reading
the file confirms your change while the interpreter runs the previous version.
This makes a revert-and-confirm-it-goes-red check produce a **false green**.
Before trusting any red/green pair on a small edit:

```bash
find . -name __pycache__ -type d -not -path './.venv/*' -exec rm -rf {} +
``` A standalone script is not: `python foo.py` puts *the
script's own directory* on `sys.path[0]`, so a script living in a scratch
directory imports main's code and says nothing.

This is exactly the shape of an ad-hoc GPU verification script, where the
whole point is comparing one worktree against another. Such a script can
report a perfect match because it loaded the same code twice. Assert the
path before trusting any number:

```python
import sys, pathlib
EXPECTED = "/home/bot/project/<your-worktree>"
sys.path.insert(0, EXPECTED)
import runtime
assert runtime.__file__.startswith(EXPECTED), runtime.__file__
```

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
