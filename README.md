# BlackweLLM

**A single-node inference runtime built exclusively for NVIDIA Blackwell SM120.**

BlackweLLM is an inference engine whose every design decision rests on one
deliberately narrow contract: **one GPU architecture (SM120 / CC 12.0), one
machine, one process, no tensor/pipeline parallelism.** Narrowing is the point —
it is what lets the engine delete the abstractions that generic frameworks
must carry, and hand-write the paths that matter.

Production model today: **`poolside/Laguna-S-2.1-NVFP4`**.
Qwen3.6 series support is the current roadmap priority — see
[`docs/roadmap.md`](docs/roadmap.md).

[中文说明](#中文说明) · [Documentation index](docs/README.md)

---

## Status (2026-08-01)

| | |
|---|---|
| **Servable models** | `Laguna-S-2.1-NVFP4` (only) |
| **Planned** | `Qwen3.6-27B`, `Qwen3.6-25B-A3B` and derivatives |
| **Hardware** | SM120 only (RTX PRO 6000 Blackwell, RTX 5090), single GPU |
| **Dependencies** | Zero vLLM in the production path; SparkInfer for SM120 kernels |
| **Maturity** | Pre-1.0. See [Known issues](#known-issues) before deploying. |

> ⚠️ **Qwen3.6 is not currently servable.** It was supported by an earlier
> vLLM-based execution path that was retired on 2026-07-30 (that code now
> lives read-only under `oracle/qwen36_vllm/`). Re-adding it through the new
> model abstraction layer is the M2→M4 milestone.

## Why

Mainstream engines target SM90/SM100 datacenter parts first. On SM120 —
which has **no wgmma and no BF16 tensor core** — their fast paths fall back to
slow ones. BlackweLLM writes the SM120 path directly, and because
`world_size == 1` is a compile-time premise rather than a runtime option, the
entire distributed abstraction layer simply does not exist.

## Features

- **Self-built execution stack** — model graph, weight loading, runtime config,
  and forward context are all owned; the production import graph has no vLLM edge
- **SparkInfer SM120 kernels** — paged attention (FP8 KV, CUDA-graph replayable)
  and fused NVFP4 MoE
- **Own SM120 kernels** — MoE router (`.cu`), plus RoPE / RMSNorm / KV-scatter (Triton)
- **FP8 (e4m3) KV cache** — 256K context on a single 96 GB GPU
- **Fixed-slot continuous batching** — dedicated engine thread owns the CUDA
  context; the asyncio side never blocks
- **DFlash speculative decoding** — 96.3–100% acceptance on Laguna
- **CUDA Graph capture** — decode, draft, and verify graphs
- **Prefix caching** — content-addressed, reference-counted, LRU eviction
- **OpenAI + Anthropic APIs** — `/v1/chat/completions`, `/v1/completions`,
  `/v1/messages`, `/v1/models`, SSE streaming, tool calling, Prometheus `/metrics`
- **Production hardening** — client-disconnect detection, cancellation,
  request timeout, watchdog for stale slot reclamation
- **bfdiag diagnostics platform** — flight recorder, run records, run-comparability
  checks, warm-engine daemon (see [`docs/diagnostics-guide.md`](docs/diagnostics-guide.md))

## Performance

`Laguna-S-2.1-NVFP4` on **RTX PRO 6000 Blackwell Max-Q** (96 GB, 188 SMs),
FP8 KV cache, DFlash speculative decoding, CUDA Graph on, analytic decode path.
Measured 2026-07-31 and independently reproduced 2026-08-01 on the current
SparkInfer fork head (run records `fc6b3376785a`, `781e1edbf37b`): acceptance
matches to the fourth decimal on all four workloads, throughput lands inside
the recorded ranges on 7 of 8 samples. See
[`notes/2026-08-01-sparkinfer-patch-recovery-and-repro.md`](notes/2026-08-01-sparkinfer-patch-recovery-and-repro.md).

| Workload | Throughput | Acceptance |
|---|---|---|
| galaxy-4K | 395–401 tok/s | 100% |
| fox-4K | 353–357 tok/s | 96.3–97.0% |
| code-4K | 341–359 tok/s | 97.8% |
| fox-64K | 353–368 tok/s | 96.9% |

Context capacity: **[unverified for Laguna]** — the figures previously quoted
here (256K @ concurrency 2, 128K @ concurrency 4) come from Qwen3.6-27B's old
capacity table and contradict Laguna's own memory audit. A live 3-slot × 256K
configuration was measured at 94.2 / 97.9 GB on 2026-08-01; Laguna's weights
alone are 67 GB. Re-deriving Laguna's real context × concurrency envelope is
tracked as roadmap item D2 (memory planner).

> We publish kernel-level latency comparisons only under identical conditions.
> End-to-end throughput vs other engines depends on cache state, scheduling, and
> compilation overhead, so we avoid apples-to-oranges numbers.

## Quality validation

> **Read the model column.** The quality numbers below were measured on
> **Qwen3.6-27B** in July 2026, on the execution path that has since been
> retired. They are retained as evidence that the runtime approach does not
> degrade quality — **they are not current-build numbers, and the current build
> cannot serve that model.** Re-establishing them on the new path is a
> Track B verification item.

| Benchmark | Model | Runtime | Score | Reference |
|---|---|---|---|---|
| MMLU-Pro (414q, thinking, greedy) | Qwen3.6-27B-NVFP4 | BlackweLLM, 2026-07-22 | 84.54% | official card 86.2 (−1.7pp, within ±3.5% sampling noise) |
| HumanEval | Qwen3.6-27B-NVFP4 | BlackweLLM vs stock vLLM | 0.445 vs 0.433 | +1.2pp (SE ≈ ±3.9pp) |
| HumanEval+ | Qwen3.6-27B-NVFP4 | BlackweLLM vs stock vLLM | 0.433 vs 0.427 | +0.6pp |

For **Laguna-S-2.1**, the live gates are the DFlash acceptance regression,
the production CUDA Graph gate, and a bit-level router oracle — all run through
`bfdiag`. Methodology: [`notes/2026-07-22-quality-baseline-and-official-scores.md`](notes/2026-07-22-quality-baseline-and-official-scores.md).

## Repository layout

```
blackwellm/
├── runtime/           Core inference engine
│   ├── backends/          LagunaBackend + SparkInfer attn/MoE adapters,
│   │                      CUDA Graph lifecycle, DFlash speculative engine
│   ├── model/             Self-built model graph (decoder, linear, embedding, attention)
│   ├── kernels/           Own SM120 kernels (.cu) + Triton kernels
│   ├── block_pool.py      Paged KV + prefix cache (refcount, LRU)
│   ├── model_loading.py   Self-built streaming safetensors loader
│   └── laguna_config.py   Self-built runtime config (replaces vLLM's VllmConfig)
├── server/            HTTP layer
│   ├── app.py             FastAPI endpoints + /metrics
│   ├── engine.py          Admission, fixed-slot scheduling, continuous batching
│   └── formats/           OpenAI / Anthropic / streaming / tools / thinking
├── bfdiag/            Diagnostics platform (CLI: `bf`) — pure stdlib
├── bfprobe/           Runtime probes
├── loader/            Checkpoint inspection utilities
├── benchmarks/        Reproducible perf & correctness checks
├── tests/             Unit tests (CPU-only)
├── docs/              Roadmap, architecture, model support, diagnostics
├── notes/             Investigation records and evidence archive
└── oracle/            Offline reference utilities — NOT shipped, NOT importable
                       from production code (includes the retired Qwen3.6 path)
```

> **Naming:** the product and GitHub repo are **BlackweLLM** (formerly
> BlackForge); the working directory is historically `qwen-sm120-runtime`;
> environment variables still use the legacy `QSR_` prefix. Unifying all three
> on `blackwellm` / `BWLLM_` is roadmap item D5.

## Quick start

### Prerequisites

- NVIDIA Blackwell GPU, SM120 / CC 12.0 (RTX PRO 6000, RTX 5090)
- CUDA 13.x, Python 3.10+, ~96 GB GPU memory for 256K context
- A checkout of [this project's SparkInfer fork](https://github.com/jieen1/sparkinfer)
  — `master` carries two commits on top of upstream that the analytic decode path
  needs (see [`docs/sparkinfer-fork-delta.md`](docs/sparkinfer-fork-delta.md))

### Install

```bash
git clone https://github.com/jieen1/BlackweLLM.git
cd BlackweLLM
python -m pip install -e '.[cuda,dev,serving]'   # does not install vLLM

git clone https://github.com/jieen1/sparkinfer.git
python -m pip install -e ./sparkinfer             # verified at master @ 0844a4f
```

Startup runs a preflight check (GPU architecture, CUDA, torch, SparkInfer,
checkpoint) before any weights load, and refuses to start on a fatal mismatch.
`--skip-preflight` bypasses it.

### Serve

```bash
QSR_SERVER_PRODUCTION=1 \
QSR_SERVER_CAPACITY=2 \
QSR_SERVER_NUM_SLOTS=2 \
QSR_SERVER_BLOCKS_PER_SLOT=16384 \
QSR_SERVED_MODEL_NAME="laguna-s-2.1" \
python -m server.app --host 0.0.0.0 --port 8000
```

> These four variables are coupled: get them wrong and you either OOM at load
> time or silently waste VRAM. Automating this is roadmap item D2.

### Call

```bash
# OpenAI format
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"laguna-s-2.1","messages":[{"role":"user","content":"Hello!"}],"max_tokens":256,"temperature":0}'

# Anthropic format
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"laguna-s-2.1","messages":[{"role":"user","content":"Hello!"}],"max_tokens":256}'
```

## Configuration

| Env variable | Default | Description |
|---|---|---|
| `QSR_SERVER_CAPACITY` | `1` | Max concurrent requests |
| `QSR_SERVER_NUM_SLOTS` | `2` | Total internal slots (one extra for CG warmup) |
| `QSR_SERVER_BLOCKS_PER_SLOT` | `2048` | KV blocks per slot (×64 = tokens) |
| `QSR_SERVER_PRODUCTION` | `1` | Production mode: skip validation slots |
| `QSR_SERVER_ENABLE_CUDAGRAPH` | `1` | Enable CUDA Graph capture |
| `QSR_SERVER_ENABLE_PREFIX_CACHE` | `0` | Enable prefix caching |
| `QSR_SERVED_MODEL_NAME` | model ID | Advertised model name(s) |
| `QSR_DEBUG_REQUESTS` | `0` | Log raw request/response |
| `QSR_TRACE` | `0` | bfdiag flight recorder |
| `QSR_ASSERT_LEVEL` | `0` | Runtime invariant assertions |

Load-time parameters (`block_size`, `blocks_per_slot`, `gpu_memory_utilization`,
`max_model_len`) are fixed when the model loads. Changing them requires a fresh
process — a warm-engine run will not pick them up **and will not error**.

## Metrics

`GET /metrics` exposes Prometheus metrics under `blackwellm:*`, covering speed
(e2e latency, TTFT, time-per-output-token, throughput counters), stability
(running/waiting requests, success by finish reason, errors by status code, KV
utilisation), and accuracy (speculative-prefill bootstrap checks, prefix cache
hit rate). Full per-metric reference: [`server/README.md`](server/README.md#metrics).

## Development

```bash
make install        # editable install with dev + serving extras
make lint           # ruff lint gate (whole repo)
make format         # ruff auto-fix + format production packages
make test           # unit test suite
make verify-cuda    # confirm an SM120 CUDA op executes
make serve          # start the server
```

**Before writing any diagnostic code, read
[`docs/diagnostics-guide.md`](docs/diagnostics-guide.md).** This machine has one
GPU and no parallelism; a test run costs minutes, so the only lever on iteration
speed is how much each GPU run tells you. Three rules override habit: don't write
another one-off script under `benchmarks/`; run `bf diff` before comparing any two
numbers; when something fails, read the existing trace instead of re-running.

## Known issues

Tracked in [`docs/roadmap.md`](docs/roadmap.md) §1.3. As of 2026-08-01, CI, the
test suite, the dependency contract and the thinking/reasoning contract have all
been repaired; what remains open:

- **Structured output is a skeleton.** `response_format` with `json_object` /
  `json_schema` is accepted but **does not constrain generation at all** — the
  grammar mask is never applied. Requests silently come back as free text. Do not
  rely on JSON mode.
- **`stop` sequences are not implemented** on either protocol.
- **`seed` re-seeds per token** rather than advancing one generator, so it gives
  determinism but not the usual sampling semantics.
- **Anthropic reasoning is non-standard.** Reasoning is delivered as a
  `reasoning_content_delta` SSE event and a top-level field, *not* as a spec
  `thinking` content block — emitting that block requires a cryptographic
  signature we cannot produce, and a fake one makes Claude Desktop silently drop
  every subsequent content block including `tool_use`
  (see [`docs/roadmap.md`](docs/roadmap.md) §1.4).
- **One known flaky test** surfaces only in a full-suite run under machine load
  (`test_bfdiag_record.py::test_cli_ls_labels_an_unfinished_record_running`).
- **SparkInfer must be this project's fork.** A stock upstream install starts and
  runs correctly but silently loses the analytic decode path; startup preflight
  reports this as a warning.

## Limitations

- **Single model** — `Laguna-S-2.1-NVFP4` only, today
- **Single GPU** — no tensor/pipeline/expert parallelism, by design
- **SM120 only** — compute capability 12.0 required, by design
- **Sampling disables speculation** — `temperature > 0` falls back to
  autoregressive decode; only greedy uses the full speculative pipeline
- **Text only** — no vision/multimodal input

## Roadmap

Full plan: [`docs/roadmap.md`](docs/roadmap.md).

- **M1 (Aug)** — clear red CI/tests, pin the dependency contract, finalise the
  model abstraction design
- **M2 (Sep)** — land the abstraction layer with zero Laguna regression;
  Qwen3.6 fact baseline
- **M3 (Oct)** — Qwen3.6-27B correctness + serving; one-command startup
- **M4 (Nov)** — Qwen3.6-27B performance + MTP speculation; 25B-A3B bring-up
- **M5 (Dec)** — 25B-A3B serving; usability and compatibility close-out
- **M6 (Jan 2027)** — soak testing, release gates, `0.2.0`

Explicitly out of scope: multi-GPU, multi-node, non-SM120 architectures,
vision/multimodal, training/fine-tuning.

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## 中文说明

**BlackweLLM 是一个只面向 NVIDIA Blackwell SM120、只做单机部署的推理运行时。**

它的每个设计决策都建立在一份刻意收窄的硬件合同上：单一 GPU 架构
（SM120 / CC 12.0）、单机、单进程、无张量/流水线并行。收窄本身就是价值来源——
它让这个引擎可以删掉通用框架不得不背的抽象，并把关键路径手写出来。

**当前可服务模型**：`poolside/Laguna-S-2.1-NVFP4`。
**Qwen3.6 系列支持是当前路线图第一优先级**（曾经支持过，随 vLLM 剥离被摘除，
将走新的模型抽象层重新接入）。

完整中文文档见 [`docs/README.md`](docs/README.md)：

- [`docs/roadmap.md`](docs/roadmap.md) — 定位、现状盘点、轨道与里程碑、风险、待拍板事项
- [`docs/architecture.md`](docs/architecture.md) — 当前架构与目标架构
- [`docs/model-support.md`](docs/model-support.md) — 模型支持矩阵与接入新模型的操作指南
- [`docs/diagnostics-guide.md`](docs/diagnostics-guide.md) — bfdiag 诊断平台使用指南（排查问题前必读）

部署方式、配置项、性能与质量数据见上方英文段落。
**上线前请先读 [Known issues](#known-issues)**——当前 CI 是红的，
且有若干已知的行为未定义项。
