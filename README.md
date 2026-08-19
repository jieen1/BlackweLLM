# BlackweLLM

**A single-node inference runtime built exclusively for NVIDIA Blackwell SM120.**

BlackweLLM is an inference engine whose every design decision rests on one
deliberately narrow contract: **one GPU architecture (SM120 / CC 12.0), one
machine, one process, no tensor/pipeline parallelism.** Narrowing is the point —
it is what lets the engine delete the abstractions that generic frameworks
must carry, and hand-write the paths that matter.

Production models today: **`poolside/Laguna-S-2.1-NVFP4`** and the Qwen3.x
27B checkpoints served by the self-built `qwen36` backend. Qwen defaults to
DSpark K=7 with CUDA Graphs, persistent prefix cache, dynamic KV and FP8 KV;
native MTP remains an explicit rollback path. `Qwen3.6-25B-A3B` is the next
roadmap target — see
[`docs/roadmap.md`](docs/roadmap.md).

[中文说明](#中文说明) · [Documentation index](docs/README.md)

---

## Status (2026-08-05)

| | |
|---|---|
| **Servable models** | `Laguna-S-2.1-NVFP4`, Qwen3.x 27B NVFP4 checkpoints |
| **Planned** | `Qwen3.6-25B-A3B` and derivatives |
| **Hardware** | SM120 only (RTX PRO 6000 Blackwell, RTX 5090), single GPU |
| **Dependencies** | Zero vLLM in the production path; SparkInfer for SM120 kernels |
| **Maturity** | Pre-1.0. See [Known issues](#known-issues) before deploying. |

Qwen3.x 27B became servable on the self-built runtime on 2026-08-05. The
current default serving path is the measured DSpark profile; the historical
MTP K=3 quality path remains available explicitly — see [Quality
validation](#quality-validation).
The retired vLLM-era execution path remains read-only under
`oracle/qwen36_vllm/` as an offline reference.

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
- **DSpark speculative decoding (Qwen3.x)** — K=7, ragged verify + draft/verify
  CUDA Graphs captured at load and replayed in the live serving path
- **CUDA Graph capture** — decode, draft, sync, verify graphs (DFlash + MTP)
- **Prefix caching** — content-addressed, reference-counted, LRU eviction
- **OpenAI + Anthropic APIs** — `/v1/chat/completions`, `/v1/completions`,
  `/v1/messages`, `/v1/responses` (OpenAI Responses, used by Codex CLI),
  `/v1/models`, SSE streaming, tool calling, Prometheus `/metrics`
- **Agent CLI integration** — local Codex profile and Claude Code settings talk
  to the runtime directly with no proxy (see
  [Agent CLI integration](#agent-cli-integration))
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

Measured on **Qwen3.6-27B-NVFP4**, served by this repo's own runtime's
historical quality path (`qwen36` backend, MTP K=3 + CUDA Graphs + persistent
prefix cache + FP8 KV; no external engine). The 2026-08-05 rerun used parameters identical to the
historical July run and re-established the numbers on the current build —
orchestrator: [`scripts/run_qwen36_quality.sh`](scripts/run_qwen36_quality.sh)
(parallel, resumable); full evidence:
[`notes/2026-08-05-qwen36-quality-rerun.md`](notes/2026-08-05-qwen36-quality-rerun.md).

| Benchmark | 2026-07 historical | 2026-08-05 current | Verdict |
|---|---|---|---|
| MMLU-Pro (414q, thinking, 5-shot CoT, greedy, max_tokens 32768) | 84.54% | **84.54%** (same 414 question_ids, verified 100% overlap) | exact repro |
| tool (accuracy) | 1.000 (20/20) | 1.000 (20/20) | no regression |
| agent (final answer / tool invocation) | 1.000 (4/4) | 1.000 (4/4) | no regression |
| longctx (8K/32K/64K/128K needles) | 1.000 (12/12) | 1.000 (12/12) | no regression |
| code (HumanEval / HumanEval+, max_tokens 4096) | — | 0.921 / 0.884 | current gate |
| HumanEval+ 768 (README row) | 0.445 / 0.433 | 0.421 / 0.415 | within ±3.9pp SE |

The HumanEval 768 delta is run-to-run greedy variance (task-level: 50 stable
passes, 18 new, 20 regressions; raw length distributions equivalent), not
truncation. Historical methodology:
[`notes/2026-07-22-quality-baseline-and-official-scores.md`](notes/2026-07-22-quality-baseline-and-official-scores.md).

### Qwen3.6 server performance grid (historical MTP path, 2026-08-05)

Full server-side context × concurrency grid on the best serving profile
(MTP K=3, decode + MTP CUDA Graphs, persistent prefix cache, FP8 e4m3 KV,
GPU util 0.92, 3×256K; parameters identical to the historical runs): 5 strict
context lengths (4K/32K/64K/128K/250K served tokens, block-aligned) × 3
concurrencies, 15/15 cells complete with no errors. Every WARM wave hit the
persistent cache (`restores == concurrency`); the 250K cold→warm TTFT dropped
~316 s → ~0.48 s (~656×). Decode throughput with MTP on is no longer a
regression vs the MTP-off CG anchor (4K: 67.9/64.7/65.8 tok/s at c=1/2/3 vs
28.56/47.71/68.59 at c=1/2/4).

Running the grid uncovered and fixed two stacked persistent-cache bugs
(per-slot chunked prefill bypassed COW and overwrote shared scratch KV bytes;
live-slot committed aliases could not be evicted, so the 250K store silently
failed). Both are covered by new regression tests; full suite green.

Evidence: [`notes/2026-08-05-server-perf-grid-mtp-cg-prefix.md`](notes/2026-08-05-server-perf-grid-mtp-cg-prefix.md) ·
harness: [`benchmarks/server_perf_grid.py`](benchmarks/server_perf_grid.py)
(per-cell resumable) ·
results: [`benchmarks/fixtures/server_perf_grid_20260805_v2.json`](benchmarks/fixtures/server_perf_grid_20260805_v2.json)
(pre-fix failure kept as `server_perf_grid_20260805.json`).

**2026-08-06 rerun** (`server_perf_grid_opt1_exact_v2_20260806.json`): the
4K/32K/64K/128K × c1-4 exact-repeat grid (same parameters as the historical
server-path runs) now completes 16/16 cells; a third stacked persistent-cache
bug found during the rerun — the MTP scratch arena watermark was lowered by
later shorter stores, making every 64K/128K restore fail with
"persistent MTP prefix disappeared" — is fixed with a regression test
(`test_scratch_watermark_survives_a_later_shorter_store`). Warm aggregate e2e
at c=4: 4K ~190-195, 32K ~132-137, 64K ~117-124, 128K ~103-107 tok/s.
Historical DirectModelRunner anchors are 227.5 (4K W1-S, different harness),
236.69 (64K), 222.44 (128K); the 64K/128K gap is the documented GPU
verify-path cost, not host-side overhead —
[`notes/2026-08-05-128k-host-side-profile.md`](notes/2026-08-05-128k-host-side-profile.md) §9.

**2026-08-06 devicedraft optimization** (commit `9595f58`): the batched MTP
draft graph no longer round-trips its result to host; draft rows stay on
device through the next verify fill and the GPU-side accept comparison.
128K/c4 round total dropped 82.8 → 73.7 ms (median; `draft_batch` host phase
12.5 → 0.1 ms) and warm aggregate e2e moved from 143.8/163.4 to
**157.5/165.9 tok/s**. Same-parameter cells: 64K/c4 **192.8/202.1**, 4K/c4
289–324 (variance band, ~flat), 128K/c3 130–140, 128K/c2 105–113. Greedy
accept/reject stays byte-identical (device path rebuilds committed from
verifier predictions, which match accepted drafts by construction; new unit
tests assert dict-vs-tensor equality). Evidence:
[`notes/2026-08-06-128k-c4-parity-profiling.md`](notes/2026-08-06-128k-c4-parity-profiling.md)
§12 · fixtures `benchmarks/fixtures/server_perf_grid_devicedraft_fix_*.json`.

Full same-parameter grid on this build (`server_perf_grid_devicedraft_full_20260806.json`,
4K/32K/64K/128K × c1-4, 16/16 cells, 0 errors, every WARM wave a prefix-cache
hit): warm aggregate e2e at c=4 is 4K ~311–319, 32K ~225–262, 64K ~200–211,
128K ~140–146 tok/s. Cross-process variance at 128K/c4 is ~±10% (isolated
rerun 157.5/165.9); see note §12.2 for the full table.

**2026-08-06 evening — 128K/c4 追平并反超历史 headline.** 在 devicedraft
基础上继续压主机热路径：滚动 checkpoint 前缀哈希（~1.1 ms/轮全量重哈希
→ 增量 blake2b）、verify 输入预分配零分配 + lm_head 并入 verify CUDA
Graph（verify_replay 主机段 10.8 → 1.5 ms）、GDN checkpoint 96 次 clone
→ 单次 `_foreach_copy_`，以及 sparkinfer M32 raw-FP8 verifier +
`replay_page_key` 跳过回放 worklist 重建。轮时中位 73.7 → **53.4 ms**
（瓶颈已移到 GPU verify 本体 ~50 ms/轮；128K 稳态 ~15.75 token/轮）。

正式结果（capacity=4/num_slots=5/256K×3/MTP K=3/FP8 KV/CG/prefix cache/
GPU util 0.92，与历史参数一致；每波 4×256 token 全量完成、0 error、
WARM 全命中）：

| 口径 | 128K/c4 warm agg e2e tok/s | 对照 |
|---|---:|---|
| 精确重复前缀 warm（README 表口径），新进程 3 波 | 239.26 / 222.53 / 248.03 | **3/3 超过 222.44** |
| 同上，无 profile 新进程 5 波 | 237.02 / 234.82 / 232.05 / 222.20 / 238.76 | 中位 **234.82**，4/5 超过 |
| 历史协议形态（前缀 + 10240 新后缀），稳态 warm2/3 | 197.37 / 201.30 | ~89–90%（含后缀 TTFT，首波 47 s） |

全网格 c4（warm2）：4K 384/385、32K 353/340、64K **291/262**（> 历史
236.69）、128K 204/225。跑间方差 ±5–10% 为主机 CPU 争用（chrome/celery），
非 GPU；正式对比以新进程隔离样本为准。完整剖析、修复与数据：
[`notes/2026-08-06-128k-c4-parity-profiling.md`](notes/2026-08-06-128k-c4-parity-profiling.md)
§13 · fixtures `server_perf_grid_{cglogits,noprof_headline,noprof_suffix10240}_20260806.json`
· env/复现脚本 `scripts/qwen36_128k_bench_env.sh`。

For **Laguna-S-2.1**, the live gates are the DFlash acceptance regression,
the production CUDA Graph gate, and a bit-level router oracle — all run through
`bfdiag`.

## Repository layout

```
blackwellm/
├── runtime/           Core inference engine
│   ├── backends/          LagunaBackend (DFlash), Qwen36Backend (DSpark K=7
│   │                      default; native MTP rollback; persistent prefix +
│   │                      GDN recurrent state), CUDA
│   │                      Graph lifecycle, SparkInfer attn/MoE adapters
│   ├── model/             Self-built model graph (decoder, linear, embedding, attention)
│   ├── kernels/           Own SM120 kernels (.cu) + Triton kernels
│   ├── block_pool.py      Paged KV + prefix cache (refcount, LRU)
│   ├── model_loading.py   Self-built streaming safetensors loader
│   └── laguna_config.py   Self-built runtime config (replaces vLLM's VllmConfig)
├── server/            HTTP layer
│   ├── app.py             FastAPI endpoints + /metrics
│   ├── engine.py          Admission, fixed-slot scheduling, continuous batching
│   └── formats/           OpenAI / Anthropic / Responses / streaming / tools / thinking
├── bfdiag/            Diagnostics platform (CLI: `bf`) — pure stdlib
├── bfprobe/           Runtime probes
├── loader/            Checkpoint inspection utilities
├── benchmarks/        Reproducible perf & correctness checks
├── scripts/           Runbooks — `run_qwen36_quality.sh` (parallel/resumable
│                      quality rerun; server profiles incl. `best`)
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

### Serve Qwen3.x-27B (DSpark + CUDA Graph + prefix cache)

Point `QSR_SERVER_MODEL_PATH` at a local Qwen3.6/3.8-27B checkpoint. The
launcher selects the measured DSpark profile automatically: K=7, four slots,
128-token KV pages, a 256K logical ceiling, elastic 19,629,342,720-byte KV,
FP8 KV, persistent prefix cache, compact ragged verify and CUDA Graphs:

```bash
QSR_SERVER_MODEL_PATH=/path/to/Qwen3.8-27B-NVFP4 \
  python -m server.app --host 0.0.0.0 --port 8300
```

Same self-built server (no vLLM), model id `qwen3.8`, callable through
`/v1/chat/completions`, `/v1/messages`, or `/v1/responses`.

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

## Agent CLI integration

Both Codex CLI and Claude Code can drive this runtime directly, no proxy:

- **Codex** — `.codex/blackwellm.config.toml` defines profile `blackwellm`
  (`base_url = http://127.0.0.1:8300/v1`, `wire_api = "responses"`, 256K
  context). Run `CODEX_HOME="$PWD/.codex" codex exec -p blackwellm "<task>"`.
- **Claude Code** — project `.claude/settings.json` sets
  `ANTHROPIC_BASE_URL = http://127.0.0.1:8300`, `ANTHROPIC_MODEL = qwen3.8`.
  Run `claude -p "<task>"` from the repo root.
- **Laguna backend** — `.codex/laguna.config.toml` defines profile `laguna`
  (`base_url = http://127.0.0.1:8100/v1`), and
  `.claude/settings.laguna.json` points `ANTHROPIC_BASE_URL` at the same port
  (`ANTHROPIC_MODEL = laguna-s-2.1`). Run
  `CODEX_HOME="$PWD/.codex" codex exec -p laguna "<task>"` or
  `claude -p --settings .claude/settings.laguna.json "<task>"`.

Both files are local tooling state and gitignored on purpose; the exact
contents and end-to-end verification are recorded in
[`notes/2026-08-05-persistent-prefix-full-hit-fix-and-codex-integration.md`](notes/2026-08-05-persistent-prefix-full-hit-fix-and-codex-integration.md)
and
[`notes/2026-08-05-claude-code-via-local-runtime.md`](notes/2026-08-05-claude-code-via-local-runtime.md),
plus the Laguna run in
[`notes/2026-08-05-laguna-codex-cc-e2e.md`](notes/2026-08-05-laguna-codex-cc-e2e.md).

## Configuration

| Env variable | Default | Description |
|---|---|---|
| `QSR_SERVER_CAPACITY` | `1` (Laguna), `4` (Qwen) | Max concurrent requests |
| `QSR_SERVER_NUM_SLOTS` | `2` (Laguna), `4` (Qwen) | Total internal slots |
| `QSR_SERVER_BLOCKS_PER_SLOT` | `2048` | KV blocks per slot (×64 Laguna / ×128 Qwen) |
| `QSR_SERVER_PRODUCTION` | `1` | Production mode: skip validation slots |
| `QSR_SERVER_ENABLE_CUDAGRAPH` | `1` | Enable CUDA Graph capture |
| `QSR_SERVER_ENABLE_PREFIX_CACHE` | `1` | Enable prefix caching |
| `QSR_SERVER_ENABLE_MTP` | `0` | Explicit native-MTP rollback path for Qwen3.x |
| `QSR_SERVER_MTP_K` | `4` | MTP speculative depth (quality/historical profile: 3) |
| `QSR_SERVER_ENABLE_DSPARK` | `1` (Qwen), `0` (Laguna) | Enable DSpark speculative decoding |
| `QSR_SERVER_DSPARK_K` | `7` | DSpark draft depth |
| `QSR_QWEN_KV_MODE` | `elastic` (Qwen), `legacy` (Laguna) | Qwen KV allocation mode |
| `QSR_QWEN_KV_POOL_BYTES` | `19629342720` (Qwen) | Qwen DSpark physical KV budget |
| `QSR_SERVER_ENABLE_DFLASH` | `0` | Enable DFlash speculative engine (Laguna) |
| `QSR_SERVER_REQUEST_TIMEOUT_S` | `600` | Server-side request cap; `0` disables (quality/longctx profiles) |
| `QSR_SERVED_MODEL_NAME` | model ID | Advertised model name(s) |
| `QSR_DEFAULT_REASONING_EFFORT` | `medium` for native Qwen | In-memory Qwen template default; omitted requests are not rewritten, explicit request effort wins |
| `QSR_THINKING_TOKEN_BUDGET` | `8192` for Qwen | Default per-request reasoning-token cap; request-level budget/effort overrides it |
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

Tracked in [`docs/roadmap.md`](docs/roadmap.md) §1.3. As of 2026-08-05, CI, the
test suite, the dependency contract and the thinking/reasoning contract have all
been repaired; what remains open:

- **Structured output is a skeleton.** `response_format` with `json_object` /
  `json_schema` is accepted but **does not constrain generation at all** — the
  grammar mask is never applied. Requests silently come back as free text. Do not
  rely on JSON mode.
- **`seed` re-seeds per token** rather than advancing one generator, so it gives
  determinism but not the usual sampling semantics.
- **Anthropic reasoning is non-standard.** Reasoning is delivered as a
  `reasoning_content_delta` SSE event and a top-level field, *not* as a spec
  `thinking` content block — emitting that block requires a cryptographic
  signature we cannot produce, and a fake one makes Claude Desktop silently drop
  every subsequent content block including `tool_use`
  (see [`docs/roadmap.md`](docs/roadmap.md) §1.4).
- **Prefix cache is page-aware for growing prompts.** Dynamic Qwen entries keep
  the exact prompt-boundary GDN checkpoint and publish the final partial page
  as an authenticated cache entry. A longer conversation reuses the complete
  128-token pages plus the saved recurrent state, then prefills only its new
  suffix; the writable partial tail is copy-on-write detached.
- **Short `max_tokens` can be consumed by thinking.** Qwen3.6 reasons first;
  `max_tokens` is the total completion allowance, including reasoning. The
  Qwen default cap is 8192 so the default 16K allowance leaves answer
  headroom; explicit budgets still require a sufficiently large
  `max_tokens`.
- **One known flaky test** surfaces only in a full-suite run under machine load
  (`test_bfdiag_record.py::test_cli_ls_labels_an_unfinished_record_running`).
- **SparkInfer must be this project's fork.** A stock upstream install starts and
  runs correctly but silently loses the analytic decode path; startup preflight
  reports this as a warning.

## Limitations

- **Two model families** — `Laguna-S-2.1-NVFP4` and Qwen3.x 27B NVFP4;
  `Qwen3.6-25B-A3B` pending
- **Single GPU** — no tensor/pipeline/expert parallelism, by design
- **SM120 only** — compute capability 12.0 required, by design
- **Sampling disables speculation** — `temperature > 0` falls back to
  autoregressive decode; only greedy uses the full speculative pipeline
- **Text only** — no vision/multimodal input

## Roadmap

Full plan: [`docs/roadmap.md`](docs/roadmap.md).

- **M1 (Aug)** — clear red CI/tests, pin the dependency contract, finalise the
  model abstraction design — **done** (2026-08-01/02)
- **M2/M3** — model abstraction + Qwen3.6-27B correctness and serving —
  **substantially landed** on the self-built path (2026-08-05): `qwen36`
  backend, historical MTP quality path, persistent prefix cache, quality
  baseline re-established
- **M4 (in progress)** — Qwen3.x-27B performance tuning; DSpark K=7 is now the
  measured default, with native MTP retained for rollback/A-B validation
- **M5 (Dec)** — `Qwen3.6-25B-A3B` bring-up and serving
- **M6 (Jan 2027)** — soak testing, release gates, `0.2.0`
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

**当前可服务模型**：`poolside/Laguna-S-2.1-NVFP4` 与 Qwen3.x 27B NVFP4
检查点（自研 `qwen36` 后端，默认 DSpark K=7 + CUDA Graph + 持久前缀缓存 +
FP8 KV；历史 MTP K=3 质量路径仍可显式启用）。
**下一个路线图目标**是 `Qwen3.6-25B-A3B`。

完整中文文档见 [`docs/README.md`](docs/README.md)：

- [`docs/roadmap.md`](docs/roadmap.md) — 定位、现状盘点、轨道与里程碑、风险、待拍板事项
- [`docs/architecture.md`](docs/architecture.md) — 当前架构与目标架构
- [`docs/model-support.md`](docs/model-support.md) — 模型支持矩阵与接入新模型的操作指南
- [`docs/diagnostics-guide.md`](docs/diagnostics-guide.md) — bfdiag 诊断平台使用指南（排查问题前必读）

部署方式、配置项、性能与质量数据见上方英文段落。
**上线前请先读 [Known issues](#known-issues)**——当前门禁（ruff + 两套
pytest：完整 1871 passed / 3 skipped，CPU 镜像 1150 passed / 192 skipped）
为绿，但仍有若干已知的行为未定义项。
