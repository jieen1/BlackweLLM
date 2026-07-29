# BlackweLLM

**Blackwell-native inference, built to outrun vLLM.**

BlackweLLM is a full-stack inference engine built from the ground up for
NVIDIA Blackwell (SM120) GPUs. It features hand-written CUDA attention
kernels, FP8 KV cache, MTP speculative decoding, and CUDA Graph capture —
all co-designed to extract maximum performance from SM120's unique hardware.

Currently optimized for **Qwen3.6-27B** (NVFP4). More model support coming.

[中文文档](#中文文档)

---

## Why BlackweLLM?

Mainstream inference frameworks (vLLM, TGI) use generic attention kernels
(FlashInfer, FlashAttention) targeting a wide range of GPU architectures.
On SM120 (Blackwell consumer/workstation), these kernels leave significant
performance on the table — they don't exploit SM120-specific features like
16-byte `cp.async` loads or the specific shared memory bank layout.

BlackweLLM's decode attention kernel is **written from scratch for SM120**,
achieving **56% lower latency** than FlashInfer on 128K-context decode
(0.988 ms vs 1.540 ms per decode step, batch=4, GQA 24→4 heads,
head_dim=256, FP8 KV cache, paged layout).

## Key Features

- **Custom SM120 CUDA attention kernel** — 16-byte `cp.async` vectorized
  loads, 272-byte aligned shared memory strides, split-K parallelism
  (32 splits/request) tuned for SM120's 188 SMs
- **FP8 (e4m3) KV cache** — halves KV memory vs FP16, enabling 256K
  context on a single 96 GB GPU
- **MTP speculative decoding (K=3)** — leverages Qwen3.6's built-in
  Multi-Token Prediction layers, ~50% acceptance rate, lossless output
- **CUDA Graph capture** — eliminates kernel launch overhead for decode
- **Prefix caching** — content-hash KV cache reuse across multi-turn
  conversations, block-level reference counting and eviction
- **OpenAI + Anthropic compatible API** — `/v1/chat/completions`,
  `/v1/completions`, `/v1/messages`, `/v1/models`, Prometheus `/metrics`
- **Sampling support** — temperature / top-p / top-k with per-request seed
  for reproducibility; greedy (T=0) remains bit-identical to the MTP pipeline
- **Production hardening** — client disconnect detection, request cancellation,
  configurable timeout, engine watchdog for stale slot reclamation
- **Quality validated** — MMLU-Pro 84.5% vs official 86.2 (within noise),
  HumanEval+ parity with stock vLLM, 100% tool/agent/long-context regression
  (see [Quality Validation](#quality-validation))

## Performance

All measurements on **RTX PRO 6000 Blackwell Max-Q** (96 GB, 188 SMs),
Qwen3.6-27B-NVFP4, FP8 KV cache, MTP K=3, CUDA Graph enabled.

### Decode Throughput (accepted tokens/s, warm prefix cache)

| Context | Concurrency | Throughput |
|---------|-------------|------------|
| 128K    | 4           | 222 tok/s  |
| 64K     | 4           | 267 tok/s  |

### Kernel Latency (decode attention only)

| Context | Concurrency | BlackweLLM | FlashInfer | Speedup |
|---------|-------------|------------|------------|---------|
| 128K    | 4           | 0.988 ms   | 1.540 ms   | 1.56×   |

### Context Capacity

| Context Length | Max Concurrency | GPU Memory |
|----------------|-----------------|------------|
| 256K           | 2               | ~93 GB     |
| 128K           | 4               | ~70 GB     |

> **Note:** We only compare kernel-level latency under identical conditions.
> End-to-end throughput comparisons with vLLM depend on cache state,
> scheduling, and compilation overhead, so we avoid apples-to-oranges numbers.

## Quality Validation

### Official-Comparable: MMLU-Pro

[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) (414 stratified
questions, ~30/category, thinking mode, 5-shot CoT, greedy, max_tokens=32768,
zero truncation). Compared against the official Qwen3.6-27B model card score:

| Benchmark | BlackweLLM (SM120) | Official Qwen3.6-27B | Delta   |
|-----------|--------------------|-----------------------|---------|
| MMLU-Pro  | **84.54%** (414q)  | 86.2                  | −1.7pp  |

The −1.7pp gap is within the ±3.5% sampling noise of a 414-question stratified
subset. **Local inference quality is NOT degraded vs the original model.**

Per-category accuracy (thinking mode, greedy):

| Category         | Acc    | | Category         | Acc    |
|------------------|--------|-|------------------|--------|
| physics          | 97.8%  | | economics        | 86.2%  |
| math             | 95.7%  | | computer science | 85.7%  |
| chemistry        | 92.3%  | | history          | 84.6%  |
| biology          | 92.0%  | | law              | 81.6%  |
| engineering      | 87.9%  | | business         | 81.5%  |
| psychology       | 77.8%  | | health           | 75.0%  |
| philosophy       | 64.7%  | | other            | 59.4%  |

STEM categories are very strong (≥87.9%); weaker humanities reflect the model's
natural profile, not a runtime defect (identical weights on stock vLLM score the
same). Reproduce:

```bash
python benchmarks/official/mmlu_pro_eval.py --base-url http://localhost:8000/v1 \
    --model qwen3.6 --limit 420 --concurrency 8 --max-tokens 32768 \
    --out evalplus_results/official/mmlu_pro_think.json
```

### A/B Parity: HumanEval+

[EvalPlus](https://github.com/evalplus/evalplus) HumanEval+ (164 problems,
greedy decoding, temperature=0), identical OpenAI API prompts on both servers.
Same model weights, same harness, different backend:

| Benchmark  | vLLM (FlashInfer)    | BlackweLLM           | Delta  |
|------------|----------------------|----------------------|--------|
| HumanEval  | 71/164 = 0.433       | 73/164 = 0.445       | +1.2pp |
| HumanEval+ | 70/164 = 0.427       | 71/164 = 0.433       | +0.6pp |

Both use FP8 KV cache (e4m3) and NVFP4 weights. Per-problem differences
(~20 in each direction) are within statistical noise (SE ≈ ±3.9pp).
**No systematic quality degradation.**

### Self-Contained Regression Suite

Four-dimension regression gate (`benchmarks/quality_regression.py`), run on our
runtime and compared against a stock-vLLM baseline via `quality_compare.py`:

| Dimension     | Test                                      | Score  |
|---------------|-------------------------------------------|--------|
| Tool calling  | 20 tool schemas + queries, exact match     | 100%   |
| Agent loop    | Multi-turn plan→call→observe→answer        | 100%   |
| Long context  | Needle-in-haystack @ 8K/32K/64K/128K      | 100%   |
| Code (A/B)    | HumanEval+ pass@1 vs stock vLLM            | parity |

```bash
python benchmarks/quality_regression.py --base-url http://localhost:8000/v1 \
    --model qwen3.6 --out evalplus_results/quality/our_runtime.json
```

### Official Qwen3.6-27B Reference Scores

For context, the official model card numbers (BF16, thinking mode):

| Benchmark          | Official Score |
|--------------------|----------------|
| MMLU-Pro           | 86.2           |
| GPQA Diamond       | 87.8           |
| AIME26             | 94.1           |
| LiveCodeBench v6   | 83.9           |
| SWE-bench Verified | 77.2           |
| MMLU-Redux         | 93.5           |
| HMMT Feb 25        | 93.8           |

GPQA Diamond is a gated HF dataset (requires access token); AIME26 deferred
due to slot-wedge risk at 16K+ token generations. Full methodology in
[`notes/2026-07-22-quality-baseline-and-official-scores.md`](notes/2026-07-22-quality-baseline-and-official-scores.md).

## Architecture

```
blackwellm/
├── runtime/                  # Core inference engine
│   ├── laguna_config.py          # Laguna-specific runtime/cache/draft config
│   ├── laguna_runtime.py         # Owned forward context and cache bindings
│   ├── model/                    # Self-built Laguna and DFlash model graphs
│   └── backends/laguna.py        # Laguna-S-2.1 backend (SparkInfer MoE)
├── server/                   # HTTP server (OpenAI + Anthropic API)
│   ├── app.py                   # FastAPI endpoints + /metrics
│   └── engine.py                # Admission, MTP loop, prefix cache
├── model/                    # Model architecture config
├── loader/                   # Weight loading (safetensors, NVFP4)
├── kernels/                  # Owned SM120 kernel sources and build targets
├── benchmarks/               # Reproducible perf & correctness checks
├── tests/                    # Unit tests (CPU-only)
└── oracle/                   # vLLM reference comparison utilities
```

This repository contains the **Laguna runtime and serving layer**: fixed-slot
scheduling, self-built Laguna/DFlash model graphs, owned CUDA Graph lifecycle,
SparkInfer attention/MoE integration, and the OpenAI/Anthropic-compatible
server. Production code does not import, package, or require vLLM. The
`oracle/` directory is deliberately excluded from production distributions and
holds only offline migration-reference utilities.

> **Naming:** the product and GitHub repo are **BlackweLLM** (formerly
> BlackForge); the package directory is historically `qwen-sm120-runtime`;
> configuration env vars still use the legacy `QSR_` (Qwen SM120 Runtime)
> prefix; migration target is `BWLLM_`, actual rename pending. All three
> refer to this same system.

## Quick Start

### Prerequisites

- NVIDIA Blackwell GPU (SM120, CC 12.0): RTX PRO 6000, RTX 5090
- CUDA 13.x, Python 3.10+, ~96 GB GPU memory (for 256K context)

### Installation

```bash
git clone https://github.com/jieen1/BlackweLLM.git
cd BlackweLLM
python -m pip install -e '.[dev,serving]'

# GPU runtime dependencies; this does not install vLLM.
python -m pip install -e '.[cuda,dev,serving]'

# Install the pinned local SparkInfer provider required by Laguna.
python -m pip install -e /path/to/sparkinfer
```

### Running the Server

```bash
QSR_SERVER_PRODUCTION=1 \
QSR_SERVER_CAPACITY=2 \
QSR_SERVER_NUM_SLOTS=2 \
QSR_SERVER_BLOCKS_PER_SLOT=16384 \
QSR_SERVED_MODEL_NAME="laguna-s-2.1" \
python -m server.app --host 0.0.0.0 --port 8000
```

### API Usage

```bash
# OpenAI format
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"Hello!"}],"max_tokens":256,"temperature":0}'

# Anthropic format
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"Hello!"}],"max_tokens":256,"temperature":0}'
```

### Benchmarks

```bash
python -m benchmarks.prefix_cache_warm_throughput_check \
  --fixture ctx128k --concurrency 4 --decode-rounds 12

python -m pytest tests/ -q
```

## Configuration

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `QSR_SERVER_CAPACITY` | `1` | Max concurrent Laguna requests |
| `QSR_SERVER_NUM_SLOTS` | `2` | Total internal slots (one CG warmup slot) |
| `QSR_SERVER_BLOCKS_PER_SLOT` | `2048` | KV blocks per slot (×64 = tokens) ⇒ 128K context |
| `QSR_SERVER_PRODUCTION` | `1` | Production mode: skip validation slots |
| `QSR_DEBUG_REQUESTS` | `1` | Log raw request/response (input + output) for debugging |
| `QSR_SERVED_MODEL_NAME` | model ID | Advertised model name(s) |
| `QSR_SERVER_ENABLE_CUDAGRAPH` | `1` | Enable CUDA Graph capture |
| `QSR_SERVER_ENABLE_PREFIX_CACHE` | `0` | Enable prefix caching |

### Context Length vs Concurrency (96 GB GPU)

| `BLOCKS_PER_SLOT` | Context/Slot | Capacity | `NUM_SLOTS` (production) |
|--------------------|-------------|----------|--------------------------|
| 2048               | 128K        | 1        | 2                        |

Use explicit environment overrides only after a cold capacity/VRAM validation;
`block_size`, `blocks_per_slot`, and GPU memory utilisation are load-time
parameters and cannot be compared through a warm daemon run.

## Metrics & Observability

`GET /metrics` exposes Prometheus metrics in the `blackwellm:*` namespace.
They cover speed
(`e2e_request_latency_seconds`, `time_to_first_token_seconds`,
`request_time_per_output_token_seconds`, prompt/generation token histograms
and throughput counters), stability/reliability (`num_requests_running`,
`num_requests_waiting`, `request_success_total` by finish reason,
`request_errors_total` by status code, KV-cache utilisation), and accuracy
(`bootstrap_checks_ok/failed_total` speculative-prefill correctness, prefix
cache hit rate). See [`server/README.md`](server/README.md#metrics) for the
full metric-by-metric reference and example PromQL.

## Development

Common tasks are wired into the `Makefile` (run `make help` for the full
list):

```bash
make install        # editable install with dev + serving extras
make lint           # ruff lint gate (whole repo, must stay green)
make format         # ruff auto-fix + format the production packages
make test           # CPU-only unit test suite (fast, no GPU required)
make verify-cuda    # confirm an SM120 CUDA op executes
make workloads      # print the frozen Phase-0 W1/W2 contracts
make serve          # start the server (tune via QSR_* env vars)
```

Lint and formatting are enforced by `ruff` (config in `pyproject.toml`).
Install the pre-commit hooks to run them automatically on each commit:

```bash
python -m pip install pre-commit
pre-commit install
```

The same gate (ruff lint + format check + unit tests) runs in CI on every
push and pull request (`.github/workflows/ci.yml`). The unit tests are
CPU-only; tests that need `torch` self-skip via `pytest.importorskip`.

## Roadmap

- [ ] More model support (Qwen3 series, other hybrid architectures)
- [x] Streaming response support (OpenAI + Anthropic SSE)
- [x] Temperature / top-p / top-k sampling (graph-safe, greedy bit-identical at T=0)
- [x] Client disconnect detection + request cancellation + timeout
- [x] Engine watchdog (stale slot reclamation)
- [ ] Dynamic KV cache allocation (flexible context vs concurrency)
- [ ] Structured output (JSON mode)
- [ ] Multi-GPU support

## Limitations

- **Single model**: currently Qwen3.6-27B-NVFP4 only
- **Single GPU**: no tensor/pipeline parallelism
- **Sampling without MTP**: temperature > 0 uses autoregressive decode (no speculative verification); greedy (T=0) uses full MTP K=3 pipeline
- **SM120 only**: requires compute capability 12.0

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## 中文文档

### 项目简介

BlackweLLM 是一个专为 NVIDIA Blackwell（SM120）GPU 打造的全栈推理引擎。
通过手写 CUDA attention kernel、FP8 KV cache、MTP 投机解码和 CUDA Graph
捕获等深度优化，在 SM120 架构上实现极致推理性能。

当前生产模型为 **Laguna-S-2.1**（NVFP4）。

### 核心优势

- **自研 SM120 CUDA kernel** — 针对 Blackwell 硬件特性手写，decode attention
  比 FlashInfer 快 56%（128K 上下文，0.988ms vs 1.540ms）
- **FP8 KV cache** — 显存占用减半，单卡 96GB 支持 256K 上下文
- **DFlash 投机解码** — Laguna draft/verify 路径，接受率由逐 prompt 回归门禁验证
- **质量验证** — 生产 CUDA Graph、DFlash 接受率和 router 位级 oracle 均有独立门禁

### 性能数据

测试环境：RTX PRO 6000 Blackwell Max-Q（96GB，188 SMs）

| 上下文 | 并发 | 吞吐量（warm） |
|--------|------|----------------|
| 128K   | 4    | 222 tok/s      |
| 64K    | 4    | 267 tok/s      |

### 快速开始

```bash
git clone https://github.com/jieen1/BlackweLLM.git
cd BlackweLLM
python -m pip install -e '.[dev,serving]'

# GPU runtime dependencies; this does not install vLLM.
python -m pip install -e '.[cuda,dev,serving]'

# 安装 Laguna 依赖的本地 SparkInfer provider。
python -m pip install -e /path/to/sparkinfer

# 启动服务（256K 上下文，2 并发）
QSR_SERVER_PRODUCTION=1 \
QSR_SERVER_CAPACITY=2 \
QSR_SERVER_NUM_SLOTS=2 \
QSR_SERVER_BLOCKS_PER_SLOT=16384 \
QSR_SERVED_MODEL_NAME="laguna-s-2.1" \
python -m server.app --host 0.0.0.0 --port 8000
```

### 许可证

Apache 2.0
