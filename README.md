# BlackweLLM

**A single-node inference runtime built exclusively for NVIDIA Blackwell SM120 — serving Qwen3.8-27B-NVFP4 at 256K context on one 96 GB GPU.**

Every design decision rests on one deliberately narrow contract: **one GPU
architecture (SM120 / CC 12.0), one machine, one process, TP = PP = EP = 1.**
Narrowing is the point — it lets the engine delete the abstractions generic
frameworks must carry (there is no distributed layer at all) and hand-write the
paths that matter. The production import graph has **zero vLLM**: the model
graph, weight loading, and runtime config are all owned by this repo.

---

## Measured performance

Qwen3.8-27B-NVFP4 · RTX PRO 6000 Blackwell Max-Q (96 GB) · DFlash2 K=7
speculative decoding · FP8 e4m3 KV cache · CUDA Graphs · persistent prefix
cache. All numbers measured 2026-08-25/26 with
[`benchmarks/server_perf_grid.py`](benchmarks/server_perf_grid.py) against a
freshly started server; completion SHA-256 verified identical across runs.

These are the historical Qwen3.8 DSpark/DFlash2 benchmark profiles. The current
Qwen3.8 Flash-Next application has a separate one-slot/MTP launch contract and
is documented in [`docs/qwen38-flash-next-ops.md`](docs/qwen38-flash-next-ops.md);
do not use the numbers below as a Flash-Next baseline.

| Scenario | Result |
|---|---|
| Decode @ 128K context, c=4 | **169 tok/s per request** |
| Aggregate throughput @ 128K, c=4 (prefix hit) | **588 tok/s** |
| Aggregate throughput @ c=8, 1024 output tokens | **872 tok/s** |
| Cold prefill, 128K prompt × 4 concurrent | TTFT **34.2 s** |
| Warm restart of a cached 128K prefix | TTFT **< 0.15 s** |
| Speculative acceptance | ~88% (896/1020 tokens, mean 6.6 of 7 per round) |
| Steady memory, fresh load (4 × 256K slots) | **57 GiB** / 96 GiB |

Streaming TTFT on a warm slot is single-digit milliseconds; the server speaks
three API dialects simultaneously (below) and is exercised daily as the backend
of [opencode](https://opencode.ai) agent sessions running multi-hour tool-use
workloads.

## What it is

- **Qwen3.8-27B-NVFP4, end to end.** Hybrid architecture: 48 gated-delta-net
  (GDN) layers + 16 full-attention layers, served by a self-built backend with
  a recurrent-state pool, chunked GDN prefill, and paged FP8 KV.
- **DFlash2 speculative decoding**, K = 7, with a dedicated 5-layer draft model:
  draft, verify, ragged-verify and accept CUDA Graphs captured at load.
- **Native SM120 kernels**: NVFP4 W4A4 dense GEMM via CUTLASS DSL
  (bf16/fp8/fp4 tile selection per shape), MoE router (`*.cu`), RoPE /
  RMSNorm / fused KV-scatter (Triton), fused causal-conv + SiLU.
- **FP8 KV cache + persistent prefix cache** — content-addressed,
  reference-counted, LRU-evicted; cross-request prefix restore in milliseconds;
  dynamic (VMM-backed) arena so capacity is paid for on demand.
- **Fixed-slot continuous batching** — a dedicated engine thread owns the CUDA
  context; the asyncio side never blocks. Client disconnects, cancellations,
  timeouts, and stale-slot watchdogs are handled.
- **Memory that stays flat**: narrow sliding-window draft-KV rows, lazy
  dequantization-free weight paths, and no hidden BF16 up-caches — a 4-slot ×
  256K deployment fits in 57 GiB with room for concurrency 8.

## API surface

One server, three dialects — all verified daily:

| Dialect | Endpoint | Supports |
|---|---|---|
| OpenAI Chat | `/v1/chat/completions` | streaming, tool calling, `reasoning_effort` |
| OpenAI Responses | `/v1/responses` | `message` / `function_call` / `function_call_output` items, SSE event stream, `reasoning.effort` |
| Anthropic Messages | `/v1/messages` | streaming (`reasoning_content_delta`), `tool_use`, thinking budget |

Plus `/v1/completions`, `/v1/models`, `/v1/messages/count_tokens`,
Prometheus `/metrics`, and debug endpoints (`/debug/stats`, `/debug/traces`).

Thinking effort is adjustable on every dialect (`reasoning_effort` /
`reasoning.effort` / `thinking.budget_tokens`).

## Quick start

Requirements: NVIDIA SM120 GPU (RTX PRO 6000 Blackwell / RTX 5090), driver ≥
580, CUDA 13.x, Python 3.12+, [SparkInfer/b12x](https://github.com/local-inference-lab/sparkinfer)
kernels.

For the current Qwen3.8 Flash-Next checkpoint, use the reproducible
[Flash-Next operations guide](docs/qwen38-flash-next-ops.md). It pins the
validated Python 3.14 environment, `flashnext` backend, `127.0.0.1:8300`,
one-slot/256K profile, MTP `K=3`, CUDA Graphs, prefix cache and all required
SM120 tuning flags. The generic `make serve` target is intentionally only a
thin development wrapper; it does not select a model or a production profile.

```bash
cd /home/bot/project/qwen-sm120-runtime
export QSR_SERVER_MODEL_PATH=/home/bot/models/Qwen3.8-Flash-Next-NVFP4-RadixArk
export QSR_SERVER_BACKEND=flashnext
export QSR_SERVED_MODEL_NAME="qwen3.8 qwen3.8-flash-next"
export QSR_SERVER_ENABLE_CUDAGRAPH=1 QSR_SERVER_ENABLE_PREFIX_CACHE=1
export QSR_SERVER_ENABLE_MTP=1 QSR_SERVER_MTP_K=3
exec /home/bot/.venvs/torch-nightly/bin/python -m server.app \
  --host 127.0.0.1 --port 8300 --capacity 1 --num-slots 1 \
  --blocks-per-slot 2048 --qwen-kv-mode legacy --mtp --mtp-k 3
```

### Call it

```bash
# OpenAI chat completions (works with any OpenAI SDK)
curl http://127.0.0.1:8300/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 512
}'

# Anthropic messages
curl http://127.0.0.1:8300/v1/messages -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8", "max_tokens": 1024,
  "messages": [{"role": "user", "content": "Hello"}]
}'

# OpenAI Responses (Codex CLI shape)
curl http://127.0.0.1:8300/v1/responses -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8", "input": "Hello", "reasoning": {"effort": "medium"}
}'
```

### Use it as an agent backend

Any OpenAI-compatible client works point-and-shoot. For opencode, drop this as
`opencode.json` in your project:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "qwen38-local": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8300/v1" },
      "models": { "qwen3.8": { "name": "Qwen3.8 27B NVFP4 (local)" } }
    }
  },
  "model": "qwen38-local/qwen3.8"
}
```

### Reproduce the benchmarks

```bash
benchmarks/server_perf_grid.py --base-url http://127.0.0.1:8300 \
  --model qwen3.8 --contexts 128k --concurrency 4 --max-tokens 256 \
  --tokenizer-path /path/to/Qwen3.8-tokenizer \
  --out result.json
```

The grid reports cold/warm waves, per-request decode rates, aggregate
throughput, speculative acceptance histograms, and completion SHA-256 so runs
are comparable bit-for-bit.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
