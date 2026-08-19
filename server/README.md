# Server

The server exposes the fixed-slot engine (`server/engine.py`'s
`ServerEngine`, a continuous-batching wrapper around
`runtime.backends.laguna.LagunaBackend` / `runtime.backends.qwen36.Qwen36Backend`)
through OpenAI-, Anthropic- and Responses-compatible interfaces. It has no
vLLM runtime dependency; do not add multi-model or multi-GPU routing here.

## Endpoints

- `POST /v1/chat/completions` — OpenAI chat. Streaming (`stream=true`,
  SSE) and non-streaming. Thinking is streamed as `reasoning_content`
  deltas (vLLM-compatible).
- `POST /v1/completions` — OpenAI text completion (non-streaming).
- `POST /v1/messages` — Anthropic Messages API (Claude Desktop). Streaming
  (`message_start` / `content_block_*` / `message_delta` / `message_stop`
  SSE) and non-streaming. Thinking is emitted as a `thinking` content block.
- `POST /v1/messages/count_tokens` — Anthropic token counting (Claude
  Desktop calls this before sending).
- `POST /v1/responses` — OpenAI Responses API (used by Codex CLI). Streaming
  lifecycle events (`response.created` / `in_progress` / `output_text.delta` /
  `completed` / `done`) and non-streaming, with 15s idle keepalive comments.
- `GET /v1/models` — model card; `max_model_len` reports the live per-slot
  context ceiling (`capacity_tokens_per_slot`).
- `GET /metrics` — Prometheus exposition in the `blackwellm:*` namespace.
- `GET /health` — liveness + slot occupancy.
- `GET /debug/stats` — the engine's admission/round counters plus the P0
  prompt-prefix-overlap, P4a prefix-cache hit-rate, and P4b session-affinity
  instrumentation.

Decoding is greedy (DSpark/MTP verify requires a greedy match). `n != 1` is a
clean 400. A request whose `prompt + max_tokens + K` would exceed the per-slot
capacity is rejected with a clean 400 BEFORE it reaches the runtime (this is
what keeps the server from triggering the known whole-batch attention crash).
Non-streaming responses carry non-standard `debug_committed_token_ids` /
`debug_prompt_token_ids`, used solely by `benchmarks/server_e2e_check.py`
(real clients ignore them).

## Configuration

Set via env (read at import) or `python -m server.app` flags. The current
Qwen deployed profile is DSpark K=7 with four 256K logical slots, block size
128, elastic FP8 KV and persistent prefix cache; the MTP values below are
kept as historical/rollback settings.

| Env | Flag | Code default | Deployed | Meaning |
| --- | --- | --- | --- | --- |
| `QSR_SERVER_CAPACITY` | `--capacity` | 4 (Qwen), 1 (Laguna) | 4 | concurrent production slots |
| `QSR_SERVER_NUM_SLOTS` | `--num-slots` | 4 (Qwen), 2 (Laguna) | 4 | total physical slots |
| `QSR_SERVER_BLOCK_SIZE` | — | 128 (Qwen), 64 (Laguna) | 128 | KV block size (tokens/block); Laguna sparkinfer requires 64 |
| `QSR_SERVER_BLOCKS_PER_SLOT` | `--blocks-per-slot` | 2048 | 2048 | per-slot KV ceiling (`× block_size` tokens) ⇒ **256K** for Qwen, **128K** for Laguna |
| `QSR_SERVER_ENABLE_CUDAGRAPH` | `--no-cudagraph` | 1 | 1 | captured decode graph |
| `QSR_SERVER_ENABLE_PREFIX_CACHE` | `--no-prefix-cache` | 1 | 1 | persistent prefix cache (P4a) |
| `QSR_SERVER_ENABLE_SESSION_AFFINITY` | `--session-affinity` | 0 | 0 | opt-in warm-slot retention (P4b) |
| `QSR_SERVER_SESSION_TTL_S` | `--session-ttl-s` | 30.0 | 30.0 | warm-slot retention TTL seconds (P4b) |
| `QSR_SERVER_ENABLE_MTP` | `--mtp` | 0 | 0 | explicit native-MTP rollback path |
| `QSR_SERVER_MTP_K` | `--mtp-k` | 4 | 3 | MTP speculative depth (historical K=3) |
| `QSR_SERVER_ENABLE_DSPARK` | `--dspark` | 1 (Qwen), 0 (Laguna) | 1 | default Qwen speculative decoding path |
| `QSR_SERVER_DSPARK_K` | `--dspark-k` | 7 | 7 | DSpark draft depth |
| `QSR_SERVER_ENABLE_DFLASH` | `--dflash` | 0 | 0 | DFlash speculative engine (Laguna) |
| `QSR_SERVER_KV_CACHE_DTYPE` | — | fp8_e4m3 (Qwen), auto (Laguna) | fp8_e4m3 | KV cache dtype |
| `QSR_QWEN_KV_MODE` | `--qwen-kv-mode` | elastic (Qwen), legacy (Laguna) | elastic | Qwen KV allocation mode |
| `QSR_QWEN_KV_POOL_BYTES` | `--qwen-kv-pool-bytes` | 19629342720 (Qwen) | 19629342720 | Qwen DSpark physical KV budget |
| `QSR_SERVER_GPU_MEM_UTIL` | — | 0.85 | 0.92 | `gpu_memory_utilization` |
| `QSR_SERVER_PRODUCTION` | — | 1 | 1 | production slot layout (vs. diagnostic layout) |
| `QSR_SERVED_MODEL_NAME` | — | engine MODEL | `qwen3.6` | name(s) reported by `/v1/models` (space-separated list) |
| `QSR_SERVER_REQUEST_TIMEOUT_S` | — | 600 | 0 | server-side request cap; 0 disables (long generations) |
| `QSR_DEBUG_REQUESTS` | — | 1 | 1 | log raw request/response (see **Raw I/O logging**); legacy alias `QSR_DEBUG_ANTHROPIC` |

CLI also accepts `--host` / `--port` (default `127.0.0.1:8000`).

### Long context (256K)

`blocks_per_slot` is the per-slot KV capacity CEILING
(`blocks_per_slot * block_size` tokens); `capacity_ok` rejects any request
whose `prompt + max_tokens + K` would exceed it BEFORE it reaches the
runtime. The default **16384 ⇒ 262144-token (256K) ceiling**. The KV cache is
a single shared `BlockPool` of ~40000 blocks (sized in `engine.py`
`_load_model` to fit the GPU with headroom for activations/GDN snapshots);
blocks are allocated FROM that pool ON DEMAND per slot, not reserved up front,
so idle `blackwellm:kv_cache_used_blocks` is 0. Two simultaneous 256K requests
(2 × 16384 = 32768 blocks) fit within the pool with spare for the prefix
cache. The previous deployed value (4200 ⇒ 67200) wrongly rejected real
long-context requests (e.g. `prompt 25843 + max_tokens 64000`); see
`tests/test_format_regression.py`.

### Prefix cache (P4a)

Default **ON** — this is the product value: the server serves warm prefix
hits across requests via the P0→P3 persistent content-addressed cache
(`enable_block_table` + `enable_prefix_cache` +
`enable_persistent_prefix_cache` on the runner). Turn 2+ of a growing
conversation, or unrelated requests sharing a system prompt / codebase
bundle, skip re-prefilling the cached prefix. The hit depth `L` and the
prefill tokens saved are reported by `/debug/stats` under
`prefix_cache_hits` / `prefix_cache_misses` / `prefix_cache_hit_rate` /
`prefix_cache_hit_L_samples` / `prefix_cache_hit_tokens_saved`, and the
counters are exported as `blackwellm:prefix_cache_hits_total` /
`blackwellm:prefix_cache_misses_total` / `blackwellm:prefix_cache_hit_rate`.

`--no-prefix-cache` (or `QSR_SERVER_ENABLE_PREFIX_CACHE=0`) rolls back to
the pre-P4a server **byte-for-byte**. `_finish_request` still does an
unconditional `reset_slot` in P4a — the content-hash cache survives reset by
design (R10).

### Session affinity (P4b)

Default **OFF** — opt-in via `--session-affinity` (or
`QSR_SERVER_ENABLE_SESSION_AFFINITY=1`). When a request carries a `session_id`,
the server retains the finished slot **warm** for `QSR_SERVER_SESSION_TTL_S`
(default 30.0s) so the next turn of the same session continues IN PLACE with
**zero restore** — it skips the GDN-checkpoint copy + block `touch` that a
content-hash hit (P4a) otherwise performs, via the runtime's gated
`mtp_prefill_warm_continue`. The content-hash cache (P4a) remains the
correctness-bearing fallback if the next turn's prompt does not reproduce the
retained slot's committed prefix exactly. Requires the prefix cache:
`--session-affinity` + `--no-prefix-cache` is refused at startup. Without a
`session_id` (or with the flag off) behavior is byte-for-byte P4a.
`/debug/stats` reports `session_warm_continuations`, `session_retentions`,
`session_expirations`, and `session_warm_fallbacks`; a warm turn advances
`session_warm_continuations` but NOT `prefix_cache_hits` (the definitive
zero-restore signal).

## Metrics

`GET /metrics` returns Prometheus text in the vLLM naming convention
(`blackwellm:*`, all labelled `model_name`), scraped by the local Prometheus
(legacy `~/vllm_server` docker `vllm-prometheus`, job `vllm`, 15s interval).

**Performance / speed** (the core focus; app-layer, recorded per request in
`server/metrics.py`, labelled by `endpoint` ∈ {`chat`, `completions`,
`messages`, `responses`}):

| Metric | Type | Meaning |
| --- | --- | --- |
| `blackwellm:e2e_request_latency_seconds` | histogram | request received → response complete |
| `blackwellm:time_to_first_token_seconds` | histogram | streaming time to first generated token (TTFT) |
| `blackwellm:request_time_per_output_token_seconds` | histogram | `(e2e − ttft) / (gen_tokens − 1)` (TPOT) |
| `blackwellm:request_prompt_tokens` | histogram | prompt-length distribution |
| `blackwellm:request_generation_tokens` | histogram | generation-length distribution |
| `blackwellm:prompt_tokens_total` | counter | total prompt tokens processed (throughput) |
| `blackwellm:generation_tokens_total` | counter | total generation tokens produced (throughput) |

**Stability / reliability:**

| Metric | Type | Meaning |
| --- | --- | --- |
| `blackwellm:num_requests_running` | gauge | requests currently generating |
| `blackwellm:num_requests_waiting` | gauge | requests queued for a slot |
| `blackwellm:num_free_slots` | gauge | free production slots |
| `blackwellm:request_success_total` | counter | successful requests by `endpoint` + `finish_reason` (`stop`/`length`/`tool_calls`) |
| `blackwellm:request_errors_total` | counter | rejected/failed requests by `endpoint` + status `code` (400 capacity/invalid, 500 internal) |
| `blackwellm:requests_completed_total` | counter | engine-level completed requests |
| `blackwellm:kv_cache_usage_perc` | gauge | KV cache utilisation (0–1) |
| `blackwellm:kv_cache_total_blocks` / `blackwellm:kv_cache_used_blocks` | gauge | KV block pool size / in use |
| `blackwellm:capacity_tokens_per_slot` | gauge | live per-slot context ceiling (256K = 262144) |

**Accuracy / correctness:**

| Metric | Type | Meaning |
| --- | --- | --- |
| `blackwellm:bootstrap_checks_ok_total` | counter | speculative prefills matching the independent reference prefill |
| `blackwellm:bootstrap_checks_failed_total` | counter | speculative prefills that DIVERGED from reference (non-zero = correctness problem) |
| `blackwellm:prefix_cache_hit_rate` | gauge | prefix-cache hit rate (warm-restart correctness + speed) |
| `blackwellm:prefix_cache_hits_total` / `blackwellm:prefix_cache_misses_total` | counter | prefix-cache hits / misses |

Useful PromQL:
- p50/p99 latency: `histogram_quantile(0.99, sum(rate(blackwellm:e2e_request_latency_seconds_bucket[5m])) by (le))`
- p99 TTFT: `histogram_quantile(0.99, sum(rate(blackwellm:time_to_first_token_seconds_bucket[5m])) by (le))`
- output throughput (tok/s): `sum(rate(blackwellm:generation_tokens_total[1m]))`
- error rate: `sum(rate(blackwellm:request_errors_total[5m])) / sum(rate(blackwellm:request_success_total[5m]) + rate(blackwellm:request_errors_total[5m]))`

## Raw I/O logging

With `QSR_DEBUG_REQUESTS=1` (default ON) every request logs, to the service
log (stdout of the server process; the quality runbook keeps it under
`logs/quality/server_*_<label>.log`):

- `<ENDPOINT> RAW REQUEST (N bytes): <full JSON body>` — the verbatim client
  request (OpenAI and Anthropic alike).
- `<ENDPOINT> PARSED MESSAGES: <parsed chat messages>` — what the format
  layer produced.
- `<ENDPOINT> DECODED PROMPT (N ids, M chars): <text>` — the EXACT input fed
  to the model.
- `<ENDPOINT> RAW OUTPUT (N tokens, finish=..., M chars): <text>` and
  `<ENDPOINT> VISIBLE OUTPUT (M chars): <text>` — the model's raw decoded
  output and the thinking-stripped visible text.

This is how format/length bugs are diagnosed (compare RAW REQUEST → PARSED →
DECODED PROMPT to see whether a user message survived). Set
`QSR_DEBUG_REQUESTS=0` to disable in production. When a real request exposes a
bug, capture its RAW REQUEST line into `tests/fixtures/` and add a case to
`tests/test_format_regression.py`.

## Validation

`python -m benchmarks.server_e2e_check` starts the real server (uvicorn +
genuine HTTP over a real socket) and verifies: a basic round-trip,
independent-reference-replay correctness, genuine concurrent batching,
defensive rejections, the P4a turn-1/turn-2 prefix-cache hit, and the P4b
session-affinity warm-continue. `tests/test_format_regression.py` (CPU-only,
runs in CI) locks the real captured request shapes and the 256K capacity fix.
