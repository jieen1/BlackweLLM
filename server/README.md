# Server

## Current local Flash-Next profile

The currently validated local application is Qwen3.8 Flash-Next, not the
historical Laguna controller or the older Qwen3.8 DSpark benchmark profile.
The exact Python 3.14 launch command, environment contract, health checks and
OpenCode/Windows setup live in
[`docs/qwen38-flash-next-ops.md`](../docs/qwen38-flash-next-ops.md).

For orientation, the live profile is:

- checkpoint `/home/bot/models/Qwen3.8-Flash-Next-NVFP4-RadixArk`;
- `flashnext` backend on `127.0.0.1:8300`;
- one production slot (`capacity=1`, `num_slots=1`), `block_size=128`,
  `blocks_per_slot=2048` (256K per-slot ceiling);
- legacy KV mode, MTP `K=3`, CUDA Graphs and persistent prefix cache enabled.

Do not use `scripts/blackwellm_ctl.sh` for this profile: that script is a
Laguna-oriented historical launcher and defaults to port `8100`.

The server exposes the fixed-slot engine (`server/engine.py`'s
`ServerEngine`, a continuous-batching wrapper around
`runtime.backends.laguna.LagunaBackend` / `runtime.backends.qwen36.Qwen36Backend`
/ `runtime.backends.flashnext.FlashNextBackend`) through OpenAI-, Anthropic-
and Responses-compatible interfaces. It has no vLLM runtime dependency; do not
add multi-model or multi-GPU routing here.

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
  lifecycle events carry monotonically increasing `sequence_number` values;
  successful requests end at `response.completed` and token-limited requests
  at `response.incomplete` (the Realtime-only `response.done` event is not
  emitted), with 15s idle keepalive comments.
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

Set via env (read at import) or `python -m server.app` flags. The table below
keeps the historical Qwen3.8 DSpark/DFlash2 deployment values for API
compatibility and rollback reference. The current Flash-Next service uses the
separate one-slot/MTP profile in [`docs/qwen38-flash-next-ops.md`](../docs/qwen38-flash-next-ops.md);
do not infer its live settings from the historical `Deployed` column.

| Env | Flag | Code default | Deployed | Meaning |
| --- | --- | --- | --- | --- |
| `QSR_SERVER_CAPACITY` | `--capacity` | 4 (Qwen), 1 (Laguna) | 4 | concurrent production slots |
| `QSR_SERVER_NUM_SLOTS` | `--num-slots` | 4 (Qwen), 2 (Laguna) | 4 | total physical slots |
| `QSR_SERVER_MODEL_PATH` | — | Laguna HF id | deployment-specific | checkpoint path or HF id |
| `QSR_SERVER_TOKENIZER_PATH` | — | unset | deployment-specific | required for a local Qwen GGUF target; compatible HF tokenizer directory |
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
| `QSR_QWEN36_GDN_BATCH_LARGE_PROJECTIONS` | — | `auto` | `auto` | batch GDN `qkvz`/`out_proj` over the DSpark verify window for native ModelOpt W4A4 (and qualified raw-FP8); `0` restores the sequential rollback path, `1` forces the experiment |
| `QSR_QWEN36_MODEL_OPT_FP4_QUANT` | — | `flashinfer` | `local` | ModelOpt W4A4 activation quantizer; default selects SGLang's SM120 CuTe-DSL implementation, while `local` is the rollback |
| `QSR_QWEN36_DSPARK_FAST_SLOT_MAPPING` | — | `1` | `1` | use host-known contiguous position bounds in DSpark cache mapping; `0` restores generic GPU min/max validation for A/B diagnostics |
| `QSR_SERVER_ENABLE_DFLASH2` | `--dflash2` | 0 | 0 | opt-in Qwen3.8 Q6 GGUF + DFlash2 path; CUDA Graph is mandatory |
| `QSR_SERVER_DFLASH2_DRAFT_MODEL` | `--dflash2-draft-model` | `/home/bot/models/Qwen3.8-27B-DFlash2` | local | DFlash2 draft directory or cached model id |
| `QSR_SERVER_DFLASH2_K` | `--dflash2-k` | 7 | 7 | DFlash2 proposal depth (`block_size=8` → 7 proposals) |
| `QSR_GGUF_DEQUANTIZE_WEIGHTS` | — | 0 | 0 for Q6+DFlash2 | resident BF16 cuBLAS weights; adds about 26.7 GiB in the measured capacity-1 process; set 1 for the resident-BF16 rollback |
| `QSR_GGUF_NATIVE_PREFILL_DEQUANT` | — | 0 | 1 for Q6+DFlash2 | transient BF16/cuBLAS only for genuine prefill batches (M≥32); DFlash2 M=8 verify remains packed and graph-safe; 512 MiB per-projection cap by default |
| `QSR_GGUF_TC_BLOCK_M` | — | `auto` | `auto` | packed Q6 tensor-core M tile: 8 for DFlash2 verify, 32 for ordinary small batches, Q5/Q6 widen to 64 for large prefill; numeric 8/16/32/64 values are A/B overrides |
| `QSR_GGUF_NATIVE_MMQ` | — | 0 | 0 | experimental SGLang-style Q6_K MMQ; only M=8 DFlash2 verify and wide MLP shapes; latest fresh A/B is about +1.4% over a two-sample baseline, so default TC stays unchanged |
| `QSR_GGUF_NATIVE_MMQ_Q5` | — | 0 | 0 | separate Q5_K MMQ experiment for Qwen3.8's dynamic Q6_K_XL file; latest fixed 4K+DFlash2 A/B was about 1.8% slower, so it remains disabled |
| `QSR_GGUF_NATIVE_MMQ_Q8` | — | 0 | 0 | separate Q8_0 MMQ experiment; disabled by default because the fixed 4K smoke fell to 16/31 DFlash2 acceptance and changed output SHA |
| `QSR_GGUF_NATIVE_MMQ_LM_HEAD` | — | 0 | 0 | separate Q8_0 vocabulary-head MMQ A/B; the isolated kernel is faster at N=248320 but the fresh end-to-end decode result was not a stable net gain, so it remains opt-in |
| `QSR_SERVER_ENABLE_DFLASH` | `--dflash` | 0 | 0 | DFlash speculative engine (Laguna) |
| `QSR_SERVER_KV_CACHE_DTYPE` | — | fp8_e4m3 (Qwen), auto (Laguna) | fp8_e4m3 | KV cache dtype |
| `QSR_QWEN_KV_MODE` | `--qwen-kv-mode` | elastic (Qwen), legacy (Laguna) | elastic | Qwen KV allocation mode |
| `QSR_QWEN_KV_POOL_BYTES` | `--qwen-kv-pool-bytes` | 19629342720 (Qwen) | 19629342720 | Qwen DSpark physical KV budget |
| `QSR_SERVER_GPU_MEM_UTIL` | — | 0.85 | 0.92 | `gpu_memory_utilization` |
| `QSR_SERVER_PRODUCTION` | — | 1 | 1 | production slot layout (vs. diagnostic layout) |
| `QSR_SERVED_MODEL_NAME` | — | `qwen3.8` for Qwen | `qwen3.8` | name(s) reported by `/v1/models` (space-separated list) |
| `QSR_DEFAULT_REASONING_EFFORT` | — | `medium` for native Qwen/Flash-Next | `medium` | in-memory Qwen template default; omitted requests keep request kwargs empty and explicit effort wins |
| `QSR_SERVER_REQUEST_TIMEOUT_S` | — | 600 | 0 | server-side request cap; 0 disables (long generations) |
| `QSR_DEBUG_REQUESTS` | — | 1 | 1 | log raw request/response (see **Raw I/O logging**); legacy alias `QSR_DEBUG_ANTHROPIC` |

CLI also accepts `--host` / `--port` (default `127.0.0.1:8000`).

### Qwen3.8 Flash-Next reasoning effort

The shipped Flash-Next tokenizer template is the authority for this model's
thinking controls. It accepts the native effort values `low`, `medium`, and
`xhigh`; `enable_thinking=false` disables the `<think>` block. It does not
implement a distinct native `high` or `max` level. The runtime therefore
normalizes common OpenAI/OpenCode aliases at the request boundary:

| Client value | Flash-Next template value |
| --- | --- |
| `minimal` | `low` |
| `low` | `low` |
| `medium` | `medium` |
| `high` | `xhigh` |
| `xhigh` | `xhigh` |
| `max` | `xhigh` |
| `none` / `enable_thinking=false` | thinking disabled |

For OpenCode 1.x, declare the model as reasoning-capable and add variants in
the provider's `models` map. The installed local configuration uses this
shape (the model-level default is `medium`):

```json
{
  "compaction": {
    "reserved": 32003
  },
  "provider": {
    "blackwellm": {
      "npm": "@ai-sdk/openai",
      "options": { "baseURL": "http://127.0.0.1:8300/v1" },
      "models": {
        "qwen3.8-flash-next": {
          "reasoning": true,
          "interleaved": { "field": "reasoning_content" },
          "options": { "reasoningEffort": "medium" },
          "variants": {
            "none": { "reasoningEffort": "none" },
            "minimal": { "reasoningEffort": "low" },
            "low": { "reasoningEffort": "low" },
            "medium": { "reasoningEffort": "medium" },
            "high": { "reasoningEffort": "xhigh" },
            "xhigh": { "reasoningEffort": "xhigh" },
            "max": { "reasoningEffort": "xhigh" }
          },
          "limit": {
            "context": 262144,
            "output": 32000
          }
        }
      }
    }
  }
}
```

Use OpenCode's `/variants` picker, `variant_cycle` keybind, or `opencode run
--variant <name>`. The AI SDK sends the selected setting to this server as
`reasoning_effort`; direct OpenAI-compatible callers can send that field (or
`reasoning.effort` on the Responses API) themselves.

### Qwen3.8 Flash-Next image input

Flash-Next prefill uses a quality-gated **1024-token chunk** by default.  The
generic `QSR_PREFILL_CHUNK` value is clamped to 1024 for this backend unless
`QSR_FLASHNEXT_ALLOW_UNSAFE_PREFILL_CHUNK=1` is explicitly set for a numerical
experiment; larger chunks currently change recurrent/MoE reduction order and
are not production-safe.

Flash-Next target verify batches the large GDN projections across the fixed
`K+1` verify rows by default for the validated `qwen4_exp` checkpoint
(`mamba_ssm_dtype=float32`).  This removes repeated M=1 GEMM launches without
changing the FP32 recurrent-state contract.  Use the following switches only
for an explicit A/B or rollback:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `QSR_FLASHNEXT_BATCH_GDN_PROJECTIONS` | `auto` | `auto` enables the validated qwen4_exp batched path; `0` restores the per-row compatibility path; `1` forces the path only for a qualified format or an explicit BF16 validation override. |
| `QSR_FLASHNEXT_ALLOW_BF16_BATCH_PROJECTIONS` | `0` | Allows `QSR_FLASHNEXT_BATCH_GDN_PROJECTIONS=1` for an unknown BF16 checkpoint. The validated qwen4_exp checkpoint does not need this escape hatch. |

The active contract is exposed as `_cuda_graph_dbg.gdn_projections` in
`/debug/stats` (for example `batched_bf16` or `per_row`).

Flash-Next accepts still images on all three multimodal-compatible endpoints:
OpenAI Chat Completions (`image_url`), Anthropic Messages (`image` with a URL
or base64 `source`), and Responses (`input_image`). Images are decoded and
resized on the CPU before patchification, then fused through the checkpoint's
Qwen3-VL vision tower with the sglang-compatible three-axis MRoPE positions.
Video blocks are rejected explicitly.

Vision is enabled by default for the Flash-Next checkpoint. Set these before
starting the server when tuning the quality/capacity trade-off:

| Variable | Default | Meaning |
| --- | ---: | --- |
| `QSR_FLASHNEXT_VISION` | `1` | load the BF16 vision tower (about 0.84 GiB) |
| `QSR_FLASHNEXT_IMAGE_MAX_PIXELS` | `1048576` | post-decode area budget (1 MP; hard ceiling 16 MP) |
| `QSR_FLASHNEXT_IMAGE_MIN_PIXELS` | `65536` | processor's minimum area budget |
| `QSR_FLASHNEXT_IMAGE_MAX_TOKENS` | `16384` | total merged visual-token cap per request |
| `QSR_FLASHNEXT_VISION_ATTN` | `sdpa` | vision attention implementation; `eager` is the fallback |

The request log reports source dimensions, resized dimensions, and visual token
count. Visual requests carry an authenticated image fingerprint through slot
admission and prefix snapshots, so an identical image prefix can be restored
without re-prefilling it. Their teacher-forced visual rows also feed the same
MTP proposal path (including CUDA Graph replay when captured); different images
are always treated as cache misses.

### Qwen3.8 Q6_K_XL + DFlash2

The native Qwen3.8 path accepts the local `unsloth/Qwen3.8-27B-GGUF`
`Qwen3.8-27B-UD-Q6_K_XL.gguf` checkpoint and the separate DFlash2 draft
directory. It keeps Q/K/V and the full-attention reduction in F32, uses the
native SM120 GGUF kernels, and captures target decode, DFlash2 draft, and
fixed/ragged verify CUDA Graphs. DFlash2 refuses startup if any required Graph
is disabled or fails to capture; the existing NVFP4 service remains unchanged
because this path is opt-in.

The server profile defaults this explicit Q6+DFlash2 path to compact packed
weights plus transient BF16 prefill dequantization. For each genuine
prefill-sized projection, one BF16 matrix is created, used by cuBLAS, and
released; DFlash2's M=8 eager warmups and all captured verify/decode graphs stay
on the packed tensor-core path. This avoids the resident model-sized BF16
cache while materially reducing Q6 TTFT. Set
`QSR_GGUF_NATIVE_PREFILL_DEQUANT=0` to keep the fully packed path, or set
`QSR_GGUF_DEQUANTIZE_WEIGHTS=1` to select the resident-BF16 rollback. The local
`.gguf` path and the Qwen3.8 Flash-Next NVFP4 checkpoint both select the Qwen3.8
`qwen3_coder` tool parser automatically; `QSR_TOOL_CALL_PARSER` remains an
explicit override.

For a controlled packed-path experiment, set `QSR_GGUF_NATIVE_MMQ=1` together
with the Q6 split/Q8 activation settings. The route is shape-gated and does not
change resident BF16 mode, prefill, M=1 decode, or the existing NVFP4 service.
`QSR_GGUF_NATIVE_MMQ_Q5=1` is a separate opt-in for the Q5 gate matrices inside
the same dynamic Q6_K_XL file; it is not a different checkpoint and is not part
of the recommended profile.
`QSR_GGUF_NATIVE_MMQ_Q8=1` is intentionally not part of the recommended profile
until its quality regression is fixed.
`QSR_GGUF_NATIVE_MMQ_LM_HEAD=1` is a narrower vocabulary-head experiment;
although the isolated `N=248320` kernel is faster, the fresh end-to-end Q6
decode A/B did not show a stable net gain, so it remains disabled by default.

Example (use an isolated process and the `torch-nightly` environment):

```bash
QSR_SERVER_MODEL_PATH=/home/bot/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K_XL.gguf \
QSR_SERVER_TOKENIZER_PATH=/home/bot/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/9c73e2daee1d0fd494ffbd1d8753f2174a953796 \
QSR_SERVER_ENABLE_DFLASH2=1 \
QSR_SERVER_DFLASH2_DRAFT_MODEL=/home/bot/models/Qwen3.8-27B-DFlash2 \
python -m server.app --dflash2
```

The DFlash2 loader expects a model directory (or its `config.json`), not the
binary `model.safetensors` file. Under the same isolated 4K/c=1/32-token
workload, the earlier shape-aware packed-TC-only path
(`QSR_GGUF_DEQUANTIZE_WEIGHTS=0`, `QSR_GGUF_TC_BLOCK_M=auto`) reached
**84.605 tok/s decode**, **5.444 s TTFT**, and **5.814 s/request** warm mean;
the previous fixed-M=8 compact baseline was **77.465 tok/s**, **15.338 s TTFT**,
and **15.741 s/request**. The current default adds transient prefill
dequantization: a fresh A/B measured warm TTFT **1.163 s** and wall
**1.519 s** (two warm samples), with decode in the same noise band at about
**87.7 tok/s**. Resident BF16 is an explicit rollback; its earlier
fresh-process result was **81.905 tok/s** with **3.71 s TTFT**. The existing
NVFP4+DSpark profile reached **232.045 tok/s** with **0.6365 s TTFT**. The
SGLang-derived change is shape-aware: forcing M=32 everywhere improved prefill
but reduced DFlash2 decode to about 68 tok/s, so the fixed M=8 verify tile is
preserved. All Q6 variants in this A/B had 28/31 DFlash2 acceptance and the
same completion SHA; the required Graph replays remained captured. The full
quality/long-context/concurrency benchmark is a separate gate; raw methodology
and caveats are in
[`notes/2026-08-20-qwen38-q6-dflash2-performance.md`](../notes/2026-08-20-qwen38-q6-dflash2-performance.md).

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

For Qwen's dynamic arena, the exact prompt boundary is checkpointed even when
the prompt is not 128-token aligned. A growing conversation restores the
longest common page prefix and its GDN/DSpark state, then computes only the
new suffix; the final partial page is copied on write before extension.

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
