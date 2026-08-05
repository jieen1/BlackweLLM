# Qwen3.6-27B Quality Suite Rerun (2026-08-05)

Status: **complete** (all phases run 2026-08-05; results and artifacts below)

## 0. 2026-08-05 decision update: MTP + MTP CUDA Graph ON

The rerun profile is **MTP on + CUDA Graph on** (project decision): the
historical 2026-07-21/22 quality numbers were served by a stack with MTP
speculative decoding (`--speculative-config '{"method": "mtp",
"num_speculative_tokens": 3}'` in `/home/bot/vllm_server/vllm_ctl.sh`), and
this rerun must prove the current self-built runtime end-to-end with the
equivalent machinery, not with speculative decoding silently disabled.

- `QSR_SERVER_ENABLE_MTP=1`, `QSR_SERVER_MTP_K=3` (historical K=3).
- MTP anchor/draft/sync/verify CUDA Graphs: captured automatically at load
  (`runtime/backends/qwen36_mtp.py`, `QSR_QWEN36_MTP_CUDA_GRAPH` default 1).
- `QSR_SERVER_ENABLE_CUDAGRAPH=1`: decode CUDA Graph also captured.
- Enabling decode CG adds one warmup-slot requirement
  (`num_slots >= capacity + 1`); every profile below uses `num_slots =
  capacity + 1`. `capacity` — the concurrency knob — is raised to what the
  GPU fits (see 4.1), and generation parameters are unchanged. Pool sizing
  and context length do not affect greedy outputs for prompts within length.
- **Context sized to the benchmark, not 256K** (2026-08-05 follow-up
  decision): suite-fast runs 32K slots (code 4096 + prompt ≈ 4.3K max),
  longctx runs 139264-token slots (128K needle + 2048 answer), mmlu runs
  64K slots. Concurrency is raised accordingly: suite-fast capacity 8,
  longctx capacity 4, mmlu capacity 8.
- `QSR_SERVER_REQUEST_TIMEOUT_S=0`: the 600s server-side cap (added
  `a1fac04`, 2026-07-22 19:46, after the historical quality runs) is disabled
  to match the historical server, which had no request timeout. Client-side
  harness timeouts remain at 3600s as a defensive floor only.

## 1. Purpose

Rerun the Qwen3.6-27B quality evidence recorded in
[README Quality validation](../README.md) and
[`notes/2026-07-22-quality-baseline-and-official-scores.md`](2026-07-22-quality-baseline-and-official-scores.md)
on the current build, served by **this repo's own runtime only** (`server.app`
with the `qwen36` backend). No external inference engine is involved.

## 2. Environment

- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB, SM120), single GPU
- Python: `/home/bot/.venvs/vllm/bin/python` (historic venv name; hosts the
  repo's editable install + evalplus/aiohttp/datasets; no vllm import in the
  production path)
- Model: `unsloth/Qwen3.6-27B-NVFP4`
  snapshot `ccdaab7e68af2409599b8949a8f2685703c9bae5`
- Registry resolution: `backend=qwen36` (verified 2026-08-05)
- Server port: 8300
- GPU lock: `/tmp/gpu_lock.sh` (owner `qwen36-quality-rerun`)

## 3. Scripts

| Path | Role |
|---|---|
| `scripts/run_qwen36_quality.sh` | Orchestrator: env-check / server start+stop / ours-suite / ours-humaneval / mmlu / compare / all |
| `benchmarks/quality_regression.py` | 4-dim suite (tool/agent/longctx/code), per-dim jsonl checkpoints, resumable |
| `benchmarks/quality_eval.py` | HumanEval+ 768 harness (restored), incremental jsonl output, `--evaluate` |
| `benchmarks/quality_merge.py` | Merge per-dim part reports |
| `benchmarks/official/mmlu_pro_eval.py` | MMLU-Pro 414, sharded + merge, per-shard jsonl checkpoints |
| `benchmarks/quality_compare.py` | Regression comparator (baseline vs candidate) |

### 3.1 Changes made for this rerun

- `quality_regression.py`: every dimension now writes each completed item to
  `<label>.work/*.jsonl` immediately (`_parallel(on_done=...)`), so an
  interrupted run resumes item-by-item instead of restarting the dimension.
- `quality_eval.py`: restored from git history (`dfd37bf^`) with the original
  defaults (`max_tokens=768`, `concurrency=16`); outputs are appended per
  completed problem, so interruption resumes without regenerating done tasks.
- `mmlu_pro_eval.py`: `--shards N / --shard-idx i` split the 414 stratified
  subset deterministically; each shard has its own `.shardN.jsonl`; `--merge`
  combines shards into the final report.
- `mmlu_pro_eval.py` (14:1x): stratified subset fixed to **largest-remainder
  apportionment** so it reproduces the historical 414-question subset exactly.
  The plain `int()` floor produced only 408 questions (math 46 vs 47,
  physics 44 vs 45, chemistry 38 vs 39, law 37 vs 38, other 31 vs 32,
  biology 24 vs 25) and is NOT comparable to the 84.54 baseline. Verified:
  the 414 question_ids now match `mmlu_pro_think_c4.jsonl` 100%
  (0 historical-only, 0 current-only).
- `run_qwen36_quality.sh`: orchestrates phases, uses historical server
  profiles, `QSR_TOOL_CALL_PARSER=qwen3_coder` (required by Qwen3.6 tool
  calling), waits with `curl --noproxy '*'` (system proxy otherwise breaks
  health checks), and releases the GPU lock after each phase.

## 4. Parameters (identical to the historical run)

### 4.1 Server profiles

| Phase | capacity | num_slots | block_size | blocks_per_slot | KV / slot | cudagraph | MTP (K) | prefix | KV dtype | GPU util |
|---|---|---|---|---|---|---|---|---|---|---|
| suite-fast (tool/agent/code + HumanEval 768) | 8 | 9 (CG warmup slot) | 16 | 2048 | 32K | on | on (K=3) | on | fp8_e4m3 | 0.85 |
| longctx (needles 8K..128K) | 4 | 5 (CG warmup slot) | 16 | 8704 | 139264 | on | on (K=3) | on | fp8_e4m3 | 0.85 |
| mmlu | 8 | 9 (CG warmup slot) | 16 | 4096 | 64K | on | on (K=3) | on | fp8_e4m3 | 0.85 |

Shared: `QSR_SERVER_PRODUCTION=1`, `SESSION_AFFINITY=0`, `DFLASH=0`, `MTP=1 (K=3)`,
`QSR_TOOL_CALL_PARSER=qwen3_coder`, `HF_HUB_OFFLINE=1`,
model name `qwen3.6`.

### 4.2 Harness parameters

| Benchmark | Parameters |
|---|---|
| 4-dim suite | dims `tool,agent,longctx,code`; concurrency 8 (longctx capped at 4); code `max_tokens=4096`; greedy |
| HumanEval 768 | evalplus HumanEval+ 164; greedy; `max_tokens=768`; concurrency 16; evalplus sanitize + evaluate |
| MMLU-Pro | 414 stratified; thinking; 5-shot CoT; greedy; `max_tokens=32768`; 4 shards × concurrency 2 (= 8 in flight) |

## 5. Parallelism & resume

- 4-dim suite: `tool/agent/code` run as independent processes against the
  32K/capacity-8 suite-fast server; `longctx` runs as its own phase against
  the dedicated 128K+ server. Per-dim `<label>.work/{tool,agent,longctx,
  code_raw}.jsonl` checkpoints; rerun the same command to resume.
- MMLU: 4 shard processes, each `.shardN.jsonl`, then `--merge`.
- HumanEval 768: appends each completed problem to `<label>.jsonl` + `.raw.jsonl`.
- Every phase is idempotent; `KEEP_SERVER=1` keeps the server up between phases.

## 6. Run timeline

2026-08-05 (label `qwen36_20260805_mtpcg`, port 8300, own runtime only):

| Time (UTC+8) | Event |
|---|---|
| 12:05–12:08 | First suite attempts without final harness fixes (partial logs) |
| 12:36 | MTP+CG probe server (`probe_server.log`): 4096-token HumanEval 46.5s vs eager 726.9s; 2+2 21.8s→4.1s |
| 12:52–13:05 | longctx 12/12 on 256K-pool MTP+CG server (all 12; backup kept as `longctx.256k-pool.bak.jsonl`) |
| 13:04–13:41 | suite-fast server (32K×9 slots, MTP K=3, decode+MTP CG): tool 20/20, agent 4/4, code 164/164 generated; code eval failed on evalplus filename bug |
| 13:42–13:44 | Fixed `_parse_evalplus` path (`x.jsonl` → `x_eval_results.json`); code eval: 0.921/0.884; merged suite |
| 13:45–13:59 | longctx rerun on dedicated 139264-token server (capacity 4): 8/12 regenerated (all 64K/128K), 4 short cases reused from earlier checkpoint; 12/12 = 1.000 |
| 14:00–14:06 | HumanEval+ 768 on suite-fast server: 164/164 generated (0.6 problems/s), eval 0.421/0.415; fixed same evalplus path bug in `quality_eval.py` |
| 14:07–14:20 | First MMLU attempt started on wrong 408-question subset (plain int floor); stopped after 37 questions (moved to `official/.partial408/`), fixed apportionment |
| 14:21–15:04 | MMLU-Pro 414 (104/104/103/103) on 64K×8 server: 4 shards × concurrency 2; merged 15:04 |
| 15:0x | Evidence probe: suite-fast server restarted, short requests, MTP CG replay counters captured from `/debug/stats` |

## 7. Results

Suite (`evalplus_results/quality/qwen36_20260805_mtpcg.json`, merged 13:44):

| Dimension | Result | n |
|---|---|---|
| tool | accuracy 1.000, name 1.000 | 20 |
| agent | final-answer 1.000, tool-invocation 1.000 | 4 |
| longctx | 1.000 (8K/32K/64K/128K all 1.000) | 12 |
| code (4096) | HumanEval 0.921, HumanEval+ 0.884 | 164 |

HumanEval+ 768 (README row):

| Metric | 2026-07-21 historical | 2026-08-05 current | delta |
|---|---|---|---|
| HumanEval base pass@1 | 0.445 (73/164) | 0.421 (69/164) | −2.4pp |
| HumanEval+ pass@1 | 0.433 (71/164) | 0.415 (68/164) | −1.8pp |

Task-level: 50 stable passes, 18 new passes, 20 regressions, 1 partial, 73 fail/fail
— within the ±3.9pp SE the historical note attaches to this row. Raw length
distributions are equivalent (median 2380 vs 2363 chars), so the delta is not
truncation; it is run-to-run greedy variance.

MMLU-Pro 414 (`evalplus_results/official/mmlu_pro_think_qwen36_20260805_mtpcg.json`):

| Metric | 2026-07-22 historical | 2026-08-05 current |
|---|---|---|
| accuracy | **84.54%** (414) | **84.54%** (414) |
| truncated | 0 | 1 |
| no_answer | 1 | 0 |
| methodology | 5-shot CoT, thinking, greedy, max_tokens 32768 | identical (same 414 question_ids, verified 100% overlap) |

Same stratified-414 subset as the historical baseline (largest-remainder
apportionment), same per-question parameters; the 84.54 result is a direct
repro.

## 8. Artifacts

All under the repo root; sha256 truncated to 16 hex chars:

| Artifact | Bytes | Lines | sha256 |
|---|---|---|---|
| `evalplus_results/quality/qwen36_20260805_mtpcg.json` (merged suite) | 2083 | – | `d2e682e6555657c3` |
| `evalplus_results/quality/qwen36_20260805_mtpcg.part.tool.json` | 266 | – | `9582a91a42d69f3d` |
| `evalplus_results/quality/qwen36_20260805_mtpcg.part.agent.json` | 1519 | – | `3abfd99726664f4a` |
| `evalplus_results/quality/qwen36_20260805_mtpcg.part.code.json` | 415 | – | `393f1d1b5d65da17` |
| `evalplus_results/quality/qwen36_20260805_mtpcg.part.longctx.json` | 357 | – | `dc075a91f28f9eb2` |
| `evalplus_results/quality/qwen36_20260805_mtpcg.work/longctx.jsonl` | 1410 | 12 | `6c8189eaa397b569` |
| `evalplus_results/quality/qwen36_20260805_mtpcg.work/longctx.256k-pool.bak.jsonl` | 1410 | 12 | `f4f5a05f5d68b5fe` |
| `evalplus_results/quality/qwen36_20260805_mtpcg.work/code_raw.jsonl` | 2065395 | 164 | `4abe2e4058f8b4b9` |
| `evalplus_results/humaneval/our_runtime_qwen36_20260805_mtpcg.jsonl` | 42596 | 164 | `eeceda247f9fee17` |
| `evalplus_results/humaneval/our_runtime_qwen36_20260805_mtpcg.raw.jsonl` | 408288 | 164 | `a2340b7c831c61ee` |
| `evalplus_results/humaneval/our_runtime_qwen36_20260805_mtpcg_eval_results.json` | 61193 | – | `db6869044b21826d` |
| `evalplus_results/official/mmlu_pro_think_qwen36_20260805_mtpcg.json` | 1256 | – | `c59156ce8352eb00` |
| `evalplus_results/official/mmlu_pro_think_qwen36_20260805_mtpcg.shard{0..3}.jsonl` | 13869/13892/13752/13751 | 104/104/103/103 | `080e55a004c68c62` / `c1be0e1fcd6850cf` / `d5063c05eafb29ea` / `323c047ddc126db4` |
| `logs/quality/server_longctx_qwen36_20260805.log` | 3659 | 37 | `b43b0f705242c9ae` |
| `logs/quality/server_mmlu_qwen36_20260805_mtpcg.log` | 73496 | 849 | `7cf8806943ebc163` |
| `logs/quality/server_suite_qwen36_20260805_mtpcg.log` (13:04–14:06 run, overwritten by the 15:07 probe restart) | 30481 | 349 | `78e75f25d84cfc60` |
| `logs/quality/server_suite_qwen36_20260805_mtpcg.log` (15:07 evidence probe) | 2846 | 28 | `dcd5aa8f044f6b07` |
| `logs/quality/humaneval_qwen36_20260805_mtpcg.log` | 1615 | 33 | `f51d0c755112c3ff` |
| `logs/quality/mmlu_shard{0..3}_qwen36_20260805_mtpcg.log` | 1137×4 | 13×4 | `1ae966d526faff32` / `cb3d998d089af8b7` / `f4fef9763129e9c2` / `d08dcd29fd6b3928` |
| `/tmp/qwen36_mtpcg_stats.json` (MTP replay evidence, 15:10) | 2697 | – | `acb39fde849a9c4f` |

Engine-ready evidence (suite-fast, 13:15):
`capacity=8 num_slots=9 capacity_tokens_per_slot=32768 cudagraph=True prefix_cache=True ... mtp=True(K=3,resync=None)`.
MTP CUDA Graphs captured: `cg_status={'anchor': 'unused', 'draft': 'captured', 'sync': 'captured', 'verify': 'captured'}`.
Live replay counters (captured 15:10 after three short chat requests against a
fresh suite-fast server; `/tmp/qwen36_mtpcg_stats.json`):

```
_cuda_graph_dbg.decode = captured
_cuda_graph_dbg.mtp_draft / mtp_sync / mtp_verify = captured
_backend_stats_dbg.mtp_draft_graph_replays = 51
_backend_stats_dbg.mtp_verify_graph_replays = 48
_backend_stats_dbg.mtp_batched_sync_replays = 48
_backend_stats_dbg.decode_rounds / decode_graph_replays = 0
```

`decode_rounds/decode_graph_replays` stay 0 on this path because every decode
step is served by the MTP draft/sync/verify graphs; the decode CUDA Graph is
still captured (and `cudagraph=True` in the engine-ready line) as the
non-speculative path's graph. Longctx and MMLU servers show the identical
`cg_status` + `cudagraph=True` + `mtp=True(K=3)` engine-ready lines in their
logs (13:51:52 and 14:19:28).

## 9. How to rerun

```bash
bash scripts/run_qwen36_quality.sh env-check
bash scripts/run_qwen36_quality.sh all          # suite -> humaneval 768 -> mmlu -> compare
# or phase by phase:
bash scripts/run_qwen36_quality.sh ours-suite
bash scripts/run_qwen36_quality.sh ours-longctx
bash scripts/run_qwen36_quality.sh ours-humaneval
bash scripts/run_qwen36_quality.sh mmlu
bash scripts/run_qwen36_quality.sh compare
```

Override the output label with `RUN_LABEL=...`; restart after interruption with
the same label to resume from checkpoints.
