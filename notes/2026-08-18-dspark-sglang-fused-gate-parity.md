# Qwen3.8 DSpark vs SGLang: indexed GDN fused-gate parity (2026-08-18)

状态：🟡 **历史记录；cold prefill 问题已由 live-prefix publish 修复**

后续同口径结论与完整证据见
[`2026-08-19-qwen38-dspark-live-prefix-parity.md`](2026-08-19-qwen38-dspark-live-prefix-parity.md)。

## 口径

四路请求、每路 131072 prompt tokens、`max_tokens=256`、DSpark `K=7`、
CUDA Graph、prefix cache、FP8 KV、block/page size 128、Qwen3.8-27B-NVFP4，
prompt tokenizer 及 completion SHA 固定。所有结果来自完整 c4 fixture，不用
单请求或不同 prompt 的数字拼接。

SGLang 基准：`/tmp/sglang_dspark_same_128k_c4_20260818.json`，HEAD
`b296e1a503`，steady warm wall `2.609s`，即 `1024 / 2.609 = 392.49 tok/s`。
其本地 scraper 没有拿到 blackwellm 指标，所以只使用客户端 wall 和四路
completion SHA；不要把它的空 metrics 当成吞吐数据。

## 结果

| runtime / configuration | cold wall | warm 0 | warm 1 | completion SHA | DSpark stats |
|---|---:|---:|---:|---|---|
| local, vllm venv, fused gate=1 | 189.305s | 390.71 | 389.94 | 四路均 `75b43a8a...` | 411 rounds / 3060 committed |
| local, nightly, fused gate=1 | 185.645s | 442.91 | 445.46 | 四路均 `75b43a8a...` | 408 / 3060 |
| local, nightly, **default gate** | 171.630s | 425.12 | **428.66** | 四路均 `75b43a8a...` | 408 / 3060 |
| SGLang, nightly environment | 50.278s | 5.937 (JIT/overhead) | **392.49** | 四路均 `75b43a8a...` | external stats unavailable |

The default nightly run is therefore `+8.2%` on warm 0 and `+9.1%` on warm 1
against the SGLang steady value. The explicit-gate run is a faster repeat in
the same environment, but the default run is the acceptance result.

The production vllm environment is not a fair toolchain match to SGLang here:
it is Torch 2.13/CUDA 13.3/Triton 3.7/b12x 1.1.0, while SGLang and the accepted
comparison use Torch 2.15 nightly/CUDA 13.4/Triton 3.8/b12x 1.2.3. The vllm
run also admitted the warm wave as `[1, 3]`; the nightly run reached batch 4
for the steady wave. Keep these dimensions explicit in future comparisons.

## Change and verification

`runtime/model/qwen36_model.py` now defaults
`QSR_QWEN36_GDN_FUSED_GATES` to enabled through
`_qwen36_gdn_fused_gates_enabled()`. Setting the variable to `0` remains the
rollback/A-B switch. This fuses GDN gate sigmoid/softplus/exp math into the
indexed recurrence kernel, removing temporary gate launches from DSpark verify.

The default-switch regression is in
`tests/test_qwen36_gdn_spec_rollback.py` (13 passed); ruff passed for both
changed files. The benchmark artifacts are retained under
`benchmarks/fixtures/server_perf_grid_qwen38_dynamic_128k_c4_fused_gate_*` and
the matching `server_stats_*` / `server_trace_*` files.

## Historical remaining gap

At the time of this run, cold prefill was roughly `171.6s` local versus
`50.3s` SGLang. The live-prefix publish change recorded in the 2026-08-19
note reduced the same local cold run to `40.7986s`; this paragraph is retained
as the pre-fix baseline and should not be used as the current result.
