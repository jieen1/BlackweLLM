# Qwen3.8 128K decode：Phase 0 轮级 + kernel-family 归因（2026-08-15）

状态：🟢 **实测完成**（规划 §6.1 P0-P 的第一件门禁）。在 P0 显存提交
（`c1f2f11`..`a1c0c89`）之上的 K=3 生产配置实测：轮级 phase 归因解释 ≥90%
round wall，nsys 给出 kernel-family 占比，规划的判别树据此落判。

## 配置

- 模型 `unsloth/Qwen3.8-27B-NVFP4`，qwen36 backend，MTP K=3，CUDA Graphs，
  FP8 KV，elastic 4160-bundle pool，capacity=4（与
  `scripts/run_qwen38_128k_decode_bench.sh` 记录的生产配置一致）。
- 负载：131072-token digit filler（与 2026-08-15 基线同内容），c1 + c4。
- 轮级：`QSR_PROFILE_ROUNDS=2`（含 CUDA event，**模式 2 是 benchmark
  perturbation**，只用于归因不用于吞吐对比）。
- kernel-family：`nsys launch --trace=cuda --cuda-graph-trace=node` 会话采集
  c4 稳态（prefix 命中后的 warm waves）。
- 产物：`/tmp/opencode/phase0/`（server 日志含逐轮 JSON、`nsys_k3.nsys-rep`、
  perf-grid/trace/stats JSON）。

## 轮级归因（中位数）

| | B1 | B4 |
|---|---:|---:|
| round wall | 39.18 ms | 58.42 ms |
| accept_gpu_wait（等 verify GPU） | 73.2% | 78.4% |
| 其余 host phase 合计 | ~8% | ~6% |
| **verify_gpu_ms** | **29.47（75.2%）** | **46.46（79.5%）** |
| sync_gpu_ms | 2.70（6.9%） | 3.37（5.8%） |
| draft_gpu_ms | 3.79（9.7%） | 4.64（7.9%） |
| **GPU 合计 / wall** | **91.8%** | **93.2%** |

B4 的 phase 总和解释 84.4% wall，CUDA event 解释 93.2%——满足规划"解释
≥90% round wall"的门禁。**没有 host 空洞**：host 侧仅 ~7%，消 host sync /
metadata / D2H 不是优先项。

## kernel-family 归因（c4 稳态全 GPU 时间占比）

| family | 占比 | 主要 kernel |
|---|---:|---|
| **attention（b12x paged，FP8 backbone）** | **35.4%** | PagedForwardKernel ×3104，1.19 ms/次 |
| attention（b12x paged，BF16 = MTP） | 5.6% | draft/sync 的 128K attention ×618 |
| **W8A8 FP8 GEMM（attn/GDN 投影）** | 21.6% + 2.5% quant | qsr_fp8_w8a8 CUTLASS SM120 ×36520 |
| **W4A4 NVFP4 MLP** | 8.4 + 5.8 + 0.6 = 14.8% | dense_gemm f4E2M1FN（gate/up/down） |
| rms_norm（reduce+tail+elementwise） | ~4.4% | reduce MeanOps fp32 ×44872 |
| 小 GEMM/gemv（MTP head fc、lm_head） | ~4.7% | wmma 32x32 / cublas gemv |
| GDN（recurrent + conv） | ~2.6% | fused_recurrent_gdn_multistep_indexed |
| KV scatter/copy（index_put/float8_copy） | ~1.5% | |

verify body（轮时的 79.5%）内部按此比例折算：attention ≈44%，W8A8 ≈27%，
W4A4 ≈18%，GDN ≈3-5%。SplitKV merge 未进前 40 名（forward 内联或占比小）。

## 规划 §6.1 判别树落判

1. **kernel sum / wall = 93.2% > 80%** → 按 GPU ms 排序优化 →
   **attention 是最大 family（41%）**，P1-A（b12x FA4 迁移：exact
   conditional rescale、packed f32x2、tile/stage/SplitKV 联合调优）方向证实。
2. **GPU 无明显空洞**（host ~7%）→ 不先做 host sync/metadata/D2H 项。
3. **sync_gpu 5.8% < 8–10% 门槛** → P1-R 的 ragged sync 分桶**排除**。
4. **GDN ~2.6%** → §6.4 的 GDN 融合项**不进入**本轮优先（维持 profile 触发门禁）。

## 接受率与 K sweep 状态

- 本次全部运行接受率 **100%**（832/832 轮 3/3 满接受；digit filler 内容
  完全可预测）。注意：基线 note 的 58.4% 是 `f9e4a29` 修复 dynamic MTP
  page-table 代际 bug **之前**的数据，与本批 P0 改动无关。
- K=1/2/3 content-matched sweep：**K=3 已完成**（本 note）；K=2/K=1 未跑。
  在 100% 接受率下 K 的取舍可由本 note 的 profile 数直接推算：draft 仅占
  轮时 7.9%（B4），K=3 的 tokens/round=4 vs K=1 的 2，单位 token 轮时
  K=3 ≈ 14.6 ms vs K=1 ≈ 27 ms——K=3 显著占优，sweep 的决策价值低，
  留作需要精确数字时再补。

## 性能对照（同条件、无 profile）

c4 warm decode：改动前（`f9e4a29` 时代 mapping_version_fix fixture）
66.3 tok/s/req；本批 P0 显存提交后 66.39/66.29 tok/s/req——**中性（Δ≈0）**，
符合"显存优化不动计算路径"的预期。c1 cold decode 101.09 tok/s，
TTFT 62.3 s（131K prefill）。
