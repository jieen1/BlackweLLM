# Qwen3.8 Gittensor ModelOpt W4A4 激活量化 A/B

日期：2026-08-23
范围：只针对 `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` 的 ModelOpt
W4A4 路径；Unsloth mixed-precision 路径未改变。所有服务实验使用
`/home/bot/.venvs/torch-nightly`、FP8 KV、CUDA Graph、prefix cache、DSpark
K=7、128K context、隔离端口 18424；8300 未触碰。

## 结论

保留并启用一个可回滚的优化：
`QSR_QWEN36_MODEL_OPT_FP4_QUANT=flashinfer` 使用 FlashInfer 的 SM120
CuTe-DSL NVFP4 activation quantizer；现在它是 ModelOpt W4A4 默认，显式设为
`local` 即可回滚。真实服务 A/B 有约
1.7–2.0% 的方向性 decode 提升，DSpark 接受统计和 completion SHA 保持一致，
但量化出来的 FP4 code 并非逐字节相同，因此保留 `local` 回滚并继续观察更完整
质量套件；本次服务的接受率与输出 SHA 已通过。

## 微基准

真实量化形状 `K=5120` 的稳态 CUDA event 时间（ms）：

| M | 当前 Triton | FlashInfer CuTe-DSL |
|---:|---:|---:|
| 1 | 0.00586 | 0.00446 |
| 4 | 0.00595 | 0.00438 |
| 7 | 0.00586 | 0.00435 |
| 16 | 0.00803 | 0.00432 |
| 32 | 0.00800 | 0.00437 |

在合成的 `ModelOptNVFP4W4A4Linear(K=5120,N=5120)` A/B 中，融合量化器后的
线性层时间为：

| M | local (ms) | FlashInfer (ms) |
|---:|---:|---:|
| 1 | 0.24471 | 0.18695 |
| 4 | 0.19710 | 0.11582 |
| 7 | 0.23084 | 0.12091 |
| 16 | 0.22482 | 0.11755 |

该合成测试的全一 scale 输出完全一致；真实随机激活的 FP4 code 会有差异，
所以不能把合成结果当成质量证明。

## 真实服务结果

同一 Gittensor checkpoint、同一 DSpark/FP8-KV/CG/prefix-cache 配置：

| workload | local | FlashInfer | 变化 |
|---|---:|---:|---:|
| c=1 decode | 216.17 tok/s | 220.51 tok/s | +2.01% |
| c=4 warm decode（repeat） | 138.33 tok/s/req | 140.63 tok/s/req | +1.66% |
| c=4 warm aggregate（repeat） | 485.715 tok/s | 493.755 tok/s | +1.66% |

c=1 两边均为 DSpark `226/255` accepted、平均 accepted `6.647059`，
completion SHA 相同；c=4 repeat 两边 acceptance histogram 也相同（`908/1020`）。
c=4 的两次 warm wave 仍有机器抖动，因此结论只写成小幅、方向一致的收益，
不宣称微基准中的 24–48% 会端到端复现。

为避免直接肉眼比较，已把 c=4 repeat 的平均值写入隔离 bfdiag store 并运行
`bf diff --json`：

- local：`f10d6dd255b6`
- FlashInfer：`0193de2ade27`
- verdict：`comparable=true`；唯一目标配置差异为
  `extra.activation_quantizer`（另有产物标签差异），GPU SM clock 为
  487→502 MHz 的运行时差异；decode `+1.635%`，aggregate E2E `+1.628%`。

原始产物：

- `/tmp/qwen38_modelopt_fp4_gittensor_ab/server_perf_grid_qwen38_dynamic_128k_c1_modelopt_fp4_local_gittensor_c1.json`
- `/tmp/qwen38_modelopt_fp4_gittensor_ab/server_perf_grid_qwen38_dynamic_128k_c1_modelopt_fp4_flashinfer_gittensor_c1.json`
- `/tmp/qwen38_modelopt_fp4_gittensor_ab/server_perf_grid_qwen38_dynamic_128k_c4_modelopt_fp4_local_gittensor_c4_repeat2.json`
- `/tmp/qwen38_modelopt_fp4_gittensor_ab/server_perf_grid_qwen38_dynamic_128k_c4_modelopt_fp4_flashinfer_gittensor_c4_repeat.json`

## 已排除的方向

- SGLang 的 GDN `target_verify` 融合思想在当前生产 DSpark 路径已经由 b12x
  `multistep/indexed_gated` 一次 launch 覆盖；此前新增的双 CUDA stream A/B
  只有约 `-0.15%`，已撤回。
- 小行 KV scatter 的单 kernel 差异只有 0–7%，服务级无净收益，已撤回。
- 当前安装的 FlashInfer `0.6.16.post3` 在 SM120 上拒绝 `mm_fp4` 的
  `backend="cute-dsl"`；`cutlass` 会首次编译大量 SM120 变体，孤立试探超过
  7 分钟仍未进入稳态测量，已停止。没有把不可用或未经证明更快的 GEMM 后端
  接入生产。
- b12x 与 FlashInfer 的同 ABI GEMM 在已测真实形状上处于同一性能级别；换 API
  本身不是优化。

## 代码开关

- 实现：`runtime/model/modelopt_linear.py`
- 回归测试：`tests/test_loading_modelopt.py`
- 配置说明：`server/README.md`
- `flashinfer` 是默认路径；`local` 是回滚路径。该开关只影响 ModelOpt W4A4，
  不影响 Unsloth。
