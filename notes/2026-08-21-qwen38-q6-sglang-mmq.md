# Qwen3.8 Q6_K：SGLang MMQ 学习与 SM120 适配结果

> 日期：2026-08-21
> 状态：🟢 Q6/Q5 形状限定实验路径已验证；默认生产路由仍保持 TC

## 结论

SGLang 的 Q6_K CUDA 路径分成两类：M=1 的 MMVQ，以及批量 verify/prefill 的
MMQ。DFlash2 的固定 verify 是 `M=8`，因此只移植 MMQ 有意义；MMVQ 不会跨
token 复用权重，不适合这个形状。

本地参考源码：

- `/home/bot/project/sglang/sgl-kernel/csrc/quantization/gguf/mmq.cuh`
- `/home/bot/project/sglang/sgl-kernel/csrc/quantization/gguf/mmvq.cuh`
- `/home/bot/project/sglang/sgl-kernel/csrc/quantization/gguf/vecdotq.cuh`
- `/home/bot/project/sglang/python/sglang/srt/layers/quantization/gguf.py`

适配的关键不是改成标准 Q6 block，而是保留 runtime 的 Q6_SPLIT row-tail
布局：每个 Q6 block 的 `ql/qh/scales` 为 208 bytes，所有 FP16 `d` 放在整行
payload 尾部，行 stride 仍为 `blocks * 210`。MMQ loader 因此单独计算 payload
和 d 的地址，不能直接 cast 成 SGLang 的 `block_q6_K*`。

## 路由与实现

新增 SM120 native ABI：

- `qsr_gguf_gemm_q8_mmq_sm120`
- Q8_1 activation + Q6_SPLIT weight + DP4A
- SGLang 同款 tile：`MMQ_X=4`、`MMQ_Y=32`、`4 warps`
- 输出仍写 runtime 使用的 row-major `[M, N]`

Python 路由额外限制为：

- `QSR_GGUF_NATIVE_MMQ=1` 显式开启；
- `M == 8`，只覆盖 DFlash2 verify CUDA Graph；
- Q6_SPLIT、Q8 activation 开启；
- 只覆盖宽 MLP gate/up：`N >= 16384 && N >= 2*K`；
- attention、GDN、MLP down、prefill、M=1 和 ragged tail 继续使用 BF16 TC/native
  fallback。

Qwen3.8 的 `Q6_K_XL` 是动态混合 K-quant 文件，不是所有矩阵都为 Q6：实际
`blk.0.ffn_gate.weight` 为 `Q5_K`，`ffn_up/down` 为 `Q6_K`。因此增加了独立
的 `QSR_GGUF_NATIVE_MMQ_Q5=1` 实验 selector，并复用了 SGLang 的 Q5_K
`MMQ_X=4/MMQ_Y=32/4 warps` 布局；它支持同一个目标文件中的 Q5 gate/up，
不是引入另一种模型。默认值保持 `0`，关闭时 Q5 仍走已经存在的标准 native
TC/GEMV 路径。

`M >= 8` 的早期版本会把 4K prefill 也送进 MMQ，TTFT 变慢约 0.3 秒，已经
收紧为 `M == 8`，并添加 selector 回归测试。

在保持 SGLang 的 `MMQ_X=4/MMQ_Y=32/4 warps` 后，又把同一 Q6 block 的两半
Q8_1 激活 tile 合并到一次 shared-memory load/barrier 中。真实
`N=34816,K=5120,M=8` 的 kernel 微基准从旧版 `0.2537 ms` 降到
`0.2421 ms`（约 `4.6%`）；这是 kernel 层收益，不能直接等同为端到端收益。

## 真实 SM120 结果

所有服务均使用隔离 `127.0.0.1:18380`，同一 torch-nightly venv、同一 Q6
checkpoint、同一 tokenizer、4K 数字 filler、并发 1、32 output tokens、
DFlash2 K=7、prefix cache off、cold + 2 warm。现有服务未重启。

| 配置 | warm decode 1 | warm decode 2 | 均值 | warm TTFT 均值 | 输出 SHA | 接受率 |
|---|---:|---:|---:|---:|---|---|
| 原生 Q6 split + TC（最新 fresh baseline） | 77.54 | 77.39 | 77.47 | 15.338 s | `34850d3f...` | 28/31 |
| M=8 Q6 SGLang-MMQ + TC（最新 fresh A/B） | 78.38 | 78.73 | 78.56 | 15.314 s | `34850d3f...` | 28/31 |
| M=8 Q5/Q6 SGLang-MMQ + TC | 75.24 | 76.93 | 76.09 | 15.876 s | `34850d3f...` | 28/31 |
| M=8 Q6 固定形状 fast path 候选（已回滚） | 77.66 | 76.85 | 77.26 | 15.603 s | `34850d3f...` | 28/31 |

最新 fresh A/B 中 Q6-only MMQ 的均值相对 baseline 约 `+1.4%`，但仍是低个位数
且只有两次 warm 样本，不能包装成稳定收益，更不能包装成 20% 级别提升。历史上
曾出现 `+1.7%`，随后融合版本复测只有 `+0.3%`，也支持同一个判断。Q5 开启后
均值反而比 baseline 低约 `1.8%`，所以 Q5 MMQ 不进入默认路径；输出 SHA 与
DFlash2 接受统计在这组固定 workload 上均未变化。

固定形状 fast path 的真实 kernel micro 在 `N=34816,K=5120,M=8` 上约有 `2%`
平均改善，但 Qwen3.8 实际是 Q5 gate + Q6 up 的混合格式，生产路由不会形成
一个 Q6 `N=34816` 合并矩阵；实际 `N=17408` micro 没有稳定收益，fresh endpoint
也低于同轮 baseline，因此该分支已回滚。

最新复测产物：`/tmp/qwen38_q6_mmq_fused_prod_4k32_perf_20260821.json`。

## 被否决的路线

- 全量 M>=8 MMQ：端到端 warm 为 `29.32`、`53.85 tok/s`，明显慢于 TC，且
  预填充代价更高。
- `BLOCK_N=64/128`：SM120 Triton TC synthetic sweep 明显退化；`BLOCK_N=32,
  4 warps` 仍是主配置。
- `QSR_GGUF_TC_STAGES=2`：真实端到端 warm 只有 `80.52`、`61.49 tok/s`，不
  采用；默认 stages=1 保持不变。
- MMVQ：SGLang 的 `GGML_CUDA_MMV_Y=1`，M=8 时不共享权重 tile，不值得移植。
- 固定形状 Q6 MMQ fast path：合并宽度 micro 有约 2% 的局部改善，但实际混合
  Q5/Q6 路由与端到端结果不支持维护额外分支，已回滚。

## Q5_K MMQ 实验：同一动态 Q6_K_XL 文件中的伴随格式

`Q5_K` 不是误接入的第五种模型格式。Unsloth 的动态 `Q6_K_XL` 文件会按张量
重要性混合 `Q5_K/Q6_K/Q8_0`；本地 header 检查得到 gate 是 Q5、up/down 是 Q6。
因此 Q5 ABI 和 selector 是为了完整覆盖这个文件，而不是把目标改成 Q5 模型。

Q5 native MMQ 已通过随机有效 Q5 payload 与现有 Q8 tile 的数值对照，真实 gate
层 micro 的最大绝对误差为 `0.0078125`、cosine 为 `1.0`；专项测试与 CUDA
构建通过。但固定 4K+DFlash2 fresh A/B 从 baseline `77.47 tok/s` 降到
`76.09 tok/s`，所以 `QSR_GGUF_NATIVE_MMQ_Q5` 继续保持显式关闭。

## Q8_0 MMQ 实验：保留代码但不进入默认路径

按 SGLang 的 MMQ 结构补了 Q8_0/Q8_0_SPLIT ABI，用于确认“所有 GGUF 都切
MMQ”是否值得。结果把它单独锁在 `QSR_GGUF_NATIVE_MMQ_Q8=1`，默认值为
`0`：

- 对齐的 Q8_0_SPLIT kernel 微基准约比现有 tile 快 `32%`；标准 34-byte
  Q8_0 行因 stride 未对齐，反而约慢 `8.7x`，所以不能直接照搬标准布局。
- 真实 4K + DFlash2 K=7 smoke 的接受数从基线 `28/31` 降到 `16/31`，输出
  SHA 从 `34850d3f...` 变成 `d620c1...`，因此质量门禁失败。

结论是：Q8 MMQ 的“算子快”不等于“模型可用”，当前只作为后续重新设计
activation/累加精度时的实验开关，Q6 的默认/显式安全路径不受影响。

## 质量与发布边界

本轮已经通过 native MMQ 与现有 Q8 tile 的数值回归，以及固定 4K 端到端
greedy SHA/接受率对照；尚未完成 prose/code、多 prompt、MMLU、长上下文的
正式质量集。因此 MMQ 目前是显式 opt-in 的实验优化，不改变没有环境变量时
的默认 TC 行为。

验证命令与产物：

- `make build-gguf-qk PYTHON=/home/bot/.venvs/torch-nightly/bin/python`
- `/home/bot/.venvs/torch-nightly/bin/python -m pytest -q tests/test_gguf_qk.py`
- `/home/bot/.venvs/torch-nightly/bin/ruff check .`
- `/tmp/qwen38_q6_mmq_m8only_4k32_perf_20260821.json`
- `/tmp/qwen38_q6_split_tc_cache0_4k32_perf_20260821.json`
- `/tmp/qwen38_q6_split_tc_cache1_4k32_perf_20260821.json`
- `/tmp/qwen38_q6_mmq_q6only_nobarrier_4k32_perf_20260821.json`
- `/tmp/qwen38_q6_q5mmq_4k32_perf_20260821.json`
- `/tmp/qwen38_q6_q5mmq_baseline_4k32_perf_20260821.json`
- `/tmp/qwen38_q6_mmq_fastshape_4k32_perf_20260821.json`（候选 fast path，已回滚）

## 2026-08-21 follow-up：checked MMQ 与同配置五次复测

为避免把旧的两次 warm 结果当成稳定收益，重新在同一隔离服务、同一
torch-nightly、同一 4K/32-token/DFlash2-K7 配置下复测五次。当前 Q6 MMQ
kernel 保留 SGLang 的 DP4A 算术和 `NeedCheck` 编译期分支，但在本机 SM120
上将 CTA 调回 `X=8/Y=32/8 warps`；SGLang 的字面 `X=4/Y=32/4 warps` 在这个
私有 Q6_SPLIT row-tail ABI 上更慢，不能机械照搬线程几何。

| 配置 | warm decode tok/s（5 次） | 均值 | 接受率 | 输出 SHA |
|---|---|---:|---:|---|
| 原生 Q6 split + TC | 75.23, 76.56, 74.31, 77.38, 75.88 | 75.87 | 28/31 | `34850d3f...` |
| Q6 MMQ，X=8/8 warps，checked/unchecked specialization | 61.30, 60.23, 59.72, 58.48, 59.74 | 59.89 | 28/31 | `34850d3f...` |

同一轮端到端结果显示当前 MMQ 仍比 TC 慢约 `21.0%`，所以继续保持显式
`QSR_GGUF_NATIVE_MMQ=1` opt-in，不进入默认生产路由。新增 M、N 都非整 tile
的 CUDA 回归（`M=5,N=37`）覆盖 SGLang 风格 `NeedCheck=true` 分支；整 tile
走 `NeedCheck=false`，消除无条件边界判断。该优化改善了实现质量，但没有在本机
的端到端五次复测中产生收益。

产物：

- `/tmp/qwen38_q6_sglang_x8w8_baseline_nocheck_w5_4k32_perf_20260821.json`
- `/tmp/qwen38_q6_sglang_x8w8_mmq_nocheck_w5_4k32_perf_20260821.json`
- `/tmp/qwen38_q6_sglang_x4w4_mmq_4k32_perf_20260821.json`（SGLang 字面几何）

## 2026-08-21 follow-up：整数 MMA 候选

另外对照了仓库里已有的 `QSR_GGUF_MMQ_MMA=1` 候选：它把 Q6 中心化字节和
Q8_1 激活送入 `mma.sync.aligned.m16n8k16.s32.s8.s8.s32`，以 128 行权重
tile 复用数据。相同五次 warm endpoint 结果为 `73.75, 75.35, 74.35,
76.11, 73.27 tok/s`，均值 `74.57 tok/s`，仍比同配置 TC 基线 `75.87 tok/s`
低约 `1.7%`，但比 DP4A-MMQ 的 `59.89 tok/s` 高约 `24.5%`。接受率仍为
`28/31`，输出 SHA 仍为 `34850d3f...`。

因此整数 MMA 候选值得保留为显式实验开关，但当前没有证据将它设为默认：它
没有稳定超过已经存在的 BF16 tensor-core Q6 路径，且与 DP4A oracle 的归约顺序
不同，尾 tile 会出现一个 BF16 ulp 级别的允许误差。回归测试对此使用更适合
整数 MMA 归约顺序的误差门限，并继续以真实端到端 SHA/接受率作为质量门禁。

产物：`/tmp/qwen38_q6_sglang_x8w8_mmq_mma_w5_4k32_perf_20260821.json`。
