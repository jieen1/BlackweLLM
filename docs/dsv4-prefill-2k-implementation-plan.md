# DSV4 单卡 prefill ≥2000 tok/s：架构与实施计划

> 状态：**待实施**
>
> 日期：2026-08-11
>
> 目标硬件：NVIDIA RTX PRO 6000 Blackwell Max-Q（SM120，96 GB，300 W）
>
> 目标模型：当前 DeepSeek-V4-Flash GGUF（routed experts 为 IQ2_XS）

本文是达到 2000 tok/s 的执行合同，不是对当前 129–133 tok/s 路径的小幅调优清单。
现状证据与已排除方案见
[`../notes/2026-08-10-dsv4-prefill-moe-kernel-deep-dive.md`](../notes/2026-08-10-dsv4-prefill-moe-kernel-deep-dive.md)。

## 1. 目标与验收

模型完成加载、JIT 和 prewarm，服务进入 ready 后，单请求、无 prefix-cache 命中的真实 prefill
必须满足：

- 4K、16K、128K prompt 的 median 均 `>= 2000 tok/s`；
- 每个长度至少 5 次，报告 median/p10/p90，p10 不低于 1800 tok/s；
- ready 后首请求不得承担 kernel JIT；JIT 只能计入启动阶段；
- 2 slots × 128K + decode CUDA Graph 仍能在 96 GB 内加载；
- kernel 结果与现有 IQ2_XS reference 对齐，端到端 worst cosine `>=0.99`、greedy `39/39`；
- long-context、prefix cache、slot reset/reuse 和 decode graph 不回归；
- 对比必须经过 `bf diff`，确保 prompt、clock、power、配置和代码版本可比。

以下不算完成：

- 把 129 tok/s 提到 150/200；
- 用 prefix full hit 计算“等效吞吐”；
- 用 warm daemon 代替 load-time/cold-process 验证；
- 只展示随机矩阵、单 expert 或伪造 route 分布的 microbenchmark；
- 降低 top-k、层数或质量门禁。

## 2. 当前根因

当前 [`runtime/backends/dsv4.py`](../runtime/backends/dsv4.py) 每 64 tokens 调用一次完整
43 层 `_forward`。每层依次执行 HC、attention、HC、MoE。prefill MoE 在
[`runtime/model/dsv4_model.py`](../runtime/model/dsv4_model.py) 把 token×top-6 展开为 M=1 routes，
每条 route 单独做 IQ2 解码和 dp4a。

2026-08-11 真实 GGUF/SM120 实测：

- `max_q_rows=64` 稳态 131.8–133.3 tok/s；
- M=32 profile：MoE 289/350 ms（83%），attention 16 ms，HC 14 ms；
- M=32 的 192 routes 触达 131 experts，平均仅 1.47 routes/expert；
- 手写 CUDA warp-row 只比 Triton dp4a 快 1.2×；
- activation prequant 全 43 层约 5.4 ms，不是瓶颈。

结论：当前 route-M1 IQ2 指令流已接近其现实上限。调 `BLOCK_COLS`、warp 数或把 chunk 从
64 调到 96/128 都没有 15× 空间。

## 3. 可行性与预算

### 3.1 计算量

仅 routed MoE：

```text
params/token = 43 × top6 × 3 matrices × 4096 × 2048
             = 6.493B parameters/token
ops/token    = 2 × params = 12.986 GOP/token
2000 tok/s   = 25.97 TOPS
```

该卡官方 dense FP4 峰值约 1755.7 TOPS；3511 TOPS 是使用 2:4 sparsity 的值。2K 目标不是
硬件计算峰值问题，而是当前小 M/高解码指令路径没有把工作组织成 Tensor Core 能高效处理的形状。

### 3.2 权重流量

routed expert packed 权重约 74.58 GiB。1024-token superchunk 若每层大致读一遍活跃 expert
权重，相当于约 78.2 MB/token；2000 tok/s 约需 156 GB/s，低于官方 1792 GB/s 带宽。

### 3.3 端到端预算

1024 tokens 在 2000 tok/s 下总预算为 512 ms，即 11.91 ms/layer：

| 模块 | 每层预算 | 43 层预算 |
|---|---:|---:|
| routed + shared MoE | ≤6.5 ms | ≤279.5 ms |
| attention | ≤3.5 ms | ≤150.5 ms |
| 两段 HC + norm | ≤1.4 ms | ≤60.2 ms |
| embed/head/route/checkpoint/余量 | — | ≤21.8 ms |
| 总计 | ≤11.91 ms | ≤512 ms |

每个模块都有独立 kill gate；不能用一个模块的 microbenchmark 掩盖另一个模块超预算。

## 4. 目标架构

```text
1024 tokens
  -> embedding
  -> for each of 43 layers:
       HC-attention pre + norm over 1024 rows
       causal attention in sequential 64/128-row tiles
       HC post over 1024 rows
       HC-FFN pre + norm over 1024 rows
       route all rows and group by expert
       exact IQ2_XS tile decode -> INT8 Tensor-Core grouped MoE
       stable top-6 reduction + HC post
  -> HC head + LM head for the final token only
```

### 4.1 Layer-major superchunk

把“64 tokens 完整跑 43 层”改成“1024 tokens 在一层内处理完，再进入下一层”。

强制要求：

- superchunk 默认 1024；尾块固定为 64/128/256/512/1024 buckets；
- HC、norm、route、MoE 对整个 superchunk 运行；
- attention 为保持因果状态，内部仍顺序推进 64/128-row tiles；
- compressor/indexer/KV 必须在每个 tile 后保持旧语义；
- prefix checkpoint 明确绑定 superchunk boundary；
- LM head 只计算最后一个有效 token；
- 所有 scratch 由 backend 持有，循环内不分配。

64 tokens 时平均只有 `64×6/256=1.5` routes/expert；1024 tokens 时平均为
`1024×6/256=24`，使一个 IQ2 weight tile 能被约 24 行 activation 复用。这是数量级提升的来源。

### 4.2 Exact IQ2_XS → INT8 Tensor-Core grouped MoE

权重继续以 IQ2_XS 常驻；禁止 resident W8A8、NVFP4 或 BF16 副本。SparkInfer 新增语义：

```text
source_format = gguf_iq2_xs
quant_mode    = iq2_w8a8
execution     = expert-grouped, fixed M buckets
```

每个 CTA：

1. 读取某 expert 的 IQ2_XS K×N tile；
2. 把 codebook magnitude/sign 精确解码为 int8，只落 shared/register；
3. activation 按 K=32 量化为 int8 + per-row scale；
4. 使用 SM120 `mma.sync` INT8×INT8→INT32 做 K32 partial MMA；
5. 每个 partial 应用 activation scale 和 IQ2 的 `d × (0.5+nibble) × 0.25`；
6. FP32 累加，FC1 同时计算 gate/up，接 SwiGLU，再运行 FC2；
7. 按 stable expert-id/top-slot 顺序归并。

IQ2_XS 的 nibble scale 覆盖连续 32 values，正好匹配 K32 partial。权重无需重新量化，差异仅来自
accumulation order。

SparkInfer 已有 grouped routing、caller-owned scratch、W4A8/MXFP8 和 SM120 inline-MMA 骨架，
但没有现成 IQ2/INT8 grouped kernel；这是新 kernel 项目，不是 API 接线。

### 4.3 HC 与 attention

MoE 之外，HC+attention 旧 profile 已约 30 ms/M32 chunk，也必须大批次化：

- HC pre/post 对 1024 rows 合并 launch、量化、mix 和 norm；
- attention 首版保持 64-row tiles，正确后单独 A/B 128 rows；
- 128 rows 预计比 64 rows 多约 0.38 GiB shared MLA scratch，必须在
  2-slot×128K+decode graph 的实际 resident 下测；
- Q8 projections 复用现有 Tensor Core 路径；
- 禁止 tile 内 allocation、host sync 和 shape JIT。

## 5. 强制依赖

### 硬件/工具链

- NVIDIA SM120，当前目标限定 RTX PRO 6000 Blackwell Max-Q；
- CUDA 工具链能编译并验证 SM120 `mma.sync`；
- 可编辑的 `/home/bot/project/sparkinfer` fork；
- SparkInfer 改动必须在独立 worktree 开发并通过其测试；
- 使用手写 inline PTX/CuTe 时可沿用当前基础；若依赖 CUTLASS builder，则必须单独验证
  CUTLASS DSL 4.6+，不能依据 `is_supported()` 版本门禁推断功能。

### 显存

- 不得产生 model-sized second representation；
- grouped MoE scratch 上限 256 MiB；
- 若需要 SoA repack，必须在 streaming loader 中替换 raw storage，不能同时常驻两份；
- 必须保持 2 slots×128K + decode graph 在 96 GB 内。

### 运行时

- device-side expert grouping；不得有 `counts.max().item()` 等 GPU→CPU 同步；
- 固定 buckets、启动期 prewarm、运行时无分配；
- decode B1/M=1 保持现有路径，large-M prefill 才使用新 recipe；
- trace 必须记录 selected recipe、bucket、fallback reason 和分阶段 GPU 时间；
- 任何 fallback 都让 2K 性能门禁失败，不能静默回旧 kernel。

## 6. 实施阶段

### Phase 0：冻结观测和基线

- 给 bfdiag DSV4 provider 增加 `prefill_profile` operation；
- 记录每层 MoE/attention/HC GPU 时间和 real route histogram；
- 固化 4K/16K/128K token fixtures；
- 同一配置复现 129–133 tok/s，profile 分项与 wall time 差异 <5%。

改动面：`bfdiag/daemon/`、`bfprobe/`、`docs/diagnostics-guide.md`。

### Phase 1：正确性优先的 layer-major scheduler

- 在 `runtime/backends/dsv4.py` 增加 `_prefill_superchunk_logits`；
- attention 支持 caller-owned tiled output；
- HC/MoE 增加 large-M 固定形状入口；
- 旧 IQ2 kernel 暂时作为 reference，不以速度验收；
- 验证 1/63/64/65/255/256/1023/1024/1025 tokens 的 logits 和所有层状态。

### Phase 2：单层 IQ2 Tensor-Core kill gate

SparkInfer 改动面：

- `b12x/moe/_shared/execution.py`：新增 `gguf_iq2_xs`/`iq2_w8a8`；
- 新增 `b12x/moe/_shared/kernels/iq2xs_w8a8.py`；
- 复用 device-side grouping 和 planned scratch；
- 首版直接读取 74-byte block；只有 profiler 证明 alignment 是瓶颈才做等大小 SoA。

真实 layer、真实 1024-token routes 必须满足：

- gate+up+SwiGLU+down+stable reduction p50 `<=6.5 ms`、p90 `<=7.0 ms`；
- 连续 100 replay 无 host sync、allocation、JIT；
- kernel cosine `>=0.9999`；
- scratch `<=256 MiB`。

direct-packed 和等大小 SoA 两轮都失败，则停止单卡 2K 承诺，转向新 checkpoint format、较小
模型或多 GPU；不得回到 route-M1 小修。

### Phase 3：HC/attention 预算闭环

- HC+norm `<=1.4 ms/layer`；
- attention `<=3.5 ms/layer`；
- 验证 64/128-row attention tile 的性能、状态和显存；
- 未同时达标不得进入全模型集成。

### Phase 4：全模型集成和预热

- backend 生产默认 superchunk=1024；
- prewarm 64/128/256/512/1024 buckets 和 gate/up/down 三种 shape；
- decode 继续使用旧路径；
- 1024-token full-model `<=512 ms`；任何 fallback 直接失败。

### Phase 5：服务发布门禁

按以下顺序运行：unit/kernel → real single-layer → fresh 4K → fresh 16K → fresh 128K →
2-slot/128K/decode graph/slot reuse。先 `bf diff`，失败先读 trace，不盲目重跑。

4K、16K、128K 任一 median `<2000 tok/s`，都视为未完成。

## 7. 正确性与测试

### 无权重单测

- tail buckets、absolute position；
- layer-major/chunk-major state transition 等价；
- stable route grouping/reduction；
- prefix full/partial hit、reset/reuse、rollback；
- torch-free collection，torch 测试使用 `pytest.importorskip()`。

### CUDA/kernel

- 512 grid codes、128 sign patterns、所有 scale nibbles；
- M=1/8/16/24/32/64，生产 K/N=4096/2048；
- gate/up dual、SwiGLU clamp、down、router weight；
- Nsight 确认实际发射 Tensor Core 指令；
- 100 replay 无 allocation/sync/JIT。

### 全模型

- 1–1025 边界长度及 4K/16K/128K；
- ratio 0/4/128、hash layers、普通 router layers；
- 1–2 slots、reset/reuse、decode graph；
- logits、greedy、long-context NaN、prefix checkpoint；
- `ruff check .`、torch-free pytest、full pytest、`git diff --check`。

## 8. 拒绝项与风险

| 项目 | 决策 |
|---|---|
| 继续优化 route-M1 dp4a | 拒绝；warp-row 仅再快 1.2×，离目标约 15× |
| resident W8A8/NVFP4 cache | 拒绝；单层约 6 GiB，96 GB 无法常驻 |
| 全局 dequant BF16 + cuBLAS | 拒绝；GiB 级中间写流量，已有实测更慢 |
| 改 top-k/层数 | 拒绝；改变模型语义 |
| exact K32 scale/累加 | 高风险；由 Phase 2 数值+性能 kill gate 控制 |
| attention causal state | 高风险；Phase 1 boundary parity 前置 |
| 2K 未达成 | 不降低门槛宣布成功；按超预算模块继续优化或更换约束 |

## 9. 当前过渡修正

2026-08-11 已把 server/bfdiag 的旧 `max_q_rows=32` 默认统一为实测最佳的 64，并补齐
provider/parser 一致性测试。它把稳态约 90–105 提到 131.8–133.3 tok/s，且 2-slot×128K
decode graph 捕获成功。

该改动只建立更好的实施基线，不计入 2K 目标完成度。

## 10. 外部依据

- [NVIDIA RTX PRO 6000 Blackwell Max-Q 规格](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/)
- [NVIDIA RTX Blackwell PRO GPU Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf)
- [CUTLASS integer sub-byte types](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/integer_subbyte.h)
- [CUTLASS mixed-dtype GEMM example](https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_mixed_dtype_gemm.cu)
