# DSV4 单卡 prefill ≥2000 tok/s：架构与实施计划

> 状态：**可执行实施合同；Phase 2B-0 direct-folding 已触发性能 kill gate 并失败
> （9.97 ms gate+up vs 2.4 ms 目标，4.2×），I2F 前提被 Nsight 证伪；下一步按 §4.3
> 进入 two-plane candidate 或重新核算单卡 2K 预算**
>
> 日期：2026-08-11
>
> 目标硬件：NVIDIA RTX PRO 6000 Blackwell Max-Q（SM120，96 GB，300 W）
>
> 目标模型：当前 DeepSeek-V4-Flash GGUF（routed experts 为 IQ2_XS）

本文是达到 2000 tok/s 的执行合同，不是对当前 129–133 tok/s 路径的小幅调优清单。
现状证据与已排除方案见
[`../notes/2026-08-10-dsv4-prefill-moe-kernel-deep-dive.md`](../notes/2026-08-10-dsv4-prefill-moe-kernel-deep-dive.md)。

## 执行摘要：后续工程师从这里开始

### 不可改变的约束

- **只有一张** RTX PRO 6000 Blackwell Max-Q，96 GB，SM120，300 W；
- 保持 `TP=PP=EP=1`，多卡、换卡和远程 prefill 节点均不在本方案范围；
- 使用当前 DeepSeek-V4-Flash GGUF，routed experts 常驻格式为 IQ2_XS；
- 不降低 top-k、层数、上下文长度或质量门禁；
- 目标是无 prefix hit 的真实服务 prefill：4K、16K、128K median 均 `>=2000 tok/s`；
- 不能常驻第二份 W4/W8/BF16 routed weights，临时 scratch 上限 `256 MiB`。

任何后续方案若以“增加 GPU”作为达标条件，直接判为**越界**，不进入评审。

### 唯一实施顺序

```text
真实 GGUF 单层 route capture
  -> Phase 2B-0: IQ2 K-group 共尺度 Tensor-Core microkernel
       gate+up <=2.4 ms, down <=1.3 ms, cosine 达标
  -> Phase 2B-1: device-side route/group + fused SwiGLU/down/reduce
       router+routed+shared <=6.5 ms/layer
  -> Phase 1 production: 1024-token bounded layer-major superchunk
       full model <=512 ms/1024 tokens
  -> Phase 3: mHC/attention 按 SGLang DSV4 的融合和 overlap 结构收口
       HC <=1.4 ms/layer, attention <=3.5 ms/layer
  -> 4K -> 16K -> 128K 冷进程服务验收
```

顺序不能颠倒。当前 dormant superchunk 最终必须接入，但在新 MoE operator 通过
`6.5 ms/layer` 之前接 production 没有性能意义：它仍会调用
`Dsv4MoE._route_expanded_prefill()`，继续把 1024×top-6 展成 route-M1 dp4a。

### SGLang DSV4 怎么参考

SGLang 是本方案的**执行图和 operator 组织基线**，不是可以直接加载当前 GGUF 的现成后端。
调研固定基线为 `/home/bot/project/sglang` 的 `origin/main@2d193077f7`（2026-08-11）。

| SGLang 已验证结构 | 本项目的落点 | 迁移方式 |
|---|---|---|
| 4096-token chunked prefill；breakable/piecewise graph 的 1024-token bucket | `runtime/backends/dsv4.py` | 采用 1024 bounded superchunk 和固定 tail buckets；不复制 TP/DP 调度 |
| DSV4 attention 的 Q/KV、compressor、indexer 多流 overlap | `runtime/model/dsv4_attn_kernel.py`、`runtime/backends/dsv4.py` | 在单流正确性通过后增加 3 条预分配 stream 和显式 event 依赖 |
| `mhc_fused_post_pre` | `runtime/kernels/dsv4_mhc.py`、`runtime/model/dsv4_model.py` | 合并相邻层 post/pre、norm 和中间写回 |
| device-side MoE align/dispatch/combine | `runtime/model/dsv4_model.py` + 新 grouped operator | 复用固定 bucket、device counts/offsets、stable combine 的组织方式 |
| MegaMoE 的 dispatch + GEMM + activation + combine 融合边界 | 新 IQ2 grouped operator | 复用 operator 边界，不复制 MXFP4 算术 |
| prefill alt-stream 只在 breakable graph/capture 条件下开启 | DSV4 prefill recipe selector | 只作为 kernel 达标后的 overlap 优化，不能用于掩盖单 kernel 超预算 |

必须明确区分：SGLang 官方 RTX PRO 6000 路径是 MXFP4、`TP=2`；其
`flashinfer_mxfp4`/MegaMoE kernel 不能读取 74-byte/256-value IQ2_XS block。本项目直接复用的
是调度、融合、dispatch/combine 和 overlap 结构；唯一必须自建的核心是
**IQ2_XS on-the-fly Tensor-Core MoE**。

上游对照文件：

- `python/sglang/srt/models/deepseek_v4.py`：mHC、attention 多流、BCG prefill；
- `python/sglang/kernels/ops/moe/moe_align_small_numel.py`：单 launch device align；
- `test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py`：4096 chunk、breakable graph、
  1024 piecewise bucket 的组合门禁；
- commit `a0a76e4485`：BCG prefill 中允许 alt-stream 的最小改动。

### 2026-08-11 已完成的 Phase 2B 数值预筛
以下是本机单卡、真实 GGUF `blk.4.ffn_gate_exps.weight` expert 0、M=24、seed=20260811
的预筛结果。A/W 都在 K-group 内做对称 INT8 共尺度；它只证明候选值得写 CUDA kernel，
**不等于真实 route、完整 MoE 或端到端质量已经通过**。

| K group | gate projection `cos_min` | `cos_mean` | `rel_l2_max` |
|---:|---:|---:|---:|
| 32 | 0.9999694 | 0.9999718 | 0.00801 |
| 64 | 0.9999603 | 0.9999639 | 0.00898 |
| 128 | 0.9999508 | 0.9999552 | 0.00993 |
| 256 | 0.9999375 | 0.9999461 | 0.01118 |

同一 expert 的 gate+up→SwiGLU→down 误差会累积：K32/K32 的单 route
`cos_min=0.9998977`、K64/K32 为 `0.9998817`、K128/K128 为 `0.9998198`。
所以实现必须同时保留 K32/K64/K128 recipe，并在**真实 top-6 加权归并之后**选择；不能因为
单矩阵 K128 通过就把它写死为 production 默认。

### 2026-08-11 已完成的 Phase 2B-0 CUDA 实测（direct-folding）

表示证明已在真实 GGUF（blk.4，M=24）验证 K32 是满足 quality gate 的最大
K-group（全链路 down cos 0.9999）。`iq2_mma16_tc.cu` 实现 K-group=32
scale folding，数值 vs exact oracle 全部 `>=0.9999`。

**性能（E=256，M_PAD=32，真实 1024-token 路由）**：gate+up 9.97 ms、
down ~5 ms —— **kill gate（2.4/1.3 ms）未达（4.2×）**。

Nsight 证据：
- tensor core 仅 1.2%（指令），ALU 1.69G（decode 位操作）是最大单项；
- SM throughput 64.6%，tensor 12% —— 指令流水近饱和但 tensor 饿死；
- 纯 mma 下限（无 decode/staging）4.72 ms，仍超 2.4 ms kill gate 2×；
- K32 mma 减半 tensor 指令（100M→50M）但总时间反升（A 片段寄存器 + 发射率），回退；
- `__launch_bounds__` 压寄存器 → spill → 更慢，回退。

**结论**：§4.3 "I2F 串行依赖是主瓶颈" 前提被实测证伪——瓶颈是 decode 的
ALU 指令量和 L1TEX/smem 流量，不是 I2F。four 轮优化（spill 消除 → 直接
global decode → fused decode/fold → 模板 M_PAD）把 24 ms 降到 9.97 ms，
但**即使 decode 完美消除（5.89 ms 下限）也超 kill gate**。two-plane 预解码
codebook 预计 ~6-8 ms，仍超 2.4 ms。这是单卡 IQ2_XS Tensor-Core 路径的
现实下限，需重新核算 6.5 ms/layer MoE 预算或转向 §9 的 "停止 2K 承诺" 分支。

详见 `notes/2026-08-11-dsv4-phase2b-scale-amortized-falsified.md`。

### 下一个提交必须交付什么

下一位实现者不要先改 backend。第一个可合并提交只允许完成 Phase 2B-0，并必须同时提供：

1. 新的独立 candidate artifact，例如 `runtime/kernels/iq2_mma16_tc.{cu,py}`；现有
   `iq2_mma16` exact kernel 保持不变，继续作为 oracle；
2. K32/K64/K128 的真实 layer、真实权重、真实 route 数据；
3. gate、up、down 各自相对 exact oracle 的 cosine 和误差分位；
4. gate+up p50/p90、down p50/p90，100 次 replay；
5. Nsight 中的 MMA、I2F、integer decode、DRAM、occupancy 证据；
6. 运行期零 `.item()`、零分配、零 JIT，以及 scratch byte 数；
7. 明确结论：通过 kill gate 继续 Phase 2B-1，或失败后进入 two-plane candidate；不得接旧
   grouped prototype 凑端到端数字。

## 0. 2026-08-11 最新代码审计

代码审计基线为本地 `main@93aa25a`，审计时与 `origin/main` 一致；本文后续文档改动不改变该
代码基线。
从本文初版 `c24d93b` 到当前 HEAD，最新代码已经不是“完全待实施”：

| 阶段 | 已有事实 | 尚未完成 / 阻塞 |
|---|---|---|
| Phase 0 | `bfdiag/prefill_profile.py` 已实测 64 tokens 为 134.9–137 tok/s；MoE 约 71%，wall 与 profiler GPU 合计差 35.8% | 仍是独立脚本，不是 bfdiag provider operation；原定 `<5%` 分项闭合未达成 |
| Phase 1 | `_prefill_superchunk_logits` 已完成 64/65/256/1024 parity；cos 0.9972–1.0，greedy 5/5 | production 入口仍调用 `_prefill_logits`；当前 prototype 一次保留整个 prompt，不是 1024-row 有界 superchunk，也未覆盖 prefix hit/checkpoint/stats |
| Phase 2 exact | `iq2_mma16.cu` 已完成 exact IQ2_XS、dual gate/up、smem decode、4-way ILP；kernel cos 1.0、maxrel 约 1e-6 | 真实 1024-token 形状 gate+up 14.8 ms、grouped routed pipeline 32 ms（尚不含 router/shared expert），已经超过整层 router+routed+shared MoE 6.5 ms 预算约 4.9×；未接入 `Dsv4MoE.forward` |
| Phase 2 Python prototype | `grouped_moe_prefill` 已串起 group→gate/up→SwiGLU→down→reduce | 仍有 `counts.max().item()` host sync、运行时分配和 PyTorch scatter/sort；测试名声称 reference parity，但当前断言只检查 finite/nonzero |
| Phase 3–5 | 无 production 变更 | HC/attention 预算、全模型 512 ms、4K/16K/128K 服务门禁均未开始 |

因此当前 production 吞吐仍是约 130 tok/s，不能把“代码已存在”解释成“服务路径已提速”。
`0e3e10a` 的 exact kernel 是重要的正确性 oracle 和性能拆账工具，但它已经用实测证明：
**逐 K16 做 I2F+FFMA 的 exact 累加组织本身不能完成 2K 目标。**

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
| router + routed + shared MoE | ≤6.5 ms | ≤279.5 ms |
| attention | ≤3.5 ms | ≤150.5 ms |
| 两段 HC + norm | ≤1.4 ms | ≤60.2 ms |
| embed/head/checkpoint/余量 | — | ≤21.8 ms |
| 总计 | ≤11.91 ms | ≤512 ms |

每个模块都有独立 kill gate；不能用一个模块的 microbenchmark 掩盖另一个模块超预算。

## 4. 更新后的目标架构

```text
1024 tokens
  -> embedding
  -> for each of 43 layers:
       HC-attention pre + norm over 1024 rows
       causal attention in sequential 64/128-row tiles
       HC post over 1024 rows
       HC-FFN pre + norm over 1024 rows
       route all rows and group by expert
       IQ2_XS scale-amortized decode -> INT8 Tensor-Core grouped MoE
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

### 4.2 Exact kernel 的新定位：oracle，不是 production recipe

权重继续以 IQ2_XS 常驻；禁止 resident W8A8、NVFP4 或 BF16 副本。当前 exact kernel 每个
K16 partial 都把 INT32 转 FP32 并乘独立 scale。它已经达到 cos 1.0，但 E=256、6144 routes
形状下 gate+up 为 13.4–14.8 ms、grouped routed pipeline 为 32 ms；后者还没有包含 router
和 shared expert。即使 scheduler 零开销，也不可能满足 router+routed+shared MoE 的 6.5 ms/layer。

所以 exact kernel 保留用于：

- 新 recipe 的 single-layer 数值 oracle；
- 逐项定位 scale folding、route grouping 和 reduction 误差；
- 当性能门禁关闭时提供 debug fallback，但 fallback 运行不计入 2K 验收。

Phase 2B 的候选 recipe 接口暂定为；在 kernel gate 通过前，它不是 production 默认：

```text
source_format = gguf_iq2_xs
quant_mode    = iq2_tc_scale_amortized
execution     = expert-grouped, fixed M buckets, caller-owned scratch
```

### 4.3 Phase 2B：scale-amortized INT8 MMA 候选验证

新路径的核心不是放宽 2K 目标，而是消掉已证实的 I2F 串行依赖。以下算法均为
**[待验证]**，必须同时验证
`K_SCALE_GROUP={32,64,128,256}`，选择满足质量门禁的最大 group：

1. IQ2_XS 的 `d`、nibble scale、grid magnitude 和 sign 仍是唯一权重来源；
2. 把多个 K16 的 scale 因子折入 INT8 B fragment/codebook lookup，在 INT32 中累加一个
   K-scale group 后只做一次 I2F+FFMA；
3. activation scale 同步从 K32 向 K64/K128/K256 做 A/B，不能只优化权重侧而保留同量级
   的 per-K32 float dependency；
4. exact `iq2_mma16` 逐层比较，禁止用随机小矩阵替代真实 route 和真实权重；
5. direct-packed 先测；若 74-byte AoS decode/对齐仍超预算，loader 做**等驻留大小** SoA：
   64-byte code plane + 10-byte scale/meta plane，替换 raw storage，不同时常驻两份；
6. gate/up、SwiGLU、down 和 stable reduction 必须作为一个 fused/grouped budget 测量。

进入 CUDA 实现前必须先提交表示证明：INT8 fragment 的取值范围不会溢出；folding/two-plane
对 `d × (0.5+nibble) × magnitude × sign` 的重构公式明确；activation K-group 变大后的量化
误差有真实 layer 分布统计；MMA 数、I2F 数、decode 指令数给出可复核的每层下界。缺任一项，
该候选只算实验，不得称为 production recipe。

第一版 direct-folding 的明确公式如下；实现不得自行替换成另一种未记录的近似：

```text
delta_j = d * (0.5 + nibble_j) * 0.25       # IQ2 每 K16 权重 scale
sA      = max(abs(A[K-group])) / 127
sB      = 43 * max_j(abs(delta_j)) / 127    # IQ2 magnitude 只有 8/25/43
qA      = round(A / sA)
qB      = sign * round(magnitude * delta_j / sB)
acc32   = sum_{K-group}(qA * qB)
partial = float(acc32) * sA * sB
```

对 `K-group<=256`，最坏 INT32 partial 为
`127*127*256=4,129,024`，远小于 `INT32_MAX`；这只证明 accumulator 不溢出，不证明数值质量。
每个 K16 的 `{8,25,43}` 三档 q-magnitude 应只计算一次并供 codebook lookup 复用，禁止逐权重
执行 float scale。K32/K64/K128 必须编译成独立固定 recipe，运行时 selector 只选已 prewarm 的
artifact，不能把 group size 作为会触发重编译的动态参数。

优先实现两级数值方案：

- **TC64/TC128 scale folding**：首选，目标是在可接受量化误差下把 I2F 次数降低 4–8×；
- **two-plane integer decomposition**：若直接 folding 误差过大，把 IQ2 magnitude×scale-index
  分成两个 signed-INT8 fragment，在 INT32 合并后统一缩放；它增加 MMA 数量，但保留更多
  IQ2 数值信息，仍避免逐 K16 I2F。

这两条都保持原始 2-bit 级 resident footprint。W4A8/NVFP4 resident repack 仍然拒绝，因为
74.58 GiB routed weights 放大到 4-bit 后无法在 96 GB 卡上与 2×128K KV、decode graph 共存。

每个 grouped CTA/cluster：

1. 读取某 expert 的 IQ2_XS K×N tile；
2. codebook magnitude/sign 与 scale folding 只落 shared/register；
3. activation 使用选定的固定 K-scale group 量化；
4. 使用 SM120 `mma.sync` INT8×INT8→INT32，跨多个 K16 partial 保持 INT32；
5. 每个 K-scale group 只做一次 FP32 scale/accumulate；
6. FC1 dual gate/up 后立刻 SwiGLU，再进 down，避免 `[E,M,N]` FP32 全局中间张量；
7. 按 stable expert-id/top-slot 顺序归并。

### 4.4 device-side grouping 与固定 scratch

当前 `grouped_moe_prefill` 只是功能原型，不可直接接入生产。production grouping 必须：

- 固定 `{16,32,48,64}` M-tile buckets，1024 tokens 的 6144 routes 全在 device 上分桶；
- route count `>64` 的热点 expert 拆成多个固定 tile，不能截断；trace 单独报告 overflow 比例；
- 无 `.item()`、无动态 `torch.empty/zeros/arange/sort`、无 Python per-expert loop；
- backend 启动时一次性分配 `<=256 MiB` scratch，并 prewarm 所有 bucket；
- 直接写 gate/up 的 BF16 或 fused SwiGLU staging；禁止当前 `[E,M_PAD,N]` 双 FP32 输出；
- trace 记录 active experts、bucket distribution、padding ratio、recipe 和 fallback reason。

### 4.5 HC 与 attention

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
- SoA 不是 kernel 局部优化：必须增加显式 layout descriptor，并同步迁移 loader、dequant oracle、
  prefill kernel、decode kernel 和 fallback tests；decode 未保持当前性能/正确性前不得替换 AoS；
- 必须保持 2 slots×128K + decode graph 在 96 GB 内。

### 运行时

- device-side expert grouping；不得有 `counts.max().item()` 等 GPU→CPU 同步；
- 固定 buckets、启动期 prewarm、运行时无分配；
- decode B1/M=1 保持现有路径，large-M prefill 才使用新 recipe；
- trace 必须记录 selected recipe、bucket、fallback reason 和分阶段 GPU 时间；
- 任何 fallback 都让 2K 性能门禁失败，不能静默回旧 kernel。

## 6. 从当前 HEAD 继续的实施阶段

章节编号保留历史 Phase 名称，但**实际提交顺序**固定为
`Phase 0观测闭合 -> Phase 2B-0 -> Phase 2B-1 -> Phase 1 production接线 -> Phase 3 -> Phase 4 -> Phase 5`。

### 文件改动面与所有权

| 工作包 | 允许修改的主要文件 | 明确不做 |
|---|---|---|
| 2B-0 candidate kernel | 新建 `runtime/kernels/iq2_mma16_tc.cu`、`.py`、`.exports`；Makefile 独立 build target；新 CUDA tests | 不改 `dsv4.py`，不替换 exact artifact，不接 production |
| 2B-0 真实单层 gate | `bfdiag/daemon/provider.py` 或可复用 bfdiag operation、trace schema | 不在 `benchmarks/` 增加一次性脚本，不加载全模型只为取一层 |
| 2B-1 grouping/operator | `runtime/model/dsv4_model.py`；必要时新建 grouped scratch/operator 模块 | 不调用 `.item()`，不使用 Python per-expert loop，不输出双 FP32 大 staging |
| Phase 1 scheduler | `runtime/backends/dsv4.py`、prefix/checkpoint/trace 对应测试 | 新 MoE 未过 6.5ms gate 前不接默认生产入口 |
| Phase 3 mHC/attention | `runtime/kernels/dsv4_mhc.py`、`runtime/model/dsv4_attn_kernel.py`、`runtime/backends/dsv4.py` | 不先开多流再补同步；不改变 compressor/indexer 状态语义 |
| Phase 4/5 serving | server config、prewarm、bfdiag 冷测 operation、文档 | 不用 warm daemon 代替冷 prefill/显存验收 |

每个工作包独立提交、独立回滚。若同时修改候选 kernel、production scheduler 和 prefix state，
视为不可评审的大提交，必须拆分。

### Phase 0：冻结观测和基线 — **部分完成**

- 把现有 `bfdiag/prefill_profile.py` 接入 DSV4 provider operation；
- 记录每层 MoE/attention/HC GPU 时间和 real route histogram；
- 固化 4K/16K/128K token fixtures；
- 同一配置复现 129–133 tok/s，profile 分项与 wall time 差异 <5%。

改动面：`bfdiag/daemon/`、`bfprobe/`、`docs/diagnostics-guide.md`。

### Phase 1：有界 layer-major scheduler — **prototype 完成，production 未接线**

- 把现有 `_prefill_superchunk_logits` 改为 `(start_base, <=1024 rows)` 有界入口；
- outer prefill 逐 1024 rows 调用，attention 使用绝对位置 `start_base+tile_offset`；
- 补齐 prefix full/partial hit、256-token checkpoint、stats/trace、slot reset/reuse；
- attention 支持 caller-owned tiled output；
- HC/MoE 增加 large-M 固定形状入口；
- 旧 IQ2 kernel 暂时作为 reference，不以速度验收；
- 验证 1/63/64/65/255/256/1023/1024/1025 tokens 的 logits 和所有层状态。

### Phase 2A：exact IQ2 Tensor-Core — **数值完成，性能 kill gate 失败**

已完成：exact dual gate/up kernel、wrapper、build provenance、CUDA tests。实测 grouped routed
pipeline 32 ms（router/shared expert 未包含），不再继续用 A-smem/launch 微调追逐 6.5 ms；该路径
冻结为 oracle。

### Phase 2B-0：表示证明与 routed microkernel — **direct-folding 已实测失败，9.97 ms vs 2.4 ms gate**

第一轮不接 backend，只做真实权重/真实 route capture 上的 routed microkernel：

- 先完成 INT8 range、folding/two-plane 重构公式、误差界和指令下界； ✓ 已提交
  （`tools/prescreen_iq2_kgroup_fold.py`，真实 GGUF blk.4，K32..K256 sweep）
- 实现 K32/64/128/256 scale-amortization sweep； ✓ K32 通过 quality gate，
  K64/128/256 单调退化
- 先测 direct-packed；仅在 profiler 证明 AoS decode/transaction 是 blocker 后启动独立 SoA migration；
- 与 exact kernel 比较 gate/up/down 输出； ✓ cos 0.99999/0.99999/0.99993
- 报告 gate+up、SwiGLU、down、group/reduce 四段独立 GPU 时间；
- Nsight 记录 MMA、I2F、整数 decode、DRAM 和 occupancy，不用 wall time猜瓶颈。 ✓

**2026-08-11 实测结论（direct-folding 失败，kill gate 未达）**：

`iq2_mma16_tc.cu` 实现 K-group=32 scale folding（数值全部 >=0.9999），四轮
优化（模板 M_PAD 消除 spill → direct-global decode → fused decode/fold）
从 24 ms 降到 **9.97 ms gate+up**（E=256 M_PAD=32 真实路由）。Nsight 证据：

- tensor core 仅 1.2% 指令；ALU 1.69G（decode 位操作）是最大单项；
- SM throughput 64.6%，tensor 12% —— 指令流水近饱和但 tensor 饿死；
- **纯 mma 下限（无 decode/staging）4.72 ms，仍超 2.4 ms gate 2×**；
- K32 mma 减半 tensor 指令但总时间反升（寄存器/发射），回退。

**这证伪了 §4.3 "I2F 串行依赖是主瓶颈" 的前提**：瓶颈是 decode 的 ALU
指令量和 L1TEX/smem 流量。two-plane 预解码 codebook 预计 ~6-8 ms，仍超
2.4 ms。单卡 IQ2_XS Tensor-Core gate+up 在本 block 组织下现实下限 ~5 ms。

**决策点**：在投入 two-plane 前需重新核算——(a) 6.5 ms/layer MoE 预算是否
基于不现实的 gate+up 2.4 ms；(b) 是否有不同的 block 组织（更大 N/流式）
能把纯 mma 4.7 ms 降到 2.4 ms 以下。若两者都否定，则按 §9 触发"停止单卡
2K 承诺"分支。详见 `notes/2026-08-11-dsv4-phase2b-scale-amortized-falsified.md`。

6.5 ms 是 router+routed+shared MoE 的完整预算，不只是新 kernel 的预算。Phase 2B 的目标拆账如下；
这些是 **[待验证目标]**，不是当前实测：

| 子阶段 | p50 目标 |
|---|---:|
| router score/top-k | ≤0.3 ms |
| device route/group/gather + A quant | ≤0.5 ms |
| dual gate/up | ≤2.4 ms |
| SwiGLU + down-input quant | ≤0.4 ms |
| down | ≤1.3 ms |
| stable route reduction | ≤0.3 ms |
| shared expert | ≤1.0 ms |
| 余量 | ≥0.3 ms |
| router+routed+shared MoE 总计 | ≤6.5 ms |

Phase 2B-0 kernel gate：

- dual gate/up p50 `<=2.4 ms`，down p50 `<=1.3 ms`；
- 连续 100 replay 无 host sync、allocation、JIT；
- gate/up/down 各自相对 exact oracle cosine `>=0.9999`；
- scratch `<=256 MiB`。

### Phase 2B-1：完整 MoE operator gate

只有 Phase 2B-0 通过后，才接 device route/group/gather、SwiGLU、stable reduction 和 shared
expert。真实 layer、真实 1024-token routes 必须满足：

- router + routed pipeline + shared expert p50 `<=6.5 ms`、p90 `<=7.0 ms`；
- 连续 100 replay 无 host sync、allocation、JIT；
- 完整 MoE 输出相对旧路径 cosine `>=0.9999`；
- scratch `<=256 MiB`，无 model-sized second representation。

full-model worst logits cosine和 greedy 39/39 属于 Phase 4 集成门禁，不前置伪装成 kernel-only
证据。

所有 K-scale group 与 two-plane decomposition 都失败时，才停止**当前 IQ2_XS 表示**的 2K
承诺，转向新的、仍能单卡常驻的 2-bit Tensor-Core-native checkpoint format；若模型/权重格式
也不允许改变，则如实记录目标在当前约束下未证可达。多 GPU 不属于本项目 fallback。不得回到
route-M1 小修，也不得把 100–500 tok/s 宣布为完成。

### Phase 3：HC/attention 预算闭环 — **未开始**

- HC+norm `<=1.4 ms/layer`；
- attention `<=3.5 ms/layer`；
- 验证 64/128-row attention tile 的性能、状态和显存；
- 未同时达标不得进入全模型集成。

### Phase 4：全模型集成和预热 — **未开始**

- backend 生产默认 superchunk=1024；
- prewarm 64/128/256/512/1024 buckets 和 gate/up/down 三种 shape；
- decode 继续使用旧路径；
- 1024-token full-model `<=512 ms`；任何 fallback 直接失败。
- full-model worst logits cosine `>=0.99`，greedy `39/39`；

### Phase 5：服务发布门禁 — **未开始**

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
- M=1/8/16/24/32/48/64，生产 K/N=4096/2048；
- K-scale group 32/64/128/256 的误差、I2F 次数和性能曲线；
- gate/up dual、SwiGLU clamp、down、router weight；
- Nsight 确认实际发射 Tensor Core 指令；
- 100 replay 无 allocation/sync/JIT。

### 全模型

- 1–1025 边界长度及 4K/16K/128K；
- ratio 0/4/128、hash layers、普通 router layers；
- 1–2 slots、reset/reuse、decode graph；
- logits、greedy、long-context NaN、prefix checkpoint；
- `ruff check .`、torch-free pytest、full pytest、`git diff --check`。

### 接手者逐阶段执行命令

Phase 2B-0 的提交必须新增独立 target 和 test 文件。以下名称是本实施合同的一部分，接手者应按
此命名落地，避免复用 exact artifact 后无法区分结果：

```bash
# Phase 2B-0：candidate kernel。以下 target/test 由该提交新增。
make build-iq2-mma16-tc
make verify-iq2-mma16-tc
~/.venvs/vllm/bin/python -m pytest -q tests/test_iq2_mma16_tc_kernel.py

# 每次候选提交都必须确认 exact oracle 没有被破坏。
make build-iq2-mma16
~/.venvs/vllm/bin/python -m pytest -q tests/test_iq2_mma16_kernel.py

# Phase 1/2B-1/3 的局部回归。
~/.venvs/vllm/bin/python -m pytest -q \
  tests/test_dsv4_chunked_prefill.py \
  tests/test_dsv4_backend.py \
  tests/test_dsv4_slots.py \
  tests/test_dsv4_slot_arena.py

# 合并前的仓库门禁。
ruff check .
/tmp/ci-sim/bin/python -m pytest -q
~/.venvs/vllm/bin/python -m pytest -q
git diff --check
```

性能运行不得只留下终端文本。Phase 2B-0/2B-1 的真实 layer operation 必须产生 bfdiag run
record；Phase 4/5 必须为每个 fresh-process 4K/16K/128K 测试分别产生 run id。比较流程固定为：

```bash
bf show <run-id>
bf diff <baseline-run-id> <candidate-run-id>
bf trace show <candidate-run-id>
```

若 `bf diff` 判定不可比，结果作废；若性能失败，先读已有 trace，不为得到更好数字盲目重跑。

## 8. 拒绝项与风险

| 项目 | 决策 |
|---|---|
| 继续优化 route-M1 dp4a | 拒绝；warp-row 仅再快 1.2×，离目标约 15× |
| resident W8A8/NVFP4 cache | 拒绝；单层约 6 GiB，96 GB 无法常驻 |
| 全局 dequant BF16 + cuBLAS | 拒绝；GiB 级中间写流量，已有实测更慢 |
| 改 top-k/层数 | 拒绝；改变模型语义 |
| exact K32 scale/累加 | 高风险；由 Phase 2 数值+性能 kill gate 控制 |
| 继续优化 exact per-K16 I2F 路径 | 拒绝作为 production 主线；routed pipeline 32 ms 且未含 router/shared expert，已触发 kill gate，保留为 oracle |
| 直接接入当前 `grouped_moe_prefill` | 拒绝；存在 host sync、动态分配、FP32 大中间张量且没有 reference parity 断言 |
| scale-amortized IQ2 Tensor Core | 当前主线；必须同时通过 6.5 ms、cos/greedy、96 GB 三重门禁 |
| attention causal state | 高风险；Phase 1 boundary parity 前置 |
| 2K 未达成 | 不降低门槛宣布成功；按超预算模块继续优化或更换约束 |

## 9. 当前结论与下一个可提交里程碑

2026-08-11 已把 server/bfdiag 的旧 `max_q_rows=32` 默认统一为实测最佳的 64，并补齐
provider/parser 一致性测试。它把稳态约 90–105 提到 131.8–133.3 tok/s，且 2-slot×128K
decode graph 捕获成功。

该改动只建立更好的实施基线，不计入 2K 目标完成度。

截至 `93aa25a`，可以计入完成度的是“观测工具、scheduler parity prototype、exact kernel
oracle”；不能计入的是 production throughput。下一个里程碑必须是一个窄提交，且同时包含：

1. K32/64/128/256 或 two-plane 的真实单层 gate/up/down 曲线；
2. 至少一个 recipe 达到 gate+up p50 `<=2.4 ms`、down p50 `<=1.3 ms`；
3. gate/up/down 各自 exact-oracle cosine `>=0.9999`，并报告 propagated route 误差；
4. 100 replay 无 host sync/allocation/JIT 的证据；
5. 明确的 scratch/weight resident bytes，且 scratch `<=256 MiB`。

完成该 microkernel 里程碑后才进入 Phase 2B-1；只有完整
`router+routed+shared <=6.5 ms/layer` 后，才允许把 superchunk 接到 production prefill，避免把
128K correctness、prefix cache 和 slot 状态风险叠加到一个尚未过性能门的 kernel 上。

## 10. 外部依据

- [NVIDIA RTX PRO 6000 Blackwell Max-Q 规格](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/)
- [NVIDIA RTX Blackwell PRO GPU Architecture](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf)
- [CUTLASS integer sub-byte types](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/integer_subbyte.h)
- [CUTLASS mixed-dtype GEMM example](https://github.com/NVIDIA/cutlass/blob/main/examples/55_hopper_mixed_dtype_gemm/55_hopper_mixed_dtype_gemm.cu)
- [SGLang DeepSeek-V4 deployment and recipes](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx)
- [SGLang DeepSeek-V4 model execution graph](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/deepseek_v4.py)
- [SGLang DSV4 FP4 B200 end-to-end gates](https://github.com/sgl-project/sglang/blob/main/test/registered/models_e2e/test_deepseek_v4_flash_fp4_b200.py)
