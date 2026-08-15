# Qwen3.8 / SM120：从 FA4 技术到 SparkInfer/runtime 的完整优化规划

> 状态：可实施技术规划，2026-08-15
> 目标模型：`unsloth/Qwen3.8-27B-NVFP4`
> 目标硬件：单卡 NVIDIA Blackwell SM120（RTX PRO 6000 Blackwell，188 SM，约 95.6 GiB）
> FA4 快照：`fa4-v4.0.0.beta26`，commit `145b1010051dbfd4bdc41a0ae55d495b08d7a458`
> SparkInfer/b12x 快照：本机 `/home/bot/project/sparkinfer`，commit `d94b0cc`
> 目标：不降低质量、MTP 接受率和 prefill，压低 4×256K 常驻/峰值显存，并提高长上下文 accepted tok/s

## 1. 这份规划要解决什么

这不是一份“是否用 FA4 替换现有 kernel”的评估。FA4 在这里是一个经过验证的技术矿山；真正的任务是：

1. 先分清 SM100 与 SM120 的物理能力边界；
2. 把 FA4 的每项优化拆成算法、调度、数据搬运和硬件指令四层；
3. 将可行部分落到现有 b12x paged attention 和 Qwen runtime；
4. 同时处理 attention 之外更大的显存与整轮性能浪费；
5. 用真实 profile、accepted tok/s、ITL 和四口径显存证明收益。

FA4 beta26 的 SM120 实现本身不能覆盖本项目的 `paged KV + SplitKV + FP8 KV + hd256` 生产组合，
但这只是一条实现边界，**不是本规划的中心结论**。我们不替换现有 kernel；我们优化 SparkInfer 和 runtime。

本文区分三种证据：

- **实测**：本机完整服务运行得到；
- **代码精算**：由真实 shape、dtype、tensor/header 或 allocator 公式精确计算；
- **待验证推断**：有机制依据，但必须通过 fresh-process/profile A/B 才能成为结论。

## 2. SM100 与 SM120：先把物理边界说清楚

官方依据：

- [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- [PTX ISA 9.3](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUTLASS Blackwell functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
- [CUTLASS tcgen05 programming guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)
- [CUDA thread-block clusters / DSMEM](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters)
- [CUDA Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)

### 2.1 硬件能力矩阵

| 能力 | SM100 | SM120 | 对本项目的结论 |
|---|---|---|---|
| `tcgen05` / UMMA | 支持异步、单线程发射 | PTX target 不包含 SM120 | **物理不可实现**；不能复刻 FA4 的异步 QK/PV issuer |
| TMEM accumulator | 支持 | `tcgen05.ld/st/alloc/...` target 不含 SM120 | **物理不可实现**；S/P/O 必须回到 registers/SMEM |
| `tcgen05.ld.red` | 部分 SM100-family 支持 | 不支持 | **物理不可实现**；rowmax 必须 shuffle/SMEM reduction |
| 2-CTA joint MMA | `cta_group::2` | 无对应 `tcgen05` | **物理不可实现**；cluster/DSMEM 不能补出联合 MMA |
| `mma.sync` | 支持 | 支持，含 SM120 narrow/block-scaled 扩展 | 数学可实现；但它是同步 warp-level MMA，流水结构必须重写 |
| TMA `cp.async.bulk.tensor` | 支持 | 支持 | **可直接用**；b12x 已用于 paged K/V |
| TMA `tile::gather4/scatter4` | 支持相关 family | PTX feature table 不含 SM120 | **不可依赖**；paged KV 继续用 page-id + plane descriptor，而非任意 gather TMA |
| TMA multicast | 优化目标 | ISA 功能可用，但官方优化/建议 target 列表排除 SM120，并警告其他 target 可能显著降速 | **不是物理缺失，也不是默认优化项**；只能受控 A/B |
| Thread-block cluster / DSMEM | 支持 | 支持 | **可用但有成本**；不能把“无 2CTA UMMA”误写成“无 DSMEM” |
| Cluster Launch Control | 支持 | `try_cancel/query_cancel` 可用，`multicast::cluster::all` 明列 SM120 | **可试验**；先以 atomic work queue 为基线 |
| `setmaxnreg` | 支持 | 支持 | 可做 producer/math warp 寄存器再分配；必须满足一致控制流约束 |
| SMEM/SM | 228 KiB | 128 KiB | FA4 的深 KV pipeline 不能照搬 |
| 最大 SMEM/block | 227 KiB | 99 KiB | tile、stage、P staging 必须重新算；99 KiB 是硬上限 |
| 最大常驻 warps/SM | 64 | 48 | 16-warp FA4 CTA 不适合作为 SM120 默认拓扑 |
| register file | 64K × 32-bit/SM | 64K × 32-bit/SM | 无 TMEM 后 accumulator 回到 RF，SM120 对 spill/occupancy 更敏感 |
| `ex2.approx` | 支持 | 支持 | 功能可用；SM120 MUFU 吞吐没有可靠公开对比，收益必须实测 |
| packed `f32x2` arithmetic | 支持 | SM120 支持 | 可试 FA4 的 packed scale/subtract 和 polynomial exp |

### 2.2 三类迁移结论

**A. 物理不可实现，不进入 SM120 实施清单**

- `tcgen05.mma`、TMEM、TMEM copy/`ld.red`；
- 两个 CTA 联合执行一次 UMMA；
- 依赖 TMEM 的 split-P arrival 与异步 QK→PV 发射协议；
- FA4 224 KiB 级 SMEM/16-warp 原拓扑。

**B. 能实现，但必须改用 SM120 机制重构**

- QK/PV overlap：改成跨 head/跨 warp-group 的软件 ping-pong，而不是同一 UMMA issuer 异步发射；
- correction warp：若拆出，P/O 必须经 SMEM 或寄存器交换，先证明 barrier/SMEM 成本值得；
- 2-CTA 协作：只能做普通 cluster/DSMEM 数据共享或调度，不能宣称等价 2CTA MMA；
- TMA multicast：功能存在但 SM120 不是官方优化 target，必须对比独立 TMA + L2 reuse；
- persistent scheduler：可用 atomic work queue，再评估 CLC。

**C. 可直接借鉴**

- conditional rescale、packed `f32x2`、软/硬 `exp2` 混合；
- GQA packing、Q 常驻片上、P 低精度存储；
- TMA/cp.async producer-consumer、descriptor prefetch；
- stage/tile/residency 联合搜索；
- SplitKV one-wave/minimax、LPT、L2 swizzle、persistent work queue；
- dynamic compile key 和 live-length metadata。

### 2.3 SM120 应采用的执行骨架

SM120 的基线不应复制 FA4 的 16 warps，而应从当前 b12x 的窄拓扑继续收敛：

```text
1 producer warp：page id / TMA descriptor / K-V prefetch
4 math warps：mma.sync QK -> online softmax -> mma.sync PV
可选第二组 4 math warps：处理另一个 head/chunk，实现软件 ping-pong
```

CUTLASS 官方 SM120 GEMM 已证明两类软件调度是合理硬件用法：

- ping-pong：两组各 4 个 MMA warp 处理不同输出 tile，交叠 mainloop/epilogue；
- cooperative：一组 8 个 MMA warp 处理同一 tile。

对 attention 的迁移含义是“跨独立 head/chunk 交叠”，不是“SM120 拥有 FA4 的异步 UMMA”。

## 3. Qwen3.8 当前事实基线

### 3.1 模型与生产 shape

| 项目 | 值 |
|---|---:|
| hidden size / layers | 5120 / 64 |
| GDN / full attention layers | 48 / 16 |
| Q heads / KV heads / GQA | 24 / 4 / 6 |
| attention head dim | 256 |
| maximum position | 262,144 |
| speculative decoding | MTP K=3，1 个 MTP layer |
| backbone KV | FP8 E4M3 |
| MTP KV | BF16 |

三个需要分别调优的 attention family：

| family | 典型 packed Q | 关键矛盾 |
|---|---:|---|
| decode | q_len=1，GQA6 后每 KV head 6 rows | 长 KV、row0 softmax、SplitKV、尾波 |
| verify | q_len=K+1=4，GQA6 后每 KV head 24 rows | QK/PV 利用率、ragged sync、partial rows |
| prefill/extend | 512–8192 token chunk | TTFT、workspace、decode stall、large-M GEMM/attention |

### 3.2 最新实测

来源：[`../notes/2026-08-15-qwen38-128k-decode-profile.md`](../notes/2026-08-15-qwen38-128k-decode-profile.md)。

| 工作负载 | 结果 |
|---|---:|
| B1 128K warm decode | **108.1 tok/s** |
| B4 128K warm decode | **201.9 aggregate tok/s** |
| K=3 draft acceptance | **58.4%** |
| 每 slot-round 提交 | 2.753 tokens（含 target） |
| B4 round time | 约 54–56 ms |
| 启动 / B4 后显存 | 56,650 / 60,072 MiB |
| 当前 elastic KV pool | 4,160 bundles，18.28125 GiB |

这次运行没有 kernel attribution。54–56 ms 不能直接归咎于 attention、GDN、GEMM 或 host gap。

当前 round time 不变时，即使 K=3 完美接受，B4 数学上限也只有约 **292.5 tok/s**。因此若把
**1000 accepted tok/s** 作为长期目标，仅提高接受率不够；在 B4/K=3 下还需要把完美接受时的 round
降到约 16 ms，或改变并发/投机结构。规划不能拿单个 microkernel 的百分比假装已接近该目标。

## 4. 显存：先解决真正的大头

### 4.1 4×256K 的硬账

每 token：

```text
backbone = 16 × 2(K/V) × 4 × 256 × 1 byte = 32 KiB
MTP      =  1 × 2(K/V) × 4 × 256 × 2 byte =  4 KiB
total                                               = 36 KiB/token
```

page size 128 时，一个 bundle 是 4.5 MiB。strict pool 的真实公式是：

```text
1 null + 4 × 2048 business bundles + 8 watermark = 8201 bundles
8201 × 4.5 MiB = 36.03955 GiB
```

这是代码精算。动态 allocator 不能把四条真实、独立、完整 256K 上下文降到这个 KV 下界以下。

### 4.2 dynamic arena 的真实含义

当前 `strict/elastic` arena 已经存在，但“动态”的是 page ownership/refcount/hash/COW/admission；GPU
物理 tensor 在启动时按 `pool_bundles` 一次性 `torch.zeros`。所以：

- 当前 4,160 bundles 是 18.28125 GiB 常驻，不会随 live token 释放给 driver；
- 它避免 `(slots + full scratch) × 256K` 固定行，却只能承载约 532K business tokens；
- strict 4×256K 仍需 36.03955 GiB；
- 下一步是减少 pool 之外的重复布局和常驻 workspace，而不是把 ownership 动态误报为 VMM commit。

### 4.3 已识别显存分项

| 项目 | 大小 | 证据 |
|---|---:|---|
| strict backbone+MTP KV | **36.03955 GiB** | 代码精算 |
| runtime 加载的语言模型原始 tensors | **20.9509 GiB** | safetensors header 实算 |
| 默认 all-W4A4 下遗留 W4A16 派生表示 | 约 **7.84 GiB**，连 runtime/scale 上限约 8.72 GiB | 代码推算，需 allocated delta |
| GDN live rows，4 slots + scratch，K=3 | **1.47949 GiB** | 代码精算 |
| 4 个 rolling checkpoint destinations | **303 MiB** | 代码精算 |
| 每个 persistent GDN checkpoint clone | **75.75 MiB** | 代码精算 |
| 16 层 eager extend+decode arenas | **848.47 MiB** | b12x layout 精算 |
| MTP CUDA Graph private pools | 36 个，字节 unknown | 运行时分段测量 |
| W8A8 workspaces / allocator reserve | unknown | 运行时分段测量 |

### 4.4 P0-M1：跳过 all-W4A4 场景的死 W4A16 初始化

当前默认 `QSR_QWEN36_MLP_W4A4_ALL=1`，但首次 forward 仍先进入 W4A16 prepare，再生成 W4A4；之后
all-row 路径实际使用 W4A4。真实 checkpoint 的一套 NVFP4 packed+scale 表示约 7.844 GiB，此外还有
`_w4a16_graph_runtime` buffer。

实施：

1. all-W4A4 时直接 prepare W4A4；
2. 不生成 W4A16 prepared/runtime；
3. 只有确认没有任何 fallback 后才释放 W4A16/raw scale 所需数据；
4. 保留 feature flag，以 fresh-process 对照 W4A16 小 M 性能和质量。

门禁：首个 forward、全部 decode/MTP graph capture/replay、W4A4/W4A16 logits、accepted tok/s、prefill。

### 4.5 P0-M2：共享/延迟物化 eager attention workspace

加载 warmup 当前让 16 个 full-attention layer 各自 materialize extend 和 decode arena：

```text
one arena = 27,802,624 bytes = 26.5146 MiB
16 layers × 2 modes = 848.47 MiB
```

分三级实施：

1. 同 mode 跨 16 层共享：理论回收 **795.44 MiB**；
2. graph 全量捕获成功后释放仍保留的 B1–B4 eager decode drivers；
3. 生产禁用 serial fallback 时不 materialize per-layer decode workspace。

per-layer K/V descale 必须在 forward 时传入；extend/decode/verify 不能无证明混用同一 arena。

### 4.6 P0-M3：MTP graph pool 按 family 共享

主 decode B1–B4 已共享 graph pool，MTP 的 36 个 graph 仍各自捕获 private pool：

```text
target verify: 4
draft:         4
sync:         16
sync verify:  12
```

安全起点：同 family 的互斥 bucket 共享，`4→1 / 4→1 / 28→1`。不要直接跨 family 共池；verify 的
`all_hiddens/logits` 可能仍被 sync 使用。每个 family 捕获前后记录 NVML、allocated、reserved 和 graph pool bytes。

### 4.7 P1-M：GDN checkpoint 硬预算与池化

当前 rolling destination 每槽懒分配后常驻；persistent prefix 又 clone 75.75 MiB/entry。动态 arena 分支的
GDN budget 仍可被 pinned entry 超过。

实施：

- 固定 byte/entry hard cap；
- checkpoint storage 池化或 ownership transfer，避免每 entry 产生 96 个 tensor clone；
- KV hash、GDN checkpoint、final hidden 原子发布和原子淘汰；
- rolling cadence 只有 profile 证明 checkpoint round spike 后才从 16 调到 64/128。

### 4.8 P0-C：先修 partial-page MTP COW 原子性

`prepare_kv_writes()` 的 COW 当前只复制 16 层 backbone K/V；MTP pooled K/V 不在复制循环。arena 又允许
block-size 边界命中落在半个 128-token page 内。首次 suffix 写可能出现：

```text
bundle mapping：backbone 与 MTP 一起换新 bundle
content copy：只复制 backbone 旧半页
MTP：新页丢失 prefix 半页
```

这可能不破坏 target token，却会降低 draft/acceptance。任何 arena 内存优化前必须二选一：

1. 把 MTP K/V 注册为第 17 个原子 COW family；或
2. 临时把 MTP prefix restore 截断到 page128 边界。

回归覆盖 partial page、4 slots 同轮 COW、取消/复用、graph replay 和 acceptance histogram。

### 4.9 strict 4×256K 的现实目标

当前 `56.65 GiB startup - 18.281 GiB pool + 36.040 GiB strict pool ≈ 74.4 GiB` 只是容量推算。
若 W4 重复布局和 eager workspace 的静态上限确实被回收，strict 启动可能进入约 65–70 GiB 区间；graph
private pool 的收益仍 unknown。**必须 fresh process 分阶段测量，不能把推算写成实测。**

阶段验收目标：

- strict 4×256K 可启动并完成逐槽增长，不 OOM；
- 运行峰值保留至少 10 GiB driver free；
- 短请求不会因 `max_model_len=256K` 占用完整 business KV；
- 同时报 NVML used、PyTorch allocated、reserved、graph-private pool。

## 5. 当前已经实现的能力：不要重复开发

### Runtime 已有

- 主 decode B1–B4 CUDA Graph 和共享 pool；
- MTP target verify / sync / draft graph；
- anchor 折入 K+1 verify、device-direct draft seed；
- indexed multistep GDN recurrence，无 K 次 snapshot/restore；
- raw W8A8 dense path，无旧 BF16 权重 cache；
- fused qkvz projection、W4A4/W4A16 两条 NVFP4 MLP path；
- rolling prefix hash、dynamic bundle ownership、prefix alias/COW；
- uniform multi-slot prefill batching。

### b12x paged attention 已有

- paged FP8 KV、SplitKV、page128；
- GQA6 packed rows；
- K/V TMA、descriptor cache、producer warp；
- FP32 online softmax、BF16 packed P、`ex2.approx`；
- decode/verify graph live re-chunk、one-wave/minimax planner；
- stage/SMEM/CTA-residency 精确校验；
- Qwen hd256 production path。

因此“加 TMA、加 Pack-GQA、把 P 从 FP32 降 BF16、实现 SplitKV”都不是新任务。

## 6. 性能：先解释整轮，再优化 hotspot

### 6.1 P0-P：真实 profile 与判别树

第一轮必须使用现有 `QSR_PROFILE_ROUNDS=2`，解释至少 90% 的 54–56 ms round wall：

```text
setup -> verify_replay -> accept_gpu_wait -> commit_loop
      -> sync_ragged -> draft_batch -> engine bookkeeping
```

同时按 GPU kernel family 归因：attention forward/merge、W8A8、W4A4/W4A16、GDN、KV scatter、sampling。

决策：

- kernel sum / wall >80%：按 GPU ms 排前两名；
- GPU 有明显空洞：先消 host sync、metadata、D2H accept、checkpoint copy；
- attention forward+merge 不在前两名：FA4 kernel 实验保留，但不抢 runtime P1；
- 单 kernel 变快而 E2E 不变：查并发、graph、merge/host 抵消，不因一次 E2E 无变化就否定真实 hotspot。

所有 A/B 先 `bf diff`，固定 prompt、K、KV dtype、page size、prefix state、graph hit 和 sampler。

### 6.2 P0-P：K=1/2/3 content-matched sweep

58.4% acceptance 留有明显 token-yield 空间，但 K 越小 round 更短。必须直接比较：

- accepted tok/s；
- round ms、ITL；
- accepted tokens/round；
- reject-position histogram；
- verify/draft/sync GPU ms。

不以 raw acceptance 单项选 K，也不把完美接受率上限当可实现性能。

### 6.3 P1-R：无需新 attention kernel 的高价值项

1. **W4A4-all vs W4A16 小 M 路由**：decode、verify、prefill 分开。W4A4 是三次 blockscaled GEMM+
   activation quant，W4A16 是 fused dense-as-one-expert；Qwen3.8 128K 的最优点不能继承 Qwen3.6 旧结论。
2. **ragged MTP sync 分桶**：仅当 `sync_gpu_ms >= 8–10% round`，比较当前 padded single graph 与按
   q=1/2/3/4 分桶；看整 round，不看单 sync kernel。
3. **fused KV quant+scatter**：Qwen full-attention 当前 scale/cast/K scatter/V scatter 是多个小 kernel；
   复用仓库现有 `runtime/kernels/fused_kv_scatter.py`，先按 profile 数调用和总时间。
4. **post-fix 128K exact-prefix repeat**：确认 GDN checkpoint restore 真正消除 prefill；这属于 TTFT/prefix，
   不混报为 steady decode 加速。
5. **sampled MTP batching**：temperature>0 当前会让整个 batch 退到逐 slot；作为独立服务路径优化，先做
   分布正确性，再测 B1/B4。

### 6.4 GDN 的边界

旧“GDN core 0.3–0.6%”只覆盖 recurrent kernel，不代表 48 个完整 GDN block。当前又已完成 qkvz fusion、
indexed multistep、spec rows 和 graphs。因此只在新 profile 证明 elementwise/conv/状态更新是热点后尝试：

- 去掉 Q/K `repeat_interleave` 物化；
- 融合 `conv_state + depthwise conv + copy + SiLU`；
- gate/beta activation 下沉现有 FLA kernel；
- 保持 slot reset、rollback、accepted state column 的逐项一致。

## 7. FA4 技术迁移矩阵：落到 b12x 的具体任务

| FA4 技术 | SM120 可行性 | b12x 状态 | 实施级别 | 主要指标 |
|---|---|---|---|---|
| exact conditional rescale | 直接可行 | 缺失 | **P0** | skip ratio、O-FMA、kernel/round |
| threshold rescale 4/8 | 可行但改变数值路径 | 缺失 | P1 | saturation、logits、acceptance |
| packed `f32x2` scale/subtract | 直接可行 | 标量为主 | **P0** | SASS、instr、regs、latency |
| 软件/硬件 exp2 混合 | 可行，收益未知 | all hardware approx | P1，profile 触发 | MUFU/FMA busy、误差 |
| BF16 packed P | 直接可行 | **已实现** | 不重复做 | — |
| Q/GQA packing | 直接可行 | **已实现** | 仅收敛策略 | KV bytes、CTA residency |
| TMA K/V pipeline | 直接可行 | **已实现主体** | **P0 调参** | TMA wait、stage、occupancy |
| descriptor prefetch/cache | 直接可行 | 大部已实现 | P1 审计 | descriptor/page-table loads |
| page-id K/V 复用 | 直接可行 | producer 已有，generic 待审计 | P1 | loads、regs、occupancy |
| stage 1/2/3 | 受 99 KiB 限制 | FP8/多 KV warp 常为 stage1 | **P0 A/B** | wait vs 1/2 CTA/SM |
| SplitKV one-wave/minimax | 直接可行 | **已实现** | **P0 调 LUT** | forward+merge、scratch |
| head-pair KV reuse | 可行 | 实验结构已有 | P0/P1 | DRAM bytes、B1/B4、SMEM |
| LPT / full-chunk-first | 直接可行 | request-major | P1 | tail CTA、SM active、L2 |
| persistent forward CTA | 可行 | forward 缺失 | P1 | tail、atomic cost、p95 |
| CLC work stealing | 硬件支持 | 缺失 | P2 | 与 atomic queue A/B |
| TMA multicast | 功能可用但 SM120 非优化 target | 缺失 | P3 microbench | 独立 TMA/L2 对照 |
| cluster/DSMEM KV sharing | 可行但非联合 MMA | 缺失 | P3 | remote SMEM/barrier/occupancy |
| 跨-head QK/PV ping-pong | 可用软件重构 | 缺失 | P2 | tensor-pipe bubbles、SMEM |
| split-P early arrival | 依赖 TMEM/async UMMA 才自然 | register-local P 无消费者 | **不优先** | 需先证明重构价值 |
| `tcgen05`/TMEM/2CTA UMMA | 物理不可实现 | — | **不做** | — |

### 7.1 P0-A1：exact conditional rescale

第一版只做不改变 softmax 基准的 exact skip：当 `m_new == m_prev` 时，不执行 `d/o *= 1`。先落到 decode
row0 快路径，再覆盖 verify/pairwise。它应保持现有数值结果。

第二版才试 FA4 的 threshold `{4,8}`：在安全范围内继续使用旧 max，从源头避免 rescale。这会改变舍入和
underflow 路径，必须独立提交并跑长上下文质量/acceptance，不能混入 exact 版本。

### 7.2 P0-A2：packed arithmetic

将 score 的 subtract/scale 两两打包为 `f32x2`，先在 row0 kernel 落地。必须检查 SASS 确实生成 packed
指令；若 DSL 拆回标量或 packing 提高 register count 导致变慢，淘汰。

### 7.3 P0-A3：tile/stage/role/residency 联合搜索

不能单独搜索 `num_stages`。每个 workload family 联合搜索：

```text
tile N:          64 / 128
stage:           1 / 2 / 3（受99 KiB和精确traits约束）
producer role:   on / off
heads per CTA:   1 / 2
residency:       1 / 2 CTA per SM
SplitKV chunks:  planner候选集
```

decode、verify、prefill分开；B1/B2/B4、128K/256K分开。目标函数是 attention forward+merge 和 whole-round
accepted tok/s，同时限制 partial scratch 和 compile variants，不是“单 CTA cycles 最低”。

### 7.4 P1-A：调度与尾波

1. 低风险：同 request 内 full chunks 先于 tail chunk；
2. 中风险：新增显式 `partial_idx`，允许按 estimated KV work 降序，同时保持 merge mapping；
3. persistent CTA：固定 `SM × target_ctas_per_sm`，atomic counter 取 work；
4. CLC 只在 atomic 基线完成后 A/B；
5. L2 swizzle 使用本机真实 L2/冷缓存证据，不复制 FA4 的固定 50 MB 假设。

### 7.5 P2-A：SM120 版 QK/PV overlap

不能在同一 math warp 内复刻异步 UMMA。可行实验是：

- warp group A 做 head h 的 PV；
- warp group B 同时做 head h+1 的 QK；
- producer warp 为两组提前发 TMA；
- P 仍尽量 register-local；若必须经 SMEM，先做精确 byte/barrier/occupancy 预算。

只有 NCU 显示 tensor pipe 中间空泡且 KV/TMA 不是主瓶颈时进入实现。

## 8. Prefill：保持速度并继续优化

当前 uniform multi-slot 相同 chunk 已 batch；mixed/ragged suffix 仍可能逐 slot。prefill 单独执行：

1. fresh-process cold cells：1K/8K/32K/128K；
2. chunk `{512,1024,2048,4096,8192}`，按无 decode/有 decode 分别选；
3. admission 按 `(remaining chunk, GDN state regime)` 分桶，尽量维持 B×Q；
4. 无 active decode 时放大 chunk；有 active decode 时由目标 ITL、等待队列、KV reservation 给 GPU-time budget；
5. 大 M 保留 W4A4 prefill，不因小 M decode 路由实验一起关闭；
6. 共享 eager extend arena后检查 TTFT、workspace、JIT/compile-key churn；
7. attention 只有在 cold profile 占比足够高时才应用 stage/producer/QK-PV 实验。

门禁：prefill tok/s、TTFT、peak resident、active decode ITL；decode 优化不得造成可重复 >3% prefill 回退。

## 9. CUDA 13.3：可以怎样帮助这条路线

官方资料：

- [CUDA 13.3 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [CUDA Tile C++](https://developer.nvidia.com/blog/develop-high-performance-gpu-kernels-in-cpp-with-nvidia-cuda-tile/)
- [NVCC Advanced Controls](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#apply-controls-apply-controls)
- [CompileIQ](https://nvidia.github.io/CompileIQ/v1.0/index.html)
- [cuda.core 1.0](https://nvidia.github.io/cuda-python/cuda-core/latest/release/1.0.0-notes.html)

### 9.1 已完成的快速验证

本机 `/usr/local/cuda-13.3/bin/nvcc 13.3.73` 已用：

```bash
nvcc -std=c++20 --enable-tile -arch sm_120 ...
```

编译并运行一个临时 CUDA Tile C++ kernel，`max_error=0`，cubin 为 `sm_120`；临时文件已删除。这只证明
工具链可用，不是性能证据。当前 PATH 的 `nvcc` 仍是 13.2.78，而 PyTorch/runtime 是 CUDA 13.3；后续
构建必须记录绝对 nvcc 路径、版本、driver、cubin arch 和 CUTLASS component。

### 9.2 CUDA Tile C++

用途：小边界融合原型，而不是重写 paged attention。首选：

- Qwen fused FP8 KV quant+scatter；
- page/work metadata transform；
- SplitKV merge 或小 reduction；
- GDN conv/state 小融合。

每个原型保持 runtime ABI 不变，与现有 CUDA/Triton/CuTe 输入逐项对照。

### 9.3 CompileIQ / ACF

只用于 P0 profile 排名前二的 AOT `.cu` kernel；CuTe DSL/JIT paged attention 先用显式 workload autotune。
搜索目标必须同时包括 correctness、冷缓存 p50/p95、register、SMEM 和 whole-round。ACF 是 experimental；
crash、timeout、输出变化立即淘汰，不能把 control 应用到整个静态库。

### 9.4 CUDA Python 1.0

近期价值在统一 JIT/cache/graph/driver 测量工具，不直接减少模型显存或 kernel 时间。当前不新增生产依赖；
需要稳定 low-level harness 时另立设计。

### 9.5 13.3 风险门禁

对本项目相关的 cuBLAS 已知问题包括非默认 blocking stream capture heuristic、部分 M/N=1 unsupported、
CC12.x strided broadcast 读越界条件和 PDL alpha/beta WAR。算法选择在 capture 前完成；M=1 与 broadcast
形状保留 fallback；升级门禁加入 Compute Sanitizer 小形状覆盖。

## 10. 分阶段执行与文件边界

### Phase 0：证据与正确性（0.5–2 天）

- [ ] `QSR_PROFILE_ROUNDS=2` 解释 ≥90% round wall；
- [ ] post-fix 128K exact-prefix repeat；
- [ ] K=1/2/3 content-matched sweep；
- [ ] 修 MTP partial-page COW，先钉 regression；
- [ ] 分阶段显存采样：weights → W4 prepare → attention warmup → GDN K3 expand → 每个 graph family → strict KV。

### Phase 1M：显存 P0（2–5 天）

- [ ] all-W4A4 直接 prepare，跳过 W4A16 dead rep/runtime；
- [ ] 跨层共享 extend/decode eager arena，并延迟无用 fallback arena；
- [ ] MTP graph pool 按 target/draft/sync family 共享；
- [ ] persistent GDN checkpoint hard pool 与原子淘汰；
- [ ] strict 8201-bundle fresh-process 启动/逐槽增长验收。

### Phase 1R：runtime 性能（2–7 天，按 profile 选前两项）

- [ ] W4A4/W4A16 小 M 路由；
- [ ] ragged sync bucket；
- [ ] fused KV quant+scatter；
- [ ] sampled MTP batching；
- [ ] GDN/conv fusion 仅在新 profile 支持时进入。

### Phase 1A：b12x/FA4 可迁移 P0（3–10 天）

- [ ] exact conditional rescale；
- [ ] packed `f32x2` score arithmetic；
- [ ] tile/stage/producer/head-pair/residency 联合 A/B；
- [ ] SplitKV forward+merge+scratch 联合 LUT；
- [ ] page-id/descriptor load 审计。

### Phase 2：调度和深入流水（1–3 周）

- [ ] full-chunk-first/LPT + explicit partial mapping；
- [ ] persistent CTA atomic queue，再试 CLC；
- [ ] profiler 触发的软件 exp2；
- [ ] NCU 触发的跨-head QK/PV ping-pong；
- [ ] TMA multicast/DSMEM 只做 microbenchmark，不做默认依赖。

### Phase 3：CUDA 13.3 定向实验

- [ ] 一个 CUDA Tile 融合原型；
- [ ] 前二 AOT kernel 的 CompileIQ/ACF；
- [ ] 版本化 build provenance 和 sanitizer 升级门禁。

主 runtime 仓库：

- `runtime/model/qwen36_slots.py`：COW、arena、GDN/checkpoint 生命周期；
- `runtime/model/qwen36_model.py`：W4 prepare、workspace、KV scatter、GDN；
- `runtime/backends/qwen36.py`：prefix/checkpoint、主 graph pool；
- `runtime/backends/qwen36_mtp_cudagraph.py`：MTP graph pools、ragged sync；
- `server/engine.py`：strict/elastic admission、自适应 prefill；
- `bfdiag/`：复用现有 trace，只加通用分项，不新增一次性 benchmark。

SparkInfer 独立 worktree/提交：

- `b12x/attention/paged/forward_paged.py`：rescale、packed arithmetic、pipeline/head-pair；
- `b12x/attention/paged/planner.py`：SplitKV/LPT/persistent policy；
- `_forward.py` / tests：compile-key、merge、correctness/perf gates。

提交必须按 numerics、scheduler、memory-lifecycle、compiler-control 分离，避免无法归因。

## 11. 统一验收合同

### 正确性

- attention：page boundary、partial page、B1/B4、q=1/4、128K/256K、split merge；
- max/mean error、top-k logits agreement、greedy token drift；
- MTP：reject-position histogram、accepted tokens/round、state rollback；
- slots：COW、cancel、reset/reuse、prefix restore、4-slot interleave；
- graph：eager/graph parity、bucket replay、无地址生命周期污染；
- compiler candidate：sanitizer + unit/integration 后才允许测速。

### 性能

- 主指标：accepted tok/s、ITL、TTFT；
- 解释指标：round wall、kernel sum、attention forward/merge、host gap、TMA/barrier/tensor-pipe stalls；
- decode：128K/256K × B1/B4 × K=1/3；
- prefill：1K/8K/32K/128K cold；
- 每 cell 至少 3 次稳定运行，报告中位数与离散度；
- 冷缓存采用 ≥256 MiB 旋转足迹；
- 单 kernel 提升但 E2E 不变时必须给出抵消项，既不夸大，也不无证据放弃 hotspot。

### 显存

- 同时报告 NVML used、allocated、reserved、graph-private pool；
- 记录加载/prepare/warmup/GDN扩容/每个graph family/strict pool 的阶段 delta；
- 4×256K strict 不 OOM、最差保留 ≥10 GiB driver free；
- dynamic/elastic 报 physical bundles、reserved、published、prefix-cached、COW watermark，不把逻辑 capacity 当驻留。

## 12. 明确不做

- 不替换现有 production attention kernel；
- 不实现 `tcgen05`/TMEM/2CTA UMMA 的“SM120 等价物”；
- 不把普通 cluster/DSMEM 包装成 FA4 的联合 MMA；
- 不把 SM120 TMA multicast 说成物理不存在，也不在无实测时依赖它；
- 不重复实现 b12x 已有的 TMA、GQA、BF16 P、SplitKV、FP8 KV；
- 不考虑 NVFP4 KV；
- 不用降低并发、上下文、关闭 MTP 来伪装显存优化；
- 不用 warm engine 证明 cold-prefill 或 cold-memory 改善；
- 不用 `b12x.*.is_supported()` 版本 gate 作为实际 capability signal。

## 13. 第一批实际执行清单

1. 修 partial-page MTP COW；否则 dynamic prefix acceptance 数据不可信。
2. 抓两轮真实 profile，并跑同配置 K=1/2/3。
3. 对默认 all-W4A4 跳过 W4A16 prepare，目标先回收约 7.84 GiB 级重复表示。
4. 共享 16 层 eager arena，并按 family 合并 MTP graph pool。
5. fresh-process 启动 8201-bundle strict pool，取得真实 4×256K 显存账。
6. b12x 先做 exact conditional rescale 和 packed `f32x2`，每项独立提交。
7. 按 NCU 做 stage/tile/role/SplitKV 联合搜索；没有 stall 证据不盲目加 stage。
8. 再做 LPT/persistent CTA；跨-head QK/PV、CLC、TMA multicast放在后续受控实验。

这条路线把“硬件做不到”“能用不同机制做到”“已经做过”“值得立即做”明确分开。近期最大显存收益来自
runtime 生命周期和重复布局，近期 attention 收益来自 FA4 的条件 softmax、packed arithmetic 和调度思想；
真正依赖 SM100 `tcgen05`/TMEM 的部分被明确排除，不会再用错误的硬件前提指导 SM120 实现。

## 14. 实施证据索引

以下位置是开发开始前的代码入口；若代码继续演进，以符号搜索后的最新实现为准：

| 事实/任务 | 代码证据 |
|---|---|
| strict pool `1 + slots×pages + watermark` | `server/engine.py:843-848` |
| dynamic arena 启动时全量物理 tensor 分配 | `runtime/model/qwen36_slots.py:230-293` |
| backbone-only partial-page COW | `runtime/model/qwen36_slots.py:683-744` |
| partial-page prefix 命中/restore | `runtime/model/qwen36_slots.py:1203-1279` |
| all-W4A4 默认与首次 W4A16 路由 | `runtime/model/qwen36_model.py:2829-2863,3394-3400` |
| W4A16/W4A4 prepare 生命周期 | `runtime/model/qwen36_model.py:3066-3158,3200-3256` |
| eager workspace geometry/成员/warmup | `runtime/model/qwen36_model.py:1617-1649,2437-2451,4280-4306` |
| main decode graph shared pool | `runtime/backends/qwen36.py:2005-2043` |
| MTP target/draft/sync graph capture | `runtime/backends/qwen36_mtp_cudagraph.py:627-633,887-890,1383-1398` |
| rolling checkpoint destination | `runtime/model/qwen36_slots.py:1341-1371` |
| persistent GDN checkpoint clone | `runtime/backends/qwen36.py:1507-1538` |
| MTP round phase profiling | `runtime/backends/qwen36_mtp.py:1264+` |
| Qwen 分离式 KV scale/cast/scatter | `runtime/model/qwen36_model.py:2665,2742,2805` |
| 可复用 fused KV scatter | `runtime/kernels/fused_kv_scatter.py:13+` |
| b12x role producer/stage/residency | `/home/bot/project/sparkinfer/b12x/attention/paged/forward_paged.py:3205-3318,5258-5359` |
| b12x 无条件 rescale | `/home/bot/project/sparkinfer/b12x/attention/paged/forward_paged.py:2725-2735,2806-2816` |
| b12x packed BF16 P | `/home/bot/project/sparkinfer/b12x/attention/paged/forward_paged.py:2770-2771,6910-6943` |
| SplitKV worklist/LUT/scratch | `/home/bot/project/sparkinfer/b12x/attention/paged/planner.py:185-304,584-873,2279-2416` |
| kernel dynamic compile key/cache | `/home/bot/project/sparkinfer/b12x/attention/paged/_forward.py:303-408,1338-1437` |
