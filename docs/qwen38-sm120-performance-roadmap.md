# Qwen3.8 / SM120 性能优化路线：从动态 KV 到有效 M

> 状态：活文档
>
> 基线日期：2026-08-16
>
> 目标模型：`unsloth/Qwen3.8-27B-NVFP4`
>
> 目标硬件：单卡 RTX PRO 6000 Blackwell（SM120，96 GiB），TP=PP=EP=1
>
> 关系：本文件依据最新生产 profile 重新排序实施优先级；
> [`qwen38-sm120-cuda133-fa4-optimization-plan.md`](qwen38-sm120-cuda133-fa4-optimization-plan.md)
> 仍是 SM100/SM120 硬件边界、FA4 迁移矩阵和显存账本的完整来源。两者冲突时，
> 性能实施顺序以本文件为准。

## 1. 决策摘要

动态 KV 是容量和启动峰值优化，**不会直接提高单请求 decode**。当前 VMM
随机分页访问与普通 torch 分配在本机实测处于 ±2% 噪声带，增长提交摊销仅
1.7 µs/128 KiB 页。它的性能价值必须通过“更多逻辑并发 → 更大的有效 M”
兑现，而不是用 B1 token/s 验收。

最新 profile 已把真正的计算瓶颈收敛为：

- B1 128K、MTP K=3、全 CUDA Graph：约 **104–108 tok/s**；
- 每轮 39.18 ms，GPU 占 91.8%，host 仅约 8%；
- verify 占轮时 75.2%；
- W8A8 + W4A4 GEMM 占 GPU 约 54%，均接近真冷 DRAM 带宽地板；
- attention 占 B1 GPU 16.5%，生产路径达到约 79–89% DRAM 峰值；
- 因此剩余大杠杆只有：**提高有效 M、减少权重/中间字节、纵向融合**。

按预期价值排序：

1. 预构建 `K={3,5,7}` MTP family，按接受率和 batch 自适应选择；
2. 动态 KV 完成后把逻辑并发从 B4 扩到 B8；
3. decode/verify 合并 q/gate、k、v W8 投影；
4. 保留 torch reduction 顺序的 RMSNorm tail → FP8 activation quant 融合；
5. 空闲单请求 prefill chunk 从 2048 自适应提升至 8192；
6. 再做 ragged/mixed prefill + target verify，共享一次 target 权重扫描；
7. tree/top-k MTP、W8 dense → NVFP4 作为独立高风险研究项。

## 2. 证据基线与口径

### 2.1 Decode

| 指标 | B1 | B4 |
|---|---:|---:|
| round wall | 39.18 ms | 58.42 ms |
| verify GPU | 29.47 ms | 46.46 ms |
| sync GPU | 2.70 ms | 3.37 ms |
| draft GPU | 3.79 ms | 4.64 ms |
| GPU / wall | 91.8% | 93.2% |

来源：[`../notes/2026-08-15-qwen38-phase0-round-attribution.md`](../notes/2026-08-15-qwen38-phase0-round-attribution.md)。

B1 生产 node-trace 的 kernel-family 分解：

| family | ms/轮 | GPU 占比 | 判定 |
|---|---:|---:|---|
| W8A8 FP8 GEMM | 表观 10.64；校正后约 6.9 | 最大项 | 真冷 57–71% 峰值，主要靠减字节/提 M |
| W4A4 MLP | 6.68 | 20.8% | 约 70% 峰值，tile 微调空间小 |
| paged attention | 5.30 | 16.5% | 79–89% 峰值，B1 理论优化上限约 3.3% wall |
| W8 activation quant | 1.27 | 4.0% | 纵向融合候选 |
| RMSNorm | 约 0.95 | 2.9% | 只允许 bit-exact tail 融合 |
| copy/index/elementwise | 约 1.5 | 4.6% | 随 consumer 融合消除 |

来源：
[`../notes/2026-08-16-qwen38-b1-decode-kernel-attribution.md`](../notes/2026-08-16-qwen38-b1-decode-kernel-attribution.md)、
[`../notes/2026-08-16-w8a8-gemm-roofline-bandwidth-floor.md`](../notes/2026-08-16-w8a8-gemm-roofline-bandwidth-floor.md)。

### 2.2 动态 KV 的正确性能语义

VMM 已验证：

- 36 GiB VA 预留不提交物理显存；
- capture 后增页，旧 CUDA Graph 基址保持有效；
- bulk copy/read/write 与随机 128 KiB page-table walk 均在普通 torch
  allocation 的 ±2% 内；
- commit+zero 摊销 1.7 µs/128 KiB 页。

因此动态 KV 的验收分两层：

1. **直接门禁**：同 token 流、同接受率，B1/B4 性能差异在 ±2%；
2. **间接收益**：同一物理显存预算下能否支持更大的逻辑 batch，并提高
   aggregate accepted tok/s。

来源：[`../notes/2026-08-16-vllm-extensible-kv-cache.md`](../notes/2026-08-16-vllm-extensible-kv-cache.md)。

## 3. P0-1：更高且自适应的 MTP K

### 3.1 为什么它是最大机会

当前每轮用一次 `anchor + K` target verify，K=3 满接受时提交 4 token。
小 M 的 W8/W4 GEMM主要支付一次权重读取，verify 的 M 从 4 提到 6/8 不会让
权重流量线性增长；线性增长主要来自 attention、activation 和额外 MTP draft
continuation。因此，更高 K 能让一次昂贵 target 权重扫描产出更多真实 token。

当前 K 被固化在：

- GDN candidate row 数；
- `round()`/`round_batch()` 输入合同；
- verify graph 的 `qo_len=K+1`；
- draft graph 的 `K-1` 自回归 continuation；
- attention verify workspace capacity。

不能在线修改一个整数后重捕获。应参考 SGLang 的 adaptive state pattern，启动时
预构建多套 runtime/graph state，在 round 边界切换：

- `K={3,5,7}`；
- batch buckets 至少 `B={1,2,4,8}`；
- EMA 接受率按 batch bucket 独立维护；
- reject-position histogram 连续恶化时降 K，持续高接受时升 K；
- 不允许在线 capture。

本地参考：
`/home/bot/project/sglang/python/sglang/srt/speculative/adaptive_spec_params.py`、
`/home/bot/project/sglang/python/sglang/srt/speculative/eagle_worker_v2.py`。
SGLang Frozen-KV MTP 尚未直接接入 adaptive，借鉴的是“预建状态并切换”的架构，
不是复制一段可直接使用的 Qwen 实现。

### 3.2 收益模型（待实测）

根据当前 phase 时间，把随 qlen/continuation 增长的部分外推：

| K | B1 满接受轮时推算 | B1 满接受吞吐推算 | B4 aggregate 推算 |
|---:|---:|---:|---:|
| 3 | 39.18 ms（实测） | 102.1 tok/s（轮级） | 273.9 tok/s（轮级） |
| 5 | 45–49 ms | 122–133 tok/s | 304–333 tok/s |
| 7 | 50–56 ms | 143–160 tok/s | 327–372 tok/s |

以上为 `[待验证]` 推算，不是性能承诺。break-even 条件：

- K=5：平均至少接受约 3.8/5，才能超过满接受 K3；
- K=7：平均至少接受约 4.4/7，才能超过满接受 K3。

接受率必须用真实代码、长上下文检索、中文推理和网页生成提示词测量；digit filler
的 100% 接受只用于测硬件上限，不能代表线上分布。

### 3.3 显存成本

每个 GDN state row 为 75.75 MiB。4 slots + scratch 下：

- K=3：20 rows，1.47949 GiB；
- K=5：30 rows，约 2.219 GiB，增量约 0.740 GiB；
- K=7：40 rows，约 2.959 GiB，增量约 1.479 GiB。

另有 graph family private-pool 增量，必须按 target verify、draft、sync 分 family
记录 `memory_allocated/memory_reserved`，不得用 `nvidia-smi` 总差值猜分项。

### 3.4 验收

- `bf diff` 确认 K arms 除 K/graph family 外完全可比；
- greedy 输出必须与 K=3 target token 流完全一致；
- 记录每个 K/B 的 round wall、verify/sync/draft GPU、接受 histogram、
  reject-position histogram、accepted tok/s、graph hit/fallback；
- 满接受 fixture 与至少四类真实 prompt 分开报告；
- 只有 `(1 + avg_accepted) / round_time` 优于 K=3 才保留该 K；
- graph capture 新增显存与启动时间单独报告。

## 4. P0-2：把动态 KV 兑现成 B8 逻辑并发

VMM 只改变 pool 的物理提交方式。若 `num_slots`、GDN rows 和 CG buckets 仍固定
为 4，性能收益必然为零。动态 KV 完成后必须追加逻辑容量阶段：

1. 逻辑 slot 数不再由“每 slot 预留 max_model_len KV”决定；
2. 捕获 `B={1,2,4,8}`，短上下文再评估 B16；
3. GDN state、MTP graphs、page table 和 admission 同步扩展；
4. 使用常见 exact buckets；只有稀有 batch 才向上 padding；
5. KV admission 继续按真实剩余 token 保守预留，不能以超卖换吞吐。

用现有 B1/B4 满接受轮时拟合 `T(B)=32.77+6.413B ms`：

| B | K3 aggregate 推算 | 相对 B4 |
|---:|---:|---:|
| 4 | 273.9 tok/s | 1.00× |
| 8 | 380.6 tok/s | 1.39× |
| 16 | 472.7 tok/s | 1.73× |

这是 `[待验证]` 的局部外推。B8×128K KV 与 B4×256K 相同，约 36 GiB，
是当前 96 GiB 卡上的现实近期目标；B16×128K 的约 72 GiB KV 加非 KV 基座后
不能容纳，因此 B16 只适合明显更短的平均上下文。

正确的动态 KV 性能验收是：在相同显存预算内把 B4 提升到 B8，并让
aggregate decode 从约 265–274 提升到约 380–410 tok/s，而不是要求 B1 变快。

## 5. P0-3：合并 q/gate、k、v W8 投影

### 5.1 当前重复工作

每个 full-attention 调用对同一个 BF16 `hidden_states` 分别调用 q/gate、k、v
三个 W8 linear。每个 linear 内部都重新：

1. 分配/选择 FP8 activation scratch；
2. 动态 per-token quant；
3. 发起一次 W8A8 GEMM。

其中 k/v 的 `N=1024` 只有 16 CTA，真冷带宽仅 16–20% 峰值，但两者合计
绝对成本只有约 0.5 ms/round。把它们单独做微 kernel 上限约 1.3% wall；
合并 q/k/v 能同时解决重复量化、两次 launch 和 k/v 欠填充。

### 5.2 SM120 exploratory microprobe

配置：K=5120，`N={12288,1024,1024}`，E4M3 W8A8，300 MiB L2 flush，
25 次 CUDA-event 中位数。比较：

- baseline：三次 quant + 三次 GEMM；
- shared quant：一次 quant + 三次 GEMM；
- concat：一次 quant + 一次 `N=14336` GEMM。

| M | baseline | shared quant | concat | concat speedup |
|---:|---:|---:|---:|---:|
| 1 | 116.448 µs | 97.024 µs | 71.968 µs | 1.618× |
| 4 | 118.560 µs | 97.568 µs | 71.968 µs | 1.647× |
| 8 | 118.304 µs | 99.520 µs | 72.000 µs | 1.643× |

随机输入下三个输出 slice 在 M=1/4/8 均逐 bit 相同。该 probe 为本轮临时
探索，不作为合入证据；正式实现必须在 bfdiag warm engine 下复测并保留 artifact。

生产每轮约有 23 组 attention W8 投影；按 M=4 差值估算：

`(118.560 - 71.968) µs × 23 = 1.072 ms/round`

相对 39.18 ms B1 round，预计约 **2.7% 轮时 / 2.8% 吞吐**。

### 5.3 实现约束

- loader/post-load 创建一份连续 q/gate+k+v weight 和 scale storage；
- 原 q/k/v 模块改成该 storage 的 slice view，随后释放旧 storage，禁止
  永久保留第二份约 1 GiB 级全模型副本；
- decode/verify `M<=16` 走 concat；prefill 初期保持原三路，避免改变大 M
  数值轨迹；
- concat 输出用 view 切分，不复制；
- 要求 M=1/4/8/16 raw BF16 output exact、完整 greedy token stream exact、
  MTP acceptance exact；
- CUDA Graph 捕获前准备所有 storage/workspace，capture 内不允许首次分配。

## 6. P0-4：bit-exact RMSNorm tail → FP8 quant

### 6.1 安全边界

项目历史已经证明：即使 RMSNorm 只有 1–2 BF16 ULP 差异，也可能通过 64 层
放大并改变 MTP acceptance。因此不能直接照搬一个数学等价但 reduction order
不同的 Triton RMSNorm。

当前安全实现保留 torch 的 `pow→mean→rsqrt` reduction 顺序，并只把最后两个
FP32 multiply + BF16 round 放入 bit-exact `rms_norm_tail`。正确的下一步是扩展
这个 tail，而不是替换 reduction：

1. torch 继续产生完全相同的 `combined_f32` 和 `rstd`；
2. tail 在寄存器中形成完全相同的 BF16 norm value；
3. 基于该 BF16 value 做 max reduction、动态 scale、divide/clamp/E4M3 convert；
4. consumer 只需要 FP8 operand 时不写 BF16 norm 中间张量；
5. 需要 BF16 view 时可同时输出，但必须证明额外 store 的收益仍为正。

TensorRT-LLM 提供了 fused add+RMSNorm+FP8 quant 的结构参考：
`/home/bot/project/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/custom_ops/normalization/triton_fused_add_rms_norm_quant_fp8.py`。
其 NVFP4 warp-cooperative epilogue明确禁用 SM120，所以只能借鉴结构，不能复制
SM100 NVFP4 实现。

### 6.2 探索结果与收益

本轮 exploratory probe 在 H=5120、M=1/4/8 上把完整 zero-centered norm 与
dynamic FP8 quant 合并：现路径约 61–64 µs，单 kernel 约 8–10 µs，样本 FP8
输出一致。它只证明融合在 SM120 上有足够空间，**不证明 full-reduction 实现满足
token parity**。生产候选必须采用上节的“保留 reduction、只融合 tail”方案。

结合 profile 中 quant 1.27 ms、RMSNorm 约 0.95 ms 及部分 copy 流量，保守
E2E 目标为 **B1 +3–5%**。任何非 bit-exact 实现直接淘汰，不以速度换 acceptance。

## 7. P0-5：自适应 prefill chunk

server 当前固定 chunk=2048；backend geometry 支持到 8192。2048 是为已有 decode
请求控制最长 stall 的折中，不应无条件施加给空闲 B1 prefill。

现有 60K 记录：

- backend 8192 内部 chunk：prefill 25.7 s；
- chunk=512：prefill 56.1 s，产生 118 次 forward；
- 512 虽把单次 stall 降低 36×，但总 prefill 变慢 2.2×。

策略：

- 无 active decode：8192；
- 有 active decode：2048；
- decode ITL/queue latency 超门槛时才降至 1024；
- active decode 清空后，下一 chunk 回升到8192；
- 同一请求中 chunk 变化只能发生在 chunk 边界，GDN/KV 状态语义不变。

按已有 forward 数和固定开销，空闲单请求 2048→8192 预计提升 TTFT
**15–20% `[待验证]`**。必须 fresh process A/B；warm engine 不能验证 load-time
或 cold-prefill 参数。

128K+256 的现实意义：当前 TTFT 约 60.5 s、decode 约 2.5 s；即使 decode
提升 50%，总时间也只少约 0.8 s。prefill 提升 15–20% 后，E2E 才可能真正
下降约 16–20%。报告必须同时列 TTFT、decode-only 和 E2E，不能只报其中一个。

## 8. P1：混合负载与高风险路线

### 8.1 Ragged/mixed prefill + target verify

当前 engine 可先推进一个 prefill chunk，再单独执行 active slots 的 MTP target
verify。同一层权重可能在一轮中读取两遍。TP=1 下比 SGLang TBO 更符合本项目的
结构是把 prefill rows 和 `anchor+K` verify rows 打包进一个 target forward：

- W8/W4 GEMM 对 packed rows 一次执行；
- paged attention 用 ragged qo lengths/page tables；
- GDN 用 slot/source/destination indices 区分状态；
- forward 后执行原 acceptance、commit、sync、draft；
- 首版只支持一个 prefill chunk + homogeneous greedy MTP slots。

这是在线混合流量优化，对纯 B1 decode 为 0 收益。按减少一次 target 权重扫描的
比例，服务 aggregate **+10–30% `[待验证]`**，实现复杂度高，排在 P0 后。

### 8.2 Sampled MTP batching

当前只要 batched MTP 中有一个非 greedy slot，整个 batch 会退回逐 slot round。
应增加 batched sampled accept/reject，不让一个 temperature>0 请求串行化整批。
对 greedy 基准无收益；对 sampled 多请求流量预计 +10–30%，必须验证采样分布而非
要求 token exact。

### 8.3 Tree/top-k MTP

SGLang Frozen-KV MTP 已有 top-k speculative tree 和 tree verification 输入构造：
`/home/bot/project/sglang/python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py`。
这可能在高 K 线性链接受率下降时扩大有效 M，但本 runtime 尚无 tree attention
mask/verify contract，属于新架构项目。只有 K5/K7 因尾部拒绝明显达不到
break-even 时才启动。

### 8.4 W8 dense → NVFP4

checkpoint 中约 9.9 GiB FP8-channel dense weight 是剩余大字节来源。转成 NVFP4
理论上可减少约 5 GiB resident weight，并把真实约 6.9 ms/round 的 W8 GEMM
权重流量近似减半；考虑 FP4 quant/kernel 效率，保守 decode 收益目标 +5–9%。

该路线高风险：checkpoint 很可能有意把敏感 dense 投影保留为 W8。必须离线
repack、逐层 logit/top-k 对账、MMLU/code/long-context 质量和 MTP acceptance
全部过门，不能因显存或 microbench 快就默认启用。

## 9. 已证伪或不适用的方向

| 方向 | 结论 |
|---|---|
| 动态 KV 直接提升 B1 | 错误目标；它只改变容量/提交时机，访问带宽已证奇偶 |
| 替换成 FA4 kernel | 不需要且不满足生产 paged/FP8 KV合同；本路线只借鉴执行思想 |
| SM100 tcgen05/TMEM/2CTA UMMA | SM120 物理不具备，不能移植 |
| 再做 generic attention tile/stage sweep | B1 上限约3.3% wall，只有新 NCU 证据显示离开带宽峰值才重启 |
| packed `f32x2` score arithmetic | SASS 已证 compiler scalarize，性能在噪声内 |
| W8 N32 tile | 真冷 A/B 无收益，已回退 |
| W4A16 默认 decode | B1 -24%，B4仅约+5%，并增加7.88 GiB |
| `silu*mul→NVFP4 quant` 再做增量尝试 | 两次因bit parity/收益关闭；除非有新 exact epilogue 设计 |
| SGLang TBO 直接移植 | 主要服务 DP/EP 通信重叠；本项目 TP=EP=1、host gap<8% |
| host async/scheduler shaving | host只约6–8%，不是当前大杠杆 |
| torch.compile 替换手工 CG | 当前热点已在全 CG 中，收益/稳定性比低 |
| custom all-reduce | 单卡不适用 |
| 单独优化 k/v N=1024 | 绝对上限约1.3%；qkv concat 会结构性消除 |
| NVFP4 KV | 质量量化不足，按既有决定暂缓 |

## 10. 分阶段交付与 kill gates

### Phase 0：可比基线

- 固化 128K prompts、输出长度、sampling、GPU clock/pstate；
- `bf diff` 拒绝配置/prompt 不同的比较；
- B1/B4 分别记录 cold/warm；
- trace 必含 phase GPU event、accept histogram、reject-position、CG hit/fallback；
- 报告 TTFT、decode-only accepted tok/s、E2E 三套数字。

### Phase 1：低风险确定性收益

1. qkv concat；
2. bit-exact norm-tail→FP8 quant；
3. idle/concurrent adaptive prefill chunk。

Kill gates：

- 任一 greedy token、W8 raw output、accept histogram 改变：停止；
- qkv concat B1 收益 <1.5%：不扩到更多 shape；
- norm-tail fusion B1 收益 <2%：不接 FP4 consumer；
- prefill TTFT 收益 <8% 或并发 decode p99 ITL 回退 >5%：回退策略。

### Phase 2：有效 M

1. K5；
2. K7；
3. adaptive controller；
4. B8 logical slots/graphs。

Kill gates：

- K arm 不满足 `(1+avg_accepted)/round_ms` 优于 K3：不保留；
- B8 aggregate 相对 B4 <25%：先查 padding/fallback，不扩大 B16；
- 新 graph/GDN 常驻超过预算或破坏 4×256K admission：降低 family 组合；
- 任何 greedy target token drift：正确性失败，不作为“接受率变化”解释。

### Phase 3：服务混合负载

1. sampled MTP batch；
2. ragged prefill；
3. mixed prefill + target verify。

Kill gates：以 production trace 中真实 fallback/重叠占比为前置；若目标路径占总
GPU/服务 wall <10%，不实施大改。

### Phase 4：独立研究

- tree/top-k MTP；
- W8 dense → NVFP4。

两项都必须独立 PR、独立质量报告，不与 P0 性能改动混合。

## 11. 整体目标区间

这些数字是基于当前 Amdahl 与 phase 数据的规划目标，不是已实现结果：

| 完成范围 | B1 128K decode | aggregate decode | Prefill |
|---|---:|---:|---:|
| 只做确定性融合 | 约 114–117 tok/s | B4约280–290 | chunk策略另计 |
| K5/7 + 两项融合 | 约 140–165 tok/s | B4约340–400 | 不变 |
| 再接 B8 | 单请求不变 | B8约500–600（取决于K7接受率） | 不变 |
| 自适应 chunk | decode不变 | 混合流量需另测 | B1 TTFT目标 +15–20% |

单并发或 4×128K 达到 1000 tok/s 不现实。单卡接近 1000 aggregate 需要更短的
平均上下文、B16/B32、K5/7或tree MTP，并继续减少 dense weight bytes。近期可信
目标应先定为：

- B1 128K：**140–165 tok/s**；
- B8 128K：**500–600 aggregate tok/s**；
- 空闲 128K prefill：**TTFT 降低 15–20%**；
- 以上全部保持 greedy token exact、质量门禁和 4×256K 可服务性。

## 12. 更新触发条件

以下任一发生时更新本文件：

- K5/K7/B8 产出可比实测；
- qkv concat 或 norm-tail fusion 完成/被证伪；
- VMM physical KV 接入完成并有 B8 容量数据；
- Qwen3.8 checkpoint/quantization scheme 更新；
- b12x paged attention 或 W8/W4 kernel 的生产 roofline 明显变化；
- 新 profile 将 host、GDN 或 attention 推到 >25% 且离开既有性能地板。
