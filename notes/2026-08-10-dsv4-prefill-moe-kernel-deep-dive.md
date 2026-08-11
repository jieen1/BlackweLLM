# DSV4 prefill MoE 性能深挖：83 tok/s 的根因与 dp4a 正解（2026-08-10）

状态：🔴 prefill 当前 **83 tok/s**（max_q_rows=64 最优），长上下文 128k 冷 prefill 约 25 分钟，
**不可用**。根因已确证，正解（dp4a）已明确但未实施（预计 4-8 小时）。

## 1. 事实基线（真实 GGUF，SM120，96GB）

- prefill（chunked，M=32/64）：**70 → 83 tok/s**（max_q_rows 32→64；96 退化到 60）
- 单层 MoE（M=32）fused kernel（`iq2xs_dequant_gemm_batch_indexed_dual`，E=192 routes, M=1）：
  **4.3ms/层**（dual gate+up）+ 2.3ms（down）≈ 6.8ms/层
- torch.profiler M=32 单 chunk（43 层）：**MoE 289ms / 350ms = 83%**，attn 16ms，HC 14ms
- 带宽账：M=32 → top-8 → 192 routes → **131 唯一专家**，**每专家平均 1.47 token**（最大 4）
  **每 route 输出 gate/up 的 inter=2048 全行**（不是 8 行），每层读激活专家权重
  ~1.5GB（131×2048×4096×0.375B×3）；fused kernel 不落全局的带宽极限 ~5TB/s
  （当前 4.3ms/层 = 反量化计算 2.6% 效率，不是带宽）

## 2. 根因（多路径交叉确认）

**MoE prefill kernel 是反量化计算-bound，不是带宽-bound**：
- `_route_expanded_prefill` 把每 token 的 top-8 展开成 E=192 条 route、M=1，每 block 只算
  1 token × 8 行（`BLOCK_COLS=8`），同一专家权重被多条 route 重复反量化
- IQ2 反量化（`IQ2XS_GRID`/`KSIGNS` 查表 gather + 8-j 位操作）+ fp32 内积
  → 2.6% 计算效率（4.3ms/层 = 1.7 Gops/s vs SM120 ~100 Tops）

## 3. 已尝试并排除的路径（避免重复踩坑）

| 路径 | 结果 |
|---|---|
| M 批处理 kernel（block 内多 token 复用权重，3D acc） | kernel 数值 bitwise 正确（16 测试绿）但 **每专家 1.47 token 无批可复**，E=131 m_pad=2 实测 0.8×（padding 反而 1.4-4× 冗余） |
| expert-grouping 接入（sort/unique/pad + M 批处理） | **8× 退化**（9 tok/s）：`counts.max().item()` GPU→CPU 同步 + m_pad 变化触发 Triton 重新编译；已回滚 |
| BLOCK_COLS 调参（8→16/32/64） | 无影响（4.30/4.14/4.42/4.62ms）——瓶颈不是 block 组织 |
| deq（Triton IQ2→bf16）+ cuBLAS bmm（137 专家批量） | 单层 8.9ms（0.8×）：137 专家 bf16 反量化输出 2.2GB/层 **写带宽 bound**（1.6TB/s，32%），bmm 137 次小 GEMM 也慢 |
| deq（131 专家全量 bf16）+ 大 GEMM（[32,4096]×[4096,272K]） | GEMM 1.47ms（cuBLAS 仅 17 TFLOPS，M=32 计算密度低）+ **17× 冗余**（每 token 只需它的 8 专家，却算全部 131 专家） |
| max_q_rows 增大 | 64 最优 83 tok/s；96 退化（MLA scratch/overhead）；128 OOM |

**共同限制**：(a) 任何"反量化全权重并落全局（bf16/fp32）再 GEMM"的路径被 bf16 写带宽
封顶；(b) 任何 cuBLAS 大 GEMM 被 M=32 小 batch 的低计算密度 + 专家冗余（每 token 只
需 top-8）封顶；(c) M 批处理被"每专家 1.47 token"的固有稀疏路由封顶（唯一专家数太多）。
**必须保持 fused（不落全局），优化其反量化计算（对齐布局去 gather + dp4a int8 内积）**。

## 4. 正解：dp4a int8 内积（ds4 §2B 已验证方案）

参考 `notes/2026-08-10-ds4-cuda-deep-dive.md` §2B（`iq2_xxs_aligned_moe_pair_vec_kernel`）：

1. **激活预量化（preq）**：x [M, K] → int8 xq + fp32 xscale（每 32 元素 1 scale）
2. **IQ2 code 对齐重排**（离线 repack，模型加载时）：当前 74B 交错块 → code 连续对齐
3. **dp4a 内积 kernel**：int8 code × int8 激活 dp4a（一条指令 4×int8 乘加），int32 累加
   → × xscale × wscale（wscale 只每块 1 个，不是每元素反量化）
4. 可选：gate/up/mid 单 kernel 融合（双累加 acc_g/acc_u + epilogue 内 clamp/silu/weights）

**预期**：fused kernel 内积（fp32 标量）约占总时间 2/3（deq 纯反量化 1.4ms vs fused 4.3ms），
dp4a 内积 4× + 对齐去 gather 后 **fused 4.3ms → 1-2ms/层，prefill 83 → 170-330 tok/s（2-4×）**。
**不是数量级提升**：反量化查表（`IQ2XS_GRID`/`KSIGNS` gather）固有，每 route 需全 2048 行输出、
每专家平均 1.47 token 的路由稀疏性、bf16 落全局写带宽三者共同封顶。要 >1,000 tok/s 需要
换模型格式（NVFP4 专家 + tensor-core fused MoE，sglang 路线）或权重预反量化缓存（显存不足）。

实测排除：对齐布局（code 连续）对 **deq kernel 无加速**（1.40 vs 1.41ms，写 bf16 带宽 bound，
读 0.6GB 非瓶颈）；对齐对 **fused kernel**（不落全局、反量化 gather 是计算瓶颈）可能有效但
需接 repack + 改 kernel（数值对齐 bug 未修，方向已否定前不必修）。

## 4b. dp4a 已落地（2026-08-10，commit e543e41）

`tl.inline_asm_elementwise` 跑通 PTX `dp4a.s32.s32`：
- **数值 maxdiff 0**（4×int8 打包成 int32 用 int32 指针读 + dp4a）
- **内积快 2.2×**（4096 int8 pairs：dp4a 6.2μs vs torch fp32 13.4μs）
- Triton 用法要点：`constraints="=r,r,r,r"`（1 输出 + 3 输入），`args=[a, b, acc]`
  （**不含输出占位 $0**），打包用 `a_ptr.to(tl.pointer_type(tl.int32))` 直接读 4 字节

**落地结果**（commit `e543e41`）：
- 单层 gate/up：**4.07 → 1.76ms（2.3×）**；routed cos 0.999928（门禁最严阈值 0.99）
- **端到端 prefill：max_q_rows=64 76 → 129 tok/s（+70%）**；max_q_rows=32 68 → 94
- **门禁 PASS**：worst cos 0.99999988、greedy 39/39（与基线一致）
- 实现：`preq_activation`（torch，每 32 元素 int8+scale）+ `_iq2xs_dequant_gemm_indexed_dual_dp4a_kernel`
  / `_iq2xs_dequant_gemm_indexed_dp4a_kernel`（`tl.inline_asm_elementwise` dp4a.s32.s32，per-code
  scale 在 code 归约前乘）；`_route_expanded_prefill` 切换；decode B1/M=1 数值路径未动
- Triton 陷阱记录：inline_asm args 只含输入（$0 是自动输出）；constexpr `M`/`arange` 需 2 幂；
  2D 张量 inline_asm 逐元素可用；**kernel 调用缺 grid 会误报 "Cannot call @jit outside kernel"**；
  **同名 kernel 编译失败会被 Triton 磁盘缓存**（改名可绕过）

**剩余空间（已系统调研，2026-08-10）**：129 tok/s = 带宽利用 5%（每层读 200 专家 gate+up+down
~1.2GB，带宽极限 ~6,400 tok/s @ M=64）。已实测排除的快速路径：
- **对齐布局**（code 连续）：对 dp4a kernel **无收益**（交错 code 在 74B 块内本就连续；0.84 vs 0.80ms）
- **预计算 code→magnitude 表**（65536×2 int32，去 grid/ksigns 查表）：**更慢 2.5×**（256KB 表 L2 压力）
- **M 批处理 + dp4a**（E=200, M=2）：仅 1.2×（每专家 token 少，3D acc 开销）
- **num_warps**：4 最优（8 更慢）；**BR**：8 最优（16/32 更慢）
瓶颈是 kernel 计算效率（每 block 8 行×1 token 的反量化+dp4a 指令流），非带宽非查表。
**手写 CUDA warp-per-row 已实测**（`runtime/kernels/iq2_warp_row.cu`，sm_120 编译，32 lanes/warp/行）：
0.66ms vs Triton dp4a 0.80ms（仅 1.2×）；smem 查表（grid 4KB+ksigns 512B）无额外改善。
**指令吞吐分析**：每 code 解码+2×dp4a+scale ~30 指令，M=64 gate 420M code = 12.6G lane 指令，
0.66ms（192 routes）→ 约 SM120 指令吞吐的 ~30%，**kernel 已接近指令级峰值**（每 code 的
解码/查表/dp4a 是量化格式的固有指令量）。129 tok/s 是**当前 IQ2 格式 + kernel 的指令级上限**
（带宽只用了 5%，但计算已近饱和）。
**要几千 tok/s 必须减少每 code 指令量或 code 数**：(a) 换量化格式（NVFP4 专家 + tensor-core
fused MoE，sglang 路线——tensor core 吞吐 >> dp4a，且 NVFP4 解码更少指令）；或 (b) 更大量化块
（每 code 更多元素摊薄解码）。两者都是模型格式级变更，非 kernel 微调可达。

**实施顺序建议**：
1. preq 激活量化 kernel（简单，独立）
2. IQ2 对齐 repack（加载时一次）+ dp4a kernel（单专家验证 → routes 批量）
3. 数值门禁：单 kernel vs `dequantize_iq2_xs`（rel < 1e-3），端到端 prefill cos ≥ 0.99
4. 接入 `_route_expanded_prefill`（替换 `iq2xs_dequant_gemm_batch_indexed_dual` 调用）

## 5. 长上下文现状（本目标的另一半，已基本达标）

- 64k/128k 配置加载 + decode graph 捕获正常（显存实测：128k×2 slots KV 仅 1.83GB，
  load 87.4GB → graph 91.0GB → prefill 91.5GB，96GB 卡放得下）
- 长上下文 decode **无 NaN**（60000 步递增 100 步 + graph 30 次重复全过，cos=1.0；
  早前 `longctx_backend_test` 的 NaN 是其 step_time 同 pos 重复 30 次的测试假象）
- prefix cache：same-slot 命中 1.4s（冷 855s → **600×**）；跨 slot 命中在 server 流程未验证
- **唯一短板 = prefill 速度**（本 notes 主题）

## 5b. 并发 128k 内存边界（2026-08-10 实测，96GB 卡）

| 配置 | load | capture 后 reserved | 结论 |
|---|---|---|---|
| 2 slots × 128k | 90.3GB | 95.0GB | ✅ decode graph 捕获成功 |
| 3 slots × 128k | 90.3GB | 96.0GB | ❌ capture 失败（`None`，回退 eager） |
| 4 slots × 128k | 90.3GB | **98.5GB** | ❌ 超 96GB（OOM 风险） |

**4 并发 128k 的硬障碍 = decode graph 池**（每 slot 的 B=1/2/4 bucket kernel 中间张量；
3 slots 比 2 slots 的 capture 增量 ~1GB，4 slots 共超 7GB over 2 slots）。KV 本身很小
（4×128k 仅 3.66GB）。模型 88GB + 4 slots graph 池 > 96GB。

**解除路径**（按 ROI）：
1. decode graph 池优化（跨 slot/bucket 共享中间张量、按需 bucket 捕获）——深水区，收益最大
2. `QSR_SERVER_GPU_MEM_UTIL` / 负载调参（降 max_q_rows 等）——收益小
3. 3 slots 128k（B=1/2 只捕 2 bucket）——接近但 capture 失败需先省 graph 池

SoA 存储（`4d1751c`）已省 7GB，否则连 2 slots 128k 都紧张。

## 6. 相关提交

- `4d1751c` Store Q8_0 weights as aligned SoA planes（省 7GB 显存，4 slots/131072 不再 OOM）
- `6493d87` IQ2 kernel M 批处理能力（M=1 位级兼容，为 dp4a/对齐铺路）
