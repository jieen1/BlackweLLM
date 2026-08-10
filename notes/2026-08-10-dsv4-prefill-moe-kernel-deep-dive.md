# DSV4 prefill MoE 性能深挖：83 tok/s 的根因与 dp4a 正解（2026-08-10）

状态：🔴 prefill 当前 **83 tok/s**（max_q_rows=64 最优），长上下文 128k 冷 prefill 约 25 分钟，
**不可用**。根因已确证，正解（dp4a）已明确但未实施（预计 4-8 小时）。

## 1. 事实基线（真实 GGUF，SM120，96GB）

- prefill（chunked，M=32/64）：**70 → 83 tok/s**（max_q_rows 32→64；96 退化到 60）
- 单层 MoE（M=32）fused kernel（`iq2xs_dequant_gemm_batch_indexed_dual`，E=192 routes, M=1）：
  **4.3ms/层**（dual gate+up）+ 2.3ms（down）≈ 6.8ms/层
- torch.profiler M=32 单 chunk（43 层）：**MoE 289ms / 350ms = 83%**，attn 16ms，HC 14ms
- 带宽账：M=32 → top-8 → 192 routes → **131 唯一专家**，**每专家平均 1.47 token**（最大 4）
  每层读激活专家权重 ~1GB（gate+up+down 137×2.4MB×3）；HBM 极限 → **~3,800 tok/s**
  （当前 83 = 带宽/计算利用 ~2-4%）

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
| deq（Triton IQ2→bf16）+ cuBLAS bmm | 单层 8.9ms（0.8×）：137 专家 bf16 反量化输出 2.2GB/层 **写带宽 bound**（1.6TB/s，32%），bmm 137 次小 GEMM 也慢 |
| max_q_rows 增大 | 64 最优 83 tok/s；96 退化（MLA scratch/overhead）；128 OOM |

**共同限制**：任何"反量化全权重（137 专家）并落全局（bf16/fp32）再 GEMM"的路径都被
bf16 输出写带宽封顶在 ~几百 tok/s。**必须不落全局、直接量化内积**。

## 4. 正解：dp4a int8 内积（ds4 §2B 已验证方案）

参考 `notes/2026-08-10-ds4-cuda-deep-dive.md` §2B（`iq2_xxs_aligned_moe_pair_vec_kernel`）：

1. **激活预量化（preq）**：x [M, K] → int8 xq + fp32 xscale（每 32 元素 1 scale）
2. **IQ2 code 对齐重排**（离线 repack，模型加载时）：当前 74B 交错块 → code 连续对齐
3. **dp4a 内积 kernel**：int8 code × int8 激活 dp4a（一条指令 4×int8 乘加），int32 累加
   → × xscale × wscale（wscale 只每块 1 个，不是每元素反量化）
4. 可选：gate/up/mid 单 kernel 融合（双累加 acc_g/acc_u + epilogue 内 clamp/silu/weights）

**预期**：MoE prefill 计算 4-8× 减（int8 vs fp32 内积 + 免 per-element 反量化），
prefill 83 → **400-1,500 tok/s**（接近带宽极限 3,800 的 10-40%）。

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

## 6. 相关提交

- `4d1751c` Store Q8_0 weights as aligned SoA planes（省 7GB 显存，4 slots/131072 不再 OOM）
- `6493d87` IQ2 kernel M 批处理能力（M=1 位级兼容，为 dp4a/对齐铺路）
