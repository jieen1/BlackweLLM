# 阶段四盘点：让解码 kernel 变快的可用杠杆**已经用尽**

> 🔴 **本文的总结已被同日推翻（2026-08-03 晚）。**
> 见 [`2026-08-03-performance-gap-vs-historical.md`](2026-08-03-performance-gap-vs-historical.md)。
>
> 本文的推理是"CG 下已 kernel-bound ⇒ 只能让 kernel 更快 ⇒ 四条路都堵死 ⇒ 没杠杆了"。
> **错在第一步到最后一步之间那个隐含假设：把"已 kernel-bound"当成了"kernel 已最优"。**
> 同模型、同卡，历史实现的有效带宽是 **564 GB/s**，今天是 **343 GB/s**——**约 1.6× 的差距**。
> 所以问题不是"没有杠杆可用"，而是**当前 kernel 组合本身比历史那套慢**。
>
> 下面四条否定（W4A4 / W8A8 / `bf16_gemv` / sparkinfer 无 BF16 稠密 GEMM）**各自仍然成立**，
> 作为证据档案有效；**只有"因此没得优化了"这句结论作废。**

日期：2026-08-03 · 模型：`unsloth/Qwen3.6-27B-NVFP4`（标准模型）

## 前提

[`2026-08-03-decode-kernel-profile.md`](2026-08-03-decode-kernel-profile.md) 证明
**CG 解码已 kernel-bound**（GPU busy 31.01ms / 墙钟 34.67ms = 89%）。
所以再提速只有两条路：**让这些 kernel 更快**，或**少做 kernel**。
本文盘的是前者，结论是**都试完了、都走不通**。

kernel 构成：

| 路径 | 占比 | 覆盖 |
|---|---:|---|
| `W4A16FusedMoeKernel` | 35.0% | 0–55 层 MLP（NVFP4） |
| BF16 通用 GEMM/GEMV | 45.0% | 233 个 FP8 层反量化后走 `F.linear` |
| GDN 递归 | 0.6% | —— 不是量级项 |

## 已关闭的四条

### 1. NVFP4 → W4A4（35%）✗

前提成立（kernel 契约匹配、checkpoint 真声明 W4A4 且发货 `input_global_scale`），
**但数值上就是差**：单层 cosine 0.988 对 W4A16 的 0.99999，B1-R 全线不过，
一个负载发散到溢出 top-1024 捕获窗口。
见 [`2026-08-03-w4a4-blockscaled-negative-result.md`](2026-08-03-w4a4-blockscaled-negative-result.md)。

### 2. FP8 → W8A8（45%）✗

**误差下界**（只注入激活 FP8 往返，不写 kernel）就已经打穿 B1-R——
真实实现必然更差。见 [`2026-08-03-fp8-w8a8-preflight-negative.md`](2026-08-03-fp8-w8a8-preflight-negative.md)。

**1 与 2 的共同点是本文最该记住的一句**：两者都是**把激活降到 4/8 bit**。
这个模型**对激活精度敏感，对权重量化不敏感**（在跑的权重量化路径 cosine 0.99999）。
**别再找更激进的激活量化。**

### 3. `sparkinfer.gemm.bf16_gemv` —— 覆盖面太小，不值得做 ✗

它正是为这类情况写的（"narrow projections 上整块 GEMM tile 浪费 CTA"），
CUDA-graph 安全、带 `precompile`，而且**是 BF16×BF16，不动激活精度**——
所以方向上对，本该是 1/2 撞墙后的首选。**但先量了再说，结果是量不够：**

约束 `SMALL_M_MAX=8`（解码 batch 够用）、**输出维 ≤1024、输入维 ≥1024**。
按 checkpoint 真实形状筛 237 个投影：

| 投影 | out × in | 层数 | 可用 |
|---|---:|---:|:--:|
| `self_attn.k_proj` / `v_proj` | 1024 × 5120 | 16+16 | ✓ |
| `mtp.self_attn.k_proj` / `v_proj` | 1024 × 5120 | 1+1 | ✓ |
| `self_attn.q_proj` | 12288 × 5120 | 16 | ✗ |
| `self_attn.o_proj` | 5120 × 6144 | 16 | ✗ |
| `linear_attn.in_proj_qkv` | 10240 × 5120 | 48 | ✗ |
| `linear_attn.in_proj_z` | 6144 × 5120 | 48 | ✗ |
| `linear_attn.out_proj` | 5120 × 6144 | 48 | ✗ |
| `mlp.{gate,up}_proj`（56–63） | 17408 × 5120 | 16 | ✗ |
| `mlp.down_proj`（56–63） | 5120 × 17408 | 8 | ✗ |
| `lm_head` | 248320 × 5120 | 1 | ✗ |

**只有 34/237（14%）落在覆盖内，而它们恰恰是最小的那些**——
按计算量约占 FP8 GEMM 工作的 **1.7%**，即整个解码 kernel 时间的 **~0.8%**。

**这次是先量后做，几分钟就否掉了。** 对照 W4A4：完整实现完才发现过不了。

### 4. sparkinfer 里没有可用的 BF16 稠密 GEMM ✗

主导形状全是**宽输出**（12288 / 10240 / 17408 / 248320），需要真正的 SM120 稠密 GEMM。
`sparkinfer/gemm/` 逐个看过：`blockscaled`（块量化）、`tensor_fp8_linear`（**静态
per-tensor** FP8，与本 checkpoint 的 per-channel 权重 + per-token 动态激活不符）、
`mxfp8_linear`、`block_fp8_linear`、`trellis_linear`、`wo_projection`、
`mla_query_projection` —— **全是量化 GEMM**，`bf16_gemv` 是唯一的 BF16 入口且限 small-N。

而**任何 FP8/量化 GEMM 都意味着量化激活**，那正是第 2 条已证伪的东西。

自研 kernel 这条路，本仓库自己的调研早有结论：
[`2026-07-31-sm120-flash-attention-kernel-research-for-sparkinfer.md`](2026-07-31-sm120-flash-attention-kernel-research-for-sparkinfer.md)
—— 自研比 sparkinfer 慢 2.4–3×。

## 结论

**解码速度这一侧，短期内没有可动的杠杆了。** 那 24.8% 的
`cutlass_80_wmma`（为 SM80 编译、跑在 cc 12.0 上）**看着刺眼，但没有更好的替代品**：
换成任何量化 GEMM 都会撞上激活精度这堵墙。

**仍然开着、且方向不同的：**

- **FP8 KV cache** —— 省的是**显存不是 kernel 时间**。标准 checkpoint 其实发了
  `k_scale`/`v_scale`（16+16）而我们没接，KV 8192 MiB/槽是常驻里最大的单项，
  FP8 KV 直接减半。见
  [`2026-08-03-production-memory-audit.md`](2026-08-03-production-memory-audit.md)。
- **少做 kernel 而不是让 kernel 更快** —— 投机解码属于这一类，但 MTP 目前
  **根本没接进服务路径**，且 CG 把顺序解码变便宜了 4.71×，**门槛是被抬高的**；
  实测接受率只有 1.20–1.67 / K=4。见
  [`2026-08-03-mtp-acceptance-on-standard-checkpoint.md`](2026-08-03-mtp-acceptance-on-standard-checkpoint.md)。
- **首请求 TTFT 4.67s**（之后 0.25s）—— 已排除是编译（磁盘缓存本就开着），
  尚未定位。这是延迟不是吞吐。
