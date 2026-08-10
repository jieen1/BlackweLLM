# Laguna 性能优化完整路径调研

> 来源：notes/ Laguna 全部笔记 + runtime/backends/laguna*.py + runtime/kernels/* + git log。
> Laguna 证明：M=1 带宽赤字根因是 **kernel 结构 + 量化布局 + 图内元数据**。

日期：2026-08-10

## 0. 一句话结论

**图内元数据（零 Python/零 H2D）+ 单 kernel 融合 + 一次性权重布局重排** 三条，
DSV4 只有第一条已抄到位（argmax 烤进图），后两条是 15% 利用率→40% 的主要抓手。

## 1. 优化时间线

| 阶段 | 提交 | 内容 | 效果 |
|---|---|---|---|
| CG decode | 442fa91 | CG decode 落地 | 66.3 tok/s，3.2× vLLM |
| | 783617e | argmax 融入 CG | 每步 host 读一个 token |
| SWA+DFlash | bf30f76 | SWA 环形 KV | 36 滑窗层 9→0.04 GiB/槽 |
| | 2b41ae4 | verify-only 投机 | 62.9→190.2 tok/s（3×） |
| MoE 去 vLLM | 3e3b989 | B12x NVFP4 MoE 单 kernel | 融合 routing+2GEMM+SiLU+quant+reduce |
| 带宽压榨 | 21acd6d | per-step buffer GPU 预计算 | 15.06→13.95ms/step |
| | 073cac8 | Triton fused RMSNorm+residual | 12.95ms/step |
| DFlash 大提速 | 1443143 | verify _fill_buffers 向量化（消 1109 标量写） | 45→253 tok/s（5.6×） |
| 带宽账 | 07-27 | roofline 方法论 | MoE 100%、dense 86-95%、attn 37% |
| 服务器 | 09d35c97 | 预分配 fill_buffers | 无 per-round 分配 |
| 闭环 | 07-31 | MoE 带宽 100% 饱和 | "No tuning helps. Only batch amortization, L2 locality" |

## 2. 关键技术可复用细节

### 2.1 CUDA Graph decode 元数据烤进图（laguna_cuda_graph.py）
- argmax 烤进图（:460）：capture 内 `self._input_ids[0] = self._logits[0].argmax()`。
- B=1 fastpath `_fill_buffers_b1`（:279）：常量缓存、window 对齐内联。
- 固定地址纪律：输入 buffer __init__ 预分配，KV cache capture 前 _bind_kv_caches。

### 2.2 融合元数据 kernel（cg_decode_metadata.py）
`write_laguna_b1_decode_metadata`（:45）：host 写 [token_id, kv_len] 到 pinned（一次 non_blocking copy），
一个 1-block Triton kernel 写 6 个元数据张量（input_ids/positions/2×slot_mapping/2×cache_seqlens）。
SWA 对齐/ring 取模在 kernel 内。

### 2.3 向量化纪律（5.6× 教训）
verify _fill_buffers 曾用 Python for 逐元素填 page_table，64K 下 1109 次标量写 = 180ms CPU 调度。
向量化后 verify_replay 205.7→37.83ms。

### 2.4 fused_kv_scatter（fused_kv_scatter.py）
grid=(num_tokens, num_kv_heads)（:108），每 program 一 head 一行：load K/V（各自 stride）→ fp32 →
除以 scale → fp8 → scatter 进 paged cache。6 op→1 kernel（288→48 kernels/step）。

### 2.5 fused_rms_norm（fused_rms_norm.py）
一行一 program，一次 load x+residual 到寄存器，val=x+res 直接存新 residual，寄存器内归约求 rstd。
每张量一读一写。num_warps=8。

### 2.6 MoE 单 kernel（laguna_sparkinfer_moe.py）
38μs/层（CG M=1）vs eager 186μs/层。ncu：M=1 **1200 GB/s、occupancy 10.3%** —— 带宽饱和。
`SparkinferMoEOutputArena`（:307）：grow-only 输出 buffer，47 层共享。

### 2.7 带宽账方法论（07-27 note）
1. 实测卡带宽 ceiling（torch.empty 2GB + copy_ ≈1300 GB/s）。
2. 每 kernel 用真实 shape 算字节/耗时 = 有效 GB/s。
3. **必须冷缓存**：L2 128MiB 会污染（紧凑循环权重常驻 L2 测出 1122 GB/s 假数字）。
   "eager 比裸 F.linear 慢 2.4×"就是 L2 污染的假结论（flush 后 19.9→44.86μs）。

**对 DSV4 推论**：q8_0 "170 GB/s / DRAM latency-bound" 前提要先验证。Laguna 的 M=1 MoE
在 10% occupancy 打满 1200 GB/s → 低 GB/s 更可能是 load 宽度/对齐结构，不是 latency。
（后经 ds4 确认：确实是 34B 对齐问题。）

### 2.8 DFlash M>1 摊销
verify M=16 一次 forward 读一遍权重。M=16 只比 M=1 贵 2.9×（不是 16×）。
单轮固定开销（accept_reject_sync/bookkeeping/kv_precompute）不能流水线（draft 需 CPU 决策）。

### 2.9 图捕获容量
- 在代表性上下文而非最大容量捕获（更短→更小块→更多有效 CTA）。
- MultiBatchGraphManager：每 batch size 各捕获一张图，不 padding 空跑。

## 3. 对 DSV4 建议（ROI 排序）
R1 MoE down 折进 fused kernel（iq2xs ~10.4ms 最大单点）
R2 Q8_0 对齐 SoA 重打包（q8_0 18.5ms @30%）
R3 逐层数清 elementwise 融合
R4 确认热循环无 Python 标量写（向量化/上 kernel）
R5 split-K 推广大输出投影
R6 Q8_0→FP8 e4m3 原生 MMA（B 计划）
R7 推测解码 DSpark/MTP（系统级最大杠杆，需显存+MLA 回滚）
