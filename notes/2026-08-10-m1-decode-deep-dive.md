# M=1 decode 深度调研：Laguna 全路径 / FA4 执行层 / SGLang DSV4 / ds4 CUDA

日期：2026-08-10

## 背景

前一轮调研（`2026-08-10-m1-decode-bandwidth-survey.md`）确认 M=1 权重读取是瓶颈，
但错误地把 170 GB/s 归因于"DRAM latency-bound"。本轮四路深度调研推翻该结论，
并给出**根因**与**已验证的解法**。

## 关键结论（交叉印证）

### 1. q8_0 170 GB/s 的真实根因：34B 交错布局的对齐问题

- **ds4 实测**（GB10/sm_121，与 SM120 同代）：q8_0 块是 34B（2B fp16 scale + 32B int8 code），
  int8 码流只有 2B 对齐 → 每 32-bit 字被迫两次 16-bit load。
  对齐 SoA 重打包后：attn_q_b 164→235 GB/s（+43%）、mid 157→218（+27%）、
  out_a 199→230（+16%）、head 146→243（+66%）。
- **Laguna 反证**：M=1 MoE 在 occupancy 10.3% 下打满 1200 GB/s →
  "170 GB/s" 是 load 宽度/对齐问题，不是 latency 或 occupancy。
- **L2 测量陷阱**：权重缓冲 < L2 会驻留，测出 931 GB/s 假数字。必须用 ≥256MB 旋转足迹。

### 2. 正确的 Q8_0 对齐 SoA 布局（ds4 proto_q8_aligned.cu:161-167）

```
[__half dq[nblk]]        ← 所有 fp16 scale 独立成区
[pad 到 64B]
[int8  qs[nblk*32]]      ← 所有 code 连续，64B 对齐，每 block 32B
```
M=1 GEMV 用 **warp-per-row**（1 warp/行 × 32 threads），每 pass 32 lane 覆盖 32 个
block = 1024 元素，int4(16B) 对齐 load ×2/block，`__dp4a`。要求 K%1024==0（DSV4 满足）。

### 3. 激活预量化（preq / fold）

- 每层 RMS-norm 后把激活量化成 q8_1（xq+xscale）一次，q_a/kv/shared/o_proj 全部复用。
- 生产者在 norm 内核里直接 emit 量化（fold），省独立 quantize kernel + 全局回读。

### 4. MoE 融合

- IQ2 gate/up 对齐布局 + silu(gate)*up*weight epilogue 内联 → 5 launch 压成 1。
- verify 去重：w5 时 ~40% expert 重叠 → 1.5x。

### 5. HC 链融合

- `dot(w, x·s) = s·dot(w, x)`：norm 延后到 matmul 之后，dot 打 raw x + 顺带 sumsq，
  finish 核统一算 rms scale。cooperative launch 可进 CUDA graph（已验证位一致）。

### 6. FA4 执行层可移植点

- 软 exp2（magic 6291456 + add.rm + poly + 位拼）、scale_subtract_rowmax 一条 packed FMA、
  条件 rescale 跳过（M=1 省一整趟 O pass）、多级 KV smem + producer warp 超前 cp.async、
  128-bit + assume 对齐、epilogue 融合归一化。

### 7. SGLang 结论

- SGLang 无"魔法 M=1 kernel"；破墙靠 DSpark/MTP verify 批量（M=γ+1）+ FP8/NVFP4 减字节。
- SM120 用 FlashInfer `sparse_mla_sm120_decode_dsv4` 或 Triton tiled（flash_mla_sm120_triton.py 可抄）。

## 行动（按 ROI）

1. **q8_0 对齐 SoA 重打包 + warp-per-row dp4a**（+43-66%，最优先）—— 之前用 tensor-core
   tl.dot 实现失败（内存 OOM），本次用 ds4 验证的离线重打包 + dp4a 方案
2. **激活预量化一次复用**（preq/pair）
3. **IQ2 gate/up 融合 mid**（iq2xs MoE 10.4ms）
4. **HC dot 延后**（norm 移到 matmul 后）
5. **FA4 条件 rescale / 软 exp2**（软降 softmax 开销）
