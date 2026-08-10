# ds4 CUDA 内核完整调研：DSV4 M=1 decode 移植目标

> 来源：antirez/ds4 仓库深度调研。本文档是执行层细节的完整归档，
> 下次做 DSV4 M=1 decode 优化前必须先读。

日期：2026-08-10

## 0. 前置事实

- **模型格式**：DSV4 Flash GGUF = attention 投影 `Q8_0`，共享专家 `Q8_0`，
  routed MoE 的 gate/up `IQ2_XXS`、down `Q2_K`，compressor/MTP 投影 `F16`。
- **decode M=1 不用张量核**。D2R（tensor-core MMQ）是 prefill 专用：
  `d2r_min_cols()=1024`（cuda/mmq/ds4_mmq.cu:263），M=1 全走 warp-per-row GEMV。
- 实测数字出自 GB10/sm_121（与 SM120 同代），L2 防缓存旋转手法可信。

## 1. M=1 decode 完整 kernel 栈（每层 43 次）

| 阶段 | kernel | 位置 | 要点 |
|---|---|---|---|
| HC 前 | `rms_norm_plain` + f16 `matmul_f16_splitk` 24×16384 + sinkhorn | proto_m2_hc.cu:120-229 | norm 延后见 §2E |
| QKV 投影 | `dsv4_qkv_rms_norm_rows_q81_kernel`（norm q→1024/kv→512 内联发 q8_1 码）→ `matmul_q8_0_pair_preq_warp8_kernel` | proto_m2_qkv.cu:243；ds4_cuda.cu:5216 | pair 核共享预量化激活 |
| head norm+rope(q) | `head_rms_norm_rope_tail_scalars_kernel`（融合，capture-safe scalars） | proto_m2_qkv.cu:294 | |
| kv 写入 | `kv_rope_fp8_store_scalars_kernel`（rope 只碰 64 尾部，fp8 只量化 448 nope，8 并行 64-thread block） | proto_m2_qkv.cu:359 | 串行组并行化 |
| Attention | `attention_decode_mixed_kernel`（每 (token,head) 一 block）；长 ctx 用 split 在线 softmax | ds4_cuda.cu:7350/7760/9185 | fp32 KV（f16 舍入），sink+raw window(≤256)+compressed |
| attn 输出+共享专家 | `shared_mid_q8_0_preq_warp8_exact_kernel`（gate+up+clamp+SwiGLU 一次） | ds4_cuda.cu:5264 | |
| Routed MoE | iq2 对齐 gate/up/mid 融合 → q2k 对齐 down → `moe_mmq_sum_kernel` | ds4_mmq.cu:2388/3903；ds4_cuda.cu:23657 | §2B/§2C |
| Router | 3-kernel 链融合成 1 个 cooperative 核 | proto_m2_router.cu:268 | §2J |

### 激活量化体系
- **preq（预量化）**：`quantize_q8_0_f32_kernel`（ds4_cuda.cu:4959）量化激活成 xq+xscale，
  所有 `*_preq_*` GEMV 复用同一份。`acc += wscale * xscale[b] * dot(qs, xqb)`。
- **fold**：`ds4_mmq_folded_q81`（ds4_mmq.cu:144）查询注册表命中即跳过 quantize 前奏。

### 1.1 M=1 GEMV 结构（matmul_q8_0_preq_warp8_kernel, ds4_cuda.cu:5074）
```
grid=(out_dim+7)/8 × n_tok, block=256 threads=8 warps, row=blockIdx.x*8+warp
lane: for(b=lane; b<blocks; b+=32){ 34B块=[d][qs]; dot=dp4a; acc+=wd*xscale[b]*dot }
acc=warp_sum_f32(shfl_down 16,8,4,2,1); lane0 写 out
```
- 32 lane/块：每 pass 32 lane 覆盖 32 块=1024 元素，lane 内 34B 连续读 → 满 coalescing。
- 为什么到不了天花板：34B 块 2B 对齐 → int8 码流两次 16-bit load（`get_int_b2`）。
  这就是 **q8_0 170 GB/s 瓶颈的根源**。

### 1.2 DSpark（ds4_cuda.cu:13113）
- draft：`dspark_markov_argmax_kernel` = argmax(logits[i]+dot(w2[i],w1[prev]))，atomicMax+float-ordered key。
- verify：n_tokens=2..8 批处理。`matmul_q8_0_preq_batch_warp8_tok8_kernel`（:5775）
  每 warp 对同一权重行跨 8 token：w4 load 一次，8 个 dp4a 对 8 份激活。

### 1.3 CUDA Graph island 拆分（ds4_cuda.cu:838-1021）
- 每层拆 2 个位置无关 island：island0=HC 前+mix+QKV 投影（到 rope 前）；island1=attn 输出投影→FFN/MoE 尾。
- 48B graph key={il,island,variant,cur_hc,after_attn_hc,after_ffn_hc,attn_norm} 指针。
- warm→capture→replay 状态机，失败 kill 回 eager，永不丢 token。

## 2. 可复制技术（文件:行 + 结构 + 实测）

### 2A. 对齐 SoA 重打包 + warp-per-row GEMV（q8_0 带宽救星，最高 ROI）
布局（proto_q8_aligned.cu:161-167；生产 ds4_mmq.cu:3664-3674）：
```
[__half dq[nblk]] [pad 64B] [int8 qs[nblk*32]]
```
重打包核 `repack_q8_0_aligned_kernel`（ds4_repack.cu:405）。GEMV 核
`q8_0_aligned_dense_vec_kernel`（ds4_mmq.cu:3675）：32 threads/warp/行，int4(16B) load ×2/block，
8×dp4a，dq*ds*sumi。要求 K%1024==0（DSV4 满足）。

**实测（GB10，L2 旋转测法）**：
| shape | baseline | aligned | 提升 |
|---|---|---|---|
| attn_q_b 1024×32768 | 164 GB/s | 235 GB/s | +43% |
| mid 2048×4096 | 157 | 218 | +27% |
| out_a 8192×4096 | 199 | 230 | +16% |
| head 4096×129280 | 146 | 243 | +66% |

verify 宽度变体 `q8_0_aligned_dense_vec_nc_kernel<NC>`（:3717）：权重流每行读一次，
NC 列各对 col-strided 激活 dp4a（激活靠 L1/L2 广播），+17..+87%。

**坑**：L2 trap（<L2 驻留测出假数字）；必须 ≥256MB 旋转足迹。

### 2B. IQ2_XXS 对齐 + gate/up/mid 融合
布局（proto_iq2_aligned.cu:80-83；ds4_mmq.cu:4137-4142）：[dq][pad64][uint2 qs[nblk*8]]。
核 `iq2_xxs_aligned_moe_pair_vec_kernel`（ds4_mmq.cu:2320）：32 lane 每 pass 4 blocks×8 pairs，
iq2xxs_grid 查表 + unpack_ksigns + dp4a + 5bit ls 缩放。
融合 mid（:2388）：双累加 acc_g/acc_u，lane0 epilogue 内 clamp+silu(gate)*up*weights → 直接写 mid。
verify 去重（:2491）：w5 约 40% expert 重叠，first-owner 独占读 → D=18 1.53x。

### 2C. Q2_K down row-pair 对齐
布局（ds4_mmq.cu:3856-3864）：[uint2 dm2][pad64][int4 sc4][pad64][uint2 qs2]。
核 `q2_k_aligned_moe_vec_kernel`（:3903）：warp/列，每 lane 迭代 8B qs + 16B scales + 8B dm，
12 条 load 压到 3 条。实测 raw 154→214 GB/s。

### 2D. 激活量化折叠
V4（proto_m2_hc.cu:705-745）：norm 输出 warp 持有 32 连续列 → 每 warp 每 pass 恰发一个 q8 block；
amax 用 shfl_xor 蝶式，与独立 quantize 核位一致（:1006-1025 校验）。
2a（proto_m2_qkv.cu:243-291）：写 q row 后同 block 回读按 warp-contiguous 发 q8_1 → 省 quantize 前奏。

### 2E. HC 链融合（norm 延后 + cooperative）
数学：`dot(w, x·s) = s·dot(w, x)`。不对 x 先 RMS，直接对 raw x 算 24 dot 部分和 + sumsq 部分和
（hc_stage_dots_kernel, :235），finish 核（:284）统一算 rms scale、乘回、sinkhorn。
V3 单核（:508）：96 blocks×256，3 次 grid.sync()；sinkhorn warp16 并行（shfl_xor）。
**cooperative launch 可被 CUDA graph capture/replay 且位一致**（:1073-1108）。

### 2F. D2R 张量核（仅 prefill，参考）
- CTA m128×n64，8 warps，kStages=2。mma.sync.m16n8k32.s8（mma.cuh:946）。
- 权重布局教训（proto_gemm_dense_q8_d2r.cu:14-19）：全分块紧凑布局比 row-major 慢 27%
  （slice camping）。**重打包必须 row-major**。

### 2G. Router 融合（proto_m2_router.cu）
f16 split-K logits + combine + top-6 select 融合 1 cooperative 核（:268）。V2（:454）
每 warp 拿两个 tile，4 个 w 向量 load 提前 → 双倍 MLP。位一致（10 seed × 3 模式 × 5 grid）。

### 2H. decode 注意力
每 (token,head) 一 block；q 按 head 分、scores 在 shared，raw rows 环形读 fp32 KV。
长 ctx 用 tile512（16 head×16 rows）：q/kv float4 协同载入 smem（stride 516 避 bank 冲突），
严格升序 FMA 链保位一致。

## 3. 对 DSV4 M=1 建议（按 ROI）
① q8_0 对齐 SoA 重打包 + warp-per-row dp4a（+43-66%）
② 激活预量化一次复用（preq/pair）
③ IQ2 gate/up 融合 mid（5 launch→1）
④ norm 生产者 emit q8（fold）
⑤ HC dot 延后 + cooperative
⑥ decode graph island 拆分
⑦ DSpark verify 批量（权重读一次 × N token）
⑧ 权重布局 row-major（避 slice camping）
不做：D2R 张量核（M=1 无收益）；MXFP4/NVFP4 MMA（q8_0/iq2/q2k 用 dp4a 就够）
