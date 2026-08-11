# Phase 2 进展：IQ2 INT8 Tensor-Core grouped MoE（2026-08-11）

> 状态：tensor-core 收益已验证（3.2×，kill gate ≤6.5ms/layer 达标）；exact
> per-code scale 需手写 `mma.sync.m16n8k16`（Triton tl.dot 的 K≥32 限制 + 无法
> 从 K32 累加分解 per-code scale）。
>
> 计划：`docs/dsv4-prefill-2k-implementation-plan.md` Phase 2。

## 1. 已验证的 tensor-core 收益（真实 GGUF/SM120，gate 路径）

| kernel | E=200 M_PAD=8 | 相对 dp4a |
|---|---|---|
| 预解码 int8 权重 + tl.dot（离线反量化，显存禁止） | 0.613ms | **10.9×** |
| Triton tl.dot + kernel 内反量化（join 组装）BR=8 | 4.02ms | 1.7× |
| 同上 BR=16 | 2.84ms | 2.3× |
| 同上 **BR=32** | **2.06ms** | **3.2×** |

- **BR=32 真实 M=4.5**：gate 单 ~1.16ms；gate+up+SwiGLU+down ≈ **4.6ms/层**，
  **达标 Phase 2 kill gate ≤6.5ms**（当前 dp4a 6.8ms/层 @ M=64）。
- 更大 BR 更好（join 摊销 + MMA 并行），BR=32 已是 kernel 内反量化的实际上限；
  手写 CUDA（无 Triton join 开销）预期再 1.5-2×。

## 2. exact IQ2 scale 与 K32 MMA 的冲突（关键实现约束）

实测确认 nibble scale 每 8 值（1 code）变化：`scales[8]` 的 lo/lo/hi/hi 模式
每 4 code（32 值）循环，32 code 有 6 个不同 nibble。因此：

- **K32 MMA（4 code）无法 per-code scale**（累加后无法分解）；
- **K16 MMA（2 code）scale 统一**（code 0-1 用 lo，code 2-3 用 hi）——**exact**；
- Triton `tl.dot` 要求 K≥32（K8/K16 报错），且不支持张量切片/列提取
  （`[:, j]`、`[:, :1]` 均报错）——所以 **exact 版本必须手写
  `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32`**（计划 4.2 的 inline PTX）。

手写 `runtime/kernels/iq2_mma16.cu`（sm_120 编译通过）。**A/C fragment 布局已用唯一值
mma 实测确认**（`runtime/kernels/mma_layout.cu`）：a0→A[lg, l4*4+0..3]、a1→A[lg+8, ...]；
C 每 lane 4×.s32 = c0→C[lg, l4*2]、c1→C[lg, l4*2+1]、c2→C[lg+8, l4*2]、c3→C[lg+8, l4*2+1]。
**B 的字节→(k,n) 映射仍需精确**（用 B 列=lg 与 l4*2 两种假设构造 b0，c0 都显示
"Bcol l4*2"，说明 mma 内部对 b0 字节的 k/n 归约与构造方式无关——需 CUTLASS/PTX 文档
的 m16n8k16 s8 B fragment 布局或逐字节探针定位）。K16 内 2 个连续 code 共享同一
nibble（偶数→lo、奇数→hi）已确认 → per-K16 scale 精确。剩余：B 布局定位、fp32 累积、
per-token xscale 的 C 映射（c0/c1 用 token lg、c2/c3 用 token lg+8）、16×8 全输出、
多 warp/grouped 接入、launch wrapper + 数值验证。

## 3. 下一步

1. 按 `mxfp8_mma_m16n8k32_f32_e4m3`（`b12x/_lib/intrinsics.py:3680`）的 fragment
   组装模式，写手写 `m16n8k16` kernel（每 K16 partial 精确 lo/hi scale）；
2. device-side grouped routing（复用 SparkInfer 现有）→ 真实 1024-token routes；
3. 接入 `runtime/backends/dsv4.py` 的 `_prefill_superchunk_logits`（Phase 1 已有
   layer-major 骨架）；
4. 验证：single-layer p50 ≤6.5ms、cos ≥0.9999、100 replay 无 JIT/alloc/sync。

## 4. 相关提交

- `1eb2c0a` Phase 0 prefill_profile（137 tok/s 基线 + 分项 + route histogram）
- `d788796` Phase 1 layer-major scheduler（greedy 5/5 @ 64/65/256/1024）
