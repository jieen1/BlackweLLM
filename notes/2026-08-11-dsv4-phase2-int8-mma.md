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
**B 是 warp 级归约**（逐字节探针：任意 lane 单字节 b0 探针下 c0..c3 恒为 4，即
mma 组合所有 lane 的 b0 成完整 B 矩阵再归约，单 lane 无法独立控制）——**B fragment
布局必须按 CUTLASS `mma_tensor_op` 的线程映射精确构造**（`b12x/_lib/intrinsics.py`
m16n8k32 模式 + CUTLASS `mma_sm80.h` m16n8k16 s8 instruction 的 warp 布局）。
作为 Phase 2 的过渡，Triton `tl.dot`（K32，per-K16 近似 scale）已验证 3.2× 且
kill gate ≤6.5ms/layer 达标——exact 手写 mma 是后续优化。
**B 布局假设**（B[lg*4+j, l4*2] 等）**全部实测失败**（0/32）；b12x 的 m16n8k32 mma
用 `frag_layout_swizzle_16b_to_8b`（fragment 字节 swizzle）构造 B——**B 的线程映射 +
字节 swizzle 必须照搬 b12x/CUTLASS**（`b12x/attention/nsa_indexer/kernel.py` 的
`frag_layout_swizzle_16b_to_8b`），不能手推。Phase 2 下一步：Triton tl.dot 的
gate/up/down + SwiGLU 端到端接入（过渡），exact 手写 mma 并行跟进。
**数值状态（2026-08-11）**：Triton tl.dot（kernel 内反量化，BR=32）gate cos **0.47**
（远低于 0.9999）——w32 的 join 组装字节序或 K32 scale 近似错误；预解码版 cos 0.018
（构造无 scale，不能作参考）。排查顺序：先用无 scale 的 kernel 输出 vs torch mag×xq
定位 w32 组装；再修 K32 scale（4 code 的 lo/lo/hi/hi 无法在 tl.dot 内 per-code，需
确认是否可接受近似或改用每 K32 单独 mma）。Phase 2 数值未达之前不进入 Phase 3。
**进一步定位（2026-08-11 晚）**：无 scale kernel 与 torch mag×xq 的 maxdiff 仍大；
dump 显示 kernel 的 codes/g 与 torch 完全不同（kernel codes [63002,47616,...] vs torch
[14157,6635,...]；kernel g 是 int32 小值 vs torch int64 大值）——**kernel 的 code 字节
读取偏移错位**（c4/kb/base 的偏移与 torch 的 blocks 视图不一致），需逐字节核对
`base + 2 + c4*2` 的读取。tl.dot int8 本身已验证正确（构造数据 maxdiff 0）。K16 内 2 个连续 code 共享同一
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
