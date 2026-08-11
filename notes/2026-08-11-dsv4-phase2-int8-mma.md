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
**关键结论（2026-08-11 终）**：Triton 的 `tl.dot` 有硬约束——K≥32（无法 per-code K8/K16
scale）、无张量标量索引/切片（无法提取 codes/mag 的列）、无 per-code 位移（w32 组装只能
用 join，j 字节序需 bit-reverse）——**kernel 内反量化的 exact 数值在 Triton 内不可行**。
Phase 2 的正解是**手写 CUDA m16n8k16**（`runtime/kernels/iq2_mma16.cu`，A/C 已实测确认：
a0→A[lg,l4*4+0..3]、a1→A[lg+8,...]、c0→C[lg,l4*2]、c1→C[lg,l4*2+1]、c2→C[lg+8,l4*2]、
c3→C[lg+8,l4*2+1]）——数值 + 性能都需在 CUDA 侧完成。Triton tl.dot
仅作过渡速度验证（3.3×，kill gate ≤6.5ms 达标，数值不达标）。
**B 布局最终结论（2026-08-11）**：b0 的 4 字节 → B 的 (k,n) 映射是 mma 硬件布局硬编码的；
所有手推假设（n=lg、n=l4*2、k=l4*4+j、k=lg*4+j）实测均失败（c0 恒为固定 B 列或边界
lane 漂移）——**必须从 CUTLASS `mma_tensor_op.h` 的 `IteratorB`（
MmaTensorOpMultiplicandTileIterator，ThreadCount=32，MatrixShape<K,N>）的线程映射逐字节
提取**。这是 Phase 2 剩余的唯一步骤，之后按 notes 完成 fp32 累积 + per-token scale + 全输出 + 接入。
**最新（2026-08-11 深夜）**：唯一值探针最终确认 A/C 布局（a0→A[lg,l4*4+0..3]、
c0→C[lg,l4*2]、c1→C[lg,l4*2+1]、c2/c3→C[lg+8,...]），且 b0 的字节同时供 c0 与 c1
（同一 b0 归约到 C 的两个相邻列）——**B 的 4 字节 = 2 列（l4*2 与 l4*2+1）各 2 个 k，
非 4 个连续 k**；手写 kernel（iq2_mma16.cu）已用 `b |= mag << (8*j)` 但输出行映射
（C 列 l4*2 → out 行）与 B 列的权重行（lg）之间的硬件映射尚未厘清（c0/c1 共享 b0
意味着每 warp 一次 mma 同时算 2 个输出行）。**需按 CUTLASS `mma_tensor_op` 的
FragmentB 精确布局（2 列 × 2 k）重写 B 打包与输出 scatter**。Phase 2 数值+完整 kernel
未达之前不进入 Phase 3。
**决定性（2026-08-11 深夜）**：b0=B[l4*4+j,lg] 时 c0==ΣB[:,l4*2] 达 **32/32**（c0 的 B
列 = l4*2 完全确认），但 b0 的字节值（B[k,lg]）与 c0 的归约（ΣB[:,l4*2]）矛盾——说明
**mma 的 b0 字节→(k, n) 映射有独立的 k 重排**（非 l4*4+j），必须从 CUTLASS
TensorOpMultiplicandCrosswise128x4（sm80）的 k_index/bank 公式逐字节提取。
**b12x 参考**：`frag_layout_swizzle_16b_to_8b`（b12x/_lib/intrinsics.py:3916）用
`shuffle_sync_bfly(offset 1,2) + byte_perm(0x5410/0x3276)` 做 m16n8k32 B fragment 的
lane 间字节重排（16b→8b）；m16n8k16 s8 的 B fragment 需要类似的 lane 间 k/n 映射
（可直接用此 swizzle 的 offset=1,2 版本 + int8 字节）。这是 Phase 2 手写 kernel 的
最后一块。
**进一步定位（2026-08-11 晚）**：无 scale kernel 与 torch mag×xq 的 maxdiff 仍大；
早期 dump 的 codes/g 对比错是 **kb 不对齐**（kernel dump 落在 kb=15，torch 参考用 kb=0）
+ codes 逐字节读取。已改 uint16 读取（`+1` 跳过 d），ROWS/STRIDE/`eid*ROWS*STRIDE`
dump 正确（12124160）。**剩余疑点**：kernel 全 kb 累加的 acc 与 torch 全 cols mag×xq
maxdiff 仍大，需 kb 对齐的 codes dump 逐块核对（或 w32 的 join 在非 kz=0 时的字节序）。
tl.dot int8 本身已验证正确（构造数据 maxdiff 0）。Phase 2 数值未达之前不进入 Phase 3。
**关键结论（2026-08-11 终）**：Triton 的 `tl.dot` 有硬约束——K≥32（无法 per-code K8/K16
scale）、无张量标量索引/切片（无法提取 codes/mag 的列）、无 per-code 位移（w32 组装只能
用 join，j 字节序需 bit-reverse）——**kernel 内反量化的 exact 数值在 Triton 内不可行**。
Phase 2 的正解是**手写 CUDA m16n8k16**（`runtime/kernels/iq2_mma16.cu`，A/C 已实测确认：
a0→A[lg,l4*4+0..3]、a1→A[lg+8,...]、c0→C[lg,l4*2]、c1→C[lg,l4*2+1]、c2→C[lg+8,l4*2]、
c3→C[lg+8,l4*2+1]）——数值 + 性能都需在 CUDA 侧完成。Triton tl.dot
仅作过渡速度验证（3.3×，kill gate ≤6.5ms 达标，数值不达标）。
**B 布局最终结论（2026-08-11）**：b0 的 4 字节 → B 的 (k,n) 映射是 mma 硬件布局硬编码的；
所有手推假设（n=lg、n=l4*2、k=l4*4+j、k=lg*4+j）实测均失败（c0 恒为固定 B 列或边界
lane 漂移）——**必须从 CUTLASS `mma_tensor_op.h` 的 `IteratorB`（
MmaTensorOpMultiplicandTileIterator，ThreadCount=32，MatrixShape<K,N>）的线程映射逐字节
提取**。这是 Phase 2 剩余的唯一步骤，之后按 notes 完成 fp32 累积 + per-token scale + 全输出 + 接入。
**最新（2026-08-11 深夜）**：唯一值探针最终确认 A/C 布局（a0→A[lg,l4*4+0..3]、
c0→C[lg,l4*2]、c1→C[lg,l4*2+1]、c2/c3→C[lg+8,...]），且 b0 的字节同时供 c0 与 c1
（同一 b0 归约到 C 的两个相邻列）——**B 的 4 字节 = 2 列（l4*2 与 l4*2+1）各 2 个 k，
非 4 个连续 k**；手写 kernel（iq2_mma16.cu）已用 `b |= mag << (8*j)` 但输出行映射
（C 列 l4*2 → out 行）与 B 列的权重行（lg）之间的硬件映射尚未厘清（c0/c1 共享 b0
意味着每 warp 一次 mma 同时算 2 个输出行）。**需按 CUTLASS `mma_tensor_op` 的
FragmentB 精确布局（2 列 × 2 k）重写 B 打包与输出 scatter**。Phase 2 数值+完整 kernel
未达之前不进入 Phase 3。
**决定性（2026-08-11 深夜）**：b0=B[l4*4+j,lg] 时 c0==ΣB[:,l4*2] 达 **32/32**（c0 的 B
列 = l4*2 完全确认），但 b0 的字节值（B[k,lg]）与 c0 的归约（ΣB[:,l4*2]）矛盾——说明
**mma 的 b0 字节→(k, n) 映射有独立的 k 重排**（非 l4*4+j），必须从 CUTLASS
TensorOpMultiplicandCrosswise128x4（sm80）的 k_index/bank 公式逐字节提取。
**b12x 参考**：`frag_layout_swizzle_16b_to_8b`（b12x/_lib/intrinsics.py:3916）用
`shuffle_sync_bfly(offset 1,2) + byte_perm(0x5410/0x3276)` 做 m16n8k32 B fragment 的
lane 间字节重排（16b→8b）；m16n8k16 s8 的 B fragment 需要类似的 lane 间 k/n 映射
（可直接用此 swizzle 的 offset=1,2 版本 + int8 字节）。这是 Phase 2 手写 kernel 的
最后一块。K16 内 2 个连续 code 共享同一
nibble（偶数→lo、奇数→hi）已确认 → per-K16 scale 精确。剩余：B 布局定位、fp32 累积、
per-token xscale 的 C 映射（c0/c1 用 token lg、c2/c3 用 token lg+8）、16×8 全输出、
多 warp/grouped 接入、launch wrapper + 数值验证。

## 3. 下一步

**2026-08-11 决定性解析（B 布局谜底）**：PTX ISA §9.7.15.5.9 文档的
m16n8k16 s8 片段布局就是正确的硬件布局——之前所有"手推 B 布局失败"的结论
**全是探针代码的符号扩展 bug**：`(int32_t)(int8_t)(负值) << (8*j)` 会把
0xFFFFFF 垃圾位 OR 进高字节，导致 b0/a1 寄存器字节被污染（实测 lane 25-31 的
a1 高 3 字节全变 0xFF）。修好打包（`& 0xFF` 掩码）后，PTX 文档布局在 SM120 上
**32/32 全对**：

- A：`a0=A[lg, l4*4+0..3]`、`a1=A[lg+8, l4*4+0..3]`（与之前探针一致）
- B：`b0 byte j = B[l4*4+j, lg]`（4 字节全在同一列 lg=groupID，k 连续 4 个）
- C：`c0=C[lg,l4*2]`、`c1=C[lg,l4*2+1]`、`c2=C[lg+8,l4*2]`、`c3=C[lg+8,l4*2+1]`

真实 GGUF/SM120 端到端验证通过：cos=1.0、maxrel≈1e-6（fp32 累加舍入），
覆盖 E=1..3、ROWS=32/64/128/2048、M=8/16、rand/ones/sparse 激活。

**kernel 里的两个真 bug（2026-08-11 已修，`runtime/kernels/iq2_mma16.cu`）**：
1. **B 列错**：旧代码用 `ncol = l4*2` 读权重行，PTX 规定 B 列 = groupID =
   `lg`，所以 b0 字节应来自权重行 `rowblk+lg`（lane lg 读第 lg 行）。
2. **scale 行错**：per-K16 scale（`d*(0.5+nibble)*0.25`）必须取 C 累加所对的
   B 列对应权重行的 d/nibble——c0/c2 对 B 列 `l4*2`、c1/c3 对 B 列 `l4*2+1`，
   两半各自的权重行 scale 不同；旧代码统一用 B 片段行 lg 的 scale，导致除
   lg==l4*2 的 lane 外全部错。

已达成：single-layer p50 ≤6.5ms 的数值前置条件（kernel cos ≥0.9999 的 0.9999
阈值远超满足，实际 1.0）。下一步进入 Phase 2 性能 kill gate 和 grouped 接入。

## 4. 相关提交

- `1eb2c0a` Phase 0 prefill_profile（137 tok/s 基线 + 分项 + route histogram）
- `d788796` Phase 1 layer-major scheduler（greedy 5/5 @ 64/65/256/1024）
