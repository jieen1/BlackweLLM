# DSV4 注意力噪声归因与 bf16-QK 优化尝试（2026-08-08）

状态：🟡 归因闭环；优化实现 85%，topk>2 掩码问题未解（fork worktree `sparkinfer-wt-dsv4-bf16q`，分支 `work/dsv4-bf16q-20260807`，提交 `58d5208`）。

## 1. 归因（闭环）

fork 的 compressed_mla decode kernel（FlashInfer decode_dsv4 移植）的 QK-NoPE 用
**MXFP8 m16n8k32 MMA**：q 在 smem staging 时被 `s0_quantize_q_to_smem` 量化成
**FP8 e4m3**（per-head 64-tile、pow2-ue8m0 scale，与 pack kernel 同语义），K 保持
e4m3 原字节（SFA/SFB 硬件选择器折入 scale）。官方 reference 的 q 是 BF16 ——
fp8-q 是 kernel 的性能设计，偏离官方契约。

归因实验（kernel vs 参考，真实量级 q/k，64 头 × 8 token）：

| 参考变体 | cos | 结论 |
|---|---|---|
| 纯 fp32 | 0.99972481 | 基线噪声 |
| +q-fp8 仿真（act_quant_simulate 语义） | 0.99983692 | **q-fp8 确认，占 ~40%** |
| +P-fp8 仿真 | 0.99959487 | 更差 → XV 侧不是主源（sm_p_full bf16 staging 与 fp32 同吻合） |
| +q-fp8 + base-2 exp2 链 | 0.99983692 | base-2 数学等价 |

残余 ~1.6e-4 = kernel 内部 MMA 累加/exp2f 舍入，未再细分。

## 2. 优化实现（fork 改动）

`SPARKINFER_MLA_DSV4_BF16_Q=1`（默认关，编译 spec 含 key，不改变现有行为）：

1. `traits.py`：`UnifiedMLATraits.dsv4_bf16_q: bool = False`
2. `smem.py`：bf16 模式下 Q-NoPE 区域 0 字节（smem 预算装不下 bf16 staging —— 实测
   +7.4 KB 超限 1.6 KB，改为 **Q 完全不进 smem，QK A 操作数逐 lane 从 global 读**）
3. `decode_math.py`：`s1_qk_nope_dsv4_bf16` —— A=Q（global 逐 lane，PTX m16n8k16
   片段布局）+ B=K（e4m3 寄存器反量化 × ue8m0 scale，与 eager 往返逐位一致）；
   `s0_load_q_bf16_to_smem` 加 `stage_nope=False`
4. `kernel.py`：env 读取 → traits replace → H8 路径（`_kernel_body` 与 `kernel` 两个
   body！）与 generic dispatcher 的 bf16 分支

## 3. 调试链踩坑记录（都值得记住）

- **编译缓存 key 不含 decode_math 源码**：多次"修改无效"的假象，清 `~/.cache/sparkinfer`
  后才发现。fork 的 `compile_cache_info()` 可查 hits/misses。
- **fork 测试的 cosine 门禁是尺度不变的**：`p×0.5` 标记测不出来，不能作为验证信号。
- **`ld_global_nc_v2_u32` 返回二元组**：不解包直接给 MMA → 静默 garbage。
- **探针走 H8 路径（per_token_len=True 时）**：H8 的 s1 直接调 mxfp8 专用函数，
  绕过了 generic dispatcher —— 两个 body（`kernel`/`_kernel_body`）都要接分支。
- **H8 的 staged_kv_stride = 592（打包行），不是 464**：`kv_smem_stride` 必须用
  `staged_kv_stride`。

## 4. 已验证

- topk=2、32/64 头：kernel vs 参考 **cos 0.999993–0.999997**（fp8 基线 0.99972）
- 门禁基线（3×512，eager vs kernel 路径）：**流一致 1512/1539 = 98.2%**，
  最终 logits cos 最差 0.82（漂移 dips 振荡、非指数）

## 5. 未解（下一步）

- **topk>2 时 A=K 方向（实测可工作）的 a1/a3 行半（候选 8..15）逃过共享 s3 掩码**；
  A=Q 方向与 s3 语义一致但 mma 输出不流动。两者各差一步，需聚焦一轮：
  a) 给变体传 `section_len`/候选有效性，直接掩掉无效行半；
  b) 或复刻 H8 的 mxfp8 变体的行掩码语义。
- 之后：fork 基准（QK 是小的 MMA，bf16 全局读的带宽代价要实测）→ 重跑门禁对比
  （预期 logits cos 0.82 → ~0.9+、流一致 98.2% → 99%+）。
