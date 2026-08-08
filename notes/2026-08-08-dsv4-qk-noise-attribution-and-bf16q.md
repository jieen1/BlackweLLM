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

## 5. 已解决（2026-08-08 第二轮）—— topk>2 根因与验证

**根因**：`_kernel_body`（per_token_len 内核）的 s0 分发只认 NVFP4 —— bf16-q
模式下 DSV4 仍走 `s0_quantize_q_to_smem`，往 **0 字节的 q staging 区域**写
fp8 量化 q，**污染相邻的 q_rope smem** → s2（rope QK）读垃圾 → 分数巨大
（LSE ~78）→ topk>2 输出 garbage。修复：两个 kernel body 的 s0 分发都加
`or t.dsv4_bf16_q` + `stage_nope=(not t.dsv4_bf16_q)`（fork 提交 `82d12f3`）。

**验证**：
- 真实尺度探针（8 行 × 64 头 × 8 token）：kernel vs 纯 fp32-q 参考
  **0.99972 → 0.99986**（q-fp8 噪声消除）；topk 2/8 均 0.99999+
- fork 套件带/不带 env 均 21/21
- 模型级门禁 smoke（16 步，bf16-q vs eager）：
  **最差逐层 0.816 → 0.942、最终 logits 0.980 → 0.986**（漂移显著减小）
- 残余 ~1.4e-4 噪声 = XV 侧（P/V 处理），非 QK

**后续**：fork 基准（bf16 全局读的带宽代价实测）→ 完整门禁对比（后台跑）→
若 XV 侧也想清，查 s6 的 w_fp8（fp8 重量化 P）语义。
