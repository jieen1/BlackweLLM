# DSV4 注意力噪声归因与 bf16-QK 优化尝试（2026-08-08）

状态：🟢 QK 与 XV 两侧噪声均已消除（fork 提交 `679c921`）；模型级完整门禁后台运行中。

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

**XV 侧噪声源（s6_xv_nope 已确认）**：DSV4 的 XV 把 softmax 权重 W 重新量化成
e4m3 存入 w_fp8（per-head w_head_sc 后缩放），走 PLAIN fp8 MMA —— 与 QK 侧的
q-fp8 同构的 3-bit 尾数量化。NVFP4 模式已有 bf16-PV 变体
（s6_xv_nope_nvfp4_bf16）但不适用于 DSV4（V 是 e4m3 非 E2M1）。给 DSV4 做
bf16-PV 需新 s6 变体（bf16 P × e4m3 V 寄存器反量化）—— 工作量同 QK 变体，
是下一个可选增量（预期 0.99986 → ~0.9999+）。

**后续**：fork 基准（bf16 全局读的带宽代价实测）→ 完整门禁对比（后台跑）→
若 XV 侧也想清，按 s6 的分析做 bf16-PV 变体。

## 6. bf16-PV XV 变体完成（2026-08-08 第三轮）

**实现**（fork `work/dsv4-bf16q-20260807`，提交 `679c921`）：新的
`s6_xv_nope_dsv4_bf16` 不再读 S5 的 sm_p_full（H8 路径上 S5 的 fill 根本不跑，
c6d6bc4 提交的变体因此读 garbage），而是像 `s6_xv_nope_dsv4_h8_swap_ab` 一样
从 swapped-score w_pre 片段自建普通 [head, candidate] BF16 sm_p_full，barrier
后再做 BF16 MMA（A=P 直接 bf16，B=V 逐 lane 寄存器反量化 e4m3×ue8m0）。

**修掉的真 bug —— 逐候选 scale**：初版（含 c6d6bc4 与 14:23 快照）只加载
candidate `ent0` 的 footer scale 字节，却把它套用到 **四个不同候选**
（ent0, ent0+1, ent0+8, ent0+9）。footer 布局是每候选独立 7 字节 ue8m0 块
（与 QK 侧 `s1_qk_nope_dsv4_bf16` 的逐候选索引一致），所以 v1/v8/v9 全用错
scale。修复为四个候选各读各的 scale 字节。CG 重放探针（32 头，DSV4 页）：
**kernel vs 参考 cos 0.971610 → 0.999996**（fp8 基线 0.999991），eager 与
CG 重放 bit 一致。fork 套件带/不带 env 均 21/21。

**16:31 那次调试为什么没找到它**：`branch_probe.txt` 从未被写出 —— cute.jit
kernel body 是从磁盘编译缓存服务的，`decode_math.py`/`kernel.py` 的改动根本没
进执行的内核（note §3 的坑第二次踩中）。清 `~/.cache/sparkinfer/compile` 后
改动才真正生效，一测就定位到 scale bug。

**eager 门禁路径的 OOM 陷阱**：工作区里试过的 expert 反量化 LRU 缓存
（`PackedIQ2_XSExperts._expert_cache`，cap 48/模块）装不进显存 ——
43 层 × 3 模块 × ~7 路由专家 × 32 MiB fp32 ≈ 28 GiB 常驻，压在 84.9 GiB
（模型 81.9 + kernel 层）之上，余量只有 ~11 GiB，首个 eager forward 即 OOM。
已回退（`8d3c4d9`），门禁脚本默认步数定为 128（512×3 在 eager 路径上约 2.5h，
只留作最终确认）。

**状态**：模型级 3×512 完整门禁（bf16-q + bf16-pv vs eager）后台运行中，
以它为准。
