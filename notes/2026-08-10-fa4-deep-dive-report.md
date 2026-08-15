# FA4 深度调研报告 —— Blackwell 注意力内核设计穷尽式拆解（面向 SM120 M=1 Decode / MLA / MoE）

> **2026-08-15 时效更新**：本文的算法拆解仍有效，但上游能力快照已从早期 commit 更新到
> `fa4-v4.0.0.beta26`。beta21 已加入 SM120 Pack-GQA；beta26 的 SM120 forward 仍复用 SM80
> `mma.sync`，仍不支持 paged KV、SplitKV 和 FP8 KV，不能直接承载 Qwen3.8 的生产形状。
> 最新的 Qwen3.8 显存/性能映射和实施门禁见
> [`../docs/qwen38-sm120-cuda133-fa4-optimization-plan.md`](../docs/qwen38-sm120-cuda133-fa4-optimization-plan.md)。

> 日期：2026-08-10
> 资料来源：`/home/bot/project/fa4-latest`（Dao-AILab/flash-attention，commit `c75d019`，
> main 落后 origin 12 个 commit，可 fast-forward，与工作树内容一致），
> 以及 `assets/fa4_paper.pdf`（20 页，已用 pypdf 全文提取）。
> 方法：全部基于实际源码逐行阅读 + 论文交叉验证。**本报告纯调研，未写任何代码。**
> 与已有旧笔记（`fa4-sm120-adaptation-analysis.md`、`fa4-sm120-port-research.md`、
> `fa4-sm120-portability-research.md`，均为 2026-07-31）的区别：那三篇是结构扫描；
> 本篇是 9395 行关键源码的逐点拆解 + 可移植性论证，并落到本项目的 M=1 decode / MLA / MoE 场景。

---

## 0. 结论摘要（TL;DR）

1. **SM120 在 FA4 里就是 SM80 fallback**，没有任何 SM100 专属内容。`flash_fwd_sm120.py` /
   `flash_bwd_sm120.py` 各自只有 55~61 行，是 `FlashAttentionForwardSm80` /
   `FlashAttentionBackwardSm80` 的子类，只把共享内存容量从 163KB 改成 99KB、把
   `self.arch` 强制成 `sm_80`，MMA 用 `warp.MmaF16BF16Op(...,(16,8,16))`（`flash_fwd.py:589-599`）。
   与我们 runtime 用的标准 `mma.sync` 指令**完全同族**。所以 FA4 对 SM120 没有直接可复制的
   SM100 kernel。可迁移的是 warp specialization、软 exp2、条件 rescale、LPT、TMA pipeline 等
   算法/调度思想；依赖 `tcgen05`、TMEM 和 2-CTA joint UMMA 的数据流**不能**映射成等价实现。

2. **FA4 论文三大核心**（`assets/fa4_paper.pdf`）：
   - 针对 Blackwell **非对称硬件扩展**（tensor core 吞吐翻倍，smem 带宽与 MUFU 吞吐没涨）重排
     流水线，最大化 MMA 与非 MMA（softmax / 搬数）的重叠；roofline 证明 128×128 tile 下
     MMA 与 exp 单元都是 1024 cycles 瓶颈（`fa4_paper.txt` 第 244-262 行）。
   - **软 exp2**：用 Cody-Waite + 多项式在 FMA 单元上模拟 `2^x`，与 MUFU 并行，分担 exp
     瓶颈；只模拟每行 10-25% 的元素（freq/res 可调，`flash_fwd_sm100.py:99-110` 的调参表）。
   - **条件 rescale**：online softmax 里只有当新的 row_max 比旧的大超过阈值 τ（默认
     `log2(256)=8`）才 rescale O；否则跳过，最后用真实最大/和一次归一化（`softmax.py:293-297`，
     `flash_fwd_sm100.py:2542` 的 `vote_ballot` 检测）。
   - 反向额外用 tmem 存中间结果降 smem 流量、2-CTA MMA 把 dQ 的全局 atomic 减半、确定性
     模式用 semaphore 串行化 dQ 归约。

3. **对本项目最值钱的 5 个点**（详见 §6）：
   ① **M=1 decode 下用「Q 留在寄存器 + KV 走单级流水 + 一次性读到 smem」的 ping-pong 结构**
   ——FA4 的 mma 循环把 QK 与 PV 两个 GEMM 交错发射，天然适合 decode 的每步单 token；
   ② **P 写成输入 dtype 而非 FP32**（`tSrP_r2t` recast 成 q_dtype 再写回 tmem/smem）——
   直接砍掉 PV GEMM 的 B 流量一半，bandwidth-bound 的 M=1 decode 收益极大；
   ③ **软 exp2 模拟 + f32x2 packed FMA**（`utils.py:760-820`）——数学结构可迁移，但没有
   可靠公开证据证明 SM120 MUFU 比 SM100 更慢，只有 profiler 显示 MUFU 饱和时才实施；
   ④ **条件 rescale 跳过 online rescale**——省掉 O 的向量乘（对 M=1 是整条 512 宽的行）；
   ⑤ **LPT 调度 + 头部 swizzle**（`tile_scheduler.py:393-647`）——decoding 不同请求 KV 长度
   差异大时减少负载不均衡，FA 作者实测 MQA 场景 7-14% 提升。

4. **SM100 专属、完全不能移植的**：tcgen05.mma（`tcgen05.mma.cta_group::N` 指令，MLA 用
   cta_group::2）、tmem 及 `tcgen05.copy.Ld32x32bOp/St32x32bOp`、`ld.red` rowmax 硬件归约
   （`use_ldred_rowmax`，SM103 专属）、smem descriptor（`mma_sm100_desc.py` 的 idesc/smem_desc
   64/32 位打包）和 2-CTA joint UMMA。**更正**：SM120 支持普通 thread-block cluster/DSMEM；
   TMA multicast 在 PTX 功能上也可用，但官方优化/建议 target 不含 SM120，并警告其他 target
   可能显著降速。因此两者是“可实验但不能复刻 FA4 数据流”，不是物理缺失。

---

## 1. FA4 论文核心与整体架构

### 1.1 论文核心贡献（`assets/fa4_paper.pdf`，全文提取见 `/tmp/fa4_paper.txt`）

| 贡献 | 位置 | 要点 |
|---|---|---|
| 非对称硬件扩展论 | 摘要、§1、§2.2 | B200 tensor core 8192 ops/clk/SM（Hopper 的 2 倍），但 smem 读带宽仍 128 B/clk、MUFU 仍 16 ops/clk。B300/GB300 的 MUFU 才翻倍到 32 ops/clk。结论：**瓶颈从 MMA 转移到 smem 流量与 exp** |
| roofline 分析 | §3.1.1, §3.2.1 | fwd 128×128×128：MMA 1024 cycles、smem 768、exp 1024（表 1）。M=256 时全翻倍。bwd 128×128×128：smem 3328 cycles 超过 MMA 2560（表 3）；2-CTA M=256 时降到 2688（≈ 超 MMA 5%） |
| 全异步 MMA + 大 tile 流水线 | §3.1.2 | 两个 softmax warpgroup 各 128 线程、每线程整行（消除 inter-warp shuffle 归约 rowmax）；P 经 tmem 传，rescale 挪到独立的 correction warpgroup，从关键路径剔除 |
| 软 exp2 | §3.1.3 | `2^x = 2^⌊x⌋·2^{x−⌊x⌋}`，整数部分用 IEEE754 指数位 bit 操作，小数部分 Sollya 多项式（Horner+FMA）。degree-3 在 BF16 量化误差内与 MUFU 几乎无差（表 2）。只模拟 10-25% 元素 |
| 条件 rescale | §3.1.4 | 只在 `m_j − m_{j−1} > τ`（τ=log2 256=8）时 rescale，避免 warp 发散用 vote_ballot |
| 2-CTA MMA | §2.2、§3.2.3 | M=256 输出按 M 分成两个 CTA 各 128，B 矩阵按 N 分半、每 CTA 只在自己 smem 里 stage 一半；dQ 步用 DSMEM 交换一半 dS，把归约轴拆到两个 CTA，**全局 atomic 减半** |
| 确定性反向 | §3.2.4 | semaphore 锁串行化 dQ 全局归约；SPT（shortest-processing-time-first）序防首写阻塞；批外、头内 swizzle |
| LPT 调度 | §3.3 | 经典 LPT（longest-processing-time-first）：causal 下把 mblocks 逆序处理（先长后短），heads 按 L2 容量分段 swizzle，批次最外。实测 MHA 4-8%、MQA 7-14%（H200） |
| CuTe-DSL（Python） | §4 | 全 Python JIT，编译时间 fwd 2.5s vs FA3 55s（22×）、bwd 1.4s vs 45s（32×）；PTX 内联逃生舱 |

对 B300 的直接提示：论文写明 B300/GB300 的 MUFU 已经翻倍到 32 ops/clk——**我们 SM120 的
exp 瓶颈比论文测的 B200 场景要轻**，但 smem 带宽瓶颈依旧。

### 1.2 `flash_attn/cute/` 模块结构

`__init__.py` 只导出 `flash_attn_func` / `flash_attn_varlen_func`（走 `interface.py`）。

- **入口/调度**：`interface.py`（3146 行）—— arch 解析（`_parse_arch_str`，支持
  `FLASH_ATTENTION_ARCH` env override，`interface.py:68-94`），按 `arch//10` 分派到
  SM80/SM90/SM100(MLA)/SM120/hd256 专用 kernel（`interface.py:838-977`）。
- **fwd 族**：`flash_fwd.py`（SM80，`FlashAttentionForwardSm80`）、`flash_fwd_sm90.py`、
  `flash_fwd_sm100.py`（标准前向）、`flash_fwd_mla_sm100.py`（DeepSeek MLA/DSA）、
  `sm100_hd256_2cta_fmha_forward.py`（hd=256 专用）、`flash_fwd_sm120.py`（SM80 子类）、
  `flash_fwd_combine.py`（split-KV 合并）。
- **bwd 族**：`flash_bwd.py`（SM80）、`flash_bwd_sm90.py`、`flash_bwd_sm100.py`（4172 行）、
  `flash_bwd_mla_*.py` 三件套、`sm100_hd256_2cta_fmha_backward*`、`flash_bwd_sm120.py`。
- **共享组件**：`softmax.py`（online softmax + 低精度 scaling + exp2 模拟）、`mask.py`
  （`apply_mask_sm100` 等，1711 行）、`block_info.py`（因果/局部窗口的 n_block 界计算）、
  `seqlen_info.py`、`tile_scheduler.py`（persistent / LPT / CLC 调度）、`pipeline.py`
  （PipelineTmaUmma / PipelineUmmaAsync / PipelineAsyncUmma 等自定义流水线）、
  `blackwell_helpers.py`（tcgen05 MMA 的 PTX 封装）、`mma_sm100_desc.py`（指令/内存描述符
  位打包）、`pack_gqa.py`、`paged_kv.py`、`topk_gather_kv.py`、`block_sparse_utils.py`、
  `block_sparsity.py`、`compute_block_sparsity.py`、`utils.py`（含软 exp2 多项式）、
  `copy_utils.py`、`cache_utils.py`、`named_barrier.py`、`barrier.py`、`bench_utils.py` 等。

**fwd / bwd / MLA 的关系**：MLA kernel（`FlashAttentionMLAForwardSm100`）不是标准 fwd 的子类，
是独立实现，但复用 `softmax.SoftmaxSm100`、`topk_gather_kv.CpasyncGatherKVManager`、
`paged_kv.PagedKVManager`、`pack_gqa`、`tile_scheduler`。它固定 hdim=64、hdimv=512、
`use_2cta_instrs=True`（MQA-128 / DSA 假设），是**DeepSeek MLA 吸收式 + topk 稀疏的完整实现**。

---

## 2. 逐文件深入分析

### 2.1 `flash_fwd_sm100.py`（3202 行）—— 标准前向

#### 2.1.1 warp 分工（16 warps / 512 threads）

`flash_fwd_sm100.py:279-298`：

```
softmax0_warp_ids = (0,1,2,3)     # 4 warps 读 S、算 softmax、写 P
softmax1_warp_ids = (4,5,6,7)     # 4 warps，处理第 2 个 Q tile（q_stage=2 时）
correction_warp_ids = (8,9,10,11) # 4 warps，读 O 累加器、条件 rescale、写 O
mma_warp_id = 12                  # 1 warp 驱动 UMMA（tcgen05），同时管 QK 和 PV 两个 GEMM
epilogue_warp_ids = (13,)         # 1 warp TMA store O
load_warp_ids = (14,)             # 1 warp TMA load Q/K/V
empty_warp_ids = (15,)            # 1 warp 空转（register dealloc 兜底）
```

每个 warp 用 `setmaxregister_decrease/increase` 动态调整寄存器上限：
`num_regs_softmax`（如 192）、`num_regs_correction`（如 80）、`num_regs_other = 512 − 2×softmax − correction`（`flash_fwd_sm100.py:334-353`）。16 warps 时寄存器预算 512，
12 warps 时 504（MLA 里 `num_regs_per_thread = 168`，`flash_fwd_mla_sm100.py:160-161`）。

#### 2.1.2 tmem 布局与 P/S/O 复用

`flash_fwd_sm100.py:319-332`（hd=128，512 列 tmem）：

```
tmem_s_offset = [0, 128]            # S0, S1 各 128 列（FP32 累加器）
tmem_s_to_p_offset = 64             # P 从 S 内偏移 64 列开始（重叠）
tmem_p_offset = [64, 192]           # P0, P1（与 S 共享，P 是输入 dtype 更瘦）
tmem_o_offset = [256, 384]          # O0, O1 各 128 列
tmem_total = 512
```

即：S（FP32）占 0-127 / 128-255，P（输入 dtype，如 FP16，128 列只需 64 个 FP32 列宽）
重叠放在 S 内偏移 64 起；O 占 256-511。这对应论文 §3.1.2 的「两个 S tile 与 P 重叠」方案
——**tmem 只 512 列，S 用 FP32 是 P 的两倍宽，所以 P 塞进 S 的右半**，而不是另开一整块。
`tmem_vec_offset` 存 row_max/row_sum（`flash_fwd_sm100.py:331-332`）。

#### 2.1.3 mma（QK/PV 双 GEMM ping-pong 流水，`flash_fwd_sm100.py:1573-1865`）

- MMA 用 `sm100_desc.mma_op_to_idesc` 编码指令描述符，`gemm_ptx_precomputed_varname` /
  `gemm_ptx_partial` 发内联 PTX（`blackwell_helpers.py:396-616`）。关键手法：
  `elect.sync _|leader_thread` 让 warp 里**只有 1 个线程**发 `tcgen05.mma`，其余线程
  纯等待；一条 asm 内把 K 维的多次 MMA **用 smem desc 地址递增展开**（`offset_a[k]`），
  免去循环开销（`blackwell_helpers.py:474-522`）。
- 循环结构（persistent，`mma` 方法内 `while work_tile.is_valid_tile`）：
  1. **prologue**：对每个 Q stage 等 Q、等 K，发 `GEMM_QK (Q_i × K_0 → S_i)`（`1715-1740`）；
  2. **主循环**（`1751-1816`）：每个 KV block 迭代里，对每个 stage 交替发
     `GEMM_PV (P_i × V_j → O_i, accumulate)` 和 `GEMM_QK (Q_i × K_{j+1} → S_i)`——
     **PV 和下一个 QK 交错发射**，让 tensor core 在 softmax 处理上一个 S 的同时做下一个
     QK。这正是论文 §3.1.2 的 ping-pong 图（图 1）。
  3. **epilogue**（`1823-1855`）：最后一个 V、O 累加 + `pipeline_o_acc.producer_commit`。
- 2-CTA 模式（`use_2cta_instrs`）下 `mma_tiler_qk = (2·m_block, n_block, hdim)`、
  `mma_tiler_pv = (2·m_block, hdim_v, n_block)`（`193-194`），`cta_group=2`、cluster (2,1)，
  MMA 用 `tcgen05.mma.cta_group::2`（`blackwell_helpers.py:503`）——一个 CTA 的 warp 发起，
  tmem 累加器按 M 分两半、两个 CTA 各持一半，B 按 N 分半各 stage 一半。

#### 2.1.4 softmax（`softmax_loop` 1894-2263、`softmax_step` 2265-2413）

`softmax_step` 单步做：
1. 等 S（`pipeline_s_p_o.consumer_wait`），用 `tcgen05.copy.Ld32x32bOp` 从 tmem 把
   (128,128) S 拉进寄存器（`1952-1959`）；SM103 用 `LdRed32x32bOp` 硬件顺带算 rowmax
   （`use_ldred_rowmax`，`231-236`、`2326-2331`）。
2. 软 rowmax 归约：`update_row_max`（`softmax.py:314-331`），用 **rescale_threshold**：
   当 `acc_scale_ >= -rescale_threshold` 时**跳过 rescale**（`softmax.py:293-297`）。
3. `scale_subtract_rowmax`：`fma_packed_f32x2(acc, scale_log2, bias)`——**f32x2 packed FMA
   一次处理两个分数**（`softmax.py:342-358`）；FP8 时 bias 加 `max_offset=8`（×256），把
   P 的动态范围上移（Note [Low Precision Scaling]，`flash_fwd_sm100.py:84-89`、`2047`）。
4. `apply_exp2_convert`（`softmax.py:360-402`）：**软 exp2 与 MUFU exp2 按
   `ex2_emu_freq / ex2_emu_res / ex2_emu_start_frg` 混用**（`flash_fwd_sm100.py:99-110`
   的调参表），结果**直接 recast 成输入 dtype**（`tSrP_r2t`，`2376-2381`）再 St32 写回
   tmem——**P 在 tmem 里就是 FP16/BF16/FP8，PV GEMM 的 A 操作数流量直接减半**。
5. `split_P_arrive`：P 的前 3/4 列写完后先 `consumer_release` 通知 mma 开始 PV，剩余 1/4
   写完再 release（`flash_fwd_sm100.py:180-183`、`2394-2409`）——**软流水，把最后几列 P 的
   计算时间与 PV GEMM 重叠**。
6. `update_row_sum`（`softmax.py:333-341`）用 `fadd_reduce`。

#### 2.1.5 correction warpgroup（`correction_loop` 2416-2865）

- 每迭代先等 softmax stats（smem `sScale`，`flash_fwd_sm100.py:727`），
  `scale = sScale[tidx + stage*m_block]`；`should_rescale = vote_ballot_sync(scale < 1.0) != 0`
  （`2542`）——**整个 warp 只要有一行需要 rescale 就一起 rescale**（避免发散），
  从 tmem 读 O、乘 scale、写回（`correction_rescale`）。条件 rescale 让大多数迭代
  跳过这段（默认 scale=1）。
- 最终缩放+写 O：`row_sum`、`row_max` 从 smem 读，`rcp_approx` 归一化、乘 `v_descale`、
  recast 到 o_dtype，TMA store（`2579-2613`、`_store_O_to_gmem` 2867+）。

#### 2.1.6 load（`load` 1350-1571）

- Q/K/V 默认走 **TMA**（`PipelineTmaUmma`），1 个 load warp（warp 14）发全部 TMA；
  KV stage 数 `kv_stage` 由 smem 预算推导（`372-380`，`min((224KB − q/o)/每stage, 32)`）。
- `paged_kv_non_tma` 时用 `cp.async` + 128 线程（load_warp_ids=(14,15)，`309-311`），走
  `PagedKVManager`（`paged_kv.py`）：按页表算地址、`shuffle_sync` 广播页指针、每行
  `cp.async` 拷 16B（`paged_kv.py:136-247`）。SM100 的 V 在 gmem 里**预转置成
  (dv, page_size, num_pages)**，PagedKV 里 `compute_X_ptr` 分转置/非转置两分支
  （`paged_kv.py:158-171`）。
- `uneven_kv_smem`（hd=192/128 3-stage）：`smem_large, smem_small, smem_large` 交错布局，
  第 1 stage 靠 phase 加/减偏移寻址（`flash_fwd_sm100.py:384-398`、`offset_kv_smem` 3115）。

#### 2.1.7 调度（`tile_scheduler.py`）

`SingleTileLPTScheduler`（`tile_scheduler.py:393-647`）：static persistent + LPT。
关键实现：`l2_minor=swizzle`（每 section 能塞进 L2 的 head 数）、
`l2_major_divmod`，坐标映射用 FastDivmod 位运算；mblocks **逆序**（`554` 附近
`reverse block order`）实现「先长后短」。`FmhaStaticTileScheduler` 也是 LPT 变体。
`ClcDynamicPersistentTileScheduler`（CLC）是 Blackwell 硬件调度（`interface.py` 里
`use_clc_scheduler` 默认开）。

### 2.2 `flash_fwd_mla_sm100.py`（3160 行）—— DeepSeek MLA 前向（本项目最相关）

#### 2.2.1 输入张量与三组 GEMM

签名（`flash_fwd_mla_sm100.py:352-365`）：
`mQ(s,q,h,hdim=64)`、`mQv(s,q,h,hdimv=512)`、`mK(s,k,h_k,64)`、`mV(s,k,h_k,512)`、
`mO(s,q,h,512)`，可选 `mIndexTopk(topk)`、`mP`、`mRowMax`、`mPageTable`、`mCuSeqlens*`。

三组 UMMA（`196-219`）：
```
mma_tiler_QK  = (128, 128, 64)    # S += Q × K^T，A 从 smem（Major K）
mma_tiler_QvV = (128, 128, 256)   # S += Qv_i × V_i^T（hdimv 分 2 半，每半 256）
mma_tiler_PVt = (128, 256, 128)   # O_i = P × V_i（hdimv 分 2 半）
```
cluster (2,1)，`cta_tile_m = 64`，所以每 CTA 64 行、cluster 128 行。**S 累加器在 tmem 里
同时接收 QK 和 QvV 的贡献**（`mma` 里 `mma_QK` 与 `mma_QvV` 都 `zero_init` 到同一 stage，
`flash_fwd_mla_sm100.py:2328-2359`）——这就是 MLA 吸收式：Q 是 64 维的压缩 query，
QvV^T 是低秩 attention 打分（DeepSeek 用 latent 压缩 KV 做 attention bias）。

#### 2.2.2 warp 分工（16 warps，`110-142`）

```
softmax = (0,1,2,3)   epilogue = (4,5,6,7)   load = 8   mma = 9
clc_scheduler = 10    relay = 11（仅 cpasync 模式）
cpasync_load = (12,13,14,15)（仅 use_cpasync_load_KV 时，即 topk/paged 路径）
```

`relay`（`1348-1360`）：cpasync 生产者 `consumer_wait` 后由 `elect_one` 转发
`pipeline_mma.producer_commit`——**cp.async 的 mbarrier 和 UMMA 的 mbarrier 是两套
信号，中间加一个 relay warp 转接**。这是 FA4 里 cpasync 路径特有的流水线结构。

#### 2.2.3 DSA / topk-gather 路径（`load_cpasync` 1363+、`topk_gather_kv.py`）

`is_topk_gather` 时强制 `pack_gqa`、`qhead_per_kvhead == 128`（MQA-128）、
`use_cpasync_load_KV`（`77-80`）。`CpasyncGatherKVManager`（`topk_gather_kv.py`）：
- 每线程读 `topk_indices_per_thread = tile_n // 128` 个 topk 索引（`topk_gather_kv.py:102`）；
- 构造 bitmask：`1 << lane_idx` 按 warp 内 32 个 topk 有效位做 `warp_reduce(add)`，
  lane0 写 smem `sBitmask[warp_idx][stage]`（`163-192`）——**用位图代替每个无效索引
  单独掩码**，softmax 侧按位展开置 `-inf`（`mask.py:649-657`）；
- `compute_X_ptr` 把 topk 索引转成 gmem 指针（`195-216`），`load_X` 用
  `shuffle_sync` 把指针广播给同行的 4 个线程（每行 16B = 128bit 拷贝单位，`218-278`）。
- `disable_bitmask` 开关：当上层保证 topk 索引都在界内时省掉 bitmask 流水线
  （`82`、`53-71`）。

#### 2.2.4 cluster 内 rowmax 归约与 2-CTA 协作（`softmax_step` 2764-2873）

- 每 CTA 处理 64 行，cluster 128 行。rowmax 归约**跨 CTA**：每线程把自己的局部 rowmax
  写 smem `sRowMax[tidx%64, warp_idx//2]`，`softmax_barrier.arrive_and_wait()` 后读另一
  个 CTA 的值取 max（`2817-2825`）——因为 S 是 cluster 共享的 128 行，两 CTA 各算一半。
- P **写 smem 而非 tmem**（`2851-2855`：`cute.copy(smem_store_thr, rP_smem_view,
  sP_smem_view)`），然后 PVt 从 smem 读 P。原因：S/O 已经占了 tmem 384 列（`237-251`），
  P 从 smem 走可以给 mma 更大的回旋；且 P 是输入 dtype 天然就是 smem 布局。这与标准 fwd
  的「P 在 tmem」是**同一个理念的两种落点**——把 P 的 dtype 压成输入精度（128 列 FP16
  只占 64 个 FP32 列宽）。
- 2-CTA 下 V 的 stage 数 `num_stages_V = 4`（`226`），V 按 hdimv 分 2 半、cluster 内再
  分 2 半，每 CTA 只 stage 自己的 `hdimv/4` 块（`mVt_cur` 的 tiled_divide/logical_divide，
  `1482-1486`）。

#### 2.2.5 correction/epilogue（`correction_loop` 2875-3137）

- 输出 hdimv=512 分两个 256 半，各存 tmem `tmem_offsets_O = [128, 256]`（`245-248`）。
- 每个 n_block 先等 softmax stats，`should_rescale = vote_ballot(scale < 1.0)`（`2988`），
  需要时 `correction_rescale` 用 `mul_packed_f32x2` 双元素乘（`3139-3160`）。
- 最后一轮：`row_sum = row_sum0 + row_sum1`（两 CTA 各 64 行求和，`3041-3045`）、
  `rcp_approx` 归一化、乘 v_descale、recast 到 o_dtype、**每个 CTA 用 TMA 各存自己那半**
  （`3083-3131`），store 前 `cp_async_bulk_wait_group(1-split)` 双缓冲等待。

### 2.3 `sm100_hd256_2cta_fmha_forward.py`（1918 行）—— hd=256 专用 2-CTA 前向

- 只支持 (256,256)、tile 128×128、2-CTA、TMA paged KV（page_size==128）（`60-79`）。
- `qk_mma_tiler = (256, 128, 128)`（QK 一次算 256 行 = 两 CTA 各 128），K 维 128 分两
  段 `iterations_qk=2`；PV 同理（`97-109`）。
- **KV 用 TMA multicast**：`tma_partition(..., block_in_cluster_coord_vmnk[1], ...)` ——
  K/V 在 cluster 里只加载一次、硬件广播给两个 CTA（`908-925`）。这是 2-CTA 相比 1-CTA
  的核心省带宽手段。
- **K/V 共享同一物理页**：paged 路径把 K[i] 的 page_idx 存起来给 V[i-1] 用，省一次
  页表 gmem 读（`1002-1028`），并且**提前一个迭代预取下一页表项**隐藏 L2 延迟（`1009-1011`）。
- MMA 用 `cute.gemm` + `tcgen05.Field.ACCUMULATE` 开关（而非裸 PTX），QK/PV 与 softmax
  ping-pong（`1151-1312`）。softmax 用 `need_apply_mask = step == end_count-1` 的
  「最后一列才做 seqlen 掩码」优化（`1393-1395`）。

### 2.4 `flash_bwd_sm100.py`（4172 行）—— 反向思路（略读）

- 5 个 GEMM：S^T=KQ^T、dP^T=V·dO^T、dV=P^T·dO、dK=dS^T·Q、dQ=dS·K；tmem 里
  **S/P 共享一块（offset 0），dP/dS/dQ 共享另一块**（论文 §3.2.2）。
- `compute_loop`（2882+）：新流水把 **上一迭代的 dQ/dK MMA 与当前 dS 逐元素计算重叠**。
- dQ 归约：`dQacc_reduce`（3564+）从 tmem 读 dQ 累加、`thr_copy_dQaccum_r2s` 写 smem、
  TMA 全局归约；确定性模式用 `mdQ_semaphore` 锁（`_dq_semaphore_lock_value` 3510-3562，
  SPT 序让锁值 = `n_block_max-1-n_block`）。
- 2-CTA：`mma_tiler_dQ` 用 DSMEM 交换半张 dS 后做 (M/2, 2N)(2N, d)，全局 atomic 减半
  （论文图 3）。

### 2.5 `flash_fwd_sm120.py` / `flash_bwd_sm120.py` —— SM80 fallback 确认

- `FlashAttentionForwardSm120`（61 行）：`__init__` 里 `self.arch = Arch.sm_80`
  （`flash_fwd_sm120.py:19`），`can_implement` 只改 smem 容量为 `get_smem_capacity_in_bytes("sm_120")`
  = 99 KB（`56`）。MMA 继承 SM80 的 `warp.MmaF16BF16Op((16,8,16))`（`flash_fwd.py:589-599`）。
- `interface.py` 的 SM120 约束：`page_table=None`、`num_splits==1`、无 block sparsity
  （`956-960`）；tile 默认 hd<=64 用 128×128、hd>64 用 128×64（99KB 预算，`529-539`）。
- **结论：SM120 = SM80 kernel + 99KB smem。没有任何 SM100 路径。**

### 2.6 `paged_kv.py` / `topk_gather_kv.py` / `block_sparse_utils.py` / `pack_gqa.py`

- `paged_kv.py`（247 行）：`PagedKVManager`，cp.async 16B/行逐行拷；`load_page_table`
  用 `FastDivmodDivisor` 算 page/offset（`136-155`）；`shuffle_sync` 广播页指针（`237-241`）。
  SM100 与 SM90 分支：V 是否在 gmem 预转置（`v_gmem_transposed = arch != 90`，`65`）。
- `topk_gather_kv.py`（278 行）：见 2.2.3。**128 线程 4 warps 专门做 gather**。
- `block_sparse_utils.py`：`produce_block_sparse_loads_sm100`（`677`）、
  `softmax_block_sparse_sm100`（`1039`）、`handle_block_sparse_empty_tile_correction_sm100`
  （`833`）——按块稀疏 mask 跳过空 tile 的 load 与 softmax，空 tile 的 mbarrier 契约单独
  处理（`flash_fwd_sm100.py:2162` 的 NOTE 注释）。
- `pack_gqa.py`：`pack_gqa_layout`（15-40）把 GQA 的 qhead_per_kvhead 折叠进 seqlen 维
  （(qhead_per_kvhead, seqlen) 分层），让 TMA 仍保持 4D；`make_packgqa_tiled_tma_atom`
  保持 TMA 维度不变（`43-83`）。**这让我们 runtime 的 GQA/头部处理可以直接照搬布局思想**。

### 2.7 SM100 硬件原语封装

- `blackwell_helpers.py`（1115 行）：`_tcgen05_mma_kind`（13-30）映射到
  `kind::f16/tf32/i8/f8f6f4/mxf8f6f4/mxf4/mxf4nvf4`；`gemm_ptx_partial`（396-616）：
  内联 PTX 里 `elect.sync` 选 1 线程发 `tcgen05.mma.cta_group::N.kind::K [tmem_acc],
  smem_desc_a, smem_desc_b, idesc, pred`，K 维展开用 smem desc 地址递增；`gemm_ptx_partial1`
  （617-794）用 `mad.lo.u32` 把 stage 偏移并进 smem desc；TS（tensor-shared）时
  A 走 tmem 地址 + mbarrier（`534-549`，`split_arrive` 时 mma 自己 spin mbarrier）。
- `mma_sm100_desc.py`（296 行）：`make_instr_desc` 把 dtype/M/N/major 打包成 32 位 idesc
  （146-160）；`make_smem_desc_base` 把 CuTe layout + swizzle 打包成 64 位 smem desc
  （212-283）；`LayoutType` 的 SWIZZLE 编码（177-184）。
- `pipeline.py`：`PipelineTmaUmma`（337+）、`PipelineUmmaAsync`（387+）、
  `PipelineAsyncUmma`（398+）、`NamedBarrier.arrive_w_index`（163-194）——tmem↔smem 之间
  的 mbarrier 信号转接。tx_count 按 TMA 字节数。

---

## 3. SM100 专属但理念可移植的东西

| 技术 | SM100 实现 | 可移植的理念（SM120） |
|---|---|---|
| 2-CTA MMA | `tcgen05.mma.cta_group::2`，M=256 输出拆两 CTA、B 按 N 分半（论文 §2.2） | SM120 无 joint MMA；只能用普通 cluster/DSMEM 做数据共享。TMA multicast 功能存在但不是 SM120 官方优化 target，必须与两个 CTA 各自 TMA + L2 reuse 实测，不能预设省一半带宽 |
| tmem 中间缓冲 | S/P/O 全放 tmem，FP32 累加器只占 128 列/片 | **中间量留在片上不落寄存器/不反复重读**。SM120 无 tmem，等价物是：**寄存器里保持 O 累加 + P 用输入 dtype 存 smem**；或者复用我们已有的 CUDA Graph + persistent smem 缓冲 |
| 全异步 MMA 流水 | MMA 异步写 tmem，软 warp 不阻塞 | **软流水发射**：SM120 的 `mma.sync` 是同步的，但可以用 2 个独立 Q tile / 2 个 KV stage 交错，让 softmax 与下一个 QK 重叠（FA4 的 ping-pong 结构不依赖具体 MMA 指令） |
| 软 exp2 | `utils.py:760-820` 多项式 + MUFU 混用 | **纯数学，直接可移植**。SM120 的真实收益未知；仅在 MUFU/SFU profile 证明热点后启用 |
| 条件 rescale | `softmax.py:293-297` + `vote_ballot` | **直接可移植**，只是把 tmem 上的 O 换成本地寄存器 O 累加 |
| FP8/FP16 P 写入 | `apply_exp2_convert` recast 输入 dtype 再 St32 | **直接可移植**：P 在 smem/寄存器里用输入 dtype（FP8/FP16/BF16），PV GEMM 的 B 读流量减半 |
| split_P_arrive | 先写 3/4 P 就发 mbarrier | **不能直接移植**：FA4 依赖异步 UMMA + TMEM consumer；SM120 register-local P 没有可提前唤醒的独立消费者，需先重构跨 warp P staging |
| LPT 调度 | `tile_scheduler.py:393-647` | **直接可移植**（纯 host/device 端调度逻辑，不碰指令） |
| pack_gqa | 头部折叠进 seqlen 维 | **直接可移植** |
| 每线程整行 softmax | 128 线程 × 每线程一整行，免 inter-warp shuffle | **直接可移植**，M=1 时天然成立 |
| 页表预取 + K/V 同页复用 | `sm100_hd256_2cta_fmha_forward.py:1002-1028` | **直接可移植**到我们 PagedKV |

---

## 4. FA4 之外的耦合资产（黑盒扫描）

- **SM90 前向/反向**（`flash_fwd_sm90.py` 68K、`flash_bwd_sm90.py` 84K）：Hopper
  `wgmma` + cp.async，与 SM120 的 `mma.sync` 是不同指令族（wgmma 是 SM90 专属），
  **不可移植指令**，但它的 producer-consumer 流水与 FA4 同源。
- **flash_bwd_mla_*.py 三件套**（dk / dq_dqv / 主文件，共 193K）：DeepSeek MLA 反向，
  思路同 2.4。
- **flash_fwd_combine.py**：split-KV 的 LSE 合并 kernel——思路可参考，SM120 无此实现。
- **mma_sm100_desc / blackwell_helpers** 是 CUTLASS DSL 对 tcgen05 的薄封装，SM120 完全用不上。
- **git log 无任何 sm120/sm_120 相关提交**（`git log --all --grep=120` 为空）；SM120 支持
  只在 `interface.py` 的 arch 分派 + 两个 fallback 子类里，且标注「in this PR」（`interface.py:959`）。
- 仓库另有 `hopper/`（FA3 C++）、`csrc/`（老 C++ 内核）、`third_party/`（cudnn-frontend 等）。
- **没有人在这个仓库里为 SM120 写过 tcgen05 或任何 SM100 专属移植**——SM120 的「上等公民」
  内核在这套代码里不存在，这正好印证了我们「SM120 必须用标准 mma.sync 自建」的前提。

---

## 5. 完全不能移植的部分（SM100/SM103 专属）

1. **tcgen05.mma 指令族**（`tcgen05.mma.cta_group::N`，`blackwell_helpers.py:503`）——SM120 无。
2. **tmem 及 tcgen05.copy（Ld/St/LdRed）**（`softmax_step` 的 tmem→reg、reg→tmem）——SM120 无。
3. **smem descriptor / idesc 位打包**（`mma_sm100_desc.py`）——依赖 UMMA 的 smem desc 寻址。
4. **FA4 的 TMA multicast + 2-CTA UMMA 数据流**（`tma_partition(..., mcast_id, ...)`）——
   SM120 没有 joint UMMA；multicast 功能虽存在，但不是官方优化 target，不能作为等价迁移。
5. **FA4 的 DSMEM + 2-CTA joint-MMA 归约协议**（bwd dQ，论文 §3.2.3）——普通 DSMEM 在
   SM120 可用，但没有配套 joint MMA，只能另行设计并实测 remote-SMEM/barrier 成本。
6. **ld.red 硬件 rowmax**（`use_ldred_rowmax`，SM103）——tcgen05.ld.red。
7. **2-CTA tmem 分配/回收协议**（`TmemAllocator(is_two_cta=...)`，`flash_fwd_sm100.py:910-916`）。
8. **16 warps / 512 线程的大 warpgroup 结构**——不是指令问题，但 SM120 的 99KB smem 和
   `mma.sync` 的寄存器累加风格让「每线程一整行」这种 128 线程只配 softmax 的布局很难直接搬，
   需按我们 kernel 的线程数重新配比。

---

## 6. 对本项目（SM120 M=1 decode / MLA / MoE）最有价值的 5 个点

### ① M=1 decode：QK 与 PV 的 ping-pong 软流水（`flash_fwd_sm100.py:1751-1816`）

FA4 的 mma 循环每迭代**交错发射 PV_i 和 QK_{i+1}**：tensor core 算 PV 时，下一块 S 的
QK 已经在发射；softmax 与 QK/PV 三者重叠。对 M=1 decode，这对应：
```
等 Q 的 TMA → QK(=Q×K^T，每 KV tile) → softmax(每 KV tile，可与其他头重叠)
             → PV(累加 O) → 下一 KV tile 的 QK
```
即使 SM120 的 `mma.sync` 同步写寄存器，**用「Q 常驻寄存器 + KV tile 双缓冲 smem」的
结构，QK/PV 天然交错**，decode 每步的 latency 就被 PV 与下一 QK 的重叠吃掉一部分。
这是我们 DSV4 decode 最该先动的结构。

### ② P 按输入 dtype 存储，PV GEMM 的 B 读流量减半（`softmax.py:397-401`、`2376-2381`）

`apply_exp2_convert` 把 exp2 结果 recast 成 `q_dtype`（FP16/BF16/FP8）再写回，
P 在 tmem/smem 里是输入精度。M=1 decode 是 bandwidth-bound（每 token 读全 KV），
PV 的 B 操作数是 N×d 的 V；P 瘦身直接影响 QK 侧 smem 流量，而 **V 保持输入 dtype 是
已有的**。FP8 路径还叠加 `max_offset=8`（×256 动态范围上移，`flash_fwd_sm100.py:84-89`）
让 FP8 的 P 精度够用。我们的 MLA decode 直接照抄这两个点：P 存 FP8/FP16，同时用
rescale_threshold 容忍行 max 滞后。

### ③ 软 exp2 模拟（`utils.py:760-820`，Cody-Waite + degree-3 多项式 + f32x2 packed FMA）

纯数学、零硬件依赖。SM120 的 MUFU 吞吐没有足够公开证据支持具体数字；只有 profiler 证明
MUFU/SFU 饱和后，才能把 exp 判成 decode 瓶颈。实现层面有三档可抄：
`ex2_emulation_2`（`760-777`，2 元素包装）、`e2e_asm2`（`781-820`，`fma.rn.ftz.f32x2`
手写 SASS）、以及 `flash_fwd_sm100.py:99-110` 的**混用调参表**（freq=10~32、
res=6~8、start_frg，只模拟 10-25% 元素，保住寄存器）。注意 B300 系列 MUFU 翻倍到
32/clk，所以在 SM120 上软 exp2 的收益和方向都要用软/硬混合 A/B 决定，不能预设正收益。

### ④ 条件 rescale：跳过 online softmax 的 O 重标定（`softmax.py:293-297` + `2542`）

online softmax 里只有 `acc_scale_ >= -rescale_threshold` 时才真的更新 row_max 并 rescale，
否则沿用旧 max（阈值默认 log2(256)=8，即 max 滞后 8 个 log2 单位仍安全）。M=1 时
「rescale O」是整条 512 宽行的向量乘，跳过它就是省一条向量流水；decode 长上下文时
row_max 增长缓慢，**绝大多数迭代都能跳过**。我们的 MLA decode 应直接加
`vote_ballot_sync(scale < 1.0)` 式判定 + 阈值 8。

### ⑤ LPT 调度 + 头部 swizzle（`tile_scheduler.py:393-647`）

论文实测 MHA 4-8%、MQA 7-14% 提升（H200）。decoding 场景天然 load-imbalanced：
不同请求 KV 长度差很大。FA4 的做法是 **mblock 逆序（长任务先做）+ 按 L2 容量把
(head, batch) 分 section 再 swizzle + batch 最外**。我们的单卡连续 batching 直接
把「按 KV 长度降序调度」与现有 admission 结合，是纯调度改动、无 kernel 风险。

（备选第 6 点：**页表预取 + K/V 同页复用** `sm100_hd256_2cta_fmha_forward.py:1002-1028`，
对 PagedKV decode 的每次 KV 页跳转省一次页表 gmem 读——我们已有 block_pool，接上即可。）

---

## 7. 附：关键源码位置速查

| 想找什么 | 位置 |
|---|---|
| 16-warp 分工表 | `flash_fwd_sm100.py:279-298` |
| tmem 布局（S/P/O 重叠） | `flash_fwd_sm100.py:319-332` |
| QK/PV ping-pong 主循环 | `flash_fwd_sm100.py:1751-1816` |
| P 写入输入 dtype | `softmax.py:360-402`；`flash_fwd_sm100.py:2376-2381` |
| 条件 rescale | `softmax.py:293-297`；`flash_fwd_sm100.py:2542` |
| 软 exp2（多项式） | `utils.py:760-820` |
| exp2 混用调参表 | `flash_fwd_sm100.py:99-110` |
| split_P_arrive | `flash_fwd_sm100.py:180-183`、`2394-2409` |
| SM120 = SM80 fallback | `flash_fwd_sm120.py:19,56`；`interface.py:956-960`；`flash_fwd.py:589-599` |
| MLA 三组 GEMM | `flash_fwd_mla_sm100.py:196-219` |
| MLA 跨 CTA rowmax | `flash_fwd_mla_sm100.py:2817-2825` |
| topk gather + bitmask | `topk_gather_kv.py:163-192`、`218-278`；`mask.py:649-657` |
| cpasync relay | `flash_fwd_mla_sm100.py:1348-1360` |
| LPT 调度 | `tile_scheduler.py:393-647` |
| tcgen05 指令 PTX 封装 | `blackwell_helpers.py:396-616` |
| idesc/smem desc 打包 | `mma_sm100_desc.py:146-160,212-283` |
| 论文 roofline 数 | `fa4_paper.txt` 第 244-262 行（表 1）、第 450-463 行（表 3） |
| 论文 exp2 精度表 | `fa4_paper.txt` 第 324-334 行（表 2） |
| 论文 2-CTA/DSMEM/dQ 减半 | `fa4_paper.txt` 第 499-527 行 |
| 论文 LPT 收益 | `fa4_paper.txt` 第 565-569 行 |

---

## 补充：第二轮执行层细节（2026-08-10 增补）

### 软 exp2 多项式系数（utils.py:32-64，sollya fpminimax）
| deg | c0 | c1 | c2 | c3 | c4 | c5 | 最大相对误差 |
|---|---|---|---|---|---|---|---|
| 3 | 1.0 | 0.695146143436431884765625 | 0.227564394474029541015625 | 0.077119089663028717041015625 | | | 8.8e-5 |
| 4 | 1.0 | 0.693042695522308349609375 | 0.2412912547588348388671875 | 0.052225358784198760986328125 | 0.013434938155114650726318359375 | | 3.0e-6 |
| 5 | 1.0 | 0.693151414394378662109375 | 0.24016360938549041748046875 | 0.055802188813686370849609375 | 0.00901452265679836273193359375 | 0.00186810153536498546600341796875 | 8.5e-8 |

- 硬件 ex2.approx ≈ 2^-22 ≈ 2.4e-7。deg3 反而比硬件差 ~300 倍，deg5 才追平。
  FA4 用模拟**不是为精度，是为流水线平衡**（MUFU→FMA 管）。
- 魔法数 6291456.0（2^23+2^22），add.rm.ftz.f32 + 两次 rn 减 + `(bits<<23)+bits` 拼位。
- 混用：`ex2_emu_freq`（FP8 fwd 用 freq=8~32），SM103 用 freq=0 全硬件。

### scale_subtract_rowmax（softmax.py:342-357，★直接抄）
`x*scale_log2 - row_max*scale_log2 + max_offset` 三步并成一条 packed FMA
（bias = max_offset - row_max_scaled），每 2 元素 1 指令。

### 条件 rescale 的 M=1 退化（★直接抄）
M=1 时 row_max 是标量，判定从 vote_ballot(128 lane) 退化成一次标量比较。省：
① 计算 acc_scale 的 exp2；② **整趟 acc_O 的 rescale pass**（读累加器乘一遍写回）。
阈值 8.0（fp16/bf16）依赖 P 低精度存储码点余量；P 若 fp32 需调小。

### P 按输入 dtype 存储（softmax.py:359-451）
寄存器里 recast 同一块 rmem 为输入 dtype；P 与 S 共用缓冲各半幅。
M=1 收益有限（P 只是 1×N 小行），真正大的是 V。除非把多 head/batch 打包进 M。

### gather 行指针 shuffle 分发（topk_gather_kv.py:257-261）
每 lane 算自己的行指针，shuffle_sync 在同行线程组互取 → 避免重复算 topk 地址，
保证同行的 128-bit 连续加载由一组线程完成。**检查 b12x gather 是否重复读 index**。

### M=1 最肥的优化：多级 KV smem + producer warp 超前 cp.async（flash_fwd_sm100.py:1499-1560）
KV 管道可深到 32 级（kv_stage，:376），load warp 一口气超前发长串 cp.async，
计算从尾部倒序消费。SM120 没有异步 MMA，但 **cp.async 是异步的**——
producer warp 提前 2~4 block 发，计算时零等待。这是 decode 最肥的一块。
