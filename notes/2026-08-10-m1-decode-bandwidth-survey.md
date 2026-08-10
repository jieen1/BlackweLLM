# M=1 decode 带宽优化全景调研：FA4 / FlashInfer / CUTLASS / llama.cpp / ds4 / vLLM / SGLang / 本地 Laguna-Qwen3.6

日期：2026-08-10

## 背景与结论先行

DSV4 M=1 decode 每步必读权重 ≈ **9 GB**（Q8_0 全量 7.19 GiB + IQ2_XS 路由 top-6 约 1.75 GiB）。
卡实测带宽 ceiling ~1300 GB/s，纯带宽下限约 **7 ms**。当前 CUDA Graph 56ms / GPU 53ms，
**总利用率仅 ~15%**。两大块各自很低：

| 分量 | 实测 | 折算带宽 | 利用率(vs 1.3TB/s) |
|---|---|---|---|
| Q8_0 投影 | 18.5ms | ~390 GB/s | ~30% |
| IQ2_XS MoE | ~10ms | ~190 GB/s | ~14% |

**核心结论（四份调研一致）**：M=1 的带宽赤字是 kernel 结构/量化格式问题 + 权重读取方式问题，
不是 tile 配置问题。可借鉴的技术分三档：
1. **当天可做的 kernel/布局优化**：Q8_0 对齐 SoA 重打包（+12%）、MoE down 折进 fused kernel、
   采样 argmax 烤进图、HC 的 split-K 推广到大输出投影、16-byte cp.async（KWIDE）
2. **需门禁的中期**：Q8_0→FP8 e4m3 原生 MMA、FP8 P 重量化进 PV
3. **系统级最大杠杆**：推测解码（MTP/DSpark）把 M 从 1 抬到 K+1，唯一能突破带宽墙的数量级手段

---

## 1. FA4 / FlashAttention / FlashInfer / CUTLASS

### 1.1 FA4 对 SM120 的官方立场：不支持，退回 SM80
- `fa4-latest` HEAD `c75d019`。真正的 FA4/Blackwell 实现全在 `flash_attn/cute/`（CUTLASS DSL Python）：
  `flash_fwd_sm100.py`(3202行)、`flash_fwd_mla_sm100.py`(3160行)、`sm100_hd256_2cta_fmha_forward.py`(1918行)
- **SM120 支持仅** `flash_fwd_sm120.py`/`flash_bwd_sm120.py`（61/55行）= 强制 `Arch.sm_80` + 99KB smem 检查，
  用 SM80 的 m16n8k16 MMA。**无 tcgen05、无 tmem、无 cluster、无 2-CTA MMA**。
- FA4 的核心（tcgen05/tmem/2-CTA MMA/FP8 PV）依赖 SM100 专属硬件，SM120 不可移植。

### 1.2 FA4 可借鉴的设计理念（SM120 有对应物）
| 理念 | FA4 位置 | SM120 对应 |
|---|---|---|
| Warp specialization | `flash_fwd_sm100.py:863` | FlashInfer DSV4 decode 已是 8+1 warp |
| mbarrier full/empty 双缓冲 + expect_tx | `flash_fwd_sm100.py:900` | cp.async.bulk 在 SM120 可用 |
| Q 双 stage / K+V 不均衡 smem 复用 | `flash_fwd_sm100.py:3089` | SM120 只有 99/114KB smem，省 smem 更关键 |
| **V 在 gmem 预转置** | `paged_kv.py:63` | 消除 kernel 内 V 转置往返（本地 sm120 笔记记录的瓶颈） |
| split-KV + cluster 归约 + merge kernel | `flash_fwd_sm100.py:158` | SM120 1x1xN cluster 归约可用（CUTLASS 93_blackwell） |
| top-k gather + bitmask 稀疏选块 | `topk_gather_kv.py:164` | 对 DSV4 稀疏 MLA 直接适用 |

### 1.3 FlashInfer：DSV4 专属实现（最强相关）
- **`include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh`**(916行, `f9546947` + `24d7dfb2`)：
  DeepSeek V4 Flash sparse MLA decode，SM120/SM121。**8 math warp + 1 IO warp**，IO warp 用
  **cp.async.bulk（TMA gather）** 把 top-k 条目 gather 进双缓冲 smem。QK 走 FP8 m16n8k32，
  RoPE bf16 单独 m16n8k16。**FP8 P 重量化进 PV**（原子 max 求 w_head_sc）。
  split-K decode + merge kernel（smem 缓存每 split LSE、attn_sink 折叠进 normalizer）。
- 关键启示：**SM120 上 sparse top-k gather 用 cp.async.bulk 是可行且被采用的**——本地
  sm120-flash-attention 曾以"strided 不适合 TMA"否决，FlashInfer 用 per-entry bulk 绕开。
- SM120 group GEMM 三件套：`group_gemm_{nvfp4,fp8,mxfp4}_groupwise_sm120.cuh`。
  SM120 只用 1x1x1 cluster。

### 1.4 CUTLASS 4.6.1
- **`include/cutlass/gemm/kernel/gemv_blockscaled.h`**：CUDA-core FMA 路径的 FP4 GEMV
  （非张量核）。K-分裂线程布局（每行 16 线程分 K）、128-bit 访问、cp.async 双缓冲 4 stage、
  **dequant 完全融合进计算**（cvt + fma 树）。Q8_0 需自己扩展，但结构完全可复用。
- **`include/cute/arch/mma_sm120.hpp`**：SM120 原生 blockscaled FP4 MMA
  `mma.sync.aligned.kind::f8f6f4.m16n8k32`（e2m1×5 种组合）+ `mma_sm120_sparse.hpp` m16n8k64。
- split-K GEMV 模板：`gemm_with_k_reduction.h`。
- **缺失**：SM120 无 tcgen05/tmem/UMMA；`93_blackwell_low_latency_gqa`（M=1 decode 最接近）
  只支持 SM100。

### 1.5 本地 sm120-flash-attention（已收口，方法论沉淀）
- attention kernel 研发线正式停止（`bdc9e7e` A1-A4 closure），生产走 native FlashInfer。
- **KWIDE 16-byte cp.async 是最值得抄的经验**（`7d1937d`）：V_SMEM_STRIDE 260→272（16 对齐），
  2×cp.async4 换 1×cp.async16，微基准 174→272 GB/s（1.56×）。stride 256 会 8-way bank 冲突。
- 负结论：M=1 decode 瓶颈是 cp.async KV-tile 顺序等待链 + V 转置 smem 往返，不是 occupancy；
  split-KV/cp_async4/occupancy/D-并行全被实测证伪。

---

## 2. llama.cpp / ik_llama.cpp / exllamav2 / antirez/ds4

### 2.1 一个重要纠正：ds4 的 `cuda/mmq/` 是 llama.cpp 的 ggml-cuda mmq 内核 vendor
- `cuda/mmq/VENDOR.md`：pin 到 llama.cpp `5c0e946`。ds4 自研的是 `ds4_cuda.cu` 的 decode 路径、
  `proto_*` 原型系列、D2R 张量核扩展、对齐 SoA 重打包。

### 2.2 llama.cpp / ik_llama.cpp 最新动向
- **MMQ 大重构**（`6eddde06a` #24127）：按架构拆配置，Blackwell 专注 MXFP4/NVFP4 原生 FP4。
- **MoE gate/up 激活量化去重**（`5839ba352` #25441）：ne11==1 时每 token 只量化一次再 scatter。
- **MUL_MAT_VEC_Q**（`mmvq.cu:478`）：M≤8 走 vecdotq，多 warp 分 K（dp4a），warp 部分和经 smem 归约。
- **ik_llama.cpp `--prefetch-experts`**：MADV_POPULATE_READ 预热 mmap 专家权重进 page cache
  （SSD 流式场景，权重已在 GPU 则不适用的对照）。
- 融合 sinkhorn（`#2115`）、fused indexer top_k（`#2103`）——与 ds4 融合方向一致。

### 2.3 antirez/ds4：对 DSV4 Flash M=1 的可借鉴点（详细）
1. **两 tier 分治**（`DS4_CUDA_MMQ_MOE_MIN_TOKENS=2`）：M=1 decode 走向量 warp kernel，
   **M≥2 才用张量核 MMQ**。实测张量核 MMQ 在 M=1 并不赢（prefill 2.8x，decode 持平）。
   我们已 fused Q8_0 张量核 GEMV——**值得做一次同形状 A/B 对比 vecdotq warp-per-row**。
2. **对齐 SoA 重打包 = M=1 唯一的"免费"带宽杠杆**：Q8_0 34B / IQ2_XS 66B 的 code 流
   **只有 2 字节对齐**，每 32-bit 字被迫拆两次 16-bit load。拆成 scale/code 两平面、
   code 64B 对齐后可用 int4/int8 全宽 load。`proto_iq2_aligned.cu` 实测 **+12%**，
   `proto_q8_aligned.cu` 目标 attn_q_b 164→~200 GB/s。**GGUF 加载时离线条换一次，运行期零成本**。
3. **激活量化去重（q8_1 fold）**：生产者在算 norm 的 kernel 里寄存器内直接 emit q8_1 块，
   消费方按指针取走，下个 GEMV 跳过 quantize prelude（`ds4_mmq.cu:144`）。
4. **HC 链融合**（`proto_m2_hc.cu`）：`dot(w, x·s) == s·dot(w, x)` —— **rms_norm 延后**，
   raw x 上做 dot+sumsq 部分和，finish kernel 用 sumsq 推出 scale 一次修正。
   4 个 launch 压成 1 个 cooperative launch（96 blocks，4 次 grid.sync）。V4 写回时顺带量化 q8。
5. **DSpark 推测解码**（README:196）：draft 读主模型 hidden、最多提 5 token、主模型批量 verify、
   置信度阈值 0.7。**真 batch verify（非自回归）** + Markov 修正层。**绕开 M=1 带宽墙主力**。
6. **decode graphs 用 island 拆分**（`ds4_cuda.cu:887`）：每层切成 position-independent 的 2 段
   分别 capture，中间 position-dependent 部分 eager。状态机 warm→capture→replay，失败回退。
7. **mmid case-1 快速路径**（`mmid.cu:300`）：MoE down 把 assignment rows 当 `n_expert_used=1`
   的 token，专用 `<1>` 模板，20.4x。

### 2.4 exllamav2（低活跃，2026-03）
- **编译期 M 特化**：为 max_m∈{1,2,3,4,8} 各编一份全 unroll 内核，运行时选指针。
  激活驻留 shared，权重流式读，寄存器 dequant + `__hfma2` dot（无张量核）。
  M=1 时循环编译期展开为标量序列，零运行时分支。

---

## 3. vLLM / SGLang / nano-vllm

### 3.1 vLLM（最近偏多模态/TTS）
- 最近 commit 多为多模态/TTS/audio。DeepSeek 相关分散。
- decode 用 continuous batching + CUDA Graph 多 batch（业界标准做法，无 DSV4 专属新发现）。

### 3.2 SGLang：对 DeepSeek V4 有大量工作（最强相关）
- **`deepseek_v4_dspark.py`**(892行)：DSpark draft + **ragged verify**（`ragged_verify.py`）——
  多候选批量验证，M>1 带宽摊销的直接参考。
- **`deepseek_v4_nextn.py`**(283行)：MTP（next-token-n）路径。
- `[AMD] Support two batch overlap with MTP on DeepSeekV4`（`e2d021d4ab` #30238）：
  两个 batch 与 MTP 重叠。
- `perf(deepseek_v4): enable SGLANG_OPT_FP8_WO_A_GEMM on sm90`（`dee91c51cf`）：
  **FP8 WO-A GEMM** 优化开关。
- `[KDA] Add FlashInfer SM100 KDA decode + MTP`（`a649b5a9db` #30113）。
- `[mem_cache] MLATokenToKVPoolHost`（`9dd57ef8c4`）：MLA token→KV pool 重构。

### 3.3 结论
- **vLLM/SGLang 的 decode 批处理答案是 continuous batching（M 聚合）**，但 M 由并发请求数决定。
- 对单请求 M=1 场景，真正能突破带宽墙的是**推测解码（MTP/DSpark）**——SGLang 的
  `deepseek_v4_dspark.py` + ragged verify 就是完整实现。

---

## 4. 本地 runtime：Laguna / Qwen3.6 优化历程

### 4.1 Laguna（M=1 带宽受限最完整的诊断范例）
- **带宽账方法论**（`2026-07-27-dflash-bandwidth-roofline-moe-gemm-attention.md`）：
  ① 用 torch.empty+copy_ 实测卡真实带宽 ceiling（~1300 GB/s）；② 每 kernel 用真实形状算
  有效 GB/s；③ **必须冷缓存**（L2 128MiB 会污染测量）。
- batch 1→4 GEMM 仅 +17% → 证明 M=1..4 是读权重主导的带宽受限，非算力受限。
- 已落地：`cg_decode_metadata.py`（B=1 一步 pinned H2D + 单 kernel 写 6 个元数据 tensor）、
  `fused_kv_scatter.py`（288→48 kernels）、`fused_rms_norm.py`、NVFP4 MoE 单 kernel 38μs/层、
  **贪心 argmax 烤进图**（`laguna_cuda_graph.py:460`，replay 后 host 只做一次 .item()）。
- **关键判断**（`2026-07-31-session-summary.md:38`）：MoE 带宽 100% 饱和后
  "No tuning helps. Only system-level opts (batch amortization, L2 locality)"。

### 4.2 Qwen3.6（唯一真正解决"低比特小 M 带宽赤字"的模型）
- **W4A16 融合 kernel 在小 M 拿到 ~830 GB/s**（vs blockscaled W4A4 330-440、torch._scaled_mm FP8 ~400）。
  融合 FC1→act→FC2 单 kernel 是 830 GB/s 的关键。
- 小 M 带宽赤字是 **kernel pipeline 结构问题，不是 tile 配置**（swap_ab/tile/split-K 全实测失败）。
- `fp8_w8a8_sm120.cu`：自研 CUTLASS port，保留原始 E4M3 权重 + 动态 per-token E4M3 激活，
  M=1 bit-exact，常驻 20.2 GiB（消灭 49.72 GiB BF16 反量化缓存）。
- `Qwen36DecodeGraphAttention.update_replay_metadata`：decode 图回放按 live cache_seqlens
  重切 split-KV chunk。
- MTP 四图（draft/verify/sync/decode）+ K=3，接受率 ~1.2-1.5/轮。

### 4.3 它们用了、DSV4 还没用的（按可移植性）
1. **采样/argmax 烤进图**：Laguna 已做，DSV4 的 driver 返回完整 logits 让 host 决策。
2. **MoE down 折进 fused kernel**：Qwen3.6 的 830 GB/s 来自 FC1→act→FC2 单 kernel。
   DSV4 的 gate/up 已融合，**down 是独立 kernel**——折进来是 iq2xs 10ms 最大单点。
3. **Split-K GEMV 推广到大输出投影**：DSV4 只用在 HC（16KB），lm_head(129280)/wo_b/indexer 未用。
4. **Q8_0→FP8 e4m3 原生 MMA**：数值变更需门禁。torch._scaled_mm 对 M<4 不支持，需自研。
5. **L2 驻留**：小权重天然驻留；大权重不指望（Qwen3.6 swizzle 反而 +13% 慢）。

---

## 5. 优化方向建议（按 ROI 排序）

### 立即做（当天，预期各省 3-5ms）
1. **Q8_0 对齐 SoA 重打包**：34B block → scale/code 两平面、code 64B 对齐。
   目标：Q8_0 投影 390→~700 GB/s（ds4 实测思路 +12%，但我们欠利用率更多）。
   **验证**：nsys 看 DRAM 读带宽。
2. **MoE down 折进 fused kernel**：扩展 `dual_swiglu_b1` → gate/up → 寄存器内 SwiGLU →
   down GEMM → route-weight 归约，单 kernel 覆盖 routed + shared。
3. **采样 argmax 烤进图**：贪心路径把 argmax 放 capture 块，消每步 logits D2H/决策。
4. **16-byte cp.async（KWIDE）**：所有权重/KV 加载 stride 取 16 对齐且 %32≠0。

### 中期（需门禁/测量）
5. **Split-K 推广到大输出投影**：lm_head、indexer、wo_b（已有 HC 12× 先例）。
6. **Q8_0→FP8 e4m3 评估**：作为 kernel 结构修复之外的 B 计划，先做 bit-exact 预演。
7. **FP8 P 重量化进 PV**（FlashInfer DSV4 已有实现，b12x 里可能已含）。

### 系统级（最大杠杆，最远）
8. **推测解码**：DSpark（真 batch verify）或 MTP 扩容。SGLang `deepseek_v4_dspark.py` +
   ragged verify 是完整参考。把 9GB/步 摊到 K+1 token。需要先解决 MLA 递归状态 + 压缩 KV
   的回滚语义。roadmap D9 已预留 DSpark（10.9GB draft GGUF 未下载）。

### 物理下限
每步读 ~9GB，带宽 ceiling 1300GB/s → 纯下限 ~7ms。当前 53ms。1-4 项可把利用率
从 ~15% 拉到 ~40%+。

---

## 附：调研项目清单与版本

| 项目 | 本地路径 | HEAD |
|---|---|---|
| FA4/flash-attention | /home/bot/project/fa4-latest | c75d019 |
| SM120 flash-attention 分支 | /home/bot/project/sm120-flash-attention | 7d1937d |
| FlashInfer | /home/bot/project/flashinfer | 608657a7 (v0.6.15) |
| CUTLASS | /home/bot/project/cutlass-4.6.1 | e05f953 (v4.6.1) |
| llama.cpp | /home/bot/project/llama.cpp | 79bba02a |
| ik_llama.cpp | /home/bot/project/ik_llama.cpp | 1fddd12 |
| exllamav2 | /home/bot/project/exllamav2 | 7dc12af |
| antirez/ds4 | /home/bot/project/ds4（本轮新 clone） | 84cc882 |
| vLLM | /home/bot/project/vllm-omni | e0c2b9d7 |
| SGLang | /home/bot/project/sglang | b296e1a503 |
| nano-vllm | /home/bot/project/nano-vllm | bb823b3 |
| DeepGEMM | /home/bot/project/DeepGEMM | 559d79f |
| tilelang | /home/bot/project/tilelang | 9ff4ef8 |
| sparkinfer(b12x) | /home/bot/project/sparkinfer | 583e313 |
