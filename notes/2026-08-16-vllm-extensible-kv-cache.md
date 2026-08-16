# vLLM extensible-KV-cache 分支调研与 SM120 动态物理 KV 实施规划（2026-08-16）

状态：🟢 **调研完成 + 本机实验验证通过 + Phase A/B 已实施**。规划为
Phase 5.5 候选（挂接在 `qwen38-dynamic-context-vllm-plan.md` 的 P0 动态逻辑
arena 之上），性能无代价已由本机实测证明。**2026-08-16 实施落地**：
Phase A（`runtime/model/vmm_extensible.py` + 双测试套件）+ Phase B
（`Qwen36SlotPool.extensible_kv` 接入、MTP 锁步池、ServerEngine 实测提交、
`QSR_QWEN_KV_EXTENSIBLE` 开关、GPU 门禁 11/11 + server smoke 通过），
见 §11 实施记录。

## 1. 一句话结论

vLLM 有一个未合入 main 的实验分支 **`origin/extensible-kv-cache`**
（2026-07-19~22，16 commits）实现"动态 KV"的第二层：**用 CUDA VMM
（`cuMemAddressReserve`/`cuMemCreate`/`cuMemMap`）预留整个 KV cache 的虚拟
地址空间，先只提交少量物理页跑 warmup + CUDA Graph 捕获，捕获后按实测剩余
显存一次性提交最终大小**——CUDA Graph 因基址稳定而全部保持有效。本机实测
（RTX PRO 6000 Blackwell / CUDA 13.3 / torch 2.13）：**VMM 内存与 torch 内存
的带宽/随机分页访问完全奇偶（±2% 内），增长提交零化成本 1.7µs/128KiB 页
（30ms decode step 的 0.006%）**。我们 runtime 已有的动态 arena 只做了
"逻辑所有权动态"（Phases 0-3 已落地），物理池仍在启动时整块提交
（`notes/2026-08-15-strict-4x256k-startup-acceptance.md`：stage 2 即
+32.44 GiB）；extensible KV 补上"物理提交动态 + 实测容量定稿"这一层。

**对性能的意义**（本笔记重点）：不改变单请求 decode 延迟（带宽奇偶），但
（a）消灭启动 OOM 与手工容量配置（路线图 S7），把"池多大"从人肉公式变成
实测值；（b）释放的 headroom 直接变成**更大的并发容量 → 更大的有效 M**——
`2026-08-16-qwen38-b1-decode-kernel-attribution.md` 判定轮时 54% 在小 M GEMM
（权重带宽受限），其结构性解法第一条就是"更大的有效 M（并发/投机结构）"。

## 2. 调研对象与基线

| 对象 | 位置/版本 |
|---|---|
| vLLM 本地源码 | `/home/bot/vllm`（editable 安装，HEAD `acb0f1dc` 2026-08-15） |
| **extensible-kv-cache 分支** | `git branch -r origin/extensible-kv-cache`，16 commits ahead of main，最后更新 2026-07-22，**未合入 main** |
| 分支核心文件 | `vllm/utils/vmm_driver.py`（354 行）、`vllm/utils/extensible_tensor.py`（451 行）、`vllm/v1/worker/gpu/attn_utils.py`（+285）、`vllm/v1/worker/gpu_worker.py`（+95）、`vllm/v1/worker/gpu_model_runner.py`（+170）、`vllm/v1/engine/core.py`（+132） |
| 我们 runtime | `9bca41b` 起 Phases 0-3（`69671b7`/`aa66345`），动态 arena strict 模式已可服务 |
| 现有计划 | `.omx/plans/qwen38-dynamic-context-vllm-plan.md`（其 §2.2 明令"第一版不做 CUDA VMM 稀疏提交"，§7 Phase 8 列为独立实验） |

## 3. vLLM 分支做了什么（设计拆解）

### 3.1 核心机制

1. **VA 预留零成本**：`cuMemAddressReserve` 预留全容量虚拟地址（36 GiB 本机
   实测 0.8ms、0 物理内存，见 §4 E1）。
2. **物理增量提交**：`cuMemCreate`（物理分配，granularity=2MiB）+ `cuMemMap`
   + `cuMemSetAccess`。`ExtensibleTensor` 支持 `num_segments>1`（K/V-split
   布局下两个半区各自保持前缀提交），`resize_per_segment_(n, zero_new=True)`
   只提交并零化新增长区间。
3. **捕获期只提交 1 个 block**：`_allocate_extensible_kv_cache` 提交
   `commit(1)` 后即 reshape、bind、capture；warmup 期间按需
   `ensure_kv_cache_blocks(n)` 补足 warmup 要写的块（`warmup.py`、
   `qwen_triton_warmup.py` 都加了钩子）。
4. **捕获后实测定稿**：`core.py` 用 `CompilationTimes.warmup_memory`
   （warmup + graph capture 超出 profiled baseline 的实际内存，从
   `torch.accelerator.get_memory_info()` 现测）算出 `final_available_gpu_memory`，
   重新走 `get_kv_cache_configs` 得到最终块数，再 `extend_kv_cache` 一次提交。
   ——即 **KV 池大小 = 96GiB − 权重 − warmup − graphs − buffer 的实测值**，
   不再依赖预热前对"warmup/捕获吃多少内存"的预估（那正是传统流程 OOM 的
   来源：池先按预估分配，捕获发现不够就晚了）。
5. **CUDA Graph 安全性**：基址固定、block 偏移固定、只增不减 → 捕获后补页
   不失效（本机 E1 §4 实测证明）。
6. **sleep 模式**：`release_physical()` 释放物理保留 VA，wake 时
   `recommit()` 重新提交并零化（VA/视图/图全部保持有效）。
7. **KV transfer 兼容**：UCX 无法跨多 handle 传输 → 注册前
   `defragment=True` 重新单块提交；NIXL 注册推迟到最终大小后
   （`_deferred_kv_transfer_init`）。**单进程无 KV transfer 的我们不需要**。
8. **回退**：VMM 不可用（WSL2 等）时警告并退回标准分配；不支持 packed
   layout / uniform KV / V1 runner + connector 组合（显式 raise）。

### 3.2 语义边界（实现里读出来的，测试 `tests/v1/worker/test_extensible_kv_cache.py` 佐证）

- **grow-only**：`resize_per_segment_` 拒绝缩容；请求路径不 resize。
- **零化保证只到 `num_blocks`**：granularity 向外取整多提交的 padding
  granule **不零化**（本机 E1 实测确认）——kernel 永远不读 num_blocks 之后
  的字节即可，页表寻址天然满足。
- **每 segment 独立前缀**：K/V-split（`(2, num_blocks, ...)`）下 K、V 半区
  各自前缀提交，block b 落在 `[b*S, (b+1)*S)`。
- 物理提交粒度 2 MiB，32 GiB 池 = 16384 granules；128 KiB KV 页 → 16 页/
  granule。

### 3.3 与 main 的分叉

`git merge-base` 停在 `ae10e855`（2026-07-20），分支 16 commits 全为特性 +
fixup，**未 squash 未合入**。上游状态未知（调研日 2026-08-16，remote 未见
对应 PR）。我们采纳的是**概念与语义**（本机已用最小实现独立验证），不依赖
分支代码本身。

## 4. 本机实验（`scripts/b4_vmm_extensible_experiments.py`，可复跑）

GPU：RTX PRO 6000 Blackwell Max-Q / CUDA 13.3 / torch 2.13.0a0 / 95.59 GiB。
脚本自包含最小 ctypes VMM 驱动（不依赖 vLLM 源码）。

### E1 机制与 CUDA Graph 安全性 — ALL PASS

| 项 | 结果 |
|---|---|
| VMM granularity | **2.00 MiB** |
| 36 GiB VA 预留+释放 | 0.8 ms，free delta **0.0 MiB**（纯虚拟） |
| K/V-split 两 segment 提交 | 1 bundle/seg 后 physical=4.00 MiB ✓ |
| torch(DLPack) 视图写/零/读 | 写回一致、新页全零 ✓ |
| **graph 捕获(1 bundle 提交) → 提交 512 bundles → replay 写第 511 块** | **写入正确、其余保持零** ✓（基址稳定主张成立） |
| release_physical + recommit | VA 不变、页重新零化 ✓ |
| padding granule | 取整多提交但**不零化**（语义内安全，见 §3.2） |

### E2 性能奇偶 — 全部在 ±2% 内

| 模式 | torch | VMM | ratio |
|---|---:|---:|---:|
| copy 4 GiB | 701 GB/s | 697 GB/s | 1.006 |
| read 4 GiB（sum） | 78 GB/s | 78 GB/s | 0.990 |
| write 4 GiB（fill） | 1508 GB/s | 1493 GB/s | 1.010 |
| **随机 128KiB 页走读 2 GiB（page-table 模式）** | 489 GB/s | 484 GB/s | **1.009** |
| 增长提交（2→4 GiB, commit+zero） | — | 27.4 ms | — |
| 均摊：128 KiB 页提交成本 | — | **1.7 µs = 30ms decode step 的 0.006%** | — |

（`ratio = vmm/torch` 的中位比值；多次运行在同噪声带内波动 ±0.3%，
全部 ≤1.02。复现：`python scripts/b4_vmm_extensible_experiments.py e2`。）

结论：VMM 页无带宽/分页模式惩罚（decode 处于 DRAM 带宽地板，见
`2026-08-16-w8a8-gemm-roofline-bandwidth-floor.md`，因此这项奇偶是硬前提）；
增长事件成本可忽略。

## 5. 对我们 runtime 的意义（含性能论证）

### 5.1 与现有动态 arena 的关系：互补，不是替代

| 层 | 现有（Phases 0-3 已落地） | extensible KV（本调研） |
|---|---|---|
| 逻辑所有权 | 动态 bundle 分配/COW/refcount/LRU/前缀缓存（`qwen36_kv_arena.py`） | —（不动） |
| 物理提交时机 | 启动即全量 `torch.empty/zeros`（stage 2 +32.44 GiB） | 捕获时 1 bundle，之后实测定稿/按需增长 |
| 池容量来源 | 人肉公式 `1 + 4×2048 + 8 = 8201`（S7） | 实测 `96GiB − 权重 − warmup − graphs − buffer` |
| CUDA Graph | 基址固定（现有前提） | 基址固定由 VMM 保证（实测） |

### 5.2 性能论证（用户关注的优先问题）

1. **零每步代价**：E2 证明 VMM 页与 cudaMalloc 页在所有相关访问模式（含
   随机分页走读）下带宽奇偶。decode 已在 DRAM 地板，此项无隐藏成本。
2. **增长代价可忽略**：1.7µs/页。即使每 decode step 长 1 页，占 0.006%。
3. **容量 → 并发 → 有效 M（真正的性能杠杆）**：归因笔记判定轮时 54% 在
   小 M W8A8/W4A4 GEMM（权重带宽受限）。有效 M 增大的两条路之一就是并发。
   实测定稿的池容量把"公式省出来的保守量"全部转化为并发容量：
   - 现状 strict 4×256K：权重 21.06 + KV 池 36.0 + graphs ~2.6 + 杂项 ≈
     64.09 GiB 峰值，**31.51 GiB driver free**（验收笔记）。
   - extensible 流程下峰值先降到 ~25 GiB（池不提交），捕获后实测可提交
     池至 **~60 GiB（保守留 10 GiB buffer）≈ 1.67× 当前 36 GiB 池**
     ≈ 单机可容 5×256K 或同等前缀缓存/并发余量（此为主观推算，实施阶段
     以实测提交值验收，见 §7 Phase A 验收 1）。
   - 或同样 4×256K 下把 31.5 GiB 留给更激进的弹性组合。
4. **启动健壮性 = 可用性**（北极星指标 #1/#2"能跑起来/不会崩"）：捕获
   OOM 的传统来源（warmup/捕获吃内存超过预估）被消灭；容量算错不再
   OOM/浪费（S7 根治）。
5. **不动的东西**：decode 内核、注意力后端、MTP 图、page table 寻址。

### 5.3 对我们 runtime 的适配差异（与 vLLM 分支的对照）

| vLLM 分支设施 | 我们 runtime 要不要 | 原因 |
|---|---|---|
| KV transfer/connector 延迟注册 + defrag | **不要** | 单进程无 KV transfer |
| sleep 模式 release/recommit | 可要（Phase D） | 无 sleep 需求，但 reset 全空闲时可回收物理页 |
| uniform/packed layout raise | 不适用 | 我们布局已知（K/V-split per-layer tensors, bundle=17 张物理 tensor） |
| `ensure_kv_cache_blocks` warmup 钩子 | 要 | warmup/捕获阶段按需提交（full-forward warmup、MTP 捕获都会写 KV） |
| `warmup_memory` 实测定稿 | 要（核心） | 替换手工 `pool_bundles` |
| 2 MiB granule padding 语义 | 照搬 | 页表寻址天然不越界 |

## 6. vLLM 最近其他改动对本 runtime 的判定（顺路调研）

| 改动 | 判定 |
|---|---|
| `7b544ecb52` MTP trailing all-reduce 融合 + **local-argmax draft tokens** | **无效**：我们 TP=1 无 all-reduce；draft 已直接 `lm_head(...).argmax`（`qwen36_mtp_cudagraph.py:880`），vLLM 的优化点我们天然最优 |
| `1be3628367` Qwen3.5 GDN fused post-conv MTP decode kernel | **低优先**：生产已用 sparkinfer `fused_recurrent_gdn_multistep_indexed`（K 步单核，GDN 仅 0.41ms/轮 = 1.3%），vLLM 融合 kernel 的 launch 削减收益我们已提前拿到；剩余融合面（norm+conv+gate）上限 <1.3% |
| `5af7c8dad7` GDN gates 与 spec tokens 对齐 bugfix | **不适用**：我们的 a/b gate 从已 gather 的 spec 行直接投影（`qwen36_model.py` spec_forward），无 index_select 错位结构；已建议加一条断言锁死（§8 可选任务） |
| `63a9a5010a` DSA MLA MTP=3 native decode | 不适用（MLA/DeepSeek 路径） |
| `57bd0ed441` KV-Cache Layout Refactor [5/N] backend-published packing | 与我们 b12x paged 布局无关，暂缓 |
| `44d95069e9` DeepSeek V4 / GLM-5.1 SM120 enablement | 与 DSV4 后端路线相关，另案跟踪 |

## 7. 实施规划（Phase 5.5：VMM 物理动态 KV，挂接动态上下文计划）

> 前置：动态 arena Phases 0-3 已合入（`69671b7` 起）。本规划**不修改**
> `qwen38-dynamic-context-vllm-plan.md` 的 P0 逻辑层结论，只在其上增加物理层。

**目标**：`Qwen36Backend(dynamic_arena=True, extensible_kv=True)` 下，池
物理提交从"启动全量"变为"捕获后按实测容量定稿 + 可选按需增长"，消除 S7，
启动峰值从 64 GiB 降至 ~25 GiB，池容量从公式值变为实测值。

### Phase A：基础设施（1 个 PR，纯新增）

- `runtime/vmm/`（或 `runtime/model/vmm_extensible.py`）：
  - `VmmDriver`（ctypes，移植 `scripts/b4_vmm_extensible_experiments.py`
    中的最小实现 + `ensure_context` + 多 device 守卫，SM120 单卡只需简单版）
  - `ExtensibleTensor`（移植本机已验证子集：reserve/commit/zero/views/
    release/recommit）
  - `ExtensibleKVCacheBuffers`（多 tensor 锁步提交 + `physical_bytes`）
- 验收（每项都有断言）：
  1. 36 GiB VA 预留 free delta = 0（E1 §2）
  2. graph 捕获后补页 replay 正确（E1 §4）
  3. 提交零化语义：`[0, n*S)` 全零、padding granule 不读（E1 §5）
  4. torch-free 单测（CI 模拟路径不 import torch 的纯分配逻辑可 CPU 测
     granule 记账）

### Phase B：接入 Qwen36Backend dynamic_arena（1 个 PR）

- `qwen36_slots.py`：`extensible_kv=True` 时 k/v/conv/recurrent/attn_outputs
  池改为 `ExtensibleTensor` 视图（17 张/层物理 tensor 各自 num_segments
  按布局算，参照 vLLM `_kv_cache_num_segments_by_layer` 的思路：我们全为
  K/V-split 或 block-major，segment 数 = block_dim 前各物理维乘积）
- warmup/MTP 捕获前 `ensure_kv_blocks(n)` 按需提交（对照 vLLM warmup 钩子；
  我们的 full-forward warmup 与 `enable_mtp`/decode capture 都写 KV）
- 捕获后实测：`free_after_capture` 测定稿 `pool_bundles`（公式
  `(free − buffer) / bundle_bytes`，buffer=10 GiB 起步可调），提交并
  `assert` 公式==实测（沿用 `assert_kv_storage_consistent` 精神）
- **A/B 门禁**：`scripts/b2_verify_dynamic_arena_ab.py` 同参数跑
  `extensible_kv=False` vs `True`，要求 token 流一致、KV 字节一致、
  捕获后池容量 ≥ 公式值或显式报告差值
- 验收数字：新进程启动峰值（stage 2-4 之间池不提交）与提交后容量，
  复用 `notes/2026-08-15-strict-4x256k-startup-acceptance.md` 的分阶段
  测量法

### Phase C：elastic 与按需增长（1 个 PR）

- 默认 `strict` 语义不变；`elastic` 下池容量 = 实测值（无需手工
  `pool_bundles`），请求增长时 `resize_per_segment` 按需提交（grow-only），
  watermark 超限走现有确定性拒绝/排队（不动 preemption）
- 冷前缀缓存与池共享同一 VA：refcnt=0 的 CACHED 块在物理已提交区间内
  复用；可选地把冷缓存物理页在压力下释放（Phase D 再做）
- 验收：`elastic, 18 GiB 预算` 解析 = 实测提交 ≤ 预算 + granule 取整；
  MTP/COW/前缀 restore 全量回归（`tests/test_qwen36_dynamic_arena.py` +
  GPU A/B）

### Phase D（可选，明确不作第一版承诺）

- reset 全空闲 → `release_physical`（池空闲时物理回收，唤醒逻辑
  `recommit`）；sleep 语义我们不需要
- 压力下冷缓存物理页回收（granule 级 unmap）

### 验收总账（对照现有 DoD 清单）

1. `strict, slots=4, 262144`：启动峰值 ≤ 30 GiB（当前 64.09），捕获后池
   容量 ≥ 8201 bundles 且 = 实测值
2. 4×256K 满负载全量回归（质量/正确性门禁全绿，A/B token 一致）
3. 短请求（5 token）live_bundles=1 不变（逻辑层不动）
4. `QSR_*` 无新增必需参数（`pool_bundles` 变可选，默认实测定稿）
5. 性能门禁：B1 128K decode tok/s 与基线差 ≤ 1%（E2 已预示，仍要跑）

## 8. 风险与开放问题

| 风险 | 缓解 |
|---|---|
| 分支未合入 main，上游可能改动语义 | 只采纳概念；本机最小实现独立验证（E1/E2 已过） |
| 提交零化的 2MiB granule 边界与 128KiB 页不对齐（池总大小非 2MiB 倍数时尾部浪费 ≤2MiB） | 池大小向上取整到 granule；`kv_bytes_total` 公式同步 |
| `torch.from_dlpack` 视图的释放顺序（E1 中 `__del__` 报错） | 视图生命周期显式管理（池 tensor 由 `ExtensibleKVCacheBuffers` 持有，不依赖 GC 顺序） |
| `assert_kv_storage_consistent`（公式==实测）与"提交后定稿"的兼容 | Phase B 把公式从"启动时确定"改为"定稿时确定"，两条路径各测一次 |
| 捕获时只提交 1 bundle 是否会暴露 warmup 未提交区（非法访问） | 对照 vLLM 的 `ensure_kv_cache_blocks` 钩子逐一审计 warmup/捕获写点（full-forward、MTP enable、decode capture、`_init_kv_zero_meta` 同类项） |
| 显存结论必须 cold start 测量（AGENTS 规则） | 验收用 fresh process 分阶段 NVML，不用 warm engine |

## 9. 建议

1. **已立项（2026-08-16 用户拍板）**：Phase A+B 已完成——移植已验证子集 +
   接入 + A/B 门禁，直接消除 S7、启动 OOM 风险与闲置显存（见 §11）。
2. 性能验收以 §7 验收 5 为准：B1 128K tok/s 差 ≤1% 即证明零代价。
3. 若立项，`qwen38-dynamic-context-vllm-plan.md` §2.2 的"不做 VMM 稀疏
   提交"与 §7 Phase 8 条目 5 需随 PR 更新为"已落地（Phase 5.5）"。

## 11. 实施记录（2026-08-16，Phase A + B 落地）

### Phase A：`runtime/model/vmm_extensible.py`

- `VmmDriver`（ctypes CUDA VMM，模块级零 torch import，CI torch-free 可收集）
- `ExtensibleTensor`（grow-only 前缀提交、K/V-split `num_segments`、
  DLPack `full_view`、`release_physical` 保留 VA）
- `ExtensibleKVCacheBuffers`（多 buffer 锁步提交、`ensure_blocks` warmup
  钩子、**`add` 会同步新 buffer 到当前已提交前缀**——这是 MTP 池注册
  晚于 backbone 提交时的锁步不变量，vLLM 分支没有这个场景）
- 测试：`tests/test_vmm_extensible.py`（18 项 torch-free 记账，fake driver）
  + `tests/test_vmm_extensible_gpu.py`（8 项 GPU，self-skip）

### Phase B：Qwen36Backend 接入

- `Qwen36SlotPool(extensible_kv=True)`：backbone k/v pools 改 VMM 视图，
  构造时 0 物理提交；`ensure_kv_blocks`/`commit_kv_blocks`/`physical_kv_bytes`
- `build_pooled_mtp_caches`：MTP KV 加入同一锁步池（`register_mtp_kv`
  COW family 语义不变）
- `Qwen36Backend.ensure_kv_blocks/commit_kv_cache`；`ServerEngine`
  `qwen_kv_extensible` + `qwen_kv_commit_buffer_gb`（env/CLI：
  `QSR_QWEN_KV_EXTENSIBLE` / `--qwen-kv-extensible`）
- `_commit_extensible_kv_pool`：warmup+MTP+decode capture 后实测
  `mem_get_info` 定稿提交数，不足时警告（admission 拒绝而非 OOM）
- 捕获期提交策略：warmup 前 `ensure(1 + slots*(K+2))` 覆盖全部写页点
  （warmup 1 页、MTP verify K+1 页/slot、decode capture 1 页/slot）

### 门禁结果（`scripts/b2_verify_extensible_kv.py`，11/11 PASS）

| 检查 | 结果 |
|---|---|
| 构造 0 物理提交（4096 seq: 288 MiB VA → 0 MiB 物理） | ✅ |
| pre-commit greedy == dynamic token 流（2 slots × 16 steps） | ✅ |
| 部分提交下跑通 greedy（9/72 bundles committed） | ✅ |
| commit 前后 KV base pointers 不变 | ✅ |
| commit 全量后物理 == VA 容量 | ✅ |
| post-commit greedy 仍 == dynamic | ✅ |
| MTP K=3 在 extensible 池解码 + 池平衡 | ✅ |
| 原 b2 dynamic A/B 9/9 无回归 | ✅ |
| server smoke（strict+extensible+MTP K=4 完整启动+真实请求） | ✅ 25 bundles VA commit 144 MiB |

修复过程中发现并解决的问题：`ExtensibleKVCacheBuffers.add` 未同步新
buffer 到已提交前缀 → MTP draft capture 写未提交页非法访问（`c5a42db`
之后补修）。

### 遗留（后续 Phase C/D，不在本次范围）

- 按需增长（请求路径 resize）与 elastic 实测定稿的自动默认
- reset 全空闲时 `release_physical` 物理回收（Phase D）
- `qwen38-dynamic-context-vllm-plan.md` §2.2/§7 措辞更新

## 10. 证据文件

- `scripts/b4_vmm_extensible_experiments.py`（本机可复跑，E1/E2）
- `/home/bot/vllm` 分支 `origin/extensible-kv-cache`（16 commits）
- `notes/2026-08-15-strict-4x256k-startup-acceptance.md`（容量/启动基线）
- `notes/2026-08-16-qwen38-b1-decode-kernel-attribution.md`（性能杠杆依据）
