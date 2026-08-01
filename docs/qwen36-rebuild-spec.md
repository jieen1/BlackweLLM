# Qwen3.6-27B 重建规格

> 编制日期：2026-08-02 · 基线 commit：`52f9484`（`work/qwen36-trackB`）
>
> **定位**：这不是"接入一个陌生模型"的规格，是"重建一个曾经在 vLLM 上跑通、有实测数字、
> 后来因为整条 vLLM 依赖被剥离而被截肢的实现"的规格。`oracle/qwen36_vllm/` 有 8047 行、
> 11 个模块的参考代码；`docs/archive/2026-07-20-PROGRESS.md`、
> `docs/archive/2026-07-30-architecture-two-tenant.md`、`notes/2026-07-22-quality-baseline-*`
> 里有真实测过的吞吐、接受率、MMLU-Pro、HumanEval+、显存数字。本文档做三件事：
> ①把 `oracle/qwen36_vllm/` 逐模块判定"能不能用、怎么用"；②把当年的实测数字整理成
> 验收基线；③给出在今天的 Track A 抽象（`ModelBackend` 协议 / `ArchitectureSpec` /
> `SlotResourceManager`）上重建它的设计。
>
> **不做**：不写生产代码，不复活 `oracle/qwen36_vllm/`（生产代码永远不能 import 它，
> 这条合同不变）。本文档产出后，Track B0-B3 的实施者应该照着第 2、4 节的映射表写代码，
> 照着第 3 节的基线判定"打平还是退步"，照着第 6 节的风险列表决定实现顺序。
>
> **纪律**：本文档遵守 `docs/README.md` 的文档纪律——数字标日期/配置/来源；
> 没实测过的标 **[待验证]**；不重新推翻已拍板的结论（`investigation-queue.md` B-6/C-2/C-3、
> `roadmap.md` §7 D6），只核实和引用。

---

## 0. 一句话摘要

`oracle/qwen36_vllm/` 里 8047 行代码，**真正能直接复用的只有编排层与状态管理层**——
GDN checkpoint 快照/恢复、GDN 状态×投机解码的行寻址方案、accept/reject 判定算法、
块哈希前缀缓存的骨架。**模型数学本身（GDN 层 forward、mrope RoPE、attn_output_gate、
稠密 SwiGLU MLP、modelopt NVFP4 反量化）完全不在这份参考代码里**——它当年活在
vLLM pip 包自己的 `Qwen3_5ForConditionalGeneration`/`Qwen3_5MTP` 类里，从未被搬进这个仓库，
`get_model()` 只是把它租来用。这是本轮调研对任务原始判断的**最重要纠偏**：模型图层是
纯绿地工作，不是"移植"工作。好消息是：Track A 已经把很多当年需要现场设计的抽象
（`ArchitectureSpec` 的 `CACHE_PAGED_KV`/`CACHE_RECURRENT`、`_ssm_spec_row` 的 GDN×投机
行寻址）提前建好了，其中一部分甚至是**当年这套代码本身留下的，休眠在 `block_pool.py`
里没被删掉**。

---

## 1. 家底盘点：`oracle/qwen36_vllm/` 逐模块判定

判定口径（与任务要求一致）：

- **可直接搬**——不依赖 vLLM，或依赖面已被自研件替代，可近乎原样复制。
- **需改写**——逻辑/算法是对的，但绑在 `VllmConfig`/`Attention`/`get_model()`/
  `ForwardContext` 上，需要照着 `ModelBackend` 协议与现有 Laguna 基础设施重新表达。
- **已被取代**——自研框架里已有等价物（通常是为 Laguna 造的），应该泛化它而不是
  把老代码搬进来并列存在。
- **应废弃**——当年应对 vLLM 特定问题的权宜之计，新框架里这个问题不存在。

### 1.0 关键纠偏：模型数学层根本不在这份参考代码里

逐行读完 `backends/qwen36.py`（2159 行）后确认：**零 GDN 前向、零 RoPE、零
`attn_output_gate`、零 SwiGLU、零 modelopt 权重处理代码**
（`grep -in "rotary\|rope\|swiglu\|attn_output_gate\|modelopt\|def forward"` 无命中）。
整个文件是单一类 `Qwen36Backend`，13 个方法全部是 MTP draft/verify 编排。真正的模型
定义——`Qwen3_5ForConditionalGeneration`/`Qwen3_5MTP`——在
`direct_model_runner.py:487` 的 `get_model(vllm_config=vllm_config)` 里现场从 vLLM
pip 包借用，**从未被 vendor 进 `oracle/` 或本仓库任何位置**。`oracle/qwen36_vllm/` 里
四个 `nvfp4_*_patch.py` 也不是权重加载器，是运行期打进 vLLM 自己的
`CutlassNvFp4LinearKernel`/kernel 选择逻辑的 monkeypatch。

**结论**：GDN 层 forward、mrope-interleaved RoPE（partial_rotary_factor=0.25）、
`attn_output_gate`（swish 门控）、稠密 SwiGLU MLP、modelopt NVFP4 反量化——这五样是
**全新代码，只能参考 vLLM 上游源码作外部文档（不 import，只读）+
`runtime/architecture.py` 已解析出的架构事实**来写，不存在"移植"这个选项。
下面 1.1–1.9 判定的都是**编排/状态/元数据/缓存**这一层，那一层确实有大量可用的参考。

### 1.1 `gdn_state.py`（466 行）—— 逐方法判定（本文件被判定为最高价值文件，是唯一）

零 vLLM import（`from __future__ import annotations; from collections import OrderedDict;
import torch; from runtime.block_pool import _physical_slot`）。唯一"耦合"是对
`self._r`（runner）的字段依赖，这是数据结构问题，不是 vLLM 问题。

| 方法（行号） | 判定 | 新位置 / 备注 |
|---|---|---|
| `GdnStateManager.__init__`（30-31） | 可直接搬 | 持有 `self._r` 的模式直接复用 |
| `_allocate_gdn_checkpoint_pool`（33-92） | 需改写 | 按层 shape/dtype 分配 checkpoint 池、按字节预算定容量的逻辑是对的；但读的是 `self._r.kv_caches[name]`/`self._r.gdn_layer_names`（runner 内部字段），需要改成从 `ArchitectureSpec.recurrent_layers`（`runtime/architecture.py:132`）驱动 |
| `_gdn_ckpt_alloc_slot`（94-104） | 可直接搬 | 纯 free-list/LRU 记账，零外部耦合 |
| `materialize_gdn_checkpoint`（106-161） | 需改写 | `torch._foreach_copy_` 模式直接复用；只需重新绑定 `self._r.kv_caches[name]` 到新 backend 的状态存储 |
| `checkpoint_view`（163-181） | 可直接搬 | 纯字典/LRU 读取，零耦合 |
| `evict_gdn_checkpoint`（183-209） | 需改写，但对应一个**已经预留好、只差实现**的挂钩 | `runtime/block_pool.py` 的 `BlockPool._on_evict_block: Callable[[int], None] \| None`（326-329 行）正是这个回调的插槽，Laguna 侧目前是 `None`。这个方法就是这个挂钩缺的那份实现——移植逻辑、接进这个挂钩即可 |
| `_evict_gdn_checkpoints_for_budget`（211-226） | 可直接搬 | 纯记账，其自身 docstring 就写"可无 GPU 单测" |
| `reset_slot`（228-286） | 需改写 | 逻辑对，但摸了约 9 个 runner 持有的逐槽数组（`block_table`、`slot_kv_len`、`slot_gdn_initialized` 等），需要照 `runtime/backends/protocol.py` 的 `SlotStateView`/`SlotSnapshot`（74-77、111-124 行）重新表达 |
| `snapshot_gdn_state`（287-383） | 可直接搬 | GPU 常驻双缓冲 + `torch._foreach_copy_`，纯自研原语，零 vLLM 耦合，含性能设计理由 |
| `restore_gdn_state`（385-467） | 可直接搬 | 同上，含 generation-counter/消费标记/槽位不匹配防护与跨槽 fan-out 模式（`allow_cross_slot`），是真正打磨过的代码 |

**关键**：`_ssm_spec_row`/`_physical_slot`（`gdn_state.py`/`metadata_builders.py` 都依赖）
**已经原样存在于今天的 `runtime/block_pool.py`（23-24、45-79 行），未被删除**——这块
GDN×投机解码的行寻址方案不需要重新设计，直接可用。

### 1.2 `metadata_builders.py`（596 行）—— 逐函数判定

耦合点：`from oracle.qwen36_vllm.attention_compat import (GDNAttentionMetadata,
SM120GQAMetadata, ...)`——这两个不是本地 dataclass，是
`vllm.v1.attention.backends.gdn_attn.GDNAttentionMetadata`/
`vllm.v1.attention.backends.sm120_gqa.SM120GQAMetadata` 的直接引用，被 vLLM 自己
vendor 的 GDN 层代码 `isinstance` 检查。`prepare_chunk_indices`/`prepare_chunk_offsets`
来自独立 pip 包 `fla`，架构文档已把它列为可接受的直接 pin（不算 vLLM 耦合）。

| 函数（行号） | 判定 | 新位置 / 备注 |
|---|---|---|
| `build_attention_metadata`（22-75） | 已被取代 | CSR 数学（qo_indptr 等）可迁移，但容器是 vLLM 的 `SM120GQAMetadata`；形状已被 `runtime/backends/bf_attention.py` 的 `BFAttnContext`（线程局部 dict-of-tensors）+ sparkinfer 自己的元数据取代 |
| `build_attention_metadata_batch`（157-384） | 已被取代 | 同上，批量版；ragged qo_len / CUDA-graph 安全分片的设计值得作为设计笔记保留 |
| `build_gdn_metadata`（78-133） | 需改写 | 构造真实 vLLM `GDNAttentionMetadata`；CSR 构造算法可迁移，但需要自建元数据类型——新文件，如 `runtime/backends/qwen36_gdn_metadata.py` |
| `build_gdn_metadata_batch`（387-521） | 需改写 | 同上批量版；2026-07-17 的"按值不按类型判断 uniform qo_len=1 快路径"修复（409-430 行文档）值得作为教训保留，不只是代码 |
| `build_gdn_metadata_spec_batch`（524-596） | 需改写，**但是本文件最有价值的算法** | 基于 `_ssm_spec_row` 的 K+1 专用行寻址方案（已原样存在于 `runtime/block_pool.py:45-79`）是"GDN 状态×投机解码"问题的真正解法。依赖真实 kernel 契约（`spec_state_indices_tensor`/`num_accepted_tokens`），需要对新选定的具体 GDN kernel 重新验证契约是否一致 |
| `_MAX_DECODE_QO_LEN`/`_DEFAULT_PREFILL_CHUNK_SIZE`（136-154） | 可直接搬 | 纯常量，但 `_MAX_DECODE_QO_LEN=16` 绑定 vLLM 自己 `SM120GQAMetadataBuilder` 的测试范围，需对新 decode kernel 重新核实上界 |

### 1.3 `backends/qwen36.py`（`Qwen36Backend`，2159 行）—— 按子系统判定

每个方法都经 `from oracle.qwen36_vllm.vllm_compat import set_forward_context` 加上
`self._r.vllm_config`/`self._r.mtp_model`/`self._r.model`——这是全文件唯一但贯穿式的
vLLM 耦合点，每个方法的 forward 调用胶水都需要重写。

| 子系统 | 方法（行号） | 判定 | 备注 |
|---|---|---|---|
| 单槽 MTP 前向原语 | `_mtp_forward`（91-201） | 需改写 | `prior_kv_len`（attention 历史）与 `slot_draft_sync_len`（持久记账）解耦是真实修过的 bug（2026-07-17 回归记录，120-155 行），保留设计、重写管线 |
| 同步+提议漏斗 | `_mtp_sync_and_propose(_batch)`（203-260/671-902）、`_mtp_run_continuation_steps`（598-670）、`_mtp_forward_batch`（453-596） | 需改写 | 除 GDN 状态外本文件最有价值的算法：第 0 步 teacher-forced 重同步 vs. 探索性自回归步、按槽 ragged qo_len 批处理、CUDA graph step0/continuation 分裂 |
| Prefill 入口 | `mtp_prefill`（262-306）、`mtp_prefill_batch`（904-1310） | 需改写 | `mtp_prefill_batch` 有一个**当年也没解决**的开放缺口：`chunk_size` 叠加真正 ragged（每槽不同长）batch 会抛 `NotImplementedError`（1042-1050 行） |
| Verify+commit | `mtp_verify_and_commit(_batch)`（308-442/1985-2159） | 需改写，但内核判定函数**已经是可直接搬（且已经搬完了）** | accept/reject 决策本身调用的是 `determine_accept_reject(_batch)`——这**就是今天的 `runtime/mtp_accept.py`**，已经零 vLLM、已经移植完成。只有 verify forward 周围的管线需要重写 |
| 同轮前缀 fan-out | `mtp_prefill_fanout_batch`（1318-1549） | 需改写 | 真正新颖的算法（探测公共前缀→只 prefill leader 一次→在分叉点 checkpoint GDN→`reference()` 共享 attention 块 + 跨槽 `restore_gdn_state`），值得保留 |
| 会话亲和续接 | `mtp_prefill_warm_continue`（1560-1633） | 需改写，且**协议里已经预留了这个坑** | `runtime/backends/protocol.py`（22-27、202-209 行）已经在 `capabilities.warm_continue` 下声明了这个确切方法名，其 docstring 直接点名"当前没有任何 shipping backend 实现它，--session-affinity 每次都静默降级"——这是协议提前为未来的 `Qwen36Backend` 挖好的坑 |
| 带缓存的统一 prefill | `mtp_prefill_with_cache`（1635-1983） | 需改写 | 组合 hit/fan-out/cold 三条路径，同样的耦合，子件之外无新算法惊喜 |

### 1.4 `direct_model_runner.py`（2017 行）+ `cuda_graphs.py`（1118 行）—— 按子系统判定

| 子系统 | 老位置 | 新位置 | 判定 |
|---|---|---|---|
| 槽位 reset / 新鲜槽记账（非递归状态部分） | `DirectModelRunner.reset_slot` → 委托给 `GdnStateManager.reset_slot`（`direct_model_runner.py:1598-1599`） | `LagunaBackend.reset_slot`（`runtime/backends/laguna.py:2117-2138`） | 已被取代——泛化 Laguna 的模式（保留 KV、存 token 历史、防重复 reset） |
| GDN 递归状态 reset/init 标记 | `__init__`（594-643）、`GdnStateManager` | 无（Laguna 无 GDN，`laguna.py:553-563` 显式 `gdn_layer_names=[]`） | 需改写——无模板可抄，几乎要原样从 oracle 重新推导 |
| 物理槽 0 保留（`RESERVED_PHYSICAL_SLOTS=1`） | `direct_model_runner.py:44-57`；根因见 `notes/direct-model-runner-design.md` "Stage C" | `block_pool.py:17-24` 沿用 `=1`，但 **Laguna 自己本地定义 `RESERVED_PHYSICAL_SLOTS=0`**（`laguna.py:53`、`laguna_cuda_graph.py:41`） | **应废弃（待核实）**——花了四轮调试才坐实这是"vLLM 调度器从不产出物理索引 0"这一 vLLM 特定事实，不是硬件事实；Laguna 保留 0 个物理槽也能跑。新实现不应该"为了安全"照抄这条保留，应该先在自建栈上实证检查是否还需要 |
| Chunked prefill 数据结构/调用约定 | `ChunkedPrefillState`、`prefill_chunked_begin/_step`（1731-1938） | `block_pool.py:153-183`（dataclass 已共享）、`LagunaBackend.prefill_chunked_begin/_step`（`laguna.py:2221-2278`） | 契约形状已被取代；但… |
| Chunked prefill 真正的跨步状态机（与 decode 轮交织） | `direct_model_runner.py:1817-1938`（`prefill_chunked_step` 中途返回 `False` 让引擎跑一轮 decode） | Laguna 的 `prefill_chunked_step`（`laguna.py:2276-2278`）是**空操作 stub**："Laguna prefill 从不是增量的" | 需改写——无可搬内容；连 oracle 自己的 Phase B（跨步交织）也**从未完工**（`notes/2026-07-20-inv8-chunked-hit-prefill-plan.md` §"Phase B" 明确标注为更难、未确认建成的一半） |
| Decode CUDA graph 捕获/重放骨架 | `CapturedBatchDecodeGraph`（`cuda_graphs.py:21-689`） | `LagunaCudaGraphDecode`（`runtime/backends/laguna_cuda_graph.py:55-554`） | 已被取代——Laguna 版本是更成熟的同构模式（持久 buffer、`_fill_buffers`、固定地址重放、捕获前后 patch/unpatch attention 实现） |
| **GDN 状态中立的图捕获**（在保留 warmup 槽上写 dummy 数据再捕获，捕获后 reset——因为 GDN 递归状态**非幂等**，不像 attention KV 那样能安全重复 warmup） | `cuda_graphs.py:87-130`（"2026-07-17，state-neutral capture"） | 无等价物——Laguna 的 decode 图从不触碰递归状态，其 warmup 复用天然安全 | 需改写——**全新问题，无可抄**，必须重新推导，不能从 Laguna 的（不相关的）安全性论证反推 |
| MTP/投机 verify CUDA 图 | `CapturedBatchDecodeGraph` 的 `qo_len>1` 分支 + `CapturedMTPDraftStepGraph`（692-1119） | `LagunaCudaGraphVerify`（`laguna_cuda_graph.py:618-985`）+ `DFlashEngine._capture_verify_cg/_capture_draft_cg`（`laguna_dflash.py:442-470`） | 已被取代（调度模式）——Laguna 版本更精细（SWA-ring 感知、融合元数据 kernel） |
| Accept/reject 判定算法 | `determine_accept_reject`/`mtp_verify_and_commit(_batch)` | `_greedy_accept_reject`/`_verify_only_accept_reject`（`laguna_dflash.py:76-123`） | 已被取代——领域通用的最长公共前缀 vs argmax 逻辑，无 GDN 依赖，Laguna 版本是干净的参考 |
| 图捕获时的 metadata 重建（避免逐步 Python/H2D 开销） | 无——oracle 自己承认"未尝试完全无分配版本" | `write_laguna_b1_decode_metadata`（`runtime/kernels/cg_decode_metadata.py`，GPU 侧融合重建，B=1 路径零 H2D） | 已被取代，且是**实质性的改进**，重建时应直接采用而不是照抄 oracle 更差的方案 |

### 1.5 `prefix_cache.py`（357 行）判定

| 子系统 | 判定 | 备注 |
|---|---|---|
| 块哈希链、内容寻址索引、引用计数/LRU 驱逐 | 需改写（跨槽/跨请求共享部分）/ 已被取代（同会话部分） | `BlockPool`（`runtime/block_pool.py:270-519`）机制完整存在，但**实测确认只被 `benchmarks/`/`tests/`/oracle 使用**——`LagunaBackend` 从不构造 `BlockPool`，从不调用 `cache_block`/`touch`/`hash_to_block`。Laguna 自己的 `reconcile_prefix_hit`/`find_best_slot_for_prompt`（`laguna.py:2139-2219`）是**同槽线性 token 比较**，不是内容寻址哈希 |
| 前缀缓存×GDN checkpoint 联动驱逐 | 需改写 | 逻辑在未逐行读完的 `gdn_state.py::evict_gdn_checkpoint` 里（见 1.1）；`BlockPool._on_evict_block` 挂钩本身是通用的，**`block_pool.py` 里没有残留 GDN 专属代码**（与任务原始假设不同，已核实是干净的通用挂钩，不是需要清理的残迹） |

### 1.6 `vllm_compat.py`（402 行）—— 逐项拆解：它在垫哪些 vLLM 子系统

| 它垫的 vLLM 子系统 | oracle 调用点 | 新框架现状 |
|---|---|---|
| `EngineArgs`/`EngineArgs.create_engine_config()` → `VllmConfig` | `direct_model_runner.py:build_vllm_config`（255-278） | **已经替代**——`runtime/laguna_config.py`（`LagunaRuntimeConfig`/`build_laguna_config` 等）是从零写的对应件。重建目标是扩展这个文件加一个 Qwen3.6 形态的变体，不是复活 `EngineArgs` |
| `get_model(vllm_config=...)` | `direct_model_runner.py:487` | **已经替代**——`runtime/model_loading.py:load_laguna_model` 直接流式读 safetensors，自建 KV-scale post-load。其 docstring 明确写着这是"唯一路径…`QSR_LAGUNA_MODEL_LOADER=vllm` 逃生舱已被彻底删除" |
| `bind_kv_cache` | `direct_model_runner.py:224` | **已经替代**——`bind_laguna_kv_cache`（`runtime/laguna_runtime.py:63-87`），两边都是纯字典绑定+按层序排序，逻辑几乎相同，GDN 的 `(conv_state, ssm_state)` 元组绑定可直接套用同一模式 |
| `set_forward_context`/`ForwardContext`/`get_forward_context` | 每个 `_forward(_batch)` 调用点 | **已经替代**——`laguna_forward_context`（`runtime/laguna_runtime.py`）+ `bf_attn_context`（`runtime/backends/bf_attention.py:67`），拆成两个 context manager 替代 vLLM 单一全局 `ForwardContext` |
| Attention backend 注册体系 | `direct_model_runner.py:_ensure_sm120_backend_registered`、`attention_compat.py` | **已经替代**——`BFAttention`（自建 `nn.Module`）无需注册表，因为模型图是自建的，从不向 vLLM 按字符串/枚举请求 backend |
| `load_eagle_model`（MTP/draft 加载） | `direct_model_runner.py:549-553` | **对 DFlash 场景已经替代**——`load_laguna_dflash_draft_model`（`model_loading.py:187-231`）处理了同样的 embed/lm_head 共享问题。但真正的**checkpoint 内 MTP**（不是独立 draft 模型）需要自己的加载器变体，*模式*可直接套用 |
| `FlashInferMetadataBuilder`/`CommonAttentionMetadata` | oracle 文档字符串声称"被 `laguna.py` 使用" | **过期声明，已核实为假**——grep 确认 `laguna.py` 零引用；其真实注释写着"kept for CG compat, unused for prefill" |
| NVFP4 kernel 选择补丁（4 个函数） | `direct_model_runner.py:467-492` | 见 1.8；这些补丁的目标是 vLLM 自己的 kernel 注册表，一旦模型图自建（如 Laguna 已经做的），这类补丁本身就失去存在理由 |
| 网络工具（`get_open_port` 等） | 全文件 | **可直接搬（且已经是**）——`runtime/laguna_runtime.py:34-45` 是逐字节相同的自写副本，双方本来就是零 vLLM 纯 Python |

### 1.7 `attention_compat.py`（150 行）

| 内容 | 判定 |
|---|---|
| `compute_causal_conv1d_metadata`（99-150） | **可直接搬**——自身文档写明"2026-07-22 实测验证 bit-exact"，纯 numpy/torch 自写，零 vLLM 依赖 |
| `GDNAttentionMetadata`/`SM120GQAMetadata` dataclass、`AttentionBackendEnum`/`register_backend` | **需改写**——是真实 vLLM 类的直接引用（`vllm.v1.attention.backends.gdn_attn.*`），需要按 `runtime/laguna_runtime.LagunaAttentionMetadata` 的形状自建一个 GDN 版本 |

### 1.8 kernel/patch 小文件（触发理由 + 判定 + 新落点）

| 文件 | 当年在垫什么 | 判定 | 新落点 |
|---|---|---|---|
| `triton_norm_ops.py`（264 行）：核心 RMSNorm kernel | vLLM 编译的 `_C.abi3.so` 在那台机器上缺 `rms_norm`/`fused_add_rms_norm` 符号，被迫退化成 8+ 次 launch 的原生 PyTorch fallback | 已被取代 | `runtime/kernels/fused_rms_norm.py`（`rms_norm`/`fused_add_rms_norm`/`TritonRMSNorm`），同算法更干净的单 pass 版本。**[待验证]**：其 `BLOCK_SIZE=next_power_of_2(N)` 单块假设在 Qwen3.6 hidden=5120、`num_warps=8` 下的寄存器/共享内存预算，当前只在 Laguna hidden=3072 验证过 |
| `triton_norm_ops.py`：`install_triton_norm_ops()`（IR-op 注册） | 把上面 kernel 注册进 vLLM 的 op 优先级分派系统 | **应废弃** | 新框架没有这层分派表概念，无需等价物 |
| `triton_norm_ops.py`：`triton_silu_and_mul` | 融合 SwiGLU 激活，无 vLLM 特定理由 | 可直接搬 | 当前无等价物——`laguna_decoder.py:137` 是未融合的 `F.silu(gate)*up`；新落点是 Qwen3.6 新解码层文件 |
| `nvfp4_custom_gemm.py`（197 行）：自研 CUTLASS-C++ NVFP4 GEMM + 按 shape 调参表 | 号称比 vLLM 自己的 `cutlass_scaled_fp4_mm` 快 17.4%，bit-exact | 需改写 | `.cu` 源码**并未消失**——`runtime/kernels/nvfp4_gemm_sm120.cu` 是同一个 kernel，Makefile 仍保留构建规则，但**零 Python 调用方**：`runtime/nvfp4_custom_gemm.py` 与其唯一调用方 `runtime/model/nvfp4_linear.py` 都在 `a9cb932` 里被删除（当时判定为无用）。**现在与 `sparkinfer.gemm.blockscaled.mm`（`sparkinfer/gemm/blockscaled/api.py`）竞争**，后者自带按 `expected_m` 调参。**[待验证 GPU]**：在 Qwen3.6 真实稠密 shape（34816/17408/6144/5120/96，见 oracle 自带的调参表）上把两者跑一次 A/B，sparkinfer 打平或更快就直接放弃复活这个自研 `.cu` |
| `nvfp4_cudnn_patch.py`（119 行）：走 cuDNN backend 的 NVFP4 patch | 号称快 12.6%，但**默认关闭**（`QSR_A2_CUDNN=0`），自己的注释写"对真实权重不 bit-exact" | 应废弃 | 依赖 FlashInfer + vLLM 的 FlashInfer 包装类都不存在；历史上从来不是安全收益，sparkinfer 也没有 cuDNN backend 选项 |
| `nvfp4_b12x_patch.py`（80 行）：重新插入 "b12x" warp-MMA kernel 到 vLLM 优先级列表前面 | 绕过上游 CUTLASS SM121 MMA guard 把这个 kernel 排除出自动选择的 bug | 应废弃（结构性消失） | `sparkinfer/_lib/dense_gemm.py` 明确是"唯一一个 SM120/121 原生 warp-MMA 引擎，无 tcgen05/TMEM 竞品"——这个补丁在打的选择歧义问题在 sparkinfer 里不存在 |
| `nvfp4_cutlass_direct_patch.py`（62 行）：绕过 FlashInfer Python 包装、直调 C++ kernel 省调用开销 | 每 decode 轮 ~304 次 GEMM 调用的 Python 开销 | 应废弃 | 新框架的 `fp8_linear.py`/`plain_linear.py` 在 `forward()` 里直接调 GEMM 函数，本来就没有这层分派对象/间接层 |
| `gemma_norm_patch.py`（53 行）：给 vLLM 真实的 `GemmaRMSNorm` 类缓存 `weight.float()+1.0` | vLLM 的 Qwen3.6 模型定义**内部复用**了 Gemma 的零中心 RMSNorm 约定（不是真的 Gemma 模型），161 个实例 × ~2 次调用/步造成冗余 kernel launch | 需改写 | 当前 `runtime/` 里没有 `weight+1.0`/零中心 RMSNorm 模式。*缓存思路*（frozen 推理期参数，算一次派生张量）可套用 `fp8_linear.py` 自己的 `_ensure_ready()` 懒初始化模式，但需要新写一个 `GatedRMSNorm`/零中心变体（可放进 `runtime/kernels/fused_rms_norm.py` 或新文件） |
| `__init__.py` × 2（6 + 1 行） | 只是打包/隔离声明 | N/A | 无需移植，确认重建应该是干净重写，不是从 oracle import |

### 1.9 已删除、但可从 git 历史恢复的相关文件（不在 `oracle/` 里，但是同一批工作的产物）

`runtime/model/nvfp4_linear.py` 与 `runtime/nvfp4_custom_gemm.py` 曾经存在于**当前框架
的 `runtime/` 目录下**（不是 oracle），在 `a9cb932`（"Isolate retired Qwen runtime from
Laguna distribution"）里被判定为死代码删除。可用 `git show a9cb932^:runtime/model/nvfp4_linear.py`
取回。这不是 oracle 参考代码，是**几乎完工的自建 NVFP4 Linear 模块**，针对
compressed-tensors 格式（从 vLLM 的 `CompressedTensorsW4A4Fp4` + `CutlassNvFp4LinearKernel`
提炼）：

- `swizzle_blockscale`/`pad_nvfp4_weight_for_cutlass`/`slice_nvfp4_output`——纯张量操作，
  零 vLLM 依赖，**可直接搬**。已核实 `swizzle_blockscale` 的 128 行/4 列 swizzle 模式与
  sparkinfer 自己的 `sparkinfer._lib.intrinsics.swizzle_block_scale` **结构相同**（硬件定义
  的 CUTLASS SM1xx block-scaled MMA 布局，不是框架约定，这是预期之中的一致）。
- 唯一未完工的缝：激活侧量化调用 `torch.ops._C.scaled_fp4_quant`——**vLLM 编译的 CUDA
  extension 符号**，当前 `runtime/` 全树零引用，零 vLLM 世界里这个 op 不存在。**这是整条
  GEMM 路径里最具体的单个空白**：需要换成 `sparkinfer.quantization.nvfp4`
  （`sparkinfer/quantization/nvfp4/api.py`：`plan(m,k)`/`allocate_outputs`/`run`，
  CUTLASS-DSL TMA-based bf16→packed-FP4+scale 量化器，SM120/121 已支持）。
- **格式警告**：这份被删代码的参数命名（`weight_packed`/`weight_global_scale`/
  `input_global_scale`/跨分片 `.max()`-merge）是 compressed-tensors 专属。Qwen3.6 的
  modelopt checkpoint 命名/scale 语义不同——需要照真实 modelopt checkpoint 逐项确认
  （本轮未定位到 modelopt 参考实现，不猜字段名，见第 7 节待验证清单）。

### 1.10 汇总：分布与"已经休眠就位"的机制

四类判定的粗略分布（按行数）：约 15% 可直接搬（`gdn_state.py` 的快照/恢复对、
`mtp_accept.py`、`compute_causal_conv1d_metadata`、网络工具、CSR 常量）；约 45% 需改写
（GDN 状态生命周期管理、metadata builders、MTP 编排、prefix cache 跨请求共享、NVFP4
GEMM 与 RMSNorm 补丁的可迁移部分）；约 15% 已被取代（该抄 Laguna 的成熟实现而不是老
代码，CUDA graph、accept/reject、chunked-prefill 契约形状）；约 25% 应废弃（vLLM
kernel-选择补丁类，sparkinfer/自建模型图让这类问题结构性消失）。**模型数学本身
（GDN forward/RoPE/gate/MLP/modelopt 反量化）不计入以上四类，是纯新写**。

**已经在当前框架里休眠、只差接线的机制**（重建时不要重新发明）：

| 机制 | 位置 | 状态 |
|---|---|---|
| GDN 状态×投机解码 K+1 专用行寻址（`_ssm_spec_row`） | `runtime/block_pool.py:23-24,45-79` | 原样存在，未删除，Laguna 未使用（无 GDN） |
| GDN checkpoint 驱逐挂钩（`_on_evict_block`） | `runtime/block_pool.py:326-329` | 通用回调，值为 `None`，等一个 `evict_gdn_checkpoint` 实现接进去 |
| MTP accept/reject 判定算法 | `runtime/mtp_accept.py`（130 行） | **已完全移植完成**，`oracle/qwen36_vllm/backends/qwen36.py` 当年调用的就是这份逻辑 |
| conv1d 元数据构造 | `oracle/qwen36_vllm/attention_compat.py:compute_causal_conv1d_metadata`（99-150） | 已是零 vLLM 自写，可直接搬（未挪进 `runtime/`，是唯一一处还需要"搬家"动作的可直接搬代码） |
| NVFP4 权重侧张量操作 | `git show a9cb932^:runtime/model/nvfp4_linear.py`（`swizzle_blockscale` 等） | 已删除但可恢复，零 vLLM 依赖 |
| Qwen36 backend 注册表条目 | `runtime/model_registry.py:69-73`（`Qwen3_5ForConditionalGeneration → backend="qwen36"`） | 已注册，`IMPLEMENTED_BACKENDS` 未包含它（第 76 行注释"Track B flips `qwen36`"）|

---

## 2. 验收基线：Qwen3.6-vLLM 时代实测数字

> 全部来自 vLLM 剥离前（`ff4d858`/`a9cb932`，2026-07-30）的真实测量，运行在
> RTX PRO 6000 Blackwell（SM120，96 GB/97887 MiB）。**这些数字是重建的下限**——
> 新实现打不平就是退步。找不到记录的明确写"无记录"。见
> [`../notes/2026-08-02-laguna-docs-inherited-qwen36-numbers.md`](../notes/2026-08-02-laguna-docs-inherited-qwen36-numbers.md)：
> `docs/roadmap.md:27`、`docs/model-support.md:49`、`README.md:79` 当前把下面这批
> Qwen3.6 数字中的两个（MMLU-Pro 84.54%、容量表 93GB/70GB）错标成了 Laguna 的数字——
> 引用时认这份文档，不要拿那三处当"独立确认"。

### 2.1 吞吐

**长上下文（128K/64K，concurrency=4，MTP K=3，warm 前缀缓存）——headline 终值**：

| 场景 | 终值 | 测量日期 | 来源 |
|---|---|---|---|
| 128K, c=4, warm | **222.44 tok/s**（warm）/ 226.03（cold） | 2026-07-21 深夜，KWIDE+V272 kernel 优化 | `docs/archive/2026-07-20-PROGRESS.md:4239-4244` |
| 64K, c=4, warm | 236.69 tok/s（PROGRESS.md 自己独立记录的最后一个值）；文档汇编里回声为 267 tok/s | 2026-07-21（236.69）；文档编制 2026-07-22/25（267 的回声） | `PROGRESS.md:4185-4189`；`docs/archive/2026-07-30-architecture-two-tenant.md:444` |

> **267 tok/s 置信度较低**：只在架构文档里回声出现，PROGRESS.md 自己没有对应的独立
> 64K 测量条目（最后一条独立记录是 236.69）。`docs/roadmap.md:84` 已经指出"README 里的
> 222/267 tok/s 是旧数字"（那是指这两个数字曾被误搬进当前 Laguna 章节，现已修正，
> 与本节把它们当 Qwen3.6 历史基线引用不矛盾）。**引用 128K 用 222.44，引用 64K 优先
> 用 236.69（更可信），267 仅作参考**。

完整优化轨迹（用于理解基线是怎么来的，不是要复现每一步）：104.7 → 105.4 → 120.8 →
154.7-157.3 → 166.3 → 176.45 → 183.43 → **222.44**（128K）；92.3 → 132.0 → 201.4 →
225.5 → **236.69**（64K），跨越 2026-07-19 至 2026-07-21，逐次优化点：INV8 分块前缀命中、
NATIVEFP8 默认+split-KV=64、CUDA graph 内存修复、numpy `_fill_buffers`、Triton RMSNorm
混合 dtype 修复、VTRANSPOSE_ELIM、split=32。同期原生 vLLM/FlashInfer 参考：128K warm
117.4–146.85 tok/s，64K warm 175.2–222.17 tok/s（噪声大，多次测量）。**200K/c=4 双方均不可行**
（>95GB KV，`PROGRESS.md:143,3487,3507,3545,3599,3676`）。

**短上下文（W1-S，4096 in/256 out，c=4，K=3）**：原生参考 **144.54 tok/s**（2026-07-17，
`PROGRESS.md:1941`）；自研实现从 11.60 一路优化到 **142.504–166.022 tok/s**（多轮
回归门禁复测区间，2026-07-18~20），与原生的差距在优化后收窄到 1.014×。

**容量阶梯（D1，cold，batched+cudagraph，2026-07-18/19）**：

| 上下文/c=4 | 自研 | 原生 | 备注 |
|---|---:|---:|---|
| 32K | 29.522 | 32.941 | 2026-07-18 |
| 64K（chunked cold） | 13.386 | 10.800 | 2026-07-19 |
| 128K（chunked cold） | 5.014 | 3.270 | |
| 256K（chunked cold） | 1.557 | 0.580 | 双方均可行，无 OOM，82.8% 峰值显存 |

### 2.2 投机解码接受率（Qwen3.6 内置 MTP，K=3）

多次测量方法/配置不完全一致（不同上下文长度、采样方式、一次已定位的计数 bug），
**取最后一次记录作 headline**：

| 值 | 配置 | 测量日期 | 来源 |
|---|---|---|---|
| **50.3%**（约每轮产出 2 个 token） | 128K, c=4, MTP K=3, warm | **2026-07-21（终值，与 222.44 tok/s 同批测量）** | `PROGRESS.md:4241` |
| 自研 66.7%（tokens/step=3.0）vs 原生 64.2%（mean_accept_len=2.926）——自研略优 | 128K, c=4, warm，修正双计数 bug 后 | 2026-07-21 | `PROGRESS.md:4013-4015` |
| 自研 70.29%（多次回归复测重复出现的锚点值） | 4K, c=4, W1 headline | 2026-07-18~20 | `PROGRESS.md:114,232,453,480` |

> 与 Laguna 当前 DFlash 96.3–100% 接受率不可直接对比——机制完全不同（checkpoint 内 MTP
> K=3 vs. 独立 draft 模型 DFlash K=15），且 Qwen3.6 的数字本身在不同测量批次间有
> 3+pp 波动（含一次已定位的测量 bug）。**重建后的验收目标不是"匹配某个具体百分数"，
> 是"匹配贪心输出与 target 逐 token 一致"（无损性）+ "接受率不低于历史区间下限
> 约 50%"**。

### 2.3 质量

| 指标 | 值 | 配置 | 测量日期 | 来源 |
|---|---|---|---|---|
| MMLU-Pro | **84.54% vs 官方 86.2**（−1.7pp，414 题子集 ±3.5% 抽样噪声内） | 414 题分层抽样，thinking 模式，5-shot CoT，greedy，max_tokens=32768，零截断 | **2026-07-22** | `notes/2026-07-22-quality-baseline-and-official-scores.md:220`；`README.md:96` |
| HumanEval | 自研 44.5%（73/164）vs 原生 vLLM 43.3%（71/164），+1.2pp | greedy, evalplus, max_tokens=768（**方法论已知问题**：截断偏低估两侧同等，非系统性退化） | 2026-07-21 | `notes/2026-07-22-quality-baseline-and-official-scores.md:112-121` |
| HumanEval+ | 自研 43.3%（71/164）vs 原生 42.7%（70/164），+0.6pp | 同上 | 2026-07-21 | 同上 |
| AIME26 / GPQA Diamond | **无记录**——本地测过但未完成/被撤销 | — | 尝试于 2026-07-22，后撤销 | `docs/archive/2026-07-26-roadmap-vllm-removal.md:52`（"AIME26 评测撤销"）；commit `cd186fe` "Not-tested" 段 |

官方/外部参考（非本地测量）：MMLU-Pro 86.2、AIME26=94.1、GPQA Diamond=87.8、
LiveCodeBench v6=83.9、SWE-bench Verified=77.2（Qwen 官方模型卡）。官方卡**不发布**
HumanEval/HumanEval+ 数字，故上表的"vs 原生 vLLM"对比是同权重 A/B，不是 vs 官方。

**HumanEval+ max_tokens=768 的重新测量（4096 token）从未出现**——`notes/2026-07-22-quality-baseline-and-official-scores.md`
自己说要重锁基线，但搜索范围内没有对应结果，标 **[无记录]**。

### 2.4 GPU 显存占用

| 场景 | 值 | 测量日期 | 来源 |
|---|---|---|---|
| 128K, c=4, warm | 92.9 GiB 峰值（chunking 前）→ 90.7 GiB（chunking 后） | 2026-07-20 | `PROGRESS.md:77,106` |
| 64K, c=4, warm | 63–65 GiB | 2026-07-20 | `PROGRESS.md:135,3499,3639` |
| 200K, c=1, cold-populate（非 chunked） | 97.3 GiB（**溢出到 CPU 共享内存，被节流**——已知测量局限，非典型工况） | 2026-07-20 | `PROGRESS.md:202,214-217` |
| 256K, c=4, cold, chunked | 82.8% 峰值（可行，无 OOM） | 2026-07-19 | `PROGRESS.md:711,718` |
| GDN 快照缓冲区（4 槽） | ~604 MB VRAM | 设计记录，非端到端占用 | `PROGRESS.md:2923-2924` |

### 2.5 最大上下文

架构上限 **262144 tokens（256K）**，与 checkpoint `max_position_embeddings` 精确对齐
（`PROGRESS.md:716-717`）。**256K/c=4 实测可达成**（cold, chunked, 2026-07-19，
1.557 tok/s，82.8% 峰值显存，无 OOM）。200K/c=4 在两侧均被判定**不可行**（>95GB），
是显存天花板不是架构限制。

### 2.6 GDN kernel 级数据（决定重建时的优化优先级）

来自 `notes/2026-07-22-a1a-gdn-profiling.md`（2026-07-22，RTX PRO 6000 Blackwell，
K=3 MTP，eager 模式）：

| 上下文 | GDN 占 decode GPU 时间 | NVFP4 GEMM 占比 | Attention 占比 |
|---|---:|---:|---:|
| 4K（c=1） | 5.1%（fused_sigmoid_gating 2.0% + rms_norm 2.0% + causal_conv1d 0.6%） | **71.1%** | 3.5% |
| 128K（c=1） | 3.9%（delta_rule 1.5% + rms_norm 1.5% + conv1d 0.5%） | **53.7%** | **28.2%** |

**M2 优先级结论（当年的，仍然成立）**：A2（NVFP4 GEMM autotune）> attention 优化 >
MTP 链路融合 >> GDN kernel 融合。GDN 48 层合计从未超过 5.1%，**GDN kernel 本身不是
重建的性能瓶颈**，先把 NVFP4 GEMM 与 attention 做对，GDN 用现成的 FLA Triton kernel
够用（见第 5 节 B0-4）。

### 2.7 数字使用须知

- 上述吞吐/接受率数字都存在**测量间波动**（不同批次的采样、上下文长度组合、
  以及至少一次已定位的计数 bug），本节统一取"轨迹终值"作 headline，同时保留轨迹
  以便判断噪声量级——重建验收时应比"终值区间"，不是比单次数字。
  跨运行比较前先按仓库纪律 `bf diff`（当年没有 `bfdiag`，这是新框架相对当年的
  改进，重建时应该从第一天就接入）。
- 本节所有数字都是**在 vLLM 执行路径上**测的，包含 vLLM 自己的调度/内存管理开销。
  新实现走 Track A 抽象后的开销分布会不同（更接近 Laguna：无 vLLM scheduler、无
  `ForwardContext` 全局态），因此吞吐数字理论上有改善空间，但这是假设，不是承诺。

---

## 3. 用今天的框架重新设计

### 3.1 起点：Track A 已经落地了什么

`runtime/backends/protocol.py`（299 行，`ModelBackend` Protocol，13 个必需成员 +
`BackendCapabilities`）、`runtime/architecture.py`（319 行，`ArchitectureSpec`，已经
正确解析 Qwen3.6 真实 `config.json`：`layer_types`→`CACHE_PAGED_KV`/`CACHE_RECURRENT`
逐层区分、`attn_output_gate`、`partial_rotary_factor`、mrope `rope_parameters`、
modelopt quant、MTP 层数）、`runtime/model_registry.py`（151 行，已注册
`Qwen3_5ForConditionalGeneration → backend="qwen36"`，但 `IMPLEMENTED_BACKENDS`
只有 `{"laguna"}`）——**这三个文件是影子模式，已经过 `TestAgainstRealCheckpoints`
针对 4 个本地 Qwen3.6 checkpoint 验证过解析正确性，但目前没有任何东西真正驱动它们**。
`tests/test_architecture_spec.py`/`test_backend_protocol.py` 已经存在，是重建的
起点测试，不是要新写的测试。

**重建的第一条硬约束**：新实现必须是 `IMPLEMENTED_BACKENDS` 里的 `"qwen36"` 值指向的
那个类，必须通过 `runtime/backends/protocol.py::check_conformance()` 校验，**不是**
把 `DirectModelRunner` 整个搬过来再让 `ServerEngine` 认出它。

### 3.2 `Qwen36Backend` 要实现协议的哪些成员

对照第 1 节的移植结果，逐协议成员标注可用度：

| 协议成员 | 可用度 | 依据 |
|---|---|---|
| `capabilities` | 新写，简单 | `speculative_decode=True`（MTP）、`prefix_cache`（先 False，见 3.3）、`cuda_graph`（先 False，见 3.5）、`chunked_prefill=True`、`warm_continue`（先 False，同 N8 的 (c) 选项） |
| `reset_slot`/`slot_state`/`snapshot` | 需改写 | 1.4 节的槽位记账部分 + GDN 状态部分需要合并进一个实现，`SlotStateView`/`BackendSnapshot` 形状已固定 |
| `prefill`/`decode_batch_sampled` | 需改写 + 新写 | 编排抄 1.4 节，模型 forward 本身是新写（1.0 节） |
| `prefill_chunked_begin/_step` | 需改写，且要**决定做不做 Phase B** | oracle 自己都没做完跨步交织（1.4 节），Laguna 也没做；B1（正确性优先）阶段可以先做 Laguna 式的 stub（非增量），B2 再评估是否需要 |
| `reconcile_prefix_hit`/`find_best_slot_for_prompt` | 需改写，依赖 3.3 的协调者设计 | Qwen3.6 需要 GDN 状态参与前缀命中判定，不能照抄 Laguna 的同槽线性比较 |
| `has_speculative_decode`/`enable_dflash`/`mtp_verify_and_commit_batch` | 需改写（命名待定），accept/reject 逻辑**已经可直接搬** | `runtime/mtp_accept.py` 已经是这套逻辑的移植终点，见 1.10 |
| `capture_decode_cuda_graph` | 需改写，且是 B3 的门票 | 见 3.5 的"状态中立捕获"设计缺口 |
| `mtp_prefill_warm_continue` | 需改写，或按 N8 的 (c) 选项先拒绝 | 协议已预留（1.3 节），但 roadmap N8 已拍板 (c)：能力查询为 False 时启动期拒绝该 flag，不是现在就要做 (a) |

### 3.3 缓存资源协调者（对应 A3，读 `hybrid-cache-prior-art` 之后的框架）

`notes/2026-08-01-hybrid-cache-prior-art.md` 对 A3 的六条修改，落到 Qwen3.6 具体是：

1. **不做"统一分配器"**——分页 KV 用现有 `BlockPool`，递归状态需要一个新的、独立的
   `RecurrentStateAllocator`（每请求一个固定槽，不分页，形状抄 SGLang 的
   `MambaSlotAllocator` 而不是 `BlockPool`）。`_ssm_spec_row`（已在 `block_pool.py`
   休眠，见 1.10）是这个新分配器要用的行寻址原语，**不是**要塞进 `BlockPool` 本体。
2. **前缀匹配返回两个数字**：`find_prefix_match → (kv_hit, state_hit)`。GDN 没有
   前缀概念，只有终态 checkpoint，搜索方向与 attention KV 相反（右到左找最近可用
   checkpoint），`state_hit ≤ kv_hit`，两者之间的 token 即使 KV 命中也要为递归层重算。
3. **投机解码保守释放**：释放递归状态槽前，先假设本轮全部草稿 token 被拒绝
   （`num_computed_tokens -= num_speculative_blocks`），代价是有限的显存浪费，
   换来正确性——这条直接对应本文档 §5 的"最难项之一"。
4. **同一轮内不可跨请求借用递归状态**：两个请求在同一轮命中同一个全新前缀，
   只有一个能真的拿到状态命中，另一个必须假报"槽位不足"退到下一轮——这是
   **调度层**约束，`ServerEngine` 今天没有这个概念，需要新增。
5. **逐资源驱逐预算，命中测试独立验证两个资源**：KV 存活但递归状态被驱逐仍可用
   （只要状态有 backup），不是"一个没了另一个也必须没"。
6. **块大小对齐**：分页 KV 与递归状态的"命中长度"要能用同一单位表达，需要显式选
   对齐粒度，不是隐式假设两者一致。

**GDN checkpoint 驱逐挂钩已经就位**（`BlockPool._on_evict_block`，见 1.10），
但按上面第 1 条，**它可能不是这个协调者最终会用的机制**——`_on_evict_block` 假设了
"一个分配器驱逐时通知另一个"的耦合模式，而先例研究明确建议两个分配器独立、协调者
在上面管理不变量，不是分配器互相调用。这条挂钩是否直接复用还是被协调者的新设计
取代，是 A3 落地时要拍板的具体点，不是本文档预判。

### 3.4 加载器 adapter：modelopt vs compressed-tensors

`runtime/model_registry.py` 已经把 `"modelopt"` 列为 `LOADER_FOR_QUANT_METHOD` 的
一个值（第 41-44 行），但目前只有名字，没有实现。需要新写：

- **权重侧**：`git show a9cb932^:runtime/model/nvfp4_linear.py` 的
  `swizzle_blockscale`/`pad_nvfp4_weight_for_cutlass`/`slice_nvfp4_output` 三个纯
  张量函数可直接搬（1.9 节），但参数命名要从 compressed-tensors 风格
  （`weight_packed`/`weight_global_scale`）换成 modelopt 的真实命名——**本轮未确认
  modelopt 的确切张量名/scale 语义**，是 B0-2 的原始任务，不能猜（`roadmap.md`
  §待验证清单已列）。
- **激活侧量化**：`torch.ops._C.scaled_fp4_quant`（vLLM 编译扩展）必须换成
  `sparkinfer.quantization.nvfp4`（`plan(m,k)`/`allocate_outputs`/`run`）——这是
  整条 GEMM 路径里最具体的单个空白，1.9 节已定位。
- **GEMM 本身**：`sparkinfer.gemm.blockscaled.mm` vs 复活
  `runtime/kernels/nvfp4_gemm_sm120.cu`（自研，源码还在，1.8 节），需要一次
  GPU A/B（**[待验证]**，见第 7 节）。
- **333 个 vision 张量过滤**：D6 已拍板用官方 `nvidia/Qwen3.6-27B-NVFP4`，需要一个
  按 tensor 名前缀跳过 `vision.*` 的加载过滤器（一次性机械工作，`roadmap.md` B0-1a
  已列，可复用于任何带 vision tower 的衍生 checkpoint）。

### 3.5 CUDA Graph：状态中立捕获是全新问题

1.4 节已确认：Laguna 的 decode 图从不触碰递归状态，其 warmup 复用天然安全，**这条
问题在新框架里没有现成参照**。oracle 当年的解法（`2 × batch_size` 专用 warmup 槽，
`docs/archive/2026-07-30-architecture-two-tenant.md:348`）值得抄，但要重新在自建
CUDA graph 骨架（`LagunaCudaGraphDecode`/`LagunaCudaGraphVerify`）上验证一遍，
而不是假设"照抄就对"。**这是本文档判定的三大难点之一**（见第 6 节）。

### 3.6 bfdiag 耦合：先修协议依赖，Track B2 才能用诊断平台

`architecture.md` §3.5.4 已经指出 `bfdiag/daemon/provider.py`
（`LagunaEngineProvider`）直接 import 具体的 `LagunaBackend`/`DFlashEngine`，摸
`_decode_cg`/`_moe_sparkinfer_layers`/`static_forward_context` 等私有属性。**这不是
Qwen3.6 特定问题，是 Track A 步骤 5-6（迁移顺序表）的通用前置**——但对 Track B2/B3
有直接影响：`bfdiag` 的热引擎/run record/`bf diff` 是这个仓库唯一的性能与正确性
验收工具，B2/B3 的门禁（"接受率与吞吐进 bfdiag 基线"）**要求 bfdiag 先能托管
`Qwen36Backend`**，否则验收无从谈起。排期上，Track A 的协议迁移（尤其
`bfdiag/daemon/provider.py` 改为按协议持有）应该在 B2 开始前完成，不是可以延后到
B2 期间顺手做的小事。

---

## 4. 已确认的事实（汇总，逐条标来源）

- **架构事实**（config.json 实测）：64 层 = 48 linear_attention(GDN) + 16
  full_attention（interval 4）；hidden 5120；head_dim 256；24 q 头/4 kv 头（GQA 6）；
  partial_rotary_factor 0.25；mrope interleaved；`attn_output_gate: True`；稠密
  SwiGLU intermediate 17408；`mtp_num_hidden_layers: 1`；modelopt NVFP4 + fp8 KV。
  来源：`runtime/architecture.py` 已解析并通过 `TestAgainstRealCheckpoints` 验证
  （`tests/test_architecture_spec.py`），`docs/model-support.md` §3.1 交叉确认。
- **B-6 结论（不可推翻，只核实）**：6 个本地 checkpoint 的 `mtp.*` 张量清一色
  `self_attn.*`+`mlp.*`，零 GDN。但主模型 48 个 GDN 层在 verify 时照样跑，被拒
  token 照样要回滚——这条**不消除**第 6 节的最难项。来源：
  `notes/2026-08-01-b6-mtp-gdn-verification.md`。本轮 GDN 状态判定表（1.1/1.2 节）
  与此结论完全一致，未发现矛盾证据。
- **C-2 结论**：sparkinfer 的 paged kernel 显式拒绝 fp16/bf16/fp8_e4m3 之外的 KV
  dtype，NVFP4 KV today 不存在可测对象。B3 用 FP8 KV。来源：`investigation-queue.md` C-2。
- **D6 已拍板**：主线 checkpoint 用官方 `nvidia/Qwen3.6-27B-NVFP4`，需排除 333 个
  vision 张量，`validate_text_only` 已经是"接受但断言零 vision 张量被加载"的语义
  （`runtime/architecture.py:292-319` 已实现，非空文档承诺）。
- **C-3 结论**：PyPI `torch==2.13.0` 带 `sm_120`（`2.13.0+cu130`，实测确认）。
- **本轮新确认**：`fla`（flash-linear-attention v0.5.2）本地可 `import`，
  `fla.ops.gated_delta_rule.chunk`/`fused_recurrent` 两条路径均可导入成功，**无需**
  `causal_conv1d`（该包在本机测试 venv 里根本没装，import 时未报错，说明这两条
  gated_delta_rule 路径不在其关键依赖链上）。但 `pyproject.toml` 未声明 `fla` 依赖，
  且**从未在 SM120 上实跑验证**（`investigation-queue.md` §F 已记录：FLA 的 Blackwell
  相关 bug 全部是 B200/SM100，无 SM120 记录，与"未验证"一致，不是"已知能跑"）。
- **本轮新确认**：`git show a9cb932^:runtime/model/nvfp4_linear.py` 与
  `runtime/nvfp4_custom_gemm.py` 是当前框架 `runtime/` 目录曾经存在、后被判定为
  死代码删除的 NVFP4 Linear 原型，可恢复，见 1.9 节。
- **本轮新确认**：`runtime/block_pool.py` 里没有需要清理的 GDN 专属残留代码——
  `_on_evict_block` 是通用回调，`_ssm_spec_row`/`_physical_slot` 是通用寻址原语，
  两者都是干净、可直接复用的挂钩，不是"死代码待删"（与 `roadmap.md` §1.5-S4/
  `architecture.md` §2.4 的"残迹"措辞略有出入——它们是**休眠但设计良好的挂钩**，
  不是需要先清理的垃圾）。

---

## 5. 风险与未知

### 5.1 最难的三件事（按判断的难度排序）

1. **GDN 状态×投机解码回滚（主模型侧）**——`docs/roadmap.md` §2.3 与
   `investigation-queue.md` B-6 已经定性为本轨道最难项，本轮判定维持这个结论，
   但补充了两个具体好消息：①寻址方案（`_ssm_spec_row`）已经原样存在于
   `block_pool.py`，不需要重新设计；②accept/reject 决策函数（`mtp_accept.py`）已经
   完全移植完成。**真正剩下的难点是 CUDA graph 状态中立捕获**（3.5 节）——GDN
   状态非幂等这条约束在自建 CUDA graph 骨架上**从未验证过**，Laguna 的图从不碰
   递归状态，没有参照可抄。这是本文档判定的**第一难点**，因为它同时卡住 B2（服务化）
   与 B3（投机）两个阶段。
2. **GDN 状态×前缀缓存联动驱逐**——是 A3 的第一个真实用户（`architecture.md`
   §3.5.5 步骤 7 明确写"在 Track B 的递归状态到来之前它没有真实消费者"）。
   `hybrid-cache-prior-art.md` 的六条修改（3.3 节）都还只是设计，没有一条在这个
   框架里实现过。风险不在"知道怎么做"（先例研究已经把坑列全了），风险在**六条
   修改要同时正确，任何一条漏了都是"许多 token 之后才显形"的那类最难查 bug**
   （`architecture.md` §3.2-C 原话）。
3. **模型数学本身是纯新写，且验证它对不对没有捷径**——1.0 节的纠偏意味着 B1
   （正确性优先）阶段不是"移植+调试"，是"从 vLLM 上游源码读懂算法+从零实现+
   逐层 cosine 相似度对 HF transformers"。GDN 层前向（gated delta rule + conv1d +
   输出门）、mrope-interleaved 部分旋转、modelopt 反量化——任何一个写错都可能表现
   为"能跑但输出漂移"而不是崩溃，B1 的逐 token 对齐门禁存在正是为了抓这类错误，
   但抓到之前排查成本高（这台机器一次验证以分钟计，见 `AGENTS.md` 诊断纪律）。

### 5.2 其余风险（按判断的确定性排序，非难度）

- **NVFP4 GEMM 到底选自研 `.cu` 还是 sparkinfer**——需要一次 GPU A/B（3.4 节），
  在此之前无法判断这条工作是"选一个现成的"还是"两边都要调"。
- **`RESERVED_PHYSICAL_SLOTS=1` 是否还需要**——1.4 节已指出这可能是纯 vLLM 调度器
  伪影，Laguna 用 0 也能跑，但"新实现要不要保留 1"需要在自建栈上实证检查，不能
  两头都假设。
- **`_MAX_DECODE_QO_LEN=16` 等常量绑定旧 kernel 的测试范围**——换新 GEMM/attention
  kernel 后这类边界常量需要逐个重新核实，不能整包沿用。
- **modelopt 张量命名/scale 语义未确认**（B0-2，本轮未做，不是本文档职责范围但
  是紧邻的下一步）。
- **chunked prefill 跨步交织（Phase B）连 oracle 自己都没做完**——B1/B2 阶段可以
  先用 Laguna 式的非增量 stub，B2 结束前要明确决定是否补这块，不要放到 B3 才发现
  服务化门禁需要它。

---

## 6. 待验证清单（本轮不动 GPU，明确列出留给下一步）

- [ ] sparkinfer `blockscaled.mm` vs 自研 `nvfp4_gemm_sm120.cu` 在 Qwen3.6 真实
  稠密 shape（34816/17408/6144/5120/96）上的 A/B（3.4 节，1.8 节）
- [ ] FLA `gated_delta_rule`（chunk / fused_recurrent 两条路径）在 SM120 上的正确性
  与速度实测——本轮只确认了本地可 `import`，未做任何 GPU 执行
- [ ] GDN 递归状态更新在自建 CUDA graph 骨架上是否 capture-safe（3.5 节，第 6.1 节
  第一难点的具体验证动作）
- [ ] `RESERVED_PHYSICAL_SLOTS=1` 在自建栈上是否仍必要（5.2 节）
- [ ] `runtime/kernels/fused_rms_norm.py` 的 `BLOCK_SIZE`/`num_warps` 假设在
  Qwen3.6 hidden=5120 下是否仍安全（1.8 节）
- [ ] modelopt NVFP4 的 tensor 命名与 scale 语义逐项确认（B0-2，紧邻下一步，非本文档产出）
- [ ] Qwen3.6-27B 在 96 GB 上的 context × 并发可行域——**本文档第 2.4/2.5 节的数字
  是 vLLM 执行路径下测的**，新框架的 KV/递归状态显存记账方式不同（参照 Laguna 的
  `notes/2026-07-29-gpu-memory-audit.md` 式审计），需要重新测，不能直接套旧数字

---

## 7. 需要人拍板的事项

- **`oracle/qwen36_vllm/` 的处置时机**——`roadmap.md` §7 D5 已列出三个选项
  （保留只读参考 / B 完成后删除 / 现在就删）。本文档大量引用了它的具体行号，
  **建议在 Track B3 验收通过前保持 (a) 不变**，否则本文档的可核查性会打折。
- **B0-2（modelopt 张量命名确认）与 B0-4（GDN 方案三选一：FLA/移植/自研）的排期
  顺序**——本文档判定 GDN kernel 本身不是性能瓶颈（2.6 节），建议 B0-4 直接选①
  FLA 拿正确性，把"要不要自研③"推迟到 B3 profiling 之后再看，这条 roadmap 已经
  这么建议，本文档只是补充了量化证据（GDN 恒占 <5.1% decode 时间）支持这个建议，
  不改变它。
- **`docs/roadmap.md:27`、`docs/model-support.md:49`、`README.md:79` 的数字污染
  修正**——本文档第 2 节引用的 Qwen3.6 数字（MMLU-Pro 84.54%/86.2、容量表）目前在
  这三处被误标成 Laguna 当前数字，已记录在
  [`../notes/2026-08-02-laguna-docs-inherited-qwen36-numbers.md`](../notes/2026-08-02-laguna-docs-inherited-qwen36-numbers.md)，
  但**未在本次改动范围内修复**（这三个文件不在本任务授权改动范围）——需要人决定
  谁来修，以及是"重新标注为历史数字"还是"重新测一遍 Laguna 自己的数字"。这条
  直接影响 `docs/roadmap.md` §0 的论证依据（"Laguna 模型能力经评测判断为一般"
  这句话目前引用的是 Qwen3.6 的分数，不是 Laguna 的）。

---

## 8. 配套文档

- [`../oracle/qwen36_vllm/`](../oracle/qwen36_vllm/) —— 本文档第 1 节判定的对象，
  只读参考，生产代码不能 import
- [`../docs/archive/2026-07-20-PROGRESS.md`](archive/2026-07-20-PROGRESS.md) ——
  第 2 节数字的主要来源，4266 行流水账
- [`../docs/archive/2026-07-30-architecture-two-tenant.md`](archive/2026-07-30-architecture-two-tenant.md) ——
  §6.2 GDN 状态投机难题当年的解法，§12 质量/性能验证方法论
- [`../notes/2026-08-01-hybrid-cache-prior-art.md`](../notes/2026-08-01-hybrid-cache-prior-art.md) ——
  A3 协调者设计的六条修改，第 3.3 节的直接依据
- [`../notes/2026-08-01-b6-mtp-gdn-verification.md`](../notes/2026-08-01-b6-mtp-gdn-verification.md) ——
  MTP 不含 GDN 的结论与纠偏
- [`../notes/2026-08-02-laguna-docs-inherited-qwen36-numbers.md`](../notes/2026-08-02-laguna-docs-inherited-qwen36-numbers.md) ——
  本轮发现的文档数字污染记录
- [`roadmap.md`](roadmap.md) Track B —— 排期与里程碑，已同步本文档结论
- [`model-support.md`](model-support.md) §3 —— Qwen3.6 架构事实与接入六步流程
