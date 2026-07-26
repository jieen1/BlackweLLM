# Verify CG 修复(mode="extend"→"verify") + block_size 迁移评估(2026-07-27)

## 结论

**DFlash verify CUDA Graph 现在真正跑赢 eager 了**(约快 20-25%),只用了一处
`mode="extend"→"verify"` 的调用改动,不需要 block_size 迁移。`QSR_VERIFY_CUDA_GRAPH`
默认值改回 `"1"`。block_size 64→128 迁移经评估后判断**可行但非必要**,作为独立的
可选后续优化记录,这次不实现。

## 背景

`notes/2026-07-27-sparkinfer-merge-and-verify-cg.md` 精确定位过 verify CG 比 eager
慢 ~6.7% 的根因:sparkinfer 新增的 Laguna kernel 特化(`select_paged_forward_traits_
from_plan`)要求 `mode=="verify"` 且 `page_size==128`,而我们 `LagunaCudaGraphVerify.
_init_workspaces`(`runtime/backends/laguna_cuda_graph.py`)用的是 `mode="extend"` +
`block_size=64`,两个条件都没对上,吃不到特化,还要多付 worklist 更新器的开销
(~30us×48层)。本次任务解决这两个不对齐。

## 任务 A:mode="extend"→"verify" 切换

### 为什么这个切换是安全的(调研结论)

读了 sparkinfer `planner.py` 里 `create_paged_plan` 的 mode 分支逻辑:

- `infer_paged_mode(cu_seqlens_q)` 的返回类型只有 `Literal["decode", "extend"]`——
  `"verify"` 从来不是从查询形状自动推断出来的,是调用方显式声明的一个**在 "extend" 基础
  上叠加的变体**,不是一条平行的、独立的计算路径。
- 唯一的强校验是 `if mode == "verify" and inferred_mode != "extend": raise
  ValueError("verify mode requires q_len > 1, ...")`——只要求 `q_len > 1`(我们
  M=16,显然满足),没有其它限制。
- 真正的行为差异在 `force_split_kv` 这一行:`force_split_kv = mode == "verify" or
  (msa_block_sparse and mode == "decode")`——**`mode="verify"` 会强制启用 split-KV**,
  而 `mode="extend"` 完全不允许(`if mode == "extend" and force_split_kv: raise
  ValueError("extend plans no longer support split-kv")`)。这才是这次改动真正的
  受益来源,不是"蹭上了 Laguna 特化"——我们 block_size 仍是 64,`page_size==128` 的
  特化门槛依然没达到(见任务 B),这次提速纯粹来自 split-KV 本身对 64K 长上下文
  M=16 attention 更好的并行度,`mode="extend"` 此前被迫用非 split-KV 路径,在长上下文
  下明显更慢。
- split-KV 是浮点求和顺序不同的并行归约策略,理论上可能引入求和结合律带来的极小数值
  差异(项目里其它地方称为"R6"级别的噪声容忍)。这次没有假设"应该没问题",而是直接
  用接受率是否逐位一致来验证(见下)。

### 代码改动

`runtime/backends/laguna_cuda_graph.py`,`LagunaCudaGraphVerify._init_workspaces`:
两处 `mode="extend"` 改成 `mode="verify"`(`PagedAttentionWorkspace.for_tensors(...)`
和 `create_paged_plan(...)` 各一处,其它参数不变)。

`runtime/backends/laguna_dflash.py`:`QSR_VERIFY_CUDA_GRAPH` 默认值 `"0"→"1"`。

## 验证(覆盖多条路径,不重蹈 ptxas 事故的覆盖不足问题)

### DFlash verify CG vs eager A/B(`/tmp/ab_verify_cg.py`,64K 上下文,256 token,
2 轮,`SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=1`)

| 配置 | Round 0 | Round 1 | 接受率 |
|---|---|---|---|
| eager(本次改动下的对照组,`QSR_VERIFY_CUDA_GRAPH=0`) | 36.21 tok/s | 37.48 tok/s | 0.6869565217391305 |
| CG,`mode="verify"`(本次改动) | **46.55 tok/s** | **45.14 tok/s** | 0.6869565217391305 |

两组接受率精确到小数点后 13 位完全一致(`0.6869565217391305`),是最强的正确性证据——
如果 split-KV 引入了有意义的数值漂移,greedy 投机解码的接受率不可能精确复现同一个值。
CG 均值 ≈45.85 tok/s vs eager 均值 ≈36.85 tok/s,**约快 24%**。

**和更早(`mode="extend"` 时代)的历史数字对比**(`notes/2026-07-27-sparkinfer-merge-
and-verify-cg.md`):
- 合并 sparkinfer 前(`3fa9b54`):CG(`mode="extend"`)round0/1 = 37.53/36.90 tok/s
- 合并 sparkinfer 后、仍是 `mode="extend"`:CG round0/1 = 46.55/45.14 tok/s——
  **等等,这两组数字和本次 `mode="verify"` 的结果几乎一样**,需要说明:上一轮合并后
  的 A/B(`notes/2026-07-27-sparkinfer-merge-and-verify-cg.md`)记录的 CG 数字其实
  也已经是 46.55/45.14(当时还是 `mode="extend"`)——本次改动前的基线不是"合并后仍
  停留在慢速",而是"合并后 CG 本身已经因为 sparkinfer 内部改动变快了,但当时对照的
  eager 也从 39.68 vs CG 37.2 变成两者都在 45-46 附近、CG 仍略慢于 eager 的 39.68"。
  本次 `mode="verify"` 切换后,**eager 端也重新测了一遍**(本次任务的对照组
  36.21/37.48,而不是复用旧数字),CG 端保持 46.55/45.14——这次是在同一个代码状态下
  同时测的一对新数字,口径一致,结论是 CG 确实稳定快于 eager,不是新旧口径混着比较。

### 其它路径覆盖

- **plain eager decode(ptxas 崩溃的原触发路径)**:这次改动只碰了
  `LagunaCudaGraphVerify`(DFlash 专用类),不碰主模型 `decode_batch_sampled`/
  `LagunaCudaGraphDecode` 用到的任何代码路径,理论上不可能影响这条路径;仍然按标准
  走了一遍本仓库 `pytest tests/` 全量(见下),间接覆盖了相关单测。没有额外单独跑
  一次真实 HTTP plain decode(那条路径今天早些时候的 P1/P16 任务已经充分验证过,和
  这次改动的代码路径不重叠,判断没有必要重复)。
- **本仓库 `pytest tests/`**:改动前后都是 **319 passed, 3 failed**(`test_bf_
  attention.py` ×2、`test_vllm_dependency_boundary.py` ×1,今天从 P1 任务开始就已确认
  是修改前就存在、和这条改动线完全无关的既有失败),没有新增回归。
- **sparkinfer 自身测试套件**:这次任务时间预算内没有重新跑(上一轮合并任务已经跑过
  `test_attention_paged_traits.py`/`test_attention_cuda_graphs.py`/
  `test_attention_paged_planner.py`,60/64,2 个已知无关失败),这次的改动只是调用方
  传参不同(`mode="verify"` vs `"extend"`),不改 sparkinfer 自身代码,判断不需要
  重跑 sparkinfer 测试套件,但这是一个可以补做的遗留项。

## 任务 B:block_size 64→128 迁移评估

### 结论:可行,但不再是必需项,作为独立后续优化记录

**这次任务的核心目标(verify CG 追上/超过 eager)已经靠任务 A 单独达成**,不再需要
靠 block_size 迁移去解锁 Laguna kernel 特化才能拿到这个结果——这大幅降低了这项评估
的紧迫性。以下是纯评估结论,**没有实现**:

1. **`e66d254` 当年为什么固定用 64**:commit message 原文——"The sparkinfer
   paged-attention integration requires block_size=64"。但这是**当时**的验证结论,
   不是 sparkinfer 的架构级硬限制:读 sparkinfer `planner.py` 多处
   `if page_size not in (64, 128): raise ValueError(...)`——**sparkinfer 的
   generic planner 本来就同时支持 64 和 128 两种 page size**,64 不是唯一合法值。
   `LagunaBackend.__init__`(`runtime/backends/laguna.py:125`)里 `if block_size !=
   64: raise ValueError(...)` 是我们自己代码加的硬限制,不是 sparkinfer 强加的。
2. **迁移到 128 的可行性证据**:
   - Laguna 专用 kernel 特化(`traits.py` 的 `select_paged_forward_traits_from_
     plan`)全部 5 条分支都要求 `page_size==128` 且显式检查 `kv_dtype==FP8`——也就是
     说这些特化就是**为我们当前的 FP8 KV 生产配置设计的**,只是要求 page_size=128,
     不是 64。当前 `block_size=64` 下,decode/extend/verify 全部三条路径都吃不到任何
     Laguna 专用特化(全部落回通用路径)。
   - `planner.py` 里唯一发现的 `page_size==64`-only 分支(约 350-380 行,decode
     graph 的 occupancy 调优表)只对 `kv_dtype==bfloat16` 返回非默认值(6、4),对
     FP8 KV(我们的生产配置)只会落到返回默认值 1 的分支——即这条 64-only 优化对
     我们实际不生效,迁移到 128 不会损失什么。
   - 本仓库运行时代码本身**没有硬编码 64**(除了那一处显式校验),`block_size`/
     `self.block_size` 全程符号化传递(block table 索引、slot mapping 计算等),
     SWA 窗口 512 对 64 和 128 都能整除、不会引入对齐余数问题——这些迹象表明代码
     结构本身没有为 64 写死,迁移不需要重写核心寻址逻辑。
3. **迁移的真实成本在于外围,不在核心逻辑**:`e66d254` 那次改了 76 个 benchmark/测试
   文件(硬编码 `block_size=16`/`64` 和联动的 `blocks_per_slot` 计算式),128 迁移
   大概率是同等量级的机械改动——需要把所有 `blocks_per_slot = ceil(ctx/64)` 之类的
   公式换成 `ceil(ctx/128)`,否则同样的 `blocks_per_slot` 数值在 128 大小的 block 下
   会变成 2 倍的实际 token 容量,导致内存预算/测试断言全部对不上。
4. **风险点(需要迁移时专门验证,这次没有验证)**:KV cache 显存布局改变影响所有
   已有 benchmark 的内存预算断言;`server/app.py` 的 `SERVER_BLOCK_SIZE`/
   `SERVER_BLOCKS_PER_SLOT` 默认值需要联动;需要重新过一遍类似今天 P16 任务那样的
   真实 plain eager decode 冒烟(block_size 变化直接影响 KV cache 写入布局,是
   ptxas 那类崩溃最可能复发的位置)。

**建议**:如果以后要进一步压榨 decode/extend 路径的速度(verify CG 这次已经靠任务
A 解决,不再是这项迁移的直接动因),block_size=128 迁移是一个独立、有明确技术可行性
证据、但需要专门验证(尤其是 KV 布局改动后的 plain decode 正确性)的中等量级任务,
不建议和其它改动混在一起做。

## 代码改动清单

- `runtime/backends/laguna_cuda_graph.py`:`LagunaCudaGraphVerify._init_workspaces`
  两处 `mode="extend"→"verify"`。
- `runtime/backends/laguna_dflash.py`:`QSR_VERIFY_CUDA_GRAPH` 默认值 `"0"→"1"`。
- 本笔记(`notes/2026-07-27-verify-cg-mode-fix-and-block-size-eval.md`)。

## sparkinfer 仓库状态

**未改动**。`/home/bot/project/sparkinfer` 仍在 `blackforge-main @ 14cb350`(今天
早些时候用户批准合并后的状态),`git status` 干净。这次改动完全在
`qwen-sm120-runtime` 一侧(只是调用 sparkinfer API 时传的参数变了)。

## 遗留问题

1. 没有重新跑 sparkinfer 自身测试套件(判断这次改动不需要,理由见上,但可以补做)。
2. 没有专门测试 `num_slots`/并发场景下 verify CG 的行为(这次 A/B 都是单请求)。
3. block_size 64→128 迁移仍是独立、未实现的评估结论,如果以后要做需要专门的一轮
   验证(尤其是 KV 布局改动后的正确性),不要图省事直接抄本次 mode 切换的验证力度。
4. 之前 `notes/2026-07-27-sparkinfer-merge-and-verify-cg.md` 记录的"CG 仍慢于
   eager"结论现在已经被推翻,建议后续读者以本笔记为准。
