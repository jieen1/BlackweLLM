# Laguna 前缀缓存方案评估(2026-07-27,只做方案,不实现)

## 结论先行

**给 Laguna 加前缀缓存不是"移植 DirectModelRunner 那套代码"能解决的小活,而是要从头
重走一遍 DirectModelRunner 当年的 P0(block-table 间接层)阶段。** 这是一个多阶段、
需要专门验证方法论(INV 不变量、near-tie 数值比对)的大功能,规模和历史上
`notes/prefix-cache-implementation-log.md` 记录的 P0-P4 相当,不建议在一轮任务里
直接动手实现。

## 为什么不能直接搬 DirectModelRunner 那套

`runtime/backends/laguna.py:331-337` 的代码注释已经把这件事说得很清楚:

```python
# E1: mirrors DirectModelRunner.block_table's role as a per-slot
# "has this slot ever been touched" dirty flag for admission. Laguna
# has no block-table indirection (physical slot is a direct
# arithmetic mapping, see _physical_slot) -- this list is never
# populated, only kept empty/falsy so ServerEngine's shared admission
# check (`slot_kv_len[slot] != 0 or block_table[slot]`) works
# unmodified against either backend.
self.block_table: list[list[int]] = [[] for _ in range(num_slots)]
```

DirectModelRunner 的前缀缓存(`runtime/block_pool.py` 的 `BlockPool`)之所以能做内容
寻址共享,前提是它有一层**逻辑块→物理块的间接映射**(`block_table[slot]` 是真实的、
可变的物理块 id 列表,允许多个 slot 的 `block_table` 指向同一个物理块、`ref_cnt`
计数、按需分配/回收)。这套间接层本身就是历史上 P0 阶段的全部内容
(`notes/prefix-cache-implementation-log.md`"P0"节)。

Laguna 完全没有这层——`_physical_slot(slot)` 是纯算术映射(`slot`直接对应一段固定
`blocks_per_slot`大小的连续物理块区间),`self.block_table`永远是空列表,只是为了让
`ServerEngine`共享的准入检查代码在两个 backend 上都能跑而摆的一个假字段。**没有间接层,
就没有"两个 slot 共享同一段物理 KV"这件事在物理上能发生的基础**——不是加个 hash map
就完事,是要先把 Laguna 的整个内存寻址模型从"槽=固定连续区间"换成"槽=可变长度、可
共享的块列表"。

## 规模对比

| | DirectModelRunner(已完成) | Laguna(现状) |
|---|---|---|
| 地址模型 | BlockPool 动态分配+引用计数(P0-P1) | 固定 `slot * blocks_per_slot` 算术映射,零间接 |
| 前缀共享机制 | 内容寻址 hash chain + GDN checkpoint(P2-P3) | 无 |
| server 集成 | `mtp_prefill_with_cache` 生产入口(P4a/P4b) | `reconcile_prefix_hit` 永久返回 0(桩) |
| 验证方法论 | INV1-INV9 不变量 + near-tie 数值比对,每阶段独立 benchmark 门禁 | 无对应基础设施 |
| 历史工作量 | P0-P4 共 5 个阶段,`prefix-cache-implementation-log.md` 104K 字 | 从零开始 |

Laguna 还有额外复杂度是 DirectModelRunner 当年没有的:DFlash draft 模型自己的独立
KV(ring buffer,SWA window)、CUDA Graph 捕获对固定地址的依赖(`laguna_cuda_graph.py`
里 CG 捕获会缓存 Q/K/V/output 的裸地址,今天的 STATUS 文档记录过一次因为地址失效
导致接受率暴跌到 0.13% 的真实事故)——前缀缓存一旦引入块级别的动态搬迁,必须重新
证明不会踩到同一个坑。

## 建议的分阶段路径(仅方案,参照 DirectModelRunner 的 P0-P4 节奏)

1. **L-P0**:给 Laguna 引入真正的 block-table 间接层(`_physical_slot`固定映射→
   动态分配+引用计数),先做到行为完全不变(bit-identical),只是把寻址方式换掉——
   这是后面一切的地基,工作量和历史 P0 相当。
2. **L-P1**:同一轮内的 fan-out 共享(多个请求共享同一段刚计算出的前缀),不需要
   跨轮持久化,风险最低,价值验证最快。
3. **L-P2**:跨请求持久化内容寻址缓存(真正的"warm hit"),需要设计 Laguna 版本的
   GDN-equivalent 状态快照——**但 Laguna 没有 GDN**,DFlash draft KV/ring buffer 
   要不要一起做快照、怎么和主模型 KV 的 hit 边界对齐,是这一阶段独有的新问题,
   DirectModelRunner 的方案不能直接照抄。
4. **L-P3**:server 集成(`reconcile_prefix_hit` 从桩变成真实实现),接进
   `server/engine.py`。

## 这次没做的原因

- 规模上比这次 session 已经做的所有事情(复现、vLLM 版本决策、sparkinfer 合并)都大,
  不适合在无人盯着的后台任务里一次性冲完,尤其是涉及 CUDA Graph 地址失效这种已经
  出过真实事故的雷区。
- 价值取决于真实流量模式(sequential 多轮 vs 同轮 fan-out),而 Laguna 目前**连
  这个用量数据都没有**(`server/engine.py`目前甚至没有为 Laguna 记录 prompt 前缀
  重叠率——`DirectModelRunner`当年的 P0 之前就先加了这个埋点)。建议的第一步反而是
  **先加埋点、看真实命中潜力有多大,再决定值不值得投入 L-P0**,而不是假设"多轮
  对话场景一定值得做"就直接动手。

## 建议的下一步(如果要推进)

先做最小、最安全的一步:在 `server/engine.py` 的 Laguna 准入路径加上前缀重叠率的
只读埋点(照抄 DirectModelRunner P0 之前用过的做法),跑一段时间真实/模拟流量,
拿到数据再决定 L-P0 值不值得投入。这一步本身工作量很小、零行为风险,可以作为
独立小任务先做。
