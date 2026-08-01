# L-P1 设计草稿:同轮 fan-out 共享(未实现,供下一轮参考)

## 目标

多个同轮到达、共享前缀的请求,让共享部分只算一次、其余 slot 引用同一批物理
块——参照 DirectModelRunner 当年 P2(`notes/prefix-cache-implementation-log.md`
"P2" 节,`runtime/block_pool.py` 的 `BlockPool.reference()`)的模式,不需要
跨轮持久化(那是 L-P2 的事),风险和验证成本都比 L-P2/L-P3 低,是继续推进
的最佳下一步。

## 前提(已具备,L-P0 完成)

- `block_table[slot]`/`ring_block_table[slot]` 现在是真实、可变的物理块 id
  列表,主模型全部地址计算路径(eager 全注意力/SWA、decode CG、DFlash verify
  CG)都已经统一读这两个表——`reference()`/`free()` 这类引用计数原语要生效,
  下游消费者不需要再改。

## 需要新增的东西

1. **引用计数**:`Block` 结构目前不存在(Laguna 侧只有裸的 `list[int]` block
   id 列表,没有 `runtime/block_pool.py` 那种 `Block(block_id, ref_cnt,
   ...)` 对象)。L-P1 需要引入某种 ref_cnt 记账——可以是最小化版本(不需要
   `BlockPool` 全部机器,只要够 fan-out 用):一个 `dict[int, int]`(物理块
   id → 引用计数),在 `reference(block_ids)`/`free_shared(block_ids)` 时
   更新。
2. **common-prefix 检测**:同轮多个请求到达时(`server/engine.py` 的准入
   路径),检测哪些请求共享前缀(可以直接复用 `_log_prefix_overlap` 已经在
   算的 `same_round_overlap_tokens`,不需要重新写这部分逻辑——今天下午已经
   核实过这个埋点本来就在跑,数据是现成的)。
3. **leader/sibling 机制**:第一个(leader)请求正常 prefill 共享前缀部分;
   后续 sibling 请求不重新计算这部分,而是把自己 `block_table[sibling_slot]`
   的对应位置直接指向 leader 的物理块 id(引用计数 +1),只对 suffix(各自
   独有的部分)重新分配、计算。
4. **SWA ring buffer 的特殊性**:DirectModelRunner 没有 ring buffer 这个
   概念,这是 Laguna 独有的复杂度——SWA 层的"前缀"语义比全注意力层复杂:
   ring buffer 只保留最近 `window` token 的 KV,如果 leader 已经把早期 KV
   挤出了 ring(因为 leader 自己后续生成了更多 token),sibling 想复用的
   "前缀"部分可能已经不在 ring 里了。这个时序问题需要专门设计:可能的方案
   是"只在 leader 还没开始 decode(纯 prefill 阶段)时允许 fan-out 共享",
   避免 ring 覆盖问题——需要下一轮任务专门定这个规则并验证。
5. **`reset_slot`/释放时机**:引用计数意味着 `reset_slot(sibling_slot)` 不能
   再无条件 `index_fill_(...)` 清零共享的物理块(leader 或其它 sibling 可能
   还在用)——需要先做引用计数递减,只有归零才真正清零/回收。

## 验证要求(不能低于 L-P0 的标准)

- 至少 2 个 sibling 共享前缀,各自独立后缀,验证:leader 和每个 sibling 的
  独立解码输出各自正确(不串号,不会读到别的 sibling 写入的内容)。
- 引用计数正确性:fan-out 结束、所有 slot reset 后,没有物理块泄漏(所有
  block 的 ref_cnt 都归零)。
- 覆盖 SWA ring buffer 场景(不能只测全注意力层)。
- 真实 GPU 验证,git-stash A/B 或等价方法比对 fan-out 路径 vs 独立重新
  prefill 的路径,确认数值一致(near-tie 或 bit-identical,取决于是不是走
  完全相同的 kernel 路径)。

## 这次没有开始实现的原因

规模上足够大、涉及新的记账机制和 SWA ring 时序问题,值得作为独立一轮任务
来做,不适合在这轮已经交付了 L-P0 核心+扩展之后,再无人盯着地往前赶。留给
下一轮。
