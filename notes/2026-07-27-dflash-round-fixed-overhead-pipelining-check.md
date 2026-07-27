# DFlash 单轮固定开销(accept_reject_sync/draft_kv_precompute/bookkeeping)能不能流水线掉?(2026-07-27)

## 结论先行

**不能有意义地流水线掉,这三项加起来只占 round_total 的 ~4.2%(1.85ms/44.16ms),而且
其中的 GPU→CPU 同步是投机解码算法本身要求的硬依赖,不是实现疏漏。**

## 方法:读代码确认数据依赖链,不是猜测

`benchmarks/profile_dflash_round.py`(已有的阶段级 profiling 脚本)和生产代码
`runtime/backends/laguna_dflash.py:dflash_round` 的实际执行顺序完全一致:

```
verify_replay(GPU)
  -> all_argmax = verify_logits[:16].argmax(-1).tolist()   # GPU→CPU 同步点
  -> accept_reject_sync 阶段(0.50ms)= 这次 .tolist() 本身的等待时间
  -> _verify_only_accept_reject(纯 CPU/Python)             # bookkeeping 阶段(0.50ms)
  -> draft_kv_precompute(GPU,依赖 context_count 这个 CPU 决策结果)  # 0.85ms
  -> draft_replay(GPU,依赖 new_bonus/new_kv_len 这个 CPU 决策结果) # 3.59ms,是独立大项不算在这三个里
```

**关键约束**:`draft_kv_precompute` 和 `draft_replay` 都需要知道"这一轮到底接受了几个
draft token"(`context_count`/`new_bonus`)才能算——这个信息只有 verify 的 argmax 结果
出来、并且做完 CPU 端的 accept/reject 判断之后才知道。这不是"忘了重叠",是投机解码算法
本身的因果顺序:你不可能在知道"接受了几个"之前,就去算"接受之后要更新的 KV/下一个
draft"。

## 跨轮流水线呢?

也不行,原因类似:round N+1 的 `verify_tokens = [new_bonus] + next_draft_tokens`,而
`next_draft_tokens` 正是 round N 的 `draft_replay`(round N 最后一步)的输出。round N+1
的 verify 没法在 round N 的 draft_replay 完成前开始——两轮之间同样是硬依赖,不是可以
用双 CUDA stream 重叠掉的独立工作。

## 量化:就算能做,能省多少

`accept_reject_sync`(0.50ms)+ `bookkeeping`(0.50ms)= 1.00ms 是纯 CPU 端/同步开销,
理论上限也就是把这 1ms 完全消除(不可能,因为同步点是必须的),占 round_total 44.16ms
的 2.3%。`draft_kv_precompute`(0.85ms)是真实 GPU 计算,不是"开销",没有能不能重叠的
问题,只有它自己算得快不快(0.85ms 相对 round_total 已经很小,不是优化重点)。

## 结论

这个方向已经用真实代码依赖链核实过,不是继续深挖的高价值方向。真正还有空间、且已经
量化清楚的是 `notes/2026-07-27-dflash-bandwidth-roofline-moe-gemm-attention.md` 记录的
attention page_size 迁移(~5% 吞吐,中等成本/风险)。比 kernel 效率更大的杠杆是接受率
本身(不同维度,和这次 kernel 级检查正交,需要单独调研 draft 模型/策略,这次没有展开)。
