# 接受率是正式验收判据，但服务器对它没有可用的观测路径

日期：2026-08-02 · 状态：🟢 已核实，未修复 · 零 GPU（读代码 + 本地渲染）

## 结论

接受率（96.3–100%）是 `docs/implementation-plan.md` 写死的 A6 验收判据之一。
**但运行中的服务器无法报告它。** 三个独立缺陷叠在一起：

| # | 位置 | 缺陷 |
|---|---|---|
| 1 | `server/metrics.py:246` `record_mtp_acceptance` | **生产零调用方** |
| 2 | `server/metrics.py:266` `record_slot_kv_usage` | **生产零调用方** |
| 3 | `server/engine.py:449` `stats["mtp_acceptance_histogram"]` | 5 个桶，`na >= 5` 静默丢弃；生产 `K=15` |

今天所有可信的接受率数字都来自 `benchmarks/acceptance_regression.py` 走 `bf exec`——
即**离线基准**，不是服务器自己。生产服务器跑着的时候，没人能问它"你现在接受率多少"。

## 逐条证据

### 1 & 2 · 两个记录函数生产零调用

```
$ grep -rn 'record_mtp_acceptance' --include=*.py runtime/ server/ bfdiag/ | grep -v 'def '
(空)
$ grep -rn 'record_slot_kv_usage' --include=*.py runtime/ server/ bfdiag/ | grep -v 'def '
(空)
```

两者都被 `render_d2_metrics()` 导出（`server/metrics.py:281` 的直方图、`:303` 的
`blackwellm:slot_kv_usage_fraction` gauge），由 `server/app.py:1397` 在 `/metrics` 上调用。

⚠️ **比"恒为 0"更糟的是：空的时候这两条序列根本不出现。** 本地渲染实测：

```
$ python -c "import server.metrics as m; print(m.render_d2_metrics('laguna-s-2.1'))"
blackwellm:prefix_cache_hits_total 0
blackwellm:prefix_cache_misses_total 0
```

对仪表盘来说，**缺失的序列看起来像"还没数据"，而不是"这条链路是断的"**。
`record_prefix_cache_hit` / `record_prefix_cache_miss` 是接上了的（各 2 处调用），
所以 `/metrics` 看起来"部分工作"，更不容易引起怀疑。

### 3 · 引擎自己的直方图装不下生产的 K

`server/engine.py:1483`：

```python
elif 0 <= na < len(self.stats["mtp_acceptance_histogram"]):
    self.stats["mtp_acceptance_histogram"][na] += 1
```

`stats["mtp_acceptance_histogram"] = [0] * 5`（`:449`），所以有效范围是 `na ∈ 0..4`，
**`na >= 5` 落进 `elif` 的假分支，被静默丢弃**——不是记进溢出桶，是消失。

`runtime/backends/dflash_constants.py:8` 的 `NUM_SPECULATIVE_TOKENS = 15`。
一次健康的 DFlash 轮次接受 5–15 个 token 是常态，**这些轮次在直方图里完全不存在**。
换句话说：接受率越健康，被记录的比例越低。

`server/metrics.py:232` 的 `MTP_ACCEPT_BUCKETS = (0..8)` 同样装不下 9..15，
即使把 1 接上，上面三分之一仍然落在桶外。

📌 `engine.py:460` 的既有注释已经指出过这个 quirk（"silently drops any num_accepted >= 4"），
**但阈值写错了一位**：`len == 5` ⇒ 丢弃的是 `>= 5`，不是 `>= 4`。本笔记以实测为准。

## 为什么这三条要一起修

单修任何一条都不够：

- 只接 1（喂 Prometheus）→ 数据仍被 3 的 5 桶截断，且 metrics 侧 9 桶也装不下 K=15
- 只扩 3 的桶 → `/metrics` 仍然什么都不报，因为 1 没接
- 两个直方图的桶必须**由 `NUM_SPECULATIVE_TOKENS` 推导**，不是各写一个字面量——
  今天这两处一个是 5、一个是 9、真值是 15，**三个数字互不相同**，这本身就是根因

## 与既有条目的关系

同一族：**"观测能力看起来在，其实不在"**。
- RK9 / C7-2：CUDA Graph 捕获**成功**的可观测性为 0（只有失败打 warning）
- 本条：接受率与槽位 KV 使用率的可观测性为 0

也同属本轮反复撞见的另一族：**一份正确的实现存在，但生产路径没接到它**
（`BlockPool` 44 个测试零生产调用方、`prefill_sampled` 唯一调用方 `generate()` 零生产调用方、
N8 的 `mtp_prefill_warm_continue` 调用已截肢子系统）。

## 建议处置

作为一个条目一起做（估计半天，零 GPU 可写、需 GPU 验证一次）：
1. 两个直方图的桶宽从 `NUM_SPECULATIVE_TOKENS` 推导，并加一个溢出桶（丢弃必须可见）
2. 在 `server/engine.py` 已知 `na` 的那一处接上 `record_mtp_acceptance`
3. 在槽位 KV 已知的地方接上 `record_slot_kv_usage`
4. 补一条测试：**喂一个 `na > K/2` 的值，断言它出现在 `/metrics` 输出里**——
   这条在今天的代码上必然是红的，不需要另外构造回归证明
