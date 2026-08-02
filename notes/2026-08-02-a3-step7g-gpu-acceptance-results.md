# Track A 第 7 步（7-g）GPU 验收结果

日期：2026-08-02 · 执行：协调者直接执行（非子 agent）· 状态：3/4 条通过，第 4 条不成立需重做

## 判定摘要

| 门禁 | 结果 | 说明 |
|---|---|---|
| **1 · 贪心 bit-exact** | ✅ **通过** | 5 个 prompt × 64 token，切换前后**逐字节相同** |
| **2 · 接受率不回归** | ⚠️ **本次执行不构成有效门禁**（见 §3） | 跑出了真实数据，但对照组错了 |
| **3 · 前缀缓存不回归** | ✅ **通过** | `tokens_saved_ratio = 1.0000`，8 组 × 第 2–6 轮全部 |
| **4 · C-LIVE 冒烟** | ✅ **通过** | 67/67 |

## 1 · 贪心 bit-exact ✅

**方法**：main（已接协调者）起服务抓输出 → 停 → 新建 revert worktree
（`git revert --no-commit ea9d784`，确认 `self.slot_resources.` 计数归 0）→ **用完全相同的
环境变量**起服务 → 同样 5 个 prompt → SHA256 比对。

| prompt | 长度 | 结果 |
|---|---:|---|
| CUDA graph 解释 | 314 字符 | ✅ 逐字节相同 |
| `def fibonacci(n):` | 170 | ✅ |
| capital of France | 293 | ✅ |
| 三个大于 100 的质数 | 258 | ✅ |
| 英译中 | 257 | ✅ |

⚠️ **过程中修正了一次会让结果失效的错误**：第一次起 revert 服务用了默认配置，
`/health` 报 `capacity=1`，而切换后那台是 `capacity=3`。**配置不同就不可比**——而且本仓库
刚证明过 **workspace 容量会改变数值结果**（CG-vs-eager 分歧的根因正是按容量规划）。
已按 `capacity=3 / num_slots=3 / blocks_per_slot=4096 / cudagraph=1 / dflash=1 /
prefix_cache=1 / gpu_mem_util=0.95` 重起后重测。

⚠️ **另一次**：就绪检查写成 `until curl ... >/dev/null`，只看退出码。8100 前面有代理，
服务未起时返回 **502 + 空 body**，curl 退出码仍是 0 → 循环提前退出，对着没起来的服务发请求。
**改为按响应内容判断**（`grep -q '"status"'`）。同样的坑在 bfdiag daemon 上又踩了一次：
socket 文件存在 ≠ 能连接，改成真的去 `connect()`。

## 2 · 接受率 —— 数据是真的，但这不是 7-g 的门禁 ⚠️

跑了 `bf exec benchmarks/acceptance_regression.py`（13 个工作负载），与
`benchmarks/fixtures/acceptance_regression_20260731.json` 对比：

| workload | 07-31 | 08-02 | Δ |
|---|---:|---:|---:|
| code-4K | 0.586 | **0.978** | **+0.392** |
| cn-repeat-4K | 0.086 | 0.279 | +0.193 |
| qa-quicksort | 0.853 | 0.985 | +0.132 |
| qa-tcp-udp | 0.881 | 1.000 | +0.119 |
| qa-photosynthesis | 0.911 | 1.000 | +0.089 |
| ids-cycle-512 | 0.404 | 0.471 | +0.067 |
| galaxy-4K / qa-relativity | 0.978 | 1.000 | +0.022 |
| fox-4K | 0.963 | 0.970 | +0.007 |
| **fox-64K** | 1.000 | 0.961 | **−0.039** |
| **ml-4K** | 0.963 | 0.833 | **−0.130** |
| **ids-cycle-4K** | 0.351 | 0.183 | **−0.169** |
| **cn-qa** | 0.567 | 0.387 | **−0.180** |

**为什么这不构成 7-g 的门禁**：07-31 基线与今天之间**合并了 160 个提交**——Track A 第 5/6/7 步、
E2 采样与投机共存、E-N1-b0、前缀缓存工作全在其中。这个对比测的是"两天之间整个代码库变了什么"，
**不是"接入协调者是否改变了接受率"**。

**7-g 本身的接受率不变，由门禁 1 推得**：贪心输出逐字节相同意味着每一步的 accept/reject
决策相同（同模型、同草稿、同 logits），且 `SlotResourceManager` 在
`needs_two_cache_families=False` 时是纯转发。要把这条做成独立门禁，需要像门禁 1 那样
**在同一次会话里前后各跑一遍**。

⚠️ **上面 4 条下降是独立于 7-g 的真实待查项**，尤其 `ml-4K 0.963→0.833` 与 `cn-qa 0.567→0.387`。
`fox-64K` 的数字**本来就不可信**（已记录：同一负载因调用在序列中的位置不同吞吐可摆动约 60%，
该条已被撤下 A6 验收判据）。

## 3 · 前缀缓存不回归 ✅

`python -m benchmarks.prefix_cache_baseline`，8 组独立 6 轮对话：

```
turn 2..6: hit_rate=1.00  mean_tokens_saved_ratio=1.0000  (n=8 each)
```

每一次命中都 `hit_L == ideal_L`，与切换前基线（39/40 命中为 1.0）一致。

📌 **顺带证实了判据选择是对的**：整轮 `/debug/stats` 的
`prefix_cache_hit_rate = 0.506`（第 1 轮必然 miss 被计入分母），而缓存实际捕获了
**100% 的可用前缀**。**用 `hit_rate` 当门禁完全没有信号。**

## 4 · C-LIVE ✅

`scripts/c_live_smoke.py --base-url http://127.0.0.1:8100` → **67 passed, 0 failed**。

## 结论与待办

- **第 7 步的行为不变性已被证明**（门禁 1 + 3 + 4）。`implementation-plan.md` 标注的
  "爆炸半径最大"在实测上没有兑现成任何行为变化——符合设计预期（对今天每个生产 checkpoint，
  `needs_two_cache_families` 都是 `False`，协调者纯转发）。
- [ ] **重做门禁 2**：同一会话内 revert 前后各跑一次 `acceptance_regression.py`
- [ ] **独立排查 4 条接受率下降**：`ml-4K`、`cn-qa`、`ids-cycle-4K`、`fox-64K`
      —— 与 7-g 无关，但发生在最近 160 个提交之内
