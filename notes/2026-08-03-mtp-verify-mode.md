# MTP 的 verify 走错了 sparkinfer 模式（根因 + 修复 + 剩余差距）

2026-08-03。GPU：RTX PRO 6000 Blackwell Max-Q (sm120)。checkpoint：
`unsloth/Qwen3.6-27B-NVFP4`（**standard**，不是 `nvidia/`）。

## 一、根因

`Qwen36Attention.forward` 只按形状推断 paged 模式：

```python
mode = "decode" if seq_len == 1 else "extend"     # runtime/model/qwen36_model.py
```

MTP 的 verify 是「K 个 token 打在已有 KV span 上」，形状上和「K 个 token 的
prefill」**无法区分**，于是被当成 extend 规划并发射。但在 sparkinfer 里
extend 和 verify 是两个不同模式，planner 里 5 处分支
（`planner.py:469/556/571/715/727`），计划不同、内核策略标志也不同。

`Qwen36AttentionWorkspace` 的类文档当时明确写着「no `"verify"`（B1 has no
speculative decoding）」——B1 时代这是对的，MTP 接进来之后就不对了，而这行
文档正好掩盖了它。

### 症状与证据链

K=4，standard checkpoint，`scripts/b3_mtp_e2e_acceptance_throughput.py`：

| 配置 | token 序列与非投机路径一致 | 每轮接受数 (prose/code) |
|---|---|---|
| FP8 KV 开，**修复前** | **False** | 1.54 / 1.82 |
| FP8 KV 关，修复前 | True | 1.54 / 1.82 |
| K=1，FP8 KV 开，修复前 | True | 0.778 / 0.882 |
| FP8 KV 开，**修复后** | **True** | 1.54 / **2.00** |

所以「FP8 KV 和 MTP 不兼容」这个说法是错的：FP8 KV 从来没问题，是 verify
跑错了模式，而错误只在计划被 FP8 形状化之后才发散。K=1 当时能过，是因为
1 个 token 的 verify 塌回 decode 路径，而 decode 路径本来就是对的——正是这个
**extend/decode 的形状不对称**把问题指到了这里。

## 二、容量部分不是新问题，是**本仓库自己解过一遍**

`runtime/backends/laguna_sparkinfer_attn.py::SparkinferPrefillWorkspace` 早在
DFlash 上踩过完全一样的坑（`notes/2026-08-01-c1-c2-gpu-investigation.md` §C-1）。
两条结论直接移植，不重新发现：

1. **`eager_extend_work_items_capacity` 不适用于 verify**，而固定容量工作区
   是设计上「宁可硬失败也不自动扩容」的，所以会直接抛
   `fixed-capacity paged workspace exceeded`。verify 的容量改为**跑一次真实
   eager planner**（`create_paged_plan(enable_cuda_graph=False, mode="verify")`，
   就是线上每次调用会用的那个函数）打在声明的最坏情况上，读它的数字。
   - 那份调研记下的**死路也一并抄进 docstring** 以免重试：
     `plan_verify_graph_capacity` 在真实 GPU 上预测 47/112，而真实 eager 计划
     需要 96/256——图路径和 eager 路径算的是两套调度，不能互为容量来源。
2. **verify 必须排除在 `PagedPlanBudget` 之外**。`_paged_determine_cta_tile_q`
   靠 `packed_qo_len` 的精确匹配选 M64 verifier，下游若干内核策略标志
   （`use_laguna_verify_kernel`、`laguna_verify_two_wave_b1`、**FP8 PV MMA 路径**）
   都 gate 在 `plan.cta_tile_q == 64` 上。容量推导出来的 `packed_qo_len` 会错过
   这个匹配，把 verify 悄悄降到 `cta_tile_q=16`。verify 本来也没有 extend 的
   多桶问题——它的 query 长度是固定的 K 窗口，本身就是单桶。

容量按声明的 K 显式给（`declare_verify_capacity`），不猜；K 变大时丢弃旧
工作区重建。

## 三、修复带来的实测变化

`profile_mtp_round.py`，K=4，FP8 KV 开，CUDA 图开：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| `cg_status` | `{anchor: captured, draft: captured}` | `{anchor, draft, **verify**: captured}` |
| 每轮墙钟 | 303.8 ms | **154.7 ms**（−49%） |
| 每轮叶子内核 | 102.9 ms | 105.6 ms |
| GPU busy | 34% | **68%** |

verify 的 CUDA 图**现在才捕得住**——这是修复的直接结果，verify 模式的计划是
单桶稳定的，图才有得捕。

## 四、但 MTP 仍然是净亏，差距在别处

服务器端 A/B（`QSR_SERVER_ENABLE_MTP` 0/1，其余相同，CG 开，FP8 KV 开，
capacity=1）：

端到端 256 token 的总时长取自服务器日志：

| prompt | MTP off | MTP on | 比值 |
|---|---|---|---|
| prose | 256 tok / 10.323 s = **24.80 tok/s** | 256 tok / 13.142 s = **19.48 tok/s** | **0.79x** |
| code | 256 tok / 11.257 s = **22.74 tok/s** | 256 tok / 13.536 s = **18.91 tok/s** | **0.83x** |

> **测量方法的坑（先踩后改）**：第一版驱动脚本数的是 SSE delta 条数，不是
> token 数。MTP 一轮提交多个 token，服务器把它们放在**同一个 delta** 里，于是
> MTP-on 被读成「只吐了 92/97 个 token」，算出 0.28x/0.32x，还顺带诬陷了一个
> 并不存在的「服务器路径提前终止」bug。服务器日志里两边都明明白白是
> `256 tokens, finish=stop`。流式场景下 **delta 条数 ≠ token 数**，必须从
> usage 或服务端日志取 token 数。

算术：MTP-on 实测 19.48 tok/s → 51.3 ms/token；一轮产出 1.54~2.0 个 token
→ 每轮约 **103 ms**。要打平需要每轮 < 接受数 × 40.3 ms（=1/24.80），即接受率
2.0 时 < **80.6 ms**。现在约 103 ms，**差 ~25%**。

独立 profile 里一轮 154.7 ms、叶子内核 105.6 ms ≈ 2.9 个 decode step 的算力
换 1.54~2.0 个 token（该 profile 的上下文长度与服务器不同，故比 103 ms 大）。

### 结构性原因（有历史对照）

`notes/2026-08-03-historical-implementation-survey.md:114` 已经记下过：

> verify = 1 次跨槽 `verify_batch_spec`，**qo_len = k+1**
> （`oracle/.../direct_model_runner.py:1581`）
> vs 我们：每槽 **2 次** forward（anchor 推进 `[1,1]` + verify `[1,K]`）
> —— **不同**（多一次 anchor forward；当年 anchor 的 KV 写入折在 verify 里）

decode 是显存带宽瓶颈，5 个 token 的前向和 1 个 token 的前向成本几乎一样，
所以当年把 anchor 折进 verify 是**白拿**的：每轮 2 次全量前向 → 1 次。

我们其实已经在拼 `all_hiddens` `[1, K+1, H]` 了（`qwen36_mtp.py` 的 round
文档写得很清楚），只是用**两次**前向拼出来的；GDN 的 `_ssm_spec_row` 也本来
就按 K+1 行设计。

**下一步（M-1b）**：verify 改成 `[anchor] + drafts` 的 K+1 token 单次前向，
删掉 (a) 的 anchor 推进。预期每轮内核 105 → ~70 ms。届时按接受率 1.54~2.0
是 35~45 ms/token（与 plain 36.5 打平到小胜），接受率若能回到历史 ~3.0 则
约 23 ms/token（1.6x）。

**在此之前 MTP 保持默认关闭**，理由就是上表的 0.28x/0.32x。

## 五、期望收益（M-1b 之后）

每轮 2 次全量前向 → 1 次，按前向占轮次成本的主体估算，每轮 ~103 ms 应降到
**60~70 ms**，低于 80.6 ms 的打平线。届时按当前接受率 1.54~2.0 就该是**净胜**，
而不是靠把接受率先拉到历史的 3.0。接受率的提升（M-2 跨槽批处理等）是**叠加**
项，不是前提。
