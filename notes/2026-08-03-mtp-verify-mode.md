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
capacity=1）。端到端 256 token 的总时长取自服务器日志：

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

算术（**每轮提交 = 接受的 draft 数 + 1 个 bonus/recovery token**，别漏掉那个 +1）：

| prompt | 每轮接受 draft | 每轮提交 token | 验算 |
|---|---|---|---|
| prose | 1.54 | **2.54** | 32 tok / 13 轮 = 2.46 ✓ |
| code | 2.00 | **3.00** | 32 tok / 11 轮 = 2.91 ✓ |

打平线 = 每轮提交 token × 40.3 ms（=1/24.80）：prose **99.1 ms**、code **117.3 ms**。

服务器实测反推的每轮成本：prose 13.142 s / (256/2.46 轮) = **126 ms**；
code 13.536 s / (256/2.91 轮) = **154 ms**。都超线，故 0.79x/0.83x。
**要补的是 25~30%**，不是「必须先把接受率拉到 3.0」。

独立 profile 的 154.7 ms/轮与 code 那一列吻合。

> **口径提醒**（`historical-implementation-survey.md:484`）：**任何接受率数字必须
> 连生成深度一起引用**——历史实测短请求 76.81% → 4×2000 输出 94.46%。上表是
> 13~19 token prompt / 32 token 输出，即**最浅**的情形；历史那个「3.0 tokens/step」
> 是 **128K / c=4 深生成**下测的（`:458`），而且历史 K=3、我们 K=4（`:120`）。
> 直接把 1.54 和 3.0 并排比，正是那份调研专门警告过的错误。深生成下我们的
> 接受率会更高，但那要单独测，不能假定。

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
删掉 (a) 的 anchor 推进。两次全量前向变一次，砍掉的远多于需要补的 25~30%。

**在此之前 MTP 保持默认关闭**，理由就是上表的 0.79x/0.83x。

## 四·五、真正的差距是**发射空隙**，不是内核量（实测）

两个新测量把上面的推测替换掉了。

### (1) 深度几乎不抬接受率 —— 我的猜测是错的

同一脚本 `QSR_B3_N_TOKENS` 从 32 提到 256（standalone、eager、K=4、FP8 KV 开）：

| prompt | 32 token 深度 | 256 token 深度 | 每轮提交 | 加速比 |
|---|---|---|---|---|
| prose | 1.54 接受/轮 | **1.60** | 2.61 | **1.11x** |
| code | 2.00 接受/轮 | **2.04** | 3.05 | **1.35x** |

历史那条「短请求 76.81% → 4×2000 输出 94.46%」的深度效应，在这个模型的
32→256 区间**几乎不存在**。所以不能指望靠加深生成把 MTP 救回来——已实测，
不是推断。（更深的 2000+ token 未测；但要靠它就得先证明它。）

### (2) standalone 净赢、服务器净亏 —— 差别是**工作点**，不是模型

| | 非投机 | 投机 | 比值 |
|---|---|---|---|
| standalone（eager，无 CG） | 4.13 / 3.85 tok/s | 4.58 / 5.18 | **1.11x / 1.35x** |
| 服务器（CG 开） | 24.80 / 22.74 tok/s | 19.48 / 18.91 | **0.79x / 0.83x** |

同 K、同 checkpoint、同 FP8 KV。区别只有 CUDA 图。

**CUDA 图让 plain decode 快了 6 倍（242 → 40.3 ms/token），却只让 MTP 轮快了
4.4 倍（570 → ~126 ms）。** MTP 是在 eager 那个「基线本来就很慢」的工作点上赢的，
一旦基线被图化，它的优势就被吃掉了。

### (3) 把内核量和空隙分开，结论很干净

- plain decode（CG）：**89% busy**（`notes/2026-08-03-decode-kernel-profile.md`），
  40.3 ms/token → **35.9 ms 内核/token**
- MTP 轮：**68% busy**，154.7 ms 墙钟 / **105.6 ms 内核**

105.6 ÷ 35.9 = **2.94 个 decode step 的内核量**，产出 **2.61~3.05 个 token**。

**MTP 的内核工作量本身已经打平到略胜；输掉的全部是发射空隙**（32% vs 11%）。
每轮 ~49 ms 的空隙，就是 anchor / draft / verify 三个图之间那段没被图化的
Python 胶水。

这直接决定了优化顺序：先砍前向数和图切换次数（M-1b），而不是先调内核。

## 五、期望收益（M-1b 之后）

每轮 2 次全量前向 → 1 次，同时少一次图切换及其胶水——**内核量和空隙一起降**。
当前每轮 126 ms(prose)/154 ms(code)，打平线 99.1/117.3 ms，需要补 25~30%。

按 §四·五(3) 拆开算：内核 105.6 → 约 70 ms（少一个全量前向），空隙 49 → 约
33 ms（少一段图间胶水），合计约 103 ms —— 落在 prose 打平线附近、明显低于
code 的线。所以 M-1b 之后预期 prose 打平上下、code 净胜。

接受率的提升（M-2 跨槽批处理）是**叠加**项。深生成的自然增益已实测**不存在**
（§四·五(1)），不要再计入。
