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
> 直接把 1.54 和 3.0 并排比，正是那份调研专门警告过的错误。
> **这条口径提醒仍然成立，但由它推出的「深生成下我们接受率会更高」已被
> §四·五(1) 实测否掉**——32→256 token 只从 1.54/2.00 动到 1.60/2.04。

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

## 六、空隙的确切来源：每轮两次主机同步（已定位到行）

§四·五(3) 说「输在发射空隙」，但没说空隙在哪。用包装 `Tensor.tolist/item/cpu/to`
并在阻塞超过 2 ms 时记录 Python 栈的办法测出来（`find_mtp_syncs.py`）：

```
round wall 137.72 ms; blocking transfers > 2.0 ms

  66.82 ms/round  x1.0  tolist  <-  qwen36_mtp.py:618  round()
                                    predicted_tokens = all_logits.argmax(dim=-1).tolist()
  13.72 ms/round  x1.0  tolist  <-  qwen36_mtp_cudagraph.py:638  replay()
                                    return self._draft_tokens.tolist()

  80.54 ms/round  TOTAL attributed (58% of the round)
```

**这两次等待本身不是浪费**——GPU 在这 80 ms 里确实在算 anchor+verify+draft。
浪费的是**等待前后的空档**：CPU 被同步卡住时无法给 GPU 排下一批活，于是同步
一返回、Python 在跑 accept/reject 的那段时间 GPU 完全空转。整轮 137.7 ms 墙钟
对 105.6 ms 内核 = **32 ms 纯空转**（23%）。

（记一下方法：chrome trace 只说 `aten::to` 花了 94.9/42.5/14.0 ms，不说是谁调的。
一个几十毫秒的 D2H 拷贝不是在搬字节，是在**等**。要定位就得给调用点打栈，
而不是从 op 名字猜。之前我差点去改 `torch.tensor(list, device=cuda)` 那类
每次几微秒的小拷贝——那个方向是错的，先量再改省下了这次返工。）

### 依赖链其实不需要经过主机

draft 的种子 = `next_anchor` ← accept/reject ← verify 的 logits。今天这条链
**穿过主机**：logits 读回 CPU → Python 算接受数 → 把种子 token 作为 Python int
写回设备。

但 accept/reject 是纯 argmax + 前缀比较，可以整个留在设备上，`next_anchor` 用
设备端 gather 取（`predicted_tokens[m]`，`m` 是设备标量）。这样一轮到最后
才需要一次读回（提交的 token，流式输出必需），CPU 可以在 verify 还没算完时
就把 draft 图排进去。

仓库里已有一半：`runtime/mtp_accept.py::determine_accept_reject_batch`
（`[num_reqs, k+1]` 一次 argmax + 一次 `.tolist()`）**实现了但零调用**——
和 `block_pool.py`、`recurrent_state_pool.py::spec_row` 一样，是当年做过、
这次没接线的东西（`historical-implementation-survey.md:115`）。

**M-1c**：accept/reject 全程留在设备上，draft 图接受设备端种子 token。
预期回收那 32 ms 空转的大部分。与 M-1b（anchor 折进 verify）正交，两者叠加。

## 七、M-1b 落地：anchor 折进 verify（qo_len = k+1），MTP 转为净胜

按 §四·五(3)/§五 的预测做了，结果比预测好。

### 轮次成本（`profile_mtp_round.py`，K=4，FP8 KV 开，CG 开）

| | verify 走 extend | verify 模式 | **+ 合并前向** |
|---|---|---|---|
| 每轮墙钟 | 303.8 ms | 154.7 ms | **105.9 ms** |
| 叶子内核 | 102.9 ms | 105.6 ms | **76.7 ms** |
| GPU busy | 34% | 68% | 72% |
| `cg_status` | anchor+draft | +verify | anchor 变 `unused` |

内核 −27%、墙钟 −32%。

### 服务器端 A/B（256 token，token 数与时长取自服务器日志）

| 顺序 | prose | code |
|---|---|---|
| MTP-off 先 | 15.93 → 25.64 tok/s = **1.61x** | 15.85 → 31.98 = **2.02x** |
| MTP-on 先（顺序颠倒） | 16.88 → 28.84 tok/s = **1.71x** | 16.59 → 31.60 = **1.91x** |

**从 0.79x/0.83x 变成 1.61~1.71x / 1.91~2.02x。**

> **测量纪律（这次差点被坑第二次）**：这块卡是 Max-Q，时钟随热态漂。**同一份
> 代码**的 MTP-off 基线在三次运行里分别是 36.8 / 39.6 / 57.8 ms 的中位 ITL——
> 差 57%。所以**跨运行的绝对 tok/s 不可比，只有同一次运行内的比值可比**，而且
> 比值本身要用**颠倒顺序**再测一遍才算数（先跑的那个更凉）。上表两个方向都测了。
>
> 基线从早先的 24.8 tok/s 掉到 16.6~16.9，一度像是 decode 回归。排掉它靠的是
> `git diff --stat main..HEAD`：本分支只动了 `qwen36_mtp.py`、
> `qwen36_mtp_cudagraph.py`、测试和笔记，**decode 路径一行没改**。查代码比再测
> 一轮快，也更能定论。

## 八、仍未解决：服务器路径的贪心输出与非投机不一致

`b3_mtp_e2e_acceptance_throughput.py` 报 `token sequences match: True`，但它
走的是**模型级**路径，不经过 `round()` 和 CUDA 图。把服务器 MTP-on / MTP-off
在温度 0 下的完整输出逐字符比对，结果不同：

| 树 | prose | code |
|---|---|---|
| 只修 verify 模式（main @ 94d569f） | 第 **45** 字符起分叉 | 第 **232** 字符起分叉 |
| **+ 合并前向** | 第 **525** 字符起分叉 | **完全一致** |

合并前向把一致性**大幅改善**（code 从分叉到逐字节一致，prose 推迟 12 倍），
说明分叉**不是它引入的**——在只修 verify 模式的树上就已经存在，而且更严重。

尚未判定它属于哪一类：

1. 真 bug（round/图路径的簿记问题），还是
2. `docs/b1-correctness-criterion.md` 早就确立的**数值不可位精确**那一类——
   plain decode 的 CG 路径本身就冻结在 1 个 KV split 而 eager 用 4~16，两者都
   "对"但不逐位相同；verify 在 qo_len=5 下用的内核与 decode 的 qo_len=1 不同，
   归约顺序不同，近似平局会翻转。

### 已判定：是 (2)，而且对照极其干净

做了 MTP-off 自身的 CG-vs-eager 对照（`greedy_divergence_control.py`：两臂都
关 MTP，唯一变量是 `QSR_SERVER_ENABLE_CUDAGRAPH`）：

| 对比 | prose | code |
|---|---|---|
| MTP-on vs MTP-off（都开图） | 第 **525** 字符分叉 | 一致 |
| **CUDA 图 vs eager（都关 MTP）** | 第 **525** 字符分叉 | 一致 |

**同一个字符位置、同一个模式**。而且两组的文本是**逐字对应**的：

```
CG-vs-eager 对照:  cg_on  = " keeper's past self? From a ship that never "
                   cg_off = "mselves? From a lost love? A warning about t"
MTP A/B:           off    = " keeper's past self? From a ship that ne"
                   on     = "mselves? From a lost love? A warning abo"
```

即 **MTP-on 的输出等于 eager 的输出，MTP-off（开图）的输出等于 cg_on 的输出**。
分叉是 `docs/b1-correctness-criterion.md` 早就确立的那一类路径相关数值差异——
非投机路径**自己**在同一个字符上就已经有了，MTP 一点没往里加。

（这也解释了为什么合并前向"改善"了一致性：合并前 MTP 多跑一次 `[1,1]` anchor
前向，那一次走的是 decode 内核；合并后整轮只剩 verify 内核，路径更单一，与
eager 的归约顺序反而更接近。第 45/232 → 525/一致 是这个原因，不是簿记变对了。）

**结论：MTP 的正确性落在本仓库既有的、已接受的数值包络内，不是 bug。**
