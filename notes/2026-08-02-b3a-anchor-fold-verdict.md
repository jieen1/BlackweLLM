# B3-a：anchor 推进能不能折进 verify 块——结论是不能，且是实测出来的

日期：2026-08-02 · 分支 `work/b3a-anchor-20260802`（**未合并 main**）·
测量环境：RTX PRO 6000 Blackwell Max-Q，`nvidia/Qwen3.6-27B-NVFP4` 真权重

> 【实测】/【推算】标注同 `notes/2026-08-02-spec-verify-batching-bar.md` 的约定。

---

## 0. 结论先行

`docs/implementation-plan.md` §7.1 B3 把"把 anchor 推进折进 verify 块"列为
"目前唯一的大杠杆"，依据是上一轮的一句【推算】判断（"这是脚本结构的产物,不是算法必需"）。
**本轮把这句话拆开重新验证，结论是反的：这个依赖是真的，不是脚本结构的产物。**

两条独立证据：

1. **因果论证**：MTP 头给一轮起草的第一步喂 `(next_token_ids=T_p, prev_hidden=h_p)`，
   `h_p` 必须是目标模型**处理过 `T_p` 之后**的隐状态（"同一位置"配对，§1 有完整推导）。
   每一轮新确定的 anchor（拒绝时的纠正 token，或全部接受时的 bonus token）**在被确定的那一刻
   从未被当作真实输入喂给过 backbone**——它的值来自 `verify_forward` 自己的 logits（零额外算力），
   但它的 `h_p` 只能靠**真的把它塞进 64 层再跑一遍**才能拿到。这不是脚本怎么写的问题，
   是"目标模型的隐状态是这个 token 自己的前向输出"这条数学关系。
2. **实测排除了唯一的"零风险"替代方案**：把 confirm 步骤从"走 `Qwen36TextModelSelfBuilt.
   forward()` 普通前向"换成"走 `verify_forward(K=1)` + `commit_verify`"（即真正意义上的
   "让它走 verify 代码路径"），**没有变快，反而慢了 9%**（128.28ms → 139.78ms，K=1，同一进程
   连续测量）。说明"每轮两次 64 层遍历"的固定开销不是某个函数选错了，是两次遍历本身躲不掉。

**唯一在数学上真正"免费"的分支**（起草的全部 K 个 token 都被接受，且 MTP 自己额外多起草的
第 K+1 个猜测恰好等于目标模型自己的 bonus 预测）在 K=8、接受率 15%/35% 下发生概率
≈1.5^-8≈2.6e-7（prose）/ 0.35^8≈2.2e-4（code）——**实践中不会发生，不构成一个可交付的优化**。

**没有对 `runtime/` 做任何算法改动。** 唯一改动是给自己 worktree 的脚本副本修正了硬编码路径
（每个 B3 worktree的脚本副本本来就各自指向自己，这是仓库既有约定，不是本次改动引入的）。

---

## 1. 那次额外前向的依赖到底是什么

`scripts/b3_mtp_e2e_acceptance_throughput.py` 每轮末尾（L178-184，prefill 处的一次性版本在
L129-143）：

```python
new_anchor = decision["committed"][-1]
new_anchor_tensor = torch.tensor([[new_anchor]], ...)
new_anchor_logits, new_anchor_hidden = _logits_for(model, new_anchor_tensor, state)
```

`_logits_for` 调用普通 `model(token_ids, state)`——对 `new_anchor` 单独跑一次完整 64 层前向，
拿到 `new_anchor_hidden`（喂给下一轮 MTP 链的 `prev_hidden` 种子）和 `new_anchor_argmax`
（下一轮 accept/reject 的 `predicted_tokens[0]`）。

### 1.1 `prev_hidden` 的真实契约：同位置配对，不是"前一个位置"

`Qwen36MTPHead.forward`/`mtp_step` 的调用点（`runtime/model/qwen36_model.py:2332-2365`）：

```python
def mtp_step(self, next_token_ids, prev_hidden, position, mtp_cache):
    embeds = self.model.embed_tokens(next_token_ids)
    hidden = self.mtp(embeds, prev_hidden, positions, cos_sin_cache, mtp_cache)
    ...
```

链式起草规则（`mtp_step` 自己的 docstring）：**非首步**用"上一次 `mtp_step` 自己返回的
`(draft_token, hidden)` 这一对"——这一对**永远是同一次调用的输出**，即同一个"隐状态"和
"它所对应的 token"从来不会来自不同位置。**首步**（每轮唯一需要 target 提供种子的地方）
沿用同一条规则：`anchor_hidden` 必须是 backbone **处理完 `anchor_token` 之后**的输出
（`_logits_for(model, [[anchor_token]], state)` 返回的正是这个），与 `anchor_token` 严格
同位置——这不是脚本的随意选择，是链式规则内部自洽性唯一允许的写法（详见下）。

**为什么不能用"前一个已知位置的隐状态"顶替**：round N 的 `verify_forward` 对 K 个草稿位置
逐一给出 `verify_hidden[i]`（backbone 处理完 `draft_tokens[i]` 之后的隐状态，同位置，真实、
免费）。**但 round N 新确定的 anchor（`decision["committed"][-1]`）由 `runtime/mtp_accept.py`
的 `determine_accept_reject_from_predictions` 定义为"恰好一个 recovery/bonus token，从不计入
已接受的草稿"（该函数 docstring 原文）——它按构造上永远不等于任何 `draft_tokens[i]`**：

- 拒绝分支（`m<K`）：新 anchor = `predicted_tokens[m]`（目标模型在拒绝点自己的贪心预测），
  这个 token **从未作为输入喂进过任何一次前向**——它纯粹是一个 logits argmax 的产物。
- 全接受分支（`m==K`）：新 anchor = `predicted_tokens[K]`（bonus token，目标模型预测
  "`draft_tokens[K-1]` 之后是什么"）——**同样从未被当作输入喂过**，`verify_hidden[K-1]`
  只是它前一个位置（`draft_tokens[K-1]`）的隐状态，位置错了一格，不能顶替。

两个分支都一样：新 anchor 的**值**从 verify 的 logits 里免费拿到，但它的**隐状态**必须靠
把它自己塞进 backbone 再跑一遍才能拿到。**接受率 15%/35% 意味着"拒绝分支"是绝大多数轮次的
结局**（K=8 下平均每轮 1.21/2.80 个 token 被接受，远小于 K），所以这不是一个边缘情况。

### 1.2 唯一可能"零风险"的规避路径已被实测排除

理论上还剩一条低风险路径：**不消灭这次前向，只是换一条更便宜的代码路径去跑它**——
把 confirm 步骤从"普通 `forward()`"换成"`verify_forward(K=1)` + `commit_verify`"
（字面意义上的"折进 verify 块"：让它复用 verify 的机制，而不是消灭这次遍历）。

`scripts/b3a_probe_confirm_cost.py`（本轮新增，同一次模型加载、同一进程内交替计时，上下文
长度 ~28-133 token）：

| 步骤 | mean | median | min | max |
|---|---:|---:|---:|---:|
| (a) 普通 `forward()`，K=1（今天的 confirm 步骤） | 128.28ms | 137.71ms | 104.68ms | 151.09ms |
| (b) `verify_forward(K=1)` + `commit_verify`（候选"折法"） | **139.78ms** | 145.85ms | 111.61ms | 162.19ms |
| (c) `verify_forward(K=8)`（同一轮真正的 verify，作对照） | 218.46ms | 212.68ms | 179.25ms | 302.84ms |

**(b) 比 (a) 慢 9%（+11.5ms），不是快。** 机制解释：`spec_forward` 对 GDN 层额外做
K+1 份状态快照的簿记（`gdn_snapshots` 字典构建、张量克隆），这份簿记在 K=8 时被摊薄，
在 K=1 时反而是纯开销，没有可批的东西可批。

同一份数据也回答了"是不是纯粹按层数摊开销、K 不重要"：(a)/(c) = 0.587——**K=1 并不像
K=8 那样贵，两者之间存在真实的、与 K 相关的边际成本（约 13ms/token），不是纯粹的
"64 层固定开销、token 数无所谓"**。但这条边际成本恰恰说明：**每轮两次独立的 64 层遍历
（confirm 一次 + verify 一次）会把"每次遍历的固定部分"（≈100-130ms）付两遍**——这正是
上一轮判断"值得折"的直觉来源，只是"折"在事实上做不到（§1.1 的因果论证），
**换代码路径也拿不到**（本节实测）。

---

## 2. 端到端重测：真实数字（未做任何折叠，因为折不动）

`scripts/b3_mtp_e2e_acceptance_throughput.py`（本 worktree 内路径已改自指，算法零改动），
同一次模型加载：

| prompt | 接受率 | 平均接受/轮 (K=8) | 投机 tok/s | 非投机 tok/s | 加速比 | tokens_match |
|---|---:|---:|---:|---:|---:|:--:|
| prose | 15.2%（17/112） | 1.21 | 4.55 | 6.33 | **0.72×** | ✅ True |
| code | 35.0%（28/80） | 2.80 | 6.02 | 6.61 | **0.91×** | ❌ **False（首个分歧在 index 2）** |

⚠️ **两条腿都还是 <1.0×。** 与旧 fixture（prose 0.66×/code 0.61×，`benchmarks/fixtures/
qwen36_mtp_e2e_20260802.json`）比，绝对数字明显不同（prose 0.72×、code 0.91×）——
这完全符合"绝对毫秒随 GPU 热状态漂移，不能跨会话直接对比绝对值"这条本项目反复验证过的规律
（本次两个进程之间也能看到：confirm 步骤 GPU 探针测到 ~128-140ms，早前【推算】给的是
194-240ms，同一量级但不是同一个数）。**没有做任何代码改动，纯粹是重新测量。**

⚠️ **`code` prompt 这次 `tokens_match=False`。** 这不是本轮改动引入的——本轮对 `runtime/`
零改动，这是 main 分支现状本来就有的行为。脚本自己的 docstring（L27-45）已经预先点名了这类
"良性分歧"的机制：verify 阶段 full-attention 层用一次 extend-mode 核处理 K 个位置，
顺序解码用 K 次 decode-mode 核，两者是不同代码路径、不同 KV 归约顺序，`notes/2026-08-02-
eager-verify-cg-verify-divergence.md` 已经记录过同类"cos≥0.999997 但不 bit-exact"的现象。
**本轮没有去复现是否可重复、没有跑 gap-error 定量（`scripts/b3_verify_batching_logit_
agreement.py` 测的是另一项改动——上一轮的大投影批处理——不针对本次的零代码改动，重跑
不会给出新信息）**——见 §4"没能验证的东西"第 1 条，留给后续或 B3-b 一起处理。

---

## 3. 为什么"折进 verify 块"这句话本身没错，但落地条件不成立

`notes/2026-08-02-spec-verify-batching-bar.md` §6（【推算】小节）原文："这是
`scripts/b3_mtp_e2e_acceptance_throughput.py` 的实现方式，不是必然的，把 anchor 折进 verify
块可以省掉"——**这句话本身是对脚本结构的正确观察**（这次前向确实是"脚本怎么组织循环"的产物，
不是 GDN 或 MTP 头本身要求的），但**它把"可以合并到同一次 Python 调用"和"可以省掉重复的
64 层计算"混为一谈了**。

真正的因果链：

```
round N 的 verify_forward(K tokens)
        │  (确定 accept/reject，new_anchor 的【值】立即可知，零额外算力)
        ▼
confirm new_anchor：唯一途径是把它塞进 backbone 再跑一次
        │  (这一步产出 new_anchor 的 h_p —— MTP 起草下一轮 K 个 token 的必要输入)
        ▼
round N+1 的 MTP 起草（需要上一步的 h_p 才能开始）
        ▼
round N+1 的 verify_forward(K tokens)
```

"confirm" 和"round N+1 的 verify"之间存在**真实的数据依赖**（后者的输入——起草出的 K 个
token——本身就是靠 confirm 的输出算出来的），不像"round N 的 verify"和"confirm"之间那样
只是"值已知、只差隐状态"。**能合并的两个调用之间必须没有这种依赖，而这里恰好反过来**：
唯一有"值已知、只差算力"关系的两个步骤（round N 的 verify 和 round N 的 confirm）合并不了，
因为 confirm 需要的正是 verify 跑完之后才能确定的 token；唯一在数据流上顺序相邻、可以合并
调用的两个步骤（confirm 和 round N+1 的 verify）之间又是真实的先后依赖，合并等于要求
"没算完就已经能用结果"。

---

## 4. 我没能验证的东西

1. **`code` prompt 的 `tokens_match=False` 没有深入定性**——不知道是否可重复（同一 GPU 热
   状态下多跑几次）、gap error 量级是否仍在 τ 内。这是 main 分支现状，不是本轮引入的，
   但会拖累"MTP 端到端是否可信"这个更大的问题，建议下一次拿到 GPU 窗口时顺手测一次
   `scripts/b3_verify_batching_logit_agreement.py` 式的定量对比（把它的方法套到"整条
   speculative_decode 路径 vs free_greedy_decode"，而不是它现在测的"verify_forward 内部
   批处理前后"）。
2. **§1.2 的 K=1 探针只在一个上下文长度窗口（~28-133 token）测过一次**，没有像
   `scripts/b3_probe_batching_bar.py` 那样跑多个上下文长度或多次独立进程做稳定性交叉验证——
   方向（(b) 不比 (a) 快）足够清楚不需要更多重复，但绝对毫秒数不建议引用到别处。
3. **没有验证"同位置配对"契约本身是否是 checkpoint 训练时的真实设计**——本轮的论证基于
   (a) `mtp_step` 链式调用内部自洽性（同一次返回的 `(token, hidden)` 永远配对使用）和
   (b) 实测接受率 15%/35% 明显高于随机水平，间接支持当前配对是"能工作的"，但没有找到
   Qwen3.6 官方 MTP 设计文档做直接交叉验证。如果这个契约其实允许"前一个位置的隐状态"
   （即本文档否决的"近似种子"思路），那会打开一个本轮判定为"越界"的优化空间——但那需要
   算法层面的验证（训练分布匹配），不是本轮 B3-a 的范围。
4. **没有探索"减少轮数"这条路**（增大 K、让每轮起草更多 token，从而摊薄"每轮两次 64 层
   遍历"的固定开销）——这本质是接受率/K 配置问题，任务已经把它划给 B3-b，本文不越界处理。
5. **CUDA Graph 化 confirm 步骤没有探索**——§1.2 的探针证明 confirm 的成本是真实的 64 层
   核启动开销，而不是某个可以简单换个函数就省掉的东西；这类开销正是 CUDA Graph 通常用来
   解决的问题，但把 MTP+backbone 的 decode 步骤图捕获是 B2 量级的独立工作量，明显超出
   "把 anchor 折进 verify 块"这一条的范围，本轮不做，只记录为可能的未来方向。

---

## 5. 相关

- `scripts/b3a_probe_confirm_cost.py` —— §1.2 的 K=1 探针
- `scripts/b3_mtp_e2e_acceptance_throughput.py` —— §2 的端到端重测（仅路径自指改动，零算法改动）
- `runtime/mtp_accept.py::determine_accept_reject_from_predictions` —— "committed 恰好一个
  recovery/bonus token，从不计入已接受草稿"的权威定义
- `runtime/model/qwen36_model.py::Qwen36TextModelSelfBuilt.verify_forward`/`Qwen36ForCausalLM
  SelfBuilt.mtp_step` —— §1.1 因果论证的两个锚点
- `docs/implementation-plan.md` §7.1 B3 —— 需要按本文修正 B3-a 那一条的措辞
- `notes/2026-08-02-spec-verify-batching-bar.md` §6 —— 本文修正的那句【推算】
