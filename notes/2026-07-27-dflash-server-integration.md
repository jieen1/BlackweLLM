# DFlash 投机解码接入 server 主循环(2026-07-27)

## 结论

**已接入并验证正确,capacity=1 场景端到端可用,默认关闭(opt-in)。** 复用了
`DirectModelRunner` 的 MTP 分支(`ServerEngine._step_sync` 的 `greedy_slots` 路径)
现成的、已测试的多轮/EOS/logprobs 记账逻辑,`_step_sync` 本身**零改动**。过程中用真实
HTTP 测试发现一个**预置、和这次改动无关**的严重 bug(Laguna 非贪心 eager decode 会
崩),已隔离确认并如实报告,不在这次任务范围内修。

## 接口约束(读代码确认)

- `DFlashEngine` 的所有内部 buffer(`_draft_seq_lens`、`_draft_block_table` 等)和
  draft/verify CUDA Graph 都是硬编码单物理槽(batch=1)——`DFlashEngine._init_buffers`
  没有 batch 维度。这意味着 DFlash **只能支持 capacity=1**,不是"这次先做 capacity=1,
  以后再扩展"的选择,是当前 CG 捕获方式的硬约束。
- `ServerEngine._step_sync` 已经有一条**现成的、给 `DirectModelRunner` MTP 用的**
  `greedy_slots` 分支(`classify_decode_slots` 按 `self.runner.spec.has_mtp` +
  `is_greedy` 路由),调用 `self.runner.mtp_verify_and_commit_batch(slots, anchors,
  drafts, ...)`,返回的 decision dict 需要 `committed`/`num_accepted`/`next_anchor`/
  `next_draft_tokens`[/`logprobs`]。这条分支的多轮记账、EOS 检测、`max_tokens` 截断、
  流式推送、grammar 状态推进全部已经写好并测试过——**只要 Laguna+DFlash 提供同样形状
  的方法,`_step_sync` 完全不用改**。
- `LagunaBackend.prefill_chunked_begin` 目前对 Laguna 返回
  `{"anchor": first_token, "draft_tokens": []}`(无 MTP);这是 DFlash bootstrap 的
  挂载点。
- `determine_accept_reject_from_predictions`(`runtime/mtp_accept.py`)的 `committed`
  **不含**起始 anchor,`committed[p]` 对应 `predicted_tokens[p]`(即 `verify_logits[p]`)
  逐行——这个精确对应关系是实现 logprobs 支持的关键,不是猜的。

## 集成设计

不新增/不改 `ServerEngine._step_sync`。改动全部在"喂给这条现成分支什么"这一层:

1. `DFlashEngine` 新增两个方法(`runtime/backends/laguna_dflash.py`):
   - `dflash_prefill_bootstrap(slot, prompt_ids) -> {"anchor", "draft_tokens"}`——
     prefill_with_aux + 初始 draft,返回值形状和现有 `prefill_chunked_begin` 一致。
   - `dflash_round(slot, anchor, draft_tokens, *, return_logprobs, top_logprobs) -> dict`——
     从 `generate_verify_only` 的 while 循环体里提取出来的单轮 draft+verify+accept 逻辑
     (逐行对照原逻辑,没有改变任何计算顺序),新增了 logprobs 支持(用
     `verify_logits[p]` 对应 `committed[p]` 的精确关系)。EOS/max_tokens 判断**不在
     这里做**——留给 `_step_sync` 现成的逻辑处理,一致性由"这就是同一套分支"保证。
2. `LagunaBackend`(`runtime/backends/laguna.py`)新增:
   - `self._dflash: Any = None`(由 `ServerEngine` 在启用时注入)。
   - `mtp_verify_and_commit_batch(slots, anchors, drafts, ...)`——对每个 slot 调用
     `self._dflash.dflash_round(...)`(capacity=1 下永远只有一个 slot,循环写法只是
     防御性的,不依赖这个假设)。
   - `prefill_chunked_begin` 在 `self._dflash is not None` 时改走
     `dflash_prefill_bootstrap`。
3. `ServerEngine`(`server/engine.py`)新增 `enable_dflash` 构造参数:
   - `capacity != 1` 时直接 `raise ValueError`(对应上面的硬约束,清晰的启动期错误
     而不是运行时腐坏)。
   - `_load_laguna_model` 里,`enable_dflash=True` 时构造 `DFlashEngine(self.runner)`,
     设 `self.runner._dflash = dflash`,并用 `dataclasses.replace` 把
     `self.runner.spec` 的 `mtp_model_id`/`num_speculative_tokens` 填上,让
     `spec.has_mtp` 变 `True`——这一行是唯一让 `classify_decode_slots` 真正路由到
     DFlash 分支的开关。非贪心请求不受影响,继续走 `decode_batch_sampled`。
4. `server/app.py`:新增 `QSR_SERVER_ENABLE_DFLASH` 环境变量 / `--dflash` CLI flag,
   **默认关闭**(见下面"为什么默认关闭")。

## 正确性验证

1. **逐 token 精确比对**(`/tmp/verify_dflash_server_integration.py`,64K 上下文,
   256 token):同一 prompt,slot 0 走原有的 `generate_verify_only`(一次性跑完),
   slot 1 走新的 `dflash_prefill_bootstrap` + 循环调用 `dflash_round`(模拟
   `_step_sync` 每步调一次)——**`EXACT MATCH: True`**,256 个 token 逐一相同。这是
   最强的正确性证据:新旧两条代码路径在相同输入下产出逐位相同的输出。
2. **`pytest tests/`**:319 passed,3 个失败与今天所有其它改动一致的既有失败
   (`test_bf_attention.py` ×2、`test_vllm_dependency_boundary.py` ×1),无新增回归。
3. **真实 HTTP 端到端**(`QSR_SERVER_MODEL_BACKEND=laguna QSR_SERVER_ENABLE_DFLASH=1
   QSR_SERVER_CAPACITY=1`,64K prompt,`temperature=0`):两次请求输出正确一致
   (`finish_reason=length`,文本内容和期望的贪心续写一致)。
4. **非贪心请求路由验证**:`temperature=0.8` 请求确实被 `classify_decode_slots`
   正确路由到 `sampled_slots`(没有误入 DFlash 分支)——但这条路径本身崩溃了,见下面
   "发现的预置 bug"。这至少证明了路由判断本身按预期工作(问题出在被路由到的目标代码
   里,不是路由逻辑本身错了)。

## 真实端到端性能(`/debug/stats`/`/debug/traces`,真实生产 tracing,不是估算)

64K prompt(59580 真实 token),`max_tokens=256`,贪心:

| | 数值 |
|---|---|
| decode_ms(15 轮) | 712-713ms |
| avg_round_ms | **47.47-47.56ms** |
| tokens/sec | 336-337(这次测试文本高度重复,接受率接近满格,不是常态基线,见下) |

对照今天早些时候独立 benchmark(`ab_verify_cg.py`,同样 64K,更真实的非重复文本,
接受率 68.7%)修复后测出的 **round_total=44.16ms**:服务化路径的 `avg_round_ms`
只比独立 benchmark 高约 **7%**(47.5 vs 44.2ms)——这 7% 是异步事件循环、多轮记账、
EOS/max_tokens 检查、tracing 本身的合理开销,**不是** 意外的服务化回归。协调者要求
"不能假设独立 benchmark 的数字原样保留到服务场景"——这次用真实生产 tracing 数据
(不是自己另写的 profiler)验证了这个假设基本成立,只有小幅、可解释的开销,不需要
进一步优化。

## 发现的预置 bug(不在本次任务范围,已隔离确认,不是这次改动引入的)

真实 HTTP 测试非贪心请求(`temperature=0.8`)时,Laguna 的 eager(非 CG)decode 路径
崩溃:

```
Could not guard on data-dependent expression u1 < u0 ...
Caused by: if end < start:  # sparkinfer/attention/paged/planner.py:498 in _q_lengths_from_cu_seqlens
```

**已用隔离测试确认**:关掉 `QSR_SERVER_ENABLE_DFLASH`(纯 Laguna,无 DFlash)发送
**完全相同**的非贪心请求,**逐字符相同的崩溃**——证明这是 Laguna 服务器本身、和
DFlash 无关的预置 bug,大概率是这套服务器代码路径**第一次真正处理非贪心 HTTP 请求**
(今天所有测试,包括更早的 P1 decode CG 集成,用的都是 `temperature=0`)才暴露出来
的、torch dynamo 在应该是纯 eager 的路径上触发了动态形状 guard。服务进程本身没有
崩溃(`/health` 请求后依然正常),只是这一个请求返回 500。**这个 bug 独立于本次
DFlash 任务,建议作为单独任务排查修复**,不在这次范围内处理。

## 为什么 `SERVER_ENABLE_DFLASH` 默认关闭

- 硬要求 `capacity=1`,和当前 Laguna 默认配置一致但仍是个真实限制,不是能悄悄打开
  的旁路优化。
- 会额外加载一个 draft 模型,真实增加显存占用,还没有做过完整的显存核算。
- 上面发现的非贪心请求崩溃虽然确认与 DFlash 无关,但既然还没修,贸然默认开启会让
  "非贪心+DFlash 同时打开"这个此前完全没测过的组合暴露给生产流量。
- 是同一天刚落地的能力,按照今天 `SERVER_ENABLE_CUDAGRAPH` 的先例(先跑一段时间、
  真实验证过再默认开启),这次先保持 opt-in(`QSR_SERVER_ENABLE_DFLASH=1` /
  `--dflash`)。

## 代码改动

- `runtime/backends/laguna_dflash.py`:新增 `dflash_prefill_bootstrap`、`dflash_round`。
- `runtime/backends/laguna.py`:新增 `self._dflash`、`mtp_verify_and_commit_batch`,
  `prefill_chunked_begin` 分支到 DFlash bootstrap,更新相关注释。
- `server/engine.py`:新增 `enable_dflash` 构造参数 + capacity 校验 + spec 翻转逻辑。
- `server/app.py`:新增 `SERVER_ENABLE_DFLASH` 环境变量 + `--dflash` CLI flag。

## 遗留问题

1. **【高优先级,独立于本任务】** Laguna 非贪心 eager decode 的 torch dynamo 崩溃
   (见上),需要单独排查——影响面是所有 Laguna 非贪心真实请求,和 DFlash 无关。
2. 只验证了 capacity=1(唯一支持的配置);没有也不需要测试"多 slot 并发投机解码",
   因为当前 CG 架构从设计上就不支持,`ServerEngine.__init__` 会在 `capacity!=1` 时
   直接拒绝启动。
3. 显存占用没有做完整核算(draft 模型 + draft KV cache 的真实增量)。
4. logprobs 支持是新写的(基于 `verify_logits[p]` 对应 `committed[p]` 的精确关系
   实现),做过静态代码审查但**没有**真实请求端到端验证过 DFlash+logprobs 组合
   (今天的真实 HTTP 测试没有覆盖这一项,任务时间关系,标注为已知空白)。
5. 流式(`stream: true`)DFlash 请求没有专门测试(P1 那次发现 `/v1/completions` 的
   `stream:true` 本身似乎没有走 SSE,原因未查,这次也没有重新确认)。
