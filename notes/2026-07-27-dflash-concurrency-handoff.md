# DFlash 并发支持 + 深度性能优化——交接记录(2026-07-27,fork 因上下文接近上限被主动停止)

## 为什么写这份笔记

负责"DFlash 多槽并发支持(#22)"+"DFlash 深度性能优化(#23)"的 fork(agentId
`a6c9b3fd9ccb3b7a9`)运行时间很长、被多次恢复,上下文接近上限,主动停止它并在这里
记录当前真实状态,方便后续接手,不丢失已经做出的进展。

## 当前代码状态(截至停止时刻)

`/home/bot/project/qwen-sm120-runtime`(主 worktree,分支 `main`,HEAD `5f4bdb3`)有
**两个文件的未提交改动**(`git diff --stat`:`server/app.py` +10/-3,
`server/engine.py` +37/-9),GPU 空闲,无残留进程:

- **`server/app.py`**:去掉 `--dflash` 必须 `--capacity 1` 的限制,`--help` 文案改成
  说明 DFlash 的 CUDA Graph 是按 slot 顺序 replay、吞吐随 capacity 次线性增长。
- **`server/engine.py`**:
  1. `ServerEngine.__init__` 里去掉 `enable_dflash` 强制 `capacity==1` 的检查,换成
     一段注释说明真实机制:DFlash 的 draft/verify CUDA Graph 是对**一套共享 scratch
     buffer**捕获的(不是像 decode CG 那样按 batch 形状捕获),`_fill_buffers(slot,
     ...)` 在每次 replay 前都会重新从 `slot` 算出物理地址,所以并发靠的是"每个活跃
     slot 顺序各 replay 一次",不是一次批量 replay N 个 slot——没有 capacity 上限,
     但也不是真正的批量并行,是 N 次顺序单槽 replay。
  2. **一个独立但很有价值的 bug 修复**:`_engine_thread_main` 里,模型加载失败时原来
     只 `logger.exception` 然后仍然 `self._ready_event.set()`——这会让 `start()` 的
     `wait()` 正常返回、`/health` 显示"健康"(200),但实际上引擎线程已经退出、
     `_step_sync` 永远不会跑,**所有真实请求会静默地永远挂起**。修复:记录
     `self._load_error`,`start()` 检测到就重新抛出,让服务启动失败得响亮而不是
     静默假装健康。这个 bug 是在测试 DFlash capacity>1 时(一次 `blocks_per_slot`
     设置过大导致 draft 模型加载 CUDA OOM)意外发现的,和 DFlash 并发本身是两回事,
     但值得保留。

**这两处改动都还没有做完整的 GPU 验证就被中断了**(下面详细说),不确定是否可以直接
提交——接手时请先重新走一遍验证,不要假设已经验证过。

## 已经确认、有真实证据支撑的结论

1. **DFlash 多槽正确性(部分验证)**:`verify_dflash_multi_slot.py`(NUM_SLOTS=4,
   交错 vs 单独跑基线比对)报告过 `ALL_MATCH: true`——但这是在上面两处代码改动
   **之前**还是**之后**跑的,fork 的消息里没有说清楚,需要重新确认。
2. **真实 HTTP 并发测试**:`verify_dflash_concurrent_http.py` 已经写好,是否跑完、
   结果如何,最后一条消息被下面第 4 点的发现打断,**没有给出最终结果**。
3. **MoE `deterministic_output` 成本拆解**(为 #23 深度优化做准备,尚未接入真实
   profiling 数据验证):
   - 物理行分配确定性(`989723d`)是真实数值 bug 修复,预计算开销应该不大,**不能动**。
   - "phase 2"路由输出物化(`deterministic_output=True` 时 `route_output_rows`
     最多到 160 行,而非确定性模式只用 1 行 + atomic scatter)是更可能的性能大头,
     但**没有实测 kernel 级数据**,只是代码读出来的推断。`profile_moe_determinism_
     cost.py` 已经写好但没跑完。
4. **【最重要、优先级最高的发现,已经写进任务 #21】**:`decode_batch_sampled` 在
   `num_reqs>1`(一次处理超过 1 个 slot)的 eager 路径下,**必现** `torch._dynamo`
   "data-dependent expression u1 < u0" 崩溃——和贪心/非贪心无关,单纯是"eager 路径
   一次处理 >1 个请求"就会崩。因为 decode CUDA Graph 固定 `batch_size=1`,
   `capacity>1` 时只要同时有 >1 个 slot 需要 decode,batch size 就对不上 CG、精确
   匹配失败、**必然**掉回这条会崩的 eager 路径。**推论:整个 Laguna 后端(不只
   DFlash)可能从未真正支持过 `capacity>1` 的真实并发**,这个严重性远超"DFlash
   并发支持"这个任务本身。DFlash 走 `mtp_verify_and_commit_batch`(逐 slot 顺序调
   `dflash_round`),不经过 `decode_batch_sampled`,不受这个 bug 直接影响,可以
   干净交付。
5. **未解之谜**:DFlash capacity=4 测试里出现过一次 270 秒的诡异延迟,fork 怀疑
   可能和上面第 4 点同源(`torch._dynamo` 在本该是纯 eager 的路径上被意外触发
   tracing/重编译),**没有验证清楚**。

## 接手时的建议顺序

1. **先处理任务 #21(严重性升级后的版本)**——这是阻塞整个 Laguna 并发服务的核心
   问题,优先级高于把 DFlash 并发这个具体任务收尾。用一个干净的最小复现脚本
   (`decode_batch_sampled`,`num_reqs=2`,贪心,不涉及 DFlash)先确认崩溃、定位
   `torch._dynamo` 具体是在 sparkinfer 的哪个函数触发(fork 提到过
   `_q_lengths_from_cu_seqlens`,需要重新核实是不是同一个点)。
2. 重新验证上面"当前代码状态"里两处未提交改动的正确性(多槽比对 + 真实 HTTP 并发),
   确认后再决定要不要提交。
3. 补上 270 秒延迟之谜的验证。
4. 如果 #21 修复需要改 sparkinfer 代码,按今天已经建立的规矩:先把证据和方案发给
   用户批准,不要擅自合并。
5. 完成后再回到 #23(深度性能优化,目标对齐 vLLM 367 tok/s)的 kernel 级 profiling。

## 环境提醒

- `/home/bot/project/sparkinfer`:`blackforge-main @ 14cb350`,干净,今天已验证过,
  不要动,除非确认需要改动并拿到用户批准。
- `/home/bot/vllm`:`e12b91b032` + patches,今天已决定不升级到 0.26.0,不要动。
- 独立 venv `/home/bot/.venvs/vllm-repro80`,不要碰主 venv。
- GPU 独占约束依旧:同一时刻只允许一个 agent 用 GPU,现在还有另一个 agent
  (`a595e32a81e0765f0`)在独立 worktree `.claude/worktrees/laguna-prefix-cache`
  做前缀缓存 L-P0/L-P1,也在排队用 GPU,协调时要考虑到它。
