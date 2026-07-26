# 把 CUDA Graph 接入 decode_batch_sampled(P1)——2026-07-27

## 结论

**已实现并验证正确,但发现了一个更严重、无关的 sparkinfer 回归挡住了完整的服务端吞吐测量。**

## 背景

`notes/2026-07-27-l2-server-integration-gap.md` 核实过:`decode_batch_sampled`
(`runtime/backends/laguna.py:1375`,`ServerEngine._step_sync` 实际调用的方法)是纯
eager,完全不碰 `_decode_cg`——CUDA Graph replay 只在另一个独立的、给 benchmark 用
的 run-to-completion 方法里。本次任务:让 `decode_batch_sampled` 在条件匹配时安全
地走 CG replay。

## 接口约束(读代码确认,不是猜的)

- `LagunaCudaGraphDecode(backend, batch_size)`——`batch_size` 是构造时固定的容量,
  `_ensure_decode_cg()`(`laguna.py:1504`)**硬编码 `batch_size=1`**——当前只存在
  M=1 的捕获形态,这也是本次 session 里`main`所有历史"80 tok/s"benchmark 用的同一个
  实例。
- `replay(slot_ids, token_ids, kv_lengths)`(`laguna_cuda_graph.py:367`)内部
  `_fill_buffers` 只写入 `[0, len(slot_ids))`,但 `capture()` 捕获时录制的图恒定对
  `[:self.batch_size]` 的完整缓冲区跑 forward。**如果 `len(slot_ids) !=
  batch_size`,多出来的行会用上一次 replay 遗留的陈旧 slot_mapping/page_table 数据
  重新计算 attention 并写 KV cache——`_physical_slot` 是 1:1 映射到真实物理槽,这些
  陈旧行可能指向另一个正在使用的真实请求的物理槽,把垃圾写进它的 KV cache**,不只
  是浪费算力。因此只有 batch size 精确匹配才是可证明安全的,不做 padding。
- `capture()` 结尾会调用 `self.unpatch_impls()`,把 attention 层的 `impl` 恢复成
  原始 eager 实现。`replay()` 直接调 `self._graph.replay()`——CUDA Graph 重放的是
  捕获时录制的 GPU kernel 调用序列,**完全不经过 Python 的 `layer.impl.forward()`**,
  所以 replay 语义和当前 `impl` патch 状态无关。之前以为需要在 replay 前调用
  `_repatch_impls_for_cg()`——这个假设是错的,已经用真实测试推翻(见下)。
- 图捕获本身会用 `warmup_slots = [num_slots - batch_size, ..., num_slots - 1]`
  的物理槽写入 dummy 数据——只有在**任何真实请求都还没被分配到任何槽**时调用才
  安全。`ServerEngine.__init__` 的 `min_slots = capacity + (capacity if
  enable_cudagraph else 0)` 公式已经预留了刚好 `capacity` 个额外槽位给这个目的,
  但真正的安全保证来自"在 `start()` 的 admission 循环开始之前调用",不是槽位数量
  本身。
- `decode_batch_sampled` 同时服务 greedy 和非 greedy(温度采样)请求,还支持
  `return_logprobs`。CG 捕获时把 argmax(贪心)直接烤进了图里
  (`capture()`:`self._input_ids[0] = self._logits[0].argmax(...)`),只返回
  token id,没有 logits/logprobs。

## 集成设计

新增 `LagunaBackend._decode_cg_batch_eligible(slot_ids, params_list,
return_logprobs)`(`laguna.py`,`decode_batch_sampled` 前面),四个必要条件全部满足
才返回 True:
1. `self._decode_cg is not None`(已捕获)
2. `not return_logprobs`
3. `len(slot_ids) == self._decode_cg.batch_size`(精确匹配,不做 padding)
4. `all(p.is_greedy for p in params_list)`

`decode_batch_sampled` 开头检查这个条件,满足就 `self._decode_cg.replay(...)` 拿
token,补上原有的 `slot_kv_len`/`slot_committed_tokens` 记账(和 eager 分支一致),
不满足就走原有 `self._forward(...)` 路径,**逐字节不变**。

`server/engine.py._load_laguna_model` 新增:当 `self._enable_cudagraph`
(`ServerEngine` 的构造参数,`server/app.py` 里 Laguna 默认仍是 `False`,本次没有
改这个默认值)为真时,在 `LagunaBackend` 构造完成后**立即**调用
`self.runner._ensure_decode_cg()`——早于 `start()` 的 admission 循环,所以此时
所有槽定义上都是空的,capture 期间的 dummy 写入不会碰到任何真实请求。

## 正确性验证

`benchmarks/verify_decode_cg_integration.py`(已提交),在 sparkinfer
`blackforge-main@3fa9b54`(合并 Laguna 性能分支**之前**的版本,见下面的"发现的严重
问题")上跑通:

- 同一 prompt,先跑 20 步纯 eager(CG 还没捕获),再跑 20 步经过我这次改动的路径
  (CG 已捕获,`_decode_cg_batch_eligible` 逐步返回 True)——**20 个 token 逐一
  比对,`EXACT MATCH: True`**(原始输出见
  `benchmarks/fixtures/verify_decode_cg_integration_20260727.log`)。
- 非 greedy 请求:`_decode_cg_batch_eligible` 正确返回 False,退回 eager,产出
  合理 token,没有崩溃。
- `return_logprobs=True`:正确返回 False。
- `batch_size=2`(捕获时是 1):正确返回 False。
- 混合场景:slot 1 全新 prefill + 非 greedy 解码,在 CG 已捕获、**没有**调用
  `_repatch_impls_for_cg()` 的情况下正常工作——证实了上面那条"replay 和 impl patch
  状态无关"的结论。

CPU 测试:`pytest tests/` 319 passed(3 个失败是 merge 前就存在的、和这次改动无关
的 pre-existing 失败,用 `git stash` 验证过)。

## 发现的严重问题(超出本次任务范围,但必须报告)

在验证过程中,**sparkinfer `blackforge-main@478b9af`(今天早些时候另一个 fork 合并
`master` 的 Laguna kernel 性能分支之后的状态)下,plain eager M=1 decode 会
100% 复现崩溃**:

```
cutlass.base_dsl.common.DSLRuntimeError: DSLRuntimeError: 🧊🧊🧊 ICE 🧊🧊🧊
Caused exception: ... NVPTX compiler invocation failed ...
ptxas application ptx input, line 705; error: Unexpected instruction types specified for 'cvt'
```

- 复现 3/3(含清空 `~/.cache/sparkinfer` JIT 编译缓存后重跑),与上下文长度无关
  (CTX=4096 和 CTX=512 都触发)。
- 崩溃点:`decode_batch_sampled` → `_forward` → 模型 attention 层 →
  `laguna_sparkinfer_attn.py:158`(eager、`enable_cuda_graph=False` 的通用 decode
  attention 路径)→ sparkinfer 的 CuTeDSL JIT 编译。
- **确认是这次合并引入的,不是这个环境本来就有的问题**:把 sparkinfer 临时切回
  `3fa9b54`(合并前)用同一个测试脚本重跑,**完全正常**(本节上面报告的所有正确性
  验证结果都是在这个 pre-merge 状态下跑出来的)。测完已经切回
  `blackforge-main@478b9af`(当前状态),没有遗留在 pre-merge 分支上。
- 这次任务做的验证(`ab_verify_cg.py`,记录在
  `notes/2026-07-27-sparkinfer-merge-and-verify-cg.md`)用的是 `DFlashEngine`,
  从来不走主模型的 plain M=1 decode 路径,所以没有捕捉到这个回归。
- **影响面**:这不只是"CG 集成用不了"——post-merge 的 sparkinfer 上,**连不用
  CG 的最基础的 eager decode 都会崩**,理论上会影响任何触发这条 kernel 特化编译
  的真实流量。这比本任务(P1)优先级更高,需要单独排查(大概率是 traits.py 新增的
  Laguna 特化路径里 `fp8x4_e4m3_to_bfloat2x2_native_sm120` 这条原生 PTX 转换指令
  和这个环境的 ptxas/CUDA 工具链版本不兼容,但没有逐行确认到底是哪个具体 kernel
  变体触发的,留给下一个任务)。

## 性能数字(在 pre-merge sparkinfer 上测的,因为 post-merge 崩溃挡住了测量)

`benchmarks/measure_decode_cg_throughput.py`,64K 不是这次用的上下文(CTX=4096,
200 步,同一张卡):

| 路径 | tok/s | ms/step |
|---|---|---|
| 纯 eager(今天服务默认值,`enable_cudagraph` 对 Laguna 默认 False) | 0.9 | 1130.0 |
| 这次改动的 CG-routed 路径(`enable_cudagraph=True`) | 83.8 | 11.9 |

eager 这个 0.9 tok/s 数字非常反常(比本次 session 其它所有 eager 测量都慢一个数量
级以上),**目前没有拆解成"首次编译一次性开销" vs "稳态开销"**——这是本次 session
第一次测量"完全不用 CUDA Graph 的连续多步 decode_batch_sampled 循环"(之前所有
"eager"对比都是 DFlash 的 M=16 verify,不是这条路径),不排除是 sparkinfer 的
plan/kernel 缓存对每步变化的 `cache_seqlens` 标量值没有命中缓存、每步都重新编译。
94.7x 这个倍数**不应该被当作字面的每步加速比**来引用,但"CG 路径远快于不用 CG 的
连续 decode"这个定性结论是确凿的,也正是当初设计这个 CG 机制的原因。真实的服务端
吞吐增益应该在 post-merge sparkinfer 的崩溃修好之后,用真实 HTTP 流量重新测一次。

## 代码改动

- `runtime/backends/laguna.py`:新增 `_decode_cg_batch_eligible`,`decode_batch_sampled`
  开头加 CG 路由分支。
- `server/engine.py`:`_load_laguna_model` 里 `enable_cudagraph=True` 时在模型
  构造后立即调用 `_ensure_decode_cg()`;更新了两处文档字符串反映现状。
- `benchmarks/verify_decode_cg_integration.py`(新增,正确性验证脚本)
- `benchmarks/measure_decode_cg_throughput.py`(新增,吞吐对比脚本)
- `benchmarks/fixtures/verify_decode_cg_integration_20260727.log` /
  `measure_decode_cg_throughput_20260727.log`(原始输出存档)

## 遗留问题

1. **(高优先级,超出本任务范围)** sparkinfer `blackforge-main@478b9af` 的 plain
   eager decode ptxas 崩溃——需要单独排查修复,目前挡住了在当前 sparkinfer 状态下
   验证/测量本次改动的端到端效果。
2. 本次改动没有改 `server/app.py` 里 Laguna 的 `SERVER_ENABLE_CUDAGRAPH` 默认值
   (仍是 `False`)——只是让这个开关第一次真正起作用。要不要默认打开是运营决策,
   建议先解决上面的 ptxas 崩溃、修好后跑一次真实 HTTP 端到端冒烟,再决定默认值。
3. 没有测试 `num_slots > 2`(多个并发 Laguna 请求同时活跃、batch size 恰好等于某
   个 >1 的捕获值)的场景——当前 `_ensure_decode_cg()` 硬编码 `batch_size=1`,只有
   `SERVER_CAPACITY=1`(Laguna 当前默认)才能命中;如果以后提高 Laguna 并发容量,
   需要重新评估捕获多个 batch size 的图,或者接受那部分流量退回 eager。
4. 没有做真实 HTTP 端到端冒烟(受阻于上面第 1 条),只验证到 `LagunaBackend` 层面。
