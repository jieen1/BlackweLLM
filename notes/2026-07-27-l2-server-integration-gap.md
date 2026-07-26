# Laguna L2 服务集成缺口核实(2026-07-27 01:00)

## 为什么写这份笔记

任务是"优化 DFlash 表现和缓存表现"。深入之后发现:这两项要优化的东西,**在真实 HTTP
服务路径里根本不存在**——本次 session 和更早所有 DFlash/CUDA-Graph/sparkinfer 的
benchmark 数字,全部来自独立脚本直接驱动 `LagunaBackend`/`DFlashEngine`,从未经过
`server/engine.py` + `server/app.py` 这条真实服务路径。`docs/roadmap.md`(最后修改
`4d8290f`,07-26 08:44,比今天的 DFlash 排查早)第 0 节执行看板里已经把这个缺口列为
**"当前最大的结构性风险，优先级应高于继续榨 kernel 收益"**——本次核实:两天过去了,
这个结论今天(07-27)用代码重新验证依然成立,没有被修复。

## 逐项代码核实(不是抄 roadmap,是重新读代码确认)

### 1. Laguna 前缀缓存 = 永久 miss 桩实现

`runtime/backends/laguna.py:1469-1472`:
```python
def reconcile_prefix_hit(self, token_ids: list[int]) -> int:
    """E1 stub: Laguna has no persistent content-addressed prefix cache
    yet (roadmap L2/L3 TODO) -- every admission is a cold miss."""
    return 0
```
`runtime/block_pool.py` 里那一整套内容寻址前缀缓存(`PROGRESS.md` P0-P4,数百小时验证
过的系统)是给**另一个 runner**(`DirectModelRunner`,"Qwen3.6 runner")用的,`199ac67`
commit message 明确写"block_pool.py RESERVED_PHYSICAL_SLOTS=1 is separate (Qwen3.6
runner) and unchanged"——两套 runner 是分开的,Laguna 完全没有继承这套前缀缓存。

### 2. DFlash 投机解码没有接入 server 主循环

`server/engine.py` 里 `grep -n "DFlashEngine\|dflash"` **零匹配**。`ServerEngine.
_load_laguna_model`(391 行)`import`的是`LagunaBackend`,不是`runtime/backends/
laguna_dflash.py`里的`DFlashEngine`。主循环`_step_sync`调用的是`self.runner.
decode_batch_sampled`/`self.runner.mtp_verify_and_commit_batch`——这是`DirectModelRunner`
风格的方法名,`LagunaBackend`要嘛没实现要嘛是兼容桩,**不是**`DFlashEngine`那套 K=3
投机解码状态机。也就是说:本次 session 排查的 verify CG、接受率、K/V 竞态修复等等,
全部发生在一条真实请求永远不会走到的代码路径上。

### 3. Laguna 走 server 时,CUDA Graph / 前缀缓存 / SWA ring buffer 全部默认关闭

`server/app.py:77-135` 的注释非常诚实,直接引用:

> "laguna" drives the new LagunaBackend second tenant (roadmap Track E / L2) --
> it has no CUDA Graph integration yet (Lane 2 GPU work lives in
> `runtime/backends/laguna_cuda_graph.py`, not wired into the engine), no
> persistent prefix cache, and no session affinity, so those three default
> OFF for it below... It also has no SWA ring-buffer KV yet (roadmap L2 TODO)
> -- every discovered attention layer, including the 36 sliding-window ones,
> currently gets a KV cache sized for the FULL context ceiling, so per-token
> memory cost is ~4x the roadmap L0 budget note's estimate.

实测默认值:
```
SERVER_MODEL_BACKEND        默认 "qwen36",不是 "laguna"(要显式指定才会服务 Laguna)
SERVER_ENABLE_CUDAGRAPH     Laguna 默认 "0"
SERVER_ENABLE_PREFIX_CACHE  Laguna 默认 "0"
SERVER_KV_CACHE_DTYPE       Laguna 默认 "auto",不是本次全程验证的 "fp8_e4m3"
```

### 4. server 端 MoE backend——重新核实:其实不是问题

`server/engine.py:402` 的 `moe_backend=os.environ.get("QSR_MOE_BACKEND", "marlin")`
**不控制是否使用 sparkinfer MoE**。`LagunaBackend.__init__`(`runtime/backends/
laguna.py:175`)无条件调用 `self._patch_moe_sparkinfer()`,和这个 vLLM `EngineArgs`
参数完全无关——`moe_backend="marlin"` 只影响 vLLM 最初怎么加载/格式化 FusedMoE 权重,
随后立刻被我们的代码整体替换成 sparkinfer kernel。**结论:sparkinfer MoE 在 server
的 Laguna 路径上本来就一直生效,这条不是缺口,是我最初核实不够仔细写错的。**

### 4b.(更正)CUDA Graph 服务路径:比想象的缺口更大

最初以为 `decode_batch_sampled`(`ServerEngine._step_sync` 实际调用的方法)可能已经
通过 `_decode_cg_enabled` 环境变量(`QSR_DECODE_CUDA_GRAPH`,默认开启)间接吃到
CUDA Graph——**这个假设是错的,已核实推翻**。`decode_batch_sampled`
(`runtime/backends/laguna.py:1375`)直接调用 `self._forward(...)`(eager),完全
不触碰 `_decode_cg`。CUDA Graph replay 只存在于另一个独立方法(约 1540-1560 行,
`generate_fast` 风格,一次性跑完整个生成的单请求辅助方法),专门给独立 benchmark
脚本用,和 `ServerEngine` 的逐步、多槽、持续批处理调用模型不兼容。

好消息:`LagunaCudaGraphDecode.replay(slot_ids, token_ids, kv_lengths)` 本身是支持
多槽批量 replay 的(`laguna_cuda_graph.py`),所以理论上可行的修复路径是让
`decode_batch_sampled` 在条件匹配时(全 batch greedy、CG 已捕获、batch 组成和捕获时
一致)改走 `self._decode_cg.replay(...)`,不匹配则退回现有 eager 路径——这是一个有
边界、可测试的改动,但不是简单的默认值翻转,需要专门设计+验证(batch size 不匹配、
greedy/非 greedy 混合等边界情况),不在本次直接实现。

## 结论

**本次 session(以及更早)所有关于 DFlash 接受率、verify CG、80 tok/s、sparkinfer
kernel 加速的测量,都是在 backend 层 + 独立 benchmark 脚本上做的,一条都没有验证过
真实 HTTP 请求打过去会发生什么。** 真实服务路径(`QSR_SERVER_MODEL_BACKEND=laguna`
启动)现在很可能是:无 CUDA Graph(纯 eager)、无前缀缓存(每次全量重算)、无 SWA
ring buffer(每层按满上下文分配 KV,显存×4)、MoE 走 marlin 不是 sparkinfer、且没有
DFlash 投机解码——换句话说,今天验证的所有性能成果,在生产服务路径上可能一个都拿
不到。

这比"verify CG 该不该开"或者"省几百 MB 显存"的优先级都高。

## 为什么这次没有直接动手实现

把 Lane 2(CUDA Graph + sparkinfer MoE/attention + DFlash + SWA ring buffer +
前缀缓存)接入 `server/engine.py` 是一个触及请求处理主循环的结构性改动,不是可以在
一轮后台任务里安全冲完的小修改:

1. `LagunaBackend` 需不需要新实现 `decode_batch_sampled`/`mtp_verify_and_commit_batch`
   等接口方法(还是已经有兼容实现,只是没被验证过),这本身要先读清楚再动手。
2. 当前有一个并行 fork 正在改 sparkinfer(合并 master 分支、尝试修 verify CG),
   在它跑完并确认 sparkinfer 侧状态稳定之前,不应该再开一条同时改 `server/engine.py`
   主循环 + `LagunaBackend` 接口的并行改动线——两边都在动同一套 attention/CUDA-Graph
   机制,同时改容易互相踩。
3. 项目自己的门禁文化(`AGENTS.md`/`docs/roadmap.md`)要求"证据先行、一次一个变量、
   过完整回归"——这种量级的改动应该按 P0→P1→P2 拆解成可独立验证的小步(先接 sparkinfer
   MoE 默认值,再接 CUDA Graph,再接前缀缓存,再接 DFlash),不应该一次性糊在一起。

## 建议的下一步顺序(等当前 fork 完成、GPU 空出来之后)

1. ~~P0~~(已核实,不是问题,见上文 4):`moe_backend` 参数不控制 sparkinfer MoE 是否
   生效,`_patch_moe_sparkinfer()` 无条件调用,server 路径本来就在用 sparkinfer MoE。
2. **P1**(已核实,比预期工作量更大,见上文 4b):`decode_batch_sampled` 目前是纯
   eager,要接 CUDA Graph 需要新写代码——在 batch 组成匹配捕获形状且全部 greedy 时
   走 `_decode_cg.replay(...)`,否则退回现有 eager 路径,并处理 batch size/组成
   不匹配的边界情况。这是一个需要专门设计和测试的改动,不是简单摸个开关。
3. **P2**:前缀缓存——`reconcile_prefix_hit` 目前是永久 miss,要不要给 Laguna 接一套
   等价机制,是照搬 `DirectModelRunner` 的内容寻址方案还是设计更简单的版本,需要
   单独评估(工作量可能不小,历史上 `DirectModelRunner` 那套花了 P0-P4 好几轮)。
4. **P3**:DFlash 接入主循环——需要先确认 `server/engine.py` 的请求循环模型
   (`decode_batch_sampled` + `mtp_verify_and_commit_batch` 的分离式循环)能不能
   直接换成 `DFlashEngine` 的 `generate_verify_only` 状态机,还是要写一层适配。

这四步都应该分开验证、分开提交,不要合并成一个大改动。
