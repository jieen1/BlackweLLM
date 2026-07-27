# fused_kv_scatter 接线到 bf_attention.py 导致 DFlash 接受率暴跌——排查进展(未闭环)

## 背景

阶段0(#31)把3个 `reshape_and_cache_flash` 调用点换成 `runtime/kernels/fused_kv_scatter.py`
(负slot bug已修复,见 `2026-07-27-fused-kv-scatter-negative-slot-bug-fixed.md`)后,DFlash
接受率从历史稳定值 0.718182 暴跌到 0.028839(确定性,不是竞态)。

## 已经用二分法精确定位到具体调用点

同一个 fused_kv_scatter kernel,分别单独接入3个调用点测试(`ab_dflash_block_size_64_vs_128.py 64 10240`):

| 组合 | 接入的调用点 | 接受率 |
|---|---|---|
| A | `bf_attention.py` | **0.028839(故障)** |
| B | `laguna_cuda_graph.py` | 0.718182(正常) |
| C | `laguna_sparkinfer_attn.py` | 0.718182(正常) |

**bug精确定位在 `bf_attention.py` 这一处调用点,不在kernel本身**(同一个kernel在B/C两处工作完全正常)。

注:B/C两处的 `do_kv_cache_update` 看起来像未被调用的死代码(`runtime/`内搜不到调用点),
实际是被**vLLM自己的内部代码**(`qwen3_dflash.py:586`、`attention.py:784`等,duck-typed
接口)调用的,所以B/C的测试结果是真实有效的。

## 已排除的假设

### 假设1:kernel对key/value的stride处理有bug(排除,但发现一个独立的真实隐患)

读代码发现 `fused_kv_scatter.py` 的 kernel body 对 key 和 value 都用 `stride_kt/kh/kd`
(key自己的stride)去读,wrapper 传给"应该是value自己stride"的那组参数
(`stride_ct/ch/cd`)硬编码传 `0,0,0`,kernel body 从未真正使用这组参数。

真实抓取 `bf_attention.py` 调用点的100次真实生产数据(确认这100次确实来自触发故障的
配置),验证 key/value 的 stride 确实不同(key:`(1024,128,1)`,value:`(8192或11264,128,1)`),
逐一做内容级对比:**0次不一致**。

**结论:这是一个真实存在、独立于本次故障的kernel实现缺陷**(如果key/value的stride在
某个未来场景下真的以数值敏感的方式产生差异,会读到错误的内存位置),**但不是导致这次
接受率暴跌的原因**——已用真实数据排除,不是主观判断。**需要单独修复**(给value单独传
它自己的stride,kernel body对应改成用各自的stride读),不阻塞当前排查,不是这次的优先级。

### 假设2:stream同步/写后读时序问题(排除)

`bf_attention.py` 是三处调用点里唯一"KV cache写入→同一函数内紧接着调用
`self.impl.forward()`做attention读取"的调用点,怀疑Triton kernel launch和sparkinfer
attention实现之间可能存在流同步语义差异,导致写后读的时序问题。

验证:在写入后、attention读取前插入 `torch.cuda.synchronize()`(用
`torch.cuda.is_current_stream_capturing()` 保护,确保只在真正 replay 时同步,不破坏
CUDA Graph capture本身)。

结果:`verify_cg=True`(配置和baseline一致,不是被破坏后的降级路径),**接受率仍然是
0.028839,分毫未变**。

**结论:不是stream同步/时序问题,已排除。**

## 当前状态

- kernel本身内容级验证正确(100+次真实数据对比,含此前已确认的负slot修复)。
- 是`bf_attention.py`这一个具体调用点导致的,不是其它两处。
- stride不匹配(真实但独立的隐患)和stream同步都已用真实实验排除。
- 根因仍未找到,需要新的排查方向。

## 尚未验证的候选方向(供后续参考,不代表下一步一定是这个)

- `bf_attention.py`的`forward()`里,`k`/`v`是`key`/`value`经过`.view(-1, num_kv_heads, head_size)`
  得到的——这个view操作本身、以及紧随其后调用`self.impl.forward(self, q, k, v, self.kv_cache, meta, out)`
  时传入的`self.kv_cache`(不是`k_cache`/`v_cache`局部变量)是否和fused_kv_scatter实际写入的
  是同一块内存(要确认`self.kv_cache[0]`/`self.kv_cache[1]`跟`k_cache`/`v_cache`是不是共享
  存储,view/别名关系有没有被破坏)。
- `bf_attention.py`是不是同时被DFlash草稿模型和主模型共用(如果是,两者的`num_kv_heads`/
  `head_dim`等参数是否有可能在某次调用中不一致,导致shape相关的隐藏bug)。
