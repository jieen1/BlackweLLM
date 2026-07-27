# fused_kv_scatter.py 数值bug:缺少负slot(padding)跳过检查(已修复)

## 背景

阶段0(#31)把 `reshape_and_cache_flash` 换成 `runtime/kernels/fused_kv_scatter.py`
(此前从未接线过的Triton FP8 KV scatter kernel)时,DFlash接受率从稳定复现的0.718182
暴跌到0.028839(确定性,不是竞态),已revert并单独建任务(#36)排查。

## 根因(读代码直接定位,GPU最小复现验证,不需要跑完整DFlash)

对照vLLM自己的Triton参考实现(`vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:61-64`):

```python
slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
if slot_idx < 0:
    # Padding token that should be ignored.
    return
```

`slot_mapping` 里的 `-1` 是vLLM的标准"跳过这个token"占位符,CUDA Graph定长batch场景下
(实际token数少于捕获时的batch size)必然会用到——`fused_kv_scatter.py` 的docstring自己
写着"CG-compatible",正是这类场景的目标用途,但实现里完全没有这个检查。

`block_idx = slot // block_size` 对 `slot=-1` 算出 `block_idx=-1`,再乘以block stride
得到一个越界的负内存偏移,写坏了不该动的内存位置。

## 最小复现(不需要加载模型,几秒钟跑完)

`/tmp/repro_fused_kv_scatter_bug.py`(未提交,临时脚本):构造4×64×8×128的假KV cache,
分别用 `reshape_and_cache_flash` 和 `fused_kv_scatter` 写入相同的key/value:

- 全部合法slot(`[5,70,130,200]`):两边逐字节相同 ✓(核心scatter/量化逻辑本身没问题)。
- 含一个 `-1`(`[5,70,-1,200]`):**参考实现正确跳过,`fused_kv_scatter` 往
  `[block=3, slot=63]` 写入了垃圾数据**,坐实根因。

## 修复

`runtime/kernels/fused_kv_scatter.py` 加了和vLLM参考实现一致的检查:

```python
slot = tl.load(slot_mapping_ptr + pid)
if slot < 0:
    return
```

修复后重跑最小复现:单个-1、多个-1、全部-1(no-op batch)、block_size=128 四种场景
全部逐字节相同。`pytest tests/`:808 passed,2个既有失败(`test_bf_attention`×2,
与本次改动无关)。

## 状态

kernel本身的bug已修复并验证。**尚未重新接线到生产调用点**(`bf_attention.py`/
`laguna_cuda_graph.py`/`laguna_sparkinfer_attn.py` 三处 `reshape_and_cache_flash`
调用)——这是阶段0原计划留下的活,现在kernel已知是对的,可以重新走一遍接线+
真实DFlash基准的bit-exact验证。
