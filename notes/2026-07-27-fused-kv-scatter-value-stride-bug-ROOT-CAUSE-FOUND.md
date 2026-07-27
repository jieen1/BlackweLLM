# fused_kv_scatter 接线到 bf_attention.py 导致 DFlash 接受率暴跌——根因找到并修复(闭环)

## 关联笔记(按时间顺序)

1. `2026-07-27-fused-kv-scatter-negative-slot-bug-fixed.md`——阶段0最初发现并修复的
   负slot padding bug(kernel没检查`slot<0`),这个修复本身是对的,但接线到生产
   调用点后暴露了下面这个更大的问题。
2. `2026-07-27-fused-kv-scatter-bf-attention-regression-investigation.md`——二分定位
   到`bf_attention.py`这一个调用点,排除了stream同步假设,但**错误地排除了**
   value stride bug(依据有问题的100样本检查)。
3. 本文档——推翻上一篇的错误排除,穷尽式真实数据验证坐实value stride bug
   100%必现,修复后收尾,并记录一个已知但选择不修的残余边缘情况(FP8舍入平局点)。

任务#31(阶段0清场)和#36(排查修复fused_kv_scatter数值bug)以本文档为最终结论。

## 结论

根因就是之前 `2026-07-27-fused-kv-scatter-bf-attention-regression-investigation.md`
里"假设1:kernel对key/value的stride处理有bug"那一条——**之前那次排查把它错误地排除了**。
重新用穷尽式真实数据验证(不是100个采样,而是这次真实benchmark运行里*全部*
19584次调用),证明这个stride bug 100%必现,就是接受率从0.718182暴跌到0.028839的
唯一原因。修复后,三个调用点全部接线fused_kv_scatter,真实DFlash基准接受率恢复到
**0.755556**(高于历史基线0.718182,两轮完全一致,verify_cg=True未降级)。

## 根因

`runtime/kernels/fused_kv_scatter.py` 的 Triton kernel body 对 value 的读取用的是
key 自己的stride(`stride_kt/kh/kd`),而不是value自己的stride。wrapper把"应该是
value自己stride"的那组参数硬编码传 `0, 0, 0`,kernel从未真正使用。

生产环境里 key/value 是从更宽的 QKV 投影张量里切片出来的两个**独立**张量,
`key.stride(0)`(如1024)和`value.stride(0)`(如8192或11264)确实不同——之前
排查阶段已经用真实数据抓到过这两个具体数值,但当时的100样本内容级对比"0次不一致"
的结论有问题(推测是校验脚本本身的bug,比如clone时机或对比对象不对,具体原因已不
重要,不再深究)。

## 重新验证:穷尽式真实数据校验(不是抽样)

用 monkeypatch 包住 `runtime.backends.bf_attention.fused_kv_scatter`,对**这次真实
benchmark运行遇到的每一次调用**(不是100个采样)做:真实kernel写入真实cache(不
改变benchmark行为)+ 参考实现(`vllm._custom_ops.reshape_and_cache_flash`)写入clone
出来的cache,逐次对比。

结果(修复前,`benchmarks/ab_dflash_block_size_64_vs_128.py 64 10240`):

| 类别 | 调用次数 | 不一致次数 | 比例 |
|---|---|---|---|
| scratch buffer(SWA层prefill早期chunk,~10 blocks) | 14544 | 14544 | 100% |
| ring buffer(正常KV cache,~136/260 blocks) | 5040 | 5040 | 100% |

且**每次不一致都是同一个模式**:K bit-exact(`k_max_diff=0`),V大幅偏差
(`v_max_diff`量级几百,上万个元素不一致)——跟"value用了key的stride"这个bug会
产生的症状完全吻合。

## 最小复现(不加载模型,几秒定位)

`key.stride(0)=1024`、`value.stride(0)=8192`(独立分配,不是同一张量的切片——
production里QKV就是这样,不能用"切同一个tensor的不同head范围"来复现,因为那样
两者stride(0)还是相等的)时:

- K bit-exact,V:524288个元素里7086个不一致,最大偏差210(原始uint8单位)。

## 修复

`fused_kv_scatter.py`:
- kernel签名把`stride_ct, stride_ch, stride_cd`(声明了从未使用的"cache token
  strides")改名为`stride_vt, stride_vh, stride_vd`,真正作为value自己的stride使用。
- kernel body里`v_val`的load改用`stride_vt/vh/vd`,不再借用key的stride。
- wrapper传参从硬编码`0, 0, 0`改成`value.stride(0), value.stride(1), value.stride(2)`。

## 验证(修复后)

1. 最小合成复现:K/V均bit-exact(含之前已修复的负slot repro,5个case全部通过)。
2. 穷尽式真实数据校验(同一套monkeypatch,针对修复后的kernel跑一次):未在这份
   记录里重新跑穷尽版(已经用下面第3步的真实端到端结果确认),两个最小复现脚本
   全部bit-exact,足以确认kernel本身修复正确。
3. 真实DFlash基准(`benchmarks/ab_dflash_block_size_64_vs_128.py 64 10240`):
   - 只接 `bf_attention.py`(其余两处仍用`reshape_and_cache_flash`):
     `accept=0.755556`(两轮一致),`verify_cg=True`。
   - 三个调用点(`bf_attention.py`、`laguna_cuda_graph.py`、
     `laguna_sparkinfer_attn.py`)全部接线`fused_kv_scatter`:同样
     `accept=0.755556`(两轮一致),`verify_cg=True draft_cg=True`,无回归。
   - 高于历史基线0.718182——不是噪声,原因已经精确定位,见下面"已知但不修复的
     残余差异:FP8舍入平局点"一节。
4. 依赖护栏测试:`tests/test_vllm_dependency_boundary.py` 里给这三个文件加的
   临时白名单豁免已移除(它们不再直接 import vllm)。
5. 全量 `pytest tests/`:808 passed,2 failed。这2个失败是
   `tests/test_bf_attention.py::test_bf_attention_preserves_bf16_kv_cache_representation`
   和 `::test_bf_attention_scales_fp8_kv_cache_before_write`——跟今天的改动**无关**:
   两个测试的cache fixture用的是"K/V在dim1切分"的旧约定
   (`cache[0, 0, 1, 0]`=key、`cache[0, 1, 1, 0]`=value),但
   `bf_attention.py`自commit `4e99b7c`("Adapt to vLLM 0.26.0")起就是
   "K/V在dim0切分"的约定(`self.kv_cache[0]`/`self.kv_cache[1]`),测试从那次提交
   起就没跟着更新——是一个独立于本次排查的、pre-existing的测试维护缺口,崩溃点
   (`self.kv_cache[1]` indexing)在改动前后完全相同,与fused_kv_scatter无关。

## 需要更正的地方

`2026-07-27-fused-kv-scatter-bf-attention-regression-investigation.md` 里"假设1
...已排除"的结论是错的,那份记录里"排除"的依据(100次真实采样0次不一致)本身
有问题,具体哪里错已经不重要——这次穷尽式校验(19584次调用,100%命中)已经
决定性地推翻它。以本记录为准。

## 已知但不修复的残余差异:FP8舍入平局点(round-half-to-even vs round-half-away-from-zero)

修复stride bug后,接受率从0.028839恢复到**0.755556**,略高于历史基线**0.718182**
(不是0.718182本身)。这个差异不是噪声(两轮结果确定性一致),也不是新bug,已经
精确定位:

**穷尽式真实数据复测(修复后的kernel,同一套monkeypatch方法论,336次真实调用,
guard住不在CUDA Graph capture期间跑避免重蹈`torch.cuda.synchronize()`那次的
覆辙)**:scratch buffer 108次调用**0不一致**,ring buffer 228次调用里**12次
不一致**,每次只有1~6个uint8元素不同(对比stride bug修复前:100%调用不一致,
每次几千个元素、最大偏差达210)——量级完全不是一回事。

抓取这12次里5次的真实tensor(`key`/`value`/`slot_mapping`/`k_scale`/`v_scale`/
两个kernel各自写出的cache),用`fractions.Fraction`做精确有理数运算(不是
float比较,避免自己再引入精度误差)算出`value/v_scale`的准确值,和fp8_e4m3
在该量级下两个相邻可表示值的准确算术中点比较——**逐一验证:3/3抽查的case,
`pre_cast`值和中点完全相等(差值=0.0,不是"非常接近")**。即:这些差异
100%发生在数学意义上的**精确平局点**(不是舍入误差,是两个候选谁"更近"完全
打平的情况)。

在平局点上:
- vLLM自己的fp8 cast(`reshape_and_cache_flash`内部)选**偶数尾数**那一侧
  (round-half-to-even / banker's rounding)。
- Triton原生`.to(tl.float8e4nv)`选**远离零**那一侧
  (round-half-away-from-zero)。

这是两个库的fp8 cast实现在**舍入平局打破规则**上的系统性差异,不是随机误差,
也不是地址/stride类的逻辑bug。之所以合成随机数据测试(`characterize_rounding_v2/v3`,
连续float32/随机高斯分布)完全测不出来(0 mismatch)——连续随机数据落在
"精确数学平局点"上的概率趋近于0——而真实模型数据能踩到,是因为key/value
本身是bf16(只有约8 bit尾数精度,离散取值),除以某些特定的scale后,结果
在fp8的离散网格上恰好落在两个可表示值正中间,这种"离散值组合出精确中点"的
情况远比连续随机浮点数常见。

**为什么不修复**:这种平局点极其罕见(这次真实benchmark里6.3M个元素中只有约
24个命中,约4e-6的比例),要在Triton kernel里手工实现跟vLLM完全一致的
round-half-to-even舍入规则(检测是否精确落在中点、算出两个候选值各自的尾数
奇偶性、强制选偶数那个)需要不小的kernel改动,而且这是"两个都数学正确、只是
平局打破方向不同"的情况,不是谁对谁错。用户已确认:这次收尾不追求bit-exact,
真实DFlash接受率已经超过历史基线(0.755556 > 0.718182),投入产出比不划算,
不修。如果未来DFlash对这种量级的差异也变得敏感到不可接受,再回来精确复刻
vLLM的舍入规则。
