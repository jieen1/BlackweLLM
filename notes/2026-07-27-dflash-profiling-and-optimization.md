# DFlash 真实 profiling + 优化(2026-07-27)

## 结论先行

**之前"DFlash 天然比不投机解码慢"的结论是错的。真正原因是一个纯粹的软件 bug:
verify CUDA Graph 每次 replay 前用 Python for 循环给 page table 逐元素赋值,64K 上下文
下单次 replay 要做 1000+ 次零碎标量写入,烧掉 ~180ms 纯 CPU 调度开销,而这段 GPU 计算
本身只需要 ~38ms。修好这一个函数后:**

| | 修复前 | 修复后 | 倍数 |
|---|---|---|---|
| DFlash 完整生产路径(`ab_verify_cg.py`,64K,真实 `DFlashEngine.generate`) | 45.14 / 46.55 tok/s | **252.89 / 259.14 tok/s** | **~5.6×** |
| 不开 DFlash 的纯 M=1 decode CG(今天早些时候测的) | ~80 tok/s | ~80 tok/s(未变) | — |

**DFlash 现在比不投机解码快 ~3.2 倍,不是慢 43%。** 用户对"完整优化后不可能这么慢"的
质疑是对的,是我之前的分析方向错了。

## 方法论:真实分阶段 + kernel 级 profiling(不是估算)

- 环境:同一份已验证的 sparkinfer `blackforge-main`、vLLM `e12b91b032`+patch、独立
  venv `vllm-repro80`,64K 上下文(`CTX=65536`,和今天全部 A/B 测试口径一致),
  `QSR_VERIFY_CUDA_GRAPH=1`。
- 阶段计时:`benchmarks/profile_dflash_round.py`,用 `torch.cuda.Event` 精确包住
  `generate_verify_only` 主循环里的每个阶段(verify replay / accept-reject 的
  GPU→CPU 同步 / draft KV 预计算 / draft replay / 纯 Python 记账),热身 10 轮后
  统计 30 轮稳态数据。
- kernel 级 profiling:`benchmarks/profile_dflash_verify_kernels.py`,热身 8 轮后用
  `torch.profiler.profile(activities=[CPU, CUDA])` 采样 10 次 verify replay,按
  kernel 名字归类到 attention/MoE/dense GEMM/norm/其它。

## 第一轮结果:round 耗时 95% 在 verify replay(修复前)

64K 上下文,`mode="verify"`(今天早些时候已经切好),修复**前**:

| 阶段 | 均值 | 占比 |
|---|---|---|
| **verify_replay** | **205.70 ms** | **95.4%** |
| draft_replay | 3.36 ms | 1.6% |
| accept_reject_sync(GPU→CPU) | 0.44 ms | 0.2% |
| draft_kv_precompute | 0.80 ms | 0.4% |
| bookkeeping(纯 Python) | 0.50 ms | 0.2% |
| **round_total** | **215.50 ms** | 100% |

verify replay 一家独大,draft/accept-reject/记账全部可以忽略。

## 第二轮:verify replay 内部的 kernel 级真相——GPU 只花了 37.5ms

对准 verify replay 单独用 `torch.profiler` 做 kernel 级采样(同样 64K,修复前):

| 类别 | ms/replay | 占比 |
|---|---|---|
| MoE | 21.95 | 58.5% |
| dense GEMM | 6.29 | 16.8% |
| attention | 4.46 | 11.9% |
| 其它 kernel | 3.39 | 9.0% |
| memcpy/memset | 1.00 | 2.7% |
| norm | 0.40 | 1.1% |
| **GPU kernel 总计** | **37.50** | 100% |

**决定性矛盾**:CUDA-event 测出 verify replay wall-clock 是 205.70ms,但 GPU 实际
kernel 执行只有 37.50ms——**差了 5.5 倍**。这个缺口不可能来自"计算量",因为
profiler 已经把所有 GPU kernel 都算进去了。

## 根因:1109 次逐元素 Python 赋值

同一份 profiler 输出里,`aten::copy_` 一项:**CPU total 79.74%(1.825秒/10次 replay
=182.5ms/replay),但 CUDA total 只有 6.088ms**——即每次 `copy_` 调用本身的 GPU 工作
几乎为零,但 Python/ATen 调度开销高达 164.56us/次,一次 replay 里被调用 **1109 次**。

读 `runtime/backends/laguna_cuda_graph.py` 的 `LagunaCudaGraphVerify._fill_buffers`
(verify CG 每次 replay 前、graph 内部计算之前调用,不在图内,是纯 Python/eager 代码)
找到了原因——**给 page table 填值用的是逐元素 Python for 循环**,其中最大的一处:

```python
# 原来的写法(已删除)
pt = self._page_tables[group_key]
for j in range(n_blocks_full):
    pt[0, j] = full_base + j
```

64K 上下文、`block_size=64` 下 `n_blocks_full = ceil(65536/64) = 1024`——**单次
replay 光这一个循环就是 1024 次零碎 GPU tensor 元素写**,加上其它几个更小的循环
(input_ids/positions/SWA ring page table/两处 slot mapping),精确对上 1109 次
`aten::copy_`。

**同一个文件里 decode CG 的 `_fill_buffers`(`LagunaCudaGraphDecode`,136-198 行)
早就用向量化写法**(`pt[0, :n_blocks] = torch.arange(base, base + n_blocks, ...)`),
verify CG 的这个函数是后加的、没跟上这个模式——纯粹是遗漏,不是什么"M=16 计算量"
的架构限制。

## 修复

`runtime/backends/laguna_cuda_graph.py`:`LagunaCudaGraphVerify._fill_buffers` 全部
向量化:

- `input_ids`/`positions`:循环 → 一次 `torch.tensor`/`torch.arange` 赋值。
- 全注意力 page table(最大的一项,1024 元素):`for j in range(n_blocks_full): pt[0,j]=...`
  → `pt[0, :n_blocks_full] = torch.arange(full_base, full_base + n_blocks_full, ...)`。
- SWA ring page table:同样从逐元素循环换成 `torch.arange` + 向量化取模。
- 两处 slot mapping(全注意力/SWA):复用同一份 `pos_range = torch.arange(kv_len, kv_len+nt)`,
  向量化计算,不再逐元素写。

语义逐行核对过和原循环完全一致(`arange(full_base, full_base+n)` 等价于
`full_base+j for j in range(n)`,取模/整除运算保持原顺序),不是近似实现。

## 验证

### 正确性:接受率精确不变

修复前后 `ab_verify_cg.py 1`(64K,256 max_tokens,`DFlashEngine.generate`,贪心)的
`acceptance_rate` 都是 **0.6869565217391305**(13 位小数完全一致)——投机解码的接受率
对任何数值偏差都极敏感,能精确复现同一个值是最强的正确性证据。

### CPU 回归

`pytest tests/`:319 passed,3 个既有失败(`test_bf_attention.py` ×2、
`test_vllm_dependency_boundary.py` ×1),和这次改动无关,数量、名字修复前后一致。

### 阶段耗时(修复后,同样 64K、30 轮稳态)

| 阶段 | 修复前 | 修复后 | 倍数 |
|---|---|---|---|
| verify_replay | 205.70 ms | **37.83 ms** | 5.44× |
| round_total | 215.50 ms | **44.16 ms** | 4.88× |
| draft_replay / accept_reject / kv_precompute / bookkeeping | 都 <1ms(未受影响) | 同左 | — |

修复后 verify_replay(37.83ms)和修复前独立测出的"GPU kernel 真实耗时"(37.50ms,
torch.profiler)几乎完全吻合——证实修复前的 168ms 缺口就是这个循环的调度开销,
现在基本清零。

### 端到端:两种测量方法互相印证

- 用阶段耗时反推:`round_total=44.16ms`,K=15、接受率 0.687 → 每轮约 11.30 个有效
  token(`15×0.687+1`)→ 隐含吞吐 `11.30/0.04416s ≈ 256 tok/s`。
- 独立跑完整生产路径(`ab_verify_cg.py 1`,不是 profiling 脚本,是真实
  `DFlashEngine.generate_verify_only` 整个循环):**252.89 / 259.14 tok/s**。

两个独立测量互相印证(差距 <2%),不是巧合,证明这次修复的收益是真实的、可在生产
路径复现的,不是 profiling 脚本本身的假象。

### 上下文长度核实(排除"测的是短上下文"的可能性)

两个 profiling 脚本都显式 `CTX = 65536`,和当天所有 A/B 测试(`ab_verify_cg.py`)
口径一致;kernel 级 profiling(37.5ms GPU 真实耗时)和阶段耗时(修复后 37.83ms)
都是在同一个 64K 设置下测的,两者吻合本身就证明不存在"用短上下文冒充 64K"的问题。

## 回答最初的问题:M=16 到底比 M=1 贵多少

真实倍数:**37.5ms(M=16 verify 真实 GPU kernel 时间,64K)÷ 约 13ms(M=1 decode 单步,
今天早些时候 `notes/STATUS_speed_optimization_0726.md` 测的)≈ 2.9 倍**——远不是之前
文档估算的 16 倍。MoE 占了 verify replay GPU 时间的 58.5%,这部分确实会随 M 增长
(每个 token 独立路由到 top_k=10 个专家),但增长幅度远小于线性 16 倍;attention 只
占 11.9%,符合"KV 读取内存带宽受限、M=16 复用同一次读取"的预期,涨幅很小。

**这次投机解码之所以"看起来"贵,九成以上不是"M=16 算力开销"本身,而是一个和 M 完全
无关的、可以直接向量化消除的 Python 循环 bug。**

## 代码改动

- `runtime/backends/laguna_cuda_graph.py`:`LagunaCudaGraphVerify._fill_buffers` 向量化。
- `benchmarks/profile_dflash_round.py`(新增,阶段级 profiling 脚本)
- `benchmarks/profile_dflash_verify_kernels.py`(新增,kernel 级 profiling 脚本)
- `benchmarks/fixtures/dflash_round_profile_after_fix_20260727.txt`、
  `dflash_verify_kernel_profile_before_fix_20260727.txt`、
  `dflash_ab_verify_cg_after_fillbuffers_fix_20260727.txt`(原始输出存档)

## 遗留问题

1. draft CG(`_draft_cg`)的 `_fill_buffers` 没有做同等排查——draft_replay 本身已经
   很便宜(3.4-3.6ms),不是当前瓶颈,但没有确认它内部是否也有类似的未向量化循环
   (只是因为 buffer 更小所以影响没那么大),值得后续顺手查一遍。
2. 没有测试不同上下文长度(4K/128K/200K)下 verify replay 的耗时曲线,只验证了
   64K 这一个点。理论上向量化后的 page table 填充成本应该随 `blocks_per_slot` 线性
   增长(64K→128K 大约再翻一倍的 page table 元素数),但向量化后单次 `torch.arange`
   调用本身极快,预计影响很小,没有实测确认。
3. 252-259 tok/s 这个新数字目前只在这个独立 profiling/A-B 环境里测出来,还没有
   走一遍今天早些时候做的真实 HTTP server 端到端路径(P1 那次的方法论)——DFlash
   本身也还没接入 server 主循环(`任务 #12`,之前判断"DFlash 比不投机解码慢,不
   值得接入"的结论现在被推翻了,值得重新评估这项任务的优先级)。
4. 没有检查代码库里其它 CUDA Graph 相关的 `_fill_buffers`/等价函数是否有同样的
   "逐元素循环" 反模式(这次只查了 verify 这一个,是不是还有其它地方也在偷偷烧
   CPU 调度开销,没有做全仓库排查)。
