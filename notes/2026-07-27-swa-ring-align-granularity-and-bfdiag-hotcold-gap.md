# SWA ring 对齐粒度排查 + bfdiag 热/冷 gap(2026-07-27)

**⚠️ 2026-07-27 追加更正,请先读 `notes/2026-07-27-block-size-128-accept-rate-root-cause-CLOSED.md`**:
本笔记"结论先行"第 1 点("对齐粒度假设已被数据排除")是基于**逐 round 独立**的相关性
分析框架得出的,这个框架本身是对的(round 级别确实不相关,方向甚至相反),但据此推出
"对齐粒度不是主因"是**过早的**——后续用 ring 内容直接对比 + aux_hidden_states 逐层对比
证明了真正机制是"对齐粒度分叉触发一次性的临界翻转,再沿因果链级联放大,和之后每一轮
自己是否分叉无关",以及"bs=64 强制用 128 粒度对齐会把接受率拖到和 bs=128 一样差"这个
直接的因果验证。**对齐粒度确实是根因,只是通过"触发+级联"而不是"每轮独立付出对齐税"这个
机制起作用**。第 2 点(page_table/cache_seqlens 裁剪不一致的独立 bug)和 bfdiag 热/冷 gap
的记录仍然有效,未受影响。完整、最新的结论以 CLOSED 笔记为准,本笔记保留作为排查过程记录。

对 `notes/2026-07-27-block-size-128-migration-and-tie-break-noise.md` 的更新:
该笔记"纯浮点临界翻转"的结论已被 sparkinfer 团队独立找到的
`forward_extend_generic.py` `wait_group` race(`num_warps_kv>1` 分支缺了一处
从 `wait_group(1)` 改成 `wait_group(0)` 的历史修复)取代为更精确的根因描述。
修复后(sparkinfer 侧改动,非本仓库代码)接受率的 run-to-run 不稳定性消失,
稳定复现 0.452525,但和 bs=64 的 0.718182 之间仍有约 0.27 的确定性差距,
根因尚未定位。本笔记记录后续两条独立排查线的结果。

## 结论先行

1. **SWA ring 对齐粒度(`aligned_start` 按 block_size 取整)假设已被数据排除**,
   不能解释 bs=64 vs bs=128 的主 gap。生产配置(`align_gran == block_size`)
   下相关的溢出/陈旧读取机制**永远不会触发**,而主 gap 在生产配置下是完全
   确定性、稳定复现的,不带任何本次排查发现的"多值饱和"特征。
2. 排查过程中确认了一个**真实、独立、和主线无关**的 bug:SWA ring 的
   `cache_seqlens` 用未裁剪长度,`page_table` 只 populate 裁剪后的条目数,
   两者不一致时会读到 CUDA Graph 预分配 buffer 里的陈旧内容。生产配置下
   零裕量但不触发,值得记录、不用现在修。
3. bfdiag 热引擎(daemon/`bf exec`)测出的接受率和冷启动基线对不上
   (同一配置,热 0.6754 vs 冷 0.452525),是一个独立的、新工具(标注过
   "真实 provider 从未运行过")的已知问题,需要工具开发那边跟进,不在本次
   排查范围内。

## 假设来源

用户读 `laguna_cuda_graph.py:_fill_buffers_b1` 的 SWA 分支发现:

```python
window_start = max(0, kv_len - window + 1)
aligned_start = (window_start // ps) * ps      # 对齐粒度 = block_size
aligned_len   = new_kv - aligned_start
```

`ps` 从 64 变 128,对齐粒度翻倍 → `aligned_len` 最多比原来大 64,直接喂给
`cache_seqlens`。假设:如果 kernel 的 window_left 掩码没有把这多出来的部分
裁掉,block_size=128 会让 SWA 层多看到最多 64 个本该被滑窗屏蔽的旧 token。
同样的模式在三处代码里都存在:`laguna_cuda_graph.py` 的
`LagunaCudaGraphDecode._fill_buffers`/`_fill_buffers_b1`(main 模型 decode)、
`LagunaCudaGraphVerify._fill_buffers`(main 模型 verify,DFlash 实际走的路径)、
`laguna_dflash_cudagraph.py` 的 `DFlashDraftCudaGraph._fill_buffers`(draft
模型自己的 ring)。

## 证伪设计:把对齐粒度和物理 block_size 解耦

在 `laguna_cuda_graph.py`/`laguna_dflash_cudagraph.py` 加了
`QSR_DEBUG_SWA_ALIGN_GRANULARITY` 诊断开关(`_debug_swa_align_gran()`,不设
时行为完全不变,读取值替代 `bs` 作为 `aligned_start` 的取整粒度),可以独立
于物理 block_size 调节对齐粒度。

### 第一版(字面测试:`aligned_start = window_start`,完全不取整)

结果:**accept = 0.0**(全拒绝)。不是证实/证伪信号——这是我自己分析确认的
一个新 confound:page_table 按物理页(128)寻址,取整粒度设成 1 后,
`cache_seqlens`(未裁剪的 `aligned_len`)描述的区间末端会比 page_table 第
0 项实际起始位置(仍然是 floor 到物理页边界)更晚,导致 kernel 认为窗口在
**末尾**少了几十个 token(包括刚写入的自身 KV)——这是我的探针设计缺陷,
不是有效测试。

### 第二版(dose-response:粒度 128/256/512,块寻址仍安全,因为都是 128 的整数倍)

先用 bfdiag 热引擎(`bf exec`)测,发现 gran=128(生产默认)在热引擎下给出
0.6754,和冷启动基线 0.452525 对不上(见下方"bfdiag 热/冷 gap"一节)——热
引擎数字不可信,遂改用一直验证过的冷启动脚本重跑。

冷启动结果(block_size=128,CTX=10240,`benchmarks/ab_dflash_block_size_64_vs_128.py`,
2 轮):

| align_gran | Round0 | Round1 |
|---|---|---|
| 128(生产默认) | 0.452525 | 0.452525 |
| 256 | 0.502222 | 0.333333 |
| 512 | 0.502222 | 0.333333(与 256 逐位相同) |

**关键现象**:align_gran=128 在本会话所有测试里都是逐位可复现的(包括同一
会话内更早的强制 `torch.cuda.synchronize()` 测试)。粒度调宽到 256/512 后,
同一进程连续两轮给出不同结果——这是整个会话第一次在
`SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=1` 保护下观察到真正的
run-to-run 不一致。且 256 和 512 给出的两个值逐位相同,不是随粒度渐进变化,
是收敛到同一组"饱和态"。

## 定位到的真实 bug:cache_seqlens/page_table 裁剪不一致

`LagunaCudaGraphVerify._fill_buffers`(`laguna_cuda_graph.py`,DFlash verify
实际路径)及 `DFlashDraftCudaGraph._fill_buffers`(draft 自己的路径)里:

```python
n_ring = min(-(-aligned_len // bs), self._ring_blocks_per_slot)
...
pt[0, :n_ring] = ...                     # page_table 只填 n_ring 项(裁剪后)
self._cache_seqlens[group_key][0] = aligned_len   # cache_seqlens 用未裁剪值
```

当真正需要的 block 数超过 ring 物理容量(`self._ring_blocks_per_slot`)时,
`page_table` 里 `n_ring:` 之后的条目**不会被这次调用刷新**,仍是 CUDA Graph
预分配 buffer 里上一次 replay 留下的陈旧内容,但 `cache_seqlens` 却告诉
kernel"有这么多有效 token"——kernel 会读到陈旧、和当前 KV 内容无关的
page_table 条目。

**触发条件推导**(main 模型 SWA:`window=512`,`qo_max=nt=16`,
`block_size(bs)=128`,`ring_blocks_per_slot=6` 即容量 768 slots):

```
aligned_len_worst = window - 1 + qo_max + slack = 527 + slack   (slack = window_start % align_gran)
溢出条件: 527 + slack > 768  ⟺  slack > 241        (与 align_gran 本身无关,只看绝对 slack)
```

- align_gran=128(生产默认):`slack ∈ [0,127]`,**永远 ≤127 < 241,不会溢出**。
  ring 容量公式(`_ring_blocks_for_window` 里的 `+1`)本来就是专门为
  align_gran=block_size 时的最坏对齐余量设计的,零裕量但严丝合缝。
- align_gran=256:`slack ∈ [0,255]`,约 5.5% 的轮次溢出。
- align_gran=512:`slack ∈ [0,511]`,约 52.7% 的轮次溢出。

溢出时读到的"陈旧内容"具体是哪个物理 ring block,取决于此前若干轮 replay
碰巧在那个位置留下什么——这天然随 Python/CUDA 调度的轮次间细节变化,表现上
和 GPU race 无法区分,但机制上是"确定性代码路径读取了未定义状态",不是真正
的并发写后读竞争。256 和 512 触发概率差一个数量级但收敛到同一组具体数值,
说明触发后系统进入少数几个"饱和态"而非概率线性叠加——这一点具体机制还没有
完全搞清楚,不过度解读。

## 为什么这个 bug 排除了对齐假设作为主 gap 的解释

生产配置(align_gran=128)按上面的算式**从不触发**这条溢出/陈旧读取路径,
而 0.452525 vs 0.718182 的 gap 在生产配置下是完全确定性、稳定复现的,不带
本次观察到的"多值饱和"特征。因此"SWA ring 对齐让 kernel 多看到 pre-window
内容"这个假设,作为主 gap 的解释,**被数据排除**。这条 bug 是排查过程中的
真实副产物,值得记录和以后修(如果 align_gran 相关的代码路径未来被复用到
其他 window/block_size 组合、裕量不再是零的场景),但不是当前任务的根因。

**后续如果要真正修复**:让 `n_ring` 溢出时 `cache_seqlens` 同步截断到
`n_ring * bs`(而不是保留未裁剪的 `aligned_len`),或者在 ring 容量公式里
把 align_gran 也纳入余量计算——按本仓库约定这类 sparkinfer 调用惯例代码是
我们自己的(`runtime/backends/laguna_cuda_graph.py`/`laguna_dflash_cudagraph.py`),
不是 sparkinfer 源码,可以自己改,但目前没有必要(生产配置不触发)。

## bfdiag 热引擎 / 冷启动 accept-rate 不一致(工具已知问题,非本次任务范围)

排查对齐假设时先用了新合并的 `bf` CLI(`bfdiag` 热引擎)。发现原生
`bf daemon start` 不支持 `--block-size`(provider 一直硬编码走
`LagunaBackend` 默认值 64),补了这个参数(`bfdiag/daemon/{provider,server,cli}.py`
+ 3 处 CPU 单测断言,`LOAD_TIME_CONFIG_KEYS` 里加了 `block_size`,走冷重启不
允许热切换)。`pytest tests/test_bfdiag_daemon.py tests/test_bfdiag_canary.py`
52 passed。

补完之后用 `bf exec` 在热 daemon 里复现"生产默认对齐"配置
(block_size=128,CTX=10240,同一 prompt,同一 `SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=1`),
得到 **accept=0.6753623188405797**,连续两次 `bf exec` 逐位一致(自洽),但
和这个会话里反复验证过的冷启动基线 **0.452525** 对不上,差了约 0.22。

排查过、排除的可能性:
- prompt/tokenizer 不一致:核对过 `bf exec` 里 `tokenizer.encode(BASE_TEXT)`
  产出的 token 和已知的 period-11 pattern 集合一致(只是起始相位不同,是
  预期内的边界效应)。
- KV cache 内容残留:读代码确认 `LagunaBackend.reset_slot()` 对 full-attn 和
  SWA 层的 KV cache tensor 都是真正的 `.zero_()`,不只是清计数器；
  `reset_laguna_engine` 还会额外清 draft KV cache。

没排查完的方向(留给工具开发那边):daemon `load()` 阶段 CUDA Graph capture
时的 warmup 状态是否真的等价于一次全新 capture;金丝雀自检本身够不够严格
(固定短步数,不确定有没有覆盖到 SWA ring 需要跑很久才触发的 wrap-around
状态)。`docs/diagnostics-guide.md` 本身标注"真实 provider 从未运行过"——这
是这套工具第一次在 GPU 上跑,这类 gap 正是文档里提示要当"首次运行"验证的
那类问题。**在这个 gap 定性之前,不建议用 bfdiag 热引擎做 DFlash accept-rate
相关的 A/B 结论**,冷启动脚本仍是当前唯一可信的测量方式。

## 主线现状

0.452525(bs=128 生产默认)vs 0.718182(bs=64)的确定性 gap,根因**仍未
定位**。已排除:sparkinfer wait_group race(已被 sparkinfer 团队修复)、
ring/scratch 拷贝逻辑、draft KV 写入路径寻址、kernel 本身(正确 GQA 形状下
byte-identical 输入给出 cosine=1.0)、SWA ring 对齐粒度(本笔记)。下一个
二分点待定。
