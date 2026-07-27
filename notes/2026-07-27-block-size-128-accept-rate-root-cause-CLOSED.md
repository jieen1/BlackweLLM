# block_size 64→128 接受率差距:完整排查收尾(触发机制已闭环,深层数值机制仍有一处未解)(2026-07-27)

**2026-07-27 追加更新**:下方两个后续验证测试的结果放在文末新增的"两个修复前置验证"一节。
简要结论:**"对齐粒度是触发源"这件事被进一步坐实**(bs=64 强制用 128 粒度对齐后接受率从
0.718182 暴跌到 0.443333,几乎复现 bs=128 自己的 0.452525),但**"kernel 对 masked 区域的
cache_seqlens 长度数值敏感"这个具体子假设被推翻**(固定内容、固定 window_left、只变
cache_seqlens 的孤立 kernel 测试给出 cosine=1.0,逐字节相同)。这两个结果放在一起说明:
对齐粒度确实是因、但不是通过"单次 kernel 调用对 mask 区域数值敏感"这条路径起作用的,真正
的数值机制还在多轮/状态累积的某个环节,尚未定位到底层原因。下方原始正文(闭环声明)基于
这两个测试之前的证据链,读到"两个修复前置验证"一节之前请把"已闭环"理解为"触发条件已闭环,
底层数值机制部分闭环"。

对 `notes/2026-07-27-block-size-128-migration-and-tie-break-noise.md`(结论:纯浮点临界翻转)
和 `notes/2026-07-27-swa-ring-align-granularity-and-bfdiag-hotcold-gap.md`(对齐假设排查)的
最终收敛。这篇笔记是这次排查的完整记录 + 最终根因,后续如果有人重新踩到"block_size=64 vs
128 接受率不一致"这个问题,应该先读这篇,不要重新走一遍弯路。

## 结论先行

**根因链条(每一环都是直接测量到的,不是推断)**:

1. 主模型(不是 draft)verify 步骤的 SWA 分支里,`aligned_start` 按 `block_size` 取整
   (`runtime/backends/laguna_cuda_graph.py`,`LagunaCudaGraphVerify._fill_buffers`,我们自己的代码)。
   block_size 从 64 变 128,取整粒度翻倍,导致 `aligned_len`(喂给 kernel 的 `cache_seqlens`)
   在某些 kv_len 上比 bs=64 多报出最多 64 个 token 的窗口长度——这个分叉在 CTX=10240 这次
   复现里第一次发生在 **kv_len=10306**。
2. 这个 `cache_seqlens` 分叉直接导致主模型在 kv_len=10306 这一轮 verify forward 里,**自己的
   中间层 hidden state 就已经产生了差异**——从最浅的采样层开始只有 0.0156 的微小差异(和
   bf16 舍入噪声同量级),经过后续每一层**单调放大**,到最深层放大到 20.0(见下方"最终验证"
   一节的完整数字)。这不是"kernel 对不同 block_size 结果不同"(kernel 本身已经被证明无罪,
   见下),是"给 kernel 的 cache_seqlens 长度不同 → kernel 内部 split-KV/分块方式不同 →
   浮点求和顺序不同 → 即使是本该被 window_left 正确掩码为无效的那部分,也会通过改变内部
   归约方式间接影响被掩码区域自身的数值结果"这类现象(和本仓库其它地方记录的"R6 级别"
   split-KV 数值噪声同一类,只是这次触发维度是对齐粒度而不是 split-KV 开关)。
3. 这个 hidden state 差异通过 `precompute_and_store_context_kv` 写进 draft 模型的 context KV
   cache。因为 SWA 是 causal/windowed attention,一旦某个位置的内容被污染,**后续每个位置的
   计算都会 attend 到这个已经不同的位置**,差异沿因果链继续放大、不会自愈——直接测量到:
   ring buffer 里 kv_len=10306 之前的 417 个位置逐字节完全相同,10306 开始的 91 个位置
   (采样窗口上限)**无一例外**全部不同,且量级不是 FP8 舍入边界抖动(单个位置 K 向量
   128 维里 110 维不同,V 的 max_diff 到 76.0)。
4. draft 模型此后连续约 20+ 轮(kv_len 10362→10493,约 130 个 token)自己的 top-1 预测
   几乎全错(很多轮 13-15/15 miss),而 bs=64 全程没有这个现象。DFlash 的 accept/reject 机制
   兜底纠正了所有错误 draft(最终提交的 256 个 token 序列两边逐位相同),但效率因此大幅下降,
   这就是 0.452525 vs 0.718182 那个差距的**主要成因**(不是唯一成因——bs=64 自己也有 7 个
   ULP 级别的临界翻转 miss,那部分仍然是良性噪声,和这条链条无关)。

## 完整排查时间线(每一步的方法和为什么推翻/推进到下一步)

1. **最初的误判**:找到一个分叉点(kv_len=10307,top-2 候选完全相同、margin=0.125=1 ULP)
   就下结论"整个 gap 是纯浮点噪声"。**用户明确拒绝**这个"一个例子下结论"的逻辑。
2. **穷举分析纠正**(`analyze_divergences.py`):把 bs=64/bs=128 各自的完整 draft 轨迹
   (325/543 个位置)全部对照 ground truth(利用"贪心解码下绝对位置的 token 是路径无关的
   确定性函数"这个性质)。结果:bs=64 是 7/325=2.15% miss、全部 ULP 级近似平局;bs=128 是
   338/543=62.25% miss,其中只有 25 个是近似平局形状,16 个还超过 5×ULP,313 个连 ground
   truth 都不在 top-2 里——**证伪"纯噪声"结论**。
3. **kernel 隔离测试 v1**:构造相同 FP8 内容喂两种 page_size,直接走 sparkinfer 真实调用
   路径,得 cosine=1.0。但用错了 Q_HEADS(8,gqa_group_size=1),不是真实形状(应为 72/8
   或 48/8),**结论不可靠,自报存疑**。
4. **sparkinfer 团队独立发现真实 bug**(不是我发现的):`forward_extend_generic.py` 的
   `num_warps_kv>1` 分支里一处历史修复(commit d2d8cb9)漏改,FP8 KV decode 用了
   `wait_group(1)` 而不是应该统一的 `wait_group(0)`,是个真实的 producer/consumer race。
   sparkinfer 侧修复后(不是我们改的),run-to-run 不确定性消失,稳定在 0.452525,但和
   bs=64 的 0.718182 仍有约 0.27 的**确定性**差距。
5. **强制同步测试**:在 draft CG replay/verify replay/accept-reject/draft KV precompute 全部
   插入 `torch.cuda.synchronize()`,接受率纹丝不动(还是 0.452525)——**排除残留 race**,
   证明剩下的 gap 是和执行时序无关的确定性差异。
6. **kernel 隔离测试 v2**(用正确 GQA 形状重做):SWA(72/8,gqa=9,window_left=511)和
   full-attn(48/8,gqa=6,window_left=-1)两种真实形状,N_CTX=10240(真实分叉量级,不是玩具
   规模),cosine=1.0,max_abs_diff=0.000000——**kernel 本身在正确形状下确认无罪**(喂
   byte-identical 输入,两个 block_size 输出逐字节相同)。
7. **对齐假设提出**(用户读代码发现):`aligned_start` 按 block_size 取整,64→128 粒度翻倍,
   假设"多看到的 pre-window 内容"是根因。
8. **证伪测试第一版**(字面上 `aligned_start=window_start` 不取整):accept=0.0。**不是
   证伪/证实信号**,是我自己分析确认的一个新 confound——page_table 按物理页寻址,粒度设成 1
   后 `cache_seqlens`(未裁剪)和 page_table 实际起始位置不一致,kernel 认为窗口末尾少了
   几十个 token(包括自身 KV),全拒绝。**顺带定位了一个真实、独立的 bug**:
   `n_ring`(page_table 填充条目数)被 clip 到 ring 物理容量,但 `cache_seqlens` 没有同步
   裁剪——生产配置(align_gran=block_size)下永远不触发(容量公式的 `+1` 余量正好卡死),
   但粒度调宽后会触发,读到 CUDA Graph 里未刷新的陈旧 page_table 条目。已做成永久性不变量
   `bfdiag_checks.check_page_table_covers_seqlen`,接线补丁已应用到
   `LagunaCudaGraphVerify._fill_buffers`。
9. **dose-response 测试**(align_gran=128/256/512,固定 block_size=128):热引擎(`bf exec`,
   新合并的 bfdiag daemon)先测出一套数字,但发现**热引擎和冷启动同配置对不上**
   (0.6754 vs 0.452525,差了和主 gap 同量级!)——不是这次排查的东西,记进
   `2026-07-27-swa-ring-align-granularity-and-bfdiag-hotcold-gap.md`,标记"新工具已知问题,
   工具组跟进",改用一直可信的冷启动脚本重测。冷启动结果:gran=128(生产默认)0.452525
   (稳定复现),gran=256/512 两轮内部就不一致(0.502222/0.333333,首次在本次排查里看到
   `SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=1` 保护下的 run-to-run 不一致),且 256/512
   给出的两个值逐位相同——指向同一个"陈旧 page_table 读取"饱和态(第 8 步定位的那个 bug)。
10. **round 级别相关性分析**(用户带来的关键纠偏):用 `bf shapes --diff --kv-len N`(纯 CPU,
    不用 GPU)对已采集的 543 个位置逐 round 算 `aligned_len(64)` vs `aligned_len(128)` 是否
    相同,和 miss 率交叉。结果**方向和假设相反**:aligned_len 不同的轮次 miss 率反而更低
    (56.14% vs 66.67%)。但意外发现:bs=128 从 kv_len=10362 开始连续 20+ 轮几乎全错,
    和这一轮自己 aligned_len 是否相同完全无关——**这是"每轮独立付出对齐税"模型的反例,
    但指向一个更具体的"某处触发、然后持续污染"的模式**。
11. **ring 内容直接对比**(不经过 kernel,读 CUDA Graph 预分配 buffer 原始 FP8 字节):
    kv_len=10306 之前 417 个位置逐字节相同,10306 开始的 91 个位置(窗口上限)无一例外
    全部不同,且量级不是 FP8 舍入边界抖动(见结论第 3 点)。**干净的二值边界,精确对应
    round 级别分析里 aligned_len 第一次分叉的那个位置**。
12. **最终验证:aux_hidden_states 直接对比**(这次收尾的最后一环):在 kv_len=10306 这一轮,
    直接 dump 主模型 verify forward 产出的 6 层采样 hidden state(`aux_hidden_states`,
    DFlash draft 输入的来源),两边逐层对比:

    | 层(浅→深) | max_abs_diff | mean_abs_diff | cosine |
    |---|---|---|---|
    | 0 | 0.0156 | 0.00043 | 0.999984 |
    | 1 | 0.1875 | 0.00929 | 0.997476 |
    | 2 | 0.5000 | 0.03239 | 0.991631 |
    | 3 | 1.8750 | 0.16340 | 0.970483 |
    | 4 | 6.7500 | 0.34670 | 0.974111 |
    | 5(最深) | 20.0000 | 0.84631 | 0.981325 |

    **最浅层已经有差异(0.0156,和 bf16 舍入噪声同量级),逐层单调放大到最深层的 20.0**——
    教科书级别的"微小扰动经过网络深度级联放大"信号,不是随机噪声(随机噪声不会呈现这种
    干净的逐层单调增长)。这就是第 3 步"kernel 内部归约方式因 cache_seqlens 不同而改变"
    这个机制的直接、逐层的数值证据,把"位置吻合的推断"变成了"直接测量到的因果链"。

## 能不能修

**能修的部分(我们自己的代码,`runtime/backends/laguna_cuda_graph.py`/`laguna_dflash_cudagraph.py`)**:
把 `aligned_start` 的取整粒度从"等于 block_size"改成"所有支持的 block_size 的公共倍数"
(目前是 64/128,取 128 即可),让 `aligned_len`/`cache_seqlens` 在两种 block_size 下对齐到
完全相同的值,消除"block_size 变了导致喂给 kernel 的窗口长度不同"这个触发源本身。
**这是纯我们自己代码的改动,不涉及 sparkinfer,可以直接改。**

**但这个修复有一个没有验证过的风险,需要先测,不能直接上生产**:目前 bs=64 用的是它自己
"天然"的 64 粒度对齐(逐 round 最多 63 个 token 的窗口跳变),如果统一改成 128 粒度对齐,
bs=64 的窗口跳变幅度会翻倍到最多 127 个 token——如果第 2 步定位的机制("跳变幅度越大,
越容易触发 kernel 内部归约方式变化 → 临界翻转 → 级联")成立,这个改动**可能会把 bs=64 现在
干净的 0.718182 拖差**,而不是把 bs=128 拉好。也就是说,这个修复能让两个 block_size **互相
一致**,但不保证让它们都变好——需要先跑一次 A/B(bs=64 强制用 128 粒度对齐,看接受率是否
下降)才能判断这条路径是"真正的修复"还是"把问题从 bs=128 转移到 bs=64"。

**还没有直接验证、需要 sparkinfer 团队参与的部分**:第 2 步机制里"更长的 cache_seqlens
即使多出来的部分被 window_left 正确掩码,依然会改变被掩码区域自身的浮点结果"这件事,
本质上是 sparkinfer kernel 内部 split-KV/分块策略的行为,不是我们能直接改的。**决定性的
下一个测试**(还没做,成本低,不需要碰 sparkinfer 代码,只是换一种调用参数):固定
block_size、固定 window_left、固定 KV 内容,只改变喂进去的 `cache_seqlens` 长度(模拟对齐
粒度不同带来的长度差异),直接对比 masked 区域内的 kernel 输出是否变化。如果不变——说明
上面"统一对齐粒度"的修复思路是对的、干净的。如果变了——说明 window_left masking 在这条
sparkinfer 路径下对"被掩码区域是否受 cache_seqlens 长度影响"没有提供数值不变性保证,这是
kernel 行为层面的问题,需要整理证据交给 sparkinfer 团队(不自己改 sparkinfer 代码)。

## 这次排查过程中沉淀下来的工具/资产

- `bfdiag`/`bf` CLI(新合并,`docs/diagnostics-guide.md`/`docs/bfdiag-handoff.md`):`bf shapes
  --diff` 补了 `block_size` 参数(daemon 原生没支持,provider/server/cli 三处 + 3 个 CPU
  单测已修);热引擎 `bf exec` 目前有已知的热/冷 accept-rate gap,还不能用来测这类问题。
- `bfdiag_checks.check_page_table_covers_seqlen` 不变量(`QSR_ASSERT_LEVEL>=1` 时生效),
  接线补丁已应用到 `LagunaCudaGraphVerify._fill_buffers`。
- `QSR_DEBUG_SWA_ALIGN_GRANULARITY`(`laguna_cuda_graph.py`/`laguna_dflash_cudagraph.py`):
  把对齐粒度和物理 block_size 解耦的诊断开关,不设时行为不变。
- `QSR_DEBUG_RING_DUMP_KV_LEN`/`_PATH`(`laguna_dflash_cudagraph.py`):draft ring 原始内容
  落盘,这次改成了 `>=` 阈值 + 一次性 latch(不再要求精确命中 round 边界)。
- `QSR_DEBUG_AUX_DUMP_KV_LEN`/`_PATH`(`laguna_cuda_graph.py`,这次新加):主模型
  `aux_hidden_states` 落盘,同样的阈值+latch 模式。
- `QSR_DEBUG_FORCE_SYNC`(`laguna_dflash.py`):draft/verify replay 关键节点强制
  `torch.cuda.synchronize()`,用于区分 race 和确定性差异,不改变任何计算数值。

## 两个修复前置验证(2026-07-27 追加)

在动手修之前,先测了两件事,确定"统一对齐粒度"这个修复方向是否安全、以及第 2 步机制的
具体载体是不是 kernel 本身。

### 验证 1:bs=64 强制用 128 粒度对齐,会不会把 bs=64 也拖差

`QSR_DEBUG_SWA_ALIGN_GRANULARITY=128`,`block_size=64`(物理页仍是 64,只是对齐取整粒度
强制改成 128,128 是 64 的整数倍,寻址不会破坏正确性),CTX=10240,冷启动,2 轮:

```
Round 0: accept=0.443333
Round 1: accept=0.443333   (确定性复现)
```

**从 bs=64 天然的 0.718182 暴跌到 0.443333,几乎复现了 bs=128 天然的 0.452525。** 这确认了:
degradation 的开关是**对齐粒度**,不是 block_size 本身、不是 FP8 量化边界、不是 kernel 对
不同 page_size 的处理方式(这些都已经在更早的 kernel 隔离测试里逐一排除)。同时这也证明了
"把两个 block_size 统一成同一个对齐粒度"这个修复方向**不安全**——统一到 128 会把 bs=64
现在干净的数字拖到和 bs=128 一样差,不是把 bs=128 拉好,这条修复路径到此排除。

### 验证 2:固定 block_size/window_left/KV 内容,只变 cache_seqlens,masked 区域输出会不会变

`cache_seqlen_sensitivity_test.py`:block_size=128 固定,window_left=511 固定,构造一份
byte-identical 的 FP8 KV 内容(72 Q 头/8 KV 头,gqa=9,真实 SWA 形状),两种 page_table/
cache_seqlens 配置指向**同一份**底层内容——场景 A 是零冗余的理想边界(`cache_seqlens=527`,
`aligned_start` 恰好落在 window_start 上),场景 B 额外多出整整一个物理块(128 token)的
pre-window 冗余(`cache_seqlens=655`),两者都完整覆盖真实窗口,只是 B 多报告了 128 个
理论上会被 window_left 掩码掉的旧 token。直接走 sparkinfer 真实调用路径(和 kernel 隔离
测试 v2 相同的函数),不经过我们自己的任何代码。

```
max_abs_diff=0.000000  cosine similarity: 1.00000000  (16 个 query 逐一核对,全部 cosine>=0.99999988)
```

**逐字节相同。** kernel 对 masked 区域的输出,在 window_left 掩码正确的前提下,**不受
cache_seqlens 报告长度影响**——第 2 步笔记原文提出的"kernel 内部 split-KV/分块策略因
cache_seqlens 长度不同而改变、连带影响 masked 区域数值"这个具体假设,被这个孤立测试
**直接推翻**。

### 两个结果放在一起:机制被进一步限定,但底层数值路径还没找到

验证 1 证明对齐粒度确实是因(效应巨大、干净、确定性可复现,和 block_size 本身无关)。
验证 2 证明这个因不是通过"单次 kernel 调用里 masked 区域对 cache_seqlens 长度数值敏感"
这条路径起作用的——这条路径已经被两次独立、干净的孤立测试(kernel 隔离 v2 + 这次)排除。

真正的载体大概率在**多轮/状态累积**的某个环节,不是单次调用能捕捉到的——验证 2 用的是
一次性构造、静态内容的孤立调用,没有真实生产路径里"这一轮的写入(`reshape_and_cache_flash`
把新 token 的 K/V 写进 ring)和下一轮的读取(下一次 verify replay 的 page_table/
cache_seqlens)之间"的时序/状态依赖。候选方向(还没验证,留给下一步或交给 sparkinfer 团队
参考):`n_ring`(page_table 有效条目数)在对齐粒度变宽后,从一轮到下一轮**变化的频率和
幅度**都会变小但更剧烈(128 粒度下 block 边界跨越更少见,但每次跨越挪动整整 128 个 token,
而不是 64 粒度下更频繁但每次只挪 64 个)——是否是这种"跳变节奏改变"本身触发了写入路径或
CUDA Graph 里其它状态(比如 `update_prefill_graph_replay_metadata` 重算 worklist 的时机)
的某种边界条件,是接下来最值得查的方向。

### 对"能不能修"的更新判断

- **"统一对齐粒度"不是安全的修复方向**(验证 1 已经证伪,会拆东墙补西墙)。
- **不是 sparkinfer kernel 对 cache_seqlens 长度的 masked 区域数值不变性问题**(验证 2 已经
  排除,不需要以此为证据交给 sparkinfer 团队)。
- 剩下的可能载体(写入路径/CUDA Graph 状态刷新时机随跳变节奏变化)还在我们自己的代码里
  (`runtime/backends/laguna_cuda_graph.py`),原则上如果定位到具体机制,是可以自己修的,
  但目前还没有直接证据钉死是哪一处——不建议在没有更多证据前贸然改动这部分代码。
- **实用结论**:block_size=128 迁移在这类高对抗性(高度重复文本、大量临界决策点)负载下,
  接受率有真实的、有机制支撑的劣化风险,幅度可能和这次测出的 27 个百分点同量级;
  block_size=64 仍是接受率更可信的选择,除非/直到这个更深的机制被定位。

**用户决策(2026-07-27)**:生产默认回退/保持 block_size=64,block_size=128 这条线不再
现在投入(理论上限只有 8-10% 吞吐提升——attention 只占 round 时间 11.9%,MoE/dense GEMM
已经没有空间——投入产出比不划算)。block_size=128 的支持代码(guard 放宽、`page_size`
参数化等)本身验证过正确、零回归,**保留不删**,作为以后可以直接启用的能力。另见
`notes/2026-07-27-sparkinfer-generalize-kv-heads-4-to-8-spec.md`——如果 sparkinfer 团队
把 Laguna kernel 特化从 num_kv_heads=4 泛化到 8(我们生产的真实 TP=1 形状),会解锁另一块
独立的性能收益,和这条 block_size=64→128 接受率调查是两件不同的事,值得关注但不是这次的
结论范围。

## 和 vLLM 参考基准(367 tok/s)对比时的提醒

这次 CTX=10240/65536 用的是高度重复的合成文本,这种文本天然会让模型产生大量"非常自信
但和另一个候选极度接近"的临界决策,是投机解码对这类浮点/对齐敏感性最敏感的对抗场景。
真实、多样化的生产文本大概率不会有这么密集的触发点——不能假设这次测出的接受率差距幅度
能代表真实生产场景下 block_size=64 和 128 的实际差距。
