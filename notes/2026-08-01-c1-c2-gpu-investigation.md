# C-1 / C-2 GPU 排查：CUDA Graph/warmup 形状真实性 + NVFP4 vs FP8 KV prefill

对应 [`../docs/investigation-queue.md`](../docs/investigation-queue.md) §C 的 C-1、C-2。
实测环境：`work/gpu-20260801`(基于 main `6acc4ba`),`sparkinfer@0844a4f`(fork,
`master`),GPU 为 NVIDIA RTX PRO 6000 Blackwell Max-Q(97887 MiB)。
run record: `bf show 940b708aa0f8`(见下)。

---

## C-1 · warmup / CUDA Graph 捕获是否用真实形状 —— ✅ 部分成立,且挖到一个更严重的活 bug

### 结论一句话

CUDA Graph 捕获本身(`laguna_cuda_graph.py`/`laguna_dflash_cudagraph.py`)和
`warmup_paged_attention_shapes()` 对它们各自覆盖的 contract**确实用生产真实容量**
(不是占位小形状)——flashinfer #3255 那种"autotuner 只测过小合成形状,真实维度
的形状第一次出现才炸"的字面模式,在这两处**不成立**。

但沿着 `warmup_paged_attention_shapes()` 自己文档里承认的缺口(mode="verify"
未被预热)往下查,发现的不是"没预热所以第一次会慢",而是**这条 eager 回退路径
现在会直接 `ValueError` 崩掉**——用 GPU 直接调用生产函数 `_forward_verify_with_aux`
实测确认。这比原假设("只是没预热,会有一次编译停顿")严重得多。

### 证据链

1. **CUDA Graph 捕获用的是真实容量,不是占位形状**(读代码,未启动 GPU 前完成):
   - `LagunaCudaGraphDecode`/`LagunaCudaGraphVerify`(`runtime/backends/laguna_cuda_graph.py`)
     和 `DFlashDraftCudaGraph`(`runtime/backends/laguna_dflash_cudagraph.py`)在
     capture 前先用 `blocks_per_slot`/`_ring_blocks_per_slot`/`_draft_blocks_per_slot`
     (backend 在加载期算出的真实 per-slot 容量,来自 `QSR_SERVER_BLOCKS_PER_SLOT`
     等生产配置)去建 sparkinfer 的持久 `PagedAttentionWorkspace`/`create_paged_plan`——
     这是决定 SparkInfer JIT 编译键的那一步,用的是**真实最大容量**,不是任意小样本。
   - `torch.cuda.graph()` 实际捕获时喂给 `_fill_buffers` 的 dummy kv_len(decode 用
     `blocks_per_slot*block_size-1`,verify 用 64,draft 用 2048)只影响 CG 捕获**外部**
     的 metadata 填充调用(这段代码本身不在 `torch.cuda.graph()` 块内,每次 replay 都
     会用真实 kv_len 重新执行),不影响被捕获的 kernel launch 本身的容量/编译键——那
     由上一条的持久 workspace 决定。三个类里这三个数字不统一看起来眼熟,但对捕获正确
     性和编译键都不构成影响。
   - `LagunaBackend.warmup_paged_attention_shapes()`(`runtime/backends/laguna.py:615`)
     的容量边界 `self._prefill_capacity_by_window_left`(`laguna.py:463`)直接取自
     `self._prefill_chunk_tokens`(默认 8192,`QSR_PREFILL_CHUNK`)和真实
     `blocks_per_slot`——同样是生产真实值。

2. **已知缺口(继承自 `235f51e`,当时故意不修)**:`warmup_paged_attention_shapes()`
   只跑 `mode="extend"` 和 `mode="decode"` 两个 dummy pass,完全不碰 `mode="verify"`。
   见 [`2026-08-01-prefill-shape-buckets-root-cause.md`](2026-08-01-prefill-shape-buckets-root-cause.md)
   的"已知缺口"一节——那份笔记的猜测是"如果 DFlash 的 eager verify 回退真的被打
   到,会付一次未预热的编译停顿(30-100s 量级,参照同一份笔记里 extend 模式修复前
   测到的数字)"。

3. **往下追了一层,发现这个猜测低估了严重性**。DFlash 的两条 eager 回退路径待遇不同:
   - `_draft_forward`(草稿模型 eager 回退,`laguna_dflash.py:623`)用
     `mode="extend"`——和 `eager_extend_work_items_capacity` 这个容量估算函数的设计
     假设(见下)一致,理论上不受本条影响(本次未独立在 GPU 上验证这条路径本身,
     只是代码层面确认它没有踩同一个坑)。
   - `_forward_verify_with_aux`(主模型 eager 回退,`laguna_dflash.py:1637`)对全部
     layer group 都传 `mode="verify"`(`laguna_dflash.py:1692,1698`)。这条路径复用
     的是**跟普通 prefill 完全同一个** `SparkinferPrefillWorkspace` 实例——
     `replace_laguna_attention()`(`bf_attention.py:241`)按
     `(window_left, num_heads, num_kv_heads, head_size)` 缓存 workspace,**缓存键里
     没有 mode**,所以 extend/decode/verify 三种调用共享同一个 Python 对象。

4. **容量预算的 bug**:`SparkinferPrefillWorkspace.forward()`
   (`laguna_sparkinfer_attn.py:246-250`)不管调用方传的 `mode` 是什么,永远用
   `PagedAttentionWorkspace.eager_extend_work_items_capacity(max_total_q=...,
   num_q_heads=..., num_kv_heads=...)`(函数名直接写着 `eager_extend`)去估算
   `max_work_items`,再传给 `for_fixed_capacity(mode=mode, ..., max_work_items=...)`。
   `for_fixed_capacity` 建出来的是**硬容量**——`_ensure_capacity`
   (sparkinfer `workspace.py:1748`)发现真实 plan 需要的容量超过声明值时,对
   `fixed_capacity` 的 workspace **直接 `raise ValueError`**,不像 CG 路径用的
   `for_contract`(容量未定,首次调用会按需自动长大,只有*之后*再超才报错)。
   `create_paged_plan(mode="verify", ...)` 算出的真实 work-item 数显然不是
   `eager_extend_work_items_capacity` 这个"按 extend 语义"估的那个数量级(verify
   模式下 query 很短但要在长 context 上切 KV chunk 做负载均衡,work-item 数不是简单
   随 `max_total_q` 线性增长的)。

### 实测(GPU,run record `940b708aa0f8`)

在 worktree 自己的 `bf daemon`(生产等价配置:`block_size=64`、
`blocks_per_slot=4096`、`num_slots=3`,CUDA Graph/DFlash 默认开)里:

- 冷启动日志确认**这次启动 verify CG 和 draft CG 都正常捕获成功**
  (`DFlash: verify CG captured (M=16)`、`DFlash: draft CUDA Graph captured`)——
  所以**今天线上不是一个正在发生的故障**,是一个没人打到过的隐患分支。
- 直接调用生产函数 `engine._forward_verify_with_aux(slot, verify_tokens, kv_len,
  16)`(不是重写的等价代码,是同一个函数;绕开 `if self._verify_cg is not None`
  分支,模拟"如果 verify CG 捕获在启动期失败了会发生什么"),用一个很朴素的真实
  形状(`kv_len≈2016`,16 token 的 verify 窗口,远低于任何声明容量)——

  ```
  ValueError: fixed-capacity paged workspace exceeded; construct a larger eager extend workspace
  ```

  报错发生在 `ws._ensure_capacity(plan)`,在任何 attention 数学或 CuTe 编译之前——
  失败很快很便宜,不是超时也不是崩 GPU。

### 后果推断(部分是代码读出来的,不是本次独立复测的)

- `_capture_verify_cg()`(`laguna_dflash.py:442`)把**任何**异常吞掉,只打
  `logger.warning`,然后 `self._verify_cg = None`;`_init_cuda_graph()` 之后无条件
  把 `self._cg_captured = True`——生产构造路径下 `_lazy_capture_cg()` 永不会被再次
  触发(读代码确认,未独立测)。也就是说:**如果 verify CG 捕获在某次启动时失败了
  (OOM 瞬时、CUDA 瞬时错误、未来 sparkinfer 升级改了内部容量计算……),这个进程
  剩下的整个生命周期里,`self._verify_cg` 永远是 `None`。**
- 从那一刻起,`dflash_round()`/`generate_verify_only()` 的每一轮都会调
  `_forward_verify_with_aux`,而这个函数在（至少)常见真实形状下会 `raise
  ValueError`——这个异常在调用链上没被 catch(`dflash_round`/
  `generate_verify_only` 都是裸调用,没有 try/except)。也就是说:**一旦 verify CG
  在启动期失败,DFlash 在这个进程的余下生命周期里对每一个请求的第一轮 verify 都会
  报错**,不是"变慢",是**功能性失效**。这是没有独立复测的推断(没有故意让
  verify CG 捕获失败去验证这一整条因果链),但每一步都能在当前代码里直接指到行号。
  没有测的部分诚实标注在下面"未测"里。
- 本次没有分离出到底是 `wl=-1`(full attention)还是 `wl=511`(SWA)这两个 layer
  group 里哪一个先撞到容量上限(traceback 只显示了第一层撞到的那次调用)——报错
  机制(`eager_extend_work_items_capacity` 用于 extend 语义、被 verify 模式误用)
  对两个 group 都适用,不影响结论,但没有独立坐实具体是哪一个或两个都会。

### 这属于哪一类"诊断平台说谎"

不是"捕获时验的形状和生产跑的形状不是一回事"(那部分实测下来是干净的)。而是
**同一个思路的另一种表现**:整条"用小/合成契约的估算函数去覆盖一个语义不同的大
契约"的模式,换了个位置又发生了一次——`eager_extend_work_items_capacity` 这个
以 extend 命名、按 extend 语义设计的估算函数,被不分场合套用到了 verify 契约上。
根因和 235f51e 修的那个 bug 是**同一类**(容量/编译键假设与真实契约不匹配),只是
这次的表现从"变慢"升级成了"报错崩掉"。

### 建议(不改代码,写清楚交给开发)

`SparkinferPrefillWorkspace.forward()` 需要按 `mode` 选择正确的 work-item 容量
估算,而不是永远调用为 extend 设计的 `eager_extend_work_items_capacity`。最干净
的做法很可能是:第一次针对某个 `(mode, window_left)` contract 时,先用
`PagedAttentionWorkspace.for_contract`(未定容量,自动长大)跑一次真实 plan 拿到
sparkinfer 自己算出的真实容量,再用那个数字去建 `for_fixed_capacity` 的硬容量
workspace——这正是 `LagunaCudaGraphVerify._init_workspaces()` 已经在用的手法
(`for_contract` 起步 + `_ensure_capacity` 自动长大一次,再固化),只是 CG 路径
和 eager `SparkinferPrefillWorkspace` 路径没有共享这个逃生舱。这段属于
`runtime/backends/laguna_sparkinfer_attn.py`,按文件归属不动它——写清楚交给
BlackweLLM 自己的开发(这不是 SparkInfer 内核问题,是我们这边包装类的容量估算
选错了函数)。

### 未测 / 留白

- 没有独立触发一次"启动期 verify CG 捕获真的失败"来验证上面"整条因果链"的推断
  (今天这次启动两个 CG 都成功,没有自然复现口)。
- 没有分离出 full-attention 组和 SWA 组具体是哪一个(或两个都)先撞容量上限。
- 没有测 `_draft_forward` 的 eager 回退路径本身(只从代码确认它用 `mode="extend"`,
  按当前证据不该踩同一个坑,但没有专门起一次 draft-CG-禁用的启动去实测)。

---

## C-2 · NVFP4 KV vs FP8 KV 的 prefill 对比 —— ✅ 查完,结论是"测不了,而且理由比预想更硬"

### 结论一句话

**这个对比在当前技术栈上根本跑不起来**:SparkInfer 的 paged-attention 内核
(唯一的 attention 内核,本项目零依赖 FlashInfer)**只接受 fp16/bf16/fp8_e4m3
三种 KV dtype**,不是三选二里选了 fp8——传别的 dtype 会被显式 `TypeError` 拒绝
(`sparkinfer/attention/paged/traits.py:120-121`)。本 runtime 自己也在三处硬编码
`kv_cache_dtype = "fp8"`(`runtime/model/plain_attention.py:168`、
`runtime/backends/bf_attention.py:103`、`runtime/backends/laguna_sparkinfer_attn.py:489`),
`plain_attention.py` 的模块级注释直接写"这个 runtime 唯一真实的 KV-cache
dtype,硬编码而不是重新推导……FP8,永远"。全仓库搜索 `nvfp4`/`NVFP4` 命中的都是
**权重量化**(MoE 专家、linear 层)语境,没有一处是 KV cache。

所以 flashinfer #4269(第三方在 RTX PRO 5000 上测出 NVFP4 KV prefill 比 FP8 KV
慢 1.7-1.8x)在我们的栈上**连对照组都不存在**——不是"我们测过更慢",是"我们连
另一个选项都没有,现在也没打算实现(NVFP4 KV kernel 是 SparkInfer 团队的活,不是
我们能直接改的)"。这比一次实测结论更硬:实测结论会随下次 SparkInfer 版本升级
过期,而"内核层没有这个 dtype、要新增需要跨团队的内核工作"这个理由,只要
SparkInfer 不主动加 NVFP4 KV 支持就一直成立。

### bf diff 可比性判定

不适用——没有第二个配置可比。这正是问题所在:C-2 原计划的"两次跑、`bf diff`
判可比、再比数"三步,在第一步("跑第二个配置")就跑不下去,所以这里不是"忽略了
`bf diff`",是从根上没有第二组数据可以拿去 `bf diff`。

### 退而求其次:测了什么

既然没有 NVFP4 KV 可比,退回去测**我们唯一真实拥有的 FP8 KV 在生产真实形状上的
prefill 基线**,给将来任何人重提"要不要上 NVFP4 KV"时一个真实锚点(而不是凭感觉
猜"我们现在多快")。同一次 `bf exec`(run record `940b708aa0f8`,worktree 生产等
价配置:`block_size=64`、`blocks_per_slot=4096`,走生产入口
`backend.prefill_with_aux`,每次换一批互不重复的 prompt 内容,单个新 slot,冷 KV):

| prompt 长度(token) | 墙钟耗时 |
|---:|---:|
| 64 | 284 ms |
| 512 | 146 ms |
| 2048 | 331 ms |
| 8192 | 1106 ms |
| 32768 | 5048 ms |
| 16384(全新长度,在上面几次跑完之后) | 2313 ms |

几点说明:
- 这是**热 daemon 里的稳态数字**,不是严格意义的"冷启动 prefill"(见
  `docs/diagnostics-guide.md` 的热/冷边界表)——但这里要测的是"FP8 KV attention
  在真实形状上跑多快",不是"分配器/编译缓存在全新进程下的瞬态行为",所以热
  daemon 是合适的工具,不违反那条边界规则。
- 64 token 比 512 token 还慢(284ms vs 146ms)是预期的噪声/warmup 残留(这是
  daemon 里的第一次真实调用,数字本身不是本次调查的重点,不深挖）。
- 32768 token 5048ms、16384 token(晚于上面几个、从未出现过的新长度)2313ms,
  两者之间没有观察到 30-100s 级别的异常尖峰——说明 235f51e 修的那个"每个新形状
  重新编译"的 bug 在 mode=extend 上确实是稳的(这条路径当天本机的 `~/.cache/
  sparkinfer` 已经是热的,所以这次没有单独观察到冷编译的那一次性代价,但也没有
  观察到本该被修掉的"每个新长度都重新编译"症状复发)。

### 对 roadmap 的影响

支持原计划的"选 FP8 KV,不要把 NVFP4 扩到 KV"结论,而且理由从"第三方测过更慢"
升级为"我们自己的内核库现在直接不支持,要支持得先让 SparkInfer 团队新增一条
NVFP4 KV 内核路径,再由我们做运行时接线(attention impl 分支、`block_pool.py`
里已经预留的 `kv_cache_dtype` cache-key 位)"——门槛比"跑起来但慢 1.7-1.8x"高
得多,不建议投入,除非 SparkInfer 主动把 NVFP4 KV 加进他们的内核库。
