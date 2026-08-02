# Track B0 GPU 收口：B0-3 / B0-4 / B0-5

> 编制日期：2026-08-02 · worktree `work/b0-gpu-20260802`（新建自 `main`）
> 环境：`~/.venvs/vllm/bin/python`（`torch==2.13.0a0+gitcf30153`，
> `transformers==5.8.0`，`fla==0.5.2`）· GPU：`NVIDIA RTX PRO 6000 Blackwell
> Max-Q`，`torch.cuda.get_device_capability()==(12,0)`（SM120），
> `shared_memory_per_multiprocessor=102400` B。全程用 `/tmp/gpu_lock.sh`
> 独占，探针跑完立即 `release`。
>
> **只读约束遵守情况**：`/home/bot/project/sparkinfer` 全程零写入——只调用其
> 公开 API（`sparkinfer.attention.paged.{plan,bind,run}`），未改一行源码。
> `oracle/` 全程未 `import`、未读取（B0-4/B0-5 用的参照实现是**已安装的 pip
> 包** `transformers`（`transformers/models/qwen3_5/modeling_qwen3_5.py`，
> `pip show` 可查，非 oracle）与 `fla`，不违反"只能抄读 oracle、不能
> import"的约束，因为压根没碰 oracle）。
>
> 三条探针脚本随本次提交进本 worktree（`scripts/b0_probe_*.py`），可直接
> 重跑复现下面每一个数字。

---

## 结论摘要

| 条目 | 结论 | 确信度 |
|---|---|---|
| **B0-3** | **能跑，且正确**：sparkinfer 的 paged attention 对 `head_dim=256/gqa_group=6(24Q/4KV)/page_size∈{64,128}/fp8 KV` **没有硬门**（无论是 Python 侧 `select_paged_forward_traits` 还是底层 CUTLASS kernel 派发条件），decode 与 extend(prefill) 两种模式均实测跑通，精度对齐 fp32 参照（cosine≥0.99999）。**但**：这个形状组合在 sparkinfer 自己的测试套件里**从未被测过**（所有 `head_dim=256` 测试用的是 gqa=8 或 gqa=128，所有 `num_q_heads=24/num_kv_heads=4` 测试用的是 head_dim=128），走的是通用 FlashInfer 风格兜底路径而非任何"Laguna 专属"调优内核；首次调用（JIT/autotune）耗时 **decode 62-64 秒、extend 27 秒**，warmup 若不覆盖这个具体形状会在生产首个真实请求上原地卡一分钟 | 硬：GPU 实测通过/失败 + 数值对比 |
| **B0-4** | **① FLA v0.5.2 拿到手就是正确的**：`chunk_gated_delta_rule`/`fused_recurrent_gated_delta_rule` 在 Qwen3.6 真实 GDN 形状（16 K 头/48 V 头，头维 128/128，repeat_interleave×3）下与 HF transformers 自带的 torch 参照实现（`torch_chunk_gated_delta_rule`/`torch_recurrent_gated_delta_rule`，就在 `modeling_qwen3_5.py` 里）逐层对比，cosine≥0.99998，误差量级 1e-3~5e-3（浮点噪声量级，非 bug）。**追加事实**：HF 官方 `Qwen3_5GatedDeltaNet` 本身在 `fla` 可用时就是调 `fla.ops.gated_delta_rule` 这两个函数——即"HF 参照实现"与"① FLA 路径"在生产配置下**是同一份代码**，不是两个独立实现凑巧一致 | 硬：GPU 实测数值对比 |
| **B0-5** | **能，capture-safe，且已给出可直接照抄的模式**：`torch.cuda.graph()` 捕获"读持久 state buffer → `fused_recurrent_gated_delta_rule` → `state_buf.copy_(new_state)`"整段单步 decode，重放 6 步（每步换真实输入）与逐步 eager 参照**逐 bit 一致**（`max_abs_err=0`）。前提条件（HF 自己的 `transformers/cache_utils.py::LinearAttentionLayer` 已经这么做）：状态 buffer 只分配一次、`torch._dynamo.mark_static_address` 标记、**永远 `.copy_()` 写入，永远不重新绑定 Python 引用**。唯一真实的操作要求（不是新风险，是把 oracle 已经点破的坑坐实）：状态递归**非幂等**，warmup 迭代会污染 state buffer，必须在真正捕获前、真正服务前手动 `.zero_()` 一次——这是图外的普通 eager 操作，不需要额外设计 | 硬：GPU 实测 capture+replay 数值对比 |

**对 B1 的直接影响**：三条都是正面结论，B1 没有被这三条挡住。head_dim=256 全注意力可以直接用 sparkinfer 的通用 decode/extend 路径（不是"Laguna 专属"那一路,不要照抄 Laguna 那些 `head_dim_qk==128` 硬编码分支），GDN 用 FLA ①（不需要现在就做③自研），B2 的 CUDA Graph 可以按"持久 buffer + copy_ 写回 + 槶位重置时手动清零"的模式设计,不再是"未知能不能做"的风险项。

---

## B0-3 · sparkinfer paged attention：head_dim=256 / gqa=6 / page_size∈{64,128} / fp8 KV

### 结论先行

**没有发现任何硬门会拒绝这个形状。** 逐层读了三层可能拒绝的地方，均放行：

1. **Python 侧形状校验**（`sparkinfer/attention/paged/traits.py:116-126`
   `select_paged_forward_traits`）：只要求 `head_dim_qk % 16 == 0`、
   `head_dim_vo % 16 == 0`、`kv_dtype` 是 fp16/bf16/fp8_e4m3、fp8 KV 时
   `q_dtype` 必须 bf16。`head_dim=256` 满足 `256 % 16 == 0`，没有上界。
2. **decode 内核派发条件**（`forward_paged.py:3319-3333`
   `PagedForwardKernel.__init__`）：`fp8_plane_decode_dims_supported` 要求
   `head_dim_qk % 64==0 且 >=128 且 <=256`、`head_dim_vo % 128==0 且 <=256`——
   `256` 恰好卡在这个区间的**上界**（不是被排除，是刚好被允许到头）。TMA
   分支的"或"条件里，除了两个 `head_dim_qk==128` 的 Laguna 专属分支外，还有
   一个**不限定 head_dim** 的通用分支（`num_warps_kv>1 and num_warps_q==1
   and cta_tile_q==16 and stage_tile_rows==64 and num_mma_kv==1`）——decode
   永远走 `cta_tile_q=16`（`planner.py::_paged_determine_cta_tile_q`），且
   `traits.py:265-269` 对 `cta_tile_q==16` 硬编码只枚举 `num_mma_kv=1`，
   恰好落进这个通用分支。
3. **extend(prefill) 内核**（`forward_extend_generic.py:3006-3021`）：存在一个
   显式的 `traits.head_dim_qk==256 and traits.head_dim_vo==256` TMA 分支（这
   是 sparkinfer 为另一张卡 GB10/SM121、`num_q_heads=8/num_kv_heads=1` 的场景
   写的——`planner.py` 里 `_sm121_gqa8_decode_chunk_budget` 显式 `assert
   device_capability==(12,1)`——但这个 TMA 判断本身**没有**绑定
   `gqa_group_size` 或 `device_capability`，所以对我们的 `gqa=6`/SM120 同样
   命中）。测试套件里 `tests/attention/test_attention_paged_planner.py` 和
   `test_attention_paged_decode_split_graph.py` 里所有 `head_dim=256` 的用例
   全部是 `num_q_heads=8,kv_heads=1`（gqa=8）或 `num_q_heads=128,kv_heads=1`
   （gqa=128），所有 `num_q_heads=24,num_kv_heads=4` 的用例全部是
   `head_dim_qk=128`——**这个具体组合（256+24/4）在 sparkinfer 自己的测试里
   一次没出现过**，这也是为什么本轮必须真机验证，不能只读代码下结论。

### 实测（`scripts/b0_probe_paged_attention_head256.py`，可重跑复现）

用 sparkinfer 自己的 `paged.plan/bind/run` 公开生命周期（与
`tests/attention/test_paged.py::_run_eager` 同一调用序列），形状固定为
`num_q_heads=24, num_kv_heads=4, head_dim_qk=head_dim_vo=256, q_dtype=bf16,
kv_dtype=fp8_e4m3fn`：

| 场景 | page_size | 首次调用耗时(JIT/autotune) | 正确性(vs fp32参照) | 稳态吞吐(含 plan/bind 重建, eager) |
|---|---:|---:|---|---:|
| decode（batch=4, cache_seqlens=[512,1024,2048,4096]） | 64 | 63.7 s | max_abs_err=0.00024, cosine=0.9999915 | 3.58 ms/call |
| decode（同上） | 128 | 62.6 s | max_abs_err=0.00024, cosine=0.9999923 | 2.79 ms/call |
| extend/prefill（batch=2, 768 q-rows, cache_seqlens=[512,256]） | 64 | 27.4 s | max_abs_err=0.00391, cosine=0.9999923 | 3.36 ms/call |
| extend/prefill（同上） | 128 | 27.2 s | max_abs_err=0.00391, cosine=0.9999926 | 3.35 ms/call |

误差量级（2.4e-4~3.9e-3 绝对误差、cosine 5 个 9）与 sparkinfer 自己
`test_run_decode_fp8_kv_matches_reference` 的容差（≤0.06 绝对误差）相比是
**远优于**它自己认定"通过"的门槛,说明不是"勉强能跑、精度打折"的情况。

**吞吐数字的诚实说明**：上表的"稳态吞吐"是完整
`plan→scratch分配→bind→run` 的 eager 端到端耗时,包含 Python 侧重建 plan
的开销,**不是**纯 kernel-only benchmark 数字（B1 阶段"eager 无图"下这个
数字有代表性；B2 上 CUDA Graph 后,plan 只建一次,重放只剩纯 kernel 时间,
届时数字会低很多,不应把这里的数字当成 B2 的性能基线）。

**JIT 耗时是本轮最值得写进风险登记的数字**：decode 62-64 秒、extend 27 秒
的首次编译时间,直接印证了 `implementation-plan.md` §7.3 C7-3 已经点名的
担忧——"warmup/autotune 是否用生产真实形状"。如果 B1/B2 的
`warmup_paged_attention_shapes()` 不显式覆盖 `head_dim=256/gqa=6` 这个形状,
生产环境第一个真实全注意力请求会原地卡接近一分钟。**这不是本轮新发现的
风险,是把 C7-3 已经写下的担忧,用这个具体形状实测坐实成一个具体数字。**

**未验证/超出本轮范围的点**：
- decode 侧的 CUDA-Graph 专属加速路径（`_is_laguna_fp8_gqa6_analytic_decode_graph`
  等,`planner.py:307-391`）全部硬编码 `head_dim_qk==128`,head_dim=256 摸不到
  这些"Laguna 专属"调优内核,只能走通用 decode graph 容量规划
  （`plan_decode_graph_capacity`,该函数本身**没有** head_dim 上限,
  `_heuristic_decode_graph_ctas_per_sm`/`_decode_graph_heuristic_max_chunks_per_req`
  都有显式 `head_dim_qk>=256 and head_dim_vo>=256 and gqa_group_size<=8` 分支）——
  但本轮只测了 `use_cuda_graph=False` 的 eager 路径,**没有**实测
  `use_cuda_graph=True` 下 head_dim=256 decode 能否真正捕获/重放。这条留给
  B2 落地时验证,不是本轮 B0-3 范围（B0-3 原文只问"正确性与吞吐",没问
  CUDA Graph——CUDA Graph 问题在 B0-5,但 B0-5 原文明确限定在 GDN 递归状态,
  不含全注意力）。
- 只测了两个 batch/cache_seqlens 组合,没有扫描全部 page_size×batch×context
  长度矩阵（这是 B3 性能调优的范围,不是 B0 事实基线的范围）。
- 没有测 `window_left` 相关的滑窗注意力路径（Qwen3.6 的 full-attention 层
  是否用滑窗尚待 B1 读 config 确认;若不用,这条不适用）。

---

## B0-4 · GDN 方案三选一：① FLA v0.5.2 正确性

### 关键追加事实（比"选哪个方案"更根本）

`transformers/models/qwen3_5/modeling_qwen3_5.py:60-63,205-206,409-410`：

```python
if is_flash_linear_attention_available():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
else:
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
...
self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule
```

即：**HF 官方 `Qwen3_5GatedDeltaNet` 在 `fla` 可用时,本身就是调用 FLA 的这
两个函数**;`torch_chunk_gated_delta_rule`/`torch_recurrent_gated_delta_rule`
只是"没装 fla 时的兜底"。这意味着"① FLA"与"跟 HF 参照实现比对"**不是两件
独立的事**——本机 `~/.venvs/vllm` 里 `fla` 是真装的（`fla.__version__ ==
0.5.2`),所以 HF 参照实现在这台机器上跑起来,走的就是 FLA 路径。本轮的
对比测的是"FLA 路径 vs torch 兜底路径"两者互相吻合到什么程度,而不是
"FLA vs 一个完全独立的参照"。

### 实测（`scripts/b0_probe_gdn_correctness.py`,可重跑复现）

形状取 Qwen3.6 真实 GDN 参数（`config.json` text_config,B0 静态读码已确认）：
`linear_num_key_heads=16, linear_num_value_heads=48,
linear_key_head_dim=linear_value_head_dim=128`,K/Q 按 `repeat_interleave×3`
扩到 48 头（对齐 `modeling_qwen3_5.py:505-507`）。

| 场景 | core_attn_out 误差 | final_state 误差 |
|---|---|---|
| `chunk_gated_delta_rule`,prefill T=300（非 64 的整数倍,触发 padding 分支） | max_abs=9.8e-4, cosine=0.999985 | max_abs=5.0e-3, cosine=0.999994（均为 fp32） |
| `fused_recurrent_gated_delta_rule`,8 步连续 decode,从上一行的 final_state 续接 | 每步 max_abs≈2.4e-4, cosine≈0.99999 | 每步 max_abs≈3e-3~5e-3, cosine≈0.99999 |

误差量级在 8 步递归里**没有随步数增长发散**（第 0 步与第 7 步误差同一
量级),说明数值稳定,不是"越递归越飘"的坑。

**跨步 BF16 舍入的代价**（呼应记忆里"单步 FP32 计算+跨步 BF16 舍入"这条
已知事实,本轮用真实数值实测了代价而不是停在"机制上确实如此"）：把持久
state buffer 按 `transformers/cache_utils.py::LinearAttentionLayer` 的真实
做法降到 BF16（`self.recurrent_states.copy_(fp32结果)`,buffer dtype 是
BF16),8 步下来 `core_attn_out` 相对"全程 FP32 干净递归"的额外误差只有
**6.1e-5~1.2e-4**——比 FLA-vs-HF-参照本身的误差(2.4e-4)还小一个量级。
**结论**：跨步 BF16 舍入不是精度担忧的主因,B1 如果要跟 HF 逐 token 对齐,
复刻这个"FP32 算完就近舍入进 BF16 buffer"的动作代价很低,值得做（对齐
成本低,不对齐的话这一项误差会累加进"未解释的偏差"里,徒增排查难度）。

### 吞吐（用于后续判断是否需要③自研的参考数字,非正式 benchmark）

`fused_recurrent_gated_delta_rule` 单步 decode（真实 48 头/128 头维形状）：
batch=1 时 **0.035 ms/step**,batch=8 时 **0.039 ms/step**——48 层合计约
**1.6~1.9 ms**,相对 sparkinfer 全注意力 decode 一次调用的 2.8~3.6 ms
(B0-3 数字,16 层里只有部分是全注意力),量级相当,不是明显瓶颈。
`chunk_gated_delta_rule` prefill（batch=1, seq=4096,首次调用有 ~238 ms
triton autotune 开销,稳态后)约 **0.79 ms/call**,48 层合计约 **37.8 ms**
——对一次 4096 token 的 prefill 而言占比很小。**这些数字支持计划里"先用①
拿正确性,profiling 说话后再决定③"的判断**：① 本身看起来已经够快,没有
现在就投入③自研的理由;真正决定要不要③,需要等 B1 端到端跑起来后,把 GDN
耗时占比放进整层 forward 的 profile 里看,而不是孤立看 GDN 算子本身的数字
——这条留给 B3,本轮不代为拍板。

---

## B0-5 · GDN 递归状态更新是否 CUDA Graph capture-safe

### 前置证据：HF 自己的 Cache 实现已经是"怎么做才 capture-safe"的现成答案

`transformers/cache_utils.py:768-837`（`LinearAttentionLayer`）：

```python
def lazy_initialization(self, conv_states=None, recurrent_states=None):
    if recurrent_states is not None:
        self.recurrent_states = torch.zeros_like(recurrent_states, dtype=self.dtype, device=self.device)
        if not is_torchdynamo_compiling():
            torch._dynamo.mark_static_address(self.recurrent_states)   # <- 关键
        self.is_recurrent_states_initialized = True

def update_recurrent_state(self, recurrent_states, **kwargs):
    if not self.is_recurrent_states_initialized:
        self.lazy_initialization(recurrent_states=recurrent_states)
    # Note that we copy instead of assigning, to preserve the static address for cudagraphs
    self.recurrent_states.copy_(recurrent_states)          # <- 关键：copy_ 不是重新绑定
    return self.recurrent_states
```

FLA 的 kernel 本身**不做原地更新**——`fused_recurrent_gated_delta_rule_fwd`
每次调用都 `q.new_empty(N, HV, K, V, dtype=torch.float32)` 分配一个全新
tensor 作为返回值（`fused_recurrent.py:211-213`）。真正让整条链路
capture-safe 的,是调用方这一层的纪律：**buffer 只分配一次、标记
`mark_static_address`、永远 `.copy_()` 写回、永远不重新绑定 Python 引用**。
这是 HF 自己的机制,不是本仓库要发明的东西。

### 实测（`scripts/b0_probe_gdn_cudagraph_capture.py`,可重跑复现）

原样复刻上面这个模式（不经过完整 HF 模型,直接对 FLA 函数做,排除模型其余
部分的干扰）：分配一次 state buffer、`mark_static_address`、在
`torch.cuda.graph()` 内部执行"读 state → `fused_recurrent_gated_delta_rule`
→ `state_buf.copy_(new_state)`",捕获成功后拿同一个 graph 重放 6 次,每次
重放前把新的真实输入 `.copy_()` 进静态输入 buffer（这是 CUDA Graph decode
重放的标准写法,不是这里发明的特例)。

```
RESULT: torch.cuda.graph() CAPTURE SUCCEEDED
[step 0] replay vs eager: max_abs_err=0 cosine=0.99999994
[step 1] replay vs eager: max_abs_err=0 cosine=0.99999994
[step 2] replay vs eager: max_abs_err=0 cosine=1.00000000
[step 3] replay vs eager: max_abs_err=0 cosine=0.99999994
[step 4] replay vs eager: max_abs_err=0 cosine=0.99999988
[step 5] replay vs eager: max_abs_err=0 cosine=1.00000000
final recurrent_state (post replay) vs eager final_state: max_abs_err=0
```

**逐 bit 一致,不是"大致对得上"**——6 步重放跟 6 步 eager 参照的输出和最终
state 完全相等（`max_abs_err=0`）,说明状态确实在跨 replay 正确传递,不是
"图捕获成功但读的是 stale 数据"这种更隐蔽的错误。

**调试过程本身留一条方法论记录**：第一版探针出现"replay 从第 1 步开始
偏离 eager"的假阳性,根因是探针自己的 bug——`A_log`（GDN 的门控参数,真实
模型里是固定的 `nn.Parameter`）被我在每次生成输入时用未接种子的
`torch.empty(...).uniform_()` 重新随机了一遍,导致两条路径消耗全局 RNG
的次数不同、两边的"模型参数"实际上不是同一份。**这不是 CUDA Graph 的
bug,是测试脚本没有把"模型参数"和"逐步变化的输入"分开处理的 bug**——已在
`scripts/b0_probe_gdn_cudagraph_capture.py` 里改成 A_log/dt_bias 只生成一次、
探针启动时固定,复现前先说明这一点,避免后续有人重跑时被同样的假阳性坑到。

### ⚠️ 关于"不要照搬 Laguna 的 warmup 复用安全论证"（`qwen36-rebuild-spec.md:135`）

原文点破的问题：Laguna 的 decode 图从不碰递归状态,它的"warmup 可以安全
重复调用"论证建立在"重复调用不改变任何持久状态"这个前提上——GDN 的递归
状态**恰恰会被每次调用改变**（非幂等),这个前提对 GDN 不成立,不能直接搬。

本轮探针**在设计阶段就撞上了这个问题的另一面,并给出了具体解法**：探针的
warmup 阶段（3 次迭代,用于触发 triton autotune/首次分配)确实把
`state_buf` 写脏了(非零),如果直接拿这个脏 state 去捕获、然后拿去服务真实
的第一个请求,结果就是错的（第一步会读到 warmup 留下的垃圾状态,不是干净
的零状态)。**解法是一行 `state_buf.zero_()`,在 warmup 之后、捕获之前,
作为一次普通 eager 操作,不需要进图**——之所以这样就够了,是因为
`torch.cuda.graph()` 捕获的是**固定的 kernel launch 序列和固定的内存地址**,
不是"捕获时刻 buffer 里的值"（这个 kernel 也没有任何依赖 host 端可见数值
的分支逻辑,不会把某个具体值"烤"进图里)——所以捕获前 buffer 里是脏数据还
是零,不影响捕获本身能不能成功,只影响**第一次真实重放前**这个 buffer 该是
什么值。

**这条转成 B2 的一个具体、廉价的实现要求**（不是新风险,是把 oracle 那条
注释坐实成一个已验证可行的具体做法）：新槶位分配到 GDN 状态 buffer 时,
必须显式 `.zero_()`（或写入"空槶位"该有的初值),且这个操作在图外、在该
槶位第一次真实重放之前执行一次即可,不需要每次重放都做,也不需要任何
比"一次 eager `.zero_()`"更复杂的机制。这与 roadmap 里"递归状态纳入槶位
生命周期"这条本来就要做的事完全对得上,不是新增工作项。

**未验证/超出本轮范围**：
- 只测了单层 GDN 的单个 CUDA Graph,没有测"48 层 GDN + 16 层全注意力交织
  在同一个 decode 图里"的完整规模（真实场景是把整个 decode step 的所有
  64 层一起捕获进一张图,不是分层捕获)。单层验证过 capture-safe 不代表
  64 层拼一起没有其他坑（比如显存布局、kernel 数量对 graph 大小的影响),
  但至少排除了"GDN 状态更新这个机制本身有原则性问题"这个最大的不确定性。
- 没有测多槶位 batch 内、槶位生命周期跨越多次 capture/reset 的场景（比如
  某个槶位被驱逐后新请求复用同一个物理槶位,状态要不要清零、清零时机)——
  这属于 B2 服务化阶段的槶位生命周期设计,不是"capture-safe 与否"这个
  B0-5 本身要回答的问题。
- `chunk_gated_delta_rule`（prefill 路径,涉及 varlen/padding,Python 侧有
  形状相关分支)本轮**没有**测试 CUDA Graph 捕获——这符合预期,因为 decode
  CUDA Graph 只需要 `fused_recurrent_gated_delta_rule`（`seq_len==1` 路径),
  prefill 本身不在 B2 的"decode 图"范围内(prefill 是否要图是另一个问题,
  不在 B0-5 原文范围)。

---

## 复现清单

```bash
cd /home/bot/project/qsr-w-b0gpu
/tmp/gpu_lock.sh acquire <你的名字> "复现 B0-3/4/5 探针"
find . -name __pycache__ -type d -not -path './.venv/*' -exec rm -rf {} +
~/.venvs/vllm/bin/python scripts/b0_probe_paged_attention_head256.py
~/.venvs/vllm/bin/python scripts/b0_probe_gdn_correctness.py
~/.venvs/vllm/bin/python scripts/b0_probe_gdn_cudagraph_capture.py
/tmp/gpu_lock.sh release <你的名字>
```
