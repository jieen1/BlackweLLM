# bfprobe P2a:T1 归约签名层——实现记录

日期:2026-07-27
状态:CPU 可验证部分已完成并全绿;GPU 部分(Triton kernel 本体)已写出但**从未运行**。
方法:零 GPU 操作。所有代码在 `/home/bot/project/qwen-sm120-runtime/.venv`(CPU-only 可用
的 torch 2.11 + triton 3.6 + numpy 2.3 + ruff 0.15)下验证;从未 import 任何会触发 CUDA
初始化的路径,从未创建 CUDA tensor,从未跑 `nvidia-smi`。

设计依据:`notes/2026-07-27-probe-system-design-and-plan.md` §4(探针清单与成本预算)、
§5(分级探针 + 预触发冻结)。阈值一致性依据:`bfdiag/divergence/thresholds.py`(位于并行
agent 的 worktree `.claude/worktrees/agent-a0e22f11934dd2f59/bfdiag/`——本仓库这个
worktree 尚未包含 `bfdiag/`,故本包**不导入** `bfdiag`,只是复用它的深度放宽公式,详见下文
"阈值策略"一节)。

---

## 1. 交付清单

| 文件 | 内容 | 状态 |
|---|---|---|
| `bfprobe/__init__.py` | 与其他并行 agent 约定的逐字节内容 | 完成 |
| `bfprobe/_bus_stub.py` | `bus.py` 落地前的本地 stub | 完成 |
| `bfprobe/reduce.py` | `Signature`、CPU 参考实现、Triton kernel、形状/索引纯函数 | 完成(kernel 未跑) |
| `bfprobe/signature.py` | T1 签名环(host 后端)、JSON dump/load、零开销 `emit()` | 完成 |
| `bfprobe/baseline.py` | 基线记录/持久化、深度放宽阈值、纯函数判定器 `judge()` | 完成 |
| `bfprobe/scan.py` | 离线扫描、`site_id` 常量、文本/JSON 报告渲染 | 完成 |
| `bfprobe/scan_cli.py` | `bf probe scan` 子命令(`register`/`main`) | 完成 |
| `tests/test_bfprobe_reduce.py` | 31 项,CPU 归约对拍 + 形状/索引单测 | 全绿 |
| `tests/test_bfprobe_signature.py` | 17 项,环行为 + 零开销 timeit | 全绿 |
| `tests/test_bfprobe_baseline.py` | 23 项,阈值模型 + 判定器 | 全绿 |
| `tests/test_bfprobe_scan.py` | 14 项,越界定位(含 layer 31 注入) | 全绿 |

`python3 -m pytest tests/test_bfprobe_*.py -q` → **85 passed**。
`ruff check bfprobe/ tests/test_bfprobe_*.py`(line-length 100,select E,F,I,UP)→ **全绿**。
运行整个仓库的 `pytest -q` 确认:pre-existing 的 35 个失败全部是 `ModuleNotFoundError: No
module named 'vllm'`(这个 worktree 的 venv 没装 vllm)及其连带的迁移台账断言,与
`bfprobe` 无关、不是本任务引入的问题。

---

## 2. 归约设计与"一个 kernel"的论证

### 2.1 澄清"一个 kernel"到底指什么

原始设计文档写:「一轮有 48 层 × 4 抽头 = 192 次归约。如果每个统计量单独 launch,就是
192×5 = 960 次」。这句话的约束粒度是**每次归约**(即每个张量的一次签名计算),不是"整
个 round 只能有一次 kernel launch"。所以 `bfprobe/reduce.py` 的设计目标是:

> 对**一个**张量,`absmax`/`l2`/`mean`/`nan_count`/`inf_count` 五个量必须在**一次** kernel
> launch 里全部算出来,而不是拆成 5 次 `torch.xxx()` 各自触发一次 launch。

192 次归约 = 192 次 launch(仍然是 192,而不是 960),这与"P1 T0 已经在 verify 之后有一次
同步点"以及"6 KB/轮可忽略"的成本预算完全吻合。

### 2.2 为什么 NaN/Inf 不从 absmax/l2/mean 里"过滤"出去

`absmax`/`l2`/`mean` 按**全部元素**朴素计算,不做 NaN/Inf 掩码。含 NaN 时,IEEE-754 传播
规则会让这三个值本身也变成 NaN(在 CPU 参考实现和 Triton kernel 里都验证/编写了这一行
为)。这是刻意的:`bfprobe.baseline.judge()` 一旦看到 `nan_count>0` 或 `inf_count>0` 就直接
判定越界并返回,根本不会再看这三个被污染的值。如果反过来做"先掩码再算三个统计量",单个
kernel 内至少要多一次 per-element 分支(仍在一次 launch 内,不违反"一个 kernel"的硬约
束,但白白增加复杂度换不来任何判定精度上的好处)——所以选择不掩码。

`tests/test_bfprobe_reduce.py::TestReduceReferenceCorrectness` 里专门验证了这个"NaN/Inf 污
染但计数仍然正确"的行为(`test_nan_is_counted_and_poisons_reductions` 等)。

### 2.3 一个意外但值得记录的边界情况:fp32 累加器溢出

`test_fp32_accumulator_overflow_produces_inf_without_inf_count` 记录了一个真实边界:两个
`1e30` 的元素本身都不是 Inf(`inf_count == 0`),但计算 L2 时 `1e30² = 1e60` 超过 float32
的 ~3.4e38 上限,`sum_sq` 溢出为 `inf`,于是 `l2` 变成 `inf`——**即使没有任何一个元素是
Inf**。这是 CPU 参考实现和(设想中的)Triton kernel 都会呈现的真实行为(两者都在 float32
里累加,与仓库里 `runtime/triton_norm_ops.py`/`runtime/kernels/fused_rms_norm.py` 现有 kernel
的累加精度约定一致)。工程含义:极端量级张量(理论上限 sqrt(3.4e38) ≈ 1.8e19)会让 `l2`
本身变成一个可疑信号,即使 `inf_count==0`——这本身也是有用信息(说明这个张量数值已经在
爆炸的边缘),不需要额外处理。已记录于 `reduce.py` 该测试的说明里,以及本节。

### 2.4 GPU kernel 设计:单 program、单 launch、串行分块

`_signature_reduce_kernel`(`bfprobe/reduce.py`)用 `grid=(1,)`——**恰好一个** program 实
例,内部对被打平的张量做 `for off in range(0, numel, BLOCK_SIZE)` 串行分块循环,五个累加
器(`sum`/`sum_sq`/`absmax`/`nan_count`/`inf_count`)全程留在寄存器里,循环结束后一次性
写出。这个写法直接照抄了仓库里已经在跑的单 pass 风格
(`runtime/triton_norm_ops.py::_rms_norm_triton_kernel`、
`runtime/kernels/fused_rms_norm.py::_fused_add_rms_norm_kernel` 都是"一个 program,内部循
环,寄存器里累加,循环完再写"的模式)。

选择 `grid=(1,)`(而不是多 program + atomic 归约)的理由:
- **不需要跨 program 同步**:只有一个 program,天然没有"其他 program 有没有写完"的问
  题,`mean`/`l2` 的最终推导(见下）不需要额外一次 launch。
- 本包的目标张量都是**decode 单轮单 tap** 的量级(数千到数万元素:router logits
  16×256=4096,hidden state 16×3072=49152,topk 16×10=160),单 program 串行分块在这个规
  模下完全够用,不需要多 SM 并行。
- **已知限制,写入 GPU 验证待办**:这个设计**不适合 prefill 全量隐状态**(64K token ×
  3072 = 2 亿元素级)——单 program 只用一个 SM,吞吐远低于峰值,量级到了这里会明显变
  慢。设计文档 §4 本来就把 prefill 的 T1/T2 策略分开处理("prefill 只能做 T0 + T1 签
  名 + 稀疏 token 采样"),所以这不是本阶段的阻塞项,但如果未来要给 prefill 大张量也上
  T1 签名,需要一个多 program + `tl.atomic_add`/`tl.atomic_max` 的变体——**这是一条明确
  的 GPU 验证待办**(见 §6)。

### 2.5 GPU 结果的"原始累加器"与"落地 Signature"分离

`reduce_gpu()` 返回的是 GPU 常驻的 5 元素原始累加器张量 `[sum, sum_sq, absmax, nan_count,
inf_count]`,**不做** `.item()`/`.cpu()`/`torch.cuda.synchronize()`。`mean = sum/numel`、
`l2 = sqrt(sum_sq)` 这两步推导被拆到 `finalize_signature()`——一个纯函数,只在**dump 时**
（离开热路径之后）才被调用,此时才第一次把标量读回宿主端。这样"归约结果留在 GPU 上,写
进签名环,dump 时才读回"这条硬性要求在接口层面就是不可违反的:`reduce_gpu` 的返回值类型
就是一个 GPU tensor,`finalize_signature` 的输入类型就是纯 Python float 元组,两者之间没
有第三条"顺便读一下"的路径。

---

## 3. 阈值策略:与 `bfdiag/divergence/thresholds.py` 的一致性

### 3.1 为什么是"复用同一套增长模型"而不是"import 那个模块"

`bfdiag/divergence/thresholds.py` 目前只存在于另一个并行 agent 的 worktree
(`.claude/worktrees/agent-a0e22f11934dd2f59/bfdiag/`),**本 worktree 里没有 `bfdiag/`**。
如果 `bfprobe/baseline.py` 直接 `import bfdiag...`,这个包在当前 worktree 里会直接
`ModuleNotFoundError`,单测全灭。所以本模块**不做代码级依赖**,而是把同一套深度放宽公式
原样重新实现了一遍(见 `bfprobe/baseline.py` 的 `_depth_growth`),常数也刻意保持一致:

```python
_MAX_GROWTH = 3.0                  # 与 thresholds.py 完全相同
_DEPTH_GROWTH_COEFFICIENT = 0.3    # 与 thresholds.py 完全相同
growth(layer) = min(_MAX_GROWTH, 1.0 + _DEPTH_GROWTH_COEFFICIENT * sqrt(max(layer, 0)))
```

依据与 `thresholds.py` 一致:独立的逐层舍入误差按随机游走累积,标准差随层数的平方根增
长;增长设上限,避免深层阈值松到能藏住真 bug。

### 3.2 两边阈值"形状相同、量纲不同"

`thresholds.py` 判的是**跨实现**(oracle vs 引擎)的余弦相似度/top-k agreement 等下界型
指标,用 `_relax_lower_bound` 把 `(1 - base)` 的"允许误差预算"按 `growth` 放大。`bfprobe`
判的是**跨时间**(基线 vs 当前)的 `absmax`/`l2` 上界/偏差型指标,做法结构相同但方向不同:

```python
absmax_ratio_bound(layer) = 1.0 + (_BASE_ABSMAX_HEADROOM - 1.0) * growth(layer)
l2_rel_dev_bound(layer)   = _BASE_L2_REL_DEV * growth(layer)
```

`_BASE_ABSMAX_HEADROOM = 1.5`(layer 0 允许 50% 冗余,深层封顶到 `1 + 0.5*3.0 = 2.5x`),
`_BASE_L2_REL_DEV = 0.5`(layer 0 允许 50% 相对偏差,深层封顶到 150%)。

**与 `thresholds.py` 的本质区别、也是必须说明的局限**:`thresholds.py` 的每一个 layer-0
floor 都标注了具体的历史实测依据(见其文档字符串:RMSNorm 近乎逐位相同、attention
cos=0.999999、NVFP4 MoE 基线 0.95-0.97 等)。**本模块的 `_BASE_ABSMAX_HEADROOM`/
`_BASE_L2_REL_DEV` 目前没有对应的实测依据**——因为"基线 vs 重复跑同一个引擎"这件事本身
还没有在真实 GPU 上做过(需要先有一次真实的 DFlash 运行录制基线,再跑第二次同 prompt 的
运行,测量 absmax/L2 的自然波动范围)。这两个常数目前是从直觉论证给出的保守初值,**必须
列入 GPU 验证待办**(见 §6),校准方式应该和 `thresholds.py` 一样:用真实历史数字定出
layer-0 floor,而不是猜。

### 3.3 nan_count/inf_count 优先级

`judge()` 里 NaN/Inf 检查在最前面、无条件短路返回,不受深度影响——这与设计文档 §6 的
"nan_count>0 或 inf_count>0(任何情况都是致命)"要求逐字对应,也是
`tests/test_bfprobe_baseline.py::TestJudge::test_nan_is_always_fatal_regardless_of_depth`/
`test_inf_is_always_fatal_regardless_of_depth` 直接断言的行为。

---

## 4. `site_id` 分配规则(200-299 号段)

**关键设计决策**:`site_id` 标识的是**抽头种类**(tap kind),**不含层号**。层号已经是
`SignatureRecord.layer` 的独立字段,不需要、也不应该编码进 `site_id`——如果每层每种抽头
都分配一个独立 `site_id`(如任务描述里举例的「`200 + layer_idx` 之类的规则」字面理解),
48 层 × 4 抽头 = 192 个组合,会直接超出 200-299 这 100 个 id 的预算。把 `site_id` 限定为
"种类"、层号单独存放,192 个组合只需要 4 个 `site_id`,给未来新增抽头种类(逐 head
attention 输出、GDN 状态、embedding 层输出等)留出 96 个备用 id。

当前分配(定义在 `bfprobe/scan.py`):

| `site_id` | 常量 | 抽头 | 命名依据 |
|---|---|---|---|
| 200 | `SITE_INPUT_LAYERNORM` | 每层 `input_layernorm` 输出(进入 attention 前) | 与 `bfdiag/divergence/thresholds.py` 的 `INPUT_LAYERNORM` 同名 |
| 201 | `SITE_ATTN_OUT` | 每层 `self_attn` 输出 | 同 `ATTN_OUT` |
| 202 | `SITE_POST_ATTENTION_LAYERNORM` | 每层 `post_attention_layernorm` 输出(进入 MLP/MoE 前) | 同 `POST_ATTENTION_LAYERNORM` |
| 203 | `SITE_MOE_OUT` | 每层 MLP/MoE 输出(该层最终隐状态,喂给下一层) | 同 `MOE_OUT` |
| 204-299 | 保留 | 未来抽头(逐 head attention、GDN 状态、embedding 层等) | — |

选择与 `thresholds.py` 的 kind 常量同名,是为了将来 `bfdiag/divergence` 需要"先用 T1 签
名做便宜的初筛,再对可疑层做完整对拍"(设计文档附录里提到的关系)时,site 名字和
threshold kind 名字是同一个字符串,不需要一张额外的映射表。

300-399 号段属于 MoE 路由 agent,本包完全没有触碰。

---

## 5. T1 签名环设计

### 5.1 数据形状

`SignatureRecord`:`(seq, site_id, round_idx, layer, absmax, l2, mean, nan_count,
inf_count, numel)`——比任务描述的 9 元组多一个 `seq` 字段。这是刻意的增强,不是另搞一
套:9 元组本身不含任何可以用来做"丢弃检测"的东西,而"必须靠单调 seq 号可检测"是任务的硬
性要求,所以 `seq` 是这个要求在数据结构层面唯一自然的落点。

### 5.2 环形语义

- **存储后端可替换**:`SignatureRingBackend`(`typing.Protocol`)只定义 `write`/`read`
  两个方法 + `capacity` 属性。今天唯一的实现是 `HostSignatureRingBackend`(numpy 列存,
  9 个定长数组,`record()` 对每一列做原地标量赋值——不分配新对象)。将来 P3 的 GPU 常驻
  后端只需要实现同样两个方法,`SignatureRing` 本身不用改一行。
- **零分配热路径**:`SignatureRing.record()` 只接受裸标量参数(不接受 `Signature` 对
  象),避免每次调用构造一个 dataclass 实例;内部只做取模、原地赋值、整数自增,没有
  f-string、没有 dict、没有临时容器。
- **写满覆盖最旧 + 丢弃可见**:`_next_seq` 单调递增、永不重置;`idx = seq % capacity`;
  `read_all()` 用 `dropped = max(0, total_written - capacity)` 精确算出被覆盖的行数,
  绝不假装它们从未存在过。`tests/test_bfprobe_signature.py::TestSignatureRingWrapAndDrop`
  覆盖了"容量以内不丢"、"恰好等于容量不丢"、"超过容量精确计数"、"seq 单调无空洞"四种
  情况。
- **读回顺序是时间序**:`read_all()` 从 `total_written - valid` 开始按 `seq` 升序读,天
  然是"最老到最新",无需额外排序。

### 5.3 默认容量

`DEFAULT_CAPACITY = 200_000` 行(numpy 9 列 int64/float64,约 14.4 MB),在 192
条/轮的负载下可回溯 1000+ 轮——相比设计文档里 T2(256 MiB/轮量级)的显存预算,T1 环的
体量小到可以忽略,不需要像 T2 那样做"分配前检查空闲显存"的硬门禁(那是 P3 的问题)。

### 5.4 与 `bus.py` 的耦合方式

`bus.py` 在任何 worktree 里都还不存在。`signature.py` 按任务要求做
`try: from bfprobe.bus import PROBE_ENABLED, TIER_SIGNATURE except ImportError: from
bfprobe._bus_stub import ...`。**只导入这两个名字**,没有导入 `emit_signature`——因为
`bus.emit_signature` 未来的实现大概率是"调用方→bus.emit_signature→(判断
PROBE_ENABLED)→归约→写入某个 ring",即 bus 是这个包的**调用者**,而不是被这个包调用的
对象;本包反过来导入 `bus.emit_signature` 会造成方向不明的循环依赖。为了让"disabled 时
只有一次 if 判断、GPU-off 时 0 分配 0 tensor 触碰"这条验收要求在**没有 bus.py** 的情况下
也能测,`signature.py` 自己暴露了一个结构相同的入口 `emit(ring, site_id, round_idx, layer,
tensor)`,同样只做"读一次 `PROBE_ENABLED`,不满足就直接 return"。**合并时如果 `bus.py`
的真实设计是反过来（本包对外暴露一个注册函数供 bus 调用），这里需要相应调整**——已经在
`_bus_stub.py`/`signature.py` 的 docstring 里写清楚了这个假设,方便合并时核对。

`PROBE_ENABLED=False` 时 `emit()` 单次调用实测约 **33 ns**(`timeit`,500000 次取平均),
测试断言的门槛是 100 ns,有 3 倍余量。

---

## 6. 需要 GPU 才能验证的待办清单

以下事项**没有 GPU 就做不出实测数据**,均已在代码/本文档里标注,列在这里统一追踪:

1. **`_signature_reduce_kernel` 从未执行**——需要真实 GPU 验证:数值上和
   `reduce_reference` 是否逐位/近似一致(bf16/fp16 输入、各种 `BLOCK_SIZE`);实际 launch
   开销是否真的可以忽略(设计预算是 192 次/轮、~0% of 44.16ms)。
2. **prefill 规模的多 program + atomic 归约变体尚未设计/实现**——当前 `grid=(1,)` 单
   program 版本只覆盆 decode 量级(数千到数万元素);如果决定给 prefill 也上 T1 签名,需
   要一版新 kernel,且需要真实大张量实测吞吐。
3. **`_BASE_ABSMAX_HEADROOM`/`_BASE_L2_REL_DEV` 未经真实数据校准**——需要:①录制一次真
   实 DFlash 运行的逐层基线;②同 prompt 重复跑,测量 `absmax`/`l2` 的自然轮间/运行间波
   动范围;③用这个实测波动范围重新标定 layer-0 floor(参照 `thresholds.py` 的校准方
   法),而不是沿用本文档给出的直觉初值。
4. **T1 探针挂到 `laguna.py`/`laguna_dflash.py` 的真实 forward hook 之后的零假设检
   查**——本任务只交付了归约核心 + 环 + 判定逻辑,**没有**把探针实际接到生产 forward
   path(那是 P1/P2 集成阶段的工作,不在本任务文件清单内:`runtime/`、
   `bfprobe/{cli,routing,routing_compare,report}.py` 明确不归本任务修改)。接入后需要跑
   一次"探针全关 vs T1 全开"的零假设检查(输出 token 逐位相同、延迟 p50 偏差 < 0.5%),
   对应设计文档 §7 P2 阶段的"GPU 验证批次 #2 ①"。
5. **`SignatureRingBackend` 的 GPU 常驻实现(P3)未开始**——今天的 `HostSignatureRingBackend`
   全部是宿主内存;接口已经按"可替换"设计好,但 GPU 显存版本本身、以及"图内 staging →
   图外环形轮转"的双缓冲时序需要真实 CUDA Graph 环境才能验证(依赖设计文档 §1(G)的机
   制,与 `replay_with_aux` 类似)。

---

## 7. 已知偏离/取舍(供合并时核对)

- `SignatureRecord` 比任务给的 9 元组多一个 `seq` 字段(§5.1 已说明理由)。
- `judge()`/`OutOfBandVerdict` 的字段名、`Baseline`/`BaselineEntry` 的具体结构是本任务自
  行设计(任务只要求"纯函数判定器",没有给定具体接口),如果 P2b(oracle 侧路由探针)或
  其他消费者已经约定了不同的形状,合并时需要对齐。
- `bfprobe/scan.py` 内联了报告渲染(`to_json_dict`/`format_text_report`),因为任务文件清
  单明确禁止本任务创建 `bfprobe/report.py`(该文件属于另一个 agent)。
- 本任务没有创建 `bfprobe/cli.py`(按要求不碰),`scan_cli.py` 目前只能通过
  `python -m bfprobe.scan_cli` 独立运行;等 `bfprobe/cli.py` 落地后,顶层 dispatcher 需要
  `import bfprobe.scan_cli; bfprobe.scan_cli.register(subparsers)` 才能接进 `bf probe scan`。
