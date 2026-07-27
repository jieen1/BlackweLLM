# bfdiag 方案5:oracle 逐层对拍一键化 + 激活值缓存(2026-07-27)

## 动机(不重复背景,只记要点)

`notes/2026-07-27-acceptance-rate-gap-vllm-vs-ours-same-prompt.md`(注:该笔记不在本
worktree 分支快照里——本分支是从更早的 commit 切出的,那篇笔记是并行开发中另一条线之后
才落地的,本任务的动机来自协调者转述其内容,以下按转述内容处理)记录的核心发现:同一个
prompt、同样 64K 上下文/K=15/greedy,vLLM 原生 DFlash 接受率 100%,我们引擎只有 68.7%。
笔记明确给出方法论:「需要用逐层 logits 对比(类似历史上验证 BFAttention "cos=0.999999"
那次的方法论)去定位具体哪一层/哪个算子先开始分叉」,但当时没有把这套方法论工具化。

本任务(方案5)把这套方法论变成 `bf divergence` 一键命令 + 一个 oracle 激活值缓存,目标是把
"跑两边模型 + 手工比对" 从几个小时压到一条命令,精度 bug 定位从 10 次迭代压到 1 次(缓存
命中后只需重跑我们自己引擎那一侧)。

## 交付范围与优先级执行情况

按任务给定的优先级 (a) scan+thresholds+核心验收测试 → (b) report/CLI → (c) cache →
(d) capture 真实路径,四项全部完成:

- `bfdiag/divergence/scan.py` —— 纯函数扫描器,`tests/test_bfdiag_divergence.py` 8 个用例全绿。
- `bfdiag/divergence/thresholds.py` —— 复合阈值 + 深度放宽,`tests/test_bfdiag_thresholds.py` 全绿。
- `bfdiag/divergence/report.py` + `bfdiag/divergence/cli.py` —— 文本/JSON 报告 + `bf divergence`
  子命令(`register(subparsers)`),CLI 自身可独立 `python -m bfdiag.divergence.cli --help` 自测。
- `bfdiag/divergence/cache.py` —— 三层缓存(统计量/采样token/全量),`tests/test_bfdiag_oracle_cache.py` 全绿。
- `bfdiag/divergence/capture.py` —— 复用 `oracle/capture_hooks.py::ForwardCapture`;
  `FakeCaptureSource` 已在测试里跑通完整链路;真实 GPU 采集路径(`capture_engine_activations`、
  `EngineCaptureSource`)写完但**从未执行**(见文末 GPU 待办清单)。
- `oracle/comparator.py` —— 仅加法式扩展:新增 `top_k_agreements`、`activation_rms`、
  `LayerComparison`、`compare_activation`,原有 `compare_values`/`ComparisonResult`/`_as_values`/
  `_top_indices` 一字未改,`tests/test_comparator.py` 原有两个用例保持全绿。

全部命令验证方式见文末「怎么验证」一节。

## 1. 扫描算法(`scan.py`)

### 数据契约

```
ActivationTrace = Mapping[int, Mapping[str, Any]]   # layer_idx -> {submodule_name: tensor_like}
```

`tensor_like` 复用 `oracle/comparator.py::_as_values` 的鸭子类型:list、tuple、numpy 数组、
CPU 上的 torch 张量都可以,`scan_layers` 本身不 import torch,不碰 GPU,不碰模型——这是任务
"扫描必须是纯函数" 这条硬性要求的直接实现方式,也是它能在 CPU 上完整单测的原因。

### 算法

对 `oracle`/`candidate` 共有的每个 `layer_idx`(升序),再对该层里两侧共有的每个
`submodule` 名字:

1. 用 `oracle.comparator.compare_activation(reference, candidate, top_k=10)` 得到
   `ComparisonResult`(cos / max_abs_error / mean_abs_error / top-10 agreement)+ 固定的
   top-1、top-5 rank agreement。
2. 用 `oracle.comparator.activation_rms(reference)` 把 `max_abs_error` 归一化成
   `rel_max_abs_error`(见第2节为什么必须是相对误差而不是绝对误差)。
3. 用 `thresholds.threshold_for(submodule_name, layer_idx)` 取该 (层, 子模块) 的复合阈值。
4. 四项判据全部满足才算 `passed`:`cos >= min_cosine`、`rel_max_abs_error <= max_rel_abs_error`、
   `top1_agreement >= min_top1_agreement`、`top5_agreement >= min_top5_agreement`。

`DivergenceReport.first_divergent_layer` = 第一个存在 ≥1 个子模块判定失败的层(按层号升序找
第一个);`first_divergent_submodules` = 该层里失败最严重(cos 最低优先)的子模块名列表——这就
是"自动定位第一个越过阈值的层,并在该层内继续下钻到子模块"的实现。

两侧只在某一层/某个子模块上存在(capture 配置不同导致的)会被静默跳过,不计入失败——因为
oracle 缓存和引擎实时采集完全可能挂了不同的子模块集合(见第3节缓存设计)。

### 核心验收测试(`tests/test_bfdiag_divergence.py`)

构造 42 层合成激活序列,每层 5 个子模块
(`input_layernorm`/`self_attn`/`post_attention_layernorm`/`mlp`/`mlp.gate`),每个子模块的
"真值"向量是 `10*(i+1) + sin(seed + 0.37*i)`(dim=64)——刻意让分量幅值按 10 的间隔严格拉开
（而不是用一个各分量幅值接近的向量),这样"自然漂移"量级的噪声不会碰巧把 top-1/top-5 的排名
打乱(第一版用 `sin(seed+0.37*i)+2.0` 当真值时,分量幅值太接近,1e-4 量级的噪声就能让
top-1 命中率降到 0——这是个真实的度量特性,不是 bug:top-1/top-5 一致性对"分量幅值扎堆"的
向量天然脆弱,提醒了我们真实 hidden_state 若出现大量近似分量,该判据的信噪比也会变差,值得
在 GPU 验证阶段留意)。

用例:

1. **`test_injected_bias_at_layer_17_is_the_first_divergent_layer`**:仅在 layer 17 的
   `mlp` 上把候选向量整体取负(cos=-1.0,决定性发散),其余层/子模块只有随深度增长的
   微小自然漂移。断言 `first_divergent_layer == 17`、`"mlp" in first_divergent_submodules`、
   layer 17 同层其余子模块仍然 `passed`、17 层之前全部 `passed`。
2. **`test_injected_bias_at_a_different_submodule_is_still_pinpointed`**:同样手法换成
   layer 5 的 `mlp.gate`,证明下钻不是写死指向 `mlp`。
3. **`test_no_bias_at_all_means_no_divergence`**:候选与 oracle 逐层逐子模块完全相等
   （深拷贝),断言 `has_divergence is False`。
4. **`test_deep_layer_natural_drift_does_not_false_positive`**:不注入任何 bug,42 层全部
   只有随 `sqrt(layer_idx)` 增长的微小自然漂移,断言全程不发散,包括最深的 layer 41
   （并断言该层 cos < 1.0,证明确有噪声、不是和上一条测试退化成同一个用例)。
5. 另加 4 个边界测试(缺层跳过、空 trace 报错、层号不交报错、`scan_prompt` 端到端走通
   `FakeCaptureSource`)。

全部 8 个用例见 `tests/test_bfdiag_divergence.py`,`python -m pytest -q
tests/test_bfdiag_divergence.py` 全绿。

## 2. 阈值策略论证(`thresholds.py`)

### 为什么不能用单一 cos 阈值

固定阈值在这个 42~48 层的模型上必然两难:定得紧,深层天然的 fp8/bf16 舍入顺序差异会疯狂
误报;定得松,浅层的真实 bug(例如 cos=0.997)反而混进"看起来还行"的区间。必须复合判据
（cos + 相对误差 + top1/top5)且逐层放宽。

### 真实历史数字(证据来自本仓库 notes/,不是编的)

| 场景 | cos | 来源 |
|---|---|---|
| sparkinfer attention kernel vs SDPA(fp8-dequant) | 0.999999 | `notes/STATUS_dflash_acceptance.md:25`、`notes/STATUS_bf_attention_integration.md:165` |
| BFAttention 模块输出(layer 0,1)vs SDPA bf16 参考 | 0.9996 | `notes/STATUS_bf_attention_integration.md:167` |
| block_size=128 迁移后,全注意力/SWA 的 extend/decode | 0.999991~0.999993 | `notes/2026-07-27-laguna-real-shapes-correction-and-page-size-migration-plan.md:55-58` |
| fused_add_rms_norm(混合 dtype) | 0.99999624 | `notes/2026-07-21-kernel-comprehensive-review.md:296` |
| NVFP4 量化 MoE 单层 vs bf16 真值(ALPHA 路径,正确配置) | 0.9612~0.9714 | `notes/2026-07-24-phase1-ground-truth.md`(实验1、实验2) |
| 同一 MoE 层,÷2.5 combine bug(错误配置) | 0.9448(rel_norm 从 0.9712 掉到 0.8735) | `notes/2026-07-24-phase1-ground-truth.md`(实验2) |
| MoE 47 层复合(kernel 配置错误,非本任务默认场景) | 单层最高 0.94 → 端到端 ≈0.06 | `notes/2026-07-24-sparkinfer-integration-diagnosis.md` |
| 全模型 hidden-state cosine,历史上被判定为"真实 bug,不是可接受漂移" | 0.996 | `notes/2026-07-17-post-ragged-round-next-steps.md:1380-1406,1487` |

这张表给出两个关键论据:
1. **RMSNorm/attention kernel 这类确定性强的算子,正确时几乎 bit-exact(0.999999+)**——
   浅层阈值可以、也应该定得很紧。
2. **NVFP4 量化 MoE 天然有 ~0.95~0.97 的"正确基线"**——如果给 MoE 用和 attention 一样紧
   的阈值,每一层都会误报。但 0.996 这个"全模型"级别的数字曾经被认定为真实 bug——说明即使
   是"深层/全模型"级别的比较,也不能松到 0.996 都放行,阈值放宽必须有上限(见下)。

### 复合判据 + 深度放宽模型

`LayerThreshold(min_cosine, max_rel_abs_error, min_top1_agreement, min_top5_agreement)`。

逐子模块的 layer-0 基线(`_BASE_THRESHOLDS`,数值即来自上表):

| kind | min_cosine | max_rel_abs_error | min_top1 | min_top5 | 依据 |
|---|---|---|---|---|---|
| `input_layernorm` / `post_attention_layernorm` | 0.999999 | 0.001 | 1.0 | 1.0 | RMSNorm 融合核 bit-exact 级别数字 |
| `self_attn`(attn_out) | 0.9999 | 0.01 | 0.98 | 0.95 | attention kernel 0.999999,模块级(含 softmax/长上下文累加)留一档余量 |
| `mlp.gate`(router_logits) | 0.999 | 0.05 | 0.99 | 0.95 | 路由 logits 维度小、一点误差就可能翻转 top-k 专家,取接近 attention 的紧阈值 |
| `mlp`(moe_out) | 0.95 | 0.35 | 0.85 | 0.75 | NVFP4 量化 MoE 单层正确基线 0.96~0.97,留余量到 0.95 |
| `hidden_state`(整层输出/兜底) | 0.999 | 0.02 | 0.98 | 0.95 | 参照仓库里反复出现的 "cosine>=0.99 + top1" 全栈验收线,再收紧一档 |

放宽模型:独立同分布的逐层舍入误差按随机游走累积——方差可加,标准差按 `sqrt(层数)`
增长,而不是线性增长。所以允许误差预算按

```
growth(layer_idx) = min(3.0, 1 + 0.3 * sqrt(layer_idx))
min_cosine(layer)         = 1 - (1 - base.min_cosine) * growth
max_rel_abs_error(layer)  = min(0.75, base.max_rel_abs_error * growth)
min_top1/5(layer)         = max(floor, base.min_top1/5 - 不放宽预算式同上,下限 0.4/0.3)
```

放宽,`growth` 上限 3.0(不是拍脑袋:Laguna-S-2.1 共 48 层,`0.3*sqrt(47)+1 ≈ 3.06`,取
3.0 意味着到最深层误差预算最多放宽到 3 倍,不会无限放松——呼应上面"0.996 曾被判定为真实
bug"这条证据:不能让深层阈值松到连历史上认定的真实 bug 都放行)。`threshold_for` 实测数值
（`python -m bfdiag.divergence.thresholds`)节选:

| layer | self_attn min_cos | mlp min_cos | mlp.gate min_cos |
|---|---|---|---|
| 0 | 0.9999 | 0.95 | 0.999 |
| 5 | 0.99983 | 0.9165 | 0.99833 |
| 17 | 0.99978 | 0.8882 | 0.99776 |
| 47(封顶) | 0.9997 | 0.85 | 0.997 |

注意:MoE 在任何深度的阈值都不会松过 attention(`test_moe_out_never_relaxes_tighter_than_attn_out`
显式断言这一点)——即使两者各自的放宽比例相同,MoE 的 layer-0 基线本来就更松,这是真实量化
噪声决定的,不是本模块的锅。

`QSR_DIVERGENCE_THRESHOLD` 环境变量可以整体覆盖 `min_cosine`(绕开深度模型),用于人工临时
收紧/放松,不改代码。

### 已知局限(诚实说明,不是回避)

- `growth` 的系数 0.3、封顶 3.0、`_TOP1_FLOOR`/`_TOP5_FLOOR` 是"用真实历史数字论证方向、
  但没有真实 64K 逐层数据拟合曲线"的产物——这是本任务在没有 GPU 的约束下能做到的最好程度。
  GPU 待办清单里第一条就是拿一次真实的 42~48 层逐层对拍数据回来标定这几个常数。
- `hidden_state` 这个 kind 目前没有真实模块可以直接挂钩(vLLM 的 `Qwen3_5DecoderLayer`
  没有单独输出"整层结果"的子模块,整层输出就是下一层的输入),是为了"没有更细粒度信号时的
  兜底判据"预留的 kind,真实使用时预期由 capture 侧把 `model.layers.{i+1}.input_layernorm`
  的**输入**(而不是输出)重新导出成 `hidden_state`,这部分还没有实现,记入 GPU 待办清单。

## 3. 缓存策略与体积估算表(`cache.py`)—— 本任务最重要的设计取舍

### 问题规模

Laguna-S-2.1:`hidden_size=3072`、`num_experts=256`、48 层(`layer 0` 为稠密 MLP,
`layer 1-47` 为 MoE,证据:`runtime/backends/laguna_sparkinfer_moe.py:96-100`)。一次扫描
默认挂 5 类子模块(`input_layernorm`/`post_attention_layernorm`/`self_attn`/`mlp` 4 个
hidden_size=3072 的张量 + `mlp.gate` 1 个 num_experts=256 的张量,后者只在 47 层里存在)。

单个 `[seq_len, 3072]` fp32 张量在 64K 上下文下是 `65536*3072*4 ≈ 786MB`——**如果对全部
层、全部子模块原样存盘,一次 64K prompt 的 oracle dump 会超过 140GB**,不可接受。这正是
任务描述里点名的问题规模。

### 三层策略

1. **统计量(始终存)**:`SubmoduleStats`(mean/std/absmax/L2/min/max,6 个 float)+ 9 个
   分位数 + 按 token 维度求平均后的 `dim_mean` 向量(长度=dim)。代价是 `O(dim)`,**与
   `seq_len` 完全无关**——64K 上下文和 128 token 的 fixture,这一层的体积一模一样。
2. **采样 token(默认策略)**:开头 K + 结尾 K + 中间随机 R 个 token 的**完整精度**向量
   (`CaptureConfig`,默认 `K=8, R=8`,即 24 个 token)。代价是 `O((2K+R)*dim)`,同样与
   `seq_len` 无关(只要 `seq_len >= 2K+R`)。这是 `scan_layers` 默认拿去比较的信号——在
   采样到的具体位置上是满精度的,不是统计量的近似。
3. **全量(`--full`,仅小 fixture 用)**:整个 `[seq_len, dim]` 张量,随 `seq_len` 线性
   增长。

`CachedSubmodule.to_scan_vector()` 按 `full > sample_tokens > dim_mean` 的优先级摊平成
`scan_layers` 需要的单一向量——调用方(尤其是引擎侧实时采集)只要在**同样的采样位置**上
取值,两侧向量长度就能对上;这是"高保真但便宜"这个设计能落地的关键前提,记入 GPU 待办
清单(见下)。

### 体积估算表(按公式手算,未实测——见硬性约束)

单 token 全量体积(fp32,4 类 hidden_size 张量 + 1 类 router 张量,47 层带 router):

```
每 token 体积 = 48层 * 4个hidden张量 * 3072dim * 4byte
            + 47层 * 1个router张量  * 256dim  * 4byte
           = 2,359,296 + 48,128
           ≈ 2.41 MB / token
```

| 上下文长度 | 统计量层(始终) | 采样层(K=8,R=8,默认) | 全量层(`--full`,fp32) |
|---|---|---|---|
| 128 token(小 fixture) | ≈2.3 MB | ≈57.5 MB | ≈0.29 GB |
| 4096 token | ≈2.3 MB | ≈57.5 MB | ≈9.2 GB |
| 65536 token(64K,acceptance-rate 调研场景) | ≈2.3 MB | ≈57.5 MB | ≈147 GB |

关键结论:**统计量层和采样层的体积与上下文长度无关**(只要 `seq_len >= 2K+R = 24`),
默认策略下一次 64K prompt 的 oracle 缓存只要 ~60MB,而不是 147GB。`--full` 只应该在
128~4096 token 级别的小 fixture 上使用;64K 场景下"--full" 是任务描述里明确警告过的
"体积不可接受"路径,不建议使用。

### 缓存 key 与命中/未命中

`CacheKey = (model_revision, prompt_hash, layer_set, capture_config)`。落盘布局:

```
${QSR_ORACLE_CACHE:-${QSR_BFDIAG_DIR:-<repo>/.bfdiag}/oracle_cache}/<prompt_hash>/<config_hash>.safetensors
                                                                    <prompt_hash>/<config_hash>.manifest.json
```

`prompt_hash` 只由 token ids 决定(同一个目录下可以有多个 `capture_config`/`model_revision`
组合并存,靠 `config_hash` 区分文件名,不冲突)。`read_oracle_cache`:文件不存在 → 干净
未命中(`None`);文件存在但 manifest 里的 `config_hash` 和请求的 key 对不上(比如换了
K/R 采样参数或换了 revision)→ 同样当未命中处理,不猜测、不用不匹配的数据——`write_oracle_cache`/
`read_oracle_cache` 都返回 `CacheLookup(hit, path, message)`,`message` 是可以直接打印
给人看的一句话(CLI 里 `oracle cache hit: <path>` / miss 时打印明确指引,见
`bfdiag/divergence/cli.py::_run`)。

## 4. 真实模块名清单(附代码位置)

Laguna-S-2.1 走 vLLM 加载(`runtime/backends/laguna.py:170` 附近 `get_model(...)`),真实
decoder 层子模块名来自两处:

- **本仓库侧的引用证据**(证明这些名字是真实在用的,不是编的):
  - `runtime/backends/laguna.py:195-199` —— 通过 `static_forward_context` 发现注意力层,
    `hasattr(layer, "get_attn_backend")`。
  - `runtime/backends/laguna.py:456-473` —— `_patch_moe_sparkinfer` 用
    `hasattr(module, "gate") and hasattr(module, "experts")` 识别 MoE 层,
    `getattr(moe_module, "shared_expert", None)` 取共享专家。
  - `runtime/backends/laguna.py:518` —— `router_logits, _ = moe_mod.gate(hs)`:路由 logits
    来自 `mlp.gate` 子模块的输出。
  - `runtime/backends/laguna_sparkinfer_moe.py:96-102` —— `NUM_EXPERTS=256`、`TOP_K=10`、
    `HIDDEN_SIZE=3072`、`MOE_LAYER_IDS=list(range(1,48))`(layer 0 是稠密层,不在此列表)。
- **vLLM 侧模型定义**(vendored vLLM checkout,只读,未修改):
  - `vllm/model_executor/models/qwen3_5.py`(`Qwen3_5DecoderLayer`,继承
    `Qwen3NextDecoderLayer`):`self.self_attn`、`self.mlp`(`Qwen3NextSparseMoeBlock` 或
    `Qwen3NextMLP`)、`self.input_layernorm`、`self.post_attention_layernorm`。
  - `vllm/model_executor/models/qwen3_next.py`(`Qwen3NextSparseMoeBlock`):`self.gate`
    （路由,`ReplicatedLinear`)、`self.shared_expert`、`self.experts`(`FusedMoE`);
    (`Qwen3NextAttention`):`self.qkv_proj`、`self.o_proj`。

因此一次完整扫描默认挂的模块名(`capture.default_module_names`)是:每层
`model.layers.{i}.input_layernorm`、`model.layers.{i}.post_attention_layernorm`、
`model.layers.{i}.self_attn`、`model.layers.{i}.mlp`,再加上 `layer_idx != 0` 时的
`model.layers.{i}.mlp.gate`。`capture.parse_layer_submodule` 解析这些名字时,不写死
`"model."` 前缀,而是照搬 `runtime/backends/laguna.py` 里已经在用的做法——找 `"layers"`
这个 segment、取下一个 segment 当层号、剩下的部分当子模块名,这样即使真实加载时的顶层
包装前缀不是 `"model."`(不同 loader 可能不同),解析逻辑依然成立。

## 5. 需要 GPU 才能验证的待办清单

以下事项本任务**只写了代码、从未执行**(硬性约束:不允许使用 GPU,不允许加载模型/vLLM,
不允许建 CUDA tensor),需要有 GPU 权限的后续工作验证:

1. **`bfdiag/divergence/capture.py::capture_engine_activations`**——对一个真实
   `runtime.backends.laguna.LagunaBackend` 实例跑 `ForwardCapture` + `prefill`,确认
   `parse_layer_submodule` 对真实 `model.named_modules()` 输出的解析是否和假设一致
   （尤其是顶层前缀到底是不是 `"model."`,以及 `self_attn`/`mlp` 的输出是单个 tensor 还是
   tuple——`ForwardCapture._tensor_leaves` 已经处理了 tuple/dict 输出,但具体这两个模块
   在真实 forward 里返回的类型需要用真实模型确认一次)。
2. **`thresholds.py` 里的深度放宽常数**(`_DEPTH_GROWTH_COEFFICIENT=0.3`、
   `_MAX_GROWTH=3.0`、`_TOP1_FLOOR`/`_TOP5_FLOOR`)——目前是从历史数字"论证方向"推出来的,
   没有真实 42~48 层逐层数据拟合过。建议:先在一个**已知没有 bug**的 prompt 上跑一次完整
   oracle vs 引擎对拍,把每层每子模块的真实 cos/rel_err/top1/top5 记录下来,重新标定这几个
   常数,确保"正常情况下不误报"这条线画在刚好比真实自然漂移松一点点的地方,而不是本任务
   凭历史数字估出来的位置。
3. **`hidden_state` 这个 threshold kind 目前没有真实 capture 路径**——见第2节局限说明,
   需要在 capture 侧把下一层 `input_layernorm` 的**输入**重新导出成上一层的 `hidden_state`
   （或者直接给 decoder 层本身加一个 forward hook,而不是分别 hook 内部子模块),这需要在
   真实模型上试验哪种方式更准确、开销更小。
4. **oracle 侧真实采集流程**——`oracle/vllm_reference.py` 的 docstring 明确说这个 hook
   要"only after selecting the exact revision and model checkpoint" 在**独立的 vLLM
   checkout** 里实现(不是本仓库,也不是本次改动范围)。本任务的 `cache.py`/`capture.py`
   只负责消费该流程产出的 `{name: tensor}` 安全张量转储(通过 `group_named_tensors` +
   `write_oracle_cache` 摄入),但那个"跑一次 vLLM、导出安全张量"的脚本本身不在本任务
   交付范围内,需要专门跟进(该 docstring 早就把这一点写清楚了,本任务只是遵循,不重复实现)。
5. **`bfdiag/divergence/cli.py::_construct_live_engine_backend` 尚未实现**——刻意留空
   （`NotImplementedError`),因为构造一个真实 `LagunaBackend` 需要 `VllmConfig`/已加载
   checkpoint,这些只在 `server/engine.py` 的启动流程里才有意义,诊断工具不应该自己重新
   实现一遍模型加载。真实 GPU 使用时,建议从一个已经有 backend 实例的脚本直接调用
   `bfdiag.divergence.cli.scan_prompt(oracle_source, EngineCaptureSource(backend,
   module_names), prompt_token_ids)`,而不是走裸 CLI。
6. **采样位置对齐**——`cache.py` 的采样层要求引擎侧在**同样的 token 位置**上取值才能直接
   比较(见第3节)。真实采集时需要确认:引擎侧一次 prefill 之后,`ForwardCapture` 捕获到的
   `[seq_len, dim]` 张量的 token 维度顺序和位置索引,与 oracle 缓存里记录的 `sample_positions`
   是否严格对应(理论上应该,因为都是同一个 tokenized prompt 从位置 0 开始的因果 forward,
   但没有在真实模型上验证过)。

## 怎么验证(不需要 GPU 的部分,现在就能跑)

```bash
# 本任务新增/改动文件的 ruff 检查(全绿)
python -m ruff check bfdiag/ oracle/comparator.py \
  tests/test_bfdiag_divergence.py tests/test_bfdiag_oracle_cache.py tests/test_bfdiag_thresholds.py

# 本任务新增测试(核心验收测试在 test_bfdiag_divergence.py 里,42 层/layer 17 注入偏差那组)
python -m pytest -q tests/test_bfdiag_divergence.py tests/test_bfdiag_oracle_cache.py \
  tests/test_bfdiag_thresholds.py tests/test_comparator.py

# 各模块自带的 __main__ 自测入口
python -m bfdiag.divergence.scan
python -m bfdiag.divergence.thresholds
python -m bfdiag.divergence.capture
python -m bfdiag.divergence.cache
python -m bfdiag.divergence.report

# CLI 自身(在 bfdiag/cli.py 总调度落地前也能独立跑)
python -m bfdiag.divergence.cli --help
python -m bfdiag.divergence.cli --prompt <token_ids.json>   # 无缓存时给出清晰的 miss 提示，
                                                              # 不触碰 GPU/模型
```

以上命令均已在本次任务里实际执行验证(见对应章节),没有出现需要 GPU、torch.cuda、模型
权重的路径。
