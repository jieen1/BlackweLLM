# bfprobe P2b:MoE 路由探针 + vLLM oracle 侧路由抽头定位

日期:2026-07-27
状态:调研完成、代码完成(CPU 侧全绿),GPU 验证未做(硬约束:本次任务全程零 GPU)
方法:vLLM 侧结论全部来自 `/home/bot/vllm` 源码通读(只读,附 `文件:行号`);两个 checkpoint 的
`config.json` 只读取,未加载模型、未跑 CUDA。

---

## 0. 一句话结论

**vLLM 侧不需要 monkeypatch,也不需要碰源码。** vLLM 自带一个官方、公开、文档化的引擎开关
`enable_return_routed_experts`,专门用来把每层 MoE 的 `topk_ids` 通过 OpenAI 兼容 API 的
`routed_experts` 字段直接返回给客户端(base64 编码的 `.npy`)。我们这一侧已经在
`runtime/backends/laguna.py:532` 挂好了对等探针。两侧路由计算**字面上调用的是同一个 vLLM 函数**
(`fused_topk_bias`),所以专家编号约定、topk 排序、归一化、softcap 全部逐项核对一致(见 §2)。
唯一悬而未决的是一个**当前 vLLM 检出版本里的已知限制**(`enable_return_routed_experts` 尚不支持
"V2 model runner"),但针对我们这个具体模型配置(target 混合注意力、draft 均匀滑窗注意力),
V2 默认不会被选用,所以这条限制大概率不影响我们——这是本笔记里唯一需要 GPU 才能坐实的推断。

---

## 1. vLLM 侧路由抽头点:精确位置与可行性

### 1.1 先确认 vLLM 侧真正在跑的路由代码路径,不是猜的

`vllm/model_executor/models/laguna.py:120-246`(`class LagunaMoE`)是 vLLM 自带的、真实跑在
oracle 服务器上的模型类。它的 `forward`(`:234-246`)：

```python
def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    ...
    router_logits, _ = self.gate(hidden_states)
    router_logits = router_logits.float()
    softcap = getattr(self.config, "moe_router_logit_softcapping", 0.0) or 0.0
    if softcap > 0.0:
        router_logits = torch.tanh(router_logits / softcap) * softcap
    final_hidden_states = self.experts(hidden_states, router_logits)
    return final_hidden_states.view(orig_shape)
```

**这和我们自己 `runtime/backends/laguna.py:517-531` 的 `_patched_forward` 几乎逐行相同**——
说明我们的 sparkinfer 补丁本来就是照抄 vLLM 自己这份 `LagunaMoE.forward` 写的。`self.experts`
是一个 `FusedMoE` 实例(`:215-232` 构造,`scoring_func="sigmoid"`、`use_grouped_topk=False`、
`renormalize=config.norm_topk_prob`、`e_score_correction_bias=<checkpoint 里的偏置>`、
`routed_scaling_factor=self.routed_scaling_factor`、`apply_routed_scale_to_output=True`)。

`FusedMoE.forward` 最终落到 `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
的 `_apply_quant_method`(约 `:544-583`):

```python
if self.routed_experts.quant_method.is_monolithic:
    # Monolithic kernels: pass router_logits to routed_experts
    fused_out = self.routed_experts.forward_monolithic(x=hidden_states, router_logits=router_logits, ...)
else:
    # Modular kernels: select experts first, then call routed_experts
    topk_weights, topk_ids = self.router.select_experts(hidden_states=hidden_states, router_logits=router_logits, ...)
    fused_out = self.routed_experts.forward_modular(x=hidden_states, topk_weights=topk_weights, topk_ids=topk_ids, ...)
```

**这是本次调研最关键的分叉点**:如果 `is_monolithic` 为真,`topk_ids`/`topk_weights` 永远
不会在 Python 层出现,直接被塞进 CUTLASS/FlashInfer 内核内部计算——这正是设计文档里担心的
"路由融合进 kernel、Python 层拿不到" 的情形。必须先确认我们这个模型走的是哪一支。

### 1.2 确认 FlashInfer CUTLASS(我们要对拍的那条路径)不是 monolithic

- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:38-190`:`NvFp4MoeBackend` 枚举,
  `"flashinfer_cutlass"` → `NvFp4MoeBackend.FLASHINFER_CUTLASS` → `experts_cls = [FlashInferExperts]`
  (`:82-87`)。
- `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py:62`:
  `class FlashInferExperts(mk.FusedMoEExpertsModular)`。
- `vllm/model_executor/layers/fused_moe/modular_kernel.py:762-770`:
  `class FusedMoEExpertsModular` 的 `is_monolithic() -> False`(静态方法,写死)。
  对照 `:958-991` 的 `class FusedMoEExpertsMonolithic`(`is_monolithic() -> True`,只有
  `FLASHINFER_TRTLLM`/`FLASHINFER_CUTEDSL` 等其他后端走这支)。

**结论:`FLASHINFER_CUTLASS`(笔记里说的"vLLM 的 FlashInfer CUTLASS MoE")是 `FusedMoEExpertsModular`
子类,`is_monolithic() == False`。** 所以它走的是 `moe_runner.py` 里 **"Modular kernels" 那一支**——
`self.router.select_experts(...)` 确确实实在 Python 层被调用,`topk_weights`/`topk_ids` 是普通
torch 张量,和我们这边的情况完全一样。**路由没有被融合进 kernel。**

`compressed_tensors_moe_w4a4_nvfp4.py:262-274` 里那个 `assert self.is_monolithic` 的
`apply_monolithic` 方法确实存在,但只有在选中 `FLASHINFER_TRTLLM` 等 monolithic 后端时才会被
调用到——对 `FLASHINFER_CUTLASS` 是死代码,不影响结论。

### 1.3 谁在算 topk:和我们调用的是同一个函数

`vllm/model_executor/layers/fused_moe/router/router_factory.py:161-190`
(`create_fused_moe_router`)按优先级派发路由器类:因为 `LagunaMoE` 总是传
`e_score_correction_bias`(`:205-208` 构造的零张量,checkpoint 会覆盖),`use_grouped_topk=False`,
`custom_routing_function=None`,所以必然落到

```python
if e_score_correction_bias is not None or hash_indices_table is not None:
    return FusedTopKBiasRouter(...)
```

`vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:324-412`
(`class FusedTopKBiasRouter._compute_routing`)内部调用的,就是同文件 `:161-321` 定义的自由函数
**`fused_topk_bias`**——**和我们 `runtime/backends/laguna.py:23` 里
`from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias`
导入、在 `:523-531` 直接调用的,是同一个函数,同一份源码,同一个 CUDA 核(`sigmoid` 分支
`:213-223` 走 `vllm_topk_sigmoid` → `ops.topk_sigmoid`,一个自定义 CUDA op)。**

也就是说:**两条推理引擎在这一点上没有"两套不同的 MoE 路由实现"这回事——字面上是同一段代码。**
如果两边喂进去的 `router_logits` 完全一致,`topk_ids` 应该逐位相同(同一个确定性核,同样的入参)。
真正可能出现分歧的地方,是喂进这个函数之前的 `router_logits` 本身(上游 attention/GEMM 数值路径
不同导致 hidden_states 有微小差异,而 top-k 选择是输入的不连续函数,临界点附近的极小扰动就能
掀翻整个 top-10 集合)——这比"两套 MoE kernel 数值路径不同"这个原假设更精确,也更容易验证
(见 §3 的一句话总结)。

### 1.4 挂钩点:不用 monkeypatch,vLLM 自带一个官方开关

继续往回追:`FusedMoERouter`(`base_router.py`/`fused_moe_router.py`)本来就设计了一个捕获钩子:

`vllm/model_executor/layers/fused_moe/router/base_router.py:159-306`
(`class BaseRouter(FusedMoERouter)`):

```python
def set_capture_fn(self, capture_fn: Callable[[torch.Tensor], None] | None) -> None:  # :185-187
    """Set a capture callback for logical routed expert IDs."""
    self.capture_fn = capture_fn
...
def _select_experts(self, ...):
    ...
    topk_weights, topk_ids = self._compute_routing(...)
    # Capture logical ids before EPLB mapping.                                          # :296-297
    if self.capture_fn is not None:
        self.capture_fn(topk_ids)
    topk_ids = self._apply_eplb_mapping(topk_ids)          # EPLB 重映射发生在捕获之后
    ...
```

**这不是我发明的挂钩方案——vLLM 自己的引擎已经在用它了**,而且刚好是为了同一类目的
(捕获路由决策)。`vllm/v1/worker/gpu_model_runner.py:7477-7532`
(`init_routed_experts_capturer` / `_bind_routed_experts_capturer`):

```python
def _bind_routed_experts_capturer(self, capturer: RoutedExpertsCapturer) -> None:  # :7519
    for module in self.compilation_config.static_forward_context.values():
        if isinstance(module, MoERunner) and isinstance(module.router, BaseRouter):
            layer_id = module.layer_id
            def _capture_fn(topk_ids, _layer_id=layer_id, _capturer=capturer):
                _capturer.capture(_layer_id, topk_ids)
            module.router.set_capture_fn(_capture_fn)                                   # :7532
```

这就自动给**模型里每一个 MoE 层**挂好了钩子,写进一个 GPU 常驻的 `RoutedExpertsCapturer`
(`vllm/model_executor/layers/fused_moe/routed_experts_capturer.py:58-220`):设备端 buffer
形状 `(max_num_batched_tokens, num_hidden_layers, num_experts_per_tok)`,dtype `int32`;
调度器侧还有一个按物理 KV-cache slot 索引的 `RoutedExpertsManager`(同文件
`:223-349`),专门设计成能在抢占/前缀缓存复用后依然对得上同一批 token 的路由记录。

**而且这整条链路已经打通到 OpenAI 兼容 API 的响应体里**,不需要我们写任何客户端代码去接驳:

- `vllm/config/model.py:220-221`:`ModelConfig.enable_return_routed_experts: bool = False`
- `vllm/entrypoints/llm.py:142,202,329`:`LLM(..., enable_return_routed_experts=True)` 构造参数
- `vllm/engine/arg_utils.py:421`:`EngineArgs.enable_return_routed_experts`,自动生成
  `vllm serve --enable-return-routed-experts` CLI 开关
- `vllm/sampling_params.py:320-329`:每请求可选的 `routed_experts_prompt_start`(跳过前 N 个
  prompt token 的路由,多轮对话场景用)
- `vllm/entrypoints/openai/chat_completion/protocol.py:105-114` /
  `vllm/entrypoints/openai/completion/protocol.py:571-580`:
  `ChatCompletionResponseChoice.routed_experts` / `CompletionResponseChoice.routed_experts`
  ——**base64 编码的 `.npy` 字节**,解码后形状 `(num_tokens - 1, num_layers, num_experts_per_tok)`,
  dtype `uint8`(256 专家 ≤ 256,用 uint8;详见 `routed_experts_capturer.py` 里
  `RoutedExpertsManager` 的 dtype 选择逻辑),解码方式官方注释里写明:
  `np.load(io.BytesIO(base64.b64decode(s)))`。

**结论:GPU 验证批次不需要写任何 hook/monkeypatch 代码去拿 vLLM 侧的 `topk_ids`。** 只要在
起 oracle vLLM 服务时加 `--enable-return-routed-experts`,同一个 HTTP completions 请求的响应里
就会带 `routed_experts` 字段,解码即用。这比设计文档原先设想的"forward hook / monkeypatch /
CPU 侧重算 topk 对拍"都更干净、更权威(是 vLLM 自己的公开产品特性,不是我们逆向出来的私有实现
细节,版本升级也不容易失效)。

### 1.5 已知限制,必须在 GPU 批次里第一件事验证

`vllm/config/vllm.py:2149-2151`(`_get_v2_model_runner_unsupported_features`)把
`enable_return_routed_experts` 列为**"V2 model runner" 尚不支持**的特性
(`# Will be added by https://github.com/vllm-project/vllm/pull/38163`,上游 PR 还没合并)。

顺着 `use_v2_model_runner` 属性(`vllm/config/vllm.py:530-568`)往下查,V2 是否会被选中取决于:

1. `envs.VLLM_USE_V2_MODEL_RUNNER` 环境变量显式覆盖(优先级最高,直接短路)。
2. `speculative_config.method == "dspark"` → 强制 V2。**不适用**(我们用 DFlash)。
3. `_dflash_needs_multi_kv_group()`(`:576-583`)→ 强制 V2。这个函数检查的是 **draft 模型**的
   `layer_types` 是否混合了 `sliding_attention` 和非 sliding 类型
   (`0 < num_sliding < len(layer_types)`)。查了真实 draft checkpoint 配置
   (`~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-DFlash-NVFP4/.../config.json`):
   `layer_types` 是 6 层**全部** `sliding_attention`——`num_sliding == len(layer_types)`,
   条件不满足,**不强制 V2**。
4. `_is_default_v2_model_runner_model()`(`:585-601`)要求 `not model_config.is_hybrid`——但我们的
   **target** 模型(Laguna-S-2.1,12 层 full_attention + 36 层 sliding_attention)是混合注意力
   模型,`is_hybrid` 应为真,**这一步直接返回 False**,V2 默认根本不会被尝试。

**结论(源码推导,未经 GPU 验证):对 Laguna-S-2.1(混合注意力 target + 均匀滑窗 draft)这个具体
配置,`use_v2_model_runner` 默认应为 `False`,所以 `_get_v2_model_runner_unsupported_features`
那条"V2 不支持 routed-experts-capture"的限制根本不会被触发——`enable_return_routed_experts`
应该能和我们的 DFlash oracle 配置正常共存。**

**但这只是源码推导,不是 GPU 实测,必须在下一次 GPU 验证批次里第一件事确认**(见 §4)。另外
还查到两条独立的不兼容项(`vllm/config/vllm.py` 里 `enable_return_routed_experts` 校验块附近):
流水线并行(PP > 1)、以及任何 KV connector(PD 分离部署或 KV offload)——我们是单 GPU 单实例,
大概率不涉及,但也要在起服务的命令行里确认没有意外开启。

### 1.6 如果 §1.5 的推导错了怎么办:替代方案

万一 GPU 验证发现 `enable_return_routed_experts` 和我们实际用的 DFlash 启动方式确实冲突
(比如启动脚本设了 `VLLM_USE_V2_MODEL_RUNNER=1`,或上游还有别的隐藏耦合),按优先级排列的
替代方案:

1. **最简单**:在同一 prompt 上跑一次**不开 DFlash 的普通 vLLM 解码**
   (`enable_return_routed_experts=True`,无 speculative_config),得到 target 模型在这个
   prompt 每个位置的路由 ground truth。DFlash 的 verify 轮本质是"target 模型对同一段
   context+token 序列做一次前向",只要 token 序列一致,路由应该和逐 token 跑普通解码一致——
   这样可以完全绕开 V2 model runner 的任何潜在限制,GPU 批次成本也更低(不需要跑 DFlash
   的复杂 CUDA Graph 组合)。**建议作为第一优先级方案**,即使 §1.5 的推导是对的也值得先用它
   validate 一遍(更简单、更快出结果)。
2. 用 `set_capture_fn`(§1.4)自己手写一个薄的绑定层,复刻
   `gpu_model_runner.py:_bind_routed_experts_capturer` 的逻辑,但不依赖
   `enable_return_routed_experts` 整条官方流水线(绕开它可能触发的 V2/PP/KV-connector 校验),
   只在自己的一次性验证脚本里手动 `for module in ...: if isinstance(module.router, BaseRouter):
   module.router.set_capture_fn(...)`。这是"只读 + hook"级别的改动,不碰 vLLM 源码。
3. 最后手段(不推荐,仅记录):monkeypatch
   `vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router.fused_topk_bias`
   模块属性本身,用一层包装函数记录调用参数和返回值——因为它是模块级自由函数,
   `FusedTopKBiasRouter._compute_routing` 通过模块全局命名空间解析调用,替换模块属性即可
   拦截,不需要动任何类定义。这个方案额外收益是**同时**拿到 `router_logits`(函数入参)和
   `topk_weights`(函数返回值第一项),比 `set_capture_fn`(只給 `topk_ids`)信息量更大,
   但因为是我们自己发明的挂钩方式,不如官方 `enable_return_routed_experts` 稳。

**以上三条都不需要修改 `/home/bot/vllm` 或 `/home/bot/project/sparkinfer` 的任何源码。**

---

## 2. 两侧张量语义核对表

逐项核对,结论标注在每一行末尾。

| 项 | 我们这侧 | vLLM 侧(oracle) | 结论 |
|---|---|---|---|
| 路由函数 | `fused_topk_bias`(直接 import vLLM 的自由函数,`laguna.py:23,523-531`) | `FusedTopKBiasRouter._compute_routing` 内部调用同一个 `fused_topk_bias`(`fused_topk_bias_router.py:378-391`) | **同一份代码,同一个 CUDA 核** |
| 专家编号约定 | `fused_topk_bias` 返回的原始 id,`[0, 256)`,不做任何重映射 | `set_capture_fn` 捕获点在 `_apply_eplb_mapping` **之前**(`base_router.py:296-300`),即 pre-EPLB 逻辑 id;若走 `enable_return_routed_experts` 官方通道,`RoutedExpertsCapturer.capture` 直接写入 `select_experts` 返回前经 `capture_fn` 拿到的同一份 id | **一致**,前提是 oracle 未开 EPLB(`enable_eplb` 默认 False,需要在启动命令行里确认没有 `--enable-eplb`) |
| top-k 排序约定 | `ops.topk_sigmoid`(自定义 CUDA 核)内部顺序,未额外重排 | 同一个 `ops.topk_sigmoid` 调用 | **一致**(同一个核,同一种"原生顺序");`bfprobe/routing_compare.py` 仍然把"集合是否一致"和"顺序是否一致"分开报告,不假设顺序语义,防止未来任一侧换核导致误判 |
| 归一化(renormalize) | `_renorm = getattr(hf_config, "norm_topk_prob", True)`,checkpoint 里是 `true` | `renormalize=config.norm_topk_prob`(`laguna.py:221`,vLLM 模型定义里),同一个 `norm_topk_prob=true` 字段 | **一致** |
| softcap | `_softcap = getattr(hf_config, "moe_router_logit_softcapping", 0.0) or 0.0`;checkpoint 里 `moe_router_logit_softcapping=0.0` | 同一份 `hf_config` 字段,同样为 0.0(`vllm/model_executor/models/laguna.py:241`) | **一致,且对本 checkpoint 是 no-op**(两边分支都不触发 tanh 软限幅) |
| `routed_scaling_factor`(2.5,checkpoint 里 `moe_routed_scaling_factor=2.5`) | 路由调用时硬编码 `routed_scaling_factor=1.0`(`laguna.py:530`),缩放挪到**输出**上(`if _scaling != 1.0: routed_out = routed_out * _scaling`) | `LagunaMoE` 用 `apply_routed_scale_to_output=True` 构造 `FusedMoE`;`layer.py:293-299` 的工厂函数因此把传给路由器的 `routed_scaling_factor` 强制改成 `1.0`("so it ends up being a nop"),真正的 2.5 倍缩放挪到 `moe_runner.py:390-406` 的 `_maybe_apply_routed_scale_to_output`,同样是**输出**级别 | **一致**——这条本来是最容易踩的坑(担心两边 `topk_weights` 原始值相差 2.5 倍),追完源码后确认两边都把 2.5 倍缩放推迟到输出,路由器返回的 `topk_weights` 本身都是未缩放、归一化到 sum≈1 的值,可以直接比较,不需要额外除以/乘以 2.5 |
| `moe_apply_router_weight_on_input` | `_apply_on_input = getattr(hf_config, "moe_apply_router_weight_on_input", False)`;checkpoint 里 `false` | 同一份 `hf_config` 字段,`apply_router_weight_on_input=bool(config.moe_apply_router_weight_on_input)`(`laguna.py:226`) | **一致,且为 False**——两边都不会把路由权重提前乘进输入 |
| 共享专家(shared expert) | `_shared(hs)` 单独跑,`_shared is not None` 时输出相加(不进 topk_ids) | `LagunaMoE.shared_expert` 作为独立 nn.Module 传给 `FusedMoE(shared_experts=...)`,**不是** `num_fused_shared_experts`(那是另一种把共享专家 id 拼进 topk_ids 数组的机制,本模型未使用) | **一致**——两边 `topk_ids`/`topk_weights` 数组长度都恰好是 `top_k=10`,不含共享专家的额外槽位 |
| 路由是否融合进 kernel(monolithic) | 不适用(sparkinfer 是我们自己接的 kernel,路由永远在 Python 层) | `FLASHINFER_CUTLASS` 后端的 `FlashInferExperts` 是 `FusedMoEExpertsModular`,`is_monolithic()==False`(`modular_kernel.py:770`)| **未融合**,`topk_ids`/`topk_weights` 在 vLLM 侧也是 Python 层的普通张量(`moe_runner.py:566-574`) |

**唯一还需要 GPU 上实测确认的语义项**:oracle 实际跑 100% 接受率那次的启动命令行有没有开
`--enable-eplb`(会引入专家副本重映射,需要改用 pre-EPLB 捕获点)、有没有设
`VLLM_USE_V2_MODEL_RUNNER=1`(会触发 §1.5 的限制)。这两条都是"读一下启动脚本/环境变量"级别的
核对,不需要重新推理,但仍然只能在有 GPU/有权限查看该次运行环境的场合完成。

---

## 3. 对头号未解问题的影响(重要,写清楚以免误导后续调研方向)

`notes/2026-07-27-acceptance-rate-gap-vllm-vs-ours-same-prompt.md` 的头号假设是
"我们用 sparkinfer MoE,vLLM 用 FlashInfer CUTLASS,两条数值路径不同"。**本次调研的发现让这个
假设需要修正:两条路径在"路由决策"这一步,字面上调用的是同一个 vLLM 函数、同一个 CUDA 核**
(§1.3)。所以如果两边的 `router_logits` 完全相同,`topk_ids` 应该逐位相同——真正可能不同的,
只有专家计算本身(sparkinfer kernel vs. FlashInfer CUTLASS kernel 的 GEMM/量化数值路径),或者
更上游的 `router_logits` 本身因为 attention/其他 GEMM 数值差异而产生了微小漂移,而 top-k 选择
对输入是不连续函数,临界点附近的极小扰动就会掀翻整个专家集合。

`bfprobe` 这一对探针(我们侧 `laguna.py:532` + vLLM 侧 `enable_return_routed_experts`)刚好能把
这两种可能性分开:

- 如果两侧 `topk_ids` 逐层逐 token 完全一致 → 说明分歧在专家计算本身(sparkinfer vs CUTLASS
  的数值路径),原假设成立,该去查 GEMM/量化数值细节。
- 如果两侧 `topk_ids` 在某个 (layer, token) 开始分叉 → 说明分歧其实起源于**更上游**的
  `router_logits`(或者更上游的 hidden_states),路由本身没有独立的数值 bug,但路由的"选择"
  这个离散操作把上游的微小差异放大成了完全不同的专家组合——这是一个更精确、更容易继续往上游
  追的定位(可以用同一套 `router_logits` 张量去对拍,比如检查是不是从某一层开始 cos 相似度掉
  下来的)。

`bfprobe/routing_compare.py` 的 `first_divergence` 字段就是为了回答"从哪里开始不一致"这个问题
设计的。

---

## 4. 我们这侧的集成点与开销

### 4.1 集成(实际改动,不是设计)

`runtime/backends/laguna.py` 改动 **2 行**(远低于 ≤5 行预算):

```python
# 顶部 import 区(:22)
from bfprobe.routing import capture_routing

# _patched_forward 内部,fused_topk_bias 调用之后(:532)
capture_routing(router_logits, topk_ids, topk_weights)  # bfprobe P-TOPK
```

`PROBE_ENABLED=False` 时,`capture_routing` 第一行就 `return`(单次布尔判断,不触碰任何张量),
`bfprobe/routing.py:63-76` 的 docstring 和实现都体现了这一点。

### 4.2 站点表(全局唯一,300-399 号段)

| site_id | 名称 | 层级 | 形状 | 语义 |
|---|---|---|---|---|
| 300 | `SITE_ROUTER_LOGITS` | T2 | `(M, 256)` float32 | softcap 之后的 gate 输出(P-ROUTER-LOGITS) |
| 301 | `SITE_TOPK_IDS` | T2 | `(M, 10)` int32 | pre-EPLB 逻辑专家 id,router kernel 原生顺序(P-TOPK,最高价值站点) |
| 302 | `SITE_TOPK_WEIGHTS` | T2 | `(M, 10)` float32 | 归一化后的路由权重(未乘 `routed_scaling_factor`,见 §2 表) |

**排序约定**:不显式发送层索引。`capture_routing` 每层调用一次,MoE 层在模型加载时按
checkpoint 层号升序被 patch 一次(`_patch_moe_sparkinfer` 的 `model.named_modules()` 遍历),
每轮前向都按同样固定顺序重新触发这些闭包——离线消费者靠这个固定调用顺序 + 已知层数(47)
重建 `(round, layer, token)` 网格,和设计文档里 T1 每层签名环的约定一致(同样没有显式层索引)。

### 4.3 成本

47 个 MoE 层 × 16 verify token × 10 专家 × (4+4) B = **60 KB/轮**,相对 44.16 ms/轮预算是
**0.0001%**,和设计文档 §4 的预算完全一致(未重新测量,沿用既有估算,因为张量形状和调用频率
没有变化)。

---

## 5. 需要 GPU 才能验证的待办清单

按优先级排列,建议合并进下一次 GPU 验证批次(批次 #2,MoE 路由专项):

1. **【最优先,最简单】** 不开 DFlash,起一次普通 vLLM 解码(`--enable-return-routed-experts`),
   同一个 64K 重复短语 prompt,拿到 target 模型逐 token 的 `routed_experts` 字段,解码验证格式
   与本笔记描述一致(shape、dtype、`num_tokens - 1` 语义)。
2. 确认 `enable_return_routed_experts` 能否和 vLLM 原生 DFlash 同时开启而不报错/不被静默降级
   ——验证 §1.5 的"V1 model runner 默认生效,V2-only 限制不触发"这条纯源码推导。如果报错或
   行为异常,退回 §1.6 的替代方案 1(逐 token 普通解码代替 DFlash verify 轮)。
3. 确认 oracle 那次 100% 接受率实测用的启动命令行/环境变量:有没有 `--enable-eplb`、有没有
   `VLLM_USE_V2_MODEL_RUNNER=1`、有没有 PP>1 或 KV connector 相关开关。
4. 用同一个 64K 重复短语 prompt,两侧都打开 T2 探针(我们侧 `PROBE_ENABLED=True`,vLLM 侧
   `enable_return_routed_experts=True`),跑一次 DFlash round(或 §1.6 方案 1 的普通解码),
   把两侧 `topk_ids` 喂进 `bfprobe/routing_compare.py::compare_routing`,读 `first_divergence`
   和 `verdict`。
5. 零假设门禁:确认开启 `PROBE_ENABLED=True`(我们侧)后,输出 token 逐位相同、延迟 p50 偏差
   < 0.5%(design doc §6 的通用要求,这里只是把它套到 P2b 这一对探针上)。
6. 如果 §5 发现 `topk_ids` 一致但 `router_logits`/`topk_weights` 有数值差异,进一步核对是否是
   `routed_scaling_factor` 处理路径的隐藏差异被 §2 表格漏掉了(理论上不应该,但值得在真实数据
   上复核一次 cos/max_abs)。

---

## 6. 交付清单

- `bfprobe/__init__.py`——按共享契约逐字节写死的一行 docstring。
- `bfprobe/_bus_stub.py`——本地最小 bus 桩,供本任务单测使用;真实 `bfprobe/bus.py`(P1 agent
  产出)落地后,`routing.py` 的 `try/except ImportError` 会自动切到真实实现,无需再改
  `routing.py`。
- `bfprobe/routing.py`——`capture_routing(router_logits, topk_ids, topk_weights)`,站点
  300/301/302。
- `bfprobe/routing_compare.py`——纯函数 `compare_routing`,输出 `RoutingComparisonResult`
  (top1/set/sequence 匹配率、Jaccard、`first_divergence`、权重 cos/max_abs、一句话 verdict)。
- `bfprobe/report.py`——`to_dict`/`render_text`/`render_json`。
- `bfprobe/cli.py`——`register(subparsers)` 挂 `bf probe routing`(当前从 `.npy` 文件读两侧
  张量;等 `bfprobe/bus.py` 的存储后端落地后,把数据加载换成读 `${QSR_BFDIAG_DIR}` 里的真实
  run,对比逻辑不用动)。
- `runtime/backends/laguna.py`——2 行加法式集成(`:22` import,`:532` 调用)。
- `tests/test_bfprobe_routing.py`——4 个用例,覆盖:站点 id 稳定、关闭时零发射、关闭时不触碰
  参数、开启时顺序与内容正确、多层调用顺序保持。
- `tests/test_bfprobe_routing_compare.py`——9 个用例,含任务要求的三个核心验收(用例
  A/B/C:完全一致 / layer23-token7 单点专家替换 / 仅顺序不同集合相同),外加形状校验、
  Jaccard 部分重叠、权重 cos/max_abs、权重两侧必须同时提供或同时省略。

**验证命令**(全部 CPU-only,已跑通):

```bash
python -m pytest -q tests/test_bfprobe_routing.py tests/test_bfprobe_routing_compare.py
ruff check bfprobe/ tests/test_bfprobe_routing.py tests/test_bfprobe_routing_compare.py
```

14/14 测试通过,ruff 全绿(仅限本任务清单内的文件;`runtime/backends/laguna.py` 里两处历史遗留
的 `I001`/`E501` 是改动前就存在的,不在本次改动引入范围内,已用 `git show HEAD:...` 核实)。
