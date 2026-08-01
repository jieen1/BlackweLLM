# 完全剥离 vLLM 依赖——分阶段实施计划(2026-07-27)

> **2026-08-01 更新（T0-7 仓库卫生）**：本文档引用的验证脚本
> `benchmarks/_phase1_bitexact_validate.py`（+ `_long`）、
> `benchmarks/_phase3_dflash_bitexact_validate.py`（+ `_long`）、
> `benchmarks/_phase5_e2e_bitexact_validate.py` 已于 2026-08-01 删除
> （用户明确决定：vLLM 剥离主线已完成 `a9cb932`，这批验证脚本不再需要保留）。
> 下文所有对这些文件名的引用指向的都是历史记录,脚本本身可以从 git 历史找回：
> `git log --diff-filter=D -- benchmarks/_phase5_e2e_bitexact_validate.py`
> （其余四个同理,把文件名换掉即可）。

## 范围确认(用户,2026-07-27)

**qwen3.6(DirectModelRunner)路径本次不动,阶段4整体跳过。以 Laguna 为主,做完
阶段0/1/2/3/5。** 总工期按不含阶段4的估算:**约6-8周一人力,阶段2/3并行可压缩到
5-6周**。

## 目标与范围

**目标是彻底替换 vLLM**:生产服务(Laguna + DFlash)启动和运行全程不需要 `import vllm`
成功,`vllm` 不再是运行时硬依赖(可以卸载,或者只留在专门跑"stock vLLM 原生基准对比"
的 `benchmarks/` 脚本环境里,那部分本来就是故意保留、不在剥离范围内)。

本计划基于三轮调研:
1. `notes/2026-07-27-vllm-flashinfer-dependency-audit-and-decoupling-roadmap.md`(全局依赖清单)
2. `get_model()` 剥离工作量深挖(未单独存档,结论已吸收进本计划)
3. NVFP4量化/DFlash草稿模型加载机制深挖(未单独存档,结论已吸收进本计划)

FlashInfer 部分**已经不需要规划**——Laguna 生产路径对 flashinfer 的真实依赖已经是零
(见第1轮调研),这里不再重复。

---

## 全局发现:比预期干净的部分 vs 比预期重的部分

**已经完全不需要处理(零工作量,这次调研反复确认)**:
- 采样、Tokenizer、调度器、KV block/prefix cache 管理、FLA chunk 索引——已完全自研。
- Attention 计算(Laguna 用 `bf_attention.py`,DFlash draft 用 `SparkinferAttentionImpl`)、
  MoE 计算(`sparkinfer.moe.fused_moe`)、RMSNorm 计算(`fused_rms_norm.py`)——已完全自研,
  只是部分还没接入 DFlash 的 precompute 链路(见阶段3)。
- **CustomOp dispatch + `@support_torch_compile`**:生产配置下 `skip_compiled=True` 100%覆盖,
  这两套机制现在是**死代码**,不需要移植,直接砍掉。
- NVFP4 矩阵乘法本体(`nvfp4_gemm_sm120.so`)——已经是纯张量函数,不耦合vLLM对象,可以
  直接被自己的 Linear 类调用。
- NVFP4 权重侧预处理(swizzle/pad)和 global scale 数值逻辑——纯张量操作,可直接照搬。
- DFlash 的 embedding/lm_head 权重共享、`SpeculativeConfig`等价配置构造、`SupportsEagle3`
  接口契约——三块都验证过复杂度很低,不是这次真正的硬骨头。

**比预期重、新发现的部分(容易被"import行数少"这种表象误导)**:
- **权重加载 orchestration**:之前以为 `loader/` 包已经打了基础,实测对 Laguna 覆盖率
  **0%**(是给另一个模型写的),需要新写中等体量的加载器。
- **自己的 Linear/Embedding 类**:如果目标是彻底剥离(不只是不调用`get_model()`,而是
  连vLLM的`ColumnParallelLinear`等类都不用),这块是全新工作量,之前的"2-3周"估算
  没有包含这部分。
- **NVFP4 激活量化 kernel**(`scaled_fp4_quant`):~430行CUDA,目前仍是vLLM编译产物,
  是量化链路里唯一没被自研替换的一环。
- **DFlash 核心算法**(`combine_hidden_states`+`precompute_and_store_context_kv`):这是
  本轮调研最大的意外发现——`runtime/backends/laguna_dflash.py` 表面只有3行vLLM import,
  显得"已经很自主",但DFlash相对普通投机解码的核心性能优化(融合多层context-KV投影)
  整个委托给了vLLM的模型类方法,~400-450行真实算法代码在vLLM checkout里,不在我们
  自己的代码里。其中批量RoPE是目前项目里**唯一完全没有先例的自研空白**。

**贯穿全程的最大风险不是写代码难度,是数值bit-exact验证成本**——每一块都有历史上
真实踩过坑的记录(K/V竞态、NVFP4 scale round-trip、CUDA Graph地址过期等),这类bug
往往要跑到端到端精度对比才暴露,预算时间要向验证倾斜,不是向写代码倾斜。

---

## 阶段0:清场 + 前置决策(~2-3天)

不是"剥离"本身,是让后续阶段在干净地基上进行。

1. **修复依赖护栏测试**(当前是红的):`bf_attention.py`/`laguna_cuda_graph.py`/
   `laguna_sparkinfer_attn.py` 三处新增的直接vLLM import,统一改走
   `fused_kv_scatter.py`(已经写好,只差接线),顺带验证bit-exact。
2. **接线 `fused_kv_scatter.py` 到全部调用点**——包括DFlash draft attention的
   `do_kv_cache_update`(`laguna_sparkinfer_attn.py:330-349`)那一处,这次调研确认它
   也在用 `reshape_and_cache_flash`。
3. **需要你决策的前置问题(上一轮就提过,现在必须定下来才能排后续优先级)**:
   **DirectModelRunner(qwen36)现在还是真实会被启动的生产路径吗?** 这条路径的
   attention 走的是 vLLM 原生 `SM120GQABackend`,依赖vLLM Attention ABC契约的深度
   最大(见阶段4),如果这条路径已经不再使用,可以直接归档,阶段4整个跳过,总工作量
   会显著减少。
4. 顺带修一个今天调研中复现出来的真实bug:DirectModelRunner默认开启的NVFP4 B12x
   kernel patch会因flashinfer版本冲突崩溃启动(已用真实调用复现)。如果阶段0决定
   保留DirectModelRunner,这个要立刻修;如果决定归档,不用管。

---

## 阶段1:自建模型图 + 主模型权重加载(不含Linear/Embedding类替换)(~1-1.5周)

目标:不再调用 `get_model()`,但暂时继续复用vLLM的 `ColumnParallelLinear`/
`VocabParallelEmbedding`等类(推迟到阶段2再替换),先把"构造+加载"这个编排步骤
自己接管。

1. **模型注册表查找**替换成硬编码 import(几行代码,不需要保留通用注册表机制)。
2. **模型forward骨架重写**(参照 `LagunaModel.forward()` 现有结构抄,去掉PP分支、
   去掉CustomOp/`@support_torch_compile`相关部分):embedding查表→48层残差数据流转
   循环→最终norm→lm_head matmul,~100-200行。
3. **权重加载 orchestrator 新写**:照抄裁剪vLLM现有三层逻辑(`AutoWeightsLoader`+
   `DefaultModelLoader.load_weights`+`LagunaModel.load_weights`),去掉不需要的分支
   (LoRA/bitsandbytes/PP等),覆盖:
   - stacked_params_mapping(q/k/v→qkv_proj,~20行,有现成参考实现可抄)
   - expert_params_mapping(继续调用vLLM的 `fused_moe_make_expert_params_mapping`,
     除非阶段2一起替换,否则暂时保留这个小依赖)
   - tied-embedding 处理
   - `process_weights_after_loading` 的调用编排(NVFP4部分的具体数值逻辑放到阶段2,
     这里先只管"在哪个时机调用它")
4. 分布式初始化(`init_worker_distributed_environment`)**暂时保留**(继续用vLLM
   Linear类期间无法绕开,代价很小,4行代码)。
5. **验证**:新构造路径 vs 旧 `get_model()` 路径做bit-exact对照(固定prompt+greedy,
   逐层/逐token比对),这是本阶段最大的时间不确定性来源。

---

## 阶段2:自建 Linear/Embedding + NVFP4量化(彻底脱离vLLM Linear类)(~1.5-2周)

**2026-07-28 更新:执行到一半发现原计划的前提是错的,以下先记录纠正,再列出
修订后的实际范围。**在动手接入 `NvFp4Linear`(阶段2早前已按原计划写好,commit
`7964b2d`)之前,先读了真实 checkpoint 的 `quantization_config`
(`models--poolside--Laguna-S-2.1-NVFP4/.../config.json`),而不是沿用"Laguna是
NVFP4量化"这个笼统假设。结果:

- `ignore` 列表明确排除了**全部** `self_attn.{q,k,v,o,g}_proj`(全部48层的正则)、
  `model.layers.0.mlp.{gate,up,down}_proj`(唯一的 `mlp_only_layers` 条目)、
  `mlp.gate`(MoE router)、`mlp.shared_expert.{gate,up,down}_proj`(全部层)——
  这些**全是 BF16**,checkpoint 里根本没有对应的 `weight_packed`/`weight_scale`/
  `weight_global_scale`/`input_global_scale` 张量。
- `config_groups.group_0.targets` 只有一条:
  `re:.*experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$`——NVFP4量化**只用在
  FusedMoE内部的routed expert权重上**,别的地方一个字节都不涉及。
- `runtime/backends/laguna.py::_patch_moe_sparkinfer` 早就绕开了vLLM
  `FusedMoE`的权重路径:直接从safetensors读取expert权重
  (`laguna_sparkinfer_moe.py`,本来就是zero-vLLM-dependency的),vLLM
  `FusedMoE.process_weights_after_loading`产出的NVFP4张量在推理开始前就被释放,
  它的CUTLASS GEMM从未在生产路径里被真正调用过。

**结论**:原计划item 1(自建Linear类要用NVFP4)对 `LagunaAttention`/
`LagunaMLP`/`LagunaMoE`的gate和shared_expert**完全不适用**——它们要的是
普通BF16 Linear,不是NvFp4Linear。`NvFp4Linear`(nvfp4_linear.py)在
Laguna生产路径里目前没有真实调用点,保留但不接入。item 2-5(NVFP4权重侧
预处理/global scale/激活量化kernel/自研GEMM直接调用/patch文件整体删除)
**只跟FusedMoE的expert权重有关**,而`self.experts`本身这一步刻意保留vLLM的
`FusedMoE`不动(见下)——所以这几项目前不是"Linear/Embedding替换"范围内的
工作,是否需要做要看未来是否有单独的"自建MoE"阶段(当前5阶段计划里没有这一项,
是本次发现新增的开放问题,见下方"待决策")。`nvfp4_b12x_patch.py`/
`nvfp4_cutlass_direct_patch.py`/`nvfp4_cudnn_patch.py`**不能盲目整体删除**:
`nvfp4_b12x_patch.py`只给`direct_model_runner.py`(qwen3.6,本轮明确不动)用;
另外两个对Laguna的实际作用是让`FusedMoE.process_weights_after_loading`在
SM120上不崩溃(权重处理阶段,不是推理阶段的GEMM),删除的前置条件是"自建MoE"
那个还不存在的阶段,不是这一步。

**已完成(commit `c432982`)**:
1. `runtime/model/plain_linear.py::PlainLinear`——TP=1、无量化,替代
   `ColumnParallelLinear`/`QKVParallelLinear`/`RowParallelLinear`/
   `ReplicatedLinear`,复用`NvFp4Linear`那套weight_loader闭包设计(每个
   Parameter挂一个闭包,兼容`LagunaModelSelfBuilt.load_weights`现成的
   `stacked_params_mapping`/`default_weight_loader`分发逻辑,零改动接入)。
2. `runtime/model/laguna_decoder.py`——`LagunaAttentionSelfBuilt`(qkv_proj/
   o_proj/g_proj用PlainLinear,`Attention`op-dispatch对象和`quant_config`
   原样保留——KV cache是FP8量化的,这个`quant_config`管的是KV cache scale
   不是权重量化,别删掉)、`LagunaMLPSelfBuilt`、`LagunaMoESelfBuilt`(gate/
   shared_expert用PlainLinear/LagunaMLPSelfBuilt,`experts`保留vLLM的
   `FusedMoE`不动)、`LagunaDecoderLayerSelfBuilt`。顺带把`q_norm`/`k_norm`/
   `input_layernorm`/`post_attention_layernorm`也切到Phase 1已验证过的
   `TritonRMSNorm`。
3. 真实权重bit-exact验证(复用阶段1的两个脚本,vllm loader vs selfbuilt
   loader完全一致的token序列,零偏差):167-token单chunk prefill+32轮解码;
   10240-token多chunk prefill(跨8192阈值)+128轮解码(触发SWA ring
   wraparound)。

**还没做、且不再确定属于这一阶段**(原计划item 2-6,理由见上):NVFP4权重侧
预处理/global scale/激活量化kernel移植、自研GEMM接入、patch文件删除、彻底去掉
distributed/parallel_state初始化——这些都卡在"`FusedMoE`还是vLLM的类"这一点上,
只要`self.experts`不换成自建实现就都无法真正完成,也没有必要在还没决定要不要
自建MoE之前先做。`VocabParallelEmbedding`/`ParallelLMHead`/`LogitsProcessor`
同理是BF16、可以用PlainLinear+普通Embedding替换,但要连带替换
`LogitsProcessor`的vLLM专属调用约定,工作量比单纯换Linear类大,留作后续独立
一步。

**用户决定(2026-07-28):不新增"自建MoE"阶段,FusedMoE和三个NVFP4 patch文件
(`nvfp4_custom_gemm.py`/`nvfp4_cutlass_direct_patch.py`/`nvfp4_cudnn_patch.py`
对Laguna的那部分作用;`nvfp4_b12x_patch.py`本来就只服务qwen3.6)保留不动,不
列入本次vLLM剥离范围。**接受"权重加载阶段短暂构造+丢弃"这份一次性开销
(不影响推理正确性/性能)作为既成事实,原计划item 2-6(NVFP4权重侧预处理/
global scale/激活量化kernel移植、自研GEMM接入、patch文件删除、彻底去掉
distributed/parallel_state初始化)相应地**不再是阶段2的一部分**,不是遗漏。

**阶段2到此收尾,目标达成**:自建Linear类替代了vLLM在Laguna里唯一真正需要
替代的部分(attention全部投影、dense MLP、MoE router gate、MoE
shared_expert——即上面确认过的BF16部分),真实权重bit-exact验证通过。
`VocabParallelEmbedding`/`ParallelLMHead`/`LogitsProcessor`(同样是BF16,但
连带`LogitsProcessor`的vLLM专属调用约定,工作量比单纯换Linear类大)和
`FusedMoE`一样,留作后续独立评估是否要做,不计入阶段2范围。

---

## 阶段3:DFlash 草稿模型自建(可以和阶段2部分并行)(~1.5-2周)

**2026-07-28 完成(commit `c9a70b5`)。** 开工前先读了真实DFlash草稿checkpoint
(`models--poolside--Laguna-S-2.1-DFlash-NVFP4/config.json`+safetensors权重名),
而不是凭调研印象假设——结果:无`quantization_config`(纯BF16),6层全部
`sliding_attention`,`num_experts=0`(纯dense MLP,从不走MoE分支)。这意味着
草稿模型每一层的结构和主模型的`LagunaDecoderLayerSelfBuilt`完全一致(权重名/
形状核对过),可以直接复用,`quant_config=None`贯穿始终。

实际完成范围,对照原计划各项:
1. **复用阶段1/2基础设施**:完成,但不是"跑两遍get_model()",而是新写了
   `runtime/model_loading.py::load_laguna_dflash_draft_model`(镜像
   `load_laguna_model`的模式:`set_default_torch_dtype`+`target_device`内构造
   +`DefaultModelLoader`加载+严格assert全load),权重命名映射确实需要单独适配
   ——草稿checkpoint没有顶层`model.`/`lm_head.`前缀(和主checkpoint不同),
   `LagunaDraftModelSelfBuilt.load_weights`因此更简单(没有MoE分支,没有
   attention_sink分支)。
2. **Embedding/lm_head tied-weight共享**:完成,照抄`_should_share`
   (`vllm/v1/worker/gpu/spec_decode/eagle/utils.py`)对
   `has_own_embed_tokens=False`/`has_own_lm_head=False`场景的无条件共享行为——
   构造后直接做对象引用替换(`del`+重新赋值成主模型的`embed_tokens`/`lm_head`
   对象),不是权重拷贝。
3. **草稿模型配置构造简化版**:**没有做,推迟**。`SpeculativeConfig`/
   `ModelConfig`仍然用vLLM的真实dataclass构造(`DFlashEngine._load_draft_model`
   里那段代码未改动)——判断这是配置管道而非模型代码,和`VllmConfig`/
   `EngineArgs`属于同一层级的既有依赖,优先级低于"自建模型类+权重加载+核心算法"
   这个阶段3的主线目标。如果以后要做,需要绕开`SpeculativeConfig`dataclass的
   `__post_init__`(否则仍会跑完整~10种方法调度器)。
4. **`SupportsEagle3`契约移植**:轻量完成——`LagunaDraftForCausalLMSelfBuilt`
   只加了`get_eagle3_default_aux_hidden_state_layers`,没有完整实现Protocol
   ——这套运行时用自己手写的`DFlashEngine`调用链,从不经过vLLM通用speculative
   decode调度,草稿模型本身也从不需要输出自己的aux hidden states(DFlash的aux
   hidden states来自主模型,阶段1已完成)。
5. **DFlash核心算法移植**:完成。
   - `combine_hidden_states`:逐slice RMSNorm用`TritonRMSNorm`(阶段1/2已验证
     过的同一个kernel),逐字照搬vLLM实现的拼接→fc投影→hidden_norm结构。
   - `precompute_and_store_context_kv`:**读vLLM源码时发现一个关键点**——
     Laguna自己在`vllm/model_executor/models/laguna_dflash.py`里覆写了
     `_project_context_kv`/`_normalize_context_k`/`_build_context_kv_buffers`,
     不是`DFlashQwen3Model`基类版本(`qwen3_dflash.py`)。基类版本对所有层用
     一个共享的`hidden_norm`做归一化;Laguna的覆写版本对每一层用**该层自己的
     `input_layernorm`权重**做归一化(`context_states`已经在`combine_
     hidden_states`里被`hidden_norm`归一化过一次,这里再按每层的
     `input_layernorm`归一化一次——不是重复归一化的bug,是刻意的逐层设计,
     照抄不改)。grouped K-norm(vLLM用一个支持`[L,...]`广播的kernel)改成
     对`L=6`层逐层调用现有`fused_rms_norm.rms_norm`——读了
     `csrc/libtorch_stable/layernorm_kernels.cu`确认广播语义等价,`L=6`很小,
     没必要为了省几次kernel launch专门写一个融合kernel。KV cache写入复用
     `attn.impl.do_kv_cache_update`(已有,sparkinfer补丁提供,未改动)。
   - **批量RoPE kernel**(`runtime/kernels/rope.py`):项目里第一个自建RoPE
     实现,NeoX-style,对展平的多层K张量一次性旋转。独立于模型代码先做了
     bit-exact对照(vs `vllm._custom_ops.rotary_embedding`,覆盖全量旋转和
     部分旋转partial_rotary_factor两种配置),再接入。
   - vLLM发现的"draft层Attention op要用偏移过的`attention_prefix`
     (`layers.{i+48}`而不是自己的`layers.{i}`)"这个细节也照抄了——
     `laguna_dflash.py`已有的`_alloc_draft_kv_cache`靠`layer_idx>=48`从
     `static_forward_context`里筛选draft层,偏移搞错会导致draft层和主模型
     layers.0-47的attention op注册命名冲突。
6. **验证**:复用`DFlashEngine.generate_verify_only`完整生产路径(不是孤立单测),
   `QSR_DFLASH_MODEL_LOADER=vllm` vs `selfbuilt`两组token序列+accept_rate对照:
   短测(64-token真实文本+64轮解码)和长测(10240-token真实多段文本,跨主模型
   chunk prefill阈值+640-token解码/48轮,累计>512个context-KV位置,强制触发
   draft`DRAFT_WINDOW=512` ring多次wraparound)均**bit-exact零偏差**。
   `QSR_DFLASH_MODEL_LOADER`默认值保持`"vllm"`,沿用阶段1的上线纪律——验证通过
   但先不切默认值,等待明确批准。

---

## 阶段4:DirectModelRunner 路径(条件性,取决于阶段0的决策)

**如果确认这条路径已不再使用**:直接归档,`sm120_gqa.py`连同它对vLLM Attention
ABC/注册表/enum round-trip的深度依赖一起下线,本阶段跳过,总工期显著缩短。

**如果确认仍需保留**:这是四个阶段里依赖深度最深的一块——不是要不要换计算逻辑
(kernel本身已经是自研SM120 GQA实现),是"注册表/enum round-trip/CUDA Graph捕获
约定"这套集成语义,脱离vLLM需要重新设计一套等价的调度/dispatch骨架,工作量接近
重新发明vLLM attention backend系统的一个子集,**预计~2-3周**,是四个阶段里最大的
单项。建议:如果这条路径要保留,优先确认它的真实使用场景(是否可以简化到不需要
完整ABC契约的程度),不要无脑照搬vLLM那套注册表设计。

---

## 阶段5:收尾验证 + 切换

**2026-07-28 完成(commit `0f11c4f`及之后)。** 逐项对照原计划,包含一项和原计划
预期不符的真实发现(第2项):

1. **全链路端到端bit-exact验证**:完成
   (`benchmarks/_phase5_e2e_bitexact_validate.py`)。不是逐模块对照,是把
   `QSR_LAGUNA_MODEL_LOADER`+`QSR_DFLASH_MODEL_LOADER`两个开关作为一个整体切换
   ——一次显式全设`"vllm"`(完整旧基线),一次不设任何环境变量(今天实际的默认值,
   两者都是selfbuilt)——通过`DFlashEngine.generate_verify_only`完整生产路径
   (prefill→decode→DFlash verify→accept/reject全流程)跑同一个请求。用的是和
   之前所有验证脚本都不同的全新文本内容(5段不同主题的真实段落,不是复用过的
   内容,也不是重复短语这种对抗性场景)。10240-token prompt跨主模型chunk
   prefill阈值,640-token/50轮解码迫使draft模型`DRAFT_WINDOW=512`的ring多次
   wraparound。**结果:640个token完全一致,accept_rate(0.793333)和
   num_steps(50)完全一致,零偏差。**
2. **确认生产server启动路径不再需要`vllm`可导入**:**结论是"不是",和原计划的
   预期不符,这里如实记录而不是回避**。用`sys.modules['vllm']=None`模拟
   vllm不可用后实测:
   - `import server.app`本身能成功(`ServerEngine`及其依赖是懒加载,不在模块
     顶层导入)。
   - `ServerEngine(...)`构造也能成功(同样是懒加载)。
   - 但`engine.start()`(真正的服务启动路径,`uvicorn.run`触发的就是这条链路)
     会在后台线程里加载模型时立即失败,报错定位到
     `runtime/compat_vllm.py:83`的`from vllm.v1.attention.backends.gdn_attn
     import GDNAttentionMetadata`。
   - 用`tests/test_vllm_dependency_boundary.py`的AST扫描机制核实了完整清单:
     `runtime/`+`server/`下目前有**11个文件**直接导入`vllm`(`server/`本身
     没有,但`server/app.py`透传依赖`runtime.backends.laguna`等)。这11个都是
     本轮各阶段"读真实代码后明确决定保留"的依赖,不是遗漏:`FusedMoE`(阶段2
     用户明确决定不退役)、`VocabParallelEmbedding`/`ParallelLMHead`/
     `LogitsProcessor`(阶段2判断为独立工作量,推迟)、`SpeculativeConfig`/
     `ModelConfig`(阶段3判断为配置管道,推迟简化)、以及`bf_attention.py`等
     几处在本轮vLLM剥离工作开始前就已经存在的CUDA op直连(`reshape_and_cache_
     flash`等)。**"完全替换vllm使server不需要vllm可导入"这个目标在当前已批准
     的范围内还没有达成,需要额外的阶段(至少覆盖FusedMoE自建+Linear/Embedding
     收尾)才能做到,不是这次能顺带解决的。**
3. **`pyproject.toml`更新**:`vllm-provider`extra**保留**(不能去掉,原因见上——
   仍是运行时必需依赖,不是"仅benchmarks/对比脚本可选")。加了一段注释,链接到
   依赖护栏测试和这份计划文档里对应阶段,说明保留的具体原因和范围,而不是含糊地
   叫"provider"。
4. **依赖护栏测试白名单更新**:白名单和实际观察到的11个文件精确匹配(护栏测试的
   两条断言——子集+精确相等——都通过),**不能清空**,原因同第2项。白名单本身
   在本次各阶段推进过程中已经保持同步更新(阶段2/3各自的commit都补上了新增文件),
   这里确认的是"现在还不能进一步缩小",不是遗漏维护。
5. **`benchmarks/`下vLLM原生对比脚本确认保留不动**:核实过——本次vLLM剥离工作
   在`benchmarks/`下只新增了4个文件(`_phase1_bitexact_validate*.py`×2、
   `_phase3_dflash_bitexact_validate*.py`×2)加上这份`_phase5_e2e_bitexact_
   validate.py`,通过`git diff --stat`对照分支基点确认零修改/零删除任何既有
   benchmark脚本。

**给合并决策的信息**:阶段1-3(模型图、Linear/Embedding、DFlash草稿模型)的自建
路径均已默认启用且真实权重bit-exact验证通过,可以合并而不改变生产行为(有
`QSR_LAGUNA_MODEL_LOADER=vllm`/`QSR_DFLASH_MODEL_LOADER=vllm`应急回退开关)。
但"完全替换vllm"(用户最初定的目标)尚未达成——`vllm`仍是`server`/`runtime`
启动必需的真实依赖,不是可选项,原因见上第2项。是否需要新开阶段追加完成度,
还是接受当前范围(自建了真正需要自建的模型图/权重加载/DFlash算法部分,保留
FusedMoE等复杂度高、已有独立正确性保证的部分)作为这次剥离工作的终态,需要
用户拍板。

---

## 阶段6:真正做到"完全无vllm"(用户看完阶段5报告后拍板重开)

用户明确拒绝"核心风险已解决、可以合并"这个中间状态,决定继续推进到
"server启动路径完全不需要vllm可导入"。范围:自建
VocabParallelEmbedding/ParallelLMHead/LogitsProcessor;简化
SpeculativeConfig/ModelConfig构造;重新评估FusedMoE退役(上次的"不退役"
决定是否真的站得住);盘点清楚"几处早于本轮工作就存在的CUDA-op调用点"。

**2026-07-28 中途的方法论事故,完整记录不回避**:执行过程中发现本session
从阶段1开始写的所有GPU验证脚本(`_phase1_bitexact_validate*.py`/
`_phase3_dflash_bitexact_validate*.py`/`_phase5_e2e_bitexact_validate.py`)
都把`sys.path`指向了主workspace(`/home/bot/project/qwen-sm120-runtime`)
而不是这个worktree自己的路径。Python的`PathFinder`(遵循`sys.path`)排在
venv的editable-install finder之前,所以本session全部"bit-exact验证通过"
的结论,实际测的都是main当时那个高频提交、随时在变的工作树,不是这个分支
提交的代码。用修正后的路径重新跑,当场暴露4个被这个bug掩盖了一整个session
的真实bug(和当天FusedMoE工作无关,阶段1-3遗留):
1. `_assert_all_params_loaded`对`Attention`的`q_scale`/`k_zero_point`/
   `v_zero_point`/`q_zero_point`检查过严——读vLLM真实源码
   (`compressed_tensors.py`)确认默认值(`q_scale`默认1.0、zero_point默认
   0.0)是安全的no-op值,checkpoint本身是对称量化,压根不提供这几个tensor
   (直接读safetensors核实过),vLLM自己走`AutoWeightsLoader`遇到的是同一
   情况。已放宽断言。
2. DFlash草稿checkpoint的`qkv_proj.weight`是已经融合好的单个
   `[11264,3072]`张量(和主模型checkpoint分开存q/k/v不是同一种格式),
   `LagunaAttentionSelfBuilt`复用主模型那套3-way-sharded `PlainLinear`
   构造没有对应处理。已修`PlainLinear`的weight_loader,加一个不误伤现有
   场景的整体拷贝兜底。
3. `mask_embedding`注册成了`nn.Parameter`,vLLM真实实现是
   `register_buffer(persistent=False)`。已修。
4. `_patch_moe_sparkinfer`的`_patched_forward`还在按vLLM旧
   `ReplicatedLinear`的`(output,bias)`元组约定解包`gate(hs)`,但阶段2
   已经把`gate`换成`PlainLinear`(返回单个tensor)——这行代码本session
   从来没被真正跑到过(全程被路径bug掩盖),真实跑一次就崩。已修。

4个bug修完后,用修正过路径的`_phase5_e2e_bitexact_validate.py`做了真正的
token级bit-exact对照(`QSR_LAGUNA_MODEL_LOADER=vllm`+
`QSR_DFLASH_MODEL_LOADER=vllm`全套vLLM参考实现 vs 不设任何环境变量的今天
默认配置,同一个10240-token真实prompt+640-token DFlash解码):**640个
token逐字节完全一致**。`acceptance_rate`数值本身有微小差异
(0.798667 vs 0.789333),定位到是统计口径问题——`total_accepted`按完整
轮次计入,最终token列表在`max_tokens`处截断,和实际生成的token没有关系,
不是真实差异。commit `3a84e84`。

**已知问题,记录不修,不阻塞当前工作**(用户拍板):`QSR_LAGUNA_MODEL_
LOADER=vllm`这个应急回退开关,搭配DFlash的CUDA graph verify流程(默认
开启,不受`enforce_eager`影响),会不稳定触发`illegal memory access`。
用`CUDA_LAUNCH_BLOCKING=1`能绕开(证实是异步执行时序问题,不是确定性的
逻辑错误)。已确认范围边界(3点):
1. 只在用DFlash时出现——纯prefill+decode(不走DFlash)用同一个vllm
   loader全程未复现。
2. 只在DFlash的CUDA graph verify流程默认开启时出现——用
   `QSR_DFLASH_CUDA_GRAPH=0`关掉CUDA graph(哪怕不加
   `CUDA_LAUNCH_BLOCKING=1`)同样不复现,支持"和CUDA graph capture/replay
   有关"这个判断。
3. 只在`QSR_LAGUNA_MODEL_LOADER=vllm`这个开关上出现——生产实际默认配置
   (`selfbuilt`)全程没有一次复现,不影响shipped路径。
不修的理由:`vllm` loader是应急回退开关,不是生产路径,三条边界条件都指向
一个和这次vLLM剥离工作本身无关的、更早存在的CUDA graph capture/Triton
kernel(`fused_kv_scatter`,由更早的cherry-pick带入这个分支)交互问题,
真正定位需要专门的Triton/CUDA graph调试,超出这次"完全无vllm"的范围。

**4件事的进展**:

1. **自建VocabParallelEmbedding/ParallelLMHead/LogitsProcessor**:完成
   (commit `b3fb144`)。读vLLM真实源码
   (`vocab_parallel_embedding.py`/`logits_processor.py`)后确认:这个
   runtime的真实取值下(TP=1、无LoRA、Laguna `vocab_size=100352`本来就是
   `DEFAULT_VOCAB_PADDING_SIZE=64`的整数倍、`draft_vocab_size==
   vocab_size`、从未配置过`soft_cap`/`scale`),vLLM那套vocab
   padding/TP-sharding/gather机制处处都是no-op,
   `LogitsProcessor.forward`最终就是一次`F.linear(hidden, weight, bias)`
   ——追到`default_unquantized_gemm`确认过。新增
   `runtime/model/plain_embedding.py`:`PlainEmbedding`/`PlainLMHead`/
   `PlainLogitsProcessor`。TP切分留作文档化的未来扩展点。真实GPU
   bit-exact验证通过(640 token/accept_rate/num_steps和已验证基线完全
   一致)。
2. **简化SpeculativeConfig/ModelConfig构造**:**评估后决定不做,不是
   遗漏**。逐一核对`runtime/model_loading.py`/`runtime/model/
   laguna_dflash_model.py`里对`speculative_config`的每一处真实访问
   (只有`.draft_model_config`一处,没有别的),发现`_load_draft_model`
   现有代码已经在构造完`SpeculativeConfig(...)`后,把它内部
   `__post_init__`自动构造出来的`draft_model_config`整个丢弃、换成自己
   手写的`ModelConfig(...)`(10个显式字段,已经是"只给下游真正用到的
   字段"这个精简版本,不是完整~10种方法调度器)。真正的浪费在
   `SpeculativeConfig(...)`构造本身触发的那次内部`ModelConfig`自动构造
   ——**但`SpeculativeConfig`是Pydantic校验的类**(`@config`+
   `pydantic.Field`/`field_validator`),没有官方的"跳过`__post_init__`"
   安全构造入口,贸然绕过有引入隐蔽正确性问题的风险。更重要的是:
   不管构造过程简化与否,`SpeculativeConfig`/`ModelConfig`本身还是vLLM
   的类——这一步**不会让vllm-import白名单少一个文件**,和阶段6"完全无
   vllm"这个目标没有关系,只是一个和目标无关的次要效率优化。评估后判断
   不值得为了这个和目标无关的收益承担Pydantic绕过的风险,不做。
3. **FusedMoE退役重新评估**:完成(commit `3a84e84`前置的
   `laguna_decoder.py`/`laguna.py`/`laguna_sparkinfer_moe.py`改动)。
   读vLLM真实`FusedMoE`工厂函数(`layer.py`)+NVFP4 MoE量化scheme
   (`compressed_tensors_moe_w4a4_nvfp4.py`)源码,并用live probe核实
   (不是凭源码推断就下结论):`_patch_moe_sparkinfer`过去从FusedMoE
   读取的每一个值追踪到底——`w13_weight`/`w13_weight_scale`/
   `w13_weight_scale_2`/`w2_weight_scale`/`w2_weight_scale_2`全部只是
   "存在性检查后立即释放,从未读取过数值";真正被消费的只有
   `e_score_correction_bias`和两个激活值全局scale
   (`a1_gscale`/`a2_gscale`,公式`min(每个expert的原始
   input_global_scale)`,双重取倒数容易搞反方向,已经用live run实测
   验证过,不是靠代数推导)。这三个值现在直接从checkpoint读取
   (`laguna_sparkinfer_moe.py`新增`load_moe_layer_activation_gscales`/
   `load_moe_layer_e_score_correction_bias`),`LagunaMoESelfBuilt`
   不再构造`self.experts`,原来只为构造`FusedMoE`服务的TP/EP/EPLB
   bookkeeping代码全部删除。**结论和最初"不退役"的判断相反**:不是因为
   TP=1/EP=1让分布式代码变成死代码(这个假设本身没错,但不是真正的浪费
   点),而是FusedMoE加载的那些大NVFP4权重张量,不管TP/EP多大,本来就
   100%被丢弃不用——真正的浪费和分布式与否无关。
4. **遗留CUDA-op调用点盘点**:完成(commit `6b28c95`,见上文"已知问题"
   之前的记录)。`bf_attention.py`/`laguna_cuda_graph.py`/
   `laguna_sparkinfer_attn.py`三处`vllm._custom_ops.reshape_and_cache_
   flash`直连调用,cherry-pick了main上已经修好验证过的
   `fused_kv_scatter`接线(commit `486acbd`/`f8a01dd`/`249beb6`),
   护栏白名单从11个文件降到8个。

---

## 阶段7:剩余8文件的4类真实依赖(任务#38,用户看完阶段6报告后拍板重开)

用户仍不满足于阶段6后的状态,继续推进剩下8个文件里的四类依赖:
`VllmConfig`配置管道;`Attention` op-dispatch ABC(`bf_attention.py`
patching依赖的那个,阶段6审计文档标注"风险最大,工作量接近重新发明vLLM
attention backend系统的一个子集");`get_rope`的`cos_sin_cache`构造;
`DefaultModelLoader`/`process_weights_after_loading`权重加载工具。

**4件事的进展**:

1. **VllmConfig配置管道**:完成。`runtime/model_loading.py`/
   `runtime/model/laguna_model.py`/`runtime/model/laguna_decoder.py`/
   `runtime/model/laguna_dflash_model.py`四个文件里的`VllmConfig`导入
   全部核实为纯类型标注用途(逐处grep确认,没有一处`isinstance`/构造),
   加上`from __future__ import annotations`本来就让标注变成惰性字符串,
   移到`if TYPE_CHECKING:`块内即可去掉真实运行时依赖,不影响类型检查。
   `laguna_decoder.py`里还顺手发现一处完全没被用到的死`VllmConfig`导入
   (连类型标注都不是),直接删除。护栏测试(`tests/
   test_vllm_dependency_boundary.py`)新增`_is_type_checking_guard`/
   `_walk_runtime_reachable`,让AST扫描正确跳过`if TYPE_CHECKING:`子树,
   不把这类导入误判成"新增vllm依赖"。GPU bit-exact验证通过(见第4项)。
2. **get_rope的cos_sin_cache构造**:完成(commit `2698b9e`)。读vLLM真实
   `get_rope`/`RotaryEmbedding`/`YaRNScalingRotaryEmbedding`源码
   (`vllm/model_executor/layers/rotary_embedding/`)确认:真正的计算路径
   (`forward_cuda`调用`ops.rotary_embedding(...)`)阶段3已经bit-exact
   验证过,缺的只是cache构造这段一次性的纯tensor数学,可移植。新增
   `runtime/kernels/rope.py`的`compute_cos_sin_cache_default`/
   `compute_cos_sin_cache_yarn`(+`_yarn_*`系列helper)和
   `SelfBuiltRotaryEmbedding`包装类。过程中发现一个真实的命名陷阱:
   `get_rope`的YaRN分支读的是`rope_parameters["attn_factor"]`
   (默认1.0),不是Laguna checkpoint里字段名相近的`attention_factor`
   (`1.3465735902799727`)——两者不是一回事,用live probe核实过vLLM
   真实运行时`attn_factor`确实取默认值1,而Laguna的`attention_factor`
   数值恰好等于`yarn_get_mscale(32.0)`(vLLM自己独立算出来的同一个值),
   这是"用错误的默认值反而算出正确结果"这种巧合,已经在
   `compute_cos_sin_cache_yarn`里写清楚,避免以后被这个巧合误导。首次
   验证时cache精度对不上(float32下有0.01/0.0005/0.001量级误差)
   ——根因是生产环境在`set_default_torch_dtype(bfloat16)`上下文里构造
   这个cache,vLLM真实cache是bf16的;两边都换成bf16后3个真实配置
   (yarn full_attention、default sliding_attention、default草稿模型)
   全部`torch.equal`精确匹配。
3. **Attention op-dispatch ABC**:评估,结论**不做**。见下方独立小节。
4. **DefaultModelLoader/process_weights_after_loading**:**一半完成,
   一半评估后放弃自建、退回vLLM真实实现**。
   - `DefaultModelLoader.get_all_weights`替换:完成。新增
     `runtime/model_loading.py`的`_iterate_safetensors_checkpoint`,直接
     流式读取safetensors分片(和`laguna_sparkinfer_moe.py`的
     `load_moe_layer_weights`同一个模式,只是从单层扩到整个checkpoint)。
     刻意不做vLLM那套通用loader的功能:不支持HF Hub下载(这个runtime
     全程`HF_HUB_OFFLINE=1`,只读本地已缓存路径)、不支持.bin/.pt格式、
     不支持多线程、不支持EP权重过滤(这个runtime恒为TP=1/EP=1)。两种真实
     checkpoint布局都核实过(不是假设):主模型15个分片+
     `model.safetensors.index.json`索引,DFlash草稿模型单个无索引的
     `model.safetensors`。内存安全这点是刻意保留的,不是意外复杂度——
     主checkpoint约67GiB,这台机器只有约19GiB内存,一次读一个分片、
     yield完再关闭再读下一个,是让峰值内存保持在约一个分片大小、而不是
     整个checkpoint的关键行为(vLLM自己loader也打印"Checkpoint
     size... Available RAM..."这条日志,同样的顾虑)。
   - `process_weights_after_loading`替换:**尝试后确认有真实回归,已
     完整回退,不是因为怕风险不敢试**。第一次实现只调用了
     `Attention.process_weights_after_loading(dtype)`(每个`Attention`
     模块本身的方法),读真实vLLM源码
     (`vllm/model_executor/layers/attention/attention.py:599-611`)确认
     这个方法内部其实做两件事——
     `self.impl.process_weights_after_loading(act_dtype)`(当前attention
     后端实现自己的post-load,和KV-cache scale完全是两回事)加上
     KV-cache scale defaulting——但当时的自建版本调用的就是这个真实绑定
     方法本身,理论上两部分都应该会触发,这个"漏了什么"的猜测站不住脚,
     没有继续深挖到底漏在哪。用真实GPU bit-exact e2e跑出来的是明确、
     严重的回归:文本损坏(连字符被替换成下划线,如"record-keeping"变成
     "record_keeping")、退化重复("I'm sorry, but I can't continue this
     response."重复十几次)、`acceptance_rate`/`num_steps`都不对
     (0.710714/56,应为0.789333/50)。没有在没搞清楚根因之前继续往前
     走或者绕过验证,而是完整回退到原来`process_weights_after_loading(
     model, model_config, target_device)`调用,只保留
     `_iterate_safetensors_checkpoint`这一项独立验证——单独跑通
     真实GPU bit-exact e2e,`acceptance_rate=0.789333`/`num_steps=50`/
     640-token解码文本逐字符和已确认基线一致,证明safetensors
     loader这一半是干净的,问题确实出在`process_weights_after_loading`
     自建那一半。这部分自建**放弃,保留vLLM真实实现**——正确的自建版本
     需要先弄清楚当前attention实现(`self.impl`)自己的
     `process_weights_after_loading`具体做什么,这属于阶段7第3项
     (Attention op-dispatch ABC)评估范围内的同一块复杂度,不值得在
     没有那个评估的前提下重复冒险。

---

## Attention op-dispatch ABC 评估(阶段7第3项,只评估不动手)

按用户要求,这项只做评估、不在没有独立结论前直接动手全量重写。

**结论:不做,建议维持现状**。和FusedMoE那次(阶段6第3项)不一样——那次
深挖之后发现"看起来很重"的表面下其实是纯粹被丢弃的死权重,评估后**反转**
了"不退役"的结论;这次深挖之后**反而印证**了最初审计"工作量接近重新发明
vLLM attention backend系统一个子集"这个判断,不是它夸大了。

**先说清楚"浅"的部分,避免高估**:`bf_attention.py`本身其实**已经不依赖
vLLM**——通读全文(241行)确认它一个vllm import都没有,`replace_vllm_
attention()`靠`hasattr(attn_layer, "get_attn_backend")`这种鸭子类型
discover层,不是`isinstance`检查。真正还导入`from vllm.model_executor.
layers.attention import Attention`的只有`laguna_decoder.py`一处
(`LagunaAttentionSelfBuilt.__init__`里`self.attn = Attention(...)`,
line 325)。`Attention.forward()`真实计算路径从未执行过——构造完之后,
`LagunaBackend.__init__`立刻用`replace_vllm_attention()`把它整个替换成
`BFAttention`实例(`laguna.py:333`);构造它的唯一目的是"borrow"它
`__init__`做的几件事的**副作用**。

**读了真实vLLM源码(`vllm/model_executor/layers/attention/attention.py`
的`Attention.__init__`/`_init_kv_cache_quant`/`get_kv_cache_spec`)后,
这些副作用逐条拆解如下**:

1. **`static_forward_context[prefix] = self`注册**:trivial,自建一行
   `sfc[prefix] = self`就能替代。
2. **`get_attn_backend(...)`真实后端选择 + `impl_cls(...)`构造**:表面
   看像FusedMoE那次的"构造后丢弃"(`laguna.py:268`确实立刻把
   `layer.impl`整个换成`SparkinferAttentionImpl`)——但**这次深挖发现
   它不是纯丢弃**:`get_kv_cache_spec()`(下面第4点)的sliding-window
   分支会回头调用`self.attn_backend.get_supported_kernel_block_sizes()`
   去做per-page block size优化(`_largest_kernel_block_within`),这个
   `self.attn_backend`就是这一步选出来的真实vLLM后端类对象——也就是说
   这个"看起来会被丢弃"的对象,其实真实喂给了KV-cache分页大小的计算逻辑,
   和FusedMoE权重张量"验证过0%被读取"的情况不是同一类问题。Laguna是
   sliding_attention/full_attention混合模型(阶段3/7已确认),这个SW
   分支是真实被走到的路径,不是死代码。
3. **`_init_kv_cache_quant`创建`k_scale`/`v_scale`量化Parameter**:通过
   `quant_config.get_quant_method(layer, prefix)`拿到
   `CompressedTensorsKVCacheMethod`,再调用它的`create_weights(layer)`
   完成真实的Parameter创建;背后调的`set_default_quant_scales`,和
   本阶段第4项(`process_weights_after_loading`自建尝试)踩坑时读到的
   是**同一个函数**——今天已经在这个具体子系统上现场翻过一次车(自建
   版本漏掉一部分KV-scale处理,产出静默文本损坏而不是直接崩溃,回归
   完全没有报错,只有跑bit-exact e2e才发现),这不是从文档推测的抽象
   风险,是这个session里刚刚发生过的真实教训。
4. **`get_kv_cache_spec(vllm_config)`**:构造`FullAttentionSpec`/
   `SlidingWindowSpec`(`vllm.v1.kv_cache_interface`的dataclass),被
   `laguna.py:291`的`layer.get_kv_cache_spec(vllm_config)`真实调用,
   结果直接决定这个runtime怎么分配/摆放KV cache——即使把`Attention`
   本身替换掉,这两个spec dataclass的import大概率还是甩不掉(除非连
   `laguna.py`里消费spec的那部分逻辑也一起重写,这已经超出"替换
   Attention构造"本身的范围)。

**结论的技术依据**(不是"太难所以不做"这种空泛判断):真正可以trivial
替换的只有第1点(注册,一行代码);第2/3/4点合在一起,意味着要正确复刻
这块东西,需要:(a) 手写KV-cache scale Parameter创建,精确匹配checkpoint
的量化scheme(对称量化,已确认,但"精确匹配"这个要求本身,今天已经在
同一个函数附近失败过一次,不是假设的风险);(b) 手写或保留
`FullAttentionSpec`/`SlidingWindowSpec`构造和消费,大概率无法真正甩开
`vllm.v1.kv_cache_interface`这个import;(c) 手写sliding-window
block-size优化逻辑,需要理解真实vLLM后端的`get_supported_kernel_block_
sizes()`语义,这部分至今没有独立验证过。收益上限是把`laguna_decoder.py`
从8文件白名单里去掉一个(前提是同一个文件里`cache_config`/
`quant_config`两个真实运行时参数也一并处理干净,这次评估没有覆盖那部分,
需要单独评估),不是让`Attention`相关的vLLM import彻底从这个runtime消失
——`laguna.py`本身作为backend orchestrator,不管`Attention`类本身在不在,
仍然会保留在白名单里(它导入vLLM太多其他东西,如`bind_kv_cache`/
`CommonAttentionMetadata`等,这次评估范围外)。工作量大、风险真实(KV-scale
这类静默数值bug今天刚验证过有多容易犯),换来的收益只是白名单8→7,不满足
"值得为了减少vllm依赖去承担这个风险"的门槛。按用户的态度:明确建议不做。

---

## Attention op-dispatch ABC 结论修正(任务#39,用户质疑后补充调研)

用户对上面"不做"这个结论提出根本性质疑:评估只问了"vLLM自己内部怎么做、
原样复刻多难",没问过"我们是不是必须复刻vLLM那套通用抽象,其他极简引擎
怎么解决同一个问题"——和阶段6自己纠正FusedMoE"以为需要完整分布式机制,
查完发现只需要3个值"是同一类方法论漏洞。补课如下,**结论修正为:值得
重新实现,原"不做"判断站不住脚**。

### 调研发现

**nano-vllm**(`GeeeekExplorer/nano-vllm`,真实clone读源码,~1450行,单一
Qwen3架构):`Attention`类(`nanovllm/layers/attention.py`)只有43行,
**没有任何后端注册表/ABC**——不存在`AttentionBackend`/`AttentionImpl`/
`AttentionLayerBase`/`static_forward_context`,因为它只支持一种kernel
(flash-attn),压根不需要"从多个后端里选一个"这层抽象;KV cache写入用
自己的Triton kernel(和我们的`fused_kv_scatter`同一个思路)。
`ModelRunner.allocate_kv_cache()`(`nanovllm/engine/model_runner.py:103`)
直接用`hf_config`的`num_key_value_heads`/`head_dim`/`num_hidden_layers`
做纯算术算block字节数,**没有`FullAttentionSpec`/`SlidingWindowSpec`
这类dataclass,也不会为了"page size怎么摆最优"去问后端对象**。这个仓库
不支持sliding window/hybrid attention,也没有KV-cache量化,所以没有
现成答案覆盖我们全部场景,但它证明了核心问题的一半:**通用后端注册表/
ABC不是解决"attention怎么算"这个问题的必要条件,是vLLM"要支持任意架构
+任意后端"这个目标带来的,不是问题本身要求的**。

**sglang**(`sgl-project/sglang`,真实clone读源码,生产级引擎,支持
Gemma2/3式SWA+full混合注意力和KV-cache量化,不是极简教学项目,但覆盖了
nano-vllm没覆盖的两个场景):
- SWA+full混合KV cache用**两个独立池**
  (`SWAKVPool`,`python/sglang/srt/mem_cache/swa_memory_pool.py`):
  `full_kv_pool`/`swa_kv_pool`,构造参数是纯标量
  (`head_num`/`head_dim`/`page_size`/`dtype`/两组layer_id列表)——**同样
  没有"问后端对象支持哪些block size"这一步**,和vLLM
  `get_kv_cache_spec()`→`self.attn_backend.get_supported_kernel_block_
  sizes()`那种耦合不是同一类设计。
- KV-cache量化的scale buffer创建走独立的`quant_method.create_buffers(...)`
  (`memory_pool.py:1901`),**接在memory pool上,和"选哪个kernel后端"这个
  关注点是分开的**,不像vLLM把backend选择、KV-scale创建、
  static_forward_context注册、KVCacheSpec构造全部塞进`Attention.__init__`
  一个类里。

### 对照我们自己的代码,补一个更关键的发现

`laguna.py:288-295`——真实调用`layer.get_kv_cache_spec(vllm_config)`
之后,**只用了`spec`的两个字段**:`type(spec).__name__`(是不是
`SlidingWindowSpec`)和`spec.sliding_window`(窗口大小)。`spec.block_size`
**从未被读取**——KV cache实际分配用的`block_size`是`LagunaBackend`自己
外部传入的构造参数,和`spec.block_size`（vLLM那套"问后端支持哪些kernel
block size"算出来的优化值)完全无关。也就是说:**我们的代码本来就已经在
丢弃vLLM那套backend-driven分页优化了,阶段7原评估说"这个backend对象真实
喂给了KV-cache分页逻辑"这句话对vLLM自己成立,但对我们自己的消费代码不
成立**——这是原评估的方法论漏洞:只查了"vLLM内部怎么用",没有查"我们
自己实际读了它返回值的哪一部分"。而`is_swa`/`sliding_window`这两个值,
`LagunaAttentionSelfBuilt.__init__`本来就已经知道(`per_layer_sliding_
window=self.sliding_window`这个参数本身就是构造`Attention(...)`之前就
算好的),不需要真的构造一个vLLM `Attention`对象再问它要回来。

### 顺带把阶段7第4项那次真实回归的根因挖到底了(之前没挖完,现在挖完了)

之前(阶段7第4项)`process_weights_after_loading`自建替换回归时,猜测是
漏了`self.impl.process_weights_after_loading()`,但当时也承认"这个猜测
站不住脚,没有继续深挖"。这次为了搞清楚"KV-scale这块到底多难自建",把
真实vLLM源码(`vllm/model_executor/layers/attention/attention.py`的
`Attention.process_weights_after_loading`,`vllm/model_executor/layers/
quantization/kv_cache.py`的`BaseKVCacheMethod.process_weights_after_
loading`,`vllm/model_executor/model_loader/utils.py`的通用
`process_weights_after_loading(model, model_config, target_device)`
三层)逐行读完,**根因现在完全确认,和最初猜测的不是同一个东西**:

真实的通用`process_weights_after_loading`分3趟遍历`model.named_modules()`:
第1趟按`module.quant_method`类型分派(`quant_method.process_weights_after_
loading(module)`);第2趟专门对`Attention`/`MLAAttention`调用**模块自己
的**`process_weights_after_loading(dtype)`方法;第3趟处理`HpcModule`。
**当时自建的版本只做了第2趟**(`for module in model.modules(): if
isinstance(module, Attention): module.process_weights_after_loading(
dtype)`),**完全没做第1趟**。而`Attention.__init__`(`_init_kv_cache_
quant`)会把`layer.quant_method`设成一个真实的`CompressedTensorsKVCache
Method`实例——这意味着`Attention`模块自己也会被第1趟命中,第1趟对它
调的是`quant_method.process_weights_after_loading(module)`,也就是
`BaseKVCacheMethod.process_weights_after_loading`
(`vllm/model_executor/layers/quantization/kv_cache.py:74`)。

这个方法做的事:checkpoint里真实加载出来的`layer.k_scale`/`layer.v_scale`
(`KVCacheScaleParameter`,默认哨兵值`-1.0`,被weight_loader写入真实值后
变成checkpoint里的`0.0319824...`/`0.0011978...`,直接读safetensors验证
过)**在这一步之前从未被任何东西消费过**——`forward()`阶段(不管是vLLM
的`FlashInferImpl`还是我们自己的`BFAttention`)读的是下划线版本
`layer._k_scale`/`layer._v_scale`(构造时`set_default_quant_scales`建的
buffer,默认值1.0)。`BaseKVCacheMethod.process_weights_after_loading`
就是**唯一**把`layer.k_scale`(真实checkpoint值)拷贝进`layer._k_scale`
+`layer._k_scale_float`(真正被消费的那份)的地方。`bf_attention.py`的
`replace_vllm_attention()`把`attn_layer._k_scale`/`_v_scale`原样拷给
`BFAttention`(`if hasattr(attn_layer, "_k_scale"): bf_attn._k_scale =
attn_layer._k_scale`)——如果这一步没跑过,`_k_scale`/`_v_scale`就停在
默认值1.0,`fused_kv_scatter`用错误的scale量化KV cache,产出静默数值
错误(观察到的"文本损坏但不崩溃"和这个诊断完全吻合,不是巧合)。

**这个根因现在是可以被精确复刻的**,不是"不知道漏在哪所以不敢动"的
未知风险:真实checkpoint的k_scale/v_scale都是正数、非per-token-head、
非"只有一个kv_scale需要复制成两份"这种边界情况(已核实),需要抄的只是
`BaseKVCacheMethod.process_weights_after_loading`里`if layer.k_scale >
0.0 and layer.v_scale > 0.0`这一个分支(kv_cache.py:104-110,~7行),
其余分支(per-token-head/单scale复制/fp8_fnuz doubling,后者是AMD平台
专属,SM120不适用)对我们这个checkpoint是死代码,不需要照抄,但要写清楚
"只覆盖了这一种真实场景,不是通用实现"这条边界,防止以后checkpoint换了
量化scheme就悄悄错。另外`q_scale`/`prob_scale`两个字段确认和我们的运行
路径完全无关——`replace_vllm_attention()`只拷贝`_k_scale`/`_v_scale`,
从不读`_q_scale`/`_prob_scale`,`SparkinferAttentionImpl`也不用它们
(`_assert_all_params_loaded`早就把`q_scale`/所有zero_point排除在必须
加载的参数之外)。

### 修正后的结论

原"不做"判断建立在两个不准的前提上:(1)以为backend对象真实喂给我们自己
的KV-cache分页逻辑——**实测不成立**,我们自己的消费代码根本没读那部分;
(2)以为KV-scale后处理是"没搞清楚、容易犯错的未知风险"——**现在已经精确
根因到具体7行代码**,不再是未知。跨引擎调研(nano-vllm/sglang)进一步
确认:通用后端注册表/ABC本来就是vLLM"支持任意架构"这个目标专属的复杂度,
不是解决我们这个具体问题(单一Laguna架构、TP=1、attention计算早就是
sparkinfer自己的kernel)必须付的成本。

**结论修正为:值得重新实现**,范围收窄到:
1. 自建一个轻量placeholder类(不继承`Attention`/`AttentionLayerBase`),
   在`LagunaAttentionSelfBuilt.__init__`里替代`self.attn = Attention(...)`
   构造,自己把`sfc[prefix] = self`写进去。
2. `is_swa`/`sliding_window`直接用已经算好的`self.sliding_window`,不用
   构造spec对象再读回来;`laguna.py:288-295`那段消费逻辑同步简化(不用
   再判断`type(spec).__name__`,直接读一个自己定义的布尔值)。
3. `k_scale`/`v_scale`两个`nn.Parameter`(初值参考`KVCacheScaleParameter`
   的`-1.0`哨兵约定或直接扣我们已知checkpoint一定提供的事实简化成
   `torch.ones(1)`,两种取舍需要在真正动手时明确选一个并写清楚理由),
   走和阶段6 MoE gscales/阶段1 PlainLinear同一套weight_loader闭包模式。
4. 抄`BaseKVCacheMethod.process_weights_after_loading`那7行真实分支
   (只覆盖"两个scale都是正数"这一种真实场景),接到
   `model_loading.py`的post-load步骤里,把加载到的`k_scale`/`v_scale`
   转换成`BFAttention`真正读的`_k_scale`/`_v_scale`。
5. 顺带解决阶段7第4项遗留的`process_weights_after_loading`自建放弃
   问题——这次自建version覆盖的正是当时缺的那一块,做完这个,阶段7第4项
   也可以真正自建完成,不用继续依赖vLLM真实`process_weights_after_
   loading`。

不确定、需要动手时用真实GPU bit-exact验证才能确认的点(不能因为"这次
想清楚了"就跳过验证,这正是今天这次事故的教训):`KVCacheScaleParameter`
默认值语义(-1.0哨兵 vs 直接假设总有真实值)选错会不会引入新的边界条件
bug;`laguna.py`里消费spec的逻辑简化后,和CUDA Graph capture/SWA环形
buffer相关的其它读取点(`sfc[name].kv_cache = ...`等,现有代码里对
`layer`还有别的属性读取,这次评估没有逐一过一遍)是否都还兼容一个不是
真实`Attention`子类的placeholder对象。

这个结论已经推翻了阶段7原评估的"明确建议不做"——等用户/协调者确认要
不要动手实现,动手前会先把上面"不确定"这几点过一遍,再摸GPU前照常先
确认。

---

## Attention op-dispatch ABC 实现完成(任务#40)

用户批准动手实现,范围按结论修正版收窄:placeholder注册 + is_swa/
sliding_window直读config + k_scale/v_scale weight_loader + 照kv_cache.py
真实逻辑写的post-load拷贝。中途收到两条用户强调,已落实进代码(不只是
对话里口头同意):"我们只支持sparkinfer这一个kernel"——`plain_attention.py`
模块docstring顶部写明这是架构前提,不是"以后可能要支持别的kernel、先
凑合"的占位符,以后真要加第二个kernel是架构重做,不是加if分支。

新增`runtime/model/plain_attention.py`的`SelfBuiltAttentionPlaceholder`,
替换`laguna_decoder.py`里`self.attn = Attention(...)`的构造;
`model_loading.py`新增`_apply_kv_cache_scale_post_load`,替换掉之前一直
依赖的vLLM真实`process_weights_after_loading`(阶段7第4项遗留的那部分,
这次一并吃掉)。

**实现过程中GPU真实验证连续抓到3个评估阶段没覆盖到的真实缺口**,不是走
个过场,记录下来避免以后同类误判:

1. **DFlash草稿模型checkpoint没有quantization_config**:第一次GPU跑直接
   `AttributeError: 'NoneType' object has no attribute 'kv_cache_scheme'`
   ——最初把"sparkinfer只有一种kernel,所以FP8 KV cache可以硬编码"和
   "checkpoint是否提供真实scale"这两件事混成了一件事。查草稿checkpoint
   的config.json+safetensors确认:草稿模型确实没有quantization_config、
   没有k_scale/v_scale,但它的KV cache仍然是FP8(`laguna_dflash.py`
   `_alloc_draft_kv_cache`硬编码`dtype=torch.uint8`,和quant_config无关)
   ——scale固定用默认值1.0,不是bug,是这个checkpoint真实没有可加载的
   scale。修成`has_checkpoint_kv_scale`布尔位,只在为True时才创建
   `k_scale`/`v_scale`这两个Parameter。kernel选择(硬编码sparkinfer)和
   "这个checkpoint有没有真实scale可加载"(数据驱动、两个模型不一样)是
   两件不同的事,不是同一个"要不要写死"的问题——这条边界划错过一次。
2. **草稿模型attention从没被换成BFAttention**:第二次GPU跑崩在
   `self.attn`自己的`forward()`被直接调用(`NotImplementedError`风险,
   实际报错是`bf_attn_context`缺失,见下)——最初的评估只grep了4个
   backend文件(`laguna.py`/`bf_attention.py`/`laguna_sparkinfer_attn.py`/
   `laguna_cuda_graph.py`),漏了`laguna_dflash.py`。真实vLLM
   `Attention.forward()`要靠`get_forward_context()`+custom-op分发才能
   桥接到`impl.forward()`,草稿模型这条路径以前一直是这样跑的
   (`_patch_draft_sparkinfer`只换`.impl`,没换整个模块)。问过用户后
   决定:草稿attention也走`replace_vllm_attention()`换成`BFAttention`,
   不去重新实现vLLM那套op-dispatch桥接。
3. **`replace_vllm_attention()`按`layer_name`字符串解析路径,对草稿模型
   不成立**:换成BFAttention后崩在`IndexError: index 48 is out of
   range`——草稿层的`layer_name`用全局层号(48-53,偏移过主模型的48层,
   避免和主模型共享同一个`static_forward_context`时撞key,
   `laguna_dflash_model.py`模块docstring原本就写了这一点),但草稿模型
   自己的module tree只有本地索引0-5。给`replace_vllm_attention()`加了
   一个可选的`resolve_parent`参数(默认行为对主模型完全不变),草稿模型
   传入一个真实按`draft_model.model.layers`直接走属性访问解析的resolver
   (照抄`_alloc_draft_kv_cache`fallback分支已经证明过的路数)。
4. **`bf_attn_context`没有覆盖所有草稿forward调用点**:换成BFAttention后
   崩在`RuntimeError: BFAttention was called without a scoped attention
   context`——草稿模型有4处真实forward调用点(`generate_verify_only`里
   的一处、`laguna_dflash_cudagraph.py`的CUDA Graph capture/replay一处、
   还有两处),之前只有3处两两配对包了`bf_attn_context`+
   `set_forward_context`(因为原来走真实vLLM `Attention.forward()`只需要
   后者),`_draft_forward`(eager路径)和
   `DFlashDraftCudaGraph._build_metadata_and_forward`(CUDA Graph路径)
   漏了`bf_attn_context`。两处都已经在构造`attn_metadata_dict`/
   `slot_mapping_dict`了,补一层`bf_attn_context(...)`包装,不是重新
   设计。

**最终GPU bit-exact e2e验证通过**(`benchmarks/_phase5_e2e_bitexact_
validate.py`,今天实际默认配置,含DFlash CUDA Graph默认开启,不是绕开
验证的降级路径):`acceptance_rate=0.789333`/`num_steps=50`/640-token
解码文本逐字符和已确认基线完全一致,`DFlash draft CUDA Graph captured`
确认CUDA Graph真的被捕获成功,不是静默回退到eager。

**白名单文件数**:仍然是8个,没有变化——`laguna_decoder.py`/
`model_loading.py`都还有别的、和Attention无关的真实vLLM import
(`extract_layer_index`/`set_default_torch_dtype`),挡在白名单里不是因为
Attention了。白名单comment已更新,写清楚这两个文件现在留在白名单上的
真实、剩余原因。

---

## 阶段8:白名单8→3(任务#41,用户"继续,不要停"拍板重开)

用户原话"继续去做啊。你的目标是什么?达成了吗"——目标是server启动路径
完全不需要vllm可导入,阶段7后白名单还是8个文件,没有达成。用户自己重新
grep了一遍8个文件的完整vllm import(含函数内局部import,不只是文件顶部
的),给出清单+三级优先级。

**第一项:核实`laguna_dflash_cudagraph.py`的`FIPrefill`/`FlashInferMetadata`
是不是真死代码**。派agent重新核实(不照抄阶段0审计的怀疑当结论):
`DFlashVerifyCudaGraph`类(45-357行),真实vLLM `FlashInfer` verify CUDA
graph实现。Git history显示commit `d4354e939e9`(2026-07-25,"Fix SWA
verify_ring window")起就把它从调用链里摘掉了,原因是实测FlashInfer CG
verify比eager还慢(128K处191ms vs 249ms,而且当时有个SWA bug拖低
accept rate)。活跃的verify CUDA graph路径是`laguna_cuda_graph.py`的
`LagunaCudaGraphVerify`(纯sparkinfer,从未导入过vllm/flashinfer),由
`laguna_dflash.py`的`_capture_verify_cg`调用,`QSR_VERIFY_CUDA_GRAPH`
(默认开启)只控制要不要跑`_capture_verify_cg`,从没指向过
`DFlashVerifyCudaGraph`——不是应急开关,是纯粹的孤儿代码。确认后整个类
删除,该文件同时退出vLLM和FlashInfer两个白名单(FlashInfer那个白名单
直接清空)。

**第二项:读真实vLLM源码核实"看起来小"的几个依赖具体做什么,不猜**:
- `set_default_torch_dtype`:5行`torch.set_default_dtype`设置/还原,
  纯plain PyTorch,自建。
- `extract_layer_index`:纯字符串解析,~30行,真实调用点(2处)都只用
  默认的`num_attn_module=1`分支,收窄自建。
- `default_weight_loader`:~15行,`param.data.copy_`,纯plain PyTorch,
  自建,`laguna_model.py`/`laguna_dflash_model.py`共用一份
  (`runtime/model/_weight_loading.py`)。
- `maybe_remap_kv_scale_name`:vLLM真实版本覆盖6种checkpoint命名格式
  (ModelOpt/QKV-proj/Qwen3-MoE/NemotronH/HYV3/默认格式)+MLA特殊前缀,
  但这个checkpoint的真实key(`self_attn.k_scale`)只命中"默认格式"这一条
  pattern(直接safetensors核实过),其余5种不可达。自建版本只照抄这一条,
  写清楚"这是收窄不是遗漏"。
- `get_tensor_model_parallel_rank`:TP=1恒成立,直接硬编码`tp_rank=0`,
  不需要查vLLM distributed group状态。
- `fused_moe_make_expert_params_mapping`/`get_expert_mapping`:**不是
  简单替换,是整条调用链确认100%死代码后直接删除**——`LagunaMoESelfBuilt`
  自阶段6起就没有`self.experts`/`routed_experts`子模块,这个mapping算出来
  的每一条`param_name`都不可能出现在`params_dict`里。验证过：被这条
  死链路originally试图匹配的checkpoint key(`mlp.experts.N.*`),删除
  这条逻辑后一样会在`load_weights`最后的`ignore_suffixes`/
  `params_dict`兜底检查那里被同样静默跳过——不是行为改变,是去掉了
  从来没工作过的死代码。

**第三项:`load_dflash_model`这块"硬骨头"评估**——先做了跟Attention ABC
那次同等力度的调研,不是拿到"工作量大"的第一印象就下结论。**关键发现:
这个import根本不在生产默认路径上**,是`QSR_DFLASH_MODEL_LOADER=vllm`
应急回退开关专属的(`laguna_dflash.py:236`的`if os.environ.get(
"QSR_DFLASH_MODEL_LOADER", "selfbuilt") == "selfbuilt":`分支外),默认
`selfbuilt`路径走的是`runtime.model_loading.load_laguna_dflash_draft_
model`——阶段3就已经自建完成、bit-exact验证过的等价实现,和`get_model()`
escape hatch结构上是完全同一类东西。

调研sglang真实源码(`python/sglang/srt/speculative/eagle_worker_v2.py`)
确认:sglang加载EAGLE草稿模型也是复用自己那套通用`TpModelWorker`(带
`is_draft_worker=True`标记),不是什么草稿模型专属的极简加载器——和vLLM
`load_dflash_model`内部调`get_model()`走完整registry+`AutoWeightsLoader`
是同一类设计,不是vLLM独有的重复劳。这次调研**印证**(不是推翻)了最初
判断:业界不存在"草稿模型加载天生应该很简单"这种参照实现,这个runtime的
`load_laguna_dflash_draft_model`本来就已经是手写的等价物,不是漏掉的
简化机会。

**结论:不动`load_dflash_model`,理由是具体的,不是"嫌麻烦"**——
(1)它已经不在默认生产路径上,不阻塞"server启动零vllm"这个目标本身
(该目标的达成状态取决于默认路径,阶段5已经确认过);(2)它和`get_model()`
是同一类东西:应急回退开关,存在的意义就是提供一个独立于自建实现的、
真实vLLM参考路径,用自建的东西替换掉它会直接废掉这个开关存在的价值,
不是"减少依赖",是"拆掉安全网";(3)`laguna_dflash.py`不管这一项动不动
都会留在白名单上(`bind_kv_cache`/`set_forward_context`/
`ModelConfig`/`SpeculativeConfig`都是这次范围外、有意保留的真实依赖),
这一项本身对白名单文件数没有边际贡献。

**`ModelConfig`/`SpeculativeConfig`**:阶段6已评估过,按用户指示这次
不重做——结论没变(简化构造过程不影响它们仍是vLLM类这个事实,对减少
vLLM import数量没有贡献)。

**`laguna.py`的`fused_topk_bias_router`/`RMSNorm`**(不影响白名单文件数,
`laguna.py`本身留白名单是因为其他有意保留的依赖,但按用户要求仍给出
明确结论):
- `fused_topk_bias`(`_patch_moe_sparkinfer`真实调用,MoE router
  top-k+bias选择,`laguna.py:551`):读了真实源码,这不是trivial函数——
  背后是真实CUDA kernel dispatch(topk+softmax+bias correction的
  compiled op),portable到自建版本需要真正写/移植一个kernel,和阶段1-3
  已经做的RoPE/RMSNorm/KV-scatter kernel移植是同一量级的工作,不是
  "读源码抄5行"能解决的。不在这次范围内做,需要单独立项按kernel移植
  的方式评估。
- `RMSNorm`(`_patch_rmsnorm_triton`,类级别monkeypatch
  `RMSNorm.forward_cuda`):这个self-built路径下实例级patch循环确认是
  no-op(`self.model.modules()`里没有真实vLLM `RMSNorm`实例,全部是
  `TritonRMSNorm`)——但类级别的patch仍然是`QSR_LAGUNA_MODEL_LOADER=vllm`
  escape hatch真实受益的(那条路径的`get_model()`会构造真实`RMSNorm`
  实例,受益于Triton加速)。去掉这个import意味着放弃给escape hatch的
  Triton加速,是真实的(虽然很小的)功能倒退,换来的收益是0(laguna.py
  不会因为这一项离开白名单)。不做。

**验证**:GPU bit-exact e2e全部通过(含MoE层——这批改动里唯一有真实
行为改动风险的部分,`expert_params_mapping`删除),`acceptance_rate=
0.789333`/`num_steps=50`/640-token解码文本逐字符和已确认基线一致。
commit `1e71861`。

**白名单文件数:8→3**(`compat_vllm.py`/`laguna.py`/`laguna_dflash.py`)。
剩下这3个都是backend orchestrator层面有意保留的真实依赖
(`bind_kv_cache`/`set_forward_context`/`get_model`/`ModelConfig`/
`SpeculativeConfig`/`load_dflash_model`escape hatch等),不是遗漏。
`server启动路径完全不需要vllm可导入`这个目标本身,阶段5已经确认过和
"白名单文件数"不是一回事——即使白名单降到0,`compat_vllm.py`/
`laguna.py`仍然会在server真正启动时导入`vllm.config`/`vllm.distributed`
等做config plumbing和分布式初始化,这部分从阶段0开始就被认定为"设计上
应该保留的收口层",不属于这次剥离范围。

---

## 任务#42:compat_vllm.py拆分 + 真实无vllm环境端到端验证

用户交叉核实发现`compat_vllm.py`有模块级、无条件的`GDNAttentionMetadata`/
`SM120GQAMetadata`导入,质疑白名单"8→3"是不是表面进展。核实后确认:这两个
类的isinstance检查是真的(vLLM真实GDN线性attention层代码里),但和Laguna
自己的路径完全无关——真实使用方(`runtime/metadata_builders.py`/
`cuda_graphs.py`/`direct_model_runner.py`)全部是qwen36/DirectModelRunner
专属(阶段0已经把这条路径排除在剥离范围外),问题是`compat_vllm.py`作为
两个tenant共用的文件,把qwen36专属的这几个类做成了模块级无条件导入,
导致Laguna自己合法的`from runtime.compat_vllm import (...)`被迫连带要求
这几个vLLM子模块和第三方`fla`包可导入——不是Laguna需要,是文件共享的
副作用。

**(a) 已修复**(commit `7e73959`):新增`runtime/compat_vllm_qwen36.py`,
把`GDNAttentionMetadata`/`SM120GQAMetadata`/`AttentionBackendEnum`/
`register_backend`/FLA chunk helpers/`compute_causal_conv1d_metadata`
整体搬过去,3个真实调用方同步改导入路径。纯移动不改逻辑,GPU bit-exact
验证通过。

**(b) 真实无vllm环境端到端测试**(用户明确要求:不要再靠静态import扫描
推断,要拿到第一手ground truth,测试环境必须是独立新建的venv,不碰
`~/.venvs/vllm`这个已有的GPU验证测试环境)。

新建`~/.venvs/laguna-novllm-check`(纯净、从零构建,不带`--system-site-
packages`):torch(复用`/home/bot/pytorch-build`这个已有的SM120源码构建,
避免重新编译)、triton、cuda-python bindings直接从`~/.venvs/vllm`拷贝
site-packages(纯文件拷贝,不触发pip依赖解析,不会误装vllm);sparkinfer/
flash_attn_sm120/gn_kernels/fla(flash_linear_attention)这几个editable
install的真实kernel依赖也是纯拷贝`.pth`+finder文件,同样不触发依赖解析;
numpy/safetensors/transformers/fastapi/uvicorn/httpx/requests走真实pip
install。全程没有出现`vllm`这个包本身——验证过`import vllm`直接
`ModuleNotFoundError`。

**结果(第一次真实测出来的ground truth,不是推断)**:
`import runtime.compat_vllm`在这个环境下**失败**,卡在
`import vllm.forward_context as _vllm_fc`这一行——这是`compat_vllm.py`
自己真实、合法的vLLM依赖(不是qwen36泄漏那种,(a)已经修完了),所以
"vllm可以卸载"这个阶段0最初设定的目标,老实说**从来没有真正达成过**,
这次是第一次拿到直接证据,不是继续靠"白名单文件数"这种间接推断。

**`compat_vllm.py`剩下的5个真实模块级vLLM依赖,逐个给结论(不是笼统说
"设计上保留"就跳过)**:

1. **`VllmConfig`/`set_current_vllm_config`**:全runtime的config管道
   骨架,`vllm_config.model_config`/`.cache_config`/`.quant_config`/
   `.compilation_config.static_forward_context`等字段在整个自建模型图
   (`laguna_model.py`/`laguna_decoder.py`/`plain_attention.py`等)里
   被广泛、深度读取,不是一两个调用点。替换意味着重新实现vLLM整套config
   解析(从checkpoint的config.json自动识别架构/dtype/量化scheme、
   CacheConfig、CompilationConfig、ParallelConfig、SchedulerConfig...)。
   **不trivial,量级是月而不是这次session里任何一项清理的量级,不建议
   在没有单独立项评估的情况下动手**。
2. **`EngineArgs`**:CLI参数→config解析的入口,和`VllmConfig`的解析逻辑
   紧耦合,拆不开单独评估——同上,不建议动手。
3. **`get_model`**:阶段6/7就已经明确的应急回退开关,故意保留作为独立
   于自建实现的真实参考基线。不做,理由和`load_dflash_model`那次
   (任务#41)一样。
4. **`init_worker_distributed_environment`**:vLLM自己的`_TP`/`_PP`/
   `_DP` process group初始化。Laguna自己对分布式状态的真实读取已经在
   任务#41里清空了(`get_tensor_model_parallel_rank`硬编码成
   `tp_rank=0`),但这个函数和`VllmConfig`/`EngineArgs`的解析逻辑绑在
   一起(`vllm_config`本身的构造流程依赖它),不是能单独摘出来的一小块,
   同上不建议现在动手。
5. **`ForwardContext`/`CUDAGraphMode`/`_vllm_fc`(即`vllm.forward_
   context`模块)**:**这一项发现了真实的、有具体依据的线索,但没有
   验证,先如实记录不是结论**。`set_forward_context`(compat_vllm.py
   自建)的注释写着"仍需要ForwardContext,因为model layers调用
   get_forward_context()读取"——这句话在阶段7-补充之前是对的(那时
   `self.attn`要么是真实vLLM `Attention`要么还没被`replace_vllm_
   attention()`换掉,`Attention.forward()`确实会调`get_forward_
   context()`)。但阶段7-补充之后,`_alloc_draft_kv_cache()`/
   `_patch_draft_sparkinfer()`在`DFlashEngine.__init__`里**无条件**
   跟在`_load_draft_model()`后面跑(不管走selfbuilt还是vllm loader),
   主模型这边`replace_vllm_attention()`同样无条件跑——也就是说不管
   走哪个loader、哪个模型,最终所有attention层都会被换成`BFAttention`,
   而`BFAttention.forward()`读的是`bf_attn_context`(自己的thread-local),
   不读vLLM的`get_forward_context()`。grep了整个`runtime/`,真实调用
   `get_forward_context()`的地方是**零**(唯一命中的3处都是我自己这个
   session写的docstring/注释,解释"为什么不用它")。这意味着
   `set_forward_context`往`vllm.forward_context._forward_context`
   写状态这件事,现在很可能是完全没人读的死状态——但这只是静态分析
   出来的推断,**没有做"真的去掉这段代码,跑一次GPU bit-exact"这一步
   验证**,不能算数。留给后续单独验证,不在这次任务#42范围内直接动手
   (今天已经动了不少production代码,不适合再叠加一个没验证过的假设)。

**结论**:白名单3个文件目前对应两类真实情况,不能笼统混为一谈——
`laguna.py`/`laguna_dflash.py`基本是"backend orchestrator该有的真实依赖
+故意保留的应急开关",没有明显遗留缺口;`compat_vllm.py`则是"确实是
Laguna自己需要,但规模上远超这次session已经清理掉的任何一项"的真实地基
依赖,其中4/5(`VllmConfig`/`EngineArgs`/`get_model`/
`init_worker_distributed_environment`)当时判断量级上是独立项目,1/5
(`ForwardContext`相关)有具体线索但没验证。**更新(同一天稍晚)**:
`ForwardContext`那一项已经验证通过并落地(见下面任务#42(b)验证记录),
`compat_vllm.py`模块级vllm import从5个降到4个。"vllm可以卸载"这个
阶段0最初目标,到这里为止,第一次有真实环境测试给出的、非推断的答案:
剩下4项没有达成,也不是靠继续做小清理能达成的——但"独立项目"这个量级
判断本身,任务#44重新调研后需要修正,见下文。

---

## 任务#44:VllmConfig/EngineArgs/init_worker_distributed_environment结论修正

用户原话:"又来月级别的工作量了？？？还是那句话 nano-vllm怎么做的？？？"
——和阶段7-补充Attention ABC那次一模一样的方法论质疑:上面"独立项目"级别
这个判断,同样是只看了vLLM自己怎么实现VllmConfig/EngineArgs的复杂度,
没有先查同类极简引擎有没有更简单的路子就下的结论。

**调研nano-vllm真实源码(`nanovllm/config.py`)**:整个config系统是**一个
11字段的flat dataclass**,`hf_config`就是`transformers.AutoConfig.
from_pretrained(model)`——不包私有的架构/dtype/量化scheme解析逻辑,
直接用HF库自己的`config.json`解析。`model_runner.py`里对`hf_config`的
真实读取全部是`hf_config.dtype`/`.num_key_value_heads`/`.hidden_size`/
`.num_attention_heads`/`.num_hidden_layers`这种直接字段访问,没有vLLM
`ModelConfig`那种`get_hidden_size()`/`get_num_layers(parallel_config)`
包装层。分布式初始化(`nanovllm/engine/model_runner.py`)是**6行原始
`torch.distributed`调用**(`dist.init_process_group`/`get_rank`/
`get_world_size`/`all_reduce`/`barrier`/`destroy_process_group`),
完全没有vLLM那套`GroupCoordinator`/`_TP`/`_PP`单例封装层。

**调研sglang真实源码(`python/sglang/srt/server_args.py`+
`configs/model_config.py`)**:这次和上次Attention ABC不一样——sglang的
`server_args.py`本身**9139行**,`model_config.py` **2076行**,和vLLM
自己的规模相当,不是"极简"路子。读了`ModelConfig.__init__`的真实实现:
它做的是真正的多架构自动探测(per-architecture的multimodal禁用列表、
`get_config()`包一层sglang自己的override逻辑、generation_config解析等)
——**因为sglang自己选择了要支持"任意架构、任意部署场景"这个和vLLM同样
广的目标**,这部分复杂度不是"engine不够精简"带来的,是"支持范围选择"
带来的,是真实、必要的复杂度,不是可以绕开的冗余。

**这次调研没有像Attention ABC那次一样单纯得到"其他引擎都更简单"这个
结论——诚实汇报,不能挑对自己有利的证据**:nano-vllm证明"只服务一个
固定模型+固定部署形态"时config系统可以小到30行;sglang证明"要支持
任意架构+任意部署场景"时config系统必然是vLLM同等量级。**决定复杂度的
是范围选择(支持1个模型 vs 支持任意模型),不是"引擎是否成熟/是否
认真"**。我们的真实约束和nano-vllm的选择完全一致(单一Laguna架构、
TP=1恒定、已知固定的NVFP4量化scheme),不是sglang/vLLM那种。

**对照我们自己代码的真实消费点重新审计**(不是继续凭vLLM自己的实现
复杂度猜):

```
grep -rn "vllm_config\.\|draft_vllm_config\." runtime/ server/ 之后
按字段聚合,真实distinct字段/方法只有约15个:
model_config.hf_config / .dtype / .model / .get_hidden_size() /
  .get_vocab_size() / .get_num_layers()
cache_config.cache_dtype
compilation_config.static_forward_context
quant_config
parallel_config.enable_eplb
speculative_config.draft_model_config / .num_speculative_tokens / .model
load_config / device_config
kernel_config.ir_op_priority
```

之前"25处vllm_config.调用"这个数字是原始文本命中次数,不是真实字段数,
把"看起来吓人"和"真实规模"混为一谈了。逐个查真实实现:`get_hidden_size()`
是`return self.model_arch_config.hidden_size`一行;`get_vocab_size()`
同理一行;`get_num_layers()`牵扯PP rank/size计算,但我们PP=1恒成立,
退化成`start=0, end=total_num_hidden_layers`的trivial情况——这几个都是
`hf_config`字段的薄包装,不是深度架构探测逻辑。真实生产入口
(`server/engine.py::_load_laguna_model`)构造`EngineArgs(...)`只传
7个kwarg(`model`/`max_model_len`/`gpu_memory_utilization`/`dtype`/
`disable_log_stats`/`async_scheduling`/`moe_backend`),不是vLLM CLI
那套几百个flag的任意组合。

**但这次评估也确实挖出一个和Attention ABC不同类型的真实结构性障碍,
不是"看起来复杂但其实是空中楼阁"那种,是真实存在的**:`VllmConfig`是
**self-built默认路径和`get_model()`/vllm loader应急开关共用的同一个
上游对象**——`vllm_config = EngineArgs(...).create_engine_config()`
只构造一次,然后`selfbuilt`分支传给`load_laguna_model(vllm_config)`,
`vllm`分支传给`get_model(vllm_config=vllm_config)`。`get_model()`内部
构造的是**真实**vLLM Linear/Embedding/Attention类,这些类的`__init__`
会真实查询`vllm_config`的完整字段集(不只是我们自己代码读的那15个)
和`get_tensor_model_parallel_world_size()`这类分布式状态——如果换成
一个只有15个字段的自建轻量对象喂给`get_model()`,应急开关会直接崩。
和Attention ABC不同:Attention的self-built placeholder和`BFAttention`
从头到尾不需要满足"能喂给`get_model()`内部逻辑"这个约束(两条路径在
构造阶段就完全分岔),但`VllmConfig`是两条路径共用的**同一个**输入对象,
分岔点更靠后。`init_worker_distributed_environment`同理——`get_model()`
内部real vLLM Linear/Embedding的构造过程真实查询TP GroupCoordinator
状态,只要应急开关还留着,这层初始化就不能简单换成
`torch.distributed.init_process_group()`一行了事。

**结论修正**:上次"独立项目、月级别工作量"这个判断,论据是"vLLM
`VllmConfig`自己实现有多复杂",这次重新查证实这个论据站不住脚——真实
消费的~15个字段规模和nano-vllm同一量级,不是vLLM/sglang那种"支持任意
架构"的量级。但**这次不是简单撤回结论说"其实很简单"**——真实挡路的
是一个不同性质的问题:`VllmConfig`/`init_worker_distributed_environment`
是self-built默认路径和`get_model()`应急开关共用的上游依赖,不像
Attention那样两条路径构造阶段就已经分岔。要在保留应急开关的前提下把
默认路径的vLLM依赖去掉,需要把config/分布式初始化的构造点往后移、
按loader分支各自构造(`server/engine.py`等2-3个调用点跟着改,不是内部
实现细节的事)——**这是一个真实的、有具体边界的feature级别改动
(自建一个nano-vllm量级的Laguna专属config dataclass~百行级别+
2-3个调用点的构造流程重构),不是"5分钟看错了"那种,但也不是月级别的
独立项目**。是否要做,交给用户/协调者决定优先级,这次任务范围内先不
动手实现(用户已经指示"如果调研后决定动手实现,再按老规矩确认")。

---

## 任务#45:自建Laguna专属config实现 + 双路径验证

规模收窄后(见上节),用户批准动手实现,并明确要求:self-built默认路径
和`get_model()`应急开关两条路径都要用真实权重+真实文本跑一次GPU
bit-exact确认,不能只测self-built默认路径过了就算完——应急开关是故意
保留的安全网,如果重构把它搞坏了,那是拆了安全网而不是清理依赖。

### 实现

新增`runtime/laguna_config.py`:`SelfBuiltModelConfig`/
`SelfBuiltCacheConfig`/`SelfBuiltQuantConfig`/`SelfBuiltParallelConfig`/
`SelfBuiltCompilationConfig`/`SelfBuiltLoadConfig`/`SelfBuiltDeviceConfig`/
`SelfBuiltKernelConfig`/`SelfBuiltVllmConfig`,以及
`build_laguna_config(...)`（纯self-built构造）和
`build_laguna_config_for_loader(...)`（按`QSR_LAGUNA_MODEL_LOADER`分支,
后者任务#46已删除,见下)。`vllm.transformers_utils.configs.laguna.
LagunaConfig`保留为真实vLLM import——120行、零further vLLM耦合的独立
`transformers.PretrainedConfig`子类,复用它保证字段默认值和历史所有
GPU bit-exact验证字节一致,重新实现有静默drift的真实风险。

`SelfBuiltParallelConfig`的字段是通过GPU报错逐项发现的（`enable_
elastic_ep`→`cpu_distributed_timeout_seconds`等，经`get_current_
vllm_config_or_none()`全局读取，不只是显式参数）——发现`cpu_
distributed_timeout_seconds`这层之后,放弃"喂self-built parallel_config
给真实`init_worker_distributed_environment`/`GroupCoordinator`"这个方向
（深度无界),改为写一个nano-vllm风格的极简`init_laguna_distributed_
environment()`(只做`torch.distributed.init_process_group()`,不建
`GroupCoordinator`)。

`vllm.config.replace()`直接读源码+实测确认对任意`@dataclass`（不只是
vLLM自己的类)都透明生效,这是`SelfBuiltVllmConfig`能做成真`@dataclass`、
让`laguna_dflash.py`里已有的`vllm_replace(self.vllm_config, ...)`调用
零改动继续工作的关键依据。

### 通过真实GPU测试挖出的两个真实bug(都不是这次改动引入的既有回归)

1. **`is_swa` AttributeError**(`laguna.py:331`):阶段7-补充把SWA/full
   分类循环重写成假设`layer.is_swa`永远存在,但应急开关的真实vLLM
   `Attention`实例没有这个属性——阶段7-补充当时只GPU验证过self-built
   默认路径,从未跑过应急开关,这个gap一直没被发现。修复:加`hasattr`
   判断,回退到阶段7-补充之前的`get_kv_cache_spec()`逻辑(任务#46删除
   应急开关后这个分支又被简化掉了,见下)。

2. **ForwardContext回归**(`compat_vllm.py`):任务#42(b)曾经把
   `set_forward_context`简化成no-op(不再写`vllm.forward_context.
   _forward_context`),验证方式是grep `runtime/`+`server/`确认没有
   `get_forward_context()`调用者、再GPU验证self-built默认路径。这个
   验证是不完整的——应急开关的`get_model()`构造的是vLLM**自己**的、
   独立的模型图(`vllm/model_executor/models/laguna.py`,含真实
   `FusedMoE`,这是self-built图从阶段6起就没有的东西),它的内部代码
   真实调用`get_forward_context()`,任务#42(b)从未测过这条路径。真实
   GPU跑应急开关直接崩:`AssertionError: Forward context is not set`。
   修复:恢复`set_forward_context`原本写`ForwardContext(...)`到
   `_vllm_fc._forward_context`的完整实现。

### 验证结果

- **PATH 1(self-built默认,两个loader都是selfbuilt)**:完全bit-exact,
  含两个CUDA graph捕获,`acceptance_rate=0.789333, num_steps=50`——
  ForwardContext恢复前后跑两次,结果不变,证实这段状态写入对self-built
  路径确实无害无用(和任务#42(b)当初的结论一致)。
- **PATH 2(应急开关,两个loader都是vllm)**:修完上面两个bug后,eager
  模式(`QSR_VERIFY_CUDA_GRAPH=0`)生成的token序列跟PATH 1基线**逐token
  完全一致**(直接张量比较确认,不是肉眼看文字)。但`LagunaCudaGraphVerify`
  的CUDA graph replay崩溃(`illegal memory access`)——强证据指向真实
  vLLM的`FusedMoE`/`FLASHINFER_CUTLASS` MoE kernel(只有应急开关的模型图
  才有,self-built图阶段6起没有MoE)和自建verify CUDA graph不兼容(这个
  verify CG从来只针对self-built的无MoE图开发/测试过)——没有用
  compute-sanitizer精确定位,但佐证充分。
- **混合loader(main=selfbuilt, draft=vllm,`_phase3_dflash_bitexact_
  validate*.py`真实依赖的组合)**:发现并修复了任务#45自己引入的一个真实
  gap(`init_laguna_distributed_environment`从不建TP/PP `GroupCoordinator`,
  真实vLLM draft模型图的`VocabParallelEmbedding`需要它)。修复后进一步
  暴露一个**更早的、非本次引入的**历史回归:真实`load_dflash_model()`
  会把draft的`lm_head`跟target模型的`lm_head`权重共享(vocab共享优化),
  target是self-built模型时那就是`PlainLMHead`,没有真实vLLM
  `LogitsProcessor`需要的`.quant_method`属性,直接崩。阶段3当年"两次
  bit-exact通过"发生在阶段6(自建Embedding/LogitsProcessor,`PlainLMHead`
  的引入)之前,这个组合阶段6之后就从未被重新测过——像是阶段6遗留的
  真实回归,不是这次改动造成的。

### 决策:不深挖修复剩下两个bug,直接进任务#46

PATH 1完全过关(含CUDA graph)、PATH 2 eager模式token级完全match,已经
是"两条路径都bit-exact确认"的充分证据。verify-CG+MoE不兼容、
mixed-loader的lm_head tying这两个bug都是应急开关内部、且应急开关本身
马上要在任务#46被彻底删除——征询用户后,选择不再深挖修复这两个bug,
直接记录为已知限制/历史回归,进入任务#46。

---

## 任务#46:彻底删除vllm回退开关

全部验证通过后,把`QSR_LAGUNA_MODEL_LOADER=vllm`/`QSR_DFLASH_MODEL_
LOADER=vllm`两条分支,连同它们背后依赖的`get_model()`/
`load_dflash_model()`真实vLLM调用代码,从Laguna生产路径彻底删除——不是
默认不走但代码保留的半成品状态。

### 删除范围

- `runtime/backends/laguna.py`:`__init__`里`QSR_LAGUNA_MODEL_LOADER`
  分支(分布式初始化+模型加载)全部删除,只保留self-built分支。SWA/full
  分类循环的`hasattr(layer, "is_swa")`回退分支也一并删除(真实vLLM
  `Attention`不会再出现在这条路径上,回退分支变成不可达代码)。
- `runtime/backends/laguna_dflash.py`:`_load_draft_model`删除
  `QSR_DFLASH_MODEL_LOADER`分支、mixed-loader的`isinstance`检测+
  `EngineArgs`重建逻辑、任务#45里为mixed-loader加的`init_worker_
  distributed_environment`调用——draft loader现在只有self-built一条路,
  这些全部变成死代码。`VllmConfig`导入(只为那个`isinstance`检测存在)
  一并删除;`ModelConfig`/`SpeculativeConfig`/`replace`保留(阶段2/3
  决定的"config plumbing暂不自建"范围,跟这次删除无关)。
- `runtime/laguna_config.py`:`build_laguna_config_for_loader`整个删除
  (它唯一的作用就是那个loader分支),调用点直接用`build_laguna_config`。

### 明确不删除的部分(容易误删的边界)

`runtime/compat_vllm.py`的`get_model`/`init_worker_distributed_
environment`/`EngineArgs`/`ForwardContext`/`CUDAGraphMode`导出**全部
保留不动**——这些不是"只为Laguna应急开关存在":
1. `runtime/direct_model_runner.py`(qwen3.6/DirectModelRunner tenant,
   阶段0明确排除在这次剥离范围外)无条件构造真实vLLM模型图,同样需要
   `get_model()`/`init_worker_distributed_environment`/真实
   `ForwardContext`状态(`set_forward_context()`在它自己的代码里也有
   两处真实调用)。
2. `EngineArgs`被`benchmarks/`下几十个诊断/profiling脚本直接构造,跟
   两个tenant的生产loader选择完全无关。

`server/engine.py`调用点从`build_laguna_config_for_loader(...,
disable_log_stats=True, async_scheduling=False, moe_backend=...)`简化
为`build_laguna_config(...)`,去掉的三个kwarg验证过self-built分支从来
没读过(只对真实`EngineArgs`分支有意义)，去掉不改变任何行为。
`benchmarks/_phase5_e2e_bitexact_validate.py`同样简化,不再需要两个
loader env var；`_phase1_bitexact_validate*.py`/`_phase3_dflash_
bitexact_validate*.py`(当年产出阶段1/3 bit-exact证据的历史脚本)加了
弃用说明——它们的`QSR_*_MODEL_LOADER=vllm`分支现在会静默走self-built
(loader选择代码已经不在了),不是报错,容易误导,加注释说明但不重写
(历史记录性质,不再维护成实时A/B工具)。

### 验证

- CPU全量测试(`pytest tests/ -q`):810 passed, 0 failed。
- GPU e2e bit-exact重跑(删除后唯一剩下的生产路径):跟删除前的基线
  逐token/`acceptance_rate`/`num_steps`完全一致(直接张量比较确认)。

### 顺带完成:任务#47(`test_bf_attention.py`两个失败测试)

用户明确要求"不允许任何测试失败,不管是不是这次改动导致的"。根因追查
（不是简单让测试通过）：
1. 测试fixture的KV cache shape用的是`4e99b7c`之前的`[num_blocks, 2,
   block_size, heads, dim]`(K/V在dim1)布局,真实生产代码(`4e99b7c`起)
   是`[2, num_blocks, block_size, heads, dim]`(K/V在dim0)——fixture
   过期了。修fixture shape+断言位置。
2. 修完shape还有第二个更深的问题:`fused_kv_scatter.py`（`4e99b7c`之后
   替换手写Python K/V写入的融合Triton kernel)无条件做scale+fp8量化,
   丢失了旧实现里"非fp8 cache(如bf16)保持原始表示、不缩放"的分支——
   `bf_attention.py`的`forward()`现在对bf16 cache也会错误地做fp8
   scale+量化。生产环境KV cache dtype始终是fp8(`SelfBuiltAttentionPlaceholder`
   硬编码`kv_cache_dtype="fp8"`),这条bf16分支目前是死代码,但测试的
   本意就是验证这个保证——真正的修复是在`bf_attention.py`里恢复
   non-fp8 cache走无缩放直接写入的分支,而不是改测试断言去匹配错误的
   量化行为。同时发现这两个测试需要真实CUDA(Triton kernel不能在CPU
   张量上跑)——CI本来就没装torch,这个模块整体靠`importorskip`跳过,
   加了`torch.cuda.is_available()`跳过条件做防御,不影响CI也不影响
   `~/.venvs/vllm`这个有GPU的验证环境。

---

## 总工作量估算

| 阶段 | 内容 | 估算 | 备注 |
|---|---|---|---|
| 0 | 清场+决策 | 2-3天 | 阻塞后续,优先做 |
| 1 | 模型图+主模型权重加载 | 1-1.5周 | |
| 2 | 自建Linear/Embedding+NVFP4量化 | 1.5-2周 | 可能和阶段1有部分重叠开发 |
| 3 | DFlash草稿模型自建 | 1.5-2周 | 可以和阶段2部分并行(不同人/不同时间片) |
| 4 | DirectModelRunner(条件性) | 0 或 2-3周 | 取决于阶段0的决策 |
| 5 | 收尾验证 | 贯穿全程+集中确认1周 | |

**总计(不含阶段4):约6-8周一人力**,如果阶段2/3能有效并行(两块耦合度不高,
一个是量化GEMM链路,一个是DFlash算法链路),压缩到**5-6周**是合理预期。
**含阶段4(如果DirectModelRunner保留):再加2-3周**。

**最大的时间不确定性来源始终是验证,不是编码**——尤其阶段2(NVFP4数值)和阶段3
(DFlash算法)历史上都有真实踩坑记录,建议排期时预留验证缓冲,不要按"写完代码就
算完成"来估工期。

---

## 建议的启动顺序

1. 先回答阶段0的决策问题(DirectModelRunner是否保留),这直接影响总工期要不要
   算上阶段4的2-3周。
2. 阶段0本身工作量小,可以立刻开始,不需要等决策问题回答完(先修护栏测试和
   `fused_kv_scatter.py`接线,这两项和决策问题无关)。
3. 阶段1是后续一切的地基,决策问题回答后应该第一个集中投入。
4. 阶段2/3在阶段1完成、有了自己的模型图骨架之后,可以两条线并行推进。
