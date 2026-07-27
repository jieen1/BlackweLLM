# 完全剥离 vLLM 依赖——分阶段实施计划(2026-07-27)

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

这是"折中方案"和"彻底剥离"分道扬镳的地方,目标是连vLLM的Linear/Embedding类和
`process_weights_after_loading`/量化GEMM调度机制都不要。

1. **自建简化版 Linear/Embedding 类**(TP=1场景,不需要vLLM那套TP切分感知的复杂
   `weight_loader`协议,可以写得比vLLM原版简单得多——多GPU支持是路线图里明确的
   远期项(`docs/roadmap.md` B6),这里先只覆盖当前真实场景,TP切分作为以后独立
   任务的扩展点,不要在这里为了"以防万一"预先做通用化)。
2. **移植NVFP4权重侧预处理**(`swizzle_blockscale`+`pad_nvfp4_weight_for_cutlass`,
   ~120行纯张量函数,可以逐字复制,零风险)。
3. **移植global scale数值逻辑**(~50行,**必须原样保留"gate_proj/up_proj不融合"
   这个设计决策**——这不是vLLM特有的limitation,是"per-Linear量化checkpoint+融合
   Linear层"这个组合的数学问题,任何框架下都存在,合并会不可避免损失精度,唯一的
   规避方式就是不融合,自己实现时不要尝试"修好"这个问题,原样沿用现有设计)。
4. **移植/重写NVFP4激活量化kernel**(`scaled_fp4_quant`,~430行CUDA,swizzled
   fp8 block scale输出布局)——这是本阶段唯一的"从零写CUDA"工作,但项目已经有
   写`nvfp4_gemm_sm120.cu`的经验,是同类型的第二个kernel,不是新技能。
5. **让自研GEMM kernel直接被自己的Linear类调用**,不再需要`nvfp4_b12x_patch.py`/
   `nvfp4_cutlass_direct_patch.py`/`nvfp4_cudnn_patch.py`这些monkey-patch vLLM内部
   注册表的中间层——这些patch文件可以在这个阶段结束后**整体删除**,连带解决了
   之前调研指出的"最脆弱、随vLLM升级容易静默失效"的那个风险点。
6. **彻底去掉分布式/`parallel_state`初始化**——一旦不再用vLLM的Linear/Embedding类,
   这个依赖就没有存在理由了,`init_worker_distributed_environment`调用点可以删除。
7. **验证**:重点是量化数值精度(bit-exact vs 当前CUTLASS baseline,所有patch文件
   都以此为验收标准),以及"gate/up不融合"这个设计在新实现里确实被遵守。

---

## 阶段3:DFlash 草稿模型自建(可以和阶段2部分并行)(~1.5-2周)

1. **复用阶段1/2的模型图+权重加载基础设施跑草稿模型**——草稿模型本来就走同一套
   `get_model()`流程,阶段1/2的工作对它同样适用,只是要跑两遍(主模型+草稿模型),
   不是新增复杂度类别,但工作量要按两次算(权重命名映射规则不同,需要单独适配)。
2. **Embedding/lm_head tied-weight共享**(~50行,纯Python对象引用操作,直接照搬
   `_should_share`的逻辑,零风险)。
3. **草稿模型配置构造简化版**(不需要vLLM `SpeculativeConfig.__post_init__`的完整
   ~10种speculative方法调度器,只需要11个字段+2处从draft checkpoint的hf_config
   读`causal`/`attention_backend`字段这两行逻辑)。
4. **`SupportsEagle3`契约移植**(<60行,已经是最简单的一块)。
5. **DFlash核心算法移植**(本阶段主体工作量,~400-450行):
   - `combine_hidden_states`(aux hidden states逐slice RMSNorm→拼接→fc投影→
     hidden_norm)——RMSNorm计算部分**接入已有的`fused_rms_norm.py`**(已经写好,
     只是没接入这条链路)。
   - `precompute_and_store_context_kv`(所有draft层KV投影权重concat成一个大矩阵、
     一次fused GEMM算出全部层K/V、grouped RMSNorm、批量RoPE、写入各层KV cache)——
     KV cache写入部分**接入`fused_kv_scatter.py`**(阶段0已经接线),RMSNorm部分同上
     复用现有kernel,**唯一真正从零开始的是批量RoPE kernel**(对展平的多层K张量
     一次性做旋转,这个"多层融合"的批处理形态目前项目里完全没有先例,是本阶段
     真正的新增工作)。
6. **验证**:接受率/贪心一致性对照(这套方法论今天在block_size排查里被反复验证有效,
   直接复用——同一逻辑位置的输出必须bit-exact,任何偏差都要narrow down到具体子步骤)。

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

贯穿前面每个阶段,但需要一次集中的最终确认:
1. 全链路端到端bit-exact验证(不是逐模块,是完整请求从prefill到decode到DFlash
   verify的完整输出对照)。
2. 确认生产server启动路径(`server/app.py`)不再需要`vllm`可导入。
3. `pyproject.toml`更新(去掉`vllm-provider`这个extra,或者明确标注为"仅benchmarks/
   对比脚本可选依赖")。
4. 依赖护栏测试(`tests/test_vllm_dependency_boundary.py`)更新白名单,理想情况下
   白名单最终清空(`runtime/`+`server/`下不再有任何被批准的直接vLLM import)。
5. `benchmarks/`下的vLLM原生对比脚本**保留不动**,它们的存在价值不受这次剥离影响。

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
