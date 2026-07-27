# vLLM / FlashInfer 依赖全面审计 + 剥离规划(2026-07-27)

## 一、两个需要立刻确认/处理的红线问题(独立于长期规划,现在就该看)

### 1. 依赖护栏测试当前是红的(架构纪律漂移)

仓库自带 `tests/test_vllm_dependency_boundary.py`,用 AST 扫描冻结了"谁被批准直接
`import vllm`/`import flashinfer`"的白名单。实测:

```
FAILED test_vllm_direct_imports_are_an_explicit_migration_ledger
New direct vLLM import outside the migration ledger:
['runtime/backends/bf_attention.py', 'runtime/backends/laguna_cuda_graph.py',
 'runtime/backends/laguna_sparkinfer_attn.py']
```

这三个文件都是过去两天(K/V竞态修复、DFlash 接受率排查期间)新增的直接 `import vllm`
(具体是 `reshape_and_cache_flash`),护栏测试本体最后更新在更早之前,没跟上。**说明
"依赖必须经 `compat_vllm.py` 收口"这条架构约束,在这几天的高速迭代里已经出现真实
漏管**。

处理选项:①把这3个文件补进白名单(承认现状,最快)②把这3处 `reshape_and_cache_flash`
调用改走已经写好但从未接线的 `runtime/kernels/fused_kv_scatter.py`(Triton 自研实现,
文档目标就是替换这几个调用点),顺带验证 bit-exact 后彻底解决。**建议选②**,反正
这是"剥离vLLM"规划里"容易"那一档的活,顺手做掉比先斩后奏地放宽护栏更干净。

### 2. DirectModelRunner 路径今天可能直接崩溃启动(已用真实 Python 调用复现)

FlashInfer 调研 agent 实际执行了触发路径,复现出:

```
RuntimeError: flashinfer-cubin version (0.6.13) does not match flashinfer version (0.6.15).
Set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass this check.
```

调用链:`DirectModelRunner` 默认启用的 `patch_nvfp4_prefer_b12x()`
(`nvfp4_b12x_patch.py`,`QSR_A2_B12X` 默认开)→ `FlashInferB12xNvFp4LinearKernel.is_supported()`
→ `has_flashinfer_cutedsl()` → `importlib.util.find_spec("flashinfer.cute_dsl")`——对子模块
`find_spec` 会先 import 父包 `flashinfer`,今天早些时候修的那个 `flashinfer-python`
(0.6.15) vs `flashinfer-cubin`(0.6.13,PyPI无更新版本)版本冲突在这里被真实触发。
**这个 `RuntimeError` 不是 `ImportError`,不会被现有的 `try/except ImportError`
兜住**,且全仓库搜索确认生产启动路径没有任何地方预置 `FLASHINFER_DISABLE_VERSION_CHECK=1`。

**这意味着:如果现在真的启动 DirectModelRunner(`QSR_SERVER_MODEL_BACKEND=qwen36`,
而且这是 `server/app.py` 里的默认值),大概率会在这一步崩溃。**

**需要你确认一个前提性问题**:`server/app.py` 注释里 `QSR_SERVER_MODEL_BACKEND` 默认
就是 `"qwen36"`(DirectModelRunner),Laguna 被代码注释描述为"第二租户"(需要显式选择
才启用)。这和这几天所有工作默认把"Laguna 是生产路径"当前提,可能有出入——**DirectModelRunner
现在到底还是不是一个会被真实启动的路径?** 如果是历史遗留、已经不再启动,这个崩溃
风险可以先记录、不用紧急修;如果它仍然是某个默认/兜底路径,这个 bug 需要马上修
(最简单的办法是给 `is_supported()` 外面包一层 `try/except Exception`,或者干脆确认
B12x 从不生效后直接删掉这条 patch)。

---

## 二、vLLM 依赖现状(按剥离难度分类)

### 已经完全剥离,不需要规划工作
- 采样(`runtime/sampling.py`,自研 dataclass + CUDA-Graph-safe 采样逻辑,零 vLLM/flashinfer 依赖)
- Tokenizer(直接用 `transformers.AutoTokenizer`,绕开 vLLM 的 TokenizerGroup)
- 调度器/cache coordinator(`ServerEngine` 自研定长 slot 调度,零 vLLM `Scheduler` 依赖)
- KV block/prefix cache 管理(`block_pool.py`,设计参考 vLLM 但纯自研实现)
- FLA(线性注意力)chunk 索引(已切到上游 `fla` 包,不再用 vLLM 内嵌版本,有 bit-exact 验证记录)

### 容易(浅依赖,已有替代雏形,优先级高、工作量小)
- `reshape_and_cache_flash`(vLLM C++ 算子,6处调用):`fused_kv_scatter.py` 已经写好
  纯 Triton 替代实现,**但从未被接线使用**。这是这次审计里最现成的"低垂果实"。
- 上面第一部分的护栏漂移(3处新增直接 import)。

### 已有部分收口,但收口层本身尚未被自己的实现替代
`runtime/compat_vllm.py`(482行)是一个已经分好三级的成熟抽象层(自写薄依赖/中等
re-export/厚 re-export),这是整个项目里最值得复用的架构资产——但"厚"这一档
(`get_model`、`load_eagle_model`)只是标注了"最后才替换",还没真正动手。

### 难啃,是真正的架构工作
1. **`get_model()` / 模型图构建**——单点最重依赖。加载模型时真实跑一遍 vLLM 的层级
   wiring、CustomOp(RMSNorm)绑定、`@support_torch_compile` 装饰、分布式/`parallel_state`
   初始化(即使 TP=1 也要走全套)。这四个子问题本质是一个问题:只要还用 `get_model()`
   造模型实例,CustomOp monkey-patch、torch.compile 装饰器、分布式初始化就都甩不掉。
2. **DFlash 推测解码的模型加载**(`SpeculativeConfig`/`ModelConfig`/`load_dflash_model`)——
   **完全绕开了 `compat_vllm` 这层收口**,直接依赖 vLLM spec-decode 子系统的私有
   权重共享/eagle3 接口约定。是"生产关键路径 + 未被收口 + 未被自研替代"三重叠加的
   最脆弱点。
3. **NVFP4 kernel 选择**——运行时改写 vLLM 内部**可变全局注册表**
   (`_POSSIBLE_NVFP4_KERNELS`)来强制优先级,这类 monkey patch 在 vLLM 内部数据结构
   改名/改形状时(0.25→0.26 已经证明过会发生)会静默失效,是最脆弱的一类。
4. **`sm120_gqa.py` 对 vLLM Attention ABC 契约的深度模拟**(仅 DirectModelRunner 路径
   使用)——不是要不要换计算逻辑(已经是自研 kernel),是"注册表/enum round-trip/
   CUDA Graph 捕获约定"这套集成语义,脱离 vLLM 需要重新设计一套等价的调度骨架,
   工作量接近重新发明 vLLM attention backend 系统的一个子集。

### 两条后端路径的关键差异
| | DirectModelRunner(qwen36) | LagunaBackend/DFlashEngine |
|---|---|---|
| compat_vllm 收口纪律 | 100%守规矩,全文零直接 import vllm | 5处绕开(见红线1) |
| Attention 抽象依赖 | 深(靠 vLLM 注册表/enum/ABC) | 已剥离(自研 `bf_attention.py`) |
| 模型图构建 | `get_model()`(厚) | `get_model()`(厚,相同瓶颈) |

两条路径在"模型图构建"这个最重依赖上完全一样,这是剥离工作的公共瓶颈,不管保留
哪条路径都绕不开。

### 附带发现
`pyproject.toml` 声明 `vllm==0.26.0`,但生产实际锁定 `0.25.0+patch`(有实测8.6%吞吐
回归的决策记录)——规划替换顺序时要以 0.25.0+patch 的真实行为为准,不能参考
pyproject 声明版本。

---

## 三、FlashInfer 依赖现状

**结论先给**:Laguna 生产路径(attention + MoE + CUDA Graph)**对 flashinfer 的真实
依赖已经是零**——不是"接近替代完成",是已经替代完成,有代码证据支撑:

- `laguna_sparkinfer_attn.py`、`laguna_sparkinfer_moe.py`、`laguna_cuda_graph.py`
  三个核心生产实现文件,逐行读过,**零 flashinfer import**。
- 唯一一处 CI 白名单里"批准"的直接 flashinfer import(`laguna_dflash_cudagraph.py`
  里的 `DFlashVerifyCudaGraph` 类)**从未被任何生产代码实例化,是死代码**——建议直接
  删除这段代码而不是"替代"它,顺便简化护栏账本。
- `init_flashinfer_workspace` 这个函数名带"flashinfer"字样,但实际读源码是纯 vLLM
  通用 buffer manager,不碰真正的 flashinfer 包——纯粹是历史命名遗留,容易误导。
- sparkinfer 自己(`pyproject.toml` 依赖列表)**不依赖 flashinfer**,内部唯一出现
  flashinfer import 的地方是它自己测试套件用来做正确性 oracle 对照的辅助文件,不在
  推理前向路径上。

DirectModelRunner 路径:attention 走的是**第三条独立血统**(vLLM 原生
`SM120GQABackend`,既不是 flashinfer 也不是 sparkinfer),所以 attention 层面也不
依赖 flashinfer。真正会碰 flashinfer 的是 NVFP4 GEMM kernel 选择逻辑(见红线2的
那次崩溃复现)——这块如果要迁移,理论上目标应该是"改用自研/sparkinfer的GEMM kernel
选择,不需要vLLM的B12x路径",而不是"用sparkinfer的attention/MoE去替代"(因为这条
路径的attention本来就不是flashinfer)。

---

## 四、剥离规划建议(阶段化)

**阶段0(立刻做,风险处理,不是"剥离"本身)**:
- 确认 DirectModelRunner 是否还是真实会启动的路径(需要你回答)。
- 视情况修复红线2(包 try/except 或删除B12x patch)。
- 处理红线1(接线 `fused_kv_scatter.py`,清掉3处ledger违规)。

**阶段1(容易,清理噪音,为后续统计打基础)**:
- 接线 `fused_kv_scatter.py` 替换全部6处 `reshape_and_cache_flash`,做 bit-exact 验证。
- 删除 `laguna_dflash_cudagraph.py` 里的死代码(`DFlashVerifyCudaGraph`),同步收窄
  flashinfer 白名单。
- 如果确认 DirectModelRunner 不再需要,评估直接整体归档这条路径(连同 `sm120_gqa.py`
  的深度ABC依赖一起免于处理)。

**阶段2(核心攻坚,工作量最大,建议集中投入)**:
- 自己实现模型图构建,替代 `get_model()`——这是唯一的、真正意义上的"深度架构工作",
  解决之后能连带解决 CustomOp dispatch monkey-patch、`@support_torch_compile` 装饰、
  分布式/`parallel_state` 初始化这三个连带问题。
- DFlash 推测解码模型加载改用自己的实现,不再依赖 vLLM spec-decode 私有约定。

**阶段3(可以和阶段2并行,是独立子问题)**:
- NVFP4 kernel 选择逻辑改成自己维护的注册表,不再运行时 monkey-patch vLLM 内部
  可变全局状态。

---

## 五、给你的开放问题(需要回答才能定优先级)

1. **DirectModelRunner(qwen36)现在还是真实会被启动的生产路径吗?** 这决定红线2的
   紧急程度,也决定阶段1里"是否要整体归档这条路径"这个选项是否可行。
2. **剥离vLLM这件事的目标范围是什么?** 是"完全不依赖vLLM(连`get_model()`都自己实现)"
   这种彻底剥离,还是"把已知的脆弱点/风险点(NVFP4 monkey-patch、DFlash未收口依赖)
   处理掉,模型图构建这种巨大工作量的部分先不动"这种务实收敛?两者工作量差异极大
   (阶段2是这次审计里公认最重的部分)。
