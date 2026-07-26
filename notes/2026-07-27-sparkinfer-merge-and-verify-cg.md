# sparkinfer master 合并 + DFlash verify CG 复测(2026-07-27 01:20)

## 结论(先说)

- **sparkinfer 合并成功**,`blackforge-main` 现在同时包含正确性修复和 master 的
  Laguna 性能改进,状态干净、可用。
- **verify CG 仍然比 eager 慢约 6-7%**,和 `cac38ab` 记录的历史结论(慢 7%)一致。
  合并本身**没有解决**这个问题。
- **不改动 `QSR_VERIFY_CUDA_GRAPH` 默认值**(保持 `"0"`,eager)。现有数据不支持
  改成默认开启 CG。
- 根因高置信定位到:我们当前调用点用 `mode="extend"`(非 `"verify"`)+
  `block_size=64`(非 `128`),而 sparkinfer 里能跳过 worklist 更新开销的
  analytic-schedule 快速路径要求两者都满足 `mode=="verify"` 且 `page_size==128`。
  这不是"改一行代码"能修的,需要更大的结构性改动,本次任务范围内不做。

## 1. sparkinfer 合并

在 `/home/bot/project/sparkinfer` 执行 `git merge master`(从 `blackforge-main`
分支,提交 `3fa9b54`),合并出新的 `blackforge-main` HEAD `478b9af`
("Merge branch 'master' into blackforge-main: pull in Laguna SM120
graph-pipeline perf work")。

- **合并范围**:master 相对 blackforge-main 领先的 6 个 commit
  (`83a5844`..`ccea985`):`83a5844`(perf(indexer): keep exact long-context
  top-k on fast path)、`a5bc746`(feat(gemm): add BF16 MLA query projection)、
  `7a35259`(perf(paged-attn): optimize SM120 graph pipelines——核心提交)、
  `0519715`(fix(paged-attn): accept narrower copied graph tables)、
  `c39b806`(bench: add production model tuning probes)、`ccea985`
  (Merge branch 'blackforge-main' into master,把我们的 3 个正确性修复带进
  master)。**无冲突**(`git merge-tree` 提前确认过,`<<<<<<<` 计数为 0),
  `git merge` 用 `ort` 策略干净完成。
- **不需要重新编译**:sparkinfer 没有 `setup.py`/`CMakeLists.txt` 里的
  `ext_modules`/`CUDAExtension`,全部是 `@triton.jit` 运行时 JIT 和纯 Python,
  合并后直接 `import sparkinfer` 正常。
- **测试**:跑了 `tests/attention/test_attention_paged_traits.py`、
  `test_attention_paged_planner.py`、`test_attention_cuda_graphs.py`
  (需要真实 GPU,`CUDA_VISIBLE_DEVICES=''` 会直接报
  `RuntimeError: No CUDA GPUs are available`,所以是带 GPU 跑的,不是纯 CPU)。
  结果:**60 passed, 2 failed, 1 xfailed, 2 xpassed**(耗时 257.9s)。

  2 个失败都是同一个根因,**不是这次合并引入的新问题**:
  `test_laguna_page128_split_graph_traits_consume_one_physical_page` 和
  `test_laguna_page128_traits_reject_same_gqa_nonproduction_head_counts`
  在 `sparkinfer/attention/paged/traits.py:439` 抛
  `AttributeError: 'types.SimpleNamespace' object has no attribute
  'window_left'`——测试用的 mock plan(`SimpleNamespace`)没有覆盖新加的
  decode gqa=6 分支要检查的 `window_left` 字段。这两个测试文件本身就是
  `master` 的 `7a35259` 提交带来的,合并只是把这个已存在的测试 fixture
  缺口原样带进来,没有改变其逻辑。建议后续单独修 sparkinfer 上游,不在这次
  任务范围内处理。

- 未跑 sparkinfer 全量测试套件(时间成本高),只跑了和这次任务直接相关的
  attention/traits/planner/cuda_graphs 四个文件。

## 2. 三个技术问题的确凿结论

### 2.1 `mode="extend"` vs `mode="verify"` 语义

读 `sparkinfer/attention/paged/planner.py` 的 `create_paged_plan`:
- `mode="verify"` 要求 `inferred_mode == "extend"`(即多 token,非纯 decode),
  否则报错——verify 本质是 extend 的一个特化标记,不是完全独立的形状类别。
- `force_split_kv = mode == "verify" or (msa_block_sparse and mode == "decode")`
  ——**切到 `mode="verify"` 会强制 `split_kv=True`**,这是一个真实的执行策略
  变化,不只是打标签。我们目前的 `LagunaCudaGraphVerify` 用 `mode="extend"`
  (`runtime/backends/laguna_cuda_graph.py`),没有强制 split_kv。
- `traits.py` 里的 5 条 Laguna 专用特化分支中,verify 分支
  (`plan.mode == "verify"`)恰好就是靠 `mode` 字段区分的,`mode="extend"`
  永远走不到这条 verify 特化,也永远碰不到
  `_uses_laguna_verify_analytic_schedule()` 那条"跳过 worklist 更新器"的
  快速路径(该函数在 `workspace.py` 里首先检查 `self.mode == "verify"`)。
- **本次任务没有实际切换成 `mode="verify"` 做实测**——这涉及执行策略变化
  (强制 split_kv),需要专门验证正确性(输出是否仍然一致)和性能,超出本次
  A/B 验证的时间预算,留作后续任务。

### 2.2 `page_size` 是什么、64 vs 128

- `create_paged_plan` 里 `page_size` 直接来自 `k_cache.shape[1]`
  (`num_pages, page_size, num_kv_heads, head_dim_k = k_cache.shape`),就是我们
  自己说的 KV cache `block_size`——**同一个东西,不是两个概念**。
- API 层面 `if page_size not in (64, 128): raise ValueError(...)`——64 和 128
  都是合法值,不是只支持 128。
- 但 `traits.py` 里**全部 5 条** Laguna 专用 kernel 特化(decode×2、
  extend×2、verify×1)都硬编码 `and plan.page_size == 128`,没有一条支持
  64。也就是说:**只要我们还在用 `block_size=64`,不管 mode 怎么传,一条
  Laguna 特化都吃不到**,包括之前调研提到的、也包括这次想验证的
  verify analytic-schedule 快速路径。
- `runtime/backends/laguna.py` 生产用 `block_size=64` 是 `e66d254` 的结论
  ("The sparkinfer paged-attention integration requires block_size=64",
  当时把 76 个 benchmark/test 从 16 改成 64)。本次在合并后的代码里检索了
  `page_size`/`block_size` 相关的硬编码校验,**没有找到任何强制"必须是 64"
  的断言**——`planner.py` 里另有一个不相关的 `_heuristic_decode_graph_ctas_
  per_sm` 函数专门检查 `page_size == 64` 走特定 occupancy 启发式,但那是
  decode CTA 数量调优,和 Laguna kernel trait 特化是两件事。这意味着
  `e66d254` 当时"必须 64"的结论,更可能是基于实测行为(某些 kernel 在别的
  page_size 下产出错误结果或崩溃),而非当前代码里的硬编码限制——**这次任务
  没有时间实测 128 是否仍然正确/必需仍是 64**,不确定当时的限制在合并后的
  代码上是否依然成立。
- **迁移 block_size 64→128 是一次结构性改动**(KV cache 内存布局、block
  table 尺寸公式、`e66d254` 当时改了 76 个文件的同类工作量),按照任务要求
  **本次不动手**,只如实报告:这是解锁全部 5 条 Laguna 特化(而不仅是 verify
  这一条)的必要前提,值得作为独立评估的后续任务,但成本不低。

### 2.3 analytic-schedule 快速路径目前可不可达

**不可达。** 需要同时满足 `mode=="verify"` 且 `page_size==128`,我们当前两条
都不满足(`mode="extend"`,`page_size=64`)。这是这次合并没能改善 7% 差距的
直接原因——`update_prefill_graph_replay_metadata` 的 worklist 更新开销依然
会被调用,合并只是把它带来的其它路径优化(比如 forward_extend_generic.py
的 PTX 转换改进)顺带包含进来,但那些同样大多数是靠 `traits.py` 里 shape 匹配
触发的,不匹配就还是走通用路径。

## 3. A/B 实测数据

脚本:`benchmarks/ab_dflash_verify_cg_vs_eager.py`(64K 上下文,重复
"The quick brown fox..." 构造的确定性 prompt,`DFlashEngine.generate`,
`max_tokens=256`,2 轮)。独立 venv `/home/bot/.venvs/vllm-repro80`,
合并后的 sparkinfer(`blackforge-main@478b9af`),`main` 分支运行时代码。

| 配置 | Round | tok/s | 接受率 | wall(s) | 备注 |
|---|---|---|---|---|---|
| eager(`QSR_VERIFY_CUDA_GRAPH=0`) | 0 | 0.69 | 68.7% | 750.9 | **异常值**,首次调用撞上 JIT/首轮编译开销,不代表稳态 |
| eager | 1 | **39.68** | 68.7% | 20.4 | 稳态 |
| CG(`QSR_VERIFY_CUDA_GRAPH=1`) | 0 | 37.53 | 68.7% | 20.8 | 稳态(CG 没有 eager 那种首轮异常,两轮都正常) |
| CG | 1 | 36.90 | 68.7% | 20.5 | 稳态 |

原始 JSON:`benchmarks/fixtures/ab_dflash_verify_eager_20260727.json`、
`benchmarks/fixtures/ab_dflash_verify_cg_20260727.json`。

**稳态对比**:eager 39.68 tok/s vs CG 均值 37.2 tok/s(37.53/36.90)——
**CG 慢约 6.7%**,和 `cac38ab` 记录的"慢 7%"基本一致。

**接受率说明**:这次两种配置都是 68.7%,内部可比;但和 `cac38ab` 原始记录
的 88.5% 不是同一个 prompt/条件,**不能跨记录直接比较绝对 tok/s 数字**,
只能比较同一批测试内 eager vs CG 的相对差距,而这个相对差距(约 6-7%)两次
测量高度吻合,增强了"这是真实、稳定的效应,不是噪声"的信心。

## 4. 决策

- `runtime/backends/laguna_dflash.py` 里 `QSR_VERIFY_CUDA_GRAPH` 默认值
  **保持 `"0"`(eager),不修改**。当前证据不支持默认开启 CG。
- sparkinfer 合并本身有价值(带来 Laguna kernel 特化的代码基础、
  `plan_extend_graph_capacity`/`compile_paged_attention` 等新 API、`3fa9b54`
  的 CG-capture 兼容修复现在双向都在),**予以保留**,不回滚。
- 若要真正解决 verify CG 的 7% 差距,需要按顺序独立评估两件事(缺一不可):
  1. `block_size` 64→128 迁移的可行性与成本(结构性改动,上面已说明)
  2. `LagunaCudaGraphVerify` 切到 `mode="verify"` 的正确性验证(`force_split_kv=True`
     的行为变化需要专门测试,不能假设无害)

## 5. sparkinfer / 本仓库最终状态

- `/home/bot/project/sparkinfer`:分支 `blackforge-main`,HEAD `478b9af`,
  working tree clean,领先 `origin/blackforge-main` 8 个 commit(**未 push**)。
- `/home/bot/project/qwen-sm120-runtime`:本次改动只有新增文档和 benchmark
  脚本/fixture,`runtime/backends/laguna_dflash.py` **未改动**。
- `/home/bot/vllm`、`/home/bot/.venvs/vllm`(主 venv)**未触碰**。
- GPU 全程只跑了一个进程(每次操作前用 `nvidia-smi`/`ps aux` 确认过)。

## 6. 遗留问题清单

1. **block_size 64→128 迁移**:是否可行、成本多大、对现有 KV cache 内存
   布局/block table/其它 76 个 benchmark 和 test 的影响面,需要单独评估
   (这是解锁全部 5 条 Laguna kernel 特化的必要前提,不只是 verify CG 一项)。
2. **`mode="extend"→"verify"` 切换的正确性验证**:`force_split_kv=True`
   之后需要确认 verify 输出(logits/接受率)与现在完全一致,再谈性能。
3. **sparkinfer 上游测试 bug**(`traits.py:439` 的 `SimpleNamespace` 缺
   `window_left`):不是我们这次合并造成的,但影响这两个测试的可信度,
   建议反馈给 sparkinfer 上游修掉测试 fixture。
4. **`decode`/`extend` 路径是否也吃不到特化**:这次只验证了 verify CG,
   `LagunaCudaGraphDecode`(纯 decode CG)和主 prefill 的 extend 路径是否
   同样因为 `page_size=64` 吃不到 Laguna 特化,本次没有单独测,值得后续
   跟进(如果 decode/extend 也一样被 page_size 卡住,那 block_size 迁移的
   收益就不只是 verify CG 这一条,优先级可能更高)。
5. 只跑了 sparkinfer attention 相关 4 个测试文件,没有跑全量测试套件。
