# 待排查清单（2026-08-01 汇总）

> 本轮会话中浮现的、需要**逐条排查或拍板**的事项，集中在一处以便按顺序处理。
>
> 与另外两份文档的分工：[`roadmap.md`](roadmap.md) 是目标与理由，
> [`implementation-plan.md`](implementation-plan.md) 是路线图的执行视图。
> **本文档是"外部输入 + 自查"产生的队列**——它们大多不在原路线图里，
> 是读上游代码和跑生态扫描时冒出来的。处理完的条目应并入 implementation-plan 或删除。
>
> 每条标注：**去向**（我们做 / 交 SparkInfer / 需你拍板）、**是否需要 GPU**、**依据**。

---

## A. 需要你拍板（阻塞其它工作）—— ✅ 三条已于 2026-08-01 拍板

- [x] **A-1 · D3 GPU CI 形态** —— ✅ **已拍板 (b) 本地 pre-push 门禁 + 人工签核**。理由：单 GPU 机器上
  自托管 runner（选项 a）本身也要抢卡，不解决"GPU 验收天然串行"这条约束；(c) 门禁太松。
  详见 [`roadmap.md`](roadmap.md) §7/D3、[`implementation-plan.md`](implementation-plan.md) §4/C-1、§7.3/C4。

- [x] **A-2 · D6 Qwen3.6 主线 checkpoint** —— ✅ **已拍板：官方 `nvidia/Qwen3.6-27B-NVFP4`**。
  真正的取舍是 **官方 provenance + 需排除 333 个 vision 张量** vs **社区量化 + 天生文本版**（0 vision、
  单文件）；provenance 不可逆、vision 过滤是一次性机械工作，社区版留作交叉验证。
  详见 [`roadmap.md`](roadmap.md) §7/D6、[`implementation-plan.md`](implementation-plan.md) §4/C-2、§7.1/B0-1。

- [x] **A-3 · N8 `--session-affinity` 静默失效** —— ✅ **已拍板 (c) 启动期拒绝该 flag**。
  `mtp_prefill_warm_continue` 只存在于已退役的 `oracle/qwen36_vllm/`；把静默降级变成显式失败，
  (a) 留给 Track A 能力查询落地后重新评估。详见 [`architecture.md`](architecture.md) §3.5.6、
  [`implementation-plan.md`](implementation-plan.md) §6.1（落地清单，尚待实现）。

---

## B. 自查 —— 不需要 GPU（本轮已完成的标 ✅，余下待做）

- [x] **B-1 测试是否该 skip 却 fail** —— ✅ **按合同 N/A**。我们靠启动门控（`preflight.py:224`、`laguna_router.py:85`）而非测试门控；合同就是 sm120-only，所以在别的卡上失败是正确行为。对应 sglang #29900 / #31365。
- [x] **B-2 是否有格式敏感的打包权重切分** —— ✅ **今天不存在**。`model_loading.py` 完全不做打包权重切分。**Track B（GDN）+ A4（modelopt adapter）时变活**,已排期进 B0-2。对应 sglang #31720。
- [x] **B-3 是否踩 sm120 cuBLAS FP8 限制** —— ✅ **两重 N/A**。上游 issue 已关且维护者确认 SM120 上可跑；我们直接调 `torch._scaled_mm`(`fp8_linear.py:114`),不走 FlashInfer 的 autotuned `bmm_fp8`。对应 flashinfer #3255。
- [x] **B-4 text-only 判据是否选对** —— ✅ **被独立验证**。sglang #27212 举的例子正是 `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`;他们栽在按架构名强制 multimodal 且关不掉。我们按 `language_model_only` 判定 + 检测与策略分离是对的。

- [x] **B-5 · Makefile 的 gencode 形式拿不到架构特性**（本轮实测发现） —— ✅ **已修复并验证**
  `Makefile:51`（旧）用 `-gencode arch=compute_120,code=sm_120a`。**实测坐实**：门控在虚拟架构上，不在
  `code=` 上。用 CUDA 13.2 + 本地 `cutlass-4.6.1` 头文件实际编译 `runtime/kernels/nvfp4_gemm_sm120.cu`
  （未接入构建、纯探测）三种组合对比：
  - `arch=compute_120,code=sm_120a`（旧 flag）—— **编译成功但 4 个 block-scaled MMA 核体全部退化成
    `BPT.TRAP` + `EXIT`**（`.so` 体积 1,373,984 字节，SASS 仅 362 行，4 个函数每个约 90 行的跳转到
    trap 的桩）。这不是"编不出来"的编译错误，是**编译干净但一跑就崩**的运行期陷阱——比编译错误更危险。
  - `arch=compute_120a,code=sm_120a` 与 `arch=compute_120f,code=sm_120f` —— 两者产出**SASS
    指令级完全一致**（`.so` 体积同为 1,783,584 字节；`cuobjdump --dump-sass` diff 后除
    `.headerflags` 里 `EF_CUDA_ACCELERATORS` 标记（`a` 有、`f` 无，纯兼容性分类标记，非功能差异）外
    零差异）。`f` 变体的 ELF `sm=` 标记是通用 `sm_120`（可跨 120 家族加载，含 sm_121/DGX Spark），
    `a` 变体标记 `sm_120a`（只能在精确 sm_120 上加载）。
  - Router 本身（`laguna_router_sm120.cu`，只用 `__shfl_xor_sync` 基础 ISA）在旧 flag 与新 flag 下
    **SASS 逐行相同**，证实今天切换零风险、零回归。
  **结论**：`arch=compute_120,code=sm_120a` 是全项目里一个活的定时炸弹——只要有人把 router 现有 flag
  复制到 `nvfp4_gemm_sm120.cu`（或任何用 block-scaled MMA 的核）的构建规则上，产出的 `.so` 会通过编译、
  通过任何不启动 kernel 的检查，直到真正调用时才在 GPU 上炸。**已改为 `arch=compute_120f,code=sm_120f`**
  （`Makefile` `ROUTER_FLAGS` + manifest payload 的 `"target_sm"` 字段同步改为 `"sm_120f"`）。
  `make build-laguna-router` 与 `make verify-laguna-router` 均已重跑通过，产物 SASS 与旧 flag 逐行相同
  （功能零回归）。**去向**：已完成。**GPU**：不需要（仅编译对比，未上机执行）。

- [x] **B-6 · Qwen3.6 的 MTP 层是否带 GDN** —— ✅ **确认不带，但不能删掉 B3 那一项**
  详见 [`../notes/2026-08-01-b6-mtp-gdn-verification.md`](../notes/2026-08-01-b6-mtp-gdn-verification.md)。
  **事实**：本地全部 6 个 checkpoint 变体（`nvidia`/`unsloth`/`sakamakismile`/`morosystems`/官方
  `Qwen3.6-27B-FP8`/`cyankiwi` AWQ-INT4）的 `mtp.*` 张量集清一色是 `self_attn.{q,k,v,o}_proj` +
  `mlp.{gate,up,down}_proj` + 层归一化——与主模型 `full_attention` 层张量结构一致，**零个**
  `linear_attn.*`/`A_log`/`conv1d`/`dt_bias` 之类的 GDN 张量。`config.json` 本身不够（没有
  `mtp_layer_types` 这种字段，只有 `mtp_num_hidden_layers=1`），要靠张量名才能坐实。
  **但队列原本的推论有误，已纠正**：vLLM 那条注释（"draft models have no mamba layers, so no eagle
  shift"）说的是**草稿模型自身**的递归状态管理（drafting 阶段不需要 shift 自己的 GDN 状态，因为它没有）——
  这确实被本条证据消掉了。但 **verify 阶段仍然要把 MTP 提出的候选 token 整段跑一遍主模型的完整 64 层
  （含 48 层 GDN）**，一旦部分候选被拒绝，主模型的 GDN 递归状态已经被"没发生过的" token 污染，且这个更新
  不可逆——这个问题跟 MTP 头本身有没有 GDN 完全无关。用 WebSearch 核实到 vLLM 自己的
  `vllm-project/vllm#47572`（ReplaySSM RFC）原话："Speculative decoding must roll back rejected draft
  tokens, but the SSM state update is irreversible... the current implementation keeps a separate
  recurrent state per draft token"——这正是本文档 **D-3**（ReplaySSM，显存 11.5GB→1.8GB）已经独立记录
  的同一个问题。**对 roadmap 的影响**：Track B3"MTP draft / verify...含 GDN 递归状态的推测回滚"**不删**，
  但应改写为"主模型侧"问题并与 D-3 合并排期；可以删掉/减轻的是**草稿侧**的递归状态管理（MTP 头不需要自己
  的 conv/ssm state，不需要 eagle-shift 类操作）——这是比原队列设想更小但仍然真实的简化。
  **去向**：已完成（结论 + 纠偏）。**GPU**：不需要。

---

## C. 自查 —— 需要 GPU（留给开发执行）

> **状态（2026-08-02，晚间更新）**：C-1/C-2/C-3 三条已全部查完并结案。C-1 沿途挖出的
> 数值分歧（eager verify vs CG verify，kv_len≥400 起 argmax 翻转）**现已根因并定级**：
> 稠密 fp32 attention oracle 判定**两条路径在 attention 算子层面都是对的**，严重性从
> "静默送错 token"下调为"两条路径输出不同"，见下方条目与 [`../notes/2026-08-02-eager-verify-cg-verify-divergence.md`]
> (../notes/2026-08-02-eager-verify-cg-verify-divergence.md)。C-1 同时印证了 `roadmap.md` §6
> RK9（冷启动/首次真实形状路径的系统性覆盖不足）是真实存在的一类盲区，而不只是从 JIT
> 那个 bug 泛化出来的猜测。

- [x] **C-1 · warmup / autotune / CUDA Graph 捕获是否用真实形状** —— ✅ **成立，已修，但修复过程中挖到一个更严重的活 bug，未根因**
  详见 [`../notes/2026-08-01-c1-c2-gpu-investigation.md`](../notes/2026-08-01-c1-c2-gpu-investigation.md#c-1)
  和 [`../notes/2026-08-02-eager-verify-cg-verify-divergence.md`](../notes/2026-08-02-eager-verify-cg-verify-divergence.md)。
  CUDA Graph 捕获（`laguna_cuda_graph.py`/`laguna_dflash_cudagraph.py`）和
  `warmup_paged_attention_shapes()` 对它们覆盖的 contract 确实用生产真实容量，不是占位小形状——
  flashinfer #3255 字面那种模式在这两处不成立。但沿着 `warmup_paged_attention_shapes()`
  自己承认的缺口（`mode="verify"` 未被预热）往下查，**GPU 实测坐实**：DFlash 主模型的
  eager verify 回退（`_forward_verify_with_aux`）直接调用生产函数会 `ValueError` 崩掉——
  `SparkinferPrefillWorkspace.forward()` 不分 mode 永远用为 `extend` 设计的
  `eager_extend_work_items_capacity` 估算容量，套到 `verify` 契约上低估了。**已修**：改为按
  mode 分派，`verify` 用一次真实 eager planner（`create_paged_plan(enable_cuda_graph=False,
  mode="verify")`）在该组声明的最大容量上跑出真实数字（第一次尝试用 sparkinfer 的
  `plan_verify_graph_capacity` 也是错的——那是为 CUDA Graph 重放设计的不同调度策略，实测同样
  低估，已改用真实 eager planner 本身）。真正不足的维度是 `max_partial_rows`（硬编码 0），
  不是 `max_work_items`。

  **但修完之后做"贪心位精确交叉验证"时挖到更严重的问题**：容量修好后 eager verify 能跑了，
  但跟 CG-verify 路径数值不一致——kv_len=64 bit-exact，kv_len≥400 起 argmax 真的选错 token
  （峰值 raw logit 差 26.7），分界点不是 SWA window=512，双 slot 隔离排除了测试脚本副作用。
  **修容量的 bug 把一个响亮失败（`ValueError`）变成了沉默失败（悄悄送错 token）**。
  触发面确认（读代码）：今天 eager verify 只有一条触发路径——verify CG 启动期捕获失败——
  是潜伏风险，不是正在发生的活跃故障（本次所有冷启动 verify CG 都捕获成功）。**响应**：
  `QSR_DFLASH_REQUIRE_CG` 默认值从 `0`（降级但响亮）改成 `1`（拒绝启动），直到这个数值分歧
  被根因排查并修掉。根因排查需要 `bf divergence` 逐层定位，**独立立项，不在本次任务范围**。
  run record: `bf show 940b708aa0f8`。

  **根因与定级（2026-08-02 晚，独立立项已完成）**：造了一个稠密 fp32 causal attention
  oracle（`bfdiag/dense_attention_oracle.py`，完全不碰 sparkinfer 的 `create_paged_plan` /
  `paged_attention_forward`），在单层隔离、不回灌模型的条件下三方对比。结论：**kv_len=
  64/400/500 上 eager 与 CG-style 对 oracle 都是 cos ≥ 0.999997**，高于本仓库对"正确 kernel
  变体"的 ≥0.999991 标准——**两种分块策略在 attention 算子层面都是对的**，split-KV merge
  没有 bug。全模型那个 26.7 logit 差与 argmax 翻转来自 **MoE 的 top-10/256 离散路由把
  微小数值差放大**（本仓库历史数据显示正常工作的 MoE 本身就在 cos ~0.95–0.97，远比
  attention 的 ~0.999999 嘈杂）。

  **方法上的一个坑，值得记住**：第一次尝试把 oracle 接进全部 12 个 full-attention 层，在
  **kv_len=64**——那个已被证明 CG 与 eager 完全 bit-exact 的点——oracle 与它们的共识只有
  cos=0.845。没有采信：两个独立实现且互相 bit-exact 的路径共享同一个 bug，远不如"新写的
  oracle 有 bug"来得可能。改单层隔离后 oracle 对真实 eager 输出 cos=0.999999，证明 oracle
  数学本身没错，混淆源正是 MoE 放大。**多层同时替换 + 活体前向 = 不可用的判据。**

  同时更正上一轮一个真实错误：之前的 chunk-size 探测用了 sparkinfer 自带的 TP=2 分片形状
  提示（24Q/4KV），而本部署真实是 TP=1（48Q/8KV，见 `model-support.md`）；数字已更正
  （17600 → 37632 tokens），mismatch 计数结论不变。

  **由此改变的决策**：`QSR_DFLASH_REQUIRE_CG=1` **保持默认，但理由换了**——不再是"merge
  内核未经验证"，而是"本运行时承诺贪心逐位可复现，中途回退到 eager 会静默毁掉这个承诺"。
  同一类失败，从可复现性而非算术抵达。方向 1（让 eager 也塌成单 chunk）现在风险更低，
  但**只会缩小、不会消除**分歧，因为 MoE 会放大任何残余实现差异。

  **诚实的缺口**：12 个 full-attention 层只逐层验证了第 0 层；一个手搓的 "CG-style"
  workspace helper 有容量估算 bug，导致 OOM 后 `cudaErrorIllegalAddress` 污染了 daemon 的
  CUDA context，遂停手而非反复重试（该 helper 未进仓库）。
  详见 [`../notes/2026-08-02-eager-verify-cg-verify-divergence.md`](../notes/2026-08-02-eager-verify-cg-verify-divergence.md)。

- [x] **C-2 · NVFP4 KV vs FP8 KV 在我们卡上的 prefill 对比** —— ✅ **查完：这个对比在当前技术栈上跑不起来，理由比预想更硬**
  详见 [`../notes/2026-08-01-c1-c2-gpu-investigation.md`](../notes/2026-08-01-c1-c2-gpu-investigation.md#c-2)。
  SparkInfer 的 paged-attention 内核（唯一的 attention 内核，零 FlashInfer 依赖）只接受
  fp16/bf16/fp8_e4m3 三种 KV dtype（`sparkinfer/attention/paged/traits.py:120-121` 显式
  `TypeError`），本 runtime 自己也三处硬编码 `kv_cache_dtype="fp8"`。所以 flashinfer #4269
  （第三方在 RTX PRO 5000 上测的 NVFP4 KV prefill 慢 1.7–1.8x）在我们的栈上**连对照组都不
  存在**——`bf diff` 判可比性这步在"跑第二个配置"就跑不下去，不是被忽略。退而测了唯一真实
  存在的 FP8 KV 在生产真实形状（`block_size=64`、`blocks_per_slot=4096`，走
  `backend.prefill_with_aux`）上的 prefill 基线：64 tok 284ms、512 tok 146ms、2048 tok
  331ms、8192 tok 1106ms、32768 tok 5048ms、16384 tok(全新长度)2313ms——未观察到
  30-100s 级别的重编译尖峰。**结论支持原计划：选 FP8 KV，不扩 NVFP4 到 KV**，且门槛从
  "跑得起来但慢" 升级为 "内核库现在直接不支持，要支持得先让 SparkInfer 团队新增内核路径"。
  run record: `bf show 940b708aa0f8`（同一次 `bf exec`）。

- [x] **C-3 · PyTorch 2.13.0 PyPI wheel 是否带 `sm_120`** —— ✅ **带，自编译要求终结**
  详见 [`../notes/2026-08-01-c3-torch-pypi-wheel-sm120.md`](../notes/2026-08-01-c3-torch-pypi-wheel-sm120.md)。
  干净 venv 跑指定命令，实测：
  ```
  2.13.0+cu130 ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
  ```
  `sm_120` 在列。`pyproject.toml` 对 `torch` pin 的注释里已经预判"若装了公开版 wheel 也满足合同"，
  这条把预判坐实成实测结论。**未验证的部分**（未过度声称）：本机参考环境跑的是自编译
  `2.13.0a0+gitcf30153`（对齐 CUDA 13.3），PyPI wheel 走 `nvidia-*-cu13` 传递依赖（CUDA 13.0）——
  两者 CUDA 子版本有差，数值/调优路径是否逐 bit 一致未验证；`nvidia-cutlass-dsl` 等其余 `cuda` extras
  是否与纯 wheel 版 torch 一起装能装成也未重新验证（本条只测了 torch 单项，按队列给的命令原样执行）。
  **对 roadmap 的影响**：**H1"依赖可从公开源安装"仍然被 sparkinfer 上游化（RK2）卡着**，不是被 torch
  卡着——这条不解锁 H1，但去掉了"torch 也要求自编译"这一层叠加风险。**RK6**（依赖链漂移）不因此关闭，
  CUDA 13.0 vs 13.3 的子版本差本身就是 RK6 类风险的一个小实例。`pyproject.toml` 该注释段落建议由该文件
  owner 更新为"已实测确认"而非"预期满足"（本次未改 `pyproject.toml`，超出本轮文件归属范围）。

---

## D. 值得吸收的上游做法（ADOPT 候选，按价值排序）—— ✅ 2026-08-01 已全部消化

> 分工：**kernel 形状 → 写清楚交给 SparkInfer 团队**（按 `AGENTS.md` 规矩不直接改其源码）；
> 调度 / 缓存 / API / 模型支持 → 我们自己做。
>
> **本节 D-1～D-8 已全部处理**：分派到 `roadmap.md` 对应 Track 与 `implementation-plan.md`
> 的执行条目，保留在此处作为triage 记录（不删除，同 D-1 的先例）。

- [x] **D-1 · 混合缓存（KV + 递归状态）先例** —— ✅ **已读并转成 A3 设计输入**
  见 [`../notes/2026-08-01-hybrid-cache-prior-art.md`](../notes/2026-08-01-hybrid-cache-prior-art.md)。
  产出 6 条对 A3 的修改（抽象拆分、双数字前缀匹配、投机保守释放、同轮不可跨请求借用、逐资源驱逐预算、块大小对齐）。
  **A3 动工前必读。**

- [x] **D-2 · DSpark：置信度驱动的自适应 verify 窗口** —— ✅ **升级为 P1，见 `roadmap.md` Track F /
  `implementation-plan.md` §7.6/F1**。分块半自回归起草，**用草稿自身置信度决定每个 verify 窗口
  大小**。SGLang v0.5.16 报 383.7 tok/s、接受长度约 5；vLLM 也有，两边都有 tracking（sglang #30344），
  参数 `--speculative-dspark-block-size`。**为什么对我们成立**：我们 DFlash 是固定
  `NUM_SPECULATIVE_TOKENS=15`,而接受率实测 **96.3–100%**——限制吞吐的**可能是固定窗口而不是接受率**。
  **警告**：vllm #49369 报告 DSpark 在某负载上比不开投机还慢，不是白捡。**去向**：我们做（投机策略），
  已排出两步（先静态调宽，再考虑自适应）。

- [x] **D-3 · ReplaySSM Ring Spec-Verify：投机 scratch 显存降 6.4×** —— ✅ **升级为 P1，见 `roadmap.md`
  Track F / `implementation-plan.md` §7.6/F2**。11.5 GB → 1.8 GB 是别人的卡、别人的形状，不能直接当
  我们的数字用；Laguna 权重已占 67 GB（`notes/2026-07-29-gpu-memory-audit.md` 逐项相加，非猜测），
  96 GB 卡上给 KV + 投机 scratch 的预算很紧——投机 scratch 在和 KV 抢这块。**去向**：我们做（先补显存
  审计 + 判断多少能在调度层拿到），若节省在 kernel 侧则转 SparkInfer。

- [x] **D-4 · 每 KV-cache group 选不同 attention backend；sliding-window 作为显式 backend capability**
  —— ✅ **验证了 Track A 的既有设计方向，追加一条设计备忘**，见 `roadmap.md` Track A。
  vLLM v0.26.0。与我们今天落地的 A2 `BackendCapabilities` 独立收敛到同一设计；A1 已按层类型序列
  描述架构，Qwen3.6 的 16 full + 48 GDN 混合正是这个设计要接住的形状。**唯一具体补充**：滑窗应从
  "模型图内部隐式处理"提升为 `ModelSpec`/能力查询里可查询的显式字段。**去向**：我们做（A1/A2 实现时）。

- [x] **D-5 · Hybrid (SWA + full) DFlash drafters；投机专用 `kv_cache_dtype`** —— ✅ **读代码后核实：
  不完全对，已从待办移除**，见 `roadmap.md`/`implementation-plan.md` §7.6。①投机专用 `kv_cache_dtype`
  **已经是现状**——`laguna_dflash.py` 里 draft KV cache 按 `# Self-allocated: FP8 as uint8` 分配，
  与主模型该层自己的 dtype 选择独立；②我们的 draft 模型走的是另一条路——固定 6 层全 SWA
  （window=512）、bf16 权重，KV cache 只有 0.007 GB（`notes/2026-07-29-gpu-memory-audit.md`），已经
  靠"专用小 draft 模型"达到类似的省显存效果，没有证据表明改成 vLLM 式 hybrid drafter 会更好。
  **这不是降级，是核实后发现已经做到。**

- [x] **D-6 · FlashAttention 的 SM120 路径进展** —— ✅ **保持"持续观察，不是现在动"**，已写入
  `roadmap.md` Track F（含完整 T0 触发条件）。维护者已合入 sm120 PR（#2413，"WIP"），并有**面向 5090
  的 TMA + warp specialization** PR 在做（#2440）——正是 `implementation-plan.md` F4（原编号 F2，
  Track F 重排后顺延）计划要移植的技法。但：**FA4 算法本体上不了 SM120**（缺 tcgen05/TMEM）；当前
  sm120 路径只有 FP16/BF16、`main` 上部分路径仍报错、在 5090 上比 FA2 **慢约 5%**。**T0 触发条件**：
  那批 PR 落到 main 且在 sm120 上跑赢 FA2 —— 到那时才从"自己移植"变成"评估采纳"。

- [x] **D-7 · NVFP4 per-token online MoE 量化**（vLLM v0.26.0）+ CuTe-DSL MXFP4 —— ✅ **排入
  `implementation-plan.md` §7.6/F10**。Laguna 是 256 专家 NVFP4 MoE，直接可比。**去向**：kernel 形状
  → **写清楚交给 SparkInfer**，我们自己只写一份技术提案文档，不实现。

- [x] **D-8 · chunked input-logprob 默认开启（削峰值显存）**（SGLang v0.5.16）—— ✅ **排入
  `roadmap.md` Track E / `implementation-plan.md` §7.5/E5，提前到 M2**。我们有 logprobs 路径、双协议
  都暴露 `top_logprobs`;单卡上长 prompt 的峰值显存是真问题。小而自足，不依赖 Track A。
  **去向**：我们做（Track E / 显存）。

---

## E. 依赖与工具链（低优先，但会静默变坏）

- [ ] **E-1 · Triton 3.7.1 / 3.7.0 已发布**,本机 3.6.0，钉的 `>=3.6,<4` 允许。
  我们的 Triton kernel（`rope` / `fused_rms_norm` / `fused_kv_scatter`）和 FLA 都吃 codegen 行为。升级前先读 changelog。
- [ ] **E-2 · CUTLASS 4.6.1 / 4.6.0 / 4.5.3 已发布**,本机 4.5.2，钉 `>=4.5,<5` 允许。
  关注点：SM120 blockscaled GEMM 变化；**打包结构变化会破坏 `laguna_sparkinfer_moe.py` 的 sys.path workaround**。
- [ ] **E-3 · vLLM v0.26.0 release notes 尚未通读** —— 本轮只读了摘要。混合前缀缓存部分尤其值得读（本地检出是 0.25.1，落后一个版本）。
- [ ] **E-4 · 生态扫描未覆盖的源**：Tier 2 的 A 组（nano-vllm / llama.cpp / TileLang）与 C 组（量化工具链）。
  本地都有检出：`/home/bot/project/{nano-vllm,llama.cpp,tilelang,exllamav2,TensorRT-LLM,DeepGEMM,SageAttention,Model-Optimizer,KIVI,sm120-flash-attention}`。

---

## F. 已确认无需行动（记录以免重复排查）

- **FLA v0.5.2 就是本地版本**,我们是最新的。Blackwell 相关 bug（#790/#945/#999/#727）**全部是 B200/SM100，没有一条 SM120**,且多已关闭 —— FLA 在这张卡上仍**未经验证**,与基线一致。
- **sparkinfer 上游 0 领先**。`3bd3a2e` 本身是把上游 `b0976b7` 合进来的 merge，fork 另有 23 个自有提交。
- **没有新一代 Laguna**。poolside 最近修改的十个模型全是 2.1 系；存在 `Laguna-XS-2.1`（更小，不是更强），过不了模型雷达门槛。
- **`sm_121` 是 DGX Spark**（flashinfer #3170）—— 这就是 `compute_120f` 家族前向兼容所指向的芯片。
- **SGLang / FlashInfer 确实在做 SM120**,但目标是大 MoE / MLA + 多卡栈；单卡固定槽位仍是我们的面。
  **注意**：这条**不构成"差异化收窄"的结论**——他们支不支持 SM120 不是发现，他们**做出了什么**才是（见 D 节）。

---

## 处理建议顺序

1. ~~**A 节三条拍板**~~ —— ✅ **2026-08-01 已完成**（A-3 的 (c) 已实现并合入 main，见
   `server/engine.py` / `server/app.py` 的启动期拒绝，以及 `tests/test_engine_session_affinity.py`）
2. ~~**B-5、B-6**~~ —— ✅ **2026-08-01 已处理**：B-5 已改 Makefile 并验证（旧 gencode 组合会让
   block-scaled MMA kernel body 退化成 `BPT.TRAP` 桩——干净编译背后的运行时崩溃）；
   B-6 确认 MTP 张量零 GDN，但**纠正了"会删掉 B3 一整项"的推论**——vLLM 那条注释指的是
   draft 模型自己的递归状态，不是主模型的 48 个 GDN 层，后者在 verify 时照样跑、照样需要
   回滚方案。见 `notes/2026-08-01-b6-mtp-gdn-verification.md`。
3. ~~**C-3**~~ —— ✅ **2026-08-01 已处理**：PyPI `torch==2.13.0` 带 `sm_120`，自编译要求终结
   （不解锁 H1，仍卡 sparkinfer 上游化 / RK2）。见 `notes/2026-08-01-c3-torch-pypi-wheel-sm120.md`。
4. **A3 动工前读 D-1 的笔记** —— 已读完，见 D-1
5. ~~其余按 Track 排期并入 implementation-plan.md~~ —— ✅ **D 节全部（D-2～D-8）已排期**，见 §D
   逐条的分派记录；C-1/C-2 结论已回（见 §C），其下游（Track B3 措辞收窄、与 D-3 合并排期）
   已在 roadmap 更新
