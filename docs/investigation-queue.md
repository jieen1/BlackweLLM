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

- [ ] **B-5 · Makefile 的 gencode 形式拿不到架构特性**（本轮实测发现）
  `Makefile:51` 用 `-gencode arch=compute_120,code=sm_120a`。实测：**门控在虚拟架构上，不在 `code=` 上**——
  `compute_120,code=sm_120a` 编不出 NVFP4 block-scaled MMA，`compute_120f/a` 才能。
  今天**无功能 bug**（router 只用 `__shfl_xor_sync`,属基础 ISA；`nvfp4_gemm_sm120.cu` 无构建规则）。
  代价：拿了 `a` 的兼容性限制（编不到 sm_121 / DGX Spark），没拿到 `a` 的特性；且 manifest 记 `"target_sm": "sm_120a"` 会强化误解。
  **建议改 `arch=compute_120f,code=sm_120f`**：特性相同、多一份家族可移植、语义与实际相符。
  **去向**：我们做。**GPU**：不需要（仅编译）。

- [ ] **B-6 · Qwen3.6 的 MTP 层是否带 GDN** —— **可能删掉 B3 最难的一项**
  vLLM 注释：*"draft models have no mamba layers, so no eagle shift"*——若我们的 MTP 层不含 GDN，**递归状态推测回滚这件事根本不存在**。
  查法：读 checkpoint `config.json` 里 MTP 段的 `layer_types` + `mtp.*` 张量名。
  **去向**：我们做。**GPU**：不需要。**应在 B3 排期前答掉。**

---

## C. 自查 —— 需要 GPU（留给开发执行）

> **状态（2026-08-01）**：C-1/C-2/C-3 三条**由另一个并行 agent 在查**——`roadmap.md`/
> `implementation-plan.md` 已把下游结论（B3 的 GDN 分支、KV dtype 选型、RK6/H1 的 sm_120 wheel）
> 写成显式 [待验证]，**不预判**任何一条的结论。C-1 与 `roadmap.md` §6 RK9（冷启动/首次真实形状
> 路径的系统性覆盖不足）是同一类问题，C7 审计（`implementation-plan.md` §7.3）已把 DFlash
> 的一个具体已知缺口（verify 路径预热覆盖）纳入这条调查范围。

- [ ] **C-1 · warmup / autotune / CUDA Graph 捕获是否用真实形状**
  依据 flashinfer #3255：失败**不在** autotuner 第一个小的合成形状上，而在后面一个匹配真实模型维度的形状上。
  本项目已有同型伤疤（fp8 舍入平局：合成随机数据复现不出真实数据的 bug）。
  查：CUDA Graph 捕获与 DFlash warmup 用的是生产形状还是占位形状。

- [ ] **C-2 · NVFP4 KV vs FP8 KV 在我们卡上的 prefill 对比**
  依据 flashinfer #4269：第三方在 RTX PRO 5000 Blackwell 上实测 **NVFP4 KV 的 paged causal prefill 比 FP8 KV 慢 1.7–1.8 倍**,而 decode 更快（带宽瓶颈）。
  不是我们的卡也不是我们的形状。用 `bf diff` 判可比性后再比数。**支持 B3 选 FP8 KV，并警告别把 NVFP4 扩到 KV。**

- [ ] **C-3 · PyTorch 2.13.0 PyPI wheel 是否带 `sm_120`** —— *其实不需要 GPU*
  ```bash
  python -m venv /tmp/torch-probe && /tmp/torch-probe/bin/pip install -q torch==2.13.0 \
    && /tmp/torch-probe/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
  ```
  带 `sm_120` → 自编译要求终结，解锁 RK6 与 H1"可从公开源安装"。不带 → 确认自编译留着。

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

1. ~~**A 节三条拍板**~~ —— ✅ **2026-08-01 已完成**（A-3 的 (c) 实现清单仍待落地，见
   [`implementation-plan.md`](implementation-plan.md) §6.1）
2. **B-5、B-6**（零 GPU，各半天，B-6 可能删掉 B3 一整项）—— **B-6 由另一 agent 在查，本轮不重复**
3. **C-3**（零 GPU，一条命令）—— **由另一 agent 在查，本轮不重复**
4. **A3 动工前读 D-1 的笔记** —— 已读完，见 D-1
5. ~~其余按 Track 排期并入 implementation-plan.md~~ —— ✅ **D 节全部（D-2～D-8）已排期**，见 §D
   逐条的分派记录；C-1/C-2 由另一 agent 在查，不预判结论，结论回来后其下游（Track B3、RK6/H1）
   会需要再更新一轮
