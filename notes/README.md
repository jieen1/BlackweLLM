# notes/ 索引

> 编制日期：2026-08-01 · 共 120 篇（`git ls-files notes | wc -l`）

`notes/` 是**调查记录与证据档案**，不是文档。它的价值在于：当一个结论被
质疑时，能翻出当初的实测数据、复现命令和被排除的假设。

**这些文件不做物理归档**——`runtime/`、`server/`、`block_pool.py` 等处的
代码注释直接按路径引用它们（例如 `runtime/model_loading.py` 引用
`notes/2026-07-27-vllm-complete-removal-implementation-plan.md`），
移动会把这些引用全部打断，代价大于收益。取而代之，本索引标注每篇的**时效状态**。

## 状态标记

| 标记 | 含义 |
|---|---|
| 🟢 **有效** | 结论仍然成立，可直接引用 |
| 🟡 **历史证据** | 描述的是已经改变的状态，但实测数据仍有参考价值 |
| 🔴 **已过时** | 前提已不成立（vLLM 已剥离 / Qwen3.6 租户已摘除 / 结论已被推翻） |

---

## 1. 当前有效的设计与方法论 🟢

接入新模型、排查问题时**应该先读这些**。

| 文件 | 内容 |
|---|---|
| `2026-07-27-probe-system-design-and-plan.md` | bfdiag 探针系统设计（`docs/diagnostics-guide.md` 的设计背景） |
| `2026-07-27-bfdiag-*.md`（7 篇） | 飞行记录仪 / run record / 可比性判定 / 热引擎 / 形状推导 / 确定性 / oracle 分歧 |
| `2026-07-27-bfprobe-*.md`（2 篇） | 探针签名与 MoE 路由抓取 |
| `prefix-cache-design.md` | 前缀缓存设计 |
| `2026-07-27-laguna-prefix-cache-lp1-design-draft.md` | L-P1 同轮 fan-out 设计草稿，**当时未实现**。2026-08-01 从 `laguna-prefix-cache` 分支抢救（该分支的 L-P0 block-table 改动已在 main，只有这份草稿没有） |
| `sglang-radix-cache-study.md` | radix cache 的对照研究 |
| `2026-07-23-e1-model-abstraction-design.md` | **模型抽象层的早期设计**——Track A 的先验，必读 |
| `2026-07-23-gpu-verification-checklist.md` | GPU 验证清单 |
| `reference-map.md` | 参考实现地图 |

## 2. 已定案的根因分析 🟢

- [2026-08-03 CUDA Graph vs eager 解码吞吐 = 4.71×](2026-08-03-cudagraph-vs-eager-decode-throughput.md) —— 🟢 **在册每个吞吐数字都是 eager 的，比运行时实际能力低约 5 倍**。服务路径同 prompt 同参数只切一个开关：CG **28.848** vs eager **6.120 tok/s**，且 CG 还少用 5.30 GiB。捕获本身不贵（启动 +4s）。⚠️ **所有基于 ~6 tok/s 的优化判断都需重估**；另纠正一条我自己的误判（311s 冷启动是磁盘读 22 GiB，不是 JIT 前置）
- [2026-08-02 GDN spec_forward 批处理](2026-08-02-gdn-spec-forward-batching.md) —— 🟡 19.9→12.0ms 的计时仍有效；但"大投影不能批处理"的否决与"torch.bmm 512 上限"这条推广**已被下一条推翻**
- [2026-08-02 批处理大投影：bit-exact 不是这里的判据](2026-08-02-spec-verify-batching-bar.md) —— `verify_forward` 里的 bit-exact 早已不存在（layer 1 状态差 0.0117 / 73% 元素）；全模型 672 步的 B1-R gap error 与 shipped 路径**完全相同**（p90=0.250，bar 0.5）；单层 K=16 再快 3.15×、全模型 verify 1.40–1.57×；e2e【推算】0.79–0.91×，仍 <1.0×
- [GDN 多步融合 kernel 规格](2026-08-02-handoff-sparkinfer-gdn-multistep-kernel.md) —— ⚠️ 其 ~6.8ms 归因已被实测推翻（真值 ~0.8–1.0ms）；kernel 已实现并 bit-exact，但不是决定性项
- [2026-08-02 Qwen3.6 显存底线由反量化缓存决定](2026-08-02-qwen36-dequant-cache-memory-floor.md) —— 一次完整前向后常驻 19GB→54GB+，`GPU_MEM_UTIL`/`num_slots` 全部管不到
- [2026-08-02 B1 对齐门禁的证据链（原门禁已作废）](2026-08-02-b1-greedy-alignment-fails.md) —— 分歧只差 1–2 个 bf16 ULP，证明原门禁要求的东西不存在；含一条被自我推翻的误报
- [2026-08-02 第 7 步 GPU 验收结果](2026-08-02-a3-step7g-gpu-acceptance-results.md) —— bit-exact/前缀缓存/C-LIVE 三条通过；接受率那条对照组错了需重做，另发现 4 条与 7-g 无关的下降
- [2026-08-02 Qwen3.6 历史性能记录](2026-08-02-qwen36-historical-performance-record.md) —— 稳态 decode 曾快于 vLLM 1.33×，输在 TTFT；并厘清哪些"超越 vLLM"其实是 Laguna 的数字
- [2026-08-02 接受率没有可用的观测路径](2026-08-02-acceptance-rate-has-no-working-observability.md) —— 正式验收判据，但两个记录函数生产零调用 + 直方图 5 桶装不下 K=15
- [2026-08-02 anchor token 恒为贪心 argmax](2026-08-02-anchor-token-ignores-sampling-params.md) —— 所有 temperature>0 用户的首 token 是确定性的；使 E-N1 的 (b) 方案作废
这些是"教训档案"，接入新模型时逐条对照（见 `docs/model-support.md` §6）。

| 文件 | 教训 |
|---|---|
| `2026-07-27-block-size-128-accept-rate-root-cause-CLOSED.md` | 页大小变更导致接受率静默下降 |
| `2026-07-27-block-size-128-migration-and-tie-break-noise.md` | 同上，含浮点平局噪声 |
| `2026-07-27-fused-kv-scatter-value-stride-bug-ROOT-CAUSE-FOUND.md` | KV scatter stride 写错，长上下文才显形 |
| `2026-07-27-fused-kv-scatter-negative-slot-bug-fixed.md` | 负 slot 索引 |
| `2026-07-27-fused-kv-scatter-bf-attention-regression-investigation.md` | 同族回归调查 |
| `2026-07-23-laguna-bos-root-cause.md` | BOS token 处理 |
| `2026-07-27-ptxas-ice-diagnosis.md` | ptxas 编译器崩溃诊断 |
| `2026-07-29-cg-slot-doubling-fix.md` | CUDA Graph 槽位重复分配 |
| `2026-07-24-moe-b12x-cudagraph-incompat-fix.md` | MoE kernel 与 CUDA Graph 不兼容 |
| `2026-08-01-bfdiag-assertion-audit.md` | **bfdiag 断言可信度审计(Track C0)**——`reset_slot` 隔离保证失效(checkpoint/restore + daemon reset 两处都依赖已经不成立的"reset_slot 会清零 KV");两份手册误称 `reconcile_prefix_hit` 是 stub(实为生产代码);已删除的 `bug_found_not_fixed` 条目;`bfdiag/` 其余模块的真假断言普查 |
| `2026-08-01-prefill-shape-buckets-root-cause.md` | **prefill 每轮 30+ 秒卡顿**——真正的编译缓存键轴是 sparkinfer `page_table` 宽度（`kv_len+qo_len` 的函数），不是最初怀疑的 `q.shape[0]`（源码 + 本机实测编译缓存直接证伪）；修复是让 `SparkinferPrefillWorkspace` 按 `(mode, window_left)` 建固定容量 workspace，不是按形状重建 |
| `2026-08-01-c1-c2-gpu-investigation.md` | **C-1/C-2 GPU 排查**——CG 捕获/warmup 用的都是生产真实容量，但顺着 warmup 缺口往下查，发现 DFlash 的 eager verify 回退（`_forward_verify_with_aux`）会直接 `ValueError` 崩掉（GPU 实测坐实，容量估算函数按 extend 语义误用到 verify 契约，后续在 2026-08-02 那份笔记里发现连"按 verify 图容量规划器估"这个第一次修复也是错的，真正修法是跑一次真实 eager planner）；NVFP4 KV vs FP8 KV prefill 对比测不了——SparkInfer 内核只认 fp16/bf16/fp8，这个 runtime 也三处硬编码 fp8 KV，没有第二个配置可比 |
| `2026-08-02-eager-verify-cg-verify-divergence.md` | **eager verify 修完容量能跑了，但和 CG-verify 数值不一致**（未根因，独立立项）——kv_len=64 bit-exact，kv_len≥400 起 argmax 真的选错 token（峰值 raw logit 差 26.7），分界点不是 SWA window=512；双 slot 隔离排除了测试脚本副作用假设。把"响亮失败"变成了"沉默失败"，`QSR_DFLASH_REQUIRE_CG` 默认值因此从 `0` 改成 `1`（拒绝启动）。触发面确认：今天生产里 eager verify 只有一条触发路径（verify CG 捕获失败），全仓库搜索排除了其它旁路——是潜伏风险，不是正在发生的活跃故障 |

## 3. SM120 kernel 研究 🟢

Track F（性能，机会主义）的输入。

| 文件 | 内容 |
|---|---|
| `2026-07-31-research-synthesis.md` | **8 个方向的研究综述**——FA4 可移植性、CUTLASS SM120、MoE 带宽、FlashInfer 分派、Warp Decode |
| `2026-07-31-sm120-flash-attention-kernel-research-for-sparkinfer.md` | 自研 attention kernel vs sparkinfer 对比（自研慢 2.4–3×） |
| `2026-07-31-sm120-nvfp4-gemm-research.md` | NVFP4 GEMM |
| `2026-07-31-fa4-optimization-roadmap.md` | FA4 技法路线 |
| `fa4-sm120-portability-research.md` / `fa4-sm120-port-research.md` / `fa4-sm120-adaptation-analysis.md` | FA4 移植性三连 |
| `research-warp-decode-flashinfer-deepgemm.md` | Warp Decode / DeepGEMM |
| `2026-07-23-cutlass-fp4-moe-fusion-design.md` | CUTLASS FP4 MoE 融合 |
| `2026-07-27-dflash-bandwidth-roofline-moe-gemm-attention.md` | 带宽 roofline |

## 4. 最近的性能与状态记录 🟢

| 文件 | 内容 |
|---|---|
| `2026-07-31-session-summary.md` | **当前最佳配置与性能基线**（4K 353–401 tok/s、64K 353–368、接受率 96.3–100%） |
| `2026-07-29-perf-optimization-results.md` / `-perf-profiling-analysis.md` | 优化结果与 profiling |
| `2026-07-29-acceptance-regression-baseline.md` | 接受率回归基线 |
| `2026-07-29-dflash-acceptance-incident-summary.md` | **DFlash M=16 接受率事故的停机状态记录**——2026-08-01 从 `vllm-removal-phase1` worktree 的未提交状态中抢救，此前从未被任何提交收录 |
| `2026-07-29-dflash-m16-execution-plan.md` | 同上批抢救。含 vLLM 参考的 prompt hash、接受率 0.687、warm 吞吐三次采样——这组数字是后续所有 DFlash 对比的口径来源 |
| `2026-07-29-gpu-memory-audit.md` | 显存审计 |
| `2026-07-29-moe-ab-test.md` | MoE A/B |
| `2026-07-22-quality-baseline-and-official-scores.md` | 质量基线与官方分数对标方法论 |
| `2026-08-02-evaluation-artifact-provenance.md` | **评测产物归属存根**——五份产物的 `model` 字段全是 `qwen3.6`，仓库里没有任何 Laguna 评测数据。因 `evalplus_results/` 被 gitignore、随时可能消失而固化 |
| `2026-08-02-track-a-step5-gpu-verification.md` | Track A 第 5 步 GPU 验收：贪心 bit-exact 用真实 revert worktree 实测确认（非推理）；接受率逐工作负载几乎精确匹配基线；**fox-64K tok/s 低基线 ~19%，与第 5 步无关**（bfdiag provider 绕过 `ServerEngine`）**，用直接 A/B 排除了 C-1 verify 容量修复**（回退容量计算、保留 `REQUIRE_CG=1`，数字与信号形状原样复现），**又靠读代码（零 GPU）排除了前缀缓存假设**（`enable_prefix_cache=False` 在两层都生效，`_prefill_with_prefix_hit` 在这个 benchmark 里根本不可达），仍未根因；对基线数字本身的可信度也给了直接判断；记录了复现基线所需的完整环境变量三件套（漏一个就测不准） |
| `2026-08-02-prefix-cache-baseline.md` | Track A 7-g「前缀缓存命中率不回归」的**切换前基线**（`main@1cf482f`，未合并）。核实并纠正了 `docs/implementation-plan.md` 阻塞记录里对 `hits=1/misses=10/hit_rate=0.0909` 成因的猜测（读代码证实：admission bootstrap check 走独立的 `runner.prefill(ref_slot,...)`，从不经过计数路径；真正原因是 e2e 脚本多个子测试共享同一进程级、无重置、跨会话的计数器）；论证 `hit_rate` 即便正确限定范围仍对"浅命中"不敏感，推荐改用逐轮 `hit_L / ideal_L`（`ideal_L` = 上一轮 `prompt_tokens` 按 block_size 取整）当回归信号；固定负载（8 组独立 6 轮增长对话，各带随机 marker 防串味）+ 逐轮完整序列（不只汇总），配套脚本 `benchmarks/prefix_cache_baseline.py` 供 7-g 后原样重跑做 A/B |

## 5. Laguna 专属实现记录 🟡

Laguna 仍是生产模型，这些记录有效；但其中的 SWA ring、MoE、DFlash 等
机制**不适用于 Qwen3.6**（无滑窗、稠密 FFN、MTP 而非 DFlash）。

`2026-07-22-laguna-l0-*`、`2026-07-23-laguna-*`、`2026-07-23-swa-ring-kv-investigation.md`、
`2026-07-24-sparkinfer-*`、`2026-07-27-dflash-*`（7 篇）、`2026-07-27-laguna-*`、
`2026-07-27-swa-ring-align-granularity-*`、`2026-07-27-verify-cg-mode-fix-*`、
`2026-07-27-decode-cg-server-integration.md`、`2026-07-27-sparkinfer-*`（5 篇）、
`2026-07-28-laguna-m16-baseline-plan.md`、`2026-07-27-speed-repro-verified.md`、
`2026-07-27-allocator-sensitivity.md`

> 其中 `2026-07-27-dflash-net-negative-conclusion-superseded.md` 文件名已自带
> "superseded" 标记——该结论后来被推翻，DFlash 是净正收益。

## 6. vLLM 剥离主线 🟡

**剥离已完成**（`a9cb932`，2026-07-30）。这些是执行记录，不是待办。

`2026-07-27-vllm-complete-removal-implementation-plan.md`（仍被 `runtime/model_loading.py`
等处的注释引用，**不要移动**）、`2026-07-27-vllm-flashinfer-dependency-audit-and-decoupling-roadmap.md`、
`2026-07-27-vllm-version-decision.md`、`2026-07-29-vllm-removal-status.md`、
`2026-07-22-vllm-fork-archive.md` + `.patch` + `-sm120_gqa-backup.py`、
`2026-07-23-vllm-baseline-final.md`、`2026-07-23-vllm-dflash-baseline.md`、
`2026-07-23-vllm-fused-moe-cutlass-flashinfer-analysis.md`、
`2026-07-27-acceptance-rate-gap-vllm-vs-ours-same-prompt.md`、
`2026-07-27-dflash-fair-comparison-vllm-parity.md`

## 7. Qwen3.6 时代的记录 🔴 → 但对 Track B 是先验

前提已不成立（那套实现依赖 vLLM，已移入 `oracle/qwen36_vllm/`），
**但 Qwen3.6 重新接入时这些是最有价值的参考**，尤其是 GDN 相关的。

| 文件 | 为什么还要读 |
|---|---|
| `2026-07-22-a1a-gdn-profiling.md` | **GDN profiling 数据**——Track B 的 GDN 方案选型直接输入 |
| `direct-model-runner-design.md` | 旧 Qwen3.6 runner 的整体设计 |
| `2026-07-22-a3-mtp-fusion-analysis.md` | MTP 融合分析——Track B3 的先验 |
| `2026-07-20-inv8-chunked-hit-prefill-plan.md` | 分块 prefill 与前缀命中（含 GDN 状态处理） |
| `phase-0-baseline.md` / `2026-07-24-phase1-ground-truth.md` | 早期基线 |
| `2026-07-19-p3-implementation-plan.md` | 前缀缓存 P3 实施计划（含 GDN checkpoint 池设计） |

## 8. 已过时的状态快照 🔴

这些是"某个时刻的进度快照"，全部已被 `docs/roadmap.md` 取代。
保留仅作历史，**不要据此判断当前状态**。

`FULL_STATUS_20260726.md`、`STATUS_dflash_acceptance.md`（其中 35.83% 接受率的
问题已解决，当前 96.3–100%）、`STATUS_bf_attention_integration.md`、
`STATUS_speed_optimization_0726.md`、`STATUS_speed_reproduction_0726.md`、
`ANALYSIS_speed_regression.md`、`2026-07-17-post-ragged-round-next-steps.md`、
`2026-07-19-comprehensive-audit-and-forward-plan.md`、
`2026-07-20-comprehensive-optimization-plan.md`、
`2026-07-20-evidence-based-optimization-plan.md`、
`2026-07-20-kernel-optimization-plan.md`、
`2026-07-20-cold-prefill-*.md`、`2026-07-21-kernel-comprehensive-review.md`、
`2026-07-24-comprehensive-benchmark-results.md`、`2026-07-24-step-latency-analysis.md`、
`2026-07-24-baseline-marlin.md`、`2026-07-22-quality-review.md`、
`2026-07-22-a2-gemm-autotune-investigation.md`、`2026-07-22-a6-attention-split-k-investigation.md`、
`2026-07-23-a2-gemm-baseline-and-cudnn-findings.md`、`2026-07-23-a3-nsys-conclusion.md`、
`2026-07-23-cudagraph-determinism.md`、`2026-07-27-l2-server-integration-gap.md`、
`2026-07-27-p1-http-e2e-and-thinking-strip-bug.md`、`b12x_investigation.md`、
`compile_integration.md`、`prefix-cache-implementation-log.md`

## 9. 仓库卫生 🟢

| 文件 | 内容 |
|---|---|
| `2026-08-01-t0-7-branch-worktree-survey.md` | T0-7 分支/worktree 调研清单（只调研未执行删除）——16 个分支已合并可安全删，16 个有独有提交需人工确认 |
| `2026-08-02-laguna-docs-inherited-qwen36-numbers.md` | **文档数字污染**：`docs/roadmap.md:27`/`docs/model-support.md:49` 的 MMLU-Pro 84.54% 与 `README.md:79` 的容量表都是 Qwen3.6 时代数字被误标成 Laguna 当前数字（前者与 Laguna 自己的显存审计矛盾）；未修复，只记录，供 roadmap/model-support/README owner 处理 |

## 10. Track B0 事实基线（Qwen3.6 重建）🟢

零 GPU、只读 safetensors header/JSON config 的事实核实，是
`qwen36-rebuild-spec.md` B0-2/B0-6/B0-7 的证据来源。

| 文件 | 内容 |
|---|---|
| `2026-08-02-qwen36-b0-fact-baseline.md` | **B0-2**：`nvidia/Qwen3.6-27B-NVFP4` 是混合精度（GDN/self_attn 投影 FP8，稠密 MLP+lm_head 才是 NVFP4 weight-only/W4A16），逐张量命名与 Laguna compressed-tensors 对照表；**最大发现**——checkpoint 声明 `kv_cache_quant_algo=FP8` 但零 k_scale/v_scale 张量（Laguna 对照组确认这不是量化格式的通例，是这份 checkpoint 特有的缺口）；333 个 vision 张量前缀 `model.visual.` 精确复现。**B0-6**：mrope-interleaved 纯文本退化为标准 1D RoPE，附 `transformers==5.8.0` 的 `modeling_qwen3_5.py` 源码行号证据（`Qwen3_5TextModel.forward`/`compute_3d_position_ids` 的 position_ids 恒等链路）。**B0-7**：GDN 递归状态每槽固定 ~75–150 MiB、与上下文长度无关；权重实测仅 18.8 GiB（纯文本，比 Laguna 67 GiB 小很多），据此给出 context×并发可行域算术推导（非 GPU 实测）。**同日第二轮追加**（§4-8）：逐个读完 `sparkinfer/gemm/` 9 个 op 后确认 `moe._shared.kernels.w4a16` 已原生支持 modelopt NVFP4 weight-only 语义（`prepare_w4a16_modelopt_nvfp4_weights`，明确是为 GLM 服务的 A4-prefill/A16-decode 场景写的），`num_experts=1` 在整条代码路径上没有下限限制，可走公开 `moe.fused_moe(quant_mode="w4a16")` API 退化成稠密 GEMM，不需要 SparkInfer team 写新代码（[待验证 GPU]）；GDN 状态 dtype 证据链纠偏（本机真装了 `fla`，参考实现走真 kernel 而非 torch 兜底，真 kernel 内部用 FP32 算，但 HF 通用 Cache 类落盘时按 conv_states 先锁定的 BF16 把结果降精度存回——净效果"落盘 BF16"结论不变，但机制是"单步 FP32 计算+跨步 BF16 舍入"，B1 若要 bit-exact 需要复刻这个舍入动作）；读了 vLLM/SGLang 源码确认两者对"声明 FP8 KV 但无 scale 张量"都是默认 1.0+告警（vLLM 正在弃用运行时校准选项，官方声明"以后就是有则读、没有就 1.0"） |
| `2026-08-01-b6-mtp-gdn-verification.md` | **B0-8**/`investigation-queue.md` B-6：六个本地 Qwen3.6-27B checkpoint 变体的 `mtp.*` 张量清一色 `self_attn`+`mlp`+norm，零 GDN 张量（`linear_attn.*`/`A_log`/`conv1d`/`dt_bias`），确认 MTP 头本身不含 GDN；但纠正"因此 B3 最难项消失"的推论——verify 阶段仍要把 MTP 候选整段跑主模型 64 层（含 48 层 GDN），候选被拒绝时主模型 GDN 状态已被不可逆更新，这是主模型侧问题、跟 MTP 头无关（独立佐证 vLLM `#47572` ReplaySSM RFC，与本仓库 D-3 同一问题） |
| `2026-08-02-trackB-b0-facts.md` | B0-2/B0-6/B0-7/B0-8 四条**独立复现**（不是抄上面两篇——每个数字/行号本轮重新跑出来），用于把 `docs/implementation-plan.md` §7.1 的四个复选框从未打勾状态收口打勾；额外核实了 `unsloth/Qwen3.6-27B-NVFP4` 作为 B0-8 的第二个 checkpoint 交叉验证 |

---

## 写新 note 的约定

1. 文件名 `YYYY-MM-DD-<主题>.md`，日期用当天真实日期。
2. 开头写清楚：**问题是什么、用什么配置测的、结论是什么、什么被排除了**。
3. 数字必须附复现命令。跨运行比较必须先 `bf diff`。
4. 结论被推翻时，**改旧文件**（加 `SUPERSEDED` 段落指向新结论），
   不要只写一篇新的——否则半年后没人知道哪个是对的。
5. 写完在本索引里加一行。
