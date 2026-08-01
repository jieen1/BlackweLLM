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

---

## 写新 note 的约定

1. 文件名 `YYYY-MM-DD-<主题>.md`，日期用当天真实日期。
2. 开头写清楚：**问题是什么、用什么配置测的、结论是什么、什么被排除了**。
3. 数字必须附复现命令。跨运行比较必须先 `bf diff`。
4. 结论被推翻时，**改旧文件**（加 `SUPERSEDED` 段落指向新结论），
   不要只写一篇新的——否则半年后没人知道哪个是对的。
5. 写完在本索引里加一行。
