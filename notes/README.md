# notes/ 索引

> 编制日期：2026-08-15 · 共 234 篇（`git ls-files notes | wc -l`，包含本次新增）

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
| `2026-08-07-deepseek-v4-flash-iq2xs-gguf-implementation-plan.md` | **DeepSeek-V4-Flash-0731（GGUF IQ2_XS）接入实施方案**——事实基线、D1–D10 决策、六阶段计划、oracle 链与风险登记；参考材料在 `dsv4flash-ref/` |
| `2026-08-09-dsv4-cudagraph-decode-driver.md` | **DSV4 CUDA-Graph decode 驱动进展**——`torch.where` 修掉 compressor 迁移 `0 * -inf = NaN`，ratio-4 / ratio-128 回归已加；compressed pack 写址修正、indexer 自有 compressor 推进、decode graph driver/backend/engine 接入完成；grouped Q8 prefill 和 MoE prefill 真实模型阻断修复；真实权重 3 workload × 12-step gate 通过（worst cosine 0.99999988、token 12/12、eager 449.1ms vs graph 272.1ms，历史 136ms 口径不可直接比较）；P1 仍待统一诊断基线 |
| `2026-08-10-fa4-deep-dive-report.md` | **FA4 穷尽式深度调研**（纯调研，未写码）——逐行拆解 16-warp/TMEM、QK-PV ping-pong、软 exp2、条件 rescale、LPT、2-CTA/topk-gather-MLA。**2026-08-15 硬件更正**：SM120 物理缺失 `tcgen05`/TMEM/2-CTA joint UMMA，但支持普通 cluster/DSMEM；TMA multicast 功能存在而非优化 target，不能笼统写成“不支持”。beta26 与 Qwen3.8/b12x 的完整迁移矩阵见 `docs/qwen38-sm120-cuda133-fa4-optimization-plan.md`。 |
| `2026-08-10-ds4-cuda-deep-dive.md` | **antirez/ds4 CUDA 内核完整调研**——DSV4 M=1 decode 执行层细节。核心结论：**q8_0 170 GB/s 瓶颈根因是 34B 交错布局的 2B 对齐**，对齐 SoA 重打包 + warp-per-row dp4a 实测 +43~66%（attn_q_b 164→235、head 146→243 GB/s）。含对齐布局定义、preq 激活预量化、IQ2 gate/up 融合、HC dot 延后、DSpark verify 批量、CUDA Graph island 拆分的全部代码结构 |
| `2026-08-10-sglang-dsv4-deep-dive.md` | **SGLang DeepSeek V4 完整调研**——MLA/MoE/HC 结构、SM120 注意力 kernel 选择（FlashInfer sparse_mla_sm120 + Triton tiled 备选）、MTP/DSpark verify 批量机制、decode 调度。结论：SGLang 无魔法 M=1 kernel，破墙靠 DSpark/MTP（M=γ+1）+ FP8/NVFP4 减字节 |
| `2026-08-10-laguna-optimization-path.md` | **Laguna 性能优化完整路径调研**——CG decode 元数据烤进图、融合元数据 kernel、向量化纪律（5.6×）、fused_kv_scatter/fused_rms_norm、MoE 单 kernel 38μs/层、带宽账方法论（必须冷缓存）、DFlash M=16 摊销原理。DSV4 已抄 argmax 烤进图，缺 kernel 融合 + 布局重排 |
| `2026-08-10-m1-decode-deep-dive.md` | **M=1 decode 深度调研结论总览**——四路交叉印证：q8_0 对齐是根因（非 latency）、ds4 实测数字、行动清单按 ROI |
| `2026-08-10-dsv4-q8-warprow-gate-regression.md` | **DSV4 Q8_0 warp-per-row 优化未过门禁的完整说明**——单 kernel 正确（与 tl.dot maxdiff 0）但门禁 3-workload 漂移（0.99903 vs 基线 0.99999988）。含现象、已确证事实、尝试方案、未解疑点（soa_planes 懒构建与 graph pool 交互、逐层定位建议）。已回滚，门禁 PASS。**接手者先读此文档** |
| `2026-08-10-dsv4-prefill-moe-kernel-deep-dive.md` | **DSV4 prefill MoE kernel 性能深挖**——dp4a 已落地并将 76 提到 129 tok/s；2026-08-11 复测 64-row 稳态 131.8–133.3 tok/s。MoE 占 M32 chunk 83%，route-M1 IQ2 解码已接近指令流上限；warp-row 只再快 1.2×。达到 1K 必须改为 bounded superchunk + expert-grouped Tensor-Core 路线。含长上下文状态和显存边界。 |
| `2026-08-11-dsv4-phase2-int8-mma.md` | **Phase 2：IQ2 INT8 Tensor-Core grouped MoE**——exact `mma.sync.m16n8k16` 已落地并达到 cos 1.0；真实 1024-token 形状 gate+up 14.8 ms、grouped routed pipeline 32 ms（router/shared expert 未包含），未过旧 2K 的 router+routed+shared `<=6.5 ms/layer` kill gate。exact 路径保留为 oracle；K32 在 1K 新预算下重新成为主线，当前合同见 `../docs/dsv4-prefill-1k-implementation-plan.md`。 |
| `2026-08-12-dsv4-prefill-row-w8-replan-evidence.md` | **DSV4 单卡 2K row-W8 路线最终失败证据**——审计 `main@882d209`，记录 128 MiB L2、row-W8/A8 数值预筛、global W8 流量账，以及 C0 最终 gate+up 28.2 ms、W8 resident 6.4 GB、consumer L2 hit 23% 的三重失败。新 1K 合同只保留严格条件下的 CTA-local 研究门，不复活 global/circular W8。 |
| `2026-08-12-dsv4-4x256k-capacity-plan.md` | **DSV4 单卡 4×256K 显存容量实施规划**——严格核对当前双份主 compressor 历史、逐层 RoPE 和第五 CG 槽；主路线保留 packed pages、删除 5.40625 GiB BF16 主缓存镜像并把 RoPE 收敛为两份，目标已知持久占用 `86.52000322565436 GiB`；定义四槽、图池、prefix 和 fresh-process 实机验收门槛。 |
| `2026-08-14-dsv4-dynamic-moe-nan-bug.md` | **DSV4 routes>64 NaN 已修复，1024-token cold mean 达 509.2 tok/s**——device tile-loop 关闭未写 tail；position sync 647→1、Q8 直接读 SoA、indexer launches 40,320→630、cooperative gather 87.1→0.584 ms。20-run median 541.3 / trimmed 519.0，最好 563.9 接近 llama.cpp 567.8，但 mean 仍差 10.3% 且尾部抖动未解决。 |

## 2. 已定案的根因分析 🟢

- [2026-08-15 Qwen3.8 dynamic MTP page-table 代际 bug](2026-08-15-qwen38-dynamic-mtp-page-table-regression.md) —— 🟢 **fresh-process 128K 接受率回退已修复**：dynamic arena remap 物理 bundle 后，MTP draft/sync CUDA Graph 仍按 slot id 缓存 capture 阶段旧 page table；改用 `(slot,page_table_version)` 后首条 128K 从 `[7,31,21,31]` / 2.844 committed-per-round / 74.95 tok/s 恢复到 `[0,0,0,64]` / 4.000 / 101.58 tok/s，prefix-on cold/warm 也均满接受，输出 SHA 不变。另修 128K chained hash O(n²)：5.352s→10.8ms，persistent-hit TTFT 13.97s→~52ms。
- [2026-08-05 Qwen3.6-27B 质量套件重跑（MTP+CG 路径）](2026-08-05-qwen36-quality-rerun.md) —— 🟢 **质量无回退**：MMLU-Pro 414 精确复现 **84.54%**（与历史同 question_ids）；tool/agent/longctx 均 1.000；HumanEval 768 在 ±3.9pp SE 内。参数与历史一致（MTP K=3、block_size 16、FP8 KV、GPU util 0.92）；并行分片 + 断点续跑，`bash scripts/run_qwen36_quality.sh all` 可复现
- [2026-08-05 Qwen3.6 服务端性能网格（严格上下文 × 并发，MTP+CG+prefix cache）](2026-08-05-server-perf-grid-mtp-cg-prefix.md) —— 🟢 **15/15 cell 全完成、WARM 全命中**：5 档严格上下文（4K–250K，block 对齐）× 3 并发，参数与历史一致；250K 冷→热 TTFT ~316 s → ~0.48 s（655.6×），MTP-on 解码（4K c=1/2/3 = 67.9/64.7/65.8 tok/s）不再有修复前 7.80 tok/s 的回退。**过程中修掉两个叠加的 persistent cache bug**（per-slot chunked prefill 缺 COW 写穿 scratch、live-slot 已提交别名无法逐出致 250K store 静默失败），回归测试已加；`benchmarks/server_perf_grid.py` 逐 cell 续跑，结果 JSON 与失败对照版一并入库
- [2026-08-05 persistent prefix 完整命中路径修复 + Codex 接入](2026-08-05-persistent-prefix-full-hit-fix-and-codex-integration.md) —— 🟢 **完整命中必错的根因**：scratch restore 把 live GDN 列一起清零；次 bug 是 slot-local checkpoint 覆盖 one-to-one hash 索引导致隔次必 miss。修复 + 回归测试 + Codex CLI（Responses 协议）接入方式；流式生命周期事件缺顶层 `type` 导致 CLI 反复重连，已修并重跑通过
- [2026-08-05 Claude Code CLI 接入本地 runtime](2026-08-05-claude-code-via-local-runtime.md) —— 🟢 项目 `.claude/settings.json` 直连 `/v1/messages`（Anthropic 协议，无代理）+ workspace trust；Claude Code 多轮工具循环端到端任务退出码 0；带工具流式冒烟 3/3
- [2026-08-05 Laguna 模型 Codex + Claude Code 端到端测试](2026-08-05-laguna-codex-cc-e2e.md) —— 🟢 与 Qwen3.6 对等：Laguna 生产配置（DFlash K=15、3×256K、三图全 captured、prefix 命中 17/4）下 Codex CLI（Responses）与 Claude Code CLI（Anthropic）各完成真实 agentic 任务，退出码 0，hello 产物逐字节精确；Codex 总结 31 行略超 30 行要求，如实记录
- [2026-08-05 128K 解码回退主机侧剖析](2026-08-05-128k-host-side-profile.md) —— 🟢 每轮 ~127 ms 中 GPU 内核仅 ~52.6 ms；定位 accept 等待/checkpoint 克隆/ragged 同步/draft 续跑的多次 D2H；落地 array('I') 前缀哈希与 device-direct draft seed（轮中位 83.5→75.6 ms）
- [2026-08-06 128K/c4 历史性能追平剖析](2026-08-06-128k-c4-parity-profiling.md) —— 🟢 **追平并反超历史 headline**：精确重复前缀 warm（README 表口径）128K/c4 新进程样本 **238/243/227、239/223/248、237/235/232/222/239**（中位 234.8，4–5/波 ≥ 222.44）；64K 291/262 > 236.69；+10240 后缀历史协议形态稳态 197–201（~89–90%，含后缀 TTFT 与内容差异）。修复链：滚动 checkpoint 哈希（−1.1 ms/轮）→ verify 输入零分配 + lm_head 入图（verify_replay 10.8→1.5 ms）→ GDN checkpoint 单次拷贝 → sparkinfer M32 raw-FP8 verifier + replay worklist 跳过。瓶颈已移到 GPU verify ~50 ms/轮；跑间方差 ±5–10% 为主机 CPU 争用（chrome/celery），非 GPU
- [2026-08-03 交织 prefill + 全前向 warmup](2026-08-03-interleaved-prefill-and-warmup.md) —— 🟢 **两项实测收益,都照搬 `oracle/` 已跑通的实现**:首请求 TTFT **4.67s → 0.538s**(warmup 只暖 attention,MLP/GDN 全冷);长 prefill 对并发请求的停顿 **24.9s → 1.2s 且 ITL 零退化**(chunk 2048)。含一处自我修正:块大小在**交织语义下**是关键旋钮,不能拿"Phase A 只值 −10.7%"否掉
- [2026-08-03 ~~FP8 KV 打破投机 token 一致性~~（已撤回）](2026-08-03-fp8kv-breaks-speculative-token-identity.md) —— 🔴 **归因错误，结论已推翻**：现象是真的，原因不是 FP8 KV，而是 MTP 的 verify 跑了 sparkinfer 的 `mode="extend"`。见 [verify 模式](2026-08-03-mtp-verify-mode.md)。保留作推理链记录，勿再引用
- [2026-08-03 MTP verify 走错 sparkinfer 模式](2026-08-03-mtp-verify-mode.md) —— 🟢 **根因 + 修复 + 实测**：verify 被当成 extend 规划，导致 FP8 KV 下投机 token 不一致；改走 `mode="verify"` 后一致性恢复、接受率 1.54/2.00、每轮 303.8→154.7 ms、verify 图首次捕获成功。但服务器端仍 0.79x/0.83x —— 实测**输在发射空隙不是内核量**（68% vs plain decode 的 89% busy），下一步 M-1b：anchor 折进 verify（历史 qo_len=k+1）
- [2026-08-03 MTP 一轮实测分解](2026-08-03-mtp-round-profile.md) —— 🟢 **GPU 只有 31% 忙,瓶颈是主机侧拷贝不是接受率**:一轮 276.8ms/leaf kernel 87.0ms,`aten::copy_` 113.3ms(41%);`verify_forward` 占 70% 且没进图。⚠️ 反直觉:**verify 4 个 token 只花单次 decode 的 1.2×,投机前提在 GPU 侧成立**
- ⭐ [2026-08-03 历史实现完整测绘](2026-08-03-historical-implementation-survey.md) —— 🟢 **要查历史做过什么、量到多少，先读这份,别再回老树一次问一个问题**。含 12 条按收益排序的"今天缺什么"、每个实测数字的完整配置与可比性标注。三条纠偏:① 历史终点是 `a9cb932`(07-30)不是 `8f5c195`,中间 330 个提交;② **代码没丢——就在今天仓库的 `oracle/qwen36_vllm/`(8047 行)和 `docs/archive/2026-07-20-PROGRESS.md`**;③ 跨步交织 chunked prefill **建过**(`a8bd167`),今天是 stub

- [2026-08-03 解码 kernel profiling：CG 下已 kernel-bound](2026-08-03-decode-kernel-profile.md) —— 🟢 GPU busy 31.01ms / CG 墙钟 34.67ms = **89%**；eager 只有 21% 忙（CPU 侧 paged 元数据 ~34ms/step）。按调用次数精确归属：NVFP4 融合 MLP **56 次/35%**，FP8 层反量化后的 BF16 GEMM **233 次/45%**（233 = 预期数，一个不差），**GDN 递归仅 0.6%**。含一处自我纠正（首版对 `key_averages()` 求和，重复计数）
- [2026-08-03 FP8 KV cache：过了 B1-R，省 ~12.3 GiB](2026-08-03-fp8-kv-cache.md) —— 🟢 **今天唯一的正面结论**。标准 checkpoint 一直发着 16+16 个 `k_scale`/`v_scale` 而无人消费(反向 loader 检查首次真实运行抓到)；历史上 FP8 KV 就是本模型的默认值。判据每条 2–8× 余量、**无捕获窗口溢出**——与 W4A4/W8A8 形成对照:**量化 KV 没事，量化激活不行**。KV 8192→4096 MiB/槽。途中修掉两个静默出错(Q 被一起量化成 FP8、`index_copy_` 不支持 fp8)。当前默认关，建议翻开
- [2026-08-03 128K 解码 profiling](2026-08-03-128k-decode-profile.md) —— 🟢 **实测,非推断**:128K 下 **attention 占 58.5%**(短上下文下几乎不可见——两个区间结论不可互推);🔴 **FP8 KV 让 attention 慢 19%**(44.97→53.64ms),读的字节减半而 kernel 更慢,**推翻了我自己'它是长上下文最大提速杠杆'的预测**;另记单次 forward 上限 ≈61,681 token(int32 溢出)
- [2026-08-03 与历史性能的差距](2026-08-03-performance-gap-vs-historical.md) —— 🟢 **接受长度确定退化 2.75×**(历史 3.3 → 今天 1.20，已定位到 (token,hidden) 配对 bug)；**原始 decode 步按有效带宽慢约 1.6×**(历史 564 GB/s vs 今天 343，且历史还多扛 128K attention)。含两处自我纠正
- [2026-08-03 阶段四盘点：解码 kernel 杠杆已用尽](2026-08-03-stage4-kernel-levers-exhausted.md) —— 🔴 **总结已被上一条推翻**（"已 kernel-bound"≠"kernel 已最优"）；四条否定本身仍有效。原摘要：四条路逐一关闭：W4A4 ✗、W8A8 ✗（两者都是降激活精度，这个模型对此敏感）、`bf16_gemv` 只覆盖 34/237 投影约 **0.8% kernel 时间**（先量后做，几分钟否掉）、sparkinfer 里除它之外**全是量化 GEMM**。**那 24.8% 的 SM80 kernel 没有更好替代品。**
- [2026-08-03 FP8 W8A8 预演：误差下界都过不了 B1-R](2026-08-03-fp8-w8a8-preflight-negative.md) —— 🟢 **负面定案，且省掉一整轮实现**：不写 kernel，只注入激活 FP8 往返（真实 W8A8 误差的下界），`instruction` 负载即溢出 top-1024 捕获窗口。⚠️ 那些看似过线的 bar 是**排除掉最差负载后**算的（已修该假绿）。含一条对我自己的纠正：引用量化数字前先确认它测自哪个 checkpoint（今日第二次同型错误）
- [2026-08-03 生产显存审计](2026-08-03-production-memory-audit.md) —— 🟢 标准模型、配置逐项写明：CG **72.39 GiB** vs eager 77.69。**反量化缓存只解决了一半**——FP8 侧 237 个张量的 BF16 缓存 19.99 GiB 与 FP8 原件 9.99 GiB 同时常驻，`forward` 只读前者
- [2026-08-03 W4A4 blockscaled 走不通](2026-08-03-w4a4-blockscaled-negative-result.md) —— 🟢 **负面定案**：kernel 契约完全匹配、checkpoint 真是 W4A4，但单层 cosine 0.988 对 W4A16 的 0.99999（差 30×），**B1-R 全线不过**，一个负载发散到溢出 top-1024 窗口。生产未动。含陷阱留档：两个 global scale 都**直接用不取倒数**，与 W4A16 约定相反，4 种组合实测 3 种直接爆
- [2026-08-04 Qwen3.6 W4A16 profile 与分相试验](2026-08-04-qwen36-w4a16-profile-and-split-plan.md) —— 🟢 当前 W1-S trace 的可复现 phase 分解、W4A4 明确排除、已提交的 barrier-clear 小修；已纠正 modelopt 分相未命中 Unsloth packed 路径的错误归因。Nsight 已定位真实 packed kernel 为 **54.27 KiB/CTA、单 block/SM、16.67% occupancy**。**occupancy 候选同日实测结案（负结果）**：放开 pin 后 3 blocks/SM、单层 kernel −18%，但全模型 verify 图相位 +6%，隔离收益被 barrier/tile_k 开销吃掉；下一步方向改为发射结构。
- [2026-08-03 MTP 接受率在标准 checkpoint 上重测](2026-08-03-mtp-acceptance-on-standard-checkpoint.md) —— 🟢 **checkpoint 假说被证伪**：prose 每轮平均接受 nvidia 1.21 vs 标准 1.20，几乎相同。⚠️ 但其基线同样是 eager，e2e 比值不能外推到生产
- [2026-08-03 CUDA Graph vs eager 解码吞吐 = 4.71×](2026-08-03-cudagraph-vs-eager-decode-throughput.md) —— 🟢 **在册每个吞吐数字都是 eager 的，比运行时实际能力低约 5 倍**。服务路径同 prompt 同参数只切一个开关：CG **28.848** vs eager **6.120 tok/s**，且 CG 还少用 5.30 GiB。捕获本身不贵（启动 +4s）。⚠️ **所有基于 ~6 tok/s 的优化判断都需重估**；另纠正一条我自己的误判（311s 冷启动是磁盘读 22 GiB，不是 JIT 前置）
- [2026-08-03 MTP 服务路径 GPU 验证：接受率涨了，但 CG 下吞吐反而 0.28×](2026-08-03-mtp-serving-gpu-verification.md) —— 🟢 **(token,hidden) 配对修复方向对但幅度有限**：prose 接受率 0.300→0.385、code 0.417→0.455，32 token 逐 token 相同。**决定性负面结论**：`Qwen36MTPEngine.round` 全程 eager，从不经过 `decode_batch_sampled` 那条被捕获的 CUDA Graph——MTP 打开后服务路径 tok/s **28.0→7.80，倒退到 0.28×**，比接受率本身更能决定 MTP 该不该上生产。C-LIVE 持平/略好（66/67 vs 64/67 基线）。附一条诚实记录但未下定论的发现：~2000 token 长生成下 MTP-on 与 MTP-off 逐 token 比对**确实分叉**，但溯源到两个各自独立、早已在案的 kernel 数值噪声源（CG-vs-eager 在 MTP 完全不参与时同一位置也分叉；extend-mode-verify-vs-decode-mode 是 B3 判据脚本自己文档里预先声明过的"不算 bug"类分歧），不是 accept/reject 实现本身的缺陷。重同步 A/B 本轮未测，预算耗尽。
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
| `2026-08-15-b12x-packed-f32x2-falsified.md` | 🔴 **P0-A2 packed f32x2 负面定案**：cutlass-dsl 4.7.0 把 `fma_packed_f32x2` 标量化——SASS 零 FFMA2/FMUL2/FADD2、净指令差仅 2 条、128K B1 warm 在 ±3-5% 噪声内（110.2 vs 108.8 tok/s），且 fused 舍入移动数值（plain 60 token 3 个平局翻转）。按规划 §7.2 淘汰条款回退；exact conditional rescale（bit-exact）保留，见 sparkinfer `8f74740` |
| `2026-08-15-qwen38-phase0-round-attribution.md` | 🟢 **Phase 0 轮级 + kernel-family 归因实测**（规划 §6.1 门禁）：K=3 生产配置 128K，轮级 phase 解释 ≥90% round wall（B4 GPU 合计 93.2%、verify 79.5%、host 仅 ~7%）；nsys kernel-family：attention 41% 最大、W8A8 24%、W4A4 14.8%、GDN ~2.6%。判别树落判——P1-A（b12x attention）方向证实，ragged sync 分桶（sync 5.8%<8–10%）与 GDN 融合**排除**。接受率 100%（832/832）；c4 warm 66.3→66.4 tok/s/req 对 P0 显存提交**中性**。K=2/K=1 sweep 留作补测 |
| `2026-08-15-strict-4x256k-startup-acceptance.md` | 🟢 **strict 4×256K fresh-process 启动验证实测通过**（规划 §4.9 门禁）：P0-M1/M2/M3/C 落地后，8201-bundle strict pool 完整启动序列峰值 **NVML 64.09 GiB / driver free 31.51 GiB**（优于规划 65–70 GiB 预估）；KV 合计 36.0 GiB 与代码精算 36.04 吻合；5-token 短请求只占 1 bundle；逐槽增长严格成比例（每 128 tokens 1 bundle）；decode graph 捕获阶段 Δallocated≈0，直接证明 P0-M2 step 2-3 释放抵消了 graph pool 增量 |
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
