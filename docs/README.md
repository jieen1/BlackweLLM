# BlackweLLM 文档索引

> 最后整理：2026-09-01 · 增补 Qwen3.8 Flash-Next 当前启动与 OpenCode 运维手册

## 先读哪一份

| 你想做什么 | 读这份 |
|---|---|
| 了解项目现在是什么、要去哪 | [`roadmap.md`](roadmap.md) |
| **知道下一个动作是什么、谁卡着谁** | [`implementation-plan.md`](implementation-plan.md) |
| **把 DSV4 单卡 prefill 提到 >=1000 tok/s** | [`dsv4-prefill-1k-implementation-plan.md`](dsv4-prefill-1k-implementation-plan.md) |
| **优化 Qwen3.8 / SM120 的 4×256K 显存、decode 与 prefill** | [`qwen38-sm120-cuda133-fa4-optimization-plan.md`](qwen38-sm120-cuda133-fa4-optimization-plan.md) |
| **按最新 profile 实施 Qwen3.8 性能优化（MTP K、B8、qkv/norm fusion、prefill）** | [`qwen38-sm120-performance-roadmap.md`](qwen38-sm120-performance-roadmap.md) |
| **有哪些待排查 / 待拍板的事项** | [`investigation-queue.md`](investigation-queue.md) |
| **E2E 测试 / 兼容性 / 稳定性怎么分期、GPU 窗口怎么排** | [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) |
| 了解系统怎么搭的、准备改核心代码 | [`architecture.md`](architecture.md) |
| 接入一个新模型 / 想知道支持哪些模型 | [`model-support.md`](model-support.md) |
| **重建 Qwen3.6-27B 支持（Track B）的历史规格与基线**（重建主体已落地，见 `../notes/2026-08-05-qwen36-quality-rerun.md`） | [`qwen36-rebuild-spec.md`](qwen36-rebuild-spec.md) |
| **排查任何问题、写任何诊断代码之前** | [`diagnostics-guide.md`](diagnostics-guide.md) |
| **启动当前 Qwen3.8 Flash-Next 服务 / 接入 OpenCode** | [`qwen38-flash-next-ops.md`](qwen38-flash-next-ops.md) |
| 部署和调参（通用 API / 历史 backend） | [`../README.md`](../README.md) + [`../server/README.md`](../server/README.md) |
| 查某个指标的定义 | [`../server/README.md`](../server/README.md#metrics) |
| 翻某个历史决策的来龙去脉 | [`archive/README.md`](archive/README.md) |
| 找某次调查的原始数据 | [`../notes/README.md`](../notes/README.md) |

## 文档清单

### 活文档（反映当前状态，有变更就要更新）

| 文件 | 内容 | 更新触发条件 |
|---|---|---|
| [`roadmap.md`](roadmap.md) | 定位、现状盘点、轨道与里程碑、风险、待拍板事项 | 里程碑推进、优先级变化、拍板落地 |
| [`implementation-plan.md`](implementation-plan.md) | roadmap 的执行视图：按优先级排序的实施清单、状态核实、阻塞依赖速查 | 条目完成、拍板落地、发现新阻塞 |
| [`investigation-queue.md`](investigation-queue.md) | 待排查 / 待拍板队列：来自上游代码阅读与生态扫描的输入，不在原路线图里 | 新增外部输入、自查完成、条目并入 implementation-plan |
| [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) | roadmap Track C/Track E 的执行细节展开：E2E 三层金字塔、兼容性/稳定性分期、"反复审查"节奏机制、GPU 窗口调度 | Track C/E 阶段完成、新增门禁、里程碑节奏调整 |
| [`architecture.md`](architecture.md) | 当前架构、目标架构、五个关键抽象、迁移不变量 | 核心执行路径或分层变化 |
| [`model-support.md`](model-support.md) | 支持矩阵、各模型架构事实、接入新模型的六步流程、跨模型陷阱 | 新模型接入、支持状态变化 |
| [`qwen36-rebuild-spec.md`](qwen36-rebuild-spec.md) | Track B 重建规格：`oracle/qwen36_vllm/` 逐模块判定与新位置映射、Qwen3.6-vLLM 时代验收基线（吞吐/接受率/MMLU-Pro/HumanEval+/显存）、在 Track A 抽象上的重建设计、风险与待验证清单 | `oracle/` 判定变化、新实测基线产出、Track B 里程碑推进 |
| [`2026-08-04-qwen36-w4a16-current-status.md`](2026-08-04-qwen36-w4a16-current-status.md) | Qwen3.6 当前 W4A16 热点、Nsight 证据与下一道性能门禁 | 新 profiling 结论或路径判定变化 |
| [`dsv4-prefill-1k-implementation-plan.md`](dsv4-prefill-1k-implementation-plan.md) | DSV4 单卡 prefill >=1000 tok/s：K32 grouped IQ2 complete MoE + bounded superchunk；包含 all-in operator、prefix、workspace、质量和服务 kill gates | 任一阶段完成、性能预算变化、目标约束变化 |
| [`qwen38-sm120-cuda133-fa4-optimization-plan.md`](qwen38-sm120-cuda133-fa4-optimization-plan.md) | Qwen3.8-27B-NVFP4 的 SM100/SM120 硬件边界、FA4→b12x 技术迁移矩阵、4×256K 显存账及 runtime/decode/prefill 实施门禁 | Qwen3.8 profile、内存基线、FA4/CUDA 版本或实施阶段变化 |
| [`qwen38-sm120-performance-roadmap.md`](qwen38-sm120-performance-roadmap.md) | Qwen3.8-27B-NVFP4 最新 profile 驱动的性能路线：MTP K5/7、B8 逻辑并发、qkv/norm fusion、自适应 prefill、收益与 kill gates | K/B sweep、融合实验、VMM/B8 容量或性能基线变化 |
| [`qwen38-flash-next-ops.md`](qwen38-flash-next-ops.md) | 当前 Qwen3.8 Flash-Next 的 SM120 启动命令、显存/并发边界、健康检查、OpenCode/Windows 接入与故障排查 | 当前服务 profile、端口、checkpoint 或客户端接入方式变化 |
| [`dsv4-prefill-2k-implementation-plan.md`](dsv4-prefill-2k-implementation-plan.md) | **SUPERSEDED**：DSV4 单卡 2K 路线的失败合同和历史证据 | 仅在历史证据被纠正时更新 |
| [`diagnostics-guide.md`](diagnostics-guide.md) | bfdiag 使用指南、三条黄金法则、温冷引擎边界 | bfdiag 能力变化 |
| [`../README.md`](../README.md) | 面向外部的项目介绍、快速开始、配置 | 面向用户的行为变化 |
| [`../AGENTS.md`](../AGENTS.md) | 给 agent 的仓库约定（结构、命令、诊断纪律、风格） | 目录结构或工程约定变化 |
| [`../server/README.md`](../server/README.md) | API 与指标的逐项参考 | 端点或指标变化 |
| [`../notes/README.md`](../notes/README.md) | 223 篇调查记录的分类索引与时效标记 | 新增 note |

### 归档（不反映当前状态，只作历史）

见 [`archive/README.md`](archive/README.md)。

## 文档纪律

1. **数字必须标日期和来源。** 性能/质量数字写明测量日期、硬件、配置、复现命令。
   没有来源的数字不写。
2. **假设标 `[待验证]`。** 没在本机实测过的，一律显式标注，不混进结论。
3. **过时就归档，不要就地烂。** 一份文档的前提不成立了，移进 `archive/`
   并在 `archive/README.md` 里写明原因和替代品——而不是留在原地误导下一个人。
4. **`notes/` 不搬家。** 代码注释按路径引用它们，移动会打断引用；
   用 `notes/README.md` 的状态标记代替物理归档。
5. **结论被推翻时改旧文件**（加 `SUPERSEDED` 段落指向新结论），
   而不是只写一篇新的。
