# BlackweLLM 文档索引

> 最后整理：2026-08-01 · 基线 commit `ce21eb5`

## 先读哪一份

| 你想做什么 | 读这份 |
|---|---|
| 了解项目现在是什么、要去哪 | [`roadmap.md`](roadmap.md) |
| **知道下一个动作是什么、谁卡着谁** | [`implementation-plan.md`](implementation-plan.md) |
| **有哪些待排查 / 待拍板的事项** | [`investigation-queue.md`](investigation-queue.md) |
| 了解系统怎么搭的、准备改核心代码 | [`architecture.md`](architecture.md) |
| 接入一个新模型 / 想知道支持哪些模型 | [`model-support.md`](model-support.md) |
| **重建 Qwen3.6-27B 支持（Track B）：`oracle/qwen36_vllm/` 哪些能搬、验收基线数字、风险** | [`qwen36-rebuild-spec.md`](qwen36-rebuild-spec.md) |
| **排查任何问题、写任何诊断代码之前** | [`diagnostics-guide.md`](diagnostics-guide.md) |
| 部署和调参 | [`../README.md`](../README.md) + [`../server/README.md`](../server/README.md) |
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
| [`architecture.md`](architecture.md) | 当前架构、目标架构、五个关键抽象、迁移不变量 | 核心执行路径或分层变化 |
| [`model-support.md`](model-support.md) | 支持矩阵、各模型架构事实、接入新模型的六步流程、跨模型陷阱 | 新模型接入、支持状态变化 |
| [`qwen36-rebuild-spec.md`](qwen36-rebuild-spec.md) | Track B 重建规格：`oracle/qwen36_vllm/` 逐模块判定与新位置映射、Qwen3.6-vLLM 时代验收基线（吞吐/接受率/MMLU-Pro/HumanEval+/显存）、在 Track A 抽象上的重建设计、风险与待验证清单 | `oracle/` 判定变化、新实测基线产出、Track B 里程碑推进 |
| [`diagnostics-guide.md`](diagnostics-guide.md) | bfdiag 使用指南、三条黄金法则、温冷引擎边界 | bfdiag 能力变化 |
| [`../README.md`](../README.md) | 面向外部的项目介绍、快速开始、配置 | 面向用户的行为变化 |
| [`../AGENTS.md`](../AGENTS.md) | 给 agent 的仓库约定（结构、命令、诊断纪律、风格） | 目录结构或工程约定变化 |
| [`../server/README.md`](../server/README.md) | API 与指标的逐项参考 | 端点或指标变化 |
| [`../notes/README.md`](../notes/README.md) | 116 篇调查记录的分类索引与时效标记 | 新增 note |

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
