# BlackweLLM 端到端测试 / 兼容性 / 稳定性 排期

> 编制日期：2026-08-02 · 基线 `main @ 52f9484` · 分支 `work/e2e-20260801`
>
> 本文档是 [`roadmap.md`](roadmap.md) Track C（稳定性）/ Track E（兼容性）与
> [`implementation-plan.md`](implementation-plan.md) §7.3/§7.5 的**执行细节展开**——回答
> "分几步、每步怎么证明它真的在测东西、GPU 窗口怎么排、多久审一次"。roadmap 仍是目标与理由
> 的权威，implementation-plan 仍是"下一个动作是什么"的权威；本文档不复述它们已经写清楚的
> 内容，只做排期、分层与"反复审查"机制设计。
>
> 本次任务范围：**只规划，不写实现代码**。所有 GPU 相关条目在此仅排期，不代跑；
> 标 **[待验证]** 的数字/假设没有本机实测记录。

---

## 0. 出发点：核实过的两处修正

写排期之前，先纠正两处会让排期算错优先级的过期结论（都是读代码/读文档核实的，不是猜测）：

1. **roadmap §1.3/Track E 把 `n>1` 和 usage token 统计列为"仍待核查"，其实已经查完**。
   `docs/api-layer-design.md` §5.3/§5.5（`fix/t0b-api`，2026-08-01）已经把两条都答了：
   - `n>1`：`server/app.py:447` 的 `_build_sampling_params` 显式检查并 400 拒绝，不是静默截断
     成 1 个——**已关闭，行为正确，不需要新工作**。
   - usage token 统计"基本正确"，只有两个已记录、低优先级的小缺口：(a) 响应里没有
     `usage.completion_tokens_details.reasoning_tokens` 细分字段；(b) `<usage>` 标签剥离在
     流式/非流式路径下语义不统一（训练数据污染的窄 artifact，实践中极少触发）。
   这两条不该再挂在"待核查"状态——本文档 §2 按这个更准的现状重新排期。

2. **N1（结构化输出）的真实阻塞点比"接进采样环"这个说法更具体，也更危险**。
   `runtime/structured_output.py` 模块头 docstring + `docs/api-layer-design.md` §7.1
   （2026-08-01）记录了三条不可达路径：admission 阶段裸 `argmax`（不经过任何
   `SamplingParams`）、CUDA Graph 贪心重放（贪心已经烤进已捕获的 graph，没有逐 token
   logits 张量可掩码）、eager 贪心分支直接绕过 `sample_from_logits`（本模块唯一能钩的缝）。
   本运行时默认 `temperature=0.0`，即"要保证的 JSON"这种最常见请求形态恰好总是走这三条
   不可达路径之一，包括第一个 token。**文档明确警告**：只接通那条唯一可达的窄缝
   （`temperature>0` 时第 2 个及以后的 token）比完全不接更危险——它会让默认场景看起来已经
   支持却仍然不受约束，是同一类静默失败换了个位置。**这不是一步"接上"就完的工作，需要先
   解决 admission 阶段没有掩码钩子 + CUDA Graph 贪心重放这两个架构冲突**，本文档 §2.3 按
   这个更准的现状重排。

3. **顺带发现，标 [待验证]，不代为断定**：`roadmap.md` §0 引用"Laguna-S-2.1 MMLU-Pro
   84.5%"作为"模型能力评测为一般"的证据。本仓库里唯一一次真实跑过的 MMLU-Pro 数字记录在
   `notes/2026-07-22-quality-baseline-and-official-scores.md` §7——**那次测的是已退役的
   Qwen3.6-27B + 上游 vLLM，不是 Laguna**，最终结果是"84.54%"，且呈现同样的"STEM 强
   （math/physics/chemistry/biology ≥92%）、人文弱（philosophy 65%/other 59%）"形状。
   仓库里另有两处不相关的"84.5%"是 DFlash 接受率数字（`notes/2026-07-29-*.md`）。三个数字
   同时存在，不足以断定 roadmap §0 的引用是误引，但足以怀疑——**Laguna 本身从未跑过 MMLU-Pro
   的可能性不能排除**。本文档把"针对 Laguna 真正跑一次 MMLU-Pro"列为 §3.3/C9 的第一步；
   跑完后无论数字如何，roadmap §0 的引用都应该换成有日期/commit/复现命令的新数字——这个
   替换动作留给该节的所有者（本次改动不碰 roadmap §0，只是提请注意，因为它恰好是本文档
   想要杜绝的那类"没人真的验证过、但被当结论用"的数字）。

---

## 1. E2E 三层金字塔

三层按**触发频率递减、单次成本递增**排列。每层必须能回答"它红了说明什么坏了"，
新增门禁必须给出"如何证明它会红"的方法（见 §4 的通用规则）。

| 层 | 内容 | 触发时机 | 单次成本 | 归属 | 状态 |
|---|---|---|---|---|---|
| **L0 · 活服务器冒烟** | C-LIVE 六项检查（冷启动 `/metrics`、忙时 `/metrics`、丢唤醒 200ms 窗口、双协议×流式×工具调用、thinking 首字符契约、`/v1/completions` 逐字返回） | 每次改 `server/` 或 `runtime/backends/`，push 前 | 秒级（对已热身的服务器发 HTTP） | **另一 agent 在落地**（`scripts/c_live_smoke.py` + `make smoke`，`work/trackA-20260801` 分支 `c83c41d`/`9db317c`/`ad5ef6b`，尚未合并 main） | 已有，本文档不重做，只在 §3 里把它当作 Track C 的 C-LIVE 锚点引用 |
| **L1 · SDK 一致性矩阵**（本文档新增） | 真实客户端 SDK/工具跑一遍真实会话：openai-python、anthropic-sdk-python、LiteLLM、Claude Code、Cline/Roo、OpenWebUI | 每次改 `server/formats/`；每个里程碑边界全量跑一遍 | 分钟级（一个热身服务器 + 每个客户端跑 3-5 个动作） | Track E / E3，见 §2.4 | **不存在，是本文档提出的结构性缺口填补** |
| **L2 · 质量与长稳回归** | MMLU-Pro（分层子集）+ evalplus HumanEval+/MBPP+ 对 Laguna 跑一次；24h soak | 每个里程碑边界（质量）；M3/M6 前（soak） | 小时到天级（独占 GPU） | Track C / C9（质量）、C5（soak），见 §3.3 | 质量 harness **已存在但从未指向 Laguna**（`benchmarks/official/mmlu_pro_eval.py` + `quality_regression.py`，`92f8b34`，2026-07-22，当时指向 Qwen3.6/vLLM）；soak 未做 |

**为什么 L1 是本文档要填的结构性缺口**：现有的 `tests/test_api_compat.py` /
`scripts/c_live_smoke.py` 都是手搓 `http.client`/`urllib` 直接读 HTTP 响应体——它们验证的是
"响应体符合我们自己认为的形状"，不是"一个真实 SDK 的严格解析器/pydantic 模型/流式状态机能不能
接受这个响应"。这两者不等价：真实 SDK 会做我们自己写的断言不会做的事——严格字段校验、
未知字段容忍度、SSE 事件顺序状态机、错误类型到异常类的映射、超时行为。本机已装
`openai==2.34.0`、`anthropic==0.99.0`、`httpx==0.28.1`（协调者核实，**[待验证]** 具体安装
在哪个 venv，需要落地时确认——这些应作为**测试期依赖**引入，不进 `pyproject.toml` 的
`serving`/`cuda` extras，协议实现本身不依赖任何厂商 SDK，这条边界必须保持）。

---

## 2. Track E（兼容性）分期

原则：每个阶段独立可交付、独立可验证，不出现"一次做完整个 Track E"。

### 2.1 已关闭条目（不再重复排查，只记录）

| 条目 | 状态 | 证据 |
|---|---|---|
| N2 `stop`/`stop_sequences` | ✅ 已接通 | `server/formats/stop.py` + `docs/api-layer-design.md` §7.2；覆盖贪心 MTP 提交批量路径 + 自回归采样路径 + admission anchor token；**已知且可接受的近似**：CUDA Graph 贪心重放路径不需要单独覆盖（路由规则决定，配了 `stop` 的 slot 本来就不走 CG 重放）；批内歧义文本整批扣住只影响延迟，从不泄漏 |
| N3 `seed` | ✅ 已接通 | `runtime/sampling.py::PersistentSeed`，按对象身份在同请求多轮解码间持续前进同一个 generator，见 `docs/api-layer-design.md` §7.3 |
| `n>1` | ✅ 已关闭 | `server/app.py:447` 显式 400 拒绝，`docs/api-layer-design.md` §5.3 |
| N1 危险性 | ✅ 已消除 | `server/app.py::_reject_unsupported_response_format` 三处显式 400，不再静默失败；功能缺口仍在，见 §2.3 |

### 2.2 E2 · 采样与投机共存（消灭 S8，"最明显的功能缺口"）

现状：`temperature>0` 时 DFlash 直接退化为无投机自回归解码（`_greedy_accept_reject`，
`runtime/backends/laguna_dflash.py:76`，只处理贪心接受/拒绝）。要接上需要拒绝采样
（rejection sampling）或 typical acceptance 这类"投机验证步骤本身允许非贪心采样"的机制
（标准做法，Leviathan et al. 投机解码论文族 + vLLM/SGLang 已有实现可参考）。

- **E2-a（M2，零 GPU）**：拒绝采样/typical-acceptance 算法本身的正确性可以在 CPU 上用
  合成分布验证（给定草稿分布 q 和目标分布 p，验证接受概率 `min(1, p(x)/q(x))` 与拒绝后
  重采样残差分布的数学性质），不需要真实模型。产出：`runtime/mtp_accept.py`（现有模块，
  当前只有贪心版本）旁边的非贪心版本设计 + 纯 Python/CPU-tensor 单测。
- **E2-b（M2→M3，需 GPU，蹭 Track A 剩余窗口或 Track F 的 F1 窗口——见 §5）**：接进
  `laguna_dflash.py` 的验证步骤，替换 `temperature>0` 时"直接退化成非投机"的分支。
  **完成判据**：(1) 接受率进 bfdiag 基线，与贪心路径的接受率（96.3–100%）分别记录，
  不要求相同；(2) 采样 token 的分布统计学上匹配非投机路径的采样分布（同 prompt、同
  temperature，跑 N 次，KS 检验或类似方法，而不是肉眼看"看起来差不多"）。
- **如何验证会红**：这条目前**天生是红的**——今天的代码在 `temperature>0` 时压根不会
  尝试投机（`is_greedy` 分支直接短路），所以"验证 E2-b 完成"的测试从写下来那一刻就会失败，
  不需要历史回放。

### 2.3 E-N1 · 结构化输出真正生效（架构性阻塞，需先拍板再实现）

见 §0-2 的现状修正。这不是一步走的工作：

- **E-N1-a（M2，零 GPU，决策备忘）**：写清楚两个选项交给需要拍板的人（不在本文档自行
  决定，因为这是产品语义判断）：
  - (a) 等到 admission 阶段掩码钩子 + CUDA Graph 贪心重放的架构冲突都解决后一次性接通
    （`docs/api-layer-design.md` §7.1 的"可推翻的条件"）；
  - (b) 先做一个**显式受限**的中间态——只接受 `temperature>0` 的结构化输出请求，
    `temperature=0`（默认）时对 `json_object`/`json_schema` 继续 400 拒绝（而不是静默
    只保护部分路径）。(b) 的关键是**显式**：客户端会立刻知道自己撞到了限制，而不是拿到
    一个看起来生效但没生效的响应。
  - **不选的选项**：文档已经明确警告过的"只接可达窄缝但不改拒绝逻辑"——那是原地不动
    换个位置藏起来，不是一个真的选项。
- **E-N1-b（(a) 或 (b) 拍板后，M3→M4，需 GPU，前提是 `runtime/backends/laguna.py` /
  `laguna_cuda_graph.py` 在 Track A 的第 5–8 步之后稳定下来——见 §5 的调度原则）**：
  实现选定的方案。
- **完成判据**：生成 N 个样本（N ≥ 20，覆盖至少 3 种 json_schema 复杂度），100% 通过
  JSON Schema 校验；同时验证方案 (b) 下 `temperature=0` 的拒绝行为没有被误改成静默通过。
- **如何验证会红**：给 `GrammarState` 喂一个刻意设计的、会被违反的 grammar（比如要求
  `enum: ["a","b"]` 但模型大概率会生成别的值)，在实现前跑这个校验会失败（因为完全没有
  约束）；实现后同一个校验必须变绿。这个"实现前失败"就是这条门禁最初的红色证明，不需要
  额外构造。

### 2.4 E3 · 客户端 SDK 一致性矩阵（本文档提出的新结构性资产）

现状盘点里协调者提出的判断（"没有一套真实客户端 SDK 的一致性矩阵"）核实为真：
`grep -rn "^import openai\|^import anthropic" tests/ scripts/` 零命中——所有现有 E2E/兼容性
资产（`test_api_compat.py`/`test_tool_calls.py`/`test_real_world.py`/`test_format_regression.py`/
`c_live_smoke.py`）都是手搓 `http.client`，没有一个用真实厂商 SDK 解析响应。

分四个子阶段，**便宜的、已经装好的先做**：

- **E3-a（M2）**：openai-python + anthropic-sdk-python。两者本机已装（协调者核实版本
  2.34.0/0.99.0，**[待验证]** 具体在哪个 venv），零新增安装成本。每个跑：
  非流式 chat、流式 chat、一次工具调用、一次故意触发的 400/422 错误（确认 SDK 把我们的
  错误体正确映射成它自己的异常类型，不是解析失败）。**首次真机运行蹭 Track A 第 5 步或
  C-LIVE 的 GPU 窗口**（见 §5）。产出：一份锁定的基线记录（bfdiag run record 或等价的
  带日期/commit 的 transcript），之后按 §4 的节奏刷新。
- **E3-b（M3）**：LiteLLM。选它排第二的理由——它是一个纯协议归一化代理，同时理解
  OpenAI 和 Anthropic 两套协议，能捕获"归一化层"这一类两个原生 SDK 各自都不会触发的 bug
  （比如它对 `tool_calls`/`content` 数组做的二次转换）。
- **E3-c（M4）**：Claude Code 本身（本项目在用的工具）指向 Anthropic 兼容端点跑一次真实
  编码会话——这是"吃自己的狗粮"，给出的是一个真实 agentic workload 的高信号样本，而不是
  单条请求。
- **E3-d（M5）**：Cline/Roo + OpenWebUI。排最后，因为它们是下游集成（IDE 插件 / Chat UI），
  已经建立在前三步验证过的协议行为之上，风险相对低，但需要更多环境搭建成本
  （长 system prompt、频繁工具循环、会话恢复这些是它们特有的流量形状）。

**每一步的完成判据**：该客户端的一次完整会话（至少：列模型、单轮对话、流式对话、一次
工具调用）无异常跑完，产出一份写进兼容性表的结果行（成功/部分成功/失败 + 具体现象）。
**如何验证会红**：对每个已修复的历史 bug（N2 `stop` 未实现前、错误体双重包裹进
`{"detail":...}` 前、thinking 泄漏 bug 修复前），可以在该 bug 的**父提交**上重跑同一个
SDK 客户端脚本，确认它在父提交上失败、在修复提交上通过——这是把 C-LIVE B-4 已经用过的
"父提交必须变红"验收标准，原样搬到 SDK 矩阵上，不是发明新方法。

**RK9 交叉引用**：E3 的每个客户端脚本第一次跑，天然就是"该客户端 SDK 打到这台服务器
的第一个请求"——如果撞上冷启动 JIT 编译停顿（30–100 秒，见 roadmap RK9），某些 SDK 的
默认请求超时可能比这个停顿更短，会报超时而不是报错误的 JSON。这是手搓测试（自己控制
超时参数）不会暴露、但真实客户端会暴露的一类 bug——E3 的完成判据应包含"第一次请求"和
"稳态请求"两种场景，不能只测稳态（见 §3.1 C7 的交叉引用）。

### 2.5 E4 与其余小缺口

- **E4 reasoning 暴露**：OpenAI 侧已接（`reasoning_content`，`c86858a`）。Anthropic 侧
  维持非标准 `reasoning_content_delta` 事件，理由见 roadmap §1.4（`f13fd4a` 生产事故，
  伪造签名导致 Claude Desktop 静默丢弃后续所有 content block）。**这不是待办，是一条
  需要长期守住的契约**——建议加一条**零成本、随时生效**的标准回归测试（如果
  `test_thinking_reasoning.py` 还没有，补一条）：断言 Anthropic 流式响应路径的任何代码
  路径都不产出规范形态的、未签名的 `thinking` content block。这条不需要里程碑节奏，
  写一次、常驻 CI。
- **usage token 两个小缺口**（见 §0-1）：
  - `usage.completion_tokens_details.reasoning_tokens` 细分字段——独立、小、可随时排期，
    不依赖任何其它条目，M3 前后顺手做。
  - `<usage>` 标签流式/非流式语义统一——需要先决定"统一到哪一种语义"（产品判断，不由
    本文档代为决定，`docs/api-layer-design.md` §6 已经把这个问题留白），决定后是小改动。

---

## 3. Track C（稳定性）分期

roadmap 已有 C0–C7 骨架。本节把它们排成可执行的阶段，并新增 C8/C9。

### 3.1 排期表

| 阶段 | 内容 | 里程碑 | 需要 GPU | 完成判据 |
|---|---|---|---|---|
| C0R | bfdiag code_ref 行号漂移审计残余（见 implementation-plan §7.3） | M1 | ❌ | 5 个高命中文件改为按符号名引用；两条无调用点的不变量接线或标记 |
| C1-backend | 故障面清单里 backend 层的部分：显存不足、CUDA Graph 捕获失败、kernel JIT 失败、并发抢占 | M1→M2 | ✅（蹭 Track A 第 5–8 步窗口，见 §5） | 每项都有检测点+指标+日志+用户可见错误+恢复动作，逐项写进 `docs/diagnostics-guide.md` 或等价文档 |
| C1-protocol | 故障面清单里协议层的部分：客户端断连、tokenizer 边界、非法采样参数、长请求超时 | M2→M3 | 🟡 小（可与 E3 的 harness 工作共享同一个热身服务器） | 同上，逐项覆盖测试 |
| C1-slot | 槽位卡死 | M2（Track A A3 第 7 步落地时，同一窗口） | ✅ | 覆盖测试 + 至少一次故障注入复现 |
| C2 | 分级降级（CG→eager、投机→非投机、前缀缓存命中→冷 prefill）三级各出指标 | 设计 M2 零 GPU，逐个触发点接线随各自 GPU 窗口 M2→M4 | 部分 | 三个降级事件都能在 `snapshot()`/`/debug/stats` 里查到（复用 D-3 能力查询形状） |
| C3 | 看门狗覆盖 + 故障注入 | M2→M3 | 🟡 主要靠 mock（在 Python 层注入异常模拟 CUDA 错误，不需要真实 OOM），小规模真机确认见缝插针 | 每个已知故障面至少一条注入测试 |
| C4 | bit-exact 回归门禁落地 | M2（`make gate-local`，随 Track A 第 5 步首次真机验证） | ✅ | `make gate-local` 跑通，PR 签核勾选项生效 |
| C5 | 24h soak | **M3 末、M6 末两个检查点**（见 §5 的独占窗口安排） | ✅ 独占一整天 | 显存碎片/host 内存/槽位分布/指标漂移在 24h 内不越界，产出一份 run record |
| C6 | 崩溃可诊断（进程级异常留 bfdiag run record） | 设计+纯异常场景验证 M2 零 GPU；真实 CUDA 崩溃场景确认见缝插针 | 部分 | 一次注入的未捕获异常产出完整 run record，不是裸 traceback |
| C7-1/2/3 | 冷启动/首次真实形状路径审计（RK9，implementation-plan §7.3 已有拆解） | C7-2 随 Track A 第 5 步捆绑（M2）；C7-1/C7-3 见缝插针 | ✅ | 见 implementation-plan §7.3 原文，不重复 |
| **C8**（新增） | 门禁可信度周期审计 | **M2、M4、M6**（每两个里程碑一次） | ❌ | 见 §3.2 |
| **C9**（新增） | 质量回归：MMLU-Pro（分层子集）+ HumanEval+/MBPP+ 对 Laguna | **M2 首次基线**；M3/M4 起加 Qwen3.6 覆盖；M6 发布前全量 | ✅（蹭窗口，见 §5） | 见 §3.3 |

### 3.2 C8 · 门禁可信度周期审计（新增，直接回应 N4/C0 的教训）

**动机**：用户点名的 `bfdiag/checkpoint` 回归测试——从不调用真实函数、一直是绿的、
守着一个早就不存在的 bug——不该是一次性发现后一次性修复的事故，应该是一个**常设的
自我审查动作**，否则同类问题会在别的模块里原样再发生一次（N4 本身就是"发现了一次"，
不代表以后不会再有第二次）。

**做法**：每次审计抽取 N 个（建议 N=12–15，约相当于 `bfdiag/` 一次抽样覆盖一轮）现有
测试/门禁，对每一个回答两个问题：

1. **这条测试在 git 历史上真的失败过吗？**（`git log` + CI 记录，能找到一次真实的红，
   说明它至少一次真的挡住了什么。）
2. **如果它从没红过，能不能构造一个输入让它红？** 如果试了却红不了（比如它测的函数已经
   被替换、mock 对象已经和真实类型漂移、断言路径实际不可达），标记为**可疑**——不是立刻
   删除或重写，是记入一份"门禁债务清单"，供下一轮排期时处理（同 N4 当年被发现的方式，
   但变成周期性而非偶然）。

抽样范围优先 `bfdiag/`（历史上唯一一次真实中过这个问题的地方），其次是新落地不久的门禁
（C-LIVE、SDK 矩阵、E2/E-N1 的新测试——越新越可能有"看起来测了但没测到"的缝）。

**如何验证这条机制本身有用**：第一轮审计（M2）应该**主动**去审计 N4 那次事件涉及的
`bfdiag/checkpoint` 模块群，确认修复后的状态经得起这两个问题的检验——如果连这个已知
案例都审不出问题，说明审计方法本身需要调整，而不是说明"这次没找到新问题"。

### 3.3 C9 · 质量回归（新增，MMLU-Pro / HumanEval+ 与长稳同级对待）

**为什么和 C5 放一起**：两者都是里程碑量级、独占 GPU、成本以小时到天计——不能像
C-LIVE 一样"每次改代码就跑"，只能按里程碑节奏。质量回归回答"输出还聪明吗"，
soak 回答"服务还稳吗"，两者合在一起构成北极星指标第 3 条（"输出可信"）与第 2 条
（"不会崩"）的周期性证据。

- **C9-a（M2，需 GPU，蹭窗口）**：针对现有基础设施做**存在性验证**——
  `benchmarks/official/mmlu_pro_eval.py --base-url <laguna-server>` 与
  `benchmarks/quality_regression.py` 直接指向 Laguna 跑一次小规模子集（比如 §0 提到的
  414 题分层子集的一个更小切片，先确认接口兼容，不追求完整规模）。**这一步的产出本身就是
  §0-3 那条 [待验证] 的答案**——如果这次是 Laguna 第一次真正跑 MMLU-Pro，产出的数字应该
  替换掉 roadmap §0 那个疑似误引的"84.5%"（替换动作留给该节所有者）。
- **C9-b（M2→M3）**：把 C9-a 的结果包装成 bfdiag run record（而不是散落的
  `evalplus_results/*.json`），纳入 `bf diff` 的可比性纪律——避免重复 2026-07-27 那次
  "两个不可比的数字被当成打平证据"的教训（`diagnostics-guide.md`）。锁定为回归基线。
- **C9-c（M3→M4，Track B B1/B2 落地后）**：同一套 harness 加一份 Qwen3.6-27B 覆盖——
  两个模型系列各自有一份质量回归基线，不是共享一份"平均"数字。
- **C9-d（M6，发布前）**：全量或接近全量规模跑一次（不再是分层子集），作为 Track H 发布
  门禁 H1 的一项输入。
- **完成判据**：每次运行产出一份带日期/commit/base-url/并发配置的 bfdiag run record；
  与上一次里程碑基线的差异用 `bf diff` 判定是否在容忍带内。
- **如何验证会红**：`benchmarks/quality_compare.py` 已经是"比较两份报告、超出容忍度退出
  非零"的现成实现（`92f8b34`）——验证它会红最直接的方法是拿一份历史上真实退化过的报告
  对比一份健康报告（AGENTS.md 列出的陷阱清单：block_size-128 接受率回归、
  fused_kv_scatter value-stride bug、FP8 舍入平局——这些都在 git 历史里有对应的坏提交，
  可以在坏提交上跑一次质量回归子集，确认分数确实掉，而不是假设它会掉）。

---

## 4. "反复审查"机制：节奏 × 成本 × 触发条件 × 责任人

用户原话是"反复"，不是"做一次"。下表把 §1–§3 的所有层级放进同一个节奏体系，
避免出现"写完门禁但没人知道该多久跑一次"的情况。

| 节奏 | 触发条件 | 内容 | 单次成本 | 责任人 |
|---|---|---|---|---|
| **每次提交**（最快） | 改动 `server/` 或 `runtime/backends/`，push 前 | L0 · C-LIVE 六项检查 | 秒级 | 该改动的作者（`make gate-local` + PR 签核勾选项，D3 已拍板的机制） |
| **每次协议改动** | 改动 `server/formats/**`（新字段、新端点、错误形状变化） | L1 · SDK 矩阵里已经建好的那部分（不要求全量六个客户端，先跑 E3-a 已装好的两个） | 分钟级 | 该改动的作者 |
| **每个里程碑边界**（M1→M6 各一次） | 里程碑收口 | L1 全量六客户端矩阵 + L2 · C9 质量回归子集 + C1 故障面清单刷新 + C8 门禁可信度审计 | 小时级，需要一个专门预留的 GPU 时段（见 §5） | 该里程碑的收口责任人（人或指定的"里程碑收口" agent 角色）——**显式指定，不能是"大家都以为别人会跑"** |
| **发布/重大切换前**（Track A 第 5 步 Registry 切换、B2 服务化合并、`0.2.0` 发布） | 一次性、日程化，不是机会性 | C5 24h soak + C9 全量质量回归 + 全量 SDK 矩阵 + C7 冷启动审计复跑 | 天级，独占 GPU | Track H 发布协调者（对 `0.2.0` 而言）或该次切换的负责人 |
| **机会性/见缝插针** | 任何已经在排队的 GPU 窗口（Track A/B/F 的窗口） | RK9 相关审计（C7-1/C7-3）、C1/C2/C3 的 GPU 确认段 | 零增量 GPU 成本 | 持有该窗口的人，顺手带一句 |

**通用规则（适用于本文档提出的所有新门禁）**：任何新增门禁在合入前必须给出"如何证明
它会红"的方法，并写进它的 commit message 或 docstring——三种可接受的方法（§2/§3 每条
都用了其中一种）：

1. **历史回放**：在某个已修复 bug 的父提交上跑，确认变红，在修复提交上变绿
   （C-LIVE B-4 与 E3 已经在用这个方法）。
2. **构造性证明**：该检查的目标行为今天压根不存在，所以从写下来那一刻就是红的
   （E2-b、E-N1-a 属于这类）。
3. **已知坏输入回放**：喂一个已知会违反约束的输入，确认检查真的会拒绝它，而不是
   静默通过（C9 的质量回归对比、E-N1 的 grammar 违反测试属于这类）。

任何新门禁如果三种方法都用不上（既没有历史 bug 可回放，也无法构造出会失败的场景），
应该被怀疑——这正是 C8 存在的理由。

---

## 5. GPU 窗口调度：不与 Track B 抢卡

硬约束（来自 roadmap RK5）：单 GPU，所有需要 GPU 的验收天然串行。本文档的原则——
**"蹭窗口"而不是"申请专用时段"**，与 Track F 的 F1/F2 已经用的策略一致。

**关键调度原则：把 Laguna 侧的验证前置到 M1→M2（Track A 还在占用 Laguna GPU 窗口的
阶段），把 M3 之后的 GPU 需求默认蹭 Track B 的 Qwen3.6 窗口**——因为 M3 起 GPU 压力的
主体转移到 Track B，继续单独申请 Laguna 专用时段就是在制造第二个抢卡方（RK5 的原话）。

| 本文档条目 | GPU 需求 | 蹭哪个窗口 | 理由 |
|---|---|---|---|
| E2-b（采样+投机） | Laguna DFlash 会话 | Track A 第 5–8 步窗口 或 Track F F1 窗口（同一台 Laguna warm engine） | 同一个模型、同一批代码区域，天然同窗口 |
| E-N1-b（结构化输出实现） | Laguna decode 循环 | Track A 第 5–8 步之后的**任意**窗口（不早于——必须等 laguna.py/laguna_cuda_graph.py 在 Track A 手里稳定下来，否则两边改同一批文件会打架） | 文件归属边界（`docs/api-layer-design.md` §5.1 已经点出这个边界） |
| E3-a 首次真机运行 | 任意已加载模型的热身服务器 | Track A 第 5 步 或 C-LIVE B-4 窗口 | 与模型无关，蹭"服务器已经起来了"这个状态即可 |
| E3-b/c/d 后续矩阵刷新 | 同上 | M3 起默认蹭 **Track B 的窗口**（B2/B3，无论当时跑的是 Laguna 还是 Qwen3.6，矩阵对着"当前在跑的模型"验证即可，不要求同时验两个模型） | GPU 压力主体转移；矩阵关心的是协议层，模型是谁不重要 |
| C1-backend/C1-slot/C2/C4 GPU 确认段 | Laguna backend 内部状态 | Track A 第 5–8 步窗口（同批改动，同批验证） | 这些本来就是 Track A 落地要验的东西的一部分 |
| C9-a/b（Laguna 质量基线） | Laguna 稳态推理 | Track A 第 8 步 A6 验收窗口之后、Track F F1/F2 窗口 | A6 验收本身就要跑一批真实 prompt，质量回归可以搭这班车 |
| C9-c（Qwen3.6 质量基线） | Qwen3.6 稳态推理 | Track B B1/B2 窗口 | 只有 Qwen3.6 能跑起来之后才有意义，天然同窗口 |
| C5 24h soak（两个检查点） | 独占一整天 | **不能蹭，必须日程化预留**：建议 M3 末（B1/B2 验收后，B3 性能冲刺前的空档）与 M6 末（发布前）——这两个点是 Track B 自身节奏里天然的间隙，不是额外抢占 | 24h 独占是本清单里唯一真正需要专属时段的条目，必须提前协调，不能假设"顺路" |
| C8 门禁可信度审计 | 无 | 不需要 | 纯静态/历史分析 |

---

## 6. 新增条目一览（roadmap/implementation-plan 原来没有的）

| 条目 | 位置 | 为什么新增 |
|---|---|---|
| L1 · SDK 一致性矩阵 / E3-a~d 分期 | Track E | roadmap 已经写了"E3 客户端验证矩阵"这个目标，但没有分期、没有"怎么验证它真的测到了协议层而不是我们自己的假设"的方法；本文档把它从一句话目标变成可执行、可验证红的四步 |
| E2-a/b、E-N1-a/b 的显式分期 | Track E | 原来是一句话条目（"E2 采样+投机共存""E1-N1 结构化输出真正实现"），读代码后发现每条都有一个非显然的架构阻塞点，一步走的排期会低估工作量或做出"看起来接上但没接上"的东西 |
| **C8 门禁可信度周期审计** | Track C | 直接回应用户点名的 `bfdiag/checkpoint` 教训——把"发现一次假门禁"变成"定期抽查是否有新的假门禁"，是"值得反复审查"这条要求在 Track C 里的具体落地 |
| **C9 质量回归**（MMLU-Pro/HumanEval+ 对 Laguna） | Track C | roadmap 完全没有这条——现有的 `benchmarks/official/`、`quality_regression.py` 从 2026-07-22 起就在，但从未指向过 Laguna，是一个"已经建好但没人用在当前生产模型上"的资产；同时揪出了 roadmap §0 "84.5%"引用疑似误引这条待核实项 |
| C1 的三段拆分（C1-backend/C1-protocol/C1-slot） | Track C | 原来是一条九项失败面的大清单，没有排期；按"跟哪个 GPU 窗口走"重新分组，让它可执行 |
| §4 的"如何证明会红"三分类方法论 | 跨 Track C/E | 把 C-LIVE B-4 已经用过的方法（父提交必须变红）显式提升为适用于**所有**新门禁的通用规则，不止 C-LIVE 自己用 |
| §5 的"前置到 M1-M2、M3 起蹭 Track B 窗口"调度原则 | 跨 Track C/E | roadmap/implementation-plan 已经在 F1/F2 上用了"蹭窗口"的说法，本文档把它推广成一条贯穿全排期的显式规则，并给出具体的窗口映射表 |

---

## 7. 待拍板 / 待验证 移交清单（本文档不代为决定）

- [ ] **E-N1-a**：结构化输出的中间态选择——(a) 等全量修复 vs (b) 先做显式受限的
  `temperature>0`-only 支持。产品语义判断，见 §2.3。
- [ ] **usage token 缺口 2**：`<usage>` 标签剥离统一到哪种语义（流式的"冻结"还是
  非流式的"删除后继续"）。见 §2.5。
- [ ] **[待验证]** roadmap §0 "Laguna-S-2.1 MMLU-Pro 84.5%"的引用来源——疑似误引已退役
  Qwen3.6/vLLM 的评测结果（`notes/2026-07-22-quality-baseline-and-official-scores.md`
  §7，84.54%）。C9-a 跑完后应该有一个真正针对 Laguna 的数字来核实或替换这条引用，
  但替换 roadmap §0 本身不在本次改动范围内。
- [ ] **[待验证]** `openai==2.34.0`/`anthropic==0.99.0`/`httpx==0.28.1` 具体安装在哪个
  venv（协调者转述，未在本次任务中核实）；E3-a 落地时需要确认，并决定以什么形式声明为
  测试期依赖（不进 `pyproject.toml` 的 `serving`/`cuda` extras）。

---

## 8. 配套文档

- [`roadmap.md`](roadmap.md) — Track C/Track E 的目标与理由（权威）
- [`implementation-plan.md`](implementation-plan.md) — §7.3/§7.5 的执行清单
- [`api-layer-design.md`](api-layer-design.md) — N1/N2/N3/n>1/usage token 的逐项审计证据
- [`diagnostics-guide.md`](diagnostics-guide.md) — `bf diff`/run record 的使用纪律
- [`../notes/2026-07-22-quality-baseline-and-official-scores.md`](../notes/2026-07-22-quality-baseline-and-official-scores.md) — 现有质量回归 harness 的原始设计记录（Qwen3.6/vLLM 时代）
