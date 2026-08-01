# T0-7 分支清理：执行记录（2026-08-01）

前置调研：`notes/2026-08-01-t0-7-branch-worktree-survey.md`（同一 worktree 下）。
本文档记录**实际执行**的删除操作、判定依据、以及为什么绝大多数候选分支
这一轮**没有**被删除。

## 0. 关键发现：几乎所有分支都被 worktree 占用

执行前 `git worktree list` 显示 **33 个 worktree**（含主工作区），对应
**31 个本地分支**处于"被 worktree 占用"状态——覆盖了前一份调研里 A 组
（16 个"零独有提交,可安全删"）和 B 组（16 个"需人工确认"）里的几乎全部
条目。用 `git branch -d bfdiag-integration` 实测验证过：

```
error: cannot delete branch 'bfdiag-integration' used by worktree at
'/home/bot/project/qwen-sm120-runtime/.claude/worktrees/bfdiag-integration'
```

任务指令明确要求"任何被 worktree 占用的分支都不要删"，这是硬规则，不因
分支内容"没用"而豁免。因此：**本轮删分支操作只覆盖了唯一一个没有
worktree 占用的分支**（`test-reproduce-80`）。其余 30 个分支（含
`vllm-removal-phase1`）全部完成了价值判断,但因 worktree 占用被保留,
判断结果记录在下面,供 worktree 收尾后据此执行。

远端分支同理：`origin` 上现存的 9 个分支（`cleanup/vllm-dep-and-dead-code`、
`fix/t0-runtime-hardening`、`fix/tests-and-ci-20260727`、
`fix/tool-call-streaming-json-args`、`main`、`perf/c394-forward-20260730`、
`vllm-removal-phase1`、`worktree-fix-tests-ci`、`worktree-rename-blackwellm`）
除 `main`/`fix/t0-runtime-hardening` 外,对应的本地分支全部被 worktree
占用——本轮**未删除任何远端分支**,保持与本地一致的谨慎口径（删除本地
分支和删除远端副本这次绑定处理,没有本地先例的情况下不单独动远端）。

## 1. 实际删除（1 个）

| 分支 | 删除前 HEAD SHA | 远端副本 | 判定依据 |
|---|---|---|---|
| `test-reproduce-80` | `486b00b6a45ad50f6244b64ca1d657cfb41dae7a` | 无 | 见下 |

判定过程：
- 无 worktree 占用（`git worktree list` 里找不到这个分支——是唯一一个
  裸分支）。
- `ahead(origin/main..test-reproduce-80) = 3`,乍看像"有独有提交需要人工
  确认"，但看完 3 个提交的内容后发现它们本身就是 cherry-pick：
  - `486b00b vllm-compat: CG capture impl restore (cherry-pick c90b009)`
  - `183b3cd vllm-compat: BFAttention replacement (cherry-pick ef7a036)`
  - `92179f6 vllm-compat: self-allocated KV cache (cherry-pick 8e5c504)`
  用 `git merge-base --is-ancestor <sha> origin/main` 验证：`c90b009`、
  `ef7a036`、`8e5c504` **三个都已经是 `origin/main` 的祖先**。也就是说
  `test-reproduce-80` 的"独有提交"是把 main 上已有的修复反向 cherry-pick
  进这个实验分支自用,不是这个分支贡献了新东西给 main。该分支落后 main
  258 个提交,是 2026-07-25 前后的一次性 benchmark 实验分支（提交历史里
  能看到 SparkInfer/FlashInfer attn 对比、block_size 调参等一次性记录）。
  **零独有价值,删除无损失。**
- 无远端副本,`git branch -D` 后无需处理远端。

执行命令与结果：
```
$ git rev-parse test-reproduce-80
486b00b6a45ad50f6244b64ca1d657cfb41dae7a
$ git branch -D test-reproduce-80
Deleted branch test-reproduce-80 (was 486b00b).
```

取回方式（如果判断有误）：`git branch test-reproduce-80 486b00b6a45ad50f6244b64ca1d657cfb41dae7a`
（对象未被 gc,short window 内可直接取回）。

## 2. 保留：因 worktree 占用而未删除，但已完成价值判断

以下分支**全部**当前被 worktree 占用（见附表），因此本轮不删除,不动
远端。价值判断按任务给的标准（是否已通过其他形式进 main / 内容现在还
有没有价值 / 拿不准就保留）执行,供将来 worktree 收尾后参考。

### 2.1 零独有提交,已完全并入 origin/main（16 个,worktree 收尾后可安全删）

`ahead(origin/main..branch) = 0`——每个提交都是 `origin/main` 的祖先,
删分支不会丢任何东西。

| 分支 | worktree 路径 | 远端副本 |
|---|---|---|
| `bfdiag-integration` | `.claude/worktrees/bfdiag-integration` | 无 |
| `cleanup/vllm-dep-and-dead-code` | `qwen-sm120-runtime-vllm-cleanup` | 有 |
| `fix/tests-and-ci-20260727` | `.claude/worktrees/fix-tests-ci-20260727` | 有 |
| `fix/tests-ci-20260729` | `qwen-sm120-runtime-tests-20260729` | 无 |
| `fix/tool-call-streaming-json-args` | `qwen-sm120-runtime-fix-tool-stream-json` | 有 |
| `investigate/sparkinfer-upgrade` | `.claude/worktrees/investigate-sparkinfer-upgrade` | 无 |
| `perf/baseline-forward-20260730` | `qwen-sm120-runtime-baseline-forward` | 无 |
| `worktree-agent-a0e22f11934dd2f59` | `.claude/worktrees/agent-a0e22f11934dd2f59` | 无 |
| `worktree-agent-a5733b6ca216a6d1f` | `.claude/worktrees/agent-a5733b6ca216a6d1f` | 无 |
| `worktree-agent-a658f629b4d476fa5` | `.claude/worktrees/agent-a658f629b4d476fa5` | 无 |
| `worktree-agent-a854c70175fad2426` | `.claude/worktrees/agent-a854c70175fad2426` | 无 |
| `worktree-agent-acd4fed94943f1e15` | `.claude/worktrees/agent-acd4fed94943f1e15` | 无 |
| `worktree-agent-af25c9341a59de5cb` | `.claude/worktrees/agent-af25c9341a59de5cb` | 无 |
| `worktree-laguna-e1-server-integration` | `.claude/worktrees/laguna-e1-server-integration` | 无 |
| `worktree-laguna-mid-conversation-system` | `.claude/worktrees/laguna-mid-conversation-system` | 无 |
| `worktree-rename-blackwellm` | `.claude/worktrees/rename-blackwellm` | 有 |

执行建议（不是本轮范围）：先确认对应 worktree 确实不再使用
（`git worktree remove <path>`,如目录有未提交改动 git 会拒绝,需人工看一眼)，
再 `git branch -D <branch>`,有远端的再 `git push origin --delete <branch>`。
这 16 个是零风险操作,唯一的前置条件是清掉 worktree。

### 2.2 有独有提交，逐个判断（15 个，worktree 占用中）

#### `vllm-removal-phase1` —— 已完全并入 main，历史过程分支，结论明确

- **独有提交**：49 个（相对 `origin/main`）,`behind=166`。
- **HEAD SHA**（供将来核实/取回）：`6349751c127ad022ffe33d82b9572cd2da5a870c`。
- **结论：内容已 100% 通过手工分批合并进 `origin/main`,是已完成工作的
  历史过程分支,没有独有价值。**

  证据链：
  1. `origin/main` 历史里有连续完整的 11 个"Merge batch"提交
     （`de56b63` batch1 → `04554e6` batch2 → `e9266e5` batch3 →
     `7e44c87` batch4 → `b7d869c` batch5 → `bd45a3f` batch6+7 →
     `98737eb` batch8 → `cce3a3e` batch9 → `368f277` batch10 →
     `571dc6f` batch11，均在 2026-07-29 14:58–15:22 之间提交,序号
     连续无缺口）,每条提交信息都写明"from vllm-removal-phase1"。
     用 `git merge-base --is-ancestor` 逐个验证,全部 11 个都在
     `origin/main` 上。
  2. `vllm-removal-phase1` 的 49 个独有提交总共触碰 63 个文件
     （`git diff <merge-base> vllm-removal-phase1 --name-only`）。
     把这 63 个文件与 11 个 batch 提交实际改动的文件集合
     （`git diff-tree --name-only -r <batch-sha>`)做差集,只剩 4 个文件
     没被任何 batch 直接触碰：
     - `benchmarks/fixtures/full_comparison_ours.json`（数据 fixture）
     - `docs/diagnostics-guide.md`（文档）
     - `notes/2026-07-27-fused-kv-scatter-negative-slot-bug-fixed.md`（历史笔记）
     - `runtime/kernels/fused_kv_scatter.py`（唯一的代码文件,值得深查）
  3. 对 `runtime/kernels/fused_kv_scatter.py` 单独核实：
     `vllm-removal-phase1` 上有两个提交修复过这个文件
     （`f9529f4` "value load used key's strides, corrupting V"、
     `2ba6d45` "skip negative (padding) slot_mapping entries"）。
     检查 `origin/main` 当前版本,两个修复**都在**——
     `grep -n "if slot < 0"` 命中"Padding token"处理逻辑,
     `grep -n "stride_v"` 确认 value load 用的是 `stride_v*` 不是
     `stride_k*`。`git diff` 两边只剩纯格式化差异（多行参数被后续 ruff
     格式化合并成单行,逻辑零变化,`git diff --shortstat` = 16 insertions/
     41 deletions,全是换行风格）。**这两个修复已经通过别的提交路径
     （不是这 11 个 batch,可能是独立重新发现或代码格式化时顺带带过)
     落地了,不是缺口。**
  4. `origin/main vllm-removal-phase1` 两边同名文件（63 个里的 44 个）
     确实存在较大差异（44 files changed, 6118 insertions(+), 3688
     deletions(-)）,但这是**main 在 batch11（2026-07-29 15:22）之后继续
     独立演进**造成的,不是"vllm-removal-phase1 有东西没进main"：用
     `git log 571dc6f..origin/main -- <这63个文件>` 查到 batch11 之后
     还有 79 个提交碰过这些文件,都是后续独立工作（前缀缓存实现、
     TURBO_ATTN 实验、DFlash verify-graph 分叉边界研究、NVFP4 micro
     split 实验等),日期全部晚于 2026-07-29 15:22。
  5. 与 `docs/roadmap.md` §1.1 对照：现状盘点写"vLLM 完全剥离 ✅ 生产
     路径零 vLLM 依赖"，与"vllm-removal-phase1 的工作已经落地"的结论
     一致。

  **建议**：worktree（`.claude/worktrees/vllm-removal-phase1`）清理后,
  这个分支可以安全删除,不需要保留期。如果之后发现遗漏,可从
  `6349751c127ad022ffe33d82b9572cd2da5a870c` 找回具体提交。

#### 其余 14 个：内容像是仍在进行的 T0 系列工作，判定"保留/存疑"，不建议删

这些分支的独有提交都不多（1–9 个）,但看内容**不像一次性实验的残留**,
更像是这次 T0/T0b 并行任务序列（`fix/t0-*`）里其它并行 agent 的工作
产物,或者是配套的诊断/性能验证记录。没有做逐一的"是否已经落地 main"
深挖（因为反正被 worktree 占用,本轮不能删,深挖的边际价值低于花的时间),
只记录内容概况和保留理由,供人工在收尾这些 T0 任务时判断。

| 分支 | 独有提交数 | 内容概况 | 建议 |
|---|---|---|---|
| `fix/t0-runtime-hardening` | 20 | 本轮 PR #7,评审中 | **明确排除,不判断** |
| `fix/t0b-api` / `fix/t0b-diag` / `fix/t0b-hyg` | 各 20 | 本轮 T0-7 三个并行 agent 工作区 | **明确排除,不判断** |
| `fix/t0-ci-tests` | 5 | CI 增加 CPU-torch job、修复 DFlashEngine 测试 fixture drift、恢复 CPU-only 单测契约、ruff format 四个测试模块 | 看起来是 T0 系列 CI 修复工作,可能已被后续 T0 分支重做或部分吸收,需要人工核对是否与当前 CI 状态重复 |
| `perf/c394-forward-20260730` | 5 | BF16 router 精确路由从"gate 开关"变成"生产默认路径"（`6bc24b8` "Adopt exact BF16 routing as Laguna's production path"） | 性能主线工作,需人工确认是否已通过 `perf/repro-2ce5-baseline-20260730` 或其它路径落地 |
| `fix/t0-api-thinking` | 4 | thinking/reasoning span 检测修复 + API 错误体格式修复 + API 分层设计文档 | 与 `fix/t0b-api`（本轮正在做 API 的 agent）主题重叠,很可能是前置工作,人工确认是否已被 t0b-api 吸收 |
| `fix/t0-deps` | 4 | 依赖契约修复 + 启动 preflight 检查 + SparkInfer patch 审计文档改名 | 独立的依赖/启动检查工作,内容具体（`runtime/preflight.py` 738 行 + 测试424行),看起来完整,需人工确认是否已合并 |
| `perf/repro-2ce5-baseline-20260730` | 9 | 一系列"Batch 1-4"性能基线复现记录 + decode 循环微优化,与 main 当前基线对比（"Verify main-merged perf...64K=310 tok/s"） | 性能验证记录性质,可能是historical baseline archive,内容多为记录/验证而非新功能,价值主要是历史证据,不建议直接删但优先级低 |
| `laguna-prefix-cache` | 3 | L-P0 block-table indirection + L-P1 设计草稿（未实现) | 前缀缓存已在 main 上实现（`9db2647` "Implement prefix cache for LagunaBackend" 等一系列近期 commit),需要核对这个分支的 L-P0/L-P1 设计是不是被那批 main 提交取代了——**没有做深挖,存疑保留** |
| `worktree-fix-tests-ci` | 1 | CI 里 actions/checkout、setup-python 升到 v7 (Node 24) | 小改动,需核对当前 `.github/workflows/ci.yml` 是否已经是 v7,如果是则可安全删 |
| `worktree-agent-a1bfdbfb374855cc2` | 1 | 修复 `LagunaEngineProvider` 静默加载期默认值问题（2026-07-27 事故复盘) + daemon/provider 改动 + 434 行事故笔记 | 具体的 bug 修复,需核对是否已通过其它提交落地到 `bfdiag/daemon/provider.py` |
| `worktree-agent-a1e6266c057312be7` | 1 | bfprobe P2b：MoE 路由探针 + vLLM oracle tap 调研 | 调研性质,vLLM oracle 现已隔离到 `oracle/qwen36_vllm/`（`a9cb932`),这个调研可能已经过时（因为 vLLM 路径已经不在生产范围内),但没有确认改动本身是否已落地,存疑保留 |
| `fix/t0-ci-thinking-deps-api` | 1 | 与 `docs/roadmap.md`/`docs/model-support.md`/`notes/README.md` 相关的文档改写("Rewrite documentation baseline for the SM120 runtime pivot") | 文档基线改写,需核对当前 `docs/roadmap.md`（2026-08-01 版本,基线 commit `ce21eb5`)是不是已经取代了这个分支的版本——**很可能已经被取代**（当前 roadmap 明显更新更完整),但没有做逐字核对 |

**没有一个判定为"直接删除"**——按任务要求"拿不准就保留",这 14 个都
列进保留清单,不建议在没人工核对内容的情况下删除,尤其是 `fix/t0-*` 系列
和 `laguna-prefix-cache`，因为它们的主题与本轮/近期真实工作（T0 系列、
前缀缓存实现）高度重叠,很可能是还没吸收完的真实工作,不是可以安全丢弃
的残留。

## 3. 需要人拍板的项

1. **worktree 清理节奏**：本轮完全没有清理 worktree（任务范围内不允许)。
   30 个 worktree 里,§2.1 的 16 个一旦确认对应工作已完成,可以走
   "先 `git worktree remove` 再删分支"的标准流程,零风险。§2.2 的 14 个
   （不含 4 个明确排除的）需要先人工核对内容是否已被吸收,再决定要不要
   连 worktree 一起清。
2. **`worktree-agent-*` 和 `agent-*` 系列的归属**：这批分支名带
   `worktree-agent-<hash>` 的,看起来是 Claude Code Agent 工具
   `isolation: "worktree"` 功能产生的历史会话残留,但本次没有找到能确认
   "这些 agent 会话已经结束,产出已经处理完"的证据——只是通过内容判断
   （零独有提交或内容已被别的提交覆盖)。如果人工能确认这些会话早就
   完结,可以更有信心地按 §2.1/§2.2 的判断执行清理。
3. **`perf/*-20260730` 三个性能分支**（`baseline-forward`、`c394-forward`、
   `repro-2ce5-baseline`）看起来是同一批性能验证工作的不同阶段/角度,
   建议放在一起由熟悉这轮性能工作的人一并核对,而不是逐个孤立判断。
4. 本文档只对 `vllm-removal-phase1` 做了逐文件级别的深度核实,§2.2 表格
   里的其余 14 个分支判断力度较浅（主要基于 commit message 和
   diffstat,没有逐文件核对是否被 main 吸收)。如果需要同等确定性的
   结论,需要对每个分支重复"批次合并/cherry-pick 检测 + 关键文件内容
   核对"的方法论,工作量与 `vllm-removal-phase1` 那次分析相当。

## 4. 附：本轮未触碰的东西（按任务要求，明确记录）

- `main`、`fix/t0-runtime-hardening`、`fix/t0b-api`、`fix/t0b-diag`、
  `fix/t0b-hyg`：未做任何分支操作。
- 主工作区（`/home/bot/project/qwen-sm120-runtime`）：只跑了只读 git
  查询（`git fetch --all --prune`、`git log`、`git diff`、`git branch --merged`
  等),没有 `git add`/`git commit`/`git checkout`/`git stash`,没有编辑任何
  文件。主工作区里已有的未提交改动（`AGENTS.md`、`README.md`、
  `docs/roadmap.md` 等)在整个过程中原样未动。
- 所有 30 个仍被 worktree 占用的分支：未删除,未 `git worktree remove`。
- 远端 `origin` 上的所有分支：未删除任何一个。
