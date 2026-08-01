# T0-7 分支与 worktree 清单（调研，不执行删除）

问题：`git branch -a` 有 32 个本地分支，`git worktree list` 有 30+ 个
worktree。T0-7 仓库卫生要求给出"哪些能删"的清单，但**删分支不可逆，且这台
机器上可能有别人正在用的 worktree**，所以这份 note 只调研、不执行。

方法：对每个本地分支跑 `git rev-list --count origin/main..<branch>`
（独有提交数）和反向（落后提交数），并用 `git branch --merged origin/main`
确认是否已通过快进/合并提交的方式进入 main（squash-merge 的分支这个检测
会漏——如果某个"有独有提交"的分支其实早就被 squash 进 main 了，人工确认
时应该先核对内容而不是直接删）。调研时间：2026-08-01，基于 `git fetch
--all --prune` 之后的状态。

## 结论摘要

- **16 个分支可安全删除**：已合并进 `origin/main`（`git branch --merged`
  为真）且无独有提交（`ahead=0`）。
- **16 个分支需要人工确认**：有独有提交（`ahead>0`），无论是否显示"未合并"——
  任何一个都可能是被丢弃的实验，也可能是还没来得及提 PR 的真实工作。
- **worktree**：4 个已经是 git 认定的 `prunable`（工作目录已在磁盘上消失，
  只剩 `.git/worktrees/` 元数据），`git worktree prune` 可以直接清，零风险。
  剩下的 worktree 全部对应仍然存在的分支，删除节奏应该跟分支删除节奏走
  （先删分支就没有意义了的 worktree，再 `git worktree remove`）。

## 分支清单

### A 组 · 已合并、零独有提交 → 可安全删除（16 个）

| 分支 | 落后 main | 备注 |
|---|---|---|
| `bfdiag-integration` | 170 | worktree 在 `.claude/worktrees/bfdiag-integration` |
| `cleanup/vllm-dep-and-dead-code` | 274 | 有 `origin/` 对应分支，一并删远端 |
| `fix/tests-and-ci-20260727` | 169 | 同上，有远端分支 |
| `fix/tests-ci-20260729` | 112 | |
| `fix/tool-call-streaming-json-args` | 1 | 有远端分支，worktree 在 `qwen-sm120-runtime-fix-tool-stream-json` |
| `investigate/sparkinfer-upgrade` | 231 | worktree 在 `.claude/worktrees/investigate-sparkinfer-upgrade` |
| `perf/baseline-forward-20260730` | 85 | worktree 在 `qwen-sm120-runtime-baseline-forward` |
| `worktree-agent-a0e22f11934dd2f59` | 209 | `.claude/worktrees/agent-a0e22f11934dd2f59` |
| `worktree-agent-a5733b6ca216a6d1f` | 185 | `.claude/worktrees/agent-a5733b6ca216a6d1f` |
| `worktree-agent-a658f629b4d476fa5` | 209 | `.claude/worktrees/agent-a658f629b4d476fa5` |
| `worktree-agent-a854c70175fad2426` | 209 | `.claude/worktrees/agent-a854c70175fad2426` |
| `worktree-agent-acd4fed94943f1e15` | 185 | `.claude/worktrees/agent-acd4fed94943f1e15` |
| `worktree-agent-af25c9341a59de5cb` | 185 | `.claude/worktrees/agent-af25c9341a59de5cb` |
| `worktree-laguna-e1-server-integration` | 293 | `.claude/worktrees/laguna-e1-server-integration` |
| `worktree-laguna-mid-conversation-system` | 240 | `.claude/worktrees/laguna-mid-conversation-system` |
| `worktree-rename-blackwellm` | 256 | 有远端分支，worktree 在 `qwen-sm120-runtime-vllm-cleanup`... 需核对（见下方"存疑"） |

> `worktree-rename-blackwellm` 的分支名和它对应的 worktree 路径
> (`/home/bot/project/qwen-sm120-runtime-vllm-cleanup` 实际 checkout 的是
> `cleanup/vllm-dep-and-dead-code`) 看起来不匹配——执行删除前请用
> `git worktree list` 重新核对每个分支到底有没有 worktree 在用它，
> 这份清单是分支和 worktree 分开跑的,两张表按名字对不一定完全对得上路径。

### B 组 · 有独有提交 → 需人工确认（16 个）

**任何一个都不要直接删。** `ahead` = 相对 `origin/main` 的独有提交数。

| 分支 | 独有提交 | 落后 main | 备注 |
|---|---|---|---|
| `vllm-removal-phase1` | **49** | 166 | 独有提交数最大的一个；worktree 在 `.claude/worktrees/vllm-removal-phase1`。vLLM 剥离主线已在 `a9cb932`（2026-07-30）完成——需要确认这 49 个提交是否是被合并前的重复历史（这种情况删安全）还是包含没有并入主线的额外内容 |
| `fix/t0-ci-tests` | 5 | 4 | |
| `perf/c394-forward-20260730` | 5 | 86 | 有远端分支；worktree 在 `qwen-sm120-runtime-c394-forward` |
| `fix/t0-api-thinking` | 4 | 4 | |
| `fix/t0-deps` | 4 | 4 | worktree 在 `qwen-sm120-runtime-wt-deps` |
| `fix/t0-runtime-hardening` | 20 | 0 | **与 `fix/t0b-api`/`fix/t0b-diag`/`fix/t0b-hyg`（本分支）指向同一个提交 `43d574e`**——这是三个并行 T0b 工作流的共同分叉点，本身可能是要被合并回去的父分支，不是可以孤立判断的候选；worktree 在 `/tmp/merge-trial`，**目录仍然存在且今天 15:16 有修改**，可能有人正在用 |
| `perf/repro-2ce5-baseline-20260730` | 9 | 34 | worktree 在 `qwen-sm120-runtime-2ce5-baseline` |
| `laguna-prefix-cache` | 3 | 212 | worktree 在 `.claude/worktrees/laguna-prefix-cache` |
| `test-reproduce-80` | 3 | 258 | **没有对应 worktree**（`git worktree list` 里找不到这个分支名），可能是早期实验分支，孤儿分支相对好确认 |
| `worktree-fix-tests-ci` | 1 | 242 | 有远端分支；worktree 在 `.claude/worktrees/fix-tests-ci` |
| `worktree-agent-a1bfdbfb374855cc2` | 1 | 186 | `.claude/worktrees/agent-a1bfdbfb374855cc2` |
| `worktree-agent-a1e6266c057312be7` | 1 | 210 | `.claude/worktrees/agent-a1e6266c057312be7` |
| `fix/t0-ci-thinking-deps-api` | 1 | 4 | worktree 在 `qwen-sm120-runtime-t0-api` |
| `fix/t0b-api` | 20 | 0 | **本次 T0-7 并行任务之一**（另一个 agent 在改 API），不要删；等它合并完再回收 |
| `fix/t0b-diag` | 20 | 0 | **本次 T0-7 并行任务之一**（另一个 agent 在改诊断），不要删；等它合并完再回收 |
| `fix/t0b-hyg` | 20 | 0 | 就是本 note 所在的分支（这次仓库卫生工作），当前 HEAD |

`main` 本身也显示 `behind=3`（落后 `origin/main` 3 个提交）——本地 `main`
分支指针没有跟最新的 `origin/main` 同步，这和分支清理无关，但建议顺手
`git fetch && git checkout main && git merge --ff-only origin/main`。

## worktree 清单

### 已经 prunable（工作目录已删除，纯元数据，`git worktree prune` 零风险）

- `/tmp/qsr-66d`（HEAD `66d5913`，对应 `benchmarks/repro_80tok_m1_decode_cg.py`
  文档里提到的"replicating commit 66d5913 methodology"）
- `/tmp/qsr-fd333`（HEAD `fd33368`）
- `/tmp/qsr-legacy-634-0AwCfs`（HEAD `6349751`，与 `vllm-removal-phase1`
  分支尖端一致）
- `/tmp/qsr-review-c4RYlh`（HEAD `fd33368`，与 `qsr-fd333` 同一个提交）

### 看起来仍在使用，不要删

- `/tmp/merge-trial` — `fix/t0-runtime-hardening`，**今天 15:16 有文件修改**，
  时间点在本次 T0-7 三个并行 agent 启动前后,可能是协调工作用的临时合并区。
- `/home/bot/project/qwen-sm120-runtime`（主工作区）、
  `qwen-sm120-runtime-w2-api`、`qwen-sm120-runtime-w2-diag`、
  `qwen-sm120-runtime-w2-hyg`（本 worktree）— 这四个是本轮 T0-7 三个并行
  agent 的工作区，都不要动。

### 其余（对应 A/B 组分支，删除节奏跟分支走）

其余 `/home/bot/project/qwen-sm120-runtime-*` 与
`/home/bot/project/qwen-sm120-runtime/.claude/worktrees/*` 均对应上面
A/B 组里列出的分支，一一对应关系见 `git worktree list` 输出，删除时先删
worktree（`git worktree remove <path>`）再删分支,顺序反了 git 会因为
"branch is checked out" 拒绝删分支。

### ⚠️ 异常发现：本 session 的 scratchpad 目录内有一个未知来源的 worktree

`git worktree list` 里有一条指向
`/tmp/claude-1002/.../afb58670-9850-4f48-be9d-100cdcad03b5/scratchpad/decoy-worktree`
的 detached-HEAD worktree（HEAD `43d574e`，即本次 T0b 三分支共同的分叉点）。
**这不是本次任务创建的**——本 note 的作者在开始工作前就发现这个 scratchpad
目录里已经有一批非本任务产物的文件（`decoy-worktree/`、`ab_subset_probe.py`、
`groupA_*` / `groupB_*` 日志与探针脚本、`sitecustom/`、`w2-diag-ci/` 等，
时间戳集中在今天 13:49–15:33,即本任务开始之前）。这些文件**没有被读取内容
或执行**,只用 `ls`/`stat` 确认了存在性,原样保留未做任何改动。写出来是因为：
(1) 这不应该出现在一个"session 专用"的 scratchpad 里；(2) 如果这是别的
agent/进程的真实工作产物,删除前需要先确认归属,不能当作噪声清掉。

## 建议的执行顺序（不是本 note 的范围，仅供参考）

1. `git worktree prune` — 清掉 4 个已 prunable 的纯元数据条目,零风险。
2. A 组 16 个分支：`git worktree remove <path>`（如果有对应 worktree）→
   `git branch -d <branch>` → 如果有远端对应分支,`git push origin
   --delete <branch>`。
3. B 组 16 个分支：逐个人工核对（尤其 `vllm-removal-phase1` 的 49 个独有
   提交、`fix/t0-runtime-hardening` 与三个并行 T0b 分支的关系、
   `test-reproduce-80` 这个孤儿分支），确认哪些内容已经通过别的路径落地
   （例如 squash-merge，`git branch --merged` 检测不到）。
4. `decoy-worktree` 和 scratchpad 里的其它异常文件：先确认来源,不要当作
   本次任务的清理对象处理。
