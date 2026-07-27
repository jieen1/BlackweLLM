# sparkinfer 分支变更:blackforge-main 已合并进 origin/master,master 现在是唯一权威分支(2026-07-27)

## 变更

用户把 `blackforge-main` 的全部内容合并进了 `origin/master`(PR #3,merge commit
`8db352c`)。`/home/bot/project/sparkinfer`(本项目 venv 编辑安装指向的共享目录)
现在应该跟踪 `master`,不再是 `blackforge-main`。

## 验证过程(过程记录,供以后类似情况参考)

排查 DFlash 深度优化任务时发现共享目录当时停在一个游离的 `origin/master`
(`12c66a6`),缺少 3 个当时只在 `blackforge-main` 上的修复(`3fa9b54` MoE
bincount→scatter_add、`6e906b0` FP8→BF16 widening、`14cb350` ptxas ISA 修复)。
这不是意外——用户当时正在把这些修复合并回 origin/master、并计划切换到 master 作为
canonical 分支,只是合并还没做完,我查到的是中间状态。

合并完成后重新核实(`git fetch origin` + `git merge-base --is-ancestor`):

- `master`(`8db352c`)包含 `d2d8cb9`/`989723d`/`3fa9b54`/`6e906b0`/`14cb350`
  全部 5 个修复。
- `git log --oneline blackforge-main..master` 非空(还多出几个 `master` upstream
  的 pcie 相关提交,和 Laguna 无关),`git log --oneline master..blackforge-main`
  为空——**`master` 是 `blackforge-main` 的严格超集,没有丢任何东西**。

## 结论

- 生产依赖分支:**`master`**,不再是 `blackforge-main`。
- `runtime/backends/laguna_sparkinfer_moe.py` 顶部文档注释已更新。
- 本仓库历史 notes 里大量提到 `blackforge-main @ <commit>` 的记录(复现步骤、
  版本决策等)是当时的真实状态,不需要回改——但**以后新的复现/环境核实,应该
  确认 `sparkinfer` 在 `master` 分支上**,不要再假设 `blackforge-main` 存在或
  是权威分支。
- GPU 验证(`decode_batch_sampled` num_reqs>1 崩溃修复的复测)在这次分支确认
  之后重新跑,确保不是针对中间态跑的。
