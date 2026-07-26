# sparkinfer 性能分支合并完成(2026-07-27,用户批准后执行)

## 结论

用户明确批准后,把 ptxas 修复(`6e906b0`)合并进了 `/home/bot/project/sparkinfer` 的
`blackforge-main`。合并提交 `14cb350`,`git merge --no-ff`,无冲突。用主 checkout
(不是诊断阶段用的临时 worktree)重新跑了一次崩溃复现脚本(`/tmp/repro_ice_isolated.py`,
CTX=512,eager decode 5 步)——**`ALL STEPS OK -- NO CRASH`**,确认合并后的生产分支
本身不崩,不只是隔离 worktree 里的验证。

临时诊断 worktree(`/tmp/sparkinfer-ice-repro`)已用 `git worktree remove --force`
清理。

## 当前状态

- `/home/bot/project/sparkinfer`:`blackforge-main @ 14cb350`,干净,包含:
  - K/V 竞态修复(`d2d8cb9`)、MoE 确定性修复(`989723d`)、bincount→scatter_add
    修复(`3fa9b54`)——这三项本来就在
  - master 的 Laguna SM120 kernel 性能特化(`478b9af`,traits.py 特化、PTX FP8→BF16
    转换优化、compact_sync_rows、`plan_extend_graph_capacity`/`compile_paged_attention`
    等,详见 `notes/2026-07-27-*sparkinfer*.md` 系列前几篇笔记的调研记录)
  - ptxas ICE 修复(`6e906b0`,`forward_paged.py` 6 处调用换成 portable fallback)
- 未 push 到 origin(本次任务范围内没有要求推送,保持本地领先状态)。

## 这打开了什么

`notes/2026-07-27-sparkinfer-merge-and-verify-cg.md` 里 P2(block_size 64→128 迁移)
和 P3(mode="verify" 切换验证)之前因为合并被回退而暂停——现在合并已经带着修复重新
落地,这两项评估任务(`任务 #13`/`#14`)重新解封,可以恢复推进,目标是真正让 verify
CG 吃到 Laguna kernel 特化、追上甚至超过 eager。
