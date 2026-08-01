# SparkInfer 性能补丁：丢失、恢复、复现

日期：2026-08-01
结论：**基线已复现。** 补丁不是"未上游"，是从未提交且丢失，已找回并合入
`jieen1/sparkinfer` 的 `origin/master`。

---

## 1. 问题

`notes/2026-07-31-session-summary.md` 记录的性能基线依赖对 SparkInfer 的
gating 放宽改动。2026-08-01 复查时，`/home/bot/project/sparkinfer` 工作区干净、
`git log --all` 搜不到任何相关提交，最初判定为"补丁丢失、基线不可复现"。

**那个判定是对现象的正确描述，但结论下早了。**

## 2. 补丁在哪

`git log --all` 不会走到**无引用（unreachable）的提交**。补丁在悬空对象里：

| 提交 | 内容 | 改动量 |
|---|---|---|
| `1e306d7` | Unlock Laguna analytic decode kernel for TP=1 (48/8 heads, page_size=64) | `_forward.py` + `planner.py`，−12/+6 |
| `ec8bb1eb` | FA4-inspired: decouple FP8 PV from FP8 QK for verify path | `_forward.py`，−2/+7 |

两个都带完整的 commit body（Constraint / Rejected / Tested 齐全），基于当时的
upstream `a2a8214`。它们从未被任何分支引用，工作区一清就只剩悬空对象，
随时可能被 `git gc` 回收。

找法：

```bash
cd /home/bot/project/sparkinfer
for c in $(git fsck --lost-found | awk '/dangling commit/{print $3}'); do
  git show $c --stat --format='' | grep -q 'paged/_forward.py' && \
    git show -s --format='%h %ad %s' --date=short $c
done
```

## 3. 补丁做了什么

核心只有 4 处准入谓词放宽：把 `num_q_heads==24 and num_kv_heads==4 and
page_size==128`（TP=2 的确切形状）换成 `gqa_group_size==6 and
page_size in (64,128)`。

**kernel 本体本来就是按 runtime `group_size` / `page_size` 参数化的**，
绝对头数只活在准入谓词里 —— 这是补丁安全的根据，也是它只有 20 行的原因。

Laguna 的真实形状是 TP=1：full-attention 层 48 q / 8 kv（gqa=6，**命中**），
SWA 层 72 q / 8 kv（gqa=9，**按设计不命中**，走通用路径）。

## 4. 恢复过程

```bash
# 1. 先导出成 patch 文件，脱离悬空对象的生命周期
git format-patch -o <safe-dir> a2a8214..ec8bb1eb

# 2. rebase 到用户新同步的 upstream 基线
git worktree add --detach <tmp> 3bd3a2e
git am --3way <patches>          # 零冲突

# 3. 合入 master 并推送
git merge --ff-only <tmp-branch>
git push origin master
```

结果：`origin/master` = upstream `3bd3a2e` + `7a1d69d` + `0844a4f`。

`blackforge-main` 分支（本地 + 远端）已删除 —— 相对 `origin/master` 有 0 个
独有提交，无损。**sparkinfer 从此只维护 `master` 一条线**，我们的 delta 就在
上面，不再有需要同步的 fork 分支。

## 5. 复现结果

配置（与 2026-07-31 相同）：

```
QSR_VERIFY_CG_MAX_PAGES=1040
QSR_VERIFY_CG_CTAS_PER_SM=4
# 不设 SPARKINFER_TURBO_ATTN
```

Run records：`fc6b3376785a`、`781e1edbf37b`（两次独立测量，
fingerprint 均为 `qwen-sm120-runtime@40e9cdd` + `sparkinfer@0844a4f dirty=false`）。

| 工作负载 | 07-31 基线 | 实测 1 | 实测 2 | 判定 |
|---|---|---|---|---|
| galaxy-4K | 395–401 | **398.3** | 387.4 | 一次区间内，一次低 1.9% |
| fox-4K | 353–357 | **357.1** | **360.8** | ✅ |
| code-4K | 341–359 | **350.7** | **347.8** | ✅ |
| fox-64K | 353–368 | **359.7** | **360.5** | ✅ |

接受率（比吞吐更重要，是质量门禁）：

| 工作负载 | 07-31 基线 | 实测 | 判定 |
|---|---|---|---|
| galaxy-4K | 100% | 1.0000 | ✅ |
| fox-4K | 96.3–97.0% | 0.9704 | ✅ |
| code-4K | 97.8% | 0.9778 | ✅ |
| fox-64K | 96.9% | 0.9686 | ✅ |

**接受率四项全部小数点级吻合**；吞吐 8 个数字里 7 个落在基线区间内，
唯一例外 galaxy-4K 的 387.4 低于下沿 1.9%，而同一工作负载另一次是 398.3
（区间内），属测量散布。

**结论：基线复现。** 而且这是在**新的 upstream 基线**（`3bd3a2e`，比 07-31
当时的 `a2a8214` 多 28 个提交，其中 3 个动过 `attention/paged/`）上复现的，
所以同时说明那 28 个提交没有破坏这条路径。

## 6. 没做的：B 组对照

未测量"不打补丁"的对照组，所以**补丁的定量价值（吞吐 delta）没有本次实测数据**。
两个原因：

1. 每换一次 SparkInfer commit 就要付一次完整的 CuTe DSL JIT 冷编译
   （实测约 10 分钟以上 8 核满载、GPU 闲置）。新 upstream 的
   `36cade0 compiler: key persistent cache by device UUID` 改了缓存键，
   所有缓存都是冷的。
2. 更硬的原因见第 7 节：**切换 SparkInfer 的机制本身是坏的。**

定性结论仍然成立：`runtime/preflight.py` 的
`check_sparkinfer_analytic_decode_gate` 用真实生产形状（48 q / 8 kv / page 64 /
fp8 KV）去探活的 gate，打补丁后返回 OPEN，纯 upstream 返回 closed。

## 7. 过程中发现的两个真问题

### 7.1 `bf` 在任何 worktree 里都加载主工作区的代码

venv 的 `bf` console script 在 `~/.venvs/vllm/bin/bf`。Python 解析
`import bfdiag` 时 `sys.path[0]` 是**脚本自己所在目录**（venv 的 bin/），
那里没有项目代码，于是落到 pip-editable 的 finder —— 而那个 finder 把
`/home/bot/project/qwen-sm120-runtime`（主工作区）**硬编码**成了包源位置。

**净效果**：`cd <某个 worktree> && bf daemon start` 会静默加载**另一个
checkout** 的 `bfdiag` / `runtime` / `server` / `benchmarks`，run record 和
`.bfdiag` 状态也都来自错误的 checkout。**不报错，不警告。**

这对一个"诊断平台"是严重问题：从 worktree 跑的任何测量都可能在测错的代码。

绕法：`scripts/bf-t0.sh` 用 `PYTHONPATH` 指向当前 worktree —— PYTHONPATH 条目
由 stdlib 的 PathFinder 处理，它在 `sys.meta_path` 里排在 pip-editable 的
MetaPathFinder **之前**。

### 7.2 `BF_SPARKINFER_PATH` 是个坏掉的逃生口

这个环境变量只被两个文件读：`laguna_sparkinfer_attn.py` 和
`laguna_sparkinfer_moe.py`，各自在自己的 `import sparkinfer...` 之前做
`sys.path.insert(0, ...)`。

但 `runtime/backends/laguna.py:593` 的 `_patch_moe_sparkinfer` 有一个**自己的**
直接导入 `from sparkinfer.moe.fused_moe._impl import allocate_tp_moe_workspace_pool`，
而那是整个 daemon 进程里对 `sparkinfer` 名字的**第一次触碰**（用 `__import__`
traceback hook 确认过）。导入结果进 `sys.modules` 之后，后面再改 `sys.path`
**无法回溯重定向**。

所以 `BF_SPARKINFER_PATH` 在真实的 Laguna 启动路径上**根本不生效**，
而且同样不报错。README / 文档里任何"用这个变量切换 SparkInfer"的说法都是错的。

`scripts/bf_sparkinfer_bootstrap/sitecustomize.py` 是绕过用的 shim
（在进程最早期生效）。**但正确的修法是修 `laguna.py` 那处直接导入**，
让所有 sparkinfer 导入走同一条受控路径 —— 这条留给 Track 0 收尾。

## 8. 遗留

- [ ] B 组定量对照（需要先修 7.2）
- [ ] 修 `laguna.py:_patch_moe_sparkinfer` 的直接导入，让 `BF_SPARKINFER_PATH` 真的生效
- [ ] `bf` 的 worktree 解析问题：`scripts/bf-t0.sh` 是绕法不是修法
- [ ] 还有 9 处 gate 未放宽，见 [`../docs/sparkinfer-fork-delta.md`](../docs/sparkinfer-fork-delta.md)
