# 紧急回退:sparkinfer master 性能合并导致 eager decode 100% 崩溃(2026-07-27)

## 发生了什么

今天早些时候(见 `notes/2026-07-27-sparkinfer-merge-and-verify-cg.md`)把 sparkinfer
`master` 分支的 Laguna kernel 性能特化合并进了 `blackforge-main`(`3fa9b54` →
`478b9af`)。那次验证只测了 DFlash verify CG 的 A/B(`ab_verify_cg.py`,走
`DFlashEngine`),确认没有回归就保留了合并。

后续做 P1(CUDA Graph 接入 `decode_batch_sampled`)任务时,验证脚本第一次真正跑了
"plain eager M=1 decode_batch_sampled 连续多步循环"(不经过 `DFlashEngine`,主模型
最基础的解码路径)——**在 `478b9af` 上 100% 复现崩溃**:

```
cutlass.base_dsl.common.DSLRuntimeError: DSLRuntimeError: 🧊🧊🧊 ICE 🧊🧊🧊
Caused exception: ... NVPTX compiler invocation failed ...
ptxas application ptx input, line 705; error: Unexpected instruction types specified for 'cvt'
```

- 复现 3/3(含清空 `~/.cache/sparkinfer` JIT 缓存后重跑),与上下文长度无关(4096/512
  均触发)。
- 崩溃点:`decode_batch_sampled` → `_forward` → 主模型 attention 层 →
  `laguna_sparkinfer_attn.py:158`(eager、`enable_cuda_graph=False` 的通用 decode
  attention 路径)→ sparkinfer 的 CuTeDSL JIT 编译。
- **已用 bisect 式对照确认是这次合并引入的**:临时切回 `3fa9b54`(合并前),同一脚本
  完全正常。

`ab_verify_cg.py` 那次验证之所以没抓到,是因为它走的是 `DFlashEngine`,从来不触碰
主模型这条最基础的 plain eager decode 路径——两次验证覆盖的代码路径不重叠。

## 处理

**立即把 sparkinfer `blackforge-main` 分支指针退回 `3fa9b54`**(合并前、本次 session
从头到尾验证过的已知良好状态),把有问题的合并提交保留为 tag
`broken-478b9af-ptxas-ice`(不删除,留给后续排查),不留在 detached HEAD 上:

```bash
cd /home/bot/project/sparkinfer
git tag broken-478b9af-ptxas-ice 478b9af
git branch -f blackforge-main 3fa9b54
git checkout blackforge-main
```

当前状态:`blackforge-main @ 3fa9b54`,干净,`broken-478b9af-ptxas-ice` tag 指向问题
提交供以后排查用。

## 为什么这么处理,而不是先排查

- 崩溃点是主模型**最基础的 eager decode 路径**,不是什么边缘功能——任何触发这条
  kernel 编译的真实流量都会崩,风险面远大于"verify CG 还没修好"这种性能问题。
- 合并至今**没有任何已经兑现的收益**(verify CG A/B 测出来还是慢,详见
  `notes/2026-07-27-sparkinfer-merge-and-verify-cg.md`),只有确认的崩溃风险,继续
  停留在 `478b9af` 没有任何理由。
- 排查 ptxas ICE 需要读 PTX/CuTeDSL 生成细节,不是能立刻确定要多久的小活;先回退止血,
  再单独排查,不应该让整个仓库在排查期间处于已知会崩的状态。

## 对之前决策的影响

- `notes/2026-07-27-sparkinfer-merge-and-verify-cg.md` 里"sparkinfer 合并本身予以
  保留(无回归)"这条结论**需要撤销**——当时的验证范围不够,漏掉了最基础的 decode
  路径,不是无回归,是有严重回归。
- 待办 #13(block_size 64→128 迁移)、#14(mode="verify" 切换验证)两项原本是为了
  吃到这次合并带来的 Laguna kernel 特化收益——**在崩溃修好、重新合并之前,这两项都没
  有意义,暂停**。
- P1(CUDA Graph 接入 `decode_batch_sampled`,commit `9ca7612`)本身的实现和正确性
  验证是在 pre-merge sparkinfer(`3fa9b54`)上做的,和这次回退不冲突,现在
  `blackforge-main` 也回到了同一个版本,应该可以正常做完整的端到端验证了(之前被
  post-merge 崩溃挡住)。

## 下一步

1. 单独排查 `478b9af` 引入的 ptxas ICE(大概率是 `traits.py` 新增的
   `fp8x4_e4m3_to_bfloat2x2_native_sm120` 原生 PTX 转换指令和当前 ptxas/CUDA 工具链
   版本不兼容,需要逐个 kernel 变体确认,不是猜的结论,需要专门排查)。
2. 排查修好之前,`blackforge-main` 留在 `3fa9b54`,不要重新合并 `master`。
3. 排查修好后,重新走一遍合并 + 完整验证(这次要覆盖 plain eager decode,不能只测
   DFlash verify CG 一条路径)。
4. 现在 sparkinfer 已经回到验证过的状态,可以补做 P1 的真实端到端 HTTP 冒烟(之前
   被崩溃挡住,只验证到 `LagunaBackend` 层面)。
