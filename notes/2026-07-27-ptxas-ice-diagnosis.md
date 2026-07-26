# ptxas ICE 根因诊断与修复(2026-07-27)

## 结论

**根因已精确定位,修复已验证,但修复尚未合并进 `blackforge-main`——最后一步合并操作
被权限分类器拦截,需要人工/协调者显式批准后完成。**

## 精确复现

在独立 worktree(`/tmp/sparkinfer-ice-repro`,checkout 自 tag `broken-478b9af-ptxas-ice`,
不影响主 `/home/bot/project/sparkinfer` checkout)上,用 `benchmarks/measure_decode_cg_throughput.py`
同款的 plain eager `decode_batch_sampled` 连续解码复现:

```
[rank0]: cutlass._mlir._mlir_libs._site_initialize.<locals>.MLIRError: Failure while executing pass pipeline:
[rank0]: error: unknown: NVPTX compiler invocation failed, error log: ptxas application ptx input, line 705; error   : Unexpected instruction types specified for 'cvt'
[rank0]:   ptxas application ptx input, line 706; error   : Unexpected instruction types specified for 'cvt'
... (数百个成对出现的同类错误)
```

调用栈:`decode_batch_sampled` → `_forward` → `laguna_sparkinfer_attn.py:158`(eager
decode attention)→ `sparkinfer/attention/paged/_forward.py:1452` `paged_attention_forward`
→ `sparkinfer/_lib/compiler.py` 的 CuTeDSL JIT 编译(`compile_and_jit`/`launch`)。

## 根因(不是猜的,有直接证据)

用环境变量 `CUTE_DSL_KEEP=ptx CUTE_DSL_DUMP_DIR=/tmp/cute_dump` 把实际生成的 PTX 导出,
崩溃 kernel(`forward_paged` 系列)的 PTX 头是:

```
.version 9.1
.target sm_120a
```

而触发崩溃的指令 `cvt.rn.bf16x2.e4m3x2`(来自 `sparkinfer/_lib/intrinsics.py` 的
`fp8x4_e4m3_to_bfloat2x2_native_sm120`,该函数自己的 docstring 就写明
"requires PTX ISA 9.2")**需要 PTX ISA 9.2,但生成的模块只声明了 9.1**。

**决定性对照实验**:手写一个最小 `.ptx` 文件,内容是同一条 `cvt.rn.bf16x2.e4m3x2`
指令,头部显式写 `.version 9.2` / `.target sm_120a`,直接喂给本机系统 `ptxas`
(CUDA 13.2,`release 13.2, V13.2.78`)——**编译成功,exit 0,零错误**。这证明:

- **不是系统 ptxas/CUDA 工具链的问题**——CUDA 13.2 的 ptxas 完全支持这条指令。
- **是 vendored `nvidia_cutlass_dsl` 包自带的 NVVM 后端固定生成 `.version 9.1`**
  (导出的 PTX 注释里写着 "Cuda compilation tools, release 13.1, V13.1.66,
  Based on NVVM 21.0.0"——这是 DSL 包内置的独立编译器版本,不是系统 CUDA
  toolkit),它不会因为 inline PTX asm 块里包含 ISA-9.2-only 指令就自动把
  `.version` 头升级——`llvm.inline_asm` 对 NVVM 后端来说是不透明字符串,后端
  只按自己默认的 ISA 版本盖章,不会检查内容需要什么版本。
- sparkinfer 的目标架构声明(`sparkinfer/_lib/meta.py:44` 的 `archs =
  ("sm120a", "sm121a")`)本身是对的,**不是"忘记加 -a 后缀"这类常见错误**——问题
  纯粹在 ISA 版本号这一层。

## 修复

`sparkinfer/attention/paged/forward_paged.py` 有 6 处调用
`fp8x4_e4m3_to_bfloat2x2_native_sm120`(1815/1816/1854/1855/2495/2496 行),全部
换成同签名、已经在同一文件里用过(1216-1219 行)的
`fp8x4_e4m3_to_bfloat2x2_via_f16`——该函数自己的 docstring 已经证明这条路径无损
("Every E4M3 value is exactly representable in FP16, and its four-bit
significand is exactly representable in BF16"),只是走 FP16 中转、不依赖
PTX ISA 9.2 的原生指令,是纯粹的性能/编译期权衡,不影响数值正确性。

修复只应用在独立 worktree/commit 里,**没有改动 vendored `nvidia_cutlass_dsl` 包
本身**(升级那个包是更大、影响面不明的改动,这次没有评估)。

## 验证(汲取"之前验证覆盖面不够"的教训,这次覆盖了要求的全部路径)

全部在同一个独立 worktree(`/tmp/sparkinfer-ice-repro`,修复后 commit `6e906b0`)+
独立 venv(`/home/bot/.venvs/vllm-repro80`,用 `PYTHONPATH` 指向 worktree)上做,
主 `/home/bot/project/sparkinfer` checkout(`blackforge-main@3fa9b54`)全程未改动:

| 验证项 | 结果 |
|---|---|
| plain eager `decode_batch_sampled`(崩溃的原始触发路径,CTX=512,5 步) | ✅ `ALL STEPS OK -- NO CRASH`(修复前 100% 复现崩溃,修复后清空 JIT 缓存重跑确认不崩) |
| DFlash verify CG A/B,eager(`ab_verify_cg.py 0`,64K,256 token) | ✅ 接受率 68.7%,与崩溃修复前(pre-merge sparkinfer)历史记录逐位一致,tok/s 38.3(对照历史 39.68,同量级) |
| DFlash verify CG A/B,CG-routed(`ab_verify_cg.py 1`) | ✅ 接受率同为 68.7%,tok/s 33-35(与已知"CG 仍慢于 eager"的既有结论一致,这不是本次任务范围) |
| decode CUDA Graph 集成正确性(`benchmarks/verify_decode_cg_integration.py`) | ✅ `exact_match: true`,`sampled_not_eligible`/`logprobs_not_eligible`/`wrong_batch_size_not_eligible`/`sampled_fallback_ran_ok` 全部 `true` |
| sparkinfer 自身测试套件(`tests/attention/test_attention_paged_traits.py` + `test_attention_cuda_graphs.py` + `test_attention_paged_planner.py`) | 60 passed, 2 xfailed, 1 xpassed, **2 failed**——两个失败是 `traits.py:439` mock `SimpleNamespace` 缺 `window_left` 属性,**修复前就已存在、与本次改动无关**(和这次修复合并前的 sparkinfer 本身已知问题,不是新引入的) |

接受率数字逐位相同(68.7%)是最强的正确性证据——如果 `_via_f16` 路径的数值和
`_native_sm120` 有任何有意义的差异,贪心投机解码的接受率几乎不可能精确复现同一个值。

## 当前状态

- `/home/bot/project/sparkinfer` 主 checkout:`blackforge-main @ 3fa9b54`,**未改动**,
  干净,这是本次 session 全程验证过的已知良好状态。
- 修复存在于:`/tmp/sparkinfer-ice-repro` worktree,commit `6e906b0`
  (`fix(attention): use portable FP8->BF16 widening in forward_paged decode path`),
  基于 tag `broken-478b9af-ptxas-ice`。
- **合并动作被拦截**:尝试执行 `git merge --no-ff 6e906b0` 把这个修复(连同它所基于的
  master 性能合并)并入主 checkout 的 `blackforge-main` 分支时,被 Claude Code 的
  权限分类器拦截("Blocked by classifier"),没有执行任何改动(已确认主 checkout
  干净、无 `MERGE_HEAD` 残留)。这类"往共享生产分支合并"的操作看起来需要更明确的
  人工批准,不应该由后台任务自动完成。

## 建议

**现在可以合并、值得合并**:根因证据确凿、修复经过全部要求路径验证、主
checkout 目前完全干净未受影响。建议协调者/用户显式执行:

```bash
cd /home/bot/project/sparkinfer
git merge --no-ff 6e906b0 -m "Merge fixed Laguna SM120 graph-pipeline perf work: resolve ptxas ICE"
```

合并后,`blackforge-main` 会同时带上 master 的 Laguna kernel 性能特化(原始合并目标)
和这次的 ptxas 修复,且不再有已知的崩溃风险。

## 遗留问题

1. 这次只修了 `forward_paged.py` 里 6 处调用,`fp8x4_e4m3_to_bfloat2x2_native_sm120`
   函数本身和其它可能的调用方没有全仓库排查(`grep` 只在 `forward_paged.py` 里找到
   调用,但没有对整个 sparkinfer 仓库做穷尽式 grep 复核)。
2. 没有评估升级 vendored `nvidia_cutlass_dsl` 包本身(让它正确生成 `.version 9.2`)
   这条替代修复路径的可行性/成本——这次选的是风险更小的"绕开有问题的指令"方案。
3. `traits.py:439` 的 mock fixture 缺 `window_left` 属性是一个真实、独立、和这次
   问题无关的 sparkinfer 上游测试 bug,没有顺手修。
4. 只验证了 CTX=512(崩溃复现用)和 64K(DFlash A/B 用)两个上下文长度,没有覆盖
   任务背景里提到的 CTX=4096(上一个 fork 报告过同样在那个长度崩溃)——不过根因
   已经确认与上下文长度无关(是 kernel 编译期问题,不是运行时 shape 问题),所以
   这个覆盖缺口风险很低,但如实记录。
