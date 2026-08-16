# Nightly 工具链（torch 2.15.0.dev+cu134 / triton 3.8.0 / py3.14）性能与数值对比验证（2026-08-16）

状态：🟢 **有效**

## 背景与结论一句话

把 runtime 迁移到新 venv（`~/.venvs/torch-nightly`：torch 2.15.0.dev20260815+cu134、
triton 3.8.0+gitdf3f91dd、Python 3.14.4）后，用 `scripts/run_qwen38_128k_decode_bench.sh`
（Qwen3.8-27B-NVFP4、131072 上下文、MTP K=3、CUDA Graph、FP8 KV、elastic 4160-bundle
pool、capacity=4）与 2026-08-15 参考数字对比。**结论：没有性能回归，也没有可修的代码
问题；"掉速"与"prefill +16%"均为机器 run-to-run 漂移的测量伪差。** 新工具链引入的
唯一实质变化是 kernel 级数值漂移（1-ULP 类，见 §4），不影响已过质量门禁。

## 1. 环境

| | 参考（08-15） | nightly |
|---|---|---|
| venv | `~/.venvs/vllm` | `~/.venvs/torch-nightly` |
| torch | 2.13.0a0+gitcf30153（cu133） | 2.15.0.dev20260815+cu134 |
| triton | 3.7.1（5d6048aa） | 3.8.0+gitdf3f91dd |
| python | 3.12 | 3.14.4 |
| b12x | 1.0.1/1.1.0 | 1.2.3（master @375cca44） |
| nvcc（自研 .so） | 13.3.73 | 13.3.73（同，5 个产物全部重建，manifest 已记录） |

依赖补齐：`blackwellm[dev,serving]`（跳过 `cuda` extra 以免 torch==2.13.0 pin 降级，
其非 torch 依赖手动装）、b12x/sparkinfer editable、fla 0.5.2、tilelang 0.1.12、aiohttp。
preflight 版本契约已扩展为接受 {2.13.0, 2.15.0} 两个已验证环境（`VERIFIED_TORCH_VERSIONS`）。

## 2. 性能对比（同脚本、同配置、同模型快照）

| 指标 | 参考 08-15 | nightly 今天 | 旧环境今天 | 参考代码今天 |
|---|---|---|---|---|
| c1 cold decode tok/s | 101.09 | 97.45 | 97.30 | 98.79 |
| c1 cold TTFT | 62.3 s | 74.42 s（伪差） | 63.94 s | 63.57 s |
| c4 warm decode tok/s | 66.32 / 66.33 | 64.50 / 64.93 | 64.58 / 63.81 | 65.02 / 64.42 |

### 2.1 机器漂移是主变量（关键证据）

同一台机器、同一服务器进程（参考代码 a1c0c89）、连续三组 c4：

    65.02 → 62.83 → 63.04（5 分钟内，零代码/环境变化）

同服务器 run-to-run 漂移 ≈3%，**大于**所有"代码差异"（~1%）与"工具链差异"（~0%）。
四种 代码×环境 组合的交错测量结果相互覆盖，测量时刻决定了谁快。08-15 参考数字落在
今天漂移带的快端。**教训：任何性能对比必须同服务器多次重复取中位数，单次测量不可判。**

### 2.2 prefill "+16%" 是伪差

nightly 首次冷 prefill 74.4s 是一次性慢点。干净背对背（前缀缓存关闭、各测 2 次）：

| | 第 1 次 | 第 2 次 |
|---|---|---|
| nightly | 57.2 s | 59.3 s |
| 旧环境 | 58.2 s | 59.0 s |

无差异。nsys kernel 构成也一致（attn 63.5% / W8A8 11% / W4A4 8.5% / torch ops ~9%）。

### 2.3 今天代码（08-16）无回归

runtime 的 VMM 提交（c5a42db/49d8eb7）全部 gated 在 `extensible_kv=True`（bench 未开），
热路径未触碰；sparkinfer rescale-skip（8f74740）自述 bit-exact 且性能中性。代码审查 +
实测（参考代码 a1c0c89+6ea48ad vs 今天代码）均落在漂移带内。

## 3. 测试套件

nightly 全量：2455 passed / 10 failed / 8 skipped。10 个失败中 9 个是
`sparkinfer`→`b12x` 改名后的测试引用缺口（旧 venv 残留旧名 editable 包故通过/跳过），
与工具链无关；1 个是真数值漂移（见 §4）。

## 4. 工具链数值漂移（真实存在，1-ULP 类，质量无影响）

| 项目 | 漂移量 | 说明 |
|---|---|---|
| DSV4 blk.3 ratio-128 prefill | o0 max diff 0.03125 | **参考模型（纯 torch 数学）在两 torch 版本间也漂 0.03125**——cuBLAS 13.3→13.4 归约顺序变化，非 triton 问题。kv cache 漂 2⁻¹⁰ |
| `fused_add_rms_norm` | 2⁻⁷ | triton 3.7.1 vs 3.8.0 kernel-vs-kernel。已排除 LLVM SLP（对 sm120 打 sm90 式 disable 补丁后输出位等不变，补丁已回滚）；ttgir 两版一致，差异在更下层 codegen |
| `rms_norm` / `rms_norm_tail` | **位等** | 不受影响 |
| `nvfp4_quant`（bit-exact gate） | **位等** | tests/test_nvfp4_quant_triton.py 5/5 通过 |

含义：DSV4 测试容差（2e-3）是按旧工具链校准的，nightly 下需重校或 DSV4 工作留在旧
venv；Qwen3.6/3.8/Laguna 生产路径不受影响（bench 100% MTP 接受率、质量维度全绿）。

## 5. 工具链侧可用的优化盘点（2026-08-16 时点）

| 项 | 状态 |
|---|---|
| triton 3.8 sm_120a 目标名修复 / ptxas-blackwell 13.3 / reduce_forward 提速 / 原子合并 / membar 消除 | 已在用（nightly kernel 全新编译，构成与 3.7.1 一致） |
| torch 2.15 新功能（cuDNN varlen paged KV、FA4 mixed head dims、scaled_mm 修复） | 不适用（attention/GEMM 全走自研 b12x/自研 .so，torch 侧只有 elementwise/cat/conv ~9%） |
| packed arithmetic（ttng.packed_arith，Gluon API） | 记入 Qwen3.8 路线图（tl.* 语言面还没有） |
| 真正的大杠杆 | 仍是 roadmap 的有效 M / 字节削减 / 融合（104→108 tok/s 路径已验证） |

## 6. 复现命令

```bash
# 服务器（nightly）：
QSR_BENCH_PYTHON=/home/bot/.venvs/torch-nightly/bin/python \
  scripts/run_qwen38_128k_decode_bench.sh server
# 基准 cell：
QSR_BENCH_PYTHON=/home/bot/.venvs/torch-nightly/bin/python \
  scripts/run_qwen38_128k_decode_bench.sh c1   # 及 c4
# 注意：curl 检查端口会走 127.0.0.1:7890 代理（~/.curlrc），须 --noproxy '*'
# 注意：pkill -f "server.app" 会匹配自身命令行，用 ps+grep+awk 按 PID 杀
```

fixtures：`benchmarks/fixtures/server_perf_grid_qwen38_dynamic_128k_c{1,4}_20260816_*.json`
（nightly 基线），nsys 报告在 `/tmp/opencode/nsys_{nightly,oldenv}.nsys-rep`。
