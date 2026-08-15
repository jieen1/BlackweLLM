# 2026-08-15：CUDA 13.3 / CUTLASS 4.7 / triton 3.7.1 工具链升级

环境修复、版本升级、DSV4 数值修复，以及 Qwen3.8 128K 端到端观察的记录。

## 1. 背景：编译链实际用的是 CUDA 13.2，不是 13.3

系统里 `/usr/local/cuda` 符号链接指向 **13.3**（`update-alternatives`），
torch 也是 `2.13.0a0+gitcf30153`（CUDA 13.3 编译），但：

- `~/.bashrc` 硬编码 `PATH=/usr/local/cuda-13.2/bin`、`CUDA_HOME=/usr/local/cuda-13.2`
- 因此所有 `make` 编译的 runtime kernel（fp8_w8a8 / iq2_mma16 / laguna_router）
  和 sparkinfer JIT（cutlass-dsl 读 `CUDA_HOME`）都用 **13.2 nvcc** 编译
- venv 里还有独立的 `nvidia-cuda-nvcc==13.2.78` pip 包（13.2.78 版 nvcc）
- 磁盘上 13.2 编译的 `.so` 全部标记 `release 13.2`

### 修复

1. `~/.bashrc` 改为指向 `/usr/local/cuda` 符号链接（= 13.3）：
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   export CUDA_HOME=/usr/local/cuda
   ```
   （新交互终端生效；非交互 shell 需显式 export。）
2. 用 CUDA 13.3 nvcc 重编译全部 Makefile 管理的 kernels，确认 `release 13.3`。
3. 删除 3 个无加载器的 13.2 遗留产物：
   `runtime/kernels/nvfp4_gemm_sm120.so`、
   `runtime/kernels/_generated/{nvfp4_gemm_sm120,nvfp4_w4a4_quant_sm120}.so`
   （git 未跟踪、runtime 无引用、2026-07/08 旧编译）。
4. 清空 sparkinfer 编译缓存（`~/.cache/b12x/compile`，旧产物 `CUDA_HOME=13.2`），
   以 13.3 重建，确认新缓存 `CUDA_HOME=/usr/local/cuda` 且 `.o` 为 `release 13.3`。

**cutlass-dsl 的 nvcc 解析**（`nvidia_cutlass_dsl/dsl_packages/cutlass/base_dsl/env_manager.py`）：
只用 `CUDA_HOME`/`CUDA_PATH` 或 PATH 里的 `nvcc`，**不引用** venv 的
`nvidia-cuda-nvcc` pip 包——所以真正决定 JIT 编译版本的是调用进程的
`CUDA_HOME`，这解释了为什么"torch 是 13.3"但"kernel 是 13.2"。

## 2. 版本升级

| 组件 | 之前 | 之后 | 备注 |
|---|---|---|---|
| CUDA 编译 | 13.2.78 | **13.3** | `.bashrc` + 重编译全部 kernels |
| CUTLASS C++ headers | 4.6.1 (`/home/bot/project/cutlass-4.6.1`) | **4.7.0** | 下载 GitHub v4.7.0，`Makefile CUTLASS_ROOT` 更新 |
| nvidia-cutlass-dsl | 4.6.0 | **4.7.0** | 含全部 `-libs-{base,core,cu12,cu13}` 4.7.0 |
| triton | 3.6.0 | **3.7.1** | 见下节 DSV4 数值修复 |
| torch | 2.13.0a0+gitcf30153 | 不变 | 已是 2.13.0 RELEASE 的预发布版，preflight 认可 |

sparkinfer（b12x fork）侧同步：`pyproject.toml` 钉
`nvidia-cutlass-dsl==4.7.0` 系，`b12x/_lib/gating.py` 的 `MIN_CUTLASS_DSL`
4.6.0 → 4.7.0。

## 3. triton 3.7.1 引入的 DSV4 数值回归与修复

triton 3.7 改变了 `tl.sum` 的 fp32 归约树（线性链 → 平衡树），导致 DSV4
IQ2_XS kernels 的 fp32 累加顺序变化，破坏两个 bit-exact 契约：

### 3a. `iq2xs_dequant_gemm_batch_indexed_dual_swiglu_b1`（gate/up+SwIGLU 融合）

融合 kernel 内嵌 SwiGLU 尾部改变了编译器对 GEMM 累加循环的寄存器分配，
使 gate/up 的 `tl.sum` 归约树与 split 参考路径不同 → gate/up 累加值微差
（如 3034 vs 3040）→ bf16 舍入边界翻转（2/2048 元素）。

**修复**：融合 kernel 不再内嵌 SwiGLU——改为输出 raw fp32 gate/up
（`_iq2xs_dequant_gemm_batch_indexed_dual_b1_kernel_rawfp32`，与参考 dual
kernel 结构逐字一致），外部调 `swiglu_bf16`。这样 fused 路径与 split 参考
**完全同构**，任何 triton 版本下 bit-exact（已验证 diff=0）。

### 3b. `iq2xs_dequant_gemm_batch_indexed`（down GEMM 的 M>1 批量）

原 3D 归约 `tl.sum(valj[None,:,:] * xv[:,None,:], axis=2)`（[M,8,32]）在
triton 3.7 下与 M_PAD==1 的 2D 归约树不同 → M=4 批量与 M=1 单次 bit 不一致。

**修复**：M>1 分支改为 `for mm in tl.static_range(M_PAD)` 逐行 2D 归约
（与 M_PAD==1 分支同构），保证跨 triton 版本一致。代价是 M>1 prefill 时
每个 token 重读 packed weight（3D 路径的权重复用被牺牲），正确性优先。

### 3c. `test_cuda_forward_decode_batch_matches_concatenated_b1_oracle`

该测试对比 M=4 批量 `forward_decode_batch` 与 4 次 M=1 串联。修复 3a/3b 后
route（gate/up/down/reduce）全部 bit 一致，剩余差异**纯在 shared-expert
投影**：cuBLAS fp32 matmul 的 M=1 vs M=4 特化差异 ~1.5e-5，经
w1→swiglu→w2 链放大后在 bf16 舍入边界变成 1 ULP（0.00195）。

**关键确认**：该差异在 triton 3.6 和 3.7 下**完全相同**（
`(s4-s1).abs().max()=6.7e-6` 两版本一致）——**不是 triton 3.7 引入**，是
cuBLAS M 特化的固有差异。测试 atol=1e-6 恰好卡在 bf16 舍入边界上（3.6
通过是边界运气）。

**修复**：测试门禁改为 `atol=1e-6, rtol=1e-2`（容忍 1 个相对 bf16 ULP，
注释说明原因）——不是掩盖，是承认 M=1 vs M=4 的 cuBLAS 固有差异
（测试注释本来就预告 "can move one tiny BF16 output by a single
representable step"）。

## 4. 完整验证

- 全量测试：**2373 passed, 26 skipped**（含 DSV4 363 passed）
- DSV4 专项：363 passed, 5 skipped
- 所有 runtime kernel `.so` 确认 `release 13.3`
- sparkinfer JIT 缓存确认 `CUDA_HOME=/usr/local/cuda`（=13.3）

## 5. Qwen3.8 128K 端到端观察（MTP 接受率）

用 `scripts/run_qwen38_128k_decode_bench.sh`（同一 128K prompt）跑：
`server_perf_grid_qwen38_dynamic_128k_c1/c4_20260815_rerun{,2,3}.json`

### 5a. warm（prefix hit）场景：无回退

| run | 每轮 tokens | decode tok/s |
|---|---|---|
| 基线 20260815（cold） | 3.94 | 103.58 |
| rerun2（warm） | 4.0 | 108.31 |
| rerun3（warm） | 4.0 | 107.28 |

warm decode 107-108 ≥ 基线 103，**MTP 每轮满额 4.0 tokens**。

### 5b. cold 场景：观察到接受率下降（待确认）

| run | 每轮 tokens | decode tok/s |
|---|---|---|
| 基线 20260815（cold） | 3.94 | 103.58 |
| rerun（cold） | 2.82 | 70.75 |

rerun 的 MTP acceptance histogram `[7,31,21,31]`（90 rounds）= 平均 1.84/轮，
明显低于 warm 的 2.89/轮。**cold 长上下文（128K）prefill 后的首个 decode
段的 MTP draft 预测质量下降** → 拒绝率上升 → 每轮产出减少。

**注意**：仅一次 cold 观测（n=1），且 warm 无回退，不能确定是 triton
3.7.1 / 13.3 编译回归还是 MTP 对 cold 内容的固有波动。基线 c4 的
acceptance histogram `[24,532,256,304]` 平均仅 1.75/轮——**基线本身在
c1/c4 间就有接受率波动**，说明这个 128K filler prompt 的 MTP 接受率天然
不稳定。

**待办**：重启服务连续 2-3 次 cold 重测，确认 70 vs 103 是否稳定复现；
若复现，调查 triton 3.7.1 / 13.3 编译的 MTP draft/verify kernel 数值是否
变化。
