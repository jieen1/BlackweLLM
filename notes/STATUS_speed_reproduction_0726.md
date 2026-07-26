# Speed Reproduction Status — 2026-07-26 23:25

## 目标

复现历史最佳 80.4 tok/s (64K M=1 decode CG)，并完整记录所有步骤确保可重复。

## 历史最佳环境 (2026-07-22 记录)

| 组件 | 版本/状态 |
|------|-----------|
| 代码 | commit `66d5913` (worktree: `/tmp/qsr-66d`) |
| vLLM | v0.25.0 (commit `e12b91b03`) + 5个本地patch + sm120_gqa.py |
| vLLM 路径 | `/home/bot/vllm` (editable install) |
| sparkinfer | `0a7b143` (Fix decode graph capacity underestimation) |
| PyTorch | 2.13.0a0+gitcf30153 (editable: `/home/bot/pytorch-build`) |
| venv | `/home/bot/.venvs/vllm` |
| 结果 | **80.4 tok/s, 12.45 ms/step** |

## 当前复现进度

### 已完成

1. ✅ `/home/bot/vllm` 切回 v0.25.0 (`e12b91b03`) + `git stash apply stash@{0}` (local-patches-v0.25)
2. ✅ `sm120_gqa.py` 确认存在 (untracked, 60152 bytes)
3. ✅ sparkinfer 切回 `0a7b143`
4. ✅ 使用 v0.25.0 Python + v0.26.0 编译的 .so 测试 → **74.5 tok/s** (差距来自C++扩展)
5. ✅ 备份 v0.26.0 .so 到 `/home/bot/vllm_so_backup_v026/`
6. ⏳ 正在重新编译 vLLM v0.25.0 C++ 扩展 (MAX_JOBS=8, 开始于 23:23)

### 测试结果汇总

| 环境 | tok/s | ms/step | 备注 |
|------|-------|---------|------|
| 历史最佳 (v0.25.0+patches, 原始.so) | **80.4** | 12.45 | 2026-07-22 记录 |
| v0.25.0 Python + patches + v0.26.0 .so | 74.5 | 13.42 | 刚测 (23:21) |
| main + vLLM 0.26.0 + sparkinfer 3fa9b54 | 70.8 | 14.13 | 今天早些时候 |
| 66d5913 + stock vLLM 0.25.1 + sparkinfer 3fa9b54 | 73.7 | 13.57 | 今天早些时候 |
| main + vLLM 0.26.0 修复前 | 64.8 | 15.4 | 基线 |

### 差距分析

- 80.4 → 74.5 (差 5.9 tok/s): C++ 扩展版本不同 (v0.25.0 原始 vs v0.26.0 编译)
- 74.5 → 70.8 (差 3.7 tok/s): vLLM Python 代码版本 (v0.25.0+patches vs v0.26.0)
- 正在重编 v0.25.0 C++ 扩展以验证是否能恢复剩余 5.9 tok/s

## 复现命令

### Benchmark 脚本

`/tmp/bench_m1_66d.py` — M=1 decode CG, 64K context, 128 tokens × 3 rounds

```bash
cd /tmp/qsr-66d && \
CUDA_VISIBLE_DEVICES=0 \
USE_LIBUV=0 \
HF_HUB_OFFLINE=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
/home/bot/.venvs/vllm/bin/python /tmp/bench_m1_66d.py
```

### 环境切换命令

```bash
# vLLM → v0.25.0 + patches
cd /home/bot/vllm
git checkout e12b91b03
git stash apply stash@{0}  # local-patches-v0.25

# sparkinfer → 0a7b143
cd /home/bot/project/sparkinfer
git checkout 0a7b143

# 恢复 vLLM 0.26.0 (之后用)
cd /home/bot/vllm
git checkout v0.26.0
cp /home/bot/vllm_so_backup_v026/*.so vllm/
cp /home/bot/vllm_so_backup_v026/_vllm_fa*.so vllm/vllm_flash_attn/

# sparkinfer → latest
cd /home/bot/project/sparkinfer
git checkout blackforge-main
```

### 编译命令 (当前正在执行)

```bash
cd /home/bot/vllm
PATH="/home/bot/.venvs/vllm/bin:$PATH" MAX_JOBS=8 \
  /home/bot/.venvs/vllm/bin/python -m pip install -e . --no-build-isolation
```

## Profiler 数据 (vLLM 0.26.0, M=1 decode CG, 4K context, 20 steps)

```
Kernel                                              Time(ms)  Calls  %
─────────────────────────────────────────────────────────────────────
sparkinfer MoE (cutlass_kernel)                       43.7     940   20.9%
cutlass_80_wmma_tensorop_bf16 (SM80!)                 43.5     960   20.8%
cutlass_80_wmma_tensorop_s1616 (SM80!)                40.7     980   19.4%
gemvx (GEMV)                                         27.7    3800   13.3%
gemvx (variant)                                       8.6     980    4.1%
sparkinfer attention paged forward                    6.9     960    3.3%
topkGating                                            4.6     940    2.2%
_fused_add_rms_norm_kernel                            3.9    1920    1.9%
rotary_embedding                                      3.5     960    1.7%
elementwise (various)                                 3.2    1900    1.5%
_rms_norm_kernel                                      2.7    1940    1.3%
sparkinfer attention mergePartial                     2.5     960    1.2%
reshape_and_cache_flash                               2.4     960    1.2%
─────────────────────────────────────────────────────────────────────
Total CUDA                                           209.4   20 steps
Per step                                             10.47 ms
```

**关键发现**: `cutlass_80_wmma` (SM80 Ampere kernel) 占 GPU 时间 40%。
这是 cuBLAS 在 SM120 硬件上选择了 SM80 的 WMMA kernel，而非 SM120 原生 kernel。
自定义 NVFP4 GEMM patch 已验证生效（patch 到 `cutlass_scaled_fp4_mm`），
但 vLLM 0.26.0 的 kernel 选择优先级变了（FlashInferCuteDsl > FlashInferCutlass > Cutlass），
可能导致实际走了不同的代码路径。

## 待确认

1. 重编 v0.25.0 后能否恢复 80.4 tok/s
2. 如果不能，差距来自 PyTorch 版本变化（原始测试可能用了不同 PyTorch 编译的 .so）
3. SM80 WMMA kernel 选择问题是否是 v0.25.0 和 v0.26.0 共有的

## 文件位置

| 文件 | 用途 |
|------|------|
| `/tmp/bench_m1_66d.py` | M=1 decode CG benchmark (指向 /tmp/qsr-66d) |
| `/tmp/bench_m1_decode.py` | 同上但指向 main repo |
| `/tmp/qsr-66d/` | commit 66d5913 worktree |
| `/home/bot/vllm_so_backup_v026/` | v0.26.0 编译的 .so 备份 |
| `/home/bot/vllm-0251/` | stock vLLM 0.25.1 (已编译) |
| `/home/bot/vllm-025/` | patched fork (v0.25.1 tag, 有 sm120_gqa.py) |
| `notes/STATUS_speed_optimization_0726.md` | 优化分析文档 |
| `notes/2026-07-22-vllm-fork-archive.md` | vLLM fork 完整存档 |
| `notes/2026-07-22-vllm-fork-diff.patch` | 原始 patch 文件 |
