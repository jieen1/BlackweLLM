# BlackweLLM 全面状态文档 — 2026-07-26 23:35

## 一、当前正在执行的事情

| 任务 | 状态 | 详情 |
|------|------|------|
| vLLM v0.25.0 重编译 | ⏳ 进行中 | `/home/bot/vllm` 已切到 `e12b91b03` + patches，MAX_JOBS=8 编译中 (session 72636) |
| 复现 80.4 tok/s | ⏳ 等编译完 | 编译完后用 `/tmp/bench_m1_66d.py` 跑 64K M=1 decode CG |

**编译完成后立即执行：**
```bash
cd /tmp/qsr-66d && CUDA_VISIBLE_DEVICES=0 USE_LIBUV=0 HF_HUB_OFFLINE=1 \
  FLASHINFER_DISABLE_VERSION_CHECK=1 \
  /home/bot/.venvs/vllm/bin/python /tmp/bench_m1_66d.py
```

---

## 二、核心问题清单

### P0: 速度复现 (80.4 tok/s → 当前 70-74 tok/s)

**根因已定位：**
- 历史 80.4 tok/s 环境 = vLLM v0.25.0 + 本地 patches + 原始编译的 .so + sparkinfer 0a7b143
- 当前环境 = vLLM 0.26.0 + 新编译 .so + sparkinfer 3fa9b54 → 70.8 tok/s
- 差距来源：
  1. **C++ 扩展版本** (贡献 ~6 tok/s): v0.25.0 原始 .so vs v0.26.0 新编译
  2. **vLLM Python 代码** (贡献 ~4 tok/s): v0.25.0+patches vs v0.26.0
  3. **SM80 WMMA kernel 选择** (profiler 确认): cuBLAS 在 SM120 上选了 SM80 kernel，占 GPU 时间 40%

**当前进度：**
- ✅ 已切回 v0.25.0 + patches + sparkinfer 0a7b143
- ✅ 用 v0.26.0 .so 测试得到 74.5 tok/s (证明 Python 代码差异贡献 ~4 tok/s)
- ⏳ 正在重编 v0.25.0 C++ 扩展，验证是否能恢复剩余 ~6 tok/s

### P1: DFlash 接受率

**状态：基本修复，但有残留问题**

| 场景 | 接受率 | 状态 |
|------|--------|------|
| 64K eager | 87% | ✅ |
| 64K draft CG | 86% | ✅ (修复了 binding 地址缓存 bug) |
| 64K cold 多轮 | 84-90% (偶尔 15%) | ⚠️ 残留状态污染 |
| 128K warm r1+ | 15-26% | ❌ full-prefix-hit bug |

**已修复的 bug：**
1. CG binding 地址缓存 → draft 输出错误 (commit 30675d2)
2. Verify-only 接受/拒绝状态机错误 (mtp_accept.py 重写)
3. CG capture impl 泄漏 (unpatch_impls)
4. Draft position offset

**未修复：**
1. Full prefix hit 时 draft KV 未重建 (`laguna_dflash.py:1068` 无条件清空 + `continue_prefill_with_aux` 返回 None)
2. 主模型 SWA ring KV 在 full hit 后未恢复
3. Cold r2 偶尔掉到 15% (retained-state 问题，候选：MoE workspace / draft CG buffer)

### P2: Verify CUDA Graph 被禁用

- 当前 verify 走 eager (qo=16)，因为 sparkinfer 在 qo>1 时 ring cache_seqlens 跨页让捕获的 worklist 失效
- 有 CG 时 37.77 ms/步 vs 无 CG 314 ms/步 → 8 倍差距
- 投机解码当前是**负收益**：关闭 DFlash 80 tok/s，开启只有 42-45 tok/s
- **修好 verify CG 是 DFlash 真正产生加速的前提**

### P3: 显存优化

- `RESERVED_PHYSICAL_SLOTS=1` 白占一个槽 (128K 时 3.0 GiB)
- `_bind_kv_caches` 算完从不调用 `ws.bind_kv()`，dummy fp8 缓存常驻 256-512 MiB
- 动态 KV 分配未实现 (短请求也占满 128K 配额)

---

## 三、已完成的工作

### 本次 session (2026-07-26)

| 提交 | 内容 |
|------|------|
| `4e99b7c` | 适配 vLLM 0.26.0: RMSNorm dispatch 修复 + C++ KV scatter (288→48 kernels, -0.97ms) |
| `8e04775` | 记录 profiler 数据和优化状态文档 |
| `c0f318d` | 记录速度回归根因分析 |
| `cac38ab` | 禁用 verify CG (比 eager 慢 7%) |
| `6ee3d42` | 记录 188 SM split-k 需要重新 sweep |
| `c282c36` | 修正 SM 数量 132→188 |
| `814b049` | P3 显存修复 + verify CG worklist 修复 |
| `e9cf99d` | 修复 verify CG stale worklist |
| `199ac67` | 回收 reserved physical slot + 释放 dummy CG caches |
| `e66d254` | 修复 block_size=16→64 (76个 benchmarks/tests) |
| `c22699c` | 回退 lint 改动恢复 87% DFlash 接受率 |
| `30675d2` | 修复 CG binding 地址缓存: 接受率 19%→86% |

### Profiler 数据 (vLLM 0.26.0, M=1 decode CG, 64K, per step)

```
GPU compute:           13.03 ms  (84%)
CPU fill_buffers:       1.36 ms  (9%)
.item() D2H sync:       0.18 ms  (1%)
Other overhead:         1.01 ms  (6%)
Total wall:            15.58 ms  → 64.2 tok/s

Top GPU kernels:
- sparkinfer MoE:              2.29ms (18%)
- cutlass_80_wmma_bf16:        2.28ms (18%) ← SM80 kernel on SM120!
- cutlass_80_wmma_s1616:       2.17ms (17%) ← SM80 kernel on SM120!
- gemvx (GEMV):               1.99ms (15%)
- sparkinfer attention:        1.59ms (12%)
- Triton fused norm:           0.36ms (3%)
- reshape_and_cache_flash:     0.13ms (1%)
```

### 速度测试汇总

| 环境 | tok/s | ms/step | 日期 |
|------|-------|---------|------|
| 历史最佳 (v0.25.0+patches, 原始.so, sparkinfer 0a7b143) | **80.4** | 12.45 | 07-22 |
| v0.25.0 Python + patches + v0.26.0 .so + sparkinfer 0a7b143 | 74.5 | 13.42 | 07-26 23:21 |
| 66d5913 + stock vLLM 0.25.1 + sparkinfer 3fa9b54 | 73.7 | 13.57 | 07-26 |
| main + vLLM 0.26.0 + sparkinfer 3fa9b54 (修复后) | 70.8 | 14.13 | 07-26 22:47 |
| main + vLLM 0.26.0 + sparkinfer 3fa9b54 (修复前) | 64.8 | 15.4 | 07-26 |

---

## 四、未完成 / 待做

### 紧急 (本次编译完成后)

1. **跑 benchmark 验证 v0.25.0 重编后速度** → 确认能否复现 80.4
2. **完整记录复现步骤** → 确保任何人可以重复
3. **恢复环境到 v0.26.0** (如果复现完成)
   ```bash
   cd /home/bot/vllm && git checkout v0.26.0
   cp /home/bot/vllm_so_backup_v026/*.so vllm/
   cp /home/bot/vllm_so_backup_v026/_vllm_fa*.so vllm/vllm_flash_attn/
   cd /home/bot/project/sparkinfer && git checkout blackforge-main
   ```

### 短期 (性能优化)

4. **GEMM kernel 选择修复** — SM80 WMMA → SM120 原生 kernel (潜在 +2ms/step → +15% tok/s)
   - 自研 NVFP4 GEMM patch 已验证生效，但 vLLM 0.26.0 的 kernel 选择优先级变了
   - 需要 patch `FlashInferCutlassNvFp4LinearKernel` 或强制选择 `CutlassNvFp4LinearKernel`
5. **GEMM autotune for Laguna shapes** — M=1 decode 形状针对性优化
6. **Triton fused norm 端到端性能验证** — 当前 M=1 中性 (0.36ms vs 0.20ms C++)，M>1 应有优势
7. **CPU overhead 削减** — fill_buffers 1.36ms (SWA ring 计算、page table 更新)

### 中期 (DFlash 完善)

8. **Verify CG 修复** — 解决 qo>1 时 ring worklist 失效问题 (P0 收益点)
9. **Full prefix hit draft KV 重建** — `laguna_dflash.py:1068` + `laguna.py:1593`
10. **Cold 多轮 retained-state 排查** — MoE workspace / CG buffer 状态泄漏

### 长期 (全面超越 vLLM)

11. **全面性能对比测试**: 64K/128K/200K × prefix-cache+CG × 3轮 × 同 prompt
12. **动态 KV 分配** — 全局池按需分配，短请求不占满 128K 配额
13. **sparkinfer upstream 优化同步** — 3 个修复已在 blackforge-main (attention race, MoE deterministic, bincount→scatter_add)
14. **去 vLLM 依赖** — compat_vllm.py 收口，逐步替换

---

## 五、环境清单

### 仓库

| 路径 | 当前状态 | 用途 |
|------|----------|------|
| `/home/bot/project/qwen-sm120-runtime` | main @ `8e04775` | 主仓库 |
| `/home/bot/vllm` | **v0.25.0 (e12b91b03) + patches** (正在重编) | 生产 vLLM (editable) |
| `/home/bot/vllm-0251` | v0.25.1 (已编译) | 备用 stock 0.25.1 |
| `/home/bot/vllm-025` | v0.25.1 tag (有 sm120_gqa.py) | 参考 patched fork |
| `/home/bot/project/sparkinfer` | **0a7b143** (detached HEAD) | sparkinfer |
| `/tmp/qsr-66d` | commit 66d5913 worktree | 历史最佳代码 |
| `/home/bot/pytorch-build` | PyTorch 2.13.0a0 (editable) | PyTorch |

### Venvs

| Venv | PyTorch | vLLM | 用途 |
|------|---------|------|------|
| `/home/bot/.venvs/vllm` | 2.13.0 | editable → /home/bot/vllm | 生产 |
| `/home/bot/.venvs/vllm025` | 2.13.0 | 0.25.1 → /home/bot/vllm-0251 | 备用 |

### 关键备份

| 路径 | 内容 |
|------|------|
| `/home/bot/vllm_so_backup_v026/` | v0.26.0 编译的 6 个 .so 文件 (1.4GB) |
| `/home/bot/vllm_so_backup_v026/inplace/` | 从 vllm/ 移出的额外 .so |
| `notes/2026-07-22-vllm-fork-diff.patch` | 原始 5 个 patch 的完整 diff |
| `notes/2026-07-22-vllm-fork-archive.md` | vLLM fork 完整存档文档 |

### 硬件常数 (实测)

| 参数 | README 旧值 | 实测值 |
|------|-------------|--------|
| SM 数量 | 132 | **188** |
| L2 缓存 | — | 128 MiB |
| SMEM/block | — | 99 KiB |
| GPU | RTX PRO 6000 Blackwell 96GB | compute capability 12.0 |

---

## 六、关键脚本

| 脚本 | 用途 | 命令 |
|------|------|------|
| `/tmp/bench_m1_66d.py` | M=1 decode CG 64K (指向 /tmp/qsr-66d) | 见上方复现命令 |
| `/tmp/bench_m1_decode.py` | 同上但指向 main repo | 同上但换 sys.path |
| `benchmarks/full_comparison_ours.py` | 完整对比 (prefix+CG+DFlash) | `python -m benchmarks.full_comparison_ours [ctx]` |
| `benchmarks/full_comparison_vllm.py` | vLLM 基线对比 | `python -m benchmarks.full_comparison_vllm [ctx]` |
| `/tmp/test_acceptance_repro.py` | DFlash 接受率测试 | `python /tmp/test_acceptance_repro.py 65536 256` |

---

## 七、sparkinfer 版本说明

| Commit | 分支 | 内容 | 状态 |
|--------|------|------|------|
| `0a7b143` | detached HEAD (当前) | Fix decode graph capacity for windowed attn | 用于复现 80.4 |
| `d2d8cb9` | blackforge-main | 修复 attention paged extend K/V 双缓冲竞争 | 正确性修复 |
| `989723d` | blackforge-main | MoE 确定性物理行分配 | 正确性修复 |
| `3fa9b54` | blackforge-main (最新) | bincount→scatter_add (CG 兼容) | CG 兼容性 |

**复现完成后必须恢复：**
```bash
cd /home/bot/project/sparkinfer && git checkout blackforge-main
```

---

## 八、测试状态

- CPU 测试: **315 passed, 5 failed** (test_cudagraph_buffers.py 检查已移除的旧 FlashInfer 接口)
- DFlash 接受率: 64K cold 84-90% (偶尔波动到 15%)
- E2E 正确性: ✅ "The capital of France is" → " Paris"
- MoE 确定性: ✅ (SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=1)

---

## 九、决策记录

1. **不降级 PyTorch** — 保持 2.13.0，即使原始 80.4 可能用了不同 PyTorch 编译的 .so
2. **适配 v0.26.0 为主** — 复现 80.4 是为了定位差距，最终目标是在 0.26.0 上达到/超越
3. **sparkinfer 用 blackforge-main** — 包含 3 个正确性修复，生产环境必须用
4. **DFlash verify CG 暂停** — eager verify + draft CG 是当前最佳组合
5. **自研 runtime 不依赖 vLLM 推理路径** — 只用 vLLM 做 config/model loading，推理全走自研
