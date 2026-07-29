# vLLM 剥离状态 (2026-07-29)

## 发现

vLLM 剥离工作在 `vllm-removal-phase1` 分支完成，但**从未合并到 main**。

### 分支上的关键提交

| 提交 | 内容 |
|------|------|
| `7e73959` | 任务#42(a): 拆分compat_vllm.py，qwen36专属import不再泄漏进Laguna |
| `e2ba8ec` | 任务#42(b): 去掉set_forward_context对vllm.forward_context的状态写入 |
| `bb9d519` | 任务#45/#46/#47: 自建Laguna config + 彻底删除vllm回退开关 |
| `fd33368` | 任务#48: 64K DFlash+前缀缓存+CUDA Graph性能测试(vllm剥离后版本) |

### 分支上的自建替代

- `runtime/laguna_config.py` — 替代 `EngineArgs.create_engine_config()`
- `runtime/laguna_runtime.py` — 替代 `vllm.forward_context` 状态管理
- compat_vllm.py 从 482 行减到 402 行

### 当前 main 的 vLLM 依赖 (未剥离)

1. **模型构建**: `get_model()`, `EngineArgs`, `VllmConfig` — 整个模型图在 vLLM
2. **前向上下文**: `set_forward_context`, `ForwardContext` — 每层 attention 读取
3. **Attention metadata**: `GDNAttentionMetadata`, `SM120GQAMetadata`
4. **NVFP4 kernel**: `cutlass_scaled_fp4_mm`, `scaled_fp4_quant`
5. **DFlash 加载**: `load_dflash_model`
6. **杂项**: `RMSNorm` isinstance, `FusedTopkBiasRouter`

### 合并建议

分支有 49 个提交、63 个文件、8419 行改动。直接 merge 会与性能优化冲突。
建议：cherry-pick 3 个关键剥离提交，或等性能优化稳定后做完整 merge。

### stash 残留

`stash@{0}: vllm-removal-wip` 包含 laguna_cuda_graph.py 和
laguna_dflash_cudagraph.py 的修改（333行删减），是剥离工作的未完成部分。
