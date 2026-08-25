# Qwen3.8 ModelOpt W4A4 GDN verify batching (2026-08-23)

状态：🟢 **已进入默认 `auto` 路径；Unsloth 路径保持原有选择。**

## 结论

Gittensor 的 Qwen3.8 ModelOpt W4A4 checkpoint 在 DSpark verify 中，GDN 的
`in_proj_qkvz` 和 `out_proj` 都是原生 block-scaled W4A4。旧的逐候选位置循环
会把每个投影拆成多次 M=1 block-scaled GEMM；这不是该格式的合理执行形态。

`Qwen36GatedDeltaNet.spec_forward()` 现在在 `auto` 模式下识别这两个
`ModelOptNVFP4W4A4Linear`，把整个 verify window 作为一个矩阵执行（K=7 时
有效 M=8），同时保留 `QSR_QWEN36_GDN_BATCH_LARGE_PROJECTIONS=0` 的精确
旧路径回滚开关。Unsloth 的 mixed compressed-tensors FP8 模块不满足
ModelOpt W4A4 类型条件；它仍只沿用原先的 raw-FP8 contract 选择。

## 同口径实测

RTX PRO 6000 Blackwell SM120，torch nightly，Qwen3.8 Gittensor ModelOpt W4A4，
DSpark K=7，CUDA Graph，prefix cache，FP8 KV，128K prompt，c=1，256 output
tokens，same completion fixture。先用 `bf diff` 校验两个 run record，结果为
`comparable=true`，唯一配置差异是 `extra.gdn_batch_large_projections`。

| 路径 | warm decode | warm E2E | DSpark accepted/committed | completion SHA |
|---|---:|---:|---:|---|
| batch-large=0 | 96.90 tok/s | 93.75 tok/s | 226/255 | `75b43a8a…f306` |
| batch-large=auto/1 | 160.32 tok/s | 151.11 tok/s | 226/255 | `75b43a8a…f306` |

相对旧路径，decode **+65.4%**，请求级 warm E2E **+61.2%**；DSpark 接受
统计和 completion SHA 没有变化。服务只在隔离端口 `18424` 启动，验证完成后
已停止，未触碰现有服务。

## 代码位置与验证产物

- 选择器和三个 verify call site：`runtime/model/qwen36_model.py`
- CPU 回归：`tests/test_qwen36_gdn_spec_rollback.py`
- A/B run records：`bf diff 4f063e591a41 12a1f47d0535`
- 原始 JSON/trace：`/tmp/qwen38_gdn_ab/`

其它已审计候选（context-KV graph fusion、ragged tier、RMS/FP8 fusion）在
`notes/2026-08-19-qwen38-dspark-optimization-stop.md` 中已有同口径负结果；
当前没有第二个同等高收益且已过质量门禁的候选。
