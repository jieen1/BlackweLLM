# Laguna server 默认打开 decode CUDA Graph(2026-07-27)

## 改动

`server/app.py`:
- `SERVER_ENABLE_CUDAGRAPH`:Laguna 默认从 `"0"` 改成 `"1"`。P1 已经把 CUDA Graph
  接入了 `decode_batch_sampled`(commit `9ca7612`)并用真实 HTTP 请求验证过
  (`notes/2026-07-27-p1-http-e2e-and-thinking-strip-bug.md`)——eager decode 已经
  不再是生产路径唯一跑过的配置,继续默认关闭没有理由。`QSR_SERVER_ENABLE_CUDAGRAPH=0`
  / `--no-cudagraph` 仍然可以回滚到 eager。
- `SERVER_NUM_SLOTS`:Laguna 默认从 `1` 提到 `2`。`ServerEngine.__init__` 的准入公式
  `min_slots = capacity + (capacity if enable_cudagraph else 0)`——capacity=1 且
  cudagraph 打开时 min_slots=2,原来的默认值 1 会导致默认配置直接启动失败
  (`ValueError`)。这两个默认值必须一起改,不能只改一个。

## 验证

用**完全默认配置**(只设 `QSR_SERVER_MODEL_BACKEND=laguna`,不显式传任何
`QSR_SERVER_ENABLE_CUDAGRAPH`/`QSR_SERVER_NUM_SLOTS`)启动真实 server:

- 启动正常,`/health` 就绪。
- `POST /v1/completions`(同一 prompt,两轮):第一轮 73.8s(JIT 首次编译热身,
  已知现象,非 bug),第二轮 **1.08s**(CG 稳态速度),两轮输出文本完全一致、正确
  ("Paris. \n\n..."),没有 P1 那次发现的 thinking-strip 空输出问题。
- 请求完成后 `/health` 显示 `active:0, free_slots:1`,槽状态干净,没有泄漏。
- `pytest tests/`:319 passed,3 个既有失败(和这次改动无关,数量、名字都对得上)。

## 结论

Laguna server 现在默认就会用 decode CUDA Graph(仍可用
`QSR_SERVER_ENABLE_CUDAGRAPH=0` 回滚),真实端到端验证通过。
