# Qwen3.8 最大思考导致简单问题长思考：复现与方案验证

日期：2026-08-19
模型：`unsloth/Qwen3.8-27B-NVFP4`，本地 snapshot
`9c73e2daee1d0fd494ffbd1d8753f2174a953796`
机器：RTX PRO 6000 Blackwell Max-Q，SM120，CUDA 13.4，torch nightly  2.15.0.dev20260815+cu134
服务：本 runtime，plain decode，CUDA Graph 开启，4 slots，`block_size=32`

## 结论

问题真实存在，但要拆成两个问题看：

1. Qwen3.8 的 chat template 默认把 thinking effort 设成 `xhigh`。它只是向
   模型注入较强的思考指令，不是硬 token budget；简单问题也可能反复检查答案
   格式。
2. 本 runtime 之前没有 `ChatCompletionRequest.reasoning_effort` 字段。Pydantic
   默认忽略未知字段，所以客户端传顶层 `reasoning_effort=low/none` 时 HTTP
   返回 200，却仍然使用模板默认 `xhigh`。只有
   `chat_template_kwargs` 能生效。

第一点是模型/模板行为，第二点是 runtime 的确定性兼容 bug。第二点已修复：
顶层 Chat Completions 字段、Responses 的 `reasoning.effort` 都会映射到模板；
显式 `chat_template_kwargs` 优先；`none` 映射为 Qwen 的
`enable_thinking=false`。默认不传参数的行为保持不变。

## 外部证据

- Qwen 官方文档确认 hybrid Qwen3 默认开启 thinking，`enable_thinking=false`
  是硬开关，并区分 `/think`/`/no_think` 软开关；同时给出 thinking mode 的
  `temperature=0.6, top_p=0.95, top_k=20` 建议：
  <https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/quickstart.md>
- Qwen 官方 issue #1887 用简单的 `323/23` 算术复现了 reasoning loop；报告称
  thinking 开启时会反复讨论答案格式，关闭 `enable_thinking` 后正常结束：
  <https://github.com/QwenLM/Qwen3/issues/1887>
- Qwen3.8 仓库 issue #145 记录了 Qwen3.5 系列在参考 reasoning sampling
  参数下进入循环的现象，且运行环境包含 vLLM 与 SGLang：
  <https://github.com/QwenLM/Qwen3.8/issues/145>
- 当前 vLLM 文档明确支持 request-level `reasoning_effort` 到
  `enable_thinking` 的映射、server-level chat-template defaults，以及
  `thinking_token_budget`；这正是本 runtime 之前缺失的对照契约：
  <https://docs.vllm.ai/en/stable/features/reasoning_outputs/>
- 社区对 Qwen3.8-27B 的实测反馈也集中在 xhigh 简单问题耗时很长，以及
  `reasoning_effort` 没有被转发时静默回落 xhigh；社区结果作为旁证，不作为
  模型质量结论：
  <https://www.reddit.com/r/LocalLLaMA/comments/1vpotfv/qwen_38_27b_unusable_long_thinking/>
  · <https://www.reddit.com/r/LocalLLaMA/comments/1vokl82/llamacpp_reasoning_effort_not_forwarded_to_chat/>

## 本地模板确认

对本地 tokenizer 的 `chat_template.jinja` 做 CPU 渲染，prompt 为：

> `What is 2+2? Answer with just the number.`

模板实际行为：

| kwargs | prompt 结果 |
|---|---|
| 不传 | `reasoning_effort` 默认 `xhigh`，注入 xhigh 指令 |
| `{"reasoning_effort":"low"}` | 注入 low 指令 |
| `{"reasoning_effort":"medium"}` | 合法，但模板没有额外 medium 指令 |
| `{"enable_thinking":false}` | 生成 prompt 带闭合 think，完全关闭思考 |

这排除了“runtime 只是解析错 reasoning 输出”的可能：差异在模型收到的
prompt 里已经存在。

## 修复前真实端到端复现

固定 prompt：

```text
You are a concise assistant. Never use internal monologue, reasoning, or
"Analyze the Request." Respond immediately with one sentence without
explanation. Do not use tools unless specifically requested. What is 323/23?
Answer with a number.
```

固定 `max_tokens=2048, temperature=0.6, top_p=0.95, top_k=20`，服务启动参数
相同，仅改变 reasoning 字段。修复前结果：

| 请求 | wall time | completion tokens | 结果 |
|---|---:|---:|---|
| 不传（模板 xhigh） | 7.721 s | 299 | 正确答案，但反复检查格式 |
| 顶层 `reasoning_effort=low` | 8.283 s | 343 | **与 low 无关，仍是 xhigh** |
| `chat_template_kwargs.reasoning_effort=low` | 3.276 s | 131 | 低 effort 生效 |
| `chat_template_kwargs.reasoning_effort=medium` | 3.444 s | 134 | medium 合法，但不保证更短 |
| `enable_thinking=false` | 0.370 s | 11 | 无 reasoning，立即回答 |

顶层 low 返回 200 但与默认 xhigh 一样，是最直接的 runtime bug 证据。

## 修复后同口径 A/B

修复后 fresh server，固定相同 prompt、`seed=0`、
`max_tokens=2048, temperature=0.6, top_p=0.95, top_k=20`：

| 请求 | wall time | prompt tokens | completion tokens | reasoning chars | content |
|---|---:|---:|---:|---:|---|
| 默认 xhigh | 6.575 s | 104 | 238 | 501 | `14.043478260869565` |
| 顶层 `reasoning_effort=low` | 4.105 s | 92 | 159 | 369 | `323/23 ≈ 14.0435` |
| kwargs low | 4.085 s | 92 | 159 | 369 | 与顶层 low 完全一致 |
| 顶层 `reasoning_effort=medium` | 6.349 s | 62 | 249 | 591 | medium 本次反而更长 |
| 顶层 `reasoning_effort=none` | 0.685 s | 64 | 18 | 0 | `14.043478260869565` |
| kwargs `enable_thinking=false` | 0.576 s | 64 | 18 | 0 | 与顶层 none 一致 |

另测 Responses API 的标准 `reasoning.effort=none`：HTTP 200，0 reasoning
token，0.349 s，内容为 `4`。非法 effort 在 GPU 前直接返回 HTTP 400，不再静默
回落。

## 排除的伪方案

仅把 `max_tokens` 从 2048 改成 64 不能解决问题：同一请求耗时 1.819 s，
`finish_reason=length`，completion 64，正文为空，只有未完成的 reasoning。
因此它是截断，不是“思考预算后继续给答案”。

官方 Qwen 的可靠 thinking-budget 方案是两阶段：达到 budget 后注入一段
early-stopping 指令和 `</think>`，再继续生成最终答案；官方 quickstart 明确
指出开源框架需要自行实现这一流程。上面的复现记录的是该能力落地前的状态：
当时 runtime 还没有 token-level `thinking_token_budget`，只能依赖
`reasoning_effort` 或普通 `max_tokens`。vLLM 已提供同类能力，后续实现需要
同时覆盖非流式、流式、reasoning token 统计、取消/超时和多请求 batch。

## 当前建议

这是经过本地实测的优先级：

1. **已落地**：支持顶层 `reasoning_effort`，调用方对简单任务显式使用
   `low`；需要绝对低延迟时使用 `none`/`enable_thinking=false`。
2. 不把 `medium` 当作“比 xhigh 一定更快”；Qwen3.8 当前模板对 medium 没有
   额外短思考指令，本次 A/B 甚至比 low 更长。
3. **已落地（2026-08-19）**：若产品需要“保留 thinking 但硬上限”，使用
   低层 `thinking_token_budget`；它在单次连续生成中强制 `</think>`，不把
   普通 `max_tokens` 冒充预算，也不依赖循环检测。
4. 采样参数按 Qwen 官方建议配置；sampling 主要是质量/稳定性控制，不能替代
   reasoning effort 路由。

## 回归收口（2026-08-19）

请求层修复完成后，先跑原失败集合并逐项处理了依赖重命名和数值测试问题：

- bfdiag 与 SparkInfer 测试统一到当前实际包名 `b12x`；MoE 测试同步到当前
  导出的 `b12x_moe_fp4` 符号。
- GDN batch-vs-single 的断言改为 `torch.testing.assert_close`（该测试验证
  行隔离，不要求不同 batch 形状下逐 bit 相同）。
- DSV4 HCA prefill 保留 `rtol=2e-3`，绝对误差按 BF16 的 `2 * eps` 处理；复核
  的最大差异为 `0.015625`，正好是约 2.0 附近的一个 BF16 ULP。
- 全量回归中额外发现 `runtime/kernels/iq2_mma16.cu` 每个 K-block 间缺少
  shared-memory 屏障，造成 `up` 分支间歇性错误；已补 `__syncthreads()` 并
  重建 SM120 artifact。修复后 E=2/M=32 重复检查 100 次通过。

最终验证：

```text
2516 passed, 8 skipped, 0 failed in 433.99s
```

变更文件的 Ruff 检查、Python 编译检查和 `git diff --check` 均通过；Ruff 使用
`/home/bot/.venvs/torch-nightly/bin/python -m ruff`，因为 shell 中没有全局
`ruff` 命令。

## 低层 token-level 实现与真实 E2E（2026-08-19）

后续实现参考 vLLM 的状态机，但把强制逻辑放在本 runtime 的 sampler/调度器
边界：`ThinkingBudgetState` 跟踪 prompt 与已提交 output 的最新
`<think>`/`</think>` span；普通 decode 在 logits 后强制 token，MTP 在 verify
位置图上强制 token，并在多 token end marker 的中间状态从位置 0 继续。预算
计算包含 prompt 中已打开 think block 后的 token，因此和 vLLM 的 token 语义一致。

本地 Qwen3.8 snapshot、RTX PRO 6000 Blackwell Max-Q（SM120）实测：

- plain decode、`thinking_token_budget=8`：completion 的前 8 个思考区 token
  以 `248069`（`</think>`）结束，随后同一请求继续输出 `4`；没有第二次
  `submit`。
- plain streaming、budget 4：SSE 正确分出 reasoning 与 content；服务端原始
  序列为 `The user asks</think>\n8`。
- MTP `K=3`：draft/sync/verify CUDA Graph 均捕获；budget 8 非流式输出
  `The user asks for a simple arithmetic</think>\n13`，budget 4 流式输出 `5`。
- Anthropic `/v1/messages` 的 `thinking.budget_tokens=4` 和 Responses
  `/v1/responses` 的 `reasoning.budget_tokens=4` 均在各自单次请求中生效；
  `thinking_token_budget=0` 在生成前返回 HTTP 400。

因此该能力是 sampler-level 的真实 token 约束，而不是响应层截断、字符串拼接
或二次请求续生成。
