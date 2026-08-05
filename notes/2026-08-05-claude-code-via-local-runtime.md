# 2026-08-05：Claude Code CLI 接入本地 runtime（Anthropic 协议直连）

Status: **verified**。Claude Code 2.1.221 通过项目级配置直连本地 best 服务
（端口 8300，qwen36 backend，MTP K=3 + CUDA Graph + prefix cache + 3×256K），
完成一次真实 agentic 任务（读仓库 → 写文件），退出码 0，无任何代理层。

## 1. 配置（无任何代理，直接转发到 runtime 的 /v1/messages）

项目级 `.claude/settings.json`（本仓库内）：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8300",
    "ANTHROPIC_AUTH_TOKEN": "local-runtime",
    "ANTHROPIC_MODEL": "qwen3.6",
    "ANTHROPIC_SMALL_FAST_MODEL": "qwen3.6",
    "DISABLE_TELEMETRY": "1"
  },
  "permissions": { "allow": ["Bash", "Edit", "Write", "Read", "Glob", "Grep", "TodoWrite"] }
}
```

另外在 `~/.claude.json` 的
`projects["/home/bot/project/qwen-sm120-runtime"].hasTrustDialogAccepted` 置
`true`（否则 Claude Code 忽略 settings.json 里的 permissions.allow，提示
"this workspace has not been trusted"）。两者都完成之后，headless 运行**无需**
`--dangerously-skip-permissions`。

Claude Code 走的端点是：

- `POST /v1/messages?beta=true`（27 个工具、stream、max_tokens=32000）
- `POST /v1/messages/count_tokens?beta=true`

runtime 侧零改动：`server/formats/anthropic.py` 已支持 system/文本块、
tool_use/tool_result 多轮、Claude Code 的 billing header 剥离、server-side
工具跳过；`server/app.py` 的 SSE 已输出标准 `message_start → content_block_*
→ message_delta → message_stop`；`QSR_TOOL_CALL_PARSER=qwen3_coder` 负责把
模型输出解析成 `<tool_call>` → Anthropic tool_use 块。

## 2. 协议冒烟测试（带工具）

`/v1/messages` 流式 + `get_weather` 工具：3/3 次返回标准 tool_use
（`content_block_start` + `input_json_delta` + `content_block_stop`），
`stop_reason=tool_use`。非流式路径同样正确。

## 3. Claude Code 端到端任务

首次（trust 未设置，带 `--dangerously-skip-permissions`）：

```bash
cd /home/bot/project/qwen-sm120-runtime
claude -p --dangerously-skip-permissions \
  "通读仓库代码并总结，至少读 README.md、docs/architecture.md、
   runtime/backends/qwen36.py 的入口与状态机；把不超过30行的中文总结写到
   /tmp/claude-e2e-summary.md；创建 /tmp/claude-e2e-hello.txt，内容恰好是
   runtime-e2e-ok；完成后报告读了哪些文件"
```

结果：退出码 0；`/tmp/claude-e2e-summary.md` 20 行（≤30 行 ✓）；
`/tmp/claude-e2e-hello.txt` 恰好 `runtime-e2e-ok`（14 字节 ✓）。Claude Code
实际执行了多轮工具循环：服务端日志 `msgs=2 → 4 → 6 → 9 → 11`，全部
`POST /v1/messages?beta=true 200 OK`。

二次（trust 已设置，**无任何 flag**）：创建 `/tmp/claude-e2e-trusted.txt`，
内容恰好 `trusted-ok`（10 字节 ✓），退出码 0。证明纯配置即可自主运行。

## 4. 服务端健康度（同一 best 进程）

- `mtp_verify_graph_replays=1536`：MTP verify CUDA Graph 在真实 Anthropic
  服务路径持续回放。
- prefix cache：完整命中只在 block_size=16 边界生效。16-token 对齐 prompt
  连续 3 次，run 2/3 `cache_read_input_tokens=16` 完整命中，
  `prefix_persistent_restores=2`（修复后的 persistent 路径正常）；
  Claude Code 的对话每轮增长、多数请求不在边界上，因此统计上以 miss 为主，
  属预期行为，不影响正确性。
- 冒烟测试注意：短 `max_tokens`（如 32）可能全部被思考 token 消耗，可见文本
  为空；真实 Claude Code 任务 max_tokens=32000，不受影响。

## 5. 运行方式备忘

```bash
cd /home/bot/project/qwen-sm120-runtime
claude -p "<任务>"          # 依赖项目 .claude/settings.json + workspace trust
claude                       # 交互式同配置
```
