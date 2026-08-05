# 2026-08-05：Laguna 模型 Codex CLI + Claude Code CLI 端到端测试

Status: **verified**。与 Qwen3.6 对等：同一个 runtime（`server.app` /
Laguna backend），Codex CLI（OpenAI Responses 协议）与 Claude Code CLI
（Anthropic 协议）各自完成一次真实 agentic 任务（通读仓库 → 写文件），
退出码均为 0，无任何代理层，无 vLLM。

## 1. 服务配置（与 Laguna 生产配置一致）

`scripts/blackwellm_ctl.sh start`（端口 8100，PID 见日志
`logs/server_20260805_173334.log`）：

| 配置项 | 值 |
|---|---|
| 槽位 | 3 × 256K（`blocks_per_slot=4096` × `block_size=64`） |
| 投机引擎 | DFlash K=15（draft / verify / decode 三个 CUDA Graph 全部 captured） |
| decode CUDA Graph | 开 |
| prefix cache | 开 |
| KV | FP8 |
| GPU_MEM_UTIL | 0.95（实测显存 92.6/97.9 GiB） |
| 工具解析器 | `poolside_v1` |
| 服务模型名 | `laguna-s-2.1`（别名 `qwen3.6`），`max_model_len=262144` |

## 2. 配置文件（均被 .gitignore 忽略，仅本机工具状态）

`.codex/laguna.config.toml`：

```toml
# Independent Codex profile for the Laguna backend of this repo's runtime
# (server.app / laguna backend on http://127.0.0.1:8100/v1).
model = "laguna-s-2.1"
model_provider = "laguna"
model_context_window = 262144
model_auto_compact_token_limit = 200000
approval_policy = "never"
sandbox_mode = "danger-full-access"

[projects."/home/bot/project/qwen-sm120-runtime"]
trust_level = "trusted"

[model_providers.laguna]
name = "BlackweLLM Laguna-S-2.1 (DFlash + CUDA Graph, 3x256K)"
base_url = "http://127.0.0.1:8100/v1"
wire_api = "responses"
```

调用方式：`CODEX_HOME="$PWD/.codex" codex exec -p laguna "<task>"`。

`.claude/settings.laguna.json`（独立配置，因为默认
`.claude/settings.json` 被 qwen36 占用；通过 `--settings` 显式指定）：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8100",
    "ANTHROPIC_AUTH_TOKEN": "local-runtime",
    "ANTHROPIC_MODEL": "laguna-s-2.1",
    "ANTHROPIC_SMALL_FAST_MODEL": "laguna-s-2.1",
    "DISABLE_TELEMETRY": "1"
  },
  "permissions": {
    "allow": ["Bash", "Edit", "Write", "Read", "Glob", "Grep", "TodoWrite"]
  }
}
```

调用方式：
`claude -p --settings .claude/settings.laguna.json "<task>"`。

## 3. 协议冒烟测试（带工具，3 个端点全过）

- `/v1/chat/completions` + `get_weather`：`finish=tool_calls`，
  `{"city":"北京"}`（约 0.8s）。
- `/v1/messages` + `get_weather`：`stop=tool_use`，标准 tool_use 块。
- `/v1/responses`：`Say exactly: laguna-ok` 输出精确 `laguna-ok`。

## 4. Codex CLI 端到端（Responses 协议）

命令：

```bash
CODEX_HOME="$PWD/.codex" codex exec -p laguna \
  -C /home/bot/project/qwen-sm120-runtime \
  "通读…总结…"
```

结果：

- 退出码 0；服务端 `/v1/responses` 16 次全部 200 OK。
- `/tmp/codex-laguna-hello.txt` = `laguna-codex-ok`（16 字节，精确匹配 ✓）。
- `/tmp/codex-laguna-summary.md` = 31 行。
  ⚠️ 任务要求 ≤30 行，实际 31 行，轻微超标；端到端功能验证本身成立，
  如实记录，未重跑（如需严格对齐可重跑一次）。

## 5. Claude Code CLI 端到端（Anthropic 协议）

命令：

```bash
claude -p --settings .claude/settings.laguna.json \
  "通读…总结…"
```

结果：

- 退出码 0；服务端 `/v1/messages` 13 次 200 OK，Claude Code 执行了
  27 次工具调用的多轮工具循环。
- `/tmp/cc-laguna-hello.txt` = `laguna-cc-ok`（13 字节，精确匹配 ✓）。
- `/tmp/cc-laguna-summary.md` = 28 行（≤30 行 ✓）。

## 6. 服务端健康度（同一 Laguna 进程，Prometheus 指标）

```
blackwellm:requests_completed_total 21
blackwellm:prefix_cache_hits_total 17
blackwellm:prefix_cache_misses_total 4   (命中率 ~81%)
blackwellm:dflash_cg_captured{graph="decode"} 1
blackwellm:dflash_cg_captured{graph="draft"}  1
blackwellm:dflash_cg_captured{graph="verify"} 1
```

DFlash 三个 CUDA Graph 全部 captured（无 eager fallback），prefix cache
在实际 agent 对话中命中 17/21，证明端到端路径完整可用。

## 7. 复现命令

```bash
cd /home/bot/project/qwen-sm120-runtime
scripts/blackwellm_ctl.sh start
CODEX_HOME="$PWD/.codex" codex exec -p laguna -C "$PWD" "<任务>"
claude -p --settings .claude/settings.laguna.json "<任务>"
curl -s http://127.0.0.1:8100/metrics | rg 'requests_completed|prefix_cache|dflash_cg'
```

## 8. 测试后恢复默认工作流

本机默认 agent 工作流指向 qwen36 best（8300）。恢复：

```bash
scripts/blackwellm_ctl.sh stop        # 停 Laguna（8100）
scripts/run_qwen36_quality.sh server start best   # 起 qwen36 best（8300）
```

之后 Codex 默认 profile `blackwellm` 与 `.claude/settings.json` 直接可用。
