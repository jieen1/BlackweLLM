# Qwen3.8 Flash-Next 本地启动与 OpenCode 运维手册

> 状态：当前本机可复现配置（2026-09-02）
>
> 适用模型：`Qwen3.8-Flash-Next-NVFP4-RadixArk`（`qwen4_exp`）
>
> 适用硬件：单张 NVIDIA Blackwell SM120，约 96 GiB 显存

这份手册是当前 Flash-Next 服务的启动事实来源。仓库里的
[`scripts/blackwellm_ctl.sh`](../scripts/blackwellm_ctl.sh) 是历史 Laguna
控制脚本，默认端口为 `8100`，不会启动下面这套 Flash-Next 配置；不要把它
当作当前应用的启动入口。

## 当前验证过的服务形态

| 项目 | 当前值 |
| --- | --- |
| Python | `/home/bot/.venvs/torch-nightly/bin/python`（3.14 nightly） |
| checkpoint | `/home/bot/models/Qwen3.8-Flash-Next-NVFP4-RadixArk` |
| backend | `flashnext` |
| 监听 | `127.0.0.1:8300` |
| 并发/物理 slot | `2 / 2`（FP8 QSA 冷启动 + 两路并发 smoke 已验证） |
| KV block | `128 tokens × 2048 blocks/slot = 262144 tokens`（256K ceiling） |
| KV 模式 | `legacy`（当前实测稳定配置） |
| MTP | 开启，`K=3` |
| CUDA Graph | 开启 |
| persistent prefix cache | 开启 |
| 视觉输入 | 开启（默认 1 MP 后处理面积上限） |
| 两槽冷启动显存 | `explicit=85.11 GiB`，`torch_reserved=86.76 GiB`，driver free 约 `6.05 GiB` |

256K 是每个 slot 的容量上限，不代表模型加载时立即为每个 token 分配显存。
当前 `capacity=2` / `num_slots=2` 已经过本机 FP8 冷启动、CUDA Graph 捕获和两路
并发请求门禁；启动后余量约 6.05 GiB。此前三槽冷启动虽能通过，但余量只有约
1.9 GiB，当前运行配置有意降为两槽。不要在不重新冷启动的情况下提高并发、
上下文或 PLE 配额。

## 启动

先确认没有占用 `8300` 的旧 Flash-Next 进程。下面的环境变量是当前运行服务
实际使用的完整 profile；保持 `QSR_QWEN_KV_MODE=legacy`，不要为了“省显存”
擅自改权重精度或切换到未经验证的 KV 配置。

```bash
cd /home/bot/project/qwen-sm120-runtime

export HF_HUB_OFFLINE=1
export QSR_SERVER_MODEL_PATH=/home/bot/models/Qwen3.8-Flash-Next-NVFP4-RadixArk
export QSR_SERVER_BACKEND=flashnext
export QSR_SERVED_MODEL_NAME="qwen3.8 qwen3.8-flash-next"
export QSR_SERVER_PRODUCTION=1
export QSR_SERVER_CAPACITY=2
export QSR_SERVER_NUM_SLOTS=2
export QSR_SERVER_BLOCK_SIZE=128
export QSR_SERVER_BLOCKS_PER_SLOT=2048
export QSR_SERVER_ENABLE_CUDAGRAPH=1
export QSR_SERVER_ENABLE_PREFIX_CACHE=1
export QSR_SERVER_ENABLE_MTP=1
export QSR_SERVER_MTP_K=3
export QSR_QWEN_KV_MODE=legacy
# Flash-Next QSA main-attention K/V: row-scaled FP8 E4M3.  BF16 is only for
# explicit reference A/B runs; QSR_QWEN_KV_MODE does not select this dtype.
export QSR_FLASHNEXT_QSA_KV_DTYPE=fp8_e4m3
export QSR_SERVER_GPU_MEM_UTIL=0.90
export QSR_DISABLE_GC=1

# Flash-Next 的已验证批量/图/PLE 配置
export QSR_FLASHNEXT_BATCH_GDN_RECURRENCE=1
export QSR_FLASHNEXT_BATCH_LM_HEAD=1
export QSR_FLASHNEXT_BATCH_GDN_PROJECTIONS=1
export QSR_FLASHNEXT_MTP_CONTINUATION_GRAPH=1
export QSR_FLASHNEXT_MTP_SPARSE_GRAPH=1
export QSR_FLASHNEXT_PLE_CACHE_ROWS=4194304
export QSR_FLASHNEXT_PLE_IO=io_uring
export QSR_FLASHNEXT_PLE_IO_WORKERS=32
# Native QSA top-k emits an unordered set; keep score-order reranking enabled
# unless running an explicit performance-only A/B experiment.
export QSR_FLASHNEXT_QSA_TOPK_RERANK=1
export QSR_FLASHNEXT_HC_NORM_FUSION=0
export QSR_FLASHNEXT_HC_NORM_APPLY_FUSION=1
export QSR_FLASHNEXT_HC_POINTWISE_FUSION=1
export QSR_TRACE=1

exec /home/bot/.venvs/torch-nightly/bin/python -m server.app \
  --host 127.0.0.1 --port 8300 \
  --capacity 2 --num-slots 2 --blocks-per-slot 2048 \
  --qwen-kv-mode legacy --mtp --mtp-k 3
```

首次加载 checkpoint、编译 CUDA kernel 和捕获 Graph 需要等待；启动期间不要
重复启动第二个实例，也不要用相同端口发送大量测试请求。服务是前台进程，使用
`Ctrl-C` 停止。后台启动时先用 `pgrep -af 'server.app.*--port 8300'` 找到
精确 PID，再只终止该 PID。

## 启动验证

另开终端执行：

```bash
curl -fsS http://127.0.0.1:8300/health
curl -fsS http://127.0.0.1:8300/v1/models | python -m json.tool
curl -fsS http://127.0.0.1:8300/debug/stats | python -m json.tool
```

`/v1/models` 应至少列出：

```text
qwen3.8
qwen3.8-flash-next
```

最小 OpenAI Chat smoke：

```bash
curl -N http://127.0.0.1:8300/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    "max_tokens": 8,
    "stream": true,
    "reasoning_effort": "low"
  }'
```

需要检查真实端到端耗时、MTP 接受率和 prefix 命中时，查看
`/debug/stats`、`/debug/traces` 与 `/metrics`；不要把单次 trace 中不一致的
旧计时字段当作端到端基线。性能对比必须使用相同 prompt、上下文长度、输出长度
和启动 profile，并按 `docs/diagnostics-guide.md` 先做 `bf diff`。

### 默认采样参数

服务不会把 Qwen3.8 Flash-Next 的省略参数错误地降成贪心温度 0。根据
[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#recommended-sampling-parameters)，
默认思考模式使用 `temperature=1.0, top_p=0.95, top_k=20`；请求显式
关闭思考（`reasoning_effort=none`、`enable_thinking=false` 或等价的
`chat_template_kwargs`）时使用非思考/指令模式的
`temperature=0.7, top_p=0.80, top_k=20`。客户端显式提供的每个字段都会覆盖对应
默认值，因此需要确定性输出时仍可传 `temperature=0`。当前 runtime 暴露的采样
字段只有这三项；模型卡中的 `min_p`、presence penalty 和 repetition penalty
尚未作为 runtime 采样字段实现，不能假装已经生效。

Prefix cache 的 admission key 会同时携带 `sampled`/`greedy` 解码契约。两种模式
都会保存 target + MTP 状态；采样命中时恢复真实 MTP 前缀、覆盖旧 anchor 并重新
采样 draft/q 分布，greedy 命中则复用确定性 draft。模式不匹配的 checkpoint 不会
被恢复，避免把另一请求的随机 proposal 状态带进来；仍要求 token 前缀和视觉
cache key 完全一致。

## OpenCode / Windows 客户端

OpenCode provider 的最小配置如下，模型名使用服务实际暴露的
`qwen3.8-flash-next`：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "compaction": {
    "auto": true,
    "prune": true,
    "reserved": 32003,
    "preserve_recent_tokens": 12000,
    "tail_turns": 15
  },
  "provider": {
    "blackwellm": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8300/v1" },
      "models": {
        "qwen3.8-flash-next": {
          "name": "Qwen3.8 Flash-Next (local)",
          "reasoning": true,
          "interleaved": { "field": "reasoning_content" },
          "options": { "reasoningEffort": "medium" },
          "variants": {
            "none": { "reasoningEffort": "none" },
            "minimal": { "reasoningEffort": "low" },
            "low": { "reasoningEffort": "low" },
            "medium": { "reasoningEffort": "medium" },
            "high": { "reasoningEffort": "xhigh" },
            "xhigh": { "reasoningEffort": "xhigh" },
            "max": { "reasoningEffort": "xhigh" }
          },
          "limit": {
            "context": 262144,
            "input": 262144,
            "output": 32000
          }
        }
      }
    }
  },
  "model": "blackwellm/qwen3.8-flash-next"
}
```

这里使用 `@ai-sdk/openai-compatible`，因为本地服务的稳定入口是
`/v1/chat/completions`；不要把它替换成会优先走 `/v1/responses` 的
`@ai-sdk/openai`。`limit.input` 必须保留为 `262144`：OpenCode 1.18.25 的
V1 overflow 公式是 `input - compaction.reserved`，因此当前配置得到
`262144 - 32003 = 230141`，正好对齐服务 `/v1/models` 的安全输入上限（其中
32000 是最大输出，3 是 MTP speculative tail）。不要把 `input` 改成
`230141`，否则会提前丢掉 32003 个可用上下文 token。

服务的流式结束 chunk 也会携带 `usage.prompt_tokens`、
`usage.completion_tokens` 和 `usage.total_tokens`；这是 OpenCode 判断
overflow 并触发自动压缩所需的协议字段。若自定义代理剥离了该字段，OpenCode
会把该轮 token 记为 0，自动压缩就无法按上下文增长触发。

Flash-Next 默认将 `preserve_thinking` 设为 `false`。OpenCode 会把上一轮的
`reasoning_content` 原样回传；而该模型模板默认会把所有历史 reasoning 再放回
新的 `<think>` 块，长工具会话会因此反复进入同一段思路。需要完整保留隐藏推理
历史时，可在请求的 `chat_template_kwargs` 中显式设置
`{"preserve_thinking": true}`，或在服务启动前设置
`QSR_FLASHNEXT_PRESERVE_THINKING=1`。这不会关闭当前轮思考，只移除旧轮的隐藏
reasoning；`reasoning_effort` 仍照常生效。

Flash-Next tokenizer 原生支持 `low`、`medium`、`xhigh`；OpenCode 的
`minimal`/`high`/`max` 是 runtime 在请求边界归一化的别名。修改 Windows
配置后必须完全退出并重新打开 OpenCode（包括托盘进程），配置不会热加载。

如果 OpenCode 工作区位于 WSL 的 UNC 路径，并出现每轮等待约 80–90 秒、日志含
`failed to add snapshot files`，在 Windows 的
`%USERPROFILE%\\.config\\opencode\\opencode.jsonc` 中加入：

```jsonc
"snapshot": false
```

这是绕过 WSL symlink 无法加入 Git snapshot 的客户端兼容性设置，会关闭
OpenCode 的 undo/revert snapshot；不会改变 runtime、模型或推理质量。加入后
同样需要完全重启 OpenCode。

## 常见错误

- `model unavailable`：先确认 `/v1/models`，并使用精确模型 id
  `qwen3.8-flash-next`；不要连接旧端口 `8100` 或历史 Laguna 服务。
- 请求长时间卡住：先看服务进程是否仍在加载/捕获 Graph，再看 Windows
  OpenCode 日志是否有 snapshot symlink 错误；runtime 已返回时不要重复提交请求。
- OOM：不要在现有服务上动态增大 slot、并发或 256K 配额；停止后按新 profile
  冷启动，逐项做显存与质量验证。
- 输出/思考异常：保留 `QSR_TRACE=1`，采集 `/debug/traces` 和原始请求；先按
  相同 prompt 做 `bf diff`，避免把不同 effort、prefix 命中状态或上下文长度混为
  一个性能结论。
