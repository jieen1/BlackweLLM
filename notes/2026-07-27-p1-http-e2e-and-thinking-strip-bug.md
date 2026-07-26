# P1 真实 HTTP 端到端冒烟 + 发现并修复一个预置 bug(2026-07-27)

## 背景

`notes/2026-07-27-decode-cg-server-integration.md`(P1,commit `9ca7612`)把 decode CUDA
Graph 接入了 `decode_batch_sampled`,但当时受阻于 sparkinfer 的 ptxas 崩溃,只验证到
`LagunaBackend` 层面,没有跑过真实 HTTP 请求。sparkinfer 崩溃已回退(见
`notes/2026-07-27-sparkinfer-revert-ptxas-crash.md`),现在补做这次冒烟——这也是
`docs/roadmap.md` 明确标记过的缺口:"`_load_laguna_model` 从未过 GPU 冒烟"。

## 第一次真实请求就发现了一个真 bug(不是这次改动引入的)

用 `QSR_SERVER_MODEL_BACKEND=laguna QSR_SERVER_ENABLE_CUDAGRAPH=1` 启动真实
`server/app.py`,发 `POST /v1/completions`(prompt="The capital of France is",
temperature=0):

- 服务端日志的 `RAW OUTPUT` 是正确、连贯的文本("Paris. \n\n...")。
- 但最终返回给客户端的 `text` 字段是**空字符串**,日志里 `VISIBLE OUTPUT (0 chars)`。
- 这次请求还异常慢:69 秒(应该 <2 秒)。

### 根因(空输出):`server/app.py` 无条件假设所有 backend 都是 Qwen3.6 的 thinking 模型

`/v1/completions`(以及 `/v1/messages`)的响应处理里有这样一段:

```python
_raw_comp_full = (
    _raw_comp
    if _raw_comp.startswith("<think>")
    else ("<think>" + "\n" + _raw_comp)   # 无条件拼接!
)
text = strip_thinking(_raw_comp_full)
```

`strip_thinking`(`server/formats/thinking.py`)的注释明确写着这是给 **Qwen3.6 chat
template** 设计的("The Qwen3.6 chat template injects a `<think>` tag at the start of
assistant generation")。当模型输出本身不含 `<think>` 时,这段代码会**强行拼接一个
`<think>\n` 前缀**,然后 `strip_thinking` 的"未闭合 think 块"规则
(`_UNCLOSED_THINK_RE = r"<think>.*\Z"`)因为找不到对应的 `</think>`,会把从拼接的
`<think>` 到字符串末尾的**全部内容**当成"未完成的思考过程"删掉——Laguna 的普通
completion 输出本来就没有 `</think>`,于是整段正文被吃空。

聊天补全端点(`/v1/chat/completions`)已经有一个类似的绕过逻辑(`_non_thinking`,通过
客户端 `chat_template_kwargs={"enable_thinking": False}` 触发),但那是**客户端选择性
关闭**,不是**按 backend 区分**——而且 `/v1/completions` 和 `/v1/messages` 两个端点
完全没有这层保护,无条件假设。这是一个**预置于本次改动之前、和 CUDA Graph 集成无关**
的真实 bug,只是因为"从没有真实请求打过 Laguna 路径"(`docs/roadmap.md` 的原话)才
一直没暴露。

### 修复

在三个端点(`/v1/chat/completions`、`/v1/completions`、`/v1/messages`)都加上
`engine.backend_name != "qwen36"` 的判断:非 qwen36 backend 直接跳过"假设是 thinking
输出"的包装/strip 逻辑,原样(只做 `�` 替换字符清理)返回文本——和聊天端点已有
的 `_non_thinking` 分支做的事完全一致,只是触发条件从"客户端显式关闭"扩展到"backend
本来就不是 thinking 模型"。

修复后重新起服务、同样的请求:输出正确("Paris. \n\n...",303 字符,和 RAW OUTPUT
完全一致),日志显示 `VISIBLE OUTPUT (303 chars)`。

## 关于那次异常的 69 秒:不是 bug,是首次 JIT 编译热身

修复 strip_thinking 后重启服务测的第一次请求,同样的 prompt 只用了 **1.10-1.14 秒**
(3 轮全部一致)。推断:第一次的 69 秒是当时那个服务进程第一次真正触发某些 kernel
JIT 编译路径的一次性开销(本次 session 之前也见过好几次同类现象,比如
`notes/2026-07-27-speed-repro-verified.md` 里 round 0 的异常延迟),JIT 缓存
(`~/.cache/sparkinfer`/CUTLASS DSL 缓存)落盘后,同一台机器上后续进程的首次请求就
不用重新付这个代价了——不是 CG 集成本身的问题。

## CG 在真实服务路径生效的证据

修复后 3 轮 `POST /v1/completions`(64 token,temperature=0):稳定 **~1.1 秒**(含
6-token prefill + 64 步 decode + HTTP/JSON 开销),换算 decode 吞吐和独立 benchmark
测出的 CG-routed 数字(`benchmarks/measure_decode_cg_throughput.py`,83.8 tok/s)
量级一致,远快于同一路径此前测出的纯 eager 连续 decode(~1 tok/s 量级)。三轮输出
逐字节相同(温度=0 的确定性预期)。这是 `decode_batch_sampled` 真正在真实 HTTP 请求
下走通 CUDA Graph 路径的第一次直接证据(此前只验证到 `LagunaBackend` 单元层面)。

## 验证

- `pytest tests/` 全量:319 passed,3 failed(`test_bf_attention.py` ×2、
  `test_vllm_dependency_boundary.py` ×1)——这 3 个失败和这次改动无关,P1 那次任务已经
  用 `git stash` 确认是修改前就存在的既有失败,这次重跑确认数量、名字都对得上,没有
  新增回归。
- 真实 HTTP 服务器起停正常(`QSR_SERVER_MODEL_BACKEND=laguna
  QSR_SERVER_ENABLE_CUDAGRAPH=1`,capacity=1,num_slots=2),`/health` 就绪检测正常。

## 遗留问题

1. **流式路径有同类风险,未修**:`server/formats/stream.py` 的 `StreamProcessor.
   finalize()`(约 285-292 行)有几乎一样的模式("Prepend `<think>` for consistent
   processing"),对非 thinking backend 的流式请求 finalize 阶段大概率会有同样的
   吃空问题。增量 delta 路径(`get_delta`/等价方法)本身已经对"完全没有 think 标签"
   做了优雅处理("should not happen with Qwen3.6 but handle gracefully"),只有
   `finalize()` 没有。`StreamProcessor` 目前拿不到 `engine.backend_name`,修复需要
   多线一个参数进去,这次没有顺手做——需要单独验证(这次测试发现 `/v1/completions`
   带 `"stream": true` 时实际上没有走 SSE,返回的还是一次性 JSON,原因未查,可能是
   这个端点本来就不支持流式,需要另外确认,不在这次范围内)。
2. 只测了单请求、单 slot、`num_slots=2`(capacity=1 对应的最小配置)——没有测试并发
   多请求场景。
3. `server/app.py` 默认的 `SERVER_MODEL_BACKEND` 仍是 `"qwen36"`,`SERVER_ENABLE_
   CUDAGRAPH` 对 Laguna 仍默认 `False`——这次冒烟用显式环境变量启动验证,没有改这两个
   默认值(要不要默认打开是运营决策,建议至少 CUDA Graph 这项现在可以考虑打开了,
   因为已经有真实 HTTP 端到端证据)。
