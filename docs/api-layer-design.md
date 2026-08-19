# API 层设计：现状、契约、扩展路径

> 编制日期：2026-08-01 · 基线 commit：`40e9cdd`（`fix/t0-api-thinking` 分支）
> 配套：[`docs/roadmap.md`](roadmap.md) §1.2 R3/R4、§4 Track E、§7 D1；
> [`docs/architecture.md`](architecture.md) §3（目标架构，覆盖模型/调度层，
> 不覆盖本文档描述的协议层内部设计）。
>
> 本文档只覆盖 **HTTP 协议层**（`server/app.py` + `server/formats/`）。
> 模型抽象（`ModelSpec` / `ModelBackend` / `SlotResourceManager`）已经在
> `docs/architecture.md` §3.2 里设计过，这里不重复。

---

## 0. 结论先行

1. **thinking/reasoning 契约已修**（本次改动的主任务）：服务端不再对最终文本
   跑贪婪正则，改为生成流上的状态机（`StreamProcessor`），非流式路径复用同一
   状态机的结果。契约：`content`/`text` 永不含 reasoning；OpenAI 走
   `reasoning_content`（delta/message），Anthropic 走非标准的
   `reasoning_content_delta` 事件 / 顶层 `reasoning_content` 字段（**不是**
   spec 的 `thinking` content block —— 见 §1.4 的历史事故还原）。
   `QSR_REASONING_MODE=expose|strip` 控制是否暴露，默认 `expose`。
2. **现状分层比看起来整齐，但流式路径完全绕开了 formats 层**——两套协议的
   SSE 序列化各自手写在 `app.py` 里，`server/formats/openai.py` 和
   `anthropic.py` 里对应的 `build_sse_chunks`/`build_sse_events` 是死代码。
   加一种新协议，流式这块必须整个重写，没有可复用的骨架。
3. **本次顺手做的低风险修复**：三个协议无关的 bug——HTTPException 被
   FastAPI 包了一层多余的 `{"detail": ...}`、pydantic 422 完全没有协议形状、
   `_validate_capacity` 的 metrics 会被新加的统一处理器重复计数——都已经修好，
   现在两套协议共享同一套错误处理路径（§4.3）。
4. **任务 3 核查结论**（§5）：**结构化输出（json_object/json_schema）是纯骨架，
   实际不生效**——`GrammarState.apply_mask*` 从未被 `server/engine.py` 调用过，
   语法从不真正约束采样。`stop` 序列**完全没有实现**（两套协议都没有）。
   `seed` 能保证确定性，但实现方式（每步重新播种，不是连续推进一个流）和
   OpenAI 的隐含语义有细微偏差。

---

## 1. thinking / reasoning 契约：问题与修复

### 1.1a Qwen3.8 request-level effort（2026-08-19）

Qwen3.8 的官方模板把 `reasoning_effort` 作为 Jinja 变量，模板文件默认值是
`xhigh`；runtime 在加载 native Qwen tokenizer 时将这个模板默认改为
`medium`（可用 `QSR_DEFAULT_REASONING_EFFORT` 覆盖），这不是一个硬 token budget。OpenAI Chat Completions 的顶层
`reasoning_effort`、Responses API 的 `reasoning.effort` 现在由
`server/app.py::_resolve_chat_template_kwargs` 映射到模板：`low`、`medium`、
`high`、`xhigh` 保持思考并传递 effort，`none` 映射为
`enable_thinking=false`。显式 `chat_template_kwargs` 优先级最高；未传任何字段时
请求 kwargs 保持为空，直接使用 runtime 的模板默认 `medium`。

不要把 `max_tokens` 当作 thinking budget：它限制整个 completion，达到上限时
可能只有未闭合 reasoning 而没有正文。当前 runtime 的
`thinking_token_budget` 采用更底层的单次连续生成路径：调度器跟踪
`<think>`/`</think>` token span，并在 plain decode 或 MTP verify 的 logits 边界
强制输出 close marker；不会通过二次 HTTP 请求或响应层字符串拼接续生成。官方
两阶段 early-stop 方案仍是可选的上层策略，当前实现细节和真实 GPU E2E 证据见
`notes/2026-08-19-qwen38-thinking-overrun-reproduction.md`。

### 1.1 坏在哪（修复前，`fix/t0-api-thinking` 分支基线 `40e9cdd`）

`docs/roadmap.md` R3/R4 已经点出问题，这里补完整证据链：

- `d52a3b1`（"Strip thinking tags from all API responses"）让三个端点
  （`/v1/chat/completions`、`/v1/completions`、`/v1/messages` 的非流式路径）
  统一调用 `strip_thinking(raw_text)`，但 commit message 写 "Tested: unit
  tests pass"——`tests/test_laguna_server_integration.py::
  test_laguna_chat_response_preserves_generated_think_tags` 当时就是红的
  （断言"逐字保留"，和这个改动的"整段剥离"直接矛盾）。
- `ce21eb5`（"Fix streaming: enable thinking strip in SSE paths"）把两处
  `StreamProcessor(engine.tok, thinking_capable=False)` 改成
  `thinking_capable=True`——这是对 `notes/2026-07-27-p1-http-e2e-and-
  thinking-strip-bug.md` 里那次真实事故修复的**直接回退**：那次事故的根因
  正是"无条件假设所有 backend 都是 Qwen3.6 的 thinking 模型"，把 Laguna
  的普通输出整段吃空。
- `server/formats/thinking.py` 的两个正则是这一切的病灶：
  - `_ORPHAN_CLOSE_RE = r"\A.*?</think>\s*"`——匹配"字符串开头到第一个
    `</think>`"，不管中间是什么。对着一段没有真正 think 块、只是**碰巧提到**
    `</think>` 字面量的正文（比如"这个标签用 `</think>` 闭合"），会把它之前的
    全部正文删掉。
  - `_UNCLOSED_THINK_RE = r"<think>.*\Z"`——匹配"任意 `<think>` 到字符串末尾"，
    同样不看位置。正文里讨论 `<think>` 标签用法的请求（对一个主要服务
    代码/agent 场景的运行时是高频场景）会被从第一次提到 `<think>` 起截断。
  - 模块 docstring 写着"The Qwen3.6 chat template injects a `<think>` tag"——
    但 Qwen3.6 早已被移出生产路径（`docs/roadmap.md` R7），当前生产模型是
    Laguna，且已用真实 GPU 输出验证过 **Laguna 的 chat template 不注入
    `<think>`**（`notes/2026-07-27-p1-http-e2e-and-thinking-strip-bug.md`）。
    `<think>` 只会作为 Laguna **自己选择生成**的内容出现——这正是
    `test_laguna_chat_response_preserves_generated_think_tags` 这个测试名字
    里"generated"两个字的来源。

### 1.2 新契约

由用户裁定（对应 `docs/roadmap.md` §7 D1 的三个选项 (a)/(b)/(c)：这次选
**(b) 按协议暴露，外加 (c) 的开关**，不是 (a) 一律剥离）：

| | OpenAI | Anthropic |
|---|---|---|
| 正文 | `message.content` / `delta.content`：**永不含 reasoning** | `text` content block：**永不含 reasoning** |
| reasoning（`expose` 模式，默认） | `message.reasoning_content`（非流式）/ `delta.reasoning_content`（流式），vLLM `--reasoning-parser` 同款约定 | 顶层 `reasoning_content` 字段（非流式）/ 自定义 SSE 事件 `reasoning_content_delta`（流式）—— **不是** spec 的 `thinking` content block，见 §1.4 |
| reasoning（`strip` 模式） | 不出现 `reasoning_content` 字段 | 不出现 `reasoning_content` 字段 / 不发 `reasoning_content_delta` 事件 |
| 开关 | `QSR_REASONING_MODE=expose\|strip`（`server/app.py` 的 `SERVER_REASONING_MODE`，默认 `expose`） | 同上 |

对当前客户端的可见行为和"一律剥离"是一致的（正文依然干净），区别是
reasoning 不再被销毁，且不会误伤正文。

### 1.3 判定规则：什么时候算"reasoning 段"

判定逻辑集中在 `server/formats/thinking.py::find_reasoning_span`（纯文本
函数，`server/formats/stream.py::StreamProcessor` 的流式/非流式两条路径都
调用它，不是各写一遍）：

> **只有 `<think>` 是生成文本的第一个字符时，才认为存在 reasoning 段**
> （或者 `thinking_capable=True`——chat template 把 `<think>` 注进了
> prompt，这种情况下生成的 token 从第一个字符起就已经隐式在 think 块内，
> 见 §1.5）。任何不在位置 0 出现的 `<think>`/`</think>` 都是普通正文。

这一条规则同时解决了：

- **R4 的误伤 bug**：正文中途提到的 `<think>` 不再被当成信号——见
  `tests/test_thinking_reasoning.py::TestFindReasoningSpan::
  test_think_not_at_position_zero_is_not_a_span` 和
  `tests/test_laguna_server_integration.py::
  test_laguna_chat_response_literal_think_mid_body_not_truncated`（后者是
  任务要求的强制回归用例，走真实的 `chat_completions` handler）。
- **未闭合（撞 max_tokens）**：`find_reasoning_span` 返回 `closed=False`，
  `end=len(text)`——整段都是 reasoning，正文为空
  （`test_unclosed_think_at_max_tokens_yields_no_content`）。
- **孤儿闭合标签**（没有匹配的开标签）：不满足"`<think>` 在位置 0"，直接
  判定为无 reasoning 段，原样保留——这是对旧版 `_ORPHAN_CLOSE_RE` 的直接
  替换（`test_orphan_close_tag_is_ordinary_content`）。
- **模板注入的开标签不在输出里**：`thinking_capable=True` 时，
  `StreamProcessor._get_raw()` 会在检测到解码文本不以 `<think>` 开头时
  自己拼一个（这部分逻辑没变，只是现在默认关闭，因为没有 backend 会用到
  它——见 §1.5）。

### 1.4 Anthropic 侧为什么不是 spec 的 `thinking` content block

这是**任务书原文和历史事故直接冲突**的地方，需要明确记录，因为下一个改
这块代码的人（人或 agent）必须知道这不是疏漏：

`tests/test_format_regression.py::test_anthropic_sse_no_thinking_blocks`
和 commit `f13fd4a`（"Fix Anthropic SSE stream: remove invalid thinking
blocks that caused Claude Desktop to drop tool_use"）记录了一次真实的生产
事故：

```
Claude Desktop validates the cryptographic signature on thinking blocks.
Our fake 32-char hex signature was rejected, causing the client to DROP
all subsequent content blocks -- including tool_use (e.g. AskUserQuestion).
...
Rejected: Emit thinking with fake signature | Claude Desktop validates and drops subsequent blocks
Rejected: Emit thinking with empty signature | Same validation failure
Directive: Do NOT re-add thinking block emission without a valid signature source
```

我们不是 Anthropic 官方后端，没有对应的签名私钥，**无法产出一个真实客户端
会接受的 `signature_delta`**——这个约束在两年后的今天依然成立。这次任务的
指令要求"Anthropic 侧：reasoning 作为 thinking content block"，但字面执行
会精确重现这次事故（用户的工具调用结果会被 Claude Desktop 静默丢弃）。

**这次的选择**：保留"暴露而不是销毁"的精神，但走一个不冒充 spec content
block 的通道——非流式在响应体顶层加 `reasoning_content` 字段（不在
`content` 数组里，普通 SDK 的宽松解析会忽略未知字段）；流式发一个自定义
`event: reasoning_content_delta`（不在 Anthropic 文档定义的事件类型集合
里，符合规范的 SSE 消费者按 `event:` 名字分发、忽略不认识的名字）。
`tests/test_format_regression.py::test_anthropic_sse_no_thinking_blocks`
和新增的
`tests/test_laguna_server_integration.py::
test_anthropic_stream_reasoning_via_custom_event_not_thinking_block` 一起
锁住这个边界：两个测试都断言输出里没有 `"type": "thinking"` 和
`signature_delta`。

**这是需要人拍板的点**：如果确实需要字面意义的 Anthropic `thinking` block
（比如判断某些客户端能容忍无签名的 block，或者接受 Claude Desktop 场景
下降级），需要显式决定覆盖这条历史 directive，而不是靠下一次改动"顺手"
改回去——那正是这次事故复现的路径。

### 1.5 `thinking_capable` 参数：两种机制，不是"是否支持思考"

`StreamProcessor(tokenizer, thinking_capable: bool = True)`——这个参数名
容易望文生义，实际含义是"chat template 有没有把 `<think>` 注进 prompt"：

- `thinking_capable=True`：模板在 prompt 里注入了开标签，生成的 token 从
  第一个字符起就已经隐式在 think 块内（`<think>` 字面量不会出现在生成文本
  里）。当前**没有 backend 用这个模式**（Qwen3.6 已被移出生产路径，见
  `docs/roadmap.md` R7），但机制保留：Track B 若把 Qwen3.6 或任何走同样
  模板约定的模型接回来，直接把这个参数设 `True` 就行,不需要改状态机。
- `thinking_capable=False`（Laguna 的默认值，`server/app.py` 的
  `SERVER_THINKING_CAPABLE`）：模板不注入，`<think>` 只可能是模型自己选择
  生成的内容——用 §1.3 的"位置 0"规则判定。

`server/app.py` 里四个 `StreamProcessor(...)` 构造点（chat 流式/非流式、
messages 流式/非流式）现在统一传 `thinking_capable=SERVER_THINKING_CAPABLE`，
不再各自硬编码——这是本次顺手做的、把"这个值该是什么"收敛成一个决策点
的改动。

### 1.6 流式边界情况：avoid "先发后撤"

`StreamProcessor` 是逐 token 增量吃入的，两类边界情况在写状态机时才会
暴露（原来的一次性正则实现看不到）：

1. **开标签的歧义前缀**：`thinking_capable=False` 时，解码文本还不够长
   （比如只有 `"<th"`，3 个字符）时，无法确定它会不会长成 `"<think>"`。
   `_reasoning_span(raw, final=False)` 在这个窗口返回哨兵值 `"pending"`，
   `drain_content()`/`drain_thinking()` 都原样等待，不猜测。generation
   结束时（`final=True`）不再有这个歧义——不管长成什么样，就是最终答案。
2. **闭标签的部分匹配**（这是本次实现时才发现的真 bug，不在原始任务范围
   但同一类问题，已经修复并有回归测试
   `tests/test_thinking_reasoning.py::
   TestNonStreamingReusesSameStateMachine`）：`drain_thinking()` 增量吐出
   reasoning 文本时，如果 `</think>` 还没完整到达，"目前看到的 thinking
   文本"里会暂时包含 `</thin` 这样的部分闭标签字符。如果直接把这些字符
   当作 delta 发给客户端，等标签补全后再从"已发送长度"里减掉是不可能的
   （SSE 是单向的，发出去的没法撤回）。修复：`_trim_ambiguous_tail`
   （`server/formats/stream.py`）在闭标签未完整确认前，从待发送文本尾部
   剪掉任何"可能是 `</think>` 前缀"的部分，等它确认变成真正的闭标签（被
   排除）或者确认不是闭标签（补发）再决定。`<usage>` 的同类部分匹配逻辑
   顺手复用了同一个 helper（之前是内联的 for 循环，行为不变，只是去重）。

### 1.7 `<usage>` 剥离：判断为不同类问题，未改行为

任务要求判断 `<usage>` 剥离和 U+FFFD 清理是不是和 `<think>` 同一类问题：

- **U+FFFD 清理**：和 thinking 完全无关，是 tokenizer 解码层面的噪音清理
  （字节级 BPE token 解码出不完整 UTF-8 序列）。任何位置出现都清理是安全
  的——U+FFFD 不会被合法内容使用。未改动。
- **`<usage>` 剥离**：和 `<think>` 是**不同类问题**。`<think>` 是模型可能
  被要求讨论的合法概念（正文里讨论"这个标签怎么用"很常见）；`<usage>` 是
  一个训练数据污染的窄artifact（docstring 原话："model artifact from
  training data that included Claude sub-agent output format"），没有
  证据表明有正常请求会让模型讨论字面意义的 `<usage>` 标签。因此"在任意
  位置出现都剥离"（不像 `<think>` 那样要求位置 0）在实践中风险低得多，
  保留原行为：
  - 非流式：`server/formats/thinking.py::strip_usage_artifacts`，正则
    整段删除（支持一次响应里出现多个 `<usage>` 块）。
  - 流式：`StreamProcessor.drain_content()` 里"发现 `<usage>` 就冻结此后
    全部输出"（和检测到工具调用 XML 时的机制一致）——**注意这和非流式的
    "删除该块、后续内容继续输出"不是同一种语义**，这是重构前就存在的不
    一致，不是这次引入的，见 §5.5。
  - 唯一的风险点：`_UNCLOSED_USAGE_RE`（未闭合 `<usage>` 吃到字符串末尾）
    理论上和 `_UNCLOSED_THINK_RE` 是同一种"贪婪到底"模式，但触发条件更窄
    （需要模型正好在生成一个未闭合的 `<usage>` 字面量时撞上 max_tokens），
    未在这次改动中处理，记在 §5.5 留给以后。

---

## 2. 现状分层：证据与问题清单

### 2.1 现状分层图

```
server/app.py（1402 行，本次改动后）
  ├─ FastAPI 路由装饰器 + 6 个端点 handler（直接在 app.py 里，不是独立模块）
  ├─ 3 个全局异常处理器（HTTPException / RequestValidationError / Exception）
  ├─ 请求校验（_build_sampling_params 等）—— 协议无关，两套协议共享 ✅
  ├─ 非流式响应构建 —— 委托给 server/formats/{openai,anthropic}.py::build_response ✅
  └─ 流式响应构建 —— ⚠️ 内联在每个 handler 的闭包里，不经过 formats 层

server/formats/
  ├─ content.py    —— 协议无关的内容块解析，两套协议共享 ✅
  ├─ tools.py      —— 协议无关的工具调用解析/格式化 ✅
  ├─ stream.py     —— StreamProcessor：协议无关的生成流状态机 ✅
  ├─ thinking.py   —— 协议无关的 reasoning/usage 文本处理 ✅
  ├─ openai.py     —— 请求解析 ✅ + 非流式响应构建 ✅ + 流式构建（死代码 ⚠️）
  └─ anthropic.py  —— 同上
```

**结论**：请求解析、内容块/工具调用解析、非流式响应构建这几层的分工是
干净的（协议特定逻辑封在各自的 `formats/*.py` 里，`app.py` 只做路由和
拼装）。**流式响应构建完全没有走这套分层**——这是加新协议时最大的成本
来源，见 §2.2。

### 2.2 具体问题（都有 file:line 证据）

| # | 问题 | 证据 | 加一种新协议时的影响 |
|---|---|---|---|
| L1 | 流式 SSE 序列化内联在 `app.py` 里，不经过 `formats/*.py` | `server/app.py:622-770`（OpenAI `_sse()` 闭包，148 行）、`server/app.py:1211-1366`（Anthropic `_anthropic_sse()` 闭包，155 行）——手写 JSON 结构 | 新协议的流式支持=从零手写一整套闭包，不能复用任何现有骨架 |
| L2 | `formats/openai.py::build_sse_chunks` / `formats/anthropic.py::build_sse_events` 是死代码 | `grep -n "build_sse_chunks\|build_sse_events" server/` 只在 `tests/test_regression_unit.py` 和 `tests/test_format_regression.py` 命中，`app.py` 从不调用 | 这两个函数会误导新协议的实现者——它们的签名（"给一段完整文本，一次性吐出假装是流式的 chunk"）和真实的增量流式（`StreamProcessor` 逐 token 吃入）不匹配，是早期设计遗留，不是当前真实契约的参考实现 |
| L3 | 每个端点各自决定何时调用 `StreamProcessor.drain_thinking/drain_content/drain_tool_deltas`，各自决定怎么包装成协议特定的 JSON | `server/app.py` 里 OpenAI 流式循环（622-770 行）和 Anthropic 流式循环（1211-1366 行）结构几乎相同但完全独立维护 | 加新协议＝再复制一份"读三个 drain 方法、包装成协议 JSON"的循环，两处逻辑分别演进，容易像 R3/R4 一样出现"改了一处忘了另一处" |
| L4 | debug 日志按端点各自调用，tag 字符串手写 | `_debug_log_input("OPENAI /v1/chat/completions", ...)` / `_debug_log_input("ANTHROPIC /v1/messages", ...)` 等 6+ 处调用，每处手写协议名字符串 | 低风险但每加一个端点要记得抄一遍调用 |
| L5 | 请求级校验混合了协议特定分支 | `_build_sampling_params` 本身协议无关，但被 `/v1/messages`（`server/app.py:1183`）和两个 OpenAI 端点共享时，Anthropic 侧的 400 曾经要手写自己的 JSONResponse（本次已收敛，见 §4.3）——提醒：协议特定的"如何呈现校验失败"曾经泄漏进本该协议无关的校验函数调用点 | 本次已修好这一条；新协议只需要在 `_protocol_error_body` 里加一个分支 |
| L6 | `SERVER_MODEL_BACKEND`/`SERVER_THINKING_CAPABLE` 等运行期常量硬编码为 Laguna 的值 | `server/app.py:80`（`SERVER_MODEL_BACKEND = "laguna"`）、本次新增的 `SERVER_THINKING_CAPABLE = False` | 这是 `docs/architecture.md` Track A（模型抽象层）要解决的问题，不是协议层问题——协议层的 `thinking_capable` 参数已经是"按 backend 能力"设计的（§1.5），只是"这个能力从哪读出来"目前还是硬编码常量，等 Track A 的 `ModelSpec`/`ModelRegistry` 落地后应该改成从模型描述里读 |

### 2.3 "加一种新协议现在要动几处"

以当前分层现状（含本次修复）为基准，接入协议 N（假设是流式+非流式+工具
调用，参照 OpenAI/Anthropic 的完整度）：

1. `server/formats/protocol_n.py`：请求解析（`parse_*`）+ 非流式响应构建
   （`build_response`，接受 `text` + `reasoning_content` + 其他字段，内部
   调 `parse_tool_calls`）——**有 openai.py/anthropic.py 可抄，工作量小**。
2. `server/app.py`：新增 3-5 个路由 handler：
   - 非流式：`engine.submit(...)` → `StreamProcessor.content_text()` /
     `.reasoning_content()` → 调 `protocol_n.build_response(...)`——**这条
     路径干净，可以照抄现有两个端点的非流式部分**。
   - 流式：**从零手写一整套 SSE/等价格式的增量循环**（L1/L3 的直接后果）——
     这是当前唯一没有"照抄即可"的部分,需要理解 `StreamProcessor` 的四个
     drain 方法契约（见 `server/formats/stream.py` 的类 docstring）。
3. `server/app.py` 模块顶部：新增协议特定的 Pydantic 请求 schema
   （若该协议走 JSON body + 类型校验；若像 Anthropic 一样手动 `await
   request.json()`，可以跳过这步，但也失去 pydantic 自动 422 校验）。
4. `_protocol_error_body`（`server/app.py`）：加一个 `path.startswith(...)`
   分支，让该协议的错误响应形状正确——**一行改动，本次审计后统一在一处**。
5. 如果该协议有 reasoning 概念：在 `formats/protocol_n.py` 决定"reasoning
   放哪"（顶层字段？专有 event？正规 content block？）——**这是一个需要
   对该协议的真实客户端做兼容性验证的决策点，不是纯技术问题**（参照
   §1.4 的 Anthropic 教训：字面上"最像 spec"的做法不一定是能用的做法）。

**低风险 vs 需要单独排期**：1/3/4 是机械劳动，可以在加协议时顺手做。
2 的流式部分是真正的工作量所在——**这正是本次故意不做的"投机性大爆炸重构"
的边界**：把 L1-L3 抽成一个"给定 StreamProcessor 的 drain 输出，序列化成
协议 X 的流式事件"的适配器，需要设计一个跨两种已知协议（未来第三种形状
未知）都适用的中间表示，而目前只有两个具体实现可供归纳，抽象没有第二个
独立现实案例做交叉验证（`AGENTS.md`："一个只为'将来可能'存在的抽象是
负债"）——留给 Track E 或有第三个协议的真实需求出现时再做。

### 2.4 补充：`tool_parsers/` 注册表是"按模型"，不是"按协议"

本文档成稿的同时，main 上独立落地了 `server/formats/tool_parsers/`
（`85cdf9d` / `00d0990`）：一个按模型分派的工具调用解析器注册表，
`ToolCallParser` + `get_active_parser()` / `set_active_parser()`，
由 `QSR_TOOL_CALL_PARSER` 选择（对标 vLLM 的 `--tool-call-parser`），
当前有 `poolside_v1` 和 `qwen3_coder` 两个实现。

**这是一个和本节正交的轴，不要混淆**：

| 轴 | 变化的是 | 抽象点 |
|---|---|---|
| **协议**（OpenAI / Anthropic / 未来的 N） | 客户端怎么收发 | 请求解析 + 响应/流式事件序列化 |
| **模型**（Laguna / Qwen3.6 / …） | 模型怎么吐工具调用 | `ToolCallParser`：`open_tag` + 解析成结构化参数 |

`StreamProcessor` 同时贯穿两轴：它从 `tool_parser` 拿模型侧的形状，
把结果交给协议侧序列化。所以 Track A 的多模型接入会往 `tool_parsers/`
加条目，而加一种新协议不会——**两边各自扩展，互不牵扯**。这个注册表也是
一个有用的先例：它是从"第二个真实模型形状出现"归纳出来的，
不是预先设计的，正是 §2.3 结尾说的那种"等第二个独立现实案例"再抽象。

---

## 3. 内部统一表示：`StreamProcessor` 已经是它了

不需要新设计一个"内部统一表示"——`server/formats/stream.py::
StreamProcessor` 已经是这个角色，只是没有被明确地当作"协议 adapter 的
输入契约"来描述。一个协议 adapter 需要实现：

```
请求解析（parse_*）
  外部协议的 JSON body → chat_messages: list[dict]（role/content，
  与 openai.parse_chat_messages / anthropic.parse_messages 同一形状）
       │
       ▼ engine.submit() / engine.submit_stream()（协议无关，已经如此）
       │
StreamProcessor（协议无关的中间表示，已经如此）
  .add_tokens(token_ids)                     —— 喂入
  .drain_thinking() -> list[str]              —— 流式：reasoning 增量
  .drain_content() -> list[str]               —— 流式：正文增量（工具调用 XML 已剔除）
  .drain_tool_deltas() -> list[dict]           —— 流式：工具调用增量
  .content_text() / .reasoning_content()      —— 非流式：一次性获取上面两者的全量结果
  .finalize() -> (visible_text, tool_calls)    —— 非流式：文本 + 已解析的工具调用列表
       │
       ▼
响应序列化（build_response / 流式序列化）
  中间表示 → 该协议的 JSON 形状（非流式已经统一走这条路；流式见 §2.2 L1）
```

新协议要做的事：**请求解析** 输出同一个 `chat_messages` 形状；
**响应序列化** 消费同一个 `StreamProcessor` 契约。中间这一层不需要为
"支持更多格式"发明新东西。

---

## 4. 本次顺手修复的低风险改动（已完成）

### 4.1 HTTPException 的 `{"detail": ...}` 包裹（新发现，不在原始任务范围）

用 `TestClient` 实测验证（`docs/api-layer-design.md` 撰写过程中做的实验，
回归测试在 `tests/test_laguna_server_integration.py::
test_openai_error_shapes_via_real_http_dispatch`）：`_invalid_request()`
构造的 `HTTPException(status_code=400, detail={"error": {...}})`，在没有
自定义 `exception_handler(HTTPException)` 的情况下，FastAPI 的默认处理会
把 `detail` 原样塞进 `{"detail": <detail>}`——最终响应体是
`{"detail": {"error": {"message": ..., "type": "invalid_request_error"}}}`，
**双重嵌套，两套协议的 SDK 都无法按预期解析**。修复：新增
`@app.exception_handler(HTTPException)`（`server/app.py:497`），解包
`detail`，按 `request.url.path` 重新套上正确协议的外壳。

### 4.2 RequestValidationError 的 422（新发现）

pydantic 请求体校验失败（比如漏了必填的 `messages` 字段）时，FastAPI 默认
返回 `{"detail": [{"loc":..., "msg":..., "type":...}, ...]}`——同样两套
协议都不认。修复：`@app.exception_handler(RequestValidationError)`
（`server/app.py:521`），拼成一句话消息，套用同一个 `_protocol_error_body`。

### 4.3 统一错误外壳 + 消灭 metrics 重复计数

新增 `_protocol_error_body(path, err)`（`server/app.py:489`），三个异常
处理器和之前 Anthropic 手写的两处 `JSONResponse`（"no messages
provided"、"prompt too long"）现在都调它。副作用：`_validate_capacity`
原来在抛异常前自己调一次 `metrics.record_error`，现在统一处理器会给
**每一个**被抛出的 `HTTPException` 记一次错误——如果校验函数自己也记，
就会重复计数。已经把 `_validate_capacity` 和两处 Anthropic 手写校验里的
显式 `metrics.record_error` 调用删掉，回归测试：
`tests/test_laguna_server_integration.py::
test_validate_capacity_error_metric_not_double_counted`。

以上三项全部落在 `server/app.py`，改动范围小、有测试锁定，符合"只做修
thinking 所必需的、低风险的结构调整"的授权范围——之所以顺手做，是因为
审计 thinking 契约的错误路径时，用同一个 `TestClient` 手段顺带发现的，
不是计划外的范围扩张。

---

## 5. 任务 3：已知缺口核查结论

> **2026-08-01 更新（`fix/t0b-api`）**：本节 §5.1/§5.2/§5.4 三个缺口在这一批
> 全部处理完毕，结论见 §7。本节原文保留不改——它是审计当时的证据记录，
> §7 记录的是后续基于这份证据做出的决定和落地细节。

### 5.1 结构化输出（`runtime/structured_output.py`）—— **只有骨架，不生效**

`server/engine.py:693`（`_activate_slot`）确实会在 `response_format` 存在
时创建 `GrammarState`（`fmt = ResponseFormat.from_api(req.response_format);
if fmt.is_constrained: self.active[slot]["grammar"] = GrammarState(fmt,
self.tok)`），并在每个 committed token 后调用 `grammar.accept(tok)`
（`server/engine.py:1058, 1128`）。

但是：**`GrammarState.apply_mask()` / `.apply_mask_batch()`
（`runtime/structured_output.py:127-141`，真正把语法约束应用到 logits 的
地方）在 `server/engine.py` 里从未被调用过**——`grep -n "apply_mask"
server/engine.py` 零命中。采样/贪心解码前从来没有对 logits 做任何掩码。

结论：一个带 `response_format={"type": "json_object"}` 的请求，服务端
会创建语法状态机、在每个 token 生成后"喂"给它，但从来没有在生成**之前**
用它约束候选 token——模型完全不受任何 JSON 语法约束地自由生成，
`GrammarState` 只是在事后记账（且因为 `accept_token()` 在 token 不合语法
时静默返回 `False`——已用本机安装的 xgrammar 版本验证其文档行为，不会
抛异常——记账错了也不会报错，不会导致请求崩溃，但也不会有任何提示）。

**这不是"低风险顺手能修"的范围**：真正接上需要理解 `server/engine.py`
的解码循环里贪心/采样两条路径分叉的具体位置（`classify_decode_slots`
之后，`sample_from_logits`/`argmax` 调用之前），是一处核心解码路径的改动，
按文件归属边界（`runtime/**` 改动需要最小化+写清楚理由）不在这次任务
范围内实施，只诊断记录，交给排期。

### 5.2 `stop` 序列 —— **完全没有实现**

`ChatCompletionRequest`/`CompletionRequest`（`server/app.py`）都没有
`stop` 字段；Anthropic 的 `/v1/messages` 也从不读取请求体里的
`stop_sequences`。所有响应里的 `stop_sequence` 字段都硬编码为 `None`
（`server/formats/anthropic.py` 三处）。客户端传 `stop`/`stop_sequences`
会被静默忽略，不会报错——这本身也是一个问题（无提示的功能缺失比明确的
400 更容易被误用）。修复需要在 `server/engine.py` 的 decode 循环里维护一个
滑动窗口匹配停止串（协议层已经解析了 `stop`，但没有传给 engine），是
Track E1（`docs/roadmap.md`）里明确排了期的项，不在这次范围内实施。

### 5.3 `n>1` —— 正确处理（拒绝，非静默忽略）

`_build_sampling_params`（`server/app.py:447`）显式检查 `n is not None and
n != 1`，返回 400 明确拒绝。这是正确的行为——比"静默只返回 1 个"更安全，
不需要改动。

### 5.4 `seed` —— 能保证确定性，但不是"连续推进一个流"

`runtime/sampling.py::make_generator(seed)` 在
`runtime/backends/laguna.py` 的三个采样调用点（1500/1831/1920 行）**每次
都重新创建**一个用同一个 `seed` 播种的 `torch.Generator`，而不是维护一个
贯穿整段生成、随每个 token 前进的生成器状态。实际效果：同一 prompt + 同一
seed，逐 token 的 logits 分布不同 ⇒ 结果仍然是确定且可复现的（满足
`seed` 最常见的用途）。但语义上不是 OpenAI 文档隐含的"一个连续前进的随机
流"——如果两个不同位置恰好算出完全相同的 logits 分布，会采出完全相同的
"随机"结果，这是一个理论上的、影响极窄的偏差，不构成当前的正确性问题，
记录供 Track E1 审计时参考。这是 `runtime/backends/laguna.py` 的改动，
不在本次 `server/**` 范围内。

### 5.5 usage token 统计准确性 —— 基本正确，两个小缺口

- `prompt_tokens`/`completion_tokens`（`server/engine.py:711-712`）分别取
  `len(req.prompt_ids)` 和 `len(committed_tokens)`——即使前缀缓存命中，
  `prompt_tokens` 依然报告完整的逻辑 prompt 长度（缓存收益单独通过
  `cache_read_input_tokens`/`prefix_cache_hit_tokens` 上报），这是正确的
  语义，两套协议都符合。
- **缺口 1**：`completion_tokens` 包含 reasoning token（符合 vLLM 的惯例），
  但响应里没有 OpenAI 真实 API 提供的
  `usage.completion_tokens_details.reasoning_tokens` 细分字段——客户端拿
  不到"这次生成里有多少 token 花在 reasoning 上"。低优先级 nice-to-have，
  本次未实现（不是 bug，是功能缺口）。
- **缺口 2**（§1.7 提到）：`<usage>` 剥离在流式和非流式路径下的语义不一致
  （流式是"遇到即冻结此后全部输出"，非流式是"删除该块、后续内容继续
  展示"）——如果一次响应里 `<usage>` 块**不是**最后内容（比如模型先吐一段
  垃圾 `<usage>` 块又继续正常回答），流式客户端会看到比非流式客户端更少
  的内容。这是重构前就存在的行为差异，不是本次引入，本次也未修改（判断
  为低风险：`<usage>` 是训练数据污染的窄 artifact，实践中极少出现在内容
  中段），记录留档。

### 5.6 错误码语义 —— 本次已修复三处（见 §4），核查后无更多发现

- 400（参数校验、容量超限）：两套协议现在都返回各自 spec 的错误形状（本次
  修复，§4.1/4.3）。
- 422（pydantic 校验失败，仅 OpenAI 端点会触发，因为 Anthropic 端点手动
  解析 body 不用 pydantic model）：本次修复（§4.2）。
- 500（未捕获异常）：本次修复（§4.1 的姊妹修复）。
- 404（未匹配路由）：未特别处理——Starlette 默认 404 也会经过新的
  `_http_exception_handler`（因为 404 也是一个 `HTTPException`），会被
  转成 `{"error": {"message": "Not Found", "type": "invalid_request_error"}}`
  这种形状——"type" 字段语义不完全精确（"Not Found" 不是"参数不合法"），
  但至少不再是 FastAPI 默认的 `{"detail": "Not Found"}`。这是一个可以
  接受的近似，未进一步细化（不是这次审计的重点路径：客户端不太可能真的
  在这两个端点上打错路径）。

---

## 6. 分阶段迁移路径

**这次做的（本次改动已落地）**：
- thinking/reasoning 状态机重写（§1）。
- 三处错误处理/形状修复（§4）。

**低风险，值得下一次顺手做，但这次没做**（避免范围扩张）：
- 把 `formats/openai.py::build_sse_chunks` / `formats/anthropic.py::
  build_sse_events` 标记废弃或删除——它们是死代码，留着只会误导人（见
  §2.2 L2）。删除前需要确认 `tests/test_regression_unit.py` 里引用它们的
  测试是否要一并删除或改写（那个文件不在本次任务的文件归属范围内）。
- `<usage>` 流式/非流式语义统一（§5.5 缺口 2）——工作量小，但需要决定
  统一到哪一种语义，涉及产品判断，留给下次一起排。

**需要单独排期，不建议顺手做**：
- L1-L3（流式序列化抽成协议无关的 adapter）：需要真实的第三个协议需求
  出现后再抽象，现在做是投机性通用化。
- 结构化输出真正生效（§5.1）：核心解码循环改动，需要性能/正确性双重验证
  （xgrammar 掩码在 CUDA Graph 路径下的兼容性也要确认）。
- `stop` 序列实现（§5.2）：同样是 engine 解码循环改动。
- `seed` 语义修正（§5.4）：`runtime/backends/laguna.py` 改动，且当前
  "至少确定性正确"，收益/风险比不如前两项紧迫。
- Track A 落地后，`SERVER_THINKING_CAPABLE`/`SERVER_MODEL_BACKEND` 等
  硬编码常量应该改成从 `ModelSpec`/`ModelRegistry` 读取（§2.2 L6）——
  这是模型抽象层的工作，不是协议层的工作，协议层这边的接口
  （`StreamProcessor(thinking_capable=...)`）已经是"按 backend 能力"设计的，
  不需要再改。

## 7. N1/N2/N3 落地（`fix/t0b-api`，2026-08-01）

在 §5 审计基础上，这一批把 N1/N2/N3 三个缺口都处理完毕。文件归属边界
（另有两个 agent 并行改 `runtime/backends/**`/CUDA Graph 相关代码）不变，
所以三条都在不碰那些文件的前提下完成——其中 N1 的结论是"确认接不通，
选择响亮失败"，N2/N3 是"确认能在 `server/**` + `runtime/sampling.py` 范围
内接通，已接通"。

### 7.1 N1（结构化输出）—— 选择"响亮失败"，不是"接上"

深入到具体调用点后，结论比 §5.1 当时更细：`GrammarState.apply_mask()`/
`apply_mask_batch()` 本身逻辑没问题（bitmask 解包、掩码应用都是对的），
真正的障碍是**这条解码循环里根本没有可用的掩码注入点**，覆盖到会实际生效
的路径：

- `runtime/backends/laguna.py::prefill_chunked_begin`/`_forward` 计算的
  admission 阶段 anchor token（每个请求的第一个 token，不分是否请求了
  `response_format`）是裸的无约束 `argmax`，连 `SamplingParams` 都没有传进去。
- `decode_batch_sampled` 里 CUDA Graph 重放路径把贪心 argmax 直接烤进了
  已捕获的 graph，没有逐 token 的 logits 张量可掩码。
- 同一函数里 eager 路径自己的 `if params.is_greedy: argmax(...)` 分支
  直接绕过了 `sample_from_logits`——这是本模块唯一可能钩进去的缝。
- 本运行时未显式指定 `temperature` 时默认值是 `0.0`（贪心），也就是说
  "给我保证的 JSON"这种最常见请求形态，恰好总是走上面三条不可达路径，
  包括第一个 token。

即使只接通那条唯一可达的缝（`sample_from_logits`，只有 `temperature>0`
时的第 2 个及以后 token 才会经过），默认/常见场景仍然完全不受约束——
这和"完全没接"在实践中几乎没有区别，却会让人误以为已经接上了，是同一类
静默失败换了个位置，不是修复。

**决定**：`server/app.py::_reject_unsupported_response_format` 在
`/v1/chat/completions`（流式 + 非流式）和 `/v1/completions` 三个位置，
请求体里 `response_format.type` 是 `json_object`/`json_schema` 时直接
400（`invalid_request_error`），在真正开始生成前拒绝。Anthropic
`/v1/messages` 协议本身没有 `response_format` 字段，不需要改。

配套清理：`server/engine.py` 里原来那套"创建 `GrammarState`、每 token
`accept()`、`classify_decode_slots` 的 `grammar_slots`"全部移除——response_format
被拒绝在 API 层之后，那套代码永远不会被触发,继续留着只是另一种形式的
"看起来接上了"。`classify_decode_slots` 函数签名保留 `grammar_slots`
参数（`tests/test_laguna_server_integration.py` 仍覆盖，调用处永远传
`[]`），给以后真正接通留一个已测试的钩子。`runtime/structured_output.py`
不删——`GrammarState`/`ResponseFormat`/bitmask 解包都是对的，模块顶部
docstring 记录了现在为什么没接、接通需要什么条件。

**可推翻的条件**：有人在 `runtime/backends/laguna.py` 里加一个从
admission 阶段就能拿到的掩码钩子，并解决 CUDA Graph 重放下"贪心已经烤进
graph"的架构冲突（关掉该批次的 CG 重放，或者把逐 token 变化的 bitmask
做成 graph 的一个输入）。在此之前不要重新"顺手接上"§5.1 提到的窄路径。

### 7.2 N2（`stop` / `stop_sequences`）—— 已接通

思路：stop 是**文本**层面的语义，但解码是**token**层面的——沿用
`server/formats/stream.py::_trim_ambiguous_tail` 已有的"扣住可能还会变成
标记的尾部字节"思路，泛化到 N 个候选串（新模块
`server/formats/stop.py::find_earliest_stop_match`/
`trim_ambiguous_stop_tail`），再在 `server/engine.py` 里给每个配置了
`stop` 的 slot 维护一个私有的 `StreamProcessor`（`_stop_check_token`/
`_flush_stop_pending`/`_drop_stop_pending_from_committed`），逐 token 判断：

- 完整匹配 → 截断 `committed_tokens`（连同 `logprobs_acc`），以 `"stop"`
  结束，Anthropic 侧记录命中的具体串。
- 是某个候选串的严格前缀（歧义，可能被后续 token 补全）→ 扣住，不进
  stream channel，也不算进最终结果。
- 排除歧义后确认安全 → 一次性 flush 给 stream channel（顺序不变，从未
  出现"先发后撤"）。

**与 reasoning 状态机的交互**：只对 content 生效，不对 reasoning 生效——
私有 tracker 复用 `StreamProcessor.thinking_done`；只要还在 reasoning 阶段
（`thinking_done=False`），token 立即 flush（不因为 stop 逻辑给 reasoning
显示增加延迟），完全不进入 stop 匹配。理由：OpenAI 真实 API 的
`reasoning_content`/思维链内容不受 `stop` 截断，这是主流实现的行为；本次
选择跟随这个约定而不是自创语义。

**覆盖的解码路径**：贪心 MTP 验证/提交批量路径（`mtp_verify_and_commit_batch`，
一轮可提交多个 token，逐 token 检查、命中即截断丢弃该轮剩余草稿）、
自回归采样路径（`decode_batch_sampled`，逐 token）、admission 阶段的
anchor token（每个请求的第一个 token，同样跑一遍检查，且必须无条件喂给
tracker——否则后续匹配会静默漏掉这个 token 的贡献）。**未覆盖**：
CUDA Graph 贪心重放路径——但这不是短板，是路由规则的必然结果：
`decode_batch_sampled` 的 CG 重放要求整批 `all(p.is_greedy)`且无
logprobs，配了 `stop_sequences` 的 slot 走的是同一份逐 token 检查代码，
不依赖 CG 是否命中；CG 命中与否只影响"贪心怎么拿到 token"，不影响"拿到
token 之后怎么判断 stop"。

**未精确处理、已知且可接受的近似**：一个 pending 批次里，如果匹配点之前
还夹着"安全"文本（同一批被扣住的 token 里，前面部分其实不歧义），本次
选择整批一起扣住/一起 flush，不去拆某个 token 内部的字符边界——因为
stream channel 是按 token id 传的，拆到字符级需要把文字重新编码回
token，不可靠。这只影响**延迟**（安全文本多等一轮才 flush），从不导致
泄漏（stop 序列本身或其后内容永不发出）。同理，命中匹配截断
`committed_tokens`/`logprobs_acc` 时，如果 pending 缓冲区里第一个 token
恰好是 anchor（没有对应的 logprobs 条目），可能多裁一条 `logprobs_acc`——
方向上安全（从不残留错位数据，最多少算一两条）。

### 7.3 N3（`seed`）—— 已接通，不改 `runtime/backends/laguna.py`

根因和 §5.4 一致：`runtime/backends/laguna.py` 三个采样调用点
（`prefill_sampled`/`decode_sampled`/`decode_batch_sampled`）每次都是
`gen = make_generator(params.seed); sample_from_logits(..., generator=gen)`，
`seed` 是裸 `int` 时 `make_generator` 每次都新建一个 generator 并
`manual_seed(seed)`——同一个 seed 每一步都被"重置"回同一个初始状态,而不是
沿着一条流前进。

修法不改调用点（调用点在 `runtime/backends/laguna.py`，不在本次范围内），
改**调用点传进去的值本身**：新增 `runtime/sampling.py::PersistentSeed`，
包一层薄壳在 `.seed` 字段上——它是一个持有"惰性创建的 `torch.Generator`"
的普通对象，`make_generator()` 识别到 `PersistentSeed` 时直接返回它内部
缓存的 generator（第一次调用才真正创建+播种），不重新播种。因为
`GenerationRequest.sampling_params`（因此 `.seed`）在一个请求的全部解码轮
之间是**同一个对象**，`runtime/backends/laguna.py` 里那三处调用点在同一个
请求的不同轮次里，拿到的是同一个 `PersistentSeed` 实例，从而拿到同一个、
持续前进的 generator——完全不需要改调用点的代码，因为调用点做的事
(`make_generator(params.seed)`) 没变，变的是 `params.seed` 这个值自己的
身份和行为。

`server/app.py::_build_sampling_params` 是唯一的构造点：每个 HTTP 请求
`seed is not None` 时新建一个 `PersistentSeed(seed)` 实例（按对象身份而非
数值区分，两个恰好都传 `seed=42` 的并发请求各自拿到独立实例、独立
generator，不会互相污染随机流）。

**贪心位精确不受影响**：`SamplingParams.is_greedy`（`temperature<=0`）
分支在 `sample_from_logits`/`decode_batch_sampled` 里都是直接
`logits.argmax(...)`，完全不触碰 `make_generator`/`seed`——`PersistentSeed`
对贪心路径是彻底惰性的，从未被实例化的 generator 也就没有任何东西可以
影响。`tests/test_sampling.py::TestPersistentSeed::
test_greedy_path_never_touches_seed` 直接断言这一点。
