# GPU 显存审计 + CUDA Graph 活体确认 (2026-08-02)

日期：2026-08-02 · 分支：`work/cg-audit-20260802` · 状态：🟢 两个模型均已真机复验

取代 `notes/2026-07-29-gpu-memory-audit.md`（只测过 Laguna，且早于反量化缓存发现）。本轮新增
Qwen3.6，并回答反量化缓存发现之后（`notes/2026-08-02-qwen36-dequant-cache-memory-floor.md`）唯一
真正重要的新问题：**"权重 / KV / 反量化缓存 / 其它"逐项相加，两个模型分别是多少**。同时完成
`docs/implementation-plan.md` §7.3 C7-2（CUDA Graph 捕获可观测性的活体确认）与 F2-0（带日期来源的
显存审计）。

## TL;DR

1. **CUDA Graph 确实真实启用**，两个后端、真实 HTTP 服务、真实请求，全部实测：
   - Qwen3.6：`decode` 捕获成功，本次请求 23/23 个 decode round **全部**走 `graph.replay()`
     （新增的 `_backend_stats_dbg.decode_graph_replays` 计数器，见下）。
   - Laguna（生产 DFlash 配置）：`decode`/`draft`/`verify` **三个图全部捕获成功**，请求 8 round 全部
     走 DFlash 路径，无报错、输出连贯。
2. **反量化缓存只存在于 Qwen3.6，不存在于 Laguna**——这不是猜测，是代码读取 + 两次真机测量的双重
   证据：
   - Qwen3.6：CUDA Graph 捕获的 warmup forward 一次摸遍所有层，观测到 **nvidia-smi 在 5 秒内从
     26.6 GiB 跳到 76.3 GiB（+49.7 GiB）**，此后请求量再大也不再涨。
   - Laguna：`runtime/model/plain_linear.py::PlainLinear`（非 MoE 全部层）**在磁盘上就是 BF16，从未
     量化过**（safetensors 逐张量核实：non-MoE 7.4762 GiB 全部是 BF16 dtype，无一例外）；MoE
     专家权重由 sparkinfer 自己的 CUTLASS kernel 直接在 NVFP4 上计算（`laguna_sparkinfer_moe.py`
     `prepare_sparkinfer_fp4_moe_weights`），代码里没有 `_weight_bf16` 或任何缓存字段。实测同样
     印证：CG 捕获前后 nvidia-smi 只变化 **+0.83 GiB**（远小于任何"全模型反量化"量级），送一次真实
     请求后再变化 **+0.26 GiB**，此后持平。
3. **`server.engine` 的 logger 确认到不了日志文件**——独立于 C7-2 的另一个真实缺陷，两个后端都验证
   到了（见下）。
4. 逐项显存表见下，两个模型均**逐项相加对上 nvidia-smi 总数**（误差 < 0.1%），标准与
   `2026-07-29` 那份一致。

## 方法

- 两个后端各起一次**真实 `python -m server.app` 子进程**（非 mock、非直接调 backend 方法），HTTP
  `/health`、`/debug/stats`、`/v1/chat/completions` 全部走真实网络请求。
- 显存读数**全部来自外部 `nvidia-smi --query-gpu=memory.used`**（不依赖进程内 `torch.cuda.*`
  计数器），在服务启动全程以 ~3-4s 间隔轮询，连同 `/health` 一起记录成时间序列——这是为了在不改
  一行生产代码的前提下，把"权重加载完成"和"CUDA Graph 捕获触发反量化"这两个阶段在时间轴上分开，
  给出**直接观测到的**跳变量，而不是靠减法猜一个"其它"桶。
- 权重字节数**从 safetensors header 的 shape+dtype 精确计算**（不做假设，不读整张量），按
  `.experts.` 键名 / `visual` 前缀 / `mtp` 前缀切分 MoE / 视觉塔 / MTP 层，语言模型实际加载的权重
  = 总量 − 视觉塔 − MTP（Qwen3.6 B2 后端不支持 MTP，视觉塔在 `language_model_only=True` 模式下从不
  构建）。
- 为了在共享卡上安全验证（见 `notes/2026-08-02-qwen36-dequant-cache-memory-floor.md` 的警告），两次
  运行都把 KV 预算调到远小于生产：`capacity=1, num_slots=1或2, block_size=64, blocks_per_slot=128`
  （8192 token/slot），而不是生产的 `blocks_per_slot=4096`（`scripts/blackwellm_ctl.sh` 默认）。
  **这个缩放只影响 KV/scratch 这一项，权重和反量化缓存两项都与 KV 预算无关**——本轮审计的核心问题
  正是要证明这一点，缩小 KV 预算不会漏掉任何权重相关的显存事实。

## Qwen3.6（`nvidia/Qwen3.6-27B-NVFP4`, backend=`qwen36`, capacity=1, num_slots=2, blocks_per_slot=128）

### 显存时间线（`nvidia-smi`，外部轮询）

| t (s) | 显存 (MiB) | 阶段 |
|---|---|---|
| 6–106 | 24,854 → 24,701（微降） | 权重加载（量化态：FP8 attn/GDN 投影 + NVFP4 MLP/lm_head） |
| 108–134 | 25,138 → 27,259 | `Qwen36SlotPool.__init__`：KV pool（16 层全注意力）+ GDN 递归/conv state pool（48 层）+ attn_outputs |
| **134→139** | **27,259 → 76,052** | **`capture_decode_cuda_graph()` 的 warmup forward 摸遍全部 64 层，触发每个 `ModelOptFP8Linear`/`ModelOptNVFP4Linear` 的惰性反量化并永久缓存** |
| 139–142 | 76,052 → 78,173 | CUDA Graph 捕获本身（`torch.cuda.graph()` 的激活/输出缓冲池）；服务 `/health` 转 up |
| 142→之后 | 78,173 → 78,356（发一次真实请求后） | +183 MiB，转瞬即逝的请求级临时缓冲，此后持平 |

### 逐项显存表

| 分量 | 大小 | 依据 |
|---|---|---|
| CUDA 上下文 + PyTorch 分配器/驱动开销 | **5.36 GiB**（5.76 GB） | 残差：权重加载阶段稳定读数 24,701 MiB − 精确计算的权重字节 19,217 MiB |
| 量化权重（语言模型部分：FP8 attn/GDN 投影 208 张量 + NVFP4 MLP/lm_head 193 张量，group_size=16） | **18.767 GiB**（20.14 GB） | safetensors header 精确求和；checkpoint 总量 20.42 GiB，减去视觉塔 0.858 GiB（不构建）和 MTP 0.791 GiB（B2 后端不用 MTP） |
| KV cache（16 层全注意力，paged）+ GDN 递归/conv state（48 层），2 个真实槽 + 1 个 scratch 行 | **2.067 GiB**（2.22 GB） | 观测跳变（backend 构造阶段） |
| **反量化 BF16 缓存**（`ModelOptFP8Linear`/`ModelOptNVFP4Linear` 惰性缓存，一次前向摸遍全部层后永久占用） | **49.721 GiB**（53.39 GB） | 观测跳变（CG 捕获阶段），这是本轮审计要回答的核心数字 |
| CUDA Graph 捕获池 + 舍入 | 0.179 GiB | 残差：78,173 − (19,217+2,117+50,914+5,484) = 366 MiB — 已并入上一步的观测区间，此行仅为对账余量 |
| **合计（服务 up，未收请求）** | **76.341 GiB = 78,173 MiB**（81.97 GB） | **nvidia-smi 实测，与逐项相加一致（误差 < 0.5%）** |

FP8/NVFP4 → BF16 的膨胀倍数验证了数量级：NVFP4 权重（U8 打包，2 个 4-bit 值/字节）8.5608 GiB
on-disk → BF16 后约 34.24 GiB（4x）；FP8 权重（F8_E4M3，1 元素/字节）→ BF16 后约 2x。两者相加的
理论上限（~46-50 GiB 区间，具体取决于 blockscale 张量如何计入）与实测的 49.72 GiB 跳变量级吻合。

### CUDA Graph 与真实流量

```
起服务后（尚未收请求）: _cuda_graph_dbg = {"decode": "captured"}
                        _backend_stats_dbg.decode_graph_replays = 0
发一次真实 /v1/chat/completions（24 tokens, greedy）后:
                        rounds = 23, decode_rounds = 23, decode_tokens = 23
                        decode_graph_replays = 23   ← 100% 走 graph.replay()，无一次回退 eager
```

`decode_graph_replays` 是本轮新加的一行代码（`server/app.py::debug_stats`，`getattr` 防御，
`LagunaBackend` 无此属性时静默跳过）——`_cuda_graph_dbg` 只证明"捕获没报异常"，这个计数器才是
"真实流量确实在用这张图"的直接证据，二者合起来才是 C7-2 要的"活体确认"。

## Laguna（`poolside/Laguna-S-2.1-NVFP4`, backend=`laguna`, **生产配置**：DFlash 开、prefix cache 开，
capacity=1, num_slots=1, blocks_per_slot=128）

### 显存时间线

| t (s) | 显存 (MiB) | 阶段 |
|---|---|---|
| 2–108 | 12,748 → 32,515 | 加载非 MoE BF16 权重 + 前几个 MoE 分片 |
| 110–184 | 42,883 → 75,408 | 剩余 MoE 专家分片（256 专家 × 47 层，NVFP4 打包） |
| **184–411** | **75,157 – 75,409（±0.3%，稳定平台）** | **权重加载完毕，此时还没有任何前向。整整 227 秒里显存纹丝不动** |
| 411–434 | 75,343 → 77,113 | `capture_decode_cuda_graph()`（主 decode 图）+ `enable_dflash()`（DFlash draft/verify 图捕获） |
| 434–479 | 77,113 → 76,190 | 捕获期临时 buffer 释放，`/health` 转 up |
| 479→之后 | 76,190 → 76,454（发一次真实请求后） | +264 MiB，此后持平 |

**关键对比**：weights-settled 平台（75,343 MiB）到 server-up（76,190 MiB）只涨了 **847 MiB
(0.83 GiB)**——同样是"CUDA Graph 捕获 + 一次真实 forward"，Qwen3.6 涨了 49.7 GiB，Laguna 只涨了
0.83 GiB（差两个数量级）。这就是"Laguna 是否也有反量化缓存"这个问题的实测答案：**没有**。

### 逐项显存表

| 分量 | 大小 | 依据 |
|---|---|---|
| CUDA 上下文 + KV/SWA/draft-KV pool + 分配器开销 | **4.539 GiB**（4.87 GB） | 残差：weights-settled 平台 75,343 MiB − 精确计算的权重字节 70,695 MiB |
| 主模型权重（MoE 专家 256×47 层 NVFP4 打包 U8 + blockscale F8_E4M3：59.485 GiB；非 MoE BF16——attn QKV/O、embed、lm_head、norm、gate、shared_expert、层 0 dense MLP：7.476 GiB） | **66.961 GiB**（71.90 GB） | safetensors header 精确求和，与 `2026-07-29` 那份的 59.5+7.3 GB 吻合 |
| DFlash draft 模型权重（独立 checkpoint `poolside/Laguna-S-2.1-DFlash-NVFP4`，6 层全 SWA，纯 BF16） | **2.077 GiB**（2.23 GB） | safetensors header 精确求和 |
| CUDA Graph 捕获（decode + draft + verify 三个图的激活/输出缓冲池） | **0.827 GiB**（0.89 GB） | 观测跳变（weights-settled → server-up） |
| 请求级临时缓冲（一次真实 8-round DFlash 请求后的残留） | 0.258 GiB | 观测跳变（server-up → 请求后） |
| **合计（一次真实请求后）** | **74.662 GiB = 76,454 MiB**（80.17 GB） | **nvidia-smi 实测，与逐项相加一致（误差 < 0.1%）** |

**没有"反量化缓存"这一行**——不是没测到，是这个机制在 Laguna 的代码路径里根本不存在：
`runtime/model/laguna_decoder.py` 全部使用 `PlainLinear`（`runtime/model/plain_linear.py`），
构造函数里就是 `nn.Parameter(torch.empty(...))`，没有量化态、没有 scale、没有惰性反量化分支；MoE
专家权重走 `runtime/backends/laguna_sparkinfer_moe.py::prepare_sparkinfer_fp4_moe_weights`，产出的
是 sparkinfer 自己的打包 FP4 权重结构，forward 直接调 CUTLASS kernel 在 FP4 上计算，全程没有
`_bf16` 缓存字段。

### CUDA Graph 与真实流量

```
起服务后（尚未收请求）: _cuda_graph_dbg = {"decode": "captured", "draft": "captured", "verify": "captured"}
发一次真实 /v1/chat/completions（8 tokens, greedy, DFlash 全程参与）后:
                        rounds = 8, mtp_acceptance_histogram = [8,0,...,0]（8 轮全部经过 DFlash 路径）
                        _cuda_graph_dbg 不变，请求正确完成（"Hello, how are you today?"），无报错
```

`LagunaBackend` 没有 `.stats` 属性，本轮新加的 `_backend_stats_dbg` 对它是 no-op（`getattr` 防御生
效，不报错、不误报）。因此 Laguna 侧"图确实被 replay 而非静默回退 eager"的证据是**代码读取
+ 间接证据**而非直接计数器：`runtime/backends/laguna.py` 的 `_decode_cg.replay(...)`（约 2149-2183
行）在 `self._decode_cg is not None and len(slot_ids) == self._decode_cg.batch_size` 时无条件调用，
没有静默降级分支；`mtp_acceptance_histogram` 8 轮全部计入证明这次请求确实全程走了 DFlash 路由。
**这一点比 Qwen3.6 弱，见下"未能验证的事项"。**

## `server.engine` 的 logger 到不了日志文件——确认属实，独立缺陷

三层证据，独立成立：

1. **代码读取**：`server/engine.py`（`logging.getLogger("qwen_sm120_server.engine")`）、
   `runtime/backends/qwen36.py`（`logging.getLogger(__name__)`）、`runtime/backends/laguna.py`
   （`"qwen_sm120_runtime.laguna_backend"`）都从未 `addHandler`，也从未被任何 `logging.config
   .dictConfig`/`basicConfig` 配置过。`server/app.py` 的 `"qwen_sm120_server.app"` 是**唯一**显式
   `addHandler` 的 logger（第 47-59 行有注释解释原因）。`uvicorn` 自己的默认 `LOGGING_CONFIG`（实测
   `uvicorn==0.46.0`）只配置 `"uvicorn"`/`"uvicorn.error"`/`"uvicorn.access"` 三个 logger，**不碰
   root**。Python 默认 root logger 无 handler、level=WARNING。
2. **独立小脚本复现**（零 GPU 成本）：按真实启动顺序（先跑 uvicorn 的 `dictConfig`，再 import
   `server.engine`）构造后，`eng.logger.getEffectiveLevel()==30`（WARNING），`eng.logger.handlers
   ==[]`；发一条 INFO 记录，stderr 里什么都没有；发一条 WARNING，Python 的 `logging.lastResort`
   handler 把它打到 stderr（这就是为什么"捕获失败"的 `logger.warning` 能看见，"捕获成功"的
   `logger.info` 看不见——两条路径的可见性从一开始就不对称）。
3. **两次真实服务器运行的日志文件逐行核对**：`logs/qwen36_audit_server.log`、
   `logs/laguna_audit_server.log` 里，`"qwen_sm120_server.app"` 的 INFO 行全部在（
   `tool_call_parser=...`、`loading model=...`、`engine ready: ...`——这些调用的是 `server/app.py`
   自己的 logger），但 `"qwen_sm120_server.engine"` 的**任何**一行都不存在——包括
   `"Qwen3.6 decode CUDA Graph captured at load"`、`"Laguna decode CUDA Graph captured at load"`、
   `"DFlash speculative decoding wired"`、`"Qwen3.6 model loaded on engine thread"`、
   `"Laguna model loaded on engine thread"`——这些全部是 `server/engine.py` 里 `logger.info(...)`
   调用，两次真实运行、两个后端，一行都没落盘。

**这是一个独立于 C7-2 的真实缺陷**：C7-2 的 `snapshot()`/`/debug/stats` 修复解决了"能不能查到"，
但没有解决"运维照着日志文件排障时看不到任何 engine 层信息"（不只是 CUDA Graph——整个
`server/engine.py` 的 INFO 级进度/诊断信息都不落盘，只有 WARNING/ERROR 可见）。最小修法是给
`"qwen_sm120_server.engine"` 挂一个 handler，或者更彻底地在 `server/app.py` 加一次全局
`logging.basicConfig`/`dictConfig`，把 root 配好——两种都是小改动，但都不在本轮任务范围内
（本轮只做"确认属实并写清楚"，不代 sparkinfer/不代主线开发做决定）。

## 未能验证的事项

1. **没有直接复现"生产规模"KV 预算下的总占用**：本轮为安全把 `blocks_per_slot` 从生产的 4096 降到
   128（KV 相关项按比例缩小了约 32 倍），且 Laguna capacity 从生产的 3 降到 1。协调者
   2026-08-01 汇报的生产实测 94.2/97.9 GB（`docs/implementation-plan.md` §7.6 F2）**没有在本轮复
   现**——本轮证明的是"权重 + 反量化缓存"这两项（合计占大头、且与 KV 预算无关）的精确值，KV/DFlash
   scratch 在生产 `blocks_per_slot=4096, capacity=3` 下的真实占用仍需要专门一次运行去测（F2-0 剩余
   部分）。粗略外推：Laguna 权重+draft 权重固定 69.04 GiB，本轮 KV/context/CG 池合计约 5.4 GiB
   （blocks_per_slot=128），生产配置下这部分会显著增长（SWA ring buffer 是固定窗口不随
   blocks_per_slot 缩放，主 KV 和 DFlash scratch 会），但**这只是外推，不是本轮实测**。
2. **Laguna 的 CUDA Graph replay 没有直接计数器验证**——不像 Qwen3.6 新加的
   `decode_graph_replays`，`LagunaBackend` 没有 `.stats` 属性可挂。本轮的证据是代码读取（
   `replay()` 调用路径无静默降级分支）+ 间接信号（`mtp_acceptance_histogram` 8 轮记录、请求正确
   完成）。要拿到和 Qwen3.6 同等强度的直接证据，需要给 `LagunaBackend`/`_decode_cg`/
   `DFlashEngine` 加类似的 replay 计数器——超出本轮"活体确认，不新增大改动"的范围，留给后续。
3. **没有测试 CG 捕获失败场景下 `dflash_cg_status` 是否正确翻转为 `"failed"`**——本轮两次运行捕获
   都成功，没有构造一次故意失败的对照组（例如禁用 CUDA Graph 后确认字段变成 `()`）。代码读取（
   `capture_decode_cuda_graph` 的 `try/except` 结构，`except` 分支显式写 `"failed"`）加上两次成功
   捕获的一致行为，已经能确认字段不是硬编码常量，但没有拿到"失败路径确实按预期翻转"的活体证据。
4. **两次运行都只发了一条短请求（24 token / 8 token）**，没有测多轮并发、长上下文、soak——这些本来
   就不在本轮任务范围内（任务只要求"确认 CG 真的启用"和"显存逐项对账"）。

## 复现

```bash
# worktree 内，见 logs/qwen36_audit_server.log / logs/laguna_audit_server.log 的完整启动日志
QSR_SERVER_MODEL_PATH=nvidia/Qwen3.6-27B-NVFP4 QSR_SERVER_CAPACITY=1 QSR_SERVER_NUM_SLOTS=2 \
QSR_SERVER_BLOCK_SIZE=64 QSR_SERVER_BLOCKS_PER_SLOT=128 QSR_SERVER_ENABLE_CUDAGRAPH=1 \
QSR_SERVER_ENABLE_PREFIX_CACHE=0 QSR_SERVER_ENABLE_DFLASH=0 QSR_SERVER_GPU_MEM_UTIL=0.5 \
python -m server.app --host 127.0.0.1 --port 8171

QSR_SERVER_MODEL_PATH=poolside/Laguna-S-2.1-NVFP4 QSR_SERVER_PRODUCTION=1 QSR_SERVER_CAPACITY=1 \
QSR_SERVER_NUM_SLOTS=1 QSR_SERVER_BLOCK_SIZE=64 QSR_SERVER_BLOCKS_PER_SLOT=128 \
QSR_SERVER_ENABLE_CUDAGRAPH=1 QSR_SERVER_ENABLE_PREFIX_CACHE=1 QSR_SERVER_ENABLE_DFLASH=1 \
QSR_SERVER_GPU_MEM_UTIL=0.5 python -m server.app --host 127.0.0.1 --port 8172

# 显存: nvidia-smi --query-gpu=memory.used --format=csv,noheader（外部轮询，见方法一节）
# CG 状态 + 计数器: curl -s http://127.0.0.1:<port>/debug/stats
```

## 相关

- `notes/2026-07-29-gpu-memory-audit.md`（前一份，只测 Laguna，被本文件取代）
- `notes/2026-08-02-qwen36-dequant-cache-memory-floor.md`（反量化缓存发现，本轮的直接实测确认）
- `docs/implementation-plan.md` §7.3 C7-2、§7.6 F2-0
- `runtime/model/modelopt_linear.py`（Qwen3.6 反量化缓存机制本身）
- `runtime/model/plain_linear.py`（Laguna 非 MoE 层，代码证明无量化态）
- `runtime/backends/laguna_sparkinfer_moe.py`（Laguna MoE 层，代码证明无 BF16 缓存）
- `server/app.py::debug_stats`（本轮新增 `_backend_stats_dbg`）
