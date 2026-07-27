# 独立探测系统(bfprobe)——设计、代码核验与实施规划

日期:2026-07-27
状态:设计定稿待批准,尚未实施
方法:全程零 GPU 操作。所有结论来自源码核验(附 `文件:行号`)与已有笔记的实测数据。

---

## 0. 为什么这件事优先级最高

**硬约束:只有一块 GPU,任何工作都无法并行。**

这条约束决定了一切。它意味着:

> **单位时间的进展 = 每次 GPU 运行的信息产出量 ÷ 单次运行耗时**

分母已经压不动了(权重加载 26.76s + draft 3.97s,见 `our_server.log:39,49`,再加 CUDA Graph 捕获)。**唯一还有数量级空间的是分子。**

现状是分子极小:一次运行通常只回答一个是非题("接受率是多少""这个开关有没有用"),答完就扔。想多知道一件事,就得改脚本再跑一次。`benchmarks/` 下 144 个一次性诊断脚本、32710 行,就是这个模式的物证。

更糟的是,**跑不起第二次**。`notes/2026-07-27-dflash-concurrency-handoff.md` 里那个「270 秒的诡异延迟」至今「没有验证清楚」,原因就是它是偶发的、当时没有记录、事后无法复现。

所以探测系统的设计目标不是"能查问题",而是:

> **让每一次 GPU 运行,默认就把这次运行里发生的一切都记下来。**
> 不是「出问题了才打开抓取」,而是「永远开着,出问题时数据已经在了」。

---

## 1. 代码事实核验

设计里每一条技术前提,都必须对着真实代码验过。以下是核验结果。

### 1.1 ✅ 已证实可行(而且是已在生产路径运行的机制)

**(A) 从「已捕获的 CUDA Graph」里取出中间张量 —— 这个机制他们已经在跑**

`runtime/backends/laguna_cuda_graph.py:324-328`:
```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    self._logits = self._build_metadata_and_forward()
    self._input_ids[0] = self._logits[0].argmax(dim=-1).to(torch.long)
```
而 `_build_metadata_and_forward` 内部(`:279-283`):
```python
if isinstance(result, tuple):
    hidden_states, self._aux_hidden_states = result
```
`replay_with_aux()`(`:386-394`、`:766-774`)在 `.replay()` 之后直接返回 `self._aux_hidden_states`。

**并且这条路径就是生产热路径** —— `laguna_dflash.py:1403`:
```python
verify_logits, verify_aux = self._verify_cg.replay_with_aux(slot, verify_tokens, kv_len)
```

**结论:「图内产生中间张量 → 图外读出」不是我要发明的新机制,是他们已经验证过、天天在跑的机制。探针系统只是把它一般化。** 这一条极大降低了整个方案的技术风险。

原理:图内张量地址在捕获时固定,每次 replay 把新值写到同一批地址,图外按固定地址读即可。

**(B) 从图内写入「图外预分配」的张量 —— 也已证实**

`laguna_cuda_graph.py:326` 写的 `self._input_ids` 是在 `__init__`(`:56`)里、捕获之前分配的:
```python
self._input_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
```
**这正是探针 memcpy 需要的模式:目的地缓冲区在捕获前预分配,捕获时把 copy 烘焙进图,replay 时自动写入。**

**(C) 模型是真实的 nn.Module 树,可挂 forward hook**

`laguna_cuda_graph.py:275` 走的是 `backend.model.forward(input_ids, positions)`,且 `set_forward_context(..., skip_compiled=True)`(`:273`)绕开了编译路径。`oracle/capture_hooks.py::ForwardCapture` 已经在用 forward hook 采集模块输出。

⚠ 关键细节:**在 CUDA Graph 捕获期间,forward hook 的 Python 代码只在捕获那一次执行**;它 enqueue 的 copy 算子会被烘焙进图,之后每次 replay 自动执行。这正是我们要的语义,但必须明确写进实现约定,否则容易误以为 hook 每次 replay 都会跑。

**(D) MoE 路由决策是一个普通 torch 张量,就在 Python 层 —— 这是全代码库最有价值的探针点**

`runtime/backends/laguna.py:530-546`:
```python
def _patched_forward(hidden_states: torch.Tensor) -> torch.Tensor:
    orig_shape = hidden_states.shape
    hs = hidden_states.view(-1, hidden_states.shape[-1])
    router_logits, _ = moe_mod.gate(hs)            # ← 探针点 P-ROUTER-LOGITS
    router_logits = router_logits.float()
    if _softcap > 0:
        router_logits = torch.tanh(router_logits / _softcap) * _softcap
    topk_weights, topk_ids = fused_topk_bias(...)  # ← 探针点 P-TOPK(最高价值)
    routed_out = _si_layer.forward(hs, topk_ids, topk_weights)
    ...
```

**为什么这是最高价值的探针点**:当前头号未解问题是「同一 prompt,vLLM 接受率 100%,我们 68.7%」(`notes/2026-07-27-acceptance-rate-gap-vllm-vs-ours-same-prompt.md`),头号假设是「我们用 sparkinfer MoE,vLLM 用 FlashInfer CUTLASS,数值路径不同」。

如果**两边路由到的专家 id 就不一样**,那根本不是浮点舍入,是路由问题 —— 一发就能分辨。而 `topk_ids` 在这里已经是现成张量,抓它的成本见 §4(60 KB/轮,可忽略)。

**(E) DFlash 一轮的结构清晰,探针点明确**

`laguna_dflash.py:1371-1462`(`dflash_round`)逐行标注:

| 行 | 内容 | 探针 |
|---|---|---|
| 1402-1407 | `if self._verify_cg is not None:` CG / eager 二选一 | **P-PATH-VERIFY**(那个「静默掉 eager」就是这个 if) |
| 1409 | `all_argmax = verify_logits[...].argmax(dim=-1).tolist()` | 见下方 ⭐ |
| 1410 | `_verify_only_accept_reject(all_argmax, draft_tokens, bonus_token)` | **P-ACCEPT**(`reject_position` 在这里可无成本导出) |
| 1426-1442 | aux → draft KV 预计算 | P-AUX |
| 1444-1446 | `slot_kv_len += context_count`;`committed_tokens.append` | **不变量检查点** |
| 1449-1452 | `if self._draft_cg is not None:` 又一个 CG / eager 分叉 | **P-PATH-DRAFT** |

⭐ **重要发现:第 1409 行的 `.tolist()` 已经是一次 device→host 同步。** 也就是说生产路径每轮**本来就有**一次同步点。在这个点之后读取几个宿主端标量,**额外成本为零**。T0 层探针可以完全免费。

### 1.2 ❌ 需要修正的两条(我上一轮说的不完全对)

**(F) 修正:「预留 2GB GPU 环形缓冲」—— 显存预算比我说的紧,必须按配置分档**

依据 `notes/2026-07-22-laguna-l0-memory-budget.md` 的实测预算合同:

| 场景 | 权重+KV+预留 | 剩余 | 减去 draft 模型 2.1 GiB 后 | 探针环可用 |
|---|---:|---:|---:|---|
| 2 槽 × 200K | 80.4 GiB | 15.2 GiB | ~13.1 GiB | ✅ 宽裕 |
| 2 槽 × 256K | 83.0 GiB | 12.6 GiB | ~10.5 GiB | ✅ 宽裕 |
| 4 槽 × 128K | 83.1 GiB | 12.5 GiB | ~10.4 GiB | ✅ 宽裕 |
| 4 槽 × 200K | 89.9 GiB | 5.7 GiB | ~3.6 GiB | ⚠ 需限额 |
| 4 槽 × 256K | 95.1 GiB | ~0.5 GiB | **负** | ❌ 必须整体禁用 GPU 环 |

**修正后的约定**:
- 默认 **256 MiB**,而不是 2 GiB
- 通过 `QSR_PROBE_RING_MIB` 配置,硬上限 1 GiB
- **分配前必须检查空闲显存,不足则拒绝分配并明确告警,绝不静默降级或 OOM**
- 显存预算必须作为一个显式条目写进 L0 预算合同,不能偷偷占用

**(G) 修正:「环形缓冲的轮转可以在图内完成」—— 不行**

图内张量地址在捕获时**固定**。同一个图每次 replay 都写到**同一个地址**。所以「写指针每轮前进、覆盖最旧」这个环形语义,**在图内无法实现**(除非用设备端索引 + scatter kernel,或改 graph exec node params —— 前者要多一次 kernel launch,后者 torch 没有干净的接口)。

**修正后的正确设计(而且和 (A) 的既有机制完全一致)**:

```
图内:  探针 copy → 固定的 staging buffer(双缓冲,地址捕获时固定)
       ↓  replay 结束
图外:  Python 侧把 staging → ring[write_pos],write_pos 每轮前进
       ↓  侧流 + event 定序,与下一轮 replay 重叠
```

这就是 `replay_with_aux` 已经在做的事情的推广 —— **零新机制风险**。双缓冲是为了让第 N+1 轮的 replay 不会覆盖第 N 轮还没排完的 staging。

---

## 2. 三条设计原则

> **1. 写侧极笨,读侧极聪明。** 热路径只做「把 N 字节 copy 到固定地址」。不解析、不格式化、不聚合、不判断。所有语义离线还原。
>
> **2. 探针在数据路径上,不在控制路径上。** 探针可以看数据,但它的成败**绝不能影响引擎的任何决策**。写满就丢,丢了记计数器。**能卡住被观测系统的探针系统,比没有探针系统更糟。**
>
> **3. 消费者在进程外。** 分析工具崩溃/重写/attach/detach,引擎一无所知;引擎崩溃(他们踩过 ptxas ICE),环里数据还在 —— 这才是黑匣子。

---

## 3. 架构

```
┌─ 引擎进程 ──────────────────────────────────────────────────┐
│                                                              │
│  探针总线 bfprobe.bus —— 唯一的写入 API,全局单例             │
│    emit_scalar(site_id, **fields)      T0,宿主端            │
│    emit_reduction(site_id, tensor)     T1,GPU 归约          │
│    emit_tensor(site_id, tensor)        T2,GPU copy          │
│         ↓                                                    │
│  ┌────────────┬──────────────────┬────────────────────────┐ │
│  │ T0 宿主环  │ T1 GPU 签名环    │ T2 GPU staging → 张量环 │ │
│  │ numpy 列存 │ 48×4×32B/轮      │ 双缓冲 + 图外轮转       │ │
│  └────────────┴──────────────────┴────────────────────────┘ │
│         ↓ 侧流 + event 定序,不阻塞主循环                     │
│  排水线程 → pinned host 环 → /dev/shm 共享内存环             │
│    · 引擎线程完全不参与                                       │
│    · 跟不上就丢并计数,绝不阻塞                                │
└─────────────────────↓────────────────────────────────────────┘
   ┌─ 独立消费进程(可随时启停/崩溃/重写)────────────────────┐
   │  bf probe watch   实时生命体征 TUI                       │
   │  bf probe dump    落盘为 run artifact                    │
   │  bf probe scan    离线扫描 + 触发判定                     │
   │  离线解码:靠版本化探针表还原语义(NanoLog 思路)          │
   └──────────────────────────────────────────────────────────┘
```

**关键点**:T1/T2 的环在**显存**里,不在宿主内存里。数据先在显存落地(µs 级),往外搬是排水线程的事,与引擎主循环彻底解耦。

---

## 4. 探针清单与成本预算(真实参数)

模型参数(`config.json` 实测):48 层(12 full_attention + 36 sliding_attention,窗口 512)、47 个 MoE 层(1..47)、hidden 3072、bf16、256 experts top-10、vocab 100352。
一轮 DFlash:K=15 → 16 个 verify token,实测 44.16 ms/轮。

| 级别 | 探针 | 挂载点 | 每轮字节 | 拷贝耗时 | 占 44.16ms |
|---|---|---|---:|---:|---:|
| **T0** | P-PATH-VERIFY / P-PATH-DRAFT | `laguna_dflash.py:1402,1449` | ~16 B | 0 | 0% |
| **T0** | P-ACCEPT(含 `reject_position`) | `laguna_dflash.py:1409-1410` | ~48 B | **0**(已有同步点) | 0% |
| **T0** | P-BOOKKEEPING(kv_len/committed) | `laguna_dflash.py:1444-1446` | ~24 B | 0 | 0% |
| **T1** | 每层 4 个张量签名(absmax/L2/mean/NaN·Inf 计数) | forward hook | 48×4×32 = **6 KB** | ~0 | ~0% |
| **T2** | **P-TOPK 专家 id + 权重** | `laguna.py:537` | 47×16×10×(4+4) = **60 KB** | ~0.05 µs | 0.0001% |
| **T2** | P-ROUTER-LOGITS | `laguna.py:533` | 47×16×256×4 = **770 KB** | ~0.5 µs | 0.001% |
| **T2** | P-HIDDEN 每层隐状态 | forward hook | 48×16×3072×2 = **4.7 MB** | **~3 µs** | **0.007%** |
| **T2** | P-LOGITS | `laguna_dflash.py:1403` | 16×100352×2 = **3.2 MB** | ~2 µs | 0.005% |

**结论 1:decode 阶段「每层全量抓」的成本是 0.007%,远在测量噪声之下。默认就该常开。**

**结论 2:prefill 抓不起。** 64K prefill 全层隐状态 = 65536 × 48 × 3072 × 2 B = **19.3 GB**。且 `LagunaBackend.prefill_chunked_begin`(`laguna.py:1652-1679`)明确注明 Laguna **没有真正的增量分块**,整个 prompt 一次前向。所以 prefill 只能做 T0 + T1 签名 + 稀疏 token 采样,**这是设计的硬分水岭**。

**环容量**(T2 默认只抓 hidden + routing,不抓 logits → 4.8 MB/轮):

| 环大小 | 可回溯轮数 |
|---|---:|
| 256 MiB(默认) | ~53 轮 |
| 512 MiB | ~106 轮 |
| 1 GiB(上限) | ~213 轮 |

53 轮的预触发深度对「症状前几步找原因」完全够用。

---

## 5. 分级探针 + 预触发冻结

| 级别 | 内容 | 何时开 | 成本 |
|---|---|---|---|
| **T0** 事件 | 轮次/slot/path/accept/reject_position/kv_len | **永远开** | 0 |
| **T1** 签名 | 每张量 absmax / L2 / mean / **NaN·Inf 计数** | **永远开** | ~0 |
| **T2** 全量 | hidden / router_logits / **topk_ids** / logits | 滚动覆盖最近 N 轮 | 0.007%/轮 |

**T1 签名是灵魂**:32 字节给每个中间张量一个指纹。不存张量,就能回答「第几轮、第几层、哪个算子开始不对劲」。

**预触发冻结(逻辑分析仪 / 行车记录仪思路)**:

> 「输出突然变垃圾」时,**原因永远在症状之前几步**。
> 传统做法:发现异常 → 打开抓取 → 重跑。在单 GPU 上这意味着又是几分钟,而且偶发问题未必复现。
> 正确做法**反过来**:T2 一直在写并被覆盖。异常触发时**冻结环**,而不是打开环。
> 于是手上直接有了异常发生**之前** 53 轮的全部中间张量 —— 不需要重跑。

触发器是排水线程里对 T1 签名求值的纯函数,不碰主循环:
```
trigger: any(nan_inf_count > 0)
      or absmax > absmax_baseline * 100
      or accept_rate_window(8) < 0.5
      or round_ms > round_ms_p50 * 5        # ← 那个 270 秒会在这里跳出来
action:  freeze_ring; dump_to_run_artifact; keep_running
```

---

## 6. 正确性保证:探针系统怎么自证无害

一个没人敢信的探针系统等于没有。三个机制,**缺一不可**:

1. **零假设门禁(CI 硬性)**:同一 workload,探针全关 vs 全开,断言
   ① 输出 token **逐位相同** ② 延迟 p50 相对偏差 < 0.5%。
   **探针改变了答案,就是探针坏了。** 这条必须能一键跑,且进 GPU 验证批次。
2. **丢弃可见性**:每条记录带单调 seq,读侧检测空洞并显式报告 `dropped=N`。
   **绝不允许静默丢数据** —— 静默丢数据制造出的假象比原 bug 更难查。
3. **显存预算硬门禁**:分配前检查空闲显存,不足则拒绝并告警,绝不静默降级、绝不 OOM。

---

## 7. 实施规划

### 阶段划分原则

- **GPU 验证批次化**:单 GPU 是最稀缺资源。每个阶段的 GPU 验证需求**攒成一张清单,集中在一次会话里跑完**,而不是零散触发。
- 每个阶段都能独立交付价值,不做"全做完才有用"的设计。
- CPU 可验证的部分尽量前置。

### P0 —— T0 宿主环 + 不变量断言 【进行中】

已由并行 agent 在做(`bfdiag/trace/`、`bfdiag/invariants/`)。已下发扩展位要求:记录头能寻址非标量负载、存储后端可替换、schema 版本号 + 离线解码字典。

**产出**:T0 环、事件 schema、`bf trace show` 面板、不变量断言框架。
**GPU 验证需求**:接入 `dflash_round` 后的零假设检查。

### P1 —— 探针总线 + T0 正式接入生产路径

| 项 | 内容 |
|---|---|
| 交付 | `bfprobe/bus.py`(全局单例,三个 emit API)、探针表 `bfprobe/sites.py`(版本化)、T0 六个探针接入 `dflash_round` |
| 关键约束 | 每个接入点 ≤ 3 行;`QSR_PROBE=0` 时只有一次 if 判断 |
| 验收(CPU) | 探针表 schema 单测;disabled 路径 timeit < 100 ns/轮;离线解码往返一致 |
| **GPU 验证批次 #1** | ① 零假设:探针全关 vs T0 全开,输出 token 逐位相同、p50 偏差 < 0.5% ② `bf trace show` 在真实 64K DFlash 运行上出图 ③ 确认 `reject_position` 分布与聚合 acceptance 一致 |
| 预估 | 3-4 天 + 1 次 GPU 批次 |

### P2 —— T1 签名 + MoE 路由探针 ⭐ 最高价值

**这一阶段直接对准当前头号未解问题(68.7% vs 100%)。**

| 项 | 内容 |
|---|---|
| 交付 | GPU 归约 kernel(absmax/L2/mean/NaN·Inf,融合成一个 kernel);T1 签名环;**P-TOPK / P-ROUTER-LOGITS 探针接入 `laguna.py:533,537`**;`bf probe scan` 离线扫描 |
| 关键设计 | 归约必须是**一个** kernel,不能每个统计量一个 launch;签名写入固定 staging,图外轮转 |
| 验收(CPU) | 归约 kernel 用合成张量对拍 numpy;签名环往返;扫描器在合成数据上定位注入的异常层 |
| **GPU 验证批次 #2** | ① 零假设(T0+T1 全开)② 在 64K 重复短语 workload 上,**导出我们这一侧每层每 token 的 top-10 专家 id** ③ 对照 vLLM 侧同一 prompt 的路由(需要 oracle 侧对等探针,见 P2b)④ 判定:路由是否分叉、在第几层、第几个 token |
| 预估 | 4-5 天 + 1 次 GPU 批次 |

**P2b(可与 P2 并行设计)**:vLLM oracle 侧的对等路由探针。vLLM 的 MoE 走 FlashInfer CUTLASS,路由张量在它那侧的位置需要单独定位。**注意:不能修改 `/home/bot/vllm` 源码**,只能用 forward hook / monkeypatch,和 `oracle/capture_hooks.py` 同样的纪律。

### P3 —— T2 全量 + 预触发冻结

| 项 | 内容 |
|---|---|
| 交付 | GPU staging 双缓冲 + 图外轮转 + 侧流定序;T2 张量环;触发器求值;`freeze` + dump |
| 关键约束 | 显存预算硬门禁(§6.3);默认 256 MiB;`QSR_PROBE_RING_MIB` 可配,硬上限 1 GiB |
| 验收(CPU) | 双缓冲时序逻辑用 fake stream 单测;触发器纯函数单测;显存不足时拒绝分配的路径单测 |
| **GPU 验证批次 #3** | ① 零假设(全部三级全开)② 人为注入 NaN 验证触发-冻结-dump 全链路 ③ 确认冻结出来的是**异常之前**的数据,不是之后的 ④ 显存占用与预算表一致 |
| 预估 | 5-6 天 + 1 次 GPU 批次 |

### P4 —— 进程外消费者

| 项 | 内容 |
|---|---|
| 交付 | `/dev/shm` 共享内存环;`bf probe watch` 实时 TUI;`bf probe dump`;崩溃后残留环的恢复读取 |
| 价值 | 引擎崩溃时数据还在(黑匣子);可以在长跑 benchmark 期间实时盯着而不打扰它 |
| 验收(CPU) | 生产者/消费者用两个进程 + fake 数据端到端;kill -9 生产者后消费者仍能读出完整数据 |
| **GPU 验证批次 #4** | ① 真实运行期间 attach/detach 不影响结果 ② kill -9 引擎后能读出最后 N 轮 |
| 预估 | 3-4 天 + 1 次 GPU 批次 |

### P5 —— 单轮确定性回放 ⭐ 终局

**思路借鉴 `rr` 时间旅行调试。这是把"我需要那个中间张量"从"重跑 3 分钟"变成"200 毫秒"的一步。**

不记录所有中间状态,而是记录**重放一轮所需的最小输入集**:该轮 token ids、slot KV 状态快照引用、RNG seed、路由决策。每轮几十 KB。

然后 `bf replay <run_id> --round 51`:离线、eager、单独重放第 51 轮,任意加 instrumentation、任意逐层 dump,**完全不需要重跑 64K prefill**。

前提是确定性,而**他们已经具备一大半**:sparkinfer `deterministic_output=True`(commit `989723d`)、greedy 采样、固定 seed。

| 验收(CPU) | 回放输入集 schema;重放驱动器用 fake engine 端到端 |
|---|---|
| **GPU 验证批次 #5** | ① 录制真实 64K 运行 ② 重放第 N 轮,断言输出 logits 与原运行**逐位相同** ③ 若不逐位相同,定位非确定性来源(这本身就是有价值的发现) |
| 预估 | 1 周 + 1 次 GPU 批次 |

### 总览

| 阶段 | 内容 | CPU 开发 | GPU 批次 | 累计价值 |
|---|---|---|---|---|
| P0 | T0 环 + 不变量【进行中】 | — | 并入 #1 | 轮次级可见性 |
| P1 | 探针总线 + T0 接入 | 3-4 天 | #1 | 「静默掉 eager」不再静默 |
| **P2** | **T1 签名 + MoE 路由** | **4-5 天** | **#2** | **直击 68.7% 之谜** |
| P3 | T2 全量 + 预触发冻结 | 5-6 天 | #3 | 偶发问题不再需要复现 |
| P4 | 进程外消费者 | 3-4 天 | #4 | 崩溃存活 + 实时盯盘 |
| P5 | 单轮确定性回放 | 1 周 | #5 | 「重跑 3 分钟」→「200 毫秒」 |

**合计约 4-5 周 CPU 开发 + 5 次 GPU 验证批次。**

如果只能做一段,做 **P1 + P2** —— 它们合计约 8 天,直接对准当前最重要的未解问题。

---

## 8. 风险与未决问题

| # | 风险 | 严重性 | 缓解 |
|---|---|---|---|
| 1 | forward hook 在 CG 捕获期只跑一次的语义被误用 | 高 | 写进实现约定 + 一个专门的验证用例 |
| 2 | 探针 copy 改变了 CUDA Graph 的内存池布局,导致捕获失败或行为变化 | 高 | 零假设门禁;探针 buffer 一律**捕获前**分配在池外 |
| 3 | 4 槽 × 256K 配置下显存无空间 | 中 | 硬门禁拒绝分配 + 明确告警;该配置下降级为 T0+T1(仅宿主) |
| 4 | vLLM oracle 侧路由探针的挂载点未定位 | 中 | P2b 单独排查;禁止改 vllm 源码,只用 hook |
| 5 | 归约 kernel 本身引入的 launch 开销超预期 | 低 | 融合成单 kernel;GPU 批次 #2 实测 |
| 6 | 排水线程与主循环争抢 PCIe/copy engine | 低 | 用独立 stream + 低优先级;批次 #1 用零假设检查兜底 |

**未决问题(需要决定)**:
- P2b 的 vLLM 侧路由探针,是否值得投入?如果不做,P2 只能看到我们自己的路由,无法直接对拍。**建议做** —— 因为「两边路由是否相同」是一个二值答案,信息量极大。
- 是否把 T0+T1 设为**服务器生产路径也默认开启**?成本为零,但会持续写盘。建议:默认开,带日志轮转。

---

## 附:与现有 bfdiag 工作的关系

本文档的 bfprobe 是 bfdiag 平台的一个子系统,与并行开发中的四项互补:

| 子系统 | 关系 |
|---|---|
| `bfdiag/record`(已完成,commit `02a0645`) | 每次运行的档案与配置指纹;探针数据作为 run artifact 挂在 RunRecord 下 |
| `bfdiag/trace`(进行中) | **就是本文档的 T0 层**,已下发扩展位要求 |
| `bfdiag/daemon`(进行中) | 热引擎;探针在 daemon 里是常驻的,金丝雀自检可复用 T1 签名 |
| `bfdiag/divergence`(进行中) | oracle 逐层对拍;**T1 签名是它的廉价前置筛选**(先用签名定位可疑层,再对那一层做完整对拍) |
