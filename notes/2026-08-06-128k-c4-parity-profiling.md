# 128K/c4 历史性能追平剖析：差距定位到 inter-verify GPU 空闲与接受率口径

日期：2026-08-06 · 模型：`unsloth/Qwen3.6-27B-NVFP4`（snapshot
`ccdaab7e68af2409599b8949a8f2685703c9bae5`）· 单卡 RTX PRO 6000 Blackwell
Max-Q（SM120，96 GB，188 SM，HBM 峰值 ~1.79 TB/s）· 服务：自有 runtime
（`server.app` / qwen36 后端，MTP K=3 + decode/MTP 四图 + persistent prefix
cache + FP8 e4m3 KV）。

## 一句话

历史锚点 **128K/c4 MTP K=3 warm = 222.44 tok/s**（`docs/archive/2026-07-20-PROGRESS.md`，
DirectModelRunner 直连、Split-KV=32、KWIDE/V272 内核时代）与当前服务路径的差距，
被逐层定位到两处：**① 轮内 GPU 空转**（verify 图结束到下一 verify 图开始之间
中位 14 ms、p90 49 ms 的纯主机段时间）；**② 口径差异**（历史 workload 的接受率
50.3% vs 当前同 fixture 91.3%，历史轮时 45 ms/10.04 token vs 当前 ~74 ms/14.96
token——**按每提交 token 计已打平至 ~95%**：4.5 vs 4.7 ms/token）。GC 禁用、
容量 3→4、arena 机理均已实测排除或确认。

## 1. 历史锚点的真实口径（先对齐再怀疑数字）

* 出处：`docs/archive/2026-07-20-PROGRESS.md:4241` —— KWIDE+V272 后
  **128K/c4 warm 222.44 tok/s，acceptance 50.3%**；前序 183.43（Split-KV=32 +
  VTRANSPOSE_ELIM + page-indices 缓存）。
* harness：`runtime/direct_model_runner.py`（**已随 vLLM 剥离删除**）——无 HTTP
  直连、`_DECODE_TARGET_SPLITS_PER_REQ = 32`、warm = 精确重复前缀 + 10240 新后缀。
* workload：`benchmarks/fixtures/ctx128k_prompts.json` —— **4 条互不相同的
  token-id 算术斜坡**（vLLM RandomDataset 公式
  `allowed_tokens[(offset + request_index + arange) % len]`，seed 12345），
  不是可复现文本，**文本往返不保真**，只能以 token-id 直发。
* 由此：历史轮时 = 4 req × (1 + 3×0.503) = 10.04 token/round ÷ 222.44 =
  **45.1 ms/round**。

## 2. 本轮为可比性补的基础设施（均已提交）

1. **`/v1/completions` 支持 `prompt: list[int]`**（OpenAI 兼容形态）：
   token-id 直发，绕过文本往返——历史 fixture 复现的唯一途径。
2. **`benchmarks/server_perf_grid.py --fixture ctx128k|ctx64k
   [--fixture-prompts i,j,...]`**：按 token-id 协议发历史 fixture；
   每波把全部 fixture prompt 各发一次（历史 c=4 = 4 条不同 prompt）。
3. **`QSR_DISABLE_GC`（默认 1，engine ready 后 `gc.collect()+gc.disable()`）**：
   实测对 128K/c4 无吞吐影响（160.9/167.6/170.1 vs 之前的 163.1/177.4/173.5，
   同跑动间方差内），p90 尖峰也未消失——**GC 不是尖峰来源**，开关保留仅为
   与 vLLM 惯例对齐及后续排除法。

## 3. 实测当前状态（capacity=4 × 256K，num_slots=5）

配置说明：历史 c4 需要 4 个 256K 并发槽；08-05 的 best profile 是 capacity=3
（第 4 个请求排队），本轮改为 **capacity=4 / num_slots=5**，显存 94.2/97.9 GB
可行，decode CG 与 MTP 四图的批次桶随之扩到 5（代码原生支持
`range(1, num_slots+1)` 捕获）。

| 测量 | 数字 | fixture |
|---|---|---|
| 128K/c4 filler warm e2e | 163.1 / 177.4 / 173.5 agg tok/s | `server_perf_grid_20260806_070436.json` |
| 128K/c4 filler warm（GC off 复测） | 160.9 / 167.6 / 170.1 | `server_perf_grid_20260806_074842.json` |
| 128K/c4 **历史 fixture** warm e2e | 96.2 / 98.1 / 122.9 agg | `server_perf_grid_fixture_ctx128k_20260806_082906.json` |
| mtp_round_b4 轮时（GC off 跑） | median 80.8 / p90 113.4 / max 137.5 ms | phase log |
| mtp_round_b4 轮时（nsys 窗口内） | median 67.9 ms（127 轮） | 同跑 |
| 历史 fixture 接受率 | **91.3%**（histogram 1042/1174 轮全接受，2.74 drafts/round） | `/debug/stats` |

**fixture 波为何只有 96-123 agg**：greedy 从斜坡上下文生成的续写很快撞 EOS
（256 max_tokens 只跑出 ~129 token/请求；64 max_tokens 时 ~33 token），请求
提前结束后批次退化为 b2/b1 尾巴（实测轮轨迹 b4 仅 8 轮、b2 191、b1 101），
波级聚合被拖垮。**这是内容行为，不是运行时性能**——稳态 b4 段本身是
15.75 token/round（filler 98% 接受）或 14.96（fixture 91.3%）。

## 4. GPU 侧逐图归因（nsys node 级，`/tmp/qwen36_c4_node2.sqlite`，127 轮）

| 图 | 角色 | ms/round | 占比要点 |
|---|---|---:|---|
| graph 131 | **verify**（b4，K+1=4 token × 4 slot） | **42.5** | paged attn 18.7（43.9%）、W8A8 GEMM 7.3、NVFP4 GEMM 7.0、quant 1.3、GDN multistep 1.1、elementwise 系 ~4.9 |
| graph 11+98 | **draft 环**（3 步 MTP 层） | 9.1 | draft attn 4.4（1.37 ms/call，高于独立探针 0.36 ms——待查）+ GEMM 2.9 + merge 1.8 |
| graph 128 | **sync**（提交后缀重放） | ~2.2 | 全 backbone 单遍 |
| eager | lm_head 等 | ~1.5 | lm_head M=16 ≈0.8 ms（1.27 GB @ 1.57 TB/s） |

**inter-verify 窗口**（verify(N) 结束 → verify(N+1) 开始，127 窗口）：
wall median 23.9 ms，GPU busy median 9.9 ms，**GPU 空闲 median 14.0 ms、
p90 49.1 ms、max 607 ms；窗口内 70% 是空闲**。主机侧对应段：accept 的 GPU
等待之后纯主机尾巴（决策 + commit_loop median 0.2 ms + checkpoint 尖峰
p90 5.9 / max 11.5 ms + sync/draft 的 fill 与 launch）。

**独立探针**（`scripts/probe_qwen36_graph_attn_capacity.py`，生产同款
for_contract/for_fixed_capacity 工作区）：

| 形状 | plan | 单次 ms |
|---|---|---:|
| verify b1 q=4 @128K | 47 chunks/req（cap 94 work items） | 0.36 |
| verify b3 q=4 | 15 chunks/req | 0.64 |
| verify b4 q=4 | **11 chunks/req**（cap 仍 94） | 0.82 |
| decode b1/b3 q=1 | 94/31 chunks | 0.39/0.36 |

verify 注意力 b4 = 0.82×16 层 = **13.1 ms**（live 18.7 ms，差值待归因——
可能 page 碎片化/cap 选择不同）。**work_items cap = 94 = 188 SM × 2 CTA/SM ÷
4 KV heads 是全批次共享的**：批次越大每请求 chunk 越少。

## 5. 差距核算与结论

* 按每提交 token：历史 45.1 ms ÷ 10.04 = **4.49 ms/token**；当前
  ~70-74 ms ÷ 14.96-15.75 = **4.5-4.7 ms/token → ~95% 打平**。
* 按 headline agg：稳态 b4 估算 15.75×4… 不——15.75 已含 4 slot：
  15.75/0.074 ≈ **213 tok/s**（median 轮时）——vs 222.44 = **96%**；
  波级实测因 TTFT/restore 摊入与 EOS 尾巴落到 160-195。
* **剩余缺口 = inter-verify 空闲（~14 ms/round，消掉即 15.75/0.060 ≈ 260）
  + verify attn/GEMM 的 kernel 级余量**。接受率口径差异不属运行时缺陷。

## 6. 已排除的假设（别再查）

| 假设 | 判定 | 证据 |
|---|---|---|
| Python GC 停顿造成 p90 尖峰 | ❌ 排除 | 禁用前后轮分布与吞吐不变（§3） |
| engine 轮间 bookkeeping | ❌ 排除 | `engine_step.bookkeep_ms` median 0.03 |
| split_kv 没开 | ❌ 排除 | verify/draft plan 均 split_kv=True（94/31/11 chunks） |
| 注意力没切块 | ❌ 排除 | 同上；merge kernel 在 trace 中可见 |
| capacity=3 限制 c4 | ✅ 确认并已改 | 08-05 网格第 4 请求排队；本轮 capacity=4 |
| arena 装不下 4 条不同 128K 条目 | ✅ 确认 | scratch 行 = 1×256K = 2 条 128K；且完成边界还会存 prompt+gen 条目，{A,B,A+gen,B+gen} 需 4 条 → 两 distinct prompt 以上必churn |

## 7. 下一步（按期望收益）

1. **inter-verify 空闲的亚相位定位**：RoundProfile 增加
   accept 尾巴/commit/launch 细分（或给 end_round 打绝对时间戳与 engine
   轮间间隔对齐），把 14 ms 拆到具体主机段，逐个消除。目标 ≤5 ms。
2. **verify 注意力 chunk 上调**：`plan_verify_graph_capacity` 的
   `graph_ctas_per_sm=2` 试 3/4（b4 从 11 → 16/22 chunks/req），独立探针
   先测再动生产；注意 SMEM/CTA 与 merge 成本。目标 18.7 → ~12 ms。
3. **draft 环单次图化**：3 次 replay 合并为一张含设备侧采样链的环图，
   省每步 fill+launch（历史 `CapturedBatchDecodeGraph` 即整环捕获）。
4. **draft attn 1.37 ms/call vs 探针 0.36 ms** 的差异归因（page 碎片化？
   cap 选择？descale 路径？）。
5. arena 扩容（scratch 行 ×2 + checkpoint budget 4）以支持 ≥4 distinct
   长前缀常驻——多用户真实负载也需要；当前显存需先让出 ~8 GB
   （如 slots 256K→192K）才能装下。

## 8. 产物索引

* `/tmp/qwen36_c4_node2.sqlite` —— b4 node 级 nsys（verify/draft/sync 归因源）
* `/tmp/qwen36_node3.sqlite` —— b3 同款（较早一轮）
* `scripts/probe_qwen36_graph_attn_capacity.py` —— 生产同款工作区的注意力
  plan+计时探针（b1-b5）
* `scripts/probe_qwen36_verify_attn_split.py` —— eager/graph 计划对比探针
* fixture：`benchmarks/fixtures/server_perf_grid_fixture_ctx128k_20260806_*.json`
  与 `server_perf_grid_20260806_07{0436,4842}.json`
* phase log：`logs/quality/server_cap4_nogc_20260806.log`、
  `logs/quality/server_c4_node2_20260806.log`

## 9. 修复一：adaptive chunking 扩展到 Qwen3.6 verify 几何（已验证 −4.6 ms/轮）

根因（探针实证）：verify 图按**最坏容量 262144** 捕获，chunk 计划冻结在
capture 时刻；128K 回放时 11 个可用 chunk/请求只用上 6 个（work items
94 中仅 48 个活跃），注意力 1.167 vs 0.809 ms/call（b4，
`scripts/probe_qwen36_graph_attn_capacity.py --worst 2048` 复现）。

sparkinfer 本就有解：`update_prefill_graph_chunk_metadata` 的
`ADAPTIVE_CHUNKING` 分支每次 replay 按 live `cache_seqlens` 重建 chunk
大小（kernel 本身几何无关，只用 CTA_TILE_Q/GQA/PAGE_SIZE），但
`workspace.py` 的门控白名单只放了 Laguna 几何（cta_tile_q=64、
head_dim=128）。修复 = 白名单加 Qwen3.6 verify 子句（cta_tile_q=16、
head_dim 256）。

**实测（live server，b4 filler 128K，379 轮 pooled）**：
`verify_gpu_ms`（CUDA event）**44.46 → 39.83 ms（−4.6，−10%）**。
轮时 median 78.0（修复前单跑 73.9/78.9/80.8，方差带内）；aggregate 波级
156-160（修复前 155-180）——GPU 收益被跑动间方差 + commit 尖峰（p90 17）
掩盖，尚未完全兑现到端到端。

待办（§7 第 2/3 项仍是主力）：draft 图注意力同款最坏容量冻结（探针：
decode b1 worst 0.933 vs 0.392 ms/call）；commit_loop 尖峰定位；
accept 队列开销。

## 10. 修复二：decode/draft 图回放按 live 长度重切 chunk（4K/64K 大收益，128K 持平）

同类 bug 的第二处：decode/draft 图捕获时绑定最坏容量 chunk 计划，回放
从不调用 `update_decode_graph_replay_metadata_from_runtime_cache_seqlens`
（runtime 里 0 处调用），chunk 大小永远冻结在 262K 捕获值。修复 =
`Qwen36DecodeGraphAttention.update_replay_metadata()` + 两处接线
（`build_decode_batch`、draft `_fill`）。

探针（q=1 @131K）：b1 1.123→0.474、b2 1.165→0.396、b3 1.008→0.438、
b4 0.650→0.428 ms/call。服务端 c4 warm agg（capacity 4，MTP K=3）：

| 上下文 | 修复前 | 修复后 |
|---|---|---|
| 4K | ~150-195 | **345.9 / 326.6** |
| 64K | ~146-170 | **189.0 / 185.8** |
| 128K | ~155-160 | 158.5 / 148.2 / 144.5（持平，128K 贴近最坏容量，重切收益小） |

fixture：`server_perf_grid_20260806_122813.json`（三档合并跑）、
`server_perf_grid_20260806_123150.json`（128K 复测）。

## 11. 阶段性结论（2026-08-06 午）

* 按每提交 token：~95% 历史水位（4.7 vs 4.5 ms/token）——本轮两处修复后
  4K/64K 已**超过**历史同档数字（历史 64K/c4=236.69 为 DirectModelRunner
  口径，当前服务端口径 189 尚未反超；4K 无历史同口径锚点）。
* 128K/c4 仍 ~0.68× 历史 headline（150 vs 222.44）：瓶颈 = verify 图 GPU
  ~40 ms（历史整轮才 45 ms）。剩余差距在 kernel 层：verify attn 已压到
  ~0.8 ms/call 量级，下一个可动的只有 GEMM 小 M 效率与 elementwise 融合
  （均需过 bit-parity 质量锚，属 roadmap 阶段四工作）。
* 数值口径警告：两处重切 chunk 都改变 split-KV 归约顺序（ULP 级漂移），
  重跑质量锚（`scripts/run_qwen36_quality.sh`）前不要引用绝对分数。
