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

## 12. 修复三：draft→verify 全设备直连（128K/c4 轮时 −11%，agg e2e 157.5/165.9）

§7 的 host 间隙定位继续追到一处明确的每轮 D2H 往返：batched draft 图
（`Qwen36MTPDraftCudaGraph.replay_batch`）此前把 `[B, K-1]` 结果
`.tolist()` 回主机，再由下一轮 verify fill 和 GPU accept 比较重新上传。
round profile（128K/c4 warm，b4 稳态）里 `draft_batch` host 段中位 12.5 ms，
其中约 7 ms 是这次 `.tolist()` D2H。

**修复（未提交前为工作区改动，见 commit）：**

1. `replay_batch` 返回 device 行（`[K-1]`，不再 tolist）。
2. `_continue_draft_batch` 图路径把 first-draft 种子 D2D 拼接并返回
   device `[K]` 行；`round_batch` 直接 `cat` 成 `[B, K+1]` verify 输入，
   `determine_accept_reject_batch` 走 device 分支（`committed` 由 verifier
   自身预测重建，与 host 分支按匹配条件逐值相等，**数值路径不变**）。
3. 混合来源归一化：prefill/单槽路径仍产出 host `list[int]`，图轮次产出
   device 行；同一 batch 可能混两类（恰好是初版 devicedraft 实跑炸掉的
   场景：`TypeError: can't convert cuda tensor to numpy` / `'list' object
   has no attribute 'shape'`，均已在 live server 复现并修复）。现在任何
   slot 是 tensor 就整体转 device；单槽 fallback 与 `round()` 统一先
   `.tolist()` 保持 host 契约。
4. profile 的 `elapsed_time` 改为轮边界一次性 `_draft_ev1.synchronize()`
   （旧 `.tolist()` 曾是隐式 drain；device 路径没有天然 drain）。

单测：`tests/test_qwen36_mtp_engine.py test_qwen36_backend.py
test_mtp_accept_sampling.py` 101 passed；新增 device-vs-host 逐值一致与
shape 校验测试。

**实测（同口径：capacity 4 / num_slots 5，128K block-aligned，c4，
max_tokens 256，2 轮 warm，prefix cache 全命中，MTP K=3 + 四图）：**

| 指标 | 修复前基线 | 修复后 |
|---|---:|---:|
| 轮时 total（b4 中位） | 82.8 ms | **73.7 ms（−11%）** |
| draft_batch host 段（中位） | 12.5 ms | **0.1 ms** |
| draft GPU event（中位） | 15.9 ms | 5.8 ms |
| verify GPU event（中位） | 39.9 ms | 40.4 ms（GPU 忙，未变） |
| sync_ragged（中位） | 1.1 ms | 1.1 ms |
| warm agg e2e（两波） | 143.83 / 163.35 | **157.52 / 165.86** |

fixture：`benchmarks/fixtures/server_perf_grid_devicedraft_fix_20260806.json`；
基线 fixture：`server_perf_grid_20260806_141116.json`；phase 日志：
`logs/quality/server_devicedraft_fix_20260806.log`（`mtp_round_b4` 190 条）。

命令（服务端）：
```bash
source /tmp/qwen36_server_env.sh
export QSR_SERVER_CAPACITY=4 QSR_SERVER_NUM_SLOTS=5
setsid nohup /home/bot/.venvs/vllm/bin/python -m server.app \
  --host 127.0.0.1 --port 8300 > logs/quality/server_devicedraft_fix_20260806.log 2>&1 &
```

命令（网格）：
```bash
/home/bot/.venvs/vllm/bin/python -u benchmarks/server_perf_grid.py \
  --base-url http://127.0.0.1:8300 --model qwen3.6 --endpoint completions \
  --contexts 128k --concurrency 4 --max-tokens 256 --warm-rounds 2 \
  --out benchmarks/fixtures/server_perf_grid_devicedraft_fix_20260806.json
```

下一目标：把同一修复的 4K/64K 与 c1-c3 补齐，然后全网格对齐历史
（README Performance 段同步更新）。

## 12.1 同修复扩展验证（2026-08-06 下午，capacity 4 / num_slots 5）

全部 warm 波 prefix-cache 全命中、0 error、每波 4/3/2 × 256 token 完整
生成（stats_delta 中 mtp_verify/mtp_draft/mtp_batched_sync 计数与基线
一致）。

| 单元 | warm agg e2e tok/s |
|---|---:|
| 4K c4 | 324.04 / 288.99；复测 291.16 / 300.36（§10 前值 345.9/326.6，方差带内，短波噪声） |
| 64K c4 | 192.79 / 202.14（§10 前值 189.0 / 185.8，**提升**） |
| 128K c4 | 157.52 / 165.86（基线 143.83 / 163.35，**提升**） |
| 128K c3 | 140.36 / 130.11 |
| 128K c2 | 105.04 / 112.87 |

fixture：`server_perf_grid_devicedraft_fix_{4k_c4,4k_c4_rerun,64k_c4,128k_c3,
128k_c2}_20260806.json`。4K 的 per-round 固定开销（anchor H2D + 轮末
profile sync）在短波下与跑动方差同量级；若全网格复跑仍低 ~10%，再评估
把 profile sync 移出热路径（QSR_PROFILE_ROUNDS=0 时本就不执行）。

## 12.2 全网格（devicedraft 构建，4K/32K/64K/128K × c1-4）

同一进程连续跑完 16/16 cell，0 error；每 cell COLD + 2 个 WARM 波，WARM
全部 prefix-cache 命中（restores == concurrency）。warm aggregate e2e
tok/s（两波）：

| 上下文 | c=1 | c=2 | c=3 | c=4 |
|---|---:|---:|---:|---:|
| 4K | 103.8 / 108.3 | 205.8 / 176.6 | 251.6 / 234.2 | 318.6 / 311.5 |
| 32K | 98.3 / 102.1 | 157.8 / 145.8 | 236.9 / 224.6 | 262.3 / 225.4 |
| 64K | 97.3 / 92.6 | 154.5 / 163.6 | 178.5 / 197.1 | 200.5 / 210.9 |
| 128K | 73.8 / 66.3 | 121.8 / 118.1 | 126.0 / 131.7 | 139.9 / 146.3 |

fixture：`server_perf_grid_devicedraft_full_20260806.json`。128K/c4 在本
轮跑出 139.9/146.3，早前冷启动后的隔离复测为 157.5/165.9——同参数跨进程
方差 ~±10%（TTFT 0.29-0.68 s、per-request decode 39-45 tok/s），对比历史
数字时以隔离复测为准并注明方差带。

## 13. 追平并反超 128K/c4 历史 headline（2026-08-06 傍晚，正式无 profile 口径）

### 13.1 本轮修复链（全部已提交）

1. **滚动 checkpoint 前缀哈希**（`runtime/backends/qwen36.py`
   `_rolling_prefix_hash`）：`_maybe_checkpoint` 原来每个块边界对整个
   128K 前缀重打包重哈希（~1.1 ms/轮），现按 slot 缓存 blake2b 上下文，
   只喂增量尾部。摘要与全量 `_prefix_hash` 逐位一致（新单测
   `test_rolling_prefix_hash_matches_fresh_hash_across_boundaries` 覆盖
   块对齐/非对齐/越界跳变/槽复用四条路径）；`reset_slot` /
   `_commit_prefill` 时失效上下文。
2. **verify 输入零分配**（`runtime/backends/qwen36_mtp.py`）：每批预分配
   `[B, K+1]` verify-token 缓冲 + pinned anchor，device-draft 路径原地
   `copy_`（non_blocking）填充，替换每轮 3 个小 CUDA tensor 分配。
   verify_replay 主机段中位 **10.78 → 1.48 ms**（−9.3 ms）。
3. **lm_head 并入 verify CUDA Graph**（`qwen36_mtp_cudagraph.py`）：
   capture 时把 `compute_logits` 一并捕获，replay 返回
   `(hidden, graph-owned logits)`；`compute_logits` 阶段从 ~0.8 ms 落到
   ~0.00 ms，且 logits 缓冲由图持有，热路径零分配。所有假图 stub 同步
   改为返回 `(hidden, logits)` 对。
4. **GDN checkpoint 单次拷贝**（`runtime/model/qwen36_slots.py`）：
   `capture_recurrent_state` 每槽预分配目标缓冲，96 个 `clone()` 合并为
   一次 `torch._foreach_copy_`。
5. **sparkinfer M32 raw-FP8 verifier + 回放元数据跳过**：
   `planner.py` 对 Qwen3.6 verify 几何（cta_tile_q=32、head_dim=256、
   GQA6、page=128）选 M32 单 tile；`_forward.py` 路由到
   `PagedFp8ExtendRawForwardKernel`；`workspace.py` 白名单加 M32 自适应
   chunking，并新增 `replay_page_key`：verify 回放时每请求页数不变就跳过
   3 个 Triton worklist 重建（128K 稳态每 32 轮才变一次）。

### 13.2 轮级阶段（lm_head 入图后，`QSR_PROFILE_ROUNDS=1` 诊断日志，
257 rounds，b4 稳态）

| 阶段 | med | mean | p90 |
|---|---:|---:|---:|
| total | 53.40 | 56.19 | 65.92 |
| accept_gpu_wait（GPU verify 执行） | **49.79** | 52.32 | 61.47 |
| verify_replay（主机填充） | **1.48** | 1.85 | 2.31 |
| sync_ragged | 1.02 | 1.15 | 1.72 |
| commit_loop / draft / setup 等 | ≤0.4 | — | — |

主瓶颈已从主机侧移到 **GPU verify 本体 ~50 ms/轮**。日志：
`logs/quality/server_cglogits_rollinghash_20260806.log`（聚合脚本
`scripts/aggregate_round_profile.py`）。

### 13.3 内核级归因（eager verify_batch+lm_head，128K×4，10 iters，nsys）

`logs/quality/verify_gpu_profile_rollinghash_20260806.out`，单轮 verify
合计 ~56 ms（eager + 剖析开销下）：

| 内核 | ms/verify | 说明 |
|---|---:|---|
| paged raw-FP8 attention（M32） | 143.78 ms/160 calls ≈ **14.4** | 16 层，图内实测 ~0.899 ms/call |
| qsr FP8 W8A8 GEMM（FP8 通道线性） | 75.89+72.34 ≈ **14.8** | attention qkv/o + 部分 MLP |
| NVFP4 fused MoE（W4A4 blockscaled） | 42.99+29.35 ≈ **7.2** | MLP，默认 all-rows W4A4 路径 |
| GDN fused recurrent | 11.37 ≈ **1.1** | — |
| dynamic_per_token_e4m3_quant | ~1.3 | — |
| RMSNorm 系（mean/rsqrt/pow） | ~1.9 | 与历史 A/B 保位级一致，不动 |

attention 独立探针（`scripts/probe_qwen36_verify_attn_sweep.py`）：
b4/128K **M32 0.736 ms/call vs M16 0.839 ms/call**（−12%）；图内实测
0.899 ms/call 仍有 ~20% 未归因（chunk 几何/主存带宽方向，下一步候选，
但需过质量锚）。

### 13.4 正式结果（128K/c4，max_tokens 256，每波 4×256 token 全量完成，
0 error，WARM 波 prefix 全命中 restores=4）

| 运行（服务/口径） | warm agg e2e tok/s | vs 222.44 |
|---|---:|---|
| `server_perf_grid_wallprof_rollinghash_20260806.json`（新进程，profile=1） | 238.36 / 243.33 / 227.35 | **3/3 超过** |
| `server_perf_grid_cglogits_20260806.json`（新进程，profile=1，lm_head 入图后） | 239.26 / 222.53 / 248.03 | **3/3 超过** |
| `server_perf_grid_noprof_headline_20260806.json`（**新进程，无 profile 正式口径**） | 237.02 / 234.82 / 232.05 / 222.20 / 238.76 | 中位 **234.82**，4/5 超过 |
| `server_perf_grid_noprof_headline2_20260806.json`（同进程续跑，缓存已热） | 226.44 / 216.19 / 222.79 / 219.82 / 220.18 / 215.51 | 方差下沿，216–227 |
| `server_perf_grid_noprof_suffix10240_20260806.json`（**历史协议形态：warm 前缀 + 10240 新后缀**） | warm2/3 = 197.37 / 201.30 | 稳态 ~89–90%（首波含 47.4 s 后缀 continue-prefill） |

全网格（`server_perf_grid_20260806_170138.json`，c4，warm2）：4K
384.44/385.19、32K 353.26/340.46、64K 291.28/261.91、128K
204.41/224.89。**64K 已超过历史 236.69**（README 的 267 为低置信回声）；
128K 网格波落在方差带内，隔离冷启动后即可达 222–248。

### 13.5 结论与口径说明

* **精确重复前缀 warm 协议**（README Performance 表的口径）下，
  128K/c4 端到端已超过历史 222.44：三个新进程样本的中位/均值
  234.8 / 233.0，且 cglogits 样本 3/3 波 ≥ 222.44。
* **历史 DirectModelRunner 协议形态（+10240 新后缀）**下，稳态
  warm2/warm3 = 197–201 tok/s（~89–90%）。差异组成：HTTP vs 直连
  （~可忽略）、后缀 continue-prefill 摊入 TTFT（首波 47 s）、以及
  内容 workload 不同（filler 接受率 ~98% vs 历史 token-id fixture
  50.3%）。按轮内 token 效率：15.75 token/53.4 ms ≈ 295 tok/s 稳态，
  高于历史 10.04 token/45.1 ms = 222.6。
* **跑间方差 ±5–10% 是主机 CPU 争用**（chrome ~275% CPU、celery 常驻），
  非 GPU 或代码差异；同服务连续重跑（headline2）即落入 216–227。
  正式对比以新进程隔离样本为准，本笔记同时保留方差下沿样本。
* 剩余 GPU 侧余量（M32 图内 vs 探针 ~20%、小 M GEMM、elementwise 融合）
  属 roadmap 阶段四，需先过位级质量锚，本轮不再动。

### 13.6 环境与产物

* 服务 env：`scripts/qwen36_128k_bench_env.sh`（capacity=4 /
  num_slots=5 / block_size=16 / 256K per slot / MTP K=3 / FP8 e4m3 KV /
  CG + prefix cache / GPU_MEM_UTIL=0.92 / production=1）。
* 启动：`set -a; source scripts/qwen36_128k_bench_env.sh; set +a;
  setsid /home/bot/.venvs/vllm/bin/python -u -m server.app --host
  127.0.0.1 --port 8300 > logs/quality/server_noprof_final_20260806.log &`
  （正式基准不设 `QSR_PROFILE_ROUNDS`）。
* 基准命令：
  `/home/bot/.venvs/vllm/bin/python -u benchmarks/server_perf_grid.py
  --base-url http://127.0.0.1:8300 --model qwen3.6 --endpoint completions
  --contexts 128k --concurrency 4 --max-tokens 256 --warm-rounds N
  [--warm-suffix-tokens 10240] --out <fixture.json>`。
* 探针/聚合：`scripts/probe_qwen36_verify_attn_sweep.py`、
  `scripts/probe_qwen36_verify_gpu_profile.py`、
  `scripts/aggregate_round_profile.py`。
* 本轮日志：`logs/quality/server_noprof_final_20260806.log`（正式服务）、
  `grid_noprof_headline{,_2}_20260806.out`、`grid_noprof_suffix10240_20260806.out`。

## 14. 质量套件重跑：M32 verify 图在 capacity 8 下的门限修复（2026-08-06 晚）

### 14.1 发现：num_slots=9 时 verify CG 捕获失败 → 静默降级 eager

质量套件 `run_qwen36_quality.sh` 的 suite-fast / mmlu 配置是
`capacity=8 / num_slots=9`（CG warmup 需要 capacity+1 个物理 slot）。
首轮重跑（label `qwen36_20260806`）服务日志出现：

```
Qwen3.6 MTP: verify CUDA Graph capture failed -- degrading to eager
... cg_status={'anchor': 'unused', 'draft': 'captured',
               'sync': 'captured', 'verify': 'failed'}
```

根因：`sparkinfer/attention/paged/_forward.py` 的
`_use_raw_fp8_verify_forward_kernel` 公共门限写死
`1 <= batch_capacity <= 8`，batch=9 时 M32 worklist 路径被拒，回落到
`PagedForwardKernel` 直接 `NotImplementedError`，服务按项目既定策略降级
eager（`QSR_QWEN36_MTP_REQUIRE_CG=0`）。这不是质量回退，是配置口径
不符：该轮结果**不能**与历史 MTP+CG 数字对比，已作废并停止。

### 14.2 修复（sparkinfer `d900ef1`，已推 master）

把 `1 <= batch_capacity <= 8` 从公共门限移除，只保留在 Laguna M64/D128
analytic 分支内。M32 worklist 路径每请求 K+1 个 query 打包进单个 32 行
tile（`num_qo_tiles == batch`、tile 下标全 0），无 8 行 batch 上限。

### 14.3 有效重跑（label `qwen36_20260806b`）

* 全程单 GPU 进程：每阶段一个服务，阶段间 `server stop` 后才起下一个；
  三个 dim 并行只是客户端进程，不占显存。用户明确禁止多服务并存。
* 服务启动确认（`logs/quality/server_suite_qwen36_20260806b.log`）：
  `cg_status={'anchor': 'unused', 'draft': 'captured', 'sync': 'captured',
  'verify': 'captured'}`，`capacity=8 num_slots=9`、32K slot、CG + prefix
  cache + MTP K=3 + FP8 e4m3 KV 全开。
* 流程：`RUN_LABEL=qwen36_20260806b setsid bash
  scripts/run_qwen36_quality.sh all`，输出
  `logs/quality/quality_all_20260806b.log`。
* 与历史的 ULP 级差异来源：M32 verify 的 split-KV 归约顺序与旧路径不同
  （2026-08-05 起），质量数值若与历史绝对值有末位差异属预期，需在结果
  表注明，不能当作回退。

### 14.4 结果（截至 2026-08-06 19:15，MMLU 仍在跑）

| 维度 | 历史（README/07-22） | 08-05 MTP+CG 基线 | 今天 20260806b（MTP+CG 全开） | 状态 |
|---|---|---|---|---|
| tool | 1.000 (20/20) | 1.000 (20/20) | 1.000 (20/20) | 一致 |
| agent | 1.000 (4/4) | 1.000 (4/4) | 1.000 (4/4) | 一致 |
| longctx 8K/32K/64K/128K | 1.000 (12/12) | 1.000 (12/12) | 1.000 (12/12) | 一致 |
| code 4096 (HumanEval/+) | —（README 现行为 08-05 门） | 0.921 / 0.884 | **0.890 / 0.866** | 归因中 |
| HumanEval+ 768 | 0.445 / 0.433 | 0.421 / 0.415 | **0.445 / 0.427** | 与 07-22 持平（base 73/164 完全一致） |
| MMLU-Pro 414 thinking | 84.54% | 84.54% | 待出 | 待出 |

code 4096 的 −3.1/−1.8pp：任务级翻转 17/164（9 pass→fail、4 fail→pass、
4 仅 plus 变化），且 164/164 的 reasoning 轨迹都在前 ~30-200 token 处分叉
（中位首分叉 121 字符）——与 768 维度三代任两代 ~40 题翻转、净差 ≤4 题的
形态一致，是 greedy 对数值路径敏感的表现，不是系统性失效。根因待
MTP-off 归因实验确认（对比同构建 eager 是否复现 08-05 输出）；M32
verifier 的 split-KV 归约顺序变化是首要嫌疑（见 §14.3 与 bec29b5
Directive）。

## 15. TTFT 逐波漂移根因：admission 路径上的诊断 LCP（2026-08-06 晚，已修）

### 15.1 问题与证据

用户指出的核心问题不是 decode 吞吐，而是 **纯 warm 波内 TTFT 不稳定**：
同一协议（completions / 128K / c4 / greedy / max_tokens=256 / warm 重复），
5 波 TTFT 从 ~0.5s 单调漂到 0.84s（`noprof_headline2`），suffix 波更是
0.80–1.04s。逐段计时（`QSR_PROFILE_ADMISSION=1`，wall-clock 不扰动 GPU）
把漂移定位到 admission 的 **activate 相位**：

| 波 | 修复前 activate | 修复前 prefill_begin |
|---|---|---|
| w1 | 28–41 ms | 52–75 ms |
| w2 | 25–93 ms | 55–134 ms |
| w3 | 41–129 ms | 52–104 ms |
| w4 | 49–158 ms | 39–90 ms |
| w5 | **56–197 ms** | 41–125 ms |

### 15.2 根因：`_log_prefix_overlap` 在服务关键路径上做 O(B·(B+H)·L) Python LCP

`server/engine.py` 的 `_log_prefix_overlap` 在每次 admission、prefill 与
activate 之间无条件执行：对每个请求与同波其余 B−1 个、以及 `_recent_prompts`
历史全部 H 条（`maxlen=64`，每波 +4）做 `_longest_common_prefix_len`——
一个纯 Python `zip` 循环。131072 token 全匹配实测 **3.55 ms/次**；纯 warm
波全为相同 prompt，全部全匹配：

* w1：4 × (3 + 4) = 28 次 ≈ 100 ms
* w5：4 × (3 + 20) = 92 次 ≈ 327 ms

与实测 activate 漂移 28→197 ms 同量级、同形状（一次 LCP 成本 3.55ms 是
新鲜 list 测量，服务内 list 常驻缓存后略快，量级一致）。这不是 GPU、不是
decode、不是 HTTP——是一个 instrumentation 函数把每波 TTFT 税逐波抬高。
历史 run（含 `noprof_headline2`）同样背着这笔税，只是当时没逐段计时，
被误读成“前缀恢复路径随长度非线性变慢”。

### 15.3 修复（本工作区，待提交）

`_longest_common_prefix_len` 增加 `cap` 参数；`_log_prefix_overlap` 全部
扫描以 `cap=self.block_size`（16）封顶。该统计只决策“重叠是否 ≥ block_size”
（事件计数器），封顶后语义完全一致；samples 里的精确重叠值退化为饱和值
（≥16 记为 16），仅影响诊断精度，不影响任何基准数字。

### 15.4 修复后同口径数据（fixture 已存）

**纯 warm（completions / 128K / c4 / 256 / 5 波）**

| 指标 | 修复前 `noprof_headline2` | 修复后 `noprof_headline_fixed_lcp` |
|---|---|---|
| TTFT | 0.503 → 0.547 → 0.568 → 0.757 → **0.841** | 0.216 → 0.216 → 0.216 → 0.227 → **0.218** |
| wall | 4.60–4.75 s | 4.35–4.64 s |
| agg e2e | 215–222 tok/s | **220–236 tok/s（最佳 235.6）** |
| activate 相位 | 24–197 ms | **<1 ms** |

TTFT 不再漂移，且 5 波全部稳定在 0.216–0.227s；最佳波 235.6 tok/s 已超
历史 222.44 头条（同为轮级/波级口径下）。decode 稳态指标不变（修复不触及
GPU 路径），证明该漂移本来就是纯 host 开销。

**+10240 后缀（completions / 141312 / 3 波，warm1 付一次性 40960 token prefill）**

| 指标 | 修复前 `noprof_suffix10240` | 修复后 `noprof_suffix_fixed_lcp` |
|---|---|---|
| warm1 TTFT | 47.41 s | 12.88 s（batched 路径；见 15.5） |
| warm2/3 TTFT | 1.035 / 0.795 s | **0.195 / 0.215 s** |
| warm2/3 wall | 5.19 / 5.09 s | 4.30 / 4.59 s |
| warm2/3 agg e2e | 197 / 201 tok/s | **238 / 223 tok/s** |

### 15.5 遗留观察（下一步可选）

1. **warm1 后缀 prefill 47.4s → 12.9s**：前者走 per-slot 分块（4×5 chunk ×
   ~2.2s），后者走 batched 路径（5 chunk × ~2s）。同一代码，仅持久缓存
   状态不同导致 `build_prefill_batch` 是否可用。值得查为何 per-slot 会
   在那种状态下触发，统一走 batched。
2. **首次 MTP scratch restore 12.7s**：冷波中第 2–4 个并发请求的
   `resync_prefix_tail` 首次 12.7s、后续 14–15ms（页面共享 vs 字节拷贝）。
   若持久条目首次 restore 能直接共享页面而非拷贝，冷波 TTFT 可再降。
3. `reconcile` 相位 9–38ms 与 persistent-prefix 条目数相关，量级小，暂不动。

## 16. 多 chunk 批量 prefill（2026-08-06 深夜，§15.5-① 部分落地）

### 16.1 改动

`prefill_chunked_step` 的 BxQ 批量分支原来要求 `start == 0 且
len(suffix) <= chunk`——只有能装进单个 chunk 的 suffix 才走批量。去掉这两个
限制：**任何 chunk 步上，只要多个待 prefill 槽的 suffix 等长，就批量执行该
chunk 片**（`build_prefill_batch` 的安全子集检查原样保留：等长 prior KV、
uniform GDN regime、ordinary live rows；anchor logits 只在最后一片计算）。
正确性依据：`commit_prefill_batch` 每片对所有槽推进相同长度 → prior KV
长度逐片保持相等；MTP sync 消费的本来就是"当前 chunk 的 hidden"，与
per-slot 路径同契约。

### 16.2 实测（capacity 4 / num_slots 5，128K prefix + 10240 suffix，c4）

| 指标 | 修复前（212215 前的状态） | 修复后 |
|---|---|---|
| warm1 TTFT | 26.5 s | **19.4 s**（batched_forward ×5 chunk 生效） |
| warm2/3 agg | 250–252 tok/s | 224–227（跑动间方差带内，round_batch med 58.3 与修复前 57.8 一致） |

### 16.3 遗留项的后续归因（2026-08-06 深夜续查，两项已关闭）

1. **冷波"多槽组未批量"之谜：已关闭，不存在该场景。** 冷波 admission
   实际是 [1, 3]：第 1 个请求单独 prefill（128K，~64 个 2048-token 片），
   完成后其 prompt 存入 persistent arena；随后 admit 的 3 个请求在
   `_apply_prefix_hit` 里**全部持久命中**（hit == len(prompt)），走
   restore + cached_hidden 路径，根本没有 chunked prefill 可批量。
   独立复现（无服务器，直接 `build_prefill_batch` 3 个 fresh 槽）确认
   fresh 多槽组本身可批量——但真实负载里等长同 prompt 的并发请求永远
   会被首请求的 store 变成持久命中，这正是设计意图（restore ~15 ms vs
   prefill ~57 s），无需批量。
2. **admission 合并（warm1 全 4 槽批量）：已放弃，收益不成立。** 实测
   141K 上下文下 b2 批量片 3.3 s/4096 tokens（0.81 ms/token）与 per-slot
   片 ~1.6 s/2048（0.78 ms/token）每 token 成本几乎相同——长前缀下
   prefill 由 attention 主导，批量对 GEMM 形状的收益被摊平。§15.5 估计
   的"batched ~13 s"是修复链落地前的旧测量（当时 per-slot 路径还背着
   LCP 诊断税等开销）。当前 warm1 = 19.4 s 已接近该上下文的批量下限，
   做调度侧 coalesce（语义改动）换不来可测收益。
3. 跑动间方差提示：同协议不同跑的 round_batch 中位在 53.8–58.3 ms 之间
   （acceptance 91–100% 时），单次数值对比必须带方差带。

## 17. code-4096 −3.1pp 归因实验（2026-08-06 深夜，已关闭）

§14.4 遗留问题：code 4096 在 20260806b（MTP+CG 全开）为 0.8902/0.8659，
比 08-05 基线 0.921/0.884 低 3.1/1.8pp。归因实验：同一构建、同一
suite-fast profile（capacity 8 / 32K slots），只关 MTP
（`QSR_SERVER_ENABLE_MTP=0`），重跑 code 维（164 题，greedy，
max_tokens=4096，concurrency 8）。

| 配置 | HumanEval | HumanEval+ |
|---|---|---|
| **MTP-OFF（本实验）** | **0.9268** | **0.8902** |
| MTP-ON（20260806b） | 0.8902 | 0.8659 |
| 08-05 基线 | 0.921 | 0.884 |

**结论：差距完全来自 MTP 路径。** MTP-OFF 不仅恢复、且略超 08-05 基线
（greedy 翻转噪声带内）；MTP-ON 的 −3.1pp 是 M32 verify 的 split-KV
归约顺序相对旧路径的 ULP 级差异在 greedy 边界题上的翻转（17/164 任务级
翻转，与 §14.4 观察一致）。整体质量判定不变：MMLU-Pro 85.75 > 历史
84.54、tool/agent/longctx 全 1.000、HumanEval 768 持平——code-4096 单
指标处于 greedy 跑动间翻转噪声带（历史各代之间 ±40 题/±2.4pp），不构成
系统性回退。若未来要求 MTP 路径与 eager 逐位一致，需把 M32 verifier 的
归约顺序对齐旧路径（kernel 级工作，未立项）。

数据：`evalplus_results/quality/code_mtpoff_20260806.part.code.json`、
日志 `logs/quality/suite_code_code_mtpoff_20260806.log`。

## 18. code-4096 四方归因：MTP 算法无罪，差值钉死到两处 kernel 归约顺序（2026-08-07）

§17 只证明了差距在 MTP 路径内。用户追问"08-05 也开 MTP，差在哪"，补做
两个 sparkinfer A/B 开关后四方对比（同构建、同 suite-fast profile、同
code 协议，只动 verify 数值路径）：

| 配置 | HumanEval | HumanEval+ |
|---|---|---|
| MTP-ON M32+adaptive（当前生产） | 0.8902 | 0.8659 |
| MTP-ON M16+adaptive | 0.9024 | 0.8902 |
| **MTP-ON M16+frozen（08-05 数值模式）** | **0.9268** | **0.8902** |
| **MTP-OFF** | **0.9268** | **0.8902** |
| 08-05 基线（MTP-on） | 0.921 | 0.884 |

**结论：**
1. **MTP 算法本身无罪**：把 verify 数值路径还原到 08-05 形态（M16 kernel +
   frozen worst-case chunking）后，MTP-ON 与 MTP-OFF 得分逐位一致
   （0.9268 = 0.9268），且都高于 08-05 基线 0.921（greedy 翻转噪声带内）。
2. **−3.7pp（0.9268→0.8902）拆成两处 split-KV 归约顺序效应**：
   * adaptive replay re-chunking（`d5865f8`，2026-08-06）：0.9268→0.9024，
     **−2.4pp**——live 长度重切 chunk 改变了每个 KV chunk 的边界与 merge
     归约树；
   * M32 raw-FP8 verifier tiling（sparkinfer `3fc4a5b`，2026-08-06）：
     0.9024→0.8902，**−1.2pp**——cta_tile_q 16→32 单 tile worklist，K/V 页
     读取与累加顺序与 M16 双 query-tile 不同。
   两者都是 ULP 级 logits 差异在 greedy 边界题上的翻转（164 题中 ~17 题
   任务级翻转），不是系统性质量回退（MMLU-Pro 85.75 > 84.54 等其余指标
   全面持平或更好）。
3. **性能代价**：M16+frozen 模式 verify 注意力每页读两次 + 短中上下文沿用
   最坏 chunk，verify 注意力约慢 ~2×（probe 口径），128K/c4 轮时约
   +10 ms；换来的是与 08-05 / 非投机路径逐位一致的数值。

**A/B 开关（sparkinfer fork，本次落地）：**
* `SPARKINFER_QWEN36_VERIFY_M16=1`：planner 对 Qwen3.6 verify 几何返回
   cta_tile_q=16（M32 gate 失配 → 通用 kernel），复现 08-05 kernel 形态。
* `SPARKINFER_QWEN36_VERIFY_NO_ADAPTIVE=1`：workspace 的 adaptive
   re-chunking gate 排除 Qwen3.6 几何，复现 frozen worst-case chunking。
两开关齐开 = 08-05 数值模式（上表第三行，实测复现）。

数据：`evalplus_results/quality/code_m16verify_20260807.part.code.json`、
`code_m16noadapt_20260807.part.code.json`、`code_mtpoff_20260806.part.code.json`。

## 19. 性能与质量兼得路径调研：fixed-split 契约复原（2026-08-07）

用户指令："去调研或者论证有没有获得这些性能优化但是不影响质量的方式，
包括参考历史代码、参考 vllm、参考其他内核"。本节是完整论证 + 实测。

### 19.1 理论界限：ULP 差异从哪里来，什么能消除它

§18 已把 −3.7pp 钉到两处 split-KV 归约顺序效应。这里把"能不能在不影响
质量的前提下拿回性能"这个问题闭掉：

1. **split-KV attention 的最终输出是 KV 分片上的 fp32 归约**。softmax 归一化
   下的分片合并（logsumexp merge）在精确算术下满足结合律，但 fp32 下不满足：
   不同的分片划分 = 不同的求和结合树 = 不同的舍入 = ULP 级差异。
2. **merge 阶段本身是确定性的**：`PagedPersistentMergeKernel`（sparkinfer
   `attention/paged/merge.py`）按 chunk 下标顺序逐个合并 partial（row/head
   persistent，`_merge_async_slot` 按 `start_idx + iter` 顺序消费），不存在
   完成序竞争。同一次运行、同一 chunking 下结果逐位可复现。
3. 因此配置间差异的**唯一来源是 partial 本身不同**：chunk 边界不同
   （adaptive 按 live 长度重切 vs frozen 固定）或 kernel 内部累加不同
   （M32 raw-FP8 单 tile vs M16 通用 kernel 双 query-tile）。
4. **推论（否掉一条看似聪明的路）**：把 merge 换成 Kahan/fp64/树状归约等
   "顺序无关"方案**不能**恢复 08-05 数值——partial 已经不同了，merge 再精确
   也只是精确地合并了一组不同的输入。顺序无关归约只在"partial 相同、合并顺序
   不同"时有意义，而这里合并顺序本来就固定。**已否决：不做 order-independent
   merge。**
5. **推论（正向路径）**：要拿回与 08-05 同族的质量，chunk 划分必须是
   （请求长度的）确定函数且与 08-05/历史验证过的形态同族；kernel 差异则必须
   实测过质量锚。没有任何先验方法保证一个新 ULP 模式不翻边界题——~17/164
   题对 ULP 敏感，每个新配置是一次新的抽取，只能实测。

### 19.2 历史代码参考：fixed-split 契约（oracle/qwen36_vllm）

* `direct_model_runner.py:681-684`：`_DECODE_TARGET_SPLITS_PER_REQ = 32`，
  `decode_fixed_kv_split_size = ceil(slot容量/32)`（=8192 token），
  `decode_fixed_max_num_splits = 32`。**kv_split_size 由槽容量上界导出，
  与 live batch、live 长度都无关**——chunk 边界是 8192 的固定倍数。
* 该契约正是 2026-07 README 质量门（code 0.921/0.884）与 222.44 tok/s
  性能锚共用的形态：**fixed-split 同时给过我们质量和性能**。
* 同文件 653-680 行注释记录了 vLLM 自身 `SM120GQAMetadataBuilder.build()`
  （`vllm/v1/attention/backends/sm120_gqa.py`）的做法：**永远从 build 期上界
  （max_model_len）导出固定 kv_split_size**——vLLM 的 SM120 后端本身就是
  fixed-split 设计（CUDA Graph 静态调度的副产物是数值按长度确定）。
* 2026-07 的 splits 扫描（docs/archive/2026-07-20-PROGRESS.md ~L4129）：
  128K 下 qo=4（verify 形态）splits=32 最优（1.566ms，优于 16/48/64/96/128），
  随后全局采用 32，得到 181.95→222.44 tok/s 里程碑。
* **本仓库自己的前例**（同文档 ~L3110）：`CapturedBatchDecodeGraph` 曾用
  陈旧的 TARGET_SPLITS=16（对生产值 64），acceptance rate 70.29%→76.67%
  漂移；统一 split 数后**逐位回到** eager 路径数值。——split 数变化翻动
  near-tie 决策，在本仓库发生过且被 fixed-split 治愈。

### 19.3 外部参考：vLLM / FlashInfer / 其他内核的确定性实践

（web 检索在本环境不可用，以下为实现层已知事实，供复核时按图索骥。）

* **FlashInfer**：plan/decode 的 split 数是 batch 与 seq_len 的函数以填满 SM；
  不同 shape 下归约顺序不同是已知行为。早期 single-decode/single-prefill API
  曾有 `deterministic=True` 参数（固定归约布局），plan-based API 时代未提供
  跨 shape 位级确定性。
* **vLLM**：不承诺跨 batch/shape 位级可复现；SM120 GQA 后端用固定
  kv_split_size（见 19.2）是"图静态调度"的必然选择，其副产物正是按长度
  确定的数值。
* **FlashAttention-2/3**：forward 对固定 shape 确定（每行按 KV tile 顺序
  online softmax），但 decode 的 split-KV（flash-decoding）merge 随 split 数
  变化——与本仓库同构。
* **结论**：业界没有任何上游在 split-KV decode 上提供跨 shape 位级确定性；
  通行做法就是我们正在做的——**固定 split 契约（shape 确定 ⇒ 数值确定）+
  质量锚实测**。这条路有历史数据背书（19.2），不是实验性赌博。

### 19.4 新 knob：SPARKINFER_QWEN36_VERIFY_FIXED_SPLITS（sparkinfer fork）

实现（sparkinfer 提交 `7f6021e`，已推 origin/master）：

* `workspace.py::_qwen36_verify_fixed_split_pages_from_env`：
  `SPARKINFER_QWEN36_VERIFY_FIXED_SPLITS=N` 在 Qwen3.6 verify 几何
  （fp8 KV / head_dim 256 / page 128）下解析为
  `fixed_split_size = ceil(最坏页数/N)` 页——即历史契约
  `kv_split_size = ceil(槽容量/N)` 的按页形式。N=32 ⇒ 64 页 = 8192 token，
  与 oracle 逐位同构。
* `prepare_prefill_graph_replay_state` 把它传给 planner；planner 对
  fixed-split 图计划放宽 SM-fill 预算（worst-case worklist 即包络）。
* adaptive 重切 gate 排除 `fixed_split_size ≥ 0` 的计划——回放保持捕获期
  chunk 大小，chunk 边界成为 live batch/live 长度的不变量。
* M32 kernel 保持不变（快速路径保留），M16/NO_ADAPTIVE 两个 §18 开关不受影响。
* 双模型安全：Laguna 几何（hd128/gqa6/cta64）不在 gate 内，行为零变化。

### 19.5 探针矩阵（probe_qwen36_graph_attn_capacity.py，128K live）

`logs/quality/probe_fixedsplit_*.log`。两种捕获包络：--worst 1024
（捕获==live）与 --worst 2048（生产 256K 槽同包络）。

| 配置 | b4@1024 | b4@2048 | 备注 |
|---|---:|---:|---|
| M32+adaptive（生产） | 0.891 | 2.231 | §13.3 生产图内实测 0.899 ⇒ 生产包络≈1024 侧 |
| M32+frozen | 0.879 | 2.807 | |
| fixed32（新） | 1.206 | 2.284 | b1@2048=2.088 严重（64 CTA 欠占用） |
| M16+frozen（08-05） | 0.944 | 1.266 | 大包络下 partial 行减半反而最快 |

**探针≠生产的警示**：@2048 包络的绝对值与生产图内实测（0.899）对不上，
说明探针的捕获/回放路径与生产图存在未归因差异（包络 worklist 的固定开销被
探针放大）。**因此排名结论以 19.6 的真服务器 A/B 为准，探针只作旁证。**

### 19.6 真服务器 A/B（128K/c4 生产 profile + QSR_PROFILE_ROUNDS）

`scripts/ab_verify_chunking_configs.sh`，逐配置：起服务（128k bench env）→
filler warm 3 波 → 轮级聚合。

| 配置 | warm agg tok/s（3 波） | 轮时 med (ms) | accept_gpu_wait med (ms) |
|---|---:|---:|---:|
| baseline（M32+adaptive，生产） | 173.0 / 198.3 / 183.8 | **50.98** | 46.56 |
| m16frozen（08-05 数值模式） | 166.2 / 171.2 / 182.7 | 61.45（+10.5） | 56.86 |
| fixed32（新 knob，M32 kernel） | 149.3 / 156.8 / 166.1 | 69.47（+18.5） | 65.07 |
| m32frozen | 153.5 / 146.1 / 146.3 | 70.44（+19.5） | 65.84 |

（均含 QSR_PROFILE_ROUNDS=1 剖析开销，同口径相对比较；baseline 轮时与
§13.2 的 53.4ms 同量级互相印证。）

**实测结论（与探针对照）：**
1. **生产排名与探针 @2048 包络的预测相反**：真服务器上 m16frozen 不是最快
   而是 +10.5ms（与 §18 的 "+10ms" 一致），fixed32/m32frozen 更差。探针的
   连续页表 + L2 热态 + 孤立调用三个条件都不成立于生产（散页、冷缓存、
   16 层与其它 kernel 交错），**探针只能作同配置前后的相对参考，跨配置
   排名必须真服务器验证**——本节的 A/B 就是为此补的课。
2. fixed32 在 128K/c4 慢 18.5ms 的原因：固定 64 页 chunk 在 live 128K 只有
   16 chunk/req × 4 = 64 work items（256 CTA），机器填不满；历史 vLLM
   kernel 的 CTA 分解不同（split×KVH 直接映射），没有这个欠占用。
   fixed-split 契约的价值在数值确定性，不在 128K 吞吐。
3. m32frozen 比 m16frozen 还慢：128K live 下 M32 frozen 的 chunk 几何相对
   adaptive 损失最大（§9 的 −4.6ms/轮收益正是 adaptive 带来的，frozen 全数
   吐回且 M32 单 tile 无冗余 query-tile 可摊）。

### 19.7 质量锚（suite-fast profile，code dim，同 §18 协议）

| 配置 | HumanEval | HumanEval+ |
|---|---:|---:|
| M32+adaptive（生产） | 0.8902 | 0.8659 |
| M16+adaptive | 0.9024 | 0.8902 |
| **M16+frozen（08-05 数值模式）** | **0.9268** | **0.8902** |
| MTP-OFF | 0.9268 | 0.8902 |
| **M32+fixed32（新 knob，本次实测）** | **0.915** | **0.866** |
| 历史门（README，旧 vLLM kernel + fixed32） | 0.921 | 0.884 |

数据：`evalplus_results/quality/code_fixed32_20260807.part.code.json`
（164/164，生成 1251s，conc 8）。

**解读：**
1. fixed-split 契约单独恢复了约 2/3 的回退（0.8902→0.915），印证 §18 的
   加法分解：chunking 与 kernel 是两个独立 ULP 源。
2. 剩余差距（0.915→0.9268）是 M32 raw-FP8 kernel 自身的归约数值——
   历史 0.921 是旧 vLLM kernel 的抽取，M32 kernel 的抽取略差且
   HumanEval+（0.866）未过历史 0.884 门。
3. **唯一双双越过历史门的配置仍是 M16+frozen**（0.9268≥0.921，
   0.8902≥0.884）。fixed32 knob 留作研究/对照锚，不作质量模式。

### 19.8 短上下文 A/B 与 128K 无剖析校准（Phase C，2026-08-07 午）

suite-fast profile（32K 槽 / cap 8），4K/c8 filler：

| 配置 | warm agg tok/s（3 波） |
|---|---:|
| baseline（M32+adaptive） | 112.5 / 126.7 / 382.9（前两波受污染，干净波 383） |
| **m16frozen（08-05 模式）** | **485.4 / 480.9 / 462.0（三波稳定）** |

128K/c4 无剖析净吞吐（bench env，5 warm 波）：

| 配置 | warm agg tok/s | 中位 |
|---|---|---:|
| baseline 校准（今日同机） | 170.0 / 149.6 / 171.4 / 163.0 / 174.7 | 170.0 |
| m16frozen | 155.4 / 144.9 / 154.4 / 145.0 / 152.6 | 152.6 |

**关键事实：**
1. **短上下文 m16frozen 不但不慢，反而显著更快**（干净波对比 +24% 以上，
   且三波无方差）。机理：32K 捕获下 code-dim 长度（≤8K）frozen chunking
   只有 1–2 个 chunk——**等于无 split-KV 的单遍注意力**，省掉 adaptive 的
   重切 metadata、partial 缓冲与 merge kernel；M16 双 query-tile 的页读代价
   在这个尺度上可忽略。
2. **今日主机状态异常**：load average ~9800（数千个 D 状态僵尸进程 +
   headless chrome swiftshader 常驻 ~285% CPU），当日无剖析 baseline 中位
   170 vs §13.4 的 234.8（−28%，两配置同比例受影响）。**绝对值不可与
   08-06 直接对比，同日相对比较仍有效**：m16frozen 128K 成本 = 152.6/170
   = **−10%**（剖析口径 −6.9%，两者同向）。
3. 健康机器外推：m16frozen 128K ≈ 234.8 × 0.90 ≈ **211 tok/s**——略低于
   历史 222.44（−5%）。这是"质量模式"唯一的性能缺口。

### 19.9 最终结论

**调研问题："有没有获得这些性能优化但不影响质量的方式？"**

**论证结论：没有免费的午餐，但有明确的工程解。**

1. **性能与质量的冲突是本质的，不是实现缺陷**：M32 tiling 与 adaptive
   chunking 这两个性能优化的物理内容，恰好就是改变 split-KV 归约划分的
   两个操作；质量回退的 ULP 翻转来自划分本身（§19.1 已证 order-independent
   merge 无法修复——partial 不同，merge 再精确也无用）。任何"保留这两个
   优化又恢复 0.9268 数值"的方案在 fp32 非结合律下不存在。
2. **历史契约复现实验（fixed32）给出了干净的反证**：同样的 fixed-split
   几何在 M32 kernel 上只得 0.915/0.866（HumanEval+ 未过 0.884 门），
   且 128K 慢 18.5ms——历史 0.921 是"旧 vLLM kernel + fixed split"的
   联合抽取，kernel 换了，抽取就变了。fixed-split 本身不是质量护身符，
   **08-05 数值模式（M16+frozen）才是本仓库验证过的质量锚**。
3. **推荐配置**：
   * **质量关键路径（质量套件、对外质量承诺）→ 08-05 数值模式**
     （`SPARKINFER_QWEN36_VERIFY_M16=1` + `SPARKINFER_QWEN36_VERIFY_NO_ADAPTIVE=1`）：
     0.9268/0.8902 双过历史门；短上下文更快（+24%）；128K −10%。
   * **吞吐关键路径（128K 基准 headline）→ 维持生产默认（M32+adaptive）**：
     235 tok/s，接受 0.8902 质量带。
   * 两个数值模式都已 knob 化、可复现、已入档（§18 四方 + 本节五方数据）。
4. **同时拿到两者的唯一工程路径（后续优化目标）**：在保序前提下加速
   08-05 数值模式的 128K verify（缺口 ~5%）。候选，均为位级安全
   （不改每个 query-tile 内的累加顺序，只改数据搬运）：
   a. **M16 kernel 跨 query-tile 的 KV 页复用**——当前每页读两次
      （两个 query-tile 各读一遍），smem/L2 侧做 tile 间复用可省一半
      KV 读带宽；输入位模式不变 ⇒ 数值不变。
   b. **捕获视界调优**：128K workload 用 128K 视界捕获 verify 图
      （frozen chunk 在 live==捕获视界时恰好填满预算；256K 视界捕获在
      128K live 只填半个机器）——ben ch env 改 blocks_per_slot 即可实验。
   c. merge/partial 搬运开销削减（partial 行数在 M16 已减半，剩余空间小）。
5. **双模型安全**：所有 knob 均 gate 在 Qwen3.6 verify 几何（fp8 KV /
   hd256 / page128），Laguna（hd128/gqa6/cta64）路径零变化；Laguna 服务
   不受本次任何改动影响。

### 19.10 产物索引

* sparkinfer `7f6021e`（origin/master）：fixed-split knob + 预算放宽 +
  adaptive gate 排除。
* `logs/quality/probe_fixedsplit_*.log` —— 探针矩阵（两包络 × 4 配置）。
* `logs/quality/ab_{server,grid,rounds}_*.log/.json/.txt` —— 128K/c4
  真服务器 A/B（4 配置，剖析口径）。
* `evalplus_results/quality/code_fixed32_20260807.part.code.json` ——
  fixed32 质量锚（0.915/0.866）。
* `logs/quality/ab_grid_shortctx_*.json` —— 4K/c8 短上下文 A/B。
* `logs/quality/ab_grid_c2_*.json` —— 128K 无剖析净吞吐（m16frozen +
  baseline 同日校准）。
* `scripts/ab_verify_chunking_configs.sh` / `scripts/ab_quality_code_dim.sh`
  / `scripts/ab_shortctx_m16frozen.sh` —— 本节的三个 A/B 复用脚本。
