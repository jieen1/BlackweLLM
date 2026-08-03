# BlackweLLM 路线图（2026-08 → 2027-01）

> 编制日期：2026-08-01 · 基线 commit：`ce21eb5` · 本文档取代
> [`docs/archive/2026-07-26-roadmap-vllm-removal.md`](archive/2026-07-26-roadmap-vllm-removal.md)
>
> 本文档中所有"现状"数字均为 2026-08-01 在本仓库实测所得，来源在正文标注。
> 标注 **[待验证]** 的条目是尚未在本机跑过的假设，不作为决策依据，只作为待办。
>
> **2026-08-01 二次修订**（基线推进到 `6acc4ba`）：消化
> [`investigation-queue.md`](investigation-queue.md) §D 的上游调研结论，重排优先级；
> §7 的 D3（GPU CI 形态）、D6（Qwen3.6 主线 checkpoint）拍板，加上 N8
> （`--session-affinity`，见 [`implementation-plan.md`](implementation-plan.md) §6.1）拍板；
> 新增风险 RK9（冷启动/首次真实形状路径）。本轮那三条 [待验证] 事项**已全部答完**
> （B0-8 GDN、KV dtype 选型、torch wheel 是否带 `sm_120`），结论见下方待验证清单与
> [`investigation-queue.md`](investigation-queue.md) C-1/C-2/C-3。

---

## 0. 定位变更（这是本次路线图重写的原因）

**旧定位**：只服务 `Laguna-S-2.1-NVFP4` 一个模型，把 SM120 上的极限性能榨干，
拿一个漂亮的数字去发布。

**新定位**：**Blackwell SM120 单机推理运行时**。硬件面收窄到极致（只有
SM120、只有单机），换取在这个窄面上把**稳定性、易用性、模型兼容性**做到
生产可用；性能从"主线目标"降级为"机会主义优化"。

变更依据：**Laguna-S-2.1 的模型能力经使用判断为一般，继续在它身上做深度优化
的边际收益不足以支撑一次发布。这是产品判断，不是评测结论**——本仓库**从未**
对 Laguna 跑过 MMLU-Pro 或任何官方对标评测（`evalplus_results/official/` 下
全部三份结果的 `model` 字段都是 `qwen3.6`，2026-07-22）。

> 本节初稿曾写"MMLU-Pro 84.5%"作为依据，那是 **Qwen3.6-27B** 2026-07-22 那次
> 跑分（`mmlu_pro_think_c4.json`：`model=qwen3.6, acc=84.54, n=414`）被误标到
> Laguna 名下。转向的决定本身不受影响，但**它没有评测数据支撑**，不要再引用
> 那个数字论证 Laguna 的能力。补测 Laguna 的质量基线已排为 Track C 的 C9。
但为它建起来的这套东西——自研 SM120 执行栈、固定槽位调度、CUDA Graph
生命周期、前缀缓存、双协议 API、bfdiag 诊断平台——是**与模型无关的资产**，
值得围绕它重新组织目标。

### 收窄的硬件合同（不再讨论，作为公理）

| 维度 | 合同 | 含义 |
|---|---|---|
| GPU 架构 | **仅 SM120 / CC 12.0** | 不做 SM90/SM100/消费级以外的兼容；启动即检测，不匹配直接拒绝启动 |
| 拓扑 | **单机、单进程** | 无 TP / PP / EP / 多机；`world_size=1` 是硬编码前提，不是配置项 |
| 卡数 | **单卡优先，多卡不在本路线图内** | 96 GB 单卡是容量规划基准 |
| 权重精度 | **NVFP4 优先，FP8 次之** | SM120 无 BF16 tensor core，BF16 权重不是一等公民 |
| KV 精度 | **FP8 e4m3 优先** | |

这个合同的价值在于**它允许我们删代码**：任何为"未来可能的多卡/多架构"
保留的抽象，都应该被删掉，而不是留着长草。

### 北极星指标（按优先级）

1. **能跑起来**：拿到一个受支持的 checkpoint 路径，一条命令启动，不需要人肉算
   `blocks_per_slot`。
2. **不会崩**：24 小时连续压测无 slot wedge、无显存泄漏、无需重启。
3. **输出可信**：与参考实现（HF transformers / 上游框架）在贪心解码下 token 级对齐，
   有回归门禁看着。
4. **够快**：在上述三条成立的前提下，再谈 tok/s。

---

## 0.5 主线（2026-08-02 晚，用户定调）：完整支持 NVFP4 Qwen3.6，然后压榨 SM120

标准模型是 **`unsloth/Qwen3.6-27B-NVFP4`**（不是 `nvidia/...`——今天的脚本指向后者是偏离，
历史 47 个 benchmark 脚本指向前者才是对的）。

**四个阶段，顺序不能颠倒**：

> **2026-08-03 复核**：本节多数条目在写下后已被完成，但勾一直没打。以下每条都
> 重新对代码核实过（不是凭记忆），证据写在条目里。**阶段 4 的原处方被实测推翻**，
> 见该节。

### 阶段 1 · 完整支持 —— ✅ 三条全通
- [x] `resolve_checkpoint(unsloth)` 放行 —— `mixed-precision` 已回到
      `SUPPORTED_QUANT_FORMATS`（`runtime/model_registry.py:108`）。当日移除它的理由
      （"门禁不该承诺 loader 做不到的事"）成立，但**解法选错了**：应该补 loader 而不是
      缩门禁；缩门禁把标准模型锁在了门外。门禁改为按 `(quant_method, format)` 二元组
      判定（`11b0e70`），`tests/test_registry_quant_format_gate.py` 钉住。
- [x] `load_weights` 零缺失零多余 —— adapter 已合入并验证（`ca50017`）：
      `runtime/loading/compressed_tensors.py` 的 `MixedPrecisionQuantMap` +
      `runtime/model/compressed_tensors_linear.py`。原先缺的 168 个 `weight_packed` 归零。
- [x] 能出连贯输出 + C-LIVE 通过 —— `"!!!!!!!!!!!!"` 已定位并修复:**根因是
      `weight_global_scale` 是 `weight_scale_2` 的倒数**（不是改名），按改名处理就会把
      每个权重缩放到爆。C-LIVE **64/67**，3 个失败逐个查实为模型自身行为而非服务缺陷
      （见 `../notes/2026-08-03-std-model-serving-acceptance.md`）。
      ⚠️ 顺带纠正一条工具链默认值：默认 `QSR_TOOL_CALL_PARSER=poolside_v1` 是给 Laguna 调的，
      标准模型要用 `qwen3_coder`——这一项当时吃掉了 10 个失败里的 7 个。
- **判据**：这三条全通才算"支持"，不是"能加载" —— **已达成**

### 阶段 2 · 系统质量与结构（进行中）
- [x] 把 unsloth 纳入 registry/loader 的常规测试面，而不是特例 ——
      `tests/test_qwen36_mixed_precision_checkpoint.py` +
      `tests/test_registry_quant_format_gate.py`（后者用合成 config，空 HF 缓存下也成立）
- [x] ~~现有 B1/B2/B3 的脚本与 fixture 统一切到标准模型~~ —— **已完成**（`cbd6c03`）。原记录：
      实测 22 个脚本硬编码 `models--nvidia--`（modelopt 格式），只有 1 个指向标准模型，
      且每个脚本自带一份带 snapshot 哈希的 `MODEL_PATH`，无统一解析点。
      ⚠️ **不能无脑替换**：两个 checkpoint 是不同量化格式，专门验 modelopt adapter 的脚本
      必须留在 `nvidia/` 并注明原因。
- [x] `IMPLEMENTED_BACKENDS` 与实际可服务范围一致 ——
      `frozenset({"laguna", "qwen36"})`（`runtime/model_registry.py:152`），两者都可服务。
      注：qwen36 不支持 DFlash（`ServerEngine._load_qwen36_model` 会显式抛错），
      这是 backend 内的子能力差异，不是 backend 清单不一致。

### 阶段 3 · 输出速度
- [x] ~~**首要项：消除反量化缓存**~~ —— **已消除**。原文记录的是
      "CG warmup 5 秒内 27,259 → 76,052 MiB，其中 49.72 GiB 是永久缓存的 BF16 反量化权重，
      模型本体只有 18.77 GiB"。现在稠密 NVFP4 层直接吃打包 FP4（见阶段 4），
      并在融合权重备好后释放原始 NVFP4 参数（`free_nvfp4_raw_params()`，
      `runtime/model/qwen36_model.py:2011`，`5fce64e`）。**76.34 → 53.08 GiB。**
      ⚠️ 原文"NVFP4 量化完全白做"的判断在当时成立，现在不再成立。
- [x] ~~CUDA Graph 捕获下的 decode 吞吐从来没测过~~ —— **已测，结论是本阶段最大的一条：
      在册的每个吞吐数字都比运行时的实际能力低约 5 倍。**
      服务路径、标准模型、同 prompt 同参数同槽位、只切 `QSR_SERVER_ENABLE_CUDAGRAPH`：
      **CG 28.848 tok/s vs eager 6.120 tok/s = 4.71×**，且 **CG 还少用 5.30 GiB**
      （72.39 vs 77.69）。捕获本身不贵（启动 24.9s vs 21.0s）。
      详见 [`../notes/2026-08-03-cudagraph-vs-eager-decode-throughput.md`](../notes/2026-08-03-cudagraph-vs-eager-decode-throughput.md)。
      ⚠️ **所有基于 ~6 tok/s 做出的优化判断都需要重估**，起点是 28.85 不是 6.1。
- [x] ~~先做 profiling 定位瓶颈，而不是继续猜~~ —— **已做**，见
      [`../notes/2026-08-03-decode-kernel-profile.md`](../notes/2026-08-03-decode-kernel-profile.md)。
      **CG 下 GPU busy 31.01ms / 墙钟 34.67ms = 89%，已经 kernel-bound**；
      eager 下只有 21% 忙（CPU 侧 paged 元数据规划约 34.2ms/step，`plan_metadata_to_device`
      一项就 24.3ms）——这就是 4.71× 的机制。
      **GDN 递归 kernel 只占 0.6%**，独立证实"GDN 不是决定项"，别再往那投。
- [x] ~~查清那些 BF16 kernel 来自哪些层~~ —— **已按调用次数精确归属，无猜测成分**：

      | kernel 组 | 次/step | ms/step | 是什么 |
      |---|---:|---:|---|
      | `W4A16FusedMoeKernel` | **56** | 10.752 (35.0%) | 0–55 层 MLP（NVFP4，每层一次） |
      | `cutlass_80_wmma` ×3 + `gemvx` ×2 | **233** | 13.830 (45.0%) | **全部 FP8 层反量化后的 `F.linear`** |
      | GDN 递归 | — | 0.187 (0.6%) | 不是量级项 |

      FP8 层预期数 = full_attn 16×4 + GDN 48×3 + lm_head 1 + 56–63 层 MLP 8×3 = **233**，
      与实测调用数**一个不差**；NVFP4 侧 56 预期 = 56 实测。
      ⚠️ 其中 24.8% 跑在 `cutlass_80_wmma_tensorop` 上——**为 SM80 编译的 Ampere 代
      kernel，在 cc 12.0 的卡上**。cuBLAS 按形状在 `gemvx`/SM80 WMMA 之间分派。
- [x] ~~**杠杆②：NVFP4 层改走 W4A4**（35%，10.75 ms/step）~~ —— 🔴 **试过了，不可用。**
      前提确实是对的（kernel 契约完全匹配、checkpoint 真的声明 W4A4 且发货
      `input_global_scale`），**但数值上就是差**：单层 cosine 0.988–0.989 对现有 W4A16 的
      0.999984–0.999990，**差约 30×**；B1-R **全线不过**（median gap 0.5 / 判据 0.25，
      p90 0.875 / 0.5，最差负载 `mean_kl_topk` 7–8e-3 / 5e-3），且 `instruction` 负载
      25–65 步内就**发散到溢出 top-1024 捕获窗口——比 B1-R 校准集里任何注入 bug 都差**。
      未测速度（正确性已不过，测速只会诱使放宽判据）。生产路径未动。
      详见 [`../notes/2026-08-03-w4a4-blockscaled-negative-result.md`](../notes/2026-08-03-w4a4-blockscaled-negative-result.md)。
- [x] ~~**杠杆①：FP8 层改走 W8A8**（45%，13.83 ms/step，233 次调用）~~
      —— 🔴 **也不可用。判据预演奏效了，省掉一整轮 kernel 实现。**
      方法：不写 kernel，只在现有 forward 里插入激活的 FP8 往返
      （权重侧两种设计取值相同，**新增误差的主导项就是激活量化**），
      这是真实 W8A8 误差的**下界**。已验证不是空操作（93% 激活元素被改变）。
      单层 cosine 0.9996；B1-R：`median_gap_error` **0.25 贴着判据零余量**
      （干净跑约 0.125），而 **`instruction` 负载发散到溢出 top-1024 捕获窗口、
      根本没产出数字**——**所以那些"在线下"的 bar 都是排除掉最差负载后算的**。
      与 W4A4 同一失败签名。未测速度。
      详见 [`../notes/2026-08-03-fp8-w8a8-preflight-negative.md`](../notes/2026-08-03-fp8-w8a8-preflight-negative.md)。
- [ ] 🔴 **阶段四两根杠杆都撞墙了，且撞在同一处：这个模型对激活精度敏感。**

      | 杠杆 | 占解码 kernel | 结论 |
      |---|---:|---|
      | NVFP4 → W4A4 | 35% | ✗ B1-R 全线不过 |
      | FP8 → W8A8 | 45% | ✗ 误差下界即溢出捕获窗口 |

      两者都是"把**激活**也降到 4/8 bit"；而现有的**权重**量化路径 cosine 0.99999，
      毫无问题。**别再找更激进的激活量化，这个方向已两次撞墙。**
      **① 换 Blackwell 原生 BF16 kernel —— 也查过了，没有可用的。**
      `sparkinfer.gemm.bf16_gemv` 方向对（BF16×BF16，不动激活精度、CUDA-graph 安全），
      **但先量后做**：它限 out≤1024/in≥1024，按真实形状筛 237 个投影只中 **34 个(14%)**，
      且都是最小的那些——**约占 FP8 GEMM 工作的 1.7%、整个解码 kernel 时间的 ~0.8%**，
      不值得做。主导形状全是宽输出(12288/10240/17408/248320)，需要真正的稠密 GEMM；
      而 `sparkinfer/gemm/` 里除 `bf16_gemv` 外**全是量化 GEMM**，
      用它们就等于量化激活——正是上面已证伪的那堵墙。自研更慢 2.4–3×（本仓库既有调研）。
      **⇒ 那 24.8% 的 `cutlass_80_wmma` 看着刺眼，但没有更好的替代品。**
      ② 回收 FP8 反量化缓存 —— ✅ 已做（见生产显存审计那条，实收 5.79 GiB）。
      详见 [`../notes/2026-08-03-stage4-kernel-levers-exhausted.md`](../notes/2026-08-03-stage4-kernel-levers-exhausted.md)。
- [ ] 🔴 **但"因此没得优化了"这个总结已被推翻(同日晚)——decode 步比历史慢约 1.6×。**
      按每步实际读取量换算有效带宽:**历史 564 GB/s(128K/c=4)vs 今天 343 GB/s(短上下文/c=4)**,
      而且历史还额外扛着 128K 的 attention。**同模型、同卡。**
      **"已 kernel-bound"不等于"kernel 已最优"**——上面四条否定各自仍成立,
      但缺口在**当前 kernel 组合本身比历史那套慢**,不在"没有杠杆"。
      见 [`../notes/2026-08-03-performance-gap-vs-historical.md`](../notes/2026-08-03-performance-gap-vs-historical.md)。
      **下一步:在 128K/c=4 同口径实测坐实,然后按历史文档的分解逐项对账。**
- [x] ~~**MTP 接进服务路径**~~ —— 已接(默认关),但 🔴 **不可上生产,原因是架构不是接受率**:
      服务路径实测 **MTP 关 28.0 tok/s vs MTP 开 7.80 tok/s = 0.28×,慢 3.6×**。
      根因:`Qwen36MTPEngine.round` **全程 eager,从不走 `decode_batch_sampled`**
      ——那是被捕获的 CUDA Graph 唯一重放的路径。MTP 一开,`classify_decode_slots`
      把每个 slot 都路由过去,**捕获没被关掉,只是不可达**。
      等于拿 4.71× 的 CG 收益换 1.54/4 的接受率。
      DFlash 靠自建 draft/verify CUDA Graph 规避了这个,`Qwen36MTPEngine` 没有对应物。
- [ ] 🎯 **MTP 要能用,必须先给 verify/draft 路径做 CUDA Graph 捕获**(参照 DFlash)。
      在那之前接受率再高也没意义。
- [x] ~~接受长度退化~~ —— (token,hidden) 配对 bug 已修并实测:
      prose 1.20 → **1.54**,code 1.67 → **1.82**。真实改善但幅度有限,
      **远不足以抵消上面那 3.6×**。C-LIVE **66/67**(基线 64/67,无退化)。
- [x] ~~**`assert_all_params_loaded` 是单向的**~~ —— **已补反向检查**（`0ddab29`：`warn_on_unconsumed_tensor_families`，按名字尾部分族，警告而非抛错）。原记录：：保证"每个模型参数都拿到 checkpoint 张量"，
      **不保证"每个 checkpoint 张量都被消费"**。`input_global_scale` 因此被静默丢弃了很久
      （W4A4 调查时才发现，本次无害但盲区是真的）。补一个反向检查。
- [x] ~~持久化 kernel 编译缓存~~ —— **本来就有，而且默认开着；我先前说"`cache_key`
      只是进程内记忆化"是错的，现予纠正。**
      `sparkinfer/_lib/compiler.py` 有三层：spec memo、内存 LRU、**磁盘缓存**。
      `cache_key` 喂给 `KernelCompileSpec.from_key(...)`，就是磁盘缓存键的一部分。
      开关 `SPARKINFER_COMPILE_DISK_CACHE`（**默认 `"1"`**）、目录
      `SPARKINFER_COMPILE_CACHE_DIR` 或 `$XDG_CACHE_HOME/sparkinfer`。
      佐证：`~/.cache/sparkinfer` 有 297 MB，而今天多次起服务**零文件写入**——
      是命中，不是没用。`compile_cache_info()` 可读 hits/misses 逐项核实。
- [ ] 那么首请求 TTFT 4.67s（之后稳定 0.25s）**不是编译**，是别的
      （`sparkinfer/gemm/bf16_gemv/_kernel.py:207` 提到 "first-launch lazy module load"）。
      **重新定位这 4.4 秒**再谈优化，别按"编译慢"去治。
- [x] ~~并发/批量下的 CG 收益未测~~ —— **已测**（capacity 1/2/4，同 harness 同变量）：

      | capacity | CG tok/s | eager tok/s | 比值 |
      |---:|---:|---:|---:|
      | 1 | 28.56 | 5.95 | **4.80×** |
      | 2 | 47.71 | 12.34 | **3.87×** |
      | 4 | 68.59 | 19.58 | **3.50×** |

      **优势随并发收窄但不消失**：CG 消掉的是每步固定的 CPU 侧开销，并发越高越被摊薄，
      所以 eager 能追回一部分——但 cap4 仍差 3.50×，**"并发上来就不需要 CG"是错的**。
      CG 自身扩展次线性（1→2→4 并发只给到 1.67×/2.40×），与"已 kernel-bound"互为印证。
- [ ] MTP：默认 K 已从 8 改到 4（`f616029`，K 曲线实测 prose 1.11× / code 1.38×）；
      重同步 A/B 数据待回（代码在 `work/mtp-resync-20260802` 的 `aed0e2d`，无数据）
- [x] ~~全部 MTP 接受率数字都测自非标准 checkpoint，需在标准模型上重测~~
      —— **已重测，checkpoint 发布方假说被证伪。**
      标准模型 K=4：prose 接受率 30.0%、每轮平均接受 **1.20**；code 41.7%、**1.67**。
      与 `nvidia/` K=8 的记录对齐着看跨 K 可比的"每轮平均接受"：
      **prose 1.21 vs 1.20，几乎完全相同。** 草稿头在哪个 checkpoint 上都只蒙对 ~1.2 个
      token。历史 "~4.0/4" 换 checkpoint 也追不回，**别再当靶子**。
      正确性通过（投机与非投机 committed 序列逐 token 相同）。
      详见 [`../notes/2026-08-03-mtp-acceptance-on-standard-checkpoint.md`](../notes/2026-08-03-mtp-acceptance-on-standard-checkpoint.md)。
      ⚠️ code 那一行 K 不同（8 vs 4）**不构成结论**，需同 K 重测。
- [ ] ⚠️ **MTP 根本没接进服务路径——所以"MTP 在 CG 下的 e2e"不是一次待做的测量，
      而是一项待做的实现。**（2026-08-03 核实：`Qwen36Backend.capabilities.
      speculative_decode = False`，`server/engine.py::_load_qwen36_model` 的 docstring
      也明说了。MTP 只存在于 `scripts/b3*`，从未进过 `server/app.py`。）
      现有全部 MTP 数字的基线因此都是 eager 的（本次非投机基线 5.97–6.02 tok/s），
      而服务路径开 CG 是 28.85 tok/s。这对投机**不是无关缩放**：MTP 赚不赚取决于
      "一次 verify 是否比 K 次顺序 decode 便宜"，CG 恰好把 decode 那一侧变便宜了 4.71×
      ——**门槛是被抬高而不是降低了**。
      加上已实测的接受率只有 1.20–1.67 / K=4，**先别投实现**；
      要评估就先只做一次"verify 成本 vs K 次 CG decode 成本"的对照，再决定。
      （接受率结论不受影响：那是权重性质，与 CG 无关。）
      **本 session 第二次踩到同一形状——基线跑 eager，结论当成运行时性质。**

### 阶段 4 · Kernel 深度适配，压榨 SM120
- [x] **已达成"不反量化"这个目标**：`Qwen36MLP` 把 gate/up/down 融成一次退化的
      1-expert/top-1 MoE 调用，走 `sparkinfer.moe._shared.kernels.w4a16` 的
      `run_w4a16_moe`，**直接在打包 FP4 上算**——即原文"Laguna 做对了"的那个性质。
- 🔴 **（事实记录，非待办）原处方（`gemm.blockscaled.mm`）并没有被推翻——我 2026-08-03 早些时候
      在本文写的"处方是错的"本身才是错的，现予纠正。**
      当时的依据是 `runtime/model/modelopt_linear.py:76-84` 记录的失败实验：
      `blockscaled.mm` 要求两个操作数都量化，而"checkpoint 声明 weight-only"。
      **那条记录说的是 `nvidia/` checkpoint，我把它当成了"所有 checkpoint"。**
      实测两者的 `config_groups.group_1`：

      | | 权重 | 激活 |
      |---|---|---|
      | `nvidia/`（那次实验用的） | W4 | **A=None** → weight-only，`blockscaled.mm` 确实不适用 |
      | `unsloth/`（**标准模型**） | W4 | **A=4F → W4A4** |

      而且标准 checkpoint 的 **`input_global_scale` 是实际发货的张量**（0–55 层
      `mlp.(gate|up|down)_proj` 各有一份）。那次失败是因为在 `nvidia/` 上动态量化激活
      "没有 checkpoint 侧对应物、纯属引入误差"——**在标准模型上有对应物。**
      所以 **W4A4 路径对标准模型是开着的，且很可能是阶段四最大的一根杠杆**：
      那 56 层 MLP 现在走 W4A16（kernel 内把权重反量化去乘 BF16 激活），
      实测 **10.75 ms/step、占 kernel 时间 35%**。
      ⚠️ 动手前必须先过 B1-R 的 gap-error 判据——上次就是栽在那里。
- 📌 **（事实记录，非待办）**层构成实测（2026-08-03，读自标准 checkpoint 的 `config_groups`）：
      **group_1 (NVFP4/W4A4)** = `.*mlp\.(gate|up|down)_proj$` 全量，
      但 **group_0 把 56–63 层的 MLP 覆盖回 FP8**，故 NVFP4 实际覆盖 **0–55 层的 MLP**；
      **group_0 (FP8 W8A8, 权重 channel / 激活 per-token)** = attention 的 q/k/v/o、
      `linear_attn` 的 in_proj_qkv/in_proj_z/out_proj、`lm_head`、以及 56–63 层 MLP。
      ⚠️ `scripts/mtpfix_unsloth_checkpoint_probe.py` 的 docstring 把这个**说反了**
      （称 NVFP4 "只覆盖 56–63 层"），已在该文件更正。
- [x] sparkinfer 现在**可以直接改**（2026-08-02 起解除，只动 `origin`）。
      已合入先例：`fused_recurrent_gated_delta_rule_multistep`（`1fd76d1`，17 单测 bit-exact）；
      w4a16 scratch 欠额修复（`8242340`，104 条容量断言）。
- [ ] GDN 侧的剩余项见 §7.1 B3；⚠️ 但**硬上限已排除 GDN 是 MTP 的决定项**
- （FP8 W8A8 接入**不在此处重复列**——它就是上面阶段 3 那条"杠杆①"。
  原记录：单层 cosine 0.9996，比 NVFP4 路径差 30–40×，需要一轮专门的全模型验证。
  **2026-08-03 起改为先做判据预演再谈实现**，理由见杠杆①那条。）

**阶段 3/4 曾被短暂叫停**（用户："不要去反量化啥的 就正常跑"），现按新指示恢复，
**但严格排在阶段 1/2 之后**——先完整支持，再谈速度。

## 1. 现状盘点（2026-08-01 实测）

### 1.1 已经建成的（真资产）

| 资产 | 状态 | 证据 |
|---|---|---|
| vLLM 完全剥离 | ✅ 生产路径零 vLLM 依赖 | `runtime/model_loading.py` / `runtime/laguna_config.py` 自建；vLLM 仅存在于 `oracle/`，已排除出 wheel |
| 自研模型图 | ✅ Laguna 全栈自建 | `runtime/model/`（decoder / linear / embedding / attention 占位 / RoPE） |
| 固定槽位连续批处理 | ✅ | `server/engine.py`，独立引擎线程持有 CUDA context，asyncio 侧无锁 deque + pipe 唤醒 |
| CUDA Graph 生命周期 | ✅ decode / draft / verify 三类图 | `runtime/backends/laguna_cuda_graph.py`（1106 行）、`laguna_dflash_cudagraph.py` |
| 前缀缓存 | ✅ 内容寻址 + 引用计数 + LRU | `runtime/block_pool.py`；同槽 KV 复用 + SWA ring 重建 |
| DFlash 投机解码 | ✅ 接受率 96.3–100% | `runtime/backends/laguna_dflash.py`（1707 行） |
| OpenAI + Anthropic 双协议 | ✅ 含流式 / 工具调用 / logprobs | `server/formats/` |
| Prometheus 指标 | ✅ `blackwellm:*` 命名空间 | `server/metrics.py` |
| bfdiag 诊断平台 | ✅ 飞行记录仪 / run record / 可比性判定 / 热引擎 | `bfdiag/`，CLI `bf`，见 [`diagnostics-guide.md`](diagnostics-guide.md) |
| 自研 SM120 kernel | ✅ router（.cu）+ RoPE / RMSNorm / KV scatter（Triton） | `runtime/kernels/` |

**Laguna 当前性能**（2026-07-31 实测，**2026-08-01 在当前 SparkInfer fork HEAD 上复现确认**，
analytic decode 路径，无 TURBO）：

| 工作负载 | tok/s | 接受率 |
|---|---|---|
| fox-64K | 353–368 ⚠️ **不可比,勿作判据** | 96.9% |
| fox-4K | 353–357 | 96.3–97.0% |
| galaxy-4K | 395–401 | 100% |
| code-4K | 341–359 | 97.8% |

> README 里的 222 / 267 tok/s 是旧数字，已在本次文档整理中更正。

> ⚠️ **fox-64K 那一行不可用作验收判据**（2026-08-02 认定）：同一负载**调用在序列中的
> 位置**能让它摆动约 60%（首次 ≈480 tok/s，其后 ≈298 tok/s，工作量相同），而原记录只有
> 汇总数字、没有逐轮顺序，无从还原它测的是哪种状态。已排除 verify 容量修复与前缀缓存
> 两个嫌疑。**重建时必须发布逐轮序列。** 其余三个负载多次重启复现一致，可信。复现数据、过程、
> 以及过程中发现的两个诊断链路问题见
> [`../notes/2026-08-01-sparkinfer-patch-recovery-and-repro.md`](../notes/2026-08-01-sparkinfer-patch-recovery-and-repro.md)。

### 1.2 曾经是红灯的（Track 0 止血，2026-08-01 处理）

R1–R6、R8 已在 2026-08-01 的 Track 0 批次里解决，保留在表里是为了记录**问题的形状**——
下一次同类问题该往哪看。R7、R9 仍然开着。

| # | 问题 | 证据 | 状态 |
|---|---|---|---|
| R1 | **CI 是红的** | 原诊断是 `tests/test_swa_scratch_lifecycle.py` 等裸 `import torch` 导致 pytest 收集期 ImportError。**实测后修正：流水线根本没走到 pytest**——`ruff check .` 这一步就先红了（`benchmarks/quick_check.py` 自 `e6793bc` 起有 4 个未使用 import）。而且真正违反 CPU-only 契约的模块不是 3 个而是 **5 个**，其余 4 个是经 `bfdiag.workloads` 摸到 sparkinfer、需要 fastapi、`monkeypatch.setattr` 的字符串目标会真的 import sparkinfer 子模块——**只有真跑才暴露，grep 看不见** | ✅ 已解决 |
| R2 | **4 个测试失败** | 装了 torch 后 `926 passed, 4 failed`：3× `test_bfdiag_ring.py::TestVerifyOnlyTrace`（假 backend 缺 `block_size`/`device`，测试替身漂移，非生产 bug）；1× thinking 契约冲突 | ✅ 已解决 |
| R3 | **thinking 标签契约自相矛盾** | `d52a3b1` "Strip thinking tags from all API responses" 与断言"保留 think 标签"的测试同时存在于 main；该 commit 写着 "Tested: unit tests pass"，但那个测试当时就是红的 | ✅ 已定契约，见 §1.4 |
| R4 | **thinking 剥离逻辑有误伤风险** | 根因比"贪婪"更准确：两条正则**没有锚定**，把文本里任何位置的 `<think>`/`</think>` 都当成删除信号。`_ORPHAN_CLOSE_RE = r"\A.*?</think>"` 删掉任何 `</think>` 之前的全部内容 | ✅ 已解决 |
| R5 | **sparkinfer 的性能补丁不可复现** | 2026-07-31 的 gating 放宽**从未提交到任何分支**，工作区被清后丢失（所以按分支做 pickaxe 搜索找不到）。已从悬空提交 `1e306d7`/`ec8bb1eb` 恢复，rebase 到 upstream `3bd3a2e`，现为 `jieen1/sparkinfer` `origin/master` 的 `7a1d69d`/`0844a4f` | ✅ 已解决，见 [`sparkinfer-fork-delta.md`](sparkinfer-fork-delta.md) |
| R6 | **torch 版本合同不一致** | `pyproject.toml` 钉 `torch==2.11.0`；实测环境 `2.13.0a0`；sparkinfer 要求 `>=2.12` | ✅ 钉 `torch==2.13.0` |
| R7 | **Qwen3.6 支持已被摘除** | `ff4d858` / `a9cb932` 把 Qwen3.6 + DirectModelRunner 整体移入 `oracle/qwen36_vllm/`（8370 行，仍依赖 vLLM）；`ServerEngine.__init__` 对 `backend != "laguna"` 直接抛 `ValueError` | ✅ **已重新接入**（2026-08-03 复核）。`IMPLEMENTED_BACKENDS = {"laguna","qwen36"}`（`runtime/model_registry.py:152`），`ServerEngine._load_qwen36_model`（`server/engine.py:526`，522 处分发）。标准模型已实际服务通过 C-LIVE 64/67，不是"能加载" |
| R8 | **文档全面过期** | `AGENTS.md` 指名的 4 个模块都已不存在；README 英文段说"Currently optimized for Qwen3.6-27B"，中文段说"当前生产模型为 Laguna-S-2.1" | ✅ 已解决 |
| R9 | **仓库卫生** | `server/engine.py.bak` / `.orig`、`runtime/backends/laguna.py.bak`、根目录 9 个 `*.log`、`build/`、21 个残留分支 / worktree | 🟡 **大部分已清**（2026-08-03）：根目录 `*.log` 与 `build/` 已无；**worktree 42 → 3**（清理前把 4 个有未提交改动的树各自提交到自己分支再删，分支全部保留，没有丢东西）。剩：3 个 `.bak`/`.orig`（**未跟踪、已在 `.gitignore` 第 37–38 行、无任何代码引用、2026-07-20~23 的旧备份**）——它们是主工作区里别人的未跟踪文件，**留待用户自己决定是否删**，不擅自动 |

**这一批的方法论教训**（值得比修复本身更认真地记住）：

- **三个分支各自全绿、合起来是红的。** api 分支在装了 fastapi 的环境里验证，ci 分支在没装的环境里验证，直到合并才暴露 `test_format_regression.py` 违反了它自己 docstring 声明的 CPU-only 契约。**并行分工必须配一次真实的合并验证**，否则每个分支的"绿"都是局部的。
- **诊断要跑，不能读。** R1 的原始诊断（我方）是错的——错在只看代码不看流水线实际死在哪一步。
- **提交信息里的 `Tested:` 是有约束力的。** `d52a3b1` 声称 "unit tests pass" 却带着红测试进了 main，直接制造了 R2/R3。

### 1.3 这一批新发现的问题（尚未解决）

| # | 问题 | 证据 | 归属 |
|---|---|---|---|
| N1 | ~~**结构化输出是空壳**~~ | `runtime/structured_output.py` 的 `GrammarState.apply_mask()` / `apply_mask_batch()` 在 `server/engine.py` 里**从未被调用**。`json_object` / `json_schema` 请求会被正常接受，但**完全不约束生成**——静默失败，客户端拿到的是普通文本 | ✅ **已解决（走"响亮失败"分支，不是"接上"分支）**，2026-08-03 复核。`server/app.py:562` 的 `_reject_unsupported_response_format` 在进引擎前就对 `json_object`/`json_schema` 返回 400，错误信息直说本运行时不强制结构化输出。**原文最危险的性质——静默——已经没有了。** 为什么选拒绝而不是接上，三条阻塞原因记在 `runtime/structured_output.py` 的模块 docstring 里（prefill 锚点 token 是 `laguna.py` 深处的裸 argmax、CG 重放把 argmax 烤进图里、eager 贪心走捷径绕过 `sample_from_logits`）——**重新接线前先复核那三条是否仍成立** |
| N2 | ~~**`stop` 序列完全未实现**~~ | 两套协议都是 | ✅ **已实现**（2026-08-03 复核）：`server/formats/stop.py` 的 `find_earliest_stop_match` / `trim_ambiguous_stop_tail`，`server/app.py:767` 归一化两套协议的 `stop`/`stop_sequences` 后下发（`max_count=4`） |
| N3 | **`seed` 语义可疑** | 每个 token 重新播种，而不是推进同一个 generator | Track E |
| N4 | ~~**bfdiag 的隔离保证可能已经失效**~~ | `bfdiag/checkpoint/state.py` 有一条 `"bug_found_not_fixed"` 手册条目 + 专门的回归测试，指向 `laguna.py:1647,1653` 的张量轴错误。但真实的 `reset_slot` 已被重写，为前缀缓存保留而**完全不再清零 KV 内存**。连带问题：`bfdiag/checkpoint/restore.py` 明确依赖 `reset_slot` 清掉 checkpoint 范围外的残留来保证恢复隔离性 | ✅ **已解决**（2026-08-03 复核）。三处都对上了：① `restore.py:69` 现在**显式 `.zero_()`** 本槽位 KV，不再依赖 `reset_slot`，且 78–82 行写明"`reset_slot` 已改为保留 KV 内容故不再清零，本函数总是覆盖完整范围并清掉 `_prefix_cache_tokens`/`_prefix_cache_kv_len` 记账"；② `bug_found_not_fixed` 条目已从 `bfdiag/` 移除，`daemon/session.py:65` 记着它**曾经说错了什么**；③ `reset_slot` 现在 `laguna.py:2221`，docstring 自己声明"KV **NOT** zeroed"。**原始担忧"诊断平台自己说谎"不再成立** |
| N5 | **Anthropic 侧拿不到规范形态的 reasoning** | 见 §1.4 | Track E |
| N6 | **全套件下的 flaky** | `test_bfdiag_record.py::test_cli_ls_labels_an_unfinished_record_running`。已缩窄：单文件 8/8 过；bfdiag 子集 + 12 路 CPU 满载 3/3 过；只在**全套件**且机器有 GPU 负载时出现（3/5）。排除了两个显而易见的猜测——标签逻辑是 `finished_at is None → "running"`，**与时间无关**，不是老化阈值；`default_store()`/`bfdiag_dir()` 每次调用都重读环境变量，不是缓存 store。结论：来自 `tests/test_bfdiag_*` 之外某个测试的跨测试副作用 | Track 0 收尾 |
| N7 | ~~**`FakeEngineProvider.load` 与 Protocol 不符**~~ | 没接 `EngineProvider` 声明、`LagunaEngineProvider` 实现了的 `on_stage` 参数。当前休眠（调用点没传），改了就炸 | ✅ **已修**（2026-08-03 复核）：`bfdiag/daemon/provider.py` 三处 `load` 签名一致（196/288/423），288 处的注释明写 "before this fix, this method took no `on_stage` parameter"。同类漂移现由 `tests/test_fake_runner_signatures.py` 按 `ModelBackend` 协议自动覆盖 |

### 1.4 thinking / reasoning 契约（D1 已定案）

**契约**：`content` / `text` 永不包含 reasoning；OpenAI 侧走 `reasoning_content`（delta / message）；
`QSR_REASONING_MODE=expose|strip`，默认 `expose`。判定规则从"对最终文本跑正则"改成
**生成流上的锚定状态机**——只有当 `<think>` 是生成文本的第一个字符时才认定存在 reasoning 段，
`StreamProcessor` 是这条规则的唯一实现，非流式路径复用同一个状态机。

**Anthropic 侧是非标准的**，这是一个有据可查的取舍而非疏忽：`f13fd4a`（2026-07-22）记录了一次
真实生产事故——Claude Desktop 会校验 thinking block 的加密签名，伪造的 32 位十六进制签名被拒后，
客户端**静默丢弃后续所有 content block，包括 tool_use**，用户的工具选择返回 "(no content)"。
那次修复留下了明确指令：`Do NOT re-add thinking block emission without a valid signature source`。
签名是服务端加密产物，我们造不出来。所以 Anthropic 侧发的是非标准的
`reasoning_content_delta` 事件 + 顶层字段，而不是规范的 `thinking` content block。

**可推翻的条件**：拿到合法签名来源。在那之前不要"顺手改回规范形态"——那正是 `f13fd4a` 修掉的 bug。

### 1.5 结构性短板（不是 bug，是设计债）

| # | 短板 | 具体表现 |
|---|---|---|
| S1 | **模型是硬编码的，不是配置** | `ServerEngine.MODEL = "poolside/Laguna-S-2.1-NVFP4"`；`BACKEND = "laguna"` 且拒绝其他值；`server/app.py` 里 `SERVER_MODEL_BACKEND = "laguna"` |
| S2 | **没有 backend 协议** | `LagunaBackend` 有 50+ 公开方法，`ServerEngine` 直接调用，没有任何接口约束。加第二个模型时无从下手 |
| S3 | **ModelSpec 是空壳** | `runtime/model_spec.py` 88 行，只有层名列表和 MTP 开关；不描述层类型序列、RoPE 类型、量化格式、MLP 类型 |
| S4 | **只有一类缓存** | `block_pool.py` 只管 paged KV；GDN/SSM 递归状态的挂钩（`evict_gdn_checkpoint` 等）是为 Qwen3.6 写的**休眠原语**（措辞已于 2026-08-02 更正，此前误称"残迹"——它经过验证、有测试，不是待清理物），当前 Laguna 无 GDN 层，这条路径没有活代码。第 7 步复用它，不重写 |
| S5 | **加载器只认一种量化格式** | 只支持 compressed-tensors（Laguna）；Qwen3.6 NVFP4 是 modelopt 格式 |
| S6 | **router kernel 写死 Laguna** | `runtime/laguna_router.py`：`EXPERTS = 256`、`TOP_K = 10` 是模块级常量 |
| S7 | **容量配置要人肉算** | 启动要同时设对 `QSR_SERVER_CAPACITY` / `NUM_SLOTS` / `BLOCKS_PER_SLOT` / `PRODUCTION`，四者有耦合约束，算错就是 OOM 或白白浪费显存 |
| S8 | **采样与投机互斥** | `temperature > 0` 直接退化成无投机自回归解码；只有贪心走完整 MTP 流水线 |
| S9 | **benchmarks/ 已经失控** | 136 个脚本，绝大多数是一次性诊断残留（bfdiag 的存在就是为了取代它们，但旧脚本没清） |
| S10 | **环境变量前缀仍是 `QSR_`** | 产品叫 BlackweLLM，目录叫 `qwen-sm120-runtime`，变量叫 `QSR_`，三套命名 |

---

## 2. 目标模型清单

### 2.1 本路线图覆盖

| 模型 | 架构 | 优先级 | 备注 |
|---|---|---|---|
| `Laguna-S-2.1-NVFP4` | MoE + SWA/Full 注意力 | P0（保持不回归） | 现有唯一生产模型，是重构的**回归基准** |
| `Qwen3.6-27B`（NVFP4 / 文本版） | Hybrid GDN + Full 注意力，稠密 MLP | P1 | 本地已有 4 个 checkpoint 变体 |
| `Qwen3.6-25B-A3B` | **[待验证]** 推测为 Hybrid + MoE | P1 | 本地无 checkpoint，需先拉 config |
| 上述两者的衍生微调版 | 同上 | P2 | 只要 `config.json` 架构字段一致即应自动可用 |

### 2.2 Qwen3.6-27B 架构事实（读自本地 `nvidia/Qwen3.6-27B-NVFP4` 的 `config.json`）

```
architectures : Qwen3_5ForConditionalGeneration   (model_type: qwen3_5)
num_hidden_layers : 64  =  48 linear_attention  +  16 full_attention  (interval 4)
hidden_size       : 5120        intermediate_size : 17408  (稠密 MLP，非 MoE)
num_attention_heads : 24        num_key_value_heads : 4    (GQA group = 6)
head_dim          : 256         partial_rotary_factor : 0.25
attn_output_gate  : True        output_gate_type : swish
linear_*          : conv_kernel_dim 4 / key_head_dim 128 × 16 heads
                    / value_head_dim 128 × 48 heads / ssm dtype fp32
rope              : mrope interleaved, mrope_section [11,11,10], theta 1e7
mtp_num_hidden_layers : 1       max_position_embeddings : 262144
vocab_size        : 248320      quant_method : modelopt (NVFP4) + fp8 kv
vision_config     : 存在（多模态）— 本路线图只做文本版
```

### 2.3 与 Laguna 的差异矩阵（这就是工作量的来源）

| 维度 | Laguna-S-2.1 | Qwen3.6-27B | 差距 |
|---|---|---|---|
| 注意力层构成 | 48 层全是注意力（36 SWA + 12 Full） | 16 层 Full + **48 层 GDN 线性注意力** | 🔴 **GDN 目前 0% 覆盖** |
| head_dim | 128 | 256 | 🟡 sparkinfer planner 有 `head_dim>=256 & gqa_group<=8` 分支，**[待验证]** 实测 |
| FFN | MoE 256 专家 top-10 + 共享专家 | 稠密 SwiGLU | 🟢 更简单，但需要稠密 NVFP4 GEMM 路径 |
| 注意力输出 | 普通 | **门控输出**（swish gate） | 🟡 模型图改动 |
| RoPE | yarn，partial 0.5，分层不同 theta | mrope interleaved，partial 0.25 | 🟡 需要 RoPE 变体 |
| 滑窗 | sliding_window 512（SWA ring KV） | 无滑窗 | 🟢 可省掉整套 SWA ring 机制 |
| 量化格式 | compressed-tensors | **modelopt** | 🟡 加载器分支 |
| 投机解码 | DFlash（独立 draft 模型） | **MTP**（1 层，在 checkpoint 内） | 🔴 不同机制，且 GDN 状态回滚是难点 |
| 词表 | 100352 | 248320 | 🟢 |

**结论**：GDN（48/64 层）是 Qwen3.6 支持的**单点最大风险**。它不只是一个 kernel，
而是引入了**第二类缓存**——长度无关的递归状态（conv state + ssm state），
它要和固定槽位、前缀缓存、CUDA Graph、投机解码这四套已有机制全部对接。
`docs/archive/2026-07-30-architecture-two-tenant.md` §6.2 记录过当年 Qwen3.6 时代
对这个问题的解法，是可复用的先验。

---

## 3. 不做清单

明确写下来，是为了让"要不要顺手支持一下"这个问题不再重复出现。

| 不做 | 理由 |
|---|---|
| 多卡 TP / PP / EP | 硬件合同外；引入的抽象会污染整个执行栈 |
| 多机 | 同上 |
| SM120 以外的架构 | 项目的全部价值来自架构专用化 |
| 视觉 / 多模态输入 | Qwen3.6 有 vision tower，本路线图只做文本版；多模态是另一条产品线 |
| 训练 / 微调 / LoRA 热加载 | 纯推理运行时 |
| AWQ / GPTQ / INT4 等其他量化格式 | 除非某个目标模型只有这种格式；不做通用量化框架 |
| 通用 HF 架构自动支持 | 每个架构显式接入，宁可少而正确 |
| 把 `oracle/qwen36_vllm/` 复活 | 它依赖 vLLM，与"零 vLLM"合同冲突。Qwen3.6 走新抽象层重新接入，旧代码只作参考读物 |

---

## 4. 轨道与优先级

九条轨道（0、A–H），按优先级排列。轨道内部是有序的，轨道之间大量并行。

### Track 0 · 止血（P0，M1 内必须清零）

把 §1.2 的红灯全部解决。这是所有后续工作的前置——在一个 CI 红、
测试红、依赖不可复现的仓库上做架构重构，等于没有护栏。

- ✅ **T0-1 CI 恢复绿灯**。**两条路都走了**：保留 CPU-only job 作为契约守门人
  （5 个违规模块改 `pytest.importorskip`），另加一个装 CPU torch 的 job 扩大覆盖面。
  第二个 job 有个非显然的前提：光装 torch 不够——多个模块假定"有 torch 就有
  numpy/safetensors/huggingface_hub/transformers/triton"，这在完整 GPU 环境里成立，
  对裸 torch wheel 不成立，最小可用集是 `.[dev,serving]` + CPU torch wheel + triton。
  另外修掉 `benchmarks/quick_check.py` 的 4 个未使用 import——`ruff check .` 是 CI
  的第一步，它红着，流水线从来没走到 pytest。
- ✅ **T0-2 修 bfdiag_ring 失败**：确认是测试替身漂移而非生产 bug
  （`DFlashEngine.__init__` 合法地从 backend 取 `device`/`block_size`/`_draft_blocks_per_slot`，
  测试用 `object.__new__` 绕过了构造函数）。顺带审了全部假对象，`test_bf_attention.py` /
  `test_cudagraph_buffers.py` 的同类写法对得上真实类，无需改动。
- ✅ **T0-3 / T0-4 thinking 契约定稿并重做**：见 §1.4。
- ✅ **T0-5 sparkinfer 补丁**。**结论与原计划不同**：补丁不是"未上游"，是**从未提交、已丢失**。
  已从悬空提交恢复、rebase 到新 upstream、合入 `jieen1/sparkinfer` 的 `origin/master`
  （fork 归我们所有，`upstream` 才是另一个团队的——原计划把这两者搞混了）。
  启动期校验保留：`check_sparkinfer_analytic_decode_gate` 用真实生产形状去探活的 gate，
  探测到关闭报 **warning 而非 fatal**（性能显著下降但功能正常，不该拦住启动）。
- ✅ **T0-6 依赖版本合同统一**：`torch==2.13.0`；补上三个漏声明的直接依赖
  （`huggingface_hub` / `nvidia-cutlass-dsl` / `triton`）；`transformers` 从 `serving` 移到 `cuda`；
  sparkinfer 钉 `origin/master @ 0844a4f`；`runtime/preflight.py` 九项启动期校验，
  接在 `server/app.py:main()` 的 `uvicorn.run` 之前。
- 🔴 **T0-7 仓库卫生**：未做。删 `.bak` / `.orig` / 根目录日志；清理已合并分支与
  `.claude/worktrees/` 残留；`benchmarks/` 分流（保留的进 `benchmarks/`，
  一次性诊断残留删除或转为 `bf exec` 脚本）。
- 🔴 **T0-8 收尾**（本批新增）：N6 的全套件 flaky 定位；N7 的 Protocol 不符。

**体量**：约 0.5 个月，其中 T0-1～T0-6 已于 2026-08-01 完成。

### Track A · 模型抽象层（P0，M1→M2）

这是"支持更多模型"的全部前置条件，也是"易用性"的根。设计目标不是通用性，
是**让接入第 N 个模型的成本可预测**。

- **A1 `ModelSpec` 升级为架构描述**：从 checkpoint 的 `config.json` 解析出
  层类型序列、每层的注意力/线性注意力/MLP 类型、RoPE 配置、量化格式、
  MTP 配置、缓存需求（每层需要 paged KV 还是递归状态）。
  校验前置：不支持的架构在**加载前**报错，不是跑到一半 NaN。
- **A2 Backend 协议**：把 `LagunaBackend` 的公开面收敛成一个显式协议
  （prefill / chunked prefill / decode / decode_batch / reset_slot /
  prefix 匹配与回放 / spec-decode 生命周期 / CUDA Graph 捕获）。
  先用 Laguna 做唯一实现，**协议由现有实现倒推**，不预设未来。
- **A3 缓存资源抽象**：`block_pool` 从"KV 分页器"升级为"槽位资源管理器"，
  统一管理两类资源——分页 KV（长度相关）与递归状态（长度无关、每槽固定）。
  前缀缓存的驱逐必须对两类资源联动（这正是 §1.5-S4 里那些休眠原语当年要解决的问题）。
- **A4 加载器抽象**：compressed-tensors / modelopt 两套 NVFP4 布局的
  tensor 命名与 scale 语义分离成两个 adapter，公共部分（分片流式读取、
  参数全覆盖断言、KV scale post-load）保持不变。
- **A5 模型注册表 + 自动识别**：给定 checkpoint 路径 → 读 config → 匹配架构
  → 选 backend + loader + spec 策略。`ServerEngine` 不再有 `MODEL` 常量。
- **A6 Laguna 迁到新抽象，零回归**：门禁 = 贪心输出 bit-exact + 性能不低于
  基线 3%（bfdiag run record 对比，`bf diff` 判可比性）。

**体量**：约 1.5 个月。**风险**：这是一次动到核心执行路径的重构，
Laguna 的性能与位精确是硬约束，必须逐步切换而非一次性替换。

**2026-08-01 补充**（`investigation-queue.md` D-4，vLLM v0.26.0 "每 KV-cache group 选不同 attention
backend；滑窗作为显式 backend capability"）：这条**验证了 A1/A2 的设计方向，不改变它**——A1 本就按
层类型序列描述架构（full / sliding / linear-attention 逐层区分），Qwen3.6 的 16 full + 48 GDN 混合
正是这个设计要接住的形状。**唯一的具体补充**：`BackendCapabilities`（§3.5.3）目前只有五个布尔标志，
没有把"滑窗"显式建模成一等能力——当前 Laguna 的 SWA 是通过层类型隐式处理的。A1/A2 落地时应把滑窗
参数（窗口大小、per-layer 是否滑窗）提升成 `ModelSpec`/能力查询里可查询的字段，而不是留在模型图内部。
不新开条目，作为 A1/A2 实现时的一条设计备忘。

### Track B · Qwen3.6-27B 重建（P1，M2→M4，有参考实现）

**这不是"接入一个陌生模型"，是"重建一个曾经在 vLLM 上跑通、有实测数字的实现"。**
`oracle/qwen36_vllm/` 有 8047 行、11 个模块的参考代码；`docs/archive/2026-07-20-PROGRESS.md`
等处有当年的真实吞吐/接受率/质量/显存数字。完整的逐模块判定（可直接搬/需改写/已被取代/
应废弃，逐项标新位置）、验收基线的完整来源表、以及在今天 Track A 抽象上的重建设计，见
[`qwen36-rebuild-spec.md`](qwen36-rebuild-spec.md)——本节只摘要结论，细节与行号引用一律
以那份文档为准。

**2026-08-02 关键纠偏**（读完 `oracle/qwen36_vllm/backends/qwen36.py` 全部 2159 行后确认）：
**模型数学本身（GDN 层 forward、mrope-interleaved RoPE、`attn_output_gate`、稠密 SwiGLU
MLP、modelopt NVFP4 反量化）完全不在这份参考代码里**——它当年活在 vLLM pip 包自己的
`Qwen3_5ForConditionalGeneration`/`Qwen3_5MTP` 类里，从未被 vendor 进本仓库，`get_model()`
只是现场把它借来用。**oracle 里真正能复用的是编排层与状态管理层**：GDN checkpoint
快照/恢复（`gdn_state.py`，466 行，判定为最高价值文件）、GDN 状态×投机解码的行寻址方案
（`_ssm_spec_row`，**已经原样存在于 `runtime/block_pool.py:45-79`，未被删除，休眠等接线**）、
accept/reject 判定算法（**已经完全移植完成**，就是今天的 `runtime/mtp_accept.py`）、块哈希
前缀缓存骨架。这意味着 B1（模型图 + 正确性）比原计划更接近纯新写，B0/B2/B3（状态管理 /
CUDA Graph 编排 / 投机回滚机制）比原计划有更多现成参照——工作量没有减少，是**性质变了**。

分四段，每段有独立的验收，不允许"边写边猜"。

#### B0 · 事实基线（M2，约 2 周）

- ✅ **主线 checkpoint 已拍板（见 §7 D6）：官方 `nvidia/Qwen3.6-27B-NVFP4`**。本地其余变体：
  `unsloth/...`（compressed-tensors，不是 modelopt——量化格式必须逐 checkpoint 读，不能按架构推断）、
  `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`（留作交叉验证 baseline）、`morosystems/ThinkingCap-...`。
  衍生任务：写一个按 tensor 名前缀跳过 `vision.*` 的加载过滤器（333 个张量，一次写好可复用）；
  A1 的 `validate_text_only` 语义要跟着调整为"允许 `vision_config` 存在但断言零 vision 张量被加载"
  （见 RK8；`runtime/architecture.py:292-319` 已经是这个语义，非空文档承诺）。
- Tensor 清单与 modelopt scale 语义逐项确认（不猜命名）——`git show a9cb932^:runtime/model/nvfp4_linear.py`
  可取回一份当年针对 compressed-tensors 格式写的、几乎完工的 NVFP4 Linear 原型（权重侧张量操作
  `swizzle_blockscale` 等可直接搬），但参数命名要按 modelopt 真实 checkpoint 重新逐项确认，
  见 `qwen36-rebuild-spec.md` §1.9/§3.4。
- **[待验证]** sparkinfer paged attention 在 `head_dim=256 / gqa_group=6 /
  page_size ∈ {64,128} / fp8 KV` 下的正确性与吞吐——planner 里已有对应分支，
  但没有本机实测记录。**验证时应同时检查 warmup/autotune 是否覆盖了这个真实形状**
  （见 RK9、`investigation-queue.md` C-1，同一类"首次真实形状才暴露代价"的问题）。
- GDN 方案选型三选一，**倾向已加强为"先 1，晚点再看 3"**：
  1. 依赖 `flash-linear-attention`（本地已有 v0.5.2；本轮新确认 `fla.ops.gated_delta_rule` 的
     `chunk`/`fused_recurrent` 两条路径本地均可 `import` 成功、且无需 `causal_conv1d`——但**从未在
     SM120 上实跑**，`investigation-queue.md` §F 记录的 Blackwell 相关 bug 全部是 B200/SM100，
     无 SM120 记录，"未验证"不等于"已知能跑"）；
  2. 从 `oracle/qwen36_vllm/` 的 vLLM 路径移植（**注：GDN 层 forward 本身不在这份代码里**，
     真正能移植的是 `gdn_state.py` 的状态管理，不是算子本身，见 `qwen36-rebuild-spec.md` §1.0）；
  3. 自研 Triton kernel。
  **[待验证，本轮新增数据支持]**：`notes/2026-07-22-a1a-gdn-profiling.md` 实测 GDN 48 层合计
  decode GPU 时间占比恒定在 **3.9%–5.1%**（4K 与 128K 上下文均如此，NVFP4 GEMM 才是主导，
  71.1%→53.7%）——GDN kernel 本身**不是**性能瓶颈，支持"先 1 拿正确性，profiling 说话后再决定
  要不要 3"这条既有建议，不改变它。
- ✅ **Qwen3.6 的 MTP 层是否带 GDN**（`investigation-queue.md` B-6）——**已确认：不带**。
  6 个本地 checkpoint 变体的 `mtp.*` 张量清一色 `self_attn.*`+`mlp.*`，零 `linear_attn.*`。
  **但这不消除 B3 的 GDN 回滚项**——vLLM 那条"draft models have no mamba layers"的注释指的是
  draft 模型自己的递归状态，草稿侧因此确实可以少做（不需要为 MTP 头单独管理 conv/ssm 状态或做
  eagle-shift 类操作），但**主模型的 48 个 GDN 层在 verify 时照样跑完整 64 层前向**，被拒 token
  照样污染了不可逆的递归状态更新，回滚问题原样保留在 B3，只是范围从"草稿+主模型两侧"收窄到
  "只有主模型侧"。详见 `notes/2026-08-01-b6-mtp-gdn-verification.md`；B3 不再需要按"带/不带 GDN"
  写两个分支，只有一个分支（见下）。
- **[待验证，另一 agent 在查，本文档不预判]** NVFP4 KV vs FP8 KV 在我们卡上的对比
  （`investigation-queue.md` C-2）：上游第三方在 RTX PRO 5000 上的数字（NVFP4 KV prefill 慢
  1.7–1.8×）不是我们的卡也不是我们的形状，只作参考，不作决定。
- 容量测算：64 层 / 256K 上下文 / 96 GB 下的 KV + 递归状态显存账，
  给出 context × 并发的可行域。**旧参照数字**（vLLM 执行路径下测的，不能直接当新框架的数字用，
  仅作方向参考）：128K/c=4/warm 约 90.7–92.9 GiB，64K/c=4/warm 约 63–65 GiB，256K/c=4/cold
  可行（82.8% 峰值，无 OOM），200K/c=4 两侧均不可行（>95GB）——完整来源见
  `qwen36-rebuild-spec.md` §2.4/§2.5。

**验收**：一份事实基线文档，把上述每项写成"实测值 + 复现命令"。

#### B1 · 正确性优先（M2→M3，约 1 个月）

刻意放弃所有性能：eager、batch=1、无 CUDA Graph、无投机、无前缀缓存。
**性质提醒**：以下五项里的 GDN 层/RoPE/MLP/门控/modelopt 加载**全部是新写代码**
（`oracle/qwen36_vllm/` 不含模型数学，只含编排层，见本节开头的纠偏），不要按"移植"的
工作量估计排期。

- GDN 层（conv1d state + gated delta rule + 输出门）
- Full attention 层（走 sparkinfer paged）
- 稠密 SwiGLU MLP（NVFP4）
- RoPE：partial_rotary_factor 0.25 + mrope-interleaved
- modelopt 权重加载
- 注意力输出门控

**验收门禁**：与 HF transformers 参考实现在贪心解码下**逐 token 对齐**
（至少 3 个工作负载 × 512 token）；逐层 logits 余弦相似度记录进 bfdiag。
**质量验收基线**（Qwen3.6-vLLM 时代实测，2026-07-21/22，完整来源见
`qwen36-rebuild-spec.md` §2.3）：MMLU-Pro **84.54%**（vs 官方 86.2，−1.7pp 噪声内）；
HumanEval **44.5%** / HumanEval+ **43.3%**（vs 同权重 stock vLLM 43.3%/42.7%，无退化）。
**⚠️ 这三个数字目前在 `docs/model-support.md:49` 里被错标成 Laguna 的当前质量数字**——
引用时以本节和 `README.md` 的历史数字表为准，不要拿 `model-support.md` 当独立确认，
详见 [`../notes/2026-08-02-laguna-docs-inherited-qwen36-numbers.md`](../notes/2026-08-02-laguna-docs-inherited-qwen36-numbers.md)。

#### B2 · 服务化（M3，约 1 个月）

- 接入固定槽位调度 + 连续批处理
- 递归状态纳入槽位生命周期（reset / 复用 / 看门狗回收）——协调者设计（对应 A3）见
  `qwen36-rebuild-spec.md` §3.3，六条修改（不做统一分配器、两数字前缀匹配、投机保守释放、
  同轮不可跨请求借用、逐资源驱逐预算、块大小对齐）来自
  [`hybrid-cache-prior-art.md`](../notes/2026-08-01-hybrid-cache-prior-art.md)
- CUDA Graph 捕获（decode 路径；GDN 的状态更新是否 graph-safe 是关键 **[待验证]**——
  本轮确认这条**没有可抄的参照**：Laguna 的 decode 图从不触碰递归状态，其 warmup 复用天然
  安全，oracle 当年的解法是保留 `2×batch_size` 专用 warmup 槽（GDN 状态非幂等，不能用真实
  请求槽热身），这个设计要在自建 CUDA Graph 骨架上重新验证，不是照抄就对，见
  `qwen36-rebuild-spec.md` §3.5、§6.1 判定的**第一难点**）
- 前缀缓存（含递归状态 checkpoint 与 KV 块的联动驱逐——A3 的第一个真实用户，
  `BlockPool._on_evict_block` 挂钩已就位但值为 `None`，是否直接接线还是被协调者新设计取代
  留给 A3 落地时拍板）
- 并发 ≥ 2
- **前置条件（本批新增）**：`bfdiag/daemon/provider.py` 目前直接持有具体的 `LagunaBackend`/
  `DFlashEngine` 类型，B2 的验收依赖 bfdiag（run record / `bf diff`），需要 Track A 把
  `bfdiag` 的 provider 改成按协议持有，**应在 B2 开始前完成**，不是可以顺手拖到 B2 期间做的小事
  （见 `architecture.md` §3.5.4，`qwen36-rebuild-spec.md` §3.6）

**验收**：HTTP 端到端，OpenAI + Anthropic 双协议回归全绿；
与 B1 的 eager 路径贪心 bit-exact。

#### B3 · 性能与投机（M4，约 1 个月）

**只有一个分支**（B0 的 B-6 结论已定：MTP 头本身无 GDN，但主模型侧回滚问题原样保留，
不再按"带/不带 GDN"写两个分支）：

- MTP draft / verify（Qwen3.6 自带 1 层 MTP，草稿侧因头部无 GDN 而少一块状态管理），
  含**主模型 48 层 GDN 递归状态的推测回滚**——寻址方案（`_ssm_spec_row`）与 accept/reject
  判定算法已经现成可用（见本节开头），真正要重新解决的只是把这两者接到自建 CUDA Graph +
  自建模型图上，与 D-3（ReplaySSM Ring Spec-Verify）合并排期
- GDN kernel 调优或自研（依据 B0/B2 的 profiling；本轮数据显示 GDN 恒占 <5.1% decode 时间，
  优先级应低于 NVFP4 GEMM 与 attention 调优，见 B0）
- **KV dtype 待定**：[待验证]，取决于 B0 里 NVFP4 KV vs FP8 KV 的本机对比结果，不写死 "FP8 KV"
- 长上下文（128K / 256K）容量与吞吐

**验收**：接受率与吞吐进 bfdiag 基线；与上游框架同 prompt 同参数做 A/B。
**性能验收基线**（Qwen3.6-vLLM 时代实测终值，完整轨迹与噪声说明见
`qwen36-rebuild-spec.md` §2.1/§2.2）：

| 指标 | 历史基线（终值） | 配置 / 日期 |
|---|---|---|
| 吞吐（128K, c=4, warm） | **222.44 tok/s** | MTP K=3，2026-07-21，`PROGRESS.md:4239-4244` |
| 吞吐（64K, c=4, warm） | **236.69 tok/s**（更可信）/ 267 tok/s（较低置信度，仅架构文档回声） | 同上，2026-07-21 |
| MTP 接受率（128K, c=4, warm） | **50.3%**（约每轮 2 token） | 与 222.44 tok/s 同批测量，2026-07-21 |
| 256K, c=4（cold, chunked） | 1.557 tok/s，双方可行，82.8% 峰值显存 | 2026-07-19 |

**新实现打不平这些数字就是退步，但对比前必须先确认口径一致**——这些数字全部是在 vLLM
执行路径下测的（含 vLLM scheduler/ForwardContext 开销），新框架走 Track A 抽象后开销分布
不同，理论上有改善空间但不是承诺；且接受率在当年不同测量批次间本身有 3+pp 波动（含一次
已定位的计数 bug），不要用单次数字判定退步/进步，按仓库纪律先 `bf diff` 再比较。

### Track C · 稳定性（P1，贯穿 M1→M6）

不是一个阶段，是一条持续的轨道。核心思路：**把每一种失败都变成一个
有名字、有指标、有降级路径的已知状态**。

> **2026-08-02 补充**：本节原来是一份没有排期的清单（"C0-C7"只是编号，不是阶段）。
> 详细的分期、GPU 窗口调度（如何不与 Track B 抢卡）、以及"如何证明新门禁真的会红"
> 的方法论，已展开写进 [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md)（§3/§4/§5）；
> 本节保留目标与理由，阶段归属见下表。

| 阶段 | 里程碑 | 需要 GPU |
|---|---|---|
| C0R（bfdiag 自证据审计残余） | M1 | ❌ |
| C1（故障面清单，按"跟哪个 GPU 窗口走"拆成 backend/protocol/slot 三段） | M1→M3 | 部分 |
| C2（分级降级三级指标化） | M2→M4 | 部分 |
| C3（看门狗覆盖 + 故障注入） | M2→M3 | 主要靠 mock |
| C4（bit-exact 门禁落地） | M2 | ✅ |
| C5（24h soak，两个独占检查点） | M3 末、M6 末 | ✅ 独占 |
| C6（崩溃留 bfdiag run record） | M2 | 部分 |
| C7（冷启动/首次真实形状路径审计，RK9） | M2 起，见缝插针 | ✅ |
| **C8**（新增：门禁可信度周期审计） | M2/M4/M6 | ❌ |
| **C9**（新增：质量回归 MMLU-Pro/HumanEval+） | M2 起，随 Track B 加 Qwen3.6 覆盖 | ✅ |

- **C0 诊断平台自身的可信度**（本批新增，**优先于本轨道其它条目**）：N4 揭示的问题是
  `bfdiag/checkpoint` 依赖一个已经不成立的前提（`reset_slot` 会清零 KV），而守护它的
  回归测试从不调用真实函数，所以一直是绿的。**一个会说谎的诊断平台比没有诊断平台更危险**——
  它让错误结论带着"有测试保证"的权重传播。要做的：审计 `bfdiag/` 里所有"对真实
  backend 行为的断言"，区分哪些真的在验证真实代码、哪些只是在合成数据上复现抽象模式；
  后者必须显式标注成"模式演示"而不是"回归门禁"。**这不该是一次性事故处理**——见下方 C8。
- **C1 故障面清单**：显存不足、槽位卡死、CUDA Graph 捕获失败、kernel JIT 失败、
  长请求超时、客户端断连、tokenizer 边界、非法采样参数、并发抢占。
  每一项要有：检测点、指标、日志、用户可见错误、恢复动作。
  **2026-08-02 拆分**：backend 层（显存/CG 捕获/kernel JIT/并发抢占，蹭 Track A 第 5–8
  步窗口）、协议层（断连/tokenizer 边界/非法参数/超时，与 Track E 的 E3 共享热身服务器）、
  槽位层（卡死，随 A3 第 7 步落地同窗口验证）——三段各自独立可交付，理由与调度见
  [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) §3.1/§5。
- **C2 分级降级**：CUDA Graph → eager；投机 → 非投机；前缀缓存命中 → 冷 prefill。
  每级降级都要出指标（现在部分已有，需成体系）。指标形状复用 D-3 能力查询
  （`BackendCapabilities`）；三个触发点各自蹭对应子系统落地时的 GPU 窗口，不单独申请。
- **C3 看门狗覆盖**：已有 stale slot 回收，需要覆盖测试 + 故障注入。**大部分故障注入可以
  在 Python 层 mock 掉 CUDA 异常来做，不需要真实 OOM**——只有少量场景需要真机确认，
  见缝插针即可，不是这条的主要成本来源。
- **C4 确定性与可复现**：per-request seed 已有；补 bit-exactness 回归门禁，
  纳入 CI。**D3 已拍板 (b)**：本地 pre-push 门禁 + 人工签核，落地机制（`make gate-local` +
  PR 签核勾选项）见 [`implementation-plan.md`](implementation-plan.md) §7.3。
- **C5 长稳测试**：24h soak，监控显存碎片、host 内存、槽位分布、指标漂移。**这是本清单
  里唯一真正需要独占一整天 GPU、不能蹭窗口的条目**——排在 M3 末（Track B 的 B1/B2 验收
  完成、B3 性能冲刺开始前的天然间隙）与 M6 末（发布前）两个检查点，需要提前协调，不能
  假设"顺路"。
- **C6 崩溃可诊断**：进程级异常要留下 bfdiag run record，不是只有一行 traceback。
- **C7 冷启动 / 首次真实形状路径审计**（本批新增，见 RK9）：`235f51e` 修的是"每个未见过的
  page-table 宽度都触发 30–100s JIT 重编译"，但修复自己的提交记录留了一条明确未闭合的口子——
  DFlash 的 eager verify 回退路径不在启动期预热覆盖范围内，而且 CUDA Graph 捕获**成功**的
  可观测性目前是 0（只有失败会显式可见）。这不是孤立 bug，是"首次遇到真实形状/真实路径的代价被
  系统性低估"这一模式的又一个实例——`investigation-queue.md` C-1（sparkinfer warmup/autotune
  是否用真实形状）与 B0-3 的验证范围都属于同一类别。详细任务拆解见
  [`implementation-plan.md`](implementation-plan.md) §7.3/C7。**2026-08-02 交叉引用**：
  Track E 的 E3（客户端 SDK 矩阵）第一次对新代码发请求时天然处于"冷启动窗口"，其中一些
  真实 SDK 的默认超时可能比 JIT 停顿更短——E3 的完成判据应包含"首请求"场景，作为 C7 的
  又一个验证入口，见 [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) §2.4。
- **C8 门禁可信度周期审计**（本批新增，直接回应用户"值得反复审查"的要求）：N4/C0 揪出的
  "从不调用真实函数、一直是绿的假门禁"不该被当成一次性事故处理——设计成每两个里程碑
  （M2/M4/M6）抽查一轮现有测试/门禁，对每条回答"它真的红过吗""如果没红过，能不能构造
  一个会让它红的输入"，答不出来的记入门禁债务清单。方法与首轮范围（优先审计 `bfdiag/`）
  见 [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) §3.2。
- **C9 质量回归**（本批新增）：MMLU-Pro（分层子集）+ evalplus HumanEval+/MBPP+ 对 **Laguna**
  跑一次——现有 harness（`benchmarks/official/mmlu_pro_eval.py` + `quality_regression.py`，
  `92f8b34`，2026-07-22）从建成起就没有指向过当前生产模型，测的是已退役的 Qwen3.6/vLLM。
  **误引已于 2026-08-02 证实并更正**（`c53bd7c`）：§0 原先引用的"Laguna MMLU-Pro 84.5%"
  确系 Qwen3.6 那次跑分（`evalplus_results/official/mmlu_pro_think_c4.json`：
  `model=qwen3.6, acc=84.54, n=414`）。所以 **Laguna 至今没有任何质量基线数字**，
  C9 不是"核实一个可疑数字"，而是**首次建立**它。M2 起首次基线，
  M3/M4 起随 Track B 加 Qwen3.6 覆盖，M6 发布前全量跑一次。方法与调度见
  [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) §3.3。

### Track D · 易用性（P1，M2→M5）

现状是"必须读源码才能正确启动"。目标是"读一页文档就能上线"。

- **D1 单命令启动**：`blackwellm serve <model-path-or-id>`，自动推导槽位与块数。
- **D2 显存规划器**：给定「模型 + 目标上下文 + 目标并发」算出配置并校验；
  或给定显存反推可行域。这直接消灭 §1.5-S7 那个四变量耦合陷阱。
- **D3 启动前置检查**：SM120 检测、显存、CUDA / driver、sparkinfer 版本、
  checkpoint 完整性、架构是否受支持——**全部在加载权重之前**，
  失败给出人能读懂的、带修复建议的错误。
- **D4 配置文件**：YAML 配置取代十几个环境变量（环境变量保留为覆盖手段）。
- **D5 命名统一**：`QSR_` → `BWLLM_`，带一个版本的兼容期与弃用警告；
  包目录 `qwen-sm120-runtime` → `blackwellm` 的重命名时机需要拍板（见 D4）。
- **D6 文档三件套**：安装部署、配置调参、故障排查。

### Track E · 兼容性（P2，M3→M6）

> **2026-08-02 补充**：本节的现状比 2026-08-01 的记录更精确——`docs/api-layer-design.md`
> §5/§7（`fix/t0b-api`，2026-08-01）已经把 N1/N2/N3/`n>1`/usage token 五条逐项核实过，
> 下面按核实后的真实状态重写，并把仍开着的条目拆成独立可交付的小阶段。完整分期、
> "每步怎么证明测到了协议层而不是我们自己的假设"、GPU 窗口调度，展开写进
> [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) §2/§4/§5。

**已关闭，不再重复排查**：N2 `stop` 序列已接通（含跨 token 边界匹配，`server/formats/stop.py`）；
N3 `seed` 已修为同请求内持续前进同一个 generator（`PersistentSeed`）；`n>1` 已被显式 400 拒绝
（`server/app.py:447`），不是待核查项而是已验证的正确行为；N1 的**危险性**已消除（显式 400
拒绝，不再静默失败），但功能缺口仍在——见下方 E-N1。错误码语义三处已修（FastAPI 曾把错误体
双重包进 `{"detail": ...}`）。

- **E-N1 结构化输出真正生效**：`GrammarState.apply_mask()`/`apply_mask_batch()` 逻辑本身没问题，
  真正的阻塞是**解码循环里没有可用的掩码注入点**——admission 阶段的第一个 token 是裸
  `argmax`、CUDA Graph 贪心重放把贪心烤进了已捕获的 graph、eager 贪心分支绕过
  `sample_from_logits`。本运行时默认 `temperature=0.0`，"要保证的 JSON"这种最常见请求
  恰好总是走这三条不可达路径。**已明确排除的选项**：只接通 `temperature>0` 时唯一可达的
  窄缝——文档已经论证过这比完全不接更危险（默认场景看起来接上了但仍不受约束）。分两步：
  先拍板中间态选择（等全量修复 vs 显式限定只支持 `temperature>0`），再实施，且实施要等
  Track A 对 `laguna.py`/`laguna_cuda_graph.py` 的改动稳定下来才能动（文件归属边界）。
  详细分期见 [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) §2.3。
- **E2 采样 + 投机共存**（消灭 §1.5-S8，**当前最明显的功能缺口**）：`temperature>0` 时
  DFlash 直接退化为无投机自回归解码（`_greedy_accept_reject`）。需要拒绝采样 / typical
  acceptance 让投机验证步骤本身允许非贪心采样。分两段：算法正确性可在 CPU 上用合成分布
  验证，不需要真实模型；GPU 集成蹭 Track A 剩余窗口或 Track F 的 F1 窗口。这条**天生是
  红的**（今天的代码在 `temperature>0` 时压根不会尝试投机），不需要历史回放就能证明门禁
  有效。详见 [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) §2.2。
- **E3 客户端 SDK 一致性矩阵**（本批扩充为结构性资产，非一句话目标）：现有的
  `test_api_compat.py`/`c_live_smoke.py` 全部手搓 `http.client`，**没有一个用真实厂商
  SDK 解析响应**——一个符合我们自己断言的响应体不等于一个能被 openai-python/
  anthropic-sdk-python 严格解析器接受的响应体。分四步，便宜且已装好的先做：
  openai-python + anthropic-sdk-python（M2）→ LiteLLM（协议归一化层，M3）→ Claude Code
  本身跑一次真实编码会话（M4，吃自己的狗粮）→ Cline/Roo + OpenWebUI（M5，下游集成，
  风险相对低）。每一步的完成判据、"如何证明会红"的方法（历史 bug 父提交回放，同 C-LIVE
  B-4 用过的标准）、以及与 RK9 冷启动路径的交叉检查（真实 SDK 的默认超时可能比 JIT 停顿
  更短），见 [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) §2.4。
- **E4 reasoning 内容的正确暴露**：OpenAI 的 `reasoning_content` 已接（`c86858a`）。
  Anthropic 侧维持非标准 `reasoning_content_delta` 事件而不是规范 `thinking` block——
  这不是待办，是一条需要长期守住的契约（理由见 §1.4，`f13fd4a` 生产事故）。**建议补一条
  零成本、常驻生效的回归测试**：断言 Anthropic 流式路径任何代码路径都不产出未签名的
  规范形态 `thinking` block——写一次，不需要里程碑节奏。
- **E5 chunked input-logprob 默认开启**（本批新增，`investigation-queue.md` D-8，来自
  SGLang v0.5.16）：削峰值显存，我们已有 logprobs 路径、双协议都暴露 `top_logprobs`。
  **小而自足，不依赖 Track A，也不用等 M3**——建议提前排进 M2，跟 Track F 的 F1 窗口扫测
  蹭同一个 GPU 验证窗口一起做（见 Track F）。
- **usage token 两个小缺口**（本批核实后从"待核查"降级为"已知的两个小缺口"）：
  (1) 缺 `usage.completion_tokens_details.reasoning_tokens` 细分字段——独立、小，可随时
  排期；(2) `<usage>` 标签剥离在流式/非流式路径下语义不统一——需要先决定统一到哪种语义
  （产品判断，留待拍板）。均不构成当前正确性问题，详见
  [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) §2.5。

### Track F · 性能（P2，机会主义，M3→M6 —— 但两条例外见下）

**降级为机会主义轨道**：只在有明确 roofline 依据、且不损害 Track C/D 的
前提下做。已知的候选方向（来自 2026-07-31 的研究记录，以及本批 2026-08-01 消化的
`investigation-queue.md` §D 上游调研）：

**2026-08-01 本批新增两条例外，从"机会主义/P2"提升到 P1、排进 M2**——不是因为它们比
Track A 更急，是因为它们**不依赖 Track A、成本低，且直指本项目当前两个最硬的约束**
（吞吐上限、显存上限），有本机实测数据支撑，不是纯粹的"顺手试试"：

- **接受率 96.3–100% 但投机窗口固定**（D-2）：Laguna 的接受率实测 96.3–100%（§1.1，
  2026-07-31/08-01 复现），而 DFlash 的 `NUM_SPECULATIVE_TOKENS` 固定为 15——这个组合
  说明**限制吞吐的很可能是窗口本身，不是接受率上限**。第一步是便宜的：不实现自适应
  控制器（DSpark 风格），先把窗口从 15 静态调大，`bf diff` 测吞吐与接受率；只有静态调大
  见顶了才上置信度驱动的自适应窗口——vllm #49369 报告 DSpark 在某些负载上比不开投机还慢，
  不是白捡，必须按工作负载分别 A/B。
- **显存是硬约束，且比想象中紧**（D-3）：Laguna 权重 66.8 GB ≈ 67 GB（59.5 GB MoE +
  7.3 GB non-MoE，`notes/2026-07-29-gpu-memory-audit.md`），96 GB 卡上给 KV + 投机
  scratch 的预算很紧。协调者在本轮任务中的实时汇报：生产服务实测显存 94.2/97.9 GB
  （**98.8% 占用**）；2026-07-29 的静态审计（1 slot/131K 配置）测得的是 76.0/95.6 GB
  （79.5% 占用），两次配置不同、不能直接对比，但方向一致——**投机 scratch 在跟 KV 抢一块
  越来越紧的预算**。ReplaySSM Ring Spec-Verify 报告的 11.5 GB → 1.8 GB 是别人的卡、
  别人的形状，不能当我们的数字用，第一步是补一次带日期来源的本机审计，再判断这个技巧
  有多少能在我们自己的调度/scratch 复用层面拿到（我们做），有多少要动 sparkinfer 的
  kernel 内部（转 SparkInfer，写清楚交接，不直接改源码）。这条的结论应该喂给 Track A 的
  A3 协调者设计——投机 scratch 迟早要变成 A3 管理的资源类型之一。**2026-08-02 更新**（
  [`../notes/2026-08-02-gpu-memory-audit.md`](../notes/2026-08-02-gpu-memory-audit.md)）：权重
  数字重新核实（主模型 66.96 GiB + DFlash draft 2.08 GiB，逐张量精确求和，与上面的 66.8 GB 一致）
  并**排除了一个混杂因素**——Qwen3.6 有反量化缓存机制（一次前向后常驻涨 ~50 GiB），Laguna **实测确
  认没有**（`PlainLinear` 从磁盘就是 BF16，MoE 专家直接在 NVFP4 上计算，代码和两次真机测量双重印
  证）。生产规模（`blocks_per_slot=4096/capacity=3`）下的 94.2/97.9 GB 仍未重新复现，见该笔记"未能
  验证的事项"。
- **调度纪律**：这两条不需要 Track A，可以现在做；但都需要真机 GPU 时间，**应优先蹭
  P0-E 第 5 步或 C-LIVE 的 GPU 窗口，不单独申请专用时段**——本机只有一块 GPU，任何需要
  GPU 的验收项天然串行（RK5），这也是 D3 选 (b) 而不是 (a) 的同一条理由。
- 完整任务拆解（F1/F2 的分步骤清单）见 [`implementation-plan.md`](implementation-plan.md) §7.6。

**已核实、从待办移除的一条**：`investigation-queue.md` D-5（hybrid SWA+full DFlash
drafter + 投机专用 `kv_cache_dtype`）读代码后发现**不完全对**——投机专用 `kv_cache_dtype`
已经是现状（draft KV cache 按 FP8/uint8 分配，与主模型该层自己的 dtype 选择独立）；
"hybrid drafter"这个 vLLM 新能力我们走的是另一条路（固定 6 层全 SWA 的专用小 draft
模型，KV cache 只有 0.007 GB），已经用不同手段达到类似的省显存效果。**这不是降级，
是核实后发现已经做到**，见 [`implementation-plan.md`](implementation-plan.md) §7.6。

其余仍是机会主义、不设强制时间表：

- TURBO_ATTN（FP8 QK MMA）的质量回归修复：per-head descale / Hadamard 旋转 /
  自适应 FP8-BF16 切换。收益 +6%，但 code 工作负载接受率会从 97.8% 掉到 58.6%，
  当前默认关闭。
- FA4 技法用于 prefill / extend 路径（TMA、persistent scheduler、FP8 softmax）。
  **T0 触发条件**（`investigation-queue.md` D-6，**保持观察，不要提前动**）：FlashAttention
  维护者已合入 sm120 PR（#2413，"WIP"），并有面向 5090 的 TMA + warp specialization PR
  在做（#2440，正是这里计划要移植的技法）。但 FA4 算法本体上不了 SM120（缺 tcgen05/TMEM）；
  当前 sm120 路径只有 FP16/BF16、`main` 上部分路径仍报错、在 5090 上比 FA2 **慢约 5%**。
  触发条件是"那批 PR 落到 main 且在 sm120 上跑赢 FA2"——到那时才从"自己移植"变成
  "评估采纳"。
- FP8 attention 的 `num_stages≥2`（SMEM 36 KB « 99 KB，有余量）。
- MoE 输出中心并行（Warp Decode 类方案），2–4 周量级，长期备选。
- GDN kernel 自研（依赖 Track B 的 profiling 结论）。
- **NVFP4 per-token online MoE 量化**（`investigation-queue.md` D-7，vLLM v0.26.0 +
  CuTe-DSL MXFP4）：Laguna 是 256 专家 NVFP4 MoE，直接可比。**kernel 形状 → 写清楚交给
  SparkInfer 团队评估，不直接改其源码**（按 `AGENTS.md` 规矩）——这条我们自己要做的部分
  只是写一份技术提案文档，不是实现。
- **sparkinfer 里还有 9 处未放宽的 gate**（本批发现）：`7a1d69d` 只放宽了 13 处中的 4 处
  （decode / prefill 的 analytic graph dispatch 谓词），其余 9 处——verify-graph 识别、
  各 CTA trait 选择分支、SWA budget、graph-replay 路径选择——是当初**刻意留下**的
  scope 限制，不是遗漏。逐项清单与安全性分析见
  [`sparkinfer-fork-delta.md`](sparkinfer-fork-delta.md)。
  **动它之前必须知道的一条硬风险**：`planner.py` 的 grid occupancy 预算常量是按
  `num_kv_heads=4` 推导的，用到 8 不是放宽谓词就够，需要重新推导。

**纪律**：任何性能改动必须走 `bf diff` 判可比性 + 接受率与质量回归门禁，
2026-07-27 那次"两个不可比的接受率被当成打平证据、损失一整天"的教训写在
[`diagnostics-guide.md`](diagnostics-guide.md) 里。

### Track G · Qwen3.6-25B-A3B（P1，M4→M5）

前置：**先拿到 `config.json`**。当前本地无此 checkpoint，架构参数全部未知。

预期工作（**[待验证]**，以拿到 config 为准）：

- 若为 Hybrid GDN + MoE：GDN 复用 Track B 成果；MoE 走 sparkinfer `moe.fused_moe`
  （NVFP4 已支持），但 **router kernel 需要泛化**——现在
  `runtime/laguna_router.py` 的 `EXPERTS=256 / TOP_K=10` 是模块级常量。
- 若专家数 / top-k 与 Laguna 不同，需决定：泛化现有 kernel，还是为新形状
  再做一个特化。SM120 上 MoE 已被测定为带宽饱和，泛化的性能代价 **[待验证]**。

### Track H · 发布（P2，M5→M6）

- 发布门禁：CI 绿 + 长稳通过 + 两个模型系列的质量回归 + 文档三件套齐备 +
  依赖可从公开源安装（即 sparkinfer 上游化完成）。
- 素材纪律：只发实测数字，标注硬件 / 配置 / 复现命令；不做 apples-to-oranges 对比。
- 版本：`0.2.0` 作为"多模型 + 生产可用"的第一个公开版本。

---

## 5. 里程碑（月度体量）

| 里程碑 | 时间 | 交付 | 验收 |
|---|---|---|---|
| **M1** | 2026-08 | Track 0 全清；Track A 设计定稿；文档基线（本次） | CI 绿、测试全绿、依赖可复现、抽象层设计评审通过 |
| **M2** | 2026-09 | Track A 落地，Laguna 迁移零回归；Track B0 事实基线（含 B0-8 GDN-in-MTP 结论）；Track B1 起步；Track F 的 F1-1/F2-0 机会窗口测试（蹭 A6/C-LIVE 的 GPU 时段）；Track E 的 E5 | Laguna 贪心 bit-exact + 性能不低于基线 3%；Qwen3.6 事实基线文档 |
| **M3** | 2026-10 | Qwen3.6-27B B1 正确性验收 + B2 服务化；Track D 第一批（D1/D2/D3） | Qwen3.6 逐 token 对齐；HTTP 端到端双协议绿；一条命令能起服务 |
| **M4** | 2026-11 | Qwen3.6-27B B3 性能与 MTP；25B-A3B B0/B1 | 接受率与吞吐进基线；25B-A3B 正确性对齐 |
| **M5** | 2026-12 | 25B-A3B 服务化；Track D 收口（D4/D5/D6）；Track E 客户端矩阵 | 三个模型系列同一套配置流程；兼容性表全绿 |
| **M6** | 2027-01 | 长稳、发布门禁、`0.2.0` | 24h soak 通过；发布 checklist 全项 |

**并行度提示**：Track C（稳定性）和 Track F（性能）没有独立的里程碑，
它们是贯穿的；每个里程碑的验收里都含有对应条目。

---

## 6. 风险登记

| # | 风险 | 影响 | 应对 |
|---|---|---|---|
| RK1 | **GDN 是最大未知数**：kernel 性能未知、递归状态与投机/前缀缓存/CUDA Graph 的交互复杂 | 可能拖垮 M3/M4 | B0 阶段先做可行性验证再承诺时间；先用 FLA 拿正确性，把"性能"和"能跑"解耦 |
| RK2 | **sparkinfer 本地补丁未上游** | 发布阻塞；换机器复现不出性能 | T0-5 优先；在合入前用版本钉 + 启动校验让问题显性 |
| RK3 | **抽象层重构回归 Laguna 性能** | 唯一的生产模型退化 | A6 的 bit-exact + 性能门禁作为硬约束；逐步切换、每步可回滚 |
| RK4 | **25B-A3B 配置未知** | Track G 无法排期 | 尽早拉 config；在此之前 Track G 的时间是占位而非承诺 |
| RK5 | **单 GPU、无并行的开发环境** | 每次验证成本以分钟计，是迭代速度的硬上限；**2026-08-01 补充**：任何需要 GPU 的验收项（A6 bit-exact、C-LIVE、F1/F2 的实测、B0/B3 的 profiling）**天然串行，不能靠并行 agent 压缩工期**——本轮已出现多个并行 agent 同时想起服务的风险，协调者已加互斥锁 + 显存守卫应对。这条也是 D3 选 (b) 而不是 (a) 的直接支撑：单卡机器上一个自托管 runner 本身就是又一个要排队抢卡的进程，不会绕开这条串行约束，只会再制造一个抢卡方 | 严格执行 bfdiag 三条法则（不写一次性脚本、比数前先 `bf diff`、失败先读 trace）；GPU 验收任务按优先级排队，不并发申请 |
| RK6 | **依赖链漂移**（torch / cutlass-dsl / sparkinfer / transformers） | 静默变慢或变错 | T0-6 版本合同 + 启动期校验 + CI 锁定；`investigation-queue.md` C-3（PyTorch 2.13.0 wheel 是否带 `sm_120`）**另一 agent 在查，[待验证]，不预判**——若带，自编译要求终结，直接解这条风险 |
| RK7 | ~~**GPU CI 缺失**~~ | 位精确与性能门禁只能人工跑 | ✅ **2026-08-01 已拍板 (b)**：本地 pre-push 门禁 + 人工签核（理由见 §7 D3）。RK5 补充的"GPU 验收天然串行"是这个选择成立的前提——自托管 runner（选项 a）不解决串行问题，只是把它挪到另一个进程里，还多了排队开销。机制落地见 [`implementation-plan.md`](implementation-plan.md) §7.3/C4 |
| RK8 | **Qwen3.6 多模态字段** | 文本版 checkpoint 与多模态版共用架构名，加载器可能误判 | A1 的架构校验要显式拒绝带 vision tower 的权重，给明确错误。**2026-08-01 更新**：D6 拍板选了official `nvidia/Qwen3.6-27B-NVFP4`——这份 checkpoint **本身带 vision tower**，所以"拒绝带 vision tower 的权重"这条规则要改窄：不是"config 里出现 `vision_config` 就整体拒绝"，是"接受该 checkpoint，但要求 loader 显式处于 `language_model_only=True` 模式，断言零 vision 张量被实际加载"。这条留给 A1 落地时处理，不改 `architecture.md`，见 [`implementation-plan.md`](implementation-plan.md) §4/C-2 与 §7.1/B0-1b |
| RK9 | **冷启动/首次真实生产形状路径系统性覆盖不足**（本批新增，2026-08-01） | `235f51e` 修的是"每个未见过的 page-table 宽度都触发 30–100s JIT 重编译"，而这个修复自己的提交记录留了一条**尚未坐实的同类缺口**：DFlash 的 eager verify 回退路径（`mode="verify"`）不在启动期预热覆盖范围内，且 CUDA Graph 捕获**成功**的可观测性目前是 0（只有失败会打 warning，成功只打 info，默认日志配置下不可见）——这不是一次性 bug，是一个模式："首次遇到的真实形状/真实路径"这一类代价一直系统性地被低估，直到真机流量把它暴露出来。`investigation-queue.md` C-1（sparkinfer warmup/autotune 是否用真实形状，另一 agent 在查）与 B0-3（sparkinfer paged attention 的验证范围）都属于同一类别 | 见 [`implementation-plan.md`](implementation-plan.md) §7.3/C7：C7-1（DFlash verify 路径预热覆盖，需 GPU 复现）、~~C7-2（CUDA Graph 捕获成功可观测性）~~ ✅ **2026-08-02 已落地并真机活体复验**（两个后端、真实 HTTP 服务，见 [`../notes/2026-08-02-gpu-memory-audit.md`](../notes/2026-08-02-gpu-memory-audit.md)；顺带坐实一个独立缺陷——`server.engine` 的 logger 到不了日志文件，INFO 级别全部丢失）、C7-3（呼应 investigation-queue C-1，纳入 B0-3 的验证范围）|

---

## 7. 待拍板事项

这些是需要人做决定、不该由实现者顺手选一个的分叉点。

| # | 议题 | 选项 |
|---|---|---|
| ~~**D1**~~ | ~~thinking / reasoning 的产品契约~~ | ✅ **已定案 2026-08-01**：按协议暴露 + `QSR_REASONING_MODE` 开关。Anthropic 侧因签名不可伪造而采用非标准事件，理由与可推翻条件见 §1.4 |
| ~~**D2**~~ | ~~CI 与 torch 的关系~~ | ✅ **已定案 2026-08-01**：两条都要——CPU-only job 守契约，CPU-torch job 扩覆盖 |
| ~~**D3**~~ | ~~**GPU CI 形态**~~ | ✅ **已定案 2026-08-01：(b) 本地 pre-push 门禁 + 人工签核**。理由：这台机器只有一块 GPU（RK5），自托管 runner（选项 a）本身也要抢卡，不解决"GPU 验收天然串行"这条约束，只是换一个进程排队；而"一次验证以分钟计"正是本项目全部效率问题的根源（RK5），选项 (a) 会再制造一个抢卡方，不是治它。(c)（只在里程碑人工全量跑）门禁太松，位精确回归会在里程碑之间悄悄漂移而没人发现。落地机制（`make gate-local` + PR 签核勾选项）见 [`implementation-plan.md`](implementation-plan.md) §4/C-1、§7.3/C4 |
| **D4** | **重命名时机** | 包目录 `qwen-sm120-runtime` → `blackwellm`、环境变量 `QSR_` → `BWLLM_`：随 Track D 一起做，还是推到 `0.2.0` 发布前一次性做 |
| **D5** | **`oracle/qwen36_vllm/` 的处置** | (a) 保留为只读参考（当前）；(b) Track B 完成后整体删除；(c) 现在就删，需要时从 git 历史取 |
| ~~**D6**~~ | ~~**Qwen3.6 主线 checkpoint**~~ | ✅ **已定案 2026-08-01：官方 `nvidia/Qwen3.6-27B-NVFP4`**。前提已被更正——两份候选（官方版、社区版 `sakamakismile/...-Text-NVFP4-MTP`）都带 15 个 `mtp.*` 张量，真正的取舍不是"谁能投机"，是 **provenance vs 过滤 333 个 vision 张量**。排除 vision 张量是一次性机械工作（按 tensor 名前缀跳过 `vision.*`），衍生模型（微调版、下一代 Qwen）迟早都会带 vision tower，这个过滤器无论选哪份 checkpoint 都要写；反过来 provenance 不可逆，发布时官方来源比社区量化站得住。社区文本版留作交叉验证 baseline，不弃用。<br>**衍生影响**：这个决定要求 A1 的 `validate_text_only`（RK8）从"检测到 `vision_config` 就整体拒绝"改成"接受该 checkpoint，但要求 loader 处于 `language_model_only=True` 并断言零 vision 张量被实际加载"，见 RK8 与 [`implementation-plan.md`](implementation-plan.md) §7.1/B0-1 |
| **N8**（原属 [`implementation-plan.md`](implementation-plan.md) §6.1） | **`--session-affinity` 静默失效** | ✅ **已定案 2026-08-01：(c) 启动期拒绝该 flag**。它调的 `mtp_prefill_warm_continue` 只存在于已退役的 `oracle/qwen36_vllm/`，异常被 `try/except` 吞掉 → 每次静默回退冷 prefill、指标恒为 0、零测试覆盖——把静默降级变成显式失败。真要做 warm-continue，等 Track A 的能力查询（§3.5.3 `BackendCapabilities.warm_continue`）落地后再评估 (a) 才划算；现在做，协议地基未定，很可能要重写。落地清单见 [`implementation-plan.md`](implementation-plan.md) §6.1 |

---

## 8. 待验证清单

以下条目在本文档编制时**没有本机实测记录**，一律不作为决策依据。
它们是 Track B0 的主要内容。

- [x] ~~sparkinfer paged attention 在 `head_dim=256 / gqa_group=6 / fp8 KV` 下的正确性与吞吐~~ —— ✅ **B0-3 已答**：能跑且正确（对 fp32 参照 cosine ≥ 0.99999），page_size 64/128 都通。⚠️ 该组合在 sparkinfer 自己的测试套件里从未被测过，首次 JIT decode 62–64s / extend 27s
- [x] ~~FLA `gated_delta_rule` 在 SM120 上能否跑通、速度如何（chunk 与 fused_recurrent 两条路~~ —— ✅ **B0-4 已答**：用 FLA v0.5.2，cosine ≥ 0.99998，48 层 decode 约 1.6–1.9ms。⚠️ 两条路数值不同：`seq_len==1 且有状态` 走 fused_recurrent、其余走 chunk，logits 可差约 30 ULP
- [x] ~~GDN 递归状态更新是否 CUDA Graph capture-safe~~ —— ✅ **B0-5 已答：capture-safe**。捕获+重放 6 步与 eager 逐 bit 一致（max_abs_err=0）。唯一操作要求：新槽位分配时递归状态必须显式 `.zero_()` 一次
- [x] ~~Qwen3.6 modelopt NVFP4 的 tensor 命名与 scale 语义逐项确认~~ —— ✅ **B0-2 已答**：checkpoint 是混合精度（GDN/self_attn 投影 FP8 W8A8；稠密 MLP 与 lm_head NVFP4 W4A16 block=16）。最大发现：`kv_cache_quant_algo=FP8` 有声明但**零个** k_scale/v_scale
- [x] ~~Qwen3.6-27B 在 96 GB 上的 context × 并发可行域（含递归状态显存）~~ —— ✅ **B0-7 已答**（conv 状态数已于 B1 实测更正为完整 kernel size）。⚠️ **但真正的显存底线不是 KV**：反量化缓存让常驻从 19GB 涨到 54GB+，且不受任何显存旋钮控制，见 [`../notes/2026-08-02-qwen36-dequant-cache-memory-floor.md`](../notes/2026-08-02-qwen36-dequant-cache-memory-floor.md)
- [x] ~~mrope-interleaved 在纯文本输入下能否退化为标准 1D RoPE~~ —— ✅ **B0-6 已答：精确退化**。纯文本下 T/H/W position_ids 是同一个 `.expand()` 视图，`apply_interleaved_mrope` 是空操作。可直接写成断言
- [ ] `Qwen3.6-25B-A3B` 的 `config.json`（专家数 / top-k / 是否 hybrid / 是否带 MTP）
- [ ] sparkinfer `moe.fused_moe` 在非 256/top-10 形状下的可用性与性能
- [x] ~~现有 4 个失败测试各自的"正确期望"是什么（尤其 thinking tag 那个）~~ —— ✅ **已解决**：T0-1 已让 CI 恢复绿灯，三个模块都补了 `pytest.importorskip`
- [x] ~~Qwen3.6 的 MTP 层是否带 GDN~~ —— ✅ **已答（B-6 + B0-8 两轮独立核实，两个 checkpoint
  的 `mtp.*` 张量全是 `self_attn.*`/`mlp.*`，零 `linear_attn.*`/`A_log`/`conv1d`）**。
  ⚠️ **但由此推不出"B3 最难的一项消失"**——原判断把草稿侧与验证侧混为一谈了：verify 是把候选
  token 整段跑一遍**主模型完整 64 层（含 48 层 GDN）**，候选被拒时主模型递归状态已被不可逆推进，
  这跟 MTP 头的架构无关。真正消失的只有草稿侧那块。B3 体量**不下修**，见
  [`implementation-plan.md`](implementation-plan.md) §7.1/B3 与 `investigation-queue.md` D-3（ReplaySSM）
- [x] ~~NVFP4 KV vs FP8 KV 在我们卡上的对比~~ —— ✅ **已答：这个对比在当前技术栈上没有对照组**。
  sparkinfer paged-attention（唯一的 attention 内核）只接受 fp16/bf16/fp8_e4m3 三种 KV dtype，
  NVFP4 KV 会直接 `TypeError`。所以 B3 的"KV dtype 选型"**不是一个待选项**，FP8 是唯一可行值。
  见 `investigation-queue.md` C-2
- [x] ~~PyTorch 2.13.0 PyPI wheel 是否带 `sm_120`~~ —— ✅ **带**（干净 venv 实测
  `2.13.0+cu130 [... 'sm_100', 'sm_120']`）。**自编译要求终结**，RK6 与 H1"可从公开源安装"解除
  这一处阻塞。见 `investigation-queue.md` C-3
- [x] ~~**（本批新增）** 当前生产配置下的真实显存占用带日期来源的审计（Track F/F2-0）~~
  —— **已做**（2026-08-03，标准模型，配置逐项写明）：
  [`../notes/2026-08-03-production-memory-audit.md`](../notes/2026-08-03-production-memory-audit.md)。
  capacity=1/num_slots=2/max_context=131072：**CG 72.39 GiB，eager 77.69 GiB**（CG 少用 5.30）。
  原记录的"94.2/97.9"与"76.0/95.6"配置不同、未交叉验证，两者都不描述当前标准模型。
  🔴 **主要发现：反量化缓存只解决了一半。** `free_nvfp4_raw_params()` 只覆盖 NVFP4 的
  56 层 MLP；**FP8 那 237 个张量的 BF16 缓存是永久的，且 FP8 原件同时常驻**——
  按 checkpoint 真实尺寸算：原件 9.99 GiB + BF16 缓存 19.99 GiB = **两份 29.98 GiB**，
  而 `forward` 只读 BF16 那份。**照搬 NVFP4 做法可立刻回收 ~9.99 GiB，无需任何 kernel 工作。**
  ✅ **已实施**（W8A8 预演否定后不再需要保留 FP8 原件）：`free_fp8_raw_weights()`，
  真机实测**释放 233 个 Linear、常驻 44,626 → 38,698 MiB、实收 5.79 GiB**，输出不变。
  （实收小于预估 9.99 GiB 是分配器复用所致，与 NVFP4 那轮同一现象，不是没生效。）
- [ ] 🔴 **比上一条大得多的一笔：标准 checkpoint 发了 FP8 KV scale，我们没用。**
  新加的反向检查在第一次真实运行就报出 `k_scale x16, v_scale x16` 无人消费。
  `qwen36_model.py` 里"本 checkpoint 发货零个 k_scale/v_scale（B0-2）"**对 `nvidia/`
  成立、对标准模型是错的**（实测 nvidia 0/0，unsloth 16/16），而"用 BF16 KV 绕开
  scale 问题"这个设计正建立在那个前提上。标准 checkpoint 发的是完整静态 per-tensor
  对称 FP8 KV 方案。**KV 是 8192 MiB/槽、审计里最大的单项，FP8 KV 直接减半**
  （num_slots=2 省约 8 GiB，capacity=4 省约 20 GiB）。
  ⚠️ BF16 KV 现在是正确且在跑的，这是未兑现的机会而非 bug；动手前必须过 B1-R。
- [x] ~~DFlash 的 eager verify 回退路径是否真的会在生产流量下被打到~~ —— ✅ **已答：不会**。
  今天它只有一条触发路径（verify CG 启动期捕获失败），是潜伏风险不是活跃故障；且
  `QSR_DFLASH_REQUIRE_CG=1` 已让它在生产中不可达。沿途挖出的 eager-vs-CG 数值分歧**也已结案**：
  稠密 fp32 oracle 判定两条路径在 attention 算子层面都对（cos ≥ 0.999997，高于本仓库 ≥0.999991 的
  标准），分歧来自 MoE 离散路由放大微小数值差，**不是 split-KV merge 有 bug**。见
  `investigation-queue.md` C-1
- [x] ~~**接受率与槽位 KV 使用率的可观测性为 0**~~ —— ✅ **已修（`b27915b`）**：桶宽改为从 `NUM_SPECULATIVE_TOKENS` 推导（此前引擎 5、metrics 9、真值 15 三个互不相同的字面量）；`na >= 5` 从静默丢弃改为钳进溢出桶；两个记录函数已接。测试钉的是关系而非数字，反向验证过（桶宽改回 9 则变红）

- [~] **CUDA Graph 捕获成功的可观测性** —— 🟡 **代码已修（B3），但未活体复验**。`Qwen36Backend.snapshot()` 原本硬编码 `dflash_cg_status=()`，Laguna 的非 DFlash decode-CG 路径也未被跟踪；两者现在都记录捕获成功/失败并经 `snapshot()` 与 `/debug/stats` 暴露。⚠️ **活体复验需要一次完整前向，而那正是触发反量化缓存显存爆掉的动作**，在共享卡上未做

- [ ] **（本批新增）** `NUM_SPECULATIVE_TOKENS` 从 15 静态调大是否能在不损失接受率的前提下提升吞吐
  （Track F/F1-1）

---

## 9. 与本文档配套的其他文档

- [`e2e-and-quality-plan.md`](e2e-and-quality-plan.md) — Track C（稳定性）/ Track E（兼容性）的
  详细分期、GPU 窗口调度、"反复审查"节奏机制
- [`architecture.md`](architecture.md) — 当前架构与目标架构
- [`model-support.md`](model-support.md) — 模型支持矩阵 + 接入新模型的操作指南
- [`qwen36-rebuild-spec.md`](qwen36-rebuild-spec.md) — Track B 重建规格：`oracle/qwen36_vllm/`
  逐模块判定与新位置映射、Qwen3.6-vLLM 时代验收基线、在 Track A 抽象上的重建设计、风险清单
- [`diagnostics-guide.md`](diagnostics-guide.md) — bfdiag 使用指南（仍然有效，必读）
- [`archive/README.md`](archive/README.md) — 已归档文档索引及归档原因
- [`../notes/README.md`](../notes/README.md) — 116 篇调查记录的分类索引
