# Flash-Next（qwen4_exp）自研 runtime 原生支持实施方案

日期：2026-08-27。基线与靶子见
`2026-08-27-flashnext-sglang-baseline.md`（sglang 20.5 tok/s，靶子 100）。
权重：`/home/bot/models/Qwen3.8-Flash-Next-NVFP4-RadixArk`（126 GiB）。
sglang 参考实现：`/home/bot/project/sglang` `qwen4-support` 分支
（`python/sglang/srt/models/qwen4_exp.py` / `qwen4_exp_mtp.py` /
`qwen4_ple_nvme.py` + QSA 后端）。**不 import，只作对照**。

## 1. 架构地图（config + 权重清点实测）

48 层，布局 12 × (3×(GDN→MoE) + 1×(QSA→MoE))；hidden 2560；
每层：hyper-connection 混合器（4 分支、低秩 320、398 张量）→ 注意力
（36 GDN / 12 QSA）→ MoE（512 专家 top-10 + shared，sigmoid 门）；
第 2 层注入 PLE n-gram 嵌入（132 张量，47.7 GiB FP8 表）；
MTP 1 层（QSA）；mrope-interleaved（纯文本退化为 1D，qwen36 已有先例）；
vocab 248320；262K 上下文。

权重命名（与 Laguna 的差异已核）：
`model.language_model.layers.N.mlp.experts.E.{gate,up,down}_proj.`
`{weight_packed, weight_scale, weight_global_scale, input_scale}`
——前缀多一层 `language_model.`，激活尺度键是 `input_scale`（Laguna 是
`input_global_scale`）。注意力/其余为 BF16。visual 333 张量，文本路径跳过。

## 2. 资产复用表

| 子系统 | 现有资产 | 差距 |
|---|---|---|
| GDN | `qwen36_model` GDN + 融合 conv 内核 | 输出门 silu→**sigmoid**；形状差（48 V 头/16 QK 头 ×128） |
| MoE | `qwen38_moe.QwenMoeLayer` + `qwen38_sparkinfer_moe`（E=512 已验证）+ 我们的路由 | 装载命名适配（上节）；b12x 路径直接可用 |
| PLE/engram | `runtime/model/engram.py` 哈希数学（已对照 SGLang/vLLM 钉死） | NVMe/mmap 流式 + **图安全 pinned gather**（自己写，sglang 没做成的我们做） |
| QSA 稀疏注意力 | `dsv4_attention.py` 的 indexer/稀疏选择经验 | 全新实现：微块级选择（预算 2048 tok/512 块、压缩比 4、MQA indexer 4Q/1K×128） |
| hyper-connection | 无 | 新小模块（低秩 320、4 分支、silu 门） |
| MTP | `qwen36_mtp` 全套（K=3、状态回滚、图） | MTP 层是 QSA 注意力 |
| KV/状态池 | `qwen36_kv_arena` 混合双家族 | 加 QSA KV（2 头×256）+ indexer KV（1×128） |
| NVFP4 GEMM | `nvfp4_gemm_sm120` + b12x fused MoE | 直接用 |

## 3. 分阶段（每阶段有正确性闸门）

**P1 eager 文本路径**：装载适配 → embed+PLE（先小表驻留）→ 48 层
（GDN-sigmoid、QSA、hyper-conn、MoE）→ lm_head。**闸门：对 sglang
服务器逐层/逐 token 对照**（sglang 基线服务可随时重拉当 oracle）。

**P2 CUDA Graph（最大单点收益）**：PLE gather 用固定地址 pinned staging
+ 图内 gather kernel（参考 sglang `_gather_ple_embedding_from_pinned_kernel`
的 Triton 思路，放进我们自己的图捕获）；decode 图覆盖 48 层+MoE。
sglang 在这里卡死（捕获期非 pinned 拷贝），我们从设计上避开。

**P3 MTP**：复用 qwen36_mtp 回滚机制，MTP 层走 QSA；GDN 状态布局自持，
fp32/bf16 自选——打破 sglang "MTP×flashinfer 互斥" 限制。

**P4 调优**：chunked prefill 尺寸、QSA decode kernel、PLE 预取与
decoder 重叠、接受率调参。

## 4. 性能预算（100 tok/s 的来源）

6B 激活 × ~0.55 B/param ≈ 3.3 GB/token；~1.5 TB/s 带宽上限 ≈ 450 tok/s
理论。20.5→100 的构成估计：CUDA Graph 消除 48 层×MoE 的启动开销
（~1.8–2.2×）+ b12x SM120 融合 MoE/GDN（~1.3×）+ MTP 兼得快 decode 后端
（~1.3×）+ PLE 预取重叠（消除盘读停顿）。

## 4.5 进展（2026-08-27 当日）

- **MoE 地基已通过数值闸门**：`runtime/backends/flashnext_moe.py`
  （modelopt 原生尺度约定：`weight_scale_2` 原样、`a*_gscale =
  1/input_scale`，直接读自 b12x `_prepare_modelopt_nvfp4_runtime_alphas`）
  + `tests/test_flashnext_moe.py`（真实 RadixArk 层 0，对手工解量化参考
  cosine 0.989 / 幅度比 0.96，W4A4 激活量化下的合理残差）。错误约定的
  失败形态已钉死：方向反 = 2e15 或全 0（差 8 个数量级，不会误判）。
- 权重命名/形状全部实测钉死（见 §1）；Laguna 兼容弃用（模型质量差），
  旧 `qwen38_sparkinfer_moe.py` 仅保留其 `SparkinferMoELayer`/arena 容器
  供复用。
- 决策：ngram/PLE 支持**双模式**——NVMe 流式（当前 23 GB 内存）+
  全量驻留内存（用户正在加内存），加载期按可用内存自动/显式切换。
- **hyper-connection（GatedResidual）完成**：
  `runtime/model/flashnext/hyper_connection.py`，mix/combine/grouped
  Gemma-RMSNorm 公式从参考实现逐行钉死，6 项单测（零权重恒等式手推）全绿。
- **PLE 完成**：`runtime/model/flashnext/ple.py`——
  哈希（splitmix64 乘子 + XOR mix + 每头素数模 + EOS 段重置，常量直接
  取自 checkpoint）、128 分片 FP8 表 mmap 流式 gather + `make_resident()`
  驻留模式、注入层（key/value proj、分组 norm、sqrt-soft sigmoid 门、
  dilation=3 因果 conv1d + 逐请求状态）。6 项测试全绿（含 decode conv 与
  prefill conv 逐位一致、gather 与直读分片一致）。
  注意：本模型哈希与 LongCat/vLLM 族**不同**，早期 `runtime/model/engram.py`
  的 LongCat 数学不适用于此模型（保留作家族参考）。
- **GDN 可复用确认**：`Qwen36GatedDeltaNet` 按 config 字典参数化
  （`linear_num_value_heads=48`/`linear_key_head_dim=128`/conv 4 全部
  匹配），Qwen3.8-27B 同族已验证，几何换参即可。
- **QSA 完成（bring-up 级）**：`runtime/model/flashnext/qsa.py`——
  indexer（index_qk_proj；Q 立即做 Gemma-RMSNorm + 部分 RoPE，raw token K
  先按 4 行做 FP32 均值、再做 norm + 组首位置 RoPE；64/128 NeoX 半区布局，
  文本 mrope 退化 → ReLU 打分求和/√128 →
  因果掩码 → top-512 块）+ 主注意力（q/k/v + sigmoid 输出门、24Q/2KV
  ×256、partial RoPE 64/256、gather 式稀疏 decode + o_proj）。
  5 项测试全绿（真实层 3 权重）。稀疏内核与 paged 版留到优化阶段
  （当前 bring-up 用 gather+稠密注意力，语义等价）。
- **GDN bring-up 通过**：`Qwen36GatedDeltaNet` 直接吃真实层 0 权重，
  prefill/decode 状态一致性过（注意 conv_state 宽度是完整
  kernel_size=4，非 3——`new_state` 里有推导注记）。

**剩余工作（下次会话起）**：
1. 整机组装（embed → 48 层 hyper-conn/PLE/GDN/QSA/MoE → 末级
   mixer.mix → final norm → lm_head）+ 全权重装载器；
2. 后端接线（slot、KV/GDN 状态池、QSA 块选择缓存、批处理、服务循环）；
3. 对 sglang 逐层/端到端 logits 对照（正确性闸门）；
4. CUDA Graph（PLE gather 图安全化——sglang 卡死处，我们的突破口）+
   MTP；5. 性能调优至 100 tok/s。

测试现状：`pytest -k flashnext` 20 passed（MoE 1 + hyper-conn 6 +
PLE 6 + GDN 2 + QSA 5），全在 CPU/单卡可跑。

## 4.6 整机 bring-up 成功（2026-08-27，里程碑）

`scripts/fn1_full_model_bringup.py`：48 层全权重装载（74.9 GiB），
greedy 生成 **"The capital of France is Paris."** 正确。期间修掉的三个
数值 bug（都已钉死）：
1. **RoPE 布局**：QSA/indexer 最初写成相邻交错对，生产验证的是 **NeoX
   半区布局**（`x[i], x[i+half]`，见 `runtime/kernels/rope.py`
   `_rope_kernel`）——改 `qsa.py` 两处 `_rope`。
2. **GDN 输出门**：Qwen3.8-27B 是 `swish`、Flash-Next 是 **`sigmoid`**
   （config `output_gate_type`）；给 `Qwen36RMSNormGated` 加了
   `gate_act` 参数（默认 silu，生产不变），GDN 从 config 透传。
3. **MoE 输出 arena**：`SparkinferMoEOutputArena` 默认 3072（Laguna），
   Flash-Next hidden 2560，装载器显式传 `cfg.hidden_size`。
另：embed 转 bf16、lm_head 用 fp32 算 logits。

**遗留（与本工作无关）**：`test_architecture_spec.py` 的
`test_quantization_is_compressed_tensors_with_fp8_kv` 失败——本地
Laguna checkpoint `config.json` 于 2026-08-27 15:05 被外部重写、丢了
`kv_cache_scheme` 声明（stash 验证过与我无关）；Laguna 已弃用，未追。

## 4.8 性能爬坡实录（2026-08-27 下午）

| 阶段 | decode tok/s | 备注 |
|---|---|---|
| eager 增量 | 6.5 | Python 逐层发射 |
| CUDA Graph | 13.55 | PLE 冷读串行 ~45ms/token 成为主瓶颈 |
| + 并行冷读 | **19.41** | 16 分片缺页等待重叠（83→27ms 实测） |
| sglang 基线 | 20.5 | （含 MTP+图） |

关键教训：
1. fn3 profile 揭示 GPU 每步实际只忙 ~18.6ms，其余全是发射空隙——图化吃掉；
2. 图化三件事必须做对：QSA 池化定长（全池+因果掩码）、状态原地更新
   （PLE conv 加 in-place 变体）、动态 cat 全部换成固定地址池写入；
3. RoPE 是 NeoX 半区布局、GDN 门是 sigmoid 而非 swish、MoE 尺度是
   modelopt 原生约定——三个错都曾造成乱码，已被单测钉死；
4. MTP bring-up 完成（fn4）：draft logits 正确生成（" Paris." 链），
   但 1-step acceptance 仅 0.31 vs sglang 0.68——gap 定位为 draft 层缺少
   teacher-forced 历史 KV 重放（sglang `_sync_real_suffix`/
   `_continue_draft` 机制），修正需把 FlashNextMTP 接入引擎级
   spec-rows/pool 设施（qwen36_mtp.py 的既有机制，复用而非重写）。
5. lm_head/embed 统一 bf16。

## 4.9 MTP 阶段成果（2026-08-27 晚）

- `runtime/model/flashnext/mtp.py`：FlashNextMTP 草稿模型完成——
  融合（`_fuse_residual_linear_shared`：embedding 与 hc-hidden 各自
  pre-fc-norm + 线性投影后逐分支相加）、单层 QSA（自带 indexer）、
  **BF16 MoE**（MTP 专家不量化，gate_up_proj [512,1280,2560] +
  down_proj [512,2560,640] 打包布局）、末级 mixer；lm_head 与主模型共享。
  fn4 验证：装载/前向/有限性 ✓，" Paris"→"." 草稿正确。
- 修掉三个 MTP 相关 bug：pre-fusion norm 应为**整条普通 GemmaRMSNorm**
  （非分组）、QSA KV 缓存必须保留 [S,KV,D] 维、lm_head 转 bf16 后
  同步修正调用方（mtp_sync 不能再 .float() 输入）。
- **acceptance 探针（fn5）**：同位判定 0.31 → **0.40**
  （teacher-forced sync 机制：每步把真实 (token, hc_hidden) 重放穿过
  MTP 层——引擎 `_sync_real_suffix` 的等价物）。与 sglang 0.68 的
  剩余差距锁定为 draft 链式语义（K≥2 需批量 verify + 链式草稿）。
- **重要工程结论：K=1 在逐步图结构下无加速**——每 token 仍需一次
  完整 replay；收益必须走 K≥2 链式草稿 + 一次 qo_len=K+1 批量 verify
  （qwen36_mtp `verify_batch_spec` 同款），并解决 GDN 状态快照/回滚
  （B3 机制可复用）。这是下一步引擎集成的核心工作。

PLE 常驻（内存到位后）预计把前奏压到 ~0.2ms → 纯图 replay ≈ 36+ tok/s，
叠加修正后的 MTP 即达 100 靶子的路径明确。

**下一步**：增量 decode（持久化 GDN/QSA-KV/PLE-conv 状态）测真实
tok/s → 对 sglang logits 数值门 → CUDA Graph + MTP 冲 100。

## 4.7 增量 decode 打通 + 首个自研性能数（2026-08-27）

`runtime/model/flashnext/model.py` 的 `decode_step` + `FlashNextSession`：
GDN 递归态、QSA 的 indexer-K/主 KV 缓存、PLE 卷积态全部持久化，
逐 token 增量前向。修掉四个 bring-up bug：PLE gather 要 CPU ids、
全程保持 2-D 形状（1-D 会打穿 router 的 rank-2 校验）、embed 输入
维度、QSA KV 首步要保留 `[1, KV, D]` 维。

**结果**：`scripts/fn1_full_model_bringup.py` 增量生成
`"The capital of France is Paris. ..."`（正确），
**自研 eager decode = 5.56 tok/s**（无图、无 MTP、QSA 每步 O(n)
重池化 + Python 块循环、PLE 每步 NVMe mmap gather）。

对比：**sglang 20.5 tok/s**（MTP + CUDA Graph + 融合内核）；靶子 100。
差距来源（待 profile 确认权重）：Python/eager 启动开销、无图、
无 MTP、QSA 选择未融合、PLE 盘读在关键路径。

## 5. 风险

- QSA 微块选择语义必须与参考逐位对齐（indexer 的 jaccard 在 sglang
  测试里是 0.975/0.991 底线，我们直接对照参考实现）。
- PLE 哈希必须与参考逐位一致（engram.py 已验证数学，装载后要用真实
  表做端到端抽查）。
- 23 GB 内存：PLE 流式 + 8 GB 缓存照搬基线策略；加载期分片流式处理。

## 4.10 K≥2 批量 verify 的工程结论（2026-08-27 深夜）

fn6 尝试在脚本层缝 K=3 链式草稿 + 批量 verify，暴露三个结构性冲突：
1. 无状态 forward 与会话池脱节（GDN/QSA/PLE 状态不走 sess 池）；
2. PLE 的 n-gram 窗口必须滑过候选 token 序列，与 sess.window 同步；
3. 部分接受时按 token 重放恢复状态的成本（≤K 次完整 replay）吃掉
   投机收益的一半以上。

结论：批量 verify 必须做进图引擎层（qwen36_mtp `verify_batch_spec`
+ `Qwen36MTPGDNRows` + MTP 页池的同构物），即：
- `decode_body` 扩展 qo_len=K+1 批量图（GDN chunk 路径天然支持多
  token；QSA 批量选择与 attention；PLE 批量窗口 gather）；
- spec rows 快照：GDN conv/recurrent + PLE conv 的 per-candidate
  列快照与回写（qwen36 的 B3 机制同构）；
- MTP 池页化 + 链式草稿常驻图。

这是下一阶段的主工程（预计 2-3 天），完成后叠加 PLE 常驻即达
100 tok/s 靶子。当前 19.41 tok/s（无 MTP，PLE 盘读未解）已记录为
基线对照点。

## 4.11 K=3 固定状态行 verify 落地与质量闸门（2026-08-28）

OpenCode 会话停在“给 model.forward live pools”的断点后，已把验证链路
正式落到 `runtime/model/flashnext/spec.py`：

- GDN 36 层与 PLE conv 各有固定地址 candidate rows；QSA index/K/V 直接
  写 append-only pool，部分接受只提交逻辑长度；commit 不再重放整模。
- `anchor + K drafts` 一张 CUDA Graph，MTP teacher-forced sync 使用
  **shifted real token** 与对应 target HC hidden，MTP QSA 有独立持久池。
- QSA 修正了两个参考语义错误：raw token K 必须先 FP32 分组平均再
  norm/RoPE；top-k 的无效 block 必须保持 `-1`，并显式追加最多 3 个
  未压缩 tail token。短上下文不再把 `-inf` tie 当作真实 block。
- 新增 `fn6_spec_batch.py` 真 checkpoint 闸门，同时比较投机提交流与普通
  M=1 teacher-forced 流的 logits、GDN/PLE/QSA 全状态。

真卡结果（RTX PRO 6000 96GB，prompt `The capital of France is`，24 轮）：

| verify 形状 | accept rate | tok/s | M=1 top-1 | state/logit 误差 | 结论 |
|---|---:|---:|---:|---:|---|
| 全部 M=4 批量 | 0.306 | 15.13 | 0.804 | 非零 | 不可用 |
| stateless M=1，GDN/lm_head 仍批量 | 0.222 | 11.54 | 0.900 | 非零 | 定位用 |
| **全 M=1 bit-path + 固定行 commit** | **0.472** | **15.54** | **1.000** | **0** | 质量基线 |
| 仅主 NVFP4 MoE 恢复 M=4 | 0.486 | 17.98 | 0.864 | 非零 | 不可默认 |

最终默认是全 M=1 bit-path：58 个提交 token 的 logits 逐位相等，GDN conv/
recurrent、PLE conv、QSA index/K/V 全部 cosine=1、max_abs=0，且无 greedy
mismatch。峰值显存 82.41 GiB，余量 13.18 GiB。`batch_main_moe` 保留为显式
实验开关但默认关闭；b12x 当前没有无损 BF16 expert-weight 接口，不能把 MTP
BF16 专家冒充为其 W4A16（W4 权重 + BF16 激活）路径。

性能结论：状态拓扑、shifted MTP 和 rollback-free commit 已证正确，但 exact
verify 仍把每个候选按 M=1 执行，15.54 tok/s 低于无 MTP 图基线 19.41 tok/s，
尚不应接到正式 server 默认路径。下一步必须做 shape-invariant 的 HC/GDN/MoE
小 M 内核或逐组件数值资格化；质量闸门不得放宽。

## 4.12 接受率复核与 PLE 热路径重写（2026-08-28）

4.11 的 0.472 不是可代表线上使用的接受率：当时直接把裸文本
`The capital of France is` 喂给模型，生成很快进入特殊 token 重复区。`fn6`
现已固定使用 checkpoint 自带 chat template 和长技术问答提示，64 轮 K=3：

- draft acceptance **139/192 = 72.4%**；逐位置 **82.8% / 68.8% /
  65.6%**；accepted-length 直方图 `[11, 9, 2, 42]`；
- 每轮含 bonus 平均提交 **3.17 token**，略高于同 checkpoint 的 SGLang
  基线 3.05；
- 203 个提交 token 对普通 M=1 target 流 top-1 100%，logits cosine 1.0，
  GDN/QSA/PLE 全状态 cosine 1.0、max-abs 0，无 greedy mismatch。

PLE 的旧 mmap + torch advanced-index 路径即使页已热，同一批 4×16 行仍需
48.9 ms。`runtime/model/flashnext/ple.py` 改为：

1. 把 160-byte 行映射成 4 KiB page key，整批去重；
2. 持久 32-worker `pread` 队列一次提交所有缺页（不再逐 gather 建线程池）；
3. 行结果用 bytes 一次拼接，删除线程内逐行 torch 赋值；
4. 双 pinned staging，原始 FP8 bytes 异步 H2D 后在 GPU 转 BF16；
5. 修复 FIFO 覆盖槽位后旧 key 仍指向新 row 的错误映射，并补回归测试；
6. K+1 PLE recurrence 合成一次 depthwise conv，输出与逐行实现 bit-exact。

同一进程、同一行集微基准：48.9 → **3.57 ms（13.7×）**；GPU 端到端
3.82 ms 且 CPU/GPU 输出逐 bit 相等。整模 64 轮 PLE preparation：
10.506 → **0.182–0.190 s（55–58×，2.8–3.0 ms/轮）**。整链 verify 自身
在重复运行间有 27.7–32.0 s 波动，因此只把 PLE 分项作为确定收益；端到端
观测为 5.51 → 5.99–7.00 tok/s，不把 GPU 波动归因给 PLE。

缓存/QD 扫描也已收口：16 workers 的 P95 12.7 ms，32/48/64 均约 4.5 ms，
32 尾延迟最稳；真实生成的 page-cache hit 为 0，因此 page cache 保留可配但
默认关闭，避免复制 OS cache，131072-row（约 20 MiB raw）缓存保持默认。

负结果同样钉死：PLE M=4 注入局部可从 0.915 降到 0.414 ms，但 BF16
GEMM 归约顺序变化会被后续层放大；整模 acceptance 72.4%→66.7%、target
top-1 100%→95.8%、state max-abs 13.375，故该实验开关已删除。`torch.compile`
注入虽有 2.6× 局部收益，也产生 0.0078–0.0156 的输出漂移，未进入生产路径。

## 4.13 PLE oracle 修复、真实接受率与无损批量边界（2026-08-28）

4.12 的 72.4% 仍然不是可信质量结论：它只证明 draft 与自研 target 一致，
没有证明 target 与模型参考实现一致。同一份 109-token chat-template TCP 提示送入
SGLang 后，oracle 以高置信度从 `We need answer user: ...` 开始；当时自研 target
却进入重复垃圾。逐公式复核 PLE 后定位到四条自研路径都漏掉了参考实现的
`SiLU(short_conv_output)`：prefill、单步 decode、图内 decode 和 K+1 verify 都把
原始深度卷积值直接加回了 widened hidden。

修复后，同一提示的自研输出与 SGLang oracle 从 `We need answer user: ...` 开始
一致，K=1 的 verify/ordinary-decode 状态逐位一致。正确 target 下重新跑 K=3、
64 轮：

- draft acceptance **120/192 = 62.5%**；逐位置 **76.6% / 57.8% /
  53.1%**；accepted-length 直方图 `[15, 12, 3, 34]`；
- 每轮含 bonus 平均提交 **2.875 token**；
- 184 个提交 token 对普通 M=1 target 流 top-1 100%，logits cosine 1.0，
  GDN/QSA/PLE 全状态 cosine 1.0、max-abs 0，无 greedy mismatch；
- 端到端 **21.57 tok/s**（8.529 s / 184 committed tokens）；verify 7.039 s，
  MTP 1.427 s，PLE preparation **0.226 s（3.53 ms/轮）**。

因此 62.5% 是修正 target 后的真实内部接受率，不是测试数据错位；它对应平均
2.875-token commit，具备实际投机收益。SGLang 旧记录的 68%/3.05 使用的是另一
条基准请求，不能直接判定 5.5 个百分点为实现差距，后续必须用同 prompt 做
`bf diff` 等价对照。

PLE 读路径继续补齐了固定输出缓冲：`gather(out=...)` / `embed(out=...)` 直接写
decode/verify 的地址稳定 BF16 buffer，删除临时 GPU embedding 与额外 copy；随机
页仍按 shard/连续 page run 合并读取。真实 trace 的跨轮 page hit 仍为 0，所以
默认保持 32 workers、page cache 关闭、131072-row cache，不能用重复同一页的
微基准把 512 MiB page cache 冒充线上收益。key/value projection 也必须保持两次
参考 GEMM：合并权重虽少一次 launch，但实测 M=1 value 已有 0.00390625 漂移，
M=4 key 最大漂移 0.0625，已拒绝。

另发现 b12x 包改名后 runtime 仍只设置旧 `SPARKINFER_*` 环境变量，而当前实现
读取 `B12X_DYNAMIC_DETERMINISTIC_OUTPUT` / `B12X_ENABLE_DYNAMIC_DOWN_SCALE`；
现已在 import 前同时设置新旧命名。对 routed expert 单层做 M=4：0.600 ms vs
四次 M=1 的 2.258 ms（3.76×），但输出并非逐位一致。进一步只批 routed expert、
router/shared expert 保持 M=1 的整模 CUDA Graph 实验仍累积为 NaN 和连续 `!`，
故该批量入口已删除，不能进入默认路径。`fn6` 新增 logits/state 全量 finite 闸门，
以后 NaN 会直接失败，不再被 `argmax(NaN)` 伪报成 top-1 100%。

## 4.14 MTP prompt 批量化、短上下文 GQA 与同请求 SGLang 对照（2026-08-28）

同一份 109-token chat-template TCP 提示、temperature=0、K=3、batch=1 已在
两个 runtime 上复跑。SGLang 为 TTFT 0.570 s、HTTP 总时长 5.168 s、
35.80 completion tok/s、post-first 40.02 tok/s；scheduler 采样接受率 0.56、
平均接受长度 2.67。自研 runtime 的旧有效结果为 TTFT 2.464 s、请求 36.677 s、
5.04 completion tok/s、steady 7.66 tok/s、接受率 0.625、平均每轮提交 2.875。
因此差距首先是执行时间，不是“接受率太低”。

本轮收敛了两个真实热路径：

1. MTP 的 BF16 MoE 不再执行 `weight[expert_ids]`。109 行 × top-10 会为两个
   专家矩阵物化约 10 GiB 临时权重；改为按 expert 稳定排序、两个
   `grouped_mm`、再恢复 token-major 路由顺序。CPU 保留 indexed reference。
2. 2048-token QSA budget 内的 MTP sync 不再走每行固定 2051 槽的 decode gather，
   也不再把 2 个 KV 头 `repeat_interleave` 为 24 个头。新 causal-prefix GQA
   直接在原始 KV 头上计算短前缀，超过 budget 仍回落到完整 QSA 稀疏路径。
3. Flash-Next target verify 的 GDN 恢复普通单-token decode 递推到私有 work row，
   每步写固定 candidate row。GDN multistep 虽“批量”，但真模型状态 cosine
   0.964 且出现 greedy mismatch，并且 verify 更慢，不能使用。

最终 64 轮真卡结果（RTX PRO 6000 96GB）：

- acceptance **120/192 = 62.5%**，逐位置 76.6% / 57.8% / 53.1%，平均每轮
  提交 2.875 token，与优化前完全相同；
- 184 committed tokens 对普通 target decode：top-1 100%，logits cosine 1.0，
  GDN/QSA/PLE 状态 cosine 1.0、max-abs 0，无 greedy mismatch；
- TTFT **2.466 s**，initial MTP **0.056 s**，decode **6.364 s**，请求
  **8.886 s / 185 completion tokens = 20.82 tok/s**，decode **28.91 tok/s**；
- verify **4.924 s / 64 = 76.9 ms/轮**，MTP 1.389 s，PLE 0.196 s；
- 最终驻留 82.15 GiB、空闲 13.44 GiB；此前固定宽度 MTP gather 的运行末尾
  驻留达到 95.59 GiB、空闲为 0。

相对旧 runtime，E2E 5.04→20.82（4.13×），steady 7.66→28.91（3.77×），
initial MTP 10.196→0.056 s（约 182×），verify 352.4→76.9 ms/轮（4.58×）。
相对同请求 SGLang，E2E 尚慢 41.8%，post-first 尚慢 27.8%；剩余主差距是
target prefill（2.466 vs 0.570 s）和 exact verify，而不是 PLE 或 acceptance。

两个进一步小 M 实验已整模否决并从生产代码撤回：HC mix 的独立行 bmm 在局部
真实权重上逐 bit 相同，但进 CUDA Graph 后 16 轮状态 cosine 降到 0.964；仅批量
HC combine 短跑全等，64 轮后仍累积出 3 个 greedy mismatch。QSA 只批量 FP32
attention core、保留逐行 `o_proj` 也产生 greedy mismatch。以后不能用局部
M=4/短跑数据恢复这些分支；必须以 64 轮全状态回放为准。

## 4.15 同请求 SGLang Nsight profile 与状态 dtype 修复（2026-08-28）

用同一个 109-token TCP chat prompt、temperature=0、64 completion token 和
NEXTN K=3 对本机 SGLang 做 `cudaProfilerApi` 范围采集。HTTP 为 TTFT
**0.475 s**、总时长 **2.260 s**、首 token 后 **35.87 tok/s**；64 token 用
26 个 verify 轮完成，平均每轮提交 **2.462 token**，由此可推算 draft acceptance
约 **48.7%**。这说明同请求下“较低接受率”仍可得到更高吞吐，轮执行时间才是
首要变量。

Nsight trace `/tmp/sglang-fn-profile-20260828-r3.nsys-rep` 给出了精确调用数：

- target routed MoE 的两类主 FP4 CUTLASS GEMM 各 1296 次，正好是
  `48 layers × (1 prefill + 26 verify)`；verify 的每层 MoE 是一次真正 M=4；
- 36 个 GDN 层的 fused recurrence 共 936 次、12.681 ms，即每轮全部 GDN
  **0.488 ms**；causal conv 同为 936 次、每轮 **0.132 ms**；
- 相比之下，自研 exact trace 两轮有 384 次 b12x MoE kernel，并有数千次
  GEMV/elementwise。瓶颈不是 GDN 递推本身，而是候选行把整层外围重复四次。

同时发现 checkpoint 的 `text_config.mamba_ssm_dtype` 明确为 `float32`，SGLang
状态池也实际分配 FP32 SSM；自研 `new_layer_states()` 却硬编码 BF16。现已把 dtype
纳入 `FlashNextTextConfig` 并据配置分配 recurrent state，conv state 仍为 BF16。
真模型峰值只增加到约 84.3 GiB，96GB 卡仍留 11.3 GiB，不存在用错误 BF16
契约省显存的理由。

## 4.16 SGLang-compatible FP32 GDN verify 与全 M=4 默认（2026-08-28）

直接复用 b12x multistep 不可行：它会明确拒绝 FP32 persisted state；其 BF16
逐步 round-trip 正是旧 Qwen3.6 bit-path 的一部分。于是从 SGLang
`fused_sigmoid_gating_recurrent.py` 提取固定范围，在
`runtime/kernels/flashnext_gdn_verify.py` 实现 SM120 所需子集：线性 K=3 draft、
GQA、一个 kernel 内 FP32 recurrence、每位置 FP32 candidate state。移植时修正了
SGLang state pool `[V,K]` 与本 runtime `[K,V]` 的物理布局差异。独立 SM120
测试以 4 token、2→6 GQA heads、128×128 state 对逐步 FP32 reference 验证，
所有 candidate state 通过 `rtol=atol=2e-5`，BF16 output 通过 2e-2 门。

只替换 GDN、其余仍用旧逐行浮点路径是错误混搭：verify 76.9→71.7 ms/轮，
但出现 5 个 greedy mismatch，steady 只有 27.09 tok/s，已否决。完整采用
SGLang-compatible M=4 target 数值路径后，128 轮长跑为：

- 322 committed token，全部 logits/state finite，持续生成正常技术推理；
- acceptance **194/384 = 50.5%**，平均提交 **2.516 token/轮**；
- verify **6.226/128 = 48.64 ms/轮**，decode **9.350 s = 34.44 tok/s**；
- 峰值 **84.27 GiB**，空闲 **11.32 GiB**；
- 相对旧 exact 28.91 tok/s 提升 **19.1%**，相对同请求 SGLang 35.87 tok/s
  只差 **4.0%**。

旧串行 target 回放与这一批量数值路径 top-1 为 90.7%，不能再被解释成“batch
必须 bit-identical”：SGLang 自己使用的就是批量 CUTLASS/FP32 fused recurrence，
而非本 runtime 的逐行 FLA/b12x 浮点顺序。逐行路径保留为显式诊断兼容模式；
生产 FlashNext engine 默认选择已长跑验证的全 M=4 + FP32 candidate state。
剩余最大差距已经转为 TTFT（约 2.44 s vs SGLang 0.475 s）。

## 4.17 large-M target prefill 与端到端追平（2026-08-28）

旧 TTFT 2.44 s 的根因是把 109-token prompt 当成 109 次 M=1 CUDA Graph
decode。`prefill_session()` 现在一次运行整段 prompt，同时写回 graph decode 共用的
GDN、QSA、PLE 状态，并返回 initial NEXTN sync 所需的全部 HC hidden rows。
顺序 chunk sweep 曾给出 0.172 s，但该数字继承了前序 shape 的 PLE/OS cache 和
首次 kernel 开销，不能作为独立冷样本；随后用新进程只运行 whole-prompt batch：

- prefill **0.390 s**，TTFT **0.408 s**，initial MTP 0.039 s；
- 128 轮提交 319 token，acceptance **191/384 = 49.7%**，平均提交
  2.492 token/轮；
- verify 6.194 s、MTP 2.947 s、PLE 0.523 s，decode **34.42 tok/s**；
- 请求总时长 **9.714 s / 320 completion token = 32.94 tok/s**；
- 全量 verify logits/state finite，输出持续为正常 TCP 技术推理，峰值
  **84.14 GiB**，仍留 11.45 GiB。

这使 TTFT 从 2.440 降到 0.408 s（**-83.3%，5.98×**），相同 prompt 下首次
低于 SGLang 的 0.475 s；steady decode 与 SGLang 35.87 tok/s 相差 **4.0%**。

批量 prefill 另做了显式串行交叉门禁。两者首个 greedy token 都为 1596，但
并非逐位等价：final logits cosine 0.944、HC hidden cosine 0.966、最差 GDN
state cosine 0.947。该差异来自 large-M 与 M=1 的不同 GEMM/recurrent 数值顺序；
SGLang 生产路径本身也采用 large-M prefill。因为新路径的 128 轮 acceptance
49.7% 与 SGLang 48.7% 接近、长跑 finite 且输出正常，性能驱动脚本默认启用
whole-prompt batch prefill；`FN_BATCH_PREFILL=0` 保留串行诊断回退。这个门禁只证明
当前固定提示的稳定性，不替代后续正式质量集与 server/backend 接入验证。

## 4.18 新默认路径 kernel ledger（2026-08-28）

对 large-M prefill + 全 M=4 verify 默认路径再次运行 `torch.profiler`，产物在
`/tmp/flashnext-fast-profile-20260828/`。Profiler 本身把 wall time 严重放大，
所以只使用 CUDA kernel 自耗时与调用数做归因，不拿被插桩后的 2.35 tok/s 当性能。

8 个 speculative round 的 CUDA kernel 自耗时合计 **428.574 ms（53.57
ms/轮）**。其中：

- 两类最高 elementwise kernel 合计 **149.415 ms（18.68 ms/轮，34.9%）**；
- b12x target routed MoE **41.609 ms（5.20 ms/轮，384 次 = 48 层 × 8）**；
- `aten::mm` **35.432 ms**，但有 **25,064 次（3,133 次/轮）**，说明剩余热点
  是 HC/投影/MTP 的大量小 GEMM 与 elementwise 碎片；
- 新 FP32 GDN verify kernel 仅 **3.075 ms（0.384 ms/轮）**，已不是优化对象。

Target prefill 的 GPU 自耗时合计约 68.7 ms，其中 b12x large-M MoE
31.386 ms（45.7%）、GDN causal-conv 11.893 ms（17.3%）、`aten::mm`
9.605 ms（14.0%）；非 profiler 新进程 wall TTFT 已为 0.408 s。由此下一阶段
优先级明确为减少/融合 HC 与小投影算子、给 MTP sync/continuation 建立图安全的
固定 metadata ABI；继续优化 GDN 或 PLE 不会填平剩余 4% steady gap。

## 4.19 23K 长提示、EOS 口径与 prefill shape 收敛（2026-08-29）

使用与 SGLang 长上下文记录完全相同的 `sglang-long` chat prompt（23,473
tokens）、`max_seq=32768`、K=3、token-major chunk=1024、verify CUDA Graph
开启、prefill MLP Graph 关闭。当前安全路径的新进程结果为：

- target prefill **5.734 s / 4093.54 tok/s**，TTFT 5.771 s；另一新进程样本为
  5.265 s / 4458.55 tok/s，说明 OS page cache、JIT 和权重加载热度仍会造成明显
  跨进程波动，不能把两者差值归因给单一代码改动；
- 首次 EOS 位于第 32 个 speculative round，EOS 前 108 completion tokens，
  acceptance **76/96 = 79.2%**，逐位置 **90.6% / 75.0% / 71.9%**；
- EOS 前 verify-vs-ordinary target 为 top-1 **98.1%**、最低 logits cosine
  **0.931453**，107 行中 2 个 greedy mismatch；完整输出仍能结束在正确答案；
- EOS 后强行继续到固定 64 轮会把 acceptance 拉低到 64.1%，因此固定轮数
  aggregate 不再作为用户可见质量或接受率结论。`fn6` 已同时输出
  `request_to_eos` 和 `quality_to_eos`。

同进程 ABAB prefill 对照确认 chunk=1024 相对 chunk=512 为
**4489.13 vs 2909.85 tok/s（1.543×）**。但两种 large-M shape 的 final logits
cosine 为 0.867358、最差 state cosine 为 0.780731；把 routed MoE 单独切回
512-row 分块后漂移完全不变，说明差异不是 routed MoE 单点造成。layer-major
虽然相对 token-major 512 再快 18.4%，但 logits cosine 仅 0.763382，故两条更
激进路径都不进入隐式默认值；生产默认保持经过长提示 EOS/finite 门验证的
token-major 1024。

b12x stable expert rank 的 batched fastpath 则通过同进程 sort/batched ABAB：
target-only prefill **3051→3402 tok/s（+11.5%）**，logits/hidden 逐位相同，已
保留。相反，尝试把整条 prompt 的 PLE 请求跨 chunk 提前排队没有得到可重复
收益：一次受控冷进程 sweep 中 prefill 4.790→5.297 s、请求 8.539→9.286 s。
该 sweep harness 没有生成 bfdiag run record，无法用 `bf diff` 形成正式归因，
所以跨-chunk ahead-prefetch 代码已删除；每个 chunk 内原有的异步 PLE prefetch
继续保留。

`fn6` 现在额外打印 `benchmark_config`，明确记录 prefill shape、verify/MTP
graph、GDN batch、MoE backend、PLE mode 和 stable-rank 开关，防止再次把不同
配置或 EOS 后续跑的数据当作同口径 A/B。随后同配置 32-worker 冷进程又测得
6.992 s / **3357 tok/s**（EOS 前 acceptance 77.1%、quality top-1 99.1%），
说明 23K prefill 的可复现区间目前只能写成约 **3.36–4.46K tok/s**，不能再把
4.1–4.5K 写成稳定下界。把 PLE workers 从 32 增到 128 的 prefill-only 样本更慢：
8.614 s / **2725 tok/s**，虽然只产生 1627 个 page miss；Python 线程争用没有
形成 SGLang io_uring 的高效 QD，故默认仍保持 32。

当前证据能支持“23K prefill 已进入或高于所记录的 SGLang 2–4K 区间”，但不能
支持“prefill 稳定 2×”或“整段 decode 已比 SGLang 快 2×”。同提示短请求的
稳态 decode 仍是 34.44 vs 35.87 tok/s，下一阶段必须集中在 HC/MTP 小 GEMM
与长上下文 sparse MTP 图化。全 PLE 表 resident 需要约 51 GiB pinned host RAM，
本机只有 23 GiB，不能用这台机器做 resident 实测；stream 路径仍是正式约束。

现成 b12x BF16 small-M GEMV 也按真实 shared-expert M=4 形状做了 graph replay
微基准：gate/up `1280x2560` 为 7.23→8.02 us（0.90×），down `2560x640`
为 4.30→10.74 us（0.40×），只有标量 gate `1x2560` 为 2.67→2.28 us
（1.17×）。因此不能把已有 small-N GEMV 泛化替换 shared expert；总体收益不足
一次 launch，且两个主矩阵都会退化，该方向已拒绝。

## 4.20 冷/热 prefill 归因与长上下文 sparse MTP Graph（2026-08-29）

同一个 23,473-token `sglang-long` prompt、同一进程、同一 token-major
chunk=1024 做一次 warmup 后再正式计时，第一次为 **7.396 s / 3173.67
tok/s**，第二次为 **4.719 s / 4974.51 tok/s**，即稳态比首个大-M
调用快 **56.7%**。另一个相同 warmup 进程为 5.523→4.576 s（最终
**5129.17 tok/s**）。因此之前跨进程 3.1–4.5K 的“波动”主要混入了首次
large-M kernel/JIT/autotune 成本；今后必须明确区分 cold first-request 和
shape-warmed steady prefill，不能再把两者写成同一条性能线。

PLE 也做了独立拆分。完整 prompt 的 375,568 个 lookup 只有 1,568 个唯一 row、
1,627 个唯一 4-KiB page；32-worker stream 的 CPU hash + gather 总计约
**0.197 s**。同一批冷页的现有 `pread` 获取为 91.3 ms，本地 SGLang Rust
io_uring reader（QD=512）为 17.4 ms，页读取快 5.26×，但绝对只省约 74 ms。
这证明 io_uring 是可用的后续优化，却解释不了第一次与稳态之间 2.7 s 的差距；
`PLE resident` 也不可能把 target prefill 从 3.1K 直接变成 5K，当前优先级应低于
large-M warmup 和 MTP/HC 路径。

长上下文 MTP 新增了显式 opt-in 的 fixed-shape sparse graph 原型
`FN_MTP_SPARSE_GRAPH=1`，默认仍关闭。它把 proposal/continuation 的 graph
capacity 从 dense-QSA 的 2048 扩到 `max_seq`，并使用固定 8192-group score、
512-block select 和 2051-token gather buffer。真实 23K/K=3 capture/replay 成功，
显存从旧图的约 81.95 GiB 增到 81.99 GiB。相同 warmup、32 轮对照为：

- eager MTP：acceptance **77/96 = 80.2%**，MTP 0.675 s，decode
  1.831 s / 59.52 committed tok/s；
- 第一版 sparse graph：acceptance **64/96 = 66.7%**，MTP 0.233 s，decode
  1.275 s / 75.30 committed tok/s。

图化把 MTP 时间降低 65.5%，并把每轮 wall 从 57.2 ms 降到 39.8 ms，但第一版
错误地让每个 continuation draft 重新 score/select block；生产 eager 语义是复用
teacher 最后一行的 sparse indices 并只追加 causal tail。因此该原型虽然更快、
target quality top-1 仍为 97.9%，acceptance 却明显改变，当前不能默认启用。
下一步必须把 teacher indices/captured length 放进静态 buffer，让 graph continuation
与 eager reuse 逐项等价，再重跑同 prompt acceptance/状态门。

复用语义修正后的同配置真机复测（同一 `sglang-long` prompt、K=3、32 轮）为：

- target prefill **4.255 s / 5516.77 tok/s**；此前同进程 sweep 已把相同
  large-M shape 预热，因此这是 shape-warmed 数字，不代表 cold first-request；
- sparse graph MTP **0.223 s**，decode **1.268 s**，提交 105 token，
  **82.82 committed tok/s**；相对 eager 的 59.52 tok/s 提升 **39.1%**；
- acceptance **73/96 = 76.0%**，逐位置 **90.6% / 68.8% / 68.8%**，
  已从错误版的 66.7% 明显恢复，但仍比 eager 的 80.2% 少 4 个 accepted token；
- verify quality top-1 **97.1%**、最低 logits cosine **0.933234**，全部输出 finite；
  capture 后显存约 81.86 GiB，完整请求峰值占满 allocator 可见余量但进程正常结束。

剩余 4-token acceptance 差异不再来自 sparse indices/tail 语义：fixed score、select、
gather、reuse 的回归测试均与 dynamic reference 逐项相等。真实 MTP 权重的独立对照
显示 graph-safe Triton expert matvec 与 eager grouped-GEMM 的 cosine 约 0.99999，
但逐行相对误差最高约 0.5%；三步 recurrent draft 会把少量边界 logits 放大成 token
分叉。因此 sparse graph 继续保持显式 opt-in，不能仅凭单提示把它改成默认路径。
下一门禁是让 graph/eager 使用同一数值合同或扩大质量集验证，而不是再次改 sparse
索引语义来追 acceptance。

## 4.21 MTP 批量归约、HC 确定性与最终 SGLang 对照（2026-08-29）

前三条优化线已分别完成实现或否决门禁：

1. target routed MoE 的现有 b12x M=4 已是正确批量边界。真实形状微基准为
   **0.10579 ms**，四次 M=1 合计 **0.51075 ms**，即 **4.83×**，且逐 bit
   相同；动态 route 替代为 0.14675 ms 且数值更差，因此没有为了“看起来做了
   fusion”保留退化代码。
2. 长 prompt 初始 MTP sync 的 grouped expert 输出不再恢复成 `[T,K,H]` 再整体
   转 FP32；新 Triton reducer 直接按 inverse route 以固定 K 顺序归约到 `[T,H]`。
   在 `23473×10×2560` 上，10.769→0.890 ms（**12.10×**），临时峰值
   5.596 GiB→0.114 GiB，cosine 0.99999994。MTP graph-direct 同时从逐行 launch
   改成 M≤4 一次批量 kernel，0.464→0.294 ms（**1.58×**），输出逐 bit 相同。
3. sparse proposal/continuation graph 的 fixed score/select/gather/reuse 语义已修正，
   cache capacity 和 graph→eager captured-length 也有回归门。它仍保持显式 opt-in，
   因为短 TCP 对 ordinary target 的 top-1 仍为 97.6%，未达到 98% 默认门。

重复冷进程还暴露了一个此前被 acceptance 波动掩盖的 HC 根因：persistent mix 的
down 投影把 40 个 K tile 用 FP32 `atomic_add` 汇到同一位置，CTA 调度顺序改变时，
边界 draft logits 会跨 token。旧原子版的两次相同 TCP 运行 acceptance 为
65.4%/66.7%，但 top-1 94.8%→97.4%、最低 cosine 0.863→0.956。现在每个 K tile
写独立 partial，再按固定升序归约；barrier counter 也改成每 CUDA stream 独占。
M=4 对逐行 HC 输出已通过逐 bit 相同门，两个独立冷进程的 acceptance、质量指标和
mismatch 位置全部完全一致。

严格使用同一个 109-token TCP prompt、K=3、26 verify rounds 的最终两次记录为
`26bf63e9f38c` 和 `e4d1f2ee6937`，`bf diff` 判定 comparability-critical 字段
全部一致：

- 两次均为 86 completion token，accepted **59/78 = 75.6%**，top-1
  **97.6%**，最低 logits cosine **0.971556**，相同的 2 个 greedy mismatch；
- decode-only 为 **99.60–102.48 committed tok/s**；把 initial MTP 也计入首
  token 后用户可见区间为 **94.65–97.70 tok/s**；
- 对同请求 SGLang 的 **35.87 tok/s**，严格首-token-after 口径为
  **2.64–2.72×**；第二次 TTFT **0.499 s**，SGLang 为 0.475 s；
- 第二次完整请求 1.370 s / 86 token；SGLang 同 26 rounds 为 2.260 s /
  64 token。输出长度不同，所以不把两者总时长比伪装成同 token-count latency。

最终当前代码的 23,473-token/32-round 记录为 `4f84debcbad7`：cold prefill
**5.547 s / 4231.44 tok/s**，decode **1.112 s / 93.54 committed tok/s**，
acceptance **72/96 = 75.0%**，verify top-1 **100%**、零 greedy mismatch。
相对用户给出的 SGLang 约 2K prefill 是 **2.12×**；首 token 后连同 initial MTP
为 63.49 tok/s。此前同 prompt eager MTP 的 59.52 committed tok/s 对照下，
当前图路径是 **1.57×**，尚不能把长 prompt 同路径写成 2×。

最后，长 prompt initial sync 的一次性 allocator workspace 会把 reserved 从约
79.6 GiB 留在 90.4 GiB。teacher sync 结束后执行一次 `empty_cache()`，正式
`bf diff`（`a0e2b78be9e0`→`13193428a1f7`）显示请求结束 driver used
**93.06→82.25 GiB**、free **2.54→13.34 GiB**，回收 **10.81 GiB**；最终
32-round 请求结束仍有 **13.21 GiB** 空闲，QA 完成后有 10.43 GiB。回收只发生
在初始 MTP 后一次，图 replay 的固定地址不受影响。

## 4.22 M=4 MoE token 分区与 shared-slot 否决（2026-08-30）

对当前 target verify 的真实 routed-MoE shape（`M=4, H=2560, I=640,
E=512, top_k=10`），b12x micro kernel 的 160 个 CTA 原先按全局 task id
跨 token 轮转，导致同一个 CTA 的后继 FC1 task 通常换到另一个 token，不能复用
已量化输入。干净隔离 worktree
`/tmp/sparkinfer-flashnext-m4-clean-20260830` 新增精确 shape gate，把 160 CTA
分成每 token 40 CTA；每个 CTA 只处理该 token 的 2–3 个 FC1 task，FC2 调度不变。

256 MiB L2 冷缓存的 shape-only CUDA Graph 基准为 **108.5→106.1 us
（+2.3%）**。`40/80/120/160` cooperative-grid sweep 仍由 160 胜出。GPU
oracle、env off/on 和 graph replay 均通过逐 bit 门；短 TCP 26 rounds 的接受率、
top-1、logits/state 和 mismatch 位置全部不变。更长的同 prompt 64-round 正式 A/B
记录 `ea784c51a687`（off）与 `52f9aa30b781`（on）由 `bf diff` 判定可比：

- accepted 均为 **103/192 = 53.65%**，top-1 均为 **94.61%**，9 个 mismatch
  完全一致；
- decode **2.1872→2.1318 s**，committed throughput **76.35→78.34 tok/s
  （+2.6%）**；
- 显存账逐项不变。

候选因此只对上述精确 shape 默认启用，并保留
`B12X_MICRO_M4_TOKEN_PARTITION=0` 回退；它是稳定的小幅收益，不是两位数主线。
正式运行验证另用 `/tmp/sparkinfer-flashnext-integrated-20260830` 叠加 SparkInfer
主树现有未提交依赖补丁，避免直接污染主工作树。

同时验证了 SGLang 风格的 shared-expert slot：把 BF16 shared expert 离线量化为
第 513 个 NVFP4 expert 后，真实 layer-0、M=4 的完整 MLP cosine 只有
**0.95446**、relative norm error **0.29847**、max abs **1.11914**；shared-only
cosine 更只有 **0.65012**。误差来自 BF16→NVFP4 本身，不是 append route plumbing。
该路径已完整撤回，没有留下生产 flag 或未使用接口。

## 4.23 MTP 数值 tile 与 HC norm+mix 融合否决（2026-08-30）

真实 MTP 权重的 teacher+12-step 对拍把 graph-direct 与 eager grouped-GEMM 的
第一个分叉定位到 routed expert MLP：teacher 输入的 pre-MLP tensor 逐 bit 相同，
但默认 gate/up `block_k=64` 的 MLP 输出 max abs 已为 **0.001953125**；下一步
递归后 mixed max abs 放大到 **0.28125**。continuation CUDA Graph replay 与同一
dense-direct 路径的 8-step token 序列完全一致，因此 graph capture/replay 本身
不是新增误差源。

`block_k=32/64/128/256` sweep 中，128 把 teacher MLP max abs 降到
**0.00048828125**，rows=3 热 kernel 延迟也为 215.06→198.45 us；但真实严格
TCP A/B 否决了这个局部代理指标。记录 `cdc24939a15b`（64）与
`3d785dc7a617`（128）仅改变该 tile（另有 GPU 时钟指纹差异），`bf diff`
判定 comparability-critical 字段一致：

- accepted 均为 **59/78 = 75.64%**，top-1 均为 **97.65%**；
- 最低 logits/state cosine、两个 greedy mismatch 及其位置全部相同；
- committed throughput **100.82→100.28 tok/s**，没有性能收益。

所以 128 只是把合成递归漂移推迟两步，没有移动真实 speculative 接受边界。
实验 env seam 和依赖本地 checkpoint 的重回归均已删除，生产继续固定 64。

HC grouped Gemma RMSNorm→persistent mix 也做了单-kernel 候选。rows=1/4 的
mixed 与 normed 可以逐 bit 对齐，但 graph replay 分别从 16.49→25.17 us、
14.63→22.83 us，退化 **34.5%/35.9%**；更少 CTA 的版本退化超过 60%。固定
归约需要的 barrier/scratch 成本高于省掉一个 graph node，候选已完整撤回。

剩余最大路径因此不是继续扫 Triton tile，而是让 MTP capture/eager 复用同一个
BF16 MoE 数值合同。本地 b12x 没有 BF16 routed-MoE；PyTorch `F.grouped_mm`
public API 也没有 `out=`/静态 workspace，fallback 还会读取 CPU offsets。可行的
下一步是从本地 PyTorch `GroupMM.cu` fast path 外提薄 custom op，暴露静态
output/workspace/device pointer metadata，并复用同一 CUTLASS grouped-GEMM
实现；这比另写一套近似 Triton matvec 更有机会同时保住 graph 性能与 eager 数值。

## 4.24 接受率口径纠正、逐轮 trace 与 GroupMM 否决（2026-08-30）

4.23 末尾把 MTP eager/graph 数值合同列为剩余最大路径，是基于不可比记录和局部
数值代理做出的过强结论，本节用严格同 prompt A/B 取代它。

首先，历史 **80.2%** 记录 `6c63461174c2` 是 23,473-token `sglang-long`、
32 rounds、`max_model_len=32768`；当前 **75.6%** 记录是 109-token TCP、
26 rounds、`max_model_len=4096`。prompt hash、长度、rounds 和 prefill 配置均不同，
不能把二者之差写成接受率回退。同一 TCP prompt 的多次当前记录都稳定为
**59/78 = 75.641%**。

为了不再靠总数猜测，`fn6_spec_batch.py` 现在把每轮 draft、target prediction、
reject position、teacher suffix、commit、下一轮 draft、target/MTP position 和分段
时间写入 `fn6_rounds.jsonl`，并把同一轮的标量摘要接入 `bf trace show`。正式记录
`b48ac7505b05` 的 histogram 为：full accept 17 轮，reject@0 4 轮，reject@1 2 轮，
reject@2 3 轮；前 12 轮连续 full accept，拒绝集中在后半段生成内容。
所有 26 轮均满足 `target_pos_after == mtp_sync_len_after`，没有 rollback、teacher shift
或 position off-by-one 证据。

两组有意只改变图开关的反证进一步锁定了接受率含义。`bf diff` 会正确地因为
`cuda_graph_status` 不同而标为 NOT COMPARABLE，因此性能只作为单变量实验观察，
而质量结论另由逐轮 artifact 的结构字段全等证明：

- target verify CG：`b48ac7505b05`（on）与 `15945a090ef5`（off）的 26 轮
  draft/prediction/reject/commit 全等，均 accepted 59、top-1 97.647%、相同两个
  greedy mismatch；CG 把 decode **2.7853→0.8593 s**、committed throughput
  **30.52→98.92 tok/s（3.24×）**；
- MTP proposal/continuation sparse graph：`b48ac7505b05`（on）与
  `6a82a0922631`（off）的接受率和全部质量指标也完全一致；图路径把 decode
  **17.732→0.859 s（20.6×）**。因此当前 TCP 的 75.6% 是这个生成区段的真实
  draft 命中率，不是 CG 或批量接受逻辑造成的下降。

同时完成了 4.23 所提 GroupMM 方向的执行门。SM120 上实际 profile 的
`torch.nn.functional.grouped_mm` 不是一个可直接外提的 CUTLASS grouped kernel：
它先做一次 device-to-host offsets copy，再按活动 group 发多个 cuBLAS GEMV kernel，
这正是 public API 不能 capture 的原因。独立 SM90/SM100 CUTLASS route-as-group
候选虽然能编译，`gemm.initialize()` 均返回 `Error Internal`；selected-weight
graph-safe 路径的真实 TCP 记录 `5b8a3ccfcff7` 则与基线接受率/质量逐项相同，
但增加约 1 GiB graph reserved 且没有性能收益。原型、env seam 和生成库已删除。

所以当前最大空间转回 target verify 的真实 wall 热点及 PLE stream I/O；本机只有
23 GiB RAM，47.7 GiB 表不能 resident，必须在 stream 约束下优化，而不是把
不可用的全内存配置当成生产结论。

逐行数学 oracle `dc4fadb9aa1a` 也给出了明确的质量/性能边界：同一 TCP 配置把
target verify 的外围 M=4 运算恢复为 M=1 后，accepted 从 **59→64/78
（75.6%→82.1%）**，但生成路径随之改变，仍有 2 个相对普通 decode 的 greedy
mismatch；decode 从 **0.859→1.645 s**，throughput 从 **98.92→54.71 tok/s**。
它证明批量浮点合同会影响 draft 边界，但不是可直接默认的优化：生产 M=4 路径在
同 prompt 已是 SGLang 35.87 tok/s 的 **2.76×**，逐行 oracle 只有 **1.53×**。
后续若追更高接受率，必须做组件级质量/性能 Pareto，而不能把整个 exact 模式误称为
“修复”。

## 4.25 PLE io_uring/O_DIRECT 落地与 shared-expert 小 M 否决（2026-08-30）

在 4.24 确认 PLE stream I/O 是剩余最大 CPU 路径后，`FlashNextPleTable` 接入
SGLang 已编译 Rust storage extension 的 `IoUringReader`。生产默认 `auto`，可用时
走 `io_uring`，不可用时回退原 `pread`；同时保留 `QSR_FLASHNEXT_PLE_IO=pread`
和 `io_uring` 的显式 A/B 开关。实现不 import SGLang Python 包，而是直接解析并
加载 `_storage*.so`，避免 Transformers/cache patch 和错误版本污染
`sys.modules`。io_uring 使用独立 `O_DIRECT` fd，普通 mmap/pread fd 保持不变；
close 现在与 gather 串行、幂等，并释放 direct fd、resident pinned table 和缓存。

严格同 TCP/K=3/26 rounds、同图与稳定路由配置的正式 `bf diff`：

- `pread`：`86869b6d1c7b`，accepted **59/78 = 75.641%**，decode
  **0.8524 s**，**99.72 committed tok/s**，prefill **0.4493 s / 242.57 tok/s**；
- production `auto`（本机选择 io_uring/O_DIRECT）：`aabb9fa41094`，accepted
  仍为 **59/78**，decode **0.7988 s**，**106.41 committed tok/s（+6.7%）**，
  prefill **0.3427 s / 318.07 tok/s（+31.1%）**；
- PLE decode 累计段 **79→27 ms（2.93×）**，request total
  **1.4077→1.2436 s（-11.7%）**；26 轮逐轮 draft/prediction/reject/commit
  artifact 完全一致，top-1、logits/state、greedy mismatch 和显存账逐项不变。
  两个 record 的 `ple_source_sha256` 均为 `8f752a1bb1fd...`，避免 untracked
  FlashNext 文件只显示同一个 dirty git SHA 而掩盖源码变化。

同一源码 SHA 的 23,473-token 最终记录 `000e350ac02a` 给出 prefill
**4.097 s / 5729.86 tok/s**，是用户给出的 SGLang 约 2K tok/s 的 **2.86×**；
32-round decode 为 **95.46 committed tok/s**，是同 prompt SGLang
**35.87 tok/s 的 2.66×**。该生成段 accepted **81/96 = 84.38%**；到 EOS
口径 accepted **79/96 = 82.29%**、verify top-1 **98.2%**。长请求图/allocator
峰值仍接近整卡（peak reserved 96.72 GiB），请求阶段最终 driver free 11.73 GiB，
质量 replay 结束后 free 0.50 GiB；因此不要在这条验证旁并发第二个 GPU 任务。

同一轮还验证了 target shared-expert 的 rows<=4 Triton 两-kernel 候选。局部
shared-expert 微基准为 rows=1 **92.48→24.76 us（3.74×）**、rows=4
**103.90→33.02 us（3.15×）**，但完整 TCP 把接受率从 **59→49/78**，decode
**0.856→0.886 s**，committed throughput **99.29→84.61 tok/s**，质量和性能都
退化。因此候选 kernel、默认分支和测试已全部删除；这再次说明 recurrent target
路径不能用局部 cosine/max-abs 门代替端到端接受率门。

更保守的“把 scalar gate 一行拼到原 BF16 gate/up GEMM”也未保留：rows=1
虽约 **1.28×**，但生产 verify 的 rows=4 只有 **0.67×**，cuBLAS 在 N=1281
选择的算法反而更慢。因为主路径固定 M=4，实验 flag、加载分支和测试均已删除。
