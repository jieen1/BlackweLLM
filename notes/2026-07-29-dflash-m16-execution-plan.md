# DFlash M=16 执行计划（Sol 审查后）

## 目标与唯一口径

- 目标：在 vLLM quick-brown 64K 合同上取得至少 `330 completion/output tok/s`，并保证输出质量与 DFlash 的实际价值。
- vLLM 参考：prompt hash `501c19f22bb6244b2d008ffab1e05ad53406c4f587ba4c6c097b4871a283af85`、64K、K=15、greedy、256 completion tokens、prefix cache、CUDA Graph；warm output tok/s 为 `267.7/277.1/293.1`，接受率 `0.6869565`。
- 当前正式 runtime 基线是 `8e9255937cae` 的 canonical 合同；`b836d97a54a9` 的 `321–327 tok/s`仅是“当前路径可恢复到 320+”的证据，不能与 vLLM quick-brown 作端到端胜负对照。
- 任何数值结论前必须先跑 `bf diff`；不同 prompt hash、load-time 几何、runtime/SparkInfer SHA 的记录不作性能结论。

## 已证伪或禁止重试的方向

1. 不把 p128 作为现阶段优化：虽可局部修复 CUDA Graph parity，但完整 E2E 明显不可用。
2. 不关闭 `SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT` 求速：它同时承载 dynamic-down-scale 的稳定物理路由行正确性，不能作为性能开关。
3. 不把 M=16 导入现有 W4A16 hybrid direct-topk kernel：它只支持小 M、使用 cooperative barrier，历史试验已显示死锁/正确性风险。
4. 不在没有同合同 trace 证据时修改 SparkInfer 主 MoE GEMM；它不是当前优先级最高的可证伪瓶颈。

## 连续执行步骤

### 1. 建立正式 runtime + DFlash quick-brown server-step 基线

使用已有的 `bfdiag.workloads.run_historical_dflash_m16`：固定 quick-brown 64K、K=15、greedy、256 completion tokens，计时生产 `dflash_prefill_bootstrap + dflash_round`。这是当前实际 server M=16 路径，不走已阻塞的 exact-prefix continuation；pre-fill 只用于建立每轮相同的 prompt 边界，不计入 steady-state decode。

门禁：prompt hash 必须为 `501c19…af85`；block size `64`、blocks/slot `1536`、max model len `262144`；记录 `tok_s`、平均 round 时间、接受率、输出 hash。该记录是与 no-DFlash lane 的正式同路径对照，不直接宣称与 vLLM global-prefix E2E 等价。

已完成（2026-07-29）：record `95f455b5167a`，三轮均为 `76.96 output tok/s`、`42.40 ms/round`、接受率 `15.30%`，234/234 轮命中 M=16 CUDA Graph。`bf diff 95f455b5167a b836d97a54a9` 明确拒绝把它与 320+ 记录作结论（prompt hash 与 prefix-cache 合同不同）。这不是 kernel step 速度退化：trace 显示 90/234 轮在第一个 draft token 即拒绝。

同 prompt、同几何、同 Git SHA 的历史 profile `f2cdb7dd71cb` 记录为 warmup 24 轮生成 276 token、后续 8 轮接受率 `62.5%`；当前常驻服务复跑为 `cdab0f6d7586`，仅生成 127 token、后续接受率 `5.83%`。`bf diff cdab0f6d7586 f2cdb7dd71cb` 判定合同可比。两个 record 的 Git SHA 相同但没有捕获脏工作树内容；当前 `runtime/backends/laguna_dflash.py` 在低接受率 run 前数分钟被修改。因此从现在起，dirty runtime source 是必须逐层定位的回归面，不能再把 SHA 相同误读为代码相同。

trace 的首个确定性分叉边界：每次 reset 后，前 8 个 DFlash round 接受率为 `72.5%`（95 个 completion token）；第 9--16 round 已降至 `8.33%`，之后持续低位。该形状将用于最小状态对拍，而非重新跑吞吐扫描。

### 1a. full-prefix 功能阻塞（与性能基线分离）

已复现：slot-local draft ring 容量 `640`、window `512`，256-token 回复后回退 prompt 边界需回退 `255`，超过 `128` 个 spare positions，必然降级为 MISS。即使从 `dflash_prefill_bootstrap` 直接形成完整 hit，`continue_prefill_with_aux` 的 eager M=1 continuation 在 64K 状态卡住；尝试既有 M=1 CUDA Graph 也在 DFlash M=16 graphs 已存在时卡在 capture。两条失败均不产生可比性能 record，已撤回临时代码。

因此在新 Sol 审查给出独立的 prefix-state 方案前，不把 full-prefix 加入性能胜负结论；该问题的日志证据保留在 `.bfdiag/failed-vllm-quickbrown-full-hit-20260729.log`。

### 2. 建立同合同 runtime no-DFlash lane

复用固定 quick-brown token IDs、同一 prefix-cache 热态、同一 completion 语义，测主模型非投机 decode 的 output tok/s 和输出质量。

门禁：先与步骤 1 的 record 执行 `bf diff`，确认只差 `dflash` 这一目标变量。若 DFlash 没有实质超过 no-DFlash，停止任何“DFlash 性能获胜”叙事，转入接受率/状态根因。

### 3. 同合同 phase trace 与归因

用现有 `diagnose_dflash_m16_round_timing_gap` 及 daemon-side `QSR_TRACE=1` 记录同 prompt 的 round 时间、CG 命中、reject-position、verify/draft/commit 阶段。

门禁：先读已产生 trace；只在 host gap 或 metadata/copy 明确占优时优化对应路径。若 GPU 仍为主导，进入步骤 4。

当前归因结果：host/CG 不是首要解释（M=16 CG hit 100%、round p50 `42.10 ms`）；优先执行 captured/eager 和状态对拍。首次执行已有 `diagnose_dflash_verify_cg_divergence(prefix_steps=8)` 时，eager 分支在第二个 64K state construction 后卡死，未产生 record，诊断服务已正常终止；该卡死不得计作 parity 通过或失败。下一次诊断必须复用首分叉状态、避免双重 64K prefill，再检查 verify logits/aux/KV 的首个差异。

### 4. 按证据选择一个优化方向并完整验证

- 若 trace 指向接受率/状态：先做 captured/eager parity 与逐层 divergence，定位 first divergent layer/operator；只修复已定位状态、KV、aux 或数值路径。
- 若 trace 指向 host/metadata：只改该数据物化或 copy 边界。
- 只有接受率接近参考且 GPU step 仍是主要差额时，才评估 SparkInfer kernel 路径。

每个改动必须依次通过：相关 CPU/unit tests → 最小 GPU parity/graph replay → 固定 quick-brown 合同 → `bf diff` → 输出 hash/质量/接受率门禁。每次只改一个可归因边界；失败立即回退该方向并记录证伪 run id。

### 5. 终止条件与下一轮

当步骤 4 的候选均被证伪、收益上限不足以达成目标、或 trace 没有给出可验证新假设时：停止代码试验，自动安排独立 Sol agent 做新的只读审查；将其计划追加到本文件并从步骤 1 的相同门禁重新执行。禁止无计划参数扫描或跨合同对比。
