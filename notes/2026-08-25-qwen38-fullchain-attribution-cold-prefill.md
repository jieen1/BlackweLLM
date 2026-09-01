# Qwen3.8 DFlash2 128K 全链路归因收口：prefill 账本与剩余杠杆（2026-08-25 晚）

状态：🟢 **归因完成，无新默认路径改动。** 本轮把 2026-08-25 早间 note 留下的
"下一轮做 verify-attention"问题升级成了一次全链路（cold prefill + warm decode）
的 GPU 级归因，并证伪/确认了几个候选方向。所有测量在 RTX PRO 6000 Blackwell
SM120、torch nightly（2.15.0.dev20260815+cu134）、flashinfer 0.6.16.post3、
Gittensor `Qwen3.8-27B-NVFP4-RTX5090-gittensor` + DFlash2 K=7 + FP8 KV +
CUDA Graph + persistent prefix cache 的 HEAD（`e9b001b`）上完成。

## 基线复现（本轮自己的 fresh-process 口径）

`benchmarks/server_perf_grid.py`，128K、c=4、max_tokens=256、warm=2，
隔离端口 18511：

| 波次 | wall | mean TTFT | decode/req | aggregate E2E |
|---|---:|---:|---:|---:|
| COLD | 39.97 s | 38.09 s | 145.8 tok/s | 25.6 tok/s |
| WARM1 | 1.94 s | 0.18 s | 153.9 tok/s | 526.9 tok/s |
| WARM2 | 2.00 s | 0.17 s | 151.4 tok/s | 511.6 tok/s |

与早间 note 的 169/585 相比低约 4–8%，属于时钟/口径差异（warm_rounds 不同）；
后续 A/B 必须以同进程重跑的 control 为准。

## Cold prefill 的 GPU 账本（nsys，37.7 s GPU busy / 96%+ busy）

来源：`/tmp/qwen38_cold_prefill_trace.nsys-rep`（sqlite 导出在
`/tmp/opencode/qwen38_cold.sqlite`），窗口 t≈89–129s = cold 波次的单槽 128K
chunked prefill（16×8192）+ 其余三槽 restore。

| kernel 族 | 时间 | 占比 |
|---|---:|---:|
| FlashInfer FA2 paged-prefill（FP8 KV, causal, MaskMode1, CTA_Q=64） | **21.50 s** | **57.0%** |
| b12x W4A4 dense GEMM（三个变体合计） | 8.19 s | 21.7% |
| elementwise/cat/copy/rms-tail/nvfp4-quant 杂项 | ~4.3 s | ~11% |
| GDN causal-conv（torch depthwise 兜底实现） | 0.75 s | 2.0% |
| flashinfer GDN delta-rule prefill | ~0.7 s | ~2% |

### "20 秒消失时间"的结论：不存在

早间接入 `QSR_PROFILE_ADMISSION=1` 时看到 admission `activate=36.9s` 远大于
batched_forward 段总和（17.3s），疑似隐藏拷贝或 host 气泡。nsys 归因证明：
**GPU 在整个 prefill 窗口 96–100% busy**，缺口全部是 FA2 attention 的真实执行
时间——wall-clock 分段计时因为异步发射 + 段边界处的隐式 synchronize 把
attention 成本记在了段间隙里。prefill 就是 attention 主导，没有第二嫌疑人。

### Prefill attention 已在实用天花板

- 生产 trace 中 252 次 FA2 调用（16 层 × ~15.75 chunk）总长 21.5s；
- 同几何独立 microbench（`/tmp/opencode/bench_prefill_attn.py`）逐 chunk 复现
  了生产数字（±18%，散页略慢）：B=1 期望值 = 实测值；
- 该 kernel 有效吞吐 ~160 TFLOPS；本卡实测 BF16 K=256 skinny-GEMM 上限
  （cuBLAS，QK^T [32768,256]×[256,131072]）= **168 TFLOPS**，PV 形状 192；
  8192³ dense BF16 峰值 297 TFLOPS。
- 即 **FA2 ≈ cuBLAS 同形状上限的 96%**，tile/plan 层面没有可捡的钱。
- B→T 线性外推验证过：T(B=4) ≈ 4×T(B=1)，kernel 在 B=1 大 M 下已饱和，
  多槽合批不省 attention 总时间。

### FP8-MMA attention 在 SM120 上不可用（代码级确认）

prefill/decode attention 要突破 BF16 天花板需要 FP8 QK^T（实测
`torch._scaled_mm` 同形状 248 TFLOPS，1.48×）。三条路全部关闭：

1. fa2 backend：`AssertionError: fp8 tensor core is not supported in fa2 backend`；
2. trtllm-gen：编译失败根因已定位——`fmhaKernels.cuh` 以
   `#if CUDA_VERSION >= 13030` 引用三个新版头文件才有的 oversized-smem 枚举；
   本机全部 toolkit（13.2/13.3）都没有。本地 patch 掉该 Rubin-only 分支后可
   编译、可 plan，但运行时被
   `FLASHINFER_CHECK(mSM == kSM_100 || kSM_103 || kSM_107)` 拒绝：
   **trtllm-gen FMHA 只支持 SM100/103/107（tcgen05 家族），SM120 物理不在列**。
   与 roadmap"SM120 无 tcgen05"一致。补丁已回滚，环境保持干净；
3. cutlass backend：本 build 未对 paged prefill 接线
   (`backend must be fa2 or fa3`)。

## Decode 侧复核

- fullchain nsys（早间，pre-FP8-head）显示 warm decode 波内 GPU busy
  **87–90%**，空闲集中在请求波边界，不是每 round 的结构性气泡；
- verify attention（FA2 FP8-KV split-KV）846µs/call vs KV 读地板 ~700µs
  （~85%），`use_fp16_qk_reduction`、静态 layout、page-count-only replan 等
  候选已在早间 note 里以同口径否决；
- W4A4 decode GEMM 近带宽地板（roadmap §2.1）；lm_head FP8 已落地；
- acceptance 904/1020（95%，mean committed 7.5/round）接近结构上限。

## 剩余杠杆（按可信收益排序，均为"苦活"）

| 杠杆 | 预期 | 风险 |
|---|---|---|
| ✅ **B8 逻辑并发——本轮已实测成立**（见下节） | 长 gen aggregate **+23%** | 冷启动显存余量、混合长度负载待门禁 |
| prefill elementwise/cat/copy 融合（~4.3s 中约一半可及） | TTFT −4~6% | 低（无数值序问题），但需逐个归因 |
| decode RMSNorm 链 pow/mean/rsqrt 融合 | decode +3~5% | 高：bit-exact 减少序是已知的坑（P0-4 只允许 tail 融合，tail 已做） |
| GDN causal-conv Triton 化 | ~1.5% | 低；但 fused dflash2_conv 先例未测得稳定 E2E 收益 |

## B8 实测：roadmap P0-2 在 128K DFlash2 上成立（2026-08-25 晚补测）

`--capacity 8` + `QSR_SERVER_NUM_SLOTS=8`（注意：`num_slots` 默认恒为 4，
不随 capacity 自动放大，首次启动因此 `ValueError`）。动态 arena 不扩容：
rows=9 时公式仍为 18.28 GiB；进程稳态显存 **85.5 / 97.9 GiB**。全部 graph
family（draft_b1..b8、verify、verify_ragged）capture 成功，零 fallback。

同服务器、prefix-warm、128K、1024 输出 token 的同口径 A/B：

| 配置 | decode/req | aggregate E2E | mean accepted/round | completion SHA |
|---|---:|---:|---:|---|
| c=4 | 173.5–174.7 tok/s | 655–658 tok/s | — | `4542da20…` |
| c=8 | 122.6–124.5 tok/s | **800–819 tok/s (+23%)** | 6.97/7（优于 c4 的 6.65） | 同左 |

纯 decode 窗口聚合速率：652 → **921 tok/s（+41%）**。

两个必须同时报告的口径事实：

1. **256-token 短生成下 aggregate_e2e 持平（502 vs 512）**——不是缩放失败，
   是 0.8s 的 warm TTFT（8 slot 串行 restore/logits/commit）在短输出上摊不掉。
   报告吞吐结论时必须用长生成或 decode 窗口口径。
2. roadmap P0-2 的 +39% 外推基于"GEMM 主导的轮次"，在 128K 下 attention KV
   流量随 B 线性增长本应吃掉收益；实测 decode 窗口 +41% 说明权重摊薄收益
   仍然占优。P0-2 的"仅适用于短上下文"担忧不成立。

未完成的发布门禁：冷启动（非 prefix-hit）满 8 slot 显存峰值复测、混合长度/
混合并发稳定性、watchdog 与 timeout 行为、质量语料抽检。本轮仅证明性能与
数值一致性，不改任何默认配置。

## 明确不要再走的路

- trtllm-gen / fp8-MMA attention on SM120（架构门，见上）;
- FA2 tile/plan 参数再扫（已在本轮 microbench 内含）;
- 多槽 batched prefill 省 attention（线性缩放，省不了）;
- 隐藏拷贝/同步气泡类怀疑（GPU busy 96–100%/87–90% 已排除）。

## 产物

- `/tmp/qwen38_cold_prefill_trace.nsys-rep` + `/tmp/opencode/qwen38_cold.sqlite`
- `/tmp/opencode/bench_prefill_attn.py`、`bench_verify_backends.py`（复现脚本）
- `/tmp/opencode/server_perf_grid_qwen38_dflash2_baseline{,_prof}_128k_c4.json`
- `/tmp/opencode/qwen38_server_18511.log`（QSR_PROFILE_ADMISSION 分段）
- B8：`server_perf_grid_qwen38_dflash2_128k_c{4,8}_cap8_long.json`（1024-token
  同口径）、`…_c8_cap8.json`（256-token 短生成对照）、服务器日志
  `qwen38_server_18513.log`

## 追加（2026-08-25 深夜）：decode 轮内节点级归因 + GDN conv 融合落地

### DFlash2 decode 单轮归因（nsys --cuda-graph-trace=node，干净轮周期）

| 组件 | ms/轮 | 占比 | 判定 |
|---|---:|---:|---|
| W4A4 GEMM（b12x，222 次调用） | 10.81 | 23% | 纯权重流地板 ≈8.3ms（13.7GB @1.65TB/s 读带宽），余量 ~2.5ms |
| FlashInfer verify attention（16 层） | 10.60 | 23% | KV 纯读地板 ≈10.4ms——**已在绝对地板** |
| lm_head/nvjet 族 | ~2.6 | 5.6% | NVFP4 化可省 ~0.9ms（质量门禁） |
| GDN multistep | 1.28 | 2.8% | |
| RMSNorm 链（pow/mean/rsqrt/recip/mul/tail） | ~1.8 | 3.9% | bit-exact 减少序是硬门 |
| 小 index/copy/cat/fill | ~1.8 | 3.9% | 可融合，无数值风险 |
| nvfp4 quantize | 0.41 | 0.9% | |
| 其余 | ~2.9 | 6% | |

结论：**verify attention 与 W4A4 都已贴地板**。decode 剩余可回收空间 =
小算子融合 ~4ms/轮（+9%）+ host gap ~3-5ms/轮，均为多小时级苦活；不存在
单点大鱼。

### 波次边界 restore 的隐藏成本（非每轮）

`restore_prefix_from_scratch`/`_copy_prefix_to_scratch`
（runtime/backends/qwen36_dspark.py:591-657）用 `cache[:, page_tensor] =
cache[:, source_pages]` 高级索引整槽拷贝 draft KV：每次 268.4M 元素
（=2048 页×128×8 头×128 维×K/V），~950µs、有效带宽仅 ~280GB/s，
每波每恢复槽 10-11 次 → 波边界 ~10ms×槽数。对长请求稳态可摊销，
对短波/频繁换 prefix 的 agent 流量是真实成本。低风险修法：页重映射
（改 page table 视图）代替物理拷贝。

### GDN causal conv+SiLU Triton 融合（已落地，未提交）

`runtime/kernels/gdn_conv.py` + `_conv1d_window` 接线
（`QSR_QWEN36_GDN_CONV_FUSION=0` 回滚）：

- **逐位一致**：与 eager 对在生产布局（transpose 视图→generic depthwise
  kernel）下 84M 元素 0 差异；SiLU 必须用 ATen 形式 `x/(1+exp(-x))`；
  fp32/GGUF 路径因 FMA 缩并差异不融合（保持 eager）。
- **5.17×**：[1,10240,8195] 1746→338µs；B=4 时 7.1×。
- **prefill 前向实测**：16 chunk Σ batched_forward 16.713s → 15.529s
  （**−1.18s，−7.1% of forward**）；TTFT 配对方向一致（37.52 vs 37.73s）
  但低于 ±2s 运行间噪声，按段级口径计入。
- 测试：tests/test_qwen36_gdn_conv_fusion.py（4 passed）+ qwen36_backend/
  dflash2/dspark 回归全绿。

### 当天最终账本

| 项 | 结果 |
|---|---|
| cold prefill 前向 | −1.18s（conv 融合，段级实测） |
| decode 单 token | 无变化（所有大项在硬件地板） |
| B8 容量 | aggregate +23%（配置级） |


### v2 终测（双分支融合，段级口径）

`QSR_PROFILE_ADMISSION=1` 同口径复测（18526）：

| 指标 | EAGER 基线 | FUSED-v2 | Δ |
|---|---:|---:|---|
| 16-chunk prefill Σ batched_forward | 16.713 s | **14.852 s** | **−1.86 s（−11.1%）** |
| cold TTFT | 36.28–39.72（当日带） | **36.60 s** | 带内最优 |
| warm decode/req | 144–163（当日带） | 156.95 / 160.57 | 无回归 |
| completion SHA | `75b43a8a…` | 相同 | 逐位一致 |

decode 窗口 step kernel 单测 [4,10240,4]：6.4µs vs eager 7.7µs，逐位一致；
通道非整除（C=2560/3000）mask 用例已补。回归：qwen36_backend/dflash2/
dspark/recurrent_state_pool 共 103 passed；ruff 全绿。


### 显存优化：DFlash2 draft KV 窄行环形映射（已落地，未提交）

`qwen_kv_dspark_draft_bytes` 原为 **12.50 GiB**（5 层 × [2,(slots+1)×2048 页,
128,8,128] fp8），而 DFlash2 draft 滑窗只有 2048 token——98% 是死重。
`QSR_QWEN36_DSPARK_DRAFT_NARROW_ROWS=1`（默认开，`=0` 回滚）把每槽物理行宽
缩到 `draft_row_pages=19`（窗口跨度 18 页 + 1 个保留页给 fused-context
epilogue 的 scratch 行），所有绝对页号经 `abs % 18` 环形映射进窄行：

- 映射是绝对页号的纯函数 → prefix preserve/restore/slot_mapping/attention
  page table 全部自洽；restore 只拷最后 min(pages,18) 页（波边界大 gather
  从 268M 元素降到 ~4.7M，60×）。
- 波及面：qwen36_dspark.py（分配/页表/slot_mapping/fused-context 元数据/
  三条 copy 路径）、qwen36_dspark_cudagraph.py（B=1 与 batched 图的
  page_table/_slot_mapping/_slot_page_tables/impl max_pages）、
  flashinfer impl 构造参数。

实测（cap=4，128K/c4 fixture，SHA `75b43a8a…` 逐位一致、接受率 896/1020
与基线相同）：

| 指标 | 前 | 后 |
|---|---:|---:|
| draft KV | 12.50 GiB | **0.116 GiB** |
| 整卡（fresh load） | 68.5 GiB | **57.3 GiB** |
| cold TTFT | 当日带 36.3–39.7 s | **34.20 s** |
| warm decode/req | 144–163 tok/s | **169.3 tok/s** |
| cold wall | 38.0–41.9 s | **36.04 s** |

capacity=8：fresh load 85.5 → **62.8 GiB**（draft 家族省 ~23.9 GiB），
c8×1024-token aggregate 809 → **871.8 tok/s**。回归 108 passed、ruff 绿。


### 追加（2026-08-26）：opencode 真实任务评测与一次审计误报的更正

用本地 runtime 跑 opencode 三轮复杂任务（JSON Schema 校验器 1442 行/107 测试、
mini-git 1645 行/38 测试、压测工具 1105 行/27 测试），累计 **86+ 次工具调用、
196+ 请求、服务端零错误**。

审计更正：我曾判定 mygit checkout 缺少未提交修改保护。复测证明该判断错误——
同 commit 间切换带脏工作区本就应允许（git 语义）；真正危险场景（目标树不同 +
未提交修改）实测 exit=1、列出受影响文件、工作区保留，实现完全正确。教训：
对抗探针自身要先对照参照语义校验期望值，否则会把正确实现误判为 bug。

effort 参数：机制通但效应量小（难难题 low/high 思考长度 ±3%），为模型属性。
