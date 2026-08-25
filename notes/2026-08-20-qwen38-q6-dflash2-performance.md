# Qwen3.8 Q6 GGUF + DFlash2 性能实测

> 日期：2026-08-20
> 状态：🟡 原生路径、CUDA Graph 和 4K 快速 A/B 已完成；正式质量、长上下文、并发矩阵仍是独立发布门禁

## 结论

Q6+DFlash2 已经不是“只接上能跑”的路径：target decode、DFlash2 draft、
fixed verify 和 ragged verify 都在隔离 SM120 进程中实际 capture/replay，
固定 workload 下三种配置均为 `28/28` 接受，输出 SHA 一致。

Q6 有两个原生执行模式：

| 模式 | warm decode 均值 | warm TTFT 均值 | warm wall 均值 | 显存代价 |
|---|---:|---:|---:|---:|
| packed tensor-core（`QSR_GGUF_DEQUANTIZE_WEIGHTS=0`） | **51.255 tok/s** | **21.1175 s** | **21.7252 s** | 紧凑 GGUF packed 权重 |
| resident BF16（服务端 Q6+DFlash2 默认） | **81.905 tok/s** | **3.7117 s** | **4.0935 s** | 比 packed 新鲜进程多约 **26.7 GiB** |

resident 模式在 eager warmup 中逐层把 Q/K 权重反量化为 BF16，释放对应
packed payload，后续 projection 走 BF16 `F.linear`/cuBLAS 路径；它不是
eager fallback，也没有关闭 CUDA Graph。容量 1 的新鲜进程实测显存为
`85,490 / 97,887 MiB`，约剩 `12,013 MiB`，因此当前 Q6+DFlash2 的默认
服务配置可以装下，但增加 slots 或 KV 预算前必须重新做显存门禁。

## 与现有 NVFP4 + DSpark 的严格 A/B

两组使用同一张卡、同一个 `/home/bot/.venvs/torch-nightly`、同一 tokenizer、
4K prompt、并发 1、32 output tokens、prefix cache off、cold + 2 warm
rounds。NVFP4 和 Q6 都使用 K=7 speculative decoding；Q6 使用 DFlash2，
NVFP4 使用 DSpark。

| 配置 | warm1 decode | warm2 decode | warm decode 均值 | warm TTFT 均值 | warm wall 均值 | e2e tok/s |
|---|---:|---:|---:|---:|---:|---:|
| NVFP4 + DSpark K=7 | 225.48 | 238.61 | **232.045** | **0.6365 s** | **0.7733 s** | **41.38** |
| Q6 packed + DFlash2 K=7 | 51.32 | 51.19 | **51.255** | **21.1175 s** | **21.7252 s** | **1.47** |
| Q6 resident BF16 + DFlash2 K=7 | 80.48 | 83.33 | **81.905** | **3.7117 s** | **4.0935 s** | **7.81** |

相对 NVFP4，resident Q6 的 decode 慢 **2.83x**，TTFT 慢 **5.83x**，端到端
吞吐约慢 **5.3x**。因此 Q6 现在适合作为质量/格式对照和显存允许时的
备选服务，不具备替换 NVFP4+DSpark 的性能条件；DFlash2 本身没有被接受率
拖慢，瓶颈已经从 packed projection 转移到 Q6 的 F32 target 图和算子组合。

## 优化证据

1. 初始 native packed Q6+DFlash2 为 `24.495 tok/s`，4K warm TTFT
   `99.2053 s`。F32 packed projection 的 tensor-core decoder（M>=8，覆盖
   DFlash2 固定 verify 宽度）将其改善到 `51.255 tok/s`、`21.1175 s`。
2. 逐层 op profile 显示 512-token prefill 中 packed MLP/projection 占主要
   GPU 时间；不是单独 FLA GDN 的问题。实际形状 `M=4096,N=10240,K=5120`
   的 packed tensor-core GEMM 约 `44 ms`，BF16 dense GEMM 约 `2 ms`。
3. resident BF16 fresh-process 验证把 4K warm TTFT 从 `21.1175 s` 降至
   `3.7117 s`，decode 从 `51.255` 提升至 `81.905 tok/s`；512-token
   短请求也保持 `7/7` 接受和相同输出 SHA。resident 路径在短 decode
   上不因单层 microbench 的优势而虚报全模型收益，最终数字以 4K A/B 为准。

## CUDA Graph 与正确性门禁

- Q6 resident fresh process 启动完成：target decode、DFlash2 draft
  `draft_b1`/`draft`、`verify_ragged`/`verify` 均为 `captured`。
- 4K 请求每次 4 个 speculative rounds，接受 `28/28`（7/round）；draft
  和 verify Graph replay 分别为 5/4。
- Q6 packed、Q6 resident、NVFP4 的固定 4K completion 输出 SHA-256 均为
  `34850d3f903fd71918a3db8ba1dd257b20f56b3e49405b4893ed100322c28e85`。
- `tests/test_gguf_qk.py` 覆盖 Q4/Q5/Q6/Q8 的 packed tensor-core 数值、
  非 padded M=8 verify、resident BF16 释放/重用；server flag 测试覆盖
  Q6 默认 resident 和显式 packed rollback。

## 小型质量门禁

使用同一个 `benchmarks/quality_regression.py`、同一个
`qwen3_coder` tool parser、并发 1 和同一组 20 个工具调用 + 4 个 agent
数学场景，结果如下：

| 配置 | 工具名准确率 | 工具完整准确率 | agent 最终答案 | 工具调用率 |
|---|---:|---:|---:|---:|
| NVFP4 + DSpark | 20/20 | 19/20 | 4/4 | 4/4 |
| Q6 resident + DFlash2 | 10/20 | 9/20 | 4/4 | 2/4 |
| Q6 packed + DFlash2 | 13/20 | 12/20 | 4/4 | 2/4 |

这是小样本 smoke，不足以替代正式 MMLU/code/长上下文套件；但它没有显示
Q6 的质量优势，且暴露了 Q6 在当前 tool-call 口径下的格式/调用不稳定性。
因此 Q6 当前不满足“性能明显提升或质量明显更好”的替换条件。

## 口径与产物

- Python：`/home/bot/.venvs/torch-nightly/bin/python`
- Q6 target：`/home/bot/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K_XL.gguf`
- DFlash2 draft：`/home/bot/models/Qwen3.8-27B-DFlash2`
- 1 slot、block size 128、max context 8192、CUDA Graph、prefix cache off
- 原有 `benchmarks/server_perf_grid.py`：4K prompt、并发 1、32 output
  tokens、cold + 2 warm；没有新增 benchmark 脚本
- HTTP JSON 产物：
  `/tmp/qwen38_q6_tc_m8_4k_completions_perf.json`、
  `/tmp/qwen38_q6_dequant_4k_perf.json`、
  `/tmp/qwen38_nvfp4_dspark_4k_perf.json`、
  `/tmp/qwen38_q6_dflash2_quality_packed_qwen3coder.json`、
  `/tmp/qwen38_q6_dflash2_quality_qwen3coder.json`、
  `/tmp/qwen38_nvfp4_dspark_quality_qwen3coder.json`
- 这些 HTTP 产物不是 bfdiag run record，因此没有伪造 `bf diff`；三组比较
  仍严格固定了脚本、环境、tokenizer、prompt 和服务参数。

## 未完成门禁

数字 filler 会显著有利于 speculative acceptance，不能代表 prose/code 的
一般接受率。正式结论还需要真实质量集、16K/32K/128K 长上下文、并发
2–4、prefix hit、冷启动显存压力和 resident 模式下的多 slot 约束。现有
服务没有被重启或修改；本轮验证只使用隔离端口 `127.0.0.1:18380`，完成后
已停止隔离进程。
