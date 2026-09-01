# Qwen3.8-Flash-Next sglang 基线（自研 runtime 的性能靶子）

日期：2026-08-27。机器：RTX Pro 6000 Blackwell（96 GB）+ 23 GB RAM，
权重 `RadixArk/Qwen3.8-Flash-Next-NVFP4`（126 GiB，专家 NVFP4 + 注意力
BF16，`lovedheart` 的 MIXED_PRECISION 版已验证不可用并删除）。

## 基线配置（唯一跑通的组合）

```
sglang 0.5.18.dev696（main + PR#36497 模型支持 + #36556 SM120 QSA
+ #36567 NVMe PLE，合并在 /home/bot/project/sglang 的 qwen4-support 分支）

env:
  SGLANG_QWEN4_PLE_NVME_PATH=<ckpt>      # 47.7 GiB FP8 PLE 表走 NVMe
  SGLANG_QWEN4_PLE_NVME_BACKEND=io_uring
  SGLANG_QWEN4_PLE_NVME_QUEUE_DEPTH=512
  SGLANG_QWEN4_PLE_NVME_CACHE_PAGES=2097152   # 8 GB RAM LRU 页缓存
  MAX_JOBS=2                                  # 23 GB 内存，JIT 防 swap 雪崩

flags:
  --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass
  --linear-attn-prefill-backend flashinfer --linear-attn-decode-backend triton
  --page-size 64 --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64
  --chunked-prefill-size 512 --max-running-requests 1 --context-length 32768
  --mem-fraction-static 0.91
  --speculative-algorithm NEXTN --speculative-num-steps 3
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled
  --reasoning-parser auto
另需卸载 torchcodec（缺 FFmpeg）。
```

## 实测数字（2026-08-27）

| 指标 | 值 |
|---|---|
| decode（短 prompt，512 tok） | **20.5 tok/s**（无 MTP 基线 ~13） |
| MTP accept rate / accept len | 0.68 / 3.05 |
| prefill（23,473 tok prompt） | ~2–4K tok/s（chunked 512） |
| 长上下文 | 23.5K prompt 正确作答，32768 上限内可用 |
| 显存占用 | ~90/96 GB（专家 63 + 其余 12 + KV/缓冲） |
| 内存占用 | ~15/23 GB（含 8 GB PLE 缓存） |

## 已钉死的限制（自研 runtime 要逐条打破）

1. **CUDA Graph × NVMe PLE 不兼容**：捕获期非 pinned CPU→CUDA 拷贝直接
   失败（`Cannot copy between CPU and CUDA tensors during CUDA graph
   capture`）。模型代码有 `_graph_prefetch_buffers`/breakable-graph 意图
   但实际路径没接通。→ 自研版必须自己把 PLE gather 做成图安全
   （pinned staging + 图内固定地址），这是最大的单点收益。
2. **MTP × flashinfer decode 互斥**：MTP 的 GDN 状态检查点强制
   `state_checkpoints` fp32（`extra_buffer` 策略），而
   `--linear-attn-decode-backend flashinfer` 在 SM100+ 强制
   `--mamba-ssm-dtype bfloat16`（server_args.py:6252 的
   `_handle_linear_attn_backend`）。二者只能选一，被迫用慢的 triton
   decode。→ 自研版控制自己的 GDN 状态布局，两者可兼得。
3. **PLE 表 47.7 GiB 放不进 23 GB 内存**，只能 NVMe 流式 + 8 GB 缓存；
   io_uring 直读每步都过盘（缓存未命中时）。→ 自研版可做更激进的
   行级量化缓存/预取重叠。
4. 23 GB 内存使任何"全表进内存"方案不可行；JIT 编译必须限并发。

## 靶子

自研 runtime 原生支持 `qwen4_exp`，decode **100 tok/s**（~5× 基线）。
收益来源预算：CUDA Graph（sglang 拿不到）+ b12x SM120 融合 MoE +
MTP 与 flashinfer 级 GDN decode 兼得 + PLE 图安全预取重叠。
实施计划见 `2026-08-27-flashnext-runtime-support-plan.md`。

## 同提示 profile 复测（2026-08-28）

为消除旧基线与自研 runtime 提示不一致的问题，使用实施计划 4.14 固定的
109-token TCP chat prompt、temperature=0、64 completion token 复测：

| 指标 | SGLang | 自研全 M=4（128 轮长跑） |
|---|---:|---:|
| TTFT | 0.475 s | **0.408 s** |
| post-first decode | **35.87 tok/s** | 34.44 tok/s |
| 平均提交 token/轮 | 2.462 | 2.492 |
| 推算/实测 draft acceptance | 48.7% | 49.7% |

SGLang trace：`/tmp/sglang-fn-profile-20260828-r3.nsys-rep`。该采集关闭 decode/
prefill CUDA Graph，与启动参数基线一致。trace 范围还包含一次 128-token server
warmup，所以只用能由调用数严格分离的 verify 统计；不把整个 range 的 GPU sum
冒充单请求时间。详细 kernel 归因见实施计划 4.15。

自研列的 TTFT 已切换为新进程 whole-prompt large-M prefill；128 轮 decode 为
34.42 tok/s（与此前串行 prefill 后的 34.44 tok/s 等价波动），完整请求为
320 completion token / 9.714 s = 32.94 tok/s。两边 completion 长度不同，表中
只把 TTFT、稳态速率和每轮接受形态作为可比项，不拿请求总时长做伪精确比较。
