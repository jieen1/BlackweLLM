# CUDA Graph vs eager 解码吞吐（标准模型，服务路径）：**4.71×**

日期：2026-08-03 · 模型：`unsloth/Qwen3.6-27B-NVFP4`（标准模型）· 单卡 RTX PRO 6000
Blackwell Max-Q，`/tmp/gpu_lock.sh` 下独占 · `capacity=1`（单并发）

## 一句话

**在册的每一个吞吐数字都是 eager 的，比这个运行时打开 CUDA Graph 后的实际速度低约 5 倍。**

| | 中位数 | 三轮 | 常驻 |
|---|---:|---|---:|
| **CUDA Graph** | **28.848 tok/s** | 28.695 / 28.848 / 28.912 | 72.39 GiB |
| eager | 6.120 tok/s | 6.098 / 6.208 / 6.120 | 77.69 GiB |
| | **4.71×** | | **CG 反而少 5.30 GiB** |

## 为什么之前没测出来

两件事叠在一起：

1. **w4a16 融合 MLP 路径的 CG 捕获今天才刚修好。** 在此之前捕获会失败，而失败是
   **静默的**——退回 eager 继续跑，日志里什么都不说。根因是 sparkinfer 的
   `plan_w4a16_buffers` 给 fc1/fc2 `c_tmp` 只按打包路径的界分配（本部署 decode
   形状 9 槽），而 decode 的 direct-topk 快路径不打包、需要 16 槽。
   现由 `tests/test_w4a16_scratch_contract.py` 守着。
2. **既有测量脚本本来就只测 eager。** `scripts/measure_nvfp4_gemm_memory_and_throughput.py`
   是裸 forward 循环，不走服务路径，也从不捕获 CUDA Graph——它的
   docstring 自己写着 "what does eager (no CUDA graph) greedy decode throughput
   look like"。**它没做错任何事，只是从来没人补上另一半。**

于是"实测吞吐 ~6 tok/s"被当成了运行时的能力，并据此得出过"NVFP4 量化白做"之类的判断。

## 测量方式（可比性是这次唯一必须做对的事）

同一个进程启动方式、同一个 prompt、同一组采样参数、**同样的 `num_slots=2`**
（eager 其实不需要那个 CG warmup 槽，但**保持一致比省一个槽重要**），
**只切 `QSR_SERVER_ENABLE_CUDAGRAPH` 一个变量**。

- prompt: `"Write a Python function that merges two sorted lists into one sorted list."`
- `max_tokens=256`，`temperature=0.0`，流式
- **解码时间从首 token 起算**到末 token，而不是"总 token / 总墙钟"——否则 prefill
  会混进去。每轮均满 256 token，所以两侧统计口径完全一致。
- 每臂先跑一次 warm 请求丢弃，再连测 3 轮取中位数。

harness 在 `$CLAUDE_JOB_DIR/tmp/cg_vs_eager.py`（一次性，未入库）。

⚠️ **与在册数字的关系**：本次 eager 臂 6.120 tok/s 与既有记录的 5.819 / 6.442 /
6.547 tok/s 相符，**这是佐证，不是同一测量**——那些数字出自另一个 harness（裸
forward 循环）。**4.71× 这个比值只在本表内部成立**，因为只有这两臂是同 harness 的。

## 顺带纠正两条

- **CG 捕获不贵。** 启动：CG 24.9s vs eager 21.0s，差约 4 秒。
  本轮更早一次冷启动是 311 秒，我一度以为是捕获/JIT 前置的代价，**那个推断是错的**
  ——那是**首次从磁盘读 22 GiB checkpoint**；`~/.cache/sparkinfer` 在整个过程中
  零文件写入，page cache 暖起来之后就是 21–25 秒。
- **CG 更省显存**，不是更费：72.39 vs 77.69 GiB。

## 现在的位置与下一步

权重常驻约 20.2 GiB，28.85 tok/s 对应有效显存带宽 **约 582 GB/s**。这张卡的峰值在
1.8 TB/s 量级，即目前约在 **roofline 三成**。**headroom 是真的，而且现在起点是
28.85 不是 6.1——之前所有基于 ~6 tok/s 做的优化判断都需要重估。**

未决：

- 首 token 4.67s（每进程第一次请求，之后稳定 0.25s）。
  ⚠️ **本文最初把它归给"没有持久化编译缓存"，那是错的**：`sparkinfer/_lib/compiler.py`
  有 spec memo / 内存 LRU / **磁盘缓存**三层，`SPARKINFER_COMPILE_DISK_CACHE` 默认开，
  `cache_key` 正是磁盘缓存键的一部分（`KernelCompileSpec.from_key`）。
  `~/.cache/sparkinfer` 297 MB 且今天多次起服务零写入——**是命中**。
  所以这 4.4 秒**不是编译**，需另行定位（`gemm/bf16_gemv/_kernel.py:207` 提到
  first-launch lazy module load）。`compile_cache_info()` 可逐项核实 hits/misses。
- `KV 8192 MiB/slot × 2 槽 = 16 GiB`，因为默认 `max_context=131072`。
  常驻 72 GiB 里这是一大块，且是配置选择而非缺陷——但值得给出按需配置的指引。
## 并发下 CG 收益会收窄，但不会消失（2026-08-03 补测）

同一 harness，capacity 1/2/4 各扫一遍，每档仍然只切 `QSR_SERVER_ENABLE_CUDAGRAPH`
一个变量，`num_slots = capacity + 1` 两臂相同。并发用 N 个同时发出的相同流式请求驱动，
吞吐按"所有流的 token 数 / 有流在解码的时间窗"聚合。

| capacity | CG tok/s | eager tok/s | 比值 |
|---:|---:|---:|---:|
| 1 | 28.56 | 5.95 | **4.80×** |
| 2 | 47.71 | 12.34 | **3.87×** |
| 4 | 68.59 | 19.58 | **3.50×** |

（cap1 的 28.56 与本文上半部分独立测得的 28.85 吻合，harness 可复现。）

**两件事同时成立，别只记住一件：**

1. **CG 的相对优势随并发收窄**（4.80 → 3.87 → 3.50）。这正是预期方向：CG 消掉的是
   **每步固定的 CPU 侧开销**（paged 元数据 ~34 ms/step），并发越高这笔开销被越多 token
   摊薄，eager 自己就能追回一部分。
2. **但 CG 在 cap4 仍然赢 3.50×，没有变成边际收益。** "并发上来了就不需要 CG"是错的。

**CG 自身的扩展是次线性的**：1→2→4 并发给到 28.56 → 47.71 → 68.59，即 **1.67× / 2.40×**。
eager 的相对扩展反而更好（2.07× / 3.29×）——因为它起点离 kernel 上限太远，有的是空间可追。
CG 的次线性正说明它已经贴近计算/带宽上限，与
[`2026-08-03-decode-kernel-profile.md`](2026-08-03-decode-kernel-profile.md) 的
"CG 下 GPU 89% 忙、已 kernel-bound"是同一个结论的两个侧面。

## 相关

- [`2026-08-03-nvfp4-gemm-memory-audit.md`](2026-08-03-nvfp4-gemm-memory-audit.md) —— eager 侧 5.819 tok/s 的来源与内存审计
- [`2026-08-03-std-model-serving-acceptance.md`](2026-08-03-std-model-serving-acceptance.md) —— 标准模型首次服务通过（C-LIVE 64/67）
- `tests/test_w4a16_scratch_contract.py` —— 守住"必须配已修复的 sparkinfer"这条依赖
