# Strict 4×256K fresh-process startup acceptance (2026-08-15)

状态：🟢 **实测通过**。`docs/qwen38-sm120-cuda133-fa4-optimization-plan.md`
§4.9 / Phase 1M 的启动验收门禁，在 P0-M1（跳过 W4A16 死表示）、P0-M2
（共享 eager arena + 捕获后释放）、P0-M3（MTP graph pool 按 family 共享）
和 P0-C（MTP COW 修复）全部落地后，用 fresh process 分阶段实测。

## 配置

- 模型：`unsloth/Qwen3.6-27B-NVFP4`（本机 snapshot），`max_seq_len=262144`，
  `enable_mtp=True`，默认 all-W4A4（`QSR_QWEN36_MLP_W4A4_ALL=1`）。
- 后端：`Qwen36Backend(dynamic_arena=True, pool_bundles=8201, watermark=8,
  num_slots=4, block_size=64)` —— strict 公式 `1 + 4×2048 + 8 = 8201`。
- 序列完全复刻 `ServerEngine._load_qwen36_model`：load → backend →
  full-forward warmup → enable_mtp(K=3) → capture_decode_cuda_graph。
- 脚本：`/tmp/opencode/accept_strict_4x256k.py`（一次性验收脚本，未入库；
  判据与下表以本文为准）。

## 分阶段显存（NVML / torch allocated / reserved）

| 阶段 | NVML used | allocated | reserved | Δallocated |
|---|---:|---:|---:|---:|
| 0 CUDA baseline | 2.73 GiB | 0.00 | 0.00 | — |
| 1 模型加载（含 attention warmup） | 25.37 GiB | 21.06 | 22.62 | +21.06 |
| 2 strict backend（8201 bundles） | 57.42 GiB | 53.50 | 54.70 | +32.44 |
| 3 full-forward warmup | 57.74 GiB | 54.41 | 55.00 | +0.91 |
| 4 enable_mtp + MTP graph capture | 63.83 GiB | 60.54 | 61.02 | +6.12 |
| 5 decode graph capture | 63.90 GiB | 60.51 | 61.04 | −0.02 |
| 6 短请求（5 tokens） | 63.90 GiB | 60.54 | 61.04 | +0.03 |
| 7 逐槽增长（4 槽至 512..2048 tokens） | 64.09 GiB | 61.15 | 61.24 | +0.61 |

设备总量 95.59 GiB，**峰值 NVML 64.09 GiB，driver free 31.51 GiB**。

## 增量解读（对账规划的代码精算）

- **阶段 2 的 +32.44 GiB = backbone KV only**：8201 bundles × 4 MiB
  （16 层 × K/V × 4 kv heads × hd256 × FP8 × 128 tokens）≈ 32.0 GiB，
  加 conv/recurrent/attn_outputs 池与 page table。MTP 池**不在**这一步——
  `build_pooled_mtp_caches` 在 `Qwen36MTPEngine.__init__`（阶段 4）才分配。
- **阶段 4 的 +6.12 GiB = MTP KV + MTP graph pools**：MTP 池
  8201 × 0.5 MiB ≈ 4.0 GiB；其余 ~2 GiB 是 36 个 MTP graph 的捕获开销
  （P0-M3 后为 3 个共享 family pool，不再是 36 个独立 private pool）。
- **KV 合计 32.0 + 4.0 = 36.0 GiB**，与规划 §4.1 的代码精算
  36.03955 GiB 吻合——动态 allocator 确实降不到这个下界以下，也没有降。
- **阶段 5 的 Δallocated ≈ 0**：backbone decode graph 的 pool 增量被
  P0-M2 step 2-3 的释放（B1–B4 eager drivers + 共享 decode arena）抵消。
  这是释放钩子生效的直接证据。
- **模型加载 21.06 GiB**：权重常驻。P0-M1 实测省下的 7.88 GiB
  （W4A16 死表示）没有出现在任何阶段——all-W4A4 直接 prepare，首次
  forward 不再经过 W4A16。若无该修复，阶段 3 会多出 ~7.9 GiB。

## 验收判据（规划 §4.9）

| 判据 | 结果 |
|---|---|
| strict 4×256K 可启动并完成逐槽增长，不 OOM | ✅ 启动至峰值 64.09 GiB；4 槽分别增长到 512/1024/1536/2048 tokens |
| 运行峰值保留至少 10 GiB driver free | ✅ 31.51 GiB free |
| 短请求不因 max_model_len=256K 占用完整 business KV | ✅ 5-token 请求 → live_bundles=1（不是 2048） |
| 同时报 NVML used / allocated / reserved / graph pool | ✅ 上表；graph pool 体现在阶段 4/5 增量 |
| 逐槽增长成比例 | ✅ 512/1024/1536/2048 tokens → 4/12/24/40 live bundles（每 128 tokens 1 bundle，严格线性） |

## 与规划预估对比

规划 §4.9 预估"strict 启动可能进入约 65–70 GiB 区间"（当时 graph private
pool 收益 unknown）。实测峰值 **64.09 GiB**，优于区间下界。剩余大头：
GDN checkpoint/persistent clone（P1-M，未做）与 W8A8 workspace 审计仍未
回收；若 P1-M 落地，峰值还有 ~1–2 GiB 空间。

## 未覆盖

- 4 槽**全长** 256K 同时写满（需要 ~1M tokens 的 prefill，本次只增长到
  2048 tokens/槽；容量公式已由 8201 bundles 的分配保证，写满路径的
  COW/发布语义由 `tests/test_qwen36_dynamic_arena.py` 的 CPU 测试覆盖）。
- 服务级 HTTP 运行（本次直接走 backend API，等价于 ServerEngine 的
  runner 调用路径，但不含 admission/调度层）。
- MTP acceptance 在 256K 全长下的表现（属于 Phase 0 profile/K-sweep）。
