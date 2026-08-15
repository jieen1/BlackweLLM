# Qwen3.8 128K：W4A4-all vs W4A16 小 M 路由 A/B（Phase 1R-1，2026-08-16）

状态：🟢 **实测定案——保留 all-W4A4 默认**。规划 §6.3 第 1 项明确"Qwen3.8
128K 的最优点不能继承 Qwen3.6 旧结论"，本 A/B 就是那个实测。

## 前置修复

`QSR_QWEN36_MLP_W4A4_ALL=0`（W4A16 小 M 路径）原本**不可服务**：
`_forward_w4a16_fused` 没把 `buffers.expert_counts` 传给 `run_w4a16_moe`，
route-packing 在 graph capture 里拒绝临时分配 → MTP verify 捕获失败 →
eager 退化 → 128K B1 只有 15.8 tok/s。修复（`116ec2c`，一个参数）后
verify/draft/sync 全部捕获，路径可用。

## A/B 数据（128K，K=3，两边接受率均 100%，同 prompt）

| 配置 | B1 warm | B4 warm（每请求） | 显存代价 |
|---|---:|---:|---|
| W4A4-all（默认） | ~109 tok/s（108.0-110.3，n≥3） | 66.3-66.4 tok/s（agg ~235-245） | 基线 |
| W4A16 小 M（ALL=0） | 82.5 tok/s（**−24%**） | 69.75-70.04 tok/s（agg ~256，**+5%**） | **+7.88 GiB**（W4A16 rep） |

## 判定

1. **B1 大幅落后**：单请求场景 W4A16 慢 24%，Qwen3.6 时代"W4A16 小 M
   更快（830 vs 330-440 GB/s）"的结论在 Qwen3.8 + CUTLASS 4.7
   blockscaled kernel 上**不成立**。
2. **B4 仅 +5%**：且方向上 B1/B4 交叉（M=1 与 M=4 的 kernel 优劣相反），
   混合路由（按 batch 分路径）理论可行但复杂度不低。
3. **显存代价与项目目标冲突**：+7.88 GiB 会吃掉 4×256K 验收留下的
   31.5 GiB 余量的四分之一——Phase 1M 的全部工作就是为了腾出这个空间，
   拿它换 5% B4 吞吐是本末倒置。

结论：**默认保持 all-W4A4**；W4A16 路径保留为可工作的诊断对照
（`QSR_QWEN36_MLP_W4A4_ALL=0`），不再作为生产候选，除非未来 kernel
格局变化（新 blockscaled/W4A16 kernel）时重测。

## 证据文件

- `/tmp/opencode/phase0/server_perf_grid_qwen38_dynamic_128k_c1_w4a16fix_*.json`
- `/tmp/opencode/phase0/server_perf_grid_qwen38_dynamic_128k_c4_w4a16fix_*.json`
- W4A4 基线：同日 `packed_c1*`/`base2_c1*`（B1）与 `k3np_c4`（B4）fixture
- 修复前失败证据：`/tmp/opencode/phase0/server_w4a16.log`（verify capture
  RuntimeError: expert_counts not initialized）
