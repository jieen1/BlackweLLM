# DSV4 CUDA-Graph decode 驱动：b12x 合并、捕获成功、compressor 状态机 bug

日期：2026-08-09

## 背景

上一阶段 decode 已优化到 ~254ms（fused Q8_0 + wo_a grouped + HC fused，见
`2026-08-09-dsv4-serving-e2e-and-bf16q-prefill-bug.md`）。本轮目标：**消除 decode
的 CPU launch 开销**（profiler 显示 GPU 实际只忙 92ms，~160ms 是 CPU 侧 launch），
并顺带完成 sparkinfer fork 的上游合并。

## 完成的工作

### 1. SparkInfer fork 合并上游（已推送到 origin/master）

上游 master 18 个提交合并进 fork master（`3d353ff`），其中 `e65e9d6` 把包目录
**`sparkinfer/` 重命名为 `b12x/`**（230 文件）。runtime 侧 46 处 `import sparkinfer.*`
全部改为 `import b12x.*`（提交 `a997e1d`），函数改名同步处理：

- `plan/prepare_sparkinfer_fp4_moe_weights` → `plan/prepare_b12x_fp4_moe_weights`
- `sparkinfer_moe_fp4` → `b12x_moe_fp4`
- `is_sparkinfer` → `is_b12x`
- fork 独有模块（`gated_delta_rule`、`tensor_fp8_channel_linear`）移入 b12x 树

全量测试 **1810 passed, 34 skipped**，lint 干净。

### 2. DSV4 work 分支 rebase 到 b12x

`work/dsv4-bf16q-20260807`（4 个 bf16-Q/PV 实验提交）rebase 到新 b12x master，
**0 冲突**（git 自动识别目录重命名），已推送。服务用缺省 kernel（bf16_q=0），
不依赖该分支。

### 3. CUDA-Graph decode：43 层捕获成功，eager 311ms → CG 136ms（2.3×）

新增 capture 基础设施：

- `pack_latent_kv(validate_ids=False)`：跳过 `.item()` bounds check（GPU→CPU 同步，
  捕获中非法）。`runtime/kernels/dsv4_kv_pack.py`
- **MoE decode 分支去掉 `.tolist()`**：expert gather 改 GPU 索引
  `packed.reshape(-1, rb)[eids]`（原 `eids = indices[0].tolist()` 是每层每 token 的
  GPU→CPU 同步，也是 CPU 开销来源）。`runtime/model/dsv4_model.py`
- **decode 索引 GPU kernel**：`runtime/kernels/dsv4_decode_indices.py`——`decode_swa_
  indices`（窗口环序）与 `decode_comp_indices`（压缩条目）从 GPU `pos` 标量生成，
  与 eager `window_topk_idxs`/`compress_topk_idxs` 位级一致（pos>=1 全部验证）
- **compressor `forward_graph`**：GPU `pos` 驱动的 capture-safe decode 步，
  用 mask 替代 Python 分支（`should_compress`、slot、state 迁移）
- **indexer `forward_graph`**：固定-k topk + -inf mask
- `Dsv4AttnKernelLayer.forward` 支持 `pos_tensor`（GPU pos 读 `freqs_cis`）与
  `capture` 标志

**速度验证**：完整 43 层 decode 捕获成功，replay 136ms vs eager 311ms（连续跑时
eager 更快些但 CG 仍 2.3×）。

## 已知 bug（本轮未修）

**compressor 连续多步后 `score_state` 迁移产生 nan**。单步对比（pos 40-43 各自
独立从 0 状态跑）eager vs graph **完全一致**（diff=0）；但连续多步（40→41→42→43
一步接一步）graph 的 `score_state` 前 4 行出现 nan，kv_cache 写入 nan。

定位进展：`score_state[:, :ratio] = score_state[:, :ratio]*(1-sc) + score_state[:,
ratio:]*sc` 的迁移在非 compress 步（sc=0）手动复现也出现 3072 个 nan（用独立
测试脚本手动逐行执行同样操作，`scs nan after migration: 3072`）。根因**未确定**——
怀疑与 `kvs[:, idx] = ...` 的 GPU advanced-index in-place 写 + `score_state` 初始
`-inf` 的 softmax 传播有关，或迁移公式的 `(1-sc)`/`sc` 广播在 `-inf` 上产生
`0 * -inf = nan`。

> 注：`sc * (-inf)` 在 sc=0 时是 `0 * -inf = nan`！迁移公式
> `scs[:, :ratio]*(1-sc) + scs[:, ratio:]*sc` 中，`scs[:, ratio:]` 含 `-inf`
> 元素（未写入的 slot），`sc * (-inf)` 当 sc=0 时为 `0*(-inf)=nan`，
> 但该项与 `(1-sc)*scs[:ratio]` 相加——nan 会传播。这是根因。修复方向：
> 迁移公式应改为 `torch.where(should_compress, scs[:, ratio:], scs[:, :ratio])`
> 而非乘法 mask，避免 `0*(-inf)`。

## 下一步

1. **修复迁移 nan**：`torch.where` 替代乘法 mask（见上注）
2. 修复后重跑多步正确性验证，达到 eager/graph 逐位一致
3. 实现完整 `Dsv4DecodeGraphDriver`（预分配 buffer + capture/replay 生命周期），
   接入 `DeepseekV4Backend` decode 路径
4. 端到端质量验证（cos + argmax + 服务器实测）
