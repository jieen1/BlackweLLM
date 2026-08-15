# b12x packed f32x2 score arithmetic（P0-A2）：SASS 证伪（2026-08-15）

状态：🔴 **负面定案**——按规划 §7.2 的淘汰条款（"若 DSL 拆回标量…淘汰"）回退。

## 实验

在 `b12x/attention/paged/forward_paged.py` 的两个 row0 decode 变体
（`_literal_update_mdo_states_fp32_pack_p_row0` / `_row0_1x1`）把逐元素
`(s - m_new) * sm_scale_log2`（FSUB+FMUL）替换为
`cute.arch.fma_packed_f32x2((s0,s1), (scale,scale), (bias,bias))`（FA4
`scale_subtract_rowmax` 的同构移植）。worktree
`/home/bot/project/spark-w-rescale`（branch
`work/p0a1-exact-rescale-20260815`，未提交该改动；同分支的 exact
conditional rescale 已合并，见 sparkinfer `8f74740`）。

## 证据

1. **SASS 无 packed 指令**：ncu 对 `test_run_decode_fp8_kv_matches_reference`
   （GQA8 decode FP8，命中被改的 row0 路径）dump PagedForwardKernel SASS：
   patched 与 unpatched 均为 **0 条 FFMA2/FMUL2/FADD2**。DSL 把
   fma_packed_f32x2 标量化成逐元素 FFMA。
2. **标量指令数几乎不变**：patched FMUL+FADD+FFMA = 289+17+199 = 505，
   unpatched = 292+20+195 = 507——净差 2 条，远小于理论省下的 4 条/调用，
   说明编译器已把原式部分融合，packed 移植没有指令面收益。
3. **性能在噪声内**：128K B1 warm decode（n=2/变体）：rescale-skip 基线
   107.93/109.57，packed 110.24/110.27 tok/s（+1.4%，小于 ±3-5% 跑间方差）。
4. **数值移动**：packed 版改变舍入（fused vs 两步），smoke plain decode
   60 token 中 3 个平局翻转（MTP 流 0/60）——无系统偏差（51 个 b12x
   reference 测试全过），但既然无收益，移动数值就是纯成本。

## 结论

SM120 的 CuTe DSL（cutlass-dsl 4.7.0）不会把 f32x2 packed intrinsic 落到
Blackwell 消费卡的 packed FP32 指令；该路径的收益前提不成立。P0-A2
从实施清单移除。若未来 DSL 支持 packed 发射（或手写 CUDA/PTX 路径），
可用本 note 的 SASS 对照方法复验。
