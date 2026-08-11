# Phase 2B-0 scale-amortized microkernel：表示证明通过，性能前提被实测证伪

状态：**K-group=32 scale folding 数值通过（cos>=0.9999），但性能不达标且比
exact kernel 更慢**。本文记录 Nsight 证据，作为 Phase 2B 方向修正的依据。

日期：2026-08-11

## 1. 表示证明（真实 GGUF，blk.4，M=24，seed=20260811）

`tools/prescreen_iq2_kgroup_fold.py` 在真实 IQ2_XS 权重上验证文档 §4.3 的
direct-folding 公式（`qB = sign*round(mag*delta_j/sB)`，每 K-group 一次
I2F+FFMA）：

| K group | gate cos | 全链路(K/K) down cos |
|---:|---:|---:|
| 32 | 0.9999713 | 0.9999053 |
| 64 | 0.9999636 | 0.9998705 |
| 128 | 0.9999555 | 0.9998342 |
| 256 | 0.9999458 | 0.9997896 |

**K32 是满足 quality gate（>=0.9999）的最大可行 group**。表示证明通过。

## 2. CUDA microkernel（runtime/kernels/iq2_mma16_tc.cu）

实现：decode 时算 per-code delta 存 smem，fold pass 折入 qB，K-group=32 内
2×m16n8k16 INT32 累加，每 group 一次 `facc += float(cg)*sB*xs`。

数值（vs exact oracle，真实 GGUF）：gate cos 0.99999、up 0.99999、
down 0.99993 —— 全部通过。

修过的 bug：sB 必须按 C 输出行取（c0/c2→行 l4*2，c1/c3→行 l4*2+1），
与 exact kernel 的同类 bug 一致。

## 3. 性能（SM120，E=256 M_PAD=48，6144-route 形状）

| kernel | gate+up | down |
|---|---:|---:|
| exact（per-K16 scale） | 14.8 ms | — |
| **tc fold（K32）** | **24.0 ms** | 21.1 ms |
| Phase 2B-0 kill gate | <=2.4 ms | <=1.3 ms |

**tc fold 比 exact 慢 1.6×**，离 kill gate 差 10×。文档 §4.3 的假设
（"I2F 串行依赖是主瓶颈，scale folding 消掉后可达 6×"）**被实测证伪**。

## 4. Nsight 证据（tc kernel，E=256 M_PAD=48）

```
l1tex__throughput                    65.8%   (瓶颈)
l1tex__data_pipe_lsu_wavefronts      4.90G
sm__inst_executed.sum                6.03G
sm__inst_executed_pipe_alu           2.19G   (decode 位操作)
sm__inst_executed_pipe_fma           1.37G   (scale)
sm__inst_executed_pipe_tensor        0.10G   (mma，仅 1.7% 指令)
sm__throughput                       28.8%
occupancy                            24.9%  (smem 23168B → 3 blocks/SM)
```

关键结论：
- **tensor core 只占 1.7% 指令**。99% 的指令是 ALU(decode)/FMA(scale)/LSU(smem)。
- fold 优化 I2F 无意义：exact 的瓶颈同样在 L1TEX/指令，不在 I2F。
- 让 gate+up 到 2.4ms 需要 L1TEX 流量或总指令降 ~10×，必须根本减少
  decode/scale 指令，不是调整 I2F 位置。

## 5. 方向修正建议

1. **decode 指令必须消失或极简化**：6.03G 指令里 ALU 2.19G 是 IQ2 decode 的
   位操作（grid/ksigns 查表 + 移位 + 符号）。若能做到 **zero-decode**（权重
   在加载时预解码为 int8 codebook，常驻），L1TEX/ALU 大幅下降。但文档禁止
   resident W8A8（96GB 放不下 74.58GiB 权重放大）——需评估只解码**活跃
   expert** 的可行性。
2. **减少 LSU**：4.9G wavefronts 来自 smem 写读。更大的 N-tile/warp 或
   register-only decode 可减少。
3. **重新校准 kill gate**：exact 14.8ms 是当前组织的下限。2.4ms 需要不同
   的组织，不是微调。

## 6. 相关提交

- `cc3e35c` Phase 2B-0: 表示证明 + tc kernel（数值通过，性能不达标）

## 7. 优化迭代与 K32 mma 探索（2026-08-11 续）

从 24 ms 到 ~10 ms 的四轮实测优化（E=256 M_PAD=32 真实路由）：
1. M_PAD 编译期模板 → facc 寄存器驻留，消除 48% L2 的 register spill
   （24 → 16 ms）
2. direct-global decode（去掉 smem raw staging pass）：16 → 13 ms
3. fused decode+fold 单 pass：13 → 10 ms
4. m16n8k32 mma（每 K-group 1 条指令）：tensor 指令减半（100M→50M）但
   总时间反升（10 → 12.8 ms），A 片段 4 寄存器 + tensor 利用率 12%→5.7%，
   回退

下限测量：
- 纯 mma（无 decode/staging，const B）：4.72 ms
- nodecode（decode 用常量）：5.89 ms
- fold kernel：9.97 ms

**结论**：即使 decode/staging 完美消除，纯 mma + scale + sid 读已 4.7 ms，
仍超 kill gate（2.4 ms）2 倍。mma 指令发射本身（每 block 768 mma × 16384
blocks）是下限。two-plane（预解码 codebook）最多把 ALU 1.69G 砍到 ~0.3G，
预计 ~6-8 ms，仍超 2.4 ms。

**必须重新评估**：单卡 IQ2_XS Tensor-Core MoE 在 M_PAD=32、E=256、6144
routes 下，gate+up 的现实下限 ~5 ms（纯 mma）。kill gate 2.4 ms 需要 mma
指令量降 2 倍（更大 M 或不同 mma 形状）且 decode 近零。文档 §9 的
"两轮都失败则停止单卡 2K 承诺" 条件已接近触发；在转向 checkpoint format
前，剩余高 ROI 路径是增大每 block 的 N（行）以摊薄 decode，或接受
10 ms gate+up 并重新核算 6.5 ms/layer 预算。

## 8. N_ROWS=128 探索（2026-08-11 续）

尝试每 block 128 行（4 warp × 4 N8-tile）摊薄 decode 固定成本：

- v1（b_g[4][2]+sB[4][2] 全寄存器）：147 regs + 128B stack → 数值错（cos 0.5）
- **根因**：facc 缺 tile 维，4 个 tile 共享累加器 + scatter 同一值写 4 tile
- v3（tile 最外层循环 + facc[4][M][4]）：cos 0.99999 正确，但 216 regs + 512B stack，
  E=32 实测 2.79ms vs n32 1.39ms（0.5x）——寄存器压力 + smem 16K→64K 抵消 decode 收益

**结论**：增大 N 摊薄 decode 的方向失败。pure-mma 4.72ms 下限中 decode 非唯一瓶颈；
mma + sB 重算 + facc 伴随指令是主项。two-plane 预解码 + 预存 sB 是唯一能同时消
decode 和 sB 重算的路径，但其上限 = pure-mma ≈ 4.72ms（mma 指令量由问题形状决定，
two-plane 不改变），仍超 kill gate 2.4ms 约 2 倍。
