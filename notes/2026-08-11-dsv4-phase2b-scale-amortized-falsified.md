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
