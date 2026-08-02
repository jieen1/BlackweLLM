# FP8 W8A8 也走不通：**误差下界都过不了 B1-R**（负面定案）

日期：2026-08-03 · 模型：`unsloth/Qwen3.6-27B-NVFP4`（标准模型）·
分支 `work/fp8-preflight-20260803` @ `6e97958`

## 结论

**不可用。** 阶段四最后一根杠杆（FP8 层 45% 的解码 kernel 时间）**关掉了**。

## 这次没有先写 kernel——这是本文最值得复用的部分

W4A4 那轮是**完整实现完才发现过不了**
（[`2026-08-03-w4a4-blockscaled-negative-result.md`](2026-08-03-w4a4-blockscaled-negative-result.md)）。
这次改成**先做判据预演**：

今天 FP8 权重被**精确**反量化成 BF16，`F.linear(x_bf16, w_bf16)`。真 W8A8 会把激活也
量化成 FP8、做 FP8×FP8 并 FP32 累加。**权重侧的值两种设计下相同**（checkpoint 本来就是
FP8，今天是精确反量化），所以 **W8A8 相对现状的新增误差，主导项就是激活量化那一步**。

于是不必写 kernel，只需在现有 forward 里插一段往返：

```python
x_rt = dequantize_fp8(quantize_fp8_per_token(x))   # 按 checkpoint 自己的
out  = F.linear(x_rt, w_bf16)                      # group_0: per-token, dynamic
```

**这是真实 W8A8 误差的下界**（真实版还要叠加累加顺序差异、FP8×FP8 的乘积舍入）。
所以"下界都过不了"⇒ **真实实现必然也过不了**，一整轮 kernel 实现被省掉。

**并且验证了它不是空操作**——空操作会给出毫无意义的 PASS：约 **93% 的激活元素被改变**，
最大绝对变化 ~0.003–0.004。

## 数字

**单层**（真实 checkpoint 权重，三种形状 `self_attn.q_proj` / `linear_attn.in_proj_qkv` /
56–63 层的 `mlp.gate_proj`，M=1..512）：cosine ≈ **0.9996**。

**B1-R 全模型**（3 负载 × 65 步，oracle = 今天的生产 forward，candidate = 同一 forward
+ 仿真，强制走 oracle 自己的 token）：

| 指标 | 实测 | 判据 | |
|---|---:|---:|---|
| `median_gap_error` | 0.25 | 0.25 | **贴着判据，零余量**（干净跑约 0.125） |
| `p90_gap_error` | 0.375 | 0.5 | 在线下 ⚠️ |
| `p90_logprob_error` | 0.3686 | 0.5 | 在线下 ⚠️ |
| 最差负载 `mean_kl_topk` | **测不出来** | 5e-3 | 见下 |
| `disagreement_rate` | 0.0385 | 0.03 | 超（该 bar 自述"宽松、非硬判据"，单独不承重） |

⚠️ **那两个"在线下"必须带着下面这句读，否则会被误读成"只差一点"：**

**`instruction` 负载的 candidate 轨迹发散到溢出了诊断自己的 top-1024 捕获窗口，
根本没产出任何数字**——它在进入统计前就被跳过了。**所以上表每一个 bar 都是在
"排除掉最差负载"之后算出来的。** 发散到不可测量，比超过某个 bar 更硬：
B1-R 自己的校准扫描里，**任何注入 bug 都没有产生过这种情况**（全部落在 top-64 以内，
`docs/b1-correctness-criterion.md` §5.3）。

**这与 W4A4 是同一个失败签名。**

### 顺带修掉一个会造成假绿的缺陷

判据脚本原先把 `passes_calibrated_bars` **只在存活负载上**判定。也就是说：重跑时若
1 个负载发散到不可测、另外 2 个都在 bar 内，它会记录 `passes: true`——
**门禁存在的目的（抓这种发散）反而制造了绿灯**。已修（`6e97958`）：
任一负载溢出捕获窗口即总体判失败，并在 reasons 里写明"上面的 bar 只在幸存者上算过"。

## 一处对我给出事实的纠正（值得单独记）

我下发任务时引用的"FP8 W8A8 单层 cosine 0.9996"，出自
`scripts/verify_fp8_tensor_gemm_single_layer.py` /
[`2026-08-03-nvfp4-raw-param-free-and-fp8-w8a8-probe.md`](2026-08-03-nvfp4-raw-param-free-and-fp8-w8a8-probe.md)
——**那是在 `nvidia/` checkpoint 的静态 per-tensor FP8 方案上测的**
（`dynamic: false`，每模块一个标量 `input_scale`），**不是**标准 checkpoint 的
`CompressedTensorsFP8ChannelLinear`（per-channel 权重 scale + per-token **动态**激活 scale）。
不同 checkpoint、不同 Linear 类、不同方案。重新在真正相关的层上测，独立落到同一个
~0.9996 量级，**所以结论不受影响**——但那个数此前从未在这个 checkpoint/方案上验证过。

🔴 **这是今天第二次同型错误**：第一次是我把 `blockscaled.mm` 在 `nvidia/` 上的失败
推广成"所有 checkpoint"（见 roadmap 阶段 4 的纠正）。
**在这个仓库里引用任何量化相关的实测数字之前，先确认它测自哪个 checkpoint。**
两个发布方的方案是真的不同，不是同一份权重的两次量化。

## 留下什么

- `emulate_fp8_activation_round_trip()`（`runtime/model/compressed_tensors_linear.py`），
  由 `QSR_EMULATE_FP8_ACTIVATION` 控制，**默认关**，且**每次调用时读取**
  （不能被陈旧的 shell 变量卡住，也不会漏进生产）
- `scripts/verify_fp8_w8a8_activation_emulation_{single_layer,full_model_gap}.py`
- `tests/test_fp8_w8a8_activation_emulation.py`（7 例，CPU-only，钉住默认关 + 往返非空操作）

**没有测速度**——正确性已不过，测速只会变成放宽判据的诱因（W4A4 那轮的明确教训）。

## 阶段四的处境

两根杠杆（合计 80% 的解码 kernel 时间）**都试过、都不可用**：

| 杠杆 | 占比 | 结论 |
|---|---:|---|
| NVFP4 → W4A4 | 35% | ✗ B1-R 全线不过，一个负载溢出捕获窗口 |
| FP8 → W8A8 | 45% | ✗ 误差下界即溢出捕获窗口 |

**共同点：两者都是"把激活也降到 4/8 bit"。** 这个模型对**激活精度**很敏感，
而对权重量化不敏感（现有 W4A16/FP8-权重 路径 cosine 0.99999）。
**下一步不该继续找"更激进的激活量化"，那个方向已经两次撞墙。**
仍未探索的是不降激活精度的路子——例如给这些 BF16 GEMM 换 Blackwell 原生 kernel
（目前 24.8% 的 kernel 时间跑在为 SM80 编译的 `cutlass_80_wmma` 上，
见 [`2026-08-03-decode-kernel-profile.md`](2026-08-03-decode-kernel-profile.md)），
以及回收 FP8 反量化缓存那 9.99 GiB
（见 [`2026-08-03-production-memory-audit.md`](2026-08-03-production-memory-audit.md)）。
