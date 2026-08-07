# DSV4-Flash Phase 3 设计备忘 · 三域槽位池（dsv4_slots）

状态：设计稿（Phase 3 第 1 件，先于所有 kernel）。
依据：实施计划 Phase 3 §1-2、事实基线 §2/§6/§9、`runtime/model/qwen36_slots.py`
先例（ allocate-once / never-rebind / reset 显式清零 / geometry 上报）。

## 1. 层分类（43 层，config 驱动）

| 类别 | 层号 | 数量 | 缓存成分 |
|---|---|---:|---|
| WIN（ratio 0） | 0, 1 | 2 | 仅窗口环 |
| CSA（ratio 4） | 偶数 2..42 | 21 | 窗口环 + 压缩区(seq/4) + indexer 区(seq/4) |
| HCA（ratio 128） | 奇数 3..41 | 20 | 窗口环 + 压缩区(seq/128) |

窗口环 = 128 条 512 维 latent；压缩区条目同 512 维 latent；
indexer 区条目为 128 维（indexer compressor 的 head_dim）。

## 2. 每槽缓存区域与布局

五个独立池（形状不同，不强行统一 —— 与 Qwen36SlotPool "分离分配器 +
协调层" 的结论一致）：

| 池 | 形状（S 槽，L=max_seq） | dtype | 每槽字节 |
|---|---|---|---|
| window | [S, 43, 128, 512] | KV 布局(§3) | 43·128·entry |
| csa_comp | [S, 21, L/4, 512] | KV 布局 | 21·L/4·entry |
| hca_comp | [S, 20, L/128, 512] | KV 布局 | 20·L/128·entry |
| idx_k | [S, 21, L/4, 128] | bf16 | 21·L/4·256 |
| comp_state | 每槽 fp32 定长 | fp32 | 11.2 MB（§2.1） |

### 2.1 压缩器 decode 状态（每槽定长，reset 必须清零）

- CSA 层 attn compressor（overlap，coff=2，ratio=4）：
  kv_state/score_state 各 [8, 1024] fp32 → 21 层 × 2 × 32 KB = 1.37 MB
- HCA 层 attn compressor（coff=1，ratio=128）：
  各 [128, 512] fp32 → 20 层 × 2 × 256 KB = 10.5 MB
- indexer compressor（coff=2，ratio=4，head_dim=128）：
  各 [8, 256] fp32 → 21 层 × 2 × 8 KB = 0.34 MB

合计 ~12.2 MB/槽。这些是**递推状态**：与 Qwen36 GDN 同理，reset_slot
必须显式 zero（score_state 置 -inf），不能只标记。

## 3. latent KV 条目布局（两种）

- **bf16 参照布局**（Phase 3 初期/kernel 对拍用）：512×2B = 1024 B/条。
  与 eager 图当前数值逐位一致，用于 kernel 上线前的逐位对拍。
- **FP8 混合生产布局**（计划 Phase 3 §2，与 eager 的 act_quant_simulate
  语义一致）：nope 448 维 → e4m3 字节 + 每 64 维一个 ue8m0 scale 字节
  （7 B），rope 64 维保持 bf16（128 B）→ **583 B/条**（0.57×）。
  写入 kernel 必须复现 `act_quant_simulate(x, 64, ue8m0=True)` 的量化
  语义（dsv4_attention.py 是 executable definition），不得自创新舍入。

窗口环是否也用 FP8 布局：窗口条目同样经过 act_quant_simulate（attention
kv 路径），**是**，同布局。

## 4. 显存预算与并发上限（2026-08-07 精确计算）

条目字节 entry(bf16)=1024，entry(fp8)=583。每槽 =
43·128·e + 21·L/4·e + 20·L/128·e + 21·L/4·256 + 12.2 MB。

| 上下文 | bf16/槽 | fp8/槽 |
|---:|---:|---:|
| 16K | 0.122 GiB | 0.083 GiB |
| 128K | 0.856 GiB | 0.563 GiB |
| 160K | 1.066 GiB | 0.700 GiB |
| 256K | 1.696 GiB | 1.112 GiB |

可用预算：95.6 GiB 卡 − 权重常驻（packed 81.9 GiB + CUDA/scratch ~2-3 GiB，
llama.cpp 2K 冒烟实测总占用 85,909 MiB 佐证）≈ **KV 预算 ~10.5-11.5 GiB**。

| 用户目标 | fp8 需求 | 判定 |
|---|---:|---|
| 10 × 256K | 11.1 GiB | **可行（贴边）** |
| 20 × 156K(160K) | 14.0 GiB | 不可行；上限 ~15×160K 或 20×128K(11.3 GiB) |

进一步压缩的候选（未决策）：indexer 区 bf16→fp8（省 ~0.16 GiB/槽 @256K）、
rope 段 fp8（质量风险，需对齐实验）。**10×256K 贴边成立的前提是生产布局
必须上 FP8 混合**；bf16 布局只作对拍参照，不用于生产容量规划。

## 5. API 草案

```python
class Dsv4SlotPool:
    def __init__(self, config, num_slots, max_seq_len, *, layout="bf16", device): ...
    def slot_window(self, slot) -> Tensor      # [43, 128, 512] 视图
    def slot_csa_comp(self, slot) -> Tensor    # [21, L/4, 512]
    def slot_hca_comp(self, slot) -> Tensor    # [20, L/128, 512]
    def slot_idx_k(self, slot) -> Tensor       # [21, L/4, 128]
    def slot_comp_state(self, slot) -> CompState  # 数据类，各状态张量视图
    def reset_slot(self, slot) -> None         # comp_state 清零/-inf；KV 不清（前缀复用）
    def geometry(self) -> Dsv4SlotPoolGeometry  # 上报 /metrics
```

- 静态分配：slot s 的区域切片终身不变（同 Qwen36/Laguna 先例），
  视图经 `.narrow()` 派生，不新建存储。
- kernel 消费契约：注意力 kernel 收到的是**每槽视图 + 序列长度**，
  gather 索引（window_topk ∪ compressed idx）由 attention 接线层生成。
- CUDA Graph：所有池地址 mark_static；decode 图按"定形不定值"纪律捕获。

## 6. 与 eager 图的关系

eager 图（`Dsv4Transformer`，模块自带缓存）保留为**数值参照**：Phase 3
每个 kernel 上线都要过"池+kernel vs eager"逐位/容差对拍，正如 Phase 2 用
官方 reference 对拍 eager。Backend 走池路径，eager 路径不进服务。

## 7. 未决项

1. indexer 区量化（bf16 vs fp8）——等对齐 harness 有数据后做质量实验。
2. 压缩区页化粒度（连续条 vs 256 条目页）：llama.cpp 用连续条；我们初版
   也用连续条（gather kernel 按索引取），页化留性能迭代。
3. ratio-128 层的 compressed idx 是顺序全取（compress_topk_idxs），
   kernel 侧可做连续读优化分支。
