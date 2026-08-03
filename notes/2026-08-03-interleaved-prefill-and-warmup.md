# 跨步交织 prefill + 全前向 warmup：两项实测收益

日期：2026-08-03 · 模型：`unsloth/Qwen3.6-27B-NVFP4`（标准模型）·
capacity=2 / num_slots=3 / CUDA Graph 开 / FP8 KV 默认开 ·
单卡 RTX PRO 6000 Blackwell Max-Q

两项都不是新设计——**都是照着 `oracle/qwen36_vllm/` 里已经跑通过的实现写的**。

## ① 首请求 TTFT：4.67s → 0.538s

| | 首请求 | 第二请求 | 落差 |
|---|---:|---:|---:|
| 修复前 | **4.67 s** | 0.25 s | **18×** |
| 修复后 | **0.538 s** | 0.499 s | **1.08×** |

**根因**：`warmup_attention_shapes` **只暖 attention kernel**（其 docstring 自陈，
理由是 GDN 递归状态顺序相关、真实前向会留下暖过的状态）。于是
**w4a16 融合 MoE（56 层 NVFP4 MLP，含一次性融合权重准备）、GDN 递归 kernel、
`lm_head` 全是冷的**，由第一个真实请求买单。

**答案在历史代码里**（`oracle/qwen36_vllm/direct_model_runner.py:860`）：

```python
def _warmup(self):
    try:
        self.prefill(0, [0, 0, 0, 0, 0])   # 真实全模型前向
    finally:
        self.reset_slot(0)                 # 清零递归状态
```

**那个 GDN 顾虑正是被 `reset_slot` 解决的**——清零递归状态是 B0-5 给出
"捕获安全"结论时附的唯一运行要求。所以真实前向安全，只要之后 reset。

## ② 跨步交织 prefill：长 prefill 不再饿死在跑的请求

60,000 token 的 prompt，在另一个请求已经稳态解码时提交：

| | 最大停顿 | 长 prefill | 并发 ITL 中位数 | 期间产出 |
|---|---:|---:|---:|---:|
| one-shot（原状态） | **24,939 ms** | 25.7 s | 35.5 ms | 18 tok |
| chunk 512 | 688 ms | 56.1 s | **366.8 ms** | 123 tok |
| **chunk 2048（采用）** | **1,225 ms** | **30.9 s** | **35.5 ms** | 46 tok |

**短请求跑完 220 token 的总时长**：one-shot ≈ 33s，chunk 2048 ≈ **9s（3.7×）**，
chunk 512 ≈ 80s。

### 块大小是真正的旋钮，而且我一开始搞错了

引擎侧的状态机一直是完整的（`_pending_prefill`、每轮推进一块、`done` 前不激活），
但 `prefill_chunked_begin` 自称 "one-shot"、丢弃 `chunk_size`、永远返回 `done=True`
——**整条分支是不可达的死代码**。

改成真正交织后，第一版沿用引擎传来的 `chunk_size=512`：**停顿确实塌了 36×，
但 60K prompt 从 8 次前向变成 118 次，prefill 慢 2.2×、并发 ITL 差 10×，
总账反而更差（33s → 80s）。**

按实测推导：prefill 吞吐 60000/25.7s ≈ **2335 tok/s**，把单块限制在约一秒的
prefill 得 **2048**。实测证实：停顿 1.2s、prefill 只慢 20%、
**并发 ITL 与 one-shot 完全相同（35.5 ms）**。

⚠️ **这修正了本仓库既有的一条判断，也修正了我自己引用它的方式**：
`notes/2026-07-20-comprehensive-optimization-plan.md` 记录"块内分块（Phase A）
只值 −10.7%"——**那是对的，但它说的是不交织的情况**。我拿它否掉过块大小这个维度。
**一旦有了交织，块大小恰恰决定交织到底划不划算。**

## 门禁

`tests/test_qwen36_interleaved_prefill.py` 钉的是**结构性质**而非输出：
退回 one-shot 时输出逐 token 相同，唯一症状是别人等多久——没有任何常规测试会发现。
同时钉住第一版丢掉的两个守卫（脏槽位 / 空后缀），
丢掉前者会让 GDN 从**另一个序列的递归状态**继续：不报错、不 NaN，只是错误的续接（INV-A3-1）。
