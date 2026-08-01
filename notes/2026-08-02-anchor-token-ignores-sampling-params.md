# 每个请求的首 token 恒为贪心 argmax，与 temperature 无关

日期：2026-08-02 · 状态：🟢 已核实，未修复 · 零 GPU（纯读代码）

## 结论

**生产路径上，每个请求生成的第一个 token（"anchor"）都是无约束的 `argmax`，
不查客户端的 `temperature` / `top_p` / `top_k`。**

`temperature=1.5` 的请求，首 token 与 `temperature=0` 的请求逐位相同。

## 调用链（逐环核实，非推断）

```
server/engine.py:1117   prefill_chunked_begin(slots, prompts_per_slot, chunk_size)
                        └─ 签名里没有 SamplingParams
runtime/backends/laguna.py:2272
      prefill_chunked_begin
        ├─ DFlash 开：  laguna_dflash.py:1679  dflash_prefill_bootstrap(slot, prompt_ids, *, prefix_hit=0)
        │                 └─ 签名里没有 SamplingParams；bonus_token = first_token
        └─ DFlash 关：  laguna.py:1698         prefill_with_aux(slot, prompt)
                          └─ laguna.py:1818    first_token = int(logits[-1].argmax(dim=-1).item())
        → result[slot] = {"anchor": first_token, "draft_tokens": [...]}
```

`laguna.py` 里共有六处 `first_token = ... argmax(...)`（1679 / 1818 / 1954 / 2637 / 2755，
及 2035 的 `next_token`），**没有一处查采样参数**。

## 存在一条采样版本，但它是死代码

`laguna.py:1684` 有 `prefill_sampled(slot, prompt_ids, params: SamplingParams)`，
`laguna.py:2514` 里 `generate()` 会在 `temperature != 0` 时调它：

```python
if temperature == 0:
    first = self.prefill(slot, prompt_ids)
else:
    first = self.prefill_sampled(slot, prompt_ids, params)
```

**但 `generate()` 在 `server/` 与 `runtime/` 里零调用方**——它是独立/遗留方法，不在
`ServerEngine` 的准入路径上。所以 `prefill_sampled` 实际也是死代码。

⚠️ 这正是本仓库反复出现的形状：**一个正确的实现存在，但生产路径没接到它**
（对照 `block_pool.py` 的 `BlockPool` —— 44 个通过的测试、零生产调用方；
以及 N8 的 `mtp_prefill_warm_continue` —— 调用了另一个已截肢子系统的方法）。
测试全绿、代码存在，都不等于生产路径在用它。

## 影响

- **所有 `temperature>0` 的用户**：输出的第一个 token 是确定性的。补全越短，
  这个 token 占比越大。
- **同 prompt 多次采样**：每次的首 token 都相同，多样性从第二个 token 才开始。
- **不影响** `temperature=0`：贪心本来就该是 argmax，行为正确。

## 与 E-N1（结构化输出）的关系 —— 这条使原方案 (b) 不成立

`docs/e2e-and-quality-plan.md` §2.3 原本给出两个选项，其中 (b) 是
"只接受 `temperature>0` 的结构化输出请求"。

**(b) 按原样不成立**：anchor token 的无约束与 temperature 无关，所以放开
`temperature>0` **并不能约束首 token**。而对 JSON 来说首 token 恰恰是 `{`——
是最需要被约束的那一个。放开后客户端会拿到一个"看起来结构化输出生效了"的响应，
而首 token 从未被约束——正是 `server/app.py:534` 那段拒绝逻辑存在的理由，
只是换了个更窄的位置藏起来。

`server/app.py` 的 docstring 其实已经写明了这点（"Wiring only the narrow reachable
slice (temperature > 0, decode tokens 2+)"），只是 e2e 计划 §2.3 转述成选项时把
"decode tokens 2+" 这个限定丢了。

## 建议的处置

**先修 anchor token 走采样参数，它是两个选项共同的前置**，且**独立于结构化输出
就有价值**（今天这是一个真实的采样正确性缺陷）。修完之后 E-N1 的选项才是真的二选一。

具体：让 `prefill_chunked_begin` / `dflash_prefill_bootstrap` 接收并使用
`SamplingParams`。`prefill_sampled` 已经有一份可抄的实现，但**不要直接复活
`generate()`** —— 它是遗留路径，接的不是今天的准入流程。

## 相关

- `docs/e2e-and-quality-plan.md` §2.3（E-N1，本笔记使其 (b) 选项作废）
- `docs/api-layer-design.md` §7.1（N1 三条不可达路径的原始记录）
- `server/app.py:534` `_reject_unsupported_response_format`（现行拒绝逻辑与理由）
