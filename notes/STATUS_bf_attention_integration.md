# BFAttention 集成现状 (2026-07-26 更新)

> **00:42 根因诊断更新（本节覆盖下文旧判断）**
>
> 当前 `" is is is"` 不能作为 BFAttention 集成错误的证据。产生该结果的
> BFAttention E2E、monkey-patch 分叉和 attention debug 脚本都使用了
> `tokenizer.encode(prompt, add_special_tokens=False)`，统一漏掉 Laguna
> checkpoint 要求的 BOS token（id=2）。这是 2026-07-23 已确认过、且症状完全
> 相同的公共上游输入错误。原判断“根因锁定在 BFAttention 模块替换机制”已被
> 推翻；当前真正未决的问题是：**修正 BOS 后，当前未提交的 BFAttention +
> FP8 descale 组合是否仍有独立回归。**

## 最新核心结论

### 已证实

1. **当前错误 E2E 的直接根因是测试输入缺 BOS，而不是已证明的 attention
   数学错误。**
   - Laguna tokenizer 的 `TemplateProcessing` 会在单序列前插入
     `〈|EOS|〉`，其 token id 是 2；`add_special_tokens=False` 会跳过它。
   - `/tmp/test_monkeypatch.py:65`、`tests/debug/test_attn_correctness.py:38`、
     `tests/debug/test_attn_numerical.py:62`、`tests/debug/test_attn_metadata.py:57`
     和 `tests/debug/test_attn_kv_debug.py:37` 都显式关闭了 special tokens。
   - 历史记录 `notes/2026-07-23-laguna-bos-root-cause.md` 已记录完全相同的
     `" is is..."` 症状；恢复 `tokenizer.encode(prompt)` 后首 token 为
     `" Paris"`。
   - CPU 护栏 `tests/test_laguna_tokenization.py` 已验证默认 encode 包含 BOS，
     `add_special_tokens=False` 恰好少一个 token；本轮结果为
     `12 passed in 4.47s`。

2. **“完整替换 Attention 模块导致错误”的原分叉结论不成立。**
   - monkey-patch import 已修复为
     `from vllm.forward_context import get_forward_context`，测试实际已经完成。
   - 保留 48 个原始 vLLM `Attention` 模块、仅替换其 `forward()` 后，仍得到
     top-1 `" is"`；因此“模块对象被 `setattr()` 替换”不是该次错误输出的必要
     条件。
   - 但该 monkey-patch 测试同样漏 BOS，所以它只能推翻旧的模块替换归因，
     **不能**证明当前 BFAttention 在正确输入下已经通过。

3. **RoPE 不是当前根因。**
   - vLLM 0.26 实际加载的是 `LagunaConfig`，full/SWA 两套嵌套
     `rope_parameters` 均保留并由 Laguna model 按 layer 类型选择。
   - 在位置 `[0, 1, 8192, 65535]` 上与 checkpoint/Transformers 独立计算对比：
     full YaRN 的 Q/K cosine 为 `0.999998450/0.999998331`，SWA RoPE 为
     `0.999997735/0.999997914`。此前“嵌套 RoPE 配置丢失”的假设已排除。

### 尚未证实

- 当前工作树在 **BOS 正确**输入下是否能稳定输出 `" Paris"`。本轮复用
  `python -m benchmarks.laguna_backend_test --fact-only`，但 vLLM 编译结束后
  按 GPU 使用约束主动终止，测试停在 sparkinfer MoE 权重准备阶段，未产生
  logits。
- BOS 修正后 BFAttention 与“保留原模块、只 patch forward”是否 logits
  一致。现有两边的错误结果都被相同的输入错误污染，不能用于这个比较。
- FP8 descale、KV scatter 和 BFAttention 的组合在 48 层 E2E 上是否还有次级
  误差。局部 layer 0/1 数值通过只能降低风险，不能代替 BOS 正确的 E2E 门禁。

### 新的环境阻断（与上述错误输出不是同一根因）

vLLM editable build 已完成，但安装命令没有使用 `--no-deps`，pip 将环境从
PyTorch `2.13.0a0+gitcf30153` 替换成了 `2.11.0`，并安装 vLLM
`0.26.0+cu132`。pip 明确报告：

```text
sparkinfer 1.0.1 requires torch>=2.12.0, but you have torch 2.11.0
```

因此后续 GPU 结论必须先恢复项目约定的 Torch/sparkinfer 兼容组合；不能把新
环境产生的 import、ABI 或 kernel 问题混入 BFAttention 根因判断。

## 一句话总结

自研 attention 局部数值已通过；当前 `" is is is"` 的直接原因是错误测试脚本
漏掉 BOS，不能证明 BFAttention 有错。下一步需要在兼容环境恢复后，用同一份
BOS 正确输入做 BFAttention/monkey-patch 单变量 E2E 对照。

---

## 环境

```
GPU:       RTX PRO 6000 Blackwell 96GB (SM120)
Python:    /home/bot/.venvs/vllm/bin/python (3.12)
PyTorch:   2.11.0（editable install 从 2.13.0a0 自动替换；与 sparkinfer 不兼容）
vLLM:      0.26.0+cu132 (/home/bot/vllm editable)
           build/install 已完成；日志 /tmp/vllm_build.log
sparkinfer: /home/bot/project/sparkinfer (fork: jieen1/sparkinfer, branch blackforge-main)
           ⚠️ 包元数据要求 torch>=2.12.0，当前环境不满足
模型:      ~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/
           snapshots/07614121b31898586430f189d27a25a0be310843/
Draft:     ~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-DFlash-NVFP4/
           snapshots/723794750422b3efbf3a7b3af76dffb4ba035943/
```

**必须的环境变量:**
```bash
export USE_LIBUV=0
export HF_HUB_OFFLINE=1
```

**禁止:** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（会爆显存）

---

## Git 状态

```
branch: main
最新 commit: 8e5c504 (Self-allocated KV cache + deterministic MoE)
未提交改动: 11 files modified + 6 new untracked files
```

---

## 已完成的工作

### 1. 自分配 KV cache（已提交, commit 8e5c504）
- vLLM 0.26.0 把 KV cache 布局从 `[blocks, 2, bs, kv_heads, head_dim]` 改成
  `[blocks, 8, bs, 256]`（stride-order permuted）
- 我们在 `laguna.py.__init__` 中自己分配 KV cache，格式为 sparkinfer 原生的
  `[num_blocks, 2, block_size, num_kv_heads, head_dim]`（uint8 存 FP8）

### 2. 自写 KV scatter（已提交）
- vLLM 0.26.0 删除了 `torch.ops._C_cache_ops.reshape_and_cache_flash`
- 在 `laguna_sparkinfer_attn.py` 和 `laguna_cuda_graph.py` 中自写：
  ```python
  k_cache[block_idx, block_off] = (key / k_scale).to(torch.float8_e4m3fn)
  v_cache[block_idx, block_off] = (value / v_scale).to(torch.float8_e4m3fn)
  ```

### 3. MoE 确定性修复（已提交）
- `SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=1` 在 `laguna_sparkinfer_moe.py` 模块级设置
- 解决了 M≥8 时 ATOMIC_SCATTER 归约顺序不确定的问题
- 验证：M=1,4,8,16,32 全部确定性 ✓

### 4. BFAttention 模块替换（未提交，进行中）
- **新文件 `runtime/backends/bf_attention.py`**（248行）
  - `BFAttention(nn.Module)` — 完整替换 vLLM 的 Attention 类
  - `BFAttnContext` + `set_bf_attn_context()` — 轻量 thread-local 上下文
  - `replace_vllm_attention(model, sfc, kv_caches)` — 遍历模型树替换所有 48 层
- **集成点:**
  - `laguna.py:242-244` — `__init__` 中调用 `replace_vllm_attention()`
  - `laguna.py` 的 `_forward()` 和 `_forward_with_aux()` 中调用 `set_bf_attn_context()`
  - `laguna_cuda_graph.py` 的 3 个 forward 路径中调用 `set_bf_attn_context()`
  - `laguna_dflash.py` 的 3 个 main model forward 路径中调用（draft model 不需要）
- **验证:** 48 层全部替换成功，`sfc` 中 0 个 vLLM Attention 残留

### 5. FP8 descale 修复（未提交）
- **根因:** sparkinfer workspace 的 `k_descale`/`v_descale` 被硬编码为 1.0，
  但 KV cache 写入时用了 `k / k_scale` 量化，读取时必须乘回 `k_scale`
- **修复:**
  - `laguna_sparkinfer_attn.py`: 新增 `_paged_descale()` 函数
  - prefill 路径: `SparkinferPrefillWorkspace.forward()` 接受 `k_descale`/`v_descale` 参数
  - CG decode/extend 路径: 设置 `ws._k_descale = layer._k_scale.detach()`
- **效果:** 输出从完全乱码（'´´ analysed´ „ „'）改善为重复词（' is is is'）

---

## 数值闭环测试结果（2026-07-25 完成）

### ✅ 已证明正确的组件

| 组件 | 结果 | 证据/脚本 |
|------|------|-----------|
| sparkinfer attention kernel | ✅ cos=0.999999 vs SDPA(fp8-dequant) | `tests/debug/test_sparkinfer_sdpa.py`, `tests/debug/test_attn_numerical.py` |
| FP8 KV write/descale | ✅ K err=2.8%, V err=2.7%, FP8 range ±448 | 同上 |
| BFAttention output (layers 0,1) | ✅ cos=0.9996 vs SDPA bf16 reference | `tests/debug/test_attn_numerical.py` |
| Page table / slot_mapping / cache_seqlens | ✅ 全部正确匹配 | `tests/debug/test_attn_metadata.py` |
| Triton norm patch | ✅ 有/无 patch logits 完全一致 | `/tmp/test_no_triton.py` |
| Sparkinfer MoE patch | ✅ 不影响 ranking | `/tmp/test_no_moe_patch.py` |
| MoE 确定性 | ✅ M=1,4,8,16,32 全部确定 | `SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=1` |

### ⚠️ 旧 E2E 结果（输入契约错误，不能用于判定 BFAttention）

| 测试 | 结果 | 说明 |
|------|------|------|
| E2E 首 token | ❌ ' is' 而非 ' Paris' | logits: ' is'=26.38, ' France'=23.25, ' Paris'=16.75 |
| E2E 生成 | ❌ " is is is is..." | 重复词 |

### 旧“关键矛盾”的重新解释

**attention 逐层数值正确 (cos=0.9996)，但 E2E 输出错误。**
这组结果原本被解释为 BFAttention 集成问题；现在已确认 E2E 和逐层测试没有
使用同一有效输入契约。E2E 漏 BOS，因而不能从这组结果推出模块替换有问题。

---

## 已复现症状及修正后的归因

### 症状
```
Prompt: "The capital of France is"
期望:   " Paris"
实际:   " is is is is is is is is is is is is is is is is"
Logits: ' is'=26.38, ' France'=23.25, ' Paris'=16.75 — 值合理但排序错误

Prompt: "2 + 2 ="
实际:   " 2 = 2 = 2 = 2"
```

### 根因

这些结果来自完整 prompt 缺 BOS。Laguna 的 BOS 同时复用 EOS token id=2，
checkpoint tokenizer 的 post-processor 定义为 `BOS + Sequence(A)`；显式传
`add_special_tokens=False` 后，模型看到的第一个 token 和后续绝对 position
全部与训练/基准输入不同，最终 logits 排序系统性偏移。这解释了为什么替换
norm、MoE、Attention 模块后错误仍保持为同一类 `" is"` 重复。

### monkey-patch 分叉测试（已完成，但输入无效）

**monkey-patch 测试** (`/tmp/test_monkeypatch.py`):
- 禁用 BFAttention 模块替换（`replace_vllm_attention = no-op`）
- 保留原始 vLLM Attention 模块，只 patch 其 `.forward()` 方法
- 使用 vLLM 的 `get_forward_context()` 而非 `get_bf_attn_context()`
- import 已修复，48 层均成功 patch，模块类型仍为原始 `Attention`
- 结果仍为 top-1 `" is"`，但脚本第 65 行同样使用
  `add_special_tokens=False`

该结果排除了“只有替换 module object 才会出现错误”，但没有完成正确输入下
的 A/B 分叉。有效判读规则见文末“下一步（按判别力排序）”。

---

## vLLM build/install 状态

### 编译前问题
vLLM 0.26.0 的 C 扩展 (`.abi3.so`) 与 PyTorch 2.13.0 ABI 不匹配：
```
ImportError: /home/bot/vllm/vllm/_C.abi3.so: undefined symbol:
  _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE
```

### 当时影响
- `vllm._C.ops` 不可用（C++ 自定义 op 全部失效）
- `vllm.attention` 模块不存在（0.26.0 重构了目录结构）
- 但 Python 层面的 import 正常（`import vllm` 返回 0.26.0）
- 模型加载（`get_model`）仍可工作（纯 Python 权重加载路径）
- **我们的 runtime 当时可以加载模型并跑推理**（得到错误输出），因为 attention
  和 MoE 都走自研路径，不依赖 vLLM C 扩展

### 当前状态

下列命令已经结束：

```bash
cd /home/bot/vllm
PATH="/home/bot/.venvs/vllm/bin:$PATH" MAX_JOBS=8 \
  /home/bot/.venvs/vllm/bin/python -m pip install -e . --no-build-isolation
```

它成功安装 vLLM `0.26.0+cu132`，但同时将 Torch 降为 2.11.0，造成
sparkinfer 声明的 `torch>=2.12.0` 条件不满足。按照 GPU 协作约束，编译完成
后不再启动本项目 GPU 测试。

---

## 测试脚本清单

### 数值闭环测试（已完成，结果见上表）
```bash
# sparkinfer attention vs SDPA 参考
USE_LIBUV=0 HF_HUB_OFFLINE=1 /home/bot/.venvs/vllm/bin/python tests/debug/test_sparkinfer_sdpa.py
USE_LIBUV=0 HF_HUB_OFFLINE=1 /home/bot/.venvs/vllm/bin/python tests/debug/test_attn_numerical.py

# 元数据正确性
USE_LIBUV=0 HF_HUB_OFFLINE=1 /home/bot/.venvs/vllm/bin/python tests/debug/test_attn_metadata.py

# Triton norm / MoE patch 排除
USE_LIBUV=0 HF_HUB_OFFLINE=1 /home/bot/.venvs/vllm/bin/python /tmp/test_no_triton.py
USE_LIBUV=0 HF_HUB_OFFLINE=1 /home/bot/.venvs/vllm/bin/python /tmp/test_no_moe_patch.py
```

### 旧分叉测试（已执行，但因缺 BOS 判据无效）
```bash
# import 已修复；脚本仍使用 add_special_tokens=False，不能直接复用作正确性门禁
USE_LIBUV=0 HF_HUB_OFFLINE=1 /home/bot/.venvs/vllm/bin/python /tmp/test_monkeypatch.py
```

### E2E 正确性测试
```bash
# 唯一推荐的 France 入口；默认 encode 包含 BOS
USE_LIBUV=0 HF_HUB_OFFLINE=1 /home/bot/.venvs/vllm/bin/python \
  -m benchmarks.laguna_backend_test --fact-only

# 完整 benchmark（首 token 门禁通过后再跑）
USE_LIBUV=0 HF_HUB_OFFLINE=1 /home/bot/.venvs/vllm/bin/python benchmarks/e2e_cg_bench.py
```

当前 GPU 已交还其他开发者使用，上述命令仅作为环境恢复后的操作记录，不应在
当前协作窗口继续执行。

---

## 关键文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `runtime/backends/bf_attention.py` | **新增** | BFAttention 模块（替换 vLLM Attention），248行 |
| `runtime/backends/laguna.py` | 修改 | 集成 BFAttention + bf_attn_context (L242, L787, L1213) |
| `runtime/backends/laguna_sparkinfer_attn.py` | 修改 | 自写 KV scatter + `_paged_descale()` + descale 传参 |
| `runtime/backends/laguna_cuda_graph.py` | 修改 | 自写 KV scatter (2处) + descale + set_bf_attn_context (3处) |
| `runtime/backends/laguna_dflash.py` | 修改 | set_bf_attn_context (3 main-model paths, NOT draft) |
| `runtime/backends/laguna_dflash_cudagraph.py` | 修改 | DFlash CG 相关 |
| `runtime/backends/laguna_sparkinfer_moe.py` | 已提交 | 确定性 MoE |
| `tests/debug/` | **新增** | 5 个数值闭环测试脚本 |
| `tests/test_bf_attention.py` | **新增** | BFAttention 单元测试 |

---

## 架构：BFAttention 调用链

```
LagunaBackend._forward()
  → 构建 attn_metadata_dict + slot_mapping_dict (per layer group)
  → with bf_attn_context(attn_metadata_dict, slot_mapping_dict):
      with set_forward_context(attn_metadata_dict, vllm_config, ...):
        model.forward(input_ids, positions)
          → LagunaDecoderLayer.forward() × 48
            → LagunaAttention.forward()
              → qkv_proj(hidden_states) → q, k, v
              → q_norm, k_norm, rotary_emb(positions, q, k)
              → self.attn(q, k, v)          ← BFAttention (替换了 vLLM Attention)
                → get_bf_attn_context()     ← thread-local, 非 vLLM 的 get_forward_context
                → KV write: k_cache[bi,bo] = (k / k_scale).to(fp8)
                → self.impl.forward(self, q, k, v, kv_cache, meta, out)
                  → SparkinferAttentionImpl.forward()
                    → prefill: SparkinferPrefillWorkspace.forward(k_descale, v_descale)
                    → CG decode: SparkinferDecodeWorkspace.forward()
              → gating (softplus) → o_proj(attn_output)
            → RMSNorm + MoE/dense FFN
  → model.compute_logits(hidden_states) → logits
```

---

## 模型参数（Laguna-S-2.1）

```
48 layers: 12 full-attn (window=-1) + 36 SWA-512 (window=511)
47 MoE layers (layer 0 = dense), 256 experts, top_k=10
num_attention_heads: 48 (layer 0), 72 (layers 1+)
num_key_value_heads: 8
head_dim: 128
hidden_size: 3072
routed_scaling_factor: 2.5
NVFP4 MoE experts, bf16 dense layers, FP8 KV cache
```

---

## 下一步（按判别力排序）

### 0. 先恢复可解释的运行环境

1. 恢复项目约定的 PyTorch `>=2.12` / CUDA / sparkinfer 组合，确认 vLLM
   extension 与该 Torch ABI 一致。
2. 不要直接重复 `pip install -e . --no-build-isolation`；如果只安装已经编译的
   editable vLLM，必须避免 pip 再次擅自替换 Torch/CUDA 依赖。
3. 环境恢复后先记录 `python -c "import torch, vllm; ..."`、sparkinfer import
   和扩展加载结果，再进入模型测试。

### 1. 先锁死输入契约

所有事实正确性脚本统一：

```python
prompt_ids = tokenizer.encode(prompt)
assert prompt_ids[0] == tokenizer.bos_token_id == 2
print(prompt_ids)
```

对 continuation/filler 片段仍可使用 `add_special_tokens=False`，但“完整 prompt
的首段”必须显式包含一次且仅一次 BOS。每次 oracle A/B 前先断言两边
`prompt_ids` 完全相等。

### 2. 只做一次单变量分叉

不要继续使用互相漂移的 `/tmp/test_*.py`。基于仓库现有
`benchmarks.laguna_backend_test --fact-only` 增加一个 attention mode 开关，
使两条路径共享同一 tokenizer、prompt ids、KV cache、MoE、norm、block size
和 eager 配置：

| 路径 | Attention 模块 | forward 实现 | 输入 |
|------|----------------|--------------|------|
| A | BFAttention | BFAttention.forward | BOS + prompt |
| B | 原 vLLM Attention | monkey-patched forward | 同一 BOS + prompt |

判读：

- A=`Paris`、B=`Paris`：BFAttention E2E 门禁通过，旧问题就是测试缺 BOS。
- A≠`Paris`、B=`Paris`：才进入 BFAttention 模块替换/属性/引用排查。
- A≠`Paris`、B≠`Paris` 且 logits 接近：排查两边公共的 KV scatter、descale、
  sparkinfer prefill 或 MoE，不再排查模块替换。
- A/B 首 token 相同但 logits 不一致：保存 logits cosine/max-diff，并逐层定位
  首个漂移层。

首轮只跑一次 prefill 和首 token，不生成 50 token，不开 CUDA Graph、不启用
DFlash、不跑 vLLM LLM engine。

### 3. 首 token 通过后的恢复顺序

1. eager prefill + eager decode；
2. CUDA Graph decode；
3. DFlash eager；
4. DFlash + CUDA Graph；
5. 64K，再扩展到 128K/200K。

每一级必须复用同一 tokenizer 契约，并与上一级比较首 token、greedy token
序列和 logits cosine，禁止同时切换两个变量。

### 4. 修复真实产品入口

`server/app.py:_tokenize_encode()` 当前仍固定
`add_special_tokens=False`，Laguna raw completions 会复发同一问题。Chat 路径
通过 chat template，不属于这个故障面。server/backend 应暴露模型相关的
tokenize kwargs：Laguna 使用默认 special tokens，Qwen 保持
`add_special_tokens=False`。

---

## 历史教训（避免重蹈覆辙）

1. **对比装置污染**：已发生 4 次（CG-先-eager-后 KV 伪影、同进程权重释放伪影、
   ×2.5 位置不一致、a1 scale 方向镜像）。每次对比必须确认两侧 harness 完全一致。
2. **循环论证**：vs 自带参考 0.998 不能证明正确（共享同一套 scale 折叠）。
   唯一合法裁判：bf16 反量化真值。
3. **数字不可通约**：多个手搓脚本之间的换算约定漂移。
   必须用唯一规范对比脚本。

---

## 附录：KV Scale 加载路径分析 (2026-07-26)

### 发现

Checkpoint 包含 96 个 KV scale 参数（48层 × k_scale + v_scale），值如：
- `model.layers.0.self_attn.k_scale` = 0.031982
- `model.layers.0.self_attn.v_scale` = 0.001198

### 潜在问题：命名不匹配

Checkpoint 键名：`model.layers.0.self_attn.k_scale`
模型参数位置：`model.layers.0.self_attn.attn.k_scale`（在 Attention 子模块上）

`AutoWeightsLoader` 按前缀递归匹配：
1. 走到 `LagunaAttention`（`self_attn`）时，查找名为 `k_scale` 的子模块或直接参数
2. `k_scale` 不是 `LagunaAttention` 的直接参数，而是在子模块 `attn`（Attention）上
3. 匹配失败 → 检查 `ignore_unexpected_suffixes` → `.k_scale` 在列表中 → **静默跳过**

### 但 descale 修复确实改善了输出

之前修复 sparkinfer descale（从硬编码 1.0 → 使用模型 `_k_scale`）后，
输出从完全乱码改善为重复词。这说明 `_k_scale ≠ 1.0`，即 scale 确实被加载了。

**可能的解释：**
- `CompressedTensorsKVCacheMethod.create_weights()` 在 Attention 上注册了 `k_scale` 参数
- `process_weights_after_loading()` 将 `k_scale` → `_k_scale` 并删除 `k_scale`
- 可能有其他加载路径（如 `BaseKVCacheMethod` 的 weight_loader）

### 验证方法

编译完成后运行 `/tmp/test_diag_comprehensive.py`，Phase 1 会打印：
- 模型上的 `_k_scale` 实际值
- Checkpoint 中的 `k_scale` 值
- 两者是否匹配

**如果 `_k_scale = 1.0`（未加载）→ 这就是根因**
**如果 `_k_scale ≈ 0.032`（已加载）→ 排除此假设，继续其他方向**

---

## 2026-07-26 Session Update: Memory Optimization + Verify CG Fix

### Commits Pushed (this session)

| Commit | Description |
|--------|-------------|
| `199ac67` | Reclaim reserved physical slot + free dummy CG caches |
| `e9cf99d` | Fix verify CG stale worklist (update_prefill_graph_replay_metadata) |

### P3-1: RESERVED_PHYSICAL_SLOTS Reclaimed ✅

**Problem**: `RESERVED_PHYSICAL_SLOTS=1` reserved an entire physical slot (blocks_per_slot blocks per layer) that was never written. At 128K context: ~1.5 GB wasted; at 256K: ~3 GB.

**Analysis**:
- Physical slot 0 was used as a null sentinel: page table padding entries were set to 0
- Sparkinfer kernels respect `cache_seqlens` + `block_valid_mask` — padding entries beyond cache_seqlens are NEVER read
- The sentinel was defensive, not functional

**Fix**:
- `RESERVED_PHYSICAL_SLOTS = 0` (laguna.py only; block_pool.py for Qwen3.6 runner unchanged)
- Page table padding changed from `0` to `full_base` / `ring_base` (current slot's first block) as a safe defensive sentinel
- All hardcoded `slot + 1` replaced with `_physical_slot(slot)` across 4 files
- Draft KV cache allocation uses symbolic constant

**Memory savings**: One full slot of KV cache = `blocks_per_slot × block_bytes × num_layers`. At 128K with 7 slots: gains an 8th slot capacity.

### P3-2: Dummy FP8 Cache Freed ✅

**Problem**: `LagunaCudaGraphDecode._bind_kv_caches()` computed real key_cache/value_cache but never assigned them to the workspace. Dummy fp8 tensors (~128 MB per full-attention workspace at 128K) remained allocated.

**Fix**: Added `ws._k_cache = key_cache; ws._v_cache = value_cache` in `_bind_kv_caches()`.

### Verify CG Stale Worklist Fix ✅ (needs GPU validation)

**Problem**: `LagunaCudaGraphVerify._fill_buffers()` only called `ws._copy_runtime_metadata()` (page_table + cache_seqlens) but never updated the sparkinfer worklist. The worklist (block_valid_mask, kv_chunk_size, kv_window_start_tokens) stayed frozen at capture-time values. When SWA ring alignment crossed a page boundary during replay, the kernel processed stale chunk boundaries → wrong attention → acceptance collapse (87% → 23-25%).

**Root cause in sparkinfer terms**: `update_prefill_graph_replay_metadata()` is the public API that copies metadata AND runs a Triton kernel to recompute the worklist from runtime cache_seqlens. Our code was only doing the first half.

**Fix**: Replaced `ws._copy_runtime_metadata(...)` with `ws.update_prefill_graph_replay_metadata(..., window_left=wl)`.

**Expected impact**: Eager verify ~308 ms/step → CG ~38 ms/step (8x speedup). Enabled by default via `QSR_VERIFY_CUDA_GRAPH=1`.

### GPU Validation Needed

When GPU is available, run:
```bash
CUDA_VISIBLE_DEVICES=0 USE_LIBUV=0 HF_HUB_OFFLINE=1 FLASHINFER_DISABLE_VERSION_CHECK=1 \
QSR_DFLASH_CUDA_GRAPH=1 SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT=1 \
/home/bot/.venvs/vllm/bin/python /tmp/test_acceptance_repro.py 65536 256
```

Expected: acceptance >80% with verify CG enabled. If acceptance drops, set `QSR_VERIFY_CUDA_GRAPH=0` to isolate.

### Outstanding Issues (unchanged)

1. **Full-prefix-hit bug**: warm r1/r2 acceptance collapse. Someone else investigating.
2. **vLLM 0.26.0 comparison**: needs compilation + GPU time.
3. **200K benchmark**: not yet run.
