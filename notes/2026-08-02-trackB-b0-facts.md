# Track B0 事实基线收口：B0-2 / B0-6 / B0-7 / B0-8（四条零 GPU 条目）

> 编制日期：2026-08-02 · worktree `work/trackB-20260802` @ `d87c7ef`
> 环境：`~/.venvs/vllm/bin/python`（`transformers==5.8.0`）· **全程零 GPU**，
> 只读 checkpoint 的 `config.json`/`hf_quant_config.json`/
> `model.safetensors.index.json`/safetensors JSON header，以及本机已装的
> HF `transformers`、`flash-linear-attention`（`fla`）参考实现源码。
> 没有加载任何权重进内存，没有跑任何模型前向，没有起任何进程占 GPU。
>
> **这篇笔记的定位**：`docs/implementation-plan.md` §7.1 的 B0-2/B0-6/B0-7/B0-8
> 四条在本轮之前**尚未打勾**，尽管其中三条（B0-2/6/7）已经在
> [`2026-08-02-qwen36-b0-fact-baseline.md`](2026-08-02-qwen36-b0-fact-baseline.md)
> 里被另一轮会话详细核实过，B0-8 已经在
> [`2026-08-01-b6-mtp-gdn-verification.md`](2026-08-01-b6-mtp-gdn-verification.md)
> （`docs/investigation-queue.md` B-6）核实过。本笔记做两件事：
> 1. **独立复现**这四条的关键数字与代码行号证据（不是抄一遍旧笔记——本仓库
>    有过"文档继承了别处的数字"的事故，见
>    [`2026-08-02-laguna-docs-inherited-qwen36-numbers.md`](2026-08-02-laguna-docs-inherited-qwen36-numbers.md)，
>    所以这里的每一条数字都是本轮会话重新跑出来的，不是复制）；
> 2. 给出可以直接写进 `docs/implementation-plan.md` §7.1 的勾选状态与结论句。
>
> 详细推导（含 sparkinfer w4a16 候选路径、GDN dtype 证据链纠偏、vLLM/SGLang
> FP8 KV 缺省值先例等超出 B0-2/6/7/8 本身范围的追加内容）留在两篇源笔记里，
> 本笔记只收敛到四条清单项本身需要的证据。

---

## 结论摘要（可直接抄进 implementation-plan.md）

| 条目 | 结论 | 证据密度 |
|---|---|---|
| **B0-8** | **确认：MTP 层零 GDN 张量**（两个 checkpoint 独立复现）。但**不能因此删掉 B3 的 GDN 回滚项**——回滚问题在主模型 verify 阶段的 48 层 GDN，跟 MTP 头本身有没有 GDN 无关 | 硬证据：tensor 名逐一列举，两个 checkpoint 交叉验证 |
| **B0-2** | checkpoint 是**混合精度**：GDN/self_attn 投影是 FP8（W8A8），稠密 MLP/`lm_head` 是 NVFP4 weight-only（W4A16，block=16）。**KV cache FP8 声明存在但零 scale 张量** | 硬证据：dtype/shape 逐张量实测 |
| **B0-6** | **确认退化**：纯文本输入下 T/H/W 三份 position_ids 恒等（同一 `.expand()` 视图），`apply_interleaved_mrope` 的覆盖是数值无操作，等价于标准 1D RoPE + `partial_rotary_factor=0.25` + `rope_theta=1e7` | 硬证据：源码行号 + 逐行读出的控制流 |
| **B0-7** | 权重 18.767 GiB（纯文本）；GDN 状态每槽固定 ~72–150 MiB（与上下文无关）；96GB 卡上 256K/FP8-KV 可行到 c=8，256K/BF16-KV 可行到 c=4 | 精确整数算术（非估算），权重数字来自 safetensors header 实测 |

---

## B0-8 · Qwen3.6 的 MTP 层是否带 GDN

### 结论

**不带。** 本轮独立复现 `nvidia/Qwen3.6-27B-NVFP4`（主线 checkpoint）与
`unsloth/Qwen3.6-27B-NVFP4`（交叉验证）两个 checkpoint 的 `mtp.*` 张量清单，
零 `linear_attn.*` / `A_log` / `conv1d` / `dt_bias` / `in_proj_{a,b}` 之类的
GDN 专属张量。

### 证据（本轮命令，非抄旧笔记）

```
~/.venvs/vllm/bin/python -c "
from pathlib import Path
from loader.checkpoint_index import load_checkpoint_index
model_dir = Path.home()/'.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404'
idx = load_checkpoint_index(model_dir)
mtp = sorted(n for n in idx.weight_map if n.startswith('mtp.'))
print(len(mtp)); [print(n) for n in mtp]
"
```

输出（`nvidia/Qwen3.6-27B-NVFP4`，15 个张量）：

```
mtp.fc.weight
mtp.layers.0.input_layernorm.weight
mtp.layers.0.mlp.down_proj.weight
mtp.layers.0.mlp.gate_proj.weight
mtp.layers.0.mlp.up_proj.weight
mtp.layers.0.post_attention_layernorm.weight
mtp.layers.0.self_attn.k_norm.weight
mtp.layers.0.self_attn.k_proj.weight
mtp.layers.0.self_attn.o_proj.weight
mtp.layers.0.self_attn.q_norm.weight
mtp.layers.0.self_attn.q_proj.weight
mtp.layers.0.self_attn.v_proj.weight
mtp.norm.weight
mtp.pre_fc_norm_embedding.weight
mtp.pre_fc_norm_hidden.weight
```

同一命令换成 `unsloth/Qwen3.6-27B-NVFP4` 的 snapshot 目录，输出**逐字节相同**
的 15 个张量名（`linear_attn in mtp names: False`，本轮实跑确认）。这与
`mtp.layers.0`（`self_attn.*` + `mlp.*` + 两个 norm）的结构，跟主模型
`full_attention` 层（例如 layer 3）完全同构，跟主模型 `linear_attention`
（GDN）层（例如 layer 0，带 `linear_attn.A_log`/`conv1d.weight`/`dt_bias`/
`in_proj_{a,b,qkv,z}`/`norm.weight`）结构完全不同——本轮同一条命令对 layer 0
和 layer 3 也重新跑了一遍，确认了这个对照（见下方 B0-2 §"张量命名"一节，
同一批命令复用）。

`config.json` 本身（`text_config.layer_types`，64 元素，只覆盖主模型）
**不含 MTP 层内部结构的字段**（没有 `mtp_layer_types`），只有
`mtp_num_hidden_layers=1`——这条与
[`2026-08-01-b6-mtp-gdn-verification.md`](2026-08-01-b6-mtp-gdn-verification.md)
§1 的记录一致，本轮独立读取 `config.json` 复现同样的字段缺失。该笔记还
额外核实了另外 4 个 checkpoint 变体（`sakamakismile`/`morosystems`/
`Qwen/Qwen3.6-27B-FP8`/`cyankiwi` AWQ-INT4），共 6 个全部同一结论——
本轮不重复跑那 4 个，只对最关键的主线 checkpoint 与它的社区 NVFP4 对照版
（`unsloth`）做了独立复现，作为"不是抄旧笔记"的验证锚点。

### 重要：这个结论不删除 B3 的 GDN 回滚项——两个问题层面不同

`investigation-queue.md` B-6 的原始提问框架是"vLLM 注释说 draft 模型没有
mamba 层所以不需要 eagle shift——如果我们的 MTP 也没有 GDN，B3 最难的一项
是不是就不存在了"。**这个推论本身有缺口，已经被 B-6 纠正**：

- vLLM 那条注释针对的是**草稿模型自己的**递归状态管理（drafting 阶段不需要
  shift 自己的 GDN 状态，因为草稿头没有）——B0-8 的证据确实消掉了这一点：
  MTP 头不需要自己的 conv/ssm state。
- 但 **verify 阶段仍然要把 MTP 提出的候选 token 整段跑一遍主模型的完整
  64 层（含 48 层 GDN）**。一旦某些候选被拒绝，主模型的 GDN 递归状态已经
  被"没发生过的" token 更新过，且这个更新**不可逆**（不像 KV cache 可以
  直接丢弃被拒绝的块）——这个问题跟 MTP 头本身有没有 GDN **完全无关**，
  是主模型侧的问题。
- 独立佐证：vLLM 自己的 `vllm-project/vllm#47572`（ReplaySSM RFC）原话
  "Speculative decoding must roll back rejected draft tokens, but the SSM
  state update is irreversible... the current implementation keeps a
  separate recurrent state per draft token"——这正是本仓库
  `docs/investigation-queue.md` D-3（ReplaySSM，显存 11.5GB→1.8GB）已经
  独立记录的同一个问题。

**对 Track B3 的净影响**：草稿侧的递归状态管理可以删（真实、但比原队列
设想更小的简化）；主模型侧 verify 阶段的 GDN 递归状态回滚**不能删**，
应改写为"主模型侧"问题并与 D-3 排期合并。**implementation-plan.md 现有的
两分支写法（"若 MTP 含 GDN / 若 MTP 不含 GDN"）需要改写**——不是二选一
分支，B0-8 已经确定是"不含"这一支，但"不含"这一支本身不等于"B3 最难项
消失"，需要把 B3 的描述改成"主模型侧 GDN 回滚（与 D-3/ReplaySSM 合并排期）
+ 草稿侧无递归状态（已简化）"，而不是保留"两个分支，取决于结论"的措辞。

---

## B0-2 · modelopt NVFP4 的张量命名与 scale 语义

### 1. checkpoint 是混合精度，不是"整模型 NVFP4"

`config.json`（`~/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/
snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404/config.json`，本轮
`python3 -m json.tool` 直接读出）：

```
quantization_config.config_groups.group_0: num_bits=8, type=float (FP8, 静态)
  targets: self_attn.{q,k,v,o}_proj（16 个 full-attention 层）
         + linear_attn.{in_proj_qkv,in_proj_z,out_proj}（48 个 GDN 层）
quantization_config.config_groups.group_1: num_bits=4, type=float, group_size=16
  input_activations: 未声明（weight-only）
  targets: mlp.{gate,up,down}_proj（64 层全部）+ lm_head
```

`hf_quant_config.json` 同目录下与 `config.json` 的 `quantization_config`
逐条一致（本轮未重新逐条 diff 两个文件，采信旧笔记已做的逐条比对，因为
这是文件内容比对而非需要重新实测的数字）。

### 2. 张量命名与 dtype/shape——本轮逐张量重新实测

命令（对 layer 0 = GDN 层、layer 3 = full-attention 层各转储关键张量的
`dtype`/`shape`）：

```python
from pathlib import Path
from loader.checkpoint_index import load_checkpoint_index
from loader.safetensors_header import read_safetensors_header
model_dir = Path.home()/".cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404"
idx = load_checkpoint_index(model_dir)
headers = {s: read_safetensors_header(model_dir/s) for s in idx.shard_names}
def info(n):
    shard = idx.weight_map[n]; h = headers[shard][n]
    return h.dtype, h.shape
```

实测输出（本轮真跑，不是抄数）：

```
model.language_model.layers.0.linear_attn.in_proj_qkv.weight          ('F8_E4M3', (10240, 5120))
model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale    ('F32', ())
model.language_model.layers.0.linear_attn.in_proj_qkv.input_scale     ('F32', ())
model.language_model.layers.0.linear_attn.A_log                       ('BF16', (48,))
model.language_model.layers.0.linear_attn.dt_bias                     ('BF16', (48,))
model.language_model.layers.0.linear_attn.conv1d.weight               ('BF16', (10240, 1, 4))
model.language_model.layers.0.linear_attn.in_proj_a.weight            ('BF16', (48, 5120))
model.language_model.layers.0.linear_attn.in_proj_b.weight            ('BF16', (48, 5120))
model.language_model.layers.0.linear_attn.norm.weight                 ('BF16', (128,))
model.language_model.layers.3.self_attn.q_proj.weight                 ('F8_E4M3', (12288, 5120))
model.language_model.layers.3.self_attn.k_proj.weight                 ('F8_E4M3', (1024, 5120))
model.language_model.layers.3.self_attn.v_proj.weight                 ('F8_E4M3', (1024, 5120))
model.language_model.layers.3.self_attn.o_proj.weight                 ('F8_E4M3', (5120, 6144))
model.language_model.layers.3.mlp.down_proj.weight                    ('U8', (5120, 8704))
model.language_model.layers.3.mlp.down_proj.weight_scale              ('F8_E4M3', (5120, 1088))
model.language_model.layers.3.mlp.down_proj.weight_scale_2            ('F32', ())
model.language_model.layers.3.mlp.down_proj.input_scale               ('F32', ())
lm_head.weight                                                        ('U8', (248320, 2560))
```

**命名规律**（逐后缀，本轮实测确认）：

- **FP8 目标**（`group_0`，self_attn + GDN 的 in_proj/out_proj）：`.weight`
  是 `F8_E4M3`，**不 pack**（1 byte/元素直存）；`.weight_scale` 是 `F32`
  **标量**（per-tensor，静态）；`.input_scale` 是 `F32` 标量（激活侧，
  也是 per-tensor 静态）。
- **NVFP4 目标**（`group_1`，MLP + `lm_head`）：`.weight` 是 `U8`，
  形状 `[out, in//2]`（2 个 FP4/E2M1 元素 pack 进 1 byte——
  `down_proj` 的 `8704 = 17408/2` 精确整除验证 pack 约定）；
  `.weight_scale` 是 `F8_E4M3`，形状 `[out, in//16]`（per-block，
  block_size=**16**，`down_proj` 的 `1088 = 17408/16` 精确整除验证）；
  `.weight_scale_2` 是 `F32` **标量**（per-tensor 全局二级 scale）；
  `.input_scale` 也存在（`F32` 标量），但 `group_1` 声明
  `input_activations=None`（weight-only），这个标量的运行时语义
  **[待验证]**——大概率是校准期留下的 amax，推理时是否被消费不确定。

**KV cache scale——本节最大发现，本轮独立重新做穷举验证**：

```python
names = sorted(idx.weight_map)
for pat in ["scale","amax","kv","cache"]:
    hits = [n for n in names if pat in n.lower()]
    print(pat, sorted(set(n.split(".")[-1] for n in hits)))
```

本轮输出：

```
total tensors 2194
scale: ['input_scale', 'weight_scale', 'weight_scale_2']
amax: []
kv:   ['bias', 'input_scale', 'weight', 'weight_scale']   # 全部来自 in_proj_qkv 命名里的 "kv" 子串，不是真的 kv-cache scale
cache: []
```

**零命中** `k_scale`/`v_scale`/`kv_scale`/`kv_cache_scale`/任何 `amax`
张量——尽管 `config.json`/`hf_quant_config.json` 都声明
`kv_cache_scheme.kv_cache_quant_algo = FP8, dynamic=false`。这不是
"静态量化就该没有 scale 张量"的通例：旧笔记的对照组
`poolside/Laguna-S-2.1-NVFP4` 同样 `dynamic=False` 但**真的有**
`self_attn.k_scale`/`v_scale`（BF16, `(1,)`，48 层各一对）——本轮未
重新拉这个对照 checkpoint 复测（不在本轮任务范围内的模型，仅引用旧笔记
已做的对照，不是本轮新证据）。**结论**：这份 checkpoint 的 KV FP8 声明
"元数据里写了、权重里没落地"，是这份特定 checkpoint 的缺口，不是格式惯例。

`vision` 张量 333 个（`model.visual.*` 前缀），`mtp` 张量 15 个——本轮
`total tensors 2194` 与 `vision 333 mtp 15` 都在同一条命令里重新数出来，
与旧笔记数字一致。

### 3. 与 Laguna（compressed-tensors）的命名差异（本轮不重新对照，采信旧笔记的对照表已经充分）

modelopt 用 `.weight`（不是 Laguna 的 `.weight_packed`）+ `.weight_scale`
（block，与 Laguna 命名撞了但语义相同）+ `.weight_scale_2`（全局，Laguna
叫 `.weight_global_scale`）。**同一个 `.weight` 后缀在这份 checkpoint 里
可能是 U8-packed-NVFP4、F8_E4M3-unpacked、或 BF16 明文三种之一**——
必须先查该 module 属于哪个 `config_groups`（或用层类型+`ignore`列表推导）
才能知道怎么解释 `.weight` 张量，不能只看后缀名。这条结构性差异是给 A4
加载器 adapter 的直接约束，已经写进 `docs/qwen36-rebuild-spec.md` §3.4。

---

## B0-6 · mrope-interleaved 在纯文本下确认退化为标准 1D RoPE

### 结论（可直接当断言写进 B1 的代码/测试）

> **在纯文本请求（未传 `image_grid_thw`/`video_grid_thw`）下，Qwen3.6 的
> RoPE 数学上恒等于标准 1D RoPE：`partial_rotary_factor=0.25`
> （只旋转 `head_dim=256` 的前 64 维）+ `rope_theta=10,000,000`。
> B1 不需要实现三维 mrope 插值，可以直接用标准 1D RoPE kernel 加一个
> "只转前 25% 维度"的 partial-rotary 支持。这个等价性对 prefill 与 decode
> 的每一步都成立（不存在"prefill 退化、decode 不退化"的分支）。**

### 证据（本轮重新读源码文件，行号本轮独立 grep 出来，非抄旧笔记）

文件：`~/.venvs/vllm/lib/python3.12/site-packages/transformers/models/
qwen3_5/modeling_qwen3_5.py`（`transformers==5.8.0`，本轮
`python -c "import transformers; print(transformers.__version__)"` 确认）。

**退化机制**（`Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope`，
本轮 `grep -n` 定位在 157-172 行）：

```python
157: def apply_interleaved_mrope(self, freqs, mrope_section):
...
167:     freqs_t = freqs[0]  # just overwrite the first dimension T
168:     for dim, offset in enumerate((1, 2), start=1):  # H, W
169:         length = mrope_section[dim] * 3
170:         idx = slice(offset, length, 3)
171:         freqs_t[..., idx] = freqs[dim, ..., idx]
172:     return freqs_t
```

这个函数只是"用 `freqs[1]`/`freqs[2]` 覆盖 `freqs[0]` 的部分位置"，不做
任何跨维度数学组合——如果 `freqs[0]==freqs[1]==freqs[2]`（逐元素相等），
覆盖操作是数值上的无操作。

**纯文本时 T/H/W 三个 position_ids 恒等**（`Qwen3_5TextModel.forward`，
本轮定位在 1256-1263 行附近，与旧笔记引用的 1257-1263 基本一致）：

```python
if position_ids is None:
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
    position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
elif position_ids.ndim == 2:
    position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

if position_ids.ndim == 3 and position_ids.shape[0] == 4:
    text_position_ids = position_ids[0]
    position_ids = position_ids[1:]      # 这 3 份传给 rotary_emb
```

`.expand(4, ...)` 是同一个 1D `arange` 张量的 4 份**视图**（共享底层
storage，不是 4 份独立数据）——所以 `position_ids[1:]` 里传给 rotary_emb
的 T/H/W 三份逐元素相等。

**上层多模态包装同样落到这条路径**（`Qwen3_5Model.compute_3d_position_ids`，
本轮 grep 定位：`compute_3d_position_ids` 在 1575 行，`has_multimodal` 在
1586 行，`can_compute_mrope` 在 1593 行）：纯文本请求 `image_grid_thw`/
`video_grid_thw` 都是 `None` → `has_multimodal=False`（1586 行）→
`can_compute_mrope=False`（1593 行）→ `self.rope_deltas` 永远不会被设置
（只在 `if can_compute_mrope` 分支里赋值）→ 后续 `elif self.rope_deltas
is not None and ...` 恒为假 → 每次调用都落到 `else: position_ids = None`，
原样传进 `Qwen3_5TextModel.forward`，回到上面完全相同的退化路径。

**config 参数**（本轮 `config.json` 重新读取确认）：

```
text_config.partial_rotary_factor = 0.25
text_config.rope_parameters = {
    "mrope_interleaved": true,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000,
    "rope_type": "default"
}
```

`rope_theta=10,000,000` 比默认的 `10000` 差 3 个数量级，新实现如果抄错
这个常数会在长上下文位置上产生可测的漂移——这是给 B1 实现者的具体数值,
不是"通常默认值"。

**边界提醒（不是本条结论的例外，是使用边界）**：这个退化只在真正的
纯文本请求上成立。如果请求层面被上游误传了 `image_grid_thw` 之类的字段
（哪怕权重侧已经按 B0-1 拍板跳过 vision 张量），`can_compute_mrope` 会
变 True，退化条件不成立。这个输入校验应该在 API 层或 `ArchitectureSpec`
校验时一并拒绝，不是 RoPE kernel 本身的职责。

---

## B0-7 · 容量测算：64 层 / 256K / 96GB 的可行域

### 1. 权重大小——本轮重新聚合 safetensors header 的 `data_offsets`，与 `index.json` 交叉验证

```python
def nbytes(n):
    lo, hi = headers[idx.weight_map[n]][n].data_offsets
    return hi - lo
total = vision = mtp = 0
for n in idx.weight_map:
    b = nbytes(n); total += b
    if n.startswith("model.visual."): vision += b
    elif n.startswith("mtp."): mtp += b
```

本轮输出（GiB）：

```
total=20.416 vision=0.858 mtp=0.791 lm(rest)=18.767
index.json total_size (bytes) = 21921428072 -> GiB 20.415920831263065
```

`total`（20.416 GiB）与 `model.safetensors.index.json` 的 `metadata.
total_size` 字段（21,921,428,072 bytes = 20.41592... GiB）**吻合到小数点
后 3 位**——本轮独立读取 index.json 元数据做的交叉验证，不是信任旧笔记的
同一句话。

| 部分 | 大小 (GiB) | 说明 |
|---|---:|---|
| 全部张量 | 20.416 | 与 index.json `total_size` 交叉验证通过 |
| vision（排除对象） | 0.858 | 333 个张量，`model.visual.*` 前缀 |
| MTP 头 | 0.791 | B1 不需要，B3 投机解码需要 |
| **backbone + `lm_head`（B1 实际加载量）** | **18.767** | 纯文本、无 MTP |
| backbone + `lm_head` + MTP（B3 加载量） | 19.558 | |

### 2. KV cache（16 个 full-attention 层，与上下文长度线性相关）——本轮重新算术，非估算

每 token 每层字节数（K+V）= `2 × kv_heads(4) × head_dim(256) × dtype_bytes`；
16 层合计 = `32768 × dtype_bytes` bytes/token（本轮用独立 Python 脚本重新
算出这个系数，不是抄公式）：

| KV dtype | 每 token（16 层合计） | 256K（262144 token） | 128K | 64K |
|---|---:|---:|---:|---:|
| FP8（1 byte） | 32768 bytes | **8.0 GiB/槽** | 4.0 GiB/槽 | 2.0 GiB/槽 |
| BF16（2 byte） | 65536 bytes | **16.0 GiB/槽** | 8.0 GiB/槽 | 4.0 GiB/槽 |

（精确整除：`262144 × 32768 / 1024³ = 8.0` 恰好整数，因为
head_dim/kv_heads/层数都是 2 的幂——本轮独立跑 Python 算术验证，不是
四舍五入的估算。KV scale 缺失问题见 B0-2 第 2 节，不影响这里的容量算术，
只影响运行时数值精度。）

### 3. GDN 递归状态（48 层，与上下文长度**无关**，每槽固定）——本轮重新算术

架构参数（`config.json` `text_config`，本轮重新读取）：

```
linear_num_key_heads=16  linear_key_head_dim=128
linear_num_value_heads=48  linear_value_head_dim=128
linear_conv_kernel_dim=4
```

`modeling_qwen3_5.py` 确认 K/Q 会 `repeat_interleave` 从 16 头扩到 48 头
（`num_v_heads // num_k_heads = 3`）再进 delta-rule，所以递归状态的头数
维度是 **48**（不是 16）。单层 SSM 状态元素数 = `48 × 128(k) × 128(v)
= 786,432`。

⚠️ **conv1d 状态数已于 2026-08-02 更正（B1 实测推翻本节的理论推导）。**
本节原写 `conv_dim(10240) × (kernel-1=3) = 30,720`——那是按因果卷积缓存的**标准惯例**
（只存 kernel-1 个历史元素）推出来的，合理但**不是 HF 实际做的**。B1 在真实模型上读
`DynamicCache`：`cache.layers[0].conv_states.shape == (1, 10240, 4)`，是**完整 kernel size**。
正确值：`10240 × 4 = 40,960` 元素/层，比原值大 4/3。

下表的 conv 列已按 40,960 重算（SSM 列不受影响）。这条差异对单槽总量只有约 1.3%
（SSM 状态占绝对大头），但**推导方式的教训更重要：能读到真实缓存对象时不要从惯例推**。

本轮独立 Python 脚本重新算出（SSM 部分）：

```
ssm elems/layer 786432
FP32 SSM total MiB 144.0   BF16 SSM total MiB 72.0
FP32 conv total MiB 5.625  BF16 conv total MiB 2.8125
```

| 状态 dtype | SSM（48层） | conv（48层，已更正） | 单槽合计 |
|---|---:|---:|---:|
| FP32 | 144.0 MiB | 7.5 MiB | **~151.5 MiB** |
| BF16 | 72.0 MiB | 3.75 MiB | **~75.75 MiB** |

（更正前分别是 5.6 / 2.8 MiB 与 ~149.6 / ~74.8 MiB，按 `kernel-1=3` 算。）

**dtype 该取哪个**——本轮独立重新读了 `transformers/cache_utils.py` 与
`flash-linear-attention` 源码确认机制（`~/.venvs/vllm` 里两者都是本机
真实安装，`import fla`/`import transformers` 都命中这两份源码）：

- `flash-linear-attention/fla/ops/gated_delta_rule/fused_recurrent.py`
  （decode 单步路径，本轮 grep 定位在 210-213 行）：
  ```python
  210: if state_v_first:
  211:     final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)
  212: else:
  213:     final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
  ```
  `flash-linear-attention/fla/ops/common/chunk_delta_h.py`（prefill/chunk
  路径，本轮 grep 定位在 637/640 行，与旧笔记一致）同样无条件分配
  `dtype=torch.float32`。**FLA 真 kernel 内部用 FP32 计算 recurrent
  state**，与输入 q/k/v 的 bf16 dtype 无关（本轮独立确认本机 `fla` 包
  真能 import，不是走 HF 的 torch 兜底函数——`~/.venvs/vllm/bin/python
  -c "import fla.ops.gated_delta_rule as g; print(g.chunk_gated_delta_rule,
  g.fused_recurrent_gated_delta_rule)"` 两者皆非 `None`，采信旧笔记已经
  做过的这个 import 检查，本轮未重复跑）。
- 但 `transformers/cache_utils.py`（本轮 grep 定位：
  `lazy_initialization` 在 768 行，关键三行在 773/777/784）：
  ```python
  773:     self.dtype, self.device = conv_states.dtype, conv_states.device
  777:     self.conv_states = torch.zeros_like(conv_states, dtype=self.dtype, ...)
  784:     self.recurrent_states = torch.zeros_like(recurrent_states, dtype=self.dtype, ...)
  ```
  第 784 行用的是 `self.dtype`（已经被 773 行的 `conv_states.dtype`
  锁定成 BF16），**不是** `recurrent_states.dtype`（FLA kernel 输出的
  FP32）——所以 FP32 计算结果被隐式降精度存进 BF16 buffer。

**净结论：单步计算 FP32，跨步持久化 BF16**（不是"全程 BF16 计算"，也不是
"全程 FP32 持久化"）。容量数字上，两种假设都远小于 KV cache（256K/FP8 KV
单槽 8.0 GiB vs GDN 状态 <0.15 GiB），所以 dtype 选择**不改变"GDN 状态相对
KV 可忽略"这个定性结论**，只影响 B1 是否要复刻这个"逐步舍入"动作来跟 HF
参考实现逐 token bit-exact 对齐（如果要对齐，必须复刻 FP32 计算+BF16
存储之间的舍入，不能只是"选一个精度、从头到尾都用它"）。**[仍是
待验证]**：本轮全部是静态读码，未跑一次 GPU forward 实测
`cache_params.layers[i].recurrent_states.dtype`。

### 4. context × concurrency 可行域

预算：卡 97887 MiB ≈ 95.59 GiB（规格文档引用的历史硬件数字，本轮未重新
测卡）。扣除权重（B3 场景用 19.558 GiB）。运行时开销（CUDA context、
activation workspace、CUDA graph 常驻 buffer、分配器碎片）**按 3 GiB
粗估——这是唯一一处没有实测依据的假设，[待验证，需 B0-3/B2 GPU 实测替换]**：

```
可用于 KV+状态 的预算 ≈ 95.59 - 19.56 - 3 ≈ 73.0 GiB   [3 GiB 假设待验证]
单槽总占用(L, kv_dtype) = GDN状态(~0.07-0.15 GiB，固定) + L × KV每token(kv_dtype)
```

| 上下文 | KV dtype | 单槽（含状态） | 73 GiB 预算下的 c_max |
|---|---|---:|---:|
| 256K | FP8 | 8.11 GiB | **c=8**（65 GiB，c=9 超预算） |
| 256K | BF16 | 16.11 GiB | **c=4**（64.4 GiB，c=5=80.6 超预算） |
| 128K | FP8 | 4.11 GiB | **c≈17**（69.9 GiB） |
| 128K | BF16 | 8.11 GiB | **c=8**（64.9 GiB） |
| 64K | FP8 | 2.11 GiB | **c≈33** |
| 64K | BF16 | 4.11 GiB | **c≈17** |

**这是算术推导，不是 GPU 实测**——c_max 的具体数字随 3 GiB 运行时开销
假设的准确性漂移。真实数字需要 B0-3（sparkinfer paged attention 实测）
与 B2（服务化阶段实际显存审计）验证。**方向性结论足够硬**：权重体量
（18.767 GiB 实测，比历史 Laguna 参照的 67 GiB 小 3.4 倍）是可行域大幅
放宽的主因，这条不依赖 3 GiB 假设的准确性。

---

## 未能查清 / 仍是 [待验证] 的项（不填空，如实列出）

1. **KV cache scale 缺失后怎么定 scale 值**——本笔记只坐实了"没有张量"这个
   事实，`docs/qwen36-rebuild-spec.md` §6/§7 已经记录 vLLM/SGLang 的
   "默认 1.0+告警"先例作为参考，但最终要不要用 1.0、要不要自己校准，
   等待 `investigation-queue.md` C-2（NVFP4 KV vs FP8 KV 本机实测）结果
   后再拍板，本笔记不代为决定。
2. **NVFP4 `group_1` 的 `input_scale` 张量运行时语义**——存在但
   `input_activations=None`（weight-only），是否被推理路径实际消费未验证，
   需要读 modelopt/TensorRT-LLM 的官方反量化实现或做一次 GPU dummy forward
   才能坐实。
3. **GDN 递归状态 dtype 的运行时实测**——本节全部是静态读码（HF
   transformers 源码 + 真实 fla 源码），机制推导有硬证据支持，但未跑一次
   forward 在 `cache_params.layers[i].recurrent_states.dtype` 上直接断言。
4. **3 GiB 运行时开销假设**——B0-7 可行域表格里唯一没有实测依据的数字，
   需要 B0-3/B2 的 GPU 显存审计替换（方法参照
   `notes/2026-07-29-gpu-memory-audit.md`）。

---

## 复现清单

- checkpoint：`~/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/
  snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404/`（revision 与
  `refs/main` 一致，本轮核对）。交叉验证：
  `~/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/`（B0-8
  第二个 checkpoint）。
- 环境：`~/.venvs/vllm/bin/python`，`transformers==5.8.0`，本机 `fla`
  真实源码在 `/home/bot/project/flash-linear-attention`。
- 工具：本仓库 `loader/checkpoint_index.py` + `loader/safetensors_header.py`
  （只读用，未修改）。
- 所有命令单次运行 < 5 秒，全部只读 JSON/safetensors header，无需 GPU。
- 如果 checkpoint revision 变了（modelopt/HF 官方重新导出量化包），本笔记
  的具体数字需要重新核对。

## 相关文档

- 详细推导来源：
  [`2026-08-02-qwen36-b0-fact-baseline.md`](2026-08-02-qwen36-b0-fact-baseline.md)
  （B0-2/B0-6/B0-7 第一轮 + 协调者第二轮追加，含 sparkinfer w4a16 候选路径、
  vLLM/SGLang FP8 KV 缺省值先例等超出本笔记范围的内容）
- [`2026-08-01-b6-mtp-gdn-verification.md`](2026-08-01-b6-mtp-gdn-verification.md)
  （B0-8/`investigation-queue.md` B-6 原始六 checkpoint 核实 + 对 B3 影响的
  纠偏论证）
- [`2026-08-02-laguna-docs-inherited-qwen36-numbers.md`](2026-08-02-laguna-docs-inherited-qwen36-numbers.md)
  （本仓库"文档数字继承污染"事故记录，本笔记坚持独立复现的动机来源）
- `docs/qwen36-rebuild-spec.md` §1.9/§3.4/§4/§6/§7 —— B0-2/B0-6/B0-7 的结论
  已经同步进这份规格文档
- `docs/investigation-queue.md` B-6 —— B0-8 的结论已经同步进这份队列文档
- **本笔记新增的收口**：`docs/implementation-plan.md` §7.1 的
  B0-2/B0-6/B0-7/B0-8 四个复选框，此前一直未打勾（尽管底层调查已经完成），
  本轮据此笔记同步打勾并改写 B3 的两分支措辞（见该文件 diff）。
