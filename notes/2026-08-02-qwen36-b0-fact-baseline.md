# Qwen3.6 Track B0 事实基线：B0-2（modelopt 张量/scale 语义）· B0-6（mrope 退化）· B0-7（容量测算）

> 编制日期：2026-08-02 · worktree `work/trackB-20260802` @ `22579b65`
> 环境：`~/.venvs/vllm/bin/python`（transformers `5.8.0`）· **全程零 GPU**，只读
> safetensors header/JSON config，用 `loader/safetensors_header.py` +
> `loader/checkpoint_index.py`（未改动这两个文件，只当工具用）。
>
> **checkpoint**：`~/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/
> snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404/`（revision `0893e16...`）。
> 对照组：`~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/
> snapshots/07614121.../`。
>
> **问题**：`docs/qwen36-rebuild-spec.md` §1.9/§3.4/§6 把 modelopt 的张量命名、
> scale 语义、KV cache scale 键名标成"本轮未定位到 modelopt 参考实现，不猜字段名"；
> §5.1/roadmap B0-6 需要确认 mrope 在纯文本下是否退化成 1D RoPE；roadmap B0-7 需要
> 一份 KV+GDN 状态显存账。本笔记逐项核实，**不代人拍板**，标注哪些是硬证据、
> 哪些仍是 [待验证]。
>
> **结论摘要**（详见下文）：
> 1. B0-2：checkpoint 是**混合精度**，不是"整模型 NVFP4"——GDN 与 self_attn 的
>    投影层是 **FP8**（权重+激活都量化），只有稠密 MLP 与 `lm_head` 是 **NVFP4
>    weight-only（W4A16）**。KV cache scale **在这份 checkpoint 里不存在任何张量**，
>    尽管 `quantization_config`/`hf_quant_config.json` 都声明了 `kv_cache_quant_algo:
>    FP8`——这是本笔记最大的一条纠偏。
> 2. B0-6：**确认退化**，有 HF transformers 源码行号证据：纯文本时 T/H/W 三个
>    mrope 维度的 position_ids 是同一个 `.expand()` 出来的视图，数值恒等，
>    `apply_interleaved_mrope` 的插值是对相同值的无操作重写。
> 3. B0-7：GDN 递归状态是**每槽固定 ~72–150 MiB**（取决于状态 dtype，[待验证]，
>    见下文），与上下文长度无关；权重只有 **18.8 GiB**（纯文本，不含 vision/MTP）
>    ——比 Laguna 的 67 GiB 权重小得多，96GB 卡上的可行域比 Laguna 宽松很多。

---

## 0. 方法

```bash
cd /home/bot/project/qsr-w-trackB
~/.venvs/vllm/bin/python -c "
from pathlib import Path
from loader.checkpoint_index import load_checkpoint_index
from loader.safetensors_header import read_safetensors_header
model_dir = next((Path.home()/'.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots').iterdir())
idx = load_checkpoint_index(model_dir)
headers = {s: read_safetensors_header(model_dir/s) for s in idx.shard_names}
# ... 见下文每节的具体查询
"
```

只读了 `config.json`、`hf_quant_config.json`、`model.safetensors.index.json`、
三个 shard 的 safetensors JSON header（8 字节长度 + JSON，不读 tensor 数据），
以及本机已装的 `transformers==5.8.0` 的 `models/qwen3_5/modeling_qwen3_5.py` 源码。
**没有加载任何权重进内存，没有跑任何模型前向，没有碰 GPU。**

---

## 1. B0-2 · modelopt NVFP4 的张量命名与 scale 语义

### 1.1 checkpoint 是混合精度，不是"整模型 NVFP4"

`config.json` 顶层 `quantization_config`（与同目录 `hf_quant_config.json` 的
`quantization.*` **完全一致**，401 条 `quantized_layers` 条目逐一相同，`ignore`/
`exclude_modules` 都是 `['mtp*', 'mtp.layers.0*']`——两份文件互为镜像，没有冲突,
**以哪个为准这个问题在这份 checkpoint 上不存在分歧**；本仓库加载器走
`config.json`，因为那是 `transformers`/我们自己 `runtime/model_registry.py` 已经在读的路径，`hf_quant_config.json` 是 TensorRT-LLM/ModelOpt 生态的旁路副本，不需要单独解析）：

```
quant_algo (top level) = "MIXED_PRECISION"
producer = {"name": "modelopt", "version": "0.45.0"}
kv_cache_scheme (config.json) / kv_cache_quant_algo (hf_quant_config.json) = FP8, dynamic=False
ignore / exclude_modules = ['mtp*', 'mtp.layers.0*']
```

两个 `config_groups`：

| group | num_bits | type | group_size | input_activations | targets |
|---|---|---|---|---|---|
| `group_0` | 8 | float | — | 同样 8-bit float（静态） | `self_attn.{q,k,v,o}_proj`（16 个 full-attention 层）+ `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`（48 个 GDN 层）——**208 个目标** |
| `group_1` | 4 | float | 16 | `None`（未声明——weight-only） | `mlp.{gate,up,down}_proj`（64 层全部）+ `lm_head`——**193 个目标** |

`quantized_layers`（`hf_quant_config.json`）给每个目标标了具体算法名，验证了上表：
GDN/self_attn 投影层清一色 `"quant_algo": "FP8"`；MLP/lm_head 清一色
`"quant_algo": "W4A16_NVFP4", "group_size": 16`。**"W4A16" 是 modelopt 自己的命名
约定——weight 4-bit、activation 16-bit，即 weight-only 量化**，不是 Laguna 那种
weight+activation 都量化的 W4A4。

**这与规格文档 §4"已确认的事实"里"modelopt NVFP4 + fp8 KV"这句话不够精确**——
准确说法应该是：**只有稠密 MLP 与 lm_head 是 NVFP4（且是 weight-only），
全部注意力投影（self_attn 与 GDN 的 in_proj/out_proj）是 FP8（权重+激活都量化），
KV cache 的 FP8 声明没有对应的实际 scale 张量**（见 1.3）。GDN 的
`A_log`/`dt_bias`/`conv1d.weight`/`in_proj_a`/`in_proj_b`/`linear_attn.norm.weight`
以及所有 RMSNorm/`q_norm`/`k_norm`/`embed_tokens` 都是 **BF16，完全不量化**。
`mtp.*`（15 个张量）整体不量化（`exclude_modules` 命中），也是 BF16。

### 1.2 张量命名规律（逐后缀）

复现（对 layer 0 = GDN 层、layer 3 = full-attention 层各转储全部张量名）：

```bash
~/.venvs/vllm/bin/python -c "
from pathlib import Path
from loader.checkpoint_index import load_checkpoint_index
model_dir = next((Path.home()/'.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots').iterdir())
idx = load_checkpoint_index(model_dir)
names = sorted(idx.weight_map)
for n in names:
    if n.startswith('model.language_model.layers.0.') or n.startswith('model.language_model.layers.3.'):
        print(n)
"
```

FP8（`group_0`，self_attn 与 GDN 的 in_proj/out_proj）三件套：

```
<prefix>.weight          F8_E4M3   [out, in]           # 直接 1 byte/元素，不 pack
<prefix>.weight_scale    F32       ()                  # per-tensor 标量，静态
<prefix>.input_scale     F32       ()                  # per-tensor 标量，静态（激活侧）
```

NVFP4（`group_1`，MLP 与 lm_head）四件套（双层 scale）：

```
<prefix>.weight          U8        [out, in//2]        # 2 个 FP4(E2M1) 元素 pack 进 1 byte
<prefix>.weight_scale    F8_E4M3   [out, in//16]        # per-block 尺度，block=16（沿 in 维）
<prefix>.weight_scale_2  F32       ()                  # per-tensor 全局尺度（二级 scale）
<prefix>.input_scale     F32       ()                  # 标量，**存在但 group_1 的 input_activations=None**——
                                                        # 大概率是校准期留下的 amax，推理时未必被消费，[待验证]
```

实测（`mlp.down_proj`：逻辑形状 `[5120, 17408]`）：

```
weight:          U8      shape=(5120, 8704)     # 8704 = 17408/2 ✓ pack 验证
weight_scale:     F8_E4M3 shape=(5120, 1088)     # 1088 = 17408/16 ✓ group_size=16 验证
weight_scale_2:   F32     shape=()
input_scale:      F32     shape=()
```

不量化的 GDN 专属张量（每层 1 份，48 层）：

```
linear_attn.A_log        BF16  (48,)              # num_v_heads=48，delta-rule 的对数衰减参数
linear_attn.dt_bias      BF16  (48,)              # 同上，时间步偏置
linear_attn.conv1d.weight BF16 (10240, 1, 4)       # 10240=key_dim*2+value_dim，depthwise，kernel=4
linear_attn.in_proj_a.weight BF16 (48, 5120)       # 只到 num_v_heads=48（每头一个标量），未量化
linear_attn.in_proj_b.weight BF16 (48, 5120)       # 同上（delta-rule 的 beta 门）
linear_attn.norm.weight   BF16  (128,)             # head_v_dim=128 的 gated RMSNorm
```

**这套命名与规格文档 §1.9 提到的、从 `a9cb932^` 恢复的 compressed-tensors 版
`nvfp4_linear.py`（`weight_packed`/`weight_global_scale`/`input_global_scale`）
完全不同**——modelopt 用 `weight`（不是 `weight_packed`）+ `weight_scale`（block，
不是 Laguna 那种"weight_scale=block、weight_global_scale=全局"的二段命名，
modelopt 是 `weight_scale`=block、`weight_scale_2`=全局）。移植那份代码时不能
按名字模式匹配，必须按新命名重写字段访问。

### 1.3 KV cache scale：声明存在，张量不存在（本节最大发现）

`config.json`/`hf_quant_config.json` 都写 `kv_cache_quant_algo: "FP8"`,
`kv_cache_scheme.dynamic: false`（静态）。但对全部 2194 个张量名做穷举
grep（`scale`/`amax`/`kv`/`cache` 四个子串），**零命中 `k_scale`/`v_scale`/
`kv_scale`/`kv_cache_scale` 或任何 "amax" 张量**：

```bash
~/.venvs/vllm/bin/python -c "
from pathlib import Path
from loader.checkpoint_index import load_checkpoint_index
model_dir = next((Path.home()/'.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots').iterdir())
idx = load_checkpoint_index(model_dir)
names = sorted(idx.weight_map)
for pat in ['scale','amax','kv','cache']:
    hits=[n for n in names if pat in n.lower()]
    print(pat, sorted(set(n.split('.')[-1] for n in hits)))
"
# -> scale: ['input_scale','weight_scale','weight_scale_2']  （零 kv 相关后缀）
# -> kv:    ['bias','input_scale','weight','weight_scale']   （全部来自 in_proj_qkv 命名里的 "kv" 子串，
#           不是真的 kv-cache scale）
# -> amax, cache: 空
```

**对照组坐实这不是"静态量化就该没有 scale 张量"的一般规律**：`poolside/
Laguna-S-2.1-NVFP4` 的 `kv_cache_scheme` 也是 `dynamic: False`，但它**真的有**
`model.layers.N.self_attn.k_scale`/`v_scale`（BF16，shape `(1,)`，48 层各一对，
实测验证）。所以 modelopt 这份 checkpoint 的 KV FP8 声明是"元数据里写了、
权重里没落地"，不是这个格式的通例。

**推论（[待验证]，需要人拍板或后续实测排查）**：
1. 要么这份 checkpoint 假设下游推理栈（TensorRT-LLM？）自己跑一遍 KV 校准
   补上 scale，公开的 HF 权重故意不带；
2. 要么默认约定是 scale=1.0（直接 cast，不做真正的仿射变换）；
3. 要么这份 checkpoint 的 FP8 KV 声明对我们不适用，B3 阶段该用 BF16 KV
   或自己校准一遍 FP8 scale。

这条不由本笔记下结论——`investigation-queue.md` C-2 的 NVFP4 KV vs FP8 KV
调查如果得出"用 FP8 KV"，会立刻撞上"scale 从哪来"这个新问题，需要一并考虑。

### 1.4 vision 张量：333 个，前缀唯一且干净

```bash
~/.venvs/vllm/bin/python -c "
from pathlib import Path
from loader.checkpoint_index import load_checkpoint_index
model_dir = next((Path.home()/'.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots').iterdir())
idx = load_checkpoint_index(model_dir)
names = sorted(idx.weight_map)
vision = [n for n in names if n.startswith('model.visual.')]
stray = [n for n in names if 'vis' in n.lower() and not n.startswith('model.visual.')]
print(len(vision), stray)   # -> 333 []
"
```

**333 精确复现**（与 roadmap/spec 引用的数字一致），前缀恒为 `model.visual.`，
没有任何"vision"字样张量落在这个前缀之外——**排除过滤器可以简单到
`name.startswith("model.visual.")`**，不需要更复杂的正则或白名单。vision
张量本身是 BF16，未量化，总大小 0.858 GiB（占整包 20.4 GiB 的 4.2%）。

顶层 `config.json` 的 `language_model_only: false`——checkpoint 自己的默认配置
**不会**帮我们把 vision 关掉，`validate_text_only`/loader 必须主动做这件事，
不能假设 checkpoint 元数据已经声明了纯文本模式。

### 1.5 MTP 头：15 个张量，不量化，且比 backbone 多两个 norm

`mtp.*` 张量清单（15 个，全部 BF16，无 `*_scale` 伴生张量）：

```
mtp.fc.weight                        (5120, 10240)   # 10240=2×5120：拼接(prev_hidden, next_token_embed)
mtp.pre_fc_norm_embedding.weight     (5120,)
mtp.pre_fc_norm_hidden.weight        (5120,)
mtp.norm.weight                      (5120,)
mtp.layers.0.{input_layernorm,post_attention_layernorm}.weight
mtp.layers.0.self_attn.{q,k,v,o}_proj.weight   # 无量化伴生张量，纯 BF16
mtp.layers.0.self_attn.{q,k}_norm.weight
mtp.layers.0.mlp.{gate,up,down}_proj.weight    # 无量化伴生张量，纯 BF16
```

`mtp.fc.weight` 形状 `(5120, 10240)` 与 EAGLE 式 MTP 头的标准做法吻合：
拼接"当前位置的真实下一个 token 的 embedding"与"上一层 backbone 隐状态"
（各 5120，共 10240）后投影回 5120。`pre_fc_norm_embedding`/`pre_fc_norm_hidden`
是拼接前分别作用在两路输入上的两个 RMSNorm——这与已有结论
（`notes/2026-08-01-b6-mtp-gdn-verification.md`：MTP 层零 GDN）完全一致，
本轮未发现矛盾证据，只是补充了张量级细节。**这批 MTP 张量全程 BF16，
说明 B3 实现 MTP draft/verify 时这部分不需要走任何 FP8/NVFP4 反量化路径**。

### 1.6 与 Laguna（compressed-tensors）逐项对照表

对照来源：本轮对 `poolside/Laguna-S-2.1-NVFP4` 的实测（同方法），交叉验证
`runtime/model/_weight_loading.py`、`notes/2026-07-24-sparkinfer-moe-integration.md`、
`docs/architecture.md` §3.2-D。

| 维度 | Laguna（compressed-tensors） | Qwen3.6-27B-NVFP4（modelopt） |
|---|---|---|
| `quant_method` | `compressed-tensors` | `modelopt` |
| 格式标签 | `format: "nvfp4-pack-quantized"` | 无 `format` 字段；`producer.name=modelopt`, `quant_algo=MIXED_PRECISION` |
| 哪些层是 NVFP4 | 仅 MoE 专家 FFN（`experts.N.{gate,up,down}_proj`），正则 target；**W4A4**（`input_activations` 也声明 4-bit/group16/dynamic=local） | 稠密 MLP（`{gate,up,down}_proj`，全 64 层）+ `lm_head`；**W4A16**（`input_activations=None`，weight-only） |
| 哪些层是 FP8 | 无（`config_groups` 里没有 FP8 组） | self_attn 全部投影 + GDN 的 `in_proj_qkv/in_proj_z/out_proj`（W8A8，权重+激活都静态量化） |
| 哪些层不量化（BF16） | self_attn q/k/v/o/g_proj、MoE gate、shared_expert、`lm_head`、layer0 MLP（`ignore` 列表命中） | `mtp.*` 全部；GDN 的 `A_log/dt_bias/conv1d/in_proj_a/in_proj_b/norm`；`embed_tokens`；所有 RMSNorm/`q_norm`/`k_norm` |
| 权重张量名 | `.weight_packed`（U8 pack）+ `.weight_scale`（F8_E4M3, block=16）+ `.weight_global_scale`（F32 标量） | NVFP4 目标：`.weight`（U8 pack）+ `.weight_scale`（F8_E4M3, block=16）+ `.weight_scale_2`（F32 标量）；FP8 目标：`.weight`（F8_E4M3，不 pack）+ `.weight_scale`（F32 标量） |
| 激活 scale 名 | `.input_global_scale`（F32 标量，真实参与 W4A4 的动态量化） | `.input_scale`（F32 标量；FP8 组必需且被消费；NVFP4 组存在但语义未定，[待验证]） |
| KV cache scale | **真实张量**：`.self_attn.k_scale`/`.self_attn.v_scale`，BF16，`(1,)`，48 层各一对 | **不存在任何张量**，尽管声明了 `kv_cache_quant_algo=FP8`（1.3 节） |
| block scale dtype | F8_E4M3 | F8_E4M3（相同） |
| 全局 scale dtype | F32 | F32（相同） |
| block/group_size | 16 | 16（相同） |
| pack 约定 | U8，2 个 FP4(E2M1) 元素/byte | 相同约定（NVFP4 目标才 pack；FP8 目标不 pack，1 byte/元素直存） |
| MoE / 稠密 | MoE（`mlp.experts.N.*`） | 稠密（每层单一 `gate/up/down_proj`，无专家维度） |
| declared where | 仅 `config.json` `quantization_config` | `config.json` 与 `hf_quant_config.json` 两份，内容一致 |
| loader 额外过滤 | 无 | 必须过滤 `model.visual.*`（333 个）；`mtp*` 层要走"零 scale 张量"的无量化分支 |

**给 A4 加载器 adapter 的直接结论**：不能把 compressed-tensors 的
`weight_packed`/`weight_global_scale` 字段名简单换成 modelopt 的
`weight`/`weight_scale_2`就完事——**还要按每个 module 的量化算法
（FP8 / NVFP4-weight-only / 不量化）走三条不同的加载分支**，因为同一个
`.weight` 后缀在这份 checkpoint 里可能是 U8-packed-NVFP4、也可能是
F8_E4M3-unpacked、也可能是 BF16 明文——**必须先查 `quantization_config.
quantized_layers[name]`（或用命名规则推导所属层类型+ignore 列表）才能知道
怎么解释 `.weight` 张量本身，不能只看后缀名**。这是与 Laguna 加载器最大的
结构性差异：Laguna 的 `weight_packed` 后缀本身就唯一标识了"这是 NVFP4"，
modelopt 没有这种自解释后缀。

### 1.7 一个跟量化无关、但同样是"命名不能猜"的发现：`q_proj` 输出宽度是 2×

实测 `model.language_model.layers.3.self_attn.q_proj.weight` 形状
`(12288, 5120)`；`k_proj`/`v_proj` 都是 `(1024, 5120)` = `4×256`（符合
`num_key_value_heads=4, head_dim=256`）；`o_proj` 输入维 `6144` =
`24×256`（符合 `num_attention_heads=24`）。但 `q_proj` 输出 `12288 = 2×6144`，
不是预期的 `6144`。

查 `transformers/models/qwen3_5/modeling_qwen3_5.py:643-644`：

```python
self.q_proj = nn.Linear(
    config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
)
```

`:669-672,700`：

```python
query_states, gate = torch.chunk(
    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
)
...
attn_output = attn_output * torch.sigmoid(gate)
```

**`attn_output_gate` 的门控信号是从 `q_proj` 里"顺便"多算出来的一半**，
不是独立的权重矩阵——每个 head 的 `q_proj` 输出被切成前 256 维是真正的 Q，
后 256 维是这个 head 的 gate logit，`sigmoid(gate)` 乘在 attention 输出（不是
`silu`；`text_config.output_gate_type: "swish"` 这个字段名容易让人以为是
`silu(x)=x*sigmoid(x)`，但实际代码是纯 `sigmoid` 门控，不是 swish 激活）。
这条不是 B0-2 的量化命名，但同样是"加载器/模型图必须知道的具体形状事实，
不能按 `num_heads*head_dim` 直接假设"，一并记录在这里给 B1 模型图实现者。

同一文件还确认了另一条已有的猜测：`Qwen3_5RMSNorm.forward`
（:722-736）用 `output * (1.0 + self.weight)`（零中心 gamma），与
`oracle/qwen36_vllm/gemma_norm_patch.py` 的说法一致——本轮在 HF 参考实现里
直接找到代码确认，不再是"oracle 自称"。

---

## 2. B0-6 · mrope-interleaved 在纯文本下确认退化为标准 1D RoPE

**结论：确认退化，有代码行号证据，不是"应该可以"。**

### 2.1 退化机制

`Qwen3_5TextRotaryEmbedding.apply_interleaved_mrope`
（`modeling_qwen3_5.py:157-172`）：

```python
def apply_interleaved_mrope(self, freqs, mrope_section):
    freqs_t = freqs[0]  # 先整体取 T（text/temporal）维
    for dim, offset in enumerate((1, 2), start=1):  # H, W
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]   # 用 H/W 维的值覆盖对应位置
    return freqs_t
```

`freqs` 的 shape 是 `(3, bs, seq, head_dim//2)`，三个切片分别来自
`position_ids[0/1/2]`（T/H/W）。**这个函数纯粹是"用 freqs[1]/freqs[2] 覆盖
freqs[0] 的部分位置"，不做任何跨维度的数学组合**——如果 `freqs[0]==freqs[1]
==freqs[2]`（逐元素相等），覆盖操作是数值上的无操作（overwrite 到相同的值）。

### 2.2 纯文本时 T/H/W 三个 position_ids 恒等

**入口 A：`Qwen3_5TextModel.forward`（纯文本 decoder backbone，
`modeling_qwen3_5.py:1220-1294`）**——这是 `language_model` 子模块本身的
forward，行 1257-1263：

```python
# the hard coded `4` is for text, temporal, height and width.
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

`.expand(4, ...)` 是同一个 1D `arange` 张量的 4 份**视图**（不是 4 份独立数据），
所以 `position_ids[1:]` 里的 T/H/W 三份逐元素相等——`apply_interleaved_mrope`
覆盖的是相同的数值，`freqs_t` 最终等于纯粹用单一线性位置序列算出来的
标准 RoPE（叠加 `partial_rotary_factor=0.25`，只旋转 `head_dim*0.25=64` 维，
这是与 mrope 无关的正交细节）。

**入口 B：`Qwen3_5Model.compute_3d_position_ids`（多模态外层包装，
`modeling_qwen3_5.py:1575-1622`）**——纯文本请求时 `image_grid_thw`/
`video_grid_thw` 都是 `None`，`has_multimodal = False`（行 1586），
`can_compute_mrope = False`（行 1593）。因为 `self.rope_deltas` 在没有任何
多模态帧参与计算的会话里永远不会被设置（只在 `if can_compute_mrope` 分支
里赋值，行 1595-1603），后续的 `elif self.rope_deltas is not None and ...`
（行 1608）恒为假，**每一次调用（prefill 与每一步 decode）都落到最后的
`else: position_ids = None`**（行 1619-1621）。这个 `None` 原样传进入口 A
的 `Qwen3_5TextModel.forward`，回到上面完全相同的退化路径。**prefill 和
decode 路径都验证过是同一条退化路径，不存在"prefill 退化、decode 不退化"
的分歧**。

### 2.3 结论对 Track B 的影响

纯文本场景下，Qwen3.6 的 RoPE 数学上等价于：**标准 1D RoPE + partial_rotary_
factor=0.25（仅旋转 head_dim 的前 64 维）+ rope_theta=10,000,000**（`config.
json` 的 `rope_parameters.rope_theta`，注意不是默认的 10000，差 3 个数量级，
新实现如果抄错这个常数会在长上下文位置上产生可测的漂移）。**可以直接复用
现有 `runtime/kernels/rope.py` 的标准 1D RoPE kernel，加一个"只转前 25%
维度"的 partial-rotary 支持即可，不需要实现真正的三维 mrope 插值逻辑**——
这条决定性地回答了 B0-6 的原始问题，`mrope_section=[11,11,10]`/`mrope_
interleaved=true` 这两个 config 字段在纯文本 checkpoint 加载路径上可以被
安全忽略（只在真的送入图像/视频时才生效，而 D6 已经拍板走纯文本 checkpoint
且断言零 vision 张量加载，所以这两个字段在我们的服务场景里永远不会被触发）。

**唯一需要注意的边界情况**：如果未来某个请求携带了 `image_grid_thw`
之类的字段（哪怕文本 checkpoint 忽略 vision 权重，请求层面理论上仍可能被
上游传入这些字段），`can_compute_mrope` 会变 True，退化条件就不成立。
D6 的"断言零 vision 张量加载"只保证权重侧干净，不保证请求侧不会误传
多模态字段——这个输入校验应该在 API 层或 `ArchitectureSpec` 校验时一并
拒绝，不是本笔记职责范围，只在此提醒。

---

## 3. B0-7 · 容量测算：64 层 / 256K / 96GB 的可行域

### 3.1 权重大小（实测，不是估算）

对三个 shard 的 `data_offsets` 求和（不读数据，只读 header 里每个张量声明的
字节区间）：

```bash
~/.venvs/vllm/bin/python -c "
from pathlib import Path
from loader.checkpoint_index import load_checkpoint_index
from loader.safetensors_header import read_safetensors_header
model_dir = next((Path.home()/'.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/snapshots').iterdir())
idx = load_checkpoint_index(model_dir)
headers = {s: read_safetensors_header(model_dir/s) for s in idx.shard_names}
def nbytes(n):
    lo,hi = headers[idx.weight_map[n]][n].data_offsets
    return hi-lo
total=vision=mtp=0
for n in idx.weight_map:
    b=nbytes(n); total+=b
    if n.startswith('model.visual.'): vision+=b
    elif n.startswith('mtp.'): mtp+=b
GiB=1024**3
print(f'total={total/GiB:.3f} vision={vision/GiB:.3f} mtp={mtp/GiB:.3f} lm(rest)={(total-vision-mtp)/GiB:.3f}')
"
```

结果（GiB）：

| 部分 | 大小 | 说明 |
|---|---:|---|
| 全部 3 个 shard 总和 | 20.416 | 与 `model.safetensors.index.json` 的 `total_size` 字段完全一致（交叉验证通过） |
| vision（`model.visual.*`，排除对象） | 0.858 | D6 拍板要跳过的 333 个张量 |
| MTP 头（`mtp.*`） | 0.791 | B1 不需要，B3 投机解码需要 |
| **backbone + `lm_head`（纯文本、无 MTP，实际驻留显存的权重）** | **18.767** | B1 阶段实际要加载的量 |
| backbone + `lm_head` + MTP 头 | 19.558 | B3 阶段实际要加载的量 |

**这比 Laguna 的 67 GiB 权重小得多**（本任务背景数字，未在本笔记重新验证
Laguna 那侧，只做量级对比）——Qwen3.6-27B 参数量本身更小，且混合精度下
大部分参数（稠密 MLP）是 4-bit，注意力投影是 8-bit，平均每参数字节数明显
低于 Laguna。这直接决定了下面的可行域比 Laguns 宽松很多。

### 3.2 KV cache（16 个 full-attention 层，与上下文长度线性相关）

每 token 每层：`2（K+V）× kv_heads(4) × head_dim(256) × dtype_bytes`。
16 层合计：`16 × 2 × 4 × 256 × dtype_bytes = 32768 × dtype_bytes` bytes/token。

| KV dtype | 每 token 每槽 | 256K（262144 token） | 128K | 64K |
|---|---:|---:|---:|---:|
| FP8（1 byte，**scale 来源见 §1.3 待验证**） | 32 KiB | 8.0 GiB/槽 | 4.0 GiB/槽 | 2.0 GiB/槽 |
| BF16（2 byte） | 64 KiB | 16.0 GiB/槽 | 8.0 GiB/槽 | 4.0 GiB/槽 |

（这些是精确算术，不是估算：`262144×32768/1024^3 = 8.0` 恰好整除，
因为 head_dim/kv_heads/layer 数都是 2 的幂。）

### 3.3 GDN 递归状态（48 层，与上下文长度**无关**，每槽固定）

架构参数（`config.json` `text_config`，逐字段实测）：

```
linear_num_key_heads=16  linear_key_head_dim=128
linear_num_value_heads=48  linear_value_head_dim=128
linear_conv_kernel_dim=4
```

`transformers/models/qwen3_5/modeling_qwen3_5.py:505-507` 确认 K/Q 会
`repeat_interleave` 从 16 头扩到 48 头（`num_v_heads // num_k_heads = 3`）
再进 delta-rule，所以递归状态（累积外积矩阵）的头数维度是 **48**，不是 16：

```python
if self.num_v_heads // self.num_k_heads > 1:
    query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
```

单层 SSM 状态元素数：`48 heads × 128(k) × 128(v) = 786,432` 元素。
单层 conv1d 状态：`conv_dim(10240) × (kernel-1=3) = 30,720` 元素
（`conv_dim = key_dim×2 + value_dim = 128×16×2 + 128×48 = 4096+6144=10240`，
与实测的 `conv1d.weight` shape `(10240,1,4)` 完全对应）。

**状态 dtype 是本节唯一的 [待验证] 项**：`config.json` 声明
`mamba_ssm_dtype: "float32"`，但穷举 `transformers/models/qwen3_5/
modeling_qwen3_5.py` 全文 grep `mamba_ssm_dtype`/`ssm_dtype`**零命中**——
这个字段在当前装的 `transformers==5.8.0` 参考实现里**未被读取**。真正决定
存储 dtype 的是 `transformers/cache_utils.py:773,777,784`：

```python
self.dtype, self.device = conv_states.dtype, conv_states.device   # :773，来自调用方传入的张量
self.conv_states = torch.zeros_like(conv_states, dtype=self.dtype, ...)      # :777
self.recurrent_states = torch.zeros_like(recurrent_states, dtype=self.dtype, ...)  # :784
```

`conv_states`/`recurrent_states` 的 dtype 由**调用方**（`mixed_qkv`/
`in_proj_a`/`in_proj_b` 的输出，即模型的 bf16 计算 dtype）决定——也就是说
**HF 参考实现的持久化状态实际是 BF16，`mamba_ssm_dtype: float32` 这个
config 字段目前看起来是摆设**（未在本笔记范围内实跑代码验证这个结论，
只是静态读码，标 **[待验证，建议后续用一次 CPU-only 的 dummy forward
实测确认]**）。生产实现（vLLM 的 Mamba2Cache、FLA 的 kernel）历史上惯例是
用 fp32 累积递归状态以保数值稳定性，这条"该用 fp32 还是 bf16"的选择会
直接影响 B1"与 HF 参考逐 token 对齐"这条门禁该对齐到哪个精度基准，
不是纯粹的性能选择——**这是留给 B1 实现者的一个决定点，本笔记只把两边的
证据摆出来，不代为拍板**。

单槽总大小（48 层合计，两种 dtype 假设）：

| 状态 dtype 假设 | SSM 状态（48层） | conv 状态（48层，假设同 dtype） | 单槽合计 |
|---|---:|---:|---:|
| FP32（4 byte，匹配 `mamba_ssm_dtype` 声明 / 生产惯例） | 144.0 MiB | 5.6 MiB | **~149.6 MiB** |
| BF16（2 byte，匹配当前 HF 参考实测代码路径） | 72.0 MiB | 2.8 MiB | **~74.8 MiB** |

**交叉验证**：规格文档 §2.4 记录了 vLLM 时代的历史实测数字
"GDN 快照缓冲区（4 槽）~604 MB VRAM"，即 **~151 MB/槽**——与本节 FP32 假设
算出的 149.6 MiB/槽**几乎精确吻合**（差 <1%，量级在预期的分配对齐/padding
误差范围内）。这个吻合强烈支持"当年 vLLM 实现用的是 FP32 递归状态"，
即历史基线更接近 FP32 那一行。**不论选哪个 dtype，这个数字都是两位数到
三位数 MiB 量级，在下面 3.4 节的容量表里完全被 KV cache 淹没——dtype 选择
不改变"GDN 状态相对 KV 可忽略"这个定性结论，只影响 ~75 MiB 的具体数字**。

### 3.4 context × concurrency 可行域（算术推导，非 GPU 实测）

预算：卡是 97887 MiB ≈ 95.59 GiB（规格文档 §2 引用的历史硬件数字，本笔记
未重新测卡，只做单位换算）。扣除权重（用 3.1 节的 backbone+lm_head+MTP =
19.558 GiB，B3 场景，最保守）。**运行时开销（CUDA context、activation
workspace、CUDA graph 常驻 buffer、分配器碎片）本笔记按 3 GiB 粗估
——这是唯一一处没有实测依据的假设，[待验证，需 B0-3/B2 GPU 实测替换]**：

```
可用于 KV+状态 的预算 ≈ 95.59 - 19.56 - 3 ≈ 73.0 GiB   [待验证的 3 GiB 假设]
单槽总占用(L, kv_dtype) = GDN状态(~0.073~0.146 GiB，固定) + L × KV每token(kv_dtype)
c 槽总占用 = c × 单槽总占用
```

代入几个 headline 长度（GDN 状态取两种 dtype 假设的中点 ~0.11 GiB，
对总量影响 <0.5%，不影响下表任何一个可行/不可行的判断）：

| 上下文 | KV dtype | 单槽（含状态） | c=1 | c=2 | c=4 | c=8 | 73 GiB 预算下的 c_max |
|---|---|---:|---:|---:|---:|---:|---:|
| 256K | FP8 | 8.11 GiB | 8.1 | 16.2 | 32.4 | 64.9 | **c=8**（65 GiB，c=9 超预算） |
| 256K | BF16 | 16.11 GiB | 16.1 | 32.2 | 64.4 | 128.9 ✗ | **c=4**（64.4 GiB，c=5=80.6 超预算） |
| 128K | FP8 | 4.11 GiB | — | — | 16.4 | 32.9 | **c≈17**（69.9 GiB） |
| 128K | BF16 | 8.11 GiB | — | — | 32.4 | 64.9 | **c=8**（64.9 GiB） |
| 64K | FP8 | 2.11 GiB | — | — | 8.4 | 16.9 | **c≈33** |
| 64K | BF16 | 4.11 GiB | — | — | 16.4 | 32.9 | **c≈17** |

**对照历史参照点**：规格文档 §2.4 记录 vLLM 时代 Laguna(sic，实为 Qwen3.6
时代自己的历史数字，见 spec §2.4 表)在 128K/c=4/warm 实测 90.7–92.9 GiB
——那是 67 GiB 权重 + 4 槽 KV/状态 挤爆 96GB 卡的结果。**本笔记的算术推导
显示新框架下同样 128K/c=4 只需要 ~32.4 GiB（FP8 KV）——差距主要来自权重
从 67 GiB 降到 19.6 GiB（省了 47 GiB），不是 KV 记账方式本身有本质变化**。
这与 roadmap §7 D-something 及 spec §6 的提醒一致："旧参照数字不能直接套
新框架"——本节提供的是**方向一致但数值更宽松**的新算术基线，仍然
**不是 GPU 实测**，c_max 的具体数字随 3 GiB 运行时开销假设的准确性而漂移，
真正的数字需要 B0-3（sparkinfer paged attention 实测）与 B2（服务化阶段
实际显存审计，参照 `notes/2026-07-29-gpu-memory-audit.md` 的方法）验证。

### 3.5 对规格文档 §2.4/§2.5"旧参照数字不能直接套用"的定量支持

上一节的推导定量印证了规格 §6 待验证清单里那句预警——不仅"记账方式不同"，
连**权重体量本身就差 3.4 倍**（67 GiB vs 19.6 GiB），这是可行域大幅放宽的
主因，值得写进结论而不是只留一句"需要重新测"。

---

## 4. 给 `docs/qwen36-rebuild-spec.md` 的具体更正（已同步过去，见该文件 diff）

1. §1.9"格式警告"："本轮未定位到 modelopt 参考实现，不猜字段名"这句已过时，
   替换为指向本笔记 §1 的具体命名表。
2. §3.4"加载器 adapter"：补充真实字段名与"按量化算法分三支"的结论（本笔记 §1.6）。
3. §4"已确认的事实"："modelopt NVFP4 + fp8 KV" 改成准确描述（混合精度 +
   KV scale 缺失，本笔记 §1.1/§1.3）。
4. §6 待验证清单：勾掉 B0-2/B0-6/B0-7 对应条目，新增：
   - KV cache scale 缺失的处理方式（§1.3）
   - GDN 状态 dtype：fp32 声明 vs bf16 实测代码路径（§3.3）
   - modelopt `input_scale` 在 W4A16 组的语义（§1.2）
   - `q_proj` 2× 宽度融合 gate 的加载器/模型图含义（§1.7）

---

## 5. 复现清单（给下一个查这个 checkpoint 的人）

- 所有命令见各节代码块，均只读 `config.json`/`hf_quant_config.json`/
  `model.safetensors.index.json`/safetensors JSON header，无需 GPU，
  单次运行 <5 秒。
- checkpoint revision：`0893e1606ff3d5f97a441f405d5fc541a6bdf404`
  （`~/.cache/huggingface/hub/models--nvidia--Qwen3.6-27B-NVFP4/refs/main`）。
  **如果这个 revision 变了，本笔记的具体数字需要重新核对**——modelopt/HF
  官方偶尔会重新导出量化 checkpoint。
- HF transformers 版本：`5.8.0`（`~/.venvs/vllm/bin/python`）。B0-6 的结论
  绑定这个版本的 `modeling_qwen3_5.py` 具体实现；未来 transformers 升级
  如果重写了 mrope 实现，需要重新核对 §2 的行号引用（虽然"纯文本三维
  position_ids 恒等"这个数学事实大概率不会变，除非官方主动加了"即使
  纯文本也走不同 position_ids"的新行为）。
