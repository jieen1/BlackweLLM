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
>
> **2026-08-02 第二轮追加（同日，协调者跟进三个待拍板问题）**——见 §6/§7/§8：
> 4. sparkinfer 的 `moe._shared.kernels.w4a16` **已经原生支持 modelopt NVFP4
>    weight-only 语义**（`prepare_w4a16_modelopt_nvfp4_weights`，明确是为
>    "GLM 服务需要 A4 prefill + A16 decode 共享同一份权重"这个真实场景写的），
>    且底层 kernel 对 `num_experts=1` **没有任何下限限制**——不需要新功能，
>    通过公开的 `moe.fused_moe(quant_mode="w4a16")` API 配退化单专家路由即可，
>    **[待验证 GPU]** 未实跑。
> 5. GDN 递归状态 dtype：**原结论（BF16 持久化）成立，但原来的证据链是错的**
>    ——本机装的是真的 `fla`，实际走的是 FLA 真 kernel（内部用 FP32 计算），
>    不是我 8-2 第一轮读到的 torch 兜底函数；FP32 结果被 HF 通用 `Cache` 类
>    按 `conv_states` 先落的 dtype（BF16）**逐步降精度存回**——净效果是"单步
>    FP32 计算、跨步 BF16 落盘"，不是简单的"全程 BF16"或"全程 FP32"。
> 6. FP8 KV scale 缺失怎么办：vLLM/SGLang 都是**默认 1.0 + 告警**，vLLM 官方
>    还在弃用运行时校准选项，明确说"以后就是有则读、没有就 1.0"。

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
| FP8（1 byte，**scale 来源见 §1.3；上游框架怎么办见 §8**） | 32 KiB | 8.0 GiB/槽 | 4.0 GiB/槽 | 2.0 GiB/槽 |
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

**状态 dtype——2026-08-02 第二轮已核实，结论修正见 §7，这里保留第一轮原文
以留痕（第一轮读的是错误的代码分支，但巧合地得出了正确的落盘 dtype 结论）**：
`config.json` 声明 `mamba_ssm_dtype: "float32"`，但穷举 `transformers/models/qwen3_5/
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
**HF 参考实现的持久化状态实际是 BF16**。**[第一轮的推理有缺口，第二轮已补——
见 §7]**：第一轮以为这是因为 delta-rule 本身用了 bf16 计算的 torch 兜底函数；
实际上本机 `fla` 包真的能 import，参考实现走的是 FLA 的真 kernel，**FLA 的
kernel 内部用 FP32 计算 recurrent state**，是 HF 通用 `Cache` 类在写回持久化
buffer 时按 `conv_states` 先落的 dtype（BF16）把这个 FP32 结果**降精度存回**——
净效果同样是"落盘 BF16"，但机制是"单步 FP32 计算 + 跨步 BF16 舍入"，不是
"全程 BF16 计算"。这条区别对 B1 想做逐 token bit-exact 对齐是实质性的，见 §7。
生产实现（vLLM 的 Mamba2Cache、FLA 的 kernel 本身）历史上惯例用 fp32 累积
递归状态以保数值稳定性，这条"该用 fp32 还是 bf16"的选择直接影响 B1"与 HF
参考逐 token 对齐"这条门禁该对齐到哪个精度基准，不是纯粹的性能选择——
**这是留给 B1 实现者的一个决定点，本笔记只把两边的证据摆出来，不代为拍板**。

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

## 4. （2026-08-02 第二轮）sparkinfer 能不能吃 Qwen3.6 稠密 MLP 的 W4A16 语义

**问题**：Qwen3.6 稠密 MLP/`lm_head` 是 modelopt NVFP4 **weight-only**（§1.2，
激活留在 bf16，不量化到 FP4）。协调者判定 `sparkinfer.gemm.blockscaled.mm`
是对称 block-scaled GEMM（两个操作数都要 scale），吃不了这个语义，但指出
`sparkinfer/moe/_shared/kernels/w4a16/` 底下有 W4A16 实现（README 第 43 行：
"BF16 activations, inline FP4 weight dequant — no activation-scale math"），
可能被封装在 MoE 的 API 形状下。四个具体问题，逐一查（零 GPU，只读
`/home/bot/project/sparkinfer`，只读，未改动任何文件）。

### 4.0 先把 `sparkinfer/gemm/` 下所有 op 过一遍（协调者问题 4，优先做）

逐个读 `sparkinfer/gemm/*/api.py` 的模块 docstring 与依赖，9 个子模块：

| op | 底层原语 | 操作数 dtype 组合 | 能否给我们用 |
|---|---|---|---|
| `gemm.blockscaled` | `sparkinfer._lib.dense_gemm.dense_gemm` | `lhs`/`rhs` 都是 `(value, scale)` 二元组（签名 `lhs: Tuple[Tensor,Tensor], rhs: Tuple[Tensor,Tensor]`，`dense_gemm.py:6660-6668`）——**结构上强制两边都要 scale** | 不能，协调者判断已核实为真 |
| `gemm._bmm` | `_shared.mxfp8_bmm`，硬编码单一特化 | `a_dtype='bfloat16', b_dtype='float8_e4m3fn', sf_dtype='float8_e8m0fnu'`（`api.py:14-19`） | 不能——8-bit 权重不是 4-bit，且仍要 `sf_dtype`（B 侧有 scale） |
| `gemm.bf16_gemv` | 自成一体的小 N GEMV kernel | 纯 bf16×bf16，零量化；只服务 `N<=1024且K>=1024` 的窄投影（`api.py:21-23`，专为 GDN 的 `in_proj_ba` 这类窄层设计） | 不能——我们的 MLP 是 `N=17408`（gate/up）宽投影，且是权重量化不是纯 bf16 |
| `gemm.block_fp8_linear` | `_shared.block_fp8`（`dense_gemm`/`dense_gemm_fused_quant_a` + `quantize_block_fp8_linear_input_mxfp8`） | 权重 128×128 block-FP8，**激活也动态量化到 MXFP8**（`quantize_input` 是导出符号，见 `block_fp8_linear/api.py`） | 不能——8-bit 不是 4-bit，且激活仍被量化，不是 weight-only |
| `gemm.mxfp8_linear` | `_kernel.py`，同样基于 `dense_gemm` + `quantize_block_fp8_linear_input_mxfp8` | 权重 MXFP8，激活同样动态量化到 MXFP8（`_kernel.py:1-18` import 链） | 不能，同上 |
| `gemm.tensor_fp8_linear` | `_kernel.py`，基于 `dense_gemm` | 权重 per-tensor E4M3 + 静态 `scale_mma`/`output_scale`（`TensorFP8LinearWeight`），激活走同一 `dense_gemm` 路径（同样需要 A 侧 scale） | 不能——8-bit，且是对称量化 |
| `gemm.mla_query_projection` | `_shared.mxfp8_bmm` + 自建 `_bf16` 分支 | 权重可以是纯 BF16 **或** MXFP8；但 `q_nope [H,M,192]`/`q_pe [M,H,64]`/`out [M,H,576]` 是 **DeepSeek MLA 专用的硬编码维度**（`api.py:20-27`） | 不能——维度锁死在 MLA 的 192/64/576，不是通用 dense GEMM，且即便有 BF16 权重分支也是 8-bit 量化家族，不是 NVFP4 |
| `gemm.wo_projection` | `_shared.wo_mxfp8`（同样基于 `dense_gemm`） | MXFP8 权重+激活，"grouped WO-projection...用于 MLA attention"（README） | 不能——MLA 专用（协调者猜的候选，已排除），且仍是 8-bit 对称量化 |
| `gemm.trellis_linear` | **直接 import `moe._shared.kernels.w4a16.kernel.run_trellis256_dense`** | BF16/FP16 激活 × **EXL3 trellis 编码**权重（`suh`/`svh` Hadamard 旋转 + `mcg`/`mul1` 编码本，`trellis_linear/api.py:1-40`），3/4/5/6-bit，**不是 NVFP4 block-scale** | 语义最接近（bf16 激活、低比特权重、无激活 scale 数学）但**数值格式不对**——EXL3 trellis 码本+Hadamard 旋转，与 modelopt 的"NVFP4 block(16)+全局 scale"完全不同的编码，不能直接复用，要把 checkpoint 重新量化成 EXL3 才能用，代价太大、也偏离"官方 checkpoint 原样加载"的既定路线 |

**结论**：9 个 `gemm.*` op 里，除 `trellis_linear`（数值格式不对）外，其余全部
基于同一个对称 `dense_gemm` 原语，**结构性地要求两个操作数都带 scale**——
协调者的判断完全正确，**`sparkinfer/gemm/` 下没有第三条路**（问题 4 的答案：
过了一遍，没有漏掉的候选；`wo_projection`/`mla_query_projection` 都不是，
原因不是"形状不对"这么简单，是"专用于 MLA 的维度硬编码 + 仍是 8-bit 对称量化"
两条都不满足）。真正的候选只能是协调者已经指对的 `moe/_shared/kernels/w4a16/`。

### 4.1 `moe/_shared/kernels/w4a16/` 是不是 shape-generic（问题 1）

**分层看，答案是"底层 kernel 通用，唯一现成的高层 dense 入口不通用"**：

- **底层 kernel 发起类 `_compile_w4a16_gemm_launch`（`kernel.py:680` 起的类，
  编译/启动这个 CuTe DSL persistent-grid GEMM 的核心）是真正的 shape/layout
  通用件**：`num_experts`/`top_k` 只是普通 `int` 参数，唯一的下限检查是
  `"num_experts must be positive"`（`kernel.py:9378`），**没有任何 "E>=2"
  或"必须走 MoE 路由"的限制**。`weight_layout` 独立于 MoE 路由概念，是
  一个纯粹的权重编码格式选择，合法值至少有 `"packed"`（sparkinfer 原生/
  compressed-tensors 风格）、`"modelopt"`（modelopt 原生格式，`kernel.py:711,
  4618,9842` 三处独立校验分支）、`"trellis3_t256"`（EXL3）、`"nf3_2p1"`。
  `dense_route_fast_path: bool` 是**通用**验证的字段（`kernel.py:852-863`）：
  ```python
  self.dense_route_fast_path = bool(dense_route_fast_path)
  if self.dense_route_fast_path and (
      self.direct_topk_routes or self.single_token_route_fast_path
      or self.num_experts != 1 or self.top_k != 1 or self.mul_topk_weights
  ):
      raise ValueError("dense_route_fast_path requires E=1, top_k=1, ...")
  ```
  这条校验**不检查 `weight_layout`**——`dense_route_fast_path=True` 与
  `weight_layout="modelopt"` 组合在校验层面是合法的，没有代码路径会拒绝它。
- **权重准备函数用统一的 `[num_experts, out, in]` 张量约定，`num_experts=1`
  是这个约定里的合法值，不是特例**：`prepare.py:608` `num_experts =
  int(weight.shape[0])`——直接读 shape[0]，没有下限检查；`prepare.py:2057-2059`
  甚至对 trellis dense 路径显式要求 `trellis.shape[0] == 1`（"trellis3_t256
  dense payload requires E=1"）——**E=1 是这个代码库里被断言、被使用的正常
  形状，不是要绕过的边界情况**。
- **`prepare_w4a16_modelopt_nvfp4_weights`（`prepare.py:987-1020`）已经原生
  支持我们要的确切语义**：docstring 直接写"Prepare ModelOpt NVFP4 tensors
  into the W4A16 packed runtime layout. The per-block scales are the normal
  NVFP4 K/16 scale grid"——group_size=16、block scale、全局 scale，与 Qwen3.6
  MLP 的 `weight`/`weight_scale`/`weight_scale_2`（B0-2，§1.2）逐字段对应。
  姐妹函数 `prepare_w4a16_modelopt_native_weights`（`prepare.py:1023-1046`）
  的 docstring 更进一步：**"This is the memory-safe path for GLM serving
  that needs A4 prefill and A16 decode in the same process"**——这是
  sparkinfer 团队**已经为一个真实生产模型（GLM 系列）的 weight-only NVFP4
  场景写过这条路径**，不是理论上"应该能行"，是有真实动机、被验证过的代码。
- **但唯一现成的"dense（非 MoE 路由）"高层入口 `run_trellis256_dense`
  （`kernel.py:9669`）是硬编码给 EXL3 trellis 格式的**——它的实现
  `_run_trellis256_dense_current_device`（`kernel.py:9500-9666`）内部确实
  调用 `_compile_w4a16_gemm_launch(..., num_experts=1, top_k=1,
  dense_route_fast_path=True, weight_layout="trellis3_t256", ...)`，**这正是
  协调者设想的那个退化配置的真实、已经在跑的代码**——但函数体里夹杂了 EXL3
  专属的 Hadamard 输入/输出旋转（`hadamard_128(x_f16, rotated_f16,
  prepared_dense.suh, ...)`），这是 EXL3 格式本身需要的步骤，NVFP4 不需要
  （NVFP4 没有旋转，直接 block-scale 反量化），所以这个具体函数不能直接换
  `weight_layout` 复用——**限制在这一个 Python 包装函数里，不在底层 kernel**。

**结论（问题 1）**：不是"绑死 MoE 语义"，是"底层通用，唯一现成的高层 dense
包装函数选错了权重格式（写死 trellis，不是限制）"。

### 4.2 能不能用 `num_experts=1` 退化配置当稠密 GEMM 用（问题 2）

**能，而且不需要碰 sparkinfer 一行代码，走公开 API 就够**——4.1 节已确认
`num_experts=1` 在整个 `w4a16` 子树里是合法、被使用的形状，且
`sparkinfer.moe.fused_moe` 公开 API 本身就有 `quant_mode="w4a16"`
（`moe/fused_moe/_impl.py:937,1222,1298,1452` 等处 `_normalize_quant_mode`/
`caps.quant_mode == "w4a16"` 分支）：

```python
# 推断出的调用形状（未跑，[待验证 GPU]）——不是新写 sparkinfer 代码，
# 是用它公开 API 的正常参数
from sparkinfer.moe import fused_moe

wplan = fused_moe.plan_weights(
    quant_modes="w4a16", source_format="modelopt_nvfp4", ...,
)
experts = fused_moe.prepare_weights(plan=wplan, ...)   # num_experts=1 的权重
plan = fused_moe.plan(fused_moe.Caps(
    max_tokens=..., num_topk=1, quant_mode="w4a16", weight_plan=wplan, ...,
))
# topk_ids: 全 0（只有一个"专家"）；topk_weights: 全 1.0（float32）
binding = fused_moe.bind(plan, a=x, experts=experts,
                         topk_weights=torch.ones(m, 1),
                         topk_ids=torch.zeros(m, 1, dtype=torch.int32), ...)
out = fused_moe.run(binding=binding)
```

`TPMoEScratchCaps.__post_init__`（`_impl.py:760-762`）把 `num_topk` 的下限
钳到 `max(int(self.num_topk), 1)`——`num_topk=1` 是**下限**，不是要绕过的
特例。全仓库 grep 找 `num_experts` 相关的最小值断言，唯一命中是
`"num_experts must be positive"`（`kernel.py:9378`）和两处 `ep_moe`（专家并行）
模块的 `"global_num_experts must be positive"`——**没有任何地方写
`num_experts >= 2`**。

**这条路径走的是 sparkinfer 公开、文档化的 `moe.fused_moe` API，用真实存在
的 `quant_mode="w4a16"`，不是绕过公开接口去调用下划线开头的私有函数**——
比协调者问题 2 里设想的"退化配置 hack"更干净：不需要碰任何私有 API，
就是把 `num_experts` 这个参数设成 1，`topk`/路由张量设成"全部路由到 0 号
专家、权重恒 1"这个平凡情况。**唯一的代价**：会走通用 MoE 路由/permutation
的记账路径（`route_pack.py` 构造 `packed_route_indices`/`block_expert_ids` 等），
不会命中 `dense_route_fast_path` 那条零开销捷径（那条捷径目前只接在
`run_trellis256_dense` 这一个函数上，见 4.1）——**功能上应该完全正确，
性能上有一点可避免的路由开销，但排除这点开销不是 B1（正确性优先）阶段
需要解决的问题**。

**[待验证 GPU，本轮未跑一行 CUDA 代码，纯读码判断]**：上面的调用形状是
从 API 签名与内部校验推断出的，不是复制自任何现成的调用样例（未在
sparkinfer 仓库里找到 `num_experts=1` 的 fused_moe 单测或 benchmark 作为
现成参照）——**第一次真跑很可能要调几个参数名/张量 dtype 细节，但不应该
撞到"这个 API 根本不支持 E=1"这类结构性障碍**，4.1/4.2 的证据链已经排除了
这种可能性。

### 4.3 需不需要给 SparkInfer 团队提需求（问题 3）

**不需要阻塞性的需求**——4.1/4.2 已经证明现有公开 API 覆盖我们的确切数值
语义（modelopt NVFP4 weight-only、group_size=16），不需要 SparkInfer team
写任何新代码。

**可以提一条非阻塞、低优先级的性能优化建议**（供协调者决定是否要提，
本笔记不代为拍板要不要真的提这个 issue，只把材料准备好）：

> **给 SparkInfer 团队的材料（如果决定要提）**：
> `moe._shared.kernels.w4a16` 已经有一个真正零开销的 dense 单专家路径
> （`_compile_w4a16_gemm_launch(num_experts=1, top_k=1,
> dense_route_fast_path=True, ...)`），目前只通过 `gemm.trellis_linear`
> 暴露给 EXL3 trellis 权重格式（`run_trellis256_dense`）。我们的用例
> （Qwen3.6-27B 稠密 MLP，`hidden=5120`、`intermediate=17408`，modelopt
> NVFP4 weight-only、`group_size=16`、`F8_E4M3` block scale + `F32` 全局
> scale，bf16 激活，**无激活侧 scale 数学**）与 `run_trellis256_dense`
> 的语义几乎一致，只是权重编码换成 `weight_layout="modelopt"` 且不需要
> EXL3 的 Hadamard 输入/输出旋转步骤。建议：提供一个
> `run_modelopt_w4a16_dense`（或类似命名）作为 `gemm.*` 下的新 op，直接调用
> 已经存在的 `dense_route_fast_path` 机制配 `weight_layout="modelopt"`，
> 跳过通过 `moe.fused_moe` 走完整 MoE 路由/permutation 记账的可避免开销。
> **这是性能优化，不是功能缺口**——目前通过公开的 `moe.fused_moe(quant_mode=
> "w4a16", num_experts=1)` 已经可以正确跑通（[待验证 GPU]），只是多付一点
> 路由开销。

---

## 5. （2026-08-02 第二轮）GDN 递归状态 dtype：修正证据链，结论不变但更精确

**协调者要求复核**："transformers 实际存 BF16"这条结论是否在**纯文本推理
路径**上成立，不是某个我们不走的分支——查完发现：**结论（落盘 BF16）成立，
但第一轮（§3.3）的证据链找错了函数**，机制比第一轮描述的更细。

### 5.1 第一轮的缺口：本机装的是真的 `fla`，不会走 torch 兜底函数

`Qwen3_5GatedDeltaNet.__init__`（`modeling_qwen3_5.py:409-410`）：

```python
self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule
self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule
```

`chunk_gated_delta_rule`/`fused_recurrent_gated_delta_rule` 是从真实的
`fla` 包 import 的（`modeling_qwen3_5.py:60-61`：`from fla.ops.gated_delta_rule
import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule`）。本机
`fla` 包在 `/home/bot/project/flash-linear-attention`，`~/.venvs/vllm` 里
可以真实 import 成功（B0 第一轮已确认，本轮复核：`~/.venvs/vllm/bin/python
-c "import fla.ops.gated_delta_rule as g; print(g.chunk_gated_delta_rule,
g.fused_recurrent_gated_delta_rule)"` 两个都不是 `None`）——所以上面两行
`or` 的**左边**为真，`self.chunk_gated_delta_rule`/`self.recurrent_gated_
delta_rule` 绑定的是**真实 FLA kernel**，不是第一轮读的 `torch_chunk_
gated_delta_rule`/`torch_recurrent_gated_delta_rule` 那两个纯 PyTorch
兜底函数（那两个函数只在 `causal_conv1d_fn`/`causal_conv1d_update` 缺失时
才被用到，且缺失的只是 conv1d 那一步，跟 gated-delta-rule 本身的函数选择
是独立的两个 `or` 判断）。**这是第一轮的错误**——引用了正确的落盘结论，
但证据取自不会被执行的分支。

### 5.2 真实 FLA kernel 的 recurrent state 是 FP32，不是 BF16

`flash-linear-attention/fla/ops/gated_delta_rule/fused_recurrent.py:209-212`
（decode 单步路径）：

```python
if state_v_first:
    final_state = q.new_empty(N, HV, V, K, dtype=torch.float32)
else:
    final_state = q.new_empty(N, HV, K, V, dtype=torch.float32)
```

`flash-linear-attention/fla/ops/common/chunk_delta_h.py:637,640`
（chunk/prefill 路径）：

```python
final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32) if output_final_state else None
final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None
```

**两条路径（decode 的 `fused_recurrent_gated_delta_rule` 与 prefill 的
`chunk_gated_delta_rule`）都无条件把 `final_state` 分配成 `torch.float32`，
与 q/k/v 的输入 dtype（bf16）无关。** 这与第一轮读到的 torch 兜底函数
（`dtype=value.dtype`，即 bf16）刚好相反。

### 5.3 但落盘时被 HF 通用 `Cache` 类降精度round-trip回 BF16——净结论不变

`transformers/cache_utils.py:772-784`（`LinearAttentionLayer.
lazy_initialization`，与第一轮引用的行号一致）：

```python
if conv_states is not None:
    self.dtype, self.device = conv_states.dtype, conv_states.device   # 先落 conv 的 dtype
    ...
if recurrent_states is not None:
    self.recurrent_states = torch.zeros_like(recurrent_states, dtype=self.dtype, ...)  # 用 self.dtype，不是 recurrent_states.dtype！
```

**关键**：`recurrent_states` 分支用的是 `dtype=self.dtype`，不是
`dtype=recurrent_states.dtype`！`self.dtype` 在**同一次 forward 调用内**
已经被 `conv_states` 分支提前锁定成 BF16（因为 `Qwen3_5GatedDeltaNet.forward`
里 conv1d 的更新总是发生在 delta-rule 之前，`cache_params.update_conv_state`
/`causal_conv1d_update` 先于 `cache_params.update_recurrent_state`，见
`modeling_qwen3_5.py:455-473` vs `:534-535`）。所以：

1. FLA 真 kernel 算出 `last_recurrent_state`（FP32）。
2. `cache_params.update_recurrent_state(last_recurrent_state, ...)` 调用
   `update_recurrent_state`（`cache_utils.py:823-834`）。
3. 若是第一次调用（prefill），`lazy_initialization` 用**已经被 conv 锁定的
   `self.dtype`=BF16**分配 `self.recurrent_states` 缓冲区，然后
   `self.recurrent_states.copy_(recurrent_states)`——**FP32 值被隐式转换
   降精度存进 BF16 buffer**。
4. 下一步 decode 时，从这个 BF16 buffer 读出的 `initial_state` 传回 FLA
   kernel，FLA kernel 内部再次以 FP32 精度计算这一步更新，产出新的 FP32
   `final_state`，再被同一个 BF16 buffer 的 `.copy_()` 降精度存回。

**净效果：每一步的数值计算发生在 FP32，但跨步骤"携带"的状态在 BF16 精度
上取整**——不是"全程 BF16 计算"（第一轮的表述过粗），也不是"全程 FP32
持久化"（如果真要对齐这个参考实现的字面行为）。**这条机制对 B1 的
"逐 token bit-exact 对齐" 门禁是实质性的**：如果我们自己的实现选择"全程
FP32 状态、从不降精度"，长上下文/多步之后会因为缺失这个逐步舍入而与参考
实现产生可测的漂移——**要跟这份参考"位对齐"，必须复刻"FP32 计算、BF16
存储之间的舍入"这个具体动作，不能只是"选一个精度、从头到尾都用它"**。
这是本轮新增、比第一轮更具体的技术要求，留给 B1 实现者。

### 5.4 结论修正（覆盖 §3.3 的原表述，容量数字不变）

- `mamba_ssm_dtype: "float32"` 这个 config 字段——**依然是摆设**，不是因为
  delta-rule 计算不用 fp32（它确实用 fp32），而是因为**跨步持久化**这个层面
  被 HF 通用 Cache 类的降精度规则覆盖了，config 字段本身在这条链路上从未
  被读取（第一轮的这条子结论未变）。
- 单槽容量数字**不变**：落盘确实是 BF16，3.3 节"~74.8 MiB/槽"那一行仍是
  匹配当前参考实现实际行为的数字，"~149.6 MiB/槽"那一行对应的是"如果
  跨步也存 FP32"的假设场景（生产实现可能会选，只是不match这份参考实现的
  字面行为）——3.3 节两行数字本身都不需要改，改的是"为什么落盘是 BF16"
  这条因果链，以及新增了"FP32 计算+BF16 舍入"这条 B1 精度对齐的具体要求。
- **[仍是 [待验证]]**：本节全部是静态读码（HF `transformers` 源码 +
  真实 `flash-linear-attention` 源码），未跑一次 GPU forward 验证实际张量
  dtype。建议 B1 起步时用一次 CPU-only 或单 GPU 的 dummy forward，
  在 `cache_params.layers[i].recurrent_states.dtype` 上直接断言一次，
  把"读码推断"升级成"实测确认"。

---

## 6. （2026-08-02 第二轮）FP8 KV 声明但无 scale 张量时，vLLM/SGLang 怎么办

**查法**：本机 vLLM 源码在 `/home/bot/vllm`（`import vllm` 直接命中），
SGLang 源码在 `/home/bot/project/sglang`。零 GPU，只读源码。

### 6.1 vLLM：默认 1.0 + 告警，且正在把"运行时校准"这条路废弃

`vllm/model_executor/layers/quantization/kv_cache.py`
（`BaseKVCacheMethod.process_weights_after_loading`，21-197 行）：

- `create_weights` 把 `k_scale`/`v_scale`（以及 `q_scale`/`prob_scale`）
  初始化成 **`-1.0`**（"无效哨兵值"，`kv_cache.py:65-69`）。checkpoint 里若有
  真实值，权重加载时会覆盖这个哨兵。
- `process_weights_after_loading`（:100-152）三分支：
  1. `k_scale>0 and v_scale>0`——checkpoint 提供了真实值，直接用。
  2. **`k_scale<0 and v_scale<0`（两个都还是哨兵，即"checkpoint 没提供"）
     ——`k_scale = v_scale = 1.0`**（:111-115，**这正是 Qwen3.6-27B-NVFP4
     这份 checkpoint 会落进的分支**：声明 `kv_cache_quant_algo=FP8`，但
     零 `k_scale`/`v_scale` 张量，见 B0 第一轮 §1.3）。
  3. 只有一个标量 `kv_scale`（老格式）——复制成 k/v 各一份。
  - 分支 2 触发时还会打一条运行时告警（:147-152）：*"Using KV cache
    scaling factor 1.0 for fp8_e4m3. If this is unintended, verify that
    k/v_scale scaling factors are properly set in the checkpoint."*
- **`--calculate-kv-scales`**（`CacheConfig.calculate_kv_scales`，
  `vllm/config/cache.py:111`）是运行时动态校准 k/v scale 的旧机制
  （首次 forward 时用真实 Q/K/V 的 amax 算一个 scale，`attention.py:
  690-706` 的 `maybe_calc_kv_scales`/`calc_kv_scales`）。**但这条路径正在
  被弃用**——`vllm/config/cache.py:261-272` 的字段校验器在这个 flag 为
  True 时打出明确的弃用告警：*"The `--calculate-kv-scales` option is
  deprecated and will be removed in v0.19. The scales will be loaded from
  the model checkpoint if available, otherwise they default to 1.0."*
  ——**这句话就是 vLLM 项目对"declared FP8 KV 但没 scale 张量"这个问题
  官方、当前、明确的处理方针**：有就读，没有就 1.0，不再费力自动校准。

### 6.2 SGLang：与 vLLM 逐字节相同的策略（文件头注明是从 vLLM 搬的）

`sglang/python/sglang/srt/layers/quantization/kv_cache.py` 文件头
（第 2-3 行）：

```python
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kv_cache.py
```

逻辑逐行对应：`k_scale`/`v_scale` 初始化成 `torch.tensor(-1.0, ...)`
（:39-44），`process_weights_after_loading` 同样三分支
（`k_scale<0 and v_scale<0` → **`k_scale=v_scale=1.0`**，:59-63）。**这不是
两个框架独立收敛到同一答案——是 SGLang 直接照抄 vLLM 的实现**，所以只算
一条证据链，不是两条独立确认，但仍说明这是当前主流推理框架实际在跑、
没有人在用更复杂方案的成熟做法，不是权宜之计。

### 6.3 结论：建议采纳"默认 1.0 + 告警"，不建议自己搭校准流水线

vLLM 自己都在把运行时校准废弃掉，没有理由我们反而去新建一套。**建议**（
不代为最终拍板，见 spec §7 的待拍板条目）：Qwen3.6 的 FP8 KV 如果最终启用，
`k_scale`/`v_scale` 默认 `1.0`，日志打一条与 vLLM 同款的告警提示"若非预期
请检查 checkpoint"。**唯一需要盯的风险**：`1.0` 是否对 Qwen3.6 的真实 K/V
数值范围合适（RMSNorm 后的量级），如果逐 token 对齐/质量评测掉点明显，
再考虑要不要自己跑一次校准——**但不要在验证到"1.0 不够"之前就先去写校准
代码，遵守"先证据、后工程"的仓库纪律**。

---

## 7. 给 `docs/qwen36-rebuild-spec.md` 的具体更正（已同步过去，见该文件 diff）

**第一轮（B0-2/6/7）**：

1. §1.9"格式警告"："本轮未定位到 modelopt 参考实现，不猜字段名"这句已过时，
   替换为指向本笔记 §1 的具体命名表。
2. §3.4"加载器 adapter"：补充真实字段名与"按量化算法分三支"的结论（本笔记 §1.6）。
3. §4"已确认的事实"："modelopt NVFP4 + fp8 KV" 改成准确描述（混合精度 +
   KV scale 缺失，本笔记 §1.1/§1.3）。
4. §6 待验证清单：勾掉 B0-2/B0-6/B0-7 对应条目，新增 KV cache scale 缺失、
   GDN 状态 dtype、modelopt `input_scale` 语义、`q_proj` 2× 宽度四项。

**第二轮（本轮，协调者跟进）**：

5. §3.4/§5.2"NVFP4 GEMM 到底选自研 `.cu` 还是 sparkinfer"：**候选集进一步
   收窄，不再是"两个 FP4×FP4 kernel 选一个"或"混合精度 kernel 待找"**——
   `sparkinfer.moe.fused_moe(quant_mode="w4a16", source_format=
   "modelopt_nvfp4")` 配 `num_experts=1` 已经是现成、公开、覆盖我们确切数值
   语义的路径（本笔记 §4），**[待验证 GPU]** 但不再是"完全没有候选"的状态。
6. §6/§7 待验证/待拍板清单：GDN 状态 dtype 从"两个候选，不知道哪个对"收窄为
   "落盘 BF16 已确认（机制是 FP32 计算+BF16 舍入，不是简单的 bf16 全程），
   B1 若要 bit-exact 需要复刻这个舍入动作"（本笔记 §5）。
7. §7 待拍板"FP8 KV 的 scale 从哪来"：新增 vLLM/SGLang 的现成答案（默认
   1.0+告警，本笔记 §6）作为参考先例，不改变"这条仍待 C-2 结果后再最终拍板"
   的状态，但把"要不要自己搭校准流水线"这个子问题基本排除（先用 1.0）。

---

## 8. 复现清单（给下一个查这个 checkpoint 的人）

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
- **第二轮追加的源码位置**（均只读，未改动任何文件）：
  - sparkinfer：`/home/bot/project/sparkinfer`（按任务约定只读，不 import
    进生产代码）——关键文件 `sparkinfer/gemm/*/api.py`（9 个 op 的入口）、
    `sparkinfer/_lib/dense_gemm.py:6660`（对称 `dense_gemm` 原语签名）、
    `sparkinfer/moe/_shared/kernels/w4a16/kernel.py`（11430 行，
    `_compile_w4a16_gemm_launch` 类在 :680、`run_trellis256_dense` 在
    :9669、`run_w4a16_moe` 在 :9740）、`.../w4a16/prepare.py`（2271 行，
    `prepare_w4a16_modelopt_nvfp4_weights` 在 :987）、
    `sparkinfer/moe/fused_moe/_impl.py`（`TPMoEScratchCaps` 在 :741）。
  - `flash-linear-attention`（`fla`）真实源码：`/home/bot/project/
    flash-linear-attention`（`~/.venvs/vllm` 里可 `import fla`，指向这份
    源码，不是 pip 装的隔离副本）——`fla/ops/gated_delta_rule/
    fused_recurrent.py:209-212`、`fla/ops/common/chunk_delta_h.py:637,640`。
  - vLLM：`/home/bot/vllm`（`~/.venvs/vllm` 里 `import vllm` 命中这份源码；
    与本仓库"不依赖 vLLM"的方向无关，纯粹当参考实现读，不 import 进本仓库
    任何生产代码）——`vllm/model_executor/layers/quantization/kv_cache.py`
    全文件、`vllm/config/cache.py:261-272`、`vllm/model_executor/layers/
    attention/attention.py:690-706`。
  - SGLang：`/home/bot/project/sglang`——`python/sglang/srt/layers/
    quantization/kv_cache.py` 全文件。
