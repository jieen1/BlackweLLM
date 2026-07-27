# bfdiag.shapes:kernel 隔离测试的 shape 自动推导

## 背景 / 动机(原话)

> kernel 隔离测试的 shape 自动从真实 config 推导 —— 文档里没有覆盖,`bf divergence`
> 做的是逐层对拍(和 oracle 比),不是"page_size A/B、shape 必须从真实模型形状自动生成"
> 这种。今天 kernel 隔离测试就是因为手填 shape 错了才走了弯路,这个还是真缺口。

交付:`bfdiag/shapes/{model,attention,moe,gemm,harness,cli}.py` + 6 个测试文件
(`tests/test_bfdiag_shapes_*.py`,64 个测试,CPU 全绿)+ 本笔记。

```python
from bfdiag.shapes import model_shapes
S = model_shapes(block_size=128)
q, k, v, pt, seqlens = S.decode_attention(group="sliding", kv_len=65536).empty_tensors()
```

```bash
bf shapes                                          # 两个 block_size(64/128)都打印
bf shapes --block-size 128
bf shapes --block-size 64 --block-size 128 --diff  # 只列出变化的形状
bf shapes --json
```

## 开发环境说明(worktree 落后 main,已同步)

本 worktree 分配时的基线是 `ceb7ec8`,落后 `main`(`0504a96`)24 个提交 ——
`bfdiag/` 整个包、`docs/diagnostics-guide.md`、`runtime/backends/laguna.py` 的
block_size=128 迁移都不存在。协调者确认后执行了 `git merge --ff-only main`
(纯快进,HEAD 是 main 的祖先,无冲突、无 merge commit)。之后只新增了
`bfdiag/shapes/*` 和本笔记 + 测试文件,没有碰 `runtime/`、`bfdiag/cli.py`、
`pyproject.toml`、`bfprobe/`。

## 规格 vs 代码的分歧(以代码为准,记录在此)

1. **头数**:任务原话给的是"跑班 num_attention_heads=48"这种全局值。真实
   `config.json` 里 `num_attention_heads_per_layer` 是**逐层**列表 ——
   full_attention 层 48 头、sliding_attention 层 72 头,KV 头数两者都是 8
   (GQA group size 6 / 9)。这和 `notes/2026-07-27-laguna-real-shapes-correction-and-page-size-migration-plan.md`
   记录的纠正一致(旧文档曾错误写成"24 Q头/8 KV头")。`bfdiag/shapes/model.py`
   读的是逐层列表,不是任务原话给的单一数字。
2. **block_size 校验范围**:合并后的 `runtime/backends/laguna.py:142` 是
   `if block_size not in (64, 128): raise`(migration 已完成),不再是旧版的
   `!= 64`。`bfdiag.shapes` 的 `--block-size` 就是围绕这两个值设计的默认值,
   但函数签名本身不限制取值(任何正整数都能推导,只是"当前生产只接受 64/128"
   这条约束体现在 CLI 默认值上,不体现在库函数里)。
3. **`group="swa"` vs `"sliding"`**:任务示例用的是 `group="swa"`,真实代码
   (`runtime/backends/laguna.py` 的 `_layer_groups`)用 `layer_types` 里的字符串
   `"sliding_attention"`。`bfdiag.shapes` 以 `"sliding"` 为主键名(直接对应
   config 的 `layer_types` 取值,减少一层翻译),`"swa"` 保留为别名兼容任务里
   给的调用写法。两者行为完全相同。

## 推导规则逐条(附代码出处)

### 1. 层分组(`bfdiag/shapes/model.py::load_laguna_config`)

依据 `layer_types`(逐层字符串)+ `sliding_window`(全局窗口)+
`num_attention_heads_per_layer`(逐层头数,缺失时退化为全局
`num_attention_heads` 并标记 `heads_per_layer_source="config_uniform"`),
复刻 `runtime/backends/laguna.py:192-220`(`_layer_groups` 构造,按
`(window_left, num_qo_heads, num_kv_heads)` 分组)的分组逻辑,但不 import 它 ——
独立从同样的 config 字段重新推一遍。

真实结果(在这台机器上验证,`tests/test_bfdiag_shapes_model.py::
test_real_config_layer_grouping`):12 层 full_attention(位置 0,4,8,…,44)+
36 层 sliding_attention(window=512)。

### 2. SWA 环大小(静态容量):`ring_blocks_for_window`

出处:`runtime/backends/laguna.py:47-50`
```python
SWA_QO_MAX = 16
def _ring_blocks_for_window(window: int, block_size: int, qo_max: int = SWA_QO_MAX) -> int:
    return -(-(window - 1 + qo_max) // block_size) + 1  # cdiv + 1
```
`bfdiag/shapes/attention.py::ring_blocks_for_window` 是**独立重新实现**
(不 import 上面这个函数),测试里(`test_ring_blocks_matches_real_formula`)
又用 `math.ceil` **第三次**独立写一遍公式做对拍 —— 三份互相独立的代码,
不会因为改一份忘了改另一份而悄悄分叉。

`runtime/backends/laguna_dflash.py:294-295` 里 draft 模型的环用同一个函数、
同样的 `qo_max=NUM_QUERY_PER_REQ`:
```python
draft_blocks_per_slot = _ring_blocks_for_window(DRAFT_WINDOW, self.block_size, NUM_QUERY_PER_REQ)
```
`DRAFT_WINDOW == sliding_window == 512`、`NUM_QUERY_PER_REQ == SWA_QO_MAX == 16`
在这个模型上刚好相等,所以主模型 SWA 环容量和 draft 环容量数值相同(见下表),
但这是巧合,不是保证 —— `bfdiag.shapes` 分别推导两者,不假设相等。

### 3. SWA 环对齐算法(每一步实际用到的 aligned_len / n_ring)

出处(decode,M=1,不设 cap):`runtime/backends/laguna_cuda_graph.py:200-235`
(`LagunaCudaGraphDecode._fill_buffers_b1`)
```python
new_kv = kv_len + 1
window_start = max(0, kv_len - window + 1)
aligned_start = (window_start // ps) * ps
aligned_len = new_kv - aligned_start
n_ring = (aligned_len + ps - 1) // ps          # 不封顶
```
出处(verify,M=16,封顶):`runtime/backends/laguna_cuda_graph.py:611-648`
(`LagunaCudaGraphVerify._fill_buffers`)
```python
new_kv_len = kv_len + nt
window_start = max(0, kv_len - self._swa_window + 1)
aligned_start = (window_start // bs) * bs
aligned_len = new_kv_len - aligned_start
n_ring = min(-(-aligned_len // bs), self._ring_blocks_per_slot)   # 封顶
```
`bfdiag/shapes/attention.py::swa_alignment` 是这两段的**统一独立重推**:
`qo_len` 参数化(decode 传 1、verify 传 16),`ring_blocks_per_slot=None`
复刻 decode 的不封顶,传具体值复刻 verify 的 `min(...)`。

**⭐ 这是本次任务里价值最高的一条**:`aligned_start` 向下取整到 block_size
的整数倍,取整丢掉多少"零头"取决于 `window_start` 落在 block 网格的什么位置 ——
所以 `aligned_len` / `n_ring` **不是**简单地随 block_size 减半。见下面的对照表。

### 4. Full attention 分页数:`full_attention_pages`

出处:`LagunaCudaGraphDecode._fill_buffers_b1` 的 `n_blocks = (new_kv + ps - 1) // ps`
和 `LagunaCudaGraphVerify._fill_buffers` 的 `n_blocks_full`,统一成
`cdiv(kv_len + qo_len, block_size)`。

### 5. Prefill SWA 暂存区(scratch buffer)

出处:`runtime/backends/laguna.py:300-326`
```python
_scratch_tokens = swa_window + prefill_chunk_tokens   # QSR_PREFILL_CHUNK 默认 8192
swa_scratch_blocks = min(blocks_per_slot, cdiv(_scratch_tokens, block_size))
shape = (2, swa_scratch_blocks, block_size, num_kv_heads, head_dim)
```
`bfdiag/shapes/attention.py::prefill_swa_scratch` 原样复刻。full attention 的
prefill 走的是**分页** attention(不是 scratch),用
`full_attention_call(kv_len=kv_len_before, qo_len=chunk_tokens)`;
`ModelShapes.prefill_attention(group="sliding", ...)` 会主动报错,提示改用
`prefill_swa_scratch(...)` —— 这是真实代码里两条路径本质不同(分页 vs 扁平
scratch)的直接体现,不是偷懒统一成一个假的"对称" API。

### 6. NVFP4 打包后的 MoE 专家权重形状

出处:`runtime/backends/laguna_sparkinfer_moe.py`(`load_moe_layer_weights`/
`prepare_sparkinfer_layer`)—— 但具体数字是**直接读这台机器上真实 checkpoint 的
safetensors header**(只读 shape/dtype 元数据,不实体化张量数据,和读
config.json 一样是纯文件读取,不违反"不许用 GPU"):

```
model.layers.{1..47}.mlp.experts.{0..255}.gate_proj.weight_packed  [1024, 1536]  uint8
model.layers.{1..47}.mlp.experts.{0..255}.gate_proj.weight_scale   [1024,  192]  f8_e4m3
model.layers.{1..47}.mlp.experts.{0..255}.down_proj.weight_packed  [3072,  512]  uint8
model.layers.{1..47}.mlp.experts.{0..255}.down_proj.weight_scale   [3072,   64]  f8_e4m3
```
打包规则:`weight_packed = [out, in//2]`(NVFP4 每字节 2 个值),
`weight_scale = [out, in//group_size]`(`group_size=16`,从
`quantization_config.config_groups.*.weights.group_size` 读出,不硬编码)。
`sparkinfer` 融合后的 `w13_fp4 = [E, 2*moe_intermediate_size, hidden//2]`
(gate+up 沿 out 维拼接,`torch.cat([up_w, gate_w], dim=1)`)。

### 7. 稠密 GEMM(qkv/o/g proj、MLP、router、lm_head)

同样直接读这台机器上真实权重张量 shape 校验过(不是凭架构印象手填):
主模型 full 层 `q_proj=[6144,3072]`(48*128)、sliding 层
`q_proj=[9216,3072]`(72*128)、两者 `k_proj/v_proj=[1024,3072]`(8*128)、
`g_proj`(per-head gate,config `"gating":"per-head"`)分别是 `[48,3072]`/
`[72,3072]`、layer 0 稠密 MLP `gate/up=[12288,3072]` `down=[3072,12288]`、
router `gate=[256,3072]`、`shared_expert` `gate/up=[1024,3072]`
`down=[3072,1024]`、`lm_head=[100352,3072]`。

**draft 模型是一个真实的架构差异,不是"复用主模型公式就行"**:draft 用
**融合的** `qkv_proj=[11264,3072]`(72*128+2*8*128,不是分开的 q/k/v_proj),
`fc.weight=[3072,18432]`(EAGLE 式融合 6 个 aux hidden state,
`18432 = 6 * hidden_size`,对应 `eagle_aux_hidden_state_layer_ids` 长度),
且**没有自己的 lm_head/embed_tokens**(`draft_vocab_size==vocab_size`,复用
主模型的 tied lm_head)。这些都是从 draft 的
`~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-DFlash-NVFP4` 真实
`model.safetensors` header 里读出来的,不是猜的。

## block_size=64 vs 128:完整形状对照表

用 `bf shapes --block-size 64 --block-size 128 --diff --kv-len 65536`
(65536 = `benchmarks/ab_dflash_block_size_64_vs_128.py` 的默认 CTX)在这台机器
上跑出的真实结果:

| key | bs=64 | bs=128 | 变了? |
|---|---|---|---|
| `decode/full.k_cache` (=KV cache 张量) | `(1025, 64, 8, 128)` | `(513, 128, 8, 128)` | ✅ |
| `decode/full.max_pages` | 1025 | 513 | ✅(简单减半+1) |
| `decode/sliding.n_ring` | 9 | 5 | ✅ |
| `decode/sliding.aligned_len` | 513 | 513 | **不变**(kv_len=65536 时巧合相同,见下) |
| `verify/sliding.n_ring` | 9 | 5 | ✅ |
| `ring_capacity/sliding`(静态环容量) | **10** | **6** | ✅(不是简单减半,10→5 才是"减半") |
| `draft_ring_capacity` | 10 | 6 | ✅(和主模型 SWA 环容量数值相同,见规则2) |
| `prefill_swa_scratch.scratch_blocks` | 136 | 68 | ✅(这个是精确减半,136=2*68) |
| `kv_cache/sliding` | `(2, 10, 64, 8, 128)` | `(2, 6, 128, 8, 128)` | ✅ |
| `gemm/*`(全部 18 个稠密 GEMM) | — | — | **完全不变** |
| `moe/*`(专家权重/router/topk) | — | — | **完全不变** |
| `decode/full.q` / `verify/*.q` | — | — | **完全不变** |

**⭐ 关键发现 1:`ring_capacity/sliding` 从 10 到 6,不是从 10 到 5。**
`ring_blocks_for_window(512, 64, 16) = cdiv(527,64)+1 = 9+1 = 10`;
`ring_blocks_for_window(512, 128, 16) = cdiv(527,128)+1 = 5+1 = 6`。
`cdiv` 的向上取整在 block_size 变大时不是线性缩放的,这类"看起来该减半但没有
精确减半"的量正是手填 shape 最容易抄错的地方。

**⭐ 关键发现 2:`aligned_len` 在默认 kv_len=65536 处"恰好"两个 block_size 下
相同(513),但这是这个特定 kv_len 的巧合,不是普遍规律。** 换一个 kv_len
(比如 65600,`bf shapes --diff 64 128 --kv-len 65600`)立刻能看到真实分叉:

| key | bs=64 (kv_len=65600) | bs=128 (kv_len=65600) |
|---|---|---|
| `decode/sliding.aligned_start` | 65088 | 65024 |
| `decode/sliding.aligned_len` | 513 | 577 |
| `verify/sliding.aligned_len` | 528 | 592 |

513 vs 577 既不是 2 倍关系也不是 0.5 倍关系(测试
`test_swa_alignment_diverges_nontrivially_across_block_size` 显式断言了这
一点)。**这正是当前 bs=64→128 接受率排查(见
`notes/2026-07-27-block-size-128-migration-and-tie-break-noise.md`)的一个
候选机制**:`aligned_len` 不同 → attention kernel 内部 V 向量加权归约覆盖的
KV 范围/分页粒度不同 → 浮点求和顺序不同 → 该笔记记录的"浮点临界 argmax 翻转"
在数学上完全说得通(该笔记的结论是"不是 bug,是浮点噪声",这份对照表提供了
"噪声从哪来"的一个具体、可算的候选来源,但**没有**声称已经证实因果关系 ——
见下面 GPU 待办清单)。

完整机器可读版本:`bf shapes --diff 64 128 --kv-len 65536 --json` /
`--kv-len 65600 --json`。非 diff 模式 `bf shapes --block-size <N>` 打印每个
block_size 下的全量形状(decode/verify/draft/prefill/gemm/moe 共 ~103 项)。

## 如何在隔离测试里用

```python
from bfdiag.shapes import model_shapes

# 1. 声明意图,不抄数字
S = model_shapes(block_size=128)

# 2. 主模型 SWA 层 decode,在 kv_len=65536 处的注意力核形状
call = S.decode_attention(group="sliding", kv_len=65536)  # 或 group="swa"(别名)
q, k, v, page_table, cache_seqlens = call.empty_tensors()
# q: [1, 72, 128] bf16      k/v: [max_pages, 128, 8, 128] uint8(fp8 视图)
# page_table: [1, max_pages] int32   cache_seqlens: [1] int32
# 想看具体数字/诊断字段:
print(call.max_pages, call.swa.aligned_len, call.swa.n_ring)

# 3. DFlash verify(16 token),full attention 组
verify_call = S.verify_attention(group="full", kv_len=65536)

# 4. draft 模型自己的形状
draft_call = S.draft_verify_attention(kv_len=512)

# 5. 稠密 GEMM / MoE(与 block_size 无关,但同一套真实 config 推导)
for g in S.dense_gemms(num_tokens=16):
    print(g.name, g.m, g.n, g.k)          # 直接喂给你的 GEMM 微基准
expert_shapes = S.moe_expert_shapes()      # NVFP4 打包后的专家权重形状

# 6. 只依赖 torch,CPU-only,不需要 vllm/GPU
```

排查 page_size A/B 时:
```bash
bf shapes --block-size 64 --block-size 128 --diff --kv-len <你的真实 CTX>
```

`bfdiag.shapes` 全程只依赖 `torch` + 标准库(`bfdiag/shapes/harness.py` 里
`torch.empty`/`randn`/`zeros` 默认 CPU,`device="cuda"` 会被
`RuntimeError` 主动拒绝,除非显式设置 `BF_SHAPES_ALLOW_CUDA=1`)。

## 需要 GPU 才能验证的待办清单

以下是**这份代码本身没有、也不可能在 CPU 上验证**的部分,留给有 GPU 访问权限
的后续工作(现在这台机器的 GPU 正被用户占用排查 bs=64 vs 128 问题,本任务
全程没有碰 GPU):

1. **把 `bf shapes` 推出来的形状真的喂给 SparkInfer 的
   `create_paged_plan`/paged attention kernel**,确认 kernel 接受这些形状
   (尤其 fp8 KV cache 的 `uint8` 分配 + `.view(torch.float8_e4m3fn)` 这套
   约定)而不报错、数值上和真实 decode/verify 路径一致(cos 相似度对拍)。
2. **draft 模型的"decode"语义存疑**:本模块假设 draft 模型也有一个 qo_len=1
   的逐 token 自回归步骤(`draft_decode_attention`),但 DFlash draft config
   里出现的 `mask_token_id`(=12)暗示可能是掩码/并行生成而非纯自回归 ——
   需要读一遍真实 GPU 运行时 draft 生成循环的代码路径(或者直接跑一次
   trace)确认 `draft_decode_attention` 描述的场景是否真实存在,还是 draft
   模型每轮永远走 16-token verify 形状的那条路径。
3. **验证"aligned_len 分叉是接受率差异候选来源"这个假设的因果性**:这份
   笔记只证明了 aligned_len 在数学上确实随 block_size 非平凡变化,**没有**
   证明这就是 `2026-07-27-block-size-128-migration-and-tie-break-noise.md`
   里观测到的浮点临界翻转的直接原因。需要在 GPU 上对同一个 kv_len 分别用
   bs=64/128 跑一次真实 attention kernel,读出中间 V 加权和,确认求和顺序
   / 累加路径确实因为 aligned_len 不同而不同。
4. `ring_capacity/sliding`(静态环容量,10 vs 6)**理论上**应该始终 ≥ 运行时
   任意 kv_len 算出的 `n_ring`(见规则 3 的"不封顶"注释里的自洽性论证)——
   没有在真实长上下文(64K/256K)运行里验证过这个不等式在所有轮次里都成立。
5. `prefill_swa_scratch` 的 `blocks_per_slot_cap` 参数在真实部署里传的具体
   数值(`blocks_per_slot`)会不会让 `min(...)` 真的生效(即 `scratch_blocks`
   被外部capacity 限制而不是被 `window+chunk_tokens` 限制)—— 本模块两条分支
   都实现了,但没有验证真实部署实际落在哪一条。
6. `bfdiag/shapes/model.py` 里 `kv_cache_dtype` 从
   `quantization_config.kv_cache_scheme.num_bits==8` 推出 `"fp8_e4m3"`,但
   真实 KV cache 的两种可能物理格式(`torch.uint8` 分配 + 事后 view,还是
   直接 `torch.float8_e4m3fn` 分配)有微妙的 stride/对齐差异,值得在 GPU 上
   确认 `bfdiag.shapes.harness` 默认给出的 `kv_dtype=torch.uint8` 张量能不能
   直接喂给真实 kernel 调用(不需要额外 reinterpret 步骤)。
7. 目前只验证了 `poolside/Laguna-S-2.1-NVFP4` + 对应 DFlash draft checkpoint
   这一套真实 config;换模型/换量化格式(比如非 NVFP4)后 `moe.py` 的打包
   公式(2 值/字节、`group_size` 分组)需要重新用真实 safetensors header
   核实,不能假设所有模型都是这个打包约定。
