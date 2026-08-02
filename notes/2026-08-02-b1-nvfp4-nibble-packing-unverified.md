# NVFP4 nibble 打包顺序：B1 结论可信度的天花板

> 编制日期：2026-08-02 · worktree `work/b1-qwen36-20260802`
> 环境：`~/.venvs/vllm/bin/python`（`torch==2.13.0a0+gitcf30153`）· GPU：`NVIDIA
> RTX PRO 6000 Blackwell Max-Q`
>
> **单独成文的原因**：这条在 B1 的 GPU 验证过程中反复出现（`runtime/loading/
> modelopt.py` 的模块 docstring、`scripts/b1_verify_nvfp4_dequant.py` 的运行
> 结果、`scripts/b1_verify_greedy_alignment.py` 的开场白都各提了一遍），但
> 分散在各处容易被忽略。这条不是"B1 特有的小瑕疵"——任何后续要碰 Qwen3.6 的
> NVFP4 权重（B3 GEMM kernel 选型、量化精度评测、未来别的 NVFP4 模型接入）都
> 会撞上同一个问题，所以按协调者的要求单独写一份，供后来者直接引用。

## 一句话结论

**`runtime/loading/modelopt.py::unpack_nvfp4_to_fp32` 的 E2M1 数值表是数学上
确定的，不是猜的；但它的"哪个 nibble 对应哪个元素"打包顺序，在本机环境下**
**没有任何独立 oracle 可以验证**——本轮已经系统性排查过全部候选验证渠道，
逐个说明为什么用不了，最后退回到"端到端生成结果连贯且事实正确"这个间接证据。
这是当前 B1 结论里**唯一**没有直接数值交叉验证支撑的一环。

## 问题的确切形状

modelopt 的 NVFP4 weight-only 量化把两个 4-bit E2M1 编码打包进一个 uint8
字节：

```
byte = (high_nibble << 4) | low_nibble
```

`runtime/loading/modelopt.py::unpack_nvfp4_to_fp32` 假设 **low nibble = 偶数
下标元素（第 2k 个），high nibble = 奇数下标元素（第 2k+1 个）**——这是
CUTLASS/TensorRT-LLM 生态里"第一个元素放低位"的通行约定，但**打包顺序是
数据布局约定，不是数学问题**，同一个格式规范下理论上两种顺序都"合法"，
只有生产该 checkpoint 的具体工具链（这里是 NVIDIA modelopt）的实际选择才
是唯一正确答案。

E2M1 本身的数值表（16 个 4-bit 码字对应的浮点值）**不是猜的**：

```
nibble:  0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15
value:   0.0  0.5  1.0  1.5  2.0  3.0  4.0  6.0 -0.0 -0.5 -1.0 -1.5 -2.0 -3.0 -4.0 -6.0
```

这张表由 E2M1 格式定义本身（1 符号位 + 2 指数位[bias=1] + 1 尾数位）唯一
确定——`0.5 = 2^0 * 1.0 * 2^{-1}`（subnormal）、`1.0-1.5` 对应 exp=1、
`2.0-3.0` 对应 exp=2、`4.0-6.0` 对应 exp=3，这是浮点格式的算术推导，任何
遵循 OCP Microscaling 规范的实现都只能得到这一张表，没有第二种可能。
**这条不是本笔记要讨论的风险点**——风险完全集中在打包顺序上。

## 本轮排查过的每一条验证渠道，为什么都用不了

### 1. torch 原生 `float4_e2m1fn_x2` cast——存在但功能不可用

```
~/.venvs/vllm/bin/python -c "
import torch
b = torch.tensor([0x21,0x67,0x00,0xFF], dtype=torch.uint8, device='cuda')
v = b.view(torch.float4_e2m1fn_x2)
f = v.to(torch.float32)   # <- 崩这里
"
```

结果：**两个方向都失败**，且是不同的失败模式：

- **反量化方向**（`float4_e2m1fn_x2 -> float32`）：`../c10/core/DynamicCast.h:79:
  fetch_and_cast: ... Assertion 'false' failed`（CUDA device-side assert，
  会让整个进程的 CUDA context 从此报废，后续任何 CUDA 调用都会连带失败——
  `scripts/b1_verify_nvfp4_dequant.py` 因此把两个方向的探测拆进独立子进程，
  否则第二个探测会被第一个的 context 污染而给出假阴性）。
- **量化方向**（`float32 -> float4_e2m1fn_x2`）：`RuntimeError: copy_() does
  not support casting Float4_e2m1fn_x2 to different types`（明确的
  "未实现"，不是断言崩溃）。

结论：这个 dtype 在 `torch==2.13.0a0+gitcf30153` 这个开发版构建里**存在但
两个方向的逐元素 cast 都不可用**——大概率是一个只作为"给 CUTLASS kernel 用
的不透明标记类型"存在的占位 dtype，没有配 ATen 的通用 cast dispatch 实现。
**这条本来是最干净的独立验证路径（PyTorch/CUDA 自己的 OCP E2M1 解码，不是
本模块自己写的），失效后没有同等干净的替代**。

### 2. HF transformers 的 modelopt 量化器——没装

```
~/.venvs/vllm/bin/python -c "
from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING
print(sorted(AUTO_QUANTIZER_MAPPING.keys()))
"
# ['aqlm', 'auto-round', 'awq', 'bitnet', 'bitsandbytes_4bit', 'bitsandbytes_8bit',
#  'compressed-tensors', 'eetq', 'fbgemm_fp8', 'fouroversix', 'fp8', 'fp_quant',
#  'gptq', 'higgs', 'hqq', 'metal', 'mxfp4', 'quanto', 'quark', 'sinq', 'spqr',
#  'torchao', 'vptq']
```

没有 `"modelopt"`。`nvidia-modelopt` pip 包本身也没装（`pip show
nvidia-modelopt` 无输出）。这意味着**这台机器上没有第二个独立实现的 NVFP4
反量化器可以拿来做逐元素 diff**——不是"没找对调用方式"，是这条路径在这个
环境里压根不存在。

### 3. sparkinfer 自己的 NVFP4 代码——只有量化方向，且是 CUTLASS-DSL 不是普通
Triton

`sparkinfer/quantization/nvfp4/_kernel.py` 里唯一跟 E2M1 相关的是
`cvt_e2m1x8_f32`，底层是内联 PTX：

```
cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;
```

这是**量化方向**（float32 -> 打包 E2M1，两个 float32 寄存器 -> 一个字节），
不是反量化方向——sparkinfer 全仓库没有一处需要把 E2M1 解码回 float32（它的
真实 GEMM kernel 直接消费打包后的 uint8/E2M1，从不需要先解包）。理论上可以
反过来用：拿几个"E2M1 精确可表示"的 float32 值（比如 0.5、1.0、-6.0）过这个
量化指令，看它们编码进哪个字节的哪个 nibble，从而反推打包顺序——但这条路径
是 **CUTLASS-DSL**（`cute.jit`/`dsl_user_op`，NVIDIA 自己的 Python-到-LLVM
DSL），不是普通 Triton kernel，独立复刻一遍这套编译管线只为验证一个 nibble
顺序问题，本轮判断投入产出比不划算，没有做。**这是一条已识别但未走通的路径，
留给下一个有更多时间预算的人**。

### 4. 端到端连贯生成——本轮实际采信的证据，但性质不同

`scripts/b1_verify_full_model_smoke.py` 用真实 checkpoint 跑贪心生成，三个
prompt 全部产出连贯、语法正确、且事实正确的英文：

```
"The first president of the United States was" ->
  " George Washington. He was born in Virginia in 1732. He was a farmer
  and a soldier. He was a leader in the American Revolution."
```

（华盛顿生于弗吉尼亚、1732 年，两条事实都对。)

**这是有分量的证据，但不是同一类证据**：NVFP4 覆盖稠密 MLP（64 层
`gate/up/down_proj`，模型里数量占绝大多数的参数）+ `lm_head`——如果打包顺序
搞反了，相当于把每两个相邻权重的高低位对调，这是一个**结构性、系统性**的
错误（不是随机噪声），几乎不可能产出语法正确、事实正确的连贯输出，更可能
是乱码或高频重复的退化输出。所以"生成结果连贯且正确"是**排除了"完全搞反"
这一类最严重的错误**，但**不能排除**"某个更细微、只在特定 bit 模式下才
现形的打包 bug"（比如只在符号位处理上有细微差异，对多数权重影响很小、
不足以摧毁连贯性，但会让数值精度打折）。**证据强度：能挡住"完全错"，
挡不住"部分错"。**

## 对可信度的实际影响——诚实分级

| 结论 | 证据强度 | 可信度 |
|---|---|---|
| E2M1 数值表（16 个码字的浮点值） | 格式定义的数学推导 | **确定** |
| 打包顺序（low nibble = 偶数下标） | 行业惯例 + 连贯生成的间接佐证 | **较低置信度，唯一的真正风险点** |
| FP8（W8A8 投影层）反量化 | 单层数值对比 HF 参照，cosine>=0.9999 | **高**（有直接数值证据，见 B1 GPU 验证记录） |
| 模型数学本身（GDN/attention/RoPE/gate） | 单层数值对比 HF 参照 + 全模型连贯生成 | **高** |

**这条风险不影响 B1 已经做完的部分的可信度**（GDN、attention、loader 的
张量覆盖检查都有独立的直接数值证据），**只影响 NVFP4 反量化这一个具体环节**。

## 给后来者的具体建议（不是本笔记来拍板，只列出候选）

1. **优先级最低成本的下一步**：如果后续量化工作恰好装上了
   `nvidia-modelopt` pip 包（用于别的目的，比如真的要跑 modelopt 自己的
   校准流程），第一件事就是拿它自己的反量化函数对同一批真实 checkpoint
   张量跑一遍，和 `unpack_nvfp4_to_fp32` 逐元素 diff——这比任何本轮列出的
   路径都直接。
2. **中等成本**：写一个独立的 CUTLASS-DSL 探针（复刻 `cvt_e2m1x8_f32` 的
   量化方向，喂 E2M1 精确可表示的已知值，读出打包字节），反推打包顺序。
   本轮判断投入产出比不够，但如果 B3 真的要深入 NVFP4 GEMM kernel 选型
   （`docs/qwen36-rebuild-spec.md` §3.4/§5.2 已经在排期这件事），届时
   反正要碰 CUTLASS-DSL，这个探针可以顺手做。
3. **最高置信度但最贵**：如果 B1 的 `scripts/b1_verify_greedy_alignment.py`
   跑起来后（本轮未跑，见 `notes/`/commit `83b1362`）发现某个 workload 的
   逐层余弦在 MLP 层系统性地比 attention/GDN 层差一截（哪怕仍然"通过"阈值），
   这会是"打包顺序有问题但没有严重到摧毁连贯性"的具体信号，值得回来重新
   检查这条笔记。**这不是本轮的观测结果**（本轮没有跑逐层对比，是给后来者
   的一个具体检查清单项）。
4. **如果反过来想验证"顺序错了会造成多大伤害"**：可以直接把
   `runtime/loading/modelopt.py::unpack_nvfp4_to_fp32` 里的
   `out[:, 0::2]`/`out[:, 1::2]` 互换，重跑一次
   `scripts/b1_verify_full_model_smoke.py`，看输出退化到什么程度——如果
   互换后输出仍然"看起来还行"（连贯但质量下降），说明当前这个方向的选择
   本身就没有决定性的自证据；如果互换后输出明显崩掉（乱码/高频重复），
   反而能给现在这个方向增加一点信心（"至少两个方向不是同样合理，现在选的
   这个更好"）。**本轮未做这个 A/B**，留作候选，不代为判断结果。

## 相关文件

- `runtime/loading/modelopt.py`——`_FP4_E2M1_LUT`/`unpack_nvfp4_to_fp32`/
  `dequantize_nvfp4`，本笔记引用的全部结论的原始出处，模块 docstring 里
  有更详细的逐条证据链
- `scripts/b1_verify_nvfp4_dequant.py`——本笔记"排查过的验证渠道"第 1 条
  的可重跑脚本，两个方向的失败都能复现
- `scripts/b1_verify_full_model_smoke.py`——本笔记"端到端连贯生成"证据的
  来源脚本
- `scripts/b1_verify_greedy_alignment.py`——尚未跑，跑起来后的逐层余弦
  数字是本笔记第 4 条建议的直接检查对象
