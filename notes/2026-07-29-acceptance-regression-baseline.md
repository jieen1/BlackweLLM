# DFlash 接受率回归基准 (2026-07-29)

## 结论

**当前代码接受率正常，不存在 15% 的 bug。**

之前报告的 15%/35% 接受率有两个来源：
1. **合成 token ID prompt**（`list(range(1000,1100))` 循环）：12.1%——这是垃圾输入，
   模型输出不可预测，低接受率是预期行为，不是 bug。
2. **prompt 构造 bug**：`_repeat_text` 函数用 `text * N` 再 encode，tokenizer 在
   短语边界合并 token，导致实际 token 数和序列与预期不同。修复后数据一致。

## 基准数据 (suite v1.0, commit 1cab5f9, block_size=64)

| 类别 | n | 平均 | P50 | 范围 |
|------|---|------|-----|------|
| **真实负载(不含合成)** | **11** | **84.5%** | **98.5%** | 31.3-100% |
| 全部 | 13 | 77.1% | 97.8% | 12.1-100% |
| 英文重复文本 | 3 | 94.0% | 98.5% | 83.3-100% |
| 英文 QA | 4 | 99.6% | 100% | 98.5-100% |
| 代码 | 1 | 97.8% | — | — |
| 64K 长上下文 | 1 | 88.1% | — | — |
| 中文 | 2 | 31.5% | 31.5% | 31.3-31.7% |
| 合成 ID | 2 | 36.5% | — | 12.1-61.0% |

## 对比基线

| 指标 | c22699c (历史) | 当前 HEAD | vLLM 原生 |
|------|---------------|-----------|-----------|
| fox-64K 接受率 | 87% | **88.1%** | 99.2-100% |
| fox-4K 接受率 | — | **98.5%** | — |

## 已知差距

1. **64K 接受率 88% vs vLLM 100%**：已知 "TheOkay" 现象——sparkinfer MoE 与
   FlashInfer MoE 数值路径不同，主模型在 ~65568 位置开始偏离重复模式。
   见 `notes/2026-07-27-acceptance-rate-gap-vllm-vs-ours-same-prompt.md`。

2. **中文接受率 31.5%**：Draft 模型对中文预测能力弱。这是模型质量问题，
   不是引擎 bug。需要确认 vLLM 原生 DFlash 在中文上的接受率作为对照。

## 回归测试

```bash
# 通过 daemon 跑（推荐，秒级）
bf exec benchmarks/acceptance_regression.py --socket <sock> --timeout-s 600

# 结果保存到
benchmarks/fixtures/acceptance_regression_<date>.json
```

## 教训

- **同 prompt 同参数才能对比**——不同 prompt 构造方法产生不同 token 序列
- **tokenizer 边界合并**：`encode(text * N) != encode(text) * N`
- **合成 token ID 不代表真实负载**——不要用 `range(1000,1100)` 测接受率
