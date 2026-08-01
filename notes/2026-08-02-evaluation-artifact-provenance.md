# 评测产物归属清单（证据固化）

日期：2026-08-02

## 为什么有这份文档

`c53bd7c` 更正了"Laguna MMLU-Pro 84.5%"这个误标，它引用的证据是
`evalplus_results/` 下的结果文件。但**那个目录是 gitignore 的**
（`.gitignore`:「Evaluation results (large, reproducible)」），随时可能被
清理、被覆盖、或在另一台机器上根本不存在。

一个更正如果只指向一个不受版本控制的文件，那它和它要纠正的错误一样脆弱。
所以把关键字段抄进来。**这不是结果的替代品，是归属的存根。**

## 清单（2026-08-02 逐个 `json.load` 读出，非文件名推断）

| 文件 | `model` | 准确率 | n | 日期 |
|---|---|---|---|---|
| `official/mmlu_pro_think_c4.json` | **`qwen3.6`** | **84.54** | 414 | 2026-07-22 |
| `official/mmlu_pro_think_test.json` | `qwen3.6` | 85.71 | 14 | 2026-07-22 |
| `official/mmlu_pro_nt_smoke.json` | `qwen3.6` | 77.44 | 133 | 2026-07-22 |
| `quality/our_runtime_fast.json` | `qwen3.6` | — | — | 2026-07-22 |
| `quality/our_runtime_longctx.json` | `qwen3.6` | — | — | 2026-07-22 |

**五份全部是 `qwen3.6`。仓库里不存在任何 `model` 字段为 Laguna 的评测产物。**

## 由此确定的三件事

1. **84.54% 属于 Qwen3.6-27B**，测于 2026-07-22，跑在此后已退役的
   vLLM 执行路径上。`docs/roadmap.md` §0 曾把它当作"Laguna 能力一般"的依据，
   已于 `c53bd7c` 更正。
2. **Laguna 没有任何官方对标评测数据**。不是"数据不好找"，是从未跑过。
   建立它是 Track C 的 C9。
3. 那次评测的 harness（`benchmarks/official/mmlu_pro_eval.py` +
   `quality_regression.py`）从建成起就只指向过 Qwen3.6。C9 要做的第一件事
   是把它指向当前生产模型，这一步本身没有先例可循。

## 复现方式

```bash
python -c "
import json, glob
for f in sorted(glob.glob('evalplus_results/**/*.json', recursive=True)):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if 'model' in d:
        print(f, {k: d[k] for k in ('model','accuracy','n','label') if k in d})
"
```

产物本身可由 `benchmarks/official/mmlu_pro_eval.py` 重跑（数小时量级，
需要 GPU 与目标模型），所以它们被 gitignore 是对的；需要固化的从来只是
**哪个数字属于哪个模型**这件事。

## 相关

- `docs/roadmap.md` §0（转向依据，已更正）与 Track C 的 C9（首次建立 Laguna 基线）
- `docs/model-support.md` §2（Laguna 质量段，已更正）
- `notes/2026-07-22-quality-baseline-and-official-scores.md`（当年的方法论，
  其中"我们的运行时"指的是 Qwen3.6/vLLM 路径，不是今天的自研栈）
- `notes/2026-08-02-laguna-docs-inherited-qwen36-numbers.md`（另一路 agent
  独立发现同一问题的记录）
