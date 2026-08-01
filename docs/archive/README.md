# 归档文档索引

这里存放**已经不再反映当前状态**的文档。归档而非删除，是因为它们记录了
决策过程和当时的实测数据，在追溯"为什么当初这么做"时仍有价值。

> ⚠️ **不要用这里的任何文档判断当前状态。** 当前状态见
> [`../roadmap.md`](../roadmap.md) 与 [`../architecture.md`](../architecture.md)。

归档日期：2026-08-01（基线 commit `ce21eb5`）

---

| 文件 | 原路径 | 最后更新 | 归档原因 | 被谁取代 |
|---|---|---|---|---|
| `2026-07-26-roadmap-vllm-removal.md` | `docs/roadmap.md` | 2026-07-26 | 整份路线图围绕「B7 去 vLLM 化」主线组织，而该主线已于 2026-07-30（`a9cb932`）完成；Track E 的多模型规划仍假设 Qwen3.6 租户存在，而它已被摘除 | [`../roadmap.md`](../roadmap.md) |
| `2026-07-30-architecture-two-tenant.md` | `docs/architecture.md` | 2026-07-30 | 描述的是「剥离进行中 + Qwen3.6/Laguna 双租户 + OpRegistry 渐进替换」的中间态，三个前提全部已不成立；文中引用的 `model/qwen36_model.py`、`gdn_layer.py` 等文件从未存在或已删除 | [`../architecture.md`](../architecture.md) |
| `2026-07-20-PROGRESS.md` | `PROGRESS.md` | 2026-07-20 | 251 KB 的流水账式进度日志，冻结在 2026-07-20；此后 300+ 次提交、vLLM 完全剥离、Qwen3.6 摘除均未记录。其中的 kernel 实验数据（FP4 KV decode 等）仍有参考价值 | [`../roadmap.md`](../roadmap.md) §1 现状盘点 |
| `2026-07-18-项目实施规划-qwen36-only.md` | `项目实施规划.md` | 2026-07-18 | 项目最初的实施规划，合同是「只服务 Qwen3.6-27B-NVFP4 + SM120 + 单卡 + 并发≤4」。生产模型已换成 Laguna，Qwen3.6 将以完全不同的方式（走抽象层，零 vLLM）重新接入 | [`../roadmap.md`](../roadmap.md) |
| `2026-07-27-bfdiag-handoff.md` | `docs/bfdiag-handoff.md` | 2026-07-27 | 一次性的团队交接说明，内容已被完整的使用指南覆盖 | [`../diagnostics-guide.md`](../diagnostics-guide.md) |

---

## 从归档里还能取到什么

| 想找 | 去哪 |
|---|---|
| 为什么选择完全剥离 vLLM，而不是继续 fork | `2026-07-26-roadmap-vllm-removal.md` §5 B7 |
| 为什么从 Qwen3.6 换成 Laguna 作为生产模型 | `2026-07-26-roadmap-vllm-removal.md` §8 E3「为什么换：与 HY3 的集成成本对比」 |
| 旧的双平面 / OpRegistry 渐进替换策略 | `2026-07-30-architecture-two-tenant.md` §2.3 |
| GDN 递归状态在投机解码下的回滚方案（Qwen3.6 时代） | `2026-07-30-architecture-two-tenant.md` §6.2 —— **Track B3 的重要先验** |
| 权重加载流水线的终态设计 | `2026-07-30-architecture-two-tenant.md` §3.4 |
| FP4 KV decode kernel 的实验结论（FP4 对 decode 速度不是赢家） | `2026-07-20-PROGRESS.md` 开头 |
| 项目最初的分阶段设计（Phase 0–11） | `2026-07-18-项目实施规划-qwen36-only.md` |
