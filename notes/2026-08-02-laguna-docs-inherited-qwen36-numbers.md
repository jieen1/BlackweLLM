# Two live "current" docs still carry Qwen3.6-era numbers mislabeled as Laguna

> Found 2026-08-02 while mining Qwen3.6-vLLM-era numbers for
> [`../docs/qwen36-rebuild-spec.md`](../docs/qwen36-rebuild-spec.md) §3 (acceptance baseline).
> Not fixed here — out of scope for that task (only `docs/roadmap.md` Track B, `docs/README.md`
> index, and the new spec file were in scope). Recorded so whoever owns `docs/roadmap.md` §0 /
> `docs/model-support.md` §2 / `README.md`'s Performance section sees it before citing either
> number again.

## Finding 1: MMLU-Pro 84.54% is Qwen3.6's score, not Laguna's

- `docs/model-support.md:49` (§2, "Laguna-S-2.1（当前生产）"): **"质量：MMLU-Pro 84.54%（414 题分层抽样）vs 官方 86.2"**
- `docs/roadmap.md:27`: **"Laguna-S-2.1 的模型能力经评测后判断为一般（MMLU-Pro 84.5%，STEM 强、人文弱）"** — this is
  part of the roadmap's own stated rationale (§0) for de-prioritizing deep performance work on Laguna.

Both attribute this score to Laguna-S-2.1. It is not. The number's actual origin, unambiguously:

- `notes/2026-07-22-quality-baseline-and-official-scores.md:220`: **"Overall accuracy = 84.54% vs
  official Qwen3.6-27B = 86.2"** — same 414-question count, same 86.2 reference, same per-category
  breakdown pattern (STEM strong / humanities weak) reproduced verbatim in
  `docs/archive/2026-07-30-architecture-two-tenant.md:446-460`, a doc whose entire subject is
  Qwen3.6-27B (dated 2026-07-22, before Laguna became the production model).
- `README.md:85-98` — the one place that gets this right — has an explicit callout: *"Read the model
  column. The quality numbers below were measured on Qwen3.6-27B in July 2026, on the execution path
  that has since been retired... they are not current-build numbers, and the current build cannot
  serve that model."* Table row: `MMLU-Pro (414q, thinking, greedy) | Qwen3.6-27B-NVFP4 | BlackweLLM,
  2026-07-22 | 84.54% | official card 86.2`.

Laguna-S-2.1 and Qwen3.6-27B are different models from different vendors (poolside vs. Qwen), different
sizes, different architectures (48 SWA + 12 full attention + MoE vs. 48 GDN + 16 full attention, dense).
An independent Laguna MMLU-Pro run landing on the exact same 84.54%/86.2/414-question/per-category
profile as Qwen3.6's run is not plausible; this is a copy-paste carried across the tenant swap, not a
coincidence. **No record found of Laguna's own MMLU-Pro number anywhere in `docs/` or `notes/`.**

Consequence: `docs/roadmap.md` §0's rationale for calling Laguna's model quality "一般" and demoting
performance work is currently resting on a number that was never measured against Laguna.

## Finding 2: the "256K@2→~93GB, 128K@4→~70GB" capacity line is Qwen3.6's table, not Laguna's

`README.md:79`, under `## Performance` / "`Laguna-S-2.1-NVFP4` on RTX PRO 6000...": **"Context
capacity: 256K @ concurrency 2 (~93 GB), 128K @ concurrency 4 (~70 GB)."**

This is verbatim the same shape as `docs/archive/2026-07-30-architecture-two-tenant.md` §5.1's Qwen3.6
capacity table (`blocks_per_slot=16384 → 256K, concurrency 2, ≈93 GB`; `blocks_per_slot=8192 → 128K,
concurrency 4, ≈70 GB`; block_size=16 tokens there).

It contradicts Laguna's own, independently-derived memory audit,
`notes/2026-07-29-gpu-memory-audit.md`, which uses a completely different accounting (MoE weights
59.5 GB dominate; `block_size=64`; KV cache only 3.05 GB at 1 slot / 131K) and produces a completely
different capacity table:

| Laguna's own table (`notes/2026-07-29-gpu-memory-audit.md`) | README's current claim |
|---|---|
| 2×256K → KV 12.0 GB → **87 GB total** (needs `gpu_mem≥0.92`) | 256K @ concurrency 2 → **~93 GB** |
| 2×128K → KV 6.0 GB → **81 GB total** | 128K @ concurrency 4 → **~70 GB** |

Neither the concurrency values nor the GB totals line up — expected, since Laguna's footprint is
weight-dominated (67 GB of weights before any KV) while Qwen3.6's is a dense 27B model with a much
larger per-token KV footprint (head_dim=256, hidden=5120, no MoE offload) — the two capacity curves
have no reason to coincide, let alone match to the GB.

## Recommendation (not actioned here)

Whoever owns these files should either (a) attribute both numbers correctly to Qwen3.6 and mark them
historical the way `README.md`'s quality table already does, or (b) replace them with a real
Laguna-specific measurement. `notes/2026-07-29-gpu-memory-audit.md` already has the raw numbers needed
for (b) on the capacity line; there is no local record of a real Laguna MMLU-Pro run for (b) on the
quality line — that would need a fresh eval.

## Why this matters for Track B

The Qwen3.6-rebuild acceptance baseline (`docs/qwen36-rebuild-spec.md` §3) cites this same 84.54%/86.2
number — correctly, as Qwen3.6's own historical score, sourced from
`notes/2026-07-22-quality-baseline-and-official-scores.md` and `README.md`'s table, not from the two
contaminated current-doc locations above. Do not "cross-check" the rebuild's future Qwen3.6 MMLU-Pro
re-measurement against `docs/roadmap.md:27` or `docs/model-support.md:49` — both are echoes of the same
original number, not independent confirmation.
