# GPU Memory Audit (2026-07-29)

## 完整显存分解 (1 slot, 131K context, block_size=64)

| Component | Size | Note |
|-----------|------|------|
| Sparkinfer MoE weights | **59.5 GB** | 47层 × (w1_fp4=768MB + w1_scale=96MB + w2_fp4=384MB + w2_scale=48MB) |
| Non-MoE params | 7.3 GB | attn QKV/O, embed, lm_head, norm, gate, shared_expert |
| Main KV cache | 3.05 GB | blocks_per_slot=2049, 12 full-attn layers bf16 + 36 SWA fp8 ring |
| SWA scratch | 0.6 GB | window + chunk overlap buffer |
| Draft KV cache | 0.007 GB | 6-layer draft, ring buffer |
| Unaccounted | 5.5 GB | CUDA context, sparkinfer workspaces, fragmentation |
| **Total allocated** | **76.0 GB** | |
| **GPU total** | **95.6 GB** | RTX PRO 6000 Blackwell |
| **Free** | **15.7 GB** | |

## MoE 权重格式 (SPARKINFERFP4ExpertWeights)

每层:
- `w1_fp4`: [256, 2048, 1536] uint8 = 768 MB (gate+up, NVFP4 repacked)
- `w1_blockscale`: [256, 2048, 192] fp8 = 96 MB
- `w2_fp4`: [256, 3072, 512] uint8 = 384 MB (down, NVFP4 repacked)
- `w2_blockscale`: [256, 3072, 64] fp8 = 48 MB
- **每层合计: 1296 MB ≈ 1.27 GB**

## 扩展性分析

KV per token (full-attn only, SWA is ring-buffered fixed):
- 12 layers × 8 KV heads × 128 head_dim × 2 bytes = 24,576 bytes = 24 KB/token

| Config | KV needed | Total needed | Fits? |
|--------|-----------|-------------|-------|
| 1×128K | 3.0 GB | 78 GB | ✓ |
| 1×200K | 4.7 GB | 80 GB | ✓ |
| 1×256K | 6.0 GB | 81 GB | ✓ |
| 2×128K | 6.0 GB | 81 GB | ✓ |
| 2×200K | 9.4 GB | 85 GB | ✓ (tight at 0.90) |
| 2×256K | 12.0 GB | 87 GB | ✓ (needs gpu_mem≥0.92) |

## 关键发现

1. **MoE权重是显存大户 (59.5 GB = 78%)**，不是KV cache
2. KV cache 只占 3 GB (131K)，扩展到 2×256K 也只需 12 GB
3. sparkinfer 将 NVFP4 权重 repack 为 uint8 格式，大小不变（不是翻倍）
4. 2×256K 理论上可行，需要 gpu_memory_utilization ≥ 0.92
