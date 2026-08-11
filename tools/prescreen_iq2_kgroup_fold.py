"""Phase 2B-0 representation proof, exact IQ2 folding path.

Folds the per-K16 IQ2 delta into the INT8 B fragment directly from the
packed 74-byte blocks (plan §4.3 direct-folding formula):
    delta_j = d * (0.5 + nibble_j) * 0.25
    sA = max|A[K-group]|/127
    sB = 43 * max_j|delta_j|/127
    qA = round(A/sA)
    qB = sign * round(magnitude * delta_j / sB)
    acc32 = sum_{K-group}(qA*qB);  partial = float(acc32) * sA * sB

Compares against the exact dequant oracle per K-group {32,64,128,256}.
"""
import sys
from pathlib import Path

import torch

DEVICE = "cuda"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loader.gguf_quant_tables import IQ2XS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS  # noqa: E402
from runtime.model.dsv4_quant import dequantize_iq2_xs  # noqa: E402

MODEL = Path("/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/"
             "DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf")
GATE_NAME = "blk.4.ffn_gate_exps.weight"
UP_NAME = "blk.4.ffn_up_exps.weight"
DOWN_NAME = "blk.4.ffn_down_exps.weight"
EXPERTS, INTER, HIDDEN = 256, 2048, 4096
IQ2 = 256
IQ2_BYTES = 74

_tables_cache: dict = {}


def _tables() -> dict:
    if not _tables_cache:
        _tables_cache["grid"] = torch.tensor(IQ2XS_GRID, dtype=torch.int64)
        _tables_cache["ksigns"] = torch.tensor(KSIGNS_IQ2XS, dtype=torch.int32)
        _tables_cache["kmask"] = torch.tensor(KMASK_IQ2XS, dtype=torch.int32)
        _tables_cache["subblock"] = torch.arange(32) // 4
        _tables_cache["low_half"] = (torch.arange(32) % 4) < 2
    return _tables_cache


def load_expert_packed(name: str, expert: int, rows: int, cols: int) -> torch.Tensor:
    from runtime.loading.gguf import load_gguf_tensors
    t = load_gguf_tensors(MODEL, {name})[name]
    row_bytes = (cols // IQ2) * IQ2_BYTES
    start = expert * rows * row_bytes
    return t.data[start:start + rows * row_bytes].cuda()


def iq2_values(packed: torch.Tensor, rows: int, cols: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed magnitudes [rows*cols] and deltas [rows*cols] (fp32)."""
    flat = packed.reshape(-1)
    blocks = flat.reshape(-1, IQ2_BYTES)
    tabs = {k: v.to(DEVICE) for k, v in _tables().items()}
    d = blocks[:, :2].view(torch.float16).squeeze(-1).to(torch.float32)              # per block
    codes = blocks[:, 2:66].view(torch.int16).to(torch.int32) & 0xFFFF
    scales = blocks[:, 66:74].to(torch.int32)
    entries = tabs["grid"][codes & 511]
    magnitudes = torch.stack([(entries >> (8 * j)) & 0xFF for j in range(8)], dim=-1).float()
    sign_bytes = tabs["ksigns"][codes >> 9]
    sign_mask = (sign_bytes.unsqueeze(-1) & tabs["kmask"]) != 0
    signed = torch.where(sign_mask, -magnitudes, magnitudes)              # [B, 32, 8]
    lo = (scales & 0xF).float()
    hi = (scales >> 4).float()
    db0 = d.unsqueeze(1) * (0.5 + lo) * 0.25                              # [B, 8]
    db1 = d.unsqueeze(1) * (0.5 + hi) * 0.25
    # expand each scale byte to its 4 codes: subblock index within the 8 scale bytes
    sub = torch.arange(32, device=DEVICE) // 4
    low = (torch.arange(32, device=DEVICE) % 4) < 2
    deltas = torch.where(low, db0[:, sub], db1[:, sub])                    # [B, 32]
    # expand to per-value
    signed_v = signed.reshape(-1)                                         # [B*256]
    deltas_v = deltas.unsqueeze(-1).expand(-1, -1, 8).reshape(-1)         # [B*256]
    return signed_v, deltas_v


def k_group_fold_gemm_iq2(
    A: torch.Tensor,
    packed: torch.Tensor,
    rows: int,
    cols: int,
    K_GROUP: int,
) -> torch.Tensor:
    """Scale-amortized GEMM directly on packed IQ2_XS (plan §4.3)."""
    M, K = A.shape
    N = rows
    signed, deltas = iq2_values(packed, rows, cols)   # [N*K], [N*K]
    signed = signed.view(N, K)
    deltas = deltas.view(N, K)
    n_grp = K // K_GROUP
    # A [M, K] -> [M, n_grp, G]
    A_g = A.view(M, n_grp, K_GROUP)
    sA = A_g.abs().max(dim=-1, keepdim=True).values / 127.0
    sA = torch.clamp(sA, min=1e-8)
    qA = (A_g / sA).round().clamp(-128, 127).float()      # [M, n_grp, G]
    sg = signed.view(N, n_grp, K_GROUP)
    dl = deltas.view(N, n_grp, K_GROUP)
    sB = 43 * dl.abs().max(dim=-1, keepdim=True).values / 127.0
    sB = torch.clamp(sB, min=1e-8)
    qB = (sg * dl / sB).round().clamp(-128, 127).float()  # [N, n_grp, G]
    # per-group dot: out_g[m,n,g] = sum_g qA[m,g,:] * qB[n,g,:]
    acc = torch.einsum('mgk,ngk->mng', qA, qB)            # [M, N, n_grp]
    # sA [M, n_grp], sB [N, n_grp] -> scale [M, N, n_grp]
    scale = sA.squeeze(-1).unsqueeze(1) * sB.squeeze(-1).unsqueeze(0)
    return (acc * scale).sum(dim=-1)


def report(name: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    cos = (got * ref).sum() / (got.norm() * ref.norm() + 1e-9)
    # per-row cosine
    cos_row = (got * ref).sum(dim=-1) / (got.norm(dim=-1) * ref.norm(dim=-1) + 1e-9)
    rel = (got - ref).abs() / ref.abs().clamp_min(1e-3)
    print(f"  {name}: cos={cos.item():.7f} cos_min_row={cos_row.min().item():.7f} "
          f"rel_l2_max={rel.norm(dim=-1).max().item():.4f}")


def main() -> None:
    print(f"model: {MODEL.name} ({MODEL.stat().st_size/1e9:.1f} GB)")
    torch.manual_seed(20260811)
    A = (torch.randn(24, HIDDEN) * 0.1).cuda()
    pkg = load_expert_packed(GATE_NAME, 0, INTER, HIDDEN)
    pku = load_expert_packed(UP_NAME, 0, INTER, HIDDEN)
    pkd = load_expert_packed(DOWN_NAME, 0, HIDDEN, INTER)

    # exact oracle
    Wg = dequantize_iq2_xs(pkg).reshape(INTER, HIDDEN)
    Wu = dequantize_iq2_xs(pku).reshape(INTER, HIDDEN)
    Wd = dequantize_iq2_xs(pkd).reshape(HIDDEN, INTER)
    ref_g = A @ Wg.t()
    ref_u = A @ Wu.t()
    h_ref = (torch.nn.functional.silu(torch.clamp(ref_g, max=10.0))
             * torch.clamp(ref_u, min=-10.0, max=10.0))
    ref_d = h_ref @ Wd.t()

    print("\n=== direct IQ2 K-group folding, gate ===")
    for kg in (32, 64, 128, 256):
        got = k_group_fold_gemm_iq2(A, pkg, INTER, HIDDEN, kg)
        report(f"K{kg} gate", got, ref_g)

    print("\n=== direct IQ2 K-group folding, gate+up->SwiGLU->down ===")
    for kg in (32, 64, 128, 256):
        gate = k_group_fold_gemm_iq2(A, pkg, INTER, HIDDEN, kg)
        up = k_group_fold_gemm_iq2(A, pku, INTER, HIDDEN, kg)
        h = (torch.nn.functional.silu(torch.clamp(gate, max=10.0))
             * torch.clamp(up, min=-10.0, max=10.0))
        down = k_group_fold_gemm_iq2(h, pkd, HIDDEN, INTER, kg)
        print(f"  --- K{kg} full chain ---")
        report(f"K{kg} gate", gate, ref_g)
        report(f"K{kg} up", up, ref_u)
        report(f"K{kg} down", down, ref_d)


if __name__ == "__main__":
    main()
