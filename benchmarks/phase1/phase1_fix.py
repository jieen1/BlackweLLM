"""Phase 1 FIX: correct runtime-alpha path + compare with folding + truth."""
import os, sys
os.environ["USE_LIBUV"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
import torch
import torch.nn.functional as F
torch.set_grad_enabled(False)

HIDDEN = 3072; INTER = 1024; NUM_EXP = 256; TOP_K = 10; BLOCK = 16
FP4_LUT = torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6], dtype=torch.float32)

def unpack_fp4(packed):
    p = packed.cpu()
    low = (p & 0x0F).long(); high = ((p >> 4) & 0x0F).long()
    return torch.stack([FP4_LUT[low], FP4_LUT[high]], dim=-1).reshape(p.shape[0], -1).to(packed.device)

def dequant_weight(w_packed, w_sf, w_gs):
    fp4 = unpack_fp4(w_packed)
    sf_exp = w_sf.float().repeat_interleave(BLOCK, dim=1)
    return fp4 * sf_exp / w_gs

from runtime.backends.laguna_sparkinfer_moe import _find_checkpoint, load_moe_layer_weights
ckpt = _find_checkpoint()
raw = load_moe_layer_weights(ckpt, 1, "cuda")
gate_gs = raw["gate_gs"]

# Select experts spanning gs range
sorted_idx = gate_gs.argsort()
step = NUM_EXP // TOP_K
selected = sorted_idx[::step][:TOP_K]

M = 4
torch.manual_seed(42)
hidden = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1
topk_ids = selected.unsqueeze(0).expand(M, -1).contiguous()
topk_weights = torch.ones(M, TOP_K, device="cuda") / TOP_K

# ── Ground truth ──
print("Computing bf16 ground truth...")
truth_out = torch.zeros(M, HIDDEN, dtype=torch.float32, device="cuda")
for t in range(M):
    x = hidden[t].float()
    for k in range(TOP_K):
        eid = topk_ids[t, k].item(); w = topk_weights[t, k].item()
        gw = dequant_weight(raw["gate_w"][eid], raw["gate_sf"][eid], raw["gate_gs"][eid].item())
        uw = dequant_weight(raw["up_w"][eid], raw["up_sf"][eid], raw["up_gs"][eid].item())
        dw = dequant_weight(raw["down_w"][eid], raw["down_sf"][eid], raw["down_gs"][eid].item())
        truth_out[t] += w * (dw @ (F.silu(gw @ x) * (uw @ x)))
truth_bf16 = truth_out.to(torch.bfloat16)
print(f"  truth norm = {truth_bf16.float().norm().item():.6f}")

from sparkinfer._lib.intrinsics import swizzle_block_scale
from sparkinfer.moe.fused_moe._impl import (
    allocate_tp_moe_workspace_pool, plan_sparkinfer_fp4_moe_weights,
    prepare_sparkinfer_fp4_moe_weights,
)
from runtime.backends.laguna_sparkinfer_moe import SparkinferMoELayer, prepare_sparkinfer_layer
workspace = allocate_tp_moe_workspace_pool()

def compare(label, out, ref):
    o = out.float().view(-1); r = ref.float().view(-1)
    cos = F.cosine_similarity(o.unsqueeze(0), r.unsqueeze(0)).item()
    rel_norm = (o.norm() / r.norm()).item()
    rel_err = ((o - r).norm() / r.norm()).item()
    per_tok = F.cosine_similarity(out.float(), ref.float(), dim=1)
    print(f"  {label}: cos={cos:.6f}, rel_norm={rel_norm:.6f}, rel_err={rel_err:.6f}")
    print(f"    per-token: {[f'{v:.4f}' for v in per_tok.tolist()]}")

# ── Config A: Folding (baseline, known working) ──
print("\n=== Config A: FOLDING (bs/gs → fp8, unit alpha) ===")
experts_fold = prepare_sparkinfer_layer(raw, "cuda")
si_fold = SparkinferMoELayer(experts_fold, workspace, "cuda")
fold_out = si_fold.forward(hidden, topk_ids.to(torch.int32), topk_weights)
print(f"  norm = {fold_out.float().norm().item():.6f}")
compare("FOLD vs truth", fold_out, truth_bf16)

# ── Config B: Runtime-alpha with CORRECT scales ──
# Key insight: a1_gscale controls activation quantization (keep 1.0 like folding)
# w1_global_scale = 1/checkpoint_gs (the correct dequant factor)
print("\n=== Config B: ALPHA (a1_gscale=1.0, w1_gs=1/checkpoint_gs) ===")
gate_sf_sw = swizzle_block_scale(raw["gate_sf"].clone().contiguous())
up_sf_sw = swizzle_block_scale(raw["up_sf"].clone().contiguous())
down_sf_sw = swizzle_block_scale(raw["down_sf"].clone().contiguous())
w13_fp4 = torch.cat([raw["gate_w"], raw["up_w"]], dim=1).contiguous()
w13_sf = torch.cat([gate_sf_sw, up_sf_sw], dim=1).contiguous()

# w1_global_scale = 1/checkpoint_gs (the runtime alpha we want)
w1_gs_correct = (1.0 / raw["gate_gs"]).float().contiguous()
w2_gs_correct = (1.0 / raw["down_gs"]).float().contiguous()
ones_0 = torch.ones((), dtype=torch.float32, device="cuda")

print(f"  w1_global_scale range: [{w1_gs_correct.min():.2e}, {w1_gs_correct.max():.2e}]")

wplan = plan_sparkinfer_fp4_moe_weights(
    quant_modes="nvfp4", source_format="modelopt_nvfp4",
    activation="silu", params_dtype=torch.bfloat16,
    num_experts=NUM_EXP, hidden_size=HIDDEN,
    intermediate_size=INTER, w13_layout="w13",
)
experts_alpha = prepare_sparkinfer_fp4_moe_weights(
    plan=wplan,
    w1_global_scale=w1_gs_correct, w2_global_scale=w2_gs_correct,
    w1_fp4=w13_fp4, w1_blockscale=w13_sf,
    w2_fp4=raw["down_w"].clone().contiguous(), w2_blockscale=down_sf_sw,
    a1_gscale=ones_0, a2_gscale=ones_0,  # unit → dynamic activation quant
    params_dtype=torch.bfloat16,
)
print(f"  w1_alphas: [{experts_alpha.w1_alphas.min():.2e}, {experts_alpha.w1_alphas.max():.2e}]")
print(f"  w1_alphas zeros: {(experts_alpha.w1_alphas == 0).sum().item()}")

si_alpha = SparkinferMoELayer(experts_alpha, workspace, "cuda")
alpha_out = si_alpha.forward(hidden, topk_ids.to(torch.int32), topk_weights)
print(f"  norm = {alpha_out.float().norm().item():.6f}")
compare("ALPHA vs truth", alpha_out, truth_bf16)
compare("ALPHA vs FOLD", alpha_out, fold_out)

# ── Config C: vLLM-style (ws2 as w_global_scale, input_scale as a_gscale) ──
# This mimics prepare_sparkinfer_layer_from_vllm but from checkpoint
# ws2 = 1/checkpoint_gs, input_scale ≈ 1.0 (no calibration data)
# runtime_alpha = ws2 / input_scale = 1/checkpoint_gs
print("\n=== Config C: vLLM-style (ws2=1/gs, a_gscale=1.0) ===")
# Same as B since input_scale=1.0 → identical
print("  (identical to Config B with unit input_scale)")

# ── Per-expert comparison ──
print("\n=== Per-expert norms ===")
print(f"  {'Expert':>6} {'gs':>8} {'truth':>10} {'fold':>10} {'alpha':>10} {'f/t':>8} {'a/t':>8}")
for eid in selected.tolist():
    gs = raw["gate_gs"][eid].item()
    ids1 = torch.tensor([[eid]], dtype=torch.int32, device="cuda")
    w1 = torch.ones(1, 1, device="cuda")
    x = hidden[:1]
    
    # Truth
    gw = dequant_weight(raw["gate_w"][eid], raw["gate_sf"][eid], gs)
    uw = dequant_weight(raw["up_w"][eid], raw["up_sf"][eid], raw["up_gs"][eid].item())
    dw = dequant_weight(raw["down_w"][eid], raw["down_sf"][eid], raw["down_gs"][eid].item())
    t = (dw @ (F.silu(gw @ x[0].float()) * (uw @ x[0].float()))).norm().item()
    
    f = si_fold.forward(x, ids1, w1).float().norm().item()
    a = si_alpha.forward(x, ids1, w1).float().norm().item()
    
    print(f"  {eid:6d} {gs:8.0f} {t:10.4f} {f:10.4f} {a:10.4f} {f/t:8.4f} {a/t:8.4f}")

# ── Check block scale precision ──
print("\n=== Block scale precision analysis ===")
for eid in [selected[0].item(), selected[-1].item()]:
    gs = raw["gate_gs"][eid].item()
    bs = raw["gate_sf"][eid].float()
    bs_folded = (bs / gs).to(torch.float8_e4m3fn).float()
    bs_orig_fp8 = bs.to(torch.float8_e4m3fn).float()
    
    # Precision loss from folding
    fold_err = ((bs_folded * gs - bs) / bs).abs()
    orig_err = ((bs_orig_fp8 - bs) / bs).abs()
    
    print(f"  Expert {eid} (gs={gs:.0f}):")
    print(f"    bs range: [{bs.min():.4f}, {bs.max():.4f}]")
    print(f"    folded range: [{bs_folded.min():.2e}, {bs_folded.max():.2e}]")
    print(f"    fold precision loss: mean={fold_err.mean():.4f}, max={fold_err.max():.4f}")
    print(f"    orig fp8 precision loss: mean={orig_err.mean():.4f}, max={orig_err.max():.4f}")
    print(f"    folded zeros: {(bs_folded == 0).sum().item()}/{bs_folded.numel()}")

print("\n✓ Done")
