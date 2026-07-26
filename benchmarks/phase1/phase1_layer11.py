"""Test layer 11 (max gs=17024) — folding vs alpha precision at fp16 boundary."""
import os, sys
os.environ["USE_LIBUV"] = "0"; os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
import torch, torch.nn.functional as F
torch.set_grad_enabled(False)

HIDDEN=3072; INTER=1024; NUM_EXP=256; TOP_K=10; BLOCK=16
FP4_LUT = torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6], dtype=torch.float32)

def unpack_fp4(p):
    p = p.cpu(); lo=(p&0xF).long(); hi=((p>>4)&0xF).long()
    return torch.stack([FP4_LUT[lo],FP4_LUT[hi]],dim=-1).reshape(p.shape[0],-1).to(p.device if hasattr(p,'device') else 'cpu')

def dequant(w,sf,gs):
    return unpack_fp4(w).to(sf.device) * sf.float().repeat_interleave(16,dim=1) / gs

from runtime.backends.laguna_sparkinfer_moe import _find_checkpoint, load_moe_layer_weights
ckpt = _find_checkpoint()
raw = load_moe_layer_weights(ckpt, 11, "cuda")
gs = raw["gate_gs"]
print(f"Layer 11 gate_gs: min={gs.min():.0f}, max={gs.max():.0f}")
print(f"  Experts with gs > 16384: {(gs > 16384).sum().item()}/{NUM_EXP}")

# Pick experts: some below, some above 16384
below = (gs < 16384).nonzero().squeeze()[:5]
above = (gs >= 16384).nonzero().squeeze()[:5]
if above.dim() == 0: above = above.unsqueeze(0)
test_experts = torch.cat([below, above]).tolist()
print(f"  Test experts: {test_experts}")
print(f"  Their gs: {[f'{gs[e].item():.0f}' for e in test_experts]}")

M = 2
torch.manual_seed(42)
hidden = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1
topk_ids = torch.tensor([test_experts[:10]] * M, dtype=torch.long, device="cuda")
topk_weights = torch.ones(M, len(test_experts[:10]), device="cuda") / len(test_experts[:10])

# Ground truth
truth = torch.zeros(M, HIDDEN, dtype=torch.float32, device="cuda")
for t in range(M):
    x = hidden[t].float()
    for k in range(topk_ids.shape[1]):
        eid = topk_ids[t,k].item(); w = topk_weights[t,k].item()
        gw=dequant(raw["gate_w"][eid],raw["gate_sf"][eid],raw["gate_gs"][eid].item())
        uw=dequant(raw["up_w"][eid],raw["up_sf"][eid],raw["up_gs"][eid].item())
        dw=dequant(raw["down_w"][eid],raw["down_sf"][eid],raw["down_gs"][eid].item())
        truth[t] += w * (dw @ (F.silu(gw@x)*(uw@x)))
truth = truth.to(torch.bfloat16)

from sparkinfer._lib.intrinsics import swizzle_block_scale
from sparkinfer.moe.fused_moe._impl import (
    allocate_tp_moe_workspace_pool, plan_sparkinfer_fp4_moe_weights,
    prepare_sparkinfer_fp4_moe_weights,
)
from runtime.backends.laguna_sparkinfer_moe import SparkinferMoELayer, prepare_sparkinfer_layer
ws = allocate_tp_moe_workspace_pool()

# Folding
ef = prepare_sparkinfer_layer(raw, "cuda")
sf = SparkinferMoELayer(ef, ws, "cuda")
fold_out = sf.forward(hidden, topk_ids.to(torch.int32), topk_weights)

# Alpha (correct)
gsw = swizzle_block_scale(raw["gate_sf"].clone().contiguous())
usw = swizzle_block_scale(raw["up_sf"].clone().contiguous())
dsw = swizzle_block_scale(raw["down_sf"].clone().contiguous())
w13f = torch.cat([raw["gate_w"],raw["up_w"]],dim=1).contiguous()
w13s = torch.cat([gsw,usw],dim=1).contiguous()
wp = plan_sparkinfer_fp4_moe_weights(quant_modes="nvfp4",source_format="modelopt_nvfp4",
    activation="silu",params_dtype=torch.bfloat16,num_experts=NUM_EXP,
    hidden_size=HIDDEN,intermediate_size=INTER,w13_layout="w13")
ea = prepare_sparkinfer_fp4_moe_weights(plan=wp,
    w1_global_scale=(1.0/raw["gate_gs"]).float().contiguous(),
    w2_global_scale=(1.0/raw["down_gs"]).float().contiguous(),
    w1_fp4=w13f,w1_blockscale=w13s,
    w2_fp4=raw["down_w"].clone().contiguous(),w2_blockscale=dsw,
    a1_gscale=torch.ones((),dtype=torch.float32,device="cuda"),
    a2_gscale=torch.ones((),dtype=torch.float32,device="cuda"),
    params_dtype=torch.bfloat16)
sa = SparkinferMoELayer(ea, ws, "cuda")
alpha_out = sa.forward(hidden, topk_ids.to(torch.int32), topk_weights)

def cmp(label, out, ref):
    o=out.float().view(-1); r=ref.float().view(-1)
    cos=F.cosine_similarity(o.unsqueeze(0),r.unsqueeze(0)).item()
    rn=(o.norm()/r.norm()).item()
    re=((o-r).norm()/r.norm()).item()
    print(f"  {label}: cos={cos:.6f}, rel_norm={rn:.6f}, rel_err={re:.6f}")

print(f"\ntruth norm={truth.float().norm():.6f}")
print(f"fold  norm={fold_out.float().norm():.6f}")
print(f"alpha norm={alpha_out.float().norm():.6f}")
cmp("FOLD vs truth", fold_out, truth)
cmp("ALPHA vs truth", alpha_out, truth)
cmp("ALPHA vs FOLD", alpha_out, fold_out)

# Per-expert: below vs above 16384
print("\nPer-expert (below vs above fp16 boundary 16384):")
print(f"  {'E':>4} {'gs':>8} {'truth':>10} {'fold':>10} {'alpha':>10} {'f/t':>8} {'a/t':>8}")
for eid in test_experts:
    g = raw["gate_gs"][eid].item()
    ids1 = torch.tensor([[eid]], dtype=torch.int32, device="cuda")
    w1 = torch.ones(1,1,device="cuda")
    x = hidden[:1]
    gw=dequant(raw["gate_w"][eid],raw["gate_sf"][eid],g)
    uw=dequant(raw["up_w"][eid],raw["up_sf"][eid],raw["up_gs"][eid].item())
    dw=dequant(raw["down_w"][eid],raw["down_sf"][eid],raw["down_gs"][eid].item())
    t=(dw@(F.silu(gw@x[0].float())*(uw@x[0].float()))).norm().item()
    f=sf.forward(x,ids1,w1).float().norm().item()
    a=sa.forward(x,ids1,w1).float().norm().item()
    marker = " ← >16K" if g > 16384 else ""
    print(f"  {eid:4d} {g:8.0f} {t:10.4f} {f:10.4f} {a:10.4f} {f/t:8.4f} {a/t:8.4f}{marker}")

# Block scale folding damage for high-gs expert
high_eid = test_experts[-1]
g = raw["gate_gs"][high_eid].item()
bs = raw["gate_sf"][high_eid].float()
bs_fold = (bs / g).to(torch.float8_e4m3fn).float()
n_sub = ((bs_fold > 0) & (bs_fold < 1.95e-3)).sum().item()
n_zero = (bs_fold == 0).sum().item()
print(f"\nExpert {high_eid} (gs={g:.0f}) folding damage:")
print(f"  bs range: [{bs.min():.1f}, {bs.max():.1f}]")
print(f"  folded: [{bs_fold.min():.2e}, {bs_fold.max():.2e}]")
print(f"  subnormal fp8: {n_sub}/{bs_fold.numel()}, zeros: {n_zero}")
print(f"  fold precision loss: mean={((bs_fold*g-bs)/bs).abs().mean():.4f}")

print("\n✓ Layer 11 done")
