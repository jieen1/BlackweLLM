"""Simplified vLLM comparison: call expert kernel directly."""
import os, sys, time
os.environ["USE_LIBUV"]="0"; os.environ["HF_HUB_OFFLINE"]="1"
os.environ["QSR_MOE_SPARKINFER"]="0"; os.environ["QSR_A2_CUTLASS_DIRECT"]="0"
os.environ["QSR_A2_CUSTOM_GEMM"]="0"
sys.path.insert(0, "/home/bot/project/qwen-sm120-runtime")
import torch, torch.nn.functional as F
torch.set_grad_enabled(False)
HIDDEN=3072; INTER=1024; NUM_EXP=256; TOP_K=10; BLOCK=16
FP4_LUT=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float32)
def unpack_fp4(p):
    p=p.cpu();lo=(p&0xF).long();hi=((p>>4)&0xF).long()
    return torch.stack([FP4_LUT[lo],FP4_LUT[hi]],dim=-1).reshape(p.shape[0],-1)
def dequant_cpu(w,sf,gs):
    return (unpack_fp4(w).to(sf.device)*sf.float().repeat_interleave(16,dim=1)/gs).to(torch.bfloat16)

print("Loading model...")
t0=time.time()
from runtime.legacy_qwen36_vllm import (EngineArgs, set_current_vllm_config, get_model,
    get_distributed_init_method, get_open_port, init_worker_distributed_environment,
    init_flashinfer_workspace, set_forward_context)
from runtime.nvfp4_cutlass_direct_patch import patch_nvfp4_prefer_cutlass_direct
from runtime.nvfp4_custom_gemm import patch_nvfp4_custom_gemm
patch_nvfp4_prefer_cutlass_direct(); patch_nvfp4_custom_gemm()
args=EngineArgs(model="poolside/Laguna-S-2.1-NVFP4",dtype="bfloat16",
    enforce_eager=True,max_model_len=256,gpu_memory_utilization=0.90)
vc=args.create_engine_config()
torch.cuda.set_device(0)
with set_current_vllm_config(vc):
    init_worker_distributed_environment(vc,rank=0,
        distributed_init_method=get_distributed_init_method("127.0.0.1",get_open_port()),local_rank=0)
    model=get_model(vllm_config=vc)
init_flashinfer_workspace(torch.device("cuda:0"))
print(f"  Loaded in {time.time()-t0:.0f}s")

layer1=model.model.layers[1]; moe=layer1.mlp
routed=moe.experts.routed_experts if hasattr(moe.experts,'routed_experts') else moe.experts
shared=getattr(moe,'shared_expert',None)
scaling=getattr(moe,'routed_scaling_factor',1.0)
print(f"  scaling={scaling}")
print(f"  w13_weight_scale_2: [{routed.w13_weight_scale_2.min():.2e}, {routed.w13_weight_scale_2.max():.2e}]")
print(f"  w13_input_scale: [{routed.w13_input_scale.min():.6e}, {routed.w13_input_scale.max():.6e}]")

M=4; torch.manual_seed(42)
hidden=torch.randn(M,HIDDEN,dtype=torch.bfloat16,device="cuda")*0.1

# Get router output
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias
e_bias=getattr(routed,"e_score_correction_bias",None)
rl,_=moe.gate(hidden); rl=rl.float()
tw,ti=fused_topk_bias(hidden,rl,"sigmoid",e_bias,TOP_K,True,routed_scaling_factor=1.0)

# Call vLLM MoE with forward context
print("\nRunning vLLM CUTLASS MoE (with forward context)...")
with set_current_vllm_config(vc):
    with set_forward_context(attn_metadata={}, vllm_config=vc):
        vllm_out = moe(hidden)
print(f"  vLLM norm = {vllm_out.float().norm().item():.6f}")

# sparkinfer from checkpoint
print("\nPreparing sparkinfer...")
from sparkinfer._lib.intrinsics import swizzle_block_scale
from sparkinfer.moe.fused_moe._impl import (allocate_tp_moe_workspace_pool,
    plan_sparkinfer_fp4_moe_weights, prepare_sparkinfer_fp4_moe_weights)
from runtime.backends.laguna_sparkinfer_moe import SparkinferMoELayer, _find_checkpoint, load_moe_layer_weights
ws=allocate_tp_moe_workspace_pool()
ckpt=_find_checkpoint(); raw=load_moe_layer_weights(ckpt,1,"cuda")
gsw=swizzle_block_scale(raw["gate_sf"].clone().contiguous())
usw=swizzle_block_scale(raw["up_sf"].clone().contiguous())
dsw=swizzle_block_scale(raw["down_sf"].clone().contiguous())
w13f=torch.cat([raw["gate_w"],raw["up_w"]],dim=1).contiguous()
w13s=torch.cat([gsw,usw],dim=1).contiguous()
wp=plan_sparkinfer_fp4_moe_weights(quant_modes="nvfp4",source_format="modelopt_nvfp4",
    activation="silu",params_dtype=torch.bfloat16,num_experts=NUM_EXP,
    hidden_size=HIDDEN,intermediate_size=INTER,w13_layout="w13")
ones0=torch.ones((),dtype=torch.float32,device="cuda")
ea=prepare_sparkinfer_fp4_moe_weights(plan=wp,
    w1_global_scale=(1.0/raw["gate_gs"]).float().contiguous(),
    w2_global_scale=(1.0/raw["down_gs"]).float().contiguous(),
    w1_fp4=w13f,w1_blockscale=w13s,
    w2_fp4=raw["down_w"].clone().contiguous(),w2_blockscale=dsw,
    a1_gscale=ones0,a2_gscale=ones0,params_dtype=torch.bfloat16)
sa=SparkinferMoELayer(ea,ws,"cuda")
si_routed=sa.forward(hidden,ti.to(torch.int32),tw)
shared_out=shared(hidden) if shared else None
si_correct=si_routed*scaling+(shared_out if shared_out is not None else 0)
si_wrong=si_routed*scaling+(shared_out/scaling if shared_out is not None else 0)

# bf16 truth
print("Computing bf16 truth...")
truth=torch.zeros(M,HIDDEN,dtype=torch.float32,device="cuda")
for t in range(M):
    x=hidden[t].float()
    for k in range(TOP_K):
        eid=ti[t,k].item();w=tw[t,k].item()
        gw=dequant_cpu(raw["gate_w"][eid],raw["gate_sf"][eid],raw["gate_gs"][eid].item()).to("cuda")
        uw=dequant_cpu(raw["up_w"][eid],raw["up_sf"][eid],raw["up_gs"][eid].item()).to("cuda")
        dw=dequant_cpu(raw["down_w"][eid],raw["down_sf"][eid],raw["down_gs"][eid].item()).to("cuda")
        truth[t]+=w*(dw.float()@(F.silu(gw.float()@x)*(uw.float()@x)))
truth_full=(truth*scaling+(shared_out.float() if shared_out is not None else 0)).to(torch.bfloat16)
truth_routed=truth.to(torch.bfloat16)

def cmp(l,o,r):
    o=o.float().view(-1);r=r.float().view(-1)
    c=F.cosine_similarity(o.unsqueeze(0),r.unsqueeze(0)).item()
    rn=(o.norm()/r.norm()).item();re=((o-r).norm()/r.norm()).item()
    pt=F.cosine_similarity(o.view(M,-1),r.view(M,-1),dim=1)
    print(f"  {l}: cos={c:.6f} rn={rn:.6f} re={re:.6f} pt={[f'{v:.4f}' for v in pt.tolist()]}")

print(f"\n{'='*60}\nFULL MoE (routed*{scaling} + shared)\n{'='*60}")
print(f"  truth={truth_full.float().norm():.6f} vLLM={vllm_out.float().norm():.6f} si={si_correct.float().norm():.6f} si_wrong={si_wrong.float().norm():.6f}")
cmp("vLLM vs truth", vllm_out, truth_full)
cmp("si(correct) vs truth", si_correct, truth_full)
cmp("si(wrong÷2.5) vs truth", si_wrong, truth_full)
cmp("si(correct) vs vLLM", si_correct, vllm_out)
cmp("si(wrong) vs vLLM", si_wrong, vllm_out)

print(f"\n{'='*60}\nROUTED ONLY\n{'='*60}")
cmp("si_routed vs truth_routed", si_routed, truth_routed)

print("\n✓ Done")
