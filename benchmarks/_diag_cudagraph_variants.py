#!/usr/bin/env python3
"""Test cudagraph wrapper variants to find what causes cosine 0.93.
Variant 1: cudagraph + disable_split_kv=True
Variant 2: cudagraph + use_tensor_cores=False  
Variant 3: non-cudagraph wrapper run through CUDA graph capture (our own graph, FI non-cudagraph wrapper)
"""
from __future__ import annotations
import os, sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("QSR_A2_CUSTOM_GEMM", "0")

import torch
MODEL = "poolside/Laguna-S-2.1-NVFP4"

def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item()

def main():
    from runtime.legacy_qwen36_vllm import EngineArgs
    args = EngineArgs(
        model=MODEL, max_model_len=4096, gpu_memory_utilization=0.80,
        enforce_eager=True, dtype="bfloat16", disable_log_stats=True,
        async_scheduling=False,
    )
    config = args.create_engine_config()

    from runtime.backends.laguna import LagunaBackend
    backend = LagunaBackend(config, num_slots=4, block_size=64, blocks_per_slot=64)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt_ids = tokenizer.encode("The capital of France is")

    # Eager reference
    backend.reset_slot(0)
    first = backend.prefill(0, prompt_ids)
    kv_len = backend.slot_kv_len[0]
    eager_logits = backend._forward([0], [first], [kv_len], qo_len=1, is_decode=True)[0].clone()
    print(f"Eager: top1={tokenizer.decode([eager_logits.argmax().item()])!r}, top5={torch.topk(eager_logits.float(), 5).values.tolist()}")

    # === Variant 3: Non-cudagraph FI wrapper + our own torch.cuda.CUDAGraph capture ===
    # This isolates: is it FlashInfer's cudagraph kernel, or torch.cuda.CUDAGraph itself?
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
    from vllm.v1.attention.backends.flashinfer import FIDecode, FlashInferMetadata, fast_plan_decode
    from runtime.legacy_qwen36_vllm import set_forward_context

    backend.reset_slot(0)
    backend.prefill(0, prompt_ids)

    bs = 1
    device = torch.device("cuda")
    # Create NON-cudagraph wrappers (one per group)
    nc_wrappers = {}
    nc_workspaces = []
    for group_key, builder in backend._metadata_builders.items():
        ws = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        nc_workspaces.append(ws)
        wrapper = BatchDecodeWithPagedKVCacheWrapper(
            ws, "NHD",
            use_cuda_graph=False,  # NON-cudagraph
            use_tensor_cores=True,
        )
        nc_wrappers[group_key] = wrapper

    # Pre-allocate buffers
    indptr_cpu = torch.tensor([0, 1], dtype=torch.int32)
    indices_gpu = torch.tensor([256], dtype=torch.int32, device=device)
    last_page_len_cpu = torch.tensor([7], dtype=torch.int32)
    slot_mapping = torch.tensor([4102], dtype=torch.long, device=device)
    input_ids = torch.tensor([first], dtype=torch.long, device=device)
    positions = torch.tensor([kv_len], dtype=torch.long, device=device)

    def plan_and_forward():
        for group_key, builder in backend._metadata_builders.items():
            wl, nqh, nkvh = group_key
            fast_plan_decode(
                nc_wrappers[group_key],
                indptr_cpu=indptr_cpu, indices=indices_gpu,
                last_page_len_cpu=last_page_len_cpu,
                num_qo_heads=nqh, num_kv_heads=nkvh, head_dim=128, page_size=16,
                pos_encoding_mode="NONE", window_left=wl, logits_soft_cap=None,
                q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn,
                sm_scale=builder.sm_scale, non_blocking=True,
                fixed_split_size=-1, disable_split_kv=False,
            )
        attn_dict = {}
        sm_dict = {}
        for group_key in nc_wrappers:
            meta = FlashInferMetadata(
                num_actual_tokens=1, slot_mapping=slot_mapping,
                q_data_type_prefill=torch.bfloat16, q_data_type_decode=torch.bfloat16,
                num_decodes=1, num_decode_tokens=1, num_prefills=0, num_prefill_tokens=0,
                causal=True, use_cascade=False, prefill=None,
                decode=FIDecode(wrapper=nc_wrappers[group_key]),
                cascade_wrapper=None,
            )
            for name in backend._layer_groups[group_key]:
                attn_dict[name] = meta
                sm_dict[name] = slot_mapping
        with set_forward_context(attn_dict, backend.vllm_config, slot_mapping=sm_dict):
            h = backend.model.forward(input_ids, positions)
        return backend.model.compute_logits(h)

    # Warmup
    for _ in range(3):
        plan_and_forward()
    torch.cuda.synchronize()

    # Eager (no graph) with non-cudagraph wrappers
    eager_nc_logits = plan_and_forward()[0].clone()
    cos_nc = cos_sim(eager_nc_logits, eager_logits)
    print(f"\nVariant 3a (non-cudagraph FI wrapper, no torch graph): cos={cos_nc:.6f}")

    # Now capture with torch.cuda.CUDAGraph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_logits = plan_and_forward()
    graph.replay()
    torch.cuda.synchronize()
    graph_nc_logits = captured_logits[0].clone()
    cos_gnc = cos_sim(graph_nc_logits, eager_logits)
    print(f"Variant 3b (non-cudagraph FI wrapper + torch.cuda.CUDAGraph): cos={cos_gnc:.6f}")

    # Replay again to check determinism
    graph.replay()
    torch.cuda.synchronize()
    graph_nc_logits2 = captured_logits[0].clone()
    cos_det = cos_sim(graph_nc_logits2, graph_nc_logits)
    print(f"Variant 3b replay determinism: cos={cos_det:.6f}")

    # === Variant 1: cudagraph FI wrapper + disable_split_kv=True ===
    from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode
    # Monkey-patch to test disable_split_kv=True
    original_run_plan = LagunaCudaGraphDecode._run_plan

    def patched_run_plan(self, slot_ids, kv_lengths):
        from vllm.v1.attention.backends.flashinfer import fast_plan_decode as fpd
        bk = self.backend
        bs2 = len(slot_ids)
        for gk, wrapper in self._decode_wrappers.items():
            wl, nqh, nkvh = gk
            kv_dtype = torch.float8_e4m3fn if "fp8" in bk._cache_dtype_str else torch.bfloat16
            builder_sm_scale = bk._metadata_builders[gk].sm_scale
            fpd(wrapper,
                indptr_cpu=self._fi_indptr_cpu[:bs2 + 1],
                indices=self._fi_indices_gpu,
                last_page_len_cpu=self._fi_last_page_len_cpu[:bs2],
                num_qo_heads=nqh, num_kv_heads=nkvh, head_dim=128, page_size=16,
                pos_encoding_mode="NONE", window_left=wl, logits_soft_cap=None,
                q_data_type=torch.bfloat16, kv_data_type=kv_dtype,
                sm_scale=builder_sm_scale, non_blocking=True,
                fixed_split_size=2048,
                disable_split_kv=True,  # <-- KEY CHANGE
            )
            wrapper._sm_scale = builder_sm_scale

    LagunaCudaGraphDecode._run_plan = patched_run_plan

    backend.reset_slot(0)
    backend.prefill(0, prompt_ids)
    cg = LagunaCudaGraphDecode(backend, batch_size=1)
    cg.capture()
    cg.reset()
    backend.reset_slot(0)
    backend.prefill(0, prompt_ids)
    cg.replay([0], [first], [kv_len])
    dskv_logits = cg._logits[0].clone()
    cos_dskv = cos_sim(dskv_logits, eager_logits)
    print(f"\nVariant 1 (cudagraph FI wrapper + disable_split_kv=True): cos={cos_dskv:.6f}")
    print(f"  top5={torch.topk(dskv_logits.float(), 5).values.tolist()}")

    # Restore
    LagunaCudaGraphDecode._run_plan = original_run_plan

    # === Variant 2: cudagraph FI wrapper + use_tensor_cores=False ===
    # Need to re-create wrappers without tensor cores
    backend.reset_slot(0)
    backend.prefill(0, prompt_ids)
    cg2 = LagunaCudaGraphDecode(backend, batch_size=1)
    # Override _init_wrappers to disable tensor cores
    original_init = LagunaCudaGraphDecode._init_wrappers
    def patched_init(self):
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
        bk = self.backend
        bs3 = self.batch_size
        for gk, builder in bk._metadata_builders.items():
            ws = torch.empty(
                builder._get_workspace_buffer().numel(), dtype=torch.uint8, device=self.device)
            wrapper = BatchDecodeWithPagedKVCacheWrapper(
                ws, "NHD", use_cuda_graph=True,
                paged_kv_indptr_buffer=self._fi_indptr_gpu[:bs3 + 1],
                paged_kv_indices_buffer=self._fi_indices_gpu,
                paged_kv_last_page_len_buffer=self._fi_last_page_len_gpu[:bs3],
                use_tensor_cores=False,  # <-- KEY CHANGE
            )
            self._decode_wrappers[gk] = wrapper
            self._workspaces.append(ws)
    LagunaCudaGraphDecode._init_wrappers = patched_init
    cg2.capture()
    cg2.reset()
    backend.reset_slot(0)
    backend.prefill(0, prompt_ids)
    cg2.replay([0], [first], [kv_len])
    notc_logits = cg2._logits[0].clone()
    cos_notc = cos_sim(notc_logits, eager_logits)
    print(f"\nVariant 2 (cudagraph FI wrapper + use_tensor_cores=False): cos={cos_notc:.6f}")
    print(f"  top5={torch.topk(notc_logits.float(), 5).values.tolist()}")

    LagunaCudaGraphDecode._init_wrappers = original_init

if __name__ == "__main__":
    main()
