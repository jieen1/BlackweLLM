#!/usr/bin/env python3
"""Test cudagraph wrapper variants: disable_split_kv and use_tensor_cores."""
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
    from oracle.qwen36_vllm.vllm_compat import EngineArgs
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
    print(f"Eager: top5={torch.topk(eager_logits.float(), 5).values.tolist()}")

    from runtime.backends.laguna_cuda_graph import LagunaCudaGraphDecode
    from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

    # === Variant 1: cudagraph + disable_split_kv=True ===
    original_run_plan = LagunaCudaGraphDecode._run_plan
    def patched_run_plan_dskv(self, slot_ids, kv_lengths):
        from vllm.v1.attention.backends.flashinfer import fast_plan_decode as fpd
        bk = self.backend
        bs2 = len(slot_ids)
        for gk, wrapper in self._decode_wrappers.items():
            wl, nqh, nkvh = gk
            kv_dtype = torch.float8_e4m3fn if "fp8" in bk._cache_dtype_str else torch.bfloat16
            bsm = bk._metadata_builders[gk].sm_scale
            fpd(wrapper,
                indptr_cpu=self._fi_indptr_cpu[:bs2+1], indices=self._fi_indices_gpu,
                last_page_len_cpu=self._fi_last_page_len_cpu[:bs2],
                num_qo_heads=nqh, num_kv_heads=nkvh, head_dim=128, page_size=16,
                pos_encoding_mode="NONE", window_left=wl, logits_soft_cap=None,
                q_data_type=torch.bfloat16, kv_data_type=kv_dtype,
                sm_scale=bsm, non_blocking=True,
                fixed_split_size=2048, disable_split_kv=True)
            wrapper._sm_scale = bsm
    LagunaCudaGraphDecode._run_plan = patched_run_plan_dskv

    backend.reset_slot(0); backend.prefill(0, prompt_ids)
    cg1 = LagunaCudaGraphDecode(backend, batch_size=1)
    cg1.capture(); cg1.reset()
    backend.reset_slot(0); backend.prefill(0, prompt_ids)
    cg1.replay([0], [first], [kv_len])
    v1_logits = cg1._logits[0].clone()
    print(f"\nV1 (cudagraph + disable_split_kv=True): cos={cos_sim(v1_logits, eager_logits):.6f}")
    print(f"  top5={torch.topk(v1_logits.float(), 5).values.tolist()}")
    del cg1
    torch.cuda.empty_cache()

    LagunaCudaGraphDecode._run_plan = original_run_plan

    # === Variant 2: cudagraph + use_tensor_cores=False ===
    original_init = LagunaCudaGraphDecode._init_wrappers
    def patched_init_notc(self):
        bk = self.backend
        bs3 = self.batch_size
        for gk, builder in bk._metadata_builders.items():
            ws = torch.empty(builder._get_workspace_buffer().numel(), dtype=torch.uint8, device=self.device)
            wrapper = BatchDecodeWithPagedKVCacheWrapper(
                ws, "NHD", use_cuda_graph=True,
                paged_kv_indptr_buffer=self._fi_indptr_gpu[:bs3+1],
                paged_kv_indices_buffer=self._fi_indices_gpu,
                paged_kv_last_page_len_buffer=self._fi_last_page_len_gpu[:bs3],
                use_tensor_cores=False)
            self._decode_wrappers[gk] = wrapper
            self._workspaces.append(ws)
    LagunaCudaGraphDecode._init_wrappers = patched_init_notc

    backend.reset_slot(0); backend.prefill(0, prompt_ids)
    cg2 = LagunaCudaGraphDecode(backend, batch_size=1)
    cg2.capture(); cg2.reset()
    backend.reset_slot(0); backend.prefill(0, prompt_ids)
    cg2.replay([0], [first], [kv_len])
    v2_logits = cg2._logits[0].clone()
    print(f"\nV2 (cudagraph + use_tensor_cores=False): cos={cos_sim(v2_logits, eager_logits):.6f}")
    print(f"  top5={torch.topk(v2_logits.float(), 5).values.tolist()}")
    del cg2
    torch.cuda.empty_cache()

    LagunaCudaGraphDecode._init_wrappers = original_init

    # === Variant 0 (control): current cudagraph (use_tensor_cores=True, disable_split_kv=False) ===
    backend.reset_slot(0); backend.prefill(0, prompt_ids)
    cg0 = LagunaCudaGraphDecode(backend, batch_size=1)
    cg0.capture(); cg0.reset()
    backend.reset_slot(0); backend.prefill(0, prompt_ids)
    cg0.replay([0], [first], [kv_len])
    v0_logits = cg0._logits[0].clone()
    print(f"\nV0 (current cudagraph, control): cos={cos_sim(v0_logits, eager_logits):.6f}")
    print(f"  top5={torch.topk(v0_logits.float(), 5).values.tolist()}")

if __name__ == "__main__":
    main()
