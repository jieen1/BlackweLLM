"""B0-3 probe: does sparkinfer paged attention run for Qwen3.6 full-attention
shapes (head_dim=256, gqa_group=6 i.e. 24 Q heads / 4 KV heads, page_size in
{64,128}, fp8 KV)? If it runs, check correctness against the fp32 reference
and measure throughput.

Read-only against sparkinfer: only calls its public `paged` API
(paged.plan/bind/run) exactly as sparkinfer's own tests do
(tests/attention/test_paged.py::_run_eager). No sparkinfer source is
modified.

Run with: ~/.venvs/vllm/bin/python scripts/b0_probe_paged_attention_head256.py
"""

from __future__ import annotations

import time
import traceback

import torch
from sparkinfer.attention import paged
from sparkinfer.attention.paged.reference import paged_attention_reference

NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")


def make_inputs(
    *,
    q_seqlens: list[int],
    cache_seqlens: list[int],
    page_size: int,
    seed: int = 0,
):
    torch.manual_seed(seed)
    batch = len(q_seqlens)
    total_q = sum(q_seqlens)
    q = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE) / 4

    pages_per_request = [(c + page_size - 1) // page_size for c in cache_seqlens]
    max_pages = max(pages_per_request, default=0)
    total_pages_needed = sum(pages_per_request)
    num_pages = max(1, total_pages_needed * 2)

    k_cache = (
        torch.randn(num_pages, page_size, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        / 4
    )
    v_cache = (
        torch.randn(num_pages, page_size, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE)
        / 4
    )
    page_table = torch.zeros(batch, max_pages, dtype=torch.int32, device=DEVICE)
    page_order = torch.randperm(num_pages, device=DEVICE)
    cursor = 0
    for request_idx, num_req_pages in enumerate(pages_per_request):
        if num_req_pages == 0:
            continue
        page_ids = page_order[cursor : cursor + num_req_pages].to(torch.int32)
        cursor += num_req_pages
        page_table[request_idx, :num_req_pages] = page_ids
        page_table[request_idx, num_req_pages:] = page_ids[-1]

    cache_seqlens_t = torch.tensor(cache_seqlens, dtype=torch.int32, device=DEVICE)
    q_offsets = [0]
    for q_len in q_seqlens:
        q_offsets.append(q_offsets[-1] + q_len)
    cu_seqlens_q = torch.tensor(q_offsets, dtype=torch.int32, device=DEVICE)
    return q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q


def quantize_e4m3(k_cache, v_cache, page_table, cache_seqlens, page_size):
    batch = page_table.shape[0]
    finfo = torch.finfo(torch.float8_e4m3fn)
    k_quant = torch.empty_like(k_cache, dtype=torch.float8_e4m3fn)
    v_quant = torch.empty_like(v_cache, dtype=torch.float8_e4m3fn)
    k_descale = torch.ones((batch, NUM_KV_HEADS), dtype=torch.float32, device=DEVICE)
    v_descale = torch.ones((batch, NUM_KV_HEADS), dtype=torch.float32, device=DEVICE)
    for request_idx in range(batch):
        cache_len = int(cache_seqlens[request_idx].item())
        num_pages = (cache_len + page_size - 1) // page_size
        if num_pages == 0:
            continue
        page_ids = page_table[request_idx, :num_pages].to(torch.long)
        k_pages = k_cache.index_select(0, page_ids).to(torch.float32)
        v_pages = v_cache.index_select(0, page_ids).to(torch.float32)
        k_scale = k_pages.abs().amax(dim=(0, 1, 3)) / finfo.max
        v_scale = v_pages.abs().amax(dim=(0, 1, 3)) / finfo.max
        k_scale = torch.where(k_scale > 0, k_scale, torch.ones_like(k_scale))
        v_scale = torch.where(v_scale > 0, v_scale, torch.ones_like(v_scale))
        k_descale[request_idx] = k_scale
        v_descale[request_idx] = v_scale
        k_quant[page_ids] = (k_pages / k_scale.view(1, 1, -1, 1)).to(torch.float8_e4m3fn)
        v_quant[page_ids] = (v_pages / v_scale.view(1, 1, -1, 1)).to(torch.float8_e4m3fn)
    return k_quant, v_quant, k_descale, v_descale


def run_eager(
    q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
    *, mode, k_descale=None, v_descale=None,
):
    plan = paged.plan(
        paged.Caps(
            device=q.device,
            mode=mode,
            dtype=q.dtype,
            kv_dtype=k_cache.dtype,
            num_q_heads=q.shape[1],
            num_kv_heads=k_cache.shape[2],
            head_dim_qk=q.shape[2],
            head_dim_vo=v_cache.shape[3],
            page_size=k_cache.shape[1],
            max_total_q=q.shape[0],
            max_batch=page_table.shape[0],
            max_page_table_width=page_table.shape[1],
            max_work_items=4096,
            max_partial_rows=65536,
            num_cache_pages=k_cache.shape[0],
            use_cuda_graph=False,
        )
    )
    spec = plan.scratch_specs()[0]
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=q.device)
    output = torch.empty((q.shape[0], q.shape[1], v_cache.shape[3]), dtype=q.dtype, device=q.device)
    binding = paged.bind(
        plan,
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        active_total_q=int(q.shape[0]),
        k_descale=k_descale,
        v_descale=v_descale,
    )
    out, lse = paged.run(binding=binding)
    return out, lse


def try_case(name, *, mode, q_seqlens, cache_seqlens, page_size, fp8):
    print(f"\n=== {name} ===")
    print(
        f"mode={mode} q_seqlens={q_seqlens} cache_seqlens={cache_seqlens} "
        f"page_size={page_size} fp8={fp8}"
    )
    try:
        q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q = make_inputs(
            q_seqlens=q_seqlens, cache_seqlens=cache_seqlens, page_size=page_size
        )
        k_descale = v_descale = None
        if fp8:
            k_cache, v_cache, k_descale, v_descale = quantize_e4m3(
                k_cache, v_cache, page_table, cache_seqlens_t, page_size
            )
        torch.cuda.synchronize()
        cold_start = time.perf_counter()
        out, lse = run_eager(
            q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q,
            mode=mode, k_descale=k_descale, v_descale=v_descale,
        )
        torch.cuda.synchronize()
        cold_elapsed = time.perf_counter() - cold_start
        print(f"RESULT: first-call (JIT/autotune incl.) wall time = {cold_elapsed*1e3:.1f} ms")
        ref_out, _ = paged_attention_reference(
            q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q,
            k_descale=k_descale, v_descale=v_descale, causal=True,
        )
        torch.cuda.synchronize()
        abs_err = (out - ref_out).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            out.float().reshape(-1), ref_out.float().reshape(-1), dim=0
        ).item()
        print(f"RESULT: RUNS. max_abs_err={abs_err:.5f} cosine={cos:.7f}")

        # Throughput: repeat the full plan->bind->run eager lifecycle N times
        # after warmup (this is the "eager, no CUDA graph" cost real B1 would
        # pay per forward call; JIT/autotune warmup cost is reported
        # separately below).
        n_warmup, n_iters = 5, 50
        for _ in range(n_warmup):
            run_eager(q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q,
                       mode=mode, k_descale=k_descale, v_descale=v_descale)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            run_eager(q, k_cache, v_cache, page_table, cache_seqlens_t, cu_seqlens_q,
                       mode=mode, k_descale=k_descale, v_descale=v_descale)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        total_q_val = int(q.shape[0])
        print(f"RESULT: throughput {elapsed/n_iters*1e3:.4f} ms/call "
              f"({total_q_val} q-rows/call, incl. plan+bind rebuild overhead each call)")
        return True
    except Exception as exc:
        print(f"RESULT: FAILS -- {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def main():
    print("torch:", torch.__version__)
    print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    import sparkinfer
    print("sparkinfer:", sparkinfer.__file__)

    results = {}
    for page_size in (64, 128):
        results[("decode", page_size)] = try_case(
            f"decode page_size={page_size} fp8_kv",
            mode="decode",
            q_seqlens=[1, 1, 1, 1],
            cache_seqlens=[512, 1024, 2048, 4096],
            page_size=page_size,
            fp8=True,
        )
        results[("extend", page_size)] = try_case(
            f"extend(prefill) page_size={page_size} fp8_kv",
            mode="extend",
            q_seqlens=[512, 256],
            cache_seqlens=[512, 256],
            page_size=page_size,
            fp8=True,
        )

    print("\n=== SUMMARY ===")
    for key, ok in results.items():
        print(key, "OK" if ok else "FAILS")


if __name__ == "__main__":
    main()
