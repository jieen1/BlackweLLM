#!/usr/bin/env python3
"""Probe: replicate Qwen36MTPDraftCudaGraph's decode-attention driver
exactly (for_contract + prepare_decode_graph_replay_state, worst-case
binding) and time it at 128K live length -- tells whether the draft
graph's 1.37 ms/call (live trace) comes from the replay-state chunking
or from elsewhere.

Run: ~/.venvs/vllm/bin/python scripts/probe_draft_graph_attn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _ROOT)

import torch  # noqa: E402
from sparkinfer.attention.paged._forward import paged_attention_forward  # noqa: E402
from sparkinfer.attention.paged._scratch import build_paged_attention_binding  # noqa: E402
from sparkinfer.attention.paged.workspace import PagedAttentionWorkspace  # noqa: E402

# Qwen3.6 MTP layer full-attention geometry
NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
PAGE_SIZE = 128
PAGES_PER_SLOT = 2048  # 262144 tokens
NUM_SLOTS = 5  # 4 + scratch, capacity-4 profile
NUM_CACHE_PAGES = NUM_SLOTS * PAGES_PER_SLOT
LIVE_SEQ = 131072 + 128
DEVICE = torch.device("cuda")
torch.manual_seed(7)
torch.set_grad_enabled(False)


def time_arm(batch: int, iters: int = 50) -> tuple[float, object]:
    ws = PagedAttentionWorkspace.for_contract(
        mode="decode",
        device=DEVICE,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim_qk=HEAD_DIM,
        head_dim_vo=HEAD_DIM,
        page_size=PAGE_SIZE,
        max_total_q=batch,
        num_cache_pages=NUM_CACHE_PAGES,
        use_cuda_graph=True,
    )
    ws.prepare_decode_graph_replay_state(
        batch=batch,
        max_page_table_width=PAGES_PER_SLOT,
        total_q_capacity=batch,
        max_cache_page_count=PAGES_PER_SLOT,
        window_left=-1,
    )
    plan = ws._plan
    pt = (
        torch.arange(PAGES_PER_SLOT, dtype=torch.int32, device=DEVICE).unsqueeze(0).repeat(batch, 1)
    )
    cs = torch.full((batch,), LIVE_SEQ, dtype=torch.int32, device=DEVICE)
    cu = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE)
    ws._copy_runtime_metadata(pt, cs, cu)
    ws.update_decode_graph_replay_metadata_from_runtime_cache_seqlens()
    kc = (torch.randn(NUM_CACHE_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=DEVICE) * 0.5).to(
        torch.float8_e4m3fn
    )
    vc = (torch.randn(NUM_CACHE_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=DEVICE) * 0.5).to(
        torch.float8_e4m3fn
    )
    q = torch.randn(batch, 1, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    out = torch.empty_like(q)
    binding = build_paged_attention_binding(
        scratch=ws,
        q=q.reshape(batch, NUM_Q_HEADS, HEAD_DIM),
        k_cache=kc,
        v_cache=vc,
        output=out.reshape(batch, NUM_Q_HEADS, HEAD_DIM),
        k_descale=torch.ones(1, device=DEVICE),
        v_descale=torch.ones(1, device=DEVICE),
    )
    for _ in range(5):
        paged_attention_forward(binding=binding)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        paged_attention_forward(binding=binding)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters, plan


def main() -> None:
    for batch in (1, 2, 3, 4):
        ms, plan = time_arm(batch)
        print(
            f"decode-graph-replay b{batch} q=1 @{LIVE_SEQ}: {ms:.3f} ms/call "
            f"(split_kv={plan.split_kv}, kv_chunk_size={plan.kv_chunk_size}, "
            f"new_batch_size={plan.new_batch_size})"
        )


if __name__ == "__main__":
    main()
