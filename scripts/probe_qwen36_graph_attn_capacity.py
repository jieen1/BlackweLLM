#!/usr/bin/env python3
"""Probe v2: what chunking do the PRODUCTION graph-mode capacity planners
choose for Qwen3.6 MTP verify/draft attention at 128K, and how long do
those exact kernel configurations take -- measured, not assumed.

Replicates Qwen36VerifyGraphAttention (for_contract +
prepare_prefill_graph_replay_state) and Qwen36DecodeGraphAttention
(plan_decode_graph_capacity) exactly, then times them at real 128K
geometry. This is the ground truth the optimization decision needs:
whether the captured graphs already split KV, and how much headroom a
different chunk count would buy.

Run: ~/.venvs/vllm/bin/python scripts/probe_qwen36_graph_attn_capacity.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _ROOT)

import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__}"
)

import sys  # noqa: E402

import torch  # noqa: E402

SCATTER = "--scatter" in sys.argv

assert torch.cuda.is_available(), "probe requires the GPU"

from sparkinfer.attention.paged._forward import paged_attention_forward  # noqa: E402
from sparkinfer.attention.paged._scratch import build_paged_attention_binding  # noqa: E402
from sparkinfer.attention.paged.planner import (  # noqa: E402
    plan_decode_graph_capacity,
    plan_verify_graph_capacity,
)
from sparkinfer.attention.paged.workspace import PagedAttentionWorkspace  # noqa: E402

NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
PAGE_SIZE = 128
NUM_PAGES = 1024  # 128K tokens
MAX_KV = NUM_PAGES * PAGE_SIZE
K = 3
DEVICE = torch.device("cuda")
LOCK_NAME = "graph-attn-capacity-probe"
torch.manual_seed(7)
torch.set_grad_enabled(False)

WORST_PAGES = NUM_PAGES
for _i, _a in enumerate(sys.argv):
    if _a == "--worst" and _i + 1 < len(sys.argv):
        WORST_PAGES = int(sys.argv[_i + 1])


def main() -> None:
    r = subprocess.run(
        ["/tmp/gpu_lock.sh", "acquire", LOCK_NAME, "120"], capture_output=True, text=True
    )
    print(r.stdout.strip() or r.stderr.strip())
    locked = r.returncode == 0
    try:
        kc = (torch.randn(WORST_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=DEVICE) * 0.5).to(
            torch.float8_e4m3fn
        )
        vc = (torch.randn(WORST_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, device=DEVICE) * 0.5).to(
            torch.float8_e4m3fn
        )

        print("\n--- plan_verify_graph_capacity (MTP verify, query_len=K+1) ---")
        for batch in (1, 2, 3, 4, 5):
            cap = plan_verify_graph_capacity(
                device=DEVICE,
                q_dtype=torch.bfloat16,
                kv_dtype=torch.float8_e4m3fn,
                num_q_heads=NUM_Q_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                head_dim_qk=HEAD_DIM,
                head_dim_vo=HEAD_DIM,
                page_size=PAGE_SIZE,
                batch=batch,
                query_len=K + 1,
                max_cache_page_count=NUM_PAGES,
            )
            chunks = (MAX_KV + cap.kv_chunk_size_pages * PAGE_SIZE - 1) // (
                cap.kv_chunk_size_pages * PAGE_SIZE
            )
            print(
                f"  b{batch}: work_items={cap.max_work_items} "
                f"partial_rows={cap.max_partial_rows} "
                f"kv_chunk_pages={cap.kv_chunk_size_pages} (~{chunks} chunks/req) "
                f"cta_tile_q={cap.cta_tile_q} ctas_per_sm={cap.graph_ctas_per_sm}"
            )

        print("\n--- plan_decode_graph_capacity (draft/decode, q=1) ---")
        for batch in (1, 2, 3):
            cap = plan_decode_graph_capacity(
                device=DEVICE,
                q_dtype=torch.bfloat16,
                kv_dtype=torch.float8_e4m3fn,
                num_q_heads=NUM_Q_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                head_dim_qk=HEAD_DIM,
                head_dim_vo=HEAD_DIM,
                page_size=PAGE_SIZE,
                batch=batch,
                max_cache_page_count=NUM_PAGES,
                window_left=-1,
            )
            print(
                f"  b{batch}: work_items={cap.max_work_items} "
                f"partial_rows={cap.max_partial_rows} "
                f"chunk_pages_lut[:4]={list(cap.chunk_pages_lut[:4])} "
                f"lut[-1]={cap.chunk_pages_lut[-1]} (pages -> "
                f"{cap.chunk_pages_lut[-1] * PAGE_SIZE} tok/chunk at 128K => "
                f"~{(NUM_PAGES + cap.chunk_pages_lut[-1] - 1) // cap.chunk_pages_lut[-1]} chunks)"
            )

        print(
            f"\n--- timed: verify graph workspace @128K "
            f"(capture capacity={WORST_PAGES * PAGE_SIZE} tokens) ---"
        )
        for batch in (1, 3, 4):
            ws = PagedAttentionWorkspace.for_contract(
                mode="verify",
                device=DEVICE,
                dtype=torch.bfloat16,
                kv_dtype=torch.float8_e4m3fn,
                num_q_heads=NUM_Q_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                head_dim_qk=HEAD_DIM,
                head_dim_vo=HEAD_DIM,
                page_size=PAGE_SIZE,
                max_total_q=batch * (K + 1),
                num_cache_pages=WORST_PAGES,
                use_cuda_graph=True,
            )
            cu = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE) * (K + 1)
            ws.prepare_prefill_graph_replay_state(
                batch=batch,
                total_q_capacity=batch * (K + 1),
                max_page_table_width=WORST_PAGES,
                max_cache_seqlen=WORST_PAGES * PAGE_SIZE,
                cu_seqlens_q=cu,
                window_left=-1,
            )
            base = torch.arange(NUM_PAGES, dtype=torch.int32, device=DEVICE)
            if SCATTER:
                perm = torch.randperm(NUM_PAGES, dtype=torch.int32, device=DEVICE)
                pt = perm.unsqueeze(0).repeat(batch, 1).contiguous()
            else:
                pt = base.unsqueeze(0).repeat(batch, 1)
            cs = torch.tensor([MAX_KV] * batch, dtype=torch.int32, device=DEVICE)
            ws.update_prefill_graph_replay_metadata(pt, cs, cu)
            q = torch.randn(
                batch * (K + 1), NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
            )
            out = torch.empty_like(q)
            binding = build_paged_attention_binding(
                scratch=ws,
                q=q,
                k_cache=kc,
                v_cache=vc,
                output=out,
                k_descale=torch.ones(1, device=DEVICE),
                v_descale=torch.ones(1, device=DEVICE),
            )
            for _ in range(5):
                paged_attention_forward(binding=binding)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(50):
                paged_attention_forward(binding=binding)
            e.record()
            torch.cuda.synchronize()
            print(f"  verify b{batch} q={K + 1} @128K: {s.elapsed_time(e) / 50:.3f} ms/call")

        print("\n--- timed: production-style decode graph workspace @128K (draft) ---")
        for batch in (1, 3):
            cap = plan_decode_graph_capacity(
                device=DEVICE,
                q_dtype=torch.bfloat16,
                kv_dtype=torch.float8_e4m3fn,
                num_q_heads=NUM_Q_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                head_dim_qk=HEAD_DIM,
                head_dim_vo=HEAD_DIM,
                page_size=PAGE_SIZE,
                batch=batch,
                max_cache_page_count=NUM_PAGES,
                window_left=-1,
            )
            ws = PagedAttentionWorkspace.for_fixed_capacity(
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
                max_batch=batch,
                max_page_table_width=NUM_PAGES,
                max_work_items=cap.max_work_items,
                max_partial_rows=cap.max_partial_rows,
                num_cache_pages=NUM_PAGES,
                use_cuda_graph=False,
            )
            pt = (
                torch.arange(NUM_PAGES, dtype=torch.int32, device=DEVICE)
                .unsqueeze(0)
                .repeat(batch, 1)
            )
            cs = torch.tensor([MAX_KV] * batch, dtype=torch.int32, device=DEVICE)
            cu = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE)
            q = torch.randn(batch, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
            out = torch.empty_like(q)
            from sparkinfer.attention.paged.planner import create_paged_plan

            plan = create_paged_plan(
                q,
                kc,
                vc,
                pt,
                cs,
                cu,
                mode="decode",
                enable_cuda_graph=True,
                graph_chunk_policy=True,
                max_batch_size_if_split=cap.max_work_items,
                window_left=-1,
            )
            ws._ensure_capacity(plan)
            ws._copy_runtime_metadata(pt, cs, cu)
            ws._copy_plan_metadata(plan)
            ws._plan = plan
            binding = build_paged_attention_binding(
                scratch=ws,
                q=q,
                k_cache=kc,
                v_cache=vc,
                output=out,
                k_descale=torch.ones(1, device=DEVICE),
                v_descale=torch.ones(1, device=DEVICE),
            )
            for _ in range(5):
                paged_attention_forward(binding=binding)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(50):
                paged_attention_forward(binding=binding)
            e.record()
            torch.cuda.synchronize()
            print(
                f"  decode b{batch} q=1 @128K: {s.elapsed_time(e) / 50:.3f} ms/call "
                f"(plan split_kv={plan.split_kv} items={plan.new_batch_size})"
            )
    finally:
        if locked:
            subprocess.run(["/tmp/gpu_lock.sh", "release", LOCK_NAME], capture_output=True)


if __name__ == "__main__":
    main()
