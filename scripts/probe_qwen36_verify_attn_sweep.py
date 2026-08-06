"""Sweep the Qwen3.6 MTP verify attention over cta_tile_q and chunk budgets.

Production geometry: ``batch`` slots x 131072-token prefix, MTP K=3, verify
query ``[batch, anchor+3]`` against the pooled full-attention KV, FP8 e4m3
KV, page_size=128.  The CUDA-graph workspace is captured against the
worst-case 262144-token capacity but re-chunked adaptively at the live
length; this probe times the raw ``paged_attention_forward`` call for both
the M16 (legacy) and M32 (2026-08-06) verifier routes at the live 128K
geometry so the two can be compared head-to-head without a full server run.

Synthetic only -- no model load, no weight files.  Needs ~1.5 GB GPU.

Usage:
    /home/bot/.venvs/vllm/bin/python scripts/probe_qwen36_verify_attn_sweep.py

Env knobs:
    QSR_PROBE_CAPACITY_PAGES=2048   worst-case workspace capacity
    QSR_PROBE_LIVE_PAGES=1024       128K live prefix
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)

import sparkinfer.attention.paged.planner as planner  # noqa: E402
import torch  # noqa: E402
from sparkinfer.attention.paged._forward import paged_attention_forward  # noqa: E402
from sparkinfer.attention.paged._scratch import (  # noqa: E402
    build_paged_attention_binding,
)
from sparkinfer.attention.paged.workspace import (  # noqa: E402
    PagedAttentionWorkspace,
)

NUM_Q, NUM_KV, HD = 24, 4, 256
PAGE = 128
CAPACITY_PAGES = int(os.environ.get("QSR_PROBE_CAPACITY_PAGES", "2048"))
LIVE_PAGES = int(os.environ.get("QSR_PROBE_LIVE_PAGES", "1024"))
K = 3
DEV = torch.device("cuda")

_orig_cta_tile_q = planner._paged_determine_cta_tile_q  # noqa: SLF001


def _graph_ws(batch: int, qo: int, force_cta: int) -> PagedAttentionWorkspace:
    ws = PagedAttentionWorkspace.for_contract(
        mode="verify",
        device=DEV,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=NUM_Q,
        num_kv_heads=NUM_KV,
        head_dim_qk=HD,
        head_dim_vo=HD,
        page_size=PAGE,
        max_total_q=batch * qo,
        num_cache_pages=CAPACITY_PAGES,
        use_cuda_graph=True,
    )
    cu = torch.arange(batch + 1, dtype=torch.int32, device=DEV) * qo
    ws.prepare_prefill_graph_replay_state(
        batch=batch,
        total_q_capacity=batch * qo,
        max_page_table_width=CAPACITY_PAGES,
        max_cache_seqlen=CAPACITY_PAGES * PAGE,
        cu_seqlens_q=cu,
        window_left=-1,
    )
    return ws


def _build(ws, q, kc, vc, pt, cs, cu, out):
    ws.update_prefill_graph_replay_metadata(pt, cs, cu)
    return build_paged_attention_binding(
        scratch=ws,
        q=q,
        k_cache=kc,
        v_cache=vc,
        output=out,
        k_descale=torch.ones(1, device=DEV),
        v_descale=torch.ones(1, device=DEV),
    )


def _time_binding(b, iters=50):
    for _ in range(5):
        paged_attention_forward(binding=b)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        paged_attention_forward(binding=b)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def _chunk_override(count: int):
    """Return a planner max-chunks override pinned to ``count``."""

    def _override(**kwargs):
        return count

    return _override


def main() -> None:
    lock_ok = subprocess.run(
        ["/tmp/gpu_lock.sh", "acquire", "qwen-verify-sweep", "600"],
        capture_output=True,
        text=True,
    )
    print(lock_ok.stdout.strip() or lock_ok.stderr.strip(), flush=True)
    locked = lock_ok.returncode == 0
    try:
        torch.set_grad_enabled(False)
        kc = (torch.randn(CAPACITY_PAGES, PAGE, NUM_KV, HD, device=DEV) * 0.5).to(
            torch.float8_e4m3fn
        )
        vc = (torch.randn(CAPACITY_PAGES, PAGE, NUM_KV, HD, device=DEV) * 0.5).to(
            torch.float8_e4m3fn
        )
        results = {}
        for force_cta in (32, 16):
            planner._paged_determine_cta_tile_q = (  # noqa: SLF001
                lambda *args, _cta=force_cta, **kwargs: _cta
            )
            for batch in (1, 2, 3, 4):
                qo = K + 1
                q = torch.randn(batch * qo, NUM_Q, HD, dtype=torch.bfloat16, device=DEV) * 0.3
                pt = (
                    torch.arange(CAPACITY_PAGES, dtype=torch.int32, device=DEV)
                    .unsqueeze(0)
                    .repeat(batch, 1)
                )
                cs = torch.tensor([LIVE_PAGES * PAGE] * batch, dtype=torch.int32, device=DEV)
                cu = torch.tensor([qo * i for i in range(batch + 1)], dtype=torch.int32, device=DEV)
                out = torch.empty(batch * qo, NUM_Q, HD, dtype=torch.bfloat16, device=DEV)
                for chunks in (None, 11, 16, 23, 31, 47):
                    orig = planner._decode_graph_heuristic_max_chunks_per_req
                    if chunks is not None:
                        planner._decode_graph_heuristic_max_chunks_per_req = _chunk_override(chunks)
                    ws = _graph_ws(batch, qo, force_cta)
                    b = _build(ws, q, kc, vc, pt, cs, cu, out)
                    ms = _time_binding(b)
                    p = ws.plan
                    results[(batch, force_cta, chunks)] = (
                        ms,
                        p.cta_tile_q,
                        p.new_batch_size,
                        p.kv_chunk_size,
                    )
                    planner._decode_graph_heuristic_max_chunks_per_req = orig
                    print(
                        f"b{batch} cta={force_cta} chunks={chunks!s:>4} "
                        f"plan_cta={p.cta_tile_q:2d} items={p.new_batch_size:3d} "
                        f"kv_chunk={p.kv_chunk_size:5d} => {ms:.3f} ms/call",
                        flush=True,
                    )
            planner._paged_determine_cta_tile_q = _orig_cta_tile_q  # noqa: SLF001
        print("\nsummary (b4, live 128K):")
        for k, v in sorted(
            ((k, v) for k, v in results.items() if k[0] == 4),
            key=lambda kv: (kv[0][1], kv[1][0]),
        ):
            print(f"  cta={k[1]:2d} chunks={k[2]!s:>4} plan_cta={v[1]:2d} => {v[0]:.3f} ms/call")
    finally:
        planner._paged_determine_cta_tile_q = _orig_cta_tile_q  # noqa: SLF001
        if locked:
            subprocess.run(
                ["/tmp/gpu_lock.sh", "release", "qwen-verify-sweep"],
                capture_output=True,
            )


if __name__ == "__main__":
    main()
