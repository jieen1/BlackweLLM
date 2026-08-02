"""Probe: does a *novel* prompt length re-trigger a sparkinfer CuTe compile?

Runs ONE `Qwen36Attention`-shaped attention layer (real geometry: 24 Q heads /
4 KV heads / head_dim=256 / page_size=128 / BF16 KV) against a sweep of
mutually-distinct, never-before-seen prompt lengths, and reports for each:

* wall time of the `paged_attention_forward` call,
* whether sparkinfer's in-process compile cache took a MISS (the ground truth
  for "a new compile key appeared" -- wall time alone lies, because
  ``~/.cache/sparkinfer`` serves a warm on-disk hit in a few hundred ms and
  makes a genuine new key look cheap),
* the plan attributes that feed sparkinfer's compile cache key.

No model weights are loaded; this needs only a few hundred MiB of KV cache, so
it is cheap to re-run. Set ``SPARKINFER_COMPILE_DISK_CACHE=0`` to force every
new key to pay a real compile (the honest worst case, i.e. what a fresh machine
or a cache-invalidating sparkinfer upgrade sees).

Run with: ~/.venvs/vllm/bin/python scripts/b1_probe_extend_jit_buckets.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sparkinfer._lib.compiler import compile_cache_info  # noqa: E402
from sparkinfer.attention.paged.planner import create_paged_plan  # noqa: E402

from runtime.model.qwen36_model import (  # noqa: E402
    Qwen36AttentionWorkspace,
    Qwen36PagedAttentionCache,
)

NUM_Q_HEADS = 24
NUM_KV_HEADS = 4
HEAD_DIM = 256
MAX_SEQ_LEN = 4096

# Deliberately all-distinct, never-repeating lengths, spanning the range real
# agent prompts live in. 5 is first because it is the length the B1 smoke test
# happened to use first, so the "before" run reproduces its exact history.
LENGTHS = [5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584]


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    disk_cache = os.environ.get("SPARKINFER_COMPILE_DISK_CACHE", "1")
    print(f"SPARKINFER_COMPILE_DISK_CACHE={disk_cache}")
    print(f"lengths: {LENGTHS}")

    cache = Qwen36PagedAttentionCache(
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=MAX_SEQ_LEN,
        dtype=dtype,
        device=device,
    )
    workspaces: dict[str, Qwen36AttentionWorkspace] = {}

    def workspace_for(mode: str) -> Qwen36AttentionWorkspace:
        if mode not in workspaces:
            workspaces[mode] = Qwen36AttentionWorkspace(
                mode=mode,
                num_q_heads=NUM_Q_HEADS,
                num_kv_heads=NUM_KV_HEADS,
                head_dim=HEAD_DIM,
                page_size=cache.page_size,
                max_total_q=MAX_SEQ_LEN,
                max_page_table_width=cache.num_pages,
                num_cache_pages=cache.num_pages,
                dtype=dtype,
                kv_dtype=cache.dtype,
                device=device,
            )
        return workspaces[mode]

    header = (
        f"{'seq_len':>8} {'mode':>7} {'wall_s':>8} {'compile?':>9} "
        f"{'cta_tile_q':>11} {'split_kv':>9} {'new_batch':>10} {'pt_width':>9}"
    )
    print("\n" + header)
    print("-" * len(header))

    rows = []
    for seq_len in LENGTHS:
        # Each length is an independent "fresh prompt": reset the cache so
        # cache_seqlens == seq_len, exactly like a cold request.
        cache.seq_len = 0
        key = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
        value = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=dtype, device=device)
        cache.append(key, value)
        q = torch.randn(seq_len, NUM_Q_HEADS, HEAD_DIM, dtype=dtype, device=device)
        out = torch.empty_like(q)
        cache_seqlens = torch.tensor([seq_len], dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        mode = "decode" if seq_len == 1 else "extend"
        ws = workspace_for(mode)

        # Read-only: what does the eager planner decide for this shape?
        plan = create_paged_plan(
            q,
            cache.k_cache,
            cache.v_cache,
            cache.page_table,
            cache_seqlens,
            cu_seqlens_q,
            mode=mode,
            enable_cuda_graph=False,
            window_left=-1,
            plan_budget=getattr(ws, "_plan_budget", None),
        )

        before = compile_cache_info()
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        ws.forward(
            q=q,
            k_cache=cache.k_cache,
            v_cache=cache.v_cache,
            output=out,
            page_table=cache.page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        after = compile_cache_info()
        misses = int(after["memory_cache_misses"]) - int(before["memory_cache_misses"])
        disk_hits = int(after["disk_cache_hits"]) - int(before["disk_cache_hits"])
        tag = "-" if misses == 0 else (f"DISK x{misses}" if disk_hits else f"JIT x{misses}")
        print(
            f"{seq_len:>8} {mode:>7} {elapsed:>8.3f} {tag:>9} "
            f"{plan.cta_tile_q:>11} {str(bool(plan.split_kv)):>9} "
            f"{plan.new_batch_size:>10} {plan.page_table_shape[1]:>9}"
        )
        rows.append((seq_len, elapsed, misses))

    total_compiles = sum(1 for _, _, m in rows if m)
    slow = [(n, t) for n, t, _ in rows if t > 1.0]
    print(
        f"\nnovel lengths tried: {len(rows)}  "
        f"lengths that hit a new compile key: {total_compiles}  "
        f"lengths costing >1s: {len(slow)} {slow}"
    )


if __name__ == "__main__":
    main()
