"""Probe: Laguna's prefill (``mode="extend"``) ``cta_tile_q`` buckets, the
compiles they cost, and whether collapsing them changes any arithmetic.

Same root cause as ``scripts/b1_probe_extend_jit_buckets.py`` found for
Qwen3.6, re-measured on Laguna's own geometry (48 Q / 8 KV heads,
``head_dim=128``, FP8 KV, ``page_size=64``, full-attention and the
``window_left=511`` SWA group): ``plan.cta_tile_q`` is part of sparkinfer's
compile cache key and the eager planner derives it from the LIVE query
length, so ``PagedAttentionWorkspace.for_fixed_capacity`` -- which pins every
*buffer* shape -- cannot pin it.

Three sections, all model-free (a few hundred MiB of dummy KV cache, no
27B checkpoint):

1. ``buckets``  -- which query lengths pay a compile, with and without
   ``PagedPlanBudget``. Run under ``SPARKINFER_COMPILE_DISK_CACHE=0`` so a
   new compile key costs a real compile instead of a warm on-disk replay.
2. ``exact``    -- bitwise A/B of the attention output at the pre-fix
   (live-shape) tile versus the post-fix (capacity) tile, plus both against
   an fp32 dense reference. This is the guard that the fix is arithmetic-free.
3. ``verify``   -- read-only planner check showing why ``mode="verify"``
   must NOT get the budget: its ``cta_tile_q`` selection is an exact-match
   specialisation on ``packed_qo_len == 48``, which a capacity-derived
   length would destroy.

Run with: ~/.venvs/vllm/bin/python scripts/laguna_probe_extend_jit_buckets.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sparkinfer._lib.compiler import compile_cache_info  # noqa: E402
from sparkinfer.attention.paged.planner import (  # noqa: E402
    PagedPlanBudget,
    create_paged_plan,
)

from runtime.backends.laguna_sparkinfer_attn import SparkinferPrefillWorkspace  # noqa: E402

# poolside/Laguna-S-2.1-NVFP4 config.json + LagunaBackend defaults.
NUM_Q_HEADS = 48
NUM_KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 64  # LagunaBackend(block_size=64), ServerEngine's default too
SLIDING_WINDOW = 512  # -> window_left = 511 for the SWA layer group
MAX_TOTAL_Q = 8192  # QSR_PREFILL_CHUNK default
NUM_PAGES = 512  # << blocks_per_slot(4096); page_table width is already pinned

FP8 = torch.float8_e4m3fn

# All distinct, deliberately dense around the tile boundaries the planner has
# for this geometry so the table shows exactly where a bucket flips.
LENGTHS = [2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 17, 29, 47, 83, 149, 251, 512, 1024, 2048]


def make_cache(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (NUM_PAGES, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
    k = torch.randn(shape, dtype=torch.bfloat16, device=device).to(FP8)
    v = torch.randn(shape, dtype=torch.bfloat16, device=device).to(FP8)
    return k, v


def make_workspace(device: torch.device, *, budget: bool) -> SparkinferPrefillWorkspace:
    ws = SparkinferPrefillWorkspace(
        device, max_total_q=MAX_TOTAL_Q, max_page_table_width=NUM_PAGES
    )
    if not budget:
        # Emulate the pre-fix planner call, which passed no budget at all.
        ws._extend_plan_budget = None
    return ws


def section_buckets(device: torch.device, *, budget: bool, window_left: int) -> None:
    disk = os.environ.get("SPARKINFER_COMPILE_DISK_CACHE", "1")
    print(
        f"\n=== buckets: budget={budget} window_left={window_left} "
        f"SPARKINFER_COMPILE_DISK_CACHE={disk} ==="
    )
    k_cache, v_cache = make_cache(device)
    page_table = torch.arange(NUM_PAGES, dtype=torch.int32, device=device).unsqueeze(0)
    ws = make_workspace(device, budget=budget)

    header = f"{'qo_len':>7} {'wall_s':>8} {'compile?':>9} {'cta_tile_q':>11} {'split_kv':>9}"
    print(header)
    print("-" * len(header))
    compiles = 0
    for qo_len in LENGTHS:
        q = torch.randn(qo_len, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        out = torch.empty_like(q)
        cache_seqlens = torch.tensor([qo_len], dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
        plan = create_paged_plan(
            q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
            mode="extend", enable_cuda_graph=False, window_left=window_left,
            plan_budget=getattr(ws, "_extend_plan_budget", None),
        )
        before = compile_cache_info()
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        ws.forward(
            q, k_cache, v_cache, out, page_table, cache_seqlens, cu_seqlens_q,
            window_left=window_left, mode="extend",
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        after = compile_cache_info()
        missed = int(after["memory_cache_misses"]) - int(before["memory_cache_misses"])
        compiles += bool(missed)
        print(
            f"{qo_len:>7} {elapsed:>8.3f} {('-' if not missed else f'JIT x{missed}'):>9} "
            f"{plan.cta_tile_q:>11} {str(bool(plan.split_kv)):>9}"
        )
    print(f"lengths tried: {len(LENGTHS)}   lengths that hit a new compile key: {compiles}")


def reference(q, k_cache, v_cache, page_table, kv_len, window_left):
    """fp32 dense causal (optionally sliding-window) attention over the pages."""
    pages = page_table[0, : (kv_len + PAGE_SIZE - 1) // PAGE_SIZE].tolist()
    k = torch.cat([k_cache[p].float() for p in pages], dim=0)[:kv_len]
    v = torch.cat([v_cache[p].float() for p in pages], dim=0)[:kv_len]
    rep = NUM_Q_HEADS // NUM_KV_HEADS
    k = k.repeat_interleave(rep, dim=1).transpose(0, 1)  # [H, kv, D]
    v = v.repeat_interleave(rep, dim=1).transpose(0, 1)
    qf = q.float().transpose(0, 1)  # [H, qo, D]
    qo = qf.shape[1]
    att = qf @ k.transpose(-1, -2) * (1.0 / math.sqrt(HEAD_DIM))
    q_pos = torch.arange(kv_len - qo, kv_len, device=q.device).unsqueeze(1)
    k_pos = torch.arange(kv_len, device=q.device).unsqueeze(0)
    allowed = k_pos <= q_pos
    if window_left >= 0:
        allowed &= k_pos >= q_pos - window_left
    att = att.masked_fill(~allowed, float("-inf"))
    return (att.softmax(-1) @ v).transpose(0, 1)


def section_exact(device: torch.device, *, window_left: int) -> int:
    print(f"\n=== exact: bitwise A/B, window_left={window_left} ===")
    k_cache, v_cache = make_cache(device)
    page_table = torch.arange(NUM_PAGES, dtype=torch.int32, device=device).unsqueeze(0)
    ws_no = make_workspace(device, budget=False)
    ws_yes = make_workspace(device, budget=True)

    header = (
        f"{'qo_len':>7} {'tile_pre':>9} {'tile_post':>10} {'bitexact':>9} "
        f"{'max|diff|':>11} {'cos_pre':>11} {'cos_post':>11}"
    )
    print(header)
    print("-" * len(header))
    bad = 0
    for qo_len in [2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 17, 47, 149, 512, 1024]:
        q = torch.randn(qo_len, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        cache_seqlens = torch.tensor([qo_len], dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
        tiles, outs = [], []
        for ws in (ws_no, ws_yes):
            budget = getattr(ws, "_extend_plan_budget", None)
            plan = create_paged_plan(
                q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q,
                mode="extend", enable_cuda_graph=False, window_left=window_left,
                plan_budget=budget,
            )
            tiles.append(int(plan.cta_tile_q))
            out = torch.empty_like(q)
            ws.forward(
                q, k_cache, v_cache, out, page_table, cache_seqlens, cu_seqlens_q,
                window_left=window_left, mode="extend",
            )
            outs.append(out)
        ref = reference(q, k_cache, v_cache, page_table, qo_len, window_left)
        cos = [
            float(torch.nn.functional.cosine_similarity(
                o.float().flatten(), ref.flatten(), dim=0).item())
            for o in outs
        ]
        exact = torch.equal(outs[0], outs[1])
        diff = float((outs[0].float() - outs[1].float()).abs().max().item())
        if not exact:
            bad += 1
        print(
            f"{qo_len:>7} {tiles[0]:>9} {tiles[1]:>10} {str(exact):>9} "
            f"{diff:>11.3e} {cos[0]:>11.8f} {cos[1]:>11.8f}"
        )
    print("RESULT:", "all bitwise identical" if bad == 0 else f"{bad} length(s) DIFFER")
    return bad


def _policy_probe(
    device, *, mode, num_q_heads, num_kv_heads, page_size, qo_len, window_left, max_total_q
):
    """Read-only: what cta_tile_q does the planner pick, with and without a budget?"""
    shape = (NUM_PAGES, page_size, num_kv_heads, HEAD_DIM)
    k_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device).to(FP8)
    v_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device).to(FP8)
    page_table = torch.arange(NUM_PAGES, dtype=torch.int32, device=device).unsqueeze(0)
    q = torch.zeros(qo_len, num_q_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    tiles = []
    for budget in (None, PagedPlanBudget(max_total_q=max_total_q, max_batch=1)):
        plan = create_paged_plan(
            q, k_cache, v_cache, page_table,
            torch.tensor([max(qo_len, 1024)], dtype=torch.int32, device=device),
            torch.tensor([0, qo_len], dtype=torch.int32, device=device),
            mode=mode, enable_cuda_graph=False, window_left=window_left,
            plan_budget=budget,
        )
        tiles.append(int(plan.cta_tile_q))
    return tiles


def section_verify(device: torch.device) -> None:
    """Read-only: the other modes/models sharing this code path, and what the
    budget would do to each. Only the first row is supposed to change."""
    print("\n=== policy: cta_tile_q without / with a plan budget ===")
    cases = [
        ("main extend (48q/8kv, page64, wl=-1, qo=64)",
         dict(mode="extend", num_q_heads=48, num_kv_heads=8, page_size=64,
              qo_len=64, window_left=-1, max_total_q=MAX_TOTAL_Q)),
        ("main extend, SWA group (wl=511, qo=64)",
         dict(mode="extend", num_q_heads=48, num_kv_heads=8, page_size=64,
              qo_len=64, window_left=SLIDING_WINDOW - 1, max_total_q=MAX_TOTAL_Q)),
        ("main extend, short prompt (qo=5)",
         dict(mode="extend", num_q_heads=48, num_kv_heads=8, page_size=64,
              qo_len=5, window_left=-1, max_total_q=MAX_TOTAL_Q)),
        ("DFlash draft extend (72q/8kv, page128, wl=511, qo=16, cap=16)",
         dict(mode="extend", num_q_heads=72, num_kv_heads=8, page_size=128,
              qo_len=16, window_left=511, max_total_q=16)),
        ("verify (48q/8kv, page64, qo=8)",
         dict(mode="verify", num_q_heads=48, num_kv_heads=8, page_size=64,
              qo_len=8, window_left=-1, max_total_q=MAX_TOTAL_Q)),
        ("verify (48q/8kv, page128, qo=8) -- MUST NOT get the budget",
         dict(mode="verify", num_q_heads=48, num_kv_heads=8, page_size=128,
              qo_len=8, window_left=-1, max_total_q=MAX_TOTAL_Q)),
        ("decode (48q/8kv, page64, qo=1)",
         dict(mode="decode", num_q_heads=48, num_kv_heads=8, page_size=64,
              qo_len=1, window_left=-1, max_total_q=MAX_TOTAL_Q)),
    ]
    print(f"{'case':<62} {'no budget':>10} {'budget':>8}")
    for label, kwargs in cases:
        no_b, with_b = _policy_probe(device, **kwargs)
        print(f"{label:<62} {no_b:>10} {with_b:>8}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section", choices=("buckets", "exact", "verify", "all"), default="all"
    )
    parser.add_argument(
        "--no-budget",
        action="store_true",
        help="buckets section only: emulate the pre-fix planner call",
    )
    parser.add_argument("--window-left", type=int, default=-1)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(0)
    rc = 0
    if args.section in ("buckets", "all"):
        section_buckets(device, budget=not args.no_budget, window_left=args.window_left)
    if args.section in ("exact", "all"):
        rc |= section_exact(device, window_left=args.window_left)
    if args.section in ("verify", "all"):
        section_verify(device)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
