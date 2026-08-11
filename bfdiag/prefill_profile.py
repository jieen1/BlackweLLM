"""Phase 0: DSV4 prefill per-layer profile + real route histogram.

Usage (warm daemon):  bf exec bfdiag/prefill_profile.py --tokens 64
Usage (direct):       python bfdiag/prefill_profile.py --gguf <path> --tokens 64

Outputs, per Phase 0 of docs/dsv4-prefill-2k-implementation-plan.md:
  - per-layer MoE / attention / HC GPU time (torch.profiler)
  - real route histogram (unique experts / routes-per-expert per layer)
  - a reproducible fixture: the exact token sequence and wall time

The same invocation must reproduce 131-133 tok/s (max_q_rows=64) with
profiler-split vs wall-time delta < 5% before later phases start.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time

import torch

DEFAULT_GGUF = (
    "/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/"
    "DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf"
)


def load_backend(gguf: str, max_q_rows: int) -> object:
    from runtime.backends.dsv4 import DeepseekV4Backend
    from runtime.model.dsv4_model import load_dsv4_from_gguf

    model, _ = load_dsv4_from_gguf(gguf, max_seq_len=65536, device="cuda")
    return DeepseekV4Backend(
        model, model.config, num_slots=2, max_seq_len=65536,
        max_q_rows=max_q_rows, device="cuda",
    )


def profile_prefill(backend: object, tokens: int, seed: int) -> dict:
    """Run one tokens-row prefill under torch.profiler, return split + hist."""
    from torch.profiler import ProfilerActivity, profile

    rng = random.Random(seed)
    ids = torch.tensor([[rng.randrange(100000) for _ in range(tokens)]],
                       dtype=torch.long, device="cuda")
    hist: dict = {}
    model = backend.model
    for lid, block in enumerate(model.blocks):
        orig = block.moe.gate.forward

        def make(lid: int, orig) -> object:
            def wrapped(x, input_ids):
                w, idx = orig(x, input_ids)
                hist[lid] = {
                    "topk": int(idx.shape[1]),
                    "unique": int(torch.unique(idx).numel()),
                    "routes": int(idx.numel()),
                }
                return w, idx
            return wrapped

        block.moe.gate.forward = make(lid, orig)
    backend._forward(0, ids, 0)  # warm (JIT, also populates route_hist)
    torch.cuda.synchronize()
    t0 = time.time()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        backend._forward(0, ids, 0)
    torch.cuda.synchronize()
    wall_ms = (time.time() - t0) * 1000

    bykey: dict[str, float] = collections.defaultdict(float)
    for e in prof.key_averages():
        bykey[e.key] += max(e.self_device_time_total, e.device_time_total) / 1000
    split = {
        "moe_dp4a_dual": sum(v for k, v in bykey.items()
                             if "dual_dp4a" in k or "dual_kernel" in k),
        "moe_dp4a_single": sum(v for k, v in bykey.items()
                               if "indexed_dp4a" in k or "indexed_kernel" in k),
        "attn_q8_tc": sum(v for k, v in bykey.items() if "q8_0_dequant_gemm_tc" in k),
        "hc_pre": bykey.get("_hc_pre_kernel", 0.0),
        "other": sum(v for k, v in bykey.items()
                     if not any(t in k for t in (
                         "dp4a", "indexed", "q8_0_dequant_gemm_tc", "_hc_pre_kernel"))),
    }
    split["total"] = sum(split.values())

    return {"wall_ms": wall_ms, "split": split, "fixture_seed": seed, "route_hist": hist}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--max-q-rows", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    backend = load_backend(args.gguf, args.max_q_rows)
    result = profile_prefill(backend, args.tokens, args.seed)
    print(json.dumps(result, indent=2))
    wall = result["wall_ms"]
    split = result["split"]
    total = split["total"]
    hist = result.get("route_hist", {})
    print(f"\n=== prefill_profile {args.tokens} tokens (max_q_rows={args.max_q_rows}) ===")
    print(f"wall: {wall:.1f} ms -> {args.tokens / (wall / 1000):.1f} tok/s")
    print(f"profiler total: {total:.1f} ms (delta vs wall {abs(total - wall) / wall * 100:.1f}%)")
    for k, v in split.items():
        if k != "total":
            print(f"  {k:22s} {v:8.1f} ms  {v / total * 100:5.1f}%")
    if hist:
        uniq = [h["unique"] for h in hist.values()]
        routes = [h["routes"] for h in hist.values()]
        rpe = [r / u for r, u in zip(routes, uniq)]
        print(f"\nroute histogram ({len(hist)} layers, topk={hist[0]['topk']}):")
        print(f"  unique experts: min {min(uniq)} max {max(uniq)} mean {sum(uniq)/len(uniq):.1f}")
        print(f"  routes/expert:  min {min(rpe):.2f} max {max(rpe):.2f} "
              f"mean {sum(rpe)/len(rpe):.2f}")


if __name__ == "__main__":
    main()
