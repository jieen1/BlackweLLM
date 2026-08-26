"""FN0: does b12x fused MoE (NVFP4) accept the Qwen3.8-MoE family geometry?

Day-0 prep for Qwen3.8-Flash-Next (125B-A6B, dropping 2026-08-26 23:00 CST,
see notes/2026-08-26-qwen38-flash-next-day0-survey.md). Production has only
ever run this kernel at Laguna's geometry (256 experts / hidden 3072 /
intermediate 1024). The Flash-Next family geometry, inferred from the fetched
Qwen3.8-2.4T-A95B config.json, is 512 experts / top-10 / intermediate 2048.
This probe answers the longest-lead-time question BEFORE the weights land:
does plan/prepare/run accept num_experts=512 at decode and verify shapes, or
does sparkinfer need kernel work first?

This is a cold-start / load-time-shape question, so the warm engine is the
wrong tool (AGENTS.md warm-vs-cold table); standalone single-layer probe with
synthetic weights, following scripts/b3_probe_gdn_spec_rollback.py precedent
(one layer, no checkpoint, shared-card friendly: ~15 GB steady).

Claim values do not matter here (weights are random): the claim under test is
finite output + correct shape at M in {1, 4, 8, 32, 64}.

Run: ~/.venvs/vllm/bin/python scripts/fn0_probe_b12x_moe_e512.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__}"
)

import torch  # noqa: E402

from runtime.backends._sparkinfer_import import ensure_sparkinfer_path  # noqa: E402

# Must precede the b12x import, mirroring laguna_sparkinfer_moe.py: the
# deterministic route path and the FC2 dynamic down-scale are import-time
# kernel selections, not runtime options.
os.environ.setdefault("SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT", "1")
os.environ.setdefault("SPARKINFER_ENABLE_DYNAMIC_DOWN_SCALE", "1")
ensure_sparkinfer_path()

from b12x._lib.intrinsics import swizzle_block_scale  # noqa: E402
from b12x.moe.fused_moe._impl import (  # noqa: E402
    allocate_tp_moe_workspace_pool,
    b12x_moe_fp4,
    build_tp_moe_fp4_binding,
    plan_b12x_fp4_moe_weights,
    prepare_b12x_fp4_moe_weights,
)

NUM_EXPERTS = int(os.environ.get("FN0_EXPERTS", "512"))
TOP_K = 10
HIDDEN = int(os.environ.get("FN0_HIDDEN", "8192"))
INTERMEDIATE = int(os.environ.get("FN0_INTER", "2048"))
DEVICE = "cuda"


def _rand_fp8(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.rand(shape, device=DEVICE) + 0.5).to(torch.float8_e4m3fn)


def _synthetic_raw(e: int, hidden: int, inter: int) -> dict[str, torch.Tensor]:
    """Random tensors in the exact layout load_moe_layer_weights returns."""
    raw: dict[str, torch.Tensor] = {}
    for name in ("gate_w", "up_w"):
        raw[name] = torch.randint(0, 255, (e, inter, hidden // 2), dtype=torch.uint8, device=DEVICE)
    raw["down_w"] = torch.randint(0, 255, (e, hidden, inter // 2), dtype=torch.uint8, device=DEVICE)
    for name in ("gate_sf", "up_sf"):
        raw[name] = _rand_fp8((e, inter, hidden // 16))
    raw["down_sf"] = _rand_fp8((e, hidden, inter // 16))
    for name in ("gate_gs", "down_gs"):
        raw[name] = torch.rand(e, dtype=torch.float32, device=DEVICE) + 0.5
    return raw


def main() -> None:
    torch.manual_seed(0)
    dev = torch.device(DEVICE)
    free0, total = torch.cuda.mem_get_info()
    print(
        f"geometry: E={NUM_EXPERTS} top_k={TOP_K} hidden={HIDDEN} "
        f"intermediate={INTERMEDIATE}; GPU free {free0 / 2**30:.1f} GiB "
        f"of {total / 2**30:.1f} GiB"
    )

    t0 = time.time()
    # Mirror prepare_sparkinfer_layer's exact convention (swizzle-then-cat,
    # reciprocal alphas, realistic activation gscales); only geometry varies.
    raw = _synthetic_raw(NUM_EXPERTS, HIDDEN, INTERMEDIATE)
    gate_sf_sw = swizzle_block_scale(raw["gate_sf"].clone().contiguous())
    up_sf_sw = swizzle_block_scale(raw["up_sf"].clone().contiguous())
    down_sf_sw = swizzle_block_scale(raw["down_sf"].clone().contiguous())
    w13_fp4 = torch.cat([raw["up_w"], raw["gate_w"]], dim=1).contiguous()
    w13_sf = torch.cat([up_sf_sw, gate_sf_sw], dim=1).contiguous()
    w1_alpha = (1.0 / raw["gate_gs"]).float().contiguous()
    w2_alpha = (1.0 / raw["down_gs"]).float().contiguous()
    w2_fp4 = raw["down_w"].clone().contiguous()
    del raw
    torch.cuda.synchronize()
    print(f"synthetic weights built in {time.time() - t0:.1f}s")

    wplan = plan_b12x_fp4_moe_weights(
        quant_modes="nvfp4",
        source_format="modelopt_nvfp4",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        w13_layout="w13",
    )
    experts = prepare_b12x_fp4_moe_weights(
        plan=wplan,
        w1_global_scale=w1_alpha,
        w2_global_scale=w2_alpha,
        w1_fp4=w13_fp4,
        w1_blockscale=w13_sf,
        w2_fp4=w2_fp4,
        w2_blockscale=down_sf_sw,
        a1_gscale=torch.tensor(0.0005, dtype=torch.float32, device=dev),
        a2_gscale=torch.tensor(0.0013, dtype=torch.float32, device=dev),
        params_dtype=torch.bfloat16,
    )
    workspace = allocate_tp_moe_workspace_pool()
    torch.cuda.synchronize()
    free1, _ = torch.cuda.mem_get_info()
    print(
        f"plan+prepare ok in {time.time() - t0:.1f}s; VRAM used {(free0 - free1) / 2**30:.1f} GiB"
    )

    out = torch.empty(64, HIDDEN, dtype=torch.bfloat16, device=dev)
    for m in (1, 4, 8, 32, 64):
        a = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device=dev)
        topk_ids = torch.randint(0, NUM_EXPERTS, (m, TOP_K), dtype=torch.int32, device=dev)
        topk_weights = torch.softmax(torch.randn(m, TOP_K, dtype=torch.float32, device=dev), dim=-1)
        torch.cuda.synchronize()
        ts = time.time()
        binding = build_tp_moe_fp4_binding(
            scratch=workspace,
            a=a,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_mode="nvfp4",
            input_scales_static=True,
            output=out[:m],
        )
        r = b12x_moe_fp4(binding=binding)
        torch.cuda.synchronize()
        finite = torch.isfinite(r.float()).all().item()
        print(
            f"M={m:<3} shape={tuple(r.shape)} finite={finite} "
            f"absmax={r.float().abs().max().item():.4g} "
            f"{(time.time() - ts) * 1e3:.2f} ms"
        )
        assert tuple(r.shape) == (m, HIDDEN), r.shape
        assert finite, "non-finite output"

    print("FN0 PASS: b12x fused MoE accepts E=512 Flash-Next-family geometry")


if __name__ == "__main__":
    main()
