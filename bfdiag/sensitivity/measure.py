"""One isolated DFlash measurement under one named perturbation.

Process isolation is the method, not an implementation detail: a second
generate() in the same process leaves allocator and engine state behind,
which is the variable under test. ``bf sensitivity sweep`` therefore
re-execs this module once per perturbation and pays a model load each
time. That is the price of a measurement you can trust.

GPU-only. Every torch/runtime import is deferred into function bodies so
importing this module (e.g. from the CLI, or from CPU unit tests that
only exercise perturbations/verdict) never needs torch.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

DEFAULT_MODEL = (
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/"
    "snapshots/07614121b31898586430f189d27a25a0be310843/"
)
DEFAULT_PHRASE = "The quick brown fox jumps over the lazy dog. "


@dataclass(frozen=True)
class LoadShape:
    """The load-time geometry, derived the one way the benchmarks derive it.

    Duplicating this formula is how a warm daemon ended up measuring
    ``blocks_per_slot=4096`` against a script's 130 and calling the two
    acceptance rates comparable. Derive, never default.
    """

    block_size: int
    ctx: int
    max_tokens: int
    max_model_len: int
    blocks_per_slot: int


def derive_shape(ctx: int, block_size: int, max_tokens: int = 256, margin: int = 4096) -> LoadShape:
    mml = ctx + max_tokens + 2048
    bps = -(-mml // block_size) + -(-margin // block_size)
    return LoadShape(block_size, ctx, max_tokens, mml, bps)


def allocator_snapshot() -> dict:
    import torch

    s = torch.cuda.memory_stats()
    return {
        "alloc_mib": round(torch.cuda.memory_allocated() / 2**20, 2),
        "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 2),
        "segments": s.get("segment.all.current"),
        "active_blocks": s.get("active.all.current"),
        "inactive_split_blocks": s.get("inactive_split.all.current"),
    }


def _prepare_env() -> None:
    os.environ.setdefault("USE_LIBUV", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("QSR_DFLASH_CUDA_GRAPH", "1")
    os.environ.setdefault("QSR_VERIFY_CUDA_GRAPH", "1")
    os.environ.setdefault("SPARKINFER_DYNAMIC_DETERMINISTIC_OUTPUT", "1")


def run_one(perturbation: str, shape: LoadShape, model: str = DEFAULT_MODEL) -> dict:
    """Load a fresh engine, apply ``perturbation``, run one generation."""
    _prepare_env()
    import torch
    from transformers import AutoTokenizer

    from bfdiag.sensitivity.perturbations import build
    from runtime.backends.laguna import LagunaBackend
    from runtime.backends.laguna_dflash import DFlashEngine
    from runtime.laguna_config import build_laguna_config

    torch.set_grad_enabled(False)
    model_path = os.path.expanduser(model)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    chunk = tok.encode(DEFAULT_PHRASE, add_special_tokens=False)
    prompt: list[int] = []
    while len(prompt) < shape.ctx:
        prompt.extend(chunk)
    prompt = prompt[: shape.ctx]

    backend = LagunaBackend(
        build_laguna_config(
            model_path,
            dtype="bfloat16",
            max_model_len=shape.max_model_len,
            gpu_memory_utilization=0.88,
            enforce_eager=True,
            trust_remote_code=True,
        ),
        num_slots=1,
        block_size=shape.block_size,
        blocks_per_slot=shape.blocks_per_slot,
    )
    engine = DFlashEngine(backend)

    holder: list = []
    build(perturbation, backend=backend, holder=holder)()
    before = allocator_snapshot()

    t0 = time.perf_counter()
    tokens, stats = engine.generate(prompt, max_tokens=shape.max_tokens)
    return {
        "perturbation": perturbation,
        "shape": asdict(shape),
        "acceptance_rate": stats["acceptance_rate"],
        "tok_per_s": round(stats["tok_per_s"], 3),
        "wall_s": round(time.perf_counter() - t0, 3),
        "output_hash": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "allocator": before,
        "verify_cg": engine._verify_cg is not None,
        "draft_cg": engine._draft_cg is not None,
    }


def _main(argv: list[str]) -> int:
    """``python -m bfdiag.sensitivity.measure <pert> <bs> <ctx> <out.json>``"""
    pert, bs, ctx, out = argv[0], int(argv[1]), int(argv[2]), argv[3]
    row = run_one(pert, derive_shape(ctx, bs))
    with open(out, "w") as f:
        json.dump(row, f, indent=2)
    print(
        f"RESULT {pert:>12} bs={bs} accept={row['acceptance_rate']:.6f} "
        f"out={row['output_hash'][:12]} reserved={row['allocator']['reserved_mib']}MiB",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
