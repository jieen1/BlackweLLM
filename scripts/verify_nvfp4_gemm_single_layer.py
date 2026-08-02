"""Single-MLP-block validation: Qwen36MLP's fused NVFP4 W4A16 kernel forward
path (``sparkinfer.moe._shared.kernels.w4a16.kernel.run_w4a16_moe``,
degenerate 1-expert/top-1 MoE) vs the legacy per-Linear BF16-dequant forward
path, on one *real* checkpoint layer's gate/up/down_proj -- real quantized
weights + block scales + global scales, not synthetic.

Not a pytest test (needs the GPU lock + a real checkpoint on disk) -- run
manually, one shot, under /tmp/gpu_lock.sh. Reports cosine similarity and
max abs error between the two forward paths across several M (decode-like
M=1 up to prefill-like M=512).

Unit granularity changed from the first attempt on this branch: that one
compared per-Linear (down_proj/gate_proj/up_proj individually) because
ModelOptNVFP4Linear.forward() itself ran the candidate GEMM. This attempt's
candidate kernel (run_w4a16_moe) is shaped like one full gated-MLP block
(FC1=w13 fused gate+up -> silu -> FC2=w2 down) in a single launch -- there
is no per-Linear granularity to compare at anymore, so this compares the
whole Qwen36MLP block's output instead. See runtime/model/qwen36_model.py's
Qwen36MLP docstring for why.

**Two checkpoint formats (2026-08-03 follow-up, ``work/std-model-fuse-
20260803``)**: originally hardcoded to nvidia's modelopt checkpoint's own
tensor suffixes (``weight``/``weight_scale``/``weight_scale_2``). unsloth's
``unsloth/Qwen3.6-27B-NVFP4`` (compressed-tensors mixed-precision format)
uses different suffixes (``weight_packed``/``weight_scale``/
``weight_global_scale``) AND a different global-scale convention (the
RECIPROCAL of modelopt's -- see ``runtime/model/compressed_tensors_linear.py``'s
``CompressedTensorsNVFP4Linear`` docstring). ``--model-path`` now selects the
checkpoint; the tensor-suffix set and Linear class are auto-detected from
that checkpoint's own ``quantization_config.quant_method`` via
``runtime.model.qwen36_model._quantized_layers_map_for_checkpoint`` (the same
classifier ``load_qwen36_model`` uses for a real load), not guessed -- so
this script never has to be told which format it's looking at.

*** MUST be run with PYTHONPATH pointing at this worktree -- see
``scripts/verify_nvfp4_gemm_full_model_gap.py``'s docstring for why.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _ROOT)
import runtime  # noqa: E402

assert runtime.__file__.startswith(_ROOT), (
    f"editable install shadowed the worktree: runtime.__file__={runtime.__file__} "
    f"-- rerun with PYTHONPATH={_ROOT}"
)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402

from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.loading.compressed_tensors import QUANT_ALGO_MP_NVFP4  # noqa: E402
from runtime.loading.modelopt import QUANT_ALGO_NVFP4  # noqa: E402
from runtime.model._weight_loading import default_weight_loader  # noqa: E402
from runtime.model.compressed_tensors_linear import CompressedTensorsNVFP4Linear  # noqa: E402
from runtime.model.modelopt_linear import ModelOptNVFP4Linear  # noqa: E402
from runtime.model.qwen36_model import (  # noqa: E402
    Qwen36MLP,
    _quantized_layers_map_for_checkpoint,
)
from runtime.model_loading import _build_qwen36_model_config  # noqa: E402

# Genuinely format-agnostic (see ``_detect_algo``/``_SUFFIXES_FOR_ALGO``
# above): auto-detects the checkpoint's quant format rather than assuming
# one, so this defaults to the standard checkpoint like every other
# non-format-pinned script this round, but ``--model-path`` (below) can
# still point it at nvidia's modelopt checkpoint -- e.g. to reproduce
# modelopt-specific numbers, or via ``modelopt_checkpoint_path()`` from
# ``runtime.checkpoints``.
DEFAULT_CKPT_GLOB = standard_checkpoint_path()
LAYER = 5
DEVICE = "cuda"
HIDDEN_ACT = "silu"

#: Tensor suffix set + owning Linear class per algo string this script
#: knows how to build a standalone MLP for -- keeps the format-specific
#: knowledge in one place rather than scattered `if` branches below.
_SUFFIXES_FOR_ALGO: dict[str, tuple[str, ...]] = {
    QUANT_ALGO_NVFP4: ("weight", "weight_scale", "weight_scale_2"),
    QUANT_ALGO_MP_NVFP4: ("weight_packed", "weight_scale", "weight_global_scale"),
}
_LINEAR_CLASS_FOR_ALGO: dict[str, type] = {
    QUANT_ALGO_NVFP4: ModelOptNVFP4Linear,
    QUANT_ALGO_MP_NVFP4: CompressedTensorsNVFP4Linear,
}
#: Which suffix holds the packed weight tensor (the one whose shape tells us
#: hidden_size/intermediate_size) -- differs by format.
_WEIGHT_SUFFIX_FOR_ALGO: dict[str, str] = {
    QUANT_ALGO_NVFP4: "weight",
    QUANT_ALGO_MP_NVFP4: "weight_packed",
}


def _find_ckpt(ckpt_path: str) -> Path:
    """``ckpt_path`` may be a HF ``.../snapshots`` directory (pick the first
    -- and normally only -- entry, the original default's own shape) OR a
    concrete snapshot directory itself (recognized by holding
    ``model.safetensors.index.json`` directly -- what ``--model-path``
    callers pass when pointing at a specific checkpoint, e.g. the other
    scripts' ``MODEL_PATH``/``--model-path`` convention)."""
    root = Path(ckpt_path)
    if (root / "model.safetensors.index.json").exists():
        return root
    snaps = sorted(root.iterdir())
    assert snaps, f"no snapshot under {root}"
    return snaps[0]


def _detect_algo(ckpt: Path, layer: int) -> str:
    """Classify this checkpoint's ``mlp.gate_proj`` at ``layer`` the same
    way a real ``load_qwen36_model`` call would -- reused rather than
    re-derived, so this script can never disagree with the real loader
    about which format a checkpoint is."""
    config = _build_qwen36_model_config(str(ckpt))
    quantized = _quantized_layers_map_for_checkpoint(config)
    algo = quantized.get(f"model.language_model.layers.{layer}.mlp.gate_proj")
    if algo not in _SUFFIXES_FOR_ALGO:
        raise ValueError(
            f"layer {layer}'s mlp.gate_proj classified as {algo!r}, which this "
            f"script does not know how to build a standalone MLP for (known: "
            f"{sorted(_SUFFIXES_FOR_ALGO)}) -- pick a different --layer (e.g. "
            "unsloth's layers 56-63 are FP8-channel, not NVFP4)."
        )
    return algo


def load_mlp_tensors(ckpt: Path, layer: int, algo: str) -> dict[str, dict[str, torch.Tensor]]:
    suffixes = _SUFFIXES_FOR_ALGO[algo]
    with open(ckpt / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp"
    projs = ("gate_proj", "up_proj", "down_proj")
    needed: dict[str, None] = {}
    for proj in projs:
        for suffix in suffixes:
            needed[f"{prefix}.{proj}.{suffix}"] = None
    shards = {weight_map[k] for k in needed}
    raw: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(ckpt / shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in needed:
                    raw[k] = f.get_tensor(k)
    assert set(raw) == set(needed), set(needed) - set(raw)
    out: dict[str, dict[str, torch.Tensor]] = {}
    for proj in projs:
        out[proj] = {suffix: raw[f"{prefix}.{proj}.{suffix}"] for suffix in suffixes}
    return out


def build_mlp(ckpt: Path, layer: int) -> tuple[Qwen36MLP, int, int]:
    algo = _detect_algo(ckpt, layer)
    linear_cls = _LINEAR_CLASS_FOR_ALGO[algo]
    weight_suffix = _WEIGHT_SUFFIX_FOR_ALGO[algo]
    print(f"  detected checkpoint algo={algo!r} -> {linear_cls.__name__}")

    tensors = load_mlp_tensors(ckpt, layer, algo)
    gate_out, gate_packed_in = tensors["gate_proj"][weight_suffix].shape
    hidden_size = gate_packed_in * 2
    intermediate_size = gate_out
    down_out, down_packed_in = tensors["down_proj"][weight_suffix].shape
    assert down_out == hidden_size, (down_out, hidden_size)
    assert down_packed_in * 2 == intermediate_size, (down_packed_in, intermediate_size)

    config = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "hidden_act": HIDDEN_ACT,
    }
    quantized = {
        f"model.language_model.layers.{layer}.mlp.{proj}": algo
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    mlp = Qwen36MLP(config, layer, quantized)
    assert mlp._nvfp4_fused, "checkpoint tensors did not classify as all-NVFP4"
    assert isinstance(mlp.gate_proj, linear_cls)

    for proj_name, proj_module in (
        ("gate_proj", mlp.gate_proj),
        ("up_proj", mlp.up_proj),
        ("down_proj", mlp.down_proj),
    ):
        t = tensors[proj_name]
        for suffix in _SUFFIXES_FOR_ALGO[algo]:
            # default_weight_loader, not a raw `.data.copy_()`: unsloth's
            # `weight_global_scale` checkpoint tensor is shape `[1]` but the
            # Parameter (matching modelopt's `weight_scale_2`) is shape
            # `()` -- default_weight_loader's scalar-numel special case is
            # exactly what real loading (`Qwen36ForCausalLMSelfBuilt.
            # load_weights`) relies on for this, so this diagnostic script
            # should too rather than reimplementing the reshape.
            default_weight_loader(getattr(proj_module, suffix), t[suffix].to(DEVICE))

    mlp = mlp.to(DEVICE)
    # run_case (below) calls legacy_forward (each submodule's own
    # ModelOptNVFP4Linear.forward()/_ensure_ready(), reading raw
    # .weight/.weight_scale/.weight_scale_2 directly) BEFORE mlp(x) on every
    # M in main()'s loop, all on this ONE mlp instance -- but the fused path
    # frees those raw Parameters by default the first time it runs (see
    # Qwen36MLP.__init__'s docstring on `_keep_raw_nvfp4_weights`), which
    # would break every M after the first. Opt out: this script's whole
    # point is comparing the two paths against each other repeatedly.
    mlp._keep_raw_nvfp4_weights = True
    return mlp, hidden_size, intermediate_size


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.reshape(-1).double()
    b64 = b.reshape(-1).double()
    return (a64 @ b64 / (a64.norm() * b64.norm() + 1e-30)).item()


def legacy_forward(mlp: Qwen36MLP, x: torch.Tensor) -> torch.Tensor:
    return mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))


def run_case(mlp: Qwen36MLP, hidden_size: int, m: int, seed: int) -> None:
    torch.manual_seed(seed)
    x = (torch.randn(m, hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.02).contiguous()

    ref = legacy_forward(mlp, x)
    # legacy_forward's own Linears cache BF16 dequants -- drop them so they
    # don't become a resident cache the fused path doesn't need.
    for sub in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        sub._weight_bf16 = None

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_iters = 20 if m <= 8 else 5
    for _ in range(n_iters):
        out = mlp(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iters

    max_abs_err = (out.double() - ref.double()).abs().max().item()
    cos = cosine(out, ref)
    ref_norm = ref.double().norm().item()
    rel_err = max_abs_err / (ref.double().abs().max().item() + 1e-30)
    print(
        f"  M={m:4d}  cosine={cos:.6f}  max_abs_err={max_abs_err:.6f}  "
        f"rel_to_max={rel_err:.6f}  ref_norm={ref_norm:.2f}  "
        f"latency={dt * 1e6:.1f}us/call"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default=DEFAULT_CKPT_GLOB)
    ap.add_argument("--layer", type=int, default=LAYER)
    args = ap.parse_args()

    ckpt = _find_ckpt(args.model_path)
    print(f"checkpoint: {ckpt}")
    print(f"=== layer {args.layer} MLP (fused gate/up/down_proj) ===")
    mlp, hidden_size, intermediate_size = build_mlp(ckpt, args.layer)
    print(f"  hidden_size={hidden_size} intermediate_size={intermediate_size}")
    for m in (1, 2, 8, 32, 128, 512):
        run_case(mlp, hidden_size, m, seed=1234 + m)
    del mlp
    torch.cuda.empty_cache()
    mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"\npeak allocated during this script: {mem:.3f} GiB")


if __name__ == "__main__":
    main()
