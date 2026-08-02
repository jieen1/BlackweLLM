"""Single-MLP-block W4A4 numerical check: does routing the standard
checkpoint's (``unsloth/Qwen3.6-27B-NVFP4``) NVFP4 MLP through a genuine
block-scaled W4A4 GEMM (``sparkinfer.gemm.blockscaled.mm``, both operands
pre-quantized) match the current W4A16 fused path and a plain BF16
reference, on one *real* checkpoint layer's gate/up/down_proj?

This is deliberately narrower than ``scripts/verify_nvfp4_gemm_single_layer.py``:
this checkpoint's ``config_groups.group_1`` (see
``runtime/loading/compressed_tensors.py``'s module docstring) declares
``input_activations`` too (``num_bits=4, strategy=tensor_group,
group_size=16, dynamic="local"``) -- a genuine, checkpoint-declared W4A4
scheme, unlike nvidia's modelopt checkpoint (weight-only W4A16, no
``input_activations`` at all). Only :class:`CompressedTensorsNVFP4Linear`
carries the ``input_global_scale`` tensor this needs, so this script only
builds MLPs from that format -- run it against a modelopt checkpoint and it
fails loudly rather than silently comparing the wrong thing.

**The one open question this script exists to answer**: does
``input_global_scale`` follow the same "use directly, do not reciprocate"
convention as ``weight_global_scale`` (see
``CompressedTensorsNVFP4Linear.nvfp4_w4a4_components_for_fuse``'s docstring
for the algebraic derivation of the weight side)? Both conventions are
tried for both scales (4 combinations); only one should produce sane
(cosine near 1, small max-abs-err) output -- the other three are expected to
look like the ``"!!!!!!!!!!!!"`` bug this repo already hit once. This
script is diagnostic only -- it does not gate anything by itself, and it
does not run under CI or ``/tmp/ci-sim`` (needs the GPU lock + a real
checkpoint on disk, same as its W4A16 sibling).

Run (under ``/tmp/gpu_lock.sh acquire``):
    PYTHONPATH=<this worktree> ~/.venvs/vllm/bin/python -u \\
        scripts/verify_nvfp4_w4a4_gemm_single_layer.py [--layer N]

*** MUST be run with PYTHONPATH pointing at this worktree -- see
``scripts/verify_nvfp4_gemm_full_model_gap.py``'s docstring for why.
"""

from __future__ import annotations

import argparse
import itertools
import sys
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

from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.model.compressed_tensors_linear import CompressedTensorsNVFP4Linear  # noqa: E402

sys.path.insert(0, str(Path(_ROOT) / "scripts"))
from verify_nvfp4_gemm_single_layer import (  # noqa: E402
    _find_ckpt,
    build_mlp,
    cosine,
    legacy_forward,
)

DEVICE = "cuda"
LAYER = 5


def load_input_global_scales(ckpt: Path, layer: int) -> dict[str, torch.Tensor]:
    """Read ``input_global_scale`` straight off the safetensors shards for
    one layer's gate/up/down_proj -- the same "read the real checkpoint
    tensor, don't guess" discipline as ``load_mlp_tensors`` in the W4A16
    sibling script, just for the one extra suffix that script's
    ``_SUFFIXES_FOR_ALGO`` does not know about (that script is genuinely
    format-agnostic across modelopt/compressed-tensors; this one is
    compressed-tensors-only, so it does not need to be)."""
    import json

    from safetensors import safe_open

    with open(ckpt / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp"
    projs = ("gate_proj", "up_proj", "down_proj")
    needed = {f"{prefix}.{proj}.input_global_scale": proj for proj in projs}
    shards = {weight_map[k] for k in needed}
    out: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(ckpt / shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in needed:
                    out[needed[k]] = f.get_tensor(k)
    assert set(out) == set(projs), set(needed.values()) - set(out)
    return out


def quantize_activation_nvfp4(x_bf16: torch.Tensor, global_scale: torch.Tensor):
    """Dynamically quantize a ``[M, K]`` BF16 activation to NVFP4 using the
    same pure-Torch grouped quantizer ``blockscaled.mm``'s own oracle tests
    build operands with (``sparkinfer._lib.intrinsics.
    quantize_grouped_nvfp4_torch``) -- num_groups=1 (this is a plain dense
    GEMM, not a grouped/MoE one)."""
    from sparkinfer._lib.intrinsics import quantize_grouped_nvfp4_torch

    m = x_bf16.shape[0]
    row_counts = torch.full((1,), m, dtype=torch.int32, device=x_bf16.device)
    packed, scale_view = quantize_grouped_nvfp4_torch(
        x_bf16.unsqueeze(0), row_counts, global_scale.reshape(1)
    )
    return packed, scale_view


def prepare_weight_operand(
    weight_packed: torch.Tensor, weight_scale: torch.Tensor, out_dim: int, in_dim: int
):
    """``weight_packed``/``weight_scale`` are the checkpoint's own raw
    tensors (unswizzled, no group axis). Reshape/swizzle into exactly what
    ``blockscaled.mm`` expects for a num_groups=1 dense operand -- the same
    transform ``quantize_grouped_nvfp4_torch`` applies to a freshly
    quantized weight, just applied to an already-quantized one."""
    from sparkinfer._lib.intrinsics import as_grouped_scale_view, swizzle_block_scale

    b_packed = weight_packed.unsqueeze(-1).contiguous()
    swizzled = swizzle_block_scale(weight_scale.unsqueeze(0).contiguous())
    b_sf = as_grouped_scale_view(swizzled.view(torch.uint8), out_dim, in_dim)
    return b_packed, b_sf


def blockscaled_linear(
    x_bf16: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_gs: torch.Tensor,
    input_gs: torch.Tensor,
    out_dim: int,
    in_dim: int,
) -> torch.Tensor:
    from sparkinfer.gemm import blockscaled

    a_packed, a_sf = quantize_activation_nvfp4(x_bf16, input_gs)
    b_packed, b_sf = prepare_weight_operand(weight_packed, weight_scale, out_dim, in_dim)
    alpha = (1.0 / (input_gs.to(torch.float32) * weight_gs.to(torch.float32))).reshape(1)
    out = blockscaled.mm(
        (a_packed, a_sf),
        (b_packed, b_sf),
        alpha=alpha,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16",
        sf_vec_size=16,
    )
    return out[:, :, 0]


def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.double() - b.double()).abs().max().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default=standard_checkpoint_path())
    ap.add_argument("--layer", type=int, default=LAYER)
    args = ap.parse_args()

    ckpt = _find_ckpt(args.model_path)
    print(f"checkpoint: {ckpt}")
    print(f"=== layer {args.layer} MLP (W4A4 blockscaled vs W4A16 fused vs BF16 reference) ===")

    mlp, hidden_size, intermediate_size = build_mlp(ckpt, args.layer)
    if not isinstance(mlp.gate_proj, CompressedTensorsNVFP4Linear):
        raise SystemExit(
            f"layer {args.layer}'s gate_proj is {type(mlp.gate_proj).__name__}, not "
            "CompressedTensorsNVFP4Linear -- this script only supports the standard "
            "(unsloth) checkpoint's genuine W4A4 scheme. Pick a checkpoint via "
            "--model-path or a different --layer."
        )
    print(f"  hidden_size={hidden_size} intermediate_size={intermediate_size}")

    igs_cpu = load_input_global_scales(ckpt, args.layer)
    igs = {k: v.to(DEVICE) for k, v in igs_cpu.items()}
    for proj, t in igs_cpu.items():
        print(f"  {proj}.input_global_scale = {t.item():.6g}")
    linears = (
        ("gate_proj", mlp.gate_proj),
        ("up_proj", mlp.up_proj),
        ("down_proj", mlp.down_proj),
    )
    for proj, lin in linears:
        print(f"  {proj}.weight_global_scale = {lin.weight_global_scale.data.item():.6g}")

    conventions = {
        "direct": lambda t: t,
        "reciprocal": lambda t: 1.0 / t,
    }

    for w_conv_name, a_conv_name in itertools.product(conventions, conventions):
        w_conv = conventions[w_conv_name]
        a_conv = conventions[a_conv_name]

        def igs_for(proj: str) -> torch.Tensor:
            return a_conv(igs[proj])

        def components_with_convention(lin):
            packed, scale, gs, _ = lin.nvfp4_w4a4_components_for_fuse()
            return packed, scale, w_conv(gs)

        print(f"\n-- weight_gs={w_conv_name}  input_gs={a_conv_name} --")
        for m in (1, 8, 128):
            torch.manual_seed(1234 + m)
            x = (
                torch.randn(m, hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.02
            ).contiguous()

            ref = legacy_forward(mlp, x)
            for sub in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
                sub._weight_bf16 = None

            gate_w, gate_scale, gate_gs = components_with_convention(mlp.gate_proj)
            up_w, up_scale, up_gs = components_with_convention(mlp.up_proj)
            down_w, down_scale, down_gs = components_with_convention(mlp.down_proj)

            gate_out = blockscaled_linear(
                x, gate_w, gate_scale, gate_gs, igs_for("gate_proj"), intermediate_size, hidden_size
            )
            up_out = blockscaled_linear(
                x, up_w, up_scale, up_gs, igs_for("up_proj"), intermediate_size, hidden_size
            )
            inter = (F.silu(gate_out.float()) * up_out.float()).to(torch.bfloat16)
            down_out = blockscaled_linear(
                inter,
                down_w,
                down_scale,
                down_gs,
                igs_for("down_proj"),
                hidden_size,
                intermediate_size,
            )

            cos = cosine(down_out, ref)
            err = max_abs_err(down_out, ref)
            ref_max = ref.double().abs().max().item()
            print(
                f"  M={m:4d}  cosine={cos:.6f}  max_abs_err={err:.6f}  "
                f"rel_to_max={err / (ref_max + 1e-30):.6f}  ref_max={ref_max:.4f}  "
                f"out_max={down_out.double().abs().max().item():.4f}"
            )

    print("\n=== also compare current W4A16 fused path against the same BF16 reference ===")
    for m in (1, 8, 128):
        torch.manual_seed(1234 + m)
        x = (torch.randn(m, hidden_size, device=DEVICE, dtype=torch.bfloat16) * 0.02).contiguous()
        ref = legacy_forward(mlp, x)
        for sub in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
            sub._weight_bf16 = None
        w4a16_out = mlp(x)
        cos = cosine(w4a16_out, ref)
        err = max_abs_err(w4a16_out, ref)
        print(f"  M={m:4d}  cosine={cos:.6f}  max_abs_err={err:.6f}")


if __name__ == "__main__":
    main()
