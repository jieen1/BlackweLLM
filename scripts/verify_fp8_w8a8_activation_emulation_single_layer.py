"""Single-Linear sanity check for the FP8 W8A8 pre-flight's activation
emulation (``runtime/model/compressed_tensors_linear.py::
emulate_fp8_activation_round_trip``, gated by ``QSR_EMULATE_FP8_ACTIVATION``):
does round-tripping the activation through per-token FP8 quantize/dequantize
before ``F.linear`` look like a plausible stand-in for a genuine W8A8 GEMM's
error, on real checkpoint weights -- and does it actually round-trip at all
(a no-op emulation would silently pass everything downstream)?

This is deliberately narrower than ``scripts/verify_fp8_tensor_gemm_single_layer.py``:
that script measures a REAL FP8xFP8 kernel (``sparkinfer.gemm.
tensor_fp8_linear``) against the MODELOPT (nvidia) checkpoint's *static
per-tensor* FP8 scheme (``config_groups.group_0``: ``dynamic: false`` for
both weights and activations, one scalar ``input_scale`` per module).
This script instead targets the STANDARD (unsloth) checkpoint's FP8-channel
Linears (:class:`~runtime.model.compressed_tensors_linear.
CompressedTensorsFP8ChannelLinear`), whose scheme is genuinely different --
verified directly against that checkpoint's own ``config.json``, 2026-08-03:
``config_groups.group_0.input_activations`` = ``{num_bits: 8, type: float,
strategy: "token", dynamic: true, symmetric: true}``, i.e. a per-TOKEN
DYNAMIC scale (no checkpoint-side ``input_scale`` tensor at all -- computed
at runtime, one scale per row), not modelopt's per-tensor static scale. The
two checkpoints' FP8 layers are not the same measurement and the modelopt
script's 0.9996 cosine number must not be read as already answering this
question -- that is exactly what this script exists to check for the
standard checkpoint's own scheme.

No FP8xFP8 kernel is built or called here (that is explicitly out of scope
for this pre-flight -- see the module docstring on
``runtime/model/compressed_tensors_linear.py``). Only the activation side is
perturbed (per-token FP8 round-trip); the weight side is dequantized exactly
from the checkpoint's real FP8 values, identical on both sides of this
script's comparison, and the GEMM itself still runs BF16xBF16 via the same
``F.linear`` today's production path already uses. This makes the measured
gap a LOWER bound on real W8A8 error (a real kernel's FP8xFP8 accumulation
order differs too) -- see
``runtime/model/compressed_tensors_linear.py::emulate_fp8_activation_round_trip``'s
docstring for why a lower bound is the right tool for a pre-flight negative
check.

Targets: one ``self_attn.q_proj`` (full-attention), one ``linear_attn.
in_proj_qkv`` (GDN), and one ``mlp.gate_proj`` from the layers-56-63 overlap
band where FP8 (not NVFP4) wins per-checkpoint -- the three distinct shapes
that make up the profiled 233 FP8-layer calls/decode-step
(``notes/2026-08-03-decode-kernel-profile.md``).

Run (under ``/tmp/gpu_lock.sh acquire``):
    PYTHONPATH=<this worktree> ~/.venvs/vllm/bin/python -u \\
        scripts/verify_fp8_w8a8_activation_emulation_single_layer.py

*** MUST be run with PYTHONPATH pointing at this worktree -- see
``scripts/verify_nvfp4_gemm_full_model_gap.py``'s docstring for why.
"""

from __future__ import annotations

import json
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
from safetensors import safe_open  # noqa: E402

from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.model.compressed_tensors_linear import (  # noqa: E402
    CompressedTensorsFP8ChannelLinear,
    emulate_fp8_activation_round_trip,
)

CKPT = Path(standard_checkpoint_path())
DEVICE = "cuda"

#: (checkpoint dotted prefix, printed label) -- one of each of the three
#: shapes that make up the 233 FP8-layer calls/step this pre-flight is
#: about (see module docstring).
TARGETS = (
    ("model.language_model.layers.3.self_attn.q_proj", "layer3 self_attn.q_proj"),
    ("model.language_model.layers.0.linear_attn.in_proj_qkv", "layer0 linear_attn.in_proj_qkv"),
    ("model.language_model.layers.60.mlp.gate_proj", "layer60 mlp.gate_proj (FP8 overlap band)"),
)


def load_linear(ckpt: Path, prefix: str) -> CompressedTensorsFP8ChannelLinear:
    with open(ckpt / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    needed = {f"{prefix}.{suf}": None for suf in ("weight", "weight_scale")}
    shards = {weight_map[k] for k in needed}
    raw: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(ckpt / shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in needed:
                    raw[k] = f.get_tensor(k)
    assert set(raw) == set(needed), set(needed) - set(raw)

    out_f, in_f = raw[f"{prefix}.weight"].shape
    lin = CompressedTensorsFP8ChannelLinear(in_f, out_f, bias=False).to(DEVICE)
    lin.weight.data.copy_(raw[f"{prefix}.weight"].to(DEVICE))
    lin.weight_scale.data.copy_(raw[f"{prefix}.weight_scale"].to(DEVICE))
    return lin


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.reshape(-1).double()
    b64 = b.reshape(-1).double()
    return (a64 @ b64 / (a64.norm() * b64.norm() + 1e-30)).item()


def run_case(lin: CompressedTensorsFP8ChannelLinear, in_features: int, m: int, seed: int) -> None:
    torch.manual_seed(seed)
    x = (torch.randn(m, in_features, device=DEVICE, dtype=torch.bfloat16) * 0.02).contiguous()

    lin._ensure_ready()
    ref = F.linear(x, lin._weight_bf16, lin.bias)

    x_rt = emulate_fp8_activation_round_trip(x)
    # A no-op emulation (e.g. a scale/dtype bug that silently returns `x`
    # unchanged) would make every downstream number below meaningless -- a
    # PASS for the wrong reason. Fail loud instead of reporting one.
    changed_frac = (x_rt != x).float().mean().item()
    activation_max_abs_change = (x_rt.float() - x.float()).abs().max().item()
    assert changed_frac > 0.5, (
        f"round-trip changed only {changed_frac:.4%} of activation elements for a real "
        f"BF16 activation (M={m}) -- this looks like a no-op, not a genuine FP8 "
        "quantize/dequantize; the emulation would be measuring nothing"
    )

    out = F.linear(x_rt, lin._weight_bf16, lin.bias)

    max_abs_err = (out.double() - ref.double()).abs().max().item()
    cos = cosine(out, ref)
    ref_max = ref.double().abs().max().item()
    rel_err = max_abs_err / (ref_max + 1e-30)
    print(
        f"  M={m:4d}  cosine={cos:.6f}  max_abs_err={max_abs_err:.6f}  "
        f"rel_to_max={rel_err:.6f}  ref_max={ref_max:.4f}  "
        f"activation_changed_frac={changed_frac:.4f}  "
        f"activation_max_abs_change={activation_max_abs_change:.6f}"
    )


def main() -> None:
    ckpt = CKPT
    print(f"checkpoint: {ckpt}")
    print("=== FP8 W8A8 pre-flight: per-token activation round-trip emulation vs today's path ===")
    for prefix, label in TARGETS:
        print(f"\n=== {label} ===")
        lin = load_linear(ckpt, prefix)
        print(f"  in_features={lin.input_size} out_features={lin.output_size}")
        for m in (1, 2, 8, 32, 128, 512):
            run_case(lin, lin.input_size, m, seed=1234 + m)
        del lin
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
