"""Single-Linear validation: does routing ``ModelOptFP8Linear`` through
sparkinfer's real static per-tensor FP8xFP8 GEMM
(``sparkinfer.gemm.tensor_fp8_linear``) stay numerically close to the
current B1 behavior (dequantize weight to BF16 once, cache forever, run
plain BF16xBF16 ``F.linear``, activations never quantized)?

Why this kernel and not another (2026-08-03, FP8 follow-up to the
NVFP4-GEMM round): checked the real checkpoint's own declared scheme
first, not assumed -- ``config_groups.group_0`` (every
``self_attn.{q,k,v,o}_proj`` / ``linear_attn.{in_proj_qkv,in_proj_z,
out_proj}``) declares BOTH ``weights`` AND ``input_activations`` as
8-bit float, ``dynamic: false`` (static scales), and the real safetensors
headers confirm every one of those layers ships an actual ``input_scale``
tensor (scalar F32) alongside ``weight``/``weight_scale`` -- this is a
genuine calibrated W8A8 checkpoint, unlike the NVFP4 MLP's W4A16
(weight-only) scheme that made ``sparkinfer.gemm.blockscaled.mm`` the
wrong tool on the previous round (that kernel needs both operands
quantized; this checkpoint's NVFP4 layers only declare a weight scale, no
input_scale-bearing scheme -- see ``runtime/loading/modelopt.py``'s module
docstring). ``sparkinfer.gemm.tensor_fp8_linear`` is a "Static per-tensor
FP8 linear for SM12x" (its own module docstring) -- weight scale and
input scale both single scalars, both static -- an exact match for what
this checkpoint's FP8 layers actually are, not an approximation forced
onto a weight-only checkpoint.

**This is NOT the same situation as the NVFP4 fusion, precision-wise,
and this script exists specifically to measure that gap rather than
assume it away**: NVFP4's fused w4a16 kernel dequantizes the weight
*inside* the kernel against an un-quantized BF16 activation -- exactly
what B1-R's calibration reference (HF's BF16-throughout forward, weights
copied from this runtime's own dequantized tensors) also does, so there
was no new source of error, just a different reduction order. Here, using
``tensor_fp8_linear`` genuinely quantizes the ACTIVATION to FP8 too
(using the checkpoint's own static ``input_scale`` -- the checkpoint's
intended execution path, not an invented one) -- something B1-R's
calibration baseline and B1's current FP8 implementation both never do.
That is real additional lossiness relative to what B1-R was calibrated
against, even though it is what the checkpoint format was designed for.
Whether it stays inside B1-R's calibrated gap-error bars is an empirical
question this script (plus the full-model gap script, if this passes) is
here to answer, not to assume from "the scheme matches" alone.

Not a pytest test (needs the GPU lock + a real checkpoint on disk) -- run
manually, one shot, under /tmp/gpu_lock.sh. Reports cosine similarity and
max abs error between the two forward paths across several M (decode-like
M=1 up to prefill-like M=512), for one real ``self_attn.q_proj`` (layer 3)
and one real ``linear_attn.in_proj_qkv`` (layer 0) -- two different real
shapes, matching ``scripts/b3_probe_batching_bar.py``'s choice of layers
for the same two module types.

*** MUST be run with PYTHONPATH pointing at this worktree -- see
``scripts/verify_nvfp4_gemm_full_model_gap.py``'s docstring for why.
"""

from __future__ import annotations

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
from safetensors import safe_open  # noqa: E402

from runtime.backends._sparkinfer_import import ensure_sparkinfer_path  # noqa: E402
from runtime.checkpoints import modelopt_checkpoint_path  # noqa: E402
from runtime.model.modelopt_linear import ModelOptFP8Linear  # noqa: E402

ensure_sparkinfer_path()
from sparkinfer.gemm import tensor_fp8_linear  # noqa: E402

# Deliberately modelopt (nvidia), not the standard checkpoint: this script
# imports ``ModelOptFP8Linear`` directly and reads the checkpoint's own
# ``config_groups`` scheme to confirm this format's static per-tensor FP8
# scales (see the module docstring above) -- both are modelopt-specific;
# the standard checkpoint's FP8 layers use a different, per-channel scale
# layout (see ``runtime/model/compressed_tensors_linear.py``'s module
# docstring) that ``ModelOptFP8Linear`` does not know how to read.
CKPT = Path(modelopt_checkpoint_path())
DEVICE = "cuda"

#: (checkpoint dotted prefix, layer index used only for the printed label)
TARGETS = (
    ("model.language_model.layers.3.self_attn.q_proj", "layer3 self_attn.q_proj"),
    ("model.language_model.layers.0.linear_attn.in_proj_qkv", "layer0 linear_attn.in_proj_qkv"),
)


def _find_ckpt() -> Path:
    """``modelopt_checkpoint_path()`` already resolved the one real snapshot
    directory (or raised a clear error) -- nothing left to glob for here."""
    return CKPT


def load_linear(ckpt: Path, prefix: str) -> ModelOptFP8Linear:
    with open(ckpt / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    needed = {f"{prefix}.{suf}": None for suf in ("weight", "weight_scale", "input_scale")}
    shards = {weight_map[k] for k in needed}
    raw: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(ckpt / shard), framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in needed:
                    raw[k] = f.get_tensor(k)
    assert set(raw) == set(needed), set(needed) - set(raw)

    out_f, in_f = raw[f"{prefix}.weight"].shape
    lin = ModelOptFP8Linear(in_f, out_f, bias=False).to(DEVICE)
    lin.weight.data.copy_(raw[f"{prefix}.weight"].to(DEVICE))
    lin.weight_scale.data.copy_(raw[f"{prefix}.weight_scale"].to(torch.float32).to(DEVICE))
    lin.input_scale.data.copy_(raw[f"{prefix}.input_scale"].to(torch.float32).to(DEVICE))
    return lin


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.reshape(-1).double()
    b64 = b.reshape(-1).double()
    return (a64 @ b64 / (a64.norm() * b64.norm() + 1e-30)).item()


def legacy_forward(lin: ModelOptFP8Linear, x: torch.Tensor) -> torch.Tensor:
    lin._ensure_ready()
    return torch.nn.functional.linear(x, lin._weight_bf16, lin.bias)


def quantize_activation_fp8(x_bf16: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    """Static per-tensor FP8 (E4M3) activation quantization, same
    ``value_fp8 = round(value / scale)`` convention
    ``runtime/loading/modelopt.py::dequantize_fp8`` uses in reverse
    (``value_bf16 = value_fp8 * scale``) -- checked directly against that
    function's own docstring, not assumed independently."""
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    scaled = (x_bf16.float() / input_scale.float()).clamp(-fp8_max, fp8_max)
    return scaled.to(torch.float8_e4m3fn)


def kernel_forward(lin: ModelOptFP8Linear, packed, x: torch.Tensor) -> torch.Tensor:
    x_fp8 = quantize_activation_fp8(x, lin.input_scale.data)
    return tensor_fp8_linear.mm(x_fp8, packed, out_dtype=torch.bfloat16)


def run_case(lin: ModelOptFP8Linear, packed, in_features: int, m: int, seed: int) -> None:
    torch.manual_seed(seed)
    x = (torch.randn(m, in_features, device=DEVICE, dtype=torch.bfloat16) * 0.02).contiguous()

    ref = legacy_forward(lin, x)
    lin._weight_bf16 = None  # don't let the legacy cache linger past this call

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n_iters = 20 if m <= 8 else 5
    for _ in range(n_iters):
        out = kernel_forward(lin, packed, x)
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
    # sparkinfer.gemm.tensor_fp8_linear.is_supported() reports False on
    # this machine's installed nvidia-cutlass-dsl (4.5.2): it gates on
    # sparkinfer._lib.gating.MIN_CUTLASS_DSL == "4.6.0" via
    # default_is_supported -> has_cutlass_dsl, a version-string floor, not
    # a live capability probe. Checked directly (not assumed) that the
    # floor is overly conservative for THIS op on THIS hardware:
    # cute.nvgpu.warp.MmaMXF8Op (what
    # sparkinfer.gemm.tensor_fp8_linear._kernel.is_tensor_fp8_linear_supported
    # actually probes for) is present, and a synthetic pack_weight+mm call
    # (matching sparkinfer's own tests/gemm/test_tensor_fp8_linear.py
    # fixture) produces the expected result within that test's own
    # tolerance (rtol=1e-2, atol=2e-3; measured max abs diff ~1.5e-5 on a
    # [7,128]x[64,128] synthetic case, run 2026-08-03). sparkinfer's
    # MIN_CUTLASS_DSL = "4.6.0" (sparkinfer/_lib/gating.py) is a flat
    # version-string floor with no comment explaining why 4.6.0
    # specifically -- not evidence this op is broken on 4.5.2, just an
    # untested combination from sparkinfer's own perspective. So this
    # script calls pack_weight/mm directly rather than gating on
    # is_supported(). If this is ever wrong (a real functional gap in
    # 4.5.2 that 4.6.0 fixes), this script's own cosine/max_abs_err
    # numbers below on REAL checkpoint weights would show it.
    print(f"tensor_fp8_linear.is_supported(): {tensor_fp8_linear.is_supported()} (see comment)")

    ckpt = _find_ckpt()
    print(f"checkpoint: {ckpt}")
    for prefix, label in TARGETS:
        print(f"\n=== {label} ===")
        lin = load_linear(ckpt, prefix)
        print(f"  in_features={lin.input_size} out_features={lin.output_size}")
        print(
            f"  weight_scale={lin.weight_scale.item():.6g} input_scale={lin.input_scale.item():.6g}"
        )
        output_scale = (lin.input_scale.data * lin.weight_scale.data).reshape(1).contiguous()
        packed = tensor_fp8_linear.pack_weight(lin.weight.data, output_scale)
        for m in (1, 2, 8, 32, 128, 512):
            run_case(lin, packed, lin.input_size, m, seed=1234 + m)
        del lin, packed
        torch.cuda.empty_cache()

    mem = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"\npeak allocated during this script: {mem:.3f} GiB")


if __name__ == "__main__":
    main()
