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

By default no FP8xFP8 kernel is built or called here.  With
``--raw-fp8-kernel``, it additionally exercises the explicit preflight
composition in ``CompressedTensorsFP8ChannelLinear``: SparkInfer's raw FP8
dot product followed by this checkpoint's separate per-token activation and
per-output-channel weight scales.  It compares that result to the emulated
W8A8 arithmetic, not to the default BF16 serving path, so any mismatch
localizes this wrapper's layout/scale composition rather than intended W8A8
quantization error.  This remains diagnostic-only and does not change
production routing.  The default emulation comparison still makes the
measured gap a LOWER bound on real W8A8 error -- see
``runtime/model/compressed_tensors_linear.py::emulate_fp8_activation_round_trip``'s
docstring for why a lower bound is the right tool for a pre-flight negative
check.

``--marlin-w8a16`` is intentionally a different experiment: it keeps the
activation in BF16 and runs a *weight-only* E4M3 kernel, using vLLM's Marlin
implementation strictly as an offline oracle for the no-vLLM port.  It is
the relevant candidate after W8A8's activation round-trip failed B1-R: raw
FP8 weights stay compressed and are unpacked inside the GEMM, but no new
activation quantization is introduced.  This script must never be imported
by production code; it exists to nail the exact packing/scale contract before
we re-home that kernel behind the runtime's own backend.

``--historical-quant-oracle`` is a differential gate: it invokes the archived
vLLM dynamic-token E4M3 operator only from this diagnostic and requires
byte-for-byte equality of its FP8 codes and FP32 scales with our self-owned
CUDA quantizer; it also reports the serving fallback's mismatch count.  It is
never a production dependency.

``--historical-gemm-oracle`` extends that differential gate through the actual
historical CUTLASS W8A8 operator.  Both operators consume the *same* archived
E4M3 activation bytes, per-token FP32 scales, checkpoint E4M3 weight bytes,
and checkpoint channel scales.  Thus an output difference can only be in the
scaled-GEMM implementation; it cannot be explained by a weight conversion or
by activation quantization.  This is diagnostic-only and is the prerequisite
for changing the production raw-FP8 path.

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
import time
from ctypes import CDLL, POINTER, byref, c_int, c_size_t, c_void_p
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
from torch import nn  # noqa: E402

from runtime.checkpoints import standard_checkpoint_path  # noqa: E402
from runtime.fp8_w8a8 import ABI_VERSION, NativeFP8W8A8Library  # noqa: E402
from runtime.model.compressed_tensors_linear import (  # noqa: E402
    CompressedTensorsFP8ChannelLinear,
    emulate_fp8_activation_round_trip,
    quantize_fp8_activation_per_token,
)

CKPT = Path(standard_checkpoint_path())
DEVICE = "cuda"
NATIVE_W8A8_LIBRARY = _ROOT + "/runtime/kernels/_generated/fp8_w8a8_sm120.so"

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


def native_w8a8_scaled_mm(
    x: torch.Tensor,
    linear: CompressedTensorsFP8ChannelLinear,
    *,
    batch_invariant: bool,
) -> torch.Tensor:
    """Call the self-owned raw-pointer SM120 W8A8 GEMM diagnostic ABI.

    This is deliberately script-local until its single-layer and full-model
    gates qualify it.  It neither imports nor links vLLM.
    """
    library = Path(NATIVE_W8A8_LIBRARY)
    if not library.is_file():
        raise RuntimeError(
            f"native W8A8 library is missing: {library}; build it before requesting --native-w8a8"
        )
    x_fp8, activation_scale = quantize_fp8_activation_per_token(x)
    weight_scale = linear.weight_scale.data.t().to(torch.float32)
    output = torch.empty(
        (x_fp8.shape[0], linear.output_size), dtype=x.dtype, device=x.device
    )
    native = CDLL(str(library))
    abi = native.qsr_fp8_w8a8_abi_version
    abi.argtypes = ()
    abi.restype = c_int
    if abi() != ABI_VERSION:
        raise RuntimeError(
            f"native W8A8 ABI mismatch: expected {ABI_VERSION}, got {abi()}"
        )
    workspace_size = native.qsr_fp8_w8a8_workspace_size_sm120
    workspace_size.argtypes = (POINTER(c_size_t), c_int, c_int, c_int, c_int)
    workspace_size.restype = c_int
    workspace_bytes = c_size_t()
    status = workspace_size(
        byref(workspace_bytes),
        x_fp8.shape[0],
        linear.output_size,
        linear.input_size,
        int(batch_invariant),
    )
    if status != 0:
        raise RuntimeError(f"native W8A8 workspace-size query returned status {status}")
    workspace = torch.empty(workspace_bytes.value, dtype=torch.uint8, device=x.device)
    op = native.qsr_fp8_w8a8_scaled_mm_sm120
    op.argtypes = (
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_size_t,
        c_int,
        c_int,
        c_int,
        c_int,
        c_void_p,
    )
    op.restype = c_int
    status = op(
        c_void_p(output.data_ptr()),
        c_void_p(x_fp8.data_ptr()),
        c_void_p(linear.weight.data.t().data_ptr()),
        c_void_p(activation_scale.data_ptr()),
        c_void_p(weight_scale.data_ptr()),
        c_void_p(workspace.data_ptr()) if workspace.numel() else None,
        workspace.numel(),
        x_fp8.shape[0],
        linear.output_size,
        linear.input_size,
        int(batch_invariant),
        c_void_p(torch.cuda.current_stream(x.device).cuda_stream),
    )
    if status != 0:
        raise RuntimeError(f"native W8A8 GEMM returned CUTLASS status {status}")
    return output


def assert_historical_quant_oracle(x: torch.Tensor) -> None:
    """Prove dynamic E4M3 codes and scales match the historical operator.

    vLLM is imported only after explicit diagnostic opt-in.  Production has no
    path through this function or its dependency.
    """
    try:
        from vllm import _custom_ops as historical_ops
    except ImportError as exc:
        raise RuntimeError(
            "--historical-quant-oracle requires the archived vLLM diagnostic environment"
        ) from exc

    historical_codes, historical_scale = historical_ops.scaled_fp8_quant(
        x, use_per_token_if_dynamic=True
    )
    serving_codes, serving_scale = quantize_fp8_activation_per_token(x)
    native = NativeFP8W8A8Library.load()
    native_codes = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    native_scale = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    native.quantize_per_token(x, native_codes, native_scale)
    torch.cuda.synchronize(x.device)

    def mismatch_count(left: torch.Tensor, right: torch.Tensor) -> int:
        return int((left != right).sum().item())

    historical_bytes = historical_codes.view(torch.uint8)
    serving_code_mismatch = mismatch_count(serving_codes.view(torch.uint8), historical_bytes)
    serving_scale_mismatch = mismatch_count(serving_scale, historical_scale)
    native_code_mismatch = mismatch_count(native_codes.view(torch.uint8), historical_bytes)
    native_scale_mismatch = mismatch_count(native_scale, historical_scale)
    print(
        "    historical_quant_oracle: "
        f"torch_codes={serving_code_mismatch} torch_scales={serving_scale_mismatch} "
        f"native_codes={native_code_mismatch} native_scales={native_scale_mismatch}"
    )
    assert native_code_mismatch == native_scale_mismatch == 0, (
        "self-owned CUDA dynamic FP8 quantizer differs from historical vLLM"
    )


def assert_historical_gemm_oracle(
    x: torch.Tensor,
    linear: CompressedTensorsFP8ChannelLinear,
) -> None:
    """Compare the current CUDA scaled-MM implementation with historical CUTLASS.

    The archived quantizer produces the common left operand deliberately.  Do
    not replace it with our Torch or native quantizer here: that would make a
    GEMM disagreement ambiguous.  The checkpoint-native E4M3 weight remains a
    transposed view, exactly as historical ``process_weights_after_loading``
    leaves it for CUTLASS; no BF16 matrix is read by either tested operator.
    """
    try:
        from vllm import _custom_ops as historical_ops
    except ImportError as exc:
        raise RuntimeError(
            "--historical-gemm-oracle requires the archived vLLM diagnostic environment"
        ) from exc

    historical_codes, historical_scale = historical_ops.scaled_fp8_quant(
        x, use_per_token_if_dynamic=True
    )
    weight_column_major = linear.weight.data.t()
    weight_scale = linear.weight_scale.data.t().to(torch.float32)
    historical_output = historical_ops.cutlass_scaled_mm(
        historical_codes,
        weight_column_major,
        scale_a=historical_scale,
        scale_b=weight_scale,
        out_dtype=torch.bfloat16,
    )
    torch_output = torch._scaled_mm(
        historical_codes,
        weight_column_major,
        scale_a=historical_scale,
        scale_b=weight_scale,
        out_dtype=torch.bfloat16,
    )
    if isinstance(torch_output, tuple):
        torch_output = torch_output[0]
    torch.cuda.synchronize(x.device)
    mismatch = int((torch_output != historical_output).sum().item())
    max_abs = (torch_output.float() - historical_output.float()).abs().max().item()
    print(
        "    historical_gemm_oracle: "
        f"torch_output_mismatch={mismatch} max_abs={max_abs:.6f} "
        f"cosine={cosine(torch_output, historical_output):.9f}"
    )


class _MarlinW8A16Layer(nn.Module):
    """Minimal vLLM Marlin setup object used only by this diagnostic.

    The weight is checkpoint-native E4M3 ``[N, K]`` and the scale is this
    checkpoint's BF16 per-output-channel ``[N, 1]``.  Marlin mutates both
    tensors into its packed launch layout, so this object owns clones and
    cannot affect the production Linear under test.
    """

    def __init__(self, linear: CompressedTensorsFP8ChannelLinear) -> None:
        super().__init__()
        self.input_size_per_partition = linear.input_size
        self.output_size_per_partition = linear.output_size
        self.orig_dtype = torch.bfloat16
        self.weight_block_size = None
        self.logical_widths = [linear.output_size]
        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        self.weight_scale = nn.Parameter(
            linear.weight_scale.detach().clone(), requires_grad=False
        )
        self.register_parameter("input_scale", None)


def prepare_marlin_w8a16(
    linear: CompressedTensorsFP8ChannelLinear,
) -> tuple[_MarlinW8A16Layer, object]:
    """Prepare the historical W8A16 kernel without touching runtime state."""
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_fp8_marlin_linear,
            prepare_fp8_layer_for_marlin,
        )
    except ImportError as exc:
        raise RuntimeError(
            "--marlin-w8a16 requires the local vLLM oracle environment; it is "
            "diagnostic-only and is not a production dependency"
        ) from exc

    layer = _MarlinW8A16Layer(linear)
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    return layer, apply_fp8_marlin_linear


def forward_marlin_w8a16(
    x: torch.Tensor,
    layer: _MarlinW8A16Layer,
    apply: object,
) -> torch.Tensor:
    """Invoke the historical BF16-activation / FP8-weight oracle."""
    return apply(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        workspace=layer.workspace,
        size_n=layer.output_size_per_partition,
        size_k=layer.input_size_per_partition,
        input_dtype=None,
        bias=None,
    )


def run_case(
    lin: CompressedTensorsFP8ChannelLinear,
    in_features: int,
    m: int,
    seed: int,
    *,
    raw_fp8_kernel: bool,
    torch_scaled_mm: bool,
    native_w8a8: bool,
    marlin_w8a16: bool,
    historical_quant_oracle: bool,
    historical_gemm_oracle: bool,
    benchmark_iters: int,
) -> None:
    torch.manual_seed(seed)
    x = (torch.randn(m, in_features, device=DEVICE, dtype=torch.bfloat16) * 0.02).contiguous()

    if historical_quant_oracle:
        assert_historical_quant_oracle(x)
    if historical_gemm_oracle:
        assert_historical_gemm_oracle(x, lin)

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

    if raw_fp8_kernel:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        kernel_out = lin.forward_fp8_channel_kernel(x, expected_m=m)
        torch.cuda.synchronize()
        kernel_s = time.perf_counter() - t0
        kernel_max_abs_err = (kernel_out.double() - out.double()).abs().max().item()
        kernel_rel_err = kernel_max_abs_err / (out.double().abs().max().item() + 1e-30)
        kernel_cos = cosine(kernel_out, out)
        print(
            f"    raw_fp8_kernel_vs_emulated: cosine={kernel_cos:.6f} "
            f"max_abs_err={kernel_max_abs_err:.6f} rel_to_max={kernel_rel_err:.6f} "
            f"first_call_latency={kernel_s * 1e6:.1f}us"
        )
        if benchmark_iters:
            with torch.inference_mode():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(benchmark_iters):
                    F.linear(x, lin._weight_bf16, lin.bias)
                torch.cuda.synchronize()
                bf16_us = (time.perf_counter() - t0) * 1e6 / benchmark_iters

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(benchmark_iters):
                    lin.forward_fp8_channel_kernel(x, expected_m=m)
                torch.cuda.synchronize()
                raw_fp8_us = (time.perf_counter() - t0) * 1e6 / benchmark_iters
            print(
                f"    steady_latency: bf16_dequant_cache={bf16_us:.1f}us "
                f"raw_fp8_composed={raw_fp8_us:.1f}us "
                f"ratio={raw_fp8_us / bf16_us:.3f}x "
                f"iters={benchmark_iters}"
            )

    if torch_scaled_mm:
        x_fp8, activation_scale = quantize_fp8_activation_per_token(x)
        try:
            torch_output = torch._scaled_mm(
                x_fp8,
                # CUDA scaled_mm's B operand is column-major [K, N], so the
                # transposed contiguous [N, K] checkpoint tensor must retain
                # its stride-0==1 view; making that view contiguous changes
                # the physical contract and is rejected by the kernel.
                lin.weight.data.t(),
                scale_a=activation_scale,
                scale_b=lin.weight_scale.data.t().to(torch.float32),
                out_dtype=torch.bfloat16,
            )
            if isinstance(torch_output, tuple):
                torch_output = torch_output[0]
            torch.cuda.synchronize()
            torch_cos = cosine(torch_output, out)
            torch_max_abs = (torch_output.double() - out.double()).abs().max().item()
            torch_rel = torch_max_abs / (out.double().abs().max().item() + 1e-30)
            print(
                f"    torch_scaled_mm_vs_emulated: cosine={torch_cos:.6f} "
                f"max_abs_err={torch_max_abs:.6f} rel_to_max={torch_rel:.6f}"
            )
            if benchmark_iters:
                with torch.inference_mode():
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(benchmark_iters):
                        F.linear(x, lin._weight_bf16, lin.bias)
                    torch.cuda.synchronize()
                    bf16_us = (time.perf_counter() - t0) * 1e6 / benchmark_iters

                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(benchmark_iters):
                        torch._scaled_mm(
                            x_fp8,
                            lin.weight.data.t(),
                            scale_a=activation_scale,
                            scale_b=lin.weight_scale.data.t().to(torch.float32),
                            out_dtype=torch.bfloat16,
                        )
                    torch.cuda.synchronize()
                    scaled_mm_us = (time.perf_counter() - t0) * 1e6 / benchmark_iters
                print(
                    f"    steady_latency: bf16_dequant_cache={bf16_us:.1f}us "
                    f"torch_scaled_mm={scaled_mm_us:.1f}us "
                    f"ratio={scaled_mm_us / bf16_us:.3f}x iters={benchmark_iters}"
                )
        except RuntimeError as exc:
            print(f"    torch_scaled_mm_unavailable: {exc}")

    if native_w8a8:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        native_output = native_w8a8_scaled_mm(x, lin, batch_invariant=False)
        torch.cuda.synchronize()
        native_first_us = (time.perf_counter() - t0) * 1e6
        native_cos = cosine(native_output, out)
        native_max_abs = (native_output.double() - out.double()).abs().max().item()
        native_rel = native_max_abs / (out.double().abs().max().item() + 1e-30)
        print(
            f"    native_w8a8_vs_emulated: cosine={native_cos:.6f} "
            f"max_abs_err={native_max_abs:.6f} rel_to_max={native_rel:.6f} "
            f"first_call_latency={native_first_us:.1f}us"
        )

    if marlin_w8a16:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        marlin_layer, marlin_apply = prepare_marlin_w8a16(lin)
        marlin_out = forward_marlin_w8a16(x, marlin_layer, marlin_apply)
        torch.cuda.synchronize()
        first_call_s = time.perf_counter() - t0
        marlin_max_abs = (marlin_out.double() - ref.double()).abs().max().item()
        marlin_rel = marlin_max_abs / (ref.double().abs().max().item() + 1e-30)
        marlin_cos = cosine(marlin_out, ref)
        print(
            f"    marlin_w8a16_vs_bf16_dequant: cosine={marlin_cos:.6f} "
            f"max_abs_err={marlin_max_abs:.6f} rel_to_max={marlin_rel:.6f} "
            f"prepare_plus_first_call={first_call_s * 1e6:.1f}us"
        )
        if benchmark_iters:
            with torch.inference_mode():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(benchmark_iters):
                    F.linear(x, lin._weight_bf16, lin.bias)
                torch.cuda.synchronize()
                bf16_us = (time.perf_counter() - t0) * 1e6 / benchmark_iters

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(benchmark_iters):
                    forward_marlin_w8a16(x, marlin_layer, marlin_apply)
                torch.cuda.synchronize()
                marlin_us = (time.perf_counter() - t0) * 1e6 / benchmark_iters
            print(
                f"    steady_latency: bf16_dequant_cache={bf16_us:.1f}us "
                f"marlin_w8a16={marlin_us:.1f}us "
                f"ratio={marlin_us / bf16_us:.3f}x iters={benchmark_iters}"
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-fp8-kernel",
        action="store_true",
        help="also run the explicit dynamic-W8A8 raw-FP8 GEMM preflight",
    )
    parser.add_argument(
        "--torch-scaled-mm",
        action="store_true",
        help="probe CUDA row/column FP8 scaled_mm as a fused-epilogue alternative",
    )
    parser.add_argument(
        "--native-w8a8",
        action="store_true",
        help="probe the self-owned SM120 raw-pointer W8A8 GEMM diagnostic ABI",
    )
    parser.add_argument(
        "--marlin-w8a16",
        action="store_true",
        help="offline oracle: BF16 activations x native FP8 weights, no activation quantization",
    )
    parser.add_argument(
        "--historical-quant-oracle",
        action="store_true",
        help="require byte-exact dynamic FP8 codes/scales against archived vLLM",
    )
    parser.add_argument(
        "--historical-gemm-oracle",
        action="store_true",
        help="compare Torch scaled-MM to archived vLLM CUTLASS with identical FP8 operands",
    )
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        default=(1, 2, 8, 32, 128, 512),
        help="activation row counts to probe",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=0,
        help="after raw-FP8 warmup, report steady BF16 and raw-FP8 latency (0 disables)",
    )
    args = parser.parse_args()

    ckpt = CKPT
    print(f"checkpoint: {ckpt}")
    print("=== FP8 W8A8 pre-flight: per-token activation round-trip emulation vs today's path ===")
    for prefix, label in TARGETS:
        print(f"\n=== {label} ===")
        lin = load_linear(ckpt, prefix)
        print(f"  in_features={lin.input_size} out_features={lin.output_size}")
        for m in args.m:
            run_case(
                lin,
                lin.input_size,
                m,
                seed=1234 + m,
                raw_fp8_kernel=args.raw_fp8_kernel,
                torch_scaled_mm=args.torch_scaled_mm,
                native_w8a8=args.native_w8a8,
                marlin_w8a16=args.marlin_w8a16,
                historical_quant_oracle=args.historical_quant_oracle,
                historical_gemm_oracle=args.historical_gemm_oracle,
                benchmark_iters=args.benchmark_iters,
            )
        del lin
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
