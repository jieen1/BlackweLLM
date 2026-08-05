"""``QSR_EMULATE_FP8_ACTIVATION`` must default OFF, and when ON must
actually perturb the activation -- not silently be a no-op.

Why this needs a dedicated, CPU-only test rather than relying on the GPU
pre-flight scripts (``scripts/verify_fp8_w8a8_activation_emulation_*.py``)
to catch a regression: those need the GPU lock and a ~55 GiB real
checkpoint, so nothing runs them on every change. A bug in the env-flag
gate -- e.g. a future edit that reads the flag once at import time instead
of per-call, or inverts the condition, or applies the round-trip
unconditionally -- would silently start paying FP8 W8A8's known-worse
activation-quantization error (single-layer cosine ~0.9996 on the sibling
modelopt scheme, see ``runtime/model/compressed_tensors_linear.py``'s
module docstring) on the hottest 233 Linear calls/decode-step
(``notes/2026-08-03-decode-kernel-profile.md``) with nothing in CI noticing
until a full GPU B1-R run does -- which is exactly the invisible-until-
expensive failure mode this pre-flight exists to avoid building blind.

A second, equally silent failure mode this guards against: an emulation
that accidentally quantizes to something indistinguishable from the input
(e.g. a scale so large every value floors to 0, or a scale bug that leaves
values unchanged) would make every downstream cosine/gap-error number a
false PASS for the wrong reason -- the emulation would be measuring
nothing. ``test_the_round_trip_actually_changes_a_real_activation`` pins
that it does not.

Torch-required (FP8 tensor casts), CUDA-free -- ``float8_e4m3fn`` casts and
``F.linear`` on bf16 both work on CPU (verified directly, not assumed), so
this whole file runs under CI without a GPU. Skips cleanly where torch is
not installed, matching ``tests/test_loading_compressed_tensors_mixed_precision.py``'s
own convention.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from runtime.model.compressed_tensors_linear import (  # noqa: E402
    QSR_EMULATE_FP8_ACTIVATION_ENV,
    CompressedTensorsFP8ChannelLinear,
    emulate_fp8_activation_round_trip,
    quantize_fp8_activation_per_token,
)

ENV = QSR_EMULATE_FP8_ACTIVATION_ENV


def _linear(in_features: int = 16, out_features: int = 8) -> CompressedTensorsFP8ChannelLinear:
    torch.manual_seed(0)
    lin = CompressedTensorsFP8ChannelLinear(in_features, out_features, bias=False)
    lin.weight.data.copy_((torch.randn(out_features, in_features) * 3).to(torch.float8_e4m3fn))
    lin.weight_scale.data.copy_(torch.full((out_features, 1), 0.3, dtype=torch.bfloat16))
    return lin


def _activation(m: int = 4, in_features: int = 16) -> torch.Tensor:
    torch.manual_seed(1)
    return (torch.randn(m, in_features, dtype=torch.bfloat16) * 0.5).contiguous()


class TestDefaultIsOff:
    def test_env_var_unset_forward_matches_plain_bf16_linear(self, monkeypatch):
        """The exact regression this file exists to catch: production must
        never pay the activation-quantization error unless a diagnostic
        script or test explicitly asked for it."""
        monkeypatch.delenv(ENV, raising=False)
        lin = _linear()
        x = _activation()

        out = lin(x)

        lin._ensure_ready()
        expected = F.linear(x, lin._weight_bf16, lin.bias)
        assert torch.equal(out, expected)

    def test_env_var_set_to_something_other_than_1_stays_off(self, monkeypatch):
        """Only the literal string "1" enables it -- "true"/"yes"/"0" must
        not, matching this codebase's existing ``QSR_*`` flag convention
        (e.g. ``QSR_DECODE_CUDA_GRAPH``'s `!= "0"` is the one exception;
        this flag is opt-in, so it follows the opt-in idiom instead)."""
        for bad_value in ("true", "yes", "0", ""):
            monkeypatch.setenv(ENV, bad_value)
            lin = _linear()
            x = _activation()
            out = lin(x)
            lin._ensure_ready()
            expected = F.linear(x, lin._weight_bf16, lin.bias)
            assert torch.equal(out, expected), f"value {bad_value!r} unexpectedly enabled it"


class TestFlagOnActuallyChangesForward:
    def test_env_var_1_changes_forward_output(self, monkeypatch):
        monkeypatch.setenv(ENV, "1")
        lin = _linear()
        x = _activation()
        emulated_out = lin(x)

        monkeypatch.delenv(ENV, raising=False)
        baseline_out = lin(x)

        assert not torch.equal(emulated_out, baseline_out), (
            "QSR_EMULATE_FP8_ACTIVATION=1 must change forward()'s output for a real "
            "activation -- if it does not, the flag is wired to nothing"
        )

    def test_flag_reads_per_call_not_cached_at_import(self, monkeypatch):
        """Guards the specific design choice in this module's docstring:
        the flag must be re-read every forward() call so a test (or a
        script running several workloads in one process) can toggle it
        mid-run and see the effect immediately, without reconstructing the
        Linear or reimporting the module."""
        lin = _linear()
        x = _activation()

        monkeypatch.delenv(ENV, raising=False)
        off_out = lin(x)

        monkeypatch.setenv(ENV, "1")
        on_out = lin(x)

        monkeypatch.delenv(ENV, raising=False)
        off_again = lin(x)

        assert not torch.equal(off_out, on_out)
        assert torch.equal(off_out, off_again)


class TestRoundTripFunction:
    def test_split_quantizer_reconstructs_the_round_trip(self):
        x = _activation(m=3, in_features=16)
        x_fp8, scale = quantize_fp8_activation_per_token(x)

        reconstructed = (x_fp8.float() * scale).to(x.dtype)
        assert torch.equal(reconstructed, emulate_fp8_activation_round_trip(x))
        assert x_fp8.dtype == torch.float8_e4m3fn
        assert scale.shape == (3, 1)

    def test_the_round_trip_actually_changes_a_real_activation(self):
        """A no-op emulation (e.g. a scale computed as 1.0 everywhere, or a
        dtype bug that casts to fp8 and immediately back without ever
        rounding) would make every cosine/gap-error number measured against
        it meaningless -- a PASS for the wrong reason. Most elements of a
        real-looking activation must actually move."""
        x = _activation(m=8, in_features=32)
        x_rt = emulate_fp8_activation_round_trip(x)
        changed_frac = (x_rt != x).float().mean().item()
        assert changed_frac > 0.5, (
            f"only {changed_frac:.1%} of elements changed -- looks like a no-op"
        )

    def test_scale_is_per_token_not_per_tensor(self):
        """This checkpoint's own declared scheme (``config_groups.group_0.
        input_activations.strategy == "token"``) is one scale per ROW, not
        one for the whole activation matrix -- verified directly against
        the standard checkpoint's config.json, 2026-08-03 (see this
        function's docstring in ``runtime/model/compressed_tensors_linear.py``).
        A regression to a single tensor-wide scale would silently let one
        outlier row's magnitude blow the quantization grid's precision for
        every other row -- exactly the kind of scale-convention mistake
        this codebase has hit before (NVFP4's reciprocal/direct mixup, see
        that module's docstring). Pinned here: a row 1000x the magnitude of
        another must not visibly degrade the small row's relative
        precision, which is only true if each row gets its own scale.
        """
        small_row = torch.full((1, 16), 0.01, dtype=torch.bfloat16)
        big_row = torch.full((1, 16), 100.0, dtype=torch.bfloat16)
        x = torch.cat([small_row, big_row], dim=0)

        x_rt = emulate_fp8_activation_round_trip(x)

        small_rel_err = (
            (x_rt[0].float() - small_row[0].float()).abs().max() / small_row[0].float().abs().max()
        ).item()
        # FP8 E4M3 has ~2-3 bits of mantissa; a per-token scale keeps this
        # row's own relative error in that same small-percent regime
        # regardless of the other row's magnitude. A shared/per-tensor
        # scale sized to the big row (100.0) would instead quantize the
        # small row's 0.01 values almost entirely to 0 -- a ~100% relative
        # error -- which is the failure this test distinguishes from.
        assert small_rel_err < 0.25, (
            f"small row's relative error is {small_rel_err:.3f} -- looks like the big "
            "row's magnitude leaked into its scale (per-tensor, not per-token)"
        )

    def test_output_dtype_matches_input(self):
        x = _activation()
        x_rt = emulate_fp8_activation_round_trip(x)
        assert x_rt.dtype == x.dtype
        assert x_rt.shape == x.shape


class TestExplicitKernelPreflightBoundary:
    def test_kernel_preflight_rejects_cpu_weights(self):
        """The experimental GEMM must not become an accidental CPU fallback."""
        lin = _linear()
        with pytest.raises(RuntimeError, match="requires CUDA-resident weights"):
            lin.prepare_fp8_channel_kernel()

    def test_kernel_preflight_rejects_released_raw_weight(self):
        lin = _linear()
        lin.free_fp8_raw_weight()
        with pytest.raises(RuntimeError, match="raw FP8 weight was released"):
            lin.prepare_fp8_channel_kernel()


class TestExplicitKernelPreflightRouting:
    def test_m1_prefers_fused_channel_epilogue_when_available(self, monkeypatch):
        lin = _linear(in_features=32, out_features=8)
        lin._fp8_channel_packed_weight = object()
        lin._fp8_channel_fused_packed_weight = object()
        lin._fp8_channel_kernel_weight_scale = torch.ones((1, 8), dtype=torch.float32)
        monkeypatch.setattr(lin, "prepare_fp8_channel_kernel", lambda: None)

        calls: list[tuple[str, tuple[int, ...]]] = []

        def fake_fused_mm(x_fp8, packed_weight, activation_scale, **kwargs):
            assert packed_weight is lin._fp8_channel_fused_packed_weight
            assert activation_scale.shape == (1,)
            calls.append(("fused", tuple(x_fp8.shape)))
            return torch.full((x_fp8.shape[0], 8), 5, dtype=torch.bfloat16)

        def fake_linear_mm(*args, **kwargs):
            calls.append(("fallback", ()))
            raise AssertionError("M=1 should not hit the fallback path")

        sparkinfer_mod = types.ModuleType("sparkinfer")
        gemm_mod = types.ModuleType("sparkinfer.gemm")
        gemm_mod.tensor_fp8_channel_linear = types.SimpleNamespace(mm=fake_fused_mm)
        gemm_mod.tensor_fp8_linear = types.SimpleNamespace(mm=fake_linear_mm)
        monkeypatch.setitem(sys.modules, "sparkinfer", sparkinfer_mod)
        monkeypatch.setitem(sys.modules, "sparkinfer.gemm", gemm_mod)

        x = _activation(m=1, in_features=32)
        actual = lin.forward_fp8_channel_kernel(x, expected_m=1)

        assert calls == [("fused", (1, 32))]
        assert torch.equal(actual, torch.full((1, 8), 5, dtype=torch.bfloat16))

    def test_multi_row_preflight_keeps_scalar_fallback(self, monkeypatch):
        lin = _linear(in_features=32, out_features=4)
        lin._fp8_channel_packed_weight = object()
        lin._fp8_channel_fused_packed_weight = object()
        lin._fp8_channel_kernel_weight_scale = torch.full((1, 4), 0.5, dtype=torch.float32)
        monkeypatch.setattr(lin, "prepare_fp8_channel_kernel", lambda: None)

        calls: list[tuple[str, tuple[int, ...]]] = []

        def fake_fused_mm(*args, **kwargs):
            calls.append(("fused", ()))
            raise AssertionError("M>1 must stay on the existing fallback path")

        def fake_linear_mm(x_fp8, packed_weight, **kwargs):
            assert packed_weight is lin._fp8_channel_packed_weight
            calls.append(("fallback", tuple(x_fp8.shape)))
            return torch.ones((x_fp8.shape[0], 4), dtype=torch.bfloat16)

        sparkinfer_mod = types.ModuleType("sparkinfer")
        gemm_mod = types.ModuleType("sparkinfer.gemm")
        gemm_mod.tensor_fp8_channel_linear = types.SimpleNamespace(mm=fake_fused_mm)
        gemm_mod.tensor_fp8_linear = types.SimpleNamespace(mm=fake_linear_mm)
        monkeypatch.setitem(sys.modules, "sparkinfer", sparkinfer_mod)
        monkeypatch.setitem(sys.modules, "sparkinfer.gemm", gemm_mod)

        x = _activation(m=2, in_features=32)
        _, activation_scale = quantize_fp8_activation_per_token(x)
        actual = lin.forward_fp8_channel_kernel(x, expected_m=2)
        expected = (torch.ones((2, 4), dtype=torch.float32) * activation_scale * 0.5).to(
            torch.bfloat16
        )

        assert calls == [("fallback", (2, 32))]
        assert torch.equal(actual, expected)


@pytest.mark.skipif(
    os.environ.get("QSR_RUN_FP8_CHANNEL_KERNEL_TEST") != "1",
    reason="explicit single-GPU FP8-channel kernel preflight only",
)
def test_fp8_channel_kernel_matches_emulated_checkpoint_arithmetic():
    """Check the raw-FP8 composition, not the unquantized BF16 default.

    This is opt-in because the first invocation can compile a SparkInfer
    kernel.  The reference has the same per-token activation round-trip and
    per-output-channel weight dequantization, so a failure localizes the
    wrapper's scale/layout composition rather than measuring W8A8's intended
    quantization error against the legacy BF16 serving route.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    lin = _linear(in_features=256, out_features=256).cuda()
    lin.weight_scale.data.copy_(torch.linspace(0.01, 0.5, 256, device="cuda").reshape(256, 1))
    x = (torch.randn(4, 256, device="cuda", dtype=torch.bfloat16) * 0.5).contiguous()

    lin._ensure_ready()
    reference = F.linear(emulate_fp8_activation_round_trip(x), lin._weight_bf16, lin.bias)
    actual = lin.forward_fp8_channel_kernel(x, expected_m=4)
    torch.cuda.synchronize()

    cosine = F.cosine_similarity(
        actual.float().reshape(1, -1), reference.float().reshape(1, -1)
    ).item()
    rel_max_error = (actual.float() - reference.float()).abs().max().item() / (
        reference.float().abs().max().item() + 1e-12
    )
    assert cosine > 0.999, f"cosine={cosine:.7f}"
    assert rel_max_error < 0.05, f"relative max error={rel_max_error:.5f}"
