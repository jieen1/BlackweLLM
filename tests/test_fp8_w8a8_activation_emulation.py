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

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from runtime.model.compressed_tensors_linear import (  # noqa: E402
    QSR_EMULATE_FP8_ACTIVATION_ENV,
    CompressedTensorsFP8ChannelLinear,
    emulate_fp8_activation_round_trip,
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
