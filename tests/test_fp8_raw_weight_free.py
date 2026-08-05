"""The FP8 half of the dequant-cache fix, which was missing for months.

`free_nvfp4_raw_params` drops the raw NVFP4 parameters once `Qwen36MLP` has
built its fused representation, and took the resident set from 76.34 GiB to
53.08 GiB. It covered the 56 NVFP4 MLP layers and nothing else.

The other 237 quantized tensors -- attention q/k/v/o, the GDN projections,
`lm_head`, and layers 56-63's MLP -- are FP8, and
`CompressedTensorsFP8ChannelLinear` dequantizes each to BF16 on first use and
caches it forever *while keeping the FP8 original resident*. `forward` reads
only the BF16 copy. Measured against the standard checkpoint's real tensor
shapes: 10.73B parameters, so 9.99 GiB of FP8 originals held for nothing
behind 19.99 GiB of cache (`notes/2026-08-03-production-memory-audit.md`).

Nothing surfaced that for as long as it existed, because the symptom is
purely a number in `nvidia-smi` that nobody had attributed. The NVFP4 fix
looked complete -- resident memory dropped a lot, the headline number moved,
and the remaining gap was not broken down until the audit.

What these tests pin, in order of what would actually regress:

1. `forward` still works after the free, and its output is unchanged -- the
   whole premise is that `.weight` is dead once `_weight_bf16` exists, and if
   that premise is wrong the model silently produces different numbers.
2. The freed storage is actually released, not merely reassigned to something
   the same size.
3. The model-level sweep reaches a non-zero number of Linears. A sweep that
   matches nothing passes every other assertion while saving nothing.
4. It is idempotent and safe before any forward, since it is called at load
   time and the dequantization is lazy.

CPU-only: tiny Linears, no checkpoint, no GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch-free CI job")

from runtime.model.compressed_tensors_linear import (  # noqa: E402
    QSR_NATIVE_W8A8_FP8_CHANNEL_ENV,
    QSR_TORCH_SCALED_MM_FP8_CHANNEL_ENV,
    CompressedTensorsFP8ChannelLinear,
)


@pytest.fixture(autouse=True)
def _use_explicit_legacy_fallback(monkeypatch):
    """This file locks the retained BF16 fallback, not CUDA serving."""
    monkeypatch.setenv(QSR_TORCH_SCALED_MM_FP8_CHANNEL_ENV, "0")


def _linear(out_features: int = 8, in_features: int = 16) -> CompressedTensorsFP8ChannelLinear:
    lin = CompressedTensorsFP8ChannelLinear(in_features, out_features, bias=False)
    torch.manual_seed(20260803)
    raw = (torch.randn(out_features, in_features) * 0.1).to(torch.float8_e4m3fn)
    lin.weight.data = raw
    lin.weight_scale.data = torch.full((out_features, 1), 0.5, dtype=torch.bfloat16)
    return lin


class TestFreeingDoesNotChangeResults:
    def test_forward_output_is_identical_after_freeing(self):
        """The premise of the whole change: forward never reads .weight again."""
        lin = _linear()
        x = torch.randn(4, 16, dtype=torch.bfloat16)

        before = lin(x)
        lin.free_fp8_raw_weight()
        after = lin(x)

        assert torch.equal(before, after), (
            "freeing the FP8 original changed forward's output, so something "
            "still reads .weight -- this change is not safe as written"
        )

    def test_forward_works_when_freed_before_any_forward(self):
        """Called at load time, before the lazy dequantization would happen."""
        lin = _linear()
        assert lin._weight_bf16 is None
        lin.free_fp8_raw_weight()
        assert lin._weight_bf16 is not None, "freeing must materialize the cache first"
        assert lin(torch.randn(2, 16, dtype=torch.bfloat16)).shape == (2, 8)


class TestStorageIsActuallyReleased:
    def test_the_raw_weight_storage_is_emptied(self):
        lin = _linear()
        assert lin.weight.data.numel() == 8 * 16
        lin.free_fp8_raw_weight()
        assert lin.weight.data.numel() == 0, (
            "storage was not released -- the memory saving does not happen"
        )

    def test_the_parameter_still_exists(self):
        """Reassign .data rather than delete: anything walking the module tree
        (assert_all_params_loaded, state_dict, the checkpoint-family check)
        must not trip over a missing entry."""
        lin = _linear()
        lin.free_fp8_raw_weight()
        assert "weight" in dict(lin.named_parameters())

    def test_it_is_idempotent(self):
        lin = _linear()
        lin.free_fp8_raw_weight()
        lin.free_fp8_raw_weight()
        assert lin(torch.randn(1, 16, dtype=torch.bfloat16)).shape == (1, 8)


def _require_qwen36_model_stack() -> None:
    """Guard for the model-level sweep below.

    It imports runtime.model.qwen36_model, which imports fla and sparkinfer
    at module scope.  The Linear-level classes above only need
    compressed_tensors_linear and must keep running in the cpu-torch CI
    job, so the guard is per-test, matching
    tests/test_laguna_sparkinfer_attn.py's function-level convention.
    """
    pytest.importorskip("fla")
    pytest.importorskip("sparkinfer")


class TestModelLevelSweep:
    def test_default_cuda_contract_preserves_all_raw_fp8_weights(self, monkeypatch):
        _require_qwen36_model_stack()
        """Serving must not materialize a BF16 matrix merely during load."""

        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = _linear()
                self.nested = torch.nn.Module()
                self.nested.b = _linear()

        from runtime.model.qwen36_model import Qwen36ForCausalLMSelfBuilt

        monkeypatch.delenv(QSR_TORCH_SCALED_MM_FP8_CHANNEL_ENV)
        model = Fake()
        freed = Qwen36ForCausalLMSelfBuilt.free_fp8_raw_weights(model)

        assert freed == 0
        assert model.a.weight.data.numel() == 8 * 16
        assert model.nested.b.weight.data.numel() == 8 * 16
        assert model.a._weight_bf16 is None
        assert model.nested.b._weight_bf16 is None

    def test_the_sweep_reports_how_many_it_freed(self):
        _require_qwen36_model_stack()
        """A sweep that silently matches nothing passes every other test here
        while saving nothing at all -- the count is what makes it checkable."""

        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = _linear()
                self.nested = torch.nn.Module()
                self.nested.b = _linear()
                self.unrelated = torch.nn.Linear(4, 4)

        from runtime.model.qwen36_model import Qwen36ForCausalLMSelfBuilt

        model = Fake()
        freed = Qwen36ForCausalLMSelfBuilt.free_fp8_raw_weights(model)

        assert freed == 2, f"expected both FP8 Linears, got {freed}"
        assert model.a.weight.data.numel() == 0
        assert model.nested.b.weight.data.numel() == 0
        assert model.unrelated.weight.numel() == 16, "a plain nn.Linear must be untouched"

    def test_scaled_mm_opt_in_keeps_only_the_selected_raw_mlp_weight(self, monkeypatch):
        _require_qwen36_model_stack()

        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.selected = _linear(out_features=17408, in_features=16)
                self.other = _linear()

        from runtime.model.qwen36_model import Qwen36ForCausalLMSelfBuilt

        monkeypatch.setenv(QSR_TORCH_SCALED_MM_FP8_CHANNEL_ENV, "1")
        model = Fake()
        freed = Qwen36ForCausalLMSelfBuilt.free_fp8_raw_weights(model)

        assert freed == 1
        assert model.selected.weight.data.numel() == 17408 * 16
        assert model.other.weight.data.numel() == 0

    def test_scaled_mm_all_opt_in_keeps_every_raw_fp8_weight(self, monkeypatch):
        _require_qwen36_model_stack()

        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = _linear()
                self.b = _linear()

        from runtime.model.qwen36_model import Qwen36ForCausalLMSelfBuilt

        monkeypatch.setenv(QSR_TORCH_SCALED_MM_FP8_CHANNEL_ENV, "all")
        model = Fake()
        freed = Qwen36ForCausalLMSelfBuilt.free_fp8_raw_weights(model)

        assert freed == 0
        assert model.a.weight.data.numel() == 8 * 16
        assert model.b.weight.data.numel() == 8 * 16
        assert model.a._weight_bf16 is None
        assert model.b._weight_bf16 is None

    def test_native_w8a8_all_opt_in_keeps_every_raw_fp8_weight(self, monkeypatch):
        _require_qwen36_model_stack()
        """The native route cannot need a BF16 cache just to load a model."""

        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = _linear()
                self.b = _linear()

        from runtime.model.qwen36_model import Qwen36ForCausalLMSelfBuilt

        monkeypatch.setenv(QSR_NATIVE_W8A8_FP8_CHANNEL_ENV, "all")
        model = Fake()
        freed = Qwen36ForCausalLMSelfBuilt.free_fp8_raw_weights(model)

        assert freed == 0
        assert model.a.weight.data.numel() == 8 * 16
        assert model.b.weight.data.numel() == 8 * 16
        assert model.a._weight_bf16 is None
        assert model.b._weight_bf16 is None

    def test_weight_only_executor_can_preserve_every_raw_weight(self):
        _require_qwen36_model_stack()
        """A raw-FP8 executor must not create a BF16 cache simply to load."""

        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = _linear()
                self.nested = torch.nn.Module()
                self.nested.b = _linear()

        from runtime.model.qwen36_model import Qwen36ForCausalLMSelfBuilt

        model = Fake()
        freed = Qwen36ForCausalLMSelfBuilt.free_fp8_raw_weights(model, keep_all_raw=True)

        assert freed == 0
        assert model.a.weight.data.numel() == 8 * 16
        assert model.nested.b.weight.data.numel() == 8 * 16
        assert model.a._weight_bf16 is None
        assert model.nested.b._weight_bf16 is None
