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
    CompressedTensorsFP8ChannelLinear,
)


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


class TestModelLevelSweep:
    def test_the_sweep_reports_how_many_it_freed(self):
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
