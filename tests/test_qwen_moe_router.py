"""Triton softmax + top-k router parity for the Qwen MoE family contract.

Reference semantics: ``Qwen3NextTopKRouter`` (softmax in FP32, top-k on the
probabilities, optional renormalization) -- see
``runtime/kernels/qwen_moe_router.py`` for the pinned source and
``notes/2026-08-26-qwen38-flash-next-day0-survey.md`` for why this contract
(not Laguna's sigmoid router) serves Qwen3.8-Flash-Next.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("qwen MoE router kernel requires a CUDA device", allow_module_level=True)

from runtime.kernels.qwen_moe_router import (  # noqa: E402
    qwen_moe_router_reference,
    qwen_moe_softmax_topk,
)


@pytest.fixture(autouse=True)
def _release_router_cuda_cache():
    yield
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _check(logits: torch.Tensor, top_k: int, *, renormalize: bool = True) -> None:
    expected_weights, expected_ids = qwen_moe_router_reference(
        logits, top_k, renormalize=renormalize
    )
    weights, ids = qwen_moe_softmax_topk(logits, top_k, renormalize=renormalize)
    torch.cuda.synchronize()
    assert ids.dtype == torch.int32
    assert weights.dtype == logits.dtype
    torch.testing.assert_close(ids, expected_ids)
    torch.testing.assert_close(weights.float(), expected_weights.float(), rtol=1e-4, atol=1e-6)


def _check_near_tie_tolerant(logits: torch.Tensor, top_k: int, *, renormalize: bool = True) -> None:
    """BF16 logits quantize coarsely, so after the FP32 softmax many experts
    land within 1 ulp of each other; the kernel's reduction order is not
    required to reproduce torch's exact near-tie ordering, and the emitted
    weights carry the logits dtype's rounding (bf16 ulp ~ 2^-8 relative).
    The contract is instead: every selected expert's true probability equals
    its emitted weight up to the renormalization denominator and the output
    dtype's rounding, and the sorted weight spectra agree with the
    reference."""
    probs = torch.softmax(logits.float(), dim=-1)
    expected_weights, _ = qwen_moe_router_reference(logits, top_k, renormalize=renormalize)
    weights, ids = qwen_moe_softmax_topk(logits, top_k, renormalize=renormalize)
    torch.cuda.synchronize()
    # Two bf16 code positions: the kernel's FP32 softmax matches torch's to
    # ~1e-6, but the bf16 output rounds each weight independently of the
    # reference's fp32 renormalization.
    rtol = 8e-3 if logits.dtype == torch.bfloat16 else 1e-4
    chosen = probs.gather(1, ids.long())
    if renormalize:
        denom = chosen.sum(dim=-1, keepdim=True)
        denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        chosen = chosen / denom
    torch.testing.assert_close(weights.float(), chosen, rtol=rtol, atol=1e-6)
    torch.testing.assert_close(
        weights.float().sort(descending=True).values,
        expected_weights.float().sort(descending=True).values,
        rtol=rtol,
        atol=1e-6,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 4, 37, 128])
def test_matches_reference_flash_next_geometry(dtype, rows):
    torch.manual_seed(0)
    logits = torch.randn(rows, 512, dtype=dtype, device="cuda") * 3.0
    if dtype == torch.float32:
        _check(logits, top_k=10)
    else:
        _check_near_tie_tolerant(logits, top_k=10)


@pytest.mark.parametrize("num_experts,top_k", [(256, 10), (200, 6), (128, 8), (512, 1)])
def test_matches_reference_other_geometries(num_experts, top_k):
    torch.manual_seed(1)
    logits = torch.randn(23, num_experts, dtype=torch.float32, device="cuda") * 3.0
    _check(logits, top_k)


def test_exact_ties_break_toward_lower_expert():
    logits = torch.zeros(3, 512, dtype=torch.float32, device="cuda")
    weights, ids = qwen_moe_softmax_topk(logits, 10)
    torch.cuda.synchronize()
    expected_ids = torch.arange(10, dtype=torch.int32, device="cuda").expand(3, 10)
    torch.testing.assert_close(ids, expected_ids.contiguous())
    torch.testing.assert_close(
        weights, torch.full((3, 10), 0.1, dtype=weights.dtype, device="cuda")
    )


def test_renormalize_off_keeps_raw_probabilities():
    torch.manual_seed(2)
    logits = torch.randn(16, 512, dtype=torch.float32, device="cuda") * 3.0
    _check(logits, top_k=10, renormalize=False)
    _, ids = qwen_moe_softmax_topk(logits, 10, renormalize=False)
    probs = torch.softmax(logits, dim=-1)
    raw = probs.gather(1, ids.long())
    weights, _ = qwen_moe_softmax_topk(logits, 10, renormalize=False)
    torch.testing.assert_close(weights.float(), raw, rtol=1e-4, atol=1e-6)


def test_caller_owned_output_arenas():
    torch.manual_seed(3)
    logits = torch.randn(64, 512, dtype=torch.bfloat16, device="cuda") * 3.0
    weights_out = torch.zeros(64, 10, dtype=torch.bfloat16, device="cuda")
    ids_out = torch.zeros(64, 10, dtype=torch.int32, device="cuda")
    weights, ids = qwen_moe_softmax_topk(logits, 10, weights_out=weights_out, ids_out=ids_out)
    assert weights.data_ptr() == weights_out.data_ptr()
    assert ids.data_ptr() == ids_out.data_ptr()
    _check_near_tie_tolerant(logits, 10)


def test_empty_batch_returns_empty_outputs():
    logits = torch.empty(0, 512, dtype=torch.float32, device="cuda")
    weights, ids = qwen_moe_softmax_topk(logits, 10)
    assert tuple(weights.shape) == (0, 10)
    assert tuple(ids.shape) == (0, 10)


def test_validation_rejects_bad_contracts():
    logits = torch.randn(4, 512, device="cuda")
    with pytest.raises(ValueError, match="top_k"):
        qwen_moe_softmax_topk(logits, 0)
    with pytest.raises(ValueError, match="top_k"):
        qwen_moe_softmax_topk(logits, 513)
    with pytest.raises(ValueError, match="rank-2"):
        qwen_moe_softmax_topk(logits.view(-1), 10)
    with pytest.raises(ValueError, match="BF16 or FP32"):
        qwen_moe_softmax_topk(logits.half(), 10)
    with pytest.raises(ValueError, match="contiguous CUDA"):
        qwen_moe_softmax_topk(logits.cpu(), 10)
    with pytest.raises(ValueError, match="weights_out"):
        qwen_moe_softmax_topk(logits, 10, weights_out=torch.empty(3, 10, device="cuda"))
    with pytest.raises(ValueError, match="ids_out"):
        qwen_moe_softmax_topk(logits, 10, ids_out=torch.empty(4, 10, device="cuda"))
