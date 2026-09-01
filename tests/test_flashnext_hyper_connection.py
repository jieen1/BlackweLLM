"""Hyper-connection unit tests: hand-derived zero-weight identities + shapes."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.model.flashnext.hc_kernels import (  # noqa: E402
    hc_fusion_supported,
    hc_norm_apply_fusion_supported,
    hc_norm_fusion_supported,
    hc_pointwise_fusion_supported,
)
from runtime.model.flashnext.hyper_connection import (  # noqa: E402
    GatedResidual,
    GroupedGemmaRMSNorm,
)


def test_hc_fusion_is_not_used_without_a_reference_equivalence_gate():
    assert hc_fusion_supported(torch.zeros(1, 4)) is False


def test_hc_norm_fusion_is_cuda_dtype_gated():
    assert hc_norm_fusion_supported(torch.zeros(1, 4)) is False
    assert hc_norm_apply_fusion_supported(torch.zeros(1, 4)) is False
    assert hc_pointwise_fusion_supported(torch.zeros(1, 4)) is False


def test_grouped_gemma_norm_zero_weight_is_plain_rms():
    norm = GroupedGemmaRMSNorm(8, eps=0.0, group_size=4)
    x = torch.tensor([[2.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0]])
    out = norm(x)
    # per-group rms: 2 and 4 -> normalized to ones; (1 + 0) scaling
    assert torch.allclose(out, torch.ones_like(x), atol=1e-5)


def test_grouped_gemma_norm_weight_shift():
    norm = GroupedGemmaRMSNorm(4, eps=0.0, group_size=4)
    with torch.no_grad():
        norm.weight.fill_(0.5)
    x = torch.tensor([[3.0, 3.0, 3.0, 3.0]])
    out = norm(x)
    assert torch.allclose(out, torch.full_like(x, 1.5), atol=1e-5)


def test_mix_zero_gates_give_half_branch_mean():
    hc = GatedResidual(2, 4, lowrank=3, dtype=torch.float32)
    with torch.no_grad():
        hc.input_mix_weight_down.weight.zero_()
        hc.input_mix_weight_up.weight.zero_()
    x = torch.randn(5, 8)
    mixed, (raw, normed) = hc.mix(x)
    assert tuple(mixed.shape) == (5, 4)
    expected = 0.5 * hc.hc_norm(x).unflatten(-1, (2, 4)).mean(dim=-2)
    assert torch.allclose(mixed, expected, atol=1e-5)
    assert raw is x


def test_combine_zero_gates_add_block_to_every_branch():
    hc = GatedResidual(2, 4, lowrank=3, dtype=torch.float32)
    with torch.no_grad():
        hc.block_inject_weight.weight.zero_()
    x = torch.randn(5, 8)
    block = torch.randn(5, 4)
    _, residuals = hc.mix(x)
    out = hc.combine(block, residuals)
    assert tuple(out.shape) == (5, 8)
    expected = (x.unflatten(-1, (2, 4)) + block.unsqueeze(-2)).flatten(-2)
    assert torch.allclose(out.to(expected.dtype), expected, atol=1e-5)


def test_roundtrip_shapes_and_dtype():
    hc = GatedResidual(4, 16, lowrank=8, dtype=torch.bfloat16)
    with torch.no_grad():
        for p in hc.parameters():
            p.normal_(0, 0.02)
    x = torch.randn(7, 64, dtype=torch.bfloat16)
    mixed, residuals = hc.mix(x)
    assert mixed.dtype == torch.bfloat16
    out = hc.combine(mixed, residuals)
    assert tuple(out.shape) == (7, 64)
    assert torch.isfinite(out.float()).all()


def test_final_mixer_no_combine_build():
    hc = GatedResidual(4, 16, lowrank=8, use_combine=False)
    assert not hasattr(hc, "block_inject_weight") or not hc.use_combine
    x = torch.randn(3, 64)
    mixed, _ = hc.mix(x)
    assert tuple(mixed.shape) == (3, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Triton CUDA kernels")
def test_cuda_fused_hyper_connection_matches_torch_reference(monkeypatch):
    # The reduction-fused experiment is intentionally opt-in; this test
    # qualifies the default reduction-preserving pointwise epilogues.
    monkeypatch.setenv("QSR_FLASHNEXT_HC_NORM_FUSION", "0")
    torch.manual_seed(19)
    branches, hidden, lowrank, rows = 4, 256, 32, 4
    hc = GatedResidual(branches, hidden, lowrank=lowrank, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        for parameter in hc.parameters():
            parameter.normal_(0, 0.02)
    x = torch.randn(rows, branches * hidden, device="cuda", dtype=torch.bfloat16)
    block = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)

    mixed, residuals = hc.mix(x)
    got = hc.combine(block, residuals)

    xf = x.float().view(rows, branches, hidden)
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    normed = (
        xf
        * torch.rsqrt(variance + hc.hc_norm.eps)
        * (1.0 + hc.hc_norm.weight.float().view(branches, hidden))
    ).flatten(-2).to(torch.bfloat16)
    down = torch.nn.functional.linear(normed, hc.input_mix_weight_down.weight) / branches
    up = torch.nn.functional.linear(
        torch.nn.functional.silu(down), hc.input_mix_weight_up.weight
    )
    expected_mixed = (
        torch.sigmoid(up.float()).view(rows, branches, hidden).to(torch.bfloat16)
        * normed.view(rows, branches, hidden)
    ).mean(dim=1)
    inject = torch.nn.functional.linear(normed, hc.block_inject_weight.weight) / branches
    inject = (2.0 * torch.sigmoid(inject.float())).to(torch.bfloat16)
    expected = (
        x.view(rows, branches, hidden)
        + block.unsqueeze(1) * inject.unsqueeze(-1)
    ).flatten(-2)

    # The fused kernels intentionally retain the reference BF16 cast before
    # branch multiply.  Keep this exact gate so future edits cannot silently
    # reintroduce a long-run speculative-decoding drift.
    assert torch.equal(mixed, expected_mixed)
    assert torch.equal(got, expected)
    assert torch.nn.functional.cosine_similarity(
        got.float().flatten(), expected.float().flatten(), dim=0
    ) > 0.9999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Triton CUDA kernels")
def test_cuda_grouped_norm_apply_fusion_matches_reference_exactly():
    torch.manual_seed(31)
    norm = GroupedGemmaRMSNorm(4 * 2560, eps=1e-6, group_size=2560).cuda()
    with torch.no_grad():
        norm.weight.normal_(0.0, 0.02)
    x = torch.randn(4, 4 * 2560, device="cuda", dtype=torch.bfloat16)
    got = norm(x)
    xf = x.float().reshape(4, 4, 2560)
    expected = (
        xf
        * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + norm.eps)
        * (1.0 + norm.weight.float().reshape(4, 2560))
    ).flatten(-2).to(torch.bfloat16)
    # The reduction stays on ATen; the Triton kernel only applies the two
    # post-reduction multiplies.  Keep this bitwise so a future edit cannot
    # silently move the numerical contract back to a reduction-fused path.
    assert torch.equal(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Triton CUDA kernels")
def test_cuda_hc_combine_supports_production_width():
    torch.manual_seed(29)
    branches, hidden, rows = 4, 2560, 4
    hc = GatedResidual(branches, hidden, lowrank=320, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        hc.block_inject_weight.weight.normal_(0, 0.02)
    residual = torch.randn(
        rows,
        branches * hidden,
        device="cuda",
        dtype=torch.bfloat16,
    )
    normed = hc.hc_norm(residual)
    block = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)

    got = hc.combine(block, (residual, normed))
    inject = torch.nn.functional.linear(normed, hc.block_inject_weight.weight) / branches
    expected = (
        residual.float().view(rows, branches, hidden)
        + block.float().unsqueeze(1)
        * (2.0 * torch.sigmoid(inject.float())).unsqueeze(-1)
    ).flatten(-2).to(torch.bfloat16)

    assert torch.allclose(got, expected, rtol=0.02, atol=0.02)
    assert torch.nn.functional.cosine_similarity(
        got.float().flatten(), expected.float().flatten(), dim=0
    ) > 0.99999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Triton CUDA kernels")
def test_cuda_persistent_hc_mix_is_stream_isolated():
    torch.manual_seed(37)
    branches, hidden, lowrank, rows = 4, 512, 64, 4
    hc = GatedResidual(branches, hidden, lowrank=lowrank, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        for parameter in hc.parameters():
            parameter.normal_(0, 0.02)
    inputs = [
        torch.randn(
            rows,
            branches * hidden,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(2)
    ]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    concurrent = []
    for stream, value in zip(streams, inputs, strict=True):
        with torch.cuda.stream(stream):
            mixed, _ = hc.mix(value)
            concurrent.append(mixed)
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    serial = [hc.mix(value)[0] for value in inputs]
    for got, expected in zip(concurrent, serial, strict=True):
        torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires Triton CUDA kernels")
def test_cuda_persistent_hc_mix_matches_rowwise_bit_exact():
    torch.manual_seed(41)
    branches, hidden, lowrank, rows = 4, 512, 64, 4
    hc = GatedResidual(branches, hidden, lowrank=lowrank, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        for parameter in hc.parameters():
            parameter.normal_(0, 0.02)
    value = torch.randn(
        rows,
        branches * hidden,
        device="cuda",
        dtype=torch.bfloat16,
    )

    batched, _ = hc.mix(value)
    rowwise = torch.cat([hc.mix(value[row : row + 1])[0] for row in range(rows)])

    torch.testing.assert_close(batched, rowwise, rtol=0, atol=0)
