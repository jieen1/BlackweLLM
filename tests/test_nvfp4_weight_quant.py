"""NVFP4 offline weight quantization: contract, accuracy, expert chain.

Day-0 prep for Qwen3.8-Flash-Next's NVFP4 serving path (see
notes/2026-08-26-qwen38-flash-next-day0-survey.md). The end-to-end test
closes the full self-quantization chain: BF16 expert tensors -> checkpoint
triple -> parameterized b12x prepare -> fused MoE forward.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
    pytest.skip(
        "NVFP4 weight quantization tests require an SM120 CUDA device",
        allow_module_level=True,
    )

from runtime.nvfp4_weight_quant import (  # noqa: E402
    dequantize_nvfp4_weight,
    quantize_nvfp4_weight,
)


@pytest.fixture(autouse=True)
def _release_quant_cuda_cache():
    yield
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def test_checkpoint_triple_contract():
    torch.manual_seed(0)
    w = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
    packed, scale, gs = quantize_nvfp4_weight(w)
    assert tuple(packed.shape) == (256, 256)
    assert packed.dtype == torch.uint8
    assert tuple(scale.shape) == (256, 32)
    assert scale.dtype == torch.float8_e4m3fn
    assert tuple(gs.shape) == ()
    assert gs.dtype == torch.float32
    # 2688 / amax convention (2688 = 6 * 448).
    expected_gs = 2688.0 / float(w.float().abs().amax())
    assert abs(float(gs) - expected_gs) / expected_gs < 1e-3


def test_roundtrip_accuracy():
    torch.manual_seed(1)
    w = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
    packed, scale, gs = quantize_nvfp4_weight(w)
    deq = dequantize_nvfp4_weight(packed, scale, gs)
    ref = w.float()
    # NVFP4 (block-16 e2m1) on unit-Gaussian weights: nRMSE lands ~0.09-0.10
    # and cosine ~0.995. Per-element relative error is NOT the gate here --
    # near-zero weights blow it up regardless of correctness.
    nrmse = ((deq - ref).norm() / ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(deq.reshape(1, -1), ref.reshape(1, -1)).item()
    assert nrmse < 0.15, nrmse
    assert cos > 0.99, cos


def test_zero_weight_is_safe():
    w = torch.zeros(64, 128, dtype=torch.bfloat16, device="cuda")
    packed, scale, gs = quantize_nvfp4_weight(w)
    assert int(packed.abs().sum()) == 0
    assert float(gs) == 1.0
    deq = dequantize_nvfp4_weight(packed, scale, gs)
    assert torch.isfinite(deq).all().item()
    assert float(deq.abs().amax()) == 0.0


def test_rejects_non_finite_and_bad_shapes():
    w = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    w[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        quantize_nvfp4_weight(w)
    with pytest.raises(ValueError, match="rank-2"):
        quantize_nvfp4_weight(torch.randn(4, 8, 16, device="cuda"))


def test_self_quantized_experts_run_through_b12x():
    """Full chain at the proven Laguna geometry: BF16 experts -> checkpoint
    triple -> parameterized prepare -> fused forward."""
    from runtime.backends.qwen38_sparkinfer_moe import (
        allocate_tp_moe_workspace_pool,
        make_qwen_moe_expert_layer,
    )
    from runtime.model.qwen38_moe import QwenMoeGeometry

    geom = QwenMoeGeometry(
        num_experts=256,
        top_k=10,
        hidden_size=3072,
        moe_intermediate_size=1024,
        shared_expert_intermediate_size=1024,
    )
    torch.manual_seed(2)
    raw: dict[str, torch.Tensor] = {}
    for name, n, k in (
        ("gate", geom.moe_intermediate_size, geom.hidden_size),
        ("up", geom.moe_intermediate_size, geom.hidden_size),
        ("down", geom.hidden_size, geom.moe_intermediate_size),
    ):
        packed_list, sf_list, gs_list = [], [], []
        for _ in range(geom.num_experts):
            w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
            packed, sf, gs = quantize_nvfp4_weight(w)
            packed_list.append(packed)
            sf_list.append(sf)
            gs_list.append(gs)
            del w
        raw[f"{name}_w"] = torch.stack(packed_list)
        raw[f"{name}_sf"] = torch.stack(sf_list)
        raw[f"{name}_gs"] = torch.stack(gs_list).float()
        del packed_list, sf_list, gs_list

    layer = make_qwen_moe_expert_layer(
        raw,
        geom,
        allocate_tp_moe_workspace_pool(),
        a1_gscale=0.0005,
        a2_gscale=0.0013,
    )
    a = torch.randn(2, geom.hidden_size, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, geom.num_experts, (2, geom.top_k), dtype=torch.int32, device="cuda")
    weights = torch.softmax(torch.randn(2, geom.top_k, device="cuda"), dim=-1).float()
    out = layer.forward(a, ids, weights)
    torch.cuda.synchronize()
    assert tuple(out.shape) == (2, geom.hidden_size)
    assert torch.isfinite(out.float()).all().item()
