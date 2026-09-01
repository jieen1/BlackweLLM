"""Flash-Next routed-expert numerical gate: b12x fused MoE vs a hand
dequantized torch reference on real RadixArk layer-0 weights.

Threshold rationale: the b12x path is W4A4 (activations quantized to NVFP4
in-kernel) while the reference keeps BF16 activations, so exact parity is
impossible; cosine > 0.985 plus magnitude agreement is the gate. The scale
convention itself (weight_scale_2 raw, a*_gscale = 1/input_scale) is pinned
from b12x source (_prepare_modelopt_nvfp4_runtime_alphas) -- a wrong
convention fails by 8+ orders of magnitude (2e15 or all-zero), not 15%.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

torch = pytest.importorskip("torch")

CKPT = pathlib.Path("/home/bot/models/Qwen3.8-Flash-Next-NVFP4-RadixArk")

_HAS_CUDA_MOE = (
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0) and CKPT.is_dir()
)
if _HAS_CUDA_MOE:
    from safetensors import safe_open  # noqa: E402

    from runtime.backends.flashnext_moe import (  # noqa: E402
        FlashInferMoELayer,
        SparkinferMoEOutputArena,
        allocate_tp_moe_workspace_pool,
        load_flashinfer_cutlass_moe_ops,
        load_flashnext_experts,
        prepare_flashnext_cutlass_experts,
        prepare_flashnext_experts,
    )
    from runtime.backends.laguna_sparkinfer_moe import SparkinferMoELayer  # noqa: E402
    from runtime.model.flashnext.model import FlashNextMlp, SharedExpert  # noqa: E402
    from runtime.model.qwen38_moe import QwenMoeGeometry  # noqa: E402

    GEOM = QwenMoeGeometry(512, 10, 2560, 640, 640)
_FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


@pytest.mark.skipif(not _HAS_CUDA_MOE, reason="requires the SM120 Flash-Next checkpoint")
def test_flashnext_enables_current_b12x_stability_contract():
    assert os.environ["B12X_DYNAMIC_DETERMINISTIC_OUTPUT"] == "1"
    assert os.environ["B12X_ENABLE_DYNAMIC_DOWN_SCALE"] == "0"


@pytest.mark.skipif(not _HAS_CUDA_MOE, reason="requires the SM120 Flash-Next checkpoint")
def test_batched_stable_expert_ranks_match_flattened_pair_order():
    from b12x.moe._shared.routing import stable_expert_ranks_batched

    torch.manual_seed(2908)
    ids = torch.stack(
        [torch.randperm(GEOM.num_experts, device="cuda")[: GEOM.top_k] for _ in range(512)]
    ).to(torch.int32)
    flat_ids = ids.flatten().contiguous()
    ranks = torch.empty_like(flat_ids)
    assert stable_expert_ranks_batched(
        flat_ids,
        ranks,
        num_experts=GEOM.num_experts,
        num_topk=GEOM.top_k,
    )

    counts = [0] * GEOM.num_experts
    expected = []
    for expert_id in flat_ids.cpu().tolist():
        expected.append(counts[expert_id])
        counts[expert_id] += 1
    torch.testing.assert_close(ranks.cpu(), torch.tensor(expected, dtype=torch.int32))


def _weight_map() -> dict[str, str]:
    with open(CKPT / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def _load_tensor(name: str, weight_map: dict[str, str]) -> torch.Tensor:
    with safe_open(str(CKPT / weight_map[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def _dequant_expert0(proj: str, weight_map: dict[str, str]) -> torch.Tensor:
    prefix = f"model.language_model.layers.0.mlp.experts.0.{proj}"
    packed = _load_tensor(f"{prefix}.weight", weight_map)
    sf = _load_tensor(f"{prefix}.weight_scale", weight_map)
    scale2 = float(_load_tensor(f"{prefix}.weight_scale_2", weight_map))
    codes = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(packed.shape[0], -1)
    vals = _FP4_LUT[(codes & 7).long()] * (1.0 - 2.0 * ((codes >> 3) & 1).float())
    return vals * sf.float().repeat_interleave(16, dim=-1) * scale2


def _make_b12x_layer(
    raw: dict[str, torch.Tensor],
    geometry: QwenMoeGeometry,
) -> SparkinferMoELayer:
    return SparkinferMoELayer(
        prepare_flashnext_experts(
            raw,
            geometry,
            "cuda",
        ),
        allocate_tp_moe_workspace_pool(),
        "cuda",
        output_arena=SparkinferMoEOutputArena(geometry.hidden_size),
    )


@pytest.fixture(scope="module")
def b12x_layer():
    if not _HAS_CUDA_MOE:
        pytest.skip("requires the SM120 Flash-Next checkpoint")
    raw = load_flashnext_experts(CKPT, 0, GEOM, "cuda")
    layer = _make_b12x_layer(raw, GEOM)
    del raw
    torch.cuda.empty_cache()
    yield layer
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.mark.skipif(not _HAS_CUDA_MOE, reason="requires the SM120 Flash-Next checkpoint")
def test_expert_forward_matches_dequant_reference(b12x_layer):
    weight_map = _weight_map()
    gate_w = _dequant_expert0("gate_proj", weight_map)
    up_w = _dequant_expert0("up_proj", weight_map)
    down_w = _dequant_expert0("down_proj", weight_map)

    torch.manual_seed(7)
    x = torch.randn(2, GEOM.hidden_size) * 0.02
    hidden = torch.nn.functional.silu(x @ gate_w.T) * (x @ up_w.T)
    ref = hidden @ down_w.T

    ids = torch.zeros(2, GEOM.top_k, dtype=torch.int32, device="cuda")
    weights = torch.zeros(2, GEOM.top_k, device="cuda")
    weights[:, 0] = 1.0
    out = b12x_layer.forward(x.bfloat16().cuda(), ids, weights).float().cpu()
    torch.cuda.synchronize()

    cos = torch.nn.functional.cosine_similarity(out.reshape(1, -1), ref.reshape(1, -1)).item()
    rel = (out - ref).norm().item() / ref.norm().item()
    assert cos > 0.985, f"cosine {cos}"
    assert rel < 0.16, f"rel {rel}"
    ratio = out.abs().max().item() / ref.abs().max().item()
    assert 0.8 < ratio < 1.2, f"magnitude ratio {ratio}"


@pytest.mark.skipif(not _HAS_CUDA_MOE, reason="requires the SM120 Flash-Next checkpoint")
def test_b12x_small_batch_stays_close_to_decode_rows(b12x_layer):
    """The M=4 optimization may change W4A4 rounding, but not semantics."""
    torch.manual_seed(2808)
    x = torch.randn(4, GEOM.hidden_size, dtype=torch.bfloat16, device="cuda")
    ids = torch.stack(
        [torch.randperm(GEOM.num_experts, device="cuda")[: GEOM.top_k] for _ in range(4)]
    ).to(torch.int32)
    weights = torch.rand(4, GEOM.top_k, dtype=torch.bfloat16, device="cuda")
    weights = (weights / weights.sum(dim=-1, keepdim=True)).contiguous()

    batched = b12x_layer.forward(x, ids.contiguous(), weights).clone()
    decode_rows = torch.cat(
        [
            b12x_layer.forward(
                x[row : row + 1],
                ids[row : row + 1].clone(),
                weights[row : row + 1].clone(),
            ).clone()
            for row in range(x.shape[0])
        ]
    )
    torch.cuda.synchronize()
    cosine = torch.nn.functional.cosine_similarity(batched.float(), decode_rows.float(), dim=-1)
    max_abs = (batched.float() - decode_rows.float()).abs().max()
    assert float(cosine.min()) > 0.998
    assert float(max_abs) < 0.02


@pytest.mark.skipif(not _HAS_CUDA_MOE, reason="requires the SM120 Flash-Next checkpoint")
def test_flashnext_prefill_moe_is_bitwise_stable(b12x_layer):
    """M=512 must keep deterministic route reduction across replays."""
    torch.manual_seed(2908)
    rows = 512
    x = torch.randn(rows, GEOM.hidden_size, dtype=torch.bfloat16, device="cuda")
    ids = torch.stack(
        [torch.randperm(GEOM.num_experts, device="cuda")[: GEOM.top_k] for _ in range(rows)]
    ).to(torch.int32)
    weights = torch.rand(rows, GEOM.top_k, dtype=torch.float32, device="cuda")
    weights = (weights / weights.sum(dim=-1, keepdim=True)).contiguous()

    outputs = [b12x_layer.forward(x, ids, weights).clone() for _ in range(4)]
    torch.cuda.synchronize()
    for output in outputs[1:]:
        assert torch.equal(outputs[0], output)


@pytest.mark.skipif(not _HAS_CUDA_MOE, reason="requires the SM120 Flash-Next checkpoint")
def test_flashnext_complete_mlp_prefill_graph_is_bitwise_exact(b12x_layer):
    torch.manual_seed(2908)
    mlp = FlashNextMlp(
        GEOM.hidden_size,
        GEOM.num_experts,
        GEOM.top_k,
        b12x_layer,
    ).cuda()
    mlp.shared = SharedExpert(
        GEOM.hidden_size,
        GEOM.shared_expert_intermediate_size,
    ).cuda()
    x = torch.randn(
        512,
        GEOM.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected = mlp(x).clone()
    mlp.capture_prefill_graph(512, pool=torch.cuda.graph_pool_handle())
    actual = mlp(x).clone()
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not _HAS_CUDA_MOE, reason="requires the SM120 Flash-Next checkpoint")
def test_flashnext_batched_rank_fastpath_matches_sort_fallback(b12x_layer):
    torch.manual_seed(2909)
    rows = 512
    x = torch.randn(rows, GEOM.hidden_size, dtype=torch.bfloat16, device="cuda")
    ids = torch.stack(
        [torch.randperm(GEOM.num_experts, device="cuda")[: GEOM.top_k] for _ in range(rows)]
    ).to(torch.int32)
    weights = torch.rand(rows, GEOM.top_k, dtype=torch.float32, device="cuda")
    weights = (weights / weights.sum(dim=-1, keepdim=True)).contiguous()

    os.environ["B12X_DYNAMIC_BATCHED_STABLE_RANKS"] = "0"
    fallback = b12x_layer.forward(x, ids, weights).clone()
    os.environ["B12X_DYNAMIC_BATCHED_STABLE_RANKS"] = "1"
    fast = b12x_layer.forward(x, ids, weights).clone()
    torch.cuda.synchronize()
    assert torch.equal(fast, fallback)


@pytest.mark.skipif(not _HAS_CUDA_MOE, reason="requires the SM120 Flash-Next checkpoint")
def test_flashnext_cutlass_prepare_shapes_match_kernel_contract():
    raw = load_flashnext_experts(CKPT, 0, GEOM, "cuda")
    prepared = prepare_flashnext_cutlass_experts(raw, GEOM, "cuda")
    assert prepared.w13_weight.shape == (
        GEOM.num_experts,
        2 * GEOM.moe_intermediate_size,
        GEOM.hidden_size // 16,
    )
    assert prepared.w2_weight.shape == (
        GEOM.num_experts,
        GEOM.hidden_size,
        GEOM.moe_intermediate_size // 16,
    )
    assert len(prepared.quant_scales) == 6
    assert prepared.quant_scales[0].shape == torch.Size([])
    assert prepared.quant_scales[1].dtype == torch.int32
    assert prepared.quant_scales[2].shape == (GEOM.num_experts,)
    assert prepared.quant_scales[3].shape == torch.Size([])
    assert prepared.quant_scales[4].dtype == torch.int32
    assert prepared.quant_scales[5].shape == (GEOM.num_experts,)


@pytest.mark.skipif(not _HAS_CUDA_MOE, reason="requires the SM120 Flash-Next checkpoint")
def test_flashnext_cutlass_layer_stays_close_to_b12x(b12x_layer):
    ops = load_flashinfer_cutlass_moe_ops()
    if ops is None:
        pytest.skip("flashinfer CUTLASS MoE unavailable")
    raw = load_flashnext_experts(CKPT, 0, GEOM, "cuda")
    cutlass_layer = FlashInferMoELayer(
        prepare_flashnext_cutlass_experts(raw, GEOM, "cuda"),
        "cuda",
        output_arena=SparkinferMoEOutputArena(GEOM.hidden_size),
    )

    torch.manual_seed(2910)
    rows = 512
    x = torch.randn(rows, GEOM.hidden_size, dtype=torch.bfloat16, device="cuda")
    ids = torch.stack(
        [torch.randperm(GEOM.num_experts, device="cuda")[: GEOM.top_k] for _ in range(rows)]
    ).to(torch.int32)
    weights = torch.rand(rows, GEOM.top_k, dtype=torch.float32, device="cuda")
    weights = (weights / weights.sum(dim=-1, keepdim=True)).contiguous()

    cutlass = cutlass_layer.forward(x, ids, weights).float()
    b12x = b12x_layer.forward(x, ids, weights).float()
    torch.cuda.synchronize()
    cosine = torch.nn.functional.cosine_similarity(cutlass, b12x, dim=-1)
    max_abs = (cutlass - b12x).abs().max()
    rel = (cutlass - b12x).norm() / b12x.norm()
    assert float(cosine.min()) > 0.998
    assert float(max_abs) < 0.05
    assert float(rel) < 0.03


@pytest.mark.skipif(not _HAS_CUDA_MOE, reason="requires the SM120 Flash-Next checkpoint")
def test_flashnext_cutlass_prefill_batch_is_replay_stable():
    ops = load_flashinfer_cutlass_moe_ops()
    if ops is None:
        pytest.skip("flashinfer CUTLASS MoE unavailable")
    raw = load_flashnext_experts(CKPT, 0, GEOM, "cuda")
    cutlass_layer = FlashInferMoELayer(
        prepare_flashnext_cutlass_experts(raw, GEOM, "cuda"),
        "cuda",
        output_arena=SparkinferMoEOutputArena(GEOM.hidden_size),
    )

    torch.manual_seed(2911)
    rows = 512
    x = torch.randn(rows, GEOM.hidden_size, dtype=torch.bfloat16, device="cuda")
    ids = torch.stack(
        [torch.randperm(GEOM.num_experts, device="cuda")[: GEOM.top_k] for _ in range(rows)]
    ).to(torch.int32)
    weights = torch.rand(rows, GEOM.top_k, dtype=torch.float32, device="cuda")
    weights = (weights / weights.sum(dim=-1, keepdim=True)).contiguous()

    outputs = [cutlass_layer.forward(x, ids, weights).clone() for _ in range(3)]
    torch.cuda.synchronize()
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[0], outputs[2])
