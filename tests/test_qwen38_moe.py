"""Qwen MoE family block: production parity, graph parity, family geometry.

Day-0 prep for Qwen3.8-Flash-Next (see
notes/2026-08-26-qwen38-flash-next-day0-survey.md). Three claims:

* the geometry-parameterized expert preparation reproduces the proven
  production ``prepare_sparkinfer_layer`` path BIT-FOR-BIT at Laguna's own
  geometry (256 / 3072 / 1024);
* ``QwenMoeLayer`` (gate -> Triton softmax router -> b12x experts + shared
  expert) is CUDA-Graph capture-safe: replayed output is bit-identical to
  the eager output;
* the Flash-Next family geometry (512 / 8192 / 2048) plans, prepares, and
  runs end to end through the assembled layer.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
    pytest.skip("Qwen MoE block tests require an SM120 CUDA device", allow_module_level=True)

from runtime.backends.laguna_sparkinfer_moe import (  # noqa: E402
    SparkinferMoELayer,
    prepare_sparkinfer_layer,
)
from runtime.backends.qwen38_sparkinfer_moe import (  # noqa: E402
    allocate_tp_moe_workspace_pool,
    make_qwen_moe_expert_layer,
    prepare_qwen_moe_experts,
)
from runtime.model.qwen38_moe import QwenMoeGeometry, QwenMoeLayer  # noqa: E402

LAGUNA = QwenMoeGeometry(
    num_experts=256,
    top_k=10,
    hidden_size=3072,
    moe_intermediate_size=1024,
    shared_expert_intermediate_size=1024,
)
FLASH_NEXT = QwenMoeGeometry(
    num_experts=512,
    top_k=10,
    hidden_size=8192,
    moe_intermediate_size=2048,
    shared_expert_intermediate_size=2048,
)


@pytest.fixture(autouse=True)
def _release_moe_cuda_cache():
    yield
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _synthetic_raw(geometry: QwenMoeGeometry, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    e, h, i = geometry.num_experts, geometry.hidden_size, geometry.moe_intermediate_size
    raw: dict[str, torch.Tensor] = {}
    for name in ("gate_w", "up_w"):
        raw[name] = torch.randint(
            0, 255, (e, i, h // 2), dtype=torch.uint8, device="cuda", generator=g
        )
    raw["down_w"] = torch.randint(
        0, 255, (e, h, i // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    for name in ("gate_sf", "up_sf"):
        raw[name] = (torch.rand((e, i, h // 16), device="cuda", generator=g) + 0.5).to(
            torch.float8_e4m3fn
        )
    raw["down_sf"] = (torch.rand((e, h, i // 16), device="cuda", generator=g) + 0.5).to(
        torch.float8_e4m3fn
    )
    for name in ("gate_gs", "down_gs"):
        raw[name] = torch.rand(e, dtype=torch.float32, device="cuda", generator=g) + 0.5
    return raw


def test_generalized_prepare_matches_production_bit_for_bit():
    raw = _synthetic_raw(LAGUNA)
    a = torch.randn(7, LAGUNA.hidden_size, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(0, LAGUNA.num_experts, (7, LAGUNA.top_k), dtype=torch.int32, device="cuda")
    weights = torch.softmax(torch.randn(7, LAGUNA.top_k, device="cuda"), dim=-1).float()

    prod_experts = prepare_sparkinfer_layer(raw, "cuda", a1_gscale=0.0005, a2_gscale=0.0013)
    new_experts = prepare_qwen_moe_experts(raw, LAGUNA, "cuda", a1_gscale=0.0005, a2_gscale=0.0013)
    workspace = allocate_tp_moe_workspace_pool()
    prod = SparkinferMoELayer(prod_experts, workspace, "cuda")
    new = SparkinferMoELayer(new_experts, workspace, "cuda")

    r_prod = prod.forward(a, ids, weights)
    r_new = new.forward(a, ids, weights)
    torch.cuda.synchronize()
    torch.testing.assert_close(r_new, r_prod, rtol=0, atol=0)


def _seeded_layer(geometry: QwenMoeGeometry, max_rows: int) -> QwenMoeLayer:
    torch.manual_seed(0)
    layer = QwenMoeLayer(geometry, max_rows=max_rows)
    for p in layer.parameters():
        p.data.normal_(0.0, 0.02)
    raw = _synthetic_raw(geometry)
    expert = make_qwen_moe_expert_layer(
        raw,
        geometry,
        allocate_tp_moe_workspace_pool(),
        a1_gscale=0.0005,
        a2_gscale=0.0013,
    )
    layer.attach_experts(expert)
    return layer


def test_moe_layer_cuda_graph_replay_is_bit_identical():
    layer = _seeded_layer(LAGUNA, max_rows=8)
    x = torch.randn(4, LAGUNA.hidden_size, dtype=torch.bfloat16, device="cuda")

    eager = layer(x)
    torch.cuda.synchronize()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            layer(x)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = layer(x)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


def test_moe_layer_flash_next_geometry_smoke():
    layer = _seeded_layer(FLASH_NEXT, max_rows=4)
    x = torch.randn(1, FLASH_NEXT.hidden_size, dtype=torch.bfloat16, device="cuda")
    out = layer(x)
    torch.cuda.synchronize()
    assert tuple(out.shape) == (1, FLASH_NEXT.hidden_size)
    assert torch.isfinite(out.float()).all().item()


def test_moe_layer_rejects_oversized_batch():
    layer = _seeded_layer(LAGUNA, max_rows=2)
    x = torch.randn(3, LAGUNA.hidden_size, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="router arena"):
        layer(x)


def test_moe_layer_requires_attached_experts():
    layer = QwenMoeLayer(LAGUNA, max_rows=2)
    x = torch.randn(1, LAGUNA.hidden_size, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="not attached"):
        layer(x)
