"""The installed sparkinfer must size w4a16 scratch for the decode path we use.

`Qwen36MLP` routes its NVFP4 MLP through sparkinfer's fused w4a16 kernel as a
degenerate 1-expert / top-1 MoE, and relies on `plan_w4a16_buffers` to size
the fc1/fc2 `c_tmp` scratch. Two kernel modes write into that one buffer and
they need different amounts:

* the packed/grouped route path shares a block between routes hitting the
  same expert, so `max_packed_route_slots` bounds it tightly;
* decode's direct-topk / TC-decode fast path never packs, so every routed row
  reserves a full `block_size_m` tile -- `m * topk * block_size_m`.

`plan_w4a16_buffers` sized only for the first. At this deployment's decode
shape that is 9 slots against a real need of 16, and the two failure modes
diverge sharply: eager silently absorbs the shortfall through a fallback
allocation, while CUDA Graph capture correctly refuses. The observable
symptom was therefore not a crash but capture failing and the server quietly
running decode in eager -- slower, with nothing in the logs saying so.

Fixed in sparkinfer by taking the union of the two bounds, and BlackweLLM's
own persistent-scratch workaround was removed on the strength of that fix.

Which leaves an ordering hazard worth guarding: this repo's `main` now
*depends* on a fixed sparkinfer, and nothing else here would notice an
unfixed one. No test exercises `torch.cuda.graph` around `Qwen36MLP` -- that
needs a GPU and the full model -- so a stale `BF_SPARKINFER_PATH`, a reverted
branch, or a fresh clone would reintroduce silent eager fallback with every
gate still green.

sparkinfer has its own capacity tests; this one is deliberately duplicative
in subject and different in direction. Theirs assert the allocator is
self-consistent across a broad shape grid, and travel with the code that
could break it. This one asserts *the sparkinfer this repo is pointed at*
satisfies the requirement `Qwen36MLP` actually places on it, at this model's
real dimensions -- so it fails on a correct-but-old checkout, which is the
case their suite cannot see and the one that bit here.

Arithmetic over sparkinfer's public API: no GPU, no model, no real weights.
Skips where sparkinfer is absent, which is the CI job that has no torch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

sparkinfer_host = pytest.importorskip(
    "sparkinfer.moe._shared.kernels.w4a16.host",
    reason="sparkinfer is not installed in the torch-free CI job",
)

max_packed_route_slots = sparkinfer_host.max_packed_route_slots
packed_gemm_scratch_elements = sparkinfer_host.packed_gemm_scratch_elements
plan_w4a16_buffers = sparkinfer_host.plan_w4a16_buffers

# Qwen3.6-27B's MLP as `Qwen36MLP` hands it to the MoE kernel: hidden 5120,
# intermediate 17408, gated (fc1 emits a fused gate+up), one expert, top-1.
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408
SMS = 128


def _prepared() -> SimpleNamespace:
    """The fields `plan_w4a16_buffers` reads off prepared weights."""
    return SimpleNamespace(
        num_experts=1,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        is_gated=True,
    )


def _direct_route_slots(m: int, topk: int, block_size_m: int) -> int:
    """What decode's direct-topk path reserves: no packing, one tile per row.

    Mirrors `run_w4a16_moe`'s `route_slots_for_scratch`, which the packed-mode
    branch overwrites but the direct/TC-decode path leaves at this value.
    """
    return m * topk * block_size_m


def _scratch_needed(*, fc1_cols: int, route_slots: int, block_size_m: int) -> tuple[int, int]:
    """fc1/fc2 `c_tmp` elements the kernel launches with for a given slot count."""
    return (
        packed_gemm_scratch_elements(
            size_n=fc1_cols, route_slots=route_slots, moe_block_size=block_size_m, sms=SMS
        ),
        packed_gemm_scratch_elements(
            size_n=HIDDEN_SIZE, route_slots=route_slots, moe_block_size=block_size_m, sms=SMS
        ),
    )


class TestAllocatorCoversDecode:
    """`plan_w4a16_buffers` -- the function the fix changed -- at our shapes."""

    @pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
    @pytest.mark.parametrize("topk", [1])
    def test_planned_scratch_covers_the_direct_topk_path(self, m, topk):
        plan = plan_w4a16_buffers(_prepared(), m=m, topk=topk, route_num_experts=1, sms=SMS)
        needed_fc1, needed_fc2 = _scratch_needed(
            fc1_cols=plan.fc1_cols,
            route_slots=_direct_route_slots(m, topk, plan.block_size_m),
            block_size_m=plan.block_size_m,
        )
        context = (
            f"m={m} topk={topk} block_size_m={plan.block_size_m} -- CUDA Graph "
            "capture of Qwen36MLP fails and decode silently falls back to eager. "
            "Is the sparkinfer at BF_SPARKINFER_PATH missing the w4a16 scratch "
            "union fix?"
        )
        assert plan.fc1_c_tmp_elements >= needed_fc1, (
            f"fc1_c_tmp_elements={plan.fc1_c_tmp_elements} < required {needed_fc1} at {context}"
        )
        assert plan.fc2_c_tmp_elements >= needed_fc2, (
            f"fc2_c_tmp_elements={plan.fc2_c_tmp_elements} < required {needed_fc2} at {context}"
        )

    def test_the_deployment_shape_that_broke_capture(self):
        """Pin the live failure: the packed bound gives 9 slots, decode needs 16.

        Decode batch 2, one expert, top-1. If this stops under-counting, the
        union is no longer load-bearing at our shape and this file has stopped
        covering the regression it was written for.
        """
        assert max_packed_route_slots(2 * 1, 8, 1) == 9
        assert _direct_route_slots(m=2, topk=1, block_size_m=8) == 16

    @pytest.mark.parametrize("m", [1, 2, 3, 4, 8])
    def test_the_union_is_free_not_a_trade(self, m):
        """Packing can only ever need fewer slots than the direct path, never more.

        This is what makes `max(packed, direct)` safe to adopt: covering decode
        costs the packed path nothing.
        """
        for block_size_m in (8, 16, 32, 64):
            assert max_packed_route_slots(m, block_size_m, 1) <= _direct_route_slots(
                m, 1, block_size_m
            )
