from __future__ import annotations

import pytest

from bfdiag.workloads import (
    HISTORICAL_DFLASH_M16_CONTEXT_TOKENS,
    HISTORICAL_DFLASH_PREFIX_BASE_TOKENS,
    HISTORICAL_DFLASH_PREFIX_CONTEXT_TOKENS,
    HISTORICAL_DFLASH_PREFIX_SUFFIX_TOKENS,
    HISTORICAL_M1_CONTEXT_TOKENS,
    HISTORICAL_M1_SUFFIX_TOKENS,
    _tensor_content_hash,
    capture_dynamic_route_tile_trace,
    capture_laguna_route_histograms,
    capture_laguna_target_logits_oracle,
    check_b1_metadata_fastpath_parity,
    check_dflash_server_step_parity,
    check_dflash_verify_cg_parity,
    diagnose_dflash_verify_cg_divergence,
    historical_dflash_m16_prompt_ids,
    historical_dflash_prefix_prompt_ids,
    historical_m1_prompt_ids,
    profile_historical_dflash_m16,
    profile_historical_m1_decode_cg,
    profile_laguna_target_shape_matrix,
    profile_rms_norm_m16,
    reset_dflash_workload_state,
    restore_dflash_daemon_state,
    summarize_dynamic_route_tile_trace,
    summarize_moe_route_ids,
    summarize_sparkinfer_workspace_pools,
)


def test_tensor_content_hash_is_value_and_layout_stable():
    torch = pytest.importorskip("torch")
    contiguous = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    non_contiguous = contiguous.t()

    assert _tensor_content_hash(contiguous) == _tensor_content_hash(contiguous.clone())
    assert _tensor_content_hash(non_contiguous) != _tensor_content_hash(contiguous)


def test_historical_m1_prompt_is_fixed_and_repeating():
    prompt = historical_m1_prompt_ids(205)
    assert prompt[:4] == [1000, 1001, 1002, 1003]
    assert prompt[99:102] == [1099, 1000, 1001]
    assert (
        len(historical_m1_prompt_ids(HISTORICAL_M1_CONTEXT_TOKENS)) == HISTORICAL_M1_CONTEXT_TOKENS
    )


def test_workspace_audit_attributes_a_shared_arena_by_view() -> None:
    class Tensor:
        def numel(self):
            return 100

        def element_size(self):
            return 2

    class Plan:
        implementation = "dynamic"
        deterministic_output = True
        routed_rows = 20
        num_topk = 10

    class Arena:
        plan = Plan()
        shared_arena = Tensor()

    class Pool:
        core_arenas = {"key": Arena()}

    report = summarize_sparkinfer_workspace_pools(
        [Pool(), Pool()],
        view_mapper=lambda _plan: ({"name": "route_output", "nbytes": 120},),
    )

    assert report == {
        "pool_count": 2,
        "core_arenas": [
            {
                "workspace_key": "'key'",
                "arena_nbytes": 200,
                "implementation": "dynamic",
                "deterministic_output": True,
                "routed_rows": 20,
                "num_topk": 10,
                "views": [{"name": "route_output", "nbytes": 120}],
            },
            {
                "workspace_key": "'key'",
                "arena_nbytes": 200,
                "implementation": "dynamic",
                "deterministic_output": True,
                "routed_rows": 20,
                "num_topk": 10,
                "views": [{"name": "route_output", "nbytes": 120}],
            },
        ],
    }
    assert len(historical_m1_prompt_ids(HISTORICAL_M1_SUFFIX_TOKENS)) == HISTORICAL_M1_SUFFIX_TOKENS


def test_dynamic_route_tile_trace_reports_exact_order_dependency_cycles() -> None:
    trace = summarize_dynamic_route_tile_trace(
        token_map=[0, 3, 2, 1],
        expert_row_counts=[2, 2],
        expert_tile_base=[0, 1, 2],
        physical_tiles_capacity=2,
        num_topk=2,
    )

    assert trace == {
        "routed_rows": 4,
        "tile_m": 2,
        "active_tiles": 2,
        "dependency_edges": 2,
        "cyclic_components": 1,
        "largest_cyclic_component_tiles": 2,
        "largest_cyclic_component_route_rows": 4,
    }


def test_capture_dynamic_route_tile_trace_requires_a_live_backend() -> None:
    class EmptyBackend:
        _moe_sparkinfer_layers = ()

    with pytest.raises(RuntimeError, match="no SparkInfer MoE layers"):
        capture_dynamic_route_tile_trace(EmptyBackend())


def test_historical_m1_prompt_rejects_negative_length():
    with pytest.raises(ValueError, match="non-negative"):
        historical_m1_prompt_ids(-1)


def test_historical_dflash_prompt_is_tokenizer_derived_and_fixed_length():
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert text == "The quick brown fox jumps over the lazy dog. "
            assert add_special_tokens is False
            return [41, 42, 43]

    assert historical_dflash_m16_prompt_ids(Tokenizer(), 8) == [41, 42, 43, 41, 42, 43, 41, 42]
    assert (
        len(historical_dflash_m16_prompt_ids(Tokenizer())) == HISTORICAL_DFLASH_M16_CONTEXT_TOKENS
    )


def test_historical_dflash_prompt_rejects_invalid_length_or_empty_encoding():
    class EmptyTokenizer:
        def encode(self, text, *, add_special_tokens):
            return []

    with pytest.raises(ValueError, match="non-negative"):
        historical_dflash_m16_prompt_ids(EmptyTokenizer(), -1)
    with pytest.raises(ValueError, match="empty"):
        historical_dflash_m16_prompt_ids(EmptyTokenizer())


def test_historical_prefix_cache_prompt_matches_archived_base_suffix_contract():
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return [11, 12] if text.startswith("The quick brown fox") else [21, 22, 23]

    base_ids, full_ids = historical_dflash_prefix_prompt_ids(Tokenizer())
    assert len(base_ids) == HISTORICAL_DFLASH_PREFIX_BASE_TOKENS
    assert len(full_ids) == HISTORICAL_DFLASH_PREFIX_CONTEXT_TOKENS
    assert full_ids[:4] == [11, 12, 11, 12]
    assert full_ids[HISTORICAL_DFLASH_PREFIX_BASE_TOKENS:][:6] == [21, 22, 23, 21, 22, 23]
    assert len(full_ids) - len(base_ids) == HISTORICAL_DFLASH_PREFIX_SUFFIX_TOKENS


def test_dflash_workload_reset_undoes_m1_patch_before_clearing_kv(monkeypatch):
    calls = []

    class Backend:
        def _unpatch_impls_for_prefill(self):
            calls.append("unpatch")

    class Engine:
        backend = Backend()

    monkeypatch.setattr(
        "bfdiag.daemon.session.reset_laguna_engine",
        lambda engine: calls.append("reset_kv"),
    )
    reset_dflash_workload_state(Engine())
    assert calls == ["unpatch", "reset_kv"]


def test_dflash_daemon_restore_repatches_after_reset(monkeypatch):
    calls = []

    class Backend:
        def _unpatch_impls_for_prefill(self):
            calls.append("unpatch")

        def _repatch_impls_for_cg(self):
            calls.append("repatch")

    class Engine:
        backend = Backend()

    monkeypatch.setattr(
        "bfdiag.daemon.session.reset_laguna_engine",
        lambda engine: calls.append("reset_kv"),
    )
    restore_dflash_daemon_state(Engine())
    assert calls == ["unpatch", "reset_kv", "repatch"]


def test_dflash_server_step_parity_rejects_a_single_token_budget():
    with pytest.raises(ValueError, match="at least two"):
        check_dflash_server_step_parity(object(), object(), max_tokens=1)


def test_dflash_verify_cg_parity_rejects_non_positive_steps():
    with pytest.raises(ValueError, match="positive"):
        check_dflash_verify_cg_parity(object(), object(), steps=0)


def test_dflash_verify_cg_diagnosis_rejects_negative_prefix_steps():
    with pytest.raises(ValueError, match="non-negative"):
        diagnose_dflash_verify_cg_divergence(object(), object(), prefix_steps=-1)


def test_m1_profiler_requires_a_captured_graph():
    class Backend:
        _decode_cg = None

    with pytest.raises(RuntimeError, match="capture"):
        profile_historical_m1_decode_cg(Backend())


def test_dflash_profiler_rejects_invalid_round_count():
    with pytest.raises(ValueError, match="positive"):
        profile_historical_dflash_m16(object(), object(), profile_rounds=0)
    with pytest.raises(ValueError, match="non-negative"):
        profile_historical_dflash_m16(object(), object(), warmup_rounds=-1)


def test_target_shape_matrix_rejects_invalid_shape_contract():
    with pytest.raises(ValueError, match="positive token"):
        profile_laguna_target_shape_matrix(object(), object(), shapes=(0,))
    with pytest.raises(ValueError, match="unique and sorted"):
        profile_laguna_target_shape_matrix(object(), object(), shapes=(2, 1))
    with pytest.raises(ValueError, match="replays_per_shape"):
        profile_laguna_target_shape_matrix(object(), object(), replays_per_shape=0)
    with pytest.raises(TypeError, match="bool or None"):
        profile_laguna_target_shape_matrix(object(), object(), nvfp4_split_decode=1)


def test_target_shape_matrix_rejects_insufficient_live_capacity():
    class ModelConfig:
        max_model_len = HISTORICAL_DFLASH_M16_CONTEXT_TOKENS + 16

    class RuntimeConfig:
        model_config = ModelConfig()

    class Backend:
        block_size = 64
        blocks_per_slot = 1_024
        runtime_config = RuntimeConfig()

    class Engine:
        backend = Backend()

    with pytest.raises(ValueError, match="at least 1025 blocks_per_slot"):
        profile_laguna_target_shape_matrix(Engine(), object(), shapes=(16,))


def test_moe_route_summary_counts_every_topk_pair() -> None:
    report = summarize_moe_route_ids([[[1, 2], [2, 2]]])

    assert report["moe_layer_count"] == 1
    assert report["mean_unique_experts"] == 2
    assert report["layers"] == [
        {
            "moe_layer": 1,
            "routed_pairs": 4,
            "unique_experts": 2,
            "max_pairs_per_expert": 3,
            "expert_pair_counts": [0, 1, 3] + [0] * 253,
        }
    ]


def test_route_histogram_rejects_non_dynamic_shapes():
    with pytest.raises(ValueError, match="dynamic serving"):
        capture_laguna_route_histograms(object(), object(), shapes=(6,))


def test_target_logits_oracle_requires_a_valid_contract():
    with pytest.raises(ValueError, match="variant"):
        capture_laguna_target_logits_oracle(object(), object(), variant="", token_count=1)
    with pytest.raises(ValueError, match="token_count"):
        capture_laguna_target_logits_oracle(object(), object(), variant="m1", token_count=0)
def test_dflash_profiler_reports_the_quick_prompt_contract_for_bad_geometry():
    class Backend:
        block_size = 1
        blocks_per_slot = 1

        class RuntimeConfig:
            class ModelConfig:
                max_model_len = 1

            model_config = ModelConfig()

        runtime_config = RuntimeConfig()

    class Engine:
        backend = Backend()

    with pytest.raises(ValueError, match="dflash-m16-64k-quick-brown-fox"):
        profile_historical_dflash_m16(Engine(), object(), profile_rounds=1)


def test_rms_norm_profiler_rejects_invalid_arguments():
    with pytest.raises(ValueError, match="iterations"):
        profile_rms_norm_m16(iterations=0)
    with pytest.raises(ValueError, match="positive"):
        profile_rms_norm_m16(num_warps_variants=(0,))


def test_b1_metadata_fastpath_parity_rejects_invalid_token_count():
    with pytest.raises(ValueError, match="positive"):
        check_b1_metadata_fastpath_parity(object(), tokens=0)
