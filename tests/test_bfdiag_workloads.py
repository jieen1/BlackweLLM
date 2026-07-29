from __future__ import annotations

import pytest

from bfdiag.workloads import (
    HISTORICAL_DFLASH_M16_CONTEXT_TOKENS,
    HISTORICAL_DFLASH_PREFIX_BASE_TOKENS,
    HISTORICAL_DFLASH_PREFIX_CONTEXT_TOKENS,
    HISTORICAL_DFLASH_PREFIX_SUFFIX_TOKENS,
    HISTORICAL_M1_CONTEXT_TOKENS,
    HISTORICAL_M1_SUFFIX_TOKENS,
    check_b1_metadata_fastpath_parity,
    check_dflash_server_step_parity,
    check_dflash_verify_cg_parity,
    diagnose_dflash_verify_cg_divergence,
    historical_dflash_m16_prompt_ids,
    historical_dflash_prefix_prompt_ids,
    historical_m1_prompt_ids,
    profile_historical_dflash_m16,
    profile_historical_m1_decode_cg,
    profile_rms_norm_m16,
    reset_dflash_workload_state,
)


def test_historical_m1_prompt_is_fixed_and_repeating():
    prompt = historical_m1_prompt_ids(205)
    assert prompt[:4] == [1000, 1001, 1002, 1003]
    assert prompt[99:102] == [1099, 1000, 1001]
    assert (
        len(historical_m1_prompt_ids(HISTORICAL_M1_CONTEXT_TOKENS)) == HISTORICAL_M1_CONTEXT_TOKENS
    )
    assert len(historical_m1_prompt_ids(HISTORICAL_M1_SUFFIX_TOKENS)) == HISTORICAL_M1_SUFFIX_TOKENS


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
