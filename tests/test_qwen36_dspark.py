"""Construction and checkpoint-name tests for the Qwen3.8 DSpark draft."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fla")
pytest.importorskip("b12x")

from runtime.backends.qwen36_dspark import (  # noqa: E402
    Qwen36DSparkEngine,
    _flatten_target_taps_ragged,
)
from runtime.dspark_config import DSparkDraftConfig  # noqa: E402
from runtime.model.qwen36_dspark import (  # noqa: E402
    Qwen36DSparkDraftForCausalLM,
)


def _tiny_config() -> DSparkDraftConfig:
    return DSparkDraftConfig.from_dict(
        {
            "architectures": ["DSparkDraftModel"],
            "block_size": 3,
            "dflash_config": {
                "attention_mode": "gqa",
                "markov_rank": 4,
                "mask_token_id": 47,
                "projector_type": "dspark",
                "target_layer_ids": [1, 3],
            },
            "enable_confidence_head": True,
            "hidden_size": 32,
            "intermediate_size": 64,
            "layer_types": ["full_attention", "full_attention"],
            "max_position_embeddings": 64,
            "model_type": "qwen3",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "markov_head_type": "vanilla",
            "markov_rank": 4,
            "rms_norm_eps": 1e-6,
            "vocab_size": 50,
            "rope_parameters": {
                "rope_theta": 10000,
                "rope_type": "default",
                "partial_rotary_factor": 1.0,
            },
        }
    )


def test_official_flat_parameter_shapes_are_represented() -> None:
    model = Qwen36DSparkDraftForCausalLM(_tiny_config(), target_layer_count=4)

    assert tuple(model.model.fc.weight.shape) == (32, 64)
    assert tuple(model.markov_head.markov_w1.weight.shape) == (50, 4)
    assert tuple(model.markov_head.markov_w2.weight.shape) == (50, 4)
    assert tuple(model.confidence_head.proj.weight.shape) == (1, 36)
    assert tuple(model.confidence_head.proj.bias.shape) == (1,)


def test_qkv_shards_and_dspark_heads_load_from_official_names() -> None:
    model = Qwen36DSparkDraftForCausalLM(_tiny_config(), target_layer_count=4)
    weights = [
        ("layers.0.self_attn.q_proj.weight", torch.zeros(32, 32)),
        ("layers.0.self_attn.k_proj.weight", torch.zeros(16, 32)),
        ("layers.0.self_attn.v_proj.weight", torch.zeros(16, 32)),
        ("fc.weight", torch.zeros(32, 64)),
        ("markov_head.markov_w1.weight", torch.zeros(50, 4)),
        ("markov_head.markov_w2.weight", torch.zeros(50, 4)),
        ("confidence_head.proj.weight", torch.zeros(1, 36)),
        ("confidence_head.proj.bias", torch.zeros(1)),
    ]

    loaded = model.load_weights(weights)

    assert "model.layers.0.self_attn.qkv_proj.weight" in loaded
    assert "model.fc.weight" in loaded
    assert "markov_head.markov_w1.weight" in loaded
    assert "confidence_head.proj.bias" in loaded
    assert torch.count_nonzero(model.model.layers[0].self_attn.qkv_proj.weight) == 0


def test_greedy_markov_block_has_gamma_rows() -> None:
    model = Qwen36DSparkDraftForCausalLM(_tiny_config(), target_layer_count=4)
    hidden = torch.zeros(1, model.gamma, 32)
    tokens, logits, confidence = model.sample_greedy_block(hidden, anchor_tokens=torch.tensor([1]))

    assert tokens.shape == (1, 3)
    assert logits.shape == (1, 3, 50)
    assert confidence is not None and confidence.shape == (1, 3)


def test_shared_callable_lm_head_path_is_used() -> None:
    model = Qwen36DSparkDraftForCausalLM(_tiny_config(), target_layer_count=4)
    shared_head = torch.nn.Linear(32, 50, bias=False)
    model.lm_head = shared_head
    hidden = torch.randn(1, 3, 32)

    assert torch.equal(model.compute_base_logits(hidden), shared_head(hidden))


def test_cuda_graph_health_is_vacuously_true_before_capture() -> None:
    engine = Qwen36DSparkEngine.__new__(Qwen36DSparkEngine)

    engine.cg_status = {}
    assert engine.cuda_graphs_healthy() is True

    engine.cg_status = {"draft": "captured", "verify": "captured"}
    assert engine.cuda_graphs_healthy() is True

    engine.cg_status["verify"] = "failed"
    assert engine.cuda_graphs_healthy() is False


def test_compact_without_dynamic_policy_uses_full_ragged_width() -> None:
    engine = Qwen36DSparkEngine.__new__(Qwen36DSparkEngine)
    engine.verify_mode = "compact"
    engine._dynamic_planner = False
    engine.k = 7

    assert engine._verify_widths([0, 1, 2, 3]) == [7, 7, 7, 7]


def test_ragged_target_taps_skip_padded_request_tail() -> None:
    tap0 = torch.arange(12, dtype=torch.float32).view(6, 2)
    tap1 = tap0 + 100

    compact = _flatten_target_taps_ragged(
        [tap0, tap1],
        batch_size=2,
        accepted_counts=[2, 1],
        verify_lens=[4, 2],
        expected_features=4,
    )

    assert compact.tolist() == [
        [0.0, 1.0, 100.0, 101.0],
        [2.0, 3.0, 102.0, 103.0],
        [8.0, 9.0, 108.0, 109.0],
    ]


def test_prefix_lifecycle_preserves_and_copies_the_draft_kv_family() -> None:
    engine = Qwen36DSparkEngine.__new__(Qwen36DSparkEngine)
    engine.backend = SimpleNamespace(num_slots=2)
    engine.page_size = 4
    engine.pages_per_slot = 3
    engine.max_seq_len = 12
    engine._draft_kv_len = [8, 0]
    engine._cached_prefix_len = [0, 0]
    cache = torch.zeros((2, 6, 4, 1, 1), dtype=torch.uint8)
    cache[:, :3].fill_(7)
    engine._draft_kv_caches = {"layer": cache}

    assert engine.preserve_prefix(0, 8) is True
    assert engine.can_restore_prefix(0, 8) is True
    engine._draft_kv_len[0] = 0

    engine.copy_prefix(0, 1, 8)

    assert engine._draft_kv_len == [0, 8]
    assert engine._cached_prefix_len == [8, 0]
    assert torch.equal(cache[:, 3:5], cache[:, :2])
    assert torch.count_nonzero(cache[:, 5]) == 0
    assert engine.can_restore_prefix(1, 8) is False

    assert engine.preserve_prefix(1, 9) is False
    assert engine.can_restore_prefix(1, 8) is False


def test_persistent_prefix_snapshot_round_trips_through_the_draft_scratch_row() -> None:
    engine = Qwen36DSparkEngine.__new__(Qwen36DSparkEngine)
    engine.backend = SimpleNamespace(num_slots=2)
    engine.page_size = 4
    engine.pages_per_slot = 3
    engine.max_seq_len = 12
    engine.scratch_row = 2
    engine.device = torch.device("cpu")
    engine._draft_kv_len = [8, 0]
    engine._cached_prefix_len = [8, 0]
    engine._scratch_valid_pages = set()
    cache = torch.zeros((2, 9, 4, 1, 1), dtype=torch.uint8)
    cache[:, :2].fill_(11)
    engine._draft_kv_caches = {"layer": cache}

    assert engine.snapshot_prefix_to_scratch(0, 8, scratch_pages=(1, 2)) is True
    cache[:, :2].zero_()
    assert engine.restore_prefix_from_scratch(1, 8, scratch_pages=(1, 2)) is True
    assert torch.count_nonzero(cache[:, 3:5] - 11) == 0

    engine.release_prefix_scratch((1, 2))
    assert engine.restore_prefix_from_scratch(1, 8, scratch_pages=(1, 2)) is False
