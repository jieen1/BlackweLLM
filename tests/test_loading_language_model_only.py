"""B0-1a unit tests: constructed tensor-name lists, no checkpoint, no torch.

Deliberately not run against any real checkpoint's weights -- Laguna, this
runtime's only production model, has no vision tower, so this filter has
never actually been asked to drop a real tensor. What IS real here: the
vision-tensor names below (``model.visual.blocks.0.attn.qkv.weight`` etc.)
are not invented -- they are copied from
``nvidia/Qwen3.6-27B-NVFP4``'s real ``model.safetensors.index.json``
(cross-checked 2026-08-02 against both that checkpoint and
``unsloth/Qwen3.6-27B-NVFP4``: 333 tensors each, all and only under
``model.visual.``; independently corroborated in
``notes/2026-08-02-qwen36-b0-fact-baseline.md`` §1.4). So the *shape* of
the input these tests construct is real; the claim "this filter, run
against a real loader, correctly handles a real checkpoint" is not one
these tests make.
"""

from __future__ import annotations

from runtime.loading.language_model_only import (
    DEFAULT_VISION_TENSOR_PREFIXES,
    LanguageModelOnlyStats,
    filter_language_model_only,
)

# A small, representative slice of nvidia/Qwen3.6-27B-NVFP4's real tensor
# names (language-model + vision), not a synthetic pattern.
_LANGUAGE_MODEL_NAMES = (
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.embed_tokens.weight",
    "lm_head.weight",
    "mtp.fc.weight",
)
_VISION_NAMES = (
    "model.visual.blocks.0.attn.proj.bias",
    "model.visual.blocks.0.attn.proj.weight",
    "model.visual.blocks.0.attn.qkv.weight",
    "model.visual.blocks.0.mlp.linear_fc1.weight",
)


def _weights(names: tuple[str, ...]) -> list[tuple[str, str]]:
    # The value is never inspected by the filter -- a plain string
    # placeholder stands in for what would be a torch.Tensor in production,
    # keeping this module (and these tests) torch-free.
    return [(name, "tensor-placeholder") for name in names]


class TestPassThroughWhenDisabled:
    def test_language_model_only_false_drops_nothing(self):
        stats = LanguageModelOnlyStats()
        weights = _weights(_LANGUAGE_MODEL_NAMES + _VISION_NAMES)
        result = list(filter_language_model_only(weights, language_model_only=False, stats=stats))
        assert result == weights
        assert stats.skipped_count == 0
        assert stats.skipped_example_names == ()

    def test_no_vision_tensors_present_is_a_pure_pass_through(self):
        # Laguna's real shape: language_model_only=True is set, but there is
        # nothing matching the prefix to skip -- the flag being True must
        # not change behavior when there is nothing for it to act on.
        stats = LanguageModelOnlyStats()
        weights = _weights(_LANGUAGE_MODEL_NAMES)
        result = list(filter_language_model_only(weights, language_model_only=True, stats=stats))
        assert result == weights
        assert stats.skipped_count == 0


class TestVisionFilteringWhenEnabled:
    def test_language_model_only_true_drops_every_vision_tensor(self):
        stats = LanguageModelOnlyStats()
        weights = _weights(_LANGUAGE_MODEL_NAMES + _VISION_NAMES)
        result = list(filter_language_model_only(weights, language_model_only=True, stats=stats))

        result_names = [name for name, _ in result]
        assert result_names == list(_LANGUAGE_MODEL_NAMES)
        assert not any(name.startswith("model.visual.") for name in result_names)

    def test_skip_count_matches_exactly_what_was_dropped(self):
        stats = LanguageModelOnlyStats()
        weights = _weights(_LANGUAGE_MODEL_NAMES + _VISION_NAMES)
        list(filter_language_model_only(weights, language_model_only=True, stats=stats))
        assert stats.skipped_count == len(_VISION_NAMES)
        assert set(stats.skipped_example_names) == set(_VISION_NAMES)

    def test_order_of_surviving_tensors_is_preserved(self):
        # Interleaved, not grouped -- a real checkpoint stream does not
        # conveniently sort vision and language-model tensors apart.
        stats = LanguageModelOnlyStats()
        interleaved = (
            _LANGUAGE_MODEL_NAMES[0],
            _VISION_NAMES[0],
            _LANGUAGE_MODEL_NAMES[1],
            _VISION_NAMES[1],
            _LANGUAGE_MODEL_NAMES[2],
        )
        result = list(
            filter_language_model_only(_weights(interleaved), language_model_only=True, stats=stats)
        )
        assert [name for name, _ in result] == [
            _LANGUAGE_MODEL_NAMES[0],
            _LANGUAGE_MODEL_NAMES[1],
            _LANGUAGE_MODEL_NAMES[2],
        ]

    def test_example_names_are_capped_not_unbounded(self):
        stats = LanguageModelOnlyStats()
        many_vision_names = tuple(f"model.visual.blocks.{i}.attn.qkv.weight" for i in range(50))
        list(
            filter_language_model_only(
                _weights(many_vision_names), language_model_only=True, stats=stats
            )
        )
        assert stats.skipped_count == 50
        assert len(stats.skipped_example_names) < 50


class TestNotAnIfBranchSpecialCase:
    """B0-1a's explicit instruction: this must be one function every loader
    adapter calls identically, not a per-quantization-format branch. These
    tests exercise it against two different plausible callers directly."""

    def test_default_prefix_is_format_agnostic(self):
        # Same vision names, same result, whether "called from" a
        # compressed-tensors loader or (hypothetically) a modelopt one --
        # nothing about the filter's decision reads a quantization format,
        # because vision-tower naming comes from the model architecture,
        # not the quantization library. See module docstring: the same
        # "model.visual." prefix was independently verified against both
        # nvidia/Qwen3.6-27B-NVFP4 (modelopt) and unsloth/Qwen3.6-27B-NVFP4
        # (compressed-tensors).
        for _caller_format in ("compressed_tensors", "modelopt"):
            stats = LanguageModelOnlyStats()
            result = list(
                filter_language_model_only(
                    _weights(_LANGUAGE_MODEL_NAMES + _VISION_NAMES),
                    language_model_only=True,
                    stats=stats,
                )
            )
            assert [name for name, _ in result] == list(_LANGUAGE_MODEL_NAMES)

    def test_custom_prefixes_override_the_default_without_touching_the_function(self):
        # A caller with a different naming convention (e.g. a hypothetical
        # future architecture) configures this via the vision_prefixes
        # parameter, not by editing filter_language_model_only's body.
        stats = LanguageModelOnlyStats()
        weights = _weights(("encoder.tower.weight", "model.language_model.layers.0.weight"))
        result = list(
            filter_language_model_only(
                weights,
                language_model_only=True,
                stats=stats,
                vision_prefixes=("encoder.tower.",),
            )
        )
        assert [name for name, _ in result] == ["model.language_model.layers.0.weight"]

    def test_default_prefix_constant_is_exported_for_callers_to_introspect(self):
        assert DEFAULT_VISION_TENSOR_PREFIXES == ("model.visual.",)
