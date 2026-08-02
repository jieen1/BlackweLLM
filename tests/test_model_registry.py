"""Step 4 shadow tests: resolution must equal today's hardcoded choice.

The claim is narrow and checkable: pointed at the checkpoint the server
actually serves, the registry picks the backend, loader, and speculative
strategy that ``server/engine.py`` and ``server/app.py`` currently hardcode.
Nothing calls the registry yet, so a failure here is the registry's.
"""

from __future__ import annotations

import pytest

from runtime.architecture import UnsupportedArchitectureError
from runtime.model_registry import IMPLEMENTED_BACKENDS, resolve_checkpoint, resolve_config
from tests.test_architecture_spec import (
    HUB,
    LAGUNA,
    QWEN_OFFICIAL,
    QWEN_TEXT_MTP,
    QWEN_UNSLOTH,
    load_config,
    minimal_config,
)


def checkpoint_dir(repo: str):
    matches = sorted((HUB / repo).glob("snapshots/*/config.json"))
    if not matches:
        pytest.skip(f"{repo} not present in the local HF cache")
    return matches[0].parent


class TestLagunaResolvesToTodaysChoice:
    def test_resolution_matches_the_hardcoded_backend(self):
        # server/engine.py's BACKEND = "laguna" class attribute and
        # server/app.py's SERVER_MODEL_BACKEND = "laguna" module constant
        # both disappeared at step 5 -- this was the assertion that they
        # could, and now that they are gone, the assertion that
        # runtime.model_registry (their replacement) still agrees.
        resolution = resolve_config(load_config(LAGUNA))
        assert resolution.backend == "laguna"

    def test_loader_matches_the_only_format_the_loader_supports(self):
        resolution = resolve_config(load_config(LAGUNA))
        assert resolution.loader == "compressed_tensors"

    def test_speculative_strategy_matches_the_dflash_wiring(self):
        # Laguna's speculation comes from a separate draft model, not from
        # layers inside the checkpoint.
        resolution = resolve_config(load_config(LAGUNA))
        assert resolution.speculative == "dflash"

    def test_resolving_by_directory_reads_the_config(self):
        resolution = resolve_checkpoint(checkpoint_dir(LAGUNA))
        assert resolution.backend == "laguna"
        assert resolution.spec.num_hidden_layers == 48


class TestRefusalsHappenBeforeAnyWeightIsRead:
    def test_unregistered_architecture_lists_what_is_supported(self):
        with pytest.raises(UnsupportedArchitectureError) as excinfo:
            resolve_config(minimal_config(architectures=["MysteryForCausalLM"]))
        assert "LagunaForCausalLM" in str(excinfo.value)

    def test_unknown_quantization_names_the_supported_methods(self):
        config = minimal_config(
            architectures=["LagunaForCausalLM"],
            quantization_config={"quant_method": "awq"},
        )
        with pytest.raises(UnsupportedArchitectureError, match="awq"):
            resolve_config(config)

    def test_vision_tower_no_longer_blocks_resolution_on_its_own(self):
        # B0-1b (2026-08-02): resolve_config always validates with
        # language_model_only=True (runtime.model_registry.resolve_config),
        # since this runtime's loaders never build a vision tower -- so a
        # vision_config alone must not be what blocks resolution anymore.
        # This directly replaces the old "vision tower is refused by
        # resolution too" claim, which is no longer the intended behavior:
        # the official Qwen3.6 checkpoint the project committed to (D6) has
        # exactly this shape and must resolve past this gate.
        config = minimal_config(
            architectures=["LagunaForCausalLM"],
            vision_config={"depth": 1},
            quantization_config={"quant_method": "compressed-tensors"},
        )
        resolution = resolve_config(config)
        assert resolution.spec.has_vision_tower is True
        assert resolution.backend == "laguna"

    def test_missing_config_json_says_what_was_expected(self, tmp_path):
        with pytest.raises(UnsupportedArchitectureError, match="config.json"):
            resolve_checkpoint(tmp_path)


class TestQwen36Resolves:
    """Track B / B2 flipped this class's premise.

    It used to be ``TestQwen36IsRefusedHonestly`` and asserted the refusal:
    ``qwen36`` was registered so the error could name it, and
    ``IMPLEMENTED_BACKENDS`` withheld it so nothing pretended to work. B2
    landed ``runtime.backends.qwen36.Qwen36Backend``, wired
    ``ServerEngine._load_qwen36_model``, and served real OpenAI and
    Anthropic requests against the real ``nvidia/Qwen3.6-27B-NVFP4``
    checkpoint, so the refusal is now the dishonest answer.

    The claim that survives the flip unchanged is the one that was never
    about implementedness: the vision gate does not fire for these
    checkpoints (B0-1b -- resolution validates with
    ``language_model_only=True`` unconditionally)."""

    def test_text_only_checkpoint_resolves_to_the_qwen36_backend(self):
        resolution = resolve_config(load_config(QWEN_TEXT_MTP))
        assert resolution.backend == "qwen36"
        assert resolution.spec.has_vision_tower is False

    def test_multimodal_builds_resolve_and_the_vision_gate_does_not_fire(self):
        # B0-1b: a vision tower in config.json is no longer a refusal --
        # the loader runs language_model_only and the tensors are filtered.
        #
        # QWEN_UNSLOTH was dropped from this loop on 2026-08-02 (its
        # compressed-tensors/mixed-precision layout refused at resolve time
        # because no loader adapter existed for it) and restored the same
        # day the adapter landed (runtime/loading/compressed_tensors.py's
        # MixedPrecisionQuantMap + runtime/model/compressed_tensors_linear.py
        # -- see runtime/model_registry.py's SUPPORTED_QUANT_FORMATS comment
        # for the evidence).
        for repo in (QWEN_OFFICIAL, QWEN_UNSLOTH):
            resolution = resolve_config(load_config(repo))
            assert resolution.backend == "qwen36"
            assert resolution.spec.has_vision_tower is True

    def test_unsloth_resolves_to_the_compressed_tensors_loader(self):
        # unsloth's checkpoint is the actual reason SUPPORTED_QUANT_FORMATS
        # needed a "mixed-precision" entry at all -- pin the loader choice
        # directly, not just that resolution doesn't raise.
        resolution = resolve_config(load_config(QWEN_UNSLOTH))
        assert resolution.loader == "compressed_tensors"

    def test_resolving_unsloth_by_directory_reads_the_config(self):
        resolution = resolve_checkpoint(checkpoint_dir(QWEN_UNSLOTH))
        assert resolution.backend == "qwen36"
        assert resolution.loader == "compressed_tensors"

    def test_an_unregistered_backend_is_still_named_rather_than_pretending(self):
        # The refusal path must stay reachable and stay honest now that
        # qwen36 no longer exercises it -- otherwise flipping the flag also
        # silently retired the only test of that message.
        config = minimal_config(
            architectures=["Qwen3_5ForConditionalGeneration"],
            language_model_only=True,
            quantization_config={"quant_method": "modelopt"},
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "runtime.model_registry.IMPLEMENTED_BACKENDS",
                IMPLEMENTED_BACKENDS - {"qwen36"},
            )
            with pytest.raises(UnsupportedArchitectureError) as excinfo:
                resolve_config(config)
        message = str(excinfo.value)
        assert "qwen36" in message
        assert "not implemented" in message


class TestSpeculativeStrategyFollowsTheCheckpoint:
    def test_mtp_family_without_mtp_layers_resolves_to_no_speculation(self):
        # Same architecture, no mtp_num_hidden_layers. The family says "mtp",
        # the checkpoint does not carry it, and claiming otherwise would defer
        # the discovery to graph capture.
        config = minimal_config(
            architectures=["Qwen3_5ForConditionalGeneration"],
            language_model_only=True,
            quantization_config={"quant_method": "modelopt"},
        )
        resolution = resolve_config(config)
        assert resolution.spec.has_mtp is False
        assert resolution.speculative is None

    def test_mtp_family_with_mtp_layers_keeps_the_strategy(self):
        config = minimal_config(
            architectures=["Qwen3_5ForConditionalGeneration"],
            language_model_only=True,
            quantization_config={"quant_method": "modelopt"},
            mtp_num_hidden_layers=1,
        )
        resolution = resolve_config(config)
        assert resolution.speculative == "mtp"
