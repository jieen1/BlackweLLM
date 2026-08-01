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

    def test_vision_tower_is_refused_by_resolution_too(self):
        # The gate belongs in resolve(), not only in callers who remember it.
        config = minimal_config(
            architectures=["LagunaForCausalLM"],
            vision_config={"depth": 1},
        )
        with pytest.raises(UnsupportedArchitectureError, match="vision"):
            resolve_config(config)

    def test_missing_config_json_says_what_was_expected(self, tmp_path):
        with pytest.raises(UnsupportedArchitectureError, match="config.json"):
            resolve_checkpoint(tmp_path)


class TestQwen36IsRefusedHonestly:
    """Registered, refused, and the message says which of those it is."""

    def test_unimplemented_backend_is_named_rather_than_pretending(self):
        resolution_target = load_config(QWEN_TEXT_MTP)
        with pytest.raises(UnsupportedArchitectureError) as excinfo:
            resolve_config(resolution_target)
        message = str(excinfo.value)
        assert "qwen36" in message
        assert "not implemented" in message

    def test_multimodal_builds_fail_the_vision_gate_first(self):
        # Ordering is deliberate: a user pointing at the official checkpoint
        # should be told it carries a vision tower, not that a backend is
        # missing -- the first is actionable, the second is not.
        for repo in (QWEN_OFFICIAL, QWEN_UNSLOTH):
            with pytest.raises(UnsupportedArchitectureError, match="vision"):
                resolve_config(load_config(repo))


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
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "runtime.model_registry.IMPLEMENTED_BACKENDS",
                IMPLEMENTED_BACKENDS | {"qwen36"},
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
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "runtime.model_registry.IMPLEMENTED_BACKENDS",
                IMPLEMENTED_BACKENDS | {"qwen36"},
            )
            resolution = resolve_config(config)
        assert resolution.speculative == "mtp"
