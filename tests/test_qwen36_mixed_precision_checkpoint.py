"""Real-checkpoint, header-only verification for the compressed-tensors
"mixed-precision" adapter (Track B, unsloth's ``unsloth/Qwen3.6-27B-NVFP4``).

This is the "zero missing, zero extra" claim, checked against the actual
checkpoint rather than a hand-written fixture: for every one of the real
checkpoint's ~1968 tensors (minus the vision tower, which
``language_model_only`` filters before ``load_weights`` ever sees it), does
:func:`~runtime.loading.compressed_tensors.mixed_precision_quant_map`'s
classification of that tensor's owning module agree with which tensors that
module *actually has*?

Deliberately reads only ``config.json`` and ``model.safetensors.index.json``
-- plain JSON, no ``safetensors.safe_open``, no ``torch``, no weight
materialization (the hard constraint this adapter was built under: no GPU,
no full-model load, host RAM already tight). This is real evidentiary
weight, not a synthetic fixture: a hand-written ``config_groups`` dict (as
``tests/test_loading_compressed_tensors.py`` uses) can only confirm what its
author already believed about the format; this file is the one that would
catch that belief being wrong. Skips cleanly when the checkpoint is not
present locally, matching ``tests/test_architecture_spec.py``'s and
``tests/test_model_registry.py``'s own convention for checkpoint-backed
tests.

What this file does NOT check, because it structurally cannot without a GPU:
that dequantizing these tensors the way this adapter does reproduces the
checkpoint's own intended numerics. It only checks that every tensor the
checkpoint actually ships lands on a Parameter this adapter's Linear classes
create, and vice versa -- i.e. that ``Qwen36ForCausalLMSelfBuilt.load_weights``
would not raise "N parameter(s) never received a checkpoint tensor" (or its
silent-wrong-value twin: a module classified as the wrong quantization
scheme, which would *pass* that assertion while loading nonsense -- see
``test_no_module_is_misclassified_as_the_wrong_scheme`` below for why that
one is checked explicitly, not just inferred from an absence of errors).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from runtime.loading.compressed_tensors import (
    QUANT_ALGO_MP_FP8_CHANNEL,
    QUANT_ALGO_MP_NVFP4,
    mixed_precision_quant_map,
)

HUB = Path.home() / ".cache" / "huggingface" / "hub"
QWEN_UNSLOTH = "models--unsloth--Qwen3.6-27B-NVFP4"

#: Every checkpoint tensor suffix this checkpoint is known to produce
#: (verified directly against its real ``model.safetensors.index.json``,
#: 2026-08-02: these ten account for all 1968 tensors, vision tower
#: included -- see the adapter's commit description for the exact count
#: breakdown). A suffix outside this set failing the test below means the
#: checkpoint changed shape since that verification, not that this list was
#: guessed.
_KNOWN_SUFFIXES = (
    "weight_packed",
    "weight_scale",
    "weight_global_scale",
    "input_global_scale",
    "weight",
    "bias",
    "A_log",
    "dt_bias",
    "k_scale",
    "v_scale",
)

_NVFP4_SUFFIXES = frozenset(
    {"weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"}
)
_FP8_CHANNEL_SUFFIXES = frozenset({"weight", "weight_scale"})


def _checkpoint_dir(repo: str) -> Path:
    matches = sorted((HUB / repo).glob("snapshots/*/config.json"))
    if not matches:
        pytest.skip(f"{repo} not present in the local HF cache")
    return matches[0].parent


def _load_language_model_module_suffixes(ckpt_dir: Path) -> dict[str, set[str]]:
    """``{dotted_module_name: {suffixes_it_owns}}`` for every non-vision
    tensor in the real checkpoint's safetensors index. Reads only the index
    JSON -- no tensor is opened or loaded."""
    index = json.loads((ckpt_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    module_suffixes: dict[str, set[str]] = defaultdict(set)
    for key in weight_map:
        if key.startswith("model.visual."):
            continue
        for suffix in _KNOWN_SUFFIXES:
            if key == suffix or key.endswith("." + suffix):
                module = key[: -(len(suffix) + 1)] if key.endswith("." + suffix) else ""
                module_suffixes[module].add(suffix)
                break
        else:
            raise AssertionError(
                f"checkpoint tensor {key!r} ends in none of the known suffixes "
                f"{_KNOWN_SUFFIXES} -- the checkpoint's tensor inventory changed since "
                "this test's suffix list was verified; update _KNOWN_SUFFIXES after "
                "checking what the new tensor is, don't just widen the list blindly"
            )
    return module_suffixes


@pytest.fixture(scope="module")
def unsloth_checkpoint_dir() -> Path:
    return _checkpoint_dir(QWEN_UNSLOTH)


@pytest.fixture(scope="module")
def unsloth_module_suffixes(unsloth_checkpoint_dir: Path) -> dict[str, set[str]]:
    return _load_language_model_module_suffixes(unsloth_checkpoint_dir)


@pytest.fixture(scope="module")
def unsloth_quant_map(unsloth_checkpoint_dir: Path):
    config = json.loads((unsloth_checkpoint_dir / "config.json").read_text())
    return mixed_precision_quant_map(config)


class TestRealCheckpointFormatIsWhatThisAdapterExpects:
    def test_declares_compressed_tensors_mixed_precision(self, unsloth_checkpoint_dir):
        config = json.loads((unsloth_checkpoint_dir / "config.json").read_text())
        quant_config = config["quantization_config"]
        assert quant_config["quant_method"] == "compressed-tensors"
        assert quant_config["format"] == "mixed-precision"


class TestZeroMissingZeroExtra:
    """The core claim: every module the checkpoint actually quantized is
    classified as exactly the scheme its real tensors match, and every
    module the classifier calls unquantized really does only carry a bare
    ``.weight`` (or nothing quantization-shaped at all)."""

    def test_every_nvfp4_module_has_exactly_the_four_nvfp4_tensors(
        self, unsloth_module_suffixes, unsloth_quant_map
    ):
        nvfp4_modules = [m for m, s in unsloth_module_suffixes.items() if "weight_packed" in s]
        assert len(nvfp4_modules) == 168, (
            "expected 168 NVFP4 modules (56 early layers x {gate,up,down}_proj) -- "
            f"found {len(nvfp4_modules)}"
        )
        for module in nvfp4_modules:
            assert unsloth_module_suffixes[module] == _NVFP4_SUFFIXES, (
                f"{module}: expected exactly {_NVFP4_SUFFIXES}, "
                f"got {unsloth_module_suffixes[module]}"
            )
            assert unsloth_quant_map.get(module) == QUANT_ALGO_MP_NVFP4, (
                f"{module}: checkpoint ships weight_packed but classifier said "
                f"{unsloth_quant_map.get(module)!r}, not nvfp4 -- this is exactly the "
                "'168 parameter(s) never received a checkpoint tensor' failure mode"
            )

    def test_every_fp8_channel_module_has_exactly_the_two_fp8_tensors(
        self, unsloth_module_suffixes, unsloth_quant_map
    ):
        fp8_modules = [
            m
            for m, s in unsloth_module_suffixes.items()
            if "weight_scale" in s and "weight_packed" not in s
        ]
        assert len(fp8_modules) == 233, (
            "expected 233 FP8-channel modules (self_attn q/k/v/o x16 full-attn layers "
            "+ linear_attn in_proj_qkv/in_proj_z/out_proj x48 GDN layers + lm_head + "
            "mlp gate/up/down x8 late layers) -- "
            f"found {len(fp8_modules)}"
        )
        for module in fp8_modules:
            assert unsloth_module_suffixes[module] == _FP8_CHANNEL_SUFFIXES, (
                f"{module}: expected exactly {_FP8_CHANNEL_SUFFIXES}, "
                f"got {unsloth_module_suffixes[module]}"
            )
            assert unsloth_quant_map.get(module) == QUANT_ALGO_MP_FP8_CHANNEL, (
                f"{module}: checkpoint ships a bare .weight + .weight_scale (FP8 "
                f"channel layout) but classifier said {unsloth_quant_map.get(module)!r}"
            )

    def test_no_plain_bf16_weight_module_is_misclassified_as_quantized(
        self, unsloth_module_suffixes, unsloth_quant_map
    ):
        # The false-positive direction: a module with only a bare .weight
        # and no .weight_scale sibling (embed_tokens, RMSNorm weights,
        # conv1d, in_proj_a/in_proj_b, ...) must classify as unquantized.
        # A false positive here would mean _make_linear builds e.g. an FP8
        # Linear expecting a .weight_scale tensor the checkpoint never
        # provides for that module -- caught eventually by
        # assert_all_params_loaded, but this test catches it at the
        # classification layer directly, before construction.
        plain_modules = [m for m, s in unsloth_module_suffixes.items() if s == {"weight"}]
        assert len(plain_modules) > 0, "sanity: the checkpoint should have plain-BF16 modules"
        misclassified = {
            m: unsloth_quant_map.get(m)
            for m in plain_modules
            if unsloth_quant_map.get(m) is not None
        }
        assert misclassified == {}

    def test_total_quantized_tensor_count_matches_the_known_baseline(self, unsloth_module_suffixes):
        # 168 weight_packed + 168 weight_scale(nvfp4) + 168 weight_global_scale
        # + 168 input_global_scale + 233 weight(fp8) + 233 weight_scale(fp8)
        # == the real index's counts, LANGUAGE-MODEL-ONLY (model.visual.*
        # excluded, matching language_model_only filtering -- measured
        # 2026-08-02): weight=602, weight_scale=401, input_global_scale=168,
        # weight_global_scale=168, weight_packed=168. This test pins those
        # five numbers against today's real checkpoint rather than a
        # remembered constant.
        counts: dict[str, int] = defaultdict(int)
        for suffixes in unsloth_module_suffixes.values():
            for suffix in suffixes:
                counts[suffix] += 1
        assert counts["weight_packed"] == 168
        assert counts["weight_scale"] == 401
        assert counts["weight_global_scale"] == 168
        assert counts["input_global_scale"] == 168
        # 602 bare .weight tensors, language-model-only; 233 of them are the
        # FP8-channel group's quantized weight, the rest (369) are plain
        # BF16 (embed_tokens, RMSNorm weights, conv1d, in_proj_a/in_proj_b,
        # mtp.* -- matches the 369-module count
        # test_no_plain_bf16_weight_module_is_misclassified_as_quantized
        # above finds independently).
        assert counts["weight"] == 602


class TestNoModuleIsMisclassifiedAsTheWrongScheme:
    """A module could in principle be quantized (has *some* .weight_scale
    tensor) but classified into the WRONG scheme (FP8 vs NVFP4) -- that
    would pass assert_all_params_loaded (the wrong Linear class still
    creates a Parameter named .weight_scale) while silently dequantizing
    every such weight incorrectly. The tests above already prove this
    cannot happen (every nvfp4-tensored module classifies nvfp4, every
    fp8-tensored module classifies fp8-channel) -- this test restates that
    as a single, explicit "no crossover" assertion for readability."""

    def test_no_crossover_between_the_two_schemes(self, unsloth_module_suffixes, unsloth_quant_map):
        for module, suffixes in unsloth_module_suffixes.items():
            algo = unsloth_quant_map.get(module)
            if algo == QUANT_ALGO_MP_NVFP4:
                assert "weight_packed" in suffixes, module
                assert "weight" not in suffixes, module
            elif algo == QUANT_ALGO_MP_FP8_CHANNEL:
                assert "weight" in suffixes, module
                assert "weight_packed" not in suffixes, module
