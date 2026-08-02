"""Checkpoint path -> which backend, loader, and speculative strategy to use.

Step 4 of the Track A migration (``docs/architecture.md`` §3.5.5) landed this
in shadow mode: it resolved, and the tests asserted the resolution equalled
the choice the server hardcoded. Step 5 gave it its first real production
consumer -- ``server/app.py``'s ``lifespan()`` now calls
:func:`resolve_checkpoint` to decide ``backend`` instead of reading the
hardcoded ``ServerEngine.MODEL`` / ``BACKEND`` class attributes and
``server/app.py``'s ``SERVER_MODEL_BACKEND`` constant, all three of which are
gone. ``server/engine.py`` validates its ``backend`` parameter against
:data:`IMPLEMENTED_BACKENDS` below rather than a single hardcoded string.

Torch-free, like :mod:`runtime.architecture`, because the whole point is to
decide -- and to refuse -- before any weight is read.

What the resolution keys on
---------------------------
Architecture name alone is not enough, and this is not a hypothetical: all
four local Qwen3.6 checkpoints declare ``Qwen3_5ForConditionalGeneration``,
yet three carry a vision tower and one does not, and three are quantized with
modelopt while unsloth's is compressed-tensors. So the architecture selects
the *backend*, while the loader is selected from the checkpoint's own
``quantization_config``, and the vision check is a separate gate.

Registering a family here is deliberately explicit. Per ``roadmap.md`` §3 the
project does not do generic HF architecture support: fewer architectures,
each actually correct.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.architecture import (
    ArchitectureSpec,
    UnsupportedArchitectureError,
    parse_architecture,
    validate_text_only,
)

#: Quantization method -> loader adapter name. Read per checkpoint.
LOADER_FOR_QUANT_METHOD = {
    "compressed-tensors": "compressed_tensors",
    "modelopt": "modelopt",
}

#: ``compressed-tensors`` is a container, not a payload format, and the formats
#: inside it are not interchangeable. Measured 2026-08-02 across the local
#: checkpoints, the per-layer tensor names differ outright:
#:
#:   nvfp4-pack-quantized  Laguna, production. Symmetric NVFP4, no zero point.
#:   mixed-precision       unsloth's Qwen3.6. FP8 channel-wise + NVFP4.
#:   pack-quantized        cyankiwi's Qwen3.6-AWQ-INT4. ``num_bits=4, type=int,
#:                         group_size=32`` -- group-wise INT4 with an
#:                         **asymmetric weight_zero_point**.
#:
#: Note the names: ``nvfp4-pack-quantized`` and ``pack-quantized`` differ by a
#: prefix and by whether a zero point exists. Nothing in this runtime models a
#: zero point.
#:
#: Selecting a loader on ``quant_method`` alone accepted the asymmetric one,
#: and the danger is not a crash -- every tensor name the loader looks for is
#: present, so ``assert_all_params_loaded`` would pass while each weight came
#: back dequantized as if it were symmetric.
#:
#: So the gate is the (method, format) pair, and an unlisted format is refused
#: rather than assumed compatible. ``None`` means the method carries no
#: sub-format (modelopt).
#: ``mixed-precision`` was listed here on 2026-08-02 and removed the same
#: day: the registry accepted it, and then ``load_weights`` failed with
#: "168 parameter(s) never received a checkpoint tensor" (168 == unsloth's
#: ``weight_packed`` tensor count), because ``runtime/loading/
#: compressed_tensors.py`` at the time only handled Laguna's
#: ``nvfp4-pack-quantized``. It was restored the same day the adapter
#: landed (same commit as this comment's edit): ``runtime/loading/
#: compressed_tensors.py`` now also carries unsloth's naming knowledge
#: (:class:`~runtime.loading.compressed_tensors.MixedPrecisionQuantMap`,
#: :func:`~runtime.loading.compressed_tensors.dequantize_fp8_channel`), and
#: ``runtime/model/qwen36_model.py``'s ``_make_linear`` dispatches to
#: ``runtime/model/compressed_tensors_linear.py``'s Linear classes for it --
#: see those modules' docstrings for the measured evidence (real safetensors
#: headers) this rests on, and ``tests/test_qwen36_mixed_precision_checkpoint.py``
#: for the real-checkpoint, header-only "every module classifies to exactly
#: the tensors it actually has, zero missing, zero extra" cross-check that
#: earns this format its place back in this frozenset (not merely "the
#: registry no longer says no").
#:
#: Why a specific known-but-unsupported format is refused. Generic text would
#: be worse than useless here: the refusal below and the one above fail for
#: opposite reasons, and telling them apart is what decides whether adding
#: the format is a loader task or a correctness problem.
_WHY_REFUSED: dict[str | None, str] = {
    "pack-quantized": (
        "Refusing rather than loading: this is group-wise INT4 with an "
        "asymmetric zero point (weight_zero_point), which nothing here "
        "models. Every tensor "
        "the loader looks for is present, so it would pass the "
        "all-params-loaded assertion and still dequantize every weight as if "
        "it were symmetric -- wrong output, no error."
    ),
}


SUPPORTED_QUANT_FORMATS: dict[str, frozenset[str | None]] = {
    "compressed-tensors": frozenset({"nvfp4-pack-quantized", "mixed-precision", None}),
    "modelopt": frozenset({None}),
}


@dataclass(frozen=True)
class ArchitectureFamily:
    """One explicitly supported architecture."""

    #: Matched against ``config.json``'s ``architectures[0]``.
    architecture: str
    backend: str
    #: ``"dflash"`` (separate draft model), ``"mtp"`` (in-checkpoint layers),
    #: or ``None``. What the family *can* do; whether a given checkpoint does
    #: is decided per checkpoint in :func:`resolve`.
    speculative: str | None


REGISTRY: tuple[ArchitectureFamily, ...] = (
    ArchitectureFamily(
        architecture="LagunaForCausalLM",
        backend="laguna",
        speculative="dflash",
    ),
    # Registered so resolution is testable and the error for it is honest.
    # `supported=False` below is what keeps it from claiming to work.
    ArchitectureFamily(
        architecture="Qwen3_5ForConditionalGeneration",
        backend="qwen36",
        speculative="mtp",
    ),
)

#: Backends that actually exist today. Track B flips ``qwen36``.
#:
#: B2 status (2026-08-02): ``runtime.backends.qwen36.Qwen36Backend`` exists
#: and conforms to ``ModelBackend``, and ``ServerEngine._load_qwen36_model``
#: can construct it. What this frozenset is waiting on is not code, it is
#: **evidence**: a real request served end-to-end through the HTTP layer on
#: the real checkpoint. Adding the string is one line; adding it before that
#: run means a user pointing at a Qwen3.6 checkpoint is served by a path
#: nothing has exercised, which is the failure mode this repo keeps
#: re-learning (N8: a capability claimed by silence, swallowed by
#: try/except, unnoticed for three years).
IMPLEMENTED_BACKENDS = frozenset({"laguna", "qwen36"})


@dataclass(frozen=True)
class Resolution:
    spec: ArchitectureSpec
    backend: str
    loader: str
    #: The strategy this checkpoint can actually run, after checking that the
    #: checkpoint carries what the family's strategy needs.
    speculative: str | None


def _family_for(spec: ArchitectureSpec) -> ArchitectureFamily:
    for family in REGISTRY:
        if family.architecture == spec.architecture:
            return family
    known = sorted(family.architecture for family in REGISTRY)
    raise UnsupportedArchitectureError(
        f"architecture {spec.architecture!r} is not registered; "
        f"supported architectures are {known}. Adding one is a deliberate "
        f"step (model graph + loader adapter + spec entry), not automatic."
    )


def _loader_for(spec: ArchitectureSpec) -> str:
    loader = LOADER_FOR_QUANT_METHOD.get(spec.quant.method)
    if loader is None:
        raise UnsupportedArchitectureError(
            f"quantization method {spec.quant.method!r} has no loader adapter; "
            f"supported methods are {sorted(LOADER_FOR_QUANT_METHOD)}"
        )
    allowed = SUPPORTED_QUANT_FORMATS.get(spec.quant.method, frozenset({None}))
    if spec.quant.format not in allowed:
        raise UnsupportedArchitectureError(
            f"quantization method {spec.quant.method!r} format "
            f"{spec.quant.format!r} has no loader adapter; supported formats for "
            f"this method are {sorted(f for f in allowed if f is not None) or ['(none)']}. "
            + _WHY_REFUSED.get(
                spec.quant.format,
                "Refusing rather than loading: the formats inside a quantization "
                "method carry different per-layer tensors and are not "
                "interchangeable.",
            )
        )
    return loader


def resolve_config(config: dict[str, Any]) -> Resolution:
    """Resolve an already-loaded ``config.json`` dict.

    Order matters: parse, then reject what cannot be served, then choose. A
    resolution that came back at all is one where every check passed, so a
    caller never has to ask whether some field was validated.
    """
    spec = parse_architecture(config)
    # B0-1b: this runtime's loaders always run in language_model_only mode
    # -- there is no code path anywhere that builds a vision tower, and
    # roadmap.md §1 commits to that being permanent, not "not implemented
    # yet". So this is not a per-checkpoint choice to plumb through
    # Resolution; it is a fixed fact about how every loader this registry
    # can select actually behaves, asserted here once. See
    # runtime.architecture.validate_text_only's docstring for what this
    # value promises and where that promise is actually kept.
    validate_text_only(spec, language_model_only=True)

    family = _family_for(spec)
    if family.backend not in IMPLEMENTED_BACKENDS:
        raise UnsupportedArchitectureError(
            f"{spec.architecture!r} resolves to the {family.backend!r} backend, "
            f"which is not implemented yet; implemented backends are "
            f"{sorted(IMPLEMENTED_BACKENDS)}"
        )

    # A family may support MTP while a given checkpoint does not carry the
    # layers -- claiming otherwise would mean discovering it during capture.
    speculative = family.speculative
    if speculative == "mtp" and not spec.has_mtp:
        speculative = None

    return Resolution(
        spec=spec,
        backend=family.backend,
        loader=_loader_for(spec),
        speculative=speculative,
    )


def resolve_checkpoint(path: str | Path) -> Resolution:
    """Resolve a checkpoint directory by reading its ``config.json``."""
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        raise UnsupportedArchitectureError(
            f"no config.json under {path}; a checkpoint directory is expected, "
            f"not a weight file or a repo id"
        )
    return resolve_config(json.loads(config_path.read_text()))
