"""Checkpoint path -> which backend, loader, and speculative strategy to use.

Step 4 of the Track A migration (``docs/architecture.md`` §3.5.5), in shadow
mode: it resolves, and the tests assert the resolution equals the choice the
server hardcodes today. Nothing calls it yet. ``ServerEngine.MODEL`` and
``BACKEND`` disappear at step 5, which is the first step that changes
behavior.

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
IMPLEMENTED_BACKENDS = frozenset({"laguna"})


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
    return loader


def resolve_config(config: dict[str, Any]) -> Resolution:
    """Resolve an already-loaded ``config.json`` dict.

    Order matters: parse, then reject what cannot be served, then choose. A
    resolution that came back at all is one where every check passed, so a
    caller never has to ask whether some field was validated.
    """
    spec = parse_architecture(config)
    validate_text_only(spec)

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
