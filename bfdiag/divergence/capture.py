"""Turn ``oracle.capture_hooks.ForwardCapture`` output into scan-ready traces.

Reuses ``ForwardCapture`` (a read-only forward hook, see
oracle/capture_hooks.py) rather than reimplementing hooking; this module's
job is purely the naming/grouping glue between "a bag of named tensors" and
the ``layer_idx -> {submodule: tensor}`` shape ``bfdiag.divergence.scan``
expects.

The real, GPU-backed capture path (``capture_engine_activations``) requires
a live ``runtime.backends.laguna.LagunaBackend`` with a loaded CUDA model.
It is written and type-checked but **never executed** by this package's test
suite -- there is no GPU available in this environment. See
notes/2026-07-27-bfdiag-oracle-divergence.md's GPU-verification checklist.
``FakeCaptureSource`` below is what makes the rest of the pipeline (cache,
scan, report, CLI wiring) fully testable on CPU without it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from oracle.capture_hooks import CapturedTensor, ForwardCapture

#: ``layer_idx -> {submodule_name: tensor_like}``, matching
#: ``bfdiag.divergence.scan.ActivationTrace``.
ActivationTrace = Mapping[int, Mapping[str, Any]]


def parse_layer_submodule(name: str) -> tuple[int, str] | None:
    """Split a captured module name into ``(layer_idx, submodule)``.

    Mirrors the exact parsing convention already used in this repository
    (see ``runtime/backends/laguna.py``'s layer-group discovery: it splits
    on ``.``, finds a ``"layers"`` segment, and reads the next segment as
    the integer layer index) rather than assuming a fixed prefix like
    ``"model."`` -- the wrapping module path can vary by loader.

    Returns ``None`` for names outside the per-layer decoder stack (e.g.
    ``model.embed_tokens``, ``model.norm``, ``lm_head``).
    """
    parts = name.split(".")
    for index, part in enumerate(parts):
        if part == "layers" and index + 1 < len(parts):
            try:
                layer_idx = int(parts[index + 1])
            except ValueError:
                return None
            submodule = ".".join(parts[index + 2 :])
            return (layer_idx, submodule) if submodule else None
    return None


def group_captured_tensors(tensors: Iterable[CapturedTensor]) -> dict[int, dict[str, Any]]:
    """Group ``ForwardCapture.tensors()`` output into an ``ActivationTrace``."""
    grouped: dict[int, dict[str, Any]] = {}
    for item in tensors:
        parsed = parse_layer_submodule(item.name)
        if parsed is None:
            continue
        layer_idx, submodule = parsed
        grouped.setdefault(layer_idx, {})[submodule] = item.tensor
    return grouped


def group_named_tensors(tensors: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """Same grouping as ``group_captured_tensors``, for a plain ``{name:
    tensor}`` mapping (e.g. a safetensors dict loaded straight off disk,
    with no ``CapturedTensor`` wrapper)."""
    grouped: dict[int, dict[str, Any]] = {}
    for name, tensor in tensors.items():
        parsed = parse_layer_submodule(name)
        if parsed is None:
            continue
        layer_idx, submodule = parsed
        grouped.setdefault(layer_idx, {})[submodule] = tensor
    return grouped


def default_module_names(
    num_layers: int, *, moe_layer_ids: Sequence[int] | None = None
) -> tuple[str, ...]:
    """Module names to hook for a full per-layer divergence scan.

    Mirrors the real Laguna-S-2.1 decoder layer tree: ``input_layernorm``,
    ``post_attention_layernorm``, and ``self_attn`` on every layer, plus a
    ``mlp`` block with a ``mlp.gate`` router on every layer except layer 0
    (which uses a dense MLP with no router). Evidence:
    ``runtime/backends/laguna_sparkinfer_moe.py``'s
    ``MOE_LAYER_IDS = list(range(1, 48))``, and
    ``runtime/backends/laguna.py``'s ``_patch_moe_sparkinfer`` (skips
    ``layer_idx == 0``, reads router logits via ``moe_mod.gate(hs)``). The
    vendored vLLM model source for ``Qwen3_5DecoderLayer`` confirms the
    ``input_layernorm``/``post_attention_layernorm``/``self_attn``/``mlp``
    attribute names themselves. See
    notes/2026-07-27-bfdiag-oracle-divergence.md for exact file/line cites.
    """
    if moe_layer_ids is None:
        moe_layer_ids = range(1, num_layers)
    moe_set = set(moe_layer_ids)
    names: list[str] = []
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"
        names.append(f"{prefix}.input_layernorm")
        names.append(f"{prefix}.post_attention_layernorm")
        names.append(f"{prefix}.self_attn")
        names.append(f"{prefix}.mlp")
        if layer_idx in moe_set:
            names.append(f"{prefix}.mlp.gate")
    return tuple(names)


@runtime_checkable
class CaptureSource(Protocol):
    """Anything that can produce a per-layer activation trace for a prompt."""

    def capture(self, prompt_token_ids: Sequence[int]) -> dict[int, dict[str, Any]]: ...


@dataclass
class FakeCaptureSource:
    """Deterministic in-memory capture source for CPU-only unit tests.

    Returns the same pre-built ``trace`` for every prompt and records every
    prompt it was asked to capture, so tests can assert on call wiring
    without a model, a GPU, or ``oracle.capture_hooks``.
    """

    trace: dict[int, dict[str, Any]]
    seen_prompts: list[tuple[int, ...]] = field(default_factory=list)

    def capture(self, prompt_token_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        self.seen_prompts.append(tuple(prompt_token_ids))
        return self.trace


@dataclass
class EngineCaptureSource:
    """Live capture source wrapping an already-constructed engine backend.

    ``backend`` is duck-typed (``backend.model``, ``backend.reset_slot``,
    ``backend.prefill``) so this module never imports
    ``runtime.backends.laguna`` -- constructing a real backend requires a
    ``VllmConfig``/loaded checkpoint that only makes sense inside the
    server's own bootstrap path (``server/engine.py``), which is out of
    scope for a diagnostics tool. GPU-only; not exercised by tests.
    """

    backend: Any
    module_names: Sequence[str]
    slot: int = 0

    def capture(self, prompt_token_ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        return capture_engine_activations(
            self.backend, prompt_token_ids, self.module_names, slot=self.slot
        )


def capture_engine_activations(
    backend: Any,
    prompt_token_ids: Sequence[int],
    module_names: Sequence[str],
    *,
    slot: int = 0,
) -> dict[int, dict[str, Any]]:
    """Capture our own engine's per-layer activations for one prompt.

    GPU-only / not exercised by the test suite: requires a live
    ``runtime.backends.laguna.LagunaBackend`` with a loaded model on CUDA.
    ``backend`` is duck-typed on purpose (``.model``, ``.reset_slot(slot)``,
    ``.prefill(slot, token_ids)``) so this module has zero import-time
    coupling to ``runtime``/``vllm``/``sparkinfer``.
    """
    capture = ForwardCapture(backend.model, tuple(module_names))
    try:
        backend.reset_slot(slot)
        backend.prefill(slot, list(prompt_token_ids))
        tensors = capture.tensors()
    finally:
        capture.close()
    return group_captured_tensors(tensors)


if __name__ == "__main__":
    fake = FakeCaptureSource(trace={0: {"self_attn": [1.0, 2.0, 3.0]}})
    result = fake.capture([1, 2, 3])
    print("captured layers:", sorted(result))
    print("seen prompts:", fake.seen_prompts)
    print("parse_layer_submodule example:", parse_layer_submodule("model.layers.17.mlp.gate"))
    print("default_module_names(3) sample:", default_module_names(3))
