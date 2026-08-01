"""B0-1a: skip vision-tower tensors when a loader runs in
``language_model_only`` mode.

Torch-free on purpose, unlike the rest of ``runtime/loading/`` -- this
filter only ever inspects the checkpoint tensor *name*, never the tensor
value, so it can be unit-tested under the CPU-only CI job with constructed
name lists (no checkpoint, no GPU) rather than only under
``~/.venvs/vllm``. That is a deliberate, not incidental, way to keep this
step's zero-GPU work zero-GPU.

Why this exists (``docs/implementation-plan.md`` §4/C-2, §7.1/B0-1a,
``docs/architecture.md``'s RK8 update): the Track A/B checkpoint decision
picked the official ``nvidia/Qwen3.6-27B-NVFP4``, which ships a vision
tower this runtime has no intention of ever serving (``roadmap.md`` §1:
this runtime is text-only, by design, permanently -- not "not implemented
yet"). ``runtime.architecture.validate_text_only`` (B0-1b) now *allows* a
checkpoint that carries a vision tower through, on the condition that the
loader that will actually read its weights runs with
``language_model_only=True``. This module is that loader-side half of the
contract: the thing that makes "zero vision tensors are actually loaded"
true, not just asserted.

**Not validated against any real checkpoint tensor stream yet.** Laguna --
this runtime's only production model -- has no vision tower at all, so
:func:`filter_language_model_only` is always called with
``language_model_only=False`` in production today (a no-op pass-through),
and the ``True`` branch below has only ever been exercised by the
constructed-name unit tests in ``tests/test_loading_language_model_only.py``.
The default prefix was cross-checked against two *real* checkpoints without
loading their weights (``model.safetensors.index.json`` only, both give
exactly 333 tensors, all and only under this prefix):
``nvidia/Qwen3.6-27B-NVFP4`` (modelopt) and ``unsloth/Qwen3.6-27B-NVFP4``
(compressed-tensors) -- same prefix regardless of quantization format,
which makes sense: vision-tower naming comes from the model architecture
(``transformers``' Qwen3.5/3.6 implementation), not from whichever
quantization library wrote the checkpoint. Independently corroborated in
``notes/2026-08-02-qwen36-b0-fact-baseline.md`` §1.4 (same 333, same
prefix, "排除过滤器可以简单到 ``name.startswith("model.visual.")``"). None
of this is a claim that the *filter itself* has been run against those
checkpoints -- only that the prefix it filters on is real, not guessed.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TypeVar

_T = TypeVar("_T")

#: Verified against real checkpoint indices, not guessed -- see module
#: docstring. A tuple (not a single string) because the contract is
#: "one or more prefixes", even though exactly one is known to matter today.
DEFAULT_VISION_TENSOR_PREFIXES: tuple[str, ...] = ("model.visual.",)

#: How many skipped names :class:`LanguageModelOnlyStats` keeps verbatim.
#: Diagnostic-only cap -- large enough to eyeball a real skip, small enough
#: to never matter for memory on a checkpoint with hundreds of matches.
_EXAMPLE_CAP = 8


@dataclass
class LanguageModelOnlyStats:
    """Mutated in place while :func:`filter_language_model_only`'s returned
    iterator is drained. Read the totals only after the caller has fully
    consumed that iterator (e.g. after ``model.load_weights(...)`` has
    returned) -- reading it mid-stream sees a partial count, not a bug in
    this class.
    """

    skipped_count: int = 0
    skipped_example_names: tuple[str, ...] = ()
    _example_cap: int = field(default=_EXAMPLE_CAP, repr=False)

    def _record_skip(self, name: str) -> None:
        self.skipped_count += 1
        if len(self.skipped_example_names) < self._example_cap:
            self.skipped_example_names = (*self.skipped_example_names, name)


def filter_language_model_only(
    weights: Iterable[tuple[str, _T]],
    *,
    language_model_only: bool,
    stats: LanguageModelOnlyStats,
    vision_prefixes: tuple[str, ...] = DEFAULT_VISION_TENSOR_PREFIXES,
) -> Iterator[tuple[str, _T]]:
    """Drop every ``(name, value)`` pair whose name starts with a vision
    prefix, when ``language_model_only`` is True. A transparent pass-through
    when it is False -- callers on a checkpoint that has no vision tower at
    all (Laguna, today, always) get back the exact same stream either way.

    Deliberately not written as an ``if quant_format == "compressed_tensors":
    ... elif ... :`` special case inside a loader adapter (B0-1a's explicit
    instruction): this is one function with one contract that any adapter
    -- compressed-tensors today, modelopt whenever Track B builds it --
    calls the same way. The quantization format never appears in this
    decision, which is the point: vision-tower naming is an architecture
    fact, not a quantization fact (see module docstring).

    This function *is* the enforcement of "zero vision tensors are actually
    loaded" (B0-1b): a name matching ``vision_prefixes`` never reaches the
    caller's ``model.load_weights(...)`` at all when ``language_model_only``
    is True, so there is no separate downstream re-assertion to also get
    wrong. ``stats`` exists for observability (how many were skipped, and a
    handful of examples) -- it is not itself the safety mechanism.
    """
    for name, value in weights:
        if language_model_only and name.startswith(vision_prefixes):
            stats._record_skip(name)
            continue
        yield name, value
