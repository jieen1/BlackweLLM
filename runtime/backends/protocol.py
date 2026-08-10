"""Execution-layer contract between the scheduler and a model backend.

This module is **deliberately torch-free** so the CPU-only test job can import
it: it holds only typing declarations and frozen value types. Importing a
concrete backend (which does need torch) is the caller's problem.

Scope, and why it is this small
-------------------------------
``LagunaBackend`` defines 50 methods, 24 of them public. ``ServerEngine``
reaches the execution layer through exactly one attribute -- ``self.runner``
-- and touches only **13** members of it. Those 13 are the contract; the rest
is incidental surface that a second backend must not be forced to reproduce.
See ``docs/architecture.md`` §3.5.1 for the derivation and call counts.

Optional capability families
----------------------------
Not every backend can do everything. A backend advertises what it supports
through :attr:`ModelBackend.capabilities`, and the scheduler must consult
that **before** calling into a family -- never ``try/except AttributeError``.

That rule is not theoretical. ``server/engine.py`` currently calls
``mtp_prefill_warm_continue`` inside a bare ``except Exception``; no shipping
backend defines it (it survives only under ``oracle/qwen36_vllm/``), so every
``--session-affinity`` warm continue raises, is swallowed, and silently falls
back to a cold prefill. Outputs stay correct and the counter stays at zero,
which is why it went unnoticed. ``warm_continue`` below is that bug expressed
as a fact the scheduler can read. See ``docs/architecture.md`` §3.5.6 (N8).

Naming debt
-----------
Two member names still carry model-specific vocabulary inherited from the
current implementation: ``mtp_verify_and_commit_batch``,
``mtp_prefill_warm_continue``. They are declared here under today's names on
purpose -- this module ships as a shadow contract that must hold with **zero**
edits to call sites.

Step 5 landed (registry became the source of truth for backend selection;
``ServerEngine.MODEL``/``BACKEND``/``SERVER_MODEL_BACKEND`` are gone) without
taking this rename -- deliberately: renaming methods on ``LagunaBackend``
touches GPU-executed code this migration step did not otherwise need to
touch, widening the change under the same bit-exact gate for no benefit to
step 5's own claim. The rename to neutral names
(``verify_and_commit_batch``, ``prefill_warm_continue``) stays available to
take whenever call sites next change for an unrelated reason (naturally:
step 7's A3 coordinator, or whichever step first touches these two call
sites again). See §3.5.5.

``enable_dflash`` used to be a third member here, governed by
``speculative_decode`` -- B3 (2026-08-03) removed it from this protocol
entirely rather than renaming it: unlike the two members above, it is a
LOAD-TIME wiring call (``server/engine.py``'s ``_load_laguna_model``, once,
never from the recurring per-round scheduler path), and the qwen36 backend's
own equivalent (``Qwen36Backend.enable_mtp``) has a genuinely different
signature (an extra ``enable_resync`` kwarg, ``-> None`` not ``-> bool``),
not just a different name -- there was no single shape left to declare here
that both backends could honestly satisfy. See ``CAPABILITY_MEMBERS``'s own
docstring below for the full reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps this module torch-free
    from runtime.block_pool import ChunkedPrefillState
    from runtime.sampling import SamplingParams


@dataclass(frozen=True)
class BackendCapabilities:
    """What a backend can do, as data rather than as duck typing.

    Chosen over a ``supports("spec_decode")`` string lookup: a mistyped string
    fails silently and cannot be checked statically, while these fields can be
    type-checked, completed by an editor, and -- because this is a plain frozen
    dataclass -- serialized straight into a bfdiag run record or ``/metrics``.
    That turns "what was actually enabled for this run" into a recorded fact
    instead of something to reconstruct from logs.

    These describe what the backend *can* do, not what is currently switched
    on. ``speculative_decode=True`` means the backend's own speculative-decode
    wiring call may be made (``LagunaBackend.enable_dflash`` /
    ``Qwen36Backend.enable_mtp`` -- backend-specific, not part of this shared
    protocol; see the module docstring's "Naming debt" section); ask
    :attr:`ModelBackend.has_speculative_decode` whether it is active right now.
    """

    speculative_decode: bool
    prefix_cache: bool
    cuda_graph: bool
    chunked_prefill: bool
    warm_continue: bool


@dataclass(frozen=True)
class PrefixHit:
    """Result of matching a prompt against warm cache state.

    ``kv_hit`` is the longest block-aligned prefix whose attention KV is
    physically present and referenceable. ``state_hit`` is the longest
    boundary at or below ``kv_hit`` that also has a matching recurrent-state
    (GDN) checkpoint. A backend with no recurrent layers has no second
    resource to disagree with the first, so ``state_hit`` always equals
    ``kv_hit`` for it (INV-A3-6, ``ArchitectureSpec.needs_two_cache_families
    is False`` -- see ``docs/a3-cache-coordinator-design.md`` §2).

    Two fields rather than one ``int``, following ``BackendCapabilities``'s
    own precedent for choosing a dataclass over a bare tuple/string: the
    coordinator (step 7-d) and ``/metrics``/bfdiag need the KV-side number
    for observability (see ``docs/a3-cache-coordinator-design.md`` §3) even
    though the scheduler only ever acts on :attr:`effective`. A bare
    ``tuple[int, int]`` would let ``result[0]``/``result[1]`` be swapped at a
    call site with no error; this cannot be, because the field names are
    part of the type.

    INV-A3-2: ``0 <= state_hit <= kv_hit`` always holds -- enforced here,
    not left as a convention call sites must remember.
    """

    kv_hit: int
    state_hit: int

    def __post_init__(self) -> None:
        if self.state_hit > self.kv_hit:
            raise ValueError(
                f"INV-A3-2 violated: state_hit={self.state_hit} > kv_hit={self.kv_hit}"
            )
        if self.state_hit < 0 or self.kv_hit < 0:
            raise ValueError(
                f"prefix hit lengths must be non-negative: "
                f"kv_hit={self.kv_hit}, state_hit={self.state_hit}"
            )

    @property
    def effective(self) -> int:
        """The length the scheduler should treat as safe to skip prefill for.

        Always ``state_hit`` (never ``kv_hit``): the region
        ``[state_hit, kv_hit)`` has KV physically resident but no matching
        recurrent-state checkpoint to resume from, so treating it as a hit
        would run a recurrent layer's forward from a state that is stale for
        those positions. See ``docs/a3-cache-coordinator-design.md`` §3.
        """
        return self.state_hit


@dataclass(frozen=True)
class SlotSnapshot:
    slot: int
    kv_len: int
    is_fresh: bool


@dataclass(frozen=True)
class PrefixSnapshot:
    slot: int
    cached_kv_len: int
    cached_tokens: int
    #: First few cached token ids, for eyeballing which prefix a slot holds.
    #: Bounded on purpose -- a snapshot is a diagnostic, not a transcript.
    head: tuple[int, ...]


@dataclass(frozen=True)
class BackendSnapshot:
    """Values, not references -- the shape observability is allowed to depend on.

    ``server/app.py`` currently reads ``runner._prefix_cache_kv_len`` and
    ``runner._prefix_cache_tokens`` (both private) and assumes ``slot_kv_len``
    is list-shaped. Both ``/metrics`` 500s on 2026-08-01 came out of that seam:
    one from an aggregate dict whose key count diverged, one from reading a
    list as a mapping. Handing out frozen values means a differently-shaped
    backend can no longer take the monitoring signal down with it.

    ``slots`` and ``prefix`` are indexed by slot and cover every slot, so a
    caller may zip them or index them directly without a bounds check.
    See ``docs/architecture.md`` §3.5.2.

    ``dflash_cg_status`` is ``()`` when a backend has not attempted any CUDA
    Graph capture yet -- never missing, never raising -- so ``/metrics`` can
    iterate it unconditionally. Each entry is ``(graph_name,
    "captured"|"failed")``.

    **Not DFlash-specific despite the name** (kept as-is, 2026-08-02, B3 step
    0 -- ``docs/implementation-plan.md`` §7.3 C7-2): the name predates a
    second producer. ``LagunaBackend`` populates it from
    ``DFlashEngine.cg_status`` (see notes/2026-08-01-c1-c2-gpu-investigation.md),
    which happens to key one of its entries ``"decode"`` even though that
    graph has nothing to do with DFlash verify/draft -- the field was always
    "every CUDA Graph this backend attempted to capture," not "every DFlash
    graph." ``Qwen36Backend`` (which has no DFlash) now populates the same
    field from its own ``self.cg_status``, keyed ``"decode"`` for its
    (DFlash-free) decode graph. Renaming the field was considered and
    rejected: it is read by ``server/app.py`` (``/metrics`` and
    ``/debug/stats``) and asserted on by ``tests/test_metrics.py``/
    ``tests/test_qwen36_backend.py``/``tests/test_dflash_engine.py``; a
    rename buys clarity at the cost of touching every one of those for no
    behavior change. Tuple-of-pairs rather than a dict to keep this
    dataclass's fields all plain immutable values, matching
    ``slots``/``prefix`` above.

    ``runtime_stats`` and ``cg_fallback_reasons`` use the same immutable
    tuple-of-pairs shape.  They are optional because older backends do not
    maintain these counters; observability callers can iterate them without
    reaching into backend-private dictionaries or special-casing a model.
    """

    slots: tuple[SlotSnapshot, ...]
    prefix: tuple[PrefixSnapshot, ...]
    dflash_cg_status: tuple[tuple[str, str], ...] = ()
    runtime_stats: tuple[tuple[str, int], ...] = ()
    cg_fallback_reasons: tuple[tuple[str, int], ...] = ()


@runtime_checkable
class SlotStateView(Protocol):
    """Read-only server view of one slot.

    ``LagunaSlotState`` already satisfies this structurally, and its own
    docstring already describes itself this way ("must not mutate Laguna's
    cache bookkeeping directly"), so this promotes an existing idea rather
    than introducing one.
    """

    kv_len: int
    committed_tokens: tuple[int, ...]

    @property
    def is_fresh(self) -> bool: ...


class ModelBackend(Protocol):
    """The 12 members ``ServerEngine`` actually depends on, plus capabilities.

    Deliberately *not* ``@runtime_checkable``: ``isinstance`` against a
    Protocol only checks that attribute names exist, which would report a
    backend as conforming while its signatures disagree. Use
    :func:`check_conformance` instead -- it compares parameters.
    """

    # -- always required ---------------------------------------------------

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def reset_slot(self, slot: int) -> None: ...

    def slot_state(self, slot: int) -> SlotStateView: ...

    def snapshot(self) -> BackendSnapshot: ...

    def prefill(self, slot: int, prompt_ids: list[int]) -> int: ...

    def decode_batch_sampled(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
        params_list: list[SamplingParams],
        *,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> list[int] | tuple[list[int], list[dict]]: ...

    # -- capabilities.chunked_prefill --------------------------------------

    def prefill_chunked_begin(
        self,
        slots: list[int],
        prompts_per_slot: list[list[int]],
        chunk_size: int = 512,
        *,
        params_per_slot: dict[int, SamplingParams] | None = None,
    ) -> ChunkedPrefillState: ...

    def prefill_chunked_step(self, state: ChunkedPrefillState) -> bool: ...

    # -- capabilities.prefix_cache -----------------------------------------

    def reconcile_prefix_hit(self, token_ids: list[int]) -> PrefixHit: ...

    def find_best_slot_for_prompt(
        self,
        token_ids: list[int],
        free_slots: list[int],
    ) -> tuple[int, int]: ...

    # -- capabilities.speculative_decode -----------------------------------

    @property
    def has_speculative_decode(self) -> bool: ...

    def mtp_verify_and_commit_batch(
        self,
        slots: list[int],
        anchors: dict[int, int],
        drafts: dict[int, list[int]],
        *,
        params_per_slot: dict[int, SamplingParams] | None = None,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> dict[int, dict]: ...

    # -- capabilities.cuda_graph -------------------------------------------

    def capture_decode_cuda_graph(self) -> int | None: ...

    # -- capabilities.warm_continue ----------------------------------------

    def mtp_prefill_warm_continue(
        self,
        slot: int,
        prompt: list[int],
        prior_len: int,
    ) -> dict: ...


#: Which protocol members each capability flag governs. Members not listed
#: here are unconditionally required.
#:
#: B3 (2026-08-03): ``speculative_decode`` used to also govern
#: ``enable_dflash`` -- a LOAD-TIME wiring method (called once from
#: ``server/engine.py``'s ``_load_laguna_model``, never from the recurring
#: ``_step_sync`` scheduler path every other governed member here is reached
#: through), not a steady-state per-round contract member. That was
#: harmless while Laguna+DFlash was the only speculative-decode-capable
#: backend; adding a second one (qwen36+MTP, ``Qwen36Backend.enable_mtp``)
#: exposed it as Laguna-specific naming leaked into a supposedly
#: backend-agnostic contract -- ``enable_mtp``'s signature (an extra
#: ``enable_resync`` kwarg, ``-> None`` not ``-> bool``) could never satisfy
#: a conformance check pinned to ``enable_dflash``'s exact shape, and
#: renaming it to match would have been the tail wagging the dog. Each
#: backend's own "how do I turn this on" method keeps its own honest name
#: (``enable_dflash``/``enable_mtp``), same asymmetry
#: ``_load_laguna_model``/``_load_qwen36_model`` already have; only the
#: steady-state members (``has_speculative_decode``,
#: ``mtp_verify_and_commit_batch``) are cross-backend contract.
CAPABILITY_MEMBERS: dict[str, tuple[str, ...]] = {
    "chunked_prefill": ("prefill_chunked_begin", "prefill_chunked_step"),
    "prefix_cache": ("reconcile_prefix_hit", "find_best_slot_for_prompt"),
    "speculative_decode": (
        "has_speculative_decode",
        "mtp_verify_and_commit_batch",
    ),
    "cuda_graph": ("capture_decode_cuda_graph",),
    "warm_continue": ("mtp_prefill_warm_continue",),
}

REQUIRED_MEMBERS: tuple[str, ...] = (
    "capabilities",
    "reset_slot",
    "slot_state",
    "snapshot",
    "prefill",
    "decode_batch_sampled",
)


def _signature_of(obj: Any, name: str) -> Any:
    import inspect

    attr = inspect.getattr_static(obj, name, None)
    if attr is None:
        return None
    if isinstance(attr, property):
        return "property"
    try:
        return inspect.signature(attr)
    except (TypeError, ValueError):  # pragma: no cover - builtins/slots
        return None


def _comparable(sig: Any) -> Any:
    """Parameter names, kinds, and whether each has a default.

    Annotations are excluded on purpose: ``from __future__ import annotations``
    leaves them as strings whose spelling ("list[int]" vs "List[int]") differs
    without any difference in contract.
    """
    if sig is None or sig == "property":
        return sig
    return tuple(
        (p.name, p.kind, p.default is not p.empty)
        for p in sig.parameters.values()
        if p.name != "self"
    )


def check_conformance(backend_cls: type, capabilities: BackendCapabilities) -> list[str]:
    """Return a list of contract violations; empty means the class conforms.

    Checks presence *and* parameters. A member governed by a capability that is
    ``False`` is not required -- but if the class defines it anyway, it is still
    checked, because a backend that advertises "no" while carrying a mismatched
    implementation is exactly the drift this is meant to catch.
    """
    problems: list[str] = []
    governed = {m: cap for cap, members in CAPABILITY_MEMBERS.items() for m in members}

    for name in (*REQUIRED_MEMBERS, *governed):
        expected = _comparable(_signature_of(ModelBackend, name))
        actual_raw = _signature_of(backend_cls, name)
        cap = governed.get(name)

        if actual_raw is None:
            if cap is None:
                problems.append(f"{name}: missing, but unconditionally required")
            elif getattr(capabilities, cap):
                problems.append(f"{name}: missing, but capabilities.{cap} is True")
            continue

        actual = _comparable(actual_raw)
        if expected == "property" and actual != "property":
            problems.append(f"{name}: protocol declares a property, class defines a method")
        elif actual == "property" and expected != "property":
            problems.append(f"{name}: class defines a property, protocol declares a method")
        elif expected != actual:
            problems.append(
                f"{name}: signature differs\n    protocol: {expected}\n    class:    {actual}"
            )

    return problems
