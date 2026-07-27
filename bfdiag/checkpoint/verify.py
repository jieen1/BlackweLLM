"""The safety valve: a deterministic greedy replay whose output must match
the baseline recorded at save time, or restore refuses to hand the slot
back.

This is not optional plumbing -- per the task brief, a restored slot that
*looks* fine but silently diverges numerically is worse than a crash (it
produces plausible-looking wrong conclusions). Every state item in
``state.py`` that is NOT separately persisted (the ring write-phase, the
next-round anchor/draft-tokens) is "safe" precisely because it is a
deterministic function of the items that ARE persisted -- this module is
what actually checks that claim on every restore, rather than merely
asserting it in a docstring.

No ``runtime.*``/``torch`` import needed here at all: everything operates
through ``engine``/``backend`` duck-typed attribute access (``dflash_round``,
``_draft_forward``, ``slot_committed_tokens``, ...), exactly like
``bfdiag/daemon/session.py::reset_laguna_engine``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class CheckpointVerificationError(RuntimeError):
    """A restored slot's deterministic replay diverged from the baseline
    recorded at save time. Callers (``restore.restore_checkpoint``) MUST
    NOT hand the slot back to the caller when this is raised."""


@dataclass
class BaselineResult:
    """One deterministic ``dflash_round`` probe: ``steps`` rounds starting
    from ``anchor_before``, flattened into ``committed_tokens`` (the exact
    token-for-token output a real caller would have seen), plus the
    (anchor, draft_tokens) pair for whatever round comes *after* this probe
    -- so a caller (``restore.restore_checkpoint``) can hand back a slot
    that is immediately ready to keep generating from where the probe left
    off, rather than "wasting" the verification rounds.
    """

    steps: int
    anchor_before: int
    committed_tokens: list[int]
    round_context_counts: list[int] = field(default_factory=list)
    final_anchor: int = 0
    final_draft_tokens: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "anchor_before": self.anchor_before,
            "committed_tokens": list(self.committed_tokens),
            "round_context_counts": list(self.round_context_counts),
            "final_anchor": self.final_anchor,
            "final_draft_tokens": list(self.final_draft_tokens),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineResult:
        return cls(
            steps=int(data["steps"]),
            anchor_before=int(data["anchor_before"]),
            committed_tokens=list(data.get("committed_tokens") or []),
            round_context_counts=list(data.get("round_context_counts") or []),
            final_anchor=int(data.get("final_anchor", 0)),
            final_draft_tokens=list(data.get("final_draft_tokens") or []),
        )


def next_round_inputs(engine: Any, slot: int) -> tuple[int, list[int]]:
    """Derive ``(anchor, draft_tokens)`` for the NEXT ``dflash_round`` call
    from the slot's CURRENT state -- mirrors ``dflash_prefill_bootstrap``'s
    own tail (``runtime/backends/laguna_dflash.py:1342-1347``) exactly:
    ``anchor`` is simply the last committed token, ``draft_tokens`` is a
    fresh deterministic draft-model forward pass over the (just-restored,
    in the restore case) draft KV ring.

    See ``state.py``'s "next-round (anchor, draft_tokens) pair" item for
    the full derivation and why neither value needs separate persistence.
    """
    backend = engine.backend
    committed = backend.slot_committed_tokens[slot]
    if not committed:
        raise ValueError(f"slot {slot} has no committed tokens -- nothing to derive a round from")
    anchor = committed[-1]
    kv_len = backend.slot_kv_len[slot]
    draft_cg = getattr(engine, "_draft_cg", None)
    if draft_cg is not None:
        draft_tokens = draft_cg.replay(slot, anchor, kv_len)
    else:
        draft_tokens = engine._draft_forward(slot, anchor, kv_len)
    return anchor, draft_tokens


def run_probe(engine: Any, slot: int, *, steps: int) -> BaselineResult:
    """Run ``steps`` real ``dflash_round`` calls starting from the slot's
    current state and return the flattened, deterministic output.

    MUTATES the live slot (advances kv_len/committed_tokens by however many
    tokens the ``steps`` rounds commit) -- this is unavoidable (there is no
    way to "ask what the model would output" without actually running it)
    and is the same documented side effect both ``store.save_checkpoint``
    (building the baseline) and ``restore.restore_checkpoint`` (re-checking
    it) rely on.
    """
    anchor, draft_tokens = next_round_inputs(engine, slot)
    anchor_before = anchor
    all_committed: list[int] = []
    context_counts: list[int] = []
    for _ in range(steps):
        decision = engine.dflash_round(slot, anchor, draft_tokens)
        all_committed.extend(decision["committed"])
        context_counts.append(decision["context_count"])
        anchor, draft_tokens = decision["next_anchor"], decision["next_draft_tokens"]
    return BaselineResult(
        steps=steps,
        anchor_before=anchor_before,
        committed_tokens=all_committed,
        round_context_counts=context_counts,
        final_anchor=anchor,
        final_draft_tokens=list(draft_tokens),
    )


def verify_restored_slot(engine: Any, slot: int, manifest: Any) -> BaselineResult:
    """Re-run the exact probe recorded in ``manifest.baseline`` against the
    just-restored slot and compare token-for-token. Raises
    :class:`CheckpointVerificationError` (naming the first divergence) on
    ANY mismatch -- ``restore.restore_checkpoint`` must not hand the slot
    back to its caller if this raises.

    Returns the fresh :class:`BaselineResult` on success (its
    ``final_anchor``/``final_draft_tokens`` are what the caller should use
    to keep generating -- the verification rounds are real, deterministic
    generation, not throwaway work).
    """
    baseline = BaselineResult.from_dict(manifest.baseline)
    replay = run_probe(engine, slot, steps=baseline.steps)

    if replay.anchor_before != baseline.anchor_before:
        raise CheckpointVerificationError(
            f"checkpoint {manifest.name!r} verification failed before any decode ran: "
            f"derived anchor={replay.anchor_before!r} != recorded baseline anchor="
            f"{baseline.anchor_before!r} -- slot_committed_tokens was not restored "
            "correctly (or the checkpoint was written by an incompatible engine that "
            "the fingerprint check failed to catch)."
        )

    if replay.committed_tokens != baseline.committed_tokens:
        first_diff = next(
            (
                i
                for i, (a, b) in enumerate(zip(replay.committed_tokens, baseline.committed_tokens))
                if a != b
            ),
            min(len(replay.committed_tokens), len(baseline.committed_tokens)),
        )
        raise CheckpointVerificationError(
            f"checkpoint {manifest.name!r} verification failed: replayed token "
            f"sequence diverges from the saved baseline at position {first_diff} "
            f"(replay={replay.committed_tokens!r}, baseline={baseline.committed_tokens!r}). "
            "Refusing to hand this slot back -- restored state does not reproduce "
            "recorded behavior."
        )

    # Belt-and-suspenders: if committed_tokens matched exactly, the
    # deterministic derivation chain guarantees final_anchor/
    # final_draft_tokens match too. A mismatch here despite the above
    # passing would indicate a bug in this verification code itself, not
    # in the checkpoint -- worth asserting rather than silently trusting.
    if replay.final_anchor != baseline.final_anchor or (
        replay.final_draft_tokens != baseline.final_draft_tokens
    ):
        raise CheckpointVerificationError(
            f"checkpoint {manifest.name!r}: committed token sequence matched but "
            f"final_anchor/final_draft_tokens diverged (replay final_anchor="
            f"{replay.final_anchor!r} vs baseline {baseline.final_anchor!r}) -- this "
            "should be impossible given matching committed_tokens; treat as a bug in "
            "verify.py itself, not the checkpoint."
        )

    return replay
