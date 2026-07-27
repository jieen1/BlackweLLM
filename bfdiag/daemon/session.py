"""Reset semantics for the real Laguna + DFlash engine: what "pristine"
means between two experiments sharing the same warm daemon process.

This module holds two things:

1. ``RESET_CHECKLIST`` -- a structured, human- and machine-readable record
   of every piece of state that was found (by reading ``runtime/``, not
   guessing) to need clearing, plus the state that the task spec named but
   which turned out not to apply to the Laguna backend. Keep this in sync
   with ``notes/2026-07-27-bfdiag-warm-daemon.md``'s "reset 清单" table --
   that document is prose derived from this list, not an independent
   source.
2. ``reset_laguna_engine`` -- the actual reset routine. It is real,
   GPU-touching code (calls into ``runtime.backends.laguna``/
   ``laguna_dflash`` objects), written and reviewed against the current
   on-disk source but, per this task's hard no-GPU constraint, never
   executed. It has no import-time dependency on torch/runtime (the
   ``engine``/``backend`` objects are passed in already-constructed), so
   this module is safely importable in a torch-free environment -- only
   *calling* ``reset_laguna_engine`` requires the real engine.

Everything here is scoped to what ``bfdiag``'s ``LagunaEngineProvider``
loads: ``runtime/backends/laguna.py::LagunaBackend`` +
``runtime/backends/laguna_dflash.py::DFlashEngine`` (+ their CUDA Graph
helpers). It does NOT cover ``runtime/direct_model_runner.py`` (the
separate Qwen3.6 runner with GDN state and the content-addressed
``BlockPool``/prefix-cache machinery) -- see the checklist entries below
for why those two, despite being named explicitly in the task spec, do not
apply here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResetStep:
    """One entry in the reset checklist.

    ``applies_to_laguna`` distinguishes state that ``reset_laguna_engine``
    actually clears from state the task spec mentioned by name but which,
    on reading the current code, turned out to belong to a different
    runtime path (documented per project convention: spec vs. code reality
    diverges, code wins, divergence is recorded and we move on).
    """

    name: str
    location: str
    applies_to_laguna: bool
    note: str


RESET_CHECKLIST: tuple[ResetStep, ...] = (
    ResetStep(
        name="slot_kv_len / slot_committed_tokens / full-attention + SWA-ring KV cache blocks",
        location="runtime/backends/laguna.py:1504 LagunaBackend.reset_slot",
        applies_to_laguna=True,
        note=(
            "Per logical slot: zeros the physical full-attention and SWA "
            "ring-KV blocks for that slot and clears slot_kv_len/"
            "slot_committed_tokens. Must be called for EVERY slot in "
            "range(backend.num_slots), not just whichever slot(s) the "
            "previous experiment happened to touch -- see the CUDA Graph "
            "capture entry below for why slot 0 in particular is never "
            "safe to skip."
        ),
    ),
    ResetStep(
        name="DFlash draft KV cache (ring buffer, 6 SWA draft-model layers)",
        location=(
            "runtime/backends/laguna_dflash.py:297 "
            "DFlashEngine._alloc_draft_kv_cache (owns self._draft_kv_caches); "
            "zeroed by hand in benchmarks/diag_acceptance_v2.py:87-88"
        ),
        applies_to_laguna=True,
        note=(
            "NOT touched by LagunaBackend.reset_slot -- it belongs to the "
            "draft model, one level up in DFlashEngine, not the backend. "
            "Existing diagnostic scripts already had to zero this by hand "
            "between test cases (see diag_acceptance_v2.py's comment "
            "'RESET DRAFT KV CACHE'), which is precisely the boilerplate "
            "this task exists to eliminate. Ring-buffer addressing means a "
            "shorter/equal-length subsequent request should never read "
            "positions it did not itself write, so omitting this has not "
            "been shown to change results -- but it is untested across "
            "differently-shaped context-length sweeps, which is exactly the "
            "kind of silent contamination the canary exists to catch if "
            "that assumption is ever wrong."
        ),
    ),
    ResetStep(
        name="CUDA Graph capture-time warmup residue in real logical slots",
        location=(
            "runtime/backends/laguna_cuda_graph.py:294 "
            "(LagunaCudaGraphDecode.capture warms up "
            "backend.num_slots-batch_size .. backend.num_slots), :702 "
            "(LagunaCudaGraphVerify.capture warms up slot 0, warmup_kv=64); "
            "runtime/backends/laguna_dflash_cudagraph.py:301 "
            "(DFlashVerifyCudaGraph.capture _fill_buffers(0, capture_kv)), "
            ":544 (DFlashDraftCudaGraph.capture _fill_buffers(0, capture_kv))"
        ),
        applies_to_laguna=True,
        note=(
            "RESERVED_PHYSICAL_SLOTS = 0 for Laguna (runtime/backends/"
            "laguna.py:40) -- unlike the DirectModelRunner/BlockPool path, "
            "there are no physical slots set aside purely for warmup. Every "
            "CUDA Graph capture routine therefore writes dummy token-id=1 "
            "(or MASK_TOKEN_ID) KV data DIRECTLY into real logical slot 0 "
            "(all four CG classes) and into the tail "
            "num_slots-batch_size..num_slots logical slots (the M=1 decode "
            "CG). This means DFlashEngine.__init__ / LagunaEngineProvider."
            "load() does NOT return a pristine engine by construction -- "
            "load() must call reset() as its last step, unconditionally, "
            "specifically to clean this up. Skipping this is a real, "
            "load-bearing risk: the FIRST canary/experiment run after a "
            "cold daemon start would otherwise silently execute against a "
            "slot 0 that still holds full-length dummy KV content from "
            "graph capture."
        ),
    ),
    ResetStep(
        name="Laguna's own lightweight per-slot prefix-cache reuse (find_prefix_match)",
        location="runtime/backends/laguna_dflash.py:1043 DFlashEngine.generate_verify_only",
        applies_to_laguna=True,
        note=(
            "generate_verify_only's default enable_prefix_cache=True means "
            "calling DFlashEngine.generate()/generate_verify_only() twice on "
            "the SAME slot does NOT start clean by default: it calls "
            "backend.find_prefix_match(slot, prompt_ids) and, on a partial "
            "or full hit, reuses cached KV/ring state instead of "
            "re-prefilling. For the canary (and any provider.generate() "
            "call meant to be a pure function of the fixed prompt), this "
            "must be called with enable_prefix_cache=False AND after an "
            "explicit backend.reset_slot(slot) -- otherwise a canary run "
            "immediately following an experiment that used the same slot "
            "with a shared prefix would spuriously 'hit' that leftover "
            "state instead of testing the model cold, which defeats the "
            "canary's entire purpose. LagunaEngineProvider.generate() does "
            "both explicitly (see provider.py); this is the single most "
            "important 'read the code, don't assume' finding for this task."
        ),
    ),
    ResetStep(
        name="GDN (Gated DeltaNet) recurrent conv/ssm state",
        location="runtime/gdn_state.py; runtime/direct_model_runner.py:1593",
        applies_to_laguna=False,
        note=(
            "Does not apply to the Laguna backend: LagunaBackend passes "
            "gdn_layer_names=[] when constructing DirectModelRunner-style "
            "shared machinery (runtime/backends/laguna.py:383, comment "
            "'E1: no GDN layers -- Laguna has no GDN/SSM recursive state'). "
            "This state belongs to the OTHER model runner "
            "(runtime/direct_model_runner.py, the Qwen3.6 path), which "
            "bfdiag's LagunaEngineProvider does not load at all. Listed "
            "here only because the task spec named gdn_state.py explicitly "
            "as an example -- recorded per the project convention of noting "
            "spec-vs-code-reality divergences and moving on rather than "
            "stopping to ask."
        ),
    ),
    ResetStep(
        name="Content-addressed persistent prefix cache (BlockPool hash index)",
        location="runtime/block_pool.py; runtime/prefix_cache.py",
        applies_to_laguna=False,
        note=(
            "Does not apply either: LagunaBackend.reconcile_prefix_hit is an "
            "explicit stub ('E1: Laguna has no persistent content-addressed "
            "prefix cache yet (roadmap L2/L3 TODO) -- every admission is a "
            "cold miss', runtime/backends/laguna.py:1520-1523). "
            "block_pool.py/prefix_cache.py belong to DirectModelRunner, same "
            "as GDN above -- not loaded by LagunaEngineProvider. If/when "
            "Laguna grows a real persistent prefix cache (see "
            "notes/2026-07-27-laguna-prefix-cache-scoping.md), this "
            "checklist and reset_laguna_engine() must grow a matching step."
        ),
    ),
    ResetStep(
        name="LagunaBackend.generate()'s own CUDA-Graph greedy-decode step counter",
        location="runtime/backends/laguna.py:1643 self._decode_cg.reset()",
        applies_to_laguna=False,
        note=(
            "BUG FOUND while researching this checklist (code-read only; "
            "NOT fixed -- runtime/ is out of scope for this task, flagged "
            "for whoever owns runtime/backends/laguna.py): "
            "LagunaCudaGraphDecode (runtime/backends/laguna_cuda_graph.py) "
            "defines no reset() method at all (grep confirms zero "
            "occurrences of 'reset' in that file), so LagunaBackend."
            "generate() would raise AttributeError the first time it takes "
            "its greedy + CUDA-Graph-decode branch -- which is the DEFAULT "
            "branch, since QSR_DECODE_CUDA_GRAPH defaults to '1' "
            "(runtime/backends/laguna.py:342). Because of this, "
            "LagunaEngineProvider does NOT call backend.generate() at all; "
            "it drives DFlashEngine.generate_verify_only() directly (the "
            "production DFlash entrypoint, which has its own, separate, "
            "working CUDA Graph objects and never calls this buggy "
            "self._decode_cg.reset()). Recorded here so nobody wires "
            "bfdiag through the plain (non-DFlash) generate() path assuming "
            "it works."
        ),
    ),
)


def describe_reset_checklist() -> list[dict[str, Any]]:
    """JSON-safe dump of ``RESET_CHECKLIST``, e.g. for ``bf daemon status``
    or ad-hoc inspection from an exec'd diagnostic snippet."""
    return [
        {
            "name": step.name,
            "location": step.location,
            "applies_to_laguna": step.applies_to_laguna,
            "note": step.note,
        }
        for step in RESET_CHECKLIST
    ]


def reset_laguna_engine(engine: Any) -> list[str]:
    """Return a ``DFlashEngine`` (and its ``.backend``) to a pristine state.

    Real, GPU-touching code -- see module docstring for why it has never
    been executed. ``engine`` is expected to be a
    ``runtime.backends.laguna_dflash.DFlashEngine`` instance (duck-typed
    here so this module never needs to import torch/runtime at module
    scope).

    Performs, in order:

    1. ``backend.reset_slot(slot)`` for every slot in
       ``range(backend.num_slots)`` -- not just slots an experiment named,
       because CUDA Graph capture dirties slot 0 (and the tail slots) as a
       side effect that has nothing to do with which slots any particular
       experiment used (see RESET_CHECKLIST's CUDA Graph entry).
    2. Zeroing every tensor in ``engine._draft_kv_caches`` (the DFlash
       draft model's ring KV cache), matching established practice in
       ``benchmarks/diag_acceptance_v2.py``.

    Returns the list of step names actually performed (for logging /
    ``bf daemon status``); an engine with no draft KV cache (e.g. a
    non-DFlash backend passed in by mistake) just skips that step rather
    than raising, so this function stays safe to call defensively.
    """
    backend = engine.backend
    performed: list[str] = []

    for slot in range(backend.num_slots):
        backend.reset_slot(slot)
    performed.append("slot_kv_len/slot_committed_tokens/full+SWA KV cache blocks (all slots)")

    draft_kv_caches = getattr(engine, "_draft_kv_caches", None)
    if draft_kv_caches:
        for kv_tensor in draft_kv_caches.values():
            kv_tensor.zero_()
        performed.append(f"DFlash draft KV cache ({len(draft_kv_caches)} layers)")

    return performed


if __name__ == "__main__":
    for step in RESET_CHECKLIST:
        marker = "APPLIES" if step.applies_to_laguna else "n/a for Laguna"
        print(f"[{marker}] {step.name}\n    {step.location}\n")
