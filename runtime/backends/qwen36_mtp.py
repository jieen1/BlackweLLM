"""B3/serving: MTP speculative decoding wired into ``Qwen36Backend``.

This is the engine-side counterpart to the model-level primitives
``runtime/model/qwen36_model.py`` already ships (``Qwen36MTPHead``,
``mtp_step``, ``spec_forward``, ``verify_forward``, ``commit_verify``) and
the accept/reject logic in ``runtime/mtp_accept.py``. Those pieces were
correct and tested in isolation, but nothing called them from the serving
path -- ``Qwen36Backend.capabilities.speculative_decode`` was ``False`` and
every request went through plain per-token decode. This module is the
missing per-slot round driver, structured the same way
``runtime/backends/laguna_dflash.py``'s ``DFlashEngine`` drives DFlash for
Laguna: a separate, backend-owned object with its own per-slot state,
reached through ``Qwen36Backend.mtp_verify_and_commit_batch`` -- the same
hook ``server/engine.py``'s ``_step_sync`` MTP branch already calls for
Laguna+DFlash, unmodified.

**The (token, hidden) pairing bug this module fixes.** Every existing B3
script (``scripts/b3_mtp_e2e_acceptance_throughput.py``,
``scripts/b3b_*.py``, ``scripts/mtpfix_*.py``, and the resync port on
``work/mtp-resync-20260802``) seeds each round's first draft step with
``mtp_step(next_token_ids=anchor_token, prev_hidden=anchor_hidden, ...)``
where ``anchor_hidden`` is the hidden state produced BY processing
``anchor_token`` (i.e. the state ``anchor_token``'s OWN forward pass
produced). ``mtp_step``'s own docstring says the opposite is required:
``prev_hidden`` must be "the hidden state from the position immediately
before" -- the state that PREDICTED ``anchor_token`` as its own greedy
argmax, one position earlier.

This is not a matter of interpretation. vLLM's native Qwen3.6 MTP
integration (``/home/bot/vllm/vllm/v1/worker/gpu/spec_decode/autoregressive/
speculator.py``, ``_prepare_prefill_inputs_kernel``, lines ~510-519) shifts
the draft model's input_ids left by one against the target's UNSHIFTED
hidden states before every draft-model prefill call ("Shift
target_input_ids by one"): row ``j`` of the draft forward pairs
``target_hidden_states[j]`` (produced by processing ``target_input_ids[j]``)
with ``target_input_ids[j+1]`` (the token ONE position ahead). That is
exactly what ``mtp_step``'s docstring already specifies and exactly what
every prior script in this repo violates. The historical vLLM-based runner
(``qsr-hist`` @ ``8f5c195``, ``_mtp_sync_and_propose``) implements the same
shift -- its own docstring calls it out explicitly ("matches vLLM's real
``_prepare_prefill_inputs_kernel`` shift-by-one mechanism"). The
``work/mtp-resync-20260802`` port's commit message even names the
discrepancy without recognizing it as a bug: "adapted to this file's own
(unshifted) token/hidden pairing convention".

Chained continuation steps (draft step 1..K-1 within one round) are NOT
affected -- vLLM's own ``update_draft_inputs`` feeds each step's own output
hidden state paired with the token sampled FROM that same output back into
the next step, i.e. same-position pairing, which is exactly what
``mtp_step``'s chaining already does. Only the ONE handoff per round --
target hidden -> draft head's first step -- needs the shift, and it is
the seed for every position drafted, so getting it wrong degrades the
entire chain, not just one token. This one bug plausibly accounts for most
of the "far below expected" acceptance measured in every prior script (see
``notes/2026-08-02-mtpfix-historical-comparison.md``): even
``scripts/b3b_teacher_forced_head_quality.py``'s "zero compounding" ceiling
measurement (62.9%/82.4%/71.1%) uses the same wrong pairing (its own
docstring: "hiddens[i] is ... produced immediately after consuming
tokens[i] -- i.e. exactly the prev_hidden MTP needs to predict tokens[i+1]"
-- exactly the unshifted convention), so even that ceiling may understate
the head's true single-step quality.

The fix costs nothing extra: the correctly-shifted hidden state is always
already available from a computation this round needed anyway (either the
prefill's own last-position hidden -- :meth:`Qwen36MTPEngine
.draft_after_prefill` -- or, within :meth:`Qwen36MTPEngine.round`, the SAME
verify/anchor forward every round already performs for accept/reject). No
additional target forward pass is added to get it right.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from runtime.logprobs import compute_logprobs
from runtime.mtp_accept import determine_accept_reject_from_predictions, sample_accept_reject
from runtime.sampling import SamplingParams, compute_sampling_distribution, make_generator

if TYPE_CHECKING:
    from runtime.backends.qwen36 import Qwen36Backend
    from runtime.model.qwen36_model import Qwen36PagedAttentionCache


#: QSR_SERVER_MTP_RESYNC (default off): per-round re-grounding of the
#: interior accepted draft positions with the target's own real
#: verify_hidden, instead of leaving them holding the draft head's
#: self-chained hidden state from the exploratory loop that first drafted
#: them. Ported in spirit from the historical runner's per-round resync
#: (``qsr-hist`` @ ``8f5c195``, ``_mtp_sync_and_propose`` /
#: ``mtp_verify_and_commit``) -- see :meth:`Qwen36MTPEngine._resync`'s
#: docstring for why this module's indexing differs from that port's
#: (``work/mtp-resync-20260802`` @ ``aed0e2d``) own, which carried the
#: same unshifted pairing convention documented above. Default OFF and
#: independently toggleable from MTP itself so it can be A/B measured on
#: real hardware separately from the pairing fix.
def _resync_env_default() -> bool:
    return os.environ.get("QSR_SERVER_MTP_RESYNC", "0") != "0"


class Qwen36MTPEngine:
    """Owns per-slot MTP draft-head state (its own small KV cache per slot)
    and drives one draft+verify+accept/reject+re-draft round, the way
    :class:`runtime.backends.laguna_dflash.DFlashEngine` drives DFlash for
    Laguna. Kept as a separate object rather than folded into
    :class:`runtime.backends.qwen36.Qwen36Backend` / :class:`runtime.model
    .qwen36_slots.Qwen36SlotPool` for the same reason DFlash is: this is
    opt-in, backend-private bookkeeping layered on top of the always-on
    slot pool, not a change to it.

    Sequential per slot, not batched -- matches DFlash's own documented
    precedent (``LagunaBackend.mtp_verify_and_commit_batch``'s docstring:
    "the loop below still handles >1 slots correctly (just sequentially, no
    batched replay)"). ``Qwen36ForCausalLMSelfBuilt``'s whole model graph is
    batch=1 by construction (B1's own scope), so there is no batched verify
    path to reach for here the way the historical vLLM-based runner's
    ``verify_batch_spec``/``_ssm_spec_row`` had -- ``runtime.
    recurrent_state_pool``'s ``spec_row`` addressing (built for exactly that
    kind of batched multi-slot GDN spec verify) stays uncalled; wiring it in
    is a real future throughput lever, not attempted here (out of scope --
    this landing is about reachability, not batched-verify throughput).
    """

    def __init__(
        self,
        backend: Qwen36Backend,
        *,
        num_speculative_tokens: int,
        enable_resync: bool | None = None,
    ) -> None:
        model = backend.model
        if model.mtp is None:
            raise ValueError(
                "Qwen36MTPEngine requires a model loaded with enable_mtp=True "
                "(runtime.model_loading.load_qwen36_model(..., enable_mtp=True)); "
                "this model instance has no mtp head"
            )
        self.backend = backend
        self.model = model
        self.k = num_speculative_tokens
        self.enable_resync = _resync_env_default() if enable_resync is None else enable_resync
        if self.enable_resync and not hasattr(model, "mtp_resync_step"):
            # Fail here, not on round 3. `_resync` calls
            # `self.model.mtp_resync_step(...)`, and that method exists only
            # on the unmerged branch `work/mtp-resync-20260802` (@ aed0e2d) --
            # never on main. So this flag could only ever produce an
            # AttributeError partway through a live request, after the server
            # had come up and started answering. A flag whose sole reachable
            # behaviour is a mid-request crash should refuse at construction.
            #
            # Not fixed by porting the method: that is ~166 lines which also
            # reimplements `mtp_step` in terms of itself, on the MTP hot path,
            # in service of an optimization that has never been A/B measured.
            # It needs its own evidence before it earns that risk.
            raise RuntimeError(
                "QSR_SERVER_MTP_RESYNC is set, but this model has no "
                "`mtp_resync_step` -- the per-round resync was ported only on "
                "branch work/mtp-resync-20260802 and never merged, so the flag "
                "has no working implementation here. Unset it (default off) to "
                "run MTP without resync, which is the only measured path."
            )
        self.device = backend.device
        self.dtype = backend.dtype
        self.vocab_size = int(model.config["vocab_size"])

        #: One persistent MTP self-attention cache per real slot (no
        #: scratch row -- unlike Qwen36SlotPool, nothing here is
        #: CUDA-graph-captured, so there is no padding-row aliasing hazard
        #: to guard against). Allocated once, reset (not reallocated) by
        #: :meth:`reset_slot`, same discipline B0-5 established for the
        #: backbone's own recurrent-state buffers.
        self._caches: list[Qwen36PagedAttentionCache] = [
            model.mtp_new_cache(device=self.device, dtype=self.dtype)
            for _ in range(backend.num_slots)
        ]

        self.stats: dict[str, int] = {
            "rounds": 0,
            "resync_rounds": 0,
            "sampled_rounds": 0,
        }

    # -- slot lifecycle ------------------------------------------------

    def reset_slot(self, slot: int) -> None:
        """Return ``slot``'s MTP cache to fresh -- called from
        ``Qwen36Backend.reset_slot`` alongside the backbone's own reset.

        Pointer-only, like the backbone's attention KV reset (not the GDN
        recurrent-state zero): bytes past ``seq_len`` are never read by a
        causal cache, and the MTP self-attention's positions are relative
        to wherever ``seq_len`` starts each generation (see the module
        docstring on why an absolute-position offset is harmless for
        RoPE -- q/k dot products depend only on the difference between
        positions, and every position ever written into one slot's MTP
        cache shares the same offset for that slot's whole generation).
        """
        self._caches[slot].seq_len = 0

    # -- drafting --------------------------------------------------------

    def _draft_loop(self, slot: int, seed_token: int, seed_hidden: torch.Tensor) -> list[int]:
        """Draft ``self.k`` tokens, chained, starting from ``seed_token``
        (already committed) paired with ``seed_hidden`` -- the hidden state
        that PREDICTED ``seed_token``, one position before it (the fix this
        module exists for; see module docstring). Steps 1..k-1 chain the
        head's own previous output, unaffected by that fix (see module
        docstring's "chained continuation steps... NOT affected").
        """
        cache = self._caches[slot]
        mtp_hidden = seed_hidden
        next_input = torch.tensor([[seed_token]], dtype=torch.long, device=self.device)
        drafts: list[int] = []
        for _ in range(self.k):
            draft_token, mtp_hidden = self.model.mtp_step(
                next_input, mtp_hidden, cache.seq_len, cache
            )
            drafts.append(int(draft_token.item()))
            next_input = draft_token.view(1, 1)
        return drafts

    def draft_after_prefill(
        self, slot: int, *, first_token: int, pred_hidden: torch.Tensor
    ) -> list[int]:
        """Seed the very first round's drafts right after a (cold or
        prefix-cache-hit) prefill.

        ``pred_hidden``: the hidden state at the LAST prompt/suffix
        position -- i.e. the same hidden whose argmax IS ``first_token``
        (``Qwen36Backend._prefill_forward``'s own last-position output,
        which every existing caller already discards after taking its
        argmax; MTP is the first caller that needs it kept). This is
        already the correctly-shifted seed the module docstring describes
        -- no extra forward needed, unlike every prior script's bootstrap
        (which discarded exactly this value and recomputed a
        WRONG one instead -- see module docstring).

        Caller must have reset this slot's MTP cache first (a fresh
        ``prefill_chunked_begin`` admission always does, via
        ``Qwen36Backend.reset_slot``).
        """
        if self._caches[slot].seq_len != 0:
            raise RuntimeError(
                f"draft_after_prefill: slot {slot} mtp cache is not fresh "
                f"(seq_len={self._caches[slot].seq_len}); caller must reset_slot first"
            )
        return self._draft_loop(slot, first_token, pred_hidden)

    # -- verify + commit + re-draft (the hot path) ------------------------

    def round(
        self,
        slot: int,
        anchor_token: int,
        drafts: list[int],
        *,
        params: SamplingParams | None = None,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> dict[str, Any]:
        """One draft-verify-accept/reject-redraft round for ``slot``.

        Returns the same decision-dict contract
        ``LagunaBackend.mtp_verify_and_commit_batch``/``DFlashEngine
        .dflash_round`` already established: ``committed`` (the newly
        committed real token ids, accepted-draft prefix plus exactly one
        recovery/bonus token), ``num_accepted``, ``next_anchor``,
        ``next_draft_tokens``[, ``logprobs``].

        Every quantity needed to seed the NEXT round's draft-loop step 0
        (:meth:`_draft_loop`'s ``seed_hidden``) comes from THIS round's own
        anchor-advance + verify forward -- see the module docstring: the
        target hidden that predicted ``anchor_token`` (from THIS round's
        own bootstrap or the previous round's own such value) is combined
        with ``verify_hidden`` into one ``[1, k+1, H]`` tensor
        (``all_hiddens``) where row ``m`` (``m`` = accepted count) is
        exactly the hidden state that predicted ``next_anchor`` -- no extra
        forward pass beyond the one every round already needs to advance
        ``state`` through ``anchor_token`` before ``verify_forward`` can
        run (``Qwen36TextModelSelfBuilt.verify_forward``'s own precondition:
        "state already reflects having processed the anchor").
        """
        pool = self.backend.pool
        state = pool.slot_state(slot)
        cache = self._caches[slot]
        k = len(drafts)
        entering_seq_len = cache.seq_len
        round_mtp_start = entering_seq_len - k
        if round_mtp_start < 0:
            raise RuntimeError(
                f"round: slot {slot} mtp cache (seq_len={entering_seq_len}) is shorter "
                f"than k={k} drafts -- draft_after_prefill/round must have written "
                "exactly k rows before this call"
            )

        # -- (a) advance the target through anchor_token. Structurally
        # required regardless of the fix below: verify_forward's own
        # precondition is that `state` already reflects having processed
        # the anchor, and nothing else in this round's flow ever forwards
        # anchor_token through the target -- accept/reject only ever
        # produces token IDS, never runs them through the model. This is
        # the SAME forward every prior script already did here; the only
        # change is that this module also KEEPS its hidden state for the
        # correctly-shifted purpose the module docstring describes,
        # instead of discarding it and reusing anchor_hidden (hidden AFTER,
        # not before, anchor_token) for that purpose.
        anchor_input = torch.tensor([[anchor_token]], dtype=torch.long, device=self.device)
        anchor_hidden = self.model(anchor_input, state)  # [1, 1, H]
        anchor_logits = self.model.compute_logits(anchor_hidden)[0]  # [1, vocab]

        # -- (b) verify the K drafts in one pass.
        past_len = state.num_tokens_seen
        draft_tensor = torch.tensor([drafts], dtype=torch.long, device=self.device)
        verify_hidden, gdn_snapshots = self.model.verify_forward(draft_tensor, state)
        verify_logits = self.model.compute_logits(verify_hidden)[0]  # [k, vocab]

        all_hiddens = torch.cat([anchor_hidden, verify_hidden], dim=1)  # [1, k+1, H]
        all_logits = torch.cat([anchor_logits, verify_logits], dim=0)  # [k+1, vocab]

        sampled = params is not None and not params.is_greedy
        if sampled:
            draft_probs = torch.zeros(k, self.vocab_size, dtype=torch.float32, device=self.device)
            draft_rows = torch.arange(k, device=self.device)
            draft_cols = torch.tensor(drafts, device=self.device)
            draft_probs[draft_rows, draft_cols] = 1.0
            target_probs = compute_sampling_distribution(all_logits.float(), params)
            generator = make_generator(params.seed, str(self.device))
            decision = sample_accept_reject(
                list(drafts), draft_probs, target_probs, generator=generator
            )
            self.stats["sampled_rounds"] += 1
        else:
            predicted_tokens = all_logits.argmax(dim=-1).tolist()
            decision = determine_accept_reject_from_predictions(
                [anchor_token] + list(drafts), predicted_tokens
            )

        m = decision["num_accepted"]
        committed: list[int] = decision["committed"]
        new_anchor = committed[-1]

        # -- (c) commit: roll the target's GDN/attention state back to the
        # accepted prefix (m real tokens past the anchor).
        self.model.commit_verify(state, gdn_snapshots, past_len=past_len, accepted_count=m)

        # -- KV/committed-token bookkeeping, "committed ahead of kv by one"
        # (the same invariant DFlash's own round already keeps for Laguna
        # -- runtime/backends/laguna_dflash.py's dflash_round): the FINAL
        # committed token (recovery/bonus) is client-visible now but its
        # own KV write is deferred to the NEXT round's anchor-advance step
        # (a)); `committed` extends slot_committed_tokens in full, while
        # slot_kv_len only advances by what commit_verify actually wrote.
        pool.slot_kv_len[slot] = state.num_tokens_seen
        pool.slot_committed_tokens[slot].extend(committed)
        self.backend._maybe_checkpoint(slot)  # noqa: SLF001 -- friend-class access, same pattern DFlash uses

        # -- (d) truncate mtp_cache's exploratory tail; optionally resync
        # the interior accepted positions first.
        cache.seq_len = round_mtp_start + m
        if self.enable_resync and m >= 2:
            self._resync(cache, round_mtp_start, drafts, all_hiddens, m)
            self.stats["resync_rounds"] += 1

        # -- (e) draft the NEXT round, correctly-shifted seed (module
        # docstring's fix): row `m` of all_hiddens is exactly the hidden
        # state that predicted new_anchor (row 0 = anchor_hidden, used when
        # m==0; rows 1..k = verify_hidden[0..k-1], used when m>=1 -- the
        # SAME row verify_argmax[m-1] was read from to decide new_anchor's
        # own identity, so this is never a fresh computation, only a kept
        # reference to one already made).
        pred_hidden_next = all_hiddens[:, m : m + 1, :]
        next_drafts = self._draft_loop(slot, new_anchor, pred_hidden_next)

        self.stats["rounds"] += 1

        result: dict[str, Any] = {
            "committed": committed,
            "num_accepted": m,
            "next_anchor": new_anchor,
            "next_draft_tokens": next_drafts,
        }
        if return_logprobs:
            result["logprobs"] = [
                compute_logprobs(all_logits[p : p + 1], [committed[p]], top_k=top_logprobs)[0]
                for p in range(len(committed))
            ]
        return result

    # -- optional per-round resync (QSR_SERVER_MTP_RESYNC) ----------------

    def _resync(
        self,
        cache: Qwen36PagedAttentionCache,
        round_mtp_start: int,
        drafts: list[int],
        all_hiddens: torch.Tensor,
        m: int,
    ) -> None:
        """Re-ground mtp_cache rows ``round_mtp_start+1 .. round_mtp_start+
        m-1`` (the ``m-1`` INTERIOR accepted draft positions) with the
        target's own real hidden states from this round's verify forward,
        overwriting the draft head's self-chained (possibly drifted)
        hidden state from the exploratory loop that first drafted them.

        Row ``round_mtp_start`` (``drafts[0]``) is never touched -- it was
        already written using the correctly-shifted seed in the PREVIOUS
        round's draft loop (:meth:`_draft_loop`'s ``seed_hidden``), so it
        is already grounded in real target data, not drifted. The row for
        ``new_anchor`` (position ``round_mtp_start+m``) does not exist yet
        at all -- it is written by THIS round's own re-draft step (e),
        using ``pred_hidden_next`` as its seed, which is itself real target
        data (row ``m`` of ``all_hiddens``). So exactly ``m-1`` rows -- the
        accepted drafts strictly between the first and the last -- ever
        need rewriting.

        Uses the correctly-shifted pairing throughout (module docstring):
        row ``round_mtp_start+j`` (``j=1..m-1``) is re-written with
        ``(token=drafts[j], hidden=verify_hidden[j-1])`` -- the hidden
        state that PREDICTED ``drafts[j]``, one position before it, not
        the hidden state produced BY processing it. This differs from
        ``work/mtp-resync-20260802``'s port (``aed0e2d``,
        ``scripts/mtp_resync_ab_sweep.py``), which pairs
        ``(drafts[i], verify_hidden[i])`` for ``i=0..m-2`` starting at
        ``round_mtp_start+1`` -- same starting row, but the unshifted
        pairing that port's own commit message names explicitly. Not a
        byte-for-byte reuse of that port on purpose.
        """
        resync_tokens = torch.tensor([drafts[1:m]], dtype=torch.long, device=self.device)
        resync_hidden = all_hiddens[:, 1:m, :]  # verify_hidden[0 .. m-2]
        cache.seq_len = round_mtp_start + 1
        self.model.mtp_resync_step(resync_tokens, resync_hidden, round_mtp_start + 1, cache)
        if cache.seq_len != round_mtp_start + m:
            raise RuntimeError(
                f"resync: mtp cache landed at {cache.seq_len}, expected {round_mtp_start + m}"
            )
