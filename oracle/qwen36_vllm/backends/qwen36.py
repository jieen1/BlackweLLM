"""E1 Phase 2: Qwen3.6-27B MTP backend.

Extracted from DirectModelRunner -- contains all MTP (Multi-Token Prediction)
draft-model methods: forward primitives, sync/propose coordinators, and
high-level prefill/verify entry points.

The backend holds a reference to the runner (``self._r``) for accessing
shared infrastructure (KV caches, block pool, CUDA graphs, GDN state, etc.).
This is a deliberate composition choice: the MTP logic is model-specific
but depends on model-agnostic infrastructure that stays in the runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from oracle.qwen36_vllm.direct_model_runner import DirectModelRunner

from oracle.qwen36_vllm.metadata_builders import (
    _MAX_DECODE_QO_LEN,
    build_attention_metadata,
    build_attention_metadata_batch,
)
from oracle.qwen36_vllm.vllm_compat import set_forward_context
from runtime.logprobs import compute_logprobs
from runtime.mtp_accept import determine_accept_reject, determine_accept_reject_batch
from server.metrics import (
    record_mtp_acceptance,
    record_prefix_cache_hit,
    record_prefix_cache_miss,
    record_slot_kv_usage,
)


class Qwen36Backend:
    """Qwen3.6-27B MTP draft-model operations.

    All methods were mechanically extracted from DirectModelRunner.
    ``self._r`` is the owning runner instance.
    """

    def __init__(self, runner: DirectModelRunner) -> None:
        self._r = runner

    def verify_batch_spec(
        self,
        slot_ids: list[int],
        draft_token_ids: list[list[int]],
        kv_lengths: list[int],
        *,
        num_accepted_tokens_prev: list[int],
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """MTP/speculative-decode verify via the REAL spec-decode GDN
        mechanism (Phase 2, 2026-07-18) -- ``verify_batch``'s sibling,
        originally for ``mtp_verify_and_commit_batch`` only, and (Phase B,
        same day) for ``mtp_verify_and_commit`` too (called at
        ``len(slot_ids)==1``) -- both production verify paths share this
        method now. Same call shape/return convention as ``verify_batch``
        (raw logits AND hidden states, request-then-position order,
        ``commit=False`` -- caller advances
        ``slot_kv_len``/``slot_num_accepted_tokens`` after accept/reject).
        The only difference is GDN metadata construction: K+1 dedicated
        SSM state rows per slot (``build_gdn_metadata_spec_batch``,
        ``_ssm_spec_row``) instead of the chunked/prefill-shaped path,
        so a partial reject needs no snapshot/restore or recompute-forward
        repair -- the "wrong" candidates' rows are simply never read by
        any future round. ``num_accepted_tokens_prev[i]`` is slot
        ``slot_ids[i]``'s real committed length from its own last verify
        round (or exactly 1 on a slot's first-ever verify right after a
        real prefill -- see ``build_gdn_metadata_spec_batch``'s
        docstring). See notes/2026-07-17-post-ragged-round-next-steps.md
        section 10/11 for the derivation and validation history that
        underlies this method."""
        qo_len = len(draft_token_ids[0]) if draft_token_ids else 0
        return self._r._forward_batch(
            slot_ids,
            draft_token_ids,
            kv_lengths,
            qo_len=qo_len,
            commit=False,
            return_hidden=return_hidden,
            fixed_kv_split_size=self._r.decode_fixed_kv_split_size,
            fixed_max_num_splits=self._r.decode_fixed_max_num_splits,
            gdn_spec_num_accepted_tokens_prev=num_accepted_tokens_prev,
        )

    def _mtp_forward(
        self,
        slot: int,
        token_ids: list[int],
        hidden_states_in: torch.Tensor,
        start_pos: int,
        *,
        prior_kv_len: int,
        is_decode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Real draft-model (``Qwen3_5MTP``) forward for ONE slot -- the
        low-level primitive the centralized MTP-cycle coordinator methods
        (``mtp_prefill``/``mtp_verify_and_commit``) build on.

        ``prior_kv_len`` (2026-07-17 fix -- see below) is an EXPLICIT
        caller-supplied argument, NOT read from ``self._r.slot_draft_sync_len``
        internally as an earlier version of this method did. This method
        does NOT touch ``self._r.slot_draft_sync_len`` itself either way (the
        caller decides whether this call's advance represents the real
        synced history -- step 0, teacher-forced with the target's own
        just-computed hidden state -- or a throwaway exploratory propose
        step -- steps 1..K-1, autoregressive on the draft's own previous
        hidden state/token -- that must NOT be counted as committed). This
        is what makes the draft model's own KV cache need no explicit
        rollback on accept/reject, unlike GDN: an exploratory step's
        positions are simply overwritten by the next round's real sync
        call, exactly like attention's own content/position-addressed
        reasoning elsewhere in this file.

        **2026-07-17 real bug, caught by an independent Codex-sol review
        and independently re-verified by the coordinator before being
        relayed**: this method used to read ``prior_kv_len=self
        .slot_draft_sync_len[slot]`` directly. That field is deliberately
        NOT updated after step 0 (see above) -- correct for THAT field's
        job (tracking the real committed sync length across rounds), but
        WRONG when reused as this call's OWN attention-metadata history
        length for the exploratory loop's 2nd-and-later steps: those
        steps' actual physical write position (``start_pos``, which DOES
        advance every exploratory iteration in
        ``_mtp_sync_and_propose``) drifts away from the frozen
        ``slot_draft_sync_len``, so the attention metadata told the
        kernel a SMALLER history length than where the write actually
        landed -- the exploratory step's own query would then fail to
        attend to the PREVIOUS exploratory step's just-written K/V (it
        wasn't in the "prior" range the metadata declared), silently
        computing over an incomplete/wrong causal history for every
        exploratory step from the 2nd one onward. K=3 (this project's
        real production setting) has exactly 2 exploratory steps, so the
        1st (which happens to immediately follow step 0, where the frozen
        field and the real position still coincide) was fine, but the 2nd
        was not -- meaning every real K=3 proposal's 3rd draft token was
        computed against a subtly wrong causal history. Fixed by making
        the caller (``_mtp_sync_and_propose``) track its own LOCAL running
        prior-length counter that DOES advance every exploratory
        iteration, passed in here explicitly, while ``self
        .slot_draft_sync_len`` itself still only updates once (after step
        0) -- decoupling "what this call's attention needs" from "what the
        cross-round bookkeeping should remember" fixes both correctly at
        once. Not shape-checkable (see notes/direct-model-runner-design.md's
        2026-07-17 methodology-fix entry): this bug produced the right
        SHAPE and vocab-range output at every step, only the CONTENT was
        wrong from the 2nd exploratory step on -- exactly why the
        verification gradient's steps 3-4 (shape/length checks only) never
        caught it, and why the fix needed a per-step oracle-aligned logits
        comparison, not another shape check, to confirm.

        ``hidden_states_in`` must have exactly ``len(token_ids)`` rows
        (``Qwen3_5MultiTokenPredictor.forward()`` concatenates it against
        the embedded ``input_ids`` along the hidden-size dim, so the
        sequence-length dim must already match)."""
        if self._r.mtp_model is None:
            raise RuntimeError(
                "no MTP draft model loaded -- build_vllm_config(speculative_config=...) first"
            )
        num_new_tokens = len(token_ids)
        # P1 (notes/prefix-cache-design.md sec 5): the draft layer shares
        # the SAME block-id namespace as the target's attention group (sec
        # 3.1), so its own forward must grow self._r.block_table[slot] too --
        # using prior_kv_len (== start_pos at every real call site, see
        # this method's docstring) + num_new_tokens, exactly what
        # build_attention_metadata below uses as its own new_kv_len.
        if self._r.enable_block_table:
            self._r._ensure_blocks(slot, prior_kv_len + num_new_tokens)
        attn_meta = build_attention_metadata(
            prior_kv_len=prior_kv_len,
            num_new_tokens=num_new_tokens,
            is_decode=is_decode,
            slot=slot,
            block_size=self._r.block_size,
            blocks_per_slot=self._r.blocks_per_slot,
            device=self._r.device,
            block_table=self._r.block_table[slot] if self._r.enable_block_table else None,
        )
        attn_metadata_dict = {name: attn_meta for name in self._r.mtp_attn_layer_names}
        slot_mapping = self._r._slot_mapping(slot, start_pos, num_new_tokens)
        slot_mapping_dict = {name: slot_mapping for name in self._r.mtp_attn_layer_names}

        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self._r.device)
        positions = torch.arange(
            start_pos, start_pos + num_new_tokens, dtype=torch.long, device=self._r.device
        )

        with set_forward_context(
            attn_metadata_dict, self._r.vllm_config, slot_mapping=slot_mapping_dict
        ):
            hidden_states_out = self._r.mtp_model.forward(input_ids, positions, hidden_states_in)
        # 2026-07-17, Phase 3: see ``_forward``'s docstring/comment -- same
        # same-stream-ordering reasoning applies to the draft model's own
        # forward+compute_logits pair.
        logits = self._r.mtp_model.compute_logits(hidden_states_out)
        return logits, hidden_states_out

    def _mtp_sync_and_propose(
        self,
        slot: int,
        shifted_input_ids: list[int],
        target_hidden_states: torch.Tensor,
        start_pos: int,
        num_new_tokens: int,
        k: int,
    ) -> list[int]:
        """The centralized sync+propose funnel every MTP-aware entry point
        (``mtp_prefill``/eventual ``mtp_decode``) routes through -- per
        2026-07-17's sol-refined design, this is the ONE place draft-sync
        logic lives, not duplicated per public entry point. Step 0 is the
        real sync (teacher-forced with the target's OWN just-computed
        hidden states, covering this step's FULL real query range --
        matches vLLM's real ``_prepare_prefill_inputs_kernel`` shift-by-one
        mechanism); steps 1..k-1 are genuinely autoregressive on the
        draft's own previous hidden state/token, and are NOT committed to
        ``self._r.slot_draft_sync_len`` (see ``_mtp_forward``'s docstring).

        **2026-07-17 fix**: tracks its OWN local ``running_prior_kv_len``
        counter, separate from ``self._r.slot_draft_sync_len`` -- the local
        counter advances every exploratory iteration (matching where each
        step's write actually lands), while the persistent field only
        ever advances once, after step 0. Passing the persistent field
        directly into every exploratory `_mtp_forward` call (the previous,
        buggy version) left steps 2..k-1's attention metadata pointing at
        a stale, non-advancing history length -- see `_mtp_forward`'s
        docstring for the full analysis of what that broke."""
        step0_logits, step0_hidden = self._mtp_forward(
            slot,
            shifted_input_ids,
            target_hidden_states,
            start_pos,
            prior_kv_len=self._r.slot_draft_sync_len[slot],
            is_decode=(num_new_tokens == 1),
        )
        self._r.slot_draft_sync_len[slot] += num_new_tokens
        draft_tokens = [int(step0_logits[-1].argmax(dim=-1).item())]
        prev_hidden = step0_hidden[-1:]
        prev_token = draft_tokens[0]
        next_pos = start_pos + num_new_tokens
        running_prior_kv_len = self._r.slot_draft_sync_len[slot]
        for _ in range(1, k):
            step_logits, step_hidden = self._mtp_forward(
                slot,
                [prev_token],
                prev_hidden,
                next_pos,
                prior_kv_len=running_prior_kv_len,
                is_decode=True,
            )
            prev_token = int(step_logits[-1].argmax(dim=-1).item())
            draft_tokens.append(prev_token)
            prev_hidden = step_hidden[-1:]
            next_pos += 1
            running_prior_kv_len += 1
        return draft_tokens

    def mtp_prefill(self, slot: int, prompt_token_ids: list[int]) -> dict:
        """Unified MTP cycle funnel point for a fresh prefill: real target
        prefill (with hidden states) -> draft KV sync (step 0, teacher-
        forced shift over the WHOLE prompt) -> K-1 more autoregressive
        draft steps. Returns the anchor (target's own greedy next token,
        matching plain ``prefill()``'s contract -- not yet written into
        the target's own KV) and the K proposed draft tokens, ready for
        the caller to submit through ``mtp_verify_and_commit``."""
        if self._r.mtp_model is None or self._r.num_speculative_tokens is None:
            raise RuntimeError("no MTP draft model loaded")
        if self._r.slot_kv_len[slot] != 0 or self._r.slot_draft_sync_len[slot] != 0:
            raise RuntimeError(f"slot {slot} is not fresh")
        # Phase 2/Phase B (2026-07-18), defense in depth: bootstrap value
        # for the spec-decode GDN mechanism's first-ever verify round on
        # this slot -- already 1 via __init__/reset_slot for any slot that
        # actually went through one of those, set explicitly here too so
        # this invariant holds regardless of how the slot got to "fresh"
        # (mirrors mtp_prefill_batch's identical defense-in-depth line).
        self._r.slot_num_accepted_tokens[slot] = 1
        target_logits, target_hidden = self._r._forward(
            slot, prompt_token_ids, start_pos=0, is_decode=False, return_hidden=True
        )
        anchor = int(target_logits[-1].argmax(dim=-1).item())
        shifted_input_ids = prompt_token_ids[1:] + [anchor]
        draft_tokens = self._mtp_sync_and_propose(
            slot,
            shifted_input_ids,
            target_hidden,
            start_pos=0,
            num_new_tokens=len(prompt_token_ids),
            k=self._r.num_speculative_tokens,
        )
        self._r.slot_pending_draft_tokens[slot] = draft_tokens
        # P3 populate-on-completion (attention half): publish this prefill's
        # full committed blocks to the content index. The GDN completion
        # checkpoint is NOT materialized here -- a single-shot prefill's live
        # GDN state is at prompt_len, not at the block-aligned completion
        # boundary G = block_align_down(prompt_len - 1), so a correct checkpoint
        # at G needs a forward that ENDS at G (mtp_prefill_with_cache's two-phase
        # cold path / the fan-out leader at Lc). Publishing attention only here
        # is safe: a later hit finds A>0 but G=0 => compute miss (L=0, cold
        # recompute), never a wrong-prefix serve (sec 3.4).
        if self._r.enable_persistent_prefix_cache:
            self._r._publish_committed_blocks(slot, prompt_token_ids, len(prompt_token_ids))
        return {"anchor": anchor, "draft_tokens": draft_tokens}

    def mtp_verify_and_commit(self, slot: int, anchor: int, draft_tokens: list[int]) -> dict:
        """Unified MTP cycle funnel point for verify+commit+resync+propose
        -- the ONE method a real multi-round loop calls repeatedly for a
        SINGLE slot (no separate "decode" coordinator needed; see the
        design note below on why).

        **2026-07-18, Phase B migration** (independent review's
        ``notes/2026-07-18-session-review-and-next-steps.md`` Phase B,
        option (a); see this session's own addendum for the falsifier
        check and the result): now uses the REAL spec-decode GDN mechanism
        (``verify_batch_spec``/``build_gdn_metadata_spec_batch``/
        ``_ssm_spec_row``) -- the exact same mechanism
        ``mtp_verify_and_commit_batch`` adopted in Phase 2, applied here at
        batch_size=1 -- instead of the old chunked-GDN-metadata +
        ``snapshot_gdn_state``/``restore_gdn_state`` + recompute-forward
        mechanism this method used through 2026-07-18. This was the last
        production call site of the old mechanism. ``snapshot_gdn_state``/
        ``restore_gdn_state`` are NOT deleted as a result: the falsifier
        check for this migration (before touching any code) confirmed
        ``benchmarks/mtp_gdn_rollback_check.py`` tests them directly as
        primitives (snapshot/restore around a real multi-step "detour",
        with no MTP verify call involved at all), and several other
        diagnostics (``mtp_real_draft_check.py``, ``mtp_trace_driven_probe.py``,
        ``mtp_slot_identity_pinpoint_diag.py``, ``mtp_batch_divergence_diag.py``,
        ``phase0_nsys_gap_ledger_diag.py``) call them directly too -- they
        remain in the codebase as tested, live (if no longer
        production-verify-path-connected) primitives. The old chunked
        ``verify_batch`` is retained for the same reason (still called
        directly by several diagnostics and by ``decode_batch``'s qo_len=1
        path via ``build_gdn_metadata_batch``).

        Why no accept/reject branch is needed any more (mirrors
        ``mtp_verify_and_commit_batch``'s own docstring): GDN's recurrent
        state, under the real spec-decode kernel, computes a causally-valid
        PER-POSITION output for every one of the K+1 candidate positions in
        a single verify forward, unconditionally -- only the recurrent
        STATE COMMIT (which physical row survives to be read next round) is
        acceptance-aware, via ``num_accepted_tokens``/``_ssm_spec_row``.
        This slot's hidden states for positions ``0..committed_len-1`` are
        therefore already sitting in ``verify_hidden``, correct, from the
        ONE verify forward this method issues -- a plain slice (never a
        second forward pass) is all the draft resync step needs, for a
        full accept exactly as much as for any partial reject.

        Persistent per-slot bookkeeping this method updates:
        ``self._r.slot_num_accepted_tokens[slot]`` (this slot's real committed
        length from ITS OWN last verify round, or bootstrap 1 right after a
        real ``mtp_prefill``) -- read by ``build_gdn_metadata_spec_batch``
        (via ``verify_batch_spec``) to select which of last round's K+1
        dedicated SSM rows holds the valid state to resume from.

        Recompute input alignment (2026-07-17, unchanged by this
        migration -- see notes/direct-model-runner-design.md): the token
        whose OWN K/V gets written at position ``kv_len_before + i`` is
        the i-th QUERY INPUT of the verify forward, matching
        ``verify_batch``/``verify_batch_spec``'s shared convention where
        ``draft[0]=anchor``'s K/V lands at ``kv_len_before`` (mirroring
        ``prefill()``/``decode()``'s established contract: the
        anchor/greedy-next token is NOT written into KV until it is fed
        back in as the FOLLOWING call's input). ``decision["committed"]``
        is ``[accepted_draft_0, ..., accepted_draft_{n-1}, recovery]`` --
        the recovery/bonus token is, symmetrically, NOT yet written into KV
        either (it becomes the next round's own anchor-equivalent). So the
        real input tokens for positions ``kv_len_before..+committed_len-1``
        are ``real_new_tokens = [anchor] + committed[:-1]`` (anchor +
        accepted drafts, dropping the not-yet-written recovery token) --
        NOT ``committed`` itself, which would silently write the WRONG
        token content into the KV cache while still looking correct on
        every shape/length/bookkeeping check.

        Draft catch-up + next-round propose, folded into ONE call
        (2026-07-17 multi-round design, unchanged by this migration): after
        committing, the draft's own KV is behind by exactly
        ``real_new_tokens`` (it was last synced at the END of the PREVIOUS
        round -- ``mtp_prefill``/this same method -- so
        ``slot_draft_sync_len`` always equals ``slot_kv_len`` from BEFORE
        this round's commit). Syncing the draft over ``real_new_tokens``
        (shifted by one, ending in the recovery/bonus token as the final
        candidate -- exactly ``_mtp_sync_and_propose``'s existing step-0
        pattern, just generalized from ``mtp_prefill``'s "whole prompt"
        range to "this round's newly committed range") both catches the
        draft's KV up to ``slot_kv_len`` again (restoring the invariant)
        AND, at that same call's LAST position (processing the
        recovery/bonus token as a candidate against the target's hidden
        state up through the last real position), produces the FIRST draft
        token for the NEXT round -- for free, no extra forward call.
        ``_mtp_sync_and_propose`` then runs the usual K-1 further
        autoregressive steps on top.

        Returns the accept/reject decision plus ``next_anchor`` (the
        recovery/bonus token -- feed this as ``anchor`` to the NEXT
        ``mtp_verify_and_commit`` call) and ``next_draft_tokens`` (K fresh
        proposed tokens for that next call)."""
        k = len(draft_tokens)
        draft = [anchor] + draft_tokens
        kv_len_before = self._r.slot_kv_len[slot]
        num_accepted_prev = self._r.slot_num_accepted_tokens[slot]

        verify_logits, verify_hidden = self.verify_batch_spec(
            [slot],
            [draft],
            [kv_len_before],
            num_accepted_tokens_prev=[num_accepted_prev],
            return_hidden=True,
        )
        decision = determine_accept_reject(draft, verify_logits)
        record_mtp_acceptance(decision["num_accepted"])
        committed_len = decision["num_accepted"] + 1
        # Real input tokens for positions kv_len_before..+committed_len-1:
        # anchor followed by the accepted drafts (NOT the recovery token --
        # see the docstring above).
        real_new_tokens = [anchor] + decision["committed"][:-1]

        self._r.slot_kv_len[slot] = kv_len_before + committed_len
        self._r.slot_num_accepted_tokens[slot] = committed_len
        # P3.2 decode-position populate: publish any newly-FULL committed blocks
        # now that slot_kv_len advanced by the REAL committed length (only
        # committed tokens; INV4). No-op off-flag.
        self._r.publish_committed_decode_blocks(slot, real_new_tokens)
        # Ragged slice of the ONE verify forward's hidden states -- valid
        # for a full accept exactly as much as for any partial reject, see
        # the docstring above.
        real_new_hidden = verify_hidden[:committed_len]

        next_anchor = decision["committed"][-1]
        next_draft_tokens = self._mtp_sync_and_propose(
            slot,
            real_new_tokens[1:] + [next_anchor],
            real_new_hidden,
            start_pos=self._r.slot_draft_sync_len[slot],
            num_new_tokens=committed_len,
            k=k,
        )
        self._r.slot_pending_draft_tokens[slot] = next_draft_tokens
        return {**decision, "next_anchor": next_anchor, "next_draft_tokens": next_draft_tokens}

    # ------------------------------------------------------------------
    # True cross-slot batched MTP (2026-07-17 round): one shared forward
    # pass -- draft model included, not just the target -- across every
    # listed slot, instead of a Python loop calling the single-slot
    # methods above once per slot. Mirrors how ``_forward_batch``/
    # ``decode_batch``/``verify_batch`` already batch the plain
    # (non-MTP-aware) target-only path; these are the MTP analogues.
    # ------------------------------------------------------------------

    def _mtp_forward_batch(
        self,
        slots: list[int],
        token_ids,
        hidden_states_in: torch.Tensor,
        prior_kv_lens: list[int],
        start_pos_list: list[int],
        *,
        qo_len: int | list[int],
        is_decode: bool,
        logits_last_position_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched analogue of ``_mtp_forward`` for the draft model
        (``Qwen3_5MTP``) -- ONE batched attention-metadata object (scoped to
        ``self._r.mtp_attn_layer_names``, no GDN metadata since the draft model
        registers no GDN layers -- see ``__init__``'s ``mtp_attn_layer_names``
        derivation) and ONE ``mtp_model.forward()`` call covering every
        listed slot. ``qo_len`` may be a per-slot RAGGED list (2026-07-17,
        for the recompute-fallback batching round -- mirrors
        ``build_attention_metadata_batch``'s identical generalization,
        forwarded through as-is); a scalar broadcasts to a uniform list,
        so every existing call site is byte-for-byte unaffected.

        ``prior_kv_lens``/``start_pos_list`` are explicit per-slot lists,
        kept as TWO separate parameters (not collapsed into one) to mirror
        ``_mtp_forward``'s own explicit-argument design (its 2026-07-17 bug
        fix -- see that method's docstring): at every real call site in this
        file the two values coincide, but they mean different things
        (attention-metadata history length vs. physical write/embedding
        position) and collapsing them would silently re-introduce the same
        class of bug that fix addressed, just in the batched path instead
        of the single-slot one.

        ``token_ids`` is a flat list (one token id per slot, in ``slots``
        order) when ``qo_len`` is the literal scalar ``1``, or a list of
        per-slot token-id lists (each of length ``qo_len[i]``, possibly
        ragged) otherwise -- same convention ``_forward_batch`` uses.
        Returns logits/hidden_states shaped ``[sum(qo_lens), ...]`` in
        request-then-position order.

        ``logits_last_position_only`` (2026-07-18, D1-followup fix, default
        ``False`` preserving every existing call site byte-for-byte): same
        contract as ``_forward_batch``'s identical parameter -- when
        ``True``, BOTH the returned ``logits`` AND the returned
        ``hidden_states_out`` are gathered down to only each slot's own
        last position (shape ``[num_reqs, ...]``, not
        ``[sum(qo_lens), ...]``) before/after ``compute_logits``. Safe here
        specifically because ``_mtp_sync_and_propose_batch`` (this
        method's only caller) already discards every non-last-position row
        of both return values via its own ``index_select`` immediately
        after calling this method, for every existing call site -- so
        gathering earlier (before the vocab-head projection, instead of
        after) changes nothing observable except removing the wasted
        compute/memory. Only ``_mtp_sync_and_propose_batch``'s step-0 call
        passes this, and only when its own caller (``mtp_prefill_batch``)
        requests it -- see that call site.
        """
        if self._r.mtp_model is None:
            raise RuntimeError(
                "no MTP draft model loaded -- build_vllm_config(speculative_config=...) first"
            )
        num_reqs = len(slots)
        if not (len(prior_kv_lens) == num_reqs and len(start_pos_list) == num_reqs):
            raise ValueError("slots/prior_kv_lens/start_pos_list must have equal length")
        qo_lens = [qo_len] * num_reqs if isinstance(qo_len, int) else list(qo_len)
        if len(qo_lens) != num_reqs:
            raise ValueError("qo_len list must have exactly one entry per slot")
        if isinstance(qo_len, int) and qo_len == 1:
            if len(token_ids) != num_reqs:
                raise ValueError("token_ids must have one entry per slot when qo_len == 1")
            flat_token_ids = token_ids
        else:
            if not (
                len(token_ids) == num_reqs
                and all(len(t) == qo for t, qo in zip(token_ids, qo_lens))
            ):
                raise ValueError("every slot's token_ids must have exactly qo_len[i] entries")
            flat_token_ids = [tok for slot_tokens in token_ids for tok in slot_tokens]

        # P1 (notes/prefix-cache-design.md sec 5): same shared block-id
        # namespace reasoning as ``_mtp_forward`` -- grow every listed
        # slot's block_table to cover prior_kv_lens[i] + qo_lens[i] before
        # building metadata/slot-mapping below.
        if self._r.enable_block_table:
            for s, kv_len, qo in zip(slots, prior_kv_lens, qo_lens):
                self._r._ensure_blocks(s, kv_len + qo)

        attn_meta = build_attention_metadata_batch(
            slots=slots,
            prior_kv_lens=prior_kv_lens,
            block_size=self._r.block_size,
            blocks_per_slot=self._r.blocks_per_slot,
            device=self._r.device,
            qo_len=qo_len,
            is_decode=is_decode,
            # 2026-07-17: always pass this runner's fixed split-KV config
            # (see __init__'s comment) -- harmless when is_decode=False
            # (decode_qo_len ends up 0 either way, so the decode-kernel
            # dispatch never reads kv_split_size/max_num_splits for that
            # call), and gives the draft model's own decode/verify-shaped
            # calls real split-KV parallelism instead of collapsing to
            # max_num_splits=1.
            fixed_kv_split_size=self._r.decode_fixed_kv_split_size,
            fixed_max_num_splits=self._r.decode_fixed_max_num_splits,
            block_tables=(
                [self._r.block_table[s] for s in slots] if self._r.enable_block_table else None
            ),
        )
        attn_metadata_dict = {name: attn_meta for name in self._r.mtp_attn_layer_names}
        # Reuses ``_slot_mapping_batch`` (built for the target model's own
        # batched path) unchanged: its ``kv_lengths`` parameter is used only
        # as "each request's own write-start position", i.e. exactly what
        # ``start_pos_list`` means here -- the formula is identical to
        # concatenating per-slot ``_slot_mapping(slot, start_pos, qo_len)``
        # calls, verified by inspection (both compute
        # ``_physical_slot(slot) * blocks_per_slot + pos // block_size``
        # over the same ``start_pos + j`` positions).
        slot_mapping = self._r._slot_mapping_batch(slots, start_pos_list, qo_len=qo_len)
        slot_mapping_dict = {name: slot_mapping for name in self._r.mtp_attn_layer_names}

        input_ids = torch.tensor(flat_token_ids, dtype=torch.long, device=self._r.device)
        positions = torch.tensor(
            [start_pos + j for start_pos, qo in zip(start_pos_list, qo_lens) for j in range(qo)],
            dtype=torch.long,
            device=self._r.device,
        )
        with set_forward_context(
            attn_metadata_dict, self._r.vllm_config, slot_mapping=slot_mapping_dict
        ):
            hidden_states_out = self._r.mtp_model.forward(input_ids, positions, hidden_states_in)
        # 2026-07-17, Phase 3: see ``_forward``'s docstring/comment -- this
        # is the batched draft-model analogue, called up to K times per
        # round (previously 2 syncs each); same same-stream-ordering
        # reasoning applies.
        if logits_last_position_only:
            # 2026-07-18, D1-followup fix: see this parameter's docstring.
            last_idx = torch.tensor(
                [sum(qo_lens[: i + 1]) - 1 for i in range(num_reqs)],
                dtype=torch.long,
                device=self._r.device,
            )
            hidden_states_out = hidden_states_out.index_select(0, last_idx)
        logits = self._r.mtp_model.compute_logits(hidden_states_out)
        return logits, hidden_states_out

    def _mtp_run_continuation_steps(
        self,
        slots: list[int],
        draft_tokens: dict[int, list[int]],
        prev_tokens: list[int],
        prev_hidden: torch.Tensor,
        next_pos_list: list[int],
        running_prior_kv_len: list[int],
        k: int,
    ) -> None:
        """The k-1 batched autoregressive draft-continuation steps that
        follow step 0, appending each step's greedy token to
        ``draft_tokens[slot]`` IN PLACE (``draft_tokens`` must already hold
        step 0's own token for every slot in ``slots`` -- this method only
        APPENDS the remaining k-1).

        Extracted (2026-07-19, chunked-prefill round,
        ``notes/2026-07-18-session-review-and-next-steps.md`` section 19)
        from ``_mtp_sync_and_propose_batch``'s own tail as a shared helper:
        pure code motion, not a behavior change -- ``_mtp_sync_and_propose_batch``'s
        call site below passes exactly the same local variables its old
        inlined body used, in the same order. This is what lets
        ``mtp_prefill_batch``'s new chunked path (which computes step 0
        itself, chunk by chunk, so each chunk's target hidden states can be
        fed to the draft model as soon as they exist, rather than needing
        the whole prompt's hidden states materialized at once) reuse the
        EXACT same, already-verified continuation logic afterward instead
        of a second, independently-written copy of it -- the two entry
        points (whole-prompt step 0 vs. chunked step 0) differ only in how
        ``prev_tokens``/``prev_hidden``/``next_pos_list``/
        ``running_prior_kv_len`` were produced, not in what happens to them
        next.

        ``next_pos_list``/``running_prior_kv_len`` are each slot's
        position/prior-kv-len immediately after step 0 -- numerically
        identical to each other at entry in every real call (both start
        equal right after step 0 and both advance by exactly 1 every
        iteration below), the same invariant
        ``CapturedMTPDraftStepGraph``'s own docstring documents."""
        num_reqs = len(slots)
        draft_step_graph = (
            self._r._get_draft_step_graph(num_reqs) if self._r.enable_cudagraph else None
        )
        for _ in range(1, k):
            if draft_step_graph is not None:
                step_logits, step_hidden = draft_step_graph.replay_incremental(
                    slots, prev_tokens, prev_hidden, running_prior_kv_len
                )
            else:
                step_logits, step_hidden = self._mtp_forward_batch(
                    slots,
                    prev_tokens,
                    prev_hidden,
                    running_prior_kv_len,
                    next_pos_list,
                    qo_len=1,
                    is_decode=True,
                )
            # qo_len=1 uniform -> step_logits/step_hidden already have
            # exactly one row per slot in ``slots`` order (request-then-
            # position order degenerates to plain per-request order here),
            # so no index_select/cat needed -- a single batched argmax
            # over the whole tensor plus ONE ``.tolist()`` covers every
            # slot in this step.
            new_prev_tokens = step_logits.argmax(dim=-1).tolist()
            for i in range(num_reqs):
                draft_tokens[slots[i]].append(new_prev_tokens[i])
            prev_tokens = new_prev_tokens
            prev_hidden = step_hidden
            for i in range(num_reqs):
                next_pos_list[i] += 1
                running_prior_kv_len[i] += 1

    def _mtp_sync_and_propose_batch(
        self,
        slots: list[int],
        shifted_input_ids_per_slot: list[list[int]],
        target_hidden_states: torch.Tensor,
        start_pos_list: list[int],
        num_new_tokens: int | list[int],
        k: int,
        step0_logits_last_position_only: bool = False,
    ) -> dict[int, list[int]]:
        """Batched analogue of ``_mtp_sync_and_propose``: one batched step-0
        sync call, followed by ``k-1`` batched autoregressive steps (each
        contributing exactly 1 new token per slot -- always uniform,
        regardless of ``num_new_tokens``).

        ``num_new_tokens`` may be a per-slot RAGGED list (2026-07-17, for
        the recompute-fallback batching round -- mirrors
        ``_mtp_forward_batch``'s identical generalization): different
        slots' committed lengths from a partial-reject round genuinely
        differ, so a single shared scalar can no longer describe every
        real call site (``mtp_prefill_batch``'s uniform-prompt-length
        group and ``mtp_verify_and_commit_batch``'s full-accept group
        still pass a uniform value -- a scalar broadcasts to a uniform
        list, so those call sites are byte-for-byte unaffected;
        ``mtp_verify_and_commit_batch``'s recompute group is the new,
        genuinely-ragged caller).

        Tracks ONE ``running_prior_kv_len`` PER SLOT (a list, not a scalar)
        -- the direct batched generalization of the single-slot 2026-07-17
        fix documented on ``_mtp_forward``: each slot's own draft-sync
        length can differ even though every slot shares this call's ``k``,
        so a single shared counter would silently reintroduce the exact
        bug that fix addressed. Returns a dict keyed by slot id (not a
        positional list) so callers can freely pass a SUBSET of the
        runner's active slots (e.g. only the full-accept group from
        ``mtp_verify_and_commit_batch``) without index confusion.

        ``step0_logits_last_position_only`` (2026-07-18, D1-followup fix,
        default ``False`` preserving every existing call site byte-for-byte):
        forwarded to step 0's eager ``_mtp_forward_batch`` call (see its
        identical parameter's docstring) -- only ``mtp_prefill_batch`` sets
        this, since it already only ever reads each slot's own last-position
        draft token below, and it is the caller whose ``num_new_tokens``
        (a full prompt length) makes the full-sequence vocab-head
        projection this avoids a real cost. **Only actually takes effect
        when step 0 takes the EAGER branch** (``step0_graph is None``) --
        when a caller's ``num_new_tokens`` is uniform AND small enough for
        the captured-graph branch above (e.g. ``mtp_verify_cudagraph_check
        .py``'s deliberately-short-prompt regression test, with
        ``enable_cudagraph=True``), this parameter is silently a no-op: the
        graph path always returns the full, un-gathered shape (see
        ``step0_already_last_only`` below), which is harmless since that
        branch is only ever reached for ``num_new_tokens <=
        _MAX_DECODE_QO_LEN``, far too small for the projection cost to
        matter anyway.
        """
        num_reqs = len(slots)
        if not (len(shifted_input_ids_per_slot) == num_reqs and len(start_pos_list) == num_reqs):
            raise ValueError(
                "slots/shifted_input_ids_per_slot/start_pos_list must have equal length"
            )
        num_new_tokens_list = (
            [num_new_tokens] * num_reqs if isinstance(num_new_tokens, int) else list(num_new_tokens)
        )
        if len(num_new_tokens_list) != num_reqs:
            raise ValueError("num_new_tokens list must have exactly one entry per slot")
        if not all(len(t) == n for t, n in zip(shifted_input_ids_per_slot, num_new_tokens_list)):
            raise ValueError(
                "every slot's shifted_input_ids must have exactly num_new_tokens[i] entries"
            )

        prior_kv_lens_step0 = [self._r.slot_draft_sync_len[s] for s in slots]
        # 2026-07-17, Phase 3 round 2: step 0 (resync) is graph-capturable
        # whenever num_new_tokens is UNIFORM across this call's slots --
        # always true for the full-accept group (always k+1) and
        # sometimes true for the recompute group (see
        # ``CapturedMTPDraftStepGraph``'s docstring for the full
        # rationale, including why using the fast decode-kernel dispatch
        # here instead of eager's general-kernel choice is still
        # numerically correct). ``prior_kv_lens_step0 == start_pos_list``
        # always holds here (both derive from the same
        # ``slot_draft_sync_len`` snapshot), matching the class's own
        # single-length-list contract.
        # SAFETY (caught before ever running this in anger): mtp_prefill_batch
        # calls this SAME function with num_new_tokens=prompt_len -- a
        # LARGE uniform value (e.g. 4096 for a real W1-S prompt), which
        # would also pass the "uniform" check below. Forcing
        # decode_qo_len=qo_len unconditionally (this class's whole design,
        # matching CapturedBatchDecodeGraph's own convention) is only valid
        # within the real decode/verify kernel's tested range
        # (_MAX_DECODE_QO_LEN=16, the SAME bound
        # build_attention_metadata_batch's eager path already enforces
        # before routing to that kernel) -- MUST NOT be used for a genuine
        # long prefill, which needs the general/chunked kernel instead.
        step0_graph = None
        step0_qo_len_padded = 0
        if self._r.enable_cudagraph and max(num_new_tokens_list) <= _MAX_DECODE_QO_LEN:
            step0_qo_len_padded = max(num_new_tokens_list)
            step0_graph = self._r._get_draft_step_graph(num_reqs, step0_qo_len_padded)
        # 2026-07-18, D1-followup fix: whether step0's OWN return is already
        # gathered to last-position-only. NOT simply
        # ``step0_logits_last_position_only`` -- a caller (e.g.
        # ``mtp_prefill_batch`` invoked with a SHORT, uniform prompt, as
        # ``mtp_verify_cudagraph_check.py`` deliberately does to regression-
        # test this exact graph path) can request the optimization while
        # STILL legitimately taking the captured-graph branch above (short
        # prompt + ``enable_cudagraph=True`` -> ``num_new_tokens_list[0] <=
        # _MAX_DECODE_QO_LEN`` is true). The graph path always returns the
        # full, un-gathered ``[sum(qo_lens), ...]`` shape (its own
        # docstring/contract), so the optimization must only be treated as
        # "applied" when the eager ``_mtp_forward_batch`` branch actually
        # ran with the flag set -- otherwise this method's own last-index
        # bookkeeping below would silently read the wrong rows. Falling
        # back to full (correct, pre-existing) behavior on the graph path
        # is harmless: that path is only ever reached for small
        # ``num_new_tokens`` (<= ``_MAX_DECODE_QO_LEN``), where the
        # full-position vocab-head cost this fix targets was never large
        # enough to matter in the first place.
        step0_already_last_only = False
        if step0_graph is not None:
            step0_qo_len = step0_qo_len_padded
            is_uniform = len(set(num_new_tokens_list)) == 1
            if is_uniform:
                tokens_for_graph = (
                    [t[0] for t in shifted_input_ids_per_slot]
                    if step0_qo_len == 1
                    else shifted_input_ids_per_slot
                )
                hidden_for_graph = target_hidden_states
            else:
                tokens_for_graph = []
                hidden_rows = []
                row_start = 0
                for i, n in enumerate(num_new_tokens_list):
                    slot_tokens = list(shifted_input_ids_per_slot[i])
                    if n < step0_qo_len:
                        slot_tokens = slot_tokens + [slot_tokens[-1]] * (step0_qo_len - n)
                    tokens_for_graph.append(slot_tokens)
                    slot_hidden = target_hidden_states[row_start : row_start + n]
                    if n < step0_qo_len:
                        pad_rows = slot_hidden[-1:].expand(step0_qo_len - n, -1)
                        slot_hidden = torch.cat([slot_hidden, pad_rows], dim=0)
                    hidden_rows.append(slot_hidden)
                    row_start += n
                hidden_for_graph = torch.cat(hidden_rows, dim=0)
            step0_logits, step0_hidden = step0_graph.replay(
                slots, tokens_for_graph, hidden_for_graph, prior_kv_lens_step0
            )
            if not is_uniform:
                real_last_indices = torch.tensor(
                    [i * step0_qo_len + num_new_tokens_list[i] - 1 for i in range(num_reqs)],
                    dtype=torch.long,
                    device=step0_logits.device,
                )
                step0_logits = step0_logits.index_select(0, real_last_indices)
                step0_hidden = step0_hidden.index_select(0, real_last_indices)
                step0_already_last_only = True
        else:
            step0_logits, step0_hidden = self._mtp_forward_batch(
                slots,
                shifted_input_ids_per_slot,
                target_hidden_states,
                prior_kv_lens_step0,
                start_pos_list,
                qo_len=num_new_tokens,
                is_decode=all(n == 1 for n in num_new_tokens_list),
                logits_last_position_only=step0_logits_last_position_only,
            )
            step0_already_last_only = step0_logits_last_position_only
        for s, n in zip(slots, num_new_tokens_list):
            self._r.slot_draft_sync_len[s] += n

        # Per-request row offsets into step0's flattened [sum(num_new_tokens), ...]
        # output (request-then-position order) -- generalizes the old
        # ``i * num_new_tokens + num_new_tokens - 1`` uniform-stride formula
        # to a ragged cumulative-sum offset; reduces to the exact same
        # indices when num_new_tokens_list is uniform.
        row_offsets = [0]
        for n in num_new_tokens_list:
            row_offsets.append(row_offsets[-1] + n)

        # 2026-07-17, Phase 3: batched on-GPU argmax instead of a per-slot
        # Python loop each calling ``.argmax(dim=-1).item()`` separately
        # (``num_reqs`` sequential host round-trips per step, ``k`` steps
        # per round -- up to ``num_reqs * k`` total). ``index_select`` +
        # ONE ``.argmax(dim=-1)`` over every last-position row computes
        # every slot's step-0 draft token in one kernel launch; ONE
        # ``.tolist()`` is the single host round-trip for this step.
        draft_tokens: dict[int, list[int]] = {s: [] for s in slots}
        if step0_already_last_only:
            # Already gathered to [num_reqs, ...] (one row per slot, its
            # own last position) inside ``_mtp_forward_batch`` -- see that
            # method's docstring. Re-indexing with the OLD full-length
            # ``row_offsets`` formula here would read the wrong (or
            # out-of-bounds) rows, so use the tensors directly. Keyed off
            # ``step0_already_last_only`` (which branch ACTUALLY ran), not
            # the raw ``step0_logits_last_position_only`` parameter -- see
            # that variable's own definition above for why they can differ.
            last_logits = step0_logits
            prev_hidden = step0_hidden
        else:
            last_idx_tensor = torch.tensor(
                [row_offsets[i + 1] - 1 for i in range(num_reqs)],
                dtype=torch.long,
                device=step0_logits.device,
            )
            last_logits = step0_logits.index_select(0, last_idx_tensor)
            prev_hidden = step0_hidden.index_select(0, last_idx_tensor)
        prev_tokens = last_logits.argmax(dim=-1).tolist()
        for i in range(num_reqs):
            draft_tokens[slots[i]].append(prev_tokens[i])

        next_pos_list = [sp + n for sp, n in zip(start_pos_list, num_new_tokens_list)]
        running_prior_kv_len = [
            prior_kv_lens_step0[i] + num_new_tokens_list[i] for i in range(num_reqs)
        ]
        # 2026-07-17, Phase 3 round 2: this loop's ``prior_kv_lens`` and
        # ``start_pos_list`` are always numerically identical here (both
        # start equal right after step 0 and both advance by exactly 1
        # every iteration below) -- confirmed by inspection, see
        # ``CapturedMTPDraftStepGraph``'s docstring -- which is what makes
        # a single-length-list captured graph replay valid for this
        # specific loop.
        # 2026-07-19, chunked-prefill round: the actual k-1 loop body is now
        # ``_mtp_run_continuation_steps`` (pure code motion, see its
        # docstring) -- shared with ``mtp_prefill_batch``'s new chunked
        # path, which computes step 0 itself (chunk by chunk) and needs
        # this exact same, already-verified tail afterward.
        self._mtp_run_continuation_steps(
            slots, draft_tokens, prev_tokens, prev_hidden, next_pos_list, running_prior_kv_len, k
        )
        return draft_tokens

    def mtp_prefill_batch(
        self,
        slots: list[int],
        prompts_per_slot: list[list[int]],
        chunk_size: int | None = None,
    ) -> dict[int, dict]:
        """Batched analogue of ``mtp_prefill``: ONE real target prefill
        forward (``_forward_batch``, now able to accept never-forwarded
        slots -- see its 2026-07-17 GDN-init-guard relaxation) covering
        every listed slot, followed by ONE batched draft-sync+propose
        funnel (``_mtp_sync_and_propose_batch``).

        **2026-07-19, continuous-batching round (ragged-length prefill):**
        prompts no longer need the SAME length. Through this round, this
        method hard-asserted every slot's prompt was exactly the same
        length (true for the W1-S/W2-S frozen fixtures, but not for real
        async serving, where different requests genuinely arrive with
        different prompt lengths). This is now generalized -- see the
        ``elif is_uniform_len`` / ``else`` split in the body below -- by
        reusing the SAME per-slot-ragged ``qo_len``/``num_new_tokens`` LIST
        mechanism ``_forward_batch``/``_mtp_sync_and_propose_batch`` already
        built and verified for the 2026-07-17 recompute-fallback batching
        round (see their own docstrings: ``build_attention_metadata_batch``/
        ``build_gdn_metadata_batch`` already construct correct CSR/
        cu_seqlens-style metadata for a ragged per-request qo_len list; the
        general/chunked attention kernel this routes to is already
        documented, in ``vllm/v1/attention/backends/sm120_gqa.py``, as
        correct for "arbitrary mixed prefill+decode batches" -- ragged
        MULTI-REQUEST prefill lengths are exactly that kernel's designed use
        case, not new territory). No new kernel/metadata mechanism was
        needed for this -- confirmed by direct reading before writing this
        branch, the same discipline the chunking round below already
        established. Every EXISTING (uniform-length) caller takes the
        EXACT SAME code path as before, byte-for-byte (the ``elif
        is_uniform_len`` branch is the untouched pre-2026-07-19 code) --
        only a genuinely ragged batch exercises the new ``else`` branch.

        ``chunk_size`` is NOT YET generalized to ragged batches: the
        chunked loop below advances every slot's chunk boundary in
        lockstep from a single shared ``running_kv_len``/
        ``running_draft_len`` counter, which assumes uniform length by
        construction. Combining ragged lengths with ``chunk_size`` raises
        ``NotImplementedError`` with an explicit message rather than
        silently mis-chunking -- a precisely scoped, real follow-on (see
        ``notes/2026-07-18-session-review-and-next-steps.md`` section 21),
        not something this round needed: real async admission prompts are
        far short of the 8K+ context where chunking's memory benefit
        matters, so this is a deliberate scope boundary, not a gap in the
        common case.

        **2026-07-18, D1-followup fix**: both the target model's prefill
        forward and the draft model's step-0 sync forward now pass
        ``logits_last_position_only=True`` -- this method only ever reads
        each slot's OWN last-position logits (the anchor / first draft
        token), never any other position's, so projecting every position
        through the vocab head was 100% wasted work. Found by direct
        instrumentation (``benchmarks/mtp_prefill_batch_memory_diag.py``)
        of the exact c=4/16K-context shape
        ``notes/2026-07-18-session-review-and-next-steps.md`` section 12
        flagged as 4.85x slower than native and peaking at 99.2% of GPU
        memory: at that shape (``qo_len=16384``, ``concurrency=4``, vocab
        248320) EACH of the two ``compute_logits`` calls this method used
        to make allocated an unused ``[65536, 248320]`` bf16 tensor (31040
        MiB, ~30.3 GiB) of which only 4 rows were ever read -- ~60 GiB of
        pure waste, and the second such allocation (already competing with
        the first plus the model+KV-cache baseline for the remaining
        headroom) took 15.2s by itself, direct evidence of near-OOM
        allocator pressure compounding the waste. See that section's
        follow-up entry for the full before/after numbers.

        ``chunk_size`` (2026-07-19, chunked-prefill round, default ``None``
        preserving every existing call site byte-for-byte -- see
        ``notes/2026-07-18-session-review-and-next-steps.md`` section 19
        for the full design/verification writeup): when given, and the
        prompt is longer than ``chunk_size``, splits the single giant
        target-model-forward-then-draft-model-step-0-forward this method
        otherwise does into multiple sequential ``chunk_size``-token
        pieces, so no one ``model.forward()`` call ever processes more
        than ``chunk_size * len(slots)`` tokens at once -- the fix for the
        16K/32K-context near-OOM activation-memory scaling section 12-18
        already root-caused and quantified (peak transient working set
        scales with ``qo_len * concurrency``; chunking bounds ``qo_len``
        to ``chunk_size`` regardless of total prompt length). Matches
        native vLLM's own ``--max-num-batched-tokens=8192`` chunked-prefill
        convention by default (``_DEFAULT_PREFILL_CHUNK_SIZE`` below).

        Both attention's paged KV cache and GDN's recurrent
        ``conv_state``/``ssm_state`` carry over correctly across chunks of
        the SAME slot's own prefill with **no new mechanism** -- both were
        already fully general, just never exercised this way before
        (confirmed by direct reading, not assumed, before writing this):
        ``_forward_batch``'s existing ``kv_lengths``/``commit=True``
        parameters already make each chunk's attention forward a genuine
        paged-KV continuation of the previous chunk (the SM120 general/
        FP8-KV kernel's own module docstring, ``vllm/v1/attention/backends
        /sm120_gqa.py``, already documents itself as correct for "pure
        prefill, chunked-prefill continuation, and arbitrary mixed
        prefill+decode batches" -- chunking here is new USAGE of that
        kernel path, not new kernel work); and GDN's ``has_initial_state``
        (built from ``self._r.slot_gdn_initialized`` inside
        ``build_gdn_metadata_batch``) is already False only for a
        genuinely fresh slot and unconditionally set True at the end of
        EVERY ``_forward_batch`` call regardless of ``qo_len`` (see that
        method's last few lines) -- so chunk 1 of a fresh slot's prefill
        correctly gets ``has_initial_state=False`` (matching today's
        single-shot behavior) and chunk 2 onward correctly gets
        ``has_initial_state=True``, reading back exactly the
        conv/ssm-state row chunk 1's own forward pass wrote, with zero new
        code in either metadata builder. This is the SAME per-physical-slot
        flag every decode/verify round already relies on for
        cross-ROUND state continuity -- generalized here, for the first
        time, to WITHIN-one-prefill continuity.

        The draft model's own step-0 sync is chunked in lockstep with the
        target model (same chunk boundaries, each chunk's target hidden
        states fed into that SAME chunk's draft forward) -- the other
        piece ``notes/2026-07-18-session-review-and-next-steps.md`` section
        18.6 identified as genuinely unbuilt. Only the LAST chunk's
        anchor/step-0 draft token are ever read; the K-1 further
        autoregressive draft continuation steps after step 0 are
        unaffected (still one uniform, small ``qo_len=1`` loop) and reuse
        ``_mtp_run_continuation_steps`` -- the exact same, already-verified
        tail ``_mtp_sync_and_propose_batch`` itself uses."""
        if self._r.mtp_model is None or self._r.num_speculative_tokens is None:
            raise RuntimeError("no MTP draft model loaded")
        num_reqs = len(slots)
        if len(prompts_per_slot) != num_reqs:
            raise ValueError("slots and prompts_per_slot must have equal length")
        if num_reqs == 0:
            return {}
        prompt_lens = [len(p) for p in prompts_per_slot]
        is_uniform_len = len(set(prompt_lens)) <= 1
        # 2026-07-19: chunking's loop below advances every slot's chunk
        # boundary from a SINGLE shared running counter -- a genuine
        # uniform-length assumption, not yet generalized (see this method's
        # docstring). Ragged batches that don't actually need chunking
        # (every prompt already fits in one chunk) are unaffected.
        needs_chunking = chunk_size is not None and max(prompt_lens) > chunk_size
        if needs_chunking and not is_uniform_len:
            raise NotImplementedError(
                "mtp_prefill_batch: chunk_size is not yet supported together with "
                "ragged (per-slot different-length) prompts -- either omit "
                "chunk_size (fine for real async-admission prompt lengths, far "
                "short of where chunking's memory benefit matters) or prefill "
                "this slot alone with chunk_size set. See "
                "notes/2026-07-18-session-review-and-next-steps.md section 21."
            )
        for s in slots:
            if self._r.slot_kv_len[s] != 0 or self._r.slot_draft_sync_len[s] != 0:
                raise RuntimeError(f"slot {s} is not fresh")
            # Phase 2 (2026-07-18), defense in depth: bootstrap value for
            # the spec-decode GDN mechanism's first-ever verify round on
            # this slot -- already 1 via __init__/reset_slot for any slot
            # that actually went through one of those, set explicitly here
            # too so this invariant holds regardless of how the slot got
            # to "fresh".
            self._r.slot_num_accepted_tokens[s] = 1

        if not needs_chunking and is_uniform_len:
            # Unchanged from before 2026-07-19: one giant forward each for
            # the target model and the draft model's step-0 sync. This
            # branch is byte-for-byte the prior implementation -- every
            # existing caller that never passes ``chunk_size`` (or whose
            # prompt already fits in one chunk) takes this exact path.
            prompt_len = prompt_lens[0]
            target_logits, target_hidden = self._r._forward_batch(
                slots,
                prompts_per_slot if prompt_len > 1 else [p[0] for p in prompts_per_slot],
                [0] * num_reqs,
                qo_len=prompt_len,
                commit=True,
                return_hidden=True,
                is_decode=False,
                logits_last_position_only=True,
            )
            anchors: dict[int, int] = {}
            shifted_per_slot = []
            for i, s in enumerate(slots):
                # target_logits is [num_reqs, vocab] (last-position-only, see
                # logits_last_position_only above) -- row i IS slot i's last
                # (and only returned) position, no further offset needed.
                row = target_logits[i]
                anchor = int(row.argmax(dim=-1).item())
                anchors[s] = anchor
                shifted_per_slot.append(prompts_per_slot[i][1:] + [anchor])

            draft_tokens_by_slot = self._mtp_sync_and_propose_batch(
                slots,
                shifted_per_slot,
                target_hidden,
                [0] * num_reqs,
                num_new_tokens=prompt_len,
                k=self._r.num_speculative_tokens,
                step0_logits_last_position_only=True,
            )
            for s in slots:
                self._r.slot_pending_draft_tokens[s] = draft_tokens_by_slot[s]
            # P3 populate-on-completion (attention half): publish each slot's
            # full committed blocks. GDN completion checkpoint deferred to the
            # two-phase cold path (see mtp_prefill's identical note) -- a
            # single-shot prefill's live GDN state is at prompt_len, not G.
            if self._r.enable_persistent_prefix_cache:
                for i, s in enumerate(slots):
                    self._r._publish_committed_blocks(s, prompts_per_slot[i], prompt_lens[i])
            return {
                s: {"anchor": anchors[s], "draft_tokens": draft_tokens_by_slot[s]} for s in slots
            }

        if not needs_chunking:
            # NEW (2026-07-19, continuous-batching round): genuinely ragged
            # per-slot prompt lengths, single-shot (no chunking needed).
            # Mirrors the uniform branch above exactly, except ``qo_len``/
            # ``num_new_tokens`` are passed as a per-slot LIST instead of a
            # shared scalar -- both ``_forward_batch`` and
            # ``_mtp_sync_and_propose_batch`` already generalize to this
            # (built 2026-07-17 for the recompute-fallback batching round;
            # see this method's docstring), so no new mechanism is added
            # here, only new USAGE of an already-verified one. Always passes
            # ``prompts_per_slot``/``shifted_per_slot`` as nested per-slot
            # lists (never the uniform branch's flattened-scalar-1 special
            # case) -- correct for every prompt_len, including a degenerate
            # length-1 slot, since a per-slot qo_len list of all-1s is
            # already treated identically to the scalar case by both
            # ``build_attention_metadata_batch`` and ``build_gdn_metadata_
            # batch`` (value-based, not type-based -- see their docstrings).
            target_logits, target_hidden = self._r._forward_batch(
                slots,
                prompts_per_slot,
                [0] * num_reqs,
                qo_len=prompt_lens,
                commit=True,
                return_hidden=True,
                is_decode=False,
                logits_last_position_only=True,
            )
            anchors = {}
            shifted_per_slot = []
            for i, s in enumerate(slots):
                row = target_logits[i]
                anchor = int(row.argmax(dim=-1).item())
                anchors[s] = anchor
                shifted_per_slot.append(prompts_per_slot[i][1:] + [anchor])

            draft_tokens_by_slot = self._mtp_sync_and_propose_batch(
                slots,
                shifted_per_slot,
                target_hidden,
                [0] * num_reqs,
                num_new_tokens=prompt_lens,
                k=self._r.num_speculative_tokens,
                step0_logits_last_position_only=True,
            )
            for s in slots:
                self._r.slot_pending_draft_tokens[s] = draft_tokens_by_slot[s]
            # P3 populate-on-completion (attention half): publish each slot's
            # full committed blocks. GDN completion checkpoint deferred to the
            # two-phase cold path (see mtp_prefill's identical note) -- a
            # single-shot prefill's live GDN state is at prompt_len, not G.
            if self._r.enable_persistent_prefix_cache:
                for i, s in enumerate(slots):
                    self._r._publish_committed_blocks(s, prompts_per_slot[i], prompt_lens[i])
            return {
                s: {"anchor": anchors[s], "draft_tokens": draft_tokens_by_slot[s]} for s in slots
            }

        # Chunked path (2026-07-19). Processes the prompt in
        # ``ceil(prompt_len / chunk_size)`` sequential pieces. Each chunk's
        # target-model forward is a genuine paged-KV-cache continuation of
        # the previous chunk (growing ``kv_lengths``, ``commit=True``) and
        # each chunk's draft-model forward is fed that SAME chunk's target
        # hidden states, mirroring the whole-prompt case's own
        # target_hidden -> draft-model wiring one chunk at a time.
        # Reaching here requires is_uniform_len (checked above), so
        # slots[0]'s own prompt length speaks for the whole batch.
        prompt_len = prompt_lens[0]
        anchors = {}
        step0_logits: torch.Tensor | None = None
        step0_hidden: torch.Tensor | None = None
        chunk_start = 0
        while chunk_start < prompt_len:
            chunk_end = min(chunk_start + chunk_size, prompt_len)
            this_chunk_len = chunk_end - chunk_start
            is_last_chunk = chunk_end == prompt_len
            chunk_tokens_per_slot = [p[chunk_start:chunk_end] for p in prompts_per_slot]

            # Uniform prompt length -> every slot's kv_len/draft_sync_len
            # advances identically chunk to chunk, so reading slots[0]'s
            # own counters is exactly this chunk's shared running value
            # for every slot (already asserted equal-length above).
            running_kv_len = self._r.slot_kv_len[slots[0]]
            target_logits_chunk, target_hidden_chunk = self._r._forward_batch(
                slots,
                chunk_tokens_per_slot
                if this_chunk_len > 1
                else [p[0] for p in chunk_tokens_per_slot],
                [running_kv_len] * num_reqs,
                qo_len=this_chunk_len,
                commit=True,
                return_hidden=True,
                is_decode=False,
                logits_last_position_only=True,
            )

            if is_last_chunk:
                for i, s in enumerate(slots):
                    anchors[s] = int(target_logits_chunk[i].argmax(dim=-1).item())
                shifted_chunk_per_slot = [
                    prompts_per_slot[i][chunk_start + 1 : prompt_len] + [anchors[slots[i]]]
                    for i in range(num_reqs)
                ]
            else:
                # Not yet at the anchor position -- this chunk's shifted
                # (draft-model-input) tokens are simply the next real
                # prompt tokens, no anchor needed yet.
                shifted_chunk_per_slot = [
                    prompts_per_slot[i][chunk_start + 1 : chunk_end + 1] for i in range(num_reqs)
                ]

            running_draft_len = self._r.slot_draft_sync_len[slots[0]]
            draft_logits_chunk, draft_hidden_chunk = self._mtp_forward_batch(
                slots,
                # ``_mtp_forward_batch`` special-cases the literal scalar
                # ``qo_len == 1`` to expect a FLAT one-token-per-slot list
                # (matching ``_forward_batch``'s own convention, and the
                # equivalent flattening already applied to the target
                # model's call above) -- only matters for a final remainder
                # chunk of length 1 with a non-default chunk_size; every
                # real fixture/chunk-size combination this round uses
                # divides evenly, so this is a defensive correctness
                # guard, not something exercised by today's measurements.
                shifted_chunk_per_slot
                if this_chunk_len > 1
                else [t[0] for t in shifted_chunk_per_slot],
                target_hidden_chunk,
                [running_draft_len] * num_reqs,
                [running_draft_len] * num_reqs,
                qo_len=this_chunk_len,
                is_decode=False,
                # Every chunk (not just the last) only needs each slot's
                # own last-position output kept -- earlier chunks' full
                # per-position hidden/logits are never read (the physical
                # KV-cache write that lets the NEXT chunk continue is a
                # side effect of the forward call itself, not something
                # this method needs the returned tensor for).
                logits_last_position_only=True,
            )
            for s in slots:
                self._r.slot_draft_sync_len[s] += this_chunk_len

            if is_last_chunk:
                step0_logits, step0_hidden = draft_logits_chunk, draft_hidden_chunk

            # P3.2 chunk-boundary GDN checkpoints (Fork-2 coarse, step 5): at
            # each NON-FINAL block-aligned chunk_end (every chunk_size boundary,
            # default 8192), the target GDN forward has just ended at chunk_end,
            # so its live state IS the state at chunk_end -- a FREE checkpoint
            # point (no extra forward) giving 8192-granular cross-request partial
            # sharing. Publish each slot's [.., chunk_end) attention blocks, then
            # materialize the persistent GDN checkpoint keyed by the chunk_end
            # tail block (same chained hash => INV3 agreement). The completion-
            # boundary checkpoint (two-phase cold path / fan-out leader) remains
            # separate. The chunked path is uniform-length (asserted above), so
            # every slot shares these boundaries; each slot checkpoints its OWN
            # physical GDN state.
            if (
                self._r.enable_persistent_prefix_cache
                and not is_last_chunk
                and chunk_end % self._r.block_size == 0
            ):
                num_chunk_blocks = chunk_end // self._r.block_size
                for i, s in enumerate(slots):
                    self._r._publish_committed_blocks(s, prompts_per_slot[i], chunk_end)
                    self._r.materialize_gdn_checkpoint(
                        s,
                        key=self._r.block_table[s][num_chunk_blocks - 1],
                        hash_value=self._r.slot_block_hashes[s][num_chunk_blocks - 1].value,
                        num_tokens=chunk_end,
                    )

            chunk_start = chunk_end

        assert step0_logits is not None and step0_hidden is not None
        prev_tokens = step0_logits.argmax(dim=-1).tolist()
        draft_tokens: dict[int, list[int]] = {s: [prev_tokens[i]] for i, s in enumerate(slots)}
        # Matches _mtp_sync_and_propose_batch's own invariant: both counters
        # equal ``prompt_len`` here (0 + prompt_len), identical to each
        # other, exactly what _mtp_run_continuation_steps expects at entry.
        next_pos_list = [self._r.slot_draft_sync_len[s] for s in slots]
        running_prior_kv_len = [self._r.slot_draft_sync_len[s] for s in slots]
        self._mtp_run_continuation_steps(
            slots,
            draft_tokens,
            prev_tokens,
            step0_hidden,
            next_pos_list,
            running_prior_kv_len,
            self._r.num_speculative_tokens,
        )
        for s in slots:
            self._r.slot_pending_draft_tokens[s] = draft_tokens[s]
        # P3 populate-on-completion (attention half) for the chunked path. The
        # GDN completion checkpoint at G is materialized only by a forward that
        # ends at G (two-phase cold path); chunk boundaries (P3.2) add the rest.
        if self._r.enable_persistent_prefix_cache:
            for i, s in enumerate(slots):
                self._r._publish_committed_blocks(s, prompts_per_slot[i], prompt_lens[i])
        return {s: {"anchor": anchors[s], "draft_tokens": draft_tokens[s]} for s in slots}

    # =====================================================================
    # A5/B4: Incremental chunked prefill API (2026-07-22)
    # Allows the engine to interleave prefill chunks with decode rounds,
    # preventing long prefills from starving active decode slots.
    # =====================================================================

    def mtp_prefill_fanout_batch(
        self,
        slots: list[int],
        prompts_per_slot: list[list[int]],
        min_shared_prefix_tokens: int | None = None,
    ) -> dict[int, dict]:
        """P2 fan-out fork -- Pattern A same-round prefix sharing
        (``notes/prefix-cache-design.md`` sec 5, "P2 -- Fan-out fork (Pattern
        A, same-round sharing; self-contained)", and sec 3.5/3.6).

        When ``enable_prefix_cache`` is on AND >=2 same-round requests share a
        token prefix of at least one full block, the shared prefix is computed
        ONCE and referenced by all siblings instead of being recomputed N
        times:

        1. Detect the common token prefix among the same-round admit batch by
           direct comparison (``_common_prefix_len``, cheap for <=4 requests).
        2. Prefill the group LEADER (``slots[0]``) over ``[0, Lc)`` -- where
           ``Lc`` is the block-aligned common-prefix length, capped at
           ``min(prompt_len) - 1`` so every request keeps >=1 suffix token to
           recompute for its own logits (sec 3.8) -- forcing a GDN checkpoint
           boundary there, then ``snapshot_gdn_state`` at ``Lc``.
        3. Continue-prefill the leader's own suffix ``[Lc, leader_len)`` and
           draft-sync the leader over its whole prompt (this writes the draft
           layer's KV for ``[0, Lc)`` into the same shared blocks -- the draft
           layer is in the attention group, sec 3.1).
        4. For each SIBLING: reference the leader's ``[0, Lc)`` attention
           blocks (``BlockPool.reference``, ``ref_cnt += 1``, all 17 attention
           layers share the one block-id namespace), ``restore_gdn_state`` the
           leader's snapshot (``allow_cross_slot=True``), set
           ``slot_draft_sync_len = Lc``, and continue-prefill ONLY the
           sibling's suffix ``[Lc, sibling_len)`` through the already-validated
           chunked-prefill continuation machinery (``_forward_batch`` with
           ``kv_lengths=[Lc]``/``commit``/``is_decode=False`` + the
           ``_mtp_sync_and_propose_batch`` draft funnel).

        No persistent hash index, no eviction -- the shared entry lives only
        for this one admission round (P3 builds the persistent cache). Reuses
        ONLY the P1 block-table/ref-counting substrate + the existing GDN
        snapshot primitive + the chunked-continuation path; nothing here is a
        parallel copy of those.

        **Rollback-safe / byte-identical-to-P1 gate**: with ``enable_prefix_
        cache`` off, OR fewer than 2 requests, OR a block-aligned common
        prefix shorter than ``min_shared_prefix_tokens`` (default
        ``block_size`` -- one full shareable block), this falls back to the
        exact ``mtp_prefill_batch`` path, byte-for-byte P1 behavior.

        Returns the same ``{slot: {"anchor": int, "draft_tokens": list[int]}}``
        shape ``mtp_prefill_batch`` returns, for the leader AND every sibling.
        """
        if self._r.mtp_model is None or self._r.num_speculative_tokens is None:
            raise RuntimeError("no MTP draft model loaded")
        num_reqs = len(slots)
        if len(prompts_per_slot) != num_reqs:
            raise ValueError("slots and prompts_per_slot must have equal length")
        if num_reqs == 0:
            return {}

        # Fork gate (rollback-safe boundary -- see docstring). Anything that
        # does not clear it takes the exact P1 path, byte-for-byte.
        if not self._r.enable_prefix_cache or num_reqs < 2:
            return self.mtp_prefill_batch(slots, prompts_per_slot)
        threshold = (
            self._r.block_size if min_shared_prefix_tokens is None else min_shared_prefix_tokens
        )
        common = self._r._common_prefix_len(prompts_per_slot)
        min_prompt_len = min(len(p) for p in prompts_per_slot)
        lc = (min(common, min_prompt_len - 1) // self._r.block_size) * self._r.block_size
        if lc < self._r.block_size or lc < threshold:
            return self.mtp_prefill_batch(slots, prompts_per_slot)

        # Defense in depth (R1): every prompt must really share [0, Lc) --
        # true by construction (Lc <= common), asserted cheaply for <=4 reqs.
        for p in prompts_per_slot:
            if p[:lc] != prompts_per_slot[0][:lc]:
                raise RuntimeError("fan-out fork: a request does not share the detected prefix")

        leader = slots[0]
        siblings = slots[1:]
        leader_prompt = prompts_per_slot[0]
        leader_len = len(leader_prompt)
        num_prefix_blocks = lc // self._r.block_size

        # Same fresh-slot contract as mtp_prefill_batch (every slot starts at
        # kv_len 0 / draft_sync_len 0).
        for s in slots:
            if self._r.slot_kv_len[s] != 0 or self._r.slot_draft_sync_len[s] != 0:
                raise RuntimeError(f"slot {s} is not fresh")
            self._r.slot_num_accepted_tokens[s] = 1

        # --- Leader phase 1: prefill the shared prefix [0, Lc), checkpoint
        # the GDN state there (the fork point). ---
        _, leader_hidden_prefix = self._r._forward_batch(
            [leader],
            [leader_prompt[:lc]],
            [0],
            qo_len=lc,
            commit=True,
            return_hidden=True,
            is_decode=False,
            logits_last_position_only=True,
        )
        leader_snapshot = self._r.snapshot_gdn_state(leader)
        shared_blocks = list(self._r.block_table[leader][:num_prefix_blocks])
        # P3 populate (cross-cutting decision 1): the fan-out leader's phase-1
        # forward ENDS at Lc, so its live GDN state IS the state at Lc -- a
        # correct completion checkpoint for the shared prefix. Publish the
        # leader's [0, Lc) attention blocks + materialize the persistent GDN
        # checkpoint at Lc so a FUTURE round (or another request sharing this
        # prefix) can hit it. Purely additive under the flag (P2 unchanged off).
        if self._r.enable_persistent_prefix_cache and num_prefix_blocks > 0:
            self._r._publish_committed_blocks(leader, leader_prompt, lc)
            self._r.materialize_gdn_checkpoint(
                leader,
                key=self._r.block_table[leader][num_prefix_blocks - 1],
                hash_value=self._r.slot_block_hashes[leader][num_prefix_blocks - 1].value,
                num_tokens=lc,
            )

        # --- Leader phase 2: continue-prefill the leader's own suffix
        # [Lc, leader_len) (validated chunked-prefill continuation). Lc is
        # capped at min_prompt_len - 1 <= leader_len - 1, so a non-empty
        # suffix always exists here. ---
        leader_logits_suffix, leader_hidden_suffix = self._r._forward_batch(
            [leader],
            [leader_prompt[lc:]],
            [lc],
            qo_len=leader_len - lc,
            commit=True,
            return_hidden=True,
            is_decode=False,
            logits_last_position_only=True,
        )
        leader_anchor = int(leader_logits_suffix[0].argmax(dim=-1).item())
        leader_hidden = torch.cat([leader_hidden_prefix, leader_hidden_suffix], dim=0)

        # Leader draft sync over the WHOLE prompt (step-0 resync + K-1
        # continuation steps) -- exactly mtp_prefill_batch's uniform-path
        # draft funnel. Writes the draft layer's KV for [0, leader_len) into
        # the leader's blocks, INCLUDING the shared [0, Lc) blocks the
        # siblings reference next (draft layer is in the attention group).
        leader_drafts = self._mtp_sync_and_propose_batch(
            [leader],
            [leader_prompt[1:] + [leader_anchor]],
            leader_hidden,
            [0],
            num_new_tokens=leader_len,
            k=self._r.num_speculative_tokens,
            step0_logits_last_position_only=True,
        )
        self._r.slot_pending_draft_tokens[leader] = leader_drafts[leader]
        # P3 populate: publish the leader's remaining full blocks [Lc, leader_len)
        # (the cursor advances from num_prefix_blocks); attention only -- the
        # leader's live GDN state is now at leader_len, not a block boundary.
        if self._r.enable_persistent_prefix_cache:
            self._r._publish_committed_blocks(leader, leader_prompt, leader_len)
        result: dict[int, dict] = {
            leader: {"anchor": leader_anchor, "draft_tokens": leader_drafts[leader]}
        }

        # --- Siblings: reference the leader's [0, Lc) attention blocks +
        # restore the leader's GDN snapshot, then continue-prefill each
        # sibling's own suffix [Lc, sibling_len). ---
        sibling_prompts = prompts_per_slot[1:]
        suffix_per_slot = [p[lc:] for p in sibling_prompts]
        suffix_lens = [len(sfx) for sfx in suffix_per_slot]
        for s in siblings:
            # Reference (ref_cnt += 1) the leader's [0, Lc) blocks -- all 17
            # attention layers share the one block-id namespace (sec 3.1), so
            # one reference call covers target + draft KV for the prefix.
            self._r.block_table[s] = list(shared_blocks)
            self._r.block_pool.reference(shared_blocks)
            # Restore the leader's GDN state at Lc into this sibling (sec 3.5
            # step 2); cross-slot by design (source = leader, dest = sibling).
            self._r.restore_gdn_state(s, leader_snapshot, allow_cross_slot=True)
            # Bookkeeping reproduces exactly the state computing [0, Lc) fresh
            # would have produced (sec 3.5 steps 3-4).
            self._r.slot_kv_len[s] = lc
            self._r.slot_gdn_initialized[s] = True
            self._r.slot_draft_sync_len[s] = lc
            self._r.slot_num_accepted_tokens[s] = 1

        # P3.2 decode-position populate: seed each sibling's committed-token
        # record (its full prompt) so a later verify-commit can hash decode-
        # produced blocks correctly (they may straddle the prompt tail + decode
        # head). The shared [0, Lc) blocks are the leader's, already published
        # by the leader under the flag; the incremental publish a decode round
        # triggers is idempotent for them (same chained hash) and fresh for the
        # suffix. Purely additive bookkeeping under the flag (the fan-out test
        # runs with the persistent flag off => complete no-op there).
        if self._r.enable_persistent_prefix_cache:
            for j, s in enumerate(siblings):
                self._r.slot_committed_tokens[s] = list(sibling_prompts[j])

        # Batched target continue-prefill over the (ragged) suffixes: the
        # validated chunked-prefill continuation (kv_lengths=[Lc], commit,
        # is_decode=False; GDN has_initial_state=True since every sibling's
        # slot_gdn_initialized is now True). Fresh suffix KV writes go to
        # freshly-allocated PRIVATE blocks appended to each sibling's table.
        sibling_logits, sibling_hidden = self._r._forward_batch(
            siblings,
            suffix_per_slot,
            [lc] * len(siblings),
            qo_len=suffix_lens,
            commit=True,
            return_hidden=True,
            is_decode=False,
            logits_last_position_only=True,
        )
        anchors = {s: int(sibling_logits[i].argmax(dim=-1).item()) for i, s in enumerate(siblings)}
        shifted_suffix_per_slot = [
            sibling_prompts[i][lc + 1 :] + [anchors[siblings[i]]] for i in range(len(siblings))
        ]

        # Batched draft step-0 sync over the suffixes + K-1 continuation
        # steps. prior_kv_lens_step0 = slot_draft_sync_len = Lc for each
        # sibling; the draft attends over the referenced [0, Lc) blocks and
        # writes its own suffix KV into fresh private blocks.
        sibling_drafts = self._mtp_sync_and_propose_batch(
            siblings,
            shifted_suffix_per_slot,
            sibling_hidden,
            [lc] * len(siblings),
            num_new_tokens=suffix_lens,
            k=self._r.num_speculative_tokens,
            step0_logits_last_position_only=True,
        )
        for s in siblings:
            self._r.slot_pending_draft_tokens[s] = sibling_drafts[s]
            result[s] = {"anchor": anchors[s], "draft_tokens": sibling_drafts[s]}
        return result

    # ------------------------------------------------------------------
    # P3.1 -- Persistent content-addressed prefix cache
    # (notes/2026-07-19-p3-implementation-plan.md, "P3.1 -- Persistent-cache
    # hit equivalence"). Write path: populate-on-completion (attention) +
    # completion GDN checkpoint. Read path: reconciliation (L = G <= A) +
    # restore-and-continue. All behind enable_persistent_prefix_cache
    # (default False => byte-for-byte P2; L=0 => P2 fan-out/cold).
    # ------------------------------------------------------------------

    def mtp_prefill_warm_continue(self, slot: int, prompt: list[int], prior_len: int) -> dict:
        # P4b session affinity -- zero-restore continuation of a WARM slot. The
        # slot already holds [0, prior_len) KV + GDN LIVE (turn-1's committed
        # content; it was retained, never reset_slot-ed), so prefill ONLY the
        # suffix [prior_len, prompt_len). Mirrors _prefill_hit_with_cache MINUS
        # the restore_cached_prefix call -- there is nothing to restore, because
        # the boundary state IS turn-1's live state (even more faithful than a
        # restore). Gated: only the server's session-affinity admission path ever
        # calls this; with the flag off it is never reached, so the frozen
        # P0-P3 + P4a paths stay byte-for-byte untouched. Reuses ONLY the private
        # primitives the hit path already uses (_forward_batch,
        # _mtp_sync_and_propose_batch, _publish_committed_blocks).
        if not self._r.enable_persistent_prefix_cache:
            raise RuntimeError("mtp_prefill_warm_continue requires the persistent prefix cache")
        if self._r.mtp_model is None or self._r.num_speculative_tokens is None:
            raise RuntimeError("mtp_prefill_warm_continue: no MTP draft model loaded")
        if self._r.slot_kv_len[slot] != prior_len or not self._r.slot_gdn_initialized[slot]:
            raise RuntimeError(
                f"mtp_prefill_warm_continue: slot {slot} is not warm at prior_len={prior_len} "
                f"(kv_len={self._r.slot_kv_len[slot]}, "
                f"gdn_init={self._r.slot_gdn_initialized[slot]})"
            )
        # Authoritative prefix match: the slot's committed-token record (P1+C1)
        # must equal the new prompt through prior_len. Any mismatch => the caller
        # must fall back to the cold/restore path (the server catches this raise).
        if self._r.slot_committed_tokens[slot][:prior_len] != prompt[:prior_len]:
            raise RuntimeError(
                "mtp_prefill_warm_continue: prefix mismatch -- caller must fall back"
            )
        # Reset draft state to the committed boundary; discard turn-1's draft-ahead.
        # _mtp_sync_and_propose_batch requires prior_kv_lens_step0 == start_pos_list
        # (its own contract), so slot_draft_sync_len MUST be prior_len before the
        # call below with start_pos_list=[prior_len]. The committed draft KV
        # [0, prior_len) is valid (the draft layer is the 17th attention-group
        # member sharing block_table[slot]); the speculative [prior_len, old) is
        # overwritten by the suffix step-0 sync + K-1 continuation steps, and any
        # stale KV beyond the new draft_sync_len is never read. slot_num_accepted_
        # tokens=1 bootstraps the spec-decode GDN mechanism for the next verify
        # round, exactly as restore_cached_prefix and the fanout sibling do.
        self._r.slot_draft_sync_len[slot] = prior_len
        self._r.slot_num_accepted_tokens[slot] = 1
        self._r.slot_pending_draft_tokens[slot] = None
        prompt_len = len(prompt)
        suffix_len = prompt_len - prior_len
        k = self._r.num_speculative_tokens
        suffix_tokens = prompt[prior_len:]
        suffix_logits, suffix_hidden = self._r._forward_batch(
            [slot],
            [suffix_tokens] if suffix_len > 1 else [suffix_tokens[0]],
            [prior_len],
            qo_len=suffix_len,
            commit=True,
            return_hidden=True,
            is_decode=False,
            logits_last_position_only=True,
        )
        anchor = int(suffix_logits[0].argmax(dim=-1).item())
        draft_tokens_by_slot = self._mtp_sync_and_propose_batch(
            [slot],
            [prompt[prior_len + 1 :] + [anchor]],
            suffix_hidden,
            [prior_len],
            num_new_tokens=suffix_len,
            k=k,
            step0_logits_last_position_only=True,
        )
        # Publish the suffix's full committed blocks (attention) so future longer
        # requests can hit deeper. The completion GDN checkpoint at the new boundary
        # is deferred (live GDN is at prompt_len, not a block boundary -- mirrors
        # the hit path); warm sessions continue in place anyway, and the content-
        # hash fallback still hits at turn-1's completion checkpoint.
        self._r._publish_committed_blocks(slot, prompt, prompt_len)
        self._r.slot_pending_draft_tokens[slot] = draft_tokens_by_slot[slot]
        return {"anchor": anchor, "draft_tokens": draft_tokens_by_slot[slot]}

    def mtp_prefill_with_cache(
        self,
        slots: list[int],
        prompts_per_slot: list[list[int]],
        chunk_size: int | None = None,
    ) -> dict[int, dict]:
        # P3.3 -- UNIFIED production prefill entrypoint (test-driven in P3.1;
        # production-wired in P3.3a). ONE batched path composes persistent-hit
        # + P2 same-round fan-out + cold:
        #   * Per slot, reconcile_prefix_hit yields L.
        #   * HIT set (L>0): restore_cached_prefix each (references the [0,L_s)
        #     attention blocks + restores the GDN checkpoint at L_s), then
        #     continue-prefill ALL hit slots' RAGGED suffixes in ONE batched
        #     _forward_batch + ONE _mtp_sync_and_propose_batch -- the proven P2
        #     fan-out sibling ragged-suffix pattern (mtp_prefill_fanout_batch's
        #     sibling block) generalized from a SHARED Lc to PER-SLOT L_s
        #     (ragged kv_lengths=[L_s], qo_len=[suffix_len_s]). A single hit
        #     slot reduces TOKEN-IDENTICALLY to _prefill_hit_with_cache: a
        #     1-element qo_len LIST normalizes to the same flat tokens /
        #     positions / GDN metadata as the scalar qo_len the helper passes
        #     (qo_len==1 takes build_gdn_metadata_batch's decode branch either
        #     way; qo_len>1 resolves slot_initialized=[True] either way).
        #   * COLD set (L==0): >=2 cold slots hand to the EXISTING
        #     mtp_prefill_fanout_batch (P2 same-round fork detection among cold
        #     slots, falling back to mtp_prefill_batch); a single cold slot uses
        #     the two-phase _prefill_cold_with_populate so its COMPLETION GDN
        #     checkpoint is materialized (a future re-request must hit it --
        #     mtp_prefill_batch/fanout publish attention blocks but materialize
        #     no per-slot completion checkpoint, so routing a lone cold slot
        #     there would silently disable hit-after-cold). A persistent hit
        #     always wins over a same-round fork (hits are removed before the
        #     cold hand-off).
        # Returns the same {slot: {"anchor","draft_tokens"}} shape as
        # mtp_prefill_batch. Rollback spine: flag off => delegate to
        # mtp_prefill_batch (byte-for-byte P2); a slot's L=0 => the cold/fanout
        # path => byte-for-byte P2 OUTPUT for it.
        if self._r.mtp_model is None or self._r.num_speculative_tokens is None:
            raise RuntimeError("no MTP draft model loaded")
        if len(slots) != len(prompts_per_slot):
            raise ValueError("slots and prompts_per_slot must have equal length")
        if not self._r.enable_persistent_prefix_cache:
            return self.mtp_prefill_batch(slots, prompts_per_slot, chunk_size)
        if len(slots) == 0:
            return {}

        # Per-slot reconciliation (sec 3.4): L = G <= A.
        L_per_slot = [self._r.reconcile_prefix_hit(p) for p in prompts_per_slot]
        # D2: record prefix cache hit/miss metrics
        for _L in L_per_slot:
            if _L > 0:
                record_prefix_cache_hit(_L // self._r.block_size)
            else:
                record_prefix_cache_miss()
        hit_idx = [i for i, L in enumerate(L_per_slot) if L > 0]
        cold_idx = [i for i, L in enumerate(L_per_slot) if L == 0]

        result: dict[int, dict] = {}

        # --- HIT set: restore each, then batched ragged continue-prefill. ---
        # When chunk_size is given and any hit suffix exceeds it, the hit
        # block switches to a chunked continue-prefill (Phase A INV8 lift,
        # 2026-07-20): uniform-suffix batches use the batched chunked pattern
        # from mtp_prefill_batch's cold chunked path; ragged-suffix batches
        # process each slot independently with chunking. Suffixes that fit
        # in one chunk take the EXISTING monolithic path, byte-for-byte.
        if hit_idx:
            hit_slots = [slots[i] for i in hit_idx]
            hit_prompts = [prompts_per_slot[i] for i in hit_idx]
            hit_L = [L_per_slot[i] for i in hit_idx]
            k = self._r.num_speculative_tokens
            # Same fresh-slot contract as _prefill_hit_with_cache; restore the
            # [0, L_s) attention blocks + GDN checkpoint at L_s (reserve-and-
            # touch before any forward, R4/INV9) and set the bookkeeping to
            # exactly what computing [0, L_s) fresh would have produced.
            for s, p, L in zip(hit_slots, hit_prompts, hit_L):
                if self._r.slot_kv_len[s] != 0 or self._r.slot_draft_sync_len[s] != 0:
                    raise RuntimeError(f"slot {s} is not fresh")
                self._r.restore_cached_prefix(s, p, L)
            # Ragged suffix continue-prefill (generalizes the fan-out sibling
            # block): kv_lengths=[L_s], qo_len=[suffix_len_s]. Fresh suffix KV
            # writes go to freshly-allocated PRIVATE blocks appended to each
            # slot's table; GDN has_initial_state=True (restore initialized it).
            suffix_per_slot = [p[L:] for p, L in zip(hit_prompts, hit_L)]
            suffix_lens = [len(sfx) for sfx in suffix_per_slot]

            use_chunked_hit = chunk_size is not None and max(suffix_lens) > chunk_size

            if not use_chunked_hit:
                # === MONOLITHIC PATH (byte-for-byte unchanged) ===
                suffix_logits, suffix_hidden = self._r._forward_batch(
                    hit_slots,
                    suffix_per_slot,
                    list(hit_L),
                    qo_len=suffix_lens,
                    commit=True,
                    return_hidden=True,
                    is_decode=False,
                    logits_last_position_only=True,
                )
                anchors = {
                    s: int(suffix_logits[i].argmax(dim=-1).item()) for i, s in enumerate(hit_slots)
                }
                shifted_suffix_per_slot = [
                    hit_prompts[i][hit_L[i] + 1 :] + [anchors[hit_slots[i]]]
                    for i in range(len(hit_slots))
                ]
                # Batched draft step-0 sync over the suffixes + K-1 continuation
                # steps. prior_kv_lens_step0 = slot_draft_sync_len = L_s for each
                # slot; the draft attends over the referenced [0, L_s) blocks and
                # writes its own suffix KV into fresh private blocks.
                hit_drafts = self._mtp_sync_and_propose_batch(
                    hit_slots,
                    shifted_suffix_per_slot,
                    suffix_hidden,
                    list(hit_L),
                    num_new_tokens=suffix_lens,
                    k=k,
                    step0_logits_last_position_only=True,
                )
                for i, s in enumerate(hit_slots):
                    # Publish the suffix's full committed blocks (attention) so
                    # future longer requests can hit deeper; the completion GDN
                    # checkpoint at the new boundary is deferred (live GDN state is
                    # at prompt_len, not a block boundary).
                    self._r._publish_committed_blocks(s, hit_prompts[i], len(hit_prompts[i]))
                    self._r.slot_pending_draft_tokens[s] = hit_drafts[s]
                    result[s] = {"anchor": anchors[s], "draft_tokens": hit_drafts[s]}

            elif len(set(suffix_lens)) == 1:
                # === UNIFORM SUFFIX CHUNKED PATH ===
                # All hit slots share the same suffix length (the common
                # benchmark scenario). Follows the EXACT pattern from
                # mtp_prefill_batch's cold chunked path, but with per-slot
                # running_kv_lens (hit_L values may differ per slot even
                # though suffix_len is uniform).
                suffix_len = suffix_lens[0]
                num_hit = len(hit_slots)
                # Bound TOTAL tokens per chunk at ~chunk_size (matching native
                # vLLM's max_num_batched_tokens): each slot gets chunk_size //
                # num_hit tokens per chunk, so the batched forward processes
                # num_hit × (chunk_size // num_hit) ≈ chunk_size tokens total.
                effective_chunk = max(1, chunk_size // num_hit)
                running_kv_lens = list(hit_L)
                running_draft_lens = list(hit_L)
                anchors = {}
                step0_logits = None
                step0_hidden = None
                chunk_start = 0
                while chunk_start < suffix_len:
                    chunk_end = min(chunk_start + effective_chunk, suffix_len)
                    this_chunk_len = chunk_end - chunk_start
                    is_last_chunk = chunk_end == suffix_len
                    chunk_tokens_per_slot = [
                        p[hit_L[i] + chunk_start : hit_L[i] + chunk_end]
                        for i, p in enumerate(hit_prompts)
                    ]

                    target_logits_chunk, target_hidden_chunk = self._r._forward_batch(
                        hit_slots,
                        chunk_tokens_per_slot
                        if this_chunk_len > 1
                        else [t[0] for t in chunk_tokens_per_slot],
                        list(running_kv_lens),
                        qo_len=this_chunk_len,
                        commit=True,
                        return_hidden=True,
                        is_decode=False,
                        logits_last_position_only=True,
                    )
                    for i in range(num_hit):
                        running_kv_lens[i] += this_chunk_len

                    if is_last_chunk:
                        for i, s in enumerate(hit_slots):
                            anchors[s] = int(target_logits_chunk[i].argmax(dim=-1).item())
                        shifted_chunk_per_slot = [
                            hit_prompts[i][hit_L[i] + chunk_start + 1 : hit_L[i] + suffix_len]
                            + [anchors[hit_slots[i]]]
                            for i in range(num_hit)
                        ]
                    else:
                        shifted_chunk_per_slot = [
                            hit_prompts[i][hit_L[i] + chunk_start + 1 : hit_L[i] + chunk_end + 1]
                            for i in range(num_hit)
                        ]

                    draft_logits_chunk, draft_hidden_chunk = self._mtp_forward_batch(
                        hit_slots,
                        shifted_chunk_per_slot
                        if this_chunk_len > 1
                        else [t[0] for t in shifted_chunk_per_slot],
                        target_hidden_chunk,
                        list(running_draft_lens),
                        list(running_draft_lens),
                        qo_len=this_chunk_len,
                        is_decode=False,
                        logits_last_position_only=True,
                    )
                    for i, s in enumerate(hit_slots):
                        self._r.slot_draft_sync_len[s] += this_chunk_len
                        running_draft_lens[i] += this_chunk_len

                    if is_last_chunk:
                        step0_logits, step0_hidden = draft_logits_chunk, draft_hidden_chunk

                    if self._r.enable_persistent_prefix_cache and not is_last_chunk:
                        for i, s in enumerate(hit_slots):
                            abs_chunk_end = hit_L[i] + chunk_end
                            if abs_chunk_end % self._r.block_size == 0:
                                num_chunk_blocks = abs_chunk_end // self._r.block_size
                                self._r._publish_committed_blocks(s, hit_prompts[i], abs_chunk_end)
                                self._r.materialize_gdn_checkpoint(
                                    s,
                                    key=self._r.block_table[s][num_chunk_blocks - 1],
                                    hash_value=self._r.slot_block_hashes[s][
                                        num_chunk_blocks - 1
                                    ].value,
                                    num_tokens=abs_chunk_end,
                                )

                    chunk_start = chunk_end

                assert step0_logits is not None and step0_hidden is not None
                prev_tokens = step0_logits.argmax(dim=-1).tolist()
                hit_drafts: dict[int, list[int]] = {
                    s: [prev_tokens[i]] for i, s in enumerate(hit_slots)
                }
                next_pos_list = [self._r.slot_draft_sync_len[s] for s in hit_slots]
                running_prior_kv_len = [self._r.slot_draft_sync_len[s] for s in hit_slots]
                self._mtp_run_continuation_steps(
                    hit_slots,
                    hit_drafts,
                    prev_tokens,
                    step0_hidden,
                    next_pos_list,
                    running_prior_kv_len,
                    k,
                )
                for i, s in enumerate(hit_slots):
                    self._r._publish_committed_blocks(s, hit_prompts[i], len(hit_prompts[i]))
                    self._r.slot_pending_draft_tokens[s] = hit_drafts[s]
                    result[s] = {"anchor": anchors[s], "draft_tokens": hit_drafts[s]}

            else:
                # === RAGGED SUFFIX PER-SLOT CHUNKED PATH ===
                # Different suffix lengths per slot: process each hit slot
                # independently with chunking. Simpler and correct, though
                # less efficient than batched ragged chunking.
                for idx, s in enumerate(hit_slots):
                    suffix_len = suffix_lens[idx]
                    L = hit_L[idx]
                    prompt = hit_prompts[idx]
                    running_kv_len = L
                    running_draft_len = L
                    chunk_start = 0
                    step0_logits_s = None
                    step0_hidden_s = None
                    anchor_s = None
                    while chunk_start < suffix_len:
                        chunk_end = min(chunk_start + chunk_size, suffix_len)
                        this_chunk_len = chunk_end - chunk_start
                        is_last_chunk = chunk_end == suffix_len
                        chunk_tokens = prompt[L + chunk_start : L + chunk_end]

                        target_logits_chunk, target_hidden_chunk = self._r._forward_batch(
                            [s],
                            [chunk_tokens] if this_chunk_len > 1 else [chunk_tokens[0]],
                            [running_kv_len],
                            qo_len=this_chunk_len,
                            commit=True,
                            return_hidden=True,
                            is_decode=False,
                            logits_last_position_only=True,
                        )
                        running_kv_len += this_chunk_len

                        if is_last_chunk:
                            anchor_s = int(target_logits_chunk[0].argmax(dim=-1).item())
                            shifted_chunk = prompt[L + chunk_start + 1 : L + suffix_len] + [
                                anchor_s
                            ]
                        else:
                            shifted_chunk = prompt[L + chunk_start + 1 : L + chunk_end + 1]

                        draft_logits_chunk, draft_hidden_chunk = self._mtp_forward_batch(
                            [s],
                            [shifted_chunk] if this_chunk_len > 1 else [shifted_chunk[0]],
                            target_hidden_chunk,
                            [running_draft_len],
                            [running_draft_len],
                            qo_len=this_chunk_len,
                            is_decode=False,
                            logits_last_position_only=True,
                        )
                        self._r.slot_draft_sync_len[s] += this_chunk_len
                        running_draft_len += this_chunk_len

                        if is_last_chunk:
                            step0_logits_s, step0_hidden_s = draft_logits_chunk, draft_hidden_chunk

                        if (
                            self._r.enable_persistent_prefix_cache
                            and not is_last_chunk
                            and (L + chunk_end) % self._r.block_size == 0
                        ):
                            abs_end = L + chunk_end
                            num_blocks = abs_end // self._r.block_size
                            self._r._publish_committed_blocks(s, prompt, abs_end)
                            self._r.materialize_gdn_checkpoint(
                                s,
                                key=self._r.block_table[s][num_blocks - 1],
                                hash_value=self._r.slot_block_hashes[s][num_blocks - 1].value,
                                num_tokens=abs_end,
                            )

                        chunk_start = chunk_end

                    assert step0_logits_s is not None and step0_hidden_s is not None
                    prev_tokens_s = step0_logits_s.argmax(dim=-1).tolist()
                    draft_tokens_s: dict[int, list[int]] = {s: [prev_tokens_s[0]]}
                    next_pos_s = [self._r.slot_draft_sync_len[s]]
                    prior_kv_s = [self._r.slot_draft_sync_len[s]]
                    self._mtp_run_continuation_steps(
                        [s],
                        draft_tokens_s,
                        prev_tokens_s,
                        step0_hidden_s,
                        next_pos_s,
                        prior_kv_s,
                        k,
                    )
                    self._r._publish_committed_blocks(s, prompt, len(prompt))
                    self._r.slot_pending_draft_tokens[s] = draft_tokens_s[s]
                    result[s] = {"anchor": anchor_s, "draft_tokens": draft_tokens_s[s]}

        # --- COLD set: P2 same-round fan-out fork when >=2 cold slots; a lone
        #     cold slot uses the two-phase populate prefill (completion GDN
        #     checkpoint required for hit-after-cold; see docstring). ---
        if cold_idx:
            cold_slots = [slots[i] for i in cold_idx]
            cold_prompts = [prompts_per_slot[i] for i in cold_idx]
            if len(cold_slots) >= 2:
                result.update(self.mtp_prefill_fanout_batch(cold_slots, cold_prompts))
            else:
                result[cold_slots[0]] = self._r._prefill_cold_with_populate(
                    cold_slots[0], cold_prompts[0]
                )

        return result

    def mtp_verify_and_commit_batch(
        self,
        slots: list[int],
        anchors: dict[int, int],
        draft_tokens: dict[int, list[int]],
        *,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> dict[int, dict]:
        """Batched analogue of ``mtp_verify_and_commit`` -- **Phase 2,
        2026-07-18 rewrite** (``notes/2026-07-17-post-ragged-round-next-steps.md``
        section 10/11): now uses the REAL spec-decode GDN mechanism
        (``verify_batch_spec`` / ``build_gdn_metadata_spec_batch`` /
        ``_ssm_spec_row``) instead of Phase 0-3's chunked-GDN-metadata +
        snapshot/restore + recompute-forward mechanism. This eliminates an
        entire class of extra work this function used to do on every
        partial-reject round (``benchmarks/mtp_batch_recompute_cost_diag.py``
        found that was 84.4% of real rounds, ~56% of round wall time):
        there is no more snapshot, no more restore, no more separate
        recompute forward pass, and (as a direct consequence) no more
        full-accept/recompute GROUP SPLIT at all -- every slot in this
        call, regardless of its own accept/reject outcome, is handled by
        the exact same code path below.

        Why one uniform code path is now correct (not just simpler): GDN's
        recurrent state, under the real spec-decode kernel
        (``fused_sigmoid_gating_delta_rule_update_kernel`` -- re-verified
        directly against source this round, see ``_ssm_spec_row``'s
        docstring), computes a causally-valid PER-POSITION output for
        every one of the K+1 candidate positions in a single verify
        forward, unconditionally, regardless of which candidates later
        turn out to be real -- exactly like attention already was
        (content/position-addressed, so a rejected position's KV is
        simply never read again). Only the recurrent STATE COMMIT (which
        physical row is read next round) is acceptance-aware, via
        ``num_accepted_tokens``/``_ssm_spec_row`` -- the per-position
        OUTPUT itself needs no rollback or recomputation. This means
        every slot's hidden states for positions ``0..committed_len-1``
        are already sitting in ``verify_hidden``, correct, from the ONE
        verify forward this method issues -- a straight ragged SLICE
        (never a second forward pass) is all the draft resync step needs,
        for full-accept slots (``committed_len == k+1``) exactly as much
        as for any partial-reject slot (``committed_len < k+1``).

        Persistent per-slot bookkeeping this round introduces:
        ``self._r.slot_num_accepted_tokens`` (this slot's real committed
        length from ITS OWN last verify round, or bootstrap 1 right after
        a real prefill) -- read by ``build_gdn_metadata_spec_batch`` to
        select which of last round's K+1 dedicated SSM rows holds the
        valid state to resume from, and updated here after every round.

        **2026-07-18, Phase 2 CUDA-graph reconciliation**: the verify
        forward now goes through a CUDA-graph replay
        (``CapturedBatchDecodeGraph``, via ``self._r._get_verify_graph``,
        rebuilt this round to fill its GDN metadata via the same
        spec-decode mechanism this method uses -- see that class's
        docstring) whenever ``self._r.enable_cudagraph`` is on AND this
        runner has enough spare slot capacity (``num_slots >= 2*len(slots)``).
        Falls back to the eager ``verify_batch_spec`` path -- correctly,
        not silently -- whenever the graph isn't available for this
        call's batch_size (``enable_cudagraph`` off, matching every
        existing correctness suite; or a caller's active-slot count isn't
        one this runner precaptured). Since Phase 2 removed the separate
        recompute forward entirely, there is only ONE verify-shaped
        forward per round now, always at ``qo_len=k+1`` -- unlike Phase 3's
        old recompute-forward-graph-reuse special case (which needed a
        DIFFERENT qo_len per ragged recompute group), this graph lookup is
        now always the exact same shape every round, for every slot.

        **2026-07-18, Phase B**: ``mtp_verify_and_commit`` (the
        singular/looped sibling) was, through this point, intentionally
        NOT migrated -- it used the old chunked + snapshot/restore +
        recompute-forward mechanism unconditionally. Phase B (see that
        method's own docstring) migrated it to this SAME spec-decode
        mechanism, applied at batch_size=1 -- both production verify paths
        now share one mechanism. ``snapshot_gdn_state``/``restore_gdn_state``
        remain in the codebase regardless (a falsifier check confirmed
        ``benchmarks/mtp_gdn_rollback_check.py`` and several other
        diagnostics test/use them directly, independent of either verify
        path), just no longer called from ANY production verify path.

        Returns a dict keyed by slot id, each value shaped exactly like
        ``mtp_verify_and_commit``'s own return dict (plus ``next_anchor``/
        ``next_draft_tokens``) -- the external contract is unchanged."""
        k = len(draft_tokens[slots[0]])
        drafts_by_slot = {s: [anchors[s]] + draft_tokens[s] for s in slots}
        drafts = [drafts_by_slot[s] for s in slots]
        kv_lens_before = {s: self._r.slot_kv_len[s] for s in slots}
        num_accepted_prev = [self._r.slot_num_accepted_tokens[s] for s in slots]

        graph = self._r._get_verify_graph(len(slots), k + 1) if self._r.enable_cudagraph else None
        if graph is not None:
            verify_logits, verify_hidden = graph.replay(
                slots,
                drafts,
                [kv_lens_before[s] for s in slots],
                commit=False,
                return_hidden=True,
                num_accepted_tokens_prev=num_accepted_prev,
            )
        else:
            verify_logits, verify_hidden = self.verify_batch_spec(
                slots,
                drafts,
                [kv_lens_before[s] for s in slots],
                num_accepted_tokens_prev=num_accepted_prev,
                return_hidden=True,
            )

        decisions = determine_accept_reject_batch(slots, drafts_by_slot, verify_logits, k)
        for _s in slots:
            record_mtp_acceptance(decisions[_s]["num_accepted"])

        real_new_tokens = {s: [anchors[s]] + decisions[s]["committed"][:-1] for s in slots}
        next_anchors = {s: decisions[s]["committed"][-1] for s in slots}
        committed_lens = {s: decisions[s]["num_accepted"] + 1 for s in slots}

        for s in slots:
            self._r.slot_kv_len[s] = kv_lens_before[s] + committed_lens[s]
            self._r.slot_num_accepted_tokens[s] = committed_lens[s]
            record_slot_kv_usage(
                s,
                self._r.slot_kv_len[s] // self._r.block_size,
                self._r.blocks_per_slot,
            )
        # P3.2 decode-position populate (per slot): publish any newly-FULL
        # committed blocks now that slot_kv_len advanced by each slot's REAL
        # committed length (only committed tokens; INV4). No-op off-flag.
        if self._r.enable_persistent_prefix_cache:
            for s in slots:
                self._r.publish_committed_decode_blocks(s, real_new_tokens[s])

        # Ragged slice of the ONE verify forward's hidden states -- see
        # this method's docstring for why this is valid for EVERY slot
        # regardless of committed_len, not just full-accept ones.
        real_new_hidden: dict[int, torch.Tensor] = {}
        for i, s in enumerate(slots):
            real_new_hidden[s] = verify_hidden[i * (k + 1) : i * (k + 1) + committed_lens[s]]

        shifted = [real_new_tokens[s][1:] + [next_anchors[s]] for s in slots]
        hidden_concat = torch.cat([real_new_hidden[s] for s in slots], dim=0)
        start_pos_list = [self._r.slot_draft_sync_len[s] for s in slots]
        next_drafts_batch = self._mtp_sync_and_propose_batch(
            slots,
            shifted,
            hidden_concat,
            start_pos_list,
            num_new_tokens=[committed_lens[s] for s in slots],
            k=k,
        )

        # C2: compute logprobs from verify logits if requested
        logprobs_by_slot: dict[int, list[dict]] = {}
        if return_logprobs:
            for i, s in enumerate(slots):
                n_committed = committed_lens[s]
                slot_logits = verify_logits[i * (k + 1) : i * (k + 1) + n_committed]
                committed = decisions[s]["committed"]
                logprobs_by_slot[s] = compute_logprobs(
                    slot_logits,
                    committed,
                    top_k=top_logprobs,
                )

        result: dict[int, dict] = {}
        for s in slots:
            self._r.slot_pending_draft_tokens[s] = next_drafts_batch[s]
            result[s] = {
                **decisions[s],
                "next_anchor": next_anchors[s],
                "next_draft_tokens": next_drafts_batch[s],
            }
            if return_logprobs:
                result[s]["logprobs"] = logprobs_by_slot[s]
        return result
