"""B5 模块化：CUDA Graph 域。

从 direct_model_runner.py 提取的 CapturedBatchDecodeGraph + CapturedMTPDraftStepGraph。
纯移动不改逻辑（B5 parity 门禁）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from runtime.direct_model_runner import DirectModelRunner

from runtime.block_pool import RESERVED_PHYSICAL_SLOTS, _physical_slot, _ssm_spec_row
from runtime.legacy_qwen36_attention import GDNAttentionMetadata, SM120GQAMetadata
from runtime.legacy_qwen36_vllm import set_forward_context


class CapturedBatchDecodeGraph:
    """CUDA-graph-captured batch decode/verify for a FIXED batch size and
    FIXED ``qo_len`` (1 = pure decode, >1 = MTP/speculative-decode verify,
    e.g. 4 for K=3 draft + 1 bonus token), replayable at ANY per-slot
    kv_len up to this runtime's per-slot capacity (``blocks_per_slot *
    block_size``) -- not just whatever dummy shape was used at capture
    time.

    2026-07-16, CUDA Graph round: this project's own read of
    ``vllm/v1/attention/backends/sm120_gqa.py``'s documented history (a
    real illegal-memory-access crash, root-caused to metadata tensors
    without fixed addresses) plus its OTHER documented lesson
    (``kv_split_size``/``max_num_splits`` frozen at capture time going
    stale under a later, larger real kv_len) directly motivate this
    class's two central design points:

    1. Every tensor a captured kernel launch reads (metadata CSR tensors,
       input_ids, positions, slot_mapping) is a PERSISTENT, fixed-address
       buffer, allocated once in ``__init__``. ``replay()`` writes freshly
       computed REAL values into these SAME buffers via ``.copy_()`` --
       it never reallocates them. This is what makes replaying at a
       kv_len the buffers were never filled with at capture time safe.
    2. ``kv_split_size``/``max_num_splits`` are derived ONCE from this
       runtime's configured per-slot page-table limit (``blocks_per_slot *
       block_size`` -- a software ceiling the caller chose when
       constructing ``DirectModelRunner``, NOT a GPU hardware limit), via
       ``build_attention_metadata_batch``'s ``fixed_kv_split_size``/
       ``fixed_max_num_splits`` parameters -- see that function's
       docstring for the correctness proof that this bounds every real
       kv_len up to that configured limit, not just the capture-time
       value.

    2026-07-16, MTP extension (qo_len>1), **superseded 2026-07-18 (Phase 2
    CUDA-graph reconciliation)**: originally this replicated GDN's
    chunked/"prefill" metadata (``build_gdn_metadata_batch``'s
    ``chunk_indices``/``chunk_offsets``/``nums_dict``/``batch_ptr``/
    ``token_chunk_offset_ptr``/``has_initial_state`` fields), matching
    what the eager verify path did at the time. The eager path has since
    moved to the REAL spec-decode GDN mechanism
    (``build_gdn_metadata_spec_batch`` -- K+1 dedicated SSM rows,
    acceptance-aware addressing, no snapshot/restore/recompute-forward;
    see ``mtp_verify_and_commit_batch``'s docstring) for the exact same
    reason this class exists at all: qo_len>1 here ONLY ever means MTP
    verify. This class now mirrors that mechanism instead:
    ``spec_query_start_loc``/``spec_sequence_masks`` are CONSTANT for a
    fixed (batch_size, qo_len) pair (computed once, same reasoning as
    ``qo_indptr`` above); ``spec_state_indices_tensor`` (a FIXED
    per-(slot, column) row mapping by construction -- see
    ``_ssm_spec_row`` -- but slot IDENTITY varies per replay, since one
    graph object is reused across calls with different active slot sets
    at the same batch_size) and ``num_accepted_tokens`` (the real
    per-round accept/reject outcome, inherently different every replay)
    are both refilled in ``_fill_buffers`` every replay, exactly like
    ``kv_page_indices``/``state_indices`` already were.

    A replayed CUDA graph is a pre-recorded sequence of GPU kernel
    launches, NOT a re-execution of Python control flow -- ``model
    .forward()`` (the Python function) is only ever actually called
    during ``capture()`` (plus its warmup iterations), never during
    ``replay()``. This means whatever kernel-dispatch branch
    ``SM120GQAImpl.forward()`` takes (decode-kernel vs general, FP8 vs
    NVFP4, MMA vs v2 vs scalar, ...) must be identical for every real
    kv_len this graph will ever replay at -- true here because dispatch
    depends only on ``qo_len``/kv-cache dtype/model config, all fixed for
    a given (batch_size, qo_len) graph, never on the live kv_len itself.

    2026-07-17, state-neutral capture (correctness fix, found via an
    independent review this project's coordinator commissioned and
    personally verified): ``capture()``'s warmup runs 3 REAL executions
    on a side stream before the graph trace (the trace itself, inside
    ``with torch.cuda.graph(g):``, executes nothing -- confirmed against
    the sibling project's own kernel-level CUDA-graph test). Attention's
    paged KV cache tolerates redundant warmup writes fine (same position,
    same value, overwritten harmlessly) -- but GDN's recurrent/chunked
    state update reads-old-state-and-writes-new-state each call, so it is
    NOT idempotent under repeated identical input; running warmup against
    slots a caller will later actually replay against silently advances
    those slots' real GDN state by 3 extra (unaccounted, un-bookkept)
    applications before any real replay happens. This was a genuine gap
    in this project's own initial qo_len=1/MTP test scripts (both reused
    real/twin-established slots for warmup) -- their empirical PASS
    results are not proof this doesn't matter, only evidence it didn't
    surface for that specific signal-probe task (plausibly because
    full-attention layers dominate identity recall, masking a GDN
    perturbation a GDN-sensitive task might not tolerate).

    Fix: this class now reserves ``batch_size`` of the runner's logical
    slots PERMANENTLY for its own exclusive, disposable warmup use (the
    LAST ``batch_size`` slots of ``runner.num_slots`` -- see
    ``self._warmup_slots``) -- ``capture()`` takes no external slot/token/
    kv_length arguments at all anymore, so this can no longer depend on
    caller discipline the way the original design implicitly did. Callers
    must size ``runner.num_slots >= 2 * batch_size`` and never pass this
    graph's reserved warmup slots to ``replay()`` or any other runner
    method.

    Also fixed the same round: ``replay()`` no longer calls
    ``torch.cuda.synchronize()`` (see that method's docstring for why this
    is safe and why the removed blanket device-wide sync worked against
    the whole point of using a captured graph to cut CPU-side dispatch
    overhead), and ``_fill_buffers`` now computes per-replay values via
    plain Python arithmetic instead of round-tripping through
    ``build_attention_metadata_batch``/``build_gdn_metadata_batch``/
    ``DirectModelRunner._slot_mapping_batch`` (which each construct several
    of their own intermediate GPU tensors -- real, avoidable per-replay
    allocation overhead on what should be a lean hot path). This is a
    partial mitigation (each static buffer's ``.copy_()`` source is still
    a freshly constructed small tensor, not a persistent pinned staging
    buffer written in place) -- a fully allocation-free version is a
    further optimization, not attempted this round.
    """

    def __init__(
        self,
        runner: DirectModelRunner,
        batch_size: int,
        qo_len: int = 1,
        warmup_slots: list[int] | None = None,
    ) -> None:
        if warmup_slots is not None:
            if len(warmup_slots) != batch_size:
                raise ValueError(
                    f"warmup_slots must have exactly batch_size ({batch_size}) "
                    f"entries, got {len(warmup_slots)}"
                )
            self._external_warmup = True
        else:
            if runner.num_slots < 2 * batch_size:
                raise ValueError(
                    f"runner.num_slots={runner.num_slots} must be >= 2*batch_size "
                    f"({2 * batch_size}) when warmup_slots is not provided"
                )
            warmup_slots = list(range(runner.num_slots - batch_size, runner.num_slots))
            self._external_warmup = False
        self.runner = runner
        self.batch_size = batch_size
        self.qo_len = qo_len
        device = runner.device
        blocks_per_slot = runner.blocks_per_slot
        # 2026-07-18, Phase 2 CUDA-graph reconciliation: this class used to
        # derive its own split-KV config from a stale local
        # ``TARGET_SPLITS = 16`` constant (a leftover from an earlier
        # round, predating the later real-production tuning to 64
        # splits/request -- see __init__'s split-KV comment on
        # ``DirectModelRunner``). That staleness was harmless as long as
        # this graph's verify-step replay was never actually exercised in
        # production (true through this round's reconciliation); wiring it
        # into the real ``mtp_verify_and_commit_batch`` path makes it
        # matter for real -- a genuinely DIFFERENT split-KV count changes
        # the attention kernel's reduction order, which this project has
        # already established can flip near-tie accept/reject decisions
        # (found via a real W1-S run: a 70.3%->76.7% draft-acceptance-rate
        # shift between the eager and graph-replayed paths, too large to
        # be ordinary near-tie noise, before this fix). Now uses the SAME
        # runner-computed, currently-tuned value every eager caller and
        # ``CapturedMTPDraftStepGraph`` already use, instead of maintaining
        # a second, independently-stale derivation.
        self.fixed_kv_split_size = runner.decode_fixed_kv_split_size
        self.fixed_max_num_splits = runner.decode_fixed_max_num_splits

        self._warmup_slots = warmup_slots
        self._last_num_pages_per_req: list[int] | None = None
        self._last_state_slot_ids: list[int] | None = None

        num_reqs = batch_size
        n_tokens = num_reqs * qo_len

        # Attention metadata static buffers -- worst-case sized (a request
        # could in principle use this slot's entire page capacity).
        # qo_indptr is CONSTANT for a fixed (batch_size, qo_len) pair
        # ([0, qo_len, 2*qo_len, ..., num_reqs*qo_len]) -- computed once,
        # never refilled.
        self.static_qo_indptr = (
            torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device) * qo_len
        )
        self.static_kv_page_indptr = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        self.static_kv_page_indices = torch.zeros(
            num_reqs * blocks_per_slot, dtype=torch.int32, device=device
        )
        self.static_kv_last_page_len = torch.zeros(num_reqs, dtype=torch.int32, device=device)

        # GDN metadata static buffers. non_spec_query_start_loc is
        # likewise constant; state_indices is per-replay-filled (depends
        # on slot_ids, not just batch_size/qo_len).
        self.static_state_indices = torch.zeros(num_reqs, dtype=torch.int32, device=device)
        self.static_non_spec_qsl = (
            torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device) * qo_len
        )

        # Model I/O static buffers.
        self.static_input_ids = torch.zeros(n_tokens, dtype=torch.long, device=device)
        self.static_positions = torch.zeros(n_tokens, dtype=torch.long, device=device)
        self.static_slot_mapping = torch.zeros(n_tokens, dtype=torch.long, device=device)

        # 2026-07-18, Phase 2 CUDA-graph reconciliation: for qo_len>1 this
        # class's ONLY real remaining use is MTP verify, which the eager
        # path (``verify_batch_spec``) now does via the REAL spec-decode
        # GDN mechanism, not the old chunked/"prefill" one this class used
        # to replicate here (``_const_gdn_extra``, removed). Static buffers
        # mirror ``build_gdn_metadata_spec_batch``'s fields:
        # ``spec_query_start_loc``/``spec_sequence_masks`` are CONSTANT for
        # a fixed (batch_size, qo_len) pair (computed once, like
        # ``static_qo_indptr`` above); ``spec_state_indices_tensor`` (a
        # FIXED per-(slot, column) row mapping -- see ``_ssm_spec_row``'s
        # docstring -- but slot IDENTITY varies per replay, since this
        # class is reused across calls with potentially different active
        # slot sets at the same batch_size) and ``num_accepted_tokens``
        # (the real per-round accept/reject outcome, inherently different
        # every replay) are both per-replay-filled in ``_fill_buffers``.
        self.num_spec = runner.num_speculative_tokens
        self.total_physical_slots = runner.num_slots + RESERVED_PHYSICAL_SLOTS
        self.static_spec_query_start_loc: torch.Tensor | None = None
        self.static_spec_sequence_masks: torch.Tensor | None = None
        self.static_spec_state_indices: torch.Tensor | None = None
        self.static_num_accepted_tokens: torch.Tensor | None = None
        if qo_len > 1:
            if self.num_spec is None:
                raise RuntimeError(
                    "CapturedBatchDecodeGraph with qo_len>1 requires MTP "
                    "(runner.num_speculative_tokens) to be configured -- "
                    "this shape only ever means spec-decode verify"
                )
            self.static_spec_query_start_loc = (
                torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device) * qo_len
            )
            self.static_spec_sequence_masks = torch.ones(num_reqs, dtype=torch.bool, device=device)
            self.static_spec_state_indices = torch.zeros(
                (num_reqs, qo_len), dtype=torch.int32, device=device
            )
            self.static_num_accepted_tokens = torch.zeros(
                num_reqs, dtype=torch.int32, device=device
            )

        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_logits: torch.Tensor | None = None
        # 2026-07-17, Phase 3: captured alongside logits (see
        # ``_forward_no_sync``/``capture()``) so ``replay(..., return_hidden=True)``
        # can hand back the target model's own hidden states -- needed by
        # ``mtp_verify_and_commit_batch`` to feed the MTP draft model's next
        # resync step without an extra eager forward. ``None`` until
        # ``capture()`` has run.
        self._static_hidden_states: torch.Tensor | None = None
        # Test-observability only (2026-07-17, consolidation pass): counts
        # real replay() calls so a correctness test can directly confirm a
        # graph was actually EXERCISED, not merely constructed/precaptured
        # (precapture populates the cache regardless of whether a given
        # round ever replays that specific shape) -- see
        # benchmarks/mtp_verify_cudagraph_check.py.
        self.replay_count = 0

        cpu = torch.device("cpu")
        self._cpu_kv_page_indptr = torch.zeros(
            num_reqs + 1, dtype=torch.int32, device=cpu, pin_memory=True
        )
        self._cpu_kv_page_indices = torch.zeros(
            num_reqs * blocks_per_slot, dtype=torch.int32, device=cpu, pin_memory=True
        )
        self._cpu_kv_last_page_len = torch.zeros(
            num_reqs, dtype=torch.int32, device=cpu, pin_memory=True
        )
        self._cpu_state_indices = torch.zeros(
            num_reqs, dtype=torch.int32, device=cpu, pin_memory=True
        )
        self._cpu_input_ids = torch.zeros(n_tokens, dtype=torch.long, device=cpu, pin_memory=True)
        self._cpu_positions = torch.zeros(n_tokens, dtype=torch.long, device=cpu, pin_memory=True)
        self._cpu_slot_mapping = torch.zeros(
            n_tokens, dtype=torch.long, device=cpu, pin_memory=True
        )
        self._np_kv_page_indptr = self._cpu_kv_page_indptr.numpy()
        self._np_kv_page_indices = self._cpu_kv_page_indices.numpy()
        self._np_kv_last_page_len = self._cpu_kv_last_page_len.numpy()
        self._np_state_indices = self._cpu_state_indices.numpy()
        self._np_input_ids = self._cpu_input_ids.numpy()
        self._np_positions = self._cpu_positions.numpy()
        self._np_slot_mapping = self._cpu_slot_mapping.numpy()
        if qo_len > 1:
            self._cpu_spec_state_indices = torch.zeros(
                (num_reqs, qo_len), dtype=torch.int32, device=cpu, pin_memory=True
            )
            self._cpu_num_accepted_tokens = torch.zeros(
                num_reqs, dtype=torch.int32, device=cpu, pin_memory=True
            )

    def _fill_buffers(
        self,
        slot_ids: list[int],
        token_ids,
        kv_lengths: list[int],
        *,
        num_accepted_tokens_prev: list[int] | None = None,
    ) -> None:
        """Write real, per-replay-varying values into the persistent static
        buffers. Computes everything via plain Python arithmetic (CPU-only,
        no GPU allocation) instead of calling
        ``build_attention_metadata_batch``/``build_gdn_metadata_batch``/
        ``DirectModelRunner._slot_mapping_batch`` -- those each construct
        several of their own intermediate GPU tensors (dataclass fields the
        caller doesn't need here), real avoidable overhead on a hot path
        meant to be lean. Each static buffer's ``.copy_()`` source below is
        still a freshly built small tensor (a partial mitigation, not a
        fully allocation-free design -- see the class docstring).

        ``num_accepted_tokens_prev`` (2026-07-18, Phase 2 CUDA-graph
        reconciliation, required when ``self.qo_len > 1``): each slot's
        real committed length from ITS OWN last verify round (or bootstrap
        1 right after a real prefill) -- see ``build_gdn_metadata_spec_batch``'s
        docstring. Used to fill ``static_num_accepted_tokens``;
        ``static_spec_state_indices`` is filled from ``slot_ids`` alone
        (``_ssm_spec_row`` is a fixed per-(slot, column) mapping, but slot
        IDENTITY genuinely varies per replay -- this class is reused
        across calls with potentially different active slot sets at the
        same batch_size)."""
        runner = self.runner
        qo_len = self.qo_len
        block_size = runner.block_size
        blocks_per_slot = runner.blocks_per_slot

        if qo_len == 1:
            flat_token_ids = token_ids
        else:
            flat_token_ids = [tok for slot_tokens in token_ids for tok in slot_tokens]

        new_kv_lens = [kv_len + qo_len for kv_len in kv_lengths]
        num_pages_per_req = [(kv_len + block_size - 1) // block_size for kv_len in new_kv_lens]
        for slot, kv_len, num_pages in zip(slot_ids, new_kv_lens, num_pages_per_req):
            if num_pages > blocks_per_slot:
                raise RuntimeError(
                    f"slot {slot} kv_len {kv_len} exceeds this slot's "
                    f"{blocks_per_slot * block_size}-token capacity"
                )

        # P1 (notes/prefix-cache-design.md sec 5): grow every replayed
        # slot's block_table to cover this replay's own new_kv_len BEFORE
        # reading it below (INV5 -- the captured launch itself never bakes
        # in a physical block id, only this Python-side refill does, every
        # replay).
        if runner.enable_block_table:
            for slot, kv_len in zip(slot_ids, new_kv_lens):
                runner._ensure_blocks(slot, kv_len)

        pages_unchanged = (
            self._last_num_pages_per_req is not None
            and num_pages_per_req == self._last_num_pages_per_req
            and slot_ids == getattr(self, "_last_slot_ids", None)
        )
        if not pages_unchanged:
            kv_page_indptr_list = [0]
            for num_pages in num_pages_per_req:
                kv_page_indptr_list.append(kv_page_indptr_list[-1] + num_pages)

            page_indices_list: list[int] = []
            for slot, num_pages in zip(slot_ids, num_pages_per_req):
                if runner.enable_block_table:
                    page_indices_list.extend(runner.block_table[slot][:num_pages])
                else:
                    first_block = _physical_slot(slot) * blocks_per_slot
                    page_indices_list.extend(range(first_block, first_block + num_pages))

            n_indptr = len(kv_page_indptr_list)
            self._np_kv_page_indptr[:n_indptr] = kv_page_indptr_list
            self.static_kv_page_indptr.copy_(self._cpu_kv_page_indptr, non_blocking=True)
            self.static_kv_page_indices.zero_()
            n_pages = len(page_indices_list)
            if n_pages:
                self._np_kv_page_indices[:n_pages] = page_indices_list
                self.static_kv_page_indices[:n_pages].copy_(
                    self._cpu_kv_page_indices[:n_pages], non_blocking=True
                )
            self._last_num_pages_per_req = list(num_pages_per_req)
            self._last_slot_ids = list(slot_ids)

        last_page_len_list = [
            kv_len - (num_pages - 1) * block_size
            for kv_len, num_pages in zip(new_kv_lens, num_pages_per_req)
        ]
        positions_list = [kv_len + j for kv_len in kv_lengths for j in range(qo_len)]

        slot_mapping_list: list[int] = []
        for slot, kv_len in zip(slot_ids, kv_lengths):
            table = runner.block_table[slot] if runner.enable_block_table else None
            first_block = _physical_slot(slot) * blocks_per_slot
            for j in range(qo_len):
                pos = kv_len + j
                block_id = (
                    table[pos // block_size]
                    if table is not None
                    else first_block + pos // block_size
                )
                offset = pos % block_size
                slot_mapping_list.append(block_id * block_size + offset)
        n_reqs = len(last_page_len_list)
        self._np_kv_last_page_len[:n_reqs] = last_page_len_list
        self.static_kv_last_page_len.copy_(self._cpu_kv_last_page_len, non_blocking=True)
        slots_changed = slot_ids != self._last_state_slot_ids
        if slots_changed:
            state_indices_list = [_physical_slot(slot) for slot in slot_ids]
            self._np_state_indices[:n_reqs] = state_indices_list
            self.static_state_indices.copy_(self._cpu_state_indices, non_blocking=True)
            self._last_state_slot_ids = list(slot_ids)
        n_tok = len(flat_token_ids)
        self._np_input_ids[:n_tok] = flat_token_ids
        self.static_input_ids.copy_(self._cpu_input_ids, non_blocking=True)
        self._np_positions[:n_tok] = positions_list
        self.static_positions.copy_(self._cpu_positions, non_blocking=True)
        self._np_slot_mapping[:n_tok] = slot_mapping_list
        self.static_slot_mapping.copy_(self._cpu_slot_mapping, non_blocking=True)

        if qo_len > 1:
            if num_accepted_tokens_prev is None:
                raise ValueError(
                    "num_accepted_tokens_prev is required when qo_len > 1 "
                    "(this shape only ever means spec-decode verify)"
                )
            if slots_changed or not hasattr(self, "_spec_cached"):
                spec_indices_list = [
                    [
                        _ssm_spec_row(slot, col, self.total_physical_slots, self.num_spec)
                        for col in range(qo_len)
                    ]
                    for slot in slot_ids
                ]
                self._cpu_spec_state_indices[:n_reqs] = torch.tensor(
                    spec_indices_list, dtype=torch.int32
                )
                self.static_spec_state_indices.copy_(
                    self._cpu_spec_state_indices, non_blocking=True
                )
                self._spec_cached = True
            self._cpu_num_accepted_tokens[:n_reqs] = torch.tensor(
                num_accepted_tokens_prev, dtype=torch.int32
            )
            self.static_num_accepted_tokens.copy_(self._cpu_num_accepted_tokens, non_blocking=True)

    def _static_metadata_dicts(self) -> tuple[dict, dict]:
        runner = self.runner
        n_tokens = self.batch_size * self.qo_len
        attn_meta = SM120GQAMetadata(
            num_actual_tokens=n_tokens,
            num_reqs=self.batch_size,
            qo_indptr=self.static_qo_indptr,
            kv_page_indptr=self.static_kv_page_indptr,
            kv_page_indices=self.static_kv_page_indices,
            kv_last_page_len=self.static_kv_last_page_len,
            page_size=runner.block_size,
            is_pure_decode=(self.qo_len == 1),
            kv_split_size=self.fixed_kv_split_size,
            max_num_splits=self.fixed_max_num_splits,
            decode_qo_len=self.qo_len,
        )
        if self.qo_len == 1:
            gdn_meta = GDNAttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=self.batch_size,
                num_decode_tokens=self.batch_size,
                num_spec_decodes=0,
                num_spec_decode_tokens=0,
                num_actual_tokens=self.batch_size,
                non_spec_query_start_loc=self.static_non_spec_qsl,
                non_spec_state_indices_tensor=self.static_state_indices,
            )
        else:
            # 2026-07-18, Phase 2 CUDA-graph reconciliation: this class's
            # only real remaining qo_len>1 use is MTP verify, which the
            # eager path now does via the REAL spec-decode GDN mechanism
            # (build_gdn_metadata_spec_batch), not the old chunked/
            # "prefill" one this branch used to replicate (removed along
            # with self._const_gdn_extra). See __init__'s comment for the
            # static-buffer rationale.
            gdn_meta = GDNAttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=0,
                num_decode_tokens=0,
                num_spec_decodes=self.batch_size,
                num_spec_decode_tokens=n_tokens,
                num_actual_tokens=n_tokens,
                spec_query_start_loc=self.static_spec_query_start_loc,
                spec_state_indices_tensor=self.static_spec_state_indices,
                spec_sequence_masks=self.static_spec_sequence_masks,
                num_accepted_tokens=self.static_num_accepted_tokens,
            )
        attn_metadata_dict = {name: attn_meta for name in runner.attn_layer_names}
        attn_metadata_dict.update({name: gdn_meta for name in runner.gdn_layer_names})
        slot_mapping_dict = {name: self.static_slot_mapping for name in runner.attn_layer_names}
        return attn_metadata_dict, slot_mapping_dict

    def _forward_no_sync(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Same op sequence as ``DirectModelRunner._forward_batch``, minus
        the ``torch.cuda.synchronize()`` calls -- calling those DURING
        capture is a documented CUDA-graph-capture violation (raises
        ``cudaErrorStreamCaptureUnsupported``), the same error class the
        sibling project already hit and documented for a different op (a
        boolean-mask-select) during its own CUDA Graph work.

        Returns ``(logits, hidden_states)`` (2026-07-17, Phase 3 addition --
        previously only logits were returned/captured; ``hidden_states`` is
        now ALSO captured so ``replay(..., return_hidden=True)`` can hand it
        back, matching ``_forward_batch``'s own ``return_hidden`` contract).
        This is a backward-compatible extension: the only caller inside
        ``capture()``'s warmup loop discards both return values either way."""
        runner = self.runner
        attn_metadata_dict, slot_mapping_dict = self._static_metadata_dicts()
        with set_forward_context(
            attn_metadata_dict, runner.vllm_config, slot_mapping=slot_mapping_dict
        ):
            hidden_states = runner.model.forward(self.static_input_ids, self.static_positions)
        logits = runner.model.compute_logits(hidden_states)
        return logits, hidden_states

    def capture(self) -> None:
        """Warm up (uncaptured, on a side stream -- required by
        ``torch.cuda.graph`` before capture) then capture the graph, using
        this object's OWN permanently reserved, disposable warmup slots
        (``self._warmup_slots``) -- NEVER any slot a caller will later pass
        to ``replay()``. This is what makes capture state-neutral for real
        traffic: the only slots capture()'s 3 real warmup executions (plus
        the graph-trace call, which itself executes nothing -- see the
        class docstring) can touch are these reserved slots.

        Warmup content is disposable and never checked for correctness --
        any valid token id works, so a fixed dummy prompt is used (``[0,
        0, 0, 0, 0]``, matching ``DirectModelRunner._warmup``'s own
        convention). This is also why fixed-sizing kv_split_size/
        max_num_splits is required in the first place: the real kv_len
        distribution ``replay()`` sees is expected to differ, often
        drastically, from this disposable warmup shape."""
        if self._graph is not None:
            raise RuntimeError("already captured")
        runner = self.runner
        warmup_slots = self._warmup_slots
        for slot in warmup_slots:
            if runner.slot_kv_len[slot] != 0:
                raise RuntimeError(
                    f"reserved warmup slot {slot} is not fresh -- capture() "
                    "must run before anything else touches this graph's "
                    "own warmup slots, and exactly once per graph object"
                )
            runner.prefill(slot, [0, 0, 0, 0, 0])
        warmup_kv_lengths = [runner.slot_kv_len[s] for s in warmup_slots]
        if self.qo_len == 1:
            warmup_token_ids = [0] * self.batch_size
            warmup_num_accepted_tokens_prev = None
        else:
            warmup_token_ids = [[0] * self.qo_len for _ in range(self.batch_size)]
            # Bootstrap value (column 0 -- the row runner.prefill() above
            # just wrote into), matching real usage's own first-ever-verify
            # convention (see build_gdn_metadata_spec_batch's docstring).
            warmup_num_accepted_tokens_prev = [1] * self.batch_size
        self._fill_buffers(
            warmup_slots,
            warmup_token_ids,
            warmup_kv_lengths,
            num_accepted_tokens_prev=warmup_num_accepted_tokens_prev,
        )

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._forward_no_sync()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._static_logits, self._static_hidden_states = self._forward_no_sync()
        self._graph = g

        if self._external_warmup:
            for slot in self._warmup_slots:
                runner.reset_slot(slot)

    def replay(
        self,
        slot_ids: list[int],
        token_ids,
        kv_lengths: list[int],
        *,
        commit: bool = True,
        return_hidden: bool = False,
        num_accepted_tokens_prev: list[int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Replay the captured graph at REAL (slot_ids, token_ids,
        kv_lengths) data -- may (and, per this round's explicit test
        scope, deliberately does) differ drastically from capture()'s
        warmup data, including kv_len values much larger or smaller than
        whatever was used at capture time. Returns logits shaped
        ``[batch_size * qo_len, vocab]`` (request-then-position order).

        ``return_hidden`` (2026-07-17, Phase 3 addition, default ``False``
        preserving every existing caller's behavior byte-for-byte): when
        ``True``, returns ``(logits, hidden_states)`` instead of just
        ``logits`` -- mirrors ``_forward_batch``'s own ``return_hidden``
        parameter. Needed by ``mtp_verify_and_commit_batch`` so a
        graph-captured verify replay can still feed the MTP draft model's
        next resync step (which needs the target model's hidden states,
        not just its logits) without an extra eager forward call.

        ``num_accepted_tokens_prev`` (2026-07-18, Phase 2 CUDA-graph
        reconciliation, required when ``self.qo_len > 1``): each slot's
        real committed length from ITS OWN last verify round (bootstrap 1
        right after a real prefill) -- forwarded to ``_fill_buffers`` to
        select which of last round's K+1 dedicated SSM rows holds the
        valid state to resume from. Mirrors ``verify_batch_spec``'s own
        parameter of the same name on the eager path.

        ``commit`` (2026-07-17 fix, Codex-sol review, confirmed real):
        mirrors ``_forward_batch``'s own ``commit`` parameter -- this
        method used to advance ``self.runner.slot_kv_len`` by
        ``self.qo_len`` UNCONDITIONALLY, the exact physical-write-vs-
        committed conflation already fixed on the eager path (see
        ``_forward_batch``'s docstring) but left inconsistent here, since
        this class has its own separate captured-graph call path that
        never goes through ``_forward_batch``. For ``qo_len==1`` (plain
        decode) this default is harmless (never ambiguous). For
        ``qo_len>1`` (MTP verify), the caller (``mtp_verify_and_commit_batch``,
        2026-07-18) passes ``commit=False`` and applies the real
        committed-length correction itself (plus updates
        ``slot_num_accepted_tokens`` for the NEXT round) after determining
        accept/reject on the returned logits -- exactly like the eager
        ``verify_batch_spec`` path.

        No ``torch.cuda.synchronize()`` here (removed 2026-07-17, a
        correctness-review finding): ``_fill_buffers``'s ``.copy_()`` calls
        and ``self._graph.replay()`` are all issued on the SAME (default)
        CUDA stream, so CUDA's own stream-ordering already guarantees the
        graph's kernels observe the freshly-copied buffer contents, and
        that the NEXT call's ``_fill_buffers`` won't overwrite data this
        replay is still reading -- no explicit device-wide sync is needed
        for that. The caller gets an implicit, narrowly-scoped sync for
        free the moment it actually reads back a value (e.g.
        ``.argmax(dim=-1).item()`` on the returned logits). A blanket
        ``torch.cuda.synchronize()`` here would additionally block on any
        OTHER unrelated work queued on the device -- directly working
        against the whole point of using a captured graph to cut CPU-side
        launch/dispatch overhead."""
        if not self._external_warmup and (
            slot_ids == self._warmup_slots or set(slot_ids) & set(self._warmup_slots)
        ):
            raise RuntimeError(
                f"slot(s) {set(slot_ids) & set(self._warmup_slots)} are this "
                "graph's own reserved warmup slots -- never replay() against "
                "them, they exist solely for capture()'s internal use"
            )
        if self._graph is None:
            raise RuntimeError("capture() must be called first")
        if not (len(slot_ids) == self.batch_size == len(token_ids) == len(kv_lengths)):
            raise ValueError("slot_ids/token_ids/kv_lengths must match batch_size")
        for slot, kv_len in zip(slot_ids, kv_lengths):
            if kv_len != self.runner.slot_kv_len[slot]:
                raise RuntimeError(
                    f"slot {slot}: caller-provided kv_length {kv_len} != "
                    f"tracked {self.runner.slot_kv_len[slot]}"
                )
            if not self.runner.slot_gdn_initialized[slot]:
                raise RuntimeError(f"slot {slot} has no GDN state yet (needs a prior prefill)")
        self._fill_buffers(
            slot_ids, token_ids, kv_lengths, num_accepted_tokens_prev=num_accepted_tokens_prev
        )
        self._graph.replay()
        self.replay_count += 1
        for slot in slot_ids:
            if commit:
                self.runner.slot_kv_len[slot] += self.qo_len
            self.runner.slot_gdn_initialized[slot] = True
        if return_hidden:
            return self._static_logits, self._static_hidden_states
        return self._static_logits


class CapturedMTPDraftStepGraph:
    """CUDA-graph-captured decode/resync step for the MTP DRAFT model
    (``runner.mtp_model``), for a FIXED ``batch_size`` and FIXED ``qo_len``
    (1 = the autoregressive continuation steps 1..k-1; >1 = step 0's
    resync, when its ``num_new_tokens`` happens to be UNIFORM across the
    batch -- always true for the full-accept group, which is always
    exactly k+1, and sometimes true for the recompute group, when every
    recompute slot's committed_len happens to coincide).
    2026-07-17, Phase 3 round 2 (coordinator-directed fast-iteration pass,
    picking off ``notes/2026-07-17-post-ragged-round-next-steps.md``'s
    Phase 3 candidate 1: "then the K draft steps") -- generalized from an
    initial qo_len=1-only version (same day, same round) once it became
    clear step 0 is ALSO uniform-shaped often enough to be worth capturing,
    exactly mirroring how ``mtp_verify_and_commit_batch``'s recompute
    forward reuses ``CapturedBatchDecodeGraph`` for its own uniform special
    case (see that method's comment) -- this class is the draft-model
    analogue of that same idea, generalized in place rather than as a
    parallel copy.

    Narrower than ``CapturedBatchDecodeGraph`` in two ways, both because
    the draft model's own call shape is simpler than the target model's:
    1. No GDN metadata at all -- ``runner.mtp_attn_layer_names`` never
       includes a GDN layer (the draft model registers only its own small
       full-attention layer(s); see ``DirectModelRunner.__init__``).
    2. Takes an EXTRA static input, ``static_hidden_states_in`` (shape
       ``[batch_size*qo_len, hidden_size]``), since ``Qwen3_5MultiTokenPredictor
       .forward()`` needs the running hidden-state carry between steps
       that the target model's own ``forward()`` does not.

    Always dispatches via ``decode_qo_len=qo_len``/``is_pure_decode=(qo_len==1)``
    (the fast decode/verify kernel path) regardless of what the EAGER path
    would have used for the same call -- for step 0 specifically, the eager
    ``_mtp_forward_batch`` passes ``is_decode=all(n==1 for n in
    num_new_tokens_list)``, which is ``False`` whenever ``qo_len>1``,
    routing eager step-0 calls through the general/chunked attention
    kernel instead. This is a DELIBERATE dispatch difference, not a bug:
    the underlying math (causal attention over a query range against a KV
    cache) is identical either way -- this project's own
    ``build_attention_metadata_batch`` docstring already establishes that
    the decode-kernel and general-kernel paths are numerically
    equivalent, just different kernel choices -- and using the fast
    decode/verify kernel here is a legitimate additional (small) win on
    top of the graph-capture win itself. Verified via
    ``benchmarks/mtp_verify_cudagraph_check.py``'s real content comparison
    against an independent eager reference, not assumed.

    Stateless w.r.t. runner bookkeeping: unlike ``CapturedBatchDecodeGraph
    .replay()``, this class's ``replay()`` does NOT touch
    ``runner.slot_kv_len``/``slot_gdn_initialized`` at all -- mirroring
    ``_mtp_forward_batch`` itself, which also never does (the caller
    tracks its own running length counters as plain local Python
    variables, not persistent per-slot runner state). ``prior_kv_len ==
    start_pos`` always holds for every real call site in this codebase
    (confirmed by inspection: ``_mtp_sync_and_propose_batch`` always
    derives ``start_pos_list`` from the exact same ``slot_draft_sync_len``
    snapshot its own internal ``prior_kv_lens_step0`` uses, and the
    qo_len=1 continuation loop advances both counters by the same amount
    every iteration), so this class only needs ONE per-slot length list,
    not two.

    Same warmup-slot-reservation and state-neutral-capture discipline as
    ``CapturedBatchDecodeGraph`` (see that class's docstring for the full
    rationale): the last ``batch_size`` logical slots of ``runner.num_slots``
    are reserved, used only by this graph's own ``capture()``, and reset
    immediately afterward. Warmup content is fully disposable (a dummy
    all-zeros hidden-state tensor and dummy token ids) -- the draft
    model's own attention KV is content/position-addressed like the
    target model's, so redundant warmup writes are harmless, and there is
    no GDN-style non-idempotent recurrent state in this class's scope at
    all (point 1 above), so the correctness concern that motivated
    ``CapturedBatchDecodeGraph``'s state-neutral-capture fix does not even
    apply here -- reserved slots are used anyway, for consistency with the
    established pattern and so a caller can safely reuse the SAME
    physical reserved-slot range across both graph classes (each resets
    its own warmup slots right after its own ``capture()``)."""

    def __init__(
        self,
        runner: DirectModelRunner,
        batch_size: int,
        qo_len: int = 1,
        warmup_slots: list[int] | None = None,
    ) -> None:
        if runner.mtp_model is None:
            raise RuntimeError("no MTP draft model loaded")
        if warmup_slots is not None:
            if len(warmup_slots) != batch_size:
                raise ValueError(
                    f"warmup_slots must have exactly batch_size ({batch_size}) entries"
                )
            self._external_warmup = True
        else:
            if runner.num_slots < 2 * batch_size:
                raise ValueError(
                    f"runner.num_slots={runner.num_slots} must be >= "
                    f"2*batch_size ({2 * batch_size})"
                )
            warmup_slots = list(range(runner.num_slots - batch_size, runner.num_slots))
            self._external_warmup = False
        self.runner = runner
        self.batch_size = batch_size
        self.qo_len = qo_len
        device = runner.device
        blocks_per_slot = runner.blocks_per_slot
        self.fixed_kv_split_size = runner.decode_fixed_kv_split_size
        self.fixed_max_num_splits = runner.decode_fixed_max_num_splits

        self._warmup_slots = warmup_slots

        n_tokens = batch_size * qo_len
        self.static_qo_indptr = (
            torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
        )
        self.static_kv_page_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        self.static_kv_page_indices = torch.zeros(
            batch_size * blocks_per_slot, dtype=torch.int32, device=device
        )
        self.static_kv_last_page_len = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.static_input_ids = torch.zeros(n_tokens, dtype=torch.long, device=device)
        self.static_positions = torch.zeros(n_tokens, dtype=torch.long, device=device)
        self.static_slot_mapping = torch.zeros(n_tokens, dtype=torch.long, device=device)
        cpu = torch.device("cpu")
        self._cpu_kv_page_indptr = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=cpu, pin_memory=True
        )
        self._cpu_kv_page_indices = torch.zeros(
            batch_size * blocks_per_slot, dtype=torch.int32, device=cpu, pin_memory=True
        )
        self._cpu_kv_last_page_len = torch.zeros(
            batch_size, dtype=torch.int32, device=cpu, pin_memory=True
        )
        self._cpu_input_ids = torch.zeros(n_tokens, dtype=torch.long, device=cpu, pin_memory=True)
        self._cpu_positions = torch.zeros(n_tokens, dtype=torch.long, device=cpu, pin_memory=True)
        self._cpu_slot_mapping = torch.zeros(
            n_tokens, dtype=torch.long, device=cpu, pin_memory=True
        )
        self._np_kv_page_indptr = self._cpu_kv_page_indptr.numpy()
        self._np_kv_page_indices = self._cpu_kv_page_indices.numpy()
        self._np_kv_last_page_len = self._cpu_kv_last_page_len.numpy()
        self._np_input_ids = self._cpu_input_ids.numpy()
        self._np_positions = self._cpu_positions.numpy()
        self._np_slot_mapping = self._cpu_slot_mapping.numpy()
        self.hidden_size = runner.vllm_config.model_config.get_hidden_size()
        # Dtype/device matched lazily on first real hidden_states_in at
        # capture time (mirrors how the class discovers its own model's
        # activation dtype rather than assuming one).
        self.static_hidden_states_in: torch.Tensor | None = None

        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_logits: torch.Tensor | None = None
        self._static_hidden_states_out: torch.Tensor | None = None
        # Test-observability only -- see CapturedBatchDecodeGraph's
        # identical field for the rationale.
        self.replay_count = 0
        self._last_slot_ids: list[int] | None = None
        self._last_kv_lengths: list[int] | None = None

    def _fill_buffers(
        self, slot_ids: list[int], token_ids, hidden_states_in: torch.Tensor, kv_lengths: list[int]
    ) -> None:
        runner = self.runner
        block_size = runner.block_size
        blocks_per_slot = runner.blocks_per_slot
        qo_len = self.qo_len

        if qo_len == 1:
            flat_token_ids = token_ids
        else:
            flat_token_ids = [tok for slot_tokens in token_ids for tok in slot_tokens]

        new_kv_lens = [kv_len + qo_len for kv_len in kv_lengths]
        num_pages_per_req = [(kv_len + block_size - 1) // block_size for kv_len in new_kv_lens]
        for slot, kv_len, num_pages in zip(slot_ids, new_kv_lens, num_pages_per_req):
            if num_pages > blocks_per_slot:
                raise RuntimeError(
                    f"slot {slot} kv_len {kv_len} exceeds this slot's "
                    f"{blocks_per_slot * block_size}-token capacity"
                )

        # P1 (notes/prefix-cache-design.md sec 5): same reasoning as
        # CapturedBatchDecodeGraph._fill_buffers -- grow before reading.
        if runner.enable_block_table:
            for slot, kv_len in zip(slot_ids, new_kv_lens):
                runner._ensure_blocks(slot, kv_len)

        kv_page_indptr_list = [0]
        for num_pages in num_pages_per_req:
            kv_page_indptr_list.append(kv_page_indptr_list[-1] + num_pages)
        page_indices_list: list[int] = []
        for slot, num_pages in zip(slot_ids, num_pages_per_req):
            if runner.enable_block_table:
                page_indices_list.extend(runner.block_table[slot][:num_pages])
            else:
                first_block = _physical_slot(slot) * blocks_per_slot
                page_indices_list.extend(range(first_block, first_block + num_pages))
        last_page_len_list = [
            kv_len - (num_pages - 1) * block_size
            for kv_len, num_pages in zip(new_kv_lens, num_pages_per_req)
        ]
        positions_list = [kv_len + j for kv_len in kv_lengths for j in range(qo_len)]
        slot_mapping_list = []
        for slot, kv_len in zip(slot_ids, kv_lengths):
            table = runner.block_table[slot] if runner.enable_block_table else None
            first_block = _physical_slot(slot) * blocks_per_slot
            for j in range(qo_len):
                pos = kv_len + j
                block_id = (
                    table[pos // block_size]
                    if table is not None
                    else first_block + pos // block_size
                )
                offset = pos % block_size
                slot_mapping_list.append(block_id * block_size + offset)

        n_indptr = len(kv_page_indptr_list)
        self._np_kv_page_indptr[:n_indptr] = kv_page_indptr_list
        self.static_kv_page_indptr.copy_(self._cpu_kv_page_indptr, non_blocking=True)
        self.static_kv_page_indices.zero_()
        n_pages = len(page_indices_list)
        if n_pages:
            self._np_kv_page_indices[:n_pages] = page_indices_list
            self.static_kv_page_indices[:n_pages].copy_(
                self._cpu_kv_page_indices[:n_pages], non_blocking=True
            )
        self._np_kv_last_page_len[: len(last_page_len_list)] = last_page_len_list
        self.static_kv_last_page_len.copy_(self._cpu_kv_last_page_len, non_blocking=True)
        self._np_input_ids[: len(flat_token_ids)] = flat_token_ids
        self.static_input_ids.copy_(self._cpu_input_ids, non_blocking=True)
        self._np_positions[: len(positions_list)] = positions_list
        self.static_positions.copy_(self._cpu_positions, non_blocking=True)
        self._np_slot_mapping[: len(slot_mapping_list)] = slot_mapping_list
        self.static_slot_mapping.copy_(self._cpu_slot_mapping, non_blocking=True)
        if self.static_hidden_states_in is None:
            self.static_hidden_states_in = torch.zeros_like(hidden_states_in)
        self.static_hidden_states_in.copy_(hidden_states_in)

    def _static_metadata_dict(self) -> dict:
        runner = self.runner
        attn_meta = SM120GQAMetadata(
            num_actual_tokens=self.batch_size * self.qo_len,
            num_reqs=self.batch_size,
            qo_indptr=self.static_qo_indptr,
            kv_page_indptr=self.static_kv_page_indptr,
            kv_page_indices=self.static_kv_page_indices,
            kv_last_page_len=self.static_kv_last_page_len,
            page_size=runner.block_size,
            is_pure_decode=(self.qo_len == 1),
            kv_split_size=self.fixed_kv_split_size,
            max_num_splits=self.fixed_max_num_splits,
            decode_qo_len=self.qo_len,
        )
        return {name: attn_meta for name in runner.mtp_attn_layer_names}

    def _forward_no_sync(self) -> tuple[torch.Tensor, torch.Tensor]:
        runner = self.runner
        attn_metadata_dict = self._static_metadata_dict()
        slot_mapping_dict = {name: self.static_slot_mapping for name in runner.mtp_attn_layer_names}
        with set_forward_context(
            attn_metadata_dict, runner.vllm_config, slot_mapping=slot_mapping_dict
        ):
            hidden_states_out = runner.mtp_model.forward(
                self.static_input_ids, self.static_positions, self.static_hidden_states_in
            )
        logits = runner.mtp_model.compute_logits(hidden_states_out)
        return logits, hidden_states_out

    def capture(self) -> None:
        if self._graph is not None:
            raise RuntimeError("already captured")
        runner = self.runner
        warmup_slots = self._warmup_slots
        dummy_prompt = [0, 0, 0, 0, 0]
        dummy_hidden = torch.zeros(
            len(warmup_slots) * len(dummy_prompt),
            self.hidden_size,
            dtype=runner.vllm_config.model_config.dtype,
            device=runner.device,
        )
        # Establishes the DRAFT model's own attention KV history on these
        # slots directly (bypassing the target model entirely -- warmup
        # content is disposable, see class docstring) so a subsequent
        # qo_len-token "step" at position len(dummy_prompt) is a genuine,
        # valid continuation rather than an out-of-range access.
        runner._mtp_forward_batch(
            warmup_slots,
            [dummy_prompt] * len(warmup_slots),
            dummy_hidden,
            [0] * len(warmup_slots),
            [0] * len(warmup_slots),
            qo_len=len(dummy_prompt),
            is_decode=False,
        )
        warmup_kv_lengths = [len(dummy_prompt)] * len(warmup_slots)
        if self.qo_len == 1:
            warmup_token_ids = [0] * self.batch_size
        else:
            warmup_token_ids = [[0] * self.qo_len for _ in range(self.batch_size)]
        warmup_hidden = torch.zeros(
            self.batch_size * self.qo_len,
            self.hidden_size,
            dtype=dummy_hidden.dtype,
            device=runner.device,
        )
        self._fill_buffers(warmup_slots, warmup_token_ids, warmup_hidden, warmup_kv_lengths)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._forward_no_sync()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._static_logits, self._static_hidden_states_out = self._forward_no_sync()
        self._graph = g

        if self._external_warmup:
            for slot in self._warmup_slots:
                runner.reset_slot(slot)

    def replay(
        self, slot_ids: list[int], token_ids, hidden_states_in: torch.Tensor, kv_lengths: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay at REAL data. Returns ``(logits, hidden_states_out)`` --
        unlike ``CapturedBatchDecodeGraph.replay()``, ``return_hidden`` is
        not optional here since every real caller needs the hidden state
        to feed the NEXT step (or, for a step-0-only recompute-group
        replay, to hand back to the caller). No runner bookkeeping is
        touched (see class docstring) -- purely a stateless tensor-in/
        tensor-out call, same as the eager ``_mtp_forward_batch`` it
        replaces. ``token_ids`` follows ``_mtp_forward_batch``'s own
        convention: flat (one id per slot) when ``qo_len==1``, else a
        list of per-slot ``qo_len``-length lists."""
        if not self._external_warmup and set(slot_ids) & set(self._warmup_slots):
            raise RuntimeError(
                f"slot(s) {set(slot_ids) & set(self._warmup_slots)} are this "
                "graph's own reserved warmup slots -- never replay() against them"
            )
        if self._graph is None:
            raise RuntimeError("capture() must be called first")
        if not (len(slot_ids) == self.batch_size == len(token_ids) == len(kv_lengths)):
            raise ValueError("slot_ids/token_ids/kv_lengths must match batch_size")
        self._fill_buffers(slot_ids, token_ids, hidden_states_in, kv_lengths)
        self._graph.replay()
        self.replay_count += 1
        self._last_slot_ids = slot_ids
        self._last_kv_lengths = kv_lengths
        return self._static_logits, self._static_hidden_states_out

    def replay_incremental(
        self, slot_ids: list[int], token_ids, hidden_states_in: torch.Tensor, kv_lengths: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optimized replay for draft continuation steps where KV length
        increased by 1 from the previous replay. Skips rebuilding the
        expensive kv_page_indices array when no slot crossed a page boundary."""
        if self._graph is None:
            raise RuntimeError("capture() must be called first")
        runner = self.runner
        block_size = runner.block_size
        can_skip_pages = (
            self._last_slot_ids is not None
            and self._last_slot_ids == slot_ids
            and self._last_kv_lengths is not None
            and len(self._last_kv_lengths) == len(kv_lengths)
            and all(
                new - old == self.qo_len
                and (old + self.qo_len) // block_size == (new + self.qo_len) // block_size
                for old, new in zip(self._last_kv_lengths, kv_lengths)
            )
        )
        if can_skip_pages:
            self._fill_buffers_incremental(slot_ids, token_ids, hidden_states_in, kv_lengths)
        else:
            self._fill_buffers(slot_ids, token_ids, hidden_states_in, kv_lengths)
        self._graph.replay()
        self.replay_count += 1
        self._last_slot_ids = slot_ids
        self._last_kv_lengths = kv_lengths
        return self._static_logits, self._static_hidden_states_out

    def _fill_buffers_incremental(
        self, slot_ids: list[int], token_ids, hidden_states_in: torch.Tensor, kv_lengths: list[int]
    ) -> None:
        """Fast path: only update last_page_len, positions, slot_mapping,
        input_ids, hidden_states. Skip kv_page_indptr and kv_page_indices."""
        runner = self.runner
        block_size = runner.block_size
        qo_len = self.qo_len
        new_kv_lens = [kv_len + qo_len for kv_len in kv_lengths]
        num_pages_per_req = [(kv_len + block_size - 1) // block_size for kv_len in new_kv_lens]
        if runner.enable_block_table:
            for slot, kv_len in zip(slot_ids, new_kv_lens):
                runner._ensure_blocks(slot, kv_len)
        last_page_len_list = [
            kv_len - (num_pages - 1) * block_size
            for kv_len, num_pages in zip(new_kv_lens, num_pages_per_req)
        ]
        positions_list = [kv_len + j for kv_len in kv_lengths for j in range(qo_len)]
        slot_mapping_list = []
        for slot, kv_len in zip(slot_ids, kv_lengths):
            table = runner.block_table[slot] if runner.enable_block_table else None
            first_block = _physical_slot(slot) * runner.blocks_per_slot
            for j in range(qo_len):
                pos = kv_len + j
                block_id = (
                    table[pos // block_size]
                    if table is not None
                    else first_block + pos // block_size
                )
                slot_mapping_list.append(block_id * block_size + pos % block_size)
        if qo_len == 1:
            flat_token_ids = token_ids
        else:
            flat_token_ids = [tok for slot_tokens in token_ids for tok in slot_tokens]
        self._np_kv_last_page_len[: len(last_page_len_list)] = last_page_len_list
        self.static_kv_last_page_len.copy_(self._cpu_kv_last_page_len, non_blocking=True)
        self._np_input_ids[: len(flat_token_ids)] = flat_token_ids
        self.static_input_ids.copy_(self._cpu_input_ids, non_blocking=True)
        self._np_positions[: len(positions_list)] = positions_list
        self.static_positions.copy_(self._cpu_positions, non_blocking=True)
        self._np_slot_mapping[: len(slot_mapping_list)] = slot_mapping_list
        self.static_slot_mapping.copy_(self._cpu_slot_mapping, non_blocking=True)
        if self.static_hidden_states_in is None:
            self.static_hidden_states_in = torch.zeros_like(hidden_states_in)
        self.static_hidden_states_in.copy_(hidden_states_in)
