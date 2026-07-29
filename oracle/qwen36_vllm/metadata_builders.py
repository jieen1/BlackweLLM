"""B5 模块化：元数据构建域。

从 direct_model_runner.py 提取的 build_*_metadata* 函数。
纯移动不改逻辑（B5 parity 门禁）。
"""

from __future__ import annotations

import torch

from oracle.qwen36_vllm.attention_compat import (
    FLA_CHUNK_SIZE,
    GDNAttentionMetadata,
    SM120GQAMetadata,
    compute_causal_conv1d_metadata,
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from runtime.block_pool import _physical_slot, _ssm_spec_row


def build_attention_metadata(
    *,
    prior_kv_len: int,
    num_new_tokens: int,
    is_decode: bool,
    slot: int,
    block_size: int,
    blocks_per_slot: int,
    device: torch.device,
    block_table: list[int] | None = None,
) -> SM120GQAMetadata:
    """Hand-built SM120GQAMetadata for one request in one fixed slot.
    ``DirectModelRunner`` tracks ``prior_kv_len`` itself via
    ``self.slot_kv_len``.

    ``block_table`` (P0, 2026-07-19, ``notes/prefix-cache-design.md`` sec
    5): optional per-slot list of physical block ids, indexed by LOGICAL
    page position. ``None`` (the default) preserves the exact prior
    arange-based addressing byte-for-byte. ``DirectModelRunner`` passes its
    own ``self.block_table[slot]`` here only when constructed with
    ``enable_block_table=True``.
    """
    new_kv_len = prior_kv_len + num_new_tokens
    page_size = block_size
    first_block = _physical_slot(slot) * blocks_per_slot
    num_pages = (new_kv_len + page_size - 1) // page_size
    if num_pages > blocks_per_slot:
        raise RuntimeError(
            f"slot {slot} kv_len {new_kv_len} exceeds this slot's "
            f"{blocks_per_slot * page_size}-token capacity"
        )
    qo_indptr = torch.tensor([0, num_new_tokens], dtype=torch.int32, device=device)
    kv_page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    if block_table is not None:
        kv_page_indices = torch.tensor(block_table[:num_pages], dtype=torch.int32, device=device)
    else:
        kv_page_indices = torch.arange(
            first_block, first_block + num_pages, dtype=torch.int32, device=device
        )
    last_page_len = new_kv_len - (num_pages - 1) * page_size
    kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)
    return SM120GQAMetadata(
        num_actual_tokens=num_new_tokens,
        num_reqs=1,
        qo_indptr=qo_indptr,
        kv_page_indptr=kv_page_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_len=kv_last_page_len,
        page_size=page_size,
        is_pure_decode=is_decode and num_new_tokens == 1,
        kv_split_size=max(new_kv_len, 1),
        max_num_splits=1,
        decode_qo_len=num_new_tokens if is_decode else 0,
    )


def build_gdn_metadata(
    *,
    slot_initialized: bool,
    num_new_tokens: int,
    is_decode: bool,
    slot: int,
    device: torch.device,
) -> GDNAttentionMetadata:
    """Hand-built GDNAttentionMetadata for one request in one fixed slot --
    see ``build_attention_metadata``'s docstring for why this is a shared
    function, not a second copy, between ``DirectModelRunner`` and Stage C.
    """
    state_indices = torch.tensor([_physical_slot(slot)], dtype=torch.int32, device=device)
    if is_decode:
        assert num_new_tokens == 1
        non_spec_qsl = torch.tensor([0, 1], dtype=torch.int32, device=device)
        return GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=1,
            num_decode_tokens=1,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=1,
            non_spec_query_start_loc=non_spec_qsl,
            non_spec_state_indices_tensor=state_indices,
        )

    query_start_loc = torch.tensor([0, num_new_tokens], dtype=torch.int32, device=device)
    query_start_loc_cpu = query_start_loc.cpu()
    has_initial_state = torch.tensor([slot_initialized], dtype=torch.bool, device=device)
    chunk_indices = prepare_chunk_indices(query_start_loc, FLA_CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(query_start_loc, FLA_CHUNK_SIZE)
    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        query_start_loc_cpu, device=device
    )
    return GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=num_new_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_new_tokens,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=query_start_loc,
        non_spec_state_indices_tensor=state_indices,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        prefill_query_start_loc=query_start_loc,
        prefill_state_indices=state_indices,
        prefill_has_initial_state=has_initial_state,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )


_MAX_DECODE_QO_LEN = 16
# Matches the real SM120GQAMetadataBuilder's own _MAX_DECODE_QO_LEN
# (vllm/v1/attention/backends/sm120_gqa.py) -- the decode/verify-shaped
# fast kernel's tested qo_len upper bound. Every real call in this project
# stays well under this (K+1 <= 4), so this cap was previously a latent,
# never-exercised gap (the old unconditional ``decode_qo_len = qo_len``
# formula had no cap at all) -- added here as part of the 2026-07-17
# ragged-qo_len generalization below, for faithfulness to the real
# formula, not because it currently changes any real call's outcome.

_DEFAULT_PREFILL_CHUNK_SIZE = 8192
# The suggested value for ``mtp_prefill_batch``'s ``chunk_size`` parameter
# (2026-07-19, chunked-prefill round) -- matches native vLLM's own
# ``--max-num-batched-tokens=8192`` default (``sm120-flash-attention/
# vllm_integration/launch_test_server.py``), so a chunked run and native's
# own chunked-prefill scheduler bound a single forward call's token count
# identically. Not itself used as ``mtp_prefill_batch``'s default (that
# stays ``None`` = unchunked, for full backward compatibility) -- callers
# that want chunking pass this explicitly.


def build_attention_metadata_batch(
    *,
    slots: list[int],
    prior_kv_lens: list[int],
    block_size: int,
    blocks_per_slot: int,
    device: torch.device,
    qo_len: int | list[int] = 1,
    is_decode: bool = True,
    fixed_kv_split_size: int | None = None,
    fixed_max_num_splits: int | None = None,
    block_tables: list[list[int]] | None = None,
) -> SM120GQAMetadata:
    """Hand-built SM120GQAMetadata for a real batch of requests spanning
    multiple fixed physical slots in a SINGLE metadata object, each
    contributing the SAME ``qo_len`` new query tokens this step (uniform
    across the batch -- the normal production case, since
    ``num_speculative_tokens`` is a global engine config, not
    per-request). ``qo_len=1`` (the default) is the batched analogue of
    ``build_attention_metadata``'s ``is_decode`` case (2026-07-16, verified
    through the full batch=1/2/3/4/varlen/reuse/continuous-generation
    ladder). ``qo_len>1`` is MTP/speculative-decode verify (K draft tokens
    + 1 bonus token per request, e.g. qo_len=4 for K=3) -- the real
    ``SM120GQAImpl.forward()`` dispatches this to
    ``flash_attn_sm120_fwd_v2_decode_fp8kv_paged`` (already
    production-hardened for qo_len in 2..4) whenever
    ``attn_metadata.decode_qo_len`` is in that range; this function only
    needs to construct the metadata correctly, not touch the kernel.

    Field construction (CSR qo_indptr/kv_page_indptr/kv_page_indices/
    kv_last_page_len, is_pure_decode, decode_qo_len) matches the real
    ``SM120GQAMetadataBuilder.build()`` (vllm/v1/attention/backends/
    sm120_gqa.py) generalized from vLLM's dense block-table -> CSR
    conversion to this project's fixed-slot addressing. At ``qo_len=1``
    every formula below reduces exactly to the previously-verified
    qo_len=1-only formulas (same numeric values, same tensors) -- this is
    a generalization, not a parallel implementation, so the existing
    ladder's results remain valid evidence for the qo_len=1 case.

    Kept as a SEPARATE function from ``build_attention_metadata`` (not a
    generalization of it) so this new batch path cannot regress the
    already-verified single-request path (2026-07-16 slot-0-reservation
    fix, Stage C/D 20/20) -- the two are cross-checked instead via the
    batch=1 equivalence test in ``benchmarks/batch_decode_regression.py``.

    ``is_decode`` (2026-07-17 addition, default ``True`` preserving
    ``decode_batch``/``verify_batch``'s existing behavior exactly): gates
    ``decode_qo_len``/``is_pure_decode`` exactly like
    ``build_attention_metadata``'s own ``is_decode`` parameter already
    does (``decode_qo_len = qo_len if is_decode else 0``,
    ``is_pure_decode = is_decode and qo_len == 1``) -- a real, pre-existing
    gap this function had before this fix: it used to set
    ``decode_qo_len = qo_len`` UNCONDITIONALLY, correct for every call site
    that existed before 2026-07-17 (``decode_batch``'s qo_len=1 decode and
    ``verify_batch``'s qo_len=k+1 MTP verify -- both genuinely
    decode/verify-shaped), but wrong for a genuine chunked/prefix PREFILL
    forward (e.g. ``mtp_prefill_batch``'s target-model call, or
    ``_mtp_sync_and_propose_batch``'s draft-model step-0 sync call when
    ``num_new_tokens > 1``): telling the kernel ``decode_qo_len=N`` for an
    N-token PREFILL falsely routes it through the decode/verify-shaped
    kernel path (confirmed against the real, authoritative
    ``SM120GQAMetadataBuilder.build()`` in ``vllm/v1/attention/backends/
    sm120_gqa.py``, whose own formula is
    ``decode_qo_len = cm.max_query_len if (is_uniform_qo_len and
    cm.max_query_len <= _MAX_DECODE_QO_LEN) else 0`` -- i.e. it is NEVER
    unconditional either). Caught via a real numerical-twin divergence
    (``benchmarks/mtp_batch_divergence_diag.py``): a batch=1 call through
    the new ``mtp_prefill_batch``/``_mtp_forward_batch`` path diverged
    from the long-verified single-slot ``mtp_prefill``/``_mtp_forward``
    path even at batch size 1, which is what proved this was a genuine
    formula gap and not a batch-size-dependent kernel numerics difference
    (the other candidate explanation this project's own established
    near-tie precedent would have made plausible).

    ``fixed_kv_split_size``/``fixed_max_num_splits`` (both None by
    default, preserving the existing per-call-derived behavior for the
    eager decode_batch/verify_batch paths): REQUIRED for CUDA-graph
    capture. Per this project's own read of
    ``vllm/v1/attention/backends/sm120_gqa.py``'s documented history (a
    real, previously-hit illegal-memory-access crash): a captured kernel
    launch's scalar arguments (kv_split_size/max_num_splits are plain
    Python ints, not device tensors) freeze to whatever value was live at
    capture time. If kv_split_size were still derived per-call from live
    ``new_kv_lens`` (as the default path does), replaying the SAME
    captured launch at a LARGER kv_len than capture time would silently
    use a stale, too-small split boundary -- the real backend's own fix
    (and this project's, when these are supplied) is to derive
    kv_split_size ONCE from a build-time-fixed upper bound L (this
    project's own ``blocks_per_slot * block_size``, i.e. the per-slot
    page-table limit THIS RUNTIME'S CALLER configured when constructing
    ``DirectModelRunner`` -- a software-chosen ceiling, not a GPU hardware
    limit; already enforced by the RuntimeError check below) via
    ``kv_split_size = ceil(L / target_splits)``,
    ``max_num_splits = target_splits``. Proof this stays correct for
    EVERY real kv_len from 1 up to L, not just the capture-time value:
    for split_size s = ceil(L/target_splits) and any real kv_len k <= L,
    num_splits(k) = ceil(k/s) <= ceil(L/s) <= target_splits (s >= L/target_splits
    by construction) -- so a single fixed pair is a valid upper bound for
    the entire decode lifetime of any request in this slot.

    ``qo_len`` as a RAGGED per-request list (2026-07-17 generalization,
    for the recompute-fallback batching round): previously a single
    scalar shared by the whole batch (the uniform case, e.g. verify's
    K+1 or a single decode token). A ``list[int]`` (one value per slot)
    is now also accepted -- this is what lets
    ``mtp_verify_and_commit_batch``'s recompute-fallback group (each
    slot needing a DIFFERENT number of real committed tokens replayed)
    batch into ONE call instead of one single-slot call per affected
    slot. A scalar ``qo_len`` is treated as a uniform list (broadcast to
    every slot) -- this is a strict generalization, not a parallel code
    path: every existing scalar call site produces byte-identical
    tensors to before.

    ``decode_qo_len``/``is_pure_decode`` now match the real
    ``SM120GQAMetadataBuilder.build()`` formula exactly, including its
    non-uniform-batch behavior: ``decode_qo_len = max(qo_lens) if
    (is_decode and is_uniform and max(qo_lens) <= _MAX_DECODE_QO_LEN)
    else 0``. A RAGGED (non-uniform) qo_lens list therefore ALWAYS gets
    ``decode_qo_len=0`` -- this deliberately routes ragged batches
    through the SAME general/chunked-prefix attention kernel
    (``flash_attn_sm120_fp8_kv_paged`` et al.) that this project's own
    genuine multi-token PREFILL calls already use, NOT a new kernel path
    invented for this round: that kernel is already documented (source
    comment in ``vllm/v1/attention/backends/sm120_gqa.py``) as "correct
    for pure prefill, chunked-prefill continuation, and ARBITRARY MIXED
    prefill+decode batches" -- i.e. real, ragged, per-request-varying
    query lengths within one batched call are exactly its designed use
    case, not new territory for the kernel itself, only for how THIS
    project's Python-side metadata construction reaches it.

    ``block_tables`` (P0, 2026-07-19, ``notes/prefix-cache-design.md`` sec
    5): optional list of per-slot physical block-id lists, ONE per entry in
    ``slots`` (same order), each indexed by LOGICAL page position. ``None``
    (the default) preserves the exact prior arange-based addressing
    byte-for-byte. Passed by ``DirectModelRunner`` only when constructed
    with ``enable_block_table=True``.
    """
    num_reqs = len(slots)
    if len(prior_kv_lens) != num_reqs:
        raise ValueError("slots and prior_kv_lens must have equal length")
    qo_lens = [qo_len] * num_reqs if isinstance(qo_len, int) else list(qo_len)
    if len(qo_lens) != num_reqs:
        raise ValueError("qo_len list must have exactly one entry per slot")
    is_uniform = len(set(qo_lens)) <= 1

    page_size = block_size
    new_kv_lens = [kv_len + qo for kv_len, qo in zip(prior_kv_lens, qo_lens)]
    num_pages_per_req = [(kv_len + page_size - 1) // page_size for kv_len in new_kv_lens]
    for slot, kv_len, num_pages in zip(slots, new_kv_lens, num_pages_per_req):
        if num_pages > blocks_per_slot:
            raise RuntimeError(
                f"slot {slot} kv_len {kv_len} exceeds this slot's "
                f"{blocks_per_slot * page_size}-token capacity"
            )

    qo_indptr_list = [0]
    for qo in qo_lens:
        qo_indptr_list.append(qo_indptr_list[-1] + qo)
    qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device=device)

    kv_page_indptr_list = [0]
    for num_pages in num_pages_per_req:
        kv_page_indptr_list.append(kv_page_indptr_list[-1] + num_pages)
    kv_page_indptr = torch.tensor(kv_page_indptr_list, dtype=torch.int32, device=device)

    if block_tables is not None:
        if len(block_tables) != num_reqs:
            raise ValueError("block_tables must have exactly one entry per slot")
        page_index_chunks = [
            torch.tensor(table[:num_pages], dtype=torch.int32, device=device)
            for table, num_pages in zip(block_tables, num_pages_per_req)
        ]
    else:
        page_index_chunks = [
            torch.arange(
                _physical_slot(slot) * blocks_per_slot,
                _physical_slot(slot) * blocks_per_slot + num_pages,
                dtype=torch.int32,
                device=device,
            )
            for slot, num_pages in zip(slots, num_pages_per_req)
        ]
    kv_page_indices = (
        torch.cat(page_index_chunks)
        if page_index_chunks
        else torch.empty(0, dtype=torch.int32, device=device)
    )
    kv_last_page_len = torch.tensor(
        [
            kv_len - (num_pages - 1) * page_size
            for kv_len, num_pages in zip(new_kv_lens, num_pages_per_req)
        ],
        dtype=torch.int32,
        device=device,
    )
    if fixed_kv_split_size is not None:
        # CUDA-graph-safe path: fixed once from a build-time bound, never
        # from this call's live data -- see the docstring's proof.
        kv_split_size = fixed_kv_split_size
        max_num_splits = fixed_max_num_splits if fixed_max_num_splits is not None else 1
    else:
        # Conservative, correctness-first choice matching the single-request
        # function: kv_split_size >= every request's own new_kv_len forces
        # num_splits == 1 for all of them (no cross-request split-size tuning
        # yet -- a performance follow-on, not a correctness concern, exactly
        # the same tradeoff build_attention_metadata already makes). NOT
        # CUDA-graph-safe (see docstring) -- only used by the eager
        # decode_batch/verify_batch paths.
        kv_split_size = max(max(new_kv_lens, default=1), 1)
        max_num_splits = 1

    max_qo_len = max(qo_lens) if qo_lens else 0
    decode_qo_len = (
        max_qo_len if (is_decode and is_uniform and max_qo_len <= _MAX_DECODE_QO_LEN) else 0
    )
    return SM120GQAMetadata(
        num_actual_tokens=sum(qo_lens),
        num_reqs=num_reqs,
        qo_indptr=qo_indptr,
        kv_page_indptr=kv_page_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_len=kv_last_page_len,
        page_size=page_size,
        is_pure_decode=(is_decode and max_qo_len == 1),
        kv_split_size=kv_split_size,
        max_num_splits=max_num_splits,
        decode_qo_len=decode_qo_len,
    )


def build_gdn_metadata_batch(
    *,
    slots: list[int],
    device: torch.device,
    qo_len: int | list[int] = 1,
    slot_initialized: list[bool] | None = None,
) -> GDNAttentionMetadata:
    """Hand-built GDNAttentionMetadata for a real batch of requests, each
    contributing the SAME ``qo_len`` new query tokens this step (see
    ``build_attention_metadata_batch``'s docstring for the uniform-qo_len
    scope rationale shared by both functions).

    ``qo_len=1`` (the default) is the batched analogue of
    ``build_gdn_metadata``'s ``is_decode`` case (2026-07-16, verified
    through the full batch ladder) -- matches the real
    ``GDNAttentionMetadataBuilder.build()``'s pure non-spec-decode branch
    (gdn_attn.py): only num_decodes/num_decode_tokens/
    non_spec_query_start_loc/non_spec_state_indices_tensor are populated,
    everything else stays None. At qo_len=1 the formulas below reduce
    exactly to the previously-verified values -- a generalization, not a
    parallel implementation.

    **2026-07-17 real bug, found via a batch=1 forced-reject equivalence
    test** (``benchmarks/mtp_ragged_recompute_verify_check.py``): an
    earlier version of this fast path was gated on
    ``isinstance(qo_len, int) and qo_len == 1`` -- i.e. a UNIFORM list
    where every entry happens to be 1 (exactly what the new
    ragged-recompute-batching caller always constructs, even for a
    single recompute slot with ``committed_len == 1``, since it always
    passes a list) fell through to the chunked/general branch instead,
    unlike a bare scalar ``1``. That asymmetry is NOT what
    ``build_attention_metadata_batch``'s own analogous ``decode_qo_len``
    logic does (it already treats a uniform list and a scalar
    identically, via ``is_uniform``/``max_qo_len``) -- this function's
    old condition was a real, narrower special-case that diverged from
    the sibling function's already-correct generalization, and the
    chunked/general GDN path is NOT a drop-in numerically-equivalent
    substitute for the fast single-token decode path (confirmed by the
    test finding real committed-content divergence between the two,
    not just a near-tie). Fixed below by making the fast-path condition
    VALUE-based (uniform, all entries equal to 1) instead of
    TYPE-based (bare scalar only) -- this treats scalar ``1`` and a
    uniform ``[1, 1, ...]`` list identically, exactly mirroring
    ``build_attention_metadata_batch``'s own already-correct pattern.

    ``qo_len>1`` (or a ragged per-request list, 2026-07-17 generalization
    -- see below) is MTP/speculative-decode verify. Rather than
    replicating the real builder's much more involved ``spec_decode``
    branch (accept/reject bookkeeping, sorting spec vs non-spec tokens --
    explicitly out of scope this round, see notes/direct-model-runner-
    design.md), this generalizes ``build_gdn_metadata``'s OTHER existing
    branch instead: the ``is_decode=False`` ("prefill"/chunked) case,
    which the real builder's own ``split_decodes_and_prefills`` would also
    select for any request with query_len>1 when no draft-acceptance info
    is supplied -- i.e. this treats an MTP verify step as an ordinary
    chunked continuation of ``qo_len`` new tokens per request, which is
    numerically correct (the chunked FLA kernel handles arbitrary query
    length, GDN state update included) even though it foregoes the
    real builder's spec-decode-specific optimizations.

    ``qo_len`` as a RAGGED per-request list (2026-07-17, for the
    recompute-fallback batching round -- mirrors
    ``build_attention_metadata_batch``'s identical generalization):
    ``query_start_loc`` is built from a per-request cumulative sum
    instead of ``arange(n+1) * qo_len`` -- a strict generalization (a
    scalar ``qo_len`` broadcasts to a uniform list, reducing to the exact
    same tensor as before). Crucially, this is NOT the same padding
    concern flagged when this generalization was first scoped: a
    genuinely ragged CSR construction feeds EVERY request EXACTLY its own
    real token count into the chunked FLA kernel (``prepare_chunk_indices``/
    ``prepare_chunk_offsets``/``compute_causal_conv1d_metadata`` are all
    already CSR/``cu_seqlens``-generic -- this project's own prior
    uniform-only usage was a special case of what these functions already
    support, not a hand-restriction on them). There is no padding token
    ever fed to any request's GDN state under this design, so the
    recurrent-state-corruption concern that made a padding-based ragged
    batch design hard does not apply here -- it only would have applied
    to a design that forced a shared qo_len via padding, which this is
    deliberately NOT doing.
    """
    num_reqs = len(slots)
    state_indices = torch.tensor(
        [_physical_slot(slot) for slot in slots], dtype=torch.int32, device=device
    )
    qo_lens = [qo_len] * num_reqs if isinstance(qo_len, int) else list(qo_len)
    if len(qo_lens) != num_reqs:
        raise ValueError("qo_len list must have exactly one entry per slot")
    if num_reqs > 0 and all(qo == 1 for qo in qo_lens):
        non_spec_qsl = torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device)
        return GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=num_reqs,
            num_decode_tokens=num_reqs,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=num_reqs,
            non_spec_query_start_loc=non_spec_qsl,
            non_spec_state_indices_tensor=state_indices,
        )

    if slot_initialized is None or len(slot_initialized) != num_reqs:
        raise ValueError("slot_initialized (one bool per slot) is required when qo_len != 1")
    qsl_list = [0]
    for qo in qo_lens:
        qsl_list.append(qsl_list[-1] + qo)
    query_start_loc = torch.tensor(qsl_list, dtype=torch.int32, device=device)
    query_start_loc_cpu = query_start_loc.cpu()
    has_initial_state = torch.tensor(slot_initialized, dtype=torch.bool, device=device)
    chunk_indices = prepare_chunk_indices(query_start_loc, FLA_CHUNK_SIZE)
    chunk_offsets = prepare_chunk_offsets(query_start_loc, FLA_CHUNK_SIZE)
    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        query_start_loc_cpu, device=device
    )
    num_actual_tokens = sum(qo_lens)
    return GDNAttentionMetadata(
        num_prefills=num_reqs,
        num_prefill_tokens=num_actual_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_actual_tokens,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=query_start_loc,
        non_spec_state_indices_tensor=state_indices,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        prefill_query_start_loc=query_start_loc,
        prefill_state_indices=state_indices,
        prefill_has_initial_state=has_initial_state,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )


def build_gdn_metadata_spec_batch(
    *,
    slots: list[int],
    device: torch.device,
    qo_len: int,
    num_accepted_tokens_prev: list[int],
    total_physical_slots: int,
    num_spec: int,
) -> GDNAttentionMetadata:
    """Real spec-decode GDN metadata (Phase 2, 2026-07-18) -- the
    ``num_prefills=0, num_decodes=0, num_spec_decodes=len(slots)`` branch
    of the real ``GDNAttentionMetadataBuilder.build()`` (``gdn_attn.py``),
    hand-built for our fixed-slot runtime instead of vLLM's paged block
    table. Replaces ``build_gdn_metadata_batch``'s chunked/prefill-shaped
    treatment of an MTP verify step -- originally for
    ``mtp_verify_and_commit_batch`` only (``mtp_verify_and_commit``, the
    singular/looped sibling, kept using the chunked path +
    snapshot/restore/recompute -- an intentional, documented divergence,
    see notes doc section 10/11), then (Phase B, same day) for
    ``mtp_verify_and_commit`` too, called at ``len(slots)==1``: both
    production verify paths now share this mechanism.

    ``qo_len`` here is always ``num_spec + 1`` (the K draft continuations
    + 1 bonus/anchor position) -- unlike ``build_gdn_metadata_batch``,
    this is NOT generalized to a ragged per-request list: every slot in
    one spec-decode verify call always submits the SAME K+1-token draft
    (a global engine config), matching ``verify_batch``'s own existing
    uniform-qo_len contract.

    ``spec_state_indices_tensor[i, col] = _ssm_spec_row(slots[i], col,
    total_physical_slots, num_spec)`` -- a FIXED per-(slot, column)
    physical-row mapping, unchanging round to round (see
    ``_ssm_spec_row``'s docstring for why this matches the real kernel's
    own block-table-derived addressing). ``num_accepted_tokens_prev[i]``
    is slot ``slots[i]``'s real committed length from its OWN previous
    verify round (or exactly ``1`` for a slot's first-ever verify right
    after a real prefill -- the bootstrap case, selecting column 0, the
    row the chunked prefill forward itself wrote into).

    ``spec_token_indx``/``non_spec_token_indx``/``has_initial_state``
    are deliberately left ``None`` -- confirmed by direct reading of
    ``qwen_gdn_linear_attn.py``'s ``_forward_core`` that the
    ``num_prefills=0 and num_decodes=0`` branch takes ``mixed_qkv_spec =
    mixed_qkv`` directly (no ``index_select``) and never reads
    ``has_initial_state`` at all in that branch (real drafts always
    resume from real prior state -- there is no "fresh" spec-decode
    request in this project's design, since every slot has already gone
    through a real ``mtp_prefill_batch`` before its first verify call)."""
    num_reqs = len(slots)
    if len(num_accepted_tokens_prev) != num_reqs:
        raise ValueError("num_accepted_tokens_prev must have exactly one entry per slot")
    spec_query_start_loc = torch.arange(0, num_reqs + 1, dtype=torch.int32, device=device) * qo_len
    state_indices_list = [
        [_ssm_spec_row(slot, col, total_physical_slots, num_spec) for col in range(qo_len)]
        for slot in slots
    ]
    spec_state_indices_tensor = torch.tensor(state_indices_list, dtype=torch.int32, device=device)
    spec_sequence_masks = torch.ones(num_reqs, dtype=torch.bool, device=device)
    num_accepted_tokens = torch.tensor(num_accepted_tokens_prev, dtype=torch.int32, device=device)
    num_actual_tokens = num_reqs * qo_len
    return GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=num_reqs,
        num_spec_decode_tokens=num_actual_tokens,
        num_actual_tokens=num_actual_tokens,
        spec_query_start_loc=spec_query_start_loc,
        spec_state_indices_tensor=spec_state_indices_tensor,
        spec_sequence_masks=spec_sequence_masks,
        num_accepted_tokens=num_accepted_tokens,
    )
