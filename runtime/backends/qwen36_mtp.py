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

The complete historical contract is stronger than a one-row handoff: MTP
teacher-forces every newly real target position into its own cache, then
generates only the remaining speculative continuation.  The target hidden
rows already exist in prefill/verify, so that restoration adds no backbone
forward; it merely keeps the MTP cache's real-prefix boundary in lockstep
with target KV and GDN state.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch

from runtime.backends.qwen36_mtp_cudagraph import (
    Qwen36MTPAnchorCudaGraph,
    Qwen36MTPBatchedSync,
    Qwen36MTPDraftCudaGraph,
    Qwen36MTPVerifyCudaGraph,
    attempt_mtp_cg_capture,
    build_pooled_mtp_caches,
)
from runtime.logprobs import compute_logprobs
from runtime.model.qwen36_model import GdnLayerState
from runtime.mtp_accept import (
    determine_accept_reject_batch,
    determine_accept_reject_from_predictions,
    sample_accept_reject,
)
from runtime.recurrent_state_pool import spec_row
from runtime.round_profile import round_profile
from runtime.sampling import SamplingParams, compute_sampling_distribution, make_generator

if TYPE_CHECKING:
    from runtime.backends.qwen36 import Qwen36Backend


class Qwen36MTPGDNRows:
    """Permanent ``K+1`` GDN candidate rows used by one MTP engine.

    The old verify path cloned every recurrent state once per speculative
    position and copied one clone back after accept/reject. A candidate state
    can instead live at its fixed ``spec_row`` address and the next round
    selects the accepted row. MTP configures the slot pool before any decode
    graph capture, so column zero is the ordinary live row in that same
    graph-safe allocation.

    The verify has ``k+1`` positions (anchor + K drafts), so it owns exactly
    ``k+1`` output rows: column ``p`` is the state after verify position
    ``p``.  The incoming prefill state already resides in ordinary live
    column zero, is selected before the forward, and is then safely
    overwritten by the anchor's candidate state.  After accepting ``m``
    drafts, column ``m`` is the state after the anchor plus its accepted
    prefix.  This is the historical
    ``num_accepted_tokens_prev - 1`` addressing, with no extra input row.
    """

    def __init__(self, backend: Qwen36Backend, num_speculative_tokens: int) -> None:
        self.backend = backend
        self.pool = backend.pool
        self.k = num_speculative_tokens
        # ``spec_row`` addresses column zero plus exactly ``num_spec``
        # dedicated candidate rows: K speculative drafts therefore means
        # K+1 rows for anchor + drafts.  Column zero is the slot pool's
        # ordinary live state, never a copied private bootstrap row.
        self.num_spec_cols = self.k
        self.total_physical_slots = self.pool.num_slots + 1
        self.total_rows = self.total_physical_slots * (self.num_spec_cols + 1)
        self.pool.enable_mtp_gdn_rows(self.k)
        self.conv_pools = self.pool.conv_pools
        self.recurrent_pools = self.pool.recurrent_pools
        self._states: dict[int, list[list[GdnLayerState]]] = {}

        for layer in self.pool.model.model.layers:
            if layer.layer_type != "linear_attention":
                continue
            per_layer = [
                self.pool.mtp_gdn_columns(layer.layer_idx, slot)
                for slot in range(self.total_physical_slots)
            ]
            self._states[layer.layer_idx] = per_layer

        if not self._states:
            raise ValueError("MTP GDN row allocation requires a linear-attention layer")

    def rows_for_slot(self, slot: int) -> dict[int, list[GdnLayerState]]:
        return {layer_idx: per_slot[slot] for layer_idx, per_slot in self._states.items()}

    def row_for_slot(self, slot: int, col: int) -> int:
        return spec_row(slot, col, self.total_physical_slots, self.num_spec_cols)

    def activate(self, slot: int, col: int) -> None:
        self.pool.activate_mtp_gdn_state(slot, col)

    def sync_from_live(self, slot: int) -> None:
        """Validate that prefill already wrote MTP's aliased column zero."""
        state = self.pool.slot_state(slot)
        for layer_idx, per_slot in self._states.items():
            live = state.gdn_states[layer_idx]
            if live is not per_slot[slot][0]:
                raise RuntimeError(
                    f"slot {slot} GDN layer {layer_idx} is not on MTP column zero after prefill"
                )

    def reset_slot(self, slot: int) -> None:
        for per_slot in self._states.values():
            for state in per_slot[slot]:
                state.conv_state.zero_()
                state.recurrent_state.zero_()
                state.has_previous_state = False
        self.activate(slot, 0)


class Qwen36MTPEngine:
    """Owns per-slot MTP draft-head state (its own small KV cache per slot)
    and drives one draft+verify+accept/reject+re-draft round, the way
    :class:`runtime.backends.laguna_dflash.DFlashEngine` drives DFlash for
    Laguna. Kept as a separate object rather than folded into
    :class:`runtime.backends.qwen36.Qwen36Backend` / :class:`runtime.model
    .qwen36_slots.Qwen36SlotPool` for the same reason DFlash is: this is
    opt-in, backend-private bookkeeping layered on top of the always-on
    slot pool, not a change to it.

    The hot path now matches the historical split: the target verify batches
    across active slots, and the autoregressive draft head batches each
    chained step across those same slots. Only sampled fallback retains the
    proven single-slot round path. GDN state uses the fixed ``spec_row``
    topology: column zero is the live state and columns one through K are
    candidate destinations. This removes the old per-round snapshot/restore
    copies while preserving the same accepted-prefix semantics as the
    historical vLLM runner.
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
        if enable_resync:
            raise ValueError(
                "QSR_SERVER_MTP_RESYNC is retired: MTP now always teacher-forces every "
                "newly committed target suffix, so the old interior-only repair is invalid."
            )
        # Kept as a read-only compatibility attribute for load-time logging.
        self.enable_resync = False
        self.device = backend.device
        self.dtype = backend.dtype
        self.vocab_size = int(model.config["vocab_size"])

        #: 2026-08-03 CUDA-Graph follow-up (see
        #: runtime.backends.qwen36_mtp_cudagraph's module docstring): CUDA
        #: Graph capture is only attempted on a real CUDA device with a
        #: real model -- a CPU/stub model (this repo's own
        #: tests/test_qwen36_mtp_engine.py) has no
        #: model.mtp.layers[0].self_attn geometry to build a pool from,
        #: and every stub test's ``model.mtp_new_cache`` fake would break
        #: under a pooled allocation it was never written to model. The
        #: eager per-slot cache (unchanged from before this follow-up)
        #: stays the CPU/stub/opt-out path; QSR_QWEN36_MTP_CUDA_GRAPH=0
        #: forces it even on a real GPU (diagnostic escape hatch, same
        #: convention as QSR_DFLASH_CUDA_GRAPH/QSR_VERIFY_CUDA_GRAPH).
        self._use_cuda_graph = (
            self.device.type == "cuda" and os.environ.get("QSR_QWEN36_MTP_CUDA_GRAPH", "1") != "0"
        )
        #: cg_status / scratch_row / mtp_k_pool / mtp_v_pool / mtp_page_size
        #: / mtp_pages_per_slot are only ever populated in the CUDA-graph
        #: branch below; they stay at these CPU/stub-safe defaults
        #: otherwise, and every graph-replay call site below is itself
        #: gated on ``self._anchor_cg``/``self._draft_cg`` being non-None,
        #: so nothing downstream needs to re-check ``self._use_cuda_graph``.
        self.cg_status: dict[str, str] = {}
        self.scratch_row: int | None = None
        self.mtp_k_pool: torch.Tensor | None = None
        self.mtp_v_pool: torch.Tensor | None = None
        self.mtp_page_size: int | None = None
        self.mtp_pages_per_slot: int | None = None
        self._spec_rows: Qwen36MTPGDNRows | None = None
        if any(
            layer.layer_type == "linear_attention" and hasattr(layer.linear_attn, "spec_forward")
            for layer in backend.pool.model.model.layers
        ):
            self._spec_rows = Qwen36MTPGDNRows(backend, self.k)
        self._spec_state_col = [0] * (backend.num_slots + 1)
        # MTP attention has two lengths.  ``cache.seq_len`` is the
        # teacher-forced real prefix plus the ``K-1`` continuation rows; the
        # first draft is predicted by the final teacher-forced row and does
        # not occupy a speculative cache row of its own. ``_sync_len`` is
        # the real-prefix boundary. The distinction is load-bearing: a
        # partial accept must overwrite, rather than reuse, draft-head rows
        # produced from speculative hidden states.
        self._sync_len = [0] * backend.num_slots
        # The backend's prefix cache is same-slot reuse: reset is pointer-only
        # and therefore leaves this one-layer MTP KV allocation intact too.
        # Retain only its authenticated real-prefix length; the backend will
        # call ``restore_prefix`` only after its own token+GDN checkpoint has
        # proven the same prefix identity.
        self._cached_prefix_sync_len = [0] * backend.num_slots
        self._anchor_cg: Qwen36MTPAnchorCudaGraph | None = None
        self._draft_cg: Qwen36MTPDraftCudaGraph | None = None
        self._batched_sync: Qwen36MTPBatchedSync | None = None
        self._verify_cg = None
        self._cg_captured = False
        #: Per-batch preallocated ``[B, K+1]`` verify-token buffers and
        #: pinned anchor staging.  The hot path used to build
        #: ``verify_tokens`` with ``torch.tensor``/``torch.cat`` every
        #: round; at ~94/96 GiB resident memory those tiny per-round
        #: allocations cost allocator round-trips that show up as host
        #: time in the replay fill (measured 2026-08-06: ~10 ms/round).
        #: Reusing the buffers keeps the batched device path
        #: allocation-free.
        self._verify_tokens_buf: dict[int, torch.Tensor] = {}
        self._anchor_host: dict[int, torch.Tensor] = {}
        self._anchor_host_views: dict[int, object] = {}
        for _batch in range(1, backend.num_slots + 1):
            self._verify_tokens_buf[_batch] = torch.zeros(
                _batch, self.k + 1, dtype=torch.long, device=self.device
            )
            _anchor = torch.zeros(_batch, 1, dtype=torch.long, device="cpu", pin_memory=True)
            self._anchor_host[_batch] = _anchor
            self._anchor_host_views[_batch] = _anchor.numpy()

        if self._use_cuda_graph:
            #: One persistent MTP self-attention cache per real slot, plus
            #: one scratch row -- POOLED (module docstring's "one
            #: allocation, two addressings", scaled down to MTP's 1-layer
            #: head), required so :class:`Qwen36MTPDraftCudaGraph` can
            #: serve every slot from a single captured graph (a CUDA Graph
            #: bakes in the exact address its kernels read; an
            #: independently-allocated per-slot cache could only ever be
            #: replayed correctly for the ONE slot it was captured
            #: against).
            #
            # Phase 2 dynamic arena: MTP shares the backbone's global
            # bundle pool and page table (plan §6.1 -- the bundle's MTP
            # K/V pages live at the same logical-to-physical mapping as
            # the backbone's), so backbone and MTP KV are lock-step by
            # construction.
            dynamic = getattr(backend.pool, "dynamic_arena", False)
            (
                self._caches,
                self.mtp_k_pool,
                self.mtp_v_pool,
                self.mtp_page_size,
                self.mtp_pages_per_slot,
            ) = build_pooled_mtp_caches(
                model,
                num_slots=backend.num_slots,
                device=self.device,
                dtype=self.dtype,
                pool_bundles=backend.pool.pool_bundles if dynamic else None,
                page_table=backend.pool._global_page_table if dynamic else None,  # noqa: SLF001
                extensible_buffers=(
                    backend.pool.extensible_buffers
                    if getattr(backend.pool, "extensible", False)
                    else None
                ),
            )
            if dynamic:
                # Register the pooled MTP KV as the 17th atomic COW family
                # (plan §4.8): MTP shares the backbone's bundle mapping, so a
                # partial-page COW detach must clone MTP bytes too, or the
                # MTP prefix half is lost while the backbone stays intact.
                backend.pool.register_mtp_kv(self.mtp_k_pool, self.mtp_v_pool)
            self.scratch_row = backend.num_slots
            # Historical `_mtp_sync_and_propose_batch` is also the B=1
            # production path.  Keep one static bucket for that shape so a
            # single active request does not fall back to eager
            # `model.mtp_forward` between the captured verify and draft
            # phases.
            self._batched_sync = Qwen36MTPBatchedSync(self)
        else:
            #: One persistent MTP self-attention cache per real slot (no
            #: scratch row -- nothing here is CUDA-graph-captured on this
            #: path, so there is no padding-row aliasing hazard to guard
            #: against). Allocated once, reset (not reallocated) by
            #: :meth:`reset_slot`, same discipline B0-5 established for
            #: the backbone's own recurrent-state buffers.
            self._caches = [
                model.mtp_new_cache(device=self.device, dtype=self.dtype)
                for _ in range(backend.num_slots)
            ]

        self.stats: dict[str, int] = {
            "rounds": 0,
            "sampled_rounds": 0,
            "verify_graph_replays": 0,
            "verify_graph_slots": 0,
            "batched_verify_replays": 0,
            "draft_graph_replays": 0,
            "draft_graph_slots": 0,
            "batched_draft_replays": 0,
            "batched_sync_replays": 0,
            "batched_sync_slots": 0,
        }

        if self._use_cuda_graph:
            self.capture_cuda_graphs()

    def _record_verify_graph_replay(self, batch_size: int) -> None:
        """Record actual replay use, separately from capture health.

        Capture status establishes eligibility only.  This counter is the
        runtime proof that a c>1 scheduler round reached M-2's fused body
        instead of a fallback, and is mirrored into backend stats because
        that is the existing ``/debug/stats`` observability surface.
        """
        self.stats["verify_graph_replays"] += 1
        self.stats["verify_graph_slots"] += batch_size
        self.backend.stats["mtp_verify_graph_replays"] += 1
        self.backend.stats["mtp_verify_graph_slots"] += batch_size
        if batch_size > 1:
            self.stats["batched_verify_replays"] += 1
            self.backend.stats["mtp_batched_verify_replays"] += 1

    def _record_draft_graph_replay(self, batch_size: int) -> None:
        """Record use of the B-wide chained draft graph, not just capture."""
        self.stats["draft_graph_replays"] += 1
        self.stats["draft_graph_slots"] += batch_size
        self.backend.stats["mtp_draft_graph_replays"] += 1
        self.backend.stats["mtp_draft_graph_slots"] += batch_size
        if batch_size > 1:
            self.stats["batched_draft_replays"] += 1
            self.backend.stats["mtp_batched_draft_replays"] += 1

    # -- CUDA Graph capture (2026-08-03 follow-up) -----------------------

    def capture_cuda_graphs(self) -> None:
        """Capture the draft-loop and batched-verify graphs, before any
        production slot is used -- same "capture while every slot is
        definitionally empty" window ``Qwen36Backend.enable_mtp``'s own
        docstring already documents, and the same one
        ``DFlashEngine.capture_cuda_graphs``/``_init_cuda_graph`` rely on.
        Idempotent; a no-op if ``self._use_cuda_graph`` is False (CPU/stub
        or QSR_QWEN36_MTP_CUDA_GRAPH=0) or capture already ran.

        Each site goes through :func:`attempt_mtp_cg_capture` -- status
        recorded in :attr:`cg_status`, degrading to that path's existing
        eager fallback (never crashing) unless QSR_QWEN36_MTP_REQUIRE_CG=1.
        See ``runtime.backends.qwen36_mtp_cudagraph``'s module docstring
        for why this defaults to "degrade", the opposite of DFlash's
        default.
        """
        if not self._use_cuda_graph or self._cg_captured:
            return
        strict = os.environ.get("QSR_QWEN36_MTP_REQUIRE_CG", "0") == "1"

        # The anchor graph is no longer captured: the round folded the anchor
        # into the verify forward (qo_len=k+1), so there is no separate
        # [1,1] anchor pass left for it to accelerate. Capturing it anyway
        # would cost capture time and a pool of graph-private buffers for a
        # graph nothing replays. Qwen36MTPAnchorCudaGraph itself is kept --
        # it is still the reference for how a single-token MTP step writes
        # the KV/GDN pools, and scripts still construct it directly.
        self.cg_status["anchor"] = "unused"

        def _do_capture_draft() -> None:
            cg = Qwen36MTPDraftCudaGraph(self)
            cg.capture()
            self._draft_cg = cg

        status = attempt_mtp_cg_capture("draft", _do_capture_draft, strict=strict)
        self.cg_status["draft"] = status
        if status == "failed":
            self._draft_cg = None

        if self._batched_sync is None:
            self.cg_status["sync"] = "unused"
        else:
            status = attempt_mtp_cg_capture("sync", self._batched_sync.capture, strict=strict)
            self.cg_status["sync"] = status

        def _do_capture_verify() -> None:
            if self._spec_rows is None:
                raise RuntimeError(
                    "verify CUDA Graph requires real Qwen3.6 GDN row-addressed state"
                )
            cg = Qwen36MTPVerifyCudaGraph(self)
            cg.capture()
            self._verify_cg = cg

        status = attempt_mtp_cg_capture("verify", _do_capture_verify, strict=strict)
        self.cg_status["verify"] = status
        if status == "failed":
            self._verify_cg = None

        self._cg_captured = True

    def cuda_graphs_healthy(self) -> bool:
        """True iff every MTP CUDA Graph capture attempted so far
        succeeded -- vacuously True before capture runs or when CUDA
        Graphs are disabled entirely, matching
        ``DFlashEngine.cuda_graphs_healthy``'s own contract."""
        # ``anchor`` is deliberately ``"unused"``: the anchor token is
        # folded into the K+1 verify body.  It is descriptive lifecycle
        # state, not a failed capture, so it must not make a healthy draft +
        # verify pair report unhealthy forever.
        return all(status == "captured" for status in self.cg_status.values() if status != "unused")

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
        self._sync_len[slot] = 0
        if self._spec_rows is not None:
            self._spec_rows.reset_slot(slot)
        self._spec_state_col[slot] = 0

    def preserve_prefix(self, slot: int, kv_len: int) -> None:
        """Remember the real MTP prefix retained in this slot's KV bytes."""
        if self._sync_len[slot] != kv_len:
            raise RuntimeError(
                f"MTP prefix for slot {slot} has sync_len={self._sync_len[slot]}, "
                f"but target KV has {kv_len} rows"
            )
        self._cached_prefix_sync_len[slot] = kv_len

    def can_restore_prefix(self, slot: int, kv_len: int) -> bool:
        return self._cached_prefix_sync_len[slot] >= kv_len

    def restore_prefix(self, slot: int, kv_len: int) -> None:
        if not self.can_restore_prefix(slot, kv_len):
            raise RuntimeError(f"slot {slot} has no retained MTP prefix of length {kv_len}")
        self._caches[slot].seq_len = kv_len
        self._sync_len[slot] = kv_len

    def restore_prefix_from_arena(self, slot: int, kv_len: int) -> bool:
        """Activate MTP rows retained in dynamic arena bundles.

        Dynamic Qwen allocates backbone and MTP KV as one lock-step bundle
        mapping.  ``Qwen36SlotPool.restore_prefix_from_arena`` has already
        revived the physical bundles and installed the target page-table
        row, so restoring MTP is metadata-only; copying would waste both
        bandwidth and a second full-context allocation.
        """
        if not self.backend.pool.dynamic_arena:
            return False
        pages = (kv_len + self.mtp_page_size - 1) // self.mtp_page_size
        row = self.backend.pool._page_table_host[slot]  # noqa: SLF001
        if kv_len <= 0 or any(page == 0 for page in row[:pages]):
            return False
        cache = self._caches[slot]
        cache.seq_len = kv_len
        self._sync_len[slot] = kv_len
        self._cached_prefix_sync_len[slot] = 0
        if self._spec_rows is not None:
            # The backend restored the recurrent checkpoint into the live
            # GDN column before activating the retained MTP pages.  Clearing
            # the speculative rows here would also erase that checkpoint.
            # Candidate columns are overwritten by verify, so only restore
            # the active-column pointer.
            self._spec_rows.activate(slot, 0)
        self._spec_state_col[slot] = 0
        return True

    def mtp_write_index(self, slot: int, kv_len: int) -> int:
        """Physical flattened KV row for one logical MTP token position.

        Legacy (fixed rows): the historical ``slot * pages_per_slot``
        formula. Phase 2 dynamic arena: the bundle mapping is shared with
        the backbone, so the physical page comes from the backbone pool's
        host page-table row (``Qwen36SlotPool.write_index`` on the MTP
        pool would be wrong -- MTP KV is a separate tensor; what is shared
        is the bundle *mapping*). Raises if the logical page has no bundle
        yet (writes must have been prepared first).
        """
        if self.mtp_page_size is None or self.mtp_pages_per_slot is None:
            raise RuntimeError("MTP KV geometry not configured")
        dynamic = getattr(self.backend.pool, "dynamic_arena", False)
        if not dynamic:
            from runtime.backends.qwen36_mtp_cudagraph import decode_write_index

            return decode_write_index(
                slot, kv_len, self.mtp_page_size, self.mtp_pages_per_slot
            )
        row = self.backend.pool._page_table_host[slot]  # noqa: SLF001
        logical_page = kv_len // self.mtp_page_size
        if logical_page >= len(row):
            raise RuntimeError(
                f"MTP write at {kv_len} needs logical page {logical_page} "
                f"but the page row has {len(row)} entries"
            )
        physical_page = row[logical_page]
        if physical_page == 0:
            raise RuntimeError(
                f"MTP write at {slot}:{kv_len} has no physical bundle for logical page "
                f"{logical_page}; prepare backbone KV writes first (bundle mapping is shared)"
            )
        return physical_page * self.mtp_page_size + kv_len % self.mtp_page_size

    def copy_prefix(self, source_slot: int, target_slot: int, kv_len: int) -> None:
        """Copy a retained causal MTP prefix into another slot.

        The target backbone/GDN restore is only valid when this second causal
        cache follows it.  This uses the same page-table-aware whole-page
        copy discipline as ``Qwen36SlotPool.copy_prefix_kv``; the next MTP
        sync writes the suffix tail before it can be consumed.
        """
        if not self.can_restore_prefix(source_slot, kv_len):
            raise RuntimeError(
                f"source slot {source_slot} has no retained MTP prefix of length {kv_len}"
            )
        if source_slot == target_slot:
            self.restore_prefix(target_slot, kv_len)
            return
        source = self._caches[source_slot]
        target = self._caches[target_slot]
        pages = (kv_len + source.page_size - 1) // source.page_size
        source_pages = source.page_table[0, :pages].to(dtype=torch.long)
        target_pages = target.page_table[0, :pages].to(dtype=torch.long)
        target.k_cache[target_pages] = source.k_cache[source_pages]
        target.v_cache[target_pages] = source.v_cache[source_pages]
        target.seq_len = kv_len
        self._sync_len[target_slot] = kv_len
        # This is a live request's restored state, not a retained snapshot.
        # Its cache becomes reusable only when Qwen36Backend.reset_slot calls
        # preserve_prefix after the request has finished.
        self._cached_prefix_sync_len[target_slot] = 0
        if self._spec_rows is not None:
            self._spec_rows.reset_slot(target_slot)
        self._spec_state_col[target_slot] = 0

    def drop_prefix(self, slot: int) -> None:
        self._cached_prefix_sync_len[slot] = 0

    def snapshot_prefix_to_scratch(
        self,
        source_slot: int,
        kv_len: int,
        *,
        scratch_pages: tuple[int, ...] | None = None,
    ) -> bool:
        """Copy a real MTP causal prefix into the pooled scratch cache.

        The backbone prefix arena lives in ``Qwen36SlotPool.scratch_row``.
        A persistent restore is valid only when this second causal cache has
        the same boundary, so expose the matching bounded snapshot here
        instead of letting the backend reach into MTP private buffers.  CPU
        / eager MTP intentionally has no scratch row and returns ``False``;
        callers then retain the existing same-slot cache behaviour.
        """
        if self.scratch_row is None or self.mtp_page_size is None:
            return False
        if self._sync_len[source_slot] < kv_len:
            return False
        source = self._caches[source_slot]
        scratch = self._caches[self.scratch_row]
        pages = (kv_len + source.page_size - 1) // source.page_size
        if scratch_pages is None:
            scratch_pages = tuple(range(pages))
        if len(scratch_pages) != pages or len(set(scratch_pages)) != pages:
            raise ValueError("MTP scratch prefix needs one distinct page per logical page")
        if any(page < 0 or page >= scratch.k_cache.shape[0] for page in scratch_pages):
            raise ValueError("MTP scratch prefix page is outside the scratch arena")
        source_pages = source.page_table[0, :pages].to(dtype=torch.long)
        scratch_page_tensor = torch.tensor(scratch_pages, dtype=torch.long, device=self.device)
        scratch.k_cache[scratch_page_tensor] = source.k_cache[source_pages]
        scratch.v_cache[scratch_page_tensor] = source.v_cache[source_pages]
        # The scratch row hosts one snapshot per persistent entry at
        # disjoint page offsets.  ``seq_len`` is a validity WATERMARK for
        # the whole arena, not the last-stored entry's length: a later
        # shorter store must not make an earlier longer entry un-restorable
        # (measured 2026-08-06: storing 4K/32K after 64K/128K dropped the
        # watermark below the long entries and every 64K/128K restore
        # failed with "persistent MTP prefix disappeared").  Reachable
        # entries keep their bytes because their page offsets are disjoint
        # and only freed at eviction, which removes the entry itself.
        scratch.seq_len = max(scratch.seq_len, kv_len)
        return True

    def restore_prefix_from_scratch(
        self,
        target_slot: int,
        kv_len: int,
        *,
        scratch_pages: tuple[int, ...] | None = None,
    ) -> bool:
        """Restore the pooled scratch snapshot into a real MTP cache row."""
        if self.scratch_row is None or self.mtp_page_size is None:
            return False
        scratch = self._caches[self.scratch_row]
        if scratch.seq_len < kv_len:
            return False
        target = self._caches[target_slot]
        pages = (kv_len + target.page_size - 1) // target.page_size
        if scratch_pages is None:
            scratch_pages = tuple(range(pages))
        if len(scratch_pages) != pages or len(set(scratch_pages)) != pages:
            raise ValueError("MTP scratch prefix needs one distinct page per logical page")
        if any(page < 0 or page >= scratch.k_cache.shape[0] for page in scratch_pages):
            raise ValueError("MTP scratch prefix page is outside the scratch arena")
        scratch_page_tensor = torch.tensor(scratch_pages, dtype=torch.long, device=self.device)
        target_pages = target.page_table[0, :pages].to(dtype=torch.long)
        target.k_cache[target_pages] = scratch.k_cache[scratch_page_tensor]
        target.v_cache[target_pages] = scratch.v_cache[scratch_page_tensor]
        target.seq_len = kv_len
        self._sync_len[target_slot] = kv_len
        self._cached_prefix_sync_len[target_slot] = 0
        if self._spec_rows is not None:
            # Column zero IS the target's live GDN state, and the persistent
            # restore path has already copied the checkpoint into it before
            # this call.  ``reset_slot`` would zero that state along with the
            # speculative candidates, so a full prefix hit would resume from
            # an empty GDN recurrence and emit wrong logits that nothing
            # downstream can detect (the full-hit corruption seen on
            # 2026-08-05).  The verify overwrites every candidate column
            # (destination rows 0..K) before reading them, so stale bytes
            # there are harmless; only the source-column pointer needs
            # pinning back to the live row.
            self._spec_rows.activate(target_slot, 0)
        self._spec_state_col[target_slot] = 0
        return True

    # -- drafting --------------------------------------------------------

    def _sync_real_suffix(
        self, slot: int, shifted_token_ids: list[int], target_hidden: torch.Tensor
    ) -> tuple[int, torch.Tensor]:
        """Write real target context into the MTP cache and return step 0.

        This is the native equivalent of the historical
        ``_mtp_sync_and_propose_batch`` step-0 forward.  It deliberately
        rewinds the physical cache to ``_sync_len`` first: rows beyond that
        point are speculative and must never become context merely because
        they happened to be accepted by the target later.
        """
        if not shifted_token_ids:
            raise ValueError("MTP synchronisation requires at least one real position")
        if target_hidden.dim() != 3 or target_hidden.shape[:2] != (1, len(shifted_token_ids)):
            raise ValueError("MTP target hidden must have shape [1, q, hidden_size]")
        cache = self._caches[slot]
        sync_len = self._sync_len[slot]
        cache.seq_len = sync_len
        inputs = torch.tensor([shifted_token_ids], dtype=torch.long, device=self.device)
        # Teacher-forcing needs every MTP hidden row to populate its causal
        # KV, but only the final row predicts draft step 0.  Projecting all
        # ``q`` rows through the 248k-vocabulary head is otherwise pure
        # work; the batched CUDA-graph path follows the same rule below.
        logits, mtp_hidden = self.model.mtp_forward(
            inputs,
            target_hidden,
            sync_len,
            cache,
            logits_last_position_only=True,
        )
        expected = sync_len + len(shifted_token_ids)
        if cache.seq_len != expected:
            raise RuntimeError(
                f"MTP sync for slot {slot} wrote {cache.seq_len - sync_len} rows; "
                f"expected {len(shifted_token_ids)}"
            )
        self._sync_len[slot] = expected
        return int(logits[0, -1].argmax(dim=-1).item()), mtp_hidden[:, -1:]

    def _sync_real_suffix_batch(
        self,
        slots: list[int],
        shifted_token_ids: list[list[int]],
        target_hidden_rows: list[torch.Tensor],
    ) -> dict[int, tuple[int, torch.Tensor]]:
        """Synchronise one equal-length acceptance group B-wide.

        A target verify is always uniform ``K+1``, but MTP's newly-real
        suffix is ``m+1`` and therefore ragged across slots.  Grouping equal
        lengths mirrors the historical batch driver without padding a real
        prefix with fake tokens.  CPU/stub and B=1 paths retain the proven
        per-slot primitive; pooled CUDA caches take the B-wide fast path.
        """
        if not slots or self._batched_sync is None:
            return {
                slot: self._sync_real_suffix(slot, tokens, target_hidden_rows[index])
                for index, (slot, tokens) in enumerate(zip(slots, shifted_token_ids, strict=True))
            }
        query_len = len(shifted_token_ids[0])
        if query_len <= 0 or any(len(tokens) != query_len for tokens in shifted_token_ids):
            raise ValueError("batched MTP synchronisation requires non-empty equal-length suffixes")
        if len(target_hidden_rows) != len(slots):
            raise ValueError("batched MTP target hidden requires one row-major tensor per slot")
        if any(
            row.dim() != 3 or tuple(row.shape[:2]) != (1, query_len) for row in target_hidden_rows
        ):
            raise ValueError(
                "batched MTP target hidden must have shape [1, q, hidden_size] per slot"
            )
        target_hidden = torch.cat(target_hidden_rows, dim=0)
        starts = [self._sync_len[slot] for slot in slots]
        for slot, start in zip(slots, starts, strict=True):
            self._caches[slot].seq_len = start
        first_drafts, mtp_hidden = self._batched_sync.replay(
            slots, shifted_token_ids, target_hidden, starts
        )
        first_draft_values = first_drafts.tolist()
        expected = [start + query_len for start in starts]
        result: dict[int, tuple[int | torch.Tensor, torch.Tensor]] = {}
        for index, (slot, end) in enumerate(zip(slots, expected, strict=True)):
            self._caches[slot].seq_len = end
            self._sync_len[slot] = end
            result[slot] = (
                int(first_draft_values[index]),
                mtp_hidden[index : index + 1, -1:],
            )
        self.stats["batched_sync_replays"] += 1
        self.stats["batched_sync_slots"] += len(slots)
        self.backend.stats["mtp_batched_sync_replays"] += 1
        self.backend.stats["mtp_batched_sync_slots"] += len(slots)
        return result

    def _sync_real_suffix_batch_ragged(
        self,
        slots: list[int],
        shifted_token_ids: list[list[int]],
        target_hidden_rows: list[torch.Tensor],
    ) -> dict[int, tuple[int | torch.Tensor, torch.Tensor]]:
        """Synchronise one all-slot ragged real suffix body when available.

        The historical fast path pads every slot to one fixed ``max_q`` MTP
        body, then picks each slot's own last valid logits/hidden instead of
        splitting by accepted length first.  This engine mirrors that shape
        when the pooled helper exposes ``replay_ragged`` and otherwise falls
        back to the previous equal-length grouping/per-slot implementation.

        On the batched graph path the first element is the **device** first-
        draft tensor row (``[1]``): converting it to a host int here forces a
        mid-round D2H sync before the draft graph can be enqueued.  The
        caller's ``_continue_draft_batch`` stages it D2D and converts after
        the draft replay has already synchronized the stream, so one round
        keeps at most one blocking D2H (the draft results).  The eager
        fallback paths below still return plain ``int``.
        """
        if not (len(slots) == len(shifted_token_ids) == len(target_hidden_rows)):
            raise ValueError("ragged MTP sync requires equal slot/token/hidden list lengths")
        if not slots:
            return {}
        if any(
            not tokens
            or hidden.dim() != 3
            or hidden.shape[0] != 1
            or hidden.shape[1] != len(tokens)
            for tokens, hidden in zip(shifted_token_ids, target_hidden_rows, strict=True)
        ):
            raise ValueError(
                "ragged MTP sync requires one non-empty [1, q, hidden_size] tensor per slot"
            )
        if not slots or self._batched_sync is None:
            return {
                slot: self._sync_real_suffix(slot, tokens, target_hidden_rows[index])
                for index, (slot, tokens) in enumerate(zip(slots, shifted_token_ids, strict=True))
            }

        replay_ragged = getattr(self._batched_sync, "replay_ragged", None)
        if replay_ragged is None:
            grouped: dict[int, list[tuple[int, list[int], torch.Tensor]]] = {}
            for slot, tokens, hidden in zip(
                slots, shifted_token_ids, target_hidden_rows, strict=True
            ):
                grouped.setdefault(len(tokens), []).append((slot, tokens, hidden))
            result: dict[int, tuple[int, torch.Tensor]] = {}
            for group in grouped.values():
                if len(group) == 1:
                    # A legacy helper without `replay_ragged` only promised
                    # the old B>=2 equal-length entrypoint.  Preserve its
                    # singleton eager fallback rather than treating the
                    # absence of ragged support as permission to call an
                    # incompatible one-row `replay` method.  The real graph
                    # helper implements `replay_ragged`, so production B=1
                    # still takes the unified graph path below.
                    slot, tokens, hidden = group[0]
                    result[slot] = self._sync_real_suffix(slot, tokens, hidden)
                    continue
                result.update(
                    self._sync_real_suffix_batch(
                        [entry[0] for entry in group],
                        [entry[1] for entry in group],
                        [entry[2] for entry in group],
                    )
                )
            return result

        starts = [self._sync_len[slot] for slot in slots]
        for slot, start in zip(slots, starts, strict=True):
            self._caches[slot].seq_len = start
        first_drafts, mtp_hidden = replay_ragged(
            slots, shifted_token_ids, target_hidden_rows, starts
        )
        result: dict[int, tuple[int, torch.Tensor]] = {}
        for index, (slot, start, tokens) in enumerate(
            zip(slots, starts, shifted_token_ids, strict=True)
        ):
            end = start + len(tokens)
            self._caches[slot].seq_len = end
            self._sync_len[slot] = end
            result[slot] = (
                first_drafts[index],
                mtp_hidden[index : index + 1],
            )
        self.stats["batched_sync_replays"] += 1
        self.stats["batched_sync_slots"] += len(slots)
        self.backend.stats["mtp_batched_sync_replays"] += 1
        self.backend.stats["mtp_batched_sync_slots"] += len(slots)
        return result

    def _continue_draft(self, slot: int, first_draft: int, first_hidden: torch.Tensor) -> list[int]:
        """Produce the remaining ``K-1`` autoregressive draft tokens.

        Step 0 is emitted by :meth:`_sync_real_suffix` from real target
        context.  Only these continuation steps are speculative and eligible
        for the fixed-shape draft CUDA Graph.

        The single-slot graph wrapper returns a device row; this path's
        contract is a host ``list[int]`` (the caller stores it in request
        state and single-slot ``round`` consumes it on host), so convert
        here rather than leaking CUDA scalars into the list.
        """
        if self.k <= 1:
            return [first_draft]
        cache = self._caches[slot]
        if self._draft_cg is not None:
            tail = self._draft_cg.replay(slot, first_draft, first_hidden, cache.seq_len)
            if isinstance(tail, torch.Tensor):
                tail = tail.tolist()
            expected_tail = self.k - 1
            if len(tail) != expected_tail:
                raise RuntimeError(
                    f"MTP draft graph returned {len(tail)} continuation tokens for slot {slot}; "
                    f"expected K-1={expected_tail} after teacher-forced step 0"
                )
            self._record_draft_graph_replay(1)
            return [first_draft, *tail]
        drafts = [first_draft]
        next_input = torch.tensor([[first_draft]], dtype=torch.long, device=self.device)
        mtp_hidden = first_hidden
        for _ in range(1, self.k):
            draft_token, mtp_hidden = self.model.mtp_step(
                next_input, mtp_hidden, cache.seq_len, cache
            )
            drafts.append(int(draft_token.item()))
            next_input = draft_token.view(1, 1)
        return drafts

    def _continue_draft_batch(
        self,
        slots: list[int],
        first_drafts: list[int | torch.Tensor],
        first_hiddens: torch.Tensor,
    ) -> dict[int, list[int] | torch.Tensor]:
        """Batch MTP's K-1 speculative continuation steps across slots.

        ``first_drafts`` accepts host ints (eager path) or per-slot device
        ``[1]`` rows (ragged-sync graph path).  In the graph path the device
        seeds are staged D2D into the draft graph's owned buffer and the
        K-token result is returned as a device row: the next verify fill
        copies it D2D and the GPU-side accept comparison slices it, so no
        host round-trip is needed at all.  Host ints remain the contract for
        the eager path (and for single-slot ``round``), where the values are
        genuinely consumed on the host.
        """

        def _as_int(value: int | torch.Tensor) -> int:
            return int(value) if isinstance(value, torch.Tensor) else value

        if len(slots) != len(first_drafts):
            raise ValueError("MTP draft slots and first draft tokens must have equal length")
        if self.k <= 1:
            return {slot: [_as_int(token)] for slot, token in zip(slots, first_drafts, strict=True)}
        if self._draft_cg is not None:
            starts = [self._caches[slot].seq_len for slot in slots]
            tails = self._draft_cg.replay_batch(slots, first_drafts, first_hiddens, starts)
            expected_tail = self.k - 1
            for slot in slots:
                tail = tails[slot]
                tail_len = int(tail.shape[0]) if isinstance(tail, torch.Tensor) else len(tail)
                if tail_len != expected_tail:
                    raise RuntimeError(
                        "MTP draft graph returned "
                        f"{tail_len} continuation tokens for slot {slot}; expected "
                        f"K-1={expected_tail} after teacher-forced step 0"
                    )
            self._record_draft_graph_replay(len(slots))
            if all(isinstance(tails[slot], torch.Tensor) for slot in slots):
                # Device graph path: combine the seeds and tails on device so
                # the next verify fill and accept comparison never touch host.
                if all(isinstance(token, torch.Tensor) for token in first_drafts):
                    first_rows = torch.cat([token.reshape(1) for token in first_drafts]).reshape(
                        -1, 1
                    )
                else:
                    # Host-seeded fallback (eager sync path): the seeds are
                    # already host ints, so one tiny H2D is cheaper than
                    # forcing per-token syncs inside the graph helper.
                    first_rows = torch.tensor(
                        [_as_int(token) for token in first_drafts],
                        dtype=torch.long,
                        device=self.device,
                    ).reshape(-1, 1)
                return {
                    slot: torch.cat([first_rows[index], tails[slot]], dim=0)
                    for index, slot in enumerate(slots)
                }
            first_values = [_as_int(token) for token in first_drafts]
            return {slot: [first_values[index], *tails[slot]] for index, slot in enumerate(slots)}
        return {
            slot: self._continue_draft(slot, _as_int(first_draft), first_hiddens[index : index + 1])
            for index, (slot, first_draft) in enumerate(zip(slots, first_drafts, strict=True))
        }

    def sync_prefill_chunk(
        self, slot: int, *, shifted_token_ids: list[int], target_hidden: torch.Tensor, final: bool
    ) -> list[int]:
        """Synchronise one prefill chunk; propose only after its final chunk."""
        first_draft, first_hidden = self._sync_real_suffix(slot, shifted_token_ids, target_hidden)
        if not final:
            return []
        if self._spec_rows is not None:
            self._spec_rows.sync_from_live(slot)
            self._spec_state_col[slot] = 0
        return self._continue_draft(slot, first_draft, first_hidden)

    def resync_prefix_tail(
        self, slot: int, *, shifted_token_ids: list[int], target_hidden: torch.Tensor
    ) -> list[int]:
        """Re-teacher-force the final prefix row after a full-prompt restore.

        The scratch snapshot already holds every row the cold prefill
        synced -- including this suffix's -- so the restore must back up
        one row before re-syncing it, or the MTP cache ends up one row
        ahead of the backbone KV and ``preserve_prefix``'s lockstep
        invariant breaks at reset.
        """
        if self._sync_len[slot] < 1:
            raise RuntimeError("full-prompt restore needs at least one synced MTP row")
        self._sync_len[slot] -= 1
        self._caches[slot].seq_len = self._sync_len[slot]
        return self.sync_prefill_chunk(
            slot, shifted_token_ids=shifted_token_ids, target_hidden=target_hidden, final=True
        )

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

        Every target hidden row needed to synchronise the newly real span is
        produced by this round's ``anchor + K`` verify forward.  No extra
        target forward is needed: the accepted prefix plus the recovery/bonus
        token becomes MTP's teacher-forced suffix, and only its later draft
        continuation remains speculative.
        """
        if isinstance(drafts, torch.Tensor):
            drafts = drafts.tolist()
        pool = self.backend.pool
        state = pool.slot_state(slot)
        k = len(drafts)
        if k != self.k:
            raise ValueError(f"round received {k} drafts; engine requires K={self.k}")
        cache = self._caches[slot]
        expected_speculative_len = self._sync_len[slot] + max(k - 1, 0)
        if cache.seq_len != expected_speculative_len:
            raise RuntimeError(
                f"round: slot {slot} MTP cache has seq_len={cache.seq_len}; expected "
                f"real prefix {self._sync_len[slot]} + K-1={max(k - 1, 0)} continuation rows"
            )

        # -- (a) ONE forward over [anchor] + drafts, qo_len = k+1.
        #
        # This used to be TWO full 64-layer forwards: a [1,1] pass to advance
        # the target through anchor_token, then a [1,k] verify. Decode here is
        # memory-bandwidth-bound, so the k+1-token forward costs about what
        # the 1-token one alone did, and the anchor's KV write simply happens
        # at position 0 of this pass. The historical implementation was built
        # this way from the start -- `verify_batch_spec` at qo_len=k+1,
        # oracle/.../direct_model_runner.py:1581 -- and the extra forward was
        # measured at ~35 ms of the ~137 ms round
        # (notes/2026-08-03-mtp-verify-mode.md).
        #
        # `all_hiddens` is row-for-row aligned with `verify_tokens`: row 0
        # was produced while consuming the anchor and predicts drafts[0]; row
        # `m` predicts the recovery/bonus token after `m` accepted drafts.
        #
        # `past_len` is now the length BEFORE the anchor, so the commit below
        # keeps `m + 1` positions (the anchor is always accepted), not `m`.
        past_len = state.num_tokens_seen
        verify_tokens = [anchor_token, *drafts]
        if self._verify_cg is not None:
            all_hiddens, all_logits = self._verify_cg.replay([slot], [verify_tokens], [past_len])
            all_logits = all_logits[0]  # [k+1, vocab]
            self._record_verify_graph_replay(1)
            gdn_snapshots = None
        else:
            verify_input = torch.tensor([verify_tokens], dtype=torch.long, device=self.device)
            if self._spec_rows is not None:
                all_hiddens, gdn_snapshots = self.model.verify_forward(
                    verify_input,
                    state,
                    spec_state_rows=self._spec_rows.rows_for_slot(slot),
                )
            else:
                all_hiddens, gdn_snapshots = self.model.verify_forward(verify_input, state)
        if self._verify_cg is None:
            all_logits = self.model.compute_logits(all_hiddens)[0]  # [k+1, vocab]

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

        # -- (b) commit: roll the target's GDN/attention state back to the
        # accepted prefix. That prefix is `m + 1` positions, not `m`: this
        # round's forward started at the anchor, and the anchor is committed
        # unconditionally (accept/reject only ever rejects DRAFTS). past_len
        # is correspondingly the length before the anchor.
        self.model.commit_verify(state, gdn_snapshots, past_len=past_len, accepted_count=m + 1)
        if self._spec_rows is not None:
            # Column m is the state after position m: anchor plus m accepted
            # drafts.  The incoming state occupied column zero only before
            # this verify and has just been overwritten by the anchor result.
            self._spec_rows.activate(slot, m)
            self._spec_state_col[slot] = m

        # -- KV/committed-token bookkeeping, "committed ahead of kv by one"
        # (the same invariant DFlash's own round already keeps for Laguna
        # -- runtime/backends/laguna_dflash.py's dflash_round): the FINAL
        # committed token (recovery/bonus) is client-visible now but its
        # own KV write is deferred to the NEXT round -- which since the
        # anchor was folded into the verify forward means position 0 of the
        # next round's own (a), rather than a separate anchor step. The
        # invariant is unchanged; only the place that discharges it moved.
        # `committed` extends slot_committed_tokens in full, while
        # slot_kv_len only advances by what commit_verify actually wrote.
        pool.slot_kv_len[slot] = state.num_tokens_seen
        pool.slot_committed_tokens[slot].extend(committed)
        self.backend._maybe_checkpoint(slot)  # noqa: SLF001 -- friend-class access, same pattern DFlash uses

        # -- (c) re-synchronise the MTP head with every newly real target
        # position.  The target verify already produced exactly these hidden
        # rows, so this is no extra backbone forward.  It replaces the old
        # tail-only cache rewind (and its optional interior-only resync),
        # restoring the historical invariant ``mtp_sync_len == target KV
        # length`` before adding the next speculative tail.
        real_new_tokens = [anchor_token, *committed[:-1]]
        shifted = [*real_new_tokens[1:], new_anchor]
        first_draft, first_hidden = self._sync_real_suffix(
            slot, shifted, all_hiddens[:, : m + 1, :]
        )
        next_drafts = self._continue_draft(slot, first_draft, first_hidden)

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

    def round_batch(
        self,
        slots: list[int],
        anchors: dict[int, int],
        drafts_by_slot: dict[int, list[int] | torch.Tensor],
        *,
        params_per_slot: dict[int, SamplingParams] | None = None,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> dict[int, dict[str, Any]]:
        """Run one uniform ``anchor + K`` verify for every active slot.

        M-2's graph has a fixed query length but a variable request count.
        The target verify and greedy acceptance batch across slots; M-4 also
        batches each position of the intrinsically chained MTP draft head.
        Sampling remains on the proven single-slot path until its probability
        tensors are made batched too.
        """
        if not slots:
            return {}
        if self._verify_cg is None or any(
            params_per_slot is not None
            and params_per_slot.get(slot) is not None
            and not params_per_slot[slot].is_greedy
            for slot in slots
        ):
            # The single-slot fallback consumes drafts on host; a device row
            # would splat into a list of CUDA scalars and break the verify
            # fill's numpy conversion.  Normalise once here.
            drafts_by_slot = {
                slot: (value.tolist() if isinstance(value, torch.Tensor) else value)
                for slot, value in drafts_by_slot.items()
            }
            return {
                slot: self.round(
                    slot,
                    anchors[slot],
                    drafts_by_slot[slot],
                    params=(params_per_slot.get(slot) if params_per_slot else None),
                    return_logprobs=return_logprobs,
                    top_logprobs=top_logprobs,
                )
                for slot in slots
            }

        # Drafts arrive from two sources: the prefill/eager path stores host
        # ``list[int]``, while a previous graph round stores device ``[K]``
        # rows.  A request that finished prefill this round can therefore
        # mix both kinds in one batch.  The verify graph fill accepts either
        # an all-host or all-device payload, never a per-slot mix -- so
        # normalise every slot to device once any slot is already a tensor.
        # Values are bit-identical (the tensor is built from the same ints),
        # and the device accept path reconstructs ``committed`` from the
        # verifier's own predictions, which match the accepted drafts by the
        # greedy match condition.
        if any(isinstance(drafts_by_slot[slot], torch.Tensor) for slot in slots):
            normalized: dict[int, torch.Tensor] = {}
            for slot in slots:
                value = drafts_by_slot[slot]
                if isinstance(value, torch.Tensor):
                    if tuple(value.shape) != (self.k,):
                        raise ValueError("batched MTP verify requires uniform [K] device drafts")
                    normalized[slot] = value
                else:
                    if len(value) != self.k:
                        raise ValueError(
                            "batched MTP verify requires the engine's uniform K drafts per slot"
                        )
                    normalized[slot] = torch.tensor(
                        [int(token) for token in value], dtype=torch.long, device=self.device
                    )
            drafts_by_slot = normalized
        else:
            # Defensive: a caller may have built "host" lists whose elements
            # are CUDA scalars (``list(device_tensor)``).  The verify fill's
            # numpy conversion cannot take those; normalise to device once
            # rather than leaking a second mixed-type failure mode.
            if any(
                any(isinstance(token, torch.Tensor) for token in drafts_by_slot[slot])
                for slot in slots
            ):
                drafts_by_slot = {
                    slot: torch.tensor(
                        [int(token) for token in drafts_by_slot[slot]],
                        dtype=torch.long,
                        device=self.device,
                    )
                    for slot in slots
                }
        if isinstance(drafts_by_slot[slots[0]], torch.Tensor):
            if any(tuple(drafts_by_slot[slot].shape) != (self.k,) for slot in slots):
                raise ValueError("batched MTP verify requires uniform [K] device drafts per slot")
        elif any(len(drafts_by_slot[slot]) != self.k for slot in slots):
            raise ValueError("batched MTP verify requires the engine's uniform K drafts per slot")

        round_profile.begin_round()
        pool = self.backend.pool
        states = [pool.slot_state(slot) for slot in slots]
        caches = [self._caches[slot] for slot in slots]
        past_lens = [state.num_tokens_seen for state in states]
        round_profile.phase("setup")
        if any(
            cache.seq_len != self._sync_len[slot] + max(self.k - 1, 0)
            for slot, cache in zip(slots, caches)
        ):
            raise RuntimeError(
                "batched MTP verify received a cache without its K-1-row continuation tail"
            )

        first_value = drafts_by_slot[slots[0]]
        if isinstance(first_value, torch.Tensor):
            # Device drafts: fill the preallocated [B, K+1] verify input in
            # place (pinned anchor row + per-slot D2D draft rows).  This is
            # allocation-free on the hot path -- the historical
            # torch.tensor/torch.cat construction allocated three small
            # CUDA tensors per round and, under 94/96 GiB resident memory,
            # the allocator round-trips landed on the host-critical replay
            # fill.
            batch = len(slots)
            verify_tokens = self._verify_tokens_buf[batch]
            anchor_view = self._anchor_host_views[batch]
            anchor_view.reshape(-1)[:] = [anchors[slot] for slot in slots]
            verify_tokens[:, 0].copy_(self._anchor_host[batch].reshape(-1), non_blocking=True)
            for index, slot in enumerate(slots):
                verify_tokens[index, 1:].copy_(drafts_by_slot[slot], non_blocking=True)
            drafts_arg = verify_tokens
        else:
            verify_tokens = [[anchors[slot], *drafts_by_slot[slot]] for slot in slots]
            drafts_arg = {slot: [anchors[slot], *drafts_by_slot[slot]] for slot in slots}
        _verify_ev0 = _verify_ev1 = None
        if round_profile.cuda_events:
            _verify_ev0 = torch.cuda.Event(enable_timing=True)
            _verify_ev1 = torch.cuda.Event(enable_timing=True)
            _verify_ev0.record()
        all_hiddens, all_logits = self._verify_cg.replay(slots, verify_tokens, past_lens)
        if round_profile.cuda_events:
            _verify_ev1.record()
        round_profile.phase("verify_replay")
        self._record_verify_graph_replay(len(slots))
        round_profile.phase("compute_logits")
        if round_profile.cuda_events:
            _post_verify_ev = torch.cuda.Event(enable_timing=True)
        decisions = determine_accept_reject_batch(
            slots,
            drafts_arg,
            all_logits.reshape(-1, all_logits.shape[-1]),
            self.k,
        )
        if round_profile.cuda_events:
            _post_verify_ev.record()
            _post_verify_ev.synchronize()
            round_profile.note("post_verify_ms", _verify_ev1.elapsed_time(_post_verify_ev))
        round_profile.phase("accept_decision")
        if round_profile.cuda_events and _verify_ev0 is not None and _verify_ev1 is not None:
            round_profile.note("verify_gpu_ms", _verify_ev0.elapsed_time(_verify_ev1))
        results: dict[int, dict[str, Any]] = {}
        sync_slots: list[int] = []
        sync_tokens: list[list[int]] = []
        sync_hidden_rows: list[torch.Tensor] = []
        for request, slot in enumerate(slots):
            state = states[request]
            decision = decisions[slot]
            m = decision["num_accepted"]
            committed: list[int] = decision["committed"]
            new_anchor = committed[-1]
            self.model.commit_verify(state, None, past_len=past_lens[request], accepted_count=m + 1)
            if self._spec_rows is not None:
                self._spec_rows.activate(slot, m)
                self._spec_state_col[slot] = m
            pool.slot_kv_len[slot] = state.num_tokens_seen
            pool.slot_committed_tokens[slot].extend(committed)
            self.backend._maybe_checkpoint(slot)  # noqa: SLF001 - backend owns checkpoint policy
            real_new_tokens = [anchors[slot], *committed[:-1]]
            shifted = [*real_new_tokens[1:], new_anchor]
            sync_slots.append(slot)
            sync_tokens.append(shifted)
            sync_hidden_rows.append(all_hiddens[request : request + 1, : m + 1])
            result: dict[str, Any] = {
                "committed": committed,
                "num_accepted": m,
                "next_anchor": new_anchor,
            }
            if return_logprobs:
                result["logprobs"] = [
                    compute_logprobs(
                        all_logits[request, position : position + 1],
                        [committed[position]],
                        top_k=top_logprobs,
                    )[0]
                    for position in range(len(committed))
                ]
            results[slot] = result

        round_profile.phase("commit_loop")
        if round_profile.cuda_events:
            _sync_ev0 = torch.cuda.Event(enable_timing=True)
            _sync_ev1 = torch.cuda.Event(enable_timing=True)
            _sync_ev0.record()
        first_by_slot = self._sync_real_suffix_batch_ragged(
            sync_slots, sync_tokens, sync_hidden_rows
        )
        if round_profile.cuda_events:
            _sync_ev1.record()
        round_profile.phase("sync_ragged")
        first_drafts = [first_by_slot[slot][0] for slot in slots]
        first_hiddens = [first_by_slot[slot][1] for slot in slots]

        if round_profile.cuda_events:
            _draft_ev0 = torch.cuda.Event(enable_timing=True)
            _draft_ev1 = torch.cuda.Event(enable_timing=True)
            _draft_ev0.record()
        next_drafts_by_slot = self._continue_draft_batch(
            slots, first_drafts, torch.cat(first_hiddens, dim=0)
        )
        if round_profile.cuda_events:
            _draft_ev1.record()
        round_profile.phase("draft_batch")
        for slot in slots:
            results[slot]["next_draft_tokens"] = next_drafts_by_slot[slot]

        self.stats["rounds"] += len(slots)
        if round_profile.cuda_events:
            _ms = torch.cuda.memory_stats()
            round_profile.note("device_alloc", _ms["num_device_alloc"])
            round_profile.note("device_free", _ms["num_device_free"])
            # One blocking sync at the round boundary.  The historical path
            # drained the stream with the draft-row ``.tolist()`` before
            # these elapsed-time reads; the device-direct path has no
            # natural drain, so complete both timed spans explicitly (the
            # events are stream-ordered, so syncing the last one finishes
            # both).
            _draft_ev1.synchronize()
            round_profile.note("sync_gpu_ms", _sync_ev0.elapsed_time(_sync_ev1))
            round_profile.note("draft_gpu_ms", _draft_ev0.elapsed_time(_draft_ev1))
        round_profile.end_round(label=f"mtp_round_b{len(slots)}")
        return results
