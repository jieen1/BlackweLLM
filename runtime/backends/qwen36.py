"""Qwen3.6 serving backend -- Track B / B2 (``docs/implementation-plan.md``
§7.1: 固定槽位 + 连续批处理 · 递归状态纳入槽位生命周期 · CUDA Graph ·
前缀缓存联动驱逐 · 并发 ≥ 2).

What is new here relative to B1
--------------------------------
B1 (``runtime/model/qwen36_model.py``) built a correct **batch=1, caller-owns-
the-state** model graph and said so in its own docstring ("No slot manager, no
continuous batching, no ``BlockPool``/``RecurrentStatePool`` integration ...
Wiring this into ``ModelBackend`` is B2 scope, not attempted here"). This
module is that wiring, and it is the **first real user of Track A's step-7
coordinator**: ``ArchitectureSpec.needs_two_cache_families`` is ``True`` for a
Qwen3.6 checkpoint, which is the branch
``runtime/slot_resource_manager.SlotResourceManager`` used to raise
``NotImplementedError`` on.

The second cache family, concretely
------------------------------------
48 of this checkpoint's 64 layers are GDN (recurrent). Their state is not a
cache in the KV sense:

* KV bytes past ``kv_len`` are simply never read, so a stale slot is harmless
  and ``reset_slot`` deliberately preserves them (that *is* the prefix cache).
* A GDN state IS read on the first step of the next sequence to use the slot.
  A stale one produces a plausible, non-crashing, **wrong** continuation. So
  :meth:`reset_slot` zeroes it -- the one operational requirement B0-5
  attached to its capture-safe verdict.

And it is not block-composable: there is no way to "start from position 400 of
a 900-token resident KV region" unless a checkpoint was taken at exactly that
boundary. That asymmetry is the whole reason
:class:`runtime.backends.protocol.PrefixHit` carries two numbers, and why
``.effective`` is ``state_hit`` and never ``kv_hit``
(``docs/a3-cache-coordinator-design.md`` §3).

Checkpoint policy: one rolling, block-aligned checkpoint per slot
------------------------------------------------------------------
A checkpoint here is ~77 MiB for this checkpoint's geometry (48 layers x
(conv ``[10240,4]`` + ssm ``[48,128,128]``) in BF16 -- recomputed from the
real config, not inherited from ``notes/prefix-cache-design.md``'s
"~151MB/checkpoint", which §4 of the A3 design explicitly flags as an
old-hardware number that must not be copied forward).

Taken at every ``block_size``-th token a slot commits, overwriting the
previous one. Cost: one ~77 MiB device-to-device clone per ``block_size``
tokens -- ~0.1% of a 27B decode step at ``block_size=64``, measured against
the step time, not assumed. Benefit: when the slot is reused by a
continuation of the same conversation (the workload this runtime actually
serves), the deepest reusable boundary is within ``block_size`` tokens of the
whole previous turn.

Why *rolling* rather than "every boundary, LRU-evicted": with 48 GDN layers a
single extra retained checkpoint costs more than the entire KV cache of a
short request, and the deepest boundary is the only one a same-slot
continuation can use. The byte-budgeted, LRU, lockstep-evicted
:class:`runtime.recurrent_state_pool.RecurrentStatePool` still governs *which
slots* may hold one -- that is what makes it a real allocator rather than a
per-slot attribute, and it is what implements INV-A3-3's两个方向 (a KV-side
prefix drop cascades into the checkpoint; the checkpoint's own byte budget
never reclaims live KV, it only "turns a future would-be hit into a safe
compute miss").

Speculative decoding / MTP (B3, wired 2026-08-03)
---------------------------------------------------
``capabilities.speculative_decode`` is ``True``: MTP is reachable through
:meth:`Qwen36Backend.enable_mtp`, wired the same way Laguna's DFlash is
(``LagunaBackend.enable_dflash`` / ``has_speculative_decode``). The round
driver itself -- per-slot MTP KV cache, draft/verify/accept-reject/re-draft
-- lives in :mod:`runtime.backends.qwen36_mtp` (:class:`Qwen36MTPEngine`),
kept separate from this class for the same reason ``DFlashEngine`` is kept
separate from ``LagunaBackend``: opt-in bookkeeping layered on top of the
always-on slot pool, not a change to it. See that module's docstring for
the (token, hidden) pairing bug found and fixed while wiring this in --
every prior standalone B3 script measured acceptance far below what the
model API's own documented contract implies, because every one of them
violated that contract the same way.
``runtime/recurrent_state_pool.py``'s ``spec_row`` addressing (built for a
BATCHED multi-slot GDN spec verify, matching the historical vLLM-based
runner's ``verify_batch_spec``) still has no caller here: this backend's
model graph is batch=1 by construction, so MTP rounds are driven
sequentially per slot (matching DFlash's own documented precedent), not
through a fused batched verify. A real future throughput lever, not
attempted in this landing.

Still deliberately out of scope, stated rather than implied
--------------------------------------------------------------
* **Warm continue.** ``capabilities.warm_continue`` is ``False`` -- the same
  honest ``False`` that ``protocol.py``'s docstring says the Laguna path
  should have been carrying all along (N8).
* **Cross-slot KV sharing.** Static per-slot pages, per §8 decision 5.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from runtime.backends.protocol import (
    BackendCapabilities,
    BackendSnapshot,
    PrefixHit,
    PrefixSnapshot,
    SlotSnapshot,
)
from runtime.block_pool import ChunkedPrefillState
from runtime.logprobs import compute_logprobs
from runtime.model.qwen36_model import Qwen36ForCausalLMSelfBuilt
from runtime.model.qwen36_slots import Qwen36SlotPool
from runtime.recurrent_state_pool import RecurrentStatePool
from runtime.sampling import SamplingParams, make_generator, sample_from_logits

if TYPE_CHECKING:
    from runtime.backends.qwen36_mtp import Qwen36MTPEngine

logger = logging.getLogger(__name__)

#: Default byte budget for the recurrent-checkpoint pool, expressed as a
#: multiple of one checkpoint. Two is the smallest number that makes the
#: pool's eviction path reachable at ``num_slots >= 3`` -- a budget nothing
#: can ever exceed is a budget whose eviction code has never run, which is
#: the C8 discipline this repo already applies to its gates ("没红过的门禁,
#: 能不能构造一个让它红的输入").
DEFAULT_CHECKPOINT_BUDGET_MULTIPLE = 2


def _prefix_hash(token_ids: list[int], length: int) -> str:
    """Content identity of ``token_ids[:length]``.

    A hash, not the token list itself, because this value is the key both
    resources are co-keyed on (INV-A3-3) and it gets stored per checkpoint;
    keeping whole prompts alive in a side table is how a "cache" becomes a
    transcript. blake2b over the little-endian int32 encoding -- stable
    across processes, unlike Python's salted ``hash()``.
    """
    buf = bytearray()
    for tok in token_ids[:length]:
        buf += int(tok).to_bytes(4, "little", signed=False)
    return hashlib.blake2b(bytes(buf), digest_size=16).hexdigest()


@dataclass
class Qwen36SlotStateView:
    """Read-only view of one slot, satisfying ``protocol.SlotStateView``."""

    kv_len: int
    committed_tokens: tuple[int, ...]

    @property
    def is_fresh(self) -> bool:
        return self.kv_len == 0 and not self.committed_tokens


#: int32's positive range -- the width of the element count in the cutlass DSL
#: memref descriptor sparkinfer's w4a16 fused MoE builds. Exceeding it raises
#: OverflowError from inside the kernel launch, not from anything we control.
_W4A16_MEMREF_ELEMENT_LIMIT = 2**31 - 1

#: Preferred tokens per prefill forward. Matches LagunaBackend's own
#: ``_prefill_chunk_tokens`` default; capped by model geometry in
#: :meth:`Qwen36Backend._prefill_chunk_tokens`.
_PREFERRED_PREFILL_CHUNK_TOKENS = 8192


class Qwen36Backend:
    """Fixed-slot, continuously-batched serving backend for Qwen3.6.

    Conforms to :class:`runtime.backends.protocol.ModelBackend` -- checked
    mechanically by ``tests/test_qwen36_backend.py`` via
    ``check_conformance``, not by eye.
    """

    def __init__(
        self,
        model: Qwen36ForCausalLMSelfBuilt,
        *,
        num_slots: int = 4,
        max_seq_len: int = 4096,
        block_size: int = 64,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_prefix_cache: bool = True,
        checkpoint_byte_budget: int | None = None,
        batched_decode: bool = True,
    ) -> None:
        self.model = model
        self.num_slots = num_slots
        self.block_size = block_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.enable_prefix_cache = enable_prefix_cache
        self.batched_decode = batched_decode

        self.pool = Qwen36SlotPool(
            model,
            num_slots=num_slots,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        if self.pool.page_size % block_size != 0:
            # §1.7 of the A3 design says this divisibility holds today by
            # coincidence of two independently chosen defaults and must be
            # re-verified, not assumed, the moment a checkpoint-boundary
            # policy is chosen. This is that re-verification, as an assert
            # rather than a sentence in a document.
            raise ValueError(
                f"page_size={self.pool.page_size} must be a multiple of "
                f"block_size={block_size}: prefix-cache boundaries are "
                "block-aligned and must not straddle a page"
            )
        self.max_seq_len = self.pool.max_seq_len
        if self.device.type == "cuda":
            # Pre-build every batch size's eager driver so no request pays
            # a first-call allocation. On CPU they are built lazily instead:
            # a CPU-device backend is not a fallback execution path (the
            # kernels are CUDA-only and a forward there fails loudly), it
            # exists so the slot/prefix/checkpoint bookkeeping below -- pure
            # Python, and where the silent-corruption failure modes live
            # (INV-A3-1/2/3) -- can be tested deterministically against a
            # stub model instead of only against a 50 GiB checkpoint on a
            # contended GPU. See tests/test_qwen36_backend.py.
            self.pool.ensure_decode_workspaces(max_batch=num_slots)

        # -- prefix cache bookkeeping (same shape as LagunaBackend's) ------
        self._prefix_cache_tokens: list[list[int] | None] = [None] * num_slots
        self._prefix_cache_kv_len: list[int] = [0] * num_slots
        self._pending_prefix_hits: dict[int, PrefixHit] = {}

        # -- second cache family -------------------------------------------
        ckpt_bytes = self.pool.recurrent_checkpoint_nbytes()
        if checkpoint_byte_budget is None:
            checkpoint_byte_budget = ckpt_bytes * DEFAULT_CHECKPOINT_BUDGET_MULTIPLE
        self.checkpoint_pool = RecurrentStatePool(
            checkpoint_byte_budget,
            should_drop_kv_hash=self._checkpoint_kv_is_free,
            drop_kv_hash=self._drop_kv_for_checkpoint,
        )
        self._checkpoint_bytes = ckpt_bytes
        #: slot -> cloned tensors of the checkpoint currently registered for
        #: it. Kept out of :class:`RecurrentStatePool` on purpose: that class
        #: is torch-free bookkeeping (its own docstring), and giving it real
        #: tensors would make it untestable in the CPU-only CI job.
        self._checkpoint_tensors: dict[int, list[torch.Tensor]] = {}
        self._checkpoint_len: dict[int, int] = {}

        self.stats: dict[str, int] = {
            "prefix_kv_hit_tokens": 0,
            "prefix_state_hit_tokens": 0,
            "prefix_hit_split_events": 0,
            "checkpoints_taken": 0,
            "checkpoints_evicted_by_budget": 0,
            "checkpoints_evicted_by_kv": 0,
            "decode_rounds": 0,
            "decode_tokens": 0,
            "decode_graph_replays": 0,
        }

        self._decode_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._decode_graph_logits: dict[int, torch.Tensor] = {}
        self._graph_pool = None
        #: B3 step 0 (``docs/implementation-plan.md`` §7.3 C7-2): whether the
        #: decode CUDA Graph actually captured in THIS process, made
        #: queryable rather than only visible as a ``logger.info``/
        #: ``logger.exception`` line that this runtime's default log config
        #: does not persist. Same shape as ``DFlashEngine.cg_status``
        #: (``"captured"``/``"failed"``, key = graph name) -- surfaced
        #: through :meth:`snapshot`'s ``dflash_cg_status`` field, which despite
        #: its name is not DFlash-specific (see that field's updated
        #: docstring in ``runtime/backends/protocol.py``).
        self.cg_status: dict[str, str] = {}

        #: B3 (2026-08-03): the MTP round driver, wired via :meth:`enable_mtp`.
        #: ``None`` until then -- see :mod:`runtime.backends.qwen36_mtp`.
        self._mtp: Qwen36MTPEngine | None = None

        self.pool.reset_all()

    # -- protocol: always required ----------------------------------------

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            speculative_decode=True,
            prefix_cache=self.enable_prefix_cache,
            cuda_graph=True,
            chunked_prefill=True,
            warm_continue=False,
        )

    @property
    def has_speculative_decode(self) -> bool:
        """Whether MTP-verified rounds are active on THIS instance.

        ``capabilities.speculative_decode`` above says this backend CAN do
        MTP (:meth:`enable_mtp` may be called); this says whether it IS
        active right now -- same split Laguna's own ``has_speculative_decode``
        docstring documents for DFlash.
        """
        return self._mtp is not None

    def enable_mtp(self, *, num_speculative_tokens: int, enable_resync: bool | None = None) -> None:
        """Attach the owned MTP round driver before any production slot is
        used -- same "capture/wire before real use" window
        ``LagunaBackend.enable_dflash`` and this class's own
        ``capture_decode_cuda_graph`` already rely on. Requires the model
        to have been loaded with ``enable_mtp=True``
        (``runtime.model_loading.load_qwen36_model``) -- ``Qwen36MTPEngine``
        raises a clear error otherwise rather than this method silently
        doing nothing.
        """
        if self._mtp is not None:
            return
        from runtime.backends.qwen36_mtp import Qwen36MTPEngine

        self._mtp = Qwen36MTPEngine(
            self, num_speculative_tokens=num_speculative_tokens, enable_resync=enable_resync
        )

    def slot_state(self, slot: int) -> Qwen36SlotStateView:
        return Qwen36SlotStateView(
            kv_len=self.pool.slot_kv_len[slot],
            committed_tokens=tuple(self.pool.slot_committed_tokens[slot]),
        )

    def snapshot(self) -> BackendSnapshot:
        slots = tuple(
            SlotSnapshot(
                slot=s,
                kv_len=self.pool.slot_kv_len[s],
                is_fresh=self.slot_state(s).is_fresh,
            )
            for s in range(self.num_slots)
        )
        prefix = tuple(
            PrefixSnapshot(
                slot=s,
                cached_kv_len=self._prefix_cache_kv_len[s],
                cached_tokens=len(self._prefix_cache_tokens[s] or ()),
                head=tuple((self._prefix_cache_tokens[s] or ())[:8]),
            )
            for s in range(self.num_slots)
        )
        return BackendSnapshot(
            slots=slots, prefix=prefix, dflash_cg_status=tuple(sorted(self.cg_status.items()))
        )

    def reset_slot(self, slot: int) -> None:
        """Release ``slot``, saving its prefix cache; zero its GDN state.

        The double-reset guard is Laguna's, for the same reason
        (``laguna.py`` ``reset_slot``: "finish -> reset saves cache;
        admission -> reset would clear it"). What is NOT Laguna's, and is
        the whole point of this backend existing, is the recurrent half:
        the saved KV is left in place because a future request may reuse
        it, while the live recurrent state is zeroed because a future
        request must never inherit it. Whatever recurrent state deserves
        to survive was already cloned into the checkpoint pool at a
        block boundary; the live buffer is not that clone.
        """
        if self.pool.slot_committed_tokens[slot] and self.pool.slot_kv_len[slot] > 0:
            self._prefix_cache_tokens[slot] = list(self.pool.slot_committed_tokens[slot])
            self._prefix_cache_kv_len[slot] = self.pool.slot_kv_len[slot]
        self.pool.reset_slot(slot)
        self._pending_prefix_hits.pop(slot, None)
        if self._mtp is not None:
            self._mtp.reset_slot(slot)

    def drop_prefix_cache(self, slot: int) -> None:
        """Forget ``slot``'s saved KV prefix -- and, in lockstep, its
        recurrent checkpoint (INV-A3-3, forward direction).

        Unconditional cascade: once the KV side has decided those bytes no
        longer describe the tokens it thought they did, a checkpoint keyed
        to them can only produce a wrong resume. This is
        ``oracle/qwen36_vllm/direct_model_runner.py:590``'s
        ``block_pool._on_evict_block = evict_gdn_checkpoint`` hook,
        expressed as a call instead of an assignment.
        """
        self._prefix_cache_tokens[slot] = None
        self._prefix_cache_kv_len[slot] = 0
        if slot in self._checkpoint_tensors:
            self.stats["checkpoints_evicted_by_kv"] += 1
        self._evict_checkpoint(slot)

    # -- protocol: prefill --------------------------------------------------

    def prefill(self, slot: int, prompt_ids: list[int]) -> int:
        """Cold prefill of ``prompt_ids`` into ``slot``; greedy first token.

        Always cold, from position 0 -- matching ``LagunaBackend.prefill``'s
        own contract ("This method always does a full cold prefill"). The
        prefix-cache-aware entry point is
        :meth:`prefill_chunked_begin`, which consults
        :meth:`reconcile_prefix_hit`'s pending side table the same way
        Laguna's does.
        """
        logits = self._prefill_forward(slot, prompt_ids, prefix_hit=0)
        first_token = int(logits[-1].argmax(dim=-1).item())
        self._commit_prefill(slot, prompt_ids, first_token)
        return first_token

    def prefill_sampled(self, slot: int, prompt_ids: list[int], params: SamplingParams) -> int:
        logits = self._prefill_forward(slot, prompt_ids, prefix_hit=0)
        gen = make_generator(params.seed)
        first_token = int(sample_from_logits(logits[-1].unsqueeze(0), params, generator=gen).item())
        self._commit_prefill(slot, prompt_ids, first_token)
        return first_token

    def _prefill_forward(
        self,
        slot: int,
        prompt_ids: list[int],
        *,
        prefix_hit: int,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the prompt's suffix through the model, return ``[q, vocab]``.

        ``prefix_hit`` tokens are taken as already resident: the slot's KV
        bytes for ``[0, prefix_hit)`` are reused in place and its recurrent
        state has already been restored from the co-keyed checkpoint by
        :meth:`_apply_prefix_hit`. Only ``prompt_ids[prefix_hit:]`` is
        forwarded.

        This is B1's eager single-sequence path, unchanged: same
        ``Qwen36GenerationState``, same one-shot extend call. The only
        difference is that the state object's tensors are views into the
        slot pool instead of freshly allocated -- which is what makes
        "bit-exact against B1 eager" a structural claim about this path
        rather than a coincidence.

        ``return_hidden`` (B3, default False -- every pre-MTP caller keeps
        the old ``logits``-only return byte-for-byte): also returns
        ``hidden`` (``[q, hidden_size]``, the SAME forward's post-norm
        hidden states this method already computes and previously
        discarded). MTP needs the LAST row of this -- the hidden state
        that predicted ``first_token``'s own argmax, one position before
        it -- to correctly seed its first draft step (see
        ``runtime.backends.qwen36_mtp``'s module docstring for why every
        prior B3 script got this wrong by recomputing a DIFFERENT, later
        hidden state instead of keeping this one).
        """
        state = self.pool.slot_state(slot)
        if self.pool.slot_kv_len[slot] != prefix_hit:
            # The slot must be at exactly the position this prefill intends to
            # continue from: 0 for a cold prefill, ``prefix_hit`` after a
            # restore. Anything else means the caller skipped a reset, and the
            # GDN layers would start from another sequence's recurrent state
            # -- no exception, no NaN, just a wrong continuation
            # (INV-A3-1's symptom, and the reason reset_slot zeroes).
            raise RuntimeError(
                f"slot {slot} is at kv_len={self.pool.slot_kv_len[slot]}, but this prefill "
                f"continues from {prefix_hit}; the caller must reset_slot first"
            )
        suffix = prompt_ids[prefix_hit:]
        if not suffix:
            raise ValueError(
                f"prefill for slot {slot} has nothing to compute "
                f"(prefix_hit={prefix_hit} == len(prompt)); the caller must leave "
                "at least one token so there are logits to sample from"
            )
        # Chunked, and NOT optional. A single forward over the whole suffix
        # overflows int32 inside sparkinfer's w4a16 fused MoE: it builds a
        # memref whose element count is `m * fc1_cols`, and with
        # fc1_cols = 2 * intermediate_size = 34816 that exceeds 2**31-1 at
        # m = 61,682. Above that the kernel raises OverflowError deep in the
        # cutlass DSL, `ServerEngine` swallows it, and the client gets HTTP
        # 200 with `finish=stop` and ZERO tokens -- no error anywhere. The
        # model advertises max_context=131072 per slot, so every prompt
        # between ~61.7k and 131k silently returned nothing.
        #
        # This method previously ran one forward for the whole suffix and
        # `prefill_chunked_begin` documented itself as "one-shot", accepting
        # and ignoring the caller's chunk_size. The one-shot *protocol* is
        # kept (still returns done=True in a single round, which is what lets
        # ServerEngine activate slots immediately) -- only the forward is
        # split. Chunking here is exactly what a continuation already does:
        # each call advances `state`, and only the final chunk's rows are
        # returned, because only the last position's logits are sampled from.
        chunk = self._prefill_chunk_tokens()
        hidden = None
        for start in range(0, len(suffix), chunk):
            piece = suffix[start : start + chunk]
            input_ids = torch.tensor([piece], dtype=torch.long, device=self.device)
            hidden = self.model(input_ids, state)
        assert hidden is not None  # `suffix` is non-empty, checked above
        logits = self.model.compute_logits(hidden[0])
        if return_hidden:
            return logits, hidden[0]
        return logits

    def _prefill_chunk_tokens(self) -> int:
        """Tokens per prefill forward: large for speed, capped for correctness.

        The cap is derived from the model's own geometry rather than pinned to
        a constant, so a different `intermediate_size` cannot silently walk
        back into the overflow. `_W4A16_MEMREF_ELEMENT_LIMIT` is int32's
        positive range, which is what the cutlass DSL memref descriptor
        actually holds.

        The preferred size is Laguna's own `_prefill_chunk_tokens` default,
        for consistency; on this model the derived cap (61,681) is far above
        it, so the cap only ever binds for a model with a much wider MLP.
        """
        # `Qwen36ForCausalLMSelfBuilt.config` is the raw checkpoint config
        # dict, not an attribute object -- see that class's `__init__`.
        #
        # Falls back to the preferred size when the model does not expose an
        # `intermediate_size`, which is only ever a test double standing in
        # for the model. Deliberately a fallback rather than a raise: the cap
        # exists to bound a REAL long prefill, and making every stub in the
        # suite carry a field it has no other use for would couple unrelated
        # tests to this bound. A stub that never prefills 61k tokens cannot
        # trip the limit the cap protects against. Any real model built from
        # a checkpoint always has the key.
        config = getattr(self.model, "config", None)
        intermediate = (config or {}).get("intermediate_size") if isinstance(config, dict) else None
        if not intermediate:
            return _PREFERRED_PREFILL_CHUNK_TOKENS
        hard_cap = _W4A16_MEMREF_ELEMENT_LIMIT // (2 * int(intermediate))
        return max(1, min(_PREFERRED_PREFILL_CHUNK_TOKENS, hard_cap))

    def _commit_prefill(self, slot: int, prompt_ids: list[int], first_token: int) -> None:
        self.pool.slot_kv_len[slot] = len(prompt_ids)
        self.pool.slot_committed_tokens[slot] = list(prompt_ids) + [first_token]
        self._maybe_checkpoint(slot)

    # -- protocol: chunked prefill -----------------------------------------

    def prefill_chunked_begin(
        self,
        slots: list[int],
        prompts_per_slot: list[list[int]],
        chunk_size: int = 512,
        *,
        params_per_slot: dict[int, SamplingParams] | None = None,
    ) -> ChunkedPrefillState:
        """Prefill every admitted slot, honouring pending prefix hits.

        One-shot, like ``LagunaBackend``'s non-DFlash path (its own
        ``_prefill_chunk_tokens`` default is 8192 and the engine comment at
        the construction site records prefill as "one-shot"); ``chunk_size``
        is accepted and ignored so the protocol signature holds. Returning
        ``done=True`` immediately is what tells ``ServerEngine`` to activate
        the slots in the same round.

        When MTP is enabled (:meth:`enable_mtp`), also drafts this slot's
        first round of speculative tokens right after prefill -- see
        ``runtime.backends.qwen36_mtp.Qwen36MTPEngine.draft_after_prefill``.
        Without it, ``draft_tokens`` stays ``[]`` (byte-for-byte prior
        behavior) and ``ServerEngine`` routes every round for this slot
        through plain sampled decode (``classify_decode_slots``).
        """
        params_per_slot = params_per_slot or {}
        requested = chunk_size if chunk_size and chunk_size > 0 else self._prefill_chunk_tokens()
        chunk = min(requested, self._prefill_chunk_tokens())

        suffixes: list[list[int]] = []
        for slot, prompt in zip(slots, prompts_per_slot):
            hit = self._apply_prefix_hit(slot, prompt)
            # Both guards moved here from `_prefill_forward`, which the chunked
            # path no longer goes through. Dropping them is not cosmetic: a
            # slot at the wrong kv_len makes the GDN layers continue from
            # ANOTHER sequence's recurrent state, which raises nothing and
            # produces no NaN -- just a wrong continuation (INV-A3-1). That is
            # the exact silent-wrongness this repo keeps being bitten by, and
            # tests/test_qwen36_backend.py's dirty-slot case caught their loss
            # the moment this method stopped calling `_prefill_forward`.
            if self.pool.slot_kv_len[slot] != hit:
                raise RuntimeError(
                    f"slot {slot} is at kv_len={self.pool.slot_kv_len[slot]}, but this prefill "
                    f"continues from {hit}; the caller must reset_slot first"
                )
            suffix = list(prompt[hit:])
            if not suffix:
                raise ValueError(
                    f"prefill for slot {slot} has nothing to compute "
                    f"(prefix_hit={hit} == len(prompt)); the caller must leave "
                    "at least one token so there are logits to sample from"
                )
            suffixes.append(suffix)

        state = ChunkedPrefillState(
            done=False,
            result={},
            slots=list(slots),
            prompts_per_slot=list(prompts_per_slot),
            suffix_per_slot=suffixes,
            chunk_size=chunk,
            chunk_start=0,
            total_len=max((len(s) for s in suffixes), default=0),
            anchors=dict(params_per_slot),
        )
        # Advance once here, so a prompt that fits in one chunk still finishes
        # within this round. ``ServerEngine`` activates slots immediately on
        # done=True, which is what every short request did before and must
        # keep doing -- interleaving is for long prompts, not a new round-trip
        # for short ones.
        self.prefill_chunked_step(state)
        return state

    def prefill_chunked_step(self, state: ChunkedPrefillState) -> bool:
        """Advance every pending slot by one chunk. Returns ``done``.

        A5/B4 -- cross-step interleaved chunked prefill. This used to be a
        no-op returning ``True``, with :meth:`prefill_chunked_begin`
        documenting itself as "one-shot" and discarding the caller's
        ``chunk_size``. That made ``ServerEngine``'s entire incremental branch
        (``self._pending_prefill``, ``server/engine.py`` ~1310) unreachable
        dead code: nothing ever returned ``done=False``, so no prefill was
        ever advanced across rounds, and a long admission monopolised the
        engine until it finished.

        The cost is the largest single item on record. With prefill blocking
        the round, a 128K admission starves every active slot's decode for its
        whole duration -- historically TTFT 25.7s against native's 4.4s, which
        ``notes/2026-07-20-comprehensive-optimization-plan.md`` attributes
        **60-70% of the end-to-end gap** to. The same document records that
        chunking *within* one admission (Phase A) bought only -10.7%: the win
        is in yielding between chunks, not in the chunk size. The historical
        implementation of this state machine is in this repo, at
        ``oracle/qwen36_vllm/direct_model_runner.py:1731-1938``.

        Only the LAST chunk's logits matter -- earlier chunks exist to advance
        the slot's KV and recurrent state -- so anchor sampling,
        :meth:`_commit_prefill` and the MTP first draft all happen on the
        final call for that slot, exactly as they did when this ran in one
        shot. Slots whose prompts differ in length finish on different calls;
        each is committed when it finishes and skipped thereafter.
        """
        params_per_slot = state.anchors or {}
        start = state.chunk_start
        chunk = state.chunk_size
        finished = True

        for slot, prompt, suffix in zip(
            state.slots, state.prompts_per_slot, state.suffix_per_slot
        ):
            if start >= len(suffix):
                continue  # shorter prompt: already committed on an earlier call
            end = min(start + chunk, len(suffix))
            is_last = end >= len(suffix)
            if not is_last:
                finished = False

            input_ids = torch.tensor([suffix[start:end]], dtype=torch.long, device=self.device)
            hidden = self.model(input_ids, self.pool.slot_state(slot))
            if not is_last:
                continue

            logits = self.model.compute_logits(hidden[0])
            params = params_per_slot.get(slot)
            if params is None or params.is_greedy:
                token = int(logits[-1].argmax(dim=-1).item())
            else:
                gen = make_generator(params.seed)
                token = int(
                    sample_from_logits(logits[-1].unsqueeze(0), params, generator=gen).item()
                )
            self._commit_prefill(slot, prompt, token)
            draft_tokens: list[int] = []
            if self._mtp is not None:
                pred_hidden = hidden[0][-1:].unsqueeze(0)  # [1, 1, H]
                draft_tokens = self._mtp.draft_after_prefill(
                    slot, first_token=token, pred_hidden=pred_hidden
                )
            state.result[slot] = {"anchor": token, "draft_tokens": draft_tokens}

        state.chunk_start = start + chunk
        state.done = finished
        return finished

    # -- protocol: decode ---------------------------------------------------

    def decode_batch_sampled(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
        params_list: list[SamplingParams],
        *,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> list[int] | tuple[list[int], list[dict]]:
        """One decode round for every active slot -- continuous batching.

        ``kv_lengths`` is accepted for protocol compatibility and
        cross-checked against this backend's own bookkeeping rather than
        trusted: a scheduler and a backend disagreeing about how long a
        sequence is is exactly the class of bug that shows up as a
        plausible wrong answer rather than an exception.
        """
        if not slot_ids:
            return ([], []) if return_logprobs else []
        for slot, expected in zip(slot_ids, kv_lengths):
            actual = self.pool.slot_kv_len[slot]
            if expected != actual:
                raise RuntimeError(
                    f"slot {slot}: scheduler says kv_len={expected}, backend has {actual}"
                )

        if self.batched_decode:
            logits = self._decode_forward_batched(slot_ids, token_ids)
        else:
            logits = self._decode_forward_serial(slot_ids, token_ids)

        next_tokens: list[int] = []
        for i, params in enumerate(params_list):
            if params.is_greedy:
                next_tokens.append(int(logits[i].argmax(dim=-1).item()))
            else:
                gen = make_generator(params.seed)
                next_tokens.append(
                    int(sample_from_logits(logits[i].unsqueeze(0), params, generator=gen).item())
                )
        for slot in slot_ids:
            self._maybe_checkpoint(slot)

        self.stats["decode_rounds"] += 1
        self.stats["decode_tokens"] += len(slot_ids)

        if return_logprobs:
            lp = [
                compute_logprobs(logits[i].unsqueeze(0), [next_tokens[i]], top_k=top_logprobs)[0]
                for i in range(len(next_tokens))
            ]
            return next_tokens, lp
        return next_tokens

    def _decode_forward_batched(self, slot_ids: list[int], token_ids: list[int]) -> torch.Tensor:
        batch, b = self.pool.build_decode_batch(slot_ids, token_ids)
        graph = self._decode_graphs.get(b)
        if graph is not None:
            graph.replay()
            self.stats["decode_graph_replays"] += 1
            return self._decode_graph_logits[b]
        return self.model.decode_batch(batch)

    def _decode_forward_serial(self, slot_ids: list[int], token_ids: list[int]) -> torch.Tensor:
        """One slot at a time through B1's single-sequence path.

        Kept as a first-class, selectable path (``batched_decode=False``),
        not as dead fallback code: it is the control group the bit-exact
        gate needs. "Batched decode matches eager decode" is only a
        meaningful claim if both can be run against the same slot pool in
        the same process, and the difference between them is exactly one
        constructor keyword.
        """
        rows: list[torch.Tensor] = []
        for slot, token in zip(slot_ids, token_ids):
            state = self.pool.slot_state(slot)
            input_ids = torch.tensor([[token]], dtype=torch.long, device=self.device)
            hidden = self.model(input_ids, state)
            rows.append(self.model.compute_logits(hidden[0])[-1])
            self.pool.slot_kv_len[slot] += 1
            self.pool.slot_committed_tokens[slot].append(int(token))
        return torch.stack(rows, dim=0)

    # -- protocol: MTP speculative decode -----------------------------------

    def mtp_verify_and_commit_batch(
        self,
        slots: list[int],
        anchors: dict[int, int],
        drafts: dict[int, list[int]],
        *,
        params_per_slot: dict[int, SamplingParams] | None = None,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> dict[int, dict]:
        """B3: Qwen3.6's sibling of ``LagunaBackend.mtp_verify_and_commit_batch``
        -- called by the SAME ``ServerEngine._step_sync`` MTP branch
        (``classify_decode_slots`` routes here once ``self.has_speculative_decode``
        is true, i.e. once :meth:`enable_mtp` has been called).

        Sequential per slot, not a fused batched verify -- see
        ``runtime.backends.qwen36_mtp.Qwen36MTPEngine``'s class docstring
        for why (this backend's whole model graph is batch=1 by
        construction; DFlash's own docstring documents the identical
        choice for the same structural reason).

        ``params_per_slot``: optional per-slot ``SamplingParams``, forwarded
        to :meth:`Qwen36MTPEngine.round`. A missing entry (or ``None``
        altogether) takes the greedy path for that slot.
        """
        if self._mtp is None:
            raise RuntimeError("mtp_verify_and_commit_batch called without enable_mtp()")
        return {
            slot: self._mtp.round(
                slot,
                anchors[slot],
                drafts[slot],
                params=(params_per_slot.get(slot) if params_per_slot else None),
                return_logprobs=return_logprobs,
                top_logprobs=top_logprobs,
            )
            for slot in slots
        }

    # -- protocol: prefix cache --------------------------------------------

    def reconcile_prefix_hit(self, token_ids: list[int]) -> PrefixHit:
        """Deepest reusable prefix across all slots, as ``(kv_hit, state_hit)``.

        Structure ported (read, not imported) from
        ``oracle/qwen36_vllm/prefix_cache.py:131-166``'s ``L = G <= A``:
        compute the attention-side match first, then search **downward
        from it** for a recurrent boundary. Two differences from the
        oracle, both forced by this runtime's addressing rather than
        chosen:

        * ``A`` comes from a same-slot linear token comparison, not a
          chained block hash, because this backend uses static per-slot
          pages and never shares KV across slots
          (``docs/a3-cache-coordinator-design.md`` §8 decision 5).
        * The downward search has at most one candidate to consider,
          because the checkpoint policy keeps one rolling boundary per
          slot (module docstring).

        ``A>0, G=0`` is a compute miss, not a partial hit -- the oracle's
        own rule (``prefix_cache.py:135-139``) and the only safe reading
        of :attr:`PrefixHit.effective`.
        """
        if not self.enable_prefix_cache or not token_ids:
            return PrefixHit(kv_hit=0, state_hit=0)
        best = PrefixHit(kv_hit=0, state_hit=0)
        best_slot = -1
        for slot in range(self.num_slots):
            hit = self._prefix_hit_for_slot(token_ids, slot)
            if (hit.effective, hit.kv_hit) > (best.effective, best.kv_hit):
                best = hit
                best_slot = slot
        if best.effective > 0 and best_slot >= 0:
            self._pending_prefix_hits[best_slot] = best
        self.stats["prefix_kv_hit_tokens"] += best.kv_hit
        self.stats["prefix_state_hit_tokens"] += best.state_hit
        if best.state_hit < best.kv_hit:
            # §3 asks for exactly this signal: "state_hit < kv_hit 多久发生
            # 一次、差多少" is what tells Track B whether the checkpoint
            # boundary policy above is too coarse. Recorded here, in the
            # only object that outlives a request, because the coordinator
            # is reconstructed on every access (server/engine.py:507).
            self.stats["prefix_hit_split_events"] += 1
        return best

    def find_best_slot_for_prompt(
        self, token_ids: list[int], free_slots: list[int]
    ) -> tuple[int, int]:
        """``(slot, hit_depth)`` for cache-aware assignment among ``free_slots``.

        ``hit_depth`` is the **effective** (recurrent-bounded) depth, not
        the KV depth: a slot whose KV matches 900 tokens but whose
        recurrent checkpoint only reaches 0 is worth exactly as much as a
        cold slot, and ranking it first would trade a real hit elsewhere
        for an imaginary one here.
        """
        if not free_slots:
            raise ValueError("find_best_slot_for_prompt requires at least one free slot")
        if not self.enable_prefix_cache:
            return free_slots[0], 0
        best_slot = free_slots[0]
        best = PrefixHit(kv_hit=0, state_hit=0)
        for slot in free_slots:
            hit = self._prefix_hit_for_slot(token_ids, slot)
            if (hit.effective, hit.kv_hit) > (best.effective, best.kv_hit):
                best = hit
                best_slot = slot
        return best_slot, best.effective

    def prefix_hit_for_slot(self, token_ids: list[int], slot: int) -> PrefixHit:
        """Per-slot ``(kv_hit, state_hit)`` -- the coordinator's ranking input.

        Public because :class:`runtime.slot_resource_manager.
        SlotResourceManager` needs per-slot numbers to rank free slots by
        ``.effective`` rather than by whatever a backend's own
        ``find_best_slot_for_prompt`` happens to rank by.
        """
        return self._prefix_hit_for_slot(token_ids, slot)

    def _prefix_hit_for_slot(self, token_ids: list[int], slot: int) -> PrefixHit:
        cached = self._prefix_cache_tokens[slot]
        if not self.enable_prefix_cache or cached is None:
            return PrefixHit(kv_hit=0, state_hit=0)
        cached_kv_len = self._prefix_cache_kv_len[slot]
        if cached_kv_len <= 0:
            return PrefixHit(kv_hit=0, state_hit=0)
        prompt_len = len(token_ids)
        limit = min(prompt_len, len(cached), cached_kv_len)
        match_len = 0
        for i in range(limit):
            if token_ids[i] != cached[i]:
                break
            match_len += 1
        kv_hit = (match_len // self.block_size) * self.block_size
        if kv_hit >= prompt_len:
            # Always leave one token to compute, so there are logits to
            # sample the first output from (Laguna's rule, same reason).
            kv_hit = ((prompt_len - 1) // self.block_size) * self.block_size
        if kv_hit <= 0:
            return PrefixHit(kv_hit=0, state_hit=0)

        state_hit = 0
        ckpt_len = self._checkpoint_len.get(slot, 0)
        if 0 < ckpt_len <= kv_hit and slot in self._checkpoint_tensors:
            key = (slot, ckpt_len)
            if key in self.checkpoint_pool:
                expected = self.checkpoint_pool.get_by_hash(_prefix_hash(token_ids, ckpt_len))
                # Identity check, not just length: a checkpoint whose bytes
                # were produced by a *different* prefix of the same length
                # would resume from a state that is wrong in a way nothing
                # downstream can detect. This is the oracle's "a
                # wrong-prefix checkpoint is REJECTED" rule
                # (prefix_cache.py's restore_cached_prefix).
                if expected == key:
                    state_hit = ckpt_len
        return PrefixHit(kv_hit=kv_hit, state_hit=state_hit)

    def _apply_prefix_hit(self, slot: int, prompt_ids: list[int]) -> int:
        """Consume the pending hit for ``slot``; return tokens to skip.

        Restores the recurrent checkpoint **before** returning, so the
        caller's forward starts from a state that genuinely corresponds to
        position ``effective``. Returns 0 (full recompute) if anything
        about the restore does not line up -- a compute miss is always
        available and always correct, which is why every failure here
        degrades to it rather than raising.
        """
        hit = self._pending_prefix_hits.pop(slot, None)
        if hit is None or hit.effective <= 0:
            return 0
        length = hit.effective
        tensors = self._checkpoint_tensors.get(slot)
        if tensors is None or self._checkpoint_len.get(slot) != length:
            return 0
        if self.checkpoint_pool.get_by_hash(_prefix_hash(prompt_ids, length)) != (slot, length):
            return 0
        self.pool.restore_recurrent_state(slot, tensors)
        self.pool.rewind_slot(slot, length)
        self.checkpoint_pool.touch((slot, length))
        return length

    # -- second cache family: checkpoints ----------------------------------

    def _maybe_checkpoint(self, slot: int) -> None:
        """Refresh ``slot``'s rolling checkpoint if it is on a boundary.

        Called after every commit (prefill and each decode token). The
        boundary test is on ``kv_len``, i.e. on positions actually written
        to the KV cache, so the two resources are co-keyed on the same
        number by construction rather than by two call sites agreeing.
        """
        if not self.enable_prefix_cache:
            return
        kv_len = self.pool.slot_kv_len[slot]
        if kv_len <= 0 or kv_len % self.block_size != 0:
            return
        if self._checkpoint_len.get(slot) == kv_len:
            return
        tokens = self.pool.slot_committed_tokens[slot]
        if len(tokens) < kv_len:
            return
        self._evict_checkpoint(slot)
        # Nothing is ever ``pin``ned. INV-A3-4 protects resources with a live
        # reference, and a checkpoint is not one: the live recurrent state is
        # the slot's own pool row, which is not a cache and is not reachable
        # by any eviction path at all. A checkpoint is a *copy* taken at an
        # older boundary, so evicting one while its slot generates costs a
        # future would-be hit and nothing else. Pinning every busy slot's
        # checkpoint would instead make the budget unenforceable exactly when
        # it matters (all slots busy), which is the opposite of a budget.
        self.checkpoint_pool.evict_for_budget(self._checkpoint_bytes)
        key = (slot, kv_len)
        self.checkpoint_pool.register(
            key,
            hash_value=_prefix_hash(tokens, kv_len),
            num_tokens=kv_len,
            nbytes=self._checkpoint_bytes,
        )
        self._checkpoint_tensors[slot] = self.pool.capture_recurrent_state(slot)
        self._checkpoint_len[slot] = kv_len
        self.stats["checkpoints_taken"] += 1

    def _evict_checkpoint(self, slot: int) -> None:
        length = self._checkpoint_len.pop(slot, None)
        self._checkpoint_tensors.pop(slot, None)
        if length is not None:
            self.checkpoint_pool.evict((slot, length))

    def _checkpoint_kv_is_free(self, key: object) -> bool:
        """Reverse-direction lockstep predicate (INV-A3-3).

        Answers "may this checkpoint's eviction also drop the co-keyed KV
        hash?" -- true only when the slot holds no live sequence. A live
        slot keeps its KV: dropping a checkpoint while its KV survives
        "merely turns a future would-be hit into a safe compute miss"
        (``oracle/qwen36_vllm/gdn_state.py:196``), whereas reclaiming a
        live slot's KV would be the symmetric version this asymmetry
        exists to forbid.
        """
        slot = key[0] if isinstance(key, tuple) else key
        return self.pool.slot_kv_len[slot] == 0

    def _drop_kv_for_checkpoint(self, key: object) -> None:
        slot = key[0] if isinstance(key, tuple) else key
        self._checkpoint_tensors.pop(slot, None)
        self._checkpoint_len.pop(slot, None)
        self._prefix_cache_tokens[slot] = None
        self._prefix_cache_kv_len[slot] = 0
        self.stats["checkpoints_evicted_by_budget"] += 1

    # -- protocol: CUDA Graph ----------------------------------------------

    def capture_decode_cuda_graph(self) -> int | None:
        """Capture one decode graph per batch size ``1..num_slots``.

        Returns the largest batch size captured, or ``None`` if capture was
        not possible. Nothing here is best-effort about correctness: a
        failed capture drops the whole set and falls back to eager, because
        a half-captured set would silently serve some batch sizes from a
        graph and others eagerly, which is precisely the "looks right,
        is wrong" shape INV-A3-7 warns about.

        **Every slot is zeroed afterwards, unconditionally.** Capture runs
        real forwards, which write real recurrent state into whichever rows
        they touch; leaving that behind is the non-idempotent-state trap
        B0-5 flagged. This is the same "capture before any real slot use"
        window ``ServerEngine`` already relies on for Laguna, plus the one
        extra step Laguna does not need because it has no recurrent state.
        """
        if self.device.type != "cuda" or not self.batched_decode:
            return None
        try:
            captured_batch = self._capture_decode_graphs()
            self.cg_status["decode"] = "captured" if captured_batch else "failed"
            return captured_batch
        except Exception:  # pragma: no cover - depends on driver/kernel support
            logger.exception("Qwen3.6 decode CUDA Graph capture failed; falling back to eager")
            self.cg_status["decode"] = "failed"
            self._decode_graphs.clear()
            self._decode_graph_logits.clear()
            # Uninstall every graph driver too. Leaving one installed would
            # make the eager path write metadata into a replay-mode
            # workspace whose plan was never captured -- a state that reads
            # as "fell back safely" and computes garbage.
            self.pool.graph_attn.clear()
            return None
        finally:
            self.pool.reset_all()
            for slot in range(self.num_slots):
                self._prefix_cache_tokens[slot] = None
                self._prefix_cache_kv_len[slot] = 0
            self._pending_prefix_hits.clear()

    def _capture_decode_graphs(self) -> int:
        captured = 0
        for b in range(1, self.num_slots + 1):
            slots = list(range(b))
            tokens = [0] * b
            for slot in slots:
                self.pool.reset_slot(slot)
            # Warm eagerly first, at this exact batch size: a kernel's
            # first-ever launch may JIT, and a JIT compile inside a capture
            # is not capturable. This is also what pays the FLA/Triton
            # first-call compile once instead of inside the graph.
            warm_batch, _ = self.pool.build_decode_batch(slots, tokens)
            self.model.decode_batch(warm_batch)
            torch.cuda.synchronize(self.device)

            # Install the graph-replay driver BEFORE building the batch the
            # graph is captured against: it owns the metadata buffers the
            # capture bakes in, and those must be the same buffers every
            # later step writes.
            self.pool.graph_attn[b] = self.pool.build_graph_attention_driver(b)
            for slot in slots:
                self.pool.reset_slot(slot)
            batch, _ = self.pool.build_decode_batch(slots, tokens)
            # One eager run through the graph driver too: sparkinfer's
            # replay-mode kernel has its own first-launch compile, distinct
            # from the eager planner's.
            self.model.decode_batch(batch)
            for slot in slots:
                self.pool.reset_slot(slot)
            batch, _ = self.pool.build_decode_batch(slots, tokens)
            torch.cuda.synchronize(self.device)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._graph_pool):
                logits = self.model.decode_batch(batch)
            if self._graph_pool is None:
                self._graph_pool = graph.pool()
            self._decode_graphs[b] = graph
            self._decode_graph_logits[b] = logits
            captured = b
        return captured
