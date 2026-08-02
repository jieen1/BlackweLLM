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

Deliberately out of scope, stated rather than implied
------------------------------------------------------
* **Speculative decoding / MTP.** ``capabilities.speculative_decode`` is
  ``False``; B3 owns it. ``runtime/recurrent_state_pool.py``'s ``spec_row``
  addressing is built and tested for it but nothing here calls it.
* **Warm continue.** ``capabilities.warm_continue`` is ``False`` -- the same
  honest ``False`` that ``protocol.py``'s docstring says the Laguna path
  should have been carrying all along (N8).
* **Cross-slot KV sharing.** Static per-slot pages, per §8 decision 5.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

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
            self.pool.ensure_decode_workspaces(max_batch=num_slots)
        # else: sparkinfer's paged workspaces are CUDA-only, so there is
        # nothing to build. A CPU-device backend is not a fallback execution
        # path -- any forward will fail inside the kernel, loudly. It exists
        # so the slot/prefix/checkpoint bookkeeping below, which is pure
        # Python and is where the silent-corruption failure modes live
        # (INV-A3-1/2/3), can be tested deterministically against a stub
        # model instead of only against a 50 GiB checkpoint on a contended
        # GPU. See tests/test_qwen36_backend.py.

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

        self.pool.reset_all()

    # -- protocol: always required ----------------------------------------

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            speculative_decode=False,
            prefix_cache=self.enable_prefix_cache,
            cuda_graph=True,
            chunked_prefill=True,
            warm_continue=False,
        )

    @property
    def has_speculative_decode(self) -> bool:
        return False

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
        return BackendSnapshot(slots=slots, prefix=prefix, dflash_cg_status=())

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

    def prefill_sampled(
        self, slot: int, prompt_ids: list[int], params: SamplingParams
    ) -> int:
        logits = self._prefill_forward(slot, prompt_ids, prefix_hit=0)
        gen = make_generator(params.seed)
        first_token = int(sample_from_logits(logits[-1].unsqueeze(0), params, generator=gen).item())
        self._commit_prefill(slot, prompt_ids, first_token)
        return first_token

    def _prefill_forward(
        self, slot: int, prompt_ids: list[int], *, prefix_hit: int
    ) -> torch.Tensor:
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
        """
        state = self.pool.slot_state(slot)
        suffix = prompt_ids[prefix_hit:]
        if not suffix:
            raise ValueError(
                f"prefill for slot {slot} has nothing to compute "
                f"(prefix_hit={prefix_hit} == len(prompt)); the caller must leave "
                "at least one token so there are logits to sample from"
            )
        input_ids = torch.tensor([suffix], dtype=torch.long, device=self.device)
        hidden = self.model(input_ids, state)
        return self.model.compute_logits(hidden[0])

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
        """
        params_per_slot = params_per_slot or {}
        result: dict[int, dict] = {}
        for slot, prompt in zip(slots, prompts_per_slot):
            hit = self._apply_prefix_hit(slot, prompt)
            logits = self._prefill_forward(slot, prompt, prefix_hit=hit)
            params = params_per_slot.get(slot)
            if params is None or params.is_greedy:
                token = int(logits[-1].argmax(dim=-1).item())
            else:
                gen = make_generator(params.seed)
                token = int(
                    sample_from_logits(logits[-1].unsqueeze(0), params, generator=gen).item()
                )
            self._commit_prefill(slot, prompt, token)
            result[slot] = {"anchor": token, "draft_tokens": []}
        return ChunkedPrefillState(
            done=True,
            result=result,
            slots=list(slots),
            prompts_per_slot=list(prompts_per_slot),
            chunk_size=chunk_size,
        )

    def prefill_chunked_step(self, state: ChunkedPrefillState) -> bool:
        """No-op: :meth:`prefill_chunked_begin` always finishes in one go."""
        return True

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

    def _decode_forward_batched(
        self, slot_ids: list[int], token_ids: list[int]
    ) -> torch.Tensor:
        batch, b = self.pool.build_decode_batch(slot_ids, token_ids)
        graph = self._decode_graphs.get(b)
        if graph is not None:
            graph.replay()
            self.stats["decode_graph_replays"] += 1
            return self._decode_graph_logits[b]
        return self.model.decode_batch(batch)

    def _decode_forward_serial(
        self, slot_ids: list[int], token_ids: list[int]
    ) -> torch.Tensor:
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
            return self._capture_decode_graphs()
        except Exception:  # pragma: no cover - depends on driver/kernel support
            logger.exception("Qwen3.6 decode CUDA Graph capture failed; falling back to eager")
            self._decode_graphs.clear()
            self._decode_graph_logits.clear()
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
            # Warm the shapes eagerly first: a kernel's first-ever launch
            # may JIT, and JIT inside a capture is not capturable.
            warm_batch, _ = self.pool.build_decode_batch(slots, tokens)
            self.model.decode_batch(warm_batch)
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
