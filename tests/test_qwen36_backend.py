"""Track B / B2: :class:`runtime.backends.qwen36.Qwen36Backend`.

Two kinds of claim live in this file, and they are kept apart on purpose:

* **Contract shape** -- ``check_conformance`` against ``ModelBackend``.
  Mechanical, no model needed.
* **Slot / prefix-cache / checkpoint bookkeeping** -- run against a stub
  model on CPU. This is where the two-cache-family invariants live, and
  every one of them fails *silently* when broken (INV-A3-1/2/3: "不是崩溃
  ——是某个请求的输出因为另一个请求的写入而改变"). Pinning them to a
  deterministic fake is the only way to get a red light out of them at
  all; a GPU run of a 27B checkpoint would report the same bug as slightly
  worse output quality, if at all.

What is deliberately NOT here: anything about numerics. "Batched decode is
bit-exact against B1's eager path", "the CUDA Graph replays what eager
computes", "concurrency >= 2 actually runs" are claims about a real
checkpoint on a real GPU and are made by ``scripts/b2_verify_serving.py``,
not faked here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fla")
pytest.importorskip("sparkinfer")

from runtime.backends.protocol import (  # noqa: E402
    BackendCapabilities,
    PrefixHit,
    check_conformance,
)
from runtime.backends.qwen36 import Qwen36Backend  # noqa: E402
from runtime.sampling import SamplingParams  # noqa: E402

_VOCAB = 32
_CONV_DIM = 8
_CONV_K = 4
_V_HEADS = 2
_HEAD_DIM = 4


class _StubModel:
    """A model-shaped object that advances state the way the real graph does.

    It reproduces the two side effects ``Qwen36Backend`` depends on and
    would otherwise be silently assuming: ``state.num_tokens_seen`` and
    each attention cache's ``seq_len`` advance by the number of tokens
    forwarded. Logits are a deterministic function of the last input token
    so a test can assert *which* token was sampled without pretending to
    model anything.
    """

    def __init__(self, layer_types: list[str]) -> None:
        layers = []
        for i, kind in enumerate(layer_types):
            if kind == "linear_attention":
                linear_attn = SimpleNamespace(
                    conv_dim=_CONV_DIM,
                    conv_kernel_size=_CONV_K,
                    num_v_heads=_V_HEADS,
                    head_k_dim=_HEAD_DIM,
                    head_v_dim=_HEAD_DIM,
                )
                self_attn = None
            else:
                linear_attn = None
                self_attn = SimpleNamespace(num_kv_heads=2, head_dim=_HEAD_DIM, num_heads=4)
            layers.append(
                SimpleNamespace(
                    layer_idx=i, layer_type=kind, linear_attn=linear_attn, self_attn=self_attn
                )
            )
        self.model = SimpleNamespace(layers=layers)
        self.forward_lengths: list[int] = []

    def __call__(self, input_ids, state):
        seq_len = int(input_ids.shape[1])
        self.forward_lengths.append(seq_len)
        state.num_tokens_seen += seq_len
        for cache in state.attn_caches:
            if cache is not None:
                cache.seq_len += seq_len
        return input_ids.to(torch.float32).unsqueeze(-1)  # [1, seq, 1]

    def decode_batch(self, batch):
        # The batched path's bookkeeping is advanced by the pool before the
        # forward, so this only has to produce logits -- deliberately by the
        # same rule as compute_logits, so a test can compare the two decode
        # paths' *token* choices without the stub encoding an opinion.
        self.forward_lengths.append(int(batch.input_ids.shape[0]))
        out = torch.zeros(batch.input_ids.shape[0], _VOCAB)
        for i, tok in enumerate(batch.input_ids[:, 0].tolist()):
            out[i, (int(tok) + 1) % _VOCAB] = 1.0
        return out

    def compute_logits(self, hidden):
        # hidden is [seq, 1]; produce a one-hot-ish row per position whose
        # argmax is (last_token + 1) % vocab -- deterministic and distinct.
        seq = hidden.shape[0]
        out = torch.zeros(seq, _VOCAB)
        for i in range(seq):
            out[i, (int(hidden[i, 0]) + 1) % _VOCAB] = 1.0
        return out


def _backend(num_slots: int = 3, block_size: int = 64, **kw) -> Qwen36Backend:
    model = _StubModel(["full_attention", "linear_attention"])
    return Qwen36Backend(
        model,
        num_slots=num_slots,
        max_seq_len=512,
        block_size=block_size,
        device="cpu",
        dtype=torch.float32,
        **kw,
    )


def _run(backend: Qwen36Backend, slot: int, prompt: list[int], steps: int) -> list[int]:
    """Prefill + ``steps`` greedy decode tokens through the public API."""
    params = SamplingParams()
    state = backend.prefill_chunked_begin([slot], [prompt], params_per_slot={})
    token = state.result[slot]["anchor"]
    out = [token]
    for _ in range(steps):
        token = backend.decode_batch_sampled(
            [slot], [token], [backend.slot_state(slot).kv_len], [params]
        )[0]
        out.append(token)
    return out


class TestContractShape:
    def test_conforms_to_the_model_backend_protocol(self) -> None:
        caps = BackendCapabilities(
            speculative_decode=False,
            prefix_cache=True,
            cuda_graph=True,
            chunked_prefill=True,
            warm_continue=False,
        )
        assert check_conformance(Qwen36Backend, caps) == []

    def test_capabilities_are_honest_about_what_is_not_implemented(self) -> None:
        backend = _backend()
        caps = backend.capabilities
        # protocol.py's own docstring (N8) is about a capability claimed by
        # silence and swallowed by try/except for three years. These two are
        # False on purpose, and B3 is where they change.
        assert caps.speculative_decode is False
        assert caps.warm_continue is False
        assert backend.has_speculative_decode is False

    def test_page_size_must_be_a_multiple_of_block_size(self) -> None:
        # §1.7: the divisibility that holds today holds by coincidence of two
        # independently chosen defaults, and must be checked rather than
        # assumed the moment a checkpoint-boundary policy depends on it.
        with pytest.raises(ValueError, match="multiple of"):
            _backend(block_size=48)


class TestSlotLifecycle:
    def test_fresh_backend_reports_every_slot_fresh(self) -> None:
        backend = _backend()
        assert all(backend.slot_state(s).is_fresh for s in range(3))
        assert all(snap.is_fresh for snap in backend.snapshot().slots)

    def test_prefill_then_decode_advances_kv_len_by_one_per_token(self) -> None:
        backend = _backend()
        _run(backend, 0, [1, 2, 3, 4], steps=3)
        assert backend.slot_state(0).kv_len == 4 + 3

    def test_reset_zeroes_recurrent_state_of_that_slot_only(self) -> None:
        backend = _backend()
        _run(backend, 0, [1, 2, 3], steps=1)
        _run(backend, 1, [9, 9, 9], steps=1)
        for gdn in backend.pool.slot_state(0).gdn_states:
            if gdn is not None:
                gdn.recurrent_state.fill_(3.0)
        for gdn in backend.pool.slot_state(1).gdn_states:
            if gdn is not None:
                gdn.recurrent_state.fill_(4.0)
        backend.reset_slot(0)
        assert torch.all(backend.pool.recurrent_pools[1][0] == 0.0)
        assert torch.all(backend.pool.recurrent_pools[1][1] == 4.0)

    def test_reset_preserves_the_prefix_cache_and_double_reset_does_not_clear_it(self) -> None:
        backend = _backend()
        _run(backend, 0, [1, 2, 3], steps=1)
        backend.reset_slot(0)
        saved = list(backend._prefix_cache_tokens[0] or [])
        assert saved
        backend.reset_slot(0)  # admission-time second reset
        assert backend._prefix_cache_tokens[0] == saved

    def test_decode_refuses_a_scheduler_kv_length_it_disagrees_with(self) -> None:
        backend = _backend()
        _run(backend, 0, [1, 2, 3], steps=0)
        with pytest.raises(RuntimeError, match="scheduler says"):
            backend.decode_batch_sampled([0], [7], [999], [SamplingParams()])

    def test_empty_decode_round_is_a_no_op(self) -> None:
        assert _backend().decode_batch_sampled([], [], [], []) == []


class TestPrefixCacheTwoFamilies:
    def test_cold_backend_reports_no_hit(self) -> None:
        assert _backend().reconcile_prefix_hit([1, 2, 3]) == PrefixHit(kv_hit=0, state_hit=0)

    def test_disabled_prefix_cache_reports_no_hit_and_takes_no_checkpoint(self) -> None:
        backend = _backend(enable_prefix_cache=False)
        _run(backend, 0, list(range(70)), steps=0)
        backend.reset_slot(0)
        assert backend.reconcile_prefix_hit(list(range(70))) == PrefixHit(kv_hit=0, state_hit=0)
        assert backend.stats["checkpoints_taken"] == 0
        assert backend.capabilities.prefix_cache is False

    def test_kv_hits_without_a_checkpoint_are_a_compute_miss_not_a_partial_hit(self) -> None:
        # The oracle's own rule (oracle/qwen36_vllm/prefix_cache.py:135-139):
        # A>0, G=0 is a miss. Using kv_hit here is the INV-A3-2 violation --
        # it does not crash, it makes the GDN layers resume from a state that
        # is stale for [state_hit, kv_hit).
        backend = _backend(block_size=64)
        prompt = list(range(100))
        _run(backend, 0, prompt, steps=0)  # kv_len=100, no 64-boundary crossed after prefill?
        backend._evict_checkpoint(0)  # force the "KV is there, state is not" case
        backend.reset_slot(0)
        hit = backend.reconcile_prefix_hit(prompt + [999])
        assert hit.kv_hit == 64
        assert hit.state_hit == 0
        assert hit.effective == 0
        assert backend.stats["prefix_hit_split_events"] == 1

    def test_a_checkpointed_boundary_becomes_a_real_state_hit(self) -> None:
        backend = _backend(block_size=64)
        prompt = list(range(64))  # prefill lands exactly on a boundary
        _run(backend, 0, prompt, steps=0)
        assert backend.stats["checkpoints_taken"] == 1
        backend.reset_slot(0)
        hit = backend.reconcile_prefix_hit(prompt + [777, 778])
        assert hit.kv_hit == 64
        assert hit.state_hit == 64
        assert hit.effective == 64

    def test_a_hit_actually_shortens_the_forward(self) -> None:
        backend = _backend(block_size=64)
        prompt = list(range(64))
        _run(backend, 0, prompt, steps=0)
        backend.reset_slot(0)
        follow_up = prompt + [777, 778]
        backend.reconcile_prefix_hit(follow_up)  # populates the pending side table
        backend.model.forward_lengths.clear()
        backend.prefill_chunked_begin([0], [follow_up])
        # Only the two novel tokens are forwarded, not all 66.
        assert backend.model.forward_lengths == [2]
        assert backend.slot_state(0).kv_len == 66

    def test_a_checkpoint_from_a_different_prefix_of_the_same_length_is_rejected(self) -> None:
        # Length agreement is not identity. A checkpoint produced by other
        # tokens resumes from a state that is wrong in a way nothing
        # downstream can detect.
        backend = _backend(block_size=64)
        _run(backend, 0, list(range(64)), steps=0)
        backend.reset_slot(0)
        impostor = [5] * 64 + [1]
        hit = backend.reconcile_prefix_hit(impostor)
        assert hit.state_hit == 0

    def test_find_best_slot_prefers_the_resumable_slot_over_the_deeper_kv_one(self) -> None:
        backend = _backend(num_slots=3, block_size=64)
        # slot 0: long KV match, checkpoint deliberately dropped.
        long_prompt = list(range(200))
        _run(backend, 0, long_prompt, steps=0)
        backend.reset_slot(0)
        backend._evict_checkpoint(0)
        # slot 1: shorter match, checkpoint intact.
        _run(backend, 1, long_prompt[:64], steps=0)
        backend.reset_slot(1)
        slot, depth = backend.find_best_slot_for_prompt(long_prompt + [1], [0, 1, 2])
        assert (slot, depth) == (1, 64)

    def test_reconcile_records_the_split_signal_the_design_asks_for(self) -> None:
        backend = _backend(block_size=64)
        _run(backend, 0, list(range(200)), steps=0)
        backend.reset_slot(0)
        backend._evict_checkpoint(0)
        before = backend.stats["prefix_hit_split_events"]
        backend.reconcile_prefix_hit(list(range(200)) + [1])
        assert backend.stats["prefix_hit_split_events"] == before + 1
        assert backend.stats["prefix_kv_hit_tokens"] > backend.stats["prefix_state_hit_tokens"]


class TestCheckpointLockstep:
    def test_dropping_the_kv_prefix_cascades_into_the_checkpoint(self) -> None:
        # INV-A3-3 forward direction, unconditional: the KV side has decided
        # those bytes no longer describe the tokens it thought they did, so a
        # checkpoint keyed to them can only produce a wrong resume.
        backend = _backend(block_size=64)
        prompt = list(range(64))
        _run(backend, 0, prompt, steps=0)
        backend.reset_slot(0)
        assert (0, 64) in backend.checkpoint_pool
        backend.drop_prefix_cache(0)
        assert (0, 64) not in backend.checkpoint_pool
        assert backend.reconcile_prefix_hit(prompt + [1]) == PrefixHit(kv_hit=0, state_hit=0)
        assert backend.stats["checkpoints_evicted_by_kv"] == 1

    def test_budget_pressure_evicts_the_oldest_checkpoint_of_an_idle_slot(self) -> None:
        # Reverse direction of INV-A3-3, idle case: the co-keyed KV carries
        # no live reference, so dropping its hash alongside the checkpoint is
        # allowed (oracle/qwen36_vllm/gdn_state.py:205-209's `ref_cnt == 0`
        # branch). What is NOT allowed is reclaiming *live* KV -- the next
        # test covers that side.
        backend = _backend(num_slots=3, block_size=64)
        prompts = [list(range(64)), list(range(100, 164)), list(range(200, 264))]
        for slot, prompt in enumerate(prompts):
            _run(backend, slot, prompt, steps=0)
            backend.reset_slot(slot)
        # Budget is 2 checkpoints (DEFAULT_CHECKPOINT_BUDGET_MULTIPLE); the
        # third registration must have pushed the first one out.
        assert len(backend.checkpoint_pool) == 2
        assert (0, 64) not in backend.checkpoint_pool
        assert backend.stats["checkpoints_evicted_by_budget"] == 1
        assert backend.reconcile_prefix_hit(prompts[0] + [1]) == PrefixHit(kv_hit=0, state_hit=0)
        # The two younger slots are untouched.
        assert backend.reconcile_prefix_hit(prompts[2] + [1]).state_hit == 64

    def test_budget_pressure_never_touches_a_live_slots_kv(self) -> None:
        # Reverse direction, live case: "losing only the checkpoint ... merely
        # turns a future would-be hit into a safe compute miss (L = G <= A
        # still holds)" -- gdn_state.py:196. Slot 0 is mid-generation when its
        # checkpoint is evicted, so its KV must survive and later show up as
        # the kv_hit > state_hit split the design asks to be observable.
        backend = _backend(num_slots=3, block_size=64)
        prompts = [list(range(64)), list(range(100, 164)), list(range(200, 264))]
        for slot, prompt in enumerate(prompts):
            _run(backend, slot, prompt, steps=0)  # every slot stays live
        assert (0, 64) not in backend.checkpoint_pool
        assert backend.slot_state(0).kv_len == 64
        backend.reset_slot(0)
        hit = backend.reconcile_prefix_hit(prompts[0] + [1])
        assert hit.kv_hit == 64
        assert hit.state_hit == 0

    def test_a_live_slots_checkpoint_is_not_chosen_by_budget_eviction(self) -> None:
        # INV-A3-4: a resource with a live reference is never evicted by
        # either allocator. Slot 0 is mid-generation here.
        backend = _backend(num_slots=3, block_size=64)
        _run(backend, 0, list(range(64)), steps=0)
        assert backend.checkpoint_pool.is_pinned((0, 64)) is False
        # Being live is expressed as kv_len > 0; the reverse-lockstep
        # predicate must refuse to touch such a slot's KV.
        assert backend._checkpoint_kv_is_free((0, 64)) is False
        backend.reset_slot(0)
        assert backend._checkpoint_kv_is_free((0, 64)) is True

    def test_a_slots_checkpoint_rolls_forward_rather_than_accumulating(self) -> None:
        backend = _backend(num_slots=1, block_size=64)
        _run(backend, 0, list(range(64)), steps=64)
        # Two boundaries crossed (64 and 128), one checkpoint retained.
        assert backend.stats["checkpoints_taken"] == 2
        assert len(backend.checkpoint_pool) == 1
        assert backend._checkpoint_len[0] == 128


class TestObservability:
    def test_snapshot_covers_every_slot_and_holds_values_not_references(self) -> None:
        backend = _backend(num_slots=3)
        _run(backend, 1, [1, 2, 3], steps=1)
        snap = backend.snapshot()
        assert len(snap.slots) == 3
        assert len(snap.prefix) == 3
        assert snap.dflash_cg_status == ()
        assert snap.slots[1].kv_len == 4
        # Frozen values: mutating the backend afterwards must not change it.
        _run(backend, 2, [7, 7], steps=0)
        assert snap.slots[2].kv_len == 0
