"""A long prefill must yield between chunks, not monopolise the engine round.

`ServerEngine` has a complete incremental-prefill state machine: it keeps
`self._pending_prefill`, advances it one chunk per round via
`prefill_chunked_step`, and only activates slots when that returns `done`
(`server/engine.py` ~1310). Decode rounds for already-active slots run in the
same `_step`, so a prefill that returns `done=False` interleaves with them
automatically.

None of it ran. `Qwen36Backend.prefill_chunked_begin` documented itself as
"one-shot", discarded the caller's `chunk_size`, and returned `done=True`
always; `prefill_chunked_step` was `return True`. So the engine branch was
unreachable dead code, and every admission blocked the round for its whole
duration -- `server/engine.py:607` even carried the comment "unused:
Qwen36Backend prefill is one-shot".

The cost is the largest single item on record for this runtime. A 128K
admission starves every active slot's decode while it runs: historically TTFT
25.7s against native's 4.4s, which
`notes/2026-07-20-comprehensive-optimization-plan.md` attributes **60-70% of
the end-to-end gap** to. That document also records that chunking *within* one
admission bought only -10.7% -- the win is specifically in yielding between
chunks, which is the property these tests pin.

Why a permanent gate: nothing else would notice a regression to one-shot.
Output is identical either way -- same tokens, same anchor, same drafts -- and
the only symptom is that other requests wait. A throughput test would need
concurrency plus a long prompt plus a GPU to see it; these assert the
structural property directly, on a fake model, in milliseconds.

CPU-only: a fake model that records the shape of each forward. No GPU, no
checkpoint.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fla")
pytest.importorskip("sparkinfer")

from runtime.backends.qwen36 import Qwen36Backend  # noqa: E402
from runtime.sampling import SamplingParams  # noqa: E402

_VOCAB = 64
_HEAD_DIM = 4


class _RecordingModel:
    """Records the token count of every forward; hidden value == token id."""

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    layer_idx=0,
                    layer_type="full_attention",
                    linear_attn=None,
                    self_attn=SimpleNamespace(num_kv_heads=2, head_dim=_HEAD_DIM, num_heads=4),
                ),
                SimpleNamespace(
                    layer_idx=1,
                    layer_type="linear_attention",
                    linear_attn=SimpleNamespace(
                        conv_dim=8,
                        conv_kernel_size=4,
                        num_v_heads=2,
                        head_k_dim=_HEAD_DIM,
                        head_v_dim=_HEAD_DIM,
                    ),
                    self_attn=None,
                ),
            ]
        )
        self.mtp = None
        self.config = {"vocab_size": _VOCAB, "intermediate_size": 17408}
        self.forwards: list[int] = []

    def __call__(self, input_ids, state):
        n = int(input_ids.shape[1])
        self.forwards.append(n)
        state.num_tokens_seen += n
        for cache in state.attn_caches:
            if cache is not None:
                cache.seq_len += n
        return input_ids.to(torch.float32).unsqueeze(-1)

    def compute_logits(self, hidden):
        out = torch.zeros(*hidden.shape[:-1], _VOCAB)
        flat_h, flat_o = hidden.reshape(-1), out.reshape(-1, _VOCAB)
        for i in range(flat_h.shape[0]):
            flat_o[i, (int(flat_h[i].item()) + 1) % _VOCAB] = 1.0
        return out


def _backend() -> tuple[Qwen36Backend, _RecordingModel]:
    model = _RecordingModel()
    backend = Qwen36Backend(
        model,
        num_slots=3,
        max_seq_len=4096,
        block_size=64,
        device="cpu",
        dtype=torch.float32,
        enable_prefix_cache=False,
    )
    return backend, model


class TestItYieldsBetweenChunks:
    def test_a_long_prompt_is_not_finished_in_one_round(self):
        """The property the whole change exists for: done=False, so the engine
        gets the round back and can run decode for other slots."""
        backend, _ = _backend()
        state = backend.prefill_chunked_begin(
            [0], [list(range(2000))], chunk_size=512, params_per_slot={0: SamplingParams()}
        )
        assert state.done is False, (
            "a 2000-token prompt finished in one round -- prefill has regressed "
            "to one-shot and ServerEngine's incremental branch is dead again"
        )

    def test_it_completes_across_successive_steps(self):
        backend, model = _backend()
        state = backend.prefill_chunked_begin(
            [0], [list(range(2000))], chunk_size=512, params_per_slot={0: SamplingParams()}
        )
        guard = 0
        while not state.done:
            backend.prefill_chunked_step(state)
            guard += 1
            assert guard < 100, "prefill did not converge"
        # Count the model's own forwards, not loop iterations -- `begin`
        # already performed one, so a loop counter is off by one and would
        # pin the wrong number.
        assert model.forwards == [512, 512, 512, 464], (
            f"2000 tokens at chunk 512 should be four forwards, got {model.forwards}"
        )
        assert sum(model.forwards) == 2000, "chunking must not drop or duplicate tokens"

    def test_a_short_prompt_still_finishes_immediately(self):
        """Interleaving is for long prompts. A short one must not pay an extra
        round-trip -- ServerEngine activates on done=True in the same round."""
        backend, model = _backend()
        state = backend.prefill_chunked_begin(
            [0], [list(range(100))], chunk_size=512, params_per_slot={0: SamplingParams()}
        )
        assert state.done is True
        assert model.forwards == [100]
        assert 0 in state.result

    def test_the_anchor_comes_from_the_last_chunk(self):
        """Only the final chunk's logits are sampled -- earlier chunks exist to
        advance KV/recurrent state. Getting this wrong samples the anchor from
        the middle of the prompt and is invisible in the shapes."""
        backend, _ = _backend()
        prompt = list(range(1000))
        state = backend.prefill_chunked_begin(
            [0], [prompt], chunk_size=512, params_per_slot={0: SamplingParams()}
        )
        while not state.done:
            backend.prefill_chunked_step(state)
        # `compute_logits` puts the argmax at (hidden + 1) % vocab, and hidden
        # == the token id, so the anchor must derive from the LAST token.
        assert state.result[0]["anchor"] == (prompt[-1] + 1) % _VOCAB


class TestRaggedAndGuards:
    def test_slots_with_different_lengths_each_finish_when_ready(self):
        backend, _ = _backend()
        state = backend.prefill_chunked_begin(
            [0, 1],
            [list(range(600)), list(range(1600))],
            chunk_size=512,
            params_per_slot={0: SamplingParams(), 1: SamplingParams()},
        )
        assert state.done is False
        n = 1
        while not backend.prefill_chunked_step(state):
            n += 1
            assert n < 50
        assert set(state.result) == {0, 1}, "both slots must be committed exactly once"

    def test_a_dirty_slot_is_still_refused(self):
        """This guard lived in `_prefill_forward`, which the chunked path no
        longer calls. Losing it lets GDN continue from another sequence's
        recurrent state: no exception, no NaN, just a wrong continuation
        (INV-A3-1). It was dropped in the first version of this change and
        caught only because tests/test_qwen36_backend.py already pinned it."""
        backend, _ = _backend()
        backend.pool.slot_kv_len[0] = 7
        with pytest.raises(RuntimeError, match="reset_slot"):
            backend.prefill_chunked_begin(
                [0], [list(range(100))], chunk_size=512, params_per_slot={0: SamplingParams()}
            )

    def test_a_fully_cached_prompt_is_refused(self):
        """Nothing to compute means no logits to sample an anchor from."""
        backend, _ = _backend()
        prompt = list(range(100))
        backend.pool.slot_kv_len[0] = 0
        with pytest.raises(ValueError, match="nothing to compute"):
            backend.prefill_chunked_begin(
                [0], [[]], chunk_size=512, params_per_slot={0: SamplingParams()}
            )
        del prompt
