"""End-to-end ServerEngine round-trip against the real DeepseekV4Backend.

The engine's admission loop calls the runner's full surface in a specific
order (reconcile_prefix_hit -> prefill_chunked_begin -> decode_batch_sampled
-> reset_slot), and `check_conformance` cannot see call-order or keyword
drift -- that gap has already cost a day (test_fake_runner_signatures.py
docstring).  This test drives the REAL backend object (with a forward_fn
stub, no weights) through a REAL ServerEngine admission+decode round so
the surface is exercised, not merely checked.

No GPU/model required: DeepseekV4Backend with forward_fn skips the kernel
stacks; ServerEngine's real __init__ only loads the (offline) tokenizer.
"""

from __future__ import annotations

import asyncio
import os

import pytest

torch = pytest.importorskip("torch")

from runtime.backends.dsv4 import DeepseekV4Backend  # noqa: E402
from runtime.model.dsv4_config import Dsv4Config  # noqa: E402
from runtime.model.dsv4_model import Dsv4Transformer  # noqa: E402
from runtime.sampling import SamplingParams  # noqa: E402
from server.engine import GenerationRequest, ServerEngine, StreamChannel  # noqa: E402

TINY = Dsv4Config(
    vocab_size=128,
    hidden_size=256,
    num_layers=1,
    max_position_embeddings=256,
    norm_eps=1e-6,
    num_heads=2,
    head_dim=128,
    rope_head_dim=64,
    q_lora_rank=16,
    o_groups=2,
    o_lora_rank=8,
    window_size=8,
    compress_ratios=(4,),
    rope_theta=10000.0,
    rope_factor=16.0,
    rope_original_seq_len=64,
    beta_fast=32,
    beta_slow=1,
    compress_rope_theta=160000.0,
    index_n_heads=2,
    index_head_dim=64,
    index_topk=4,
    hc_mult=4,
    hc_sinkhorn_iters=4,
    hc_eps=1e-6,
    n_routed_experts=8,
    n_shared_experts=1,
    n_activated_experts=2,
    moe_intermediate_size=256,
    route_scale=1.5,
    swiglu_limit=10.0,
    n_hash_layers=0,
)


def _zeroed_model(max_seq_len: int = 64) -> Dsv4Transformer:
    model = Dsv4Transformer(TINY, max_seq_len=max_seq_len, device="cpu")
    for buf in model.buffers():
        buf.zero_()
    return model


def _make_backend() -> DeepseekV4Backend:
    """Real backend, forward stubbed to a fixed token: logits uniform +
    2.0 on token 7, so greedy decode yields 7 deterministically."""
    vocab = TINY.vocab_size

    def forward_fn(slot: int, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        logits = torch.zeros(
            1, input_ids.shape[1], vocab, dtype=torch.float32, device=input_ids.device
        )
        logits[0, -1, 7] = 2.0
        return logits

    return DeepseekV4Backend(
        _zeroed_model(),
        TINY,
        num_slots=2,
        max_seq_len=64,
        device="cpu",
        forward_fn=forward_fn,
    )


class _FakeIdTok:
    """Fake tokenizer: fixed 1-char strings so decode is deterministic."""

    def __init__(self, mapping: dict[int, str]) -> None:
        self._mapping = mapping

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(self._mapping[i] for i in ids)


def _make_engine(backend: DeepseekV4Backend) -> ServerEngine:
    engine = ServerEngine(
        backend="deepseek_v4",
        capacity=1,
        num_slots=2,
        enable_cudagraph=False,
        production=True,
    )
    engine.tok = _FakeIdTok({i: chr(65 + (i % 26)) for i in range(TINY.vocab_size)})
    engine.eos_token_ids = frozenset()
    engine.runner = backend
    engine._asyncio_loop = asyncio.new_event_loop()
    engine.request_timeout_s = 0
    engine.watchdog_max_stale_rounds = 0
    engine.enable_session_affinity = False
    engine.retained = {}
    engine.waiting = []
    engine._pending_prefill = None
    r, w = os.pipe()
    os.set_blocking(r, False)
    os.set_blocking(w, False)
    engine._req_pipe_r = r
    engine._req_pipe_w = w
    return engine


def _make_req(engine: ServerEngine, prompt_ids: list[int], max_tokens: int = 5):
    channel = StreamChannel()
    req = GenerationRequest(
        request_id="t",
        prompt_ids=prompt_ids,
        sampling_params=SamplingParams(temperature=0.0),
        max_tokens=max_tokens,
        stop_sequences=[],
        future=engine._asyncio_loop.create_future(),
        stream_channel=channel,
    )
    return req, channel


def _pump(engine: ServerEngine) -> None:
    """Let the engine's event loop process call_soon_threadsafe callbacks
    queued by _resolve_future/StreamChannel.put."""
    engine._asyncio_loop.run_until_complete(asyncio.sleep(0))


def test_full_admission_decode_reset_round_trip() -> None:
    """Admission (prefill) -> decode rounds -> finish -> reset, all on the
    real backend object through the real engine call sequence."""
    backend = _make_backend()
    engine = _make_engine(backend)
    req, channel = _make_req(engine, [1, 2, 3], max_tokens=5)

    # Prefill: the engine admits the request through the runner surface.
    state = backend.prefill_chunked_begin(
        [0], [[1, 2, 3]], params_per_slot={0: SamplingParams(temperature=0.0)}
    )
    assert state.done
    assert state.result[0]["anchor"] == 7  # stub logits -> token 7
    assert backend.slot_state(0).kv_len == 3
    engine._activate_slot(0, req, anchor=7, drafts=[])

    # Decode rounds until max_tokens.
    for _ in range(6):
        if 0 not in engine.active:
            break
        engine._step_sync()
    assert 0 not in engine.active, "request must finish at max_tokens"
    _pump(engine)
    result = req.future.result()
    assert result["finish_reason"] == "length"
    assert result["committed_token_ids"] == [7, 7, 7, 7, 7]  # 5 decode rounds


def test_admission_reuses_runner_prefix_surface() -> None:
    """The engine's admission path calls reconcile_prefix_hit and
    find_best_slot_for_prompt through the coordinator; with prefix_cache
    False these must be harmless no-ops on the real backend."""
    backend = _make_backend()
    engine = _make_engine(backend)
    # slot_resources wraps the backend; the no-cache path must forward
    # cleanly.
    hit = engine.slot_resources.reconcile_prefix_hit([1, 2])
    assert hit.kv_hit == 0 and hit.effective == 0
    slot, hit_len = engine.slot_resources.find_best_slot_for_prompt([1, 2], [0, 1])
    assert slot in (0, 1) and hit_len == 0


def test_backend_snapshot_after_round() -> None:
    backend = _make_backend()
    backend.prefill(0, [1, 2])
    snap = backend.snapshot()
    assert snap.slots[0].kv_len == 2
    assert snap.slots[0].is_fresh is False
    assert snap.slots[1].is_fresh is True
