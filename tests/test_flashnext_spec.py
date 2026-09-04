from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from runtime.model.flashnext.model import (  # noqa: E402
    FlashNextGraphEngine,
    new_layer_states,
    prefill_session,
    prefill_session_layer_major,
)
from runtime.model.flashnext.spec import (  # noqa: E402
    FlashNextSpecEngine,
    FlashNextVerifyGraph,
    allocate_verify_buffers,
    verify_body,
)
from runtime.sampling import PersistentSeed, SamplingParams  # noqa: E402


def test_verify_graph_can_share_fixed_scratch_between_serial_slots() -> None:
    model = SimpleNamespace(
        cfg=SimpleNamespace(hidden_size=4),
        layers=[],
    )
    sess = SimpleNamespace(
        qsa_k_pool={0: torch.empty(16, 1)},
        token_buf=torch.empty(1, dtype=torch.long),
        ple_conv_state=None,
    )
    buffers = allocate_verify_buffers(model, sess, qo_len=4, device="cpu")

    first = FlashNextVerifyGraph(model, sess, "cpu", k=3, buffers=buffers)
    second = FlashNextVerifyGraph(model, sess, "cpu", k=3, buffers=buffers)

    assert first.buffers is buffers
    assert second.buffers is buffers


def test_verify_graph_rejects_shared_scratch_for_a_different_k() -> None:
    model = SimpleNamespace(
        cfg=SimpleNamespace(hidden_size=4),
        layers=[],
    )
    sess = SimpleNamespace(
        qsa_k_pool={0: torch.empty(16, 1)},
        token_buf=torch.empty(1, dtype=torch.long),
        ple_conv_state=None,
    )
    buffers = allocate_verify_buffers(model, sess, qo_len=4, device="cpu")

    with pytest.raises(ValueError, match="wrong fixed row count"):
        FlashNextVerifyGraph(model, sess, "cpu", k=2, buffers=buffers)


def test_recomputed_verify_state_keeps_one_layer_scratch_not_per_layer_rows() -> None:
    layers = [
        SimpleNamespace(
            is_qsa=False,
            layer_idx=idx,
            attn=SimpleNamespace(num_v_heads=2, head_v_dim=3),
        )
        for idx in range(2)
    ]
    model = SimpleNamespace(cfg=SimpleNamespace(hidden_size=4), layers=layers)
    sess = SimpleNamespace(
        ple_conv_state=None,
        gdn={
            f"gdn_{idx}": SimpleNamespace(
                conv_state=torch.zeros(1, 5, 2),
                recurrent_state=torch.zeros(1, 2, 4, 3),
                has_previous_state=True,
            )
            for idx in range(2)
        },
    )

    buffers = allocate_verify_buffers(
        model,
        sess,
        qo_len=4,
        device="cpu",
        allocate_sequential_work=False,
        recompute_recurrent_state=True,
    )

    assert buffers.gdn_recompute_scratch.shape == (4, 2, 4, 3)
    assert buffers.gdn_recompute_output.shape == (1, 4, 2, 3)
    assert all(rows.numel() == 0 for rows in buffers.gdn_recurrent_rows.values())
    assert all(
        row.recurrent_state.numel() == 0
        for candidates in buffers.gdn_rows.values()
        for row in candidates
    )


def test_recomputed_verify_state_does_not_share_graph_owned_commit_inputs() -> None:
    layer = SimpleNamespace(
        is_qsa=False,
        layer_idx=0,
        attn=SimpleNamespace(num_v_heads=2, head_v_dim=3),
    )
    model = SimpleNamespace(cfg=SimpleNamespace(hidden_size=4), layers=[layer])
    sess = SimpleNamespace(
        qsa_k_pool={0: torch.empty(16, 1)},
        token_buf=torch.empty(1, dtype=torch.long),
        ple_conv_state=None,
        gdn={
            "gdn_0": SimpleNamespace(
                conv_state=torch.zeros(1, 5, 2),
                recurrent_state=torch.zeros(1, 2, 4, 3),
                has_previous_state=True,
            )
        },
    )
    buffers = allocate_verify_buffers(
        model,
        sess,
        qo_len=4,
        device="cpu",
        allocate_sequential_work=False,
        recompute_recurrent_state=True,
    )

    first = FlashNextVerifyGraph(
        model,
        sess,
        "cpu",
        k=3,
        recompute_recurrent_state=True,
        buffers=buffers,
    )
    second = FlashNextVerifyGraph(
        model,
        sess,
        "cpu",
        k=3,
        recompute_recurrent_state=True,
        buffers=buffers,
    )

    assert first.buffers is second.buffers
    assert first._gdn_commit_inputs is not second._gdn_commit_inputs
    assert first._gdn_commit_inputs[0] is not second._gdn_commit_inputs[0]


from runtime.model.qwen36_model import GdnLayerState  # noqa: E402


def test_new_layer_states_honors_checkpoint_ssm_dtype():
    gdn = SimpleNamespace(
        conv_dim=8,
        conv_kernel_size=4,
        num_v_heads=2,
        head_k_dim=3,
        head_v_dim=5,
    )
    model = SimpleNamespace(
        cfg=SimpleNamespace(mamba_ssm_dtype="float32"),
        layers=[SimpleNamespace(is_qsa=False, layer_idx=0, attn=gdn)],
    )

    state = new_layer_states(model, "cpu")["gdn_0"]

    assert state.conv_state.dtype == torch.bfloat16
    assert state.recurrent_state.dtype == torch.float32


class _FakeMixer:
    def mix(self, value):
        return value, (value, value)

    def combine(self, value, residuals):
        del residuals
        return value


class _FakeAttn:
    def __init__(self) -> None:
        self.spec_forward_calls = 0
        self.forward_calls = 0

    def __call__(self, hidden_states, state):
        self.forward_calls += 1
        state.conv_state.add_(1)
        state.recurrent_state.add_(2)
        return hidden_states + 5

    def spec_forward(
        self,
        hidden_states,
        state,
        *,
        spec_state_rows,
        batch_large_projections,
        fp32_intermediate_states=None,
    ):
        self.spec_forward_calls += 1
        assert tuple(hidden_states.shape) == (1, 3, 4)
        assert batch_large_projections is False
        assert fp32_intermediate_states is None
        for row in spec_state_rows:
            row.conv_state.copy_(state.conv_state + 1)
            row.recurrent_state.copy_(state.recurrent_state + 2)
            row.has_previous_state = True
        return hidden_states + 5, None


class _FakePrefillAttn:
    def __call__(self, hidden_states, state):
        seen = int(state.recurrent_state.flatten()[0].item())
        steps = hidden_states.shape[1]
        offsets = torch.arange(
            seen + 1,
            seen + steps + 1,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ).view(1, steps, 1)
        state.conv_state.fill_(seen + steps)
        state.recurrent_state.fill_(seen + steps)
        state.has_previous_state = True
        return hidden_states + offsets


def test_verify_body_exact_row_math_uses_decode_gdn_forward():
    attn = _FakeAttn()
    layer = SimpleNamespace(
        ple=None,
        attn_hc=_FakeMixer(),
        mlp_hc=_FakeMixer(),
        attn=attn,
        mlp=lambda x: x,
        is_qsa=False,
        layer_idx=0,
    )
    model = SimpleNamespace(
        cfg=SimpleNamespace(hidden_size=4, hc_count=1),
        layers=[layer],
        embed_tokens=lambda token_ids: token_ids.to(torch.bfloat16).unsqueeze(-1).repeat(1, 4),
        final_mixer=_FakeMixer(),
        lm_head=lambda x: x.to(torch.float32),
    )
    live_state = GdnLayerState(
        conv_state=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        recurrent_state=torch.zeros(1, 1, 1, 2, dtype=torch.bfloat16),
        has_previous_state=True,
    )
    candidate_rows = [
        GdnLayerState(
            conv_state=torch.empty_like(live_state.conv_state),
            recurrent_state=torch.empty_like(live_state.recurrent_state),
            has_previous_state=False,
        )
        for _ in range(3)
    ]
    sess = SimpleNamespace(
        gdn={"gdn_0": live_state},
        ple_conv_state=None,
        qsa_idx_k_pool={},
        qsa_k_pool={},
        qsa_v_pool={},
        qsa_attn={},
        qsa_pad=0,
    )
    buffers = SimpleNamespace(
        token_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        positions=torch.tensor([7, 8, 9], dtype=torch.long),
        ple_embeddings=torch.zeros(3, 4, dtype=torch.bfloat16),
        gdn_rows={0: candidate_rows},
        gdn_work={
            0: GdnLayerState(
                conv_state=torch.empty_like(live_state.conv_state),
                recurrent_state=torch.empty_like(live_state.recurrent_state),
                has_previous_state=False,
            )
        },
        ple_rows=[],
    )

    hc_hidden, logits = verify_body(model, sess, buffers, exact_row_math=True)

    assert attn.spec_forward_calls == 0
    assert attn.forward_calls == 3
    torch.testing.assert_close(hc_hidden, model.embed_tokens(buffers.token_ids) + 5, rtol=0, atol=0)
    torch.testing.assert_close(logits, hc_hidden.to(torch.float32), rtol=0, atol=0)
    for index, row in enumerate(candidate_rows, start=1):
        torch.testing.assert_close(row.conv_state, live_state.conv_state + index, rtol=0, atol=0)
        torch.testing.assert_close(
            row.recurrent_state, live_state.recurrent_state + 2 * index, rtol=0, atol=0
        )


def test_verify_qsa_reuses_decode_mrope_cache():
    """Target verify must pass the same MRoPE cache ABI as M=1 decode."""

    class _Indexer:
        compress_ratio = 2
        block_topk = 1
        calls = []

        def project_qk(self, hidden_states, positions):
            rows = hidden_states.shape[0]
            return (
                torch.zeros(rows, 1, 2, dtype=hidden_states.dtype),
                torch.zeros(rows, 2, dtype=hidden_states.dtype),
            )

        def update_index_cache_fixed(
            self, raw_cache, pooled_cache, keys, positions, **kwargs
        ):
            del raw_cache, pooled_cache, keys
            self.calls.append((positions.clone(), kwargs))

        def score_blocks(self, q, pooled_cache, row_block_ends):
            del pooled_cache, row_block_ends
            return torch.zeros(q.shape[0], 1, dtype=q.dtype)

        def select_blocks(self, logits, row_block_ends):
            del row_block_ends
            return torch.zeros(logits.shape[0], 1, dtype=torch.long)

        def batch_decode_gather_indices(self, blocks, positions, pad_to):
            rows = blocks.shape[0]
            return (
                torch.zeros(rows, pad_to, dtype=torch.long),
                torch.ones(rows, pad_to, dtype=torch.bool),
            )

    class _Attention:
        def project(self, hidden_states, positions):
            rows = hidden_states.shape[0]
            shape = (rows, 1, 2)
            value = torch.arange(
                rows * 2,
                dtype=hidden_states.dtype,
            ).reshape(shape)
            return value, value + 10, value + 20, value + 30

    class _QsaDecode:
        def __call__(self, q, gate, *args):
            del gate, args
            return torch.zeros(q.shape[0], 4, dtype=q.dtype)

    indexer = _Indexer()
    layer = SimpleNamespace(
        ple=None,
        attn_hc=_FakeMixer(),
        mlp_hc=_FakeMixer(),
        attn=SimpleNamespace(indexer=indexer, attn=_Attention()),
        mlp=lambda x: x,
        is_qsa=True,
        layer_idx=0,
    )
    model = SimpleNamespace(
        cfg=SimpleNamespace(hidden_size=4, hc_count=1),
        layers=[layer],
        embed_tokens=lambda token_ids: token_ids.to(torch.bfloat16).unsqueeze(-1).repeat(1, 4),
        final_mixer=_FakeMixer(),
        lm_head=lambda x: x.to(torch.float32),
    )
    sess = SimpleNamespace(
        qsa_idx_k_pool={0: torch.zeros(16, 2, dtype=torch.bfloat16)},
        qsa_pooled_k_pool={0: torch.zeros(8, 2, dtype=torch.bfloat16)},
        qsa_idx_rope_pool={0: torch.zeros(16, 3, dtype=torch.long)},
        qsa_k_pool={0: torch.zeros(16, 1, 2, dtype=torch.bfloat16)},
        qsa_v_pool={0: torch.zeros(16, 1, 2, dtype=torch.bfloat16)},
        qsa_k_scale_pool={0: torch.ones(16, 1, dtype=torch.float16)},
        qsa_v_scale_pool={0: torch.ones(16, 1, dtype=torch.float16)},
        qsa_attn={0: _QsaDecode()},
        qsa_pad=3,
    )
    buffers = SimpleNamespace(
        token_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        positions=torch.tensor([7, 8, 9], dtype=torch.long),
        ple_embeddings=torch.zeros(3, 4, dtype=torch.bfloat16),
        gdn_rows={},
        gdn_work={},
        ple_rows=[],
    )

    verify_body(model, sess, buffers, exact_row_math=True)

    assert len(indexer.calls) == 1
    positions, kwargs = indexer.calls[0]
    torch.testing.assert_close(positions, buffers.positions)
    assert kwargs["rope_cache"] is sess.qsa_idx_rope_pool[0]
    torch.testing.assert_close(kwargs["rope_positions"], buffers.positions)
    expected_k = torch.arange(2, dtype=torch.bfloat16).reshape(1, 1, 2).expand(3, -1, -1) + 10
    expected_v = torch.arange(2, dtype=torch.bfloat16).reshape(1, 1, 2).expand(3, -1, -1) + 20
    torch.testing.assert_close(sess.qsa_k_pool[0][buffers.positions], expected_k)
    torch.testing.assert_close(sess.qsa_v_pool[0][buffers.positions], expected_v)


class _FakePrefillModel(torch.nn.Module):
    def __init__(self, layer) -> None:
        super().__init__()
        self._anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.cfg = SimpleNamespace(hidden_size=4, hc_count=1, ngram_size=4)
        self.layers = [layer]
        self.final_mixer = _FakeMixer()
        self.lm_head = lambda x: x.to(torch.float32)

    def embed_tokens(self, token_ids):
        return token_ids.to(torch.bfloat16).unsqueeze(-1).repeat(1, 4)


def _fake_prefill_session():
    return SimpleNamespace(
        gdn={
            "gdn_0": SimpleNamespace(
                conv_state=torch.full((1, 1, 1), -7, dtype=torch.bfloat16),
                recurrent_state=torch.full((1, 1, 1, 1), -9, dtype=torch.bfloat16),
                has_previous_state=True,
            )
        },
        qsa_k_pool={0: torch.zeros(16, 1, dtype=torch.bfloat16)},
        qsa_v_pool={0: torch.zeros(16, 1, dtype=torch.bfloat16)},
        qsa_idx_k_pool={0: torch.zeros(16, 1, dtype=torch.bfloat16)},
        qsa_pooled_k_pool={0: torch.zeros(4, 1, dtype=torch.bfloat16)},
        qsa_attn={},
        ple_conv_state=None,
        window=[],
        pos=0,
    )


def test_layer_major_prefill_matches_chunked_reference():
    layer = SimpleNamespace(
        ple=None,
        attn_hc=_FakeMixer(),
        mlp_hc=_FakeMixer(),
        attn=_FakePrefillAttn(),
        mlp=lambda x: x * 2,
        is_qsa=False,
        layer_idx=0,
    )
    model = _FakePrefillModel(layer)
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)

    chunked_sess = _fake_prefill_session()
    layer_major_sess = _fake_prefill_session()
    chunked_logits, chunked_hidden = prefill_session(model, input_ids, chunked_sess)
    layer_major_logits, layer_major_hidden = prefill_session_layer_major(
        model,
        input_ids,
        layer_major_sess,
        attention_chunk_size=2,
    )

    torch.testing.assert_close(layer_major_logits, chunked_logits, rtol=0, atol=0)
    torch.testing.assert_close(layer_major_hidden, chunked_hidden, rtol=0, atol=0)
    torch.testing.assert_close(
        layer_major_sess.gdn["gdn_0"].conv_state,
        chunked_sess.gdn["gdn_0"].conv_state,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        layer_major_sess.gdn["gdn_0"].recurrent_state,
        chunked_sess.gdn["gdn_0"].recurrent_state,
        rtol=0,
        atol=0,
    )
    assert layer_major_sess.window == chunked_sess.window
    assert layer_major_sess.pos == chunked_sess.pos


def test_graph_engine_prefill_defaults_to_token_major(monkeypatch):
    import runtime.model.flashnext.model as flashnext_model

    calls = []

    def fake_prefill(model, tokens, sess):
        del model, sess
        calls.append(tokens.tolist())
        return torch.tensor([float(tokens[-1])]), tokens[:, None].to(torch.float32)

    def reject_layer_major(*args, **kwargs):
        del args, kwargs
        raise AssertionError("unqualified layer-major prefill must not run")

    monkeypatch.setattr(flashnext_model, "prefill_session", fake_prefill)
    monkeypatch.setattr(
        flashnext_model,
        "prefill_session_layer_major",
        reject_layer_major,
    )
    engine = object.__new__(FlashNextGraphEngine)
    engine.model = SimpleNamespace(layers=[])
    engine.sess = SimpleNamespace(window=[])

    logits, hidden = engine.prefill([1, 2, 3, 4, 5], chunk_size=2)

    assert calls == [[1, 2], [3, 4], [5]]
    torch.testing.assert_close(logits, torch.tensor([5.0]))
    torch.testing.assert_close(
        hidden,
        torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]]),
    )


def test_spec_engine_sparse_graph_opt_in_uses_max_seq_capacity(monkeypatch):
    import runtime.model.flashnext.spec as flashnext_spec

    prepared = {}
    continuation = {}
    proposals = {}

    class _Verify:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def capture(self):
            return None

    class _ContGraph:
        def __init__(
            self,
            model,
            mtp,
            sess,
            *,
            device,
            graph_capacity,
            continuation_steps,
            sparse_qsa,
        ) -> None:
            del model, mtp, sess, device
            continuation["graph_capacity"] = graph_capacity
            continuation["continuation_steps"] = continuation_steps
            continuation["sparse_qsa"] = sparse_qsa
            self.graph_capacity = graph_capacity

        def capture(self):
            return None

    class _ProposalGraph:
        def __init__(
            self,
            model,
            mtp,
            sess,
            *,
            device,
            graph_capacity,
            query_len,
            k,
            sparse_qsa,
        ) -> None:
            del model, mtp, sess, device, k
            proposals[query_len] = (graph_capacity, sparse_qsa)
            self.query_len = query_len
            self.graph_capacity = graph_capacity

        def capture(self):
            return None

    monkeypatch.setattr(flashnext_spec, "FlashNextVerifyGraph", _Verify)
    monkeypatch.setattr(
        flashnext_spec,
        "new_mtp_session",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    def fake_prepare(mtp, sess, *, max_rows, device):
        del mtp, sess
        prepared["max_rows"] = max_rows
        prepared["device"] = device

    monkeypatch.setattr(flashnext_spec, "prepare_mtp_sparse_graph_buffers", fake_prepare)
    monkeypatch.setattr(flashnext_spec, "FlashNextMtpContinuationGraph", _ContGraph)
    monkeypatch.setattr(flashnext_spec, "FlashNextMtpProposalGraph", _ProposalGraph)

    model = SimpleNamespace(cfg=SimpleNamespace(hc_count=1, hidden_size=4))
    mtp = SimpleNamespace(indexer=SimpleNamespace(block_topk=512, compress_ratio=4))
    engine = FlashNextSpecEngine(
        model,
        mtp,
        target_session=SimpleNamespace(),
        max_seq=32768,
        device="cpu",
        k=3,
        mtp_continuation_graph=True,
        mtp_sparse_graph=True,
    )

    assert prepared == {"max_rows": 4, "device": "cpu"}
    assert continuation == {
        "graph_capacity": 32768,
        "continuation_steps": 2,
        "sparse_qsa": True,
    }
    assert proposals == {
        1: (32768, True),
        2: (32768, True),
        3: (32768, True),
        4: (32768, True),
    }
    assert engine.mtp_continuation_graph.graph_capacity == 32768


def test_continue_draft_uses_sparse_graph_past_dense_budget():
    engine = object.__new__(FlashNextSpecEngine)
    engine.k = 3
    engine.device = "cpu"
    engine.max_seq = 32768
    engine.mtp_session = SimpleNamespace(pos=23472)
    replay = {}

    class _Graph:
        graph_capacity = 32768

        def replay(self, first_draft, hidden, position):
            replay["args"] = (first_draft, hidden.clone(), position)
            return torch.tensor([88, 99], dtype=torch.long)

    engine.mtp_continuation_graph = _Graph()

    drafts = FlashNextSpecEngine.continue_draft(
        engine,
        first_draft=77,
        first_hidden=torch.ones(1, 4, dtype=torch.bfloat16),
    )

    assert drafts == [77, 88, 99]
    assert replay["args"][0] == 77
    assert replay["args"][2] == 23472
    assert engine.mtp_session.pos == 23474


def test_sync_and_propose_uses_sparse_graph_past_dense_budget():
    engine = object.__new__(FlashNextSpecEngine)
    engine.k = 3
    engine.max_seq = 32768
    engine.mtp_session = SimpleNamespace(sync_len=23472, pos=0)
    replay = {}

    class _Graph:
        query_len = 2
        graph_capacity = 32768

        def replay(self, shifted_token_ids, target_hidden, position):
            replay["args"] = (list(shifted_token_ids), target_hidden.clone(), position)
            return torch.tensor([11, 22, 33], dtype=torch.long)

    engine.mtp_proposal_graphs = {2: _Graph()}

    drafts = FlashNextSpecEngine.sync_and_propose(
        engine,
        shifted_token_ids=[5, 6],
        target_hc_hidden=torch.ones(2, 4, dtype=torch.bfloat16),
    )

    assert drafts == [11, 22, 33]
    assert replay["args"][0] == [5, 6]
    assert replay["args"][2] == 23472
    assert engine.mtp_session.sync_len == 23474
    assert engine.mtp_session.pos == 23476


def test_sync_real_suffix_uses_precomputed_input_embeds() -> None:
    engine = object.__new__(FlashNextSpecEngine)
    engine.device = torch.device("cpu")
    engine.k = 3
    engine.max_seq = 16
    engine.mtp_session = SimpleNamespace(sync_len=5, pos=0)
    captured: dict[str, torch.Tensor] = {}

    def forward(embeds, target_hc_hidden, positions, sess, **kwargs):
        del sess, kwargs
        captured["embeds"] = embeds.clone()
        captured["target_hc_hidden"] = target_hc_hidden.clone()
        captured["positions"] = positions.clone()
        mixed = torch.zeros(2, 4, dtype=torch.bfloat16)
        own_hc = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        return mixed, own_hc

    engine.mtp = SimpleNamespace(forward=forward)
    engine.model = SimpleNamespace(
        cfg=SimpleNamespace(hc_count=1, hidden_size=4),
        embed_tokens=lambda _tokens: (_ for _ in ()).throw(
            AssertionError("embed_tokens must not run when input_embeds is provided")
        ),
        lm_head=lambda mixed: torch.tensor(
            [[0.0, 1.0], [0.0, 3.0]],
            dtype=torch.float32,
        )[: mixed.shape[0]],
    )

    input_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    first_draft, hidden = FlashNextSpecEngine.sync_real_suffix(
        engine,
        shifted_token_ids=[7, 8],
        target_hc_hidden=torch.ones(2, 4, dtype=torch.bfloat16),
        input_embeds=input_embeds,
    )

    assert first_draft == 1
    assert torch.equal(captured["embeds"], input_embeds.to(dtype=torch.bfloat16))
    assert torch.equal(captured["positions"], torch.tensor([5, 6], dtype=torch.long))
    assert torch.equal(hidden, torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)[-1:])
    assert engine.mtp_session.sync_len == 7
    assert engine.mtp_session.pos == 7


def test_sync_real_suffix_can_advance_state_without_lm_head() -> None:
    """Intermediate chunk sync must not run a discarded vocabulary matmul."""
    engine = object.__new__(FlashNextSpecEngine)
    engine.device = torch.device("cpu")
    engine.k = 3
    engine.max_seq = 16
    engine.mtp_session = SimpleNamespace(sync_len=5, pos=0)

    def forward(embeds, target_hc_hidden, positions, sess, **kwargs):
        del embeds, target_hc_hidden, positions, sess, kwargs
        return torch.zeros(2, 4, dtype=torch.bfloat16), torch.ones(2, 4)

    engine.mtp = SimpleNamespace(forward=forward)

    def lm_head(_mixed):
        raise AssertionError("intermediate sync must skip lm_head")

    engine.model = SimpleNamespace(
        cfg=SimpleNamespace(hc_count=1, hidden_size=4),
        embed_tokens=lambda tokens: tokens.to(torch.bfloat16).unsqueeze(-1).repeat(1, 4),
        lm_head=lm_head,
    )

    first_draft, hidden = FlashNextSpecEngine.sync_real_suffix(
        engine,
        shifted_token_ids=[7, 8],
        target_hc_hidden=torch.ones(2, 4, dtype=torch.bfloat16),
        return_first_token=False,
    )

    assert first_draft == 0
    assert hidden.shape == (1, 4)
    assert engine.mtp_session.sync_len == 7
    assert engine.mtp_session.pos == 7


def test_sync_and_propose_replays_graph_with_precomputed_input_embeds() -> None:
    engine = object.__new__(FlashNextSpecEngine)
    engine.k = 3
    engine.max_seq = 32768
    engine.mtp_session = SimpleNamespace(sync_len=23472, pos=0)

    class _Graph:
        query_len = 2
        graph_capacity = 32768

        graph = object()

        def replay(self, tokens, hidden, position, *, input_embeds=None):
            captured["graph_tokens"] = list(tokens)
            captured["graph_hidden"] = hidden.clone()
            captured["graph_position"] = position
            captured["graph_input_embeds"] = input_embeds.clone()
            return torch.tensor([7, 8, 9], dtype=torch.long)

    captured: dict[str, object] = {}
    engine.mtp_proposal_graphs = {2: _Graph()}

    input_embeds = torch.ones(2, 4, dtype=torch.bfloat16)
    drafts = FlashNextSpecEngine.sync_and_propose(
        engine,
        shifted_token_ids=[5, 6],
        target_hc_hidden=torch.ones(2, 4, dtype=torch.bfloat16),
        input_embeds=input_embeds,
    )

    assert drafts == [7, 8, 9]
    assert captured["graph_tokens"] == [5, 6]
    assert captured["graph_position"] == 23472
    assert torch.equal(captured["graph_input_embeds"], input_embeds)
    assert engine.mtp_session.sync_len == 23474
    assert engine.mtp_session.pos == 23476


def test_sampled_sync_and_propose_keeps_the_exact_draft_distributions() -> None:
    """Sampled MTP must expose q for every sampled draft, not argmax one-hot."""

    engine = object.__new__(FlashNextSpecEngine)
    engine.device = torch.device("cpu")
    engine.k = 3
    engine.max_seq = 32
    engine.mtp_session = SimpleNamespace(sync_len=0, pos=0)
    engine.mtp_proposal_graphs = {}
    engine.mtp_continuation_graph = None

    def forward(embeds, target_hc_hidden, positions, sess, **kwargs):
        del target_hc_hidden, positions, sess, kwargs
        rows = embeds.shape[0]
        mixed = torch.zeros(rows, 4, dtype=torch.bfloat16)
        own_hc = torch.arange(rows * 4, dtype=torch.bfloat16).reshape(rows, 4)
        return mixed, own_hc

    calls = iter((1, 2, 3))

    def lm_head(mixed):
        logits = torch.zeros(mixed.shape[0], 5, dtype=torch.float32)
        logits[:, next(calls)] = 2.0
        return logits

    engine.mtp = SimpleNamespace(forward=forward)
    engine.model = SimpleNamespace(
        cfg=SimpleNamespace(hc_count=1, hidden_size=4),
        embed_tokens=lambda tokens: torch.zeros(tokens.shape[0], 4, dtype=torch.bfloat16),
        lm_head=lm_head,
    )
    params = SamplingParams(
        temperature=0.8,
        top_k=0,
        top_p=1.0,
        seed=PersistentSeed(17),
    )

    drafts = FlashNextSpecEngine.sync_and_propose(
        engine,
        shifted_token_ids=[11, 12],
        target_hc_hidden=torch.ones(2, 4, dtype=torch.bfloat16),
        params=params,
    )

    assert len(drafts) == 3
    assert engine.pending_draft_probs is not None
    assert engine.pending_draft_probs.shape == (3, 5)
    assert torch.allclose(engine.pending_draft_probs.sum(dim=-1), torch.ones(3))
    assert torch.all(engine.pending_draft_probs.gather(1, torch.tensor(drafts).unsqueeze(1)) > 0)
    assert engine.mtp_session.sync_len == 2
    assert engine.mtp_session.pos == 4


def test_sampled_round_uses_rejection_sampling_and_preserves_commit_shape() -> None:
    engine = object.__new__(FlashNextSpecEngine)
    engine.device = torch.device("cpu")
    engine.k = 2
    engine.target_session = SimpleNamespace(pos=5)
    engine.mtp_session = SimpleNamespace(sync_len=5)
    logits = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
        ]
    )
    params = SamplingParams(temperature=0.7, seed=PersistentSeed(3))
    from runtime.sampling import compute_sampling_distribution

    engine.pending_draft_probs = compute_sampling_distribution(logits[:2], params)
    engine.verify = SimpleNamespace(
        replay=lambda tokens, past_len: (torch.ones(3, 8), logits),
        eager=lambda tokens, past_len: (torch.ones(3, 8), logits),
        commit=lambda count: None,
        last_ple_seconds=0.0,
    )
    proposed: dict[str, object] = {}

    def sync(tokens, hidden, *, params=None):
        proposed["tokens"] = list(tokens)
        proposed["params"] = params
        return [8, 9]

    engine.sync_and_propose = sync
    result = FlashNextSpecEngine.round(
        engine,
        anchor_token=0,
        drafts=[1, 2],
        params=params,
    )

    assert result["num_accepted"] == 2
    assert result["committed"][:2] == [1, 2]
    assert len(result["committed"]) == 3
    assert result["next_draft_tokens"] == [8, 9]
    assert proposed["params"] is params


@pytest.mark.parametrize(
    ("prediction_ids", "expected_accepted", "expected_reject", "expected_teacher"),
    [
        ([11, 99, 98, 97], 0, 0, [11]),
        ([21, 22, 99, 97], 2, 2, [21, 22, 99]),
        ([21, 22, 23, 24], 3, -1, [21, 22, 23, 24]),
    ],
)
def test_round_returns_complete_acceptance_trace_metadata(
    prediction_ids,
    expected_accepted,
    expected_reject,
    expected_teacher,
):
    engine = object.__new__(FlashNextSpecEngine)
    engine.k = 3
    engine.target_session = SimpleNamespace(pos=7)
    engine.mtp_session = SimpleNamespace(sync_len=7)
    logits = torch.full((4, 128), -1.0)
    for row, token in enumerate(prediction_ids):
        logits[row, token] = 1.0
    engine.verify = SimpleNamespace(
        replay=lambda tokens, past_len: (torch.ones(4, 8), logits),
        eager=lambda tokens, past_len: (torch.ones(4, 8), logits),
        commit=lambda count: None,
        last_ple_seconds=0.0,
    )
    proposed = [31, 32, 33]
    engine.sync_and_propose = lambda tokens, hidden: proposed

    result = FlashNextSpecEngine.round(engine, 20, [21, 22, 23])

    assert result["num_accepted"] == expected_accepted
    assert result["reject_position"] == expected_reject
    assert result["verify_tokens"] == [20, 21, 22, 23]
    assert result["verify_prediction_ids"] == expected_teacher
    assert result["teacher_tokens"] == expected_teacher
    assert result["bonus_token"] == expected_teacher[-1]
    assert result["next_draft_tokens"] == proposed


def test_sparse_graph_continuation_body_reuses_captured_sync_row():
    calls = []

    mtp = SimpleNamespace()

    def forward(embeds, hidden, position, sess, **kwargs):
        del embeds, sess
        calls.append((position.clone(), kwargs))
        mixed = torch.zeros(1, 4, dtype=torch.bfloat16)
        next_hidden = hidden + 1
        return mixed, next_hidden

    mtp.forward = forward
    model = SimpleNamespace(
        cfg=SimpleNamespace(hc_count=1, hidden_size=4),
        embed_tokens=lambda token: torch.zeros(1, 4, dtype=torch.bfloat16),
        lm_head=lambda mixed: torch.tensor([[7.0]], dtype=torch.float32),
    )
    from runtime.model.flashnext.spec import FlashNextMtpContinuationGraph

    graph = FlashNextMtpContinuationGraph(
        model,
        mtp,
        SimpleNamespace(),
        device="cpu",
        graph_capacity=32768,
        continuation_steps=2,
        sparse_qsa=True,
    )

    tokens, hidden = graph._body()

    assert tokens.tolist() == [0, 0]
    assert hidden.shape == (1, 4)
    assert [call[0].tolist() for call in calls] == [[0], [1]]
    for _, kwargs in calls:
        assert kwargs["reuse_sparse_indices"] is True
        assert kwargs["graph_sparse_capacity"] == 32768
        assert kwargs["graph_dense_capacity"] is None


def test_sparse_graph_proposal_body_captures_then_reuses_sync_row():
    calls = []

    mtp = SimpleNamespace()

    def forward(embeds, hidden, position, sess, **kwargs):
        del embeds, sess
        calls.append((position.clone(), kwargs))
        rows = position.shape[0]
        mixed = torch.zeros(rows, 4, dtype=torch.bfloat16)
        next_hidden = hidden + 1
        return mixed, next_hidden

    mtp.forward = forward
    model = SimpleNamespace(
        cfg=SimpleNamespace(hc_count=1, hidden_size=4),
        embed_tokens=lambda token: torch.zeros(token.shape[0], 4, dtype=torch.bfloat16),
        lm_head=lambda mixed: torch.zeros(mixed.shape[0], 1, dtype=torch.float32),
    )
    from runtime.model.flashnext.spec import FlashNextMtpProposalGraph

    graph = FlashNextMtpProposalGraph(
        model,
        mtp,
        SimpleNamespace(),
        device="cpu",
        graph_capacity=32768,
        query_len=2,
        k=3,
        sparse_qsa=True,
    )

    drafts = graph._body()

    assert drafts.tolist() == [0, 0, 0]
    assert calls[0][0].tolist() == [0, 1]
    assert calls[0][1]["capture_sparse_indices"] is True
    assert calls[0][1].get("reuse_sparse_indices", False) is False
    assert calls[0][1]["graph_sparse_capacity"] == 32768
    assert calls[1][0].tolist() == [2]
    assert calls[2][0].tolist() == [3]
    for _, kwargs in calls[1:]:
        assert kwargs["reuse_sparse_indices"] is True
        assert kwargs["graph_sparse_capacity"] == 32768
        assert kwargs["graph_dense_capacity"] is None


def test_proposal_graph_replay_populates_teacher_embedding_buffer():
    """Visual teacher rows must reach the captured MTP graph input pointer."""

    from runtime.model.flashnext.spec import FlashNextMtpProposalGraph

    model = SimpleNamespace(
        cfg=SimpleNamespace(hc_count=1, hidden_size=4),
        embed_tokens=lambda tokens: tokens.to(torch.bfloat16).unsqueeze(-1).repeat(1, 4),
    )
    graph = object.__new__(FlashNextMtpProposalGraph)
    graph.graph = object()
    graph.query_len = 2
    graph.k = 3
    graph.graph_capacity = 32
    graph.tokens = torch.zeros(2, dtype=torch.long)
    graph.target_hidden = torch.zeros(2, 4, dtype=torch.bfloat16)
    graph.teacher_embeds = torch.zeros(2, 4, dtype=torch.bfloat16)
    graph.position = torch.zeros(1, dtype=torch.long)
    graph.sess = SimpleNamespace(mtp_k_pool=torch.zeros(32, 1))
    graph.model = model
    replayed = []
    graph.graph = SimpleNamespace(replay=lambda: replayed.append(True))
    graph._drafts = torch.tensor([8, 9, 10], dtype=torch.long)

    visual_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    drafts = graph.replay(
        [5, 6],
        torch.ones(2, 4, dtype=torch.bfloat16),
        7,
        input_embeds=visual_embeds,
    )

    assert drafts.tolist() == [8, 9, 10]
    assert replayed == [True]
    assert torch.equal(graph.teacher_embeds, visual_embeds.to(torch.bfloat16))
    assert graph.tokens.tolist() == [5, 6]


def test_verify_replay_rejects_cache_overflow_before_replay():
    from runtime.model.flashnext.spec import FlashNextVerifyGraph

    verify = object.__new__(FlashNextVerifyGraph)
    verify.k = 3
    verify.qo_len = 4
    verify.graph = SimpleNamespace(replay=lambda: None)
    verify.sess = SimpleNamespace(pos=0, window=[])
    verify.device = "cpu"
    verify.max_seq = 6
    verify._prepare_ple = lambda token_ids: None
    verify.buffers = SimpleNamespace(
        token_ids=torch.zeros(4, dtype=torch.long),
        positions=torch.zeros(4, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="verify exceeds target cache capacity"):
        FlashNextVerifyGraph.replay(verify, [1, 2, 3, 4], past_len=3)


def test_verify_eager_rejects_cache_overflow_before_body():
    from runtime.model.flashnext.spec import FlashNextVerifyGraph

    verify = object.__new__(FlashNextVerifyGraph)
    verify.k = 3
    verify.qo_len = 4
    verify.sess = SimpleNamespace(pos=0, window=[])
    verify.device = "cpu"
    verify.max_seq = 6
    verify._prepare_ple = lambda token_ids: None
    verify.buffers = SimpleNamespace(
        token_ids=torch.zeros(4, dtype=torch.long),
        positions=torch.zeros(4, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="verify exceeds target cache capacity"):
        FlashNextVerifyGraph.eager(verify, [1, 2, 3, 4], past_len=3)


def test_verify_eager_keeps_cuda_graph_output_handles(monkeypatch):
    """An eager oracle must not rebind buffers returned by a live graph.

    Graph replay writes into the allocations captured by ``capture`` and
    ``replay`` returns the handles stored on the graph object.  The eager
    correctness path is allowed to return its own tensors, but must leave
    those graph-owned handles untouched so a later replay cannot return a
    stale eager allocation.
    """
    import runtime.model.flashnext.spec as flashnext_spec

    graph_hidden = torch.tensor([[1.0]])
    graph_logits = torch.tensor([[2.0]])
    eager_hidden = torch.tensor([[3.0]])
    eager_logits = torch.tensor([[4.0]])

    def fake_verify_body(*args, **kwargs):
        del args, kwargs
        return eager_hidden, eager_logits

    monkeypatch.setattr(flashnext_spec, "verify_body", fake_verify_body)

    verify = object.__new__(FlashNextVerifyGraph)
    verify.model = SimpleNamespace()
    verify.sess = SimpleNamespace(pos=0, window=[])
    verify.device = "cpu"
    verify.k = 3
    verify.qo_len = 4
    verify.max_seq = None
    verify.exact_row_math = True
    verify.batch_lm_head = False
    verify.batch_gdn_recurrence = False
    verify.batch_gdn_projections = False
    verify._gdn_commit_inputs = {}
    verify._prepare_ple = lambda token_ids: None
    verify.buffers = SimpleNamespace(
        token_ids=torch.zeros(4, dtype=torch.long),
        positions=torch.zeros(4, dtype=torch.long),
    )
    verify._hc_hidden = graph_hidden
    verify._logits = graph_logits

    returned_hidden, returned_logits = FlashNextVerifyGraph.eager(
        verify,
        [10, 11, 12, 13],
        past_len=0,
    )

    assert returned_hidden is eager_hidden
    assert returned_logits is eager_logits
    assert verify._hc_hidden is graph_hidden
    assert verify._logits is graph_logits


def test_verify_commit_rolls_back_unaccepted_qsa_rows():
    """Rejected verify rows must not leak into the next target decode."""
    from runtime.model.flashnext.qsa import QSAIndexer

    indexer = QSAIndexer(
        hidden_size=4,
        n_heads=1,
        kv_heads=1,
        head_dim=2,
        rotary_dim=2,
        compress_ratio=2,
        block_topk=2,
        dtype=torch.bfloat16,
    )
    layer = SimpleNamespace(
        layer_idx=0,
        is_qsa=True,
        attn=SimpleNamespace(indexer=indexer),
    )
    raw = torch.arange(32, dtype=torch.bfloat16).view(16, 2)
    pooled = torch.full((8, 2), 7, dtype=torch.bfloat16)
    k_pool = torch.ones(16, 1, 2, dtype=torch.bfloat16)
    v_pool = torch.ones_like(k_pool)
    k_scale = torch.full((16, 1), 3, dtype=torch.float16)
    v_scale = torch.full_like(k_scale, 3)
    sess = SimpleNamespace(
        qsa_idx_k_pool={0: raw.clone()},
        qsa_pooled_k_pool={0: pooled.clone()},
        qsa_idx_rope_pool={},
        qsa_k_pool={0: k_pool.clone()},
        qsa_v_pool={0: v_pool.clone()},
        qsa_k_scale_pool={0: k_scale.clone()},
        qsa_v_scale_pool={0: v_scale.clone()},
    )
    verify = object.__new__(FlashNextVerifyGraph)
    verify.model = SimpleNamespace(layers=[layer])
    verify.sess = sess
    verify.qo_len = 4
    verify._last_past_len = 3

    verify._rollback_speculative_qsa(committed_end=5)

    # Rows 5 and 6 were speculative; row 4 remains the committed prefix.
    raw_after = sess.qsa_idx_k_pool[0]
    pooled_after = sess.qsa_pooled_k_pool[0]
    k_after = sess.qsa_k_pool[0]
    v_after = sess.qsa_v_pool[0]
    k_scale_after = sess.qsa_k_scale_pool[0]
    v_scale_after = sess.qsa_v_scale_pool[0]
    torch.testing.assert_close(raw_after[5:7], torch.zeros_like(raw_after[5:7]))
    torch.testing.assert_close(k_after[5:7], torch.zeros_like(k_after[5:7]))
    torch.testing.assert_close(v_after[5:7], torch.zeros_like(v_after[5:7]))
    torch.testing.assert_close(
        k_scale_after[5:7], torch.ones_like(k_scale_after[5:7])
    )
    torch.testing.assert_close(
        v_scale_after[5:7], torch.ones_like(v_scale_after[5:7])
    )
    # Group 1 (tokens 2,3) is complete and rebuilt from raw keys; group 2 is
    # partial after the rollback and must not retain the rejected pooled key.
    expected_group = indexer.pool_key_groups_at_positions(
        raw_after[2:4].unsqueeze(0),
        torch.tensor([2]),
    )[0]
    torch.testing.assert_close(pooled_after[1], expected_group)
    torch.testing.assert_close(pooled_after[2:4], torch.zeros_like(pooled_after[2:4]))
