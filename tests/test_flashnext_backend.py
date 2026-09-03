from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from runtime.backends.flashnext import (  # noqa: E402
    FlashNextBackend,
    FlashNextPrefixSnapshot,
    _flashnext_batch_gdn_projections_enabled,
    _flashnext_gdn_projection_mode,
)


def test_flashnext_gdn_projection_batching_is_safe_by_default(monkeypatch) -> None:
    """BF16 Flash-Next GDNs must never enter the drift-prone batched path."""

    model = SimpleNamespace(
        layers=[
            SimpleNamespace(
                is_qsa=False,
                attn=SimpleNamespace(
                    in_proj_qkvz=nn.Linear(8, 16),
                    out_proj=nn.Linear(16, 8),
                ),
            )
        ]
    )
    monkeypatch.delenv("QSR_FLASHNEXT_BATCH_GDN_PROJECTIONS", raising=False)
    assert _flashnext_batch_gdn_projections_enabled(model) is False

    # A stale force flag must not re-enable a path that is unsupported for the
    # loaded BF16 projection format.
    monkeypatch.setenv("QSR_FLASHNEXT_BATCH_GDN_PROJECTIONS", "1")
    assert _flashnext_batch_gdn_projections_enabled(model) is False

    # The BF16 path is available only through a separately named, explicit
    # validation override.  ``FN_BATCH_GDN_PROJECTIONS`` belongs to the
    # standalone fn6 diagnostic and must never leak into serving implicitly.
    monkeypatch.setenv("QSR_FLASHNEXT_ALLOW_BF16_BATCH_PROJECTIONS", "1")
    assert _flashnext_batch_gdn_projections_enabled(model) is True


def test_flashnext_qwen4_exp_bf16_contract_batches_by_default(monkeypatch) -> None:
    """The validated Flash-Next checkpoint must not fall back to M=1 GEMMs."""

    model = SimpleNamespace(
        cfg=SimpleNamespace(model_type="qwen4_exp", mamba_ssm_dtype="float32"),
        layers=[
            SimpleNamespace(
                is_qsa=False,
                attn=SimpleNamespace(
                    in_proj_qkvz=nn.Linear(8, 16),
                    out_proj=nn.Linear(16, 8),
                ),
            )
        ],
    )
    monkeypatch.delenv("QSR_FLASHNEXT_BATCH_GDN_PROJECTIONS", raising=False)
    monkeypatch.delenv("QSR_FLASHNEXT_ALLOW_BF16_BATCH_PROJECTIONS", raising=False)
    assert _flashnext_gdn_projection_mode(model) == "batched_bf16"
    assert _flashnext_batch_gdn_projections_enabled(model) is True

    # The explicit rollback remains available for numerical A/B tests.
    monkeypatch.setenv("QSR_FLASHNEXT_BATCH_GDN_PROJECTIONS", "0")
    assert _flashnext_gdn_projection_mode(model) == "disabled"
    assert _flashnext_batch_gdn_projections_enabled(model) is False


def test_prefill_mlp_graph_capture_forwards_to_shared_target(monkeypatch) -> None:
    backend = object.__new__(FlashNextBackend)
    backend.device = torch.device("cuda")
    captured: list[int] = []
    backend._targets = [
        SimpleNamespace(capture_prefill_mlp_graphs=lambda rows: captured.append(rows))
    ]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    backend.capture_prefill_mlp_graphs(1024)

    assert captured == [1024]


def test_chunked_prefill_syncs_partial_tail_at_its_real_width() -> None:
    backend = object.__new__(FlashNextBackend)
    backend.max_seq_len = 64
    backend.device = torch.device("cpu")
    backend._slot_tokens = [[]]
    backend._last_logits = [None]
    backend.stats = {
        "prefill_requests": 0,
        "decode_rounds": 0,
        "decode_tokens": 0,
        "mtp_rounds": 0,
        "mtp_accepted_tokens": 0,
        "prefill_chunks": 0,
        "prefill_tokens": 0,
        "prefill_target_ns": 0,
        "prefill_mtp_sync_ns": 0,
        "prefill_mtp_draft_ns": 0,
        "prefill_trim_ns": 0,
        "prefill_last_chunks": 0,
        "prefill_last_tokens": 0,
        "prefill_last_target_ns": 0,
        "prefill_last_mtp_sync_ns": 0,
        "prefill_last_mtp_draft_ns": 0,
        "prefill_last_trim_ns": 0,
    }
    backend._reset_runtime = lambda slot: None
    backend._trim_prefill_cuda_cache = lambda prompt_tokens: None

    class _Target:
        def prefill(self, token_ids):
            return torch.zeros(8), torch.zeros(len(token_ids), 4)

    sync_widths: list[tuple[int, int]] = []

    class _Spec:
        def sync_real_suffix(self, token_ids, hidden):
            sync_widths.append((len(token_ids), hidden.shape[0]))
            assert len(token_ids) == hidden.shape[0]
            return 7, torch.zeros(1, 4)

        def continue_draft(self, first, hidden):
            return [first]

    backend._targets = [_Target()]
    backend._specs = [_Spec()]

    result = backend._prefill_slot(
        0,
        list(range(10)),
        forced_token=42,
        chunk_size=4,
    )

    assert sync_widths == [(4, 4), (4, 4), (2, 2)]
    assert result == {"anchor": 42, "draft_tokens": [7]}


def test_chunked_prefill_submits_one_ple_chunk_ahead() -> None:
    """The long-prompt path queues only the current and next PLE reads."""

    backend = object.__new__(FlashNextBackend)
    backend.max_seq_len = 64
    backend.device = torch.device("cpu")
    backend.enable_prefix_cache = False
    backend._slot_tokens = [[]]
    backend._last_logits = [None]
    backend._specs = [None]
    backend.stats = {
        "prefill_requests": 0,
        "prefill_chunks": 0,
        "prefill_tokens": 0,
        "prefill_target_ns": 0,
        "prefill_mtp_sync_ns": 0,
        "prefill_mtp_draft_ns": 0,
        "prefill_trim_ns": 0,
        "prefill_last_chunks": 0,
        "prefill_last_tokens": 0,
        "prefill_last_target_ns": 0,
        "prefill_last_mtp_sync_ns": 0,
        "prefill_last_mtp_draft_ns": 0,
        "prefill_last_trim_ns": 0,
    }
    backend._reset_runtime = lambda slot: None
    backend._trim_prefill_cuda_cache = lambda prompt_tokens: None
    backend.model = SimpleNamespace(cfg=SimpleNamespace(ngram_size=3))

    prefetched: list[tuple[list[int], list[int]]] = []
    consumed: list[tuple[list[int], object | None]] = []

    class _Target:
        sess = SimpleNamespace(window=[99])

        def start_ple_prefetch(self, token_ids, *, history_tokens, prefix_hint=False):
            prefetched.append((list(token_ids), list(history_tokens)))
            return object()

        def prefill(self, token_ids, **kwargs):
            consumed.append((list(token_ids), kwargs.get("_ple_pending")))
            return torch.zeros(8), torch.zeros(len(token_ids), 4)

    backend._targets = [_Target()]
    result = backend._prefill_slot(0, list(range(6)), forced_token=42, chunk_size=2)

    assert [tokens for tokens, _history in prefetched] == [[0, 1], [2, 3], [4, 5]]
    assert [history for _tokens, history in prefetched] == [
        [99, 0, 1],
        [0, 1, 2, 3],
        [1, 2, 3, 4, 5],
    ]
    assert [tokens for tokens, _pending in consumed] == [[0, 1], [2, 3], [4, 5]]
    assert all(pending is not None for _tokens, pending in consumed)
    assert result == {"anchor": 42, "draft_tokens": []}


def test_chunked_visual_prefill_keeps_external_embeddings_for_every_chunk() -> None:
    """Chunking must not turn image rows back into plain token embeddings."""

    backend = object.__new__(FlashNextBackend)
    backend.max_seq_len = 64
    backend.device = torch.device("cpu")
    backend.enable_prefix_cache = False
    backend.num_slots = 1
    backend._slot_tokens = [[]]
    backend._last_logits = [None]
    backend._specs = [None]
    backend.stats = {
        "prefill_requests": 0,
        "prefill_chunks": 0,
        "prefill_tokens": 0,
        "prefill_target_ns": 0,
        "prefill_mtp_sync_ns": 0,
        "prefill_mtp_draft_ns": 0,
        "prefill_trim_ns": 0,
        "prefill_last_chunks": 0,
        "prefill_last_tokens": 0,
        "prefill_last_target_ns": 0,
        "prefill_last_mtp_sync_ns": 0,
        "prefill_last_mtp_draft_ns": 0,
        "prefill_last_trim_ns": 0,
    }
    backend._reset_runtime = lambda slot: None
    backend._trim_prefill_cuda_cache = lambda prompt_tokens: None
    calls: list[tuple[list[int], torch.Tensor]] = []

    class _Target:
        def prefill(self, token_ids, *, input_embeds=None, **kwargs):
            del kwargs
            assert input_embeds is not None
            calls.append((list(token_ids), input_embeds.clone()))
            return torch.tensor([0.0, 1.0]), torch.zeros(len(token_ids), 4)

    multimodal = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    backend.model = SimpleNamespace(
        encode_multimodal=lambda prompt_ids, vision_inputs: multimodal.clone(),
    )
    backend._targets = [_Target()]

    result = backend._prefill_slot(
        0,
        [10, 11, 12, 13, 14],
        chunk_size=2,
        vision_inputs=SimpleNamespace(image_cache_keys=("img-a",)),
    )

    assert [tokens for tokens, _embeds in calls] == [[10, 11], [12, 13], [14]]
    torch.testing.assert_close(
        torch.cat([embeds for _tokens, embeds in calls]),
        multimodal,
    )
    assert result == {"anchor": 1, "draft_tokens": []}


def test_flashnext_prefix_snapshot_restores_recurrent_state_without_qsa_copy() -> None:
    """A retained prefix keeps fixed KV storage and restores only small state."""

    backend = object.__new__(FlashNextBackend)
    backend.enable_prefix_cache = True
    backend.num_slots = 1
    backend._slot_tokens = [[]]
    backend._last_logits = [None]
    backend._prefix_cache = [None]
    backend._prefix_cache_tokens = [None]
    backend._prefix_cache_kv_len = [0]
    backend._pending_prefix_hits = {}
    backend._specs = [None]
    backend.stats = {}

    class _Target:
        def __init__(self) -> None:
            self.qsa = torch.full((16, 2), 7.0)
            self.sess = SimpleNamespace(
                gdn={
                    "gdn_0": SimpleNamespace(
                        conv_state=torch.tensor([[1.0, 2.0]]),
                        recurrent_state=torch.tensor([[3.0, 4.0]]),
                        has_previous_state=True,
                    )
                },
                ple_conv_state=torch.tensor([[5.0, 6.0]]),
                rope_next=torch.tensor([11, 12, 13]),
                window=[8, 9],
                pos=4,
                qsa_k={},
                qsa_v={},
                qsa_idx_k={},
            )

        def _zero_state(self, *, clear_kv: bool = True) -> None:
            state = self.sess.gdn["gdn_0"]
            state.conv_state.zero_()
            state.recurrent_state.zero_()
            state.has_previous_state = True
            self.sess.ple_conv_state.zero_()
            self.sess.pos = 0
            self.sess.window = []
            self.sess.rope_next = None
            if clear_kv:
                self.qsa.zero_()

    target = _Target()
    backend._targets = [target]

    logits = torch.tensor([0.1, 0.9])
    backend._capture_prefix_snapshot(
        0,
        [101, 102, 103, 104],
        anchor=17,
        draft_tokens=[18, 19, 20],
        anchor_logits=logits,
        mtp_ready=False,
    )
    target.sess.gdn["gdn_0"].conv_state.zero_()
    target.sess.gdn["gdn_0"].recurrent_state.zero_()
    target.sess.ple_conv_state.zero_()
    target.sess.pos = 0
    target.sess.window = []
    target.sess.rope_next = None
    backend._slot_tokens[0] = []

    qsa_before = target.qsa.clone()
    assert backend._prefix_hit_for_slot([101, 102, 103, 104, 105], 0).effective == 4
    assert backend._prefix_hit_for_slot([101, 999], 0).effective == 0
    backend._reset_runtime(0, preserve_prefix=True)
    backend._restore_prefix_snapshot(0, [101, 102, 103, 104, 105], 4)

    assert torch.equal(target.qsa, qsa_before)
    assert torch.equal(target.sess.gdn["gdn_0"].conv_state, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(target.sess.gdn["gdn_0"].recurrent_state, torch.tensor([[3.0, 4.0]]))
    assert torch.equal(target.sess.ple_conv_state, torch.tensor([[5.0, 6.0]]))
    assert target.sess.pos == 4
    assert target.sess.window == [8, 9]
    assert torch.equal(target.sess.rope_next, torch.tensor([11, 12, 13]))


def test_flashnext_prefix_history_selects_shorter_authenticated_checkpoint() -> None:
    """A compacted prompt can resume from an older checkpoint, not only latest."""

    backend = object.__new__(FlashNextBackend)
    backend.enable_prefix_cache = True
    backend.num_slots = 1
    backend.prefix_cache_checkpoints_per_slot = 4
    backend._slot_tokens = [[]]
    backend._last_logits = [None]
    backend._prefix_cache = [None]
    backend._prefix_cache_history = [[]]
    backend._prefix_cache_tokens = [None]
    backend._prefix_cache_kv_len = [0]
    backend._pending_prefix_hits = {}
    backend._specs = [None]
    backend.stats = {}

    class _Target:
        def __init__(self) -> None:
            self.sess = SimpleNamespace(
                gdn={
                    "gdn_0": SimpleNamespace(
                        conv_state=torch.tensor([[1.0]]),
                        recurrent_state=torch.tensor([[2.0]]),
                        has_previous_state=True,
                    )
                },
                ple_conv_state=torch.tensor([[3.0]]),
                rope_next=None,
                window=[],
                pos=0,
                qsa_k={},
                qsa_v={},
                qsa_idx_k={},
            )

        def _zero_state(self, *, clear_kv: bool = True) -> None:
            del clear_kv
            state = self.sess.gdn["gdn_0"]
            state.conv_state.zero_()
            state.recurrent_state.zero_()
            self.sess.ple_conv_state.zero_()
            self.sess.pos = 0
            self.sess.window = []
            self.sess.rope_next = None

    target = _Target()
    backend._targets = [target]
    logits = torch.tensor([0.1, 0.9])
    backend._capture_prefix_snapshot(
        0,
        [1, 2, 3, 4],
        anchor=5,
        draft_tokens=[],
        anchor_logits=logits,
        mtp_ready=False,
        target_only_reusable=True,
        decode_mode="greedy",
    )
    target.sess.gdn["gdn_0"].conv_state.fill_(7.0)
    target.sess.gdn["gdn_0"].recurrent_state.fill_(8.0)
    target.sess.ple_conv_state.fill_(9.0)
    backend._capture_prefix_snapshot(
        0,
        [1, 2, 3, 4, 5, 6, 7, 8],
        anchor=9,
        draft_tokens=[],
        anchor_logits=logits,
        mtp_ready=False,
        target_only_reusable=True,
        decode_mode="greedy",
    )

    request = [1, 2, 3, 4, 10, 11]
    greedy_key = backend.prefix_cache_key_for_sampling(None, sampled=False)
    assert backend._prefix_hit_for_slot(request, 0, prefix_cache_key=greedy_key).effective == 4
    hit, entry = backend._prepare_prefill_prefix(
        0,
        request,
        prefix_cache_key=greedy_key,
    )
    assert hit == 4
    assert entry is not None and entry.kv_len == 4
    assert torch.equal(target.sess.gdn["gdn_0"].conv_state, torch.tensor([[1.0]]))
    assert torch.equal(target.sess.gdn["gdn_0"].recurrent_state, torch.tensor([[2.0]]))
    assert torch.equal(target.sess.ple_conv_state, torch.tensor([[3.0]]))


def test_flashnext_prefix_history_is_bounded_and_keeps_early_anchor() -> None:
    """History retention is bounded so long prompts cannot clone unbounded state."""

    backend = object.__new__(FlashNextBackend)
    backend.enable_prefix_cache = True
    backend.num_slots = 1
    backend.prefix_cache_checkpoints_per_slot = 3
    backend._prefix_cache = [None]
    backend._prefix_cache_history = [[]]
    backend._prefix_cache_tokens = [None]
    backend._prefix_cache_kv_len = [0]
    backend._specs = [None]
    backend.stats = {}
    backend._targets = [
        SimpleNamespace(
            sess=SimpleNamespace(
                gdn={},
                ple_conv_state=None,
                rope_next=None,
                window=[],
                pos=0,
            )
        )
    ]
    logits = torch.tensor([0.1, 0.9])
    for length in (4, 8, 12, 16, 20):
        backend._capture_prefix_snapshot(
            0,
            list(range(length)),
            anchor=1,
            draft_tokens=[],
            anchor_logits=logits,
            mtp_ready=False,
            target_only_reusable=True,
        )
    retained = backend._prefix_cache_history[0]
    assert len(retained) == 3
    assert [entry.kv_len for entry in retained] == [4, 16, 20]


def test_flashnext_target_only_checkpoint_retains_mtp_cursor_without_proposal() -> None:
    """Teacher-forced history entries must not be mistaken for proposals."""

    backend = object.__new__(FlashNextBackend)
    backend.enable_prefix_cache = True
    backend.num_slots = 1
    backend.prefix_cache_checkpoints_per_slot = 3
    backend._prefix_cache = [None]
    backend._prefix_cache_history = [[]]
    backend._prefix_cache_tokens = [None]
    backend._prefix_cache_kv_len = [0]
    backend.stats = {}
    backend._specs = [
        SimpleNamespace(mtp_session=SimpleNamespace(sync_len=8, pos=8))
    ]
    backend._targets = [
        SimpleNamespace(
            sess=SimpleNamespace(
                gdn={},
                ple_conv_state=None,
                rope_next=None,
                window=[],
            )
        )
    ]

    backend._capture_prefix_snapshot(
        0,
        list(range(8)),
        anchor=1,
        draft_tokens=[],
        anchor_logits=torch.tensor([0.1, 0.9]),
        mtp_ready=False,
        mtp_prefix_ready=True,
        target_only_reusable=True,
    )

    entry = backend._prefix_cache_history[0][0]
    assert entry.mtp_ready is False
    assert entry.mtp_prefix_ready is True
    assert entry.target_only_reusable is True
    assert entry.mtp_sync_len == 8
    assert entry.mtp_pos == 8


def test_vision_prefix_cache_requires_matching_image_keys() -> None:
    backend = object.__new__(FlashNextBackend)
    backend.enable_prefix_cache = True
    backend.num_slots = 1
    backend._prefix_cache = [
        FlashNextPrefixSnapshot(
            token_ids=(101, 102, 103, 104),
            kv_len=4,
            gdn={},
            ple_conv_state=None,
            rope_next=None,
            window=(),
            anchor=17,
            draft_tokens=(),
            anchor_logits=torch.tensor([0.1, 0.9]),
            vision_cache_key=("img-a",),
            mtp_ready=False,
        )
    ]
    backend._specs = [None]

    assert backend._prefix_hit_for_slot([101, 102, 103, 104, 105], 0).effective == 0
    assert (
        backend._prefix_hit_for_slot(
            [101, 102, 103, 104, 105],
            0,
            prefix_cache_key=("img-a",),
        ).effective
        == 4
    )
    assert (
        backend._prefix_hit_for_slot(
            [101, 102, 103, 104, 105],
            0,
            prefix_cache_key=("img-a", "img-b"),
        ).effective
        == 4
    )
    assert (
        backend._prefix_hit_for_slot(
            [101, 102, 103, 104, 105],
            0,
            prefix_cache_key=("img-z",),
        ).effective
        == 0
    )


def test_flashnext_visual_prefix_cache_never_reuses_text_checkpoint() -> None:
    backend = object.__new__(FlashNextBackend)
    backend.enable_prefix_cache = True
    backend.num_slots = 1
    backend._slot_tokens = [[]]
    backend._last_logits = [None]
    backend._prefix_cache = [None]
    backend._prefix_cache_tokens = [None]
    backend._prefix_cache_kv_len = [0]
    backend._pending_prefix_hits = {}
    backend._specs = [None]
    backend.stats = {}
    backend._targets = [
        SimpleNamespace(
            sess=SimpleNamespace(
                gdn={},
                ple_conv_state=None,
                rope_next=None,
                window=[],
                pos=0,
                qsa_k={},
                qsa_v={},
                qsa_idx_k={},
            )
        )
    ]
    backend._capture_prefix_snapshot(
        0,
        [101, 102, 103],
        anchor=104,
        draft_tokens=[],
        anchor_logits=torch.tensor([0.1, 0.9]),
        vision_cache_key=None,
        mtp_ready=False,
    )

    assert backend._prefix_hit_for_slot([101, 102, 103, 105], 0).effective == 3
    assert (
        backend._prefix_hit_for_slot(
            [101, 102, 103, 105],
            0,
            prefix_cache_key=("image-a",),
        ).effective
        == 0
    )


def test_flashnext_visual_prefix_cache_key_must_match() -> None:
    backend = object.__new__(FlashNextBackend)
    backend.enable_prefix_cache = True
    backend.num_slots = 1
    backend._slot_tokens = [[]]
    backend._last_logits = [None]
    backend._prefix_cache = [None]
    backend._prefix_cache_tokens = [None]
    backend._prefix_cache_kv_len = [0]
    backend._pending_prefix_hits = {}
    backend._specs = [None]
    backend.stats = {}
    backend._targets = [
        SimpleNamespace(
            sess=SimpleNamespace(
                gdn={},
                ple_conv_state=None,
                rope_next=None,
                window=[],
                pos=0,
                qsa_k={},
                qsa_v={},
                qsa_idx_k={},
            )
        )
    ]
    backend._capture_prefix_snapshot(
        0,
        [201, 202, 203],
        anchor=204,
        draft_tokens=[],
        anchor_logits=torch.tensor([0.3, 0.7]),
        vision_cache_key=("image-a",),
        mtp_ready=False,
    )

    assert (
        backend._prefix_hit_for_slot(
            [201, 202, 203, 205],
            0,
            prefix_cache_key=("image-a",),
        ).effective
        == 3
    )
    assert (
        backend._prefix_hit_for_slot(
            [201, 202, 203, 205],
            0,
            prefix_cache_key=("image-b",),
        ).effective
        == 0
    )


def test_flashnext_sampled_prefix_key_allows_target_only_checkpoint() -> None:
    backend = object.__new__(FlashNextBackend)
    backend.enable_prefix_cache = True
    backend.num_slots = 1
    backend._prefix_cache = [
        FlashNextPrefixSnapshot(
            token_ids=(301, 302, 303),
            kv_len=3,
            gdn={},
            ple_conv_state=None,
            rope_next=None,
            window=(),
            anchor=304,
            draft_tokens=(),
            anchor_logits=torch.tensor([0.3, 0.7]),
            vision_cache_key=None,
            mtp_ready=False,
        )
    ]
    backend._specs = [object()]

    assert backend._prefix_hit_for_slot([301, 302, 303, 305], 0).effective == 0
    sampled_key = backend.prefix_cache_key_for_sampling(None, sampled=True)
    greedy_key = backend.prefix_cache_key_for_sampling(None, sampled=False)
    assert (
        backend._prefix_hit_for_slot(
            [301, 302, 303, 305], 0, prefix_cache_key=sampled_key
        ).effective
        == 3
    )
    assert (
        backend._prefix_hit_for_slot([301, 302, 303, 305], 0, prefix_cache_key=greedy_key).effective
        == 0
    )


def test_flashnext_sampled_mtp_restore_rewinds_cached_anchor_row() -> None:
    """A sampled cache hit must overwrite the old anchor before proposing."""

    backend = object.__new__(FlashNextBackend)
    backend.num_slots = 1
    backend._prefix_cache = [
        FlashNextPrefixSnapshot(
            token_ids=(1, 2, 3, 4),
            kv_len=4,
            gdn={},
            ple_conv_state=None,
            rope_next=None,
            window=(),
            anchor=5,
            draft_tokens=(6, 7, 8),
            anchor_logits=torch.tensor([0.1, 0.9]),
            mtp_sync_len=4,
            mtp_pos=6,
            mtp_ready=True,
            decode_mode="sampled",
            mtp_teacher_hidden=torch.ones(1, 4),
        )
    ]
    backend._targets = [
        SimpleNamespace(
            sess=SimpleNamespace(
                gdn={},
                ple_conv_state=None,
                window=[],
                pos=0,
                rope_next=None,
            )
        )
    ]
    backend._specs = [
        SimpleNamespace(
            mtp_session=SimpleNamespace(sync_len=0, pos=0),
            verify=SimpleNamespace(),
        )
    ]
    backend.stats = {}

    backend._restore_prefix_snapshot(0, [1, 2, 3, 4, 9], 4)

    assert backend._specs[0].mtp_session.sync_len == 3
    assert backend._specs[0].mtp_session.pos == 3


def test_flashnext_backend_advertises_sampled_mtp() -> None:
    assert FlashNextBackend.supports_sampled_speculative_decode is True


def test_flashnext_shift_teacher_force_embeds_handles_overlap() -> None:
    backend = object.__new__(FlashNextBackend)
    backend.device = torch.device("cpu")
    backend.model = SimpleNamespace(
        embed_tokens=lambda ids: torch.tensor(
            [[float(ids[0]), float(ids[0]) + 0.5]],
            dtype=torch.float32,
        )
    )
    embeds = torch.tensor(
        [
            [1.0, 1.5],
            [2.0, 2.5],
            [3.0, 3.5],
        ],
        dtype=torch.float32,
    )

    shifted = backend._shift_teacher_force_embeds(embeds, anchor=9)

    assert shifted is embeds
    assert torch.equal(
        shifted,
        torch.tensor(
            [
                [2.0, 2.5],
                [3.0, 3.5],
                [9.0, 9.5],
            ],
            dtype=torch.float32,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA allocator")
def test_trim_prefill_cuda_cache_only_for_long_prompts(monkeypatch):
    backend = object.__new__(FlashNextBackend)
    backend.device = torch.device("cuda")
    calls: list[object] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    monkeypatch.setenv("QSR_FLASHNEXT_TRIM_PREFILL_CACHE_TOKENS", "2048")

    backend._trim_prefill_cuda_cache(2047)
    assert calls == []
    backend._trim_prefill_cuda_cache(2048)
    assert calls == [torch.device("cuda"), "empty"]

    monkeypatch.setenv("QSR_FLASHNEXT_TRIM_PREFILL_CACHE", "0")
    backend._trim_prefill_cuda_cache(4096)
    assert calls == [torch.device("cuda"), "empty"]


def test_flashnext_backend_defaults_to_captured_verify_state(monkeypatch):
    monkeypatch.delenv("QSR_FLASHNEXT_RECOMPUTE_VERIFY_STATE", raising=False)

    captured: dict[str, object] = {}

    def fake_new_session(model, device):
        return SimpleNamespace(
            qsa_k_pool={0: torch.empty(16, 1)},
            qsa_v_pool={0: torch.empty(16, 1)},
            qsa_idx_k_pool={0: torch.empty(16, 1)},
            qsa_pooled_k_pool={0: torch.empty(16, 1)},
            gdn={},
            ple_conv_state=None,
            token_buf=torch.empty(1, dtype=torch.long),
            pos_buf=None,
            ends_buf={},
            hc_hidden_buf=None,
            ple_emb_buf=None,
            window=[],
            pos=0,
        )

    def fake_prepare_graph_buffers(*args, **kwargs):
        return None

    def fake_graph_engine(*args, **kwargs):
        return SimpleNamespace()

    def fake_load_flashnext_mtp(*args, **kwargs):
        return SimpleNamespace()

    def fake_spec_engine(*args, **kwargs):
        captured["recompute_recurrent_state"] = kwargs["recompute_recurrent_state"]
        captured["batch_gdn_projections"] = kwargs["batch_gdn_projections"]
        return SimpleNamespace(
            verify=SimpleNamespace(buffers=SimpleNamespace()),
            mtp_continuation_graph=None,
            mtp_proposal_graphs={},
        )

    import runtime.model.flashnext.model as flashnext_model_mod
    import runtime.model.flashnext.mtp as flashnext_mtp_mod
    import runtime.model.flashnext.spec as flashnext_spec_mod

    monkeypatch.setattr(flashnext_model_mod, "new_session", fake_new_session)
    monkeypatch.setattr(flashnext_model_mod, "prepare_graph_buffers", fake_prepare_graph_buffers)
    monkeypatch.setattr(flashnext_model_mod, "FlashNextGraphEngine", fake_graph_engine)
    monkeypatch.setattr(flashnext_mtp_mod, "load_flashnext_mtp", fake_load_flashnext_mtp)
    monkeypatch.setattr(flashnext_spec_mod, "FlashNextSpecEngine", fake_spec_engine)

    model = SimpleNamespace(
        cfg=SimpleNamespace(),
        layers=[
            SimpleNamespace(
                is_qsa=False,
                attn=SimpleNamespace(
                    in_proj_qkvz=nn.Linear(8, 16),
                    out_proj=nn.Linear(16, 8),
                ),
            )
        ],
    )
    backend = FlashNextBackend(
        model,
        num_slots=1,
        max_seq_len=64,
        device="cpu",
        checkpoint_path="dummy",
        enable_mtp=True,
    )

    assert captured["recompute_recurrent_state"] is False
    assert captured["batch_gdn_projections"] is False
    assert backend._cg_status["gdn_projections"] == "per_row"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_memory_breakdown_deduplicates_verify_row_views() -> None:
    backend = object.__new__(FlashNextBackend)
    backend.device = torch.device("cuda")

    class _TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = []

    backend.model = _TinyModel()
    backend._mtp_model = None
    backend._targets = [
        SimpleNamespace(
            sess=SimpleNamespace(
                qsa_k_pool={},
                qsa_v_pool={},
                qsa_idx_k_pool={},
                qsa_pooled_k_pool={},
                qsa_k={},
                qsa_v={},
                qsa_idx_k={},
                gdn={},
                ple_conv_state=None,
                token_buf=None,
                pos_buf=None,
                ends_buf={},
                hc_hidden_buf=None,
                ple_emb_buf=None,
            ),
            _logits=None,
        )
    ]

    recurrent_rows = torch.zeros(2, 3, dtype=torch.float32, device="cuda")
    verify_buffers = SimpleNamespace(
        token_ids=torch.zeros(2, dtype=torch.long, device="cuda"),
        positions=torch.zeros(2, dtype=torch.long, device="cuda"),
        ple_embeddings=torch.zeros(2, 4, dtype=torch.bfloat16, device="cuda"),
        gdn_rows={
            0: [
                SimpleNamespace(
                    conv_state=torch.zeros(1, 2, dtype=torch.bfloat16, device="cuda"),
                    recurrent_state=recurrent_rows[0:1],
                ),
                SimpleNamespace(
                    conv_state=torch.zeros(1, 2, dtype=torch.bfloat16, device="cuda"),
                    recurrent_state=recurrent_rows[1:2],
                ),
            ]
        },
        gdn_work={
            0: SimpleNamespace(
                conv_state=torch.zeros(1, 2, dtype=torch.bfloat16, device="cuda"),
                recurrent_state=torch.zeros(1, 3, dtype=torch.float32, device="cuda"),
            )
        },
        gdn_recurrent_rows={0: recurrent_rows},
        ple_rows=[
            torch.zeros(1, 2, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(1, 2, dtype=torch.bfloat16, device="cuda"),
        ],
    )
    backend._specs = [
        SimpleNamespace(
            mtp_session=SimpleNamespace(
                mtp_k_pool=None,
                mtp_v_pool=None,
                mtp_idx_k_pool=None,
                mtp_pooled_k_pool=None,
                shared_sparse_indices=None,
                shared_sparse_valid=None,
                sparse_graph_buffers=None,
            ),
            verify=SimpleNamespace(
                buffers=verify_buffers,
                _hc_hidden=None,
                _logits=None,
            ),
            mtp_continuation_graph=None,
            mtp_proposal_graphs={},
        )
    ]

    def _nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.untyped_storage().nbytes())

    expected_verify_bytes = sum(
        _nbytes(tensor)
        for tensor in (
            verify_buffers.token_ids,
            verify_buffers.positions,
            verify_buffers.ple_embeddings,
            recurrent_rows,
            verify_buffers.gdn_rows[0][0].conv_state,
            verify_buffers.gdn_rows[0][1].conv_state,
            verify_buffers.gdn_work[0].conv_state,
            verify_buffers.gdn_work[0].recurrent_state,
            verify_buffers.ple_rows[0],
            verify_buffers.ple_rows[1],
        )
    )

    breakdown = backend.memory_breakdown()

    assert breakdown["model_tensor_bytes"] == 0
    assert breakdown["target_session_tensor_bytes"] == 0
    assert breakdown["mtp_verify_state"] == expected_verify_bytes
    assert breakdown["session_tensor_bytes"] == expected_verify_bytes
    assert breakdown["explicit_tensor_bytes"] == expected_verify_bytes
    assert breakdown["torch_allocated"] >= breakdown["explicit_tensor_bytes"]
    assert breakdown["torch_reserved"] >= breakdown["torch_allocated"]
