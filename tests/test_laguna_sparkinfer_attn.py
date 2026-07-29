"""CPU-only contract tests for SparkInfer KV descale normalization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from runtime.backends.laguna_cuda_graph import _SparkinferCGExtendImpl  # noqa: E402
from runtime.backends.laguna_sparkinfer_attn import (  # noqa: E402
    SparkinferPrefillWorkspace,
    _paged_descale,
)


def test_paged_descale_normalizes_rank_zero_dflash_default_scale():
    descale = _paged_descale(torch.tensor(0.5), batch_size=1, num_kv_heads=8)

    assert descale.shape == (1,)
    assert torch.equal(descale, torch.tensor([0.5]))


def test_paged_descale_expands_per_head_scale_for_each_request():
    per_head = torch.arange(1, 9, dtype=torch.float32)
    descale = _paged_descale(per_head, batch_size=2, num_kv_heads=8)

    assert descale.shape == (2, 8)
    assert torch.equal(descale[0], per_head)
    assert torch.equal(descale[1], per_head)


def test_paged_descale_rejects_ambiguous_scale_shape():
    with pytest.raises(ValueError, match="KV descale"):
        _paged_descale(torch.ones(3), batch_size=2, num_kv_heads=8)


def test_cuda_graph_attention_rebinds_when_model_buffers_move(monkeypatch):
    bindings = []

    def build_binding(**kwargs):
        binding = SimpleNamespace(**kwargs)
        bindings.append(binding)
        return binding

    monkeypatch.setattr(
        "sparkinfer.attention.paged._scratch.build_paged_attention_binding",
        build_binding,
    )
    monkeypatch.setattr(
        "sparkinfer.attention.paged._forward.paged_attention_forward",
        lambda *, binding: None,
    )

    impl = _SparkinferCGExtendImpl(workspace=object(), num_tokens=2)
    layer = SimpleNamespace(
        _k_scale=torch.tensor(1.0),
        _v_scale=torch.tensor(1.0),
    )
    kv_cache = torch.zeros((2, 2, 64, 1, 4), dtype=torch.uint8)
    key = value = torch.empty((2, 1, 4), dtype=torch.bfloat16)
    query = torch.empty((2, 1, 4), dtype=torch.bfloat16)
    output = torch.empty_like(query)
    metadata = SimpleNamespace(num_actual_tokens=2)

    impl.forward(layer, query, key, value, kv_cache, metadata, output)
    impl.forward(layer, query, key, value, kv_cache, metadata, output)
    assert len(bindings) == 1

    moved_query = torch.empty_like(query)
    moved_output = torch.empty_like(output)
    impl.forward(
        layer,
        moved_query,
        key,
        value,
        kv_cache,
        metadata,
        moved_output,
    )

    assert len(bindings) == 2
    assert bindings[-1].q.data_ptr() == moved_query.data_ptr()
    assert bindings[-1].output.data_ptr() == moved_output.data_ptr()


def test_prefill_workspace_reuses_plan_only_within_one_metadata_object(monkeypatch):
    """Layer-group sharing must not reuse a plan after the next forward."""
    calls: list[str] = []

    class FakeWorkspace:
        @classmethod
        def for_tensors(cls, **kwargs):
            calls.append("workspace")
            return cls()

        def _ensure_capacity(self, plan):
            calls.append("capacity")

        def _copy_runtime_metadata(self, *args):
            calls.append("runtime")

        def _copy_plan_metadata(self, plan):
            calls.append("plan_metadata")

    monkeypatch.setattr(
        "sparkinfer.attention.paged.workspace.PagedAttentionWorkspace",
        FakeWorkspace,
    )
    monkeypatch.setattr(
        "sparkinfer.attention.paged.planner.create_paged_plan",
        lambda *args, **kwargs: calls.append("plan") or object(),
    )
    monkeypatch.setattr(
        "sparkinfer.attention.paged._scratch.build_paged_attention_binding",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "sparkinfer.attention.paged._forward.paged_attention_forward",
        lambda *, binding: calls.append("forward"),
    )

    workspace = SparkinferPrefillWorkspace(torch.device("cpu"))
    q = torch.empty(2, 1, 4)
    cache = torch.empty(3, 2, 1, 4)
    output = torch.empty_like(q)
    metadata = object()

    def run(cache_key):
        workspace.forward(
            q=q,
            k_cache=cache,
            v_cache=cache,
            output=output,
            page_table=torch.empty(1, 2, dtype=torch.int32),
            cache_seqlens=torch.empty(1, dtype=torch.int32),
            cu_seqlens_q=torch.empty(2, dtype=torch.int32),
            plan_cache_key=cache_key,
        )

    run(metadata)
    run(metadata)
    run(object())

    assert calls.count("workspace") == 1
    assert calls.count("plan") == 2
    assert calls.count("runtime") == 2
    assert calls.count("plan_metadata") == 2
    assert calls.count("forward") == 3
