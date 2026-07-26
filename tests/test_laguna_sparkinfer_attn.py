"""CPU-only contract tests for SparkInfer KV descale normalization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from runtime.backends.laguna_cuda_graph import _SparkinferCGExtendImpl
from runtime.backends.laguna_sparkinfer_attn import _paged_descale


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
