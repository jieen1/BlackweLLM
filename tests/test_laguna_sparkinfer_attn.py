"""CPU-only contract tests for SparkInfer KV descale normalization."""

from __future__ import annotations

import pytest
import torch

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
