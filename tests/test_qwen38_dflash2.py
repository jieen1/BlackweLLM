import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

import runtime.model.qwen38_dflash2 as qwen38_dflash2  # noqa: E402
from runtime.backends.flashinfer_dspark_attn import compact_dflash_window  # noqa: E402
from runtime.dflash2_config import DFlash2DraftConfig  # noqa: E402
from runtime.model.qwen38_dflash2 import (  # noqa: E402
    DFlash2CandidateSelector,
    Qwen38DFlash2DecoderLayer,
    _dflash2_flashinfer_topk,
    _dflash2_hf_config,
    _dflash2_topk,
    _grouped_dynamic_convolve,
    _score_edges,
)
from tests.test_dflash2_config import _config  # noqa: E402


def _official_grouped_conv(
    hidden: torch.Tensor,
    dynamic: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    groups = hidden.shape[-1] // group_size
    blocks = hidden.reshape(-1, groups, group_size)
    taps = base.shape[0]
    dynamic = dynamic.reshape(-1, taps, groups, 1)
    output = torch.zeros_like(blocks)
    positions = torch.arange(hidden.reshape(-1, hidden.shape[-1]).shape[0])
    if block_size & (block_size - 1) == 0:
        positions = positions & (block_size - 1)
    else:
        positions = positions.remainder(block_size)
    for tap in range(taps):
        if tap == 0:
            values = blocks
        else:
            values = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
            values = values * (positions >= tap).view(-1, 1, 1)
        kernel = base[tap].reshape(1, groups, group_size).to(hidden.dtype)
        output = output + kernel * values
        output = torch.addcmul(output, dynamic[:, tap], values)
    return output.flatten(-2).reshape_as(hidden)


def test_grouped_dynamic_convolution_matches_official_block_reset_equation():
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(2, 8, 8, generator=generator)
    dynamic = torch.randn(2, 8, 2, 4, generator=generator)
    base = torch.randn(2, 8, generator=generator)
    actual = _grouped_dynamic_convolve(hidden, dynamic, base, 2, 8)
    expected = _official_grouped_conv(hidden, dynamic, base, 8, 2)
    torch.testing.assert_close(actual, expected)


def test_grouped_dynamic_convolution_resets_at_non_power_of_two_blocks():
    generator = torch.Generator().manual_seed(12)
    hidden = torch.randn(2, 6, 6, generator=generator)
    dynamic = torch.randn(2, 6, 3, 3, generator=generator)
    base = torch.randn(3, 6, generator=generator)
    actual = _grouped_dynamic_convolve(hidden, dynamic, base, 2, 6)
    expected = _official_grouped_conv(hidden, dynamic, base, 6, 2)
    torch.testing.assert_close(actual, expected)


def test_grouped_dynamic_convolution_preserves_bfloat16_reference_order():
    """Keep the official multiply/addcmul order at the draft's real dtype."""

    generator = torch.Generator().manual_seed(19)
    hidden = torch.randn(2, 8, 8, generator=generator, dtype=torch.bfloat16)
    dynamic = torch.randn(2, 8, 2, 1, generator=generator, dtype=torch.bfloat16)
    base = torch.randn(2, 8, generator=generator, dtype=torch.bfloat16)
    actual = _grouped_dynamic_convolve(hidden, dynamic, base, 8, 8)
    expected = _official_grouped_conv(hidden, dynamic, base, 8, 8)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_selector_greedy_path_matches_official_edge_score_reference():
    config = DFlash2DraftConfig.from_dict(_config())
    with torch.random.fork_rng():
        torch.manual_seed(17)
        selector = DFlash2CandidateSelector(config)
    generator = torch.Generator().manual_seed(13)
    hidden = torch.randn(2, 4, config.hidden_size, generator=generator)
    logits = torch.randn(2, 4, config.vocab_size, generator=generator)
    anchor = torch.tensor([3, 7])

    path, candidates, _ = selector.select(hidden, logits, anchor)
    unary, expected_candidates = torch.topk(logits, config.selector_top_k, dim=-1, sorted=True)
    assert torch.equal(candidates, expected_candidates)
    projected = selector.hidden_projection(hidden)
    edges = _score_edges(
        selector.predecessor_codebook.weight,
        selector.successor_codebook.weight,
        candidates,
        unary,
        projected,
        anchor,
        config.selector_top_k,
    )

    predecessor_index = torch.zeros(hidden.shape[0], dtype=torch.long)
    expected_path = []
    for position in range(hidden.shape[1]):
        choice = (
            edges[:, position]
            .gather(1, predecessor_index[:, None, None].expand(-1, 1, config.selector_top_k))[:, 0]
            .argmax(dim=-1)
        )
        expected_path.append(candidates[:, position].gather(1, choice[:, None])[:, 0])
        predecessor_index = choice
    torch.testing.assert_close(path, torch.stack(expected_path, dim=1))


def test_selector_sampling_returns_normalized_sparse_distributions():
    config = DFlash2DraftConfig.from_dict(_config())
    selector = DFlash2CandidateSelector(config)
    hidden = torch.randn(1, 2, config.hidden_size)
    logits = torch.randn(1, 2, config.vocab_size)
    with torch.no_grad():
        for parameter in selector.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=torch.Generator().manual_seed(15))
            )
    tokens, candidates, probs = selector.select(
        hidden,
        logits,
        torch.tensor([2]),
        temperature=0.7,
        generator=torch.Generator().manual_seed(14),
    )
    assert tokens.shape == (1, 2)
    assert candidates.shape == (1, 2, config.selector_top_k)
    assert probs is not None
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1, 2))


def test_dflash2_topk_preserves_position_axes_for_batched_logits():
    generator = torch.Generator().manual_seed(21)
    logits = torch.randn(2, 3, 17, generator=generator)
    values, indices = _dflash2_topk(logits, 5)
    expected_values, expected_indices = torch.topk(
        logits, 5, dim=-1, sorted=True
    )
    assert values.shape == (2, 3, 5)
    assert indices.shape == (2, 3, 5)
    torch.testing.assert_close(values, expected_values)
    assert torch.equal(indices, expected_indices)


def test_dflash2_flashinfer_topk_flattens_position_axes(monkeypatch):
    calls = []

    def fake_flashinfer_topk(logits, top_k):
        calls.append(logits.shape)
        return torch.topk(logits, top_k, dim=-1, sorted=True)

    monkeypatch.setattr(qwen38_dflash2, "_flashinfer_top_k", fake_flashinfer_topk)
    logits = torch.randn(2, 3, 17)
    values, indices = _dflash2_flashinfer_topk(logits, 5)
    expected_values, expected_indices = torch.topk(logits, 5, dim=-1, sorted=True)
    assert calls == [(6, 17)]
    torch.testing.assert_close(values, expected_values)
    assert torch.equal(indices, expected_indices)


def test_dflash2_decoder_uses_bfloat16_draft_kv_cache():
    config = DFlash2DraftConfig.from_dict(_config())
    layer = Qwen38DFlash2DecoderLayer(
        _dflash2_hf_config(config),
        cache_config=None,
        prefix="model.layers.0",
        layer_idx=0,
        attention_prefix="model.layers.4",
        conv_kernel_size=config.conv_kernel_size,
        conv_group_size=config.conv_group_size,
    )
    attention = layer.self_attn.attn
    assert attention.kv_cache_dtype == "bfloat16"
    assert attention.kv_cache_torch_dtype is torch.bfloat16
    assert not attention.has_checkpoint_kv_scale


@pytest.mark.parametrize(
    ("sequence_length", "expected_local_length", "expected_first_page"),
    [
        (1024, 1024, 0),
        (2048, 2048, 0),
        (2049, 2049, 0),
        (4096, 2048, 32),
        (4097, 2049, 32),
    ],
)
def test_compact_dflash_window_rebases_only_the_attention_page_view(
    sequence_length: int, expected_local_length: int, expected_first_page: int
):
    assert compact_dflash_window(
        sequence_length, window_size=2048, page_size=64
    ) == (expected_local_length, expected_first_page)
