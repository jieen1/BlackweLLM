"""Regression tests for the scoped BFAttention forward context."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from runtime.backends.bf_attention import (  # noqa: E402
    BFAttention,
    bf_attn_context,
    clear_bf_attn_context,
    get_bf_attn_context,
)

# The KV write goes through vLLM's compiled reshape_and_cache_flash, which is
# registered for the CUDA backend only, so the two KV-cache tests below cannot
# run on CPU. The forward-context tests above them stay CPU-only.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="reshape_and_cache_flash is a CUDA-only vLLM op",
)


class _CopyQueryImpl:
    def forward(self, _layer, query, _key, _value, _cache, _metadata, output):
        output.copy_(query)


def _attention_with_cache(cache: torch.Tensor, kv_cache_dtype: str) -> BFAttention:
    attention = object.__new__(BFAttention)
    nn.Module.__init__(attention)
    attention.layer_name = "layer"
    attention.num_heads = 1
    attention.head_size = 2
    attention.head_size_v = 2
    attention.num_kv_heads = 1
    attention.kv_cache_dtype = kv_cache_dtype
    attention._k_scale = torch.tensor([0.5], device=cache.device)
    attention._v_scale = torch.tensor([0.25], device=cache.device)
    attention.kv_cache = cache
    attention.impl = _CopyQueryImpl()
    return attention


def test_bf_attn_context_is_cleared_after_scope() -> None:
    clear_bf_attn_context()

    with bf_attn_context({"layer": "metadata"}, {"layer": torch.tensor([3])}):
        context = get_bf_attn_context()
        assert context.attn_metadata["layer"] == "metadata"
        assert context.slot_mapping["layer"].tolist() == [3]

    with pytest.raises(RuntimeError, match="without a scoped attention context"):
        get_bf_attn_context()


def test_bf_attn_context_restores_an_outer_scope() -> None:
    clear_bf_attn_context()

    with bf_attn_context({"outer": 1}, {"outer": torch.tensor([1])}):
        with bf_attn_context({"inner": 2}, {"inner": torch.tensor([2])}):
            assert get_bf_attn_context().attn_metadata == {"inner": 2}

        assert get_bf_attn_context().attn_metadata == {"outer": 1}


# KV cache layout is [2, num_blocks, block_size, num_kv_heads, head_dim] so
# that kv_cache[0] / kv_cache[1] are the contiguous K/V slices the C++ op
# requires; slot 1 therefore lands at [., block 0, offset 1, head 0].
@requires_cuda
def test_bf_attention_preserves_bf16_kv_cache_representation() -> None:
    cache = torch.zeros((2, 1, 2, 1, 2), dtype=torch.bfloat16, device="cuda")
    attention = _attention_with_cache(cache, kv_cache_dtype="auto")
    query = torch.tensor([[5.0, 6.0]], dtype=torch.bfloat16, device="cuda")
    key = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16, device="cuda")
    value = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16, device="cuda")

    with bf_attn_context({"layer": object()}, {"layer": torch.tensor([1], device="cuda")}):
        output = attention(query, key, value)

    assert torch.equal(cache[0, 0, 1, 0], key[0])
    assert torch.equal(cache[1, 0, 1, 0], value[0])
    assert torch.equal(output, query)


@requires_cuda
def test_bf_attention_scales_fp8_kv_cache_before_write() -> None:
    cache = torch.zeros((2, 1, 2, 1, 2), dtype=torch.uint8, device="cuda")
    attention = _attention_with_cache(cache, kv_cache_dtype="fp8_e4m3")
    query = torch.tensor([[5.0, 6.0]], dtype=torch.bfloat16, device="cuda")
    key = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16, device="cuda")
    value = torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16, device="cuda")

    with bf_attn_context({"layer": object()}, {"layer": torch.tensor([1], device="cuda")}):
        attention(query, key, value)

    fp8_cache = cache.view(torch.float8_e4m3fn)
    assert torch.equal(fp8_cache[0, 0, 1, 0].float(), torch.tensor([2.0, 4.0], device="cuda"))
    assert torch.equal(fp8_cache[1, 0, 1, 0].float(), torch.tensor([12.0, 16.0], device="cuda"))
