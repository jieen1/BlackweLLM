from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
    pytest.skip(
        "Qwen3.8 native F32 attention requires an SM120 CUDA device", allow_module_level=True
    )

from runtime.kernels.gguf_paged_attention import paged_f32_attention  # noqa: E402


@pytest.fixture(autouse=True)
def _release_native_attention_cuda_cache():
    yield
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def _reference_attention(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    batch_size, query_tokens, num_q_heads, head_dim = 2, 3, 24, 256
    num_kv_heads = 4
    gqa = num_q_heads // num_kv_heads
    rows = []
    for request in range(batch_size):
        physical_page = int(page_table[request, 0].item())
        cache_len = int(cache_seqlens[request].item())
        keys = k_cache[physical_page, :cache_len].transpose(0, 1)
        values = v_cache[physical_page, :cache_len].transpose(0, 1)
        request_query = query[request * query_tokens : (request + 1) * query_tokens]
        scores = (
            torch.einsum(
                "qhd,hkd->qhk",
                request_query,
                keys.repeat_interleave(gqa, dim=0),
            )
            / head_dim**0.5
        )
        scores = scores.masked_fill(
            torch.arange(cache_len, device=query.device)[None, None, :]
            > positions[request * query_tokens : (request + 1) * query_tokens, None, None],
            float("-inf"),
        )
        rows.append(
            torch.einsum(
                "qhk,hkd->qhd", scores.softmax(dim=-1), values.repeat_interleave(gqa, dim=0)
            )
        )
    return torch.cat(rows, dim=0)


def test_native_f32_attention_matches_reference_and_replays_graph() -> None:
    device = torch.device("cuda")
    batch_size, query_tokens = 2, 3
    num_q_heads, num_kv_heads, head_dim, page_size = 24, 4, 256, 128
    query = torch.randn(batch_size * query_tokens, num_q_heads, head_dim, device=device)
    k_cache = torch.randn(2, page_size, num_kv_heads, head_dim, device=device)
    v_cache = torch.randn_like(k_cache)
    page_table = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([3, 3], dtype=torch.int32, device=device)
    positions = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int32, device=device)
    output = paged_f32_attention(
        query,
        k_cache,
        v_cache,
        page_table,
        cache_seqlens,
        positions,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        tokens_per_request=query_tokens,
        output=torch.empty_like(query),
    )
    expected = _reference_attention(query, k_cache, v_cache, page_table, cache_seqlens, positions)
    torch.cuda.synchronize()
    assert torch.cosine_similarity(output.flatten(), expected.flatten(), dim=0).item() > 0.99999

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        graph.capture_begin()
        paged_f32_attention(
            query,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            positions,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            tokens_per_request=query_tokens,
            output=output,
        )
        graph.capture_end()
        graph.replay()
    capture_stream.synchronize()
    assert torch.isfinite(output).all()
