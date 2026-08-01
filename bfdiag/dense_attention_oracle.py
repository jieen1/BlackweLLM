"""Dense (non-paged, non-split-KV) reference attention impl.

Root-cause probe for the open question in
notes/2026-08-02-eager-verify-cg-verify-divergence.md: does DFlash's eager
verify path (fresh, fine split-KV chunking) or its CUDA-Graph path (frozen,
coarse -- in practice always single-chunk -- chunking) match a dense,
unchunked attention reference?  This is a from-scratch fp32
matmul -> mask -> softmax -> matmul.  It never calls
``sparkinfer.attention.paged.planner.create_paged_plan`` or
``paged_attention_forward`` at all, so it cannot inherit any split-KV
chunk-boundary or merge-kernel bug either path might have -- that
independence is the entire point (see
``bfdiag.workloads.diagnose_dflash_verify_dense_oracle``, which wires this
in as a temporary ``BFAttention.impl`` replacement for the full-attention
layer group only, then restores the original impl).

Not placed under ``oracle/``: that tree is the read-only reference-capture
scaffold for the separate vLLM-vs-engine divergence effort
(``oracle/vllm_reference.py``'s docstring says as much). This module is
specific to the DFlash verify investigation and needs to live somewhere
that gets edited, so it goes in ``bfdiag/`` instead.

Deliberately narrow: only supports what this investigation needs --
window_left=-1 (full attention, not SWA -- the prior note already ruled
out the SWA window as a factor; this dense reference is not built for the
SWA ring buffer's wrapped page layout), a single request (batch=1, which is
DFlash verify's only real shape), and FP8 e4m3 KV cache (Laguna's only KV
cache dtype in production, see server/app.py's default cache config).
"""

from __future__ import annotations

import torch


def _dequant_scale(scale: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Reshape a vLLM-style KV scale to broadcast against ``[T, num_kv_heads, head_dim]``.

    Mirrors ``runtime/backends/laguna_sparkinfer_attn.py::_paged_descale``'s
    accepted shapes (scalar or per-kv-head) minus the batch dimension --
    this module's gather already collapses everything to one flat
    ``[T, H, D]`` tensor (batch is always 1 here).
    """
    scale = scale.detach().to(dtype=torch.float32)
    count = scale.numel()
    if count == 1:
        return scale.reshape(1, 1, 1)
    if count == num_kv_heads:
        return scale.reshape(1, num_kv_heads, 1)
    raise ValueError(
        "dense oracle only supports scalar or per-kv-head KV descale, got shape "
        f"{tuple(scale.shape)} for num_kv_heads={num_kv_heads}"
    )


class DenseCausalOracleImpl:
    """Reference attention: gather paged KV, dequantize, plain causal fp32
    attention. No split-KV, no chunking, no paged-attention CuTe kernel.

    Same call signature as
    ``runtime.backends.laguna_sparkinfer_attn.SparkinferAttentionImpl.forward``
    / ``runtime.backends.laguna_cuda_graph._SparkinferCGExtendImpl.forward``
    so it drops in as a ``BFAttention.impl`` replacement without touching
    ``BFAttention.forward`` itself (which does the KV cache *write* directly
    via ``fused_kv_scatter``, unconditional on which impl is installed --
    only the attention *read*/compute below is swapped).

    ``gqa_head_mapping`` picks how query heads map to KV heads when
    expanding K/V for GQA:
      - ``"interleaved"`` (default): query head ``h`` reads KV head
        ``h // gqa_group_size`` (the near-universal HF/vLLM ``repeat_kv``
        convention -- ``torch.repeat_interleave``).
      - ``"blocked"``: query head ``h`` reads KV head ``h % num_kv_heads``
        (``torch.Tensor.repeat``).
    sparkinfer's actual CuTe kernel convention is not visible from Python
    (it's compiled device code); ``diagnose_dflash_verify_dense_oracle``
    calibrates this choice empirically at kv_len=64 (the one point already
    proven bit-exact between CG and eager) before trusting any comparison
    at a diverging kv_len -- see that function's docstring.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        *,
        gqa_head_mapping: str = "interleaved",
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        if gqa_head_mapping not in ("interleaved", "blocked"):
            raise ValueError(f"unknown gqa_head_mapping {gqa_head_mapping!r}")
        self.gqa_head_mapping = gqa_head_mapping

    def process_weights_after_loading(self, act_dtype):
        pass

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        # BFAttention.forward never calls this (it writes the cache itself
        # before invoking impl.forward) -- kept only for interface parity
        # with SparkinferAttentionImpl in case some other caller expects it.
        from runtime.kernels.fused_kv_scatter import fused_kv_scatter

        k_cache = kv_cache[0].view(torch.float8_e4m3fn)
        v_cache = kv_cache[1].view(torch.float8_e4m3fn)
        fused_kv_scatter(
            key, value, k_cache, v_cache, slot_mapping, layer._k_scale, layer._v_scale
        )

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if attn_metadata is None:
            return output.fill_(0)
        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        if num_actual_tokens == 0:
            return output.fill_(0)
        window_left = int(getattr(attn_metadata, "window_left", -1))
        if window_left >= 0:
            raise NotImplementedError(
                "DenseCausalOracleImpl only supports window_left=-1 (full attention); "
                "SWA layers must keep their normal impl for this investigation"
            )

        page_table = attn_metadata.page_table
        cache_seqlens = attn_metadata.cache_seqlens
        if int(page_table.shape[0]) != 1 or int(cache_seqlens.numel()) != 1:
            raise NotImplementedError("DenseCausalOracleImpl only supports batch=1")

        key_cache, value_cache = kv_cache.unbind(0)
        if key_cache.dtype == torch.uint8:
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
        page_size = int(key_cache.shape[1])
        num_kv_heads = int(key_cache.shape[2])
        head_dim = int(key_cache.shape[3])

        cache_seqlen = int(cache_seqlens[0].item())
        n_blocks = (cache_seqlen + page_size - 1) // page_size
        block_ids = page_table[0, :n_blocks].to(torch.long)
        k_gathered = key_cache[block_ids].reshape(n_blocks * page_size, num_kv_heads, head_dim)
        v_gathered = value_cache[block_ids].reshape(n_blocks * page_size, num_kv_heads, head_dim)
        k_gathered = k_gathered[:cache_seqlen]
        v_gathered = v_gathered[:cache_seqlen]

        k_scale = _dequant_scale(layer._k_scale, num_kv_heads)
        v_scale = _dequant_scale(layer._v_scale, num_kv_heads)
        k_full = k_gathered.float() * k_scale
        v_full = v_gathered.float() * v_scale

        q = query[:num_actual_tokens].float()  # [T, num_q_heads, head_dim]
        num_q_heads = q.shape[1]
        gqa_group_size = num_q_heads // num_kv_heads
        if gqa_group_size * num_kv_heads != num_q_heads:
            raise ValueError("num_q_heads must be an exact multiple of num_kv_heads")

        if self.gqa_head_mapping == "interleaved":
            k_exp = k_full.repeat_interleave(gqa_group_size, dim=1)
            v_exp = v_full.repeat_interleave(gqa_group_size, dim=1)
        else:
            k_exp = k_full.repeat(1, gqa_group_size, 1)
            v_exp = v_full.repeat(1, gqa_group_size, 1)

        kv_len_before = cache_seqlen - num_actual_tokens
        q_pos = kv_len_before + torch.arange(num_actual_tokens, device=q.device)
        kv_pos = torch.arange(cache_seqlen, device=q.device)
        causal_mask = kv_pos.unsqueeze(0) <= q_pos.unsqueeze(1)  # [T, KV], True = allowed

        qh = q.transpose(0, 1)  # [num_q_heads, T, head_dim]
        kh = k_exp.transpose(0, 1)  # [num_q_heads, KV, head_dim]
        vh = v_exp.transpose(0, 1)
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, vh)  # [num_q_heads, T, head_dim]
        out = out.transpose(0, 1).to(output.dtype)  # [T, num_q_heads, head_dim]
        output[:num_actual_tokens] = out
        return output
