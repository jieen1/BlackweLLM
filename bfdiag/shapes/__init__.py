"""bfdiag.shapes -- derive kernel-isolation-test shapes from the real model config.

The point: stop hand-typing ``num_heads=48, head_dim=128, ...`` into a
microbenchmark. Instead::

    from bfdiag.shapes import model_shapes
    S = model_shapes(block_size=128)
    q, k, v, pt, seqlens = S.decode_attention(group="sliding", kv_len=65536).empty_tensors()

See ``notes/2026-07-27-bfdiag-shape-derivation.md`` for the full derivation
rules (with code citations), the block_size=64-vs-128 shape comparison
table, and the GPU-verification TODO list. ``block_size`` (KV page_size) is
always an explicit, required argument -- this package never assumes 64 or
128, it computes both and lets ``bf shapes --diff`` show what changed.

Submodules:

- ``bfdiag.shapes.model``: parses the real ``config.json`` (target + DFlash
  draft) into structural dataclasses. No hardcoded architecture defaults --
  missing fields raise :class:`~bfdiag.shapes.model.LagunaConfigError`.
- ``bfdiag.shapes.attention``: decode/verify/prefill attention shapes,
  ``block_size`` explicit, SWA ring math re-derived independently of
  ``runtime/backends/laguna.py``/``laguna_cuda_graph.py`` (read-only
  references, not imported).
- ``bfdiag.shapes.moe``: MoE router + NVFP4-packed expert weight shapes.
- ``bfdiag.shapes.gemm``: dense GEMM M/N/K for qkv/o/g proj, MLPs, lm_head.
- ``bfdiag.shapes.harness``: realize any of the above as real (CPU-only)
  ``torch`` tensors.
- ``bfdiag.shapes.cli``: ``bf shapes`` / ``bf shapes --diff``.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfdiag.shapes.attention import (
    SWA_QO_MAX,
    AttentionCallShape,
    PrefillSwaScratch,
    full_attention_call,
    prefill_swa_scratch,
    ring_blocks_for_window,
    swa_attention_call,
)
from bfdiag.shapes.attention import (
    kv_cache_shape as _attention_kv_cache_shape,
)
from bfdiag.shapes.gemm import (
    GemmShape,
    draft_dense_gemms,
    target_dense_gemms,
)
from bfdiag.shapes.model import (
    DEFAULT_DRAFT_MODEL_ID,
    DEFAULT_MODEL_ID,
    DraftModelConfig,
    LagunaConfigError,
    LagunaModelConfig,
    LayerGroup,
    load_draft_config,
    load_laguna_config,
)
from bfdiag.shapes.moe import (
    Nvfp4PackedGemm,
    expert_projection_shapes,
    router_shapes,
    shared_expert_gemms,
    sparkinfer_w13_shapes,
    stacked_expert_shapes,
)

try:
    # The one real runtime constant this package can't derive from
    # config.json (K=15 speculative tokens / qo_len=16 per verify round is a
    # project hyperparameter, not a model architecture field). Read straight
    # from runtime/backends/dflash_constants.py -- a torch/vllm-free module
    # written explicitly so CPU-only tools can import it (see its own
    # docstring). This is a read; nothing in runtime/ is modified.
    from runtime.backends.dflash_constants import NUM_QUERY_PER_REQ, NUM_SPECULATIVE_TOKENS
except ImportError as _exc:  # pragma: no cover - only if run outside the repo root
    raise LagunaConfigError(
        "could not import runtime.backends.dflash_constants (NUM_QUERY_PER_REQ). "
        "bfdiag.shapes must be run with the repo root on sys.path."
    ) from _exc

_GROUP_ALIASES = {"full": "full", "sliding": "sliding", "swa": "sliding"}


def _resolve_group(config: LagunaModelConfig, group: str) -> LayerGroup:
    key = _GROUP_ALIASES.get(group)
    if key is None or key not in config.groups:
        raise ValueError(
            f"unknown attention group {group!r}; valid groups for {config.model_id}: "
            f"{sorted(_GROUP_ALIASES)} (have {sorted(config.groups)} in this config)"
        )
    return config.groups[key]


@dataclass(frozen=True)
class ModelShapes:
    """Front door: one object per ``block_size``, all shape derivations hang off it."""

    block_size: int
    config: LagunaModelConfig
    draft_config: DraftModelConfig

    # ---- main model attention ----

    def decode_attention(
        self, *, group: str, kv_len: int, batch_size: int = 1
    ) -> AttentionCallShape:
        """M=1 decode step. ``kv_len`` is the pre-existing committed context
        length (matches ``LagunaCudaGraphDecode._fill_buffers_b1``'s ``kv_len``
        parameter exactly -- the call being shaped decodes token ``kv_len``,
        extending context to ``kv_len + 1``)."""
        g = _resolve_group(self.config, group)
        if g.window is None:
            return full_attention_call(
                label=f"decode/{g.kind}",
                num_qo_heads=g.num_qo_heads,
                num_kv_heads=g.num_kv_heads,
                head_dim=g.head_dim,
                block_size=self.block_size,
                kv_len=kv_len,
                qo_len=1,
                batch_size=batch_size,
            )
        return swa_attention_call(
            label=f"decode/{g.kind}",
            num_qo_heads=g.num_qo_heads,
            num_kv_heads=g.num_kv_heads,
            head_dim=g.head_dim,
            block_size=self.block_size,
            kv_len=kv_len,
            window=g.window,
            qo_len=1,
            batch_size=batch_size,
            ring_blocks_per_slot=None,  # real decode path (_fill_buffers_b1) does not cap
        )

    def verify_attention(
        self, *, group: str, kv_len: int, batch_size: int = 1
    ) -> AttentionCallShape:
        """DFlash verify: M=``NUM_QUERY_PER_REQ`` (16 = 1 bonus + 15 draft)
        tokens per request, matching ``LagunaCudaGraphVerify``."""
        g = _resolve_group(self.config, group)
        if g.window is None:
            return full_attention_call(
                label=f"verify/{g.kind}",
                num_qo_heads=g.num_qo_heads,
                num_kv_heads=g.num_kv_heads,
                head_dim=g.head_dim,
                block_size=self.block_size,
                kv_len=kv_len,
                qo_len=NUM_QUERY_PER_REQ,
                batch_size=batch_size,
            )
        ring_cap = self.ring_capacity(group)
        return swa_attention_call(
            label=f"verify/{g.kind}",
            num_qo_heads=g.num_qo_heads,
            num_kv_heads=g.num_kv_heads,
            head_dim=g.head_dim,
            block_size=self.block_size,
            kv_len=kv_len,
            window=g.window,
            qo_len=NUM_QUERY_PER_REQ,
            batch_size=batch_size,
            ring_blocks_per_slot=ring_cap,  # real verify path DOES cap (min(...))
        )

    def prefill_attention(
        self, *, group: str, kv_len_before: int, chunk_tokens: int
    ) -> AttentionCallShape:
        """Chunked prefill for the full-attention group: q/k/v over one chunk
        of ``chunk_tokens`` new tokens, page table covering the whole context
        (``kv_len_before + chunk_tokens``). For the sliding group, prefill
        does NOT use paged attention -- it uses a flat scratch buffer, see
        :meth:`prefill_swa_scratch`; calling this with the sliding group
        raises on purpose."""
        g = _resolve_group(self.config, group)
        if g.window is not None:
            raise ValueError(
                f"prefill_attention(group={group!r}) is a paged full-attention call; the "
                "sliding group's prefill path uses a flat scratch buffer instead "
                "(runtime/backends/laguna.py's _swa_scratch), not a page table -- "
                "use ModelShapes.prefill_swa_scratch(...) for that."
            )
        return full_attention_call(
            label=f"prefill/{g.kind}",
            num_qo_heads=g.num_qo_heads,
            num_kv_heads=g.num_kv_heads,
            head_dim=g.head_dim,
            block_size=self.block_size,
            kv_len=kv_len_before,
            qo_len=chunk_tokens,
            batch_size=1,
        )

    def prefill_swa_scratch(
        self, *, chunk_tokens: int, blocks_per_slot_cap: int | None = None
    ) -> PrefillSwaScratch:
        """The persistent SWA prefill scratch buffer
        (``runtime/backends/laguna.py:305-312``): ``shape = (2,
        min(blocks_per_slot_cap, cdiv(window + chunk_tokens, block_size)),
        block_size, num_kv_heads, head_dim)``."""
        g = self.config.groups.get("sliding")
        if g is None:
            raise ValueError(f"{self.config.model_id} has no sliding_attention group")
        return prefill_swa_scratch(
            block_size=self.block_size,
            num_kv_heads=g.num_kv_heads,
            head_dim=g.head_dim,
            window=g.window,
            chunk_tokens=chunk_tokens,
            blocks_per_slot_cap=blocks_per_slot_cap,
        )

    def ring_capacity(self, group: str) -> int:
        """Static ring-buffer capacity in blocks:
        ``ring_blocks_for_window(window, block_size, qo_max=NUM_QUERY_PER_REQ)``
        -- the fixed-address CUDA-Graph allocation size, independent of any
        particular ``kv_len``. This is the number ``bf shapes --diff`` flags
        first: it does not simply halve when block_size doubles."""
        g = _resolve_group(self.config, group)
        if g.window is None:
            raise ValueError(f"group {group!r} is full attention, has no ring capacity")
        return ring_blocks_for_window(g.window, self.block_size, qo_max=NUM_QUERY_PER_REQ)

    def kv_cache_shape(
        self, *, group: str, num_slots: int, blocks_per_slot: int | None = None
    ) -> tuple[int, int, int, int, int]:
        """The persistent KV cache tensor `[2, num_blocks, block_size,
        num_kv_heads, head_dim]`. For the sliding group ``num_blocks`` is
        fully determined by config (``num_slots * ring_capacity``) -- for
        full attention it is a deployment choice and must be passed in
        explicitly (no silent default; see ``server/app.py``'s
        ``SERVER_BLOCKS_PER_SLOT`` for how the real server picks one)."""
        g = _resolve_group(self.config, group)
        if g.window is None:
            if blocks_per_slot is None:
                raise ValueError(
                    "full-attention KV cache shape requires an explicit blocks_per_slot "
                    "(context-capacity) argument -- this is a deployment choice "
                    "(runtime/backends/laguna.py's blocks_per_slot constructor arg), "
                    "not a model-architecture constant bfdiag.shapes can derive."
                )
            num_blocks = num_slots * blocks_per_slot
        else:
            num_blocks = num_slots * self.ring_capacity(group)
        return _attention_kv_cache_shape(
            block_size=self.block_size,
            num_kv_heads=g.num_kv_heads,
            head_dim=g.head_dim,
            num_blocks=num_blocks,
        )

    # ---- draft model attention ----

    def draft_ring_capacity(self) -> int:
        """``_ring_blocks_for_window(DRAFT_WINDOW, block_size,
        NUM_QUERY_PER_REQ)`` (``runtime/backends/laguna_dflash.py``) -- one
        fixed ring, sized for the worst-case 16-wide verify burst, reused for
        every draft-side attention call (decode-style and verify-style
        alike; see the GPU-verification TODO in the design notes about
        whether the draft model ever issues a true qo_len=1 call)."""
        return ring_blocks_for_window(
            self.draft_config.sliding_window, self.block_size, qo_max=NUM_QUERY_PER_REQ
        )

    def draft_decode_attention(self, *, kv_len: int, batch_size: int = 1) -> AttentionCallShape:
        dc = self.draft_config
        return swa_attention_call(
            label="draft_decode/sliding",
            num_qo_heads=dc.num_attention_heads,
            num_kv_heads=dc.num_key_value_heads,
            head_dim=dc.head_dim,
            block_size=self.block_size,
            kv_len=kv_len,
            window=dc.sliding_window,
            qo_len=1,
            batch_size=batch_size,
            ring_blocks_per_slot=self.draft_ring_capacity(),
        )

    def draft_verify_attention(self, *, kv_len: int, batch_size: int = 1) -> AttentionCallShape:
        dc = self.draft_config
        return swa_attention_call(
            label="draft_verify/sliding",
            num_qo_heads=dc.num_attention_heads,
            num_kv_heads=dc.num_key_value_heads,
            head_dim=dc.head_dim,
            block_size=self.block_size,
            kv_len=kv_len,
            window=dc.sliding_window,
            qo_len=NUM_QUERY_PER_REQ,
            batch_size=batch_size,
            ring_blocks_per_slot=self.draft_ring_capacity(),
        )

    # ---- GEMMs ----

    def dense_gemms(self, *, num_tokens: int) -> list[GemmShape]:
        return target_dense_gemms(self.config, num_tokens=num_tokens)

    def draft_gemms(self, *, num_tokens: int) -> list[GemmShape]:
        return draft_dense_gemms(self.draft_config, num_tokens=num_tokens)

    def shared_expert_gemms(self, *, num_tokens: int) -> list[GemmShape]:
        return shared_expert_gemms(self.config, num_tokens=num_tokens)

    # ---- MoE ----

    def moe_expert_shapes(self) -> dict[str, Nvfp4PackedGemm]:
        return expert_projection_shapes(self.config)

    def moe_stacked_expert_shapes(self) -> dict[str, tuple[int, ...]]:
        return stacked_expert_shapes(self.config)

    def moe_sparkinfer_shapes(self) -> dict[str, tuple[int, ...]]:
        return sparkinfer_w13_shapes(self.config)

    def moe_router_shapes(self, *, num_tokens: int) -> dict[str, tuple[int, ...]]:
        return router_shapes(self.config, num_tokens=num_tokens)


def model_shapes(
    block_size: int,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    draft_model_id: str = DEFAULT_DRAFT_MODEL_ID,
    model_path: str | None = None,
    draft_model_path: str | None = None,
) -> ModelShapes:
    """Build a :class:`ModelShapes` for a given KV page_size (``block_size``),
    reading both the target and DFlash draft model's real ``config.json``.
    Raises :class:`LagunaConfigError` if either config can't be found or is
    missing a required field -- never falls back to a hardcoded default.
    """
    config = load_laguna_config(model_id, path_override=model_path)
    draft_config = load_draft_config(draft_model_id, path_override=draft_model_path)
    return ModelShapes(block_size=block_size, config=config, draft_config=draft_config)


__all__ = [
    "SWA_QO_MAX",
    "AttentionCallShape",
    "DEFAULT_DRAFT_MODEL_ID",
    "DEFAULT_MODEL_ID",
    "DraftModelConfig",
    "GemmShape",
    "LagunaConfigError",
    "LagunaModelConfig",
    "LayerGroup",
    "ModelShapes",
    "NUM_QUERY_PER_REQ",
    "NUM_SPECULATIVE_TOKENS",
    "Nvfp4PackedGemm",
    "PrefillSwaScratch",
    "load_draft_config",
    "load_laguna_config",
    "model_shapes",
    "ring_blocks_for_window",
]
