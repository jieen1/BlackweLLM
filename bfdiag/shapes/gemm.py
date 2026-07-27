"""bfdiag.shapes.gemm -- dense GEMM M/N/K shapes for Laguna's linear layers.

Convention: a linear layer with weight ``[out_features, in_features]``
computes ``Y[M, out_features] = X[M, in_features] @ W.T``; GEMM shape is
``(M, N, K) = (M, out_features, in_features)``. All N/K values below come
from the real model config (``bfdiag.shapes.model``) and were cross-checked
against the actual safetensors weight tensor shapes on this machine (see
``notes/2026-07-27-bfdiag-shape-derivation.md``), not hand-typed from
architecture folklore.

Real per-layer-group shapes confirmed against
``~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4`` weights::

    full_attention (12 layers):    q_proj.weight=[6144,3072]  (48*128)
    sliding_attention (36 layers): q_proj.weight=[9216,3072]  (72*128)
    both groups:                   k_proj/v_proj.weight=[1024,3072] (8*128)
    o_proj:                        [3072, N*128]   (N = group's qo heads)
    g_proj (per-head gate, config "gating": "per-head"): [N, 3072]
    dense mlp (layer 0 only, mlp_only_layers=[0]):
        gate_proj/up_proj=[12288,3072]  down_proj=[3072,12288]
    MoE router gate (layers 1-47):  [256, 3072]
    shared_expert (layers 1-47):    gate/up=[1024,3072]  down=[3072,1024]
    lm_head / embed_tokens:         [100352, 3072]

DFlash draft model (``poolside/Laguna-S-2.1-DFlash-NVFP4``) has a *different*
attention projection layout -- fused ``qkv_proj``, not separate q/k/v::

    self_attn.qkv_proj.weight = [11264, 3072]
        = 72*128 (q) + 8*128 (k) + 8*128 (v) = 9216 + 1024 + 1024
    self_attn.o_proj.weight = [3072, 9216]
    self_attn.g_proj.weight = [72, 3072]
    mlp.{gate,up}_proj = [12288, 3072], mlp.down_proj = [3072, 12288]  (dense, no MoE)
    fc.weight = [3072, 18432] = [hidden, len(aux_hidden_state_layer_ids) * hidden]
        (EAGLE-style fusion of the 6 captured target-model aux hidden states)

    No lm_head/embed_tokens tensor exists in the draft checkpoint --
    draft_vocab_size == target vocab_size, so the draft model reuses the
    target model's tied lm_head/embed_tokens.

All shapes above were read from the real safetensors header metadata (shape
only, no tensor data materialized) on this machine, not guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

from bfdiag.shapes.model import DraftModelConfig, LagunaModelConfig, LayerGroup


@dataclass(frozen=True)
class GemmShape:
    """One dense GEMM: ``Y[M, N] = X[M, K] @ W[N, K].T``."""

    name: str
    m: int
    n: int
    k: int

    @property
    def weight_shape(self) -> tuple[int, int]:
        return (self.n, self.k)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "x": (self.m, self.k),
            "weight": (self.n, self.k),
            "y": (self.m, self.n),
        }


def attention_proj_gemms(
    group: LayerGroup, *, hidden_size: int, num_tokens: int
) -> list[GemmShape]:
    """q_proj / k_proj / v_proj / o_proj / g_proj for one attention group
    (full or sliding) -- separate projections, matching the *target* model's
    real layout (see :func:`draft_qkv_proj_gemm` for the draft model's fused
    layout, which is different)."""
    qh, kvh, hd = group.num_qo_heads, group.num_kv_heads, group.head_dim
    return [
        GemmShape(f"{group.kind}.q_proj", m=num_tokens, n=qh * hd, k=hidden_size),
        GemmShape(f"{group.kind}.k_proj", m=num_tokens, n=kvh * hd, k=hidden_size),
        GemmShape(f"{group.kind}.v_proj", m=num_tokens, n=kvh * hd, k=hidden_size),
        GemmShape(f"{group.kind}.o_proj", m=num_tokens, n=hidden_size, k=qh * hd),
        GemmShape(f"{group.kind}.g_proj", m=num_tokens, n=qh, k=hidden_size),
    ]


def dense_mlp_gemms(
    *, hidden_size: int, intermediate_size: int, num_tokens: int, prefix: str = "mlp"
) -> list[GemmShape]:
    """gate_proj / up_proj / down_proj for one dense MLP (target layer 0, or
    any MoE layer's shared_expert -- same shape, different
    ``intermediate_size``)."""
    return [
        GemmShape(f"{prefix}.gate_proj", m=num_tokens, n=intermediate_size, k=hidden_size),
        GemmShape(f"{prefix}.up_proj", m=num_tokens, n=intermediate_size, k=hidden_size),
        GemmShape(f"{prefix}.down_proj", m=num_tokens, n=hidden_size, k=intermediate_size),
    ]


def router_gemm(*, hidden_size: int, num_experts: int, num_tokens: int) -> GemmShape:
    return GemmShape("moe.router_gate", m=num_tokens, n=num_experts, k=hidden_size)


def lm_head_gemm(*, hidden_size: int, vocab_size: int, num_tokens: int) -> GemmShape:
    return GemmShape("lm_head", m=num_tokens, n=vocab_size, k=hidden_size)


def target_dense_gemms(config: LagunaModelConfig, *, num_tokens: int) -> list[GemmShape]:
    """All dense (non-expert) GEMMs for the target model at a given token
    count: qkv/o/g proj for both attention groups, layer-0's dense MLP, the
    MoE router + one representative shared_expert MLP, and lm_head."""
    gemms: list[GemmShape] = []
    for group in config.groups.values():
        gemms.extend(
            attention_proj_gemms(group, hidden_size=config.hidden_size, num_tokens=num_tokens)
        )
    gemms.extend(
        dense_mlp_gemms(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_tokens=num_tokens,
            prefix="layer0.dense_mlp",
        )
    )
    if config.num_experts:
        gemms.append(
            router_gemm(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                num_tokens=num_tokens,
            )
        )
        gemms.extend(
            dense_mlp_gemms(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                num_tokens=num_tokens,
                prefix="moe_layer.shared_expert",
            )
        )
    gemms.append(
        lm_head_gemm(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            num_tokens=num_tokens,
        )
    )
    return gemms


def draft_qkv_proj_gemm(config: DraftModelConfig, *, num_tokens: int) -> GemmShape:
    """Draft model's *fused* qkv_proj -- real checkpoint shape
    ``[11264, 3072] = [72*128 + 2*8*128, 3072]``. This is a real architecture
    difference from the target model's separate q/k/v_proj; a hand-typed
    kernel test that copies the target model's per-proj shapes here would be
    wrong in a way that's easy to not notice."""
    qh, kvh, hd = config.num_attention_heads, config.num_key_value_heads, config.head_dim
    n = qh * hd + 2 * kvh * hd
    return GemmShape("draft.qkv_proj", m=num_tokens, n=n, k=config.hidden_size)


def draft_attention_gemms(config: DraftModelConfig, *, num_tokens: int) -> list[GemmShape]:
    qh, hd = config.num_attention_heads, config.head_dim
    return [
        draft_qkv_proj_gemm(config, num_tokens=num_tokens),
        GemmShape("draft.o_proj", m=num_tokens, n=config.hidden_size, k=qh * hd),
        GemmShape("draft.g_proj", m=num_tokens, n=qh, k=config.hidden_size),
    ]


def draft_fc_gemm(config: DraftModelConfig, *, num_tokens: int) -> GemmShape:
    """EAGLE-style aux-hidden-state fusion: real checkpoint shape
    ``fc.weight = [3072, 18432]`` = ``[hidden_size, len(aux_hidden_state_layer_ids)
    * hidden_size]`` (6 captured target-model layers, concatenated, projected
    back to hidden_size)."""
    n_aux = len(config.eagle_aux_hidden_state_layer_ids)
    return GemmShape(
        "draft.fc", m=num_tokens, n=config.hidden_size, k=n_aux * config.hidden_size
    )


def draft_dense_gemms(config: DraftModelConfig, *, num_tokens: int) -> list[GemmShape]:
    """All dense GEMMs for one draft-model layer's worth of work, plus the
    shared ``fc`` fusion projection (once, not per-layer). No lm_head here --
    the draft model reuses the target model's tied lm_head/embed_tokens."""
    gemms = draft_attention_gemms(config, num_tokens=num_tokens)
    gemms.extend(
        dense_mlp_gemms(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_tokens=num_tokens,
            prefix="draft.mlp",
        )
    )
    gemms.append(draft_fc_gemm(config, num_tokens=num_tokens))
    return gemms
