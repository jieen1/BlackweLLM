"""Runner configuration for the DeepSeek-V4-Flash (GGUF) model graph.

The GGUF header KV is the config source (there is no config.json). Values
here are the verified facts from notes/2026-08-07-dsv4flash-fact-baseline.md;
``config_from_gguf_kv`` reads them out of a parsed header and refuses
unsupported shapes the same way the registry does -- an unsupported
checkpoint must fail before a single weight is read.

This is the DSV4 sibling of ``model/qwen36_config.py``: a narrow contract
reader consumed by the model graph, deliberately separate from
``runtime.architecture.ArchitectureSpec`` (which decides *which* backend).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from runtime.architecture import UnsupportedArchitectureError

#: Compression ratios with implementations (reference + llama.cpp agree).
SUPPORTED_RATIOS = (0, 4, 128)


@dataclass(frozen=True)
class Dsv4Config:
    """Everything the DSV4 model graph needs, nothing it does not."""

    vocab_size: int = 129280
    hidden_size: int = 4096
    num_layers: int = 43
    max_position_embeddings: int = 1048576
    norm_eps: float = 1e-6

    # MLA-variant attention: q via low-rank projection into 64 heads over a
    # 512-dim latent; single latent KV per token (448 nope + 64 rope).
    num_heads: int = 64
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128

    #: Per main layer: 0 = window-only, 4 = CSA (window + indexer top-k over
    #: seq/4 compressed entries), 128 = HCA (window + all seq/128 entries).
    #: 43 entries; the GGUF array's trailing MTP-stage entries are excluded.
    compress_ratios: tuple[int, ...] = field(default_factory=tuple)

    # RoPE: YaRN on attention and compressed KV (different thetas); window-only
    # layers use the base theta without YaRN (verified reference behavior).
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    rope_original_seq_len: int = 65536
    beta_fast: int = 32
    beta_slow: int = 1
    compress_rope_theta: float = 160000.0

    # Indexer (ratio-4 layers only): scores compressed K with its own low-rank
    # query + per-head weights, keeps top-k positions.
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512

    # Hyper-Connections: hc_mult residual streams, Sinkhorn-projected mixing.
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # MoE: 256 routed + 1 shared, top-6, sqrtsoftplus scoring with noaux_tc
    # selection bias, renormalized weights scaled by route_scale. swiglu_limit
    # clamps up to [-limit, limit] and gate to (-inf, limit] (reference
    # Expert.forward -- note the asymmetric clamp).
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    moe_intermediate_size: int = 2048
    route_scale: float = 1.5
    swiglu_limit: float = 10.0

    #: Layers below this index route by tid2eid[token_id] instead of top-k
    #: (weights are still gathered from gate logits -- "skip selection, not
    #: the gate"; verified bit-exact against the reference).
    n_hash_layers: int = 3

    @property
    def nope_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def hc_dim(self) -> int:
        return self.hc_mult * self.hidden_size

    @property
    def hc_mix_dim(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    @property
    def hash_layer_ids(self) -> tuple[int, ...]:
        return tuple(range(min(self.n_hash_layers, self.num_layers)))

    def layer_ratio(self, layer_id: int) -> int:
        return self.compress_ratios[layer_id]

    def has_compressor(self, layer_id: int) -> bool:
        return self.layer_ratio(layer_id) != 0

    def has_indexer(self, layer_id: int) -> bool:
        return self.layer_ratio(layer_id) == 4

    def compressor_coeff(self, layer_id: int) -> int:
        """Overlap doubles the compressor width on ratio-4 layers."""
        return 2 if self.layer_ratio(layer_id) == 4 else 1

    def validate(self) -> None:
        if len(self.compress_ratios) != self.num_layers:
            raise UnsupportedArchitectureError(
                f"compress_ratios has {len(self.compress_ratios)} entries but "
                f"num_layers is {self.num_layers}"
            )
        for index, ratio in enumerate(self.compress_ratios):
            if ratio not in SUPPORTED_RATIOS:
                raise UnsupportedArchitectureError(
                    f"layer {index} has compress_ratio {ratio}; supported "
                    f"ratios are {list(SUPPORTED_RATIOS)}"
                )
        # Load-bearing fields must be present in the GGUF metadata: a zero
        # here does not mean "absent by design", it means the file does not
        # describe this model, and empty modules would build silently.
        required = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "n_routed_experts": self.n_routed_experts,
            "n_activated_experts": self.n_activated_experts,
            "moe_intermediate_size": self.moe_intermediate_size,
        }
        missing = sorted(name for name, value in required.items() if value <= 0)
        if missing:
            raise UnsupportedArchitectureError(
                f"GGUF metadata is missing or zero for required fields: {missing}"
            )


def config_from_gguf_kv(kv: dict) -> Dsv4Config:
    """Build and validate the runner config from parsed GGUF metadata."""
    num_layers = int(kv.get("deepseek4.block_count", 0))
    ratios_raw = kv.get("deepseek4.attention.compress_ratios")
    if not isinstance(ratios_raw, list) or len(ratios_raw) < num_layers:
        raise UnsupportedArchitectureError(
            "GGUF 'deepseek4.attention.compress_ratios' is missing or shorter "
            f"than block_count ({num_layers})"
        )
    tokens = kv.get("tokenizer.ggml.tokens")
    config = Dsv4Config(
        vocab_size=len(tokens) if isinstance(tokens, list) else 0,
        hidden_size=int(kv.get("deepseek4.embedding_length", 0)),
        num_layers=num_layers,
        max_position_embeddings=int(kv.get("deepseek4.context_length", 0)),
        norm_eps=float(kv.get("deepseek4.attention.layer_norm_rms_epsilon", 1e-6)),
        num_heads=int(kv.get("deepseek4.attention.head_count", 0)),
        head_dim=int(kv.get("deepseek4.attention.key_length", 0)),
        rope_head_dim=int(kv.get("deepseek4.rope.dimension_count", 64)),
        q_lora_rank=int(kv.get("deepseek4.attention.q_lora_rank", 1024)),
        o_groups=int(kv.get("deepseek4.attention.output_group_count", 8)),
        o_lora_rank=int(kv.get("deepseek4.attention.output_lora_rank", 1024)),
        window_size=int(kv.get("deepseek4.attention.sliding_window", 128)),
        compress_ratios=tuple(int(r) for r in ratios_raw[:num_layers]),
        rope_theta=float(kv.get("deepseek4.rope.freq_base", 10000.0)),
        rope_factor=float(kv.get("deepseek4.rope.scaling.factor", 16.0)),
        rope_original_seq_len=int(kv.get("deepseek4.rope.scaling.original_context_length", 65536)),
        beta_fast=int(kv.get("deepseek4.rope.scaling.yarn_beta_fast", 32)),
        beta_slow=int(kv.get("deepseek4.rope.scaling.yarn_beta_slow", 1)),
        compress_rope_theta=float(kv.get("deepseek4.attention.compress_rope_freq_base", 160000.0)),
        index_n_heads=int(kv.get("deepseek4.attention.indexer.head_count", 64)),
        index_head_dim=int(kv.get("deepseek4.attention.indexer.key_length", 128)),
        index_topk=int(kv.get("deepseek4.attention.indexer.top_k", 512)),
        hc_mult=int(kv.get("deepseek4.hyper_connection.count", 4)),
        hc_sinkhorn_iters=int(kv.get("deepseek4.hyper_connection.sinkhorn_iterations", 20)),
        hc_eps=float(kv.get("deepseek4.hyper_connection.epsilon", 1e-6)),
        n_routed_experts=int(kv.get("deepseek4.expert_count", 0)),
        n_shared_experts=int(kv.get("deepseek4.expert_shared_count", 0)),
        n_activated_experts=int(kv.get("deepseek4.expert_used_count", 0)),
        moe_intermediate_size=int(kv.get("deepseek4.expert_feed_forward_length", 0)),
        route_scale=float(kv.get("deepseek4.expert_weights_scale", 1.0)),
        n_hash_layers=int(kv.get("deepseek4.hash_layer_count", 0)),
    )
    config.validate()
    return config
