"""Architecture description parsed from a checkpoint's ``config.json``.

Step 3 of the Track A migration (``docs/architecture.md`` §3.5.5), in shadow
mode: this parses and validates, and nothing drives off it yet. The tests
assert it reproduces the values the runtime currently hardcodes, which is the
only claim shadow mode makes.

Deliberately torch-free -- it reads a dict. That keeps it importable in the
CPU-only job, and it is also the point: **an unsupported checkpoint must fail
before a single weight is read**, so this cannot depend on anything that
needs a GPU to import.

This does not replace ``runtime/model_spec.py``. That one is the Qwen3.6-era
runner spec (layer *names* discovered from a live model, plus MTP wiring) and
is still what ``LagunaBackend`` constructs. Step 5 landed
(``runtime.model_registry`` -- which uses this module -- became
``server/engine.py``'s and ``server/app.py``'s real backend-selection source
of truth) without merging the two: that step only needed *which* backend to
pick, not to change what ``LagunaBackend`` itself builds internally. The
merge remains open for whichever step next needs ``LagunaBackend`` to
consume ``ArchitectureSpec`` directly.

Why the layout is the way it is
-------------------------------
Both supported families keep their real architecture in different places, and
the difference is not cosmetic:

* Laguna's ``config.json`` is flat.
* Qwen3.6's nests everything under ``text_config``, with ``vision_config``
  alongside it. Reading Qwen3.6's ``num_hidden_layers`` at the top level
  yields nothing at all, silently.

Verified against four local Qwen3.6 checkpoints and one Laguna checkpoint on
2026-08-01; see ``TestAgainstRealCheckpoints`` for what was observed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Attention kinds that consume paged KV, keyed by the ``layer_types`` spelling.
#: ``csa_attention`` / ``hca_attention`` are DeepSeek-V4's compressed attention
#: layers (ratio-4 with learned indexer / ratio-128 all-compressed); they are
#: still length-proportional KV storage, hence the paged-KV family -- the
#: per-layer heterogeneity (window ring + compressed region + indexer region)
#: is a backend concern, not a cache-family concern.
PAGED_KV_ATTENTION = (
    "full_attention",
    "sliding_attention",
    "csa_attention",
    "hca_attention",
)
#: Attention kinds that carry length-independent recurrent state instead.
RECURRENT_ATTENTION = ("linear_attention",)

CACHE_PAGED_KV = "paged_kv"
CACHE_RECURRENT = "recurrent"


class UnsupportedArchitectureError(ValueError):
    """Raised before any weight is read, naming the field that disqualified it."""


@dataclass(frozen=True)
class LayerSpec:
    index: int
    attention: str
    mlp: str
    #: Which resource this layer needs from the slot manager. This is the
    #: field ``SlotResourceManager`` (step 7) is built around: a checkpoint
    #: mixing both is what makes two cache families necessary rather than
    #: hypothetical.
    cache: str


@dataclass(frozen=True)
class RopeSpec:
    rope_type: str
    theta: float
    partial_rotary_factor: float
    factor: float | None = None
    original_max_position_embeddings: int | None = None


@dataclass(frozen=True)
class QuantSpec:
    #: ``compressed-tensors`` or ``modelopt``. Read per checkpoint, never
    #: inferred from the architecture -- of four local Qwen3.6 NVFP4
    #: checkpoints, three are modelopt and unsloth's is compressed-tensors.
    method: str
    format: str | None
    kv_num_bits: int | None
    kv_type: str | None


@dataclass(frozen=True)
class MoESpec:
    num_experts: int
    top_k: int
    intermediate_size: int | None
    shared_expert_intermediate_size: int | None


@dataclass(frozen=True)
class ArchitectureSpec:
    """Everything needed to decide whether a checkpoint can be served, and how."""

    architecture: str
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    max_position_embeddings: int

    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    sliding_window: int | None
    attn_output_gate: bool

    layers: tuple[LayerSpec, ...]
    rope: dict[str, RopeSpec]
    quant: QuantSpec
    moe: MoESpec | None
    mtp_layers: int

    #: True when the checkpoint carries a vision tower. Detection and policy
    #: are kept apart on purpose: see :func:`validate_text_only`.
    has_vision_tower: bool
    #: The checkpoint's own claim, when it makes one (``language_model_only``).
    declares_language_model_only: bool | None

    @property
    def has_mtp(self) -> bool:
        return self.mtp_layers > 0

    @property
    def is_moe(self) -> bool:
        return self.moe is not None

    @property
    def paged_kv_layers(self) -> tuple[int, ...]:
        return tuple(layer.index for layer in self.layers if layer.cache == CACHE_PAGED_KV)

    @property
    def recurrent_layers(self) -> tuple[int, ...]:
        return tuple(layer.index for layer in self.layers if layer.cache == CACHE_RECURRENT)

    @property
    def needs_two_cache_families(self) -> bool:
        return bool(self.paged_kv_layers) and bool(self.recurrent_layers)

    def count_attention(self, kind: str) -> int:
        return sum(1 for layer in self.layers if layer.attention == kind)


def _text_section(config: dict[str, Any]) -> dict[str, Any]:
    """Qwen3.6 nests the language model under ``text_config``; Laguna does not."""
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def _parse_rope(text: dict[str, Any]) -> dict[str, RopeSpec]:
    """Return RoPE settings keyed by layer type.

    Laguna gives a ``rope_parameters`` mapping with a distinct entry per layer
    type (yarn for full attention, default for sliding). A model with one
    global setting is normalized to the single key ``"default"`` so callers do
    not branch on which shape the checkpoint used.
    """
    params = text.get("rope_parameters")
    if isinstance(params, dict) and params and all(isinstance(v, dict) for v in params.values()):
        return {
            layer_type: RopeSpec(
                rope_type=str(entry.get("rope_type", "default")),
                theta=float(entry.get("rope_theta", 10000.0)),
                partial_rotary_factor=float(entry.get("partial_rotary_factor", 1.0)),
                factor=entry.get("factor"),
                original_max_position_embeddings=entry.get("original_max_position_embeddings"),
            )
            for layer_type, entry in params.items()
        }

    scaling = text.get("rope_scaling") if isinstance(text.get("rope_scaling"), dict) else {}
    return {
        "default": RopeSpec(
            rope_type=str(scaling.get("rope_type", "default")),
            theta=float(text.get("rope_theta", 10000.0)),
            partial_rotary_factor=float(text.get("partial_rotary_factor", 1.0)),
            factor=scaling.get("factor"),
            original_max_position_embeddings=scaling.get("original_max_position_embeddings"),
        )
    }


def _parse_quant(config: dict[str, Any]) -> QuantSpec:
    quant = config.get("quantization_config")
    if not isinstance(quant, dict):
        return QuantSpec(method="none", format=None, kv_num_bits=None, kv_type=None)
    kv = quant.get("kv_cache_scheme")
    kv = kv if isinstance(kv, dict) else {}
    return QuantSpec(
        method=str(quant.get("quant_method", "unknown")),
        format=quant.get("format"),
        kv_num_bits=kv.get("num_bits"),
        kv_type=kv.get("type"),
    )


def _parse_moe(text: dict[str, Any]) -> MoESpec | None:
    num_experts = text.get("num_experts")
    top_k = text.get("num_experts_per_tok")
    if not num_experts or not top_k:
        return None
    return MoESpec(
        num_experts=int(num_experts),
        top_k=int(top_k),
        intermediate_size=text.get("moe_intermediate_size"),
        shared_expert_intermediate_size=text.get("shared_expert_intermediate_size"),
    )


def _cache_for(attention: str) -> str:
    if attention in PAGED_KV_ATTENTION:
        return CACHE_PAGED_KV
    if attention in RECURRENT_ATTENTION:
        return CACHE_RECURRENT
    raise UnsupportedArchitectureError(
        f"layer_types contains unknown attention kind {attention!r}; "
        f"known kinds are {[*PAGED_KV_ATTENTION, *RECURRENT_ATTENTION]}"
    )


def _parse_layers(text: dict[str, Any], num_layers: int) -> tuple[LayerSpec, ...]:
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or not layer_types:
        raise UnsupportedArchitectureError(
            "config.json has no 'layer_types' list; the per-layer attention "
            "sequence cannot be inferred, and guessing it is what produces "
            "NaN halfway through a run instead of an error at load time"
        )
    if len(layer_types) != num_layers:
        raise UnsupportedArchitectureError(
            f"layer_types has {len(layer_types)} entries but num_hidden_layers "
            f"is {num_layers}; the checkpoint contradicts itself"
        )

    mlp_types = text.get("mlp_layer_types")
    if not isinstance(mlp_types, list) or len(mlp_types) != num_layers:
        # Qwen3.6-27B is uniformly dense and says so only via the absence of
        # MoE fields, so fall back rather than rejecting.
        mlp_types = ["sparse" if text.get("num_experts") else "dense"] * num_layers

    return tuple(
        LayerSpec(
            index=i,
            attention=str(attention),
            mlp=str(mlp_types[i]),
            cache=_cache_for(str(attention)),
        )
        for i, attention in enumerate(layer_types)
    )


#: DSV4 compression ratios this runtime understands (llama.cpp's deepseek4
#: loader enforces the same set; the reference implementation defines them).
_DSV4_RATIOS = {0: "sliding_attention", 4: "csa_attention", 128: "hca_attention"}

#: GGUF ggml type names that are quantized payloads (everything else is a
#: plain tensor: norms, sinks, APE, embeddings-as-bf16, routing tables, ...).
_DSV4_QUANT_TYPES = ("iq2_xs", "q8_0")

# Qwen3.8's UD Q6_K_XL file is a standard GGML mixed K-quant file. The
# importance-mix is part of the checkpoint identity, just like DSV4's
# IQ2_XS/Q8_0 mix above; a different mix must not silently enter this loader.
_QWEN35_QUANT_TYPES = ("q4_k", "q5_k", "q6_k", "q8_0")


def parse_qwen35_gguf_architecture(
    kv: dict[str, Any],
    *,
    tensor_type_names: frozenset[str],
) -> ArchitectureSpec:
    """Build an :class:`ArchitectureSpec` for a text-only Qwen3.8 GGUF.

    ``qwen35.block_count`` includes the one NextN/MTP block. The target trunk
    is therefore ``block_count - nextn_predict_layers``; making that boundary
    explicit is important because the extra block has a different role even
    though it shares the same GGUF namespace.
    """

    block_count = int(kv.get("qwen35.block_count", 0))
    nextn_layers = int(kv.get("qwen35.nextn_predict_layers", 0))
    trunk_layers = block_count - nextn_layers
    if block_count <= 0 or nextn_layers <= 0 or trunk_layers <= 0:
        raise UnsupportedArchitectureError(
            "Qwen3.8 GGUF requires positive qwen35.block_count and "
            "qwen35.nextn_predict_layers with a non-empty target trunk"
        )
    interval = int(kv.get("qwen35.full_attention_interval", 0))
    if interval <= 0:
        raise UnsupportedArchitectureError(
            "Qwen3.8 GGUF is missing a positive qwen35.full_attention_interval"
        )

    layers = tuple(
        LayerSpec(
            index=index,
            attention="full_attention" if (index + 1) % interval == 0 else "linear_attention",
            mlp="dense",
            cache=(
                CACHE_PAGED_KV
                if (index + 1) % interval == 0
                else CACHE_RECURRENT
            ),
        )
        for index in range(trunk_layers)
    )
    quant_types = sorted(
        name.lower() for name in tensor_type_names if name.lower() in _QWEN35_QUANT_TYPES
    )
    unknown_types = sorted(
        name.lower()
        for name in tensor_type_names
        if name.lower() not in _QWEN35_QUANT_TYPES
        and not name.lower().startswith(("f32", "bf16", "f16", "i8", "i16", "i32", "i64"))
    )
    if unknown_types:
        raise UnsupportedArchitectureError(
            f"Qwen3.8 GGUF contains unsupported tensor types {unknown_types}; "
            f"supported quant types are {_QWEN35_QUANT_TYPES}"
        )
    vocab = kv.get("tokenizer.ggml.tokens")
    key_length = int(kv.get("qwen35.attention.key_length", 0))
    rope_dim = int(kv.get("qwen35.rope.dimension_count", 0))
    return ArchitectureSpec(
        architecture="Qwen3_5ForConditionalGeneration",
        model_type=str(kv.get("general.architecture", "")),
        vocab_size=len(vocab) if isinstance(vocab, list) else 0,
        hidden_size=int(kv.get("qwen35.embedding_length", 0)),
        num_hidden_layers=trunk_layers,
        max_position_embeddings=int(kv.get("qwen35.context_length", 0)),
        num_attention_heads=int(kv.get("qwen35.attention.head_count", 0)),
        num_key_value_heads=int(kv.get("qwen35.attention.head_count_kv", 0)),
        head_dim=key_length,
        sliding_window=None,
        attn_output_gate=True,
        layers=layers,
        rope={
            "default": RopeSpec(
                rope_type="default",
                theta=float(kv.get("qwen35.rope.freq_base", 10000.0)),
                partial_rotary_factor=rope_dim / key_length if key_length else 1.0,
            )
        },
        quant=QuantSpec(
            method="gguf",
            format="+".join(quant_types) or "none",
            kv_num_bits=None,
            kv_type=None,
        ),
        moe=None,
        mtp_layers=nextn_layers,
        has_vision_tower=False,
        declares_language_model_only=None,
    )


def parse_dsv4_gguf_architecture(
    kv: dict[str, Any],
    *,
    tensor_type_names: frozenset[str],
    mtp_layers: int = 0,
) -> ArchitectureSpec:
    """Build the ArchitectureSpec for a DeepSeek-V4 GGUF checkpoint.

    GGUF has no ``config.json``; its header KV pairs are the config. This is
    the GGUF sibling of :func:`parse_architecture` -- same contract: parse and
    reject before a single weight is read, name the offending field.

    ``tensor_type_names`` is the set of ggml type names present in the tensor
    index (e.g. ``{"IQ2_XS", "Q8_0", "F32", "BF16", "I32"}``); it drives the
    quant format gate the same way ``quantization_config`` does for
    safetensors -- different GGUF quant mixes of the same model are not
    interchangeable, so the mix is part of the identity.
    """
    block_count = int(kv.get("deepseek4.block_count", 0))
    if block_count <= 0:
        raise UnsupportedArchitectureError("GGUF metadata has no positive 'deepseek4.block_count'")
    ratios_raw = kv.get("deepseek4.attention.compress_ratios")
    if not isinstance(ratios_raw, list) or len(ratios_raw) < block_count:
        raise UnsupportedArchitectureError(
            "GGUF 'deepseek4.attention.compress_ratios' is missing or shorter "
            f"than block_count ({block_count}); the file may carry extra MTP "
            "entries, but it must at least cover every main layer"
        )
    layers: list[LayerSpec] = []
    for index in range(block_count):
        ratio = int(ratios_raw[index])
        attention = _DSV4_RATIOS.get(ratio)
        if attention is None:
            raise UnsupportedArchitectureError(
                f"layer {index} has compress_ratio {ratio}; supported ratios "
                f"are {sorted(_DSV4_RATIOS)}"
            )
        layers.append(
            LayerSpec(index=index, attention=attention, mlp="sparse", cache=CACHE_PAGED_KV)
        )

    quant_types = sorted(
        name.lower() for name in tensor_type_names if name.lower() in _DSV4_QUANT_TYPES
    )
    unknown_types = sorted(
        name.lower()
        for name in tensor_type_names
        if name.lower() not in _DSV4_QUANT_TYPES
        and not name.lower().startswith(("f32", "bf16", "f16", "i8", "i16", "i32", "i64"))
    )
    format_name = "+".join([*quant_types, *unknown_types]) or "none"

    vocab = kv.get("tokenizer.ggml.tokens")
    rope_theta = float(kv.get("deepseek4.rope.freq_base", 10000.0))
    scaling_type = str(kv.get("deepseek4.rope.scaling.type", "default"))
    factor = kv.get("deepseek4.rope.scaling.factor")
    original_ctx = kv.get("deepseek4.rope.scaling.original_context_length")
    return ArchitectureSpec(
        # The HF spelling (config.json architectures[0]) is what the registry
        # keys on; the GGUF arch name ("deepseek4") is recorded as model_type.
        architecture="DeepseekV4ForCausalLM",
        model_type=str(kv.get("general.architecture", "")),
        vocab_size=len(vocab) if isinstance(vocab, list) else 0,
        hidden_size=int(kv.get("deepseek4.embedding_length", 0)),
        num_hidden_layers=block_count,
        max_position_embeddings=int(kv.get("deepseek4.context_length", 0)),
        num_attention_heads=int(kv.get("deepseek4.attention.head_count", 0)),
        num_key_value_heads=int(kv.get("deepseek4.attention.head_count_kv", 0)),
        head_dim=int(kv.get("deepseek4.attention.key_length", 0)),
        sliding_window=kv.get("deepseek4.attention.sliding_window"),
        attn_output_gate=False,
        layers=tuple(layers),
        rope={
            "default": RopeSpec(
                rope_type=scaling_type,
                theta=rope_theta,
                partial_rotary_factor=1.0,
                factor=float(factor) if factor is not None else None,
                original_max_position_embeddings=(
                    int(original_ctx) if original_ctx is not None else None
                ),
            ),
            # Compressed-KV entries are re-rotated with their own theta
            # (verified against reference Compressor + GGUF KV).
            "compressed": RopeSpec(
                rope_type=scaling_type,
                theta=float(kv.get("deepseek4.attention.compress_rope_freq_base", 160000.0)),
                partial_rotary_factor=1.0,
                factor=float(factor) if factor is not None else None,
                original_max_position_embeddings=(
                    int(original_ctx) if original_ctx is not None else None
                ),
            ),
        },
        quant=QuantSpec(method="gguf", format=format_name, kv_num_bits=None, kv_type=None),
        moe=MoESpec(
            num_experts=int(kv.get("deepseek4.expert_count", 0)),
            top_k=int(kv.get("deepseek4.expert_used_count", 0)),
            intermediate_size=kv.get("deepseek4.expert_feed_forward_length"),
            shared_expert_intermediate_size=kv.get("deepseek4.expert_feed_forward_length"),
        ),
        mtp_layers=mtp_layers,
        has_vision_tower=False,
        declares_language_model_only=None,
    )


def parse_architecture(config: dict[str, Any]) -> ArchitectureSpec:
    """Parse a ``config.json`` dict. Raises on anything it cannot describe.

    Raising beats returning a half-filled spec: a missing field here becomes a
    wrong tensor shape thousands of tokens later, which is the failure mode
    this whole layer exists to move earlier.
    """
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise UnsupportedArchitectureError("config.json has no 'architectures' list")

    text = _text_section(config)
    num_layers = text.get("num_hidden_layers")
    if not isinstance(num_layers, int):
        where = "text_config" if text is not config else "top level"
        raise UnsupportedArchitectureError(
            f"'num_hidden_layers' missing or non-integer at {where} of config.json"
        )

    return ArchitectureSpec(
        architecture=str(architectures[0]),
        model_type=str(config.get("model_type", "")),
        vocab_size=int(text.get("vocab_size", 0)),
        hidden_size=int(text.get("hidden_size", 0)),
        num_hidden_layers=num_layers,
        max_position_embeddings=int(text.get("max_position_embeddings", 0)),
        num_attention_heads=int(text.get("num_attention_heads", 0)),
        num_key_value_heads=int(text.get("num_key_value_heads", 0)),
        head_dim=int(text.get("head_dim", 0)),
        sliding_window=text.get("sliding_window"),
        attn_output_gate=bool(text.get("attn_output_gate", False)),
        layers=_parse_layers(text, num_layers),
        rope=_parse_rope(text),
        quant=_parse_quant(config),
        moe=_parse_moe(text),
        mtp_layers=int(text.get("mtp_num_hidden_layers", 0) or 0),
        has_vision_tower=isinstance(config.get("vision_config"), dict),
        declares_language_model_only=config.get("language_model_only"),
    )


def validate_text_only(spec: ArchitectureSpec, *, language_model_only: bool) -> None:
    """Enforce RK8: refuse a vision-bearing checkpoint unless the loader that
    will read its weights runs in ``language_model_only`` mode.

    Detection lives in :func:`parse_architecture`; the policy lives here, so
    that changing the policy does not require changing the parser.

    The reason this needs its own check is that the architecture name does not
    carry the answer. All four local Qwen3.6 checkpoints declare
    ``Qwen3_5ForConditionalGeneration`` / ``qwen3_5``, yet three ship a vision
    tower and one does not -- so a registry keyed on architecture alone would
    happily accept a multimodal checkpoint and fail much later, on a tensor
    name it did not expect.

    **2026-08-02 semantics change (B0-1b,** ``docs/implementation-plan.md``
    **§4/C-2 and §7.1/B0-1):** this used to reject *any* checkpoint carrying
    ``vision_config``, full stop. That was written before D6 picked
    ``nvidia/Qwen3.6-27B-NVFP4`` as the Track B checkpoint -- which ships a
    real vision tower (333 tensors, verified against its own
    ``model.safetensors.index.json``; see
    ``notes/2026-08-02-qwen36-b0-fact-baseline.md`` §1.4) -- so the original
    rule would have refused the checkpoint the project committed to. The rule
    is now: a vision tower is fine *if* the caller can show the loader will
    not actually load it. ``language_model_only`` is that caller-side
    guarantee, not a config.json field -- it says "the loader that is about
    to read this checkpoint's weights will run
    ``runtime.loading.language_model_only.filter_language_model_only`` with
    ``language_model_only=True``" (B0-1a). This function cannot itself verify
    that promise (it is torch-free and runs before any weight is read, by
    design -- see module docstring); it only gates on the caller having made
    it. The real "zero vision tensors loaded" guarantee is enforced at
    weight-load time by the filter itself, structurally (a name that matches
    a vision prefix never reaches ``model.load_weights(...)`` at all when the
    filter runs with ``language_model_only=True`` -- see that module's
    docstring for the precise claim and what has/has not been verified
    against a real checkpoint).

    ``language_model_only: true`` in ``config.json`` (``spec.
    declares_language_model_only``) is a *different* thing from the
    ``language_model_only`` parameter here, and is treated as authoritative
    on its own when present: it is the checkpoint stating what it is, and on
    the one local checkpoint that sets it, it agrees with the absence of both
    ``vision_config`` and any ``visual.*`` tensor. The parameter is this
    runtime's own operating mode, decided by the caller regardless of what
    the checkpoint claims about itself -- necessarily so, since the official
    Qwen3.6 checkpoint's own ``config.json`` sets ``language_model_only:
    false`` (verified) while still being exactly the checkpoint this
    parameter is meant to let through.
    """
    if spec.declares_language_model_only is True:
        return
    if not spec.has_vision_tower:
        return
    if language_model_only:
        return
    raise UnsupportedArchitectureError(
        f"checkpoint declares a vision tower (config.json has 'vision_config', "
        f"and 'language_model_only' is {spec.declares_language_model_only!r}); "
        f"this runtime serves text only. Either use a text-only build of "
        f"{spec.architecture} -- one that sets 'language_model_only': true "
        f"and omits 'vision_config' -- or load with language_model_only=True "
        f"so the loader filters vision.* tensors instead of loading them."
    )
