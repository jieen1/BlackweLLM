"""Self-built replacement for vLLM's ``get_model()`` — Phase 1, extended
阶段7 (weight-loading + post-load processing self-built too; see
runtime/kernels/rope.py and runtime/model/laguna_decoder.py for the
Linear/Embedding/RoPE side of that same phase).

See notes/2026-07-27-vllm-complete-removal-implementation-plan.md.
The ONLY path for both the main model and the DFlash draft model since
阶段1/阶段3 respectively; the ``QSR_LAGUNA_MODEL_LOADER=vllm``/
``QSR_DFLASH_MODEL_LOADER=vllm`` escape hatches back to vLLM's
``get_model()``/``load_dflash_model()`` were removed entirely (任务#46)
once 任务#45's GPU validation confirmed this self-built path bit-exact.

Remaining vLLM dependency here: ``DefaultModelLoader.get_all_weights`` and
``process_weights_after_loading`` are BOTH now replaced
(``iterate_safetensors_checkpoint``/``apply_kv_cache_scale_post_load``,
阶段7, moved to ``runtime/loading/common.py`` at Track A step 6 -- see
that module's docstring for why they belong in the format-agnostic common
layer rather than here). The first attempt at the latter (calling only
``Attention.process_weights_after_loading(dtype)`` per-module) produced a
confirmed, severe regression on a real GPU e2e run (garbled text,
degenerate repetition, wrong accept rate) versus the established
bit-exact baseline, and was reverted rather than shipped half-understood.
Root-caused afterward (阶段7-补充, prompted by a coordinator challenge
to the "don't replace Attention" call this had led to -- see
notes/2026-07-27-vllm-complete-removal-implementation-plan.md's
"Attention op-dispatch ABC 结论修正"): the missing piece was never
``Attention.process_weights_after_loading`` itself (confirmed a no-op
for this model -- no attention sinks), but
``CompressedTensorsKVCacheMethod.process_weights_after_loading``
(vllm/model_executor/layers/quantization/compressed_tensors/
compressed_tensors.py:1121-1147), which is what actually copies the
loaded ``k_scale``/``v_scale`` checkpoint values into the ``_k_scale``/
``_v_scale`` buffers ``BFAttention`` reads at runtime.
``apply_kv_cache_scale_post_load`` replicates exactly that (see
its docstring and runtime/model/plain_attention.py's module docstring
for the full contract), and ``SelfBuiltAttentionPlaceholder``
(runtime/model/plain_attention.py) replaces ``Attention`` construction
itself in runtime/model/laguna_decoder.py. ``VllmConfig`` remains a
pure type-annotation usage in this file -- see the ``TYPE_CHECKING``
import below.

任务#41 (阶段8): ``set_default_torch_dtype`` is now self-built too
(``runtime.loading.common.default_torch_dtype``) -- vLLM's real version
(vllm/utils/torch_utils.py) is a 5-line ``@contextlib.contextmanager``
wrapping ``torch.set_default_dtype``/restore, nothing vLLM-specific
about it.

Track A step 6 (``docs/architecture.md`` §3.5.5): this module used to
define all of the above itself, plus the compressed-tensors-specific
scale-name knowledge that actually lived one file over in
``runtime/model/laguna_model.py``/``runtime/model/_weight_loading.py``.
It is now the orchestrator: format-agnostic pieces come from
``runtime.loading.common``, and it accepts a ``language_model_only``
flag (B0-1a/B0-1b) that it threads through to
``runtime.loading.language_model_only.filter_language_model_only``
before any weight reaches ``model.load_weights(...)``. Both real call
sites (``runtime/backends/laguna.py``, ``runtime/backends/laguna_dflash.py``)
keep the default (``False``): Laguna's checkpoints have no vision tower to
filter, so this is a real, wired parameter with no real trigger yet -- see
the ``language_model_only`` module's docstring for what that does and does
not prove.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from loader.gguf_header import read_gguf_header
from runtime.dflash2_config import load_dflash2_config, validate_dflash2_target
from runtime.dspark_config import load_and_validate_dspark_pair
from runtime.loading.common import (
    apply_kv_cache_scale_post_load,
    assert_all_params_loaded,
    default_torch_dtype,
    iterate_safetensors_checkpoint,
    record_checkpoint_tensor_names,
    warn_on_unconsumed_tensor_families,
)
from runtime.loading.gguf import PACKED_QUANT_TYPES, iterate_gguf_checkpoint
from runtime.loading.language_model_only import (
    LanguageModelOnlyStats,
    filter_language_model_only,
)
from runtime.model.laguna_dflash_model import LagunaDraftForCausalLMSelfBuilt
from runtime.model.laguna_model import LagunaForCausalLMSelfBuilt
from runtime.model.qwen36_dspark import Qwen36DSparkDraftForCausalLM
from runtime.model.qwen36_model import Qwen36ForCausalLMSelfBuilt
from runtime.model.qwen38_dflash2 import Qwen38DFlash2DraftForCausalLM


def load_laguna_model(
    runtime_config: Any,
    *,
    language_model_only: bool = False,
) -> LagunaForCausalLMSelfBuilt:
    """Construct + load a Laguna model instance without vLLM's ``get_model()``.

    Distributed initialization is performed by ``LagunaBackend`` before this
    call. The owned model receives its configuration explicitly; it does not
    read a global vLLM context.

    ``language_model_only`` (B0-1a/B0-1b, default False): when True, every
    checkpoint tensor whose name starts with a vision-tower prefix is
    dropped before ``model.load_weights(...)`` ever sees it -- see
    ``runtime.loading.language_model_only`` for the exact contract and why
    this has never been exercised against a real vision-bearing checkpoint.
    Laguna's real checkpoint has no such tensors, so the default (False)
    reproduces this function's pre-step-6 behavior exactly regardless of
    which value is passed; the flag exists so the loader is *capable* of
    the mode Track B's official Qwen3.6 checkpoint will need, not because
    Laguna needs it today.
    """
    model_config = runtime_config.model_config
    load_config = runtime_config.load_config
    device_config = runtime_config.device_config

    load_device = device_config.device if load_config.device is None else load_config.device
    target_device = torch.device(load_device)

    with default_torch_dtype(model_config.dtype), target_device:
        model = LagunaForCausalLMSelfBuilt(runtime_config=runtime_config)

        vision_filter_stats = LanguageModelOnlyStats()
        weights = filter_language_model_only(
            iterate_safetensors_checkpoint(model_config.model),
            language_model_only=language_model_only,
            stats=vision_filter_stats,
        )
        loaded_param_names = model.load_weights(weights)
        assert_all_params_loaded(model, loaded_param_names, context="load_laguna_model")

        apply_kv_cache_scale_post_load(model)

    return model.eval()


def load_laguna_dflash_draft_model(
    target_model: LagunaForCausalLMSelfBuilt,
    draft_runtime_config: Any,
    *,
    language_model_only: bool = False,
) -> LagunaDraftForCausalLMSelfBuilt:
    """Construct + load the DFlash draft model without vLLM's ``get_model()``
    / ``load_dflash_model()``. See runtime/model/laguna_dflash_model.py's
    module docstring for the two subtle wiring details this must get right
    (attention layer naming offset, Laguna-specific context-KV projection).

    Distributed initialization is already complete. The draft model receives
    its configuration explicitly and does not read a global vLLM context.

    ``language_model_only``: same contract as :func:`load_laguna_model`.
    The draft model's checkpoint has no ``quantization_config`` at all
    (see ``runtime/model/plain_attention.py``'s module docstring) and,
    like the main model, no vision-tower tensors -- so this is here for
    interface symmetry, with the same "never triggered by a real
    checkpoint yet" caveat.
    """
    draft_model_config = draft_runtime_config.speculative_config.draft_model_config
    load_config = draft_runtime_config.load_config
    device_config = draft_runtime_config.device_config

    load_device = device_config.device if load_config.device is None else load_config.device
    target_device = torch.device(load_device)

    with default_torch_dtype(draft_model_config.dtype), target_device:
        model = LagunaDraftForCausalLMSelfBuilt(runtime_config=draft_runtime_config)

        vision_filter_stats = LanguageModelOnlyStats()
        weights = filter_language_model_only(
            iterate_safetensors_checkpoint(draft_model_config.model),
            language_model_only=language_model_only,
            stats=vision_filter_stats,
        )
        loaded_param_names = model.load_weights(weights)
        assert_all_params_loaded(
            model,
            loaded_param_names,
            context="load_laguna_dflash_draft_model",
            expected_unloaded=frozenset({"model.embed_tokens.weight", "lm_head.weight"}),
        )

        apply_kv_cache_scale_post_load(model)

    # embed_tokens/lm_head are shared with the target model, not loaded from
    # this checkpoint (verified: it has no such keys at all). Matches
    # vLLM's `_should_share` (vllm/v1/worker/gpu/spec_decode/eagle/utils.py):
    # unconditional sharing, since has_own_embed_tokens/has_own_lm_head are
    # False -- this is an object-reference swap, not a weight copy.
    del model.model.embed_tokens
    model.model.embed_tokens = target_model.model.embed_tokens
    del model.lm_head
    model.lm_head = target_model.lm_head

    # One-time fused-buffer construction for context-KV precompute. Must
    # run after weight loading (and after the embed/lm_head swap, though
    # those aren't read by this step) -- see laguna_dflash_model.py.
    model.model._build_fused_kv_buffers()

    return model.eval()


def _build_qwen36_model_config(model_path: str) -> dict[str, Any]:
    """Merge ``config.json``'s ``text_config`` (Qwen3.6 nests the language
    model's own fields there -- ``runtime.architecture._text_section``
    already establishes this convention) with the top-level
    ``quantization_config`` (which is NOT nested under ``text_config`` --
    verified directly against the real checkpoint) into one flat dict,
    since :class:`runtime.model.qwen36_model.Qwen36ForCausalLMSelfBuilt`
    reads both from a single ``config`` argument.
    """
    raw = json.loads((Path(model_path) / "config.json").read_text())
    text_config = raw.get("text_config")
    merged = dict(text_config) if isinstance(text_config, dict) else dict(raw)
    merged["quantization_config"] = raw.get("quantization_config")
    # The self-built model consumes the flattened text config, but the
    # DFlash2 pair validator also needs the family identity after loading.
    # Preserve these top-level fields when Qwen3.5 nests the language config;
    # without them an NVFP4 target looked like an anonymous dict and could not
    # be paired with the otherwise format-agnostic DFlash2 draft.
    for key in ("architectures", "model_type"):
        if key in raw:
            merged[key] = raw[key]
    return merged


def _build_qwen38_gguf_model_config(model_path: str | Path) -> dict[str, Any]:
    """Build the flat Qwen3.8 target config from a ``qwen35`` GGUF header."""

    header = read_gguf_header(Path(model_path))
    kv = header.kv
    if kv.get("general.architecture") != "qwen35":
        raise ValueError(
            f"Qwen3.8 GGUF loader expects general.architecture='qwen35', got "
            f"{kv.get('general.architecture')!r}"
        )
    block_count = int(kv.get("qwen35.block_count", 0))
    if block_count != 65:
        raise ValueError(
            f"Qwen3.8 GGUF loader expects 65 blocks (64 trunk + 1 NextN), got {block_count}"
        )
    vocab = kv.get("tokenizer.ggml.tokens")
    vocab_size = len(vocab) if isinstance(vocab, list) else 0
    if vocab_size != int(kv.get("qwen35.vocab_size", vocab_size or 0)):
        raise ValueError("Qwen3.8 GGUF tokenizer vocabulary metadata is inconsistent")

    packed_types = {tensor.type_name for tensor in header.tensors} & PACKED_QUANT_TYPES
    format_name = "+".join(sorted(type_name.lower() for type_name in packed_types))
    if not format_name:
        raise ValueError("Qwen3.8 GGUF has no supported packed quantized tensors")

    quantized: dict[str, str] = {}
    embedding_type: str | None = None

    for tensor in header.tensors:
        type_name = tensor.type_name
        if tensor.name == "token_embd.weight":
            embedding_type = type_name
            continue
        if tensor.name == "output.weight":
            if type_name in PACKED_QUANT_TYPES:
                quantized["lm_head"] = f"gguf_{type_name.lower()}"
            continue
        pieces = tensor.name.split(".")
        if len(pieces) < 3 or pieces[0] != "blk":
            continue
        layer_idx = int(pieces[1])
        if layer_idx >= 64:
            continue
        suffix = ".".join(pieces[2:])
        root = f"model.language_model.layers.{layer_idx}"
        mapping = {
            "attn_q.weight": f"{root}.self_attn.q_proj",
            "attn_k.weight": f"{root}.self_attn.k_proj",
            "attn_v.weight": f"{root}.self_attn.v_proj",
            "attn_output.weight": f"{root}.self_attn.o_proj",
            "attn_qkv.weight": f"{root}.linear_attn.in_proj_qkv",
            "attn_gate.weight": f"{root}.linear_attn.in_proj_z",
            "ssm_alpha.weight": f"{root}.linear_attn.in_proj_a",
            "ssm_beta.weight": f"{root}.linear_attn.in_proj_b",
            "ssm_out.weight": f"{root}.linear_attn.out_proj",
            "ffn_gate.weight": f"{root}.mlp.gate_proj",
            "ffn_up.weight": f"{root}.mlp.up_proj",
            "ffn_down.weight": f"{root}.mlp.down_proj",
        }
        dotted_name = mapping.get(suffix)
        if dotted_name is not None and type_name in PACKED_QUANT_TYPES:
            quantized[dotted_name] = f"gguf_{type_name.lower()}"

    if embedding_type not in PACKED_QUANT_TYPES:
        raise ValueError(
            f"Qwen3.8 GGUF token_embd.weight must use a supported packed type, got "
            f"{embedding_type!r}"
        )

    layer_types = [
        "full_attention" if index % 4 == 3 else "linear_attention" for index in range(64)
    ]
    raw_rope_sections = kv.get("qwen35.rope.dimension_sections")
    if not isinstance(raw_rope_sections, list) or len(raw_rope_sections) != 4:
        raise ValueError(
            "Qwen3.8 GGUF must declare four qwen35.rope.dimension_sections values"
        )
    rope_sections = tuple(int(section) for section in raw_rope_sections)
    rotary_dim = int(kv["qwen35.rope.dimension_count"])
    if sum(rope_sections) * 2 != rotary_dim:
        raise ValueError(
            "Qwen3.8 GGUF MRoPE sections do not cover dimension_count: "
            f"sections={rope_sections}, dimension_count={rotary_dim}"
        )
    merged: dict[str, Any] = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "weight_format": "gguf",
        "gguf_path": str(model_path),
        "gguf_embedding_type": embedding_type,
        "gguf_quantized": quantized,
        "quantization_config": {"quant_method": "gguf", "format": format_name},
        "vocab_size": vocab_size,
        "hidden_size": int(kv["qwen35.embedding_length"]),
        "intermediate_size": int(kv["qwen35.feed_forward_length"]),
        "num_hidden_layers": 64,
        "num_attention_heads": int(kv["qwen35.attention.head_count"]),
        "num_key_value_heads": int(kv["qwen35.attention.head_count_kv"]),
        "head_dim": int(kv["qwen35.attention.key_length"]),
        "max_position_embeddings": int(kv["qwen35.context_length"]),
        "rms_norm_eps": float(kv["qwen35.attention.layer_norm_rms_epsilon"]),
        # Qwen3.8 GGUF stores ordinary RMSNorm scales (around 1.0).  The
        # legacy Qwen3.6 safetensors path uses the zero-centred Gemma-style
        # convention and therefore defaults to True in the model graph.
        "rms_norm_zero_centered": False,
        # qwen35 GGUF stores the fixed GDN scalars in their llama.cpp form:
        # F32 conv/norm/dt parameters and ssm_a = -exp(A_log).  Preserve that
        # representation through the model instead of round-tripping it
        # through a BF16 log parameter.
        "gguf_gdn_scalar_dtype": "float32",
        "gguf_ssm_a_is_negative": True,
        "hidden_act": "silu",
        "attention_bias": False,
        "attn_output_gate": True,
        "tie_word_embeddings": False,
        "layer_types": layer_types,
        "rope_parameters": {
            "rope_type": "interleaved",
            "rope_theta": float(kv["qwen35.rope.freq_base"]),
            "partial_rotary_factor": int(kv["qwen35.rope.dimension_count"])
            / int(kv["qwen35.attention.key_length"]),
            "mrope_section": rope_sections,
        },
        "full_attention_interval": int(kv["qwen35.full_attention_interval"]),
        "linear_num_key_heads": int(kv["qwen35.ssm.group_count"]),
        "linear_num_value_heads": int(kv["qwen35.ssm.time_step_rank"]),
        "linear_key_head_dim": int(kv["qwen35.ssm.state_size"]),
        "linear_value_head_dim": int(kv["qwen35.ssm.state_size"]),
        "linear_conv_kernel_dim": int(kv["qwen35.ssm.conv_kernel"]),
        "mtp_num_hidden_layers": int(kv.get("qwen35.nextn_predict_layers", 1)),
    }
    return merged


def load_qwen38_gguf_model(
    model_path: str | Path,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    max_seq_len: int = 4096,
    warmup_attention: bool = False,
) -> Qwen36ForCausalLMSelfBuilt:
    """Load the Qwen3.8 Q6_K_XL GGUF target in an isolated process.

    This entry point intentionally has no relationship to the existing
    NVFP4 service defaults. On SM120/BF16, packed Q4_K/Q5_K/Q6_K/Q8_0
    tensors dispatch to the native raw-pointer Q/K kernels and stay packed
    for the model lifetime. GGML's recurrent scalars remain F32, while the
    graph compute dtype defaults to the caller's ``dtype``.  The explicit
    ``QSR_GGUF_COMPUTE_DTYPE=bf16`` override is useful for the production
    SM120 fast path: it keeps the file-format scalar contract but avoids
    forcing every recurrent/attention intermediate through F32.  The
    ``QSR_GGUF_COMPUTE_DTYPE=float32`` setting restores the exact bring-up
    path. ``QSR_GGUF_NATIVE=0`` is the explicit torch dequantization oracle
    used for numerical diagnosis and CPU-compatible fallback tests.
    """

    torch.set_grad_enabled(False)
    model_config = _build_qwen38_gguf_model_config(model_path)
    compute_dtype = os.environ.get("QSR_GGUF_COMPUTE_DTYPE", "").strip().lower()
    if compute_dtype and compute_dtype not in {"float32", "bf16", "bfloat16"}:
        raise ValueError(
            "QSR_GGUF_COMPUTE_DTYPE must be float32 or bf16, "
            f"got {compute_dtype!r}"
        )
    if compute_dtype in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    elif compute_dtype == "float32":
        dtype = torch.float32
    model_config["gguf_compute_dtype"] = "float32" if dtype == torch.float32 else "bfloat16"
    target_device = torch.device(device)
    with default_torch_dtype(dtype), target_device:
        model = Qwen36ForCausalLMSelfBuilt(
            model_config,
            max_seq_len=max_seq_len,
            enable_mtp=False,
            enable_fp8_kv=False,
        )
        loaded_param_names = model.load_gguf_weights(
            iterate_gguf_checkpoint(Path(model_path), device=target_device)
        )
        assert_all_params_loaded(model, loaded_param_names, context="load_qwen38_gguf_model")
    model.eval()
    if warmup_attention and target_device.type == "cuda":
        model.warmup_attention_shapes(device=target_device, dtype=dtype)
        model.warmup_gdn_prefill_shapes(device=target_device, dtype=dtype)
    return model


#: FP8 KV cache gate (2026-08-03 follow-up). The standard checkpoint consumes
#: real per-layer ``k_scale``/``v_scale`` tensors; the ModelOpt W4A4
#: checkpoint has no such tensors and uses the explicit 1.0 fallback in
#: ``runtime/model/qwen36_model.py``. ``load_qwen36_model``'s own
#: ``enable_fp8_kv`` parameter reads this only when left at its own default
#: (``None``), same two-layer "explicit argument wins, env var is only the
#: outermost default" pattern ``runtime/checkpoints.py``'s env vars use.
QSR_QWEN36_FP8_KV_ENV = "QSR_QWEN36_FP8_KV"


def load_qwen36_model(
    model_path: str,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_seq_len: int = 4096,
    language_model_only: bool = True,
    warmup_attention: bool = True,
    enable_mtp: bool = False,
    enable_fp8_kv: bool | None = None,
    keep_fp8_raw_weights: bool = False,
) -> Qwen36ForCausalLMSelfBuilt:
    """Construct + load a Qwen3.6-27B (``Qwen3_5ForConditionalGeneration``,
    text-only) model instance. Track B / B1 -- see ``runtime/model/
    qwen36_model.py``'s module docstring for the exact scope this covers
    (eager, batch=1, BF16-dequantized quantized Linears; KV cache is BF16
    by default, FP8-e4m3 when ``enable_fp8_kv`` resolves True).

    Unlike :func:`load_laguna_model`, ``language_model_only`` defaults to
    **True** here, not False: this runtime's only real Qwen3.6 checkpoint
    (``nvidia/Qwen3.6-27B-NVFP4``, D6) ships a real vision tower this
    runtime never builds (B0-1a/b) -- there is no analogue of Laguna's
    "no vision tensors exist at all" case for this family, so silently
    defaulting to False here would mean the common call forgets to filter.
    Passing False is only for a hypothetical vision-tower-free Qwen3.6
    variant and should be a deliberate caller choice, not a silent default.

    ``max_seq_len`` sizes every full-attention layer's single-page KV
    cache (``Qwen36PagedAttentionCache``) upfront -- see that class's
    docstring for why this is a hard cap, not a growable buffer, at B1's
    batch=1/no-CUDA-graph scope.

    ``warmup_attention`` (default True, CUDA only) runs
    :meth:`Qwen36ForCausalLMSelfBuilt.warmup_attention_shapes` and
    :meth:`Qwen36ForCausalLMSelfBuilt.warmup_gdn_prefill_shapes` before
    returning, so SparkInfer/FlashInfer attention and FLA/Triton GDN chunk
    compilation are both paid here rather than inside whichever request
    arrives first. It adds nothing to steady-state cost and, once the kernel
    caches are warm, little to load time; pass False only when a caller
    genuinely wants to measure cold first-request compilation (as the
    prefill JIT probe scripts do).

    ``enable_mtp`` (default False, B3): construct and load
    :attr:`Qwen36ForCausalLMSelfBuilt.mtp`, the MTP draft head, from this
    same checkpoint's ``mtp.*`` tensors. False reproduces B1/B2's exact
    behavior (those tensors are counted in ``model.skipped_mtp_count`` and
    otherwise ignored) -- only B3 callers doing real speculative decoding
    need True.

    ``enable_fp8_kv`` (default ``None``, 2026-08-03 follow-up): FP8-e4m3
    KV cache for every backbone full-attention layer (never the MTP head,
    which remains BF16 unless its separate experimental switch is enabled).
    ``None`` resolves from ``QSR_QWEN36_FP8_KV`` (``"1"`` -> True, anything
    else including unset -> False) at call time, so a caller that never
    mentions this parameter gets the env-driven default; pass an explicit
    ``True``/``False`` to override the environment for one call. Standard
    checkpoints use their real ``self_attn.k_scale``/``v_scale`` tensors;
    ModelOpt W4A4 checkpoints use 1.0 when those tensors are absent, matching
    SGLang's ModelOpt KV-cache method.

    Disables autograd globally (``torch.set_grad_enabled(False)``), same
    as ``LagunaBackend.__init__`` (``runtime/backends/laguna.py:266``) --
    this is not optional cleanup: sparkinfer's paged-attention kernel
    exports its input tensors via ``__dlpack__``, which torch refuses for
    any tensor requiring grad (confirmed directly: omitting this raises
    ``BufferError: Can't export tensors that require gradient`` from
    inside ``sparkinfer/attention/paged/_forward.py``, not a hypothetical
    concern). Every ``nn.Parameter`` in this model graph defaults to
    ``requires_grad=True`` unless a class explicitly overrides it (only
    the quantized Linears in ``modelopt_linear.py`` do), so this is
    process-global, not per-parameter, on purpose -- matching Laguna's
    same choice, for the same reason (this runtime never trains).

    ``keep_fp8_raw_weights`` is an explicit opt-in for a weight-only FP8
    executor.  The normal B1 path creates a persistent BF16 cache and then
    releases the original FP8 weight storage; a native FP8 executor instead
    consumes those checkpoint values directly and must request that residency
    at its call site.  This is an argument rather than an environment flag
    because it changes the model's memory contract by roughly 10 GiB.
    """
    torch.set_grad_enabled(False)
    if enable_fp8_kv is None:
        # Default ON as of 2026-08-03. Measured on the standard checkpoint,
        # positive on all three axes that matter, so there is nothing left to
        # trade off:
        #   correctness  B1-R clears every bar by 2-8x, no capture-window
        #                overflow (notes/2026-08-03-fp8-kv-cache.md)
        #   memory       58.64 -> 46.63 GiB resident, KV 8192 -> 4096 MiB/slot
        #   speed        1.047x at a 100k-token prompt under CUDA Graph
        #                (25.058 -> 26.229 tok/s)
        # An earlier version of this comment claimed FP8 KV breaks
        # speculative decoding's token-identity guarantee, measured the same
        # day. The measurement was real; the attribution was wrong. MTP's
        # verify was running sparkinfer's mode="extend" instead of
        # mode="verify", and only the FP8-shaped plan diverged. With the
        # mode routed correctly, FP8 KV ON gives identical speculative and
        # non-speculative tokens at K=4, with acceptance 1.54/2.00 (better
        # than the 1.54/1.82 that turning FP8 KV off used to give). See
        # notes/2026-08-03-mtp-verify-mode.md; the retracted note is
        # notes/2026-08-03-fp8kv-breaks-speculative-token-identity.md.
        #
        # It also matches what this model shipped with historically
        # (qsr-hist's direct_model_runner.py defaulted kv_cache_dtype to
        # fp8_e4m3). `QSR_QWEN36_FP8_KV=0` opts back out.
        #
        # The speed figure is worth stating because it contradicts the
        # prediction made from an eager profile, where FP8 KV made
        # sparkinfer's PagedForwardKernel 19% slower (44.97 -> 53.64 ms) and
        # that was expected to surface as loss once decode became
        # kernel-bound under CUDA Graph. It did not; it became a small gain.
        # Trust the CG measurement, not the eager extrapolation.
        enable_fp8_kv = os.environ.get(QSR_QWEN36_FP8_KV_ENV, "1") != "0"
    model_config = _build_qwen36_model_config(model_path)
    target_device = torch.device(device)

    with default_torch_dtype(dtype), target_device:
        model = Qwen36ForCausalLMSelfBuilt(
            model_config,
            max_seq_len=max_seq_len,
            enable_mtp=enable_mtp,
            enable_fp8_kv=enable_fp8_kv,
        )

        vision_filter_stats = LanguageModelOnlyStats()
        weights = filter_language_model_only(
            iterate_safetensors_checkpoint(model_path),
            language_model_only=language_model_only,
            stats=vision_filter_stats,
        )
        # Recorded AFTER the vision filter: tensors it drops were dropped on
        # purpose and must not be reported as unconsumed.
        seen_checkpoint_names: set[str] = set()
        weights = record_checkpoint_tensor_names(weights, seen_checkpoint_names)
        loaded_param_names = model.load_weights(weights)
        assert_all_params_loaded(model, loaded_param_names, context="load_qwen36_model")
        warn_on_unconsumed_tensor_families(
            model, seen_checkpoint_names, context="load_qwen36_model"
        )

    model._vision_filter_stats = vision_filter_stats
    model.eval()
    if warmup_attention and target_device.type == "cuda":
        model.warmup_attention_shapes(device=target_device, dtype=dtype)
        model.warmup_gdn_prefill_shapes(device=target_device, dtype=dtype)
        # Only on CUDA, and only alongside warmup: this materializes every FP8
        # Linear's BF16 cache to free the originals, which is a real cost to
        # pay eagerly and pointless on a CPU-side construction (a test fixture,
        # a shape probe) that may never run a forward at all.
        if not keep_fp8_raw_weights:
            freed = model.free_fp8_raw_weights()
            if freed:
                print(f"freed raw FP8 weights on {freed} Linear(s) (BF16 cache retained)")
    return model


def load_qwen36_dspark_draft_model(
    target_model: Qwen36ForCausalLMSelfBuilt,
    *,
    target_model_path: str,
    draft_model_path: str,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen36DSparkDraftForCausalLM:
    """Load the separate Qwen3.8 DSpark draft and share target modules.

    The target config is validated against the draft config before the first
    draft parameter is allocated.  The draft checkpoint intentionally has no
    embedding or LM-head tensors; those two modules are attached by object
    reference after loading, exactly like the existing Laguna DFlash path.
    """

    draft_config = load_and_validate_dspark_pair(target_model_path, draft_model_path)
    target_layer_count = len(target_model.model.layers)
    target_device = torch.device(device)
    with default_torch_dtype(dtype), target_device:
        model = Qwen36DSparkDraftForCausalLM(
            draft_config,
            target_layer_count=target_layer_count,
        )
        loaded_param_names = model.load_weights(iterate_safetensors_checkpoint(draft_model_path))
        assert_all_params_loaded(
            model,
            loaded_param_names,
            context="load_qwen36_dspark_draft_model",
            expected_unloaded=frozenset({"embed_tokens.weight", "lm_head.weight"}),
        )
    model.attach_shared_modules(
        embed_tokens=target_model.model.embed_tokens,
        lm_head=target_model.lm_head,
    )
    return model.eval()


def load_qwen38_dflash2_draft_model(
    target_model: Qwen36ForCausalLMSelfBuilt,
    *,
    draft_model_path: str | Path,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen38DFlash2DraftForCausalLM:
    """Load the DFlash2 draft and attach the Qwen3.8 target's shared heads.

    The target model is already constructed by either the GGUF or compressed-
    tensors Qwen3.8 loader; its flattened config is the source of truth for
    the pair check.  This function only reads the separate DFlash2 safetensors
    file and never changes service defaults or a running backend.
    """

    draft_config = load_dflash2_config(draft_model_path)
    validate_dflash2_target(target_model.config, draft_config)
    target_layer_count = len(target_model.model.layers)
    target_device = torch.device(device)
    with default_torch_dtype(dtype), target_device:
        model = Qwen38DFlash2DraftForCausalLM(
            draft_config,
            target_layer_count=target_layer_count,
        )
        loaded_param_names = model.load_weights(
            iterate_safetensors_checkpoint(str(draft_model_path))
        )
        assert_all_params_loaded(
            model,
            loaded_param_names,
            context="load_qwen38_dflash2_draft_model",
            expected_unloaded=frozenset({"embed_tokens.weight", "lm_head.weight"}),
        )
    model.attach_shared_modules(
        embed_tokens=target_model.model.embed_tokens,
        lm_head=target_model.lm_head,
    )
    return model.eval()
