"""Self-built, Laguna-specific replacement for constructing vLLM's
``VllmConfig`` via ``EngineArgs.create_engine_config()`` -- 任务#45 (vLLM
removal plan). Mirrors ``VllmConfig``'s nested attribute SHAPE (so every
existing self-built call site -- ``load_laguna_model``,
``SelfBuiltAttentionPlaceholder``, ``LagunaBackend.__init__``,
``laguna_dflash.py`` -- keeps working with zero changes, since they all
already treat ``vllm_config`` as a duck-typed object, never
``isinstance``-check it) but populates it directly from the checkpoint's
own ``config.json`` and this runtime's fixed TP=1/PP=1/no-EPLB/sparkinfer
deployment shape, instead of vLLM's general-purpose CLI-args-to-config
resolution pipeline.

任务#44 evaluated this as "month-scale, don't touch" first, then reversed
that after a coordinator challenge (nano-vllm cross-check, same lesson as
阶段7-补充's Attention ABC reversal): real field usage across this
runtime is ~15 distinct fields, mostly trivial ``hf_config`` wraps or
constants for our fixed deployment shape, comparable to nano-vllm's own
11-field flat ``Config`` dataclass for the same "one model, one
deployment shape" scope -- not vLLM/sglang's "support any architecture"
scope, which is where their real config-system size comes from.

Every field here was verified against a REAL constructed vLLM
``VllmConfig`` (via ``EngineArgs(...).create_engine_config()`` for
Laguna's real checkpoint) before being hardcoded, not guessed from
vLLM's field names -- see notes/2026-07-27-vllm-complete-removal-
implementation-plan.md's 任务#44/#45 sections for the live-probe values
this was checked against.

**任务#46**: the ``QSR_LAGUNA_MODEL_LOADER=vllm``/``QSR_DFLASH_MODEL_
LOADER=vllm`` escape hatches (real ``vllm.config.VllmConfig`` via
``EngineArgs.create_engine_config()`` + real ``get_model()``/
``load_dflash_model()``) that used to run alongside this module have
been removed entirely from ``runtime/backends/laguna.py``/
``laguna_dflash.py`` -- this ``SelfBuiltVllmConfig`` is now the only
config Laguna's production path ever constructs. They served their
purpose as a reference implementation during 任务#45's own bit-exact
validation (both paths confirmed correct against real weights); kept
alive past that point they were accumulating real, unfixed regressions
(a TP/PP GroupCoordinator gap, and a deeper lm_head/quant_method tying
incompatibility between vLLM's load_dflash_model() and this runtime's
self-built PlainLMHead) in code nobody was going to keep maintaining.
Real ``EngineArgs``/``get_model()`` etc. remain available from
``runtime/compat_vllm.py`` for the separate qwen3.6/DirectModelRunner
tenant (out of scope, 阶段0) and for one-off benchmark/diagnostic
scripts -- only Laguna's own production loader selection was removed.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

# ---------------------------------------------------------------------------
# LagunaConfig is loaded from the checkpoint's custom code via transformers
# a real vLLM import deliberately, NOT replaced with a from-scratch
# transformers.AutoConfig.from_pretrained() call: it is a tiny (~120-line),
# fully standalone `transformers.PretrainedConfig` subclass with zero further
# vLLM coupling (no VllmConfig/EngineArgs/model_executor imports at all) --
# reusing it verbatim guarantees byte-identical field defaults to what every
# GPU bit-exact validation this whole session has run against, versus a
# real (if narrow) risk of silent default-value drift from re-deriving the
# same class from scratch. This is the ONE vLLM import this module keeps,
# and it is a completely different order of dependency than
# vllm.config/vllm.engine.arg_utils/vllm.model_executor.model_loader.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _EplbConfigShim:
    communicator: str = "torch_nccl"


@dataclasses.dataclass
class SelfBuiltParallelConfig:
    """TP=1/PP=1/no-EPLB always -- see module docstring. Fields match what
    the real ``init_worker_distributed_environment`` (vLLM's function --
    kept, unchanged, for the escape hatches used by other tenants/
    diagnostics, but no longer called from laguna.py/laguna_dflash.py
    since 任务#46) reads off its ``parallel_config`` argument AND off
    ``get_current_vllm_config_or_none()`` internally (``init_distributed_
    environment`` reads the
    GLOBAL current-vllm-config, set by ``set_current_vllm_config`` right
    before this runs -- not just its own explicit args; verified by
    reading vllm/distributed/parallel_state.py's real source, not
    guessed) -- fields below are the union of both, all trivial TP=1/
    PP=1/single-node constants for this runtime's fixed deployment shape.
    """

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    data_parallel_size: int = 1
    world_size: int = 1
    distributed_timeout_seconds: int | None = None
    disable_custom_all_reduce: bool = True
    enable_eplb: bool = False
    eplb_config: _EplbConfigShim = dataclasses.field(default_factory=_EplbConfigShim)
    enable_elastic_ep: bool = False
    distributed_executor_backend: str | None = "uni"
    nnodes: int = 1
    nnodes_within_dp: int = 1
    max_parallel_loading_workers: int | None = None
    ray_workers_use_nsight: bool = False
    placement_group: Any = None


@dataclasses.dataclass
class SelfBuiltCacheConfig:
    """``cache_dtype`` starts "auto" and is mutated to "fp8" by
    ``SelfBuiltAttentionPlaceholder`` itself (runtime/model/
    plain_attention.py) -- matches the real vLLM ``Attention.__init__``
    side effect this replaces (verified live: real ``cache_config.
    cache_dtype`` is "auto" straight out of ``EngineArgs.
    create_engine_config()`` too, never resolved to "fp8" at this stage).
    """

    cache_dtype: str = "auto"
    calculate_kv_scales: bool = False


@dataclasses.dataclass
class SelfBuiltQuantConfig:
    """Only ``.kv_cache_scheme`` is ever read (runtime/model/
    plain_attention.py) -- read directly from the checkpoint's
    ``config.json`` ``quantization_config.kv_cache_scheme`` field, same
    checkpoint-direct-read pattern already used throughout 阶段6/7/8 for
    other NVFP4/MoE values, not vLLM's ``CompressedTensorsConfig``
    parsing machinery (config_groups/ignore lists/etc -- none of which
    this runtime reads through ``vllm_config.quant_config`` at all).
    """

    kv_cache_scheme: dict[str, Any] | None = None


@dataclasses.dataclass
class SelfBuiltCompilationConfig:
    static_forward_context: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class SelfBuiltLoadConfig:
    device: str | None = None


@dataclasses.dataclass
class SelfBuiltDeviceConfig:
    device: str = "cuda"


class _NoOpIrOpPriority:
    """``vllm_config.kernel_config.ir_op_priority.set_default()`` is
    called unconditionally in laguna.py. A no-op here: ``TritonRMSNorm``
    (this runtime's actual RMSNorm, used throughout the self-built model
    graph) never dispatches through vLLM's IR-op registry at all -- only
    real vLLM ``RMSNorm`` CustomOp instances (constructed by the real
    ``get_model()``, which Laguna's own loader no longer calls since
    任务#46) read it.
    """

    def set_default(self) -> None:
        pass


@dataclasses.dataclass
class SelfBuiltKernelConfig:
    ir_op_priority: _NoOpIrOpPriority = dataclasses.field(default_factory=_NoOpIrOpPriority)
    moe_backend: str | None = None


@dataclasses.dataclass
class SelfBuiltModelConfig:
    """Field list is the union of what THIS runtime's own code reads
    (``hf_config``/``dtype``/``model``/``get_hidden_size()``/
    ``get_vocab_size()``/``get_num_layers()`` -- verified by grepping
    every ``vllm_config.model_config.*`` access across runtime/ and
    server/) AND what the real vLLM ``SpeculativeConfig.__post_init__``
    reads off ``target_model_config`` when constructing (and then
    discarding, per 阶段6's finding) its own auto-derived draft
    ``ModelConfig`` (runtime/backends/laguna_dflash.py passes this object
    as ``target_model_config`` -- ``SpeculativeConfig``'s Pydantic fields
    for it use ``Annotated[ModelConfig, SkipValidation()]``, confirmed by
    inspecting the real dataclass fields, so it never isinstance-checks
    this against the real class, only does plain attribute access at
    runtime -- verified directly against vllm/config/speculative.py's
    real source for every ``self.target_model_config.X`` it touches, not
    guessed).
    """

    hf_config: object  # LagunaConfig from checkpoint
    dtype: torch.dtype
    model: str
    max_model_len: int
    gpu_memory_utilization: float
    trust_remote_code: bool
    quantization: str | None
    tokenizer: str
    tokenizer_mode: str = "auto"
    tokenizer_revision: str | None = None
    seed: int = 0
    enforce_eager: bool = False
    max_logprobs: int = 20
    allowed_local_media_path: str = ""
    allowed_media_domains: list[str] | None = None
    config_format: str = "auto"
    hf_overrides: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def hf_text_config(self) -> object:  # LagunaConfig
        # Real vLLM's get_hf_text_config() only differs from hf_config
        # itself for nested-text-config multimodal architectures (e.g.
        # a separate `.text_config`) -- Laguna is not one (verified: no
        # such attribute on the real checkpoint's config.json), so this
        # is an alias, matching the real, live-probed value
        # (hf_text_config.model_type == "laguna" == hf_config.model_type).
        return self.hf_config

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_vocab_size(self) -> int:
        return self.hf_config.vocab_size

    def get_num_layers(self, parallel_config: SelfBuiltParallelConfig) -> int:
        # Real vLLM's version does PP start/end index math
        # (get_pp_indices); PP=1 always here collapses that to the whole
        # range trivially -- verified by reading vllm/config/model.py's
        # real get_num_layers()/get_layers_start_end_indices() before
        # simplifying, not assumed.
        del parallel_config
        return self.hf_config.num_hidden_layers


@dataclasses.dataclass
class LagunaDraftModelConfig:
    """Configuration surface consumed by the self-built DFlash model.

    This deliberately models only the draft checkpoint fields the local
    loader reads.  It is not a replacement for vLLM's general ModelConfig.
    """

    model: str
    hf_config: object
    tokenizer: str
    tokenizer_mode: str
    trust_remote_code: bool
    dtype: torch.dtype
    seed: int
    max_model_len: int
    spec_target_max_model_len: int
    enforce_eager: bool = True
    runner: str = "draft"


@dataclasses.dataclass
class LagunaDFlashConfig:
    """Owned DFlash settings required by ``LagunaDraftForCausalLMSelfBuilt``."""

    model: str
    method: str
    num_speculative_tokens: int
    target_model_config: SelfBuiltModelConfig
    target_parallel_config: SelfBuiltParallelConfig
    draft_model_config: LagunaDraftModelConfig


@dataclasses.dataclass
class SelfBuiltVllmConfig:
    """Duck-typed stand-in for ``vllm.config.VllmConfig``, used only for
    the ``selfbuilt`` (default) loader path -- see module docstring.
    """

    model_config: SelfBuiltModelConfig
    cache_config: SelfBuiltCacheConfig
    quant_config: SelfBuiltQuantConfig
    parallel_config: SelfBuiltParallelConfig
    compilation_config: SelfBuiltCompilationConfig
    load_config: SelfBuiltLoadConfig
    device_config: SelfBuiltDeviceConfig
    kernel_config: SelfBuiltKernelConfig
    speculative_config: Any = None
    # Required by the former worker-initialization contract; always unused.
    ec_transfer_config: Any = None


def build_laguna_config(
    model: str,
    *,
    dtype: str = "bfloat16",
    max_model_len: int,
    gpu_memory_utilization: float = 0.9,
    trust_remote_code: bool = False,
    enforce_eager: bool = False,
) -> SelfBuiltVllmConfig:
    """Self-built replacement for ``EngineArgs(model=..., ...).
    create_engine_config()`` -- only the kwargs this runtime's real call
    sites actually pass (server/engine.py's ``_load_laguna_model``,
    benchmarks/_phase5_e2e_bitexact_validate.py; verified by grepping
    every real ``EngineArgs(...)`` call site in the production/
    validation paths, not vLLM's full CLI surface).
    """
    model_dir = Path(model)
    hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)

    checkpoint_config = json.loads((model_dir / "config.json").read_text())
    quantization_config = checkpoint_config.get("quantization_config")
    quantization = quantization_config.get("quant_method") if quantization_config else None
    kv_cache_scheme = quantization_config.get("kv_cache_scheme") if quantization_config else None

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        dtype
    ]

    model_config = SelfBuiltModelConfig(
        hf_config=hf_config,
        dtype=torch_dtype,
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
        quantization=quantization,
        tokenizer=model,
        enforce_eager=enforce_eager,
    )
    return SelfBuiltVllmConfig(
        model_config=model_config,
        cache_config=SelfBuiltCacheConfig(),
        quant_config=SelfBuiltQuantConfig(kv_cache_scheme=kv_cache_scheme),
        parallel_config=SelfBuiltParallelConfig(),
        compilation_config=SelfBuiltCompilationConfig(),
        load_config=SelfBuiltLoadConfig(),
        device_config=SelfBuiltDeviceConfig(),
        kernel_config=SelfBuiltKernelConfig(),
    )


def build_laguna_dflash_config(
    runtime_config: SelfBuiltVllmConfig,
    *,
    model: str,
    hf_config: object,
    num_speculative_tokens: int,
    max_model_len: int,
) -> SelfBuiltVllmConfig:
    """Return a DFlash-specific copy of a Laguna runtime configuration.

    The draft loader only consumes its checkpoint identity/configuration and
    the target model's fixed single-GPU geometry.  Keeping this narrow avoids
    importing vLLM's broad speculative configuration machinery at startup.
    """
    target = runtime_config.model_config
    draft_model_config = LagunaDraftModelConfig(
        model=model,
        hf_config=hf_config,
        tokenizer=target.tokenizer,
        tokenizer_mode=target.tokenizer_mode,
        trust_remote_code=target.trust_remote_code,
        dtype=target.dtype,
        seed=target.seed,
        max_model_len=max_model_len,
        spec_target_max_model_len=target.max_model_len,
    )
    speculative_config = LagunaDFlashConfig(
        model=model,
        method="dflash",
        num_speculative_tokens=num_speculative_tokens,
        target_model_config=target,
        target_parallel_config=runtime_config.parallel_config,
        draft_model_config=draft_model_config,
    )
    return dataclasses.replace(runtime_config, speculative_config=speculative_config)


def load_laguna_draft_hf_config(model: str) -> object:
    """Load and normalize the DFlash checkpoint configuration locally.

    The draft checkpoint deliberately has no ``auto_map`` entry.  Its
    generic Transformers config preserves the draft's sliding-attention RoPE
    settings, while vLLM historically supplied a few absent Laguna defaults.
    Keep that exact split: replacing the config class with the target's
    custom class changes the draft RoPE default and breaks acceptance.
    """
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    defaults = {
        "qkv_bias": False,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [0],
        "norm_topk_prob": True,
        "partial_rotary_factor": 1.0,
        "swa_attention_sink_enabled": False,
        "swa_rope_parameters": None,
    }
    for name, value in defaults.items():
        if not hasattr(config, name):
            setattr(config, name, value)
    # The generic config materializes a uniform list; the legacy DFlash
    # config intentionally normalized that representation to ``None``.
    config.num_attention_heads_per_layer = None
    return config


def init_laguna_distributed_environment(
    rank: int, distributed_init_method: str, local_rank: int = 0
) -> None:
    """Self-built replacement for vLLM's real ``init_worker_distributed_
    environment`` -- this is the only distributed-init path
    ``runtime/backends/laguna.py``'s ``__init__`` calls since 任务#46
    removed the ``QSR_LAGUNA_MODEL_LOADER=vllm`` escape hatch that used
    to call the real vLLM function instead.

    Real ``init_worker_distributed_environment`` (vllm/v1/worker/
    gpu_worker.py) does five things: ``init_batch_invariance()`` (no-op
    unless ``VLLM_BATCH_INVARIANT`` env var is set -- verified against
    its real source, never set anywhere in this runtime);
    ``override_envs_for_eplb(...)`` (no-op: this runtime has no EPLB,
    verified 阶段6); ``set_custom_all_reduce(...)`` (irrelevant at TP=1 --
    custom all-reduce only matters for multi-GPU tensor parallelism);
    ``init_distributed_environment(...)`` + ``ensure_model_parallel_
    initialized(...)`` (the actual ``torch.distributed``/vLLM
    ``GroupCoordinator`` setup); ``ensure_ec_transfer_initialized(...)``
    (no-op: ``vllm_config.ec_transfer_config`` is always ``None`` here,
    no encoder-decoder/disaggregation).

    Attempted first to just feed the real ``init_worker_distributed_
    environment`` a self-built ``parallel_config`` duck-type (matching
    the rest of this module's approach) -- abandoned after it kept
    needing more fields several GroupCoordinator-construction-internals
    layers deep (``cpu_distributed_timeout_seconds`` etc, read via
    vLLM's OWN global ``get_current_vllm_config_or_none()``, not even
    this function's own explicit args) with no clear bound on how many
    more there might be. Confirmed via grep (阶段8/任务#44) that nothing
    in this runtime's own code reads vLLM's ``GroupCoordinator``/``get_
    tp_group()``/``get_pp_group()`` state at all -- so replicating just
    the one thing that's actually needed (a working ``torch.distributed``
    process group at world_size=1) is both simpler and lower-risk than
    continuing to chase vLLM-internal attribute requirements with no
    established bound, same "one real kernel/one real deployment shape"
    simplification nano-vllm's own distributed init makes (6 raw
    ``torch.distributed`` calls, no ``GroupCoordinator``-equivalent
    layer at all -- verified against nanovllm/engine/model_runner.py's
    real source, 任务#44).
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=distributed_init_method,
            world_size=1,
            rank=rank,
        )
    torch.cuda.set_device(local_rank if local_rank >= 0 else 0)
