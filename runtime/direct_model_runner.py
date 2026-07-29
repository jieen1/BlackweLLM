"""Direct (non-HTTP) model runner: this process owns GPU KV/GDN state itself
and drives ``model.forward()`` directly, replacing an earlier prototype's
HTTP bridge to a separate vLLM server (commit ``b28942c``).

Design and the four reused vLLM primitives this depends on (``EngineArgs
.create_engine_config()``, ``get_model()``, ``bind_kv_cache()``,
``set_forward_context()``) are documented in
``notes/direct-model-runner-design.md`` -- read that first.

Scope this round (see the design doc's "explicitly out of scope" section):
only slot 0 is exercised, no CUDA graph, no real multi-request batching, no
MTP. Metadata is hand-built for exactly one request at a time, not through
the production ``SM120GQAMetadataBuilder``/``GDNAttentionMetadataBuilder``
(those handle concerns -- persistent CUDA-graph-safe buffers, spec-decode,
multi-request batching -- this round's scope does not need).
"""

from __future__ import annotations

import torch

from runtime.compat_vllm import (
    EngineArgs,
    VllmConfig,
    bind_kv_cache,
    get_distributed_init_method,
    get_model,
    get_open_port,
    init_worker_distributed_environment,
    set_current_vllm_config,
    set_forward_context,
)
from runtime.compat_vllm_qwen36 import (
    AttentionBackendEnum,
    GDNAttentionMetadata,
    SM120GQAMetadata,
    register_backend,
)
from runtime.sampling import SamplingParams, make_generator, sample_from_logits

NUM_SLOTS = 4
_SM120_BACKEND_PATH = "vllm.v1.attention.backends.sm120_gqa.SM120GQABackend"

# Physical index 0 (block index / GDN state index) is never used for real
# request data -- confirmed empirically from a real vLLM SchedulerOutput
# dump (block_ids=([1], [2], [3], [4]) for the first-ever scheduled
# request; see notes/direct-model-runner-design.md's "Stage C field diff"
# section). Root cause of the 100%-deterministic wrong output this round:
# our hand-built metadata hardcoded physical index = logical slot (so slot
# 0 -> physical index 0), which real vLLM's convention never produces --
from runtime.block_pool import (  # noqa: E402
    RESERVED_PHYSICAL_SLOTS,
    BlockHash,
    BlockPool,
    ChunkedPrefillState,
    _physical_slot,
)


def _ensure_sm120_backend_registered() -> None:
    """register_backend() is a plain dict write (see registry.py's
    _ATTN_OVERRIDES) -- safe to call more than once."""
    register_backend(AttentionBackendEnum.CUSTOM, _SM120_BACKEND_PATH)


def profile_kv_cache_blocks(
    static_forward_context: dict,
    vllm_config: VllmConfig,
    device: torch.device,
    *,
    num_slots: int,
    block_size: int,
    gpu_memory_utilization: float = 0.85,
    num_speculative_tokens: int = 0,
) -> int:
    """Profile available GPU memory and return the maximum number of KV
    cache blocks that fit, following vLLM's memory-profiling approach.

    After model loading, measures free GPU memory and calculates how many
    attention KV blocks fit within the ``gpu_memory_utilization`` budget.
    GDN state (conv + ssm) is allocated per-physical-slot (not per-block),
    so its cost is subtracted separately.

    Returns the total number of attention KV blocks (including reserved).
    """
    free_mem, total_mem = torch.cuda.mem_get_info(device)
    budget = int(total_mem * gpu_memory_utilization)
    already_used = total_mem - free_mem
    reserved_for_kv = budget - already_used
    # Reserve 15% of the KV budget for forward-pass activations, temporary
    # tensors, and CUDA allocator overhead. Without this margin the first
    # forward pass OOMs because all GPU memory is consumed by KV cache.
    activation_margin = int(reserved_for_kv * 0.30) + 5 * 2**30
    reserved_for_kv -= activation_margin
    if reserved_for_kv <= 0:
        raise RuntimeError(
            f"GPU memory budget exhausted after model load: "
            f"used={already_used / 2**30:.1f} GiB, budget={budget / 2**30:.1f} GiB, "
            f"free={free_mem / 2**30:.1f} GiB (utilization={gpu_memory_utilization})"
        )

    attn_layer_names = []
    gdn_layer_names = []
    for name, layer in static_forward_context.items():
        if hasattr(layer, "get_state_shape"):
            gdn_layer_names.append(name)
        else:
            attn_layer_names.append(name)

    gdn_bytes = 0
    total_physical_slots = num_slots + RESERVED_PHYSICAL_SLOTS
    if gdn_layer_names:
        layer = static_forward_context[gdn_layer_names[0]]
        conv_shape, ssm_shape = layer.get_state_shape()
        conv_dtype, ssm_dtype = layer.get_state_dtype()
        conv_elem = torch.tensor([], dtype=conv_dtype).element_size()
        ssm_elem = torch.tensor([], dtype=ssm_dtype).element_size()
        conv_size = 1
        for d in conv_shape:
            conv_size *= d
        ssm_size = 1
        for d in ssm_shape:
            ssm_size *= d
        ssm_rows_per_slot = 1 + num_speculative_tokens
        gdn_bytes = len(gdn_layer_names) * (
            total_physical_slots * conv_size * conv_elem
            + total_physical_slots * ssm_rows_per_slot * ssm_size * ssm_elem
        )

    attn_budget = reserved_for_kv - gdn_bytes
    if attn_budget <= 0:
        raise RuntimeError(
            f"GDN state alone ({gdn_bytes / 2**30:.1f} GiB) exceeds KV budget "
            f"({reserved_for_kv / 2**30:.1f} GiB)"
        )

    if not attn_layer_names:
        raise RuntimeError("no attention layers found for KV cache profiling")

    any_attn = static_forward_context[attn_layer_names[0]]
    backend_cls = any_attn.get_attn_backend()
    num_kv_heads = any_attn.num_kv_heads
    head_size = any_attn.head_size
    cache_dtype_str = vllm_config.cache_config.cache_dtype
    shape = backend_cls.get_kv_cache_shape(1, block_size, num_kv_heads, head_size, cache_dtype_str)
    torch_dtype = any_attn.kv_cache_torch_dtype
    elem_size = torch.tensor([], dtype=torch_dtype).element_size()
    per_block_elems = 1
    for d in shape:
        per_block_elems *= d
    per_block_bytes = per_block_elems * elem_size * len(attn_layer_names)

    num_blocks = max(1, attn_budget // per_block_bytes)
    return num_blocks


def allocate_fixed_slot_kv_caches(
    static_forward_context: dict,
    vllm_config: VllmConfig,
    device: torch.device,
    *,
    num_slots: int,
    block_size: int,
    blocks_per_slot: int,
    num_speculative_tokens: int = 0,
    num_blocks_override: int | None = None,
) -> dict[str, object]:
    """Allocate our own num_slots-fixed-slot KV (attention) and state (GDN)
    tensors and bind them via vLLM's own real ``bind_kv_cache()`` -- this is
    the only thing that differs from vLLM's own tensor allocation, everything
    else stays real. Returns the same ``dict[str, tensor|tuple]``
    bind_kv_cache expects, keyed by layer name.
    """
    attn_layer_names = []
    gdn_layer_names = []
    for name, layer in static_forward_context.items():
        if hasattr(layer, "get_state_shape"):
            gdn_layer_names.append(name)
        else:
            attn_layer_names.append(name)

    kv_caches: dict[str, object] = {}

    if attn_layer_names:
        any_attn = static_forward_context[attn_layer_names[0]]
        backend_cls = any_attn.get_attn_backend()
        num_kv_heads = any_attn.num_kv_heads
        head_size = any_attn.head_size
        cache_dtype_str = vllm_config.cache_config.cache_dtype
        if num_blocks_override is not None:
            num_blocks = num_blocks_override
        else:
            num_blocks = (num_slots + RESERVED_PHYSICAL_SLOTS) * blocks_per_slot
        shape = backend_cls.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
        )
        torch_dtype = any_attn.kv_cache_torch_dtype
        for name in attn_layer_names:
            kv_caches[name] = torch.zeros(shape, dtype=torch_dtype, device=device)

    for name in gdn_layer_names:
        layer = static_forward_context[name]
        conv_shape, ssm_shape = layer.get_state_shape()
        conv_dtype, ssm_dtype = layer.get_state_dtype()
        total_physical_slots = num_slots + RESERVED_PHYSICAL_SLOTS
        conv_state = torch.zeros(
            (total_physical_slots, *conv_shape), dtype=conv_dtype, device=device
        )
        # Phase 2 (2026-07-18): SSM/recurrent state gets num_speculative_tokens
        # EXTRA dedicated rows per physical slot -- one per non-anchor MTP
        # candidate position -- on top of the ordinary one row per physical
        # slot ("column 0", shared with the non-spec/chunked/prefill path).
        # See _ssm_spec_row's docstring for the addressing scheme and its
        # direct verification against the real spec-decode GDN kernel.
        # num_speculative_tokens=0 (no MTP configured) reduces this to
        # exactly the previous allocation -- byte-for-byte unaffected.
        ssm_rows_per_slot = 1 + num_speculative_tokens
        ssm_state = torch.zeros(
            (total_physical_slots * ssm_rows_per_slot, *ssm_shape), dtype=ssm_dtype, device=device
        )
        kv_caches[name] = (conv_state, ssm_state)

    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_caches, static_forward_context, runner_kv_caches)
    return kv_caches


from runtime.metadata_builders import (  # noqa: E402
    build_attention_metadata,
    build_attention_metadata_batch,
    build_gdn_metadata,
    build_gdn_metadata_batch,
    build_gdn_metadata_spec_batch,
)


def _install_triton_norm_ops_once() -> None:
    """Install Triton-fused RMSNorm ops (vLLM C ext lacks them on this machine).
    Must be called AFTER create_engine_config() because that call resets
    IR op priorities via KernelConfig.ir_op_priority.set_priority()."""
    try:
        from runtime.triton_norm_ops import install_triton_norm_ops

        install_triton_norm_ops()
    except Exception:
        pass
    try:
        from runtime.gemma_norm_patch import patch_gemma_rms_norm

        patch_gemma_rms_norm()
    except Exception:
        pass


def build_vllm_config(
    *,
    model: str,
    kv_cache_dtype: str = "fp8_e4m3",
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.5,
    speculative_config: dict | None = None,
) -> VllmConfig:
    _ensure_sm120_backend_registered()
    args = EngineArgs(
        model=model,
        kv_cache_dtype=kv_cache_dtype,
        attention_backend=AttentionBackendEnum.CUSTOM,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        disable_log_stats=True,
        language_model_only=True,
        async_scheduling=False,
        speculative_config=speculative_config,
    )
    config = args.create_engine_config()
    _install_triton_norm_ops_once()
    return config


from runtime.backends.qwen36 import Qwen36Backend  # noqa: E402
from runtime.cuda_graphs import CapturedBatchDecodeGraph, CapturedMTPDraftStepGraph  # noqa: E402
from runtime.gdn_state import GdnStateManager  # noqa: E402
from runtime.logprobs import compute_logprobs  # noqa: E402
from runtime.model_spec import ModelSpec  # noqa: E402
from runtime.prefix_cache import PrefixCacheOps  # noqa: E402
from server.metrics import (  # noqa: E402
    record_prefix_cache_hit,
    record_prefix_cache_miss,
)


class DirectModelRunner:
    """Owns the model, the 4-slot KV/GDN state tensors, and drives forward
    passes directly. This round: single request, slot 0 only."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_slots: int = NUM_SLOTS,
        block_size: int = 64,
        blocks_per_slot: int = 128,
        num_blocks: int | None = None,
        auto_profile_blocks: bool = False,
        gpu_memory_utilization: float = 0.85,
        enable_cudagraph: bool = False,
        enable_block_table: bool = False,
        enable_prefix_cache: bool = False,
        enable_persistent_prefix_cache: bool = False,
        gdn_checkpoint_byte_budget: int = 8 * 2**30,
    ) -> None:
        # 2026-07-18, D3 memory-growth fix: this whole class is a pure
        # inference runtime (never computes a backward pass) but, unlike
        # real vLLM's ``GPUModelRunner`` (whose ``execute_model`` always
        # runs under ``@torch.inference_mode()``), NOTHING in this
        # hand-rolled runner ever disabled autograd -- confirmed by
        # grepping this whole file for "grad" before this fix: zero hits.
        # Every real (non-CUDA-graph) forward call (``_forward``/
        # ``_forward_batch``/``_mtp_forward``/``_mtp_forward_batch``,
        # exercised every round via the eager step-0 fallback whenever
        # active slots' committed lengths are ragged -- the common case at
        # real draft-acceptance rates < 100%) therefore built a full
        # autograd graph rooted at the model's own parameters
        # (``requires_grad=True`` by default, never explicitly frozen by
        # this project's loading path). Root-caused via
        # ``benchmarks/memory_growth_diag.py``: ``torch.cuda
        # .memory_allocated()`` (NOT just ``memory_reserved()``) grew
        # continuously and monotonically round over round with no
        # plateau -- real live-tensor growth, not allocator fragmentation
        # -- reaching 69055 MiB allocated / 97261 MiB nvidia-smi (99.3% of
        # the 97887 MiB card) after 3 W1-S passes, matching the review's
        # reported near-OOM figure almost exactly. ``torch.set_grad_enabled
        # (False)`` (process-global, not a context manager that needs a
        # matching exit -- this runner's process never needs grad) is the
        # standard fix for this exact class of bug and is set as early as
        # possible, before any model construction or forward call.
        torch.set_grad_enabled(False)

        self.vllm_config = vllm_config
        self.num_slots = num_slots
        self.block_size = block_size
        self.blocks_per_slot = blocks_per_slot
        self._auto_profile_blocks = auto_profile_blocks
        self._gpu_memory_utilization = gpu_memory_utilization
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        # P0 (2026-07-19, notes/prefix-cache-design.md sec 5 -- "P0 --
        # block-table indirection substrate"): block_table[slot] is a
        # per-logical-slot list of physical block ids, indexed by logical
        # page position. Built unconditionally (cheap: num_slots small
        # Python lists) so the dedicated equivalence tests can check it
        # regardless of enable_block_table; only CONSULTED by the
        # metadata/slot-mapping/CUDA-graph-fill code paths below when
        # enable_block_table=True. Default False preserves every existing
        # caller's behavior byte-for-byte (this project's established
        # feature-flag convention -- see enable_cudagraph above).
        #
        # P1 (2026-07-19, notes/prefix-cache-design.md sec 5 -- "P1 --
        # Dynamic free-list allocator + reference counting"): P0's
        # ``_initial_block_table`` static per-slot partition (every slot
        # pre-populated with its own fixed contiguous blocks_per_slot-sized
        # range, byte-identical to the old arange addressing) is REPLACED
        # here by a real ``BlockPool`` -- a free queue + ref-counting
        # allocator over the shared pool of physical blocks, excluding
        # reserved physical block 0 (INV7). Every slot now starts with an
        # EMPTY block_table and grows it ON DEMAND (see ``_ensure_blocks``,
        # called from every attention-metadata/slot-mapping/CUDA-graph-fill
        # call site that used to just read ``self.block_table[slot]``
        # as-is) as its kv_len actually grows, instead of every slot
        # permanently reserving its whole blocks_per_slot capacity
        # up front. ``_initial_block_table`` itself is kept, UNCHANGED, as
        # a standalone function -- ``benchmarks/prefix_cache_block_table_
        # check.py``'s arange-equivalence check still imports and calls it
        # directly (it never was, and still isn't, about what
        # DirectModelRunner's own initial state looks like) -- it is simply
        # no longer what populates ``self.block_table`` here.
        #
        # Still NO cross-slot sharing this phase: every block, once
        # allocated, has exactly one referencer (``Block.ref_cnt`` is
        # always 0 or 1) -- see ``BlockPool``'s docstring. This is what
        # keeps end-to-end *behavior* identical to P0/pre-P0 while making
        # *placement* genuinely dynamic (a single slot's own blocks may be
        # non-contiguous after any churn of allocate/free cycles -- the
        # thing this phase's own dedicated tests prove the block-table +
        # CUDA-graph path tolerates, not just P0's trivial contiguous case).
        self.enable_block_table = enable_block_table
        # P2 (2026-07-19, notes/prefix-cache-design.md sec 5, "P2 -- Fan-out
        # fork (Pattern A, same-round sharing)"): OPT-IN, default False --
        # preserves every existing caller's behavior byte-for-byte (this
        # project's established feature-flag convention, see enable_block_
        # table/enable_cudagraph above). When True, ``mtp_prefill_fanout_
        # batch`` detects a common token prefix among a same-round admit
        # batch and forks it (leader prefills the shared prefix once,
        # siblings reference the leader's [0, Lc) attention blocks + restore
        # the leader's GDN snapshot + continue-prefill only their own
        # suffixes). Requires ``enable_block_table=True`` (the fork reuses
        # the P1 block-table/ref-counting substrate -- it manipulates
        # ``block_table``/``BlockPool.reference`` directly); with the flag
        # off, OR when fewer than two same-round requests share at least one
        # full block of prefix, ``mtp_prefill_fanout_batch`` falls back to
        # the exact ``mtp_prefill_batch`` path -- byte-identical to P1.
        self.enable_prefix_cache = enable_prefix_cache
        if enable_prefix_cache and not enable_block_table:
            raise ValueError(
                "enable_prefix_cache=True requires enable_block_table=True "
                "(the fan-out fork reuses the P1 block-table/ref-counting substrate)"
            )
        # P3 persistent content-addressed prefix cache (notes/2026-07-19-p3-
        # implementation-plan.md, P3.1): OPT-IN, default False -- preserves
        # every existing caller's behavior byte-for-byte (rollback spine:
        # flag off => byte-for-byte P2; persistent lookup L=0 => P2 fan-out/
        # cold). Requires enable_prefix_cache=True (it builds on the P2 fan-out
        # substrate: block_table/BlockPool/restore_gdn_state(allow_cross_slot)),
        # raising on misconfiguration exactly like the P2 guard above. When on,
        # populate-on-completion writes a content index + persistent GDN
        # checkpoint pool, and mtp_prefill_with_cache serves restore-and-
        # continue hits -- exercised ONLY by the dedicated test in P3.1 (the
        # production prefill entrypoint is untouched this round).
        self.enable_persistent_prefix_cache = enable_persistent_prefix_cache
        if enable_persistent_prefix_cache and not enable_prefix_cache:
            raise ValueError(
                "enable_persistent_prefix_cache=True requires enable_prefix_cache=True "
                "(the persistent cache builds on the P2 fan-out/ref-counting substrate)"
            )
        # kv_cache_dtype is carried in every block's chained hash extra_keys so
        # fp8 vs nvfp4 KV can never collide on the same token prefix (R7).
        self.kv_cache_dtype = vllm_config.cache_config.cache_dtype
        self.gdn_checkpoint_byte_budget = gdn_checkpoint_byte_budget
        self.block_table: list[list[int]] = [[] for _ in range(num_slots)]
        self._num_blocks_override = num_blocks
        _effective_num_blocks = (
            num_blocks
            if num_blocks is not None
            else (num_slots + RESERVED_PHYSICAL_SLOTS) * blocks_per_slot
        )
        self.block_pool = BlockPool(
            num_blocks=_effective_num_blocks,
            reserved=RESERVED_PHYSICAL_SLOTS,
        )

        # 2026-07-17, Phase 3 (notes/2026-07-17-post-ragged-round-next-steps.md):
        # OPT-IN, default False -- preserves every existing caller's
        # behavior byte-for-byte (every correctness suite in this project
        # constructs a runner with ``num_slots`` sized to its OWN real slot
        # count, no spare capacity reserved for a captured graph's
        # disposable warmup slots; turning this on unconditionally would
        # break them, since ``CapturedBatchDecodeGraph`` permanently
        # reserves the LAST ``batch_size`` logical slots of ``num_slots``
        # for its own warmup -- see that class's docstring -- and several
        # existing tests use those exact slot indices as real,
        # independent reference slots, e.g. ``mtp_batch_verify_check.py``'s
        # ``ref_slots = [4, 5, 6, 7]`` at ``num_slots=8``). A caller that
        # wants ``mtp_verify_and_commit_batch`` to graph-capture its verify
        # forward must pass ``enable_cudagraph=True`` AND size ``num_slots``
        # to at least twice the real concurrency it plans to use (the extra
        # half is reserved warmup capacity, never touched by real request
        # traffic) -- see ``_get_verify_graph``.
        self.enable_cudagraph = enable_cudagraph
        self._verify_graphs: dict[tuple[int, int], CapturedBatchDecodeGraph] = {}
        self._draft_step_graphs: dict[tuple[int, int], CapturedMTPDraftStepGraph] = {}

        # A2: prefer B12x NVFP4 kernel on SM120+ (bit-exact, ~6% faster)
        from runtime.nvfp4_b12x_patch import patch_nvfp4_prefer_b12x

        patch_nvfp4_prefer_b12x()

        # A2: option to force direct CUTLASS (bypasses FlashInfer Python wrapper)
        from runtime.nvfp4_cutlass_direct_patch import patch_nvfp4_prefer_cutlass_direct

        patch_nvfp4_prefer_cutlass_direct()

        # A2: 自研 SM120 NVFP4 GEMM kernel (per-shape tile config, bit-exact)
        from runtime.nvfp4_custom_gemm import patch_nvfp4_custom_gemm

        patch_nvfp4_custom_gemm()

        with set_current_vllm_config(vllm_config):
            init_method = get_distributed_init_method("127.0.0.1", get_open_port())
            init_worker_distributed_environment(
                vllm_config, rank=0, distributed_init_method=init_method, local_rank=0
            )
            self.model = get_model(vllm_config=vllm_config)

        # A2: patch NVFP4 GEMM to cuDNN backend (12.6% faster, bit-exact)
        from runtime.nvfp4_cudnn_patch import patch_nvfp4_to_cudnn

        patch_nvfp4_to_cudnn()

        sfc = vllm_config.compilation_config.static_forward_context
        self.static_forward_context = sfc
        self.attn_layer_names: list[str] = []
        self.gdn_layer_names: list[str] = []
        for name, layer in sfc.items():
            if hasattr(layer, "get_state_shape"):
                self.gdn_layer_names.append(name)
            else:
                self.attn_layer_names.append(name)
        if not self.attn_layer_names or not self.gdn_layer_names:
            raise RuntimeError(
                f"expected both attention and GDN layers, got "
                f"{len(self.attn_layer_names)} attn / {len(self.gdn_layer_names)} gdn"
            )

        # E1 Phase 1: explicit model spec (frozen architecture parameters)
        self.spec = ModelSpec.from_runner_init(
            model_id=vllm_config.model_config.model,
            architecture=getattr(
                vllm_config.model_config,
                "architecture",
                "Qwen3_5ForConditionalGeneration",
            ),
            attn_layer_names=self.attn_layer_names,
            gdn_layer_names=self.gdn_layer_names,
            kv_dtype=self._kv_cache_dtype if hasattr(self, "_kv_cache_dtype") else "fp8_e4m3",
            block_size=self.block_size,
        )

        # E1 Phase 2: MTP backend (delegates model-specific operations)
        self.backend = Qwen36Backend(self)
        self.prefix_cache = PrefixCacheOps(self)
        self.gdn_state = GdnStateManager(self)

        # Real MTP draft model (2026-07-17, Phase 2 / sol's "Option A"),
        # loaded ONLY if the caller configured speculative decoding via
        # build_vllm_config(speculative_config=...). Uses vLLM's own real
        # loading mechanism (load_eagle_model -- also used by vLLM's real
        # MTPSpeculator, not just EAGLE) so embed_tokens/lm_head sharing
        # matches production exactly, nothing hand-rolled. Must load
        # BEFORE _allocate_and_bind_kv_caches() so the draft's own
        # attention layer registers into the SAME static_forward_context
        # this project's existing generic KV-cache-allocation machinery
        # already iterates over -- confirmed by reading vLLM's own
        # DraftModelSpeculator.load_model() (vllm/v1/worker/gpu/spec_decode
        # /speculator.py:153-170), which snapshots attention layer names
        # before/after loading the draft for the exact same reason (there
        # via get_layers_from_vllm_config(..., AttentionLayerBase); here
        # via a direct before/after diff of static_forward_context, which
        # is equivalent since every layer -- attention or GDN -- is
        # registered into that same dict).
        self.mtp_model = None
        self.mtp_attn_layer_names: list[str] = []
        self.num_speculative_tokens: int | None = None
        if vllm_config.speculative_config is not None:
            from runtime.compat_vllm import load_eagle_model

            names_before = set(sfc.keys())
            with set_current_vllm_config(vllm_config):
                self.mtp_model = load_eagle_model(self.model, vllm_config)
            names_after = set(sfc.keys())
            self.mtp_attn_layer_names = sorted(names_after - names_before)
            if not self.mtp_attn_layer_names:
                raise RuntimeError("loading the MTP draft model registered no new layers")
            for name in self.mtp_attn_layer_names:
                if hasattr(sfc[name], "get_state_shape"):
                    raise RuntimeError(f"unexpected GDN layer in MTP draft model: {name}")
            self.num_speculative_tokens = vllm_config.speculative_config.num_speculative_tokens
            # E1: update spec with MTP info
            self.spec = ModelSpec.from_runner_init(
                model_id=self.spec.model_id,
                architecture=self.spec.architecture,
                attn_layer_names=self.attn_layer_names,
                gdn_layer_names=self.gdn_layer_names,
                mtp_model_id=(
                    vllm_config.speculative_config.model
                    if hasattr(vllm_config.speculative_config, "model")
                    else "mtp"
                ),
                mtp_attn_layer_names=self.mtp_attn_layer_names,
                num_speculative_tokens=self.num_speculative_tokens,
                kv_dtype=self.spec.kv_dtype,
                block_size=self.block_size,
            )

        self._allocate_and_bind_kv_caches()
        self._allocate_gdn_snapshot_buffers()
        if self.enable_persistent_prefix_cache:
            self._allocate_gdn_checkpoint_pool()
            # P3.2 lockstep eviction (INV3/R5, both directions): when
            # BlockPool._evict_one reclaims a still-hashed attention block, drop
            # the co-keyed GDN checkpoint too. evict_gdn_checkpoint is the
            # reverse direction as well (a budget/pool-driven checkpoint eviction
            # drops the co-keyed attention block's hash if that block is free).
            # Only wired under the flag: blocks are only ever hashed when the
            # persistent cache is on, so _evict_one never invokes this otherwise.
            self.block_pool._on_evict_block = self.evict_gdn_checkpoint

        # Per-slot bookkeeping: attention kv_len (tokens actually written into
        # the paged KV cache) and GDN "has state been initialized" flag.
        self.slot_kv_len = [0] * num_slots
        self.slot_gdn_initialized = [False] * num_slots

        # Per-slot MTP state (explicit fields, not implicit -- 2026-07-17
        # Codex-sol review asked for this precisely so a live multi-round
        # loop can't silently conflate "physically written" with
        # "committed"). ``slot_kv_len``/``slot_gdn_initialized`` above
        # ARE the target's committed_len/init-state -- no separate
        # "committed_len" field is added since that would just be a second
        # name for the same quantity; what's genuinely new is the DRAFT
        # model's own sync length (a different KV cache, tracked
        # separately) and the in-flight pending proposal.
        self.slot_draft_sync_len = [0] * num_slots
        self.slot_pending_draft_tokens: list[list[int] | None] = [None] * num_slots
        self.slot_gdn_snapshot_gen = [0] * num_slots

        # Phase 2 (2026-07-18): per-slot "real committed length from this
        # slot's own last spec-decode GDN verify round" -- read by
        # build_gdn_metadata_spec_batch to select which of the previous
        # round's K+1 dedicated SSM rows holds the valid state to resume
        # from (see _ssm_spec_row/build_gdn_metadata_spec_batch). Bootstrap
        # value is 1 (not 0) for a slot's first-ever spec verify right
        # after a real prefill -- selects column 0, the same physical row
        # the chunked prefill forward itself wrote into. Reset to 1 on
        # ``reset_slot`` and explicitly re-set to 1 in both
        # ``mtp_prefill_batch`` and ``mtp_prefill`` for defense in depth.
        # Phase B (2026-07-18): also read/updated by ``mtp_verify_and_commit``
        # (the singular/looped sibling) -- both production verify paths
        # share this bookkeeping now.
        self.slot_num_accepted_tokens = [1] * num_slots

        # P3 per-slot hash-chain state (notes/2026-07-19-p3-implementation-plan
        # .md, P3.1 step 4), reset in reset_slot. slot_block_hashes[s][i] is the
        # chained BlockHash of block i (depends on all tokens 0..(i+1)*block_size);
        # slot_published_blocks[s] is the count of this slot's blocks already
        # published to the content index (the write cursor for
        # _publish_committed_blocks). Built unconditionally (cheap small Python
        # lists) so the dedicated test can inspect them regardless of the flag;
        # only MUTATED by the persistent-cache write/read paths when the flag is
        # on.
        self.slot_block_hashes: list[list[BlockHash]] = [[] for _ in range(num_slots)]
        self.slot_published_blocks: list[int] = [0] * num_slots
        # P3.2 decode-position populate: the full committed token sequence for
        # each slot (positions [0, slot_kv_len[s])). Decode-produced blocks hash
        # tokens that may straddle the prompt tail + decode head, so the whole
        # sequence must be available. Seeded with the prompt at prefill (inside
        # _publish_committed_blocks), extended on each verify-commit (publish_
        # committed_decode_blocks), reset in reset_slot. Only mutated under the
        # flag; built unconditionally (cheap small lists) like slot_block_hashes.
        self.slot_committed_tokens: list[list[int]] = [[] for _ in range(num_slots)]

        # Split-KV parallelism for decode/verify-shaped batched kernel calls
        # (2026-07-17, found via direct source comparison after the
        # coordinator's own nvidia-smi monitoring caught persistently low
        # ~30% GPU utilization in the batched MTP path despite ~95%
        # CUDA-event-measured busy time -- a DIFFERENT dimension from
        # "is a kernel running right now" (busy%) than "how much of the
        # 188-SM array does any ONE kernel call actually occupy"
        # (occupancy), and it is this second dimension that was starved).
        # `build_attention_metadata_batch`'s DEFAULT (this eager path's
        # only caller, until now) derives `kv_split_size` from the
        # request's OWN live kv_len, which forces `max_num_splits == 1`
        # (literally zero split-KV parallelism) unconditionally -- the
        # real, production `SM120GQAMetadataBuilder.build()`
        # (`vllm/v1/attention/backends/sm120_gqa.py`) NEVER does this: it
        # always derives a FIXED `kv_split_size` from a build-time bound
        # (there, `max_model_len`; here, this runner's own real per-slot
        # capacity ceiling `blocks_per_slot * block_size`, the same L the
        # CUDA-graph-safety proof in `build_attention_metadata_batch`'s
        # docstring already establishes as a valid upper bound for every
        # real kv_len this runner will ever see) targeting
        # `_DECODE_TARGET_SPLITS_PER_REQ = 32` splits/request -- a value
        # that project's own sweep (kv_len 2000-131072) found best; this
        # project's OWN (not-yet-wired-into-production) `CapturedBatchDecodeGraph`
        # class used a stale `TARGET_SPLITS = 16` from an earlier round,
        # predating that later tuning -- 64 is used here to match the
        # CURRENT best-known value, not the stale one. Confirmed the SAME
        # underlying kernel is used on both sides of the W1-S native
        # comparison (`launch_test_server.py` defaults to
        # `--attention-backend CUSTOM`, this project's own SM120GQABackend
        # unless `--baseline-flashinfer` is passed) -- so this is a
        # same-kernel, different-launch-configuration gap, not a
        # different-kernel confound.
        # NOTE: 32 was tuned when SM count was believed to be 132.
        # Actual SM120 has 188 SMs → 32×4KVH=128 CTAs fills only 68%.
        # Optimal ≈ 188/4 = 47 splits. Re-sweep needed (legacy path only;
        # Laguna/sparkinfer queries SM count dynamically and is unaffected).
        _DECODE_TARGET_SPLITS_PER_REQ = 32
        capacity = self.blocks_per_slot * self.block_size
        self.decode_fixed_kv_split_size = max(1, -(-capacity // _DECODE_TARGET_SPLITS_PER_REQ))
        self.decode_fixed_max_num_splits = _DECODE_TARGET_SPLITS_PER_REQ

        self._warmup()

        # Pre-capture every real batch_size this runner's configured spare
        # capacity supports, so the one-time capture cost (a few extra
        # warmup forward passes per size -- see ``CapturedBatchDecodeGraph
        # .capture()``) happens HERE, during construction, not inside the
        # first few timed rounds of a real measurement (matches this
        # method's own "pay setup cost once at construction" philosophy).
        # Requires MTP to be configured (``num_speculative_tokens`` is
        # unknown otherwise, and this graph is only ever used from
        # ``mtp_verify_and_commit_batch``).
        if self.enable_cudagraph and self.num_speculative_tokens is not None:
            if self.num_slots >= 2 * self.num_slots:
                self._precapture_verify_graphs()
                self._precapture_draft_step_graphs()

    def _precapture_verify_graphs(self) -> None:
        # 2026-07-18, Phase 2 CUDA-graph reconciliation: only qo_len=k+1 is
        # ever needed now. The old rationale for precapturing every
        # qo_len in 1..k+1 (the recompute-forward graph-reuse path, which
        # needed a graph at whatever committed_len 1..k a ragged recompute
        # group happened to land on) no longer applies -- Phase 2 removed
        # the separate recompute forward entirely, so
        # mtp_verify_and_commit_batch's verify step now ALWAYS replays at
        # exactly qo_len=k+1, regardless of each slot's own accept/reject
        # outcome (see that method's docstring). Precapturing the other
        # qo_len values would just be wasted capture time/GPU memory for
        # shapes nothing calls anymore.
        max_batch = self.num_slots // 2
        for batch_size in range(1, max_batch + 1):
            self._get_verify_graph(batch_size, self.num_speculative_tokens + 1)

    def _precapture_draft_step_graphs(self) -> None:
        # 2026-07-17, Phase 3 round 2: precapture qo_len=1 (the K-1
        # continuation steps) AND every qo_len in 1..k+1 (step 0's own
        # shape for the full-accept group -- always k+1 -- and the
        # recompute group's uniform special case -- 1..k) so NEITHER step
        # 0 nor the continuation loop ever lazily captures during a real
        # timed round.
        max_batch = self.num_slots // 2
        for batch_size in range(1, max_batch + 1):
            for qo_len in range(1, self.num_speculative_tokens + 2):
                self._get_draft_step_graph(batch_size, qo_len)

    def _get_draft_step_graph(
        self, batch_size: int, qo_len: int = 1
    ) -> CapturedMTPDraftStepGraph | None:
        """Lazily construct + capture (and cache, keyed by
        ``(batch_size, qo_len)``) a ``CapturedMTPDraftStepGraph`` for the
        MTP draft model's qo_len=1 continuation step OR (2026-07-17,
        generalized) step 0's resync when its own ``num_new_tokens`` is
        uniform -- see that class's docstring. Same deliberate
        ``None``-on-insufficient-capacity fallback contract as
        ``_get_verify_graph``."""
        key = (batch_size, qo_len)
        cached = self._draft_step_graphs.get(key)
        if cached is not None:
            return cached
        if batch_size > self.num_slots or self.mtp_model is None:
            return None
        if self.num_slots >= 2 * batch_size:
            graph = CapturedMTPDraftStepGraph(self, batch_size=batch_size, qo_len=qo_len)
            graph.capture()
        else:
            warmup_slots = list(range(batch_size))
            graph = CapturedMTPDraftStepGraph(
                self, batch_size=batch_size, qo_len=qo_len, warmup_slots=warmup_slots
            )
            graph.capture()
        self._draft_step_graphs[key] = graph
        return graph

    def precapture_cuda_graphs(
        self, batch_sizes: list[int] | None = None, qo_lens: list[int] | None = None
    ) -> None:
        """Pre-capture CUDA graphs during initialization, before any real
        traffic. Uses real slots 0..batch_size-1 for warmup, then resets
        them so they are fresh for real traffic. This eliminates the need
        for permanently reserved warmup slots (which doubled KV cache
        memory)."""
        if not self.enable_cudagraph:
            return
        if batch_sizes is None:
            batch_sizes = [self.num_slots]
        if qo_lens is None:
            qo_lens = [1]
            if self.num_speculative_tokens is not None:
                qo_lens.append(self.num_speculative_tokens + 1)
        draft_qo_lens = qo_lens
        if self.mtp_model is not None and self.num_speculative_tokens is not None:
            draft_qo_lens = list(range(1, self.num_speculative_tokens + 2))
        for bs in batch_sizes:
            if bs > self.num_slots:
                raise ValueError(f"batch_size {bs} exceeds num_slots {self.num_slots}")
            warmup_slots = list(range(bs))
            for qo in qo_lens:
                key = (bs, qo)
                if key not in self._verify_graphs:
                    graph = CapturedBatchDecodeGraph(
                        self,
                        bs,
                        qo_len=qo,
                        warmup_slots=warmup_slots,
                    )
                    graph.capture()
                    self._verify_graphs[key] = graph
        if self.mtp_model is not None:
            for bs in batch_sizes:
                warmup_slots = list(range(bs))
                for qo in draft_qo_lens:
                    key = (bs, qo)
                    if key not in self._draft_step_graphs:
                        graph = CapturedMTPDraftStepGraph(
                            self,
                            bs,
                            qo_len=qo,
                            warmup_slots=warmup_slots,
                        )
                        graph.capture()
                        self._draft_step_graphs[key] = graph
        for bs in batch_sizes:
            for slot in range(bs):
                if self.slot_kv_len[slot] != 0:
                    self.reset_slot(slot)

    def _get_verify_graph(self, batch_size: int, qo_len: int) -> CapturedBatchDecodeGraph | None:
        """Lazily construct + capture (and cache, keyed by
        ``(batch_size, qo_len)``) a ``CapturedBatchDecodeGraph`` for the
        target model's verify forward. Returns ``None`` -- a deliberate,
        documented eager-fallback signal, NOT an error -- when this runner
        wasn't configured with enough spare capacity
        (``num_slots >= 2*batch_size``) to reserve that graph's own
        disposable warmup slots. This is the expected, correct outcome for
        every existing (non-cudagraph) correctness suite in this project
        (``enable_cudagraph`` defaults to ``False`` there, so this method is
        never even called), and also the correct outcome for a genuinely
        unusual batch_size a graph-enabled caller never pre-captured (e.g.
        one bigger than ``num_slots // 2`` -- cannot happen from
        ``_precapture_verify_graphs``'s own range, but this method stays
        safe if called with an out-of-range size directly).

        Capturing a NEW graph resets its own reserved warmup slots
        (``runner.reset_slot``) immediately afterward -- ``capture()``
        requires its warmup slots to be fresh (``slot_kv_len == 0``), and
        different ``batch_size`` graphs' reserved-slot RANGES overlap
        (``CapturedBatchDecodeGraph`` reserves the LAST ``batch_size``
        logical slots of ``num_slots``, so e.g. batch_size=2 and
        batch_size=4 graphs share slots ``num_slots-2 .. num_slots-1``) --
        without this reset, capturing a second graph whose reserved range
        overlaps a previously-captured graph's would hit that freshness
        check and fail. This is safe because a graph's reserved slots are
        NEVER touched again after its own ``capture()`` call returns (never
        passed to ``replay()`` or any other runner method) -- resetting
        them costs nothing but bookkeeping."""
        key = (batch_size, qo_len)
        cached = self._verify_graphs.get(key)
        if cached is not None:
            return cached
        if batch_size > self.num_slots:
            return None
        if self.num_slots >= 2 * batch_size:
            graph = CapturedBatchDecodeGraph(self, batch_size=batch_size, qo_len=qo_len)
            graph.capture()
            for s in graph._warmup_slots:
                self.reset_slot(s)
        else:
            warmup_slots = list(range(batch_size))
            graph = CapturedBatchDecodeGraph(
                self, batch_size=batch_size, qo_len=qo_len, warmup_slots=warmup_slots
            )
            graph.capture()
        self._verify_graphs[key] = graph
        return graph

    def _warmup(self) -> None:
        """Real vLLM always runs a profiling/warmup forward before serving
        (see gpu_model_runner.py's warmup pass, and this project's own
        server logs: "Initial profiling/warmup run took N s"). Motivated by
        a real, isolated repro (see notes/direct-model-runner-design.md's
        "deep dive on the conv_state lead" section): causal_conv1d_fn's
        Triton kernel returns an all-zero result on its first-ever call in
        a process, in complete isolation, unrelated to this runtime's code.
        Kept here since it mirrors real vLLM's own behavior and cannot
        hurt, but -- reported honestly -- this alone does NOT fix the real
        model's wrong output (verified: neither a 1-token nor a
        shape-matched 5-token warmup changed the observed wrong completion
        for "The capital of France is"). The cold-start bug is real but
        evidently not the whole story; see the design doc for the
        follow-up isolated tests that show a messier, not-yet-characterized
        pattern (interleaved shapes don't self-correct the way repeating
        one shape does) and the next debugging steps."""
        try:
            self.prefill(0, [0, 0, 0, 0, 0])
        finally:
            self.reset_slot(0)

    def _allocate_and_bind_kv_caches(self) -> None:
        num_blocks_override = self._num_blocks_override
        if num_blocks_override is None and self._auto_profile_blocks:
            num_blocks_override = profile_kv_cache_blocks(
                self.static_forward_context,
                self.vllm_config,
                self.device,
                num_slots=self.num_slots,
                block_size=self.block_size,
                gpu_memory_utilization=self._gpu_memory_utilization,
                num_speculative_tokens=self.num_speculative_tokens or 0,
            )
            self._num_blocks_override = num_blocks_override
            self.block_pool = BlockPool(
                num_blocks=num_blocks_override,
                reserved=RESERVED_PHYSICAL_SLOTS,
            )
        self.kv_caches = allocate_fixed_slot_kv_caches(
            self.static_forward_context,
            self.vllm_config,
            self.device,
            num_slots=self.num_slots,
            block_size=self.block_size,
            blocks_per_slot=self.blocks_per_slot,
            num_speculative_tokens=self.num_speculative_tokens or 0,
            num_blocks_override=num_blocks_override,
        )

    def _allocate_gdn_snapshot_buffers(self) -> None:
        """Preallocated, GPU-resident, fixed-address storage for
        ``snapshot_gdn_state``/``restore_gdn_state`` (2026-07-17, Phase 1 of
        ``notes/2026-07-17-post-ragged-round-next-steps.md``). Replaces the
        old per-call ``.detach().to("cpu", copy=True)`` -- Phase 0's real
        ``nsys`` ledger (that doc's section 7) measured this mechanism at
        89-117ms/round of pageable D2H/H2D memcpy-engine time alone, plus a
        comparable amount of host-dispatch gap in the same phases (~30-31%
        of round wall time combined, present in every round -- snapshot
        happens unconditionally for all active slots).

        Sizing rationale (verified against the real call pattern before
        relying on it, per this round's own instructions -- both
        ``mtp_verify_and_commit`` and ``mtp_verify_and_commit_batch`` snap
        each slot in the list AT MOST ONCE per round, and any restore for
        that slot happens later in that SAME round, before the next round's
        snapshot call for that slot can be issued): at most ONE snapshot
        per logical slot is ever outstanding at a time. One buffer entry per
        logical slot (indexed 0..num_slots-1) is therefore sufficient --
        NOT a literal ping-pong double buffer (which would double the VRAM
        cost to ~1.2GB); this is deliberately the plan doc's "~604MB"
        estimate, which already assumed exactly this one-copy-per-slot
        sizing (confirmed against Phase 0's own measured D2H byte count,
        ~604MB for a 4-slot round). The persistent buffer is safe to reuse
        round-over-round without an explicit double-buffer/generation-aware
        allocation scheme because everything here runs on ONE CUDA stream
        in strict Python-issued order: a later round's snapshot() write for
        slot S can only be enqueued after every earlier statement that
        reads slot S's snapshot (i.e. that round's own restore() call, if
        any) has already been issued -- CUDA's own per-stream FIFO
        ordering, not an extra synchronization primitive, is what makes
        this correct. The three safety invariants this class already
        enforces (slot-id tag, generation counter, consumed-once flag) are
        UNCHANGED and still checked before any tensor data is read on
        restore -- they continue to guard against a caller holding a STALE
        snapshot object across rounds, which would otherwise now silently
        alias newer data through the same buffer slot (the checks reject
        it before that data is ever used, exactly as before).

        Indexed directly by LOGICAL slot (0..num_slots-1), unlike
        ``kv_caches`` (which reserves physical index 0 -- see
        ``RESERVED_PHYSICAL_SLOTS``/``_physical_slot``): that reservation
        works around a real vLLM physical-block-addressing convention this
        private buffer is not subject to, so no such offset/reservation is
        needed here.

        Fixed-address discipline (never reallocated after ``__init__``,
        only ever written into via ``copy_``) matches this file's other
        persistent GPU buffers (see ``CapturedBatchDecodeGraph``'s class
        docstring) -- this code path does not currently run inside any CUDA
        graph capture region (``mtp_verify_and_commit``/``_batch`` are
        eager-only; ``CapturedBatchDecodeGraph`` is a separate, not-yet-
        wired-in mechanism per Phase 3 of the same plan doc), but following
        the same discipline now means Phase 3 does not have to revisit this
        buffer's allocation strategy later if GDN snapshot/restore is ever
        folded into a captured graph."""
        self.gdn_snapshot_conv: dict[str, torch.Tensor] = {}
        self.gdn_snapshot_ssm: dict[str, torch.Tensor] = {}
        for name in self.gdn_layer_names:
            conv_state, ssm_state = self.kv_caches[name]
            self.gdn_snapshot_conv[name] = torch.zeros(
                (self.num_slots, *conv_state.shape[1:]),
                dtype=conv_state.dtype,
                device=self.device,
            )
            self.gdn_snapshot_ssm[name] = torch.zeros(
                (self.num_slots, *ssm_state.shape[1:]),
                dtype=ssm_state.dtype,
                device=self.device,
            )

    def _allocate_gdn_checkpoint_pool(self) -> None:
        return self.gdn_state._allocate_gdn_checkpoint_pool()

    def _gdn_ckpt_alloc_slot(self) -> int:
        return self.gdn_state._gdn_ckpt_alloc_slot()

    def materialize_gdn_checkpoint(
        self, slot: int, key: int, hash_value: int, num_tokens: int
    ) -> None:
        return self.gdn_state.materialize_gdn_checkpoint(slot, key, hash_value, num_tokens)

    def checkpoint_view(self, key: int) -> dict | None:
        return self.gdn_state.checkpoint_view(key)

    def evict_gdn_checkpoint(self, key: int) -> None:
        return self.gdn_state.evict_gdn_checkpoint(key)

    def _evict_gdn_checkpoints_for_budget(self, incoming_bytes: int) -> None:
        return self.gdn_state._evict_gdn_checkpoints_for_budget(incoming_bytes)

    def _ensure_blocks(self, slot: int, kv_len_needed: int) -> None:
        """P1 (notes/prefix-cache-design.md sec 5): grow
        ``self.block_table[slot]`` on demand from ``self.block_pool`` so it
        holds at least ``ceil(kv_len_needed / self.block_size)`` physical
        block ids -- called from every code path that is about to build
        attention metadata / a slot-mapping / a CUDA-graph fill for a write
        or read up to position ``kv_len_needed`` (single-request and
        batched target-model forward, single-request and batched MTP
        draft-model forward, both captured-graph ``_fill_buffers``
        methods). A no-op when the table already covers the request -- the
        common per-token-decode-step case, which only needs a fresh
        physical block once every ``block_size`` tokens, not every call.

        Every call site gates on ``self.enable_block_table`` before calling
        this (matching this file's existing per-call-site flag-branch
        convention) -- this method itself always consults ``self
        .block_pool`` unconditionally once called, it does not re-check the
        flag.

        Raises the same ``RuntimeError`` message shape as
        ``build_attention_metadata``/``_batch``'s own capacity check when
        ``kv_len_needed`` would need more than ``self.blocks_per_slot``
        pages -- checked here too (not just left to the metadata builder to
        catch after the fact) so a request that will be rejected anyway
        never consumes a block from the shared pool first."""
        num_pages_needed = (kv_len_needed + self.block_size - 1) // self.block_size
        if num_pages_needed > self.blocks_per_slot:
            raise RuntimeError(
                f"slot {slot} kv_len {kv_len_needed} exceeds this slot's "
                f"{self.blocks_per_slot * self.block_size}-token capacity"
            )
        table = self.block_table[slot]
        grow_by = num_pages_needed - len(table)
        if grow_by > 0:
            table.extend(self.block_pool.allocate(grow_by))

    def _attention_metadata(
        self, slot: int, *, num_new_tokens: int, is_decode: bool
    ) -> SM120GQAMetadata:
        if self.enable_block_table:
            self._ensure_blocks(slot, self.slot_kv_len[slot] + num_new_tokens)
        return build_attention_metadata(
            prior_kv_len=self.slot_kv_len[slot],
            num_new_tokens=num_new_tokens,
            is_decode=is_decode,
            slot=slot,
            block_size=self.block_size,
            blocks_per_slot=self.blocks_per_slot,
            device=self.device,
            block_table=self.block_table[slot] if self.enable_block_table else None,
        )

    def _gdn_metadata(
        self, slot: int, *, num_new_tokens: int, is_decode: bool
    ) -> GDNAttentionMetadata:
        return build_gdn_metadata(
            slot_initialized=self.slot_gdn_initialized[slot],
            num_new_tokens=num_new_tokens,
            is_decode=is_decode,
            slot=slot,
            device=self.device,
        )

    def _slot_mapping(self, slot: int, start_pos: int, num_new_tokens: int) -> torch.Tensor:
        """Flat per-token KV-cache write index: block_id * block_size + offset
        -- the same convention vLLM's own paged attention backends use (see
        attention.py's do_kv_cache_update, which reads this from
        ``forward_context.slot_mapping[layer_name]``, NOT from
        ``attn_metadata`` -- easy to miss, and missing it means K/V are never
        written into the cache at all)."""
        positions = torch.arange(
            start_pos, start_pos + num_new_tokens, dtype=torch.long, device=self.device
        )
        if self.enable_block_table:
            table = self.block_table[slot]
            block_ids = torch.tensor(
                [table[p // self.block_size] for p in range(start_pos, start_pos + num_new_tokens)],
                dtype=torch.long,
                device=self.device,
            )
        else:
            first_block = _physical_slot(slot) * self.blocks_per_slot
            block_ids = first_block + positions // self.block_size
        offsets = positions % self.block_size
        return (block_ids * self.block_size + offsets).to(torch.long)

    def _forward(
        self,
        slot: int,
        token_ids: list[int],
        start_pos: int,
        *,
        is_decode: bool,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        num_new_tokens = len(token_ids)
        attn_meta = self._attention_metadata(
            slot, num_new_tokens=num_new_tokens, is_decode=is_decode
        )
        gdn_meta = self._gdn_metadata(slot, num_new_tokens=num_new_tokens, is_decode=is_decode)
        attn_metadata_dict = {name: attn_meta for name in self.attn_layer_names}
        attn_metadata_dict.update({name: gdn_meta for name in self.gdn_layer_names})
        slot_mapping = self._slot_mapping(slot, start_pos, num_new_tokens)
        slot_mapping_dict = {name: slot_mapping for name in self.attn_layer_names}

        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        positions = torch.arange(
            start_pos, start_pos + num_new_tokens, dtype=torch.long, device=self.device
        )

        with set_forward_context(
            attn_metadata_dict, self.vllm_config, slot_mapping=slot_mapping_dict
        ):
            hidden_states = self.model.forward(input_ids, positions)
        # 2026-07-17, Phase 3 (notes/2026-07-17-post-ragged-round-next-steps.md):
        # the two ``torch.cuda.synchronize()`` calls that used to bracket
        # ``compute_logits`` here were removed -- they block the HOST
        # (Python) thread until every queued GPU op finishes, but neither
        # call was ever needed for CORRECTNESS: ``model.forward()`` and
        # ``compute_logits()`` are both issued on the SAME (default) CUDA
        # stream, so CUDA's own per-stream FIFO ordering already guarantees
        # ``compute_logits`` reads ``hidden_states`` only after `forward()`'s
        # kernels have written it -- exactly the same reasoning
        # ``CapturedBatchDecodeGraph.replay()``'s docstring already
        # established for removing ITS blanket sync (see that class,
        # 2026-07-17 correctness-review round). Any caller that actually
        # needs the values host-side (``.item()``/``.cpu()``/``torch.equal``)
        # already forces an implicit, narrowly-scoped sync at that read --
        # a blanket device-wide sync here was pure per-call dispatch
        # overhead (Phase 0's ``nsys`` ledger measured 3634 kernels/round in
        # the verify phase alone; every method in this file's hot path used
        # to insert two of these), not a safety requirement.
        logits = self.model.compute_logits(hidden_states)

        self.slot_kv_len[slot] += num_new_tokens
        self.slot_gdn_initialized[slot] = True
        if return_hidden:
            return logits, hidden_states
        return logits

    def prefill(self, slot: int, prompt_token_ids: list[int]) -> int:
        """Run the prompt through the model; returns the greedy next token id."""
        if self.slot_kv_len[slot] != 0:
            raise RuntimeError(f"slot {slot} is not fresh (kv_len={self.slot_kv_len[slot]})")
        logits = self._forward(slot, prompt_token_ids, start_pos=0, is_decode=False)
        return int(logits[-1].argmax(dim=-1).item())

    def prefill_sampled(
        self, slot: int, prompt_token_ids: list[int], params: SamplingParams
    ) -> int:
        """Run the prompt through the model; returns the sampled next token id.

        For ``temperature == 0`` this is bit-identical to ``prefill()``.
        """
        if self.slot_kv_len[slot] != 0:
            raise RuntimeError(f"slot {slot} is not fresh (kv_len={self.slot_kv_len[slot]})")
        logits = self._forward(slot, prompt_token_ids, start_pos=0, is_decode=False)
        last_logits = logits[-1].unsqueeze(0)
        gen = make_generator(params.seed)
        return int(sample_from_logits(last_logits, params, generator=gen).item())

    def decode(self, slot: int, token_id: int) -> int:
        """Consume one token, return the greedy next token id."""
        start_pos = self.slot_kv_len[slot]
        logits = self._forward(slot, [token_id], start_pos=start_pos, is_decode=True)
        return int(logits[-1].argmax(dim=-1).item())

    def decode_sampled(self, slot: int, token_id: int, params: SamplingParams) -> int:
        """Consume one token, return the sampled next token id.

        For ``temperature == 0`` this is bit-identical to ``decode()``.
        """
        start_pos = self.slot_kv_len[slot]
        logits = self._forward(slot, [token_id], start_pos=start_pos, is_decode=True)
        last_logits = logits[-1].unsqueeze(0)
        gen = make_generator(params.seed)
        return int(sample_from_logits(last_logits, params, generator=gen).item())

    def _slot_mapping_batch(
        self, slots: list[int], kv_lengths: list[int], qo_len: int | list[int] = 1
    ) -> torch.Tensor:
        """Batched analogue of ``_slot_mapping``: each request contributes
        ``qo_len`` new tokens starting at its own ``kv_lengths[i]``,
        flattened in the SAME per-request-contiguous order ``_forward_batch``
        uses for ``input_ids``/``positions`` (request 0's ``qo_len`` tokens,
        then request 1's, ...). At ``qo_len=1`` this reduces exactly to the
        previously-verified one-position-per-request mapping. ``qo_len`` may
        also be a per-slot RAGGED list (2026-07-17, mirrors
        ``build_attention_metadata_batch``'s identical generalization) --
        a scalar broadcasts to a uniform list, so every existing call site
        is unaffected."""
        num_reqs = len(slots)
        qo_lens = [qo_len] * num_reqs if isinstance(qo_len, int) else list(qo_len)
        positions = [kv_len + j for kv_len, qo in zip(kv_lengths, qo_lens) for j in range(qo)]
        slots_per_token = [slot for slot, qo in zip(slots, qo_lens) for _ in range(qo)]
        if self.enable_block_table:
            block_ids = torch.tensor(
                [
                    self.block_table[slot][pos // self.block_size]
                    for slot, pos in zip(slots_per_token, positions)
                ],
                dtype=torch.long,
                device=self.device,
            )
        else:
            block_ids = torch.tensor(
                [
                    _physical_slot(slot) * self.blocks_per_slot + pos // self.block_size
                    for slot, pos in zip(slots_per_token, positions)
                ],
                dtype=torch.long,
                device=self.device,
            )
        offsets = torch.tensor(
            [pos % self.block_size for pos in positions], dtype=torch.long, device=self.device
        )
        return block_ids * self.block_size + offsets

    def _forward_batch(
        self,
        slot_ids: list[int],
        token_ids,
        kv_lengths: list[int],
        *,
        qo_len: int | list[int] = 1,
        commit: bool = True,
        return_hidden: bool = False,
        is_decode: bool = True,
        fixed_kv_split_size: int | None = None,
        fixed_max_num_splits: int | None = None,
        gdn_spec_num_accepted_tokens_prev: list[int] | None = None,
        logits_last_position_only: bool = False,
    ) -> torch.Tensor:
        """Real batched decode/verify: ONE batched attention/GDN metadata
        object and ONE ``model.forward()`` call covering every listed slot
        -- not a Python loop calling ``_forward``/``decode`` per slot.
        ``kv_lengths`` is the caller-asserted prior KV length (before this
        step's new tokens) for each slot; cross-checked against this
        runner's own ``self.slot_kv_len`` bookkeeping to catch drift early
        rather than silently addressing the wrong cache rows.

        ``qo_len=1`` (the default, unchanged from the original decode-only
        batch path): ``token_ids`` is a flat list, one token id per slot.
        ``qo_len>1`` (MTP/speculative-decode verify, uniform across the
        batch): ``token_ids`` is a list of per-slot token-id lists, each of
        length ``qo_len`` -- the K draft tokens + 1 bonus-position
        placeholder being verified in one batched call.
        Returns logits shaped ``[num_reqs * qo_len, vocab]``, flattened in
        request-then-position order (request 0's qo_len rows, then request
        1's, ...) -- the same order ``SM120GQAImpl.forward()``'s own
        ``q_decode.reshape(num_reqs, qo_len, ...)`` expects.

        ``commit`` (default ``True``, preserving the original decode_batch
        behavior exactly): whether to advance ``self.slot_kv_len`` by
        ``qo_len`` for every listed slot. The forward pass ALWAYS
        physically writes K/V for all ``qo_len`` positions regardless of
        this flag -- ``commit`` only controls this method's own
        bookkeeping. Real MTP verify calls (``verify_batch``/
        ``verify_batch_spec``) pass ``commit=False``, since the actual
        committed length is not known until the caller's accept/reject
        decision runs on the returned logits (2026-07-17, fixing the exact
        "physically-written vs. committed" conflation Codex-sol's review
        flagged) -- the caller (``mtp_verify_and_commit``/``_batch``) is
        responsible for advancing ``slot_kv_len`` by the REAL committed
        length afterward. Attention's own KV needs no explicit rollback
        either way (content/position addressed -- positions beyond the
        real committed length are simply never read again).

        **2026-07-18, Phase B update**: GDN's recurrent state used to need
        an explicit ``snapshot_gdn_state``/``restore_gdn_state`` + a real
        recompute-forward repair on a non-full-accept outcome -- that was
        true for both ``mtp_verify_and_commit`` and
        ``mtp_verify_and_commit_batch`` through 2026-07-18, then only for
        the singular path (Phase 2 migrated the batched path off it), and
        as of Phase B is no longer true for EITHER production verify path:
        both now go through the real spec-decode GDN mechanism
        (``gdn_spec_num_accepted_tokens_prev`` below), under which the
        recurrent state's per-position OUTPUT is already causally valid
        for every candidate position regardless of which are later
        accepted -- only the STATE COMMIT (which physical row survives to
        be read next round) is acceptance-aware, so no rollback is ever
        needed. ``snapshot_gdn_state``/``restore_gdn_state`` themselves are
        retained as tested, standalone primitives (still directly exercised
        by ``benchmarks/mtp_gdn_rollback_check.py`` and several other
        diagnostics -- see ``mtp_verify_and_commit``'s docstring), just no
        longer called from any production verify path.

        ``is_decode`` (2026-07-17 addition, default ``True`` preserving
        ``decode_batch``/``verify_batch``'s existing behavior byte-for-byte):
        forwarded to ``build_attention_metadata_batch``'s own ``is_decode``
        parameter -- see that function's docstring for the real gap this
        closes (``decode_qo_len`` must be 0 for a genuine chunked/prefix
        PREFILL call, not ``qo_len`` unconditionally). Only
        ``mtp_prefill_batch`` passes ``is_decode=False`` explicitly, for its
        genuine target-model prefill forward.

        ``fixed_kv_split_size``/``fixed_max_num_splits`` (both ``None`` by
        default, forwarded as-is to ``build_attention_metadata_batch``):
        without these, that function's default branch derives
        ``kv_split_size`` from this call's own live kv_len, which forces
        ``max_num_splits == 1`` -- literally zero split-KV parallelism.
        Real MTP callers now pass ``self.decode_fixed_kv_split_size``/
        ``self.decode_fixed_max_num_splits`` (computed once in
        ``__init__``, matching native's production
        ``SM120GQAMetadataBuilder``'s own fixed-from-build-time-bound
        derivation) so the SAME decode/verify kernel gets real split-KV
        parallelism here too -- see ``__init__``'s comment for the full
        story (2026-07-17, found after the coordinator's own nvidia-smi
        monitoring caught persistently low GPU utilization in the batched
        MTP path despite high CUDA-event-measured busy time).

        ``qo_len`` as a RAGGED per-request list (2026-07-17, for the
        recompute-fallback batching round): each slot may contribute a
        DIFFERENT number of new tokens this call -- forwarded as-is to
        ``build_attention_metadata_batch``/``build_gdn_metadata_batch``
        (both already generalized for this, see their docstrings) and
        used locally to build per-slot-correct ``positions``/kv_len
        bookkeeping. A scalar ``qo_len`` broadcasts to a uniform list, so
        every existing call site is byte-for-byte unaffected.

        ``gdn_spec_num_accepted_tokens_prev`` (2026-07-18, Phase 2, default
        ``None`` preserving every existing call site byte-for-byte): when
        given (one entry per slot), GDN metadata is built via the REAL
        spec-decode mechanism (``build_gdn_metadata_spec_batch``) instead
        of the chunked/prefill-shaped ``build_gdn_metadata_batch`` --
        K+1 dedicated SSM state rows per slot, acceptance-aware addressing
        selecting which row survives to be read next round, no
        snapshot/restore or recompute-forward needed. Requires a SCALAR,
        uniform ``qo_len`` (always ``num_speculative_tokens + 1`` in
        practice) -- unlike the chunked path this is not generalized to a
        ragged per-request list, since every real spec-decode verify call
        submits the same K+1-token draft for every slot. Only
        ``verify_batch_spec`` passes this.

        ``logits_last_position_only`` (2026-07-18, D1-followup fix, default
        ``False`` preserving every existing call site byte-for-byte): when
        ``True``, ``self.model.compute_logits(...)`` is applied to ONLY the
        last position of each slot's ``qo_len`` block (gathered via
        ``index_select`` right before the vocab-head projection), instead of
        every position -- the returned ``logits`` is then shaped
        ``[num_reqs, vocab]``, NOT ``[num_reqs * qo_len, vocab]``. The full,
        un-gathered ``hidden_states`` is still returned unchanged when
        ``return_hidden=True`` -- only the tensor fed into ``compute_logits``
        is sliced. Found via direct instrumentation
        (``benchmarks/mtp_prefill_batch_memory_diag.py``) profiling the
        16K-context/c=4 shape flagged in
        ``notes/2026-07-18-session-review-and-next-steps.md`` section 12:
        at ``qo_len=16384``/``concurrency=4`` this call's own
        ``compute_logits`` alone allocates a 31040 MiB ``[65536, 248320]``
        bf16 tensor of which only 4 rows (0.006%) are ever read by any
        caller -- only ``mtp_prefill_batch`` needs the anchor logits, and
        only at each slot's OWN last prompt position. ``decode_batch``/
        ``verify_batch``/``verify_batch_spec`` genuinely need every
        position's logits (MTP verify checks every draft token against the
        target's own prediction) and MUST NOT pass this -- it is only safe
        when the caller already only reads the last row per slot, which is
        why only ``mtp_prefill_batch`` sets it.
        """
        num_reqs = len(slot_ids)
        qo_lens = [qo_len] * num_reqs if isinstance(qo_len, int) else list(qo_len)
        if len(qo_lens) != num_reqs:
            raise ValueError("qo_len list must have exactly one entry per slot")

        if isinstance(qo_len, int) and qo_len == 1:
            if not (len(token_ids) == num_reqs and len(kv_lengths) == num_reqs):
                raise ValueError("slot_ids/token_ids/kv_lengths must have equal length")
            flat_token_ids = token_ids
        else:
            if not (
                len(token_ids) == num_reqs
                and len(kv_lengths) == num_reqs
                and all(len(t) == qo for t, qo in zip(token_ids, qo_lens))
            ):
                raise ValueError(
                    "slot_ids/token_ids/kv_lengths must have equal length, and "
                    "every token_ids[i] must have exactly qo_len[i] tokens"
                )
            flat_token_ids = [tok for slot_tokens in token_ids for tok in slot_tokens]

        for slot, kv_len in zip(slot_ids, kv_lengths):
            if kv_len != self.slot_kv_len[slot]:
                raise RuntimeError(
                    f"slot {slot}: caller-provided kv_length {kv_len} != "
                    f"tracked {self.slot_kv_len[slot]}"
                )
            # kv_len == 0 legitimately means "this slot's very first forward"
            # (matches ``prefill()``'s own "fresh slot" definition) -- 2026-07-17
            # relaxation for ``mtp_prefill_batch``, the first real caller that
            # needs a batched forward covering NEVER-forwarded slots.
            # ``build_gdn_metadata_batch``'s qo_len>1 branch already accepts a
            # per-slot ``slot_initialized`` list (passed below) and handles
            # ``False`` correctly (has_initial_state=False is exactly what a
            # fresh slot's chunked GDN forward needs) -- this guard was stricter
            # than the underlying kernel actually requires, a leftover of
            # ``_forward_batch`` previously only ever being called on
            # already-prefilled slots (``decode_batch``/``verify_batch``). Any
            # OTHER "not yet initialized" case (kv_len != 0) still raises,
            # unchanged -- that combination can only mean a caller skipped a
            # real prefill while lying about kv_len, exactly what this check
            # exists to catch.
            if not self.slot_gdn_initialized[slot] and kv_len != 0:
                raise RuntimeError(f"slot {slot} has no GDN state yet (needs a prior prefill)")

        # P1 (notes/prefix-cache-design.md sec 5): grow every listed slot's
        # block_table to cover this call's own new_kv_len (kv_len + qo)
        # BEFORE building metadata/slot-mapping below, which both read
        # self.block_table[slot] as-is.
        if self.enable_block_table:
            for slot, kv_len, qo in zip(slot_ids, kv_lengths, qo_lens):
                self._ensure_blocks(slot, kv_len + qo)

        attn_meta = build_attention_metadata_batch(
            slots=slot_ids,
            prior_kv_lens=kv_lengths,
            block_size=self.block_size,
            blocks_per_slot=self.blocks_per_slot,
            device=self.device,
            qo_len=qo_len,
            is_decode=is_decode,
            fixed_kv_split_size=fixed_kv_split_size,
            fixed_max_num_splits=fixed_max_num_splits,
            block_tables=(
                [self.block_table[s] for s in slot_ids] if self.enable_block_table else None
            ),
        )
        if gdn_spec_num_accepted_tokens_prev is not None:
            if not isinstance(qo_len, int):
                raise ValueError("gdn_spec_num_accepted_tokens_prev requires a scalar qo_len")
            gdn_meta = build_gdn_metadata_spec_batch(
                slots=slot_ids,
                device=self.device,
                qo_len=qo_len,
                num_accepted_tokens_prev=gdn_spec_num_accepted_tokens_prev,
                total_physical_slots=self.num_slots + RESERVED_PHYSICAL_SLOTS,
                num_spec=self.num_speculative_tokens,
            )
        else:
            gdn_meta = build_gdn_metadata_batch(
                slots=slot_ids,
                device=self.device,
                qo_len=qo_len,
                slot_initialized=(
                    [self.slot_gdn_initialized[s] for s in slot_ids]
                    if not (isinstance(qo_len, int) and qo_len == 1)
                    else None
                ),
            )
        attn_metadata_dict = {name: attn_meta for name in self.attn_layer_names}
        attn_metadata_dict.update({name: gdn_meta for name in self.gdn_layer_names})
        slot_mapping = self._slot_mapping_batch(slot_ids, kv_lengths, qo_len=qo_len)
        slot_mapping_dict = {name: slot_mapping for name in self.attn_layer_names}

        input_ids = torch.tensor(flat_token_ids, dtype=torch.long, device=self.device)
        positions = torch.tensor(
            [kv_len + j for kv_len, qo in zip(kv_lengths, qo_lens) for j in range(qo)],
            dtype=torch.long,
            device=self.device,
        )

        with set_forward_context(
            attn_metadata_dict, self.vllm_config, slot_mapping=slot_mapping_dict
        ):
            hidden_states = self.model.forward(input_ids, positions)
        # 2026-07-17, Phase 3: see ``_forward``'s docstring/comment for why
        # the two blanket ``torch.cuda.synchronize()`` calls that used to
        # bracket ``compute_logits`` here were removed -- same-stream
        # ordering already guarantees correctness, and this method (the
        # real per-round verify/recompute/decode hot path) is exactly
        # where Phase 0's ``nsys`` ledger measured the dominant no-kernel
        # gap this removal targets.
        if logits_last_position_only:
            # 2026-07-18, D1-followup fix: project only each slot's own
            # last position through the vocab head -- see this parameter's
            # docstring. ``qo_lens`` (already computed above) gives each
            # slot's own row count; cumulative sum minus 1 is that slot's
            # last row in the request-then-position-flattened layout
            # ``model.forward`` returned.
            last_idx = torch.tensor(
                [sum(qo_lens[: i + 1]) - 1 for i in range(num_reqs)],
                dtype=torch.long,
                device=self.device,
            )
            logits_hidden = hidden_states.index_select(0, last_idx)
        else:
            logits_hidden = hidden_states
        logits = self.model.compute_logits(logits_hidden)

        for slot, qo in zip(slot_ids, qo_lens):
            if commit:
                self.slot_kv_len[slot] += qo
            self.slot_gdn_initialized[slot] = True
        if return_hidden:
            return logits, hidden_states
        return logits

    def decode_batch(
        self, slot_ids: list[int], token_ids: list[int], kv_lengths: list[int]
    ) -> list[int]:
        """Decode one token for each of several active slots via a single
        real batched forward call. Returns the greedy next token id per
        slot, in the same order as ``slot_ids``."""
        logits = self._forward_batch(slot_ids, token_ids, kv_lengths)
        return [int(logits[i].argmax(dim=-1).item()) for i in range(len(slot_ids))]

    def decode_batch_sampled(
        self,
        slot_ids: list[int],
        token_ids: list[int],
        kv_lengths: list[int],
        params_list: list[SamplingParams],
        *,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> list[int] | tuple[list[int], list[dict]]:
        """Decode one token per slot with per-request sampling params.

        Falls back to greedy argmax for any slot whose params have
        ``temperature == 0``, preserving bit-identical behavior.

        When ``return_logprobs`` is True, returns a tuple of
        ``(token_ids, logprobs_list)`` instead of just token_ids.
        """
        logits = self._forward_batch(slot_ids, token_ids, kv_lengths)
        results: list[int] = []
        for i, params in enumerate(params_list):
            if params.is_greedy:
                results.append(int(logits[i].argmax(dim=-1).item()))
            else:
                row = logits[i].unsqueeze(0)
                gen = make_generator(params.seed)
                results.append(int(sample_from_logits(row, params, generator=gen).item()))
        if return_logprobs:
            lp_list = [
                compute_logprobs(
                    logits[i].unsqueeze(0),
                    [results[i]],
                    top_k=top_logprobs,
                )[0]
                for i in range(len(results))
            ]
            return results, lp_list
        return results

    def verify_batch(
        self,
        slot_ids: list[int],
        draft_token_ids: list[list[int]],
        kv_lengths: list[int],
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """MTP/speculative-decode verify: submit ``qo_len`` draft tokens
        (K speculative + 1 bonus position) per active slot and run them all
        through ONE real batched forward call. ``draft_token_ids[i]`` is
        slot ``slot_ids[i]``'s own list of draft tokens (same length for
        every slot this step, since ``num_speculative_tokens`` is a global
        engine config). Returns raw logits shaped
        ``[num_reqs * qo_len, vocab]`` (request-then-position order) --
        accept/reject sampling against these logits is the caller's job
        (``determine_accept_reject``/``mtp_verify_and_commit``).
        ``commit=False`` is passed to ``_forward_batch`` unconditionally --
        a verify call's real committed length is never known until
        accept/reject runs on these logits, so ``slot_kv_len`` is
        deliberately NOT advanced here (2026-07-17 fix; see
        ``_forward_batch``'s docstring). Passes this runner's own fixed
        split-KV config (2026-07-17) so the decode/verify kernel gets real
        split-KV parallelism instead of collapsing to ``max_num_splits=1``
        -- see ``_forward_batch``'s docstring."""
        qo_len = len(draft_token_ids[0]) if draft_token_ids else 0
        return self._forward_batch(
            slot_ids,
            draft_token_ids,
            kv_lengths,
            qo_len=qo_len,
            commit=False,
            return_hidden=return_hidden,
            fixed_kv_split_size=self.decode_fixed_kv_split_size,
            fixed_max_num_splits=self.decode_fixed_max_num_splits,
        )

    def verify_batch_spec(
        self,
        slot_ids: list[int],
        draft_token_ids: list[list[int]],
        kv_lengths: list[int],
        *,
        num_accepted_tokens_prev: list[int],
        return_hidden: bool = False,
    ) -> torch.Tensor:
        return self.backend.verify_batch_spec(
            slot_ids,
            draft_token_ids,
            kv_lengths,
            num_accepted_tokens_prev=num_accepted_tokens_prev,
            return_hidden=return_hidden,
        )

    def reset_slot(self, slot: int) -> None:
        return self.gdn_state.reset_slot(slot)

    def snapshot_gdn_state(self, slot: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return self.gdn_state.snapshot_gdn_state(slot)

    def restore_gdn_state(
        self,
        slot: int,
        snapshot: dict[str, tuple[torch.Tensor, torch.Tensor]],
        *,
        allow_cross_slot: bool = False,
    ) -> None:
        return self.gdn_state.restore_gdn_state(
            slot,
            snapshot,
            allow_cross_slot=allow_cross_slot,
        )

    def _mtp_forward(
        self,
        slot: int,
        token_ids: list[int],
        hidden_states_in: torch.Tensor,
        start_pos: int,
        *,
        prior_kv_len: int,
        is_decode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backend._mtp_forward(
            slot,
            token_ids,
            hidden_states_in,
            start_pos,
            prior_kv_len=prior_kv_len,
            is_decode=is_decode,
        )

    def _mtp_sync_and_propose(
        self,
        slot: int,
        shifted_input_ids: list[int],
        target_hidden_states: torch.Tensor,
        start_pos: int,
        num_new_tokens: int,
        k: int,
    ) -> list[int]:
        return self.backend._mtp_sync_and_propose(
            slot,
            shifted_input_ids,
            target_hidden_states,
            start_pos,
            num_new_tokens,
            k,
        )

    def mtp_prefill(self, slot: int, prompt_token_ids: list[int]) -> dict:
        return self.backend.mtp_prefill(slot, prompt_token_ids)

    def mtp_verify_and_commit(self, slot: int, anchor: int, draft_tokens: list[int]) -> dict:
        return self.backend.mtp_verify_and_commit(slot, anchor, draft_tokens)

    def _mtp_forward_batch(
        self,
        slots: list[int],
        token_ids,
        hidden_states_in: torch.Tensor,
        prior_kv_lens: list[int],
        start_pos_list: list[int],
        *,
        qo_len: int | list[int],
        is_decode: bool,
        logits_last_position_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backend._mtp_forward_batch(
            slots,
            token_ids,
            hidden_states_in,
            prior_kv_lens,
            start_pos_list,
            qo_len=qo_len,
            is_decode=is_decode,
            logits_last_position_only=logits_last_position_only,
        )

    def _mtp_run_continuation_steps(
        self,
        slots: list[int],
        draft_tokens: dict[int, list[int]],
        prev_tokens: list[int],
        prev_hidden: torch.Tensor,
        next_pos_list: list[int],
        running_prior_kv_len: list[int],
        k: int,
    ) -> None:
        return self.backend._mtp_run_continuation_steps(
            slots,
            draft_tokens,
            prev_tokens,
            prev_hidden,
            next_pos_list,
            running_prior_kv_len,
            k,
        )

    def _mtp_sync_and_propose_batch(
        self,
        slots: list[int],
        shifted_input_ids_per_slot: list[list[int]],
        target_hidden_states: torch.Tensor,
        start_pos_list: list[int],
        num_new_tokens: int | list[int],
        k: int,
        step0_logits_last_position_only: bool = False,
    ) -> dict[int, list[int]]:
        return self.backend._mtp_sync_and_propose_batch(
            slots,
            shifted_input_ids_per_slot,
            target_hidden_states,
            start_pos_list,
            num_new_tokens,
            k,
            step0_logits_last_position_only,
        )

    def mtp_prefill_batch(
        self,
        slots: list[int],
        prompts_per_slot: list[list[int]],
        chunk_size: int | None = None,
    ) -> dict[int, dict]:
        return self.backend.mtp_prefill_batch(slots, prompts_per_slot, chunk_size)

    def prefill_chunked_begin(
        self,
        slots: list[int],
        prompts_per_slot: list[list[int]],
        chunk_size: int = 512,
    ) -> ChunkedPrefillState:
        """Start an incremental chunked prefill. Returns a state object that
        the engine advances one chunk at a time via ``prefill_chunked_step()``.

        Handles prefix cache reconciliation internally: hit slots get their
        cached prefix restored immediately; the remaining suffix (or full
        cold prompt) is processed incrementally.

        For short prompts (<= chunk_size) or ragged batches, falls back to
        the monolithic ``mtp_prefill_with_cache`` and returns a state with
        ``done=True`` immediately.
        """
        if self.mtp_model is None or self.num_speculative_tokens is None:
            raise RuntimeError("no MTP draft model loaded")
        if len(slots) != len(prompts_per_slot):
            raise ValueError("slots and prompts_per_slot must have equal length")
        if not slots:
            return ChunkedPrefillState(done=True, result={})

        prompt_lens = [len(p) for p in prompts_per_slot]
        is_uniform = len(set(prompt_lens)) == 1

        # Ragged batch or short prompts: monolithic fallback
        if not is_uniform or max(prompt_lens) <= chunk_size:
            result = self.mtp_prefill_with_cache(slots, prompts_per_slot, chunk_size)
            return ChunkedPrefillState(done=True, result=result)

        # Prefix cache reconciliation
        if self.enable_persistent_prefix_cache:
            L_per_slot = [self.reconcile_prefix_hit(p) for p in prompts_per_slot]
        else:
            L_per_slot = [0] * len(slots)

        # D2: record prefix cache hit/miss metrics
        for _L in L_per_slot:
            if _L > 0:
                record_prefix_cache_hit(_L // self.block_size)
            else:
                record_prefix_cache_miss()

        # For hit slots, restore cached prefix immediately
        for s, p, L in zip(slots, prompts_per_slot, L_per_slot):
            if L > 0:
                if self.slot_kv_len[s] != 0 or self.slot_draft_sync_len[s] != 0:
                    raise RuntimeError(f"slot {s} is not fresh")
                self.restore_cached_prefix(s, p, L)

        # Validate cold slots are fresh
        for s, L in zip(slots, L_per_slot):
            if L == 0:
                if self.slot_kv_len[s] != 0 or self.slot_draft_sync_len[s] != 0:
                    raise RuntimeError(f"slot {s} is not fresh")
                self.slot_num_accepted_tokens[s] = 1

        # Suffix = portion of prompt not covered by cache hit
        suffix_per_slot = [p[L:] for p, L in zip(prompts_per_slot, L_per_slot)]
        suffix_lens = [len(sfx) for sfx in suffix_per_slot]
        total_suffix = max(suffix_lens)

        # If suffix fits in one chunk after all, monolithic
        if total_suffix <= chunk_size:
            result = self.mtp_prefill_with_cache(slots, prompts_per_slot, chunk_size)
            return ChunkedPrefillState(done=True, result=result)

        return ChunkedPrefillState(
            done=False,
            result=None,
            slots=slots,
            prompts_per_slot=prompts_per_slot,
            suffix_per_slot=suffix_per_slot,
            suffix_lens=suffix_lens,
            kv_offsets=list(L_per_slot),
            L_per_slot=L_per_slot,
            chunk_size=chunk_size,
            chunk_start=0,
            total_len=total_suffix,
            step0_logits=None,
            step0_hidden=None,
            anchors={},
        )

    def prefill_chunked_step(self, state: ChunkedPrefillState) -> bool:
        """Advance the incremental prefill by ONE chunk. Returns True when
        the prefill is complete (``state.result`` is populated).

        Each call processes exactly one ``chunk_size`` worth of tokens through
        the target model + draft model, then returns control to the engine
        so it can run a decode round for active slots before the next chunk.
        """
        if state.done:
            return True

        slots = state.slots
        prompts_per_slot = state.prompts_per_slot
        suffix_per_slot = state.suffix_per_slot
        chunk_size = state.chunk_size
        chunk_start = state.chunk_start
        total_len = state.total_len
        num_reqs = len(slots)
        k = self.num_speculative_tokens

        chunk_end = min(chunk_start + chunk_size, total_len)
        this_chunk_len = chunk_end - chunk_start
        is_last_chunk = chunk_end >= total_len

        # Build this chunk's tokens per slot
        chunk_tokens_per_slot = [sfx[chunk_start:chunk_end] for sfx in suffix_per_slot]

        # Current kv_len per slot (grows with each chunk)
        running_kv_lens = [self.slot_kv_len[s] for s in slots]

        # Target model forward for this chunk
        target_logits_chunk, target_hidden_chunk = self._forward_batch(
            slots,
            chunk_tokens_per_slot if this_chunk_len > 1 else [t[0] for t in chunk_tokens_per_slot],
            running_kv_lens,
            qo_len=this_chunk_len,
            commit=True,
            return_hidden=True,
            is_decode=False,
            logits_last_position_only=True,
        )

        if is_last_chunk:
            for i, s in enumerate(slots):
                state.anchors[s] = int(target_logits_chunk[i].argmax(dim=-1).item())
            shifted_chunk_per_slot = [
                suffix_per_slot[i][chunk_start + 1 :] + [state.anchors[slots[i]]]
                for i in range(num_reqs)
            ]
        else:
            shifted_chunk_per_slot = [
                suffix_per_slot[i][chunk_start + 1 : chunk_end + 1] for i in range(num_reqs)
            ]

        # Draft model forward for this chunk
        running_draft_lens = [self.slot_draft_sync_len[s] for s in slots]
        draft_logits_chunk, draft_hidden_chunk = self._mtp_forward_batch(
            slots,
            shifted_chunk_per_slot
            if this_chunk_len > 1
            else [t[0] for t in shifted_chunk_per_slot],
            target_hidden_chunk,
            running_draft_lens,
            running_draft_lens,
            qo_len=this_chunk_len,
            is_decode=False,
            logits_last_position_only=True,
        )
        for s in slots:
            self.slot_draft_sync_len[s] += this_chunk_len

        if is_last_chunk:
            state.step0_logits = draft_logits_chunk
            state.step0_hidden = draft_hidden_chunk

        # P3.2 chunk-boundary GDN checkpoints (block-aligned boundaries)
        if self.enable_persistent_prefix_cache and not is_last_chunk:
            abs_kv_end = self.slot_kv_len[slots[0]]
            if abs_kv_end % self.block_size == 0:
                num_chunk_blocks = abs_kv_end // self.block_size
                for i, s in enumerate(slots):
                    self._publish_committed_blocks(s, prompts_per_slot[i], abs_kv_end)
                    self.materialize_gdn_checkpoint(
                        s,
                        key=self.block_table[s][num_chunk_blocks - 1],
                        hash_value=self.slot_block_hashes[s][num_chunk_blocks - 1].value,
                        num_tokens=abs_kv_end,
                    )

        state.chunk_start = chunk_end

        if not is_last_chunk:
            return False

        # === FINALIZE: run draft continuation steps ===
        assert state.step0_logits is not None and state.step0_hidden is not None
        prev_tokens = state.step0_logits.argmax(dim=-1).tolist()
        draft_tokens: dict[int, list[int]] = {s: [prev_tokens[i]] for i, s in enumerate(slots)}
        next_pos_list = [self.slot_draft_sync_len[s] for s in slots]
        running_prior_kv_len = [self.slot_draft_sync_len[s] for s in slots]
        self._mtp_run_continuation_steps(
            slots,
            draft_tokens,
            prev_tokens,
            state.step0_hidden,
            next_pos_list,
            running_prior_kv_len,
            k,
        )
        for s in slots:
            self.slot_pending_draft_tokens[s] = draft_tokens[s]

        # Publish committed blocks for prefix cache
        if self.enable_persistent_prefix_cache:
            for i, s in enumerate(slots):
                self._publish_committed_blocks(s, prompts_per_slot[i], len(prompts_per_slot[i]))

        state.result = {
            s: {"anchor": state.anchors[s], "draft_tokens": draft_tokens[s]} for s in slots
        }
        state.done = True
        return True

    @staticmethod
    def _common_prefix_len(prompts: list[list[int]]) -> int:
        """Longest token prefix shared by EVERY prompt in ``prompts`` (direct
        element-by-element comparison -- cheap for the <=4 same-round requests
        the fixed-slot runtime ever admits at once; ``notes/prefix-cache-design
        .md`` sec 5, "P2 -- Fan-out fork": "detect a common token prefix among
        the same-round admit_now batch by direct comparison")."""
        if not prompts:
            return 0
        first = prompts[0]
        max_len = min(len(p) for p in prompts)
        n = 0
        while n < max_len and all(p[n] == first[n] for p in prompts):
            n += 1
        return n

    def mtp_prefill_fanout_batch(
        self,
        slots: list[int],
        prompts_per_slot: list[list[int]],
        min_shared_prefix_tokens: int | None = None,
    ) -> dict[int, dict]:
        return self.backend.mtp_prefill_fanout_batch(
            slots,
            prompts_per_slot,
            min_shared_prefix_tokens,
        )

    def _publish_committed_blocks(self, slot: int, token_ids: list[int], committed_len: int) -> int:
        return self.prefix_cache._publish_committed_blocks(slot, token_ids, committed_len)

    def publish_committed_decode_blocks(self, slot: int, committed_token_ids: list[int]) -> None:
        return self.prefix_cache.publish_committed_decode_blocks(slot, committed_token_ids)

    def _compute_prompt_block_hashes(
        self, token_ids: list[int], max_tokens: int
    ) -> list[BlockHash]:
        return self.prefix_cache._compute_prompt_block_hashes(token_ids, max_tokens)

    def reconcile_prefix_hit(self, token_ids: list[int]) -> int:
        return self.prefix_cache.reconcile_prefix_hit(token_ids)

    def restore_cached_prefix(self, slot: int, token_ids: list[int], L: int) -> None:
        return self.prefix_cache.restore_cached_prefix(slot, token_ids, L)

    def _prefill_cold_with_populate(self, slot: int, prompt: list[int]) -> dict:
        return self.prefix_cache._prefill_cold_with_populate(slot, prompt)

    def _prefill_hit_with_cache(self, slot: int, prompt: list[int], L: int) -> dict:
        return self.prefix_cache._prefill_hit_with_cache(slot, prompt, L)

    def mtp_prefill_warm_continue(self, slot: int, prompt: list[int], prior_len: int) -> dict:
        return self.backend.mtp_prefill_warm_continue(slot, prompt, prior_len)

    def mtp_prefill_with_cache(
        self,
        slots: list[int],
        prompts_per_slot: list[list[int]],
        chunk_size: int | None = None,
    ) -> dict[int, dict]:
        return self.backend.mtp_prefill_with_cache(slots, prompts_per_slot, chunk_size)

    def mtp_verify_and_commit_batch(
        self,
        slots: list[int],
        anchors: dict[int, int],
        draft_tokens: dict[int, list[int]],
        *,
        return_logprobs: bool = False,
        top_logprobs: int = 0,
    ) -> dict[int, dict]:
        return self.backend.mtp_verify_and_commit_batch(
            slots,
            anchors,
            draft_tokens,
            return_logprobs=return_logprobs,
            top_logprobs=top_logprobs,
        )
