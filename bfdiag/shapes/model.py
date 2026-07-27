"""bfdiag.shapes.model -- parse Laguna's *real* config.json into structural params.

The whole point of ``bfdiag.shapes`` is to stop people from hand-typing
``num_heads=48, head_dim=128, ...`` into kernel isolation tests -- that is
exactly how the 2026-07-27 shape bug happened (see
``notes/2026-07-27-bfdiag-shape-derivation.md``). So this module never
hardcodes an architecture default: every structural number below is read
from the model's own ``config.json`` (the HF snapshot actually present on
this machine). If a required field or file is missing, :class:`LagunaConfigError`
is raised with an actionable message -- silently falling back to a hardcoded
number would defeat the entire purpose of this package.

Two real checkpoints are parsed:

- the target model (``poolside/Laguna-S-2.1-NVFP4``): 48 layers, 12
  ``full_attention`` + 36 ``sliding_attention`` (window=512), MoE from layer 1
  onward (layer 0 is dense).
- the DFlash draft model (``poolside/Laguna-S-2.1-DFlash-NVFP4``): 6 dense
  ``sliding_attention`` layers (window=512), no MoE.

Layer grouping (full vs sliding, and the per-layer head count) is computed
from ``layer_types`` + ``sliding_window`` + ``num_attention_heads_per_layer``
exactly the way ``runtime/backends/laguna.py`` (``LagunaBackend.__init__``,
the ``_layer_groups`` construction) does it -- this module does not import
that code, it independently re-derives the same grouping from the same
config fields so ``bf shapes`` and the runtime can never silently drift
apart without a test noticing.

Real-shape correction (do not "fix" this back to a uniform head count):
``runtime/backends/laguna.py``'s own docstring used to say "24 Q heads / 8 KV
heads" for every layer. Reading the actual safetensors weight shapes (not the
config field ``num_attention_heads`` alone) shows this is wrong -- see
``notes/2026-07-27-laguna-real-shapes-correction-and-page-size-migration-plan.md``.
The real per-layer split, confirmed against ``model.safetensors`` tensor
shapes on this machine:

- ``full_attention`` layers (12 of them, indices 0,4,8,...,44):
  ``q_proj.weight = [6144, 3072]`` = 48 heads * 128 head_dim.
- ``sliding_attention`` layers (36 of them): ``q_proj.weight = [9216, 3072]``
  = 72 heads * 128 head_dim.
- both groups share ``k_proj/v_proj.weight = [1024, 3072]`` = 8 KV heads *
  128 head_dim (GQA group size 6 for full, 9 for sliding).

Config already carries this as ``num_attention_heads_per_layer`` (a
per-layer list) -- this module reads that list rather than re-deriving it
from weight shapes, but the invariant test in
``tests/test_bfdiag_shapes_model.py`` cross-checks both.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "poolside/Laguna-S-2.1-NVFP4"
DEFAULT_DRAFT_MODEL_ID = "poolside/Laguna-S-2.1-DFlash-NVFP4"

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"


class LagunaConfigError(RuntimeError):
    """Raised when a real config.json can't be found or is missing a required field.

    This is the error the whole ``bfdiag.shapes`` package is built to force:
    if we can't read the real structural parameters, we refuse to guess --
    the caller gets pointed at exactly what's missing and how to fix it.
    """


def cdiv(a: int, b: int) -> int:
    """Ceiling division for positive integers (matches the ``-(-a // b)`` idiom
    used throughout ``runtime/backends/laguna.py``)."""
    if b <= 0:
        raise ValueError(f"cdiv: block size must be positive, got {b}")
    return -(-a // b)


def _hf_cache_root() -> Path:
    override = os.environ.get("BF_SHAPES_HF_HOME") or os.environ.get("HF_HOME")
    if override:
        return Path(override).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def find_snapshot_dir(model_id: str, *, path_override: str | Path | None = None) -> Path:
    """Resolve the HF cache snapshot directory holding ``config.json`` for ``model_id``.

    ``path_override`` may point directly at a snapshot directory or at a
    ``config.json`` file (its parent is used); this is how tests point at a
    synthetic fixture instead of the real HF cache.
    """
    if path_override is not None:
        p = Path(path_override).expanduser()
        if p.is_file():
            p = p.parent
        if not (p / "config.json").exists():
            raise LagunaConfigError(
                f"no config.json under override path {p} for model_id={model_id!r}. "
                "path_override must point at a directory containing config.json "
                "(or at the config.json file itself)."
            )
        return p

    cache_root = _hf_cache_root()
    model_dir = cache_root / ("models--" + model_id.replace("/", "--"))
    snapshots = model_dir / "snapshots"
    if not snapshots.is_dir():
        raise LagunaConfigError(
            f"cannot find a HF cache snapshot for {model_id!r} under {cache_root}. "
            f"Expected {snapshots}. Either download the model "
            f"(huggingface-cli download {model_id}) or pass path_override=... "
            "pointing at a directory with config.json. Refusing to fall back to "
            "hardcoded architecture defaults -- that is exactly the bug this "
            "package exists to prevent."
        )
    candidates = sorted(d for d in snapshots.iterdir() if (d / "config.json").exists())
    if not candidates:
        raise LagunaConfigError(
            f"{snapshots} exists but no snapshot subdirectory has a config.json "
            f"for {model_id!r}. Refusing to fall back to hardcoded defaults."
        )
    return candidates[0]


def load_json_config(model_id: str, *, path_override: str | Path | None = None) -> dict[str, Any]:
    """Load and parse ``config.json`` for ``model_id``. Raises :class:`LagunaConfigError`
    on any I/O or parse failure -- never returns a partial/default dict."""
    snapshot_dir = find_snapshot_dir(model_id, path_override=path_override)
    config_path = snapshot_dir / "config.json"
    try:
        with open(config_path) as f:
            return json.load(f)
    except OSError as exc:
        raise LagunaConfigError(f"failed to read {config_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LagunaConfigError(f"failed to parse {config_path} as JSON: {exc}") from exc


def _require(config: dict[str, Any], key: str, model_id: str) -> Any:
    if key not in config or config[key] is None:
        raise LagunaConfigError(
            f"config.json for {model_id!r} is missing required field {key!r}. "
            "Refusing to substitute a hardcoded default for a real structural "
            "parameter -- pass an explicit override or fix the checkpoint."
        )
    return config[key]


@dataclass(frozen=True)
class LayerGroup:
    """One (attention pattern) group of layers sharing head counts + window.

    ``kind`` is ``"full"`` or ``"sliding"``. ``window`` is ``None`` for full
    attention, else the sliding-window size in tokens (raw config value,
    e.g. 512 -- *not* pre-decremented; ``_ring_blocks_for_window`` subtracts
    the 1 itself, see ``bfdiag.shapes.attention``).
    """

    kind: str
    window: int | None
    layer_indices: tuple[int, ...]
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class LagunaModelConfig:
    """Structural parameters of the target Laguna model, parsed from its real config.json."""

    model_id: str
    config_path: Path

    hidden_size: int
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int

    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    nvfp4_group_size: int | None

    sliding_window: int
    layer_types: tuple[str, ...]
    heads_per_layer: tuple[int, ...]
    heads_per_layer_source: str  # "config_per_layer" | "config_uniform"

    dense_mlp_layer_indices: tuple[int, ...]
    moe_layer_indices: tuple[int, ...]

    kv_cache_dtype: str  # "fp8_e4m3" | "bf16" (from quantization_config.kv_cache_scheme)

    groups: dict[str, LayerGroup] = field(default_factory=dict)

    @property
    def full_layer_indices(self) -> tuple[int, ...]:
        return self.groups["full"].layer_indices

    @property
    def sliding_layer_indices(self) -> tuple[int, ...]:
        return self.groups["sliding"].layer_indices


@dataclass(frozen=True)
class DraftModelConfig:
    """Structural parameters of the DFlash draft model, parsed from its real config.json."""

    model_id: str
    config_path: Path

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    sliding_window: int
    eagle_aux_hidden_state_layer_ids: tuple[int, ...]
    dflash_verify_block_size: int | None
    """``dflash_config.block_size`` in the checkpoint -- a DFlash verify-chunk
    granularity (matches NUM_QUERY_PER_REQ=16 in practice), *not* the KV
    page_size this package's ``block_size`` parameter refers to. Kept
    separate on purpose to avoid the two "block_size"s getting conflated."""


def _kv_cache_dtype_from_config(config: dict[str, Any]) -> str:
    quant = config.get("quantization_config") or {}
    kv_scheme = quant.get("kv_cache_scheme") or {}
    num_bits = kv_scheme.get("num_bits")
    if num_bits == 8:
        return "fp8_e4m3"
    torch_dtype = config.get("torch_dtype", "bfloat16")
    return "bf16" if "bfloat16" in str(torch_dtype) else str(torch_dtype)


def _nvfp4_group_size_from_config(config: dict[str, Any]) -> int | None:
    quant = config.get("quantization_config") or {}
    groups = quant.get("config_groups") or {}
    for group in groups.values():
        weights = group.get("weights") or {}
        gs = weights.get("group_size")
        if gs is not None:
            return int(gs)
    return None


def load_laguna_config(
    model_id: str = DEFAULT_MODEL_ID, *, path_override: str | Path | None = None
) -> LagunaModelConfig:
    """Parse the target Laguna model's real config.json into :class:`LagunaModelConfig`.

    Layer grouping mirrors ``runtime/backends/laguna.py``'s ``_layer_groups``
    construction (group key = (window_left, num_qo_heads, num_kv_heads)):
    read here from ``layer_types[i]`` + ``sliding_window`` +
    ``num_attention_heads_per_layer[i]`` (falling back to the uniform
    ``num_attention_heads`` field if the per-layer list isn't present in the
    config -- some non-Laguna configs won't have it).
    """
    snapshot_dir = find_snapshot_dir(model_id, path_override=path_override)
    config = load_json_config(model_id, path_override=path_override)

    num_hidden_layers = _require(config, "num_hidden_layers", model_id)
    layer_types_raw = _require(config, "layer_types", model_id)
    if len(layer_types_raw) != num_hidden_layers:
        raise LagunaConfigError(
            f"{model_id}: len(layer_types)={len(layer_types_raw)} != "
            f"num_hidden_layers={num_hidden_layers}"
        )
    layer_types = tuple(layer_types_raw)

    sliding_window = _require(config, "sliding_window", model_id)
    num_key_value_heads = _require(config, "num_key_value_heads", model_id)
    head_dim = _require(config, "head_dim", model_id)

    heads_per_layer_raw = config.get("num_attention_heads_per_layer")
    if heads_per_layer_raw is not None:
        if len(heads_per_layer_raw) != num_hidden_layers:
            raise LagunaConfigError(
                f"{model_id}: len(num_attention_heads_per_layer)="
                f"{len(heads_per_layer_raw)} != num_hidden_layers={num_hidden_layers}"
            )
        heads_per_layer = tuple(heads_per_layer_raw)
        heads_per_layer_source = "config_per_layer"
    else:
        uniform_heads = _require(config, "num_attention_heads", model_id)
        heads_per_layer = tuple(uniform_heads for _ in range(num_hidden_layers))
        heads_per_layer_source = "config_uniform"

    full_indices: list[int] = []
    sliding_indices: list[int] = []
    full_heads: set[int] = set()
    sliding_heads: set[int] = set()
    for i, lt in enumerate(layer_types):
        if lt == FULL_ATTENTION:
            full_indices.append(i)
            full_heads.add(heads_per_layer[i])
        elif lt == SLIDING_ATTENTION:
            sliding_indices.append(i)
            sliding_heads.add(heads_per_layer[i])
        else:
            raise LagunaConfigError(f"{model_id}: unrecognized layer_types[{i}]={lt!r}")

    if len(full_heads) > 1:
        raise LagunaConfigError(
            f"{model_id}: full_attention layers have inconsistent head counts {full_heads}; "
            "bfdiag.shapes assumes one head count per attention pattern (matching "
            "runtime/backends/laguna.py's group-by-(window,heads) logic) -- extend "
            "LayerGroup handling if this model genuinely varies within a pattern."
        )
    if len(sliding_heads) > 1:
        raise LagunaConfigError(
            f"{model_id}: sliding_attention layers have inconsistent head counts "
            f"{sliding_heads}; see full_attention error above for why this is a hard error."
        )

    groups: dict[str, LayerGroup] = {}
    if full_indices:
        groups["full"] = LayerGroup(
            kind="full",
            window=None,
            layer_indices=tuple(full_indices),
            num_qo_heads=next(iter(full_heads)),
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
        )
    if sliding_indices:
        groups["sliding"] = LayerGroup(
            kind="sliding",
            window=sliding_window,
            layer_indices=tuple(sliding_indices),
            num_qo_heads=next(iter(sliding_heads)),
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
        )

    mlp_only_layers = tuple(config.get("mlp_only_layers") or ())
    moe_layer_indices = tuple(i for i in range(num_hidden_layers) if i not in mlp_only_layers)

    num_experts = config.get("num_experts", 0) or 0

    return LagunaModelConfig(
        model_id=model_id,
        config_path=snapshot_dir / "config.json",
        hidden_size=_require(config, "hidden_size", model_id),
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=_require(config, "intermediate_size", model_id),
        vocab_size=_require(config, "vocab_size", model_id),
        num_experts=num_experts,
        num_experts_per_tok=config.get("num_experts_per_tok", 0) or 0,
        moe_intermediate_size=config.get("moe_intermediate_size", 0) or 0,
        shared_expert_intermediate_size=config.get("shared_expert_intermediate_size", 0) or 0,
        nvfp4_group_size=_nvfp4_group_size_from_config(config),
        sliding_window=sliding_window,
        layer_types=layer_types,
        heads_per_layer=heads_per_layer,
        heads_per_layer_source=heads_per_layer_source,
        dense_mlp_layer_indices=mlp_only_layers,
        moe_layer_indices=moe_layer_indices if num_experts else (),
        kv_cache_dtype=_kv_cache_dtype_from_config(config),
        groups=groups,
    )


def load_draft_config(
    model_id: str = DEFAULT_DRAFT_MODEL_ID, *, path_override: str | Path | None = None
) -> DraftModelConfig:
    """Parse the DFlash draft model's real config.json into :class:`DraftModelConfig`."""
    snapshot_dir = find_snapshot_dir(model_id, path_override=path_override)
    config = load_json_config(model_id, path_override=path_override)

    sliding_window = config.get("sliding_window")
    if sliding_window is None:
        windows = config.get("sliding_windows")
        if windows:
            sliding_window = windows[0]
    if sliding_window is None:
        raise LagunaConfigError(
            f"{model_id}: neither 'sliding_window' nor 'sliding_windows' present in config.json"
        )

    dflash_cfg = config.get("dflash_config") or {}

    return DraftModelConfig(
        model_id=model_id,
        config_path=snapshot_dir / "config.json",
        hidden_size=_require(config, "hidden_size", model_id),
        num_hidden_layers=_require(config, "num_hidden_layers", model_id),
        num_attention_heads=_require(config, "num_attention_heads", model_id),
        num_key_value_heads=_require(config, "num_key_value_heads", model_id),
        head_dim=_require(config, "head_dim", model_id),
        intermediate_size=_require(config, "intermediate_size", model_id),
        vocab_size=_require(config, "vocab_size", model_id),
        sliding_window=sliding_window,
        eagle_aux_hidden_state_layer_ids=tuple(
            config.get("eagle_aux_hidden_state_layer_ids") or ()
        ),
        dflash_verify_block_size=dflash_cfg.get("block_size"),
    )
