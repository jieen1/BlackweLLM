"""Runtime-owned Laguna config loading with no vLLM/Transformers dependency.

This module owns the minimal config/model interfaces the Laguna runtime and
its DFlash path need while the broader vLLM-removal work is in flight.
Only Laguna-family checkpoints are supported here today.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_ID = "poolside/Laguna-S-2.1-NVFP4"
DEFAULT_ARCHITECTURE = "LagunaForCausalLM"

_REQUIRED_LAGUNA_FIELDS = (
    "architectures",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "layer_types",
    "sliding_window",
)


class LagunaConfigError(RuntimeError):
    """Raised when a Laguna-family config.json cannot be resolved or validated."""


def _candidate_hf_cache_roots() -> list[Path]:
    roots: list[Path] = []

    def add(path: str | None, *, append_hub: bool = False) -> None:
        if not path:
            return
        root = Path(path).expanduser()
        if append_hub:
            root = root / "hub"
        if root not in roots:
            roots.append(root)

    add(os.environ.get("HUGGINGFACE_HUB_CACHE"))
    add(os.environ.get("HF_HUB_CACHE"))
    add(os.environ.get("TRANSFORMERS_CACHE"))
    add(os.environ.get("BF_SHAPES_HF_HOME"), append_hub=True)
    add(os.environ.get("HF_HOME"), append_hub=True)
    add(str(Path.home() / ".cache" / "huggingface" / "hub"))
    return roots


def _choose_snapshot_dir(model_dir: Path) -> Path:
    if (model_dir / "config.json").is_file():
        return model_dir

    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.is_dir():
        raise LagunaConfigError(
            f"no config.json or snapshots/ directory under {model_dir}. "
            "Expected a local checkpoint directory, a snapshot directory, or a "
            "Hugging Face cache model directory."
        )

    refs_dir = model_dir / "refs"
    if refs_dir.is_dir():
        for ref_name in ("main", "master"):
            ref_path = refs_dir / ref_name
            if not ref_path.is_file():
                continue
            snapshot_name = ref_path.read_text(encoding="utf-8").strip()
            if not snapshot_name:
                continue
            snapshot_dir = snapshots_dir / snapshot_name
            if (snapshot_dir / "config.json").is_file():
                return snapshot_dir

    candidates = sorted(
        (path for path in snapshots_dir.iterdir() if (path / "config.json").is_file()),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise LagunaConfigError(
        f"{snapshots_dir} exists but no snapshot contains config.json. "
        "Refusing to guess a checkpoint layout."
    )


def _resolve_local_checkpoint_dir(candidate: str | Path) -> Path:
    path = Path(candidate).expanduser()
    if path.is_file():
        if path.name != "config.json":
            raise LagunaConfigError(
                f"{path} is a file, but not config.json. "
                "Pass a config.json file or a directory containing one."
            )
        return path.parent
    if not path.exists():
        raise LagunaConfigError(f"checkpoint path does not exist: {path}")
    if not path.is_dir():
        raise LagunaConfigError(f"checkpoint path is not a directory: {path}")
    return _choose_snapshot_dir(path)


def resolve_laguna_checkpoint_dir(
    model: str | Path = DEFAULT_MODEL_ID,
    *,
    path_override: str | Path | None = None,
) -> Path:
    """Resolve a local checkpoint/snapshot directory containing ``config.json``.

    Resolution order:
    1. ``path_override`` if provided.
    2. ``model`` when it already points at a local file/directory.
    3. Hugging Face cache lookup by model id across common cache env vars.
    """
    if path_override is not None:
        return _resolve_local_checkpoint_dir(path_override)

    if isinstance(model, Path):
        return _resolve_local_checkpoint_dir(model)

    model_str = str(model)
    local_candidate = Path(model_str).expanduser()
    if local_candidate.exists():
        return _resolve_local_checkpoint_dir(local_candidate)

    cache_suffix = "models--" + model_str.replace("/", "--")
    checked_roots: list[Path] = []
    for cache_root in _candidate_hf_cache_roots():
        checked_roots.append(cache_root)
        model_dir = cache_root / cache_suffix
        if model_dir.exists():
            return _choose_snapshot_dir(model_dir)

    checked = ", ".join(str(root) for root in checked_roots)
    raise LagunaConfigError(
        f"cannot resolve Laguna checkpoint for model={model_str!r}. "
        f"Checked local path and HF cache roots: {checked}."
    )


def resolve_laguna_config_path(
    model: str | Path = DEFAULT_MODEL_ID,
    *,
    path_override: str | Path | None = None,
) -> Path:
    """Return the resolved ``config.json`` path for a Laguna-family checkpoint."""
    return resolve_laguna_checkpoint_dir(model, path_override=path_override) / "config.json"


class LagunaConfig:
    """Minimal attribute-style Laguna config object loaded from ``config.json``.

    The runtime only needs a small attribute surface from the original
    Transformers/vLLM config object. This class preserves that shape without
    pulling either dependency into the import path.
    """

    model_type = "laguna"

    def __init__(self, **raw: Any) -> None:
        data = copy.deepcopy(raw)
        data.setdefault("model_type", self.model_type)
        data.setdefault("architectures", [DEFAULT_ARCHITECTURE])
        self._validate(data)
        super().__setattr__("_data", data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        self._data[name] = copy.deepcopy(value)

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            raise AttributeError(f"cannot delete internal attribute {name}")
        try:
            del self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __repr__(self) -> str:
        architecture = self.architectures[0] if self.architectures else "unknown"
        return (
            "LagunaConfig("
            f"architecture={architecture!r}, "
            f"num_hidden_layers={self.num_hidden_layers}, "
            f"hidden_size={self.hidden_size})"
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LagunaConfig":
        if not isinstance(raw, dict):
            raise LagunaConfigError("Laguna config must deserialize from a JSON object")
        return cls(**raw)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "LagunaConfig":
        config_path = Path(path).expanduser()
        try:
            with config_path.open(encoding="utf-8") as config_file:
                raw = json.load(config_file)
        except OSError as exc:
            raise LagunaConfigError(f"failed to read {config_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LagunaConfigError(f"failed to parse {config_path} as JSON: {exc}") from exc
        return cls.from_dict(raw)

    @staticmethod
    def _validate(raw: dict[str, Any]) -> None:
        for key in _REQUIRED_LAGUNA_FIELDS:
            if raw.get(key) is None:
                raise LagunaConfigError(f"Laguna config is missing required field {key!r}")

        architectures = raw["architectures"]
        if not isinstance(architectures, list) or not architectures:
            raise LagunaConfigError("Laguna config must declare at least one architecture")

        try:
            num_hidden_layers = int(raw["num_hidden_layers"])
        except (TypeError, ValueError) as exc:
            raise LagunaConfigError("Laguna config num_hidden_layers must be an integer") from exc

        layer_types = raw["layer_types"]
        if not isinstance(layer_types, list):
            raise LagunaConfigError("Laguna config layer_types must be a list")
        if len(layer_types) != num_hidden_layers:
            raise LagunaConfigError(
                "Laguna config layer_types length must match num_hidden_layers"
            )

        heads_per_layer = raw.get("num_attention_heads_per_layer")
        if heads_per_layer is not None and len(heads_per_layer) != num_hidden_layers:
            raise LagunaConfigError(
                "Laguna config num_attention_heads_per_layer length must match "
                "num_hidden_layers"
            )


def build_laguna_config(
    model: str | Path = DEFAULT_MODEL_ID,
    *,
    path_override: str | Path | None = None,
) -> LagunaConfig:
    """Load a Laguna-family ``config.json`` from a local path or HF cache."""
    return LagunaConfig.from_json_file(resolve_laguna_config_path(model, path_override=path_override))


@dataclass
class SelfBuiltModelConfig:
    """Minimal runtime-owned replacement for the ModelConfig surface Laguna uses."""

    model: str
    hf_config: LagunaConfig
    runner: str = "generate"
    tokenizer: str | None = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = True
    dtype: str = "bfloat16"
    seed: int = 0
    max_model_len: int | None = None
    spec_target_max_model_len: int | None = None
    enforce_eager: bool = False

    def get_hidden_size(self) -> int:
        return int(self.hf_config.hidden_size)

    def get_num_hidden_layers(self) -> int:
        return int(self.hf_config.num_hidden_layers)

    def get_num_attention_heads(self) -> int:
        return int(self.hf_config.num_attention_heads)


def build_selfbuilt_model_config(
    *,
    model: str | Path,
    hf_config: LagunaConfig | None = None,
    path_override: str | Path | None = None,
    runner: str = "generate",
    tokenizer: str | None = None,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = True,
    dtype: str = "bfloat16",
    seed: int = 0,
    max_model_len: int | None = None,
    spec_target_max_model_len: int | None = None,
    enforce_eager: bool = False,
) -> SelfBuiltModelConfig:
    """Build the runtime-owned model config from a Laguna-family checkpoint."""
    resolved_hf_config = hf_config or build_laguna_config(model, path_override=path_override)
    model_str = str(model)
    return SelfBuiltModelConfig(
        model=model_str,
        hf_config=resolved_hf_config,
        runner=runner,
        tokenizer=tokenizer or model_str,
        tokenizer_mode=tokenizer_mode,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        seed=seed,
        max_model_len=max_model_len or resolved_hf_config.get("max_position_embeddings"),
        spec_target_max_model_len=spec_target_max_model_len,
        enforce_eager=enforce_eager,
    )


@dataclass
class SelfBuiltSpeculativeConfig:
    """Minimal runtime-owned replacement for the SpeculativeConfig surface."""

    model: str
    method: str
    num_speculative_tokens: int
    target_model_config: Any
    target_parallel_config: Any
    draft_model_config: SelfBuiltModelConfig | None = None


def build_selfbuilt_speculative_config(
    *,
    model: str | Path,
    method: str,
    num_speculative_tokens: int,
    target_model_config: Any,
    target_parallel_config: Any,
    draft_model_config: SelfBuiltModelConfig | None = None,
) -> SelfBuiltSpeculativeConfig:
    return SelfBuiltSpeculativeConfig(
        model=str(model),
        method=method,
        num_speculative_tokens=num_speculative_tokens,
        target_model_config=target_model_config,
        target_parallel_config=target_parallel_config,
        draft_model_config=draft_model_config,
    )


def replace(instance: Any, /, **changes: Any) -> Any:
    """Shallow copy ``instance`` with selected fields replaced.

    Mirrors the narrow behavior this repo currently relies on from
    ``vllm.config.replace`` while staying generic enough for other runtime-owned
    config dataclasses.
    """
    if is_dataclass(instance) and not isinstance(instance, type):
        names = {field.name for field in fields(instance)}
        unknown = sorted(set(changes) - names)
        if unknown:
            raise TypeError(
                f"cannot replace unknown field(s) {unknown} on {type(instance).__name__}"
            )
        values = {field.name: getattr(instance, field.name) for field in fields(instance)}
        values.update(changes)
        return type(instance)(**values)

    clone = copy.copy(instance)
    for name, value in changes.items():
        setattr(clone, name, value)
    return clone
