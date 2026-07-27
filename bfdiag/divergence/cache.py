"""Tiered on-disk cache for oracle (vLLM) per-layer activations.

Why a tiered cache, not a raw dump
-----------------------------------
A raw per-layer activation dump does not fit on disk at realistic context
lengths. For Laguna-S-2.1 (hidden_size=3072, see
``runtime/backends/laguna_sparkinfer_moe.py``'s ``HIDDEN_SIZE``), one
``[seq_len, hidden]`` fp32 tensor at a 64K-token context is
``64_000 * 3072 * 4 bytes ~= 786 MB``. A full scan hooks ~4-5 submodules per
layer across 48 layers -- a raw dump would be on the order of **150+ GB for
one prompt**. That is the problem this cache exists to avoid; see
notes/2026-07-27-bfdiag-oracle-divergence.md for the full size table across
context lengths and tiers.

The fix is to store three independent tiers per (layer, submodule):

1. **Stats (always)** -- ``SubmoduleStats`` (mean/std/absmax/L2/min/max) plus
   9 quantiles and a per-dimension mean vector. Cost is ``O(dim)``, *not*
   ``O(seq_len)`` -- this tier's size is identical at 128 tokens and at 64K
   tokens. Always cheap, always written, always available even when nothing
   else was captured.
2. **Sampled tokens (default)** -- full-precision vectors for the first K,
   last K, and R randomly-sampled token positions (``CaptureConfig``, default
   K=8, R=8). Size is ``O((2K+R) * dim)``, independent of context length.
   This is the tier the divergence scanner actually diffs against by
   default: it is high-fidelity at the sampled positions without paying for
   every position.
3. **Full (opt-in, ``--full``)** -- the entire ``[seq_len, dim]`` tensor.
   Scales with context length; only sane for small fixtures.

Cache key = ``(model_revision, prompt_hash, layer_set, capture_config)``
(``CacheKey``). Storage layout: ``${QSR_ORACLE_CACHE}`` if set, else
``${QSR_BFDIAG_DIR:-<repo>/.bfdiag}/oracle_cache/<prompt_hash>/`` holding one
``<config_hash>.safetensors`` + ``<config_hash>.manifest.json`` pair per
distinct ``(model_revision, layer_set, capture_config)`` under that prompt --
so several capture configs for the same prompt coexist without collisions,
while the literal "one directory per prompt" layout from the shared bfdiag
storage contract is preserved.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, sqrt
from pathlib import Path
from typing import Any

from bfdiag.divergence.scan import ActivationTrace

ORACLE_CACHE_ENV = "QSR_ORACLE_CACHE"
BFDIAG_DIR_ENV = "QSR_BFDIAG_DIR"

_QUANTILE_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _repo_root() -> Path:
    # bfdiag/divergence/cache.py -> bfdiag/divergence -> bfdiag -> <repo root>
    return Path(__file__).resolve().parents[2]


def oracle_cache_root() -> Path:
    """Resolve the oracle cache root directory from the environment.

    ``QSR_ORACLE_CACHE`` (this module's own override) takes precedence;
    otherwise falls back to the shared bfdiag storage contract:
    ``${QSR_BFDIAG_DIR:-<repo>/.bfdiag}/oracle_cache``.
    """
    override = os.environ.get(ORACLE_CACHE_ENV)
    if override:
        return Path(override)
    bfdiag_dir = os.environ.get(BFDIAG_DIR_ENV)
    base = Path(bfdiag_dir) if bfdiag_dir else _repo_root() / ".bfdiag"
    return base / "oracle_cache"


# ---------------------------------------------------------------------------
# Pure reduction logic (no file I/O, no optional dependencies): fully
# CPU-testable, see tests/test_bfdiag_oracle_cache.py.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureConfig:
    """Sampling knobs for the sampled-token tier. Part of the cache key."""

    k_edge_tokens: int = 8
    r_random_tokens: int = 8
    full: bool = False
    random_seed: int = 0

    def as_key_dict(self) -> dict[str, Any]:
        return {
            "k_edge_tokens": self.k_edge_tokens,
            "r_random_tokens": self.r_random_tokens,
            "full": self.full,
            "random_seed": self.random_seed,
        }


@dataclass(frozen=True)
class CacheKey:
    """Cache key: ``(model_revision, prompt_hash, layer_set, capture_config)``."""

    model_revision: str
    prompt_hash: str
    layer_set: tuple[int, ...] | str  # explicit layer indices, or "all"
    capture_config: CaptureConfig

    def config_hash(self) -> str:
        layer_set = self.layer_set if isinstance(self.layer_set, str) else list(self.layer_set)
        payload = {
            "model_revision": self.model_revision,
            "layer_set": layer_set,
            "capture_config": self.capture_config.as_key_dict(),
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def compute_prompt_hash(prompt_token_ids: Sequence[int]) -> str:
    """Stable hash of a tokenized prompt, used as the cache directory name."""
    blob = json.dumps(list(prompt_token_ids)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class SubmoduleStats:
    """Always-on statistical summary of one captured [tokens, dim] tensor."""

    num_tokens: int
    dim: int
    mean: float
    std: float
    absmax: float
    l2: float
    min: float
    max: float
    quantiles: tuple[float, ...]


def _as_matrix(value: Any) -> list[list[float]]:
    """Duck-type a ``[tokens, dim]`` (or flat ``[dim]``) tensor-like value
    into a nested list of plain floats. Accepts CPU torch/numpy tensors or
    plain nested lists -- never touches a GPU (callers are responsible for
    only ever passing CPU-resident data, exactly like
    ``oracle.comparator._as_values``)."""
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    rows = list(value)
    if rows and not isinstance(rows[0], list | tuple):
        rows = [rows]  # a flat 1D activation is treated as a single "token"
    return [[float(item) for item in row] for row in rows]


def _quantiles(sorted_values: Sequence[float], levels: Sequence[float]) -> tuple[float, ...]:
    n = len(sorted_values)
    if n == 1:
        return tuple(sorted_values[0] for _ in levels)
    result = []
    for level in levels:
        pos = level * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        result.append(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)
    return tuple(result)


def compute_stats(matrix: Sequence[Sequence[float]]) -> tuple[SubmoduleStats, tuple[float, ...]]:
    """Reduce a ``[tokens, dim]`` activation to scalar stats + a per-dim mean
    vector. Cost and storage are ``O(dim)`` once tokens are summed away --
    the reason the stats tier stays tiny even at a 64K-token context."""
    num_tokens = len(matrix)
    dim = len(matrix[0]) if num_tokens else 0
    flat = [value for row in matrix for value in row]
    count = len(flat)
    if count == 0:
        raise ValueError("cannot compute stats for an empty activation")
    mean = fsum(flat) / count
    variance = fsum((value - mean) ** 2 for value in flat) / count
    std = sqrt(variance)
    absmax = max(abs(value) for value in flat)
    l2 = sqrt(fsum(value * value for value in flat))
    minimum = min(flat)
    maximum = max(flat)
    quantiles = _quantiles(sorted(flat), _QUANTILE_LEVELS)
    dim_mean = tuple(
        fsum(matrix[t][d] for t in range(num_tokens)) / num_tokens for d in range(dim)
    )
    stats = SubmoduleStats(num_tokens, dim, mean, std, absmax, l2, minimum, maximum, quantiles)
    return stats, dim_mean


def select_sample_positions(
    num_tokens: int, *, k_edge: int, r_random: int, seed: int = 0
) -> tuple[int, ...]:
    """First ``k_edge`` + last ``k_edge`` + ``r_random`` deterministic random
    positions from the middle, deduplicated and sorted ascending."""
    if num_tokens <= 0:
        return ()
    head = set(range(min(k_edge, num_tokens)))
    tail = set(range(max(0, num_tokens - k_edge), num_tokens))
    middle_candidates = sorted(set(range(num_tokens)) - head - tail)
    positions = head | tail
    if r_random > 0 and middle_candidates:
        rng = random.Random(seed)
        positions |= set(rng.sample(middle_candidates, min(r_random, len(middle_candidates))))
    return tuple(sorted(positions))


@dataclass(frozen=True)
class CachedSubmodule:
    """Everything cached for one (layer, submodule): stats always, sampled
    tokens by default, full tensor only with ``CaptureConfig.full=True``."""

    stats: SubmoduleStats
    dim_mean: tuple[float, ...]
    sample_positions: tuple[int, ...]
    sample_tokens: tuple[tuple[float, ...], ...]
    full: tuple[tuple[float, ...], ...] | None

    def to_scan_vector(self) -> list[float]:
        """Flatten this cache entry into the single comparable vector
        ``scan.scan_layers`` expects for one (layer, submodule): the full
        tensor if captured, else the sampled tokens (in position order),
        else the per-dim mean as a last resort."""
        if self.full is not None:
            source: Sequence[Sequence[float]] = self.full
        elif self.sample_tokens:
            source = self.sample_tokens
        else:
            source = (self.dim_mean,)
        return [value for row in source for value in row]


def build_cache_entries(
    config: CaptureConfig, trace: ActivationTrace
) -> dict[int, dict[str, CachedSubmodule]]:
    """Reduce a full ``ActivationTrace`` into the tiered, cache-ready shape.
    Pure -- no file I/O, no optional dependencies."""
    entries: dict[int, dict[str, CachedSubmodule]] = {}
    for layer_idx, submodules in trace.items():
        layer_entries: dict[str, CachedSubmodule] = {}
        for name, tensor in submodules.items():
            matrix = _as_matrix(tensor)
            stats, dim_mean = compute_stats(matrix)
            positions = select_sample_positions(
                len(matrix),
                k_edge=config.k_edge_tokens,
                r_random=config.r_random_tokens,
                seed=config.random_seed,
            )
            sample_tokens = tuple(tuple(matrix[position]) for position in positions)
            full = tuple(tuple(row) for row in matrix) if config.full else None
            layer_entries[name] = CachedSubmodule(
                stats=stats,
                dim_mean=dim_mean,
                sample_positions=positions,
                sample_tokens=sample_tokens,
                full=full,
            )
        entries[layer_idx] = layer_entries
    return entries


def to_activation_trace(
    entries: Mapping[int, Mapping[str, CachedSubmodule]],
) -> dict[int, dict[str, list[float]]]:
    """Flatten cached submodules into the plain vector-per-submodule shape
    ``scan.scan_layers`` compares directly."""
    return {
        layer_idx: {name: entry.to_scan_vector() for name, entry in submodules.items()}
        for layer_idx, submodules in entries.items()
    }


# ---------------------------------------------------------------------------
# Disk I/O: safetensors + JSON manifest. Imports numpy/safetensors lazily so
# importing this module (and using the pure functions above) never requires
# them -- matching oracle/capture_hooks.py's own lazy-import convention.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheLookup:
    """Clear hit/miss result for a cache read or write, with a human message."""

    hit: bool
    path: Path
    message: str


def _entry_paths(key: CacheKey, *, root: Path | None = None) -> tuple[Path, Path]:
    base = (root or oracle_cache_root()) / key.prompt_hash
    config_hash = key.config_hash()
    return base / f"{config_hash}.safetensors", base / f"{config_hash}.manifest.json"


def write_oracle_cache(
    key: CacheKey, trace: ActivationTrace, *, root: Path | None = None
) -> CacheLookup:
    """Persist a layered activation capture (stats + samples + optional full)."""
    import numpy as np
    from safetensors.numpy import save_file

    entries = build_cache_entries(key.capture_config, trace)
    tensors: dict[str, Any] = {}
    tensor_manifest: dict[str, dict[str, Any]] = {}
    for layer_idx, submodules in entries.items():
        for name, entry in submodules.items():
            tensor_key = f"L{layer_idx:03d}.{name}"
            tensors[f"{tensor_key}.stats"] = np.array(
                [
                    entry.stats.mean,
                    entry.stats.std,
                    entry.stats.absmax,
                    entry.stats.l2,
                    entry.stats.min,
                    entry.stats.max,
                ],
                dtype=np.float32,
            )
            tensors[f"{tensor_key}.quantiles"] = np.array(entry.stats.quantiles, dtype=np.float32)
            tensors[f"{tensor_key}.dim_mean"] = np.array(entry.dim_mean, dtype=np.float32)
            tensors[f"{tensor_key}.sample_positions"] = np.array(
                entry.sample_positions, dtype=np.int64
            )
            if entry.sample_tokens:
                tensors[f"{tensor_key}.sample_tokens"] = np.array(
                    entry.sample_tokens, dtype=np.float32
                )
            if entry.full is not None:
                tensors[f"{tensor_key}.full"] = np.array(entry.full, dtype=np.float32)
            tensor_manifest[tensor_key] = {
                "layer_idx": layer_idx,
                "submodule": name,
                "num_tokens": entry.stats.num_tokens,
                "dim": entry.stats.dim,
                "has_full": entry.full is not None,
            }

    safetensors_path, manifest_path = _entry_paths(key, root=root)
    safetensors_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(safetensors_path))
    manifest = {
        "model_revision": key.model_revision,
        "prompt_hash": key.prompt_hash,
        "layer_set": key.layer_set if isinstance(key.layer_set, str) else list(key.layer_set),
        "capture_config": key.capture_config.as_key_dict(),
        "config_hash": key.config_hash(),
        "tensors": tensor_manifest,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return CacheLookup(
        hit=False, path=safetensors_path, message=f"wrote oracle cache: {safetensors_path}"
    )


def read_oracle_cache(
    key: CacheKey, *, root: Path | None = None
) -> tuple[dict[int, dict[str, CachedSubmodule]], CacheLookup] | None:
    """Read back a cache entry, or ``None`` on a clean miss.

    A miss covers both "nothing on disk yet" and "something is on disk but
    its manifest doesn't match this key's ``capture_config``/``layer_set``"
    (a stale/incompatible cache from a previous run) -- both are treated the
    same way: recapture, don't guess.
    """
    safetensors_path, manifest_path = _entry_paths(key, root=root)
    if not safetensors_path.exists() or not manifest_path.exists():
        return None

    from safetensors.numpy import load_file

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("config_hash") != key.config_hash():
        return None

    tensors = load_file(str(safetensors_path))
    entries: dict[int, dict[str, CachedSubmodule]] = {}
    for tensor_key, info in manifest["tensors"].items():
        layer_idx = int(info["layer_idx"])
        name = info["submodule"]
        stats_vec = tensors[f"{tensor_key}.stats"].tolist()
        quantiles = tuple(tensors[f"{tensor_key}.quantiles"].tolist())
        dim_mean = tuple(tensors[f"{tensor_key}.dim_mean"].tolist())
        positions = tuple(int(p) for p in tensors[f"{tensor_key}.sample_positions"].tolist())
        sample_key = f"{tensor_key}.sample_tokens"
        sample_tokens = (
            tuple(tuple(row) for row in tensors[sample_key].tolist())
            if sample_key in tensors
            else ()
        )
        full_key = f"{tensor_key}.full"
        full = (
            tuple(tuple(row) for row in tensors[full_key].tolist())
            if full_key in tensors
            else None
        )
        stats = SubmoduleStats(
            num_tokens=int(info["num_tokens"]),
            dim=int(info["dim"]),
            mean=stats_vec[0],
            std=stats_vec[1],
            absmax=stats_vec[2],
            l2=stats_vec[3],
            min=stats_vec[4],
            max=stats_vec[5],
            quantiles=quantiles,
        )
        entries.setdefault(layer_idx, {})[name] = CachedSubmodule(
            stats=stats,
            dim_mean=dim_mean,
            sample_positions=positions,
            sample_tokens=sample_tokens,
            full=full,
        )
    lookup = CacheLookup(
        hit=True, path=safetensors_path, message=f"oracle cache hit: {safetensors_path}"
    )
    return entries, lookup


if __name__ == "__main__":
    demo_trace = {0: {"self_attn": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}}
    demo_key = CacheKey(
        model_revision="demo",
        prompt_hash=compute_prompt_hash([1, 2, 3]),
        layer_set="all",
        capture_config=CaptureConfig(k_edge_tokens=1, r_random_tokens=0),
    )
    demo_entries = build_cache_entries(demo_key.capture_config, demo_trace)
    print("stats:", demo_entries[0]["self_attn"].stats)
    print("scan vector:", to_activation_trace(demo_entries))
