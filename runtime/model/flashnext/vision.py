"""Vision preprocessing and loading for the Flash-Next checkpoint.

The text runtime deliberately keeps image handling out of the hot decode
path. This module owns the slow, request-time boundary instead:

* decode OpenAI/Anthropic/local image references safely;
* resize large images before patchification;
* cap the visual token budget so one request cannot exhaust the GPU; and
* build/load the Qwen3-VL vision tower used by the Qwen4Exp checkpoint.

The imports for Pillow, NumPy, Transformers, and safetensors are lazy. Text
only deployments therefore retain the existing lightweight import surface.
"""

from __future__ import annotations

import base64
import binascii
import functools
import hashlib
import io
import math
import os
import pathlib
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_IMAGE_BLOCK_TYPES = frozenset({"image", "image_url", "input_image"})
_VIDEO_BLOCK_TYPES = frozenset({"video", "video_url", "input_video"})
_DEFAULT_MAX_PIXELS = 1_048_576  # 1 MP: enough detail without 16 MP bursts
_DEFAULT_MIN_PIXELS = 65_536
# 16384 covers realistic agent screenshot traffic (measured 2026-09-01: an
# OpenCode turn with several ~1 MP screenshots produced 4350 visual tokens and
# was rejected by the earlier 4096 default).  The per-image 1 MP pixel budget
# still bounds each image; this total only caps the vision-tower burst, and
# the prompt-capacity check keeps the whole request inside the 256K slot.
_DEFAULT_MAX_IMAGE_TOKENS = 16_384
_DEFAULT_FETCH_MAX_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_DECODE_PIXELS = 100 * 1_000 * 1_000
_DEFAULT_MAX_DIMENSION = 16_384


@dataclass(frozen=True)
class PreparedVisionInput:
    """CPU-side image tensors ready for one target prefill.

    ``pixel_values`` is a NumPy array with the flattened patch layout
    expected by Qwen3VLVisionModel; ``image_grid_thw`` describes the original
    image boundaries in that flattened array. Keeping this object CPU-side
    until the engine thread enters the vision tower avoids retaining
    request-specific GPU buffers between rounds.
    """

    pixel_values: Any
    image_grid_thw: Any
    image_token_counts: tuple[int, ...]
    source_sizes: tuple[tuple[int, int], ...]
    resized_sizes: tuple[tuple[int, int], ...]
    max_pixels: int
    max_image_tokens: int
    image_cache_keys: tuple[str, ...] = ()
    # Filled by the server after the chat template has expanded image markers.
    # Qwen4-Exp uses interleaved 3-axis MRoPE for multimodal prompts; keeping
    # these positions with the CPU-side image batch avoids recomputing them
    # for every prefill chunk.
    rope_positions: Any | None = None
    next_rope_position: int | None = None
    spatial_merge_size: int = 2

    @property
    def total_image_tokens(self) -> int:
        return sum(self.image_token_counts)


def _positive_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def image_max_pixels() -> int:
    """Return the configured post-decode image area budget.

    The default is intentionally conservative. Operators can raise it for a
    known workload, but the hard 16 MP ceiling prevents accidentally restoring
    the checkpoint processor's 16,777,216-pixel default on a nearly-full card.
    """

    value = _positive_env("QSR_FLASHNEXT_IMAGE_MAX_PIXELS", _DEFAULT_MAX_PIXELS)
    if value > 16 * 1024 * 1024:
        raise ValueError(
            "QSR_FLASHNEXT_IMAGE_MAX_PIXELS may not exceed 16777216; "
            "larger visual bursts are unsafe for this single-GPU runtime"
        )
    return value


def image_min_pixels() -> int:
    return _positive_env("QSR_FLASHNEXT_IMAGE_MIN_PIXELS", _DEFAULT_MIN_PIXELS)


def image_max_tokens() -> int:
    return _positive_env("QSR_FLASHNEXT_IMAGE_MAX_TOKENS", _DEFAULT_MAX_IMAGE_TOKENS)


def _is_image_block(block: Mapping[str, Any]) -> bool:
    block_type = str(block.get("type", "")).lower()
    return block_type in _IMAGE_BLOCK_TYPES or (
        block_type not in _VIDEO_BLOCK_TYPES
        and ("image" in block or "image_url" in block or "source" in block)
        and block_type != "text"
    )


def extract_image_blocks(messages: list[dict] | tuple[dict, ...]) -> list[dict]:
    """Return image blocks in template order."""

    images: list[dict] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, Mapping) and _is_image_block(block):
                images.append(dict(block))
    return images


def has_video_blocks(messages: list[dict] | tuple[dict, ...]) -> bool:
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, list):
            continue
        if any(
            isinstance(block, Mapping)
            and str(block.get("type", "")).lower() in _VIDEO_BLOCK_TYPES
            for block in content
        ):
            return True
    return False


def _payload_from_block(block: Mapping[str, Any]) -> Any:
    """Extract the URL/base64/path payload from common API block shapes."""

    if "image_url" in block:
        payload = block["image_url"]
    elif "image" in block:
        payload = block["image"]
    elif "source" in block:
        payload = block["source"]
    else:
        payload = block.get("url")

    if isinstance(payload, Mapping):
        # OpenAI uses {url, detail}; Anthropic uses {type, media_type, data}.
        if (
            str(payload.get("type", "")).lower() == "base64"
            or payload.get("b64_json") is not None
        ):
            return payload
        for key in ("url", "data", "b64_json", "path"):
            if payload.get(key) is not None:
                return payload[key]
    return payload


def _read_remote(url: str, max_bytes: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "blackwellm-flashnext/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        raise ValueError(
                            f"remote image is larger than the {max_bytes} byte fetch limit"
                        )
                except ValueError as exc:
                    if "larger than" in str(exc):
                        raise
                    # A malformed Content-Length is harmless; the bounded
                    # streaming read below remains authoritative.
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(min(1024 * 1024, max_bytes - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(
                        f"remote image is larger than the {max_bytes} byte fetch limit"
                    )
                chunks.append(chunk)
            return b"".join(chunks)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"failed to fetch image URL {url!r}: {exc}") from exc


def _decode_payload(payload: Any, *, max_bytes: int) -> bytes | Any:
    """Decode one payload to bytes, or return an already-open PIL image."""

    # Avoid importing PIL for text-only requests. ``Image.Image`` is checked
    # by duck typing so tests and callers can pass a PIL image directly.
    if hasattr(payload, "size") and hasattr(payload, "convert"):
        return payload
    if isinstance(payload, (bytes, bytearray, memoryview)):
        data = bytes(payload)
        if len(data) > max_bytes:
            raise ValueError(f"image payload is larger than the {max_bytes} byte limit")
        return data
    if isinstance(payload, Mapping):
        source_type = str(payload.get("type", "")).lower()
        if source_type == "base64" or payload.get("data") is not None:
            encoded = payload.get("data") or payload.get("b64_json")
            if not isinstance(encoded, str):
                raise ValueError("base64 image source must contain a string data field")
            try:
                data = base64.b64decode("".join(encoded.split()), validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("image base64 data is malformed") from exc
            if len(data) > max_bytes:
                raise ValueError(f"image payload is larger than the {max_bytes} byte limit")
            return data
        payload = _payload_from_block(payload)

    if not isinstance(payload, str):
        raise ValueError(f"unsupported image payload type: {type(payload).__name__}")
    value = payload.strip()
    if value.startswith("data:"):
        header, separator, encoded = value.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError("only base64 data:image URLs are supported")
        try:
            data = base64.b64decode("".join(encoded.split()), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("image data URL contains malformed base64") from exc
        if len(data) > max_bytes:
            raise ValueError(f"image payload is larger than the {max_bytes} byte limit")
        return data
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme in {"http", "https"}:
        return _read_remote(value, max_bytes)
    if parsed.scheme == "file":
        value = urllib.parse.unquote(parsed.path)
    path = pathlib.Path(value).expanduser()
    if not path.is_file():
        raise ValueError(
            "image reference must be a data URL, http(s) URL, file path, or base64 source"
        )
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"failed to read image path {str(path)!r}: {exc}") from exc
    if len(data) > max_bytes:
        raise ValueError(f"image payload is larger than the {max_bytes} byte limit")
    return data


def _open_and_bound_image(payload: Any, *, max_pixels: int, max_bytes: int):
    from PIL import Image, ImageOps

    source = _decode_payload(payload, max_bytes=max_bytes)
    if hasattr(source, "convert"):
        image = source
    else:
        try:
            image = Image.open(io.BytesIO(source))
        except Exception as exc:
            raise ValueError(f"unable to decode image: {exc}") from exc
    width, height = (int(image.width), int(image.height))
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if max(width, height) > _DEFAULT_MAX_DIMENSION:
        raise ValueError(
            f"image dimensions {width}x{height} exceed the safe "
            f"{_DEFAULT_MAX_DIMENSION} pixel edge limit; resize it before sending"
        )
    decode_limit = _positive_env(
        "QSR_FLASHNEXT_IMAGE_MAX_DECODE_PIXELS", _DEFAULT_MAX_DECODE_PIXELS
    )
    if width * height > decode_limit:
        raise ValueError(
            f"image area {width * height} exceeds the safe decode limit {decode_limit}; "
            "resize it before sending"
        )
    # Validate the header before materialising pixels.  A compressed 100 MP
    # image can otherwise allocate hundreds of MiB during ``load()`` before
    # the post-decode max-pixel budget gets a chance to resize it.
    if not hasattr(source, "convert"):
        try:
            image.load()
        except Exception as exc:
            raise ValueError(f"unable to decode image: {exc}") from exc
    image = ImageOps.exif_transpose(image).convert("RGB")
    source_size = (int(image.width), int(image.height))
    area = source_size[0] * source_size[1]
    if area > max_pixels:
        scale = math.sqrt(max_pixels / area)
        resized = (
            max(1, int(round(source_size[0] * scale))),
            max(1, int(round(source_size[1] * scale))),
        )
        image = image.resize(resized, Image.Resampling.LANCZOS)
    return image, source_size, (int(image.width), int(image.height))


@functools.lru_cache(maxsize=4)
def get_image_processor(checkpoint: str):
    """Load and cache the checkpoint's Qwen2-VL image processor."""

    try:
        from transformers import Qwen2VLImageProcessor
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError(
            "image requests require transformers' Qwen2VLImageProcessor"
        ) from exc
    try:
        return Qwen2VLImageProcessor.from_pretrained(checkpoint)
    except Exception as exc:
        raise RuntimeError(
            f"failed to load Flash-Next image processor from {checkpoint!r}: {exc}"
        ) from exc


def prepare_image_inputs(
    messages: list[dict] | tuple[dict, ...],
    *,
    checkpoint: str | os.PathLike[str] | None = None,
    processor: Any | None = None,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
    max_image_tokens: int | None = None,
) -> PreparedVisionInput:
    """Decode, compress, and patchify all images in ``messages``."""

    blocks = extract_image_blocks(messages)
    if not blocks:
        raise ValueError("prepare_image_inputs called without an image block")
    if has_video_blocks(messages):
        raise ValueError(
            "video inputs are not enabled yet; send still images to the Flash-Next runtime"
        )
    max_pixels = int(image_max_pixels() if max_pixels is None else max_pixels)
    min_pixels = int(image_min_pixels() if min_pixels is None else min_pixels)
    max_image_tokens = int(
        image_max_tokens() if max_image_tokens is None else max_image_tokens
    )
    if max_pixels <= 0 or min_pixels <= 0 or max_image_tokens <= 0:
        raise ValueError("image pixel/token budgets must be positive")
    if min_pixels > max_pixels:
        raise ValueError(
            f"image min_pixels={min_pixels} cannot exceed max_pixels={max_pixels}"
        )
    fetch_max_bytes = _positive_env(
        "QSR_FLASHNEXT_IMAGE_FETCH_MAX_BYTES", _DEFAULT_FETCH_MAX_BYTES
    )
    images = []
    source_sizes: list[tuple[int, int]] = []
    resized_sizes: list[tuple[int, int]] = []
    for block in blocks:
        image, source_size, resized_size = _open_and_bound_image(
            _payload_from_block(block),
            max_pixels=max_pixels,
            max_bytes=fetch_max_bytes,
        )
        images.append(image)
        source_sizes.append(source_size)
        resized_sizes.append(resized_size)

    if processor is None:
        if checkpoint is None:
            raise ValueError("a local checkpoint is required to load the image processor")
        processor = get_image_processor(str(checkpoint))
    try:
        features = processor(
            images=images,
            return_tensors="np",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    except Exception as exc:
        raise ValueError(f"image patchification failed: {exc}") from exc

    import numpy as np

    pixel_values = np.asarray(features["pixel_values"])
    image_grid_thw = np.asarray(features["image_grid_thw"], dtype=np.int64)
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[0] != len(images):
        raise ValueError(
            "image processor returned an invalid image_grid_thw shape: "
            f"{tuple(image_grid_thw.shape)} for {len(images)} images"
        )
    merge_size = int(getattr(processor, "merge_size", 2))
    merge_area = merge_size * merge_size
    image_token_counts = tuple(
        int(grid[0] * grid[1] * grid[2] // merge_area) for grid in image_grid_thw
    )
    total_tokens = sum(image_token_counts)
    if total_tokens <= 0:
        raise ValueError("image processor produced zero visual tokens")
    if total_tokens > max_image_tokens:
        raise ValueError(
            f"images produce {total_tokens} visual tokens, exceeding the configured "
            f"limit of {max_image_tokens}; lower image resolution or raise "
            "QSR_FLASHNEXT_IMAGE_MAX_TOKENS deliberately"
        )
    image_cache_keys = build_image_cache_keys(
        pixel_values,
        image_grid_thw,
        source_sizes=tuple(source_sizes),
        resized_sizes=tuple(resized_sizes),
        max_pixels=max_pixels,
        max_image_tokens=max_image_tokens,
        spatial_merge_size=merge_size,
    )
    return PreparedVisionInput(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        image_token_counts=image_token_counts,
        source_sizes=tuple(source_sizes),
        resized_sizes=tuple(resized_sizes),
        max_pixels=max_pixels,
        max_image_tokens=max_image_tokens,
        image_cache_keys=image_cache_keys,
        spatial_merge_size=merge_size,
    )


def build_image_cache_keys(
    pixel_values: Any,
    image_grid_thw: Any,
    *,
    source_sizes: tuple[tuple[int, int], ...],
    resized_sizes: tuple[tuple[int, int], ...],
    max_pixels: int,
    max_image_tokens: int,
    spatial_merge_size: int,
) -> tuple[str, ...]:
    """Return one deterministic cache-authentication key per image.

    Prefix-cache reuse must authenticate the actual processed vision payload,
    not just the placeholder token ids in the text prompt. The key is built
    from the exact patch rows the vision tower will consume plus the geometry
    metadata that maps those rows back to image boundaries.
    """

    import numpy as np

    grids = np.asarray(image_grid_thw, dtype=np.int64)
    pixels = np.ascontiguousarray(np.asarray(pixel_values))
    if grids.ndim != 2 or grids.shape[1] != 3:
        raise ValueError(
            "image_grid_thw must be a [num_images,3] array to build cache keys, "
            f"got shape {tuple(grids.shape)}"
        )
    if len(source_sizes) != grids.shape[0] or len(resized_sizes) != grids.shape[0]:
        raise ValueError(
            "source_sizes and resized_sizes must match image_grid_thw rows: "
            f"{len(source_sizes)=}, {len(resized_sizes)=}, {grids.shape[0]=}"
        )

    def digest_array(value: Any) -> bytes:
        array = np.ascontiguousarray(np.asarray(value))
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("utf-8"))
        digest.update(memoryview(array).cast("B"))
        return digest.digest()

    offset = 0
    keys: list[str] = []
    for index, grid in enumerate(grids):
        patch_rows = int(grid[0] * grid[1] * grid[2])
        next_offset = offset + patch_rows
        if next_offset > pixels.shape[0]:
            raise ValueError(
                "pixel_values rows are shorter than image_grid_thw requires: "
                f"need {next_offset}, have {pixels.shape[0]}"
            )
        digest = hashlib.sha256()
        digest.update(digest_array(pixels[offset:next_offset]))
        digest.update(digest_array(grid))
        digest.update(str(source_sizes[index]).encode("utf-8"))
        digest.update(str(resized_sizes[index]).encode("utf-8"))
        digest.update(str(int(max_pixels)).encode("utf-8"))
        digest.update(str(int(max_image_tokens)).encode("utf-8"))
        digest.update(str(int(spatial_merge_size)).encode("utf-8"))
        keys.append(digest.hexdigest())
        offset = next_offset
    if offset != pixels.shape[0]:
        raise ValueError(
            "image_grid_thw does not consume every pixel_values row: "
            f"used {offset}, total {pixels.shape[0]}"
        )
    return tuple(keys)


def expand_image_tokens(
    prompt_ids: list[int] | tuple[int, ...],
    image_token_id: int,
    image_token_counts: tuple[int, ...] | list[int],
) -> list[int]:
    """Replace one template image marker per image with visual token slots."""

    counts = tuple(int(count) for count in image_token_counts)
    if any(count <= 0 for count in counts):
        raise ValueError(f"image token counts must be positive, got {counts!r}")
    expanded: list[int] = []
    image_index = 0
    for token in prompt_ids:
        token = int(token)
        if token != int(image_token_id):
            expanded.append(token)
            continue
        if image_index >= len(counts):
            raise ValueError(
                "chat template emitted more image markers than the image processor saw"
            )
        expanded.extend([int(image_token_id)] * counts[image_index])
        image_index += 1
    if image_index != len(counts):
        raise ValueError(
            "chat template emitted fewer image markers than the image processor saw: "
            f"markers={image_index}, images={len(counts)}"
        )
    return expanded


def build_mrope_positions(
    prompt_ids: list[int] | tuple[int, ...],
    *,
    image_token_id: int,
    vision_start_token_id: int | None = None,
    image_grid_thw: Any,
    spatial_merge_size: int = 2,
) -> tuple[Any, int]:
    """Build Qwen4-Exp's 3-axis MRoPE positions for an expanded prompt.

    This mirrors the Qwen VL implementation used by sglang's
    ``MRotaryEmbedding.get_rope_index``: text runs use the same position on
    all three axes, while each image run receives ``[T, H, W]`` coordinates
    after spatial merging.  The returned NumPy matrix is ``[3, seq]`` and is
    intentionally CPU-side so long prompts do not allocate another CUDA
    buffer before target prefill.
    """

    import numpy as np

    ids = [int(token) for token in prompt_ids]
    grids = np.asarray(image_grid_thw, dtype=np.int64)
    if grids.ndim != 2 or grids.shape[1] != 3:
        raise ValueError(f"image_grid_thw must have shape [images,3], got {grids.shape}")
    merge = int(spatial_merge_size)
    if merge <= 0:
        raise ValueError(f"spatial_merge_size must be positive, got {merge}")

    # The template emits ``vision_start, image_token * N, vision_end``.  Find
    # each run by its opening marker rather than treating every repeated token
    # as a separate image.
    runs: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(ids):
        if vision_start_token_id is not None:
            if ids[cursor] != int(vision_start_token_id):
                cursor += 1
                continue
            first = cursor + 1
            if first >= len(ids) or ids[first] != int(image_token_id):
                cursor += 1
                continue
        elif ids[cursor] != int(image_token_id):
            cursor += 1
            continue
        else:
            first = cursor
        end = first
        while end < len(ids) and ids[end] == int(image_token_id):
            end += 1
        runs.append((first, end))
        cursor = end
    if len(runs) != len(grids):
        raise ValueError(
            "expanded image markers do not match image grids: "
            f"markers={len(runs)}, grids={len(grids)}"
        )

    positions = np.ones((3, len(ids)), dtype=np.int64)
    st = 0
    next_position = 0
    for image_index, (start, end) in enumerate(runs):
        if start < st:
            raise ValueError("image marker runs overlap or are out of order")
        text_len = start - st
        if text_len:
            text_positions = np.arange(text_len, dtype=np.int64) + next_position
            positions[:, st:start] = text_positions[None, :]

        t, h, w = (int(value) for value in grids[image_index])
        llm_h = h // merge
        llm_w = w // merge
        visual_tokens = t * llm_h * llm_w
        if visual_tokens != end - start:
            raise ValueError(
                "image grid/token mismatch while building MRoPE: "
                f"image={image_index}, grid={(t, h, w)}, "
                f"expected={visual_tokens}, markers={end - start}"
            )
        t_index = np.repeat(np.arange(t, dtype=np.int64), llm_h * llm_w)
        h_index = np.tile(np.repeat(np.arange(llm_h, dtype=np.int64), llm_w), t)
        w_index = np.tile(np.arange(llm_w, dtype=np.int64), t * llm_h)
        positions[:, start:end] = np.stack(
            (t_index, h_index, w_index), axis=0
        ) + (text_len + next_position)
        next_position = int(positions[:, start:end].max()) + 1
        st = end

    if st < len(ids):
        text_positions = np.arange(len(ids) - st, dtype=np.int64) + next_position
        positions[:, st:] = text_positions[None, :]

    next_rope_position = int(positions.max()) + 1 if len(ids) else 0
    return positions, next_rope_position


def load_vision_tower(
    checkpoint: pathlib.Path,
    full_config: dict[str, Any],
    weight_map: Mapping[str, str],
    *,
    device: str,
):
    """Instantiate the official Qwen3-VL tower and load visual tensors."""

    try:
        import torch
        from safetensors import safe_open
        from transformers import Qwen3VLVisionConfig, Qwen3VLVisionModel
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError(
            "Flash-Next vision support requires Qwen3VLVisionModel and safetensors"
        ) from exc
    vision_config = dict(full_config.get("vision_config") or {})
    if not vision_config:
        raise RuntimeError("Flash-Next checkpoint has no vision_config")
    # The checkpoint advertises qwen4_exp, while Transformers' compatible
    # implementation is Qwen3-VL. The visual tensor layout is identical and
    # the Qwen3VLVisionConfig constructor deliberately normalizes model_type.
    vision_config.pop("model_type", None)
    config = Qwen3VLVisionConfig(**vision_config)
    implementation = os.environ.get("QSR_FLASHNEXT_VISION_ATTN", "sdpa").strip().lower()
    if implementation not in {"sdpa", "eager"}:
        raise ValueError(
            "QSR_FLASHNEXT_VISION_ATTN must be 'sdpa' or 'eager', "
            f"got {implementation!r}"
        )
    config._attn_implementation = implementation
    tower = Qwen3VLVisionModel(config)

    # Group by shard so loading the 333 visual tensors opens each safetensors
    # file once, rather than paying one mmap/open/close cycle per parameter.
    grouped: dict[str, list[tuple[str, Any]]] = {}
    for name, parameter in tower.named_parameters():
        checkpoint_name = f"model.visual.{name}"
        shard = weight_map.get(checkpoint_name)
        if shard is None:
            raise RuntimeError(
                f"Flash-Next visual checkpoint is missing tensor {checkpoint_name!r}"
            )
        grouped.setdefault(shard, []).append((checkpoint_name, parameter))
    with torch.no_grad():
        for shard, parameters in grouped.items():
            with safe_open(str(checkpoint / shard), framework="pt", device="cpu") as handle:
                for checkpoint_name, parameter in parameters:
                    parameter.copy_(handle.get_tensor(checkpoint_name))

    tower.to(device=device, dtype=torch.bfloat16)
    tower.eval()
    for parameter in tower.parameters():
        parameter.requires_grad_(False)
    return tower
