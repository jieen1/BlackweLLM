"""Content block parsing.

Both OpenAI and Anthropic allow content to be either a plain string
or a list of typed content blocks. This module extracts plain text and
normalises structured image/video blocks when needed.
"""

from __future__ import annotations

from typing import Any

_IMAGE_TYPES = frozenset({"image", "image_url", "input_image"})
_VIDEO_TYPES = frozenset({"video", "video_url", "input_video"})


def extract_text(field: Any) -> str:
    """Extract plain text from a flexible content field.

    Accepts:
    - None -> empty string
    - str -> returned as-is
    - list of blocks -> concatenated text from type=text entries
    - list of str -> joined with newlines
    """
    if field is None:
        return ""
    if isinstance(field, str):
        return field
    if isinstance(field, list):
        parts: list[str] = []
        for block in field:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(field)


def extract_blocks(field: Any) -> list[dict]:
    """Return the raw content blocks from a flexible content field.

    If field is a plain string it is wrapped as a text block.
    If field is already a list of dicts it is returned as-is.
    """
    if field is None:
        return []
    if isinstance(field, str):
        return [{"type": "text", "text": field}]
    if isinstance(field, list):
        out: list[dict] = []
        for block in field:
            if isinstance(block, dict):
                out.append(block)
            elif isinstance(block, str):
                out.append({"type": "text", "text": block})
        return out
    return [{"type": "text", "text": str(field)}]


def content_has_images(field: Any) -> bool:
    """Return whether a content field contains an image-bearing block."""

    return any(
        isinstance(block, dict)
        and (
            str(block.get("type", "")).lower() in _IMAGE_TYPES
            or (
                str(block.get("type", "")).lower() != "text"
                and ("image" in block or "image_url" in block or "source" in block)
            )
        )
        for block in extract_blocks(field)
    )


def content_has_videos(field: Any) -> bool:
    """Return whether a content field contains a video-bearing block."""

    return any(
        isinstance(block, dict)
        and (
            str(block.get("type", "")).lower() in _VIDEO_TYPES
            or (
                str(block.get("type", "")).lower() != "text"
                and ("video" in block or "video_url" in block)
            )
        )
        for block in extract_blocks(field)
    )


def normalize_content_blocks(field: Any) -> list[dict]:
    """Normalize provider-specific image block names for Qwen templates.

    Text and unsupported multimodal blocks remain unchanged. OpenAI's
    ``image_url`` and Responses' ``input_image`` become the single internal
    ``type='image'`` form while preserving their URL/base64 payload for the
    vision preprocessor. Keeping video blocks intact lets the request layer
    reject them explicitly instead of silently dropping them.
    """

    normalized: list[dict] = []
    for block in extract_blocks(field):
        block = dict(block)
        block_type = str(block.get("type", "")).lower()
        if block_type in {"image_url", "input_image"}:
            payload = block.get("image_url", block.get("image"))
            block = {"type": "image", "image_url": payload}
        elif block_type == "video_url":
            payload = block.get("video_url", block.get("video"))
            block = {"type": "video", "video": payload}
        elif block_type == "image" and "source" in block and "image" not in block:
            # Anthropic's source object is kept under ``image`` so all
            # adapters share one marker shape without losing base64 metadata.
            block = {"type": "image", "image": block["source"]}
        normalized.append(block)
    return normalized
