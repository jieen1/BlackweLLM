"""Unit coverage for Flash-Next image normalization and bounded preprocessing."""

from __future__ import annotations

import base64
import io

import pytest

# ``runtime.model.flashnext`` imports the CUDA-backed speculative modules from
# its package initializer.  Keep the whole module out of the torch-free CI
# collection job, as required by the repository test contract.
pytest.importorskip("torch")

from runtime.model.flashnext.vision import (
    PreparedVisionInput,
    build_image_cache_keys,
    build_mrope_positions,
    expand_image_tokens,
    prepare_image_inputs,
)
from server.formats.anthropic import parse_messages
from server.formats.openai import parse_chat_messages
from server.formats.responses import parse_input


def _data_uri(width: int = 4000, height: int = 2000) -> str:
    image = pytest.importorskip("PIL.Image").new("RGB", (width, height), "navy")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


class _FakeImageProcessor:
    merge_size = 2

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *, images, return_tensors, min_pixels, max_pixels):
        import numpy as np

        self.calls.append(
            {
                "sizes": [image.size for image in images],
                "return_tensors": return_tensors,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            }
        )
        grids = []
        patches = 0
        for image in images:
            # Qwen's 32-pixel merged grid, represented in the unmerged
            # [T,H,W] convention used by image_grid_thw.
            height = (image.height + 31) // 32
            width = (image.width + 31) // 32
            grid = [1, height * 2, width * 2]
            grids.append(grid)
            patches += grid[0] * grid[1] * grid[2]
        return {
            "pixel_values": np.zeros((patches, 1536), dtype=np.float32),
            "image_grid_thw": np.asarray(grids, dtype=np.int64),
        }


def test_high_resolution_image_is_compressed_before_patchification() -> None:
    processor = _FakeImageProcessor()
    prepared = prepare_image_inputs(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image_url": _data_uri()},
                    {"type": "text", "text": "describe"},
                ],
            }
        ],
        processor=processor,
        max_pixels=262_144,
        min_pixels=65_536,
        max_image_tokens=512,
    )

    assert prepared.source_sizes == ((4000, 2000),)
    resized = prepared.resized_sizes[0]
    assert resized[0] * resized[1] <= 262_144
    assert resized != prepared.source_sizes[0]
    assert prepared.total_image_tokens <= 512
    assert processor.calls[0]["max_pixels"] == 262_144


def test_image_token_expansion_is_strict() -> None:
    assert expand_image_tokens([1, 99, 2, 99, 3], 99, (4, 2)) == [
        1,
        99,
        99,
        99,
        99,
        2,
        99,
        99,
        3,
    ]
    with pytest.raises(ValueError, match="fewer image markers"):
        expand_image_tokens([99], 99, (2, 3))
    with pytest.raises(ValueError, match="more image markers"):
        expand_image_tokens([99, 99], 99, (2,))


def test_provider_adapters_preserve_image_blocks() -> None:
    uri = _data_uri(64, 64)
    openai = parse_chat_messages(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image_url", "image_url": {"url": uri}},
                    ],
                }
            ]
        }
    )
    assert openai[0]["content"][1] == {"type": "image", "image_url": {"url": uri}}

    anthropic = parse_messages(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(io.BytesIO().getvalue()).decode(),
                            },
                        },
                    ],
                }
            ]
        }
    )
    assert anthropic[0]["content"][1]["type"] == "image"
    assert anthropic[0]["content"][1]["image"]["type"] == "base64"

    responses = parse_input(
        {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is this?"},
                        {"type": "input_image", "image_url": uri},
                    ],
                }
            ]
        }
    )
    assert responses[0]["content"][1] == {"type": "image", "image_url": uri}


def test_provider_adapters_preserve_video_blocks_for_explicit_rejection() -> None:
    uri = "https://example.invalid/video.mp4"
    openai = parse_chat_messages(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "video_url", "video_url": {"url": uri}}],
                }
            ]
        }
    )
    assert openai[0]["content"][0] == {"type": "video", "video": {"url": uri}}

    anthropic = parse_messages(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "video", "source": {"url": uri}}],
                }
            ]
        }
    )
    assert anthropic[0]["content"][0]["type"] == "video"

    responses = parse_input(
        {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_video", "video_url": uri}],
                }
            ]
        }
    )
    assert responses[0]["content"][0] == {"type": "video", "video": uri}


def test_prepared_input_is_cpu_side_and_reports_token_budget() -> None:
    import numpy as np

    prepared = PreparedVisionInput(
        pixel_values=np.zeros((8, 1536), dtype=np.float32),
        image_grid_thw=np.asarray([[1, 2, 4]], dtype=np.int64),
        image_token_counts=(2,),
        source_sizes=((64, 64),),
        resized_sizes=((64, 64),),
        max_pixels=65_536,
        max_image_tokens=4_096,
    )
    assert prepared.total_image_tokens == 2


def test_build_image_cache_keys_changes_only_for_the_touched_image() -> None:
    import numpy as np

    pixels = np.arange(12, dtype=np.float32).reshape(3, 4)
    grids = np.asarray([[1, 1, 2], [1, 1, 1]], dtype=np.int64)
    base = build_image_cache_keys(
        pixels,
        grids,
        source_sizes=((64, 64), (32, 32)),
        resized_sizes=((64, 64), (32, 32)),
        max_pixels=65_536,
        max_image_tokens=256,
        spatial_merge_size=1,
    )
    changed = pixels.copy()
    changed[0, 0] = -1
    mutated = build_image_cache_keys(
        changed,
        grids,
        source_sizes=((64, 64), (32, 32)),
        resized_sizes=((64, 64), (32, 32)),
        max_pixels=65_536,
        max_image_tokens=256,
        spatial_merge_size=1,
    )

    assert len(base) == 2
    assert base[0] != mutated[0]
    assert base[1] == mutated[1]


def test_mrope_positions_match_qwen_vl_image_run_layout() -> None:
    # vision_start + four merged image tokens + vision_end + two text tokens
    # with a 2x2 spatial merge.  The image coordinates must advance each axis
    # independently while surrounding text remains scalar on all three axes.
    positions, next_position = build_mrope_positions(
        [10, 99, 99, 99, 99, 11, 20, 21],
        image_token_id=99,
        vision_start_token_id=10,
        image_grid_thw=[[1, 4, 4]],
        spatial_merge_size=2,
    )

    import numpy as np

    expected = np.asarray(
        [
            [0, 1, 1, 1, 1, 3, 4, 5],
            [0, 1, 1, 2, 2, 3, 4, 5],
            [0, 1, 2, 1, 2, 3, 4, 5],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(positions, expected)
    assert next_position == 6
