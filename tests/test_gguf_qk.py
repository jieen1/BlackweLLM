from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from loader.gguf_dequant import (  # noqa: E402
    Q4_K_BLOCK_BYTES,
    Q5_K_BLOCK_BYTES,
    Q6_K_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES,
)
from runtime.kernels.gguf_qk import NativeGgufQK, artifact_paths  # noqa: E402
from runtime.loading.gguf import dequantize_gguf_packed  # noqa: E402
from runtime.model.gguf_linear import (  # noqa: E402
    GgufEmbedding,
    GgufLinear,
    GgufMergedLinear,
    _native_mmq_lm_head_enabled,
    _native_mmq_q5_enabled,
    _native_mmq_q8_enabled,
    _native_mmq_q8_module_enabled,
    _native_mmq_q8_rows_enabled,
    _native_mmq_rows_enabled,
    _native_mmq_shape_enabled,
    _native_mxfp6_rows_enabled,
    _native_mxfp6_w6a8_enabled,
    _native_tensor_core_tile_major_enabled,
    _native_tensor_core_tile_major_module_enabled,
    _native_tensor_core_tile_major_rows_enabled,
    _resident_bf16_weights_enabled,
    gguf_q8_activation_cache,
)


def test_tensor_core_block_m_auto_preserves_verify_and_widens_prefill(monkeypatch) -> None:
    pytest.importorskip("triton")
    from runtime.kernels.gguf_qk_triton import (  # noqa: PLC0415
        _tensor_core_block_m,
        _tensor_core_block_n,
        _tensor_core_decode_elements,
        _tensor_core_num_stages,
        _tensor_core_num_warps,
    )

    monkeypatch.delenv("QSR_GGUF_TC_BLOCK_M", raising=False)
    monkeypatch.delenv("QSR_GGUF_TC_BLOCK_N", raising=False)
    monkeypatch.delenv("QSR_GGUF_TC_DECODE_ELEMENTS", raising=False)

    assert _tensor_core_block_m(8) == 8
    assert _tensor_core_block_m(31) == 8
    assert _tensor_core_block_m(32) == 32
    assert _tensor_core_block_m(4096) == 32
    assert _tensor_core_block_m(32, type_name="Q5_K") == 64
    assert _tensor_core_block_m(63, type_name="Q6_K_SPLIT") == 32
    assert _tensor_core_block_m(64, type_name="Q6_K_SPLIT") == 64
    assert _tensor_core_block_n(type_name="Q6_K_SPLIT", rows=8, n=5120, k=17408) == 16
    assert _tensor_core_block_n(type_name="Q6_K_SPLIT", rows=32, n=5120, k=17408) == 32
    assert _tensor_core_block_n(type_name="Q5_K", rows=8, n=5120, k=17408) == 32
    assert _tensor_core_decode_elements(4, rows=8, n=5120, k=17408) == 128
    assert _tensor_core_decode_elements(4, rows=32, n=5120, k=17408) == 256
    monkeypatch.setenv("QSR_GGUF_TC_Q6_SMALL_M_DOWN", "0")
    assert _tensor_core_block_n(type_name="Q6_K_SPLIT", rows=8, n=5120, k=17408) == 32
    assert _tensor_core_decode_elements(4, rows=8, n=5120, k=17408) == 256

    monkeypatch.setenv("QSR_GGUF_TC_BLOCK_M", "16")
    assert _tensor_core_block_m(8) == 16
    assert _tensor_core_block_m(4096) == 16

    monkeypatch.setenv("QSR_GGUF_TC_BLOCK_M", "64")
    assert _tensor_core_block_m(8) == 64

    monkeypatch.setenv("QSR_GGUF_TC_BLOCK_M", "invalid")
    assert _tensor_core_block_m(4096) == 8

    monkeypatch.delenv("QSR_GGUF_TC_WARPS", raising=False)
    monkeypatch.delenv("QSR_GGUF_TC_STAGES", raising=False)
    assert _tensor_core_num_warps("Q5_K", 8) == 4
    assert _tensor_core_num_stages("Q5_K", 8) == 1
    assert _tensor_core_num_stages("Q6_K_SPLIT", 8) == 1
    assert _tensor_core_num_warps("Q5_K", 4096) == 8
    assert _tensor_core_num_stages("Q5_K", 4096) == 2
    assert _tensor_core_num_warps("Q6_K_SPLIT", 4096) == 4
    assert _tensor_core_num_stages("Q6_K_SPLIT", 4096) == 1

    monkeypatch.setenv("QSR_GGUF_TC_WARPS", "4")
    monkeypatch.setenv("QSR_GGUF_TC_STAGES", "3")
    assert _tensor_core_num_warps("Q5_K", 4096) == 4
    assert _tensor_core_num_stages("Q5_K", 4096) == 3

    monkeypatch.setenv("QSR_GGUF_TC_WARPS", "invalid")
    monkeypatch.setenv("QSR_GGUF_TC_STAGES", "invalid")
    assert _tensor_core_num_warps("Q5_K", 4096) == 4
    assert _tensor_core_num_stages("Q5_K", 4096) == 1


def test_q8_lm_head_mmq_is_an_explicit_shape_specific_switch(monkeypatch) -> None:
    monkeypatch.delenv("QSR_GGUF_NATIVE_MMQ_LM_HEAD", raising=False)
    assert not _native_mmq_lm_head_enabled("lm_head", "Q8_0")
    monkeypatch.setenv("QSR_GGUF_NATIVE_MMQ_LM_HEAD", "1")
    assert _native_mmq_lm_head_enabled("lm_head", "Q8_0")
    assert not _native_mmq_lm_head_enabled("lm_head", "Q6_K")
    assert not _native_mmq_lm_head_enabled("model.layers.0", "Q8_0")


def test_q8_mmq_module_allowlist_is_explicit(monkeypatch) -> None:
    monkeypatch.delenv("QSR_GGUF_NATIVE_MMQ_Q8_MODULES", raising=False)
    assert _native_mmq_q8_module_enabled("model.layers.0.linear_attn.out_proj")
    monkeypatch.setenv("QSR_GGUF_NATIVE_MMQ_Q8_MODULES", "linear_attn.out_proj, in_proj_qkv")
    assert _native_mmq_q8_module_enabled("model.layers.0.linear_attn.out_proj")
    assert _native_mmq_q8_module_enabled("model.layers.0.linear_attn.in_proj_qkv")
    assert not _native_mmq_q8_module_enabled("model.layers.0.mlp.gate_proj")
    assert not _native_mmq_q8_module_enabled(None)


def test_mxfp6_w6a8_is_explicitly_limited_to_measured_rows(monkeypatch) -> None:
    monkeypatch.delenv("QSR_GGUF_MXFP6_W6A8", raising=False)
    monkeypatch.delenv("QSR_GGUF_MXFP6_ROWS", raising=False)
    assert not _native_mxfp6_w6a8_enabled()
    assert _native_mxfp6_rows_enabled(8)
    assert not _native_mxfp6_rows_enabled(1)

    monkeypatch.setenv("QSR_GGUF_MXFP6_W6A8", "1")
    monkeypatch.setenv("QSR_GGUF_MXFP6_ROWS", "8,16")
    assert _native_mxfp6_w6a8_enabled()
    assert _native_mxfp6_rows_enabled(8)
    assert _native_mxfp6_rows_enabled(16)
    assert not _native_mxfp6_rows_enabled(7)

    monkeypatch.setenv("QSR_GGUF_MXFP6_ROWS", "not-a-row")
    assert not _native_mxfp6_rows_enabled(8)


def test_tile_major_q6_is_an_explicit_m8_switch(monkeypatch) -> None:
    monkeypatch.delenv("QSR_GGUF_TC_TILE_MAJOR", raising=False)
    monkeypatch.delenv("QSR_GGUF_TC_TILE_MAJOR_ROWS", raising=False)
    monkeypatch.delenv("QSR_GGUF_TC_TILE_MAJOR_MODULES", raising=False)
    assert not _native_tensor_core_tile_major_enabled()
    assert _native_tensor_core_tile_major_rows_enabled(8)
    assert not _native_tensor_core_tile_major_rows_enabled(1)
    assert _native_tensor_core_tile_major_module_enabled("model.layers.0.mlp.up_proj")

    monkeypatch.setenv("QSR_GGUF_TC_TILE_MAJOR", "1")
    monkeypatch.setenv("QSR_GGUF_TC_TILE_MAJOR_ROWS", "8,16")
    monkeypatch.setenv("QSR_GGUF_TC_TILE_MAJOR_MODULES", "mlp,linear_attn")
    assert _native_tensor_core_tile_major_enabled()
    assert _native_tensor_core_tile_major_rows_enabled(8)
    assert not _native_tensor_core_tile_major_rows_enabled(7)
    assert _native_tensor_core_tile_major_module_enabled("model.layers.0.mlp.up_proj")
    assert _native_tensor_core_tile_major_module_enabled("model.layers.0.linear_attn.out_proj")
    assert not _native_tensor_core_tile_major_module_enabled("lm_head")


@pytest.mark.parametrize(
    ("type_name", "block_bytes"),
    [
        ("Q4_K", Q4_K_BLOCK_BYTES),
        ("Q5_K", Q5_K_BLOCK_BYTES),
        ("Q6_K", Q6_K_BLOCK_BYTES),
        ("Q8_0", Q8_0_BLOCK_BYTES),
    ],
)
def test_gguf_reference_dequant_and_cpu_modules(type_name: str, block_bytes: int) -> None:
    rows, width = 2, 256
    blocks_per_row = width // (32 if type_name == "Q8_0" else 256)
    packed = torch.zeros(rows * blocks_per_row * block_bytes, dtype=torch.uint8)
    values = dequantize_gguf_packed(packed, (rows, width), type_name)
    assert values.shape == (rows, width)
    assert values.dtype == torch.float32
    assert torch.count_nonzero(values) == 0

    linear = GgufLinear(width, rows, type_name)
    linear.weight.weight_loader(linear.weight, packed)
    output = linear(torch.ones(3, width, dtype=torch.bfloat16))
    assert output.shape == (3, rows)
    assert torch.count_nonzero(output) == 0

    embedding = GgufEmbedding(rows, width, type_name)
    embedding.weight.weight_loader(embedding.weight, packed)
    gathered = embedding(torch.tensor([0, 1], dtype=torch.long))
    assert gathered.shape == (2, width)
    assert torch.count_nonzero(gathered) == 0

    with pytest.raises(IndexError, match="outside the vocabulary"):
        embedding(torch.tensor([rows], dtype=torch.long))


def _valid_random_packed(type_name: str, rows: int, width: int) -> torch.Tensor:
    block_bytes = {
        "Q4_K": Q4_K_BLOCK_BYTES,
        "Q5_K": Q5_K_BLOCK_BYTES,
        "Q6_K": Q6_K_BLOCK_BYTES,
        "Q8_0": Q8_0_BLOCK_BYTES,
    }[type_name]
    blocks_per_row = width // (32 if type_name == "Q8_0" else 256)
    packed = torch.randint(
        0,
        256,
        (rows * blocks_per_row * block_bytes,),
        dtype=torch.uint8,
    )
    scale_bytes = torch.tensor([0x66, 0x2E], dtype=torch.uint8)  # BF16-safe FP16 0.1
    for block_start in range(0, packed.numel(), block_bytes):
        scale_offset = 208 if type_name == "Q6_K" else 0
        packed[block_start + scale_offset : block_start + scale_offset + 2] = scale_bytes
        if type_name in {"Q4_K", "Q5_K"}:
            packed[block_start + 2 : block_start + 4] = 0
    return packed


def _pad_q6_k_rows(packed: torch.Tensor, rows: int, width: int) -> torch.Tensor:
    blocks_per_row = width // 256
    source = packed.view(rows, blocks_per_row, Q6_K_BLOCK_BYTES)
    padded = torch.zeros(
        rows,
        blocks_per_row,
        224,
        dtype=packed.dtype,
        device=packed.device,
    )
    padded[..., :Q6_K_BLOCK_BYTES].copy_(source)
    return padded.reshape(-1)


def _split_q6_k_rows(packed: torch.Tensor, rows: int, width: int) -> torch.Tensor:
    blocks_per_row = width // 256
    source = packed.view(rows, blocks_per_row, Q6_K_BLOCK_BYTES)
    split = torch.empty(
        rows,
        blocks_per_row * Q6_K_BLOCK_BYTES,
        dtype=packed.dtype,
        device=packed.device,
    )
    split[:, : blocks_per_row * 208].view(rows, blocks_per_row, 208).copy_(source[..., :208])
    split[:, blocks_per_row * 208 :].view(rows, blocks_per_row, 2).copy_(source[..., 208:])
    return split.reshape(-1)


def _split_q8_0_rows(packed: torch.Tensor, rows: int, width: int) -> torch.Tensor:
    blocks_per_row = width // 32
    source = packed.view(rows, blocks_per_row, Q8_0_BLOCK_BYTES)
    split = torch.empty(
        rows,
        blocks_per_row * Q8_0_BLOCK_BYTES,
        dtype=packed.dtype,
        device=packed.device,
    )
    split[:, : blocks_per_row * 32].view(rows, blocks_per_row, 32).copy_(source[..., 2:])
    split[:, blocks_per_row * 32 :].view(rows, blocks_per_row, 2).copy_(source[..., :2])
    return split.reshape(-1)


def test_resident_bf16_dequant_releases_packed_payload() -> None:
    rows, width = 3, 256
    packed = _valid_random_packed("Q6_K", rows, width)
    linear = GgufLinear(width, rows, "Q6_K")
    linear.weight.weight_loader(linear.weight, packed)

    resident = linear._materialize_resident_bf16_weight()

    assert resident.shape == (rows, width)
    assert resident.dtype == torch.bfloat16
    assert linear.weight.numel() == 0
    assert linear._packed_weight_released
    x = torch.ones(2, width, dtype=torch.bfloat16)
    output = linear(x)
    assert torch.equal(output, F.linear(x, resident))


def test_resident_bf16_type_filter_is_selective(monkeypatch) -> None:
    monkeypatch.setenv("QSR_GGUF_DEQUANTIZE_WEIGHTS", "0")
    monkeypatch.delenv("QSR_GGUF_DEQUANTIZE_TYPES", raising=False)
    assert not _resident_bf16_weights_enabled("Q6_K")

    monkeypatch.setenv("QSR_GGUF_DEQUANTIZE_TYPES", "q6_k, Q8_0")
    assert _resident_bf16_weights_enabled("Q6_K")
    assert _resident_bf16_weights_enabled("Q8_0")
    assert not _resident_bf16_weights_enabled("Q5_K")

    monkeypatch.setenv("QSR_GGUF_DEQUANTIZE_MODULES", "ffn_up, ffn_gate")
    assert _resident_bf16_weights_enabled("Q8_0", "blk.1.ffn_up.weight")
    assert not _resident_bf16_weights_enabled("Q8_0", "blk.1.attn_qkv.weight")
    assert _resident_bf16_weights_enabled("Q6_K", "blk.1.ffn_up.weight")
    assert not _resident_bf16_weights_enabled("Q5_K", "blk.1.ffn_up.weight")

    monkeypatch.setenv("QSR_GGUF_DEQUANTIZE_WEIGHTS", "1")
    assert not _resident_bf16_weights_enabled("Q5_K")
    monkeypatch.delenv("QSR_GGUF_DEQUANTIZE_TYPES")
    monkeypatch.delenv("QSR_GGUF_DEQUANTIZE_MODULES")
    assert _resident_bf16_weights_enabled("Q5_K")


def test_merged_gguf_cpu_fallback_matches_independent_projections() -> None:
    width = 256
    rows = 2
    packed = _valid_random_packed("Q6_K", rows, width)
    left = GgufLinear(width, rows, "Q6_K")
    right = GgufLinear(width, rows, "Q6_K")
    left.weight.weight_loader(left.weight, packed)
    right.weight.weight_loader(right.weight, packed)

    merged = GgufMergedLinear(left, right)
    x = torch.randn(3, width, dtype=torch.bfloat16)

    expected = torch.cat((left(x), right(x)), dim=-1)
    actual = merged(x)

    assert torch.equal(actual, expected)


def test_mixed_format_merged_gguf_cpu_fallback_matches_independent_projections() -> None:
    width = 256
    rows = 2
    left = GgufLinear(width, rows, "Q6_K")
    right = GgufLinear(width, rows, "Q8_0")
    left.weight.weight_loader(left.weight, _valid_random_packed("Q6_K", rows, width))
    right.weight.weight_loader(right.weight, _valid_random_packed("Q8_0", rows, width))

    merged = GgufMergedLinear(left, right)
    x = torch.randn(3, width, dtype=torch.bfloat16)

    expected = torch.cat((left(x), right(x)), dim=-1)
    actual = merged(x)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_qk_matches_reference_for_non_padded_batch() -> None:
    native = NativeGgufQK.load()
    device = torch.device("cuda")
    rows, output_rows, width = 3, 5, 768
    for type_name in ("Q4_K", "Q5_K", "Q6_K", "Q8_0"):
        packed = _valid_random_packed(type_name, output_rows, width).to(device)
        x = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
        row_bytes = packed.numel() // output_rows
        actual = native.gemm(
            x,
            packed,
            m=rows,
            n=output_rows,
            k=width,
            row_bytes=row_bytes,
            type_name=type_name,
        )
        reference_weight = dequantize_gguf_packed(
            packed, (output_rows, width), type_name, dtype=torch.bfloat16
        )
        expected = F.linear(x, reference_weight)
        torch.cuda.synchronize()
        assert (
            torch.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
            > 0.999
        )
        max_expected = expected.float().abs().max().item()
        max_error = (actual.float() - expected.float()).abs().max().item()
        assert max_error <= max(4096.0, max_expected * 0.02)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_prequantized_qk_matches_regular_q8_path() -> None:
    native = NativeGgufQK.load()
    device = torch.device("cuda")
    rows, output_rows, width = 1, 5, 768
    x = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    workspace = native.quantize_q8_1(x)
    for type_name in ("Q4_K", "Q5_K", "Q6_K", "Q8_0"):
        packed = _valid_random_packed(type_name, output_rows, width).to(device)
        row_bytes = packed.numel() // output_rows
        actual = native.gemm_q8_prequantized(
            workspace,
            packed,
            m=rows,
            n=output_rows,
            k=width,
            row_bytes=row_bytes,
            type_name=type_name,
        )
        expected = native.gemm(
            x,
            packed,
            m=rows,
            n=output_rows,
            k=width,
            row_bytes=row_bytes,
            type_name=type_name,
        )
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_q8_activation_staging_matches_uncached_q8_gemv() -> None:
    native = NativeGgufQK.load()
    device = torch.device("cuda")
    rows, output_rows, width = 1, 9, 768
    x = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    workspace = native.quantize_q8_1(x)
    for type_name in ("Q4_K", "Q5_K", "Q6_K", "Q8_0"):
        packed = _valid_random_packed(type_name, output_rows, width).to(device)
        row_bytes = packed.numel() // output_rows
        expected = native.gemm_q8_prequantized(
            workspace,
            packed,
            m=rows,
            n=output_rows,
            k=width,
            row_bytes=row_bytes,
            type_name=type_name,
            cache_activation=False,
        )
        actual = native.gemm_q8_prequantized(
            workspace,
            packed,
            m=rows,
            n=output_rows,
            k=width,
            row_bytes=row_bytes,
            type_name=type_name,
            cache_activation=True,
        )
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_q8_m8_activation_staging_matches_uncached_q6_tile() -> None:
    native = NativeGgufQK.load()
    device = torch.device("cuda")
    rows, output_rows, width = 8, 37, 768
    standard = _valid_random_packed("Q6_K", output_rows, width).to(device)
    packed = _split_q6_k_rows(standard, output_rows, width)
    x = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    workspace = native.quantize_q8_1(x)
    row_bytes = standard.numel() // output_rows
    expected = native.gemm_q8_prequantized(
        workspace,
        packed,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=row_bytes,
        type_name="Q6_K_SPLIT",
        cache_activation=False,
    )
    actual = native.gemm_q8_prequantized(
        workspace,
        packed,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=row_bytes,
        type_name="Q6_K_SPLIT",
        cache_activation=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_f32_q8_path_preserves_quantized_activation_contract() -> None:
    """F32 input/output must agree with the separately quantized F32 row."""

    native = NativeGgufQK.load()
    device = torch.device("cuda")
    rows, output_rows, width = 1, 5, 768
    x = torch.randn(rows, width, dtype=torch.float32, device=device)
    workspace = native.quantize_q8_1(x)
    for type_name in ("Q4_K", "Q5_K", "Q6_K", "Q8_0"):
        packed = _valid_random_packed(type_name, output_rows, width).to(device)
        row_bytes = packed.numel() // output_rows
        direct = native.gemm_q8_f32(
            x,
            packed,
            m=rows,
            n=output_rows,
            k=width,
            row_bytes=row_bytes,
            type_name=type_name,
        )
        prequantized = native.gemm_q8_prequantized(
            workspace,
            packed,
            m=rows,
            n=output_rows,
            k=width,
            row_bytes=row_bytes,
            type_name=type_name,
            output_dtype=torch.float32,
        )
        torch.cuda.synchronize()
        assert direct.dtype == torch.float32
        assert torch.equal(direct, prequantized)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_padded_q6_prequantized_matches_standard_layout() -> None:
    native = NativeGgufQK.load()
    device = torch.device("cuda")
    rows, output_rows, width = 1, 5, 768
    x = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    workspace = native.quantize_q8_1(x)
    standard = _valid_random_packed("Q6_K", output_rows, width).to(device)
    padded = _pad_q6_k_rows(standard, output_rows, width)

    expected = native.gemm_q8_prequantized(
        workspace,
        standard,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=standard.numel() // output_rows,
        type_name="Q6_K",
    )
    actual = native.gemm_q8_prequantized(
        workspace,
        padded,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=padded.numel() // output_rows,
        type_name="Q6_K_ALIGNED",
    )
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_split_q6_matches_standard_prefill_and_decode() -> None:
    native = NativeGgufQK.load()
    device = torch.device("cuda")
    rows, output_rows, width = 8, 5, 768
    standard = _valid_random_packed("Q6_K", output_rows, width).to(device)
    split = _split_q6_k_rows(standard, output_rows, width)
    row_bytes = standard.numel() // output_rows
    x = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    expected = native.gemm(
        x,
        standard,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=row_bytes,
        type_name="Q6_K",
    )
    actual = native.gemm(
        x,
        split,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=row_bytes,
        type_name="Q6_K_SPLIT",
    )
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_split_q8_matches_standard_prefill_decode_and_gather() -> None:
    native = NativeGgufQK.load()
    device = torch.device("cuda")
    rows, output_rows, width = 8, 5, 768
    standard = _valid_random_packed("Q8_0", output_rows, width).to(device)
    split = _split_q8_0_rows(standard, output_rows, width)
    row_bytes = standard.numel() // output_rows
    x = torch.randn(rows, width, dtype=torch.bfloat16, device=device)

    expected = native.gemm(
        x,
        standard,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=row_bytes,
        type_name="Q8_0",
    )
    actual = native.gemm(
        x,
        split,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=row_bytes,
        type_name="Q8_0_SPLIT",
    )
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)

    expected_direct = native.gemm_direct(
        x[:1],
        standard,
        m=1,
        n=output_rows,
        k=width,
        row_bytes=row_bytes,
        type_name="Q8_0",
    )
    actual_direct = native.gemm_direct(
        x[:1],
        split,
        m=1,
        n=output_rows,
        k=width,
        row_bytes=row_bytes,
        type_name="Q8_0_SPLIT",
    )
    assert torch.equal(actual_direct, expected_direct)

    ids = torch.tensor([0, output_rows - 1], dtype=torch.int64, device=device)
    expected_rows = native.dequant_rows(
        ids,
        standard,
        rows=ids.numel(),
        k=width,
        row_bytes=row_bytes,
        type_name="Q8_0",
    )
    actual_rows = native.dequant_rows(
        ids,
        split,
        rows=ids.numel(),
        k=width,
        row_bytes=row_bytes,
        type_name="Q8_0_SPLIT",
    )
    torch.cuda.synchronize()
    assert torch.equal(actual_rows, expected_rows)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_split_q8_handles_m1_and_unaligned_row_stride() -> None:
    """Exercise both launch families when row N is not four-byte aligned."""

    native = NativeGgufQK.load()
    device = torch.device("cuda")
    # Three Q8_0 blocks make a 102-byte row.  If the allocation base is
    # aligned, row 1 is therefore deliberately 2-byte aligned and must take
    # the unaligned payload-load path.
    rows, output_rows, width = 8, 5, 96
    standard = _valid_random_packed("Q8_0", output_rows, width).to(device)
    split = _split_q8_0_rows(standard, output_rows, width)
    row_bytes = standard.numel() // output_rows
    assert row_bytes == 102

    x = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    expected_m1 = native.gemm(
        x[:1],
        standard,
        m=1,
        n=output_rows,
        k=width,
        row_bytes=row_bytes,
        type_name="Q8_0",
    )
    actual_m1 = native.gemm(
        x[:1],
        split,
        m=1,
        n=output_rows,
        k=width,
        row_bytes=row_bytes,
        type_name="Q8_0_SPLIT",
    )
    workspace = native.quantize_q8_1(x)
    expected_m8 = native.gemm_q8_prequantized(
        workspace,
        standard,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=row_bytes,
        type_name="Q8_0",
    )
    actual_m8 = native.gemm_q8_prequantized(
        workspace,
        split,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=row_bytes,
        type_name="Q8_0_SPLIT",
    )
    expected_direct = native.gemm_direct(
        x[:1].float(),
        standard,
        m=1,
        n=output_rows,
        k=width,
        row_bytes=row_bytes,
        type_name="Q8_0",
    )
    actual_direct = native.gemm_direct(
        x[:1].float(),
        split,
        m=1,
        n=output_rows,
        k=width,
        row_bytes=row_bytes,
        type_name="Q8_0_SPLIT",
    )
    torch.cuda.synchronize()
    assert torch.equal(actual_m1, expected_m1)
    assert torch.equal(actual_m8, expected_m8)
    assert torch.equal(actual_direct, expected_direct)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_cached_exact_gemv_matches_uncached() -> None:
    """Shared activation staging must preserve the exact decode contract."""

    native = NativeGgufQK.load()
    device = torch.device("cuda:0")
    output_rows, width = 5, 5120
    for type_name in ("Q4_K", "Q5_K", "Q6_K", "Q8_0"):
        standard = _valid_random_packed(type_name, output_rows, width).to(device)
        row_bytes = standard.numel() // output_rows
        for dtype in (torch.bfloat16, torch.float32):
            x = torch.randn(1, width, dtype=dtype, device=device)
            expected = native.gemm_direct(
                x,
                standard,
                m=1,
                n=output_rows,
                k=width,
                row_bytes=row_bytes,
                type_name=type_name,
            )
            actual = native.gemm_direct(
                x,
                standard,
                m=1,
                n=output_rows,
                k=width,
                row_bytes=row_bytes,
                type_name=type_name,
                cache_activation=True,
            )
            torch.cuda.synchronize()
            assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_cached_mixed_exact_gemv_matches_uncached() -> None:
    """Merged QKV/GateUp staging must preserve mixed-format rows exactly."""

    device = torch.device("cuda:0")
    width = 5120
    left = GgufLinear(width, 3, "Q6_K").to(device)
    right = GgufLinear(width, 2, "Q8_0").to(device)
    left.weight.weight_loader(left.weight, _valid_random_packed("Q6_K", 3, width).to(device))
    right.weight.weight_loader(right.weight, _valid_random_packed("Q8_0", 2, width).to(device))
    merged = GgufMergedLinear(left, right)
    native = NativeGgufQK.load()
    descriptors = merged._ensure_mixed_descriptors(device)
    for dtype in (torch.bfloat16, torch.float32):
        x = torch.randn(1, width, dtype=dtype, device=device)
        expected = native.gemm_direct_mixed(
            x,
            descriptors,
            projection_count=2,
            total_n=5,
            k=width,
        )
        actual = native.gemm_direct_mixed(
            x,
            descriptors,
            projection_count=2,
            total_n=5,
            k=width,
            cache_activation=True,
        )
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_mixed_merged_qk_matches_independent_projections() -> None:
    device = torch.device("cuda")
    width = 768
    left = GgufLinear(width, 3, "Q6_K").to(device)
    right = GgufLinear(width, 2, "Q8_0").to(device)
    left.weight.weight_loader(left.weight, _valid_random_packed("Q6_K", 3, width).to(device))
    right.weight.weight_loader(right.weight, _valid_random_packed("Q8_0", 2, width).to(device))
    merged = GgufMergedLinear(left, right)
    x = torch.randn(1, width, dtype=torch.bfloat16, device=device)

    expected = torch.cat((left(x), right(x)), dim=-1)
    actual = merged(x)

    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_mixed_f32_merged_qk_matches_independent_projections() -> None:
    """Keep the exact F32 mixed launch on the same contract as two linears."""

    device = torch.device("cuda")
    width = 768
    left = GgufLinear(width, 3, "Q6_K").to(device)
    right = GgufLinear(width, 2, "Q8_0").to(device)
    left.weight.weight_loader(left.weight, _valid_random_packed("Q6_K", 3, width).to(device))
    right.weight.weight_loader(right.weight, _valid_random_packed("Q8_0", 2, width).to(device))
    merged = GgufMergedLinear(left, right)
    x = torch.randn(1, width, dtype=torch.float32, device=device)

    expected = torch.cat((left(x), right(x)), dim=-1)
    actual = merged(x)

    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_mixed_f32_q8_merged_qk_reuses_one_activation_row(monkeypatch) -> None:
    """The SGLang-style F32 Q8 route also covers mixed adjacent formats."""

    monkeypatch.setenv("QSR_GGUF_NATIVE_F32_Q8", "1")
    monkeypatch.setenv("QSR_GGUF_NATIVE_Q8_ACTIVATION_CACHE", "1")
    device = torch.device("cuda")
    width = 768
    left = GgufLinear(width, 3, "Q6_K").to(device)
    right = GgufLinear(width, 2, "Q8_0").to(device)
    left.weight.weight_loader(left.weight, _valid_random_packed("Q6_K", 3, width).to(device))
    right.weight.weight_loader(right.weight, _valid_random_packed("Q8_0", 2, width).to(device))
    merged = GgufMergedLinear(left, right)
    x = torch.randn(1, width, dtype=torch.float32, device=device)

    with gguf_q8_activation_cache():
        actual = merged(x)
        workspace = NativeGgufQK.load().quantize_q8_1(x)
        expected = NativeGgufQK.load().gemm_q8_mixed(
            workspace,
            merged._ensure_mixed_descriptors(device),
            projection_count=2,
            total_n=5,
            k=width,
            output_dtype=torch.float32,
        )

    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_mixed_split_descriptors_match_independent_projections(monkeypatch) -> None:
    """Keep the mixed descriptor ABI covered for both private layouts."""

    monkeypatch.setenv("QSR_GGUF_Q6_SPLIT", "1")
    monkeypatch.setenv("QSR_GGUF_Q8_SPLIT", "1")
    device = torch.device("cuda")
    width = 768
    left = GgufLinear(width, 3, "Q6_K").to(device)
    right = GgufLinear(width, 2, "Q8_0").to(device)
    left.weight.weight_loader(left.weight, _valid_random_packed("Q6_K", 3, width).to(device))
    right.weight.weight_loader(right.weight, _valid_random_packed("Q8_0", 2, width).to(device))
    merged = GgufMergedLinear(left, right)
    x = torch.randn(1, width, dtype=torch.bfloat16, device=device)

    expected = torch.cat((left(x), right(x)), dim=-1)
    actual = merged(x)

    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_tensor_core_matches_reference_for_non_padded_batch() -> None:
    native = NativeGgufQK.load()
    device = torch.device("cuda")
    rows, output_rows, width = 8, 5, 768
    for type_name in ("Q4_K", "Q5_K", "Q6_K", "Q8_0"):
        packed = _valid_random_packed(type_name, output_rows, width).to(device)
        x = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
        row_bytes = packed.numel() // output_rows
        actual = native.gemm_tensor_core(
            x,
            packed,
            m=rows,
            n=output_rows,
            k=width,
            row_bytes=row_bytes,
            type_name=type_name,
        )
        reference_weight = dequantize_gguf_packed(
            packed, (output_rows, width), type_name, dtype=torch.bfloat16
        )
        expected = F.linear(x, reference_weight)
        torch.cuda.synchronize()
        assert (
            torch.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
            > 0.999
        )
        max_expected = expected.float().abs().max().item()
        max_error = (actual.float() - expected.float()).abs().max().item()
        assert max_error <= max(4096.0, max_expected * 0.02)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_tensor_core_reads_split_q6_and_q8_rows() -> None:
    """Keep split decode storage usable by the large-M tensor-core path."""

    native = NativeGgufQK.load()
    device = torch.device("cuda")
    rows, output_rows, width = 8, 5, 768
    x = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    for type_name, split_name, splitter in (
        ("Q6_K", "Q6_K_SPLIT", _split_q6_k_rows),
        ("Q8_0", "Q8_0_SPLIT", _split_q8_0_rows),
    ):
        standard = _valid_random_packed(type_name, output_rows, width).to(device)
        split = splitter(standard, output_rows, width)
        row_bytes = standard.numel() // output_rows
        expected = native.gemm_tensor_core(
            x,
            standard,
            m=rows,
            n=output_rows,
            k=width,
            row_bytes=row_bytes,
            type_name=type_name,
        )
        actual = native.gemm_tensor_core(
            x,
            split,
            m=rows,
            n=output_rows,
            k=width,
            row_bytes=row_bytes,
            type_name=split_name,
        )
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_q6_tile_major_matches_row_major_exactly() -> None:
    """The physical N-tile reorder must not change any Q6 result."""

    from runtime.kernels.gguf_qk_triton import _tensor_core_block_n, gguf_qk_repack_for_tensor_core

    native = NativeGgufQK.load()
    device = torch.device("cuda")
    rows, output_rows, width = 8, 37, 768
    x = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    for type_name, split_name, splitter in (
        ("Q6_K", "Q6_K", lambda packed, count, size: packed),
        ("Q6_K", "Q6_K_SPLIT", _split_q6_k_rows),
    ):
        standard = _valid_random_packed(type_name, output_rows, width).to(device)
        packed = splitter(standard, output_rows, width)
        row_bytes = standard.numel() // output_rows
        expected = native.gemm_tensor_core(
            x,
            packed,
            m=rows,
            n=output_rows,
            k=width,
            row_bytes=row_bytes,
            type_name=split_name,
        )
        block_n = _tensor_core_block_n(
            type_name=split_name,
            rows=rows,
            n=output_rows,
            k=width,
        )
        tile_major, _ = gguf_qk_repack_for_tensor_core(
            packed,
            n=output_rows,
            k=width,
            row_bytes=row_bytes,
            type_name=split_name,
            block_n=block_n,
        )
        actual = native.gemm_tensor_core_tile_major(
            x,
            tile_major,
            m=rows,
            n=output_rows,
            k=width,
            type_name=split_name,
            block_n=block_n,
        )
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_q6_mmq_matches_q8_tile_for_split_rows() -> None:
    """Keep the SGLang-style Q6 MMQ dot contract aligned with Q8 tile math."""

    native = NativeGgufQK.load()
    device = torch.device("cuda")
    rows, output_rows, width = 8, 37, 768
    standard = _valid_random_packed("Q6_K", output_rows, width).to(device)
    split = _split_q6_k_rows(standard, output_rows, width)
    x = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    activation = native.quantize_q8_1(x)
    expected = native.gemm_q8_prequantized(
        activation,
        split,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=standard.numel() // output_rows,
        type_name="Q6_K_SPLIT",
    )
    actual = native.gemm_q8_mmq(
        activation,
        split,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=standard.numel() // output_rows,
        type_name="Q6_K_SPLIT",
    )
    torch.cuda.synchronize()
    assert torch.allclose(actual, expected, rtol=2e-3, atol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_q6_mmq_checks_m_and_n_tile_tails_for_split_rows() -> None:
    """The SGLang-style checked specialization must handle ragged MMQ CTAs."""

    native = NativeGgufQK.load()
    device = torch.device("cuda")
    rows, output_rows, width = 5, 37, 768
    standard = _valid_random_packed("Q6_K", output_rows, width).to(device)
    split = _split_q6_k_rows(standard, output_rows, width)
    x = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    activation = native.quantize_q8_1(x)
    expected = native.gemm_q8_prequantized(
        activation,
        split,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=standard.numel() // output_rows,
        type_name="Q6_K_SPLIT",
    )
    actual = native.gemm_q8_mmq(
        activation,
        split,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=standard.numel() // output_rows,
        type_name="Q6_K_SPLIT",
    )
    torch.cuda.synchronize()
    assert torch.allclose(actual, expected, rtol=2e-3, atol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_q6_mmq_mma_candidate_matches_q8_tile_for_split_rows(monkeypatch) -> None:
    """Validate the opt-in llama.cpp/ds4 integer-MMA Q6 candidate."""

    monkeypatch.setenv("QSR_GGUF_MMQ_MMA", "1")
    native = NativeGgufQK.load()
    device = torch.device("cuda")
    rows, width = 8, 768
    x = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    activation = native.quantize_q8_1(x)
    for output_rows in (37, 128, 129):
        standard = _valid_random_packed("Q6_K", output_rows, width).to(device)
        split = _split_q6_k_rows(standard, output_rows, width)
        expected = native.gemm_q8_prequantized(
            activation,
            split,
            m=rows,
            n=output_rows,
            k=width,
            row_bytes=standard.numel() // output_rows,
            type_name="Q6_K_SPLIT",
        )
        actual = native.gemm_q8_mmq(
            activation,
            split,
            m=rows,
            n=output_rows,
            k=width,
            row_bytes=standard.numel() // output_rows,
            type_name="Q6_K_SPLIT",
        )
        diff = (actual.float() - expected.float()).abs()
        # Integer MMA changes the reduction order relative to the DP4A
        # oracle.  The BF16 boundary can therefore move by one ulp more than
        # the DP4A tolerance while still preserving the quantized result.
        assert torch.allclose(actual, expected, rtol=1e-2, atol=0.25), (
            output_rows,
            float(diff.max().item()),
            tuple(int(index) for index in torch.unravel_index(diff.argmax(), diff.shape)),
            float(actual.flatten()[diff.argmax()].item()),
            float(expected.flatten()[diff.argmax()].item()),
        )
    torch.cuda.synchronize()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_q8_mmq_matches_q8_tile_for_split_rows() -> None:
    """Keep the SGLang-style Q8_0 MMQ dot contract aligned with Q8 tile math."""

    native = NativeGgufQK.load()
    device = torch.device("cuda")
    rows, output_rows, width = 8, 37, 768
    standard = _valid_random_packed("Q8_0", output_rows, width).to(device)
    split = _split_q8_0_rows(standard, output_rows, width)
    x = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    activation = native.quantize_q8_1(x)
    expected = native.gemm_q8_prequantized(
        activation,
        split,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=standard.numel() // output_rows,
        type_name="Q8_0_SPLIT",
    )
    actual = native.gemm_q8_mmq(
        activation,
        split,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=standard.numel() // output_rows,
        type_name="Q8_0_SPLIT",
    )
    torch.cuda.synchronize()
    assert torch.allclose(actual, expected, rtol=2e-3, atol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_native_q5_mmq_matches_q8_tile_for_rows() -> None:
    """Keep the SGLang-style Q5_K MMQ dot contract aligned with Q8 tile math."""

    native = NativeGgufQK.load()
    device = torch.device("cuda")
    rows, output_rows, width = 8, 37, 768
    packed = _valid_random_packed("Q5_K", output_rows, width).to(device)
    x = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    activation = native.quantize_q8_1(x)
    row_bytes = packed.numel() // output_rows
    expected = native.gemm_q8_prequantized(
        activation,
        packed,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=row_bytes,
        type_name="Q5_K",
    )
    actual = native.gemm_q8_mmq(
        activation,
        packed,
        m=rows,
        n=output_rows,
        k=width,
        row_bytes=row_bytes,
        type_name="Q5_K",
    )
    torch.cuda.synchronize()
    assert torch.allclose(actual, expected, rtol=2e-3, atol=2e-2)


def test_native_mmq_selector_is_limited_to_dflash2_verify(monkeypatch) -> None:
    """Do not route prefill or ragged tails through the experimental MMQ tile."""

    monkeypatch.setenv("QSR_GGUF_NATIVE_MMQ", "1")
    assert _native_mmq_rows_enabled(8)
    assert not _native_mmq_rows_enabled(1)
    assert not _native_mmq_rows_enabled(7)
    assert not _native_mmq_rows_enabled(4096)
    assert _native_mmq_shape_enabled(34_816, 5_120)
    assert _native_mmq_shape_enabled(17_408, 5_120)
    assert not _native_mmq_shape_enabled(5_120, 5_120)
    assert not _native_mmq_shape_enabled(5_120, 17_408)
    assert not _native_mmq_q8_enabled()
    assert not _native_mmq_q5_enabled()
    monkeypatch.setenv("QSR_GGUF_NATIVE_MMQ_Q8", "1")
    assert _native_mmq_q8_enabled()
    monkeypatch.delenv("QSR_GGUF_NATIVE_MMQ", raising=False)
    assert _native_mmq_q8_rows_enabled(8)
    assert not _native_mmq_rows_enabled(8)
    monkeypatch.setenv("QSR_GGUF_NATIVE_MMQ_Q5", "1")
    assert _native_mmq_q5_enabled()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_high_level_split_decode_storage_survives_tensor_core_prefill(monkeypatch) -> None:
    """Exercise the graph-order transition that previously produced an empty payload."""

    monkeypatch.setenv("QSR_GGUF_Q6_SPLIT", "1")
    monkeypatch.setenv("QSR_GGUF_Q8_SPLIT", "1")
    monkeypatch.setenv("QSR_GGUF_NATIVE_TC", "1")
    monkeypatch.setenv("QSR_GGUF_TC_TILE_MAJOR", "1")
    native = NativeGgufQK.load()
    device = torch.device("cuda")
    width, output_rows = 768, 5
    packed = _valid_random_packed("Q6_K", output_rows, width).to(device)
    x_decode = torch.randn(1, width, dtype=torch.bfloat16, device=device)
    x_prefill = torch.randn(8, width, dtype=torch.bfloat16, device=device)

    linear = GgufLinear(width, output_rows, "Q6_K").to(device)
    linear.weight.weight_loader(linear.weight, packed)
    linear(x_decode)  # capture-order M=1 path builds and releases split storage
    actual = linear(x_prefill)  # M=8 must consume that split storage through Triton
    split = linear._native_packed_weight
    assert split is not None
    expected = native.gemm_tensor_core(
        x_prefill,
        split,
        m=8,
        n=output_rows,
        k=width,
        row_bytes=packed.numel() // output_rows,
        type_name="Q6_K_SPLIT",
    )
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)

    left = GgufLinear(width, output_rows, "Q6_K").to(device)
    right = GgufLinear(width, output_rows, "Q6_K").to(device)
    left.weight.weight_loader(left.weight, packed)
    right.weight.weight_loader(right.weight, packed)
    merged = GgufMergedLinear(left, right)
    merged(x_decode)  # same-type fusion owns and releases the source tensors
    actual_merged = merged(x_prefill)
    merged_packed = merged._q8_packed_weight
    assert merged_packed is not None
    expected_merged = native.gemm_tensor_core(
        x_prefill,
        merged_packed,
        m=8,
        n=output_rows * 2,
        k=width,
        row_bytes=packed.numel() // output_rows,
        type_name="Q6_K_SPLIT",
    )
    torch.cuda.synchronize()
    assert torch.equal(actual_merged, expected_merged)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not Path(artifact_paths()[0]).is_file(),
    reason="native GGUF Q/K artifact and SM120 CUDA are required",
)
def test_high_level_q8_disabled_merged_projection_keeps_exact_fallback(monkeypatch) -> None:
    """Disabling approximate Q8 activation quantization must be honored."""

    monkeypatch.setenv("QSR_GGUF_NATIVE_Q8", "0")
    device = torch.device("cuda")
    width, output_rows = 768, 5
    left = GgufLinear(width, output_rows, "Q6_K").to(device)
    right = GgufLinear(width, output_rows, "Q8_0").to(device)
    left.weight.weight_loader(
        left.weight, _valid_random_packed("Q6_K", output_rows, width).to(device)
    )
    right.weight.weight_loader(
        right.weight, _valid_random_packed("Q8_0", output_rows, width).to(device)
    )
    merged = GgufMergedLinear(left, right)
    x = torch.randn(1, width, dtype=torch.bfloat16, device=device)

    expected = torch.cat((left(x), right(x)), dim=-1)
    actual = merged(x)

    torch.cuda.synchronize()
    assert torch.equal(actual, expected)
