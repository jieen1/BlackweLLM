from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import runtime.kernels.dsv4_q8_gemm as q8_gemm  # noqa: E402


class _FakeKernel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append({"grid": grid, "args": args, "kwargs": kwargs})

        return launch


def _packed_bytes(out_features: int, in_features: int) -> torch.Tensor:
    row_stride = (in_features // 32) * 34
    return torch.zeros(out_features * row_stride, dtype=torch.uint8)


@pytest.mark.parametrize(
    ("rows", "in_features", "out_features", "block_m", "block_n"),
    [
        (1, 4096, 64, 8, 8),
        (2, 4096, 64, 8, 8),
        (4, 4096, 64, 16, 8),
        (1, 4096, 1024, 8, 8),
        (2, 4096, 1024, 8, 8),
        (4, 4096, 1024, 8, 8),
        (1, 1024, 8192, 8, 16),
        (2, 1024, 8192, 8, 32),
        (4, 1024, 8192, 8, 16),
        (1, 4096, 512, 8, 16),
        (4, 4096, 512, 8, 16),
        (1, 8192, 4096, 8, 16),
        (4, 8192, 4096, 8, 16),
        (8, 4096, 512, 16, 32),
        (8, 8192, 4096, 16, 32),
        (1, 1024, 32768, 16, 64),
    ],
)
def test_q8_gemm_default_selector_uses_expected_tiles(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    in_features: int,
    out_features: int,
    block_m: int,
    block_n: int,
) -> None:
    fake_kernel = _FakeKernel()
    monkeypatch.setattr(q8_gemm, "_q8_0_dequant_gemm_tc_kernel", fake_kernel)
    x = torch.zeros(rows, in_features, dtype=torch.bfloat16)
    packed = _packed_bytes(out_features, in_features)

    out = q8_gemm.q8_0_dequant_gemm(
        x,
        packed,
        out_features=out_features,
        in_features=in_features,
    )

    assert out.shape == (rows, out_features)
    call = fake_kernel.calls.pop()
    assert call["kwargs"]["BLOCK_M"] == block_m
    assert call["kwargs"]["BLOCK_N"] == block_n


def test_q8_gemm_explicit_tile_override_bypasses_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_kernel = _FakeKernel()
    monkeypatch.setattr(q8_gemm, "_q8_0_dequant_gemm_tc_kernel", fake_kernel)
    x = torch.zeros(1, 4096, dtype=torch.bfloat16)
    packed = _packed_bytes(512, 4096)

    q8_gemm.q8_0_dequant_gemm(
        x,
        packed,
        out_features=512,
        in_features=4096,
        BLOCK_M=16,
        BLOCK_N=32,
    )

    call = fake_kernel.calls.pop()
    assert call["kwargs"]["BLOCK_M"] == 16
    assert call["kwargs"]["BLOCK_N"] == 32


@pytest.mark.parametrize(
    ("rows", "in_features", "out_features", "block_cols"),
    [
        (1, 4096, 2048, 8),
        (2, 4096, 2048, 8),
        (4, 2048, 4096, 8),
        (1, 512, 1024, 32),
    ],
)
def test_q8_fp32_default_selector_uses_expected_tile(
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    in_features: int,
    out_features: int,
    block_cols: int,
) -> None:
    fake_kernel = _FakeKernel()
    monkeypatch.setattr(q8_gemm, "_q8_0_dequant_gemv_fp32_kernel", fake_kernel)
    x = torch.zeros(rows, in_features, dtype=torch.bfloat16)
    packed = _packed_bytes(out_features, in_features)

    out = q8_gemm.q8_0_dequant_gemv_fp32(
        x,
        packed,
        out_features=out_features,
        in_features=in_features,
    )

    assert out.shape == (rows, out_features)
    call = fake_kernel.calls.pop()
    assert call["kwargs"]["BLOCK_COLS"] == block_cols


@pytest.mark.parametrize(
    ("rows_per_group", "block_m", "block_n", "row_tiles"),
    [
        (1, 8, 16, 1),
        (4, 8, 16, 1),
        (5, 16, 64, 1),
        (17, 16, 64, 2),
    ],
)
def test_q8_grouped_gemm_default_selector_uses_expected_tiles(
    monkeypatch: pytest.MonkeyPatch,
    rows_per_group: int,
    block_m: int,
    block_n: int,
    row_tiles: int,
) -> None:
    fake_kernel = _FakeKernel()
    monkeypatch.setattr(q8_gemm, "_q8_0_grouped_dequant_gemm_tc_kernel", fake_kernel)
    num_groups = 8
    group_size = 1024
    in_features = 4096
    x = torch.zeros(num_groups * rows_per_group, in_features, dtype=torch.bfloat16)
    packed = _packed_bytes(num_groups * group_size, in_features)

    out = q8_gemm.q8_0_grouped_dequant_gemm(
        x,
        packed,
        num_groups=num_groups,
        group_size=group_size,
        in_features=in_features,
        rows_per_group=rows_per_group,
    )

    assert out.shape == (num_groups * rows_per_group, group_size)
    call = fake_kernel.calls.pop()
    assert call["grid"] == (num_groups, row_tiles, 64 if block_n == 16 else 16)
    assert call["kwargs"]["BLOCK_M"] == block_m
    assert call["kwargs"]["BLOCK_N"] == block_n


def test_q8_grouped_gemm_explicit_tile_override_bypasses_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_kernel = _FakeKernel()
    monkeypatch.setattr(q8_gemm, "_q8_0_grouped_dequant_gemm_tc_kernel", fake_kernel)
    num_groups = 8
    group_size = 1024
    in_features = 4096
    rows_per_group = 1
    x = torch.zeros(num_groups * rows_per_group, in_features, dtype=torch.bfloat16)
    packed = _packed_bytes(num_groups * group_size, in_features)

    q8_gemm.q8_0_grouped_dequant_gemm(
        x,
        packed,
        num_groups=num_groups,
        group_size=group_size,
        in_features=in_features,
        rows_per_group=rows_per_group,
        BLOCK_M=16,
        BLOCK_N=32,
    )

    call = fake_kernel.calls.pop()
    assert call["kwargs"]["BLOCK_M"] == 16
    assert call["kwargs"]["BLOCK_N"] == 32
