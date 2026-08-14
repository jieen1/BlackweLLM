"""Static/CPU coverage for the Triton DSV4 compressor decode kernel wrapper."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from runtime.kernels.dsv4_compressor import (  # noqa: E402
    compile_fused_decode_postgemv_batch_sm120,
    compile_fused_indexer_decode_postgemv_batch_sm120,
    fused_decode_postgemv,
    fused_decode_postgemv_batch,
    fused_indexer_decode_postgemv,
    fused_indexer_decode_postgemv_batch,
    fused_indexer_decode_postgemv_seq,
    hadamard_fp4_query,
    supports_fused_decode_postgemv,
    supports_fused_decode_postgemv_batch,
    supports_fused_indexer_decode_postgemv,
    supports_fused_indexer_decode_postgemv_batch,
)
from runtime.model.dsv4_attention import (  # noqa: E402
    apply_rotary_emb,
    fp4_act_quant_simulate,
    hadamard_transform,
    precompute_freqs_cis,
)
from runtime.model.dsv4_model import rms_norm  # noqa: E402


def _make_freqs(device: torch.device) -> torch.Tensor:
    return precompute_freqs_cis(
        64,
        384,
        original_seq_len=0,
        base=160000.0,
        factor=1.0,
        beta_fast=32,
        beta_slow=1,
        device=device,
    )


def _oracle_step(
    *,
    kv_i: torch.Tensor,
    score_i: torch.Tensor,
    position: int,
    ratio: int,
    overlap: bool,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    kv_cache: torch.Tensor,
    eps: float,
    rotate_quantize: bool = False,
) -> torch.Tensor:
    """Torch state-machine oracle for one already-projected decode token."""
    head_dim = norm_weight.numel()
    slot = position % ratio
    should_compress = (position + 1) % ratio == 0
    score_i = score_i + ape[slot].reshape(1, 1, -1)
    if overlap:
        kv_state[:, ratio + slot] = kv_i.squeeze(1)
        score_state[:, ratio + slot] = score_i.squeeze(1)
        values = torch.cat([kv_state[:, :ratio, :head_dim], kv_state[:, ratio:, head_dim:]], dim=1)
        scores = torch.cat(
            [score_state[:, :ratio, :head_dim], score_state[:, ratio:, head_dim:]], dim=1
        )
        pooled = (values * scores.softmax(dim=1)).sum(dim=1, keepdim=True)
        if should_compress:
            kv_state[:, :ratio] = kv_state[:, ratio:]
            score_state[:, :ratio] = score_state[:, ratio:]
    else:
        kv_state[:, slot] = kv_i.squeeze(1)
        score_state[:, slot] = score_i.squeeze(1)
        pooled = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)

    pooled = rms_norm(pooled.to(torch.bfloat16), norm_weight, eps)
    freqs = freqs_cis[position + 1 - ratio].unsqueeze(0)
    apply_rotary_emb(pooled[..., -64:], freqs)
    if rotate_quantize:
        pooled = hadamard_transform(pooled, head_dim**-0.5)
        pooled = fp4_act_quant_simulate(pooled, 32)
    entry = position // ratio
    if should_compress:
        kv_cache[:, entry] = pooled.squeeze(1)
    return kv_cache[:, entry : entry + 1].clone()


def _main_batch_inputs(
    *,
    batch_size: int,
    ratio: int,
    overlap: bool,
) -> dict[str, torch.Tensor | int | bool | float]:
    head_dim = 512
    full_dim = head_dim * (2 if overlap else 1)
    state_rows = ratio * (2 if overlap else 1)
    num_slots = 4
    return {
        "kv_i": torch.zeros(batch_size, 1, full_dim, dtype=torch.float32),
        "score_i": torch.zeros(batch_size, 1, full_dim, dtype=torch.float32),
        "pos": torch.tensor([ratio - 1] * batch_size, dtype=torch.int64),
        "slot_ids": torch.arange(batch_size, dtype=torch.int64),
        "ratio": ratio,
        "head_dim": head_dim,
        "rope_head_dim": 64,
        "overlap": overlap,
        "ape": torch.zeros(ratio, full_dim, dtype=torch.float32),
        "norm_weight": torch.ones(head_dim, dtype=torch.float32),
        "freqs_cis": torch.ones(512, 32, dtype=torch.complex64),
        "kv_state": torch.zeros(num_slots, state_rows, full_dim, dtype=torch.float32),
        "score_state": torch.zeros(num_slots, state_rows, full_dim, dtype=torch.float32),
        "kv_cache": torch.zeros(num_slots, 8, head_dim, dtype=torch.bfloat16),
        "out": torch.zeros(batch_size, 1, head_dim, dtype=torch.bfloat16),
        "eps": 1e-6,
    }


def _indexer_batch_inputs(batch_size: int) -> dict[str, torch.Tensor | float]:
    num_slots = 4
    return {
        "kv_i": torch.zeros(batch_size, 1, 256, dtype=torch.float32),
        "score_i": torch.zeros(batch_size, 1, 256, dtype=torch.float32),
        "pos": torch.tensor([3] * batch_size, dtype=torch.int64),
        "slot_ids": torch.arange(batch_size, dtype=torch.int64),
        "ape": torch.zeros(4, 256, dtype=torch.float32),
        "norm_weight": torch.ones(128, dtype=torch.float32),
        "freqs_cis": torch.ones(512, 32, dtype=torch.complex64),
        "kv_state": torch.zeros(num_slots, 8, 256, dtype=torch.float32),
        "score_state": torch.zeros(num_slots, 8, 256, dtype=torch.float32),
        "kv_cache": torch.zeros(num_slots, 8, 128, dtype=torch.bfloat16),
        "out": torch.zeros(batch_size, 1, 128, dtype=torch.bfloat16),
        "eps": 1e-6,
    }


def _serial_main_b1_oracle(
    *,
    kv_i: torch.Tensor,
    score_i: torch.Tensor,
    positions: tuple[int, ...],
    slot_ids: tuple[int, ...],
    ratio: int,
    head_dim: int,
    overlap: bool,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    kv_cache: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    expected = torch.empty(len(positions), 1, head_dim, device=kv_i.device, dtype=torch.bfloat16)
    for row, (position, slot_id) in enumerate(zip(positions, slot_ids, strict=True)):
        row_out = torch.empty(1, 1, head_dim, device=kv_i.device, dtype=torch.bfloat16)
        result = fused_decode_postgemv(
            kv_i=kv_i[row : row + 1],
            score_i=score_i[row : row + 1],
            pos=torch.tensor([position], device=kv_i.device, dtype=torch.int64),
            ratio=ratio,
            head_dim=head_dim,
            rope_head_dim=64,
            overlap=overlap,
            ape=ape,
            norm_weight=norm_weight,
            freqs_cis=freqs_cis,
            kv_state=kv_state[slot_id : slot_id + 1],
            score_state=score_state[slot_id : slot_id + 1],
            kv_cache=kv_cache[slot_id : slot_id + 1],
            out=row_out,
            eps=eps,
        )
        expected[row : row + 1].copy_(result)
    return expected


def _serial_indexer_b1_oracle(
    *,
    kv_i: torch.Tensor,
    score_i: torch.Tensor,
    positions: tuple[int, ...],
    slot_ids: tuple[int, ...],
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    kv_cache: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    expected = torch.empty(len(positions), 1, 128, device=kv_i.device, dtype=torch.bfloat16)
    for row, (position, slot_id) in enumerate(zip(positions, slot_ids, strict=True)):
        row_out = torch.empty(1, 1, 128, device=kv_i.device, dtype=torch.bfloat16)
        result = fused_indexer_decode_postgemv(
            kv_i=kv_i[row : row + 1],
            score_i=score_i[row : row + 1],
            pos=torch.tensor([position], device=kv_i.device, dtype=torch.int64),
            ape=ape,
            norm_weight=norm_weight,
            freqs_cis=freqs_cis,
            kv_state=kv_state[slot_id : slot_id + 1],
            score_state=score_state[slot_id : slot_id + 1],
            kv_cache=kv_cache[slot_id : slot_id + 1],
            out=row_out,
            eps=eps,
        )
        expected[row : row + 1].copy_(result)
    return expected


def test_supports_fused_decode_postgemv_requires_exact_contract() -> None:
    assert supports_fused_decode_postgemv(
        ratio=4,
        rotate=False,
        quantize=False,
        device=torch.device("cuda"),
        batch_size=1,
        seq_len=1,
        head_dim=512,
        rope_head_dim=64,
    )
    assert not supports_fused_decode_postgemv(
        ratio=4,
        rotate=False,
        quantize=False,
        device=torch.device("cpu"),
        batch_size=1,
        seq_len=1,
        head_dim=512,
        rope_head_dim=64,
    )
    assert not supports_fused_decode_postgemv(
        ratio=4,
        rotate=True,
        quantize=False,
        device=torch.device("cuda"),
        batch_size=1,
        seq_len=1,
        head_dim=512,
        rope_head_dim=64,
    )
    assert not supports_fused_decode_postgemv(
        ratio=128,
        rotate=False,
        quantize=True,
        device=torch.device("cuda"),
        batch_size=1,
        seq_len=1,
        head_dim=512,
        rope_head_dim=64,
    )
    assert not supports_fused_decode_postgemv(
        ratio=4,
        rotate=False,
        quantize=False,
        device=torch.device("cuda"),
        batch_size=2,
        seq_len=1,
        head_dim=512,
        rope_head_dim=64,
    )
    assert not supports_fused_decode_postgemv(
        ratio=4,
        rotate=False,
        quantize=False,
        device=torch.device("cuda"),
        batch_size=1,
        seq_len=1,
        head_dim=256,
        rope_head_dim=64,
    )


def test_fused_decode_postgemv_rejects_non_cuda_inputs() -> None:
    with pytest.raises(ValueError, match="exact CUDA main-compressor contract"):
        fused_decode_postgemv(
            kv_i=torch.zeros(1, 1, 1024, dtype=torch.float32),
            score_i=torch.zeros(1, 1, 1024, dtype=torch.float32),
            pos=torch.tensor([3], dtype=torch.int64),
            ratio=4,
            head_dim=512,
            rope_head_dim=64,
            overlap=True,
            ape=torch.zeros(4, 1024, dtype=torch.float32),
            norm_weight=torch.ones(512, dtype=torch.float32),
            freqs_cis=torch.ones(16, 32, dtype=torch.complex64),
            kv_state=torch.zeros(1, 8, 1024, dtype=torch.float32),
            score_state=torch.zeros(1, 8, 1024, dtype=torch.float32),
            kv_cache=torch.zeros(1, 8, 512, dtype=torch.bfloat16),
            out=torch.zeros(1, 1, 512, dtype=torch.bfloat16),
            eps=1e-6,
        )


def test_supports_fused_decode_postgemv_batch_requires_exact_contract() -> None:
    assert supports_fused_decode_postgemv_batch(
        ratio=4,
        rotate=False,
        quantize=False,
        device=torch.device("cuda"),
        batch_size=2,
        seq_len=1,
        head_dim=512,
        rope_head_dim=64,
    )
    assert supports_fused_decode_postgemv_batch(
        ratio=128,
        rotate=False,
        quantize=False,
        device=torch.device("cuda"),
        batch_size=4,
        seq_len=1,
        head_dim=512,
        rope_head_dim=64,
    )
    assert not supports_fused_decode_postgemv_batch(
        ratio=4,
        rotate=False,
        quantize=False,
        device=torch.device("cpu"),
        batch_size=2,
        seq_len=1,
        head_dim=512,
        rope_head_dim=64,
    )
    assert not supports_fused_decode_postgemv_batch(
        ratio=4,
        rotate=False,
        quantize=False,
        device=torch.device("cuda"),
        batch_size=3,
        seq_len=1,
        head_dim=512,
        rope_head_dim=64,
    )


def test_fused_decode_postgemv_batch_rejects_noncontiguous_inputs_before_cuda_check() -> None:
    inputs = _main_batch_inputs(batch_size=2, ratio=4, overlap=True)
    inputs["kv_i"] = torch.empty_strided((2, 1, 1024), (2048, 1024, 1), dtype=torch.float32)
    with pytest.raises(ValueError, match="kv_i must be contiguous"):
        fused_decode_postgemv_batch(**inputs)


def test_fused_decode_postgemv_batch_accepts_offset_integer_views() -> None:
    inputs = _main_batch_inputs(batch_size=2, ratio=4, overlap=True)
    packed = torch.tensor([91, 92, 3, 4, 1, 0], dtype=torch.int64)
    inputs["pos"] = packed[2:4]
    inputs["slot_ids"] = packed[4:6]

    assert inputs["pos"].is_contiguous() and inputs["pos"].storage_offset() != 0
    assert inputs["slot_ids"].is_contiguous() and inputs["slot_ids"].storage_offset() != 0
    with pytest.raises(ValueError, match="exact CUDA main-compressor contract"):
        fused_decode_postgemv_batch(**inputs)


@pytest.mark.parametrize(("ratio", "overlap"), [(4, True), (128, False)])
def test_compile_fused_decode_postgemv_batch_sm120_offline(
    ratio: int,
    overlap: bool,
) -> None:
    kernel = compile_fused_decode_postgemv_batch_sm120(ratio=ratio, overlap=overlap)
    assert kernel is not None


def test_supports_fused_indexer_decode_postgemv_requires_exact_contract() -> None:
    expected = {
        "ratio": 4,
        "rotate": True,
        "quantize": True,
        "device": torch.device("cuda"),
        "batch_size": 1,
        "seq_len": 1,
        "head_dim": 128,
        "rope_head_dim": 64,
    }
    assert supports_fused_indexer_decode_postgemv(**expected)
    for key, value in {
        "ratio": 128,
        "rotate": False,
        "quantize": False,
        "device": torch.device("cpu"),
        "batch_size": 2,
        "seq_len": 2,
        "head_dim": 512,
        "rope_head_dim": 32,
    }.items():
        rejected = dict(expected)
        rejected[key] = value
        assert not supports_fused_indexer_decode_postgemv(**rejected), key


def test_supports_fused_indexer_decode_postgemv_batch_requires_exact_contract() -> None:
    assert supports_fused_indexer_decode_postgemv_batch(
        ratio=4,
        rotate=True,
        quantize=True,
        device=torch.device("cuda"),
        batch_size=4,
        seq_len=1,
        head_dim=128,
        rope_head_dim=64,
    )
    assert not supports_fused_indexer_decode_postgemv_batch(
        ratio=4,
        rotate=True,
        quantize=True,
        device=torch.device("cuda"),
        batch_size=3,
        seq_len=1,
        head_dim=128,
        rope_head_dim=64,
    )


def test_fused_indexer_decode_postgemv_rejects_non_cuda_inputs() -> None:
    with pytest.raises(ValueError, match="exact CUDA contract"):
        fused_indexer_decode_postgemv(
            kv_i=torch.zeros(1, 1, 256, dtype=torch.float32),
            score_i=torch.zeros(1, 1, 256, dtype=torch.float32),
            pos=torch.tensor([3], dtype=torch.int64),
            ape=torch.zeros(4, 256, dtype=torch.float32),
            norm_weight=torch.ones(128, dtype=torch.float32),
            freqs_cis=torch.ones(16, 32, dtype=torch.complex64),
            kv_state=torch.zeros(1, 8, 256, dtype=torch.float32),
            score_state=torch.full((1, 8, 256), float("-inf"), dtype=torch.float32),
            kv_cache=torch.zeros(1, 8, 128, dtype=torch.bfloat16),
            out=torch.zeros(1, 1, 128, dtype=torch.bfloat16),
            eps=1e-6,
        )


def test_compile_fused_indexer_decode_postgemv_batch_sm120_offline() -> None:
    decode_kernel, migrate_kernel = compile_fused_indexer_decode_postgemv_batch_sm120()
    assert decode_kernel is not None
    assert migrate_kernel is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs exclusive GPU")
@pytest.mark.parametrize(
    ("ratio", "overlap", "positions"),
    [
        (4, True, (3, 4, 5, 7, 8)),
        (128, False, (127, 128, 129, 255, 256)),
    ],
)
def test_fused_decode_postgemv_matches_torch_state_machine(
    ratio: int,
    overlap: bool,
    positions: tuple[int, ...],
) -> None:
    device = torch.device("cuda")
    head_dim = 512
    full_dim = head_dim * (2 if overlap else 1)
    state_rows = ratio * (2 if overlap else 1)
    generator = torch.Generator(device=device).manual_seed(1900 + ratio)
    ape = torch.randn(ratio, full_dim, generator=generator, device=device, dtype=torch.float32)
    norm_weight = torch.randn(head_dim, generator=generator, device=device, dtype=torch.float32)
    freqs_cis = precompute_freqs_cis(
        64,
        384,
        original_seq_len=0,
        base=160000.0,
        factor=1.0,
        beta_fast=32,
        beta_slow=1,
        device=device,
    )
    kernel_kv_state = torch.randn(
        1, state_rows, full_dim, generator=generator, device=device, dtype=torch.float32
    )
    kernel_score_state = torch.randn(
        1, state_rows, full_dim, generator=generator, device=device, dtype=torch.float32
    )
    kernel_cache = torch.randn(
        1, 4, head_dim, generator=generator, device=device, dtype=torch.bfloat16
    )
    oracle_kv_state = kernel_kv_state.clone()
    oracle_score_state = kernel_score_state.clone()
    oracle_cache = kernel_cache.clone()
    out = torch.empty(1, 1, head_dim, device=device, dtype=torch.bfloat16)

    for position in positions:
        kv_i = torch.randn(1, 1, full_dim, generator=generator, device=device, dtype=torch.float32)
        score_i = torch.randn(
            1, 1, full_dim, generator=generator, device=device, dtype=torch.float32
        )
        expected = _oracle_step(
            kv_i=kv_i,
            score_i=score_i,
            position=position,
            ratio=ratio,
            overlap=overlap,
            ape=ape,
            norm_weight=norm_weight,
            freqs_cis=freqs_cis,
            kv_state=oracle_kv_state,
            score_state=oracle_score_state,
            kv_cache=oracle_cache,
            eps=1e-6,
        )
        actual = fused_decode_postgemv(
            kv_i=kv_i,
            score_i=score_i,
            pos=torch.tensor([position], device=device, dtype=torch.int64),
            ratio=ratio,
            head_dim=head_dim,
            rope_head_dim=64,
            overlap=overlap,
            ape=ape,
            norm_weight=norm_weight,
            freqs_cis=freqs_cis,
            kv_state=kernel_kv_state,
            score_state=kernel_score_state,
            kv_cache=kernel_cache,
            out=out,
            eps=1e-6,
        )
        torch.cuda.synchronize()

        assert torch.equal(actual, expected), position
        assert torch.equal(kernel_kv_state, oracle_kv_state), position
        assert torch.equal(kernel_score_state, oracle_score_state), position
        assert torch.equal(kernel_cache, oracle_cache), position


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs exclusive GPU")
@pytest.mark.parametrize(
    ("ratio", "overlap", "positions", "slot_ids"),
    [
        (4, True, (3, 4), (1, 0)),
        (4, True, (3, 4, 7, 8), (1, 0, 3, 2)),
        (128, False, (127, 128), (1, 0)),
        (128, False, (127, 128, 255, 256), (1, 0, 3, 2)),
    ],
)
def test_fused_decode_postgemv_batch_matches_serial_b1_oracle(
    ratio: int,
    overlap: bool,
    positions: tuple[int, ...],
    slot_ids: tuple[int, ...],
) -> None:
    device = torch.device("cuda")
    head_dim = 512
    full_dim = head_dim * (2 if overlap else 1)
    state_rows = ratio * (2 if overlap else 1)
    batch_size = len(positions)
    generator = torch.Generator(device=device).manual_seed(7100 + ratio + batch_size)
    ape = torch.randn(ratio, full_dim, generator=generator, device=device, dtype=torch.float32)
    norm_weight = torch.randn(head_dim, generator=generator, device=device, dtype=torch.float32)
    freqs_cis = _make_freqs(device)
    batch_kv_state = torch.randn(
        4, state_rows, full_dim, generator=generator, device=device, dtype=torch.float32
    )
    batch_score_state = torch.randn(
        4, state_rows, full_dim, generator=generator, device=device, dtype=torch.float32
    )
    batch_cache = torch.randn(
        4, 8, head_dim, generator=generator, device=device, dtype=torch.bfloat16
    )
    serial_kv_state = batch_kv_state.clone()
    serial_score_state = batch_score_state.clone()
    serial_cache = batch_cache.clone()
    kv_i = torch.randn(
        batch_size, 1, full_dim, generator=generator, device=device, dtype=torch.float32
    )
    score_i = torch.randn(
        batch_size, 1, full_dim, generator=generator, device=device, dtype=torch.float32
    )
    out = torch.empty(batch_size, 1, head_dim, device=device, dtype=torch.bfloat16)

    expected = _serial_main_b1_oracle(
        kv_i=kv_i,
        score_i=score_i,
        positions=positions,
        slot_ids=slot_ids,
        ratio=ratio,
        head_dim=head_dim,
        overlap=overlap,
        ape=ape,
        norm_weight=norm_weight,
        freqs_cis=freqs_cis,
        kv_state=serial_kv_state,
        score_state=serial_score_state,
        kv_cache=serial_cache,
        eps=1e-6,
    )
    actual = fused_decode_postgemv_batch(
        kv_i=kv_i,
        score_i=score_i,
        pos=torch.tensor(positions, device=device, dtype=torch.int64),
        slot_ids=torch.tensor(slot_ids, device=device, dtype=torch.int64),
        ratio=ratio,
        head_dim=head_dim,
        rope_head_dim=64,
        overlap=overlap,
        ape=ape,
        norm_weight=norm_weight,
        freqs_cis=freqs_cis,
        kv_state=batch_kv_state,
        score_state=batch_score_state,
        kv_cache=batch_cache,
        out=out,
        eps=1e-6,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    assert torch.equal(batch_kv_state, serial_kv_state)
    assert torch.equal(batch_score_state, serial_score_state)
    assert torch.equal(batch_cache, serial_cache)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs exclusive GPU")
def test_hadamard_fp4_query_matches_torch_exactly() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260905)
    query = torch.randn(
        3,
        1,
        64,
        128,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected = fp4_act_quant_simulate(hadamard_transform(query, 128**-0.5), 32)
    assert torch.equal(hadamard_fp4_query(query), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs exclusive GPU")
def test_fused_indexer_decode_postgemv_matches_torch_state_machine() -> None:
    """Acceptance gate for production wiring; exercises sentinel + overlap migration."""
    device = torch.device("cuda")
    ratio = 4
    head_dim = 128
    full_dim = 256
    generator = torch.Generator(device=device).manual_seed(4137)
    ape = torch.randn(ratio, full_dim, generator=generator, device=device)
    norm_weight = torch.randn(head_dim, generator=generator, device=device)
    freqs_cis = precompute_freqs_cis(
        64,
        384,
        original_seq_len=0,
        base=160000.0,
        factor=1.0,
        beta_fast=32,
        beta_slow=1,
        device=device,
    )
    kernel_kv_state = torch.randn(1, 8, full_dim, generator=generator, device=device)
    kernel_score_state = torch.full(
        (1, 8, full_dim), float("-inf"), device=device, dtype=torch.float32
    )
    kernel_score_state[:, :4] = torch.randn(1, 4, full_dim, generator=generator, device=device)
    kernel_cache = torch.randn(
        1, 4, head_dim, generator=generator, device=device, dtype=torch.bfloat16
    )
    oracle_kv_state = kernel_kv_state.clone()
    oracle_score_state = kernel_score_state.clone()
    oracle_cache = kernel_cache.clone()
    out = torch.empty(1, 1, head_dim, device=device, dtype=torch.bfloat16)

    for position in (3, 4, 5, 7, 8):
        kv_i = torch.randn(1, 1, full_dim, generator=generator, device=device)
        score_i = torch.randn(1, 1, full_dim, generator=generator, device=device)
        expected = _oracle_step(
            kv_i=kv_i,
            score_i=score_i,
            position=position,
            ratio=ratio,
            overlap=True,
            ape=ape,
            norm_weight=norm_weight,
            freqs_cis=freqs_cis,
            kv_state=oracle_kv_state,
            score_state=oracle_score_state,
            kv_cache=oracle_cache,
            eps=1e-6,
            rotate_quantize=True,
        )
        actual = fused_indexer_decode_postgemv(
            kv_i=kv_i,
            score_i=score_i,
            pos=torch.tensor([position], device=device, dtype=torch.int64),
            ape=ape,
            norm_weight=norm_weight,
            freqs_cis=freqs_cis,
            kv_state=kernel_kv_state,
            score_state=kernel_score_state,
            kv_cache=kernel_cache,
            out=out,
            eps=1e-6,
        )
        torch.cuda.synchronize()

        assert torch.equal(actual, expected), position
        assert torch.equal(kernel_kv_state, oracle_kv_state), position
        assert torch.equal(kernel_score_state, oracle_score_state), position
        assert torch.equal(kernel_cache, oracle_cache), position


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs exclusive GPU")
def test_fused_indexer_postgemv_seq_matches_per_token_kernel_bit_exactly() -> None:
    device = torch.device("cuda")
    ratio, head_dim, full_dim, rows = 4, 128, 256, 8
    generator = torch.Generator(device=device).manual_seed(20260814)
    ape = torch.randn(ratio, full_dim, generator=generator, device=device)
    norm_weight = torch.randn(head_dim, generator=generator, device=device)
    freqs_cis = precompute_freqs_cis(
        64,
        384,
        original_seq_len=0,
        base=160000.0,
        factor=1.0,
        beta_fast=32,
        beta_slow=1,
        device=device,
    )
    kv = torch.randn(1, rows, full_dim, generator=generator, device=device)
    score = torch.randn(1, rows, full_dim, generator=generator, device=device)
    initial_kv_state = torch.randn(1, 8, full_dim, generator=generator, device=device)
    initial_score_state = torch.full(
        (1, 8, full_dim), float("-inf"), device=device, dtype=torch.float32
    )
    initial_score_state[:, :4] = torch.randn(
        1, 4, full_dim, generator=generator, device=device
    )
    initial_cache = torch.randn(
        1, rows // ratio, head_dim, generator=generator, device=device, dtype=torch.bfloat16
    )

    old_kv_state = initial_kv_state.clone()
    old_score_state = initial_score_state.clone()
    old_cache = initial_cache.clone()
    old_rows = []
    scratch = torch.empty(1, 1, head_dim, device=device, dtype=torch.bfloat16)
    for position in range(rows):
        result = fused_indexer_decode_postgemv(
            kv_i=kv[:, position : position + 1],
            score_i=score[:, position : position + 1],
            pos=torch.tensor([position], device=device, dtype=torch.int64),
            ape=ape,
            norm_weight=norm_weight,
            freqs_cis=freqs_cis,
            kv_state=old_kv_state,
            score_state=old_score_state,
            kv_cache=old_cache,
            out=scratch,
            eps=1e-6,
        )
        if position % ratio == ratio - 1:
            old_rows.append(result.clone())

    new_kv_state = initial_kv_state.clone()
    new_score_state = initial_score_state.clone()
    new_cache = initial_cache.clone()
    new_out = torch.empty(1, rows // ratio, head_dim, device=device, dtype=torch.bfloat16)
    actual = fused_indexer_decode_postgemv_seq(
        kv=kv,
        score=score,
        pos0=torch.tensor([0], device=device, dtype=torch.int64),
        host_start_pos=0,
        ape=ape,
        norm_weight=norm_weight,
        freqs_cis=freqs_cis,
        kv_state=new_kv_state,
        score_state=new_score_state,
        kv_cache=new_cache,
        out=new_out,
        eps=1e-6,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, torch.cat(old_rows, dim=1))
    assert torch.equal(new_kv_state, old_kv_state)
    assert torch.equal(new_score_state, old_score_state)
    assert torch.equal(new_cache, old_cache)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs exclusive GPU")
@pytest.mark.parametrize(
    ("positions", "slot_ids"),
    [
        ((3, 4), (1, 0)),
        ((3, 4, 7, 8), (1, 0, 3, 2)),
    ],
)
def test_fused_indexer_decode_postgemv_batch_matches_serial_b1_oracle(
    positions: tuple[int, ...],
    slot_ids: tuple[int, ...],
) -> None:
    device = torch.device("cuda")
    full_dim = 256
    batch_size = len(positions)
    generator = torch.Generator(device=device).manual_seed(9100 + batch_size)
    ape = torch.randn(4, full_dim, generator=generator, device=device)
    norm_weight = torch.randn(128, generator=generator, device=device)
    freqs_cis = _make_freqs(device)
    batch_kv_state = torch.randn(4, 8, full_dim, generator=generator, device=device)
    batch_score_state = torch.full(
        (4, 8, full_dim), float("-inf"), device=device, dtype=torch.float32
    )
    batch_score_state[:, :4] = torch.randn(4, 4, full_dim, generator=generator, device=device)
    batch_cache = torch.randn(4, 8, 128, generator=generator, device=device, dtype=torch.bfloat16)
    serial_kv_state = batch_kv_state.clone()
    serial_score_state = batch_score_state.clone()
    serial_cache = batch_cache.clone()
    kv_i = torch.randn(batch_size, 1, full_dim, generator=generator, device=device)
    score_i = torch.randn(batch_size, 1, full_dim, generator=generator, device=device)
    out = torch.empty(batch_size, 1, 128, device=device, dtype=torch.bfloat16)

    expected = _serial_indexer_b1_oracle(
        kv_i=kv_i,
        score_i=score_i,
        positions=positions,
        slot_ids=slot_ids,
        ape=ape,
        norm_weight=norm_weight,
        freqs_cis=freqs_cis,
        kv_state=serial_kv_state,
        score_state=serial_score_state,
        kv_cache=serial_cache,
        eps=1e-6,
    )
    actual = fused_indexer_decode_postgemv_batch(
        kv_i=kv_i,
        score_i=score_i,
        pos=torch.tensor(positions, device=device, dtype=torch.int64),
        slot_ids=torch.tensor(slot_ids, device=device, dtype=torch.int64),
        ape=ape,
        norm_weight=norm_weight,
        freqs_cis=freqs_cis,
        kv_state=batch_kv_state,
        score_state=batch_score_state,
        kv_cache=batch_cache,
        out=out,
        eps=1e-6,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)
    assert torch.equal(batch_kv_state, serial_kv_state)
    assert torch.equal(batch_score_state, serial_score_state)
    assert torch.equal(batch_cache, serial_cache)
