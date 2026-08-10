"""FFN-half tests: expert numerics vs the reference Expert, MoE routing
wiring on a small synthetic config."""

from __future__ import annotations

import importlib.util
import random
import struct
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from runtime.loading.gguf import load_gguf_tensors  # noqa: E402
from runtime.model.dsv4_config import Dsv4Config  # noqa: E402
from runtime.model.dsv4_model import (  # noqa: E402
    DenseLinear,
    Dsv4MoE,
    PackedIQ2_XSExperts,
    PackedQ8_0Linear,
    rms_norm,
    swiglu,
)
from runtime.model.dsv4_quant import dequantize_iq2_xs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "notes" / "dsv4flash-ref" / "inference"
REAL_GGUF = Path(
    "/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf"
)


def _load_reference_model_module():
    if "kernel" not in sys.modules:
        kernel_spec = importlib.util.spec_from_file_location("kernel", REFERENCE_DIR / "kernel.py")
        kernel = importlib.util.module_from_spec(kernel_spec)
        sys.modules["kernel"] = kernel
        kernel_spec.loader.exec_module(kernel)
    spec = importlib.util.spec_from_file_location("dsv4_ref_model", REFERENCE_DIR / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_config() -> Dsv4Config:
    # hidden must be a multiple of the 256-element IQ2_XS block (contiguous dim)
    return Dsv4Config(
        vocab_size=16,
        hidden_size=256,
        num_layers=3,
        compress_ratios=(0, 4, 128),
        n_routed_experts=4,
        n_activated_experts=2,
        moe_intermediate_size=256,  # >= 256: IQ2_XS block constraint on the contiguous dim
        n_hash_layers=1,
    )


def decode_batch_config(*, n_routed_experts: int = 8) -> Dsv4Config:
    return Dsv4Config(
        vocab_size=32,
        hidden_size=256,
        num_layers=3,
        compress_ratios=(0, 4, 128),
        n_routed_experts=n_routed_experts,
        n_activated_experts=6,
        moe_intermediate_size=256,
        n_hash_layers=3,
    )


def valid_q8_0_blocks(rng: random.Random, n_blocks: int) -> bytes:
    out = bytearray()
    for _ in range(n_blocks):
        out += struct.pack("<H", rng.randrange(0x1C00, 0x2400))  # small finite d
        out += struct.pack("<32b", *(rng.randrange(-127, 128) for _ in range(32)))
    return bytes(out)


def valid_iq2_xs_blocks(rng: random.Random, n_blocks: int) -> bytes:
    out = bytearray()
    for _ in range(n_blocks):
        out += struct.pack("<H", rng.randrange(0x1C00, 0x2400))
        out += struct.pack("<32H", *(rng.getrandbits(16) for _ in range(32)))
        out += bytes(rng.getrandbits(8) for _ in range(8))
    return bytes(out)


def fill_moe(moe: Dsv4MoE, seed: int) -> None:
    rng = random.Random(seed)
    for container in (moe.gate_exps, moe.up_exps, moe.down_exps):
        data = valid_iq2_xs_blocks(rng, container.packed.numel() // 74)
        container.packed.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
    for linear in (moe.shared_w1, moe.shared_w3, moe.shared_w2):
        data = valid_q8_0_blocks(rng, linear.packed.numel() // 34)
        linear.packed = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    # ffn_gate_inp is BF16 in the file (DenseLinear container)
    gate_w = moe.gate.weight.weight
    gate_w.copy_(
        torch.randn(gate_w.shape, generator=torch.Generator().manual_seed(seed + 1)) * 0.05
    )
    if hasattr(moe.gate, "bias"):
        moe.gate.bias.copy_(
            torch.randn(moe.gate.bias.shape, generator=torch.Generator().manual_seed(seed))
        )
    else:
        moe.gate.tid2eid.copy_(
            torch.randint(0, moe.config.n_routed_experts, moe.gate.tid2eid.shape)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_dense_gate_decode_rows_match_b1_bit_exactly() -> None:
    gate = DenseLinear(256, 4096, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(20260810)
    gate.weight.copy_(
        torch.randn(gate.weight.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
        * 0.01
    )
    row = torch.randn(1, 4096, generator=generator, device="cuda", dtype=torch.bfloat16)

    expected = torch.nn.functional.linear(row.float(), gate.weight.float())
    actual_b1 = gate(row)
    actual_b4 = gate(row.repeat(4, 1))

    assert torch.equal(actual_b1, expected)
    assert torch.equal(actual_b4[0], actual_b1[0])
    assert torch.equal(actual_b4, actual_b1.expand_as(actual_b4))


def reference_moe_output(moe: Dsv4MoE, x: torch.Tensor, input_ids) -> torch.Tensor:
    """Manual transcription of the reference MoE semantics over dequanted
    weights -- the oracle for the module's gather/scatter wiring."""
    cfg = moe.config
    flat = x.reshape(-1, x.shape[-1]).float()
    gate_w = moe.gate.weight.weight.float()
    scores = torch.nn.functional.softplus(flat @ gate_w.t()).sqrt()
    if moe.gate.hashed:
        indices = moe.gate.tid2eid[input_ids.reshape(-1)].to(torch.int64)
    else:
        indices = (scores + moe.gate.bias).topk(cfg.n_activated_experts, dim=-1)[1]
    weights = scores.gather(1, indices)
    weights = weights / weights.sum(-1, keepdim=True) * cfg.route_scale

    y = torch.zeros_like(flat)
    for expert_id in range(cfg.n_routed_experts):
        token_idx, top_slot = torch.where(indices == expert_id)
        if token_idx.numel() == 0:
            continue
        w1 = moe.gate_exps.expert_weight(expert_id)
        w3 = moe.up_exps.expert_weight(expert_id)
        w2 = moe.down_exps.expert_weight(expert_id)
        xs = flat[token_idx]
        h = swiglu(xs @ w1.t(), xs @ w3.t(), cfg.swiglu_limit)
        y.index_add_(0, token_idx, (h @ w2.t()) * weights[token_idx, top_slot].unsqueeze(-1))
    shared = swiglu(moe.shared_w1(flat), moe.shared_w3(flat), cfg.swiglu_limit)
    y = y + moe.shared_w2(shared)
    return y.to(x.dtype).reshape(*x.shape[:-1], y.shape[-1])


def test_moe_matches_reference_wiring_scored() -> None:
    cfg = small_config()
    moe = Dsv4MoE(cfg, hashed=False)
    fill_moe(moe, seed=3)
    gen = torch.Generator().manual_seed(4)
    x = torch.randn(5, 256, generator=gen) * 0.1
    input_ids = torch.randint(0, 16, (5,), generator=gen)
    expected = reference_moe_output(moe, x, input_ids)
    actual = moe(x, input_ids)
    assert torch.equal(actual, expected)


def test_moe_matches_reference_wiring_hashed() -> None:
    cfg = small_config()
    moe = Dsv4MoE(cfg, hashed=True)
    fill_moe(moe, seed=5)
    gen = torch.Generator().manual_seed(6)
    x = torch.randn(7, 256, generator=gen) * 0.1
    input_ids = torch.randint(0, 16, (7,), generator=gen)
    expected = reference_moe_output(moe, x, input_ids)
    actual = moe(x, input_ids)
    assert torch.equal(actual, expected)
    # hash contract: selection comes from the table, weights from the logits
    _, indices = moe.gate(x, input_ids)
    assert torch.equal(indices, moe.gate.tid2eid[input_ids].to(torch.int64))


def test_forward_decode_batch_matches_reference_scored_cpu() -> None:
    cfg = decode_batch_config()
    moe = Dsv4MoE(cfg, hashed=False)
    fill_moe(moe, seed=13)
    gen = torch.Generator().manual_seed(14)
    x = torch.randn(4, 1, cfg.hidden_size, generator=gen, dtype=torch.bfloat16) * 0.1
    input_ids = torch.randint(0, cfg.vocab_size, (4,), generator=gen)
    expected = torch.cat(
        [moe(x[i : i + 1], input_ids[i : i + 1]) for i in range(x.shape[0])],
        dim=0,
    )
    actual = moe.forward_decode_batch(x, input_ids)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert torch.equal(actual, expected)


def test_forward_decode_batch_matches_reference_hashed_cpu() -> None:
    cfg = decode_batch_config()
    moe = Dsv4MoE(cfg, hashed=True)
    fill_moe(moe, seed=15)
    moe.gate.tid2eid.zero_()
    moe.gate.tid2eid[3] = torch.tensor([5, 1, 5, 2, 3, 0], dtype=torch.int32)
    moe.gate.tid2eid[7] = torch.tensor([4, 4, 1, 0, 2, 5], dtype=torch.int32)
    moe.gate.tid2eid[9] = torch.tensor([2, 6, 2, 1, 7, 3], dtype=torch.int32)
    moe.gate.tid2eid[11] = torch.tensor([7, 0, 7, 5, 4, 6], dtype=torch.int32)
    gen = torch.Generator().manual_seed(16)
    x = torch.randn(4, 1, cfg.hidden_size, generator=gen, dtype=torch.bfloat16) * 0.1
    input_ids = torch.tensor([[3], [7], [9], [11]], dtype=torch.int64)
    flat_ids = input_ids.reshape(-1)
    expected = torch.cat([moe(x[i : i + 1], flat_ids[i : i + 1]) for i in range(x.shape[0])], dim=0)
    actual = moe.forward_decode_batch(x, input_ids)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert torch.equal(actual, expected)
    _, indices = moe.gate(x[:, 0], input_ids.reshape(-1))
    assert torch.equal(indices, moe.gate.tid2eid[input_ids.reshape(-1)].to(torch.int64))


def test_forward_decode_batch_rejects_non_decode_shapes() -> None:
    cfg = decode_batch_config()
    moe = Dsv4MoE(cfg, hashed=False)
    bad_rank = torch.zeros(2, cfg.hidden_size, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"\[B, 1, H\]"):
        moe.forward_decode_batch(bad_rank, torch.zeros(2, dtype=torch.int64))
    bad_seqlen = torch.zeros(2, 2, cfg.hidden_size, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="seqlen=1"):
        moe.forward_decode_batch(bad_seqlen, torch.zeros(2, dtype=torch.int64))
    good_x = torch.zeros(2, 1, cfg.hidden_size, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="input_ids 3 != batch 2"):
        moe.forward_decode_batch(good_x, torch.zeros(3, dtype=torch.int64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
@pytest.mark.parametrize("hashed", [False, True])
def test_cuda_forward_decode_batch_matches_concatenated_b1_oracle(hashed: bool) -> None:
    # The fused hashed router intentionally implements only the production
    # DSV4 contract (256 experts, top-6).  Keep the cheap 8-expert fixture for
    # CPU and mocked routing tests, but exercise the real CUDA route surface
    # here so the parity gate cannot silently fall back to a synthetic shape.
    cfg = decode_batch_config(n_routed_experts=256)
    moe = Dsv4MoE(cfg, hashed=hashed, device="cuda")
    fill_moe(moe, seed=17 if hashed else 18)
    if hashed:
        moe.gate.tid2eid.zero_()
        moe.gate.tid2eid[2] = torch.tensor([6, 1, 6, 2, 3, 0], dtype=torch.int32, device="cuda")
        moe.gate.tid2eid[5] = torch.tensor([4, 4, 7, 0, 2, 5], dtype=torch.int32, device="cuda")
        moe.gate.tid2eid[8] = torch.tensor([2, 6, 2, 1, 7, 3], dtype=torch.int32, device="cuda")
        moe.gate.tid2eid[13] = torch.tensor([7, 0, 7, 5, 4, 6], dtype=torch.int32, device="cuda")
        input_ids = torch.tensor([2, 5, 8, 13], dtype=torch.int64, device="cuda")
    else:
        input_ids = torch.tensor([2, 5, 8, 13], dtype=torch.int64, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(19 if hashed else 20)
    x = (
        torch.randn(4, 1, cfg.hidden_size, generator=generator, device="cuda", dtype=torch.float32)
        * 0.1
    ).to(torch.bfloat16)
    expected = torch.cat(
        [moe(x[i : i + 1], input_ids[i : i + 1]) for i in range(x.shape[0])],
        dim=0,
    )
    actual = moe.forward_decode_batch(x, input_ids)
    # Changing the launch M from four independent M=1 calls to one M=4 call
    # can move one tiny BF16 output by a single representable step (observed
    # max 4.77e-7 on the hashed fixture).  Route ids/order are covered by the
    # strict contract test below; keep the numerical gate tight without
    # pretending the tensor-core launch shape is bitwise invariant.
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_cuda_forward_decode_batch_preserves_token_major_hash_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runtime.kernels.dsv4_iq2xs_gemm as iq2xs_kernels

    cfg = decode_batch_config()
    moe = Dsv4MoE(cfg, hashed=True, device="cuda")

    class FixedGate(torch.nn.Module):
        def forward(self, x: torch.Tensor, _input_ids):
            indices = torch.tensor(
                [
                    [5, 1, 5, 2, 3, 0],
                    [7, 0, 7, 5, 4, 6],
                ],
                dtype=torch.int64,
                device=x.device,
            )
            weights = torch.full(indices.shape, 0.25, dtype=torch.float32, device=x.device)
            return weights, indices

    seen: dict[str, torch.Tensor] = {}

    def fake_dual(
        x: torch.Tensor,
        _packed_gate_all: torch.Tensor,
        _packed_up_all: torch.Tensor,
        expert_ids: torch.Tensor,
        *,
        rows: int,
        **_kwargs,
    ) -> torch.Tensor:
        seen["dual_x"] = x.clone()
        seen["dual_eids"] = expert_ids.clone()
        return torch.zeros(expert_ids.numel(), 1, rows, dtype=torch.bfloat16, device=x.device)

    def fake_batch(_self, exps, eids: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        seen["down_eids"] = eids.clone()
        return torch.zeros(
            eids.numel(),
            xs.shape[1],
            exps.rows,
            dtype=torch.float32,
            device=xs.device,
        )

    def fake_shared(_self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)

    moe.gate = FixedGate()
    monkeypatch.setattr(iq2xs_kernels, "iq2xs_dequant_gemm_batch_indexed_dual_swiglu_b1", fake_dual)
    monkeypatch.setattr(Dsv4MoE, "_batch_expert_gemm", fake_batch)
    monkeypatch.setattr(Dsv4MoE, "_shared_forward", fake_shared)

    x = (
        torch.arange(2 * cfg.hidden_size, dtype=torch.float32, device="cuda")
        .reshape(2, 1, cfg.hidden_size)
        .to(torch.bfloat16)
    )
    input_ids = torch.tensor([3, 11], dtype=torch.int64, device="cuda")
    expected_eids = torch.tensor(
        [5, 1, 5, 2, 3, 0, 7, 0, 7, 5, 4, 6], dtype=torch.int64, device="cuda"
    )
    expected_xs = (
        x[:, 0]
        .unsqueeze(1)
        .expand(2, cfg.n_activated_experts, cfg.hidden_size)
        .reshape(2 * cfg.n_activated_experts, 1, cfg.hidden_size)
    )

    output = moe.forward_decode_batch(x, input_ids)

    assert torch.count_nonzero(output) == 0
    assert torch.equal(seen["dual_eids"], expected_eids)
    assert torch.equal(seen["down_eids"], expected_eids)
    assert torch.equal(seen["dual_x"], expected_xs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_cuda_prefill_keeps_batched_expert_ids_on_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import runtime.kernels.dsv4_iq2xs_gemm as iq2xs_kernels

    cfg = small_config()
    moe = Dsv4MoE(cfg, hashed=False, device="cuda")

    class FixedGate(torch.nn.Module):
        def forward(self, x: torch.Tensor, _input_ids):
            indices = torch.tensor([[0, 1], [1, 2], [0, 2]], dtype=torch.int64, device=x.device)
            weights = torch.full(indices.shape, 0.5, dtype=torch.float32, device=x.device)
            return weights, indices

    seen: dict[str, torch.Tensor] = {}

    def fake_dual(
        x: torch.Tensor,
        _packed_gate_all: torch.Tensor,
        _packed_up_all: torch.Tensor,
        expert_ids: torch.Tensor,
        *,
        rows: int,
        **_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seen["dual_eids"] = expert_ids.clone()
        seen["dual_x"] = x.clone()
        output = torch.zeros(expert_ids.numel(), 1, rows, dtype=torch.float32, device=x.device)
        return output, output.clone()

    def fake_batch(_self, exps, eids: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        assert isinstance(eids, torch.Tensor)
        assert eids.device == xs.device
        seen["down_eids"] = eids.clone()
        return torch.zeros(
            xs.shape[0], xs.shape[1], exps.rows, dtype=torch.float32, device=xs.device
        )

    def fake_shared(_self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    moe.gate = FixedGate()
    monkeypatch.setattr(iq2xs_kernels, "iq2xs_dequant_gemm_batch_indexed_dual", fake_dual)
    monkeypatch.setattr(Dsv4MoE, "_batch_expert_gemm", fake_batch)
    monkeypatch.setattr(Dsv4MoE, "_shared_forward", fake_shared)

    output = moe(
        torch.ones(3, cfg.hidden_size, dtype=torch.bfloat16, device="cuda"),
        torch.tensor([1, 2, 3], device="cuda"),
    )

    assert torch.count_nonzero(output) == 0
    expected_eids = torch.tensor([0, 1, 1, 2, 0, 2], dtype=torch.int64, device="cuda")
    assert torch.equal(seen["dual_eids"], expected_eids)
    assert torch.equal(seen["down_eids"], expected_eids)
    assert seen["dual_x"].shape == (6, 1, cfg.hidden_size)


def test_rms_norm_matches_reference_formula() -> None:
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(4, 64, generator=gen, dtype=torch.bfloat16)  # rms_norm is dim-agnostic
    w = torch.randn(64, generator=gen)
    expected_f = x.float()
    expected = (
        w * (expected_f * torch.rsqrt(expected_f.square().mean(-1, keepdim=True) + 1e-6))
    ).to(torch.bfloat16)
    assert torch.equal(rms_norm(x, w, 1e-6), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")
def test_cuda_rms_norm_matches_reference_formula_exactly() -> None:
    generator = torch.Generator(device="cuda").manual_seed(71)
    x = torch.randn(3, 1, 4096, generator=generator, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(4096, generator=generator, device="cuda")
    xf = x.float()
    expected = (weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6))).to(
        torch.bfloat16
    )
    assert torch.equal(rms_norm(x, weight, 1e-6), expected)


def test_load_packed_shape_guard() -> None:
    from runtime.loading.gguf import GgufTensor

    linear = PackedQ8_0Linear(8, 32)
    bad = GgufTensor(
        name="x",
        type_name="Q8_0",
        shape=(4, 32),
        data=torch.zeros(4 * 32 // 32 * 34, dtype=torch.uint8),
    )
    with pytest.raises(ValueError, match="expected"):
        linear.load_packed(bad)
    experts = PackedIQ2_XSExperts(2, 32, 256)
    bad_exp = GgufTensor(
        name="x",
        type_name="IQ2_XS",
        shape=(3, 32, 256),
        data=torch.zeros(3 * 32 * 74, dtype=torch.uint8),
    )
    with pytest.raises(ValueError, match="expected"):
        experts.load_packed(bad_exp)


@pytest.mark.skipif(not REAL_GGUF.exists(), reason="GGUF download not present")
@pytest.mark.skipif(not REFERENCE_DIR.exists(), reason="reference drop not present")
def test_real_expert_slice_matches_reference_expert() -> None:
    """One real expert (blk.3, expert 0) through the reference Expert module."""
    ref = _load_reference_model_module()
    tensors = load_gguf_tensors(
        REAL_GGUF,
        {
            "blk.3.ffn_gate_exps.weight",
            "blk.3.ffn_up_exps.weight",
            "blk.3.ffn_down_exps.weight",
        },
        device="cpu",
    )
    inter, hidden = 2048, 4096
    row_bytes = (hidden // 256) * 74
    per_expert = inter * row_bytes

    def expert_matrix(name: str) -> torch.Tensor:
        packed = tensors[name].data[:per_expert]  # expert 0 slice (expert-major)
        return dequantize_iq2_xs(packed).reshape(inter, hidden)

    w1 = expert_matrix("blk.3.ffn_gate_exps.weight")
    w3 = expert_matrix("blk.3.ffn_up_exps.weight")
    # ffn_down_exps torch shape is (256, 4096, 2048): expert 0 is [4096, 2048]
    down_bytes = hidden * (inter // 256) * 74
    w2 = dequantize_iq2_xs(tensors["blk.3.ffn_down_exps.weight"].data[:down_bytes])
    w2 = w2.reshape(hidden, inter)

    expert = ref.Expert(hidden, inter, dtype=torch.float32, swiglu_limit=10.0)
    expert.w1.weight = torch.nn.Parameter(w1)
    expert.w3.weight = torch.nn.Parameter(w3)
    expert.w2.weight = torch.nn.Parameter(w2)

    gen = torch.Generator().manual_seed(8)
    x = torch.randn(3, hidden, generator=gen) * 0.05
    expected = expert(x.float(), weights=torch.ones(3, 1))
    gate = x.float() @ w1.t()
    up = x.float() @ w3.t()
    h = swiglu(gate, up, 10.0)
    actual = h @ w2.t()
    assert torch.allclose(actual, expected, rtol=0, atol=0)
