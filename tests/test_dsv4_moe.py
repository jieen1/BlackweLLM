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
    for linear in (moe.shared_w1, moe.shared_w3, moe.shared_w2, moe.gate.weight):
        data = valid_q8_0_blocks(rng, linear.packed.numel() // 34)
        linear.packed.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
    if hasattr(moe.gate, "bias"):
        moe.gate.bias.copy_(
            torch.randn(moe.gate.bias.shape, generator=torch.Generator().manual_seed(seed))
        )
    else:
        moe.gate.tid2eid.copy_(
            torch.randint(0, moe.config.n_routed_experts, moe.gate.tid2eid.shape)
        )


def reference_moe_output(moe: Dsv4MoE, x: torch.Tensor, input_ids) -> torch.Tensor:
    """Manual transcription of the reference MoE semantics over dequanted
    weights -- the oracle for the module's gather/scatter wiring."""
    cfg = moe.config
    flat = x.reshape(-1, x.shape[-1]).float()
    gate_w = moe.gate.weight.dequantized()
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


def test_rms_norm_matches_reference_formula() -> None:
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(4, 64, generator=gen, dtype=torch.bfloat16)  # rms_norm is dim-agnostic
    w = torch.randn(64, generator=gen)
    expected_f = x.float()
    expected = (
        w * (expected_f * torch.rsqrt(expected_f.square().mean(-1, keepdim=True) + 1e-6))
    ).to(torch.bfloat16)
    assert torch.equal(rms_norm(x, w, 1e-6), expected)


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
