"""Assembly tests: HC/Block/Transformer wiring against the reference, the
zero-exemption GGUF coverage claim, and the real-weight eager smoke.

Real-file tests need the downloaded GGUF; the full-load smoke additionally
needs the GPU and is gated behind QSR_DSV4_FULL_LOAD=1 because it occupies
the whole card (81 GiB of weights).
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import struct
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from loader.gguf_header import read_gguf_header  # noqa: E402
from runtime.loading.gguf import load_gguf_tensors  # noqa: E402
from runtime.model.dsv4_config import Dsv4Config, config_from_gguf_kv  # noqa: E402
from runtime.model.dsv4_model import (  # noqa: E402
    DenseLinear,
    Dsv4Block,
    Dsv4Embedding,
    Dsv4Transformer,
    PackedIQ2_XSExperts,
    PackedQ8_0Weight,
    expected_gguf_tensor_names,
    hc_split_sinkhorn,
    load_dsv4_from_gguf,
    rms_norm,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "notes" / "dsv4flash-ref" / "inference"
REAL_GGUF = Path(
    "/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf"
)
real_gguf = pytest.mark.skipif(not REAL_GGUF.exists(), reason="GGUF not downloaded")
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def _load_reference_modules():
    if "kernel" not in sys.modules:
        kernel_spec = importlib.util.spec_from_file_location("kernel", REFERENCE_DIR / "kernel.py")
        kernel = importlib.util.module_from_spec(kernel_spec)
        sys.modules["kernel"] = kernel
        kernel_spec.loader.exec_module(kernel)
    if "dsv4_ref_model" not in sys.modules:
        spec = importlib.util.spec_from_file_location("dsv4_ref_model", REFERENCE_DIR / "model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["dsv4_ref_model"] = module
        spec.loader.exec_module(module)
    return sys.modules["dsv4_ref_model"], sys.modules["kernel"]


# ---------------------------------------------------------------------------
# synthetic-weight helpers (same block formats as test_dsv4_moe.py)
# ---------------------------------------------------------------------------


def valid_q8_0_blocks(rng: random.Random, n_blocks: int) -> bytes:
    out = bytearray()
    for _ in range(n_blocks):
        out += struct.pack("<H", rng.randrange(0x1C00, 0x2400))
        out += struct.pack("<32b", *(rng.randrange(-127, 128) for _ in range(32)))
    return bytes(out)


def valid_iq2_xs_blocks(rng: random.Random, n_blocks: int) -> bytes:
    out = bytearray()
    for _ in range(n_blocks):
        out += struct.pack("<H", rng.randrange(0x1C00, 0x2400))
        out += struct.pack("<32H", *(rng.getrandbits(16) for _ in range(32)))
        out += bytes(rng.getrandbits(8) for _ in range(8))
    return bytes(out)


def small_config() -> Dsv4Config:
    # contiguous dims honor IQ2_XS %256 and Q8_0 %32 block constraints
    return Dsv4Config(
        vocab_size=32,
        hidden_size=256,
        num_layers=3,
        max_position_embeddings=512,
        num_heads=16,
        head_dim=128,
        rope_head_dim=64,
        q_lora_rank=64,
        o_groups=4,
        o_lora_rank=64,
        window_size=8,
        compress_ratios=(0, 4, 128),
        rope_factor=4.0,
        rope_original_seq_len=64,
        index_n_heads=8,
        index_head_dim=128,
        index_topk=16,
        n_routed_experts=4,
        n_activated_experts=2,
        moe_intermediate_size=256,
        n_hash_layers=1,
    )


def fill_transformer(model: Dsv4Transformer, seed: int) -> None:
    rng = random.Random(seed)
    for module in model.modules():
        if isinstance(module, PackedQ8_0Weight):  # includes PackedQ8_0Linear
            data = valid_q8_0_blocks(rng, module.packed.numel() // 34)
            module.packed.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
        elif isinstance(module, PackedIQ2_XSExperts):
            data = valid_iq2_xs_blocks(rng, module.packed.numel() // 74)
            module.packed.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
        elif isinstance(module, DenseLinear):
            gen = torch.Generator().manual_seed(seed)
            module.weight.copy_(torch.randn(module.weight.shape, generator=gen) * 0.05)
        elif isinstance(module, Dsv4Embedding):
            data = valid_q8_0_blocks(rng, module.packed.numel() // 34)
            module.packed.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
    for name, buffer in model.named_buffers():
        if buffer.dtype == torch.float32:
            gen = torch.Generator().manual_seed(seed + hash(name) % 1000)
            buffer.copy_(torch.randn(buffer.shape, generator=gen) * 0.1)
    for block in model.blocks:
        if block.moe.gate.hashed:
            block.moe.gate.tid2eid.random_(0, model.config.n_routed_experts)


def test_transformer_synthetic_forward_deterministic() -> None:
    cfg = small_config()
    model = Dsv4Transformer(cfg, max_seq_len=64)
    fill_transformer(model, seed=11)
    ids = torch.tensor([[5, 9, 1, 17, 3]], dtype=torch.long)

    logits_a = model(ids, start_pos=0)
    assert logits_a.shape == (1, 5, cfg.vocab_size)
    assert torch.isfinite(logits_a).all()
    # same prefill again: caches are rewritten from the same inputs
    logits_b = model(ids, start_pos=0)
    assert torch.equal(logits_a, logits_b)

    # two greedy decode steps exercise the ring/compressor decode paths
    next_id = logits_a[0, -1].argmax()
    step1 = model(next_id.view(1, 1), start_pos=5)
    assert step1.shape == (1, 1, cfg.vocab_size) and torch.isfinite(step1).all()
    next_id2 = step1[0, -1].argmax()
    step2 = model(next_id2.view(1, 1), start_pos=6)
    assert step2.shape == (1, 1, cfg.vocab_size) and torch.isfinite(step2).all()


def test_expected_names_match_real_header() -> None:
    """The config-generated inventory must equal the file header exactly."""
    if not REAL_GGUF.exists():
        pytest.skip("GGUF not downloaded")
    header = read_gguf_header(REAL_GGUF)
    config = config_from_gguf_kv(header.kv)
    expected = expected_gguf_tensor_names(config)
    actual = {t.name for t in header.tensors}
    assert expected == actual, (
        f"missing from file: {sorted(expected - actual)[:5]}, "
        f"unexpected in file: {sorted(actual - expected)[:5]}"
    )
    assert len(header.tensors) == 1328
    types = {t.name: t.type_name for t in header.tensors}
    assert types["token_embd.weight"] == "Q8_0"
    assert types["blk.0.ffn_gate_inp.weight"] == "BF16"
    assert types["blk.0.ffn_gate_tid2eid.weight"] == "I32"
    assert types["blk.3.exp_probs_b.bias"] == "F32"
    assert types["blk.2.attn_compressor_ape.weight"] == "Q8_0"


# ---------------------------------------------------------------------------
# reference parity for the HC stream (real weights, GPU)
# ---------------------------------------------------------------------------


@needs_gpu
@real_gguf
def test_hc_split_sinkhorn_runtime_matches_kernel() -> None:
    ref, kernel = _load_reference_modules()
    weights = load_gguf_tensors(
        REAL_GGUF,
        {"blk.0.hc_attn_scale.weight", "blk.0.hc_attn_base.weight"},
        device="cuda",
    )
    scale = weights["blk.0.hc_attn_scale.weight"].data.to(torch.float32)
    base = weights["blk.0.hc_attn_base.weight"].data.to(torch.float32)
    gen = torch.Generator(device="cuda").manual_seed(20260810)
    mixes = torch.randn(3, 17, 24, generator=gen, device="cuda") * 2

    ref_pre, ref_post, ref_comb = kernel.hc_split_sinkhorn(
        mixes, scale, base, hc_mult=4, sinkhorn_iters=20, eps=1e-6
    )
    our_pre, our_post, our_comb = hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    assert torch.allclose(ref_pre, our_pre, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ref_post, our_post, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ref_comb, our_comb, rtol=1e-4, atol=1e-5)


class _StubAttn(torch.nn.Module):
    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        return torch.tanh(x.float()).to(x.dtype) * 0.1


class _StubMoE(torch.nn.Module):
    def forward(self, x: torch.Tensor, input_ids) -> torch.Tensor:
        return torch.sin(x.float()).to(x.dtype) * 0.1


@needs_gpu
@real_gguf
def test_block_hc_stream_matches_reference() -> None:
    """Our Block's HC reduce/expand + norm ordering vs a line-for-line
    transcription of the reference Block.forward over real layer-0 weights."""
    ref, kernel = _load_reference_modules()
    from runtime.model.dsv4_quant import dequantize_q8_0

    header = read_gguf_header(REAL_GGUF)
    config = config_from_gguf_kv(header.kv)
    names = {
        "blk.0.hc_attn_fn.weight",
        "blk.0.hc_attn_base.weight",
        "blk.0.hc_attn_scale.weight",
        "blk.0.hc_ffn_fn.weight",
        "blk.0.hc_ffn_base.weight",
        "blk.0.hc_ffn_scale.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
    }
    w = load_gguf_tensors(REAL_GGUF, names, device="cuda")
    hc_attn_fn = dequantize_q8_0(w["blk.0.hc_attn_fn.weight"].data).reshape(
        config.hc_mix_dim, config.hc_dim
    )
    hc_ffn_fn = dequantize_q8_0(w["blk.0.hc_ffn_fn.weight"].data).reshape(
        config.hc_mix_dim, config.hc_dim
    )
    params = {
        "attn_fn": hc_attn_fn,
        "ffn_fn": hc_ffn_fn,
        "attn_base": w["blk.0.hc_attn_base.weight"].data.to(torch.float32),
        "ffn_base": w["blk.0.hc_ffn_base.weight"].data.to(torch.float32),
        "attn_scale": w["blk.0.hc_attn_scale.weight"].data.to(torch.float32),
        "ffn_scale": w["blk.0.hc_ffn_scale.weight"].data.to(torch.float32),
        "attn_norm": w["blk.0.attn_norm.weight"].data.to(torch.float32),
        "ffn_norm": w["blk.0.ffn_norm.weight"].data.to(torch.float32),
    }

    block = Dsv4Block(config, 0, max_seq_len=256, device="cuda")
    block.hc_attn_fn.packed.copy_(w["blk.0.hc_attn_fn.weight"].data)
    block.hc_ffn_fn.packed.copy_(w["blk.0.hc_ffn_fn.weight"].data)
    block.hc_attn_base.copy_(params["attn_base"])
    block.hc_ffn_base.copy_(params["ffn_base"])
    block.hc_attn_scale.copy_(params["attn_scale"])
    block.hc_ffn_scale.copy_(params["ffn_scale"])
    block.attn_norm_weight.copy_(params["attn_norm"])
    block.ffn_norm_weight.copy_(params["ffn_norm"])
    block.attn = _StubAttn()
    block.moe = _StubMoE()

    eps = config.norm_eps

    def ref_hc_pre(x, hc_fn, hc_scale, hc_base):
        shape, dtype = x.size(), x.dtype
        xf = x.flatten(2).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
        mixes = torch.nn.functional.linear(xf, hc_fn) * rsqrt
        pre, post, comb = kernel.hc_split_sinkhorn(
            mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6
        )
        y = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2)
        return y.to(dtype), post, comb

    def ref_hc_post(x, residual, post, comb):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
        )
        return y.type_as(x)

    def ref_forward(x, input_ids):
        residual = x
        x, post, comb = ref_hc_pre(x, params["attn_fn"], params["attn_scale"], params["attn_base"])
        x = rms_norm(x, params["attn_norm"], eps)
        x = torch.tanh(x.float()).to(x.dtype) * 0.1
        x = ref_hc_post(x, residual, post, comb)
        residual = x
        x, post, comb = ref_hc_pre(x, params["ffn_fn"], params["ffn_scale"], params["ffn_base"])
        x = rms_norm(x, params["ffn_norm"], eps)
        x = torch.sin(x.float()).to(x.dtype) * 0.1
        x = ref_hc_post(x, residual, post, comb)
        return x

    gen = torch.Generator(device="cuda").manual_seed(20260811)
    x = torch.randn(1, 6, 4, 4096, generator=gen, device="cuda").to(torch.bfloat16) * 0.5
    input_ids = torch.randint(0, config.vocab_size, (1, 6), generator=gen, device="cuda")

    ours = block(x, 0, input_ids)
    theirs = ref_forward(x, input_ids)
    assert ours.shape == theirs.shape == (1, 6, 4, 4096)
    # bf16 stream: allow 1-2 ulp of drift from reduction-order differences
    assert torch.allclose(ours.float(), theirs.float(), rtol=2e-2, atol=2e-2)
    assert (ours.float() - theirs.float()).abs().max() < 0.05


@needs_gpu
@real_gguf
def test_hc_head_matches_reference() -> None:
    ref, kernel = _load_reference_modules()
    from runtime.model.dsv4_quant import dequantize_q8_0

    header = read_gguf_header(REAL_GGUF)
    config = config_from_gguf_kv(header.kv)
    w = load_gguf_tensors(
        REAL_GGUF,
        {"output_hc_fn.weight", "output_hc_base.weight", "output_hc_scale.weight"},
        device="cuda",
    )
    hc_fn = dequantize_q8_0(w["output_hc_fn.weight"].data).reshape(config.hc_mult, config.hc_dim)
    base = w["output_hc_base.weight"].data.to(torch.float32)
    scale = w["output_hc_scale.weight"].data.to(torch.float32)

    model = Dsv4Transformer(config, max_seq_len=256, device="cuda")
    model.hc_head_fn.packed.copy_(w["output_hc_fn.weight"].data)
    model.hc_head_base.copy_(base)
    model.hc_head_scale.copy_(scale)

    gen = torch.Generator(device="cuda").manual_seed(20260812)
    x = torch.randn(1, 3, 4, 4096, generator=gen, device="cuda").to(torch.bfloat16) * 0.5

    # reference hc_head transcription (model.py Block.hc_head)
    shape, dtype = x.size(), x.dtype
    xf = x.flatten(2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + config.norm_eps)
    mixes = torch.nn.functional.linear(xf, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * scale + base) + config.hc_eps
    expected = torch.sum(pre.unsqueeze(-1) * xf.view(shape), dim=2).to(dtype)

    ours = model.hc_head(x)
    assert ours.shape == expected.shape
    assert torch.allclose(ours.float(), expected.float(), rtol=2e-2, atol=2e-2)


# ---------------------------------------------------------------------------
# full-load eager smoke (whole card; gated)
# ---------------------------------------------------------------------------

FULL_LOAD = bool(os.environ.get("QSR_DSV4_FULL_LOAD"))


@needs_gpu
@real_gguf
@pytest.mark.skipif(not FULL_LOAD, reason="set QSR_DSV4_FULL_LOAD=1 (occupies whole card)")
def test_full_load_eager_smoke() -> None:
    model, count = load_dsv4_from_gguf(REAL_GGUF, max_seq_len=1024, device="cuda")
    assert count == 1328

    try:
        from transformers import PreTrainedTokenizerFast

        tok = PreTrainedTokenizerFast(
            tokenizer_file=str(REPO_ROOT / "notes" / "dsv4flash-ref" / "tokenizer.json")
        )
        ids = torch.tensor([tok.encode("The meaning of life is")], device="cuda")
        prompt = "The meaning of life is"
    except Exception:
        tok = None
        ids = torch.tensor([[1000, 2000, 3000, 4000, 5000]], device="cuda")
        prompt = "<raw ids>"

    logits = model(ids, start_pos=0)
    assert logits.shape == (1, ids.shape[1], model.config.vocab_size)
    assert torch.isfinite(logits).all()
    last = logits[0, -1]
    top5 = torch.topk(last, 5)
    record = {
        "prompt": prompt,
        "start_pos": 0,
        "logits_last": {
            "mean": float(last.mean()),
            "std": float(last.std()),
            "max": float(last.max()),
        },
        "top5_ids": top5.indices.tolist(),
        "top5_logits": top5.values.tolist(),
        "argmax": int(last.argmax()),
    }
    if tok is not None:
        record["top5_tokens"] = [tok.decode([i]) for i in top5.indices.tolist()]
        record["argmax_token"] = tok.decode([int(last.argmax())])

    # one greedy decode step through the decode paths
    next_id = torch.tensor([[int(last.argmax())]], device="cuda")
    step = model(next_id, start_pos=ids.shape[1])
    assert step.shape == (1, 1, model.config.vocab_size)
    assert torch.isfinite(step).all()
    record["decode_step1_argmax"] = int(step[0, -1].argmax())
    if tok is not None:
        record["decode_step1_token"] = tok.decode([record["decode_step1_argmax"]])

    print("\nDSV4 EAGER SMOKE RECORD:\n" + json.dumps(record, indent=2, ensure_ascii=False))
