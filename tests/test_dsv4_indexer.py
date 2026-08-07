"""Indexer parity: Dsv4Indexer vs the reference Indexer on real blk.2
weights (ratio-4 CSA layer), prefill + decode, exact index comparison."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

from loader.gguf_header import read_gguf_header  # noqa: E402
from runtime.loading.gguf import load_gguf_tensors  # noqa: E402
from runtime.model.dsv4_config import config_from_gguf_kv  # noqa: E402
from runtime.model.dsv4_model import Dsv4Indexer  # noqa: E402
from runtime.model.dsv4_quant import dequantize_q8_0  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "notes" / "dsv4flash-ref" / "inference"
REAL_GGUF = Path(
    "/home/bot/models/DeepSeek-V4-Flash-0731-GGUF/DeepSeek-V4-Flash-0731-IQ2_XS-Experts-Q8_0.gguf"
)
if not REAL_GGUF.exists():
    pytest.skip("GGUF download not present", allow_module_level=True)
if not REFERENCE_DIR.exists():
    pytest.skip("reference drop not present", allow_module_level=True)

LAYER = 2  # ratio-4 CSA layer with indexer


def _install_hadamard_shim() -> None:
    """fast_hadamard_transform does not build on this machine; the reference
    imports it lazily inside rotate_activation. The shim supplies our
    Sylvester implementation -- this parity run is also the first check that
    the convention (signs/ordering) agrees to within rounding."""
    import types

    from runtime.model.dsv4_attention import hadamard_transform as ours

    if "fast_hadamard_transform" in sys.modules:
        return
    shim = types.ModuleType("fast_hadamard_transform")
    shim.hadamard_transform = lambda x, scale=1.0: ours(x, scale)
    sys.modules["fast_hadamard_transform"] = shim


def _load_ref():
    _install_hadamard_shim()
    if "kernel" not in sys.modules:
        spec = importlib.util.spec_from_file_location("kernel", REFERENCE_DIR / "kernel.py")
        kernel = importlib.util.module_from_spec(spec)
        sys.modules["kernel"] = kernel
        spec.loader.exec_module(kernel)
    spec = importlib.util.spec_from_file_location("dsv4_ref_model", REFERENCE_DIR / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.scale_fmt = "ue8m0"
    module.scale_dtype = torch.float8_e8m0fnu
    return module


def _ref_args(ref):
    return ref.ModelArgs(
        max_batch_size=1,
        max_seq_len=4096,
        vocab_size=129280,
        dim=4096,
        moe_inter_dim=2048,
        n_layers=43,
        n_hash_layers=3,
        n_mtp_layers=3,
        n_heads=64,
        n_routed_experts=256,
        n_shared_experts=1,
        n_activated_experts=6,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        swiglu_limit=10.0,
        q_lora_rank=1024,
        head_dim=512,
        rope_head_dim=64,
        o_groups=8,
        o_lora_rank=1024,
        window_size=128,
        compress_ratios=tuple([0, 0] + [4, 128] * 20 + [4]),
        compress_rope_theta=160000.0,
        original_seq_len=65536,
        rope_theta=10000.0,
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        dtype="fp8",
        scale_fmt="ue8m0",
        expert_dtype=None,
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=(40, 41, 42),
        dspark_markov_rank=256,
    )


@pytest.fixture(scope="module")
def pair():
    ref = _load_ref()
    # real runtime runs under torch.set_default_dtype(bfloat16) (see
    # reference generate.py); the indexer's kv_cache buffer inherits it,
    # and the reference einsum is dtype-strict.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    config = config_from_gguf_kv(read_gguf_header(REAL_GGUF).kv)
    prefix = f"blk.{LAYER}."
    tensors = load_gguf_tensors(
        REAL_GGUF,
        {
            prefix + "indexer.attn_q_b.weight",
            prefix + "indexer.proj.weight",
            prefix + "indexer_compressor_kv.weight",
            prefix + "indexer_compressor_gate.weight",
            prefix + "indexer_compressor_ape.weight",
            prefix + "indexer_compressor_norm.weight",
        },
        device="cpu",
    )

    ref_idx = ref.Indexer(_ref_args(ref), compress_ratio=4).cuda()
    wq_b = dequantize_q8_0(tensors[prefix + "indexer.attn_q_b.weight"].data)
    proj = dequantize_q8_0(tensors[prefix + "indexer.proj.weight"].data)
    ref_idx.wq_b.weight = torch.nn.Parameter(wq_b.reshape(8192, 1024).to(torch.bfloat16).cuda())
    ref_idx.weights_proj.weight = torch.nn.Parameter(
        proj.reshape(64, 4096).to(torch.bfloat16).cuda()
    )
    comp = ref_idx.compressor
    comp.wkv.weight = torch.nn.Parameter(
        dequantize_q8_0(tensors[prefix + "indexer_compressor_kv.weight"].data)
        .reshape(256, 4096)
        .cuda()
    )
    comp.wgate.weight = torch.nn.Parameter(
        dequantize_q8_0(tensors[prefix + "indexer_compressor_gate.weight"].data)
        .reshape(256, 4096)
        .cuda()
    )
    comp.ape = torch.nn.Parameter(
        dequantize_q8_0(tensors[prefix + "indexer_compressor_ape.weight"].data)
        .reshape(4, 256)
        .cuda()
    )
    comp.norm.weight = torch.nn.Parameter(
        tensors[prefix + "indexer_compressor_norm.weight"].data.to(torch.float32).cuda()
    )

    ours = Dsv4Indexer(config, LAYER).cuda()
    ours.wq_b.load_packed(tensors[prefix + "indexer.attn_q_b.weight"])
    ours.weights_proj.load_packed(tensors[prefix + "indexer.proj.weight"])
    ours.compressor.wkv.load_packed(tensors[prefix + "indexer_compressor_kv.weight"])
    ours.compressor.wgate.load_packed(tensors[prefix + "indexer_compressor_gate.weight"])
    ours.compressor.ape.copy_(
        dequantize_q8_0(tensors[prefix + "indexer_compressor_ape.weight"].data)
        .reshape(4, 256)
        .cuda()
    )
    ours.compressor.norm_weight.copy_(
        tensors[prefix + "indexer_compressor_norm.weight"].data.to(torch.float32).cuda()
    )

    torch.set_default_dtype(prev_dtype)
    # shared rope table (compressed-KV variant, proven identical to reference)
    freqs = ref.precompute_freqs_cis(
        config.rope_head_dim,
        4096,
        config.rope_original_seq_len,
        config.compress_rope_theta,
        config.rope_factor,
        config.beta_fast,
        config.beta_slow,
    ).cuda()
    ref_idx.freqs_cis = freqs
    ours.freqs_cis = freqs
    # bf16 caches, as in the real runtime
    ref_idx.kv_cache = torch.zeros(1, 1024, 128, device="cuda", dtype=torch.bfloat16)
    ours.kv_cache = torch.zeros(1, 1024, 128, device="cuda", dtype=torch.bfloat16)
    return ref, ref_idx, ours


def test_indexer_prefill_and_decode_parity(pair) -> None:
    ref, ref_idx, ours = pair
    # the reference builds causal masks with bare torch.arange; the real
    # runtime runs under set_default_device("cuda") (model.py __main__).
    prev_device = torch.device("cpu")
    torch.set_default_device("cuda")
    try:
        _run_parity(ref_idx, ours)
    finally:
        torch.set_default_device(prev_device)


def _run_parity(ref_idx, ours) -> None:
    gen = torch.Generator(device="cuda").manual_seed(21)
    seq_len = 40
    x = torch.randn(1, seq_len, 4096, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.05
    qr = torch.randn(1, seq_len, 1024, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.05
    offset = 128  # window slots ahead of compressed entries

    ref_idxs = ref_idx(x, qr, 0, offset)
    our_idxs = ours(x, qr, 0, offset)
    assert torch.equal(our_idxs, ref_idxs), (
        f"prefill idx differ at {(our_idxs != ref_idxs).nonzero()[:5]}"
    )
    assert torch.equal(ours.kv_cache, ref_idx.kv_cache)

    for step in range(24):
        pos = seq_len + step
        x1 = torch.randn(1, 1, 4096, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.05
        qr1 = torch.randn(1, 1, 1024, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.05
        r = ref_idx(x1, qr1, pos, offset)
        o = ours(x1, qr1, pos, offset)
        assert torch.equal(o, r), f"decode idx differ at pos {pos}"
        assert torch.equal(ours.kv_cache, ref_idx.kv_cache)
        # every returned index must point past the window offset or be -1
        assert ((o == -1) | (o >= offset)).all()
