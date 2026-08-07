"""Kernel-path vs eager attention parity (Phase 3 gate, layer level).

The eager Dsv4Attention is the executable definition; Dsv4AttnKernelLayer
runs the same projections/compressor/indexer math but stores KV packed
(FP8 pages, bit-exact by test_dsv4_kv_pack) and computes attention in the
sparkinfer compressed_mla kernel. This pins the kernel path's output to
the eager path: per-step cosine + max-abs on the final attention output,
for all three layer flavors (window-only / CSA ratio-4 / HCA ratio-128),
across prefill, decode, and prefill longer than the window ring.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("sparkinfer")

from runtime.loading.gguf import GgufTensor  # noqa: E402
from runtime.model.dsv4_attn_kernel import Dsv4AttnKernelLayer  # noqa: E402
from runtime.model.dsv4_config import Dsv4Config  # noqa: E402
from runtime.model.dsv4_model import Dsv4Attention  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

#: Real DSV4-Flash per-layer compress_ratios (43 main layers; the file's
#: three trailing zeros belong to the DSpark stage, dropped here).
COMPRESS_RATIOS = (0, 0) + tuple(4 if i % 2 == 0 else 128 for i in range(41))
assert len(COMPRESS_RATIOS) == 43
CONFIG = Dsv4Config(compress_ratios=COMPRESS_RATIOS)


def q8_0_tensor(name: str, shape: tuple[int, ...], gen, device) -> GgufTensor:
    """Random Q8_0 payload with d=1.0: dequantizes to exact int8 values."""
    numel = math.prod(shape)
    blocks = numel // 32
    qs = torch.randint(-10, 10, (blocks, 32), generator=gen, device=device, dtype=torch.int8)
    d = torch.full((blocks, 1), 1.0, dtype=torch.float16, device=device)
    packed = torch.cat([d.view(torch.uint8), qs.view(torch.uint8)], dim=1).reshape(-1).contiguous()
    return GgufTensor(name=name, type_name="Q8_0", shape=shape, data=packed)


def _load_q8(module, name: str, shape: tuple[int, ...], gen, device) -> None:
    getattr(module, name).load_packed(q8_0_tensor(name, shape, gen, device))


def _load_f32(buffer: torch.Tensor, gen) -> None:
    buffer.copy_(torch.randn(buffer.shape, generator=gen, device=buffer.device))


def _load_attn_weights(attn, layer_id: int, gen, device) -> None:
    c = CONFIG
    _load_q8(attn, "wq_a", (c.q_lora_rank, c.hidden_size), gen, device)
    _load_q8(attn, "wq_b", (c.num_heads * c.head_dim, c.q_lora_rank), gen, device)
    _load_q8(attn, "wkv", (c.head_dim, c.hidden_size), gen, device)
    _load_q8(
        attn,
        "wo_a",
        (c.o_groups * c.o_lora_rank, c.num_heads * c.head_dim // c.o_groups),
        gen,
        device,
    )
    _load_q8(attn, "wo_b", (c.hidden_size, c.o_groups * c.o_lora_rank), gen, device)
    _load_f32(attn.q_norm_weight, gen)
    _load_f32(attn.kv_norm_weight, gen)
    _load_f32(attn.attn_sink, gen)
    if attn.compressor is not None:
        comp = attn.compressor
        _load_q8(comp, "wkv", (comp.coeff * comp.head_dim, c.hidden_size), gen, device)
        _load_q8(comp, "wgate", (comp.coeff * comp.head_dim, c.hidden_size), gen, device)
        _load_f32(comp.ape, gen)
        _load_f32(comp.norm_weight, gen)
    if attn.indexer is not None:
        idx = attn.indexer
        _load_q8(idx, "wq_b", (c.index_n_heads * c.index_head_dim, c.q_lora_rank), gen, device)
        _load_q8(idx, "weights_proj", (c.index_n_heads, c.hidden_size), gen, device)
        comp = idx.compressor
        _load_q8(comp, "wkv", (comp.coeff * comp.head_dim, c.hidden_size), gen, device)
        _load_q8(comp, "wgate", (comp.coeff * comp.head_dim, c.hidden_size), gen, device)
        _load_f32(comp.ape, gen)
        _load_f32(comp.norm_weight, gen)


def _make_pair(layer_id: int, max_q_rows: int = 160):
    device = "cuda"
    eager = Dsv4Attention(CONFIG, layer_id, max_seq_len=4096, device=device)
    kernel = Dsv4AttnKernelLayer(
        CONFIG, layer_id, max_seq_len=4096, max_q_rows=max_q_rows, device=device
    )
    # same seed per layer: both layers must load IDENTICAL weights
    seed = 20260807 + layer_id
    _load_attn_weights(eager, layer_id, torch.Generator(device=device).manual_seed(seed), device)
    _load_attn_weights(kernel, layer_id, torch.Generator(device=device).manual_seed(seed), device)
    return eager, kernel


def _compare(a: torch.Tensor, b: torch.Tensor, what: str) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    max_abs = (a - b).abs().max().item()
    print(f"  {what}: cosine={cos:.8f} max_abs={max_abs:.6f}")
    return cos


def _random_input(seqlen: int, gen, device) -> torch.Tensor:
    return (torch.randn(1, seqlen, CONFIG.hidden_size, generator=gen, device=device)).to(
        torch.bfloat16
    )


def _run_session(eager, kernel, gen, prefill_len: int, decode_steps: int) -> float:
    device = "cuda"
    worst = 1.0
    x = _random_input(prefill_len, gen, device)
    e = eager(x, 0)
    k = kernel(x, 0)
    worst = min(worst, _compare(e, k, "prefill"))
    pos = prefill_len
    for step in range(decode_steps):
        xt = _random_input(1, gen, device)
        e = eager(xt, pos)
        k = kernel(xt, pos)
        worst = min(worst, _compare(e, k, f"decode@{pos}"))
        pos += 1
    return worst


@pytest.mark.parametrize(
    ("layer_id", "max_q_rows"),
    [(0, 160), (2, 160), (3, 160)],
    ids=["ratio0", "ratio4", "ratio128"],
)
def test_parity_prefill_and_decode(layer_id: int, max_q_rows: int) -> None:
    eager, kernel = _make_pair(layer_id, max_q_rows=max_q_rows)
    gen = torch.Generator(device="cuda").manual_seed(1)
    worst = _run_session(eager, kernel, gen, prefill_len=32, decode_steps=6)
    # Numerical budget: the kernel contract takes bf16 q (the eager path keeps
    # fp32) and casts softmax weights to bf16 for the V MMA; measured worst
    # cosine on random-scale weights is ~0.9995-0.9997. The strict >0.99999
    # gate belongs to the real-weight model-level parity (Phase 3 step 5).
    assert worst >= 0.999, (
        f"kernel-path attention diverges from eager (layer {layer_id}, cos {worst})"
    )


def test_parity_prefill_longer_than_window() -> None:
    """Prefill 130 > window 128: ring wrap + the window covering only the
    last 128 positions must match eager exactly."""
    eager, kernel = _make_pair(0, max_q_rows=160)
    gen = torch.Generator(device="cuda").manual_seed(2)
    worst = _run_session(eager, kernel, gen, prefill_len=130, decode_steps=3)
    assert worst >= 0.999


def test_parity_decode_boundary_without_prefill() -> None:
    """A session that only decodes from an empty slot (first token at
    start_pos=0 with seqlen 1, then rolling decode)."""
    eager, kernel = _make_pair(2, max_q_rows=160)  # ratio-4: exercises the
    gen = torch.Generator(device="cuda").manual_seed(3)
    x0 = _random_input(1, gen, device="cuda")
    e, k = eager(x0, 0), kernel(x0, 0)
    worst = _compare(e, k, "decode@0")
    pos = 1
    for step in range(7):  # crosses compression boundaries at pos 3 and 7
        xt = _random_input(1, gen, device="cuda")
        e, k = eager(xt, pos), kernel(xt, pos)
        worst = min(worst, _compare(e, k, f"decode@{pos}"))
        pos += 1
    assert worst >= 0.999
