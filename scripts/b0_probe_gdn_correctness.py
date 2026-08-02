"""B0-4 probe: compare FLA v0.5.2's gated_delta_rule kernels against HF
transformers' own torch reference implementation (torch_chunk_gated_delta_rule
/ torch_recurrent_gated_delta_rule in modeling_qwen3_5.py), using Qwen3.6's
real GDN shapes:

    linear_num_key_heads=16  linear_key_head_dim=128
    linear_num_value_heads=48  linear_value_head_dim=128
    (num_v_heads // num_k_heads == 3 -> repeat_interleave factor 3, matching
    Qwen3_5GatedDeltaNet.forward lines ~505-507)

This only imports the installed `transformers` package (pip-installed,
"qwen3_5" tag == Qwen3.6) and the installed `fla` package. No oracle/ code
is read or imported (hard constraint respected).

Run with: ~/.venvs/vllm/bin/python scripts/b0_probe_gdn_correctness.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

DEVICE = torch.device("cuda")
NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_K_DIM = 128
HEAD_V_DIM = 128
REPEAT = NUM_V_HEADS // NUM_K_HEADS
assert REPEAT * NUM_K_HEADS == NUM_V_HEADS


def make_layer_inputs(batch, seq_len, *, seed=0, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    query = torch.randn(batch, seq_len, NUM_K_HEADS, HEAD_K_DIM, device=DEVICE, dtype=dtype)
    key = torch.randn(batch, seq_len, NUM_K_HEADS, HEAD_K_DIM, device=DEVICE, dtype=dtype)
    value = torch.randn(batch, seq_len, NUM_V_HEADS, HEAD_V_DIM, device=DEVICE, dtype=dtype)
    b = torch.randn(batch, seq_len, NUM_V_HEADS, device=DEVICE, dtype=dtype)
    a = torch.randn(batch, seq_len, NUM_V_HEADS, device=DEVICE, dtype=dtype)
    dt_bias = torch.ones(NUM_V_HEADS, device=DEVICE, dtype=torch.float32)
    A_log = torch.log(torch.empty(NUM_V_HEADS, device=DEVICE).uniform_(0, 16)).to(torch.float32)

    beta = b.sigmoid()
    g = -A_log.exp() * F.softplus(a.float() + dt_bias)  # matches modeling_qwen3_5.py:504

    query = query.repeat_interleave(REPEAT, dim=2)
    key = key.repeat_interleave(REPEAT, dim=2)
    return query, key, value, g, beta


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0).item()


def compare(name, fla_out, ref_out, fla_state, ref_state):
    out_err = (fla_out.float() - ref_out.float()).abs().max().item()
    out_cos = cosine(fla_out, ref_out)
    print(f"[{name}] core_attn_out: max_abs_err={out_err:.6g} cosine={out_cos:.8f}")
    if fla_state is not None and ref_state is not None:
        state_err = (fla_state.float() - ref_state.float()).abs().max().item()
        state_cos = cosine(fla_state, ref_state)
        print(
            f"[{name}] final_state ({fla_state.dtype} vs {ref_state.dtype}): "
            f"max_abs_err={state_err:.6g} cosine={state_cos:.8f}"
        )


def test_chunk_prefill():
    print("\n=== chunk_gated_delta_rule (prefill, T=300, not a multiple of chunk_size=64) ===")
    query, key, value, g, beta = make_layer_inputs(batch=2, seq_len=300, seed=11)
    fla_out, fla_state = chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    ref_out, ref_state = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    compare("chunk", fla_out, ref_out, fla_state, ref_state)
    return fla_state, ref_state


def test_recurrent_decode_multistep(init_fla_state, init_ref_state, n_steps=8):
    print(
        f"\n=== fused_recurrent_gated_delta_rule ({n_steps} sequential decode steps, "
        f"starting from the prefill's final_state) ==="
    )
    batch = init_fla_state.shape[0]
    fla_state = init_fla_state.clone()
    ref_state = init_ref_state.clone()
    for step in range(n_steps):
        query, key, value, g, beta = make_layer_inputs(batch=batch, seq_len=1, seed=100 + step)
        fla_out, fla_state = fused_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=fla_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        ref_out, ref_state = torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=ref_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()
        compare(f"decode step {step}", fla_out, ref_out, fla_state, ref_state)


def test_recurrent_decode_multistep_bf16_rounded(init_fla_state, init_ref_state, n_steps=8):
    """Same as above, but round the persisted state to BF16 between steps
    (mirroring transformers/cache_utils.py LinearAttentionLayer, which stores
    the FP32 kernel output into a BF16-dtype persistent buffer via .copy_()).
    This is the "single-step FP32 compute, cross-step BF16 rounding" mechanism
    already identified from static reading; here we measure how much it costs.
    """
    print(
        f"\n=== fused_recurrent_gated_delta_rule ({n_steps} steps) WITH BF16 "
        f"cross-step state rounding (mirrors HF cache_utils.py exactly) vs FP32-clean FLA ==="
    )
    batch = init_fla_state.shape[0]
    fla_state_bf16_persisted = init_fla_state.clone().to(torch.bfloat16)
    fla_state_fp32_clean = init_fla_state.clone()
    for step in range(n_steps):
        query, key, value, g, beta = make_layer_inputs(batch=batch, seq_len=1, seed=200 + step)
        out_bf16path, new_state = fused_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=fla_state_bf16_persisted,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        # HF's LinearAttentionLayer.update_recurrent_state does
        # self.recurrent_states.copy_(recurrent_states) into a BF16 buffer.
        fla_state_bf16_persisted = fla_state_bf16_persisted.copy_(new_state).clone()

        out_fp32path, fla_state_fp32_clean = fused_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=fla_state_fp32_clean,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()
        out_err = (out_bf16path.float() - out_fp32path.float()).abs().max().item()
        print(
            f"[step {step}] BF16-persisted-state vs FP32-clean-state core_attn_out "
            f"max_abs_err={out_err:.6g}"
        )


def main():
    print("device:", torch.cuda.get_device_name(0))
    print("fla chunk_gated_delta_rule:", chunk_gated_delta_rule)
    print("fla fused_recurrent_gated_delta_rule:", fused_recurrent_gated_delta_rule)

    fla_state, ref_state = test_chunk_prefill()
    test_recurrent_decode_multistep(fla_state, ref_state, n_steps=8)
    test_recurrent_decode_multistep_bf16_rounded(fla_state, ref_state, n_steps=8)


if __name__ == "__main__":
    main()
