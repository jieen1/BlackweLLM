"""B0-5 probe: is the GDN recurrent-state update (fla.ops.gated_delta_rule.
fused_recurrent_gated_delta_rule, the decode-step kernel selected by
Qwen3_5GatedDeltaNet.forward when seq_len==1) CUDA-Graph capture-safe?

Evidence already read (transformers/cache_utils.py, LinearAttentionLayer):
  - self.recurrent_states is allocated ONCE (lazy_initialization) with a
    fixed shape/dtype and marked via torch._dynamo.mark_static_address.
  - update_recurrent_state always does self.recurrent_states.copy_(...),
    NEVER `self.recurrent_states = new_tensor` (comment: "preserve the
    static address for cudagraphs").
  - The FLA kernel itself does NOT update state in-place; it returns a
    FRESH tensor each call (`q.new_empty(...)`) that must be copied into
    the persistent slot buffer by the caller.

This probe reproduces that exact pattern with torch.cuda.graph() directly
(bypassing the full HF model) to get a real, executable answer instead of
trusting the code comment: capture ONE decode step (read persistent state
-> fused_recurrent_gated_delta_rule -> copy_ back into the SAME persistent
buffer), replay it N times with different live inputs (mutated in the
static input buffers between replays, exactly as CUDA-graph decode replay
works in serving), and check the replayed recurrence numerically matches
an eager step-by-step reference.

Run with: ~/.venvs/vllm/bin/python scripts/b0_probe_gdn_cudagraph_capture.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

DEVICE = torch.device("cuda")
NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_K_DIM = 128
HEAD_V_DIM = 128
REPEAT = NUM_V_HEADS // NUM_K_HEADS
BATCH = 4  # simulate 4 concurrent decode slots, matching a small batch decode step


# A_log / dt_bias are model PARAMETERS in the real Qwen3_5GatedDeltaNet
# (nn.Parameter, fixed after init) -- they must NOT be re-randomized per
# call, or the "eager reference" and "graph replay" paths silently diverge
# for a reason that has nothing to do with CUDA graphs (this bit us on the
# first pass of this probe: an unseeded .uniform_() call consumed the global
# RNG stream a different number of times on each path). Generate them once.
torch.manual_seed(4242)
_DT_BIAS = torch.ones(NUM_V_HEADS, device=DEVICE, dtype=torch.float32)
_A_LOG = torch.log(torch.empty(NUM_V_HEADS, device=DEVICE).uniform_(0, 16)).to(torch.float32)


def make_step_inputs(seed: int, dtype=torch.bfloat16):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    query = torch.randn(BATCH, 1, NUM_K_HEADS, HEAD_K_DIM, device=DEVICE, dtype=dtype, generator=g)
    key = torch.randn(BATCH, 1, NUM_K_HEADS, HEAD_K_DIM, device=DEVICE, dtype=dtype, generator=g)
    value = torch.randn(BATCH, 1, NUM_V_HEADS, HEAD_V_DIM, device=DEVICE, dtype=dtype, generator=g)
    b = torch.randn(BATCH, 1, NUM_V_HEADS, device=DEVICE, dtype=dtype, generator=g)
    a = torch.randn(BATCH, 1, NUM_V_HEADS, device=DEVICE, dtype=dtype, generator=g)
    beta = b.sigmoid()
    gate = -_A_LOG.exp() * F.softplus(a.float() + _DT_BIAS)
    query = query.repeat_interleave(REPEAT, dim=2)
    key = key.repeat_interleave(REPEAT, dim=2)
    return query, key, value, gate, beta


def eager_reference_sequence(n_steps: int, seeds: list[int]):
    """Ground truth: run the recurrence eagerly step by step, no graph."""
    state = None
    outs = []
    for step in range(n_steps):
        query, key, value, gate, beta = make_step_inputs(seeds[step])
        out, state = fused_recurrent_gated_delta_rule(
            query, key, value, g=gate, beta=beta,
            initial_state=state, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        outs.append(out.clone())
    return outs, state


def main():
    print("device:", torch.cuda.get_device_name(0))
    n_steps = 6
    seeds = list(range(1000, 1000 + n_steps))

    print("\n--- eager step-by-step reference ---")
    eager_outs, eager_final_state = eager_reference_sequence(n_steps, seeds)

    print("\n--- capture ONE decode step as a CUDA graph, replay it n_steps times ---")
    # Static input buffers (the graph reads/writes these fixed addresses).
    query_buf, key_buf, value_buf, gate_buf, beta_buf = make_step_inputs(seeds[0])
    # Persistent recurrent-state buffer: allocate once, mark static, never
    # reassign -- mirrors transformers/cache_utils.py LinearAttentionLayer.
    state_shape = (BATCH, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM)
    state_buf = torch.zeros(state_shape, device=DEVICE, dtype=torch.float32)
    torch._dynamo.mark_static_address(state_buf)
    output_buf = torch.zeros(BATCH, 1, NUM_V_HEADS, HEAD_V_DIM, device=DEVICE, dtype=torch.bfloat16)

    # Warmup on a side stream first (required before capture: the kernel's
    # first invocation may allocate workspace / trigger triton autotune,
    # which must not happen during capture).
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            out, new_state = fused_recurrent_gated_delta_rule(
                query_buf, key_buf, value_buf, g=gate_buf, beta=beta_buf,
                initial_state=state_buf, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            state_buf.copy_(new_state)
            output_buf.copy_(out)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()
    # Reset state to zero after warmup pollution, matching a fresh slot.
    state_buf.zero_()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            out, new_state = fused_recurrent_gated_delta_rule(
                query_buf, key_buf, value_buf, g=gate_buf, beta=beta_buf,
                initial_state=state_buf, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            # Mirror transformers/cache_utils.py exactly: copy_, never rebind.
            state_buf.copy_(new_state)
            output_buf.copy_(out)
        torch.cuda.synchronize()
        print("RESULT: torch.cuda.graph() CAPTURE SUCCEEDED for fused_recurrent_gated_delta_rule "
              "+ copy_-into-static-buffer state update.")
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT: CAPTURE FAILED -- {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return

    state_buf.zero_()
    replay_outs = []
    for step in range(n_steps):
        q2, k2, v2, g2, b2 = make_step_inputs(seeds[step])
        query_buf.copy_(q2)
        key_buf.copy_(k2)
        value_buf.copy_(v2)
        gate_buf.copy_(g2)
        beta_buf.copy_(b2)
        graph.replay()
        torch.cuda.synchronize()
        replay_outs.append(output_buf.clone())

    print("\n--- comparing graph-replayed recurrence vs eager reference ---")
    max_err_overall = 0.0
    for step in range(n_steps):
        err = (replay_outs[step].float() - eager_outs[step].float()).abs().max().item()
        cos = F.cosine_similarity(
            replay_outs[step].float().reshape(-1), eager_outs[step].float().reshape(-1), dim=0
        ).item()
        max_err_overall = max(max_err_overall, err)
        print(f"[step {step}] replay vs eager: max_abs_err={err:.6g} cosine={cos:.8f}")
    final_state_err = (state_buf.float() - eager_final_state.float()).abs().max().item()
    print(
        f"final recurrent_state (post replay) vs eager final_state: "
        f"max_abs_err={final_state_err:.6g}"
    )

    if max_err_overall < 1e-3 and final_state_err < 1e-2:
        print("\nRESULT: CAPTURE-SAFE AND NUMERICALLY CORRECT -- graph replay reproduces the "
              "eager step-by-step recurrence within fp32/bf16 noise.")
    else:
        print("\nRESULT: CAPTURE SUCCEEDED BUT NUMERICS DIVERGED -- replay does not match the "
              "eager recurrence; state is being lost/stale between replays.")


if __name__ == "__main__":
    main()
