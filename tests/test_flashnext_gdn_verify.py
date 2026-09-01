from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
    pytest.skip("Flash-Next GDN verify kernel requires SM120 CUDA", allow_module_level=True)

from runtime.kernels.flashnext_gdn_verify import (  # noqa: E402
    flashnext_gdn_commit,
    flashnext_gdn_verify,
)


def _reference(q, k, v, a, b, a_log, dt_bias, initial_state):
    state = initial_state.clone()
    outputs = []
    states = []
    repeat = v.shape[2] // q.shape[2]
    q = q.repeat_interleave(repeat, dim=2).float()
    k = k.repeat_interleave(repeat, dim=2).float()
    for step in range(q.shape[1]):
        query = q[:, step] / torch.sqrt((q[:, step] ** 2).sum(dim=-1, keepdim=True) + 1e-6)
        key = k[:, step] / torch.sqrt((k[:, step] ** 2).sum(dim=-1, keepdim=True) + 1e-6)
        decay = -torch.exp(a_log.float()) * torch.nn.functional.softplus(
            a[:, step].float() + dt_bias.float()
        )
        beta = torch.sigmoid(b[:, step].float())
        state = state * torch.exp(decay)[..., None, None]
        delta = v[:, step].float() - torch.einsum("bhkv,bhk->bhv", state, key)
        delta = delta * beta[..., None]
        state = state + key[..., None] * delta[..., None, :]
        outputs.append(
            torch.einsum("bhkv,bhk->bhv", state, query * (q.shape[-1] ** -0.5))
        )
        states.append(state.clone())
    return torch.stack(outputs, dim=1).bfloat16(), torch.stack(states, dim=1)


def test_fp32_verify_matches_recurrent_reference_and_stores_every_step():
    torch.manual_seed(20260828)
    batch, steps, heads, value_heads, dim = 1, 4, 2, 6, 128
    q = torch.randn(batch, steps, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(
        batch, steps, value_heads, dim, device="cuda", dtype=torch.bfloat16
    )
    a = torch.randn(batch, steps, value_heads, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a_log = torch.randn(value_heads, device="cuda", dtype=torch.float32) - 2
    dt_bias = torch.randn(value_heads, device="cuda", dtype=torch.float32)
    initial = torch.randn(
        batch, value_heads, dim, dim, device="cuda", dtype=torch.float32
    ) * 0.01
    candidate_states = torch.empty(
        batch, steps, value_heads, dim, dim, device="cuda", dtype=torch.float32
    )

    expected_output, expected_states = _reference(q, k, v, a, b, a_log, dt_bias, initial)
    output = flashnext_gdn_verify(
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        a_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial,
        intermediate_states=candidate_states,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output.float(), expected_output.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(candidate_states, expected_states, rtol=2e-5, atol=2e-5)

    for accepted_count in range(1, steps + 1):
        recomputed = initial.clone()
        commit_scratch = torch.empty_like(candidate_states)
        commit_output = torch.empty_like(v)
        flashnext_gdn_commit(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            a_log=a_log,
            dt_bias=dt_bias,
            state=recomputed,
            accepted_count=accepted_count,
            scratch_states=commit_scratch,
            scratch_output=commit_output,
        )
        torch.cuda.synchronize()
        expected = candidate_states[:, accepted_count - 1]
        assert torch.equal(recomputed, expected), (
            accepted_count,
            int(torch.count_nonzero(recomputed != expected)),
            float((recomputed - expected).abs().max()),
        )
