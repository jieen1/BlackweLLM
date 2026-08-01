"""Tests for runtime/sampling.py — CPU-only, no GPU required."""

from __future__ import annotations

import pytest

from runtime.sampling import SamplingParams


class TestSamplingParams:
    def test_default_is_greedy(self):
        params = SamplingParams()
        assert params.is_greedy
        assert params.temperature == 0.0
        assert params.top_k == 0
        assert params.top_p == 1.0
        assert params.seed is None

    def test_zero_temperature_is_greedy(self):
        assert SamplingParams(temperature=0.0).is_greedy

    def test_negative_temperature_is_greedy(self):
        assert SamplingParams(temperature=-1.0).is_greedy

    def test_positive_temperature_is_not_greedy(self):
        assert not SamplingParams(temperature=0.7).is_greedy

    def test_validate_ok(self):
        SamplingParams(temperature=0.8, top_k=50, top_p=0.95).validate()

    def test_validate_negative_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            SamplingParams(temperature=-0.1).validate()

    def test_validate_negative_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            SamplingParams(top_k=-1).validate()

    def test_validate_top_p_zero(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=0.0).validate()

    def test_validate_top_p_above_one(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=1.1).validate()

    def test_frozen(self):
        params = SamplingParams()
        with pytest.raises(AttributeError):
            params.temperature = 1.0


class TestSamplingParamsTorch:
    """Tests that require torch (auto-skipped in CPU-only envs without torch)."""

    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

    def test_greedy_matches_argmax(self):
        import torch

        from runtime.sampling import sample_from_logits

        logits = torch.randn(4, 100)
        params = SamplingParams(temperature=0.0)
        result = sample_from_logits(logits, params)
        expected = logits.argmax(dim=-1)
        assert torch.equal(result, expected)

    def test_greedy_deterministic(self):
        import torch

        from runtime.sampling import sample_from_logits

        logits = torch.randn(2, 50)
        params = SamplingParams(temperature=0.0)
        r1 = sample_from_logits(logits, params)
        r2 = sample_from_logits(logits, params)
        assert torch.equal(r1, r2)

    def test_sampling_with_seed_reproducible(self):
        import torch

        from runtime.sampling import make_generator, sample_from_logits

        logits = torch.randn(1, 1000)
        params = SamplingParams(temperature=1.0, seed=42)
        gen1 = make_generator(params.seed)
        gen2 = make_generator(params.seed)
        r1 = sample_from_logits(logits, params, generator=gen1)
        r2 = sample_from_logits(logits, params, generator=gen2)
        assert torch.equal(r1, r2)

    def test_sampling_different_seeds_differ(self):
        import torch

        from runtime.sampling import make_generator, sample_from_logits

        logits = torch.randn(1, 10000)
        params = SamplingParams(temperature=1.0)
        gen1 = make_generator(1)
        gen2 = make_generator(2)
        r1 = sample_from_logits(logits, params, generator=gen1)
        r2 = sample_from_logits(logits, params, generator=gen2)
        assert not torch.equal(r1, r2)

    def test_top_k_restricts_candidates(self):
        import torch

        from runtime.sampling import make_generator, sample_from_logits

        logits = torch.zeros(1, 100)
        logits[0, 42] = 10.0
        logits[0, 7] = 9.0
        params = SamplingParams(temperature=1.0, top_k=2, seed=0)
        gen = make_generator(params.seed)
        for _ in range(20):
            result = sample_from_logits(logits, params, generator=gen)
            assert result.item() in (42, 7)

    def test_top_p_restricts_candidates(self):
        import torch

        from runtime.sampling import make_generator, sample_from_logits

        logits = torch.zeros(1, 100)
        logits[0, 0] = 100.0
        params = SamplingParams(temperature=1.0, top_p=0.5, seed=0)
        gen = make_generator(params.seed)
        for _ in range(20):
            result = sample_from_logits(logits, params, generator=gen)
            assert result.item() == 0

    def test_temperature_scaling_effect(self):
        import torch

        from runtime.sampling import make_generator, sample_from_logits

        logits = torch.tensor([[1.0, 2.0, 3.0]])
        low_temp = SamplingParams(temperature=0.01)
        high_temp = SamplingParams(temperature=100.0)
        low_counts = {0: 0, 1: 0, 2: 0}
        high_counts = {0: 0, 1: 0, 2: 0}
        for i in range(200):
            gen_low = make_generator(i)
            gen_high = make_generator(i)
            low_counts[sample_from_logits(logits, low_temp, generator=gen_low).item()] += 1
            high_counts[sample_from_logits(logits, high_temp, generator=gen_high).item()] += 1
        assert low_counts[2] > low_counts[0]
        assert high_counts[0] > 0


class TestPersistentSeed:
    """N3: seed must advance ONE generator across a request's decode
    rounds, not reseed an identical initial state at every token -- see
    ``PersistentSeed``'s docstring in runtime/sampling.py."""

    def test_repeated_make_generator_calls_return_same_object(self):
        pytest.importorskip("torch")
        from runtime.sampling import PersistentSeed, make_generator

        seed = PersistentSeed(42)
        gen1 = make_generator(seed, device="cpu")
        gen2 = make_generator(seed, device="cpu")
        assert gen1 is gen2

    def test_successive_draws_differ_unlike_plain_int_reseed(self):
        """The bug this fixes: with a plain int, calling make_generator +
        drawing repeatedly gives the SAME draw every time (reseed-per-call).
        With a PersistentSeed, successive draws through the SAME wrapper
        advance and differ."""
        torch = pytest.importorskip("torch")
        from runtime.sampling import PersistentSeed, make_generator, sample_from_logits

        params = SamplingParams(temperature=1.0)
        logits = torch.randn(1, 5000)

        # Old (buggy) behavior with a plain int: every call reseeds to the
        # same state -> identical draws.
        plain_draws = []
        for _ in range(3):
            gen = make_generator(7, device="cpu")
            plain_draws.append(sample_from_logits(logits, params, generator=gen).item())
        assert plain_draws[0] == plain_draws[1] == plain_draws[2]

        # New behavior with PersistentSeed: the SAME wrapper instance
        # advances across calls -> draws are not all identical.
        seed = PersistentSeed(7)
        persistent_draws = []
        for _ in range(3):
            gen = make_generator(seed, device="cpu")
            persistent_draws.append(sample_from_logits(logits, params, generator=gen).item())
        assert len(set(persistent_draws)) > 1

    def test_reproducible_across_two_fresh_persistent_seeds(self):
        """Two independent PersistentSeed(42) instances (e.g. two different
        requests both passing seed=42) each replay the exact same sequence
        of draws -- reproducibility is preserved, just per-instance instead
        of per-call."""
        torch = pytest.importorskip("torch")
        from runtime.sampling import PersistentSeed, make_generator, sample_from_logits

        params = SamplingParams(temperature=1.0)
        logits = torch.randn(1, 5000)

        def draw_sequence(seed_value: int, n: int) -> list[int]:
            seed = PersistentSeed(seed_value)
            out = []
            for _ in range(n):
                gen = make_generator(seed, device="cpu")
                out.append(sample_from_logits(logits, params, generator=gen).item())
            return out

        assert draw_sequence(42, 5) == draw_sequence(42, 5)

    def test_two_concurrent_requests_same_seed_do_not_share_state(self):
        """Two PersistentSeed(42) instances used INTERLEAVED (simulating two
        concurrent requests sharing the same user-supplied seed value) do
        not interfere with each other: each still reproduces its own
        independent, self-consistent stream."""
        torch = pytest.importorskip("torch")
        from runtime.sampling import PersistentSeed, make_generator, sample_from_logits

        params = SamplingParams(temperature=1.0)
        logits = torch.randn(1, 5000)

        seed_a = PersistentSeed(42)
        seed_b = PersistentSeed(42)
        draws_a, draws_b = [], []
        for _ in range(4):
            gen_a = make_generator(seed_a, device="cpu")
            draws_a.append(sample_from_logits(logits, params, generator=gen_a).item())
            gen_b = make_generator(seed_b, device="cpu")
            draws_b.append(sample_from_logits(logits, params, generator=gen_b).item())

        # Same seed value, interleaved 1:1 -> each stream independently
        # matches what an uninterrupted run of that seed alone would give.
        seed_solo = PersistentSeed(42)
        draws_solo = [
            sample_from_logits(
                logits, params, generator=make_generator(seed_solo, device="cpu")
            ).item()
            for _ in range(4)
        ]
        assert draws_a == draws_solo
        assert draws_b == draws_solo

    def test_greedy_path_never_touches_seed(self):
        """Bit-exact greedy requirement: is_greedy short-circuits to argmax
        before make_generator/seed is ever consulted."""
        torch = pytest.importorskip("torch")
        from runtime.sampling import PersistentSeed, sample_from_logits

        logits = torch.randn(3, 37)
        params = SamplingParams(temperature=0.0, seed=PersistentSeed(1))
        result = sample_from_logits(logits, params)
        assert torch.equal(result, logits.argmax(dim=-1))
