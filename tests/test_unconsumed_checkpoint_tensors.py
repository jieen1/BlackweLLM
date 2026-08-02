"""A checkpoint tensor nobody consumes must not be invisible.

``assert_all_params_loaded`` checks one direction: every model Parameter got a
checkpoint tensor. Nothing checked the reverse, and the reverse has no natural
symptom -- an unread tensor is simply never read. There is no exception, no
shape error, no log line.

That is not hypothetical. ``CompressedTensorsNVFP4Linear`` never created an
``input_global_scale`` Parameter, so the standard checkpoint's 56 activation-
scale tensors were dropped silently for as long as that class existed. It was
found only because a W4A4 investigation went looking for that scale and
noticed it was not there. It happened to be harmless -- the W4A16 path has no
use for an activation scale -- but nothing at the time distinguished "we do
not need this" from "we should be using this and are not", and a scale silently
absent from a quantized matmul does not crash, it changes the arithmetic.

Why families and not names: loaders remap. The checkpoint's spelling and the
model's Parameter names are different naming schemes, so differencing the two
name sets would compare things that were never meant to match. The trailing
component survives remapping -- if a checkpoint ships ``.input_global_scale``
and no Parameter or buffer name ends with it, then nothing consumed it no
matter how names were rewritten.

Why it warns instead of raising, unlike its sibling: a Parameter with no tensor
is unambiguously broken (the weight is random). An unconsumed tensor is often
legitimate -- checkpoints ship tensors for features a given build does not
enable. Raising would convert "this checkpoint offers more than we use" into a
load failure. Visibility was the missing thing, not enforcement.

Pure set arithmetic over tiny fake modules: no torch model, no checkpoint,
no GPU.
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch", reason="torch-free CI job")

from runtime.loading.common import (  # noqa: E402
    record_checkpoint_tensor_names,
    warn_on_unconsumed_tensor_families,
)


def _model(param_suffixes, buffer_suffixes=()):
    """Build a module whose named_parameters end in the given suffixes."""
    m = torch.nn.Module()
    for suffix in param_suffixes:
        holder = torch.nn.Module()
        holder.register_parameter(suffix, torch.nn.Parameter(torch.zeros(1)))
        m.add_module(f"proj_{suffix}", holder)
    for suffix in buffer_suffixes:
        holder = torch.nn.Module()
        holder.register_buffer(suffix, torch.zeros(1))
        m.add_module(f"buf_{suffix}", holder)
    return m


class TestDetection:
    def test_the_input_global_scale_regression_is_caught(self):
        """The exact case that went unnoticed: model has no such Parameter."""
        model = _model(("weight_packed", "weight_scale", "weight_global_scale"))
        seen = {
            f"model.layers.{i}.mlp.gate_proj.{suffix}"
            for i in range(56)
            for suffix in (
                "weight_packed",
                "weight_scale",
                "weight_global_scale",
                "input_global_scale",
            )
        }
        unconsumed = warn_on_unconsumed_tensor_families(model, seen, context="t")
        assert unconsumed == frozenset({"input_global_scale"})

    def test_fully_consumed_checkpoint_reports_nothing(self):
        model = _model(("weight_packed", "weight_scale", "weight_global_scale"))
        seen = {
            f"model.layers.0.mlp.gate_proj.{s}"
            for s in ("weight_packed", "weight_scale", "weight_global_scale")
        }
        assert warn_on_unconsumed_tensor_families(model, seen, context="t") == frozenset()

    def test_buffers_count_as_consumers(self):
        """``_k_scale``-style buffers are real consumers, not gaps.

        ``apply_kv_cache_scale_post_load`` moves checkpoint values into
        registered buffers; treating only Parameters as consumers would report
        those as dropped.
        """
        model = _model(("weight",), buffer_suffixes=("k_scale",))
        seen = {"model.layers.0.self_attn.weight", "model.layers.0.self_attn.k_scale"}
        assert warn_on_unconsumed_tensor_families(model, seen, context="t") == frozenset()

    def test_expected_unconsumed_suppresses_known_cases(self):
        """A build that deliberately ignores a family can say so."""
        model = _model(("weight",))
        seen = {"model.layers.0.weight", "mtp.0.embed.weight_scale"}
        assert (
            warn_on_unconsumed_tensor_families(
                model, seen, context="t", expected_unconsumed=frozenset({"weight_scale"})
            )
            == frozenset()
        )

    def test_it_warns_rather_than_raises(self, caplog):
        """The whole point is visibility without turning extra tensors into a failure."""
        model = _model(("weight",))
        seen = {"model.layers.0.weight", "model.layers.0.input_global_scale"}
        with caplog.at_level(logging.WARNING, logger="runtime.loading.common"):
            result = warn_on_unconsumed_tensor_families(model, seen, context="load_x")
        assert result == frozenset({"input_global_scale"})
        assert any("input_global_scale" in r.getMessage() for r in caplog.records), (
            "the warning must name the family; a count alone is not actionable"
        )


class TestRecorder:
    def test_it_passes_tensors_through_unchanged(self):
        src = [("a.weight", torch.zeros(2)), ("b.weight_scale", torch.ones(3))]
        sink: set[str] = set()
        out = list(record_checkpoint_tensor_names(iter(src), sink))
        assert [n for n, _ in out] == ["a.weight", "b.weight_scale"]
        assert torch.equal(out[1][1], torch.ones(3))
        assert sink == {"a.weight", "b.weight_scale"}

    def test_it_stays_lazy(self):
        """Must not materialize the checkpoint -- the iterator it wraps exists
        precisely so a ~67 GiB checkpoint never lands in host RAM at once."""
        sink: set[str] = set()
        consumed: list[str] = []

        def src():
            for name in ("a.weight", "b.weight", "c.weight"):
                consumed.append(name)
                yield name, torch.zeros(1)

        gen = record_checkpoint_tensor_names(src(), sink)
        next(gen)
        assert consumed == ["a.weight"], "wrapping must not drain the source eagerly"
        assert sink == {"a.weight"}
