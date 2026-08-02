"""CPU-only tests for the deliberate bug injector.

The *numeric* effect of an injection can only be seen on a real GPU with
real weights. What can and must be checked here is everything else: that
each injection is parsed correctly, finds the modules it claims to find,
actually changes behaviour while active, and -- the property the whole
sweep methodology depends on -- restores the model exactly on exit, so a
control run measured after an injected run is still a control run.

The fakes below are plain Python objects that merely *look* like the real
modules (``named_modules()``, ``q_norm``/``k_norm``, ``conv1d``/``A_log``,
``cos_sin_cache``), which is possible only because the injector discovers
by attribute shape and never by ``isinstance``.
"""

from __future__ import annotations

import pytest

from bfdiag.divergence.qwen36_bug_injection import (
    INJECTION_NAMES,
    InjectionSpec,
    find_attention_modules,
    find_gdn_modules,
    find_rope_owner,
    injected,
    parse_injection,
    sweep_specs,
)


class FakeModule:
    """Minimal stand-in for ``torch.nn.Module``'s calling convention."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - overridden
        raise NotImplementedError


class FakeNorm(FakeModule):
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def forward(self, x):
        return f"{self.tag}({x})"


class FakeAttention(FakeModule):
    def __init__(self, idx: int) -> None:
        self.idx = idx
        self.q_norm = FakeNorm("qnorm")
        self.k_norm = FakeNorm("knorm")

    def forward(self, x):
        return self.q_norm(self.k_norm(x))


class FakeState:
    def __init__(self, value: float) -> None:
        self.recurrent_state = value
        self.has_previous_state = False


class FakeGdn(FakeModule):
    """Advances ``state.recurrent_state`` by 1 per call, like the real
    layer's ``state.recurrent_state = last_state.to(...)`` writeback."""

    def __init__(self, idx: int) -> None:
        self.idx = idx
        self.conv1d = object()
        self.A_log = object()

    def forward(self, hidden_states, state):
        state.recurrent_state = state.recurrent_state + 1
        state.has_previous_state = True
        return hidden_states


class FakeTextModel(FakeModule):
    def __init__(self) -> None:
        self.cos_sin_cache = "CACHE"
        self.rotary_dim = 64
        self.config = {
            "max_position_embeddings": 262144,
            "rope_parameters": {"rope_theta": 1e7},
        }


class FakeModel:
    def __init__(self, *, num_attn: int = 2, num_gdn: int = 3) -> None:
        self.text_model = FakeTextModel()
        self.attentions = [FakeAttention(i) for i in range(num_attn)]
        self.gdns = [FakeGdn(i) for i in range(num_gdn)]

    def named_modules(self):
        yield "model", self.text_model
        for i, module in enumerate(self.attentions):
            yield f"layers.{i}.self_attn", module
            yield f"layers.{i}.self_attn.q_norm", module.q_norm
            yield f"layers.{i}.self_attn.k_norm", module.k_norm
        for i, module in enumerate(self.gdns):
            yield f"layers.{i}.linear_attn", module


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_every_declared_injection_parses() -> None:
    for name in INJECTION_NAMES:
        text = name if name in {"none", "drop-q-norm", "drop-k-norm"} else f"{name}:1"
        assert parse_injection(text).name == name


def test_parse_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown injection"):
        parse_injection("make-it-fast")


def test_parse_requires_a_magnitude_where_one_is_meaningful() -> None:
    with pytest.raises(ValueError, match="requires a ':<magnitude>' suffix"):
        parse_injection("rope-theta-rel")


def test_parse_rejects_a_magnitude_where_none_is_meaningful() -> None:
    with pytest.raises(ValueError, match="takes no magnitude"):
        parse_injection("drop-q-norm:2")


def test_parse_rejects_a_non_numeric_magnitude() -> None:
    with pytest.raises(ValueError, match="not a number"):
        parse_injection("rope-theta-rel:lots")


def test_zero_magnitude_is_a_control() -> None:
    assert parse_injection("rope-theta-rel:0").is_control
    assert parse_injection("none").is_control
    assert not parse_injection("rope-theta-rel:1e-3").is_control


def test_spec_str_round_trips() -> None:
    for text in ("none", "drop-q-norm", "rope-theta-rel:0.001", "gdn-state-stale-every:64"):
        assert str(parse_injection(text)) == text


def test_sweep_specs_starts_from_the_control() -> None:
    specs = sweep_specs("rope-theta-rel", [1e-4, 1e-3])
    assert specs[0] == InjectionSpec("none")
    assert [s.magnitude for s in specs[1:]] == [1e-4, 1e-3]


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def test_discovery_finds_the_right_modules_by_shape() -> None:
    model = FakeModel(num_attn=4, num_gdn=7)
    assert len(find_attention_modules(model)) == 4
    assert len(find_gdn_modules(model)) == 7
    assert find_rope_owner(model) is model.text_model


def test_discovery_rejects_a_model_without_named_modules() -> None:
    with pytest.raises(TypeError, match="named_modules"):
        find_attention_modules(object())


def test_rope_owner_must_be_unique() -> None:
    model = FakeModel()
    model.attentions[0].cos_sin_cache = "SECOND"
    with pytest.raises(ValueError, match="exactly one cos_sin_cache"):
        find_rope_owner(model)


def test_injection_errors_when_its_target_is_absent() -> None:
    model = FakeModel(num_attn=0, num_gdn=0)
    with pytest.raises(ValueError, match="no attention modules"):
        with injected(model, parse_injection("drop-q-norm")):
            pass
    with pytest.raises(ValueError, match="no GDN modules"):
        with injected(model, parse_injection("gdn-state-decay:0.1")):
            pass


# --------------------------------------------------------------------------
# effect + restoration
# --------------------------------------------------------------------------


def test_control_changes_nothing() -> None:
    model = FakeModel()
    with injected(model, parse_injection("none")):
        assert model.attentions[0]("x") == "qnorm(knorm(x))"


def test_drop_q_norm_bypasses_only_q_norm_and_restores() -> None:
    model = FakeModel()
    with injected(model, parse_injection("drop-q-norm")):
        for attn in model.attentions:
            assert attn("x") == "knorm(x)"
    for attn in model.attentions:
        assert attn("x") == "qnorm(knorm(x))"


def test_drop_k_norm_bypasses_only_k_norm_and_restores() -> None:
    model = FakeModel()
    with injected(model, parse_injection("drop-k-norm")):
        assert model.attentions[0]("x") == "qnorm(x)"
    assert model.attentions[0]("x") == "qnorm(knorm(x))"


def test_rope_theta_injection_swaps_the_cache_and_restores() -> None:
    model = FakeModel()
    seen: list[tuple[object, float]] = []

    def builder(owner, scale):
        seen.append((owner, scale))
        return f"CACHE@{scale}"

    spec = parse_injection("rope-theta-rel:0.01")
    with injected(model, spec, rope_builder=builder):
        assert model.text_model.cos_sin_cache == "CACHE@1.01"
    assert model.text_model.cos_sin_cache == "CACHE"
    assert seen == [(model.text_model, 1.01)]


def test_rope_positions_offset_zero_is_a_no_op() -> None:
    model = FakeModel()
    with injected(model, InjectionSpec("rope-positions-offset", 0.0)):
        assert model.text_model.cos_sin_cache == "CACHE"


def test_gdn_state_decay_scales_the_persisted_state_and_restores() -> None:
    model = FakeModel(num_gdn=1)
    gdn = model.gdns[0]
    state = FakeState(10.0)

    with injected(model, parse_injection("gdn-state-decay:0.5")):
        gdn("h", state)
        # real writeback 10 -> 11, then decayed by (1 - 0.5)
        assert state.recurrent_state == pytest.approx(5.5)

    state = FakeState(10.0)
    gdn("h", state)
    assert state.recurrent_state == pytest.approx(11.0)


def test_gdn_state_stale_drops_every_nth_update() -> None:
    model = FakeModel(num_gdn=1)
    gdn = model.gdns[0]
    state = FakeState(0.0)

    with injected(model, parse_injection("gdn-state-stale-every:3")):
        seen = []
        for _ in range(6):
            gdn("h", state)
            seen.append(state.recurrent_state)
    # steps 3 and 6 have their writeback discarded, so the state falls
    # two full updates behind over six calls.
    assert seen == [1.0, 2.0, 2.0, 3.0, 4.0, 4.0]


def test_gdn_state_stale_every_one_freezes_the_state_entirely() -> None:
    model = FakeModel(num_gdn=1)
    state = FakeState(0.0)
    with injected(model, parse_injection("gdn-state-stale-every:1")):
        for _ in range(5):
            model.gdns[0]("h", state)
        assert state.recurrent_state == 0.0


def test_gdn_stale_restores_the_has_previous_state_flag_too() -> None:
    model = FakeModel(num_gdn=1)
    state = FakeState(0.0)
    with injected(model, parse_injection("gdn-state-stale-every:1")):
        model.gdns[0]("h", state)
        assert state.has_previous_state is False
    model.gdns[0]("h", state)
    assert state.has_previous_state is True


def test_injections_restore_even_when_the_body_raises() -> None:
    model = FakeModel()
    with pytest.raises(RuntimeError):
        with injected(model, parse_injection("drop-q-norm")):
            raise RuntimeError("boom")
    assert model.attentions[0]("x") == "qnorm(knorm(x))"


def test_nested_and_repeated_injections_leave_no_residue() -> None:
    """The sweep runs many configurations against one loaded model, so a
    leaked patch would silently contaminate every later measurement --
    including the control."""
    model = FakeModel(num_gdn=1)
    for _ in range(3):
        with injected(model, parse_injection("drop-q-norm")):
            with injected(model, parse_injection("drop-k-norm")):
                assert model.attentions[0]("x") == "x"
            assert model.attentions[0]("x") == "knorm(x)"
        assert model.attentions[0]("x") == "qnorm(knorm(x))"
        state = FakeState(0.0)
        model.gdns[0]("h", state)
        assert state.recurrent_state == 1.0
