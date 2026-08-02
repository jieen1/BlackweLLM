"""Deliberate, reversible bug injection into this runtime's Qwen3.6 model.

Exists for one reason: a correctness gate nobody has ever seen go red is
not a gate. ``docs/e2e-and-quality-plan.md`` §4 makes this a standing rule
for every new gate in this repository ("任何新增门禁在合入前必须给出'如何
证明它会红'的方法"), and lists three acceptable proofs; this module is the
third one -- **known-bad input replay**. The B1-R criterion
(``bfdiag/divergence/logit_agreement.py``,
``docs/b1-correctness-criterion.md``) deliberately tolerates argmax flips
at the bf16 representation limit, because two correct implementations
provably produce them. That tolerance is exactly what has to be shown not
to swallow a real defect.

Every injection here is:

- **reversible** -- applied through a context manager that restores the
  exact objects it replaced, so one loaded model can be swept across many
  configurations without a reload (a 27B reload is minutes, and the HF
  reference side has to be evicted from the GPU to make room for it at
  all -- see ``scripts/b1_verify_greedy_alignment.py``'s docstring);
- **duck-typed** -- discovery is by attribute shape (``q_norm``/``k_norm``
  for attention, ``conv1d``+``A_log`` for GDN, ``cos_sin_cache`` for the
  text model), never by ``isinstance``, so the mechanism itself is
  testable on CPU against a lightweight fake module tree with no torch,
  no weights and no GPU (``tests/test_bfdiag_qwen36_bug_injection.py``);
- **magnitude-parameterised where it can be** -- ``rope-theta-rel`` and
  ``gdn-state-decay`` take a continuous knob, which is what turns "we
  injected a bug and the gate went red" into a *sensitivity curve*: sweep
  the knob down until the gate goes green again and you have measured the
  criterion's actual detection floor rather than asserted one.

Injections patch the **instance's** ``forward`` attribute rather than
swapping submodules out. ``torch.nn.Module.__setattr__`` refuses to
replace a registered child module with a non-Module, and swapping in a
real ``nn.Identity`` would need torch here; an instance-level ``forward``
override is honoured by ``Module.__call__``, needs nothing from torch,
and restores by deleting one attribute.

**These are bugs. Nothing in this module may ever be reachable from a
serving path** -- it is only ever invoked explicitly by name from a
verification script.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any

#: Injection names that take a ``:<magnitude>`` suffix.
_MAGNITUDE_REQUIRED = frozenset(
    {"rope-theta-rel", "rope-positions-offset", "gdn-state-decay", "gdn-state-stale-every"}
)
#: Injection names that take no magnitude.
_MAGNITUDE_FORBIDDEN = frozenset({"none", "drop-q-norm", "drop-k-norm"})

INJECTION_NAMES: tuple[str, ...] = tuple(sorted(_MAGNITUDE_REQUIRED | _MAGNITUDE_FORBIDDEN))


@dataclass(frozen=True)
class InjectionSpec:
    """A parsed ``name`` or ``name:magnitude`` injection request."""

    name: str
    magnitude: float | None = None

    def __str__(self) -> str:
        if self.magnitude is None:
            return self.name
        return f"{self.name}:{self.magnitude:g}"

    @property
    def is_control(self) -> bool:
        """True for the do-nothing configuration.

        ``rope-theta-rel:0`` counts as a control too, on purpose: sweeping
        a magnitude down to zero must land back on the unmodified model,
        and having that be the *same code path* as the real injection
        (rather than a special-cased skip) is what proves the injection
        machinery itself is not perturbing the measurement.
        """
        return self.name == "none" or (self.magnitude is not None and self.magnitude == 0.0)


def parse_injection(text: str) -> InjectionSpec:
    """Parse ``"none"``, ``"drop-q-norm"`` or ``"rope-theta-rel:1e-3"``."""
    text = text.strip()
    name, _, magnitude_text = text.partition(":")
    name = name.strip()
    if name not in INJECTION_NAMES:
        raise ValueError(f"unknown injection {name!r}; known: {', '.join(INJECTION_NAMES)}")
    if magnitude_text:
        if name in _MAGNITUDE_FORBIDDEN:
            raise ValueError(f"injection {name!r} takes no magnitude")
        try:
            magnitude = float(magnitude_text)
        except ValueError as exc:
            raise ValueError(f"magnitude {magnitude_text!r} is not a number") from exc
        return InjectionSpec(name=name, magnitude=magnitude)
    if name in _MAGNITUDE_REQUIRED:
        raise ValueError(f"injection {name!r} requires a ':<magnitude>' suffix")
    return InjectionSpec(name=name)


# --------------------------------------------------------------------------
# Discovery -- attribute shape only, never isinstance
# --------------------------------------------------------------------------


def _all_modules(model: Any) -> list[Any]:
    named = getattr(model, "named_modules", None)
    if named is None:
        raise TypeError("model must expose named_modules()")
    return [module for _name, module in named()]


def find_attention_modules(model: Any) -> list[Any]:
    """Every module carrying both ``q_norm`` and ``k_norm`` -- the
    full-attention layers (16 of Qwen3.6-27B's 64; the other 48 are GDN)."""
    return [m for m in _all_modules(model) if hasattr(m, "q_norm") and hasattr(m, "k_norm")]


def find_gdn_modules(model: Any) -> list[Any]:
    """Every module carrying both ``conv1d`` and ``A_log`` -- the gated
    delta-net layers."""
    return [m for m in _all_modules(model) if hasattr(m, "conv1d") and hasattr(m, "A_log")]


def find_rope_owner(model: Any) -> Any:
    """The single module holding the RoPE ``cos_sin_cache`` buffer."""
    owners = [m for m in _all_modules(model) if hasattr(m, "cos_sin_cache")]
    if len(owners) != 1:
        raise ValueError(f"expected exactly one cos_sin_cache owner, found {len(owners)}")
    return owners[0]


# --------------------------------------------------------------------------
# Primitive patches
# --------------------------------------------------------------------------


@contextmanager
def _patched_forward(module: Any, replacement: Callable[..., Any]) -> Iterator[None]:
    """Override ``module.forward`` on the instance, restoring exactly the
    prior state (including "there was no instance attribute at all")."""
    had_own = "forward" in vars(module)
    previous = vars(module).get("forward")
    object.__setattr__(module, "forward", replacement)
    try:
        yield
    finally:
        if had_own:
            object.__setattr__(module, "forward", previous)
        else:
            try:
                object.__delattr__(module, "forward")
            except AttributeError:  # pragma: no cover - defensive
                pass


@contextmanager
def _patched_attribute(owner: Any, name: str, value: Any) -> Iterator[None]:
    previous = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, previous)


# --------------------------------------------------------------------------
# Injections
# --------------------------------------------------------------------------


def _identity_forward(x: Any, *args: Any, **kwargs: Any) -> Any:
    return x


@contextmanager
def _drop_norm(model: Any, attribute: str) -> Iterator[None]:
    """Turn every attention layer's ``q_norm``/``k_norm`` into a pass-through.

    This is the classic "ported the layer but missed a normalisation"
    defect. It is deliberately a *structural* bug, not a numeric nudge:
    the point of including it is to establish the far end of the
    sensitivity range, so the continuous knobs' detection floors can be
    read against something unambiguous.
    """
    modules = find_attention_modules(model)
    if not modules:
        raise ValueError("no attention modules found (nothing carries q_norm/k_norm)")
    with ExitStack() as stack:
        for module in modules:
            stack.enter_context(_patched_forward(getattr(module, attribute), _identity_forward))
        yield


@contextmanager
def _rope_cache_replaced(model: Any, builder: Callable[[Any], Any]) -> Iterator[None]:
    owner = find_rope_owner(model)
    with _patched_attribute(owner, "cos_sin_cache", builder(owner)):
        yield


def _default_rope_rebuild(owner: Any, *, theta_scale: float) -> Any:
    """Rebuild the cos/sin cache with ``rope_theta * theta_scale``.

    Uses the runtime's own cache constructor (``runtime.kernels.rope``)
    rather than a re-derivation, so the injected model differs from the
    control in exactly one input value and nothing else.
    """
    from runtime.kernels.rope import compute_cos_sin_cache_default

    config = owner.config
    rope_params = config["rope_parameters"]
    cache = owner.cos_sin_cache
    return compute_cos_sin_cache_default(
        owner.rotary_dim,
        config["max_position_embeddings"],
        float(rope_params["rope_theta"]) * theta_scale,
        cache.dtype,
        device=cache.device,
    )


def _rolled_rope_cache(owner: Any, *, offset: int) -> Any:
    """Shift the whole cache up by ``offset`` rows.

    Equivalent to feeding every token ``position + offset``: row ``p`` of
    the cache is what position ``p`` would read, so reading row ``p`` from
    a cache rolled by ``-offset`` yields row ``p + offset``. The wrap-around
    at the very top is unreachable here -- ``max_position_embeddings`` is
    262144 and B1 runs a few hundred positions.
    """
    return owner.cos_sin_cache.roll(-offset, dims=0)


@contextmanager
def _gdn_state_decay(model: Any, factor: float) -> Iterator[None]:
    """Scale each GDN layer's persisted recurrent state by ``factor`` after
    every step.

    The single most B1-relevant failure shape: the recurrent state is the
    one thing that survives across decode steps, so an error in it
    *compounds* instead of averaging out. A factor of ``1 - 1e-4`` is far
    below any single step's rounding noise and is precisely the kind of
    bug a per-step comparison would miss and a trend test would not --
    which is why ``logit_agreement.WorkloadAgreement.drift_ratio`` exists.
    """
    modules = find_gdn_modules(model)
    if not modules:
        raise ValueError("no GDN modules found (nothing carries conv1d/A_log)")
    with ExitStack() as stack:
        for module in modules:
            stack.enter_context(_patched_forward(module, _decayed_gdn_forward(module, factor)))
        yield


def _decayed_gdn_forward(module: Any, factor: float) -> Callable[..., Any]:
    original = type(module).forward

    def patched(hidden_states: Any, state: Any) -> Any:
        out = original(module, hidden_states, state)
        state.recurrent_state = state.recurrent_state * factor
        return out

    return patched


@contextmanager
def _gdn_state_stale(model: Any, every: int) -> Iterator[None]:
    """Discard every ``every``-th GDN recurrent-state update.

    "GDN 状态少更新一步", made periodic so its severity is tunable: at
    ``every=1`` the state never advances at all (grossly broken); at
    ``every=64`` a single step in sixty-four is dropped, which no
    single-step logit comparison against a *matched* reference would
    ever notice on its own.
    """
    if every < 1:
        raise ValueError("'every' must be >= 1")
    modules = find_gdn_modules(model)
    if not modules:
        raise ValueError("no GDN modules found (nothing carries conv1d/A_log)")
    with ExitStack() as stack:
        for module in modules:
            stack.enter_context(_patched_forward(module, _stale_gdn_forward(module, every)))
        yield


def _stale_gdn_forward(module: Any, every: int) -> Callable[..., Any]:
    original = type(module).forward
    calls = [0]

    def patched(hidden_states: Any, state: Any) -> Any:
        previous_state = state.recurrent_state
        previous_flag = state.has_previous_state
        clone = getattr(previous_state, "clone", None)
        saved = clone() if callable(clone) else previous_state
        out = original(module, hidden_states, state)
        calls[0] += 1
        if calls[0] % every == 0:
            state.recurrent_state = saved
            state.has_previous_state = previous_flag
        return out

    return patched


@contextmanager
def injected(
    model: Any,
    spec: InjectionSpec,
    *,
    rope_builder: Callable[[Any, float], Any] | None = None,
) -> Iterator[None]:
    """Apply ``spec`` to ``model`` for the duration of the block.

    ``rope_builder(owner, theta_scale)`` overrides how the RoPE cache is
    rebuilt -- the only seam that genuinely needs torch, so overriding it
    is what lets the CPU tests exercise the rest of this module for real.
    """
    if spec.is_control:
        yield
        return

    name = spec.name
    magnitude = spec.magnitude

    if name == "drop-q-norm":
        with _drop_norm(model, "q_norm"):
            yield
    elif name == "drop-k-norm":
        with _drop_norm(model, "k_norm"):
            yield
    elif name == "rope-theta-rel":
        assert magnitude is not None
        scale = 1.0 + magnitude
        build = rope_builder or (lambda owner, s: _default_rope_rebuild(owner, theta_scale=s))
        with _rope_cache_replaced(model, lambda owner: build(owner, scale)):
            yield
    elif name == "rope-positions-offset":
        assert magnitude is not None
        offset = int(magnitude)
        if offset == 0:
            yield
            return
        with _rope_cache_replaced(model, lambda owner: _rolled_rope_cache(owner, offset=offset)):
            yield
    elif name == "gdn-state-decay":
        assert magnitude is not None
        with _gdn_state_decay(model, 1.0 - magnitude):
            yield
    elif name == "gdn-state-stale-every":
        assert magnitude is not None
        with _gdn_state_stale(model, int(magnitude)):
            yield
    else:  # pragma: no cover - parse_injection already rejects these
        raise ValueError(f"unhandled injection {name!r}")


def sweep_specs(name: str, magnitudes: Sequence[float]) -> tuple[InjectionSpec, ...]:
    """Build a magnitude sweep for one injection, control included first."""
    return (InjectionSpec("none"),) + tuple(
        InjectionSpec(name=name, magnitude=m) for m in magnitudes
    )
