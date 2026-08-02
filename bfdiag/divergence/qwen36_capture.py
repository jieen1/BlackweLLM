"""Qwen3.6 (Track B / B1) capture sources for ``bfdiag.divergence.scan``.

Wires the B1 correctness gate's "逐层 logits 余弦相似度进 bfdiag" requirement
(``docs/implementation-plan.md`` §7.1) into the *existing*, already-generic
``bfdiag.divergence`` machinery -- ``scan_layers``/``thresholds``/``report``
needed zero changes: they operate on plain ``ActivationTrace = Mapping[int,
Mapping[str, Any]]`` values (see ``bfdiag/divergence/scan.py``'s module
docstring), and ``thresholds.py`` already ships a ``HIDDEN_STATE`` kind with
a calibrated threshold (``LayerThreshold(0.999, 0.02, 0.98, 0.95)``) -- this
module is purely the capture-side glue for a second architecture, following
the exact shape ``bfdiag/divergence/capture.py``'s ``EngineCaptureSource``/
``capture_engine_activations`` already established for Laguna.

Two ``CaptureSource``-conformant sources:

- :class:`Qwen36EngineCaptureSource` wraps this runtime's own
  ``Qwen36ForCausalLMSelfBuilt`` (``runtime/model/qwen36_model.py``),
  using its ``capture_hidden_states=True`` forward option -- no forward
  hooks needed (unlike Laguna's ``ForwardCapture``), since B1's model
  graph already threads per-layer hidden states through explicitly.
- :class:`Qwen36HFOracleCaptureSource` wraps HF's own
  ``Qwen3_5ForCausalLM`` (``transformers==5.8.0``,
  ``transformers/models/qwen3_5/modeling_qwen3_5.py``, read-only
  reference per ``docs/qwen36-rebuild-spec.md`` §1.0) via its standard
  ``output_hidden_states=True`` contract -- this is the "oracle" side.

**GPU-only, not yet exercised against a real GPU/model in this pass** --
see ``tests/test_bfdiag_qwen36_capture.py`` for what IS exercised on CPU
(the trace-shape/indexing glue, via a fake stand-in object) and the B1
handoff notes for exactly what remains: pointing this at a real loaded
``Qwen36ForCausalLMSelfBuilt`` + a real HF reference for >=3 real prompts
and reading the resulting :class:`bfdiag.divergence.scan.DivergenceReport`.

**Layer-indexing contract, the one thing most likely to be gotten wrong
here without a live check** -- documented explicitly because it was
designed, not measured against a real HF forward pass in this pass:
``Qwen36ForCausalLMSelfBuilt``'s ``capture_hidden_states=True`` returns a
list where index ``i`` is the residual-stream hidden state immediately
AFTER decoder layer ``i`` (before the model's final norm) -- see
``Qwen36TextModelSelfBuilt.forward``. HF's ``output_hidden_states=True``
returns a tuple of length ``num_layers + 1`` under the standard HF
convention (index 0 = embedding output, index ``i + 1`` = after layer
``i``) -- this module reads ``hidden_states[i + 1]`` to line up with
``i``. **This offset-by-one convention is a documented HF norm, not
independently confirmed against this specific model class in this pass**
(some HF model families apply the final norm to the last
``output_hidden_states`` entry, some don't; Qwen3_5TextModel's own
behavior here was not checked against a live run before this was
written) -- the first thing to verify on GPU, before trusting any
reported divergence layer number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bfdiag.divergence.capture import ActivationTrace

#: One past the last real decoder layer -- the logits comparison point
#: (unambiguous on both sides: this runtime's own
#: ``Qwen36ForCausalLMSelfBuilt.compute_logits`` vs HF's ``outputs.logits``),
#: kept out of the 0..num_layers-1 hidden-state range so it never collides
#: with a real layer index. ``thresholds.kind_for_submodule("logits")``
#: falls through to ``_DEFAULT_KIND_THRESHOLD`` today (no dedicated
#: "logits" kind exists yet) -- a real gap, not an oversight: see this
#: module's TODO note below.
LOGITS_SENTINEL_OFFSET = 0


def _logits_layer_index(num_layers: int) -> int:
    return num_layers + LOGITS_SENTINEL_OFFSET


def build_trace_from_captured_values(
    per_layer_hidden: list[Any], last_token_logits: Any
) -> dict[int, dict[str, Any]]:
    """Pure trace-shape glue: given already-computed per-layer hidden
    states (index ``i`` = after decoder layer ``i``) and the final
    position's logits vector, build the ``{layer_idx: {submodule:
    value}}`` shape ``scan_layers`` expects, appending the logits at a
    sentinel index one past the last real layer.

    Deliberately takes no ``torch``/model/prompt arguments and does no
    forward-pass work -- this is the one piece of both capture functions
    below that has nothing to do with running a model, so it is the one
    piece exercised on CPU without ``torch`` (see
    ``tests/test_bfdiag_qwen36_capture.py``): callers pass plain Python
    values in tests, real tensors in production, this function does not
    care which.
    """
    trace: dict[int, dict[str, Any]] = {
        layer_idx: {"hidden_state": hidden} for layer_idx, hidden in enumerate(per_layer_hidden)
    }
    trace[_logits_layer_index(len(per_layer_hidden))] = {"logits": last_token_logits}
    return trace


def capture_qwen36_engine_activations(
    model: Any,
    prompt_token_ids: list[int],
    *,
    device: Any = "cuda",
    dtype: Any = None,
) -> dict[int, dict[str, Any]]:
    """Capture this runtime's own ``Qwen36ForCausalLMSelfBuilt`` per-layer
    hidden states (plus final logits) for one prompt, single prefill call.

    ``model`` is duck-typed (``Any``) on purpose, matching ``capture.py``'s
    existing ``EngineCaptureSource``/``capture_engine_activations`` style
    -- this keeps this module import-safe on CPU (no ``torch``/``runtime``
    import at module level; see the module docstring and
    ``tests/test_bfdiag_qwen36_capture.py`` for what that buys). ``torch``
    is only imported inside this function body, at call time.

    ``dtype`` defaults to ``torch.bfloat16`` (this runtime's one real
    dtype, per ``runtime/model/qwen36_model.py``'s module docstring) if
    left ``None`` -- resolved lazily so importing this module never
    requires ``torch`` to exist at all.
    """
    import torch

    if dtype is None:
        dtype = torch.bfloat16

    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    state = model.new_generation_state(device=device, dtype=dtype)
    final_hidden, per_layer_hidden = model(input_ids, state, capture_hidden_states=True)
    logits = model.compute_logits(final_hidden)

    return build_trace_from_captured_values(
        [hidden[0] for hidden in per_layer_hidden], logits[0, -1]
    )


def capture_qwen36_hf_oracle_activations(
    hf_model: Any,
    prompt_token_ids: list[int],
    *,
    device: Any = "cuda",
) -> dict[int, dict[str, Any]]:
    """Capture HF's ``Qwen3_5ForCausalLM`` per-layer hidden states (plus
    final logits) for the same prompt, via ``output_hidden_states=True``.

    ``hf_model`` is duck-typed for the same import-safety reason as
    :func:`capture_qwen36_engine_activations`. See this module's
    docstring for the layer-indexing convention this assumes and has not
    yet independently confirmed against a live HF forward pass.
    """
    import torch

    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = hf_model(
            input_ids, output_hidden_states=True, use_cache=False, logits_to_keep=1
        )

    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states) - 1
    per_layer_hidden = [hidden_states[layer_idx + 1][0] for layer_idx in range(num_layers)]
    return build_trace_from_captured_values(per_layer_hidden, outputs.logits[0, -1])


@dataclass
class Qwen36EngineCaptureSource:
    """``CaptureSource`` wrapping this runtime's own Qwen3.6 model.

    See :func:`capture_qwen36_engine_activations` for what this actually
    does; this class only adds the ``CaptureSource`` protocol shape
    (``bfdiag/divergence/capture.py``) so it drops straight into
    ``bfdiag.divergence.cli.scan_prompt`` alongside the oracle source.
    """

    model: Any
    device: Any = "cuda"
    dtype: Any = None

    def capture(self, prompt_token_ids: list[int]) -> ActivationTrace:
        return capture_qwen36_engine_activations(
            self.model, prompt_token_ids, device=self.device, dtype=self.dtype
        )


@dataclass
class Qwen36HFOracleCaptureSource:
    """``CaptureSource`` wrapping an HF ``Qwen3_5ForCausalLM`` reference."""

    hf_model: Any
    device: Any = "cuda"

    def capture(self, prompt_token_ids: list[int]) -> ActivationTrace:
        return capture_qwen36_hf_oracle_activations(
            self.hf_model, prompt_token_ids, device=self.device
        )
