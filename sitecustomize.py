"""Temporary subprocess hook for the local vLLM DSpark comparison.

This file is intentionally opt-in and is removed after the comparison run.
Python imports ``sitecustomize`` in vLLM's EngineCore subprocess, which does
not inherit the monkeypatches installed by the API-server wrapper module.
"""

import inspect
import os
from pathlib import Path

if os.environ.get("QSR_DSPARK_DUMP_ALL_LAYERS") not in (None, "", "0"):
    import torch
    from vllm.model_executor.models.interfaces import EagleModelMixin
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    _all_dump_path = Path("/tmp/qwen38-vllm-all-layers-20260817-3.pt")
    _latest_layer_states: list[torch.Tensor] = []
    _original_maybe_add_hidden_state = EagleModelMixin._maybe_add_hidden_state
    _original_propose = DSparkSpeculator.propose
    _signature = inspect.signature(_original_propose)

    def _capture_hidden_state(self, aux_hidden_states, layer_idx, hidden_states, residual):
        global _latest_layer_states
        if layer_idx == 0:
            _latest_layer_states = []
        else:
            value = hidden_states + residual if residual is not None else hidden_states
            row_cap = max(
                1, int(os.environ.get("QSR_DSPARK_DUMP_ALL_LAYERS_ROWS", "8"))
            )
            _latest_layer_states.append(value[..., :row_cap, :].detach().cpu())
            if len(_latest_layer_states) == 64:
                Path("/tmp/qwen38-vllm-layer-capture-seen").write_text(
                    f"pid={os.getpid()} shape={tuple(value.shape)}\n"
                )
        return _original_maybe_add_hidden_state(
            self, aux_hidden_states, layer_idx, hidden_states, residual
        )

    def _dump_layers_before_propose(self, *args, **kwargs):
        bound = _signature.bind(self, *args, **kwargs)
        layer_max = max(
            (float(t.float().abs().max()) for t in _latest_layer_states),
            default=0.0,
        )
        Path("/tmp/qwen38-vllm-propose-seen").write_text(
            f"pid={os.getpid()} dummy={bound.arguments.get('dummy_run', False)} "
            f"layers={len(_latest_layer_states)} max={layer_max}\n"
        )
        if (
            not bound.arguments.get("dummy_run", False)
            and not _all_dump_path.exists()
            and len(_latest_layer_states) >= 64
            and layer_max > 1e-6
        ):
            torch.save(_latest_layer_states[:64], _all_dump_path)
        return _original_propose(self, *args, **kwargs)

    EagleModelMixin._maybe_add_hidden_state = _capture_hidden_state
    DSparkSpeculator.propose = _dump_layers_before_propose
