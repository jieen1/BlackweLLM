import inspect
import multiprocessing
import os
import runpy
import sys
from pathlib import Path

import torch

from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.model_executor.models.interfaces import EagleModelMixin


_original_propose = DSparkSpeculator.propose
_signature = inspect.signature(_original_propose)
_dump_path = Path("/tmp/qwen38-vllm-taps-20260817-2.pt")
_all_dump_path = Path("/tmp/qwen38-vllm-all-layers-20260817-2.pt")
_latest_layer_states = []
_original_maybe_add_hidden_state = EagleModelMixin._maybe_add_hidden_state


def _maybe_add_hidden_state_with_capture(self, aux_hidden_states, layer_idx, hidden_states, residual):
    global _latest_layer_states
    if os.environ.get("QSR_DSPARK_DUMP_ALL_LAYERS") not in (None, "", "0"):
        if layer_idx == 0:
            _latest_layer_states = []
        else:
            value = hidden_states + residual if residual is not None else hidden_states
            row_cap = max(1, int(os.environ.get("QSR_DSPARK_DUMP_ALL_LAYERS_ROWS", "8")))
            _latest_layer_states.append(value[..., :row_cap, :].detach().cpu())
    return _original_maybe_add_hidden_state(
        self, aux_hidden_states, layer_idx, hidden_states, residual
    )


EagleModelMixin._maybe_add_hidden_state = _maybe_add_hidden_state_with_capture


def _propose_with_tap_dump(self, *args, **kwargs):
    bound = _signature.bind(self, *args, **kwargs)
    aux_hidden_states = bound.arguments.get("aux_hidden_states")
    if (
        aux_hidden_states
        and not bound.arguments.get("dummy_run", False)
        and not _dump_path.exists()
    ):
        torch.save(
            [hidden[:8].detach().cpu() for hidden in aux_hidden_states], _dump_path
        )
    if (
        not bound.arguments.get("dummy_run", False)
        and not _all_dump_path.exists()
        and len(_latest_layer_states) >= 64
    ):
        torch.save(_latest_layer_states[:64], _all_dump_path)
    return _original_propose(self, *args, **kwargs)


DSparkSpeculator.propose = _propose_with_tap_dump


if __name__ == "__main__":
    multiprocessing.set_executable(
        "/home/bot/project/qwen-sm120-runtime/.tmp_vllm_spawn_wrapper.sh"
    )
    sys.argv = [
        "vllm.entrypoints.openai.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        "8200",
        "--model",
        "/home/bot/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/9c73e2daee1d0fd494ffbd1d8753f2174a953796",
        "--trust-remote-code",
        "--max-model-len",
        "262144",
        "--served-model-name",
        "qwen3.8",
        "--generation-config",
        "vllm",
        "--block-size",
        "32",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--no-enable-prefix-caching",
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "4",
        "--spec-method",
        "dspark",
        "--spec-model",
        "/tmp/qwen38-dspark-vllm.f1IwRL",
        "--spec-tokens",
        "7",
        "--gpu-memory-utilization",
        ".92",
    ]
    if os.environ.get("QSR_VLLM_ENFORCE_EAGER") not in (None, "", "0"):
        sys.argv.append("--enforce-eager")
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
