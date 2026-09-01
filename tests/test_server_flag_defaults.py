"""Server feature-flag defaults must match what the CLI and the launcher promise.

These flags are read once at module import from the environment, so a wrong
default is invisible in production whenever the deployed launcher happens to
pin the variable -- which is exactly how the prefix cache spent an unknown
stretch defaulting OFF while its own comment, its `--no-prefix-cache` opt-out,
and `scripts/blackwellm_ctl.sh` all said ON.

The fossil: `8f27f59` collapsed `"0" if _IS_LAGUNA else "1"` down to `"0"`
when the retired Qwen branches came out, preserving Laguna's then-experimental
default. The prefix cache later became the product, the comment was updated to
say so, and the literal never followed. `blackwellm_ctl.sh` pinning
`QSR_SERVER_ENABLE_PREFIX_CACHE:=1` kept production correct and kept the bug
invisible; `python -m server.app` with no flags ran without a prefix cache, and
`--no-prefix-cache` set "0" over a value that was already falsy.

An opt-out flag is the tell. `--no-prefix-cache` is only coherent against a
default of ON, so the flag's existence is itself a specification.
"""

from __future__ import annotations

import importlib
import os
import pathlib

import pytest

pytest.importorskip("fastapi")


def test_gpu_process_lock_rejects_second_owner(tmp_path, monkeypatch):
    import server.app as app_mod

    lock_path = tmp_path / "gpu.lock"
    monkeypatch.setenv("QSR_GPU_LOCK_PATH", str(lock_path))
    first_fd = app_mod._acquire_gpu_process_lock()
    try:
        with pytest.raises(RuntimeError, match="already owns the GPU lock"):
            app_mod._acquire_gpu_process_lock()
    finally:
        import fcntl

        fcntl.flock(first_fd, fcntl.LOCK_UN)
        os.close(first_fd)


def _reimport_app_with(monkeypatch, **env: str | None):
    """Re-import server.app under a patched environment and hand back the module.

    The flags are module-level constants evaluated at import time, so the only
    way to observe a default is to import with the variable absent.
    """
    import server.app as app_mod

    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    return importlib.reload(app_mod)


class TestPrefixCacheDefault:
    def test_prefix_cache_default_is_on(self, monkeypatch):
        """No env var, no flags => the prefix cache is enabled."""
        app_mod = _reimport_app_with(monkeypatch, QSR_SERVER_ENABLE_PREFIX_CACHE=None)
        assert app_mod.SERVER_ENABLE_PREFIX_CACHE is True, (
            "prefix cache defaulted OFF -- `python -m server.app` would run "
            "without it while --no-prefix-cache and the launcher both assume ON"
        )

    def test_explicit_zero_still_turns_it_off(self, monkeypatch):
        """The documented rollback path must keep working.

        Flipping the default is only safe if the opt-out is genuinely reachable;
        this is the half of the contract that `--no-prefix-cache` depends on.
        """
        app_mod = _reimport_app_with(monkeypatch, QSR_SERVER_ENABLE_PREFIX_CACHE="0")
        assert app_mod.SERVER_ENABLE_PREFIX_CACHE is False


class TestQwenDSparkDefault:
    @pytest.mark.parametrize(
        "model_path",
        [
            "unsloth/Qwen3.6-27B-NVFP4",
            "unsloth/Qwen3.8-27B-NVFP4",
        ],
    )
    def test_qwen_model_selects_measured_dspark_profile(self, monkeypatch, model_path):
        app_mod = _reimport_app_with(
            monkeypatch,
            QSR_SERVER_MODEL_PATH=model_path,
            QSR_SERVER_CAPACITY=None,
            QSR_SERVER_NUM_SLOTS=None,
            QSR_SERVER_BLOCK_SIZE=None,
            QSR_SERVER_BLOCKS_PER_SLOT=None,
            QSR_SERVER_KV_CACHE_DTYPE=None,
            QSR_SERVER_ENABLE_CUDAGRAPH=None,
            QSR_SERVER_ENABLE_DSPARK=None,
            QSR_SERVER_ENABLE_MTP=None,
            QSR_QWEN_KV_MODE=None,
            QSR_QWEN_KV_POOL_BYTES=None,
            QSR_QWEN36_DSPARK_VERIFY_MODE=None,
            QSR_QWEN36_DSPARK_REQUIRE_CG=None,
        )

        assert app_mod.SERVER_ENABLE_DSPARK is True
        assert app_mod.SERVER_ENABLE_MTP is False
        assert app_mod.SERVER_CAPACITY == 4
        assert app_mod.SERVER_NUM_SLOTS == 4
        assert app_mod.SERVER_BLOCK_SIZE == 128
        assert app_mod.SERVER_BLOCKS_PER_SLOT == 2048
        assert app_mod.SERVER_KV_CACHE_DTYPE == "fp8_e4m3"
        assert app_mod.SERVER_QWEN_KV_MODE == "elastic"
        assert app_mod.SERVER_QWEN_KV_POOL_BYTES == 19_629_342_720
        assert app_mod.SERVER_DSPARK_K == 7
        assert app_mod.SERVER_DSPARK_VERIFY_MODE == "compact"
        assert app_mod.SERVER_DSPARK_REQUIRE_CG is True

    def test_explicit_zero_keeps_qwen_native_path_available(self, monkeypatch):
        app_mod = _reimport_app_with(
            monkeypatch,
            QSR_SERVER_MODEL_PATH="unsloth/Qwen3.8-27B-NVFP4",
            QSR_SERVER_ENABLE_DSPARK="0",
        )
        assert app_mod.SERVER_ENABLE_DSPARK is False

    def test_qwen_backend_hint_selects_profile_for_local_snapshot(self, monkeypatch):
        app_mod = _reimport_app_with(
            monkeypatch,
            QSR_SERVER_MODEL_PATH="/models/private-qwen-checkpoint",
            QSR_SERVER_BACKEND="qwen36",
            QSR_SERVER_ENABLE_DSPARK=None,
            QSR_SERVER_ENABLE_MTP=None,
            QSR_SERVER_CAPACITY=None,
            QSR_SERVER_NUM_SLOTS=None,
            QSR_SERVER_BLOCK_SIZE=None,
            QSR_SERVER_KV_CACHE_DTYPE=None,
            QSR_QWEN_KV_MODE=None,
            QSR_QWEN_KV_POOL_BYTES=None,
        )
        assert app_mod.SERVER_ENABLE_DSPARK is True
        assert app_mod.SERVER_ENABLE_MTP is False
        assert app_mod.SERVER_BLOCK_SIZE == 128
        assert app_mod.SERVER_QWEN_KV_MODE == "elastic"

    def test_qwen_gguf_defaults_to_fla_gdn_prefill(self, monkeypatch):
        app_mod = _reimport_app_with(
            monkeypatch,
            QSR_SERVER_MODEL_PATH="/models/Qwen3.8-27B-UD-Q6_K_XL.gguf",
            QSR_SERVER_BACKEND="qwen36",
            QSR_SERVER_ENABLE_DFLASH2="1",
            QSR_SERVER_ENABLE_DSPARK="0",
            QSR_SERVER_ENABLE_MTP="0",
            QSR_SERVER_ENABLE_CUDAGRAPH="1",
            QSR_QWEN36_GDN_PREFILL_BACKEND=None,
            QSR_GGUF_DEQUANTIZE_WEIGHTS=None,
            QSR_GGUF_NATIVE_PREFILL_DEQUANT=None,
        )

        app_mod._apply_qwen_dspark_runtime_defaults()
        assert os.environ["QSR_QWEN36_GDN_PREFILL_BACKEND"] == "fla"
        assert os.environ["QSR_GGUF_DEQUANTIZE_WEIGHTS"] == "0"
        assert os.environ["QSR_GGUF_NATIVE_PREFILL_DEQUANT"] == "1"

    def test_qwen_gguf_resident_bf16_opt_out_is_preserved(self, monkeypatch):
        app_mod = _reimport_app_with(
            monkeypatch,
            QSR_SERVER_MODEL_PATH="/models/Qwen3.8-27B-UD-Q6_K_XL.gguf",
            QSR_SERVER_BACKEND="qwen36",
            QSR_SERVER_ENABLE_DFLASH2="1",
            QSR_SERVER_ENABLE_DSPARK="0",
            QSR_GGUF_DEQUANTIZE_WEIGHTS="0",
            QSR_GGUF_NATIVE_PREFILL_DEQUANT=None,
        )

        app_mod._apply_qwen_dspark_runtime_defaults()
        assert os.environ["QSR_GGUF_DEQUANTIZE_WEIGHTS"] == "0"
        assert os.environ["QSR_GGUF_NATIVE_PREFILL_DEQUANT"] == "1"

    def test_qwen_gguf_transient_prefill_opt_out_is_preserved(self, monkeypatch):
        app_mod = _reimport_app_with(
            monkeypatch,
            QSR_SERVER_MODEL_PATH="/models/Qwen3.8-27B-UD-Q6_K_XL.gguf",
            QSR_SERVER_BACKEND="qwen36",
            QSR_SERVER_ENABLE_DFLASH2="1",
            QSR_SERVER_ENABLE_DSPARK="0",
            QSR_GGUF_NATIVE_PREFILL_DEQUANT="0",
        )

        app_mod._apply_qwen_dspark_runtime_defaults()
        assert os.environ["QSR_GGUF_NATIVE_PREFILL_DEQUANT"] == "0"

    def test_qwen_gguf_target_only_defaults_to_bf16_graph(self, monkeypatch):
        app_mod = _reimport_app_with(
            monkeypatch,
            QSR_SERVER_MODEL_PATH="/models/Qwen3.8-27B-UD-Q6_K_XL.gguf",
            QSR_SERVER_BACKEND="qwen36",
            QSR_SERVER_ENABLE_DFLASH2="0",
            QSR_SERVER_ENABLE_DSPARK="0",
            QSR_SERVER_ENABLE_MTP="0",
            QSR_GGUF_COMPUTE_DTYPE=None,
        )

        app_mod._apply_qwen_dspark_runtime_defaults()
        assert os.environ["QSR_GGUF_COMPUTE_DTYPE"] == "bf16"

    def test_qwen_gguf_selects_qwen3_coder_tool_parser(self, monkeypatch):
        app_mod = _reimport_app_with(
            monkeypatch,
            QSR_SERVER_MODEL_PATH="/models/Qwen3.8-27B-UD-Q6_K_XL.gguf",
            QSR_SERVER_BACKEND="qwen36",
            QSR_TOOL_CALL_PARSER=None,
        )

        assert app_mod.SERVER_TOOL_CALL_PARSER == "qwen3_coder"

    def test_flashnext_nvfp4_selects_qwen3_coder_tool_parser(self, monkeypatch):
        app_mod = _reimport_app_with(
            monkeypatch,
            QSR_SERVER_MODEL_PATH="/models/Qwen3.8-Flash-Next-NVFP4-RadixArk",
            QSR_SERVER_BACKEND=None,
            QSR_TOOL_CALL_PARSER=None,
        )

        assert app_mod.SERVER_TOOL_CALL_PARSER == "qwen3_coder"

    def test_laguna_keeps_poolside_tool_parser_default(self, monkeypatch):
        app_mod = _reimport_app_with(
            monkeypatch,
            QSR_SERVER_MODEL_PATH="poolside/Laguna-S-2.1-NVFP4",
            QSR_TOOL_CALL_PARSER=None,
        )

        assert app_mod.SERVER_TOOL_CALL_PARSER == "poolside_v1"


class TestFlagShapeMatchesDefault:
    """A CLI flag's *shape* declares what the default must be.

    `--no-x` is only meaningful if x defaults ON; `--x` is only meaningful if
    x defaults OFF. That is the invariant the prefix cache violated, and it is
    checkable without reference to any deployment.

    Deliberately *not* checked here: whether `blackwellm_ctl.sh`'s pins agree
    with the module defaults. A first draft of this test did exactly that and
    flagged `QSR_SERVER_ENABLE_DFLASH` -- whose launcher pin of 1 sits against
    a module default of 0 and is perfectly correct, because `--dflash` is an
    opt-in. A launcher pinning a value the module does not default to is a
    deployment choice, not evidence of a bug. Comparing against the flag shape
    separates the two; comparing against the launcher conflates them.
    """

    # (env var, argparse flag). The flag's `--no-` prefix is the specification.
    _FLAGS = [
        ("QSR_SERVER_ENABLE_PREFIX_CACHE", "--no-prefix-cache"),
        ("QSR_SERVER_ENABLE_CUDAGRAPH", "--no-cudagraph"),
        ("QSR_SERVER_ENABLE_DFLASH", "--dflash"),
    ]

    @pytest.mark.parametrize("env_var,flag", _FLAGS)
    def test_flag_shape_implies_default(self, monkeypatch, env_var, flag):
        app_mod = _reimport_app_with(monkeypatch, **{env_var: None})

        source = pathlib.Path(app_mod.__file__).read_text(encoding="utf-8")
        assert f'"{flag}"' in source, f"{flag} no longer exists -- update this test's table"

        expected = flag.startswith("--no-")
        constant = env_var.replace("QSR_", "")
        actual = getattr(app_mod, constant)
        assert actual is expected, (
            f"{flag} is an {'opt-out' if expected else 'opt-in'} flag, so "
            f"{constant} must default to {expected}, but it defaults to {actual}. "
            "An opt-out against a default of OFF is a no-op that reads as a "
            "working rollback switch."
        )
