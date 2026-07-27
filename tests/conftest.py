"""Pytest configuration.

test_real_world.py, test_api_compat.py, and test_e2e_256k_longctx.py are
integration/E2E test scripts that require a running server. They are excluded
from pytest collection and should be run manually:

    python tests/test_real_world.py [base_url]
    python tests/test_api_compat.py --base-url http://127.0.0.1:8000
    python tests/test_e2e_256k_longctx.py --base-url http://127.0.0.1:8000

debug/ holds standalone GPU debugging scripts with hardcoded local paths and
model checkpoints (not repeatable automated tests). They run top-level code
at import time with no importorskip guard, so pytest collection fails
whenever torch/CUDA/the model checkpoint isn't available on the machine —
run them manually instead, e.g.:

    python tests/debug/test_attn_correctness.py

The ``requires_hf_snapshot`` marker below covers the other machine-specific
dependency: bfdiag.shapes deliberately reads shapes out of the *real*
checkpoint's config.json rather than hardcoding them, so its acceptance
tests only mean anything where that checkpoint is downloaded. On a bare CI
runner they skip instead of failing; the same modules' synthetic-fixture
tests are unmarked and keep running everywhere.
"""

import pytest

collect_ignore = [
    "test_real_world.py",
    "test_api_compat.py",
    "test_e2e_256k_longctx.py",
]

collect_ignore_glob = [
    "debug/*.py",
]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_hf_snapshot(model_id): skip unless that model's config.json is "
        "present in the local HuggingFace cache",
    )


def _snapshot_available(model_id: str) -> bool:
    from bfdiag.shapes.model import LagunaConfigError, find_snapshot_dir

    try:
        find_snapshot_dir(model_id)
    except LagunaConfigError:
        return False
    return True


def pytest_collection_modifyitems(config, items):
    available: dict[str, bool] = {}
    for item in items:
        for mark in item.iter_markers("requires_hf_snapshot"):
            model_id = mark.args[0]
            if model_id not in available:
                available[model_id] = _snapshot_available(model_id)
            if not available[model_id]:
                item.add_marker(
                    pytest.mark.skip(reason=f"no local HuggingFace snapshot for {model_id}")
                )
