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
"""

collect_ignore = [
    "test_real_world.py",
    "test_api_compat.py",
    "test_e2e_256k_longctx.py",
]

collect_ignore_glob = [
    "debug/*.py",
]
