"""Importing ``server`` must make INFO from every runtime logger reachable.

Confirmed broken 2026-08-02 by reading two real server log files line by line:
neither contained a single ``qwen_sm120_server.engine`` record. uvicorn's
``dictConfig`` configures only its own loggers and leaves the root alone, so a
child logger with no handler propagated to a root that had none either and the
record was discarded.

``server/app.py`` had already hit this and worked around it for its own logger,
with a comment saying exactly why. The workaround never generalised, and what
it failed to cover was the line that matters most operationally: "CUDA Graph
captured at load" is how you find out whether capture succeeded, and it was
being written to nowhere. B2 spent a session unable to confirm CUDA Graph
engagement in a running server for precisely this reason.

The two backends log under different roots -- ``qwen_sm120_runtime.*`` and
``runtime.*`` -- which is why this checks a set rather than one name.
"""

from __future__ import annotations

import io
import logging

import pytest

LOGGER_NAMES = [
    "qwen_sm120_server.engine",
    "qwen_sm120_server.app",
    "qwen_sm120_runtime.laguna_backend",
    "runtime.backends.qwen36",
]


@pytest.mark.parametrize("logger_name", LOGGER_NAMES)
def test_info_reaches_a_handler(logger_name):
    """An INFO record emitted after importing ``server`` must land somewhere.

    Asserting on delivery rather than on configuration: a test that checked
    "does this logger have a handler" would pass on a logger whose parent
    swallows the record, and would fail on one correctly inheriting a working
    ancestor. What matters is whether the record arrives.
    """
    import server  # noqa: F401  -- the import is what installs the handlers

    captured = io.StringIO()
    probe = logging.StreamHandler(captured)

    logger = logging.getLogger(logger_name)
    # Attach to the configured ancestor so this exercises real propagation
    # instead of bypassing it.
    root_name = logger_name.split(".")[0]
    ancestor = logging.getLogger(root_name)
    ancestor.addHandler(probe)
    try:
        logger.info("probe record from %s", logger_name)
    finally:
        ancestor.removeHandler(probe)

    assert "probe record" in captured.getvalue(), (
        f"{logger_name} emitted INFO and nothing received it -- this is how "
        '"CUDA Graph captured at load" went missing from the service log'
    )


def test_info_level_is_actually_enabled():
    """A handler is useless if the level filters the record out first."""
    import server  # noqa: F401

    for logger_name in LOGGER_NAMES:
        logger = logging.getLogger(logger_name)
        assert logger.isEnabledFor(logging.INFO), (
            f"{logger_name} is not enabled for INFO, so operational lines are "
            "dropped before any handler sees them"
        )
