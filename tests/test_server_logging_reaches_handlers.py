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


def _handlers_that_would_fire(logger: logging.Logger) -> list[logging.Handler]:
    """Every handler ``callHandlers`` would reach for a record on ``logger``.

    Replicates CPython's own walk: collect the logger's handlers, then follow
    ``parent`` while ``propagate`` holds. Written out rather than probed with a
    temporary handler because ``server.app`` deliberately sets
    ``propagate = False`` -- a probe attached to its ancestor never sees its
    records, and a probe attached to the logger itself would trivially receive
    them and prove nothing. This asks the question that matters: is there a
    handler on the path a real record would take?
    """
    found: list[logging.Handler] = []
    current: logging.Logger | None = logger
    while current:
        found.extend(current.handlers)
        if not current.propagate:
            break
        current = current.parent
    return found


@pytest.mark.parametrize("logger_name", LOGGER_NAMES)
def test_info_reaches_a_handler(logger_name):
    """A record emitted here must have somewhere to land."""
    import server  # noqa: F401  -- the import is what installs the handlers

    logger = logging.getLogger(logger_name)
    handlers = _handlers_that_would_fire(logger)
    assert handlers, (
        f"{logger_name} emitted INFO would reach no handler at all -- this is "
        'how "CUDA Graph captured at load" went missing from the service log'
    )


def test_a_record_is_actually_formatted_end_to_end(caplog):
    """One end-to-end check that the walk above is not merely structural."""
    import server  # noqa: F401

    stream = io.StringIO()
    probe = logging.StreamHandler(stream)
    engine_logger = logging.getLogger("qwen_sm120_server.engine")
    ancestor = logging.getLogger("qwen_sm120_server")
    ancestor.addHandler(probe)
    try:
        engine_logger.info("CUDA Graph captured at load")
    finally:
        ancestor.removeHandler(probe)
    assert "CUDA Graph captured at load" in stream.getvalue()


def test_info_level_is_actually_enabled():
    """A handler is useless if the level filters the record out first."""
    import server  # noqa: F401

    for logger_name in LOGGER_NAMES:
        logger = logging.getLogger(logger_name)
        assert logger.isEnabledFor(logging.INFO), (
            f"{logger_name} is not enabled for INFO, so operational lines are "
            "dropped before any handler sees them"
        )
