"""OpenAI-compatible HTTP server for the fixed-slot ``DirectModelRunner`` runtime."""

import logging

#: Every module in this package logs under ``qwen_sm120_server.*``. Configure
#: that shared parent once, here, because this module is imported before any
#: submodule and a handler is resolved at emit time rather than at logger
#: creation -- so ``server.engine``'s module-level ``getLogger`` is covered
#: regardless of import order.
#:
#: Without this, INFO records from anything other than ``server.app`` were
#: dropped outright. uvicorn's ``dictConfig`` configures only its own loggers
#: and does not touch the root, so a child with no handler propagates to a
#: root that has none either. ``server/app.py`` had already worked around it
#: for its own logger and said so in a comment; the workaround simply never
#: generalised, and the cost was that "CUDA Graph captured at load" -- the
#: one line that answers whether capture succeeded -- never reached the
#: service log. Confirmed 2026-08-02 by reading two real server log files
#: line by line: not one ``qwen_sm120_server.engine`` record in either.
#: The backends log under two *different* roots -- ``laguna.py`` uses
#: ``qwen_sm120_runtime.laguna_backend`` while ``qwen36.py`` uses ``__name__``
#: (``runtime.backends.qwen36``). Inconsistent, and left that way on purpose:
#: renaming a logger silently breaks anyone filtering on the old name, and the
#: naming is not what was broken. Both are configured here instead.
#:
#: Configuring library loggers from the server package is deliberate -- the
#: server is the application and ``runtime`` is a library, so this is where the
#: "logs must go somewhere" decision belongs.
_LOGGER_ROOTS = ("qwen_sm120_server", "qwen_sm120_runtime", "runtime")

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
for _name in _LOGGER_ROOTS:
    _package_logger = logging.getLogger(_name)
    _package_logger.setLevel(logging.INFO)
    if not _package_logger.handlers:
        _package_logger.addHandler(_handler)
