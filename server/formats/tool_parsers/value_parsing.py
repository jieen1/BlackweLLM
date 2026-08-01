"""Shape-agnostic argument-value decoding, shared by every ``ToolCallParser``.

Every model's chat template JSON-encodes non-string argument values and
emits string values raw (see e.g. Laguna's ``chat_template.jinja``:
``v | tojson(ensure_ascii=False) if v is not string else v``) -- decoding
one argument value back is the same problem for every shape, so it lives
here once rather than being reimplemented per parser.
"""

from __future__ import annotations

import json
import re
from typing import Any


def repair_json(value: str) -> str:
    """Attempt to repair common JSON formatting errors from model output.

    Models occasionally produce near-valid JSON with predictable mutations:
    - ``{("key": ...)}`` instead of ``[{"key": ...}]`` (set-literal confusion)
    - Trailing commas before ``]`` or ``}``
    """
    repaired = value.strip()
    # Pattern: {("key": val, ...)}] -> [{"key": val, ...}]
    # The model sometimes wraps a dict in set-literal syntax {( ... )}
    # instead of putting it in an array [{ ... }].
    if repaired.startswith("{("):
        inner = repaired[2:]  # strip leading {(
        if inner.endswith("})]"):
            inner = inner[:-3] + "}]"
        elif inner.endswith(")}"):
            inner = inner[:-2] + "}]"
        elif inner.endswith(")"):
            inner = inner[:-1] + "}]"
        repaired = "[{" + inner
    # Trailing commas: ,] or ,}
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def parse_value(raw: str) -> Any:
    """Parse one argument value: JSON if possible, else the repaired JSON,
    else the raw string verbatim (models occasionally emit bare strings
    the template doesn't quote)."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        try:
            return json.loads(repair_json(raw))
        except (json.JSONDecodeError, ValueError):
            return raw
