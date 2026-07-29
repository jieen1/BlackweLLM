"""Shared ``maybe_prefix`` helper -- see laguna_model.py's docstring for why
this is self-written rather than imported from vLLM, and why exact-match
correctness here matters beyond cosmetics (NVFP4 ``is_layer_skipped``
matches checkpoint ``ignored_layers`` strings exactly against these).

Split into its own module (not defined in laguna_model.py, despite that
being where it originated) so laguna_decoder.py can import it too without
a circular import between the two.
"""

from __future__ import annotations


def maybe_prefix(prefix: str, name: str) -> str:
    return name if not prefix else f"{prefix}.{name}"
