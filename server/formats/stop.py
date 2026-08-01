"""Text-level stop-sequence matching (N2, docs/roadmap.md Track E).

Stop sequences are user-supplied strings; decoding is per-token. A single
sequence can span a token boundary (arrive split across two or more
tokens) or land entirely inside one token alongside other text. Matching
therefore always happens on accumulated *decoded content text*, never on
raw token ids.

This generalizes the same ambiguous-tail-withholding idea
``server/formats/stream.py::_trim_ambiguous_tail`` already uses for
``<think>``/``<usage>`` markers (one marker) to N candidate stop sequences:
a trailing strict prefix of ANY configured sequence must stay unflushed,
because more tokens could still arrive and complete a match.

Pure string functions, no tokenizer/torch dependency -- used directly by
``server/engine.py``'s per-token decode-loop bookkeeping.
"""

from __future__ import annotations


def find_earliest_stop_match(text: str, stop_sequences: list[str]) -> tuple[int, str] | None:
    """Return ``(index, matched_sequence)`` for the stop sequence that
    occurs earliest in ``text``, or ``None`` if none of them occur at all.

    Ties (two sequences starting at the same index) are broken by input
    order -- the first configured sequence wins.
    """
    best: tuple[int, str] | None = None
    for seq in stop_sequences:
        if not seq:
            continue
        idx = text.find(seq)
        if idx < 0:
            continue
        if best is None or idx < best[0]:
            best = (idx, seq)
    return best


def trim_ambiguous_stop_tail(text: str, stop_sequences: list[str]) -> str:
    """Drop the longest trailing strict prefix of any configured stop
    sequence from ``text``.

    Returns ``text`` unchanged when no suffix of it could still grow into
    a stop-sequence match (i.e. it is safe to treat everything in ``text``
    as confirmed, non-ambiguous content).
    """
    cut = len(text)
    for seq in stop_sequences:
        if not seq or len(seq) <= 1:
            continue
        for plen in range(min(len(seq) - 1, len(text)), 0, -1):
            if text.endswith(seq[:plen]):
                cut = min(cut, len(text) - plen)
                break
    return text[:cut]
