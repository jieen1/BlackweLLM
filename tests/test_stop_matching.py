"""N2: tests for server/formats/stop.py -- pure text-level stop-sequence
matching, independent of tokenizer/torch. See tests/test_engine_stop_sequences.py
for the token-boundary-spanning cases exercised against the real engine
per-token commit loop."""

from __future__ import annotations

from server.formats.stop import find_earliest_stop_match, trim_ambiguous_stop_tail


class TestFindEarliestStopMatch:
    def test_no_sequences_no_match(self):
        assert find_earliest_stop_match("hello world", []) is None

    def test_no_occurrence(self):
        assert find_earliest_stop_match("hello world", ["STOP"]) is None

    def test_simple_match(self):
        assert find_earliest_stop_match("hello STOP world", ["STOP"]) == (6, "STOP")

    def test_earliest_of_multiple_candidates_wins(self):
        text = "aaa BB aaa AA aaa"
        # "AA" occurs later than "BB"
        assert find_earliest_stop_match(text, ["AA", "BB"]) == (4, "BB")

    def test_tie_breaks_by_input_order(self):
        text = "XY"
        assert find_earliest_stop_match(text, ["XY", "X"]) == (0, "XY")
        assert find_earliest_stop_match(text, ["X", "XY"]) == (0, "X")

    def test_empty_sequence_ignored(self):
        assert find_earliest_stop_match("anything", [""]) is None

    def test_match_at_very_start(self):
        assert find_earliest_stop_match("STOPeverything", ["STOP"]) == (0, "STOP")

    def test_match_at_very_end(self):
        assert find_earliest_stop_match("everythingSTOP", ["STOP"]) == (10, "STOP")


class TestTrimAmbiguousStopTail:
    def test_no_sequences_no_trim(self):
        assert trim_ambiguous_stop_tail("hello", []) == "hello"

    def test_unrelated_text_no_trim(self):
        assert trim_ambiguous_stop_tail("hello world", ["STOP"]) == "hello world"

    def test_exact_full_ambiguous_prefix_trimmed(self):
        # "ST" is a strict prefix of "STOP" -- must be withheld entirely.
        assert trim_ambiguous_stop_tail("hello ST", ["STOP"]) == "hello "

    def test_single_char_ambiguous_suffix_trimmed(self):
        assert trim_ambiguous_stop_tail("hello S", ["STOP"]) == "hello "

    def test_longest_ambiguous_suffix_wins_across_candidates(self):
        # "STO" (3 chars) is a longer ambiguous prefix-match than "X" (1 char)
        # for a text ending in "...STO"; must trim back to before "STO".
        text = "before STO"
        assert trim_ambiguous_stop_tail(text, ["STOP", "X"]) == "before "

    def test_full_sequence_present_is_not_a_prefix_trim_case(self):
        # trim_ambiguous_stop_tail only handles STRICT prefixes (< full
        # length); a complete match is find_earliest_stop_match's job, and
        # trimming should not remove a fully-formed occurrence.
        assert trim_ambiguous_stop_tail("say STOP now", ["STOP"]) == "say STOP now"

    def test_single_char_stop_sequence_never_ambiguous(self):
        # A length-1 stop sequence has no strict, non-empty prefix shorter
        # than itself -- nothing to withhold; a full match is immediate.
        assert trim_ambiguous_stop_tail("hello", ["X"]) == "hello"

    def test_empty_text(self):
        assert trim_ambiguous_stop_tail("", ["STOP"]) == ""

    def test_fully_safe_text_returned_unchanged_object_semantics(self):
        text = "completely safe text"
        assert trim_ambiguous_stop_tail(text, ["STOP", "END"]) == text
