"""Tests for the incremental server-sent events parser."""

from bedrock_agentcore._utils.sse import SSEParser, iter_sse_data


def _parse(chunks):
    return list(iter_sse_data(chunks))


def test_a_multibyte_character_split_across_reads_is_decoded():
    data = "data: café ☕\n\n".encode("utf-8")
    split = data.index("☕".encode("utf-8")) + 1  # inside the three-byte character
    assert _parse([data[:split], data[split:]]) == ["café ☕"]


def test_crlf_split_across_reads_is_one_line_ending():
    assert _parse([b"data: one\r", b"\n\r\n", b"data: two\r\n\r\n"]) == ["one", "two"]


def test_a_lone_cr_ends_a_line():
    assert _parse([b"data: one\r\rdata: two\r\r"]) == ["one", "two"]


def test_multiple_data_lines_are_joined_with_a_newline():
    assert _parse([b"data: first\ndata: second\n\n"]) == ["first\nsecond"]


def test_only_one_leading_space_is_removed():
    assert _parse([b"data:  indented\n\n", b"data:tight\n\n"]) == [" indented", "tight"]


def test_other_fields_comments_and_empty_events_are_ignored():
    assert _parse([b": comment\nevent: update\nid: 7\nretry: 10\n\n", b"\n\n", b"data: x\n\n"]) == ["x"]


def test_unicode_line_separators_are_part_of_the_data():
    assert _parse(["data: a b\x0cc\n\n"]) == ["a b\x0cc"]


def test_a_last_event_without_a_blank_line_comes_out_on_close():
    parser = SSEParser()
    assert parser.feed(b"data: done") == []
    assert parser.close() == ["done"]


def test_text_input_is_accepted():
    assert _parse(['data: {"a": 1}\n', "\n"]) == ['{"a": 1}']
