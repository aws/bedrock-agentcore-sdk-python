"""Tests for iter_events, which reads streamed responses as the bytes arrive."""

from bedrock_agentcore.runtime.streaming import iter_events


class Raw:
    """Hands back one chunk per read, the way a socket does when data trickles in."""

    def __init__(self, chunks):
        self.chunks, self.reads = list(chunks), 0

    def read1(self, _size):
        self.reads += 1
        return self.chunks.pop(0) if self.chunks else b""


class Body:
    def __init__(self, chunks):
        self._raw_stream = Raw(chunks)
        self.closed = False

    def close(self):
        self.closed = True


def test_each_event_is_yielded_before_the_next_chunk_is_read():
    body = Body([b'data: {"step": "one"}\n\n', b'data: {"step": "two"}\n\n'])
    events = iter_events({"response": body})
    assert next(events) == {"step": "one"}
    assert body._raw_stream.reads == 1
    assert next(events) == {"step": "two"}


def test_an_event_split_across_chunks_is_joined():
    body = Body([b'data: {"st', b'ep": "one"}\n', b"\n"])
    assert list(iter_events(body)) == [{"step": "one"}]


def test_non_json_data_comes_back_as_text_and_crlf_is_accepted():
    body = Body([b"data: hello\r\n\r\n", b": comment\r\n\r\n"])
    assert list(iter_events(body)) == ["hello"]


def test_a_final_event_without_a_blank_line_is_kept_and_the_body_closed():
    body = Body([b'data: {"done": true}'])
    assert list(iter_events(body)) == [{"done": True}]
    assert body.closed


def test_a_multibyte_character_split_across_reads_is_decoded():
    raw = 'data: {"reply": "☕"}\n\n'.encode("utf-8")
    cut = raw.index("☕".encode("utf-8")) + 2
    assert list(iter_events(Body([raw[:cut], raw[cut:]]))) == [{"reply": "☕"}]
