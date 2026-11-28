"""Incremental parsing of server-sent events (SSE).

Bytes arrive in network reads that can end anywhere, including inside a multibyte
UTF-8 character or between the two characters of a CRLF. ``SSEParser`` keeps the
partial state between reads, so each event comes out whole.
"""

import codecs
import re
from typing import Iterable, Iterator, List, Union

# SSE lines end with CRLF, LF, or CR, and nothing else counts as a line break.
_LINE = re.compile(r"([^\r\n]*)(?:\r\n|\r|\n)")


class SSEParser:
    """Turn a stream of bytes or text into the ``data`` of each server-sent event.

    Consecutive ``data:`` lines of one event are joined with a newline, as the SSE
    specification requires. Other fields (``event:``, ``id:``, ``retry:``) and comment
    lines are ignored. A blank line ends an event; an event with no ``data`` line is
    skipped.

    Example:
        parser = SSEParser()
        for chunk in chunks:
            for data in parser.feed(chunk):
                handle(data)
        for data in parser.close():
            handle(data)
    """

    def __init__(self) -> None:
        """Start with no buffered input."""
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._pending = ""
        self._data: List[str] = []

    def feed(self, chunk: Union[bytes, str]) -> List[str]:
        """Add one chunk and return the data of every event it completed."""
        text = self._decoder.decode(chunk) if isinstance(chunk, bytes) else chunk
        return self._consume(self._pending + text, final=False)

    def close(self) -> List[str]:
        """Finish the stream and return the data of a last event without a blank line."""
        text = self._pending + self._decoder.decode(b"", final=True)
        events = self._consume(text, final=True)
        if self._data:
            events.append("\n".join(self._data))
            self._data = []
        return events

    def _consume(self, text: str, final: bool) -> List[str]:
        events: List[str] = []
        if not final and text.endswith("\r"):
            # A LF may still follow in the next chunk, and CRLF is one line ending.
            cut = max(text.rfind("\n", 0, len(text) - 1), text.rfind("\r", 0, len(text) - 1)) + 1
            text, self._pending = text[:cut], text[cut:]
        else:
            self._pending = ""
        lines: List[str] = []
        end = 0
        for match in _LINE.finditer(text):
            lines.append(match.group(1))
            end = match.end()
        if end < len(text):
            if final:
                lines.append(text[end:])
            else:
                self._pending = text[end:] + self._pending
        for line in lines:
            if not line:
                if self._data:
                    events.append("\n".join(self._data))
                    self._data = []
            elif line.startswith("data:"):
                value = line[len("data:") :]
                self._data.append(value[1:] if value.startswith(" ") else value)
        return events


def iter_sse_data(chunks: Iterable[Union[bytes, str]]) -> Iterator[str]:
    """Yield the data of each server-sent event in ``chunks``, as each one completes."""
    parser = SSEParser()
    for chunk in chunks:
        yield from parser.feed(chunk)
    yield from parser.close()
