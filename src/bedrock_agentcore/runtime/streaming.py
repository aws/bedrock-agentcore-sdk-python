"""Read a streamed InvokeAgentRuntime response one event at a time, as the events arrive.

An agent that answers with ``text/event-stream`` sends each event as soon as it has it.
``StreamingBody.iter_lines()`` reads in fixed size blocks and waits for each block to
fill, so short events arrive together at the end. ``iter_events`` reads whatever bytes
are available instead.

Example:
    response = client.invoke_agent_runtime(...)
    for event in iter_events(response):
        print(event)
"""

import json
from typing import Any, Iterator, Mapping, Union

from .._utils.sse import SSEParser

_READ_SIZE = 64 * 1024


def iter_events(response: Union[Mapping[str, Any], Any]) -> Iterator[Any]:
    """Yield the ``data`` of each server-sent event in a streamed response.

    Args:
        response: An InvokeAgentRuntime response, or its ``response`` body.

    Yields:
        Each event's data, parsed as JSON when it is JSON and as a string otherwise.
    """
    body = response["response"] if isinstance(response, Mapping) else response
    raw = getattr(body, "_raw_stream", body)
    read = getattr(raw, "read1", None) or (lambda size: raw.read(size))
    parser = SSEParser()
    try:
        while True:
            chunk = read(_READ_SIZE)
            if not chunk:
                break
            for data in parser.feed(chunk):
                yield _parse(data)
        for data in parser.close():
            yield _parse(data)
    finally:
        close = getattr(body, "close", None)
        if close:
            close()


def _parse(data: str) -> Any:
    try:
        return json.loads(data)
    except ValueError:
        return data
