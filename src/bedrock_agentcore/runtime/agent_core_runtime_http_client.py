"""HTTP client for invoking a deployed Bedrock AgentCore runtime.

Complements :class:`AgentCoreRuntimeClient` (which only builds WebSocket URLs
and headers) by providing bearer-token HTTP invocation, SSE streaming, and the
``InvokeAgentRuntimeCommand`` API for running shell commands inside a runtime
session.

Wire formats targeted:

- ``POST /runtimes/{arn}/invocations`` — agent invocation. Response is either
  a JSON document or Server-Sent Events (``text/event-stream``).
- ``POST /runtimes/{arn}/commands`` — ``InvokeAgentRuntimeCommand``. Response
  is the AWS EventStream binary framing
  (``application/vnd.amazon.eventstream``). Each event's payload is JSON
  wrapped under a ``chunk`` key containing one of ``contentStart``,
  ``contentDelta {stdout, stderr}``, or
  ``contentStop {exitCode, status}``.
- ``POST /runtimes/{arn}/stopruntimesession`` — session termination.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import urllib.parse
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Generator, Iterator, Optional

import urllib3
from botocore.eventstream import EventStreamBuffer

if TYPE_CHECKING:
    # urllib3 v2 returns ``BaseHTTPResponse`` from ``PoolManager.request``.
    # On urllib3 v1.26 the returned object is ``HTTPResponse`` which is
    # structurally compatible with the same attributes (``status``, ``headers``,
    # ``read``, ``stream``, ``release_conn``), so the annotation is purely for
    # type-checking under v2.
    from urllib3 import BaseHTTPResponse as _UrllibResponse

logger = logging.getLogger(__name__)


class AgentRuntimeError(Exception):
    """Raised when an AgentCore runtime returns an error response.

    Used by :class:`AgentCoreRuntimeHttpClient` for both non-2xx HTTP
    responses and in-band error events embedded in SSE streams.

    Attributes:
        error: Short machine-readable error token (for example the
            runtime's ``error`` field, or the HTTP status text).
        error_type: Category label. For HTTP failures this is
            ``"HTTP <status>"`` (e.g. ``"HTTP 404"``). For runtime error
            payloads it is whatever the server set on the ``error_type``
            field.
    """

    def __init__(self, error: str, error_type: str = "", message: str = "") -> None:
        """Initialize the exception.

        Args:
            error: Short error token (see :attr:`error`).
            error_type: Category label (see :attr:`error_type`). Defaults
                to the empty string.
            message: Human-readable message used as the exception's
                string representation. Defaults to ``error`` when empty.
        """
        self.error = error
        self.error_type = error_type
        super().__init__(message or error)


class AgentCoreRuntimeHttpClient:
    """HTTP client for invoking a deployed Bedrock AgentCore runtime.

    Use this client when you need bearer-token authentication (JWT/OAuth)
    instead of IAM/SigV4. It supports blocking invocation, synchronous and
    asynchronous streaming, shell command execution via
    ``InvokeAgentRuntimeCommand``, and session termination.

    Each method takes the ``bearer_token`` per-call so the same client can
    be reused with rotating credentials.

    Attributes:
        agent_arn: Full ARN of the target agent runtime.
        region: AWS region extracted from ``agent_arn``.
        endpoint_name: Endpoint qualifier (defaults to ``"DEFAULT"``).
        timeout: Default HTTP read timeout in seconds for non-command
            methods. ``execute_command`` derives its HTTP timeout from
            ``command_timeout`` instead (see that method).
        content_type: Request ``Content-Type`` for :meth:`invoke` and
            :meth:`invoke_streaming`.
        accept: Request ``Accept`` for :meth:`invoke` and
            :meth:`invoke_streaming`.
    """

    def __init__(
        self,
        agent_arn: str,
        endpoint_name: str = "DEFAULT",
        timeout: int = 300,
        content_type: str = "application/json",
        accept: str = "application/json",
        pool_manager: Optional[urllib3.PoolManager] = None,
    ) -> None:
        """Initialize the HTTP client.

        Args:
            agent_arn: The ARN of the agent runtime to invoke. The AWS
                region is extracted from the ARN automatically.
            endpoint_name: Endpoint qualifier sent as the ``qualifier``
                query parameter. Defaults to ``"DEFAULT"``.
            timeout: Default HTTP read timeout in seconds (used by
                :meth:`invoke`, :meth:`invoke_streaming`,
                :meth:`invoke_streaming_async`, and
                :meth:`stop_runtime_session`). ``execute_command`` derives
                its HTTP timeout internally from ``command_timeout``.
            content_type: MIME type of the request payload for invocation
                calls.
            accept: Desired MIME type for invocation responses.
            pool_manager: Optional pre-configured
                :class:`urllib3.PoolManager`. Primarily useful for tests
                and for callers who want to control connection pooling.
                A fresh manager is created when not provided.

        Raises:
            ValueError: If the ARN does not contain a parseable region
                component.
        """
        parts = agent_arn.split(":")
        if len(parts) < 4 or not parts[3]:
            raise ValueError(f"Invalid agent ARN (missing region): {agent_arn}")

        self.agent_arn = agent_arn
        self.region = parts[3]
        self.endpoint_name = endpoint_name
        self.timeout = timeout
        self.content_type = content_type
        self.accept = accept
        self._http: urllib3.PoolManager = pool_manager or urllib3.PoolManager()

    # ------------------------------------------------------------------ #
    # URL, headers, body helpers
    # ------------------------------------------------------------------ #

    def _build_url(self, path_suffix: str) -> str:
        """Build a full runtime URL.

        Args:
            path_suffix: Path under ``/runtimes/{arn}/``. Must not start
                with ``/``. Examples: ``"invocations"``, ``"commands"``,
                ``"stopruntimesession"``.

        Returns:
            Absolute URL including the qualifier query string.
        """
        escaped_arn = urllib.parse.quote(self.agent_arn, safe="")
        base = f"https://bedrock-agentcore.{self.region}.amazonaws.com/runtimes/{escaped_arn}/{path_suffix}"
        query = urllib.parse.urlencode({"qualifier": self.endpoint_name})
        return f"{base}?{query}"

    def _build_headers(
        self,
        bearer_token: str,
        session_id: str,
        accept: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> dict[str, str]:
        """Build the base request headers.

        Args:
            bearer_token: OAuth/JWT bearer token.
            session_id: Runtime session id.
            accept: Optional override for the ``Accept`` header.
            content_type: Optional override for the ``Content-Type``
                header.

        Returns:
            Header dict including ``Authorization``, ``Content-Type``,
            ``Accept``, and ``X-Amzn-Bedrock-AgentCore-Runtime-Session-Id``.
        """
        return {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": content_type or self.content_type,
            "Accept": accept or self.accept,
            "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
        }

    def _serialize_body(self, body: Any) -> bytes:
        """Serialize a request body to bytes.

        JSON content types serialize via :func:`json.dumps`. For other
        content types, ``bytes`` pass through unchanged, ``str`` is
        UTF-8-encoded, and anything else falls back to JSON serialization.

        Args:
            body: The payload to send.

        Returns:
            UTF-8 encoded request body.
        """
        if "json" in self.content_type:
            return json.dumps(body).encode("utf-8")
        if isinstance(body, bytes):
            return body
        if isinstance(body, str):
            return body.encode("utf-8")
        return json.dumps(body).encode("utf-8")

    # ------------------------------------------------------------------ #
    # invoke / invoke_streaming / invoke_streaming_async
    # ------------------------------------------------------------------ #

    def invoke(
        self,
        body: Any,
        session_id: str,
        bearer_token: str,
        headers: Optional[dict[str, str]] = None,
    ) -> str:
        """Invoke the agent and return the full response body as a string.

        Handles both JSON responses and Server-Sent Events transparently.
        For SSE responses, the decoded event data is concatenated in the
        order it arrived.

        Args:
            body: The request body to send to the agent.
            session_id: Session id for conversation continuity.
            bearer_token: Bearer token for authentication.
            headers: Optional extra headers to include. Overwrite the
                defaults on key collision.

        Returns:
            The complete response body as a string. For JSON responses
            that happen to be a JSON-encoded string, the unwrapped string
            value is returned.

        Raises:
            AgentRuntimeError: If the runtime returns a non-2xx status
                or an error event in the SSE stream.
        """
        request_headers = self._build_headers(bearer_token, session_id)
        if headers:
            request_headers.update(headers)

        response = self._http.request(
            "POST",
            self._build_url("invocations"),
            headers=request_headers,
            body=self._serialize_body(body),
            timeout=self.timeout,
            preload_content=False,
        )
        try:
            self._check_response(response)
            content_type = response.headers.get("content-type", "")
            if "text/event-stream" not in content_type:
                return self._read_non_streaming(response)
            return "".join(self._iter_sse_decoded(response))
        finally:
            response.release_conn()

    def invoke_streaming(
        self,
        body: Any,
        session_id: str,
        bearer_token: str,
        headers: Optional[dict[str, str]] = None,
    ) -> Generator[str, None, None]:
        """Invoke the agent and yield SSE chunks as they arrive.

        Args:
            body: The request body to send to the agent.
            session_id: Session id for conversation continuity.
            bearer_token: Bearer token for authentication.
            headers: Optional extra headers to include.

        Yields:
            Decoded payload strings from the SSE stream, one per
            non-empty ``data:`` line.

        Raises:
            AgentRuntimeError: If the runtime returns a non-2xx status
                or an error event in the stream.
        """
        request_headers = self._build_headers(bearer_token, session_id)
        if headers:
            request_headers.update(headers)

        response = self._http.request(
            "POST",
            self._build_url("invocations"),
            headers=request_headers,
            body=self._serialize_body(body),
            timeout=self.timeout,
            preload_content=False,
        )
        try:
            self._check_response(response)
            yield from self._iter_sse_decoded(response)
        finally:
            response.release_conn()

    async def invoke_streaming_async(
        self,
        body: Any,
        session_id: str,
        bearer_token: str,
        headers: Optional[dict[str, str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Async generator version of :meth:`invoke_streaming`.

        The underlying HTTP call is blocking; this wrapper runs it on a
        background thread and delivers chunks to the caller through an
        :class:`asyncio.Queue`, so it is safe to ``async for`` over.

        Args:
            body: The request body to send to the agent.
            session_id: Session id for conversation continuity.
            bearer_token: Bearer token for authentication.
            headers: Optional extra headers to include.

        Yields:
            Decoded payload strings from the SSE stream.

        Raises:
            AgentRuntimeError: If the runtime returns a non-2xx status
                or an error event in the stream.
        """
        chunk_queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()
        loop = asyncio.get_running_loop()

        def stream_in_thread() -> None:
            try:
                for decoded in self.invoke_streaming(
                    body=body,
                    session_id=session_id,
                    bearer_token=bearer_token,
                    headers=headers,
                ):
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, decoded)
                loop.call_soon_threadsafe(chunk_queue.put_nowait, sentinel)
            except Exception as exc:  # noqa: BLE001 — propagated to caller
                loop.call_soon_threadsafe(chunk_queue.put_nowait, exc)
                loop.call_soon_threadsafe(chunk_queue.put_nowait, sentinel)

        thread = threading.Thread(target=stream_in_thread, daemon=True)
        thread.start()

        while True:
            item = await chunk_queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
            await asyncio.sleep(0)

    # ------------------------------------------------------------------ #
    # execute_command / execute_command_streaming
    # ------------------------------------------------------------------ #

    def execute_command(
        self,
        command: str,
        session_id: str,
        bearer_token: str,
        command_timeout: Optional[int] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Run a shell command inside the runtime session and collect the full result.

        Blocking. Accumulates all ``stdout`` and ``stderr`` chunks from
        the EventStream and returns the final exit status.

        Args:
            command: Shell command to run (1 B – 64 KB per the
                ``InvokeAgentRuntimeCommand`` API).
            session_id: Runtime session id to target. The filesystem
                inside the container persists across calls, but a fresh
                shell is spawned each time, so working directory and
                environment variables do not.
            bearer_token: Bearer token for authentication.
            command_timeout: Server-side command wall-clock timeout in
                seconds (1–3600). Defaults to :attr:`timeout`. The
                HTTP read timeout is derived internally as
                ``command_timeout + 30``.
            headers: Optional extra headers to include.

        Returns:
            Dict with keys ``"stdout"`` (str), ``"stderr"`` (str),
            ``"exitCode"`` (int, ``-1`` if no ``contentStop`` was
            received), and ``"status"`` (``"COMPLETED"`` or
            ``"TIMED_OUT"``, or ``"UNKNOWN"`` if no ``contentStop`` was
            received).

        Raises:
            AgentRuntimeError: If the runtime returns a non-2xx status.
        """
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        exit_code: int = -1
        status: str = "UNKNOWN"

        for event in self.execute_command_streaming(
            command=command,
            session_id=session_id,
            bearer_token=bearer_token,
            command_timeout=command_timeout,
            headers=headers,
        ):
            if "contentDelta" in event:
                delta = event["contentDelta"]
                if "stdout" in delta:
                    stdout_parts.append(delta["stdout"])
                if "stderr" in delta:
                    stderr_parts.append(delta["stderr"])
            elif "contentStop" in event:
                exit_code = int(event["contentStop"].get("exitCode", -1))
                status = str(event["contentStop"].get("status", "UNKNOWN"))

        return {
            "stdout": "".join(stdout_parts),
            "stderr": "".join(stderr_parts),
            "exitCode": exit_code,
            "status": status,
        }

    def execute_command_streaming(
        self,
        command: str,
        session_id: str,
        bearer_token: str,
        command_timeout: Optional[int] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream AWS EventStream events from ``InvokeAgentRuntimeCommand``.

        Yields the decoded event payloads (the value inside the
        server's ``"chunk"`` envelope). Each yielded dict has exactly one
        of the keys ``"contentStart"``, ``"contentDelta"``, or
        ``"contentStop"``.

        Args:
            command: Shell command to run.
            session_id: Runtime session id.
            bearer_token: Bearer token for authentication.
            command_timeout: Server-side wall-clock timeout in seconds
                (1–3600). Defaults to :attr:`timeout`.
            headers: Optional extra headers to include.

        Yields:
            Parsed event payload dicts.

        Raises:
            AgentRuntimeError: If the runtime returns a non-2xx status.
        """
        effective_timeout = (
            command_timeout if command_timeout is not None else self.timeout
        )

        request_headers = self._build_headers(
            bearer_token,
            session_id,
            accept="application/vnd.amazon.eventstream",
            content_type="application/json",
        )
        if headers:
            request_headers.update(headers)

        response = self._http.request(
            "POST",
            self._build_url("commands"),
            headers=request_headers,
            body=json.dumps({"command": command, "timeout": effective_timeout}).encode(
                "utf-8"
            ),
            timeout=effective_timeout + 30,
            preload_content=False,
        )
        try:
            self._check_response(response)

            buf = EventStreamBuffer()
            for chunk in response.stream(4096):
                if not chunk:
                    continue
                buf.add_data(chunk)
                for event in buf:
                    payload = event.payload
                    if not payload:
                        continue
                    try:
                        decoded = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    inner = decoded.get("chunk") if isinstance(decoded, dict) else None
                    yield inner if isinstance(inner, dict) else decoded
        finally:
            response.release_conn()

    # ------------------------------------------------------------------ #
    # stop_runtime_session
    # ------------------------------------------------------------------ #

    def stop_runtime_session(
        self,
        session_id: str,
        bearer_token: str,
        client_token: Optional[str] = None,
    ) -> dict[str, Any]:
        """Terminate a runtime session.

        Args:
            session_id: The session id to stop.
            bearer_token: Bearer token for authentication.
            client_token: Idempotency token. Auto-generated as a UUID4
                when not supplied.

        Returns:
            Parsed JSON body of the response (often an empty dict).

        Raises:
            AgentRuntimeError: If the runtime returns a non-2xx status
                (for example ``HTTP 404`` for an unknown session).
        """
        request_headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
            "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
        }
        response = self._http.request(
            "POST",
            self._build_url("stopruntimesession"),
            headers=request_headers,
            body=json.dumps({"clientToken": client_token or str(uuid.uuid4())}).encode(
                "utf-8"
            ),
            timeout=self.timeout,
            preload_content=True,
        )
        self._check_response(response)
        if not response.data:
            return {}
        try:
            parsed = json.loads(response.data)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {"response": parsed}

    # ------------------------------------------------------------------ #
    # Response parsing
    # ------------------------------------------------------------------ #

    def _check_response(self, response: _UrllibResponse) -> None:
        """Raise :class:`AgentRuntimeError` for non-2xx responses.

        Attempts to parse the body as JSON and surface ``error``,
        ``error_type``, and ``message`` fields. Falls back to the raw
        body text.

        Args:
            response: The urllib3 response to inspect.

        Raises:
            AgentRuntimeError: If the status code is 400 or above.
        """
        if response.status < 400:
            return

        body_bytes = response.read()
        try:
            body: Any = json.loads(body_bytes)
        except (json.JSONDecodeError, ValueError):
            body = body_bytes.decode("utf-8", errors="replace")

        error_type = f"HTTP {response.status}"
        reason = response.reason or error_type

        if isinstance(body, dict):
            raise AgentRuntimeError(
                error=body.get("error", reason),
                error_type=body.get("error_type", error_type),
                message=body.get(
                    "message", body_bytes.decode("utf-8", errors="replace")
                ),
            )
        raise AgentRuntimeError(
            error=str(body) or reason,
            error_type=error_type,
        )

    def _read_non_streaming(self, response: _UrllibResponse) -> str:
        """Read a fully-buffered response body and unwrap JSON strings.

        Args:
            response: The urllib3 response, opened with
                ``preload_content=False``.

        Returns:
            The response body as text. If the body is a JSON-encoded
            string (e.g. ``"hello"``), the unwrapped value is returned.
        """
        data = response.read()
        text = data.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text
        return parsed if isinstance(parsed, str) else text

    def _iter_sse_decoded(self, response: _UrllibResponse) -> Iterator[str]:
        """Iterate decoded SSE payloads from a streaming response.

        Args:
            response: The urllib3 response, opened with
                ``preload_content=False``.

        Yields:
            Decoded payload strings (one per non-empty ``data:`` line
            or JSON line).
        """
        for raw_line in self._iter_lines(response):
            if not raw_line:
                continue
            decoded = self._decode_sse_line(raw_line.decode("utf-8", errors="replace"))
            if decoded:
                yield decoded

    @staticmethod
    def _iter_lines(response: _UrllibResponse) -> Iterator[bytes]:
        r"""Yield lines from a streaming urllib3 response.

        Splits on ``\n`` and strips a trailing ``\r`` so it handles
        both LF and CRLF line endings. Preserves empty lines (the caller
        filters them).

        Args:
            response: The urllib3 response, opened with
                ``preload_content=False``.

        Yields:
            Each line as bytes (with no trailing newline).
        """
        pending = b""
        for chunk in response.stream(1024):
            if not chunk:
                continue
            pending += chunk
            while True:
                idx = pending.find(b"\n")
                if idx < 0:
                    break
                line, pending = pending[:idx], pending[idx + 1 :]
                if line.endswith(b"\r"):
                    line = line[:-1]
                yield line
        if pending:
            if pending.endswith(b"\r"):
                pending = pending[:-1]
            yield pending

    def _decode_sse_line(self, line: str) -> Optional[str]:
        """Decode a single SSE or JSON-Lines line.

        Handles SSE ``data:`` lines, JSON error envelopes, JSON-encoded
        strings, and plain text. Error envelopes (with an ``error`` key)
        are raised as :class:`AgentRuntimeError`.

        Args:
            line: A single line of the streamed response.

        Returns:
            The decoded payload, or ``None`` if the line is a comment,
            empty, or carries no renderable text.

        Raises:
            AgentRuntimeError: If the line carries a JSON error payload.
        """
        line = line.strip()
        if not line or line.startswith(":"):
            return None

        if line.startswith("data:"):
            content = line[5:].strip()

            if content.startswith("{"):
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "error" in data:
                        raise AgentRuntimeError(
                            error=str(data["error"]),
                            error_type=str(data.get("error_type", "")),
                            message=str(data.get("message", data["error"])),
                        )
                except json.JSONDecodeError:
                    pass

            if content.startswith('"'):
                try:
                    unwrapped = json.loads(content)
                except json.JSONDecodeError:
                    if content.endswith('"'):
                        return content[1:-1]
                    return content
                return unwrapped if isinstance(unwrapped, str) else content
            return content

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return line

        if isinstance(data, dict):
            if "error" in data:
                raise AgentRuntimeError(
                    error=str(data["error"]),
                    error_type=str(data.get("error_type", "")),
                    message=str(data.get("message", data["error"])),
                )
            if "text" in data:
                value = data["text"]
                return value if isinstance(value, str) else str(value)
            if "content" in data:
                value = data["content"]
                return value if isinstance(value, str) else str(value)
            if "data" in data:
                return str(data["data"])
        return None
