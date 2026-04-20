"""Tests for AgentCoreRuntimeHttpClient."""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional
from unittest.mock import MagicMock

import pytest

from bedrock_agentcore.runtime.agent_core_runtime_http_client import (
    AgentCoreRuntimeHttpClient,
    AgentRuntimeError,
)

ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/my-runtime-abc"
BEARER = "test-token"
SESSION = "a" * 36


class _FakeResponse:
    """Minimal stand-in for urllib3's response used by the HTTP client.

    Exposes the exact surface the client touches: ``status``, ``reason``,
    ``headers``, ``data``, ``read``, ``stream``, and ``release_conn``.
    """

    def __init__(
        self,
        status: int = 200,
        headers: Optional[dict[str, str]] = None,
        body: bytes = b"",
        chunks: Optional[list[bytes]] = None,
        reason: str = "",
    ) -> None:
        self.status = status
        self.reason = reason
        self.headers = headers or {}
        self._body = body
        self._chunks = chunks
        self._consumed = False
        self.release_calls = 0

    @property
    def data(self) -> bytes:
        return self._body

    def read(self) -> bytes:
        if self._consumed:
            return b""
        self._consumed = True
        return self._body

    def stream(self, amt: int = 1024) -> Iterator[bytes]:
        if self._chunks is not None:
            for chunk in self._chunks:
                yield chunk
            return
        # Fall back to emitting the full body as one chunk.
        if self._body:
            yield self._body

    def release_conn(self) -> None:
        self.release_calls += 1


def _make_client(
    response: _FakeResponse,
) -> tuple[AgentCoreRuntimeHttpClient, MagicMock]:
    """Build a client whose PoolManager returns ``response``."""
    pool = MagicMock()
    pool.request.return_value = response
    client = AgentCoreRuntimeHttpClient(agent_arn=ARN, pool_manager=pool)
    return client, pool


# --------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------- #


class TestInit:
    """Tests for AgentCoreRuntimeHttpClient.__init__."""

    def test_parses_region_from_arn(self) -> None:
        """Region is extracted from the ARN."""
        client = AgentCoreRuntimeHttpClient(agent_arn=ARN)
        assert client.region == "us-west-2"

    def test_defaults(self) -> None:
        """Default field values."""
        client = AgentCoreRuntimeHttpClient(agent_arn=ARN)
        assert client.endpoint_name == "DEFAULT"
        assert client.timeout == 300
        assert client.content_type == "application/json"
        assert client.accept == "application/json"

    def test_custom_fields(self) -> None:
        """Non-default field values are stored."""
        client = AgentCoreRuntimeHttpClient(
            agent_arn=ARN,
            endpoint_name="DEV",
            timeout=42,
            content_type="application/cbor",
            accept="text/plain",
        )
        assert client.endpoint_name == "DEV"
        assert client.timeout == 42
        assert client.content_type == "application/cbor"
        assert client.accept == "text/plain"

    def test_invalid_arn_missing_region(self) -> None:
        """An ARN with no region raises ValueError."""
        with pytest.raises(ValueError, match="Invalid agent ARN"):
            AgentCoreRuntimeHttpClient(agent_arn="arn:aws:bedrock-agentcore")

    def test_invalid_arn_empty_region(self) -> None:
        """An ARN with empty region raises ValueError."""
        with pytest.raises(ValueError, match="Invalid agent ARN"):
            AgentCoreRuntimeHttpClient(
                agent_arn="arn:aws:bedrock-agentcore::123:runtime/x"
            )

    def test_pool_manager_injection(self) -> None:
        """A caller-provided PoolManager is used verbatim."""
        pool = MagicMock()
        client = AgentCoreRuntimeHttpClient(agent_arn=ARN, pool_manager=pool)
        assert client._http is pool


# --------------------------------------------------------------------- #
# _build_url / _build_headers / _serialize_body
# --------------------------------------------------------------------- #


class TestBuildUrl:
    """Tests for _build_url."""

    def test_invocations_url(self) -> None:
        """URL embeds region, URL-encoded ARN, path, and qualifier."""
        client = AgentCoreRuntimeHttpClient(agent_arn=ARN, endpoint_name="DEFAULT")
        url = client._build_url("invocations")
        assert url.startswith(
            "https://bedrock-agentcore.us-west-2.amazonaws.com/runtimes/"
        )
        assert "/invocations?qualifier=DEFAULT" in url
        assert "arn%3Aaws%3Abedrock-agentcore" in url  # colons URL-encoded

    def test_commands_url(self) -> None:
        """Different path suffix yields the commands endpoint."""
        client = AgentCoreRuntimeHttpClient(agent_arn=ARN)
        assert "/commands?qualifier=DEFAULT" in client._build_url("commands")

    def test_non_default_qualifier(self) -> None:
        """Qualifier reflects endpoint_name."""
        client = AgentCoreRuntimeHttpClient(agent_arn=ARN, endpoint_name="DEV")
        assert "qualifier=DEV" in client._build_url("invocations")


class TestBuildHeaders:
    """Tests for _build_headers."""

    def test_default_headers(self) -> None:
        """Includes auth, content-type, accept, session-id."""
        client = AgentCoreRuntimeHttpClient(agent_arn=ARN)
        headers = client._build_headers(BEARER, SESSION)
        assert headers["Authorization"] == f"Bearer {BEARER}"
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"
        assert headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] == SESSION

    def test_override_accept_and_content_type(self) -> None:
        """Per-call overrides replace the defaults."""
        client = AgentCoreRuntimeHttpClient(agent_arn=ARN)
        headers = client._build_headers(
            BEARER,
            SESSION,
            accept="application/vnd.amazon.eventstream",
            content_type="application/xml",
        )
        assert headers["Accept"] == "application/vnd.amazon.eventstream"
        assert headers["Content-Type"] == "application/xml"


class TestSerializeBody:
    """Tests for _serialize_body."""

    def test_json_dict(self) -> None:
        """Dict body is JSON-serialized when content type is JSON."""
        client = AgentCoreRuntimeHttpClient(agent_arn=ARN)
        assert client._serialize_body({"a": 1}) == b'{"a": 1}'

    def test_bytes_passthrough_non_json(self) -> None:
        """Bytes body is sent verbatim for non-JSON content types."""
        client = AgentCoreRuntimeHttpClient(
            agent_arn=ARN, content_type="application/octet-stream"
        )
        assert client._serialize_body(b"raw") == b"raw"

    def test_str_utf8_encoded_non_json(self) -> None:
        """String body is UTF-8 encoded for non-JSON content types."""
        client = AgentCoreRuntimeHttpClient(agent_arn=ARN, content_type="text/plain")
        assert client._serialize_body("héllo") == "héllo".encode("utf-8")

    def test_fallback_to_json(self) -> None:
        """Non-str, non-bytes body falls back to JSON even with non-JSON content type."""
        client = AgentCoreRuntimeHttpClient(
            agent_arn=ARN, content_type="application/cbor"
        )
        assert client._serialize_body({"x": 1}) == b'{"x": 1}'


# --------------------------------------------------------------------- #
# invoke (non-streaming JSON, non-streaming plain, SSE)
# --------------------------------------------------------------------- #


class TestInvoke:
    """Tests for invoke."""

    def test_non_streaming_json_string(self) -> None:
        """JSON string response is unwrapped."""
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=b'"hello"',
        )
        client, _ = _make_client(resp)
        assert client.invoke({"prompt": "hi"}, SESSION, BEARER) == "hello"
        assert resp.release_calls == 1

    def test_non_streaming_json_object_returns_text(self) -> None:
        """Non-string JSON response returns the raw body text."""
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=b'{"answer": 42}',
        )
        client, _ = _make_client(resp)
        assert client.invoke({"prompt": "hi"}, SESSION, BEARER) == '{"answer": 42}'

    def test_non_streaming_invalid_json_returns_text(self) -> None:
        """If the body isn't valid JSON, the raw text is returned."""
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=b"plain-text-not-json",
        )
        client, _ = _make_client(resp)
        assert client.invoke({"prompt": "hi"}, SESSION, BEARER) == "plain-text-not-json"

    def test_sse_streaming_concatenates(self) -> None:
        """Multiple SSE data lines are concatenated in order."""
        body = b"data: hello\ndata: world\n"
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "text/event-stream"},
            chunks=[body],
        )
        client, _ = _make_client(resp)
        assert client.invoke({"prompt": "hi"}, SESSION, BEARER) == "helloworld"

    def test_custom_headers_merged(self) -> None:
        """Caller-supplied headers are merged into the request."""
        resp = _FakeResponse(
            status=200, headers={"content-type": "application/json"}, body=b'"ok"'
        )
        client, pool = _make_client(resp)
        client.invoke({}, SESSION, BEARER, headers={"X-Test": "1"})
        sent_headers = pool.request.call_args.kwargs["headers"]
        assert sent_headers["X-Test"] == "1"
        assert sent_headers["Authorization"] == f"Bearer {BEARER}"

    def test_non_ok_raises(self) -> None:
        """4xx and 5xx responses become AgentRuntimeError."""
        resp = _FakeResponse(
            status=500,
            body=b'{"error": "oops", "message": "boom"}',
            reason="Server Error",
        )
        client, _ = _make_client(resp)
        with pytest.raises(AgentRuntimeError) as excinfo:
            client.invoke({}, SESSION, BEARER)
        assert excinfo.value.error == "oops"
        assert "boom" in str(excinfo.value)


# --------------------------------------------------------------------- #
# invoke_streaming
# --------------------------------------------------------------------- #


class TestInvokeStreaming:
    """Tests for invoke_streaming."""

    def test_yields_chunks(self) -> None:
        """Yields each decoded SSE payload in order."""
        body = b"data: first\ndata: second\n"
        resp = _FakeResponse(
            status=200, headers={"content-type": "text/event-stream"}, chunks=[body]
        )
        client, _ = _make_client(resp)
        assert list(client.invoke_streaming({}, SESSION, BEARER)) == ["first", "second"]

    def test_non_ok_raises(self) -> None:
        """Errors are surfaced before the generator yields."""
        resp = _FakeResponse(status=404, body=b"not found", reason="Not Found")
        client, _ = _make_client(resp)
        with pytest.raises(AgentRuntimeError):
            list(client.invoke_streaming({}, SESSION, BEARER))

    def test_custom_headers_merged(self) -> None:
        """Custom headers are merged into the request."""
        resp = _FakeResponse(
            status=200, headers={"content-type": "text/event-stream"}, chunks=[b""]
        )
        client, pool = _make_client(resp)
        list(client.invoke_streaming({}, SESSION, BEARER, headers={"X-Custom": "v"}))
        assert pool.request.call_args.kwargs["headers"]["X-Custom"] == "v"


# --------------------------------------------------------------------- #
# invoke_streaming_async
# --------------------------------------------------------------------- #


class TestInvokeStreamingAsync:
    """Tests for invoke_streaming_async."""

    @pytest.mark.asyncio
    async def test_yields_async_chunks(self) -> None:
        """Async generator yields the same chunks as the sync version."""
        body = b"data: a\ndata: b\n"
        resp = _FakeResponse(
            status=200, headers={"content-type": "text/event-stream"}, chunks=[body]
        )
        client, _ = _make_client(resp)
        out: list[str] = []
        async for chunk in client.invoke_streaming_async({}, SESSION, BEARER):
            out.append(chunk)
        assert out == ["a", "b"]

    @pytest.mark.asyncio
    async def test_propagates_exception(self) -> None:
        """Exceptions from the background thread surface to the caller."""
        resp = _FakeResponse(status=500, body=b"err", reason="Server Error")
        client, _ = _make_client(resp)
        with pytest.raises(AgentRuntimeError):
            async for _ in client.invoke_streaming_async({}, SESSION, BEARER):
                pass


# --------------------------------------------------------------------- #
# execute_command / execute_command_streaming
# --------------------------------------------------------------------- #


def _encode_eventstream_frame(payload: dict[str, Any]) -> bytes:
    """Encode a JSON payload using botocore's EventStream framing.

    Uses the same encoder that the server would use to build a valid frame
    the client can parse.
    """

    body = json.dumps(payload).encode("utf-8")
    # Build prelude manually. Easier: construct bytes that match the wire format.
    # Format: total_length (4), headers_length (4), prelude_crc (4), headers, payload, message_crc (4).
    # With no headers: headers_length = 0.
    import binascii
    import struct

    headers = b""
    headers_length = len(headers)
    total_length = 4 + 4 + 4 + headers_length + len(body) + 4  # + message CRC
    prelude = struct.pack(">II", total_length, headers_length)
    prelude_crc = struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
    message_bytes = prelude + prelude_crc + headers + body
    message_crc = struct.pack(">I", binascii.crc32(message_bytes) & 0xFFFFFFFF)
    return message_bytes + message_crc


class TestExecuteCommand:
    """Tests for execute_command (blocking)."""

    def test_aggregates_output(self) -> None:
        """stdout/stderr/exitCode/status are aggregated across events."""
        events = [
            _encode_eventstream_frame({"chunk": {"contentStart": {}}}),
            _encode_eventstream_frame({"chunk": {"contentDelta": {"stdout": "hi "}}}),
            _encode_eventstream_frame({"chunk": {"contentDelta": {"stdout": "there"}}}),
            _encode_eventstream_frame({"chunk": {"contentDelta": {"stderr": "warn"}}}),
            _encode_eventstream_frame(
                {"chunk": {"contentStop": {"exitCode": 0, "status": "COMPLETED"}}}
            ),
        ]
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            chunks=events,
        )
        client, _ = _make_client(resp)
        result = client.execute_command("echo hi there", SESSION, BEARER)
        assert result == {
            "stdout": "hi there",
            "stderr": "warn",
            "exitCode": 0,
            "status": "COMPLETED",
        }

    def test_missing_stop_has_unknown_status(self) -> None:
        """Without a contentStop event, defaults are UNKNOWN / -1."""
        events = [
            _encode_eventstream_frame({"chunk": {"contentDelta": {"stdout": "x"}}}),
        ]
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            chunks=events,
        )
        client, _ = _make_client(resp)
        result = client.execute_command("echo x", SESSION, BEARER)
        assert result == {
            "stdout": "x",
            "stderr": "",
            "exitCode": -1,
            "status": "UNKNOWN",
        }


class TestExecuteCommandStreaming:
    """Tests for execute_command_streaming."""

    def test_sends_command_body(self) -> None:
        """Request body contains command and timeout."""
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            chunks=[],
        )
        client, pool = _make_client(resp)
        list(
            client.execute_command_streaming("ls", SESSION, BEARER, command_timeout=42)
        )
        sent = pool.request.call_args
        assert sent.kwargs["body"] == b'{"command": "ls", "timeout": 42}'
        # HTTP timeout must be command_timeout + 30.
        assert sent.kwargs["timeout"] == 72

    def test_defaults_to_self_timeout(self) -> None:
        """command_timeout=None falls back to self.timeout."""
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            chunks=[],
        )
        client = AgentCoreRuntimeHttpClient(agent_arn=ARN, timeout=120)
        client._http = MagicMock()
        client._http.request.return_value = resp
        list(client.execute_command_streaming("ls", SESSION, BEARER))
        sent = client._http.request.call_args
        body = json.loads(sent.kwargs["body"])
        assert body["timeout"] == 120
        assert sent.kwargs["timeout"] == 150

    def test_yields_parsed_events(self) -> None:
        """Each EventStream frame is yielded as a dict."""
        events = [
            _encode_eventstream_frame({"chunk": {"contentStart": {}}}),
            _encode_eventstream_frame(
                {"chunk": {"contentStop": {"exitCode": 2, "status": "COMPLETED"}}}
            ),
        ]
        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            chunks=events,
        )
        client, _ = _make_client(resp)
        parsed = list(client.execute_command_streaming("echo hi", SESSION, BEARER))
        assert parsed[0] == {"contentStart": {}}
        assert parsed[1] == {"contentStop": {"exitCode": 2, "status": "COMPLETED"}}

    def test_skips_event_without_payload(self) -> None:
        """Events with empty payload are skipped silently."""
        # An eventstream frame with empty payload body.
        import binascii
        import struct

        headers = b""
        body = b""
        headers_length = 0
        total_length = 4 + 4 + 4 + headers_length + len(body) + 4
        prelude = struct.pack(">II", total_length, headers_length)
        prelude_crc = struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
        message_bytes = prelude + prelude_crc + headers + body
        message_crc = struct.pack(">I", binascii.crc32(message_bytes) & 0xFFFFFFFF)
        empty_frame = message_bytes + message_crc

        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            chunks=[empty_frame],
        )
        client, _ = _make_client(resp)
        assert list(client.execute_command_streaming("ls", SESSION, BEARER)) == []

    def test_skips_bad_json_payload(self) -> None:
        """Events whose payload is not valid JSON are skipped."""
        # Build a frame whose payload is not valid JSON.
        import binascii
        import struct

        headers = b""
        body = b"not-json"
        headers_length = 0
        total_length = 4 + 4 + 4 + headers_length + len(body) + 4
        prelude = struct.pack(">II", total_length, headers_length)
        prelude_crc = struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
        message_bytes = prelude + prelude_crc + headers + body
        message_crc = struct.pack(">I", binascii.crc32(message_bytes) & 0xFFFFFFFF)
        bad_frame = message_bytes + message_crc

        resp = _FakeResponse(
            status=200,
            headers={"content-type": "application/vnd.amazon.eventstream"},
            chunks=[bad_frame],
        )
        client, _ = _make_client(resp)
        assert list(client.execute_command_streaming("ls", SESSION, BEARER)) == []

    def test_non_ok_raises(self) -> None:
        """Non-2xx responses raise AgentRuntimeError."""
        resp = _FakeResponse(status=403, body=b"forbidden", reason="Forbidden")
        client, _ = _make_client(resp)
        with pytest.raises(AgentRuntimeError):
            list(client.execute_command_streaming("ls", SESSION, BEARER))


# --------------------------------------------------------------------- #
# stop_runtime_session
# --------------------------------------------------------------------- #


class TestStopRuntimeSession:
    """Tests for stop_runtime_session."""

    def test_sends_client_token(self) -> None:
        """Request body contains the provided client token."""
        resp = _FakeResponse(status=200, body=b"{}")
        client, pool = _make_client(resp)
        client.stop_runtime_session(SESSION, BEARER, client_token="abc")
        sent = json.loads(pool.request.call_args.kwargs["body"])
        assert sent == {"clientToken": "abc"}

    def test_autogenerates_client_token(self) -> None:
        """Missing client_token is replaced with a UUID4."""
        resp = _FakeResponse(status=200, body=b"{}")
        client, pool = _make_client(resp)
        client.stop_runtime_session(SESSION, BEARER)
        sent = json.loads(pool.request.call_args.kwargs["body"])
        # UUID4 string is 36 chars (8-4-4-4-12).
        assert len(sent["clientToken"]) == 36

    def test_returns_empty_dict_on_blank_body(self) -> None:
        """Empty response body yields an empty dict."""
        resp = _FakeResponse(status=200, body=b"")
        client, _ = _make_client(resp)
        assert client.stop_runtime_session(SESSION, BEARER) == {}

    def test_returns_parsed_dict(self) -> None:
        """Dict body is returned as-is."""
        resp = _FakeResponse(status=200, body=b'{"sessionId": "x"}')
        client, _ = _make_client(resp)
        assert client.stop_runtime_session(SESSION, BEARER) == {"sessionId": "x"}

    def test_non_dict_body_wrapped(self) -> None:
        """A JSON body that isn't a dict is wrapped under 'response'."""
        resp = _FakeResponse(status=200, body=b'["a", "b"]')
        client, _ = _make_client(resp)
        assert client.stop_runtime_session(SESSION, BEARER) == {"response": ["a", "b"]}

    def test_invalid_json_body_returns_empty(self) -> None:
        """Invalid JSON body returns empty dict rather than raising."""
        resp = _FakeResponse(status=200, body=b"not json")
        client, _ = _make_client(resp)
        assert client.stop_runtime_session(SESSION, BEARER) == {}

    def test_404_raises(self) -> None:
        """Unknown-session HTTP 404 becomes AgentRuntimeError."""
        resp = _FakeResponse(status=404, body=b"{}", reason="Not Found")
        client, _ = _make_client(resp)
        with pytest.raises(AgentRuntimeError) as excinfo:
            client.stop_runtime_session(SESSION, BEARER)
        assert excinfo.value.error_type == "HTTP 404"


# --------------------------------------------------------------------- #
# _check_response
# --------------------------------------------------------------------- #


class TestCheckResponse:
    """Tests for _check_response."""

    def test_noop_on_2xx(self) -> None:
        """2xx responses do not raise."""
        client = AgentCoreRuntimeHttpClient(agent_arn=ARN)
        client._check_response(_FakeResponse(status=200))

    def test_parses_json_error_body(self) -> None:
        """JSON error bodies populate error/error_type/message."""
        client = AgentCoreRuntimeHttpClient(agent_arn=ARN)
        resp = _FakeResponse(
            status=400,
            body=b'{"error": "bad", "error_type": "Validation", "message": "details"}',
            reason="Bad Request",
        )
        with pytest.raises(AgentRuntimeError) as excinfo:
            client._check_response(resp)
        assert excinfo.value.error == "bad"
        assert excinfo.value.error_type == "Validation"

    def test_falls_back_to_text_body(self) -> None:
        """Non-JSON bodies become the error string."""
        client = AgentCoreRuntimeHttpClient(agent_arn=ARN)
        resp = _FakeResponse(status=500, body=b"internal boom", reason="Server Error")
        with pytest.raises(AgentRuntimeError) as excinfo:
            client._check_response(resp)
        assert "internal boom" in excinfo.value.error
        assert excinfo.value.error_type == "HTTP 500"

    def test_defaults_error_type_for_dict_body(self) -> None:
        """Missing error_type in JSON body defaults to 'HTTP <status>'."""
        client = AgentCoreRuntimeHttpClient(agent_arn=ARN)
        resp = _FakeResponse(status=502, body=b'{"message": "bad gateway"}')
        with pytest.raises(AgentRuntimeError) as excinfo:
            client._check_response(resp)
        assert excinfo.value.error_type == "HTTP 502"

    def test_empty_body_uses_reason(self) -> None:
        """Empty body falls back to the response reason string."""
        client = AgentCoreRuntimeHttpClient(agent_arn=ARN)
        resp = _FakeResponse(status=503, body=b"", reason="Service Unavailable")
        with pytest.raises(AgentRuntimeError) as excinfo:
            client._check_response(resp)
        assert excinfo.value.error == "Service Unavailable"


# --------------------------------------------------------------------- #
# _decode_sse_line — all branches
# --------------------------------------------------------------------- #


class TestDecodeSseLine:
    """Tests for _decode_sse_line."""

    @pytest.fixture
    def client(self) -> AgentCoreRuntimeHttpClient:
        return AgentCoreRuntimeHttpClient(agent_arn=ARN)

    def test_empty_line(self, client: AgentCoreRuntimeHttpClient) -> None:
        """Empty / whitespace-only lines return None."""
        assert client._decode_sse_line("") is None
        assert client._decode_sse_line("   ") is None

    def test_comment_line(self, client: AgentCoreRuntimeHttpClient) -> None:
        """SSE comment lines (starting with ':') return None."""
        assert client._decode_sse_line(": keepalive") is None

    def test_data_with_json_encoded_string(
        self, client: AgentCoreRuntimeHttpClient
    ) -> None:
        """data: "hello" → hello (unwrapped)."""
        assert client._decode_sse_line('data: "hello"') == "hello"

    def test_data_with_plain_text(self, client: AgentCoreRuntimeHttpClient) -> None:
        """data: plain → plain."""
        assert client._decode_sse_line("data: plain") == "plain"

    def test_data_with_malformed_json_string(
        self, client: AgentCoreRuntimeHttpClient
    ) -> None:
        """data: "bad" but missing closing quote: strip what we can."""
        assert client._decode_sse_line('data: "oops') == '"oops'

    def test_data_with_malformed_quoted_string(
        self, client: AgentCoreRuntimeHttpClient
    ) -> None:
        """data: \"truncated\\\\\" with trailing quote but bad escape yields trimmed string."""
        # A string that starts and ends with quotes but contains an invalid escape mid-string.
        # json.loads fails; the fallback strips the outer quotes.
        assert client._decode_sse_line('data: "bad\\escape"') == "bad\\escape"

    def test_data_with_json_error_raises(
        self, client: AgentCoreRuntimeHttpClient
    ) -> None:
        """data: { "error": ... } raises AgentRuntimeError."""
        with pytest.raises(AgentRuntimeError) as excinfo:
            client._decode_sse_line('data: {"error": "boom", "error_type": "T"}')
        assert excinfo.value.error == "boom"

    def test_data_with_json_object_no_error(
        self, client: AgentCoreRuntimeHttpClient
    ) -> None:
        """data: {...} without an error key passes through as the raw JSON content."""
        assert client._decode_sse_line('data: {"foo": "bar"}') == '{"foo": "bar"}'

    def test_data_with_broken_json_passes_through(
        self, client: AgentCoreRuntimeHttpClient
    ) -> None:
        """data: { broken passes through as the content after 'data:'."""
        assert client._decode_sse_line("data: {not json") == "{not json"

    def test_plain_json_text_key(self, client: AgentCoreRuntimeHttpClient) -> None:
        """Plain JSON line with 'text' key returns that value."""
        assert client._decode_sse_line('{"text": "hi"}') == "hi"

    def test_plain_json_content_key(self, client: AgentCoreRuntimeHttpClient) -> None:
        """Plain JSON line with 'content' key returns that value."""
        assert client._decode_sse_line('{"content": "c"}') == "c"

    def test_plain_json_data_key(self, client: AgentCoreRuntimeHttpClient) -> None:
        """Plain JSON line with 'data' key returns str(that value)."""
        assert client._decode_sse_line('{"data": 42}') == "42"

    def test_plain_json_error_raises(self, client: AgentCoreRuntimeHttpClient) -> None:
        """Plain JSON line with 'error' raises."""
        with pytest.raises(AgentRuntimeError):
            client._decode_sse_line('{"error": "bad"}')

    def test_plain_json_unknown_shape_returns_none(
        self, client: AgentCoreRuntimeHttpClient
    ) -> None:
        """Plain JSON dict without known keys returns None."""
        assert client._decode_sse_line('{"something": "else"}') is None

    def test_plain_text_passes_through(
        self, client: AgentCoreRuntimeHttpClient
    ) -> None:
        """Non-JSON plain text is returned verbatim."""
        assert client._decode_sse_line("hello world") == "hello world"


# --------------------------------------------------------------------- #
# _iter_lines
# --------------------------------------------------------------------- #


class TestIterLines:
    """Tests for _iter_lines chunk splitting."""

    def test_splits_on_lf(self) -> None:
        """Splits on newline and yields each line."""
        resp = _FakeResponse(chunks=[b"one\ntwo\nthree\n"])
        lines = list(AgentCoreRuntimeHttpClient._iter_lines(resp))
        assert lines == [b"one", b"two", b"three"]

    def test_strips_crlf(self) -> None:
        """Trailing \\r is stripped."""
        resp = _FakeResponse(chunks=[b"a\r\nb\r\n"])
        lines = list(AgentCoreRuntimeHttpClient._iter_lines(resp))
        assert lines == [b"a", b"b"]

    def test_trailing_without_newline(self) -> None:
        """A trailing fragment without newline is still yielded."""
        resp = _FakeResponse(chunks=[b"done"])
        lines = list(AgentCoreRuntimeHttpClient._iter_lines(resp))
        assert lines == [b"done"]

    def test_trailing_cr_stripped(self) -> None:
        """A trailing CR without LF is stripped."""
        resp = _FakeResponse(chunks=[b"done\r"])
        lines = list(AgentCoreRuntimeHttpClient._iter_lines(resp))
        assert lines == [b"done"]

    def test_split_across_chunks(self) -> None:
        """Lines spanning chunk boundaries are reassembled."""
        resp = _FakeResponse(chunks=[b"hel", b"lo\nwo", b"rld\n"])
        lines = list(AgentCoreRuntimeHttpClient._iter_lines(resp))
        assert lines == [b"hello", b"world"]

    def test_empty_chunk_skipped(self) -> None:
        """Empty chunks in the stream are ignored."""
        resp = _FakeResponse(chunks=[b"a\n", b"", b"b\n"])
        lines = list(AgentCoreRuntimeHttpClient._iter_lines(resp))
        assert lines == [b"a", b"b"]


# --------------------------------------------------------------------- #
# AgentRuntimeError
# --------------------------------------------------------------------- #


class TestAgentRuntimeError:
    """Tests for AgentRuntimeError construction and rendering."""

    def test_stores_error_and_type(self) -> None:
        """error and error_type are accessible as attributes."""
        exc = AgentRuntimeError(error="boom", error_type="T")
        assert exc.error == "boom"
        assert exc.error_type == "T"

    def test_default_error_type_empty(self) -> None:
        """error_type defaults to the empty string."""
        exc = AgentRuntimeError(error="boom")
        assert exc.error_type == ""

    def test_str_uses_message(self) -> None:
        """str() prefers the message when provided."""
        exc = AgentRuntimeError(error="e", message="human-readable")
        assert str(exc) == "human-readable"

    def test_str_falls_back_to_error(self) -> None:
        """str() falls back to the error token when no message."""
        exc = AgentRuntimeError(error="only-error")
        assert str(exc) == "only-error"

    def test_is_exception(self) -> None:
        """Subclasses Exception so it can be raised."""
        try:
            raise AgentRuntimeError(error="x")
        except Exception as exc:
            assert isinstance(exc, AgentRuntimeError)
