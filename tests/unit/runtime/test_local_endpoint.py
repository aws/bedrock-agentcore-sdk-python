"""Tests for LocalEndpoint, which relays a browser's requests to one session port.

A plain HTTP server and a plain WebSocket server on localhost stand in for the session
URL. The relay still signs for the session host and sends that host in ``Host``.
"""

import asyncio
import hashlib
import http.client
import http.server
import threading

import boto3
import pytest
from websockets.asyncio.server import serve
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from bedrock_agentcore.runtime import local_endpoint as local_endpoint_module
from bedrock_agentcore.runtime.agent_core_runtime_client import AgentCoreRuntimeClient
from bedrock_agentcore.runtime.auth import OAuthAuth, PresignedAuth
from bedrock_agentcore.runtime.local_endpoint import LocalEndpoint
from bedrock_agentcore.runtime.session_endpoint import EMPTY_PAYLOAD_SHA256, SessionEndpointClient

SESSION_HOST = "abcdefghijklmnopqrstuvwx.runtime.bedrock-agentcore.us-west-2.on.aws"
TARGET_HOST = f"8090--{SESSION_HOST}"


def _endpoint(auth="sigv4") -> SessionEndpointClient:
    session = boto3.Session(
        aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
        aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        aws_session_token="token",
        region_name="us-west-2",
    )
    return SessionEndpointClient(f"https://{SESSION_HOST}", region="us-west-2", session=session, auth=auth)


class FakeHttpUpstream:
    """Records each request and answers with a canned response."""

    def __init__(self, status=200, body=b"hi", headers=(("Content-Type", "text/plain"),)):
        self.requests = []
        upstream = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                upstream.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": self.headers,
                        "body": self.rfile.read(length) if length else b"",
                    }
                )
                self.send_response(status)
                for name, value in headers:
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_PROPFIND = _answer

            def log_message(self, *_args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()


class FakeWebSocketUpstream:
    """Echoes every message back with a prefix, and records the handshake headers."""

    def __init__(self, subprotocols=None, refuse=False):
        self.handshakes = []
        self.port = None
        self._ready = threading.Event()
        self._subprotocols, self._refuse = subprotocols, refuse
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait(5)

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self):
        async def echo(connection):
            async for message in connection:
                await connection.send(b"echo:" + message if isinstance(message, bytes) else "echo:" + message)

        def record(connection, request):
            self.handshakes.append(request.headers)
            if self._refuse:
                return connection.respond(403, "no\n")
            return None

        self._stop = asyncio.Event()
        async with serve(echo, "127.0.0.1", 0, process_request=record, subprotocols=self._subprotocols) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    def stop(self):
        self._loop.call_soon_threadsafe(self._stop.set)


@pytest.fixture
def upstream():
    server = FakeHttpUpstream()
    yield server
    server.stop()


def _local(endpoint=None, http_port=None, ws_port=None) -> LocalEndpoint:
    local = (endpoint or _endpoint()).open_local_endpoint(port=8090, local_port=0)
    if http_port:
        local._http_base = f"http://127.0.0.1:{http_port}"
    if ws_port:
        local._ws_base = f"ws://127.0.0.1:{ws_port}"
    return local.start()


def _get(local, path="/", method="GET", body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", local.listen_port, timeout=10)
    sent = {"Host": f"localhost:{local.listen_port}"}
    sent.update(headers or {})
    connection.request(method, path, body=body, headers=sent)
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read(), response.headers
    connection.close()
    return result


class TestHttp:
    def test_relays_the_response(self, upstream):
        with _local(http_port=upstream.port) as local:
            status, _, body, raw = _get(local)
        assert (status, body) == (200, b"hi")
        assert raw["Content-Type"] == "text/plain"

    def test_signs_for_the_port_prefixed_host(self, upstream):
        with _local(http_port=upstream.port) as local:
            _get(local, "/index.html")
        sent = upstream.requests[0]["headers"]
        assert sent["Host"] == TARGET_HOST
        assert sent["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert sent["X-Amz-Content-SHA256"] == EMPTY_PAYLOAD_SHA256
        assert len(sent.get_all("Host")) == 1

    def test_a_bodyless_request_says_its_length_is_zero(self, upstream):
        with _local(http_port=upstream.port) as local:
            _get(local)
        assert upstream.requests[0]["headers"]["Content-Length"] == "0"

    def test_a_body_is_forwarded_with_its_length(self, upstream):
        with _local(http_port=upstream.port) as local:
            _get(local, "/api", method="POST", body=b"hello")
        assert upstream.requests[0]["headers"]["Content-Length"] == "5"
        assert upstream.requests[0]["body"] == b"hello"

    def test_the_method_path_and_encoded_query_are_kept(self, upstream):
        with _local(http_port=upstream.port) as local:
            _get(local, "/files/a%2Fb?x=1&y=%20", method="PROPFIND")
        assert upstream.requests[0]["method"] == "PROPFIND"
        assert upstream.requests[0]["path"] == "/files/a%2Fb?x=1&y=%20"

    def test_a_bearer_token_replaces_the_signature(self, upstream):
        with _local(_endpoint(OAuthAuth(bearer_token="jwt-value")), http_port=upstream.port) as local:
            _get(local)
        sent = upstream.requests[0]["headers"]
        assert sent["Authorization"] == "Bearer jwt-value"
        assert "X-Amz-Date" not in sent

    def test_a_client_authorization_header_is_not_forwarded(self, upstream):
        with _local(_endpoint(OAuthAuth(bearer_token="jwt-value")), http_port=upstream.port) as local:
            _get(local, headers={"Authorization": "Basic dXNlcjpwYXNz"})
        assert upstream.requests[0]["headers"].get_all("Authorization") == ["Bearer jwt-value"]

    def test_a_local_origin_is_rewritten_to_the_session_host(self, upstream):
        with _local(http_port=upstream.port) as local:
            _get(local, method="POST", body=b"{}", headers={"Origin": f"http://localhost:{local.listen_port}"})
        assert upstream.requests[0]["headers"]["Origin"] == f"https://{TARGET_HOST}"

    def test_a_binary_body_arrives_intact(self):
        payload = bytes(range(256)) * 4096
        server = FakeHttpUpstream(body=payload, headers=(("Content-Type", "application/octet-stream"),))
        try:
            with _local(http_port=server.port) as local:
                _, _, body, _ = _get(local, "/archive.zip")
        finally:
            server.stop()
        assert body == payload

    def test_repeated_response_headers_are_all_relayed(self):
        cookies = (("Set-Cookie", "a=1"), ("Set-Cookie", "b=2"))
        server = FakeHttpUpstream(headers=cookies)
        try:
            with _local(http_port=server.port) as local:
                _, _, _, raw = _get(local)
        finally:
            server.stop()
        assert raw.get_all("Set-Cookie") == ["a=1", "b=2"]

    def test_a_body_over_the_limit_is_refused(self, upstream, monkeypatch):
        monkeypatch.setattr(local_endpoint_module, "MAX_BODY_BYTES", 4)
        with _local(http_port=upstream.port) as local:
            status, _, _, _ = _get(local, method="POST", body=b"too long")
        assert status == 413
        assert upstream.requests == []

    def test_an_unreachable_session_is_a_502(self):
        with _local(http_port=1) as local:
            status, _, body, _ = _get(local)
        assert status == 502
        assert b"could not reach" in body


class TestLocalOnly:
    def test_a_foreign_host_header_is_refused(self, upstream):
        with _local(http_port=upstream.port) as local:
            status, _, _, _ = _get(local, headers={"Host": f"rebind.example:{local.listen_port}"})
        assert status == 403
        assert upstream.requests == []

    def test_a_foreign_origin_is_refused(self, upstream):
        with _local(http_port=upstream.port) as local:
            status, _, _, _ = _get(local, method="POST", body=b"{}", headers={"Origin": "https://evil.example"})
        assert status == 403

    def test_a_cross_site_navigation_without_origin_is_refused(self, upstream):
        with _local(http_port=upstream.port) as local:
            status, _, _, _ = _get(local, "/delete-everything", headers={"Sec-Fetch-Site": "cross-site"})
        assert status == 403
        assert upstream.requests == []

    @pytest.mark.parametrize("site", ["none", "same-origin", "same-site"])
    def test_requests_from_this_machine_are_allowed(self, upstream, site):
        with _local(http_port=upstream.port) as local:
            status, _, _, _ = _get(local, headers={"Sec-Fetch-Site": site})
        assert status == 200

    def test_a_foreign_referer_without_fetch_metadata_is_refused(self, upstream):
        with _local(http_port=upstream.port) as local:
            status, _, _, _ = _get(local, headers={"Referer": "https://evil.example/page"})
        assert status == 403

    def test_a_local_referer_is_allowed(self, upstream):
        with _local(http_port=upstream.port) as local:
            status, _, _, _ = _get(local, headers={"Referer": "http://localhost:8087/"})
        assert status == 200

    def test_listens_on_loopback_only(self, upstream):
        with _local(http_port=upstream.port) as local:
            assert local._server.config.host == "127.0.0.1"


class TestWebSocket:
    def test_the_upgrade_is_signed_and_messages_relay_both_ways(self):
        server = FakeWebSocketUpstream()
        try:
            with _local(ws_port=server.port) as local:
                url = f"ws://localhost:{local.listen_port}/live"
                with connect(url, origin=f"http://localhost:{local.listen_port}") as ws:
                    ws.send("hello")
                    assert ws.recv(timeout=5) == "echo:hello"
                    ws.send(b"\x00\x01")
                    assert ws.recv(timeout=5) == b"echo:\x00\x01"
        finally:
            server.stop()
        handshake = server.handshakes[0]
        assert handshake["Authorization"].startswith("AWS4-HMAC-SHA256")
        assert handshake["Origin"] == f"https://{TARGET_HOST}"
        assert len(handshake.get_all("Host")) == 1

    def test_the_subprotocol_is_negotiated_end_to_end(self):
        server = FakeWebSocketUpstream(subprotocols=["vite-hmr"])
        try:
            with _local(ws_port=server.port) as local:
                with connect(f"ws://localhost:{local.listen_port}/", subprotocols=["vite-hmr"]) as ws:
                    assert ws.subprotocol == "vite-hmr"
        finally:
            server.stop()

    def test_a_websocket_from_another_origin_is_refused(self):
        server = FakeWebSocketUpstream()
        try:
            with _local(ws_port=server.port) as local:
                with pytest.raises(InvalidStatus) as refused:
                    connect(f"ws://localhost:{local.listen_port}/live", origin="https://evil.example")
        finally:
            server.stop()
        assert refused.value.response.status_code == 403
        assert server.handshakes == []

    def test_a_refused_upstream_handshake_closes_the_client(self):
        server = FakeWebSocketUpstream(refuse=True)
        try:
            with _local(ws_port=server.port) as local:
                with pytest.raises(InvalidStatus):
                    connect(f"ws://localhost:{local.listen_port}/live")
        finally:
            server.stop()


class TestClientMethod:
    def test_the_endpoint_client_opens_a_local_endpoint(self):
        client = AgentCoreRuntimeClient(region="us-west-2", session=_endpoint().session)
        local = client.session_endpoint_client(f"https://{SESSION_HOST}").open_local_endpoint(port=8090, local_port=0)
        assert isinstance(local, LocalEndpoint)
        assert local.target_host == TARGET_HOST

    def test_a_presigned_url_is_refused(self):
        with pytest.raises(ValueError, match="presigned"):
            _endpoint(PresignedAuth())

    def test_sigv4_sign_hashes_a_body_when_no_payload_header_is_given(self):
        client = AgentCoreRuntimeClient(region="us-west-2", session=_endpoint().session)
        with_body = client._sigv4_sign("https://example.on.aws/x", method="POST", body=b"abc")
        without = client._sigv4_sign("https://example.on.aws/x", method="POST")
        assert with_body["Authorization"] != without["Authorization"]
        assert hashlib.sha256(b"").hexdigest() == EMPTY_PAYLOAD_SHA256


class TestEndpointClientAuth:
    def test_a_bearer_token_endpoint_signs_with_the_token(self):
        url, headers = _endpoint(OAuthAuth(bearer_token="jwt-value")).sign("GET", "/status", port=8081)
        assert url == f"https://8081--{SESSION_HOST}/status"
        assert headers["Authorization"] == "Bearer jwt-value"
        assert "X-Amz-Date" not in headers
