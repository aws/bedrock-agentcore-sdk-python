"""A local endpoint that lets a browser, or any client that cannot sign, use a session URL.

Every request to a session URL needs SigV4 or a bearer token, and a browser can add
neither to the requests a page makes on its own: scripts, stylesheets, images, and the
WebSocket a development server opens for hot reload. ``LocalEndpoint`` listens on
your machine, adds the credential to each request it receives, and relays the request
and its response, so ``http://localhost:8088`` behaves like the application inside the
session.

Example:
    endpoint = client.session_endpoint_client(session_endpoint_url(response))
    with endpoint.open_local_endpoint(port=8090) as local:
        print("open", local.url)
        input("press Enter to stop")

Or from a shell:
    python -m bedrock_agentcore.runtime.local_endpoint --session-url URL --port 8090

The local endpoint acts with your credentials, so it accepts connections from this machine
only, answers only requests addressed to its own local address, and refuses requests and
WebSocket connections that a page on another site starts through the browser. Run it for
as long as you need the preview and no longer.

It is an ASGI application served by Uvicorn, so request parsing, header limits, and
connection handling come from Uvicorn rather than from code here.
"""

import argparse
import asyncio
import logging
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

import urllib3
import uvicorn
import websockets
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect as websocket_connect

from .auth import OAuthAuth
from .session_endpoint import SessionEndpointClient

logger = logging.getLogger(__name__)

_CHUNK = 64 * 1024
#: The largest request body relayed. Larger requests are refused with 413.
MAX_BODY_BYTES = 100 * 1024 * 1024
_CONNECT_TIMEOUT = 60

# Request headers that describe one hop, that the signature replaces, or that the
# WebSocket client writes itself.
_DROPPED_REQUEST = {
    "host",
    "connection",
    "keep-alive",
    "proxy-connection",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "authorization",
    "x-amz-date",
    "x-amz-security-token",
    "x-amz-content-sha256",
}
_DROPPED_WEBSOCKET = _DROPPED_REQUEST | {
    "sec-websocket-key",
    "sec-websocket-version",
    "sec-websocket-extensions",
    "sec-websocket-protocol",
}
_DROPPED_RESPONSE = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer", "upgrade"}

# Sec-Fetch-Site values a browser sends for requests this machine's own pages start:
# typing the URL (none), the page itself (same-origin), or another localhost port
# such as a preview page that frames the site (same-site).
_LOCAL_FETCH_SITES = {"none", "same-origin", "same-site"}
_LOCAL_HOSTNAMES = {"localhost", "127.0.0.1"}

# Uvicorn 0.35 and later run WebSockets on the current websockets API; earlier
# versions only have the legacy implementation.
_WS_PROTOCOL = "websockets-sansio" if "websockets-sansio" in uvicorn.config.WS_PROTOCOLS else "websockets"


class LocalEndpoint:
    """Relays HTTP requests and WebSocket connections from localhost to one session port.

    Attributes:
        url: The local address to open, for example ``http://localhost:8088``.
    """

    def __init__(
        self,
        endpoint: SessionEndpointClient,
        port: Optional[int] = None,
        listen_port: int = 8088,
    ) -> None:
        """Prepare a local endpoint for one port of one session.

        Args:
            endpoint: The session's ``SessionEndpointClient``, whose credential is added
                to every request: SigV4, or the bearer token it was created with.
            port: A port from the runtime's ``allowedPorts``. Omit it for the serving port.
            listen_port: The local port to listen on. Pass 0 to pick a free one.
        """
        self.endpoint = endpoint
        self.port = port
        self.listen_port = listen_port
        self.target_host = f"{port}--{endpoint.host}" if port is not None else endpoint.host
        # Where relayed requests go. Tests point these at a local server; the Host header
        # and the signature always name the session host.
        self._http_base = f"https://{self.target_host}"
        self._ws_base = f"wss://{self.target_host}"
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        """The local address a browser or client should open."""
        return f"http://localhost:{self.listen_port}"

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> "LocalEndpoint":
        """Start listening in a background thread and return once the port is bound."""
        if self._thread and self._thread.is_alive():
            return self
        config = uvicorn.Config(
            self,
            host="127.0.0.1",
            port=self.listen_port,
            lifespan="off",
            ws=_WS_PROTOCOL,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="session-local-endpoint", daemon=True)
        self._thread.start()
        for _ in range(200):
            if self._server.started or not self._thread.is_alive():
                break
            self._thread.join(timeout=0.05)
        if not self._server.started:
            raise RuntimeError(f"Could not listen on 127.0.0.1:{self.listen_port}")
        self.listen_port = self._server.servers[0].sockets[0].getsockname()[1]
        return self

    def stop(self) -> None:
        """Stop listening and close open connections."""
        if self._server:
            self._server.should_exit = True
            self._server.force_exit = True
        if self._thread:
            self._thread.join(timeout=5)

    def serve_forever(self) -> None:
        """Listen in the calling thread until interrupted."""
        self.start()
        try:
            while self._thread and self._thread.is_alive():
                self._thread.join(timeout=1)
        except KeyboardInterrupt:
            self.stop()

    def __enter__(self) -> "LocalEndpoint":
        """Start the local endpoint for the duration of a with block."""
        return self.start()

    def __exit__(self, *_exc) -> None:
        """Stop the local endpoint at the end of a with block."""
        self.stop()

    # -- ASGI ------------------------------------------------------------------

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve one HTTP request or WebSocket connection."""
        if scope["type"] == "http":
            response = await self._relay_http(Request(scope, receive))
            await response(scope, receive, send)
        elif scope["type"] == "websocket":
            await self._relay_websocket(WebSocket(scope, receive, send))

    def _check_local(self, headers: Any) -> Optional[str]:
        """Refuse anything a page from another site could have sent through the browser."""
        local = {f"localhost:{self.listen_port}", f"127.0.0.1:{self.listen_port}"}
        if headers.get("host", "").lower() not in local:
            # A page on another site can reach this port through a host name that
            # resolves to 127.0.0.1 (DNS rebinding); its requests carry that name.
            return "the Host header must be this endpoint's local address"
        origin = headers.get("origin")
        if origin and origin.lower() not in {f"http://{address}" for address in local}:
            return "requests from other origins are refused"
        fetch_site = headers.get("sec-fetch-site")
        if fetch_site and fetch_site.lower() not in _LOCAL_FETCH_SITES:
            # Navigations and image loads from another site carry no Origin, but the
            # browser marks them cross-site.
            return "requests started by another site are refused"
        if not origin and not fetch_site:
            referer = headers.get("referer")
            if referer and urlparse(referer).hostname not in _LOCAL_HOSTNAMES:
                return "requests started by another site are refused"
        return None

    def _target(self, scope: Scope) -> str:
        path = scope.get("raw_path") or scope["path"].encode()
        query = scope.get("query_string") or b""
        return (path + (b"?" + query if query else b"")).decode("latin-1")

    def _forwarded_headers(self, headers: Any, dropped: set) -> List[Tuple[str, str]]:
        forwarded = []
        for name, value in headers.items():
            lowered = name.lower()
            if lowered in dropped:
                continue
            if lowered == "origin":
                # The application sees requests as coming from its own address.
                value = f"https://{self.target_host}"
            forwarded.append((name, value))
        return forwarded

    # -- HTTP ------------------------------------------------------------------

    async def _relay_http(self, request: Request) -> Response:
        refusal = self._check_local(request.headers)
        if refusal:
            return PlainTextResponse(refusal, status_code=403)
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BODY_BYTES:
            return PlainTextResponse("request body too large", status_code=413)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY_BYTES:
                return PlainTextResponse("request body too large", status_code=413)

        target = self._target(request.scope)
        _, signed = self.endpoint.sign(request.method, target, port=self.port)
        headers = {name: value for name, value in signed.items() if name.lower() != "user-agent"}
        headers["Host"] = self.target_host
        for name, value in self._forwarded_headers(request.headers, _DROPPED_REQUEST):
            headers[name] = value
        # Always send a length. Without one, a bodyless request has been seen to reach
        # the application as chunked with a repeated final chunk, which Node servers
        # such as Vite answer with 400.
        headers["Content-Length"] = str(len(body))

        try:
            reply = await run_in_threadpool(
                self.endpoint._http.request,
                request.method,
                self._http_base + target,
                body=bytes(body),
                headers=headers,
                preload_content=False,
                decode_content=False,
                redirect=False,
                retries=False,
                timeout=urllib3.Timeout(connect=_CONNECT_TIMEOUT, read=None),
            )
        except urllib3.exceptions.HTTPError as error:
            return PlainTextResponse(f"could not reach the session URL: {error}", status_code=502)

        response = StreamingResponse(iterate_in_threadpool(_stream(reply)), status_code=reply.status)
        response.raw_headers = [
            (name.lower().encode("latin-1"), value.encode("latin-1"))
            for name, value in reply.headers.iteritems()
            if name.lower() not in _DROPPED_RESPONSE
        ]
        return response

    # -- WebSocket -------------------------------------------------------------

    async def _relay_websocket(self, client: WebSocket) -> None:
        refusal = self._check_local(client.headers)
        if refusal:
            await client.close(code=1008, reason=refusal)  # before accept: answered with 403
            return
        target = self._target(client.scope)
        _, signed = self.endpoint.sign_websocket(target, port=self.port)
        headers: Dict[str, str] = dict(self._forwarded_headers(client.headers, _DROPPED_WEBSOCKET))
        headers.update({name: value for name, value in signed.items() if name.lower() != "user-agent"})
        subprotocols = client.scope.get("subprotocols") or None
        try:
            upstream = await websocket_connect(
                self._ws_base + target,
                additional_headers=headers,
                subprotocols=subprotocols,
                open_timeout=_CONNECT_TIMEOUT,
                max_size=None,
            )
        except (OSError, websockets.exceptions.WebSocketException, asyncio.TimeoutError) as error:
            logger.debug("WebSocket upgrade to %s refused: %s", target, error)
            await client.close(code=1011)
            return

        await client.accept(subprotocol=upstream.subprotocol)
        try:
            await _relay_frames(client, upstream)
        finally:
            await upstream.close()


async def _relay_frames(client: WebSocket, upstream: Any) -> None:
    """Copy messages both ways until either side closes."""

    async def from_client() -> None:
        try:
            while True:
                message = await client.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    await upstream.send(message["bytes"])
                elif message.get("text") is not None:
                    await upstream.send(message["text"])
        except (WebSocketDisconnect, websockets.exceptions.ConnectionClosed):
            return

    async def from_upstream() -> None:
        try:
            async for message in upstream:
                if isinstance(message, bytes):
                    await client.send_bytes(message)
                else:
                    await client.send_text(message)
            await client.close(code=upstream.close_code or 1000)
        except (WebSocketDisconnect, websockets.exceptions.ConnectionClosed, RuntimeError):
            return

    tasks = [asyncio.ensure_future(from_client()), asyncio.ensure_future(from_upstream())]
    _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


def _stream(reply: Any) -> Iterator[bytes]:
    try:
        yield from reply.stream(_CHUNK, decode_content=False)
    finally:
        reply.release_conn()


def main(argv: Optional[List[str]] = None) -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session-url", required=True, help="the session URL from InvokeAgentRuntime")
    parser.add_argument("--port", type=int, help="a port from allowedPorts; omit for the serving port")
    parser.add_argument("--listen-port", type=int, default=8088)
    parser.add_argument("--region", help="the runtime's Region, if not the default Region")
    parser.add_argument("--bearer-token-file", help="a file holding a JWT, for a JWT authorized runtime")
    args = parser.parse_args(argv)

    token = None
    if args.bearer_token_file:
        with open(args.bearer_token_file, encoding="utf-8") as handle:
            token = handle.read().strip()
    endpoint = SessionEndpointClient(args.session_url, region=args.region, auth=OAuthAuth(token) if token else "sigv4")
    local = endpoint.open_local_endpoint(port=args.port, local_port=args.listen_port)
    print(f"Open {local.url} (session port {args.port or 'serving'}). Ctrl+C stops it.")
    local.serve_forever()


if __name__ == "__main__":
    main()
