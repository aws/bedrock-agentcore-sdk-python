"""Signed HTTP access to a runtime session's session URL.

A runtime created with ``allowedPorts`` returns a session URL in the
``X-Amzn-Bedrock-AgentCore-Runtime-Session-Endpoint-Url`` response header of
``InvokeAgentRuntime`` and the other operations that start or reuse a session. SDKs
that model the header expose it as ``sessionEndpointUrl``. Requests to that URL reach
any path, with any HTTP method, on the allowed ports, so the container serves the
routes it was written with rather than one entry point.

The URL is not a public link. Every HTTP request and WebSocket handshake carries SigV4
or a bearer token. For session URL requests the canonical SigV4 payload is always the
SHA-256 hash of an empty payload, including when the request carries a body, so this
module sends that value in ``X-Amz-Content-SHA256`` and signs for it. A client that
signs over the hash of the body instead is refused, and nothing in the refusal says why.

Example:
    client = AgentCoreRuntimeClient(region="us-west-2")
    response = client.invoke_agent_runtime(
        agent_runtime_arn=runtime_arn,
        runtime_session_id=session_id,
        payload=json.dumps({"prompt": "hello"}).encode(),
    )
    endpoint = client.session_endpoint_client(session_endpoint_url(response))

    orders = endpoint.request("GET", "/orders", port=8090)
    print(orders.status, orders.json())

    url, headers = endpoint.sign_websocket("/live", port=8090)

    with endpoint.open_local_endpoint(port=8090) as local:
        print("Open", local.url)
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple
from urllib.parse import urlparse

import boto3
import urllib3

from .._utils.endpoints import SESSION_ENDPOINT_DOMAINS, InvalidRegionError, validate_aws_hostname, validate_region
from .._utils.user_agent import build_user_agent_suffix
from .auth import OAuthAuth

if TYPE_CHECKING:
    from .auth import AuthMode
    from .local_endpoint import LocalEndpoint

SIGNING_NAME = "bedrock-agentcore"

#: The canonical payload hash for every session URL request, with or without a body.
EMPTY_PAYLOAD_SHA256 = hashlib.sha256(b"").hexdigest()

#: The proxy reports its own errors here, with an empty response body. An error from
#: the proxy and the same status from the application are otherwise hard to tell apart.
PROXY_ERROR_HEADER = "x-amzn-bedrock-agentcore-runtime-session-proxy-error"

# Some service deployments may send refusals under this legacy name while the
# documentation names PROXY_ERROR_HEADER. Read both for compatibility.
_PROXY_ERROR_HEADERS = (PROXY_ERROR_HEADER, "x-amzn-bedrock-agentcore-runtime-error")

#: The response header that carries the session URL. SDKs that model it expose the
#: value as ``sessionEndpointUrl``; ``session_endpoint_url`` reads either one.
SESSION_ENDPOINT_URL_HEADER = "x-amzn-bedrock-agentcore-runtime-session-endpoint-url"

#: Send this header with the value ``h2c`` to reach an HTTP/2 or gRPC server in the
#: session. The connection to the session URL is TLS either way.
TARGET_TRANSPORT_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Target-Transport"

_PAYLOAD_HASH_HEADER = "X-Amz-Content-SHA256"

_DEFAULT_TIMEOUT = 60

logger = logging.getLogger(__name__)


class InvalidSessionEndpointError(ValueError):
    """Raised when a session URL does not resolve to an AWS host.

    Requests to the URL are signed with the caller's credentials, so a URL pointing
    somewhere else would hand a signature to that host. The URL normally comes straight
    from an InvokeAgentRuntime response, and this check is here so that a misconfigured
    endpoint override cannot turn into a credential leak.
    """


def session_endpoint_url(response: Mapping[str, Any]) -> Optional[str]:
    """Return the session URL from a Runtime operation response, or None.

    Reads the modeled ``sessionEndpointUrl`` member when botocore provides it, and the
    raw response header otherwise. The header is omitted when the runtime has no
    ``allowedPorts`` or a session URL is not available, so None is a normal answer.

    Args:
        response: The response of InvokeAgentRuntime, InvokeAgentRuntimeCommand,
            GetAgentCard, or another operation that returns the header.

    Returns:
        The session URL, or None when the response does not carry one.
    """
    modeled = response.get("sessionEndpointUrl")
    if modeled:
        return modeled
    headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
    for key, value in headers.items():
        if key.lower() == SESSION_ENDPOINT_URL_HEADER:
            return value
    return None


@dataclass
class SessionEndpointResponse:
    """One response from a session URL request.

    Attributes:
        status: The HTTP status code.
        headers: The response headers.
        data: The response body as bytes, so images, archives, and protobuf arrive intact.
            Use ``text`` or ``json()`` for text responses.
    """

    status: int
    headers: Dict[str, str] = field(default_factory=dict)
    data: bytes = b""

    @property
    def proxy_error(self) -> Optional[str]:
        """The proxy's own error code, or None when the application answered.

        Platform errors arrive with an HTTP status, an empty body, and one of these
        codes: ``MalformedRequest`` or ``ProtocolError`` (400), ``AuthenticationRequired``
        (401), ``AccessDenied`` (403, including a port outside ``allowedPorts``),
        ``SessionNotFound`` (404), ``SessionConflict`` (409), ``Throttled`` (429),
        ``InternalError`` (500), ``PortNotReachable`` (502, nothing listening on an
        allowed port), ``ServiceUnavailable`` (503) and ``RuntimeTimeout`` (504).
        """
        for key, value in self.headers.items():
            if key.lower() in _PROXY_ERROR_HEADERS:
                return value
        return None

    @property
    def reached_container(self) -> bool:
        """True when the application answered, whatever status code it chose."""
        return self.proxy_error is None

    @property
    def text(self) -> str:
        """Decode the body as UTF-8 text (replacement character on invalid bytes)."""
        return self.data.decode("utf-8", errors="replace")

    def json(self) -> Any:
        """Parse the body as JSON.

        Raises:
            json.JSONDecodeError: If the body is not valid JSON.
        """
        return json.loads(self.data)


class SessionEndpointClient:
    """Sends authenticated requests of any method to any path inside one session.

    Attributes:
        url: The session URL this object sends to.
        region: The Region used for signing.
    """

    def __init__(
        self,
        url: str,
        region: Optional[str] = None,
        session: Optional[boto3.Session] = None,
        integration_source: Optional[str] = None,
        http: Optional[urllib3.PoolManager] = None,
        client: Optional[Any] = None,
        auth: "AuthMode" = "sigv4",
    ) -> None:
        """Wrap a session URL.

        Args:
            url: The session URL returned by InvokeAgentRuntime. See
                ``session_endpoint_url``.
            region: Region to sign with. Defaults to the session's Region, then
                "us-west-2".
            session: Optional boto3 Session supplying credentials.
            integration_source: Optional integration source for user-agent telemetry.
            http: Optional urllib3 PoolManager. When passed, the caller owns it and
                ``close()`` leaves it open. By default the client creates one with
                retries turned off, because a request can change state in the
                application and is not always safe to replay.
            client: The ``AgentCoreRuntimeClient`` whose signing this uses. Created from
                ``region`` and ``session`` when omitted.
            auth: The same ``AuthMode`` as ``open_shell``: ``"sigv4"`` (default) to sign with
                SigV4, or ``OAuthAuth(bearer_token=...)`` for a runtime with a JWT authorizer.

        Raises:
            InvalidSessionEndpointError: If the URL does not resolve to an AWS host.
            ValueError: For ``PresignedAuth``, which session URLs do not accept.
        """
        from .auth import PresignedAuth

        if isinstance(auth, PresignedAuth):
            raise ValueError("Session URLs take SigV4 headers or a bearer token, not a presigned URL")
        self.auth = auth
        self.url = url.rstrip("/")
        self.host = _validate_session_url(self.url)
        self.session = session if session else boto3.Session()
        self.region = validate_region(region or self.session.region_name or "us-west-2")
        self.integration_source = integration_source
        self._owns_http = http is None
        # One signed request per call, so retries and redirects are left to the caller:
        # the signature is short-lived and a replayed POST is not always safe.
        self._http = http if http else urllib3.PoolManager(retries=urllib3.Retry(total=0, redirect=0))
        if client is None:
            from .agent_core_runtime_client import AgentCoreRuntimeClient

            client = AgentCoreRuntimeClient(
                region=self.region, session=self.session, integration_source=integration_source
            )
        self._client = client

    def sign(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        port: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """Return ``(url, headers)`` for one signed request, for any HTTP library.

        This is the same contract as ``connect_shell``: the SDK does the signing
        and the caller keeps its own HTTP client. Use ``request`` when a response
        is all that is needed. Send ``body`` exactly as passed here; it is not part of
        the signature, because the canonical payload is the empty payload.

        Args:
            method: Any HTTP method the application's routes accept.
            path: Path to request, starting with "/". A query string is allowed.
            body: Optional request body. Only used to set a default Content-Type.
            port: A port from the runtime's ``allowedPorts``. Omit it to reach the
                serving port, which then has to be in ``allowedPorts`` as well. The
                connection stays on TLS 443 either way, so the target port travels as
                a hostname prefix.
            headers: Extra headers to include in the signature, for example
                ``TARGET_TRANSPORT_HEADER``.

        Returns:
            The request URL and the headers to send with it, including
            ``X-Amz-Content-SHA256``.

        Raises:
            ValueError: If ``path`` does not start with "/", ``port`` is out of range,
                or ``headers`` carries a payload hash other than the empty payload's.
            RuntimeError: If no AWS credentials are found.
        """
        if not path.startswith("/"):
            raise ValueError(f"path must start with '/': {path!r}")
        _check_port(port)
        url = f"https://{self._host_for(port)}{path}"
        signed_headers = self._headers_to_sign(headers, body)
        signed = self._client._url_auth_headers(
            url,
            auth=self.auth,
            method=method,
            signed_headers=signed_headers,
            unsigned_headers={"User-Agent": build_user_agent_suffix(self.integration_source)},
        )
        return url, signed

    def sign_websocket(
        self,
        path: str,
        port: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """Return ``(wss_url, headers)`` for one signed WebSocket handshake.

        Pass both to a WebSocket client that can set headers on its HTTP/1.1 upgrade
        request, for example ``websockets.connect(url, additional_headers=headers)``.
        Each handshake needs its own signature, so call this again to reconnect.

        Args:
            path: The application's WebSocket path, starting with "/".
            port: A port from the runtime's ``allowedPorts``. Omit it for the serving port.
            headers: Extra headers to include in the signature.

        Returns:
            The ``wss://`` URL and the headers to send with the upgrade request.
        """
        if not path.startswith("/"):
            raise ValueError(f"path must start with '/': {path!r}")
        _check_port(port)
        url = f"https://{self._host_for(port)}{path}"
        signed = self._client._websocket_auth_headers(
            url,
            auth=self.auth,
            signed_headers=self._headers_to_sign(headers, None),
            unsigned_headers={"User-Agent": build_user_agent_suffix(self.integration_source)},
        )
        return "wss://" + url[len("https://") :], signed

    def _host_for(self, port: Optional[int]) -> str:
        return f"{port}--{self.host}" if port is not None else self.host

    def _headers_to_sign(self, headers: Optional[Dict[str, str]], body: Optional[bytes]) -> Dict[str, str]:
        """The caller's headers plus the empty payload hash that session URLs are signed over."""
        signed_headers: Dict[str, str] = {}
        for name, value in (headers or {}).items():
            if name.lower() == _PAYLOAD_HASH_HEADER.lower():
                if value != EMPTY_PAYLOAD_SHA256:
                    raise ValueError(
                        "Session URL requests are signed over the empty payload hash; "
                        f"{_PAYLOAD_HASH_HEADER} must be {EMPTY_PAYLOAD_SHA256}"
                    )
                continue
            signed_headers[name] = value
        if body is not None and not any(name.lower() == "content-type" for name in signed_headers):
            signed_headers["Content-Type"] = "application/json"
        if not isinstance(self.auth, OAuthAuth):
            # botocore uses this header as the canonical payload hash when it is present,
            # rather than hashing the body, so the signature matches what Runtime checks.
            signed_headers[_PAYLOAD_HASH_HEADER] = EMPTY_PAYLOAD_SHA256
        return signed_headers

    def open_local_endpoint(self, port: Optional[int] = None, local_port: int = 8088) -> "LocalEndpoint":
        """Return a local endpoint on this machine that forwards to one port of this session.

        A browser cannot add a SigV4 signature or a bearer token to the requests a page
        makes on its own. The local endpoint listens on ``127.0.0.1``, adds this client's
        credential to every HTTP request and WebSocket handshake it receives, and relays
        the request and its response. Start it with ``start()`` or a ``with`` block, then
        open ``local.url``.

        Args:
            port: A port from the runtime's ``allowedPorts``. Omit for the serving port.
            local_port: The port to listen on. Pass 0 to pick a free one.

        Returns:
            A ``LocalEndpoint``, not yet started.

        Example:
            with endpoint.open_local_endpoint(port=8090) as local:
                print("Open", local.url)
        """
        from .local_endpoint import LocalEndpoint

        return LocalEndpoint(self, port=port, listen_port=local_port)

    def request(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        port: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> SessionEndpointResponse:
        """Send one signed HTTP/1.1 request and return the response.

        Redirects are returned rather than followed, because a redirect destination
        needs a signature of its own.

        Args:
            method: Any HTTP method the application's routes accept.
            path: Path to request, starting with "/". A query string is allowed.
            body: Optional request body. A ``str`` is encoded as UTF-8.
            port: A port from the runtime's ``allowedPorts``.
            headers: Extra headers to include in the signature.
            timeout: Seconds to wait for the response. A request can wait while
                Runtime prepares the session's compute, so allow for startup.

        Returns:
            A ``SessionEndpointResponse``. Check ``proxy_error`` before reading the
            body, because an error from the runtime carries no body.
        """
        if isinstance(body, str):
            body = body.encode("utf-8")
        url, signed = self.sign(method, path, body=body, port=port, headers=headers)
        # Send an explicit length even with no body. Without one, the request has been seen
        # to reach the application as chunked with a repeated final chunk, which Node
        # servers such as Vite answer with 400.
        signed.setdefault("Content-Length", str(len(body or b"")))
        reply = self._http.request(method.upper(), url, body=body, headers=signed, timeout=timeout, redirect=False)
        response = SessionEndpointResponse(status=reply.status, headers=dict(reply.headers), data=reply.data)
        if response.proxy_error:
            logger.debug("Session URL request %s %s answered by the runtime: %s", method, path, response.proxy_error)
        return response

    def close(self) -> None:
        """Close the connection pool, unless it was passed in by the caller."""
        if self._owns_http:
            self._http.clear()

    def __enter__(self) -> "SessionEndpointClient":
        """Use the client in a with block that closes its connection pool."""
        return self

    def __exit__(self, *_exc) -> None:
        """Close the connection pool at the end of a with block."""
        self.close()


def _check_port(port: Optional[int]) -> None:
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"port must be between 1 and 65535: {port!r}")


def _validate_session_url(url: str) -> str:
    """Return the hostname of a session URL, or raise."""
    if urlparse(url).scheme != "https":
        raise InvalidSessionEndpointError(f"Session endpoint URL must use https: {url!r}")
    try:
        return validate_aws_hostname(url, SESSION_ENDPOINT_DOMAINS)
    except InvalidRegionError as error:
        raise InvalidSessionEndpointError(f"Session endpoint URL resolves to a non-AWS host: {url!r}") from error
