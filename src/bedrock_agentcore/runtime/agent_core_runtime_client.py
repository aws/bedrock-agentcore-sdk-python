"""Client for AgentCore Runtime authentication and invocation.

This module provides a single client for interacting with Bedrock AgentCore
Runtime endpoints:

- WebSocket URL and header generation (SigV4, SigV4 presigned, OAuth bearer).
- HTTP invocation (blocking, sync streaming, async streaming) with bearer-token
  auth.
- ``InvokeAgentRuntimeCommand`` shell-exec support (blocking and streaming).
- Session termination.

Bearer-token HTTP methods mirror the shape of
:meth:`generate_ws_connection_oauth` — ``runtime_arn`` and ``bearer_token``
are passed per call. The ``urllib3.PoolManager`` used by HTTP methods is
lazy-initialized; callers who only use URL-generation methods pay no cost.
"""

import asyncio
import base64
import datetime
import json
import logging
import secrets
import threading
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    Iterator,
    Optional,
    Tuple,
)
from urllib.parse import quote, urlencode, urlparse

import boto3
from botocore.auth import SigV4Auth, SigV4QueryAuth
from botocore.awsrequest import AWSRequest
from botocore.eventstream import EventStreamBuffer

from .._utils.endpoints import get_data_plane_endpoint
from .utils import is_valid_partition

if TYPE_CHECKING:
    # urllib3 v2 returns ``BaseHTTPResponse`` from ``PoolManager.request``.
    # On urllib3 v1.26 the returned object is ``HTTPResponse`` which is
    # structurally compatible with the same attributes (``status``, ``headers``,
    # ``read``, ``stream``, ``release_conn``), so the annotation is purely for
    # type-checking under v2.
    import urllib3
    from urllib3 import BaseHTTPResponse as _UrllibResponse

DEFAULT_PRESIGNED_URL_TIMEOUT = 300
MAX_PRESIGNED_URL_TIMEOUT = 300
DEFAULT_HTTP_TIMEOUT = 300
DEFAULT_COMMAND_TIMEOUT = 600


class AgentRuntimeError(Exception):
    """Raised when an AgentCore runtime returns an error response.

    Used by the HTTP invocation methods of :class:`AgentCoreRuntimeClient`
    for both non-2xx HTTP responses and in-band error events embedded in
    SSE streams.

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


class AgentCoreRuntimeClient:
    """Client for AgentCore Runtime authentication and invocation.

    This client supports four auth/transport modes against AgentCore Runtime:

    - SigV4-signed WebSocket URL + headers (:meth:`generate_ws_connection`).
    - SigV4 presigned WebSocket URL (:meth:`generate_presigned_url`).
    - OAuth bearer-token WebSocket URL + headers
      (:meth:`generate_ws_connection_oauth`).
    - OAuth bearer-token HTTP invocation (:meth:`invoke`,
      :meth:`invoke_streaming`, :meth:`invoke_streaming_async`,
      :meth:`execute_command`, :meth:`execute_command_streaming`,
      :meth:`stop_runtime_session`).

    The ``urllib3.PoolManager`` used by the HTTP methods is lazy-initialized
    on first access; callers who only use URL-generation methods pay no cost.

    Attributes:
        region (str): The AWS region being used.
        session (boto3.Session): The boto3 session for AWS credentials.
    """

    def __init__(self, region: str, session: Optional[boto3.Session] = None) -> None:
        """Initialize an AgentCoreRuntime client for the specified AWS region.

        Args:
            region (str): The AWS region to use for the AgentCore Runtime service.
            session (Optional[boto3.Session]): Optional boto3 session. If not provided,
                a new session will be created using default credentials.
        """
        from .._utils.endpoints import validate_region

        validate_region(region)
        self.region = region
        self.logger = logging.getLogger(__name__)

        if session is None:
            session = boto3.Session()

        self.session = session
        self._pool_manager: Optional["urllib3.PoolManager"] = None

    @property
    def _http(self) -> "urllib3.PoolManager":
        """Return a lazy-initialized ``urllib3.PoolManager`` shared across HTTP calls.

        The pool is not created until an HTTP method (``invoke``,
        ``execute_command``, etc.) is actually called. Callers that only use
        the SigV4 / OAuth URL-generation methods never trigger this import.

        Returns:
            A shared :class:`urllib3.PoolManager` instance.
        """
        if self._pool_manager is None:
            import urllib3

            self._pool_manager = urllib3.PoolManager()
        return self._pool_manager

    def _parse_runtime_arn(self, runtime_arn: str) -> Dict[str, str]:
        """Parse runtime ARN and extract components.

        Args:
            runtime_arn (str): Full runtime ARN

        Returns:
            Dict[str, str]: Dictionary with region, account_id, runtime_id

        Raises:
            ValueError: If ARN format is invalid
        """
        # Expected format: arn:aws:bedrock-agentcore:{region}:{account}:runtime/{runtime_id}
        parts = runtime_arn.split(":")

        if len(parts) != 6:
            raise ValueError(f"Invalid runtime ARN format: {runtime_arn}")

        if (
            parts[0] != "arn"
            or not is_valid_partition(parts[1])
            or parts[2] != "bedrock-agentcore"
        ):
            raise ValueError(f"Invalid runtime ARN format: {runtime_arn}")

        # Parse the resource part (runtime/{runtime_id})
        resource = parts[5]
        if not resource.startswith("runtime/"):
            raise ValueError(f"Invalid runtime ARN format: {runtime_arn}")

        runtime_id = resource.split("/", 1)[1]

        # Validate that components are not empty
        region = parts[3]
        account_id = parts[4]

        if not region or not account_id or not runtime_id:
            raise ValueError("ARN components cannot be empty")

        return {
            "region": region,
            "account_id": account_id,
            "runtime_id": runtime_id,
        }

    def _build_websocket_url(
        self,
        runtime_arn: str,
        endpoint_name: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """Build WebSocket URL with query parameters.

        Args:
            runtime_arn (str): Full runtime ARN
            endpoint_name (Optional[str]): Optional endpoint name for qualifier param
            custom_headers (Optional[Dict[str, str]]): Optional custom query parameters

        Returns:
            str: WebSocket URL with query parameters
        """
        # Get the data plane endpoint
        host = get_data_plane_endpoint(self.region).replace("https://", "")

        # URL-encode the runtime ARN
        encoded_arn = quote(runtime_arn, safe="")

        # Build base path
        path = f"/runtimes/{encoded_arn}/ws"

        # Build query parameters
        query_params = {}

        if endpoint_name:
            query_params["qualifier"] = endpoint_name

        if custom_headers:
            query_params.update(custom_headers)

        # Construct URL
        if query_params:
            query_string = urlencode(query_params)
            ws_url = f"wss://{host}{path}?{query_string}"
        else:
            ws_url = f"wss://{host}{path}"

        return ws_url

    def _build_http_url(
        self,
        runtime_arn: str,
        path_suffix: str,
        endpoint_name: Optional[str] = None,
    ) -> str:
        """Build an HTTPS URL for a data-plane API on a runtime.

        Reuses :meth:`_parse_runtime_arn` to extract the region from the ARN
        and :func:`get_data_plane_endpoint` to construct the host, matching
        the convention established by the WebSocket URL builder.

        Args:
            runtime_arn: Full runtime ARN.
            path_suffix: Path under ``/runtimes/{arn}/``. Must not start with
                ``/``. Examples: ``"invocations"``, ``"commands"``,
                ``"stopruntimesession"``.
            endpoint_name: Endpoint qualifier sent as the ``qualifier`` query
                parameter. Defaults to ``"DEFAULT"`` when not supplied so the
                API receives a valid qualifier value.

        Returns:
            Absolute URL including the ``?qualifier=...`` query string.

        Raises:
            ValueError: If the ARN format is invalid.
        """
        parsed = self._parse_runtime_arn(runtime_arn)
        base = get_data_plane_endpoint(parsed["region"])
        encoded_arn = quote(runtime_arn, safe="")
        query = urlencode({"qualifier": endpoint_name or "DEFAULT"})
        return f"{base}/runtimes/{encoded_arn}/{path_suffix}?{query}"

    def _build_bearer_headers(
        self,
        bearer_token: str,
        session_id: str,
        accept: str,
        content_type: str,
    ) -> Dict[str, str]:
        """Build request headers for a bearer-authenticated HTTP call.

        Args:
            bearer_token: OAuth/JWT bearer token.
            session_id: Runtime session id.
            accept: Value for the ``Accept`` header.
            content_type: Value for the ``Content-Type`` header.

        Returns:
            Header dict including ``Authorization``, ``Content-Type``,
            ``Accept``, and ``X-Amzn-Bedrock-AgentCore-Runtime-Session-Id``.
        """
        return {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": content_type,
            "Accept": accept,
            "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
        }

    @staticmethod
    def _serialize_body(body: Any, content_type: str) -> bytes:
        """Serialize a request body to bytes based on the content type.

        JSON content types serialize via :func:`json.dumps`. For other
        content types, ``bytes`` pass through unchanged, ``str`` is
        UTF-8-encoded, and anything else falls back to JSON serialization.

        Args:
            body: The payload to send.
            content_type: The MIME type of the request.

        Returns:
            UTF-8 encoded request body.
        """
        if "json" in content_type:
            return json.dumps(body).encode("utf-8")
        if isinstance(body, bytes):
            return body
        if isinstance(body, str):
            return body.encode("utf-8")
        return json.dumps(body).encode("utf-8")

    def generate_ws_connection(
        self,
        runtime_arn: str,
        session_id: Optional[str] = None,
        endpoint_name: Optional[str] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """Generate WebSocket URL and SigV4 signed headers for runtime connection.

        Args:
            runtime_arn (str): Full runtime ARN
                (e.g., 'arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime-abc')
            session_id (Optional[str]): Session ID to use. If None, auto-generates a UUID.
            endpoint_name (Optional[str]): Endpoint name to use as 'qualifier' query parameter.
                If provided, adds ?qualifier={endpoint_name} to the URL.

        Returns:
            Tuple[str, Dict[str, str]]: A tuple containing:
                - WebSocket URL (wss://...) with query parameters
                - Headers dictionary with SigV4 signature

        Raises:
            RuntimeError: If no AWS credentials are found.
            ValueError: If runtime_arn format is invalid.

        Example:
            >>> client = AgentCoreRuntimeClient('us-west-2')
            >>> ws_url, headers = client.generate_ws_connection(
            ...     runtime_arn='arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime',
            ...     endpoint_name='DEFAULT'
            ... )
        """
        self.logger.info("Generating WebSocket connection credentials...")

        # Validate ARN
        self._parse_runtime_arn(runtime_arn)

        # Auto-generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            self.logger.debug("Auto-generated session ID: %s", session_id)

        # Build WebSocket URL
        ws_url = self._build_websocket_url(runtime_arn, endpoint_name)

        # Get AWS credentials
        credentials = self.session.get_credentials()
        if not credentials:
            raise RuntimeError("No AWS credentials found")

        frozen_credentials = credentials.get_frozen_credentials()

        # Convert wss:// to https:// for signing
        https_url = ws_url.replace("wss://", "https://")
        parsed = urlparse(https_url)
        host = parsed.netloc

        # Create the request to sign
        request = AWSRequest(
            method="GET",
            url=https_url,
            headers={
                "host": host,
                "x-amz-date": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y%m%dT%H%M%SZ"
                ),
            },
        )

        # Sign the request with SigV4
        auth = SigV4Auth(frozen_credentials, "bedrock-agentcore", self.region)
        auth.add_auth(request)

        # Build headers for WebSocket connection
        headers = {
            "Host": host,
            "X-Amz-Date": request.headers["x-amz-date"],
            "Authorization": request.headers["Authorization"],
            "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
            "Upgrade": "websocket",
            "Connection": "Upgrade",
            "Sec-WebSocket-Version": "13",
            "Sec-WebSocket-Key": base64.b64encode(secrets.token_bytes(16)).decode(),
            "User-Agent": "AgentCoreRuntimeClient/1.0",
        }

        # Add session token if present
        if frozen_credentials.token:
            headers["X-Amz-Security-Token"] = frozen_credentials.token

        self.logger.info(
            "✓ WebSocket connection credentials generated (Session: %s)", session_id
        )
        return ws_url, headers

    def generate_presigned_url(
        self,
        runtime_arn: str,
        session_id: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        expires: int = DEFAULT_PRESIGNED_URL_TIMEOUT,
    ) -> str:
        """Generate a presigned WebSocket URL for runtime connection.

        Presigned URLs include authentication in query parameters, allowing
        frontend clients to connect without managing AWS credentials.

        Args:
            runtime_arn (str): Full runtime ARN
                (e.g., 'arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime-abc')
            session_id (Optional[str]): Session ID to use. If None, auto-generates a UUID.
            endpoint_name (Optional[str]): Endpoint name to use as 'qualifier' query parameter.
                If provided, adds ?qualifier={endpoint_name} to the URL before signing.
            custom_headers (Optional[Dict[str, str]]): Additional query parameters to include
                in the presigned URL before signing (e.g., {"abc": "pqr"}).
            expires (int): Seconds until URL expires (default: 300, max: 300).

        Returns:
            str: Presigned WebSocket URL with query string parameters including:
                - Original query params (qualifier, custom_headers)
                - SigV4 auth params (X-Amz-Algorithm, X-Amz-Credential, etc.)

        Raises:
            ValueError: If expires exceeds maximum (300 seconds).
            RuntimeError: If URL generation fails or no credentials found.

        Example:
            >>> client = AgentCoreRuntimeClient('us-west-2')
            >>> presigned_url = client.generate_presigned_url(
            ...     runtime_arn='arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime',
            ...     endpoint_name='DEFAULT',
            ...     custom_headers={'abc': 'pqr'},
            ...     expires=300
            ... )
        """
        self.logger.info("Generating presigned WebSocket URL...")

        # Validate expires parameter
        if expires > MAX_PRESIGNED_URL_TIMEOUT:
            raise ValueError(
                f"Expiry timeout cannot exceed {MAX_PRESIGNED_URL_TIMEOUT} seconds, got {expires}"
            )

        # Validate ARN
        self._parse_runtime_arn(runtime_arn)

        # Auto-generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            self.logger.debug("Auto-generated session ID: %s", session_id)

        # Add session_id to custom_headers (which become query params)
        if custom_headers is None:
            custom_headers = {}
        custom_headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] = session_id

        # Build WebSocket URL with query parameters
        ws_url = self._build_websocket_url(runtime_arn, endpoint_name, custom_headers)

        # Convert wss:// to https:// for signing
        https_url = ws_url.replace("wss://", "https://")

        # Parse URL
        url = urlparse(https_url)

        # Get AWS credentials
        credentials = self.session.get_credentials()
        if not credentials:
            raise RuntimeError("No AWS credentials found")

        frozen_credentials = credentials.get_frozen_credentials()

        # Create the request to sign
        request = AWSRequest(
            method="GET", url=https_url, headers={"host": url.hostname}
        )

        # Sign the request with SigV4QueryAuth
        signer = SigV4QueryAuth(
            credentials=frozen_credentials,
            service_name="bedrock-agentcore",
            region_name=self.region,
            expires=expires,
        )
        signer.add_auth(request)

        if not request.url:
            raise RuntimeError("Failed to generate presigned URL")

        # Convert back to wss:// for WebSocket connection
        presigned_url = request.url.replace("https://", "wss://")

        self.logger.info(
            "✓ Presigned URL generated (expires in %s seconds, Session: %s)",
            expires,
            session_id,
        )
        return presigned_url

    def generate_ws_connection_oauth(
        self,
        runtime_arn: str,
        bearer_token: str,
        session_id: Optional[str] = None,
        endpoint_name: Optional[str] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """Generate WebSocket URL and OAuth headers for runtime connection.

        This method uses OAuth bearer token authentication instead of AWS SigV4.
        Suitable for scenarios where OAuth tokens are used for authentication.

        Args:
            runtime_arn (str): Full runtime ARN
                (e.g., 'arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime-abc')
            bearer_token (str): OAuth bearer token for authentication.
            session_id (Optional[str]): Session ID to use. If None, auto-generates one.
            endpoint_name (Optional[str]): Endpoint name to use as 'qualifier' query parameter.
                If provided, adds ?qualifier={endpoint_name} to the URL.

        Returns:
            Tuple[str, Dict[str, str]]: A tuple containing:
                - WebSocket URL (wss://...) with query parameters
                - Headers dictionary with OAuth authentication

        Raises:
            ValueError: If runtime_arn format is invalid or bearer_token is empty.

        Example:
            >>> client = AgentCoreRuntimeClient('us-west-2')
            >>> ws_url, headers = client.generate_ws_connection_oauth(
            ...     runtime_arn='arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-runtime',
            ...     bearer_token='eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...',
            ...     endpoint_name='DEFAULT'
            ... )
        """
        self.logger.info("Generating WebSocket connection with OAuth authentication...")

        # Validate inputs
        if not bearer_token:
            raise ValueError("Bearer token cannot be empty")

        # Validate ARN
        self._parse_runtime_arn(runtime_arn)

        # Auto-generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            self.logger.debug("Auto-generated session ID: %s", session_id)

        # Build WebSocket URL
        ws_url = self._build_websocket_url(runtime_arn, endpoint_name)

        # Convert wss:// to https:// to get host
        https_url = ws_url.replace("wss://", "https://")
        parsed = urlparse(https_url)

        # Generate WebSocket key
        ws_key = base64.b64encode(secrets.token_bytes(16)).decode()

        # Build OAuth headers
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
            "Host": parsed.netloc,
            "Connection": "Upgrade",
            "Upgrade": "websocket",
            "Sec-WebSocket-Key": ws_key,
            "Sec-WebSocket-Version": "13",
            "User-Agent": "OAuth-WebSocket-Client/1.0",
        }

        self.logger.info(
            "✓ OAuth WebSocket connection credentials generated (Session: %s)",
            session_id,
        )
        self.logger.debug("Bearer token length: %d characters", len(bearer_token))

        return ws_url, headers

    # ------------------------------------------------------------------ #
    # HTTP invocation (bearer auth)
    # ------------------------------------------------------------------ #

    def invoke(
        self,
        runtime_arn: str,
        bearer_token: str,
        body: Any,
        *,
        session_id: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = DEFAULT_HTTP_TIMEOUT,
        content_type: str = "application/json",
        accept: str = "application/json",
    ) -> str:
        """Invoke the runtime over HTTP and return the full response body.

        Handles both JSON and Server-Sent Events responses transparently. For
        SSE responses the decoded event data is concatenated in arrival order.

        Args:
            runtime_arn: Full runtime ARN.
            bearer_token: Bearer token for authentication.
            body: Request payload (JSON-serializable for JSON content types,
                ``bytes`` or ``str`` for others).
            session_id: Runtime session id. Auto-generated as a UUID4 when
                not supplied, matching :meth:`generate_ws_connection_oauth`.
            endpoint_name: Endpoint qualifier. Defaults to ``"DEFAULT"`` on
                the wire when not supplied.
            headers: Optional extra headers merged over the auth headers.
            timeout: HTTP read timeout in seconds.
            content_type: Request ``Content-Type``.
            accept: Request ``Accept``.

        Returns:
            The complete response body as a string. For JSON responses
            whose body is a JSON-encoded string, the unwrapped string is
            returned.

        Raises:
            AgentRuntimeError: On non-2xx responses or in-band SSE error
                events.
            ValueError: If the ARN format is invalid.
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        url = self._build_http_url(runtime_arn, "invocations", endpoint_name)
        request_headers = self._build_bearer_headers(
            bearer_token, session_id, accept, content_type
        )
        if headers:
            request_headers.update(headers)

        response = self._http.request(
            "POST",
            url,
            headers=request_headers,
            body=self._serialize_body(body, content_type),
            timeout=timeout,
            preload_content=False,
        )
        try:
            self._check_response(response)
            response_content_type = response.headers.get("content-type", "")
            if "text/event-stream" not in response_content_type:
                return self._read_non_streaming(response)
            return "".join(self._iter_sse_decoded(response))
        finally:
            response.release_conn()

    def invoke_streaming(
        self,
        runtime_arn: str,
        bearer_token: str,
        body: Any,
        *,
        session_id: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = DEFAULT_HTTP_TIMEOUT,
        content_type: str = "application/json",
        accept: str = "application/json",
    ) -> Generator[str, None, None]:
        """Invoke the runtime and yield SSE chunks as they arrive.

        Args:
            runtime_arn: Full runtime ARN.
            bearer_token: Bearer token for authentication.
            body: Request payload.
            session_id: Runtime session id. Auto-generated if not supplied.
            endpoint_name: Endpoint qualifier. Defaults to ``"DEFAULT"``.
            headers: Optional extra headers.
            timeout: HTTP read timeout in seconds.
            content_type: Request ``Content-Type``.
            accept: Request ``Accept``.

        Yields:
            Decoded payload strings from the SSE stream.

        Raises:
            AgentRuntimeError: If the runtime returns a non-2xx status
                or an error event in the stream.
            ValueError: If the ARN format is invalid.
        """
        if not session_id:
            session_id = str(uuid.uuid4())
        url = self._build_http_url(runtime_arn, "invocations", endpoint_name)
        request_headers = self._build_bearer_headers(
            bearer_token, session_id, accept, content_type
        )
        if headers:
            request_headers.update(headers)

        response = self._http.request(
            "POST",
            url,
            headers=request_headers,
            body=self._serialize_body(body, content_type),
            timeout=timeout,
            preload_content=False,
        )
        try:
            self._check_response(response)
            yield from self._iter_sse_decoded(response)
        finally:
            response.release_conn()

    async def invoke_streaming_async(
        self,
        runtime_arn: str,
        bearer_token: str,
        body: Any,
        *,
        session_id: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = DEFAULT_HTTP_TIMEOUT,
        content_type: str = "application/json",
        accept: str = "application/json",
    ) -> AsyncGenerator[str, None]:
        """Async generator version of :meth:`invoke_streaming`.

        The underlying HTTP call is blocking; this wrapper runs it on a
        background thread and delivers chunks through an :class:`asyncio.Queue`,
        so it is safe to ``async for`` over.

        Args:
            runtime_arn: Full runtime ARN.
            bearer_token: Bearer token for authentication.
            body: Request payload.
            session_id: Runtime session id. Auto-generated if not supplied.
            endpoint_name: Endpoint qualifier. Defaults to ``"DEFAULT"``.
            headers: Optional extra headers.
            timeout: HTTP read timeout in seconds.
            content_type: Request ``Content-Type``.
            accept: Request ``Accept``.

        Yields:
            Decoded payload strings from the SSE stream.

        Raises:
            AgentRuntimeError: If the runtime returns a non-2xx status
                or an error event in the stream.
            ValueError: If the ARN format is invalid.
        """
        chunk_queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()
        loop = asyncio.get_running_loop()

        def stream_in_thread() -> None:
            try:
                for decoded in self.invoke_streaming(
                    runtime_arn,
                    bearer_token,
                    body,
                    session_id=session_id,
                    endpoint_name=endpoint_name,
                    headers=headers,
                    timeout=timeout,
                    content_type=content_type,
                    accept=accept,
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
        runtime_arn: str,
        bearer_token: str,
        command: str,
        *,
        session_id: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    ) -> Dict[str, Any]:
        """Run a shell command inside the runtime session and collect the full result.

        Backed by the ``InvokeAgentRuntimeCommand`` API. Blocking; accumulates
        all ``stdout`` and ``stderr`` chunks from the EventStream and returns
        the final exit status.

        Args:
            runtime_arn: Full runtime ARN.
            bearer_token: Bearer token for authentication.
            command: Shell command to run (1 B – 64 KB per the
                ``InvokeAgentRuntimeCommand`` API).
            session_id: Runtime session id. Auto-generated if not supplied.
                The filesystem inside the container persists across calls in
                the same session, but a fresh shell is spawned each time, so
                working directory and environment variables do not.
            endpoint_name: Endpoint qualifier. Defaults to ``"DEFAULT"``.
            headers: Optional extra headers.
            command_timeout: Server-side command wall-clock timeout in
                seconds (1–3600). The HTTP read timeout is derived internally
                as ``command_timeout + 30``.

        Returns:
            Dict with keys ``"stdout"`` (str), ``"stderr"`` (str),
            ``"exitCode"`` (int, ``-1`` if no ``contentStop`` was received),
            and ``"status"`` (``"COMPLETED"``, ``"TIMED_OUT"``, or
            ``"UNKNOWN"``).

        Raises:
            AgentRuntimeError: If the runtime returns a non-2xx status.
            ValueError: If the ARN format is invalid.
        """
        stdout_parts: list = []
        stderr_parts: list = []
        exit_code: int = -1
        status: str = "UNKNOWN"

        for event in self.execute_command_streaming(
            runtime_arn,
            bearer_token,
            command,
            session_id=session_id,
            endpoint_name=endpoint_name,
            headers=headers,
            command_timeout=command_timeout,
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
        runtime_arn: str,
        bearer_token: str,
        command: str,
        *,
        session_id: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream AWS EventStream events from ``InvokeAgentRuntimeCommand``.

        Yields the decoded event payloads (the value inside the server's
        ``"chunk"`` envelope). Each yielded dict has exactly one of the keys
        ``"contentStart"``, ``"contentDelta"``, or ``"contentStop"``.

        Args:
            runtime_arn: Full runtime ARN.
            bearer_token: Bearer token for authentication.
            command: Shell command to run.
            session_id: Runtime session id. Auto-generated if not supplied.
            endpoint_name: Endpoint qualifier. Defaults to ``"DEFAULT"``.
            headers: Optional extra headers.
            command_timeout: Server-side wall-clock timeout in seconds
                (1–3600).

        Yields:
            Parsed event payload dicts.

        Raises:
            AgentRuntimeError: If the runtime returns a non-2xx status.
            ValueError: If the ARN format is invalid.
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        request_headers = self._build_bearer_headers(
            bearer_token,
            session_id,
            accept="application/vnd.amazon.eventstream",
            content_type="application/json",
        )
        if headers:
            request_headers.update(headers)

        response = self._http.request(
            "POST",
            self._build_http_url(runtime_arn, "commands", endpoint_name),
            headers=request_headers,
            body=json.dumps({"command": command, "timeout": command_timeout}).encode(
                "utf-8"
            ),
            timeout=command_timeout + 30,
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
        runtime_arn: str,
        bearer_token: str,
        *,
        session_id: str,
        endpoint_name: Optional[str] = None,
        client_token: Optional[str] = None,
        timeout: int = DEFAULT_HTTP_TIMEOUT,
    ) -> Dict[str, Any]:
        """Terminate a runtime session via HTTP.

        Args:
            runtime_arn: Full runtime ARN.
            bearer_token: Bearer token for authentication.
            session_id: The session id to stop.
            endpoint_name: Endpoint qualifier. Defaults to ``"DEFAULT"``.
            client_token: Idempotency token. Auto-generated as a UUID4
                when not supplied.
            timeout: HTTP read timeout in seconds.

        Returns:
            Parsed JSON body of the response (often an empty dict).

        Raises:
            AgentRuntimeError: If the runtime returns a non-2xx status
                (for example ``HTTP 404`` for an unknown session).
            ValueError: If the ARN format is invalid.
        """
        request_headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
            "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
        }
        response = self._http.request(
            "POST",
            self._build_http_url(runtime_arn, "stopruntimesession", endpoint_name),
            headers=request_headers,
            body=json.dumps({"clientToken": client_token or str(uuid.uuid4())}).encode(
                "utf-8"
            ),
            timeout=timeout,
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
    # Response parsing helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_response(response: "_UrllibResponse") -> None:
        """Raise :class:`AgentRuntimeError` for non-2xx responses.

        Attempts to parse the body as JSON and surface ``error``,
        ``error_type``, and ``message`` fields. Falls back to the raw body
        text.

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
                error=str(body.get("error", reason)),
                error_type=str(body.get("error_type", error_type)),
                message=str(
                    body.get("message", body_bytes.decode("utf-8", errors="replace"))
                ),
            )
        raise AgentRuntimeError(
            error=str(body) or reason,
            error_type=error_type,
        )

    @staticmethod
    def _read_non_streaming(response: "_UrllibResponse") -> str:
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

    @classmethod
    def _iter_sse_decoded(cls, response: "_UrllibResponse") -> Iterator[str]:
        """Iterate decoded SSE payloads from a streaming response.

        Args:
            response: The urllib3 response, opened with
                ``preload_content=False``.

        Yields:
            Decoded payload strings (one per non-empty ``data:`` line
            or JSON line).
        """
        for raw_line in cls._iter_lines(response):
            if not raw_line:
                continue
            decoded = cls._decode_sse_line(raw_line.decode("utf-8", errors="replace"))
            if decoded:
                yield decoded

    @staticmethod
    def _iter_lines(response: "_UrllibResponse") -> Iterator[bytes]:
        r"""Yield lines from a streaming urllib3 response.

        Splits on ``\n`` and strips a trailing ``\r`` so it handles both
        LF and CRLF line endings. Preserves empty lines (the caller
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

    @staticmethod
    def _decode_sse_line(line: str) -> Optional[str]:
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
