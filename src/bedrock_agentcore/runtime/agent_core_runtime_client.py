"""Client for generating WebSocket authentication for AgentCore Runtime.

This module provides a client for generating authentication credentials
for WebSocket connections to AgentCore Runtime endpoints.
"""

import base64
import datetime
import logging
import secrets
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode, urlparse

from .shell._validation import parse_runtime_arn, validate_shell_id

if TYPE_CHECKING:
    from .shell import AuthMode, ReconnectConfig, ShellSession

import boto3
from botocore.auth import SigV4Auth, SigV4QueryAuth
from botocore.awsrequest import AWSRequest
from botocore.config import Config
from botocore.exceptions import ClientError

from .._utils.config import WaitConfig
from .._utils.endpoints import get_data_plane_endpoint, validate_region
from .._utils.polling import wait_until, wait_until_deleted
from .._utils.snake_case import accept_snake_case_kwargs, convert_kwargs
from .._utils.user_agent import build_user_agent_suffix

DEFAULT_PRESIGNED_URL_TIMEOUT = 300
MAX_PRESIGNED_URL_TIMEOUT = 300

_RUNTIME_FAILED_STATUSES = {"CREATE_FAILED", "UPDATE_FAILED"}
_ENDPOINT_FAILED_STATUSES = {"CREATE_FAILED", "UPDATE_FAILED", "DELETE_FAILED"}


class AgentCoreRuntimeClient:
    """Client for generating WebSocket authentication for AgentCore Runtime.

    This client provides authentication credentials for WebSocket connections
    to AgentCore Runtime endpoints, allowing applications to establish
    bidirectional streaming connections with agent runtimes.

    Attributes:
        region (str): The AWS region being used.
        session (boto3.Session): The boto3 session for AWS credentials.
    """

    _ALLOWED_DP_METHODS = {
        "invoke_agent_runtime",
        "stop_runtime_session",
    }

    _ALLOWED_CP_METHODS = {
        "create_agent_runtime",
        "update_agent_runtime",
        "get_agent_runtime",
        "delete_agent_runtime",
        "list_agent_runtimes",
        "create_agent_runtime_endpoint",
        "get_agent_runtime_endpoint",
        "update_agent_runtime_endpoint",
        "delete_agent_runtime_endpoint",
        "list_agent_runtime_endpoints",
        "list_agent_runtime_versions",
        "delete_agent_runtime_version",
    }

    def __init__(
        self,
        region: Optional[str] = None,
        session: Optional[boto3.Session] = None,
        integration_source: Optional[str] = None,
    ) -> None:
        """Initialize an AgentCoreRuntime client for the specified AWS region.

        Args:
            region: AWS region name. If not provided, uses the session's
                region or "us-west-2".
            session: Optional boto3 Session to use. If not provided, a
                default session is created.
            integration_source: Optional integration source for user-agent
                telemetry.
        """
        self.region = validate_region(region or (session.region_name if session else None) or "us-west-2")
        self.session = session if session else boto3.Session(region_name=self.region)
        self.integration_source = integration_source
        self.logger = logging.getLogger(__name__)

        user_agent_extra = build_user_agent_suffix(integration_source)
        client_config = Config(user_agent_extra=user_agent_extra)

        self.cp_client = self.session.client(
            "bedrock-agentcore-control",
            region_name=self.region,
            config=client_config,
        )
        self.dp_client = self.session.client(
            "bedrock-agentcore",
            region_name=self.region,
            config=client_config,
        )
        self.logger.info(
            "Initialized AgentCoreRuntimeClient for region: %s",
            self.region,
        )

    # Pass-through
    # -------------------------------------------------------------------------
    def __getattr__(self, name: str):
        """Dynamically forward allowlisted method calls to the appropriate boto3 client."""
        if name in self._ALLOWED_DP_METHODS and hasattr(self.dp_client, name):
            method = getattr(self.dp_client, name)
            self.logger.debug("Forwarding method '%s' to dp_client", name)
            return accept_snake_case_kwargs(method)

        if name in self._ALLOWED_CP_METHODS and hasattr(self.cp_client, name):
            method = getattr(self.cp_client, name)
            self.logger.debug("Forwarding method '%s' to cp_client", name)
            return accept_snake_case_kwargs(method)

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'. "
            f"Method not found on dp_client or cp_client. "
            f"Available methods can be found in the boto3 documentation for "
            f"'bedrock-agentcore' and 'bedrock-agentcore-control' services."
        )

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

    def _sigv4_sign(
        self,
        https_url: str,
        signed_headers: Optional[Dict[str, str]] = None,
        unsigned_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """SigV4-sign a WebSocket upgrade URL and return the full header dict.

        Args:
            https_url: The https:// form of the wss:// URL to sign.
            signed_headers: Headers to include inside the AWSRequest so they
                are covered by the SigV4 signature.
            unsigned_headers: Headers appended to the result after signing
                (not covered by the signature).

        Returns:
            Dict with Host, X-Amz-Date, Authorization, optionally
            X-Amz-Security-Token, plus any signed/unsigned headers passed in.
        """
        credentials = self.session.get_credentials()
        if not credentials:
            raise RuntimeError("No AWS credentials found")
        frozen = credentials.get_frozen_credentials()
        host = urlparse(https_url).netloc
        req_headers: Dict[str, str] = {
            "host": host,
            "x-amz-date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        }
        if signed_headers:
            req_headers.update({k.lower(): v for k, v in signed_headers.items()})
        request = AWSRequest(method="GET", url=https_url, headers=req_headers)
        SigV4Auth(frozen, "bedrock-agentcore", self.region).add_auth(request)
        headers: Dict[str, str] = {
            "Host": host,
            "X-Amz-Date": request.headers["x-amz-date"],
            "Authorization": request.headers["Authorization"],
        }
        if frozen.token:
            headers["X-Amz-Security-Token"] = frozen.token
        if signed_headers:
            headers.update(signed_headers)
        if unsigned_headers:
            headers.update(unsigned_headers)
        return headers

    def _presign(self, https_url: str, expires: int) -> str:
        """Sign a WebSocket URL with SigV4 query-string auth and return it as wss://.

        Args:
            https_url: The https:// URL to sign (query params already embedded).
            expires: Seconds until the presigned URL expires.

        Returns:
            Presigned wss:// URL.

        Raises:
            RuntimeError: If no AWS credentials are found.
        """
        credentials = self.session.get_credentials()
        if not credentials:
            raise RuntimeError("No AWS credentials found")
        frozen = credentials.get_frozen_credentials()
        request = AWSRequest(method="GET", url=https_url, headers={"host": urlparse(https_url).hostname})
        SigV4QueryAuth(frozen, "bedrock-agentcore", self.region, expires=expires).add_auth(request)
        return request.url.replace("https://", "wss://")

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
        parse_runtime_arn(runtime_arn)

        # Auto-generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
            self.logger.debug("Auto-generated session ID: %s", session_id)

        # Build WebSocket URL
        ws_url = self._build_websocket_url(runtime_arn, endpoint_name)
        headers = self._sigv4_sign(
            ws_url.replace("wss://", "https://"),
            unsigned_headers={
                "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
                "Upgrade": "websocket",
                "Connection": "Upgrade",
                "Sec-WebSocket-Version": "13",
                "Sec-WebSocket-Key": base64.b64encode(secrets.token_bytes(16)).decode(),
                "User-Agent": "AgentCoreRuntimeClient/1.0",
            },
        )
        self.logger.info("✓ WebSocket connection credentials generated (Session: %s)", session_id)
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
            raise ValueError(f"Expiry timeout cannot exceed {MAX_PRESIGNED_URL_TIMEOUT} seconds, got {expires}")

        # Validate ARN
        parse_runtime_arn(runtime_arn)

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
        presigned_url = self._presign(ws_url.replace("wss://", "https://"), expires)
        self.logger.info("✓ Presigned URL generated (expires in %s seconds, Session: %s)", expires, session_id)
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
        parse_runtime_arn(runtime_arn)

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

        self.logger.info("✓ OAuth WebSocket connection credentials generated (Session: %s)", session_id)
        self.logger.debug("Bearer token length: %d characters", len(bearer_token))

        return ws_url, headers

    # InvokeAgentRuntimeCommandShell auth helpers
    # -------------------------------------------------------------------------

    def _build_shell_url(
        self,
        runtime_arn: str,
        endpoint_name: Optional[str] = None,
        shell_id: Optional[str] = None,
    ) -> str:
        """Build wss:// URL for /ws/shells (shell op).

        Args:
            runtime_arn: Full runtime ARN.
            endpoint_name: Optional qualifier query param.
            shell_id: Optional shellId query param.

        Returns:
            WebSocket URL (wss://…).
        """
        host = get_data_plane_endpoint(self.region).replace("https://", "")
        encoded_arn = quote(runtime_arn, safe="")
        path = f"/runtimes/{encoded_arn}/ws/shells"

        params: Dict[str, str] = {}
        if endpoint_name:
            params["qualifier"] = endpoint_name
        if shell_id:
            params["shellId"] = shell_id

        qs = urlencode(params)
        return f"wss://{host}{path}?{qs}" if qs else f"wss://{host}{path}"

    def connect_shell(
        self,
        runtime_arn: str,
        session_id: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        shell_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """Return (wss_url, headers) for a SigV4-authenticated shell session.

        SigV4 is the correct auth path for server-side Python.  Browsers cannot
        set arbitrary headers on a WebSocket upgrade (RFC 6455); use
        ``connect_shell_oauth`` for browser-facing use cases instead.

        Args:
            runtime_arn: Full agent runtime ARN.
            session_id: Routes to an existing VM. Auto-generated UUID if omitted;
                a new VM is provisioned.
            endpoint_name: Endpoint qualifier (default: DEFAULT).
            shell_id: Client-chosen shell name (1–128 chars, no ?, #, &).
                Auto-generated UUID if omitted. **Store this value** — passing the same
                ID reconnects to the same PTY, preserving shell state and up to 256 KB
                of buffered output.

        Returns:
            ``(wss_url, headers)`` — pass both directly to any WebSocket library.

        Raises:
            ValueError: If the ARN format is invalid or ``shell_id`` is
                outside the allowed character set.
            RuntimeError: If no AWS credentials are found.

        Example:
            url, headers = client.connect_shell(
                runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/r",
                shell_id="my-debug-shell",
            )
            ws = await websockets.connect(url, additional_headers=headers)
        """
        parse_runtime_arn(runtime_arn)
        if not session_id:
            session_id = str(uuid.uuid4())
        if not shell_id:
            shell_id = str(uuid.uuid4())
        validate_shell_id(shell_id)
        ws_url = self._build_shell_url(runtime_arn, endpoint_name, shell_id)
        headers = self._sigv4_sign(
            ws_url.replace("wss://", "https://"),
            signed_headers={"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id},
            unsigned_headers={"User-Agent": build_user_agent_suffix(self.integration_source)},
        )
        self.logger.info(
            "Generated shell connection (session=%s, shell_id=%s)",
            session_id,
            shell_id,
        )
        return ws_url, headers

    def connect_shell_presigned(
        self,
        runtime_arn: str,
        session_id: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        shell_id: Optional[str] = None,
        expires: int = DEFAULT_PRESIGNED_URL_TIMEOUT,
    ) -> str:
        """Return a presigned wss:// URL for a shell session (auth in query string).

        Useful when you need to hand a URL to another process or service without
        sharing AWS credentials directly.

        Args:
            runtime_arn: Full agent runtime ARN.
            session_id: Routes to an existing VM. Auto-generated if omitted.
            endpoint_name: Endpoint qualifier (default: DEFAULT).
            shell_id: Client-chosen shell name. Store this value for
                reconnection. Auto-generated if omitted.
            expires: Seconds until the URL expires (max 300).

        Returns:
            Presigned wss:// URL — open with any WebSocket client, no extra headers needed.

        Raises:
            ValueError: If ``expires`` exceeds the maximum, the ARN is invalid, or
                ``shell_id`` contains forbidden characters.
            RuntimeError: If no AWS credentials are found.

        Example:
            url = client.connect_shell_presigned(
                runtime_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/r",
                shell_id="build-shell",
                expires=120,
            )
            ws = await websockets.connect(url)
        """
        if expires > MAX_PRESIGNED_URL_TIMEOUT:
            raise ValueError(f"Expiry timeout cannot exceed {MAX_PRESIGNED_URL_TIMEOUT} seconds, got {expires}")
        parse_runtime_arn(runtime_arn)
        if not session_id:
            session_id = str(uuid.uuid4())
        if not shell_id:
            shell_id = str(uuid.uuid4())
        validate_shell_id(shell_id)
        ws_url = self._build_shell_url(runtime_arn, endpoint_name, shell_id)
        https_url = ws_url.replace("wss://", "https://")
        # session_id rides as a query param so it is covered by the signature
        sep = "&" if "?" in https_url else "?"
        https_url += sep + urlencode({"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id})
        presigned = self._presign(https_url, expires)
        self.logger.info(
            "Generated presigned shell URL (expires=%ds, shell_id=%s)",
            expires,
            shell_id,
        )
        return presigned

    def connect_shell_oauth(
        self,
        runtime_arn: str,
        bearer_token: str,
        session_id: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        shell_id: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        """Return (wss_url, subprotocols) for an OAuth-authenticated shell session.

        This is the only valid auth path for browser clients: browsers cannot set
        arbitrary headers on a WebSocket upgrade (RFC 6455), so the bearer
        token is embedded in the ``Sec-WebSocket-Protocol`` handshake using the
        ``base64UrlBearerAuthorization`` scheme instead.

        Server-side Python callers should prefer ``connect_shell`` with
        SigV4.  Use this method when building a browser relay or xterm.js
        integration.

        Args:
            runtime_arn: Full agent runtime ARN.
            bearer_token: OAuth bearer token obtained from your identity provider.
            session_id: Routes to an existing VM. Auto-generated if omitted.
            endpoint_name: Endpoint qualifier (default: DEFAULT).
            shell_id: Client-chosen shell name. Store this value for
                reconnection. Auto-generated if omitted.

        Returns:
            ``(wss_url, subprotocols)`` — pass both to the WebSocket constructor.
            ``subprotocols`` contains the base64url-encoded token as
            ``base64UrlBearerAuthorization.<token>`` plus the sentinel
            ``base64UrlBearerAuthorization``.

        Raises:
            ValueError: If ``bearer_token`` is empty, the ARN is invalid, or
                ``shell_id`` contains forbidden characters.

        Example (server-side relay):
            url, protos = client.connect_shell_oauth(
                runtime_arn="arn:...",
                bearer_token=await get_oauth_token(),
                shell_id="inspector-shell",
            )
            ws = await websockets.connect(url, subprotocols=protos)

        Example (browser):
            # Backend returns (url, subprotocols); browser does:
            # const ws = new WebSocket(url, subprotocols)
        """
        if not bearer_token:
            raise ValueError("bearer_token cannot be empty")
        parse_runtime_arn(runtime_arn)
        if not session_id:
            session_id = str(uuid.uuid4())
        if not shell_id:
            shell_id = str(uuid.uuid4())
        validate_shell_id(shell_id)

        ws_url = self._build_shell_url(runtime_arn, endpoint_name, shell_id)
        sep = "&" if "?" in ws_url else "?"
        ws_url += sep + urlencode({"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id})

        # Bearer token embedded as a WebSocket subprotocol using the
        # base64UrlBearerAuthorization scheme — the only mechanism available to
        # browser clients for passing auth on a WS upgrade (RFC 6455).
        encoded = base64.urlsafe_b64encode(bearer_token.encode()).decode().rstrip("=")
        if len(encoded) > 4096:
            raise ValueError(
                f"bearer_token too large to embed in Sec-WebSocket-Protocol ({len(encoded)} chars encoded, max 4096)"
            )
        subprotocols: List[str] = [
            f"base64UrlBearerAuthorization.{encoded}",
            "base64UrlBearerAuthorization",
        ]

        self.logger.info("Generated OAuth shell connection (shell_id=%s)", shell_id)
        return ws_url, subprotocols

    def open_shell(
        self,
        runtime_arn: str,
        session_id: Optional[str] = None,
        shell_id: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        auth: "AuthMode" = "sigv4",
        reconnect_config: Optional["ReconnectConfig"] = None,
    ) -> "ShellSession":
        r"""Create a ``ShellSession`` for interactive shell access to an agent VM.

        Returns an async context manager that connects on ``__aenter__`` and
        sends a graceful CLOSE frame on ``__aexit__``.

        Args:
            runtime_arn: Full agent runtime ARN.
            session_id: Runtime session ID — routes to an existing VM. Auto-generated
                UUID if omitted; a new VM is provisioned.
            shell_id: Client-chosen shell name (1–128 UTF-8 chars, no
                ``?``, ``#``, or ``&``). Auto-generated if omitted. **Store this
                value alongside** ``session_id`` — both are required to reconnect to
                the same PTY. ``shell_id`` names the PTY; ``session_id`` routes to
                the VM that hosts it. Passing either without the other will not
                reconnect successfully.
            endpoint_name: Endpoint qualifier (default: DEFAULT).
            auth: Authentication mode. One of:

                - ``"sigv4"`` *(default)* — SigV4-signed headers from the boto3
                  session. Correct for all server-side Python use cases.
                - ``PresignedAuth(expires=N)`` — auth embedded in the URL query
                  string; valid for up to ``expires`` seconds (max 300). Use when
                  handing a URL to another process without sharing AWS credentials.
                - ``OAuthAuth(bearer_token="…")`` — bearer token embedded as a
                  WebSocket subprotocol. The only valid path for browser clients
                  (browsers cannot set arbitrary headers on a WS upgrade).

            reconnect_config: When provided, ``ShellSession`` automatically
                reconnects on unexpected WebSocket disconnects using the same
                ``shell_id``. The ``on_reconnect`` callback fires after each
                successful reconnect with ``reconnected=True/False`` so callers can
                react to the buffered-output burst. ``None`` (default) disables
                auto-retry — callers handle reconnection explicitly.

        Returns:
            ``ShellSession`` async context manager.

        Example — SigV4 (default):
            async with client.open_shell(runtime_arn, session_id=sid) as shell:
                await shell.send("cat /etc/os-release\\n")
                async for frame in shell:
                    if frame.channel == ShellChannel.STDOUT:
                        print(frame.text, end="")
                    elif frame.channel == ShellChannel.STATUS:
                        # Termination frames have empty metadata (no shellId).
                        if not frame.json().get("metadata", {}).get("shellId"):
                            break

        Example — presigned URL:
            async with client.open_shell(
                runtime_arn,
                auth=PresignedAuth(expires=120),
                shell_id="build-shell",
            ) as shell:
                ...

        Example — OAuth (browser relay or OAuth-only environments):
            async with client.open_shell(
                runtime_arn,
                auth=OAuthAuth(bearer_token=await get_oauth_token()),
                shell_id="inspector-shell",
            ) as shell:
                ...

        Example — auto-reconnect with callback:
            async def on_reconnect(reconnected: bool) -> None:
                print(f"reconnected={reconnected}")

            config = ReconnectConfig(max_retries=5, on_reconnect=on_reconnect)
            async with client.open_shell(
                runtime_arn,
                shell_id="debug",
                reconnect_config=config,
            ) as shell:
                async for frame in shell:
                    ...
        """
        from .shell import ShellSession

        parsed = parse_runtime_arn(runtime_arn)
        if parsed["region"] != self.region:
            raise ValueError(
                f"ARN region {parsed['region']!r} does not match client region {self.region!r}. "
                "Create a client for the same region as the runtime ARN, or use the ARN's region."
            )
        return ShellSession(
            client=self,
            runtime_arn=runtime_arn,
            session_id=session_id,
            shell_id=shell_id,
            endpoint_name=endpoint_name,
            auth=auth,
            reconnect_config=reconnect_config,
        )

    # *_and_wait methods
    # -------------------------------------------------------------------------
    def create_agent_runtime_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create an agent runtime and wait for it to reach READY status.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the create_agent_runtime API.

        Returns:
            Runtime details when READY.

        Raises:
            RuntimeError: If the runtime reaches a failed state.
            TimeoutError: If the runtime doesn't become READY within max_wait.
        """
        response = self.cp_client.create_agent_runtime(**convert_kwargs(kwargs))
        rid = response["agentRuntimeId"]
        return wait_until(
            lambda: self.cp_client.get_agent_runtime(agentRuntimeId=rid),
            "READY",
            _RUNTIME_FAILED_STATUSES,
            wait_config,
            error_field="failureReason",
        )

    def update_agent_runtime_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Update an agent runtime and wait for it to reach READY status.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the update_agent_runtime API.

        Returns:
            Runtime details when READY.

        Raises:
            RuntimeError: If the runtime reaches a failed state.
            TimeoutError: If the runtime doesn't become READY within max_wait.
        """
        response = self.cp_client.update_agent_runtime(**convert_kwargs(kwargs))
        rid = response["agentRuntimeId"]
        return wait_until(
            lambda: self.cp_client.get_agent_runtime(agentRuntimeId=rid),
            "READY",
            _RUNTIME_FAILED_STATUSES,
            wait_config,
            error_field="failureReason",
        )

    def delete_agent_runtime_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> None:
        """Delete an agent runtime and wait for deletion to complete.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the delete_agent_runtime API.

        Raises:
            TimeoutError: If the runtime isn't deleted within max_wait.
        """
        response = self.cp_client.delete_agent_runtime(**convert_kwargs(kwargs))
        rid = response["agentRuntimeId"]
        wait_until_deleted(
            lambda: self.cp_client.get_agent_runtime(agentRuntimeId=rid),
            wait_config=wait_config,
        )

    def create_agent_runtime_endpoint_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create an agent runtime endpoint and wait for it to reach READY.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the
                create_agent_runtime_endpoint API.

        Returns:
            Endpoint details when READY.

        Raises:
            RuntimeError: If the endpoint reaches a failed state.
            TimeoutError: If the endpoint doesn't become READY within
                max_wait.
        """
        converted = convert_kwargs(kwargs)
        response = self.cp_client.create_agent_runtime_endpoint(
            **converted,
        )
        rid = converted.get("agentRuntimeId")
        ename = response.get("name", kwargs.get("name", "DEFAULT"))
        return wait_until(
            lambda: self.cp_client.get_agent_runtime_endpoint(
                agentRuntimeId=rid,
                endpointName=ename,
            ),
            "READY",
            _ENDPOINT_FAILED_STATUSES,
            wait_config,
            error_field="failureReason",
        )

    def update_agent_runtime_endpoint_and_wait(
        self,
        wait_config: Optional[WaitConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Update an agent runtime endpoint and wait for READY status.

        Args:
            wait_config: Optional WaitConfig for polling behavior.
            **kwargs: Arguments forwarded to the
                update_agent_runtime_endpoint API.

        Returns:
            Endpoint details when READY.

        Raises:
            RuntimeError: If the endpoint reaches a failed state.
            TimeoutError: If the endpoint doesn't become READY within
                max_wait.
        """
        converted = convert_kwargs(kwargs)
        response = self.cp_client.update_agent_runtime_endpoint(
            **converted,
        )
        rid = converted.get("agentRuntimeId")
        ename = response.get("name", kwargs.get("endpointName", "DEFAULT"))
        return wait_until(
            lambda: self.cp_client.get_agent_runtime_endpoint(
                agentRuntimeId=rid,
                endpointName=ename,
            ),
            "READY",
            _ENDPOINT_FAILED_STATUSES,
            wait_config,
            error_field="failureReason",
        )

    # Higher-level orchestration methods
    # -------------------------------------------------------------------------
    def get_aggregated_status(
        self,
        agent_runtime_id: str,
        endpoint_name: str = "DEFAULT",
    ) -> Dict[str, Any]:
        """Get aggregated status of runtime and endpoint.

        Args:
            agent_runtime_id: The agent runtime ID.
            endpoint_name: Endpoint name (default: "DEFAULT").

        Returns:
            Dict with 'runtime' and 'endpoint' status details.
        """
        result: Dict[str, Any] = {"runtime": None, "endpoint": None}

        try:
            result["runtime"] = self.cp_client.get_agent_runtime(
                agentRuntimeId=agent_runtime_id,
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                raise
            result["runtime"] = {"error": str(e)}

        try:
            result["endpoint"] = self.cp_client.get_agent_runtime_endpoint(
                agentRuntimeId=agent_runtime_id,
                endpointName=endpoint_name,
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                raise
            result["endpoint"] = {"error": str(e)}

        return result

    def teardown_endpoint_and_runtime(
        self,
        agent_runtime_id: str,
        endpoint_name: str = "DEFAULT",
    ) -> None:
        """Delete endpoint then runtime in correct order.

        Silently ignores ResourceNotFoundException for either resource
        (already deleted).

        Args:
            agent_runtime_id: The agent runtime ID.
            endpoint_name: Endpoint name (default: "DEFAULT").
        """
        try:
            self.cp_client.delete_agent_runtime_endpoint(
                agentRuntimeId=agent_runtime_id,
                endpointName=endpoint_name,
            )
            self.logger.info(
                "Deleted endpoint '%s' for runtime %s",
                endpoint_name,
                agent_runtime_id,
            )
            wait_until_deleted(
                lambda: self.cp_client.get_agent_runtime_endpoint(
                    agentRuntimeId=agent_runtime_id,
                    endpointName=endpoint_name,
                ),
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                raise
            self.logger.info("Endpoint '%s' not found, skipping", endpoint_name)

        try:
            self.delete_agent_runtime_and_wait(
                agentRuntimeId=agent_runtime_id,
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                raise
            self.logger.info("Runtime %s not found, skipping", agent_runtime_id)
