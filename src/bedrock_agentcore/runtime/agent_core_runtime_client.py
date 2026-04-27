"""Client for generating WebSocket authentication for AgentCore Runtime.

This module provides a client for generating authentication credentials
for WebSocket connections to AgentCore Runtime endpoints.
"""

import base64
import datetime
import logging
import secrets
import uuid
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote, urlencode, urlparse

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
from .utils import is_valid_partition

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
        session = session if session else boto3.Session()
        self.region = validate_region(region or session.region_name or "us-west-2")
        self.session = session
        self.integration_source = integration_source
        self.logger = logging.getLogger(__name__)

        user_agent_extra = build_user_agent_suffix(integration_source)
        client_config = Config(user_agent_extra=user_agent_extra)

        self.cp_client = session.client(
            "bedrock-agentcore-control",
            region_name=self.region,
            config=client_config,
        )
        self.dp_client = session.client(
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

        if parts[0] != "arn" or not is_valid_partition(parts[1]) or parts[2] != "bedrock-agentcore":
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
                "x-amz-date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
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
        request = AWSRequest(method="GET", url=https_url, headers={"host": url.hostname})

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

        self.logger.info("✓ OAuth WebSocket connection credentials generated (Session: %s)", session_id)
        self.logger.debug("Bearer token length: %d characters", len(bearer_token))

        return ws_url, headers

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
