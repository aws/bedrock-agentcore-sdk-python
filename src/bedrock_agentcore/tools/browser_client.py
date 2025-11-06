"""Client for interacting with the Browser sandbox service.

This module provides a client for the AWS Browser sandbox, allowing
applications to start, stop, and automate browser interactions in a managed
sandbox environment using Playwright.
"""

import base64
import datetime
import logging
import secrets
import uuid
from contextlib import contextmanager
from typing import Dict, Generator, Optional, Tuple
from urllib.parse import urlparse

import boto3
from botocore.auth import SigV4Auth, SigV4QueryAuth
from botocore.awsrequest import AWSRequest

from .._utils.endpoints import get_control_plane_endpoint, get_data_plane_endpoint

DEFAULT_IDENTIFIER = "aws.browser.v1"
DEFAULT_SESSION_TIMEOUT = 3600
DEFAULT_LIVE_VIEW_PRESIGNED_URL_TIMEOUT = 300
MAX_LIVE_VIEW_PRESIGNED_URL_TIMEOUT = 300


class BrowserClient:
    """Client for interacting with the AWS Browser sandbox service.

    This client handles the session lifecycle and browser automation for
    Browser sandboxes, providing an interface to perform web automation
    tasks in a secure, managed environment.

    Attributes:
        region (str): The AWS region being used.
        control_plane_client: The boto3 client for control plane operations.
        data_plane_service_name (str): AWS service name for the data plane.
        client: The boto3 client for interacting with the service.
        identifier (str, optional): The browser identifier.
        session_id (str, optional): The active session ID.
    """

    def __init__(self, region: str) -> None:
        """Initialize a Browser client for the specified AWS region.

        Args:
            region (str): The AWS region to use for the Browser service.
        """
        self.region = region
        self.logger = logging.getLogger(__name__)

        # Control plane client for browser management
        self.control_plane_client = boto3.client(
            "bedrock-agentcore-control",
            region_name=region,
            endpoint_url=get_control_plane_endpoint(region),
        )

        # Data plane client for session operations
        self.data_plane_client = boto3.client(
            "bedrock-agentcore",
            region_name=region,
            endpoint_url=get_data_plane_endpoint(region),
        )

        self._identifier = None
        self._session_id = None

    @property
    def identifier(self) -> Optional[str]:
        """Get the current browser identifier."""
        return self._identifier

    @identifier.setter
    def identifier(self, value: Optional[str]):
        """Set the browser identifier."""
        self._identifier = value

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]):
        """Set the session ID."""
        self._session_id = value

    def create_browser(
        self,
        name: str,
        execution_role_arn: str,
        network_configuration: Optional[Dict] = None,
        description: Optional[str] = None,
        recording: Optional[Dict] = None,
        browser_signing: Optional[Dict] = None,
        tags: Optional[Dict[str, str]] = None,
        client_token: Optional[str] = None,
    ) -> Dict:
        """Create a custom browser with specific configuration.

        This is a control plane operation that provisions a new browser with
        custom settings including Web Bot Auth, VPC, and recording configuration.

        Args:
            name (str): The name for the browser. Must match pattern [a-zA-Z][a-zA-Z0-9_]{0,47}
            execution_role_arn (str): IAM role ARN with permissions for browser operations
            network_configuration (Optional[Dict]): Network configuration:
                {
                    "networkMode": "PUBLIC" or "VPC",
                    "vpcConfig": {  # Required if networkMode is VPC
                        "securityGroups": ["sg-xxx"],
                        "subnets": ["subnet-xxx"]
                    }
                }
            description (Optional[str]): Description of the browser (1-4096 chars)
            recording (Optional[Dict]): Recording configuration:
                {
                    "enabled": True,
                    "s3Location": {
                        "bucket": "bucket-name",
                        "keyPrefix": "path/prefix"
                    }
                }
            browser_signing (Optional[Dict]): Web Bot Auth configuration (NEW FEATURE):
                {
                    "enabled": True
                }
            tags (Optional[Dict[str, str]]): Tags for the browser
            client_token (Optional[str]): Idempotency token

        Returns:
            Dict: Response containing:
                - browserArn (str): ARN of created browser
                - browserId (str): Unique browser identifier
                - createdAt (datetime): Creation timestamp
                - status (str): Browser status (CREATING, READY, etc.)

        Example:
            >>> client = BrowserClient('us-west-2')
            >>> # Create browser with Web Bot Auth enabled
            >>> response = client.create_browser(
            ...     name="my_signed_browser",
            ...     execution_role_arn="arn:aws:iam::123456789012:role/BrowserRole",
            ...     network_configuration={"networkMode": "PUBLIC"},
            ...     browser_signing={"enabled": True},
            ...     recording={
            ...         "enabled": True,
            ...         "s3Location": {
            ...             "bucket": "my-recordings",
            ...             "keyPrefix": "browser-sessions/"
            ...         }
            ...     }
            ... )
            >>> browser_id = response['browserId']
        """
        self.logger.info("Creating browser: %s", name)

        request_params = {
            "name": name,
            "executionRoleArn": execution_role_arn,
            "networkConfiguration": network_configuration or {"networkMode": "PUBLIC"},
        }

        if description:
            request_params["description"] = description

        if recording:
            request_params["recording"] = recording

        if browser_signing:
            request_params["browserSigning"] = browser_signing
            self.logger.info("🔐 Web Bot Auth (browserSigning) enabled")

        if tags:
            request_params["tags"] = tags

        if client_token:
            request_params["clientToken"] = client_token

        response = self.control_plane_client.create_browser(**request_params)
        return response

    def delete_browser(self, browser_id: str, client_token: Optional[str] = None) -> Dict:
        """Delete a custom browser.

        Args:
            browser_id (str): The browser identifier to delete
            client_token (Optional[str]): Idempotency token

        Returns:
            Dict: Response containing:
                - browserId (str): ID of deleted browser
                - lastUpdatedAt (datetime): Update timestamp
                - status (str): Deletion status

        Example:
            >>> client.delete_browser("my-browser-abc123")
        """
        self.logger.info("Deleting browser: %s", browser_id)

        request_params = {"browserId": browser_id}
        if client_token:
            request_params["clientToken"] = client_token

        response = self.control_plane_client.delete_browser(**request_params)
        return response

    def get_browser(self, browser_id: str) -> Dict:
        """Get detailed information about a browser.

        Args:
            browser_id (str): The browser identifier

        Returns:
            Dict: Browser details including:
                - browserArn, browserId, name, description
                - createdAt, lastUpdatedAt
                - executionRoleArn
                - networkConfiguration
                - recording configuration
                - browserSigning configuration (if enabled)
                - status (CREATING, CREATE_FAILED, READY, DELETING, etc.)
                - failureReason (if failed)

        Example:
            >>> browser_info = client.get_browser("my-browser-abc123")
            >>> print(f"Status: {browser_info['status']}")
            >>> if browser_info.get('browserSigning'):
            ...     print("Web Bot Auth is enabled!")
        """
        self.logger.info("Getting browser: %s", browser_id)
        response = self.control_plane_client.get_browser(browserId=browser_id)
        return response

    def list_browsers(
        self,
        browser_type: Optional[str] = None,
        max_results: int = 10,
        next_token: Optional[str] = None,
    ) -> Dict:
        """List all browsers in the account.

        Args:
            browser_type (Optional[str]): Filter by type: "SYSTEM" or "CUSTOM"
            max_results (int): Maximum results to return (1-100, default 10)
            next_token (Optional[str]): Token for pagination

        Returns:
            Dict: Response containing:
                - browserSummaries (List[Dict]): List of browser summaries
                - nextToken (str): Token for next page (if more results)

        Example:
            >>> # List all custom browsers
            >>> response = client.list_browsers(browser_type="CUSTOM")
            >>> for browser in response['browserSummaries']:
            ...     print(f"{browser['name']}: {browser['status']}")
        """
        self.logger.info("Listing browsers (type=%s)", browser_type)

        request_params = {"maxResults": max_results}
        if browser_type:
            request_params["type"] = browser_type
        if next_token:
            request_params["nextToken"] = next_token

        response = self.control_plane_client.list_browsers(**request_params)
        return response

    def start(
        self,
        identifier: Optional[str] = DEFAULT_IDENTIFIER,
        name: Optional[str] = None,
        session_timeout_seconds: Optional[int] = DEFAULT_SESSION_TIMEOUT,
        viewport: Optional[Dict[str, int]] = None,
    ) -> str:
        """Start a browser sandbox session.

        This method initializes a new browser session with the provided parameters.

        Args:
            identifier (Optional[str]): The browser sandbox identifier to use.
                Can be DEFAULT_IDENTIFIER or a custom browser ID from create_browser.
            name (Optional[str]): A name for this session.
            session_timeout_seconds (Optional[int]): The timeout for the session in seconds.
                Range: 1-28800 (8 hours). Default: 3600 (1 hour).
            viewport (Optional[Dict[str, int]]): The viewport dimensions:
                {'width': 1920, 'height': 1080}

        Returns:
            str: The session ID of the newly created session.

        Example:
            >>> # Use system browser
            >>> session_id = client.start()
            >>>
            >>> # Use custom browser with Web Bot Auth
            >>> session_id = client.start(
            ...     identifier="my-browser-abc123",
            ...     viewport={'width': 1920, 'height': 1080},
            ...     session_timeout_seconds=7200  # 2 hours
            ... )
        """
        self.logger.info("Starting browser session...")

        request_params = {
            "browserIdentifier": identifier,
            "name": name or f"browser-session-{uuid.uuid4().hex[:8]}",
            "sessionTimeoutSeconds": session_timeout_seconds,
        }

        if viewport is not None:
            request_params["viewPort"] = viewport

        response = self.data_plane_client.start_browser_session(**request_params)

        self.identifier = response["browserIdentifier"]
        self.session_id = response["sessionId"]

        self.logger.info("✅ Session started: %s", self.session_id)
        return self.session_id

    def stop(self) -> bool:
        """Stop the current browser session if one is active.

        Returns:
            bool: True if successful or no session was active.
        """
        self.logger.info("Stopping browser session...")

        if not self.session_id or not self.identifier:
            return True

        self.data_plane_client.stop_browser_session(browserIdentifier=self.identifier, sessionId=self.session_id)

        self.logger.info("✅ Session stopped: %s", self.session_id)
        self.identifier = None
        self.session_id = None
        return True

    def get_session(self, browser_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict:
        """Get detailed information about a browser session.

        Args:
            browser_id (Optional[str]): Browser identifier (uses current if not provided)
            session_id (Optional[str]): Session identifier (uses current if not provided)

        Returns:
            Dict: Session details including:
                - sessionId, browserIdentifier, name
                - status (READY, TERMINATED)
                - createdAt, lastUpdatedAt
                - sessionTimeoutSeconds
                - sessionReplayArtifact (S3 location if recording enabled)
                - streams (automationStream, liveViewStream)
                - viewPort

        Example:
            >>> session_info = client.get_session()
            >>> print(f"Session status: {session_info['status']}")
            >>> if session_info.get('sessionReplayArtifact'):
            ...     print(f"Recording available at: {session_info['sessionReplayArtifact']}")
        """
        browser_id = browser_id or self.identifier
        session_id = session_id or self.session_id

        if not browser_id or not session_id:
            raise ValueError("Browser ID and Session ID must be provided or available from current session")

        self.logger.info("Getting session: %s", session_id)

        response = self.data_plane_client.get_browser_session(browserIdentifier=browser_id, sessionId=session_id)
        return response

    def list_sessions(
        self,
        browser_id: Optional[str] = None,
        status: Optional[str] = None,
        max_results: int = 10,
        next_token: Optional[str] = None,
    ) -> Dict:
        """List browser sessions for a specific browser.

        Args:
            browser_id (Optional[str]): Browser identifier (uses current if not provided)
            status (Optional[str]): Filter by status: "READY" or "TERMINATED"
            max_results (int): Maximum results (1-100, default 10)
            next_token (Optional[str]): Pagination token

        Returns:
            Dict: Response containing:
                - items (List[Dict]): List of session summaries
                - nextToken (str): Token for next page (if more results)

        Example:
            >>> # List all active sessions
            >>> response = client.list_sessions(status="READY")
            >>> for session in response['items']:
            ...     print(f"Session {session['sessionId']}: {session['status']}")
        """
        browser_id = browser_id or self.identifier
        if not browser_id:
            raise ValueError("Browser ID must be provided or available from current session")

        self.logger.info("Listing sessions for browser: %s", browser_id)

        request_params = {"browserIdentifier": browser_id, "maxResults": max_results}
        if status:
            request_params["status"] = status
        if next_token:
            request_params["nextToken"] = next_token

        response = self.data_plane_client.list_browser_sessions(**request_params)
        return response

    def update_stream(
        self,
        stream_status: str,
        browser_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Update the browser automation stream status.

        This is the new UpdateBrowserStream API for dynamic stream control.

        Args:
            stream_status (str): Status to set: "ENABLED" or "DISABLED"
            browser_id (Optional[str]): Browser identifier (uses current if not provided)
            session_id (Optional[str]): Session identifier (uses current if not provided)

        Example:
            >>> # Disable automation to take manual control
            >>> client.update_stream("DISABLED")
            >>> # Re-enable automation
            >>> client.update_stream("ENABLED")
        """
        browser_id = browser_id or self.identifier
        session_id = session_id or self.session_id

        if not browser_id or not session_id:
            raise ValueError("Browser ID and Session ID must be provided or available from current session")

        self.logger.info("Updating stream status to: %s", stream_status)

        self.data_plane_client.update_browser_stream(
            browserIdentifier=browser_id,
            sessionId=session_id,
            streamUpdate={"automationStreamUpdate": {"streamStatus": stream_status}},
        )

    def generate_ws_headers(self) -> Tuple[str, Dict[str, str]]:
        """Generate the WebSocket headers needed for connecting to the browser sandbox.

        Returns:
            Tuple[str, Dict[str, str]]: A tuple containing the WebSocket URL and headers.

        Raises:
            RuntimeError: If no AWS credentials are found.
        """
        self.logger.info("Generating websocket headers...")

        if not self.identifier or not self.session_id:
            self.start()

        host = get_data_plane_endpoint(self.region).replace("https://", "")
        path = f"/browser-streams/{self.identifier}/sessions/{self.session_id}/automation"
        ws_url = f"wss://{host}{path}"

        boto_session = boto3.Session()
        credentials = boto_session.get_credentials()
        if not credentials:
            raise RuntimeError("No AWS credentials found")

        frozen_credentials = credentials.get_frozen_credentials()

        request = AWSRequest(
            method="GET",
            url=f"https://{host}{path}",
            headers={
                "host": host,
                "x-amz-date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            },
        )

        auth = SigV4Auth(frozen_credentials, "bedrock-agentcore", self.region)
        auth.add_auth(request)

        headers = {
            "Host": host,
            "X-Amz-Date": request.headers["x-amz-date"],
            "Authorization": request.headers["Authorization"],
            "Upgrade": "websocket",
            "Connection": "Upgrade",
            "Sec-WebSocket-Version": "13",
            "Sec-WebSocket-Key": base64.b64encode(secrets.token_bytes(16)).decode(),
            "User-Agent": f"BrowserSandbox-Client/1.0 (Session: {self.session_id})",
        }

        if frozen_credentials.token:
            headers["X-Amz-Security-Token"] = frozen_credentials.token

        return ws_url, headers

    def generate_live_view_url(self, expires: int = DEFAULT_LIVE_VIEW_PRESIGNED_URL_TIMEOUT) -> str:
        """Generate a pre-signed URL for viewing the browser session.

        Args:
            expires (int): Seconds until URL expires (max 300).

        Returns:
            str: The pre-signed URL for viewing.

        Raises:
            ValueError: If expires exceeds maximum.
            RuntimeError: If URL generation fails.
        """
        self.logger.info("Generating live view url...")

        if expires > MAX_LIVE_VIEW_PRESIGNED_URL_TIMEOUT:
            raise ValueError(
                f"Expiry timeout cannot exceed {MAX_LIVE_VIEW_PRESIGNED_URL_TIMEOUT} seconds, got {expires}"
            )

        if not self.identifier or not self.session_id:
            self.start()

        url = urlparse(
            f"{get_data_plane_endpoint(self.region)}/browser-streams/{self.identifier}/sessions/{self.session_id}/live-view"
        )
        boto_session = boto3.Session()
        credentials = boto_session.get_credentials().get_frozen_credentials()
        request = AWSRequest(method="GET", url=url.geturl(), headers={"host": url.hostname})
        signer = SigV4QueryAuth(
            credentials=credentials, service_name="bedrock-agentcore", region_name=self.region, expires=expires
        )
        signer.add_auth(request)

        if not request.url:
            raise RuntimeError("Failed to generate live view url")

        return request.url

    def take_control(self):
        """Take control of the browser by disabling automation stream."""
        self.logger.info("Taking control of browser session...")

        if not self.identifier or not self.session_id:
            self.start()

        if not self.identifier or not self.session_id:
            raise RuntimeError("Could not find or start a browser session")

        self.update_stream("DISABLED")

    def release_control(self):
        """Release control by enabling automation stream."""
        self.logger.info("Releasing control of browser session...")

        if not self.identifier or not self.session_id:
            self.logger.warning("Could not find a browser session when releasing control")
            return

        self.update_stream("ENABLED")


@contextmanager
def browser_session(
    region: str, viewport: Optional[Dict[str, int]] = None, identifier: Optional[str] = None
) -> Generator[BrowserClient, None, None]:
    """Context manager for creating and managing a browser sandbox session.

    Args:
        region (str): AWS region.
        viewport (Optional[Dict[str, int]]): Viewport dimensions.
        identifier (Optional[str]): Browser identifier (system or custom).

    Yields:
        BrowserClient: An initialized and started browser client.

    Example:
        >>> # Use system browser
        >>> with browser_session('us-west-2') as client:
        ...     ws_url, headers = client.generate_ws_headers()
        ...
        >>> # Use custom browser with Web Bot Auth
        >>> with browser_session('us-west-2', identifier='my-signed-browser') as client:
        ...     # Automation with reduced CAPTCHA friction
        ...     pass
    """
    client = BrowserClient(region)
    start_kwargs = {}
    if viewport is not None:
        start_kwargs["viewport"] = viewport
    if identifier is not None:
        start_kwargs["identifier"] = identifier

    client.start(**start_kwargs)

    try:
        yield client
    finally:
        client.stop()
