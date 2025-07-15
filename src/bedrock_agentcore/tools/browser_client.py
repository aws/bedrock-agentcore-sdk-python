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

from .._utils.endpoints import (
    get_data_plane_endpoint,
)

DEFAULT_IDENTIFIER = "aws.browser.v1"
DEFAULT_SESSION_TIMEOUT = 3600
DEFAULT_LIVE_VIEW_PRESIGNED_URL_TIMEOUT = 300


class BrowserClient:
    """Client for interacting with the AWS Browser sandbox service.

    This client handles the session lifecycle and browser automation for
    Browser sandboxes, providing an interface to perform web automation
    tasks in a secure, managed environment.

    Attributes:
        region (str): The AWS region being used.
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
        self.data_plane_service_name = "bedrock-agentcore"
        self.client = boto3.client(
            self.data_plane_service_name, region_name=region, endpoint_url=get_data_plane_endpoint(region)
        )
        self._identifier = None
        self._session_id = None
        self.logger = logging.getLogger(__name__)

    @property
    def identifier(self) -> Optional[str]:
        """Get the current browser identifier.

        Returns:
            Optional[str]: The current identifier or None if not set.
        """
        return self._identifier

    @identifier.setter
    def identifier(self, value: Optional[str]):
        """Set the browser identifier.

        Args:
            value (Optional[str]): The identifier to set.
        """
        self._identifier = value

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID.

        Returns:
            Optional[str]: The current session ID or None if not set.
        """
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]):
        """Set the session ID.

        Args:
            value (Optional[str]): The session ID to set.
        """
        self._session_id = value

    def start(
        self,
        identifier: Optional[str] = DEFAULT_IDENTIFIER,
        name: Optional[str] = None,
        session_timeout_seconds: Optional[int] = DEFAULT_SESSION_TIMEOUT,
    ) -> str:
        """Start a browser sandbox session.

        This method initializes a new browser session with the provided parameters.

        Args:
            identifier (Optional[str]): The browser sandbox identifier to use.
                Defaults to DEFAULT_IDENTIFIER.
            name (Optional[str]): A name for this session. If not provided, a name
                will be generated using a UUID.
            session_timeout_seconds (Optional[int]): The timeout for the session in seconds.
                Defaults to DEFAULT_TIMEOUT.
            description (Optional[str]): A description for this session.
                Defaults to an empty string.

        Returns:
            str: The session ID of the newly created session.
        """
        self.logger.info("Starting browser session...")

        response = self.client.start_browser_session(
            browserIdentifier=identifier,
            name=name or f"browser-session-{uuid.uuid4().hex[:8]}",
            sessionTimeoutSeconds=session_timeout_seconds,
        )

        self.identifier = response["browserIdentifier"]
        self.session_id = response["sessionId"]

        return self.session_id

    def stop(self):
        """Stop the current browser session if one is active.

        This method stops any active session and clears the session state.
        If no session is active, this method does nothing.

        Returns:
            bool: True if no session was active or the session was successfully stopped.
        """
        self.logger.info("Stopping browser session...")

        if not self.session_id or not self.identifier:
            return True

        self.client.stop_browser_session(**{"browserIdentifier": self.identifier, "sessionId": self.session_id})

        self.identifier = None
        self.session_id = None

    def generate_ws_headers(self) -> Tuple[str, Dict[str, str]]:
        """Generate the WebSocket headers needed for connecting to the browser sandbox.

        This method creates properly signed WebSocket headers for connecting to
        the browser automation endpoint.

        Returns:
            Tuple[str, Dict[str, str]]: A tuple containing the WebSocket URL and
                the headers dictionary.

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

        auth = SigV4Auth(frozen_credentials, self.data_plane_service_name, self.region)
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

        Creates a pre-signed URL that can be used to view the current browser session.
        If no session is active, a new session will be started.

        Args:
            expires (int, optional): The number of seconds until the pre-signed URL expires.
                Defaults to DEFAULT_LIVE_VIEW_PRESIGNED_URL_TIMEOUT (300 seconds).

        Returns:
            str: The pre-signed URL for viewing the browser session.

        Raises:
            RuntimeError: If the URL generation fails.
        """
        self.logger.info("Generating live view url...")

        if not self.identifier or not self.session_id:
            self.start()

        url = urlparse(
            f"{get_data_plane_endpoint(self.region)}/browser-streams/{self.identifier}/sessions/{self.session_id}/live-view"
        )
        boto_session = boto3.Session()
        credentials = boto_session.get_credentials().get_frozen_credentials()
        request = AWSRequest(method="GET", url=url.geturl(), headers={"host": url.hostname})
        signer = SigV4QueryAuth(
            credentials=credentials, service_name=self.data_plane_service_name, region_name=self.region, expires=expires
        )
        signer.add_auth(request)

        if not request.url:
            raise RuntimeError("Failed to generate live view url")

        return request.url

    def take_control(self):
        """Take control of the browser session by disabling the automation stream.

        This method disables external automation capabilities of the browser session,
        giving this client exclusive control. If no session is active, a new session
        will be started.

        Raises:
            RuntimeError: If a session could not be found or started.
        """
        self.logger.info("Taking control of browser session...")

        if not self.identifier or not self.session_id:
            self.start()

        if not self.identifier or not self.session_id:
            raise RuntimeError("Could not find or start a browser session")

        self._update_browser_stream(self.identifier, self.session_id, "DISABLED")

    def release_control(self):
        """Release control of the browser session by enabling the automation stream.

        This method enables external automation capabilities of the browser session,
        relinquishing exclusive control. If no session exists, a warning is logged
        and the method returns without taking action.
        """
        self.logger.info("Releasing control of browser session...")

        if not self.identifier or not self.session_id:
            self.logger.warning("Could not find a browser session when releasing control")
            return

        self._update_browser_stream(self.identifier, self.session_id, "ENABLED")

    def _update_browser_stream(self, identifier: str, session_id: str, stream_status: str) -> None:
        """Update the browser stream status.

        This private helper method updates the status of the browser automation stream.

        Args:
            identifier (str): The browser identifier.
            session_id (str): The session ID.
            stream_status (str): The status to set for the automation stream.
                Valid values are "ENABLED" or "DISABLED".
        """
        self.client.update_browser_stream(
            **{
                "browserIdentifier": identifier,
                "sessionId": session_id,
                "streamUpdate": {"automationStreamUpdate": {"streamStatus": stream_status}},
            }
        )


@contextmanager
def browser_session(region: str) -> Generator[BrowserClient, None, None]:
    """Context manager for creating and managing a browser sandbox session.

    This context manager handles creating a client, starting a session, and
    ensuring the session is properly cleaned up when the context exits.

    Args:
        region (str): The AWS region to use for the Browser service.

    Yields:
        BrowserClient: An initialized and started browser client.

    Example:
        >>> with browser_session('us-west-2') as client:
        ...     browser = client.get_browser_obj()
        ...     page = browser.new_page()
        ...     page.goto('https://example.com')
    """
    client = BrowserClient(region)
    client.start()

    try:
        yield client
    finally:
        client.stop()
