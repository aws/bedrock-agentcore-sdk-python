"""Client for interacting with the Code Interpreter sandbox service.

This module provides a client for the AWS Code Interpreter sandbox, allowing
applications to start, stop, and invoke code execution in a managed sandbox environment.
"""

import uuid
from contextlib import contextmanager
from typing import Dict, Generator, Optional

import boto3

from bedrock_agentcore._utils.endpoints import get_data_plane_endpoint

DEFAULT_IDENTIFIER = "aws.codeinterpreter.v1"
DEFAULT_TIMEOUT = 900


class CodeInterpreter:
    """Client for interacting with the AWS Code Interpreter sandbox service.

    This client handles the session lifecycle and method invocation for
    Code Interpreter sandboxes, providing an interface to execute code
    in a secure, managed environment.

    Attributes:
        data_plane_service_name (str): AWS service name for the data plane.
        client: The boto3 client for interacting with the service.
        identifier (str, optional): The code interpreter identifier.
        session_id (str, optional): The active session ID.
    """

    def __init__(self, region: str) -> None:
        """Initialize a Code Interpreter client for the specified AWS region.

        Args:
            region (str): The AWS region to use for the Code Interpreter service.
        """
        self.data_plane_service_name = "bedrock-agentcore"
        self.client = boto3.client(
            self.data_plane_service_name, region_name=region, endpoint_url=get_data_plane_endpoint(region)
        )
        self._identifier = None
        self._session_id = None

    @property
    def identifier(self) -> Optional[str]:
        """Get the current code interpreter identifier.

        Returns:
            Optional[str]: The current identifier or None if not set.
        """
        return self._identifier

    @identifier.setter
    def identifier(self, value: Optional[str]):
        """Set the code interpreter identifier.

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
        session_timeout_seconds: Optional[int] = DEFAULT_TIMEOUT,
    ) -> str:
        """Start a code interpreter sandbox session.

        This method initializes a new code interpreter session with the provided parameters.

        Args:
            identifier (Optional[str]): The code interpreter sandbox identifier to use.
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
        response = self.client.start_code_interpreter_session(
            codeInterpreterIdentifier=identifier,
            name=name or f"code-session-{uuid.uuid4().hex[:8]}",
            sessionTimeoutSeconds=session_timeout_seconds,
        )

        self.identifier = response["codeInterpreterIdentifier"]
        self.session_id = response["sessionId"]

        return self.session_id

    def stop(self):
        """Stop the current code interpreter session if one is active.

        This method stops any active session and clears the session state.
        If no session is active, this method does nothing.

        Returns:
            bool: True if no session was active or the session was successfully stopped.
        """
        if not self.session_id or not self.identifier:
            return True

        self.client.stop_code_interpreter_session(
            **{"codeInterpreterIdentifier": self.identifier, "sessionId": self.session_id}
        )

        self.identifier = None
        self.session_id = None

    def invoke(self, method: str, params: Optional[Dict] = None):
        """Invoke a method in the code interpreter sandbox.

        If no session is active, this method automatically starts a new session
        before invoking the requested method.

        Args:
            method (str): The name of the method to invoke in the sandbox.
            params (Optional[Dict]): Parameters to pass to the method. Defaults to None.
            request_id (Optional[str]): A custom request ID. If not provided, a unique ID is generated.

        Returns:
            dict: The response from the code interpreter service.
        """
        if not self.session_id or not self.identifier:
            self.start()

        return self.client.invoke_code_interpreter(
            **{
                "codeInterpreterIdentifier": self.identifier,
                "sessionId": self.session_id,
                "name": method,
                "arguments": params or {},
            }
        )


@contextmanager
def code_session(region: str) -> Generator[CodeInterpreter, None, None]:
    """Context manager for creating and managing a code interpreter session.

    This context manager handles creating a client, starting a session, and
    ensuring the session is properly cleaned up when the context exits.

    Args:
        region (str): The AWS region to use for the Code Interpreter service.

    Yields:
        CodeInterpreterClient: An initialized and started code interpreter client.

    Example:
        >>> with code_session('us-west-2') as client:
        ...     result = client.invoke('listFiles')
        ...     # Process result here
    """
    client = CodeInterpreter(region)
    client.start()

    try:
        yield client
    finally:
        client.stop()
