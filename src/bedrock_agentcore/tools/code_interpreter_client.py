"""Client for interacting with the Code Interpreter sandbox service.

This module provides a client for the AWS Code Interpreter sandbox, allowing
applications to start, stop, and invoke code execution in a managed sandbox environment.
"""

import logging
import uuid
from contextlib import contextmanager
from typing import Dict, Generator, Optional

import boto3

from bedrock_agentcore._utils.endpoints import get_control_plane_endpoint, get_data_plane_endpoint

DEFAULT_IDENTIFIER = "aws.codeinterpreter.v1"
DEFAULT_TIMEOUT = 900


class CodeInterpreter:
    """Client for interacting with the AWS Code Interpreter sandbox service.

    This client handles the session lifecycle and method invocation for
    Code Interpreter sandboxes, providing an interface to execute code
    in a secure, managed environment.

    Attributes:
        region (str): The AWS region being used.
        control_plane_client: The boto3 client for control plane operations.
        data_plane_service_name (str): AWS service name for the data plane.
        client: The boto3 client for interacting with the service.
        identifier (str, optional): The code interpreter identifier.
        session_id (str, optional): The active session ID.
    """

    def __init__(self, region: str, session: Optional[boto3.Session] = None) -> None:
        """Initialize a Code Interpreter client for the specified AWS region.

        Args:
            region (str): The AWS region to use.
            session (Optional[boto3.Session]): Optional boto3 session.
        """
        self.region = region
        self.logger = logging.getLogger(__name__)

        if session is None:
            session = boto3.Session()

        # Control plane client for interpreter management
        self.control_plane_client = session.client(
            "bedrock-agentcore-control",
            region_name=region,
            endpoint_url=get_control_plane_endpoint(region),
        )

        # Data plane client for session operations
        self.data_plane_client = session.client(
            "bedrock-agentcore",
            region_name=region,
            endpoint_url=get_data_plane_endpoint(region),
        )

        self._identifier = None
        self._session_id = None

    @property
    def identifier(self) -> Optional[str]:
        """Get the current code interpreter identifier."""
        return self._identifier

    @identifier.setter
    def identifier(self, value: Optional[str]):
        """Set the code interpreter identifier."""
        self._identifier = value

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]):
        """Set the session ID."""
        self._session_id = value

    def create_code_interpreter(
        self,
        name: str,
        execution_role_arn: str,
        network_configuration: Optional[Dict] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        client_token: Optional[str] = None,
    ) -> Dict:
        """Create a custom code interpreter with specific configuration.

        This is a control plane operation that provisions a new code interpreter
        with custom settings including VPC configuration.

        Args:
            name (str): The name for the code interpreter.
                Must match pattern [a-zA-Z][a-zA-Z0-9_]{0,47}
            execution_role_arn (str): IAM role ARN with permissions for interpreter operations
            network_configuration (Optional[Dict]): Network configuration:
                {
                    "networkMode": "PUBLIC" or "VPC",
                    "vpcConfig": {  # Required if networkMode is VPC
                        "securityGroups": ["sg-xxx"],
                        "subnets": ["subnet-xxx"]
                    }
                }
            description (Optional[str]): Description of the interpreter (1-4096 chars)
            tags (Optional[Dict[str, str]]): Tags for the interpreter
            client_token (Optional[str]): Idempotency token

        Returns:
            Dict: Response containing:
                - codeInterpreterArn (str): ARN of created interpreter
                - codeInterpreterId (str): Unique interpreter identifier
                - createdAt (datetime): Creation timestamp
                - status (str): Interpreter status (CREATING, READY, etc.)

        Example:
            >>> client = CodeInterpreter('us-west-2')
            >>> # Create interpreter with VPC
            >>> response = client.create_code_interpreter(
            ...     name="my_secure_interpreter",
            ...     execution_role_arn="arn:aws:iam::123456789012:role/InterpreterRole",
            ...     network_configuration={
            ...         "networkMode": "VPC",
            ...         "vpcConfig": {
            ...             "securityGroups": ["sg-12345"],
            ...             "subnets": ["subnet-abc123"]
            ...         }
            ...     },
            ...     description="Secure interpreter for data analysis"
            ... )
            >>> interpreter_id = response['codeInterpreterId']
        """
        self.logger.info("Creating code interpreter: %s", name)

        request_params = {
            "name": name,
            "executionRoleArn": execution_role_arn,
            "networkConfiguration": network_configuration or {"networkMode": "PUBLIC"},
        }

        if description:
            request_params["description"] = description

        if tags:
            request_params["tags"] = tags

        if client_token:
            request_params["clientToken"] = client_token

        response = self.control_plane_client.create_code_interpreter(**request_params)
        return response

    def delete_code_interpreter(self, interpreter_id: str, client_token: Optional[str] = None) -> Dict:
        """Delete a custom code interpreter.

        Args:
            interpreter_id (str): The code interpreter identifier to delete
            client_token (Optional[str]): Idempotency token

        Returns:
            Dict: Response containing:
                - codeInterpreterId (str): ID of deleted interpreter
                - lastUpdatedAt (datetime): Update timestamp
                - status (str): Deletion status

        Example:
            >>> client.delete_code_interpreter("my-interpreter-abc123")
        """
        self.logger.info("Deleting code interpreter: %s", interpreter_id)

        request_params = {"codeInterpreterId": interpreter_id}
        if client_token:
            request_params["clientToken"] = client_token

        response = self.control_plane_client.delete_code_interpreter(**request_params)
        return response

    def get_code_interpreter(self, interpreter_id: str) -> Dict:
        """Get detailed information about a code interpreter.

        Args:
            interpreter_id (str): The code interpreter identifier

        Returns:
            Dict: Interpreter details including:
                - codeInterpreterArn, codeInterpreterId, name, description
                - createdAt, lastUpdatedAt
                - executionRoleArn
                - networkConfiguration
                - status (CREATING, CREATE_FAILED, READY, DELETING, etc.)
                - failureReason (if failed)

        Example:
            >>> interpreter_info = client.get_code_interpreter("my-interpreter-abc123")
            >>> print(f"Status: {interpreter_info['status']}")
        """
        self.logger.info("Getting code interpreter: %s", interpreter_id)
        response = self.control_plane_client.get_code_interpreter(codeInterpreterId=interpreter_id)
        return response

    def list_code_interpreters(
        self,
        interpreter_type: Optional[str] = None,
        max_results: int = 10,
        next_token: Optional[str] = None,
    ) -> Dict:
        """List all code interpreters in the account.

        Args:
            interpreter_type (Optional[str]): Filter by type: "SYSTEM" or "CUSTOM"
            max_results (int): Maximum results to return (1-100, default 10)
            next_token (Optional[str]): Token for pagination

        Returns:
            Dict: Response containing:
                - codeInterpreterSummaries (List[Dict]): List of interpreter summaries
                - nextToken (str): Token for next page (if more results)

        Example:
            >>> # List all custom interpreters
            >>> response = client.list_code_interpreters(interpreter_type="CUSTOM")
            >>> for interp in response['codeInterpreterSummaries']:
            ...     print(f"{interp['name']}: {interp['status']}")
        """
        self.logger.info("Listing code interpreters (type=%s)", interpreter_type)

        request_params = {"maxResults": max_results}
        if interpreter_type:
            request_params["type"] = interpreter_type
        if next_token:
            request_params["nextToken"] = next_token

        response = self.control_plane_client.list_code_interpreters(**request_params)
        return response

    def start(
        self,
        identifier: Optional[str] = DEFAULT_IDENTIFIER,
        name: Optional[str] = None,
        session_timeout_seconds: Optional[int] = DEFAULT_TIMEOUT,
    ) -> str:
        """Start a code interpreter sandbox session.

        Args:
            identifier (Optional[str]): The interpreter identifier to use.
                Can be DEFAULT_IDENTIFIER or a custom interpreter ID from create_code_interpreter.
            name (Optional[str]): A name for this session.
            session_timeout_seconds (Optional[int]): The timeout in seconds.
                Default: 900 (15 minutes).

        Returns:
            str: The session ID of the newly created session.

        Example:
            >>> # Use system interpreter
            >>> session_id = client.start()
            >>>
            >>> # Use custom interpreter with VPC
            >>> session_id = client.start(
            ...     identifier="my-interpreter-abc123",
            ...     session_timeout_seconds=1800  # 30 minutes
            ... )
        """
        self.logger.info("Starting code interpreter session...")

        response = self.data_plane_client.start_code_interpreter_session(
            codeInterpreterIdentifier=identifier,
            name=name or f"code-session-{uuid.uuid4().hex[:8]}",
            sessionTimeoutSeconds=session_timeout_seconds,
        )

        self.identifier = response["codeInterpreterIdentifier"]
        self.session_id = response["sessionId"]

        self.logger.info("✅ Session started: %s", self.session_id)
        return self.session_id

    def stop(self) -> bool:
        """Stop the current code interpreter session if one is active.

        Returns:
            bool: True if successful or no session was active.
        """
        self.logger.info("Stopping code interpreter session...")

        if not self.session_id or not self.identifier:
            return True

        self.data_plane_client.stop_code_interpreter_session(
            codeInterpreterIdentifier=self.identifier, sessionId=self.session_id
        )

        self.logger.info("✅ Session stopped: %s", self.session_id)
        self.identifier = None
        self.session_id = None
        return True

    def get_session(self, interpreter_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict:
        """Get detailed information about a code interpreter session.

        Args:
            interpreter_id (Optional[str]): Interpreter ID (uses current if not provided)
            session_id (Optional[str]): Session ID (uses current if not provided)

        Returns:
            Dict: Session details including:
                - sessionId, codeInterpreterIdentifier, name
                - status (READY, TERMINATED)
                - createdAt, lastUpdatedAt
                - sessionTimeoutSeconds

        Example:
            >>> session_info = client.get_session()
            >>> print(f"Session status: {session_info['status']}")
        """
        interpreter_id = interpreter_id or self.identifier
        session_id = session_id or self.session_id

        if not interpreter_id or not session_id:
            raise ValueError("Interpreter ID and Session ID must be provided or available from current session")

        self.logger.info("Getting session: %s", session_id)

        response = self.data_plane_client.get_code_interpreter_session(
            codeInterpreterIdentifier=interpreter_id, sessionId=session_id
        )
        return response

    def list_sessions(
        self,
        interpreter_id: Optional[str] = None,
        status: Optional[str] = None,
        max_results: int = 10,
        next_token: Optional[str] = None,
    ) -> Dict:
        """List code interpreter sessions for a specific interpreter.

        Args:
            interpreter_id (Optional[str]): Interpreter ID (uses current if not provided)
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
        interpreter_id = interpreter_id or self.identifier
        if not interpreter_id:
            raise ValueError("Interpreter ID must be provided or available from current session")

        self.logger.info("Listing sessions for interpreter: %s", interpreter_id)

        request_params = {"codeInterpreterIdentifier": interpreter_id, "maxResults": max_results}
        if status:
            request_params["status"] = status
        if next_token:
            request_params["nextToken"] = next_token

        response = self.data_plane_client.list_code_interpreter_sessions(**request_params)
        return response

    def invoke(self, method: str, params: Optional[Dict] = None):
        r"""Invoke a method in the code interpreter sandbox.

        If no session is active, automatically starts a new session.

        Args:
            method (str): The name of the method to invoke.
            params (Optional[Dict]): Parameters to pass to the method.

        Returns:
            dict: The response from the code interpreter service.

        Example:
            >>> # List files in the sandbox
            >>> result = client.invoke('listFiles')
            >>>
            >>> # Execute Python code
            >>> code = "import pandas as pd\\ndf = pd.DataFrame({'a': [1,2,3]})\\nprint(df)"
            >>> result = client.invoke('execute', {'code': code})
        """
        if not self.session_id or not self.identifier:
            self.start()

        return self.data_plane_client.invoke_code_interpreter(
            codeInterpreterIdentifier=self.identifier,
            sessionId=self.session_id,
            name=method,
            arguments=params or {},
        )


@contextmanager
def code_session(
    region: str, session: Optional[boto3.Session] = None, identifier: Optional[str] = None
) -> Generator[CodeInterpreter, None, None]:
    """Context manager for creating and managing a code interpreter session.

    Args:
        region (str): AWS region.
        session (Optional[boto3.Session]): Optional boto3 session.
        identifier (Optional[str]): Interpreter identifier (system or custom).

    Yields:
        CodeInterpreter: An initialized and started code interpreter client.

    Example:
        >>> # Use system interpreter
        >>> with code_session('us-west-2') as client:
        ...     result = client.invoke('listFiles')
        ...
        >>> # Use custom VPC interpreter
        >>> with code_session('us-west-2', identifier='my-secure-interpreter') as client:
        ...     # Secure data analysis
        ...     pass
    """
    client = CodeInterpreter(region, session=session)
    if identifier is not None:
        client.start(identifier=identifier)
    else:
        client.start()

    try:
        yield client
    finally:
        client.stop()
