"""Client for interacting with the Code Interpreter sandbox service.

This module provides a client for the AWS Code Interpreter sandbox, allowing
applications to start, stop, and invoke code execution in a managed sandbox environment.
"""

import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generator, Optional, Union
from time import sleep
import random

import boto3
from botocore.exceptions import ClientError, NoCredentialsError, EndpointConnectionError

from bedrock_agentcore._utils.endpoints import get_data_plane_endpoint

# Constants
DEFAULT_IDENTIFIER = "aws.codeinterpreter.v1"
DEFAULT_TIMEOUT = 900
DEFAULT_SESSION_NAME_PREFIX = "code-session"
SERVICE_NAME = "bedrock-agentcore"

# Retry configuration
MAX_RETRIES = 3
BASE_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 10.0

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Enumeration of possible session states."""
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class SessionInfo:
    """Container for session information."""
    identifier: Optional[str] = None
    session_id: Optional[str] = None
    state: SessionState = SessionState.INACTIVE
    
    @property
    def is_active(self) -> bool:
        """Check if session is in an active state."""
        return self.state == SessionState.ACTIVE and self.identifier and self.session_id
    
    def reset(self) -> None:
        """Reset session information to inactive state."""
        self.identifier = None
        self.session_id = None
        self.state = SessionState.INACTIVE


class CodeInterpreterError(Exception):
    """Base exception for Code Interpreter client errors."""
    pass


class SessionError(CodeInterpreterError):
    """Exception raised for session-related errors."""
    pass


class InvocationError(CodeInterpreterError):
    """Exception raised for method invocation errors."""
    pass


class CodeInterpreter:
    """Client for interacting with the AWS Code Interpreter sandbox service.

    This client handles the session lifecycle and method invocation for
    Code Interpreter sandboxes, providing an interface to execute code
    in a secure, managed environment.

    Attributes:
        region (str): The AWS region for the service.
        session (SessionInfo): Current session information.
    """

    def __init__(self, region: str) -> None:
        """Initialize a Code Interpreter client for the specified AWS region.

        Args:
            region (str): The AWS region to use for the Code Interpreter service.
            
        Raises:
            ValueError: If region is invalid.
            CodeInterpreterError: If client initialization fails.
        """
        if not region or not isinstance(region, str):
            raise ValueError("Region must be a non-empty string")
        
        self.region = region
        self.session = SessionInfo()
        self._client: Optional[boto3.client] = None
        
        # For backward compatibility, create the client immediately
        # This matches the original behavior expected by tests
        try:
            endpoint_url = get_data_plane_endpoint(self.region)
            self._client = boto3.client(
                SERVICE_NAME,
                region_name=self.region,
                endpoint_url=endpoint_url
            )
            logger.debug(f"Created boto3 client for endpoint: {endpoint_url}")
        except Exception as e:
            logger.warning(f"Failed to create boto3 client during init: {e}")
            # Don't raise here to maintain backward compatibility
        
        logger.debug(f"Initialized CodeInterpreter client for region: {region}")

    @property
    def client(self) -> boto3.client:
        """Get or create the boto3 client (lazy initialization).
        
        Returns:
            boto3.client: The initialized boto3 client.
            
        Raises:
            CodeInterpreterError: If client creation fails.
        """
        if self._client is None:
            try:
                endpoint_url = get_data_plane_endpoint(self.region)
                self._client = boto3.client(
                    SERVICE_NAME,
                    region_name=self.region,
                    endpoint_url=endpoint_url
                )
                logger.debug(f"Created boto3 client for endpoint: {endpoint_url}")
            except Exception as e:
                raise CodeInterpreterError(f"Failed to create boto3 client: {e}") from e
        
        return self._client

    @client.setter
    def client(self, value: boto3.client) -> None:
        """Set the boto3 client (for testing purposes).
        
        Args:
            value: The boto3 client to set.
        """
        self._client = value

    @property
    def identifier(self) -> Optional[str]:
        """Get the current code interpreter identifier.

        Returns:
            Optional[str]: The current identifier or None if not set.
        """
        return self.session.identifier

    @identifier.setter
    def identifier(self, value: Optional[str]) -> None:
        """Set the code interpreter identifier.

        Args:
            value: The identifier to set.
        """
        self.session.identifier = value
        # Update session state based on identifier and session_id
        if value is None:
            self.session.state = SessionState.INACTIVE
        elif value is not None and self.session.session_id is not None:
            # Both identifier and session_id are set, mark as active for backward compatibility
            self.session.state = SessionState.ACTIVE

    @property
    def session_id(self) -> Optional[str]:
        """Get the current session ID.

        Returns:
            Optional[str]: The current session ID or None if not set.
        """
        return self.session.session_id

    @session_id.setter
    def session_id(self, value: Optional[str]) -> None:
        """Set the session ID.

        Args:
            value: The session ID to set.
        """
        self.session.session_id = value
        # Update session state based on identifier and session_id
        if value is None:
            self.session.state = SessionState.INACTIVE
        elif value is not None and self.session.identifier is not None:
            # Both identifier and session_id are set, mark as active for backward compatibility
            self.session.state = SessionState.ACTIVE

    @property
    def is_active(self) -> bool:
        """Check if there's an active session.
        
        Returns:
            bool: True if session is active, False otherwise.
        """
        return self.session.is_active

    def _retry_with_backoff(self, operation, *args, **kwargs) -> Any:
        """Execute an operation with exponential backoff retry logic.
        
        Args:
            operation: The operation to retry.
            *args: Positional arguments for the operation.
            **kwargs: Keyword arguments for the operation.
            
        Returns:
            Any: The result of the successful operation.
            
        Raises:
            CodeInterpreterError: If all retries are exhausted.
        """
        last_exception = None
        
        for attempt in range(MAX_RETRIES):
            try:
                return operation(*args, **kwargs)
            except (ClientError, EndpointConnectionError) as e:
                last_exception = e
                if attempt == MAX_RETRIES - 1:
                    break
                
                # Calculate delay with jitter
                delay = min(BASE_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                jitter = random.uniform(0, delay * 0.1)
                total_delay = delay + jitter
                
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {total_delay:.2f}s"
                )
                sleep(total_delay)
        
        raise CodeInterpreterError(
            f"Operation failed after {MAX_RETRIES} attempts: {last_exception}"
        ) from last_exception

    def _generate_session_name(self, custom_name: Optional[str] = None) -> str:
        """Generate a session name.
        
        Args:
            custom_name: Optional custom name to use.
            
        Returns:
            str: The generated or provided session name.
        """
        if custom_name:
            return custom_name
        
        return f"{DEFAULT_SESSION_NAME_PREFIX}-{uuid.uuid4().hex[:8]}"

    def _validate_session_params(
        self,
        identifier: Optional[str],
        session_timeout_seconds: Optional[int]
    ) -> None:
        """Validate session parameters.
        
        Args:
            identifier: The session identifier to validate.
            session_timeout_seconds: The timeout to validate.
            
        Raises:
            ValueError: If parameters are invalid.
        """
        if identifier is not None and not isinstance(identifier, str):
            raise ValueError("Identifier must be a string")
        
        if session_timeout_seconds is not None:
            if not isinstance(session_timeout_seconds, int) or session_timeout_seconds <= 0:
                raise ValueError("Session timeout must be a positive integer")

    def start(
        self,
        identifier: Optional[str] = DEFAULT_IDENTIFIER,
        name: Optional[str] = None,
        session_timeout_seconds: Optional[int] = DEFAULT_TIMEOUT,
    ) -> str:
        """Start a code interpreter sandbox session.

        This method initializes a new code interpreter session with the provided parameters.

        Args:
            identifier: The code interpreter sandbox identifier to use.
                Defaults to DEFAULT_IDENTIFIER.
            name: A name for this session. If not provided, a name
                will be generated using a UUID.
            session_timeout_seconds: The timeout for the session in seconds.
                Defaults to DEFAULT_TIMEOUT.

        Returns:
            str: The session ID of the newly created session.
            
        Raises:
            SessionError: If session creation fails.
            ValueError: If parameters are invalid.
        """
        self._validate_session_params(identifier, session_timeout_seconds)
        
        if self.session.is_active:
            logger.warning("Session already active, stopping existing session")
            self.stop()
        
        self.session.state = SessionState.STARTING
        session_name = self._generate_session_name(name)
        
        try:
            logger.info(f"Starting session with identifier: {identifier}")
            
            def _start_session():
                return self.client.start_code_interpreter_session(
                    codeInterpreterIdentifier=identifier,
                    name=session_name,
                    sessionTimeoutSeconds=session_timeout_seconds,
                )
            
            response = self._retry_with_backoff(_start_session)
            
            self.session.identifier = response["codeInterpreterIdentifier"]
            self.session.session_id = response["sessionId"]
            self.session.state = SessionState.ACTIVE
            
            logger.info(f"Session started successfully: {self.session.session_id}")
            return self.session.session_id
            
        except Exception as e:
            self.session.state = SessionState.ERROR
            error_msg = f"Failed to start session: {e}"
            logger.error(error_msg)
            raise SessionError(error_msg) from e

    def stop(self) -> bool:
        """Stop the current code interpreter session if one is active.

        This method stops any active session and clears the session state.
        If no session is active, this method does nothing.

        Returns:
            bool: True if no session was active or the session was successfully stopped.
            
        Raises:
            SessionError: If session termination fails.
        """
        if not self.session.identifier or not self.session.session_id:
            logger.debug("No active session to stop")
            return True

        self.session.state = SessionState.STOPPING
        
        try:
            logger.info(f"Stopping session: {self.session.session_id}")
            
            def _stop_session():
                return self.client.stop_code_interpreter_session(
                    codeInterpreterIdentifier=self.session.identifier,
                    sessionId=self.session.session_id
                )
            
            self._retry_with_backoff(_stop_session)
            logger.info("Session stopped successfully")
            
        except Exception as e:
            error_msg = f"Failed to stop session: {e}"
            logger.error(error_msg)
            # Still reset session state even if stop fails
            self.session.reset()
            raise SessionError(error_msg) from e
        
        finally:
            self.session.reset()
        
        return True

    def invoke(
        self, 
        method: str, 
        params: Optional[Dict[str, Any]] = None,
        auto_start: bool = True
    ) -> Dict[str, Any]:
        """Invoke a method in the code interpreter sandbox.

        Args:
            method: The name of the method to invoke in the sandbox.
            params: Parameters to pass to the method. Defaults to None.
            auto_start: Whether to automatically start a session if none exists.

        Returns:
            dict: The response from the code interpreter service.
            
        Raises:
            InvocationError: If method invocation fails.
            SessionError: If no session is active and auto_start is False.
            ValueError: If method name is invalid.
        """
        if not method or not isinstance(method, str):
            raise ValueError("Method must be a non-empty string")
        
        if not self.session.is_active:
            if auto_start:
                logger.info("No active session, starting new session")
                self.start()
            else:
                raise SessionError("No active session and auto_start is disabled")
        
        try:
            logger.debug(f"Invoking method: {method}")
            
            def _invoke_method():
                return self.client.invoke_code_interpreter(
                    codeInterpreterIdentifier=self.session.identifier,
                    sessionId=self.session.session_id,
                    name=method,
                    arguments=params or {},
                )
            
            response = self._retry_with_backoff(_invoke_method)
            logger.debug(f"Method {method} invoked successfully")
            return response
            
        except Exception as e:
            error_msg = f"Failed to invoke method '{method}': {e}"
            logger.error(error_msg)
            raise InvocationError(error_msg) from e

    def __enter__(self) -> 'CodeInterpreter':
        """Enter the context manager by starting a session.
        
        Returns:
            CodeInterpreter: This instance.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager by stopping the session.
        
        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        try:
            self.stop()
        except Exception as e:
            logger.error(f"Error during context manager cleanup: {e}")
            # Don't suppress the original exception

    def __repr__(self) -> str:
        """Return a string representation of the client.
        
        Returns:
            str: String representation.
        """
        return (
            f"CodeInterpreter(region='{self.region}', "
            f"state={self.session.state.value}, "
            f"session_id='{self.session.session_id or 'None'}')"
        )


@contextmanager
def code_session(region: str, **kwargs) -> Generator[CodeInterpreter, None, None]:
    """Context manager for creating and managing a code interpreter session.

    This context manager handles creating a client, starting a session, and
    ensuring the session is properly cleaned up when the context exits.

    Args:
        region: The AWS region to use for the Code Interpreter service.
        **kwargs: Additional arguments to pass to the start() method.

    Yields:
        CodeInterpreter: An initialized and started code interpreter client.

    Example:
        >>> with code_session('us-west-2') as client:
        ...     result = client.invoke('listFiles')
        ...     # Process result here
        
    Raises:
        CodeInterpreterError: If client creation or session management fails.
    """
    client = CodeInterpreter(region)
    
    try:
        client.start(**kwargs)
        yield client
    finally:
        try:
            client.stop()
        except Exception as e:
            logger.error(f"Error cleaning up session in context manager: {e}")
