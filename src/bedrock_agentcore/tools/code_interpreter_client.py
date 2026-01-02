"""Client for interacting with the Code Interpreter sandbox service.

This module provides a client for the AWS Code Interpreter sandbox, allowing
applications to start, stop, and invoke code execution in a managed sandbox environment.
"""

import base64
import logging
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Union

import boto3
from botocore.config import Config

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

    Basic Usage:
        >>> from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
        >>>
        >>> client = CodeInterpreter('us-west-2')
        >>> client.start()
        >>>
        >>> # Execute code
        >>> result = client.execute_code("print('Hello, World!')")
        >>>
        >>> # Install packages
        >>> client.install_packages(['pandas', 'matplotlib'])
        >>>
        >>> # Upload and process data
        >>> client.upload_file('data.csv', csv_content, description='Sales data')
        >>>
        >>> client.stop()

    Context Manager Usage:
        >>> from bedrock_agentcore.tools.code_interpreter_client import code_session
        >>>
        >>> with code_session('us-west-2') as client:
        ...     client.install_packages(['numpy'])
        ...     result = client.execute_code('import numpy as np; print(np.pi)')
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
            config=Config(read_timeout=300),
        )

        self._identifier = None
        self._session_id = None
        self._file_descriptions: Dict[str, str] = {}

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

    def upload_file(
        self,
        path: str,
        content: Union[str, bytes],
        description: str = "",
    ) -> Dict[str, Any]:
        r"""Upload a file to the code interpreter environment.

        This is a convenience wrapper around the writeFiles method that provides
        a cleaner interface for file uploads with optional semantic descriptions.

        Args:
            path: Relative path where the file should be saved (e.g., 'data.csv',
                'scripts/analysis.py'). Must be relative to the working directory.
                Absolute paths starting with '/' are not allowed.
            content: File content as string (text files) or bytes (binary files).
                    Binary content will be base64 encoded automatically.
            description: Optional semantic description of the file contents.
                        This is stored as metadata and can help LLMs understand
                        the data structure (e.g., "CSV with columns: date, revenue, product_id").

        Returns:
            Dict containing the result of the write operation.

        Raises:
            ValueError: If path is absolute or content type is invalid.

        Example:
            >>> # Upload a CSV file
            >>> client.upload_file(
            ...     path='sales_data.csv',
            ...     content='date,revenue\n2024-01-01,1000\n2024-01-02,1500',
            ...     description='Daily sales data with columns: date, revenue'
            ... )

            >>> # Upload a Python script
            >>> client.upload_file(
            ...     path='scripts/analyze.py',
            ...     content='import pandas as pd\ndf = pd.read_csv("sales_data.csv")'
            ... )
        """
        if path.startswith("/"):
            raise ValueError(
                f"Path must be relative, not absolute. Got: {path}. Use paths like 'data.csv' or 'scripts/analysis.py'."
            )

        # Handle binary content
        if isinstance(content, bytes):
            file_content = {"path": path, "blob": base64.b64encode(content).decode("utf-8")}
        else:
            file_content = {"path": path, "text": content}

        if description:
            self.logger.info("Uploading file: %s (%s)", path, description)
        else:
            self.logger.info("Uploading file: %s", path)

        result = self.invoke("writeFiles", {"content": [file_content]})

        # Store description as metadata (available for future LLM context)
        if description:
            self._file_descriptions[path] = description

        return result

    def upload_files(
        self,
        files: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Upload multiple files to the code interpreter environment.

        This operation is atomic - either all files are written or none are.
        If any file fails, the entire operation fails.

        Args:
            files: List of file specifications, each containing:
                - 'path': Relative file path
                - 'content': File content (string or bytes)
                - 'description': Optional semantic description

        Returns:
            Dict containing the result of the write operation.

        Example:
            >>> client.upload_files([
            ...     {'path': 'data.csv', 'content': csv_data, 'description': 'Sales data'},
            ...     {'path': 'config.json', 'content': json_config}
            ... ])
        """
        file_contents = []
        for file_spec in files:
            path = file_spec["path"]
            content = file_spec["content"]

            if path.startswith("/"):
                raise ValueError(f"Path must be relative, not absolute. Got: {path}")

            if isinstance(content, bytes):
                file_contents.append({"path": path, "blob": base64.b64encode(content).decode("utf-8")})
            else:
                file_contents.append({"path": path, "text": content})

        self.logger.info("Uploading %d files", len(files))
        return self.invoke("writeFiles", {"content": file_contents})

    def install_packages(
        self,
        packages: List[str],
        upgrade: bool = False,
    ) -> Dict[str, Any]:
        """Install Python packages in the code interpreter environment.

        This is a convenience wrapper around executeCommand that handles
        pip install commands with proper formatting.

        Args:
            packages: List of package names to install. Can include version
                    specifiers (e.g., ['pandas>=2.0', 'numpy', 'scikit-learn==1.3.0']).
            upgrade: If True, adds --upgrade flag to update existing packages.

        Returns:
            Dict containing the command execution result with stdout/stderr.

        Example:
            >>> # Install multiple packages
            >>> client.install_packages(['pandas', 'matplotlib', 'scikit-learn'])

            >>> # Install with version constraints
            >>> client.install_packages(['pandas>=2.0', 'numpy<2.0'])

            >>> # Upgrade existing packages
            >>> client.install_packages(['pandas'], upgrade=True)
        """
        if not packages:
            raise ValueError("At least one package name must be provided")

        # Sanitize package names (basic validation)
        for pkg in packages:
            if any(char in pkg for char in [";", "&", "|", "`", "$"]):
                raise ValueError(f"Invalid characters in package name: {pkg}")

        packages_str = " ".join(packages)
        upgrade_flag = "--upgrade " if upgrade else ""
        command = f"pip install {upgrade_flag}{packages_str}"

        self.logger.info("Installing packages: %s", packages_str)
        return self.invoke("executeCommand", {"command": command})

    def download_file(
        self,
        path: str,
    ) -> str:
        """Download/read a file from the code interpreter environment.

        Args:
            path: Path to the file to read.

        Returns:
            File content as string.

        Raises:
            FileNotFoundError: If the file doesn't exist.

        Example:
            >>> # Read a generated file
            >>> content = client.download_file('output/results.csv')
            >>> print(content)
        """
        self.logger.info("Downloading file: %s", path)
        result = self.invoke("readFiles", {"paths": [path]})

        # Parse the response to extract file content
        # Response structure from the API
        if "stream" in result:
            for event in result["stream"]:
                if "result" in event:
                    for content_item in event["result"].get("content", []):
                        if content_item.get("type") == "resource":
                            resource = content_item.get("resource", {})
                            if "text" in resource:
                                return resource["text"]
                            elif "blob" in resource:
                                return base64.b64decode(resource["blob"]).decode("utf-8")

        raise FileNotFoundError(f"Could not read file: {path}")

    def download_files(
        self,
        paths: List[str],
    ) -> Dict[str, str]:
        """Download/read multiple files from the code interpreter environment.

        Args:
            paths: List of file paths to read.

        Returns:
            Dict mapping file paths to their contents.

        Example:
            >>> files = client.download_files(['data.csv', 'results.json'])
            >>> print(files['data.csv'])
        """
        self.logger.info("Downloading %d files", len(paths))
        result = self.invoke("readFiles", {"paths": paths})

        files = {}
        if "stream" in result:
            for event in result["stream"]:
                if "result" in event:
                    for content_item in event["result"].get("content", []):
                        if content_item.get("type") == "resource":
                            resource = content_item.get("resource", {})
                            uri = resource.get("uri", "")
                            file_path = uri.replace("file://", "")

                            if "text" in resource:
                                files[file_path] = resource["text"]
                            elif "blob" in resource:
                                files[file_path] = base64.b64decode(resource["blob"]).decode("utf-8")

        return files

    def execute_code(
        self,
        code: str,
        language: str = "python",
        clear_context: bool = False,
    ) -> Dict[str, Any]:
        """Execute code in the interpreter environment.

        This is a convenience wrapper around the executeCode method with
        typed parameters for better IDE support and validation.

        Args:
            code: The code to execute.
            language: Programming language - 'python', 'javascript', or 'typescript'.
                    Default is 'python'.
            clear_context: If True, clears all previous variable state before execution.
                        Default is False (variables persist across calls).
                        Note: Only supported for Python. Ignored for JavaScript/TypeScript.

        Returns:
            Dict containing execution results including stdout, stderr, exit_code.

        Example:
            >>> # Execute Python code
            >>> result = client.execute_code('''
            ... import pandas as pd
            ... df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            ... print(df.describe())
            ... ''')

            >>> # Clear context and start fresh
            >>> result = client.execute_code('x = 10', clear_context=True)
        """
        valid_languages = ["python", "javascript", "typescript"]
        if language not in valid_languages:
            raise ValueError(f"Language must be one of {valid_languages}, got: {language}")

        self.logger.info("Executing %s code (%d chars)", language, len(code))

        return self.invoke(
            "executeCode",
            {
                "code": code,
                "language": language,
                "clearContext": clear_context,
            },
        )

    def execute_command(
        self,
        command: str,
    ) -> Dict[str, Any]:
        """Execute a shell command in the interpreter environment.

        This is a convenience wrapper around executeCommand.

        Args:
            command: Shell command to execute.

        Returns:
            Dict containing command execution results.

        Example:
            >>> # List files
            >>> result = client.execute_command('ls -la')

            >>> # Check Python version
            >>> result = client.execute_command('python --version')
        """
        self.logger.info("Executing shell command: %s...", command[:50])
        return self.invoke("executeCommand", {"command": command})

    def clear_context(self) -> Dict[str, Any]:
        """Clear all variable state in the Python execution context.

        This resets the interpreter to a fresh state, removing all
        previously defined variables, imports, and function definitions.

        Note: Only affects Python context. JavaScript/TypeScript contexts
        are not affected.

        Returns:
            Dict containing the result of the clear operation.

        Example:
            >>> client.execute_code('x = 10')
            >>> client.execute_code('print(x)')  # prints 10
            >>> client.clear_context()
            >>> client.execute_code('print(x)')  # NameError: x is not defined
        """
        self.logger.info("Clearing Python execution context")
        return self.invoke(
            "executeCode",
            {
                "code": "# Context cleared",
                "language": "python",
                "clearContext": True,
            },
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
