from unittest.mock import MagicMock, patch

from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter, code_session


class TestCodeInterpreterClient:
    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    def test_init(self, mock_get_endpoint, mock_boto3):
        # Arrange
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_boto3.Session.return_value = mock_session
        mock_get_endpoint.return_value = "https://mock-endpoint.com"
        region = "us-west-2"

        # Act
        client = CodeInterpreter(region)

        # Assert
        mock_boto3.Session.assert_called_once()
        mock_session.client.assert_called_once_with(
            "bedrock-agentcore", region_name=region, endpoint_url="https://mock-endpoint.com"
        )
        assert client.client == mock_client
        assert client.identifier is None
        assert client.session_id is None

    @patch("bedrock_agentcore.tools.code_interpreter_client.get_data_plane_endpoint")
    def test_init_with_custom_session(self, mock_get_endpoint):
        # Arrange
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_get_endpoint.return_value = "https://mock-endpoint.com"
        region = "us-west-2"

        # Act
        client = CodeInterpreter(region, session=mock_session)

        # Assert
        mock_session.client.assert_called_once_with(
            "bedrock-agentcore", region_name=region, endpoint_url="https://mock-endpoint.com"
        )
        assert client.client == mock_client
        assert client.identifier is None
        assert client.session_id is None

    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_property_getters_setters(self, mock_boto3):
        # Arrange
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_boto3.Session.return_value = mock_session

        client = CodeInterpreter("us-west-2")
        test_identifier = "test.identifier"
        test_session_id = "test-session-id"

        # Act & Assert - identifier
        client.identifier = test_identifier
        assert client.identifier == test_identifier

        # Act & Assert - session_id
        client.session_id = test_session_id
        assert client.session_id == test_session_id

    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    @patch("bedrock_agentcore.tools.code_interpreter_client.uuid.uuid4")
    def test_start_with_defaults(self, mock_uuid4, mock_boto3):
        # Arrange
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_boto3.Session.return_value = mock_session
        mock_uuid4.return_value.hex = "12345678abcdef"

        client = CodeInterpreter("us-west-2")
        mock_response = {"codeInterpreterIdentifier": "aws.codeinterpreter.v1", "sessionId": "session-123"}
        mock_client.start_code_interpreter_session.return_value = mock_response

        # Act
        session_id = client.start()

        # Assert
        mock_client.start_code_interpreter_session.assert_called_once_with(
            codeInterpreterIdentifier="aws.codeinterpreter.v1",
            name="code-session-12345678",
            sessionTimeoutSeconds=900,
        )
        assert session_id == "session-123"
        assert client.identifier == "aws.codeinterpreter.v1"
        assert client.session_id == "session-123"

    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_start_with_custom_params(self, mock_boto3):
        # Arrange
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_boto3.Session.return_value = mock_session

        client = CodeInterpreter("us-west-2")
        mock_response = {"codeInterpreterIdentifier": "custom.interpreter", "sessionId": "custom-session-123"}
        mock_client.start_code_interpreter_session.return_value = mock_response

        # Act
        session_id = client.start(
            identifier="custom.interpreter",
            name="custom-session",
            session_timeout_seconds=600,
        )

        # Assert
        mock_client.start_code_interpreter_session.assert_called_once_with(
            codeInterpreterIdentifier="custom.interpreter",
            name="custom-session",
            sessionTimeoutSeconds=600,
        )
        assert session_id == "custom-session-123"
        assert client.identifier == "custom.interpreter"
        assert client.session_id == "custom-session-123"

    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_stop_when_session_exists(self, mock_boto3):
        # Arrange
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_boto3.Session.return_value = mock_session

        client = CodeInterpreter("us-west-2")
        client.identifier = "test.identifier"
        client.session_id = "test-session-id"

        # Act
        client.stop()

        # Assert
        mock_client.stop_code_interpreter_session.assert_called_once_with(
            codeInterpreterIdentifier="test.identifier", sessionId="test-session-id"
        )
        assert client.identifier is None
        assert client.session_id is None

    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_stop_when_no_session(self, mock_boto3):
        # Arrange
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_boto3.Session.return_value = mock_session

        client = CodeInterpreter("us-west-2")
        client.identifier = None
        client.session_id = None

        # Act
        result = client.stop()

        # Assert
        mock_client.stop_code_interpreter_session.assert_not_called()
        assert result is True

    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    @patch("bedrock_agentcore.tools.code_interpreter_client.uuid.uuid4")
    def test_invoke_with_existing_session(self, mock_uuid4, mock_boto3):
        # Arrange
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_boto3.Session.return_value = mock_session
        mock_uuid4.return_value.hex = "12345678abcdef"

        client = CodeInterpreter("us-west-2")
        client.identifier = "test.identifier"
        client.session_id = "test-session-id"

        mock_response = {"result": "success"}
        mock_client.invoke_code_interpreter.return_value = mock_response

        # Act
        result = client.invoke(method="testMethod", params={"param1": "value1"})

        # Assert
        mock_client.invoke_code_interpreter.assert_called_once_with(
            codeInterpreterIdentifier="test.identifier",
            sessionId="test-session-id",
            name="testMethod",
            arguments={"param1": "value1"},
        )
        assert result == mock_response

    @patch("bedrock_agentcore.tools.code_interpreter_client.boto3")
    def test_invoke_with_no_session(self, mock_boto3):
        # Arrange
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_boto3.Session.return_value = mock_session

        client = CodeInterpreter("us-west-2")
        client.identifier = None
        client.session_id = None

        mock_start_response = {"codeInterpreterIdentifier": "aws.codesandbox.v1", "sessionId": "session-123"}
        mock_client.start_code_interpreter_session.return_value = mock_start_response

        mock_invoke_response = {"result": "success"}
        mock_client.invoke_code_interpreter.return_value = mock_invoke_response

        # Act
        result = client.invoke(method="testMethod", params=None)

        # Assert
        mock_client.start_code_interpreter_session.assert_called_once()
        mock_client.invoke_code_interpreter.assert_called_once_with(
            codeInterpreterIdentifier="aws.codesandbox.v1",
            sessionId="session-123",
            name="testMethod",
            arguments={},
        )
        assert result == mock_invoke_response

    @patch("bedrock_agentcore.tools.code_interpreter_client.CodeInterpreter")
    def test_code_session_context_manager(self, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Act
        with code_session("us-west-2"):
            pass

        # Assert
        mock_client_class.assert_called_once_with("us-west-2", session=None)
        mock_client.start.assert_called_once()
        mock_client.stop.assert_called_once()

    @patch("bedrock_agentcore.tools.code_interpreter_client.CodeInterpreter")
    def test_code_session_context_manager_with_session(self, mock_client_class):
        # Arrange
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_session = MagicMock()

        # Act
        with code_session("us-west-2", session=mock_session):
            pass

        # Assert
        mock_client_class.assert_called_once_with("us-west-2", session=mock_session)
        mock_client.start.assert_called_once()
        mock_client.stop.assert_called_once()
